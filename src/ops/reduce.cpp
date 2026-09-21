#include "ops/reduce.h"

#include "core/check.h"
#include "core/iterate.h"
#include "ops/elementwise.h"

namespace llm {
namespace ops {
namespace {

// Прибавляет вход к накопителю, пройдя форму кусками.
//
// Накопитель — это результат, растянутый до формы входа: по сокращаемым осям
// у него шаг 0, и все элементы вдоль такой оси попадают в одну ячейку.
// Суммирование получается само собой, без разбора случаев, — и это же
// свойство позволяет схлопнуть хвост формы по обоим наборам шагов сразу.
//
// Внутри куска три случая, а не два, и различаются они не только скоростью.
//
// Оба шага единичные — поэлементная прибавка. Шаг входа единичный, а
// накопителя нулевой — весь кусок суммируется в одну ячейку, и промежуточная
// сумма берётся в double: она всё равно нужна, а точность достаётся бесплатно.
// Третий случай — любые другие шаги — считается общим циклом со шагами.
//
// Про третий стоит сказать прямо: при нулевом шаге накопителя он тоже
// суммирует кусок в одну ячейку, но уже в float, а не в double. То есть
// точность суммы зависит от того, КАК лежит вход: у плотного тензора и у вида
// с переставленными осями на одних и тех же числах получится разное последнее
// округление. Обещанию проекта это не противоречит — оно про независимость от
// машины, а раскладку задаёт программа, и на одной и той же программе ответ
// один и тот же на любом железе. Но знать об этом надо, и на общий случай
// нельзя смотреть как на медленный вариант того же самого.
void accumulate_blocks(const Shape& shape,
                       const Dims& in_strides, const float* in,
                       const Dims& acc_strides, float* acc) {
  const BlockWalkN<2> walk = block_walk2(shape, in_strides, acc_strides);
  const int64_t run = walk.run();
  const int64_t in_step = walk.run_stride(0);
  const int64_t acc_step = walk.run_stride(1);

  BlockWalkN<2>::Cursor cursor = walk.at(0);
  for (int64_t block = 0; block < walk.blocks(); ++block) {
    const float* src = in + cursor.offset(0);
    float* dst = acc + cursor.offset(1);
    if (in_step == 1 && acc_step == 1) {
      for (int64_t i = 0; i < run; ++i) {
        dst[i] += src[i];
      }
    } else if (in_step == 1 && acc_step == 0) {
      double total = 0.0;
      for (int64_t i = 0; i < run; ++i) {
        total += src[i];
      }
      *dst += static_cast<float>(total);
    } else {
      for (int64_t i = 0; i < run; ++i) {
        dst[i * acc_step] += src[i * in_step];
      }
    }
    cursor.advance();
  }
}

}  // namespace

Tensor sum(const Tensor& input, const std::vector<int>& axes, bool keepdim) {
  const int rank = input.rank();
  std::vector<bool> reduced(static_cast<std::size_t>(rank), false);
  for (std::size_t i = 0; i < axes.size(); ++i) {
    const std::size_t axis =
        static_cast<std::size_t>(input.shape().normalize_axis(axes[i]));
    // Повтор оси запрещён, хотя сама сумма его пережила бы: набор осей —
    // множество, и второе упоминание ничего не меняет. А вот mean делит на
    // произведение размеров перечисленных осей, и повтор превратил бы делитель
    // в квадрат — без падения и без признака. Проще не пустить сюда.
    LLM_CHECK_MSG(!reduced[axis],
                  "ось " << axis << " перечислена в редукции дважды");
    reduced[axis] = true;
  }

  std::vector<int64_t> kept(static_cast<std::size_t>(rank));
  for (int axis = 0; axis < rank; ++axis) {
    const std::size_t a = static_cast<std::size_t>(axis);
    kept[a] = reduced[a] ? 1 : input.dim(axis);
  }

  Tensor out = Tensor::zeros(Shape(kept));
  if (input.numel() != 0) {
    // Растягиваем результат обратно до формы входа: по сокращаемым осям это
    // даёт шаг 0, то есть все элементы вдоль такой оси попадают в одну ячейку.
    // Суммирование получается само собой, без отдельного разбора случаев.
    Tensor accumulator = out.expand(input.shape());
    accumulate_blocks(input.shape(), input.strides(), input.data(),
                      accumulator.strides(), accumulator.data());
  }

  if (keepdim) {
    return out;
  }
  std::vector<int64_t> squeezed;
  for (int axis = 0; axis < rank; ++axis) {
    if (!reduced[static_cast<std::size_t>(axis)]) {
      squeezed.push_back(input.dim(axis));
    }
  }
  return out.reshape(Shape(std::move(squeezed)));
}

Tensor mean(const Tensor& input, const std::vector<int>& axes, bool keepdim) {
  const Tensor total = sum(input, axes, keepdim);
  int64_t count = 1;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    count *= input.dim(axes[i]);
  }
  LLM_CHECK_GT(count, static_cast<int64_t>(0));
  return mul_scalar(total, 1.0f / static_cast<float>(count));
}

namespace {

std::vector<int> all_axes(int rank) {
  std::vector<int> axes;
  for (int axis = 0; axis < rank; ++axis) {
    axes.push_back(axis);
  }
  return axes;
}

}  // namespace

Tensor sum_all(const Tensor& input) {
  return sum(input, all_axes(input.rank()), false);
}

Tensor mean_all(const Tensor& input) {
  LLM_CHECK_GT(input.numel(), static_cast<int64_t>(0));
  return mul_scalar(sum_all(input), 1.0f / static_cast<float>(input.numel()));
}

Tensor reduce_to_shape(const Tensor& grad, const Shape& target) {
  if (grad.shape() == target) {
    return grad;
  }
  const int pad = grad.rank() - target.rank();
  LLM_CHECK_MSG(pad >= 0, "градиент формы " << grad.shape()
                                            << " не мог возникнуть из "
                                            << target);

  std::vector<int> axes;
  for (int axis = 0; axis < grad.rank(); ++axis) {
    if (axis < pad) {
      // Ось, которой у цели вообще не было: растяжение добавило её слева.
      axes.push_back(axis);
      continue;
    }
    const int64_t target_size = target.dim(axis - pad);
    if (target_size == 1 && grad.dim(axis) != 1) {
      axes.push_back(axis);
      continue;
    }
    LLM_CHECK_MSG(
        target_size == grad.dim(axis),
        "градиент формы " << grad.shape() << " несовместим с " << target);
  }

  const Tensor reduced = axes.empty() ? grad : sum(grad, axes, true);
  return reduced.contiguous().reshape(target);
}

}  // namespace ops
}  // namespace llm
