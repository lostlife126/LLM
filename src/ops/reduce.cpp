#include "ops/reduce.h"

#include "core/check.h"
#include "core/iterate.h"
#include "ops/elementwise.h"

namespace llm {
namespace ops {

Tensor sum(const Tensor& input, const std::vector<int>& axes, bool keepdim) {
  const int rank = input.rank();
  std::vector<bool> reduced(static_cast<std::size_t>(rank), false);
  for (std::size_t i = 0; i < axes.size(); ++i) {
    reduced[static_cast<std::size_t>(input.shape().normalize_axis(axes[i]))] =
        true;
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
    const float* in = input.data();
    float* acc = accumulator.data();
    for_each_offset2(input.shape(), input.strides(), accumulator.strides(),
                     [&](int64_t from, int64_t to) { acc[to] += in[from]; });
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
