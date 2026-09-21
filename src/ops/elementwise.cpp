#include "ops/elementwise.h"

#include <algorithm>
#include <cmath>

#include "core/check.h"
#include "core/iterate.h"
#include "core/thread_pool.h"

namespace llm {
namespace ops {
namespace {

// Сколько элементов стоит отдавать одному потоку. Поэлементная операция
// упирается в память, а не в арифметику: вход в параллельную область стоит
// около двух микросекунд, за которые поток успевает пройти порядка десяти
// килобайт.
constexpr int64_t kElementGrain = 1 << 14;

template <typename Fn>
Tensor binary_op(const Tensor& a, const Tensor& b, Fn fn) {
  const Shape result = broadcast_shapes(a.shape(), b.shape());
  Tensor out = Tensor::uninitialized(result);
  if (out.numel() == 0) {
    return out;
  }

  const Tensor a_view = a.shape() == result ? a : a.expand(result);
  const Tensor b_view = b.shape() == result ? b : b.expand(result);

  const float* a_data = a_view.data();
  const float* b_data = b_view.data();
  float* out_data = out.data();

  // Быстрый путь: оба аргумента уже нужной формы и плотные — обходимся без
  // одометра. Это самый частый случай, и он же самый горячий.
  if (a_view.is_contiguous() && b_view.is_contiguous()) {
    parallel_range(out.numel(), kElementGrain, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        out_data[i] = fn(a_data[i], b_data[i]);
      }
    });
    return out;
  }

  // Медленный путь: хотя бы один аргумент растянут или с перестановленными
  // осями. Обход идёт кусками по обоим наборам шагов сразу — у растянутого
  // аргумента шаг внутри куска оказывается нулевым, и цикл всё равно остаётся
  // простым.
  const BlockWalkN<2> walk =
      block_walk2(result, a_view.strides(), b_view.strides());
  const int64_t run = walk.run();
  const int64_t a_step = walk.run_stride(0);
  const int64_t b_step = walk.run_stride(1);
  const int64_t grain =
      std::max<int64_t>(1, kElementGrain / std::max<int64_t>(run, 1));
  parallel_range(walk.blocks(), grain, [&](int64_t begin, int64_t end) {
    BlockWalkN<2>::Cursor cursor = walk.at(begin);
    float* dst = out_data + begin * run;
    for (int64_t block = begin; block < end; ++block) {
      const float* left = a_data + cursor.offset(0);
      const float* right = b_data + cursor.offset(1);
      for (int64_t i = 0; i < run; ++i) {
        dst[i] = fn(left[i * a_step], right[i * b_step]);
      }
      dst += run;
      cursor.advance();
    }
  });
  return out;
}

template <typename Fn>
Tensor unary_op(const Tensor& input, Fn fn) {
  Tensor out = Tensor::uninitialized(input.shape());
  if (out.numel() == 0) {
    return out;
  }
  const float* in = input.data();
  float* result = out.data();

  if (input.is_contiguous()) {
    parallel_range(out.numel(), kElementGrain, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        result[i] = fn(in[i]);
      }
    });
    return out;
  }

  const BlockWalkN<1> walk = block_walk(input.shape(), input.strides());
  const int64_t run = walk.run();
  const int64_t step = walk.run_stride(0);
  const int64_t grain =
      std::max<int64_t>(1, kElementGrain / std::max<int64_t>(run, 1));
  parallel_range(walk.blocks(), grain, [&](int64_t begin, int64_t end) {
    BlockWalkN<1>::Cursor cursor = walk.at(begin);
    float* dst = result + begin * run;
    for (int64_t block = begin; block < end; ++block) {
      const float* src = in + cursor.offset(0);
      for (int64_t i = 0; i < run; ++i) {
        dst[i] = fn(src[i * step]);
      }
      dst += run;
      cursor.advance();
    }
  });
  return out;
}

// Запись или прибавка одного тензора в другой, на месте. Оба обходятся
// кусками: у цели шаги свои, и медленный поэлементный обход здесь ничем не
// оправдан — на прибавке градиента в накопитель он один из самых горячих.
void copy_or_add(const Tensor& source, Tensor* target, bool add) {
  if (target->numel() == 0) {
    return;
  }
  // У цели не должно быть нулевых шагов, и это не формальность. Нулевой шаг
  // означает, что несколько элементов формы ложатся в одну ячейку: прибавка
  // тогда накопилась бы в неё многократно, а обход идёт по потокам — то есть
  // ещё и с гонкой. Все нынешние вызывающие передают либо плотный тензор,
  // либо срез плотного, у которых нулевых шагов не бывает; проверка записывает
  // это требование, а не предполагает его.
  for (int axis = 0; axis < target->rank(); ++axis) {
    LLM_DCHECK(target->shape().dim(axis) <= 1 || target->stride(axis) != 0);
  }
  const float* in = source.data();
  float* out = target->data();
  const BlockWalkN<2> walk =
      block_walk2(target->shape(), target->strides(), source.strides());
  const int64_t run = walk.run();
  const int64_t out_step = walk.run_stride(0);
  const int64_t in_step = walk.run_stride(1);
  const int64_t grain =
      std::max<int64_t>(1, kElementGrain / std::max<int64_t>(run, 1));

  parallel_range(walk.blocks(), grain, [&](int64_t begin, int64_t end) {
    BlockWalkN<2>::Cursor cursor = walk.at(begin);
    for (int64_t block = begin; block < end; ++block) {
      float* dst = out + cursor.offset(0);
      const float* src = in + cursor.offset(1);
      if (add) {
        for (int64_t i = 0; i < run; ++i) {
          dst[i * out_step] += src[i * in_step];
        }
      } else {
        for (int64_t i = 0; i < run; ++i) {
          dst[i * out_step] = src[i * in_step];
        }
      }
      cursor.advance();
    }
  });
}

}  // namespace

Tensor add(const Tensor& a, const Tensor& b) {
  return binary_op(a, b, [](float x, float y) { return x + y; });
}

Tensor sub(const Tensor& a, const Tensor& b) {
  return binary_op(a, b, [](float x, float y) { return x - y; });
}

Tensor mul(const Tensor& a, const Tensor& b) {
  return binary_op(a, b, [](float x, float y) { return x * y; });
}

Tensor div(const Tensor& a, const Tensor& b) {
  return binary_op(a, b, [](float x, float y) { return x / y; });
}

Tensor add_scalar(const Tensor& input, float scalar) {
  return unary_op(input, [scalar](float x) { return x + scalar; });
}

Tensor mul_scalar(const Tensor& input, float scalar) {
  return unary_op(input, [scalar](float x) { return x * scalar; });
}

Tensor neg(const Tensor& input) {
  return unary_op(input, [](float x) { return -x; });
}

// Экспонента, логарифм и корень считаются библиотечными функциями, и это
// накладывает ограничение, которое стоит назвать прямо: на пути модели их
// быть не должно.
//
// Стандарт не определяет std::exp и std::log до последнего разряда, поэтому
// на машинах с разными libc они дают разные младшие биты, — а проект обещает,
// что результат от машины не зависит. Своя экспонента у проекта есть, и все
// горячие места (softmax, перекрёстная энтропия, SiLU) идут через неё, см.
// ops/fast_exp.h. Проверено: вне этого файла и его обвязки в autograd ни
// ops::exp, ни ops::log, ни ops::sqrt не вызываются нигде — только из тестов.
//
// Корень стоит в том же ряду по другой причине: std::sqrt как раз определён
// точно (IEEE 754 требует правильного округления), и он безопасен. В список
// он попал потому, что живёт здесь же, и разделять эти три функции по разным
// правилам значило бы завести правило, которое никто не вспомнит.
Tensor exp(const Tensor& input) {
  return unary_op(input, [](float x) { return std::exp(x); });
}

Tensor log(const Tensor& input) {
  return unary_op(input, [](float x) { return std::log(x); });
}

Tensor sqrt(const Tensor& input) {
  return unary_op(input, [](float x) { return std::sqrt(x); });
}

void copy_into(const Tensor& source, Tensor* target) {
  LLM_CHECK(target != nullptr);
  LLM_CHECK_MSG(source.shape() == target->shape(),
                "copy_into: формы " << source.shape() << " и "
                                    << target->shape() << " не совпадают");
  copy_or_add(source, target, false);
}

void add_into(const Tensor& source, Tensor* target) {
  LLM_CHECK(target != nullptr);
  LLM_CHECK_MSG(source.shape() == target->shape(),
                "add_into: формы " << source.shape() << " и " << target->shape()
                                   << " не совпадают");
  copy_or_add(source, target, true);
}

}  // namespace ops
}  // namespace llm
