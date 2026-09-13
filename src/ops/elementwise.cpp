#include "ops/elementwise.h"

#include <cmath>

#include "core/check.h"
#include "core/iterate.h"

namespace llm {
namespace ops {
namespace {

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
    const int64_t total = out.numel();
    for (int64_t i = 0; i < total; ++i) {
      out_data[i] = fn(a_data[i], b_data[i]);
    }
    return out;
  }

  int64_t written = 0;
  for_each_offset2(result, a_view.strides(), b_view.strides(),
                   [&](int64_t a_offset, int64_t b_offset) {
                     out_data[written++] =
                         fn(a_data[a_offset], b_data[b_offset]);
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
  int64_t written = 0;
  for_each_offset(input.shape(), input.strides(),
                  [&](int64_t offset) { result[written++] = fn(in[offset]); });
  return out;
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

Tensor exp(const Tensor& input) {
  return unary_op(input, [](float x) { return std::exp(x); });
}

Tensor log(const Tensor& input) {
  return unary_op(input, [](float x) { return std::log(x); });
}

Tensor sqrt(const Tensor& input) {
  return unary_op(input, [](float x) { return std::sqrt(x); });
}

void add_into(const Tensor& source, Tensor* target) {
  LLM_CHECK(target != nullptr);
  LLM_CHECK_MSG(source.shape() == target->shape(),
                "add_into: формы " << source.shape() << " и " << target->shape()
                                   << " не совпадают");
  const float* in = source.data();
  float* out = target->data();
  for_each_offset2(target->shape(), target->strides(), source.strides(),
                   [&](int64_t to, int64_t from) { out[to] += in[from]; });
}

}  // namespace ops
}  // namespace llm
