#include "ops/rope.h"

#include <cmath>

#include "core/check.h"

namespace llm {
namespace ops {
namespace {

Tensor apply_rotation(const Tensor& input, int64_t position_offset, float theta,
                      bool inverse) {
  LLM_CHECK_MSG(input.rank() >= 2, "RoPE нужны оси позиции и координаты");
  const int64_t head_dim = input.dim(-1);
  const int64_t sequence = input.dim(-2);
  LLM_CHECK_MSG(head_dim % 2 == 0,
                "размерность головы " << head_dim << " должна быть чётной");

  const Tensor dense = input.contiguous();
  const float* source = dense.data();
  Tensor out = Tensor::uninitialized(input.shape());
  float* result = out.data();

  const int64_t pairs = head_dim / 2;
  const int64_t blocks =
      sequence == 0 ? 0 : input.numel() / (sequence * head_dim);

  for (int64_t block = 0; block < blocks; ++block) {
    for (int64_t step = 0; step < sequence; ++step) {
      const int64_t position = step + position_offset;
      const int64_t base = (block * sequence + step) * head_dim;

      for (int64_t pair = 0; pair < pairs; ++pair) {
        const double exponent =
            -2.0 * static_cast<double>(pair) / static_cast<double>(head_dim);
        const double frequency = std::pow(static_cast<double>(theta), exponent);
        const double angle = static_cast<double>(position) * frequency;

        const float cosine = static_cast<float>(std::cos(angle));
        const float sine =
            static_cast<float>(inverse ? -std::sin(angle) : std::sin(angle));

        const float even = source[base + 2 * pair];
        const float odd = source[base + 2 * pair + 1];
        result[base + 2 * pair] = even * cosine - odd * sine;
        result[base + 2 * pair + 1] = even * sine + odd * cosine;
      }
    }
  }
  return out;
}

}  // namespace

Tensor rope(const Tensor& input, int64_t position_offset, float theta) {
  return apply_rotation(input, position_offset, theta, false);
}

Tensor rope_backward(const Tensor& grad_output, int64_t position_offset,
                     float theta) {
  return apply_rotation(grad_output, position_offset, theta, true);
}

}  // namespace ops
}  // namespace llm
