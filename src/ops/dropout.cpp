#include "ops/dropout.h"

#include "core/check.h"

namespace llm {
namespace ops {
namespace {

void check_probability(float probability) {
  LLM_CHECK_MSG(probability >= 0.0f && probability < 1.0f,
                "вероятность дропаута " << probability
                                        << " должна быть в [0, 1); единица "
                                           "занулила бы всё");
}

}  // namespace

Tensor dropout_mask(const Shape& shape, float probability, Rng* rng) {
  check_probability(probability);
  LLM_CHECK(rng != nullptr);

  Tensor out = Tensor::uninitialized(shape);
  float* data = out.data();
  const int64_t count = out.numel();
  for (int64_t i = 0; i < count; ++i) {
    // Строго «меньше»: при probability == 0 условие не выполняется никогда,
    // и маска состоит из одних единиц.
    data[i] = static_cast<float>(rng->uniform()) < probability ? 0.0f : 1.0f;
  }
  return out;
}

Tensor dropout_forward(const Tensor& input, const Tensor& mask,
                       float probability) {
  check_probability(probability);
  LLM_CHECK_MSG(input.shape() == mask.shape(),
                "маска дропаута другой формы: " << mask.shape().to_string()
                                                << " против "
                                                << input.shape().to_string());

  const Tensor dense_input = input.contiguous();
  const Tensor dense_mask = mask.contiguous();
  Tensor out = Tensor::uninitialized(input.shape());

  const float scale = 1.0f / (1.0f - probability);
  const float* source = dense_input.data();
  const float* keep = dense_mask.data();
  float* result = out.data();
  const int64_t count = out.numel();
  for (int64_t i = 0; i < count; ++i) {
    result[i] = source[i] * keep[i] * scale;
  }
  return out;
}

Tensor dropout_backward(const Tensor& grad_output, const Tensor& mask,
                        float probability) {
  // Прямой проход — умножение на постоянную (для данного шага) величину,
  // поэтому обратный выражается той же формулой.
  return dropout_forward(grad_output, mask, probability);
}

}  // namespace ops
}  // namespace llm
