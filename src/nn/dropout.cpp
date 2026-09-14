#include "nn/dropout.h"

#include <utility>

#include "core/check.h"
#include "ops/dropout.h"

namespace llm {
namespace nn {
namespace {

bool g_enabled = false;
Rng* g_rng = nullptr;

}  // namespace

TrainingScope::TrainingScope(uint64_t seed)
    : saved_enabled_(g_enabled), saved_rng_(g_rng), rng_(seed) {
  g_enabled = true;
  g_rng = &rng_;
}

TrainingScope::~TrainingScope() {
  g_enabled = saved_enabled_;
  g_rng = saved_rng_;
}

bool training_mode() { return g_enabled; }

autograd::Var dropout(const autograd::Var& input, float probability) {
  if (!g_enabled || probability <= 0.0f) {
    return input;
  }
  LLM_CHECK(g_rng != nullptr);

  // Маска считается один раз и остаётся в замыкании: обратный проход обязан
  // занулить ровно те же элементы. Сгенерировать её заново — это был бы
  // другой дропаут, и градиент перестал бы соответствовать значению.
  const Tensor mask =
      ops::dropout_mask(input.value().shape(), probability, g_rng);
  Tensor value = ops::dropout_forward(input.value(), mask, probability);

  if (!autograd::grad_enabled() || !input.requires_grad()) {
    return autograd::Var::constant(std::move(value));
  }

  const autograd::NodePtr node = input.node();
  return autograd::Var::from_op(
      std::move(value), "dropout", {node},
      [node, mask, probability](const Tensor& grad) {
        node->accumulate(ops::dropout_backward(grad, mask, probability));
      });
}

}  // namespace nn
}  // namespace llm
