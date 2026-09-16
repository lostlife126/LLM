#include "autograd/nn.h"

#include "core/check.h"
#include "ops/elementwise.h"
#include "ops/embedding.h"
#include "ops/loss.h"
#include "ops/nn.h"
#include "ops/rope.h"

namespace llm {
namespace autograd {
namespace {

bool tracking(const Var& a) { return grad_enabled() && a.requires_grad(); }

}  // namespace

Var rms_norm(const Var& input, const Var& weight, float eps) {
  Tensor value = ops::rms_norm(input.value(), weight.value(), eps);
  if (!grad_enabled() || (!input.requires_grad() && !weight.requires_grad())) {
    return Var::constant(std::move(value));
  }

  std::vector<NodePtr> parents;
  const NodePtr input_node = input.requires_grad() ? input.node() : nullptr;
  const NodePtr weight_node = weight.requires_grad() ? weight.node() : nullptr;
  if (input_node) {
    parents.push_back(input_node);
  }
  if (weight_node) {
    parents.push_back(weight_node);
  }

  const Tensor saved_input = input.value();
  const Tensor saved_weight = weight.value();

  return Var::from_op(std::move(value), "rms_norm", std::move(parents),
                      [input_node, weight_node, saved_input, saved_weight,
                       eps](const Tensor& grad) {
                        // Одно ядро отдаёт оба градиента сразу: они делят одну
                        // и ту же редукцию по строке, и считать её дважды было
                        // бы расточительно.
                        Tensor grad_input;
                        Tensor grad_weight;
                        ops::rms_norm_backward(grad, saved_input, saved_weight,
                                               eps, &grad_input, &grad_weight);
                        if (input_node) {
                          input_node->accumulate(grad_input);
                        }
                        if (weight_node) {
                          weight_node->accumulate(grad_weight);
                        }
                      });
}

Var layer_norm(const Var& input, const Var& weight, const Var& bias,
               float eps) {
  Tensor value =
      ops::layer_norm(input.value(), weight.value(), bias.value(), eps);
  if (!grad_enabled() || (!input.requires_grad() && !weight.requires_grad() &&
                          !bias.requires_grad())) {
    return Var::constant(std::move(value));
  }

  std::vector<NodePtr> parents;
  const NodePtr input_node = input.requires_grad() ? input.node() : nullptr;
  const NodePtr weight_node = weight.requires_grad() ? weight.node() : nullptr;
  const NodePtr bias_node = bias.requires_grad() ? bias.node() : nullptr;
  if (input_node) {
    parents.push_back(input_node);
  }
  if (weight_node) {
    parents.push_back(weight_node);
  }
  if (bias_node) {
    parents.push_back(bias_node);
  }

  const Tensor saved_input = input.value();
  const Tensor saved_weight = weight.value();

  return Var::from_op(std::move(value), "layer_norm", std::move(parents),
                      [input_node, weight_node, bias_node, saved_input,
                       saved_weight, eps](const Tensor& grad) {
                        Tensor grad_input;
                        Tensor grad_weight;
                        Tensor grad_bias;
                        ops::layer_norm_backward(grad, saved_input,
                                                 saved_weight, eps, &grad_input,
                                                 &grad_weight, &grad_bias);
                        if (input_node) {
                          input_node->accumulate(grad_input);
                        }
                        if (weight_node) {
                          weight_node->accumulate(grad_weight);
                        }
                        if (bias_node) {
                          bias_node->accumulate(grad_bias);
                        }
                      });
}

Var softmax(const Var& input) {
  Tensor value = ops::softmax(input.value());
  if (!tracking(input)) {
    return Var::constant(std::move(value));
  }
  // Обратному проходу нужен выход, а не вход: пересчитывать softmax заново
  // дороже, чем сохранить готовый результат.
  const Tensor saved_output = value;
  const NodePtr node = input.node();
  return Var::from_op(
      std::move(value), "softmax", {node},
      [node, saved_output](const Tensor& grad) {
        node->accumulate(ops::softmax_backward(grad, saved_output));
      });
}

Var silu(const Var& input) {
  Tensor value = ops::silu(input.value());
  if (!tracking(input)) {
    return Var::constant(std::move(value));
  }
  const Tensor saved_input = input.value();
  const NodePtr node = input.node();
  return Var::from_op(std::move(value), "silu", {node},
                      [node, saved_input](const Tensor& grad) {
                        node->accumulate(ops::silu_backward(grad, saved_input));
                      });
}

Var gelu(const Var& input) {
  Tensor value = ops::gelu(input.value());
  if (!tracking(input)) {
    return Var::constant(std::move(value));
  }
  const Tensor saved_input = input.value();
  const NodePtr node = input.node();
  return Var::from_op(std::move(value), "gelu", {node},
                      [node, saved_input](const Tensor& grad) {
                        node->accumulate(ops::gelu_backward(grad, saved_input));
                      });
}

Var masked_softmax(const Var& scores, float scale, int64_t query_offset) {
  Tensor value = ops::masked_softmax(scores.value(), scale, query_offset);
  if (!tracking(scores)) {
    return Var::constant(std::move(value));
  }
  // Как и обычному softmax, обратному проходу нужен выход, а не вход.
  const Tensor saved_output = value;
  const NodePtr node = scores.node();
  return Var::from_op(std::move(value), "masked_softmax", {node},
                      [node, saved_output, scale, query_offset](
                          const Tensor& grad) {
                        node->accumulate(ops::masked_softmax_backward(
                            grad, saved_output, scale, query_offset));
                      });
}

Var causal_mask(const Var& scores, int64_t query_offset) {
  Tensor value = ops::causal_mask(scores.value(), query_offset);
  if (!tracking(scores)) {
    return Var::constant(std::move(value));
  }
  const NodePtr node = scores.node();
  return Var::from_op(
      std::move(value), "causal_mask", {node},
      [node, query_offset](const Tensor& grad) {
        node->accumulate(ops::causal_mask_backward(grad, query_offset));
      });
}

Var embedding(const Var& weight, const std::vector<int32_t>& ids) {
  Tensor value = ops::embedding(weight.value(), ids);
  if (!tracking(weight)) {
    return Var::constant(std::move(value));
  }
  const NodePtr node = weight.node();
  const int64_t vocab = weight.shape().dim(0);
  return Var::from_op(
      std::move(value), "embedding", {node},
      [node, ids, vocab](const Tensor& grad) {
        node->accumulate(ops::embedding_backward(grad, ids, vocab));
      });
}

Var rope(const Var& input, int64_t position_offset, float theta) {
  Tensor value = ops::rope(input.value(), position_offset, theta);
  if (!tracking(input)) {
    return Var::constant(std::move(value));
  }
  const NodePtr node = input.node();
  return Var::from_op(
      std::move(value), "rope", {node},
      [node, position_offset, theta](const Tensor& grad) {
        node->accumulate(ops::rope_backward(grad, position_offset, theta));
      });
}

Var z_loss(const Var& logits) {
  Tensor value = ops::z_loss(logits.value());
  if (!tracking(logits)) {
    return Var::constant(std::move(value));
  }
  const Tensor saved_logits = logits.value();
  const NodePtr node = logits.node();
  return Var::from_op(
      std::move(value), "z_loss", {node},
      [node, saved_logits](const Tensor& grad) {
        node->accumulate(ops::z_loss_backward(saved_logits, *grad.data()));
      });
}

Var cross_entropy(const Var& logits, const std::vector<int32_t>& targets) {
  Tensor value = ops::cross_entropy(logits.value(), targets);
  if (!tracking(logits)) {
    return Var::constant(std::move(value));
  }
  const Tensor saved_logits = logits.value();
  const NodePtr node = logits.node();
  return Var::from_op(std::move(value), "cross_entropy", {node},
                      [node, saved_logits, targets](const Tensor& grad) {
                        // Градиент приходит скаляром: обычно это
                        // единица-затравка, но если потери входят в составную
                        // величину — какой-то множитель.
                        const float scale = *grad.data();
                        Tensor contribution =
                            ops::cross_entropy_backward(saved_logits, targets);
                        if (scale != 1.0f) {
                          contribution = ops::mul_scalar(contribution, scale);
                        }
                        node->accumulate(contribution);
                      });
}

}  // namespace autograd
}  // namespace llm
