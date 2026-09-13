#include "autograd/ops.h"

#include "core/check.h"
#include "ops/elementwise.h"
#include "ops/matmul.h"
#include "ops/reduce.h"

namespace llm {
namespace autograd {
namespace {

using GradFn = std::function<Tensor(const Tensor& grad_output)>;

bool tracking(const Var& a) { return grad_enabled() && a.requires_grad(); }

bool tracking(const Var& a, const Var& b) {
  return grad_enabled() && (a.requires_grad() || b.requires_grad());
}

// Обвязка бинарной операции. Вклад каждого входа приводится к его форме через
// reduce_to_shape — это и есть обратный проход растяжения: если при прямом
// проходе один элемент участвовал в ста позициях, обратно на него приходит
// сумма ста градиентов.
Var make_binary(const Var& a, const Var& b, const char* name, Tensor value,
                GradFn grad_a, GradFn grad_b) {
  if (!tracking(a, b)) {
    return Var::constant(std::move(value));
  }
  std::vector<NodePtr> inputs;
  if (a.requires_grad()) {
    inputs.push_back(a.node());
  }
  if (b.requires_grad()) {
    inputs.push_back(b.node());
  }

  const NodePtr node_a = a.requires_grad() ? a.node() : nullptr;
  const NodePtr node_b = b.requires_grad() ? b.node() : nullptr;
  const Shape shape_a = a.shape();
  const Shape shape_b = b.shape();

  return Var::from_op(
      std::move(value), name, std::move(inputs),
      [node_a, node_b, shape_a, shape_b, grad_a, grad_b](const Tensor& grad) {
        if (node_a) {
          node_a->accumulate(ops::reduce_to_shape(grad_a(grad), shape_a));
        }
        if (node_b) {
          node_b->accumulate(ops::reduce_to_shape(grad_b(grad), shape_b));
        }
      });
}

Var make_unary(const Var& input, const char* name, Tensor value,
               GradFn grad_fn) {
  if (!tracking(input)) {
    return Var::constant(std::move(value));
  }
  const NodePtr node = input.node();
  return Var::from_op(
      std::move(value), name, {node},
      [node, grad_fn](const Tensor& grad) { node->accumulate(grad_fn(grad)); });
}

}  // namespace

Var add(const Var& a, const Var& b) {
  return make_binary(
      a, b, "add", ops::add(a.value(), b.value()),
      [](const Tensor& grad) { return grad; },
      [](const Tensor& grad) { return grad; });
}

Var sub(const Var& a, const Var& b) {
  return make_binary(
      a, b, "sub", ops::sub(a.value(), b.value()),
      [](const Tensor& grad) { return grad; },
      [](const Tensor& grad) { return ops::neg(grad); });
}

Var mul(const Var& a, const Var& b) {
  // Обратный проход произведения требует значений обоих входов, поэтому они
  // захватываются замыканием и живут до конца обратного прохода. Это и есть
  // основная плата автограда по памяти.
  const Tensor a_value = a.value();
  const Tensor b_value = b.value();
  return make_binary(
      a, b, "mul", ops::mul(a_value, b_value),
      [b_value](const Tensor& grad) { return ops::mul(grad, b_value); },
      [a_value](const Tensor& grad) { return ops::mul(grad, a_value); });
}

Var div(const Var& a, const Var& b) {
  const Tensor a_value = a.value();
  const Tensor b_value = b.value();
  return make_binary(
      a, b, "div", ops::div(a_value, b_value),
      [b_value](const Tensor& grad) { return ops::div(grad, b_value); },
      [a_value, b_value](const Tensor& grad) {
        // d/db (a / b) = -a / b^2
        const Tensor quotient = ops::div(a_value, ops::mul(b_value, b_value));
        return ops::neg(ops::mul(grad, quotient));
      });
}

Var add_scalar(const Var& input, float scalar) {
  return make_unary(input, "add_scalar", ops::add_scalar(input.value(), scalar),
                    [](const Tensor& grad) { return grad; });
}

Var mul_scalar(const Var& input, float scalar) {
  return make_unary(
      input, "mul_scalar", ops::mul_scalar(input.value(), scalar),
      [scalar](const Tensor& grad) { return ops::mul_scalar(grad, scalar); });
}

Var neg(const Var& input) {
  return make_unary(input, "neg", ops::neg(input.value()),
                    [](const Tensor& grad) { return ops::neg(grad); });
}

Var exp(const Var& input) {
  // Производная экспоненты — она сама, поэтому сохраняем выход, а не вход.
  Tensor value = ops::exp(input.value());
  const Tensor saved = value;
  return make_unary(
      input, "exp", std::move(value),
      [saved](const Tensor& grad) { return ops::mul(grad, saved); });
}

Var log(const Var& input) {
  const Tensor saved = input.value();
  return make_unary(input, "log", ops::log(saved), [saved](const Tensor& grad) {
    return ops::div(grad, saved);
  });
}

Var sqrt(const Var& input) {
  Tensor value = ops::sqrt(input.value());
  const Tensor saved = value;
  return make_unary(input, "sqrt", std::move(value),
                    [saved](const Tensor& grad) {
                      // d/dx sqrt(x) = 1 / (2 * sqrt(x))
                      return ops::div(ops::mul_scalar(grad, 0.5f), saved);
                    });
}

Var sum(const Var& input, const std::vector<int>& axes, bool keepdim) {
  const Shape input_shape = input.shape();
  // Форма с сохранёнными осями нужна обратному проходу: растянуть (b, t) до
  // (b, t, d) нельзя, а (b, t, 1) — можно.
  const Shape keepdim_shape = ops::sum(input.value(), axes, true).shape();
  return make_unary(
      input, "sum", ops::sum(input.value(), axes, keepdim),
      [input_shape, keepdim_shape](const Tensor& grad) {
        // Каждый элемент входа вошёл в сумму ровно один раз,
        // поэтому градиент просто размножается обратно.
        return grad.contiguous().reshape(keepdim_shape).expand(input_shape);
      });
}

Var mean(const Var& input, const std::vector<int>& axes, bool keepdim) {
  int64_t count = 1;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    count *= input.shape().dim(axes[i]);
  }
  LLM_CHECK_GT(count, static_cast<int64_t>(0));
  const Shape input_shape = input.shape();
  const Shape keepdim_shape = ops::sum(input.value(), axes, true).shape();
  const float scale = 1.0f / static_cast<float>(count);
  return make_unary(input, "mean", ops::mean(input.value(), axes, keepdim),
                    [input_shape, keepdim_shape, scale](const Tensor& grad) {
                      return ops::mul_scalar(grad, scale)
                          .reshape(keepdim_shape)
                          .expand(input_shape);
                    });
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

Var sum_all(const Var& input) {
  return sum(input, all_axes(input.shape().rank()), false);
}

Var mean_all(const Var& input) {
  return mean(input, all_axes(input.shape().rank()), false);
}

Var matmul(const Var& a, const Var& b) {
  const Tensor a_value = a.value();
  const Tensor b_value = b.value();
  return make_binary(
      a, b, "matmul", ops::matmul(a_value, b_value),
      [b_value](const Tensor& grad) {
        // dA = dC * B^T. Транспонирование здесь — вид без копирования, и
        // matmul распознает его по единичному шагу первой оси.
        return ops::matmul(grad, b_value.transpose(-2, -1));
      },
      [a_value](const Tensor& grad) {
        // dB = A^T * dC. Если вес общий на весь батч, результат получится с
        // осью батча, и reduce_to_shape в make_binary просуммирует по ней —
        // ровно то, что нужно: вклад каждого элемента батча в общий вес.
        return ops::matmul(a_value.transpose(-2, -1), grad);
      });
}

Var reshape(const Var& input, const Shape& shape) {
  const Shape input_shape = input.shape();
  return make_unary(input, "reshape", input.value().contiguous().reshape(shape),
                    [input_shape](const Tensor& grad) {
                      return grad.contiguous().reshape(input_shape);
                    });
}

Var transpose(const Var& input, int axis_a, int axis_b) {
  return make_unary(input, "transpose", input.value().transpose(axis_a, axis_b),
                    [axis_a, axis_b](const Tensor& grad) {
                      // Перестановка двух осей обратна сама себе.
                      return grad.transpose(axis_a, axis_b);
                    });
}

Var permute(const Var& input, const std::vector<int>& order) {
  const int rank = input.shape().rank();
  LLM_CHECK_EQ(static_cast<int>(order.size()), rank);

  // Обратная перестановка: если ось j встала на место i, то в обратном проходе
  // ось i должна вернуться на место j.
  std::vector<int> inverse(static_cast<std::size_t>(rank));
  for (int i = 0; i < rank; ++i) {
    const int axis =
        input.shape().normalize_axis(order[static_cast<std::size_t>(i)]);
    inverse[static_cast<std::size_t>(axis)] = i;
  }
  return make_unary(
      input, "permute", input.value().permute(order),
      [inverse](const Tensor& grad) { return grad.permute(inverse); });
}

Var slice(const Var& input, int axis, int64_t start, int64_t count) {
  const Shape input_shape = input.shape();
  const int normalized = input_shape.normalize_axis(axis);
  return make_unary(
      input, "slice", input.value().slice(axis, start, count),
      [input_shape, normalized, start, count](const Tensor& grad) {
        // Вне среза вход на результат не влиял, там градиент нуль.
        Tensor full = Tensor::zeros(input_shape);
        Tensor target = full.slice(normalized, start, count);
        ops::add_into(grad, &target);
        return full;
      });
}

Var cat(const std::vector<Var>& parts, int axis) {
  LLM_CHECK_MSG(!parts.empty(), "cat без аргументов");
  const int normalized = parts[0].shape().normalize_axis(axis);

  int64_t total = 0;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    LLM_CHECK_MSG(parts[i].shape().rank() == parts[0].shape().rank(),
                  "cat: разный ранг у частей");
    for (int a = 0; a < parts[0].shape().rank(); ++a) {
      if (a != normalized) {
        LLM_CHECK_MSG(parts[i].shape().dim(a) == parts[0].shape().dim(a),
                      "cat: части расходятся по оси " << a);
      }
    }
    total += parts[i].shape().dim(normalized);
  }

  std::vector<int64_t> dims = parts[0].shape().dims();
  dims[static_cast<std::size_t>(normalized)] = total;
  Tensor value = Tensor::zeros(Shape(dims));

  std::vector<int64_t> offsets;
  int64_t cursor = 0;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    Tensor target =
        value.slice(normalized, cursor, parts[i].shape().dim(normalized));
    ops::add_into(parts[i].value(), &target);
    offsets.push_back(cursor);
    cursor += parts[i].shape().dim(normalized);
  }

  bool any_grad = false;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    any_grad = any_grad || parts[i].requires_grad();
  }
  if (!grad_enabled() || !any_grad) {
    return Var::constant(std::move(value));
  }

  std::vector<NodePtr> inputs;
  std::vector<NodePtr> targets;
  std::vector<int64_t> counts;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const NodePtr node = parts[i].requires_grad() ? parts[i].node() : nullptr;
    targets.push_back(node);
    counts.push_back(parts[i].shape().dim(normalized));
    if (node) {
      inputs.push_back(node);
    }
  }

  return Var::from_op(
      std::move(value), "cat", std::move(inputs),
      [targets, offsets, counts, normalized](const Tensor& grad) {
        // Каждой части достаётся её кусок градиента — ровно тот, который она
        // заняла в результате.
        for (std::size_t i = 0; i < targets.size(); ++i) {
          if (targets[i]) {
            targets[i]->accumulate(
                grad.slice(normalized, offsets[i], counts[i]));
          }
        }
      });
}

}  // namespace autograd
}  // namespace llm
