#include "ops/loss.h"

#include <cmath>
#include <limits>

#include "core/check.h"

namespace llm {
namespace ops {
namespace {

void check_arguments(const Tensor& logits,
                     const std::vector<int32_t>& targets) {
  LLM_CHECK_MSG(
      logits.rank() == 2,
      "логиты должны быть матрицей (n, vocab), а не " << logits.shape());
  LLM_CHECK_MSG(
      logits.dim(0) == static_cast<int64_t>(targets.size()),
      "логитов " << logits.dim(0) << " строк, а целей " << targets.size());
  LLM_CHECK_GT(logits.dim(0), static_cast<int64_t>(0));
}

}  // namespace

Tensor cross_entropy(const Tensor& logits,
                     const std::vector<int32_t>& targets) {
  check_arguments(logits, targets);
  const int64_t rows = logits.dim(0);
  const int64_t vocab = logits.dim(1);

  const Tensor dense = logits.contiguous();
  const float* data = dense.data();

  double total = 0.0;
  for (int64_t row = 0; row < rows; ++row) {
    const float* row_data = data + row * vocab;
    const int64_t target = targets[static_cast<std::size_t>(row)];
    LLM_CHECK_MSG(target >= 0 && target < vocab,
                  "цель " << target << " вне словаря размера " << vocab);

    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < vocab; ++i) {
      if (row_data[i] > maximum) {
        maximum = row_data[i];
      }
    }
    double sum_exp = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      sum_exp += std::exp(static_cast<double>(row_data[i] - maximum));
    }
    // -log p[target] = logsumexp(logits) - logits[target], и вычитание
    // максимума делает обе части конечными при любых логитах.
    const double log_sum_exp = static_cast<double>(maximum) + std::log(sum_exp);
    total += log_sum_exp - static_cast<double>(row_data[target]);
  }

  Tensor out = Tensor::uninitialized(Shape());
  *out.data() = static_cast<float>(total / static_cast<double>(rows));
  return out;
}

Tensor cross_entropy_backward(const Tensor& logits,
                              const std::vector<int32_t>& targets) {
  check_arguments(logits, targets);
  const int64_t rows = logits.dim(0);
  const int64_t vocab = logits.dim(1);

  const Tensor dense = logits.contiguous();
  const float* data = dense.data();

  Tensor out = Tensor::uninitialized(logits.shape());
  float* grad = out.data();
  const float scale = 1.0f / static_cast<float>(rows);

  for (int64_t row = 0; row < rows; ++row) {
    const float* row_data = data + row * vocab;
    float* grad_row = grad + row * vocab;

    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < vocab; ++i) {
      if (row_data[i] > maximum) {
        maximum = row_data[i];
      }
    }
    double sum_exp = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      const float value = std::exp(row_data[i] - maximum);
      grad_row[i] = value;
      sum_exp += value;
    }
    const float inverse = static_cast<float>(1.0 / sum_exp);
    for (int64_t i = 0; i < vocab; ++i) {
      grad_row[i] *= inverse * scale;
    }
    // Вычесть единицу в позиции правильного токена — вся разница между
    // предсказанным распределением и целевым.
    grad_row[targets[static_cast<std::size_t>(row)]] -= scale;
  }
  return out;
}

}  // namespace ops
}  // namespace llm
