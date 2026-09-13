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

namespace {

// Логарифм суммы экспонент строки, устойчивый к большим значениям.
double row_log_sum_exp(const float* row, int64_t width) {
  float maximum = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < width; ++i) {
    maximum = row[i] > maximum ? row[i] : maximum;
  }
  double sum = 0.0;
  for (int64_t i = 0; i < width; ++i) {
    sum += std::exp(static_cast<double>(row[i] - maximum));
  }
  return static_cast<double>(maximum) + std::log(sum);
}

}  // namespace

Tensor z_loss(const Tensor& logits) {
  LLM_CHECK_MSG(logits.rank() == 2, "z-loss ждёт матрицу (n, vocab)");
  const int64_t rows = logits.dim(0);
  const int64_t vocab = logits.dim(1);
  LLM_CHECK_GT(rows, static_cast<int64_t>(0));

  const Tensor dense = logits.contiguous();
  const float* data = dense.data();

  double total = 0.0;
  for (int64_t row = 0; row < rows; ++row) {
    const double z = row_log_sum_exp(data + row * vocab, vocab);
    total += z * z;
  }

  Tensor out = Tensor::uninitialized(Shape());
  *out.data() = static_cast<float>(total / static_cast<double>(rows));
  return out;
}

Tensor z_loss_backward(const Tensor& logits, float grad_output) {
  LLM_CHECK_MSG(logits.rank() == 2, "z-loss ждёт матрицу (n, vocab)");
  const int64_t rows = logits.dim(0);
  const int64_t vocab = logits.dim(1);

  const Tensor dense = logits.contiguous();
  const float* data = dense.data();

  Tensor out = Tensor::uninitialized(logits.shape());
  float* grad = out.data();

  for (int64_t row = 0; row < rows; ++row) {
    const float* row_data = data + row * vocab;
    float* grad_row = grad + row * vocab;

    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < vocab; ++i) {
      maximum = row_data[i] > maximum ? row_data[i] : maximum;
    }
    double sum_exp = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      const float value = std::exp(row_data[i] - maximum);
      grad_row[i] = value;
      sum_exp += value;
    }
    const double z = static_cast<double>(maximum) + std::log(sum_exp);
    // Производная логарифма суммы экспонент по логиту — это softmax, а
    // производная квадрата даёт множитель 2z. Деление на число строк — от
    // усреднения.
    const double scale = 2.0 * z * static_cast<double>(grad_output) /
                         (static_cast<double>(rows) * sum_exp);
    for (int64_t i = 0; i < vocab; ++i) {
      grad_row[i] = static_cast<float>(grad_row[i] * scale);
    }
  }
  return out;
}

PredictionStats prediction_stats(const Tensor& logits,
                                 const std::vector<int32_t>& targets) {
  check_arguments(logits, targets);
  const int64_t rows = logits.dim(0);
  const int64_t vocab = logits.dim(1);

  const Tensor dense = logits.contiguous();
  const float* data = dense.data();

  int64_t correct = 0;
  double entropy_total = 0.0;
  double log_z_total = 0.0;

  for (int64_t row = 0; row < rows; ++row) {
    const float* row_data = data + row * vocab;

    float maximum = -std::numeric_limits<float>::infinity();
    int64_t best = 0;
    for (int64_t i = 0; i < vocab; ++i) {
      if (row_data[i] > maximum) {
        maximum = row_data[i];
        best = i;
      }
    }
    if (best == targets[static_cast<std::size_t>(row)]) {
      ++correct;
    }

    double sum_exp = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      sum_exp += std::exp(static_cast<double>(row_data[i] - maximum));
    }
    // Энтропия через логарифм суммы экспонент: H = log(S) + max - E[x], где
    // ожидание берётся по самому распределению. Так не нужен отдельный проход
    // с материализацией вероятностей.
    const double log_sum = std::log(sum_exp);
    double weighted = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      const double probability =
          std::exp(static_cast<double>(row_data[i] - maximum)) / sum_exp;
      weighted += probability * static_cast<double>(row_data[i] - maximum);
    }
    entropy_total += log_sum - weighted;
    log_z_total += log_sum + static_cast<double>(maximum);
  }

  PredictionStats stats;
  stats.top1_accuracy = static_cast<float>(static_cast<double>(correct) /
                                           static_cast<double>(rows));
  stats.entropy = static_cast<float>(entropy_total / static_cast<double>(rows));
  stats.log_z = static_cast<float>(log_z_total / static_cast<double>(rows));
  return stats;
}

std::vector<float> per_row_loss(const Tensor& logits,
                                const std::vector<int32_t>& targets) {
  check_arguments(logits, targets);
  const int64_t rows = logits.dim(0);
  const int64_t vocab = logits.dim(1);

  const Tensor dense = logits.contiguous();
  const float* data = dense.data();

  std::vector<float> losses;
  losses.reserve(static_cast<std::size_t>(rows));
  for (int64_t row = 0; row < rows; ++row) {
    const float* row_data = data + row * vocab;
    float maximum = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < vocab; ++i) {
      maximum = row_data[i] > maximum ? row_data[i] : maximum;
    }
    double sum_exp = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      sum_exp += std::exp(static_cast<double>(row_data[i] - maximum));
    }
    const double log_sum_exp = static_cast<double>(maximum) + std::log(sum_exp);
    losses.push_back(static_cast<float>(
        log_sum_exp -
        static_cast<double>(row_data[targets[static_cast<std::size_t>(row)]])));
  }
  return losses;
}

}  // namespace ops
}  // namespace llm
