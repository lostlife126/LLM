#include "ops/loss.h"

#include <cmath>
#include <limits>

#include "core/check.h"
#include "ops/fast_exp.h"
#include "ops/parallel.h"

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

// Место под экспоненты одной строки.
//
// Экспонента считается массивом, и ей нужно куда писать; у cross-entropy
// прямого прохода готового места нет — результат там скаляр. Буфер привязан к
// потоку: строки считаются параллельно, и общий буфер пришлось бы защищать.
// Строка — это словарь, от тысячи до восьми тысяч значений, то есть десятки
// килобайт; выделять их на каждую строку было бы дороже самой экспоненты.
float* row_scratch(int64_t width) {
  thread_local std::vector<float> buffer;
  if (static_cast<int64_t>(buffer.size()) < width) {
    buffer.resize(static_cast<std::size_t>(width));
  }
  return buffer.data();
}

// Сумма экспонент строки со сдвигом на максимум и её логарифм.
//
// Сумма считается обычным последовательным циклом, а не внутри векторного
// ядра: по дорожкам она сложилась бы в другом порядке, и значение потерь
// зависело бы от набора инструкций процессора.
double log_sum_exp_into(const float* row, float maximum, float* scratch,
                        int64_t width) {
  exp_shifted(row, maximum, scratch, width);
  double sum = 0.0;
  for (int64_t i = 0; i < width; ++i) {
    sum += scratch[i];
  }
  return static_cast<double>(maximum) + std::log(sum);
}

float row_maximum(const float* row, int64_t width) {
  float maximum = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < width; ++i) {
    if (row[i] > maximum) {
      maximum = row[i];
    }
  }
  return maximum;
}

}  // namespace

Tensor cross_entropy(const Tensor& logits,
                     const std::vector<int32_t>& targets) {
  check_arguments(logits, targets);
  const int64_t rows = logits.dim(0);
  const int64_t vocab = logits.dim(1);

  const Tensor dense = logits.contiguous();
  const float* data = dense.data();

  // Сумма по строкам делится на фиксированное число частей: порядок сложения
  // влияет на младшие разряды, а значение потерь не должно зависеть от числа
  // ядер — иначе сравнение вариантов обучения теряет смысл.
  double partial[kSumParts] = {};
  for_row_parts(rows, vocab, [&](int64_t part, int64_t first, int64_t last) {
    double sum = 0.0;
    for (int64_t row = first; row < last; ++row) {
      const float* row_data = data + row * vocab;
      const int64_t target = targets[static_cast<std::size_t>(row)];
      LLM_CHECK_MSG(target >= 0 && target < vocab,
                    "цель " << target << " вне словаря размера " << vocab);

      // -log p[target] = logsumexp(logits) - logits[target], и вычитание
      // максимума делает обе части конечными при любых логитах.
      const double log_sum_exp = log_sum_exp_into(
          row_data, row_maximum(row_data, vocab), row_scratch(vocab), vocab);
      sum += log_sum_exp - static_cast<double>(row_data[target]);
    }
    partial[part] = sum;
  });

  double total = 0.0;
  for (int64_t part = 0; part < kSumParts; ++part) {
    total += partial[part];
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

  // Строки независимы: у каждой свой softmax и своя цель.
  for_rows(rows, vocab, [&](int64_t row) {
    const float* row_data = data + row * vocab;
    float* grad_row = grad + row * vocab;

    // Экспоненты пишутся прямо в градиент: место там уже есть, и лишнего
    // прохода по памяти не получается.
    exp_shifted(row_data, row_maximum(row_data, vocab), grad_row, vocab);
    double sum_exp = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      sum_exp += grad_row[i];
    }
    const float inverse = static_cast<float>(1.0 / sum_exp);
    for (int64_t i = 0; i < vocab; ++i) {
      grad_row[i] *= inverse * scale;
    }
    // Вычесть единицу в позиции правильного токена — вся разница между
    // предсказанным распределением и целевым.
    grad_row[targets[static_cast<std::size_t>(row)]] -= scale;
  });
  return out;
}

namespace {

// Логарифм суммы экспонент строки, устойчивый к большим значениям.
double row_log_sum_exp(const float* row, int64_t width) {
  return log_sum_exp_into(row, row_maximum(row, width), row_scratch(width),
                          width);
}

}  // namespace

Tensor z_loss(const Tensor& logits) {
  LLM_CHECK_MSG(logits.rank() == 2, "z-loss ждёт матрицу (n, vocab)");
  const int64_t rows = logits.dim(0);
  const int64_t vocab = logits.dim(1);
  LLM_CHECK_GT(rows, static_cast<int64_t>(0));

  const Tensor dense = logits.contiguous();
  const float* data = dense.data();

  double partial[kSumParts] = {};
  for_row_parts(rows, vocab, [&](int64_t part, int64_t first, int64_t last) {
    double sum = 0.0;
    for (int64_t row = first; row < last; ++row) {
      const double z = row_log_sum_exp(data + row * vocab, vocab);
      sum += z * z;
    }
    partial[part] = sum;
  });

  double total = 0.0;
  for (int64_t part = 0; part < kSumParts; ++part) {
    total += partial[part];
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

  for_rows(rows, vocab, [&](int64_t row) {
    const float* row_data = data + row * vocab;
    float* grad_row = grad + row * vocab;

    const float maximum = row_maximum(row_data, vocab);
    exp_shifted(row_data, maximum, grad_row, vocab);
    double sum_exp = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      sum_exp += grad_row[i];
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
  });
  return out;
}

PredictionStats prediction_stats(const Tensor& logits,
                                 const std::vector<int32_t>& targets) {
  check_arguments(logits, targets);
  const int64_t rows = logits.dim(0);
  const int64_t vocab = logits.dim(1);

  const Tensor dense = logits.contiguous();
  const float* data = dense.data();

  int64_t partial_correct[kSumParts] = {};
  double partial_entropy[kSumParts] = {};
  double partial_log_z[kSumParts] = {};

  for_row_parts(rows, vocab, [&](int64_t part, int64_t first, int64_t last) {
    int64_t correct_part = 0;
    double entropy_part = 0.0;
    double log_z_part = 0.0;
    for (int64_t row = first; row < last; ++row) {
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
        ++correct_part;
      }

      // Экспоненты считаются один раз и сохраняются. Раньше их считали
      // дважды: сначала ради суммы, потом ради взвешенного среднего, — то
      // есть вся строка словаря проходила через экспоненту два раза подряд.
      float* probabilities = row_scratch(vocab);
      exp_shifted(row_data, maximum, probabilities, vocab);
      double sum_exp = 0.0;
      for (int64_t i = 0; i < vocab; ++i) {
        sum_exp += probabilities[i];
      }
      // Энтропия через логарифм суммы экспонент: H = log(S) + max - E[x], где
      // ожидание берётся по самому распределению. Так не нужен отдельный проход
      // с материализацией вероятностей.
      const double log_sum = std::log(sum_exp);
      double weighted = 0.0;
      for (int64_t i = 0; i < vocab; ++i) {
        weighted += (static_cast<double>(probabilities[i]) / sum_exp) *
                    static_cast<double>(row_data[i] - maximum);
      }
      entropy_part += log_sum - weighted;
      log_z_part += log_sum + static_cast<double>(maximum);
    }
    partial_correct[part] = correct_part;
    partial_entropy[part] = entropy_part;
    partial_log_z[part] = log_z_part;
  });

  int64_t correct = 0;
  double entropy_total = 0.0;
  double log_z_total = 0.0;
  for (int64_t part = 0; part < kSumParts; ++part) {
    correct += partial_correct[part];
    entropy_total += partial_entropy[part];
    log_z_total += partial_log_z[part];
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

  std::vector<float> losses(static_cast<std::size_t>(rows), 0.0f);
  for_rows(rows, vocab, [&](int64_t row) {
    const float* row_data = data + row * vocab;
    const double log_sum_exp = log_sum_exp_into(
        row_data, row_maximum(row_data, vocab), row_scratch(vocab), vocab);
    losses[static_cast<std::size_t>(row)] = static_cast<float>(
        log_sum_exp -
        static_cast<double>(row_data[targets[static_cast<std::size_t>(row)]]));
  });
  return losses;
}

}  // namespace ops
}  // namespace llm
