#include "train/optimizer.h"

#include <algorithm>
#include <cmath>

#include "core/check.h"

namespace llm {
namespace train {

AdamW::AdamW(std::vector<nn::NamedParameter> parameters,
             const AdamWConfig& config)
    : parameters_(std::move(parameters)), config_(config), step_(0) {
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const Tensor& value = parameters_[i].value->value();
    LLM_CHECK_MSG(value.is_contiguous(),
                  "параметр " << parameters_[i].name << " размещён неплотно");
    first_moment_.push_back(Tensor::zeros(value.shape()));
    second_moment_.push_back(Tensor::zeros(value.shape()));
    // Распад веса — только для матриц. Одномерные параметры (масштабы
    // нормировок) от него освобождены.
    apply_decay_.push_back(value.rank() >= 2);
  }
}

int64_t AdamW::decayed_parameter_count() const {
  int64_t total = 0;
  for (std::size_t i = 0; i < apply_decay_.size(); ++i) {
    if (apply_decay_[i]) {
      total += parameters_[i].value->numel();
    }
  }
  return total;
}

float AdamW::clip_grad_norm(float max_norm) {
  double sum_squares = 0.0;
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const Tensor& grad = parameters_[i].value->grad();
    if (!grad.defined()) {
      continue;
    }
    const float* data = grad.data();
    const int64_t count = grad.numel();
    for (int64_t element = 0; element < count; ++element) {
      sum_squares += static_cast<double>(data[element]) * data[element];
    }
  }
  const double norm = std::sqrt(sum_squares);
  if (norm <= static_cast<double>(max_norm) || norm == 0.0) {
    return static_cast<float>(norm);
  }

  const float scale = static_cast<float>(static_cast<double>(max_norm) / norm);
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    parameters_[i].value->node()->scale_grad(scale);
  }
  return static_cast<float>(norm);
}

void AdamW::step(float learning_rate) {
  ++step_;

  // Поправка на смещение. Моменты стартуют с нуля, поэтому первые оценки
  // занижены; деление на (1 - beta^t) это компенсирует. Без поправки первые
  // шаги были бы во много раз короче нужного.
  const double bias1 = 1.0 - std::pow(static_cast<double>(config_.beta1),
                                      static_cast<double>(step_));
  const double bias2 = 1.0 - std::pow(static_cast<double>(config_.beta2),
                                      static_cast<double>(step_));

  update_ratios_.assign(parameters_.size(), 0.0f);

  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const Tensor& grad = parameters_[i].value->grad();
    if (!grad.defined()) {
      continue;  // параметр не участвовал в этом шаге
    }

    Tensor& value = parameters_[i].value->value();
    LLM_DCHECK(grad.shape() == value.shape());

    float* weights = value.data();
    const float* gradients = grad.data();
    float* first = first_moment_[i].data();
    float* second = second_moment_[i].data();
    const int64_t count = value.numel();

    // Длины шага и самого веса копятся по ходу: отдельный проход ради них
    // стоил бы столько же, сколько сам шаг.
    double update_squares = 0.0;
    double weight_squares = 0.0;

    for (int64_t element = 0; element < count; ++element) {
      const float gradient = gradients[element];
      weight_squares +=
          static_cast<double>(weights[element]) * weights[element];

      first[element] =
          config_.beta1 * first[element] + (1.0f - config_.beta1) * gradient;
      second[element] = config_.beta2 * second[element] +
                        (1.0f - config_.beta2) * gradient * gradient;

      const double corrected_first =
          static_cast<double>(first[element]) / bias1;
      const double corrected_second =
          static_cast<double>(second[element]) / bias2;

      // Распад применяется к самому весу, а не к градиенту: в этом и состоит
      // отличие AdamW от Adam.
      if (apply_decay_[i] && config_.weight_decay != 0.0f) {
        weights[element] -=
            learning_rate * config_.weight_decay * weights[element];
      }

      const double update =
          corrected_first /
          (std::sqrt(corrected_second) + static_cast<double>(config_.eps));
      const double step = static_cast<double>(learning_rate) * update;
      update_squares += step * step;
      weights[element] -= static_cast<float>(step);
    }

    update_ratios_[i] = weight_squares > 0.0
                            ? static_cast<float>(std::sqrt(update_squares) /
                                                 std::sqrt(weight_squares))
                            : 0.0f;
  }

  // Медиана, а не среднее: у одного-двух параметров отношение бывает на
  // порядки больше, и среднее говорило бы только о них.
  std::vector<float> sorted;
  for (std::size_t i = 0; i < update_ratios_.size(); ++i) {
    if (update_ratios_[i] > 0.0f) {
      sorted.push_back(update_ratios_[i]);
    }
  }
  if (sorted.empty()) {
    last_update_ratio_ = 0.0f;
  } else {
    std::sort(sorted.begin(), sorted.end());
    last_update_ratio_ = sorted[sorted.size() / 2];
  }
}

void AdamW::zero_grad() {
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    parameters_[i].value->zero_grad();
  }
}

}  // namespace train
}  // namespace llm
