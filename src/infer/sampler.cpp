#include "infer/sampler.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "core/check.h"

namespace llm {
namespace infer {
namespace {

const float kBlocked = -std::numeric_limits<float>::infinity();

int32_t argmax(const std::vector<float>& logits) {
  int32_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[static_cast<std::size_t>(best)]) {
      best = static_cast<int32_t>(i);
    }
  }
  return best;
}

void apply_repetition_penalty(std::vector<float>* logits,
                              const std::vector<int32_t>& history,
                              float penalty, int64_t window) {
  if (penalty == 1.0f || history.empty()) {
    return;
  }
  const std::size_t count = static_cast<std::size_t>(window);
  const std::size_t begin = history.size() > count ? history.size() - count : 0;
  for (std::size_t i = begin; i < history.size(); ++i) {
    const std::size_t token = static_cast<std::size_t>(history[i]);
    if (token >= logits->size()) {
      continue;
    }
    float& value = (*logits)[token];
    // У положительного логита штраф уменьшает величину делением, у
    // отрицательного — умножением. Единое деление сделало бы отрицательный
    // логит ближе к нулю, то есть повысило бы вероятность токена.
    value = value > 0.0f ? value / penalty : value * penalty;
  }
}

void keep_top_k(std::vector<float>* logits, int64_t top_k) {
  if (top_k <= 0 || top_k >= static_cast<int64_t>(logits->size())) {
    return;
  }
  std::vector<float> sorted(*logits);
  std::nth_element(sorted.begin(),
                   sorted.begin() + static_cast<std::ptrdiff_t>(top_k - 1),
                   sorted.end(), std::greater<float>());
  const float threshold = sorted[static_cast<std::size_t>(top_k - 1)];
  for (std::size_t i = 0; i < logits->size(); ++i) {
    if ((*logits)[i] < threshold) {
      (*logits)[i] = kBlocked;
    }
  }
}

// Softmax по вектору логитов. Отдельная функция, а не ops::softmax: здесь
// работа идёт с обычным вектором, без тензора и его формы.
std::vector<float> softmax(const std::vector<float>& logits) {
  float maximum = kBlocked;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    maximum = std::max(maximum, logits[i]);
  }
  std::vector<float> probabilities(logits.size());
  double total = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const float value = std::exp(logits[i] - maximum);
    probabilities[i] = value;
    total += value;
  }
  const float inverse = static_cast<float>(1.0 / total);
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    probabilities[i] *= inverse;
  }
  return probabilities;
}

void keep_top_p(std::vector<float>* logits, float top_p) {
  if (top_p <= 0.0f || top_p >= 1.0f) {
    return;
  }
  const std::vector<float> probabilities = softmax(*logits);

  std::vector<std::size_t> order(probabilities.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  // Устойчивая сортировка: при равных вероятностях порядок определяется
  // номером токена, и выбор остаётся воспроизводимым.
  std::stable_sort(order.begin(), order.end(),
                   [&probabilities](std::size_t a, std::size_t b) {
                     return probabilities[a] > probabilities[b];
                   });

  double accumulated = 0.0;
  std::size_t kept = 0;
  for (; kept < order.size(); ++kept) {
    accumulated += probabilities[order[kept]];
    // Токен, на котором сумма перевалила за порог, тоже остаётся: иначе при
    // p меньше вероятности самого вероятного токена не осталось бы ни одного.
    if (accumulated >= static_cast<double>(top_p)) {
      ++kept;
      break;
    }
  }
  for (std::size_t i = kept; i < order.size(); ++i) {
    (*logits)[order[i]] = kBlocked;
  }
}

}  // namespace

int32_t choose_by_cumulative(const std::vector<float>& probabilities,
                             double threshold) {
  LLM_CHECK_MSG(!probabilities.empty(), "розыгрыш по пустому распределению");
  double accumulated = 0.0;
  std::size_t last_positive = probabilities.size();
  for (std::size_t i = 0; i < probabilities.size(); ++i) {
    // Нулевые пропускаются, и это не оптимизация. uniform() возвращает
    // значение из [0, 1), то есть может вернуть ровно нуль, и тогда
    // накопленная сумма догоняет порог уже на нулевом токене — даже если он
    // отсечён и вероятность его нулевая. Выбран был бы заблокированный токен.
    //
    // Прибавление нуля к сумме ничего не меняет ни на один разряд, поэтому
    // пропуск не сдвигает розыгрыш: тот же порог даёт тот же номер.
    if (probabilities[i] <= 0.0f) {
      continue;
    }
    last_positive = i;
    accumulated += probabilities[i];
    if (accumulated >= threshold) {
      return static_cast<int32_t>(i);
    }
  }
  // Сюда попадают, когда сумма вышла чуть меньше порога из-за накопленной
  // погрешности. Возвращается последний НЕотсечённый токен, а не последний
  // подряд: последним подряд почти всегда оказывается как раз отсечённый, и
  // запасная ветвь выдавала бы токен, который top-k или top-p только что
  // запретили.
  LLM_CHECK_MSG(last_positive < probabilities.size(),
                "во всём распределении нет ни одной положительной вероятности");
  return static_cast<int32_t>(last_positive);
}

Sampler::Sampler(const SamplerConfig& config)
    : config_(config), rng_(config.seed) {
  // Проверки на то, что не падает само. Ноль в штрафе делил бы логит на нуль,
  // отрицательное окно превращалось бы при приведении к беззнаковому в
  // «сколько угодно», а NaN в температуре проходил бы мимо сравнения с нулём и
  // делал бы NaN из всех логитов сразу.
  LLM_CHECK_MSG(!(config.temperature != config.temperature),
                "температура не число");
  LLM_CHECK_MSG(config.repetition_penalty > 0.0f,
                "штраф за повтор " << config.repetition_penalty
                                   << " должен быть положительным");
  LLM_CHECK_MSG(config.repetition_window >= 0,
                "окно штрафа " << config.repetition_window << " отрицательно");
  LLM_CHECK_MSG(config.top_k >= 0,
                "top-k " << config.top_k << " отрицателен; ноль — выключено");
  LLM_CHECK_MSG(config.top_p >= 0.0f && config.top_p <= 1.0f,
                "top-p " << config.top_p
                         << " вне [0, 1]; ноль и единица — выключено");
}

int32_t Sampler::sample(std::vector<float>* logits,
                        const std::vector<int32_t>& history) {
  LLM_CHECK(logits != nullptr);
  LLM_CHECK_MSG(!logits->empty(), "пустой вектор логитов");

  apply_repetition_penalty(logits, history, config_.repetition_penalty,
                           config_.repetition_window);

  if (config_.temperature <= 0.0f) {
    // Жадный выбор. Температура ноль — это предел, в котором вся вероятность
    // сходится на самом большом логите, и разыгрывать уже нечего.
    return argmax(*logits);
  }

  for (std::size_t i = 0; i < logits->size(); ++i) {
    (*logits)[i] /= config_.temperature;
  }

  keep_top_k(logits, config_.top_k);
  keep_top_p(logits, config_.top_p);

  const std::vector<float> probabilities = softmax(*logits);
  return choose_by_cumulative(probabilities, rng_.uniform());
}

}  // namespace infer
}  // namespace llm
