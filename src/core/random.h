// Генератор случайных чисел.
//
// Биты берутся у std::mt19937_64: её поведение стандарт фиксирует побитово,
// поэтому одно и то же зерно даёт одну и ту же последовательность на любом
// компиляторе.
//
// А вот распределения из <random> стандарт побитово НЕ фиксирует: реализации
// вправе считать uniform_real_distribution и normal_distribution по-разному.
// Поэтому распределения здесь свои. Иначе инициализация весов, перемешивание
// батчей и сэмплирование менялись бы при смене компилятора, и тесты на
// воспроизводимость обучения перестали бы что-либо значить.

#ifndef LLM_CORE_RANDOM_H_
#define LLM_CORE_RANDOM_H_

#include <cstdint>
#include <random>

#include "core/check.h"

namespace llm {

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : engine_(seed), has_spare_normal_(false) {}

  std::uint64_t next_bits() { return engine_(); }

  // Равномерно на [0, 1). Берём 53 старших бита — ровно столько значащих бит
  // у double, поэтому каждое представимое значение равновероятно.
  double uniform() {
    const std::uint64_t bits = next_bits() >> 11;
    return static_cast<double>(bits) * (1.0 / 9007199254740992.0);
  }

  float uniform(float low, float high) {
    return low + static_cast<float>(uniform()) * (high - low);
  }

  // Целое из [0, bound). Отбраковка остатка убирает смещение, которое даёт
  // простое взятие по модулю.
  std::uint64_t index(std::uint64_t bound) {
    LLM_DCHECK_GT(bound, static_cast<std::uint64_t>(0));
    const std::uint64_t limit = ~static_cast<std::uint64_t>(0) -
                                (~static_cast<std::uint64_t>(0) % bound);
    std::uint64_t value = next_bits();
    while (value >= limit) {
      value = next_bits();
    }
    return value % bound;
  }

  // Стандартное нормальное через преобразование Бокса — Мюллера в полярной
  // форме. За один вызов получается пара значений, второе сохраняется.
  float normal();

  float normal(float mean, float stddev) { return mean + stddev * normal(); }

 private:
  std::mt19937_64 engine_;
  bool has_spare_normal_;
  float spare_normal_ = 0.0f;
};

}  // namespace llm

#endif  // LLM_CORE_RANDOM_H_
