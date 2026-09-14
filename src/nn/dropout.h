// Режим обучения: включён ли дропаут и откуда он берёт случайность.
//
// Устроено так же, как NoGradGuard: глобальный переключатель и объект,
// включающий его на время своей жизни. Причина та же — прямой проход модели
// объявлен константным и пронизан через пять уровней вызовов, и протаскивать
// сквозь них ещё один аргумент значило бы менять пять сигнатур ради флага,
// который по смыслу относится не к модели, а к тому, что мы с ней делаем.
//
// Выключено по умолчанию. Это важнее, чем кажется: забытый включённым дропаут
// при подсчёте проверочных потерь испортил бы их случайным образом, а на
// генерации давал бы разный текст при одинаковой затравке и одинаковом зерне
// сэмплирования. Поэтому включает его только цикл обучения, и ровно на время
// шага.

#ifndef LLM_NN_DROPOUT_H_
#define LLM_NN_DROPOUT_H_

#include <cstdint>

#include "autograd/node.h"
#include "core/random.h"

namespace llm {
namespace nn {

// Включает дропаут на время своей жизни. Зерно задаёт поток случайности:
// один и тот же прогон обучения с одним зерном воспроизводится в точности.
class TrainingScope {
 public:
  explicit TrainingScope(uint64_t seed);
  ~TrainingScope();

  TrainingScope(const TrainingScope&) = delete;
  TrainingScope& operator=(const TrainingScope&) = delete;

 private:
  bool saved_enabled_;
  Rng* saved_rng_;
  Rng rng_;
};

bool training_mode();

// Дропаут, который сам решает, нужен ли он. Вне режима обучения и при нулевой
// вероятности возвращает вход как есть — без узла на ленте и без затрат.
autograd::Var dropout(const autograd::Var& input, float probability);

}  // namespace nn
}  // namespace llm

#endif  // LLM_NN_DROPOUT_H_
