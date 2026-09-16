// Замер времени — один на все бенчмарки проекта.
//
// Раньше эта дюжина строк была скопирована в каждый бенчмарк, и копии успели
// разойтись: где-то пять серий, где-то четыре, целевое время серии у всех
// разное. То есть методика, описанная в README одним абзацем, на деле была
// четырьмя разными. Здесь она одна.
//
// Почему лучшее, а не среднее. Считается всё на общей машине, и одна серия
// ловит не только свой код, но и то, чем в этот момент занят сосед по железу.
// Помехи при этом бывают только в одну сторону: чужая нагрузка может замедлить
// счёт, но не ускорить. Значит минимум по сериям и есть оценка того, на что
// машина способна, а среднее — оценка того, насколько она сегодня занята.
//
// Почему серий несколько. Разброс между запусками одной и той же сборки
// доходил до полутора раз — больше любого эффекта, который здесь измеряется.
// Одна серия такой разброс не переживает.

#ifndef LLM_APPS_MEASURE_H_
#define LLM_APPS_MEASURE_H_

#include <chrono>

namespace bench {

// Сколько независимых серий делается на один замер.
constexpr int kTrials = 5;

// Сколько времени отводится одной серии. У коротких задач разброс одного
// вызова сравним с самим временем, поэтому вызовы повторяются, пока серия не
// наберёт эту длительность.
constexpr double kSeriesSeconds = 0.15;

// Возвращает лучшее время одного вызова fn в секундах.
template <typename Fn>
double best_seconds(Fn fn, double series_seconds = kSeriesSeconds) {
  using Clock = std::chrono::steady_clock;

  // Один прогон вне замера: он прогревает кэши и страницы памяти.
  fn();

  double best = 0.0;
  for (int trial = 0; trial < kTrials; ++trial) {
    const Clock::time_point start = Clock::now();
    int repetitions = 0;
    double elapsed = 0.0;
    do {
      fn();
      ++repetitions;
      elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    } while (elapsed < series_seconds);
    const double seconds = elapsed / repetitions;
    if (best == 0.0 || seconds < best) {
      best = seconds;
    }
  }
  return best;
}

}  // namespace bench

#endif  // LLM_APPS_MEASURE_H_
