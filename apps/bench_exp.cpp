// Замер экспоненты.
//
// Смысл замера — три отношения. Первое: во сколько раз векторное ядро быстрее
// поэлементного std::exp, ради которого всё и затевалось. Второе: во сколько
// раз AVX-512 быстрее AVX2, — и это надо мерить, а не предполагать, потому что
// на широких инструкциях серверные Xeon снижают частоту. Третье выяснилось
// здесь же и не предполагалось вовсе: свой скалярный многочлен вдвое медленнее
// std::exp, и именно поэтому запасным путём стал libm, а не он.
//
// Длина массива взята такой, чтобы он помещался в L2: замеряется скорость
// арифметики, а не пропускная способность памяти. Отдельной строкой идёт
// длина, заведомо не влезающая в кэш, — чтобы было видно, где упор смещается
// с арифметики на память.
//
// Запуск: ./bench_exp

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "core/cpu.h"
#include "core/random.h"
#include "ops/fast_exp.h"

namespace {

using Clock = std::chrono::steady_clock;

// Сколько независимых серий делается на один замер и берётся лучшая. Причина
// та же, что в bench_gemm: машина общая, и одна серия ловит чужую нагрузку.
constexpr int kTrials = 5;

// Миллионов значений в секунду. Прогоняет столько раз, чтобы суммарное время
// серии превысило её долю от target_seconds: у короткой задачи разброс одного
// запуска сравним с самим временем.
template <typename Fn>
double measure(Fn fn, std::int64_t count, double target_seconds) {
  fn();  // Прогрев кэшей и страниц памяти.

  double best_seconds = 0.0;
  for (int trial = 0; trial < kTrials; ++trial) {
    int repetitions = 0;
    const Clock::time_point start = Clock::now();
    double elapsed = 0.0;
    do {
      fn();
      ++repetitions;
      elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    } while (elapsed < target_seconds / kTrials);

    const double seconds = elapsed / repetitions;
    if (best_seconds == 0.0 || seconds < best_seconds) {
      best_seconds = seconds;
    }
  }
  return static_cast<double>(count) / best_seconds / 1e6;
}

void run_case(std::int64_t count) {
  llm::Rng rng(2026);
  std::vector<float> input(static_cast<std::size_t>(count));
  std::vector<float> output(static_cast<std::size_t>(count), 0.0f);
  for (std::size_t i = 0; i < input.size(); ++i) {
    // Диапазон как в softmax: аргумент там не больше нуля.
    input[i] = rng.uniform(-30.0f, 0.0f);
  }

  std::printf("%10lld", static_cast<long long>(count));

  int kernels = 0;
  const llm::ops::ExpKernelChoice* table = llm::ops::all_exp_kernels(&kernels);
  double libm_rate = 0.0;
  double best_rate = 0.0;
  for (int i = 0; i < kernels; ++i) {
    if (!table[i].available) {
      std::printf("  %12s", "-");
      continue;
    }
    llm::ops::force_exp_kernel(&table[i].kernel);
    const double rate = measure(
        [&]() { llm::ops::exp_array(input.data(), output.data(), count); },
        count, 0.3);
    std::printf("  %12.1f", rate);
    if (std::string(table[i].kernel.name) == "libm") {
      libm_rate = rate;
    }
    best_rate = rate > best_rate ? rate : best_rate;
  }
  llm::ops::force_exp_kernel(nullptr);
  if (libm_rate > 0.0) {
    std::printf("  %7.1fx", best_rate / libm_rate);
  }
  std::printf("\n");
  std::fflush(stdout);
}

}  // namespace

int main() {
  std::printf("процессор: %s\n\n", llm::cpu_features().to_string().c_str());
  std::printf("миллионов значений в секунду, один поток\n");
  std::printf("%10s", "длина");

  int kernels = 0;
  const llm::ops::ExpKernelChoice* table = llm::ops::all_exp_kernels(&kernels);
  for (int i = 0; i < kernels; ++i) {
    std::printf("  %12s", table[i].kernel.name);
  }
  std::printf("  %8s\n", "к libm");
  std::printf(
      "--------------------------------------------------------------------"
      "----\n");

  // 4096 значений — 16 КБ, помещается в L1.
  // 262144 — один мегабайт, помещается в L2.
  // 16777216 — 64 МБ, не помещается никуда.
  const std::int64_t lengths[] = {4096, 262144, 16777216};
  for (std::size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    run_case(lengths[i]);
  }

  std::printf("\nвыбрано автоматически: %s\n",
              llm::ops::best_exp_kernel().name);
  return 0;
}
