// Замер производительности умножения матриц.
//
// Смысл замера — не абсолютная цифра, а два отношения: во сколько раз блочная
// версия быстрее наивной и какую долю от пика процессора она берёт. Первое
// показывает, что дала кэш-блокировка; второе — сколько ещё осталось на столе.
//
// Запуск: ./bench_gemm [повторов_минимум]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "measure.h"

#include "core/cpu.h"
#include "core/random.h"
#include "core/thread_pool.h"
#include "ops/gemm.h"
#include "ops/micro_kernel.h"

namespace {

struct Measurement {
  double seconds = 0.0;
  double gflops = 0.0;
};

// Замер — общий для всех бенчмарков, см. apps/measure.h: там же записано,
// почему берётся лучшая серия, а не средняя.
template <typename Fn>
Measurement measure(Fn fn, double flops, double target_seconds) {
  Measurement best;
  best.seconds = bench::best_seconds(fn, target_seconds / bench::kTrials);
  best.gflops = flops / best.seconds / 1e9;
  return best;
}

// Пропускная способность блочного умножения на задаче m x n x k.
double measure_blocked(std::int64_t m, std::int64_t n, std::int64_t k) {
  llm::Rng rng(2026);
  std::vector<float> a(static_cast<std::size_t>(m * k));
  std::vector<float> b(static_cast<std::size_t>(k * n));
  std::vector<float> c(static_cast<std::size_t>(m * n), 0.0f);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = rng.uniform(-1.0f, 1.0f);
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = rng.uniform(-1.0f, 1.0f);
  }
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k);
  return measure(
             [&]() {
               llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k,
                              b.data(), n, 0.0f, c.data(), n);
             },
             flops, 0.3)
      .gflops;
}

void run_case(const std::string& label, std::int64_t m, std::int64_t n,
              std::int64_t k, bool include_naive) {
  llm::Rng rng(2026);
  std::vector<float> a(static_cast<std::size_t>(m * k));
  std::vector<float> b(static_cast<std::size_t>(k * n));
  std::vector<float> c(static_cast<std::size_t>(m * n), 0.0f);
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = rng.uniform(-1.0f, 1.0f);
  }
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = rng.uniform(-1.0f, 1.0f);
  }

  // Каждый элемент результата — k умножений и k сложений.
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k);

  const Measurement blocked = measure(
      [&]() {
        llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n,
                       0.0f, c.data(), n);
      },
      flops, 0.3);

  Measurement naive;
  if (include_naive) {
    naive = measure(
        [&]() {
          llm::ops::gemm_naive(false, false, m, n, k, 1.0f, a.data(), k,
                               b.data(), n, 0.0f, c.data(), n);
        },
        flops, 0.3);
  }

  std::printf("%-22s %6lld %6lld %6lld  %9.2f", label.c_str(),
              static_cast<long long>(m), static_cast<long long>(n),
              static_cast<long long>(k), blocked.gflops);
  if (include_naive) {
    std::printf("  %9.2f  %7.1fx\n", naive.gflops,
                blocked.gflops / naive.gflops);
  } else {
    std::printf("  %9s  %8s\n", "-", "-");
  }
  std::fflush(stdout);
}

}  // namespace

// Сравнение микроядер между собой.
//
// Ради этого замера ядра и вынесены в таблицу с принудительным выбором. Что
// AVX-512 быстрее AVX2, кажется очевидным — оно вдвое шире. На деле многие
// серверные Xeon при широких инструкциях снижают частоту, и выигрыш съедается
// целиком. Проверять это надо на той машине, где считать.
void compare_kernels() {
  int count = 0;
  const llm::ops::MicroKernelChoice* table =
      llm::ops::all_micro_kernels(&count);

  std::printf("процессор: %s, ядер %d\n",
              llm::cpu_features().to_string().c_str(),
              llm::detect_core_count());
  std::printf("\nмикроядра на одном потоке, квадрат 512\n");
  std::printf("%-18s %10s %10s\n", "микроядро", "GFLOPS", "доступно");
  std::printf("----------------------------------------\n");

  // Ядра сравниваются на одном потоке. Иначе в цифру попадает ещё и то, как
  // задача поделилась между ядрами процессора, а сравнивать надо микроядра.
  llm::set_parallel_width(1);
  const std::int64_t size = 512;
  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      std::printf("%-18s %10s %10s\n", table[i].kernel.name, "-", "нет");
      continue;
    }
    llm::ops::force_micro_kernel(&table[i].kernel);
    const double gflops = measure_blocked(size, size, size);
    std::printf("%-18s %10.1f %10s\n", table[i].kernel.name, gflops, "да");
  }
  llm::ops::force_micro_kernel(nullptr);
  llm::set_parallel_width(0);
  std::printf("\nвыбрано автоматически: %s\n\n",
              llm::ops::best_micro_kernel().name);
}

// Как растёт пропускная способность с числом потоков.
//
// Смысл замера — не «стало быстрее», а масштабируемость: во сколько раз
// быстрее на четырёх ядрах вместо одного. Идеал — вчетверо, и разница с
// идеалом показывает, что мешает: пропускная способность памяти, ложное
// разделение строк кэша или слишком мелкое деление работы.
//
// Формы взяты и большие, и мелкие: на мелких деление может оказаться в убыток,
// и порог, ниже которого работа не делится, проверяется именно здесь.
void compare_threads() {
  struct Case {
    const char* label;
    std::int64_t m;
    std::int64_t n;
    std::int64_t k;
  };
  const Case cases[] = {
      {"квадрат 64", 64, 64, 64},         {"квадрат 128", 128, 128, 128},
      {"квадрат 256", 256, 256, 256},     {"квадрат 512", 512, 512, 512},
      {"квадрат 1024", 1024, 1024, 1024}, {"nano: qkv", 1024, 128, 128},
      {"nano: голова", 1024, 1024, 128},  {"tiny: голова", 2048, 4096, 256}};

  llm::set_parallel_width(0);
  const int max_width = llm::parallel_width();

  std::printf("\nмасштабируемость по потокам (GFLOPS)\n");
  std::printf("%-22s", "задача");
  for (int width = 1; width <= max_width; ++width) {
    std::printf(" %8d", width);
  }
  std::printf("  %8s\n", "ускор.");
  std::printf(
      "-------------------------------------------------------------------\n");

  for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    std::printf("%-22s", cases[i].label);
    double first = 0.0;
    double last = 0.0;
    for (int width = 1; width <= max_width; ++width) {
      llm::set_parallel_width(width);
      const double gflops = measure_blocked(cases[i].m, cases[i].n, cases[i].k);
      std::printf(" %8.1f", gflops);
      if (width == 1) {
        first = gflops;
      }
      last = gflops;
    }
    std::printf("  %7.2fx\n", last / first);
    std::fflush(stdout);
  }
  llm::set_parallel_width(0);
  std::printf("\nпотоков по умолчанию: %d\n\n", llm::parallel_width());
}

int main() {
  compare_kernels();
  compare_threads();
  std::printf("%-22s %6s %6s %6s  %9s  %9s  %8s\n", "задача", "m", "n", "k",
              "блочный", "наивный", "ускор.");
  std::printf(
      "-------------------------------------------------------------------"
      "----------\n");

  // Квадратные задачи: как меняется эффективность с ростом размера.
  const std::int64_t squares[] = {64, 128, 256, 512, 1024};
  for (std::size_t i = 0; i < sizeof(squares) / sizeof(squares[0]); ++i) {
    const std::int64_t size = squares[i];
    run_case("квадрат " + std::to_string(size), size, size, size, size <= 512);
  }

  std::printf("\n");

  // Формы, которые реально встретятся в модели. Пресет nano: d_model 128,
  // FFN 352, батч 16 x 64 = 1024 токена. Пресет tiny: d_model 256, FFN 704.
  run_case("nano: qkv", 1024, 128, 128, true);
  run_case("nano: ffn вверх", 1024, 352, 128, true);
  run_case("nano: ffn вниз", 1024, 128, 352, true);
  run_case("nano: голова", 1024, 1024, 128, false);
  run_case("tiny: qkv", 2048, 256, 256, false);
  run_case("tiny: ffn вверх", 2048, 704, 256, false);
  run_case("tiny: голова", 2048, 4096, 256, false);

  // Формы внимания: по одной матрице на голову, и таких на слой batch * heads.
  // Они мелкие, и на них работает прямой путь — без упаковки вовсе.
  std::printf("\n");
  run_case("nano: внимание W*V", 64, 32, 64, false);
  run_case("tiny: внимание W*V", 128, 32, 128, false);
  return 0;
}
