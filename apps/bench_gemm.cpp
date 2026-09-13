// Замер производительности умножения матриц.
//
// Смысл замера — не абсолютная цифра, а два отношения: во сколько раз блочная
// версия быстрее наивной и какую долю от пика процессора она берёт. Первое
// показывает, что дала кэш-блокировка; второе — сколько ещё осталось на столе.
//
// Запуск: ./bench_gemm [повторов_минимум]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/random.h"
#include "ops/gemm.h"

namespace {

struct Measurement {
  double seconds = 0.0;
  double gflops = 0.0;
};

using Clock = std::chrono::steady_clock;

// Прогоняет умножение столько раз, чтобы суммарное время превысило
// target_seconds: у коротких задач разброс одного запуска сравним с самим
// временем.
template <typename Fn>
Measurement measure(Fn fn, double flops, double target_seconds) {
  // Один прогон вне замера: он прогревает кэши и страницы памяти.
  fn();

  int repetitions = 0;
  const Clock::time_point start = Clock::now();
  double elapsed = 0.0;
  do {
    fn();
    ++repetitions;
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  } while (elapsed < target_seconds);

  Measurement result;
  result.seconds = elapsed / repetitions;
  result.gflops = flops / result.seconds / 1e9;
  return result;
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

int main() {
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
  return 0;
}
