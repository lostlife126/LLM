// Отпечатки вычислений — для сверки между машинами.
//
// Проект держит инвариант: результат не зависит ни от числа ядер, ни от
// набора инструкций, ни от компилятора. Внутри одной машины это проверяют
// тесты — они сверяют между собой те ядра, что на ней доступны. Но между
// машинами сверить было нечем: на x86 нет NEON, на ARM нет AVX2, и тест,
// запущенный там и там, сравнивает разные пары.
//
// Эта программа печатает отпечаток результата каждого доступного ядра.
// Запускается на двух машинах, выводы сравниваются глазами или diff-ом.
// Ядра, помеченные слитными, обязаны давать один и тот же отпечаток везде.
//
// Про входные данные — отдельно, потому что на этом здесь уже попались.
// Они строятся целочисленно и переводятся в float одним делением. Выражение
// вида a + b * c писать нельзя: на ARM компилятор склеит его в слитное
// умножение с накоплением, на базовом x86-64 не склеит, и массивы окажутся
// разными ещё до того, как начнётся то, что мы сравниваем. Первая версия
// сверки именно так и врала: расходились не результаты, а входы.
//
// Запуск: ./fingerprint

#include <cstdint>
#include <cstdio>
#include <vector>

#include "args.h"

#include "core/cpu.h"
#include "core/thread_pool.h"
#include "ops/fast_exp.h"
#include "ops/gemm.h"
#include "ops/micro_kernel.h"

namespace {

std::uint64_t fingerprint(const std::vector<float>& values) {
  // FNV-1a по байтам представления: сравнивать надо биты, а не значения.
  std::uint64_t hash = 1469598103934665603ull;
  const unsigned char* bytes =
      reinterpret_cast<const unsigned char*>(values.data());
  for (std::size_t i = 0; i < values.size() * sizeof(float); ++i) {
    hash = (hash ^ bytes[i]) * 1099511628211ull;
  }
  return hash;
}

void check_gemm() {
  // Размеры нарочно не кратны ширинам плиток: края тоже под проверкой.
  const int64_t m = 61, n = 96, k = 77;
  std::vector<float> a(m * k), b(k * n), c(m * n);
  for (int64_t i = 0; i < m * k; ++i) {
    a[i] = static_cast<float>((i * 7919 % 2003) - 1000) / 997.0f;
  }
  for (int64_t i = 0; i < k * n; ++i) {
    b[i] = static_cast<float>((i * 6271 % 1997) - 1000) / 991.0f;
  }

  std::printf("\nумножение матриц %lldx%lldx%lld\n", static_cast<long long>(m),
              static_cast<long long>(n), static_cast<long long>(k));
  int count = 0;
  const llm::ops::MicroKernelChoice* table = llm::ops::all_micro_kernels(&count);
  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      continue;
    }
    llm::ops::force_micro_kernel(&table[i].kernel);
    for (std::size_t j = 0; j < c.size(); ++j) {
      c[j] = 0.0f;
    }
    llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                   c.data(), n);
    std::printf("  %-18s слитное=%d  %016llx\n", table[i].kernel.name,
                table[i].kernel.fused ? 1 : 0,
                static_cast<unsigned long long>(fingerprint(c)));
  }
  llm::ops::force_micro_kernel(nullptr);
}

void check_exp() {
  const int64_t n = 4099;  // некратно 4, 8 и 16: хвост тоже под проверкой
  std::vector<float> x(n), y(n);
  for (int64_t i = 0; i < n; ++i) {
    // От -110 до +90 с шагом 0.05: и переполнение, и субнормальная область.
    x[i] = static_cast<float>(static_cast<int>(i % 4001) * 5 - 11000) / 100.0f;
  }

  std::printf("\nэкспонента, %lld значений\n", static_cast<long long>(n));
  int count = 0;
  const llm::ops::ExpKernelChoice* table = llm::ops::all_exp_kernels(&count);
  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      continue;
    }
    llm::ops::force_exp_kernel(&table[i].kernel);
    llm::ops::exp_array(x.data(), y.data(), n);
    std::printf("  %-18s слитное=%d  %016llx\n", table[i].kernel.name,
                table[i].kernel.fused ? 1 : 0,
                static_cast<unsigned long long>(fingerprint(y)));
  }
  llm::ops::force_exp_kernel(nullptr);
}

}  // namespace

int main(int argc, char**) {
  // Аргументов это приложение не принимает, и лишний обязан
  // получить отказ, а не быть отброшенным молча.
  bench::expect_at_most(argc, 0, "без аргументов");
  llm::set_parallel_width(1);
  std::printf("процессор: %s\n", llm::cpu_features().to_string().c_str());
  check_gemm();
  check_exp();
  std::printf(
      "\nВСЕ строки обязаны совпадать между машинами, и слитные, и"
      " раздельные.\n"
      "Раньше здесь стояла оговорка «остальные могут отличаться»: на aarch64\n"
      "компилятор сам склеивал умножение со сложением, и раздельные ядра\n"
      "оказывались слитными. Сборка идёт с -ffp-contract=off, склейки больше\n"
      "не происходит, и оговорка не нужна.\n");
  return 0;
}
