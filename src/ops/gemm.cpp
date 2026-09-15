#include "ops/gemm.h"

#include <algorithm>
#include <vector>

#include "core/check.h"
#include "ops/micro_kernel.h"

namespace llm {
namespace ops {
namespace {

// Форма плитки больше не константа: её задаёт выбранное микроядро, и от неё
// зависит раскладка упакованных панелей. Параметризация обязательна — иначе
// смена ядра молча поломала бы раскладку, а проявилось бы это неверными
// числами, а не ошибкой.

// Размеры блоков. Смысл — удержать переиспользуемые данные в нужном уровне
// кэша:
//   блок B  kBlockK x kBlockN = 128 x 256 float = 128 КБ — рассчитан на L2,
//           он переиспользуется всеми блоками строк A;
//   панель A kBlockM x kBlockK = 64 x 128 float = 32 КБ;
//   рабочий набор микроядра — 4 x 128 от A плюс 128 x 32 от B = 18 КБ,
//           рассчитан на L1 (32 КБ).
constexpr int64_t kBlockM = 64;
constexpr int64_t kBlockN = 256;
constexpr int64_t kBlockK = 128;

int64_t ceil_div(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

// Упаковка панели A: блок mc x kc матрицы op(A), разложенный панелями по
// mr строк. Внутри панели порядок (p, ii) — сначала шаг по глубине,
// потом по строке.
//
// Упаковка делает три вещи сразу:
//   1) микроядро читает память подряд, независимо от исходных шагов;
//   2) транспонирование разрешается здесь, один раз, а не в каждой итерации
//      внутреннего цикла;
//   3) хвост дополняется нулями до полной панели, поэтому микроядро не
//      нуждается в проверках границ — нули не портят сумму.
void pack_a(bool transpose_a, const float* a, int64_t lda, int64_t row0,
            int64_t col0, int64_t mc, int64_t kc, int64_t mr, float* apack) {
  const int64_t panels = ceil_div(mc, mr);
  for (int64_t panel = 0; panel < panels; ++panel) {
    float* dst = apack + panel * kc * mr;
    for (int64_t p = 0; p < kc; ++p) {
      for (int64_t ii = 0; ii < mr; ++ii) {
        const int64_t i = panel * mr + ii;
        float value = 0.0f;
        if (i < mc) {
          value = transpose_a ? a[(col0 + p) * lda + (row0 + i)]
                              : a[(row0 + i) * lda + (col0 + p)];
        }
        dst[p * mr + ii] = value;
      }
    }
  }
}

// Упаковка блока B: kc x nc матрицы op(B), панелями по nr столбцов.
void pack_b(bool transpose_b, const float* b, int64_t ldb, int64_t row0,
            int64_t col0, int64_t kc, int64_t nc, int64_t nr, float* bpack) {
  const int64_t panels = ceil_div(nc, nr);
  for (int64_t panel = 0; panel < panels; ++panel) {
    float* dst = bpack + panel * kc * nr;
    for (int64_t p = 0; p < kc; ++p) {
      for (int64_t jj = 0; jj < nr; ++jj) {
        const int64_t j = panel * nr + jj;
        float value = 0.0f;
        if (j < nc) {
          value = transpose_b ? b[(col0 + j) * ldb + (row0 + p)]
                              : b[(row0 + p) * ldb + (col0 + j)];
        }
        dst[p * nr + jj] = value;
      }
    }
  }
}

// Применение beta к C. Отдельным проходом до накопления: иначе при разбиении
// по глубине k масштабирование применилось бы к каждому блоку.
// beta == 0 означает перезапись, а не умножение на нуль: в C может лежать
// мусор или NaN, и умножение оставило бы NaN.
void scale_c(int64_t m, int64_t n, float beta, float* c, int64_t ldc) {
  if (beta == 1.0f) {
    return;
  }
  for (int64_t i = 0; i < m; ++i) {
    float* row = c + i * ldc;
    if (beta == 0.0f) {
      std::fill(row, row + n, 0.0f);
    } else {
      for (int64_t j = 0; j < n; ++j) {
        row[j] *= beta;
      }
    }
  }
}

void check_arguments(bool transpose_a, bool transpose_b, int64_t m, int64_t n,
                     int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  LLM_CHECK_GE(m, static_cast<int64_t>(0));
  LLM_CHECK_GE(n, static_cast<int64_t>(0));
  LLM_CHECK_GE(k, static_cast<int64_t>(0));
  // Шаг строки задан до транспонирования, поэтому нижняя граница зависит от
  // того, как матрица лежит в памяти, а не от формы op(A).
  LLM_CHECK_GE(lda, transpose_a ? m : k);
  LLM_CHECK_GE(ldb, transpose_b ? k : n);
  LLM_CHECK_GE(ldc, n);
}

}  // namespace

void gemm_naive(bool transpose_a, bool transpose_b, int64_t m, int64_t n,
                int64_t k, float alpha, const float* a, int64_t lda,
                const float* b, int64_t ldb, float beta, float* c,
                int64_t ldc) {
  check_arguments(transpose_a, transpose_b, m, n, k, lda, ldb, ldc);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        const float a_value = transpose_a ? a[p * lda + i] : a[i * lda + p];
        const float b_value = transpose_b ? b[j * ldb + p] : b[p * ldb + j];
        sum += a_value * b_value;
      }
      float& target = c[i * ldc + j];
      target = alpha * sum + (beta == 0.0f ? 0.0f : beta * target);
    }
  }
}

void gemm(bool transpose_a, bool transpose_b, int64_t m, int64_t n, int64_t k,
          float alpha, const float* a, int64_t lda, const float* b, int64_t ldb,
          float beta, float* c, int64_t ldc) {
  check_arguments(transpose_a, transpose_b, m, n, k, lda, ldb, ldc);

  scale_c(m, n, beta, c, ldc);
  if (m == 0 || n == 0 || k == 0 || alpha == 0.0f) {
    return;
  }

  // Микроядро выбирается по возможностям процессора — один раз, а не на
  // каждый вызов: best_micro_kernel считает выбор при первом обращении.
  const MicroKernel& kernel = best_micro_kernel();
  const int64_t mr = kernel.mr;
  const int64_t nr = kernel.nr;

  // Блоки округляются вверх до кратного плитке. Иначе последняя плитка блока
  // была бы неполной всегда, а не только на краю матрицы, и медленный путь
  // записи срабатывал бы постоянно.
  const int64_t block_m = ceil_div(kBlockM, mr) * mr;
  const int64_t block_n = ceil_div(kBlockN, nr) * nr;

  // Буферы упаковки выделяются по фактическому размеру блоков, поэтому на
  // маленьких матрицах не платим за буферы, рассчитанные на большие.
  const int64_t mc_max = std::min(block_m, m);
  const int64_t nc_max = std::min(block_n, n);
  const int64_t kc_max = std::min(kBlockK, k);
  std::vector<float> apack(
      static_cast<std::size_t>(ceil_div(mc_max, mr) * mr * kc_max));
  std::vector<float> bpack(
      static_cast<std::size_t>(ceil_div(nc_max, nr) * nr * kc_max));

  // Порядок блочных циклов: jc снаружи, затем pc, затем ic. При таком порядке
  // упакованный блок B переиспользуется всеми блоками строк A, а он самый
  // большой и дороже всех в упаковке.
  for (int64_t jc = 0; jc < n; jc += block_n) {
    const int64_t nc = std::min(block_n, n - jc);
    for (int64_t pc = 0; pc < k; pc += kBlockK) {
      const int64_t kc = std::min(kBlockK, k - pc);
      pack_b(transpose_b, b, ldb, pc, jc, kc, nc, nr, bpack.data());

      for (int64_t ic = 0; ic < m; ic += block_m) {
        const int64_t mc = std::min(block_m, m - ic);
        pack_a(transpose_a, a, lda, ic, pc, mc, kc, mr, apack.data());

        for (int64_t i = 0; i < mc; i += mr) {
          const float* apanel = apack.data() + (i / mr) * kc * mr;
          const int64_t rows = std::min(mr, mc - i);
          for (int64_t j = 0; j < nc; j += nr) {
            const float* bpanel = bpack.data() + (j / nr) * kc * nr;
            const int64_t cols = std::min(nr, nc - j);
            kernel.run(kc, apanel, bpanel, alpha, c + (ic + i) * ldc + (jc + j),
                       ldc, rows, cols);
          }
        }
      }
    }
  }
}

}  // namespace ops
}  // namespace llm
