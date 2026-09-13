#include "ops/gemm.h"

#include <algorithm>
#include <vector>

#include "core/check.h"

namespace llm {
namespace ops {
namespace {

// Размеры плитки микроядра, подобранные замером (apps/bench_gemm) на этой
// машине. 4 x 32 оказалось оптимумом и у GCC, и у clang; соседние формы
// (4 x 24, 6 x 16, 2 x 32) отстают на 5-15%, а 4 x 8 и 4 x 48 — вдвое.
constexpr int64_t kMicroM = 4;
constexpr int64_t kMicroN = 32;

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
// kMicroM строк. Внутри панели порядок (p, ii) — сначала шаг по глубине,
// потом по строке.
//
// Упаковка делает три вещи сразу:
//   1) микроядро читает память подряд, независимо от исходных шагов;
//   2) транспонирование разрешается здесь, один раз, а не в каждой итерации
//      внутреннего цикла;
//   3) хвост дополняется нулями до полной панели, поэтому микроядро не
//      нуждается в проверках границ — нули не портят сумму.
void pack_a(bool transpose_a, const float* a, int64_t lda, int64_t row0,
            int64_t col0, int64_t mc, int64_t kc, float* apack) {
  const int64_t panels = ceil_div(mc, kMicroM);
  for (int64_t panel = 0; panel < panels; ++panel) {
    float* dst = apack + panel * kc * kMicroM;
    for (int64_t p = 0; p < kc; ++p) {
      for (int64_t ii = 0; ii < kMicroM; ++ii) {
        const int64_t i = panel * kMicroM + ii;
        float value = 0.0f;
        if (i < mc) {
          value = transpose_a ? a[(col0 + p) * lda + (row0 + i)]
                              : a[(row0 + i) * lda + (col0 + p)];
        }
        dst[p * kMicroM + ii] = value;
      }
    }
  }
}

// Упаковка блока B: kc x nc матрицы op(B), панелями по kMicroN столбцов.
void pack_b(bool transpose_b, const float* b, int64_t ldb, int64_t row0,
            int64_t col0, int64_t kc, int64_t nc, float* bpack) {
  const int64_t panels = ceil_div(nc, kMicroN);
  for (int64_t panel = 0; panel < panels; ++panel) {
    float* dst = bpack + panel * kc * kMicroN;
    for (int64_t p = 0; p < kc; ++p) {
      for (int64_t jj = 0; jj < kMicroN; ++jj) {
        const int64_t j = panel * kMicroN + jj;
        float value = 0.0f;
        if (j < nc) {
          value = transpose_b ? b[(col0 + j) * ldb + (row0 + p)]
                              : b[(row0 + p) * ldb + (col0 + j)];
        }
        dst[p * kMicroN + jj] = value;
      }
    }
  }
}

// Микроядро: считает плитку kMicroM x kMicroN произведения и добавляет её в C.
//
// Главное отличие от наивного цикла — накопление в локальном массиве, а не
// прямо в C. За один проход по глубине kc плитка загружает kMicroM значений A
// и kMicroN значений B, то есть 20 чисел, и делает на них 4 * 16 * 2 = 128
// операций. У наивного варианта на каждое умножение приходится своя загрузка,
// поэтому он упирается в память, а не в арифметику.
//
// rows и cols — сколько строк и столбцов плитки реально попадают в C: панели
// дополнены нулями, поэтому считать можно всегда полную плитку, а записывать
// только действительную часть.
void micro_kernel(int64_t kc, const float* __restrict apanel,
                  const float* __restrict bpanel, float alpha,
                  float* __restrict c, int64_t ldc, int64_t rows,
                  int64_t cols) {
  static_assert(kMicroM == 4, "развёртка ниже рассчитана ровно на 4 строки");

  // Аккумуляторы каждой строки — отдельный массив, а сами строки развёрнуты
  // вручную. Это не украшательство, а разница в 6 раз, измеренная на этой
  // машине: у варианта с вложенным циклом по строкам GCC не разворачивает
  // внешний цикл, аккумуляторы остаются в стеке, и каждое умножение
  // превращается в загрузку, сложение и выгрузку — 4 GFLOPS против 23.
  // Clang справляется с обеими формами, но опираться на это нельзя: разница
  // между 4 x 16 и 4 x 32 у GCC доходила до двадцати раз, и такой обрыв под
  // фундаментом проекта недопустим.
  float acc0[kMicroN] = {};
  float acc1[kMicroN] = {};
  float acc2[kMicroN] = {};
  float acc3[kMicroN] = {};

  for (int64_t p = 0; p < kc; ++p) {
    const float* a_values = apanel + p * kMicroM;
    const float a0 = a_values[0];
    const float a1 = a_values[1];
    const float a2 = a_values[2];
    const float a3 = a_values[3];
    const float* b_values = bpanel + p * kMicroN;
    // Одна загрузка B обслуживает сразу четыре строки: именно здесь берётся
    // арифметическая интенсивность. На 36 загруженных чисел приходится
    // 4 * 32 * 2 = 256 операций.
    for (int64_t jj = 0; jj < kMicroN; ++jj) {
      const float b = b_values[jj];
      acc0[jj] += a0 * b;
      acc1[jj] += a1 * b;
      acc2[jj] += a2 * b;
      acc3[jj] += a3 * b;
    }
  }

  const float* accumulators[kMicroM] = {acc0, acc1, acc2, acc3};
  for (int64_t ii = 0; ii < rows; ++ii) {
    float* c_row = c + ii * ldc;
    const float* acc = accumulators[ii];
    for (int64_t jj = 0; jj < cols; ++jj) {
      c_row[jj] += alpha * acc[jj];
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

  // Буферы упаковки выделяются по фактическому размеру блоков, поэтому на
  // маленьких матрицах не платим за буферы, рассчитанные на большие.
  const int64_t mc_max = std::min(kBlockM, m);
  const int64_t nc_max = std::min(kBlockN, n);
  const int64_t kc_max = std::min(kBlockK, k);
  std::vector<float> apack(
      static_cast<std::size_t>(ceil_div(mc_max, kMicroM) * kMicroM * kc_max));
  std::vector<float> bpack(
      static_cast<std::size_t>(ceil_div(nc_max, kMicroN) * kMicroN * kc_max));

  // Порядок блочных циклов: jc снаружи, затем pc, затем ic. При таком порядке
  // упакованный блок B переиспользуется всеми блоками строк A, а он самый
  // большой и дороже всех в упаковке.
  for (int64_t jc = 0; jc < n; jc += kBlockN) {
    const int64_t nc = std::min(kBlockN, n - jc);
    for (int64_t pc = 0; pc < k; pc += kBlockK) {
      const int64_t kc = std::min(kBlockK, k - pc);
      pack_b(transpose_b, b, ldb, pc, jc, kc, nc, bpack.data());

      for (int64_t ic = 0; ic < m; ic += kBlockM) {
        const int64_t mc = std::min(kBlockM, m - ic);
        pack_a(transpose_a, a, lda, ic, pc, mc, kc, apack.data());

        for (int64_t i = 0; i < mc; i += kMicroM) {
          const float* apanel = apack.data() + (i / kMicroM) * kc * kMicroM;
          const int64_t rows = std::min(kMicroM, mc - i);
          for (int64_t j = 0; j < nc; j += kMicroN) {
            const float* bpanel = bpack.data() + (j / kMicroN) * kc * kMicroN;
            const int64_t cols = std::min(kMicroN, nc - j);
            micro_kernel(kc, apanel, bpanel, alpha,
                         c + (ic + i) * ldc + (jc + j), ldc, rows, cols);
          }
        }
      }
    }
  }
}

}  // namespace ops
}  // namespace llm
