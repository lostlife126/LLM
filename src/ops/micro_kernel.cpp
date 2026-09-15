#include "ops/micro_kernel.h"

#include <algorithm>

#include "core/check.h"
#include "core/cpu.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define LLM_HAS_X86_SIMD 1
#else
#define LLM_HAS_X86_SIMD 0
#endif

namespace llm {
namespace ops {
namespace {

// --- скалярное ядро ---------------------------------------------------------
//
// Остаётся навсегда: это и запасной вариант для процессора без AVX2, и эталон,
// с которым в тестах сверяются векторные.
//
// Форма 4 x 32 и ручное развёртывание строк подобраны замером ещё до
// векторизации. Аккумуляторы каждой строки — отдельный массив: у варианта с
// вложенным циклом GCC не разворачивал внешний цикл, аккумуляторы оставались в
// стеке, и выходило 4 GFLOPS против 23.

constexpr int64_t kScalarM = 4;
constexpr int64_t kScalarN = 32;

void scalar_kernel(int64_t kc, const float* __restrict apanel,
                   const float* __restrict bpanel, float alpha,
                   float* __restrict c, int64_t ldc, int64_t rows,
                   int64_t cols) {
  float acc0[kScalarN] = {};
  float acc1[kScalarN] = {};
  float acc2[kScalarN] = {};
  float acc3[kScalarN] = {};

  for (int64_t p = 0; p < kc; ++p) {
    const float* a_values = apanel + p * kScalarM;
    const float a0 = a_values[0];
    const float a1 = a_values[1];
    const float a2 = a_values[2];
    const float a3 = a_values[3];
    const float* b_values = bpanel + p * kScalarN;
    for (int64_t jj = 0; jj < kScalarN; ++jj) {
      const float b = b_values[jj];
      acc0[jj] += a0 * b;
      acc1[jj] += a1 * b;
      acc2[jj] += a2 * b;
      acc3[jj] += a3 * b;
    }
  }

  const float* accumulators[kScalarM] = {acc0, acc1, acc2, acc3};
  for (int64_t ii = 0; ii < rows; ++ii) {
    float* c_row = c + ii * ldc;
    const float* acc = accumulators[ii];
    for (int64_t jj = 0; jj < cols; ++jj) {
      c_row[jj] += alpha * acc[jj];
    }
  }
}

#if LLM_HAS_X86_SIMD

// --- AVX2 -------------------------------------------------------------------
//
// Плитка 6 x 16: шесть строк по два вектора из восьми чисел. Форма не
// произвольная, а посчитанная по числу регистров. У AVX2 их шестнадцать:
// двенадцать уходят под аккумуляторы, два под загруженные значения B, один под
// размноженное значение A — пятнадцать из шестнадцати. Плитка шире не влезет,
// и компилятор начнёт выгружать аккумуляторы в стек, теряя ровно то, ради чего
// всё затевалось.
//
// Аккумуляторы объявлены отдельными переменными, а не массивом. Причина та же,
// что у скалярного ядра: массив — приглашение компилятору положить его в
// память.

constexpr int64_t kAvx2M = 6;
constexpr int64_t kAvx2N = 16;

__attribute__((target("avx2,fma"))) void avx2_kernel(
    int64_t kc, const float* __restrict apanel, const float* __restrict bpanel,
    float alpha, float* __restrict c, int64_t ldc, int64_t rows, int64_t cols) {
  __m256 acc00 = _mm256_setzero_ps();
  __m256 acc01 = _mm256_setzero_ps();
  __m256 acc10 = _mm256_setzero_ps();
  __m256 acc11 = _mm256_setzero_ps();
  __m256 acc20 = _mm256_setzero_ps();
  __m256 acc21 = _mm256_setzero_ps();
  __m256 acc30 = _mm256_setzero_ps();
  __m256 acc31 = _mm256_setzero_ps();
  __m256 acc40 = _mm256_setzero_ps();
  __m256 acc41 = _mm256_setzero_ps();
  __m256 acc50 = _mm256_setzero_ps();
  __m256 acc51 = _mm256_setzero_ps();

  for (int64_t p = 0; p < kc; ++p) {
    const float* b_values = bpanel + p * kAvx2N;
    const __m256 b0 = _mm256_loadu_ps(b_values);
    const __m256 b1 = _mm256_loadu_ps(b_values + 8);
    const float* a_values = apanel + p * kAvx2M;

    __m256 a = _mm256_broadcast_ss(a_values + 0);
    acc00 = _mm256_fmadd_ps(a, b0, acc00);
    acc01 = _mm256_fmadd_ps(a, b1, acc01);
    a = _mm256_broadcast_ss(a_values + 1);
    acc10 = _mm256_fmadd_ps(a, b0, acc10);
    acc11 = _mm256_fmadd_ps(a, b1, acc11);
    a = _mm256_broadcast_ss(a_values + 2);
    acc20 = _mm256_fmadd_ps(a, b0, acc20);
    acc21 = _mm256_fmadd_ps(a, b1, acc21);
    a = _mm256_broadcast_ss(a_values + 3);
    acc30 = _mm256_fmadd_ps(a, b0, acc30);
    acc31 = _mm256_fmadd_ps(a, b1, acc31);
    a = _mm256_broadcast_ss(a_values + 4);
    acc40 = _mm256_fmadd_ps(a, b0, acc40);
    acc41 = _mm256_fmadd_ps(a, b1, acc41);
    a = _mm256_broadcast_ss(a_values + 5);
    acc50 = _mm256_fmadd_ps(a, b0, acc50);
    acc51 = _mm256_fmadd_ps(a, b1, acc51);
  }

  const __m256 scale = _mm256_set1_ps(alpha);
  if (rows == kAvx2M && cols == kAvx2N) {
    // Быстрый путь — полная плитка. Он же подавляющее большинство вызовов:
    // неполные плитки бывают только по краям матрицы.
    float* r0 = c;
    _mm256_storeu_ps(r0, _mm256_fmadd_ps(scale, acc00, _mm256_loadu_ps(r0)));
    _mm256_storeu_ps(r0 + 8,
                     _mm256_fmadd_ps(scale, acc01, _mm256_loadu_ps(r0 + 8)));
    float* r1 = c + ldc;
    _mm256_storeu_ps(r1, _mm256_fmadd_ps(scale, acc10, _mm256_loadu_ps(r1)));
    _mm256_storeu_ps(r1 + 8,
                     _mm256_fmadd_ps(scale, acc11, _mm256_loadu_ps(r1 + 8)));
    float* r2 = c + 2 * ldc;
    _mm256_storeu_ps(r2, _mm256_fmadd_ps(scale, acc20, _mm256_loadu_ps(r2)));
    _mm256_storeu_ps(r2 + 8,
                     _mm256_fmadd_ps(scale, acc21, _mm256_loadu_ps(r2 + 8)));
    float* r3 = c + 3 * ldc;
    _mm256_storeu_ps(r3, _mm256_fmadd_ps(scale, acc30, _mm256_loadu_ps(r3)));
    _mm256_storeu_ps(r3 + 8,
                     _mm256_fmadd_ps(scale, acc31, _mm256_loadu_ps(r3 + 8)));
    float* r4 = c + 4 * ldc;
    _mm256_storeu_ps(r4, _mm256_fmadd_ps(scale, acc40, _mm256_loadu_ps(r4)));
    _mm256_storeu_ps(r4 + 8,
                     _mm256_fmadd_ps(scale, acc41, _mm256_loadu_ps(r4 + 8)));
    float* r5 = c + 5 * ldc;
    _mm256_storeu_ps(r5, _mm256_fmadd_ps(scale, acc50, _mm256_loadu_ps(r5)));
    _mm256_storeu_ps(r5 + 8,
                     _mm256_fmadd_ps(scale, acc51, _mm256_loadu_ps(r5 + 8)));
    return;
  }

  // Край матрицы: выгружаем плитку в буфер и дописываем поэлементно. Путь
  // редкий, поэтому простота здесь важнее скорости.
  float tile[kAvx2M * kAvx2N];
  _mm256_storeu_ps(tile + 0 * kAvx2N, acc00);
  _mm256_storeu_ps(tile + 0 * kAvx2N + 8, acc01);
  _mm256_storeu_ps(tile + 1 * kAvx2N, acc10);
  _mm256_storeu_ps(tile + 1 * kAvx2N + 8, acc11);
  _mm256_storeu_ps(tile + 2 * kAvx2N, acc20);
  _mm256_storeu_ps(tile + 2 * kAvx2N + 8, acc21);
  _mm256_storeu_ps(tile + 3 * kAvx2N, acc30);
  _mm256_storeu_ps(tile + 3 * kAvx2N + 8, acc31);
  _mm256_storeu_ps(tile + 4 * kAvx2N, acc40);
  _mm256_storeu_ps(tile + 4 * kAvx2N + 8, acc41);
  _mm256_storeu_ps(tile + 5 * kAvx2N, acc50);
  _mm256_storeu_ps(tile + 5 * kAvx2N + 8, acc51);
  for (int64_t ii = 0; ii < rows; ++ii) {
    float* c_row = c + ii * ldc;
    const float* acc = tile + ii * kAvx2N;
    for (int64_t jj = 0; jj < cols; ++jj) {
      c_row[jj] += alpha * acc[jj];
    }
  }
}

// --- AVX-512 ----------------------------------------------------------------
//
// Плитка 8 x 32: восемь строк по два вектора из шестнадцати чисел. Регистров у
// AVX-512 тридцать два, поэтому шестнадцать аккумуляторов, два значения B и
// одно размноженное A оставляют запас — можно было бы взять и плитку шире,
// но это вопрос замера, а не рассуждения, и мерить его надо на конкретной
// машине: на многих серверных Xeon широкие инструкции роняют частоту, и AVX-512
// проигрывает AVX2 при вдвое большей ширине.

constexpr int64_t kAvx512M = 8;
constexpr int64_t kAvx512N = 32;

__attribute__((target("avx512f,avx512bw,avx512vl"))) void avx512_kernel(
    int64_t kc, const float* __restrict apanel, const float* __restrict bpanel,
    float alpha, float* __restrict c, int64_t ldc, int64_t rows, int64_t cols) {
  __m512 acc00 = _mm512_setzero_ps();
  __m512 acc01 = _mm512_setzero_ps();
  __m512 acc10 = _mm512_setzero_ps();
  __m512 acc11 = _mm512_setzero_ps();
  __m512 acc20 = _mm512_setzero_ps();
  __m512 acc21 = _mm512_setzero_ps();
  __m512 acc30 = _mm512_setzero_ps();
  __m512 acc31 = _mm512_setzero_ps();
  __m512 acc40 = _mm512_setzero_ps();
  __m512 acc41 = _mm512_setzero_ps();
  __m512 acc50 = _mm512_setzero_ps();
  __m512 acc51 = _mm512_setzero_ps();
  __m512 acc60 = _mm512_setzero_ps();
  __m512 acc61 = _mm512_setzero_ps();
  __m512 acc70 = _mm512_setzero_ps();
  __m512 acc71 = _mm512_setzero_ps();

  for (int64_t p = 0; p < kc; ++p) {
    const float* b_values = bpanel + p * kAvx512N;
    const __m512 b0 = _mm512_loadu_ps(b_values);
    const __m512 b1 = _mm512_loadu_ps(b_values + 16);
    const float* a_values = apanel + p * kAvx512M;

    __m512 a = _mm512_set1_ps(a_values[0]);
    acc00 = _mm512_fmadd_ps(a, b0, acc00);
    acc01 = _mm512_fmadd_ps(a, b1, acc01);
    a = _mm512_set1_ps(a_values[1]);
    acc10 = _mm512_fmadd_ps(a, b0, acc10);
    acc11 = _mm512_fmadd_ps(a, b1, acc11);
    a = _mm512_set1_ps(a_values[2]);
    acc20 = _mm512_fmadd_ps(a, b0, acc20);
    acc21 = _mm512_fmadd_ps(a, b1, acc21);
    a = _mm512_set1_ps(a_values[3]);
    acc30 = _mm512_fmadd_ps(a, b0, acc30);
    acc31 = _mm512_fmadd_ps(a, b1, acc31);
    a = _mm512_set1_ps(a_values[4]);
    acc40 = _mm512_fmadd_ps(a, b0, acc40);
    acc41 = _mm512_fmadd_ps(a, b1, acc41);
    a = _mm512_set1_ps(a_values[5]);
    acc50 = _mm512_fmadd_ps(a, b0, acc50);
    acc51 = _mm512_fmadd_ps(a, b1, acc51);
    a = _mm512_set1_ps(a_values[6]);
    acc60 = _mm512_fmadd_ps(a, b0, acc60);
    acc61 = _mm512_fmadd_ps(a, b1, acc61);
    a = _mm512_set1_ps(a_values[7]);
    acc70 = _mm512_fmadd_ps(a, b0, acc70);
    acc71 = _mm512_fmadd_ps(a, b1, acc71);
  }

  float tile[kAvx512M * kAvx512N];
  _mm512_storeu_ps(tile + 0 * kAvx512N, acc00);
  _mm512_storeu_ps(tile + 0 * kAvx512N + 16, acc01);
  _mm512_storeu_ps(tile + 1 * kAvx512N, acc10);
  _mm512_storeu_ps(tile + 1 * kAvx512N + 16, acc11);
  _mm512_storeu_ps(tile + 2 * kAvx512N, acc20);
  _mm512_storeu_ps(tile + 2 * kAvx512N + 16, acc21);
  _mm512_storeu_ps(tile + 3 * kAvx512N, acc30);
  _mm512_storeu_ps(tile + 3 * kAvx512N + 16, acc31);
  _mm512_storeu_ps(tile + 4 * kAvx512N, acc40);
  _mm512_storeu_ps(tile + 4 * kAvx512N + 16, acc41);
  _mm512_storeu_ps(tile + 5 * kAvx512N, acc50);
  _mm512_storeu_ps(tile + 5 * kAvx512N + 16, acc51);
  _mm512_storeu_ps(tile + 6 * kAvx512N, acc60);
  _mm512_storeu_ps(tile + 6 * kAvx512N + 16, acc61);
  _mm512_storeu_ps(tile + 7 * kAvx512N, acc70);
  _mm512_storeu_ps(tile + 7 * kAvx512N + 16, acc71);

  if (rows == kAvx512M && cols == kAvx512N) {
    const __m512 scale = _mm512_set1_ps(alpha);
    for (int64_t ii = 0; ii < kAvx512M; ++ii) {
      float* c_row = c + ii * ldc;
      const float* acc = tile + ii * kAvx512N;
      _mm512_storeu_ps(c_row, _mm512_fmadd_ps(scale, _mm512_loadu_ps(acc),
                                              _mm512_loadu_ps(c_row)));
      _mm512_storeu_ps(c_row + 16,
                       _mm512_fmadd_ps(scale, _mm512_loadu_ps(acc + 16),
                                       _mm512_loadu_ps(c_row + 16)));
    }
    return;
  }
  for (int64_t ii = 0; ii < rows; ++ii) {
    float* c_row = c + ii * ldc;
    const float* acc = tile + ii * kAvx512N;
    for (int64_t jj = 0; jj < cols; ++jj) {
      c_row[jj] += alpha * acc[jj];
    }
  }
}

#endif  // LLM_HAS_X86_SIMD

const MicroKernelChoice* build_table(int* count) {
  static MicroKernelChoice table[3];
  static int size = 0;
  static bool ready = false;
  if (!ready) {
    const CpuFeatures& cpu = cpu_features();

    table[size].kernel =
        MicroKernel{kScalarM, kScalarN, &scalar_kernel, "скалярное 4x32"};
    table[size].available = true;
    ++size;

#if LLM_HAS_X86_SIMD
    table[size].kernel = MicroKernel{kAvx2M, kAvx2N, &avx2_kernel, "AVX2 6x16"};
    table[size].available = cpu.has_avx2_fma();
    ++size;

    table[size].kernel =
        MicroKernel{kAvx512M, kAvx512N, &avx512_kernel, "AVX-512 8x32"};
    table[size].available = cpu.has_avx512();
    ++size;
#endif
    ready = true;
  }
  *count = size;
  return table;
}

}  // namespace

const MicroKernelChoice* all_micro_kernels(int* count) {
  LLM_CHECK(count != nullptr);
  return build_table(count);
}

namespace {

const MicroKernel* g_forced = nullptr;

}  // namespace

void force_micro_kernel(const MicroKernel* kernel) { g_forced = kernel; }

const MicroKernel& best_micro_kernel() {
  if (g_forced != nullptr) {
    return *g_forced;
  }
  static const MicroKernel chosen = []() {
    int count = 0;
    const MicroKernelChoice* table = build_table(&count);
    // Сверху вниз: последнее доступное — самое широкое. Что «шире значит
    // быстрее» на этой машине, проверяется замером в bench_gemm; если окажется
    // иначе, порядок здесь и надо менять.
    MicroKernel best = table[0].kernel;
    for (int i = 0; i < count; ++i) {
      if (table[i].available) {
        best = table[i].kernel;
      }
    }
    return best;
  }();
  return chosen;
}

}  // namespace ops
}  // namespace llm
