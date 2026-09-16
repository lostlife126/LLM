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

// NEON, в отличие от AVX2, не нуждается в проверке во время работы: в ARMv8-A
// он обязателен, и если бинарник собран под aarch64, он есть. Поэтому здесь
// нет атрибутов target и нет ветвления по признакам — только условная
// компиляция.
#if defined(__aarch64__)
#include <arm_neon.h>
#define LLM_HAS_NEON 1
#else
#define LLM_HAS_NEON 0
#endif

namespace llm {
namespace ops {
namespace {

// --- транспонирующая упаковка -----------------------------------------------

// Простой вариант: читает подряд, пишет с шагом. Служит и запасным путём, и
// эталоном — векторный обязан давать ровно то же самое, тут перестановка
// значений, а не арифметика, поэтому «то же самое» здесь буквально.
void plain_transpose_pack(const float* src, int64_t lda, int64_t lanes,
                          int64_t rows, int64_t kc, float* dst) {
  for (int64_t lane = 0; lane < rows; ++lane) {
    const float* row = src + lane * lda;
    float* out = dst + lane;
    for (int64_t p = 0; p < kc; ++p) {
      out[p * lanes] = row[p];
    }
  }
  for (int64_t lane = rows; lane < lanes; ++lane) {
    float* out = dst + lane;
    for (int64_t p = 0; p < kc; ++p) {
      out[p * lanes] = 0.0f;
    }
  }
}

#if LLM_HAS_X86_SIMD

// Транспонирование блока 8 x 8 на месте.
//
// Три яруса перестановок: сначала соседние строки чередуются по парам
// значений, потом получившиеся четвёрки собираются в нужном порядке, и
// наконец меняются местами половины регистров. Двадцать четыре перестановки на
// шестьдесят четыре значения — против ста двадцати восьми обращений к памяти у
// поэлементного варианта.
//
// Перестановка половин отдельным ярусом нужна потому, что у AVX2 почти все
// команды перемешивания работают внутри половины регистра и через её границу
// не заглядывают. Это же и причина, по которой блок именно 8 x 8: меньше не
// заполнит регистр, больше не переставить за один проход.
__attribute__((target("avx2"))) void transpose8x8(__m256* r) {
  const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]);
  const __m256 t1 = _mm256_unpackhi_ps(r[0], r[1]);
  const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]);
  const __m256 t3 = _mm256_unpackhi_ps(r[2], r[3]);
  const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]);
  const __m256 t5 = _mm256_unpackhi_ps(r[4], r[5]);
  const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]);
  const __m256 t7 = _mm256_unpackhi_ps(r[6], r[7]);

  const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
  const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
  const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
  const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
  const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
  const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

  r[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
  r[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
  r[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
  r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
  r[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
  r[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
  r[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
  r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

__attribute__((target("avx2"))) void avx2_transpose_pack(
    const float* src, int64_t lda, int64_t lanes, int64_t rows, int64_t kc,
    float* dst) {
  // Неполный блок бывает только на краю матрицы. Векторный путь там пришлось
  // бы обкладывать проверками, а выигрыш достался бы одному блоку из
  // нескольких десятков.
  if (rows < lanes) {
    plain_transpose_pack(src, lda, lanes, rows, kc, dst);
    return;
  }

  int64_t lane0 = 0;
  for (; lane0 + 8 <= lanes; lane0 += 8) {
    int64_t p = 0;
    for (; p + 8 <= kc; p += 8) {
      __m256 r[8];
      for (int64_t i = 0; i < 8; ++i) {
        r[i] = _mm256_loadu_ps(src + (lane0 + i) * lda + p);
      }
      transpose8x8(r);
      for (int64_t i = 0; i < 8; ++i) {
        _mm256_storeu_ps(dst + (p + i) * lanes + lane0, r[i]);
      }
    }
    for (; p < kc; ++p) {
      for (int64_t i = 0; i < 8; ++i) {
        dst[p * lanes + lane0 + i] = src[(lane0 + i) * lda + p];
      }
    }
  }

  // Остаток дорожек — это случай ядра AVX2, у которого плитка шириной шесть.
  // Блок всё равно транспонируется целиком, а записываются только настоящие
  // дорожки: запись под маской не трогает остальные вовсе, поэтому за границу
  // панели ничего не выходит.
  //
  // Недостающие строки блока читаются с последней действительной, а не с
  // нулей. Значения оттуда всё равно отбрасываются маской, зато чтение
  // заведомо не выходит за пределы матрицы и не нужен буфер нулей.
  const int64_t left = lanes - lane0;
  if (left <= 0) {
    return;
  }
  const __m256i mask =
      _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(left)),
                         _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
  int64_t p = 0;
  for (; p + 8 <= kc; p += 8) {
    __m256 r[8];
    for (int64_t i = 0; i < 8; ++i) {
      const int64_t lane = lane0 + (i < left ? i : left - 1);
      r[i] = _mm256_loadu_ps(src + lane * lda + p);
    }
    transpose8x8(r);
    for (int64_t i = 0; i < 8; ++i) {
      _mm256_maskstore_ps(dst + (p + i) * lanes + lane0, mask, r[i]);
    }
  }
  for (; p < kc; ++p) {
    for (int64_t i = 0; i < left; ++i) {
      dst[p * lanes + lane0 + i] = src[(lane0 + i) * lda + p];
    }
  }
}

#endif  // LLM_HAS_X86_SIMD

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

// Тот же расчёт, но A читается построчно, как лежит. Указатели на строки
// берутся заранее: недостающие показывают на последнюю действительную, и их
// вклад в C всё равно не записывается.
void scalar_kernel_rows(int64_t kc, const float* __restrict a, int64_t lda,
                        const float* __restrict b, int64_t bstride, float alpha,
                        float* __restrict c, int64_t ldc, int64_t rows,
                        int64_t cols) {
  float acc0[kScalarN] = {};
  float acc1[kScalarN] = {};
  float acc2[kScalarN] = {};
  float acc3[kScalarN] = {};

  const float* a0 = a;
  const float* a1 = a + (rows > 1 ? 1 : rows - 1) * lda;
  const float* a2 = a + (rows > 2 ? 2 : rows - 1) * lda;
  const float* a3 = a + (rows > 3 ? 3 : rows - 1) * lda;

  for (int64_t p = 0; p < kc; ++p) {
    const float va0 = a0[p];
    const float va1 = a1[p];
    const float va2 = a2[p];
    const float va3 = a3[p];
    const float* b_values = b + p * bstride;
    for (int64_t jj = 0; jj < kScalarN; ++jj) {
      const float value = b_values[jj];
      acc0[jj] += va0 * value;
      acc1[jj] += va1 * value;
      acc2[jj] += va2 * value;
      acc3[jj] += va3 * value;
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

__attribute__((target("avx2,fma"))) void avx2_kernel_rows(
    int64_t kc, const float* __restrict a, int64_t lda,
    const float* __restrict b, int64_t bstride, float alpha,
    float* __restrict c, int64_t ldc, int64_t rows, int64_t cols) {
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

  const float* a0 = a;
  const float* a1 = a + (rows > 1 ? 1 : rows - 1) * lda;
  const float* a2 = a + (rows > 2 ? 2 : rows - 1) * lda;
  const float* a3 = a + (rows > 3 ? 3 : rows - 1) * lda;
  const float* a4 = a + (rows > 4 ? 4 : rows - 1) * lda;
  const float* a5 = a + (rows > 5 ? 5 : rows - 1) * lda;

  for (int64_t p = 0; p < kc; ++p) {
    const float* b_values = b + p * bstride;
    const __m256 b0 = _mm256_loadu_ps(b_values);
    const __m256 b1 = _mm256_loadu_ps(b_values + 8);

    __m256 av = _mm256_broadcast_ss(a0 + p);
    acc00 = _mm256_fmadd_ps(av, b0, acc00);
    acc01 = _mm256_fmadd_ps(av, b1, acc01);
    av = _mm256_broadcast_ss(a1 + p);
    acc10 = _mm256_fmadd_ps(av, b0, acc10);
    acc11 = _mm256_fmadd_ps(av, b1, acc11);
    av = _mm256_broadcast_ss(a2 + p);
    acc20 = _mm256_fmadd_ps(av, b0, acc20);
    acc21 = _mm256_fmadd_ps(av, b1, acc21);
    av = _mm256_broadcast_ss(a3 + p);
    acc30 = _mm256_fmadd_ps(av, b0, acc30);
    acc31 = _mm256_fmadd_ps(av, b1, acc31);
    av = _mm256_broadcast_ss(a4 + p);
    acc40 = _mm256_fmadd_ps(av, b0, acc40);
    acc41 = _mm256_fmadd_ps(av, b1, acc41);
    av = _mm256_broadcast_ss(a5 + p);
    acc50 = _mm256_fmadd_ps(av, b0, acc50);
    acc51 = _mm256_fmadd_ps(av, b1, acc51);
  }

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

  const __m256 scale = _mm256_set1_ps(alpha);
  for (int64_t ii = 0; ii < rows; ++ii) {
    float* c_row = c + ii * ldc;
    const float* acc = tile + ii * kAvx2N;
    if (cols == kAvx2N) {
      _mm256_storeu_ps(c_row, _mm256_fmadd_ps(scale, _mm256_loadu_ps(acc),
                                              _mm256_loadu_ps(c_row)));
      _mm256_storeu_ps(c_row + 8,
                       _mm256_fmadd_ps(scale, _mm256_loadu_ps(acc + 8),
                                       _mm256_loadu_ps(c_row + 8)));
      continue;
    }
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

__attribute__((target("avx512f,avx512bw,avx512vl"))) void avx512_kernel_rows(
    int64_t kc, const float* __restrict a, int64_t lda,
    const float* __restrict b, int64_t bstride, float alpha,
    float* __restrict c, int64_t ldc, int64_t rows, int64_t cols) {
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

  const float* a0 = a;
  const float* a1 = a + (rows > 1 ? 1 : rows - 1) * lda;
  const float* a2 = a + (rows > 2 ? 2 : rows - 1) * lda;
  const float* a3 = a + (rows > 3 ? 3 : rows - 1) * lda;
  const float* a4 = a + (rows > 4 ? 4 : rows - 1) * lda;
  const float* a5 = a + (rows > 5 ? 5 : rows - 1) * lda;
  const float* a6 = a + (rows > 6 ? 6 : rows - 1) * lda;
  const float* a7 = a + (rows > 7 ? 7 : rows - 1) * lda;

  for (int64_t p = 0; p < kc; ++p) {
    const float* b_values = b + p * bstride;
    const __m512 b0 = _mm512_loadu_ps(b_values);
    const __m512 b1 = _mm512_loadu_ps(b_values + 16);

    __m512 av = _mm512_set1_ps(a0[p]);
    acc00 = _mm512_fmadd_ps(av, b0, acc00);
    acc01 = _mm512_fmadd_ps(av, b1, acc01);
    av = _mm512_set1_ps(a1[p]);
    acc10 = _mm512_fmadd_ps(av, b0, acc10);
    acc11 = _mm512_fmadd_ps(av, b1, acc11);
    av = _mm512_set1_ps(a2[p]);
    acc20 = _mm512_fmadd_ps(av, b0, acc20);
    acc21 = _mm512_fmadd_ps(av, b1, acc21);
    av = _mm512_set1_ps(a3[p]);
    acc30 = _mm512_fmadd_ps(av, b0, acc30);
    acc31 = _mm512_fmadd_ps(av, b1, acc31);
    av = _mm512_set1_ps(a4[p]);
    acc40 = _mm512_fmadd_ps(av, b0, acc40);
    acc41 = _mm512_fmadd_ps(av, b1, acc41);
    av = _mm512_set1_ps(a5[p]);
    acc50 = _mm512_fmadd_ps(av, b0, acc50);
    acc51 = _mm512_fmadd_ps(av, b1, acc51);
    av = _mm512_set1_ps(a6[p]);
    acc60 = _mm512_fmadd_ps(av, b0, acc60);
    acc61 = _mm512_fmadd_ps(av, b1, acc61);
    av = _mm512_set1_ps(a7[p]);
    acc70 = _mm512_fmadd_ps(av, b0, acc70);
    acc71 = _mm512_fmadd_ps(av, b1, acc71);
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

  const __m512 scale = _mm512_set1_ps(alpha);
  for (int64_t ii = 0; ii < rows; ++ii) {
    float* c_row = c + ii * ldc;
    const float* acc = tile + ii * kAvx512N;
    if (cols == kAvx512N) {
      _mm512_storeu_ps(c_row, _mm512_fmadd_ps(scale, _mm512_loadu_ps(acc),
                                              _mm512_loadu_ps(c_row)));
      _mm512_storeu_ps(c_row + 16,
                       _mm512_fmadd_ps(scale, _mm512_loadu_ps(acc + 16),
                                       _mm512_loadu_ps(c_row + 16)));
      continue;
    }
    for (int64_t jj = 0; jj < cols; ++jj) {
      c_row[jj] += alpha * acc[jj];
    }
  }
}

#endif  // LLM_HAS_X86_SIMD

#if LLM_HAS_NEON

// --- NEON -------------------------------------------------------------------
//
// У aarch64 тридцать два векторных регистра по четыре числа. Плитка 12 x 8
// занимает 24 аккумулятора, плюс три регистра под значения A и два под B —
// двадцать девять из тридцати двух.
//
// Ширина плитки восемь, а не двенадцать, и это не вкус. Прямой путь (без
// упаковки) включается по условию «общее число 32 делится на ширину плитки»
// — см. use_direct в gemm.cpp. Условие выбрано так, чтобы выбор пути не
// зависел от машины: у AVX2 ширина 16, у AVX-512 — 32, обе делят 32. Ширина
// 12 не делит, и тогда на ARM брался бы упакованный путь там, где на x86
// прямой, а результаты обучения разошлись бы между машинами.
//
// Умножение с накоплением берёт множитель прямо из дорожки регистра
// (vfmaq_laneq_f32), поэтому отдельной рассылки значения A по вектору, как на
// x86, здесь не нужно — это и экономит регистры под A.
//
// Порядок арифметики тот же, что у ядер x86: накопление идёт по глубине
// слитным умножением с накоплением, и в конце c += alpha * acc тоже слитно.
// Значит NEON обязан совпадать с AVX2 побитово, и это проверяется тестом.

// Транспонирование блока 4 x 4 на месте.
//
// Два яруса: сначала соседние строки чередуются по парам значений, потом
// собираются половины регистров. У NEON регистр вчетверо уже, чем у AVX-512, и
// втрое меньше ярусов, чем у блока 8 x 8 на AVX2, — зато и границы половины
// регистра, через которую там нельзя заглянуть одной командой, здесь нет:
// vtrn1q/vtrn2q работают по всему регистру.
inline void transpose4x4(float32x4_t* r) {
  const float32x4_t t0 = vtrn1q_f32(r[0], r[1]);
  const float32x4_t t1 = vtrn2q_f32(r[0], r[1]);
  const float32x4_t t2 = vtrn1q_f32(r[2], r[3]);
  const float32x4_t t3 = vtrn2q_f32(r[2], r[3]);
  r[0] = vcombine_f32(vget_low_f32(t0), vget_low_f32(t2));
  r[1] = vcombine_f32(vget_low_f32(t1), vget_low_f32(t3));
  r[2] = vcombine_f32(vget_high_f32(t0), vget_high_f32(t2));
  r[3] = vcombine_f32(vget_high_f32(t1), vget_high_f32(t3));
}

// Упаковка панели с транспонированием.
//
// Ширина плитки здесь восемь, то есть ровно два блока по четыре дорожки, и
// остатка по дорожкам не бывает — в отличие от ядра AVX2, где плитка шириной
// шесть и последний блок приходится записывать под маской.
void neon_transpose_pack(const float* src, int64_t lda, int64_t lanes,
                         int64_t rows, int64_t kc, float* dst) {
  // Неполный блок бывает только на краю матрицы: векторный путь пришлось бы
  // обкладывать проверками ради одного блока из нескольких десятков.
  if (rows < lanes) {
    plain_transpose_pack(src, lda, lanes, rows, kc, dst);
    return;
  }

  int64_t lane0 = 0;
  for (; lane0 + 4 <= lanes; lane0 += 4) {
    int64_t p = 0;
    for (; p + 4 <= kc; p += 4) {
      float32x4_t r[4];
      for (int64_t i = 0; i < 4; ++i) {
        r[i] = vld1q_f32(src + (lane0 + i) * lda + p);
      }
      transpose4x4(r);
      for (int64_t i = 0; i < 4; ++i) {
        vst1q_f32(dst + (p + i) * lanes + lane0, r[i]);
      }
    }
    // Хвост по глубине — поэлементно: он короче четырёх значений.
    for (; p < kc; ++p) {
      for (int64_t i = 0; i < 4; ++i) {
        dst[p * lanes + lane0 + i] = src[(lane0 + i) * lda + p];
      }
    }
  }
  // Хвост по дорожкам: у ширины восемь его не бывает, но ядро могло бы
  // получить другую ширину, и молча испортить панель было бы хуже.
  for (; lane0 < lanes; ++lane0) {
    const float* row = src + lane0 * lda;
    for (int64_t p = 0; p < kc; ++p) {
      dst[p * lanes + lane0] = row[p];
    }
  }
}

constexpr int64_t kNeonM = 12;
constexpr int64_t kNeonN = 8;

void neon_kernel(int64_t kc, const float* __restrict apanel,
                 const float* __restrict bpanel, float alpha,
                 float* __restrict c, int64_t ldc, int64_t rows, int64_t cols) {
  float32x4_t acc00 = vdupq_n_f32(0.0f);
  float32x4_t acc01 = vdupq_n_f32(0.0f);
  float32x4_t acc10 = vdupq_n_f32(0.0f);
  float32x4_t acc11 = vdupq_n_f32(0.0f);
  float32x4_t acc20 = vdupq_n_f32(0.0f);
  float32x4_t acc21 = vdupq_n_f32(0.0f);
  float32x4_t acc30 = vdupq_n_f32(0.0f);
  float32x4_t acc31 = vdupq_n_f32(0.0f);
  float32x4_t acc40 = vdupq_n_f32(0.0f);
  float32x4_t acc41 = vdupq_n_f32(0.0f);
  float32x4_t acc50 = vdupq_n_f32(0.0f);
  float32x4_t acc51 = vdupq_n_f32(0.0f);
  float32x4_t acc60 = vdupq_n_f32(0.0f);
  float32x4_t acc61 = vdupq_n_f32(0.0f);
  float32x4_t acc70 = vdupq_n_f32(0.0f);
  float32x4_t acc71 = vdupq_n_f32(0.0f);
  float32x4_t acc80 = vdupq_n_f32(0.0f);
  float32x4_t acc81 = vdupq_n_f32(0.0f);
  float32x4_t acc90 = vdupq_n_f32(0.0f);
  float32x4_t acc91 = vdupq_n_f32(0.0f);
  float32x4_t acc100 = vdupq_n_f32(0.0f);
  float32x4_t acc101 = vdupq_n_f32(0.0f);
  float32x4_t acc110 = vdupq_n_f32(0.0f);
  float32x4_t acc111 = vdupq_n_f32(0.0f);

  for (int64_t p = 0; p < kc; ++p) {
    const float* b_values = bpanel + p * kNeonN;
    const float32x4_t b0 = vld1q_f32(b_values);
    const float32x4_t b1 = vld1q_f32(b_values + 4);
    const float* a_values = apanel + p * kNeonM;
    const float32x4_t a0 = vld1q_f32(a_values);
    const float32x4_t a1 = vld1q_f32(a_values + 4);
    const float32x4_t a2 = vld1q_f32(a_values + 8);

    acc00 = vfmaq_laneq_f32(acc00, b0, a0, 0);
    acc01 = vfmaq_laneq_f32(acc01, b1, a0, 0);
    acc10 = vfmaq_laneq_f32(acc10, b0, a0, 1);
    acc11 = vfmaq_laneq_f32(acc11, b1, a0, 1);
    acc20 = vfmaq_laneq_f32(acc20, b0, a0, 2);
    acc21 = vfmaq_laneq_f32(acc21, b1, a0, 2);
    acc30 = vfmaq_laneq_f32(acc30, b0, a0, 3);
    acc31 = vfmaq_laneq_f32(acc31, b1, a0, 3);

    acc40 = vfmaq_laneq_f32(acc40, b0, a1, 0);
    acc41 = vfmaq_laneq_f32(acc41, b1, a1, 0);
    acc50 = vfmaq_laneq_f32(acc50, b0, a1, 1);
    acc51 = vfmaq_laneq_f32(acc51, b1, a1, 1);
    acc60 = vfmaq_laneq_f32(acc60, b0, a1, 2);
    acc61 = vfmaq_laneq_f32(acc61, b1, a1, 2);
    acc70 = vfmaq_laneq_f32(acc70, b0, a1, 3);
    acc71 = vfmaq_laneq_f32(acc71, b1, a1, 3);

    acc80 = vfmaq_laneq_f32(acc80, b0, a2, 0);
    acc81 = vfmaq_laneq_f32(acc81, b1, a2, 0);
    acc90 = vfmaq_laneq_f32(acc90, b0, a2, 1);
    acc91 = vfmaq_laneq_f32(acc91, b1, a2, 1);
    acc100 = vfmaq_laneq_f32(acc100, b0, a2, 2);
    acc101 = vfmaq_laneq_f32(acc101, b1, a2, 2);
    acc110 = vfmaq_laneq_f32(acc110, b0, a2, 3);
    acc111 = vfmaq_laneq_f32(acc111, b1, a2, 3);
  }

  if (rows == kNeonM && cols == kNeonN) {
    // Быстрый путь — полная плитка, и она же подавляющее большинство вызовов:
    // неполные бывают только по краям матрицы. Развёрнут целиком, чтобы
    // аккумуляторы не пришлось адресовать по вычисляемому индексу.
    float* row0 = c + 0 * ldc;
    vst1q_f32(row0, vfmaq_n_f32(vld1q_f32(row0), acc00, alpha));
    vst1q_f32(row0 + 4, vfmaq_n_f32(vld1q_f32(row0 + 4), acc01, alpha));
    float* row1 = c + 1 * ldc;
    vst1q_f32(row1, vfmaq_n_f32(vld1q_f32(row1), acc10, alpha));
    vst1q_f32(row1 + 4, vfmaq_n_f32(vld1q_f32(row1 + 4), acc11, alpha));
    float* row2 = c + 2 * ldc;
    vst1q_f32(row2, vfmaq_n_f32(vld1q_f32(row2), acc20, alpha));
    vst1q_f32(row2 + 4, vfmaq_n_f32(vld1q_f32(row2 + 4), acc21, alpha));
    float* row3 = c + 3 * ldc;
    vst1q_f32(row3, vfmaq_n_f32(vld1q_f32(row3), acc30, alpha));
    vst1q_f32(row3 + 4, vfmaq_n_f32(vld1q_f32(row3 + 4), acc31, alpha));
    float* row4 = c + 4 * ldc;
    vst1q_f32(row4, vfmaq_n_f32(vld1q_f32(row4), acc40, alpha));
    vst1q_f32(row4 + 4, vfmaq_n_f32(vld1q_f32(row4 + 4), acc41, alpha));
    float* row5 = c + 5 * ldc;
    vst1q_f32(row5, vfmaq_n_f32(vld1q_f32(row5), acc50, alpha));
    vst1q_f32(row5 + 4, vfmaq_n_f32(vld1q_f32(row5 + 4), acc51, alpha));
    float* row6 = c + 6 * ldc;
    vst1q_f32(row6, vfmaq_n_f32(vld1q_f32(row6), acc60, alpha));
    vst1q_f32(row6 + 4, vfmaq_n_f32(vld1q_f32(row6 + 4), acc61, alpha));
    float* row7 = c + 7 * ldc;
    vst1q_f32(row7, vfmaq_n_f32(vld1q_f32(row7), acc70, alpha));
    vst1q_f32(row7 + 4, vfmaq_n_f32(vld1q_f32(row7 + 4), acc71, alpha));
    float* row8 = c + 8 * ldc;
    vst1q_f32(row8, vfmaq_n_f32(vld1q_f32(row8), acc80, alpha));
    vst1q_f32(row8 + 4, vfmaq_n_f32(vld1q_f32(row8 + 4), acc81, alpha));
    float* row9 = c + 9 * ldc;
    vst1q_f32(row9, vfmaq_n_f32(vld1q_f32(row9), acc90, alpha));
    vst1q_f32(row9 + 4, vfmaq_n_f32(vld1q_f32(row9 + 4), acc91, alpha));
    float* row10 = c + 10 * ldc;
    vst1q_f32(row10, vfmaq_n_f32(vld1q_f32(row10), acc100, alpha));
    vst1q_f32(row10 + 4, vfmaq_n_f32(vld1q_f32(row10 + 4), acc101, alpha));
    float* row11 = c + 11 * ldc;
    vst1q_f32(row11, vfmaq_n_f32(vld1q_f32(row11), acc110, alpha));
    vst1q_f32(row11 + 4, vfmaq_n_f32(vld1q_f32(row11 + 4), acc111, alpha));
    return;
  }

  // Край матрицы. Здесь аккумуляторы всё равно уезжают в память, но путь
  // редкий, и важна на нём правильность границы, а не скорость.
  float values[kNeonM][kNeonN];
  vst1q_f32(values[0], acc00);
  vst1q_f32(values[0] + 4, acc01);
  vst1q_f32(values[1], acc10);
  vst1q_f32(values[1] + 4, acc11);
  vst1q_f32(values[2], acc20);
  vst1q_f32(values[2] + 4, acc21);
  vst1q_f32(values[3], acc30);
  vst1q_f32(values[3] + 4, acc31);
  vst1q_f32(values[4], acc40);
  vst1q_f32(values[4] + 4, acc41);
  vst1q_f32(values[5], acc50);
  vst1q_f32(values[5] + 4, acc51);
  vst1q_f32(values[6], acc60);
  vst1q_f32(values[6] + 4, acc61);
  vst1q_f32(values[7], acc70);
  vst1q_f32(values[7] + 4, acc71);
  vst1q_f32(values[8], acc80);
  vst1q_f32(values[8] + 4, acc81);
  vst1q_f32(values[9], acc90);
  vst1q_f32(values[9] + 4, acc91);
  vst1q_f32(values[10], acc100);
  vst1q_f32(values[10] + 4, acc101);
  vst1q_f32(values[11], acc110);
  vst1q_f32(values[11] + 4, acc111);
  for (int64_t ii = 0; ii < rows; ++ii) {
    float* c_row = c + ii * ldc;
    for (int64_t jj = 0; jj < cols; ++jj) {
      c_row[jj] += alpha * values[ii][jj];
    }
  }
}

// Тот же расчёт, но A читается построчно, как лежит: прямой путь без упаковки.
void neon_kernel_rows(int64_t kc, const float* __restrict a, int64_t lda,
                      const float* __restrict b, int64_t bstride, float alpha,
                      float* __restrict c, int64_t ldc, int64_t rows,
                      int64_t cols) {
  float32x4_t acc00 = vdupq_n_f32(0.0f);
  float32x4_t acc01 = vdupq_n_f32(0.0f);
  float32x4_t acc10 = vdupq_n_f32(0.0f);
  float32x4_t acc11 = vdupq_n_f32(0.0f);
  float32x4_t acc20 = vdupq_n_f32(0.0f);
  float32x4_t acc21 = vdupq_n_f32(0.0f);
  float32x4_t acc30 = vdupq_n_f32(0.0f);
  float32x4_t acc31 = vdupq_n_f32(0.0f);
  float32x4_t acc40 = vdupq_n_f32(0.0f);
  float32x4_t acc41 = vdupq_n_f32(0.0f);
  float32x4_t acc50 = vdupq_n_f32(0.0f);
  float32x4_t acc51 = vdupq_n_f32(0.0f);
  float32x4_t acc60 = vdupq_n_f32(0.0f);
  float32x4_t acc61 = vdupq_n_f32(0.0f);
  float32x4_t acc70 = vdupq_n_f32(0.0f);
  float32x4_t acc71 = vdupq_n_f32(0.0f);
  float32x4_t acc80 = vdupq_n_f32(0.0f);
  float32x4_t acc81 = vdupq_n_f32(0.0f);
  float32x4_t acc90 = vdupq_n_f32(0.0f);
  float32x4_t acc91 = vdupq_n_f32(0.0f);
  float32x4_t acc100 = vdupq_n_f32(0.0f);
  float32x4_t acc101 = vdupq_n_f32(0.0f);
  float32x4_t acc110 = vdupq_n_f32(0.0f);
  float32x4_t acc111 = vdupq_n_f32(0.0f);

  // Указатели на строки берутся заранее: недостающие показывают на последнюю
  // действительную, и их вклад в C всё равно не записывается.
  const float* r0 = a + (rows > 0 ? 0 : rows - 1) * lda;
  const float* r1 = a + (rows > 1 ? 1 : rows - 1) * lda;
  const float* r2 = a + (rows > 2 ? 2 : rows - 1) * lda;
  const float* r3 = a + (rows > 3 ? 3 : rows - 1) * lda;
  const float* r4 = a + (rows > 4 ? 4 : rows - 1) * lda;
  const float* r5 = a + (rows > 5 ? 5 : rows - 1) * lda;
  const float* r6 = a + (rows > 6 ? 6 : rows - 1) * lda;
  const float* r7 = a + (rows > 7 ? 7 : rows - 1) * lda;
  const float* r8 = a + (rows > 8 ? 8 : rows - 1) * lda;
  const float* r9 = a + (rows > 9 ? 9 : rows - 1) * lda;
  const float* r10 = a + (rows > 10 ? 10 : rows - 1) * lda;
  const float* r11 = a + (rows > 11 ? 11 : rows - 1) * lda;

  for (int64_t p = 0; p < kc; ++p) {
    const float* b_values = b + p * bstride;
    const float32x4_t b0 = vld1q_f32(b_values);
    const float32x4_t b1 = vld1q_f32(b_values + 4);
    // Значения A собираются в вектор только затем, чтобы множитель брался из
    // дорожки: загрузка каждого по отдельности стоила бы столько же.
    const float32x4_t a0 = {r0[p], r1[p], r2[p], r3[p]};
    const float32x4_t a1 = {r4[p], r5[p], r6[p], r7[p]};
    const float32x4_t a2 = {r8[p], r9[p], r10[p], r11[p]};

    acc00 = vfmaq_laneq_f32(acc00, b0, a0, 0);
    acc01 = vfmaq_laneq_f32(acc01, b1, a0, 0);
    acc10 = vfmaq_laneq_f32(acc10, b0, a0, 1);
    acc11 = vfmaq_laneq_f32(acc11, b1, a0, 1);
    acc20 = vfmaq_laneq_f32(acc20, b0, a0, 2);
    acc21 = vfmaq_laneq_f32(acc21, b1, a0, 2);
    acc30 = vfmaq_laneq_f32(acc30, b0, a0, 3);
    acc31 = vfmaq_laneq_f32(acc31, b1, a0, 3);

    acc40 = vfmaq_laneq_f32(acc40, b0, a1, 0);
    acc41 = vfmaq_laneq_f32(acc41, b1, a1, 0);
    acc50 = vfmaq_laneq_f32(acc50, b0, a1, 1);
    acc51 = vfmaq_laneq_f32(acc51, b1, a1, 1);
    acc60 = vfmaq_laneq_f32(acc60, b0, a1, 2);
    acc61 = vfmaq_laneq_f32(acc61, b1, a1, 2);
    acc70 = vfmaq_laneq_f32(acc70, b0, a1, 3);
    acc71 = vfmaq_laneq_f32(acc71, b1, a1, 3);

    acc80 = vfmaq_laneq_f32(acc80, b0, a2, 0);
    acc81 = vfmaq_laneq_f32(acc81, b1, a2, 0);
    acc90 = vfmaq_laneq_f32(acc90, b0, a2, 1);
    acc91 = vfmaq_laneq_f32(acc91, b1, a2, 1);
    acc100 = vfmaq_laneq_f32(acc100, b0, a2, 2);
    acc101 = vfmaq_laneq_f32(acc101, b1, a2, 2);
    acc110 = vfmaq_laneq_f32(acc110, b0, a2, 3);
    acc111 = vfmaq_laneq_f32(acc111, b1, a2, 3);
  }

  if (rows == kNeonM && cols == kNeonN) {
    // Быстрый путь — полная плитка, и она же подавляющее большинство вызовов:
    // неполные бывают только по краям матрицы. Развёрнут целиком, чтобы
    // аккумуляторы не пришлось адресовать по вычисляемому индексу.
    float* row0 = c + 0 * ldc;
    vst1q_f32(row0, vfmaq_n_f32(vld1q_f32(row0), acc00, alpha));
    vst1q_f32(row0 + 4, vfmaq_n_f32(vld1q_f32(row0 + 4), acc01, alpha));
    float* row1 = c + 1 * ldc;
    vst1q_f32(row1, vfmaq_n_f32(vld1q_f32(row1), acc10, alpha));
    vst1q_f32(row1 + 4, vfmaq_n_f32(vld1q_f32(row1 + 4), acc11, alpha));
    float* row2 = c + 2 * ldc;
    vst1q_f32(row2, vfmaq_n_f32(vld1q_f32(row2), acc20, alpha));
    vst1q_f32(row2 + 4, vfmaq_n_f32(vld1q_f32(row2 + 4), acc21, alpha));
    float* row3 = c + 3 * ldc;
    vst1q_f32(row3, vfmaq_n_f32(vld1q_f32(row3), acc30, alpha));
    vst1q_f32(row3 + 4, vfmaq_n_f32(vld1q_f32(row3 + 4), acc31, alpha));
    float* row4 = c + 4 * ldc;
    vst1q_f32(row4, vfmaq_n_f32(vld1q_f32(row4), acc40, alpha));
    vst1q_f32(row4 + 4, vfmaq_n_f32(vld1q_f32(row4 + 4), acc41, alpha));
    float* row5 = c + 5 * ldc;
    vst1q_f32(row5, vfmaq_n_f32(vld1q_f32(row5), acc50, alpha));
    vst1q_f32(row5 + 4, vfmaq_n_f32(vld1q_f32(row5 + 4), acc51, alpha));
    float* row6 = c + 6 * ldc;
    vst1q_f32(row6, vfmaq_n_f32(vld1q_f32(row6), acc60, alpha));
    vst1q_f32(row6 + 4, vfmaq_n_f32(vld1q_f32(row6 + 4), acc61, alpha));
    float* row7 = c + 7 * ldc;
    vst1q_f32(row7, vfmaq_n_f32(vld1q_f32(row7), acc70, alpha));
    vst1q_f32(row7 + 4, vfmaq_n_f32(vld1q_f32(row7 + 4), acc71, alpha));
    float* row8 = c + 8 * ldc;
    vst1q_f32(row8, vfmaq_n_f32(vld1q_f32(row8), acc80, alpha));
    vst1q_f32(row8 + 4, vfmaq_n_f32(vld1q_f32(row8 + 4), acc81, alpha));
    float* row9 = c + 9 * ldc;
    vst1q_f32(row9, vfmaq_n_f32(vld1q_f32(row9), acc90, alpha));
    vst1q_f32(row9 + 4, vfmaq_n_f32(vld1q_f32(row9 + 4), acc91, alpha));
    float* row10 = c + 10 * ldc;
    vst1q_f32(row10, vfmaq_n_f32(vld1q_f32(row10), acc100, alpha));
    vst1q_f32(row10 + 4, vfmaq_n_f32(vld1q_f32(row10 + 4), acc101, alpha));
    float* row11 = c + 11 * ldc;
    vst1q_f32(row11, vfmaq_n_f32(vld1q_f32(row11), acc110, alpha));
    vst1q_f32(row11 + 4, vfmaq_n_f32(vld1q_f32(row11 + 4), acc111, alpha));
    return;
  }

  // Край матрицы. Здесь аккумуляторы всё равно уезжают в память, но путь
  // редкий, и важна на нём правильность границы, а не скорость.
  float values[kNeonM][kNeonN];
  vst1q_f32(values[0], acc00);
  vst1q_f32(values[0] + 4, acc01);
  vst1q_f32(values[1], acc10);
  vst1q_f32(values[1] + 4, acc11);
  vst1q_f32(values[2], acc20);
  vst1q_f32(values[2] + 4, acc21);
  vst1q_f32(values[3], acc30);
  vst1q_f32(values[3] + 4, acc31);
  vst1q_f32(values[4], acc40);
  vst1q_f32(values[4] + 4, acc41);
  vst1q_f32(values[5], acc50);
  vst1q_f32(values[5] + 4, acc51);
  vst1q_f32(values[6], acc60);
  vst1q_f32(values[6] + 4, acc61);
  vst1q_f32(values[7], acc70);
  vst1q_f32(values[7] + 4, acc71);
  vst1q_f32(values[8], acc80);
  vst1q_f32(values[8] + 4, acc81);
  vst1q_f32(values[9], acc90);
  vst1q_f32(values[9] + 4, acc91);
  vst1q_f32(values[10], acc100);
  vst1q_f32(values[10] + 4, acc101);
  vst1q_f32(values[11], acc110);
  vst1q_f32(values[11] + 4, acc111);
  for (int64_t ii = 0; ii < rows; ++ii) {
    float* c_row = c + ii * ldc;
    for (int64_t jj = 0; jj < cols; ++jj) {
      c_row[jj] += alpha * values[ii][jj];
    }
  }
}

#endif  // LLM_HAS_NEON

const MicroKernelChoice* build_table(int* count) {
  static MicroKernelChoice table[3];
  static int size = 0;
  static bool ready = false;
  if (!ready) {
    // Признаки процессора нужны только векторным ветвям, а они есть не на
    // всякой архитектуре.
    const CpuFeatures& cpu = cpu_features();
    (void)cpu;

    // Слитное ли у скалярного ядра умножение с накоплением — зависит от
    // архитектуры, и это выяснилось замером, а не из документации.
    //
    // Ядро написано как `acc += a * b`, то есть раздельно. На базовом x86-64
    // команды слитного умножения нет, и компилятор оставляет два округления —
    // отсюда и договор, описанный в README: ядра с FMA совпадают побитово,
    // скалярное отличается на одно округление. На aarch64 слитная команда
    // входит в базовый набор, компилятор её подставляет, и скалярное ядро
    // оказывается слитным — его отпечаток совпал с отпечатком NEON и с
    // отпечатками AVX2 и AVX-512, снятыми на другой машине.
    //
    // Флаг говорит правду об арифметике, а не о намерении, и от него зависит
    // проверка: ядра, помеченные слитными, тест сверяет побитово. Если
    // когда-нибудь окажется, что на aarch64 склейки не произошло, тест об
    // этом скажет — и это верное поведение, а не ложная тревога.
    table[size].kernel = MicroKernel{kScalarM,
                                     kScalarN,
                                     &scalar_kernel,
                                     &scalar_kernel_rows,
                                     &plain_transpose_pack,
                                     "скалярное 4x32",
                                     LLM_HAS_NEON != 0};
    table[size].available = true;
    ++size;

#if LLM_HAS_NEON
    // Доступно всегда: NEON обязателен в ARMv8-A. Порядок в таблице задаёт
    // выбор по умолчанию — берётся последнее доступное ядро.
    table[size].kernel = MicroKernel{kNeonM,
                                     kNeonN,
                                     &neon_kernel,
                                     &neon_kernel_rows,
                                     &neon_transpose_pack,
                                     "NEON 12x8",
                                     true};
    table[size].available = cpu.has_neon();
    ++size;
#endif

#if LLM_HAS_X86_SIMD
    table[size].kernel = MicroKernel{kAvx2M,
                                     kAvx2N,
                                     &avx2_kernel,
                                     &avx2_kernel_rows,
                                     &avx2_transpose_pack,
                                     "AVX2 6x16",
                                     true};
    table[size].available = cpu.has_avx2_fma();
    ++size;

    table[size].kernel = MicroKernel{kAvx512M,
                                     kAvx512N,
                                     &avx512_kernel,
                                     &avx512_kernel_rows,
                                     &avx2_transpose_pack,
                                     "AVX-512 8x32",
                                     true};
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
    // Сверху вниз: последнее доступное — самое широкое.
    //
    // «Шире значит быстрее» — предположение, и на этой машине оно для
    // умножения матриц не подтвердилось: AVX-512 и AVX2 дают одинаковые 78 и
    // 79 GFLOPS, снижение частоты на широких инструкциях съедает выигрыш от
    // ширины ровно целиком. Порядок всё равно оставлен таким, потому что на
    // экспоненте AVX-512 быстрее в полтора раза, а ядро выбирается одно на
    // всё. Проверяется это замером в bench_gemm и bench_exp — там есть
    // принудительный выбор ядра ровно затем, чтобы вопрос решался измерением.
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
