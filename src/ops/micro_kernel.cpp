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

    table[size].kernel = MicroKernel{kScalarM,         kScalarN,
                                     &scalar_kernel,   &plain_transpose_pack,
                                     "скалярное 4x32", false};
    table[size].available = true;
    ++size;

#if LLM_HAS_X86_SIMD
    table[size].kernel = MicroKernel{
        kAvx2M, kAvx2N, &avx2_kernel, &avx2_transpose_pack, "AVX2 6x16", true};
    table[size].available = cpu.has_avx2_fma();
    ++size;

    table[size].kernel = MicroKernel{kAvx512M,       kAvx512N,
                                     &avx512_kernel, &avx2_transpose_pack,
                                     "AVX-512 8x32", true};
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
