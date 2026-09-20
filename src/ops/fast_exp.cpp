#include "ops/fast_exp.h"

#include <cmath>
#include <cstring>
#include <limits>

#include "core/check.h"
#include "core/cpu.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define LLM_HAS_X86_SIMD 1
#else
#define LLM_HAS_X86_SIMD 0
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#define LLM_HAS_NEON 1
#else
#define LLM_HAS_NEON 0
#endif

namespace llm {
namespace ops {
namespace {

// Приведение к степени двойки и обратно. Всё построение экспоненты держится на
// тождестве exp(x) = 2^n * exp(r): показатель n берут целым, а остаток r
// загоняют в отрезок длиной ln2 около нуля, где многочлен пятой степени уже
// даёт точность float.
constexpr float kLog2e = 1.44269504088896341f;

// ln2 разложен на две части. Если вычитать n * ln2 одним умножением, то при
// больших n произведение округляется, и погрешность попадает прямо в r —
// туда, где она стоит дороже всего. Первая часть подобрана так, что её
// двоичная запись коротка и произведение на целое n точно; вторая добирает
// остаток.
constexpr float kLn2Hi = 0.693359375f;
constexpr float kLn2Lo = -2.12194440e-4f;

// Минимаксный многочлен для exp(r) - 1 - r на отрезке [-ln2/2, ln2/2],
// коэффициенты из Cephes. Пиковая относительная погрешность около одного
// последнего разряда float.
constexpr float kP0 = 1.9875691500e-4f;
constexpr float kP1 = 1.3981999507e-3f;
constexpr float kP2 = 8.3334519073e-3f;
constexpr float kP3 = 4.1665795894e-2f;
constexpr float kP4 = 1.6666665459e-1f;
constexpr float kP5 = 5.0000001201e-1f;

// Границы, за которыми считать нечего: выше — переполнение float, ниже —
// значение меньше самого маленького субнормального числа.
constexpr float kExpMax = 88.72283905f;
constexpr float kExpMin = -103.97208f;

float bits_to_float(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// 2^k для k в диапазоне нормальных чисел. Собирается прямо из битов показателя.
float power_of_two(int k) {
  return bits_to_float(static_cast<uint32_t>(k + 127) << 23);
}

// Округление к ближайшему целому без вызова функции.
//
// Прибавить и тут же вычесть 1.5 * 2^23 — приём старый и на вид бессмысленный,
// но он и есть содержание этой функции: у float мантисса 24 разряда, поэтому
// после прибавления такой величины дробная часть просто не имеет куда
// поместиться и отбрасывается текущим режимом округления, то есть к
// ближайшему чётному. Ровно то же делает векторная инструкция округления.
//
// Появилось это не из любви к приёмам. Сначала здесь стоял std::nearbyint, и
// замер показал, что вся скалярная реализация выходит в два с половиной раза
// **медленнее** std::exp из libm: nearbyintf на базовом x86-64 — настоящий
// вызов функции, потому что инструкция округления появилась только в SSE4.1.
// То есть запасной путь, существующий для старых процессоров, на них же и
// оказывался хуже того, что заменял.
//
// Компилятору сокращать (x + magic) - magic до x нельзя: это меняет значение,
// и без -ffast-math он этого не делает.
float round_to_nearest(float x) {
  const float magic = 12582912.0f;  // 1.5 * 2^23
  return (x + magic) - magic;
}

// --- эталонная скалярная реализация -----------------------------------------
//
// Читаемая запись того, что делают векторные: они повторяют её шаг в шаг.
// Запасным путём она не служит — замер показал, что она вдвое медленнее
// std::exp, см. заголовок. Остаётся ради тестов и ради того, чтобы алгоритм
// был записан обычным кодом, а не только интринсиками.

float scalar_exp_one(float x) {
  // NaN обязан пройти насквозь. Если обучение разошлось, это надо увидеть в
  // потерях, а не получить вместо NaN правдоподобный нуль.
  if (!(x == x)) {
    return x;
  }
  if (x > kExpMax) {
    return std::numeric_limits<float>::infinity();
  }
  if (x < kExpMin) {
    return 0.0f;
  }

  const float n = round_to_nearest(x * kLog2e);
  float r = x - n * kLn2Hi;
  r = r - n * kLn2Lo;

  const float z = r * r;
  float p = kP0;
  p = p * r + kP1;
  p = p * r + kP2;
  p = p * r + kP3;
  p = p * r + kP4;
  p = p * r + kP5;
  p = p * z + r;
  p = p + 1.0f;

  // Степень двойки применяется в два приёма, когда показатель выходит за
  // границу нормальных чисел: он доходит до -150, а 2^-150 как одно число не
  // существует — там уже субнормальная область. Половинами оба множителя
  // остаются нормальными, и exp(-100) получается правильным, а не нулём.
  //
  // В обычном случае хватает одного умножения. Результат от этого не меняется:
  // умножение на точную степень двойки в нормальном диапазоне точно, в каком
  // бы порядке его ни делать.
  const int k = static_cast<int>(n);
  if (k >= -126 && k <= 127) {
    return p * power_of_two(k);
  }
  const int low = k / 2;
  return p * power_of_two(low) * power_of_two(k - low);
}

void scalar_exp_shifted(const float* input, float shift, float* output,
                        int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    output[i] = scalar_exp_one(input[i] - shift);
  }
}

void scalar_sigmoid(const float* input, float* output, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    output[i] = 1.0f / (1.0f + scalar_exp_one(-input[i]));
  }
}

// --- запасной путь: libm ----------------------------------------------------
//
// Выбирается на процессоре без AVX2. Это ровно то, что было в операциях до
// появления этого файла, поэтому на таких машинах ничего не ухудшается.

void libm_exp_shifted(const float* input, float shift, float* output,
                      int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    output[i] = std::exp(input[i] - shift);
  }
}

void libm_sigmoid(const float* input, float* output, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    output[i] = 1.0f / (1.0f + std::exp(-input[i]));
  }
}

#if LLM_HAS_NEON

// --- NEON -------------------------------------------------------------------
//
// Повторяет ветвь AVX2 шаг в шаг: те же коэффициенты, тот же порядок действий,
// то же слитное умножение с накоплением. Это не аккуратность ради аккуратности
// — от этого зависит побитовое совпадение результата между машинами, а на нём
// стоит вся воспроизводимость обучения.
//
// Соответствие команд, если сравнивать с ветвью AVX2:
//   _mm256_round_ps(..., NEAREST) -> vrndnq_f32   (к ближайшему чётному)
//   _mm256_fnmadd_ps(n, c, x)     -> vfmsq_f32(x, n, c)   = x - n * c
//   _mm256_fmadd_ps(p, r, c)      -> vfmaq_f32(c, p, r)   = c + p * r
//   _mm256_cvttps_epi32           -> vcvtq_s32_f32        (усечение к нулю)
//   _mm256_blendv_ps(a, b, m)     -> vbslq_f32(m, b, a)
//
// Проверки на NaN нет отдельной ветви: значение сравнивается само с собой.

inline float32x4_t exp4(float32x4_t x) {
  const uint32x4_t too_big = vcgtq_f32(x, vdupq_n_f32(kExpMax));
  const uint32x4_t too_small = vcltq_f32(x, vdupq_n_f32(kExpMin));
  const uint32x4_t is_nan = vmvnq_u32(vceqq_f32(x, x));

  // Значения за границами всё равно считаются, а подменяются в конце: ветвление
  // по дорожкам стоило бы дороже лишнего многочлена. Исходное x сохраняется —
  // NaN после ограничения превратился бы в границу, а вернуть надо именно NaN.
  const float32x4_t clamped =
      vminq_f32(vmaxq_f32(x, vdupq_n_f32(kExpMin)), vdupq_n_f32(kExpMax));

  const float32x4_t n = vrndnq_f32(vmulq_f32(clamped, vdupq_n_f32(kLog2e)));
  float32x4_t r = vfmsq_f32(clamped, n, vdupq_n_f32(kLn2Hi));
  r = vfmsq_f32(r, n, vdupq_n_f32(kLn2Lo));

  const float32x4_t z = vmulq_f32(r, r);
  float32x4_t p = vdupq_n_f32(kP0);
  p = vfmaq_f32(vdupq_n_f32(kP1), p, r);
  p = vfmaq_f32(vdupq_n_f32(kP2), p, r);
  p = vfmaq_f32(vdupq_n_f32(kP3), p, r);
  p = vfmaq_f32(vdupq_n_f32(kP4), p, r);
  p = vfmaq_f32(vdupq_n_f32(kP5), p, r);
  p = vfmaq_f32(r, p, z);
  p = vaddq_f32(p, vdupq_n_f32(1.0f));

  // Степень двойки в два приёма — по той же причине, что и в скалярной
  // реализации: показатель доходит до -150, а такого числа как одно значение
  // нет. Половинки считаются усечением к нулю, а не сдвигом: сдвиг округлял бы
  // отрицательные к минус бесконечности, и результат разошёлся бы со скалярным.
  const int32x4_t k = vcvtq_s32_f32(n);
  const int32x4_t half = vcvtq_s32_f32(vmulq_f32(n, vdupq_n_f32(0.5f)));
  const int32x4_t rest = vsubq_s32(k, half);
  const int32x4_t bias = vdupq_n_s32(127);
  const float32x4_t scale_low =
      vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(half, bias), 23));
  const float32x4_t scale_rest =
      vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(rest, bias), 23));
  float32x4_t result = vmulq_f32(vmulq_f32(p, scale_low), scale_rest);

  result = vbslq_f32(
      too_big, vdupq_n_f32(std::numeric_limits<float>::infinity()), result);
  result = vbslq_f32(too_small, vdupq_n_f32(0.0f), result);
  return vbslq_f32(is_nan, x, result);
}

void neon_exp_shifted(const float* input, float shift, float* output,
                      int64_t count) {
  const float32x4_t offset = vdupq_n_f32(shift);
  int64_t i = 0;
  for (; i + 4 <= count; i += 4) {
    vst1q_f32(output + i, exp4(vsubq_f32(vld1q_f32(input + i), offset)));
  }
  // Хвост доводится тем же ядром через временный буфер: скалярная реализация
  // округляет иначе, и последние элементы массива отличались бы от остальных.
  if (i < count) {
    float tail[4];
    for (int64_t j = 0; j < 4; ++j) {
      tail[j] = i + j < count ? input[i + j] : 0.0f;
    }
    vst1q_f32(tail, exp4(vsubq_f32(vld1q_f32(tail), offset)));
    for (int64_t j = 0; i + j < count; ++j) {
      output[i + j] = tail[j];
    }
  }
}

void neon_sigmoid(const float* input, float* output, int64_t count) {
  const float32x4_t one = vdupq_n_f32(1.0f);
  int64_t i = 0;
  for (; i + 4 <= count; i += 4) {
    const float32x4_t x = vld1q_f32(input + i);
    const float32x4_t e = exp4(vnegq_f32(x));
    vst1q_f32(output + i, vdivq_f32(one, vaddq_f32(one, e)));
  }
  if (i < count) {
    float tail[4];
    for (int64_t j = 0; j < 4; ++j) {
      tail[j] = i + j < count ? input[i + j] : 0.0f;
    }
    const float32x4_t x = vld1q_f32(tail);
    const float32x4_t e = exp4(vnegq_f32(x));
    vst1q_f32(tail, vdivq_f32(one, vaddq_f32(one, e)));
    for (int64_t j = 0; i + j < count; ++j) {
      output[i + j] = tail[j];
    }
  }
}

#endif  // LLM_HAS_NEON

#if LLM_HAS_X86_SIMD

// --- AVX2 -------------------------------------------------------------------

__attribute__((target("avx2,fma"))) __m256 exp8(__m256 x) {
  const __m256 too_big = _mm256_cmp_ps(x, _mm256_set1_ps(kExpMax), _CMP_GT_OQ);
  const __m256 too_small =
      _mm256_cmp_ps(x, _mm256_set1_ps(kExpMin), _CMP_LT_OQ);
  const __m256 is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

  // Значения за границами всё равно считаются, а подменяются в конце: ветвление
  // по дорожкам стоило бы дороже, чем лишний многочлен, а границы выбраны так,
  // что промежуточные величины не переполняются. Исходное x при этом нужно
  // сохранить: NaN после ограничения превращается в границу, а вернуть надо
  // именно NaN.
  const __m256 clamped = _mm256_min_ps(
      _mm256_max_ps(x, _mm256_set1_ps(kExpMin)), _mm256_set1_ps(kExpMax));

  const __m256 n =
      _mm256_round_ps(_mm256_mul_ps(clamped, _mm256_set1_ps(kLog2e)),
                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(kLn2Hi), clamped);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(kLn2Lo), r);

  const __m256 z = _mm256_mul_ps(r, r);
  __m256 p = _mm256_set1_ps(kP0);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP1));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP2));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP3));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP4));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP5));
  p = _mm256_fmadd_ps(p, z, r);
  p = _mm256_add_ps(p, _mm256_set1_ps(1.0f));

  // Деление пополам целочисленным сдвигом даёт округление к минус
  // бесконечности, а скалярный вариант делит в int, то есть округляет к нулю.
  // Для отрицательных это разные числа, поэтому половинки считаются так же,
  // как там: k - (k усечённое пополам).
  const __m256i k = _mm256_cvttps_epi32(n);
  const __m256i half =
      _mm256_cvttps_epi32(_mm256_mul_ps(n, _mm256_set1_ps(0.5f)));
  const __m256i rest = _mm256_sub_epi32(k, half);
  const __m256i bias = _mm256_set1_epi32(127);
  const __m256 scale_low =
      _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(half, bias), 23));
  const __m256 scale_rest =
      _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(rest, bias), 23));
  __m256 result = _mm256_mul_ps(_mm256_mul_ps(p, scale_low), scale_rest);

  result = _mm256_blendv_ps(
      result, _mm256_set1_ps(std::numeric_limits<float>::infinity()), too_big);
  result = _mm256_blendv_ps(result, _mm256_setzero_ps(), too_small);
  return _mm256_blendv_ps(result, x, is_nan);
}

__attribute__((target("avx2,fma"))) void avx2_exp_shifted(const float* input,
                                                          float shift,
                                                          float* output,
                                                          int64_t count) {
  const __m256 offset = _mm256_set1_ps(shift);
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 x = _mm256_sub_ps(_mm256_loadu_ps(input + i), offset);
    _mm256_storeu_ps(output + i, exp8(x));
  }
  // Хвост. Считать его скалярной реализацией нельзя: она округляет иначе (см.
  // заголовок), и тогда последние несколько элементов массива отличались бы от
  // остальных. Поэтому хвост доводится тем же ядром через временный буфер.
  if (i < count) {
    float tail[8];
    for (int64_t j = 0; j < 8; ++j) {
      tail[j] = i + j < count ? input[i + j] : 0.0f;
    }
    const __m256 x = _mm256_sub_ps(_mm256_loadu_ps(tail), offset);
    _mm256_storeu_ps(tail, exp8(x));
    for (int64_t j = 0; i + j < count; ++j) {
      output[i + j] = tail[j];
    }
  }
}

__attribute__((target("avx2,fma"))) void avx2_sigmoid(const float* input,
                                                      float* output,
                                                      int64_t count) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 sign =
      _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000u)));
  int64_t i = 0;
  for (; i + 8 <= count; i += 8) {
    const __m256 x = _mm256_loadu_ps(input + i);
    const __m256 e = exp8(_mm256_xor_ps(x, sign));
    _mm256_storeu_ps(output + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
  }
  if (i < count) {
    float tail[8];
    for (int64_t j = 0; j < 8; ++j) {
      tail[j] = i + j < count ? input[i + j] : 0.0f;
    }
    const __m256 x = _mm256_loadu_ps(tail);
    const __m256 e = exp8(_mm256_xor_ps(x, sign));
    _mm256_storeu_ps(tail, _mm256_div_ps(one, _mm256_add_ps(one, e)));
    for (int64_t j = 0; i + j < count; ++j) {
      output[i + j] = tail[j];
    }
  }
}

// --- AVX-512 ----------------------------------------------------------------

__attribute__((target("avx512f,avx512bw,avx512vl"))) __m512 exp16(__m512 x) {
  const __mmask16 too_big =
      _mm512_cmp_ps_mask(x, _mm512_set1_ps(kExpMax), _CMP_GT_OQ);
  const __mmask16 too_small =
      _mm512_cmp_ps_mask(x, _mm512_set1_ps(kExpMin), _CMP_LT_OQ);
  const __mmask16 is_nan = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);

  const __m512 clamped = _mm512_min_ps(
      _mm512_max_ps(x, _mm512_set1_ps(kExpMin)), _mm512_set1_ps(kExpMax));

  const __m512 n =
      _mm512_roundscale_ps(_mm512_mul_ps(clamped, _mm512_set1_ps(kLog2e)),
                           _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(kLn2Hi), clamped);
  r = _mm512_fnmadd_ps(n, _mm512_set1_ps(kLn2Lo), r);

  const __m512 z = _mm512_mul_ps(r, r);
  __m512 p = _mm512_set1_ps(kP0);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP1));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP2));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP3));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP4));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP5));
  p = _mm512_fmadd_ps(p, z, r);
  p = _mm512_add_ps(p, _mm512_set1_ps(1.0f));

  const __m512i k = _mm512_cvttps_epi32(n);
  const __m512i half =
      _mm512_cvttps_epi32(_mm512_mul_ps(n, _mm512_set1_ps(0.5f)));
  const __m512i rest = _mm512_sub_epi32(k, half);
  const __m512i bias = _mm512_set1_epi32(127);
  const __m512 scale_low =
      _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(half, bias), 23));
  const __m512 scale_rest =
      _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(rest, bias), 23));
  __m512 result = _mm512_mul_ps(_mm512_mul_ps(p, scale_low), scale_rest);

  result = _mm512_mask_blend_ps(
      too_big, result, _mm512_set1_ps(std::numeric_limits<float>::infinity()));
  result = _mm512_mask_blend_ps(too_small, result, _mm512_setzero_ps());
  return _mm512_mask_blend_ps(is_nan, result, x);
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) void avx512_exp_shifted(
    const float* input, float shift, float* output, int64_t count) {
  const __m512 offset = _mm512_set1_ps(shift);
  int64_t i = 0;
  for (; i + 16 <= count; i += 16) {
    const __m512 x = _mm512_sub_ps(_mm512_loadu_ps(input + i), offset);
    _mm512_storeu_ps(output + i, exp16(x));
  }
  if (i < count) {
    float tail[16];
    for (int64_t j = 0; j < 16; ++j) {
      tail[j] = i + j < count ? input[i + j] : 0.0f;
    }
    const __m512 x = _mm512_sub_ps(_mm512_loadu_ps(tail), offset);
    _mm512_storeu_ps(tail, exp16(x));
    for (int64_t j = 0; i + j < count; ++j) {
      output[i + j] = tail[j];
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl"))) void avx512_sigmoid(
    const float* input, float* output, int64_t count) {
  const __m512 one = _mm512_set1_ps(1.0f);
  // Отрицание через исключающее «или» с битом знака, а не вычитанием из нуля:
  // так -0 остаётся -0, и результат совпадает со скалярным «-x». Побитовая
  // операция берётся целочисленная: _mm512_xor_ps требует AVX512DQ, а он есть
  // не на всех процессорах с AVX-512.
  const __m512i sign = _mm512_set1_epi32(static_cast<int>(0x80000000u));
  int64_t i = 0;
  for (; i + 16 <= count; i += 16) {
    const __m512 x = _mm512_loadu_ps(input + i);
    const __m512 negated =
        _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x), sign));
    const __m512 e = exp16(negated);
    _mm512_storeu_ps(output + i, _mm512_div_ps(one, _mm512_add_ps(one, e)));
  }
  if (i < count) {
    float tail[16];
    for (int64_t j = 0; j < 16; ++j) {
      tail[j] = i + j < count ? input[i + j] : 0.0f;
    }
    const __m512 x = _mm512_loadu_ps(tail);
    const __m512 negated =
        _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x), sign));
    const __m512 e = exp16(negated);
    _mm512_storeu_ps(tail, _mm512_div_ps(one, _mm512_add_ps(one, e)));
    for (int64_t j = 0; i + j < count; ++j) {
      output[i + j] = tail[j];
    }
  }
}

#endif  // LLM_HAS_X86_SIMD

const ExpKernelChoice* build_table(int* count) {
  // Вместимость считается теми же условиями, что и заполнение. Было
  // table[4] — ровно по числу ядер на x86, без запаса, — и добавленное ядро
  // записалось бы за границу статического массива: не отказ, а порча
  // соседней памяти. То же место уже исправлено в таблице микроядер.
  constexpr int kSlots = 2  // эталонное и libm есть всегда
#if LLM_HAS_NEON
                         + 1
#endif
#if LLM_HAS_X86_SIMD
                         + 2
#endif
      ;
  static ExpKernelChoice table[kSlots];
  static int size = 0;
  static bool ready = false;
  if (!ready) {
    // Признаки процессора нужны только векторным ветвям, а они есть не на
    // всякой архитектуре. Без этого приведения сборка под ARM падает на
    // предупреждении о неиспользуемой переменной, возведённом в ошибку.
    const CpuFeatures& cpu = cpu_features();
    (void)cpu;

    // Порядок важен: выбирается последняя доступная, поэтому эталонный
    // многочлен стоит перед libm и автоматически не выбирается никогда.
    // Многочлен записан как p * r + c, то есть раздельно, и раздельным
    // остаётся: сборка идёт с -ffp-contract=off, и компилятор не склеивает
    // это в слитную команду ни на одной машине. Раньше признак зависел от
    // архитектуры — на aarch64 склейка происходила сама, — и ровно поэтому
    // обучение на разных машинах расходилось; см. комментарий у скалярного
    // микроядра.
    LLM_CHECK(size < kSlots);
    table[size].kernel = ExpKernel{&scalar_exp_shifted, &scalar_sigmoid,
                                   "эталонная", false};
    table[size].available = true;
    ++size;

    LLM_CHECK(size < kSlots);
    table[size].kernel =
        ExpKernel{&libm_exp_shifted, &libm_sigmoid, "libm", false};
    table[size].available = true;
    ++size;

#if LLM_HAS_NEON
    // Доступно всегда: NEON обязателен в ARMv8-A.
    LLM_CHECK(size < kSlots);
    table[size].kernel =
        ExpKernel{&neon_exp_shifted, &neon_sigmoid, "NEON x4", true};
    table[size].available = cpu.has_neon();
    ++size;
#endif

#if LLM_HAS_X86_SIMD
    // Ядро экспоненты умножения с накоплением не использует, поэтому от AVX2
    // требуется только сам AVX2 — но проверяется та же пара, что у микроядер:
    // без FMA процессор всё равно не из тех, на которых это имеет смысл.
    LLM_CHECK(size < kSlots);
    table[size].kernel =
        ExpKernel{&avx2_exp_shifted, &avx2_sigmoid, "AVX2 x8", true};
    table[size].available = cpu.has_avx2_fma();
    ++size;

    LLM_CHECK(size < kSlots);
    table[size].kernel =
        ExpKernel{&avx512_exp_shifted, &avx512_sigmoid, "AVX-512 x16", true};
    table[size].available = cpu.has_avx512();
    ++size;
#endif
    ready = true;
  }
  *count = size;
  return table;
}

const ExpKernel* g_forced = nullptr;

}  // namespace

const ExpKernelChoice* all_exp_kernels(int* count) {
  LLM_CHECK(count != nullptr);
  return build_table(count);
}

void force_exp_kernel(const ExpKernel* kernel) { g_forced = kernel; }

const ExpKernel& best_exp_kernel() {
  if (g_forced != nullptr) {
    return *g_forced;
  }
  static const ExpKernel chosen = []() {
    int count = 0;
    const ExpKernelChoice* table = build_table(&count);
    ExpKernel best = table[0].kernel;
    for (int i = 0; i < count; ++i) {
      if (table[i].available) {
        best = table[i].kernel;
      }
    }
    return best;
  }();
  return chosen;
}

void exp_shifted(const float* input, float shift, float* output,
                 int64_t count) {
  if (count <= 0) {
    return;
  }
  best_exp_kernel().exp_shifted(input, shift, output, count);
}

void exp_array(const float* input, float* output, int64_t count) {
  exp_shifted(input, 0.0f, output, count);
}

void sigmoid_array(const float* input, float* output, int64_t count) {
  if (count <= 0) {
    return;
  }
  best_exp_kernel().sigmoid(input, output, count);
}

float exp_scalar(float x) { return scalar_exp_one(x); }

}  // namespace ops
}  // namespace llm
