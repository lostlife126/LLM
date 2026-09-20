// Стоит ли делать веса восьмиразрядными — по скорости.
//
// Качество уже проверено: precision_scope показал, что E4M3 стоит две десятых
// процента перплексии, то есть для инференса бесплатен. Остался второй вопрос,
// и он про то, чего у восьми разрядов нет.
//
// У половинной разрядности на развёртывание есть ОДНА КОМАНДА: vcvtph2ps на
// x86, FCVTL на ARM. У восьмиразрядных форматов нет ничего ни там, ни там:
// Raspberry Pi 5 — это Cortex-A76, ARMv8.2-A, а FEAT_FP8 появился в ARMv9.2; у
// Xeon есть F16C, но не FP8. Значит распаковка программная, и вопрос в том,
// съедает ли она выигрыш от вдвое меньшего чтения.
//
// Замеряются два разных места, и ответы у них разные:
//
//   пакетная распаковка — то, что делает упаковка панелей блочного пути;
//   распаковка внутри ядра — то, что делает прямой путь, которым идёт
//   генерация по одному токену.
//
// Запуск: ./bench_fp8
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define LLM_FP8_X86 1
#else
#define LLM_FP8_X86 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "measure.h"

#include "core/cpu.h"
#include "core/util.h"
#include "core/fp8.h"
#include "core/half.h"

namespace {

// --- пакетная распаковка -----------------------------------------------------
//
// Обе стороны написаны одним приёмом — маски сдвигом, без ветвей, — потому что
// иначе сравнивались бы не форматы, а способность компилятора векторизовать
// два разных куска кода. Этот приём в проекте уже отобран замером: с ветвями
// выходило 3.8 такта на элемент, с тернарниками 5.1, с масками 1.7.
//
// Обработки бесконечности и NaN нет ни там, ни там, и для весов это допустимо:
// значения приходят из своего же округления, а оно насыщает и NaN не даёт.
// Держать её у одного и не держать у другого было бы подтасовкой — первая
// версия этого замера так и делала и показывала fp8 дешевле, чем он есть.
void expand_half(const llm::Half* source, float* destination, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    const uint32_t bits16 = source[i].bits;
    const uint32_t sign = (bits16 & 0x8000u) << 16;
    const uint32_t rest = bits16 & 0x7FFFu;
    uint32_t bits = rest << 13;
    const uint32_t field = bits & (0x7C00u << 13);
    const uint32_t is_subnormal = 0u - ((field - 1u) >> 31);
    bits += 112u << 23;
    const float normal = llm::bits_as_float(bits);
    const float subnormal = llm::bits_as_float(bits + (1u << 23)) -
                            llm::bits_as_float(113u << 23);
    const uint32_t chosen = (is_subnormal & llm::float_bits(subnormal)) |
                            (~is_subnormal & llm::float_bits(normal));
    destination[i] = llm::bits_as_float(chosen | sign);
  }
}

// E4M3 тем же приёмом. Порядок 4 разряда со смещением 7, мантисса 3: значит
// сдвиг мантиссы 20 вместо 13, а смещение порядка 127 - 7 = 120 вместо 112.
void expand_e4m3(const uint8_t* source, float* destination, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    const uint32_t code = source[i];
    const uint32_t sign = (code & 0x80u) << 24;
    const uint32_t rest = code & 0x7Fu;
    uint32_t bits = rest << 20;
    const uint32_t field = bits & (0x78u << 20);
    const uint32_t is_subnormal = 0u - ((field - 1u) >> 31);
    bits += 120u << 23;
    const float normal = llm::bits_as_float(bits);
    const float subnormal = llm::bits_as_float(bits + (1u << 23)) -
                            llm::bits_as_float(121u << 23);
    const uint32_t chosen = (is_subnormal & llm::float_bits(subnormal)) |
                            (~is_subnormal & llm::float_bits(normal));
    destination[i] = llm::bits_as_float(chosen | sign);
  }
}

void measure_bulk() {
  const int64_t count = 1 << 22;
  std::vector<llm::Half> half(count);
  std::vector<uint8_t> narrow(count);
  std::vector<float> out(count);
  std::vector<float> source(count);
  for (int64_t i = 0; i < count; ++i) {
    half[i].bits = static_cast<uint16_t>(i * 7919);
    narrow[i] = static_cast<uint8_t>(i * 31);
  }

  const double half_seconds = bench::best_seconds(
      [&]() { expand_half(half.data(), out.data(), count); }, 0.4);
  const double narrow_seconds = bench::best_seconds(
      [&]() { expand_e4m3(narrow.data(), out.data(), count); }, 0.4);
  const double copy_seconds = bench::best_seconds(
      [&]() {
        std::copy(source.begin(), source.end(), out.begin());
      },
      0.4);

  const double per = 1.0e9 / static_cast<double>(count);
  const double elements = static_cast<double>(count);

  // Байты на элемент печатаются рядом со временем, и это не украшение. Первая
  // версия этого замера подписывала копирование «нижней границей», а оно ею не
  // является: копия переносит восемь байт на элемент (читает четыре, пишет
  // четыре), распаковка половинной разрядности — шесть, восьмиразрядной — пять.
  // На Raspberry Pi из-за этой подписи копия выглядела МЕДЛЕННЕЕ распаковки,
  // хотя просто делала больше работы.
  //
  // Со скоростью в гигабайтах в секунду видно главное: упёрлось ли дело в
  // память или в команды. Если у всех трёх строк скорость примерно одна —
  // упёрлось в память, и тогда важны только байты. Если у распаковок она
  // заметно ниже, чем у копии, — упёрлось в команды, и узкий формат сам по
  // себе не спасёт.
  const auto line = [&](const char* name, double seconds, double bytes) {
    std::printf("  %s %6.3f нс  %4.0f байт  %6.2f ГБ/с\n",
              llm::pad_utf8(name, 14).c_str(),
                seconds * per, bytes, bytes * elements / seconds / 1.0e9);
  };
  std::printf("пакетная распаковка, на элемент\n");
  line("fp16 -> float", half_seconds, 6.0);
  line("e4m3 -> float", narrow_seconds, 5.0);
  line("копия float", copy_seconds, 8.0);
  std::printf("  отношение времени e4m3 к fp16: %.2fx\n\n",
              narrow_seconds / half_seconds);
}

#if LLM_FP8_X86

// --- распаковка внутри ядра --------------------------------------------------
//
// Восемь накопителей, а не два: при m = 1 цепочка накопления по k — это
// зависимость длиной в k, и с двумя накопителями ядро упёрлось бы в задержку
// умножения с накоплением, а не в то, что мерится.
constexpr int64_t kTileN = 64;

__attribute__((target("avx2,fma,f16c"))) void row_half(
    int64_t k, const float* a, const llm::Half* b, int64_t ldb, float* c) {
  __m256 acc[8];
  for (int i = 0; i < 8; ++i) acc[i] = _mm256_setzero_ps();
  for (int64_t p = 0; p < k; ++p) {
    const __m256 av = _mm256_set1_ps(a[p]);
    const __m128i* row = reinterpret_cast<const __m128i*>(b + p * ldb);
    for (int i = 0; i < 8; ++i) {
      acc[i] = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(row + i)),
                               acc[i]);
    }
  }
  for (int i = 0; i < 8; ++i) _mm256_storeu_ps(c + i * 8, acc[i]);
}

// Восемь значений E4M3 -> восемь float. Дюжина операций там, где у половинной
// разрядности одна команда, — и вот в этом весь вопрос.
__attribute__((target("avx2,fma"))) inline __m256 expand_eight(
    const uint8_t* source) {
  const __m256i raw = _mm256_cvtepu8_epi32(
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source)));
  const __m256i sign =
      _mm256_slli_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(0x80)), 24);
  const __m256i rest = _mm256_and_si256(raw, _mm256_set1_epi32(0x7F));
  __m256i bits = _mm256_slli_epi32(rest, 20);
  const __m256i field = _mm256_and_si256(bits, _mm256_set1_epi32(0x78 << 20));
  const __m256i is_subnormal =
      _mm256_cmpeq_epi32(field, _mm256_setzero_si256());
  bits = _mm256_add_epi32(bits, _mm256_set1_epi32(120 << 23));
  const __m256 normal = _mm256_castsi256_ps(bits);
  const __m256 subnormal = _mm256_sub_ps(
      _mm256_castsi256_ps(_mm256_add_epi32(bits, _mm256_set1_epi32(1 << 23))),
      _mm256_castsi256_ps(_mm256_set1_epi32(121 << 23)));
  const __m256 chosen = _mm256_blendv_ps(normal, subnormal,
                                         _mm256_castsi256_ps(is_subnormal));
  return _mm256_or_ps(chosen, _mm256_castsi256_ps(sign));
}

__attribute__((target("avx2,fma"))) void row_e4m3(int64_t k, const float* a,
                                                  const uint8_t* b,
                                                  int64_t ldb, float* c) {
  __m256 acc[8];
  for (int i = 0; i < 8; ++i) acc[i] = _mm256_setzero_ps();
  for (int64_t p = 0; p < k; ++p) {
    const __m256 av = _mm256_set1_ps(a[p]);
    const uint8_t* row = b + p * ldb;
    for (int i = 0; i < 8; ++i) {
      acc[i] = _mm256_fmadd_ps(av, expand_eight(row + i * 8), acc[i]);
    }
  }
  for (int i = 0; i < 8; ++i) _mm256_storeu_ps(c + i * 8, acc[i]);
}

void measure_kernel() {
  static const int64_t shapes[][2] = {
      {256, 256}, {256, 4096}, {512, 4096}, {384, 8192},
  };
  std::printf("распаковка внутри ядра, m = 1 (форма генерации по токену)\n");
  std::printf("%s %10s %10s %s\n", llm::pad_utf8("форма", 20).c_str(), "fp16",
              "e4m3", llm::pad_utf8_right("отношение", 10).c_str());
  for (std::size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); ++s) {
    const int64_t k = shapes[s][0];
    const int64_t n = shapes[s][1];
    // Оба ядра идут по n шагом kTileN и хвоста не разбирают: они написаны под
    // формы модели, а те кратны. Некратная не упала бы, а прочитала и
    // записала бы за границей.
    if (n % kTileN != 0) {
      std::printf("n=%lld не кратно %lld — эта форма ядрами не считается\n",
                  static_cast<long long>(n), static_cast<long long>(kTileN));
      continue;
    }
    std::vector<float> a(static_cast<std::size_t>(k), 0.01f);
    std::vector<float> c(static_cast<std::size_t>(n), 0.0f);
    std::vector<llm::Half> half(static_cast<std::size_t>(k * n));
    std::vector<uint8_t> narrow(static_cast<std::size_t>(k * n));
    for (int64_t i = 0; i < k * n; ++i) {
      half[i].bits = static_cast<uint16_t>(0x3800 + (i % 64));
      narrow[i] = static_cast<uint8_t>(0x38 + (i % 8));
    }

    const double half_seconds = bench::best_seconds(
        [&]() {
          for (int64_t j = 0; j < n; j += kTileN) {
            row_half(k, a.data(), half.data() + j, n, c.data() + j);
          }
        },
        0.3);
    const double narrow_seconds = bench::best_seconds(
        [&]() {
          for (int64_t j = 0; j < n; j += kTileN) {
            row_e4m3(k, a.data(), narrow.data() + j, n, c.data() + j);
          }
        },
        0.3);

    char label[64];
    std::snprintf(label, sizeof(label), "k=%lld n=%lld",
                  static_cast<long long>(k), static_cast<long long>(n));
    std::printf("%-20s %7.3f мс %7.3f мс %9.2fx\n", label, half_seconds * 1e3,
                narrow_seconds * 1e3, half_seconds / narrow_seconds);
  }
}

#endif  // LLM_FP8_X86

}  // namespace

int main() {
  std::printf("процессор: %s\n\n", llm::cpu_features().to_string().c_str());
  measure_bulk();
#if LLM_FP8_X86
  if (llm::cpu_features().has_avx2_f16c()) {
    measure_kernel();
  } else {
    std::printf("нет AVX2+FMA+F16C; замер ядра пропущен\n");
  }
#else
  std::printf(
      "замер ядра написан интринсиками x86; на этой машине пропущен.\n"
      "Пакетная распаковка выше переносима и меряет то же отношение.\n");
#endif
  return 0;
}
