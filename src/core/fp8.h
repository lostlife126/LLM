// Округление до восьмиразрядных форматов с плавающей точкой.
//
// Форматов два, и выбор между ними — это выбор между точностью и диапазоном
// при одном и том же числе разрядов:
//
//   E4M3   знак, 4 разряда порядка, 3 мантиссы. Смещение 7. Бесконечности
//          нет — код со всеми единицами порядка и мантиссы занят NaN, поэтому
//          наибольшее конечное значение 448, а не 512. Наименьшее нормальное
//          2^-6, субнормальные до 2^-9. Относительный шаг решётки 2^-3,
//          погрешность не больше 2^-4, то есть 6.25%.
//
//   E5M2   знак, 5 разрядов порядка, 2 мантиссы. Смещение 15. Бесконечность и
//          NaN как в IEEE. Наибольшее конечное 57344, наименьшее нормальное
//          2^-14, субнормальные до 2^-16. Погрешность не больше 2^-3, то есть
//          12.5%.
//
// Почему одна реализация на два формата, а не две. Преобразование здесь —
// единственное тонкое место: субнормальная область, округление к чётному,
// перенос при переполнении мантиссы. Написать его дважды значит завести две
// возможности ошибиться там, где трудно заметить. Поэтому алгоритм один, а
// формат приходит параметром; настоящему ядру, когда до него дойдёт дело,
// понадобится специализация, но это уже другая задача.
//
// Переполнение НАСЫЩАЕТ, а не даёт бесконечность или NaN. Выбор осознанный:
// восьмиразрядные форматы применяются с масштабом на тензор, и значение
// вылезает за верхний край только у самых больших элементов. Насыщение стоит
// им погрешности, а NaN уничтожил бы всё вычисление целиком. Спецификация OCP
// допускает оба поведения.

#ifndef LLM_CORE_FP8_H_
#define LLM_CORE_FP8_H_

#include <cstdint>

#include "core/fp16.h"

namespace llm {

struct Fp8Format {
  int exponent_bits;
  int mantissa_bits;
  // У E5M2 верхнее поле порядка занято бесконечностью и NaN, как в IEEE. У
  // E4M3 бесконечности нет, и верхнее поле — обычные числа, кроме одного кода
  // со всеми единицами мантиссы.
  bool has_infinity;
};

constexpr Fp8Format kFp8E4M3 = {4, 3, false};
constexpr Fp8Format kFp8E5M2 = {5, 2, true};

namespace fp8_detail {

inline int bias(const Fp8Format& format) {
  return (1 << (format.exponent_bits - 1)) - 1;
}

inline int max_exponent_field(const Fp8Format& format) {
  return (1 << format.exponent_bits) - 1;
}

// Поле порядка и мантисса наибольшего конечного значения.
inline int top_normal_field(const Fp8Format& format) {
  return format.has_infinity ? max_exponent_field(format) - 1
                             : max_exponent_field(format);
}

inline uint32_t top_mantissa(const Fp8Format& format) {
  const uint32_t all_ones = (1u << format.mantissa_bits) - 1u;
  // У формата без бесконечности верхний код мантиссы при верхнем порядке
  // занят NaN, поэтому наибольшее конечное на единицу меньше.
  return format.has_infinity ? all_ones : all_ones - 1u;
}

// На сколько разрядов сдвигается мантисса float, чтобы попасть в узкий формат.
inline int mantissa_shift(const Fp8Format& format) {
  return 23 - format.mantissa_bits;
}

inline uint8_t nan_code(const Fp8Format& format, uint32_t sign) {
  const uint32_t field = static_cast<uint32_t>(max_exponent_field(format));
  const uint32_t mantissa =
      format.has_infinity ? 1u : (1u << format.mantissa_bits) - 1u;
  return static_cast<uint8_t>(
      sign | (field << format.mantissa_bits) | mantissa);
}

inline uint8_t saturated_code(const Fp8Format& format, uint32_t sign) {
  return static_cast<uint8_t>(
      sign |
      (static_cast<uint32_t>(top_normal_field(format)) << format.mantissa_bits) |
      top_mantissa(format));
}

}  // namespace fp8_detail

// float -> восьмиразрядный формат, округление к ближайшему чётному.
inline uint8_t to_fp8(float value, const Fp8Format& format) {
  using namespace fp8_detail;

  const uint32_t bits = float_bits(value);
  const uint32_t sign = (bits >> 24) & 0x80u;
  const uint32_t raw_exponent = (bits >> 23) & 0xFFu;
  const uint32_t mantissa = bits & 0x7FFFFFu;
  const int shift = mantissa_shift(format);

  if (raw_exponent == 0xFFu) {
    if (mantissa != 0) {
      return nan_code(format, sign);
    }
    // Бесконечность. В формате без неё превращается в наибольшее конечное: то
    // же соображение, что и для переполнения, — NaN погубил бы всё вычисление.
    if (!format.has_infinity) {
      return saturated_code(format, sign);
    }
    return static_cast<uint8_t>(
        sign | (static_cast<uint32_t>(max_exponent_field(format))
                << format.mantissa_bits));
  }

  const int32_t exponent =
      static_cast<int32_t>(raw_exponent) - 127 + bias(format);

  if (exponent > top_normal_field(format)) {
    return saturated_code(format, sign);
  }

  if (exponent <= 0) {
    // Меньше половины наименьшего субнормального — округляется в нуль.
    if (exponent < -format.mantissa_bits) {
      return static_cast<uint8_t>(sign);
    }
    // Субнормальная область: неявная единица становится явной, и число
    // сдвигается вправо до фиксированного порядка.
    const uint32_t full = mantissa | 0x800000u;
    const uint32_t total_shift = static_cast<uint32_t>(shift + 1 - exponent);
    uint32_t result = full >> total_shift;
    const uint32_t remainder = full & ((1u << total_shift) - 1u);
    const uint32_t half = 1u << (total_shift - 1);
    if (remainder > half || (remainder == half && (result & 1u) != 0u)) {
      ++result;
    }
    return static_cast<uint8_t>(sign | result);
  }

  uint32_t result = mantissa >> shift;
  const uint32_t remainder = mantissa & ((1u << shift) - 1u);
  const uint32_t half = 1u << (shift - 1);
  int32_t final_exponent = exponent;
  if (remainder > half || (remainder == half && (result & 1u) != 0u)) {
    ++result;
    if (result == (1u << format.mantissa_bits)) {
      // Округление переполнило мантиссу: она обнуляется, порядок растёт.
      result = 0;
      ++final_exponent;
    }
  }
  if (final_exponent > top_normal_field(format) ||
      (final_exponent == top_normal_field(format) &&
       result > top_mantissa(format))) {
    return saturated_code(format, sign);
  }
  return static_cast<uint8_t>(
      sign | (static_cast<uint32_t>(final_exponent) << format.mantissa_bits) |
      result);
}

// Восьмиразрядный формат -> float. Преобразование точное: каждое значение
// восьмиразрядного формата представимо в float.
inline float from_fp8(uint8_t code, const Fp8Format& format) {
  using namespace fp8_detail;

  const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24;
  const uint32_t all_mantissa = (1u << format.mantissa_bits) - 1u;
  const uint32_t mantissa = static_cast<uint32_t>(code) & all_mantissa;
  const uint32_t field =
      (static_cast<uint32_t>(code) >> format.mantissa_bits) &
      static_cast<uint32_t>(max_exponent_field(format));
  const int shift = mantissa_shift(format);

  if (field == static_cast<uint32_t>(max_exponent_field(format))) {
    if (format.has_infinity) {
      return bits_as_float(sign | 0x7F800000u | (mantissa << shift));
    }
    if (mantissa == all_mantissa) {
      return bits_as_float(sign | 0x7FC00000u);  // единственный NaN формата
    }
    // У формата без бесконечности верхнее поле — обычные числа.
  }

  if (field == 0) {
    if (mantissa == 0) {
      return bits_as_float(sign);  // ноль со знаком
    }
    // Субнормальное: значение равно мантисса * 2^(1 - bias - mantissa_bits).
    // Нормализуется тем же приёмом, что и в fp16: сдвигом влево до появления
    // неявной единицы, с поправкой порядка на число сдвигов.
    uint32_t shifted = mantissa;
    int32_t shifts = 0;
    const uint32_t implicit = 1u << format.mantissa_bits;
    while ((shifted & implicit) == 0) {
      shifted <<= 1;
      ++shifts;
    }
    shifted &= all_mantissa;
    const int32_t result_exponent = 127 - bias(format) + 1 - shifts;
    return bits_as_float(sign |
                         (static_cast<uint32_t>(result_exponent) << 23) |
                         (shifted << shift));
  }

  const int32_t result_exponent =
      static_cast<int32_t>(field) - bias(format) + 127;
  return bits_as_float(sign | (static_cast<uint32_t>(result_exponent) << 23) |
                       (mantissa << shift));
}

// Значение, округлённое до точности восьмиразрядного формата.
inline float round_to_fp8(float value, const Fp8Format& format) {
  return from_fp8(to_fp8(value, format), format);
}

}  // namespace llm

#endif  // LLM_CORE_FP8_H_
