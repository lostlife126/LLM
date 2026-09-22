// Округление до точности половинной разрядности и обратно.
//
// Нужно затем, чтобы проверить сходимость обучения в fp16 **до** того, как
// тип элемента тензора будет переделан. Значения остаются в float, но каждое
// проходит через представление половинной разрядности и возвращается — то
// есть получается ровно та точность, которую дало бы настоящее хранение, при
// неизменном коде. Работа на полчаса вместо нескольких дней, и главный риск
// проверяется первым, а не последним.
//
// Преобразование написано разрядными операциями, а не через _Float16: тип
// появился в компиляторах недавно и есть не везде, а поведение на краях —
// переполнение, субнормальные числа, округление к чётному — должно быть
// одинаковым на всех машинах, иначе вывод о сходимости окажется машинным
// свойством, а не свойством арифметики.
//
// Формат: знак (1), показатель (5, смещение 15), мантисса (10). Наибольшее
// конечное значение 65504, наименьшее нормальное 2^-14, субнормальные — до
// 2^-24.

#ifndef LLM_CORE_FP16_H_
#define LLM_CORE_FP16_H_

#include <cstdint>
#include <cstring>

namespace llm {

inline uint32_t float_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float bits_as_float(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// float -> половинная разрядность, округление к ближайшему чётному.
inline uint16_t to_fp16(float value) {
  const uint32_t bits = float_bits(value);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t raw_exponent = (bits >> 23) & 0xFFu;
  const uint32_t mantissa = bits & 0x7FFFFFu;

  if (raw_exponent == 0xFFu) {
    // Бесконечность и NaN. У NaN обязан остаться ненулевым хоть один разряд
    // мантиссы, иначе он превратится в бесконечность; старший разряд ставится
    // явно, потому что при сдвиге на 13 младшие могли быть единственными.
    return static_cast<uint16_t>(
        sign | 0x7C00u | (mantissa != 0 ? (0x200u | (mantissa >> 13)) : 0u));
  }

  const int32_t exponent = static_cast<int32_t>(raw_exponent) - 127 + 15;

  if (exponent >= 0x1F) {
    return static_cast<uint16_t>(sign | 0x7C00u);  // переполнение
  }

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);  // меньше половины наименьшего
    }
    // Субнормальная область: неявная единица становится явной, и число
    // сдвигается вправо до фиксированного показателя.
    const uint32_t full = mantissa | 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t result = full >> shift;
    const uint32_t remainder = full & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (remainder > half || (remainder == half && (result & 1u) != 0u)) {
      ++result;
    }
    return static_cast<uint16_t>(sign | result);
  }

  uint32_t result = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1FFFu;
  int32_t final_exponent = exponent;
  if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u) != 0u)) {
    ++result;
    if (result == 0x400u) {
      // Округление переполнило мантиссу: она обнуляется, показатель растёт.
      result = 0;
      ++final_exponent;
      if (final_exponent >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00u);
      }
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(final_exponent) << 10) | result);
}

// Половинная разрядность -> float. Преобразование точное: каждое значение
// половинной разрядности представимо в float.
inline float from_fp16(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  const uint32_t exponent = (value >> 10) & 0x1Fu;
  const uint32_t mantissa = value & 0x3FFu;

  if (exponent == 0) {
    if (mantissa == 0) {
      return bits_as_float(sign);  // ноль со знаком
    }
    // Субнормальное: значение равно мантисса * 2^-24. Нормализуется сдвигом
    // влево, пока не появится разряд 2^10; после s сдвигов значение равно
    // (shifted / 1024) * 2^(-14 - s), то есть поле показателя float — это
    // 127 - 14 - s = 113 - s.
    uint32_t shifted = mantissa;
    int32_t shifts = 0;
    while ((shifted & 0x400u) == 0) {
      shifted <<= 1;
      ++shifts;
    }
    shifted &= 0x3FFu;
    const uint32_t result_exponent = static_cast<uint32_t>(113 - shifts) << 23;
    return bits_as_float(sign | result_exponent | (shifted << 13));
  }
  if (exponent == 0x1F) {
    return bits_as_float(sign | 0x7F800000u | (mantissa << 13));
  }
  const uint32_t result_exponent = (exponent - 15 + 127) << 23;
  return bits_as_float(sign | result_exponent | (mantissa << 13));
}

// Значение, округлённое до точности половинной разрядности.
inline float round_to_fp16(float value) { return from_fp16(to_fp16(value)); }

}  // namespace llm

#endif  // LLM_CORE_FP16_H_
