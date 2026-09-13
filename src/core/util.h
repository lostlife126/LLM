// Мелкие утилиты, которых нет в C++14.

#ifndef LLM_CORE_UTIL_H_
#define LLM_CORE_UTIL_H_

#include <cstddef>
#include <cstdint>

#include "core/check.h"

namespace llm {

// std::clamp появилась только в C++17.
//
// В отличие от std::clamp, возвращаем по значению, а не по const-ссылке.
// Ссылочная подпись std::clamp — известная ловушка: clamp(x, 0.0f, 1.0f)
// вернёт ссылку на временный объект. Здесь clamp применяется к скалярам
// (температура, вероятности, границы блоков), копия ничего не стоит.
template <typename T>
T clamp(T value, T low, T high) {
  LLM_DCHECK_LE(low, high);
  return value < low ? low : (high < value ? high : value);
}

// Деление с округлением вверх: сколько блоков размера divisor нужно, чтобы
// покрыть value. Используется при разбиении матриц на блоки.
inline std::size_t div_up(std::size_t value, std::size_t divisor) {
  LLM_DCHECK_NE(divisor, static_cast<std::size_t>(0));
  return (value + divisor - 1) / divisor;
}

// Ближайшее сверху число, кратное multiple.
inline std::size_t round_up(std::size_t value, std::size_t multiple) {
  return div_up(value, multiple) * multiple;
}

inline bool is_aligned(const void* pointer, std::size_t alignment) {
  LLM_DCHECK_NE(alignment, static_cast<std::size_t>(0));
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

}  // namespace llm

#endif  // LLM_CORE_UTIL_H_
