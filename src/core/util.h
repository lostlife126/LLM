// Мелкие утилиты, которых нет в C++14.

#ifndef LLM_CORE_UTIL_H_
#define LLM_CORE_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <string>

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

// Дополняет строку пробелами до нужной ширины В СИМВОЛАХ.
//
// printf с %-20s считает байты, а кириллица в UTF-8 занимает по два, поэтому
// таблицы с русскими подписями разъезжаются. Ведущие байты символа UTF-8 —
// это все, кроме продолжающих, у которых старшие два бита равны 10.
inline std::string pad_utf8(const std::string& text, std::size_t width) {
  std::size_t characters = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
      ++characters;
    }
  }
  if (characters >= width) {
    return text;
  }
  return text + std::string(width - characters, ' ');
}

// То же выравнивание, но текст прижимается вправо. Нужно для шапок числовых
// столбцов: printf выравнивает по байтам, а в кириллице их вдвое больше, чем
// символов, и шапка уезжает относительно чисел под ней.
inline std::string pad_utf8_right(const std::string& text, std::size_t width) {
  std::size_t characters = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
      ++characters;
    }
  }
  if (characters >= width) {
    return text;
  }
  return std::string(width - characters, ' ') + text;
}

inline bool is_aligned(const void* pointer, std::size_t alignment) {
  LLM_DCHECK_NE(alignment, static_cast<std::size_t>(0));
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

}  // namespace llm

#endif  // LLM_CORE_UTIL_H_
