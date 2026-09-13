// Обход тензоров с произвольными шагами.
//
// Плотный тензор обходится простым циклом по памяти, а вид с шагами — нет:
// соседние по индексу элементы могут лежать где угодно. Здесь собран «одометр»,
// который идёт по всем позициям формы в порядке плотного размещения и на каждом
// шаге выдаёт смещения сразу в нескольких наборах шагов.
//
// Несколько наборов нужны, потому что поэлементные операции работают с двумя
// входами и одним выходом, у каждого из которых свои шаги — например, после
// растяжения одного из аргументов шагом 0.

#ifndef LLM_CORE_ITERATE_H_
#define LLM_CORE_ITERATE_H_

#include <cstdint>
#include <vector>

#include "core/shape.h"

namespace llm {
namespace detail {

// Увеличивает счётчик index на единицу в порядке «младший разряд — последняя
// ось» и поправляет смещения. При переносе разряд сбрасывается в нуль, а из
// смещения вычитается весь накопленный по этой оси вклад.
//
// Корректно работает с шагом 0 (растянутая ось) и с любой перестановкой осей.
inline void advance(const Shape& shape, std::vector<int64_t>* index,
                    const std::vector<int64_t>* const* strides,
                    int64_t* offsets, int count) {
  for (int axis = shape.rank() - 1; axis >= 0; --axis) {
    const std::size_t a = static_cast<std::size_t>(axis);
    (*index)[a] += 1;
    for (int t = 0; t < count; ++t) {
      offsets[t] += (*strides[t])[a];
    }
    if ((*index)[a] < shape.dim(axis)) {
      return;
    }
    for (int t = 0; t < count; ++t) {
      offsets[t] -= (*index)[a] * (*strides[t])[a];
    }
    (*index)[a] = 0;
  }
}

}  // namespace detail

template <typename Fn>
void for_each_offset(const Shape& shape, const std::vector<int64_t>& strides,
                     Fn fn) {
  const int64_t total = shape.numel();
  if (total == 0) {
    return;
  }
  std::vector<int64_t> index(static_cast<std::size_t>(shape.rank()), 0);
  const std::vector<int64_t>* all[] = {&strides};
  int64_t offsets[] = {0};
  for (int64_t counter = 0; counter < total; ++counter) {
    fn(offsets[0]);
    detail::advance(shape, &index, all, offsets, 1);
  }
}

template <typename Fn>
void for_each_offset2(const Shape& shape, const std::vector<int64_t>& strides_a,
                      const std::vector<int64_t>& strides_b, Fn fn) {
  const int64_t total = shape.numel();
  if (total == 0) {
    return;
  }
  std::vector<int64_t> index(static_cast<std::size_t>(shape.rank()), 0);
  const std::vector<int64_t>* all[] = {&strides_a, &strides_b};
  int64_t offsets[] = {0, 0};
  for (int64_t counter = 0; counter < total; ++counter) {
    fn(offsets[0], offsets[1]);
    detail::advance(shape, &index, all, offsets, 2);
  }
}

template <typename Fn>
void for_each_offset3(const Shape& shape, const std::vector<int64_t>& strides_a,
                      const std::vector<int64_t>& strides_b,
                      const std::vector<int64_t>& strides_c, Fn fn) {
  const int64_t total = shape.numel();
  if (total == 0) {
    return;
  }
  std::vector<int64_t> index(static_cast<std::size_t>(shape.rank()), 0);
  const std::vector<int64_t>* all[] = {&strides_a, &strides_b, &strides_c};
  int64_t offsets[] = {0, 0, 0};
  for (int64_t counter = 0; counter < total; ++counter) {
    fn(offsets[0], offsets[1], offsets[2]);
    detail::advance(shape, &index, all, offsets, 3);
  }
}

}  // namespace llm

#endif  // LLM_CORE_ITERATE_H_
