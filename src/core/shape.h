// Shape — размеры тензора по осям.
//
// Отделено от Tensor намеренно: правила совместимости форм (broadcast) и
// вычисление непрерывных шагов — самостоятельная логика, которую нужно
// тестировать отдельно от владения памятью.
//
// Скаляр — это тензор ранга 0: numel() == 1, шагов нет. Такой нужен для
// значения функции потерь, от которого пойдёт обратный проход.

#ifndef LLM_CORE_SHAPE_H_
#define LLM_CORE_SHAPE_H_

#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

#include "core/check.h"
#include "core/dims.h"

namespace llm {

class Shape {
 public:
  Shape() {}
  Shape(std::initializer_list<int64_t> dims) : dims_(dims) { validate(); }
  explicit Shape(const std::vector<int64_t>& dims) : dims_(dims) { validate(); }
  explicit Shape(const Dims& dims) : dims_(dims) { validate(); }

  int rank() const { return dims_.size(); }
  bool is_scalar() const { return dims_.empty(); }
  const Dims& dims() const { return dims_; }

  // Отрицательная ось отсчитывается с конца, как в numpy: dim(-1) — последняя.
  // Внутри операций это избавляет от постоянного writing rank() - 1.
  int normalize_axis(int axis) const {
    const int normalized = axis < 0 ? axis + rank() : axis;
    // to_string(), а не operator<<: он объявлен ниже класса.
    LLM_CHECK_MSG(normalized >= 0 && normalized < rank(),
                  "ось " << axis << " вне диапазона для формы " << to_string());
    return normalized;
  }

  int64_t dim(int axis) const { return dims_[normalize_axis(axis)]; }

  int64_t operator[](int axis) const { return dim(axis); }

  int64_t numel() const;

  std::string to_string() const;

 private:
  void validate() const;

  Dims dims_;
};

bool operator==(const Shape& lhs, const Shape& rhs);
bool operator!=(const Shape& lhs, const Shape& rhs);
std::ostream& operator<<(std::ostream& os, const Shape& shape);

// Шаги плотного размещения в памяти: последняя ось идёт с шагом 1, каждая
// предыдущая — с шагом, равным произведению всех последующих размеров.
Dims contiguous_strides(const Shape& shape);

// Совместимость форм по правилам numpy: оси выравниваются справа, ось размера
// 1 растягивается до размера соседа. Возвращает false вместо падения, потому
// что вызывающий иногда хочет проверить, а не потребовать.
bool try_broadcast(const Shape& lhs, const Shape& rhs, Shape* out);

// То же, но несовместимость — ошибка логики.
Shape broadcast_shapes(const Shape& lhs, const Shape& rhs);

}  // namespace llm

#endif  // LLM_CORE_SHAPE_H_
