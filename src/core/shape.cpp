#include "core/shape.h"

#include <algorithm>
#include <limits>
#include <sstream>

namespace llm {

void Shape::validate() const {
  for (int i = 0; i < dims_.size(); ++i) {
    LLM_CHECK_MSG(dims_[i] >= 0,
                  "отрицательный размер по оси " << i << ": " << dims_[i]);
  }
}

int64_t Shape::numel() const {
  // Для скаляра (ранг 0) произведение по пустому множеству равно 1: в тензоре
  // ранга 0 ровно один элемент.
  int64_t total = 1;
  for (int i = 0; i < dims_.size(); ++i) {
    if (dims_[i] == 0) {
      return 0;
    }
    // Переполнение здесь означало бы молчаливо неверный размер буфера.
    LLM_CHECK_MSG(
        total <= std::numeric_limits<int64_t>::max() / dims_[i],
        "переполнение при подсчёте числа элементов формы " << to_string());
    total *= dims_[i];
  }
  return total;
}

std::string Shape::to_string() const {
  std::ostringstream oss;
  oss << "(";
  for (int i = 0; i < dims_.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << dims_[i];
  }
  oss << ")";
  return oss.str();
}

bool operator==(const Shape& lhs, const Shape& rhs) {
  return lhs.dims() == rhs.dims();
}

bool operator!=(const Shape& lhs, const Shape& rhs) { return !(lhs == rhs); }

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  return os << shape.to_string();
}

Dims contiguous_strides(const Shape& shape) {
  const int rank = shape.rank();
  Dims strides = Dims::zeros(rank);
  int64_t running = 1;
  for (int axis = rank - 1; axis >= 0; --axis) {
    strides[axis] = running;
    running *= shape.dim(axis);
  }
  return strides;
}

bool try_broadcast(const Shape& lhs, const Shape& rhs, Shape* out) {
  const int rank = std::max(lhs.rank(), rhs.rank());
  const int lhs_pad = rank - lhs.rank();
  const int rhs_pad = rank - rhs.rank();

  Dims dims = Dims::zeros(rank);
  for (int axis = 0; axis < rank; ++axis) {
    // Отсутствующие слева оси считаются равными 1 — это и есть выравнивание
    // справа.
    const int64_t left = axis < lhs_pad ? 1 : lhs.dim(axis - lhs_pad);
    const int64_t right = axis < rhs_pad ? 1 : rhs.dim(axis - rhs_pad);
    if (left != right && left != 1 && right != 1) {
      return false;
    }
    // Растягивается ВСЕГДА единица, и берётся размер соседа — а не больший из
    // двух. Разница видна ровно там, где ось пуста: 0 и 1 совместимы, и numpy
    // даёт 0, потому что единственная строка растягивается в ноль строк.
    // max(0, 1) дал бы 1, то есть форму с элементом, которого нет ни в одном
    // из аргументов.
    dims[axis] = left == 1 ? right : left;
  }
  if (out != nullptr) {
    *out = Shape(dims);
  }
  return true;
}

Shape broadcast_shapes(const Shape& lhs, const Shape& rhs) {
  Shape result;
  LLM_CHECK_MSG(try_broadcast(lhs, rhs, &result),
                "формы " << lhs << " и " << rhs << " несовместимы");
  return result;
}

}  // namespace llm
