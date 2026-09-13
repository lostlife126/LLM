// Span<T> — невладеющий вид на непрерывный участок памяти.
//
// Замена std::span, появившейся только в C++20. Нужен повсюду: тензор отдаёт
// свои данные ядрам операций именно так, не раскрывая, кому принадлежит буфер
// и как он размещён.

#ifndef LLM_CORE_SPAN_H_
#define LLM_CORE_SPAN_H_

#include <cstddef>
#include <type_traits>
#include <vector>

#include "core/check.h"

namespace llm {

template <typename T>
class Span {
 public:
  using value_type = typename std::remove_cv<T>::type;

  Span() : data_(nullptr), size_(0) {}
  Span(T* data, std::size_t size) : data_(data), size_(size) {}

  template <std::size_t N>
  explicit Span(T (&array)[N]) : data_(array), size_(N) {}

  // Неявное ослабление Span<T> -> Span<const T>: функция, которая только
  // читает, должна принимать const-вид, и вызывающему не нужно об этом думать.
  template <typename U, typename = typename std::enable_if<
                            std::is_same<const U, T>::value>::type>
  Span(const Span<U>& other) : data_(other.data()), size_(other.size()) {}

  T* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  T& operator[](std::size_t index) const {
    LLM_DCHECK_LT(index, size_);
    return data_[index];
  }

  T& front() const {
    LLM_DCHECK_GT(size_, static_cast<std::size_t>(0));
    return data_[0];
  }

  T& back() const {
    LLM_DCHECK_GT(size_, static_cast<std::size_t>(0));
    return data_[size_ - 1];
  }

  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }

  Span<T> subspan(std::size_t offset) const {
    LLM_DCHECK_LE(offset, size_);
    return Span<T>(data_ + offset, size_ - offset);
  }

  Span<T> subspan(std::size_t offset, std::size_t count) const {
    LLM_DCHECK_LE(offset, size_);
    LLM_DCHECK_LE(count, size_ - offset);
    return Span<T>(data_ + offset, count);
  }

 private:
  T* data_;
  std::size_t size_;
};

template <typename T>
Span<T> make_span(std::vector<T>& vec) {
  return Span<T>(vec.data(), vec.size());
}

template <typename T>
Span<const T> make_span(const std::vector<T>& vec) {
  return Span<const T>(vec.data(), vec.size());
}

}  // namespace llm

#endif  // LLM_CORE_SPAN_H_
