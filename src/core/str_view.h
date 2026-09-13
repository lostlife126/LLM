// StrView — невладеющий вид на строку.
//
// Замена std::string_view из C++17. Понадобится в токенизаторе (разбор корпуса
// и таблицы слияний) и в разборе заголовков файлов весов, где копировать каждую
// подстроку в std::string было бы расточительно.
//
// Как и std::string_view, не гарантирует завершающего нуля: c_str() здесь нет
// намеренно.

#ifndef LLM_CORE_STR_VIEW_H_
#define LLM_CORE_STR_VIEW_H_

#include <cstddef>
#include <cstring>
#include <ostream>
#include <string>

#include "core/check.h"

namespace llm {

class StrView {
 public:
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  StrView() : data_(""), size_(0) {}
  StrView(const char* str) : data_(str), size_(std::strlen(str)) {}  // NOLINT
  StrView(const char* str, std::size_t size) : data_(str), size_(size) {}
  StrView(const std::string& str)
      : data_(str.data()), size_(str.size()) {}  // NOLINT

  const char* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  char operator[](std::size_t index) const {
    LLM_DCHECK_LT(index, size_);
    return data_[index];
  }

  const char* begin() const { return data_; }
  const char* end() const { return data_ + size_; }

  StrView substr(std::size_t pos, std::size_t count = npos) const {
    LLM_DCHECK_LE(pos, size_);
    const std::size_t available = size_ - pos;
    return StrView(data_ + pos, count < available ? count : available);
  }

  bool starts_with(StrView prefix) const {
    return size_ >= prefix.size_ &&
           std::memcmp(data_, prefix.data_, prefix.size_) == 0;
  }

  bool ends_with(StrView suffix) const {
    return size_ >= suffix.size_ &&
           std::memcmp(data_ + size_ - suffix.size_, suffix.data_,
                       suffix.size_) == 0;
  }

  std::size_t find(char ch, std::size_t pos = 0) const {
    for (std::size_t i = pos; i < size_; ++i) {
      if (data_[i] == ch) {
        return i;
      }
    }
    return npos;
  }

  std::string to_string() const { return std::string(data_, size_); }

 private:
  const char* data_;
  std::size_t size_;
};

inline bool operator==(StrView lhs, StrView rhs) {
  return lhs.size() == rhs.size() &&
         (lhs.size() == 0 ||
          std::memcmp(lhs.data(), rhs.data(), lhs.size()) == 0);
}

inline bool operator!=(StrView lhs, StrView rhs) { return !(lhs == rhs); }

inline bool operator<(StrView lhs, StrView rhs) {
  const std::size_t common = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
  const int cmp = common == 0 ? 0 : std::memcmp(lhs.data(), rhs.data(), common);
  return cmp != 0 ? cmp < 0 : lhs.size() < rhs.size();
}

inline std::ostream& operator<<(std::ostream& os, StrView view) {
  return os.write(view.data(), static_cast<std::streamsize>(view.size()));
}

}  // namespace llm

#endif  // LLM_CORE_STR_VIEW_H_
