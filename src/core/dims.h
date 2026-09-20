// Dims — небольшой массив величин по осям: сама форма или шаги по ней.
//
// Зачем не std::vector. Ранг тензора в этом проекте не превышает четырёх, а
// объектов тензора за один шаг обучения строится тринадцать тысяч — это
// измерено, — и у каждого форма и шаги лежали в куче. Два выделения на объект,
// около двадцати шести тысяч мелких malloc за шаг, и ни одно из них не нужно:
// содержимое умещается в несколько машинных слов и прекрасно живёт по значению.
//
// Заодно исчезают выделения внутри обходов: счётчик индекса там тоже был
// вектором и заводился на каждый вызов.
//
// Потолок ранга проверяется, а не подразумевается: превысить его значит
// молча испортить память, и такую ошибку лучше получить сразу и по имени.

#ifndef LLM_CORE_DIMS_H_
#define LLM_CORE_DIMS_H_

#include <cstdint>
#include <vector>

#include "core/check.h"

namespace llm {

class Dims {
 public:
  // Восемь с запасом: модель пользуется четырьмя осями, шесть — уже
  // экзотика. Запас ничего не стоит, массив всё равно лежит по значению.
  static const int kMaxRank = 8;

  Dims() : size_(0) {}

  Dims(std::initializer_list<int64_t> values) : size_(0) {
    for (const int64_t* it = values.begin(); it != values.end(); ++it) {
      push_back(*it);
    }
  }

  explicit Dims(const std::vector<int64_t>& values) : size_(0) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      push_back(values[i]);
    }
  }

  // Массив из count нулей — под счётчик обхода.
  static Dims zeros(int count) {
    Dims result;
    for (int i = 0; i < count; ++i) {
      result.push_back(0);
    }
    return result;
  }

  int size() const { return size_; }
  bool empty() const { return size_ == 0; }

  void push_back(int64_t value) {
    LLM_CHECK_MSG(size_ < kMaxRank, "ранг больше " << kMaxRank
                                                   << ", это не поддерживается");
    values_[size_++] = value;
  }

  void clear() { size_ = 0; }

  int64_t& operator[](int index) {
    LLM_DCHECK(index >= 0 && index < size_);
    return values_[index];
  }

  int64_t operator[](int index) const {
    LLM_DCHECK(index >= 0 && index < size_);
    return values_[index];
  }

  const int64_t* begin() const { return values_; }
  const int64_t* end() const { return values_ + size_; }

  std::vector<int64_t> to_vector() const {
    return std::vector<int64_t>(values_, values_ + size_);
  }

 private:
  // Обнуляются все восемь ячеек, а не только занятые. Копирование Dims —
  // почленное, то есть копирует массив целиком, и ячейки за size_ читались бы
  // неинициализированными. Это неопределённое поведение, а не безобидная
  // мелочь: GCC ловит его как -Wmaybe-uninitialized, как только видит копию
  // насквозь. Восемь записей на объект измеримой цены не имеют — шаг nano с
  // обнулением и без него идёт одинаково.
  int64_t values_[kMaxRank] = {};
  int size_;
};

inline bool operator==(const Dims& lhs, const Dims& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

inline bool operator!=(const Dims& lhs, const Dims& rhs) {
  return !(lhs == rhs);
}

}  // namespace llm

#endif  // LLM_CORE_DIMS_H_
