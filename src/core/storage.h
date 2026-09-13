// Storage — владеющий буфер сырых байт с гарантированным выравниванием.
//
// Единственное место в проекте, где выделяется память под данные тензоров.
// Выравнивание на границу кэш-линии (64 байта) нужно по двум причинам: строки
// матриц не делят кэш-линию с чужими данными, и автовекторизатор вправе
// использовать выровненные загрузки.
//
// Стандарту C++14 не хватает std::aligned_alloc (она появилась в C++17), а
// posix_memalign — это POSIX, а не стандартная библиотека. Поэтому выделяем с
// запасом обычным new[] и сдвигаем указатель через std::align.

#ifndef LLM_CORE_STORAGE_H_
#define LLM_CORE_STORAGE_H_

#include <cstddef>

#include "core/check.h"
#include "core/span.h"

namespace llm {

class Storage {
 public:
  static const std::size_t kAlignment = 64;

  Storage() : raw_(nullptr), aligned_(nullptr), nbytes_(0) {}
  explicit Storage(std::size_t nbytes);
  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  Storage(Storage&& other);
  Storage& operator=(Storage&& other);

  void* data() { return aligned_; }
  const void* data() const { return aligned_; }
  std::size_t nbytes() const { return nbytes_; }

  // Обнуляет весь буфер.
  void zero();

  // Типизированный доступ. Размер буфера обязан быть кратен sizeof(T): иначе
  // последний элемент окажется усечённым, и это почти наверняка ошибка.
  template <typename T>
  Span<T> as() {
    LLM_CHECK_EQ(nbytes_ % sizeof(T), static_cast<std::size_t>(0));
    return Span<T>(static_cast<T*>(aligned_), nbytes_ / sizeof(T));
  }

  template <typename T>
  Span<const T> as() const {
    LLM_CHECK_EQ(nbytes_ % sizeof(T), static_cast<std::size_t>(0));
    return Span<const T>(static_cast<const T*>(aligned_), nbytes_ / sizeof(T));
  }

 private:
  void release();

  unsigned char* raw_;  // то, что вернул new[]; только его можно удалять
  void* aligned_;  // то, чем пользуется остальной код
  std::size_t nbytes_;
};

}  // namespace llm

#endif  // LLM_CORE_STORAGE_H_
