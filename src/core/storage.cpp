#include "core/storage.h"

#include <cstring>
#include <memory>

namespace llm {

const std::size_t Storage::kAlignment;

Storage::Storage(std::size_t nbytes)
    : raw_(nullptr), aligned_(nullptr), nbytes_(nbytes) {
  if (nbytes == 0) {
    return;
  }
  // Запас в kAlignment - 1 байт гарантирует, что внутри выделенного блока
  // найдётся выровненный адрес, с которого помещается nbytes байт.
  const std::size_t capacity = nbytes + kAlignment - 1;
  raw_ = new unsigned char[capacity];

  void* cursor = raw_;
  std::size_t space = capacity;
  aligned_ = std::align(kAlignment, nbytes, cursor, space);
  LLM_CHECK_MSG(
      aligned_ != nullptr,
      "std::align не нашла выровненный адрес для " << nbytes << " байт");
}

Storage::~Storage() { release(); }

Storage::Storage(Storage&& other)
    : raw_(other.raw_), aligned_(other.aligned_), nbytes_(other.nbytes_) {
  other.raw_ = nullptr;
  other.aligned_ = nullptr;
  other.nbytes_ = 0;
}

Storage& Storage::operator=(Storage&& other) {
  if (this != &other) {
    release();
    raw_ = other.raw_;
    aligned_ = other.aligned_;
    nbytes_ = other.nbytes_;
    other.raw_ = nullptr;
    other.aligned_ = nullptr;
    other.nbytes_ = 0;
  }
  return *this;
}

void Storage::zero() {
  if (nbytes_ != 0) {
    std::memset(aligned_, 0, nbytes_);
  }
}

void Storage::release() {
  delete[] raw_;
  raw_ = nullptr;
  aligned_ = nullptr;
  nbytes_ = 0;
}

}  // namespace llm
