#include "core/cpu.h"

#include <sstream>
#include <thread>

namespace llm {
namespace {

CpuFeatures probe() {
  CpuFeatures out;
#if defined(__x86_64__) || defined(_M_X64)
  // __builtin_cpu_init обязателен перед первым __builtin_cpu_supports, если
  // тот вызывается не из функции с целевым атрибутом. У clang он не нужен, но
  // и не мешает.
  __builtin_cpu_init();
  out.avx2 = __builtin_cpu_supports("avx2") != 0;
  out.fma = __builtin_cpu_supports("fma") != 0;
  out.f16c = __builtin_cpu_supports("f16c") != 0;
  out.avx512f = __builtin_cpu_supports("avx512f") != 0;
  out.avx512bw = __builtin_cpu_supports("avx512bw") != 0;
  out.avx512vl = __builtin_cpu_supports("avx512vl") != 0;
#endif
#if defined(__aarch64__)
  out.neon = true;
#endif
  return out;
}

}  // namespace

const CpuFeatures& cpu_features() {
  // Инициализация статической локальной переменной потокобезопасна начиная с
  // C++11: компилятор сам ставит защёлку.
  static const CpuFeatures features = probe();
  return features;
}

std::string CpuFeatures::to_string() const {
  std::ostringstream out;
  if (neon) {
    out << "neon ";
  }
  if (avx2) {
    out << "avx2 ";
  }
  if (fma) {
    out << "fma ";
  }
  if (f16c) {
    out << "f16c ";
  }
  if (avx512f) {
    out << "avx512f ";
  }
  if (avx512bw) {
    out << "avx512bw ";
  }
  if (avx512vl) {
    out << "avx512vl ";
  }
  const std::string text = out.str();
  return text.empty() ? std::string("базовый x86-64") : text;
}

int detect_core_count() {
  const unsigned int count = std::thread::hardware_concurrency();
  return static_cast<int>(count);
}

}  // namespace llm
