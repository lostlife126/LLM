#include "core/cpu.h"

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
  // Пробел ставится ПЕРЕД очередным признаком, а не после: иначе строка
  // кончается пробелом, и в шапках измерений получается «avx512vl , ядер 4».
  std::string text;
  const auto add = [&text](bool present, const char* name) {
    if (!present) {
      return;
    }
    if (!text.empty()) {
      text += ' ';
    }
    text += name;
  };
  add(neon, "neon");
  add(avx2, "avx2");
  add(fma, "fma");
  add(f16c, "f16c");
  add(avx512f, "avx512f");
  add(avx512bw, "avx512bw");
  add(avx512vl, "avx512vl");
  return text.empty() ? std::string("базовый x86-64") : text;
}

int detect_core_count() {
  const unsigned int count = std::thread::hardware_concurrency();
  return static_cast<int>(count);
}

}  // namespace llm
