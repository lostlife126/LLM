#include "core/half.h"

namespace llm {

void floats_to_half(const float* source, Half* destination, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    destination[i].bits = to_fp16(source[i]);
  }
}

void half_to_floats(const Half* source, float* destination, int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    destination[i] = from_fp16(source[i].bits);
  }
}

}  // namespace llm
