#include "core/quantize.h"

#include <cmath>
#include <cstring>

#include "core/check.h"
#include "core/fp16.h"
#include "core/fp8.h"

namespace llm {
namespace {

// Верхний край формата — то значение, на которое отображается наибольший по
// модулю элемент тензора.
float format_maximum(Precision precision) {
  switch (precision) {
    case Precision::kFp8E4M3:
    case Precision::kFp8E4M3NoScale:
      return 448.0f;
    case Precision::kFp8E5M2:
      return 57344.0f;
    default:
      return 0.0f;  // масштаб не применяется
  }
}

bool uses_scale(Precision precision) {
  return precision == Precision::kFp8E4M3 || precision == Precision::kFp8E5M2;
}

float round_one(float value, Precision precision) {
  switch (precision) {
    case Precision::kFp16:
      return round_to_fp16(value);
    case Precision::kFp8E4M3:
    case Precision::kFp8E4M3NoScale:
      return round_to_fp8(value, kFp8E4M3);
    case Precision::kFp8E5M2:
      return round_to_fp8(value, kFp8E5M2);
    case Precision::kFp32:
      return value;
  }
  return value;
}

}  // namespace

bool parse_precision(const char* name, Precision* out) {
  if (name == nullptr) {
    return false;
  }
  if (std::strcmp(name, "fp32") == 0) {
    *out = Precision::kFp32;
    return true;
  }
  if (std::strcmp(name, "fp16") == 0) {
    *out = Precision::kFp16;
    return true;
  }
  if (std::strcmp(name, "e4m3") == 0) {
    *out = Precision::kFp8E4M3;
    return true;
  }
  if (std::strcmp(name, "e5m2") == 0) {
    *out = Precision::kFp8E5M2;
    return true;
  }
  if (std::strcmp(name, "e4m3-raw") == 0) {
    *out = Precision::kFp8E4M3NoScale;
    return true;
  }
  return false;
}

const char* precision_name(Precision precision) {
  switch (precision) {
    case Precision::kFp32:
      return "fp32";
    case Precision::kFp16:
      return "fp16";
    case Precision::kFp8E4M3:
      return "e4m3";
    case Precision::kFp8E5M2:
      return "e5m2";
    case Precision::kFp8E4M3NoScale:
      return "e4m3-raw";
  }
  return "?";
}

int scale_exponent(const Tensor& tensor, Precision precision) {
  if (!uses_scale(precision) || !tensor.defined()) {
    return 0;
  }
  // Та же проверка, что у quantize, и не для симметрии. Функция читает numel()
  // значений подряд от data(), а у вида это неверно дважды: значения будут не
  // те, а у вида с нулевым шагом — какой делает expand — их окажется больше,
  // чем есть в хранилище, то есть чтение уйдёт за границу. Заявлено это было
  // только у quantize, хотя обе функции открыты.
  LLM_CHECK_MSG(tensor.is_contiguous(),
                "масштаб считается только по плотному тензору: у вида "
                "numel() не описывает того, что лежит подряд");
  const float* data = tensor.data();
  const int64_t count = tensor.numel();
  float largest = 0.0f;
  for (int64_t i = 0; i < count; ++i) {
    const float magnitude = std::fabs(data[i]);
    if (magnitude > largest) {
      largest = magnitude;
    }
  }
  if (largest == 0.0f) {
    return 0;
  }
  // frexp даёт largest/maximum = m * 2^e при m из [0.5, 1). Значит 2^e уже не
  // меньше отношения, и после деления на него ничего не вылезет за край.
  int exponent = 0;
  std::frexp(largest / format_maximum(precision), &exponent);
  return exponent;
}

void quantize(Tensor* tensor, Precision precision) {
  if (precision == Precision::kFp32 || tensor == nullptr ||
      !tensor->defined()) {
    return;
  }
  LLM_CHECK_MSG(tensor->is_contiguous(),
                "округлять можно только плотный тензор: вид на чужой буфер "
                "испортил бы данные владельца");

  const int exponent = scale_exponent(*tensor, precision);
  float* data = tensor->data();
  const int64_t count = tensor->numel();

  if (exponent == 0) {
    for (int64_t i = 0; i < count; ++i) {
      data[i] = round_one(data[i], precision);
    }
    return;
  }
  // ldexp, а не умножение на посчитанный множитель: так масштабирование точно
  // и при показателях, при которых сам множитель уже субнормален.
  for (int64_t i = 0; i < count; ++i) {
    const float scaled = std::ldexp(data[i], -exponent);
    data[i] = std::ldexp(round_one(scaled, precision), exponent);
  }
}

}  // namespace llm
