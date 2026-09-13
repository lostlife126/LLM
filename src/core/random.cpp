#include "core/random.h"

#include <cmath>

namespace llm {

float Rng::normal() {
  if (has_spare_normal_) {
    has_spare_normal_ = false;
    return spare_normal_;
  }
  // Полярная форма: берём точки в квадрате [-1, 1]^2, пока не попадём внутрь
  // единичного круга. В отличие от тригонометрической формы не требует
  // вычисления синуса и косинуса.
  double x = 0.0;
  double y = 0.0;
  double radius_squared = 0.0;
  do {
    x = 2.0 * uniform() - 1.0;
    y = 2.0 * uniform() - 1.0;
    radius_squared = x * x + y * y;
  } while (radius_squared >= 1.0 || radius_squared == 0.0);

  const double factor =
      std::sqrt(-2.0 * std::log(radius_squared) / radius_squared);
  spare_normal_ = static_cast<float>(y * factor);
  has_spare_normal_ = true;
  return static_cast<float>(x * factor);
}

}  // namespace llm
