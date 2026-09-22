#include "train/schedule.h"

#include <cmath>

#include "core/check.h"

namespace llm {
namespace train {

float learning_rate_at(const ScheduleConfig& config, int64_t step) {
  LLM_CHECK_GE(step, static_cast<int64_t>(0));
  LLM_CHECK_GT(config.total_steps, static_cast<int64_t>(0));
  LLM_CHECK_GE(config.warmup_steps, static_cast<int64_t>(0));
  LLM_CHECK_LE(config.warmup_steps, config.total_steps);
  // Нижняя граница — доля от максимума, и доля эта обязана лежать в [0, 1].
  // Отрицательная дала бы отрицательную скорость к концу прогона, то есть
  // подъём по градиенту вместо спуска, — тихо, без падения и без единого
  // NaN. Больше единицы превратила бы затухание в разгон.
  LLM_CHECK_MSG(config.min_ratio >= 0.0f && config.min_ratio <= 1.0f,
                "нижняя доля скорости " << config.min_ratio << " вне [0, 1]");
  LLM_CHECK_MSG(
      config.max_learning_rate >= 0.0f,
      "скорость обучения " << config.max_learning_rate << " отрицательна");

  const float minimum = config.max_learning_rate * config.min_ratio;

  if (step < config.warmup_steps) {
    // Линейно от нуля до максимума. Первый шаг делается не с нулевой
    // скоростью, а с одной warmup_steps-й: иначе он был бы холостым.
    const float progress =
        static_cast<float>(step + 1) / static_cast<float>(config.warmup_steps);
    return config.max_learning_rate * progress;
  }

  if (step >= config.total_steps) {
    return minimum;
  }

  const float progress =
      static_cast<float>(step - config.warmup_steps) /
      static_cast<float>(config.total_steps - config.warmup_steps);
  // Косинус от 0 до pi даёт плавный переход от 1 к 0.
  const float cosine =
      0.5f * (1.0f + std::cos(3.14159265358979323846f * progress));
  return minimum + (config.max_learning_rate - minimum) * cosine;
}

}  // namespace train
}  // namespace llm
