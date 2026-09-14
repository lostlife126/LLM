// AdamW.
//
// Отличие от Adam — в том, где применяется распад весов. У Adam его добавляют
// к градиенту, и тогда он проходит через нормировку на второй момент: у
// параметра с большими градиентами распад получается слабее, чем у параметра
// с малыми. Это не то, чего от регуляризации ждут. AdamW применяет распад
// прямо к весу, отдельно от градиентного шага — отсюда и буква W (decoupled
// weight decay).
//
// Распад применяется только к матрицам. Масштабы нормировок и любые другие
// одномерные параметры от него освобождены: тянуть масштаб RMSNorm к нулю
// значит глушить слой, а это не регуляризация, а поломка.

#ifndef LLM_TRAIN_OPTIMIZER_H_
#define LLM_TRAIN_OPTIMIZER_H_

#include <cstdint>
#include <vector>

#include "core/tensor.h"
#include "nn/model.h"

namespace llm {
namespace train {

struct AdamWConfig {
  float beta1 = 0.9f;

  // 0.95, а не привычные 0.999. Для языковых моделей длинное окно второго
  // момента мешает: распределение градиентов заметно меняется по ходу
  // обучения, и оценка не должна тянуться за давно устаревшими батчами.
  float beta2 = 0.95f;

  float eps = 1e-8f;
  float weight_decay = 0.1f;
};

class AdamW {
 public:
  AdamW(std::vector<nn::NamedParameter> parameters, const AdamWConfig& config);

  // Скорость обучения передаётся на каждом шаге: её задаёт расписание, а не
  // оптимизатор.
  void step(float learning_rate);

  void zero_grad();

  // Глобальная норма градиентов по всем параметрам. Если она больше max_norm,
  // все градиенты умножаются на max_norm / norm.
  //
  // Обрезка именно глобальная, а не по каждому параметру отдельно: важно
  // сохранить направление шага, а поэлементная или послойная обрезка его
  // искажает. Возвращается норма ДО обрезки — по ней видно, когда обучение
  // начинает срываться.
  float clip_grad_norm(float max_norm);

  int64_t step_count() const { return step_; }

  // Отношение длины шага к длине самого веса, медиана по параметрам последнего
  // шага.
  //
  // Классический диагностический признак: у здорового обучения он держится
  // около 1e-3. Заметно больше — шаг переписывает веса, и обучение вот-вот
  // сорвётся; заметно меньше — модель практически стоит, и дело либо в
  // слишком малой скорости, либо в том, что градиент уже нулевой.
  //
  // Величина полезнее нормы градиента: та говорит, насколько велик градиент
  // сам по себе, а эта — насколько велик он относительно того, что двигает.
  float last_update_ratio() const { return last_update_ratio_; }

  // Отношение по каждому параметру отдельно — чтобы видеть, какой слой
  // движется, а какой стоит.
  const std::vector<float>& update_ratios() const { return update_ratios_; }
  const std::vector<nn::NamedParameter>& parameters() const {
    return parameters_;
  }

  // Число параметров, к которым применяется распад веса.
  int64_t decayed_parameter_count() const;

  // Моменты и счётчик шагов — наружу ради возобновления прерванного обучения.
  //
  // Без них продолжить нельзя: Адам накапливает оценки первого и второго
  // момента градиента, и обучение, начатое с нуля от середины, первые сотни
  // шагов идёт вслепую. Счётчик нужен для поправки на смещение, которая от
  // него и зависит.
  const std::vector<Tensor>& first_moment() const { return first_moment_; }
  const std::vector<Tensor>& second_moment() const { return second_moment_; }

  // Восстанавливает состояние. Формы обязаны совпадать с параметрами: иначе
  // это состояние от другой модели, и продолжать с него нельзя.
  void restore(std::vector<Tensor> first, std::vector<Tensor> second,
               int64_t step);

 private:
  std::vector<nn::NamedParameter> parameters_;
  std::vector<Tensor> first_moment_;
  std::vector<Tensor> second_moment_;
  std::vector<bool> apply_decay_;
  AdamWConfig config_;
  int64_t step_;
  float last_update_ratio_ = 0.0f;
  std::vector<float> update_ratios_;
};

}  // namespace train
}  // namespace llm

#endif  // LLM_TRAIN_OPTIMIZER_H_
