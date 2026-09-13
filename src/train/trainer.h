// Цикл обучения.
//
// Вынесен из приложения, чтобы его можно было запустить из теста: проверка
// «модель переобучается на одном батче до нулевых потерь» — единственный
// способ убедиться, что оптимизатор, градиенты и модель работают вместе, а не
// по отдельности.

#ifndef LLM_TRAIN_TRAINER_H_
#define LLM_TRAIN_TRAINER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "data/dataset.h"
#include "nn/model.h"
#include "train/optimizer.h"
#include "train/schedule.h"

namespace llm {
namespace train {

struct TrainConfig {
  int64_t steps = 2000;
  int64_t batch_size = 16;
  int64_t seq_len = 0;  // 0 означает полный контекст модели

  float max_learning_rate = 3e-4f;
  int64_t warmup_steps = 100;
  float min_lr_ratio = 0.1f;
  float weight_decay = 0.1f;

  // Обрезка градиента по глобальной норме. Один неудачный батч способен дать
  // огромный градиент и одним шагом разрушить всё обученное; обрезка
  // ограничивает ущерб, сохраняя направление шага.
  float grad_clip = 1.0f;

  int64_t log_every = 50;
  int64_t eval_every = 200;
  int64_t eval_batches = 8;
  int64_t checkpoint_every = 0;  // 0 — не сохранять
  std::string checkpoint_path;

  uint64_t seed = 1234;
  bool verbose = true;
};

struct TrainReport {
  std::vector<float> train_loss;  // потери на каждом шаге
  std::vector<float> grad_norm;  // норма градиента до обрезки
  float final_train_loss = 0.0f;
  float final_validation_loss = 0.0f;
  double seconds = 0.0;
};

// Средние потери на проверочной части, по фиксированным неперекрывающимся
// окнам: сравнение прогонов должно идти на одних и тех же данных.
float evaluate(nn::Model* model, const data::TokenDataset& dataset,
               int64_t batch, int64_t seq, int64_t max_batches);

TrainReport train(nn::Model* model, const data::TokenDataset& dataset,
                  const TrainConfig& config);

// Обучение на одном и том же батче — проверка, что связка модели,
// градиентов и оптимизатора вообще способна что-то выучить. Потери обязаны
// уйти почти в нуль: батч маленький, и модель может его просто запомнить.
std::vector<float> overfit_batch(nn::Model* model,
                                 const std::vector<int32_t>& ids, int64_t batch,
                                 int64_t seq, int64_t steps,
                                 float learning_rate);

}  // namespace train
}  // namespace llm

#endif  // LLM_TRAIN_TRAINER_H_
