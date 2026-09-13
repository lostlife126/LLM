#include "train/trainer.h"

#include <chrono>
#include <cmath>
#include <cstdio>

#include "core/check.h"
#include "serialize/checkpoint.h"

namespace llm {
namespace train {
namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

float evaluate(nn::Model* model, const data::TokenDataset& dataset,
               int64_t batch, int64_t seq, int64_t max_batches) {
  LLM_CHECK(model != nullptr);
  const int64_t available = dataset.validation_batch_count(batch, seq);
  if (available == 0) {
    return 0.0f;
  }
  const int64_t count = available < max_batches ? available : max_batches;

  // Лента не нужна: считается только значение, и без неё не расходуется
  // память на промежуточные величины.
  autograd::NoGradGuard no_grad;
  double total = 0.0;
  for (int64_t index = 0; index < count; ++index) {
    const std::vector<int32_t> ids =
        dataset.validation_batch(batch, seq, index);
    total += *model->loss(ids, batch, seq).value().data();
  }
  return static_cast<float>(total / static_cast<double>(count));
}

TrainReport train(nn::Model* model, const data::TokenDataset& dataset,
                  const TrainConfig& config) {
  LLM_CHECK(model != nullptr);
  const int64_t seq =
      config.seq_len > 0 ? config.seq_len : model->config().max_seq_len;
  LLM_CHECK_LE(seq, model->config().max_seq_len);

  AdamWConfig optimizer_config;
  optimizer_config.weight_decay = config.weight_decay;
  AdamW optimizer(model->parameters(), optimizer_config);

  ScheduleConfig schedule;
  schedule.max_learning_rate = config.max_learning_rate;
  schedule.min_ratio = config.min_lr_ratio;
  schedule.warmup_steps = config.warmup_steps;
  schedule.total_steps = config.steps;

  Rng rng(config.seed);
  TrainReport report;
  const Clock::time_point start = Clock::now();

  if (config.verbose) {
    std::printf("параметров: %lld, из них с распадом веса: %lld\n",
                static_cast<long long>(model->parameter_count()),
                static_cast<long long>(optimizer.decayed_parameter_count()));
    std::printf("обучающих токенов: %lld, проверочных: %lld\n",
                static_cast<long long>(dataset.train_size()),
                static_cast<long long>(dataset.validation_size()));
  }

  for (int64_t step = 0; step < config.steps; ++step) {
    const std::vector<int32_t> ids =
        dataset.sample_batch(config.batch_size, seq, &rng, false);

    autograd::Var loss = model->loss(ids, config.batch_size, seq);
    const float loss_value = *loss.value().data();
    LLM_CHECK_MSG(std::isfinite(loss_value),
                  "потери перестали быть конечными на шаге " << step);

    optimizer.zero_grad();
    loss.backward();
    const float norm = optimizer.clip_grad_norm(config.grad_clip);
    const float learning_rate = learning_rate_at(schedule, step);
    optimizer.step(learning_rate);

    report.train_loss.push_back(loss_value);
    report.grad_norm.push_back(norm);

    const bool last = step + 1 == config.steps;
    if (config.verbose && config.log_every > 0 &&
        ((step + 1) % config.log_every == 0 || last)) {
      const double elapsed =
          std::chrono::duration<double>(Clock::now() - start).count();
      const double tokens =
          static_cast<double>((step + 1) * config.batch_size * seq);
      std::printf(
          "шаг %6lld  потери %7.4f  норма %7.3f  скорость %.2e  %.0f ток/с\n",
          static_cast<long long>(step + 1), loss_value, norm,
          static_cast<double>(learning_rate), tokens / elapsed);
      std::fflush(stdout);
    }

    if (config.eval_every > 0 &&
        ((step + 1) % config.eval_every == 0 || last)) {
      const float validation =
          evaluate(model, dataset, config.batch_size, seq, config.eval_batches);
      report.final_validation_loss = validation;
      if (config.verbose && validation > 0.0f) {
        std::printf("             проверочные потери %7.4f\n", validation);
        std::fflush(stdout);
      }
    }

    if (config.checkpoint_every > 0 && !config.checkpoint_path.empty() &&
        ((step + 1) % config.checkpoint_every == 0 || last)) {
      serialize::save_checkpoint(config.checkpoint_path, model, step + 1);
    }
  }

  report.seconds = std::chrono::duration<double>(Clock::now() - start).count();
  if (!report.train_loss.empty()) {
    report.final_train_loss = report.train_loss.back();
  }
  return report;
}

std::vector<float> overfit_batch(nn::Model* model,
                                 const std::vector<int32_t>& ids, int64_t batch,
                                 int64_t seq, int64_t steps,
                                 float learning_rate) {
  LLM_CHECK(model != nullptr);
  AdamWConfig optimizer_config;
  // Регуляризация здесь мешает: цель — именно запомнить батч.
  optimizer_config.weight_decay = 0.0f;
  AdamW optimizer(model->parameters(), optimizer_config);

  std::vector<float> history;
  for (int64_t step = 0; step < steps; ++step) {
    autograd::Var loss = model->loss(ids, batch, seq);
    history.push_back(*loss.value().data());
    optimizer.zero_grad();
    loss.backward();
    optimizer.clip_grad_norm(1.0f);
    optimizer.step(learning_rate);
  }
  return history;
}

}  // namespace train
}  // namespace llm
