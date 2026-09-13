// Обучение: расписание, оптимизатор, данные, чекпоинты и главная проверка —
// переобучение на одном батче.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "autograd/ops.h"
#include "core/random.h"
#include "serialize/checkpoint.h"
#include "testing.h"
#include "train/optimizer.h"
#include "train/schedule.h"
#include "train/trainer.h"

namespace {

using llm::autograd::Var;
using llm::nn::Model;
using llm::nn::ModelConfig;

ModelConfig test_config() {
  ModelConfig config;
  config.vocab_size = 24;
  config.d_model = 16;
  config.n_layers = 2;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.max_seq_len = 12;
  config.ffn_hidden = 24;
  config.validate();
  return config;
}

std::vector<int32_t> random_ids(int64_t count, int64_t vocab,
                                std::uint64_t seed) {
  llm::Rng rng(seed);
  std::vector<int32_t> ids;
  for (int64_t i = 0; i < count; ++i) {
    ids.push_back(
        static_cast<int32_t>(rng.index(static_cast<uint64_t>(vocab))));
  }
  return ids;
}

// Один скалярный параметр и оптимизатор над ним: на такой задаче поведение
// AdamW можно проверить в лоб.
struct ScalarProblem {
  Var parameter;
  std::vector<llm::nn::NamedParameter> handles;

  explicit ScalarProblem(float start)
      : parameter(
            Var::leaf(llm::Tensor::full(llm::Shape({1, 1}), start), true)) {
    llm::nn::NamedParameter handle;
    handle.name = "x";
    handle.value = &parameter;
    handles.push_back(handle);
  }
};

}  // namespace

LLM_TEST(Train, ScheduleWarmsUpThenDecays) {
  llm::train::ScheduleConfig config;
  config.max_learning_rate = 1.0f;
  config.min_ratio = 0.1f;
  config.warmup_steps = 10;
  config.total_steps = 100;

  // Разогрев: строго возрастает и доходит ровно до максимума.
  float previous = 0.0f;
  for (int64_t step = 0; step < 10; ++step) {
    const float value = llm::train::learning_rate_at(config, step);
    LLM_CHECK_MSG(value > previous, "разогрев не возрастает на шаге " << step);
    previous = value;
  }
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 9), 1.0, 1e-6);

  // Первый шаг не должен быть холостым.
  LLM_CHECK_GT(llm::train::learning_rate_at(config, 0), 0.0f);

  // Затухание: строго убывает и приходит к нижней границе.
  previous = 2.0f;
  for (int64_t step = 10; step < 100; ++step) {
    const float value = llm::train::learning_rate_at(config, step);
    LLM_CHECK_MSG(value < previous, "затухание не убывает на шаге " << step);
    previous = value;
  }
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 100), 0.1, 1e-5);
  // За пределами расписания скорость остаётся на нижней границе.
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 500), 0.1, 1e-5);
}

LLM_TEST(Train, ScheduleWithoutWarmup) {
  llm::train::ScheduleConfig config;
  config.max_learning_rate = 1.0f;
  config.warmup_steps = 0;
  config.total_steps = 10;
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 0), 1.0, 1e-6);
}

LLM_TEST(Train, OptimizerFindsMinimumOfQuadratic) {
  // f(x) = (x - 3)^2, минимум в 3.
  //
  // Скорость идёт по расписанию, а не постоянная, и это существенно. Шаг
  // Adam по величине близок к скорости обучения независимо от того, насколько
  // мал градиент, поэтому с постоянной скоростью он вблизи минимума не
  // останавливается, а колеблется вокруг него с амплитудой порядка этой
  // скорости. Затухание к концу — не украшение расписания, а то, что вообще
  // позволяет дойти до минимума.
  ScalarProblem problem(0.0f);
  llm::train::AdamWConfig config;
  config.weight_decay = 0.0f;
  llm::train::AdamW optimizer(problem.handles, config);

  llm::train::ScheduleConfig schedule;
  schedule.max_learning_rate = 0.1f;
  schedule.min_ratio = 0.0f;
  schedule.warmup_steps = 10;
  schedule.total_steps = 400;

  const Var target = Var::constant(llm::Tensor::full(llm::Shape({1, 1}), 3.0f));
  float halfway = 0.0f;
  for (int step = 0; step < 400; ++step) {
    const Var difference = llm::autograd::sub(problem.parameter, target);
    Var loss =
        llm::autograd::sum_all(llm::autograd::mul(difference, difference));
    optimizer.zero_grad();
    loss.backward();
    optimizer.step(llm::train::learning_rate_at(schedule, step));
    if (step == 200) {
      halfway = problem.parameter.value()(0, 0);
    }
  }
  LLM_EXPECT_NEAR(problem.parameter.value()(0, 0), 3.0, 1e-4);
  LLM_CHECK_EQ(optimizer.step_count(), static_cast<std::int64_t>(400));

  // К середине прогона уже около минимума, но точнее приходит только к концу,
  // когда скорость упала.
  LLM_CHECK_MSG(std::fabs(halfway - 3.0f) < 0.05f,
                "к середине прогона не подошли к минимуму: " << halfway);
}

LLM_TEST(Train, FirstStepSizeEqualsLearningRate) {
  // Проверка поправки на смещение моментов.
  //
  // Оба момента стартуют с нуля, поэтому после первого шага первый момент
  // равен (1 - b1) * g, а второй — (1 - b2) * g^2. Поправка делит их на
  // (1 - b1) и (1 - b2), и отношение превращается ровно в g / |g|, то есть в
  // знак градиента. Значит первый шаг обязан сдвинуть параметр ровно на
  // скорость обучения, какой бы величины ни был градиент.
  //
  // Без поправки тот же шаг вышел бы в 0.1 / sqrt(0.05) ≈ 0.45 раза короче, и
  // обучение начиналось бы медленнее, чем задано расписанием. Ошибка тихая:
  // модель всё равно учится, просто не так, как написано.
  const float gradients[] = {1.0f, 25.0f, 0.001f};
  for (std::size_t i = 0; i < 3; ++i) {
    ScalarProblem problem(0.0f);
    llm::train::AdamWConfig config;
    config.weight_decay = 0.0f;
    llm::train::AdamW optimizer(problem.handles, config);

    llm::autograd::sum_all(
        llm::autograd::mul_scalar(problem.parameter, gradients[i]))
        .backward();
    optimizer.step(0.1f);

    LLM_CHECK_MSG(std::fabs(problem.parameter.value()(0, 0) + 0.1f) < 1e-5f,
                  "при градиенте " << gradients[i]
                                   << " первый шаг сместил параметр на "
                                   << -problem.parameter.value()(0, 0)
                                   << " вместо скорости обучения 0.1");
  }
}

LLM_TEST(Train, WeightDecayPullsTowardZero) {
  // Без градиента распад веса — единственная сила, и она обязана тянуть вес
  // к нулю.
  ScalarProblem problem(1.0f);
  llm::train::AdamWConfig config;
  config.weight_decay = 0.5f;
  llm::train::AdamW optimizer(problem.handles, config);

  // Градиент нулевой, но определённый: иначе шаг пропускается целиком.
  llm::autograd::sum_all(llm::autograd::mul_scalar(problem.parameter, 0.0f))
      .backward();
  const float before = problem.parameter.value()(0, 0);
  optimizer.step(0.1f);
  const float after = problem.parameter.value()(0, 0);

  LLM_CHECK_MSG(after < before, "распад веса не уменьшил вес");
  // Шаг распада — ровно lr * wd * w.
  LLM_EXPECT_NEAR(before - after, 0.1 * 0.5 * 1.0, 1e-5);
}

LLM_TEST(Train, WeightDecaySkipsOneDimensionalParameters) {
  // Масштабы нормировок одномерны и распаду не подлежат: тянуть их к нулю
  // значит глушить слой.
  Model model(test_config(), 5);
  llm::train::AdamWConfig config;
  llm::train::AdamW optimizer(model.parameters(), config);

  const std::vector<llm::nn::NamedParameter> parameters = model.parameters();
  int64_t matrix_elements = 0;
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    if (parameters[i].value->value().rank() >= 2) {
      matrix_elements += parameters[i].value->numel();
    }
  }
  LLM_CHECK_EQ(optimizer.decayed_parameter_count(), matrix_elements);
  LLM_CHECK_LT(matrix_elements, model.parameter_count());
}

LLM_TEST(Train, ClipGradNormPreservesDirection) {
  ScalarProblem problem(0.0f);
  llm::train::AdamW optimizer(problem.handles, llm::train::AdamWConfig());

  // Градиент величиной 10 при пороге 1: обрезка обязана сделать его ровно 1,
  // не меняя знака.
  llm::autograd::sum_all(llm::autograd::mul_scalar(problem.parameter, 10.0f))
      .backward();
  const float norm = optimizer.clip_grad_norm(1.0f);
  LLM_EXPECT_NEAR(norm, 10.0, 1e-5);
  LLM_EXPECT_NEAR(problem.parameter.grad()(0, 0), 1.0, 1e-5);
}

LLM_TEST(Train, ClipGradNormLeavesSmallGradientsAlone) {
  ScalarProblem problem(0.0f);
  llm::train::AdamW optimizer(problem.handles, llm::train::AdamWConfig());

  llm::autograd::sum_all(llm::autograd::mul_scalar(problem.parameter, 0.25f))
      .backward();
  const float norm = optimizer.clip_grad_norm(1.0f);
  LLM_EXPECT_NEAR(norm, 0.25, 1e-6);
  LLM_EXPECT_NEAR(problem.parameter.grad()(0, 0), 0.25, 1e-6);
}

LLM_TEST(Train, ClipGradNormIsGlobal) {
  // Обрезка считает норму по всем параметрам сразу. Если бы она работала
  // послойно, соотношение между градиентами разных слоёв изменилось бы, то
  // есть направление шага исказилось бы.
  Var first = Var::leaf(llm::Tensor::full(llm::Shape({1, 1}), 0.0f), true);
  Var second = Var::leaf(llm::Tensor::full(llm::Shape({1, 1}), 0.0f), true);
  std::vector<llm::nn::NamedParameter> handles;
  llm::nn::NamedParameter a;
  a.name = "a";
  a.value = &first;
  llm::nn::NamedParameter b;
  b.name = "b";
  b.value = &second;
  handles.push_back(a);
  handles.push_back(b);

  llm::train::AdamW optimizer(handles, llm::train::AdamWConfig());
  llm::autograd::add(
      llm::autograd::sum_all(llm::autograd::mul_scalar(first, 3.0f)),
      llm::autograd::sum_all(llm::autograd::mul_scalar(second, 4.0f)))
      .backward();

  // Норма вектора (3, 4) равна 5.
  const float norm = optimizer.clip_grad_norm(1.0f);
  LLM_EXPECT_NEAR(norm, 5.0, 1e-5);
  // Соотношение 3:4 обязано сохраниться.
  LLM_EXPECT_NEAR(first.grad()(0, 0), 0.6, 1e-5);
  LLM_EXPECT_NEAR(second.grad()(0, 0), 0.8, 1e-5);
}

LLM_TEST(Train, DatasetSplitsValidationFromTheEnd) {
  std::vector<int32_t> tokens;
  for (int i = 0; i < 100; ++i) {
    tokens.push_back(i);
  }
  const llm::data::TokenDataset dataset(tokens, 0.2);
  LLM_CHECK_EQ(dataset.train_size(), static_cast<std::int64_t>(80));
  LLM_CHECK_EQ(dataset.validation_size(), static_cast<std::int64_t>(20));

  // Обучающая выборка не должна выдавать токены из проверочной части.
  llm::Rng rng(1);
  for (int trial = 0; trial < 50; ++trial) {
    const std::vector<int32_t> batch = dataset.sample_batch(2, 5, &rng, false);
    for (std::size_t i = 0; i < batch.size(); ++i) {
      LLM_CHECK_LT(batch[i], 80);
    }
  }
  // И наоборот.
  for (int trial = 0; trial < 50; ++trial) {
    const std::vector<int32_t> batch = dataset.sample_batch(2, 5, &rng, true);
    for (std::size_t i = 0; i < batch.size(); ++i) {
      LLM_CHECK_GE(batch[i], 80);
    }
  }
}

LLM_TEST(Train, ValidationBatchesAreDeterministicAndDisjoint) {
  std::vector<int32_t> tokens;
  for (int i = 0; i < 100; ++i) {
    tokens.push_back(i);
  }
  const llm::data::TokenDataset dataset(tokens, 0.2);

  const std::int64_t count = dataset.validation_batch_count(2, 5);
  LLM_CHECK_EQ(count, static_cast<std::int64_t>(2));

  // Повторный вызов даёт то же самое: сравнение прогонов должно идти на одних
  // и тех же данных.
  LLM_CHECK(dataset.validation_batch(2, 5, 0) ==
            dataset.validation_batch(2, 5, 0));
  // Окна не перекрываются.
  const std::vector<int32_t> first = dataset.validation_batch(2, 5, 0);
  const std::vector<int32_t> second = dataset.validation_batch(2, 5, 1);
  LLM_CHECK_LT(first.back(), second.front());
}

LLM_TEST(Train, DatasetRejectsWindowLongerThanCorpus) {
  std::vector<int32_t> tokens(10, 0);
  const llm::data::TokenDataset dataset(tokens, 0.0);
  llm::Rng rng(1);
  LLM_EXPECT_THROWS(dataset.sample_batch(1, 50, &rng, false));
}

LLM_TEST(Train, CheckpointRoundtrip) {
  const std::string path = "test_checkpoint.llmw";
  const ModelConfig config = test_config();

  Model saved(config, 101);
  // Немного обучим, чтобы веса отличались от начальных.
  llm::train::overfit_batch(&saved, random_ids(2 * 8, config.vocab_size, 1), 2,
                            8, 3, 1e-3f);
  llm::serialize::save_checkpoint(path, &saved, 42);

  LLM_CHECK(llm::serialize::read_config(path).to_string() ==
            config.to_string());

  Model loaded(config, 999);  // другое зерно, значит другие веса
  const std::int64_t step = llm::serialize::load_checkpoint(path, &loaded);
  LLM_CHECK_EQ(step, static_cast<std::int64_t>(42));

  // После загрузки обе модели обязаны выдавать одинаковые логиты побитово.
  const std::vector<int32_t> ids = random_ids(8, config.vocab_size, 2);
  const Var a = saved.forward(ids, 1, 8);
  const Var b = loaded.forward(ids, 1, 8);
  for (std::int64_t position = 0; position < 8; ++position) {
    for (std::int64_t token = 0; token < config.vocab_size; ++token) {
      LLM_CHECK_MSG(
          a.value()(0, position, token) == b.value()(0, position, token),
          "логиты разошлись после загрузки чекпоинта");
    }
  }
  std::remove(path.c_str());
}

LLM_TEST(Train, CheckpointRejectsDifferentModel) {
  const std::string path = "test_mismatch.llmw";
  Model saved(test_config(), 1);
  llm::serialize::save_checkpoint(path, &saved, 0);

  ModelConfig other = test_config();
  other.n_layers = 3;
  Model different(other, 1);
  LLM_EXPECT_THROWS(llm::serialize::load_checkpoint(path, &different));
  std::remove(path.c_str());
}

LLM_TEST(Train, CheckpointRejectsSameShapeDifferentHyperparameters) {
  // Форма совпадает, но theta для RoPE другая — веса несовместимы, а по
  // to_string() модели выглядят одинаково. Сравнение обязано идти по всем
  // полям.
  const std::string path = "test_theta.llmw";
  Model saved(test_config(), 1);
  llm::serialize::save_checkpoint(path, &saved, 0);

  ModelConfig other = test_config();
  other.rope_theta = 500000.0f;
  Model different(other, 1);
  LLM_EXPECT_THROWS(llm::serialize::load_checkpoint(path, &different));
  std::remove(path.c_str());
}

LLM_TEST(Train, ReadsOlderCheckpointFormat) {
  // Версия 1 формата не знала про архитектурные развилки. Старые чекпоинты
  // обязаны читаться: иначе смена формата заставляла бы переобучать модель.
  //
  // Файл версии 1 строится из версии 2 удалением четырёх байтов развилок и
  // правкой номера версии — так тест не зависит от того, сохранился ли где-то
  // настоящий старый файл.
  const std::string path = "test_v2.llmw";
  const ModelConfig config = test_config();
  Model saved(config, 3);
  llm::train::overfit_batch(&saved, random_ids(2 * 8, config.vocab_size, 12), 2,
                            8, 3, 1e-3f);
  llm::serialize::save_checkpoint(path, &saved, 77);

  std::string bytes;
  {
    std::ifstream input(path.c_str(), std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    bytes = buffer.str();
  }
  std::remove(path.c_str());

  // Смещение полей, добавленных после версии 1: magic, версия, метка порядка
  // байт, метка float, семь int64 и три float конфигурации, байт связывания
  // эмбеддингов.
  const std::size_t after_v1 = 4 + 4 + 4 + 4 + 7 * 8 + 3 * 4 + 1;
  // Версия 2 добавила четыре байта развилок, версия 3 — байт QK-нормы и
  // коэффициент z-loss.
  const std::size_t v2_fields = 4;
  const std::size_t v3_fields = 1 + 4;
  LLM_CHECK_GT(bytes.size(), after_v1 + v2_fields + v3_fields);
  bytes[4] = 1;  // версия 1
  bytes[5] = 0;
  bytes[6] = 0;
  bytes[7] = 0;
  bytes.erase(after_v1, v2_fields + v3_fields);

  const std::string old_path = "test_v1.llmw";
  {
    std::ofstream output(old_path.c_str(), std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }

  // Конфигурация читается с значениями по умолчанию для развилок — ровно с
  // теми, с которыми версия 1 и писалась.
  LLM_CHECK(llm::serialize::read_config(old_path) == config);

  Model loaded(config, 999);
  LLM_CHECK_EQ(llm::serialize::load_checkpoint(old_path, &loaded),
               static_cast<std::int64_t>(77));

  const std::vector<int32_t> probe = random_ids(6, config.vocab_size, 13);
  const Var a = saved.forward(probe, 1, 6);
  const Var b = loaded.forward(probe, 1, 6);
  for (std::int64_t position = 0; position < 6; ++position) {
    for (std::int64_t token = 0; token < config.vocab_size; ++token) {
      LLM_CHECK(a.value()(0, position, token) == b.value()(0, position, token));
    }
  }
  std::remove(old_path.c_str());
}

LLM_TEST(Train, CheckpointRejectsForeignFile) {
  const std::string path = "test_foreign.llmw";
  {
    std::ofstream file(path.c_str(), std::ios::binary);
    file << "это не чекпоинт, а просто текст";
  }
  Model model(test_config(), 1);
  LLM_EXPECT_THROWS(llm::serialize::load_checkpoint(path, &model));
  std::remove(path.c_str());
}

LLM_TEST(Train, OverfitsSingleBatch) {
  // Главная проверка обучения.
  //
  // Маленькая модель обязана уметь просто запомнить один батч: потери должны
  // уйти почти в нуль. Если не уходят, сломано что-то из трёх — градиенты,
  // оптимизатор или модель, — и никакое настоящее обучение работать не будет.
  const ModelConfig config = test_config();
  Model model(config, 7);
  const std::vector<int32_t> ids = random_ids(2 * 8, config.vocab_size, 3);

  const std::vector<float> history =
      llm::train::overfit_batch(&model, ids, 2, 8, 250, 3e-3f);

  const float начало = history.front();
  const float конец = history.back();
  LLM_EXPECT_NEAR(начало, std::log(static_cast<double>(config.vocab_size)),
                  0.3);
  LLM_CHECK_MSG(конец < 0.05f, "за 250 шагов потери упали только с "
                                   << начало << " до " << конец);
}

LLM_TEST(Train, TrainingReducesLossOnRealData) {
  // Настоящий, хоть и крошечный, прогон: данные повторяются, поэтому модель
  // обязана заметно продвинуться за небольшое число шагов.
  const ModelConfig config = test_config();
  Model model(config, 11);

  std::vector<int32_t> tokens;
  llm::Rng rng(13);
  // Повторяющийся узор: предсказуемый, но не тривиальный.
  for (int repeat = 0; repeat < 400; ++repeat) {
    for (int i = 0; i < 7; ++i) {
      tokens.push_back((i * 3 + 1) % static_cast<int>(config.vocab_size));
    }
  }
  const llm::data::TokenDataset dataset(tokens, 0.1);

  llm::train::TrainConfig train_config;
  train_config.steps = 60;
  train_config.batch_size = 4;
  train_config.seq_len = 8;
  train_config.warmup_steps = 5;
  train_config.max_learning_rate = 3e-3f;
  train_config.log_every = 0;
  train_config.eval_every = 60;
  train_config.verbose = false;

  const llm::train::TrainReport report =
      llm::train::train(&model, dataset, train_config);

  LLM_CHECK_EQ(report.train_loss.size(), static_cast<std::size_t>(60));
  LLM_CHECK_MSG(report.final_train_loss < report.train_loss.front() * 0.5f,
                "потери почти не снизились: " << report.train_loss.front()
                                              << " -> "
                                              << report.final_train_loss);
  // Все нормы градиента конечны: расходимость проявилась бы здесь.
  for (std::size_t i = 0; i < report.grad_norm.size(); ++i) {
    LLM_CHECK(std::isfinite(report.grad_norm[i]));
  }
}
