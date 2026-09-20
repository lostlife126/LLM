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
#include "core/cpu.h"
#include "core/thread_pool.h"
#include "serialize/checkpoint.h"
#include "testing.h"
#include "train/optimizer.h"
#include "train/resume.h"
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
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 100), 0.1, 1e-7);
  // За пределами расписания скорость остаётся на нижней границе.
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 500), 0.1, 1e-7);

  // Значение внутри затухания, а не только на концах. Монотонности и двух
  // концов мало: косинус, посчитанный от максимума без прибавления нижней
  // границы, тоже убывает и тоже приходит куда надо — потому что на шаге
  // total_steps срабатывает отдельная ветвь. Проверено: такая подмена
  // проходила обе проверки выше, и ловил её только косвенный тест на
  // качество обучения.
  //
  // Середина затухания: шаг 55 из отрезка 10..100 даёт progress = 0.5,
  // косинус от pi/2 равен нулю, множитель — ровно половина. Значит скорость
  // обязана быть min + (max - min) / 2 = 0.55.
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 55), 0.55, 1e-6);

  // Треть пути: шаг 40 даёт progress = 30/90 = 1/3, косинус от pi/3 равен
  // 0.5, множитель 0.75. Две трети: шаг 70, косинус от 2pi/3 равен -0.5,
  // множитель 0.25. Обе точки посчитаны руками, без обращения к тому же
  // косинусу, что и в реализации.
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 40),
                  0.1 + 0.9 * 0.75, 1e-6);
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 70),
                  0.1 + 0.9 * 0.25, 1e-6);
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
  LLM_EXPECT_NEAR(before - after, 0.1 * 0.5 * 1.0, 1e-6);
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
  LLM_EXPECT_NEAR(first.grad()(0, 0), 0.6, 1e-6);
  LLM_EXPECT_NEAR(second.grad()(0, 0), 0.8, 1e-6);
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

LLM_TEST(Train, CheckpointRejectsUnknownArchitectureChoice) {
  // Развилки архитектуры лежат в файле байтами, а выбираются сравнением с
  // одним значением: «это LayerNorm?». Значит незнакомый байт означал бы
  // просто другую ветвь — модель загрузилась бы и считала бы не то, что в
  // файле записано, и заметить это было бы нечем.
  const std::string path = "test_choice.llmw";
  Model saved(test_config(), 1);
  llm::serialize::save_checkpoint(path, &saved, 0);

  std::string bytes;
  {
    std::ifstream input(path.c_str(), std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    bytes = buffer.str();
  }
  std::remove(path.c_str());

  // Байт нормировки идёт сразу за заголовком и телом конфигурации версии 1:
  // magic, версия, метка порядка байт, метка float, семь int64 и три float,
  // байт связывания эмбеддингов.
  const std::size_t norm_at = 4 + 4 + 4 + 4 + 7 * 8 + 3 * 4 + 1;
  LLM_CHECK_GT(bytes.size(), norm_at);
  bytes[norm_at] = 7;  // такой нормировки не существует

  const std::string broken = "test_choice_broken.llmw";
  {
    std::ofstream output(broken.c_str(), std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  LLM_EXPECT_THROWS(llm::serialize::read_config(broken));
  std::remove(broken.c_str());
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
  // коэффициент z-loss, версия 4 — вероятность дропаута.
  const std::size_t v2_fields = 4;
  const std::size_t v3_fields = 1 + 4;
  const std::size_t v4_fields = 4;
  const std::size_t added = v2_fields + v3_fields + v4_fields;
  LLM_CHECK_GT(bytes.size(), after_v1 + added);
  bytes[4] = 1;  // версия 1
  bytes[5] = 0;
  bytes[6] = 0;
  bytes[7] = 0;
  bytes.erase(after_v1, added);

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

LLM_TEST(Train, KeepsBestCheckpointNotLast) {
  // Проверочные потери проходят минимум и дальше растут. Если сохранять
  // последний чекпоинт, на диске окажется переобученная модель, а лучшая
  // будет затёрта — ровно это случилось на первом длинном прогоне tiny, где
  // минимум пришёлся на тысячный шаг из трёх тысяч.
  //
  // Чтобы переобучение случилось быстро и наверняка, корпус делается крошечным
  // и случайным: обобщать в нём нечего, запомнить можно целиком.
  const std::string path = "test_best_checkpoint.llmw";
  ModelConfig config = test_config();
  Model model(config, 77);

  std::vector<int32_t> tokens;
  llm::Rng rng(5);
  for (int i = 0; i < 300; ++i) {
    tokens.push_back(static_cast<int32_t>(
        rng.index(static_cast<std::uint64_t>(config.vocab_size))));
  }
  const llm::data::TokenDataset dataset(tokens, 0.2);

  llm::train::TrainConfig train_config;
  train_config.steps = 400;
  train_config.batch_size = 4;
  train_config.seq_len = 8;
  train_config.warmup_steps = 5;
  train_config.max_learning_rate = 5e-3f;
  train_config.log_every = 0;
  train_config.diagnostics_every = 50;
  train_config.eval_batches = 2;
  train_config.checkpoint_path = path;
  train_config.checkpoint_every = train_config.steps;
  train_config.verbose = false;

  const llm::train::TrainReport report =
      llm::train::train(&model, dataset, train_config);

  // Проверка обязана быть содержательной: если переобучения не случилось,
  // лучший чекпоинт совпадёт с последним, и тест пройдёт, ничего не проверив.
  LLM_CHECK_MSG(report.best_step > 0 && report.best_step < train_config.steps,
                "переобучение не случилось (лучший шаг "
                    << report.best_step << " из " << train_config.steps
                    << "), и проверка ничего не значит");
  LLM_CHECK_MSG(
      report.best_validation_loss < report.final_validation_loss,
      "проверочные потери не выросли: " << report.best_validation_loss << " -> "
                                        << report.final_validation_loss);

  // На диске обязан лежать именно лучший, и узнать это можно по записанному
  // в чекпоинт номеру шага.
  Model loaded(config, 999);
  const std::int64_t step = llm::serialize::load_checkpoint(path, &loaded);
  LLM_CHECK_MSG(step == report.best_step, "сохранён шаг " << step
                                                          << ", а лучший был "
                                                          << report.best_step);

  // И это должна быть именно та модель: проверочные потери загруженной
  // обязаны совпасть с лучшими.
  const float reloaded = llm::train::evaluate(&loaded, dataset, 4, 8, 4);
  LLM_EXPECT_NEAR(reloaded, report.best_validation_loss, 1e-5);

  std::remove(path.c_str());
}

LLM_TEST(Train, WithoutValidationFallsBackToPeriodicSaving) {
  // Без проверочной выборки «лучший» определить нечем, и сохраняться должен
  // последний — иначе прогон не сохранил бы ничего вовсе.
  const std::string path = "test_periodic_checkpoint.llmw";
  const ModelConfig config = test_config();
  Model model(config, 3);

  std::vector<int32_t> tokens;
  for (int repeat = 0; repeat < 200; ++repeat) {
    for (int i = 0; i < 7; ++i) {
      tokens.push_back((i * 3 + 1) % static_cast<int>(config.vocab_size));
    }
  }
  const llm::data::TokenDataset dataset(tokens, 0.0);

  llm::train::TrainConfig train_config;
  train_config.steps = 20;
  train_config.batch_size = 4;
  train_config.seq_len = 8;
  train_config.warmup_steps = 2;
  train_config.log_every = 0;
  train_config.eval_batches = 0;
  train_config.checkpoint_path = path;
  train_config.checkpoint_every = 10;
  train_config.verbose = false;

  const llm::train::TrainReport report =
      llm::train::train(&model, dataset, train_config);
  LLM_CHECK_EQ(report.best_step, static_cast<std::int64_t>(0));

  Model loaded(config, 1);
  LLM_CHECK_EQ(llm::serialize::load_checkpoint(path, &loaded),
               static_cast<std::int64_t>(20));
  std::remove(path.c_str());
}

LLM_TEST(Train, ResumedRunMatchesUninterruptedOne) {
  // Главная проверка возобновления, и она же единственная убедительная:
  // прогон, прерванный посередине и продолженный со снимка, обязан дать ровно
  // те же потери, что и непрерывный. Побитово, а не «примерно».
  //
  // Совпасть должно всё сразу: веса, моменты Адама, счётчик шагов для поправки
  // на смещение, положение в расписании скорости и порядок батчей. Ошибка в
  // любом из пяти сдвинет числа, и видно это будет с первого же шага после
  // возобновления.
  const std::string snapshot = "test_resume.snap";
  const ModelConfig config = test_config();

  std::vector<int32_t> tokens;
  for (int repeat = 0; repeat < 300; ++repeat) {
    for (int i = 0; i < 7; ++i) {
      tokens.push_back((i * 3 + 1) % static_cast<int>(config.vocab_size));
    }
  }
  const llm::data::TokenDataset dataset(tokens, 0.1);

  llm::train::TrainConfig base;
  base.steps = 40;
  base.batch_size = 4;
  base.seq_len = 8;
  base.warmup_steps = 4;
  base.max_learning_rate = 3e-3f;
  base.log_every = 0;
  base.diagnostics_every = 0;
  base.eval_batches = 0;
  base.verbose = false;

  // Непрерывный прогон — эталон.
  Model whole(config, 21);
  const llm::train::TrainReport full = llm::train::train(&whole, dataset, base);
  LLM_CHECK_EQ(full.train_loss.size(), static_cast<std::size_t>(40));

  // Тот же прогон, прерванный на двадцатом шаге. Расписание прежнее: steps
  // по-прежнему 40, меняется только то, докуда дошли за запуск.
  std::remove(snapshot.c_str());
  Model piece(config, 21);
  llm::train::TrainConfig first = base;
  first.resume_path = snapshot;
  first.checkpoint_every = 10;
  first.stop_at_step = 20;
  const llm::train::TrainReport part_one =
      llm::train::train(&piece, dataset, first);
  LLM_CHECK_EQ(part_one.train_loss.size(), static_cast<std::size_t>(20));

  // Продолжение в свежей модели: так проверяется, что из снимка поднимается
  // всё нужное, а не что-то осталось в памяти от прошлого прогона.
  Model continued(config, 999);  // другое зерно, значит другие начальные веса
  llm::train::TrainConfig second = base;
  second.resume_path = snapshot;
  second.checkpoint_every = 10;
  const llm::train::TrainReport part_two =
      llm::train::train(&continued, dataset, second);

  LLM_CHECK_EQ(part_two.resumed_from, static_cast<std::int64_t>(20));
  LLM_CHECK_EQ(part_two.train_loss.size(), static_cast<std::size_t>(20));

  for (std::size_t i = 0; i < part_two.train_loss.size(); ++i) {
    const float expected = full.train_loss[20 + i];
    LLM_CHECK_MSG(part_two.train_loss[i] == expected,
                  "шаг " << (21 + i) << " после возобновления: "
                         << part_two.train_loss[i] << " вместо " << expected);
  }
  // Нормы градиента тоже: они чувствительнее потерь и поймают расхождение,
  // которое потери успели бы сгладить.
  for (std::size_t i = 0; i < part_two.grad_norm.size(); ++i) {
    LLM_CHECK_EQ(part_two.grad_norm[i], full.grad_norm[20 + i]);
  }
  std::remove(snapshot.c_str());
}

LLM_TEST(Train, ResumeWithoutOptimizerStateWouldDiffer) {
  // Проверка самой проверки: если бы моменты Адама не восстанавливались,
  // совпадение выше было бы недостижимо. Убеждаемся, что это действительно
  // так, а не что тест сошёлся бы при любой реализации.
  //
  // Модель поднимается из снимка обычным чекпоинтом — то есть с весами, но
  // без моментов, — и первые же шаги обязаны разойтись с эталоном.
  const std::string snapshot = "test_resume_control.snap";
  const std::string checkpoint = "test_resume_control.llmw";
  const ModelConfig config = test_config();

  std::vector<int32_t> tokens;
  for (int repeat = 0; repeat < 300; ++repeat) {
    for (int i = 0; i < 7; ++i) {
      tokens.push_back((i * 5 + 2) % static_cast<int>(config.vocab_size));
    }
  }
  const llm::data::TokenDataset dataset(tokens, 0.1);

  llm::train::TrainConfig base;
  base.steps = 40;
  base.batch_size = 4;
  base.seq_len = 8;
  base.warmup_steps = 4;
  base.max_learning_rate = 3e-3f;
  base.log_every = 0;
  base.diagnostics_every = 0;
  base.eval_batches = 0;
  base.verbose = false;

  Model whole(config, 31);
  const llm::train::TrainReport full = llm::train::train(&whole, dataset, base);

  std::remove(snapshot.c_str());
  Model piece(config, 31);
  llm::train::TrainConfig first = base;
  first.resume_path = snapshot;
  first.checkpoint_path = checkpoint;
  first.checkpoint_every = 20;
  first.stop_at_step = 20;
  llm::train::train(&piece, dataset, first);

  // Продолжение только с весов: моменты нулевые, счётчик шагов нулевой.
  Model without_moments(config, 999);
  llm::serialize::load_checkpoint(checkpoint, &without_moments);
  llm::train::TrainConfig tail = base;
  tail.steps = 20;
  tail.warmup_steps = 0;
  const llm::train::TrainReport naive =
      llm::train::train(&without_moments, dataset, tail);

  bool differs = false;
  for (std::size_t i = 0; i < naive.train_loss.size(); ++i) {
    if (naive.train_loss[i] != full.train_loss[20 + i]) {
      differs = true;
    }
  }
  LLM_CHECK_MSG(differs,
                "продолжение без моментов совпало с эталоном — значит "
                "совпадение в предыдущем тесте ничего не доказывает");

  std::remove(snapshot.c_str());
  std::remove(checkpoint.c_str());
}

LLM_TEST(Train, ResumeRefusesAnotherModel) {
  const std::string snapshot = "test_resume_other.snap";
  std::remove(snapshot.c_str());

  ModelConfig config = test_config();
  Model model(config, 5);
  llm::train::AdamWConfig adam;
  llm::train::AdamW optimizer(model.trainable_parameters(), adam);

  llm::train::ResumeState state;
  state.step = 7;
  llm::train::save_resume(snapshot, &model, optimizer, state);

  ModelConfig other = config;
  other.n_layers = 3;
  Model different(other, 5);
  llm::train::AdamW other_optimizer(different.trainable_parameters(), adam);
  LLM_EXPECT_THROWS(
      llm::train::load_resume(snapshot, &different, &other_optimizer));

  // Тот же снимок в ту же модель читается и возвращает записанное.
  Model same(config, 999);
  llm::train::AdamW same_optimizer(same.trainable_parameters(), adam);
  const llm::train::ResumeState back =
      llm::train::load_resume(snapshot, &same, &same_optimizer);
  LLM_CHECK_EQ(back.step, static_cast<std::int64_t>(7));
  LLM_CHECK_EQ(same_optimizer.step_count(), static_cast<std::int64_t>(7));

  std::remove(snapshot.c_str());
}

LLM_TEST(Train, GradNormSumDoesNotDependOnThreadCount) {
  // Сумма квадратов делится на фиксированные части и складывается по их
  // номерам, поэтому не зависит от числа потоков. Сравнение идёт в двойной
  // точности и побитово — на самой сумме, а не на весах после шага.
  //
  // Через веса эту проверку сделать нельзя, и это выяснилось попыткой: разница
  // порядков сложения здесь порядка 1e-14 относительных, а норма и веса —
  // float с разрешением 6e-8, и при сужении типа расхождение пропадает. Тест
  // на весах проходил бы при любом порядке.
  //
  // Данные подобраны так, чтобы перегруппировка вообще была видна: единица и
  // дальше числа величиной 2^-27. Их квадраты по отдельности пропадают при
  // прибавлении к единице, а сложенные между собой — нет. На обычных случайных
  // числах сумма шестидесяти тысяч квадратов к порядку нечувствительна.
  const int64_t count = 1 << 16;
  llm::Tensor gradient = llm::Tensor::uninitialized(llm::Shape({count}));
  for (int64_t i = 0; i < count; ++i) {
    gradient.data()[i] = std::ldexp(1.0f, -27);
  }
  gradient.data()[0] = 1.0f;

  double serial = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    serial += static_cast<double>(gradient.data()[i]) * gradient.data()[i];
  }

  llm::set_parallel_width(1);
  const double one = llm::train::AdamW::sum_squares(gradient);
  llm::set_parallel_width(4);
  // Проверка самой проверки — но только наполовину, и это стоит сказать
  // прямо. Она стережёт одноядерную машину: там смена ширины ни к чему не
  // приведёт, и сравнивать будет нечего. А вот того, что ширину соблюдает сам
  // обход, она не проверяет — это свойство for_row_parts, и снаружи его видно
  // только счётом занятых потоков, чего тест делать не станет.
  //
  // Различать эти два случая приходится не зря: ручка ширины для этого пути
  // однажды не работала вовсе. for_row_parts звал parallel_for на все восемь
  // частей, не спрашивая, сколько потоков разрешено занимать, и при ширине
  // один обход занимал два потока.
  LLM_CHECK_MSG(llm::parallel_width() > 1 || llm::detect_core_count() < 2,
                "ширина осталась " << llm::parallel_width()
                                   << " при " << llm::detect_core_count()
                                   << " ядрах: сравнивать нечего");
  const double four = llm::train::AdamW::sum_squares(gradient);
  llm::set_parallel_width(0);

  // Печатать эти суммы обычным способом бесполезно: они отличаются в
  // четырнадцатом знаке, а поток по умолчанию показывает шесть. Разница
  // выводится явно.
  LLM_CHECK_MSG(one != serial,
                "данные не различают порядок сложения, проверка пуста");
  LLM_CHECK_MSG(one == four, "сумма квадратов зависит от числа потоков, "
                             "разница " << (one - four));
}

LLM_TEST(Train, OptimizerDoesNotDependOnThreadCount) {
  // Шаг оптимизатора делится по потокам, и обе суммы внутри него — норма
  // градиента и длина шага — складываются по фиксированным частям. Проверка
  // на то, что порядок действительно фиксирован: те же данные, разное число
  // потоков, побитово те же веса.
  //
  // Размер взят выше порога деления работы: с маленьким тензором многопоточная
  // ветка не включилась бы, и проверка оказалась бы пустой.
  //
  // Тонкость самой суммы проверяется отдельно, тестом выше: здесь всё сужается
  // до float, и расхождение порядков сложения до весов не доходит. Этот тест
  // про другое — что вся цепочка шага целиком, включая поэлементную часть,
  // даёт один и тот же ответ при любом числе потоков.
  const int64_t count = 1 << 16;
  llm::Rng rng(7);
  llm::Tensor start = llm::Tensor::uninitialized(llm::Shape({count}));
  llm::Tensor gradient = llm::Tensor::uninitialized(llm::Shape({count}));
  for (int64_t i = 0; i < count; ++i) {
    start.data()[i] = static_cast<float>(rng.normal());
    gradient.data()[i] = static_cast<float>(rng.normal());
  }

  const auto run = [&](int width) {
    llm::set_parallel_width(width);
    Var parameter = Var::leaf(start.clone(), true);
    std::vector<llm::nn::NamedParameter> handles;
    llm::nn::NamedParameter handle;
    handle.name = "w";
    handle.value = &parameter;
    handles.push_back(handle);

    llm::train::AdamW optimizer(handles, llm::train::AdamWConfig());
    float norm = 0.0f;
    for (int step = 0; step < 3; ++step) {
      parameter.node()->accumulate(gradient);
      norm = optimizer.clip_grad_norm(1.0f);
      optimizer.step(1e-3f);
      optimizer.zero_grad();
    }
    llm::set_parallel_width(0);
    return std::make_pair(parameter.value().clone(), norm);
  };

  const std::pair<llm::Tensor, float> one = run(1);
  const std::pair<llm::Tensor, float> four = run(4);

  LLM_CHECK_MSG(one.second == four.second,
                "норма градиента разошлась: " << one.second << " и "
                                              << four.second);
  for (int64_t i = 0; i < count; ++i) {
    LLM_CHECK_MSG(one.first.data()[i] == four.first.data()[i],
                  "вес " << i << " разошёлся: " << one.first.data()[i]
                         << " и " << four.first.data()[i]);
  }
}

LLM_TEST(Train, ScheduleHandlesWarmupEqualToTotal) {
  // Разогрев во всю длину прогона — не выдумка: так выглядит прерванный
  // прогон, у которого total_steps выставлен по уже пройденному. При этом
  // косинусная ветка делит на total - warmup, то есть на ноль, и спасает её
  // только то, что обе предыдущие ветки перехватывают все достижимые шаги.
  //
  // Честно про силу этой проверки: она закрепляет свойство, которое раньше не
  // утверждалось нигде, но построить правдоподобную поломку, которую ловила
  // бы ТОЛЬКО она, не удалось. Две пробовались. Перестановка веток местами
  // NaN наружу не выпускает — ранний возврат стоит раньше использования. А
  // замена ранней ветки на зажим progress до единицы даёт NaN, но сравнение
  // с NaN ложно, и зажим случайно возвращает ту же единицу. Проверка поэтому
  // сторожевая, а не доказательная.
  llm::train::ScheduleConfig config;
  config.max_learning_rate = 1.0f;
  config.min_ratio = 0.1f;
  config.total_steps = 10;
  config.warmup_steps = 10;

  for (int64_t step = 0; step < 10; ++step) {
    const float value = llm::train::learning_rate_at(config, step);
    LLM_CHECK_MSG(value == value, "скорость обучения на шаге " << step
                                                              << " оказалась NaN");
    LLM_CHECK_GT(value, 0.0f);
    LLM_CHECK_LE(value, 1.0f);
  }
  // Последний шаг разогрева — это и есть максимум.
  LLM_EXPECT_NEAR(llm::train::learning_rate_at(config, 9), 1.0, 1e-6);
  // За концом прогона — нижняя граница, и тоже без деления на ноль.
  const float after = llm::train::learning_rate_at(config, 10);
  LLM_CHECK_MSG(after == after, "за концом прогона скорость оказалась NaN");
  LLM_EXPECT_NEAR(after, 0.1, 1e-7);

  // И вырожденный случай наоборот: разогрева нет вовсе.
  config.warmup_steps = 0;
  const float first = llm::train::learning_rate_at(config, 0);
  LLM_CHECK_MSG(first == first, "без разогрева первый шаг оказался NaN");
  LLM_EXPECT_NEAR(first, 1.0, 1e-6);
}

LLM_TEST(Train, EvaluateWithNothingToMeasureReturnsZeroNotNan) {
  // «Считать нечего» — один случай, а не два. Пустая проверочная часть давала
  // нуль, а max_batches == 0 доходил до деления на число батчей и возвращал
  // NaN: цикл не выполнялся, а делить всё равно приходилось. Тренер сюда с
  // нулём не приходит, но функция объявлена в заголовке, и NaN в проверочных
  // потерях пошёл бы дальше — в выбор лучшего снимка.
  llm::nn::ModelConfig config = test_config();
  llm::nn::Model model(config, 7);

  std::vector<std::int32_t> tokens;
  for (int i = 0; i < 400; ++i) {
    tokens.push_back(static_cast<std::int32_t>(i % config.vocab_size));
  }
  const llm::data::TokenDataset dataset(tokens, 0.5);
  LLM_CHECK_GT(dataset.validation_batch_count(2, 8), static_cast<std::int64_t>(0));

  const float nothing = llm::train::evaluate(&model, dataset, 2, 8, 0, nullptr);
  LLM_CHECK_MSG(nothing == nothing, "проверочные потери оказались NaN");
  LLM_EXPECT_NEAR(nothing, 0.0, 0.0);

  // А при непустом запросе величина настоящая и конечная.
  const float real = llm::train::evaluate(&model, dataset, 2, 8, 2, nullptr);
  LLM_CHECK_MSG(real == real, "проверочные потери оказались NaN");
  LLM_CHECK_GT(real, 0.0f);
}
