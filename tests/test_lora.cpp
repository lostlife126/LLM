// Дообучение низкоранговыми адаптерами.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "core/random.h"
#include "nn/lora.h"
#include "nn/model.h"
#include "serialize/checkpoint.h"
#include "testing.h"
#include "train/trainer.h"

namespace {

using llm::autograd::Var;
using llm::nn::LoraConfig;
using llm::nn::Model;
using llm::nn::ModelConfig;

ModelConfig test_config() {
  ModelConfig config;
  config.vocab_size = 32;
  config.d_model = 32;
  config.n_layers = 2;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.max_seq_len = 12;
  config.ffn_hidden = 48;
  config.validate();
  return config;
}

// Батч с выучиваемой закономерностью: два элемента по восемь позиций,
// последовательность задана арифметической прогрессией по модулю.
std::vector<int32_t> periodic_task(int period, int shift) {
  std::vector<int32_t> ids;
  for (int item = 0; item < 2; ++item) {
    for (int i = 0; i < 8; ++i) {
      ids.push_back(static_cast<int32_t>((i * period + shift + item) % 11));
    }
  }
  return ids;
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

void expect_identical_logits(const Var& a, const Var& b, int64_t seq,
                             int64_t vocab, const char* what) {
  for (int64_t position = 0; position < seq; ++position) {
    for (int64_t token = 0; token < vocab; ++token) {
      LLM_CHECK_MSG(
          a.value()(0, position, token) == b.value()(0, position, token),
          what << ": логиты разошлись на позиции " << position << ", токен "
               << token);
    }
  }
}

}  // namespace

LLM_TEST(Lora, ChangesNothingAtInitialisation) {
  // Определяющее свойство адаптера.
  //
  // Матрица B начинается с нуля, поэтому вклад адаптера в точности нулевой, и
  // модель в начале дообучения совпадает с исходной побитово. Если бы обе
  // матрицы инициализировались случайными, дообучение стартовало бы со
  // сломанной модели — и первые шаги уходили бы на починку того, что было
  // целым.
  const ModelConfig config = test_config();
  Model model(config, 7);
  const std::vector<int32_t> ids = random_ids(6, config.vocab_size, 1);

  const Var before = model.forward(ids, 1, 6);

  LoraConfig lora;
  lora.rank = 4;
  model.enable_lora(lora);

  const Var after = model.forward(ids, 1, 6);
  expect_identical_logits(before, after, 6, config.vocab_size,
                          "включение LoRA");
}

LLM_TEST(Lora, OnlyAdaptersAreTrainable) {
  const ModelConfig config = test_config();
  Model model(config, 11);

  const int64_t before = model.trainable_parameter_count();
  LLM_CHECK_EQ(before, model.parameter_count());

  LoraConfig lora;
  lora.rank = 4;
  model.enable_lora(lora);

  const std::vector<llm::nn::NamedParameter> trainable =
      model.trainable_parameters();
  LLM_CHECK_MSG(!trainable.empty(), "после включения LoRA нечего обучать");
  for (std::size_t i = 0; i < trainable.size(); ++i) {
    const std::string& name = trainable[i].name;
    const bool is_adapter =
        name.size() >= 6 && name.compare(name.size() - 6, 6, "lora_a") == 0;
    const bool is_adapter_b =
        name.size() >= 6 && name.compare(name.size() - 6, 6, "lora_b") == 0;
    LLM_CHECK_MSG(is_adapter || is_adapter_b,
                  "обучаемым остался не адаптер: " << name);
  }
}

LLM_TEST(Lora, TrainableShareIsSmall) {
  const ModelConfig config = ModelConfig::nano();
  Model model(config, 13);

  LoraConfig lora;
  lora.rank = 8;
  model.enable_lora(lora);

  const double share = static_cast<double>(model.trainable_parameter_count()) /
                       static_cast<double>(model.parameter_count());
  // Адаптеры на запросах и значениях при ранге 8 — доли процента от модели.
  LLM_CHECK_MSG(share < 0.03, "обучаемых " << model.trainable_parameter_count()
                                           << " из " << model.parameter_count()
                                           << ", то есть " << share * 100.0
                                           << "%");
  LLM_CHECK_GT(model.trainable_parameter_count(), static_cast<std::int64_t>(0));
}

LLM_TEST(Lora, GradientsReachAdaptersAndNotBaseWeights) {
  const ModelConfig config = test_config();
  Model model(config, 17);
  LoraConfig lora;
  lora.rank = 4;
  model.enable_lora(lora);

  model.loss(random_ids(2 * 6, config.vocab_size, 2), 2, 6).backward();

  const std::vector<llm::nn::NamedParameter> all = model.parameters();
  int adapters_with_grad = 0;
  for (std::size_t i = 0; i < all.size(); ++i) {
    if (all[i].value->requires_grad()) {
      LLM_CHECK_MSG(all[i].value->grad().defined(),
                    "адаптер " << all[i].name << " не получил градиента");
      ++adapters_with_grad;
    }
  }
  LLM_CHECK_GT(adapters_with_grad, 0);
  // Базовые веса перестали быть узлами графа: спрашивать у них градиент —
  // ошибка, а не пустой ответ.
  LLM_EXPECT_THROWS(all[0].value->grad());
}

LLM_TEST(Lora, MergeIsExactlyEquivalent) {
  // Вплавление адаптера в веса не должно менять поведение модели: иначе
  // выложенная модель отличалась бы от той, что обучали.
  const ModelConfig config = test_config();
  Model model(config, 19);
  LoraConfig lora;
  lora.rank = 4;
  model.enable_lora(lora);

  // Дообучаем, чтобы адаптер стал ненулевым.
  const std::vector<int32_t> ids = random_ids(2 * 8, config.vocab_size, 3);
  llm::train::overfit_batch(&model, ids, 2, 8, 30, 3e-3f);

  const std::vector<int32_t> probe = random_ids(6, config.vocab_size, 4);
  const Var with_adapter = model.forward(probe, 1, 6);

  model.merge_lora();
  const Var merged = model.forward(probe, 1, 6);

  // Здесь допуск, а не точное равенство: слияние заменяет два умножения на
  // узкие матрицы одним сложением в вес, и порядок арифметики меняется.
  for (int64_t position = 0; position < 6; ++position) {
    for (int64_t token = 0; token < config.vocab_size; ++token) {
      LLM_EXPECT_NEAR(merged.value()(0, position, token),
                      with_adapter.value()(0, position, token), 2e-4);
    }
  }
  // После слияния адаптеров в наборе параметров быть не должно.
  const std::vector<llm::nn::NamedParameter> all = model.parameters();
  for (std::size_t i = 0; i < all.size(); ++i) {
    LLM_CHECK(all[i].name.find("lora") == std::string::npos);
  }
}

LLM_TEST(Lora, MergeOfUntrainedAdapterChangesNothing) {
  const ModelConfig config = test_config();
  Model model(config, 23);
  const std::vector<int32_t> ids = random_ids(6, config.vocab_size, 5);
  const Var before = model.forward(ids, 1, 6);

  LoraConfig lora;
  lora.rank = 4;
  model.enable_lora(lora);
  model.merge_lora();

  const Var after = model.forward(ids, 1, 6);
  expect_identical_logits(before, after, 6, config.vocab_size,
                          "слияние нулевого адаптера");
}

LLM_TEST(Lora, AdaptsPretrainedModelToNewTask) {
  // Настоящий сценарий LoRA: модель уже чему-то обучена, и её нужно
  // приспособить к другой задаче.
  //
  // Здесь это видно в чистом виде: 896 обучаемых параметров против 16544 у
  // полного дообучения, а результат на новой задаче тот же.
  const ModelConfig config = test_config();

  Model model(config, 29);
  llm::train::overfit_batch(&model, periodic_task(3, 0), 2, 8, 400, 3e-3f);

  const std::vector<int32_t> new_task = periodic_task(5, 2);
  const float before = *model.loss(new_task, 2, 8).value().data();
  LLM_CHECK_MSG(
      before > 5.0f,
      "новая задача оказалась слишком похожей на старую: потери " << before);

  LoraConfig lora;
  lora.rank = 4;
  model.enable_lora(lora);
  const int64_t trainable = model.trainable_parameter_count();

  const std::vector<float> history =
      llm::train::overfit_batch(&model, new_task, 2, 8, 200, 5e-3f);

  LLM_CHECK_MSG(history.back() < 0.05f,
                "адаптеры (" << trainable << " параметров) не справились: "
                             << history.front() << " -> " << history.back());
  LLM_CHECK_LT(trainable, model.parameter_count() / 10);
}

LLM_TEST(Lora, CannotReplaceTrainingFromScratch) {
  // Обратная сторона: адаптеры приспосабливают выученное, но не создают его.
  //
  // На случайно инициализированной модели всё замороженное — это шум, а не
  // признаки, и адаптерам не на что опереться. Потери упираются в плато, тогда
  // как полное обучение той же задачи доходит почти до нуля.
  //
  // Это не дефект LoRA, а граница её применимости, и знать её стоит: попытка
  // обучить модель адаптерами с нуля выглядит как «обучение идёт, но плохо»,
  // а не как явная ошибка.
  const ModelConfig config = test_config();
  const std::vector<int32_t> task = periodic_task(3, 0);

  Model full(config, 29);
  const std::vector<float> full_history =
      llm::train::overfit_batch(&full, task, 2, 8, 200, 3e-3f);

  Model adapted(config, 29);
  LoraConfig lora;
  lora.rank = 4;
  adapted.enable_lora(lora);
  const std::vector<float> lora_history =
      llm::train::overfit_batch(&adapted, task, 2, 8, 200, 5e-3f);

  LLM_CHECK_MSG(full_history.back() < 0.1f,
                "полное обучение не справилось: " << full_history.back());
  LLM_CHECK_MSG(lora_history.back() > 1.0f,
                "адаптеры на случайной модели неожиданно справились: "
                    << lora_history.back()
                    << " — стоит перепроверить, что заморозка работает");
}

LLM_TEST(Lora, RejectsUselessRank) {
  const ModelConfig config = test_config();
  Model model(config, 31);
  LoraConfig lora;
  // Ранг не меньше размеров матрицы означает, что адаптер не уже исходной
  // матрицы, и вся идея теряет смысл.
  lora.rank = config.d_model;
  LLM_EXPECT_THROWS(model.enable_lora(lora));
}

LLM_TEST(Lora, ApplyingToFfnAddsMoreParameters) {
  const ModelConfig config = ModelConfig::nano();

  Model attention_only(config, 37);
  LoraConfig lora;
  lora.rank = 8;
  attention_only.enable_lora(lora);

  Model with_ffn(config, 37);
  LoraConfig wider = lora;
  wider.ffn = true;
  with_ffn.enable_lora(wider);

  LLM_CHECK_GT(with_ffn.trainable_parameter_count(),
               attention_only.trainable_parameter_count());
}

LLM_TEST(Lora, CheckpointKeepsAdapters) {
  const std::string path = "test_lora.llmw";
  const ModelConfig config = test_config();

  Model saved(config, 41);
  LoraConfig lora;
  lora.rank = 4;
  saved.enable_lora(lora);
  llm::train::overfit_batch(&saved, random_ids(2 * 8, config.vocab_size, 7), 2,
                            8, 20, 3e-3f);
  llm::serialize::save_checkpoint(path, &saved, 9);

  Model loaded(config, 999);
  loaded.enable_lora(lora);
  LLM_CHECK_EQ(llm::serialize::load_checkpoint(path, &loaded),
               static_cast<std::int64_t>(9));

  const std::vector<int32_t> probe = random_ids(6, config.vocab_size, 8);
  expect_identical_logits(saved.forward(probe, 1, 6),
                          loaded.forward(probe, 1, 6), 6, config.vocab_size,
                          "чекпоинт с адаптерами");
  std::remove(path.c_str());
}

LLM_TEST(Lora, BaseCheckpointDoesNotFitLoraModel) {
  // Базовый чекпоинт не содержит адаптеров, и загрузка в модель с ними должна
  // отказать, а не молча оставить адаптеры случайными.
  const std::string path = "test_base.llmw";
  const ModelConfig config = test_config();

  Model base(config, 43);
  llm::serialize::save_checkpoint(path, &base, 0);

  Model adapted(config, 43);
  LoraConfig lora;
  lora.rank = 4;
  adapted.enable_lora(lora);
  LLM_EXPECT_THROWS(llm::serialize::load_checkpoint(path, &adapted));
  std::remove(path.c_str());
}
