// Дропаут.
//
// Ошибки здесь тихие. Забытое деление на вероятность сохранения даёт модель,
// которая обучается и работает, но систематически занижает активации.
// Сгенерированная заново маска в обратном проходе даёт градиент, не
// соответствующий значению, — обучение при этом не падает, а просто идёт
// хуже. Поэтому проверяется не «работает ли», а каждое из этих свойств
// по отдельности.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "autograd/ops.h"
#include "core/random.h"
#include "nn/dropout.h"
#include "nn/model.h"
#include "ops/dropout.h"
#include "serialize/checkpoint.h"
#include "testing.h"
#include "train/trainer.h"

using llm::autograd::Var;
using llm::nn::ModelConfig;

namespace {

ModelConfig test_config() {
  ModelConfig config;
  config.vocab_size = 32;
  config.d_model = 32;
  config.n_layers = 2;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.max_seq_len = 16;
  config.ffn_hidden = 64;
  return config;
}

std::vector<int32_t> ids_of(int64_t count, int64_t vocab, uint64_t seed) {
  llm::Rng rng(seed);
  std::vector<int32_t> out;
  for (int64_t i = 0; i < count; ++i) {
    out.push_back(
        static_cast<int32_t>(rng.index(static_cast<std::uint64_t>(vocab))));
  }
  return out;
}

}  // namespace

LLM_TEST(Dropout, OffByDefault) {
  // Самое важное свойство: вне обучения дропаута нет. Забытым включённым он
  // испортил бы проверочные потери и сделал бы генерацию невоспроизводимой.
  LLM_CHECK(!llm::nn::training_mode());

  const llm::Tensor input = llm::Tensor::full(llm::Shape({4, 8}), 2.0f);
  const Var same = llm::nn::dropout(Var::constant(input), 0.5f);
  for (int64_t i = 0; i < input.numel(); ++i) {
    LLM_CHECK_EQ(same.value().data()[i], 2.0f);
  }
}

LLM_TEST(Dropout, ScopeEnablesAndRestores) {
  LLM_CHECK(!llm::nn::training_mode());
  {
    const llm::nn::TrainingScope scope(1);
    LLM_CHECK(llm::nn::training_mode());
    {
      const llm::nn::TrainingScope inner(2);
      LLM_CHECK(llm::nn::training_mode());
    }
    LLM_CHECK(llm::nn::training_mode());
  }
  LLM_CHECK(!llm::nn::training_mode());
}

LLM_TEST(Dropout, ZeroProbabilityIsIdentity) {
  const llm::nn::TrainingScope scope(7);
  const llm::Tensor input = llm::Tensor::full(llm::Shape({3, 5}), 1.5f);
  const Var same = llm::nn::dropout(Var::constant(input), 0.0f);
  for (int64_t i = 0; i < input.numel(); ++i) {
    LLM_CHECK_EQ(same.value().data()[i], 1.5f);
  }
}

LLM_TEST(Dropout, KeptValuesAreScaledUp) {
  // «Обратный» дропаут: делим на вероятность сохранения сразу при обучении.
  // Тогда инференс — та же формула без дропаута, и переключать нечего.
  // Проверяется буквально: уцелевшее значение равно исходному, делённому на
  // 1 - p, а зануленное равно нулю. Третьего не дано.
  const llm::nn::TrainingScope scope(11);
  const float probability = 0.25f;
  const llm::Tensor input = llm::Tensor::full(llm::Shape({40, 40}), 4.0f);
  const Var out = llm::nn::dropout(Var::constant(input), probability);

  const float expected = 4.0f / (1.0f - probability);
  int64_t zeros = 0;
  for (int64_t i = 0; i < input.numel(); ++i) {
    const float value = out.value().data()[i];
    if (value == 0.0f) {
      ++zeros;
    } else {
      LLM_EXPECT_NEAR(value, expected, 1e-5);
    }
  }
  // Доля занулённых обязана быть близка к заданной вероятности. Границы
  // широкие: проверяется масштаб, а не точное совпадение со средним.
  const double fraction =
      static_cast<double>(zeros) / static_cast<double>(input.numel());
  LLM_CHECK_MSG(fraction > 0.20 && fraction < 0.30,
                "занулено " << fraction << " вместо примерно 0.25");
}

LLM_TEST(Dropout, MeanIsPreserved) {
  // Из-за деления на вероятность сохранения среднее значение не съезжает —
  // ровно в этом смысл «обратной» записи. Если бы деления не было, среднее
  // упало бы в 1 - p раз, и модель обучалась бы на систематически заниженных
  // активациях.
  const llm::nn::TrainingScope scope(13);
  const llm::Tensor input = llm::Tensor::full(llm::Shape({200, 200}), 3.0f);
  const Var out = llm::nn::dropout(Var::constant(input), 0.3f);

  double total = 0.0;
  for (int64_t i = 0; i < input.numel(); ++i) {
    total += out.value().data()[i];
  }
  const double mean = total / static_cast<double>(input.numel());
  LLM_EXPECT_NEAR(mean, 3.0, 0.03);
}

LLM_TEST(Dropout, BackwardUsesTheSameMask) {
  // Маска обязана быть той же самой в прямом и обратном проходе. Если её
  // сгенерировать заново, градиент перестанет соответствовать значению:
  // обучение не упадёт, а просто пойдёт хуже — и заметить это будет нечем.
  //
  // Проверка прямая: там, где значение занулено, градиент обязан быть нулём,
  // а где не занулено — равен масштабу.
  const llm::nn::TrainingScope scope(17);
  const float probability = 0.5f;
  Var input = Var::leaf(llm::Tensor::full(llm::Shape({8, 8}), 1.0f), true);
  const Var out = llm::nn::dropout(input, probability);
  llm::autograd::sum_all(out).backward();

  const float scale = 1.0f / (1.0f - probability);
  for (int64_t i = 0; i < input.numel(); ++i) {
    const float value = out.value().data()[i];
    const float grad = input.grad().data()[i];
    if (value == 0.0f) {
      LLM_CHECK_MSG(grad == 0.0f,
                    "градиент " << grad << " у занулённого элемента " << i);
    } else {
      LLM_EXPECT_NEAR(grad, scale, 1e-6);
    }
  }
}

LLM_TEST(Dropout, SameSeedGivesSameMask) {
  // Воспроизводимость прогона: два прохода с одним зерном обязаны совпасть
  // побитово, с разными — разойтись.
  const llm::Tensor input = llm::Tensor::full(llm::Shape({64}), 1.0f);
  std::vector<float> first;
  std::vector<float> again;
  std::vector<float> other;

  {
    const llm::nn::TrainingScope scope(42);
    const Var out = llm::nn::dropout(Var::constant(input), 0.5f);
    first.assign(out.value().data(), out.value().data() + input.numel());
  }
  {
    const llm::nn::TrainingScope scope(42);
    const Var out = llm::nn::dropout(Var::constant(input), 0.5f);
    again.assign(out.value().data(), out.value().data() + input.numel());
  }
  {
    const llm::nn::TrainingScope scope(43);
    const Var out = llm::nn::dropout(Var::constant(input), 0.5f);
    other.assign(out.value().data(), out.value().data() + input.numel());
  }

  LLM_CHECK(first == again);
  LLM_CHECK_MSG(first != other, "разные зёрна дали одинаковую маску");
}

LLM_TEST(Dropout, ModelOutputChangesOnlyInTrainingMode) {
  ModelConfig config = test_config();
  config.dropout = 0.3f;
  llm::nn::Model model(config, 5);
  const std::vector<int32_t> ids = ids_of(8, config.vocab_size, 1);

  // Вне обучения два прохода обязаны совпасть побитово.
  const llm::Tensor a = model.forward(ids, 1, 8).value().contiguous();
  const llm::Tensor b = model.forward(ids, 1, 8).value().contiguous();
  for (int64_t i = 0; i < a.numel(); ++i) {
    LLM_CHECK_MSG(a.data()[i] == b.data()[i],
                  "модель недетерминирована вне обучения");
  }

  // В обучении — обязаны разойтись, иначе дропаут никуда не подключён.
  llm::Tensor c;
  {
    const llm::nn::TrainingScope scope(1);
    c = model.forward(ids, 1, 8).value().contiguous();
  }
  bool differs = false;
  for (int64_t i = 0; i < a.numel(); ++i) {
    if (a.data()[i] != c.data()[i]) {
      differs = true;
    }
  }
  LLM_CHECK_MSG(differs, "дропаут включён, а логиты те же — он не применяется");
}

LLM_TEST(Dropout, ZeroDropoutModelIsUnaffectedByTrainingMode) {
  // Модель без дропаута обязана вести себя одинаково в любом режиме: иначе
  // все прежние замеры поехали бы.
  llm::nn::Model model(test_config(), 5);
  const std::vector<int32_t> ids = ids_of(8, 32, 2);

  const llm::Tensor a = model.forward(ids, 1, 8).value().contiguous();
  llm::Tensor b;
  {
    const llm::nn::TrainingScope scope(1);
    b = model.forward(ids, 1, 8).value().contiguous();
  }
  for (int64_t i = 0; i < a.numel(); ++i) {
    LLM_CHECK_EQ(a.data()[i], b.data()[i]);
  }
}

LLM_TEST(Dropout, ModelStillLearnsWithDropout) {
  // Дропаут обязан мешать переобучению, а не обучению. На маленькой задаче
  // потери всё равно должны заметно упасть.
  ModelConfig config = test_config();
  config.dropout = 0.1f;
  llm::nn::Model model(config, 9);

  const std::vector<int32_t> ids = ids_of(2 * 8, config.vocab_size, 3);
  const std::vector<float> history =
      llm::train::overfit_batch(&model, ids, 2, 8, 80, 3e-3f);
  LLM_CHECK_MSG(history.back() < history.front() * 0.7f,
                "с дропаутом обучение встало: " << history.front() << " -> "
                                                << history.back());
}

LLM_TEST(Dropout, RejectsImpossibleProbability) {
  ModelConfig config = test_config();
  config.dropout = 1.0f;
  LLM_EXPECT_THROWS(config.validate());

  config.dropout = -0.1f;
  LLM_EXPECT_THROWS(config.validate());

  llm::Rng rng(1);
  LLM_EXPECT_THROWS(llm::ops::dropout_mask(llm::Shape({4}), 1.0f, &rng));
}

LLM_TEST(Dropout, SurvivesCheckpointRoundTrip) {
  // Дропаут — часть описания модели, и чекпоинт обязан его сохранять: иначе
  // возобновлённое обучение пошло бы без него и молча дало бы другой прогон.
  ModelConfig config = test_config();
  config.dropout = 0.15f;
  llm::nn::Model model(config, 1);

  const std::string path = "test_dropout_checkpoint.llmw";
  llm::serialize::save_checkpoint(path, &model, 5);
  const ModelConfig read = llm::serialize::read_config(path);
  LLM_EXPECT_NEAR(read.dropout, 0.15, 1e-7);
  LLM_CHECK(read == config);
  std::remove(path.c_str());
}
