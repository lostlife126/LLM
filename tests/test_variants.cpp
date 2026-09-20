// Архитектурные варианты: каждый обязан быть работающей моделью, а не просто
// компилироваться.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "autograd/nn.h"
#include "autograd/ops.h"
#include "core/random.h"
#include "gradcheck.h"
#include "nn/model.h"
#include "ops/loss.h"
#include "ops/nn.h"
#include "serialize/checkpoint.h"
#include "testing.h"
#include "train/trainer.h"

namespace {

using llm::autograd::Var;
using llm::nn::FfnKind;
using llm::nn::Model;
using llm::nn::ModelConfig;
using llm::nn::NormKind;
using llm::nn::PositionKind;
using llm::testing::random_tensor;
using llm::testing::weighted_sum;

ModelConfig base_config() {
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

// Все интересные варианты, каждый отличается от базового ровно одним
// решением: иначе замер отвечал бы на вопрос о комбинации, а не о решении.
std::vector<ModelConfig> all_variants() {
  std::vector<ModelConfig> variants;
  variants.push_back(base_config());

  ModelConfig layer_norm = base_config();
  layer_norm.norm = NormKind::kLayerNorm;
  variants.push_back(layer_norm);

  ModelConfig post = base_config();
  post.post_norm = true;
  variants.push_back(post);

  ModelConfig gelu = base_config();
  gelu.ffn = FfnKind::kGeluMlp;
  gelu.ffn_hidden = base_config().ffn_hidden_matching(FfnKind::kGeluMlp);
  variants.push_back(gelu);

  ModelConfig learned = base_config();
  learned.position = PositionKind::kLearned;
  variants.push_back(learned);

  ModelConfig none = base_config();
  none.position = PositionKind::kNone;
  variants.push_back(none);

  ModelConfig mha = base_config();
  mha.n_kv_heads = mha.n_heads;
  variants.push_back(mha);

  ModelConfig mqa = base_config();
  mqa.n_kv_heads = 1;
  variants.push_back(mqa);

  ModelConfig untied = base_config();
  untied.tie_embeddings = false;
  variants.push_back(untied);

  ModelConfig qk = base_config();
  qk.qk_norm = true;
  variants.push_back(qk);

  ModelConfig z = base_config();
  z.z_loss_coef = 1e-4f;
  variants.push_back(z);
  return variants;
}

// Батч с выучиваемой закономерностью: арифметическая прогрессия по модулю.
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

}  // namespace

LLM_TEST(Variants, ParameterCountMatchesFormula) {
  const std::vector<ModelConfig> variants = all_variants();
  for (std::size_t i = 0; i < variants.size(); ++i) {
    Model model(variants[i], 1);
    LLM_CHECK_MSG(model.parameter_count() == variants[i].parameter_count(),
                  "вариант " << variants[i].to_string() << ": по факту "
                             << model.parameter_count() << ", по формуле "
                             << variants[i].parameter_count());
  }
}

LLM_TEST(Variants, EveryVariantIsCausal) {
  // Каузальность — свойство не одной конфигурации, а всех. Пост-нормировка и
  // отсутствие позиционного кодирования затрагивают поток данных, и проверить
  // их отдельно стоит.
  const std::vector<ModelConfig> variants = all_variants();
  for (std::size_t v = 0; v < variants.size(); ++v) {
    Model model(variants[v], 11);
    const int64_t seq = 8;
    const std::vector<int32_t> ids = random_ids(seq, variants[v].vocab_size, 3);
    const Var original = model.forward(ids, 1, seq);

    for (int64_t changed = 1; changed < seq; ++changed) {
      std::vector<int32_t> modified = ids;
      modified[static_cast<std::size_t>(changed)] =
          (modified[static_cast<std::size_t>(changed)] + 1) %
          static_cast<int32_t>(variants[v].vocab_size);
      const Var altered = model.forward(modified, 1, seq);

      for (int64_t position = 0; position < changed; ++position) {
        for (int64_t token = 0; token < variants[v].vocab_size; ++token) {
          LLM_CHECK_MSG(original.value()(0, position, token) ==
                            altered.value()(0, position, token),
                        variants[v].to_string()
                            << ": изменение на позиции " << changed
                            << " повлияло на логит позиции " << position);
        }
      }
    }
  }
}

LLM_TEST(Variants, EveryVariantGetsGradientsEverywhere) {
  const std::vector<ModelConfig> variants = all_variants();
  for (std::size_t v = 0; v < variants.size(); ++v) {
    Model model(variants[v], 13);
    model.loss(random_ids(2 * 6, variants[v].vocab_size, 4), 2, 6).backward();

    const std::vector<llm::nn::NamedParameter> parameters = model.parameters();
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      LLM_CHECK_MSG(parameters[i].value->grad().defined(),
                    variants[v].to_string()
                        << ": параметр " << parameters[i].name
                        << " не получил градиента");
    }
  }
}

LLM_TEST(Variants, EveryVariantStartsNearUniform) {
  const std::vector<ModelConfig> variants = all_variants();
  for (std::size_t v = 0; v < variants.size(); ++v) {
    Model model(variants[v], 17);
    const Var loss =
        model.loss(random_ids(4 * 8, variants[v].vocab_size, 5), 4, 8);
    const double expected =
        std::log(static_cast<double>(variants[v].vocab_size));
    LLM_CHECK_MSG(std::fabs(*loss.value().data() - expected) < 0.2,
                  variants[v].to_string()
                      << ": начальные потери " << *loss.value().data()
                      << " вместо " << expected);
  }
}

LLM_TEST(Variants, EveryVariantOverfitsSingleBatch) {
  // Если вариант не способен запомнить один батч, дальше его измерять
  // бессмысленно.
  const std::vector<ModelConfig> variants = all_variants();
  for (std::size_t v = 0; v < variants.size(); ++v) {
    Model model(variants[v], 19);
    const std::vector<float> history = llm::train::overfit_batch(
        &model, random_ids(2 * 8, variants[v].vocab_size, 6), 2, 8, 200, 3e-3f);
    LLM_CHECK_MSG(history.back() < 0.2f, variants[v].to_string()
                                             << ": потери упали только с "
                                             << history.front() << " до "
                                             << history.back());
  }
}

LLM_TEST(Variants, ParameterMatchedFfnHasSameSize) {
  // Сравнение SwiGLU с обычным FFN должно быть о устройстве, а не о размере.
  const ModelConfig swiglu = base_config();
  ModelConfig gelu = base_config();
  gelu.ffn = FfnKind::kGeluMlp;
  gelu.ffn_hidden = swiglu.ffn_hidden_matching(FfnKind::kGeluMlp);

  // Ширина 24 у SwiGLU даёт 36 у двухматричного варианта — делится нацело,
  // значит числа параметров обязаны совпасть в точности.
  LLM_CHECK_EQ(gelu.ffn_hidden, static_cast<std::int64_t>(36));
  LLM_CHECK_MSG(swiglu.parameter_count() == gelu.parameter_count(),
                "параметры разошлись: " << swiglu.parameter_count() << " и "
                                        << gelu.parameter_count());

  // То же на настоящем пресете: 352 у SwiGLU против 528.
  const ModelConfig nano = ModelConfig::nano();
  ModelConfig nano_gelu = nano;
  nano_gelu.ffn = FfnKind::kGeluMlp;
  nano_gelu.ffn_hidden = nano.ffn_hidden_matching(FfnKind::kGeluMlp);
  LLM_CHECK_EQ(nano_gelu.ffn_hidden, static_cast<std::int64_t>(528));
  LLM_CHECK_EQ(nano_gelu.parameter_count(), nano.parameter_count());
}

LLM_TEST(Variants, LearnedPositionsActuallyDependOnPosition) {
  // Один и тот же токен на разных позициях обязан давать разные логиты —
  // иначе таблица позиций ни на что не влияет.
  ModelConfig config = base_config();
  config.position = PositionKind::kLearned;
  Model model(config, 23);

  const std::vector<int32_t> same(4, 5);
  const Var logits = model.forward(same, 1, 4);
  bool positions_differ = false;
  for (int64_t token = 0; token < config.vocab_size; ++token) {
    if (logits.value()(0, 0, token) != logits.value()(0, 3, token)) {
      positions_differ = true;
    }
  }
  LLM_CHECK(positions_differ);
}

LLM_TEST(Variants, WithoutPositionsOnlyMaskOrdersTokens) {
  // Без позиционного кодирования внимание симметрично, и единственный
  // источник порядка — каузальная маска. Следствие: первая позиция видит
  // только себя, поэтому её логиты зависят лишь от своего токена и совпадают
  // при любом продолжении.
  ModelConfig config = base_config();
  config.position = PositionKind::kNone;
  Model model(config, 29);

  std::vector<int32_t> first = random_ids(5, config.vocab_size, 7);
  std::vector<int32_t> second = random_ids(5, config.vocab_size, 8);
  second[0] = first[0];

  const Var a = model.forward(first, 1, 5);
  const Var b = model.forward(second, 1, 5);
  for (int64_t token = 0; token < config.vocab_size; ++token) {
    LLM_CHECK(a.value()(0, 0, token) == b.value()(0, 0, token));
  }
}

LLM_TEST(Variants, CheckpointRoundtripForEveryVariant) {
  const std::string path = "test_variant.llmw";
  const std::vector<ModelConfig> variants = all_variants();
  for (std::size_t v = 0; v < variants.size(); ++v) {
    Model saved(variants[v], 31);
    llm::serialize::save_checkpoint(path, &saved, 5);
    LLM_CHECK(llm::serialize::read_config(path) == variants[v]);

    Model loaded(variants[v], 999);
    LLM_CHECK_EQ(llm::serialize::load_checkpoint(path, &loaded),
                 static_cast<std::int64_t>(5));

    const std::vector<int32_t> ids = random_ids(6, variants[v].vocab_size, 9);
    const Var a = saved.forward(ids, 1, 6);
    const Var b = loaded.forward(ids, 1, 6);
    for (int64_t position = 0; position < 6; ++position) {
      for (int64_t token = 0; token < variants[v].vocab_size; ++token) {
        LLM_CHECK_MSG(
            a.value()(0, position, token) == b.value()(0, position, token),
            variants[v].to_string() << ": логиты разошлись после "
                                       "загрузки чекпоинта");
      }
    }
  }
  std::remove(path.c_str());
}

LLM_TEST(Variants, QkNormMakesScoresScaleInvariant) {
  // Определяющее свойство QK-нормы.
  //
  // Нормировка делит вектор на его собственную длину, поэтому умножение всей
  // матрицы проекции на константу не меняет нормированный результат вообще.
  // Значит внимание перестаёт зависеть от того, насколько велики веса
  // запросов — а именно их неограниченный рост и насыщает softmax по ходу
  // обучения.
  //
  // Проверяется так: увеличиваем веса запросов и ключей в сто раз и смотрим
  // на энтропию внимания. С нормировкой она обязана остаться прежней, без неё
  // — измениться сильно.
  //
  // Множитель большой намеренно. Оценка внимания — произведение запроса на
  // ключ, поэтому она растёт как квадрат множителя, а при инициализации с
  // разбросом 0.02 оценки настолько малы, что десятикратный рост softmax
  // ещё даже не замечает.
  for (int mode = 0; mode < 2; ++mode) {
    ModelConfig config = base_config();
    config.qk_norm = mode == 1;

    Model model(config, 61);
    const std::vector<int32_t> ids = random_ids(8, config.vocab_size, 21);

    llm::nn::ForwardStats before;
    model.loss(ids, 1, 8, &before);

    std::vector<llm::nn::NamedParameter> parameters = model.parameters();
    int scaled = 0;
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      if (parameters[i].name == "block.0.attention.query.weight" ||
          parameters[i].name == "block.0.attention.key.weight") {
        llm::Tensor& weight = parameters[i].value->value();
        for (int64_t e = 0; e < weight.numel(); ++e) {
          weight.data()[e] *= 100.0f;
        }
        ++scaled;
      }
    }
    LLM_CHECK_EQ(scaled, 2);

    llm::nn::ForwardStats after;
    model.loss(ids, 1, 8, &after);

    const double change =
        std::fabs(after.attention_entropy[0] - before.attention_entropy[0]);
    if (mode == 1) {
      // Инвариантность математически точная, но не побитовая: деление на
      // стократно выросший масштаб теряет младшие разряды float. Измеренное
      // расхождение — 8.7e-4, порог взят с запасом.
      LLM_CHECK_MSG(change < 5e-3,
                    "с QK-нормой энтропия внимания изменилась на "
                        << change << " при масштабировании весов");
    } else {
      // Без нормировки энтропия обрушивается с 0.99999 до 0.023: softmax
      // насытился полностью, и внимание из равномерного стало выбирать один
      // ключ. Ровно это и происходит по ходу обучения, когда веса растут.
      LLM_CHECK_MSG(change > 0.5,
                    "без QK-нормы энтропия внимания почти не изменилась ("
                        << change << ") — проверка ничего не проверяет");
    }
  }
}

LLM_TEST(Variants, QkNormAddsTwoVectorsPerLayer) {
  ModelConfig plain = base_config();
  ModelConfig normed = base_config();
  normed.qk_norm = true;

  Model plain_model(plain, 63);
  Model normed_model(normed, 63);
  // По одному вектору длины head_dim на запросы и ключи в каждом слое.
  LLM_CHECK_EQ(normed_model.parameter_count() - plain_model.parameter_count(),
               2 * plain.head_dim() * plain.n_layers);
  LLM_CHECK_EQ(normed_model.parameter_count(), normed.parameter_count());
}

LLM_TEST(Variants, ZLossKeepsLogitsNearZero) {
  // Softmax не меняется от добавления константы ко всем логитам строки,
  // поэтому без штрафа они свободно уезжают от нуля целой группой. Штраф
  // возвращает их обратно, почти не трогая разности — то есть предсказания.
  const std::vector<int32_t> ids = periodic_task(3, 0);

  float drift[2] = {0.0f, 0.0f};
  for (int mode = 0; mode < 2; ++mode) {
    ModelConfig config = base_config();
    config.z_loss_coef = mode == 1 ? 1e-2f : 0.0f;

    Model model(config, 67);
    llm::train::overfit_batch(&model, ids, 2, 8, 150, 3e-3f);

    // Логарифм суммы экспонент — это и есть то, что штрафуется.
    const Var logits = model.forward(ids, 2, 8);
    const llm::Tensor flat = logits.value().contiguous().reshape(
        llm::Shape({2 * 8, config.vocab_size}));
    drift[mode] = *llm::ops::z_loss(flat).data();
  }

  LLM_CHECK_MSG(drift[1] < drift[0],
                "штраф не уменьшил дрейф логитов: без него "
                    << drift[0] << ", с ним " << drift[1]);
}

LLM_TEST(Variants, ZLossDoesNotRuinPredictions) {
  // Штраф обязан двигать логиты как группу, а не менять их разности: иначе он
  // мешал бы обучению, а не помогал.
  const std::vector<int32_t> ids = periodic_task(3, 0);
  ModelConfig config = base_config();
  config.z_loss_coef = 1e-3f;

  Model model(config, 71);
  const std::vector<float> history =
      llm::train::overfit_batch(&model, ids, 2, 8, 200, 3e-3f);
  LLM_CHECK_MSG(history.back() < 0.2f, "со штрафом модель перестала обучаться: "
                                           << history.front() << " -> "
                                           << history.back());
}

LLM_TEST(Variants, GradZLoss) {
  const auto fn = [](const std::vector<Var>& v) {
    return llm::autograd::z_loss(v[0]);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 5}), 90)}));
}

LLM_TEST(Variants, GradLayerNorm) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::layer_norm(v[0], v[1], v[2], 1e-5f), 80);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 6}), 81),
                                    random_tensor(llm::Shape({6}), 82),
                                    random_tensor(llm::Shape({6}), 83)}));
}

LLM_TEST(Variants, LayerNormCentersAndScales) {
  // После LayerNorm с единичным весом и нулевым сдвигом каждая строка обязана
  // иметь нулевое среднее и единичную дисперсию — это и отличает её от
  // RMSNorm, которая среднее не трогает.
  const llm::Tensor input = random_tensor(llm::Shape({4, 16}), 84, -3.0f, 5.0f);
  const llm::Tensor weight = llm::Tensor::full(llm::Shape({16}), 1.0f);
  const llm::Tensor bias = llm::Tensor::zeros(llm::Shape({16}));
  const llm::Tensor result = llm::ops::layer_norm(input, weight, bias, 1e-6f);

  for (int64_t row = 0; row < 4; ++row) {
    double sum = 0.0;
    double sum_squares = 0.0;
    for (int64_t i = 0; i < 16; ++i) {
      sum += result(row, i);
      sum_squares += static_cast<double>(result(row, i)) * result(row, i);
    }
    LLM_EXPECT_NEAR(sum / 16.0, 0.0, 1e-6);
    LLM_EXPECT_NEAR(sum_squares / 16.0, 1.0, 1e-5);
  }
}
