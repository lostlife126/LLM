// Сборка модели: конфигурация, формы, каузальность, градиенты.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/fp16.h"
#include "core/random.h"
#include "nn/config.h"
#include "nn/model.h"
#include "testing.h"

namespace {

using llm::autograd::Var;
using llm::nn::Model;
using llm::nn::ModelConfig;

// Самая маленькая осмысленная конфигурация: тесты должны идти быстро, а все
// архитектурные развилки — присутствовать. Головы ключей вдвое реже голов
// запросов, поэтому GQA задействован.
ModelConfig test_config() {
  ModelConfig config;
  config.vocab_size = 32;
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

}  // namespace

LLM_TEST(Model, PresetsAreConsistent) {
  const ModelConfig presets[] = {ModelConfig::nano(), ModelConfig::tiny(),
                                 ModelConfig::small()};
  for (std::size_t i = 0; i < 3; ++i) {
    presets[i].validate();
    LLM_CHECK_EQ(presets[i].d_model % presets[i].n_heads,
                 static_cast<int64_t>(0));
    LLM_CHECK_EQ(presets[i].head_dim() % 2, static_cast<int64_t>(0));
  }
}

LLM_TEST(Model, ValidateRejectsBadConfig) {
  ModelConfig config = test_config();
  config.n_heads = 5;  // d_model на него не делится
  LLM_EXPECT_THROWS(config.validate());

  config = test_config();
  config.n_kv_heads = 3;  // число голов запросов на него не делится
  LLM_EXPECT_THROWS(config.validate());

  config = test_config();
  config.d_model = 4;
  config.n_heads = 4;  // размерность головы получилась бы нечётной для RoPE
  LLM_EXPECT_THROWS(config.validate());
}

LLM_TEST(Model, ParameterCountMatchesFormula) {
  // Формула в конфигурации и фактический набор параметров обязаны сходиться:
  // расхождение означало бы, что какой-то вес не попадает в оптимизатор.
  const ModelConfig presets[] = {test_config(), ModelConfig::nano(),
                                 ModelConfig::tiny()};
  for (std::size_t i = 0; i < 3; ++i) {
    Model model(presets[i], 1);
    LLM_CHECK_MSG(model.parameter_count() == presets[i].parameter_count(),
                  "пресет " << i << ": по факту " << model.parameter_count()
                            << ", по формуле " << presets[i].parameter_count());
  }
}

LLM_TEST(Model, NanoPresetSizeIsAsDocumented) {
  // Цифра из README: пресет nano — около 0.87 миллиона параметров.
  const ModelConfig config = ModelConfig::nano();
  LLM_CHECK_GT(config.parameter_count(), static_cast<int64_t>(800000));
  LLM_CHECK_LT(config.parameter_count(), static_cast<int64_t>(950000));
}

LLM_TEST(Model, ForwardShapes) {
  const ModelConfig config = test_config();
  Model model(config, 7);
  const int64_t batch = 3;
  const int64_t seq = 5;

  const Var logits =
      model.forward(random_ids(batch * seq, config.vocab_size, 1), batch, seq);
  LLM_CHECK(logits.shape() == llm::Shape({batch, seq, config.vocab_size}));
}

LLM_TEST(Model, ForwardRejectsWrongTokenCount) {
  Model model(test_config(), 7);
  LLM_EXPECT_THROWS(model.forward(std::vector<int32_t>(7, 0), 2, 5));
}

LLM_TEST(Model, ForwardRejectsTooLongSequence) {
  const ModelConfig config = test_config();
  Model model(config, 7);
  const int64_t too_long = config.max_seq_len + 1;
  LLM_EXPECT_THROWS(
      model.forward(random_ids(too_long, config.vocab_size, 2), 1, too_long));
}

LLM_TEST(Model, IsCausal) {
  // Главный тест модели.
  //
  // Изменение токена на позиции t не имеет права повлиять на логиты позиций
  // раньше t. Одной проверкой покрывается сразу многое: правильность маски,
  // правильность осей в RoPE, правильность перестановок голов и то, что
  // умножение запросов на ключи идёт в нужную сторону. Почти любая ошибка в
  // форме внимания ломает именно это свойство.
  //
  // Сравнение точное, а не с допуском. Закрытые позиции получают ровно нуль
  // после экспоненты минус бесконечности, а умножение на нуль даёт нуль
  // независимо от того, что изменилось: результат обязан совпасть побитово.
  const ModelConfig config = test_config();
  Model model(config, 11);
  const int64_t seq = 8;

  std::vector<int32_t> ids = random_ids(seq, config.vocab_size, 3);
  const Var original = model.forward(ids, 1, seq);

  for (int64_t changed = 1; changed < seq; ++changed) {
    std::vector<int32_t> modified = ids;
    modified[static_cast<std::size_t>(changed)] =
        (modified[static_cast<std::size_t>(changed)] + 1) %
        static_cast<int32_t>(config.vocab_size);
    const Var altered = model.forward(modified, 1, seq);

    for (int64_t position = 0; position < changed; ++position) {
      for (int64_t token = 0; token < config.vocab_size; ++token) {
        LLM_CHECK_MSG(original.value()(0, position, token) ==
                          altered.value()(0, position, token),
                      "изменение токена на позиции "
                          << changed << " повлияло на логит позиции "
                          << position << ", токен " << token);
      }
    }

    // И наоборот: на саму изменённую позицию влияние обязано быть.
    bool changed_something = false;
    for (int64_t token = 0; token < config.vocab_size; ++token) {
      if (original.value()(0, changed, token) !=
          altered.value()(0, changed, token)) {
        changed_something = true;
      }
    }
    LLM_CHECK_MSG(changed_something, "изменение токена на позиции "
                                         << changed
                                         << " ни на что не повлияло");
  }
}

LLM_TEST(Model, BatchItemsAreIndependent) {
  // Элементы батча не должны видеть друг друга. Ошибка в свёртке осей батча и
  // голов проявилась бы именно так.
  const ModelConfig config = test_config();
  Model model(config, 13);
  const int64_t seq = 6;

  const std::vector<int32_t> first = random_ids(seq, config.vocab_size, 4);
  const std::vector<int32_t> second = random_ids(seq, config.vocab_size, 5);

  std::vector<int32_t> together = first;
  together.insert(together.end(), second.begin(), second.end());

  const Var batched = model.forward(together, 2, seq);
  const Var alone = model.forward(first, 1, seq);

  for (int64_t position = 0; position < seq; ++position) {
    for (int64_t token = 0; token < config.vocab_size; ++token) {
      LLM_CHECK_MSG(batched.value()(0, position, token) ==
                        alone.value()(0, position, token),
                    "элемент батча зависит от соседа на позиции " << position);
    }
  }
}

LLM_TEST(Model, RepeatKvMapsQueryHeadToItsKvHead) {
  // Раскладка голов: голова запроса h обязана пользоваться головой ключей
  // h / repeats. Заполняем каждую голову ключей её номером и смотрим, что
  // получилось после размножения.
  const int64_t batch = 1;
  const int64_t kv_heads = 3;
  const int64_t repeats = 2;
  const int64_t seq = 2;
  const int64_t head_dim = 2;

  llm::Tensor source =
      llm::Tensor::zeros(llm::Shape({batch, kv_heads, seq, head_dim}));
  for (int64_t head = 0; head < kv_heads; ++head) {
    for (int64_t step = 0; step < seq; ++step) {
      for (int64_t channel = 0; channel < head_dim; ++channel) {
        source(0, head, step, channel) = static_cast<float>(head);
      }
    }
  }

  const Var repeated = llm::nn::repeat_kv(Var::constant(source), batch,
                                          kv_heads, repeats, seq, head_dim);
  LLM_CHECK(repeated.shape() ==
            llm::Shape({batch, kv_heads * repeats, seq, head_dim}));

  for (int64_t head = 0; head < kv_heads * repeats; ++head) {
    LLM_EXPECT_NEAR(repeated.value()(0, head, 0, 0),
                    static_cast<double>(head / repeats), 1e-6);
  }
}

LLM_TEST(Model, InitialLossIsNearUniform) {
  // Необученная модель ничего не знает и обязана предсказывать почти
  // равномерно, то есть потери около log(vocab_size). Заметное отклонение
  // означало бы, что инициализация выбивает логиты из рабочего диапазона
  // ещё до всякого обучения.
  const ModelConfig config = test_config();
  Model model(config, 17);
  const int64_t batch = 4;
  const int64_t seq = 8;

  const Var loss =
      model.loss(random_ids(batch * seq, config.vocab_size, 6), batch, seq);
  const double expected = std::log(static_cast<double>(config.vocab_size));
  LLM_EXPECT_NEAR(*loss.value().data(), expected, 0.15);
}

LLM_TEST(Model, InitialLossIsNearUniformForLargerVocab) {
  // То же на пресете с настоящим словарём: поправка инициализации выходных
  // проекций на глубину должна работать и там.
  ModelConfig config = ModelConfig::nano();
  config.max_seq_len = 16;
  Model model(config, 19);

  const Var loss = model.loss(random_ids(2 * 16, config.vocab_size, 7), 2, 16);
  LLM_EXPECT_NEAR(*loss.value().data(),
                  std::log(static_cast<double>(config.vocab_size)), 0.2);
}

LLM_TEST(Model, IsDeterministic) {
  const ModelConfig config = test_config();
  const std::vector<int32_t> ids = random_ids(6, config.vocab_size, 8);

  Model first(config, 23);
  Model second(config, 23);
  const Var a = first.forward(ids, 1, 6);
  const Var b = second.forward(ids, 1, 6);

  for (int64_t position = 0; position < 6; ++position) {
    for (int64_t token = 0; token < config.vocab_size; ++token) {
      LLM_CHECK(a.value()(0, position, token) == b.value()(0, position, token));
    }
  }
}

LLM_TEST(Model, EveryParameterReceivesGradient) {
  // Параметр без градиента означает, что он не участвует в вычислении потерь —
  // то есть либо забыт в прямом проходе, либо лишний.
  const ModelConfig config = test_config();
  Model model(config, 29);
  model.loss(random_ids(2 * 6, config.vocab_size, 9), 2, 6).backward();

  const std::vector<llm::nn::NamedParameter> parameters = model.parameters();
  LLM_CHECK_GT(parameters.size(), static_cast<std::size_t>(10));
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    LLM_CHECK_MSG(parameters[i].value->grad().defined(),
                  "параметр " << parameters[i].name << " не получил градиента");
    LLM_CHECK_MSG(
        parameters[i].value->grad().shape() == parameters[i].value->shape(),
        "у параметра " << parameters[i].name << " форма градиента "
                       << parameters[i].value->grad().shape() << " вместо "
                       << parameters[i].value->shape());
  }
}

LLM_TEST(Model, ParameterNamesAreUnique) {
  Model model(test_config(), 31);
  const std::vector<llm::nn::NamedParameter> parameters = model.parameters();
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    for (std::size_t j = i + 1; j < parameters.size(); ++j) {
      LLM_CHECK_MSG(parameters[i].name != parameters[j].name,
                    "имя параметра повторяется: " << parameters[i].name);
    }
  }
}

LLM_TEST(Model, TiedEmbeddingGetsGradientFromBothPaths) {
  // При связанных эмбеддингах одна матрица работает и на входе, и на выходе.
  // Её градиент обязан собрать вклады обоих путей — на этом проверяется, что
  // накопление в ленте действительно складывает, а не замещает.
  ModelConfig tied = test_config();
  ModelConfig untied = test_config();
  untied.tie_embeddings = false;

  Model tied_model(tied, 37);
  Model untied_model(untied, 37);

  const std::vector<int32_t> ids = random_ids(2 * 6, tied.vocab_size, 10);
  tied_model.loss(ids, 2, 6).backward();
  untied_model.loss(ids, 2, 6).backward();

  // У несвязанной модели таблица эмбеддингов получает градиент только от
  // поиска строк, и он заметно отличается от связанного случая.
  const std::vector<llm::nn::NamedParameter> tied_parameters =
      tied_model.parameters();
  const std::vector<llm::nn::NamedParameter> untied_parameters =
      untied_model.parameters();
  LLM_CHECK(tied_parameters[0].name == "token_embedding");
  LLM_CHECK(untied_parameters[0].name == "token_embedding");

  // В несвязанной модели строка неиспользованного токена остаётся без
  // градиента, в связанной — нет: она участвует в выходной проекции для
  // каждой позиции.
  bool found_difference = false;
  for (int64_t token = 0; token < tied.vocab_size; ++token) {
    if (std::fabs(tied_parameters[0].value->grad()(token, 0) -
                  untied_parameters[0].value->grad()(token, 0)) > 1e-8f) {
      found_difference = true;
    }
  }
  LLM_CHECK(found_difference);
  LLM_CHECK_EQ(untied_parameters.size(), tied_parameters.size() + 1);
}

LLM_TEST(Model, GradientsMatchNumericDifference) {
  // Сквозная численная проверка всей модели: возмущаем отдельные элементы
  // настоящих параметров и сравниваем изменение потерь с тем, что выдал
  // обратный проход.
  //
  // Проверяется не какая-то одна операция, а вся цепочка целиком, включая
  // связывание эмбеддингов, размножение голов и остаточные соединения.
  // Элементы выбираются случайно: перебирать все параметры дорого, а
  // выборки хватает, чтобы поймать систематическую ошибку.
  const ModelConfig config = test_config();
  Model model(config, 41);
  const int64_t batch = 2;
  const int64_t seq = 6;
  const std::vector<int32_t> ids =
      random_ids(batch * seq, config.vocab_size, 11);

  model.loss(ids, batch, seq).backward();

  std::vector<llm::nn::NamedParameter> parameters = model.parameters();
  llm::Rng rng(4711);
  const float step = 1e-2f;

  double worst = 0.0;
  int checked = 0;
  for (int trial = 0; trial < 60; ++trial) {
    llm::nn::NamedParameter& parameter = parameters[static_cast<std::size_t>(
        rng.index(static_cast<uint64_t>(parameters.size())))];
    llm::Tensor& values = parameter.value->value();
    const int64_t index =
        static_cast<int64_t>(rng.index(static_cast<uint64_t>(values.numel())));

    const float analytic = parameter.value->grad().data()[index];
    const float original = values.data()[index];

    llm::autograd::NoGradGuard no_grad;
    values.data()[index] = original + step;
    const double plus = *model.loss(ids, batch, seq).value().data();
    values.data()[index] = original - step;
    const double minus = *model.loss(ids, batch, seq).value().data();
    values.data()[index] = original;

    const double numeric = (plus - minus) / (2.0 * static_cast<double>(step));
    const double scale =
        std::max(1.0, std::fabs(static_cast<double>(analytic)));
    const double error =
        std::fabs(numeric - static_cast<double>(analytic)) / scale;

    worst = std::max(worst, error);
    ++checked;
    LLM_CHECK_MSG(error < 5e-3, "параметр " << parameter.name << ", элемент "
                                            << index << ": обратный проход дал "
                                            << analytic << ", численная оценка "
                                            << numeric << ", расхождение "
                                            << error);
  }
  LLM_CHECK_EQ(checked, 60);
}

// --- половинная разрядность весов --------------------------------------------
//
// Обещание то же, что у gemm_half_b, только на уровне всей модели: прямой
// проход на упакованных весах обязан дать ПОБИТОВО то же, что прямой проход на
// обычных весах, уже прошедших округление. Не «близко» — побитово.
//
// Проверять с допуском здесь особенно бессмысленно: логиты модели после
// округления весов и так сдвигаются в третьем знаке, и любой разумный допуск
// пропустил бы перепутанную транспонированную таблицу или потерянный слой.
namespace {

void round_all_parameters(Model* model) {
  const std::vector<llm::nn::NamedParameter> parameters = model->parameters();
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    llm::Tensor& value = parameters[i].value->value();
    float* data = value.data();
    for (int64_t j = 0; j < value.numel(); ++j) {
      data[j] = llm::round_to_fp16(data[j]);
    }
  }
}

std::vector<float> forward_logits(Model* model,
                                  const std::vector<int32_t>& ids,
                                  int64_t batch, int64_t seq) {
  llm::autograd::NoGradGuard no_grad;
  const Var logits = model->forward(ids, batch, seq);
  const llm::Tensor dense = logits.value().contiguous();
  return std::vector<float>(dense.data(), dense.data() + dense.numel());
}

void check_half_forward_matches(bool tie_embeddings) {
  ModelConfig config = test_config();
  config.tie_embeddings = tie_embeddings;
  config.validate();

  Model model(config, 12345);
  round_all_parameters(&model);

  const int64_t batch = 2;
  const int64_t seq = 6;
  const std::vector<int32_t> ids =
      random_ids(batch * seq, config.vocab_size, 777);

  const std::vector<float> reference = forward_logits(&model, ids, batch, seq);
  model.pack_half();
  const std::vector<float> actual = forward_logits(&model, ids, batch, seq);

  LLM_CHECK_MSG(reference.size() == actual.size(), "формы логитов разошлись");
  for (std::size_t i = 0; i < reference.size(); ++i) {
    LLM_CHECK_MSG(reference[i] == actual[i],
                  "связанные эмбеддинги = "
                      << tie_embeddings << ": логит " << i << " равен "
                      << actual[i] << " вместо " << reference[i]);
  }
}

}  // namespace

// Готовая транспонированная таблица не меняет ни одного разряда.
//
// Обещание сильное и проверяется как сильное. Упаковка с перестановкой лишь
// переставляет значения, а порядок накопления по глубине у обоих путей один и
// тот же — значит совпадать обязано побитово, а не приблизительно. Если бы
// совпадало приблизительно, это означало бы, что пути расходятся, и тогда
// логиты зависели бы от того, вызвали ли подготовку.
LLM_TEST(Model, PreparedEmbeddingGivesTheSameLogits) {
  ModelConfig config = test_config();
  config.tie_embeddings = true;
  config.validate();

  Model model(config, 555);
  const int64_t batch = 2;
  const int64_t seq = 5;
  const std::vector<int32_t> ids =
      random_ids(batch * seq, config.vocab_size, 8080);

  const std::vector<float> reference = forward_logits(&model, ids, batch, seq);
  model.prepare_inference();
  const std::vector<float> actual = forward_logits(&model, ids, batch, seq);

  LLM_CHECK_MSG(reference.size() == actual.size(), "формы логитов разошлись");
  for (std::size_t i = 0; i < reference.size(); ++i) {
    LLM_CHECK_MSG(reference[i] == actual[i],
                  "логит " << i << " равен " << actual[i] << " вместо "
                           << reference[i]);
  }
}

// Подготовка не должна задевать обучение: при включённой ленте выходная
// проекция обязана идти через сам параметр, иначе градиент до таблицы
// эмбеддингов не дойдёт и обучение молча перестанет её обновлять.
LLM_TEST(Model, PreparingInferenceDoesNotChangeTraining) {
  ModelConfig config = test_config();
  config.tie_embeddings = true;
  config.validate();

  Model model(config, 606);
  const int64_t batch = 2;
  const int64_t seq = 5;
  const std::vector<int32_t> ids =
      random_ids(batch * seq, config.vocab_size, 9090);

  Var before = model.loss(ids, batch, seq);
  before.backward();
  const std::vector<llm::nn::NamedParameter> parameters = model.parameters();
  llm::Tensor grad_before = parameters[0].value->grad().clone();
  const float loss_before = before.value().data()[0];
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    parameters[i].value->zero_grad();
  }

  model.prepare_inference();

  Var after = model.loss(ids, batch, seq);
  after.backward();
  const float loss_after = after.value().data()[0];

  LLM_CHECK_MSG(loss_before == loss_after,
                "потери стали " << loss_after << " вместо " << loss_before);
  // Первый параметр — таблица эмбеддингов, и именно её градиент исчез бы,
  // если бы выходная проекция пошла мимо параметра.
  const llm::Tensor& grad_after = parameters[0].value->grad();
  LLM_CHECK_MSG(grad_after.defined(), "градиент таблицы эмбеддингов пропал");
  for (int64_t i = 0; i < grad_before.numel(); ++i) {
    LLM_CHECK_MSG(grad_before.data()[i] == grad_after.data()[i],
                  "градиент таблицы изменился в элементе " << i);
  }
}

LLM_TEST(Model, HalfWeightsGiveTheSameLogits) {
  check_half_forward_matches(true);
  check_half_forward_matches(false);
}

// Обучение после упаковки идёт как прежде. Это не мелочь: половинная
// разрядность здесь именно копия рядом, а не замена, и если бы прямой проход с
// градиентом случайно пошёл по ней, обучение молча потеряло бы точность весов
// — ровно то, что замер сходимости запретил.
LLM_TEST(Model, PackingHalfDoesNotChangeTraining) {
  ModelConfig config = test_config();
  Model model(config, 999);

  const int64_t batch = 2;
  const int64_t seq = 6;
  const std::vector<int32_t> ids =
      random_ids(batch * seq, config.vocab_size, 4242);

  const Var before = model.loss(ids, batch, seq);
  const float before_value = before.value().data()[0];

  model.pack_half();

  const Var after = model.loss(ids, batch, seq);
  const float after_value = after.value().data()[0];

  LLM_CHECK_MSG(before_value == after_value,
                "потери после упаковки стали " << after_value << " вместо "
                                               << before_value);
}
