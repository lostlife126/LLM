// Сборка модели: конфигурация, формы, каузальность, градиенты.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/fp16.h"
#include "core/random.h"
#include "nn/config.h"
#include "nn/lora.h"
#include "nn/model.h"
#include "train/optimizer.h"
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

  // Вещественные величины проверялись не все, а ошибка в них не падает — она
  // даёт NaN где-то в середине прямого прохода.
  config = test_config();
  config.norm_eps = 0.0f;  // нулевая строка дала бы бесконечность
  LLM_EXPECT_THROWS(config.validate());

  config = test_config();
  config.norm_eps = -1e-5f;
  LLM_EXPECT_THROWS(config.validate());

  config = test_config();
  config.rope_theta = 0.0f;  // дробная степень нуля не определена
  LLM_EXPECT_THROWS(config.validate());

  config = test_config();
  config.init_std = -0.02f;
  LLM_EXPECT_THROWS(config.validate());

  config = test_config();
  config.z_loss_coef = -1e-4f;  // штраф стал бы поощрением
  LLM_EXPECT_THROWS(config.validate());

  // А нулевой разброс инициализации законен: вырожденно, но осмысленно.
  config = test_config();
  config.init_std = 0.0f;
  config.validate();
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

// Логиты последней позиции — это логиты последней позиции.
//
// forward_last — то, чем пользуется вся генерация, а проверялся он только сам
// с собой: сравнения «с кэшем против без кэша» гоняют его с обеих сторон, и
// общая ошибка прошла бы их насквозь. Срез берётся по оси номер один, а
// осей там три; перепутанная ось при батче или длине в единицу дала бы ту же
// форму и то же число, поэтому берутся обе больше единицы.
LLM_TEST(Model, ConfigRefusesNonFiniteNumbers) {
  // Сравнения в validate() ловят NaN сами: с ним ложно любое сравнение. А
  // плюс бесконечность проходит их все — она и больше нуля, и не меньше нуля.
  //
  // Прийти она может из повреждённого чекпоинта: там вещественные поля
  // читаются сырыми байтами. Падения дальше не будет. При бесконечном
  // rope_theta частоты всех пар, кроме нулевой, обращаются в нуль — поворот
  // позиций тихо выключается; при бесконечном norm_eps нормировка отдаёт нули.
  const float infinity = std::numeric_limits<float>::infinity();
  const float not_a_number = std::numeric_limits<float>::quiet_NaN();

  {
    ModelConfig config = test_config();
    config.norm_eps = infinity;
    LLM_EXPECT_THROWS(config.validate());
  }
  {
    ModelConfig config = test_config();
    config.rope_theta = infinity;
    LLM_EXPECT_THROWS(config.validate());
  }
  {
    ModelConfig config = test_config();
    config.init_std = infinity;
    LLM_EXPECT_THROWS(config.validate());
  }
  {
    ModelConfig config = test_config();
    config.z_loss_coef = infinity;
    LLM_EXPECT_THROWS(config.validate());
  }

  // NaN отвергается и без новой проверки — это видно по тому, что тест на неё
  // не опирается, но пусть будет записано, что оба случая закрыты.
  {
    ModelConfig config = test_config();
    config.norm_eps = not_a_number;
    LLM_EXPECT_THROWS(config.validate());
  }

  // Проверка непуста: исправная конфигурация проходит.
  ModelConfig good = test_config();
  good.validate();
}

LLM_TEST(Model, ForwardLastIsTheLastRowOfForward) {
  const ModelConfig config = test_config();
  Model model(config, 4242);
  const int64_t batch = 2;
  const int64_t seq = 5;
  const std::vector<int32_t> ids =
      random_ids(batch * seq, config.vocab_size, 31337);

  const Var full = model.forward(ids, batch, seq);
  const Var last = model.forward_last(ids, batch, seq);

  LLM_CHECK(last.shape() == llm::Shape({batch, config.vocab_size}));
  for (int64_t item = 0; item < batch; ++item) {
    for (int64_t token = 0; token < config.vocab_size; ++token) {
      LLM_CHECK_MSG(
          last.value()(item, token) == full.value()(item, seq - 1, token),
          "элемент батча " << item << ", токен " << token << ": "
                           << last.value()(item, token) << " вместо "
                           << full.value()(item, seq - 1, token));
    }
  }

  // Проверка непуста: у соседней позиции логиты другие, значит совпадение
  // выше — это именно последняя строка, а не «любая подойдёт».
  bool neighbour_differs = false;
  for (int64_t token = 0; token < config.vocab_size; ++token) {
    if (full.value()(0, seq - 2, token) != full.value()(0, seq - 1, token)) {
      neighbour_differs = true;
    }
  }
  LLM_CHECK(neighbour_differs);
}

LLM_TEST(Model, BatchItemsAreIndependent) {
  // Элементы батча не должны видеть друг друга. Ошибка в свёртке осей батча и
  // голов проявилась бы именно так.
  //
  // Сравнение точное, и это не само собой разумеется. matmul_into сливает
  // умножение на общий вес в одно, то есть m равно batch * seq и от размера
  // батча зависит. А от m зависит выбор пути в gemm, и пути делят глубину
  // по-разному — значит на достаточно крупной модели переход с батча 1 на
  // батч 2 сдвинул бы логиты в последних разрядах.
  //
  // Здесь этого не происходит, потому что веса маленькие: B влезает в кэш, и
  // прямой путь выбирается при любом m. Проверено: на модели покрупнее
  // (d_model 256, FFN 1024) принудительная смена пути двигает логиты на 8e-7.
  //
  // То есть тест ловит настоящую зависимость элементов батча друг от друга, а
  // не разницу округления. Если однажды он упадёт на увеличенной модели,
  // смотреть надо сюда, а не на свёртку осей.
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

namespace {

// Масштаб градиента по всему тензору параметра.
//
// По всему, а не по отдельному элементу: у отдельного градиент бывает близок
// к нулю, и деление на него превратило бы шум численной оценки в «ошибку».
// Так же устроено сравнение в tests/gradcheck.h.
double gradient_scale_of(const llm::nn::NamedParameter& parameter) {
  const llm::Tensor gradient = parameter.value->grad().contiguous();
  double largest = 0.0;
  for (int64_t i = 0; i < gradient.numel(); ++i) {
    largest = std::max(largest,
                       std::fabs(static_cast<double>(gradient.data()[i])));
  }
  return largest;
}

// Разрешение самой центральной разности: она вычитает два близких значения
// функции и делит на 2h, поэтому её собственный шум — порядка eps * |f| / h
// при машинной точности float.
double method_resolution(double plus, double minus, float step) {
  const double kFloatEpsilon = 1.1920929e-7;
  return kFloatEpsilon * std::max(std::fabs(plus), std::fabs(minus)) /
         static_cast<double>(step);
}

}  // namespace

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

    // Нижняя граница масштаба — не единица, а то, что метод вообще способен
    // различить.
    //
    // Единица стояла здесь раньше и выглядела безобидно, а на деле подменяла
    // относительное сравнение абсолютным везде, где градиенты меньше её, —
    // то есть почти везде. Ту же ошибку в tests/gradcheck.h уже нашли и
    // исправили, а здесь она осталась: масштаб нормировки собирает градиент
    // величиной 1.5e-4, и допуск 5e-3 означал для него не «полпроцента», а
    // «в тридцать раз больше самого градиента».
    //
    // Граница записана как resolution / kTolerance, а не как resolution:
    // допустимая абсолютная ошибка должна равняться разрешению метода, а не
    // быть в двести раз меньше него. У gradcheck эта ветвь не срабатывает ни
    // разу (там градиенты крупнее), поэтому разницы между двумя записями там
    // не видно; здесь она видна сразу.
    const double kTolerance = 5e-3;
    const double scale =
        std::max(gradient_scale_of(parameter),
                 method_resolution(plus, minus, step) / kTolerance);
    const double error =
        std::fabs(numeric - static_cast<double>(analytic)) / scale;

    worst = std::max(worst, error);
    ++checked;
    LLM_CHECK_MSG(error < kTolerance,
                  "параметр " << parameter.name << ", элемент " << index
                              << ": обратный проход дал " << analytic
                              << ", численная оценка " << numeric
                              << ", расхождение " << error);
  }
  LLM_CHECK_EQ(checked, 60);
  // Допуск не взят с потолка: наибольшее расхождение по этой выборке —
  // 1.154e-3, и оно одинаково на всех четырёх сборках проекта, включая
  // aarch64 под эмуляцией. Запас вчетверо.
  //
  // Что проверка теперь ловит, проверено подменой: множитель 1.01 в обратном
  // проходе matmul по второму аргументу роняет её на выходной проекции
  // внимания (расхождение 6.2e-3 при градиенте 0.0375). С прежней границей в
  // единицу та же подмена проходила насквозь: абсолютное расхождение выходило
  // 3.9e-4, то есть в тринадцать раз меньше допуска. Ошибку в 0.2% этот тест
  // по-прежнему не видит — множитель меняет градиент пропорционально, и
  // относительное расхождение равно самому множителю минус единица; за такой
  // мелочью идут численные проверки отдельных операций, где допуск 1e-3.
  LLM_CHECK_MSG(worst < 5e-3, "наибольшее расхождение " << worst);
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

// --- подсчёт параметров ничего не подготавливает обратно ---------------------
namespace {

void check_logits_equal(const std::vector<float>& expected,
                        const std::vector<float>& actual, const char* what) {
  LLM_CHECK_MSG(expected.size() == actual.size(), what << ": формы разошлись");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    LLM_CHECK_MSG(expected[i] == actual[i],
                  what << ": логит " << i << " равен " << actual[i]
                       << " вместо " << expected[i]);
  }
}

void check_counting_keeps_prepared_copies(bool tie_embeddings) {
  ModelConfig config = test_config();
  config.tie_embeddings = tie_embeddings;
  config.validate();

  Model model(config, 31337);
  const int64_t batch = 2;
  const int64_t seq = 6;
  const std::vector<int32_t> ids =
      random_ids(batch * seq, config.vocab_size, 24680);

  // Веса здесь НЕ округляются заранее — в отличие от проверки побитового
  // совпадения выше, и именно в этом весь смысл. На округлённых весах оба
  // прохода дают одно и то же, и отличить работающую копию от сброшенной
  // стало бы нечем.
  const std::vector<float> plain = forward_logits(&model, ids, batch, seq);
  model.pack_half();
  model.prepare_inference();
  const std::vector<float> half = forward_logits(&model, ids, batch, seq);

  LLM_CHECK_MSG(plain.size() == half.size(), "формы логитов разошлись");
  bool differ = false;
  for (std::size_t i = 0; i < plain.size() && !differ; ++i) {
    differ = plain[i] != half[i];
  }
  LLM_CHECK_MSG(differ,
                "связанные эмбеддинги = "
                    << tie_embeddings
                    << ": округление весов не сдвинуло ни одного логита, и "
                       "сравнения ниже ничего не проверяют");

  LLM_CHECK_GT(model.parameter_count(), static_cast<int64_t>(0));
  check_logits_equal(half, forward_logits(&model, ids, batch, seq),
                     "после parameter_count");

  LLM_CHECK_GT(model.trainable_parameter_count(), static_cast<int64_t>(0));
  check_logits_equal(half, forward_logits(&model, ids, batch, seq),
                     "после trainable_parameter_count");
}

}  // namespace

// Подсчёт параметров — чтение, и подготовку к инференсу он отменять не вправе.
//
// Обе функции подсчёта отдают наружу одно число, а параметры обходят лишь
// затем, чтобы сложить их размеры. Если бы обход при этом отзывал половинную
// разрядность и переложенную таблицу, вышла бы ровно та ошибка, ради которой
// весь этот проект и мерит: печать размера модели молча переводила бы её
// обратно в обычную разрядность, а замер скорости рядом показывал бы цифру не
// от того прохода, который заявлен. Ни падения, ни расхождения — только вдвое
// меньшая скорость, объяснимая чем угодно.
//
// Проверяется на обеих раскладках выходной проекции: при связанных
// эмбеддингах подготовлены копии уровня модели, при раздельных — ещё и
// половинный вес самой lm_head.
LLM_TEST(Model, CountingParametersKeepsThePreparedCopies) {
  check_counting_keeps_prepared_copies(true);
  check_counting_keeps_prepared_copies(false);
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

// --- недействительность подготовленных копий ------------------------------
//
// pack_half и prepare_inference делают КОПИИ весов. Если вес после этого
// изменится, копия станет неверной, и прямой проход посчитает по старым весам,
// ничего об этом не сказав. Это худший вид ошибки: не падение, а тихо неверный
// ответ.
//
// Правило поэтому такое: любой, кто получил изменяемую ссылку на вес, тем самым
// объявляет копии недействительными. Единственная точка, через которую такие
// ссылки уходят наружу, — parameters(), и она же их сбрасывает. Плюс
// merge_lora, который меняет вес сам, никого не спрашивая.
namespace {

// Первый параметр с таким окончанием имени.
llm::autograd::Var* find_parameter(
    const std::vector<llm::nn::NamedParameter>& parameters,
    const std::string& suffix) {
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    const std::string& name = parameters[i].name;
    if (name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      return parameters[i].value;
    }
  }
  return nullptr;
}

}  // namespace

// Обучение после упаковки. Оптимизатор получает веса через parameters(), то
// есть копии обязаны сброситься, и следующий прямой проход — считать по новым
// весам.
LLM_TEST(Model, TrainingAfterPackingInvalidatesTheHalfCopies) {
  ModelConfig config = test_config();
  const int64_t batch = 2;
  const int64_t seq = 5;
  const std::vector<int32_t> ids =
      random_ids(batch * seq, config.vocab_size, 1357);

  // Две модели с одним зерном — значит с одними весами. Одна проходит через
  // упаковку, другая нет; после одинакового шага обучения логиты обязаны
  // совпасть побитово.
  Model packed(config, 2468);
  Model plain(config, 2468);
  packed.pack_half();
  packed.prepare_inference();

  Model* models[2] = {&packed, &plain};
  std::vector<float> logits[2];
  for (int which = 0; which < 2; ++which) {
    Model* model = models[which];
    Var loss = model->loss(ids, batch, seq);
    loss.backward();
    llm::train::AdamWConfig adam;
    llm::train::AdamW optimizer(model->parameters(), adam);
    optimizer.step(0.1f);
    logits[which] = forward_logits(model, ids, batch, seq);
  }

  LLM_CHECK_MSG(logits[0].size() == logits[1].size(), "формы разошлись");
  for (std::size_t i = 0; i < logits[0].size(); ++i) {
    LLM_CHECK_MSG(logits[0][i] == logits[1][i],
                  "логит " << i << " после шага обучения равен " << logits[0][i]
                           << " вместо " << logits[1][i]);
  }
}

// Вплавление адаптера. Оно меняет вес само, не проходя через parameters(), и
// вдобавок снимает признак наличия адаптера — то есть именно оно возвращало
// половинную разрядность к жизни на устаревшей копии.
//
// Проверяется на отдельном слое, а не на модели, и это важно. На уровне модели
// сравнение неизбежно ложное: адаптер вешается не на все линейные слои, у
// остальных копии остаются ВЕРНЫМИ и продолжают считать в половинной
// разрядности — то есть сравнивались бы разрядности, а не свежесть копии.
// Первые две версии этого теста попались именно на это, сначала через
// связанные эмбеддинги, потом через слои без адаптера.
LLM_TEST(Linear, MergingLoraInvalidatesTheHalfCopy) {
  const int64_t in_features = 32;
  const int64_t out_features = 64;

  // Два слоя с одним зерном — значит с одним весом и одним адаптером.
  llm::Rng rng_a(4242);
  llm::Rng rng_b(4242);
  llm::nn::Linear packed(in_features, out_features, 0.05f, &rng_a);
  llm::nn::Linear plain(in_features, out_features, 0.05f, &rng_b);

  llm::Rng lora_rng_a(77);
  llm::Rng lora_rng_b(77);
  packed.enable_lora(4, 8.0f, &lora_rng_a);
  plain.enable_lora(4, 8.0f, &lora_rng_b);

  llm::nn::Linear* layers[2] = {&packed, &plain};
  std::vector<float> results[2];
  for (int which = 0; which < 2; ++which) {
    llm::nn::Linear* layer = layers[which];

    // B адаптера рождается нулевой, и вплавление нулевого адаптера ничего не
    // меняет — проверка вышла бы пустой.
    std::vector<llm::nn::NamedParameter> parameters;
    layer->collect("layer", &parameters);
    llm::autograd::Var* b = find_parameter(parameters, ".lora_b");
    LLM_CHECK_MSG(b != nullptr, "адаптер не навесился");
    llm::Tensor& value = b->value();
    for (int64_t i = 0; i < value.numel(); ++i) {
      value.data()[i] = 0.02f * static_cast<float>((i % 5) - 2);
    }

    // Упаковка ПОСЛЕ заполнения: на этот момент копия верна. Недействительной
    // её делает вплавление строкой ниже.
    if (which == 0) {
      layer->pack_half();
      LLM_CHECK_MSG(!layer->uses_half(),
                    "с адаптером половинная разрядность должна быть выключена");
    }
    layer->merge_lora();
    LLM_CHECK_MSG(!layer->uses_half(),
                  "после вплавления копия обязана стать недействительной");

    llm::autograd::NoGradGuard no_grad;
    llm::Tensor input = llm::Tensor::uninitialized(llm::Shape({3, in_features}));
    llm::Rng input_rng(11);
    for (int64_t i = 0; i < input.numel(); ++i) {
      input.data()[i] = input_rng.normal() * 0.3f;
    }
    const Var out = layer->forward(Var::constant(input));
    const llm::Tensor dense = out.value().contiguous();
    results[which].assign(dense.data(), dense.data() + dense.numel());
  }

  for (std::size_t i = 0; i < results[0].size(); ++i) {
    LLM_CHECK_MSG(results[0][i] == results[1][i],
                  "элемент " << i << " после вплавления равен " << results[0][i]
                             << " вместо " << results[1][i]);
  }
}

namespace {

// Независимый эталон одного слоя: то же вычисление, написанное заново и по
// формулам, а не по коду модели.
//
// Зачем он нужен именно здесь. Масштаб внимания 1/sqrt(head_dim) не проверялся
// ничем: подменённый на 1/head_dim, он оставлял все проверки проекта зелёными.
// Модель при этом обучается и говорит связно — просто не то, и с чужими весами
// расходится молча. Тем же приёмом (поменять одно место и посмотреть, заметит
// ли суд) выяснилось, что заодно не проверены ни eps нормировки, ни порядок
// сомножителей в SwiGLU, ни то, что остаток складывается до нормировки.
//
// Конфигурация подобрана под краткость эталона, а не под реализм: один слой,
// одна голова, позиции выключены (RoPE проверяется отдельно). Всё остальное —
// настоящее.
std::vector<double> rms_norm_row(const std::vector<double>& x,
                                 const llm::Tensor& weight, double eps) {
  double squares = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    squares += x[i] * x[i];
  }
  const double scale = 1.0 / std::sqrt(squares / static_cast<double>(x.size()) + eps);
  std::vector<double> out(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * scale * static_cast<double>(weight(static_cast<int64_t>(i)));
  }
  return out;
}

// row * matrix, где matrix хранится как (вход, выход).
std::vector<double> project(const std::vector<double>& row,
                            const llm::Tensor& matrix) {
  const int64_t out_features = matrix.dim(1);
  std::vector<double> out(static_cast<std::size_t>(out_features), 0.0);
  for (int64_t j = 0; j < out_features; ++j) {
    double sum = 0.0;
    for (std::size_t i = 0; i < row.size(); ++i) {
      sum += row[i] * static_cast<double>(matrix(static_cast<int64_t>(i), j));
    }
    out[static_cast<std::size_t>(j)] = sum;
  }
  return out;
}

// Возвращается по значению, а не по ссылке: тензор — это вид на общий буфер,
// копия его не копирует, а ссылка на элемент вектора заставляет компилятор
// подозревать висячую ссылку.
llm::Tensor by_name(const std::vector<llm::nn::NamedParameter>& all,
                    const std::string& name) {
  for (std::size_t i = 0; i < all.size(); ++i) {
    if (all[i].name == name) {
      return all[i].value->value();
    }
  }
  LLM_CHECK_MSG(false, "в модели нет параметра " << name);
  return all[0].value->value();
}

}  // namespace

LLM_TEST(Model, OneLayerMatchesAnIndependentReference) {
  ModelConfig config;
  config.vocab_size = 11;
  config.d_model = 8;
  config.n_layers = 1;
  config.n_heads = 1;
  config.n_kv_heads = 1;
  config.max_seq_len = 6;
  config.ffn_hidden = 12;
  config.position = llm::nn::PositionKind::kNone;
  config.tie_embeddings = true;
  config.validate();

  Model model(config, 123);
  const std::vector<llm::nn::NamedParameter> all = model.parameters();
  const llm::Tensor table = by_name(all, "token_embedding");
  const llm::Tensor attention_norm = by_name(all, "block.0.attention_norm.weight");
  llm::Tensor wq = by_name(all, "block.0.attention.query.weight");
  llm::Tensor wk = by_name(all, "block.0.attention.key.weight");
  llm::Tensor wv = by_name(all, "block.0.attention.value.weight");
  llm::Tensor wo = by_name(all, "block.0.attention.output.weight");
  const llm::Tensor mlp_norm = by_name(all, "block.0.mlp_norm.weight");
  llm::Tensor wgate = by_name(all, "block.0.mlp.gate.weight");
  llm::Tensor wup = by_name(all, "block.0.mlp.up.weight");
  const llm::Tensor wdown = by_name(all, "block.0.mlp.down.weight");
  const llm::Tensor final_norm = by_name(all, "final_norm.weight");

  // Веса запросов и ключей усиливаются, и это не украшение. У свежей модели
  // они малы, скалярные произведения близки к нулю, и softmax выходит почти
  // равномерным при любом масштабе — то есть тест не различал бы 1/sqrt(d) и
  // 1/d вовсе. Проверено: с исходными весами подмена масштаба этот тест
  // проходила. Усиление делает оценки величиной в единицы, и softmax
  // становится к масштабу чувствителен.
  for (int64_t i = 0; i < wq.numel(); ++i) {
    wq.data()[i] *= 20.0f;
    wk.data()[i] *= 20.0f;
    // Значения и выходную проекцию тоже: иначе вклад внимания в логиты
    // тонет рядом с путём остатка, и разница масштабов до них не доходит.
    wv.data()[i] *= 20.0f;
    wo.data()[i] *= 20.0f;
  }
  // И веса FFN — по той же причине, но про другую нелинейность. При малых
  // аргументах silu(x) ~ x/2, поэтому silu(вентиль) * up и вентиль * silu(up)
  // совпадают с точностью до второго порядка, и перестановка их местами
  // осталась бы незамеченной. Проверено: без этого усиления эталон такую
  // перестановку пропускал.
  for (int64_t i = 0; i < wgate.numel(); ++i) {
    wgate.data()[i] *= 20.0f;
    wup.data()[i] *= 20.0f;
  }

  const int64_t seq = 5;
  const std::vector<int32_t> ids = random_ids(seq, config.vocab_size, 77);
  const Var actual = model.forward(ids, 1, seq);

  const double eps = static_cast<double>(config.norm_eps);
  const std::size_t width = static_cast<std::size_t>(config.d_model);

  // 1. Эмбеддинги.
  std::vector<std::vector<double>> x(static_cast<std::size_t>(seq));
  for (int64_t t = 0; t < seq; ++t) {
    x[static_cast<std::size_t>(t)].resize(width);
    for (std::size_t i = 0; i < width; ++i) {
      x[static_cast<std::size_t>(t)][i] =
          static_cast<double>(table(ids[static_cast<std::size_t>(t)],
                                    static_cast<int64_t>(i)));
    }
  }

  // 2. Внимание: нормировка, проекции, оценки с масштабом 1/sqrt(head_dim),
  //    каузальная маска, softmax, взвешенная сумма, выходная проекция.
  std::vector<std::vector<double>> q(static_cast<std::size_t>(seq));
  std::vector<std::vector<double>> k(static_cast<std::size_t>(seq));
  std::vector<std::vector<double>> v(static_cast<std::size_t>(seq));
  for (int64_t t = 0; t < seq; ++t) {
    const std::vector<double> h =
        rms_norm_row(x[static_cast<std::size_t>(t)], attention_norm, eps);
    q[static_cast<std::size_t>(t)] = project(h, wq);
    k[static_cast<std::size_t>(t)] = project(h, wk);
    v[static_cast<std::size_t>(t)] = project(h, wv);
  }

  // Эталон считается как функция масштаба: сначала с правильным, потом с
  // заведомо неправильным. Второй прогон — проверка самой проверки: если бы
  // данные масштаба не различали, оба совпали бы с моделью одинаково хорошо,
  // и тест ничего не значил бы.
  const auto reference = [&](double scale) {
  std::vector<std::vector<double>> after(static_cast<std::size_t>(seq));
  for (int64_t t = 0; t < seq; ++t) {
    std::vector<double> scores(static_cast<std::size_t>(t + 1));
    double largest = -1e300;
    for (int64_t u = 0; u <= t; ++u) {
      double dot = 0.0;
      for (std::size_t i = 0; i < width; ++i) {
        dot += q[static_cast<std::size_t>(t)][i] * k[static_cast<std::size_t>(u)][i];
      }
      scores[static_cast<std::size_t>(u)] = dot * scale;
      largest = std::max(largest, scores[static_cast<std::size_t>(u)]);
    }
    double total = 0.0;
    for (std::size_t u = 0; u < scores.size(); ++u) {
      scores[u] = std::exp(scores[u] - largest);
      total += scores[u];
    }
    std::vector<double> mixed(width, 0.0);
    for (std::size_t u = 0; u < scores.size(); ++u) {
      const double weight = scores[u] / total;
      for (std::size_t i = 0; i < width; ++i) {
        mixed[i] += weight * v[u][i];
      }
    }
    const std::vector<double> projected = project(mixed, wo);
    after[static_cast<std::size_t>(t)].resize(width);
    for (std::size_t i = 0; i < width; ++i) {
      // Остаток складывается с выходом подслоя, а не с нормированным входом.
      after[static_cast<std::size_t>(t)][i] =
          x[static_cast<std::size_t>(t)][i] + projected[i];
    }
  }

  // 3. SwiGLU: silu на вентиле, произведение с up, затем down.
  std::vector<std::vector<double>> y(static_cast<std::size_t>(seq));
  for (int64_t t = 0; t < seq; ++t) {
    const std::vector<double> h =
        rms_norm_row(after[static_cast<std::size_t>(t)], mlp_norm, eps);
    const std::vector<double> gate = project(h, wgate);
    const std::vector<double> up = project(h, wup);
    std::vector<double> hidden(gate.size());
    for (std::size_t j = 0; j < gate.size(); ++j) {
      hidden[j] = gate[j] / (1.0 + std::exp(-gate[j])) * up[j];
    }
    const std::vector<double> down = project(hidden, wdown);
    std::vector<double> sum(width);
    for (std::size_t i = 0; i < width; ++i) {
      sum[i] = after[static_cast<std::size_t>(t)][i] + down[i];
    }
    y[static_cast<std::size_t>(t)] = rms_norm_row(sum, final_norm, eps);
  }

  // 4. Логиты: те же эмбеддинги, только транспонированные.
  double worst = 0.0;
  for (int64_t t = 0; t < seq; ++t) {
    for (int64_t token = 0; token < config.vocab_size; ++token) {
      double logit = 0.0;
      for (std::size_t i = 0; i < width; ++i) {
        logit += y[static_cast<std::size_t>(t)][i] *
                 static_cast<double>(table(token, static_cast<int64_t>(i)));
      }
      worst = std::max(worst, std::fabs(logit - static_cast<double>(
                                                   actual.value()(0, t, token))));
    }
  }
  return worst;
  };

  const double head_dim = static_cast<double>(config.head_dim());
  const double worst = reference(1.0 / std::sqrt(head_dim));
  const double with_wrong_scale = reference(1.0 / head_dim);
  // Проверка самой проверки. Замерено: с верным масштабом эталон расходится с
  // моделью на 2.4e-08, с масштабом 1/d — на 2.7e-02, то есть в миллион раз
  // сильнее. Порог поставлен на 1e-3: до него от неверного масштаба двадцать
  // семь крат, а от верного — пять порядков.
  LLM_CHECK_MSG(with_wrong_scale > 1e-3,
                "данные не различают масштаб внимания, проверка пуста: с "
                "масштабом 1/d эталон расходится с моделью лишь на "
                    << with_wrong_scale);
  // Допуск замерен, а не взят с запасом на глаз: расхождение эталона в
  // двойной точности с моделью на float составляет 2.4e-08 — одинаково под
  // gcc и под clang. Взято на три порядка больше, чтобы пережить другой
  // порядок сложений на другой машине; до того, что тест обязан ловить, всё
  // равно остаётся три порядка.
  LLM_CHECK_MSG(worst < 1e-5, "эталон разошёлся с моделью на " << worst);
}
