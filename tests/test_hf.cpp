// Чтение чужих моделей: config.json, раскладка весов, соглашения.
//
// Главное здесь — не то, что файл читается, а то, что читается правильно.
// Расхождения между нашей реализацией и transformers таковы, что неправильно
// прочитанная модель работает и выдаёт связный текст, просто не тот. Поэтому
// каждое соглашение проверяется отдельно, а не только сквозным прогоном.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nn/model.h"
#include "ops/rope.h"
#include "serialize/hf_config.h"
#include "serialize/hf_import.h"
#include "serialize/safetensors.h"
#include "testing.h"

using llm::serialize::HfConfig;
using llm::serialize::NamedTensor;
using llm::serialize::SafeTensors;

namespace {

// Конфигурация SmolLM2-135M, дословно — это настоящая маленькая модель
// архитектуры Llama, и на ней видно, какие поля встречаются в жизни.
const char* const kSmolConfig = R"({
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "eos_token_id": 0,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "is_llama_config": true,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.42.3",
  "use_cache": true,
  "vocab_size": 49152
})";

// Поворот позиционными частотами так, как его делает Llama в transformers:
// канал i в паре с каналом i + head_dim/2. Написано здесь по формуле, а не
// заимствовано из нашего кода, — иначе проверка сравнивала бы реализацию
// с самой собой.
std::vector<float> rope_by_halves(const std::vector<float>& input,
                                  int64_t head_dim, int64_t position,
                                  double theta) {
  std::vector<float> out(input.size());
  const int64_t half = head_dim / 2;
  for (int64_t i = 0; i < half; ++i) {
    const double frequency = std::pow(
        theta, -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
    const double angle = static_cast<double>(position) * frequency;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);

    const double low = input[static_cast<std::size_t>(i)];
    const double high = input[static_cast<std::size_t>(i + half)];
    out[static_cast<std::size_t>(i)] =
        static_cast<float>(low * cosine - high * sine);
    out[static_cast<std::size_t>(i + half)] =
        static_cast<float>(low * sine + high * cosine);
  }
  return out;
}

llm::nn::ModelConfig small_llama() {
  llm::nn::ModelConfig config;
  config.vocab_size = 37;
  config.d_model = 24;
  config.n_layers = 2;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.max_seq_len = 8;
  config.ffn_hidden = 40;
  config.tie_embeddings = true;
  return config;
}

}  // namespace

LLM_TEST(Hf, RopeConventionIsAPermutationOfChannels) {
  // Ключевое утверждение всего импорта: наш поворот соседних пар и поворот
  // половинками из transformers — это одно и то же вращение, записанное в
  // разном порядке каналов. Если это неверно, модель загрузится и будет
  // выдавать связный мусор.
  const int64_t head_dim = 8;
  const int64_t half = head_dim / 2;
  const int64_t position = 5;
  const double theta = 10000.0;

  std::vector<float> source(static_cast<std::size_t>(head_dim));
  for (int64_t i = 0; i < head_dim; ++i) {
    source[static_cast<std::size_t>(i)] =
        static_cast<float>(0.3 * static_cast<double>(i) - 1.1);
  }

  const std::vector<float> theirs =
      rope_by_halves(source, head_dim, position, theta);

  // Тот же вектор, переставленный в наш порядок: наш 2i — их i, наш 2i+1 —
  // их i + half.
  std::vector<float> permuted(static_cast<std::size_t>(head_dim));
  for (int64_t i = 0; i < half; ++i) {
    permuted[static_cast<std::size_t>(2 * i)] =
        source[static_cast<std::size_t>(i)];
    permuted[static_cast<std::size_t>(2 * i + 1)] =
        source[static_cast<std::size_t>(i + half)];
  }

  const llm::Tensor input =
      llm::Tensor::from_values(llm::Shape({1, head_dim}), permuted);
  const llm::Tensor ours =
      llm::ops::rope(input, position, static_cast<float>(theta));

  for (int64_t i = 0; i < half; ++i) {
    LLM_EXPECT_NEAR(ours(0, 2 * i),
                    static_cast<double>(theirs[static_cast<std::size_t>(i)]),
                    1e-6);
    LLM_EXPECT_NEAR(
        ours(0, 2 * i + 1),
        static_cast<double>(theirs[static_cast<std::size_t>(i + half)]), 1e-6);
  }
}

LLM_TEST(Hf, ChannelPermutationMovesExactlyTheRightRows) {
  // Перестановка на маленькой матрице, где результат можно выписать руками.
  // Две головы по четыре канала: у первой строки 0..3, у второй 4..7.
  const int64_t heads = 2;
  const int64_t head_dim = 4;
  std::vector<float> values;
  for (int64_t row = 0; row < heads * head_dim; ++row) {
    values.push_back(static_cast<float>(row));
  }
  const llm::Tensor input =
      llm::Tensor::from_values(llm::Shape({heads * head_dim, 1}), values);

  const llm::Tensor ours =
      llm::serialize::halves_to_pairs(input, heads, head_dim);
  // Голова 0: их 0,1,2,3 -> наши 0,2,1,3.
  LLM_EXPECT_NEAR(ours(0, 0), 0.0, 0.0);
  LLM_EXPECT_NEAR(ours(1, 0), 2.0, 0.0);
  LLM_EXPECT_NEAR(ours(2, 0), 1.0, 0.0);
  LLM_EXPECT_NEAR(ours(3, 0), 3.0, 0.0);
  // Голова 1 переставляется независимо и не подмешивает чужие каналы.
  LLM_EXPECT_NEAR(ours(4, 0), 4.0, 0.0);
  LLM_EXPECT_NEAR(ours(5, 0), 6.0, 0.0);
  LLM_EXPECT_NEAR(ours(6, 0), 5.0, 0.0);
  LLM_EXPECT_NEAR(ours(7, 0), 7.0, 0.0);

  const llm::Tensor back =
      llm::serialize::pairs_to_halves(ours, heads, head_dim);
  for (int64_t row = 0; row < heads * head_dim; ++row) {
    LLM_EXPECT_NEAR(back(row, 0), static_cast<double>(row), 0.0);
  }
}

LLM_TEST(Hf, ReadsRealisticConfig) {
  const HfConfig config =
      llm::serialize::parse_hf_config(kSmolConfig, "SmolLM2-135M", 512);
  LLM_CHECK_EQ(config.model.vocab_size, static_cast<int64_t>(49152));
  LLM_CHECK_EQ(config.model.d_model, static_cast<int64_t>(576));
  LLM_CHECK_EQ(config.model.n_layers, static_cast<int64_t>(30));
  LLM_CHECK_EQ(config.model.n_heads, static_cast<int64_t>(9));
  LLM_CHECK_EQ(config.model.n_kv_heads, static_cast<int64_t>(3));
  LLM_CHECK_EQ(config.model.ffn_hidden, static_cast<int64_t>(1536));
  LLM_CHECK_EQ(config.model.head_dim(), static_cast<int64_t>(64));
  LLM_CHECK(config.tie_embeddings);
  LLM_CHECK_EQ(config.architecture, std::string("LlamaForCausalLM"));
  LLM_EXPECT_NEAR(config.model.rope_theta, 10000.0, 1e-6);
  LLM_EXPECT_NEAR(config.model.norm_eps, 1e-5, 1e-12);

  // Окно берём своё: max_position_embeddings у модели 8192, а маска и кэш у
  // нас растут квадратом длины.
  LLM_CHECK_EQ(config.model.max_seq_len, static_cast<int64_t>(512));
}

LLM_TEST(Hf, RejectsWhatWeCannotRepresent) {
  const auto with = [](const std::string& extra) {
    return std::string(
               "{\"model_type\":\"llama\",\"vocab_size\":32,\"hidden_size\":8,"
               "\"num_hidden_layers\":1,\"num_attention_heads\":2,"
               "\"num_key_value_heads\":1,\"intermediate_size\":16,"
               "\"rms_norm_eps\":1e-5") +
           extra + "}";
  };
  // Здоровый случай читается.
  llm::serialize::parse_hf_config(with(""), "проба", 8);

  // Свободные члены: веса остались бы правильными, а ответы — нет.
  LLM_EXPECT_THROWS(llm::serialize::parse_hf_config(
      with(",\"attention_bias\":true"), "п", 8));
  LLM_EXPECT_THROWS(
      llm::serialize::parse_hf_config(with(",\"mlp_bias\":true"), "п", 8));

  // Другая активация — другой FFN.
  LLM_EXPECT_THROWS(llm::serialize::parse_hf_config(
      with(",\"hidden_act\":\"gelu\""), "п", 8));

  // Растяжение позиций молча меняет поведение на длинном входе.
  LLM_EXPECT_THROWS(llm::serialize::parse_hf_config(
      with(",\"rope_scaling\":{\"type\":\"linear\",\"factor\":2.0}"), "п", 8));

  // Модель уже в нашей раскладке каналов — перестановка её испортит.
  LLM_EXPECT_THROWS(llm::serialize::parse_hf_config(
      with(",\"rope_interleaved\":true"), "п", 8));

  // Незнакомое поле: оно может менять вычисление, и пропустить его нельзя.
  LLM_EXPECT_THROWS(llm::serialize::parse_hf_config(
      with(",\"sliding_window\":4096"), "п", 8));

  // Не Llama.
  LLM_EXPECT_THROWS(llm::serialize::parse_hf_config(
      "{\"model_type\":\"qwen2\",\"vocab_size\":32,\"hidden_size\":8,"
      "\"num_hidden_layers\":1,\"num_attention_heads\":2,"
      "\"num_key_value_heads\":1,\"intermediate_size\":16,"
      "\"rms_norm_eps\":1e-5}",
      "п", 8));

  // Несовместимая арифметика формы.
  LLM_EXPECT_THROWS(llm::serialize::parse_hf_config(
      "{\"model_type\":\"llama\",\"vocab_size\":32,\"hidden_size\":9,"
      "\"num_hidden_layers\":1,\"num_attention_heads\":2,"
      "\"num_key_value_heads\":1,\"intermediate_size\":16,"
      "\"rms_norm_eps\":1e-5}",
      "п", 8));
  LLM_EXPECT_THROWS(
      llm::serialize::parse_hf_config(with(",\"head_dim\":16"), "п", 8));
}

LLM_TEST(Hf, GeneratedConfigIsReadBack) {
  // Наш config.json обязан читаться нашим же разбором: иначе круговая
  // проверка ниже проверяла бы не то, что нужно.
  const llm::nn::ModelConfig source = small_llama();
  const HfConfig parsed = llm::serialize::parse_hf_config(
      llm::serialize::hf_config_json(source), "своё", source.max_seq_len);

  LLM_CHECK_EQ(parsed.model.vocab_size, source.vocab_size);
  LLM_CHECK_EQ(parsed.model.d_model, source.d_model);
  LLM_CHECK_EQ(parsed.model.n_layers, source.n_layers);
  LLM_CHECK_EQ(parsed.model.n_heads, source.n_heads);
  LLM_CHECK_EQ(parsed.model.n_kv_heads, source.n_kv_heads);
  LLM_CHECK_EQ(parsed.model.ffn_hidden, source.ffn_hidden);
  LLM_CHECK(parsed.model == source);

  // Дробные поля обязаны возвращаться тем же числом, а не похожим. Шести
  // значащих цифр — точности ostream по умолчанию — для theta = 1234567 уже
  // не хватает: получилось бы 1.23457e+06, то есть 1234570. Расхождение в
  // theta даёт работающую модель с неправильными ответами.
  llm::nn::ModelConfig awkward = source;
  awkward.rope_theta = 1234567.0f;
  awkward.norm_eps = 1.2345678e-5f;
  const HfConfig read_back = llm::serialize::parse_hf_config(
      llm::serialize::hf_config_json(awkward), "своё", awkward.max_seq_len);
  LLM_CHECK(read_back.model.rope_theta == awkward.rope_theta);
  LLM_CHECK(read_back.model.norm_eps == awkward.norm_eps);
}

LLM_TEST(Hf, WeightsSurviveRoundTripThroughForeignLayout) {
  // Сквозная проверка: модель записывается в чужой раскладке со всеми
  // преобразованиями, читается обратно и обязана дать те же логиты.
  //
  // Чего эта проверка не доказывает: что мы правильно поняли transformers.
  // Прямое и обратное преобразования взаимно обратны, поэтому общая ошибка
  // в них сократилась бы. За соглашение отвечает отдельный тест выше, где
  // поворот половинками написан по формуле.
  const llm::nn::ModelConfig config = small_llama();
  llm::nn::Model source(config, 4242);

  const std::vector<int32_t> ids = {1, 5, 9, 3, 7, 2};
  const llm::Tensor before = source.forward(ids, 2, 3).value().contiguous();

  const std::string path = "/tmp/llm_hf_roundtrip.safetensors";
  const std::vector<NamedTensor> exported =
      llm::serialize::export_hf_weights(&source);
  llm::serialize::write_safetensors(path, exported);

  const HfConfig read = llm::serialize::parse_hf_config(
      llm::serialize::hf_config_json(config), "своё", config.max_seq_len);
  llm::nn::Model restored(read.model, 1);

  const SafeTensors file = SafeTensors::load(path);
  const llm::serialize::HfImportReport report =
      llm::serialize::import_hf_weights(file, read, &restored);
  LLM_CHECK(report.unused.empty());
  LLM_CHECK_EQ(report.values_copied, source.parameter_count());

  const llm::Tensor after = restored.forward(ids, 2, 3).value().contiguous();
  LLM_CHECK_EQ(before.numel(), after.numel());
  for (int64_t i = 0; i < before.numel(); ++i) {
    // Побитово: перекладывание весов не считает ничего, поэтому любое
    // расхождение означает ошибку, а не потерю точности.
    LLM_CHECK_MSG(before.data()[i] == after.data()[i],
                  "логит " << i << " разошёлся: " << before.data()[i]
                           << " против " << after.data()[i]);
  }
  std::remove(path.c_str());
}

LLM_TEST(Hf, RoundTripCatchesForgottenTransposition) {
  // Проверка самой проверки: если бы транспонирование где-то потерялось,
  // сквозной тест обязан это заметить. Подменяем одну матрицу на её
  // транспонированную копию и убеждаемся, что логиты разошлись.
  //
  // Матрица выбрана квадратная (выходная проекция внимания): именно на такой
  // сверка размеров молчит, и поймать ошибку может только сравнение чисел.
  llm::nn::ModelConfig config = small_llama();
  llm::nn::Model source(config, 99);

  const std::vector<int32_t> ids = {2, 4, 6, 8};
  const llm::Tensor before = source.forward(ids, 1, 4).value().contiguous();

  std::vector<NamedTensor> exported =
      llm::serialize::export_hf_weights(&source);
  bool patched = false;
  for (std::size_t i = 0; i < exported.size(); ++i) {
    if (exported[i].name == "model.layers.0.self_attn.o_proj.weight") {
      const llm::Tensor& value = exported[i].value;
      LLM_CHECK_EQ(value.dim(0), value.dim(1));
      llm::Tensor swapped =
          llm::Tensor::uninitialized(llm::Shape({value.dim(0), value.dim(1)}));
      for (int64_t row = 0; row < value.dim(0); ++row) {
        for (int64_t column = 0; column < value.dim(1); ++column) {
          swapped(row, column) = value(column, row);
        }
      }
      exported[i].value = swapped;
      patched = true;
    }
  }
  LLM_CHECK(patched);

  const std::string path = "/tmp/llm_hf_roundtrip_broken.safetensors";
  llm::serialize::write_safetensors(path, exported);

  const HfConfig read = llm::serialize::parse_hf_config(
      llm::serialize::hf_config_json(config), "своё", config.max_seq_len);
  llm::nn::Model restored(read.model, 1);
  llm::serialize::import_hf_weights(SafeTensors::load(path), read, &restored);

  const llm::Tensor after = restored.forward(ids, 1, 4).value().contiguous();
  bool differs = false;
  for (int64_t i = 0; i < before.numel(); ++i) {
    if (before.data()[i] != after.data()[i]) {
      differs = true;
    }
  }
  LLM_CHECK_MSG(differs,
                "подменённая матрица не изменила логиты — значит сквозная "
                "проверка ничего не проверяет");
  std::remove(path.c_str());
}

// Заголовок выровнен так же, как его выравнивает эталонная реализация.
//
// Наш читатель к этому безразличен: он берёт данные по смещению от конца
// заголовка. Но файл, который мы называем safetensors, читают и другие, а те,
// кто отображает его в память, ждут, что массив float начинается с адреса,
// кратного восьми. Эталонная реализация дополняет заголовок пробелами именно
// до этого; не дополнять — значит выкладывать файл, который у половины
// читателей не откроется.
LLM_TEST(Hf, WrittenSafetensorsHeaderIsAligned) {
  const llm::nn::ModelConfig config = small_llama();
  llm::nn::Model model(config, 77);
  const std::vector<NamedTensor> exported =
      llm::serialize::export_hf_weights(&model);

  const std::string path = "test_hf_alignment.safetensors";
  llm::serialize::write_safetensors(path, exported);

  std::ifstream file(path.c_str(), std::ios::binary);
  LLM_CHECK(file.good());
  unsigned char length_bytes[8];
  file.read(reinterpret_cast<char*>(length_bytes), 8);
  LLM_CHECK_EQ(file.gcount(), static_cast<std::streamsize>(8));
  std::uint64_t header_length = 0;
  for (int i = 7; i >= 0; --i) {
    header_length = (header_length << 8) | length_bytes[i];
  }
  LLM_CHECK_MSG((8 + header_length) % 8 == 0,
                "данные начинаются с байта " << 8 + header_length
                                             << " — не кратно восьми");

  // Дополнение — пробелы, и заголовок после них остаётся разбираемым.
  std::string header(static_cast<std::size_t>(header_length), '\0');
  file.read(&header[0], static_cast<std::streamsize>(header_length));
  LLM_CHECK_EQ(header.front(), '{');
  std::size_t last = header.size();
  while (last > 0 && header[last - 1] == ' ') {
    --last;
  }
  LLM_CHECK_EQ(header[last - 1], '}');
  LLM_CHECK_MSG(header.size() - last < 8, "дополнение длиннее семи байт");
  file.close();

  // И файл по-прежнему читается нами целиком.
  const SafeTensors reread = SafeTensors::load(path);
  LLM_CHECK_EQ(reread.size(), exported.size());
  std::remove(path.c_str());
}

// Перестановка каналов требует чётной размерности головы.
//
// При нечётной половина округляется вниз, последний канал головы не
// записывается ни разу, и в результате остаётся неинициализированная память:
// веса выглядят правдоподобно, а одна строка на голову — мусор.
LLM_TEST(Hf, ChannelPermutationRefusesOddHeadDim) {
  llm::Tensor input = llm::Tensor::zeros(llm::Shape({6, 2}));
  LLM_EXPECT_THROWS(llm::serialize::halves_to_pairs(input, 2, 3));
  LLM_EXPECT_THROWS(llm::serialize::pairs_to_halves(input, 2, 3));
  LLM_EXPECT_THROWS(llm::serialize::halves_to_pairs(input, 6, 1));

  // Чётная проходит и остаётся обратимой.
  for (int64_t i = 0; i < input.numel(); ++i) {
    input.data()[i] = static_cast<float>(i);
  }
  const llm::Tensor there = llm::serialize::halves_to_pairs(input, 3, 2);
  const llm::Tensor back = llm::serialize::pairs_to_halves(there, 3, 2);
  for (int64_t i = 0; i < input.numel(); ++i) {
    LLM_CHECK_EQ(back.data()[i], input.data()[i]);
  }
}

// Чтение чужой раскладки отказывается от наших развилок так же, как запись.
//
// Перестановка каналов запросов и ключей осмысленна только при RoPE. Модель с
// обучаемыми позициями прочиталась бы: все имена весов на месте, все размеры
// сходятся, — и получила бы переставленные ни за чем запросы с ключами и
// случайную таблицу позиций. Ни одна проверка размеров этого не заметит.
LLM_TEST(Hf, ImportRefusesModelsWithOurOwnBranches) {
  const llm::nn::ModelConfig plain = small_llama();
  llm::nn::Model source(plain, 4242);
  const std::string path = "test_hf_branches.safetensors";
  llm::serialize::write_safetensors(path,
                                    llm::serialize::export_hf_weights(&source));
  const SafeTensors file = SafeTensors::load(path);
  const HfConfig read = llm::serialize::parse_hf_config(
      llm::serialize::hf_config_json(plain), "своё", plain.max_seq_len);

  llm::nn::ModelConfig learned = plain;
  learned.position = llm::nn::PositionKind::kLearned;
  learned.validate();
  llm::nn::Model other(learned, 1);
  LLM_EXPECT_THROWS(llm::serialize::import_hf_weights(file, read, &other));

  llm::nn::ModelConfig post = plain;
  post.post_norm = true;
  post.validate();
  llm::nn::Model third(post, 1);
  LLM_EXPECT_THROWS(llm::serialize::import_hf_weights(file, read, &third));

  // А обычная по-прежнему читается.
  llm::nn::Model restored(read.model, 1);
  const llm::serialize::HfImportReport report =
      llm::serialize::import_hf_weights(file, read, &restored);
  LLM_CHECK(report.unused.empty());
  std::remove(path.c_str());
}
