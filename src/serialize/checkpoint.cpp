#include "serialize/checkpoint.h"

#include <cstring>
#include <fstream>
#include <vector>

#include "core/check.h"

namespace llm {
namespace serialize {
namespace {

const uint32_t kMagic = 0x574d4c4cu;  // "LLMW" в little-endian
const uint32_t kVersion =
    4;  // 2 — развилки, 3 — QK-норма и z-loss, 4 — дропаут

// Метка представления чисел. Если файл читают на машине с другим порядком
// байтов или другим форматом float, эти два значения не совпадут, и загрузка
// откажется вместо того, чтобы выдать правдоподобный мусор.
const uint32_t kEndianMarker = 0x01020304u;
const float kFloatMarker = 1.5f;

template <typename T>
void write_pod(std::ofstream* file, const T& value) {
  file->write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_pod(std::ifstream* file, const std::string& path) {
  T value;
  file->read(reinterpret_cast<char*>(&value), sizeof(T));
  LLM_CHECK_MSG(file->good(), "файл " << path << " оборвался при чтении");
  return value;
}

void write_string(std::ofstream* file, const std::string& text) {
  write_pod<uint32_t>(file, static_cast<uint32_t>(text.size()));
  file->write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string read_string(std::ifstream* file, const std::string& path) {
  const uint32_t length = read_pod<uint32_t>(file, path);
  LLM_CHECK_MSG(length < (1u << 20), "неправдоподобная длина имени в " << path);
  std::string text(length, '\0');
  file->read(&text[0], static_cast<std::streamsize>(length));
  LLM_CHECK_MSG(file->good(), "файл " << path << " оборвался при чтении имени");
  return text;
}

// Развилка архитектуры, прочитанная из файла. Предел проверяется здесь, а не
// там, где развилка понадобится: дальше выбор делается сравнением с одним
// значением («это LayerNorm?»), и незнакомый байт молча означал бы другую
// ветвь. Чекпоинт с испорченным байтом загрузился бы и считал бы не то, что в
// нём записано, — ровно тот случай, ради которого в заголовке стоят метки
// порядка байтов и формата float.
uint8_t read_choice(std::ifstream* file, const std::string& path,
                    const char* what, uint8_t limit) {
  const uint8_t value = read_pod<uint8_t>(file, path);
  LLM_CHECK_MSG(value < limit, "в чекпоинте "
                                   << path << " развилка " << what << " равна "
                                   << static_cast<int>(value)
                                   << ", а известны значения от 0 до "
                                   << static_cast<int>(limit - 1));
  return value;
}

void write_config(std::ofstream* file, const nn::ModelConfig& config) {
  write_pod<int64_t>(file, config.vocab_size);
  write_pod<int64_t>(file, config.d_model);
  write_pod<int64_t>(file, config.n_layers);
  write_pod<int64_t>(file, config.n_heads);
  write_pod<int64_t>(file, config.n_kv_heads);
  write_pod<int64_t>(file, config.max_seq_len);
  write_pod<int64_t>(file, config.ffn_hidden);
  write_pod<float>(file, config.rope_theta);
  write_pod<float>(file, config.norm_eps);
  write_pod<float>(file, config.init_std);
  write_pod<uint8_t>(file, config.tie_embeddings ? 1 : 0);
  write_pod<uint8_t>(file, static_cast<uint8_t>(config.norm));
  write_pod<uint8_t>(file, static_cast<uint8_t>(config.position));
  write_pod<uint8_t>(file, static_cast<uint8_t>(config.ffn));
  write_pod<uint8_t>(file, config.post_norm ? 1 : 0);
  write_pod<uint8_t>(file, config.qk_norm ? 1 : 0);
  write_pod<float>(file, config.z_loss_coef);
  write_pod<float>(file, config.dropout);
}

// Версия 1 не знала про архитектурные развилки. Их значения по умолчанию —
// ровно те, с которыми она и писалась (RMSNorm, pre-norm, RoPE, SwiGLU),
// поэтому старые чекпоинты читаются без потерь, и переобучать модель из-за
// смены формата не нужно.
nn::ModelConfig read_config_body(std::ifstream* file, const std::string& path,
                                 uint32_t version) {
  nn::ModelConfig config;
  config.vocab_size = read_pod<int64_t>(file, path);
  config.d_model = read_pod<int64_t>(file, path);
  config.n_layers = read_pod<int64_t>(file, path);
  config.n_heads = read_pod<int64_t>(file, path);
  config.n_kv_heads = read_pod<int64_t>(file, path);
  config.max_seq_len = read_pod<int64_t>(file, path);
  config.ffn_hidden = read_pod<int64_t>(file, path);
  config.rope_theta = read_pod<float>(file, path);
  config.norm_eps = read_pod<float>(file, path);
  config.init_std = read_pod<float>(file, path);
  config.tie_embeddings = read_pod<uint8_t>(file, path) != 0;
  if (version >= 2) {
    config.norm =
        static_cast<nn::NormKind>(read_choice(file, path, "нормировки", 2));
    config.position =
        static_cast<nn::PositionKind>(read_choice(file, path, "позиций", 3));
    config.ffn = static_cast<nn::FfnKind>(read_choice(file, path, "FFN", 2));
    config.post_norm = read_pod<uint8_t>(file, path) != 0;
  }
  if (version >= 3) {
    config.qk_norm = read_pod<uint8_t>(file, path) != 0;
    config.z_loss_coef = read_pod<float>(file, path);
  }
  if (version >= 4) {
    // Дропаут на веса не влияет вовсе — он существует только при обучении, —
    // поэтому старые чекпоинты читаются с нулём и остаются правильными.
    config.dropout = read_pod<float>(file, path);
  }
  config.validate();
  return config;
}

// Возвращает версию формата: она нужна дальше, чтобы решить, есть ли в файле
// поля архитектурных развилок.
uint32_t read_header(std::ifstream* file, const std::string& path) {
  LLM_CHECK_MSG(read_pod<uint32_t>(file, path) == kMagic,
                "файл " << path << " не является чекпоинтом");
  const uint32_t version = read_pod<uint32_t>(file, path);
  LLM_CHECK_MSG(version >= 1 && version <= kVersion,
                "версия чекпоинта "
                    << version
                    << " не поддержана: эта сборка читает версии с 1 по "
                    << kVersion);
  LLM_CHECK_MSG(read_pod<uint32_t>(file, path) == kEndianMarker,
                "чекпоинт " << path << " записан с другим порядком байтов");
  LLM_CHECK_MSG(read_pod<float>(file, path) == kFloatMarker,
                "чекпоинт " << path << " записан с другим форматом float");
  return version;
}

}  // namespace

void save_checkpoint(const std::string& path, nn::Model* model, int64_t step) {
  LLM_CHECK(model != nullptr);
  std::ofstream file(path.c_str(), std::ios::binary);
  LLM_CHECK_MSG(file.good(), "не удалось открыть для записи: " << path);

  write_pod<uint32_t>(&file, kMagic);
  write_pod<uint32_t>(&file, kVersion);
  write_pod<uint32_t>(&file, kEndianMarker);
  write_pod<float>(&file, kFloatMarker);
  write_config(&file, model->config());
  write_pod<int64_t>(&file, step);

  const std::vector<nn::NamedParameter> parameters = model->parameters();
  write_pod<uint32_t>(&file, static_cast<uint32_t>(parameters.size()));

  for (std::size_t i = 0; i < parameters.size(); ++i) {
    const Tensor& value = parameters[i].value->value();
    LLM_CHECK(value.is_contiguous());
    write_string(&file, parameters[i].name);
    write_pod<uint32_t>(&file, static_cast<uint32_t>(value.rank()));
    for (int axis = 0; axis < value.rank(); ++axis) {
      write_pod<int64_t>(&file, value.dim(axis));
    }
    file.write(reinterpret_cast<const char*>(value.data()),
               static_cast<std::streamsize>(value.numel() * sizeof(float)));
  }
  LLM_CHECK_MSG(file.good(), "ошибка записи в " << path);
}

nn::ModelConfig read_config(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::binary);
  LLM_CHECK_MSG(file.good(), "не удалось открыть: " << path);
  const uint32_t version = read_header(&file, path);
  return read_config_body(&file, path, version);
}

int64_t load_checkpoint(const std::string& path, nn::Model* model) {
  LLM_CHECK(model != nullptr);
  std::ifstream file(path.c_str(), std::ios::binary);
  LLM_CHECK_MSG(file.good(), "не удалось открыть: " << path);

  const uint32_t version = read_header(&file, path);
  const nn::ModelConfig stored = read_config_body(&file, path, version);
  const nn::ModelConfig& current = model->config();
  // Сравнение по всем полям, а не по to_string(): та показывает только форму
  // модели, а веса несовместимы и при расхождении в theta для RoPE или в
  // epsilon нормировки, когда форма совпадает.
  LLM_CHECK_MSG(stored == current,
                "чекпоинт описывает другую модель:\n  в файле: "
                    << stored.to_string()
                    << "\n  в памяти: " << current.to_string());

  const int64_t step = read_pod<int64_t>(&file, path);
  const uint32_t count = read_pod<uint32_t>(&file, path);

  std::vector<nn::NamedParameter> parameters = model->parameters();
  LLM_CHECK_MSG(count == parameters.size(),
                "в чекпоинте " << count << " тензоров, а модели нужно "
                               << parameters.size());

  for (uint32_t i = 0; i < count; ++i) {
    const std::string name = read_string(&file, path);
    LLM_CHECK_MSG(name == parameters[i].name,
                  "порядок параметров разошёлся: в файле '"
                      << name << "', ожидался '" << parameters[i].name << "'");

    Tensor& value = parameters[i].value->value();
    const uint32_t rank = read_pod<uint32_t>(&file, path);
    LLM_CHECK_MSG(rank == static_cast<uint32_t>(value.rank()),
                  "у параметра " << name << " в файле ранг " << rank);
    for (uint32_t axis = 0; axis < rank; ++axis) {
      const int64_t size = read_pod<int64_t>(&file, path);
      LLM_CHECK_MSG(size == value.dim(static_cast<int>(axis)),
                    "у параметра " << name << " разошлась ось " << axis);
    }

    // Та же проверка, что и при записи. Читается сплошной кусок длиной в
    // numel, и для разреженного вида это означало бы запись мимо: часть
    // значений легла бы в чужие ячейки, а часть весов осталась бы прежней.
    // Молча — формы-то совпали.
    LLM_CHECK_MSG(value.is_contiguous(),
                  "параметр " << name << " лежит не сплошным куском");
    file.read(reinterpret_cast<char*>(value.data()),
              static_cast<std::streamsize>(value.numel() * sizeof(float)));
    LLM_CHECK_MSG(file.good(),
                  "файл " << path << " оборвался на параметре " << name);
  }
  return step;
}

}  // namespace serialize
}  // namespace llm
