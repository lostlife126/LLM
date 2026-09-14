#include "serialize/hf_config.h"

#include <algorithm>
#include <fstream>
#include <sstream>

#include "core/check.h"
#include "serialize/json.h"

namespace llm {
namespace serialize {
namespace {

// Поля, которые ничего не меняют в вычислении: сведения о том, чем модель
// выложена, что с ней делать дальше и как её обучали. Список явный, потому что
// смысл проверки в обратном — всё, чего здесь нет, обязано быть разобрано или
// остановить загрузку.
const char* const kHarmlessFields[] = {
    "architectures",
    "attention_dropout",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "unk_token_id",
    "initializer_range",  // только для обучения с нуля
    "pretraining_tp",  // разбиение при обучении, на веса не влияет
    "torch_dtype",  // тип в файле берётся из самого файла
    "transformers_version",  //
    "use_cache",  // относится к реализации, не к модели
    "_name_or_path",            //
    "model_type",               // проверяется отдельно
    "max_position_embeddings",  // окно задаём мы, см. заголовок
    "is_llama_config",          // помета SmolLM
    "rope_interleaved",         // проверяется отдельно
    "mlp_bias",                 // проверяется отдельно
    "attention_bias",           // проверяется отдельно
    "head_dim",                 // проверяется отдельно
    "tie_word_embeddings",  //
    "hidden_act",           // проверяется отдельно
    "rope_scaling",         // проверяется отдельно
};

bool is_harmless(const std::string& name) {
  const std::size_t count =
      sizeof(kHarmlessFields) / sizeof(kHarmlessFields[0]);
  for (std::size_t i = 0; i < count; ++i) {
    if (name == kHarmlessFields[i]) {
      return true;
    }
  }
  // Служебные поля transformers начинаются с подчёркивания.
  return !name.empty() && name[0] == '_';
}

bool truthy(const JsonValue& value) {
  if (value.is_bool()) {
    return value.boolean();
  }
  if (value.is_number()) {
    return value.number() != 0.0;
  }
  return false;
}

}  // namespace

HfConfig parse_hf_config(const std::string& text, const std::string& origin,
                         int64_t max_seq_len) {
  const JsonDocument document = JsonDocument::parse(text);
  const JsonValue root = document.root();
  LLM_CHECK_MSG(root.is_object(), origin << ": config.json не объект");

  HfConfig result;
  nn::ModelConfig& model = result.model;

  // --- то, что мы умеем ---

  const std::string model_type =
      root.has("model_type") ? root.field("model_type").text() : std::string();
  LLM_CHECK_MSG(model_type == "llama",
                origin << ": model_type=\"" << model_type
                       << "\", а прочитать мы умеем только \"llama\". Другие "
                          "семейства отличаются не только именами весов: "
                          "например, у Gemma нормировка умножает на (1 + вес), "
                          "а у Qwen2 к проекциям внимания добавлены свободные "
                          "члены");

  if (root.has("architectures")) {
    const JsonValue architectures = root.field("architectures");
    if (architectures.is_array() && architectures.size() > 0) {
      result.architecture = architectures.at(0).text();
    }
  }

  model.vocab_size = root.field("vocab_size").integer();
  model.d_model = root.field("hidden_size").integer();
  model.n_layers = root.field("num_hidden_layers").integer();
  model.n_heads = root.field("num_attention_heads").integer();
  model.n_kv_heads = root.has("num_key_value_heads")
                         ? root.field("num_key_value_heads").integer()
                         : model.n_heads;
  model.ffn_hidden = root.field("intermediate_size").integer();

  model.norm_eps = static_cast<float>(root.field("rms_norm_eps").number());
  model.rope_theta = root.has("rope_theta")
                         ? static_cast<float>(root.field("rope_theta").number())
                         : 10000.0f;

  result.tie_embeddings = root.has("tie_word_embeddings") &&
                          truthy(root.field("tie_word_embeddings"));
  model.tie_embeddings = result.tie_embeddings;

  if (root.has("bos_token_id") && root.field("bos_token_id").is_number()) {
    result.bos_token_id = root.field("bos_token_id").integer();
  }
  if (root.has("eos_token_id") && root.field("eos_token_id").is_number()) {
    result.eos_token_id = root.field("eos_token_id").integer();
  }

  // Развилки архитектуры у Llama зафиксированы, и менять их нам не нужно —
  // но записать стоит явно, чтобы было видно, с чем именно сверяемся.
  model.norm = nn::NormKind::kRmsNorm;
  model.position = nn::PositionKind::kRope;
  model.ffn = nn::FfnKind::kSwiGlu;
  model.post_norm = false;
  model.qk_norm = false;
  model.z_loss_coef = 0.0f;

  LLM_CHECK_MSG(max_seq_len > 0, "окно должно быть положительным");
  model.max_seq_len = max_seq_len;

  // --- то, чего мы не умеем: отказ с указанием поля ---

  const std::string activation = root.has("hidden_act")
                                     ? root.field("hidden_act").text()
                                     : std::string("silu");
  LLM_CHECK_MSG(
      activation == "silu" || activation == "swish",
      origin << ": hidden_act=\"" << activation << "\", а у нас SwiGLU с silu");

  LLM_CHECK_MSG(
      !root.has("attention_bias") || !truthy(root.field("attention_bias")),
      origin << ": attention_bias=true. Свободные члены у проекций "
                "внимания у нас не предусмотрены, и без них веса "
                "останутся правильными, а ответы — нет");
  LLM_CHECK_MSG(
      !root.has("mlp_bias") || !truthy(root.field("mlp_bias")),
      origin << ": mlp_bias=true, а у нас линейные слои без свободных членов");

  LLM_CHECK_MSG(
      !root.has("rope_scaling") || root.field("rope_scaling").is_null(),
      origin << ": задано rope_scaling. Это растяжение позиций ради "
                "длинного контекста, и без него модель на длинных "
                "входах поведёт себя не так, как задумано");

  // rope_interleaved — то самое различие в раскладке каналов, из-за которого
  // веса запросов и ключей приходится переставлять. Llama в transformers
  // считает половинками; если модель объявляет чередование, перестановка не
  // нужна, и наш импорт дал бы неверный результат.
  LLM_CHECK_MSG(
      !root.has("rope_interleaved") || !truthy(root.field("rope_interleaved")),
      origin << ": rope_interleaved=true — такая модель хранит "
                "каналы головы уже в нашем порядке, и перестановка "
                "при импорте её испортит");

  // У нас размерность головы выводится делением, представить иную нельзя.
  LLM_CHECK_MSG(model.n_heads > 0 && model.d_model % model.n_heads == 0,
                origin << ": hidden_size " << model.d_model
                       << " не делится на num_attention_heads "
                       << model.n_heads);
  if (root.has("head_dim") && root.field("head_dim").is_number()) {
    const int64_t declared = root.field("head_dim").integer();
    LLM_CHECK_MSG(declared == model.head_dim(),
                  origin << ": head_dim=" << declared
                         << ", а у нас он выводится делением и равен "
                         << model.head_dim());
  }
  LLM_CHECK_MSG(model.n_kv_heads > 0 && model.n_heads % model.n_kv_heads == 0,
                origin << ": num_attention_heads " << model.n_heads
                       << " не делится на num_key_value_heads "
                       << model.n_kv_heads);
  LLM_CHECK_MSG(model.head_dim() % 2 == 0,
                origin << ": размерность головы " << model.head_dim()
                       << " нечётна, RoPE так не работает");

  // --- всё остальное ---

  for (std::size_t i = 0; i < root.size(); ++i) {
    const std::string name = root.key_at(i);
    if (name == "vocab_size" || name == "hidden_size" ||
        name == "num_hidden_layers" || name == "num_attention_heads" ||
        name == "num_key_value_heads" || name == "intermediate_size" ||
        name == "rms_norm_eps" || name == "rope_theta") {
      continue;
    }
    if (is_harmless(name)) {
      result.ignored.push_back(name);
      continue;
    }
    LLM_CHECK_MSG(false,
                  origin << ": в config.json поле \"" << name
                         << "\", смысл которого нам неизвестен. Пропустить "
                            "его молча нельзя: если оно меняет вычисление, "
                            "модель загрузится и будет выдавать связный мусор");
  }

  return result;
}

HfConfig read_hf_config(const std::string& path, int64_t max_seq_len) {
  std::ifstream file(path.c_str(), std::ios::binary);
  LLM_CHECK_MSG(file.good(), "не удалось открыть " << path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return parse_hf_config(buffer.str(), path, max_seq_len);
}

}  // namespace serialize
}  // namespace llm
