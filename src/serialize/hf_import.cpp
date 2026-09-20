#include "serialize/hf_import.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "core/check.h"

namespace llm {
namespace serialize {
namespace {

std::string layer_prefix(int64_t layer) {
  std::ostringstream out;
  out << "model.layers." << layer << ".";
  return out.str();
}

// Транспонирование (rows, columns) -> (columns, rows).
//
// Отдельной функцией, а не через наши виды с перестановкой осей: результат
// должен быть плотным, потому что дальше он копируется в веса побайтно.
Tensor transpose_matrix(const Tensor& input) {
  LLM_CHECK_EQ(input.rank(), 2);
  const int64_t rows = input.dim(0);
  const int64_t columns = input.dim(1);
  Tensor out = Tensor::uninitialized(Shape({columns, rows}));
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t column = 0; column < columns; ++column) {
      out(column, row) = input(row, column);
    }
  }
  return out;
}

void copy_into(const Tensor& source, autograd::Var* destination,
               const std::string& name, int64_t* values_copied) {
  Tensor& target = destination->value();
  LLM_CHECK_MSG(source.rank() == target.rank(),
                "у " << name << " ранг " << source.rank() << ", а нужен "
                     << target.rank());
  for (int axis = 0; axis < source.rank(); ++axis) {
    LLM_CHECK_MSG(source.dim(axis) == target.dim(axis),
                  "у " << name << " ось " << axis << " равна "
                       << source.dim(axis) << ", а нужна " << target.dim(axis));
  }
  const Tensor dense = source.contiguous();
  std::copy(dense.data(), dense.data() + dense.numel(), target.data());
  *values_copied += dense.numel();
}

// Ищет параметр модели по имени. Имена у нас складываются из префиксов, и
// опечатка в префиксе дала бы «параметр не найден» на ровном месте, поэтому
// сообщение показывает, что вообще есть.
autograd::Var* find_parameter(std::vector<nn::NamedParameter>* parameters,
                              const std::string& name) {
  for (std::size_t i = 0; i < parameters->size(); ++i) {
    if ((*parameters)[i].name == name) {
      return (*parameters)[i].value;
    }
  }
  std::ostringstream available;
  for (std::size_t i = 0; i < parameters->size() && i < 6; ++i) {
    available << (i == 0 ? "" : ", ") << (*parameters)[i].name;
  }
  LLM_CHECK_MSG(false, "в модели нет параметра "
                           << name << "; есть: " << available.str() << ", ...");
  return nullptr;
}

}  // namespace

// Перестановка каналов внутри каждой головы: половинки -> соседние пары.
//
// Вход — матрица (heads * head_dim, in_features) в раскладке transformers,
// то есть до транспонирования. Строка — это выходной канал, и переставляются
// именно строки.
Tensor halves_to_pairs(const Tensor& input, int64_t heads, int64_t head_dim) {
  LLM_CHECK_EQ(input.rank(), 2);
  LLM_CHECK_EQ(input.dim(0), heads * head_dim);
  // Нечётная размерность головы — не «почти то же самое». Половина при делении
  // округлится вниз, последний канал не запишется ни разу, и в результате
  // останется мусор из неинициализированной памяти: перестановка молча выдаст
  // правдоподобные веса с одной испорченной строкой на голову.
  LLM_CHECK_MSG(head_dim > 0 && head_dim % 2 == 0,
                "размерность головы " << head_dim
                                      << " должна быть чётной и положительной");
  const int64_t width = input.dim(1);
  const int64_t half = head_dim / 2;

  Tensor out = Tensor::uninitialized(input.shape());
  for (int64_t head = 0; head < heads; ++head) {
    const int64_t base = head * head_dim;
    for (int64_t pair = 0; pair < half; ++pair) {
      for (int64_t column = 0; column < width; ++column) {
        // Наш канал 2*pair — их канал pair; наш 2*pair+1 — их pair + half.
        out(base + 2 * pair, column) = input(base + pair, column);
        out(base + 2 * pair + 1, column) = input(base + half + pair, column);
      }
    }
  }
  return out;
}

// Обратная перестановка: соседние пары -> половинки.
Tensor pairs_to_halves(const Tensor& input, int64_t heads, int64_t head_dim) {
  LLM_CHECK_EQ(input.rank(), 2);
  LLM_CHECK_EQ(input.dim(0), heads * head_dim);
  LLM_CHECK_MSG(head_dim > 0 && head_dim % 2 == 0,
                "размерность головы " << head_dim
                                      << " должна быть чётной и положительной");
  const int64_t width = input.dim(1);
  const int64_t half = head_dim / 2;

  Tensor out = Tensor::uninitialized(input.shape());
  for (int64_t head = 0; head < heads; ++head) {
    const int64_t base = head * head_dim;
    for (int64_t pair = 0; pair < half; ++pair) {
      for (int64_t column = 0; column < width; ++column) {
        out(base + pair, column) = input(base + 2 * pair, column);
        out(base + half + pair, column) = input(base + 2 * pair + 1, column);
      }
    }
  }
  return out;
}

HfImportReport import_hf_weights(const SafeTensors& file,
                                 const HfConfig& config, nn::Model* model) {
  LLM_CHECK(model != nullptr);
  const nn::ModelConfig& shape = model->config();
  // То же условие, что и у записи. Раскладка Llama описывает ровно один набор
  // развилок, и перестановка каналов ниже осмысленна только при RoPE: модели с
  // обучаемыми позициями она переставила бы запросы и ключи ни за чем, а
  // таблицу позиций оставила бы случайной — и ни одна проверка размеров этого
  // бы не заметила.
  LLM_CHECK_MSG(shape.norm == nn::NormKind::kRmsNorm &&
                    shape.position == nn::PositionKind::kRope &&
                    shape.ffn == nn::FfnKind::kSwiGlu && !shape.post_norm &&
                    !shape.qk_norm,
                "из раскладки Llama читаются только модели без наших "
                "дополнительных развилок");
  LLM_CHECK_MSG(shape.vocab_size == config.model.vocab_size &&
                    shape.d_model == config.model.d_model &&
                    shape.n_layers == config.model.n_layers &&
                    shape.n_heads == config.model.n_heads &&
                    shape.n_kv_heads == config.model.n_kv_heads &&
                    shape.ffn_hidden == config.model.ffn_hidden,
                "модель создана не по тому config.json");

  std::vector<nn::NamedParameter> parameters = model->parameters();
  HfImportReport report;
  std::vector<std::string> used;

  const int64_t head_dim = shape.head_dim();

  // Эмбеддинги: (vocab, hidden) у них, (vocab, d_model) у нас — совпадает.
  // Это единственная матрица, которую не надо транспонировать: она не
  // линейный слой, а таблица строк.
  {
    const Tensor table = file.read("model.embed_tokens.weight");
    copy_into(table, find_parameter(&parameters, "token_embedding"),
              "model.embed_tokens.weight", &report.values_copied);
    used.push_back("model.embed_tokens.weight");
  }

  for (int64_t layer = 0; layer < shape.n_layers; ++layer) {
    const std::string source = layer_prefix(layer);
    std::ostringstream ours;
    ours << "block." << layer << ".";
    const std::string target = ours.str();

    const std::string names[2][2] = {
        {source + "input_layernorm.weight", target + "attention_norm.weight"},
        {source + "post_attention_layernorm.weight",
         target + "mlp_norm.weight"},
    };
    for (int i = 0; i < 2; ++i) {
      const Tensor value = file.read(names[i][0]);
      copy_into(value, find_parameter(&parameters, names[i][1]), names[i][0],
                &report.values_copied);
      used.push_back(names[i][0]);
    }

    // Запросы и ключи: перестановка каналов, потом транспонирование.
    // Порядок важен — перестановка описана в терминах строк исходной
    // матрицы, то есть выходных каналов.
    {
      const Tensor query = file.read(source + "self_attn.q_proj.weight");
      copy_into(
          transpose_matrix(halves_to_pairs(query, shape.n_heads, head_dim)),
          find_parameter(&parameters, target + "attention.query.weight"),
          source + "self_attn.q_proj.weight", &report.values_copied);
      used.push_back(source + "self_attn.q_proj.weight");
    }
    {
      const Tensor key = file.read(source + "self_attn.k_proj.weight");
      copy_into(
          transpose_matrix(halves_to_pairs(key, shape.n_kv_heads, head_dim)),
          find_parameter(&parameters, target + "attention.key.weight"),
          source + "self_attn.k_proj.weight", &report.values_copied);
      used.push_back(source + "self_attn.k_proj.weight");
    }

    // Значения и выходная проекция не вращаются — только транспонирование.
    const std::string plain[3][2] = {
        {source + "self_attn.v_proj.weight", target + "attention.value.weight"},
        {source + "self_attn.o_proj.weight",
         target + "attention.output.weight"},
        {source + "mlp.down_proj.weight", target + "mlp.down.weight"},
    };
    const std::string more[2][2] = {
        {source + "mlp.gate_proj.weight", target + "mlp.gate.weight"},
        {source + "mlp.up_proj.weight", target + "mlp.up.weight"},
    };
    for (int i = 0; i < 3; ++i) {
      const Tensor value = file.read(plain[i][0]);
      copy_into(transpose_matrix(value),
                find_parameter(&parameters, plain[i][1]), plain[i][0],
                &report.values_copied);
      used.push_back(plain[i][0]);
    }
    for (int i = 0; i < 2; ++i) {
      const Tensor value = file.read(more[i][0]);
      copy_into(transpose_matrix(value),
                find_parameter(&parameters, more[i][1]), more[i][0],
                &report.values_copied);
      used.push_back(more[i][0]);
    }
  }

  {
    const Tensor value = file.read("model.norm.weight");
    copy_into(value, find_parameter(&parameters, "final_norm.weight"),
              "model.norm.weight", &report.values_copied);
    used.push_back("model.norm.weight");
  }

  if (!config.tie_embeddings) {
    const Tensor value = file.read("lm_head.weight");
    copy_into(transpose_matrix(value),
              find_parameter(&parameters, "lm_head.weight"), "lm_head.weight",
              &report.values_copied);
    used.push_back("lm_head.weight");
  }

  report.tensors_used = static_cast<int64_t>(used.size());

  // Чего не тронули. Обычно это lm_head.weight у модели со связанными
  // эмбеддингами, записанный на всякий случай, либо rotary_emb.inv_freq —
  // таблица частот, которую мы считаем сами. Всё прочее означает, что в
  // модели есть что-то, чего мы не заметили, и об этом надо знать.
  const std::vector<std::string> all = file.names();
  for (std::size_t i = 0; i < all.size(); ++i) {
    if (std::find(used.begin(), used.end(), all[i]) == used.end()) {
      report.unused.push_back(all[i]);
    }
  }
  return report;
}

std::vector<NamedTensor> export_hf_weights(nn::Model* model) {
  LLM_CHECK(model != nullptr);
  const nn::ModelConfig& shape = model->config();
  LLM_CHECK_MSG(shape.norm == nn::NormKind::kRmsNorm &&
                    shape.position == nn::PositionKind::kRope &&
                    shape.ffn == nn::FfnKind::kSwiGlu && !shape.post_norm &&
                    !shape.qk_norm,
                "в раскладке Llama записываются только модели без наших "
                "дополнительных развилок");

  std::vector<nn::NamedParameter> parameters = model->parameters();
  const int64_t head_dim = shape.head_dim();
  std::vector<NamedTensor> out;

  auto take = [&parameters](const std::string& name) -> Tensor {
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      if (parameters[i].name == name) {
        return parameters[i].value->value();
      }
    }
    LLM_CHECK_MSG(false, "в модели нет параметра " << name);
    return Tensor();
  };

  NamedTensor embedding;
  embedding.name = "model.embed_tokens.weight";
  embedding.value = take("token_embedding").contiguous();
  out.push_back(embedding);

  for (int64_t layer = 0; layer < shape.n_layers; ++layer) {
    const std::string destination = layer_prefix(layer);
    std::ostringstream ours;
    ours << "block." << layer << ".";
    const std::string source = ours.str();

    NamedTensor input_norm;
    input_norm.name = destination + "input_layernorm.weight";
    input_norm.value = take(source + "attention_norm.weight").contiguous();
    out.push_back(input_norm);

    NamedTensor query;
    query.name = destination + "self_attn.q_proj.weight";
    query.value = pairs_to_halves(
        transpose_matrix(take(source + "attention.query.weight")),
        shape.n_heads, head_dim);
    out.push_back(query);

    NamedTensor key;
    key.name = destination + "self_attn.k_proj.weight";
    key.value =
        pairs_to_halves(transpose_matrix(take(source + "attention.key.weight")),
                        shape.n_kv_heads, head_dim);
    out.push_back(key);

    NamedTensor value;
    value.name = destination + "self_attn.v_proj.weight";
    value.value = transpose_matrix(take(source + "attention.value.weight"));
    out.push_back(value);

    NamedTensor output;
    output.name = destination + "self_attn.o_proj.weight";
    output.value = transpose_matrix(take(source + "attention.output.weight"));
    out.push_back(output);

    NamedTensor post_norm;
    post_norm.name = destination + "post_attention_layernorm.weight";
    post_norm.value = take(source + "mlp_norm.weight").contiguous();
    out.push_back(post_norm);

    NamedTensor gate;
    gate.name = destination + "mlp.gate_proj.weight";
    gate.value = transpose_matrix(take(source + "mlp.gate.weight"));
    out.push_back(gate);

    NamedTensor up;
    up.name = destination + "mlp.up_proj.weight";
    up.value = transpose_matrix(take(source + "mlp.up.weight"));
    out.push_back(up);

    NamedTensor down;
    down.name = destination + "mlp.down_proj.weight";
    down.value = transpose_matrix(take(source + "mlp.down.weight"));
    out.push_back(down);
  }

  NamedTensor final_norm;
  final_norm.name = "model.norm.weight";
  final_norm.value = take("final_norm.weight").contiguous();
  out.push_back(final_norm);

  if (!shape.tie_embeddings) {
    NamedTensor head;
    head.name = "lm_head.weight";
    head.value = transpose_matrix(take("lm_head.weight"));
    out.push_back(head);
  }
  return out;
}

namespace {

// Запись float, которая читается обратно тем же числом.
//
// По умолчанию ostream печатает шесть значащих цифр, и для 1e-5 или 10000
// этого хватает с запасом. Но hf_config_json — половина круговой проверки:
// модель записывается в чужом формате, читается обратно и обязана дать те же
// логиты до последнего бита. Значение вроде theta = 1234567 напечаталось бы
// как 1.23457e+06 и вернулось бы другим числом — а расхождение в theta даёт
// работающую модель с неправильными ответами, о чём сказано в заголовке.
//
// Цифры добавляются по одной, пока число не начнёт читаться обратно точно.
// Девяти хватает любому float, но обычно хватает и шести, так что привычная
// запись остаётся привычной: «1e-05», а не «1.00000001e-05».
std::string exact_float(float value) {
  for (int digits = 6; digits <= 9; ++digits) {
    std::ostringstream out;
    out << std::setprecision(digits) << value;
    const std::string text = out.str();
    if (static_cast<float>(std::strtod(text.c_str(), nullptr)) == value) {
      return text;
    }
  }
  std::ostringstream out;
  out << std::setprecision(9) << value;
  return out.str();
}

}  // namespace

std::string hf_config_json(const nn::ModelConfig& config) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"architectures\": [\"LlamaForCausalLM\"],\n";
  out << "  \"attention_bias\": false,\n";
  out << "  \"hidden_act\": \"silu\",\n";
  out << "  \"hidden_size\": " << config.d_model << ",\n";
  out << "  \"intermediate_size\": " << config.ffn_hidden << ",\n";
  out << "  \"max_position_embeddings\": " << config.max_seq_len << ",\n";
  out << "  \"mlp_bias\": false,\n";
  out << "  \"model_type\": \"llama\",\n";
  out << "  \"num_attention_heads\": " << config.n_heads << ",\n";
  out << "  \"num_hidden_layers\": " << config.n_layers << ",\n";
  out << "  \"num_key_value_heads\": " << config.n_kv_heads << ",\n";
  out << "  \"rms_norm_eps\": " << exact_float(config.norm_eps) << ",\n";
  out << "  \"rope_theta\": " << exact_float(config.rope_theta) << ",\n";
  out << "  \"tie_word_embeddings\": "
      << (config.tie_embeddings ? "true" : "false") << ",\n";
  out << "  \"vocab_size\": " << config.vocab_size << "\n";
  out << "}\n";
  return out.str();
}

void write_safetensors(const std::string& path,
                       const std::vector<NamedTensor>& tensors) {
  std::ostringstream header;
  header << "{";
  uint64_t offset = 0;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& value = tensors[i].value;
    if (i != 0) {
      header << ",";
    }
    header << "\"" << tensors[i].name << "\":{\"dtype\":\"F32\",\"shape\":[";
    for (int axis = 0; axis < value.rank(); ++axis) {
      header << (axis == 0 ? "" : ",") << value.dim(axis);
    }
    const uint64_t bytes = static_cast<uint64_t>(value.numel()) * 4;
    header << "],\"data_offsets\":[" << offset << "," << offset + bytes << "]}";
    offset += bytes;
  }
  header << "}";

  // Заголовок дополняется пробелами так, чтобы данные начинались с адреса,
  // кратного восьми. Наш читатель к этому безразличен — он берёт данные по
  // смещению, — но эталонная реализация формата дополняет именно так, а
  // читатели, отображающие файл в память, на невыровненном float спотыкаются.
  // Файл, который мы называем safetensors, должен читаться не только нами.
  std::string header_text = header.str();
  const std::size_t padding = (8 - ((header_text.size() + 8) % 8)) % 8;
  header_text.append(padding, ' ');

  std::ofstream file(path.c_str(), std::ios::binary);
  LLM_CHECK_MSG(file.good(), "не удалось создать " << path);

  const uint64_t length = header_text.size();
  char bytes[8];
  for (int i = 0; i < 8; ++i) {
    bytes[i] = static_cast<char>((length >> (8 * i)) & 0xFFu);
  }
  file.write(bytes, 8);
  file.write(header_text.data(),
             static_cast<std::streamsize>(header_text.size()));

  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const Tensor dense = tensors[i].value.contiguous();
    file.write(reinterpret_cast<const char*>(dense.data()),
               static_cast<std::streamsize>(dense.numel() * sizeof(float)));
  }
  LLM_CHECK_MSG(file.good(), "запись " << path << " не удалась");
}

}  // namespace serialize
}  // namespace llm
