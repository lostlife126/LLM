#include "nn/config.h"

#include <sstream>

#include "core/check.h"

namespace llm {
namespace nn {

void ModelConfig::validate() const {
  LLM_CHECK_GT(vocab_size, static_cast<int64_t>(0));
  LLM_CHECK_GT(d_model, static_cast<int64_t>(0));
  LLM_CHECK_GT(n_layers, static_cast<int64_t>(0));
  LLM_CHECK_GT(n_heads, static_cast<int64_t>(0));
  LLM_CHECK_GT(n_kv_heads, static_cast<int64_t>(0));
  LLM_CHECK_GT(max_seq_len, static_cast<int64_t>(0));
  LLM_CHECK_GT(ffn_hidden, static_cast<int64_t>(0));

  LLM_CHECK_MSG(
      d_model % n_heads == 0,
      "d_model " << d_model << " не делится на число голов " << n_heads);
  LLM_CHECK_MSG(n_heads % n_kv_heads == 0,
                "голов запросов " << n_heads
                                  << " должно быть кратно числу голов ключей "
                                  << n_kv_heads);
  // RoPE поворачивает пары координат, поэтому размерность головы обязана быть
  // чётной.
  LLM_CHECK_MSG(head_dim() % 2 == 0,
                "размерность головы " << head_dim() << " должна быть чётной");
}

int64_t ModelConfig::parameter_count() const {
  int64_t total = vocab_size * d_model;  // таблица эмбеддингов
  if (!tie_embeddings) {
    total += vocab_size * d_model;  // отдельная выходная проекция
  }

  const int64_t attention = d_model * d_model     // Q
                            + d_model * kv_dim()  // K
                            + d_model * kv_dim()  // V
                            + d_model * d_model;  // O
  const int64_t ffn = 3 * d_model * ffn_hidden;   // gate, up, down
  const int64_t norms = 2 * d_model;  // две нормировки в блоке

  total += n_layers * (attention + ffn + norms);
  total += d_model;  // финальная нормировка
  return total;
}

std::string ModelConfig::to_string() const {
  std::ostringstream oss;
  oss << "vocab=" << vocab_size << " d_model=" << d_model
      << " layers=" << n_layers << " heads=" << n_heads << "/" << n_kv_heads
      << " head_dim=" << head_dim() << " ffn=" << ffn_hidden
      << " ctx=" << max_seq_len << " params=" << parameter_count();
  return oss.str();
}

ModelConfig ModelConfig::nano() {
  ModelConfig config;
  config.vocab_size = 1024;
  config.d_model = 128;
  config.n_layers = 4;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.max_seq_len = 64;
  config.ffn_hidden = 352;
  config.validate();
  return config;
}

ModelConfig ModelConfig::tiny() {
  ModelConfig config;
  config.vocab_size = 4096;
  config.d_model = 256;
  config.n_layers = 4;
  config.n_heads = 8;
  config.n_kv_heads = 2;
  config.max_seq_len = 128;
  config.ffn_hidden = 704;
  config.validate();
  return config;
}

ModelConfig ModelConfig::small() {
  ModelConfig config;
  config.vocab_size = 8192;
  config.d_model = 384;
  config.n_layers = 6;
  config.n_heads = 6;
  config.n_kv_heads = 2;
  config.max_seq_len = 256;
  config.ffn_hidden = 1024;
  config.validate();
  return config;
}

ModelConfig ModelConfig::by_name(const std::string& name) {
  if (name == "nano") {
    return nano();
  }
  if (name == "tiny") {
    return tiny();
  }
  if (name == "small") {
    return small();
  }
  LLM_CHECK_MSG(false, "неизвестный пресет '"
                           << name << "', доступны: nano, tiny, small");
  return nano();
}

}  // namespace nn
}  // namespace llm
