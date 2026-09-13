// Конфигурация модели.
//
// Все архитектурные решения собраны в одном месте намеренно: в M7 предстоит
// сравнивать варианты между собой, и переключаться они должны здесь, а не
// правкой кода слоёв.

#ifndef LLM_NN_CONFIG_H_
#define LLM_NN_CONFIG_H_

#include <cstdint>
#include <string>

namespace llm {
namespace nn {

struct ModelConfig {
  int64_t vocab_size = 1024;
  int64_t d_model = 128;
  int64_t n_layers = 4;
  int64_t n_heads = 4;

  // Голов ключей и значений меньше, чем голов запросов (GQA). Каждая голова
  // ключей обслуживает несколько голов запросов, поэтому KV-кэш при генерации
  // получается во столько же раз меньше. При n_kv_heads == n_heads это
  // обычное многоголовое внимание, при n_kv_heads == 1 — MQA.
  int64_t n_kv_heads = 2;

  int64_t max_seq_len = 64;

  // Ширина скрытого слоя FFN. У SwiGLU три матрицы вместо двух, поэтому при
  // равном числе параметров ширину берут около 8/3 от d_model, а не 4, как у
  // обычного FFN с двумя матрицами. Округляется вверх до кратного 32.
  int64_t ffn_hidden = 352;

  float rope_theta = 10000.0f;
  float norm_eps = 1e-5f;
  float init_std = 0.02f;

  // Матрица эмбеддингов служит и выходной проекцией. Экономит vocab * d_model
  // параметров — для маленькой модели это заметная доля.
  bool tie_embeddings = true;

  int64_t head_dim() const { return d_model / n_heads; }
  int64_t kv_dim() const { return n_kv_heads * head_dim(); }

  // Во сколько раз голов запросов больше, чем голов ключей.
  int64_t heads_per_kv() const { return n_heads / n_kv_heads; }

  int64_t parameter_count() const;

  // Проверяет согласованность: делимость размерностей, положительность.
  void validate() const;

  std::string to_string() const;

  static ModelConfig nano();
  static ModelConfig tiny();
  static ModelConfig small();

  // Пресет по имени; падает на неизвестном.
  static ModelConfig by_name(const std::string& name);
};

}  // namespace nn
}  // namespace llm

#endif  // LLM_NN_CONFIG_H_
