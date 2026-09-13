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

// Вид нормировки. RMSNorm делит на корень из среднего квадрата, LayerNorm
// дополнительно вычитает среднее и добавляет свободный член.
enum class NormKind { kRmsNorm, kLayerNorm };

// Как в модель попадает информация о позиции.
//   kRope    — поворот пар координат запроса и ключа, без обучаемых весов;
//   kLearned — обучаемая таблица позиций, прибавляется к эмбеддингам токенов;
//   kNone    — никак. Не бессмыслица: каузальная маска сама по себе задаёт
//              порядок, и интересно, насколько этого хватает.
enum class PositionKind { kRope, kLearned, kNone };

// Устройство FFN. У SwiGLU три матрицы и вентиль, у kGeluMlp — две матрицы и
// обычная активация, как в исходном трансформере.
enum class FfnKind { kSwiGlu, kGeluMlp };

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

  // Архитектурные развилки. Значения по умолчанию — то, как устроена Llama;
  // остальные существуют, чтобы их можно было измерить, а не пересказать.
  NormKind norm = NormKind::kRmsNorm;
  PositionKind position = PositionKind::kRope;
  FfnKind ffn = FfnKind::kSwiGlu;

  // Пост-нормировка: нормируется сумма остатка и подслоя, а не вход подслоя.
  // Так было в исходном трансформере.
  bool post_norm = false;

  // Матрица эмбеддингов служит и выходной проекцией. Экономит vocab * d_model
  // параметров — для маленькой модели это заметная доля.
  bool tie_embeddings = true;

  int64_t head_dim() const { return d_model / n_heads; }
  int64_t kv_dim() const { return n_kv_heads * head_dim(); }

  // Во сколько раз голов запросов больше, чем голов ключей.
  int64_t heads_per_kv() const { return n_heads / n_kv_heads; }

  // Сколько весов у одной нормировки: у LayerNorm вдвое больше из-за
  // свободного члена.
  int64_t norm_parameter_count() const {
    return norm == NormKind::kLayerNorm ? 2 * d_model : d_model;
  }

  // Сколько матриц у FFN.
  int64_t ffn_matrix_count() const { return ffn == FfnKind::kSwiGlu ? 3 : 2; }

  // Ширина FFN, при которой другой его вид даёт столько же параметров.
  // Нужна для честного сравнения: иначе сравнивались бы не устройства FFN, а
  // размеры моделей.
  int64_t ffn_hidden_matching(FfnKind other) const;

  int64_t parameter_count() const;

  // Проверяет согласованность: делимость размерностей, положительность.
  void validate() const;

  std::string to_string() const;

  // Пресет для абляций: вдвое меньше nano, чтобы прогон занимал минуты и
  // каждый вариант можно было повторить с разными зёрнами. Без повторов
  // таблица абляций бессмысленна — непонятно, что считать различием, а что
  // разбросом.
  static ModelConfig ablation();

  static ModelConfig nano();
  static ModelConfig tiny();
  static ModelConfig small();

  // Пресет по имени; падает на неизвестном.
  static ModelConfig by_name(const std::string& name);
};

// Полное сравнение, по всем полям. Нужно чекпоинтам: to_string() показывает
// только форму модели, а веса несовместимы и при расхождении в theta для RoPE
// или в epsilon нормировки — форма при этом совпадает, и подмена прошла бы
// незамеченной.
bool operator==(const ModelConfig& lhs, const ModelConfig& rhs);
bool operator!=(const ModelConfig& lhs, const ModelConfig& rhs);

}  // namespace nn
}  // namespace llm

#endif  // LLM_NN_CONFIG_H_
