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
  LLM_CHECK_MSG(dropout >= 0.0f && dropout < 1.0f,
                "вероятность дропаута " << dropout
                                        << " должна быть в [0, 1); единица "
                                           "занулила бы оба подслоя целиком");

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

  // Вещественные величины проверяются по той же причине, что и целые: ошибка в
  // них не падает, а даёт NaN где-то в середине прямого прохода, и искать его
  // придётся от конца к началу.
  //
  // Строго больше нуля именно у eps: нормировка делит на корень из суммы
  // квадратов плюс eps, и при нулевом eps строка из одних нулей даёт
  // бесконечность. У основания поворота — тоже: степень с нулевым или
  // отрицательным основанием не определена на дробных показателях, а они здесь
  // как раз дробные.
  LLM_CHECK_MSG(norm_eps > 0.0f,
                "eps нормировки " << norm_eps
                                  << " должен быть положительным: при нуле "
                                     "нулевая строка даёт бесконечность");
  LLM_CHECK_MSG(rope_theta > 0.0f,
                "основание поворота " << rope_theta
                                      << " должно быть положительным");
  // Ноль допустим: это вырожденная, но осмысленная инициализация нулями.
  LLM_CHECK_MSG(init_std >= 0.0f, "разброс инициализации " << init_std
                                                           << " отрицателен");
  LLM_CHECK_MSG(z_loss_coef >= 0.0f,
                "коэффициент z-loss " << z_loss_coef
                                      << " отрицателен: штраф стал бы "
                                         "поощрением");
}

int64_t ModelConfig::ffn_hidden_matching(FfnKind other) const {
  // Хотим, чтобы d * target * h_new совпало с d * current * h_current: иначе
  // сравнивались бы не устройства FFN, а размеры моделей.
  //
  // Округления здесь нет намеренно. Выравнивание ширины ничего не даёт — наш
  // GEMM одинаково работает с любыми размерами, — а вот округление до
  // кратного 32 легко съедает всю точность подгонки: 36 превратилось бы в 64,
  // то есть модель выросла бы в полтора раза, и сравнение перестало бы быть
  // сравнением устройств.
  const int64_t current = ffn_matrix_count();
  const int64_t target = other == FfnKind::kSwiGlu ? 3 : 2;
  // Деление с округлением к ближайшему: точное совпадение получается, когда
  // ширина делится нацело, что верно для всех пресетов проекта.
  return (ffn_hidden * current + target / 2) / target;
}

int64_t ModelConfig::parameter_count() const {
  int64_t total = vocab_size * d_model;  // таблица эмбеддингов
  if (!tie_embeddings) {
    total += vocab_size * d_model;  // отдельная выходная проекция
  }
  if (position == PositionKind::kLearned) {
    total += max_seq_len * d_model;  // таблица позиций
  }

  const int64_t attention = d_model * d_model     // Q
                            + d_model * kv_dim()  // K
                            + d_model * kv_dim()  // V
                            + d_model * d_model;  // O
  const int64_t ffn_weights = ffn_matrix_count() * d_model * ffn_hidden;
  const int64_t norms = 2 * norm_parameter_count() + qk_norm_parameter_count();

  total += n_layers * (attention + ffn_weights + norms);
  total += norm_parameter_count();  // финальная нормировка
  return total;
}

std::string ModelConfig::to_string() const {
  std::ostringstream oss;
  oss << "vocab=" << vocab_size << " d_model=" << d_model
      << " layers=" << n_layers << " heads=" << n_heads << "/" << n_kv_heads
      << " head_dim=" << head_dim() << " ffn=" << ffn_hidden
      << " ctx=" << max_seq_len;
  oss << " norm=" << (norm == NormKind::kRmsNorm ? "rms" : "layer");
  oss << (post_norm ? "/post" : "/pre");
  oss << " pos=";
  if (position == PositionKind::kRope) {
    oss << "rope";
  } else if (position == PositionKind::kLearned) {
    oss << "learned";
  } else {
    oss << "none";
  }
  oss << " ffn_kind=" << (ffn == FfnKind::kSwiGlu ? "swiglu" : "gelu");
  if (qk_norm) {
    oss << " qk_norm";
  }
  if (z_loss_coef > 0.0f) {
    oss << " z_loss=" << z_loss_coef;
  }
  if (dropout > 0.0f) {
    oss << " dropout=" << dropout;
  }
  oss << (tie_embeddings ? " tied" : " untied");
  oss << " params=" << parameter_count();
  return oss.str();
}

ModelConfig ModelConfig::ablation() {
  ModelConfig config;
  config.vocab_size = 1024;
  config.d_model = 96;
  config.n_layers = 3;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.max_seq_len = 64;
  config.ffn_hidden = 256;
  config.validate();
  return config;
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

bool operator==(const ModelConfig& lhs, const ModelConfig& rhs) {
  return lhs.vocab_size == rhs.vocab_size && lhs.d_model == rhs.d_model &&
         lhs.n_layers == rhs.n_layers && lhs.n_heads == rhs.n_heads &&
         lhs.n_kv_heads == rhs.n_kv_heads &&
         lhs.max_seq_len == rhs.max_seq_len &&
         lhs.ffn_hidden == rhs.ffn_hidden && lhs.rope_theta == rhs.rope_theta &&
         lhs.norm_eps == rhs.norm_eps && lhs.init_std == rhs.init_std &&
         lhs.tie_embeddings == rhs.tie_embeddings && lhs.norm == rhs.norm &&
         lhs.position == rhs.position && lhs.ffn == rhs.ffn &&
         lhs.post_norm == rhs.post_norm && lhs.qk_norm == rhs.qk_norm &&
         lhs.z_loss_coef == rhs.z_loss_coef && lhs.dropout == rhs.dropout;
}

bool operator!=(const ModelConfig& lhs, const ModelConfig& rhs) {
  return !(lhs == rhs);
}

ModelConfig ModelConfig::by_name(const std::string& name) {
  if (name == "ablation") {
    return ablation();
  }
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
                           << name
                           << "', доступны: ablation, nano, tiny, small");
  return nano();
}

}  // namespace nn
}  // namespace llm
