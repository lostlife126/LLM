#include "nn/model.h"

#include <cmath>
#include <sstream>

#include "autograd/nn.h"
#include "autograd/ops.h"
#include "core/check.h"

namespace llm {
namespace nn {
namespace {

using autograd::Var;

Tensor normal_tensor(const Shape& shape, float stddev, Rng* rng) {
  Tensor tensor = Tensor::uninitialized(shape);
  Span<float> values = tensor.flat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = rng->normal(0.0f, stddev);
  }
  return tensor;
}

// Веса выходных проекций инициализируются меньшими, чем остальные.
//
// Каждый слой добавляет в остаточный поток два вклада — от внимания и от FFN.
// Без поправки дисперсия потока растёт пропорционально глубине, и к последнему
// слою активации выходят из рабочего диапазона. Деление на sqrt(2 * n_layers)
// держит дисперсию примерно постоянной по глубине.
float residual_init_std(const ModelConfig& config) {
  return config.init_std /
         std::sqrt(2.0f * static_cast<float>(config.n_layers));
}

}  // namespace

Linear::Linear(int64_t in_features, int64_t out_features, float init_std,
               Rng* rng)
    : weight_(Var::leaf(
          normal_tensor(Shape({in_features, out_features}), init_std, rng),
          true, "linear.weight")) {}

Var Linear::forward(const Var& input) const {
  return autograd::matmul(input, weight_);
}

void Linear::collect(const std::string& prefix,
                     std::vector<NamedParameter>* out) {
  NamedParameter parameter;
  parameter.name = prefix + ".weight";
  parameter.value = &weight_;
  out->push_back(parameter);
}

Attention::Attention(const ModelConfig& config, Rng* rng) : config_(config) {
  const int64_t d_model = config.d_model;
  query_ =
      Linear(d_model, config.n_heads * config.head_dim(), config.init_std, rng);
  key_ = Linear(d_model, config.kv_dim(), config.init_std, rng);
  value_ = Linear(d_model, config.kv_dim(), config.init_std, rng);
  output_ = Linear(d_model, d_model, residual_init_std(config), rng);
}

Var Attention::split_heads(const Var& input, int64_t batch, int64_t seq,
                           int64_t heads) const {
  // (b, t, heads * hd) -> (b, heads, t, hd).
  //
  // Головы — это просто разбиение выходной размерности одной проекции, а не
  // отдельные матрицы. Перестановка нужна, чтобы позиция оказалась
  // предпоследней осью: этого требуют и RoPE, и умножение запросов на ключи.
  const Var reshaped =
      autograd::reshape(input, Shape({batch, seq, heads, config_.head_dim()}));
  return autograd::permute(reshaped, {0, 2, 1, 3});
}

Var repeat_kv(const Var& input, int64_t batch, int64_t kv_heads,
              int64_t repeats, int64_t seq, int64_t head_dim) {
  if (repeats == 1) {
    return input;  // обычное многоголовое внимание, размножать нечего
  }
  // Голова ключей повторяется подряд: голова запроса h пользуется головой
  // ключей h / repeats. Альтернативная раскладка (циклическая, h % kv_heads)
  // при обучении с нуля равноценна, но с чужими весами несовместима.
  const Var wide =
      autograd::reshape(input, Shape({batch, kv_heads, 1, seq, head_dim}));
  const Var repeated =
      autograd::expand(wide, Shape({batch, kv_heads, repeats, seq, head_dim}));
  return autograd::reshape(repeated,
                           Shape({batch, kv_heads * repeats, seq, head_dim}));
}

Var Attention::forward(const Var& input, int64_t position_offset,
                       KvCache* cache, int64_t layer) const {
  LLM_CHECK_MSG(
      input.shape().rank() == 3,
      "внимание ждёт (батч, позиция, канал), получено " << input.shape());
  const int64_t batch = input.shape().dim(0);
  const int64_t seq = input.shape().dim(1);
  const int64_t heads = config_.n_heads;
  const int64_t head_dim = config_.head_dim();

  Var queries = split_heads(query_.forward(input), batch, seq, heads);
  Var keys = split_heads(key_.forward(input), batch, seq, config_.n_kv_heads);
  Var values =
      split_heads(value_.forward(input), batch, seq, config_.n_kv_heads);

  // Поворот применяется к запросам и ключам, но не к значениям: позиция должна
  // влиять на то, куда смотреть, а не на то, что оттуда взять.
  queries = autograd::rope(queries, position_offset, config_.rope_theta);
  keys = autograd::rope(keys, position_offset, config_.rope_theta);

  // Ключей и значений может быть больше, чем запросов: при генерации запрос
  // один, а ключи накоплены за весь предыдущий текст.
  int64_t key_length = seq;
  if (cache != nullptr) {
    LLM_CHECK_MSG(!autograd::grad_enabled(),
                  "KV-кэш несовместим с построением ленты: градиент через "
                  "сохранённые ключи не идёт");
    Tensor cached_keys;
    Tensor cached_values;
    cache->append(layer, keys.value(), values.value(), &cached_keys,
                  &cached_values);
    keys = Var::constant(cached_keys);
    values = Var::constant(cached_values);
    key_length = cached_keys.shape().dim(2);
  }

  keys = repeat_kv(keys, batch, config_.n_kv_heads, config_.heads_per_kv(),
                   key_length, head_dim);
  values = repeat_kv(values, batch, config_.n_kv_heads, config_.heads_per_kv(),
                     key_length, head_dim);

  // Умножение матриц работает по последним двум осям, поэтому батч и головы
  // сворачиваются в одну ось: голова — такой же независимый пример, как
  // элемент батча.
  queries = autograd::reshape(queries, Shape({batch * heads, seq, head_dim}));
  keys = autograd::reshape(keys, Shape({batch * heads, key_length, head_dim}));
  values =
      autograd::reshape(values, Shape({batch * heads, key_length, head_dim}));

  // Деление на корень из размерности головы удерживает дисперсию оценок
  // около единицы. Без него при большой размерности softmax насыщается,
  // превращаясь в почти детерминированный выбор, и градиент исчезает.
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  Var scores = autograd::mul_scalar(
      autograd::matmul(queries, autograd::transpose(keys, -2, -1)), scale);

  scores = autograd::causal_mask(scores, position_offset);
  const Var weights = autograd::softmax(scores);
  Var result = autograd::matmul(weights, values);

  result = autograd::reshape(result, Shape({batch, heads, seq, head_dim}));
  result = autograd::permute(result, {0, 2, 1, 3});
  result = autograd::reshape(result, Shape({batch, seq, heads * head_dim}));
  return output_.forward(result);
}

void Attention::collect(const std::string& prefix,
                        std::vector<NamedParameter>* out) {
  query_.collect(prefix + ".query", out);
  key_.collect(prefix + ".key", out);
  value_.collect(prefix + ".value", out);
  output_.collect(prefix + ".output", out);
}

Mlp::Mlp(const ModelConfig& config, Rng* rng) {
  gate_ = Linear(config.d_model, config.ffn_hidden, config.init_std, rng);
  up_ = Linear(config.d_model, config.ffn_hidden, config.init_std, rng);
  down_ =
      Linear(config.ffn_hidden, config.d_model, residual_init_std(config), rng);
}

Var Mlp::forward(const Var& input) const {
  const Var gate = autograd::silu(gate_.forward(input));
  const Var up = up_.forward(input);
  return down_.forward(autograd::mul(gate, up));
}

void Mlp::collect(const std::string& prefix, std::vector<NamedParameter>* out) {
  gate_.collect(prefix + ".gate", out);
  up_.collect(prefix + ".up", out);
  down_.collect(prefix + ".down", out);
}

Block::Block(const ModelConfig& config, Rng* rng)
    : config_(config),
      // Масштаб нормировки начинается с единицы: на старте нормировка ничего
      // не меняет, кроме самой нормализации.
      attention_norm_(Var::leaf(Tensor::full(Shape({config.d_model}), 1.0f),
                                true, "attention_norm")),
      mlp_norm_(Var::leaf(Tensor::full(Shape({config.d_model}), 1.0f), true,
                          "mlp_norm")),
      attention_(config, rng),
      mlp_(config, rng) {}

Var Block::forward(const Var& input, int64_t position_offset, KvCache* cache,
                   int64_t layer) const {
  const Var attended = attention_.forward(
      autograd::rms_norm(input, attention_norm_, config_.norm_eps),
      position_offset, cache, layer);
  const Var after_attention = autograd::add(input, attended);

  const Var transformed = mlp_.forward(
      autograd::rms_norm(after_attention, mlp_norm_, config_.norm_eps));
  return autograd::add(after_attention, transformed);
}

void Block::collect(const std::string& prefix,
                    std::vector<NamedParameter>* out) {
  NamedParameter attention_norm;
  attention_norm.name = prefix + ".attention_norm";
  attention_norm.value = &attention_norm_;
  out->push_back(attention_norm);

  attention_.collect(prefix + ".attention", out);

  NamedParameter mlp_norm;
  mlp_norm.name = prefix + ".mlp_norm";
  mlp_norm.value = &mlp_norm_;
  out->push_back(mlp_norm);

  mlp_.collect(prefix + ".mlp", out);
}

Model::Model(const ModelConfig& config, uint64_t seed) : config_(config) {
  config_.validate();
  Rng rng(seed);

  token_embedding_ =
      Var::leaf(normal_tensor(Shape({config.vocab_size, config.d_model}),
                              config.init_std, &rng),
                true, "token_embedding");

  for (int64_t layer = 0; layer < config.n_layers; ++layer) {
    blocks_.push_back(Block(config, &rng));
  }

  final_norm_ = Var::leaf(Tensor::full(Shape({config.d_model}), 1.0f), true,
                          "final_norm");

  if (!config.tie_embeddings) {
    lm_head_ = Linear(config.d_model, config.vocab_size, config.init_std, &rng);
  }
}

Var Model::forward(const std::vector<int32_t>& ids, int64_t batch, int64_t seq,
                   int64_t position_offset, KvCache* cache) const {
  LLM_CHECK_MSG(static_cast<int64_t>(ids.size()) == batch * seq,
                "передано " << ids.size() << " токенов при батче " << batch
                            << " и длине " << seq);
  LLM_CHECK_MSG(position_offset + seq <= config_.max_seq_len,
                "позиции до " << position_offset + seq
                              << " выходят за контекст "
                              << config_.max_seq_len);

  // Эмбеддинги отдают (batch * seq, d_model) — форму с батчем восстанавливаем
  // сами, чтобы слой эмбеддингов не знал про батчи.
  Var hidden = autograd::reshape(autograd::embedding(token_embedding_, ids),
                                 Shape({batch, seq, config_.d_model}));

  for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
    hidden = blocks_[layer].forward(hidden, position_offset, cache,
                                    static_cast<int64_t>(layer));
  }
  // Длина кэша сдвигается один раз, после всех слоёв: они дописывают один и
  // тот же блок позиций, и сдвиг внутри цикла сбил бы отсчёт для следующего
  // слоя.
  if (cache != nullptr) {
    cache->advance(seq);
  }
  hidden = autograd::rms_norm(hidden, final_norm_, config_.norm_eps);

  if (config_.tie_embeddings) {
    // Та же матрица, что и на входе, только транспонированная. Строка таблицы
    // задаёт и representation токена на входе, и направление, близость к
    // которому даёт высокий логит на выходе.
    return autograd::matmul(hidden,
                            autograd::transpose(token_embedding_, 0, 1));
  }
  return lm_head_.forward(hidden);
}

Var Model::forward_last(const std::vector<int32_t>& ids, int64_t batch,
                        int64_t seq, int64_t position_offset,
                        KvCache* cache) const {
  const Var logits = forward(ids, batch, seq, position_offset, cache);
  // Генерации нужна только последняя позиция. Остальные логиты посчитаны зря
  // лишь при обработке затравки; на шаге генерации позиция и так одна.
  const Var last = autograd::slice(logits, 1, seq - 1, 1);
  return autograd::reshape(last, Shape({batch, config_.vocab_size}));
}

Var Model::loss(const std::vector<int32_t>& ids, int64_t batch,
                int64_t seq) const {
  LLM_CHECK_MSG(seq >= 2,
                "для предсказания следующего токена нужно хотя бы два");

  const Var logits = forward(ids, batch, seq);
  // Последняя позиция отбрасывается: следующего токена для неё в окне нет.
  const Var predictions = autograd::slice(logits, 1, 0, seq - 1);
  const Var flat = autograd::reshape(
      predictions, Shape({batch * (seq - 1), config_.vocab_size}));

  std::vector<int32_t> targets;
  targets.reserve(static_cast<std::size_t>(batch * (seq - 1)));
  for (int64_t item = 0; item < batch; ++item) {
    for (int64_t position = 1; position < seq; ++position) {
      targets.push_back(ids[static_cast<std::size_t>(item * seq + position)]);
    }
  }
  return autograd::cross_entropy(flat, targets);
}

std::vector<NamedParameter> Model::parameters() {
  std::vector<NamedParameter> result;

  NamedParameter embedding;
  embedding.name = "token_embedding";
  embedding.value = &token_embedding_;
  result.push_back(embedding);

  for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
    std::ostringstream prefix;
    prefix << "block." << layer;
    blocks_[layer].collect(prefix.str(), &result);
  }

  NamedParameter final_norm;
  final_norm.name = "final_norm";
  final_norm.value = &final_norm_;
  result.push_back(final_norm);

  if (!config_.tie_embeddings) {
    lm_head_.collect("lm_head", &result);
  }
  return result;
}

int64_t Model::parameter_count() {
  int64_t total = 0;
  const std::vector<NamedParameter> all = parameters();
  for (std::size_t i = 0; i < all.size(); ++i) {
    total += all[i].value->numel();
  }
  return total;
}

}  // namespace nn
}  // namespace llm
