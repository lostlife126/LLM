#include "nn/model.h"

#include <cmath>
#include <sstream>

#include "autograd/nn.h"
#include "autograd/ops.h"
#include "core/check.h"
#include "nn/dropout.h"
#include "ops/elementwise.h"
#include "ops/loss.h"
#include "ops/matmul.h"
#include "ops/nn.h"

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

Norm::Norm(const ModelConfig& config, Rng* rng)
    : kind_(config.norm), eps_(config.norm_eps) {
  // Генератор не используется: нормировка стартует с тождественного
  // преобразования, а не со случайных весов. Аргумент оставлен ради
  // единообразия с остальными слоями.
  (void)rng;
  weight_ = Var::leaf(Tensor::full(Shape({config.d_model}), 1.0f), true,
                      "norm.weight");
  if (kind_ == NormKind::kLayerNorm) {
    bias_ =
        Var::leaf(Tensor::zeros(Shape({config.d_model})), true, "norm.bias");
  }
}

Var Norm::forward(const Var& input) const {
  if (kind_ == NormKind::kLayerNorm) {
    return autograd::layer_norm(input, weight_, bias_, eps_);
  }
  return autograd::rms_norm(input, weight_, eps_);
}

void Norm::freeze() {
  weight_ = Var::constant(weight_.value());
  if (kind_ == NormKind::kLayerNorm) {
    bias_ = Var::constant(bias_.value());
  }
}

void Norm::collect(const std::string& prefix,
                   std::vector<NamedParameter>* out) {
  NamedParameter weight;
  weight.name = prefix + ".weight";
  weight.value = &weight_;
  out->push_back(weight);
  if (kind_ == NormKind::kLayerNorm) {
    NamedParameter bias;
    bias.name = prefix + ".bias";
    bias.value = &bias_;
    out->push_back(bias);
  }
}

Linear::Linear(int64_t in_features, int64_t out_features, float init_std,
               Rng* rng)
    : weight_(Var::leaf(
          normal_tensor(Shape({in_features, out_features}), init_std, rng),
          true, "linear.weight")) {}

Var Linear::forward(const Var& input) const {
  const Var base = autograd::matmul(input, weight_);
  if (!has_lora_) {
    return base;
  }
  // Порядок умножения существен: (x * A) * B требует двух умножений на узкие
  // матрицы, а (A * B) сначала построило бы полную матрицу размера
  // вход x выход — то есть ровно ту экономию, ради которой всё затевалось, и
  // потеряло бы.
  const Var adapter =
      autograd::matmul(autograd::matmul(input, lora_a_), lora_b_);
  return autograd::add(base, autograd::mul_scalar(adapter, lora_scale_));
}

void Linear::freeze() {
  // Значение остаётся, узел графа исчезает: градиент до веса больше не
  // доходит, и оптимизатор его не увидит.
  weight_ = Var::constant(weight_.value());
}

void Linear::enable_lora(int64_t rank, float alpha, Rng* rng) {
  LLM_CHECK_GT(rank, static_cast<int64_t>(0));
  const int64_t in_features = weight_.shape().dim(0);
  const int64_t out_features = weight_.shape().dim(1);
  LLM_CHECK_MSG(rank < in_features && rank < out_features,
                "ранг " << rank << " не меньше размеров матрицы "
                        << weight_.shape() << " — экономии не будет");

  lora_a_ = Var::leaf(normal_tensor(Shape({in_features, rank}), 0.02f, rng),
                      true, "lora_a");
  // B начинается с нуля: пока она нулевая, адаптер не даёт вклада, и модель в
  // начале дообучения в точности совпадает с исходной.
  lora_b_ =
      Var::leaf(Tensor::zeros(Shape({rank, out_features})), true, "lora_b");
  lora_scale_ = alpha / static_cast<float>(rank);
  has_lora_ = true;
}

void Linear::merge_lora() {
  if (!has_lora_) {
    return;
  }
  // W += scale * A * B. После этого адаптер не нужен: он растворён в весе.
  const Tensor product = ops::matmul(lora_a_.value(), lora_b_.value());
  const Tensor scaled = ops::mul_scalar(product, lora_scale_);
  Tensor updated = weight_.value();
  ops::add_into(scaled, &updated);

  const bool trainable = weight_.requires_grad();
  weight_ = trainable ? Var::leaf(updated, true, "linear.weight")
                      : Var::constant(updated);
  lora_a_ = Var();
  lora_b_ = Var();
  lora_scale_ = 0.0f;
  has_lora_ = false;
}

void Linear::collect(const std::string& prefix,
                     std::vector<NamedParameter>* out) {
  NamedParameter parameter;
  parameter.name = prefix + ".weight";
  parameter.value = &weight_;
  out->push_back(parameter);

  if (has_lora_) {
    NamedParameter a;
    a.name = prefix + ".lora_a";
    a.value = &lora_a_;
    out->push_back(a);

    NamedParameter b;
    b.name = prefix + ".lora_b";
    b.value = &lora_b_;
    out->push_back(b);
  }
}

Attention::Attention(const ModelConfig& config, Rng* rng) : config_(config) {
  const int64_t d_model = config.d_model;
  query_ =
      Linear(d_model, config.n_heads * config.head_dim(), config.init_std, rng);
  key_ = Linear(d_model, config.kv_dim(), config.init_std, rng);
  value_ = Linear(d_model, config.kv_dim(), config.init_std, rng);
  output_ = Linear(d_model, d_model, residual_init_std(config), rng);

  if (config.qk_norm) {
    // Масштабы начинаются с единицы: на старте нормировка только делит на
    // длину и ничего больше не меняет.
    query_norm_ = Var::leaf(Tensor::full(Shape({config.head_dim()}), 1.0f),
                            true, "query_norm");
    key_norm_ = Var::leaf(Tensor::full(Shape({config.head_dim()}), 1.0f), true,
                          "key_norm");
  }
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
                       KvCache* cache, int64_t layer,
                       ForwardStats* stats) const {
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

  if (config_.qk_norm) {
    // Нормировка идёт по размерности головы и до поворота. Порядок важен:
    // поворот сохраняет длину, а масштаб нормировки — нет, и переставив их
    // местами, мы получили бы другое преобразование.
    //
    // Смысл в том, что после нормировки длина запроса и ключа ограничена, и
    // скалярное произведение не может уехать настолько, чтобы softmax
    // насытился, а градиент через него исчез.
    queries = autograd::rms_norm(queries, query_norm_, config_.norm_eps);
    keys = autograd::rms_norm(keys, key_norm_, config_.norm_eps);
  }

  // Поворот применяется к запросам и ключам, но не к значениям: позиция должна
  // влиять на то, куда смотреть, а не на то, что оттуда взять.
  //
  // При других способах кодирования позиции внимание остаётся полностью
  // симметричным по позициям, и порядок задаётся только каузальной маской.
  if (config_.position == PositionKind::kRope) {
    queries = autograd::rope(queries, position_offset, config_.rope_theta);
    keys = autograd::rope(keys, position_offset, config_.rope_theta);
  }

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

  // Масштаб, маска и softmax — одной операцией. Порознь это шесть проходов по
  // тензору оценок, слитно четыре и только по видимой половине строки.
  // Результат совпадает с раздельной цепочкой побитово, и это проверяется
  // тестом: иначе замена потребовала бы пересчитать все числа обучения.
  const Var weights = autograd::masked_softmax(
      autograd::matmul(queries, autograd::transpose(keys, -2, -1)), scale,
      position_offset);
  if (stats != nullptr) {
    stats->attention_entropy.push_back(
        ops::attention_entropy(weights.value(), position_offset));
  }
  Var result = autograd::matmul(weights, values);

  result = autograd::reshape(result, Shape({batch, heads, seq, head_dim}));
  result = autograd::permute(result, {0, 2, 1, 3});
  result = autograd::reshape(result, Shape({batch, seq, heads * head_dim}));
  return output_.forward(result);
}

void Attention::enable_lora(const LoraConfig& config, Rng* rng) {
  if (config.attention_query) {
    query_.enable_lora(config.rank, config.alpha, rng);
  }
  if (config.attention_key) {
    key_.enable_lora(config.rank, config.alpha, rng);
  }
  if (config.attention_value) {
    value_.enable_lora(config.rank, config.alpha, rng);
  }
  if (config.attention_output) {
    output_.enable_lora(config.rank, config.alpha, rng);
  }
}

void Attention::merge_lora() {
  query_.merge_lora();
  key_.merge_lora();
  value_.merge_lora();
  output_.merge_lora();
}

void Attention::freeze() {
  if (config_.qk_norm) {
    query_norm_ = Var::constant(query_norm_.value());
    key_norm_ = Var::constant(key_norm_.value());
  }
  query_.freeze();
  key_.freeze();
  value_.freeze();
  output_.freeze();
}

void Attention::collect(const std::string& prefix,
                        std::vector<NamedParameter>* out) {
  if (config_.qk_norm) {
    NamedParameter query_norm;
    query_norm.name = prefix + ".query_norm";
    query_norm.value = &query_norm_;
    out->push_back(query_norm);

    NamedParameter key_norm;
    key_norm.name = prefix + ".key_norm";
    key_norm.value = &key_norm_;
    out->push_back(key_norm);
  }
  query_.collect(prefix + ".query", out);
  key_.collect(prefix + ".key", out);
  value_.collect(prefix + ".value", out);
  output_.collect(prefix + ".output", out);
}

Mlp::Mlp(const ModelConfig& config, Rng* rng) : kind_(config.ffn) {
  if (kind_ == FfnKind::kSwiGlu) {
    gate_ = Linear(config.d_model, config.ffn_hidden, config.init_std, rng);
  }
  up_ = Linear(config.d_model, config.ffn_hidden, config.init_std, rng);
  down_ =
      Linear(config.ffn_hidden, config.d_model, residual_init_std(config), rng);
}

Var Mlp::forward(const Var& input, ForwardStats* stats) const {
  if (kind_ == FfnKind::kSwiGlu) {
    // Вентиль решает, какие каналы пропустить дальше, и решение зависит от
    // входа. У обычного FFN такого выбора нет: активация применяется к
    // каждому каналу одинаково.
    const Var raw_gate = gate_.forward(input);
    if (stats != nullptr) {
      // Доля каналов, которые вентиль пропускает дальше. Схлопывание к нулю
      // или к единице означало бы, что вентиль перестал выбирать и FFN
      // выродился в обычный.
      stats->gate_open_fraction.push_back(
          ops::positive_fraction(raw_gate.value()));
    }
    const Var gate = autograd::silu(raw_gate);
    const Var up = up_.forward(input);
    return down_.forward(autograd::mul(gate, up));
  }
  return down_.forward(autograd::gelu(up_.forward(input)));
}

void Mlp::enable_lora(const LoraConfig& config, Rng* rng) {
  if (!config.ffn) {
    return;
  }
  if (kind_ == FfnKind::kSwiGlu) {
    gate_.enable_lora(config.rank, config.alpha, rng);
  }
  up_.enable_lora(config.rank, config.alpha, rng);
  down_.enable_lora(config.rank, config.alpha, rng);
}

void Mlp::merge_lora() {
  if (kind_ == FfnKind::kSwiGlu) {
    gate_.merge_lora();
  }
  up_.merge_lora();
  down_.merge_lora();
}

void Mlp::freeze() {
  if (kind_ == FfnKind::kSwiGlu) {
    gate_.freeze();
  }
  up_.freeze();
  down_.freeze();
}

void Mlp::collect(const std::string& prefix, std::vector<NamedParameter>* out) {
  if (kind_ == FfnKind::kSwiGlu) {
    gate_.collect(prefix + ".gate", out);
  }
  up_.collect(prefix + ".up", out);
  down_.collect(prefix + ".down", out);
}

Block::Block(const ModelConfig& config, Rng* rng)
    : config_(config),
      attention_norm_(config, rng),
      mlp_norm_(config, rng),
      attention_(config, rng),
      mlp_(config, rng) {}

Var Block::forward(const Var& input, int64_t position_offset, KvCache* cache,
                   int64_t layer, ForwardStats* stats) const {
  Var output;
  if (config_.post_norm) {
    // Нормируется уже сумма остатка и подслоя, то есть нормировка стоит на
    // пути остатка.
    const Var attended =
        dropout(attention_.forward(input, position_offset, cache, layer, stats),
                config_.dropout);
    const Var after_attention =
        attention_norm_.forward(autograd::add(input, attended));
    const Var transformed =
        dropout(mlp_.forward(after_attention, stats), config_.dropout);
    output = mlp_norm_.forward(autograd::add(after_attention, transformed));
  } else {
    // Пре-нормировка: путь остатка от выхода к входу — чистое сложение.
    // Дропаут стоит на выходе подслоя, до сложения с остатком. Это
    // общепринятое место: путь остатка остаётся чистым, а зануляется только
    // поправка, которую подслой к нему добавляет.
    const Var attended =
        dropout(attention_.forward(attention_norm_.forward(input),
                                   position_offset, cache, layer, stats),
                config_.dropout);
    const Var after_attention = autograd::add(input, attended);
    const Var transformed =
        dropout(mlp_.forward(mlp_norm_.forward(after_attention), stats),
                config_.dropout);
    output = autograd::add(after_attention, transformed);
  }

  if (stats != nullptr) {
    // Масштаб остаточного потока на выходе слоя. Рост в разы по глубине
    // означает, что поправка на инициализацию выходных проекций не
    // справляется.
    stats->residual_rms.push_back(ops::root_mean_square(output.value()));
  }
  return output;
}

void Block::enable_lora(const LoraConfig& config, Rng* rng) {
  attention_.enable_lora(config, rng);
  mlp_.enable_lora(config, rng);
}

void Block::merge_lora() {
  attention_.merge_lora();
  mlp_.merge_lora();
}

void Block::freeze() {
  attention_norm_.freeze();
  mlp_norm_.freeze();
  attention_.freeze();
  mlp_.freeze();
}

void Block::collect(const std::string& prefix,
                    std::vector<NamedParameter>* out) {
  attention_norm_.collect(prefix + ".attention_norm", out);
  attention_.collect(prefix + ".attention", out);
  mlp_norm_.collect(prefix + ".mlp_norm", out);
  mlp_.collect(prefix + ".mlp", out);
}

Model::Model(const ModelConfig& config, uint64_t seed) : config_(config) {
  config_.validate();
  Rng rng(seed);

  token_embedding_ =
      Var::leaf(normal_tensor(Shape({config.vocab_size, config.d_model}),
                              config.init_std, &rng),
                true, "token_embedding");

  if (config.position == PositionKind::kLearned) {
    position_embedding_ =
        Var::leaf(normal_tensor(Shape({config.max_seq_len, config.d_model}),
                                config.init_std, &rng),
                  true, "position_embedding");
  }

  for (int64_t layer = 0; layer < config.n_layers; ++layer) {
    blocks_.push_back(Block(config, &rng));
  }

  final_norm_ = Norm(config, &rng);

  if (!config.tie_embeddings) {
    lm_head_ = Linear(config.d_model, config.vocab_size, config.init_std, &rng);
  }
}

Var Model::forward(const std::vector<int32_t>& ids, int64_t batch, int64_t seq,
                   int64_t position_offset, KvCache* cache,
                   ForwardStats* stats) const {
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

  if (config_.position == PositionKind::kLearned) {
    // Обучаемая позиция прибавляется к эмбеддингу токена один раз, на входе.
    // В отличие от RoPE она одна на все слои и на весь вектор, а не действует
    // на запросы и ключи отдельно в каждом слое.
    std::vector<int32_t> positions;
    positions.reserve(static_cast<std::size_t>(seq));
    for (int64_t step = 0; step < seq; ++step) {
      positions.push_back(static_cast<int32_t>(position_offset + step));
    }
    const Var encoded =
        autograd::reshape(autograd::embedding(position_embedding_, positions),
                          Shape({1, seq, config_.d_model}));
    // Растяжение по батчу: позиция одна и та же для всех примеров.
    hidden = autograd::add(hidden, encoded);
  }

  for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
    hidden = blocks_[layer].forward(hidden, position_offset, cache,
                                    static_cast<int64_t>(layer), stats);
  }
  // Длина кэша сдвигается один раз, после всех слоёв: они дописывают один и
  // тот же блок позиций, и сдвиг внутри цикла сбил бы отсчёт для следующего
  // слоя.
  if (cache != nullptr) {
    cache->advance(seq);
  }
  hidden = final_norm_.forward(hidden);

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

Var Model::loss(const std::vector<int32_t>& ids, int64_t batch, int64_t seq,
                ForwardStats* stats) const {
  LLM_CHECK_MSG(seq >= 2,
                "для предсказания следующего токена нужно хотя бы два");

  const Var logits = forward(ids, batch, seq, 0, nullptr, stats);
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
  if (stats != nullptr) {
    const ops::PredictionStats prediction =
        ops::prediction_stats(flat.value(), targets);
    stats->top1_accuracy = prediction.top1_accuracy;
    stats->prediction_entropy = prediction.entropy;
    stats->log_z = prediction.log_z;

    // Разбивка потерь по четвертям окна. Строки идут в порядке
    // (элемент батча, позиция), поэтому позиция строки — это остаток от
    // деления на длину окна без последней позиции.
    const std::vector<float> rows = ops::per_row_loss(flat.value(), targets);
    const int64_t window = seq - 1;
    std::vector<double> sums(4, 0.0);
    std::vector<int64_t> counts(4, 0);
    for (std::size_t row = 0; row < rows.size(); ++row) {
      const int64_t position = static_cast<int64_t>(row) % window;
      int64_t quarter = position * 4 / window;
      quarter = quarter > 3 ? 3 : quarter;
      sums[static_cast<std::size_t>(quarter)] += rows[row];
      counts[static_cast<std::size_t>(quarter)] += 1;
    }
    stats->loss_by_quarter.clear();
    for (std::size_t i = 0; i < 4; ++i) {
      stats->loss_by_quarter.push_back(
          counts[i] == 0
              ? 0.0f
              : static_cast<float>(sums[i] / static_cast<double>(counts[i])));
    }
  }
  const Var cross_entropy = autograd::cross_entropy(flat, targets);
  if (stats != nullptr) {
    stats->cross_entropy = *cross_entropy.value().data();
  }
  if (config_.z_loss_coef <= 0.0f) {
    return cross_entropy;
  }
  // Оптимизируется сумма, а отчитываемся по чистой перекрёстной энтропии:
  // сравнивать прогоны надо по одной и той же величине.
  return autograd::add(
      cross_entropy,
      autograd::mul_scalar(autograd::z_loss(flat), config_.z_loss_coef));
}

std::vector<NamedParameter> Model::parameters() {
  std::vector<NamedParameter> result;

  NamedParameter embedding;
  embedding.name = "token_embedding";
  embedding.value = &token_embedding_;
  result.push_back(embedding);

  if (config_.position == PositionKind::kLearned) {
    NamedParameter positions;
    positions.name = "position_embedding";
    positions.value = &position_embedding_;
    result.push_back(positions);
  }

  for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
    std::ostringstream prefix;
    prefix << "block." << layer;
    blocks_[layer].collect(prefix.str(), &result);
  }

  final_norm_.collect("final_norm", &result);

  if (!config_.tie_embeddings) {
    lm_head_.collect("lm_head", &result);
  }
  return result;
}

void Model::enable_lora(const LoraConfig& config) {
  Rng rng(config.seed);

  // Сначала замораживается всё, потом навешиваются адаптеры. Порядок важен:
  // иначе заморозка накрыла бы и только что созданные матрицы адаптера.
  token_embedding_ = Var::constant(token_embedding_.value());
  if (config_.position == PositionKind::kLearned) {
    position_embedding_ = Var::constant(position_embedding_.value());
  }
  for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
    blocks_[layer].freeze();
  }
  final_norm_.freeze();
  if (!config_.tie_embeddings) {
    lm_head_.freeze();
  }

  for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
    blocks_[layer].enable_lora(config, &rng);
  }
}

void Model::merge_lora() {
  for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
    blocks_[layer].merge_lora();
  }
}

std::vector<NamedParameter> Model::trainable_parameters() {
  std::vector<NamedParameter> all = parameters();
  std::vector<NamedParameter> trainable;
  for (std::size_t i = 0; i < all.size(); ++i) {
    if (all[i].value->requires_grad()) {
      trainable.push_back(all[i]);
    }
  }
  return trainable;
}

int64_t Model::trainable_parameter_count() {
  int64_t total = 0;
  const std::vector<NamedParameter> all = trainable_parameters();
  for (std::size_t i = 0; i < all.size(); ++i) {
    total += all[i].value->numel();
  }
  return total;
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
