// Модель: Llama-style трансформер.
//
//   tok_emb -> L x [ x += attn(rms_norm(x)) ; x += ffn(rms_norm(x)) ]
//           -> rms_norm -> lm_head
//
// Все слои — обычные структуры, а не иерархия с виртуальными методами.
// Полиморфизм здесь ничего не даёт: набор слоёв фиксирован и известен на
// этапе компиляции, а виртуальные вызовы только спрятали бы поток данных.
//
// Параметры собираются с именами. Имя нужно и оптимизатору (чтобы различать
// веса и нормировки), и чекпоинтам, и отладке: увидеть, у какого именно
// параметра взорвался градиент, гораздо полезнее, чем увидеть его номер.

#ifndef LLM_NN_MODEL_H_
#define LLM_NN_MODEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "autograd/node.h"
#include "core/random.h"
#include "nn/config.h"
#include "nn/kv_cache.h"

namespace llm {
namespace nn {

struct NamedParameter {
  std::string name;
  autograd::Var* value;
};

// Нормировка, скрывающая выбор между RMSNorm и LayerNorm.
//
// Вынесена в отдельный класс именно ради сравнения: слоям всё равно, чем
// нормировать, а у LayerNorm есть свободный член, которого у RMSNorm нет, и
// без этой обёртки развилка расползлась бы по всем местам, где стоит
// нормировка.
class Norm {
 public:
  Norm() {}
  explicit Norm(const ModelConfig& config, Rng* rng);

  autograd::Var forward(const autograd::Var& input) const;
  void collect(const std::string& prefix, std::vector<NamedParameter>* out);

 private:
  NormKind kind_ = NormKind::kRmsNorm;
  float eps_ = 1e-5f;
  autograd::Var weight_;
  autograd::Var bias_;  // существует только у LayerNorm
};

// Линейный слой без свободного члена.
//
// Веса хранятся как (вход, выход), а не (выход, вход), как в PyTorch. Тогда
// прямой проход — это просто x * W, без транспонирования на каждом вызове.
class Linear {
 public:
  Linear() {}
  Linear(int64_t in_features, int64_t out_features, float init_std, Rng* rng);

  autograd::Var forward(const autograd::Var& input) const;
  void collect(const std::string& prefix, std::vector<NamedParameter>* out);

 private:
  autograd::Var weight_;
};

// Размножение голов ключей и значений для GQA.
//
// Вынесено из класса и открыто ради тестов: раскладка голов — это то место,
// где легко ошибиться незаметно. Обе возможные раскладки дают работающую
// модель, обученную с нуля, но соответствие между головами запросов и ключей
// у них разное.
//
// input: (batch, kv_heads, seq, head_dim) -> (batch, kv_heads * repeats, seq,
// head_dim), где голова запроса h пользуется головой ключей h / repeats.
autograd::Var repeat_kv(const autograd::Var& input, int64_t batch,
                        int64_t kv_heads, int64_t repeats, int64_t seq,
                        int64_t head_dim);

class Attention {
 public:
  Attention() {}
  Attention(const ModelConfig& config, Rng* rng);

  // position_offset — абсолютная позиция первого запроса. При обучении нуль,
  // при генерации с KV-кэшем равен длине уже накопленного контекста.
  //
  // cache задаётся только при генерации: тогда input содержит лишь новые
  // позиции, а ключи и значения всех прошлых берутся из кэша. Градиент через
  // кэш не идёт, поэтому этот путь допустим только с выключенной лентой.
  autograd::Var forward(const autograd::Var& input, int64_t position_offset,
                        KvCache* cache = nullptr, int64_t layer = 0) const;
  void collect(const std::string& prefix, std::vector<NamedParameter>* out);

 private:
  autograd::Var split_heads(const autograd::Var& input, int64_t batch,
                            int64_t seq, int64_t heads) const;

  ModelConfig config_;
  Linear query_;
  Linear key_;
  Linear value_;
  Linear output_;
};

// FFN: либо SwiGLU, либо обычный двухматричный слой с GELU.
//
// У SwiGLU три матрицы: вентиль silu(gate(x)) умножается на up(x)
// поэлементно, то есть сеть сама решает, какие каналы пропустить дальше, а
// какие подавить, и решение зависит от входа. У варианта с GELU две матрицы и
// никакого вентиля — так было в исходном трансформере.
//
// Чтобы сравнение было о устройстве, а не о размере, ширину второго варианта
// берут в полтора раза больше: тогда числа параметров совпадают.
class Mlp {
 public:
  Mlp() {}
  Mlp(const ModelConfig& config, Rng* rng);

  autograd::Var forward(const autograd::Var& input) const;
  void collect(const std::string& prefix, std::vector<NamedParameter>* out);

 private:
  FfnKind kind_ = FfnKind::kSwiGlu;
  Linear gate_;  // существует только у SwiGLU
  Linear up_;
  Linear down_;
};

// Блок трансформера.
//
// Пре-нормировка: нормируется вход подслоя, а к результату прибавляется
// НЕнормированный остаток. Пост-нормировка: нормируется уже сумма.
//
// Порядок существен. При пост-нормировке нормировка стоит на пути остатка, и
// градиент на каждом слое проходит через неё. При пре-нормировке путь остатка
// от выхода до входа — чистое сложение, и градиент доходит до первого слоя без
// искажений. Насколько это важно на четырёх слоях — вопрос к замеру, а не к
// рассуждению.
class Block {
 public:
  Block() {}
  Block(const ModelConfig& config, Rng* rng);

  autograd::Var forward(const autograd::Var& input, int64_t position_offset,
                        KvCache* cache = nullptr, int64_t layer = 0) const;
  void collect(const std::string& prefix, std::vector<NamedParameter>* out);

 private:
  ModelConfig config_;
  Norm attention_norm_;
  Norm mlp_norm_;
  Attention attention_;
  Mlp mlp_;
};

class Model {
 public:
  Model(const ModelConfig& config, uint64_t seed);

  const ModelConfig& config() const { return config_; }

  // ids — batch * seq идентификаторов в порядке (батч, позиция).
  // Возвращает логиты формы (batch, seq, vocab).
  autograd::Var forward(const std::vector<int32_t>& ids, int64_t batch,
                        int64_t seq, int64_t position_offset = 0,
                        KvCache* cache = nullptr) const;

  // Логиты только последней позиции — то, что нужно генерации. Форма
  // (batch, vocab).
  autograd::Var forward_last(const std::vector<int32_t>& ids, int64_t batch,
                             int64_t seq, int64_t position_offset = 0,
                             KvCache* cache = nullptr) const;

  // Потери на предсказании следующего токена: каждая позиция предсказывает
  // следующую, последняя отбрасывается — ей нечего предсказывать.
  autograd::Var loss(const std::vector<int32_t>& ids, int64_t batch,
                     int64_t seq) const;

  std::vector<NamedParameter> parameters();
  int64_t parameter_count();

 private:
  ModelConfig config_;
  autograd::Var token_embedding_;
  autograd::Var position_embedding_;  // только при обучаемых позициях
  std::vector<Block> blocks_;
  Norm final_norm_;
  Linear lm_head_;  // не используется при связанных эмбеддингах
};

}  // namespace nn
}  // namespace llm

#endif  // LLM_NN_MODEL_H_
