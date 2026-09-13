// Кэш ключей и значений.
//
// При генерации модель вызывается по одному токену за раз, и без кэша каждый
// вызов пересчитывал бы ключи и значения для всего уже написанного текста.
// Стоимость генерации n токенов была бы квадратичной по n, причём с большим
// множителем: это полный прямой проход модели на каждом шаге.
//
// Кэш хранит ключи и значения всех прошлых позиций, и новый шаг досчитывает
// только свою строку. Расплата — память: batch * kv_heads * ctx * head_dim
// float на каждый слой, вдвое (ключи и значения). Именно ради этого числа
// придумано GQA: голов ключей меньше, чем голов запросов, и кэш во столько же
// раз меньше.
//
// Запросы не кэшируются: запрос нужен только на своём шаге и больше никогда.

#ifndef LLM_NN_KV_CACHE_H_
#define LLM_NN_KV_CACHE_H_

#include <cstdint>
#include <vector>

#include "core/tensor.h"
#include "nn/config.h"

namespace llm {
namespace nn {

class KvCache {
 public:
  KvCache(const ModelConfig& config, int64_t batch);

  // Начать сначала, не освобождая память: буферы переиспользуются между
  // запросами.
  void reset() { length_ = 0; }

  int64_t length() const { return length_; }
  int64_t capacity() const { return capacity_; }
  int64_t batch() const { return batch_; }

  // Дописывает ключи и значения нового блока позиций для указанного слоя и
  // возвращает вид на весь накопленный префикс, включая только что
  // дописанное.
  //
  // keys и values имеют форму (batch, kv_heads, seq, head_dim). Результат —
  // (batch, kv_heads, length() + seq, head_dim).
  //
  // Длина кэша здесь НЕ меняется: все слои дописывают один и тот же блок
  // позиций, и сдвигать длину можно только после последнего из них — иначе
  // второй слой считал бы, что первый уже сдвинул позицию.
  void append(int64_t layer, const Tensor& keys, const Tensor& values,
              Tensor* keys_view, Tensor* values_view);

  // Сдвигает длину после того, как все слои дописали свой блок.
  void advance(int64_t count);

 private:
  std::vector<Tensor> keys_;
  std::vector<Tensor> values_;
  int64_t batch_;
  int64_t capacity_;
  int64_t kv_heads_;
  int64_t head_dim_;
  int64_t length_;
};

}  // namespace nn
}  // namespace llm

#endif  // LLM_NN_KV_CACHE_H_
