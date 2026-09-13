// Корпус, разобранный на токены, и выборка батчей.
//
// Корпус хранится одной длинной последовательностью токенов, а батч — это
// несколько случайных окон из неё. Так модель видит и границы предложений, и
// середину, и не привязывается к какой-то одной нарезке.
//
// Проверочная часть отрезается с конца сплошным куском, а не случайными
// окнами. Случайная выборка перемешала бы обучение и проверку: соседние окна
// перекрываются, и модель увидела бы проверочный текст при обучении.

#ifndef LLM_DATA_DATASET_H_
#define LLM_DATA_DATASET_H_

#include <cstdint>
#include <string>
#include <vector>

#include "core/random.h"
#include "tokenizer/bpe.h"

namespace llm {
namespace data {

class TokenDataset {
 public:
  TokenDataset(std::vector<int32_t> tokens, double validation_fraction);

  static TokenDataset from_text(StrView text, const Bpe& tokenizer,
                                double validation_fraction);

  int64_t train_size() const { return split_; }
  int64_t validation_size() const {
    return static_cast<int64_t>(tokens_.size()) - split_;
  }
  int64_t total_size() const { return static_cast<int64_t>(tokens_.size()); }

  // batch окон длины seq, подряд в порядке (элемент батча, позиция).
  std::vector<int32_t> sample_batch(int64_t batch, int64_t seq, Rng* rng,
                                    bool validation) const;

  // Детерминированный проход по проверочной части: окно index-е по счёту.
  // Нужен, чтобы сравнение моделей шло на одних и тех же данных.
  std::vector<int32_t> validation_batch(int64_t batch, int64_t seq,
                                        int64_t index) const;

  int64_t validation_batch_count(int64_t batch, int64_t seq) const;

 private:
  std::vector<int32_t> tokens_;
  int64_t split_;  // сколько токенов с начала отведено под обучение
};

}  // namespace data
}  // namespace llm

#endif  // LLM_DATA_DATASET_H_
