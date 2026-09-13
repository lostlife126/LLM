#include "data/dataset.h"

#include "core/check.h"

namespace llm {
namespace data {

TokenDataset::TokenDataset(std::vector<int32_t> tokens,
                           double validation_fraction)
    : tokens_(std::move(tokens)) {
  LLM_CHECK_MSG(validation_fraction >= 0.0 && validation_fraction < 1.0,
                "доля проверочной части должна быть в [0, 1), получено "
                    << validation_fraction);
  LLM_CHECK_MSG(!tokens_.empty(), "пустой корпус");
  split_ = static_cast<int64_t>(static_cast<double>(tokens_.size()) *
                                (1.0 - validation_fraction));
  LLM_CHECK_GT(split_, static_cast<int64_t>(0));
}

TokenDataset TokenDataset::from_text(StrView text, const Bpe& tokenizer,
                                     double validation_fraction) {
  return TokenDataset(tokenizer.encode(text), validation_fraction);
}

std::vector<int32_t> TokenDataset::sample_batch(int64_t batch, int64_t seq,
                                                Rng* rng,
                                                bool validation) const {
  LLM_CHECK_GT(batch, static_cast<int64_t>(0));
  LLM_CHECK_GT(seq, static_cast<int64_t>(1));

  const int64_t begin = validation ? split_ : 0;
  const int64_t end = validation ? total_size() : split_;
  const int64_t available = end - begin;
  LLM_CHECK_MSG(available > seq,
                "в части корпуса " << available << " токенов, а окно " << seq);

  std::vector<int32_t> result;
  result.reserve(static_cast<std::size_t>(batch * seq));
  // Начало окна может быть любым: последнее допустимое — такое, при котором
  // окно ещё целиком помещается.
  const uint64_t positions = static_cast<uint64_t>(available - seq);
  for (int64_t item = 0; item < batch; ++item) {
    const int64_t start = begin + static_cast<int64_t>(rng->index(positions));
    for (int64_t position = 0; position < seq; ++position) {
      result.push_back(tokens_[static_cast<std::size_t>(start + position)]);
    }
  }
  return result;
}

int64_t TokenDataset::validation_batch_count(int64_t batch, int64_t seq) const {
  const int64_t windows = validation_size() / seq;
  return windows / batch;
}

std::vector<int32_t> TokenDataset::validation_batch(int64_t batch, int64_t seq,
                                                    int64_t index) const {
  LLM_CHECK_GE(index, static_cast<int64_t>(0));
  LLM_CHECK_LT(index, validation_batch_count(batch, seq));

  std::vector<int32_t> result;
  result.reserve(static_cast<std::size_t>(batch * seq));
  // Окна идут подряд и не перекрываются: каждый проверочный токен учитывается
  // ровно один раз.
  const int64_t base = split_ + index * batch * seq;
  for (int64_t item = 0; item < batch; ++item) {
    for (int64_t position = 0; position < seq; ++position) {
      result.push_back(
          tokens_[static_cast<std::size_t>(base + item * seq + position)]);
    }
  }
  return result;
}

}  // namespace data
}  // namespace llm
