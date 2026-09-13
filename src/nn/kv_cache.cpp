#include "nn/kv_cache.h"

#include "core/check.h"
#include "ops/elementwise.h"

namespace llm {
namespace nn {

KvCache::KvCache(const ModelConfig& config, int64_t batch)
    : batch_(batch),
      capacity_(config.max_seq_len),
      kv_heads_(config.n_kv_heads),
      head_dim_(config.head_dim()),
      length_(0) {
  LLM_CHECK_GT(batch, static_cast<int64_t>(0));
  const Shape shape({batch_, kv_heads_, capacity_, head_dim_});
  for (int64_t layer = 0; layer < config.n_layers; ++layer) {
    keys_.push_back(Tensor::zeros(shape));
    values_.push_back(Tensor::zeros(shape));
  }
}

void KvCache::append(int64_t layer, const Tensor& keys, const Tensor& values,
                     Tensor* keys_view, Tensor* values_view) {
  LLM_CHECK(keys_view != nullptr && values_view != nullptr);
  LLM_CHECK_GE(layer, static_cast<int64_t>(0));
  LLM_CHECK_LT(layer, static_cast<int64_t>(keys_.size()));
  LLM_CHECK_MSG(keys.shape() == values.shape(),
                "ключи и значения разной формы: " << keys.shape() << " и "
                                                  << values.shape());
  LLM_CHECK_EQ(keys.shape().dim(0), batch_);
  LLM_CHECK_EQ(keys.shape().dim(1), kv_heads_);
  LLM_CHECK_EQ(keys.shape().dim(3), head_dim_);

  const int64_t incoming = keys.shape().dim(2);
  const int64_t total = length_ + incoming;
  LLM_CHECK_MSG(
      total <= capacity_,
      "кэш переполнен: " << total << " позиций при ёмкости " << capacity_);

  Tensor key_slot =
      keys_[static_cast<std::size_t>(layer)].slice(2, length_, incoming);
  Tensor value_slot =
      values_[static_cast<std::size_t>(layer)].slice(2, length_, incoming);
  ops::copy_into(keys, &key_slot);
  ops::copy_into(values, &value_slot);

  *keys_view = keys_[static_cast<std::size_t>(layer)].slice(2, 0, total);
  *values_view = values_[static_cast<std::size_t>(layer)].slice(2, 0, total);
}

void KvCache::advance(int64_t count) {
  LLM_CHECK_GE(count, static_cast<int64_t>(0));
  LLM_CHECK_LE(length_ + count, capacity_);
  length_ += count;
}

}  // namespace nn
}  // namespace llm
