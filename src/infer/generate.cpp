#include "infer/generate.h"

#include <chrono>

#include "core/check.h"

namespace llm {
namespace infer {
namespace {

using Clock = std::chrono::steady_clock;

std::vector<float> to_vector(const Tensor& logits) {
  const Tensor dense = logits.contiguous();
  const float* data = dense.data();
  return std::vector<float>(data, data + dense.numel());
}

}  // namespace

GenerateResult generate(nn::Model* model, const std::vector<int32_t>& prompt,
                        const GenerateConfig& config,
                        const TokenCallback& on_token, bool record_logits) {
  LLM_CHECK(model != nullptr);
  LLM_CHECK_MSG(!prompt.empty(), "затравка не может быть пустой");
  const int64_t context = model->config().max_seq_len;
  LLM_CHECK_MSG(static_cast<int64_t>(prompt.size()) < context,
                "затравка длиной " << prompt.size() << " не оставляет места в "
                                   << context);

  // Лента при генерации не нужна и только съедала бы память, удерживая все
  // промежуточные значения.
  autograd::NoGradGuard no_grad;

  GenerateResult result;
  std::vector<int32_t> tokens = prompt;
  Sampler sampler(config.sampler);
  const Clock::time_point start = Clock::now();

  nn::KvCache cache(model->config(), 1);
  nn::KvCache* cache_pointer = config.use_cache ? &cache : nullptr;

  // Затравка прогоняется одним куском: на ней кэш заполняется сразу за один
  // проход, а не по токену.
  autograd::Var logits = model->forward_last(
      tokens, 1, static_cast<int64_t>(tokens.size()), 0, cache_pointer);

  for (int64_t generated = 0; generated < config.max_tokens; ++generated) {
    std::vector<float> values = to_vector(logits.value());
    if (record_logits) {
      result.step_logits.push_back(values);
    }

    const int32_t token = sampler.sample(&values, tokens);
    tokens.push_back(token);
    result.tokens.push_back(token);
    if (on_token) {
      on_token(token);
    }

    if (static_cast<int64_t>(tokens.size()) >= context) {
      result.stop_reason = StopReason::kContextFull;
      break;
    }
    if (generated + 1 == config.max_tokens) {
      break;  // следующий прямой проход был бы холостым
    }

    if (config.use_cache) {
      // В кэше уже лежит всё до этого токена, поэтому в модель идёт он один.
      const std::vector<int32_t> step(1, token);
      logits = model->forward_last(step, 1, 1, cache.length(), cache_pointer);
    } else {
      logits = model->forward_last(
          tokens, 1, static_cast<int64_t>(tokens.size()), 0, nullptr);
    }
  }

  result.seconds = std::chrono::duration<double>(Clock::now() - start).count();
  return result;
}

}  // namespace infer
}  // namespace llm
