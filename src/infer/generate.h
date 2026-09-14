// Генерация текста.
//
// Два режима существуют не для выбора, а ради проверки: генерация с кэшем
// обязана давать тот же результат, что и генерация без него, пересчитывающая
// всё с нуля на каждом шаге. Медленный режим — эталон, по которому проверяется
// быстрый.

#ifndef LLM_INFER_GENERATE_H_
#define LLM_INFER_GENERATE_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "infer/sampler.h"
#include "nn/model.h"

namespace llm {
namespace infer {

struct GenerateConfig {
  int64_t max_tokens = 200;
  SamplerConfig sampler;

  // false пересчитывает весь контекст на каждом шаге. Нужен только для сверки.
  bool use_cache = true;

  // Токен, на котором генерация прекращается. -1 выключает.
  //
  // Нужен не для красоты. Настоящие модели обучены заканчивать текст особым
  // токеном, и без остановки на нём генерация продолжается за концом ответа:
  // модель начинает новый документ с чистого листа, и получается связный, но
  // явно посторонний хвост.
  int32_t stop_token = -1;
};

// Вызывается на каждый выбранный токен — для потокового вывода.
using TokenCallback = std::function<void(int32_t token)>;

// Почему генерация остановилась. Упереться в контекст — самая частая причина
// на маленькой модели, и молчать об этом значит выдавать обрезанный текст за
// законченный.
enum class StopReason { kMaxTokens, kContextFull, kStopToken };

struct GenerateResult {
  std::vector<int32_t> tokens;  // только сгенерированные, без затравки
  StopReason stop_reason = StopReason::kMaxTokens;
  double seconds = 0.0;
  // Логиты каждого шага до сэмплирования. Заполняются только по запросу: на
  // них держится сверка режима с кэшем и без.
  std::vector<std::vector<float>> step_logits;
};

GenerateResult generate(nn::Model* model, const std::vector<int32_t>& prompt,
                        const GenerateConfig& config,
                        const TokenCallback& on_token = TokenCallback(),
                        bool record_logits = false);

}  // namespace infer
}  // namespace llm

#endif  // LLM_INFER_GENERATE_H_
