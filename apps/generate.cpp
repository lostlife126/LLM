// Генерация текста обученной моделью.
//
// Запуск:
//   generate <чекпоинт.llmw> <словарь.bpe> ["затравка"] [токенов] [температура]
//
// Помимо текста печатает скорость генерации с KV-кэшем и без него: разница и
// есть то, ради чего кэш существует.

#include "infer/generate.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "nn/model.h"
#include "serialize/checkpoint.h"
#include "tokenizer/bpe.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "использование: %s <чекпоинт.llmw> <словарь.bpe> "
                 "[\"затравка\"] [токенов] [температура]\n",
                 argv[0]);
    return 1;
  }
  const std::string checkpoint_path = argv[1];
  const std::string vocab_path = argv[2];
  const std::string prompt_text = argc > 3 ? argv[3] : "The ";
  const int64_t max_tokens = argc > 4 ? std::atoll(argv[4]) : 200;
  const float temperature =
      argc > 5 ? static_cast<float>(std::atof(argv[5])) : 0.8f;

  const llm::Bpe tokenizer = llm::Bpe::load(vocab_path);
  const llm::nn::ModelConfig config =
      llm::serialize::read_config(checkpoint_path);
  llm::nn::Model model(config, 0);
  const int64_t step = llm::serialize::load_checkpoint(checkpoint_path, &model);

  std::printf("модель: %s\n", config.to_string().c_str());
  std::printf("обучена до шага %lld, словарь %d токенов\n",
              static_cast<long long>(step), tokenizer.vocab_size());

  std::vector<int32_t> prompt = tokenizer.encode(prompt_text);
  if (prompt.empty()) {
    prompt.push_back(llm::Bpe::kBos);
  }

  llm::infer::GenerateConfig generate_config;
  generate_config.max_tokens = max_tokens;
  generate_config.sampler.temperature = temperature;
  generate_config.sampler.top_k = 40;
  generate_config.sampler.top_p = 0.95f;
  generate_config.sampler.repetition_penalty = 1.1f;
  generate_config.sampler.seed = 2026;

  std::printf("\n--- затравка ---\n%s", prompt_text.c_str());
  std::fflush(stdout);

  const llm::infer::GenerateResult result = llm::infer::generate(
      &model, prompt, generate_config, [&tokenizer](int32_t token) {
        const std::string& bytes = tokenizer.token_bytes(token);
        std::fwrite(bytes.data(), 1, bytes.size(), stdout);
        std::fflush(stdout);
      });

  std::printf("\n\n");
  if (result.stop_reason == llm::infer::StopReason::kContextFull) {
    std::printf("[остановлено: контекст %lld токенов заполнен]\n",
                static_cast<long long>(config.max_seq_len));
  }
  std::printf("--- скорость ---\n");
  std::printf("с кэшем:  %.1f токенов/с (%zu токенов за %.2f с)\n",
              static_cast<double>(result.tokens.size()) / result.seconds,
              result.tokens.size(), result.seconds);

  // Тот же прогон без кэша — чтобы разница была видна, а не заявлена.
  llm::infer::GenerateConfig without_cache = generate_config;
  without_cache.use_cache = false;
  const llm::infer::GenerateResult slow =
      llm::infer::generate(&model, prompt, without_cache);
  std::printf("без кэша: %.1f токенов/с (%zu токенов за %.2f с)\n",
              static_cast<double>(slow.tokens.size()) / slow.seconds,
              slow.tokens.size(), slow.seconds);
  std::printf("ускорение: %.1fx\n", slow.seconds / result.seconds);
  return 0;
}
