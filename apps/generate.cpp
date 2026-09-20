// Генерация текста обученной моделью.
//
// Запуск:
//   generate <чекпоинт.llmw> <словарь> ["затравка"] [токенов] [температура]
//
// Словарь бывает двух видов: наш .bpe и чужой tokenizer.json от модели
// HuggingFace, перенесённой через import_hf. Вид определяется по имени файла.
// Различаются они ровно двумя действиями — превратить текст в номера и номер
// обратно в байты, — поэтому вместо иерархии классов здесь две функции.
//
// Помимо текста печатает скорость генерации с KV-кэшем и без него: разница и
// есть то, ради чего кэш существует.

#include "infer/generate.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "args.h"

#include "core/check.h"
#include "nn/model.h"
#include "serialize/checkpoint.h"
#include "tokenizer/bpe.h"
#include "tokenizer/hf_tokenizer.h"

namespace {

bool ends_with(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

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
  const int64_t max_tokens =
      argc > 4 ? bench::parse_int64(argv[4], "число токенов") : 200;
  const float temperature =
      argc > 5 ? static_cast<float>(bench::parse_double(argv[5], "температура"))
               : 0.8f;

  const bool foreign = ends_with(vocab_path, ".json");

  // Оба словаря живут до конца работы: функции ниже держат на них ссылки.
  llm::Bpe own;
  llm::HfTokenizer other;
  std::function<std::vector<int32_t>(const std::string&)> encode;
  std::function<std::string(int32_t)> piece;
  int64_t vocab_size = 0;
  int32_t stop_token = -1;

  if (foreign) {
    other = llm::HfTokenizer::load(vocab_path);
    vocab_size = other.vocab_size();
    encode = [&other](const std::string& text) { return other.encode(text); };
    piece = [&other](int32_t token) {
      return other.decode(std::vector<int32_t>(1, token));
    };
    // Токен конца текста: без остановки на нём модель, закончив ответ,
    // спокойно начинает следующий документ.
    for (std::size_t i = 0; i < other.added_tokens().size(); ++i) {
      const std::string& content = other.added_tokens()[i].content;
      if (content == "<|endoftext|>" || content == "<|end_of_text|>" ||
          content == "</s>") {
        stop_token = other.added_tokens()[i].id;
      }
    }
  } else {
    own = llm::Bpe::load(vocab_path);
    vocab_size = own.vocab_size();
    encode = [&own](const std::string& text) { return own.encode(text); };
    piece = [&own](int32_t token) { return own.token_bytes(token); };
  }

  const llm::nn::ModelConfig config =
      llm::serialize::read_config(checkpoint_path);
  llm::nn::Model model(config, 0);
  const int64_t step = llm::serialize::load_checkpoint(checkpoint_path, &model);

  std::printf("модель: %s\n", config.to_string().c_str());
  std::printf("обучена до шага %lld, словарь %lld токенов (%s)\n",
              static_cast<long long>(step), static_cast<long long>(vocab_size),
              foreign ? "чужой" : "свой");
  // Расхождение словаря с моделью — это не «немного не то»: номера токенов
  // поедут, и модель будет отвечать связно и мимо.
  LLM_CHECK_MSG(vocab_size <= config.vocab_size,
                "в словаре " << vocab_size << " токенов, а модель знает "
                             << config.vocab_size);

  std::vector<int32_t> prompt = encode(prompt_text);
  if (prompt.empty()) {
    prompt.push_back(foreign ? 0 : llm::Bpe::kBos);
  }

  llm::infer::GenerateConfig generate_config;
  generate_config.max_tokens = max_tokens;
  generate_config.sampler.temperature = temperature;
  generate_config.sampler.top_k = 40;
  generate_config.sampler.top_p = 0.95f;
  generate_config.sampler.repetition_penalty = 1.1f;
  generate_config.sampler.seed = 2026;
  generate_config.stop_token = stop_token;

  // Таблица эмбеддингов перекладывается один раз, а не на каждый токен.
  // Логиты от этого не меняются ни в одном разряде.
  model.prepare_inference();

  std::printf("\n--- затравка ---\n%s", prompt_text.c_str());
  std::fflush(stdout);

  const llm::infer::GenerateResult result = llm::infer::generate(
      &model, prompt, generate_config, [&piece](int32_t token) {
        const std::string bytes = piece(token);
        std::fwrite(bytes.data(), 1, bytes.size(), stdout);
        std::fflush(stdout);
      });

  std::printf("\n\n");
  if (result.stop_reason == llm::infer::StopReason::kContextFull) {
    std::printf("[остановлено: контекст %lld токенов заполнен]\n",
                static_cast<long long>(config.max_seq_len));
  }
  if (result.stop_reason == llm::infer::StopReason::kStopToken) {
    std::printf("[остановлено: модель закончила текст]\n");
  }
  std::printf("--- скорость ---\n");
  std::printf("с кэшем:  %.1f токенов/с (%zu токенов за %.2f с)\n",
              static_cast<double>(result.tokens.size()) / result.seconds,
              result.tokens.size(), result.seconds);

  // Тот же прогон без кэша — чтобы разница была видна, а не заявлена. На
  // большой модели он занял бы вдвое больше самой генерации и показал бы ровно
  // то же самое, поэтому считается только на малых.
  //
  // Считается он здесь, до pack_half, и иначе нельзя: сравнивать его не с чем,
  // кроме прогона с кэшем выше, а тот шёл на обычных весах. Стоял он ниже —
  // и «ускорение от кэша» выходило поделённым ещё и на ускорение от половинной
  // разрядности, то есть заниженным во столько же раз.
  const int64_t kSlowRunLimit = 5000000;
  if (model.parameter_count() <= kSlowRunLimit) {
    llm::infer::GenerateConfig without_cache = generate_config;
    without_cache.use_cache = false;
    const llm::infer::GenerateResult slow =
        llm::infer::generate(&model, prompt, without_cache);
    std::printf("без кэша: %.1f токенов/с (%zu токенов за %.2f с)\n",
                static_cast<double>(slow.tokens.size()) / slow.seconds,
                slow.tokens.size(), slow.seconds);
    std::printf("ускорение от кэша: %.1fx\n", slow.seconds / result.seconds);
  } else {
    std::printf("без кэша: не считалось (модель крупнее %lld параметров)\n",
                static_cast<long long>(kSlowRunLimit));
  }

  // Тот же прогон на весах половинной разрядности. Считается здесь же, а не
  // отдельной программой, потому что сравнивать надо на одной модели, одной
  // затравке и одном зерне выборки — иначе разница потеряется в разбросе.
  //
  // Веса не заменяются, а копируются: обычные остаются на месте. Значит на
  // время замера модель занимает в полтора раза больше памяти, и это цена
  // именно сравнения, а не самого инференса.
  model.pack_half();
  const llm::infer::GenerateResult half =
      llm::infer::generate(&model, prompt, generate_config);
  std::printf("половинная разрядность весов: %.1f токенов/с (%.2fx)\n",
              static_cast<double>(half.tokens.size()) / half.seconds,
              result.seconds / half.seconds);
  // Совпал ли текст. Округление весов сдвигает логиты, и там, где два
  // кандидата стояли рядом, выбор может перевернуться — расхождение после
  // какого-то токена нормально и ничего не ломает. А вот расхождение с первого
  // токена означало бы, что сломана сама упаковка.
  std::size_t common = 0;
  while (common < half.tokens.size() && common < result.tokens.size() &&
         half.tokens[common] == result.tokens[common]) {
    ++common;
  }
  std::printf("совпало токенов подряд: %zu из %zu\n", common,
              result.tokens.size());

  return 0;
}
