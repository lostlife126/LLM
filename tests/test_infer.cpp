// Инференс: KV-кэш и сэмплирование.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/random.h"
#include "infer/generate.h"
#include "infer/sampler.h"
#include "nn/kv_cache.h"
#include "testing.h"

namespace {

using llm::infer::GenerateConfig;
using llm::infer::GenerateResult;
using llm::infer::Sampler;
using llm::infer::SamplerConfig;
using llm::nn::Model;
using llm::nn::ModelConfig;

ModelConfig test_config() {
  ModelConfig config;
  config.vocab_size = 32;
  config.d_model = 16;
  config.n_layers = 2;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.max_seq_len = 24;
  config.ffn_hidden = 24;
  config.validate();
  return config;
}

std::vector<int32_t> random_ids(int64_t count, int64_t vocab,
                                std::uint64_t seed) {
  llm::Rng rng(seed);
  std::vector<int32_t> ids;
  for (int64_t i = 0; i < count; ++i) {
    ids.push_back(
        static_cast<int32_t>(rng.index(static_cast<uint64_t>(vocab))));
  }
  return ids;
}

}  // namespace

LLM_TEST(Infer, CacheMatchesFullRecompute) {
  // Главный тест инференса.
  //
  // Генерация с кэшем обязана совпасть с генерацией, пересчитывающей весь
  // контекст на каждом шаге. Сравнение точное: кэш не приближает вычисление,
  // а лишь избавляет от его повторения, поэтому расхождение даже в последнем
  // бите означало бы ошибку — сдвинутые позиции RoPE, неверное смещение маски
  // или испорченную раскладку голов.
  const ModelConfig config = test_config();
  Model model(config, 17);
  const std::vector<int32_t> prompt = random_ids(5, config.vocab_size, 1);

  GenerateConfig cached;
  cached.max_tokens = 12;
  cached.sampler.temperature = 0.8f;
  cached.sampler.seed = 99;
  cached.use_cache = true;

  GenerateConfig full = cached;
  full.use_cache = false;

  const GenerateResult with_cache = llm::infer::generate(
      &model, prompt, cached, llm::infer::TokenCallback(), true);
  const GenerateResult without_cache = llm::infer::generate(
      &model, prompt, full, llm::infer::TokenCallback(), true);

  LLM_CHECK_EQ(with_cache.tokens.size(), without_cache.tokens.size());
  LLM_CHECK_EQ(with_cache.step_logits.size(), without_cache.step_logits.size());

  for (std::size_t step = 0; step < with_cache.step_logits.size(); ++step) {
    const std::vector<float>& a = with_cache.step_logits[step];
    const std::vector<float>& b = without_cache.step_logits[step];
    LLM_CHECK_EQ(a.size(), b.size());
    for (std::size_t token = 0; token < a.size(); ++token) {
      LLM_CHECK_MSG(a[token] == b[token], "шаг " << step << ", токен " << token
                                                 << ": с кэшем " << a[token]
                                                 << ", без кэша " << b[token]);
    }
  }
  LLM_CHECK(with_cache.tokens == without_cache.tokens);
}

LLM_TEST(Infer, CacheMatchesFullRecomputeWithSingleTokenPrompt) {
  // Затравка из одного токена: кэш заполняется не блоком, а по одному.
  const ModelConfig config = test_config();
  Model model(config, 23);
  const std::vector<int32_t> prompt(1, 7);

  GenerateConfig cached;
  cached.max_tokens = 10;
  cached.sampler.temperature = 0.0f;  // жадный выбор
  cached.use_cache = true;
  GenerateConfig full = cached;
  full.use_cache = false;

  LLM_CHECK(llm::infer::generate(&model, prompt, cached).tokens ==
            llm::infer::generate(&model, prompt, full).tokens);
}

LLM_TEST(Infer, GenerationStopsAtContextLimit) {
  const ModelConfig config = test_config();
  Model model(config, 29);
  const std::vector<int32_t> prompt = random_ids(20, config.vocab_size, 2);

  GenerateConfig generate_config;
  generate_config.max_tokens = 100;  // заведомо больше, чем влезет
  generate_config.sampler.temperature = 0.0f;

  const GenerateResult result =
      llm::infer::generate(&model, prompt, generate_config);
  // Контекст 24, затравка 20 — влезет ровно четыре токена.
  LLM_CHECK_EQ(result.tokens.size(), static_cast<std::size_t>(4));
  LLM_CHECK(result.stop_reason == llm::infer::StopReason::kContextFull);

  // А при коротком запросе остановка происходит по числу токенов.
  GenerateConfig short_request = generate_config;
  short_request.max_tokens = 2;
  const GenerateResult brief = llm::infer::generate(
      &model, random_ids(3, config.vocab_size, 5), short_request);
  LLM_CHECK_EQ(brief.tokens.size(), static_cast<std::size_t>(2));
  LLM_CHECK(brief.stop_reason == llm::infer::StopReason::kMaxTokens);
}

LLM_TEST(Infer, GenerationRejectsOversizedPrompt) {
  const ModelConfig config = test_config();
  Model model(config, 31);
  GenerateConfig generate_config;
  LLM_EXPECT_THROWS(llm::infer::generate(
      &model, random_ids(config.max_seq_len, config.vocab_size, 3),
      generate_config));
  LLM_EXPECT_THROWS(
      llm::infer::generate(&model, std::vector<int32_t>(), generate_config));
}

LLM_TEST(Infer, StreamingCallbackSeesEveryToken) {
  const ModelConfig config = test_config();
  Model model(config, 37);
  GenerateConfig generate_config;
  generate_config.max_tokens = 6;
  generate_config.sampler.temperature = 0.0f;

  std::vector<int32_t> streamed;
  const GenerateResult result = llm::infer::generate(
      &model, random_ids(3, config.vocab_size, 4), generate_config,
      [&streamed](int32_t token) { streamed.push_back(token); });
  LLM_CHECK(streamed == result.tokens);
}

LLM_TEST(Infer, CacheRejectsOverflow) {
  const ModelConfig config = test_config();
  llm::nn::KvCache cache(config, 1);
  LLM_CHECK_EQ(cache.length(), static_cast<std::int64_t>(0));
  LLM_CHECK_EQ(cache.capacity(), config.max_seq_len);

  const llm::Shape block(
      {1, config.n_kv_heads, config.max_seq_len, config.head_dim()});
  const llm::Tensor keys = llm::Tensor::zeros(block);
  llm::Tensor view_a;
  llm::Tensor view_b;
  cache.append(0, keys, keys, &view_a, &view_b);
  cache.advance(config.max_seq_len);
  LLM_CHECK_EQ(cache.length(), config.max_seq_len);

  // Ещё одна позиция уже не влезает.
  const llm::Tensor one = llm::Tensor::zeros(
      llm::Shape({1, config.n_kv_heads, 1, config.head_dim()}));
  LLM_EXPECT_THROWS(cache.append(0, one, one, &view_a, &view_b));

  cache.reset();
  LLM_CHECK_EQ(cache.length(), static_cast<std::int64_t>(0));
}

LLM_TEST(Infer, CacheReturnsWhatWasWritten) {
  const ModelConfig config = test_config();
  llm::nn::KvCache cache(config, 1);
  const llm::Shape block({1, config.n_kv_heads, 2, config.head_dim()});

  llm::Tensor first = llm::Tensor::full(block, 1.0f);
  llm::Tensor second = llm::Tensor::full(block, 2.0f);
  llm::Tensor keys_view;
  llm::Tensor values_view;

  cache.append(0, first, first, &keys_view, &values_view);
  cache.advance(2);
  cache.append(0, second, second, &keys_view, &values_view);

  // Вид обязан содержать оба блока: старый и только что дописанный.
  LLM_CHECK_EQ(keys_view.dim(2), static_cast<std::int64_t>(4));
  LLM_EXPECT_NEAR(keys_view(0, 0, 0, 0), 1.0, 1e-6);
  LLM_EXPECT_NEAR(keys_view(0, 0, 1, 0), 1.0, 1e-6);
  LLM_EXPECT_NEAR(keys_view(0, 0, 2, 0), 2.0, 1e-6);
  LLM_EXPECT_NEAR(keys_view(0, 0, 3, 0), 2.0, 1e-6);

  // Слои независимы: во втором слое ничего нет.
  llm::Tensor other_keys;
  llm::Tensor other_values;
  cache.append(1, second, second, &other_keys, &other_values);
  LLM_EXPECT_NEAR(other_keys(0, 0, 0, 0), 0.0, 1e-6);
}

// --- Сэмплирование ---

LLM_TEST(Sampler, ZeroTemperatureIsGreedy) {
  SamplerConfig config;
  config.temperature = 0.0f;
  Sampler sampler(config);

  std::vector<float> logits = {1.0f, 5.0f, 2.0f, 4.9f};
  LLM_CHECK_EQ(sampler.sample(&logits, std::vector<int32_t>()), 1);
}

LLM_TEST(Sampler, IsDeterministicForSameSeed) {
  SamplerConfig config;
  config.seed = 555;
  Sampler first(config);
  Sampler second(config);

  for (int step = 0; step < 20; ++step) {
    std::vector<float> a = {0.5f, 1.0f, 0.2f, 0.8f, 0.1f};
    std::vector<float> b = a;
    LLM_CHECK_EQ(first.sample(&a, std::vector<int32_t>()),
                 second.sample(&b, std::vector<int32_t>()));
  }
}

LLM_TEST(Sampler, TopKKeepsOnlyKCandidates) {
  SamplerConfig config;
  config.top_k = 2;
  config.seed = 7;
  Sampler sampler(config);

  // Токены 1 и 3 — два самых вероятных; остальные не должны выпадать никогда.
  for (int trial = 0; trial < 500; ++trial) {
    std::vector<float> logits = {0.0f, 5.0f, 0.1f, 4.0f, 0.2f};
    const int32_t token = sampler.sample(&logits, std::vector<int32_t>());
    LLM_CHECK_MSG(token == 1 || token == 3,
                  "top-k выдал токен " << token << " вне двух лучших");
  }
}

LLM_TEST(Sampler, TopPAdaptsToConfidence) {
  // Там, где модель уверена, top-p оставляет один вариант; там, где нет, —
  // несколько. В этом его отличие от top-k, который всегда оставляет k.
  SamplerConfig config;
  config.top_p = 0.9f;
  config.seed = 11;
  Sampler confident(config);
  Sampler uncertain(config);

  for (int trial = 0; trial < 300; ++trial) {
    std::vector<float> sharp = {20.0f, 0.0f, 0.0f, 0.0f};
    LLM_CHECK_EQ(confident.sample(&sharp, std::vector<int32_t>()), 0);
  }

  bool saw_several = false;
  int32_t first_token = -1;
  for (int trial = 0; trial < 300; ++trial) {
    std::vector<float> flat = {0.0f, 0.0f, 0.0f, 0.0f};
    const int32_t token = uncertain.sample(&flat, std::vector<int32_t>());
    if (first_token < 0) {
      first_token = token;
    } else if (token != first_token) {
      saw_several = true;
    }
  }
  LLM_CHECK(saw_several);
}

LLM_TEST(Sampler, TopPBreaksTiesByTokenNumber) {
  // При равных вероятностях порядок сортировки задаёт, кто попадёт в
  // оставленную голову распределения, а кто будет отсечён. std::sort порядок
  // равных не определяет, поэтому выбор зависел бы от того, чья стандартная
  // библиотека, — и генерация с одним зерном давала бы на двух машинах разный
  // текст. Отсюда stable_sort: у равных сохраняется исходный порядок, то есть
  // номер токена.
  //
  // Проверка: шестьдесят четыре одинаковых логита, top-p = 0.25. Каждая
  // вероятность равна 1/64, сумма перешагивает 0.25 на шестнадцатом токене,
  // значит остаются ровно номера с нулевого по пятнадцатый.
  SamplerConfig config;
  config.top_p = 0.25f;
  config.seed = 20260920;
  Sampler sampler(config);

  const std::size_t width = 64;
  for (int trial = 0; trial < 2000; ++trial) {
    std::vector<float> logits(width, 0.0f);
    const int32_t token = sampler.sample(&logits, std::vector<int32_t>());
    LLM_CHECK_MSG(token >= 0 && token < 16,
                  "при равных вероятностях выпал токен " << token
                                                         << ", а голова "
                                                            "распределения — "
                                                            "номера 0..15");
  }
}

LLM_TEST(Sampler, TopPAlwaysKeepsAtLeastOneToken) {
  // Порог меньше вероятности самого вероятного токена не должен отсекать всё.
  SamplerConfig config;
  config.top_p = 0.1f;
  config.seed = 13;
  Sampler sampler(config);
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<float> logits = {10.0f, 0.0f, 0.0f};
    LLM_CHECK_EQ(sampler.sample(&logits, std::vector<int32_t>()), 0);
  }
}

LLM_TEST(Sampler, RepetitionPenaltyLowersSeenTokens) {
  SamplerConfig config;
  config.temperature = 0.0f;  // жадный выбор, чтобы видеть чистый эффект
  config.repetition_penalty = 2.0f;
  Sampler sampler(config);

  // Без штрафа победил бы токен 0; он уже встречался, поэтому его логит
  // делится пополам, и побеждает токен 1.
  std::vector<int32_t> history;
  history.push_back(0);
  std::vector<float> logits = {4.0f, 3.0f, 1.0f};
  LLM_CHECK_EQ(sampler.sample(&logits, history), 1);
}

LLM_TEST(Sampler, RepetitionPenaltyDoesNotRewardNegativeLogits) {
  // У отрицательного логита деление приблизило бы его к нулю, то есть
  // повысило бы вероятность. Штраф обязан умножать, а не делить.
  SamplerConfig config;
  config.temperature = 0.0f;
  config.repetition_penalty = 2.0f;
  Sampler sampler(config);

  std::vector<int32_t> history;
  history.push_back(0);
  std::vector<float> logits = {-1.0f, -1.5f};
  // Токен 0 встречался: -1 * 2 = -2, что хуже, чем -1.5 у токена 1.
  LLM_CHECK_EQ(sampler.sample(&logits, history), 1);
}

LLM_TEST(Sampler, RepetitionPenaltyRespectsWindow) {
  SamplerConfig config;
  config.temperature = 0.0f;
  config.repetition_penalty = 2.0f;
  config.repetition_window = 2;
  Sampler sampler(config);

  // Токен 0 встречался, но давно — за пределами окна, поэтому не штрафуется.
  std::vector<int32_t> history;
  history.push_back(0);
  history.push_back(1);
  history.push_back(1);
  std::vector<float> logits = {4.0f, 3.0f, 1.0f};
  LLM_CHECK_EQ(sampler.sample(&logits, history), 0);
}

LLM_TEST(Sampler, FollowsDistribution) {
  // Розыгрыш должен соответствовать вероятностям, а не просто выдавать
  // разнообразие. Логиты log(1), log(2), log(7) дают доли 0.1, 0.2 и 0.7.
  SamplerConfig config;
  config.seed = 2024;
  Sampler sampler(config);

  std::vector<int> counts(3, 0);
  const int trials = 30000;
  for (int trial = 0; trial < trials; ++trial) {
    std::vector<float> logits = {std::log(1.0f), std::log(2.0f),
                                 std::log(7.0f)};
    counts[static_cast<std::size_t>(
        sampler.sample(&logits, std::vector<int32_t>()))] += 1;
  }
  LLM_EXPECT_NEAR(static_cast<double>(counts[0]) / trials, 0.1, 0.01);
  LLM_EXPECT_NEAR(static_cast<double>(counts[1]) / trials, 0.2, 0.015);
  LLM_EXPECT_NEAR(static_cast<double>(counts[2]) / trials, 0.7, 0.015);
}

LLM_TEST(Sampler, TemperatureSharpensAndFlattens) {
  // Низкая температура заостряет распределение, высокая сглаживает.
  const int trials = 5000;
  int sharp_best = 0;
  int flat_best = 0;
  for (int mode = 0; mode < 2; ++mode) {
    SamplerConfig config;
    config.temperature = mode == 0 ? 0.3f : 3.0f;
    config.seed = 77;
    Sampler sampler(config);
    for (int trial = 0; trial < trials; ++trial) {
      std::vector<float> logits = {2.0f, 1.0f, 0.0f};
      if (sampler.sample(&logits, std::vector<int32_t>()) == 0) {
        (mode == 0 ? sharp_best : flat_best) += 1;
      }
    }
  }
  LLM_CHECK_MSG(sharp_best > flat_best,
                "низкая температура выбирала лучший токен реже высокой: "
                    << sharp_best << " против " << flat_best);
}
