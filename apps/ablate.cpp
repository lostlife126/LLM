// Архитектурная лаборатория: измерение вместо пересказа.
//
// Каждый вариант отличается от базового ровно одним решением, обучается с
// одинаковым бюджетом и сравнивается по перплексии на одной и той же
// проверочной выборке.
//
// Каждый вариант прогоняется несколькими зёрнами. Без этого таблица абляций
// бессмысленна: непонятно, что считать различием, а что разбросом от
// случайной инициализации. Разброс базового варианта — и есть та планка, выше
// которой различие можно обсуждать.
//
// Запуск: ablate <корпус.txt> <словарь.bpe> [шагов] [зёрен]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/check.h"
#include "core/util.h"
#include "data/dataset.h"
#include "nn/model.h"
#include "tokenizer/bpe.h"
#include "train/trainer.h"

namespace {

using llm::nn::FfnKind;
using llm::nn::ModelConfig;
using llm::nn::NormKind;
using llm::nn::PositionKind;

struct Variant {
  std::string name;
  std::string question;  // на какой вопрос отвечает этот прогон
  ModelConfig config;
};

std::vector<Variant> build_variants(int64_t vocab_size) {
  ModelConfig base = ModelConfig::ablation();
  base.vocab_size = vocab_size;

  std::vector<Variant> variants;

  Variant baseline;
  baseline.name = "базовый (Llama-style)";
  baseline.question = "RMSNorm, pre-norm, RoPE, SwiGLU, GQA 4/2, связанные";
  baseline.config = base;
  variants.push_back(baseline);

  Variant layer_norm;
  layer_norm.name = "LayerNorm";
  layer_norm.question = "даёт ли что-нибудь вычитание среднего";
  layer_norm.config = base;
  layer_norm.config.norm = NormKind::kLayerNorm;
  variants.push_back(layer_norm);

  Variant post;
  post.name = "пост-нормировка";
  post.question = "мешает ли нормировка на пути остатка";
  post.config = base;
  post.config.post_norm = true;
  variants.push_back(post);

  Variant gelu;
  gelu.name = "FFN с GELU";
  gelu.question = "стоит ли вентиль SwiGLU лишней матрицы";
  gelu.config = base;
  gelu.config.ffn = FfnKind::kGeluMlp;
  gelu.config.ffn_hidden = base.ffn_hidden_matching(FfnKind::kGeluMlp);
  variants.push_back(gelu);

  Variant learned;
  learned.name = "обучаемые позиции";
  learned.question = "выигрывает ли RoPE у таблицы позиций";
  learned.config = base;
  learned.config.position = PositionKind::kLearned;
  variants.push_back(learned);

  Variant none;
  none.name = "без позиций";
  none.question = "сколько порядка даёт одна каузальная маска";
  none.config = base;
  none.config.position = PositionKind::kNone;
  variants.push_back(none);

  Variant mha;
  mha.name = "MHA (голов ключей 4)";
  mha.question = "сколько стоит экономия на головах ключей";
  mha.config = base;
  mha.config.n_kv_heads = base.n_heads;
  variants.push_back(mha);

  Variant mqa;
  mqa.name = "MQA (голова ключей одна)";
  mqa.question = "докуда можно ужимать KV-кэш";
  mqa.config = base;
  mqa.config.n_kv_heads = 1;
  variants.push_back(mqa);

  Variant untied;
  untied.name = "несвязанные эмбеддинги";
  untied.question =
      "что даёт отдельная выходная матрица (НЕ выровнено по параметрам)";
  untied.config = base;
  untied.config.tie_embeddings = false;
  variants.push_back(untied);

  // Развязывание эмбеддингов добавляет vocab * d_model весов, поэтому строка
  // выше отвечает на вопрос «помогает ли +24% параметров», а не «помогает ли
  // развязывание». Честный ответ даёт этот вариант: те же лишние веса, но
  // потраченные на ширину FFN при связанных эмбеддингах.
  Variant wider;
  wider.name = "связанные + шире FFN";
  wider.question = "те же +24% параметров, но в FFN, а не в выходной матрице";
  wider.config = base;
  wider.config.ffn_hidden =
      base.ffn_hidden +
      (base.vocab_size * base.d_model) / (3 * base.d_model * base.n_layers);
  variants.push_back(wider);

  return variants;
}

struct Outcome {
  std::string name;
  std::string question;
  int64_t parameters = 0;
  std::vector<float> validation_loss;

  float mean() const {
    double total = 0.0;
    for (std::size_t i = 0; i < validation_loss.size(); ++i) {
      total += validation_loss[i];
    }
    return static_cast<float>(total /
                              static_cast<double>(validation_loss.size()));
  }
  float spread() const {
    if (validation_loss.size() < 2) {
      return 0.0f;
    }
    const float low =
        *std::min_element(validation_loss.begin(), validation_loss.end());
    const float high =
        *std::max_element(validation_loss.begin(), validation_loss.end());
    return high - low;
  }
};

std::string read_file(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::binary);
  if (!file.good()) {
    std::fprintf(stderr, "не удалось открыть %s\n", path.c_str());
    std::exit(1);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "использование: %s <корпус.txt> <словарь.bpe> [шагов] "
                 "[зёрен]\n",
                 argv[0]);
    return 1;
  }
  const std::string corpus_path = argv[1];
  const std::string vocab_path = argv[2];
  const int64_t steps = argc > 3 ? std::atoll(argv[3]) : 1000;
  const int seeds = argc > 4 ? std::atoi(argv[4]) : 2;
  // Необязательный фильтр по имени: позволяет догнать один вариант, не
  // пересчитывая всю таблицу.
  const std::string filter = argc > 5 ? argv[5] : std::string();

  const llm::Bpe tokenizer = llm::Bpe::load(vocab_path);
  const llm::data::TokenDataset dataset = llm::data::TokenDataset::from_text(
      read_file(corpus_path), tokenizer, 0.1);

  const std::vector<Variant> variants = build_variants(tokenizer.vocab_size());
  std::printf("абляции: %zu вариантов по %d зёрен, %lld шагов каждый\n",
              variants.size(), seeds, static_cast<long long>(steps));
  // Сброс буфера после шапки: при выводе в файл она иначе не появится до
  // первого готового прогона, и несколько минут кажется, что ничего не идёт.
  std::printf("обучающих токенов %lld, проверочных %lld\n\n",
              static_cast<long long>(dataset.train_size()),
              static_cast<long long>(dataset.validation_size()));
  std::fflush(stdout);

  std::vector<Outcome> outcomes;
  for (std::size_t v = 0; v < variants.size(); ++v) {
    if (!filter.empty() && variants[v].name.find(filter) == std::string::npos) {
      continue;
    }
    Outcome outcome;
    outcome.name = variants[v].name;
    outcome.question = variants[v].question;
    outcome.parameters = variants[v].config.parameter_count();

    for (int seed = 0; seed < seeds; ++seed) {
      llm::nn::Model model(variants[v].config,
                           static_cast<uint64_t>(1000 + seed));

      llm::train::TrainConfig train_config;
      train_config.steps = steps;
      train_config.batch_size = 16;
      train_config.warmup_steps = steps / 20 + 1;
      // Порядок данных одинаков у всех прогонов: меняется только начальная
      // инициализация, поэтому разброс отражает именно её.
      train_config.seed = 777;
      train_config.log_every = 0;
      train_config.eval_every = steps;
      train_config.eval_batches = 16;
      train_config.verbose = false;

      const llm::train::TrainReport report =
          llm::train::train(&model, dataset, train_config);
      outcome.validation_loss.push_back(report.final_validation_loss);

      std::printf("  %s зерно %d: потери %.4f  (%.1f мин)\n",
                  llm::pad_utf8(variants[v].name, 26).c_str(), seed,
                  report.final_validation_loss, report.seconds / 60.0);
      std::fflush(stdout);
    }
    outcomes.push_back(outcome);
  }

  // Итоговая таблица, отсортированная по качеству.
  std::vector<Outcome> sorted = outcomes;
  std::sort(
      sorted.begin(), sorted.end(),
      [](const Outcome& a, const Outcome& b) { return a.mean() < b.mean(); });

  LLM_CHECK_MSG(!outcomes.empty(),
                "фильтр '" << filter << "' ничего не выбрал");
  const float baseline_spread = outcomes[0].spread();

  std::printf("\n%s %10s %8s %8s %10s %8s\n",
              llm::pad_utf8("вариант", 26).c_str(), "параметров", "потери",
              "разброс", "перплексия", "к базе");
  std::printf(
      "----------------------------------------------------------------------"
      "-----------\n");
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    const float delta = sorted[i].mean() - outcomes[0].mean();
    std::printf("%s %10lld %8.4f %8.4f %10.1f %+8.4f\n",
                llm::pad_utf8(sorted[i].name, 26).c_str(),
                static_cast<long long>(sorted[i].parameters), sorted[i].mean(),
                sorted[i].spread(), std::exp(sorted[i].mean()), delta);
  }

  std::printf("\nчто проверял каждый вариант:\n");
  for (std::size_t i = 0; i < outcomes.size(); ++i) {
    std::printf("  %s %s\n", llm::pad_utf8(outcomes[i].name, 26).c_str(),
                outcomes[i].question.c_str());
  }
  std::printf(
      "\nразброс базового варианта по зёрнам: %.4f — различия меньше этого "
      "обсуждать нельзя\n",
      baseline_spread);
  return 0;
}
