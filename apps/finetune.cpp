// Дообучение низкоранговыми адаптерами.
//
// Берёт обученную модель, замораживает её веса, навешивает адаптеры и
// приспосабливает к новому корпусу. Для сравнения тот же прогон делается
// полным дообучением — чтобы разница в числе обучаемых параметров и в
// результате была видна, а не заявлена.
//
// Запуск: finetune <чекпоинт.llmw> <словарь.bpe> <новый_корпус.txt> [шагов]

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/check.h"

#include "args.h"
#include "read_file.h"
#include "data/dataset.h"
#include "infer/generate.h"
#include "nn/model.h"
#include "serialize/checkpoint.h"
#include "tokenizer/bpe.h"
#include "train/trainer.h"

namespace {

void show_sample(llm::nn::Model* model, const llm::Bpe& tokenizer,
                 const std::string& prompt_text) {
  llm::infer::GenerateConfig config;
  config.max_tokens = 60;
  config.sampler.temperature = 0.6f;
  config.sampler.top_k = 40;
  config.sampler.seed = 5;

  const std::vector<int32_t> prompt = tokenizer.encode(prompt_text);
  const llm::infer::GenerateResult result =
      llm::infer::generate(model, prompt, config);
  std::printf("    %s%s\n", prompt_text.c_str(),
              tokenizer.decode(result.tokens).c_str());
}

}  // namespace

int main(int argc, char** argv) {
  const char* const usage =
      "<чекпоинт.llmw> <словарь.bpe> <новый_корпус.txt> [шагов]";
  if (argc < 4) {
    std::fprintf(stderr, "использование: %s %s\n", argv[0], usage);
    return 1;
  }
  bench::expect_at_most(argc, 4, usage);
  const std::string checkpoint_path = argv[1];
  const std::string vocab_path = argv[2];
  const std::string target_path = argv[3];
  const int64_t steps =
      argc > 4 ? bench::parse_positive_int64(argv[4], "число шагов") : 500;

  const llm::Bpe tokenizer = llm::Bpe::load(vocab_path);
  const llm::nn::ModelConfig config =
      llm::serialize::read_config(checkpoint_path);
  const llm::data::TokenDataset target = llm::data::TokenDataset::from_text(
      bench::read_file(target_path), tokenizer, 0.1);

  // Батч один на все три прогона и на проверку ниже: разойдись они — и
  // проверка перестала бы отвечать за то, что меряют прогоны.
  const int64_t kBatch = 16;

  // Проверочная часть короче одного батча означает, что мерить нечем, и
  // evaluate честно вернёт нуль. Беда в том, что печатается он как «потери
  // 0.0000» — числом, неотличимым с виду от измеренного и притом лучшим из
  // возможных. Вся программа — сравнение трёх потерь между собой, так что
  // отказ здесь честнее.
  LLM_CHECK_MSG(
      target.validation_batch_count(kBatch, config.max_seq_len) > 0,
      "проверочная часть нового корпуса — "
          << target.validation_size() << " токенов, а на один батч нужно "
          << kBatch * config.max_seq_len << " (" << kBatch << " окон по "
          << config.max_seq_len << "): сравнивать будет нечего");

  std::printf("модель: %s\n", config.to_string().c_str());
  std::printf("новый корпус: %lld обучающих токенов, %lld проверочных\n\n",
              static_cast<long long>(target.train_size()),
              static_cast<long long>(target.validation_size()));

  const std::string prompt = "void ";

  // Отправная точка: обученная модель на чужом для неё корпусе.
  {
    llm::nn::Model model(config, 0);
    llm::serialize::load_checkpoint(checkpoint_path, &model);
    const float loss =
        llm::train::evaluate(&model, target, kBatch, config.max_seq_len, 16);
    std::printf("до дообучения: потери на новом корпусе %.4f\n", loss);
    std::printf("  что пишет:\n");
    show_sample(&model, tokenizer, prompt);
  }

  llm::train::TrainConfig train_config;
  train_config.steps = steps;
  train_config.batch_size = kBatch;
  train_config.warmup_steps = steps / 20 + 1;
  train_config.log_every = 0;
  train_config.eval_batches = 16;
  train_config.verbose = false;

  // Дообучение адаптерами.
  {
    llm::nn::Model model(config, 0);
    llm::serialize::load_checkpoint(checkpoint_path, &model);

    llm::nn::LoraConfig lora;
    lora.rank = 8;
    model.enable_lora(lora);

    llm::train::TrainConfig lora_config = train_config;
    // Адаптерам нужна большая скорость: их вклад умножается на alpha / rank, а
    // сама матрица B стартует с нуля.
    lora_config.max_learning_rate = 3e-3f;
    const llm::train::TrainReport report =
        llm::train::train(&model, target, lora_config);

    std::printf(
        "\nLoRA (ранг %lld): обучаемых %lld из %lld (%.2f%%), потери %.4f, "
        "%.1f мин\n",
        static_cast<long long>(lora.rank),
        static_cast<long long>(model.trainable_parameter_count()),
        static_cast<long long>(model.parameter_count()),
        100.0 * static_cast<double>(model.trainable_parameter_count()) /
            static_cast<double>(model.parameter_count()),
        report.final_validation_loss, report.seconds / 60.0);
    std::printf("  что пишет:\n");
    model.merge_lora();
    show_sample(&model, tokenizer, prompt);
  }

  // Полное дообучение — для сравнения.
  {
    llm::nn::Model model(config, 0);
    llm::serialize::load_checkpoint(checkpoint_path, &model);

    llm::train::TrainConfig full_config = train_config;
    full_config.max_learning_rate = 3e-4f;
    const llm::train::TrainReport report =
        llm::train::train(&model, target, full_config);

    std::printf(
        "\nполное дообучение: обучаемых %lld (100%%), потери %.4f, "
        "%.1f мин\n",
        static_cast<long long>(model.trainable_parameter_count()),
        report.final_validation_loss, report.seconds / 60.0);
    std::printf("  что пишет:\n");
    show_sample(&model, tokenizer, prompt);
  }
  return 0;
}
