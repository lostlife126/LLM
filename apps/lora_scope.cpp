// Какие проекции покрывать адаптером.
//
// `finetune` отвечает на вопрос «LoRA против полного дообучения». Эта
// программа отвечает на следующий: адаптеры на каких матрицах вообще стоит
// вешать. Классический выбор — только запросы и значения внимания; здесь он
// сравнивается с расширенными областями при одинаковом бюджете шагов.
//
// Программа написана позже таблицы, которую она считает: таблица в README
// существовала, а протокола, который можно перепроверить, за ней не стояло.
// В проекте, где сказано «утверждение, за которым не стоит воспроизводимого
// протокола, — просто утверждение», это была прореха, а не мелочь.
//
// Запуск: lora_scope <чекпоинт.llmw> <словарь.bpe> <новый_корпус.txt> [шагов]

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/util.h"
#include "args.h"
#include "read_file.h"
#include "data/dataset.h"
#include "nn/model.h"
#include "serialize/checkpoint.h"
#include "tokenizer/bpe.h"
#include "train/trainer.h"

namespace {

struct Scope {
  const char* label;
  int64_t rank;
  bool key;
  bool output;
  bool ffn;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "использование: %s <чекпоинт.llmw> <словарь.bpe> "
                 "<новый_корпус.txt> [шагов]\n",
                 argv[0]);
    return 1;
  }
  const std::string checkpoint_path = argv[1];
  const std::string vocab_path = argv[2];
  const std::string target_path = argv[3];
  const int64_t steps =
      argc > 4 ? bench::parse_int64(argv[4], "число шагов") : 500;

  const llm::Bpe tokenizer = llm::Bpe::load(vocab_path);
  const llm::nn::ModelConfig config =
      llm::serialize::read_config(checkpoint_path);
  const llm::data::TokenDataset target = llm::data::TokenDataset::from_text(
      bench::read_file(target_path), tokenizer, 0.1);

  std::printf("модель: %s\n", config.to_string().c_str());
  std::printf("новый корпус: %lld обучающих токенов, %lld проверочных\n\n",
              static_cast<long long>(target.train_size()),
              static_cast<long long>(target.validation_size()));

  // Запросы и значения включены везде: сравниваются именно расширения к ним.
  const Scope scopes[] = {
      {"q+v, ранг 8", 8, false, false, false},
      {"q+k+v+o, ранг 8", 8, true, true, false},
      {"q+v+FFN, ранг 8", 8, false, false, true},
      {"q+k+v+o+FFN, ранг 32", 32, true, true, true},
  };

  llm::train::TrainConfig train_config;
  train_config.steps = steps;
  train_config.batch_size = 16;
  train_config.warmup_steps = steps / 20 + 1;
  train_config.log_every = 0;
  train_config.eval_batches = 16;
  train_config.verbose = false;
  // Та же скорость, что у ветви LoRA в finetune: у адаптеров вклад умножается
  // на alpha / rank, а матрица B стартует с нуля, поэтому им нужно больше.
  train_config.max_learning_rate = 3e-3f;

  for (std::size_t i = 0; i < sizeof(scopes) / sizeof(scopes[0]); ++i) {
    const Scope& scope = scopes[i];
    llm::nn::Model model(config, 0);
    llm::serialize::load_checkpoint(checkpoint_path, &model);

    llm::nn::LoraConfig lora;
    lora.rank = scope.rank;
    lora.attention_key = scope.key;
    lora.attention_output = scope.output;
    lora.ffn = scope.ffn;
    model.enable_lora(lora);

    const llm::train::TrainReport report =
        llm::train::train(&model, target, train_config);

    // Ширина через pad_utf8, а не через %-22s: printf считает ширину в
    // байтах, а «ранг» занимает восемь байт на четыре знака. В таблице из
    // четырёх строк это разъезжается на глазах — у самой длинной подписи
    // места не оставалось вовсе. Доля обучаемых тоже с фиксированной
    // шириной: без неё «потери» ехали следом.
    std::printf("%s обучаемых %7lld (%6.2f%%)  потери %.4f  %.1f мин\n",
                llm::pad_utf8(scope.label, 22).c_str(),
                static_cast<long long>(model.trainable_parameter_count()),
                100.0 * static_cast<double>(model.trainable_parameter_count()) /
                    static_cast<double>(model.parameter_count()),
                report.final_validation_loss, report.seconds / 60.0);
    std::fflush(stdout);
  }
  return 0;
}
