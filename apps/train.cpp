// Обучение модели.
//
// Запуск:
//   train <корпус.txt> <словарь.bpe> [пресет] [шагов] [батч]
//
// Пресет задаёт форму модели; размер словаря берётся из файла словаря, а не
// из пресета — иначе они разошлись бы молча.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "data/dataset.h"
#include "nn/model.h"
#include "serialize/checkpoint.h"
#include "tokenizer/bpe.h"
#include "train/trainer.h"

namespace {

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
                 "использование: %s <корпус.txt> <словарь.bpe> [пресет] "
                 "[шагов] [батч] [дропаут]\n",
                 argv[0]);
    return 1;
  }
  const std::string corpus_path = argv[1];
  const std::string vocab_path = argv[2];
  const std::string preset = argc > 3 ? argv[3] : "nano";
  const int64_t steps = argc > 4 ? std::atoll(argv[4]) : 2000;
  const int64_t batch = argc > 5 ? std::atoll(argv[5]) : 16;
  const float dropout =
      argc > 6 ? static_cast<float>(std::atof(argv[6])) : 0.0f;

  const llm::Bpe tokenizer = llm::Bpe::load(vocab_path);
  llm::nn::ModelConfig config = llm::nn::ModelConfig::by_name(preset);

  // Словарь модели обязан совпадать со словарём токенизатора. Берём его из
  // файла: расхождение здесь проявилось бы не ошибкой, а бессмысленным
  // обучением.
  config.vocab_size = tokenizer.vocab_size();
  config.dropout = dropout;
  config.validate();

  std::printf("модель: %s\n", config.to_string().c_str());

  const std::string text = read_file(corpus_path);
  const llm::data::TokenDataset dataset =
      llm::data::TokenDataset::from_text(text, tokenizer, 0.1);

  llm::nn::Model model(config, 1234);

  llm::train::TrainConfig train_config;
  train_config.steps = steps;
  train_config.batch_size = batch;
  train_config.warmup_steps = steps / 20 + 1;
  train_config.checkpoint_every = steps / 4 > 0 ? steps / 4 : steps;
  // Имя чекпоинта включает дропаут: иначе прогоны с разными его значениями
  // затирали бы друг друга, и сравнивать было бы нечего.
  std::ostringstream name;
  name << "data/" << preset;
  if (dropout > 0.0f) {
    name << "_drop" << dropout;
  }
  name << ".llmw";
  train_config.checkpoint_path = name.str();

  const llm::train::TrainReport report =
      llm::train::train(&model, dataset, train_config);

  std::printf("\nготово за %.1f мин\n", report.seconds / 60.0);
  std::printf("потери: начало %.4f -> конец %.4f\n",
              report.train_loss.empty() ? 0.0f : report.train_loss.front(),
              report.final_train_loss);
  std::printf("проверочные потери: %.4f\n", report.final_validation_loss);

  // Расхождение лучшего шага с последним — это переобучение, видимое одним
  // числом. Молчать о нём нельзя: прогон выглядит успешным (обучающие потери
  // падают до конца), а модель тем временем портится.
  if (report.best_step > 0 && report.best_step < steps) {
    std::printf(
        "ЛУЧШЕЕ БЫЛО РАНЬШЕ: %.4f на шаге %lld из %lld, дальше только хуже\n",
        report.best_validation_loss, static_cast<long long>(report.best_step),
        static_cast<long long>(steps));
    std::printf("  разница с концом прогона: %+.4f\n",
                report.final_validation_loss - report.best_validation_loss);
    std::printf("  сохранён именно лучший\n");
  }
  std::printf("чекпоинт: %s\n", train_config.checkpoint_path.c_str());
  return 0;
}
