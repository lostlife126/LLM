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

#include "core/check.h"
#include "read_file.h"
#include "data/dataset.h"
#include "nn/model.h"
#include "serialize/checkpoint.h"
#include "tokenizer/bpe.h"
#include "train/resume.h"
#include "train/trainer.h"

namespace {


}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "использование: %s <корпус.txt> <словарь.bpe> [пресет] "
                 "[шагов] [батч] [дропаут] [зерно] [выход.llmw]\n",
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
  const uint64_t seed =
      argc > 7 ? static_cast<uint64_t>(std::atoll(argv[7])) : 1234;
  const std::string output = argc > 8 ? argv[8] : std::string();

  const llm::Bpe tokenizer = llm::Bpe::load(vocab_path);
  llm::nn::ModelConfig config = llm::nn::ModelConfig::by_name(preset);

  // Словарь модели обязан совпадать со словарём токенизатора. Берём его из
  // файла: расхождение здесь проявилось бы не ошибкой, а бессмысленным
  // обучением.
  config.vocab_size = tokenizer.vocab_size();
  config.dropout = dropout;
  config.validate();

  std::printf("модель: %s\n", config.to_string().c_str());

  const std::string text = bench::read_file(corpus_path);
  const llm::data::TokenDataset dataset =
      llm::data::TokenDataset::from_text(text, tokenizer, 0.1);

  llm::nn::Model model(config, seed);

  llm::train::TrainConfig train_config;
  train_config.steps = steps;
  train_config.batch_size = batch;
  train_config.warmup_steps = steps / 20 + 1;
  train_config.checkpoint_every = steps / 4 > 0 ? steps / 4 : steps;
  train_config.seed = seed;

  // Имя чекпоинта включает дропаут: иначе прогоны с разными его значениями
  // затирали бы друг друга, и сравнивать было бы нечего.
  std::ostringstream name;
  name << "data/" << preset;
  if (dropout > 0.0f) {
    name << "_drop" << dropout;
  }
  name << ".llmw";
  train_config.checkpoint_path = output.empty() ? name.str() : output;

  // Снимок для возобновления кладётся рядом с чекпоинтом. Если он уже есть,
  // обучение продолжится с него — это важнее неожиданности: прогон на несколько
  // часов переживает и перезапуск машины, и случайный Ctrl-C, а начать заново
  // всегда можно, удалив снимок.
  train_config.resume_path = train_config.checkpoint_path + ".resume";

  const bool continuing = llm::train::resume_exists(train_config.resume_path);
  if (continuing) {
    std::printf("найден снимок %s — обучение продолжится с него\n",
                train_config.resume_path.c_str());
  }

  // Защита от того, что уже случилось однажды: короткий проверочный прогон
  // запустили с тем же пресетом, и он молча затёр модель, обучавшуюся два
  // часа. Заметно это стало только по качеству текста.
  //
  // Отказ, а не предупреждение: предупреждение в потоке вывода обучения никто
  // не прочтёт. Явно указанный путь снимает проверку — значит намерение
  // выражено.
  if (output.empty() && !continuing) {
    std::ifstream existing(train_config.checkpoint_path.c_str(),
                           std::ios::binary);
    if (existing.good()) {
      existing.close();
      llm::nn::Model probe(
          llm::serialize::read_config(train_config.checkpoint_path), 1);
      const int64_t done =
          llm::serialize::load_checkpoint(train_config.checkpoint_path, &probe);
      LLM_CHECK_MSG(done < steps,
                    "в " << train_config.checkpoint_path << " лежит модель, "
                         << "обученная до шага " << done
                         << ", а этот прогон дойдёт только до " << steps
                         << ". Перезаписывать её молча нельзя; укажите путь "
                            "последним аргументом, если так и задумано");
    }
  }

  const llm::train::TrainReport report =
      llm::train::train(&model, dataset, train_config);

  std::printf("\nготово за %.1f мин\n", report.seconds / 60.0);
  if (report.resumed_from > 0) {
    std::printf("продолжено с шага %lld\n",
                static_cast<long long>(report.resumed_from));
  }
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
