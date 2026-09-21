// Перенос модели HuggingFace в наш формат.
//
// Запуск: import_hf <каталог модели> <выход.llmw> [окно]
//
// Каталог — тот, что лежит в репозитории модели: config.json рядом с
// model.safetensors. Окно по умолчанию 512; брать объявленное моделью
// (max_position_embeddings бывает и 131072) незачем — маска и KV-кэш растут
// квадратом длины, а на веса окно не влияет.
//
// Программа ничего не додумывает. Любое расхождение с тем, что мы умеем, —
// это отказ с указанием поля или тензора, а не попытка загрузить как выйдет:
// модель, собранная из неверно понятых весов, работает и выдаёт связный
// текст, и отличить его от правильного по виду нельзя.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "args.h"

#include "core/check.h"
#include "nn/model.h"
#include "serialize/checkpoint.h"
#include "serialize/hf_config.h"
#include "serialize/hf_import.h"
#include "serialize/safetensors.h"

namespace {

bool exists(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::binary);
  return file.good();
}

std::string join(const std::string& directory, const std::string& name) {
  if (directory.empty()) {
    return name;
  }
  if (directory[directory.size() - 1] == '/') {
    return directory + name;
  }
  return directory + "/" + name;
}

}  // namespace

namespace {

int run(int argc, char** argv);

}  // namespace

// Отказ здесь — это не сбой, а предусмотренный исход: каталог может оказаться
// не тем, модель — другого семейства, поле в config.json — незнакомым. Поэтому
// исключение ловится и печатается как сообщение, а не как «terminate called
// after throwing», за которым его ещё надо разглядеть.
//
// Остальные программы проекта так не делают: там на входе наши же файлы, и
// необработанное исключение с трассировкой полезнее.
int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const llm::CheckFailure& error) {
    std::fprintf(stderr, "\n%s\n", error.what());
    return 1;
  }
}

namespace {

int run(int argc, char** argv) {
  const char* const usage = "<каталог модели> <выход.llmw> [окно]";
  if (argc < 3) {
    std::fprintf(stderr, "использование: %s %s\n", argv[0], usage);
    return 1;
  }
  bench::expect_at_most(argc, 3, usage);
  const std::string directory = argv[1];
  const std::string output = argv[2];
  const int64_t max_seq_len =
      argc > 3 ? bench::parse_positive_int64(argv[3], "длина контекста") : 512;

  const std::string config_path = join(directory, "config.json");
  LLM_CHECK_MSG(exists(config_path), "в "
                                         << directory
                                         << " нет config.json — это не каталог "
                                            "модели HuggingFace");

  const std::string weights_path = join(directory, "model.safetensors");
  if (!exists(weights_path)) {
    const std::string index = join(directory, "model.safetensors.index.json");
    LLM_CHECK_MSG(!exists(index),
                  "модель разложена по нескольким файлам (есть "
                      << index
                      << "), а читать мы умеем пока один model.safetensors");
    LLM_CHECK_MSG(false, "в " << directory << " нет model.safetensors");
  }

  const llm::serialize::HfConfig config =
      llm::serialize::read_hf_config(config_path, max_seq_len);

  std::printf("модель: %s\n", config.architecture.empty()
                                  ? "(архитектура не указана)"
                                  : config.architecture.c_str());
  std::printf("форма: %s\n", config.model.to_string().c_str());
  std::printf("параметров: %lld\n",
              static_cast<long long>(config.model.parameter_count()));
  if (!config.ignored.empty()) {
    std::printf("поля config.json, не влияющие на вычисление: %zu\n",
                config.ignored.size());
  }
  std::fflush(stdout);

  const llm::serialize::SafeTensors file =
      llm::serialize::SafeTensors::load(weights_path);
  std::printf("тензоров в файле: %zu\n", file.size());
  for (std::size_t i = 0; i < file.metadata().size(); ++i) {
    std::printf("  %s: %s\n", file.metadata()[i].first.c_str(),
                file.metadata()[i].second.c_str());
  }
  std::fflush(stdout);

  // Зерно не важно: все веса будут перезаписаны. Важно, что модель создаётся
  // по прочитанной конфигурации, а не по какой-нибудь похожей.
  llm::nn::Model model(config.model, 1);
  const llm::serialize::HfImportReport report =
      llm::serialize::import_hf_weights(file, config, &model);

  std::printf("перенесено тензоров: %lld, значений: %lld\n",
              static_cast<long long>(report.tensors_used),
              static_cast<long long>(report.values_copied));
  LLM_CHECK_MSG(report.values_copied == model.parameter_count(),
                "перенесено " << report.values_copied
                              << " значений, а у модели "
                              << model.parameter_count());

  if (!report.unused.empty()) {
    // Не ошибка, но сказать надо. Обычно это lm_head.weight у модели со
    // связанными эмбеддингами или таблица частот RoPE, которую мы считаем
    // сами. Всё остальное — повод разобраться.
    std::printf("тензоры файла, оставшиеся без применения (%zu):\n",
                report.unused.size());
    for (std::size_t i = 0; i < report.unused.size() && i < 16; ++i) {
      std::printf("  %s\n", report.unused[i].c_str());
    }
  }

  llm::serialize::save_checkpoint(output, &model, 0);
  std::printf("записано: %s\n", output.c_str());

  if (config.eos_token_id >= 0) {
    std::printf("токен конца текста: %lld\n",
                static_cast<long long>(config.eos_token_id));
  }
  return 0;
}

}  // namespace
