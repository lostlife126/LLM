// Обучение словаря BPE и осмотр результата.
//
// Запуск:
//   bpe_train <корпус.txt> <словарь.bpe> [размер_словаря]
//
// После обучения печатает статистику сжатия и примеры выученных токенов —
// по ним сразу видно, разумно ли разбиение.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "args.h"
#include "read_file.h"
#include "tokenizer/bpe.h"

int main(int argc, char** argv) {
  const char* const usage = "<корпус.txt> <словарь.bpe> [размер_словаря]";
  if (argc < 3) {
    std::fprintf(stderr, "использование: %s %s\n", argv[0], usage);
    return 1;
  }
  bench::expect_at_most(argc, 3, usage);
  const std::string corpus_path = argv[1];
  const std::string vocab_path = argv[2];
  const int vocab_size =
      argc > 3
          ? static_cast<int>(bench::parse_int64(argv[3], "размер словаря"))
          : 1024;

  const std::string text = bench::read_file(corpus_path);
  std::printf("обучение словаря на %zu байтах, цель %d токенов\n", text.size(),
              vocab_size);

  const llm::Bpe bpe = llm::Bpe::train(text, vocab_size, true);
  bpe.save(vocab_path);

  const std::vector<int32_t> encoded = bpe.encode(text);
  const std::string decoded = bpe.decode(encoded);

  std::printf("\nсловарь: %d токенов (%d слияний)\n", bpe.vocab_size(),
              bpe.merge_count());
  std::printf(
      "текст: %zu байт -> %zu токенов, сжатие %.2fx\n", text.size(),
      encoded.size(),
      static_cast<double>(text.size()) / static_cast<double>(encoded.size()));
  std::printf("побайтовый roundtrip: %s\n",
              decoded == text ? "совпал" : "РАСХОЖДЕНИЕ");

  std::printf("\nсамые длинные выученные токены:\n");
  std::vector<std::pair<std::size_t, int32_t>> by_length;
  for (int32_t id = llm::Bpe::kFirstMergeToken; id < bpe.vocab_size(); ++id) {
    by_length.push_back(std::make_pair(bpe.token_bytes(id).size(), id));
  }
  std::sort(by_length.begin(), by_length.end());
  const std::size_t show = by_length.size() < 15 ? by_length.size() : 15;
  for (std::size_t i = 0; i < show; ++i) {
    const int32_t id = by_length[by_length.size() - 1 - i].second;
    std::printf("  %5d  %s\n", id, bpe.token_debug_string(id).c_str());
  }

  std::printf("\nпервые выученные слияния:\n");
  for (int32_t id = llm::Bpe::kFirstMergeToken;
       id < llm::Bpe::kFirstMergeToken + 12 && id < bpe.vocab_size(); ++id) {
    std::printf("  %5d  %s\n", id, bpe.token_debug_string(id).c_str());
  }
  return 0;
}
