#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "core/random.h"
#include "testing.h"
#include "tokenizer/bpe.h"

namespace {

std::vector<std::string> chunk_strings(llm::StrView text) {
  const std::vector<llm::StrView> chunks = llm::Bpe::pre_tokenize(text);
  std::vector<std::string> result;
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    result.push_back(chunks[i].to_string());
  }
  return result;
}

// Небольшой корпус с повторами: на нём BPE обязан выучить осмысленные
// слияния, и его хватает, чтобы тесты шли быстро.
//
// Словарь корпуса намеренно шире, чем нужно тестам: если уникальных слов
// мало, обучение упирается в потолок (сливать больше нечего) раньше, чем
// достигнет запрошенного размера, и тесты начинают проверять не то, что
// написано в их названии.
std::string sample_corpus() {
  const char* lines[] = {
      "the quick brown fox jumps over the lazy dog\n",
      "the dog barks and the fox runs away quickly\n",
      "a wandering merchant sells silver rings and copper cups\n",
      "the river carries autumn leaves toward the distant ocean\n",
      "children gather stones along the quiet mountain path\n",
      "the old library keeps forgotten letters in wooden drawers\n",
      "morning light touches the frozen windows of the tower\n",
      "travelers exchange stories about storms and hidden harbors\n",
      "the gardener plants seeds before the heavy rains arrive\n",
      "silent horses wait beside the broken stone bridge\n",
  };
  const std::size_t count = sizeof(lines) / sizeof(lines[0]);
  std::string text;
  for (int repeat = 0; repeat < 40; ++repeat) {
    for (std::size_t i = 0; i < count; ++i) {
      text += lines[i];
    }
  }
  return text;
}

}  // namespace

LLM_TEST(Bpe, EmptyVocabIsJustBytes) {
  const llm::Bpe bpe;
  LLM_CHECK_EQ(bpe.vocab_size(), llm::Bpe::kFirstMergeToken);
  LLM_CHECK_EQ(bpe.merge_count(), 0);

  // Без слияний каждый байт остаётся отдельным токеном.
  const std::vector<int32_t> ids = bpe.encode("abc");
  LLM_CHECK_EQ(ids.size(), static_cast<std::size_t>(3));
  LLM_CHECK_EQ(ids[0], static_cast<int32_t>('a'));
  LLM_CHECK_EQ(ids[2], static_cast<int32_t>('c'));
}

LLM_TEST(Bpe, SpecialTokensHaveFixedIds) {
  // Номера спецтокенов не должны зависеть от размера словаря: иначе словари
  // разных размеров стали бы несовместимы между собой.
  const llm::Bpe small = llm::Bpe::train(sample_corpus(), 300, false);
  const llm::Bpe large = llm::Bpe::train(sample_corpus(), 400, false);
  LLM_CHECK_EQ(llm::Bpe::kBos, 256);
  LLM_CHECK_EQ(llm::Bpe::kEos, 257);
  LLM_CHECK(small.token_debug_string(llm::Bpe::kBos) == "<bos>");
  LLM_CHECK(large.token_debug_string(llm::Bpe::kEos) == "<eos>");
  // И они ничего не декодируют.
  LLM_CHECK(small.decode({llm::Bpe::kBos, llm::Bpe::kEos}).empty());
}

LLM_TEST(Bpe, PreTokenizeKeepsLeadingSpaceWithWord) {
  // Ведущий пробел — часть слова. Для языка «начало слова» существенная
  // позиция, и токен " the" должен отличаться от "the" внутри слова.
  const std::vector<std::string> chunks = chunk_strings("the cat");
  LLM_CHECK_EQ(chunks.size(), static_cast<std::size_t>(2));
  LLM_CHECK(chunks[0] == "the");
  LLM_CHECK(chunks[1] == " cat");
}

LLM_TEST(Bpe, PreTokenizeSeparatesPunctuationAndDigits) {
  const std::vector<std::string> chunks = chunk_strings("cat, 42 dogs!");
  LLM_CHECK_EQ(chunks.size(), static_cast<std::size_t>(5));
  LLM_CHECK(chunks[0] == "cat");
  LLM_CHECK(chunks[1] == ",");
  LLM_CHECK(chunks[2] == " 42");
  LLM_CHECK(chunks[3] == " dogs");
  LLM_CHECK(chunks[4] == "!");
}

LLM_TEST(Bpe, PreTokenizeGroupsWhitespaceRuns) {
  // Перевод строки и отступ не приклеиваются к слову: иначе каждое слово в
  // начале строки получило бы свой отдельный токен.
  const std::vector<std::string> chunks = chunk_strings("a\n  b");
  LLM_CHECK_EQ(chunks.size(), static_cast<std::size_t>(3));
  LLM_CHECK(chunks[0] == "a");
  LLM_CHECK(chunks[1] == "\n  ");
  LLM_CHECK(chunks[2] == "b");
}

LLM_TEST(Bpe, PreTokenizeKeepsUtf8Together) {
  // Байты со старшим битом считаются буквами, поэтому кириллическое слово
  // остаётся одним куском и может быть слито в осмысленный токен.
  const std::vector<std::string> chunks = chunk_strings("привет мир");
  LLM_CHECK_EQ(chunks.size(), static_cast<std::size_t>(2));
  LLM_CHECK(chunks[0] == "привет");
  LLM_CHECK(chunks[1] == " мир");
}

LLM_TEST(Bpe, PreTokenizeCoversInputExactly) {
  // Куски обязаны покрывать текст целиком и без перекрытий: иначе
  // токенизация потеряла бы или продублировала байты.
  const std::string text = "Hello,  world!\n\tTab 123 конец.";
  std::string rejoined;
  const std::vector<std::string> chunks = chunk_strings(text);
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    rejoined += chunks[i];
  }
  LLM_CHECK(rejoined == text);
}

LLM_TEST(Bpe, TrainingLearnsFrequentSequences) {
  const std::string corpus = sample_corpus();
  const llm::Bpe bpe = llm::Bpe::train(corpus, 400, false);

  LLM_CHECK_GT(bpe.merge_count(), 50);
  LLM_CHECK_EQ(bpe.vocab_size(), 400);

  // На таком корпусе " the" обязано стать одним токеном: это самая частая
  // последовательность в нём.
  bool learned_the = false;
  for (int32_t id = llm::Bpe::kFirstMergeToken; id < bpe.vocab_size(); ++id) {
    if (bpe.token_bytes(id) == " the") {
      learned_the = true;
    }
  }
  LLM_CHECK(learned_the);
}

LLM_TEST(Bpe, TrainingCompressesText) {
  const std::string corpus = sample_corpus();
  const llm::Bpe bpe = llm::Bpe::train(corpus, 400, false);
  const std::vector<int32_t> ids = bpe.encode(corpus);
  // Смысл словаря в том, чтобы последовательность стала короче.
  LLM_CHECK_LT(ids.size(), corpus.size() / 2);
}

LLM_TEST(Bpe, RoundtripOnTrainingCorpus) {
  const std::string corpus = sample_corpus();
  const llm::Bpe bpe = llm::Bpe::train(corpus, 400, false);
  LLM_CHECK(bpe.decode(bpe.encode(corpus)) == corpus);
}

LLM_TEST(Bpe, RoundtripOnUnseenText) {
  // Байтовый алфавит замкнут: любой текст представим, даже если в обучении
  // не встречалось ни одного его слова.
  const llm::Bpe bpe = llm::Bpe::train(sample_corpus(), 400, false);
  const std::string unseen = "Совершенно другой текст: 12345 !@#$%^&*()";
  LLM_CHECK(bpe.decode(bpe.encode(unseen)) == unseen);
}

LLM_TEST(Bpe, RoundtripOnEveryByteValue) {
  // Главное свойство byte-level BPE: неизвестных символов не бывает.
  // Проверяем все 256 значений байта, включая нулевой.
  const llm::Bpe bpe = llm::Bpe::train(sample_corpus(), 400, false);
  std::string all_bytes;
  for (int value = 0; value < 256; ++value) {
    all_bytes += static_cast<char>(value);
  }
  const std::string decoded = bpe.decode(bpe.encode(all_bytes));
  LLM_CHECK_EQ(decoded.size(), static_cast<std::size_t>(256));
  LLM_CHECK(decoded == all_bytes);
}

LLM_TEST(Bpe, RoundtripOnRandomBytes) {
  const llm::Bpe bpe = llm::Bpe::train(sample_corpus(), 400, false);
  llm::Rng rng(4242);
  for (int trial = 0; trial < 40; ++trial) {
    std::string noise;
    const std::size_t length = static_cast<std::size_t>(rng.index(64));
    for (std::size_t i = 0; i < length; ++i) {
      noise += static_cast<char>(rng.index(256));
    }
    LLM_CHECK_MSG(bpe.decode(bpe.encode(noise)) == noise,
                  "roundtrip не совпал на случайной строке длины " << length);
  }
}

LLM_TEST(Bpe, RoundtripOnEmptyInput) {
  const llm::Bpe bpe = llm::Bpe::train(sample_corpus(), 400, false);
  LLM_CHECK(bpe.encode("").empty());
  LLM_CHECK(bpe.decode({}).empty());
}

LLM_TEST(Bpe, EncodingIsDeterministic) {
  const llm::Bpe first = llm::Bpe::train(sample_corpus(), 400, false);
  const llm::Bpe second = llm::Bpe::train(sample_corpus(), 400, false);
  // Обучение на одном корпусе обязано давать один и тот же словарь: без
  // явного правила разрешения ничьих порядок обхода хеш-таблицы сделал бы
  // результат случайным.
  LLM_CHECK_EQ(first.merge_count(), second.merge_count());
  for (int32_t id = llm::Bpe::kFirstMergeToken; id < first.vocab_size(); ++id) {
    LLM_CHECK_MSG(first.token_bytes(id) == second.token_bytes(id),
                  "словари разошлись на токене " << id);
  }
  LLM_CHECK(first.encode("the dog") == second.encode("the dog"));
}

LLM_TEST(Bpe, SaveAndLoadPreserveVocabulary) {
  const std::string path = "test_vocab.bpe";
  const llm::Bpe original = llm::Bpe::train(sample_corpus(), 400, false);
  original.save(path);

  const llm::Bpe restored = llm::Bpe::load(path);
  LLM_CHECK_EQ(restored.vocab_size(), original.vocab_size());
  LLM_CHECK_EQ(restored.merge_count(), original.merge_count());

  const std::string text = "the lazy fox and the dog";
  LLM_CHECK(restored.encode(text) == original.encode(text));
  LLM_CHECK(restored.decode(restored.encode(text)) == text);

  std::remove(path.c_str());
}

LLM_TEST(Bpe, LoadRejectsBrokenFile) {
  const std::string path = "test_broken.bpe";
  {
    std::ofstream file(path.c_str());
    file << "notllmbpe 1\nmerges 0\n";
  }
  LLM_EXPECT_THROWS(llm::Bpe::load(path));
  std::remove(path.c_str());

  {
    // Слияние ссылается на токен, которого на этом шаге ещё не существует.
    std::ofstream file(path.c_str());
    file << "llmbpe 1\nmerges 1\n999 100\n";
  }
  LLM_EXPECT_THROWS(llm::Bpe::load(path));
  std::remove(path.c_str());
}

LLM_TEST(Bpe, TrainingRejectsTooSmallVocab) {
  LLM_EXPECT_THROWS(llm::Bpe::train(sample_corpus(), 100, false));
}

LLM_TEST(Bpe, TrainingStopsWhenNothingRepeats) {
  // Если запросить словарь больше, чем есть повторяющихся пар, обучение
  // обязано остановиться само, а не заполнять словарь мусором.
  const llm::Bpe bpe = llm::Bpe::train("abcdef", 5000, false);
  LLM_CHECK_LT(bpe.vocab_size(), 5000);
  LLM_CHECK(bpe.decode(bpe.encode("abcdef")) == "abcdef");
}

LLM_TEST(Bpe, MergesNeverCrossChunkBoundary) {
  // Инвариант разбиения: байты любого выученного токена обязаны сами
  // разбиваться ровно на один кусок. Токен, распадающийся на два, означал бы,
  // что слияние перешагнуло границу слова — а именно этого разбиение и не
  // должно допускать.
  //
  // Соблазнительно сформулировать проще: «в токене не бывает пробела, кроме
  // первого байта». Это неверно. Подряд идущие пробелы образуют собственный
  // кусок, и токен из двух пробелов или из перевода строки с отступом вполне
  // законен — на настоящем корпусе такие и появляются.
  //
  // Корпус здесь намеренно содержит и двойные пробелы, и отступы.
  std::string corpus;
  for (int repeat = 0; repeat < 40; ++repeat) {
    corpus += sample_corpus();
    corpus += "  double  spaced  words  here\n\n\ttabbed line follows\n";
  }

  const llm::Bpe bpe = llm::Bpe::train(corpus, 600, false);
  LLM_CHECK_GT(bpe.merge_count(), 100);
  for (int32_t id = llm::Bpe::kFirstMergeToken; id < bpe.vocab_size(); ++id) {
    LLM_CHECK_MSG(llm::Bpe::pre_tokenize(bpe.token_bytes(id)).size() == 1,
                  "токен " << bpe.token_debug_string(id)
                           << " распадается на несколько кусков");
  }
}
