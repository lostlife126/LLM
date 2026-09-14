// Токенизатор чужой модели.
//
// Словарь здесь маленький и выписан руками, поэтому результат каждого разбора
// можно проверить рассуждением, а не сравнением с самим собой. Именно это
// нужно: совпадать обязано не устройство, а номера токенов, и ошибка в них
// ничем себя не выдаёт.

#include <string>
#include <vector>

#include "testing.h"
#include "tokenizer/hf_tokenizer.h"

namespace {

// Словарь на одиннадцать букв и девять слияний. Ġ (U+0120) — это пробел в
// байтовом алфавите; в JSON он записан escape-последовательностью, как и в
// настоящих файлах.
//
// Слияния подобраны так, чтобы «hello» и « world» собирались в один токен
// каждое, причём не слева направо, а по рангу: это и надо проверить.
const char* const kTinyTokenizer = R"({
  "version": "1.0",
  "added_tokens": [
    {"id": 17, "content": "<|endoftext|>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false,
    "use_regex": true
  },
  "decoder": {"type": "ByteLevel"},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6, "\u0120": 7,
      "he": 8, "ll": 9, "hell": 10, "hello": 11,
      "\u0120w": 12, "or": 13, "ld": 14, "\u0120wor": 15, "\u0120world": 16,
      "<|endoftext|>": 17
    },
    "merges": [
      "h e", "l l", "he ll", "hell o",
      "\u0120 w", "o r", "l d", "\u0120w or", "\u0120wor ld"
    ]
  }
})";

std::string with_pretokenizer(const std::string& block) {
  std::string text = kTinyTokenizer;
  const std::string old =
      "\"pre_tokenizer\": {\n"
      "    \"type\": \"ByteLevel\",\n"
      "    \"add_prefix_space\": false,\n"
      "    \"use_regex\": true\n"
      "  }";
  const std::size_t at = text.find("\"pre_tokenizer\"");
  LLM_CHECK(at != std::string::npos);
  const std::size_t end = text.find("\"decoder\"");
  LLM_CHECK(end != std::string::npos);
  (void)old;
  return text.substr(0, at) + block + ",\n  " + text.substr(end);
}

}  // namespace

LLM_TEST(HfTokenizer, ReadsVocabularyAndMerges) {
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(kTinyTokenizer, "проба");
  LLM_CHECK_EQ(tokenizer.vocab_size(), static_cast<int64_t>(18));
  LLM_CHECK_EQ(tokenizer.token_id("hello"), 11);
  LLM_CHECK_EQ(tokenizer.token_id("нет такого"), -1);
  LLM_CHECK_EQ(tokenizer.token_text(16), std::string("\xC4\xA0world"));
  LLM_CHECK_EQ(tokenizer.added_tokens().size(), static_cast<std::size_t>(1));
}

LLM_TEST(HfTokenizer, MergesAreAppliedByRankNotLeftToRight) {
  // «hello» собирается так: h+e (ранг 0), l+l (1), he+ll (2), hell+o (3).
  // Если применять первую подходящую пару слева, порядок будет другим, и на
  // более сложном словаре результат разойдётся.
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(kTinyTokenizer, "проба");

  const std::vector<int32_t> ids = tokenizer.encode("hello world");
  LLM_CHECK_EQ(ids.size(), static_cast<std::size_t>(2));
  LLM_CHECK_EQ(ids[0], 11);  // hello
  LLM_CHECK_EQ(ids[1], 16);  // Ġworld
}

LLM_TEST(HfTokenizer, SpaceBecomesPartOfTheNextToken) {
  // Ġ в словаре — это пробел. Отдельного токена «world» без пробела здесь
  // нет вовсе, и это не оплошность словаря, а следствие разбиения.
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(kTinyTokenizer, "проба");
  LLM_CHECK_EQ(tokenizer.token_id("world"), -1);
  LLM_CHECK_EQ(tokenizer.token_id("\xC4\xA0world"), 16);
}

LLM_TEST(HfTokenizer, DecodeUndoesEncode) {
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(kTinyTokenizer, "проба");
  const std::string text = "hello world";
  LLM_CHECK_EQ(tokenizer.decode(tokenizer.encode(text)), text);

  // Разбор по одному токену: видно, что пробел действительно уехал вправо.
  LLM_CHECK_EQ(tokenizer.decode(std::vector<int32_t>(1, 11)),
               std::string("hello"));
  LLM_CHECK_EQ(tokenizer.decode(std::vector<int32_t>(1, 16)),
               std::string(" world"));
}

LLM_TEST(HfTokenizer, SpecialTokensBypassMerges) {
  // Без отдельной обработки «<|endoftext|>» распался бы на обычные токены —
  // а модель ждёт один определённый номер.
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(kTinyTokenizer, "проба");
  const std::vector<int32_t> ids = tokenizer.encode("hello<|endoftext|> world");
  LLM_CHECK_EQ(ids.size(), static_cast<std::size_t>(3));
  LLM_CHECK_EQ(ids[0], 11);
  LLM_CHECK_EQ(ids[1], 17);
  LLM_CHECK_EQ(ids[2], 16);

  LLM_CHECK_EQ(tokenizer.decode(ids), std::string("hello<|endoftext|> world"));
}

LLM_TEST(HfTokenizer, UnknownPieceIsReportedNotGuessed) {
  // В настоящем словаре есть все 256 одиночных байт, поэтому такого не
  // бывает. В урезанном — бывает, и подставлять произвольный номер нельзя:
  // модель получила бы другой текст.
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(kTinyTokenizer, "проба");
  LLM_EXPECT_THROWS(tokenizer.encode("zebra"));
}

LLM_TEST(HfTokenizer, RecognizesSequenceWithSplit) {
  // Так устроен pre_tokenizer у моделей на токенизаторе Llama 3.
  const std::string block =
      std::string(
          "\"pre_tokenizer\": {\n"
          "    \"type\": \"Sequence\",\n"
          "    \"pretokenizers\": [\n"
          "      {\"type\": \"Split\", \"pattern\": {\"Regex\": \"") +
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|"
      "\\\\p{N}{1,3}| "
      "?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]"
      "+|\\\\s+(?!\\\\S)|\\\\s+" +
      "\"}, \"behavior\": \"Isolated\", \"invert\": false},\n"
      "      {\"type\": \"ByteLevel\", \"add_prefix_space\": false, "
      "\"use_regex\": false}\n"
      "    ]\n"
      "  }";
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(with_pretokenizer(block), "проба");
  LLM_CHECK(tokenizer.rule() == llm::PretokenizeRule::kGpt4);
  LLM_CHECK_EQ(tokenizer.encode("hello world").size(),
               static_cast<std::size_t>(2));
}

LLM_TEST(HfTokenizer, RefusesUnknownSplitPattern) {
  // Главная защита: незнакомое выражение — отказ, а не разбиение по-своему.
  const std::string block =
      "\"pre_tokenizer\": {\n"
      "    \"type\": \"Sequence\",\n"
      "    \"pretokenizers\": [\n"
      "      {\"type\": \"Split\", \"pattern\": {\"Regex\": \"[a-z]+\"}, "
      "\"behavior\": \"Isolated\", \"invert\": false},\n"
      "      {\"type\": \"ByteLevel\", \"add_prefix_space\": false, "
      "\"use_regex\": false}\n"
      "    ]\n"
      "  }";
  LLM_EXPECT_THROWS(llm::HfTokenizer::parse(with_pretokenizer(block), "проба"));
}

LLM_TEST(HfTokenizer, AddPrefixSpaceChangesTheFirstToken) {
  // Настройка выглядит мелочью, а меняет первый токен любого текста: с ней
  // «hello» становится « hello». Модель, обученная с одной настройкой,
  // с другой получает не тот вход.
  const std::string block =
      "\"pre_tokenizer\": {\n"
      "    \"type\": \"ByteLevel\",\n"
      "    \"add_prefix_space\": true,\n"
      "    \"use_regex\": true\n"
      "  }";
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(with_pretokenizer(block), "проба");
  LLM_CHECK(tokenizer.add_prefix_space());
  LLM_CHECK_EQ(tokenizer.decode(tokenizer.encode("world")),
               std::string(" world"));
}

LLM_TEST(HfTokenizer, RefusesWhatChangesTheSplitSilently) {
  const auto patched = [](const std::string& from, const std::string& to) {
    std::string text = kTinyTokenizer;
    const std::size_t at = text.find(from);
    LLM_CHECK(at != std::string::npos);
    return text.substr(0, at) + to + text.substr(at + from.size());
  };

  // Не BPE.
  LLM_EXPECT_THROWS(llm::HfTokenizer::parse(
      patched("\"type\": \"BPE\"", "\"type\": \"WordPiece\""), "проба"));

  // Суффикс конца слова и продолжающий префикс меняют разбор.
  LLM_EXPECT_THROWS(
      llm::HfTokenizer::parse(patched("\"end_of_word_suffix\": null",
                                      "\"end_of_word_suffix\": \"</w>\""),
                              "проба"));
  LLM_EXPECT_THROWS(
      llm::HfTokenizer::parse(patched("\"continuing_subword_prefix\": null",
                                      "\"continuing_subword_prefix\": \"##\""),
                              "проба"));

  // Dropout вносит в разбор случайность — воспроизводимости конец.
  LLM_EXPECT_THROWS(llm::HfTokenizer::parse(
      patched("\"dropout\": null", "\"dropout\": 0.1"), "проба"));

  // unk_token при байтовом алфавите не нужен, и его наличие означает другой
  // способ разбора.
  LLM_EXPECT_THROWS(llm::HfTokenizer::parse(
      patched("\"unk_token\": null", "\"unk_token\": \"<unk>\""), "проба"));
}

LLM_TEST(HfTokenizer, MergesInPairForm) {
  // Новые версии tokenizers пишут слияния парой строк, а не одной строкой с
  // пробелом. Читаться должны обе записи.
  std::string text = kTinyTokenizer;
  const std::size_t at = text.find("\"merges\"");
  LLM_CHECK(at != std::string::npos);
  const std::size_t end = text.find("]", at);
  LLM_CHECK(end != std::string::npos);
  const std::string pairs =
      "\"merges\": [\n"
      "      [\"h\",\"e\"], [\"l\",\"l\"], [\"he\",\"ll\"], [\"hell\",\"o\"],\n"
      "      [\"\\u0120\",\"w\"], [\"o\",\"r\"], [\"l\",\"d\"], "
      "[\"\\u0120w\",\"or\"], [\"\\u0120wor\",\"ld\"]\n"
      "    ";
  text = text.substr(0, at) + pairs + text.substr(end);

  const llm::HfTokenizer tokenizer = llm::HfTokenizer::parse(text, "проба");
  const std::vector<int32_t> ids = tokenizer.encode("hello world");
  LLM_CHECK_EQ(ids.size(), static_cast<std::size_t>(2));
  LLM_CHECK_EQ(ids[0], 11);
  LLM_CHECK_EQ(ids[1], 16);
}
