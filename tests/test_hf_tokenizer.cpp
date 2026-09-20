// Токенизатор чужой модели.
//
// Словарь здесь маленький и выписан руками, поэтому результат каждого разбора
// можно проверить рассуждением, а не сравнением с самим собой. Именно это
// нужно: совпадать обязано не устройство, а номера токенов, и ошибка в них
// ничем себя не выдаёт.

#include <cstdio>
#include <ctime>
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

LLM_TEST(HfTokenizer, ByteAlphabetMatchesTheReference) {
  // Отображение байтов в печатные символы у GPT-2 не произвольное, а задано
  // конкретным правилом, и словари чужих моделей записаны именно в нём.
  // Опорные точки взяты не из головы, а посчитаны эталонной функцией
  // bytes_to_unicode: печатные знаки идут сами в себя, остальные — подряд от
  // U+0100 в порядке возрастания байта.
  //
  // Видно здесь и то, почему проверять надо на краях: 0x7F непечатный и
  // уезжает в U+0121, а 0xFF печатный и остаётся собой; 0xAD (мягкий перенос)
  // выпадает из диапазона печатных, а соседние 0xAC и 0xAE — нет.
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(kTinyTokenizer, "проба");

  struct Anchor {
    int byte;
    const char* utf8;
  };
  const Anchor anchors[] = {
      {0x00, "\xC4\x80"},  // U+0100
      {0x0A, "\xC4\x8A"},  // U+010A, перевод строки
      {0x20, "\xC4\xA0"},  // U+0120, пробел — тот самый Ġ
      {0x21, "!"},         // печатный, сам в себя
      {0x7E, "~"},         //
      {0x7F, "\xC4\xA1"},  // U+0121, непечатный
      {0xA0, "\xC5\x82"},  // U+0142
      {0xA1, "\xC2\xA1"},  // U+00A1, печатный
      {0xAD, "\xC5\x83"},  // U+0143, мягкий перенос — не печатный
      {0xAE, "\xC2\xAE"},  // U+00AE, печатный
      {0xFF, "\xC3\xBF"},  // U+00FF, печатный
  };
  const std::size_t count = sizeof(anchors) / sizeof(anchors[0]);
  for (std::size_t i = 0; i < count; ++i) {
    // Проверяем через decode: один токен, состоящий из символа алфавита,
    // обязан дать ровно исходный байт.
    const std::string symbol = anchors[i].utf8;
    std::string patched = kTinyTokenizer;
    const std::size_t at = patched.find("\"h\": 0");
    LLM_CHECK(at != std::string::npos);
    patched = patched.substr(0, at) + "\"" + symbol + "\": 0" +
              patched.substr(at + 6);

    const llm::HfTokenizer with_symbol =
        llm::HfTokenizer::parse(patched, "проба");
    const std::string decoded = with_symbol.decode(std::vector<int32_t>(1, 0));
    LLM_CHECK_EQ(decoded.size(), static_cast<std::size_t>(1));
    LLM_CHECK_EQ(static_cast<int>(static_cast<unsigned char>(decoded[0])),
                 anchors[i].byte);
  }
  (void)tokenizer;
}

namespace {

// Словарь с полным байтовым алфавитом, собранный по правилу GPT-2, выписанному
// здесь заново. Заново — потому что сверять реализацию с самой собой
// бессмысленно: общая ошибка прошла бы круг незамеченной.
std::string full_alphabet_tokenizer() {
  std::string vocab;
  int next = 256;
  for (int byte = 0; byte < 256; ++byte) {
    const bool printable = (byte >= '!' && byte <= '~') ||
                           (byte >= 0xA1 && byte <= 0xAC) || (byte >= 0xAE);
    const int code = printable ? byte : next++;

    // Символ записывается escape-последовательностью: так он попадает в JSON
    // независимо от того, печатный он или нет.
    char escaped[16];
    std::snprintf(escaped, sizeof(escaped), "\\u%04x", code);
    if (!vocab.empty()) {
      vocab += ",";
    }
    vocab += std::string("\"") + escaped + "\":" + std::to_string(byte);
  }
  return std::string(
             "{\"added_tokens\":[],\"pre_tokenizer\":{\"type\":\"ByteLevel\","
             "\"add_prefix_space\":false,\"use_regex\":true},"
             "\"decoder\":{\"type\":\"ByteLevel\"},"
             "\"model\":{\"type\":\"BPE\",\"dropout\":null,"
             "\"unk_token\":null,\"vocab\":{") +
         vocab + "},\"merges\":[]}}";
}

}  // namespace

LLM_TEST(HfTokenizer, AnyBytesSurviveTheRoundTrip) {
  // Байт-левел BPE обязан быть обратим на любом входе, а не только на
  // английском тексте. Это и есть причина, по которой байты отображаются в
  // печатные символы: иначе словарь не смог бы их назвать.
  const llm::HfTokenizer tokenizer =
      llm::HfTokenizer::parse(full_alphabet_tokenizer(), "полный алфавит");
  LLM_CHECK_EQ(tokenizer.vocab_size(), static_cast<int64_t>(256));

  const char* const cases[] = {
      "Hello world, the end.",
      "привет мир",
      "\xF0\x9F\x98\x80 emoji",
      "  \t\n  разные   пробелы  ",
      "1234567890",
      "don't — тире и кавычки «ёлочки»",
      "\xFF\xFE\x00\x01 сырые байты",
  };
  const std::size_t count = sizeof(cases) / sizeof(cases[0]);
  for (std::size_t i = 0; i < count; ++i) {
    const std::string text = cases[i];
    const std::vector<int32_t> ids = tokenizer.encode(text);
    LLM_CHECK_MSG(tokenizer.decode(ids) == text,
                  "не совпало на '" << text << "': получилось '"
                                    << tokenizer.decode(ids) << "'");
  }

  // Без слияний каждый байт — свой токен, и это удобный способ убедиться, что
  // алфавит покрывает всё: длина в токенах обязана равняться длине в байтах.
  const std::string mixed = "aж日\xF0\x9F\x98\x80";
  LLM_CHECK_EQ(tokenizer.encode(mixed).size(), mixed.size());
}

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

  // Пробел достаётся каждому обычному промежутку, а не тексту целиком.
  // В эталоне особые токены вырезаются первыми, и предтокенизатор
  // запускается на каждом оставшемся куске отдельно — значит и пробел
  // добавляется каждому. Если добавить его один раз всему тексту, второе
  // «hello» останется без пробела, и модель получит другой токен.
  const std::vector<int32_t> ids = tokenizer.encode("hello<|endoftext|>hello");
  const std::vector<int32_t> expected = {7, 11, 17, 7, 11};
  LLM_CHECK(ids == expected);
  LLM_CHECK_EQ(tokenizer.decode(ids), std::string(" hello<|endoftext|> hello"));
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

LLM_TEST(HfTokenizer, RefusesTokenIdsThatDoNotFitTheTable) {
  const auto patched = [](const std::string& from, const std::string& to) {
    std::string text = kTinyTokenizer;
    const std::size_t at = text.find(from);
    LLM_CHECK(at != std::string::npos);
    return text.substr(0, at) + to + text.substr(at + from.size());
  };

  // 2^32 — случай, ради которого проверка и написана. В JSON целое доходит до
  // 2^53, а таблица индексируется int32_t: при простом приведении этот номер
  // обращался в ноль, словарь загружался БЕЗ ошибки, и токен «h» молча
  // получал место токена, стоявшего нулевым. Ошибка ничем себя не выдаёт:
  // разбор идёт, номера выходят, модель отвечает мимо.
  LLM_EXPECT_THROWS(
      llm::HfTokenizer::parse(patched("\"h\": 0", "\"h\": 4294967296"), "проба"));

  // Номер, помещающийся в int32_t, но огромный: таблица заводится длиной с
  // него, и отказ приходил из аллокатора, ни словом не упомянув файл.
  LLM_EXPECT_THROWS(llm::HfTokenizer::parse(
      patched("\"h\": 0", "\"h\": 2000000000"), "проба"));

  // Отрицательный номер отвергался и раньше — проверка остаётся.
  LLM_EXPECT_THROWS(
      llm::HfTokenizer::parse(patched("\"h\": 0", "\"h\": -1"), "проба"));

  // То же и для добавленных токенов: они расширяют ту же таблицу и как раз до
  // своего номера.
  LLM_EXPECT_THROWS(llm::HfTokenizer::parse(
      patched("{\"id\": 17,", "{\"id\": 2000000000,"), "проба"));

  // А запас должен оставаться: словарь здесь из 18 записей, и добавленный
  // токен с номером 100 — обычное дело, его отвергать нельзя.
  const llm::HfTokenizer ok =
      llm::HfTokenizer::parse(patched("{\"id\": 17,", "{\"id\": 100,"), "проба");
  LLM_CHECK_EQ(ok.token_text(100), std::string("<|endoftext|>"));
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

LLM_TEST(HfTokenizer, MergeTakesTheLeftmostOfEqualRank) {
  // Слияния применяются по одному, и каждый раз берётся пара с наименьшим
  // рангом. Одна и та же пара может стоять в нескольких местах — тогда ранг у
  // них общий, и решает, какое место выбрано. Эталон берёт самое левое, и это
  // не безразлично.
  //
  // Словарь здесь ровно про этот случай: 'a', 'aa', 'aaa' и слияния «a a»
  // (ранг 0), «aa a» (ранг 1). Разбор «aaa»:
  //   слева: a|a|a -> aa|a (пара ранга 0 на месте 0) -> aaa (пара ранга 1);
  //   справа: a|a|a -> a|aa (та же пара на месте 1), и всё — пары (a, aa) в
  //           словаре нет, разбор кончается двумя токенами вместо одного.
  const char* const kTiny = R"({
  "version": "1.0",
  "added_tokens": [],
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
    "vocab": {"a": 0, "aa": 1, "aaa": 2},
    "merges": ["a a", "aa a"]
  }
})";

  const llm::HfTokenizer tokenizer = llm::HfTokenizer::parse(kTiny, "проба");
  const std::vector<int32_t> ids = tokenizer.encode("aaa");
  const std::vector<int32_t> expected = {2};
  LLM_CHECK_MSG(ids == expected,
                "«aaa» разобралось в " << ids.size()
                                       << " токенов вместо одного");
  LLM_CHECK_EQ(tokenizer.decode(ids), std::string("aaa"));
}

LLM_TEST(HfTokenizer, LongChunkIsNotQuadratic) {
  // Правило \s+ длинный пробельный участок не режет: строка из одних пробелов
  // даёт ОДИН кусок, какой бы длины она ни была. То же и с длинным словом, и
  // с длинной строкой знаков препинания. Значит разбор одного куска обязан
  // быть по силам на любом входе, а не только на словах из пяти букв.
  //
  // Проверять приходится по часам, и это здесь единственный способ: замена
  // выбранной пары везде за один проход и замена её же по одному вхождению
  // дают ПОБИТОВО один и тот же ответ — отличается только работа.
  //
  // Но не абсолютным временем, а его РОСТОМ, и это существенно. Первая
  // попытка ставила предел в полсекунды на кусок из 8000 пробелов: на x86
  // исправленный разбор укладывался в 0.002 с, а сломанный занимал 1.45 с, и
  // запас казался семисоткратным. Под qemu тот же исправленный разбор занял
  // 1.32 с — эмуляция здесь медленнее натуральной машины в тысячу раз, — и
  // тест упал на правильном коде. Отношение времён от скорости машины не
  // зависит вовсе: вчетверо длиннее кусок — вчетверо дольше линейный разбор и
  // вшестнадцатеро квадратичный.
  //
  // Измерено, а не выведено: x86 дал 4.47, 4.31, 4.45 на исправленном и
  // 15.95, 16.22, 15.91 на замене по одному вхождению; aarch64 под qemu —
  // 4.01 трижды. Предел 8 лежит ровно посередине, с запасом почти вдвое в обе
  // стороны.
  //
  // Словарь: Ġ (пробел в байтовом алфавите) и его удвоения. Слияния устроены
  // так, что каждый проход укорачивает кусок вдвое, — то есть проходов
  // логарифм, и линейное время достижимо.
  std::string vocab = "\"\\u0120\": 0";
  std::string merges;
  std::string symbol = "\\u0120";
  for (int i = 1; i <= 12; ++i) {
    merges += (merges.empty() ? "" : ", ");
    merges += "\"" + symbol + " " + symbol + "\"";
    symbol += symbol;
    vocab += ", \"" + symbol + "\": " + std::to_string(i);
  }

  const std::string text =
      "{\"version\": \"1.0\", \"added_tokens\": [], \"normalizer\": null,"
      " \"pre_tokenizer\": {\"type\": \"ByteLevel\","
      " \"add_prefix_space\": false, \"use_regex\": true},"
      " \"decoder\": {\"type\": \"ByteLevel\"},"
      " \"model\": {\"type\": \"BPE\", \"dropout\": null,"
      " \"unk_token\": null, \"continuing_subword_prefix\": null,"
      " \"end_of_word_suffix\": null, \"vocab\": {" + vocab + "},"
      " \"merges\": [" + merges + "]}}";

  const llm::HfTokenizer tokenizer = llm::HfTokenizer::parse(text, "проба");

  // Наименьшее из трёх, а не среднее: посторонняя нагрузка может добавить
  // времени, но не убавить. Без этого отношение прыгало бы от того, что
  // творится на машине, а не от того, что написано в коде.
  const auto best_seconds = [&tokenizer](std::size_t spaces,
                                         std::vector<int32_t>* ids) {
    const std::string text_spaces(spaces, ' ');
    double best = 0.0;
    for (int round = 0; round < 3; ++round) {
      const std::clock_t start = std::clock();
      *ids = tokenizer.encode(text_spaces);
      const double seconds =
          static_cast<double>(std::clock() - start) / CLOCKS_PER_SEC;
      if (round == 0 || seconds < best) {
        best = seconds;
      }
    }
    return best;
  };

  std::vector<int32_t> short_ids;
  std::vector<int32_t> long_ids;
  const double short_seconds = best_seconds(1000, &short_ids);
  const double long_seconds = best_seconds(4000, &long_ids);

  // Сначала ответ, и только потом время: тест, проверяющий одну лишь
  // скорость, проходил бы и на разборе, который ничего не разбирает.
  //
  // Наибольшее удвоение в словаре — 4096, поэтому сборка даёт по одному
  // токену на каждую единицу в двоичной записи длины, от старшей к младшей:
  // 1000 = 512 + 256 + 128 + 64 + 32 + 8, 4000 = 2048 + 1024 + 512 + 256 +
  // 128 + 32. Номер токена — это степень двойки: Ġ это 0, ĠĠ это 1 и так
  // далее.
  const std::vector<int32_t> short_expected = {9, 8, 7, 6, 5, 3};
  const std::vector<int32_t> long_expected = {11, 10, 9, 8, 7, 5};
  LLM_CHECK_MSG(short_ids == short_expected,
                "1000 пробелов разобрались в " << short_ids.size()
                                               << " токенов вместо шести");
  LLM_CHECK_MSG(long_ids == long_expected,
                "4000 пробелов разобрались в " << long_ids.size()
                                               << " токенов вместо шести");
  LLM_CHECK_EQ(tokenizer.decode(long_ids), std::string(4000, ' '));

  // Деление на ноль исключено: самый быстрый замер — это разбор тысячи
  // пробелов, и в ноль тактов он не уложится ни на какой машине. Но если
  // часы вдруг окажутся грубее, молчать об этом нельзя: отношение станет
  // бесконечностью или нулём, и тест либо упадёт без причины, либо пройдёт
  // без проверки.
  LLM_CHECK_MSG(short_seconds > 0.0,
                "часы не различили разбор тысячи пробелов — проверять нечем");
  const double growth = long_seconds / short_seconds;
  LLM_CHECK_MSG(growth < 8.0,
                "кусок вчетверо длиннее разбирался в " << growth
                    << " раз дольше (" << short_seconds << " с против "
                    << long_seconds << " с) — похоже, снова квадрат от длины");
}
