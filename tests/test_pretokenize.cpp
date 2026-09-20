// Предтокенизация: разбиение перед слияниями BPE.
//
// Проверять это надо въедливо. Ошибка здесь не проявляется ни падением, ни
// бессмыслицей: получатся другие номера токенов, модель увидит не тот текст и
// ответит связно и мимо. А границы кусков задаются не «здравым смыслом», а
// довольно причудливым регулярным выражением с откатами, и именно его
// причуды надо воспроизвести.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "serialize/json.h"
#include "testing.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode_tables.h"
#include "tokenizer/utf8.h"

namespace {

std::vector<std::string> split(const std::string& text,
                               llm::PretokenizeRule rule) {
  const std::vector<llm::StrView> pieces = llm::pretokenize(text, rule);
  std::vector<std::string> out;
  for (std::size_t i = 0; i < pieces.size(); ++i) {
    out.push_back(pieces[i].to_string());
  }
  return out;
}

void expect_split(const std::string& text, llm::PretokenizeRule rule,
                  const std::vector<std::string>& expected) {
  const std::vector<std::string> actual = split(text, rule);
  std::string shown;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    shown += (i == 0 ? "" : " | ") + actual[i];
  }
  LLM_CHECK_MSG(actual.size() == expected.size(),
                "кусков " << actual.size() << ", ожидалось " << expected.size()
                          << ": " << shown);
  for (std::size_t i = 0; i < actual.size(); ++i) {
    LLM_CHECK_MSG(actual[i] == expected[i],
                  "кусок " << i << ": '" << actual[i] << "' вместо '"
                           << expected[i] << "' (всё: " << shown << ")");
  }
}

const llm::PretokenizeRule kGpt2 = llm::PretokenizeRule::kGpt2;
const llm::PretokenizeRule kGpt4 = llm::PretokenizeRule::kGpt4;

}  // namespace

LLM_TEST(Pretokenize, PiecesCoverTheWholeInput) {
  // Куски обязаны склеиваться обратно в исходную строку. Без этого любая
  // другая проверка бессмысленна: потерянный байт не восстановить.
  const std::string text =
      "Mixed 123 text, с кириллицей\tи\nпереводами\n\nстрок  и  пробелами.  ";
  const llm::PretokenizeRule rules[2] = {kGpt2, kGpt4};
  for (int i = 0; i < 2; ++i) {
    const std::vector<std::string> pieces = split(text, rules[i]);
    std::string joined;
    for (std::size_t j = 0; j < pieces.size(); ++j) {
      LLM_CHECK_MSG(!pieces[j].empty(), "пустой кусок ломает цикл разбора");
      joined += pieces[j];
    }
    LLM_CHECK_EQ(joined, text);
  }
}

LLM_TEST(Pretokenize, SpaceSticksToTheFollowingWord) {
  // Самая известная особенность байт-левел BPE: пробел уходит вместе со
  // следующим словом, а не с предыдущим. Отсюда и «Ġ» в чужих словарях.
  expect_split("Hello world", kGpt2, {"Hello", " world"});
  expect_split("Hello world", kGpt4, {"Hello", " world"});
}

LLM_TEST(Pretokenize, RunOfSpacesLeavesOneForTheWord) {
  // Два пробела подряд: первый уходит отдельным куском, второй достаётся
  // слову. Так работает ветвь \s+(?!\S) с откатом на один символ.
  expect_split("Hello  world", kGpt2, {"Hello", " ", " world"});
  expect_split("a    b", kGpt2, {"a", "   ", " b"});
  expect_split("a    b", kGpt4, {"a", "   ", " b"});

  // В конце текста откатываться не от чего, и пробелы берутся целиком.
  expect_split("a  ", kGpt2, {"a", "  "});
}

LLM_TEST(Pretokenize, NumbersAreCutByThreeOnlyInGpt4) {
  // Отличие, заметно улучшающее арифметику: длинное число перестаёт быть
  // одним непрозрачным токеном.
  expect_split("1234567", kGpt2, {"1234567"});
  expect_split("1234567", kGpt4, {"123", "456", "7"});
  expect_split("42", kGpt4, {"42"});
}

LLM_TEST(Pretokenize, Contractions) {
  expect_split("don't", kGpt2, {"don", "'t"});
  expect_split("we're", kGpt2, {"we", "'re"});
  expect_split("I'll", kGpt4, {"I", "'ll"});

  // GPT-4 узнаёт сокращения независимо от регистра, GPT-2 — нет. Это не
  // придирка: от этого зависят номера токенов на любом тексте с заглавными.
  expect_split("DON'T", kGpt4, {"DON", "'T"});
  expect_split("DON'T", kGpt2, {"DON", "'", "T"});
}

LLM_TEST(Pretokenize, PunctuationBeforeWordIsAttachedOnlyInGpt4) {
  // У GPT-4 перед словом может уйти любой одиночный символ, не только
  // пробел, — отсюда «(word» одним куском.
  expect_split("(word)", kGpt4, {"(word", ")"});
  expect_split("(word)", kGpt2, {"(", "word", ")"});

  // Но пробел плюс скобка остаются вместе, а слово отделяется: одиночный
  // необязательный символ уже потрачен на пробел.
  expect_split(" (word", kGpt4, {" (", "word"});
}

LLM_TEST(Pretokenize, NewlinesAreTheirOwnPieceInGpt4) {
  // У GPT-4 перевод строки обрабатывается отдельной ветвью, у GPT-2 попадает
  // в общий разряд пробелов и дробится на части.
  expect_split("a\n\nb", kGpt4, {"a", "\n\n", "b"});
  expect_split("a\n\nb", kGpt2, {"a", "\n", "\n", "b"});

  // Пробелы перед переводом строки прилипают к нему, а не к следующей строке.
  expect_split("a  \n  b", kGpt4, {"a", "  \n", " ", " b"});
}

LLM_TEST(Pretokenize, NonAsciiLettersAreLetters) {
  // Ради этого и понадобилась таблица категорий: без неё кириллица попала бы
  // в разряд «прочих символов» и разбилась бы совсем иначе.
  expect_split("привет мир", kGpt2, {"привет", " мир"});
  expect_split("日本語", kGpt2, {"日本語"});
  expect_split("Ünïcödé", kGpt2, {"Ünïcödé"});

  // Проверка на то, что таблица вообще про то, про что надо.
  LLM_CHECK(llm::is_unicode_letter(0x043F));  // п
  LLM_CHECK(llm::is_unicode_letter(0x65E5));  // 日
  LLM_CHECK(!llm::is_unicode_letter('1'));
  LLM_CHECK(llm::is_unicode_number('7'));
  LLM_CHECK(llm::is_unicode_number(0x0666));  // арабо-индийская шестёрка
  LLM_CHECK(!llm::is_unicode_number(0x043F));
  LLM_CHECK(!llm::is_unicode_letter(' '));
}

LLM_TEST(Pretokenize, BrokenUtf8DoesNotHangOrCrash) {
  // Текст приходит снаружи и правильным UTF-8 быть не обязан. Требование
  // простое: разбор двигается вперёд и ничего не теряет.
  const std::string broken =
      std::string("ab\xC3") + "\x28" + "cd\xFF\xFE" + "e";
  const llm::PretokenizeRule rules[2] = {kGpt2, kGpt4};
  for (int i = 0; i < 2; ++i) {
    const std::vector<std::string> pieces = split(broken, rules[i]);
    std::string joined;
    for (std::size_t j = 0; j < pieces.size(); ++j) {
      LLM_CHECK(!pieces[j].empty());
      joined += pieces[j];
    }
    LLM_CHECK_EQ(joined, broken);
  }
}

LLM_TEST(Pretokenize, Utf8Decoding) {
  const std::string text = "aж日\xF0\x9F\x98\x80";
  llm::Utf8Char ch = llm::decode_utf8(text.data(), text.size());
  LLM_CHECK_EQ(ch.code, static_cast<uint32_t>('a'));
  LLM_CHECK_EQ(ch.bytes, 1);

  ch = llm::decode_utf8(text.data() + 1, text.size() - 1);
  LLM_CHECK_EQ(ch.code, static_cast<uint32_t>(0x0436));
  LLM_CHECK_EQ(ch.bytes, 2);

  ch = llm::decode_utf8(text.data() + 3, text.size() - 3);
  LLM_CHECK_EQ(ch.code, static_cast<uint32_t>(0x65E5));
  LLM_CHECK_EQ(ch.bytes, 3);

  ch = llm::decode_utf8(text.data() + 6, text.size() - 6);
  LLM_CHECK_EQ(ch.code, static_cast<uint32_t>(0x1F600));
  LLM_CHECK_EQ(ch.bytes, 4);

  // Избыточно длинная запись нуля, суррогат и обрубленная последовательность
  // — всё это не символы, и каждая обязана дать ровно один байт, чтобы разбор
  // двигался.
  const std::string overlong = "\xC0\x80";
  ch = llm::decode_utf8(overlong.data(), overlong.size());
  LLM_CHECK(!ch.valid);
  LLM_CHECK_EQ(ch.bytes, 1);

  const std::string surrogate = "\xED\xA0\x80";
  ch = llm::decode_utf8(surrogate.data(), surrogate.size());
  LLM_CHECK(!ch.valid);
  LLM_CHECK_EQ(ch.bytes, 1);

  const std::string truncated = "\xE6\x97";
  ch = llm::decode_utf8(truncated.data(), truncated.size());
  LLM_CHECK(!ch.valid);
  LLM_CHECK_EQ(ch.bytes, 1);
}

LLM_TEST(Pretokenize, RecognizesKnownPatternsAndRefusesOthers) {
  llm::PretokenizeRule rule = kGpt4;
  LLM_CHECK(llm::recognize_pretokenizer(
      llm::pretokenizer_pattern(llm::PretokenizeRule::kGpt2), &rule));
  LLM_CHECK(rule == llm::PretokenizeRule::kGpt2);

  LLM_CHECK(llm::recognize_pretokenizer(
      llm::pretokenizer_pattern(llm::PretokenizeRule::kGpt4), &rule));
  LLM_CHECK(rule == llm::PretokenizeRule::kGpt4);

  // Похожее, но другое выражение узнавать нельзя: разбиение получится своё,
  // и заметить это будет нечем.
  LLM_CHECK(!llm::recognize_pretokenizer("'s|'t| ?\\p{L}+", &rule));
  LLM_CHECK(!llm::recognize_pretokenizer("", &rule));
}

LLM_TEST(Pretokenize, EmptyInput) {
  LLM_CHECK_EQ(split("", kGpt2).size(), static_cast<std::size_t>(0));
  LLM_CHECK_EQ(split("", kGpt4).size(), static_cast<std::size_t>(0));
}

LLM_TEST(Pretokenize, BrokenUtf8GoesToOtherSymbols) {
  // Обрезанный посередине буквы текст. «аб» — это D0 B0 D0 B1, и без
  // последнего байта хвостовой D0 остаётся сам по себе.
  //
  // Соблазн есть спросить разряд у того, что вернул разбор: у D0 кодовая
  // точка получится 0x00D0, а это буква Ð. Тогда мусор приклеился бы к слову,
  // и граница куска зависела бы от того, какой именно байт уцелел. Байт 0xA0
  // тем же путём ушёл бы в пробелы, а 0xB2 — в цифры.
  const std::string cut = "\xD0\xB0\xD0";
  expect_split(cut, kGpt2, {"\xD0\xB0", "\xD0"});
  expect_split(cut, kGpt4, {"\xD0\xB0", "\xD0"});

  // Тот же байт между буквами: у GPT-2 он обязан отделиться с обеих сторон,
  // потому что необязательный ведущий символ там — только пробел.
  expect_split("a\xD0" "b", kGpt2, {"a", "\xD0", "b"});
  // Байт, который иначе сошёл бы за пробел.
  expect_split("a\xA0" "b", kGpt2, {"a", "\xA0", "b"});
  // И за цифру: 0xB2 — это верхний индекс «два».
  expect_split("a\xB2" "b", kGpt2, {"a", "\xB2", "b"});

  // Годная последовательность разбирается как прежде — проверка на valid
  // касается только мусора.
  expect_split("аб", kGpt2, {"аб"});
  // Неразрывный пробел — настоящий пробельный символ, а не «прочий», и в
  // образце GPT-2 к слову прилипает только обычный пробел, поэтому он уходит
  // отдельным куском.
  expect_split("а\xC2\xA0\xD0\xB1", kGpt2, {"а", "\xC2\xA0", "б"});
}

LLM_TEST(Pretokenize, Utf8RoundTripsForEveryCodePoint) {
  // append_utf8 не проверялся ни одним тестом, хотя через него проходит вся
  // побайтовая азбука токенизатора: hf_tokenizer собирает ею символы из
  // байтов, и ошибка там сдвинула бы токенизацию целиком.
  //
  // Перебор здесь возможен, а значит обязателен — как у fp16 и fp8. Все
  // кодовые точки от нуля до 0x10FFFF, кроме суррогатов: записать и прочитать
  // обратно, сверив и саму точку, и длину записи.
  int64_t checked = 0;
  int64_t skipped = 0;
  for (uint32_t code = 0; code <= 0x10FFFFu; ++code) {
    if (code >= 0xD800u && code <= 0xDFFFu) {
      ++skipped;  // суррогаты в UTF-8 не записываются
      continue;
    }
    std::string encoded;
    llm::append_utf8(code, &encoded);

    const int expected_bytes = code < 0x80u
                                   ? 1
                                   : (code < 0x800u ? 2
                                                    : (code < 0x10000u ? 3 : 4));
    LLM_CHECK_MSG(static_cast<int>(encoded.size()) == expected_bytes,
                  "точка " << code << " записана в " << encoded.size()
                           << " байт вместо " << expected_bytes);

    const llm::Utf8Char back = llm::decode_utf8(encoded.data(), encoded.size());
    LLM_CHECK_MSG(back.valid, "точка " << code << " не прочиталась обратно");
    LLM_CHECK_MSG(back.code == code, "точка " << code << " вернулась как "
                                              << back.code);
    LLM_CHECK_MSG(back.bytes == expected_bytes,
                  "точка " << code << ": прочитано " << back.bytes
                           << " байт вместо " << expected_bytes);
    ++checked;
  }
  // Проверка самой проверки: суррогатов ровно 2048, остальное пройдено.
  LLM_CHECK_EQ(skipped, static_cast<int64_t>(2048));
  LLM_CHECK_EQ(checked, static_cast<int64_t>(0x110000 - 2048));
}

LLM_TEST(Pretokenize, JsonEscapeDecodingAgreesWithAppendUtf8) {
  // В json.cpp лежит своя копия append_utf8 — та же логика, записанная второй
  // раз. Пока они совпадают, но совпадение это ничем не закреплено, и
  // расхождение проявилось бы не падением, а разными байтами в именах токенов
  // при чтении чужого токенизатора.
  //
  // Точки подобраны по границам длины записи и вокруг них, плюс пара из
  // дополнительных плоскостей, которые в JSON приходят суррогатной парой.
  const uint32_t codes[] = {0x00u,   0x41u,   0x7Fu,    0x80u,
                            0x7FFu,  0x800u,  0x0416u,  0xFFFFu,
                            0x10000u, 0x1F600u, 0x10FFFFu};

  for (std::size_t i = 0; i < sizeof(codes) / sizeof(codes[0]); ++i) {
    const uint32_t code = codes[i];
    std::string expected;
    llm::append_utf8(code, &expected);

    // Собираем JSON-строку с \u-экранированием. Вне основной плоскости это
    // суррогатная пара — так устроен сам формат.
    std::string escaped = "\"";
    char buffer[16];
    if (code < 0x10000u) {
      std::snprintf(buffer, sizeof(buffer), "\\u%04X", code);
      escaped += buffer;
    } else {
      const uint32_t rest = code - 0x10000u;
      std::snprintf(buffer, sizeof(buffer), "\\u%04X",
                    0xD800u + (rest >> 10));
      escaped += buffer;
      std::snprintf(buffer, sizeof(buffer), "\\u%04X",
                    0xDC00u + (rest & 0x3FFu));
      escaped += buffer;
    }
    escaped += "\"";

    const llm::serialize::JsonDocument document =
        llm::serialize::JsonDocument::parse(escaped);
    LLM_CHECK(document.root().is_string());
    LLM_CHECK_MSG(document.root().text() == expected,
                  "точка " << code
                           << ": разбор JSON и append_utf8 дали разные байты");
  }
}

LLM_TEST(Pretokenize, UnicodeTablesMatchTheDatabaseByCount) {
  // Таблицы букв и цифр ищутся двоичным поиском, а он верен ровно при одном
  // условии: диапазоны отсортированы и не пересекаются. Условие записано в
  // комментарии и не проверялось ничем. Нарушь его — поиск начнёт молча
  // возвращать «не буква» на настоящих буквах, а это сдвинет границы кусков,
  // то есть номера всех токенов.
  //
  // Проверено внешней сверкой: перебор всех кодовых точек до 0x10FFFF против
  // unicodedata из Python версии 14.0.0 — той же базы, из которой таблицы
  // сгенерированы, — дал ноль расхождений и по буквам, и по цифрам.
  //
  // Постоянной проверкой оставлен счёт: Python в тестах нет, а число точек
  // ловит почти любую порчу таблицы — выпавший диапазон, переставленные
  // границы, съехавшую сортировку. Перебор идёт за доли секунды.
  int64_t letters = 0;
  int64_t numbers = 0;
  for (uint32_t code = 0; code <= 0x10FFFFu; ++code) {
    if (llm::is_unicode_letter(code)) {
      ++letters;
    }
    if (llm::is_unicode_number(code)) {
      ++numbers;
    }
  }
  LLM_CHECK_MSG(letters == 131756,
                "букв в таблице " << letters << " вместо 131756");
  LLM_CHECK_MSG(numbers == 1791,
                "цифр в таблице " << numbers << " вместо 1791");

  // И несколько точек по именам, чтобы при расхождении было видно не только
  // «число не то», но и где искать: по одной из каждой части таблицы.
  LLM_CHECK(llm::is_unicode_letter('A'));       // латиница, начало таблицы
  LLM_CHECK(llm::is_unicode_letter(0x0416u));   // кириллица Ж
  LLM_CHECK(llm::is_unicode_letter(0x4E2Du));   // китайский 中, середина
  LLM_CHECK(llm::is_unicode_letter(0x1E921u));  // адлам, конец таблицы
  LLM_CHECK(llm::is_unicode_number('7'));
  LLM_CHECK(llm::is_unicode_number(0x0669u));   // арабо-индийская девятка
  LLM_CHECK(llm::is_unicode_number(0x1FBF9u));  // сегментная девятка, конец

  // И отрицательные: пробел, знак, эмодзи и незанятая точка буквами не бывают.
  LLM_CHECK(!llm::is_unicode_letter(' '));
  LLM_CHECK(!llm::is_unicode_letter('!'));
  LLM_CHECK(!llm::is_unicode_letter(0x1F600u));
  LLM_CHECK(!llm::is_unicode_letter(0x0378u));  // не назначена
  LLM_CHECK(!llm::is_unicode_number('x'));
}
