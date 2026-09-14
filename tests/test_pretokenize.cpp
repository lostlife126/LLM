// Предтокенизация: разбиение перед слияниями BPE.
//
// Проверять это надо въедливо. Ошибка здесь не проявляется ни падением, ни
// бессмыслицей: получатся другие номера токенов, модель увидит не тот текст и
// ответит связно и мимо. А границы кусков задаются не «здравым смыслом», а
// довольно причудливым регулярным выражением с откатами, и именно его
// причуды надо воспроизвести.

#include <string>
#include <vector>

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
