#include "tokenizer/pretokenize.h"

#include <string>

#include "core/check.h"
#include "tokenizer/unicode_tables.h"
#include "tokenizer/utf8.h"

namespace llm {
namespace {

// Записи образцов дословно, как они стоят в tokenizer.json. Сравнение
// посимвольное: если модель принесёт похожее, но другое выражение, лучше
// отказать, чем разбить по-своему.
const char kGpt2Pattern[] =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!"
    "\\S)|\\s+";

const char kGpt4Pattern[] =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

bool is_space_code(uint32_t code) {
  // \s в этих выражениях — пробельные символы Unicode. Для наших задач
  // достаточно набора ASCII плюс неразрывный пробел и пробелы из блока
  // пунктуации: именно они встречаются в текстах.
  return code == ' ' || code == '\t' || code == '\n' || code == '\r' ||
         code == '\v' || code == '\f' || code == 0x85u || code == 0xA0u ||
         (code >= 0x2000u && code <= 0x200Au) || code == 0x2028u ||
         code == 0x2029u || code == 0x202Fu || code == 0x205Fu ||
         code == 0x3000u;
}

// Разряды кодовой точки спрашиваются только у годной последовательности.
//
// Иначе разряд куска определял бы сам испорченный байт, а не то, что он
// испорчен. Пример настоящий: «аб» в UTF-8 — D0 B0 D0 B1, и если текст
// обрезан посередине буквы, остаётся хвостовой байт D0. Кодовая точка U+00D0
// — это буква Ð, так что мусор молча приклеивался бы к слову, и граница куска
// зависела бы от того, какой именно байт уцелел. Байты 0x85 и 0xA0 тем же
// путём попадали бы в пробелы, а 0xB2 — в цифры.
//
// Годной последовательности проверка ничего не меняет: у неё valid всегда
// истинно. Негодная уходит в разряд прочих символов — туда, где ей и место:
// байт-левел BPE всё равно кодирует байты, а не символы.
bool is_space(const Utf8Char& ch) { return ch.valid && is_space_code(ch.code); }

bool is_letter(const Utf8Char& ch) {
  return ch.valid && is_unicode_letter(ch.code);
}

bool is_number(const Utf8Char& ch) {
  return ch.valid && is_unicode_number(ch.code);
}

bool is_newline(const Utf8Char& ch) {
  return ch.valid && (ch.code == '\n' || ch.code == '\r');
}

bool is_blank(const Utf8Char& ch) { return ch.valid && ch.code == ' '; }

// Длина совпадения одного из сокращений в позиции at. Ноль, если нет.
std::size_t match_contraction(StrView text, std::size_t at, bool ignore_case) {
  if (at >= text.size() || text[at] != '\'') {
    return 0;
  }
  static const char* const kTails[] = {"re", "ve", "ll", "s", "t", "m", "d"};
  const std::size_t count = sizeof(kTails) / sizeof(kTails[0]);
  for (std::size_t i = 0; i < count; ++i) {
    const std::string tail = kTails[i];
    if (at + 1 + tail.size() > text.size()) {
      continue;
    }
    bool same = true;
    for (std::size_t j = 0; j < tail.size(); ++j) {
      char actual = text[at + 1 + j];
      if (ignore_case && actual >= 'A' && actual <= 'Z') {
        actual = static_cast<char>(actual - 'A' + 'a');
      }
      if (actual != tail[j]) {
        same = false;
        break;
      }
    }
    if (same) {
      return 1 + tail.size();
    }
  }
  return 0;
}

// Конец пробельной цепочки, начинающейся в at.
std::size_t whitespace_end(StrView text, std::size_t at) {
  std::size_t position = at;
  while (position < text.size()) {
    const Utf8Char ch =
        decode_utf8(text.data() + position, text.size() - position);
    if (!is_space(ch)) {
      break;
    }
    position += static_cast<std::size_t>(ch.bytes);
  }
  return position;
}

// Конец цепочки символов, удовлетворяющих условию.
template <typename Predicate>
std::size_t run_end(StrView text, std::size_t at, Predicate allowed) {
  std::size_t position = at;
  while (position < text.size()) {
    const Utf8Char ch =
        decode_utf8(text.data() + position, text.size() - position);
    if (!allowed(ch)) {
      break;
    }
    position += static_cast<std::size_t>(ch.bytes);
  }
  return position;
}

bool is_other(const Utf8Char& ch) {
  return !is_space(ch) && !is_letter(ch) && !is_number(ch);
}

// \s+(?!\S): пробельная цепочка, за которой не идёт непробельный символ.
//
// Что здесь происходит на самом деле. Жадное \s+ забирает всю цепочку, затем
// проверка «дальше не непробельный» не проходит, потому что дальше как раз
// непробельный. Движок откатывается на один символ — и теперь дальше стоит
// пробел, проверка выполняется. Итог: цепочка без последнего символа.
// В конце текста откатываться не нужно, и берётся она целиком.
//
// Отсюда и знаменитое поведение «пробел прилипает к следующему слову»: из
// двух пробелов подряд первый уходит отдельным куском, а второй достаётся
// слову.
std::size_t match_trailing_whitespace(StrView text, std::size_t at) {
  const std::size_t end = whitespace_end(text, at);
  if (end == at) {
    return 0;
  }
  if (end == text.size()) {
    return end - at;
  }
  // Откат на один символ: нужно знать, где начинается последний.
  std::size_t last = at;
  std::size_t position = at;
  while (position < end) {
    last = position;
    const Utf8Char ch = decode_utf8(text.data() + position, end - position);
    position += static_cast<std::size_t>(ch.bytes);
  }
  return last - at;
}

std::size_t match_gpt2(StrView text, std::size_t at) {
  const std::size_t contraction = match_contraction(text, at, false);
  if (contraction != 0) {
    return contraction;
  }

  // Необязательный пробел, затем буквы; затем то же с цифрами; затем с
  // прочими символами. Пробел засчитывается только если за ним есть хоть
  // один символ нужного разряда.
  const Utf8Char first = decode_utf8(text.data() + at, text.size() - at);
  const std::size_t after_space =
      is_blank(first) ? at + static_cast<std::size_t>(first.bytes) : at;

  const std::size_t letters = run_end(text, after_space, is_letter);
  if (letters > after_space) {
    return letters - at;
  }
  const std::size_t numbers = run_end(text, after_space, is_number);
  if (numbers > after_space) {
    return numbers - at;
  }
  const std::size_t others = run_end(text, after_space, is_other);
  if (others > after_space) {
    return others - at;
  }

  const std::size_t trailing = match_trailing_whitespace(text, at);
  if (trailing != 0) {
    return trailing;
  }
  const std::size_t spaces = whitespace_end(text, at);
  if (spaces > at) {
    return spaces - at;
  }
  return 0;
}

std::size_t match_gpt4(StrView text, std::size_t at) {
  const std::size_t contraction = match_contraction(text, at, true);
  if (contraction != 0) {
    return contraction;
  }

  const Utf8Char first = decode_utf8(text.data() + at, text.size() - at);

  // [^\r\n\p{L}\p{N}]?\p{L}+ — необязательный один символ, не перевод строки
  // и не буква с цифрой, затем буквы. Отличие от GPT-2 существенное: сюда
  // попадает не только пробел, но и, например, открывающая скобка перед
  // словом.
  {
    const bool skippable =
        !is_newline(first) && !is_letter(first) && !is_number(first);
    if (skippable) {
      const std::size_t after = at + static_cast<std::size_t>(first.bytes);
      const std::size_t letters = run_end(text, after, is_letter);
      if (letters > after) {
        return letters - at;
      }
    }
    const std::size_t letters = run_end(text, at, is_letter);
    if (letters > at) {
      return letters - at;
    }
  }

  // \p{N}{1,3} — не больше трёх цифр подряд.
  {
    std::size_t position = at;
    int taken = 0;
    while (position < text.size() && taken < 3) {
      const Utf8Char ch =
          decode_utf8(text.data() + position, text.size() - position);
      if (!is_number(ch)) {
        break;
      }
      position += static_cast<std::size_t>(ch.bytes);
      ++taken;
    }
    if (taken > 0) {
      return position - at;
    }
  }

  // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
  {
    const std::size_t after_space =
        is_blank(first) ? at + static_cast<std::size_t>(first.bytes) : at;
    const std::size_t others = run_end(text, after_space, is_other);
    if (others > after_space) {
      const std::size_t newlines = run_end(text, others, is_newline);
      return newlines - at;
    }
  }

  // `\s*[\r\n]+` — пробельная цепочка, оканчивающаяся переводом строки.
  // Жадность с откатом даёт ровно «до последнего перевода строки в цепочке
  // включительно».
  {
    const std::size_t end = whitespace_end(text, at);
    std::size_t last_newline = std::string::npos;
    std::size_t position = at;
    while (position < end) {
      const Utf8Char ch = decode_utf8(text.data() + position, end - position);
      if (is_newline(ch)) {
        last_newline = position + static_cast<std::size_t>(ch.bytes);
      }
      position += static_cast<std::size_t>(ch.bytes);
    }
    if (last_newline != std::string::npos) {
      return last_newline - at;
    }
  }

  const std::size_t trailing = match_trailing_whitespace(text, at);
  if (trailing != 0) {
    return trailing;
  }
  const std::size_t spaces = whitespace_end(text, at);
  if (spaces > at) {
    return spaces - at;
  }
  return 0;
}

}  // namespace

std::vector<StrView> pretokenize(StrView text, PretokenizeRule rule) {
  std::vector<StrView> out;
  std::size_t at = 0;
  while (at < text.size()) {
    std::size_t length = rule == PretokenizeRule::kGpt2 ? match_gpt2(text, at)
                                                        : match_gpt4(text, at);
    if (length == 0) {
      // Ни одно правило не подошло. В обоих образцах последняя ветвь — \s+,
      // и не подойти она может только на непробельном символе, который при
      // этом не буква, не цифра и не «прочий», чего быть не может. Но если
      // разбор UTF-8 вернул что-то неожиданное, двигаться всё равно надо:
      // иначе цикл станет вечным.
      const Utf8Char ch = decode_utf8(text.data() + at, text.size() - at);
      length = static_cast<std::size_t>(ch.bytes);
    }
    out.push_back(text.substr(at, length));
    at += length;
  }
  return out;
}

bool recognize_pretokenizer(const std::string& pattern, PretokenizeRule* rule) {
  LLM_CHECK(rule != nullptr);
  if (pattern == kGpt2Pattern) {
    *rule = PretokenizeRule::kGpt2;
    return true;
  }
  if (pattern == kGpt4Pattern) {
    *rule = PretokenizeRule::kGpt4;
    return true;
  }
  return false;
}

const char* pretokenizer_pattern(PretokenizeRule rule) {
  return rule == PretokenizeRule::kGpt2 ? kGpt2Pattern : kGpt4Pattern;
}

}  // namespace llm
