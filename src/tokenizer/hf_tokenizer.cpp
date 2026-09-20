#include "tokenizer/hf_tokenizer.h"

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>

#include "core/check.h"
#include "serialize/json.h"
#include "tokenizer/utf8.h"

namespace llm {
namespace {

using serialize::JsonDocument;
using serialize::JsonValue;

// Разделитель в ключе пары. Взят символ, которого в байтовом алфавите нет:
// алфавит состоит из печатных знаков и точек выше U+0100, а пробел в нём
// отображён в «Ġ». Значит пробел как разделитель однозначен.
const char kPairSeparator = ' ';

std::string pair_key(const std::string& left, const std::string& right) {
  return left + kPairSeparator + right;
}

bool field_is_empty(const JsonValue& parent, StrView name) {
  const JsonValue value = parent.find(name);
  if (!value.defined() || value.is_null()) {
    return true;
  }
  return value.is_string() && value.text().empty();
}

}  // namespace

void HfTokenizer::build_byte_alphabet() {
  // Правило из GPT-2: печатные знаки отображаются сами в себя, остальные —
  // в кодовые точки начиная с U+0100 по порядку. Печатными считаются
  // '!'..'~', 0xA1..0xAC и 0xAE..0xFF.
  bool printable[256];
  for (int i = 0; i < 256; ++i) {
    printable[i] = false;
  }
  for (int i = '!'; i <= '~'; ++i) {
    printable[i] = true;
  }
  for (int i = 0xA1; i <= 0xAC; ++i) {
    printable[i] = true;
  }
  for (int i = 0xAE; i <= 0xFF; ++i) {
    printable[i] = true;
  }

  uint32_t next = 256;
  for (int i = 0; i < 256; ++i) {
    byte_to_code_[i] = printable[i] ? static_cast<uint32_t>(i) : next++;
    code_to_byte_[byte_to_code_[i]] = i;
  }
}

HfTokenizer HfTokenizer::parse(const std::string& text,
                               const std::string& origin) {
  const JsonDocument document = JsonDocument::parse(text);
  const JsonValue root = document.root();
  LLM_CHECK_MSG(root.is_object(), origin << ": tokenizer.json не объект");

  HfTokenizer out;
  out.build_byte_alphabet();

  // --- модель ---

  const JsonValue model = root.field("model");
  LLM_CHECK_MSG(model.field("type").text() == "BPE",
                origin << ": model.type=\"" << model.field("type").text()
                       << "\", а мы умеем только BPE");

  // Всё, что меняет разбор и чего у нас нет, — отказ. Молча пропустить
  // byte_fallback или суффикс конца слова значит разбить текст иначе, чем
  // разбивал автор модели.
  LLM_CHECK_MSG(field_is_empty(model, "unk_token"),
                origin << ": задан unk_token; при байтовом алфавите он не "
                          "нужен, и его наличие означает другой разбор");
  LLM_CHECK_MSG(field_is_empty(model, "continuing_subword_prefix"),
                origin << ": задан continuing_subword_prefix");
  LLM_CHECK_MSG(field_is_empty(model, "end_of_word_suffix"),
                origin << ": задан end_of_word_suffix");
  LLM_CHECK_MSG(!model.has("dropout") || model.field("dropout").is_null(),
                origin << ": задан dropout — он вносит случайность в разбор");
  LLM_CHECK_MSG(
      !model.has("byte_fallback") || !model.field("byte_fallback").boolean(),
      origin << ": задан byte_fallback");

  const JsonValue vocab = model.field("vocab");
  LLM_CHECK_MSG(vocab.is_object(), origin << ": model.vocab не объект");

  int32_t highest = -1;
  std::vector<std::pair<int32_t, std::string>> entries;
  entries.reserve(vocab.size());
  for (std::size_t i = 0; i < vocab.size(); ++i) {
    const int32_t id = static_cast<int32_t>(vocab.value_at(i).integer());
    LLM_CHECK_MSG(id >= 0, origin << ": отрицательный номер токена " << id);
    entries.push_back(std::make_pair(id, vocab.key_at(i)));
    highest = id > highest ? id : highest;
  }
  out.id_to_token_.assign(static_cast<std::size_t>(highest + 1), std::string());
  std::vector<bool> filled(static_cast<std::size_t>(highest + 1), false);
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const std::size_t id = static_cast<std::size_t>(entries[i].first);
    LLM_CHECK_MSG(!filled[id],
                  origin << ": номер " << entries[i].first << " занят дважды");
    out.id_to_token_[id] = entries[i].second;
    filled[id] = true;
    out.token_to_id_[entries[i].second] = entries[i].first;
  }

  // --- слияния ---

  const JsonValue merges = model.field("merges");
  LLM_CHECK_MSG(merges.is_array(), origin << ": model.merges не массив");
  for (std::size_t i = 0; i < merges.size(); ++i) {
    const JsonValue item = merges.at(i);
    std::string left;
    std::string right;
    if (item.is_string()) {
      // Старая запись: «левое правое» одной строкой.
      const std::string line = item.text();
      const std::size_t space = line.find(' ');
      LLM_CHECK_MSG(space != std::string::npos,
                    origin << ": слияние без пробела: " << line);
      left = line.substr(0, space);
      right = line.substr(space + 1);
    } else {
      // Новая запись: пара строк.
      LLM_CHECK_MSG(item.is_array() && item.size() == 2,
                    origin << ": слияние не пара");
      left = item.at(0).text();
      right = item.at(1).text();
    }
    // Ранг — это номер в списке, и дубликаты значения не меняют: побеждает
    // первое вхождение, как в эталонной реализации.
    const std::string key = pair_key(left, right);
    if (out.merge_rank_.find(key) == out.merge_rank_.end()) {
      out.merge_rank_[key] = static_cast<int32_t>(i);
    }
  }

  // --- добавленные токены ---

  if (root.has("added_tokens")) {
    const JsonValue added = root.field("added_tokens");
    LLM_CHECK_MSG(added.is_array(), origin << ": added_tokens не массив");
    for (std::size_t i = 0; i < added.size(); ++i) {
      const JsonValue item = added.at(i);
      HfAddedToken token;
      token.id = static_cast<int32_t>(item.field("id").integer());
      LLM_CHECK_MSG(token.id >= 0, origin << ": добавленный токен с номером "
                                          << token.id);
      token.content = item.field("content").text();
      token.special = item.has("special") && item.field("special").is_bool() &&
                      item.field("special").boolean();
      out.added_tokens_.push_back(token);

      // Добавленные токены могут отсутствовать в словаре модели — тогда их
      // надо знать отдельно, иначе decode вернёт пустоту.
      if (static_cast<std::size_t>(token.id) >= out.id_to_token_.size()) {
        out.id_to_token_.resize(static_cast<std::size_t>(token.id) + 1);
      }
      if (out.id_to_token_[static_cast<std::size_t>(token.id)].empty()) {
        out.id_to_token_[static_cast<std::size_t>(token.id)] = token.content;
      }
    }
    // Длинные вперёд: при пересекающихся записях выигрывать должна длинная,
    // иначе «<|endoftext|>» распадётся на «<|end» и остаток.
    //
    // Устойчивая сортировка: на разбор порядок среди записей одной длины не
    // влияет (две разные записи одной длины не могут совпасть в одном месте),
    // но added_tokens() отдаётся наружу, и пусть он не зависит от того, чья
    // это стандартная библиотека.
    std::stable_sort(out.added_tokens_.begin(), out.added_tokens_.end(),
              [](const HfAddedToken& a, const HfAddedToken& b) {
                return a.content.size() > b.content.size();
              });
  }

  // --- предтокенизатор ---

  const JsonValue pre = root.find("pre_tokenizer");
  LLM_CHECK_MSG(pre.defined() && !pre.is_null(),
                origin << ": нет pre_tokenizer, а от него зависит разбиение");

  const std::string kind = pre.field("type").text();
  bool recognized = false;
  if (kind == "ByteLevel") {
    // Классический GPT-2: сам ByteLevel применяет своё выражение.
    const bool uses_regex =
        !pre.has("use_regex") || pre.field("use_regex").boolean();
    LLM_CHECK_MSG(uses_regex,
                  origin << ": ByteLevel без use_regex и без Split — текст "
                            "не разбивается вовсе, и такого мы не видели");
    out.rule_ = PretokenizeRule::kGpt2;
    recognized = true;
    out.add_prefix_space_ =
        pre.has("add_prefix_space") && pre.field("add_prefix_space").boolean();
  } else if (kind == "Sequence") {
    const JsonValue list = pre.field("pretokenizers");
    LLM_CHECK_MSG(list.is_array(), origin << ": pretokenizers не массив");
    for (std::size_t i = 0; i < list.size(); ++i) {
      const JsonValue item = list.at(i);
      const std::string item_kind = item.field("type").text();
      if (item_kind == "Split") {
        const JsonValue pattern = item.field("pattern");
        LLM_CHECK_MSG(pattern.has("Regex"),
                      origin << ": Split не по регулярному выражению");
        const std::string expression = pattern.field("Regex").text();
        LLM_CHECK_MSG(
            recognize_pretokenizer(expression, &out.rule_),
            origin << ": незнакомый образец разбиения.\n  " << expression
                   << "\nРазбирать произвольные выражения мы не умеем: "
                      "написать половину движка значит завести собственный "
                      "набор расхождений с эталоном, а неверное разбиение "
                      "ничем себя не выдаёт. Известны два образца:\n  "
                   << pretokenizer_pattern(PretokenizeRule::kGpt2) << "\n  "
                   << pretokenizer_pattern(PretokenizeRule::kGpt4));
        recognized = true;
      } else if (item_kind == "ByteLevel") {
        out.add_prefix_space_ = item.has("add_prefix_space") &&
                                item.field("add_prefix_space").boolean();
      } else {
        LLM_CHECK_MSG(false,
                      origin << ": в Sequence неизвестный шаг " << item_kind);
      }
    }
  } else {
    LLM_CHECK_MSG(false,
                  origin << ": pre_tokenizer типа " << kind << " не знаком");
  }
  LLM_CHECK_MSG(recognized, origin << ": не удалось определить разбиение");

  // --- декодер ---

  if (root.has("decoder") && !root.field("decoder").is_null()) {
    const JsonValue decoder = root.field("decoder");
    const std::string decoder_kind = decoder.field("type").text();
    LLM_CHECK_MSG(decoder_kind == "ByteLevel" || decoder_kind == "Sequence",
                  origin << ": декодер типа " << decoder_kind
                         << " не знаком; при байтовом алфавите ожидается "
                            "ByteLevel");
  }

  return out;
}

HfTokenizer HfTokenizer::load(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::binary);
  LLM_CHECK_MSG(file.good(), "не удалось открыть " << path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return parse(buffer.str(), path);
}

int32_t HfTokenizer::token_id(const std::string& content) const {
  const std::unordered_map<std::string, int32_t>::const_iterator found =
      token_to_id_.find(content);
  return found == token_to_id_.end() ? -1 : found->second;
}

const std::string& HfTokenizer::token_text(int32_t id) const {
  LLM_CHECK_MSG(
      id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size(),
      "номер токена " << id << " вне словаря размера " << id_to_token_.size());
  return id_to_token_[static_cast<std::size_t>(id)];
}

void HfTokenizer::apply_merges(std::vector<std::string>* symbols) const {
  // Всегда применяется пара с наименьшим рангом из имеющихся, а не первая
  // слева. Порядок здесь не украшение: он и определяет, во что сложится
  // слово.
  while (symbols->size() > 1) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    std::size_t best_at = 0;
    bool found = false;

    for (std::size_t i = 0; i + 1 < symbols->size(); ++i) {
      const std::unordered_map<std::string, int32_t>::const_iterator rank =
          merge_rank_.find(pair_key((*symbols)[i], (*symbols)[i + 1]));
      if (rank != merge_rank_.end() && rank->second < best_rank) {
        best_rank = rank->second;
        best_at = i;
        found = true;
      }
    }
    if (!found) {
      return;
    }

    // Выбранная пара заменяется ВЕЗДЕ за один проход, слева направо и без
    // перекрытий, а не в одном месте с новым поиском минимума.
    //
    // Результат от этого не меняется, и это не предположение. Слияние ранга r
    // порождает токен, который в словаре появился на шаге r, поэтому входить
    // он может только в слияния более поздние: минимум ранга по ходу работы
    // не убывает. Новых вхождений самой пары замена тоже не создаёт — новые
    // соседства идут по склейке, а она ни одной из половин не равна. Значит
    // проход по всем вхождениям и есть то, что дала бы замена по одному.
    //
    // Меняется цена. По одному вхождению за поиск получался квадрат от длины
    // куска, и на куске из 8000 пробелов — а такой даёт любая строка из одних
    // пробелов, правило \s+ её не режет — разбор занимал 1.45 с при шести
    // токенах на выходе; вдвое длиннее кусок — вчетверо дольше разбор.
    // С проходом целиком тот же кусок разбирается за 0.002 с, и время растёт
    // уже линейно: 256000 пробелов — 0.051 с вместо примерно получаса.
    const std::string left = (*symbols)[best_at];
    const std::string right = (*symbols)[best_at + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols->size());
    std::size_t i = 0;
    while (i < symbols->size()) {
      if (i + 1 < symbols->size() && (*symbols)[i] == left &&
          (*symbols)[i + 1] == right) {
        merged.push_back(left + right);
        i += 2;
      } else {
        merged.push_back((*symbols)[i]);
        ++i;
      }
    }
    symbols->swap(merged);
  }
}

std::vector<int32_t> HfTokenizer::encode(StrView text) const {
  std::vector<int32_t> out;
  if (text.empty()) {
    return out;
  }

  // Особые токены вырезаются до всего остального: в слияниях они не
  // участвуют. Поиск идёт по самой длинной записи вперёд, поэтому вложенные
  // записи не дробят друг друга.
  const std::string source = text.to_string();

  std::vector<std::pair<std::size_t, std::size_t>> specials;  // начало, длина
  std::vector<int32_t> special_ids;
  for (std::size_t at = 0; at < source.size();) {
    bool matched = false;
    for (std::size_t i = 0; i < added_tokens_.size(); ++i) {
      const std::string& content = added_tokens_[i].content;
      if (!content.empty() &&
          source.compare(at, content.size(), content) == 0) {
        specials.push_back(std::make_pair(at, content.size()));
        special_ids.push_back(added_tokens_[i].id);
        at += content.size();
        matched = true;
        break;
      }
    }
    if (!matched) {
      ++at;
    }
  }

  std::size_t position = 0;
  for (std::size_t i = 0; i <= specials.size(); ++i) {
    const std::size_t stop =
        i < specials.size() ? specials[i].first : source.size();
    if (stop > position) {
      // Ведущий пробел добавляется каждому обычному промежутку, а не тексту
      // целиком. Так устроен эталон: особые токены вырезаются первыми, и
      // предтокенизатор запускается на каждом оставшемся куске отдельно —
      // значит и add_prefix_space срабатывает на каждом. Разница видна, как
      // только особый токен стоит не в начале: у «hello<|endoftext|>hello»
      // второе «hello» тоже обязано получить пробел.
      std::string piece(source, position, stop - position);
      if (add_prefix_space_ && piece[0] != ' ') {
        piece.insert(piece.begin(), ' ');
      }
      const StrView segment(piece.data(), piece.size());
      const std::vector<StrView> chunks = pretokenize(segment, rule_);
      for (std::size_t c = 0; c < chunks.size(); ++c) {
        // Каждый байт куска — отдельный символ байтового алфавита.
        std::vector<std::string> symbols;
        symbols.reserve(chunks[c].size());
        for (std::size_t b = 0; b < chunks[c].size(); ++b) {
          const unsigned char byte = static_cast<unsigned char>(chunks[c][b]);
          std::string symbol;
          append_utf8(byte_to_code_[byte], &symbol);
          symbols.push_back(symbol);
        }
        apply_merges(&symbols);

        for (std::size_t s = 0; s < symbols.size(); ++s) {
          const int32_t id = token_id(symbols[s]);
          // При байтовом алфавите каждый одиночный байт обязан быть в
          // словаре, поэтому недостижимо. Но если словарь неполный, лучше
          // сказать об этом, чем подставить произвольный номер.
          LLM_CHECK_MSG(id >= 0, "в словаре нет токена '" << symbols[s] << "'");
          out.push_back(id);
        }
      }
    }
    if (i < specials.size()) {
      out.push_back(special_ids[i]);
      position = specials[i].first + specials[i].second;
    }
  }
  return out;
}

std::string HfTokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string joined;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    joined += token_text(ids[i]);
  }

  // Обратное отображение алфавита: каждая кодовая точка — один байт.
  std::string out;
  std::size_t at = 0;
  while (at < joined.size()) {
    const Utf8Char ch = decode_utf8(joined.data() + at, joined.size() - at);
    const std::unordered_map<uint32_t, int>::const_iterator found =
        code_to_byte_.find(ch.code);
    if (found != code_to_byte_.end()) {
      out.push_back(static_cast<char>(found->second));
    } else {
      // Символ не из алфавита — так выглядят особые токены, записанные
      // обычным текстом. Отдаём как есть.
      out.append(joined, at, static_cast<std::size_t>(ch.bytes));
    }
    at += static_cast<std::size_t>(ch.bytes);
  }
  return out;
}

}  // namespace llm
