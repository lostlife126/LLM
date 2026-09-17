#include "tokenizer/bpe.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>

#include "core/check.h"

namespace llm {

// Определения вне класса. В C++14 внутриклассовый инициализатор статической
// константы не является определением, и при взятии ссылки (а список
// инициализации или сравнение через const& именно это и делают) нужен этот
// код. В Release ошибка не проявляется: оптимизатор подставляет значение и
// ссылка не возникает. Ловится только отладочной сборкой.
const int32_t Bpe::kByteTokenCount;
const int32_t Bpe::kBos;
const int32_t Bpe::kEos;
const int32_t Bpe::kPad;
const int32_t Bpe::kFirstMergeToken;

namespace {

int64_t pack_pair(int32_t left, int32_t right) {
  return (static_cast<int64_t>(left) << 32) | static_cast<uint32_t>(right);
}

// Классификация байта для разбиения на куски.
//
// Байты со старшим битом считаются буквами: так последовательность UTF-8
// остаётся внутри одного куска, и слияния могут собрать из неё осмысленный
// токен. Без этого кириллическое слово разваливалось бы по границам байтов.
bool is_space_byte(unsigned char byte) {
  return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' ||
         byte == '\v' || byte == '\f';
}

bool is_letter_byte(unsigned char byte) {
  return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
         byte >= 0x80;
}

bool is_digit_byte(unsigned char byte) { return byte >= '0' && byte <= '9'; }

}  // namespace

Bpe::Bpe() {
  token_bytes_.resize(static_cast<std::size_t>(kFirstMergeToken));
  for (int32_t byte = 0; byte < kByteTokenCount; ++byte) {
    token_bytes_[static_cast<std::size_t>(byte)] =
        std::string(1, static_cast<char>(byte));
  }
  // Спецтокены не обозначают никаких байтов: они управляющие.
}

std::vector<StrView> Bpe::pre_tokenize(StrView text) {
  // Правило: кусок — это необязательный ведущий пробел плюс однородный хвост
  // (буквы, цифры, пробелы) либо один знак препинания.
  //
  // Ведущий пробел присоединяется к слову, а не отделяется, потому что для
  // языка «начало слова» — существенная позиция: токен " the" и токен "the"
  // внутри слова должны быть разными. GPT-2 добивается того же более
  // подробным регулярным выражением; здесь правило проще, но роль у него та же.
  std::vector<StrView> chunks;
  const std::size_t size = text.size();
  std::size_t index = 0;

  while (index < size) {
    const std::size_t start = index;

    // Ведущий пробел забираем только если за ним идёт не пробел: иначе это
    // отступ или перевод строки, и он образует свой кусок.
    if (text[index] == ' ' && index + 1 < size &&
        !is_space_byte(static_cast<unsigned char>(text[index + 1]))) {
      ++index;
    }

    const unsigned char head = static_cast<unsigned char>(text[index]);
    if (is_letter_byte(head)) {
      while (index < size &&
             is_letter_byte(static_cast<unsigned char>(text[index]))) {
        ++index;
      }
    } else if (is_digit_byte(head)) {
      while (index < size &&
             is_digit_byte(static_cast<unsigned char>(text[index]))) {
        ++index;
      }
    } else if (is_space_byte(head)) {
      while (index < size &&
             is_space_byte(static_cast<unsigned char>(text[index]))) {
        ++index;
      }
    } else {
      ++index;  // одиночный знак препинания
    }

    chunks.push_back(text.substr(start, index - start));
  }
  return chunks;
}

Bpe Bpe::train(StrView text, int32_t vocab_size, bool verbose) {
  LLM_CHECK_MSG(vocab_size >= kFirstMergeToken, "словарь не может быть меньше "
                                                    << kFirstMergeToken
                                                    << " (байты и спецтокены)");
  Bpe bpe;
  const int32_t target_merges = vocab_size - kFirstMergeToken;

  // Обучение идёт по словарю уникальных кусков с их частотами, а не по всему
  // корпусу. Результат тот же — пара встречается столько раз, сколько раз
  // встретился содержащий её кусок, — но работы на порядок меньше.
  std::unordered_map<std::string, int64_t> chunk_counts;
  const std::vector<StrView> chunks = pre_tokenize(text);
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    chunk_counts[chunks[i].to_string()] += 1;
  }

  struct Word {
    std::vector<int32_t> symbols;
    int64_t count;
  };
  std::vector<Word> words;
  words.reserve(chunk_counts.size());
  for (std::unordered_map<std::string, int64_t>::const_iterator it =
           chunk_counts.begin();
       it != chunk_counts.end(); ++it) {
    Word word;
    word.count = it->second;
    word.symbols.reserve(it->first.size());
    for (std::size_t i = 0; i < it->first.size(); ++i) {
      word.symbols.push_back(
          static_cast<int32_t>(static_cast<unsigned char>(it->first[i])));
    }
    words.push_back(word);
  }

  if (verbose) {
    std::printf("корпус: %zu байт, %zu кусков, %zu уникальных\n", text.size(),
                chunks.size(), chunk_counts.size());
  }

  for (int32_t merge_index = 0; merge_index < target_merges; ++merge_index) {
    std::unordered_map<int64_t, int64_t> pair_counts;
    for (std::size_t w = 0; w < words.size(); ++w) {
      const std::vector<int32_t>& symbols = words[w].symbols;
      for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
        pair_counts[pack_pair(symbols[i], symbols[i + 1])] += words[w].count;
      }
    }
    if (pair_counts.empty()) {
      if (verbose) {
        std::printf("слияния кончились на шаге %d: пар больше нет\n",
                    merge_index);
      }
      break;
    }

    // Выбор лучшей пары. При равных частотах побеждает меньший ключ: без
    // явного правила порядок обхода unordered_map сделал бы обучение
    // невоспроизводимым.
    int64_t best_key = 0;
    int64_t best_count = -1;
    for (std::unordered_map<int64_t, int64_t>::const_iterator it =
             pair_counts.begin();
         it != pair_counts.end(); ++it) {
      if (it->second > best_count ||
          (it->second == best_count && it->first < best_key)) {
        best_count = it->second;
        best_key = it->first;
      }
    }
    if (best_count < 2) {
      // Пара, встречающаяся один раз, не сокращает последовательность в
      // среднем, а словарь расходует.
      if (verbose) {
        std::printf(
            "остановка на шаге %d: самая частая пара встречается %lld раз\n",
            merge_index, static_cast<long long>(best_count));
      }
      break;
    }

    Merge merge;
    merge.left = static_cast<int32_t>(best_key >> 32);
    merge.right = static_cast<int32_t>(best_key & 0xffffffff);
    const int32_t new_token = kFirstMergeToken + merge_index;

    bpe.merges_.push_back(merge);
    bpe.merge_rank_[best_key] = merge_index;
    bpe.token_bytes_.push_back(
        bpe.token_bytes_[static_cast<std::size_t>(merge.left)] +
        bpe.token_bytes_[static_cast<std::size_t>(merge.right)]);

    // Замена пары во всех словах. Слева направо и без перекрытий: в "aaa"
    // пара (a, a) заменяется один раз, а не полтора.
    for (std::size_t w = 0; w < words.size(); ++w) {
      std::vector<int32_t>& symbols = words[w].symbols;
      if (symbols.size() < 2) {
        continue;
      }
      std::vector<int32_t> merged;
      merged.reserve(symbols.size());
      std::size_t i = 0;
      while (i < symbols.size()) {
        if (i + 1 < symbols.size() && symbols[i] == merge.left &&
            symbols[i + 1] == merge.right) {
          merged.push_back(new_token);
          i += 2;
        } else {
          merged.push_back(symbols[i]);
          ++i;
        }
      }
      symbols.swap(merged);
    }

    if (verbose &&
        (merge_index % 200 == 0 || merge_index + 1 == target_merges)) {
      std::printf("слияние %5d: %-20s частота %lld\n", merge_index,
                  bpe.token_debug_string(new_token).c_str(),
                  static_cast<long long>(best_count));
    }
  }

  return bpe;
}

void Bpe::rebuild_lookup() {
  merge_rank_.clear();
  for (std::size_t i = 0; i < merges_.size(); ++i) {
    merge_rank_[pack_pair(merges_[i].left, merges_[i].right)] =
        static_cast<int32_t>(i);
  }
}

void Bpe::apply_merges(std::vector<int32_t>* symbols) const {
  // Слияния применяются в том же порядке, в каком были выучены: на каждом
  // шаге берётся пара с наименьшим рангом. Порядок существен — если применить
  // позднее слияние раньше, получится другое разбиение, и токенизация
  // перестанет быть воспроизводимой.
  while (symbols->size() >= 2) {
    int32_t best_rank = -1;
    for (std::size_t i = 0; i + 1 < symbols->size(); ++i) {
      const std::unordered_map<int64_t, int32_t>::const_iterator found =
          merge_rank_.find(pack_pair((*symbols)[i], (*symbols)[i + 1]));
      if (found == merge_rank_.end()) {
        continue;
      }
      if (best_rank < 0 || found->second < best_rank) {
        best_rank = found->second;
      }
    }
    if (best_rank < 0) {
      return;
    }

    const int32_t left = merges_[static_cast<std::size_t>(best_rank)].left;
    const int32_t right = merges_[static_cast<std::size_t>(best_rank)].right;
    const int32_t new_token = kFirstMergeToken + best_rank;

    std::vector<int32_t> merged;
    merged.reserve(symbols->size());
    std::size_t i = 0;
    while (i < symbols->size()) {
      if (i + 1 < symbols->size() && (*symbols)[i] == left &&
          (*symbols)[i + 1] == right) {
        merged.push_back(new_token);
        i += 2;
      } else {
        merged.push_back((*symbols)[i]);
        ++i;
      }
    }
    symbols->swap(merged);
  }
}

std::vector<int32_t> Bpe::encode(StrView text) const {
  std::vector<int32_t> result;
  const std::vector<StrView> chunks = pre_tokenize(text);
  for (std::size_t c = 0; c < chunks.size(); ++c) {
    std::vector<int32_t> symbols;
    symbols.reserve(chunks[c].size());
    for (std::size_t i = 0; i < chunks[c].size(); ++i) {
      symbols.push_back(
          static_cast<int32_t>(static_cast<unsigned char>(chunks[c][i])));
    }
    apply_merges(&symbols);
    result.insert(result.end(), symbols.begin(), symbols.end());
  }
  return result;
}

std::string Bpe::decode(const std::vector<int32_t>& ids) const {
  std::string result;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    result += token_bytes(ids[i]);
  }
  return result;
}

const std::string& Bpe::token_bytes(int32_t id) const {
  LLM_CHECK_MSG(id >= 0 && id < vocab_size(),
                "токен " << id << " вне словаря размера " << vocab_size());
  return token_bytes_[static_cast<std::size_t>(id)];
}

std::string Bpe::token_debug_string(int32_t id) const {
  if (id == kBos) {
    return "<bos>";
  }
  if (id == kEos) {
    return "<eos>";
  }
  if (id == kPad) {
    return "<pad>";
  }
  const std::string& bytes = token_bytes(id);
  std::string result = "'";
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    const unsigned char byte = static_cast<unsigned char>(bytes[i]);
    if (byte == '\n') {
      result += "\\n";
    } else if (byte == '\t') {
      result += "\\t";
    } else if (byte == '\r') {
      result += "\\r";
    } else if (byte < 0x20 || byte == 0x7f) {
      char buffer[8];
      std::snprintf(buffer, sizeof(buffer), "\\x%02x", byte);
      result += buffer;
    } else {
      result += static_cast<char>(byte);
    }
  }
  result += "'";
  return result;
}

void Bpe::save(const std::string& path) const {
  // Формат текстовый: словарь удобно открыть глазами и увидеть, чему
  // токенизатор научился. Базовые байты и спецтокены не хранятся — они
  // подразумеваются.
  std::ofstream file(path.c_str());
  LLM_CHECK_MSG(file.good(), "не удалось открыть для записи: " << path);
  file << "llmbpe 1\n";
  file << "merges " << merges_.size() << "\n";
  for (std::size_t i = 0; i < merges_.size(); ++i) {
    file << merges_[i].left << " " << merges_[i].right << "\n";
  }
  LLM_CHECK_MSG(file.good(), "ошибка записи в " << path);
}

Bpe Bpe::load(const std::string& path) {
  std::ifstream file(path.c_str());
  LLM_CHECK_MSG(file.good(), "не удалось открыть: " << path);

  std::string magic;
  int version = 0;
  file >> magic >> version;
  LLM_CHECK_MSG(magic == "llmbpe", "чужой формат словаря в " << path);
  LLM_CHECK_MSG(version == 1, "версия формата " << version << " не поддержана");

  std::string keyword;
  std::size_t count = 0;
  file >> keyword >> count;
  LLM_CHECK_MSG(keyword == "merges", "ожидалось 'merges' в " << path);

  Bpe bpe;
  bpe.merges_.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    Merge merge;
    file >> merge.left >> merge.right;
    // Именно fail(), а не good(). Числовое чтение, дошедшее ровно до конца
    // файла, выставляет eofbit, хотя число прочитано целиком; good() при
    // этом ложно. Словарь без перевода строки в конце — а такой легко
    // получить, поправив файл руками, — отвергался бы как оборванный.
    LLM_CHECK_MSG(!file.fail(), "словарь оборвался на слиянии " << i);

    const int32_t limit = kFirstMergeToken + static_cast<int32_t>(i);
    LLM_CHECK_MSG(merge.left >= 0 && merge.left < limit && merge.right >= 0 &&
                      merge.right < limit,
                  "слияние " << i << " ссылается на токен, которого ещё нет");

    bpe.merges_.push_back(merge);
    bpe.token_bytes_.push_back(
        bpe.token_bytes_[static_cast<std::size_t>(merge.left)] +
        bpe.token_bytes_[static_cast<std::size_t>(merge.right)]);
  }
  bpe.rebuild_lookup();
  return bpe;
}

}  // namespace llm
