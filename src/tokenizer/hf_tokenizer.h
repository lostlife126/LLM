// Токенизатор чужой модели: чтение tokenizer.json.
//
// Свой BPE у нас есть, и устроен он так же. Отдельный класс нужен потому, что
// совпадать обязано не устройство, а результат: модель обучалась на конкретных
// номерах токенов, и подать ей текст, разбитый чуть иначе, — всё равно что
// подать другой текст.
//
// Три части, каждая со своей ловушкой.
//
// Байтовый алфавит. Байты не могут быть ключами в JSON, поэтому каждый из 256
// отображается в печатный символ: пробел становится «Ġ», перевод строки —
// «Ċ». Отображение не произвольное, а то самое, из GPT-2, и считается оно
// здесь по тому же правилу, а не таблицей: правило короче таблицы и не
// содержит опечаток.
//
// Слияния. Пар в файле десятки тысяч, и важен их порядок: ранг пары — это её
// номер в списке. Применяются они не по одному разу слева направо, а всегда
// самая ранняя из возможных, пока возможные не кончатся.
//
// Особые токены. <|endoftext|> и подобные не участвуют в слияниях вовсе: текст
// сначала режется по ним, и только промежутки идут через обычный разбор.
// Иначе «<|endoftext|>» в тексте превратилось бы в десяток обычных токенов.

#ifndef LLM_TOKENIZER_HF_TOKENIZER_H_
#define LLM_TOKENIZER_HF_TOKENIZER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/str_view.h"
#include "tokenizer/pretokenize.h"

namespace llm {

struct HfAddedToken {
  int32_t id = -1;
  std::string content;
  bool special = false;
};

class HfTokenizer {
 public:
  static HfTokenizer load(const std::string& path);
  static HfTokenizer parse(const std::string& text, const std::string& origin);

  // Текст -> номера токенов. Особые токены сами не добавляются: шаблон
  // начала последовательности у каждой модели свой, и подставлять его молча
  // значит менять вход модели без спроса.
  std::vector<int32_t> encode(StrView text) const;

  // Номера -> текст. Обратное преобразование точное: байт-левел BPE ничего не
  // теряет, поэтому decode(encode(x)) == x для любого x.
  std::string decode(const std::vector<int32_t>& ids) const;

  int64_t vocab_size() const {
    return static_cast<int64_t>(id_to_token_.size());
  }

  // Номер токена по его записи в словаре (в байтовом алфавите). -1, если нет.
  int32_t token_id(const std::string& content) const;

  // Запись токена — для отладки и для печати по одному токену за раз.
  const std::string& token_text(int32_t id) const;

  PretokenizeRule rule() const { return rule_; }
  bool add_prefix_space() const { return add_prefix_space_; }
  const std::vector<HfAddedToken>& added_tokens() const {
    return added_tokens_;
  }

 private:
  void build_byte_alphabet();

  // Применяет слияния к одному куску, уже переведённому в байтовый алфавит.
  void apply_merges(std::vector<std::string>* symbols) const;

  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  // Ранг пары: чем меньше, тем раньше применяется.
  std::unordered_map<std::string, int32_t> merge_rank_;

  std::vector<HfAddedToken> added_tokens_;

  // byte -> кодовая точка и обратно.
  uint32_t byte_to_code_[256];
  std::unordered_map<uint32_t, int> code_to_byte_;

  PretokenizeRule rule_ = PretokenizeRule::kGpt2;
  bool add_prefix_space_ = false;
};

}  // namespace llm

#endif  // LLM_TOKENIZER_HF_TOKENIZER_H_
