// Byte-level BPE — токенизатор.
//
// Base pair encoding начинает с алфавита из 256 байтов и затем многократно
// заменяет самую частую пару соседних токенов на новый токен. Словарь растёт,
// последовательность укорачивается.
//
// Почему именно байты, а не символы. Байтовый алфавит замкнут: любая строка
// представима, неизвестных символов не бывает в принципе, и токен <unk> не
// нужен вообще. Ценой того, что редкий символ занимает несколько токенов —
// кириллическая буква в UTF-8 это два байта, и пока модель не выучит их
// слияние, она видит два токена вместо одного.
//
// Перед слияниями текст режется на куски (pre_tokenize), и слияния никогда не
// переходят границу куска. Это не оптимизация: без такого разбиения BPE
// склеил бы частые сочетания слов в один токен, и «of the» стало бы
// неделимым. Заодно обучение ускоряется на порядок, потому что считать пары
// можно по словарю уникальных кусков, а не по всему корпусу.

#ifndef LLM_TOKENIZER_BPE_H_
#define LLM_TOKENIZER_BPE_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/str_view.h"

namespace llm {

class Bpe {
 public:
  // Первые 256 идентификаторов — байты. Дальше спецтокены с фиксированными
  // номерами: если бы они стояли после слияний, их номера зависели бы от
  // размера словаря, и сохранённые словари разных размеров стали бы
  // несовместимы.
  static const int32_t kByteTokenCount = 256;
  static const int32_t kBos = 256;
  static const int32_t kEos = 257;
  static const int32_t kPad = 258;
  static const int32_t kFirstMergeToken = 259;

  Bpe();

  // vocab_size считает всё: байты, спецтокены и слияния. Минимум —
  // kFirstMergeToken, то есть словарь вообще без слияний.
  static Bpe train(StrView text, int32_t vocab_size, bool verbose);

  int32_t vocab_size() const {
    return kFirstMergeToken + static_cast<int32_t>(merges_.size());
  }
  int32_t merge_count() const { return static_cast<int32_t>(merges_.size()); }

  std::vector<int32_t> encode(StrView text) const;
  std::string decode(const std::vector<int32_t>& ids) const;

  // Байты, которые обозначает токен. У спецтокенов их нет: они управляющие и
  // при декодировании исчезают.
  const std::string& token_bytes(int32_t id) const;

  // Читаемое представление токена для отладки и вывода словаря.
  std::string token_debug_string(int32_t id) const;

  void save(const std::string& path) const;
  static Bpe load(const std::string& path);

  // Разбиение текста на куски, внутри которых разрешены слияния. Возвращает
  // виды на переданный текст: он обязан пережить результат.
  //
  // Открыто ради тестов — от этого правила напрямую зависит, какие токены
  // вообще может выучить модель.
  static std::vector<StrView> pre_tokenize(StrView text);

 private:
  struct Merge {
    int32_t left;
    int32_t right;
  };

  void rebuild_lookup();
  void apply_merges(std::vector<int32_t>* symbols) const;

  std::vector<Merge> merges_;  // в порядке обучения: индекс это ранг
  // Ключ — пара идентификаторов, упакованная в 64 бита. Значение — ранг, он же
  // определяет номер нового токена.
  std::unordered_map<int64_t, int32_t> merge_rank_;
  std::vector<std::string> token_bytes_;
};

}  // namespace llm

#endif  // LLM_TOKENIZER_BPE_H_
