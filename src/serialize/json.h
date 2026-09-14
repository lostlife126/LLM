// Разбор JSON — ровно столько, сколько нужно для чужих файлов весов.
//
// Понадобился в трёх местах сразу: заголовок safetensors — это JSON, описание
// модели в config.json — это JSON, и словарь токенизатора в tokenizer.json —
// тоже. Писать три разбора по месту значило бы три раза ошибиться в escape-
// последовательностях.
//
// Устройство: дерево хранится не узлами со ссылками друг на друга, а плоским
// массивом, где потомки указываются номерами. Причина не в скорости, а в
// правилах языка: класс, у которого есть поле std::vector<самого себя>, до
// C++17 формально недопустим — контейнер требует полного типа. Плоский массив
// снимает вопрос и заодно даёт обход без разыменований указателей.
//
// JsonValue — лёгкая ручка: документ плюс номер узла. Живёт ровно столько,
// сколько живёт документ, из которого получена; копировать ручки дёшево,
// хранить их дольше документа нельзя.
//
// Чего здесь нет намеренно: сериализации (писать JSON проекту негде),
// сохранения порядка дубликатов ключей (в наших файлах их нет) и разбора
// «на лету» без материализации (файлы читаются целиком и без того).

#ifndef LLM_SERIALIZE_JSON_H_
#define LLM_SERIALIZE_JSON_H_

#include <cstdint>
#include <string>
#include <vector>

#include "core/str_view.h"

namespace llm {
namespace serialize {

enum class JsonKind {
  kNull,
  kBool,
  kNumber,
  kString,
  kArray,
  kObject,
};

class JsonValue;

// Документ идёт первым: ручке нужен его вложенный тип узла, а вернуть ручку
// по значению можно и из объявления — для этого полный тип не требуется.
class JsonDocument {
 public:
  // Узел плоского массива. Открыт не ради пользователей, а потому, что его
  // имя нужно и разбору, и ручке.
  struct Node {
    JsonKind kind = JsonKind::kNull;
    bool boolean = false;
    double number = 0.0;
    std::string text;  // строка, либо ничего
    std::vector<std::size_t> child;  // элементы массива или значения полей
    std::vector<std::string> key;  // имена полей, параллельно child
  };

  // Разбирает текст целиком. Любая ошибка — исключение с позицией в байтах:
  // файлы приходят извне, и «не удалось разобрать» без указания места
  // бесполезно.
  //
  // Глубина вложенности ограничена: разбор рекурсивный, а файл чужой, и
  // цепочка из миллиона открывающих скобок иначе снесла бы стек.
  static JsonDocument parse(StrView text);

  JsonValue root() const;

 private:
  friend class JsonValue;

  std::vector<Node> nodes_;
};

// Ручка на один узел разобранного документа.
class JsonValue {
 public:
  JsonValue() : document_(nullptr), index_(0) {}

  JsonKind kind() const;
  bool defined() const { return document_ != nullptr; }

  bool is_null() const { return kind() == JsonKind::kNull; }
  bool is_bool() const { return kind() == JsonKind::kBool; }
  bool is_number() const { return kind() == JsonKind::kNumber; }
  bool is_string() const { return kind() == JsonKind::kString; }
  bool is_array() const { return kind() == JsonKind::kArray; }
  bool is_object() const { return kind() == JsonKind::kObject; }

  bool boolean() const;
  double number() const;

  // Целое с проверкой, что оно действительно целое и влезает в int64_t.
  // В чужих файлах размерности и смещения — целые, и молча округлить их
  // означало бы прочитать не тот кусок файла.
  int64_t integer() const;

  // Строка возвращается по значению, а не ссылкой внутрь документа. Ссылка
  // была бы быстрее и при этом ловушкой: выражение вида
  // JsonDocument::parse(text).root().text() возвращало бы ссылку на строку
  // уже уничтоженного документа, и компилятор такое не ловит. Копия строки на
  // фоне разбора мегабайтного словаря не стоит ничего.
  std::string text() const;

  // Число элементов массива или полей объекта.
  std::size_t size() const;

  // Элемент массива по номеру.
  JsonValue at(std::size_t index) const;

  // Имя и значение поля объекта по номеру — для обхода целиком.
  std::string key_at(std::size_t index) const;
  JsonValue value_at(std::size_t index) const;

  // Поле объекта по имени. find возвращает неопределённую ручку, если поля
  // нет; field требует, чтобы поле было, и называет отсутствующее имя в
  // сообщении. Разница существенна: необязательное поле и опечатка в имени —
  // разные ошибки, и путать их дорого.
  JsonValue find(StrView key) const;
  JsonValue field(StrView key) const;
  bool has(StrView key) const { return find(key).defined(); }

 private:
  friend class JsonDocument;
  JsonValue(const JsonDocument* document, std::size_t index)
      : document_(document), index_(index) {}

  const JsonDocument::Node& node() const;

  const JsonDocument* document_;
  std::size_t index_;
};

}  // namespace serialize
}  // namespace llm

#endif  // LLM_SERIALIZE_JSON_H_
