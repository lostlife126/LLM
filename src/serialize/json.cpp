#include "serialize/json.h"

#include <cmath>
#include <cstdlib>
#include <limits>

#include "core/check.h"

namespace llm {
namespace serialize {
namespace {

// Предел вложенности. Разбор рекурсивный, и без предела файл из одних
// открывающих скобок уронил бы процесс по стеку, а не по проверке.
const int kMaxDepth = 64;

class Parser {
 public:
  Parser(StrView text, std::vector<JsonDocument::Node>* nodes)
      : text_(text), nodes_(nodes) {}

  std::size_t parse_document() {
    skip_whitespace();
    const std::size_t root = parse_value(0);
    skip_whitespace();
    fail_if(position_ != text_.size(), "после значения остался мусор");
    return root;
  }

 private:
  void fail_if(bool condition, const char* what) const {
    LLM_CHECK_MSG(!condition, "JSON, байт " << position_ << ": " << what);
  }

  bool at_end() const { return position_ >= text_.size(); }

  char peek() const {
    fail_if(at_end(), "файл кончился раньше значения");
    return text_[position_];
  }

  void skip_whitespace() {
    while (position_ < text_.size()) {
      const char ch = text_[position_];
      if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
        return;
      }
      ++position_;
    }
  }

  void expect(char ch) {
    fail_if(at_end() || text_[position_] != ch, "ожидался другой символ");
    ++position_;
  }

  void expect_literal(StrView literal) {
    fail_if(!text_.substr(position_).starts_with(literal),
            "неизвестное значение");
    position_ += literal.size();
  }

  std::size_t add_node() {
    nodes_->push_back(JsonDocument::Node());
    return nodes_->size() - 1;
  }

  std::size_t parse_value(int depth) {
    fail_if(depth > kMaxDepth, "слишком глубокая вложенность");
    switch (peek()) {
      case '{':
        return parse_object(depth);
      case '[':
        return parse_array(depth);
      case '"':
        return parse_string_node();
      case 't': {
        expect_literal("true");
        const std::size_t node = add_node();
        (*nodes_)[node].kind = JsonKind::kBool;
        (*nodes_)[node].boolean = true;
        return node;
      }
      case 'f': {
        expect_literal("false");
        const std::size_t node = add_node();
        (*nodes_)[node].kind = JsonKind::kBool;
        (*nodes_)[node].boolean = false;
        return node;
      }
      case 'n': {
        expect_literal("null");
        const std::size_t node = add_node();
        (*nodes_)[node].kind = JsonKind::kNull;
        return node;
      }
      default:
        return parse_number();
    }
  }

  std::size_t parse_object(int depth) {
    expect('{');
    const std::size_t node = add_node();
    (*nodes_)[node].kind = JsonKind::kObject;

    skip_whitespace();
    if (peek() == '}') {
      ++position_;
      return node;
    }
    while (true) {
      skip_whitespace();
      fail_if(peek() != '"', "имя поля обязано быть строкой");
      std::string key = parse_string_text();
      skip_whitespace();
      expect(':');
      skip_whitespace();
      const std::size_t value = parse_value(depth + 1);

      // Ссылку на узел нельзя было брать заранее: parse_value добавляет узлы,
      // и вектор мог переехать в памяти. Поэтому обращение по номеру и здесь,
      // и всюду ниже.
      (*nodes_)[node].key.push_back(key);
      (*nodes_)[node].child.push_back(value);

      skip_whitespace();
      const char ch = peek();
      if (ch == ',') {
        ++position_;
        continue;
      }
      fail_if(ch != '}', "в объекте ожидались запятая или закрывающая скобка");
      ++position_;
      return node;
    }
  }

  std::size_t parse_array(int depth) {
    expect('[');
    const std::size_t node = add_node();
    (*nodes_)[node].kind = JsonKind::kArray;

    skip_whitespace();
    if (peek() == ']') {
      ++position_;
      return node;
    }
    while (true) {
      skip_whitespace();
      const std::size_t value = parse_value(depth + 1);
      (*nodes_)[node].child.push_back(value);

      skip_whitespace();
      const char ch = peek();
      if (ch == ',') {
        ++position_;
        continue;
      }
      fail_if(ch != ']', "в массиве ожидались запятая или закрывающая скобка");
      ++position_;
      return node;
    }
  }

  std::size_t parse_string_node() {
    std::string text = parse_string_text();
    const std::size_t node = add_node();
    (*nodes_)[node].kind = JsonKind::kString;
    (*nodes_)[node].text.swap(text);
    return node;
  }

  // Дописывает кодовую точку в UTF-8.
  static void append_utf8(uint32_t code, std::string* out) {
    if (code < 0x80) {
      out->push_back(static_cast<char>(code));
    } else if (code < 0x800) {
      out->push_back(static_cast<char>(0xC0 | (code >> 6)));
      out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code < 0x10000) {
      out->push_back(static_cast<char>(0xE0 | (code >> 12)));
      out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
      out->push_back(static_cast<char>(0xF0 | (code >> 18)));
      out->push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
  }

  uint32_t parse_hex4() {
    fail_if(position_ + 4 > text_.size(), "оборванная \\u-последовательность");
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char ch = text_[position_ + static_cast<std::size_t>(i)];
      uint32_t digit = 0;
      if (ch >= '0' && ch <= '9') {
        digit = static_cast<uint32_t>(ch - '0');
      } else if (ch >= 'a' && ch <= 'f') {
        digit = static_cast<uint32_t>(ch - 'a') + 10;
      } else if (ch >= 'A' && ch <= 'F') {
        digit = static_cast<uint32_t>(ch - 'A') + 10;
      } else {
        fail_if(true, "в \\u ожидалась шестнадцатеричная цифра");
      }
      value = (value << 4) | digit;
    }
    position_ += 4;
    return value;
  }

  std::string parse_string_text() {
    expect('"');
    std::string out;
    while (true) {
      fail_if(at_end(), "строка не закрыта");
      const char ch = text_[position_];
      if (ch == '"') {
        ++position_;
        return out;
      }
      if (ch != '\\') {
        // Управляющие символы обязаны быть экранированы. Проверка не
        // придирка: она ловит обрезанный или двоичный файл, поданный вместо
        // текста, на первом же байте, а не через мегабайт.
        fail_if(static_cast<unsigned char>(ch) < 0x20,
                "неэкранированный управляющий символ в строке");
        out.push_back(ch);
        ++position_;
        continue;
      }
      ++position_;
      fail_if(at_end(), "строка оборвалась на обратной косой черте");
      const char escape = text_[position_++];
      switch (escape) {
        case '"':
          out.push_back('"');
          break;
        case '\\':
          out.push_back('\\');
          break;
        case '/':
          out.push_back('/');
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          uint32_t code = parse_hex4();
          // Суррогатная пара: символы вне базовой плоскости записываются
          // двумя \u подряд. В словарях токенизаторов это эмодзи, и склеить
          // пару обязательно — иначе получатся два недопустимых символа.
          if (code >= 0xD800 && code <= 0xDBFF) {
            fail_if(position_ + 2 > text_.size() || text_[position_] != '\\' ||
                        text_[position_ + 1] != 'u',
                    "верхний суррогат без пары");
            position_ += 2;
            const uint32_t low = parse_hex4();
            fail_if(low < 0xDC00 || low > 0xDFFF,
                    "за верхним суррогатом идёт не нижний");
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
          } else {
            fail_if(code >= 0xDC00 && code <= 0xDFFF,
                    "нижний суррогат без пары");
          }
          append_utf8(code, &out);
          break;
        }
        default:
          fail_if(true, "неизвестная escape-последовательность");
      }
    }
  }

  std::size_t parse_number() {
    const std::size_t begin = position_;
    if (!at_end() && text_[position_] == '-') {
      ++position_;
    }
    fail_if(at_end(), "число оборвалось");
    fail_if(text_[position_] < '0' || text_[position_] > '9',
            "значение не начинается ни с чего осмысленного");
    // Ведущий нуль запрещён стандартом: "01" — не число. Правило выглядит
    // придиркой, но ловит вполне реальный случай — файл, где число записано с
    // выравниванием нулями, читается не как задумано, и лучше узнать об этом
    // сразу.
    if (text_[position_] == '0') {
      ++position_;
      fail_if(position_ < text_.size() && text_[position_] >= '0' &&
                  text_[position_] <= '9',
              "ведущий нуль в числе");
    } else {
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9') {
        ++position_;
      }
    }
    if (position_ < text_.size() && text_[position_] == '.') {
      ++position_;
      fail_if(at_end() || text_[position_] < '0' || text_[position_] > '9',
              "после точки ожидалась цифра");
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9') {
        ++position_;
      }
    }
    if (position_ < text_.size() &&
        (text_[position_] == 'e' || text_[position_] == 'E')) {
      ++position_;
      if (position_ < text_.size() &&
          (text_[position_] == '+' || text_[position_] == '-')) {
        ++position_;
      }
      fail_if(at_end() || text_[position_] < '0' || text_[position_] > '9',
              "в порядке числа ожидалась цифра");
      while (position_ < text_.size() && text_[position_] >= '0' &&
             text_[position_] <= '9') {
        ++position_;
      }
    }

    // strtod нужен завершающий нуль, а вид на текст его не гарантирует,
    // поэтому число копируется. Числа в наших файлах короткие, копия ничего
    // не стоит.
    //
    // Разделитель дробной части strtod берёт из локали. Проект её нигде не
    // меняет, то есть остаётся «C», где разделитель — точка. Полагаться на
    // это можно, но знать об этом надо.
    const std::string digits =
        text_.substr(begin, position_ - begin).to_string();
    const double value = std::strtod(digits.c_str(), nullptr);
    // Число, не помещающееся в double, — это отказ, а не бесконечность.
    //
    // Записать бесконечность в JSON нельзя: слова inf грамматика не знает.
    // Зато её можно получить литералом вроде 1e400, и до сих пор она проходила
    // насквозь. Дальше это не падает: norm_eps и rope_theta читаются из
    // number() напрямую, а проверки конфигурации бесконечность переживают —
    // она и больше нуля, и не меньше любой границы. Видно это было бы только
    // по результату: при бесконечной rope_theta все частоты, кроме нулевой
    // пары, обращаются в нуль, то есть RoPE тихо выключается почти целиком.
    //
    // Проверяется конечность, а не errno. ERANGE strtod ставит и при потере
    // значимости — 1e-400 даёт ноль, — а это законное JSON-число, и отвергать
    // его было бы неверно.
    fail_if(!std::isfinite(value), "число не помещается в double");
    const std::size_t node = add_node();
    (*nodes_)[node].kind = JsonKind::kNumber;
    (*nodes_)[node].number = value;
    return node;
  }

  StrView text_;
  std::vector<JsonDocument::Node>* nodes_;
  std::size_t position_ = 0;
};

}  // namespace

JsonDocument JsonDocument::parse(StrView text) {
  JsonDocument document;
  Parser parser(text, &document.nodes_);
  const std::size_t root = parser.parse_document();
  // Корень всегда нулевой узел: разбор начинается с него, и add_node в первый
  // раз вернул именно нуль.
  LLM_CHECK_EQ(root, static_cast<std::size_t>(0));
  return document;
}

JsonValue JsonDocument::root() const {
  LLM_CHECK_MSG(!nodes_.empty(), "документ пуст");
  return JsonValue(this, 0);
}

// Единственное место, где ручка превращается в узел.
const JsonDocument::Node& JsonValue::node() const {
  LLM_CHECK_MSG(document_ != nullptr, "обращение к неопределённому значению");
  LLM_DCHECK_LT(index_, document_->nodes_.size());
  return document_->nodes_[index_];
}

JsonKind JsonValue::kind() const { return node().kind; }

bool JsonValue::boolean() const {
  LLM_CHECK_MSG(kind() == JsonKind::kBool, "значение не логическое");
  return node().boolean;
}

double JsonValue::number() const {
  LLM_CHECK_MSG(kind() == JsonKind::kNumber, "значение не число");
  return node().number;
}

int64_t JsonValue::integer() const {
  const double value = number();
  LLM_CHECK_MSG(std::floor(value) == value,
                "ожидалось целое, получено " << value);
  // Границу берём 2^53: дальше double уже не различает соседние целые, и
  // проверка «влезает в int64_t» перестала бы что-либо значить.
  const double limit = 9007199254740992.0;
  LLM_CHECK_MSG(value >= -limit && value <= limit,
                "целое вне точного диапазона double: " << value);
  return static_cast<int64_t>(value);
}

std::string JsonValue::text() const {
  LLM_CHECK_MSG(kind() == JsonKind::kString, "значение не строка");
  return node().text;
}

std::size_t JsonValue::size() const {
  const JsonKind k = kind();
  LLM_CHECK_MSG(k == JsonKind::kArray || k == JsonKind::kObject,
                "размер есть только у массива и объекта");
  return node().child.size();
}

JsonValue JsonValue::at(std::size_t index) const {
  LLM_CHECK_MSG(kind() == JsonKind::kArray, "значение не массив");
  const JsonDocument::Node& self = node();
  LLM_CHECK_LT(index, self.child.size());
  return JsonValue(document_, self.child[index]);
}

std::string JsonValue::key_at(std::size_t index) const {
  LLM_CHECK_MSG(kind() == JsonKind::kObject, "значение не объект");
  const JsonDocument::Node& self = node();
  LLM_CHECK_LT(index, self.key.size());
  return self.key[index];
}

JsonValue JsonValue::value_at(std::size_t index) const {
  LLM_CHECK_MSG(kind() == JsonKind::kObject, "значение не объект");
  const JsonDocument::Node& self = node();
  LLM_CHECK_LT(index, self.child.size());
  return JsonValue(document_, self.child[index]);
}

JsonValue JsonValue::find(StrView key) const {
  LLM_CHECK_MSG(kind() == JsonKind::kObject, "значение не объект");
  const JsonDocument::Node& self = node();
  for (std::size_t i = 0; i < self.key.size(); ++i) {
    if (StrView(self.key[i]) == key) {
      return JsonValue(document_, self.child[i]);
    }
  }
  return JsonValue();
}

JsonValue JsonValue::field(StrView key) const {
  const JsonValue value = find(key);
  LLM_CHECK_MSG(value.defined(), "в объекте нет поля " << key);
  return value;
}

}  // namespace serialize
}  // namespace llm
