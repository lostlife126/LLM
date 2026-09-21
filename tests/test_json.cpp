// Разбор JSON. Файлы приходят извне, поэтому проверяется не только то, что
// правильное читается правильно, но и то, что неправильное отвергается, а не
// молча превращается в правдоподобный мусор.

#include <string>

#include "serialize/json.h"
#include "testing.h"

using llm::serialize::JsonDocument;
using llm::serialize::JsonKind;
using llm::serialize::JsonValue;

namespace {

JsonValue root_of(const JsonDocument& document) { return document.root(); }

}  // namespace

LLM_TEST(Json, ScalarKinds) {
  const JsonDocument null_document = JsonDocument::parse("null");
  LLM_CHECK(root_of(null_document).is_null());

  const JsonDocument yes = JsonDocument::parse("true");
  LLM_CHECK(yes.root().boolean());
  const JsonDocument no = JsonDocument::parse("false");
  LLM_CHECK(!no.root().boolean());

  const JsonDocument number = JsonDocument::parse("-12.5e2");
  LLM_EXPECT_NEAR(number.root().number(), -1250.0, 1e-9);

  const JsonDocument text = JsonDocument::parse("\"привет\"");
  LLM_CHECK_EQ(text.root().text(), std::string("привет"));
}

LLM_TEST(Json, IntegerRejectsFractional) {
  const JsonDocument whole = JsonDocument::parse("42");
  LLM_CHECK_EQ(whole.root().integer(), static_cast<int64_t>(42));

  // Размерности и смещения в чужих файлах целые. Если там оказалось дробное,
  // округлить его молча значит прочитать не тот кусок файла.
  const JsonDocument fractional = JsonDocument::parse("42.5");
  LLM_EXPECT_THROWS(fractional.root().integer());
}

LLM_TEST(Json, StringEscapes) {
  const JsonDocument document =
      JsonDocument::parse("\"a\\\"b\\\\c\\n\\t\\u0041\\u00e9\"");
  LLM_CHECK_EQ(document.root().text(), std::string("a\"b\\c\n\tA\xc3\xa9"));
}

LLM_TEST(Json, SurrogatePairBecomesOneCharacter) {
  // В словарях токенизаторов встречаются символы вне базовой плоскости, и
  // записаны они парой \u. Склеивать пару обязательно: по отдельности это два
  // недопустимых значения.
  const JsonDocument document = JsonDocument::parse("\"\\ud83d\\ude00\"");
  LLM_CHECK_EQ(document.root().text(), std::string("\xf0\x9f\x98\x80"));

  LLM_EXPECT_THROWS(JsonDocument::parse("\"\\ud83d\""));
  LLM_EXPECT_THROWS(JsonDocument::parse("\"\\ude00\""));
  LLM_EXPECT_THROWS(JsonDocument::parse("\"\\ud83d\\u0041\""));
  // Сама ветка «нижний суррогат без пары» проверена строкой выше, а вот её
  // границы — нет. Это не придирка: при условии со строгими неравенствами
  // 0xDE00 по-прежнему отвергался бы, а края диапазона проходили бы насквозь.
  LLM_EXPECT_THROWS(JsonDocument::parse("\"\\udc00\""));
  LLM_EXPECT_THROWS(JsonDocument::parse("\"\\udfff\""));
}

LLM_TEST(Json, ArrayAndObject) {
  const JsonDocument document = JsonDocument::parse(
      "{\"dtype\":\"F32\",\"shape\":[2,3],\"offsets\":[0,24]}");
  const JsonValue root = document.root();
  LLM_CHECK(root.is_object());
  LLM_CHECK_EQ(root.size(), static_cast<std::size_t>(3));
  LLM_CHECK_EQ(root.field("dtype").text(), std::string("F32"));

  const JsonValue shape = root.field("shape");
  LLM_CHECK(shape.is_array());
  LLM_CHECK_EQ(shape.size(), static_cast<std::size_t>(2));
  LLM_CHECK_EQ(shape.at(0).integer(), static_cast<int64_t>(2));
  LLM_CHECK_EQ(shape.at(1).integer(), static_cast<int64_t>(3));

  // Обход объекта по номерам — так читается заголовок safetensors, где имена
  // тензоров заранее неизвестны.
  LLM_CHECK_EQ(root.key_at(0), std::string("dtype"));
  LLM_CHECK_EQ(root.value_at(2).at(1).integer(), static_cast<int64_t>(24));
}

LLM_TEST(Json, MissingFieldIsDistinctFromAbsentValue) {
  const JsonDocument document =
      JsonDocument::parse("{\"present\":1,\"empty\":null}");
  const JsonValue root = document.root();

  // Поля нет — find возвращает неопределённую ручку, field бросает.
  LLM_CHECK(!root.find("absent").defined());
  LLM_CHECK(!root.has("absent"));
  LLM_EXPECT_THROWS(root.field("absent"));

  // Поле есть, но со значением null — это не то же самое.
  LLM_CHECK(root.has("empty"));
  LLM_CHECK(root.field("empty").is_null());
}

LLM_TEST(Json, EmptyContainers) {
  const JsonDocument document = JsonDocument::parse("{\"a\":[],\"b\":{}}");
  LLM_CHECK_EQ(document.root().field("a").size(), static_cast<std::size_t>(0));
  LLM_CHECK_EQ(document.root().field("b").size(), static_cast<std::size_t>(0));
}

LLM_TEST(Json, WhitespaceEverywhere) {
  const JsonDocument document = JsonDocument::parse(
      " {\n  \"a\" : [ 1 , 2 ] ,\r\n\t\"b\" : { \"c\" : true }\n } ");
  LLM_CHECK_EQ(document.root().field("a").at(1).integer(),
               static_cast<int64_t>(2));
  LLM_CHECK(document.root().field("b").field("c").boolean());
}

LLM_TEST(Json, WrongTypeThrows) {
  const JsonDocument document = JsonDocument::parse("{\"a\":1}");
  const JsonValue root = document.root();
  LLM_EXPECT_THROWS(root.text());
  LLM_EXPECT_THROWS(root.at(0));
  LLM_EXPECT_THROWS(root.field("a").field("b"));
  LLM_EXPECT_THROWS(root.field("a").size());
}

LLM_TEST(Json, MalformedInputThrows) {
  LLM_EXPECT_THROWS(JsonDocument::parse(""));
  LLM_EXPECT_THROWS(JsonDocument::parse("{"));
  LLM_EXPECT_THROWS(JsonDocument::parse("{\"a\"}"));
  LLM_EXPECT_THROWS(JsonDocument::parse("{\"a\":}"));
  LLM_EXPECT_THROWS(JsonDocument::parse("{\"a\":1,}"));
  LLM_EXPECT_THROWS(JsonDocument::parse("[1 2]"));
  LLM_EXPECT_THROWS(JsonDocument::parse("{a:1}"));
  LLM_EXPECT_THROWS(JsonDocument::parse("tru"));
  LLM_EXPECT_THROWS(JsonDocument::parse("01"));
  LLM_EXPECT_THROWS(JsonDocument::parse("1."));
  LLM_EXPECT_THROWS(JsonDocument::parse("1e"));
  LLM_EXPECT_THROWS(JsonDocument::parse("\"не закрыта"));
  LLM_EXPECT_THROWS(JsonDocument::parse("\"\\q\""));
  // Лишнее после значения — верный признак, что файл не тот.
  LLM_EXPECT_THROWS(JsonDocument::parse("{} {}"));
}

LLM_TEST(Json, NumberTooLargeForDoubleThrows) {
  // Бесконечность в JSON не записывается словом, зато получается литералом, и
  // до сих пор она проходила насквозь. Падения дальше не было бы: norm_eps и
  // rope_theta читаются из number() прямо, а проверки конфигурации
  // бесконечность переживают — она больше нуля и не меньше любой границы.
  LLM_EXPECT_THROWS(JsonDocument::parse("1e400"));
  LLM_EXPECT_THROWS(JsonDocument::parse("-1e400"));
  LLM_EXPECT_THROWS(JsonDocument::parse("{\"rope_theta\": 1e400}"));

  // А потеря значимости — не отказ. strtod ставит ERANGE и на неё тоже,
  // поэтому проверка по errno отвергла бы законное число; проверяется
  // конечность, и здесь она соблюдена.
  const JsonDocument tiny = JsonDocument::parse("1e-400");
  LLM_EXPECT_NEAR(tiny.root().number(), 0.0, 0.0);

  // Граница представимого проходит: это ещё число, а не бесконечность.
  const JsonDocument big = JsonDocument::parse("1e308");
  LLM_CHECK(big.root().number() > 1e307);
}

LLM_TEST(Json, RawControlCharacterInStringThrows) {
  // Двоичный файл, поданный вместо текста, обязан отвергаться на первом же
  // управляющем байте, а не через мегабайт разбора.
  const std::string broken = std::string("\"a") + '\x01' + "b\"";
  LLM_EXPECT_THROWS(JsonDocument::parse(broken));
}

LLM_TEST(Json, DeepNestingThrowsInsteadOfCrashing) {
  // Разбор рекурсивный, и глубина в файле задаётся снаружи. Без предела это
  // был бы не отказ, а падение по стеку.
  std::string deep(4096, '[');
  LLM_EXPECT_THROWS(JsonDocument::parse(deep));
}

LLM_TEST(Json, NestedStructureSurvivesReallocation) {
  // Узлы лежат в одном векторе, который по мере разбора переезжает в памяти.
  // Если бы разбор держал ссылку на узел через это перемещение, дерево вышло
  // бы испорченным — заметно это только на достаточно длинном документе.
  std::string text = "{\"items\":[";
  const int count = 512;
  for (int i = 0; i < count; ++i) {
    if (i != 0) {
      text += ",";
    }
    text += "{\"i\":" + std::to_string(i) + "}";
  }
  text += "]}";

  const JsonDocument document = JsonDocument::parse(text);
  const JsonValue items = document.root().field("items");
  LLM_CHECK_EQ(items.size(), static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    LLM_CHECK_EQ(items.at(static_cast<std::size_t>(i)).field("i").integer(),
                 static_cast<int64_t>(i));
  }
}
