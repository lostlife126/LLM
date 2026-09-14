// Чтение safetensors.
//
// Файлы для проверок собираются прямо здесь, байт за байтом. Так проверяется
// не только чтение правильного файла, но и отказ от испорченного — а именно
// это главное: тензор, прочитанный по неверному смещению, даёт не ошибку, а
// правдоподобные числа.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "serialize/safetensors.h"
#include "testing.h"

using llm::serialize::SafeTensors;

namespace {

// Путь во временном каталоге. Имя включает номер, чтобы тесты не мешали друг
// другу, если однажды пойдут не по очереди.
std::string temp_path(int number) {
  return "/tmp/llm_safetensors_test_" + std::to_string(number) + ".bin";
}

void write_file(const std::string& path, const std::string& header,
                const std::string& data) {
  std::ofstream file(path.c_str(), std::ios::binary);
  const uint64_t length = header.size();
  char bytes[8];
  for (int i = 0; i < 8; ++i) {
    bytes[i] = static_cast<char>((length >> (8 * i)) & 0xFFu);
  }
  file.write(bytes, 8);
  file.write(header.data(), static_cast<std::streamsize>(header.size()));
  file.write(data.data(), static_cast<std::streamsize>(data.size()));
}

std::string float_bytes(const std::vector<float>& values) {
  std::string out(values.size() * sizeof(float), '\0');
  if (!values.empty()) {
    std::memcpy(&out[0], values.data(), out.size());
  }
  return out;
}

std::string u16_bytes(const std::vector<uint16_t>& values) {
  std::string out(values.size() * 2, '\0');
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[2 * i] = static_cast<char>(values[i] & 0xFFu);
    out[2 * i + 1] = static_cast<char>((values[i] >> 8) & 0xFFu);
  }
  return out;
}

}  // namespace

LLM_TEST(Safetensors, ReadsFloat32) {
  const std::string path = temp_path(1);
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  write_file(path,
             "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2,3],"
             "\"data_offsets\":[0,24]}}",
             float_bytes(values));

  const SafeTensors file = SafeTensors::load(path);
  LLM_CHECK_EQ(file.size(), static_cast<std::size_t>(1));
  LLM_CHECK(file.has("weight"));
  LLM_CHECK(!file.has("bias"));

  const llm::Tensor tensor = file.read("weight");
  LLM_CHECK_EQ(tensor.rank(), 2);
  LLM_CHECK_EQ(tensor.dim(0), static_cast<int64_t>(2));
  LLM_CHECK_EQ(tensor.dim(1), static_cast<int64_t>(3));
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t column = 0; column < 3; ++column) {
      LLM_EXPECT_NEAR(tensor(row, column),
                      static_cast<double>(
                          values[static_cast<std::size_t>(row * 3 + column)]),
                      0.0);
    }
  }
  std::remove(path.c_str());
}

LLM_TEST(Safetensors, ReadsSeveralTensorsAndMetadata) {
  const std::string path = temp_path(2);
  const std::vector<float> values = {1.0f, 2.0f, 3.0f};
  write_file(path,
             "{\"__metadata__\":{\"format\":\"pt\"},"
             "\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
             "\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}",
             float_bytes(values));

  const SafeTensors file = SafeTensors::load(path);
  // __metadata__ не тензор и в список попасть не должен.
  LLM_CHECK_EQ(file.size(), static_cast<std::size_t>(2));
  LLM_CHECK_EQ(file.metadata().size(), static_cast<std::size_t>(1));
  LLM_CHECK_EQ(file.metadata()[0].first, std::string("format"));
  LLM_CHECK_EQ(file.metadata()[0].second, std::string("pt"));

  LLM_EXPECT_NEAR(file.read("a")(1), 2.0, 0.0);
  LLM_EXPECT_NEAR(file.read("b")(0), 3.0, 0.0);
  std::remove(path.c_str());
}

LLM_TEST(Safetensors, Bf16IsTopHalfOfFloat32) {
  // bfloat16 — это ровно старшие шестнадцать бит float32. Проверка на
  // известных величинах, а не на круговом преобразовании: иначе ошибка,
  // симметричная в обе стороны, осталась бы незамеченной.
  LLM_EXPECT_NEAR(llm::serialize::bf16_to_float(0x3F80), 1.0, 0.0);
  LLM_EXPECT_NEAR(llm::serialize::bf16_to_float(0xBF80), -1.0, 0.0);
  LLM_EXPECT_NEAR(llm::serialize::bf16_to_float(0x0000), 0.0, 0.0);
  LLM_EXPECT_NEAR(llm::serialize::bf16_to_float(0x4049), 3.140625, 0.0);
  LLM_EXPECT_NEAR(llm::serialize::bf16_to_float(0x4000), 2.0, 0.0);

  // Бесконечность обязана остаться бесконечностью: у весов её быть не должно,
  // но если она там есть, лучше это увидеть, чем получить большое число.
  LLM_CHECK(std::isinf(llm::serialize::bf16_to_float(0x7F80)));
  LLM_CHECK(std::isnan(llm::serialize::bf16_to_float(0x7FC0)));
}

LLM_TEST(Safetensors, F16KnownValues) {
  LLM_EXPECT_NEAR(llm::serialize::f16_to_float(0x3C00), 1.0, 0.0);
  LLM_EXPECT_NEAR(llm::serialize::f16_to_float(0xC000), -2.0, 0.0);
  LLM_EXPECT_NEAR(llm::serialize::f16_to_float(0x0000), 0.0, 0.0);
  LLM_EXPECT_NEAR(llm::serialize::f16_to_float(0x3555), 0.333251953125, 0.0);

  // Наименьшее нормальное: 2^-14.
  LLM_EXPECT_NEAR(llm::serialize::f16_to_float(0x0400), 6.103515625e-05, 0.0);

  // Субнормальные — отдельная ветвь, и именно её легче всего написать
  // неправильно: в half они денормализованы, а в float вполне нормальны.
  LLM_EXPECT_NEAR(llm::serialize::f16_to_float(0x0200), 3.0517578125e-05, 0.0);
  LLM_EXPECT_NEAR(llm::serialize::f16_to_float(0x0001), 5.9604644775390625e-08,
                  0.0);
  LLM_EXPECT_NEAR(llm::serialize::f16_to_float(0x8001), -5.9604644775390625e-08,
                  0.0);

  LLM_CHECK(std::isinf(llm::serialize::f16_to_float(0x7C00)));
  LLM_CHECK(std::isnan(llm::serialize::f16_to_float(0x7E00)));
}

LLM_TEST(Safetensors, ReadsHalfPrecisionTensors) {
  const std::string path = temp_path(3);
  write_file(path,
             "{\"h\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,4]},"
             "\"b\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[4,8]}}",
             u16_bytes({0x3C00, 0xC000, 0x3F80, 0x4000}));

  const SafeTensors file = SafeTensors::load(path);
  const llm::Tensor half = file.read("h");
  LLM_EXPECT_NEAR(half(0), 1.0, 0.0);
  LLM_EXPECT_NEAR(half(1), -2.0, 0.0);

  const llm::Tensor brain = file.read("b");
  LLM_EXPECT_NEAR(brain(0), 1.0, 0.0);
  LLM_EXPECT_NEAR(brain(1), 2.0, 0.0);
  std::remove(path.c_str());
}

LLM_TEST(Safetensors, RejectsLengthNotMatchingShape) {
  // Самая опасная разновидность порчи: заголовок разбирается, форма
  // правдоподобна, но кусок другого размера. Без проверки прочиталась бы
  // соседняя память.
  const std::string path = temp_path(4);
  write_file(path,
             "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,3],"
             "\"data_offsets\":[0,20]}}",
             float_bytes({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
  LLM_EXPECT_THROWS(SafeTensors::load(path));
  std::remove(path.c_str());
}

LLM_TEST(Safetensors, RejectsOffsetsPastEnd) {
  const std::string path = temp_path(5);
  write_file(path,
             "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],"
             "\"data_offsets\":[0,16]}}",
             float_bytes({1.0f, 2.0f}));
  LLM_EXPECT_THROWS(SafeTensors::load(path));
  std::remove(path.c_str());
}

LLM_TEST(Safetensors, RejectsOverlappingTensors) {
  const std::string path = temp_path(6);
  write_file(path,
             "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
             "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}",
             float_bytes({1.0f, 2.0f, 3.0f}));
  LLM_EXPECT_THROWS(SafeTensors::load(path));
  std::remove(path.c_str());
}

LLM_TEST(Safetensors, RejectsUnsupportedDtype) {
  // Квантованная модель обязана отвергаться с внятным сообщением, а не
  // читаться как попало.
  const std::string path = temp_path(7);
  write_file(path,
             "{\"w\":{\"dtype\":\"I8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
             std::string(4, '\x01'));
  LLM_EXPECT_THROWS(SafeTensors::load(path));
  std::remove(path.c_str());
}

LLM_TEST(Safetensors, RejectsNonsenseFiles) {
  const std::string path = temp_path(8);

  // Слишком короткий файл.
  {
    std::ofstream file(path.c_str(), std::ios::binary);
    file.write("abc", 3);
  }
  LLM_EXPECT_THROWS(SafeTensors::load(path));

  // Длина заголовка правдоподобна, а самого заголовка нет.
  {
    std::ofstream file(path.c_str(), std::ios::binary);
    const char bytes[8] = {0x40, 0, 0, 0, 0, 0, 0, 0};
    file.write(bytes, 8);
  }
  LLM_EXPECT_THROWS(SafeTensors::load(path));

  // Заголовок не JSON.
  write_file(path, "не json", std::string());
  LLM_EXPECT_THROWS(SafeTensors::load(path));

  // Заголовок — JSON, но не объект.
  write_file(path, "[1,2,3]", std::string());
  LLM_EXPECT_THROWS(SafeTensors::load(path));

  // Нет обязательного поля.
  write_file(path, "{\"w\":{\"dtype\":\"F32\",\"shape\":[1]}}",
             float_bytes({1.0f}));
  LLM_EXPECT_THROWS(SafeTensors::load(path));

  std::remove(path.c_str());
  LLM_EXPECT_THROWS(SafeTensors::load("/tmp/этого файла нет.bin"));
}

LLM_TEST(Safetensors, MissingTensorNamesWhatIsThere) {
  // Имена тензоров — первое, что расходится между реализациями. Сообщение
  // обязано показывать, что в файле есть на самом деле, иначе отладка
  // превращается в угадывание.
  const std::string path = temp_path(9);
  write_file(path,
             "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\","
             "\"shape\":[1],\"data_offsets\":[0,4]}}",
             float_bytes({1.0f}));
  const SafeTensors file = SafeTensors::load(path);

  bool threw = false;
  try {
    file.read("embed_tokens.weight");
  } catch (const llm::CheckFailure& error) {
    threw = true;
    const std::string message = error.what();
    LLM_CHECK(message.find("model.embed_tokens.weight") != std::string::npos);
  }
  LLM_CHECK(threw);
  std::remove(path.c_str());
}

LLM_TEST(Safetensors, EmptyTensorIsAllowed) {
  // Нулевая размерность законна и встречается: так выглядит, например,
  // отсутствующий свободный член, записанный на всякий случай.
  const std::string path = temp_path(10);
  write_file(path,
             "{\"w\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}}",
             std::string());
  const SafeTensors file = SafeTensors::load(path);
  LLM_CHECK_EQ(file.read("w").dim(0), static_cast<int64_t>(0));
  std::remove(path.c_str());
}
