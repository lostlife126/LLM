// Чтение safetensors — формата, в котором сегодня выкладывают веса моделей.
//
// Устройство файла предельно простое, и в этом его смысл. Прежние форматы
// (pickle у PyTorch) при загрузке выполняли произвольный код: чтобы прочитать
// чужие веса, приходилось доверять чужому файлу. Здесь читать нечего, кроме
// чисел:
//
//   8 байт      длина заголовка, little-endian
//   заголовок   JSON: имя тензора -> {dtype, shape, data_offsets}
//   данные      тензоры подряд, смещения считаются от конца заголовка
//
// Никакого сжатия, никакого выравнивания внутри, никакого порядка: тензоры
// лежат в том порядке, в каком их записали.
//
// Что здесь проверяется сверх разбора: что смещения не выходят за файл, что
// длина куска отвечает форме и типу, и что куски не налезают друг на друга.
// Пропустить любую из этих проверок значит прочитать соседний тензор вместо
// нужного и получить модель, которая работает и выдаёт связный мусор — самая
// дорогая в отладке разновидность ошибки.

#ifndef LLM_SERIALIZE_SAFETENSORS_H_
#define LLM_SERIALIZE_SAFETENSORS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/tensor.h"

namespace llm {
namespace serialize {

// Типы, которые встречаются в файлах языковых моделей. Всё остальное —
// отказ с указанием типа: молча прочитать int8 как float значит получить
// мусор вместо весов.
enum class SafeDtype {
  kF32,
  kF16,
  kBf16,
};

struct SafeTensorEntry {
  std::string name;
  SafeDtype dtype = SafeDtype::kF32;
  std::vector<int64_t> shape;
  // Смещения от начала области данных, а не от начала файла.
  uint64_t begin = 0;
  uint64_t end = 0;

  int64_t numel() const;
};

// Читает файл целиком в память.
//
// Модели, ради которых это писалось, — сотни мегабайт, и держать их целиком
// допустимо. Отображение файла в память было бы экономнее, но потребовало бы
// системных вызовов, разных на разных платформах, а выигрыш здесь
// одноразовый: веса всё равно тут же переписываются в наш формат.
class SafeTensors {
 public:
  static SafeTensors load(const std::string& path);

  const std::vector<SafeTensorEntry>& entries() const { return entries_; }
  std::size_t size() const { return entries_.size(); }

  bool has(const std::string& name) const;

  // Имена всех тензоров — для сообщения об ошибке, когда нужного нет.
  std::vector<std::string> names() const;

  // Читает тензор по имени, преобразуя к float32.
  //
  // Форма берётся из файла. Одномерные и двумерные тензоры — всё, что бывает
  // в трансформере без свёрток, но ограничения на ранг здесь нет.
  Tensor read(const std::string& name) const;

  // Метаданные из поля __metadata__, если оно есть. Туда кладут формат
  // («pt») и иногда происхождение модели; для загрузки не нужны, но в
  // сообщении об ошибке помогают понять, что за файл подсунули.
  const std::vector<std::pair<std::string, std::string>>& metadata() const {
    return metadata_;
  }

 private:
  const SafeTensorEntry& entry(const std::string& name) const;

  std::string path_;
  std::vector<SafeTensorEntry> entries_;
  std::vector<std::pair<std::string, std::string>> metadata_;
  std::vector<char> data_;
};

// Преобразования половинной точности. Вынесены наружу ради тестов: ошибка
// здесь портит все веса разом и притом незаметно — числа остаются
// правдоподобными.
float bf16_to_float(uint16_t bits);
float f16_to_float(uint16_t bits);

std::size_t dtype_size(SafeDtype dtype);
const char* dtype_name(SafeDtype dtype);

}  // namespace serialize
}  // namespace llm

#endif  // LLM_SERIALIZE_SAFETENSORS_H_
