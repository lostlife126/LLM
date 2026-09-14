#include "serialize/safetensors.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

#include "core/check.h"
#include "serialize/json.h"

namespace llm {
namespace serialize {
namespace {

// Предел на длину заголовка. Заголовок — это список тензоров; даже у модели с
// тысячами тензоров он не доходит до мегабайта. Проверка нужна не от жадности,
// а чтобы поданный по ошибке чужой файл не заставил выделить память по
// случайным восьми байтам.
const uint64_t kMaxHeaderBytes = 64u * 1024u * 1024u;

bool machine_is_little_endian() {
  const uint32_t probe = 1;
  unsigned char bytes[4];
  std::memcpy(bytes, &probe, sizeof(probe));
  return bytes[0] == 1;
}

uint64_t read_u64_le(const char* bytes) {
  uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8) |
            static_cast<uint64_t>(static_cast<unsigned char>(bytes[i]));
  }
  return value;
}

SafeDtype parse_dtype(const std::string& name, const std::string& tensor) {
  if (name == "F32") {
    return SafeDtype::kF32;
  }
  if (name == "F16") {
    return SafeDtype::kF16;
  }
  if (name == "BF16") {
    return SafeDtype::kBf16;
  }
  // Отдельным сообщением, а не общим «не поддерживается»: увидеть, что веса
  // лежат в I8 или F8, — это сразу понять, что модель квантованная, а не
  // гадать, почему файл не читается.
  LLM_CHECK_MSG(false, "тензор " << tensor << ": тип " << name
                                 << " не поддерживается, нужен F32, F16 "
                                    "или BF16");
  return SafeDtype::kF32;
}

}  // namespace

std::size_t dtype_size(SafeDtype dtype) {
  return dtype == SafeDtype::kF32 ? 4u : 2u;
}

const char* dtype_name(SafeDtype dtype) {
  switch (dtype) {
    case SafeDtype::kF32:
      return "F32";
    case SafeDtype::kF16:
      return "F16";
    case SafeDtype::kBf16:
      return "BF16";
  }
  return "?";
}

float bf16_to_float(uint16_t bits) {
  // bfloat16 — это старшие шестнадцать бит float32, и больше ничего. Ни
  // бесконечности, ни нули, ни NaN особого обращения не требуют: у них те же
  // поля, только мантисса короче.
  const uint32_t wide = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

float f16_to_float(uint16_t bits) {
  // IEEE half: знак 1 бит, порядок 5, мантисса 10. Смещение порядка 15
  // против 127 у float32, отсюда +112 в нормальном случае.
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
  const uint32_t exponent = (static_cast<uint32_t>(bits) >> 10) & 0x1Fu;
  const uint32_t mantissa = static_cast<uint32_t>(bits) & 0x3FFu;

  uint32_t wide = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      wide = sign;  // ±0
    } else {
      // Субнормальное половинной точности — вполне нормальное в одинарной:
      // разряда хватает. Нормализация сдвигает мантиссу до неявной единицы,
      // уменьшая порядок на каждый шаг.
      uint32_t normalized_exponent = 127u - 15u + 1u;
      uint32_t normalized_mantissa = mantissa;
      while ((normalized_mantissa & 0x400u) == 0) {
        normalized_mantissa <<= 1;
        --normalized_exponent;
      }
      normalized_mantissa &= 0x3FFu;
      wide = sign | (normalized_exponent << 23) | (normalized_mantissa << 13);
    }
  } else if (exponent == 0x1Fu) {
    wide = sign | 0x7F800000u | (mantissa << 13);  // бесконечность или NaN
  } else {
    wide = sign | ((exponent + 112u) << 23) | (mantissa << 13);
  }

  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

int64_t SafeTensorEntry::numel() const {
  int64_t total = 1;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    total *= shape[i];
  }
  return total;
}

SafeTensors SafeTensors::load(const std::string& path) {
  LLM_CHECK_MSG(machine_is_little_endian(),
                "safetensors — формат little-endian, а машина другая; "
                "преобразование не написано намеренно, чтобы не выдавать "
                "непроверенный результат за верный");

  std::ifstream file(path.c_str(), std::ios::binary);
  LLM_CHECK_MSG(file.good(), "не удалось открыть " << path);

  SafeTensors result;
  result.path_ = path;

  char length_bytes[8];
  file.read(length_bytes, 8);
  LLM_CHECK_MSG(file.gcount() == 8,
                path << " короче восьми байт — это не safetensors");
  const uint64_t header_bytes = read_u64_le(length_bytes);
  LLM_CHECK_MSG(header_bytes <= kMaxHeaderBytes,
                path << ": заголовок объявлен длиной " << header_bytes
                     << " байт — это не похоже на safetensors");

  std::string header(static_cast<std::size_t>(header_bytes), '\0');
  if (header_bytes != 0) {
    file.read(&header[0], static_cast<std::streamsize>(header_bytes));
    LLM_CHECK_MSG(static_cast<uint64_t>(file.gcount()) == header_bytes,
                  path << ": файл оборвался на заголовке");
  }

  // Данные — весь остаток файла.
  std::ostringstream rest;
  rest << file.rdbuf();
  const std::string data = rest.str();
  result.data_.assign(data.begin(), data.end());

  const JsonDocument document = JsonDocument::parse(header);
  const JsonValue root = document.root();
  LLM_CHECK_MSG(root.is_object(), path << ": заголовок не объект");

  for (std::size_t i = 0; i < root.size(); ++i) {
    const std::string name = root.key_at(i);
    const JsonValue value = root.value_at(i);

    if (name == "__metadata__") {
      LLM_CHECK_MSG(value.is_object(), path << ": __metadata__ не объект");
      for (std::size_t j = 0; j < value.size(); ++j) {
        const JsonValue item = value.value_at(j);
        if (item.is_string()) {
          result.metadata_.push_back(
              std::make_pair(value.key_at(j), item.text()));
        }
      }
      continue;
    }

    LLM_CHECK_MSG(value.is_object(),
                  path << ": описание тензора " << name << " не объект");

    SafeTensorEntry entry;
    entry.name = name;
    entry.dtype = parse_dtype(value.field("dtype").text(), name);

    const JsonValue shape = value.field("shape");
    LLM_CHECK_MSG(shape.is_array(),
                  path << ": форма тензора " << name << " не массив");
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      const int64_t extent = shape.at(axis).integer();
      LLM_CHECK_MSG(extent >= 0,
                    path << ": отрицательная размерность у " << name);
      entry.shape.push_back(extent);
    }

    const JsonValue offsets = value.field("data_offsets");
    LLM_CHECK_MSG(
        offsets.is_array() && offsets.size() == 2,
        path << ": data_offsets у " << name << " должны быть парой чисел");
    const int64_t begin = offsets.at(0).integer();
    const int64_t end = offsets.at(1).integer();
    LLM_CHECK_MSG(begin >= 0 && end >= begin,
                  path << ": испорченные смещения у " << name);
    entry.begin = static_cast<uint64_t>(begin);
    entry.end = static_cast<uint64_t>(end);

    // Три проверки, без которых чтение даёт связный мусор вместо отказа:
    // кусок обязан лежать внутри файла и по длине отвечать форме и типу.
    LLM_CHECK_MSG(entry.end <= result.data_.size(),
                  path << ": тензор " << name << " выходит за конец файла ("
                       << entry.end << " > " << result.data_.size() << ")");
    const uint64_t expected =
        static_cast<uint64_t>(entry.numel()) * dtype_size(entry.dtype);
    LLM_CHECK_MSG(entry.end - entry.begin == expected,
                  path << ": у тензора " << name << " кусок в "
                       << (entry.end - entry.begin) << " байт, а форма и тип "
                       << dtype_name(entry.dtype) << " требуют " << expected);

    result.entries_.push_back(entry);
  }

  // Перекрытие кусков стандарт запрещает, и проверить это дёшево. Заголовок,
  // в котором два тензора указывают на одну память, — верный признак, что файл
  // собран неправильно, и узнать об этом лучше сразу.
  std::vector<const SafeTensorEntry*> ordered;
  for (std::size_t i = 0; i < result.entries_.size(); ++i) {
    if (result.entries_[i].end > result.entries_[i].begin) {
      ordered.push_back(&result.entries_[i]);
    }
  }
  std::sort(ordered.begin(), ordered.end(),
            [](const SafeTensorEntry* a, const SafeTensorEntry* b) {
              return a->begin < b->begin;
            });
  for (std::size_t i = 1; i < ordered.size(); ++i) {
    LLM_CHECK_MSG(ordered[i]->begin >= ordered[i - 1]->end,
                  path << ": тензоры " << ordered[i - 1]->name << " и "
                       << ordered[i]->name << " перекрываются");
  }

  return result;
}

bool SafeTensors::has(const std::string& name) const {
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].name == name) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> SafeTensors::names() const {
  std::vector<std::string> result;
  result.reserve(entries_.size());
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    result.push_back(entries_[i].name);
  }
  return result;
}

const SafeTensorEntry& SafeTensors::entry(const std::string& name) const {
  for (std::size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].name == name) {
      return entries_[i];
    }
  }
  // Имя тензора — первое, что расходится между реализациями, поэтому в
  // сообщении полезно видеть, что в файле есть на самом деле. Всё перечислять
  // незачем: у большой модели это сотни строк.
  std::ostringstream available;
  for (std::size_t i = 0; i < entries_.size() && i < 8; ++i) {
    available << (i == 0 ? "" : ", ") << entries_[i].name;
  }
  if (entries_.size() > 8) {
    available << ", ... (всего " << entries_.size() << ")";
  }
  LLM_CHECK_MSG(false, "в " << path_ << " нет тензора " << name
                            << "; есть: " << available.str());
  return entries_[0];
}

Tensor SafeTensors::read(const std::string& name) const {
  const SafeTensorEntry& found = entry(name);

  Tensor out = Tensor::uninitialized(Shape(found.shape));
  const int64_t count = found.numel();
  if (count == 0) {
    // У пустого тензора и указатель на данные, и указатель в файл могут быть
    // нулевыми, а memcpy с нулевым указателем — неопределённое поведение даже
    // при нулевой длине. Санитайзер это поймал; на практике всё «работало».
    return out;
  }
  float* destination = out.data();
  const char* source = data_.data() + found.begin;

  switch (found.dtype) {
    case SafeDtype::kF32:
      // Копия, а не приведение указателя: выравнивание куска в файле нам
      // никто не обещал, а читать float по невыровненному адресу — это
      // неопределённое поведение, которое на x86 обычно работает, а потом
      // однажды нет.
      std::memcpy(destination, source,
                  static_cast<std::size_t>(count) * sizeof(float));
      break;
    case SafeDtype::kF16:
      for (int64_t i = 0; i < count; ++i) {
        uint16_t bits;
        std::memcpy(&bits, source + i * 2, sizeof(bits));
        destination[i] = f16_to_float(bits);
      }
      break;
    case SafeDtype::kBf16:
      for (int64_t i = 0; i < count; ++i) {
        uint16_t bits;
        std::memcpy(&bits, source + i * 2, sizeof(bits));
        destination[i] = bf16_to_float(bits);
      }
      break;
  }
  return out;
}

}  // namespace serialize
}  // namespace llm
