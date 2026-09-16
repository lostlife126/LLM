// Редукции по строке с фиксированным числом накопителей.
//
// Зачем не один накопитель. Сложение с плавающей точкой не ассоциативно,
// поэтому компилятор не имеет права раскладывать `sum += x[i]` по дорожкам
// сам — и не раскладывает. Остаётся цепочка, в которой каждое сложение ждёт
// предыдущего: на double это около четырёх тактов на элемент при стоимости
// самого сложения в один. Строка softmax длиной 128 уходит на ожидание, а не
// на счёт, и в профиле это видно: тело softmax без экспоненты стоило дороже
// самой экспоненты.
//
// Зачем не «столько накопителей, сколько дорожек у вектора». Число здесь —
// константа kRowParts, одинаковая на всех машинах. Если бы его задавал набор
// инструкций, AVX2 и AVX-512 складывали бы в разном порядке и давали бы
// разные последние разряды — ровно то, от чего проект отказался. Порядок
// суммирования задан исходником, а не железом, поэтому результат остаётся
// одинаковым везде. По той же причине восьмёрка не подбирается под размер
// строки: длина строки у машин одинаковая, а вот соблазн «взять побольше на
// широком векторе» сломал бы инвариант.
//
// Хвост строки, не кратный восьмёрке, раскладывается по тем же накопителям
// с начала — это тоже часть зафиксированного порядка.

#ifndef LLM_OPS_ROW_REDUCE_H_
#define LLM_OPS_ROW_REDUCE_H_

#include <cstdint>
#include <limits>

namespace llm {
namespace ops {

constexpr int kRowParts = 8;

// Свёртка восьми накопителей в одно число. Порядок фиксирован.
inline double combine_parts(const double* part) {
  double total = 0.0;
  for (int j = 0; j < kRowParts; ++j) {
    total += part[j];
  }
  return total;
}

inline double row_sum(const float* row, int64_t width) {
  double part[kRowParts] = {};
  int64_t i = 0;
  for (; i + kRowParts <= width; i += kRowParts) {
    for (int j = 0; j < kRowParts; ++j) {
      part[j] += static_cast<double>(row[i + j]);
    }
  }
  for (int j = 0; i < width; ++i, ++j) {
    part[j] += static_cast<double>(row[i]);
  }
  return combine_parts(part);
}

inline double row_sum_squares(const float* row, int64_t width) {
  double part[kRowParts] = {};
  int64_t i = 0;
  for (; i + kRowParts <= width; i += kRowParts) {
    for (int j = 0; j < kRowParts; ++j) {
      const double value = static_cast<double>(row[i + j]);
      part[j] += value * value;
    }
  }
  for (int j = 0; i < width; ++i, ++j) {
    const double value = static_cast<double>(row[i]);
    part[j] += value * value;
  }
  return combine_parts(part);
}

inline double row_dot(const float* a, const float* b, int64_t width) {
  double part[kRowParts] = {};
  int64_t i = 0;
  for (; i + kRowParts <= width; i += kRowParts) {
    for (int j = 0; j < kRowParts; ++j) {
      part[j] += static_cast<double>(a[i + j]) * b[i + j];
    }
  }
  for (int j = 0; i < width; ++i, ++j) {
    part[j] += static_cast<double>(a[i]) * b[i];
  }
  return combine_parts(part);
}

// Скалярное произведение трёх строк. Встречается в обратном проходе RMSNorm:
// там суммируется произведение градиента, веса и входа.
inline double row_dot3(const float* a, const float* b, const float* c,
                       int64_t width) {
  double part[kRowParts] = {};
  int64_t i = 0;
  for (; i + kRowParts <= width; i += kRowParts) {
    for (int j = 0; j < kRowParts; ++j) {
      part[j] += static_cast<double>(a[i + j]) * b[i + j] * c[i + j];
    }
  }
  for (int j = 0; i < width; ++i, ++j) {
    part[j] += static_cast<double>(a[i]) * b[i] * c[i];
  }
  return combine_parts(part);
}

// Максимум ассоциативен и точен, поэтому порядок здесь ни на что не влияет —
// восемь накопителей взяты только чтобы порвать цепочку сравнений.
inline float row_max(const float* row, int64_t width) {
  float part[kRowParts];
  for (int j = 0; j < kRowParts; ++j) {
    part[j] = -std::numeric_limits<float>::infinity();
  }
  int64_t i = 0;
  for (; i + kRowParts <= width; i += kRowParts) {
    for (int j = 0; j < kRowParts; ++j) {
      if (row[i + j] > part[j]) {
        part[j] = row[i + j];
      }
    }
  }
  for (int j = 0; i < width; ++i, ++j) {
    if (row[i] > part[j]) {
      part[j] = row[i];
    }
  }
  float maximum = part[0];
  for (int j = 1; j < kRowParts; ++j) {
    if (part[j] > maximum) {
      maximum = part[j];
    }
  }
  return maximum;
}

}  // namespace ops
}  // namespace llm

#endif  // LLM_OPS_ROW_REDUCE_H_
