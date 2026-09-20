// Разбор числовых аргументов командной строки — один на все приложения.
//
// Раньше везде стоял std::atoi (или atoll, atof), и у него нет способа
// сообщить об ошибке: на нечисловой строке он возвращает нуль, на «12abc» —
// двенадцать. Молча. Последствия не выдуманные:
//
//   * перепутанный порядок аргументов даёт не «здесь ожидалось число», а
//     сообщение про совсем другое — размер словаря 0 не проходит проверку
//     «меньше 259», и пользователь ищет ошибку в словаре;
//   * «2k» вместо 2000 в числе шагов делает прогон двухшаговым, и он
//     заканчивается за долю секунды, выглядя успешным;
//   * нечисловая температура становится нулём, то есть выбор молча делается
//     жадным.
//
// Ни один из этих случаев не падает. Все они портят результат и выглядят как
// нормальная работа — то есть это самая дорогая разновидность ошибки в
// проекте, где приложениями снимаются числа для README.
//
// Поэтому разбор строгий: строка обязана быть числом целиком, иначе
// программа останавливается и называет аргумент по имени.

#ifndef LLM_APPS_ARGS_H_
#define LLM_APPS_ARGS_H_

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace bench {

inline int64_t parse_int64(const char* text, const char* what) {
  if (text == nullptr || text[0] == '\0') {
    std::fprintf(stderr, "%s: пустое значение, ожидалось целое число\n", what);
    std::exit(1);
  }
  errno = 0;
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  // Хвост обязан быть пустым: «12abc» — это не двенадцать, это опечатка.
  if (end == text || *end != '\0') {
    std::fprintf(stderr, "%s: '%s' не целое число\n", what, text);
    std::exit(1);
  }
  if (errno == ERANGE) {
    std::fprintf(stderr, "%s: '%s' не помещается в целое\n", what, text);
    std::exit(1);
  }
  return static_cast<int64_t>(value);
}

// То же, но величина обязана быть положительной.
//
// Ноль в числе зёрен, шагов или элементов батча — не «ничего не делать», а
// опечатка, и последствия у неё того же рода, что у непонятого числа: ablate
// с нулём зёрен печатал целую таблицу из четырнадцати строк с потерями 0.0000
// и перплексией 1.0, то есть выдуманный результат, с виду измеренный.
inline int64_t parse_positive_int64(const char* text, const char* what) {
  const int64_t value = parse_int64(text, what);
  if (value <= 0) {
    std::fprintf(stderr, "%s: %lld, а нужно больше нуля\n", what,
                 static_cast<long long>(value));
    std::exit(1);
  }
  return value;
}

inline double parse_double(const char* text, const char* what) {
  if (text == nullptr || text[0] == '\0') {
    std::fprintf(stderr, "%s: пустое значение, ожидалось число\n", what);
    std::exit(1);
  }
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0') {
    std::fprintf(stderr, "%s: '%s' не число\n", what, text);
    std::exit(1);
  }
  if (errno == ERANGE) {
    std::fprintf(stderr, "%s: '%s' вне представимого диапазона\n", what, text);
    std::exit(1);
  }
  return value;
}

}  // namespace bench

#endif  // LLM_APPS_ARGS_H_
