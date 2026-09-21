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
//
// Разбор отделён от остановки, и это не украшение. Функции с exit нельзя
// проверить тестом — прогон кончился бы на первой же проверке, — а разбор
// аргументов не проверялся ничем вовсе. Тот же приём уже применён дважды:
// parse_thread_count отделена от getenv, parse_env_flag от него же.

#ifndef LLM_APPS_ARGS_H_
#define LLM_APPS_ARGS_H_

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace bench {

// Чем именно строка не подошла. Случаи различаются, потому что читающему
// сообщение они говорят разное: пустой аргумент — это чаще всего сдвинутый
// порядок, «не число» — опечатка, «вне диапазона» — верно набранное, но
// слишком большое, «не конечное» — почти всегда попытка что-то выключить
// словом «inf».
enum class ParseOutcome {
  kOk,
  kEmpty,
  kNotANumber,
  kOutOfRange,
  kNotFinite,  // только у вещественных: inf и nan
};

// Разбор без остановки программы. При неудаче *value не трогается.
inline ParseOutcome try_parse_int64(const char* text, int64_t* value) {
  if (text == nullptr || text[0] == '\0') {
    return ParseOutcome::kEmpty;
  }
  errno = 0;
  char* end = nullptr;
  // Основание задано явно: иначе «0x10» разобралось бы как шестнадцать, а
  // «010» — как восемь, и число из командной строки значило бы не то, что
  // набрано.
  const long long parsed = std::strtoll(text, &end, 10);
  // Хвост обязан быть пустым: «12abc» — это не двенадцать, это опечатка.
  if (end == text || *end != '\0') {
    return ParseOutcome::kNotANumber;
  }
  if (errno == ERANGE) {
    return ParseOutcome::kOutOfRange;
  }
  *value = static_cast<int64_t>(parsed);
  return ParseOutcome::kOk;
}

inline ParseOutcome try_parse_double(const char* text, double* value) {
  if (text == nullptr || text[0] == '\0') {
    return ParseOutcome::kEmpty;
  }
  errno = 0;
  char* end = nullptr;
  const double parsed = std::strtod(text, &end);
  if (end == text || *end != '\0') {
    return ParseOutcome::kNotANumber;
  }
  if (errno == ERANGE) {
    return ParseOutcome::kOutOfRange;
  }
  // strtod понимает «nan» и «inf» — для неё это законные числа, и хвоста
  // после них не остаётся, поэтому проверка выше их пропускает. Здесь они
  // законными не бывают ни разу: аргументами приходят температура, дропаут и
  // доли, и каждое из этих значений конечно по смыслу. Бесконечная
  // температура делит логиты в нуль, а NaN проходит мимо всякого сравнения с
  // границами — то есть мимо проверок, которые стоят дальше.
  if (!std::isfinite(parsed)) {
    return ParseOutcome::kNotFinite;
  }
  *value = parsed;
  return ParseOutcome::kOk;
}

namespace detail {

inline void fail(ParseOutcome outcome, const char* text, const char* what,
                 const char* kind) {
  switch (outcome) {
    case ParseOutcome::kEmpty:
      std::fprintf(stderr, "%s: пустое значение, ожидалось %s\n", what, kind);
      break;
    case ParseOutcome::kNotANumber:
      std::fprintf(stderr, "%s: '%s' не %s\n", what, text, kind);
      break;
    case ParseOutcome::kOutOfRange:
      std::fprintf(stderr, "%s: '%s' вне представимого диапазона\n", what,
                   text);
      break;
    case ParseOutcome::kNotFinite:
      std::fprintf(stderr, "%s: '%s' не конечное число\n", what, text);
      break;
    case ParseOutcome::kOk:
      return;
  }
  std::exit(1);
}

}  // namespace detail

inline int64_t parse_int64(const char* text, const char* what) {
  int64_t value = 0;
  const ParseOutcome outcome = try_parse_int64(text, &value);
  detail::fail(outcome, text, what, "целое число");
  return value;
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

// Лишние аргументы — отказ, а не мусор, который можно не заметить.
//
// Нижнюю границу argc проверяют не все приложения, а верхнюю не проверяло ни
// одно из шестнадцати, и это та же болезнь, что лечит строгий разбор выше:
// величина, понятая не так, даёт число, с виду неотличимое от измеренного.
//
// Случай не выдуманный. В scripts/pi_check.sh стояло «bench_model tiny 8 128»,
// где 128 задумывалось длиной окна, а bench_model читает только пресет и батч.
// Третий аргумент отбрасывался молча. Спасло совпадение: у пресета tiny окно и
// так 128, — то есть замер был верен случайно, и перестал бы быть верным от
// любой правки пресета.
//
// expected — наибольшее допустимое число аргументов ПОСЛЕ имени программы.
inline void expect_at_most(int argc, int expected, const char* usage) {
  if (argc - 1 <= expected) {
    return;
  }
  std::fprintf(stderr,
               "лишних аргументов: передано %d, принимается не больше %d\n",
               argc - 1, expected);
  if (usage != nullptr) {
    std::fprintf(stderr, "использование: %s\n", usage);
  }
  std::exit(1);
}

inline double parse_double(const char* text, const char* what) {
  double value = 0.0;
  const ParseOutcome outcome = try_parse_double(text, &value);
  detail::fail(outcome, text, what, "число");
  return value;
}

}  // namespace bench

#endif  // LLM_APPS_ARGS_H_
