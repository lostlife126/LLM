// Разбор аргументов командной строки и чтение файла целиком.
//
// До сих пор это не проверялось ничем, хотя через разбор проходят все числа,
// которыми задаются прогоны: число шагов, размер батча, температура, дропаут,
// длина контекста. Ошибка здесь не падает — она меняет то, ЧТО измерено, и
// прогон при этом выглядит нормальным.
//
// Проверяется вариант без остановки программы. Тот, что с остановкой, тестом
// не проверить: прогон кончился бы на первой же проверке. Поэтому разбор и
// отделён от сообщения с exit — как parse_thread_count отделена от getenv.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#include "args.h"
#include "read_file.h"
#include "testing.h"

namespace {

using bench::ParseOutcome;

int64_t parsed_int(const char* text) {
  int64_t value = 0;
  LLM_CHECK_MSG(bench::try_parse_int64(text, &value) == ParseOutcome::kOk,
                "'" << text << "' не разобралось, а должно было");
  return value;
}

double parsed_double(const char* text) {
  double value = 0.0;
  LLM_CHECK_MSG(bench::try_parse_double(text, &value) == ParseOutcome::kOk,
                "'" << text << "' не разобралось, а должно было");
  return value;
}

void expect_int_fails(const char* text, ParseOutcome expected) {
  // Сторожевое значение: при неудаче разбор не вправе трогать результат,
  // иначе вызывающий получил бы полуразобранное число.
  int64_t value = -12345;
  const ParseOutcome outcome = bench::try_parse_int64(text, &value);
  LLM_CHECK_MSG(outcome == expected,
                "'" << (text == nullptr ? "(nullptr)" : text) << "': исход "
                    << static_cast<int>(outcome) << " вместо "
                    << static_cast<int>(expected));
  LLM_CHECK_MSG(value == -12345,
                "при неудаче разбор изменил значение на " << value);
}

void expect_double_fails(const char* text, ParseOutcome expected) {
  double value = -12345.0;
  const ParseOutcome outcome = bench::try_parse_double(text, &value);
  LLM_CHECK_MSG(outcome == expected,
                "'" << (text == nullptr ? "(nullptr)" : text) << "': исход "
                    << static_cast<int>(outcome) << " вместо "
                    << static_cast<int>(expected));
  LLM_CHECK_MSG(value == -12345.0,
                "при неудаче разбор изменил значение на " << value);
}

}  // namespace

LLM_TEST(Args, IntegersParseWholeStringOrNotAtAll) {
  LLM_CHECK_EQ(parsed_int("0"), static_cast<int64_t>(0));
  LLM_CHECK_EQ(parsed_int("12"), static_cast<int64_t>(12));
  LLM_CHECK_EQ(parsed_int("-5"), static_cast<int64_t>(-5));
  LLM_CHECK_EQ(parsed_int("+7"), static_cast<int64_t>(7));

  // Ровно тот случай, ради которого разбор и переписан с atoi: «12abc» не
  // двенадцать, а опечатка, и «2k» вместо 2000 не двойка.
  expect_int_fails("12abc", ParseOutcome::kNotANumber);
  expect_int_fails("2k", ParseOutcome::kNotANumber);
  expect_int_fails("abc", ParseOutcome::kNotANumber);
  // Хвостовой пробел — тоже хвост.
  expect_int_fails("12 ", ParseOutcome::kNotANumber);
  // Вещественное там, где ждали целое: молча обрезать до 1 нельзя.
  expect_int_fails("1.5", ParseOutcome::kNotANumber);
  expect_int_fails("1e3", ParseOutcome::kNotANumber);

  expect_int_fails(nullptr, ParseOutcome::kEmpty);
  expect_int_fails("", ParseOutcome::kEmpty);
  // Один знак без цифр — не число.
  expect_int_fails("-", ParseOutcome::kNotANumber);
}

LLM_TEST(Args, IntegerBaseIsAlwaysTen) {
  // Основание задано явно, и это видно снаружи. Без него strtoll разобрала бы
  // «010» как восьмеричное, то есть как восемь: число шагов 010 стало бы
  // восемью вместо десяти, и прогон кончился бы раньше, чем задумано.
  LLM_CHECK_EQ(parsed_int("010"), static_cast<int64_t>(10));
  LLM_CHECK_EQ(parsed_int("0009"), static_cast<int64_t>(9));
  // А шестнадцатеричная запись — не число, а опечатка: хвост «x10» остаётся.
  expect_int_fails("0x10", ParseOutcome::kNotANumber);
}

LLM_TEST(Args, IntegerEdgesOfTheRange) {
  LLM_CHECK_EQ(parsed_int("9223372036854775807"),
               static_cast<int64_t>(9223372036854775807LL));
  LLM_CHECK_EQ(parsed_int("-9223372036854775808"),
               static_cast<int64_t>(-9223372036854775807LL - 1));
  // На единицу дальше — отказ, а не тихо обрезанный предел.
  expect_int_fails("9223372036854775808", ParseOutcome::kOutOfRange);
  expect_int_fails("-9223372036854775809", ParseOutcome::kOutOfRange);
  expect_int_fails("99999999999999999999999", ParseOutcome::kOutOfRange);
}

LLM_TEST(Args, DoublesParseWholeStringOrNotAtAll) {
  LLM_EXPECT_NEAR(parsed_double("0.5"), 0.5, 0.0);
  LLM_EXPECT_NEAR(parsed_double("-1.25"), -1.25, 0.0);
  LLM_EXPECT_NEAR(parsed_double("1e3"), 1000.0, 0.0);
  LLM_EXPECT_NEAR(parsed_double("0"), 0.0, 0.0);

  expect_double_fails("0.5x", ParseOutcome::kNotANumber);
  expect_double_fails("abc", ParseOutcome::kNotANumber);
  expect_double_fails("0.5 ", ParseOutcome::kNotANumber);
  expect_double_fails(nullptr, ParseOutcome::kEmpty);
  expect_double_fails("", ParseOutcome::kEmpty);
}

LLM_TEST(Args, DoublesRefuseInfinityAndNotANumber) {
  // Главная находка этой проверки. strtod понимает «nan» и «inf» как законные
  // числа, и хвоста после них не остаётся — то есть проверка «строка целиком»
  // их пропускает.
  //
  // Дальше они расходятся по программе бесшумно. NaN ложен в любом сравнении,
  // поэтому проходит мимо всех границ, которые стоят ниже: температура NaN не
  // меньше нуля и не больше нуля, дропаут NaN не в [0, 1) и не вне его.
  // Бесконечная температура делит все логиты в нуль. Ни то, ни другое не
  // падает.
  const char* const refused[] = {"nan", "NaN",  "-nan",     "inf",
                                 "INF", "-inf", "infinity", "Infinity"};
  for (std::size_t i = 0; i < sizeof(refused) / sizeof(refused[0]); ++i) {
    expect_double_fails(refused[i], ParseOutcome::kNotFinite);
  }

  // Переполнение при этом остаётся переполнением, а не «не конечным»: порядок
  // проверок в разборе именно такой, и читающему сообщение это говорит
  // разное — «слишком большое число» против «здесь написано слово».
  double value = -12345.0;
  const ParseOutcome outcome = bench::try_parse_double("1e400", &value);
  LLM_CHECK_MSG(outcome != ParseOutcome::kOk,
                "1e400 разобралось как обычное число");
  LLM_CHECK_MSG(value == -12345.0, "при неудаче значение изменилось");
}

LLM_TEST(Args, ReadFileReturnsExactlyWhatWasWritten) {
  const std::string path = "test_args_read.bin";
  // Байты нарочно не текстовые: чтение обязано идти двоичным, иначе на
  // переводах строки содержимое поедет.
  std::string source;
  for (int i = 0; i < 1000; ++i) {
    source.push_back(static_cast<char>(i % 256));
  }
  {
    std::ofstream file(path.c_str(), std::ios::binary);
    file.write(source.data(), static_cast<std::streamsize>(source.size()));
  }
  LLM_CHECK(bench::read_file(path) == source);
  std::remove(path.c_str());

  // Пустой файл — не ошибка, а пустая строка.
  const std::string empty_path = "test_args_empty.bin";
  { std::ofstream file(empty_path.c_str(), std::ios::binary); }
  LLM_CHECK(bench::read_file(empty_path).empty());
  std::remove(empty_path.c_str());
}
