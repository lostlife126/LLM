#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "core/check.h"
#include "testing.h"

LLM_TEST(Check, PassingCheckIsSilent) {
  LLM_CHECK(true);
  LLM_CHECK_EQ(1, 1);
  LLM_CHECK_LT(1, 2);
}

LLM_TEST(Check, FailingCheckThrows) { LLM_EXPECT_THROWS(LLM_CHECK(false)); }

LLM_TEST(Check, ReportIncludesBothSides) {
  // Отчёт обязан содержать сами значения: сообщение вида "a < b" не помогает
  // отлаживать, а "3 < 2" сразу показывает причину.
  std::string message;
  try {
    LLM_CHECK_LT(3, 2);
  } catch (const llm::CheckFailure& failure) {
    message = failure.what();
  }
  LLM_CHECK(message.find("3 < 2") != std::string::npos);
  LLM_CHECK(message.find("test_check.cpp") != std::string::npos);
}

LLM_TEST(Check, CustomMessage) {
  std::string message;
  try {
    LLM_CHECK_MSG(false, "форма " << 42 << " несовместима");
  } catch (const llm::CheckFailure& failure) {
    message = failure.what();
  }
  LLM_CHECK(message.find("форма 42 несовместима") != std::string::npos);
}

LLM_TEST(Check, EvaluatesArgumentsOnce) {
  // Макрос не должен вычислять аргумент дважды: иначе LLM_CHECK_EQ(next(), 1)
  // молча съедал бы два элемента.
  int calls = 0;
  const auto bump = [&calls]() { return ++calls; };
  LLM_CHECK_EQ(bump(), 1);
  LLM_CHECK_EQ(calls, 1);
}

LLM_TEST(Check, EveryRegisteredTestHasAUniqueName) {
  // Проверка про саму оснастку. LLM_TEST заводит функцию со статической
  // связностью, поэтому два теста с одинаковым набором и именем в разных
  // файлах спокойно собираются и оба попадают в список. Оба и выполнятся, но
  // в отчёте они неразличимы: читающий увидит одно имя и решит, что тест
  // один. Так теряется копия, в которую вносили правку.
  //
  // Заодно ловится пустое имя — оно получилось бы при неудачной
  // макроподстановке.
  const std::vector<llm::testing::TestCase>& tests = llm::testing::registry();
  LLM_CHECK_GT(tests.size(), static_cast<std::size_t>(100));

  std::set<std::string> seen;
  for (std::size_t i = 0; i < tests.size(); ++i) {
    LLM_CHECK_MSG(!tests[i].suite.empty(), "у теста " << i << " пустой набор");
    LLM_CHECK_MSG(!tests[i].name.empty(), "у теста " << i << " пустое имя");
    LLM_CHECK_MSG(tests[i].fn != nullptr, "у теста " << i << " нет тела");

    const std::string full = tests[i].suite + "." + tests[i].name;
    LLM_CHECK_MSG(seen.insert(full).second,
                  "имя " << full << " зарегистрировано дважды");
  }
  LLM_CHECK_EQ(seen.size(), tests.size());
}
