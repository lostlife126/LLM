#include <string>

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
