#include <vector>

#include "core/span.h"
#include "testing.h"

namespace {

// Функция, принимающая только чтение: проверяет, что Span<float> неявно
// приводится к Span<const float>.
float sum(llm::Span<const float> values) {
  float total = 0.0f;
  for (std::size_t i = 0; i < values.size(); ++i) {
    total += values[i];
  }
  return total;
}

}  // namespace

LLM_TEST(Span, DefaultIsEmpty) {
  llm::Span<int> span;
  LLM_CHECK(span.empty());
  LLM_CHECK_EQ(span.size(), static_cast<std::size_t>(0));
  LLM_CHECK(span.data() == nullptr);
}

LLM_TEST(Span, WrapsVector) {
  std::vector<int> data = {1, 2, 3, 4};
  llm::Span<int> span = llm::make_span(data);

  LLM_CHECK_EQ(span.size(), static_cast<std::size_t>(4));
  LLM_CHECK_EQ(span[0], 1);
  LLM_CHECK_EQ(span[3], 4);
  LLM_CHECK_EQ(span.front(), 1);
  LLM_CHECK_EQ(span.back(), 4);
}

LLM_TEST(Span, IsAView) {
  // Запись через Span должна менять исходный вектор, а не копию.
  std::vector<int> data = {1, 2, 3};
  llm::Span<int> span = llm::make_span(data);
  span[1] = 42;
  LLM_CHECK_EQ(data[1], 42);
}

LLM_TEST(Span, Subspan) {
  std::vector<int> data = {0, 1, 2, 3, 4, 5};
  llm::Span<int> span = llm::make_span(data);

  llm::Span<int> tail = span.subspan(4);
  LLM_CHECK_EQ(tail.size(), static_cast<std::size_t>(2));
  LLM_CHECK_EQ(tail[0], 4);

  llm::Span<int> middle = span.subspan(2, 3);
  LLM_CHECK_EQ(middle.size(), static_cast<std::size_t>(3));
  LLM_CHECK_EQ(middle[0], 2);
  LLM_CHECK_EQ(middle[2], 4);

  // Пустой подвид на самом конце допустим.
  LLM_CHECK(span.subspan(6).empty());
}

LLM_TEST(Span, ConvertsToConst) {
  std::vector<float> data = {1.0f, 2.0f, 3.5f};
  llm::Span<float> mutable_span = llm::make_span(data);
  LLM_EXPECT_NEAR(sum(mutable_span), 6.5, 1e-6);
}

LLM_TEST(Span, RangeForLoop) {
  std::vector<int> data = {1, 2, 3, 4};
  int total = 0;
  for (int value : llm::make_span(data)) {
    total += value;
  }
  LLM_CHECK_EQ(total, 10);
}
