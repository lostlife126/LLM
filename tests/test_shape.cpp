#include <cstdint>
#include <vector>

#include "core/shape.h"
#include "testing.h"

LLM_TEST(Shape, Basics) {
  const llm::Shape shape{2, 3, 4};
  LLM_CHECK_EQ(shape.rank(), 3);
  LLM_CHECK_EQ(shape.dim(0), static_cast<std::int64_t>(2));
  LLM_CHECK_EQ(shape.dim(2), static_cast<std::int64_t>(4));
  LLM_CHECK_EQ(shape.numel(), static_cast<std::int64_t>(24));
  LLM_CHECK(!shape.is_scalar());
}

LLM_TEST(Shape, ScalarHasOneElement) {
  // Тензор ранга 0 — это одно число. Таким будет значение функции потерь,
  // от которого пойдёт обратный проход.
  const llm::Shape shape;
  LLM_CHECK_EQ(shape.rank(), 0);
  LLM_CHECK(shape.is_scalar());
  LLM_CHECK_EQ(shape.numel(), static_cast<std::int64_t>(1));
  LLM_CHECK(llm::contiguous_strides(shape).empty());
}

LLM_TEST(Shape, EmptyAxisGivesNoElements) {
  const llm::Shape shape{2, 0, 4};
  LLM_CHECK_EQ(shape.numel(), static_cast<std::int64_t>(0));
}

LLM_TEST(Shape, NegativeAxis) {
  const llm::Shape shape{2, 3, 4};
  LLM_CHECK_EQ(shape.dim(-1), static_cast<std::int64_t>(4));
  LLM_CHECK_EQ(shape.dim(-3), static_cast<std::int64_t>(2));
  LLM_CHECK_EQ(shape.normalize_axis(-2), 1);
  LLM_EXPECT_THROWS(shape.dim(3));
  LLM_EXPECT_THROWS(shape.dim(-4));
}

LLM_TEST(Shape, Equality) {
  LLM_CHECK(llm::Shape({2, 3}) == llm::Shape({2, 3}));
  LLM_CHECK(llm::Shape({2, 3}) != llm::Shape({3, 2}));
  // Ранг — часть формы: (2, 3) и (1, 2, 3) описывают разные тензоры.
  LLM_CHECK(llm::Shape({2, 3}) != llm::Shape({1, 2, 3}));
}

LLM_TEST(Shape, ContiguousStrides) {
  const std::vector<std::int64_t> strides =
      llm::contiguous_strides(llm::Shape({2, 3, 4}));
  LLM_CHECK_EQ(strides.size(), static_cast<std::size_t>(3));
  LLM_CHECK_EQ(strides[0], static_cast<std::int64_t>(12));
  LLM_CHECK_EQ(strides[1], static_cast<std::int64_t>(4));
  LLM_CHECK_EQ(strides[2], static_cast<std::int64_t>(1));
}

LLM_TEST(Shape, BroadcastAlignsFromRight) {
  LLM_CHECK(llm::broadcast_shapes(llm::Shape({3, 1}), llm::Shape({1, 4})) ==
            llm::Shape({3, 4}));
  // Отсутствующие слева оси считаются равными 1.
  LLM_CHECK(llm::broadcast_shapes(llm::Shape({5, 1, 4}), llm::Shape({3, 4})) ==
            llm::Shape({5, 3, 4}));
  LLM_CHECK(llm::broadcast_shapes(llm::Shape({2, 3}), llm::Shape({2, 3})) ==
            llm::Shape({2, 3}));
  // Скаляр совместим с любой формой.
  LLM_CHECK(llm::broadcast_shapes(llm::Shape(), llm::Shape({2, 3})) ==
            llm::Shape({2, 3}));
}

LLM_TEST(Shape, BroadcastRejectsMismatch) {
  LLM_CHECK(
      !llm::try_broadcast(llm::Shape({3, 4}), llm::Shape({2, 4}), nullptr));
  LLM_CHECK(!llm::try_broadcast(llm::Shape({3}), llm::Shape({4}), nullptr));
  LLM_EXPECT_THROWS(
      llm::broadcast_shapes(llm::Shape({3, 4}), llm::Shape({2, 4})));
}

LLM_TEST(Shape, ToString) {
  LLM_CHECK(llm::Shape({2, 3}).to_string() == "(2, 3)");
  LLM_CHECK(llm::Shape().to_string() == "()");
}

LLM_TEST(Shape, RejectsNegativeDim) { LLM_EXPECT_THROWS(llm::Shape({2, -1})); }
