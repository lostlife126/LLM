#include <cstdint>
#include <vector>

#include "core/shape.h"
#include "core/tensor.h"
#include "ops/elementwise.h"
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
  const llm::Dims strides = llm::contiguous_strides(llm::Shape({2, 3, 4}));
  LLM_CHECK_EQ(strides.size(), 3);
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

LLM_TEST(Shape, RejectsTooHighRank) {
  // Предел ранга — не украшение: формы приходят из чужих файлов. Заголовок
  // safetensors задаёт форму тензора списком любой длины, и без этой проверки
  // девятая ось записалась бы за конец массива внутри Dims.
  LLM_CHECK_EQ(llm::Dims::kMaxRank, 8);
  const llm::Shape widest{1, 1, 1, 1, 1, 1, 1, 1};
  LLM_CHECK_EQ(widest.rank(), 8);
  LLM_EXPECT_THROWS(llm::Shape({1, 1, 1, 1, 1, 1, 1, 1, 1}));
}

LLM_TEST(Shape, EmptyAxisWinsOverStretchedOne) {
  // Совместимость 0 и 1 — не случай «взять больший». Единица растягивается до
  // соседа, а сосед пуст, значит пуст и результат: ровно так считает numpy.
  // Если бы бралось max, получалась бы форма с элементом, которого нет ни в
  // одном из аргументов, и растягивание пустой оси падало бы с невнятным
  // «ось 0 размера 0 нельзя растянуть до 1».
  LLM_CHECK(llm::broadcast_shapes(llm::Shape({0}), llm::Shape({1})) ==
            llm::Shape({0}));
  LLM_CHECK(llm::broadcast_shapes(llm::Shape({1}), llm::Shape({0})) ==
            llm::Shape({0}));
  LLM_CHECK(llm::broadcast_shapes(llm::Shape({2, 0, 3}),
                                  llm::Shape({1, 1, 3})) ==
            llm::Shape({2, 0, 3}));
  // Недостающая слева ось тоже считается единицей и тоже растягивается в ноль.
  LLM_CHECK(llm::broadcast_shapes(llm::Shape({0, 3}), llm::Shape({3})) ==
            llm::Shape({0, 3}));
  // Ноль с размером больше единицы по-прежнему несовместим.
  LLM_CHECK(!llm::try_broadcast(llm::Shape({0}), llm::Shape({2}), nullptr));
}

LLM_TEST(Shape, ElementwiseOnEmptyOperandStaysEmpty) {
  // Смысл предыдущего теста, доведённый до операции: сложение пустого тензора
  // с одной строкой даёт пустой тензор, а не исключение.
  const llm::Tensor empty = llm::Tensor::zeros(llm::Shape({0, 3}));
  const llm::Tensor row = llm::Tensor::zeros(llm::Shape({1, 3}));
  const llm::Tensor sum = llm::ops::add(empty, row);
  LLM_CHECK(sum.shape() == llm::Shape({0, 3}));
  LLM_CHECK_EQ(sum.numel(), static_cast<int64_t>(0));
}
