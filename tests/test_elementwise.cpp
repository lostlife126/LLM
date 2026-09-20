#include <cmath>
#include <cstdint>
#include <vector>

#include "ops/elementwise.h"
#include "ops/reduce.h"
#include "testing.h"

namespace {

llm::Tensor values(const llm::Shape& shape, const std::vector<float>& data) {
  return llm::Tensor::from_values(shape, data);
}

}  // namespace

LLM_TEST(Elementwise, AddSameShape) {
  const llm::Tensor result =
      llm::ops::add(values(llm::Shape({2, 2}), {1, 2, 3, 4}),
                    values(llm::Shape({2, 2}), {10, 20, 30, 40}));
  LLM_EXPECT_NEAR(result(0, 0), 11.0, 1e-6);
  LLM_EXPECT_NEAR(result(1, 1), 44.0, 1e-6);
}

LLM_TEST(Elementwise, BroadcastRow) {
  // Строка (1, 3) растягивается на все строки (2, 3).
  const llm::Tensor result =
      llm::ops::add(values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6}),
                    values(llm::Shape({1, 3}), {10, 20, 30}));
  LLM_CHECK(result.shape() == llm::Shape({2, 3}));
  LLM_EXPECT_NEAR(result(0, 0), 11.0, 1e-6);
  LLM_EXPECT_NEAR(result(1, 2), 36.0, 1e-6);
}

LLM_TEST(Elementwise, BroadcastColumn) {
  const llm::Tensor result =
      llm::ops::mul(values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6}),
                    values(llm::Shape({2, 1}), {10, 100}));
  LLM_EXPECT_NEAR(result(0, 2), 30.0, 1e-6);
  LLM_EXPECT_NEAR(result(1, 0), 400.0, 1e-6);
}

LLM_TEST(Elementwise, BroadcastBothSides) {
  // (3, 1) и (1, 4) вместе дают (3, 4) — растягиваются оба аргумента.
  const llm::Tensor result =
      llm::ops::mul(values(llm::Shape({3, 1}), {1, 2, 3}),
                    values(llm::Shape({1, 4}), {1, 10, 100, 1000}));
  LLM_CHECK(result.shape() == llm::Shape({3, 4}));
  LLM_EXPECT_NEAR(result(2, 3), 3000.0, 1e-6);
}

LLM_TEST(Elementwise, WorksOnNonContiguousInput) {
  // Транспонированный вид: если операция считает память плотной, результат
  // получится переставленным.
  const llm::Tensor source = values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
  const llm::Tensor result = llm::ops::mul_scalar(source.transpose(0, 1), 2.0f);
  LLM_CHECK(result.shape() == llm::Shape({3, 2}));
  LLM_EXPECT_NEAR(result(0, 0), 2.0, 1e-6);
  LLM_EXPECT_NEAR(result(0, 1), 8.0, 1e-6);
  LLM_EXPECT_NEAR(result(2, 1), 12.0, 1e-6);
}

LLM_TEST(Elementwise, UnaryFunctions) {
  const llm::Tensor source = values(llm::Shape({3}), {0.0f, 1.0f, 4.0f});
  LLM_EXPECT_NEAR(llm::ops::exp(source)(1), std::exp(1.0f), 1e-5);
  LLM_EXPECT_NEAR(llm::ops::sqrt(source)(2), 2.0, 1e-6);
  LLM_EXPECT_NEAR(llm::ops::neg(source)(2), -4.0, 1e-6);
  LLM_EXPECT_NEAR(llm::ops::log(values(llm::Shape({1}), {1.0f}))(0), 0.0, 1e-6);
}

LLM_TEST(Elementwise, RejectsIncompatibleShapes) {
  LLM_EXPECT_THROWS(
      llm::ops::add(values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6}),
                    values(llm::Shape({2, 2}), {1, 2, 3, 4})));
}

LLM_TEST(Elementwise, AddIntoWritesThroughView) {
  llm::Tensor target = llm::Tensor::zeros(llm::Shape({2, 4}));
  llm::Tensor window = target.slice(1, 1, 2);
  llm::ops::add_into(values(llm::Shape({2, 2}), {1, 2, 3, 4}), &window);

  LLM_EXPECT_NEAR(target(0, 0), 0.0, 1e-6);
  LLM_EXPECT_NEAR(target(0, 1), 1.0, 1e-6);
  LLM_EXPECT_NEAR(target(0, 2), 2.0, 1e-6);
  LLM_EXPECT_NEAR(target(0, 3), 0.0, 1e-6);
  LLM_EXPECT_NEAR(target(1, 2), 4.0, 1e-6);
}

LLM_TEST(Elementwise, SumOverAxis) {
  const llm::Tensor source = values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6});

  const llm::Tensor rows = llm::ops::sum(source, {1}, false);
  LLM_CHECK(rows.shape() == llm::Shape({2}));
  LLM_EXPECT_NEAR(rows(0), 6.0, 1e-6);
  LLM_EXPECT_NEAR(rows(1), 15.0, 1e-6);

  const llm::Tensor columns = llm::ops::sum(source, {0}, true);
  LLM_CHECK(columns.shape() == llm::Shape({1, 3}));
  LLM_EXPECT_NEAR(columns(0, 0), 5.0, 1e-6);
  LLM_EXPECT_NEAR(columns(0, 2), 9.0, 1e-6);
}

LLM_TEST(Elementwise, SumAllGivesScalar) {
  const llm::Tensor total =
      llm::ops::sum_all(values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6}));
  LLM_CHECK_EQ(total.rank(), 0);
  LLM_EXPECT_NEAR(*total.data(), 21.0, 1e-6);
}

LLM_TEST(Elementwise, Mean) {
  const llm::Tensor source = values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
  LLM_EXPECT_NEAR(*llm::ops::mean_all(source).data(), 3.5, 1e-6);
  LLM_EXPECT_NEAR(llm::ops::mean(source, {1}, false)(0), 2.0, 1e-6);
}

// Повторённая ось — отказ, а не тихо другой ответ.
//
// Сумме повтор безразличен: набор осей внутри разворачивается в множество, и
// второе упоминание ничего не добавляет. А mean делит на ПРОИЗВЕДЕНИЕ размеров
// перечисленных осей, и повтор возводит делитель в квадрат: mean по {1, 1} у
// матрицы 2x3 выдавал бы треть настоящего среднего. Ни падения, ни признака.
LLM_TEST(Elementwise, ReductionRefusesARepeatedAxis) {
  const llm::Tensor source = values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6});

  LLM_EXPECT_THROWS(llm::ops::sum(source, {1, 1}, false));
  LLM_EXPECT_THROWS(llm::ops::mean(source, {1, 1}, false));
  // Та же ось, записанная с конца: проверка обязана смотреть на нормализованный
  // номер, иначе -1 и 1 у матрицы разошлись бы.
  LLM_EXPECT_THROWS(llm::ops::sum(source, {1, -1}, false));
  LLM_EXPECT_THROWS(llm::ops::sum(source, {0, 1, 0}, true));

  // А разные оси по-прежнему складываются.
  const llm::Tensor total = llm::ops::sum(source, {0, 1}, false);
  LLM_EXPECT_NEAR(*total.data(), 21.0, 1e-6);
}

LLM_TEST(Elementwise, SumOverNonContiguous) {
  const llm::Tensor source = values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
  const llm::Tensor sums = llm::ops::sum(source.transpose(0, 1), {1}, false);
  LLM_CHECK(sums.shape() == llm::Shape({3}));
  LLM_EXPECT_NEAR(sums(0), 5.0, 1e-6);
  LLM_EXPECT_NEAR(sums(2), 9.0, 1e-6);
}

LLM_TEST(Elementwise, ReduceToShapeSumsBroadcastAxes) {
  // Обратный проход растяжения: градиент формы (2, 3) приводится к (1, 3)
  // суммированием по строкам.
  const llm::Tensor grad = values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
  const llm::Tensor reduced =
      llm::ops::reduce_to_shape(grad, llm::Shape({1, 3}));
  LLM_CHECK(reduced.shape() == llm::Shape({1, 3}));
  LLM_EXPECT_NEAR(reduced(0, 0), 5.0, 1e-6);
  LLM_EXPECT_NEAR(reduced(0, 2), 9.0, 1e-6);
}

LLM_TEST(Elementwise, ReduceToShapeDropsLeadingAxes) {
  const llm::Tensor grad =
      values(llm::Shape({2, 2, 2}), {1, 2, 3, 4, 5, 6, 7, 8});
  const llm::Tensor reduced =
      llm::ops::reduce_to_shape(grad, llm::Shape({2, 2}));
  LLM_CHECK(reduced.shape() == llm::Shape({2, 2}));
  LLM_EXPECT_NEAR(reduced(0, 0), 6.0, 1e-6);
  LLM_EXPECT_NEAR(reduced(1, 1), 12.0, 1e-6);
}

LLM_TEST(Elementwise, ReduceToShapeIsIdentityWhenShapesMatch) {
  const llm::Tensor grad = values(llm::Shape({2, 2}), {1, 2, 3, 4});
  const llm::Tensor reduced =
      llm::ops::reduce_to_shape(grad, llm::Shape({2, 2}));
  LLM_EXPECT_NEAR(reduced(1, 1), 4.0, 1e-6);
}
