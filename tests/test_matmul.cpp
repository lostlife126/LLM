#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/random.h"
#include "ops/matmul.h"
#include "testing.h"

namespace {

llm::Tensor random_tensor(llm::Rng* rng, const llm::Shape& shape) {
  llm::Tensor tensor = llm::Tensor::uninitialized(shape);
  llm::Span<float> values = tensor.flat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = rng->uniform(-1.0f, 1.0f);
  }
  return tensor;
}

// Независимый эталон на уровне тензоров: обращается через operator(), то есть
// через шаги, и ничего не знает ни про gemm, ни про упаковку. Если matmul
// неверно разбирает вид как транспонированную матрицу, эталон это заметит.
llm::Tensor reference_matmul(const llm::Tensor& a, const llm::Tensor& b) {
  const std::int64_t m = a.dim(-2);
  const std::int64_t k = a.dim(-1);
  const std::int64_t n = b.dim(-1);
  const std::int64_t batch_a = a.rank() == 3 ? a.dim(0) : 1;
  const std::int64_t batch_b = b.rank() == 3 ? b.dim(0) : 1;
  const std::int64_t batch = std::max(batch_a, batch_b);
  const bool batched = a.rank() == 3 || b.rank() == 3;

  llm::Tensor out = llm::Tensor::zeros(batched ? llm::Shape({batch, m, n})
                                               : llm::Shape({m, n}));

  for (std::int64_t item = 0; item < batch; ++item) {
    for (std::int64_t i = 0; i < m; ++i) {
      for (std::int64_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (std::int64_t p = 0; p < k; ++p) {
          const float a_value =
              a.rank() == 3 ? a(batch_a == 1 ? 0 : item, i, p) : a(i, p);
          const float b_value =
              b.rank() == 3 ? b(batch_b == 1 ? 0 : item, p, j) : b(p, j);
          sum += static_cast<double>(a_value) * static_cast<double>(b_value);
        }
        if (batched) {
          out(item, i, j) = static_cast<float>(sum);
        } else {
          out(i, j) = static_cast<float>(sum);
        }
      }
    }
  }
  return out;
}

void expect_close(const llm::Tensor& actual, const llm::Tensor& expected,
                  double tolerance) {
  LLM_CHECK_MSG(
      actual.shape() == expected.shape(),
      "формы не совпали: " << actual.shape() << " и " << expected.shape());
  double scale = 1.0;
  for (std::size_t i = 0; i < expected.flat().size(); ++i) {
    scale = std::max(scale, std::fabs(static_cast<double>(expected.flat()[i])));
  }
  for (std::size_t i = 0; i < expected.flat().size(); ++i) {
    const double diff = std::fabs(static_cast<double>(actual.flat()[i]) -
                                  static_cast<double>(expected.flat()[i]));
    LLM_CHECK_MSG(diff / scale < tolerance,
                  "элемент " << i << ": " << actual.flat()[i] << " вместо "
                             << expected.flat()[i]);
  }
}

}  // namespace

LLM_TEST(Matmul, TwoDimensional) {
  llm::Rng rng(11);
  const llm::Tensor a = random_tensor(&rng, llm::Shape({7, 5}));
  const llm::Tensor b = random_tensor(&rng, llm::Shape({5, 9}));
  const llm::Tensor result = llm::ops::matmul(a, b);

  LLM_CHECK(result.shape() == llm::Shape({7, 9}));
  expect_close(result, reference_matmul(a, b), 1e-5);
}

LLM_TEST(Matmul, KnownProduct) {
  const llm::Tensor a =
      llm::Tensor::from_values(llm::Shape({2, 3}), {1, 2, 3, 4, 5, 6});
  const llm::Tensor b =
      llm::Tensor::from_values(llm::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
  const llm::Tensor result = llm::ops::matmul(a, b);
  LLM_EXPECT_NEAR(result(0, 0), 22.0, 1e-5);
  LLM_EXPECT_NEAR(result(0, 1), 28.0, 1e-5);
  LLM_EXPECT_NEAR(result(1, 0), 49.0, 1e-5);
  LLM_EXPECT_NEAR(result(1, 1), 64.0, 1e-5);
}

LLM_TEST(Matmul, TransposedOperandNeedsNoCopy) {
  // Q * K^T в attention: K передаётся транспонированным видом. matmul обязан
  // распознать единичный шаг по первой оси и включить флаг транспонирования,
  // а не материализовать копию.
  llm::Rng rng(21);
  const llm::Tensor q = random_tensor(&rng, llm::Shape({6, 4}));
  const llm::Tensor k = random_tensor(&rng, llm::Shape({5, 4}));
  const llm::Tensor k_transposed = k.transpose(0, 1);

  LLM_CHECK(!k_transposed.is_contiguous());
  const llm::Tensor scores = llm::ops::matmul(q, k_transposed);
  LLM_CHECK(scores.shape() == llm::Shape({6, 5}));
  expect_close(scores, reference_matmul(q, k_transposed), 1e-5);
}

LLM_TEST(Matmul, BothOperandsTransposed) {
  llm::Rng rng(22);
  const llm::Tensor a = random_tensor(&rng, llm::Shape({4, 7}));
  const llm::Tensor b = random_tensor(&rng, llm::Shape({9, 4}));
  const llm::Tensor result =
      llm::ops::matmul(a.transpose(0, 1), b.transpose(0, 1));
  LLM_CHECK(result.shape() == llm::Shape({7, 9}));
  expect_close(result, reference_matmul(a.transpose(0, 1), b.transpose(0, 1)),
               1e-5);
}

LLM_TEST(Matmul, SlicedOperand) {
  // Срез по столбцам даёт неплотный вид с единичным шагом по последней оси —
  // это по-прежнему обычная матрица, просто с большим шагом строки.
  llm::Rng rng(23);
  const llm::Tensor wide = random_tensor(&rng, llm::Shape({6, 10}));
  const llm::Tensor a = wide.slice(1, 2, 4);
  const llm::Tensor b = random_tensor(&rng, llm::Shape({4, 3}));

  LLM_CHECK(!a.is_contiguous());
  expect_close(llm::ops::matmul(a, b), reference_matmul(a, b), 1e-5);
}

LLM_TEST(Matmul, PermutedOperandFallsBackToCopy) {
  // У вида с перемешанными осями нет единичного шага ни по одной из двух
  // осей матрицы, поэтому matmul обязан сделать плотную копию и всё равно
  // дать верный ответ.
  llm::Rng rng(24);
  // Срез по последней оси тензора (3, 4, 5) даёт матрицу (3, 4) с шагами
  // (20, 5): единичного шага нет ни по строкам, ни по столбцам, и gemm такую
  // матрицу принять не может.
  const llm::Tensor source = random_tensor(&rng, llm::Shape({3, 4, 5}));
  const llm::Tensor a = source.select(2, 2);
  const llm::Tensor b = random_tensor(&rng, llm::Shape({4, 2}));

  LLM_CHECK(a.shape() == llm::Shape({3, 4}));
  LLM_CHECK_EQ(a.stride(0), static_cast<std::int64_t>(20));
  LLM_CHECK_EQ(a.stride(1), static_cast<std::int64_t>(5));
  LLM_CHECK(!a.is_contiguous());
  expect_close(llm::ops::matmul(a, b), reference_matmul(a, b), 1e-5);
}

LLM_TEST(Matmul, BatchedBothSides) {
  llm::Rng rng(31);
  const llm::Tensor a = random_tensor(&rng, llm::Shape({3, 5, 4}));
  const llm::Tensor b = random_tensor(&rng, llm::Shape({3, 4, 6}));
  const llm::Tensor result = llm::ops::matmul(a, b);

  LLM_CHECK(result.shape() == llm::Shape({3, 5, 6}));
  expect_close(result, reference_matmul(a, b), 1e-5);
}

LLM_TEST(Matmul, BatchedActivationsWithSharedWeight) {
  // Самый частый случай в модели: (batch, tokens, d) на веса (d, h).
  llm::Rng rng(32);
  const llm::Tensor activations = random_tensor(&rng, llm::Shape({4, 7, 5}));
  const llm::Tensor weight = random_tensor(&rng, llm::Shape({5, 3}));
  const llm::Tensor result = llm::ops::matmul(activations, weight);

  LLM_CHECK(result.shape() == llm::Shape({4, 7, 3}));
  expect_close(result, reference_matmul(activations, weight), 1e-5);
}

LLM_TEST(Matmul, BatchOfOneIsBroadcast) {
  llm::Rng rng(33);
  const llm::Tensor a = random_tensor(&rng, llm::Shape({1, 5, 4}));
  const llm::Tensor b = random_tensor(&rng, llm::Shape({3, 4, 2}));
  const llm::Tensor result = llm::ops::matmul(a, b);

  LLM_CHECK(result.shape() == llm::Shape({3, 5, 2}));
  expect_close(result, reference_matmul(a, b), 1e-5);
}

LLM_TEST(Matmul, IntoAccumulates) {
  const llm::Tensor a =
      llm::Tensor::from_values(llm::Shape({2, 2}), {1, 0, 0, 1});
  const llm::Tensor b =
      llm::Tensor::from_values(llm::Shape({2, 2}), {1, 2, 3, 4});
  llm::Tensor out = llm::Tensor::full(llm::Shape({2, 2}), 10.0f);

  // out = 2 * A * B + 1 * out
  llm::ops::matmul_into(a, b, 2.0f, 1.0f, &out);
  LLM_EXPECT_NEAR(out(0, 0), 12.0, 1e-5);
  LLM_EXPECT_NEAR(out(0, 1), 14.0, 1e-5);
  LLM_EXPECT_NEAR(out(1, 0), 16.0, 1e-5);
  LLM_EXPECT_NEAR(out(1, 1), 18.0, 1e-5);
}

LLM_TEST(Matmul, RejectsIncompatibleShapes) {
  const llm::Tensor a = llm::Tensor::zeros(llm::Shape({2, 3}));
  const llm::Tensor b = llm::Tensor::zeros(llm::Shape({4, 5}));
  LLM_EXPECT_THROWS(llm::ops::matmul(a, b));

  const llm::Tensor batch_a = llm::Tensor::zeros(llm::Shape({2, 2, 3}));
  const llm::Tensor batch_b = llm::Tensor::zeros(llm::Shape({3, 3, 4}));
  LLM_EXPECT_THROWS(llm::ops::matmul(batch_a, batch_b));

  const llm::Tensor rank4 = llm::Tensor::zeros(llm::Shape({2, 2, 3, 4}));
  LLM_EXPECT_THROWS(llm::ops::matmul(rank4, b));
}

LLM_TEST(Matmul, IntoRejectsWrongOutputShape) {
  const llm::Tensor a = llm::Tensor::zeros(llm::Shape({2, 3}));
  const llm::Tensor b = llm::Tensor::zeros(llm::Shape({3, 4}));
  llm::Tensor wrong = llm::Tensor::zeros(llm::Shape({2, 5}));
  LLM_EXPECT_THROWS(llm::ops::matmul_into(a, b, 1.0f, 0.0f, &wrong));
}
