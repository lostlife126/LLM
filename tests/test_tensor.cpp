#include <cstdint>
#include <vector>

#include "core/tensor.h"
#include "testing.h"

namespace {

// Тензор со значениями 0, 1, 2, ... в порядке плотного размещения.
llm::Tensor iota(const llm::Shape& shape) {
  llm::Tensor tensor = llm::Tensor::uninitialized(shape);
  llm::Span<float> values = tensor.flat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i);
  }
  return tensor;
}

}  // namespace

LLM_TEST(Tensor, Zeros) {
  const llm::Tensor tensor = llm::Tensor::zeros(llm::Shape({2, 3}));
  LLM_CHECK(tensor.shape() == llm::Shape({2, 3}));
  LLM_CHECK_EQ(tensor.numel(), static_cast<std::int64_t>(6));
  LLM_CHECK(tensor.is_contiguous());
  for (std::size_t i = 0; i < tensor.flat().size(); ++i) {
    LLM_EXPECT_NEAR(tensor.flat()[i], 0.0, 0.0);
  }
}

LLM_TEST(Tensor, FromValuesChecksCount) {
  LLM_EXPECT_THROWS(
      llm::Tensor::from_values(llm::Shape({2, 2}), {1.0f, 2.0f, 3.0f}));
}

LLM_TEST(Tensor, Indexing) {
  const llm::Tensor tensor = iota(llm::Shape({2, 3, 4}));
  // Последняя ось меняется быстрее всех: элемент (i, j, k) лежит по адресу
  // 12*i + 4*j + k.
  LLM_EXPECT_NEAR(tensor(0, 0, 0), 0.0, 0.0);
  LLM_EXPECT_NEAR(tensor(0, 0, 3), 3.0, 0.0);
  LLM_EXPECT_NEAR(tensor(0, 1, 0), 4.0, 0.0);
  LLM_EXPECT_NEAR(tensor(1, 0, 0), 12.0, 0.0);
  LLM_EXPECT_NEAR(tensor(1, 2, 3), 23.0, 0.0);
}

LLM_TEST(Tensor, IndexingRejectsWrongRank) {
  const llm::Tensor tensor = iota(llm::Shape({2, 3}));
  LLM_EXPECT_THROWS(tensor(1));
  LLM_EXPECT_THROWS(tensor(1, 1, 1));
}

LLM_TEST(Tensor, ScalarTensor) {
  llm::Tensor scalar = llm::Tensor::zeros(llm::Shape());
  LLM_CHECK_EQ(scalar.rank(), 0);
  LLM_CHECK_EQ(scalar.numel(), static_cast<std::int64_t>(1));
  LLM_CHECK(scalar.is_contiguous());
  scalar.flat()[0] = 2.5f;
  LLM_EXPECT_NEAR(*scalar.data(), 2.5, 0.0);
}

LLM_TEST(Tensor, ReshapeSharesStorage) {
  llm::Tensor tensor = iota(llm::Shape({2, 6}));
  llm::Tensor view = tensor.reshape(llm::Shape({3, 4}));

  LLM_CHECK(view.shares_storage_with(tensor));
  LLM_EXPECT_NEAR(view(2, 3), 11.0, 0.0);
  // Запись через вид видна через исходный тензор — это один и тот же буфер.
  view(0, 0) = 99.0f;
  LLM_EXPECT_NEAR(tensor(0, 0), 99.0, 0.0);
}

LLM_TEST(Tensor, ReshapeRejectsSizeChange) {
  const llm::Tensor tensor = iota(llm::Shape({2, 3}));
  LLM_EXPECT_THROWS(tensor.reshape(llm::Shape({2, 4})));
}

LLM_TEST(Tensor, TransposeIsAView) {
  const llm::Tensor tensor = iota(llm::Shape({2, 3}));
  const llm::Tensor transposed = tensor.transpose(0, 1);

  LLM_CHECK(transposed.shape() == llm::Shape({3, 2}));
  LLM_CHECK(transposed.shares_storage_with(tensor));
  // Данные не двигались: изменились только форма и шаги.
  LLM_CHECK(!transposed.is_contiguous());
  LLM_EXPECT_NEAR(transposed(0, 1), tensor(1, 0), 0.0);
  LLM_EXPECT_NEAR(transposed(2, 0), tensor(0, 2), 0.0);
  LLM_CHECK_EQ(transposed.stride(0), static_cast<std::int64_t>(1));
  LLM_CHECK_EQ(transposed.stride(1), static_cast<std::int64_t>(3));
}

LLM_TEST(Tensor, ReshapeRejectsNonContiguous) {
  const llm::Tensor transposed = iota(llm::Shape({2, 3})).transpose(0, 1);
  LLM_EXPECT_THROWS(transposed.reshape(llm::Shape({6})));
}

LLM_TEST(Tensor, Permute) {
  const llm::Tensor tensor = iota(llm::Shape({2, 3, 4}));
  // Ровно та перестановка, которой attention превращает
  // (batch, seq, heads, head_dim) в (batch, heads, seq, head_dim).
  const llm::Tensor permuted = tensor.permute({1, 0, 2});

  LLM_CHECK(permuted.shape() == llm::Shape({3, 2, 4}));
  for (std::int64_t i = 0; i < 2; ++i) {
    for (std::int64_t j = 0; j < 3; ++j) {
      for (std::int64_t k = 0; k < 4; ++k) {
        LLM_EXPECT_NEAR(permuted(j, i, k), tensor(i, j, k), 0.0);
      }
    }
  }
}

LLM_TEST(Tensor, PermuteRejectsDuplicateAxis) {
  const llm::Tensor tensor = iota(llm::Shape({2, 3, 4}));
  LLM_EXPECT_THROWS(tensor.permute({0, 0, 1}));
  LLM_EXPECT_THROWS(tensor.permute({0, 1}));
}

LLM_TEST(Tensor, Slice) {
  const llm::Tensor tensor = iota(llm::Shape({4, 3}));
  const llm::Tensor rows = tensor.slice(0, 1, 2);

  LLM_CHECK(rows.shape() == llm::Shape({2, 3}));
  LLM_EXPECT_NEAR(rows(0, 0), 3.0, 0.0);
  LLM_EXPECT_NEAR(rows(1, 2), 8.0, 0.0);
  // Срез по строкам плотного тензора остаётся плотным.
  LLM_CHECK(rows.is_contiguous());

  const llm::Tensor cols = tensor.slice(1, 1, 2);
  LLM_CHECK(cols.shape() == llm::Shape({4, 2}));
  LLM_EXPECT_NEAR(cols(0, 0), 1.0, 0.0);
  LLM_EXPECT_NEAR(cols(3, 1), 11.0, 0.0);
  // А срез по столбцам — нет: между строками остаётся пропуск.
  LLM_CHECK(!cols.is_contiguous());
}

LLM_TEST(Tensor, SliceRejectsOutOfRange) {
  const llm::Tensor tensor = iota(llm::Shape({4, 3}));
  LLM_EXPECT_THROWS(tensor.slice(0, 3, 2));
  LLM_EXPECT_THROWS(tensor.slice(0, -1, 2));
}

LLM_TEST(Tensor, Select) {
  const llm::Tensor tensor = iota(llm::Shape({3, 2, 4}));
  const llm::Tensor batch_item = tensor.select(0, 1);

  LLM_CHECK(batch_item.shape() == llm::Shape({2, 4}));
  LLM_CHECK(batch_item.is_contiguous());
  LLM_EXPECT_NEAR(batch_item(0, 0), 8.0, 0.0);
  LLM_EXPECT_NEAR(batch_item(1, 3), 15.0, 0.0);
}

LLM_TEST(Tensor, ExpandUsesZeroStride) {
  const llm::Tensor row = iota(llm::Shape({1, 3}));
  const llm::Tensor expanded = row.expand(llm::Shape({4, 3}));

  LLM_CHECK(expanded.shape() == llm::Shape({4, 3}));
  // Шаг 0 по растянутой оси: все строки читают одну и ту же память.
  LLM_CHECK_EQ(expanded.stride(0), static_cast<std::int64_t>(0));
  LLM_CHECK(expanded.shares_storage_with(row));
  for (std::int64_t i = 0; i < 4; ++i) {
    LLM_EXPECT_NEAR(expanded(i, 2), 2.0, 0.0);
  }
}

LLM_TEST(Tensor, ExpandAddsLeadingAxes) {
  const llm::Tensor vector = iota(llm::Shape({3}));
  const llm::Tensor expanded = vector.expand(llm::Shape({2, 3}));
  LLM_CHECK_EQ(expanded.stride(0), static_cast<std::int64_t>(0));
  LLM_EXPECT_NEAR(expanded(1, 1), 1.0, 0.0);
}

LLM_TEST(Tensor, ExpandRejectsRealAxis) {
  const llm::Tensor tensor = iota(llm::Shape({2, 3}));
  LLM_EXPECT_THROWS(tensor.expand(llm::Shape({4, 3})));
}

LLM_TEST(Tensor, ContiguousMaterializesView) {
  const llm::Tensor tensor = iota(llm::Shape({2, 3}));
  const llm::Tensor transposed = tensor.transpose(0, 1);
  const llm::Tensor dense = transposed.contiguous();

  LLM_CHECK(dense.is_contiguous());
  LLM_CHECK(!dense.shares_storage_with(tensor));
  LLM_CHECK(dense.shape() == llm::Shape({3, 2}));
  // Порядок элементов в плотной копии — построчный обход вида.
  const std::vector<float> expected = {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    LLM_EXPECT_NEAR(dense.flat()[i], expected[i], 0.0);
  }
}

LLM_TEST(Tensor, ContiguousOnDenseTensorDoesNotCopy) {
  const llm::Tensor tensor = iota(llm::Shape({2, 3}));
  LLM_CHECK(tensor.contiguous().shares_storage_with(tensor));
}

LLM_TEST(Tensor, ContiguousOfExpandedCopiesRepeatedRows) {
  const llm::Tensor dense =
      iota(llm::Shape({1, 3})).expand(llm::Shape({2, 3})).contiguous();
  LLM_CHECK_EQ(dense.numel(), static_cast<std::int64_t>(6));
  const std::vector<float> expected = {0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    LLM_EXPECT_NEAR(dense.flat()[i], expected[i], 0.0);
  }
}

LLM_TEST(Tensor, FlatRequiresContiguous) {
  const llm::Tensor transposed = iota(llm::Shape({2, 3})).transpose(0, 1);
  LLM_EXPECT_THROWS(transposed.flat());
}

LLM_TEST(Tensor, FillThroughView) {
  llm::Tensor tensor = llm::Tensor::zeros(llm::Shape({3, 4}));
  // Заполнение вида должно затронуть только его элементы.
  tensor.slice(1, 1, 2).fill(7.0f);
  for (std::int64_t i = 0; i < 3; ++i) {
    LLM_EXPECT_NEAR(tensor(i, 0), 0.0, 0.0);
    LLM_EXPECT_NEAR(tensor(i, 1), 7.0, 0.0);
    LLM_EXPECT_NEAR(tensor(i, 2), 7.0, 0.0);
    LLM_EXPECT_NEAR(tensor(i, 3), 0.0, 0.0);
  }
}

LLM_TEST(Tensor, SizeOneAxisIsContiguous) {
  // Ось размера 1 не ограничивает размещение, поэтому такой вид остаётся
  // плотным, несмотря на "неправильный" шаг.
  const llm::Tensor tensor = iota(llm::Shape({4, 1, 3}));
  LLM_CHECK(tensor.is_contiguous());
  LLM_CHECK(tensor.transpose(0, 1).is_contiguous());
}

LLM_TEST(Tensor, EmptyTensor) {
  const llm::Tensor tensor = llm::Tensor::zeros(llm::Shape({0, 3}));
  LLM_CHECK(tensor.empty());
  LLM_CHECK_EQ(tensor.numel(), static_cast<std::int64_t>(0));
  LLM_CHECK(tensor.is_contiguous());
  LLM_CHECK_EQ(tensor.flat().size(), static_cast<std::size_t>(0));
}

LLM_TEST(Tensor, CloneIsIndependent) {
  llm::Tensor tensor = iota(llm::Shape({2, 2}));
  llm::Tensor copy = tensor.clone();
  copy(0, 0) = 42.0f;
  LLM_EXPECT_NEAR(tensor(0, 0), 0.0, 0.0);
  LLM_CHECK(!copy.shares_storage_with(tensor));
}

LLM_TEST(Tensor, ViewKeepsStorageAlive) {
  // Вид владеет буфером совместно с исходным тензором, поэтому переживает его.
  llm::Tensor view;
  {
    const llm::Tensor tensor = iota(llm::Shape({2, 3}));
    view = tensor.select(0, 1);
  }
  LLM_EXPECT_NEAR(view(0), 3.0, 0.0);
  LLM_EXPECT_NEAR(view(2), 5.0, 0.0);
}

LLM_TEST(Tensor, CloneOfViewIsDense) {
  // clone обязан выдавать плотный тензор, а не копию вида с теми же шагами.
  // На этом держится численная проверка градиентов: она перебирает элементы
  // входа по одному индексу в data(), а аналитический градиент раскладывает
  // по шагам — если бы clone сохранял шаги транспонированного вида, порядки
  // разошлись бы, и gradcheck сравнивал бы элементы не с теми.
  const llm::Tensor transposed = iota(llm::Shape({2, 3})).transpose(0, 1);
  LLM_CHECK(!transposed.is_contiguous());

  const llm::Tensor copy = transposed.clone();
  LLM_CHECK(copy.is_contiguous());
  LLM_CHECK(copy.shape() == llm::Shape({3, 2}));
  const std::vector<float> expected = {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    LLM_EXPECT_NEAR(copy.flat()[i], expected[i], 0.0);
  }

  // И растянутого вида тоже: нулевой шаг обязан превратиться в настоящие
  // повторы, иначе запись в копию меняла бы сразу все строки.
  llm::Tensor repeated =
      iota(llm::Shape({1, 3})).expand(llm::Shape({2, 3})).clone();
  LLM_CHECK(repeated.is_contiguous());
  repeated(0, 0) = 9.0f;
  LLM_EXPECT_NEAR(repeated(1, 0), 0.0, 0.0);
}
