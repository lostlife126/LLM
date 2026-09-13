#include <cstddef>
#include <cstdint>
#include <utility>

#include "core/storage.h"
#include "core/util.h"
#include "testing.h"

LLM_TEST(Storage, DefaultIsEmpty) {
  llm::Storage storage;
  LLM_CHECK_EQ(storage.nbytes(), static_cast<std::size_t>(0));
  LLM_CHECK(storage.data() == nullptr);
}

LLM_TEST(Storage, ZeroSizeAllocatesNothing) {
  llm::Storage storage(0);
  LLM_CHECK_EQ(storage.nbytes(), static_cast<std::size_t>(0));
  LLM_CHECK(storage.data() == nullptr);
}

LLM_TEST(Storage, IsAligned) {
  // Проверяем набор размеров, в том числе некратных выравниванию: std::align
  // должна сдвинуть указатель в любом случае.
  const std::size_t sizes[] = {1, 4, 17, 63, 64, 65, 1000, 4096};
  for (std::size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
    llm::Storage storage(sizes[i]);
    LLM_CHECK_EQ(storage.nbytes(), sizes[i]);
    LLM_CHECK_MSG(llm::is_aligned(storage.data(), llm::Storage::kAlignment),
                  "буфер на " << sizes[i] << " байт не выровнен");
  }
}

LLM_TEST(Storage, TypedAccessSpansWholeBuffer) {
  llm::Storage storage(10 * sizeof(float));
  llm::Span<float> values = storage.as<float>();
  LLM_CHECK_EQ(values.size(), static_cast<std::size_t>(10));

  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i);
  }
  // Второй вид на тот же буфер обязан видеть записанное.
  LLM_EXPECT_NEAR(storage.as<float>()[7], 7.0, 0.0);
}

LLM_TEST(Storage, TypedAccessRejectsPartialElement) {
  // 6 байт не делятся на sizeof(float): последний элемент был бы усечён.
  llm::Storage storage(6);
  LLM_EXPECT_THROWS(storage.as<float>());
}

LLM_TEST(Storage, Zero) {
  llm::Storage storage(8 * sizeof(std::int32_t));
  llm::Span<std::int32_t> values = storage.as<std::int32_t>();
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = -1;
  }
  storage.zero();
  for (std::size_t i = 0; i < values.size(); ++i) {
    LLM_CHECK_EQ(values[i], 0);
  }
}

LLM_TEST(Storage, MoveTransfersOwnership) {
  llm::Storage source(4 * sizeof(float));
  source.as<float>()[0] = 3.25f;
  const void* original = source.data();

  llm::Storage moved(std::move(source));
  LLM_CHECK(moved.data() == original);
  LLM_CHECK_EQ(moved.nbytes(), 4 * sizeof(float));
  LLM_EXPECT_NEAR(moved.as<float>()[0], 3.25, 0.0);

  // Источник должен остаться в пустом, но корректном состоянии.
  LLM_CHECK_EQ(source.nbytes(), static_cast<std::size_t>(0));
  LLM_CHECK(source.data() == nullptr);
}

LLM_TEST(Storage, MoveAssignReleasesOldBuffer) {
  llm::Storage target(1024);
  llm::Storage source(16);
  source.as<float>()[0] = 1.5f;

  target = std::move(source);
  LLM_CHECK_EQ(target.nbytes(), static_cast<std::size_t>(16));
  LLM_EXPECT_NEAR(target.as<float>()[0], 1.5, 0.0);

  // Самоприсваивание не должно освобождать собственный буфер.
  llm::Storage& alias = target;
  target = std::move(alias);
  LLM_CHECK_EQ(target.nbytes(), static_cast<std::size_t>(16));
  LLM_EXPECT_NEAR(target.as<float>()[0], 1.5, 0.0);
}
