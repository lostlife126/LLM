// Обход кусками проверяется сравнением с поэлементным одометром.
//
// BlockWalkN схлопывает хвост формы, на котором шаги всех наборов идут ровно,
// и дальше ходит кусками. Условие схлопывания — единственное нетривиальное
// место во всём заголовке, и ошибка в нём не падает, а молча читает не те
// ячейки. Поэтому проверяется не само условие, а следствие: последовательность
// смещений обязана совпасть с той, что даёт одометр по одному элементу.

#include <cstdint>
#include <vector>

#include "core/dims.h"
#include "core/iterate.h"
#include "core/shape.h"
#include "testing.h"

namespace {

std::vector<int64_t> by_odometer(const llm::Shape& shape,
                                 const llm::Dims& strides) {
  std::vector<int64_t> out;
  llm::for_each_offset(shape, strides,
                       [&out](int64_t offset) { out.push_back(offset); });
  return out;
}

std::vector<int64_t> by_blocks(const llm::Shape& shape,
                               const llm::Dims& strides) {
  std::vector<int64_t> out;
  const llm::BlockWalkN<1> walk = llm::block_walk(shape, strides);
  if (shape.numel() == 0) {
    return out;
  }
  llm::BlockWalkN<1>::Cursor cursor = walk.at(0);
  for (int64_t block = 0; block < walk.blocks(); ++block) {
    for (int64_t i = 0; i < walk.run(); ++i) {
      out.push_back(cursor.offset(0) + i * walk.run_stride(0));
    }
    cursor.advance();
  }
  return out;
}

void same(const llm::Shape& shape, const llm::Dims& strides) {
  const std::vector<int64_t> expected = by_odometer(shape, strides);
  const std::vector<int64_t> actual = by_blocks(shape, strides);
  LLM_CHECK_MSG(expected.size() == actual.size(),
                "форма " << shape << ": элементов " << expected.size()
                         << ", а кусками получилось " << actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    LLM_CHECK_MSG(expected[i] == actual[i],
                  "форма " << shape << ", элемент " << i << ": ожидалось "
                           << expected[i] << ", получено " << actual[i]);
  }
}

}  // namespace

LLM_TEST(Iterate, BlockWalkMatchesTheElementOdometer) {
  // Плотный — схлопывается целиком.
  same(llm::Shape({2, 3, 4}), llm::contiguous_strides(llm::Shape({2, 3, 4})));

  // Скаляр: один кусок из одного элемента.
  same(llm::Shape(), llm::Dims());

  // Перестановка (0, 2, 1, 3) — та самая, которой внимание разворачивает
  // головы. Последняя ось плотная, схлопывается только она.
  same(llm::Shape({2, 3, 4, 5}), llm::Dims({60, 5, 15, 1}));

  // Растяжение: нулевой шаг по первой оси.
  same(llm::Shape({2, 3}), llm::Dims({0, 1}));

  // Растянуто всё — весь обход ложится в одну ячейку.
  same(llm::Shape({2, 3}), llm::Dims({0, 0}));

  // Оси размера 1 не мешают схлопыванию и не добавляют смещений.
  same(llm::Shape({2, 1, 3}), llm::Dims({3, 3, 1}));

  // Ось размера 1 с шагом, который ни с чем не согласован: он никогда не
  // должен попасть в смещение, потому что индекс по ней всегда ноль.
  same(llm::Shape({2, 1, 3}), llm::Dims({3, 999, 1}));

  // Хвост с дырой: последняя ось идёт с шагом 2, схлопывать с ней нечего.
  same(llm::Shape({3, 4}), llm::Dims({100, 2}));

  // Обратный порядок осей.
  same(llm::Shape({4, 5}), llm::Dims({1, 4}));

  // Ранг 1 и ранг 8 — края допустимого.
  same(llm::Shape({7}), llm::Dims({3}));
  same(llm::Shape({2, 1, 2, 1, 2, 1, 2, 1}),
       llm::contiguous_strides(llm::Shape({2, 1, 2, 1, 2, 1, 2, 1})));
}

LLM_TEST(Iterate, CursorStartsAtAnyBlock) {
  // Потоки заводят курсор сразу на своём куске, а не проходят одометром всё
  // начало. Если бы разложение номера куска по осям было неверным, потоки
  // читали бы чужие данные — и только при нескольких потоках, то есть не
  // всегда.
  const llm::Shape shape({3, 4, 5});
  const llm::Dims strides({40, 5, 1});  // перестановка, хвост плотный
  const llm::BlockWalkN<1> walk = llm::block_walk(shape, strides);

  for (int64_t start = 0; start < walk.blocks(); ++start) {
    llm::BlockWalkN<1>::Cursor from_start = walk.at(0);
    for (int64_t step = 0; step < start; ++step) {
      from_start.advance();
    }
    const llm::BlockWalkN<1>::Cursor direct = walk.at(start);
    LLM_CHECK_MSG(direct.offset(0) == from_start.offset(0),
                  "кусок " << start << ": сразу " << direct.offset(0)
                           << ", шагами " << from_start.offset(0));
  }
}

LLM_TEST(Iterate, TwoStrideSetsCollapseOnlyTogether) {
  // Кусок годится, только если ровно идут все наборы сразу. Здесь первый
  // набор плотный и схлопнулся бы целиком, а второй растянут по средней оси —
  // значит схлопнуться должна только последняя ось.
  const llm::Shape shape({2, 3, 4});
  const llm::Dims dense({12, 4, 1});
  const llm::Dims stretched({4, 0, 1});

  const llm::BlockWalkN<2> walk = llm::block_walk2(shape, dense, stretched);
  LLM_CHECK_EQ(walk.run(), static_cast<int64_t>(4));
  LLM_CHECK_EQ(walk.blocks(), static_cast<int64_t>(6));

  // И сами смещения обоих наборов совпадают с одометром по двум наборам.
  std::vector<int64_t> left_expected;
  std::vector<int64_t> right_expected;
  llm::for_each_offset2(shape, dense, stretched,
                        [&](int64_t left, int64_t right) {
                          left_expected.push_back(left);
                          right_expected.push_back(right);
                        });

  std::vector<int64_t> left_actual;
  std::vector<int64_t> right_actual;
  llm::BlockWalkN<2>::Cursor cursor = walk.at(0);
  for (int64_t block = 0; block < walk.blocks(); ++block) {
    for (int64_t i = 0; i < walk.run(); ++i) {
      left_actual.push_back(cursor.offset(0) + i * walk.run_stride(0));
      right_actual.push_back(cursor.offset(1) + i * walk.run_stride(1));
    }
    cursor.advance();
  }

  LLM_CHECK_EQ(left_actual.size(), left_expected.size());
  for (std::size_t i = 0; i < left_expected.size(); ++i) {
    LLM_CHECK_EQ(left_actual[i], left_expected[i]);
    LLM_CHECK_EQ(right_actual[i], right_expected[i]);
  }
}

LLM_TEST(Iterate, CollapsingActuallyHappens) {
  // Предыдущие тесты остались бы зелёными, даже если бы схлопывание не
  // работало вовсе: кусок длины 1 тоже даёт верные смещения, только медленно.
  // Здесь проверяется, что оно происходит — иначе вся польза заголовка была бы
  // не проверена ничем.
  const llm::Shape shape({2, 3, 4});
  const llm::BlockWalkN<1> dense =
      llm::block_walk(shape, llm::contiguous_strides(shape));
  LLM_CHECK_EQ(dense.run(), static_cast<int64_t>(24));
  LLM_CHECK_EQ(dense.blocks(), static_cast<int64_t>(1));

  // Перестановка: схлопывается только плотный хвост.
  const llm::BlockWalkN<1> permuted =
      llm::block_walk(llm::Shape({2, 3, 4, 5}), llm::Dims({60, 5, 15, 1}));
  LLM_CHECK_EQ(permuted.run(), static_cast<int64_t>(5));
  LLM_CHECK_EQ(permuted.blocks(), static_cast<int64_t>(24));

  // Полностью растянутое схлопывается целиком, но с нулевым шагом внутри.
  const llm::BlockWalkN<1> stretched =
      llm::block_walk(llm::Shape({2, 3}), llm::Dims({0, 0}));
  LLM_CHECK_EQ(stretched.run(), static_cast<int64_t>(6));
  LLM_CHECK_EQ(stretched.run_stride(0), static_cast<int64_t>(0));
  LLM_CHECK_EQ(stretched.blocks(), static_cast<int64_t>(1));
}
