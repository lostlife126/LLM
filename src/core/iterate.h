// Обход тензоров с произвольными шагами.
//
// Плотный тензор обходится простым циклом по памяти, а вид с шагами — нет:
// соседние по индексу элементы могут лежать где угодно. Здесь собран «одометр»,
// который идёт по всем позициям формы в порядке плотного размещения и на каждом
// шаге выдаёт смещения сразу в нескольких наборах шагов.
//
// Несколько наборов нужны, потому что поэлементные операции работают с двумя
// входами и одним выходом, у каждого из которых свои шаги — например, после
// растяжения одного из аргументов шагом 0.

#ifndef LLM_CORE_ITERATE_H_
#define LLM_CORE_ITERATE_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/shape.h"

namespace llm {
namespace detail {

// Увеличивает счётчик index на единицу в порядке «младший разряд — последняя
// ось» и поправляет смещения. При переносе разряд сбрасывается в нуль, а из
// смещения вычитается весь накопленный по этой оси вклад.
//
// Корректно работает с шагом 0 (растянутая ось) и с любой перестановкой осей.
inline void advance(const Shape& shape, std::vector<int64_t>* index,
                    const std::vector<int64_t>* const* strides,
                    int64_t* offsets, int count) {
  for (int axis = shape.rank() - 1; axis >= 0; --axis) {
    const std::size_t a = static_cast<std::size_t>(axis);
    (*index)[a] += 1;
    for (int t = 0; t < count; ++t) {
      offsets[t] += (*strides[t])[a];
    }
    if ((*index)[a] < shape.dim(axis)) {
      return;
    }
    for (int t = 0; t < count; ++t) {
      offsets[t] -= (*index)[a] * (*strides[t])[a];
    }
    (*index)[a] = 0;
  }
}

}  // namespace detail

template <typename Fn>
void for_each_offset(const Shape& shape, const std::vector<int64_t>& strides,
                     Fn fn) {
  const int64_t total = shape.numel();
  if (total == 0) {
    return;
  }
  std::vector<int64_t> index(static_cast<std::size_t>(shape.rank()), 0);
  const std::vector<int64_t>* all[] = {&strides};
  int64_t offsets[] = {0};
  for (int64_t counter = 0; counter < total; ++counter) {
    fn(offsets[0]);
    detail::advance(shape, &index, all, offsets, 1);
  }
}

template <typename Fn>
void for_each_offset2(const Shape& shape, const std::vector<int64_t>& strides_a,
                      const std::vector<int64_t>& strides_b, Fn fn) {
  const int64_t total = shape.numel();
  if (total == 0) {
    return;
  }
  std::vector<int64_t> index(static_cast<std::size_t>(shape.rank()), 0);
  const std::vector<int64_t>* all[] = {&strides_a, &strides_b};
  int64_t offsets[] = {0, 0};
  for (int64_t counter = 0; counter < total; ++counter) {
    fn(offsets[0], offsets[1]);
    detail::advance(shape, &index, all, offsets, 2);
  }
}

template <typename Fn>
void for_each_offset3(const Shape& shape, const std::vector<int64_t>& strides_a,
                      const std::vector<int64_t>& strides_b,
                      const std::vector<int64_t>& strides_c, Fn fn) {
  const int64_t total = shape.numel();
  if (total == 0) {
    return;
  }
  std::vector<int64_t> index(static_cast<std::size_t>(shape.rank()), 0);
  const std::vector<int64_t>* all[] = {&strides_a, &strides_b, &strides_c};
  int64_t offsets[] = {0, 0, 0};
  for (int64_t counter = 0; counter < total; ++counter) {
    fn(offsets[0], offsets[1], offsets[2]);
    detail::advance(shape, &index, all, offsets, 3);
  }
}

// Обход формы кусками подряд идущих элементов.
//
// Одометр выше идёт по одному элементу, и на каждый элемент приходится цикл по
// осям с обращениями к вектору шагов — десяток операций на один скопированный
// float. Для плотного тензора это чистая потеря, но и для вида с
// перестановленными осями тоже: у перестановки (0, 2, 1, 3), которой внимание
// разворачивает головы, последняя ось остаётся плотной, и её можно обходить
// целым отрезком.
//
// BlockWalk схлопывает самый длинный хвост формы, на котором шаги всех наборов
// сразу идут ровно, и превращает обход в «кусок за куском». Число обращений к
// одометру падает во столько раз, какова длина куска: для головы размером 32 —
// в тридцать два. А внутри куска шаг постоянный, и такой цикл компилятор
// векторизует сам.
//
// Наборов шагов может быть несколько — столько же, сколько у for_each_offset2
// и for_each_offset3. Схлопывать хвост приходится по всем сразу: кусок годится
// только если ровно идут все участники. Ноль как шаг это правило переживает
// без исключений — растянутая ось с шагом 0 схлопывается с такой же.
//
// Курсор умеет заводиться на произвольном номере куска. Это нужно потокам:
// иначе каждый поток, чтобы добраться до своей середины, прошёл бы одометром
// всё начало.
template <int kSets>
class BlockWalkN {
 public:
  // strides[s] — набор номер s; все наборы описывают одну и ту же форму.
  BlockWalkN(const Shape& shape,
             const std::vector<int64_t>* const (&strides)[kSets]) {
    const int rank = shape.rank();
    int collapsed = 0;
    for (int set = 0; set < kSets; ++set) {
      run_strides_[set] = 1;
    }

    if (rank > 0) {
      run_ = shape.dim(rank - 1);
      collapsed = 1;
      for (int set = 0; set < kSets; ++set) {
        run_strides_[set] = (*strides[set])[static_cast<std::size_t>(rank - 1)];
      }
      // Ось присоединяется к куску, если её шаг в точности равен шагу
      // следующей оси, умноженному на её размер, — и так у всех наборов. Это
      // ровно условие того, что два куска лежат подряд и становятся одним.
      for (int axis = rank - 2; axis >= 0; --axis) {
        bool mergeable = true;
        for (int set = 0; set < kSets; ++set) {
          const std::vector<int64_t>& s = *strides[set];
          const std::size_t a = static_cast<std::size_t>(axis);
          if (s[a] != s[a + 1] * shape.dim(axis + 1)) {
            mergeable = false;
            break;
          }
        }
        if (!mergeable) {
          break;
        }
        run_ *= shape.dim(axis);
        ++collapsed;
      }
    }

    blocks_ = 1;
    for (int axis = 0; axis < rank - collapsed; ++axis) {
      const std::size_t a = static_cast<std::size_t>(axis);
      dims_.push_back(shape.dim(axis));
      for (int set = 0; set < kSets; ++set) {
        outer_strides_[set].push_back((*strides[set])[a]);
      }
      blocks_ *= shape.dim(axis);
    }
  }

  // Сколько элементов в куске и с каким шагом они идут у набора set. Шаг 1 —
  // самый частый и самый быстрый случай, шаг 0 означает, что весь кусок
  // ложится в одну ячейку, то есть суммируется.
  int64_t run() const { return run_; }
  int64_t run_stride(int set) const { return run_strides_[set]; }
  int64_t blocks() const { return blocks_; }

  class Cursor {
   public:
    int64_t offset(int set) const { return offsets_[set]; }

    void advance() {
      for (int axis = static_cast<int>(index_.size()) - 1; axis >= 0; --axis) {
        const std::size_t a = static_cast<std::size_t>(axis);
        index_[a] += 1;
        for (int set = 0; set < kSets; ++set) {
          offsets_[set] += strides_[set][a];
        }
        if (index_[a] < dims_[a]) {
          return;
        }
        for (int set = 0; set < kSets; ++set) {
          offsets_[set] -= index_[a] * strides_[set][a];
        }
        index_[a] = 0;
      }
    }

   private:
    friend class BlockWalkN;

    // Размеры и шаги копируются в курсор, а не берутся по указателю на обход.
    // Так у каждого потока свои, и чтения соседних потоков не делят одну
    // строку кэша.
    std::vector<int64_t> dims_;
    std::vector<int64_t> strides_[kSets];
    std::vector<int64_t> index_;
    int64_t offsets_[kSets] = {};
  };

  // Курсор, установленный на кусок номер block.
  Cursor at(int64_t block) const {
    Cursor cursor;
    cursor.dims_ = dims_;
    for (int set = 0; set < kSets; ++set) {
      cursor.strides_[set] = outer_strides_[set];
      cursor.offsets_[set] = 0;
    }
    cursor.index_.assign(dims_.size(), 0);
    for (int axis = static_cast<int>(dims_.size()) - 1; axis >= 0; --axis) {
      const std::size_t a = static_cast<std::size_t>(axis);
      const int64_t position = block % dims_[a];
      block /= dims_[a];
      cursor.index_[a] = position;
      for (int set = 0; set < kSets; ++set) {
        cursor.offsets_[set] += position * outer_strides_[set][a];
      }
    }
    return cursor;
  }

 private:
  std::vector<int64_t> dims_;
  std::vector<int64_t> outer_strides_[kSets];
  int64_t run_strides_[kSets] = {};
  int64_t run_ = 1;
  int64_t blocks_ = 1;
};

// Удобные обёртки: обход с одним и с двумя наборами шагов.
inline BlockWalkN<1> block_walk(const Shape& shape,
                                const std::vector<int64_t>& strides) {
  const std::vector<int64_t>* const sets[1] = {&strides};
  return BlockWalkN<1>(shape, sets);
}

inline BlockWalkN<2> block_walk2(const Shape& shape,
                                 const std::vector<int64_t>& strides_a,
                                 const std::vector<int64_t>& strides_b) {
  const std::vector<int64_t>* const sets[2] = {&strides_a, &strides_b};
  return BlockWalkN<2>(shape, sets);
}

}  // namespace llm

#endif  // LLM_CORE_ITERATE_H_
