// Tensor — многомерный массив float с шагами по осям.
//
// Ключевое решение: тензор хранит не только форму, но и шаги (strides), а сами
// данные держит через shared_ptr<Storage>. Из этого следует главное свойство —
// reshape, transpose, slice и expand не копируют данные, а лишь по-новому
// описывают тот же буфер. Именно поэтому attention может смотреть на один и тот
// же тензор как на (batch, seq, heads, head_dim) и как на (batch, heads, seq,
// head_dim), не тратя ни байта.
//
// Тип элемента фиксирован как float, и это осознанно. Обобщение по dtype
// заставило бы каждую операцию начинаться с диспетчеризации по типу, что
// похоронило бы читаемость ради квантизации, которая в этом проекте вынесена в
// необязательный хвост. Идентификаторы токенов — целые, но они живут в обычных
// векторах, а не в тензорах, поэтому второй тип элемента здесь не нужен.
//
// Поля для градиентов появятся в M2: сейчас тензор — это чистое хранилище.

#ifndef LLM_CORE_TENSOR_H_
#define LLM_CORE_TENSOR_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/shape.h"
#include "core/span.h"
#include "core/storage.h"

namespace llm {

class Tensor {
 public:
  Tensor() {}

  // Содержимое не определено: для буферов, которые тут же будут заполнены.
  static Tensor uninitialized(const Shape& shape);
  static Tensor zeros(const Shape& shape);
  static Tensor full(const Shape& shape, float value);

  // Значения перечисляются в порядке плотного размещения. Нужно в тестах,
  // чтобы задавать маленькие матрицы явно.
  static Tensor from_values(const Shape& shape,
                            const std::vector<float>& values);

  const Shape& shape() const { return shape_; }
  const std::vector<int64_t>& strides() const { return strides_; }

  int rank() const { return shape_.rank(); }
  int64_t dim(int axis) const { return shape_.dim(axis); }
  int64_t numel() const { return shape_.numel(); }
  bool empty() const { return numel() == 0; }

  int64_t stride(int axis) const {
    return strides_[static_cast<std::size_t>(shape_.normalize_axis(axis))];
  }

  // Плотное размещение: шаги в точности такие, как у contiguous_strides.
  // Только для таких тензоров имеет смысл flat() и передача сырого указателя
  // в ядра, которые ожидают непрерывную память.
  bool is_contiguous() const;

  // Указатель на элемент с нулевыми индексами по всем осям. У вида он смещён
  // внутрь буфера владельца, поэтому храним его напрямую, а не как пару
  // «база + смещение»: меньше арифметики на каждом обращении и никакого
  // сложения с нулевым указателем у пустого тензора.
  float* data() { return data_; }
  const float* data() const { return data_; }

  Span<float> flat();
  Span<const float> flat() const;

  // Доступ по индексам с учётом шагов. Для горячих циклов слишком медленно —
  // там берут data() и шаги напрямую, — но в тестах и при отладке незаменим.
  float& operator()(int64_t i0);
  float& operator()(int64_t i0, int64_t i1);
  float& operator()(int64_t i0, int64_t i1, int64_t i2);
  float& operator()(int64_t i0, int64_t i1, int64_t i2, int64_t i3);
  const float& operator()(int64_t i0) const;
  const float& operator()(int64_t i0, int64_t i1) const;
  const float& operator()(int64_t i0, int64_t i1, int64_t i2) const;
  const float& operator()(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const;

  float& at(const std::vector<int64_t>& index);
  const float& at(const std::vector<int64_t>& index) const;

  // --- Виды: результат делит буфер с исходным тензором ---

  // Требует плотного размещения: при произвольных шагах новая форма в общем
  // случае невыразима, и вызывающий обязан сначала сделать contiguous().
  Tensor reshape(const Shape& shape) const;

  // Меняет две оси местами — просто перестановка формы и шагов.
  Tensor transpose(int axis_a, int axis_b) const;

  // Произвольная перестановка осей; order[i] — какая ось исходного тензора
  // становится i-й.
  Tensor permute(const std::vector<int>& order) const;

  // Непрерывный отрезок по одной оси; ранг сохраняется.
  Tensor slice(int axis, int64_t start, int64_t count) const;

  // Один срез по оси с её удалением: ранг уменьшается на 1.
  Tensor select(int axis, int64_t index) const;

  // Растягивает оси размера 1 до нужного размера шагом 0. Шаг 0 означает, что
  // все индексы по этой оси читают одну и ту же ячейку — так broadcast
  // выражается без копирования данных.
  Tensor expand(const Shape& shape) const;

  // --- Копии ---

  // Плотная копия, если размещение не плотное; иначе тот же тензор.
  Tensor contiguous() const;

  // Всегда новая плотная копия.
  Tensor clone() const;

  void fill(float value);
  void zero() { fill(0.0f); }

  bool shares_storage_with(const Tensor& other) const {
    return storage_ && storage_ == other.storage_;
  }

  // Форма и значения, для сообщений о падении тестов. Большие тензоры
  // печатаются частично.
  std::string debug_string(int64_t max_values = 32) const;

 private:
  Tensor(std::shared_ptr<Storage> storage, float* data, const Shape& shape,
         std::vector<int64_t> strides);

  int64_t flat_offset(const int64_t* index, int count) const;

  std::shared_ptr<Storage> storage_;  // владение; у вида — общее с владельцем
  float* data_ = nullptr;
  Shape shape_;
  std::vector<int64_t> strides_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

}  // namespace llm

#endif  // LLM_CORE_TENSOR_H_
