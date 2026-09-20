#include "core/tensor.h"

#include <algorithm>
#include <cstring>
#include <ostream>
#include <sstream>

#include "core/iterate.h"
#include "core/thread_pool.h"

namespace llm {
namespace {

// Сколько элементов стоит отдавать одному потоку при копировании.
//
// Копия упирается в память, а не в арифметику: на элемент приходится чтение и
// запись четырёх байт. Вход в параллельную область стоит около двух
// микросекунд, за которые один поток успевает скопировать порядка десяти
// килобайт, — отсюда и порядок величины.
constexpr int64_t kCopyGrain = 1 << 14;

// Плотная копия тензора с произвольными шагами.
//
// Копирование идёт кусками подряд идущих элементов, а не по одному: см.
// BlockWalk. Для плотного тензора кусок один на весь тензор, и всё сводится к
// memcpy; для перестановки осей внимания кусок равен размеру головы.
void copy_dense(const Shape& shape, const Dims& strides,
                const float* in, float* out) {
  const BlockWalkN<1> walk = block_walk(shape, strides);
  const int64_t run = walk.run();
  const int64_t run_stride = walk.run_stride(0);
  const int64_t grain =
      std::max<int64_t>(1, kCopyGrain / std::max<int64_t>(run, 1));

  parallel_range(walk.blocks(), grain, [&](int64_t begin, int64_t end) {
    BlockWalkN<1>::Cursor cursor = walk.at(begin);
    float* dst = out + begin * run;
    if (run_stride == 1) {
      for (int64_t block = begin; block < end; ++block) {
        std::memcpy(dst, in + cursor.offset(0),
                    static_cast<std::size_t>(run) * sizeof(float));
        dst += run;
        cursor.advance();
      }
      return;
    }
    // Неплотный хвост: чаще всего это транспонированный вид. Куском он не
    // копируется, но шаг внутри куска постоянен, и это всё равно много лучше
    // одометра на каждый элемент.
    for (int64_t block = begin; block < end; ++block) {
      const float* src = in + cursor.offset(0);
      for (int64_t i = 0; i < run; ++i) {
        dst[i] = src[i * run_stride];
      }
      dst += run;
      cursor.advance();
    }
  });
}

}  // namespace

Tensor::Tensor(std::shared_ptr<Storage> storage, float* data,
               const Shape& shape, const Dims& strides)
    : storage_(std::move(storage)),
      data_(data),
      shape_(shape),
      strides_(strides) {
  LLM_DCHECK_EQ(strides_.size(), shape_.rank());
}

Tensor Tensor::uninitialized(const Shape& shape) {
  const int64_t count = shape.numel();
  std::shared_ptr<Storage> storage = std::make_shared<Storage>(
      static_cast<std::size_t>(count) * sizeof(float));
  float* data = count == 0 ? nullptr : storage->as<float>().data();
  return Tensor(std::move(storage), data, shape, contiguous_strides(shape));
}

Tensor Tensor::zeros(const Shape& shape) {
  Tensor result = uninitialized(shape);
  if (!result.storage_) {
    return result;
  }
  // Обнуление делится по потокам, а не идёт одним memset. За шаг обучения
  // обнуляется 15 МБ у nano и 76 МБ у tiny; в один поток это заметная часть
  // той последовательной доли, в которую упирается масштабируемость.
  const int64_t count = result.numel();
  float* data = result.data();
  parallel_range(count, kCopyGrain, [&](int64_t begin, int64_t end) {
    std::memset(data + begin, 0,
                static_cast<std::size_t>(end - begin) * sizeof(float));
  });
  return result;
}

Tensor Tensor::full(const Shape& shape, float value) {
  Tensor result = uninitialized(shape);
  result.fill(value);
  return result;
}

Tensor Tensor::from_values(const Shape& shape,
                           const std::vector<float>& values) {
  LLM_CHECK_MSG(static_cast<int64_t>(values.size()) == shape.numel(),
                "для формы " << shape << " нужно " << shape.numel()
                             << " значений, передано " << values.size());
  Tensor result = uninitialized(shape);
  if (!values.empty()) {
    std::memcpy(result.data(), values.data(), values.size() * sizeof(float));
  }
  return result;
}

bool Tensor::is_contiguous() const {
  // Оси размера 1 не ограничивают размещение: индекс по ним всегда 0, поэтому
  // их шаг никогда не участвует в вычислении адреса.
  int64_t expected = 1;
  for (int axis = rank() - 1; axis >= 0; --axis) {
    const int64_t size = shape_.dim(axis);
    if (size == 1) {
      continue;
    }
    if (strides_[static_cast<std::size_t>(axis)] != expected) {
      return false;
    }
    expected *= size;
  }
  return true;
}

Span<float> Tensor::flat() {
  LLM_CHECK_MSG(defined(), "flat() у неопределённого тензора");
  LLM_CHECK_MSG(is_contiguous(),
                "flat() требует плотного размещения; форма " << shape_);
  return Span<float>(data_, static_cast<std::size_t>(numel()));
}

Span<const float> Tensor::flat() const {
  LLM_CHECK_MSG(defined(), "flat() у неопределённого тензора");
  LLM_CHECK_MSG(is_contiguous(),
                "flat() требует плотного размещения; форма " << shape_);
  return Span<const float>(data_, static_cast<std::size_t>(numel()));
}

int64_t Tensor::flat_offset(const int64_t* index, int count) const {
  // Неопределённый тензор — это ранг 0, то есть numel() == 1, как у скаляра.
  // Без этой проверки обращение по индексу читало бы нулевой указатель.
  LLM_CHECK_MSG(defined(), "обращение по индексу к неопределённому тензору");
  LLM_CHECK_MSG(count == rank(),
                "индекс ранга " << count << " для тензора формы " << shape_);
  int64_t offset = 0;
  for (int axis = 0; axis < count; ++axis) {
    LLM_DCHECK_GE(index[axis], static_cast<int64_t>(0));
    LLM_DCHECK_LT(index[axis], shape_.dim(axis));
    offset += index[axis] * strides_[static_cast<std::size_t>(axis)];
  }
  return offset;
}

float& Tensor::operator()(int64_t i0) {
  const int64_t index[] = {i0};
  return data_[flat_offset(index, 1)];
}

float& Tensor::operator()(int64_t i0, int64_t i1) {
  const int64_t index[] = {i0, i1};
  return data_[flat_offset(index, 2)];
}

float& Tensor::operator()(int64_t i0, int64_t i1, int64_t i2) {
  const int64_t index[] = {i0, i1, i2};
  return data_[flat_offset(index, 3)];
}

float& Tensor::operator()(int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
  const int64_t index[] = {i0, i1, i2, i3};
  return data_[flat_offset(index, 4)];
}

const float& Tensor::operator()(int64_t i0) const {
  const int64_t index[] = {i0};
  return data_[flat_offset(index, 1)];
}

const float& Tensor::operator()(int64_t i0, int64_t i1) const {
  const int64_t index[] = {i0, i1};
  return data_[flat_offset(index, 2)];
}

const float& Tensor::operator()(int64_t i0, int64_t i1, int64_t i2) const {
  const int64_t index[] = {i0, i1, i2};
  return data_[flat_offset(index, 3)];
}

const float& Tensor::operator()(int64_t i0, int64_t i1, int64_t i2,
                                int64_t i3) const {
  const int64_t index[] = {i0, i1, i2, i3};
  return data_[flat_offset(index, 4)];
}

float& Tensor::at(const std::vector<int64_t>& index) {
  return data_[flat_offset(index.data(), static_cast<int>(index.size()))];
}

const float& Tensor::at(const std::vector<int64_t>& index) const {
  return data_[flat_offset(index.data(), static_cast<int>(index.size()))];
}

Tensor Tensor::reshape(const Shape& shape) const {
  LLM_CHECK_MSG(shape.numel() == numel(), "reshape " << shape_ << " -> "
                                                     << shape
                                                     << " меняет число "
                                                        "элементов");
  LLM_CHECK_MSG(is_contiguous(),
                "reshape требует плотного размещения; сначала contiguous()");
  return Tensor(storage_, data_, shape, contiguous_strides(shape));
}

Tensor Tensor::transpose(int axis_a, int axis_b) const {
  const int a = shape_.normalize_axis(axis_a);
  const int b = shape_.normalize_axis(axis_b);
  Dims dims = shape_.dims();
  Dims strides = strides_;
  std::swap(dims[a], dims[b]);
  std::swap(strides[a], strides[b]);
  return Tensor(storage_, data_, Shape(dims), strides);
}

Tensor Tensor::permute(const std::vector<int>& order) const {
  LLM_CHECK_MSG(static_cast<int>(order.size()) == rank(),
                "перестановка из " << order.size() << " осей для тензора ранга "
                                   << rank());
  std::vector<bool> seen(order.size(), false);
  Dims dims;
  Dims strides;
  for (std::size_t i = 0; i < order.size(); ++i) {
    const int axis = shape_.normalize_axis(order[i]);
    LLM_CHECK_MSG(!seen[static_cast<std::size_t>(axis)],
                  "ось " << axis << " указана в перестановке дважды");
    seen[static_cast<std::size_t>(axis)] = true;
    dims.push_back(shape_.dim(axis));
    strides.push_back(strides_[axis]);
  }
  return Tensor(storage_, data_, Shape(dims), strides);
}

Tensor Tensor::slice(int axis, int64_t start, int64_t count) const {
  const int a = shape_.normalize_axis(axis);
  LLM_CHECK_GE(start, static_cast<int64_t>(0));
  LLM_CHECK_GE(count, static_cast<int64_t>(0));
  LLM_CHECK_MSG(start + count <= shape_.dim(a),
                "срез [" << start << ", " << start + count << ") по оси " << a
                         << " выходит за границу " << shape_.dim(a));
  Dims dims = shape_.dims();
  dims[a] = count;
  float* data = data_ + start * strides_[a];
  return Tensor(storage_, data, Shape(dims), strides_);
}

Tensor Tensor::select(int axis, int64_t index) const {
  const int a = shape_.normalize_axis(axis);
  LLM_CHECK_GE(index, static_cast<int64_t>(0));
  LLM_CHECK_LT(index, shape_.dim(a));
  // Удаление оси: в новый набор переписывается всё, кроме неё.
  //
  // Счётчик назван other, а не axis: параметр функции зовётся axis, и цикл с
  // тем же именем его перекрывал. Работало это верно — a посчитан до цикла, —
  // но читалось как ошибка.
  Dims dims;
  Dims strides;
  for (int other = 0; other < rank(); ++other) {
    if (other == a) {
      continue;
    }
    dims.push_back(shape_.dim(other));
    strides.push_back(strides_[other]);
  }
  float* data = data_ + index * strides_[a];
  return Tensor(storage_, data, Shape(dims), strides);
}

Tensor Tensor::expand(const Shape& shape) const {
  LLM_CHECK_MSG(shape.rank() >= rank(), "expand не может уменьшать ранг: "
                                            << shape_ << " -> " << shape);
  // Оси выравниваются справа, как в broadcast: новые оси добавляются слева.
  const int pad = shape.rank() - rank();
  Dims strides = Dims::zeros(shape.rank());
  for (int axis = 0; axis < rank(); ++axis) {
    const int64_t from = shape_.dim(axis);
    const int64_t to = shape.dim(axis + pad);
    if (from == to) {
      strides[axis + pad] = strides_[axis];
    } else {
      // Шаг 0: все индексы по этой оси читают одну и ту же ячейку. Так
      // растяжение выражается без копирования данных.
      LLM_CHECK_MSG(from == 1, "ось " << axis << " размера " << from
                                      << " нельзя растянуть до " << to);
      strides[axis + pad] = 0;
    }
  }
  return Tensor(storage_, data_, shape, strides);
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) {
    return *this;
  }
  return clone();
}

Tensor Tensor::clone() const {
  // Копия «ничего» — это «ничего». Иначе получился бы тензор ранга 0 с одним
  // элементом, скопированным из нулевого указателя.
  if (!defined()) {
    return Tensor();
  }
  Tensor result = uninitialized(shape_);
  if (numel() == 0) {
    return result;
  }
  copy_dense(shape_, strides_, data_, result.data());
  return result;
}

void Tensor::fill(float value) {
  LLM_CHECK_MSG(defined(), "заполнение неопределённого тензора");
  if (numel() == 0) {
    return;
  }
  if (is_contiguous()) {
    // Быстрый путь: заполнение веса или буфера активаций идёт по плотной
    // памяти, и обход одометром здесь был бы на порядок дороже.
    std::fill(data_, data_ + numel(), value);
    return;
  }
  float* out = data_;
  for_each_offset(shape_, strides_,
                  [&](int64_t offset) { out[offset] = value; });
}

std::string Tensor::debug_string(int64_t max_values) const {
  // Эта функция вызывается из сообщений о непрошедших проверках, то есть
  // тогда, когда что-то уже не так. Падать ей нельзя ни при каком состоянии
  // тензора — иначе разбираемая ошибка превращается в segmentation fault без
  // единой строки отчёта. У неопределённого тензора ранг 0, значит numel()
  // равен единице, и обход прочитал бы нулевой указатель.
  if (!defined()) {
    return "Tensor(не определён)";
  }
  std::ostringstream oss;
  oss << "Tensor" << shape_;
  if (!is_contiguous()) {
    oss << " strides(";
    for (int i = 0; i < strides_.size(); ++i) {
      if (i != 0) {
        oss << ", ";
      }
      oss << strides_[i];
    }
    oss << ")";
  }
  oss << " [";
  int64_t printed = 0;
  const float* in = data_;
  for_each_offset(shape_, strides_, [&](int64_t offset) {
    if (printed < max_values) {
      if (printed != 0) {
        oss << ", ";
      }
      oss << in[offset];
    }
    ++printed;
  });
  if (printed > max_values) {
    oss << ", ... (всего " << printed << ")";
  }
  oss << "]";
  return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  return os << tensor.debug_string();
}

}  // namespace llm
