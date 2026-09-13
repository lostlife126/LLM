#include "ops/matmul.h"

#include <algorithm>
#include <vector>

#include "core/check.h"
#include "ops/gemm.h"

namespace llm {
namespace ops {
namespace {

// Описание двумерного тензора в терминах, которые понимает gemm: указатель,
// шаг строки и флаг транспонирования.
struct MatrixView {
  const float* data;
  int64_t ld;
  bool transposed;
};

// gemm умеет работать с матрицей, у которой единичный шаг по одной из двух
// осей. Это ровно два случая: плотные строки (обычная матрица) и плотные
// столбцы (транспонированный вид). Всё остальное — например, результат
// permute с перемешанными осями — приходится materializovat через
// contiguous().
bool try_as_matrix(const Tensor& tensor, MatrixView* out) {
  LLM_DCHECK_EQ(tensor.rank(), 2);
  const int64_t rows = tensor.dim(0);
  const int64_t cols = tensor.dim(1);
  int64_t row_stride = tensor.stride(0);
  int64_t col_stride = tensor.stride(1);

  // Ось размера 1 не ограничивает размещение: индекс по ней всегда 0, её шаг
  // в вычислении адреса не участвует. Подставляем значения плотной матрицы,
  // иначе такой вид отвергался бы без причины.
  if (rows == 1) {
    row_stride = cols;
  }
  if (cols == 1) {
    col_stride = 1;
  }

  if (col_stride == 1 && row_stride >= cols) {
    out->data = tensor.data();
    out->ld = row_stride;
    out->transposed = false;
    return true;
  }
  if (row_stride == 1 && col_stride >= rows) {
    out->data = tensor.data();
    out->ld = col_stride;
    out->transposed = true;
    return true;
  }
  return false;
}

// Приводит двумерный тензор к виду, пригодному для gemm, при необходимости
// делая плотную копию. Копия возвращается через fallback, чтобы вызывающий
// держал её живой до конца умножения.
MatrixView as_matrix(const Tensor& tensor, Tensor* fallback) {
  MatrixView view;
  if (try_as_matrix(tensor, &view)) {
    return view;
  }
  *fallback = tensor.contiguous();
  const bool ok = try_as_matrix(*fallback, &view);
  LLM_CHECK(ok);
  return view;
}

// Двумерный срез батча. Для матрицы без оси батча возвращает её саму — так
// один и тот же код обслуживает и общий вес, и отдельную матрицу на элемент
// батча.
Tensor batch_slice(const Tensor& tensor, int64_t index) {
  if (tensor.rank() == 2) {
    return tensor;
  }
  return tensor.select(0, tensor.dim(0) == 1 ? 0 : index);
}

int64_t batch_size(const Tensor& tensor) {
  return tensor.rank() == 3 ? tensor.dim(0) : 1;
}

Shape result_shape(const Tensor& a, const Tensor& b) {
  const int64_t m = a.dim(-2);
  const int64_t n = b.dim(-1);
  if (a.rank() == 2 && b.rank() == 2) {
    return Shape{m, n};
  }
  return Shape{std::max(batch_size(a), batch_size(b)), m, n};
}

void check_operands(const Tensor& a, const Tensor& b) {
  LLM_CHECK_MSG(a.rank() == 2 || a.rank() == 3,
                "matmul ожидает ранг 2 или 3, получено " << a.shape());
  LLM_CHECK_MSG(b.rank() == 2 || b.rank() == 3,
                "matmul ожидает ранг 2 или 3, получено " << b.shape());
  LLM_CHECK_MSG(a.dim(-1) == b.dim(-2), "несовместимые формы для matmul: "
                                            << a.shape() << " и " << b.shape());
  const int64_t batch_a = batch_size(a);
  const int64_t batch_b = batch_size(b);
  LLM_CHECK_MSG(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                "несовместимые батчи: " << a.shape() << " и " << b.shape());
}

}  // namespace

void matmul_into(const Tensor& a, const Tensor& b, float alpha, float beta,
                 Tensor* out) {
  LLM_CHECK(out != nullptr);
  check_operands(a, b);
  LLM_CHECK_MSG(out->shape() == result_shape(a, b),
                "out имеет форму " << out->shape() << ", ожидалась "
                                   << result_shape(a, b));
  LLM_CHECK_MSG(out->is_contiguous(), "out должен быть плотным");

  const int64_t m = a.dim(-2);
  const int64_t k = a.dim(-1);
  const int64_t n = b.dim(-1);
  const int64_t batch = std::max(batch_size(a), batch_size(b));

  for (int64_t index = 0; index < batch; ++index) {
    const Tensor a_slice = batch_slice(a, index);
    const Tensor b_slice = batch_slice(b, index);
    Tensor a_copy;
    Tensor b_copy;
    const MatrixView a_view = as_matrix(a_slice, &a_copy);
    const MatrixView b_view = as_matrix(b_slice, &b_copy);

    Tensor out_slice = out->rank() == 3 ? out->select(0, index) : *out;
    gemm(a_view.transposed, b_view.transposed, m, n, k, alpha, a_view.data,
         a_view.ld, b_view.data, b_view.ld, beta, out_slice.data(),
         out_slice.stride(0));
  }
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  check_operands(a, b);
  Tensor out = Tensor::uninitialized(result_shape(a, b));
  matmul_into(a, b, 1.0f, 0.0f, &out);
  return out;
}

}  // namespace ops
}  // namespace llm
