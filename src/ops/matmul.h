// Матричное умножение на уровне тензоров.
//
// Поддерживаемые формы (последние две оси — матрица):
//   (m, k) x (k, n)       -> (m, n)
//   (b, m, k) x (k, n)    -> (b, m, n)   вес один на весь батч
//   (b, m, k) x (b, k, n) -> (b, m, n)   своя матрица на каждый элемент батча
//   (m, k) x (b, k, n)    -> (b, m, n)
//
// Транспонированный вид передаётся без копирования: matmul распознаёт, что у
// вида единичный шаг по первой оси, и включает флаг транспонирования в gemm.
// Именно поэтому attention может считать Q * K^T, не материализуя K^T.

#ifndef LLM_OPS_MATMUL_H_
#define LLM_OPS_MATMUL_H_

#include "core/half.h"
#include "core/tensor.h"

namespace llm {
namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b);

// Плотная матрица весов в половинной разрядности: k строк, n столбцов,
// построчно. Не Tensor, и это то же решение, что в core/half.h: тип элемента у
// Tensor намеренно один, а вес операндом произвольной операции не бывает.
struct HalfMatrix {
  const Half* values = nullptr;
  int64_t rows = 0;
  int64_t columns = 0;
};

// Умножение на веса половинной разрядности. Транспонирование весов не
// поддерживается; если веса нужны транспонированными, хранить их надо уже
// транспонированными — при однократной подготовке это ничего не стоит.
//
// Численно совпадает с matmul, которому дали те же веса после округления, —
// побитово. Отсюда и способ проверки: округлить, посчитать обоими и сравнить
// на равенство.
Tensor matmul_half(const Tensor& a, const HalfMatrix& b);

// out = alpha * a * b + beta * out. Форма out должна совпадать с формой
// результата matmul(a, b), размещение — плотное.
void matmul_into(const Tensor& a, const Tensor& b, float alpha, float beta,
                 Tensor* out);

}  // namespace ops
}  // namespace llm

#endif  // LLM_OPS_MATMUL_H_
