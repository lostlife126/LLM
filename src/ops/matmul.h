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

#include "core/tensor.h"

namespace llm {
namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b);

// out = alpha * a * b + beta * out. Форма out должна совпадать с формой
// результата matmul(a, b), размещение — плотное.
void matmul_into(const Tensor& a, const Tensor& b, float alpha, float beta,
                 Tensor* out);

}  // namespace ops
}  // namespace llm

#endif  // LLM_OPS_MATMUL_H_
