// Умножение матриц: C = alpha * op(A) * op(B) + beta * C.
//
// Всё в row-major. op(A) имеет размер m×k, op(B) — k×n, C — m×n.
// lda, ldb, ldc — шаг строки (leading dimension) матрицы в памяти, заданный
// ДО транспонирования: при transpose_a матрица A лежит как k×m с шагом строки
// lda. Такая развязка формы и размещения позволяет передавать сюда
// транспонированный вид тензора без копирования данных.
//
// Две реализации с одинаковой семантикой:
//
//   gemm_naive — тройной цикл. Остаётся в проекте навсегда как эталон: любая
//                оптимизированная версия сверяется с ней в тестах.
//   gemm       — блочная версия с упаковкой панелей.

#ifndef LLM_OPS_GEMM_H_
#define LLM_OPS_GEMM_H_

#include <cstdint>

namespace llm {
namespace ops {

void gemm_naive(bool transpose_a, bool transpose_b, int64_t m, int64_t n,
                int64_t k, float alpha, const float* a, int64_t lda,
                const float* b, int64_t ldb, float beta, float* c, int64_t ldc);

void gemm(bool transpose_a, bool transpose_b, int64_t m, int64_t n, int64_t k,
          float alpha, const float* a, int64_t lda, const float* b, int64_t ldb,
          float beta, float* c, int64_t ldc);

}  // namespace ops
}  // namespace llm

#endif  // LLM_OPS_GEMM_H_
