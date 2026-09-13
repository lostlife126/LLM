// Нормировки и функции активации.
//
// В отличие от поэлементных операций, у этих обратный проход написан отдельным
// ядром, а не собран из примитивов. Причина в обоих случаях одна: и RMSNorm, и
// softmax содержат редукцию по последней оси, поэтому производная каждого
// элемента зависит от всей строки. Составить это из примитивов можно, но тогда
// на каждый вызов пришлось бы удерживать в памяти всю цепочку промежуточных
// тензоров, а вызывается это 2L раз за прямой проход.
//
// Правильность слитых ядер проверяется тем, что они обязаны совпадать со
// сборкой из примитивов — и по значению, и по градиентам (tests/test_nn.cpp).

#ifndef LLM_OPS_NN_H_
#define LLM_OPS_NN_H_

#include <cstdint>

#include "core/tensor.h"

namespace llm {
namespace ops {

// RMSNorm по последней оси: y_i = x_i / sqrt(mean(x^2) + eps) * w_i.
//
// В отличие от LayerNorm не вычитает среднее. Это не упрощение ради экономии:
// оказалось, что центрирование в трансформере ничего не даёт, а нормировка
// масштаба — даёт. Одна редукция вместо двух.
Tensor rms_norm(const Tensor& input, const Tensor& weight, float eps);

void rms_norm_backward(const Tensor& grad_output, const Tensor& input,
                       const Tensor& weight, float eps, Tensor* grad_input,
                       Tensor* grad_weight);

// LayerNorm по последней оси: y = (x - mean) / sqrt(var + eps) * w + b.
//
// Существует в проекте ради сравнения с RMSNorm. Отличий два: вычитается
// среднее (значит нужна вторая редукция по строке) и есть свободный член.
// Смысл сравнения в том, чтобы увидеть, даёт ли центрирование что-нибудь —
// современные модели от него отказались.
Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  float eps);

void layer_norm_backward(const Tensor& grad_output, const Tensor& input,
                         const Tensor& weight, float eps, Tensor* grad_input,
                         Tensor* grad_weight, Tensor* grad_bias);

// Softmax по последней оси. Вычитание максимума строки обязательно: без него
// exp переполняется уже на логитах порядка 90.
Tensor softmax(const Tensor& input);

// Обратный проход softmax: dx_i = y_i * (g_i - sum_j g_j y_j).
// Нужен выход, а не вход: сохранять y дешевле, чем пересчитывать.
Tensor softmax_backward(const Tensor& grad_output, const Tensor& output);

// SiLU (он же swish): x * sigmoid(x).
Tensor silu(const Tensor& input);
Tensor silu_backward(const Tensor& grad_output, const Tensor& input);

// GELU в приближении через tanh — то, что реально считают под именем GELU.
// Нужен для сравнения с SwiGLU в M7.
Tensor gelu(const Tensor& input);
Tensor gelu_backward(const Tensor& grad_output, const Tensor& input);

// --- Диагностические величины ---
//
// Считаются только по запросу: в обычном обучении лишние редукции не нужны.

// Средняя относительная энтропия строк матрицы внимания.
//
// Энтропия строки делится на максимально возможную для неё, то есть на
// логарифм числа доступных ей ключей. Единица означает равномерное внимание —
// голова ничего не выбирает; ноль — предельно сосредоточенное. Нормировка
// нужна потому, что ранние позиции видят меньше ключей, и без неё величина
// говорила бы о длине контекста, а не о поведении головы.
//
// weights: (матрицы, запросы, ключи) — выход softmax после каузальной маски.
float attention_entropy(const Tensor& weights, int64_t query_offset);

// Доля строго положительных значений. Для вентиля SwiGLU это доля открытых
// каналов.
float positive_fraction(const Tensor& input);

// Корень из среднего квадрата — масштаб активаций.
float root_mean_square(const Tensor& input);

// Каузальная маска на последние две оси (запросы x ключи): позиция запроса не
// должна видеть ключи из будущего. query_offset — номер первого запроса, он
// ненулевой при генерации с KV-кэшем, когда запрос один, а ключей уже много.
//
// Замаскированные позиции получают минус бесконечность: после exp это точный
// нуль, без подбора «достаточно большого» отрицательного числа.
Tensor causal_mask(const Tensor& scores, int64_t query_offset);
Tensor causal_mask_backward(const Tensor& grad_output, int64_t query_offset);

}  // namespace ops
}  // namespace llm

#endif  // LLM_OPS_NN_H_
