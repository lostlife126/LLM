// Перекрёстная энтропия от логитов.
//
// Слитая операция: принимает логиты, а не вероятности. Это не оптимизация ради
// скорости, а вопрос точности и памяти.
//
// Наивная сборка выглядела бы как -log(softmax(logits)[target]). Softmax уже
// поделил на сумму экспонент, а логарифм тут же это деление разворачивает —
// два перехода через экспоненту там, где хватает одного. Хуже того,
// вероятность редкого токена легко уходит в денормализованные числа, и
// логарифм от неё теряет точность. Через logsumexp этого не происходит.
//
// Обратный проход и вовсе сворачивается в одно выражение: softmax(logits)
// минус единица в позиции правильного токена. Промежуточных тензоров нет
// совсем.

#ifndef LLM_OPS_LOSS_H_
#define LLM_OPS_LOSS_H_

#include <cstdint>
#include <vector>

#include "core/tensor.h"

namespace llm {
namespace ops {

// logits: (n, vocab), targets: n индексов. Возвращает скаляр — среднее по n.
Tensor cross_entropy(const Tensor& logits, const std::vector<int32_t>& targets);

// Градиент по логитам: (softmax(logits) - onehot(targets)) / n.
Tensor cross_entropy_backward(const Tensor& logits,
                              const std::vector<int32_t>& targets);

}  // namespace ops
}  // namespace llm

#endif  // LLM_OPS_LOSS_H_
