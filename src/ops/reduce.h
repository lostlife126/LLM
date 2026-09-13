// Суммирование и усреднение по осям.
//
// Редукции нужны не только сами по себе: обратный проход любой операции с
// растяжением — это суммирование градиента по тем осям, которые растяжение
// размножило. Если при прямом проходе один вес прибавился к ста позициям, то
// в обратном на него приходит сумма ста градиентов.

#ifndef LLM_OPS_REDUCE_H_
#define LLM_OPS_REDUCE_H_

#include <vector>

#include "core/tensor.h"

namespace llm {
namespace ops {

// keepdim сохраняет сокращённые оси размера 1. Это удобно для последующего
// растяжения обратно: (b, t, 1) растягивается до (b, t, d), а (b, t) — нет.
Tensor sum(const Tensor& input, const std::vector<int>& axes, bool keepdim);
Tensor mean(const Tensor& input, const std::vector<int>& axes, bool keepdim);

// Свёртка всего тензора в скаляр ранга 0.
Tensor sum_all(const Tensor& input);
Tensor mean_all(const Tensor& input);

// Приводит градиент к форме target, суммируя по осям, размноженным при
// растяжении. Обратный проход любой бинарной операции с broadcast сводится
// к этому вызову.
Tensor reduce_to_shape(const Tensor& grad, const Shape& target);

}  // namespace ops
}  // namespace llm

#endif  // LLM_OPS_REDUCE_H_
