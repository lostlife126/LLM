// Поэлементные операции.
//
// Бинарные операции растягивают аргументы по правилам numpy: оси
// выравниваются справа, ось размера 1 размножается. Растяжение выражается
// шагом 0 и не копирует данные.
//
// Здесь только прямой проход. Обратный собран в autograd/ops.cpp из этих же
// кирпичей: производная произведения — это снова произведение, производная
// экспоненты — она сама. Отдельные ядра для обратного прохода понадобились бы
// только там, где это выгодно по памяти или скорости — например, в softmax и
// cross_entropy.

#ifndef LLM_OPS_ELEMENTWISE_H_
#define LLM_OPS_ELEMENTWISE_H_

#include "core/tensor.h"

namespace llm {
namespace ops {

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

Tensor add_scalar(const Tensor& input, float scalar);
Tensor mul_scalar(const Tensor& input, float scalar);

Tensor neg(const Tensor& input);
Tensor exp(const Tensor& input);
Tensor log(const Tensor& input);
Tensor sqrt(const Tensor& input);

// Прибавляет source к target на месте. Формы обязаны совпадать; растяжения
// здесь нет намеренно — молчаливое растяжение при накоплении градиента
// скрывало бы ошибку в формах.
void add_into(const Tensor& source, Tensor* target);

}  // namespace ops
}  // namespace llm

#endif  // LLM_OPS_ELEMENTWISE_H_
