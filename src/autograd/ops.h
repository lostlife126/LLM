// Дифференцируемые операции над Var.
//
// Каждая функция здесь делает три вещи: считает значение ядром из ops/,
// собирает замыкание обратного прохода и заводит узел ленты. Если градиент не
// нужен — ни один вход его не требует или включён NoGradGuard — узел не
// создаётся, и Var становится обычным значением.
//
// Обратный проход выражен через те же ядра, что и прямой: производная
// произведения — снова произведение, производная суммы с растяжением — сумма
// по размноженным осям. Отдельные ядра для backward появляются только там, где
// это выигрыш по памяти или точности.

#ifndef LLM_AUTOGRAD_OPS_H_
#define LLM_AUTOGRAD_OPS_H_

#include <vector>

#include "autograd/node.h"

namespace llm {
namespace autograd {

// --- Поэлементные, с растяжением по правилам numpy ---
Var add(const Var& a, const Var& b);
Var sub(const Var& a, const Var& b);
Var mul(const Var& a, const Var& b);
Var div(const Var& a, const Var& b);

Var add_scalar(const Var& input, float scalar);
Var mul_scalar(const Var& input, float scalar);
Var neg(const Var& input);
Var exp(const Var& input);
Var log(const Var& input);
Var sqrt(const Var& input);

// --- Редукции ---
Var sum(const Var& input, const std::vector<int>& axes, bool keepdim);
Var mean(const Var& input, const std::vector<int>& axes, bool keepdim);
Var sum_all(const Var& input);
Var mean_all(const Var& input);

// --- Матричное умножение ---
Var matmul(const Var& a, const Var& b);

// --- Изменение формы: значения те же, меняется только их описание ---
Var reshape(const Var& input, const Shape& shape);

// Растягивает оси размера 1 до нужного размера. Обратный проход — суммирование
// по размноженным осям: если элемент поучаствовал в ста позициях, обратно на
// него приходит сумма ста градиентов.
Var expand(const Var& input, const Shape& shape);
Var transpose(const Var& input, int axis_a, int axis_b);
Var permute(const Var& input, const std::vector<int>& order);
Var slice(const Var& input, int axis, int64_t start, int64_t count);
Var cat(const std::vector<Var>& parts, int axis);

}  // namespace autograd
}  // namespace llm

#endif  // LLM_AUTOGRAD_OPS_H_
