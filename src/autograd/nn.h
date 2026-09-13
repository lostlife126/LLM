// Дифференцируемые операции модели.
//
// Отделены от autograd/ops.h по смыслу: там общематематические операции, здесь
// то, из чего собран именно трансформер. У большинства из них обратный проход
// сделан отдельным ядром в ops/, а не собран из примитивов, поэтому обвязка
// здесь тоньше — она только заводит узел и передаёт сохранённые значения.

#ifndef LLM_AUTOGRAD_NN_H_
#define LLM_AUTOGRAD_NN_H_

#include <cstdint>
#include <vector>

#include "autograd/node.h"

namespace llm {
namespace autograd {

// Нормировка по последней оси с обучаемым масштабом.
Var rms_norm(const Var& input, const Var& weight, float eps);

// Нормировка с центрированием и свободным членом — для сравнения с RMSNorm.
Var layer_norm(const Var& input, const Var& weight, const Var& bias, float eps);

Var softmax(const Var& input);
Var silu(const Var& input);
Var gelu(const Var& input);

// Маска «не смотреть в будущее» на последние две оси.
Var causal_mask(const Var& scores, int64_t query_offset);

// Выбор строк таблицы по индексам токенов. Результат: (ids.size(), dim).
Var embedding(const Var& weight, const std::vector<int32_t>& ids);

// Поворотное кодирование позиции. Обучаемых весов нет, но градиент через него
// проходит — и поворачивается обратно.
Var rope(const Var& input, int64_t position_offset, float theta);

// Функция потерь: скаляр, от которого идёт обратный проход всего обучения.
Var cross_entropy(const Var& logits, const std::vector<int32_t>& targets);

// Вспомогательные потери, удерживающие логиты от дрейфа по абсолютной
// величине.
Var z_loss(const Var& logits);

}  // namespace autograd
}  // namespace llm

#endif  // LLM_AUTOGRAD_NN_H_
