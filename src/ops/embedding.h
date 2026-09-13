// Таблица эмбеддингов: выбор строк по индексам токенов.
//
// Идентификаторы токенов — целые, а тензоры в проекте хранят float, поэтому
// индексы передаются обычным вектором. Это не компромисс: обобщать тензор по
// типу элемента ради единственного места, где нужны целые, дороже, чем
// провести их рядом.

#ifndef LLM_OPS_EMBEDDING_H_
#define LLM_OPS_EMBEDDING_H_

#include <cstdint>
#include <vector>

#include "core/tensor.h"

namespace llm {
namespace ops {

// weight: (vocab, dim). Результат: (ids.size(), dim).
// Форму с батчем восстанавливает вызывающий через reshape.
Tensor embedding(const Tensor& weight, const std::vector<int32_t>& ids);

// Обратный проход — рассеивающее сложение: строка веса получает сумму
// градиентов всех позиций, где встретился её токен. Именно поэтому здесь
// сложение, а не присваивание: частый токен встречается в батче десятки раз.
Tensor embedding_backward(const Tensor& grad_output,
                          const std::vector<int32_t>& ids, int64_t vocab_size);

}  // namespace ops
}  // namespace llm

#endif  // LLM_OPS_EMBEDDING_H_
