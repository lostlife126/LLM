#include "ops/embedding.h"

#include <algorithm>

#include <cstring>

#include "core/check.h"
#include "ops/parallel.h"

namespace llm {
namespace ops {

Tensor embedding(const Tensor& weight, const std::vector<int32_t>& ids) {
  LLM_CHECK_MSG(weight.rank() == 2, "таблица эмбеддингов должна быть матрицей");
  LLM_CHECK(weight.is_contiguous());
  const int64_t vocab = weight.dim(0);
  const int64_t dim = weight.dim(1);

  Tensor out =
      Tensor::uninitialized(Shape({static_cast<int64_t>(ids.size()), dim}));
  const float* table = weight.data();
  float* result = out.data();

  for (std::size_t position = 0; position < ids.size(); ++position) {
    const int64_t id = ids[position];
    LLM_CHECK_MSG(id >= 0 && id < vocab,
                  "токен " << id << " вне словаря размера " << vocab);
    std::memcpy(result + static_cast<int64_t>(position) * dim, table + id * dim,
                static_cast<std::size_t>(dim) * sizeof(float));
  }
  return out;
}

Tensor embedding_backward(const Tensor& grad_output,
                          const std::vector<int32_t>& ids, int64_t vocab_size) {
  LLM_CHECK_MSG(grad_output.rank() == 2,
                "градиент эмбеддинга должен быть матрицей");
  LLM_CHECK_EQ(grad_output.dim(0), static_cast<int64_t>(ids.size()));
  const int64_t dim = grad_output.dim(1);

  const Tensor dense = grad_output.contiguous();
  const float* grad = dense.data();

  Tensor out = Tensor::zeros(Shape({vocab_size, dim}));
  float* table = out.data();

  for (std::size_t position = 0; position < ids.size(); ++position) {
    LLM_CHECK_MSG(ids[position] >= 0 && ids[position] < vocab_size,
                  "токен " << ids[position] << " вне словаря размера "
                           << vocab_size);
  }

  // Делится по столбцам таблицы, а не по позициям, и это единственное деление,
  // которое здесь работает. Два токена в батче могут оказаться одинаковыми, и
  // тогда по позициям потоки писали бы в одну и ту же строку. По столбцам
  // ячейки у потоков не пересекаются вовсе, а порядок сложения внутри столбца
  // остаётся порядком позиций — то есть результат не зависит от числа ядер.
  // Зерно в столбцах: работы на столбец ровно столько, сколько позиций.
  const int64_t grain = std::max<int64_t>(
      1, kElementGrain / std::max<int64_t>(static_cast<int64_t>(ids.size()), 1));
  parallel_range(dim, grain, [&](int64_t first, int64_t last) {
    for (std::size_t position = 0; position < ids.size(); ++position) {
      const float* source = grad + static_cast<int64_t>(position) * dim;
      float* target = table + ids[position] * dim;
      for (int64_t i = first; i < last; ++i) {
        target[i] += source[i];
      }
    }
  });
  return out;
}

}  // namespace ops
}  // namespace llm
