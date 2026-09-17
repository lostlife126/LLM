#include "ops/rope.h"

#include <cmath>
#include <vector>

#include "core/check.h"
#include "ops/parallel.h"

namespace llm {
namespace ops {
namespace {

Tensor apply_rotation(const Tensor& input, int64_t position_offset, float theta,
                      bool inverse) {
  LLM_CHECK_MSG(input.rank() >= 2, "RoPE нужны оси позиции и координаты");
  const int64_t head_dim = input.dim(-1);
  const int64_t sequence = input.dim(-2);
  LLM_CHECK_MSG(head_dim % 2 == 0,
                "размерность головы " << head_dim << " должна быть чётной");

  const Tensor dense = input.contiguous();
  const float* source = dense.data();
  Tensor out = Tensor::uninitialized(input.shape());
  float* result = out.data();

  // Пустой тензор — не ошибка, но и работы с ним нет. Проверка стоит до
  // подсчёта блоков, потому что делится там на sequence * head_dim, а пустым
  // тензор бывает как раз тогда, когда один из множителей нулевой: у формы
  // (3, 0) чётность головы соблюдена, и деление уходило в ноль — не в отказ с
  // сообщением, а в SIGFPE.
  if (input.numel() == 0) {
    return out;
  }
  const int64_t pairs = head_dim / 2;
  const int64_t blocks = input.numel() / (sequence * head_dim);

  // Синусы и косинусы считаются один раз на (позицию, пару), а не на каждый
  // элемент.
  //
  // Это не микрооптимизация. Углы зависят только от позиции в
  // последовательности и номера пары координат, а блоков — batch * heads, то
  // есть шестьдесят четыре у пресета nano. Без таблицы pow, cos и sin
  // вызывались шестьдесят четыре раза на один и тот же угол; pow вообще
  // вызывался на каждый элемент, хотя частота зависит лишь от номера пары.
  std::vector<float> cosines(static_cast<std::size_t>(sequence * pairs));
  std::vector<float> sines(cosines.size());
  for (int64_t pair = 0; pair < pairs; ++pair) {
    const double exponent =
        -2.0 * static_cast<double>(pair) / static_cast<double>(head_dim);
    const double frequency = std::pow(static_cast<double>(theta), exponent);
    for (int64_t step = 0; step < sequence; ++step) {
      const double angle =
          static_cast<double>(step + position_offset) * frequency;
      const std::size_t index = static_cast<std::size_t>(step * pairs + pair);
      cosines[index] = static_cast<float>(std::cos(angle));
      sines[index] =
          static_cast<float>(inverse ? -std::sin(angle) : std::sin(angle));
    }
  }

  // Блоки независимы, и каждый из них — batch * heads штук; это и делит работу.
  for_rows(blocks, sequence * head_dim, [&](int64_t block) {
    for (int64_t step = 0; step < sequence; ++step) {
      const int64_t base = (block * sequence + step) * head_dim;
      const float* cosine_row = cosines.data() + step * pairs;
      const float* sine_row = sines.data() + step * pairs;
      for (int64_t pair = 0; pair < pairs; ++pair) {
        const float cosine = cosine_row[pair];
        const float sine = sine_row[pair];
        const float even = source[base + 2 * pair];
        const float odd = source[base + 2 * pair + 1];
        result[base + 2 * pair] = even * cosine - odd * sine;
        result[base + 2 * pair + 1] = even * sine + odd * cosine;
      }
    }
  });
  return out;
}

}  // namespace

Tensor rope(const Tensor& input, int64_t position_offset, float theta) {
  return apply_rotation(input, position_offset, theta, false);
}

Tensor rope_backward(const Tensor& grad_output, int64_t position_offset,
                     float theta) {
  return apply_rotation(grad_output, position_offset, theta, true);
}

}  // namespace ops
}  // namespace llm
