#include "gradcheck.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "autograd/ops.h"
#include "core/check.h"
#include "core/iterate.h"
#include "core/random.h"

namespace llm {
namespace testing {
namespace {

// Значения тензора в порядке плотного размещения. Численная проверка идёт по
// элементам, и удобно обращаться к ним по одному индексу.
std::vector<float> flatten(const Tensor& tensor) {
  std::vector<float> values;
  values.reserve(static_cast<std::size_t>(tensor.numel()));
  const float* data = tensor.data();
  for_each_offset(tensor.shape(), tensor.strides(),
                  [&](int64_t offset) { values.push_back(data[offset]); });
  return values;
}

double evaluate(
    const std::function<autograd::Var(const std::vector<autograd::Var>&)>& fn,
    const std::vector<Tensor>& inputs) {
  // Лента при численной оценке не нужна: считаем только значение.
  autograd::NoGradGuard no_grad;
  std::vector<autograd::Var> vars;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    vars.push_back(autograd::Var::constant(inputs[i]));
  }
  const autograd::Var out = fn(vars);
  LLM_CHECK_MSG(out.numel() == 1,
                "gradcheck: функция обязана вернуть скаляр, получена форма "
                    << out.shape());
  return static_cast<double>(*out.value().data());
}

}  // namespace

Tensor random_tensor(const Shape& shape, std::uint64_t seed, float low,
                     float high) {
  Rng rng(seed);
  Tensor tensor = Tensor::uninitialized(shape);
  Span<float> values = tensor.flat();
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = rng.uniform(low, high);
  }
  return tensor;
}

autograd::Var weighted_sum(const autograd::Var& output, std::uint64_t seed) {
  const autograd::Var weights =
      autograd::Var::constant(random_tensor(output.shape(), seed));
  return autograd::sum_all(autograd::mul(output, weights));
}

GradCheckResult gradcheck(
    const std::function<autograd::Var(const std::vector<autograd::Var>&)>& fn,
    const std::vector<Tensor>& inputs, float step, double tolerance) {
  GradCheckResult result;

  // Аналитические градиенты: один прямой и один обратный проход.
  std::vector<autograd::Var> vars;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    vars.push_back(autograd::Var::leaf(inputs[i].clone(), true));
  }
  autograd::Var out = fn(vars);
  LLM_CHECK_MSG(out.numel() == 1,
                "gradcheck: функция обязана вернуть скаляр, получена форма "
                    << out.shape());
  out.backward();

  std::ostringstream detail;
  double worst = 0.0;

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const Tensor& analytic_tensor = vars[i].grad();
    LLM_CHECK_MSG(analytic_tensor.defined(),
                  "gradcheck: вход " << i
                                     << " не получил градиента — он не влияет "
                                        "на результат?");
    LLM_CHECK_MSG(analytic_tensor.shape() == inputs[i].shape(),
                  "gradcheck: градиент входа "
                      << i << " имеет форму " << analytic_tensor.shape()
                      << ", а сам вход — " << inputs[i].shape());

    const std::vector<float> analytic = flatten(analytic_tensor);

    // Масштаб для относительного сравнения берётся по всему тензору, а не по
    // элементу: у отдельного элемента градиент может быть близок к нулю, и
    // деление на него превратило бы шум в «ошибку».
    double scale = 1.0;
    for (std::size_t j = 0; j < analytic.size(); ++j) {
      scale = std::max(scale, std::fabs(static_cast<double>(analytic[j])));
    }

    std::vector<Tensor> perturbed;
    for (std::size_t k = 0; k < inputs.size(); ++k) {
      perturbed.push_back(inputs[k].clone());
    }
    Tensor& target = perturbed[i];
    float* data = target.data();

    for (int64_t element = 0; element < target.numel(); ++element) {
      const float original = data[element];

      data[element] = original + step;
      const double plus = evaluate(fn, perturbed);
      data[element] = original - step;
      const double minus = evaluate(fn, perturbed);
      data[element] = original;

      const double numeric = (plus - minus) / (2.0 * static_cast<double>(step));
      const double error =
          std::fabs(numeric -
                    static_cast<double>(
                        analytic[static_cast<std::size_t>(element)])) /
          scale;

      if (error > worst) {
        worst = error;
        detail.str(std::string());
        detail << "вход " << i << ", элемент " << element
               << ": обратный проход дал "
               << analytic[static_cast<std::size_t>(element)]
               << ", численная оценка " << numeric
               << ", относительное расхождение " << error << " при допуске "
               << tolerance;
      }
    }
  }

  result.max_error = worst;
  result.ok = worst <= tolerance;
  result.detail = detail.str();
  return result;
}

}  // namespace testing
}  // namespace llm
