#include "train/optimizer.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "autograd/node.h"
#include "core/check.h"
#include "ops/parallel.h"
#include "ops/row_reduce.h"

namespace llm {
namespace train {

AdamW::AdamW(std::vector<nn::NamedParameter> parameters,
             const AdamWConfig& config)
    : parameters_(std::move(parameters)), config_(config), step_(0) {
  // Настройки проверяются здесь: дальше они попадают в арифметику шага, где
  // неверное значение не падает, а выдаёт NaN или бесконечность.
  //
  // Единица в любом из бета даёт нулевую поправку на смещение — деление на
  // нуль на каждом шаге. Отрицательный beta2 умеет сделать оценку второго
  // момента отрицательной, а корень из неё — NaN. Нулевой eps оставляет
  // деление на корень без нижней границы: параметр с нулевым градиентом даёт
  // 0/0.
  LLM_CHECK_MSG(config_.beta1 >= 0.0f && config_.beta1 < 1.0f,
                "beta1 " << config_.beta1 << " вне [0, 1)");
  LLM_CHECK_MSG(config_.beta2 >= 0.0f && config_.beta2 < 1.0f,
                "beta2 " << config_.beta2 << " вне [0, 1)");
  LLM_CHECK_MSG(config_.eps > 0.0f,
                "eps " << config_.eps << " должен быть положительным");
  LLM_CHECK_MSG(config_.weight_decay >= 0.0f,
                "распад веса " << config_.weight_decay
                               << " отрицателен: он стал бы раздуванием");

  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    // Обучаемость проверяется здесь, а не там, где она понадобится.
    // У замороженного параметра нет узла графа, и обрезка нормы обратилась бы
    // к пустому указателю: сначала grad() бросил бы «запрошен градиент
    // переменной, которая его не требует», а строкой ниже scale_grad просто
    // разыменовал бы ноль. Сообщение оттуда не сказало бы, что виноват состав
    // списка, переданного оптимизатору.
    //
    // Тренер передаёт trainable_parameters и под это условие подходит всегда;
    // а вот parameters() у модели с адаптерами LoRA содержит замороженные
    // базовые веса, и передать его сюда — ошибка вызывающего.
    LLM_CHECK_MSG(parameters_[i].value->requires_grad(),
                  "параметр " << parameters_[i].name
                              << " не требует градиента: оптимизатору нужен "
                                 "список обучаемых, а не всех");
    const Tensor& value = parameters_[i].value->value();
    LLM_CHECK_MSG(value.is_contiguous(),
                  "параметр " << parameters_[i].name << " размещён неплотно");
    first_moment_.push_back(Tensor::zeros(value.shape()));
    second_moment_.push_back(Tensor::zeros(value.shape()));
    // Распад веса — только для матриц. Одномерные параметры (масштабы
    // нормировок) от него освобождены.
    apply_decay_.push_back(value.rank() >= 2);
  }
}

void AdamW::restore(std::vector<Tensor> first, std::vector<Tensor> second,
                    int64_t step) {
  LLM_CHECK_MSG(
      first.size() == parameters_.size() && second.size() == parameters_.size(),
      "в состоянии " << first.size() << " моментов, а параметров "
                     << parameters_.size());
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const Shape& expected = parameters_[i].value->value().shape();
    LLM_CHECK_MSG(
        first[i].shape() == expected && second[i].shape() == expected,
        "у параметра " << parameters_[i].name << " момент другой формы");
    // Шаг читает моменты как numel() значений подряд. Снимок отдаёт плотные,
    // но restore объявлен в заголовке, и вид на чужой буфер здесь означал бы
    // чтение мимо.
    LLM_CHECK_MSG(
        first[i].is_contiguous() && second[i].is_contiguous(),
        "у параметра " << parameters_[i].name << " момент размещён неплотно");
  }
  LLM_CHECK_GE(step, static_cast<int64_t>(0));
  first_moment_ = std::move(first);
  second_moment_ = std::move(second);
  step_ = step;
}

int64_t AdamW::decayed_parameter_count() const {
  int64_t total = 0;
  for (std::size_t i = 0; i < apply_decay_.size(); ++i) {
    if (apply_decay_[i]) {
      total += parameters_[i].value->numel();
    }
  }
  return total;
}

// Сумма квадратов одного градиента. Порядок фиксирован дважды: внутри части —
// восемью накопителями row_sum_squares, между частями — по номерам частей.
// Ни то, ни другое не зависит от числа потоков, поэтому норма получается
// побитово одинаковой на любой машине.
double AdamW::sum_squares(const Tensor& tensor) {
  // Читается numel() значений подряд, то есть тензор обязан быть плотным.
  // Оптимизатор получает только параметры и их градиенты, а те плотны всегда —
  // проверка стоит здесь затем, чтобы это было сказано, а не подразумевалось.
  LLM_DCHECK(tensor.is_contiguous());
  const float* data = tensor.data();
  double parts[ops::kSumParts] = {};
  ops::for_row_parts(
      tensor.numel(), 1, [&](int64_t part, int64_t first, int64_t last) {
        parts[part] = ops::row_sum_squares(data + first, last - first);
      });
  double total = 0.0;
  for (int part = 0; part < ops::kSumParts; ++part) {
    total += parts[part];
  }
  return total;
}

float AdamW::clip_grad_norm(float max_norm) {
  // Отрицательный предел давал бы отрицательный множитель, то есть разворот
  // всех градиентов: шаг пошёл бы ВВЕРХ по функции потерь. Нулевой обнулял бы
  // их целиком, и обучение молча вставало бы. Ни то, ни другое не «обрезка
  // выключена» — выключают её большим пределом.
  LLM_CHECK_MSG(max_norm > 0.0f, "предел нормы градиента "
                                     << max_norm
                                     << " должен быть положительным");
  double sum_squares = 0.0;
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const Tensor& grad = parameters_[i].value->grad();
    if (!grad.defined()) {
      continue;
    }
    sum_squares += AdamW::sum_squares(grad);
  }
  const double norm = std::sqrt(sum_squares);
  if (norm <= static_cast<double>(max_norm) || norm == 0.0) {
    return static_cast<float>(norm);
  }

  const float scale = static_cast<float>(static_cast<double>(max_norm) / norm);
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    parameters_[i].value->node()->scale_grad(scale);
  }
  return static_cast<float>(norm);
}

void AdamW::step(float learning_rate) {
  ++step_;

  // Поправка на смещение. Моменты стартуют с нуля, поэтому первые оценки
  // занижены; деление на (1 - beta^t) это компенсирует. Без поправки первые
  // шаги были бы во много раз короче нужного.
  const double bias1 = 1.0 - std::pow(static_cast<double>(config_.beta1),
                                      static_cast<double>(step_));
  const double bias2 = 1.0 - std::pow(static_cast<double>(config_.beta2),
                                      static_cast<double>(step_));

  update_ratios_.assign(parameters_.size(), 0.0f);

  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    const Tensor& grad = parameters_[i].value->grad();
    if (!grad.defined()) {
      continue;  // параметр не участвовал в этом шаге
    }

    Tensor& value = parameters_[i].value->value();
    LLM_DCHECK(grad.shape() == value.shape());

    float* weights = value.data();
    const float* gradients = grad.data();
    float* first = first_moment_[i].data();
    float* second = second_moment_[i].data();
    const int64_t count = value.numel();

    // Длины шага и самого веса копятся по ходу: отдельный проход ради них
    // стоил бы столько же, сколько сам шаг.
    //
    // Работа делится на фиксированные части — те же, что у сумм в ops. Части
    // считаются разными потоками, а складываются по номерам, поэтому и сами
    // веса, и отношение шага к весу не зависят от числа ядер.
    double update_parts[ops::kSumParts] = {};
    double weight_parts[ops::kSumParts] = {};

    ops::for_row_parts(count, 1, [&](int64_t part, int64_t begin, int64_t end) {
      double update_squares = 0.0;
      double weight_squares = 0.0;

      for (int64_t element = begin; element < end; ++element) {
        const float gradient = gradients[element];
        weight_squares +=
            static_cast<double>(weights[element]) * weights[element];

        first[element] =
            config_.beta1 * first[element] + (1.0f - config_.beta1) * gradient;
        second[element] = config_.beta2 * second[element] +
                          (1.0f - config_.beta2) * gradient * gradient;

        const double corrected_first =
            static_cast<double>(first[element]) / bias1;
        const double corrected_second =
            static_cast<double>(second[element]) / bias2;

        // Распад применяется к самому весу, а не к градиенту: в этом и состоит
        // отличие AdamW от Adam.
        if (apply_decay_[i] && config_.weight_decay != 0.0f) {
          weights[element] -=
              learning_rate * config_.weight_decay * weights[element];
        }

        const double update =
            corrected_first /
            (std::sqrt(corrected_second) + static_cast<double>(config_.eps));
        const double step = static_cast<double>(learning_rate) * update;
        update_squares += step * step;
        weights[element] -= static_cast<float>(step);
      }
      update_parts[part] = update_squares;
      weight_parts[part] = weight_squares;
    });

    double update_squares = 0.0;
    double weight_squares = 0.0;
    for (int part = 0; part < ops::kSumParts; ++part) {
      update_squares += update_parts[part];
      weight_squares += weight_parts[part];
    }

    // Пессимистичная имитация: веса тоже уходят в половинную разрядность, то
    // есть эталонной копии в fp32 не остаётся вовсе. Если обучение переживает
    // и это, то реализация не потребует ни эталонных весов, ни масштабирования
    // потерь — а это втрое меньше работы.
    //
    // Отдельный флаг, но он лишь добавляется к LLM_FP16: округление внутри
    // apply_fp16_simulation выключено, пока не включён основной режим. Иначе
    // получился бы бессмысленный опыт — точные активации при грубых весах.
    //
    // Округление идёт после шага, а не до: моменты и отношение шага к весу
    // считаются по тем значениям, которые оптимизатор и получил бы, а в память
    // ложится уже усечённое — ровно как при настоящем хранении в fp16.
    if (autograd::fp16_weight_simulation()) {
      autograd::apply_fp16_simulation(&value);
    }

    update_ratios_[i] = weight_squares > 0.0
                            ? static_cast<float>(std::sqrt(update_squares) /
                                                 std::sqrt(weight_squares))
                            : 0.0f;
  }

  // Медиана, а не среднее: у одного-двух параметров отношение бывает на
  // порядки больше, и среднее говорило бы только о них.
  std::vector<float> sorted;
  for (std::size_t i = 0; i < update_ratios_.size(); ++i) {
    if (update_ratios_[i] > 0.0f) {
      sorted.push_back(update_ratios_[i]);
    }
  }
  if (sorted.empty()) {
    last_update_ratio_ = 0.0f;
  } else {
    std::sort(sorted.begin(), sorted.end());
    last_update_ratio_ = sorted[sorted.size() / 2];
  }
}

void AdamW::zero_grad() {
  for (std::size_t i = 0; i < parameters_.size(); ++i) {
    parameters_[i].value->zero_grad();
  }
}

}  // namespace train
}  // namespace llm
