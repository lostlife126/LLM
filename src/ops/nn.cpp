#include "ops/nn.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "core/check.h"
#include "ops/fast_exp.h"
#include "ops/parallel.h"
#include "ops/row_reduce.h"

namespace llm {
namespace ops {
namespace {

// Число строк и длина строки для операций по последней оси.
void row_layout(const Tensor& tensor, int64_t* rows, int64_t* width) {
  LLM_CHECK_MSG(tensor.rank() >= 1,
                "операция по последней оси требует ранга >= 1");
  *width = tensor.dim(-1);
  *rows = *width == 0 ? 0 : tensor.numel() / *width;
}

const float* dense_data(const Tensor& tensor, Tensor* holder) {
  if (tensor.is_contiguous()) {
    return tensor.data();
  }
  *holder = tensor.contiguous();
  return holder->data();
}

// Множитель sqrt(2/pi) из приближения GELU.
const float kGeluCoefficient = 0.7978845608028654f;
const float kGeluCubic = 0.044715f;

}  // namespace

Tensor rms_norm(const Tensor& input, const Tensor& weight, float eps) {
  int64_t rows = 0;
  int64_t width = 0;
  row_layout(input, &rows, &width);
  LLM_CHECK_MSG(weight.rank() == 1 && weight.dim(0) == width,
                "вес RMSNorm формы " << weight.shape()
                                     << " не подходит к входу "
                                     << input.shape());

  Tensor input_holder;
  Tensor weight_holder;
  const float* x = dense_data(input, &input_holder);
  const float* w = dense_data(weight, &weight_holder);

  Tensor out = Tensor::uninitialized(input.shape());
  float* y = out.data();

  // Строки независимы: у каждой своя нормировка. Это и делит работу.
  for_rows(rows, width, [&](int64_t row) {
    const float* x_row = x + row * width;
    float* y_row = y + row * width;

    const double sum_squares = row_sum_squares(x_row, width);
    const float scale = static_cast<float>(
        1.0 / std::sqrt(sum_squares / static_cast<double>(width) + eps));

    for (int64_t i = 0; i < width; ++i) {
      y_row[i] = x_row[i] * scale * w[i];
    }
  });
  return out;
}

void rms_norm_backward(const Tensor& grad_output, const Tensor& input,
                       const Tensor& weight, float eps, Tensor* grad_input,
                       Tensor* grad_weight) {
  LLM_CHECK(grad_input != nullptr && grad_weight != nullptr);
  int64_t rows = 0;
  int64_t width = 0;
  row_layout(input, &rows, &width);

  Tensor grad_holder;
  Tensor input_holder;
  Tensor weight_holder;
  const float* g = dense_data(grad_output, &grad_holder);
  const float* x = dense_data(input, &input_holder);
  const float* w = dense_data(weight, &weight_holder);

  // Градиент по входу заполняется целиком — каждая строка, каждый элемент, —
  // поэтому обнулять его незачем. Градиент веса, наоборот, накапливается по
  // частям и обязан начинаться с нуля.
  *grad_input = Tensor::uninitialized(input.shape());
  *grad_weight = Tensor::zeros(weight.shape());
  float* dx = grad_input->data();
  float* dw = grad_weight->data();

  // Градиент по входу считается строка за строкой и делится свободно, а
  // градиент веса — сумма по всем строкам, и её приходится собирать по
  // частям: у каждой части свой накопитель, иначе потоки писали бы в один и
  // тот же массив.
  std::vector<double> partial_dw(static_cast<std::size_t>(kSumParts * width),
                                 0.0);

  for_row_parts(rows, width, [&](int64_t part, int64_t first, int64_t last) {
    double* dw_part = partial_dw.data() + part * width;
    for (int64_t row = first; row < last; ++row) {
      const float* x_row = x + row * width;
      const float* g_row = g + row * width;
      float* dx_row = dx + row * width;

      const double sum_squares = row_sum_squares(x_row, width);
      const double mean_square = sum_squares / static_cast<double>(width);
      const double scale = 1.0 / std::sqrt(mean_square + eps);

      // Вес умножается на нормированный вход, поэтому его градиент — сумма по
      // всем строкам от g * x * scale.
      for (int64_t i = 0; i < width; ++i) {
        dw_part[i] += g_row[i] * static_cast<float>(x_row[i] * scale);
      }

      // Производная по входу. Пусть h_i = g_i * w_i — градиент по
      // нормированному значению. Тогда, поскольку нормировка зависит от всей
      // строки,
      //   dx_k = scale * h_k - x_k * scale^3 * (sum_i h_i x_i) / width.
      // Первое слагаемое — «прямой» вклад, второе — через общий множитель.
      const double dot = row_dot3(g_row, w, x_row, width);
      const double correction =
          dot * scale * scale * scale / static_cast<double>(width);

      for (int64_t k = 0; k < width; ++k) {
        const double direct = static_cast<double>(g_row[k]) * w[k] * scale;
        dx_row[k] = static_cast<float>(direct - x_row[k] * correction);
      }
    }
  });

  // Частичные суммы складываются в порядке номеров частей — всегда в одном и
  // том же, независимо от того, какой поток какую часть считал.
  for (int64_t part = 0; part < kSumParts; ++part) {
    const double* dw_part = partial_dw.data() + part * width;
    for (int64_t i = 0; i < width; ++i) {
      dw[i] += static_cast<float>(dw_part[i]);
    }
  }
}

Tensor layer_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  float eps) {
  int64_t rows = 0;
  int64_t width = 0;
  row_layout(input, &rows, &width);
  LLM_CHECK_MSG(weight.rank() == 1 && weight.dim(0) == width,
                "вес LayerNorm формы " << weight.shape()
                                       << " не подходит к входу "
                                       << input.shape());
  LLM_CHECK_MSG(bias.rank() == 1 && bias.dim(0) == width,
                "свободный член формы " << bias.shape() << " не подходит");

  Tensor input_holder;
  Tensor weight_holder;
  Tensor bias_holder;
  const float* x = dense_data(input, &input_holder);
  const float* w = dense_data(weight, &weight_holder);
  const float* b = dense_data(bias, &bias_holder);

  Tensor out = Tensor::uninitialized(input.shape());
  float* y = out.data();

  for_rows(rows, width, [&](int64_t row) {
    const float* x_row = x + row * width;
    float* y_row = y + row * width;

    // Две редукции вместо одной у RMSNorm: сначала среднее, потом дисперсия
    // вокруг него.
    const double mean = row_sum(x_row, width) / static_cast<double>(width);
    const double sum_squares = row_sum_centered_squares(x_row, mean, width);
    const double scale =
        1.0 / std::sqrt(sum_squares / static_cast<double>(width) + eps);

    for (int64_t i = 0; i < width; ++i) {
      const double normalized = (static_cast<double>(x_row[i]) - mean) * scale;
      y_row[i] = static_cast<float>(normalized) * w[i] + b[i];
    }
  });
  return out;
}

void layer_norm_backward(const Tensor& grad_output, const Tensor& input,
                         const Tensor& weight, float eps, Tensor* grad_input,
                         Tensor* grad_weight, Tensor* grad_bias) {
  LLM_CHECK(grad_input != nullptr && grad_weight != nullptr &&
            grad_bias != nullptr);
  int64_t rows = 0;
  int64_t width = 0;
  row_layout(input, &rows, &width);

  Tensor grad_holder;
  Tensor input_holder;
  Tensor weight_holder;
  const float* g = dense_data(grad_output, &grad_holder);
  const float* x = dense_data(input, &input_holder);
  const float* w = dense_data(weight, &weight_holder);

  // По входу — заполняется целиком, обнулять незачем; по весу и свободному
  // члену — накапливается, начинать надо с нуля.
  *grad_input = Tensor::uninitialized(input.shape());
  *grad_weight = Tensor::zeros(weight.shape());
  *grad_bias = Tensor::zeros(weight.shape());
  float* dx = grad_input->data();
  float* dw = grad_weight->data();
  float* db = grad_bias->data();

  // Те же две суммы по строкам, что у RMSNorm, плюс свободный член. Порядок
  // суммирования фиксирован числом частей — см. for_row_parts.
  std::vector<double> partial_dw(static_cast<std::size_t>(kSumParts * width),
                                 0.0);
  std::vector<double> partial_db(static_cast<std::size_t>(kSumParts * width),
                                 0.0);

  for_row_parts(rows, width, [&](int64_t part, int64_t first, int64_t last) {
    double* dw_part = partial_dw.data() + part * width;
    double* db_part = partial_db.data() + part * width;
    for (int64_t row = first; row < last; ++row) {
      const float* x_row = x + row * width;
      const float* g_row = g + row * width;
      float* dx_row = dx + row * width;

      const double mean = row_sum(x_row, width) / static_cast<double>(width);
      const double sum_squares = row_sum_centered_squares(x_row, mean, width);
      const double scale =
          1.0 / std::sqrt(sum_squares / static_cast<double>(width) + eps);

      // Свободный член прибавляется как есть, поэтому его градиент — просто
      // сумма по строкам.
      // Градиенты веса и свободного члена и обе поправки по входу считаются
      // одним проходом: всем четырём нужно нормированное значение, а оно
      // стоит вычитания и умножения. Раньше проходов было три, и normalized
      // считалось трижды на каждый элемент.
      //
      // Накопителей по восемь у каждой из двух сумм — по той же причине, что
      // и везде: цепочка сложений иначе упирается в задержку, а порядок
      // остаётся заданным исходником. См. ops/row_reduce.h.
      double sum_h_parts[kRowParts] = {};
      double sum_hn_parts[kRowParts] = {};
      const auto accumulate = [&](int64_t i, int j) {
        const double normalized =
            (static_cast<double>(x_row[i]) - mean) * scale;
        const double h = static_cast<double>(g_row[i]) * w[i];
        db_part[i] += g_row[i];
        dw_part[i] += g_row[i] * static_cast<float>(normalized);
        sum_h_parts[j] += h;
        sum_hn_parts[j] += h * normalized;
      };
      int64_t i = 0;
      for (; i + kRowParts <= width; i += kRowParts) {
        for (int j = 0; j < kRowParts; ++j) {
          accumulate(i + j, j);
        }
      }
      for (int j = 0; i < width; ++i, ++j) {
        accumulate(i, j);
      }
      double sum_h = 0.0;
      double sum_h_normalized = 0.0;
      for (int j = 0; j < kRowParts; ++j) {
        sum_h += sum_h_parts[j];
        sum_h_normalized += sum_hn_parts[j];
      }
      const double mean_h = sum_h / static_cast<double>(width);
      const double mean_h_normalized =
          sum_h_normalized / static_cast<double>(width);

      for (int64_t k = 0; k < width; ++k) {
        const double h = static_cast<double>(g_row[k]) * w[k];
        const double normalized =
            (static_cast<double>(x_row[k]) - mean) * scale;
        dx_row[k] = static_cast<float>(
            scale * (h - mean_h - normalized * mean_h_normalized));
      }
    }
  });

  for (int64_t part = 0; part < kSumParts; ++part) {
    const double* dw_part = partial_dw.data() + part * width;
    const double* db_part = partial_db.data() + part * width;
    for (int64_t i = 0; i < width; ++i) {
      dw[i] += static_cast<float>(dw_part[i]);
      db[i] += static_cast<float>(db_part[i]);
    }
  }
}

Tensor softmax(const Tensor& input) {
  int64_t rows = 0;
  int64_t width = 0;
  row_layout(input, &rows, &width);

  Tensor input_holder;
  const float* x = dense_data(input, &input_holder);

  Tensor out = Tensor::uninitialized(input.shape());
  float* y = out.data();

  for_rows(rows, width, [&](int64_t row) {
    const float* x_row = x + row * width;
    float* y_row = y + row * width;

    // Вычитание максимума не меняет результат математически, но без него exp
    // переполняется: логит 90 уже даёт бесконечность в float.
    const float maximum = row_max(x_row, width);

    // Экспонента считается массивом: поэлементный std::exp — вызов функции
    // из libm, через который автовекторизатор не переступает, и на нём одном
    // раньше уходила заметная часть шага обучения.
    exp_shifted(x_row, maximum, y_row, width);

    // Сумма — с фиксированным числом накопителей (см. ops/row_reduce.h).
    // Порядок задан исходником, а не шириной вектора, поэтому результат
    // одинаков на любом наборе инструкций.
    const float inverse = static_cast<float>(1.0 / row_sum(y_row, width));
    for (int64_t i = 0; i < width; ++i) {
      y_row[i] *= inverse;
    }
  });
  return out;
}

Tensor masked_softmax(const Tensor& scores, float scale, int64_t query_offset) {
  LLM_CHECK_MSG(scores.rank() >= 2, "маске нужны оси запросов и ключей");
  const int64_t keys = scores.dim(-1);
  const int64_t queries = scores.dim(-2);

  Tensor out = Tensor::uninitialized(scores.shape());
  // Пустой тензор — не ошибка, но и работы с ним нет. Проверка стоит до
  // подсчёта матриц: делится там на queries * keys, а пустым тензор бывает
  // ровно тогда, когда один из множителей нулевой. Без неё получался не отказ
  // с сообщением, а SIGFPE. Соседняя attention_entropy эту проверку имеет —
  // здесь её просто забыли.
  if (scores.numel() == 0) {
    return out;
  }
  const int64_t matrices = scores.numel() / (queries * keys);

  Tensor holder;
  const float* x = dense_data(scores, &holder);
  float* y = out.data();

  // Делится по матрицам, а не по строкам, и это не мелочь. Цена строки здесь
  // растёт с номером запроса: маска открывает ему всё больше ключей. Деление
  // подряд идущими кусками отдало бы последнему потоку семь шестнадцатых
  // работы вместо четырёх, и шестикратный выигрыш на операции почти целиком
  // съелся бы перекосом. У матриц профиль одинаковый, поэтому по ним делить
  // ровно.
  for_rows(matrices, queries * keys, [&](int64_t matrix) {
    for (int64_t query = 0; query < queries; ++query) {
      const int64_t row = matrix * queries + query;
      // Сколько ключей видит этот запрос: себя и всё прошлое, но не больше, чем
      // есть ключей.
      const int64_t available = query + query_offset + 1;
      const int64_t limit =
          available < keys ? (available > 0 ? available : 0) : keys;
      const float* x_row = x + row * keys;
      float* y_row = y + row * keys;

      // Умножение на масштаб и поиск максимума — одним проходом. Масштаб
      // положителен, поэтому порядок «умножить, потом взять максимум» и
      // «взять максимум, потом умножить» дали бы одно и то же; выбран первый,
      // потому что умноженные значения всё равно нужны дальше.
      float maximum = -std::numeric_limits<float>::infinity();
      for (int64_t i = 0; i < limit; ++i) {
        y_row[i] = x_row[i] * scale;
        if (y_row[i] > maximum) {
          maximum = y_row[i];
        }
      }

      exp_shifted(y_row, maximum, y_row, limit);
      const float inverse = static_cast<float>(1.0 / row_sum(y_row, limit));
      for (int64_t i = 0; i < limit; ++i) {
        y_row[i] *= inverse;
      }
      // Закрытые позиции: раздельная цепочка ставит туда минус бесконечность и
      // получает после экспоненты нуль. Здесь нуль пишется сразу.
      for (int64_t i = limit; i < keys; ++i) {
        y_row[i] = 0.0f;
      }
    }
  });
  return out;
}

Tensor masked_softmax_backward(const Tensor& grad_output, const Tensor& output,
                               float scale, int64_t query_offset) {
  const int64_t keys = output.dim(-1);
  const int64_t queries = output.dim(-2);

  Tensor out = Tensor::uninitialized(output.shape());
  if (output.numel() == 0) {  // см. masked_softmax
    return out;
  }
  const int64_t matrices = output.numel() / (queries * keys);

  Tensor grad_holder;
  Tensor output_holder;
  const float* g = dense_data(grad_output, &grad_holder);
  const float* y = dense_data(output, &output_holder);

  float* dx = out.data();

  // По матрицам, а не по строкам — по той же причине, что и в прямом проходе.
  for_rows(matrices, queries * keys, [&](int64_t matrix) {
    for (int64_t query = 0; query < queries; ++query) {
      const int64_t row = matrix * queries + query;
      const int64_t available = query + query_offset + 1;
      const int64_t limit =
          available < keys ? (available > 0 ? available : 0) : keys;
      const float* g_row = g + row * keys;
      const float* y_row = y + row * keys;
      float* dx_row = dx + row * keys;

      const double dot = row_dot(g_row, y_row, limit);
      for (int64_t i = 0; i < limit; ++i) {
        // Порядок множителей важен: раздельная цепочка сначала считает
        // производную softmax, потом умножает её на масштаб. Другая расстановка
        // скобок дала бы другое последнее округление.
        dx_row[i] = y_row[i] * (g_row[i] - static_cast<float>(dot)) * scale;
      }
      for (int64_t i = limit; i < keys; ++i) {
        dx_row[i] = 0.0f;
      }
    }
  });
  return out;
}

Tensor softmax_backward(const Tensor& grad_output, const Tensor& output) {
  int64_t rows = 0;
  int64_t width = 0;
  row_layout(output, &rows, &width);

  Tensor grad_holder;
  Tensor output_holder;
  const float* g = dense_data(grad_output, &grad_holder);
  const float* y = dense_data(output, &output_holder);

  Tensor out = Tensor::uninitialized(output.shape());
  float* dx = out.data();

  for_rows(rows, width, [&](int64_t row) {
    const float* g_row = g + row * width;
    const float* y_row = y + row * width;
    float* dx_row = dx + row * width;

    // Якобиан softmax плотный, но его действие на вектор сворачивается в одну
    // скалярную поправку: dx_i = y_i * (g_i - sum_j g_j y_j).
    const double dot = row_dot(g_row, y_row, width);
    for (int64_t i = 0; i < width; ++i) {
      dx_row[i] = y_row[i] * (g_row[i] - static_cast<float>(dot));
    }
  });
  return out;
}

Tensor silu(const Tensor& input) {
  Tensor holder;
  const float* x = dense_data(input, &holder);
  Tensor out = Tensor::uninitialized(input.shape());
  float* y = out.data();

  // Сигмоида считается массивом прямо в результат, а потом домножается на
  // вход. Два прохода по памяти вместо одного — и всё равно втрое быстрее
  // поэлементного std::exp.
  //
  // Границы кусков на результат не влияют: ядро обрабатывает хвост куска тем
  // же кодом, что и середину, через временный буфер. Иначе результат зависел
  // бы от числа потоков, потому что куски режет parallel_range.
  parallel_range(input.numel(), kElementGrain, [&](int64_t begin, int64_t end) {
    sigmoid_array(x + begin, y + begin, end - begin);
    for (int64_t i = begin; i < end; ++i) {
      y[i] *= x[i];
    }
  });
  return out;
}

Tensor silu_backward(const Tensor& grad_output, const Tensor& input) {
  Tensor grad_holder;
  Tensor input_holder;
  const float* g = dense_data(grad_output, &grad_holder);
  const float* x = dense_data(input, &input_holder);
  Tensor out = Tensor::uninitialized(input.shape());
  float* dx = out.data();
  // Сигмоида пишется во временное место — в сам результат: он всё равно будет
  // перезаписан следующей строкой.
  parallel_range(input.numel(), kElementGrain, [&](int64_t begin, int64_t end) {
    sigmoid_array(x + begin, dx + begin, end - begin);
    for (int64_t i = begin; i < end; ++i) {
      const float s = dx[i];
      // d/dx [x * s(x)] = s + x * s * (1 - s)
      dx[i] = g[i] * (s + x[i] * s * (1.0f - s));
    }
  });
  return out;
}

Tensor gelu(const Tensor& input) {
  Tensor holder;
  const float* x = dense_data(input, &holder);
  Tensor out = Tensor::uninitialized(input.shape());
  float* y = out.data();
  for_elements(input.numel(), [&](int64_t i) {
    const float value = x[i];
    const float inner =
        kGeluCoefficient * (value + kGeluCubic * value * value * value);
    y[i] = 0.5f * value * (1.0f + std::tanh(inner));
  });
  return out;
}

Tensor gelu_backward(const Tensor& grad_output, const Tensor& input) {
  Tensor grad_holder;
  Tensor input_holder;
  const float* g = dense_data(grad_output, &grad_holder);
  const float* x = dense_data(input, &input_holder);
  Tensor out = Tensor::uninitialized(input.shape());
  float* dx = out.data();
  for_elements(input.numel(), [&](int64_t i) {
    const float value = x[i];
    const float inner =
        kGeluCoefficient * (value + kGeluCubic * value * value * value);
    const float tanh_inner = std::tanh(inner);
    const float inner_derivative =
        kGeluCoefficient * (1.0f + 3.0f * kGeluCubic * value * value);
    dx[i] = g[i] * (0.5f * (1.0f + tanh_inner) +
                    0.5f * value * (1.0f - tanh_inner * tanh_inner) *
                        inner_derivative);
  });
  return out;
}

float attention_entropy(const Tensor& weights, int64_t query_offset) {
  LLM_CHECK_MSG(weights.rank() >= 2, "матрице внимания нужны две оси");
  const int64_t keys = weights.dim(-1);
  const int64_t queries = weights.dim(-2);
  if (queries == 0 || keys == 0) {
    return 0.0f;
  }
  const int64_t matrices = weights.numel() / (queries * keys);

  Tensor holder;
  const float* data = dense_data(weights, &holder);

  double total = 0.0;
  int64_t counted = 0;
  for (int64_t matrix = 0; matrix < matrices; ++matrix) {
    for (int64_t query = 0; query < queries; ++query) {
      // Сколько ключей доступно этому запросу: он видит себя и всё прошлое.
      const int64_t available = query + query_offset + 1;
      const int64_t limit = available < keys ? available : keys;
      // Проверяется limit, а не available, и это не то же самое. Нормировка
      // ниже делит на log(limit), и при limit == 1 это делённый на нуль нуль,
      // то есть NaN во всей метрике. Условие на available закрывало бы это
      // лишь пока число ключей согласовано со смещением запроса — сейчас оно
      // согласовано, но полагаться на согласованность вызывающего здесь не на
      // что, а NaN в диагностике обнаруживается долго.
      if (limit < 2) {
        // Единственный доступный ключ — энтропия ноль по построению, и делить
        // было бы не на что.
        continue;
      }
      const float* row = data + (matrix * queries + query) * keys;
      double entropy = 0.0;
      for (int64_t key = 0; key < limit; ++key) {
        if (row[key] > 0.0f) {
          entropy -= static_cast<double>(row[key]) * std::log(row[key]);
        }
      }
      total += entropy / std::log(static_cast<double>(limit));
      ++counted;
    }
  }
  return counted == 0 ? 0.0f : static_cast<float>(total / counted);
}

float positive_fraction(const Tensor& input) {
  if (input.numel() == 0) {
    return 0.0f;
  }
  Tensor holder;
  const float* data = dense_data(input, &holder);
  int64_t positive = 0;
  for (int64_t i = 0; i < input.numel(); ++i) {
    if (data[i] > 0.0f) {
      ++positive;
    }
  }
  return static_cast<float>(static_cast<double>(positive) /
                            static_cast<double>(input.numel()));
}

float root_mean_square(const Tensor& input) {
  if (input.numel() == 0) {
    return 0.0f;
  }
  Tensor holder;
  const float* data = dense_data(input, &holder);
  const double total = row_sum_squares(data, input.numel());
  return static_cast<float>(
      std::sqrt(total / static_cast<double>(input.numel())));
}

namespace {

// Позиция (query, key) видна, если ключ не из будущего относительно запроса.
bool is_visible(int64_t query, int64_t key, int64_t query_offset) {
  return key <= query + query_offset;
}

}  // namespace

Tensor causal_mask(const Tensor& scores, int64_t query_offset) {
  LLM_CHECK_MSG(scores.rank() >= 2, "маске нужны оси запросов и ключей");
  const int64_t keys = scores.dim(-1);
  const int64_t queries = scores.dim(-2);

  Tensor out = Tensor::uninitialized(scores.shape());
  if (scores.numel() == 0) {  // см. masked_softmax
    return out;
  }
  const int64_t matrices = scores.numel() / (queries * keys);

  Tensor holder;
  const float* input = dense_data(scores, &holder);
  float* output = out.data();

  const float blocked = -std::numeric_limits<float>::infinity();
  for_rows(matrices * queries, keys, [&](int64_t row) {
    const int64_t query = row % queries;
    const int64_t base = row * keys;
    for (int64_t key = 0; key < keys; ++key) {
      output[base + key] =
          is_visible(query, key, query_offset) ? input[base + key] : blocked;
    }
  });
  return out;
}

Tensor causal_mask_backward(const Tensor& grad_output, int64_t query_offset) {
  const int64_t keys = grad_output.dim(-1);
  const int64_t queries = grad_output.dim(-2);

  Tensor out = Tensor::uninitialized(grad_output.shape());
  if (grad_output.numel() == 0) {  // см. masked_softmax
    return out;
  }
  const int64_t matrices = grad_output.numel() / (queries * keys);

  Tensor holder;
  const float* input = dense_data(grad_output, &holder);
  float* output = out.data();

  // Закрытые позиции на результат не влияли, значит их градиент — нуль.
  // Формально softmax и так обнулил бы его, но явный нуль надёжнее: он не
  // зависит от того, что стоит следующей операцией.
  for_rows(matrices * queries, keys, [&](int64_t row) {
    const int64_t query = row % queries;
    const int64_t base = row * keys;
    for (int64_t key = 0; key < keys; ++key) {
      output[base + key] =
          is_visible(query, key, query_offset) ? input[base + key] : 0.0f;
    }
  });
  return out;
}

}  // namespace ops
}  // namespace llm
