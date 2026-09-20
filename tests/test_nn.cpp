// Операции модели: слитые ядра и их обратный проход.
//
// Главная проверка здесь — не численная. Слитое ядро обязано совпадать со
// сборкой той же формулы из примитивов, значения которых уже проверены. Это
// сильнее численной разности: сравниваются точные величины, а не приближения.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "autograd/nn.h"
#include "core/thread_pool.h"
#include "autograd/ops.h"
#include "gradcheck.h"
#include "ops/elementwise.h"
#include "ops/embedding.h"
#include "ops/fast_exp.h"
#include "ops/loss.h"
#include "ops/nn.h"
#include "ops/rope.h"
#include "ops/row_reduce.h"
#include "testing.h"

namespace {

using llm::autograd::Var;
using llm::testing::random_tensor;
using llm::testing::weighted_sum;

const float kEps = 1e-5f;

// RMSNorm, собранная из примитивов: x / sqrt(mean(x^2) + eps) * w.
Var rms_norm_composed(const Var& input, const Var& weight, float eps) {
  const Var squares = llm::autograd::mul(input, input);
  const Var mean_square = llm::autograd::mean(squares, {-1}, true);
  const Var scale =
      llm::autograd::sqrt(llm::autograd::add_scalar(mean_square, eps));
  return llm::autograd::mul(llm::autograd::div(input, scale), weight);
}

// Softmax из примитивов. Без вычитания максимума — поэтому применима только к
// умеренным значениям, что для сверки и нужно.
Var softmax_composed(const Var& input) {
  const Var exponentials = llm::autograd::exp(input);
  const Var total = llm::autograd::sum(exponentials, {-1}, true);
  return llm::autograd::div(exponentials, total);
}

void expect_tensors_close(const llm::Tensor& actual,
                          const llm::Tensor& expected, double tolerance) {
  LLM_CHECK_MSG(actual.shape() == expected.shape(),
                "формы " << actual.shape() << " и " << expected.shape());
  const llm::Tensor a = actual.contiguous();
  const llm::Tensor b = expected.contiguous();
  for (int64_t i = 0; i < a.numel(); ++i) {
    LLM_CHECK_MSG(
        std::fabs(a.data()[i] - b.data()[i]) <= tolerance,
        "элемент " << i << ": " << a.data()[i] << " против " << b.data()[i]);
  }
}

}  // namespace

LLM_TEST(Nn, RmsNormMatchesComposition) {
  const llm::Tensor input_data = random_tensor(llm::Shape({3, 8}), 41);
  const llm::Tensor weight_data = random_tensor(llm::Shape({8}), 42);

  Var fused_input = Var::leaf(input_data.clone(), true);
  Var fused_weight = Var::leaf(weight_data.clone(), true);
  Var fused = llm::autograd::rms_norm(fused_input, fused_weight, kEps);

  Var composed_input = Var::leaf(input_data.clone(), true);
  Var composed_weight = Var::leaf(weight_data.clone(), true);
  Var composed = rms_norm_composed(composed_input, composed_weight, kEps);

  expect_tensors_close(fused.value(), composed.value(), 1e-5);

  weighted_sum(fused, 43).backward();
  weighted_sum(composed, 43).backward();
  expect_tensors_close(fused_input.grad(), composed_input.grad(), 1e-4);
  expect_tensors_close(fused_weight.grad(), composed_weight.grad(), 1e-4);
}

LLM_TEST(Nn, SoftmaxMatchesComposition) {
  const llm::Tensor input_data = random_tensor(llm::Shape({4, 6}), 44);

  Var fused_input = Var::leaf(input_data.clone(), true);
  Var fused = llm::autograd::softmax(fused_input);

  Var composed_input = Var::leaf(input_data.clone(), true);
  Var composed = softmax_composed(composed_input);

  expect_tensors_close(fused.value(), composed.value(), 1e-6);

  weighted_sum(fused, 45).backward();
  weighted_sum(composed, 45).backward();
  expect_tensors_close(fused_input.grad(), composed_input.grad(), 1e-5);
}

LLM_TEST(Nn, SoftmaxIsNormalised) {
  const llm::Tensor result =
      llm::ops::softmax(random_tensor(llm::Shape({5, 7}), 46));
  for (int64_t row = 0; row < 5; ++row) {
    double total = 0.0;
    for (int64_t column = 0; column < 7; ++column) {
      LLM_CHECK(result(row, column) > 0.0f);
      total += result(row, column);
    }
    LLM_EXPECT_NEAR(total, 1.0, 1e-6);
  }
}

LLM_TEST(Nn, SoftmaxSurvivesHugeLogits) {
  // Без вычитания максимума exp(1000) даёт бесконечность, а деление
  // бесконечности на бесконечность — NaN.
  const llm::Tensor logits =
      llm::Tensor::from_values(llm::Shape({1, 3}), {1000.0f, 1001.0f, 999.0f});
  const llm::Tensor result = llm::ops::softmax(logits);
  double total = 0.0;
  for (int64_t i = 0; i < 3; ++i) {
    LLM_CHECK(std::isfinite(result(0, i)));
    total += result(0, i);
  }
  LLM_EXPECT_NEAR(total, 1.0, 1e-6);
  // Наибольший логит должен получить наибольшую вероятность.
  LLM_CHECK(result(0, 1) > result(0, 0));
  LLM_CHECK(result(0, 0) > result(0, 2));
}

LLM_TEST(Nn, SiluAndGeluValues) {
  const llm::Tensor input =
      llm::Tensor::from_values(llm::Shape({3}), {-1.0f, 0.0f, 1.0f});
  const llm::Tensor silu_result = llm::ops::silu(input);
  // silu(0) = 0, silu(1) = sigmoid(1) = 0.73106
  LLM_EXPECT_NEAR(silu_result(1), 0.0, 1e-6);
  LLM_EXPECT_NEAR(silu_result(2), 0.7310586, 1e-7);
  // Слева silu уходит в ноль снизу, но не монотонно — у неё есть минимум.
  LLM_CHECK(silu_result(0) < 0.0f);

  const llm::Tensor gelu_result = llm::ops::gelu(input);
  LLM_EXPECT_NEAR(gelu_result(1), 0.0, 1e-6);
  LLM_EXPECT_NEAR(gelu_result(2), 0.8411920, 1e-6);
}

LLM_TEST(Nn, CausalMaskBlocksFuture) {
  const llm::Tensor scores = llm::Tensor::zeros(llm::Shape({3, 3}));
  const llm::Tensor masked = llm::ops::causal_mask(scores, 0);
  const llm::Tensor attention = llm::ops::softmax(masked);

  // Первый запрос видит только первый ключ.
  LLM_EXPECT_NEAR(attention(0, 0), 1.0, 1e-6);
  LLM_EXPECT_NEAR(attention(0, 1), 0.0, 0.0);
  LLM_EXPECT_NEAR(attention(0, 2), 0.0, 0.0);
  // Последний видит все три поровну.
  LLM_EXPECT_NEAR(attention(2, 0), 1.0 / 3.0, 1e-6);
  LLM_EXPECT_NEAR(attention(2, 2), 1.0 / 3.0, 1e-6);
  // Каждая строка остаётся распределением.
  for (int64_t row = 0; row < 3; ++row) {
    double total = 0.0;
    for (int64_t column = 0; column < 3; ++column) {
      total += attention(row, column);
    }
    LLM_EXPECT_NEAR(total, 1.0, 1e-6);
  }
}

LLM_TEST(Nn, CausalMaskWithOffset) {
  // Генерация с KV-кэшем: запрос один, а ключей уже четыре, и запрос стоит на
  // позиции 3 — значит видит все четыре.
  const llm::Tensor scores = llm::Tensor::zeros(llm::Shape({1, 4}));
  const llm::Tensor attention =
      llm::ops::softmax(llm::ops::causal_mask(scores, 3));
  for (int64_t key = 0; key < 4; ++key) {
    LLM_EXPECT_NEAR(attention(0, key), 0.25, 1e-6);
  }
  // А со смещением 1 — только два первых ключа.
  const llm::Tensor narrow =
      llm::ops::softmax(llm::ops::causal_mask(scores, 1));
  LLM_EXPECT_NEAR(narrow(0, 0), 0.5, 1e-6);
  LLM_EXPECT_NEAR(narrow(0, 2), 0.0, 0.0);
}

LLM_TEST(Nn, RopeEncodesRelativePosition) {
  // Определяющее свойство RoPE: после поворота скалярное произведение запроса
  // и ключа зависит только от РАЗНОСТИ их позиций. Именно из-за этого
  // относительное кодирование возникает из геометрии, без отдельных
  // параметров.
  const llm::Tensor query = random_tensor(llm::Shape({1, 8}), 47);
  const llm::Tensor key = random_tensor(llm::Shape({1, 8}), 48);
  const float theta = 10000.0f;

  const auto dot_at = [&](int64_t query_position, int64_t key_position) {
    const llm::Tensor rotated_query =
        llm::ops::rope(query, query_position, theta);
    const llm::Tensor rotated_key = llm::ops::rope(key, key_position, theta);
    double total = 0.0;
    for (int64_t i = 0; i < 8; ++i) {
      total += static_cast<double>(rotated_query(0, i)) * rotated_key(0, i);
    }
    return total;
  };

  // Разность позиций 2 — произведение обязано быть одним и тем же.
  const double reference = dot_at(2, 0);
  LLM_EXPECT_NEAR(dot_at(5, 3), reference, 1e-5);
  LLM_EXPECT_NEAR(dot_at(12, 10), reference, 1e-5);
  // А при другой разности — уже другим.
  LLM_CHECK(std::fabs(dot_at(3, 0) - reference) > 1e-3);
}

LLM_TEST(Nn, RopeOnEmptyTensorDoesNotDivideByZero) {
  // Форма (3, 0): размерность головы нулевая, а чётность при этом соблюдена,
  // так что проверка на чётность такой тензор пропускала. Дальше число
  // блоков считалось делением на sequence * head_dim, и получался не отказ с
  // сообщением, а SIGFPE.
  const llm::Tensor empty = llm::Tensor::zeros(llm::Shape({3, 0}));
  const llm::Tensor rotated = llm::ops::rope(empty, 0, 10000.0f);
  LLM_CHECK_EQ(rotated.numel(), static_cast<int64_t>(0));

  // И с другой стороны: нулевая длина последовательности.
  const llm::Tensor no_steps = llm::Tensor::zeros(llm::Shape({0, 4}));
  LLM_CHECK_EQ(llm::ops::rope(no_steps, 0, 10000.0f).numel(),
               static_cast<int64_t>(0));
}

LLM_TEST(Nn, RopePreservesLength) {
  // Поворот — ортогональное преобразование, длина вектора не меняется.
  const llm::Tensor input = random_tensor(llm::Shape({3, 8}), 49);
  const llm::Tensor rotated = llm::ops::rope(input, 7, 10000.0f);
  for (int64_t row = 0; row < 3; ++row) {
    double before = 0.0;
    double after = 0.0;
    for (int64_t i = 0; i < 8; ++i) {
      before += static_cast<double>(input(row, i)) * input(row, i);
      after += static_cast<double>(rotated(row, i)) * rotated(row, i);
    }
    LLM_EXPECT_NEAR(after, before, 1e-5);
  }
}

LLM_TEST(Nn, RopeAtPositionZeroIsIdentity) {
  // Поворот на нулевой угол ничего не меняет, поэтому первая позиция
  // последовательности всегда остаётся собой. Остальные — нет: предпоследняя
  // ось тензора и есть позиция.
  const llm::Tensor input = random_tensor(llm::Shape({3, 4}), 50);
  const llm::Tensor rotated = llm::ops::rope(input, 0, 10000.0f);

  for (int64_t i = 0; i < 4; ++i) {
    LLM_EXPECT_NEAR(rotated(0, i), input(0, i), 1e-6);
  }
  bool later_positions_changed = false;
  for (int64_t step = 1; step < 3; ++step) {
    for (int64_t i = 0; i < 4; ++i) {
      if (std::fabs(rotated(step, i) - input(step, i)) > 1e-4f) {
        later_positions_changed = true;
      }
    }
  }
  LLM_CHECK(later_positions_changed);
}

LLM_TEST(Nn, RopeBackwardUndoesRotation) {
  const llm::Tensor input = random_tensor(llm::Shape({2, 6}), 51);
  const llm::Tensor rotated = llm::ops::rope(input, 4, 10000.0f);
  expect_tensors_close(llm::ops::rope_backward(rotated, 4, 10000.0f), input,
                       1e-5);
}

LLM_TEST(Nn, EmbeddingSelectsRows) {
  const llm::Tensor table =
      llm::Tensor::from_values(llm::Shape({3, 2}), {1, 2, 3, 4, 5, 6});
  const llm::Tensor result = llm::ops::embedding(table, {2, 0, 2});
  LLM_CHECK(result.shape() == llm::Shape({3, 2}));
  LLM_EXPECT_NEAR(result(0, 0), 5.0, 1e-6);
  LLM_EXPECT_NEAR(result(1, 1), 2.0, 1e-6);
  LLM_EXPECT_NEAR(result(2, 0), 5.0, 1e-6);
}

LLM_TEST(Nn, EmbeddingRejectsOutOfRangeToken) {
  const llm::Tensor table = llm::Tensor::zeros(llm::Shape({3, 2}));
  LLM_EXPECT_THROWS(llm::ops::embedding(table, {3}));
  LLM_EXPECT_THROWS(llm::ops::embedding(table, {-1}));
}

LLM_TEST(Nn, EmbeddingGradientAccumulatesPerToken) {
  // Токен 1 встречается дважды, значит его строка должна собрать сумму двух
  // градиентов, а неиспользованный токен 0 — остаться нулевым.
  Var table = Var::leaf(llm::Tensor::zeros(llm::Shape({3, 2})), true);
  const std::vector<std::int32_t> ids = {1, 2, 1};
  llm::autograd::sum_all(llm::autograd::embedding(table, ids)).backward();

  LLM_EXPECT_NEAR(table.grad()(0, 0), 0.0, 1e-6);
  LLM_EXPECT_NEAR(table.grad()(1, 0), 2.0, 1e-6);
  LLM_EXPECT_NEAR(table.grad()(2, 0), 1.0, 1e-6);
}

LLM_TEST(Nn, CrossEntropyKnownValues) {
  // Одинаковые логиты — равномерное распределение, потери log(vocab).
  const llm::Tensor uniform = llm::Tensor::zeros(llm::Shape({1, 4}));
  LLM_EXPECT_NEAR(*llm::ops::cross_entropy(uniform, {0}).data(), std::log(4.0),
                  1e-6);

  // Уверенное правильное предсказание — потери около нуля.
  const llm::Tensor confident =
      llm::Tensor::from_values(llm::Shape({1, 3}), {20.0f, 0.0f, 0.0f});
  LLM_CHECK(*llm::ops::cross_entropy(confident, {0}).data() < 1e-6f);

  // Уверенное неправильное — потери велики.
  LLM_CHECK(*llm::ops::cross_entropy(confident, {1}).data() > 19.0f);
}

LLM_TEST(Nn, CrossEntropyGradientSumsToZero) {
  // Градиент по логитам — это (предсказание минус цель), а обе величины
  // распределения, поэтому сумма по словарю обязана быть нулём.
  const llm::Tensor logits = random_tensor(llm::Shape({4, 5}), 52);
  const std::vector<std::int32_t> targets = {0, 3, 2, 4};
  const llm::Tensor grad = llm::ops::cross_entropy_backward(logits, targets);
  for (int64_t row = 0; row < 4; ++row) {
    double total = 0.0;
    for (int64_t column = 0; column < 5; ++column) {
      total += grad(row, column);
    }
    LLM_EXPECT_NEAR(total, 0.0, 1e-6);
  }
}

LLM_TEST(Nn, LogZTracksShiftOfAllLogits) {
  // log Z — величина, слепая к предсказаниям и чувствительная к общему сдвигу.
  // Обе половины этого утверждения проверяются здесь, потому что телеметрия
  // нужна ровно за этим: увидеть дрейф, которого не видно по потерям.
  const llm::Tensor uniform = llm::Tensor::zeros(llm::Shape({1, 4}));
  LLM_EXPECT_NEAR(llm::ops::prediction_stats(uniform, {0}).log_z, std::log(4.0),
                  1e-6);

  // Сдвиг всех логитов строки на c двигает log Z ровно на c.
  const llm::Tensor shifted =
      llm::Tensor::from_values(llm::Shape({1, 4}), {7.0f, 7.0f, 7.0f, 7.0f});
  LLM_EXPECT_NEAR(llm::ops::prediction_stats(shifted, {0}).log_z,
                  std::log(4.0) + 7.0, 1e-6);

  // При этом ни потери, ни энтропия от сдвига не меняются: softmax его не
  // видит. Без телеметрии дрейф был бы невидим целиком.
  LLM_EXPECT_NEAR(*llm::ops::cross_entropy(shifted, {0}).data(),
                  *llm::ops::cross_entropy(uniform, {0}).data(), 1e-5);
  LLM_EXPECT_NEAR(llm::ops::prediction_stats(shifted, {0}).entropy,
                  llm::ops::prediction_stats(uniform, {0}).entropy, 1e-5);

  // Связь с самим штрафом: z_loss — это среднее по строкам от квадрата log Z.
  const llm::Tensor rows =
      llm::Tensor::from_values(llm::Shape({2, 2}), {5.0f, 5.0f, -3.0f, -3.0f});
  const double high = std::log(2.0) + 5.0;
  const double low = std::log(2.0) - 3.0;
  LLM_EXPECT_NEAR(*llm::ops::z_loss(rows).data(),
                  (high * high + low * low) / 2.0, 1e-4);
}

LLM_TEST(Nn, CrossEntropyRejectsMismatchedTargets) {
  const llm::Tensor logits = llm::Tensor::zeros(llm::Shape({2, 3}));
  LLM_EXPECT_THROWS(llm::ops::cross_entropy(logits, {0}));
  LLM_EXPECT_THROWS(llm::ops::cross_entropy(logits, {0, 3}));
}

// --- Численная проверка градиентов ---

LLM_TEST(Nn, GradRmsNorm) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::rms_norm(v[0], v[1], kEps), 60);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 6}), 61),
                                    random_tensor(llm::Shape({6}), 62)}));
}

LLM_TEST(Nn, GradSoftmax) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::softmax(v[0]), 63);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 5}), 64)}));
}

LLM_TEST(Nn, GradSilu) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::silu(v[0]), 65);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 4}), 66)}));
}

LLM_TEST(Nn, GradGelu) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::gelu(v[0]), 67);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 4}), 68)}));
}

LLM_TEST(Nn, GradRope) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::rope(v[0], 3, 10000.0f), 69);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 4, 6}), 70)}));
}

LLM_TEST(Nn, GradMaskedSoftmax) {
  // Маску проверяем вместе с softmax: сама по себе она выдаёт минус
  // бесконечность, и численная разность на ней не определена.
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(
        llm::autograd::softmax(llm::autograd::causal_mask(v[0], 0)), 71);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({4, 4}), 72)}));
}

LLM_TEST(Nn, GradEmbedding) {
  const std::vector<std::int32_t> ids = {2, 0, 2, 1};
  const auto fn = [&ids](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::embedding(v[0], ids), 73);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 4}), 74)}));
}

LLM_TEST(Nn, GradCrossEntropy) {
  const std::vector<std::int32_t> targets = {1, 3, 0};
  const auto fn = [&targets](const std::vector<Var>& v) {
    return llm::autograd::cross_entropy(v[0], targets);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 5}), 75)}));
}

LLM_TEST(Nn, GradThroughAttentionShapedChain) {
  // Цепочка той же формы, что и в настоящем внимании: оценки, маска, softmax,
  // взвешивание значений. Ошибка в стыке операций проявится здесь, даже если
  // каждая по отдельности верна.
  const auto fn = [](const std::vector<Var>& v) {
    const Var queries = llm::autograd::rope(v[0], 0, 10000.0f);
    const Var keys = llm::autograd::rope(v[1], 0, 10000.0f);
    const Var scores =
        llm::autograd::matmul(queries, llm::autograd::transpose(keys, -2, -1));
    const Var scaled = llm::autograd::mul_scalar(scores, 0.5f);
    const Var attention =
        llm::autograd::softmax(llm::autograd::causal_mask(scaled, 0));
    return weighted_sum(llm::autograd::matmul(attention, v[2]), 76);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({4, 4}), 77),
                                    random_tensor(llm::Shape({4, 4}), 78),
                                    random_tensor(llm::Shape({4, 3}), 79)}));
}

// Редукции по строке считаются с фиксированным числом накопителей именно
// затем, чтобы порядок сложения задавался исходником, а не шириной вектора и
// не числом потоков (см. ops/row_reduce.h). Свойство проверяется двумя
// способами, и первый важнее.
namespace {

struct WidthGuard {
  explicit WidthGuard(int width) { llm::set_parallel_width(width); }
  ~WidthGuard() { llm::set_parallel_width(0); }
};

void expect_bitwise_equal(const char* what, const llm::Tensor& lhs,
                         const llm::Tensor& rhs) {
  LLM_CHECK_MSG(lhs.numel() == rhs.numel(),
                what << ": размеры разошлись, " << lhs.numel() << " и "
                     << rhs.numel());
  for (int64_t i = 0; i < lhs.numel(); ++i) {
    LLM_CHECK_MSG(lhs.data()[i] == rhs.data()[i],
                  what << ": элемент " << i << " разошёлся, "
                       << lhs.data()[i] << " и " << rhs.data()[i]);
  }
}

}  // namespace

LLM_TEST(Nn, RowSumFollowsTheDeclaredOrder) {
  // Данные подобраны так, чтобы перегруппировка была видна. Это не
  // придирка, а условие осмысленности проверки: первая версия этого теста
  // брала обычную строку softmax и проходила даже когда число накопителей
  // меняли с восьми на четыре — сумма двухсот положительных чисел в двойной
  // точности к перегруппировке нечувствительна, и тест ничего не проверял.
  //
  // Здесь единица и дальше числа величиной 2^-53: каждое по отдельности при
  // прибавлении к единице пропадает без следа, а сложенные между собой —
  // нет. Поэтому последовательная сумма и сумма по восьми накопителям
  // расходятся, и это расхождение здесь же и проверяется.
  const int64_t width = 173;
  std::vector<float> row(static_cast<std::size_t>(width));
  row[0] = 1.0f;
  const float tiny = std::ldexp(1.0f, -53);
  for (int64_t i = 1; i < width; ++i) {
    row[static_cast<std::size_t>(i)] = tiny;
  }

  double serial = 0.0;
  for (int64_t i = 0; i < width; ++i) {
    serial += static_cast<double>(row[static_cast<std::size_t>(i)]);
  }

  double part[8] = {};
  int64_t i = 0;
  for (; i + 8 <= width; i += 8) {
    for (int j = 0; j < 8; ++j) {
      part[j] += static_cast<double>(row[static_cast<std::size_t>(i + j)]);
    }
  }
  for (int j = 0; i < width; ++i, ++j) {
    part[j] += static_cast<double>(row[static_cast<std::size_t>(i)]);
  }
  double declared = 0.0;
  for (int j = 0; j < 8; ++j) {
    declared += part[j];
  }

  LLM_CHECK_MSG(declared != serial,
                "данные не различают порядок суммирования, проверка пуста");
  LLM_CHECK_MSG(llm::ops::row_sum(row.data(), width) == declared,
                "row_sum разошёлся с заявленным порядком");
}

LLM_TEST(Nn, RowReductionsDoNotDependOnThreadCount) {
  // Размер взят выше порога деления работы (иначе многопоточная ветка просто
  // не включится и проверка окажется пустой).
  const llm::Tensor x = random_tensor(llm::Shape({256, 173}), 901);
  const llm::Tensor w = random_tensor(llm::Shape({173}), 902);
  const llm::Tensor g = random_tensor(llm::Shape({256, 173}), 903);

  llm::Tensor softmax_one;
  llm::Tensor norm_one;
  llm::Tensor softmax_back_one;
  {
    const WidthGuard guard(1);
    softmax_one = llm::ops::softmax(x);
    norm_one = llm::ops::rms_norm(x, w, kEps);
    softmax_back_one = llm::ops::softmax_backward(g, softmax_one);
  }
  {
    const WidthGuard guard(4);
    expect_bitwise_equal("softmax", softmax_one, llm::ops::softmax(x));
    expect_bitwise_equal("rms_norm", norm_one, llm::ops::rms_norm(x, w, kEps));
    expect_bitwise_equal("softmax_backward", softmax_back_one,
                         llm::ops::softmax_backward(g, softmax_one));
  }
}

LLM_TEST(Nn, MaskedSoftmaxMatchesTheSeparateChainBitwise) {
  // Слитая цепочка обязана совпадать с раздельной побитово, а не приблизительно.
  // Это и есть условие, при котором её можно ставить в модель, не пересчитывая
  // заново все числа обучения в README.
  //
  // Совпадение не случайно: закрытые позиции дают после экспоненты ровно нуль,
  // нуль не меняет ни максимум, ни сумму, а номер накопителя у элемента зависит
  // только от его индекса — значит укорочение строки не переставляет слагаемые.
  const float scale = 0.3125f;  // степень двойки не берём: округление важно
  for (int64_t offset = 0; offset <= 3; ++offset) {
    const llm::Tensor scores = random_tensor(llm::Shape({2, 13, 13}), 905);

    const llm::Tensor separate = llm::ops::softmax(
        llm::ops::causal_mask(llm::ops::mul_scalar(scores, scale), offset));
    const llm::Tensor fused = llm::ops::masked_softmax(scores, scale, offset);
    expect_bitwise_equal("masked_softmax", separate, fused);

    const llm::Tensor grad = random_tensor(llm::Shape({2, 13, 13}), 906);
    const llm::Tensor separate_back = llm::ops::mul_scalar(
        llm::ops::causal_mask_backward(
            llm::ops::softmax_backward(grad, separate), offset),
        scale);
    const llm::Tensor fused_back =
        llm::ops::masked_softmax_backward(grad, fused, scale, offset);
    expect_bitwise_equal("masked_softmax_backward", separate_back, fused_back);
  }
}

LLM_TEST(Nn, AttentionEntropyIsNormalisedToTheAvailableKeys) {
  // Энтропия внимания — диагностика, которая печатается при обучении, и
  // читать её можно только зная нормировку. Она делится на логарифм числа
  // ДОСТУПНЫХ запросу ключей, поэтому равномерное внимание даёт ровно
  // единицу на любой длине, а сосредоточенное — ноль. Основание логарифма при
  // такой нормировке сокращается, и это тоже стоит закрепить: иначе кто-нибудь
  // однажды «исправит» натуральный логарифм на двоичный в одном месте из двух.
  //
  // Прямой проверки у этой величины не было вовсе — только косвенная, через
  // сравнение двух прогонов в тестах вариантов.

  // Равномерное внимание по каузальной маске: запрос t видит t + 1 ключей.
  const int64_t length = 4;
  llm::Tensor uniform = llm::Tensor::zeros(llm::Shape({length, length}));
  for (int64_t query = 0; query < length; ++query) {
    const int64_t available = query + 1;
    for (int64_t key = 0; key < available; ++key) {
      uniform(query, key) = 1.0f / static_cast<float>(available);
    }
  }
  // Первая строка видит один ключ и в среднее не входит: энтропия там ноль по
  // построению, и делить было бы не на что.
  LLM_EXPECT_NEAR(llm::ops::attention_entropy(uniform, 0), 1.0, 1e-6);

  // Сосредоточенное внимание: вся масса на одном ключе.
  llm::Tensor sharp = llm::Tensor::zeros(llm::Shape({length, length}));
  for (int64_t query = 0; query < length; ++query) {
    sharp(query, 0) = 1.0f;
  }
  LLM_EXPECT_NEAR(llm::ops::attention_entropy(sharp, 0), 0.0, 1e-6);

  // Смещение позиций меняет число доступных ключей, а значит и нормировку.
  // Строка из двух равных половин при смещении 0 доступна только второму
  // запросу и даёт единицу; при смещении 2 доступны оба ключа обоим запросам.
  llm::Tensor pair = llm::Tensor::zeros(llm::Shape({2, 2}));
  pair(0, 0) = 0.5f;
  pair(0, 1) = 0.5f;
  pair(1, 0) = 0.5f;
  pair(1, 1) = 0.5f;
  LLM_EXPECT_NEAR(llm::ops::attention_entropy(pair, 2), 1.0, 1e-6);
}

LLM_TEST(Nn, EmptyTensorsDoNotDivideByZero) {
  // Число матриц считается как numel / (queries * keys), а пустым тензор
  // бывает ровно тогда, когда один из множителей нулевой. Получался не отказ с
  // сообщением, а SIGFPE. Соседняя attention_entropy эту проверку имела —
  // значит случай был продуман, но записан лишь в одном месте из пяти.
  //
  // Проверяются обе пустоты: ноль запросов и ноль ключей. Они разные: при нуле
  // запросов пуста ось, по которой идёт внешний цикл, при нуле ключей — та, по
  // которой считается маска.
  const llm::Shape shapes[2] = {llm::Shape({2, 0, 4}), llm::Shape({2, 3, 0})};
  for (int i = 0; i < 2; ++i) {
    const llm::Tensor scores = llm::Tensor::zeros(shapes[i]);
    LLM_CHECK_EQ(scores.numel(), static_cast<std::int64_t>(0));

    const llm::Tensor weights = llm::ops::masked_softmax(scores, 0.5f, 0);
    LLM_CHECK(weights.shape() == shapes[i]);

    const llm::Tensor back =
        llm::ops::masked_softmax_backward(scores, weights, 0.5f, 0);
    LLM_CHECK(back.shape() == shapes[i]);

    const llm::Tensor masked = llm::ops::causal_mask(scores, 0);
    LLM_CHECK(masked.shape() == shapes[i]);

    const llm::Tensor masked_back = llm::ops::causal_mask_backward(scores, 0);
    LLM_CHECK(masked_back.shape() == shapes[i]);

    // Здесь проверка была и раньше — пусть остаётся закреплённой.
    LLM_EXPECT_NEAR(llm::ops::attention_entropy(weights, 0), 0.0, 0.0);

    // И операции по последней оси на пустом входе.
    LLM_CHECK(llm::ops::softmax(scores).shape() == shapes[i]);
    LLM_CHECK(llm::ops::silu(scores).shape() == shapes[i]);
    LLM_CHECK(llm::ops::gelu(scores).shape() == shapes[i]);
    LLM_CHECK(llm::ops::rope(scores, 0, 10000.0f).shape() == shapes[i]);
  }
}

LLM_TEST(Nn, ElementwiseKernelsDoNotDependOnThreadCount) {
  // Прежняя проверка на независимость от числа потоков брала softmax,
  // rms_norm и softmax_backward — то есть операции, которые делятся ПО
  // СТРОКАМ. У строки границы заданы формой, и деление их не трогает.
  //
  // А silu и gelu делятся по ЭЛЕМЕНТАМ: parallel_range режет numel на куски,
  // и границы кусков зависят от числа потоков напрямую. Ядро сигмоиды обязано
  // считать хвост куска тем же кодом, что и середину, иначе результат поедет
  // от числа ядер. В комментарии к silu это написано, а проверено не было.
  //
  // Заодно закрыты остальные операции, до которых прежняя проверка не дошла:
  // обе нормировки в обратную сторону, маскированный softmax и функции потерь,
  // где сумма собирается по восьми частям.
  const llm::Tensor x = random_tensor(llm::Shape({256, 173}), 911);
  const llm::Tensor g = random_tensor(llm::Shape({256, 173}), 912);
  const llm::Tensor w = random_tensor(llm::Shape({173}), 913);
  const llm::Tensor b = random_tensor(llm::Shape({173}), 914);
  const llm::Tensor scores = random_tensor(llm::Shape({8, 64, 64}), 915);
  const llm::Tensor logits = random_tensor(llm::Shape({256, 173}), 916);
  std::vector<std::int32_t> targets(256);
  for (std::size_t i = 0; i < targets.size(); ++i) {
    targets[i] = static_cast<std::int32_t>((i * 37) % 173);
  }
  LLM_CHECK_GT(x.numel(), static_cast<std::int64_t>(1 << 14));

  llm::Tensor silu_one;
  llm::Tensor silu_back_one;
  llm::Tensor gelu_one;
  llm::Tensor gelu_back_one;
  llm::Tensor layer_one;
  llm::Tensor layer_dx_one;
  llm::Tensor layer_dw_one;
  llm::Tensor layer_db_one;
  llm::Tensor rms_dx_one;
  llm::Tensor rms_dw_one;
  llm::Tensor masked_one;
  llm::Tensor masked_back_one;
  llm::Tensor loss_one;
  llm::Tensor zloss_one;
  {
    const WidthGuard guard(1);
    silu_one = llm::ops::silu(x);
    silu_back_one = llm::ops::silu_backward(g, x);
    gelu_one = llm::ops::gelu(x);
    gelu_back_one = llm::ops::gelu_backward(g, x);
    layer_one = llm::ops::layer_norm(x, w, b, kEps);
    llm::ops::layer_norm_backward(g, x, w, kEps, &layer_dx_one, &layer_dw_one,
                                  &layer_db_one);
    llm::ops::rms_norm_backward(g, x, w, kEps, &rms_dx_one, &rms_dw_one);
    masked_one = llm::ops::masked_softmax(scores, 0.3125f, 0);
    masked_back_one =
        llm::ops::masked_softmax_backward(scores, masked_one, 0.3125f, 0);
    loss_one = llm::ops::cross_entropy(logits, targets);
    zloss_one = llm::ops::z_loss(logits);
  }
  {
    const WidthGuard guard(4);
    expect_bitwise_equal("silu", silu_one, llm::ops::silu(x));
    expect_bitwise_equal("silu_backward", silu_back_one,
                         llm::ops::silu_backward(g, x));
    expect_bitwise_equal("gelu", gelu_one, llm::ops::gelu(x));
    expect_bitwise_equal("gelu_backward", gelu_back_one,
                         llm::ops::gelu_backward(g, x));
    expect_bitwise_equal("layer_norm", layer_one,
                         llm::ops::layer_norm(x, w, b, kEps));

    llm::Tensor dx;
    llm::Tensor dw;
    llm::Tensor db;
    llm::ops::layer_norm_backward(g, x, w, kEps, &dx, &dw, &db);
    expect_bitwise_equal("layer_norm_backward по входу", layer_dx_one, dx);
    expect_bitwise_equal("layer_norm_backward по весу", layer_dw_one, dw);
    expect_bitwise_equal("layer_norm_backward по сдвигу", layer_db_one, db);

    llm::Tensor rms_dx;
    llm::Tensor rms_dw;
    llm::ops::rms_norm_backward(g, x, w, kEps, &rms_dx, &rms_dw);
    expect_bitwise_equal("rms_norm_backward по входу", rms_dx_one, rms_dx);
    expect_bitwise_equal("rms_norm_backward по весу", rms_dw_one, rms_dw);

    expect_bitwise_equal("masked_softmax", masked_one,
                         llm::ops::masked_softmax(scores, 0.3125f, 0));
    expect_bitwise_equal(
        "masked_softmax_backward", masked_back_one,
        llm::ops::masked_softmax_backward(scores, masked_one, 0.3125f, 0));
    expect_bitwise_equal("cross_entropy", loss_one,
                         llm::ops::cross_entropy(logits, targets));
    expect_bitwise_equal("z_loss", zloss_one, llm::ops::z_loss(logits));
  }
}

LLM_TEST(Nn, GradLayerNormOnARowLongerThanTheAccumulators) {
  // Градиент LayerNorm проверяется численно в Variants.GradLayerNorm, но там
  // строка шириной 6 — короче восьми накопителей. Значит основной цикл
  // накопления в layer_norm_backward не выполняется НИ РАЗУ: условие
  // i + 8 <= width ложно сразу, и вся работа достаётся хвостовому циклу.
  //
  // Здесь ширина 19: основной цикл проходит дважды, хвост добирает три
  // элемента. Ошибка в раскладке по накопителям видна только так.
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::layer_norm(v[0], v[1], v[2], kEps), 84);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 19}), 85),
                                    random_tensor(llm::Shape({19}), 86),
                                    random_tensor(llm::Shape({19}), 87)}));
}

LLM_TEST(Nn, GradRmsNormOnARowLongerThanTheAccumulators) {
  // То же у RMSNorm: Nn.GradRmsNorm берёт ширину 6, и row_dot3 с
  // row_sum_squares там тоже идут одним хвостом. Ширина 19 включает основной
  // цикл.
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::rms_norm(v[0], v[1], kEps), 88);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 19}), 89),
                                    random_tensor(llm::Shape({19}), 90)}));
}

LLM_TEST(Nn, GradMaskedSoftmaxFusedPath) {
  // Слитый masked_softmax — то, что стоит в модели (см. nn/model.cpp), — не
  // проверялся численно ни разу. Проверялась раздельная цепочка causal_mask +
  // softmax и то, что слитая совпадает с ней побитово на уровне ops. Этого
  // почти достаточно, но обёртка автограда остаётся непроверенной: она
  // протаскивает масштаб и смещение запроса, и перепутанный аргумент здесь не
  // поймало бы ничто.
  //
  // Ширина 9 — больше восьми накопителей, чтобы основной цикл row_dot в
  // обратном проходе выполнялся, а не только хвост.
  const float scale = 0.3125f;
  const auto fn = [scale](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::masked_softmax(v[0], scale, 0), 91);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 9, 9}), 92)}));
}

LLM_TEST(Nn, GradSoftmaxAndCrossEntropyOnLongerRows) {
  // У обоих обратный проход опирается на row_dot и row_sum, а прежние
  // проверки брали ширину 5 — меньше восьми накопителей, то есть основной
  // цикл там не выполнялся вовсе. Разница видна: та же поломка в основном
  // цикле LayerNorm прошла мимо проверки шириной 6 и была поймана только
  // шириной 19.
  {
    const auto fn = [](const std::vector<Var>& v) {
      return weighted_sum(llm::autograd::softmax(v[0]), 93);
    };
    LLM_EXPECT_GRADCHECK(
        fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 19}), 94)}));
  }
  {
    const std::vector<std::int32_t> targets = {11, 3, 17, 0};
    const auto fn = [&targets](const std::vector<Var>& v) {
      return llm::autograd::cross_entropy(v[0], targets);
    };
    LLM_EXPECT_GRADCHECK(
        fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({4, 19}), 95)}));
  }
}
