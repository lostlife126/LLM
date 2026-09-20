#include <cstdint>
#include <utility>
#include <vector>

#include "autograd/node.h"
#include "core/fp16.h"
#include "autograd/nn.h"
#include "autograd/ops.h"
#include "testing.h"

namespace {

using llm::autograd::Var;

Var leaf(float value) {
  return Var::leaf(llm::Tensor::full(llm::Shape(), value), true);
}

float scalar_of(const llm::Tensor& tensor) { return *tensor.data(); }

}  // namespace

LLM_TEST(Autograd, ChainRule) {
  // y = (a + b) * a, при a = 3, b = 4.
  // dy/da = (a + b) + a = 10, dy/db = a = 3.
  Var a = leaf(3.0f);
  Var b = leaf(4.0f);
  Var y = llm::autograd::mul(llm::autograd::add(a, b), a);

  LLM_EXPECT_NEAR(scalar_of(y.value()), 21.0, 1e-5);
  y.backward();
  LLM_EXPECT_NEAR(scalar_of(a.grad()), 10.0, 1e-5);
  LLM_EXPECT_NEAR(scalar_of(b.grad()), 3.0, 1e-5);
}

LLM_TEST(Autograd, AccumulatesOnReuse) {
  // Переменная, использованная в двух ветвях, должна получить сумму вкладов.
  // y = a * a * a, dy/da = 3a^2 = 12 при a = 2.
  Var a = leaf(2.0f);
  Var y = llm::autograd::mul(llm::autograd::mul(a, a), a);
  y.backward();
  LLM_EXPECT_NEAR(scalar_of(a.grad()), 12.0, 1e-5);
}

LLM_TEST(Autograd, DiamondGraph) {
  // Ромб: обе ветви сходятся обратно. Если топологический порядок неверен,
  // узел обработается раньше, чем в него внесут вклад обе ветви, и часть
  // градиента потеряется.
  //   b = a * 2;  c = a * 3;  y = b + c
  //   dy/da = 2 + 3 = 5
  Var a = leaf(1.0f);
  Var b = llm::autograd::mul_scalar(a, 2.0f);
  Var c = llm::autograd::mul_scalar(a, 3.0f);
  Var y = llm::autograd::add(b, c);
  y.backward();
  LLM_EXPECT_NEAR(scalar_of(a.grad()), 5.0, 1e-5);
}

LLM_TEST(Autograd, LongSharedChain) {
  // Длинная цепочка, где каждый узел используется дважды: проверяет, что
  // обход в глубину не обрабатывает узел повторно и не переполняет стек.
  Var a = leaf(1.0f);
  Var current = a;
  for (int i = 0; i < 500; ++i) {
    current = llm::autograd::add(current, current);
  }
  // После n удвоений производная равна 2^n; берём меньше шагов, чтобы не
  // выйти за пределы float, но цепочку строим длинной.
  Var b = leaf(1.0f);
  Var chain = b;
  for (int i = 0; i < 20; ++i) {
    chain = llm::autograd::add(chain, chain);
  }
  chain.backward();
  LLM_EXPECT_NEAR(scalar_of(b.grad()), 1048576.0, 1.0);
}

LLM_TEST(Autograd, ConstantsDoNotTrack) {
  Var a = Var::constant(llm::Tensor::full(llm::Shape(), 2.0f));
  Var b = Var::constant(llm::Tensor::full(llm::Shape(), 3.0f));
  Var y = llm::autograd::mul(a, b);

  LLM_CHECK(!y.requires_grad());
  LLM_EXPECT_NEAR(scalar_of(y.value()), 6.0, 1e-5);
  LLM_EXPECT_THROWS(y.backward());
}

LLM_TEST(Autograd, MixedConstantAndLeaf) {
  // Константа рядом с параметром не мешает: узел заводится, а вклад константе
  // не начисляется.
  Var a = leaf(3.0f);
  Var k = Var::constant(llm::Tensor::full(llm::Shape(), 5.0f));
  Var y = llm::autograd::mul(a, k);

  LLM_CHECK(y.requires_grad());
  y.backward();
  LLM_EXPECT_NEAR(scalar_of(a.grad()), 5.0, 1e-5);
}

LLM_TEST(Autograd, NoGradGuardStopsTape) {
  Var a = leaf(3.0f);
  Var y;
  {
    llm::autograd::NoGradGuard no_grad;
    LLM_CHECK(!llm::autograd::grad_enabled());
    y = llm::autograd::mul(a, a);
  }
  LLM_CHECK(llm::autograd::grad_enabled());
  LLM_CHECK(!y.requires_grad());
  LLM_EXPECT_NEAR(scalar_of(y.value()), 9.0, 1e-5);
}

LLM_TEST(Autograd, NoGradGuardRestoresNestedState) {
  llm::autograd::NoGradGuard outer;
  {
    llm::autograd::NoGradGuard inner;
    LLM_CHECK(!llm::autograd::grad_enabled());
  }
  // Вложенный guard не должен «включать» режим обратно.
  LLM_CHECK(!llm::autograd::grad_enabled());
}

LLM_TEST(Autograd, BackwardRequiresScalar) {
  Var a = Var::leaf(llm::Tensor::zeros(llm::Shape({2, 2})), true);
  Var y = llm::autograd::mul_scalar(a, 2.0f);
  LLM_EXPECT_THROWS(y.backward());
}

LLM_TEST(Autograd, UnusedLeafHasNoGrad) {
  Var a = leaf(1.0f);
  Var unused = leaf(7.0f);
  Var y = llm::autograd::mul_scalar(a, 3.0f);
  y.backward();
  LLM_CHECK(a.grad().defined());
  // Обратный проход до неиспользованной переменной не дошёл — это не ошибка,
  // но и градиента у неё нет.
  LLM_CHECK(!unused.grad().defined());
}

LLM_TEST(Autograd, IntermediateGradsAreFreed) {
  // Градиенты промежуточных узлов после обратного прохода не нужны, и лента
  // их освобождает: на длинной цепочке это основная экономия памяти.
  Var a = leaf(2.0f);
  Var mid = llm::autograd::mul_scalar(a, 3.0f);
  Var y = llm::autograd::mul_scalar(mid, 4.0f);
  y.backward();

  LLM_CHECK(a.grad().defined());
  LLM_CHECK(!mid.node()->grad().defined());
  LLM_EXPECT_NEAR(scalar_of(a.grad()), 12.0, 1e-5);
}

LLM_TEST(Autograd, ZeroGrad) {
  Var a = leaf(2.0f);
  llm::autograd::mul_scalar(a, 3.0f).backward();
  LLM_EXPECT_NEAR(scalar_of(a.grad()), 3.0, 1e-5);

  // Без обнуления градиенты двух шагов сложились бы.
  llm::autograd::mul_scalar(a, 3.0f).backward();
  LLM_EXPECT_NEAR(scalar_of(a.grad()), 6.0, 1e-5);

  a.zero_grad();
  LLM_CHECK(!a.grad().defined());
  llm::autograd::mul_scalar(a, 3.0f).backward();
  LLM_EXPECT_NEAR(scalar_of(a.grad()), 3.0, 1e-5);
}

LLM_TEST(Autograd, TopologicalOrderPutsConsumersFirst) {
  Var a = leaf(1.0f);
  Var b = llm::autograd::mul_scalar(a, 2.0f);
  Var c = llm::autograd::mul_scalar(b, 3.0f);

  const std::vector<llm::autograd::NodePtr> order =
      llm::autograd::topological_order(c.node());
  LLM_CHECK_EQ(order.size(), static_cast<std::size_t>(3));
  LLM_CHECK(order[0].get() == c.node().get());
  LLM_CHECK(order[1].get() == b.node().get());
  LLM_CHECK(order[2].get() == a.node().get());
}

LLM_TEST(Autograd, GradientShapeMatchesVariable) {
  // Градиент по переменной обязан иметь её форму, даже если операция
  // растянула её до большей.
  Var row = Var::leaf(llm::Tensor::full(llm::Shape({1, 3}), 2.0f), true);
  Var block = Var::leaf(llm::Tensor::full(llm::Shape({4, 3}), 1.0f), true);
  llm::autograd::sum_all(llm::autograd::mul(row, block)).backward();

  LLM_CHECK(row.grad().shape() == llm::Shape({1, 3}));
  LLM_CHECK(block.grad().shape() == llm::Shape({4, 3}));
  // Каждый элемент row поучаствовал в четырёх строках, значит собрал сумму
  // четырёх единиц.
  LLM_EXPECT_NEAR(row.grad()(0, 0), 4.0, 1e-5);
  LLM_EXPECT_NEAR(block.grad()(0, 0), 2.0, 1e-5);
}

LLM_TEST(Autograd, AccumulateDoesNotStealSomeoneElsesBuffer) {
  // Первый вклад в градиент забирается без копии — но только если это
  // временное значение, владеющее своим буфером целиком. Проверка на то, что
  // второе условие работает: вид на живой тензор тоже временное значение, а
  // забирать его нельзя.
  //
  // Без этой проверки ошибка была бы тихой и злой: градиент меняется на месте,
  // и запись в него портила бы данные постороннего тензора.
  llm::Tensor source = llm::Tensor::zeros(llm::Shape({2, 3}));
  for (int64_t i = 0; i < source.numel(); ++i) {
    source.data()[i] = static_cast<float>(i + 1);
  }

  Var variable = Var::leaf(llm::Tensor::zeros(llm::Shape({6})), true);
  // reshape отдаёт вид: временное значение, но буфер общий с source.
  variable.node()->accumulate(source.reshape(llm::Shape({6})));
  variable.node()->scale_grad(0.0f);

  for (int64_t i = 0; i < source.numel(); ++i) {
    LLM_CHECK_MSG(source.data()[i] == static_cast<float>(i + 1),
                  "накопление испортило чужой буфер на элементе " << i);
  }
}

LLM_TEST(Autograd, AccumulateAdoptsATemporaryOfItsOwn) {
  // Обратная сторона: временное значение, владеющее буфером целиком, должно
  // забираться, а не копироваться. Проверяется по указателю — единственный
  // способ отличить забранный буфер от скопированного.
  Var variable = Var::leaf(llm::Tensor::zeros(llm::Shape({4})), true);
  llm::Tensor contribution = llm::Tensor::zeros(llm::Shape({4}));
  const float* address = contribution.data();
  variable.node()->accumulate(std::move(contribution));
  LLM_CHECK_MSG(variable.grad().data() == address,
                "временный буфер скопирован вместо того, чтобы быть забранным");
}

// --- имитация половинной разрядности ----------------------------------------
//
// На этой имитации держится вывод о том, сходится ли обучение в половинной
// разрядности, — и до сих пор её ничто не проверяло. Это хуже, чем отсутствие
// проверки у обычного кода: неработающая имитация не падает, а выдаёт числа
// fp32, и вывод получается «всё сходится», сколь бы разрядность ни была мала.

namespace {

using llm::autograd::set_fp16_simulation;

// Сколько элементов НЕ представимы в половинной разрядности.
int64_t not_representable(const llm::Tensor& tensor) {
  const llm::Tensor dense = tensor.contiguous();
  int64_t count = 0;
  for (int64_t i = 0; i < dense.numel(); ++i) {
    if (llm::round_to_fp16(dense.data()[i]) != dense.data()[i]) {
      ++count;
    }
  }
  return count;
}

llm::Tensor awkward_values(int64_t count) {
  // Значения подобраны так, чтобы в половинной разрядности они НЕ были
  // представимы: иначе проверка прошла бы и при выключенной имитации.
  llm::Tensor out = llm::Tensor::uninitialized(llm::Shape({count}));
  for (int64_t i = 0; i < count; ++i) {
    out.data()[i] = 1.0f / (3.0f + static_cast<float>(i));
  }
  return out;
}

// Переключатель, возвращающий имитацию в исходное состояние при любом выходе.
// Величина глобальная, и оставить её включённой значило бы испортить все
// следующие тесты в наборе.
struct SimulationGuard {
  explicit SimulationGuard(bool enabled) : saved_(llm::autograd::fp16_simulation()) {
    set_fp16_simulation(enabled);
  }
  ~SimulationGuard() { set_fp16_simulation(saved_); }
  bool saved_;
};

}  // namespace

LLM_TEST(Autograd, Fp16SimulationRoundsOperationResults) {
  const Var a = Var::leaf(awkward_values(16), true);
  const Var b = Var::leaf(awkward_values(16), true);

  int64_t rough_without = 0;
  {
    SimulationGuard off(false);
    const Var result = llm::autograd::mul(a, b);
    rough_without = not_representable(result.value());
  }
  // Иначе проверка ниже пуста: значения и так оказались представимыми.
  LLM_CHECK_MSG(rough_without > 0,
                "без имитации все " << 16
                                    << " значений уже представимы — проверка "
                                       "ничего не проверяет");

  SimulationGuard on(true);
  const Var result = llm::autograd::mul(a, b);
  LLM_CHECK_MSG(not_representable(result.value()) == 0,
                "с имитацией осталось "
                    << not_representable(result.value())
                    << " значений, не представимых в половинной разрядности");
}

// Градиенты важнее результатов: именно они проваливаются в нуль на малых
// значениях, и ради них имитация и написана. Обратный проход идёт мимо ленты,
// через accumulate, то есть это отдельный путь и отдельная возможность
// ошибиться.
LLM_TEST(Autograd, Fp16SimulationRoundsGradients) {
  const Var a = Var::leaf(awkward_values(16), true);
  const Var b = Var::leaf(awkward_values(16), true);

  int64_t rough_without = 0;
  {
    SimulationGuard off(false);
    Var loss = llm::autograd::sum_all(llm::autograd::mul(a, b));
    loss.backward();
    rough_without = not_representable(a.grad());
    a.node()->clear_grad();
    b.node()->clear_grad();
  }
  LLM_CHECK_MSG(rough_without > 0,
                "без имитации градиент уже представим — проверка пуста");

  SimulationGuard on(true);
  Var loss = llm::autograd::sum_all(llm::autograd::mul(a, b));
  loss.backward();
  LLM_CHECK_MSG(not_representable(a.grad()) == 0,
                "с имитацией в градиенте осталось "
                    << not_representable(a.grad())
                    << " непредставимых значений");
}

// Вид на параметр имитация трогать не должна.
//
// Reshape, slice по первой оси, select дают тензоры ПЛОТНЫЕ, но делящие
// хранилище с владельцем. Запись в такой вид меняет данные владельца — а
// параметр округлять нельзя: веса в половинной разрядности включаются
// отдельным флагом, и замер сходимости показал, что обучение их не переносит.
//
// Сейчас в модели видов на параметры нет, так что тест сторожит будущее. Он же
// проверяет, что условие не выключило имитацию целиком: соседние тесты
// требуют от неё работы.
LLM_TEST(Autograd, Fp16SimulationLeavesParameterViewsAlone) {
  const Var weight = Var::leaf(awkward_values(12), true);
  const int64_t rough_before = not_representable(weight.value());
  LLM_CHECK_MSG(rough_before > 0, "параметр уже представим — проверка пуста");

  std::vector<float> before(weight.value().numel());
  for (int64_t i = 0; i < weight.value().numel(); ++i) {
    before[static_cast<std::size_t>(i)] = weight.value().data()[i];
  }

  {
    SimulationGuard on(true);
    const Var view = llm::autograd::reshape(weight, llm::Shape({3, 4}));
    // Вид плотный — значит без условия на единоличное владение имитация
    // округлила бы его, а с ним данные параметра.
    LLM_CHECK_MSG(view.value().is_contiguous(),
                  "вид перестал быть плотным, и тест больше ничего не сторожит");
  }

  for (int64_t i = 0; i < weight.value().numel(); ++i) {
    LLM_CHECK_MSG(weight.value().data()[i] == before[static_cast<std::size_t>(i)],
                  "параметр изменился в элементе " << i << ": "
                                                   << weight.value().data()[i]
                                                   << " вместо "
                                                   << before[static_cast<std::size_t>(i)]);
  }
}

// Операции, сохраняющие СВОЙ ВЫХОД для обратного прохода, обязаны округляться
// наравне с остальными.
//
// Проверка появилась после того, как её отсутствие пропустило настоящую
// ошибку. Имитация пропускала тензор, на буфер которого смотрел кто-то ещё, —
// условие выглядело как «это вид на чужие данные», а на деле под него
// попадало и «операция сохранила собственный выход». Так из-под округления
// молча вышли softmax и masked_softmax, то есть самые горячие активации
// модели, и следующий замер сходимости в половинной разрядности оказался бы
// завышенным.
//
// Проверка на mul такого не ловит: он сохраняет ВХОДЫ, а не выход.
LLM_TEST(Autograd, Fp16SimulationRoundsOperationsThatSaveTheirOutput) {
  const Var input = Var::leaf(awkward_values(16), true);

  struct Case {
    const char* name;
    Var (*apply)(const Var&);
  };
  const Case cases[] = {
      {"exp", &llm::autograd::exp},
      {"sqrt", &llm::autograd::sqrt},
      {"softmax", &llm::autograd::softmax},
  };

  for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    int64_t rough_without = 0;
    {
      SimulationGuard off(false);
      rough_without = not_representable(cases[i].apply(input).value());
    }
    LLM_CHECK_MSG(rough_without > 0,
                  cases[i].name
                      << ": без имитации значения уже представимы — проверка "
                         "ничего не проверяет");

    SimulationGuard on(true);
    const Var result = cases[i].apply(input);
    LLM_CHECK_MSG(not_representable(result.value()) == 0,
                  cases[i].name << ": с имитацией осталось "
                                << not_representable(result.value())
                                << " непредставимых значений");
  }
}

// Обратный проход редукции при любом наборе осей.
//
// Градиент суммы размножается обратно по сокращённым осям, и делается это
// через промежуточную форму с сохранёнными осями: растянуть (2, 4) до
// (2, 3, 4) нельзя, а (2, 1, 4) — можно. Форма эта считается по форме входа, и
// проверяется здесь именно она: перечислены оси с конца, по две сразу и все
// разом, а на выходе — градиент формы входа, полный единиц (для суммы) или
// одинаковых долей (для среднего).
LLM_TEST(Autograd, ReductionSpreadsTheGradientBackOverEveryAxisSet) {
  const llm::Shape shape({2, 3, 4});
  const std::vector<std::vector<int>> axis_sets = {
      {0}, {1}, {2}, {-1}, {-2}, {0, 2}, {1, -1}, {0, 1, 2}};

  for (std::size_t s = 0; s < axis_sets.size(); ++s) {
    const std::vector<int>& axes = axis_sets[s];
    int64_t reduced = 1;
    for (std::size_t i = 0; i < axes.size(); ++i) {
      reduced *= shape.dim(axes[i]);
    }

    for (int which = 0; which < 2; ++which) {
      const bool averaging = which == 1;
      for (int keep = 0; keep < 2; ++keep) {
        Var x = Var::leaf(llm::Tensor::full(shape, 2.0f), true);
        const Var reduction =
            averaging ? llm::autograd::mean(x, axes, keep != 0)
                      : llm::autograd::sum(x, axes, keep != 0);
        // Свернуть до скаляра: backward берёт начало только от него, а
        // оставшиеся оси у части наборов ещё есть.
        llm::autograd::sum_all(reduction).backward();

        const llm::Tensor& grad = x.grad();
        LLM_CHECK_MSG(grad.defined(), "набор осей " << s << ": градиента нет");
        LLM_CHECK_MSG(grad.shape() == shape,
                      "набор осей " << s << ": градиент формы " << grad.shape()
                                    << " вместо " << shape);
        const double expected =
            averaging ? 1.0 / static_cast<double>(reduced) : 1.0;
        const llm::Tensor dense = grad.contiguous();
        for (int64_t i = 0; i < dense.numel(); ++i) {
          LLM_EXPECT_NEAR(dense.data()[i], expected, 1e-6);
        }
      }
    }
  }
}
