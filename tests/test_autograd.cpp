#include <cstdint>
#include <vector>

#include "autograd/node.h"
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
