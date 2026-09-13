// Численная проверка обратного прохода каждой операции.
//
// Эталонной реализации у проекта нет, и эти тесты её заменяют: центральная
// разность ничего не знает про формулу производной, поэтому ловит ошибки,
// которые при чтении кода выглядят правдоподобно.

#include <cstdint>
#include <vector>

#include "autograd/ops.h"
#include "gradcheck.h"
#include "testing.h"

namespace {

using llm::autograd::Var;
using llm::testing::random_tensor;
using llm::testing::weighted_sum;

}  // namespace

LLM_TEST(Grad, Add) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::add(v[0], v[1]), 101);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3}), 1),
                                    random_tensor(llm::Shape({2, 3}), 2)}));
}

LLM_TEST(Grad, AddWithBroadcast) {
  // Второй аргумент растягивается по первой оси: его градиент обязан быть
  // суммой по всем строкам, а не одной из них.
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::add(v[0], v[1]), 102);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({4, 3}), 3),
                                    random_tensor(llm::Shape({1, 3}), 4)}));
}

LLM_TEST(Grad, AddWithRankBroadcast) {
  // Аргумент меньшего ранга: выравнивание справа добавляет ось слева.
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::add(v[0], v[1]), 103);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3, 4}), 5),
                                    random_tensor(llm::Shape({3, 4}), 6)}));
}

LLM_TEST(Grad, Sub) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::sub(v[0], v[1]), 104);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 2}), 7),
                                    random_tensor(llm::Shape({3, 2}), 8)}));
}

LLM_TEST(Grad, Mul) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::mul(v[0], v[1]), 105);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3}), 9),
                                    random_tensor(llm::Shape({2, 3}), 10)}));
}

LLM_TEST(Grad, MulWithBroadcast) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::mul(v[0], v[1]), 106);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 4}), 11),
                                    random_tensor(llm::Shape({3, 1}), 12)}));
}

LLM_TEST(Grad, Div) {
  // Знаменатель держим подальше от нуля: рядом с ним производная взрывается,
  // и численная оценка теряет смысл раньше, чем аналитическая.
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::div(v[0], v[1]), 107);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>(
              {random_tensor(llm::Shape({2, 3}), 13),
               random_tensor(llm::Shape({2, 3}), 14, 1.0f, 2.0f)}));
}

LLM_TEST(Grad, ScalarOps) {
  const auto fn = [](const std::vector<Var>& v) {
    const Var shifted = llm::autograd::add_scalar(v[0], 0.75f);
    const Var scaled = llm::autograd::mul_scalar(shifted, -2.5f);
    return weighted_sum(llm::autograd::neg(scaled), 108);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3}), 15)}));
}

LLM_TEST(Grad, Exp) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::exp(v[0]), 109);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3}), 16)}));
}

LLM_TEST(Grad, LogAndSqrt) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(
        llm::autograd::add(llm::autograd::log(v[0]), llm::autograd::sqrt(v[0])),
        110);
  };
  LLM_EXPECT_GRADCHECK(fn, std::vector<llm::Tensor>({random_tensor(
                               llm::Shape({2, 3}), 17, 0.5f, 2.0f)}));
}

LLM_TEST(Grad, SumOverAxis) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::sum(v[0], {1}, false), 111);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 4}), 18)}));
}

LLM_TEST(Grad, SumKeepdim) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::sum(v[0], {-1}, true), 112);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3, 4}), 19)}));
}

LLM_TEST(Grad, Mean) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::mean(v[0], {0}, false), 113);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({4, 3}), 20)}));
}

LLM_TEST(Grad, Matmul) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::matmul(v[0], v[1]), 114);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 4}), 21),
                                    random_tensor(llm::Shape({4, 2}), 22)}));
}

LLM_TEST(Grad, MatmulBatched) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::matmul(v[0], v[1]), 115);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3, 4}), 23),
                                    random_tensor(llm::Shape({2, 4, 2}), 24)}));
}

LLM_TEST(Grad, MatmulSharedWeight) {
  // Самый важный случай: один вес на весь батч. Его градиент — сумма вкладов
  // всех элементов батча, и если суммирования нет, обучение молча идёт не
  // туда.
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::matmul(v[0], v[1]), 116);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 2, 4}), 25),
                                    random_tensor(llm::Shape({4, 3}), 26)}));
}

LLM_TEST(Grad, Reshape) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::reshape(v[0], llm::Shape({3, 2})), 117);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3}), 27)}));
}

LLM_TEST(Grad, Transpose) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::transpose(v[0], 0, 1), 118);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3}), 28)}));
}

LLM_TEST(Grad, Permute) {
  // Ровно та перестановка, которой attention переставляет головы и позиции.
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::permute(v[0], {0, 2, 1, 3}), 119);
  };
  LLM_EXPECT_GRADCHECK(fn, std::vector<llm::Tensor>(
                               {random_tensor(llm::Shape({2, 3, 2, 2}), 29)}));
}

LLM_TEST(Grad, Slice) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::slice(v[0], 1, 1, 2), 120);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 4}), 30)}));
}

LLM_TEST(Grad, SliceGradientIsZeroOutside) {
  Var input = Var::leaf(random_tensor(llm::Shape({2, 4}), 31), true);
  llm::autograd::sum_all(llm::autograd::slice(input, 1, 1, 2)).backward();
  LLM_EXPECT_NEAR(input.grad()(0, 0), 0.0, 1e-6);
  LLM_EXPECT_NEAR(input.grad()(0, 1), 1.0, 1e-6);
  LLM_EXPECT_NEAR(input.grad()(0, 2), 1.0, 1e-6);
  LLM_EXPECT_NEAR(input.grad()(0, 3), 0.0, 1e-6);
}

LLM_TEST(Grad, Cat) {
  const auto fn = [](const std::vector<Var>& v) {
    return weighted_sum(llm::autograd::cat({v[0], v[1]}, 1), 121);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({2, 3}), 32),
                                    random_tensor(llm::Shape({2, 2}), 33)}));
}

LLM_TEST(Grad, ChainOfOperations) {
  // Составная функция: ошибка в любом звене вылезет здесь, даже если каждое
  // звено по отдельности прошло проверку.
  const auto fn = [](const std::vector<Var>& v) {
    const Var product = llm::autograd::matmul(v[0], v[1]);
    const Var activated =
        llm::autograd::exp(llm::autograd::mul_scalar(product, 0.5f));
    const Var pooled = llm::autograd::mean(activated, {1}, true);
    return weighted_sum(llm::autograd::div(activated, pooled), 122);
  };
  LLM_EXPECT_GRADCHECK(
      fn, std::vector<llm::Tensor>({random_tensor(llm::Shape({3, 4}), 34),
                                    random_tensor(llm::Shape({4, 3}), 35)}));
}
