// Численная проверка градиентов.
//
// Для каждого элемента каждого входа считается центральная разность
// (f(x + h) - f(x - h)) / 2h и сравнивается с градиентом, который выдал
// обратный проход. Проверка ничего не знает об устройстве операции, поэтому
// ловит ошибки, которые невозможно заметить, глядя на формулу.
//
// О достижимой точности. Данные здесь float32, и это ставит предел снизу:
// вычитание двух близких значений f теряет значащие цифры (ошибка порядка
// eps * |f| / h при машинной точности 1.2e-7), а сама центральная разность
// приближает производную с ошибкой порядка h^2. Шаг h = 1e-2 балансирует эти
// два источника.
//
// Допуск подобран замером, а не оценкой сверху. На всех проверяемых операциях
// с заведомо верным обратным проходом расхождение лежит в диапазоне от 2e-6 до
// 7.6e-5, поэтому 1e-3 оставляет больше чем десятикратный запас и при этом
// ловит ошибку уже в десятые доли процента. Для сравнения: искусственно
// внесённая ошибка в 5% даёт расхождение 0.03, то есть обнаруживается с
// тридцатикратным запасом.
//
// Если какая-то операция потребует большего допуска, его лучше передать явно
// в этом тесте, чем ослаблять значение по умолчанию для всех.

#ifndef LLM_TESTS_GRADCHECK_H_
#define LLM_TESTS_GRADCHECK_H_

#include <functional>
#include <string>
#include <vector>

#include "autograd/node.h"
#include "core/tensor.h"

namespace llm {
namespace testing {

struct GradCheckResult {
  bool ok = false;
  double max_error = 0.0;
  std::string detail;
};

// fn получает переменные, построенные из inputs, и обязана вернуть скаляр.
GradCheckResult gradcheck(
    const std::function<autograd::Var(const std::vector<autograd::Var>&)>& fn,
    const std::vector<Tensor>& inputs, float step = 1e-2f,
    double tolerance = 1e-3);

}  // namespace testing
}  // namespace llm

// Проверяет градиенты и печатает подробности при расхождении.
#define LLM_EXPECT_GRADCHECK(fn, inputs)            \
  do {                                              \
    const ::llm::testing::GradCheckResult llm_gc_ = \
        ::llm::testing::gradcheck((fn), (inputs));  \
    LLM_CHECK_MSG(llm_gc_.ok, llm_gc_.detail);      \
  } while (false)

#endif  // LLM_TESTS_GRADCHECK_H_
