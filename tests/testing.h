// Минимальный тестовый фреймворк.
//
// Сторонних библиотек в проекте нет, поэтому регистрация и запуск тестов —
// свои. Нужно ровно три вещи: объявить тест, проверить условие, получить
// внятный отчёт о падении.
//
// Проверки внутри тестов — это LLM_CHECK* из core/check.h: они бросают
// CheckFailure, а раннер её ловит. Важно, что LLM_CHECK активен и в Release,
// в отличие от LLM_DCHECK. Поэтому LLM_EXPECT_THROWS применим только к
// условиям на LLM_CHECK: проверки на LLM_DCHECK в Release-сборке исчезают, и
// такой тест падал бы в зависимости от типа сборки.

#ifndef LLM_TESTS_TESTING_H_
#define LLM_TESTS_TESTING_H_

#include <string>
#include <vector>

#include "core/check.h"
#include "core/thread_pool.h"

namespace llm {
namespace testing {

// Возвращает ширину параллелизма к автоматической при любом выходе из области
// видимости — в том числе через исключение.
//
// Без этого непрошедшая проверка внутри ветки с шириной один оставляла бы
// единицу включённой, и ВСЕ последующие тесты двоичного файла считались бы
// в один поток. Хуже всего это именно здесь: молча однопоточными стали бы как
// раз те проверки, которые ищут зависимость от числа потоков. Та же ловушка с
// принудительным выбором микроядра однажды уже сработала — см. ForcedKernel в
// test_gemm.cpp.
//
// Восстанавливается ноль, а не прежнее значение: ноль — это «выбрать
// автоматически», и именно в нём тесты должны друг друга заставать.
// parallel_width() вернул бы уже вычисленное число, и восстановление им
// закрепило бы ширину намертво.
class WidthGuard {
 public:
  explicit WidthGuard(int width) { llm::set_parallel_width(width); }
  ~WidthGuard() { llm::set_parallel_width(0); }

  WidthGuard(const WidthGuard&) = delete;
  WidthGuard& operator=(const WidthGuard&) = delete;
};

using TestFn = void (*)();

struct TestCase {
  std::string suite;
  std::string name;
  TestFn fn;
};

std::vector<TestCase>& registry();

struct Registrar {
  Registrar(const char* suite, const char* name, TestFn fn);
};

// Запускает тесты, чьё имя набора совпадает с argv[1] (если он задан).
// Возвращает 0, если все тесты прошли.
int run_all(int argc, char** argv);

}  // namespace testing
}  // namespace llm

#define LLM_TEST(suite, name)                                     \
  static void llm_test_fn_##suite##_##name();                     \
  static ::llm::testing::Registrar llm_test_reg_##suite##_##name( \
      #suite, #name, &llm_test_fn_##suite##_##name);              \
  static void llm_test_fn_##suite##_##name()

#define LLM_EXPECT_NEAR(a, b, eps)                                             \
  do {                                                                         \
    const double llm_near_a_ = static_cast<double>(a);                         \
    const double llm_near_b_ = static_cast<double>(b);                         \
    const double llm_near_diff_ = llm_near_a_ - llm_near_b_;                   \
    const double llm_near_abs_ =                                               \
        llm_near_diff_ < 0.0 ? -llm_near_diff_ : llm_near_diff_;               \
    if (!(llm_near_abs_ <= static_cast<double>(eps))) {                        \
      std::ostringstream llm_near_oss_;                                        \
      llm_near_oss_ << llm_near_a_ << " и " << llm_near_b_                     \
                    << " расходятся на " << llm_near_abs_ << ", допуск " \
                    << static_cast<double>(eps);                               \
      ::llm::detail::check_failed(__FILE__, __LINE__, #a " ~= " #b,            \
                                  llm_near_oss_.str());                        \
    }                                                                          \
  } while (false)

#define LLM_EXPECT_THROWS(statement)                                    \
  do {                                                                  \
    bool llm_throws_caught_ = false;                                    \
    try {                                                               \
      statement;                                                        \
    } catch (const ::llm::CheckFailure&) {                              \
      llm_throws_caught_ = true;                                        \
    }                                                                   \
    if (!llm_throws_caught_) {                                          \
      ::llm::detail::check_failed(__FILE__, __LINE__, #statement,       \
                                  "ожидалось исключение CheckFailure"); \
    }                                                                   \
  } while (false)

#endif  // LLM_TESTS_TESTING_H_
