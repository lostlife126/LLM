// Проверки времени выполнения.
//
// LLM_CHECK*  — активны всегда, в том числе в Release. Ими проверяются условия,
//               нарушение которых означает ошибку в логике, а не в данных.
// LLM_DCHECK* — активны только в отладочной сборке (когда не задан NDEBUG).
//               Ими проверяются горячие места: индексация тензоров, границы
//               буферов. В Release они исчезают целиком и ничего не стоят.
//
// Нарушение проверки бросает CheckFailure, а не вызывает abort(): так тестовый
// раннер может поймать её и напечатать осмысленный отчёт.

#ifndef LLM_CORE_CHECK_H_
#define LLM_CORE_CHECK_H_

#include <sstream>
#include <stdexcept>
#include <string>

namespace llm {

class CheckFailure : public std::runtime_error {
 public:
  explicit CheckFailure(const std::string& what) : std::runtime_error(what) {}
};

namespace detail {

// Собирает сообщение и бросает CheckFailure. Вынесено из макроса, чтобы
// макрос оставался коротким, а код проверки не раздувал каждую точку вызова.
[[noreturn]] void check_failed(const char* file, int line, const char* expr,
                               const std::string& detail);

}  // namespace detail
}  // namespace llm

#define LLM_CHECK(cond)                                                      \
  do {                                                                       \
    if (!(cond)) {                                                           \
      ::llm::detail::check_failed(__FILE__, __LINE__, #cond, std::string()); \
    }                                                                        \
  } while (false)

// Сообщение задаётся выражением для потока: LLM_CHECK_MSG(ok, "shape " << s);
#define LLM_CHECK_MSG(cond, message)                         \
  do {                                                       \
    if (!(cond)) {                                           \
      std::ostringstream llm_check_oss_;                     \
      llm_check_oss_ << message;                             \
      ::llm::detail::check_failed(__FILE__, __LINE__, #cond, \
                                  llm_check_oss_.str());     \
    }                                                        \
  } while (false)

// Сравнение двух значений с печатью обеих сторон: без этого отчёт "a < b"
// бесполезен, а "3 < 2" сразу говорит, что произошло.
#define LLM_CHECK_OP(a, b, op)                                           \
  do {                                                                   \
    const auto& llm_check_a_ = (a);                                      \
    const auto& llm_check_b_ = (b);                                      \
    if (!(llm_check_a_ op llm_check_b_)) {                               \
      std::ostringstream llm_check_oss_;                                 \
      llm_check_oss_ << llm_check_a_ << " " #op " " << llm_check_b_;     \
      ::llm::detail::check_failed(__FILE__, __LINE__, #a " " #op " " #b, \
                                  llm_check_oss_.str());                 \
    }                                                                    \
  } while (false)

#define LLM_CHECK_EQ(a, b) LLM_CHECK_OP(a, b, ==)
#define LLM_CHECK_NE(a, b) LLM_CHECK_OP(a, b, !=)
#define LLM_CHECK_LT(a, b) LLM_CHECK_OP(a, b, <)
#define LLM_CHECK_LE(a, b) LLM_CHECK_OP(a, b, <=)
#define LLM_CHECK_GT(a, b) LLM_CHECK_OP(a, b, >)
#define LLM_CHECK_GE(a, b) LLM_CHECK_OP(a, b, >=)

#ifdef NDEBUG

// sizeof не вычисляет выражение, но проверяет его типом: опечатка в отключённой
// проверке всё равно не соберётся, а побочных эффектов не возникнет.
#define LLM_DCHECK(cond) \
  do {                   \
    (void)sizeof(cond);  \
  } while (false)
#define LLM_DCHECK_OP(a, b, op) \
  do {                          \
    (void)sizeof((a)op(b));     \
  } while (false)

#else

#define LLM_DCHECK(cond) LLM_CHECK(cond)
#define LLM_DCHECK_OP(a, b, op) LLM_CHECK_OP(a, b, op)

#endif  // NDEBUG

#define LLM_DCHECK_EQ(a, b) LLM_DCHECK_OP(a, b, ==)
#define LLM_DCHECK_NE(a, b) LLM_DCHECK_OP(a, b, !=)
#define LLM_DCHECK_LT(a, b) LLM_DCHECK_OP(a, b, <)
#define LLM_DCHECK_LE(a, b) LLM_DCHECK_OP(a, b, <=)
#define LLM_DCHECK_GT(a, b) LLM_DCHECK_OP(a, b, >)
#define LLM_DCHECK_GE(a, b) LLM_DCHECK_OP(a, b, >=)

#endif  // LLM_CORE_CHECK_H_
