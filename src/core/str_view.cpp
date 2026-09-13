#include "core/str_view.h"

namespace llm {

// Определение вне класса нужно в C++14: без него ссылка на StrView::npos
// (например, при передаче в функцию по const&) не проходит компоновку.
constexpr std::size_t StrView::npos;

}  // namespace llm
