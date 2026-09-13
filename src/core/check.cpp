#include "core/check.h"

namespace llm {
namespace detail {

void check_failed(const char* file, int line, const char* expr,
                  const std::string& detail) {
  std::ostringstream oss;
  oss << file << ":" << line << ": проверка не прошла: " << expr;
  if (!detail.empty()) {
    oss << " (" << detail << ")";
  }
  throw CheckFailure(oss.str());
}

}  // namespace detail
}  // namespace llm
