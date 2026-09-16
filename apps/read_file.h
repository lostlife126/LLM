// Чтение файла целиком — одно на все приложения.
//
// Было скопировано в пяти программах слово в слово.

#ifndef LLM_APPS_READ_FILE_H_
#define LLM_APPS_READ_FILE_H_

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace bench {

inline std::string read_file(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::binary);
  if (!file.good()) {
    std::fprintf(stderr, "не удалось открыть %s\n", path.c_str());
    std::exit(1);
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

}  // namespace bench

#endif  // LLM_APPS_READ_FILE_H_
