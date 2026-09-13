#include "testing.h"

#include <exception>
#include <iostream>

namespace llm {
namespace testing {

std::vector<TestCase>& registry() {
  // Статик внутри функции: порядок инициализации глобальных объектов между
  // единицами трансляции не определён, а сюда пишут конструкторы Registrar из
  // разных файлов. Ленивая инициализация снимает эту проблему.
  static std::vector<TestCase> tests;
  return tests;
}

Registrar::Registrar(const char* suite, const char* name, TestFn fn) {
  TestCase test;
  test.suite = suite;
  test.name = name;
  test.fn = fn;
  registry().push_back(test);
}

int run_all(int argc, char** argv) {
  const std::string filter = argc > 1 ? std::string(argv[1]) : std::string();

  std::size_t passed = 0;
  std::vector<std::string> failed;

  for (std::size_t i = 0; i < registry().size(); ++i) {
    const TestCase& test = registry()[i];
    if (!filter.empty() && test.suite != filter) {
      continue;
    }
    const std::string full_name = test.suite + "." + test.name;
    std::cout << "[ RUN  ] " << full_name << std::endl;

    std::string error;
    try {
      test.fn();
    } catch (const std::exception& e) {
      error = e.what();
    } catch (...) {
      error = "неизвестное исключение";
    }

    if (error.empty()) {
      ++passed;
      std::cout << "[   OK ] " << full_name << std::endl;
    } else {
      failed.push_back(full_name);
      std::cout << "[ FAIL ] " << full_name << "\n         " << error
                << std::endl;
    }
  }

  std::cout << "\nпройдено: " << passed << ", упало: " << failed.size()
            << std::endl;
  for (std::size_t i = 0; i < failed.size(); ++i) {
    std::cout << "  упал: " << failed[i] << std::endl;
  }

  if (passed == 0 && failed.empty()) {
    std::cout << "не выбрано ни одного теста (фильтр: '" << filter << "')"
              << std::endl;
    return 1;
  }
  return failed.empty() ? 0 : 1;
}

}  // namespace testing
}  // namespace llm
