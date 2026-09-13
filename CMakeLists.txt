cmake_minimum_required(VERSION 3.16)
project(llm CXX)

# C++14 без расширений компилятора: проект должен собираться любым
# соответствующим стандарту компилятором, без зависимостей за пределами
# стандартной библиотеки.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Тип сборки" FORCE)
endif()

option(LLM_SANITIZE "Собрать с ASan и UBSan" OFF)

# Предупреждения — ошибки. В проекте, где вся арифметика на индексах и
# указателях, -Wsign-compare и -Wunused ловят реальные баги.
add_compile_options(-Wall -Wextra -Werror)

# -O3 обязателен: единственный источник векторизации здесь — автовекторизатор,
# ручных интринсиков в проекте нет по решению об однопоточности и читаемости.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")

if(LLM_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(llm_core
  src/core/check.cpp
  src/core/random.cpp
  src/core/shape.cpp
  src/core/storage.cpp
  src/core/str_view.cpp
  src/core/tensor.cpp
  src/autograd/node.cpp
  src/autograd/ops.cpp
  src/ops/elementwise.cpp
  src/ops/gemm.cpp
  src/ops/matmul.cpp
  src/ops/reduce.cpp
)
target_include_directories(llm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(bench_gemm apps/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE llm_core)

enable_testing()

add_executable(llm_tests
  tests/gradcheck.cpp
  tests/testing.cpp
  tests/test_main.cpp
  tests/test_autograd.cpp
  tests/test_check.cpp
  tests/test_elementwise.cpp
  tests/test_gemm.cpp
  tests/test_grad.cpp
  tests/test_matmul.cpp
  tests/test_random.cpp
  tests/test_shape.cpp
  tests/test_span.cpp
  tests/test_storage.cpp
  tests/test_str_view.cpp
  tests/test_tensor.cpp
  tests/test_util.cpp
)
target_link_libraries(llm_tests PRIVATE llm_core)
target_include_directories(llm_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Отдельная запись ctest на каждый набор: при падении сразу видно, какой
# компонент сломан, без чтения общего лога.
set(LLM_TEST_SUITES
  Autograd
  Check
  Elementwise
  Grad
  Gemm
  Matmul
  Rng
  Shape
  Span
  Storage
  StrView
  Tensor
  Util
)
foreach(suite ${LLM_TEST_SUITES})
  add_test(NAME ${suite} COMMAND llm_tests ${suite})
endforeach()
