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

# Санитайзер гонок — отдельной ручкой, потому что с ASan он несовместим.
#
# Заведён не для полноты набора. Главное обещание проекта — результат не
# зависит от числа потоков, — держится на разделении работы без общих ячеек, и
# проверить это чтением нельзя: гонка не падает, а изредка сдвигает младшие
# разряды. Прогон всего набора под ним стоит несколько минут.
#
#   cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#         -DLLM_SANITIZE_THREAD=ON && cmake --build build-tsan -j8
option(LLM_SANITIZE_THREAD "Собрать с TSan (несовместим с LLM_SANITIZE)" OFF)

# Предупреждения — ошибки. В проекте, где вся арифметика на индексах и
# указателях, -Wsign-compare и -Wunused ловят реальные баги.
add_compile_options(-Wall -Wextra -Werror)

# Слитное умножение с накоплением — только там, где оно написано явно.
#
# По умолчанию компилятор вправе склеить a += b * c в одну команду, и это не
# то же самое число: у слитной операции одно округление вместо двух. Правом
# этим он пользуется там, где команда есть: на aarch64 она входит в базовый
# набор, на базовом x86-64 её нет. Выходит, что арифметику выбирает не
# исходник, а процессор, — а проект обещает обратное.
#
# Измерено, а не выведено из рассуждения. Двадцать шагов nano с одним зерном:
# чекпоинт с x86 и чекпоинт с aarch64 (под qemu) расходились в 482502 байтах
# из 3479950 — то есть на одном и том же исходнике и одном зерне получались
# разные модели. С этим флагом те же двадцать шагов дают побайтово один и тот
# же чекпоинт.
#
# Микроядра это не затрагивает: слитная операция написана в них интринсиками
# и std::fma, а запрет касается только того, что компилятор делает САМ. На
# x86 флаг не меняет ни результата (чекпоинты совпали), ни времени (600 шагов
# nano: 20.8 с против 20.8 с).
add_compile_options(-ffp-contract=off)

# -O3 обязателен. Векторные микроядра написаны интринсиками и собираются под
# свои наборы инструкций через атрибут target, но всё вокруг них — упаковку,
# нормировки, softmax — векторизует компилятор, и без -O3 он этого не делает.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")

if(LLM_SANITIZE AND LLM_SANITIZE_THREAD)
  message(FATAL_ERROR "LLM_SANITIZE и LLM_SANITIZE_THREAD несовместимы: "
                      "ASan и TSan не работают в одной сборке")
endif()

if(LLM_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

if(LLM_SANITIZE_THREAD)
  add_compile_options(-fsanitize=thread)
  add_link_options(-fsanitize=thread)
endif()

add_library(llm_core
  src/core/check.cpp
  src/core/cpu.cpp
  src/core/dims.cpp
  src/core/half.cpp
  src/core/quantize.cpp
  src/core/random.cpp
  src/core/shape.cpp
  src/core/storage.cpp
  src/core/str_view.cpp
  src/core/thread_pool.cpp
  src/core/tensor.cpp
  src/autograd/nn.cpp
  src/autograd/node.cpp
  src/autograd/ops.cpp
  src/ops/dropout.cpp
  src/ops/elementwise.cpp
  src/ops/embedding.cpp
  src/ops/fast_exp.cpp
  src/ops/loss.cpp
  src/ops/nn.cpp
  src/ops/rope.cpp
  src/ops/gemm.cpp
  src/ops/micro_kernel.cpp
  src/ops/matmul.cpp
  src/ops/reduce.cpp
  src/nn/config.cpp
  src/nn/dropout.cpp
  src/nn/kv_cache.cpp
  src/nn/model.cpp
  src/tokenizer/bpe.cpp
  src/data/dataset.cpp
  src/serialize/checkpoint.cpp
  src/tokenizer/hf_tokenizer.cpp
  src/tokenizer/pretokenize.cpp
  src/tokenizer/unicode_tables.cpp
  src/tokenizer/utf8.cpp
  src/serialize/hf_config.cpp
  src/serialize/hf_import.cpp
  src/serialize/json.cpp
  src/serialize/safetensors.cpp
  src/train/optimizer.cpp
  src/train/schedule.cpp
  src/train/resume.cpp
  src/train/trainer.cpp
  src/infer/sampler.cpp
  src/infer/generate.cpp
)
target_include_directories(llm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Потоки. Ищутся через find_package, а не флагом -pthread напрямую: флаг у
# разных систем разный, а зависимость обязана дойти до всех, кто ссылается на
# llm_core, — поэтому PUBLIC.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(llm_core PUBLIC Threads::Threads)

add_executable(bench_gemm apps/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE llm_core)

add_executable(bench_exp apps/bench_exp.cpp)
target_link_libraries(bench_exp PRIVATE llm_core)

add_executable(bench_fp8 apps/bench_fp8.cpp)
target_link_libraries(bench_fp8 PRIVATE llm_core)

add_executable(bench_half apps/bench_half.cpp)
target_link_libraries(bench_half PRIVATE llm_core)

add_executable(bench_qk apps/bench_qk.cpp)
target_link_libraries(bench_qk PRIVATE llm_core)

add_executable(precision_scope apps/precision_scope.cpp)
target_link_libraries(precision_scope PRIVATE llm_core)

add_executable(lora_scope apps/lora_scope.cpp)
target_link_libraries(lora_scope PRIVATE llm_core)

add_executable(fingerprint apps/fingerprint.cpp)
target_link_libraries(fingerprint PRIVATE llm_core)

add_executable(bpe_train apps/bpe_train.cpp)
target_link_libraries(bpe_train PRIVATE llm_core)

add_executable(bench_generate apps/bench_generate.cpp)
target_link_libraries(bench_generate PRIVATE llm_core)

add_executable(bench_model apps/bench_model.cpp)
target_link_libraries(bench_model PRIVATE llm_core)

add_executable(train apps/train.cpp)
target_link_libraries(train PRIVATE llm_core)

add_executable(generate apps/generate.cpp)
target_link_libraries(generate PRIVATE llm_core)

add_executable(ablate apps/ablate.cpp)
target_link_libraries(ablate PRIVATE llm_core)

add_executable(finetune apps/finetune.cpp)
target_link_libraries(finetune PRIVATE llm_core)

add_executable(import_hf apps/import_hf.cpp)
target_link_libraries(import_hf PRIVATE llm_core)

enable_testing()

add_executable(llm_tests
  tests/gradcheck.cpp
  tests/testing.cpp
  tests/test_main.cpp
  tests/test_autograd.cpp
  tests/test_bpe.cpp
  tests/test_check.cpp
  tests/test_dropout.cpp
  tests/test_elementwise.cpp
  tests/test_fast_exp.cpp
  tests/test_fp16.cpp
  tests/test_fp8.cpp
  tests/test_gemm.cpp
  tests/test_grad.cpp
  tests/test_half.cpp
  tests/test_hf.cpp
  tests/test_hf_tokenizer.cpp
  tests/test_infer.cpp
  tests/test_iterate.cpp
  tests/test_json.cpp
  tests/test_lora.cpp
  tests/test_pretokenize.cpp
  tests/test_quantize.cpp
  tests/test_safetensors.cpp
  tests/test_matmul.cpp
  tests/test_model.cpp
  tests/test_nn.cpp
  tests/test_random.cpp
  tests/test_shape.cpp
  tests/test_span.cpp
  tests/test_storage.cpp
  tests/test_str_view.cpp
  tests/test_tensor.cpp
  tests/test_threads.cpp
  tests/test_train.cpp
  tests/test_variants.cpp
  tests/test_util.cpp
)
target_link_libraries(llm_tests PRIVATE llm_core)
target_include_directories(llm_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Отдельная запись ctest на каждый набор: при падении сразу видно, какой
# компонент сломан, без чтения общего лога.
#
# Список выписан руками, и это его слабое место: набор, забытый здесь, ctest
# молча не запускает, тогда как ./llm_tests без фильтра его прогоняет. Так и
# вышло с Cpu и Iterate — 34 набора из 36. Поэтому ниже добавлена ещё одна
# запись, прогоняющая двоичный файл целиком: список остался ради
# гранулярности отчёта, но правильность на нём больше не держится.
set(LLM_TEST_SUITES
  Autograd
  Bpe
  Check
  Cpu
  Dropout
  Elementwise
  FastExp
  Fp16
  Fp8
  Grad
  Half
  Hf
  HfTokenizer
  Infer
  Iterate
  Json
  Linear
  Lora
  Gemm
  Matmul
  Model
  Nn
  Pretokenize
  Quantize
  Rng
  Safetensors
  Sampler
  Shape
  Span
  Storage
  StrView
  Tensor
  Threads
  Train
  Variants
  Util
)
foreach(suite ${LLM_TEST_SUITES})
  add_test(NAME ${suite} COMMAND llm_tests ${suite})
endforeach()

# Прогон целиком — страховка от забытого набора. Запускает ровно то же, что
# разработчик запускает руками, и стоит те же три секунды.
add_test(NAME ВСЕ COMMAND llm_tests)
