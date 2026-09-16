set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Кросс-сборка под aarch64 — для Raspberry Pi 5.
#
#   cmake -S . -B build-arm -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64.cmake
#   cmake --build build-arm -j4
#   qemu-aarch64-static -L /usr/aarch64-linux-gnu ./build-arm/llm_tests
#
# Эмуляция проверяет правильность, но не скорость: время под ней не значит
# ничего. Замер производительности делается на самой машине.
