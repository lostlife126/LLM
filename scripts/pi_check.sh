#!/bin/sh
# Сборка и проверка половинной разрядности на Raspberry Pi.
#
# Что делает: собирает, прогоняет тесты, меряет генерацию в обычной и
# половинной разрядности и печатает то, без чего числа нельзя истолковать —
# признаки процессора, размеры кэшей, память, температуру и состояние
# ограничения частоты.
#
# Зачем столько про окружение. Pi умеет тихо снижать частоту при нагреве и при
# слабом питании, и тогда второй замер выходит медленнее первого просто
# потому, что он второй. Температура и get_throttled печатаются до и после
# именно поэтому: если они разошлись, числам между ними верить нельзя.
#
# Запуск:   sh scripts/pi_check.sh          — около пяти минут
#           sh scripts/pi_check.sh --full   — плюс замер шага обучения
#
# Вывод целиком можно копировать как есть.

# Десятичный разделитель. На системе с русской локалью числа печатаются через
# запятую, и всё, что их потом читает, тихо портится. Здесь это уже случалось.
LC_ALL=C
export LC_ALL

FULL=0
[ "$1" = "--full" ] && FULL=1

say() {
  echo
  echo "=============================================================="
  echo "$1"
  echo "=============================================================="
}

# Не падать на отсутствующей команде: на голой системе половины утилит нет, и
# это само по себе сведение, а не повод остановиться.
have() { command -v "$1" >/dev/null 2>&1; }

say "1. Железо и система"
uname -a
echo
if have lscpu; then
  lscpu | grep -iE "model name|architecture|cpu\(s\)|mhz|cache|byte order" | sed 's/^/  /'
else
  grep -m1 "Model" /proc/cpuinfo
  grep -m1 "model name" /proc/cpuinfo
fi
echo
echo "-- признаки набора инструкций"
echo "   (fphp и asimdhp означают арифметику половинной разрядности; для"
echo "    того, что здесь используется, довольно одного преобразования,"
echo "    а оно есть в базовом ARMv8-A всегда)"
if grep -q "Features" /proc/cpuinfo; then
  grep -m1 "Features" /proc/cpuinfo | tr ' ' '\n' | grep -E "asimd|^fp|half|hp$" | tr '\n' ' '
  echo
else
  echo "   строки Features нет — значит это не ARM"
fi
echo
echo "-- размеры кэшей"
for d in /sys/devices/system/cpu/cpu0/cache/index*; do
  [ -d "$d" ] || continue
  printf "  L%s %s: %s" "$(cat $d/level)" "$(cat $d/type)" "$(cat $d/size)"
  [ -f "$d/shared_cpu_list" ] && printf "  (общий для ядер %s)" "$(cat $d/shared_cpu_list)"
  echo
done
echo
echo "-- память"
free -m | sed 's/^/  /'
echo
echo "-- режим частоты"
found=0
for c in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  if [ -f "$c" ]; then
    echo "  $c = $(cat $c)"
    found=1
  fi
done
[ "$found" = "0" ] && echo "  cpufreq недоступен"
if have vcgencmd; then
  echo "  частота ядра: $(vcgencmd measure_clock arm)"
fi

say "2. Состояние ДО замеров"
if have vcgencmd; then
  echo "  температура: $(vcgencmd measure_temp)"
  echo "  ограничение: $(vcgencmd get_throttled)"
  echo "  (0x0 — всё хорошо; ненулевое значит, что частота снижалась)"
else
  [ -f /sys/class/thermal/thermal_zone0/temp ] &&
    echo "  температура: $(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 )) C"
  echo "  vcgencmd не найден, состояние ограничения частоты неизвестно"
fi
echo
echo "-- посторонняя нагрузка (должно быть почти пусто)"
uptime
ps -eo pcpu,comm --sort=-pcpu 2>/dev/null | head -6 | sed 's/^/  /'

say "3. Репозиторий"
git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "  не репозиторий?"
git pull --ff-only 2>&1 | tail -3
echo
git log --oneline -3

say "4. Сборка"
if ! have cmake; then
  echo "  cmake не найден: sudo apt install -y cmake build-essential"
  exit 1
fi
echo "  компилятор: $(c++ --version | head -1)"
echo "  cmake:      $(cmake --version | head -1)"
echo
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
# Сборка на четырёх ядрах и восьми гигабайтах проходит, но с запасом небольшим.
#
# Вывод складывается в переменную, а не пропускается через grep напрямую: у
# конвейера состояние выхода берётся от последней команды, то есть от grep, и
# неудавшаяся сборка проходила незамеченной. Скрипт шёл дальше и мерил СТАРЫЕ
# двоичные файлы, а числа при этом выглядели как настоящие.
build_log=$(cmake --build build-release -j4 2>&1) || {
  echo "$build_log" | grep -iE "error" | head -20
  echo "  СБОРКА НЕ УДАЛАСЬ — дальше мерить нечего"
  exit 1
}
echo "$build_log" | grep -iE "error|warning" | head -20
echo "  сборка закончена"

say "5. Тесты"
# Половинная разрядность и умножение матриц идут первыми и полностью: именно
# они отвечают за то, что число ниже — правда.
./build-release/llm_tests Half 2>&1 | tail -3
./build-release/llm_tests Gemm 2>&1 | tail -3
./build-release/llm_tests Model 2>&1 | tail -3
echo
echo "-- весь набор"
( cd build-release && ctest 2>&1 | tail -4 )

say "5b. Отпечатки: совпадает ли арифметика с другой машиной"
echo "Главная проверка этого прогона, и единственная, которую нельзя"
echo "сделать на одной машине: на x86 нет NEON, на ARM нет AVX2, и тест,"
echo "запущенный там и там, сравнивает разные пары. Здесь сравниваются"
echo "результаты, а не ядра."
echo
echo "Ожидается (снято на x86 с AVX2/AVX-512 и на aarch64 под qemu —"
echo "все значения там совпали):"
echo "  скалярное 4x32   b4d6e0cfa3c5b5ac"
echo "  NEON 12x8        77784ea338f9d63a"
echo "  эталонная        c5d9a2184fed98d5"
echo "  NEON x4          d8c845393765229e"
echo "  libm             f70948b288fd626e  <- см. оговорку ниже"
echo
./build-release/fingerprint 2>&1
echo
echo "Про libm отдельно: это чужая функция, и стандарт не определяет"
echo "std::exp до последнего разряда. На Pi своя версия glibc, и разойтись"
echo "она вправе — это НЕ поломка. Все остальные строки разойтись не вправе:"
echo "если хоть одна из четырёх не совпала, это настоящая находка, и"
echo "числа ниже можно не читать."

say "6. Выбранное микроядро и скорость умножения матриц"
LLM_THREADS=4 ./build-release/bench_gemm 2>&1 | head -24

say "6b. Прямой путь против блочного"
echo "Порог между ними записан константой, подобранной на x86, и переносить"
echo "такое число с машины на машину нельзя. Прямой путь не упаковывает, но"
echo "читает B с шагом в целую строку; блочный упаковывает, зато с этого"
echo "коммита читает подряд. Отношение больше единицы — прямой быстрее."
LLM_THREADS=4 ./build-release/bench_gemm 2>&1 | sed -n '/прямой путь против/,/наоборот/p'

say "7. Генерация: обычная разрядность против половинной"
echo "Главное число этого прогона. Скорость от значений весов не зависит,"
echo "поэтому обученный чекпоинт не нужен."
# Число токенов подобрано под размер: крупная модель и так медленная, а
# замеряется скорость на токен, а не длина текста.
for pair in "nano 200" "tiny 100" "small 60"; do
  set -- $pair
  echo
  echo "---- $1 ----"
  LLM_THREADS=4 ./build-release/bench_generate "$1" "$2" 2>&1
done

say "7b. Занимает ли генерация все ядра"
echo "Прямой путь умножения долгое время умел делить работу только по строкам,"
echo "а при генерации строка одна. Если числа ниже не растут с числом потоков,"
echo "значит работа снова достаётся одному ядру."
for t in 1 2 4; do
  echo
  echo "-- потоков $t"
  LLM_THREADS=$t ./build-release/bench_generate small 40 2>&1 | tail -4
done

say "8. Умножение матриц: веса в половинной разрядности"
echo "Разбор по формам. Ожидается выигрыш при m <= 8 и проигрыш при m от 16"
echo "до 64 — это устройство путей, а не поломка; объяснение в ops/gemm.h."
LLM_THREADS=1 ./build-release/bench_half 2>&1 | head -14

say "9. Восьмиразрядные веса: стоит ли"
echo "Качество уже проверено — E4M3 стоит две десятых процента перплексии."
echo "Здесь меряется второе: у fp16 на развёртывание есть одна команда, у fp8"
echo "нет ни одной ни на x86, ни на этом ARM (FEAT_FP8 появился в ARMv9.2, а"
echo "Cortex-A76 — это ARMv8.2). Решает отношение в пакетной распаковке."
LLM_THREADS=1 ./build-release/bench_fp8 2>&1 | head -14

if [ "$FULL" = "1" ]; then
  say "10. Шаг обучения (только с --full)"
  # Длина окна аргументом не задаётся: bench_model берёт её у пресета, а у
  # tiny она и так 128. Третий аргумент здесь стоял и молча отбрасывался.
  LLM_THREADS=4 ./build-release/bench_model tiny 8 2>&1 | tail -14
fi

say "Состояние ПОСЛЕ замеров"
if have vcgencmd; then
  echo "  температура: $(vcgencmd measure_temp)"
  echo "  ограничение: $(vcgencmd get_throttled)"
  echo "  частота ядра: $(vcgencmd measure_clock arm)"
else
  [ -f /sys/class/thermal/thermal_zone0/temp ] &&
    echo "  температура: $(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 )) C"
fi
echo
echo "Если ограничение перестало быть 0x0 или температура выросла сильно,"
echo "числа из разделов 6-8 сравнивать между собой нельзя."
