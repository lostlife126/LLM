#include "ops/gemm.h"

#include <algorithm>
#include <vector>

#include "core/check.h"
#include "core/thread_pool.h"
#include "ops/micro_kernel.h"

namespace llm {
namespace ops {
namespace {

// Форма плитки больше не константа: её задаёт выбранное микроядро, и от неё
// зависит раскладка упакованных панелей. Параметризация обязательна — иначе
// смена ядра молча поломала бы раскладку, а проявилось бы это неверными
// числами, а не ошибкой.

// Размеры блоков. Смысл — удержать переиспользуемые данные в нужном уровне
// кэша:
//   блок B  kBlockK x kBlockN = 128 x 256 float = 128 КБ — рассчитан на L2,
//           он переиспользуется всеми блоками строк A;
//   панель A kBlockM x kBlockK = 64 x 128 float = 32 КБ;
//   рабочий набор микроядра — 4 x 128 от A плюс 128 x 32 от B = 18 КБ,
//           рассчитан на L1 (32 КБ).
constexpr int64_t kBlockM = 64;
constexpr int64_t kBlockN = 256;
constexpr int64_t kBlockK = 128;

int64_t ceil_div(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

// Упаковка панели A: блок mc x kc матрицы op(A), разложенный панелями по
// mr строк. Внутри панели порядок (p, ii) — сначала шаг по глубине,
// потом по строке.
//
// Упаковка делает три вещи сразу:
//   1) микроядро читает память подряд, независимо от исходных шагов;
//   2) транспонирование разрешается здесь, один раз, а не в каждой итерации
//      внутреннего цикла;
//   3) хвост дополняется нулями до полной панели, поэтому микроядро не
//      нуждается в проверках границ — нули не портят сумму.
void pack_a(bool transpose_a, const float* a, int64_t lda, int64_t row0,
            int64_t col0, int64_t mc, int64_t kc, int64_t mr, float* apack) {
  const int64_t panels = ceil_div(mc, mr);
  for (int64_t panel = 0; panel < panels; ++panel) {
    float* dst = apack + panel * kc * mr;
    for (int64_t p = 0; p < kc; ++p) {
      for (int64_t ii = 0; ii < mr; ++ii) {
        const int64_t i = panel * mr + ii;
        float value = 0.0f;
        if (i < mc) {
          value = transpose_a ? a[(col0 + p) * lda + (row0 + i)]
                              : a[(row0 + i) * lda + (col0 + p)];
        }
        dst[p * mr + ii] = value;
      }
    }
  }
}

// Упаковка блока B: kc x nc матрицы op(B), панелями по nr столбцов.
void pack_b(bool transpose_b, const float* b, int64_t ldb, int64_t row0,
            int64_t col0, int64_t kc, int64_t nc, int64_t nr, float* bpack) {
  const int64_t panels = ceil_div(nc, nr);
  for (int64_t panel = 0; panel < panels; ++panel) {
    float* dst = bpack + panel * kc * nr;
    for (int64_t p = 0; p < kc; ++p) {
      for (int64_t jj = 0; jj < nr; ++jj) {
        const int64_t j = panel * nr + jj;
        float value = 0.0f;
        if (j < nc) {
          value = transpose_b ? b[(col0 + j) * ldb + (row0 + p)]
                              : b[(row0 + p) * ldb + (col0 + j)];
        }
        dst[p * nr + jj] = value;
      }
    }
  }
}

// Применение beta к C. Отдельным проходом до накопления: иначе при разбиении
// по глубине k масштабирование применилось бы к каждому блоку.
// beta == 0 означает перезапись, а не умножение на нуль: в C может лежать
// мусор или NaN, и умножение оставило бы NaN.
void scale_c(int64_t m, int64_t n, float beta, float* c, int64_t ldc) {
  if (beta == 1.0f) {
    return;
  }
  for (int64_t i = 0; i < m; ++i) {
    float* row = c + i * ldc;
    if (beta == 0.0f) {
      std::fill(row, row + n, 0.0f);
    } else {
      for (int64_t j = 0; j < n; ++j) {
        row[j] *= beta;
      }
    }
  }
}

// Буферы упаковки живут в потоке, а не в вызове.
//
// Обучение делает порядка сотни умножений на шаг, и на формах внимания каждое
// занимает десятки микросекунд — столько же, сколько стоит пара выделений
// памяти под буферы. Буфер, привязанный к потоку, выделяется один раз и
// дорастает до нужного размера; при этом каждый поток остаётся со своим, то
// есть упаковка не требует никакой синхронизации.
struct PackBuffers {
  std::vector<float> a;
  std::vector<float> b;
};

PackBuffers& pack_buffers() {
  thread_local PackBuffers buffers;
  return buffers;
}

// Аргументы задачи. Их много, и передавать их по одному в функцию, которая
// вызывается из лямбды пула, значило бы переписать список трижды.
struct GemmTask {
  bool transpose_a;
  bool transpose_b;
  int64_t k;
  float alpha;
  const float* a;
  int64_t lda;
  const float* b;
  int64_t ldb;
  float beta;
  float* c;
  int64_t ldc;
  int64_t mr;
  int64_t nr;
  int64_t block_m;
  int64_t block_n;
  MicroKernelFn run;
};

// Считает прямоугольник C: строки [row0, row0 + rows), столбцы
// [col0, col0 + cols). Всё, что делает эта функция, не зависит от остальных
// прямоугольников — ни по чтению, ни по записи. Из этого следует главное:
// результат не зависит от числа потоков побитово, потому что порядок
// накопления по глубине k для каждого элемента остался тем же.
void gemm_rect(const GemmTask& task, int64_t row0, int64_t rows, int64_t col0,
               int64_t cols) {
  // beta применяется здесь же, своим прямоугольником. Отдельный проход по
  // всей C был бы вторым чтением всей матрицы и не делился бы по потокам.
  scale_c(rows, cols, task.beta, task.c + row0 * task.ldc + col0, task.ldc);

  const int64_t mr = task.mr;
  const int64_t nr = task.nr;
  PackBuffers& buffers = pack_buffers();

  for (int64_t jc = 0; jc < cols; jc += task.block_n) {
    const int64_t nc = std::min(task.block_n, cols - jc);
    for (int64_t pc = 0; pc < task.k; pc += kBlockK) {
      const int64_t kc = std::min(kBlockK, task.k - pc);

      buffers.b.resize(static_cast<std::size_t>(ceil_div(nc, nr) * nr * kc));
      pack_b(task.transpose_b, task.b, task.ldb, pc, col0 + jc, kc, nc, nr,
             buffers.b.data());

      for (int64_t ic = 0; ic < rows; ic += task.block_m) {
        const int64_t mc = std::min(task.block_m, rows - ic);

        buffers.a.resize(static_cast<std::size_t>(ceil_div(mc, mr) * mr * kc));
        pack_a(task.transpose_a, task.a, task.lda, row0 + ic, pc, mc, kc, mr,
               buffers.a.data());

        for (int64_t i = 0; i < mc; i += mr) {
          const float* apanel = buffers.a.data() + (i / mr) * kc * mr;
          const int64_t tile_rows = std::min(mr, mc - i);
          for (int64_t j = 0; j < nc; j += nr) {
            const float* bpanel = buffers.b.data() + (j / nr) * kc * nr;
            const int64_t tile_cols = std::min(nr, nc - j);
            task.run(kc, apanel, bpanel, task.alpha,
                     task.c + (row0 + ic + i) * task.ldc + (col0 + jc + j),
                     task.ldc, tile_rows, tile_cols);
          }
        }
      }
    }
  }
}

// Ниже какого объёма работы делить не стоит.
//
// Вход в параллельную область стоит около двух микросекунд — это измерено, и
// это уже после того, как ожидание работы в пуле сделали вращением вместо
// сна: со сном выходило 58 микросекунд, и порог пришлось бы поднять в тридцать
// раз. При 60 GFLOPS две микросекунды — это 120 тысяч операций, и порог взят
// того же порядка. Замер подтверждает границу: на квадрате 64 деление уже
// даёт 1.4 раза, на квадрате 32 не даёт ничего.
constexpr double kMinParallelFlops = 1.0e5;

// Как поделить прямоугольник C между потоками.
//
// Делится одна ось — та, по которой больше единиц работы, и делится по
// границам плитки микроядра: внутри плитки делить нельзя, она считается
// целиком.
//
// Почему при равенстве предпочтение строкам. Деление по строкам отдаёт каждому
// потоку целые строки C, и границы попадают между строками — ложного
// разделения строки кэша между потоками не возникает. Деление по столбцам
// разрезает каждую строку C, и на стыке двух потоков строка кэша оказывается
// общей; при нескольких блоках по глубине она перезаписывается многократно,
// и потоки начинают отнимать её друг у друга.
struct Partition {
  bool split_rows = true;
  int64_t units = 1;  // сколько единиц работы на выбранной оси
  int64_t unit_size = 1;  // размер единицы в строках или столбцах
  int tasks = 1;
};

Partition choose_partition(int64_t m, int64_t n, int64_t k, int64_t mr,
                           int64_t nr) {
  Partition out;
  out.tasks = 1;

  const int width = parallel_width();
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k);
  if (width <= 1 || inside_parallel_region() || flops < kMinParallelFlops) {
    return out;
  }

  const int64_t row_units = ceil_div(m, mr);
  const int64_t col_units = ceil_div(n, nr);
  out.split_rows = row_units >= col_units;
  out.units = out.split_rows ? row_units : col_units;
  out.unit_size = out.split_rows ? mr : nr;
  out.tasks = static_cast<int>(std::min<int64_t>(width, out.units));
  return out;
}

void check_arguments(bool transpose_a, bool transpose_b, int64_t m, int64_t n,
                     int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
  LLM_CHECK_GE(m, static_cast<int64_t>(0));
  LLM_CHECK_GE(n, static_cast<int64_t>(0));
  LLM_CHECK_GE(k, static_cast<int64_t>(0));
  // Шаг строки задан до транспонирования, поэтому нижняя граница зависит от
  // того, как матрица лежит в памяти, а не от формы op(A).
  LLM_CHECK_GE(lda, transpose_a ? m : k);
  LLM_CHECK_GE(ldb, transpose_b ? k : n);
  LLM_CHECK_GE(ldc, n);
}

}  // namespace

void gemm_naive(bool transpose_a, bool transpose_b, int64_t m, int64_t n,
                int64_t k, float alpha, const float* a, int64_t lda,
                const float* b, int64_t ldb, float beta, float* c,
                int64_t ldc) {
  check_arguments(transpose_a, transpose_b, m, n, k, lda, ldb, ldc);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int64_t p = 0; p < k; ++p) {
        const float a_value = transpose_a ? a[p * lda + i] : a[i * lda + p];
        const float b_value = transpose_b ? b[j * ldb + p] : b[p * ldb + j];
        sum += a_value * b_value;
      }
      float& target = c[i * ldc + j];
      target = alpha * sum + (beta == 0.0f ? 0.0f : beta * target);
    }
  }
}

void gemm(bool transpose_a, bool transpose_b, int64_t m, int64_t n, int64_t k,
          float alpha, const float* a, int64_t lda, const float* b, int64_t ldb,
          float beta, float* c, int64_t ldc) {
  check_arguments(transpose_a, transpose_b, m, n, k, lda, ldb, ldc);

  if (m == 0 || n == 0) {
    return;
  }
  if (k == 0 || alpha == 0.0f) {
    scale_c(m, n, beta, c, ldc);
    return;
  }

  // Микроядро выбирается по возможностям процессора — один раз, а не на
  // каждый вызов: best_micro_kernel считает выбор при первом обращении.
  const MicroKernel& kernel = best_micro_kernel();

  GemmTask task;
  task.transpose_a = transpose_a;
  task.transpose_b = transpose_b;
  task.k = k;
  task.alpha = alpha;
  task.a = a;
  task.lda = lda;
  task.b = b;
  task.ldb = ldb;
  task.beta = beta;
  task.c = c;
  task.ldc = ldc;
  task.mr = kernel.mr;
  task.nr = kernel.nr;
  task.run = kernel.run;

  // Блоки округляются вверх до кратного плитке. Иначе последняя плитка блока
  // была бы неполной всегда, а не только на краю матрицы, и медленный путь
  // записи срабатывал бы постоянно.
  task.block_m = ceil_div(kBlockM, task.mr) * task.mr;
  task.block_n = ceil_div(kBlockN, task.nr) * task.nr;

  const Partition split = choose_partition(m, n, k, task.mr, task.nr);
  if (split.tasks <= 1) {
    gemm_rect(task, 0, m, 0, n);
    return;
  }

  // Единицы работы раскладываются по задачам поровну, остаток — по одной
  // первым задачам. Считать «units / tasks» и отдать остаток последней нельзя:
  // при units = 5 и tasks = 4 последняя получила бы вдвое больше работы, и
  // остальные три ядра ждали бы её.
  const int64_t units = split.units;
  const int tasks = split.tasks;
  const int64_t limit = split.split_rows ? m : n;
  parallel_for(tasks, [&](int index) {
    const int64_t begin = (units * index) / tasks * split.unit_size;
    const int64_t end =
        std::min((units * (index + 1)) / tasks * split.unit_size, limit);
    if (begin >= end) {
      return;
    }
    if (split.split_rows) {
      gemm_rect(task, begin, end - begin, 0, n);
    } else {
      gemm_rect(task, 0, m, begin, end - begin);
    }
  });
}

}  // namespace ops
}  // namespace llm
