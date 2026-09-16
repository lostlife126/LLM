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
//
// Глубину пробовали увеличивать: 512 даёт на одном потоке до 1.42 раза на
// глубоких формах — меньше проходов по C. На четырёх потоках выигрыш исчезает
// целиком (0.99-1.06), потому что там упирается уже не в ядра, а в память, и
// экономить обращения не на чем; на шаге обучения 512 стоит семь процентов на
// tiny, 256 даёт ничью с разными знаками по пресетам. Осталось 128. Подробности
// с таблицами — в README, раздел «Глубина блока: изменение, которое не прошло».
//
// Подбирать глубину под машину нельзя: сумма по k считается кусками по kBlockK,
// внутри куска накопление идёт в регистрах, между кусками — сложением в C.
// Другая глубина — другая расстановка скобок и другое последнее округление,
// то есть результат, зависящий от машины. Это константа по построению.
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
//
// Цена упаковки не мелочь. По профилю шага обучения на неё приходилось больше
// времени, чем на само умножение: внимание после разворота голов даёт десятки
// тысяч вызовов gemm на матрицах вроде 64 x 32, а у мелкой матрицы упаковка
// сравнима с умножением. Поэтому проверка границы вынесена из внутреннего
// цикла, а там, где исходные данные лежат подряд, копирование идёт целой
// строкой.
void pack_a(bool transpose_a, const float* a, int64_t lda, int64_t row0,
            int64_t col0, int64_t mc, int64_t kc, int64_t mr,
            PackTransposeFn pack_transpose, float* apack) {
  const int64_t panels = ceil_div(mc, mr);
  for (int64_t panel = 0; panel < panels; ++panel) {
    float* dst = apack + panel * kc * mr;
    const int64_t first = panel * mr;
    const int64_t rows = std::min(mr, mc - first);

    if (transpose_a) {
      // op(A) транспонировано: подряд в памяти идут строки панели, и каждая
      // копируется как есть. Добивка нулями нужна только у последней панели
      // блока, но обходится она дёшево.
      for (int64_t p = 0; p < kc; ++p) {
        const float* src = a + (col0 + p) * lda + row0 + first;
        std::copy(src, src + rows, dst + p * mr);
        std::fill(dst + p * mr + rows, dst + p * mr + mr, 0.0f);
      }
    } else {
      // Подряд идёт шаг по глубине, а нужен шаг по строке — то самое
      // транспонирование, которое упаковка и берёт на себя. Делает его
      // микроядро: раскладка панели его, и векторный способ переставить блок
      // зависит от набора инструкций.
      pack_transpose(a + (row0 + first) * lda + col0, lda, mr, rows, kc, dst);
    }
  }
}

// Упаковка блока B из весов половинной разрядности.
//
// Только для нетранспонированного B, и это не упущение: транспонированный B
// бывает во внимании, то есть у активаций, а половинная разрядность здесь
// заведена для весов. Перестановку с одновременным развёртыванием пришлось бы
// писать отдельным векторным ядром ради случая, которого нет.
//
// Развёртывание в обычную разрядность происходит здесь, при записи в панель.
// Микроядро дальше видит обычный float и ничего не знает про половинную
// разрядность — потому блочный путь и не потребовал второго набора ядер.
void pack_b_half(const Half* b, int64_t ldb, int64_t row0, int64_t col0,
                 int64_t kc, int64_t nc, int64_t nr, float* bpack) {
  const int64_t panels = ceil_div(nc, nr);
  for (int64_t panel = 0; panel < panels; ++panel) {
    float* dst = bpack + panel * kc * nr;
    const int64_t first = panel * nr;
    const int64_t cols = std::min(nr, nc - first);
    for (int64_t p = 0; p < kc; ++p) {
      const Half* src = b + (row0 + p) * ldb + col0 + first;
      half_to_floats(src, dst + p * nr, cols);
      std::fill(dst + p * nr + cols, dst + p * nr + nr, 0.0f);
    }
  }
}

// Упаковка блока B: kc x nc матрицы op(B), панелями по nr столбцов. Всё то же
// самое, что у панели A, с обратным распределением случаев: подряд лежит
// нужное как раз у нетранспонированного B.
void pack_b(bool transpose_b, const float* b, int64_t ldb, int64_t row0,
            int64_t col0, int64_t kc, int64_t nc, int64_t nr,
            PackTransposeFn pack_transpose, float* bpack) {
  const int64_t panels = ceil_div(nc, nr);
  for (int64_t panel = 0; panel < panels; ++panel) {
    float* dst = bpack + panel * kc * nr;
    const int64_t first = panel * nr;
    const int64_t cols = std::min(nr, nc - first);

    if (transpose_b) {
      // Внимание считает Q * K^T, то есть попадает сюда на каждом слое и на
      // каждой голове. Это та же перестановка, что у панели A, и делает её та
      // же функция микроядра — разница только в ширине панели.
      pack_transpose(b + (col0 + first) * ldb + row0, ldb, nr, cols, kc, dst);
    } else {
      for (int64_t p = 0; p < kc; ++p) {
        const float* src = b + (row0 + p) * ldb + col0 + first;
        std::copy(src, src + cols, dst + p * nr);
        std::fill(dst + p * nr + cols, dst + p * nr + nr, 0.0f);
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
  // Веса половинной разрядности. Ненулевой указатель означает, что читать надо
  // отсюда, а b не задан. Два поля, а не объединение с признаком: объединение
  // пришлось бы разбирать в каждом месте, где B читается, а так проверка одна
  // и в одном месте.
  const Half* b_half;
  int64_t ldb;
  float beta;
  float* c;
  int64_t ldc;
  int64_t mr;
  int64_t nr;
  int64_t block_m;
  int64_t block_n;
  MicroKernelFn run;
  MicroKernelRowsFn run_rows;
  MicroKernelRowsHalfFn run_rows_half;
  PackTransposeFn pack_transpose;
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
      if (task.b_half != nullptr) {
        pack_b_half(task.b_half, task.ldb, pc, col0 + jc, kc, nc, nr,
                    buffers.b.data());
      } else {
        pack_b(task.transpose_b, task.b, task.ldb, pc, col0 + jc, kc, nc, nr,
               task.pack_transpose, buffers.b.data());
      }

      for (int64_t ic = 0; ic < rows; ic += task.block_m) {
        const int64_t mc = std::min(task.block_m, rows - ic);

        buffers.a.resize(static_cast<std::size_t>(ceil_div(mc, mr) * mr * kc));
        pack_a(task.transpose_a, task.a, task.lda, row0 + ic, pc, mc, kc, mr,
               task.pack_transpose, buffers.a.data());

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
// раз.
//
// Само значение выбрано замером и один раз уже переезжало. Сначала граница
// стояла на 10^5, и на квадрате 64 деление давало 1.4 раза. Потом ускорилась
// упаковка панелей, однопоточный путь на этой задаче стал быстрее вдвое, и
// выигрыш от деления исчез — остались только лишние пробуждения. Поэтому
// 10^6: квадрат 64 считается одним потоком, квадрат 128 делится и даёт 2.3
// раза.
constexpr double kMinParallelFlops = 1.0e6;

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

// --- прямой путь: умножение без упаковки ------------------------------------
//
// Упаковка существует ровно затем, чтобы микроядро читало память подряд. Когда
// вся задача помещается в кэш, это перестаёт что-либо значить: чтение «не
// подряд» из L1 стоит столько же, а упаковка остаётся чистым накладным
// расходом. На формах внимания он не мелкий — замер давал 53 GFLOPS с
// упаковкой против 96 без неё.
//
// Панель A не упаковывается никогда: микроядро прямого пути читает её
// построчно, как она лежит. Панель B не упаковывается, если B лежит обычным
// образом, — тогда её панель это сама матрица, и достаточно передать шаг
// строки вместо ширины плитки. Упаковывать приходится только транспонированный
// B, то есть Q * K^T во внимании: там подряд идут ключи, а микроядру нужны
// подряд столбцы, и без перестановки не обойтись.

// До какого размера B упаковка не окупается.
//
// Порог именно по B, и это выяснилось замером, а не выводилось заранее. Прямой
// путь перечитывает B на каждую полосу строк C, и всё держится на том,
// помещается ли B в кэш. A так не мешает: её плитка mr x k остаётся в L1 на всё
// время прохода по столбцам. C записывается ровно один раз — и это второй
// источник выигрыша, потому что блочный путь при k больше 128 делит глубину и
// перечитывает C на каждый кусок.
//
// Замер на формах модели, отношение «прямой к упакованному»:
//
//   B = 128 КБ   1.02      B = 512 КБ   0.89
//   B = 176 КБ   1.11      B = 704 КБ   0.94
//   B = 176 КБ   1.57      B = 4 МБ     0.68
//
// Граница лежит между 176 и 512 килобайтами; взято 256. Кэш второго уровня
// здесь мегабайт на ядро, то есть порог заметно ниже него — за B в кэше
// борются ещё плитки A и запись C.
constexpr double kDirectMaxBBytes = 256.0 * 1024.0;

// Сколько строк результата считать прямым путём независимо от размера B.
//
// Порог выше — про то, останется ли B в кэше. Этот — про то, окупится ли
// упаковка вообще. Блочный путь упаковывает B один раз за проход, и стоит это
// k * n работы, а самого умножения на упакованную панель приходится m * k * n.
// При m = 1 упаковка стоит столько же, сколько всё умножение, и платить за неё
// нечем. Прямой путь взамен перечитывает B по разу на каждый блок строк, то
// есть ceil(m / mr) раз, — вот где он начинает проигрывать.
//
// Замер, отношение «блочный к прямому» при k = 256, n = 4096 (B = 4 МБ):
//
//                   m=1    m=4    m=8   m=12   m=16
//   скалярное mr=4  1.37   1.42   1.09   0.98   0.92
//   AVX2      mr=6  1.85   1.84   1.15   1.14   0.92
//   AVX-512   mr=8  1.93   1.93   1.89   1.30   1.29
//
// Перелом приходится на разное m, потому что mr у ядер разный. Порог обязан
// быть один на всех — иначе выбор пути стал бы машинным свойством, а пути
// делят глубину по-разному, — поэтому взят по самому мелкому ядру: восемь.
// Почти весь выигрыш при этом остаётся, а он приходится как раз на генерацию
// по одному токену, где m равно единице.
constexpr int64_t kDirectMaxRows = 8;

// Кратность ширины результата, при которой прямой путь применим.
//
// Число подобрано так, чтобы делиться на ширину плитки любого микроядра: 32 у
// скалярного и у AVX-512, 16 у AVX2. Если появится ядро с шириной, на которую
// оно не делится, прямой путь для него просто выключится — условие ниже это
// проверяет, — но лучше подобрать ядру такую ширину, чтобы делилась.
constexpr int64_t kDirectAlign = 32;

bool use_direct(const GemmTask& task, int64_t m, int64_t n) {
  if (task.run_rows == nullptr) {
    return false;
  }
  // Транспонированное op(A) уже лежит так, как нужно упакованной панели, и
  // отдельный путь для него ничего не даст.
  if (task.transpose_a) {
    return false;
  }
  // Плитка по столбцам обязана быть полной: микроядро прямого пути читает свои
  // nr значений строки B без проверок, и неполная плитка вылезла бы за её
  // конец.
  //
  // Проверяется при этом кратность не ширине плитки, а общему числу — и это не
  // придирка. Ширина плитки у ядер разная: 32 у AVX-512, 16 у AVX2. Если
  // спрашивать про неё, то при n = 48 машина с AVX2 пошла бы прямым путём, а
  // машина с AVX-512 — блочным; пути делят глубину по-разному, и обучение на
  // двух машинах разошлось бы в последних разрядах. Общее число делится на обе
  // ширины, поэтому выбор пути одинаков везде.
  if (n % kDirectAlign != 0 || kDirectAlign % task.nr != 0) {
    return false;
  }
  // Размер считается по четыре байта на вес и тогда, когда веса лежат в
  // половинной разрядности. Физически в кэше их вдвое меньше, и порог, взятый
  // по настоящим байтам, пустил бы прямым путём вдвое более широкую матрицу.
  // Так делать нельзя, и причина та же, по которой чуть выше проверяется
  // кратность общему числу, а не ширине плитки: выбор пути не должен зависеть
  // ни от чего, кроме формы задачи.
  //
  // Прямой и блочный пути делят глубину k по-разному, то есть складывают
  // слагаемые в разном порядке. Если бы разрядность хранения меняла путь, то
  // gemm_half_b перестал бы совпадать с gemm на тех же округлённых весах — а
  // это совпадение и есть то, чем половинная разрядность здесь проверяется.
  // Сверять её стало бы не с чем.
  //
  // Цена этой строгости мала: она отсекает только матрицы в полосе от 64 до
  // 128 килобайт в половинной разрядности, а у моделей этого проекта матрицы
  // либо заметно меньше нижней границы, либо на порядок больше верхней.
  if (m <= kDirectMaxRows) {
    return true;
  }
  const double b_bytes =
      4.0 * static_cast<double>(task.k) * static_cast<double>(n);
  return b_bytes <= kDirectMaxBBytes;
}

void gemm_direct(const GemmTask& task, int64_t m, int64_t n) {
  const int64_t mr = task.mr;
  const int64_t nr = task.nr;
  const int64_t k = task.k;

  const float* bbase = task.b;
  const Half* bbase_half = task.b_half;
  int64_t bstride = task.ldb;

  // Запасной путь для процессора без развёртывания половинной разрядности
  // одной командой. Веса разворачиваются целиком в буфер упаковки, и дальше
  // всё идёт обычным ядром.
  //
  // Выигрыша здесь нет — байт читается и пишется больше, чем без половинной
  // разрядности вовсе. Смысл в другом: путь обязан остаться тем же. Если бы
  // отсутствие команды переводило вычисление на блочный путь, тот поделил бы
  // глубину иначе, и результат на такой машине отличался бы от результата на
  // всех остальных.
  std::vector<float> expanded;
  if (bbase_half != nullptr && task.run_rows_half == nullptr) {
    expanded.resize(static_cast<std::size_t>(k * n));
    for (int64_t p = 0; p < k; ++p) {
      half_to_floats(bbase_half + p * task.ldb, expanded.data() + p * n, n);
    }
    bbase = expanded.data();
    bbase_half = nullptr;
    bstride = n;
  }

  if (task.transpose_b) {
    PackBuffers& buffers = pack_buffers();
    buffers.b.resize(static_cast<std::size_t>(ceil_div(n, nr) * nr * k));
    pack_b(true, task.b, task.ldb, 0, 0, k, n, nr, task.pack_transpose,
           buffers.b.data());
    bbase = buffers.b.data();
    bstride = nr;
  }

  const auto compute = [&](int64_t first, int64_t last, int64_t col0,
                           int64_t cols) {
    scale_c(last - first, cols, task.beta,
            task.c + first * task.ldc + col0, task.ldc);
    for (int64_t i = first; i < last; i += mr) {
      const int64_t rows = std::min(mr, last - i);
      for (int64_t j = col0; j < col0 + cols; j += nr) {
        if (bbase_half != nullptr) {
          task.run_rows_half(k, task.a + i * task.lda, task.lda, bbase_half + j,
                             bstride, task.alpha, task.c + i * task.ldc + j,
                             task.ldc, rows, nr);
          continue;
        }
        const float* bpanel =
            task.transpose_b ? bbase + (j / nr) * k * nr : bbase + j;
        task.run_rows(k, task.a + i * task.lda, task.lda, bpanel, bstride,
                      task.alpha, task.c + i * task.ldc + j, task.ldc, rows,
                      nr);
      }
    }
  };

  // Деление по потокам. Границы кусков выравнены на плитку, каждый элемент C
  // считает ровно один поток, порядок накопления по глубине у него тот же —
  // значит результат от числа потоков не зависит побитово.
  //
  // Делить приходится по той оси, где есть что делить, и это не мелочь. Пока
  // прямой путь включался только при малом B, строк всегда было много, и
  // деления по строкам хватало. С порогом по числу строк всё перевернулось:
  // генерация по одному токену приходит сюда с m = 1, а единиц работы по
  // строкам тогда ровно одна — и вся работа доставалась одному ядру. Замер
  // показывал это в упор: при m <= 8 числа на одном и на четырёх потоках
  // совпадали.
  //
  // При равенстве предпочтение строкам, по той же причине, что и в блочном
  // пути: деление по столбцам разрезает строку C, и на стыке двух потоков
  // строка кэша оказывается общей.
  const int width = parallel_width();
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k);
  const int64_t row_units = ceil_div(m, mr);
  // Прямой путь включается только при n, кратном общему числу, а оно кратно
  // ширине плитки любого ядра — значит столбцы делятся нацело.
  const int64_t column_units = n / nr;
  const bool split_rows = row_units >= column_units;
  const int64_t units = split_rows ? row_units : column_units;
  const int64_t unit_size = split_rows ? mr : nr;

  const int tasks =
      (width > 1 && !inside_parallel_region() && flops >= kMinParallelFlops)
          ? static_cast<int>(std::min<int64_t>(width, units))
          : 1;
  if (tasks <= 1) {
    compute(0, m, 0, n);
    return;
  }
  parallel_for(tasks, [&](int index) {
    const int64_t begin = (units * index) / tasks * unit_size;
    const int64_t end = std::min((units * (index + 1)) / tasks * unit_size,
                                 split_rows ? m : n);
    if (begin >= end) {
      return;
    }
    if (split_rows) {
      compute(begin, end, 0, n);
    } else {
      compute(0, m, begin, end - begin);
    }
  });
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

namespace {

// Общая часть обоих входов: выбор пути и деление по потокам. Отличаются входы
// только тем, откуда берётся B, а это уже записано в задаче.
void run_gemm(GemmTask& task, int64_t m, int64_t n) {
  if (use_direct(task, m, n)) {
    gemm_direct(task, m, n);
    return;
  }

  const Partition split = choose_partition(m, n, task.k, task.mr, task.nr);
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

// Заполняет всё, что не зависит от разрядности B.
bool prepare_task(GemmTask* task, bool transpose_a, bool transpose_b, int64_t m,
                  int64_t n, int64_t k, float alpha, const float* a,
                  int64_t lda, int64_t ldb, float beta, float* c, int64_t ldc) {
  if (m == 0 || n == 0) {
    return false;
  }
  if (k == 0 || alpha == 0.0f) {
    scale_c(m, n, beta, c, ldc);
    return false;
  }

  // Микроядро выбирается по возможностям процессора — один раз, а не на
  // каждый вызов: best_micro_kernel считает выбор при первом обращении.
  const MicroKernel& kernel = best_micro_kernel();

  task->transpose_a = transpose_a;
  task->transpose_b = transpose_b;
  task->k = k;
  task->alpha = alpha;
  task->a = a;
  task->lda = lda;
  task->b = nullptr;
  task->b_half = nullptr;
  task->ldb = ldb;
  task->beta = beta;
  task->c = c;
  task->ldc = ldc;
  task->mr = kernel.mr;
  task->nr = kernel.nr;
  task->run = kernel.run;
  task->run_rows = kernel.run_rows;
  task->run_rows_half = kernel.run_rows_half;
  task->pack_transpose = kernel.pack_transpose;

  // Блоки округляются вверх до кратного плитке. Иначе последняя плитка блока
  // была бы неполной всегда, а не только на краю матрицы, и медленный путь
  // записи срабатывал бы постоянно.
  task->block_m = ceil_div(kBlockM, task->mr) * task->mr;
  task->block_n = ceil_div(kBlockN, task->nr) * task->nr;
  return true;
}

}  // namespace

void gemm(bool transpose_a, bool transpose_b, int64_t m, int64_t n, int64_t k,
          float alpha, const float* a, int64_t lda, const float* b, int64_t ldb,
          float beta, float* c, int64_t ldc) {
  check_arguments(transpose_a, transpose_b, m, n, k, lda, ldb, ldc);
  GemmTask task;
  if (!prepare_task(&task, transpose_a, transpose_b, m, n, k, alpha, a, lda,
                    ldb, beta, c, ldc)) {
    return;
  }
  task.b = b;
  run_gemm(task, m, n);
}

void gemm_half_b(bool transpose_a, int64_t m, int64_t n, int64_t k, float alpha,
                 const float* a, int64_t lda, const Half* b, int64_t ldb,
                 float beta, float* c, int64_t ldc) {
  check_arguments(transpose_a, false, m, n, k, lda, ldb, ldc);
  GemmTask task;
  if (!prepare_task(&task, transpose_a, false, m, n, k, alpha, a, lda, ldb,
                    beta, c, ldc)) {
    return;
  }
  task.b_half = b;
  run_gemm(task, m, n);
}

}  // namespace ops
}  // namespace llm
