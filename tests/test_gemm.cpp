#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/random.h"
#include "core/thread_pool.h"
#include "ops/gemm.h"
#include "ops/micro_kernel.h"
#include "testing.h"

namespace {

// Тот же сторож, что и у ширины параллелизма, и по той же причине — см.
// llm::testing::WidthGuard в testing.h.
using llm::testing::WidthGuard;

// Возвращает автоматический выбор ядра при любом выходе из области
// видимости — в том числе через исключение.
//
// Без этого непрошедшая проверка внутри перебора ядер оставляла принудительный
// выбор включённым, и ВСЕ последующие тесты двоичного файла считались чужим
// ядром. Поймано при проверке нового теста подменой: одна поломка в скалярном
// ядре уронила двадцать тестов, из которых к ней относились два.
class ForcedKernel {
 public:
  explicit ForcedKernel(const llm::ops::MicroKernel& kernel) {
    llm::ops::force_micro_kernel(&kernel);
  }
  ~ForcedKernel() { llm::ops::force_micro_kernel(nullptr); }

  ForcedKernel(const ForcedKernel&) = delete;
  ForcedKernel& operator=(const ForcedKernel&) = delete;
};

std::vector<float> random_matrix(llm::Rng* rng, std::int64_t rows,
                                 std::int64_t ld) {
  std::vector<float> values(static_cast<std::size_t>(rows * ld));
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = rng->uniform(-1.0f, 1.0f);
  }
  return values;
}

// Максимальное расхождение двух матриц, нормированное на масштаб значений.
// Блочная и наивная версии складывают слагаемые в разном порядке, поэтому
// побитового совпадения не бывает — сравнивать нужно с допуском.
double max_relative_error(const std::vector<float>& reference,
                          const std::vector<float>& actual, std::int64_t rows,
                          std::int64_t cols, std::int64_t ld) {
  double worst = 0.0;
  double scale = 1.0;
  for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t j = 0; j < cols; ++j) {
      const std::size_t index = static_cast<std::size_t>(i * ld + j);
      scale = std::max(scale, std::fabs(static_cast<double>(reference[index])));
    }
  }
  for (std::int64_t i = 0; i < rows; ++i) {
    for (std::int64_t j = 0; j < cols; ++j) {
      const std::size_t index = static_cast<std::size_t>(i * ld + j);
      const double diff = std::fabs(static_cast<double>(reference[index]) -
                                    static_cast<double>(actual[index]));
      worst = std::max(worst, diff / scale);
    }
  }
  return worst;
}

// Сверяет блочный gemm с наивным на одной задаче. pad добавляется к шагу
// строки, чтобы проверить, что реализация уважает leading dimension и не
// считает матрицу плотной.
void compare_with_naive(llm::Rng* rng, bool transpose_a, bool transpose_b,
                        std::int64_t m, std::int64_t n, std::int64_t k,
                        float alpha, float beta, std::int64_t pad) {
  const std::int64_t rows_a = transpose_a ? k : m;
  const std::int64_t cols_a = transpose_a ? m : k;
  const std::int64_t rows_b = transpose_b ? n : k;
  const std::int64_t cols_b = transpose_b ? k : n;
  const std::int64_t lda = cols_a + pad;
  const std::int64_t ldb = cols_b + pad;
  const std::int64_t ldc = n + pad;

  const std::vector<float> a = random_matrix(rng, rows_a, lda);
  const std::vector<float> b = random_matrix(rng, rows_b, ldb);
  const std::vector<float> c_initial = random_matrix(rng, m, ldc);

  std::vector<float> c_reference = c_initial;
  std::vector<float> c_actual = c_initial;

  llm::ops::gemm_naive(transpose_a, transpose_b, m, n, k, alpha, a.data(), lda,
                       b.data(), ldb, beta, c_reference.data(), ldc);
  llm::ops::gemm(transpose_a, transpose_b, m, n, k, alpha, a.data(), lda,
                 b.data(), ldb, beta, c_actual.data(), ldc);

  const double error = max_relative_error(c_reference, c_actual, m, n, ldc);
  LLM_CHECK_MSG(error < 1e-5,
                "m=" << m << " n=" << n << " k=" << k << " tA=" << transpose_a
                     << " tB=" << transpose_b << " alpha=" << alpha
                     << " beta=" << beta << ": расхождение " << error);

  // Ни одна реализация не имеет права писать в добивку за пределами n
  // столбцов: там могут лежать чужие данные.
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = n; j < ldc; ++j) {
      const std::size_t index = static_cast<std::size_t>(i * ldc + j);
      LLM_CHECK_MSG(
          c_actual[index] == c_initial[index],
          "gemm записал за границу C в строке " << i << ", столбце " << j);
    }
  }
}

}  // namespace

LLM_TEST(Gemm, MatchesNaiveOnSmallSizes) {
  llm::Rng rng(1234);
  // Размеры вокруг границ плитки микроядра (4 x 16): единица, ровное деление,
  // остаток с обеих сторон.
  const std::int64_t sizes[] = {1, 2, 3, 4, 5, 7, 15, 16, 17};
  for (std::size_t mi = 0; mi < sizeof(sizes) / sizeof(sizes[0]); ++mi) {
    for (std::size_t ni = 0; ni < sizeof(sizes) / sizeof(sizes[0]); ++ni) {
      for (std::size_t ki = 0; ki < sizeof(sizes) / sizeof(sizes[0]); ++ki) {
        compare_with_naive(&rng, false, false, sizes[mi], sizes[ni], sizes[ki],
                           1.0f, 0.0f, 0);
      }
    }
  }
}

LLM_TEST(Gemm, MatchesNaiveWithTranspose) {
  llm::Rng rng(99);
  const std::int64_t sizes[] = {1, 3, 16, 17, 33};
  for (int mode = 0; mode < 4; ++mode) {
    const bool transpose_a = (mode & 1) != 0;
    const bool transpose_b = (mode & 2) != 0;
    for (std::size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
      for (std::size_t sj = 0; sj < sizeof(sizes) / sizeof(sizes[0]); ++sj) {
        compare_with_naive(&rng, transpose_a, transpose_b, sizes[si], sizes[sj],
                           sizes[(si + sj) % 5], 1.0f, 0.0f, 0);
      }
    }
  }
}

LLM_TEST(Gemm, RespectsLeadingDimension) {
  llm::Rng rng(7);
  for (int mode = 0; mode < 4; ++mode) {
    compare_with_naive(&rng, (mode & 1) != 0, (mode & 2) != 0, 20, 20, 20, 1.0f,
                       0.0f, 3);
    compare_with_naive(&rng, (mode & 1) != 0, (mode & 2) != 0, 7, 33, 5, 1.0f,
                       0.0f, 11);
  }
}

LLM_TEST(Gemm, AlphaAndBeta) {
  llm::Rng rng(555);
  const float alphas[] = {1.0f, 0.0f, -2.5f, 0.25f};
  const float betas[] = {0.0f, 1.0f, -1.0f, 0.5f};
  for (std::size_t ai = 0; ai < 4; ++ai) {
    for (std::size_t bi = 0; bi < 4; ++bi) {
      compare_with_naive(&rng, false, false, 18, 20, 9, alphas[ai], betas[bi],
                         2);
    }
  }
}

LLM_TEST(Gemm, BetaZeroOverwritesGarbage) {
  // beta == 0 обязана перезаписывать C, а не умножать на нуль: иначе NaN в
  // непроинициализированном буфере пережил бы умножение.
  const std::vector<float> a = {1.0f, 2.0f};
  const std::vector<float> b = {3.0f, 4.0f};
  std::vector<float> c = {std::nanf(""), std::nanf("")};
  llm::ops::gemm(false, false, 2, 1, 1, 1.0f, a.data(), 1, b.data(), 1, 0.0f,
                 c.data(), 1);
  LLM_EXPECT_NEAR(c[0], 3.0, 0.0);
  LLM_EXPECT_NEAR(c[1], 6.0, 0.0);
}

LLM_TEST(Gemm, CrossesBlockBoundaries) {
  llm::Rng rng(2024);
  // Блоки — 64 x 256 x 256. Берём размеры, заведомо перекрывающие несколько
  // блоков по каждой оси, с остатками.
  compare_with_naive(&rng, false, false, 130, 70, 260, 1.0f, 0.0f, 0);
  compare_with_naive(&rng, true, false, 65, 300, 257, 1.0f, 0.5f, 0);
  compare_with_naive(&rng, false, true, 64, 256, 256, 1.0f, 0.0f, 0);
}

LLM_TEST(Gemm, KnownProduct) {
  // Один пример, посчитанный руками: защита от согласованной ошибки сразу в
  // обеих реализациях.
  //   A = [[1, 2, 3],       B = [[1, 2],        A*B = [[22, 28],
  //        [4, 5, 6]]            [3, 4],               [49, 64]]
  //                              [5, 6]]
  const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> c(4, 0.0f);

  llm::ops::gemm(false, false, 2, 2, 3, 1.0f, a.data(), 3, b.data(), 2, 0.0f,
                 c.data(), 2);
  LLM_EXPECT_NEAR(c[0], 22.0, 1e-5);
  LLM_EXPECT_NEAR(c[1], 28.0, 1e-5);
  LLM_EXPECT_NEAR(c[2], 49.0, 1e-5);
  LLM_EXPECT_NEAR(c[3], 64.0, 1e-5);

  std::vector<float> c_naive(4, 0.0f);
  llm::ops::gemm_naive(false, false, 2, 2, 3, 1.0f, a.data(), 3, b.data(), 2,
                       0.0f, c_naive.data(), 2);
  for (std::size_t i = 0; i < 4; ++i) {
    LLM_EXPECT_NEAR(c_naive[i], c[i], 1e-5);
  }
}

LLM_TEST(Gemm, ZeroSizedProblem) {
  const std::vector<float> a = {1.0f};
  const std::vector<float> b = {1.0f};
  std::vector<float> c = {5.0f};
  // k == 0: произведение пустое, поэтому остаётся только beta * C.
  llm::ops::gemm(false, false, 1, 1, 0, 1.0f, a.data(), 1, b.data(), 1, 2.0f,
                 c.data(), 1);
  LLM_EXPECT_NEAR(c[0], 10.0, 0.0);
  // m == 0 или n == 0: писать некуда, падать не должно.
  llm::ops::gemm(false, false, 0, 1, 1, 1.0f, a.data(), 1, b.data(), 1, 1.0f,
                 c.data(), 1);
  llm::ops::gemm(false, false, 1, 0, 1, 1.0f, a.data(), 1, b.data(), 1, 1.0f,
                 c.data(), 1);
  LLM_EXPECT_NEAR(c[0], 10.0, 0.0);
}

LLM_TEST(Gemm, RejectsBadLeadingDimension) {
  const std::vector<float> a(16, 0.0f);
  std::vector<float> c(16, 0.0f);
  // ldb меньше числа столбцов B — строки перекрывались бы.
  LLM_EXPECT_THROWS(llm::ops::gemm(false, false, 4, 4, 4, 1.0f, a.data(), 4,
                                   a.data(), 2, 0.0f, c.data(), 4));
  LLM_EXPECT_THROWS(llm::ops::gemm(false, false, 4, 4, 4, 1.0f, a.data(), 4,
                                   a.data(), 4, 0.0f, c.data(), 2));
}

LLM_TEST(Gemm, ThreadCountDoesNotChangeResultBitForBit) {
  // Деление по потокам сделано так, что каждый элемент C считает ровно один
  // поток, и порядок накопления по глубине k у него тот же, что был бы у
  // одного потока. Отсюда — совпадение побитовое, а не с допуском.
  //
  // Требование строгое сознательно. Совпадение с допуском проверяло бы только
  // отсутствие грубой ошибки, а нам нужно большее: повторяемость обучения не
  // должна зависеть ни от числа ядер на машине, ни от того, как в этот раз
  // разошлись потоки. Иначе два запуска с одним зерном дают разные модели, и
  // сравнивать варианты становится нечем.
  llm::Rng rng(20260915);
  const std::int64_t sizes[][3] = {
      {256, 192, 160}, {512, 64, 300}, {64, 512, 300}, {129, 127, 131}};

  for (std::size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
    const std::int64_t m = sizes[i][0];
    const std::int64_t n = sizes[i][1];
    const std::int64_t k = sizes[i][2];
    const std::vector<float> a = random_matrix(&rng, m, k);
    const std::vector<float> b = random_matrix(&rng, k, n);

    std::vector<float> serial(static_cast<std::size_t>(m * n), 0.0f);
    {
      const WidthGuard guard(1);
      llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n,
                     0.0f, serial.data(), n);
    }

    const int width = llm::parallel_width();
    std::vector<float> threaded(static_cast<std::size_t>(m * n), 0.0f);
    llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                   threaded.data(), n);

    for (std::size_t index = 0; index < serial.size(); ++index) {
      LLM_CHECK_MSG(serial[index] == threaded[index],
                    "m=" << m << " n=" << n << " k=" << k << " потоков "
                         << width << ": элемент " << index << " дал "
                         << threaded[index] << " вместо " << serial[index]);
    }
  }
}

LLM_TEST(Gemm, ThreadedResultMatchesNaive) {
  // Отдельно от побитовой проверки: та сравнивает многопоточный путь с
  // однопоточным, и если оба ошибаются одинаково, она этого не заметит.
  const WidthGuard guard(0);
  llm::Rng rng(7);
  // Формы выбраны так, чтобы задеть оба способа деления: по строкам, когда их
  // больше, и по столбцам, когда больше столбцов.
  compare_with_naive(&rng, false, false, 600, 96, 200, 1.0f, 0.0f, 0);
  compare_with_naive(&rng, false, false, 96, 600, 200, 1.0f, 0.0f, 3);
  compare_with_naive(&rng, true, false, 300, 300, 300, 0.5f, 2.0f, 0);
  compare_with_naive(&rng, false, true, 300, 300, 300, 1.0f, 1.0f, 5);
}

LLM_TEST(Gemm, VectorKernelsAgreeBitForBit) {
  // Микроядра разной ширины считают одну и ту же плитку, но раскладывают её по
  // регистрам по-разному. Порядок накопления по глубине k при этом у всех один
  // и тот же — последовательный, — поэтому векторные ядра обязаны совпадать
  // побитово, а не с допуском.
  //
  // Проверка важна не сама по себе. Из неё следует, что обучение не зависит от
  // того, какой процессор достался: на машине без AVX-512 возьмётся AVX2, и
  // получится тот же файл весов. Сравнивать варианты архитектуры, посчитанные
  // на разных машинах, иначе было бы нельзя.
  //
  // Скалярное ядро в это равенство не входит, и это не упущение. Оно
  // собирается под базовый x86-64, где инструкции умножения с накоплением
  // нет, и делает умножение и сложение отдельно — с двумя округлениями вместо
  // одного. Разница мала, но она есть, и требовать здесь побитового
  // совпадения значило бы требовать невозможного. Проверяется поэтому
  // величина расхождения.
  int count = 0;
  const llm::ops::MicroKernelChoice* table =
      llm::ops::all_micro_kernels(&count);
  LLM_CHECK_GT(count, 0);

  llm::Rng rng(31337);
  const std::int64_t m = 200;
  const std::int64_t n = 150;
  const std::int64_t k = 300;
  const std::vector<float> a = random_matrix(&rng, m, k);
  const std::vector<float> b = random_matrix(&rng, k, n);

  std::vector<float> fused_reference;
  const char* fused_name = "";
  std::vector<float> plain_reference;
  int fused_compared = 0;

  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      continue;
    }
    const ForcedKernel forced(table[i].kernel);
    std::vector<float> actual(static_cast<std::size_t>(m * n), 0.0f);
    llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                   actual.data(), n);

    if (!table[i].kernel.fused) {
      plain_reference = actual;
      continue;
    }
    if (fused_reference.empty()) {
      fused_reference = actual;
      fused_name = table[i].kernel.name;
      continue;
    }
    ++fused_compared;
    for (std::size_t index = 0; index < fused_reference.size(); ++index) {
      LLM_CHECK_MSG(fused_reference[index] == actual[index],
                    "ядра '" << fused_name << "' и '" << table[i].kernel.name
                             << "' расходятся в " << index << ": "
                             << fused_reference[index] << " против "
                             << actual[index]);
    }
  }

  // Если векторных ядер на этой машине меньше двух, сравнивать нечего. Молча
  // проходить в таком случае нельзя: пусть будет видно, что проверка не
  // работала.
  LLM_CHECK_MSG(fused_compared > 0 || count < 3,
                "сравнить оказалось нечего: доступно ядер " << count);

  // Скалярное ядро против векторного: расхождение от лишнего округления
  // накапливается по k случайными знаками, то есть как корень из k, а не
  // линейно. При k = 300 это должно оставаться на уровне единиц ulp.
  if (!fused_reference.empty() && !plain_reference.empty()) {
    const double error =
        max_relative_error(plain_reference, fused_reference, m, n, n);
    LLM_CHECK_MSG(error < 1e-6, "скалярное ядро расходится с векторным на "
                                    << error
                                    << " — это слишком много для "
                                       "разницы в одном округлении");
  }
}

LLM_TEST(Gemm, DirectPathMatchesNaive) {
  // Прямой путь — тот, где упаковки нет вовсе, — выбирается по размеру B, и
  // обычные тесты в него попадают не всегда. Здесь формы подобраны так, чтобы
  // попадали наверняка: ширина результата кратна плитке любого из ядер (32 и
  // 16), а B заведомо мал.
  //
  // Проверяются ровно те места, где прямой путь отличается от блочного:
  //   - неполная полоса строк: микроядро берёт недостающие строки с последней
  //     действительной и не должно записать их в C;
  //   - шаги строк больше ширины (pad), чтобы панель B бралась по ldb, а не по
  //     ширине матрицы;
  //   - транспонированный B, где упаковка остаётся и должна сойтись с прямым
  //     чтением A;
  //   - alpha и beta, потому что beta применяется отдельным проходом по своей
  //     полосе строк.
  int count = 0;
  const llm::ops::MicroKernelChoice* table =
      llm::ops::all_micro_kernels(&count);

  llm::Rng rng(515);
  const std::int64_t widths[] = {32, 64, 128};
  const std::int64_t heights[] = {1, 7, 8, 13, 64, 129};

  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      continue;
    }
    const ForcedKernel forced(table[i].kernel);
    for (std::size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w) {
      for (std::size_t h = 0; h < sizeof(heights) / sizeof(heights[0]); ++h) {
        const std::int64_t n = widths[w];
        const std::int64_t m = heights[h];
        compare_with_naive(&rng, false, false, m, n, 40, 1.0f, 0.0f, 0);
        compare_with_naive(&rng, false, false, m, n, 33, 0.5f, 2.0f, 3);
        compare_with_naive(&rng, false, true, m, n, 40, 1.0f, 1.0f, 0);
        compare_with_naive(&rng, false, true, m, n, 17, -1.5f, 0.0f, 5);
      }
    }
  }
}

LLM_TEST(Gemm, DirectPathDoesNotDependOnThreadCount) {
  // То же требование, что и к блочному пути: число ядер не меняет результат
  // побитово. У прямого пути деление своё, поэтому проверяется отдельно.
  llm::Rng rng(90210);
  const std::int64_t m = 300;
  const std::int64_t n = 128;
  const std::int64_t k = 48;
  const std::vector<float> a = random_matrix(&rng, m, k);
  const std::vector<float> b = random_matrix(&rng, k, n);

  std::vector<float> serial(static_cast<std::size_t>(m * n), 0.0f);
  {
    const WidthGuard guard(1);
    llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                   serial.data(), n);
  }

  std::vector<float> threaded(static_cast<std::size_t>(m * n), 0.0f);
  llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                 threaded.data(), n);

  for (std::size_t index = 0; index < serial.size(); ++index) {
    LLM_CHECK_MSG(serial[index] == threaded[index],
                  "элемент " << index << ": " << threaded[index] << " вместо "
                             << serial[index]);
  }
}

// Тот же вопрос для деления по столбцам.
//
// Отдельный тест, потому что ветка другая, и добраться до неё формой задачи
// можно только нарочно: строк должно быть мало, а столбцов много. Это ровно
// форма генерации по одному токену — и ровно та, где деление по столбцам
// появилось, когда выяснилось, что при m = 1 работа доставалась одному ядру.
LLM_TEST(Gemm, DirectPathSplitByColumnsDoesNotDependOnThreadCount) {
  llm::Rng rng(31337);
  const std::int64_t m = 1;
  const std::int64_t n = 4096;  // ниже этого объёма деление не включается
  const std::int64_t k = 256;
  const std::vector<float> a = random_matrix(&rng, m, k);
  const std::vector<float> b = random_matrix(&rng, k, n);

  // beta здесь не единица нарочно. Каждый поток применяет beta к своей полосе
  // столбцов, и если бы полосы пересеклись или какая-то осталась без beta,
  // видно было бы именно по этому.
  std::vector<float> serial(static_cast<std::size_t>(m * n), 0.25f);
  {
    const WidthGuard guard(1);
    llm::ops::gemm(false, false, m, n, k, 1.5f, a.data(), k, b.data(), n, 0.5f,
                   serial.data(), n);
  }

  std::vector<float> threaded(static_cast<std::size_t>(m * n), 0.25f);
  llm::ops::gemm(false, false, m, n, k, 1.5f, a.data(), k, b.data(), n, 0.5f,
                 threaded.data(), n);

  for (std::size_t index = 0; index < serial.size(); ++index) {
    LLM_CHECK_MSG(serial[index] == threaded[index],
                  "элемент " << index << ": " << threaded[index] << " вместо "
                             << serial[index]);
  }
}

// То же для весов половинной разрядности: деление по столбцам у них общее с
// обычным путём, но ядро другое, и границы полос оно считает само.
LLM_TEST(Gemm, HalfSplitByColumnsDoesNotDependOnThreadCount) {
  llm::Rng rng(4711);
  const std::int64_t m = 1;
  const std::int64_t n = 4096;  // ниже этого объёма деление не включается
  const std::int64_t k = 256;
  const std::vector<float> a = random_matrix(&rng, m, k);
  std::vector<float> b = random_matrix(&rng, k, n);
  std::vector<llm::Half> packed(b.size());
  llm::floats_to_half(b.data(), packed.data(),
                      static_cast<std::int64_t>(b.size()));

  std::vector<float> serial(static_cast<std::size_t>(m * n), 0.25f);
  {
    const WidthGuard guard(1);
    llm::ops::gemm_half_b(false, m, n, k, 1.5f, a.data(), k, packed.data(), n,
                          0.5f, serial.data(), n);
  }

  std::vector<float> threaded(static_cast<std::size_t>(m * n), 0.25f);
  llm::ops::gemm_half_b(false, m, n, k, 1.5f, a.data(), k, packed.data(), n,
                        0.5f, threaded.data(), n);

  for (std::size_t index = 0; index < serial.size(); ++index) {
    LLM_CHECK_MSG(serial[index] == threaded[index],
                  "элемент " << index << ": " << threaded[index] << " вместо "
                             << serial[index]);
  }
}

LLM_TEST(Gemm, PathChoiceDoesNotDependOnTheKernel) {
  // Прямой и блочный пути делят глубину по-разному, поэтому дают разные
  // младшие разряды. Это допустимо — но только если выбор пути одинаков на
  // любой машине. Иначе обучение на процессоре с AVX2 и на процессоре с
  // AVX-512 разошлось бы, и сравнивать их было бы нельзя.
  //
  // Сравнение побитовое, и допуска здесь быть не может. Первая версия теста
  // требовала расхождения меньше 1e-6, считая, что разные пути дадут заметно
  // больше. Замер показал обратное: на формах, где путь действительно
  // переключается (m=96, k=300, n от 256 до 1024), пути расходятся на
  // 6e-7…8e-7 — то есть ПОД допуском. Тест не мог поймать ровно то, ради чего
  // был написан.
  //
  // Сравниваются только слитные ядра. Скалярное делает умножение и сложение
  // отдельно, с двумя округлениями вместо одного, и с векторными не совпадает
  // независимо от путей — см. VectorKernelsAgreeBitForBit. Требовать от него
  // побитового равенства значило бы требовать невозможного, а допуск вернул бы
  // ту же слепоту.
  //
  // Ширины подобраны по условию выбора пути. Прямой путь требует кратности n
  // общему числу 32 и укладывания B в кэш; проверяется и то, и другое:
  //   48   не кратно 32 — блочный путь у всех;
  //   64   кратно, B = 75 КБ — прямой;
  //   128  кратно, B = 150 КБ — прямой;
  //   256  кратно, B = 300 КБ — блочный, потому что B уже не влезает.
  // Ширина 48 здесь главная: 48 делится на 16, но не на 32, и если бы условие
  // спрашивало про ширину плитки ядра, а не про общее число, машина с AVX2
  // пошла бы прямым путём, а машина с AVX-512 — блочным.
  llm::Rng rng(1234);
  const std::int64_t widths[] = {48, 64, 128, 256};
  const std::int64_t m = 96;
  const std::int64_t k = 300;

  for (std::size_t w = 0; w < sizeof(widths) / sizeof(widths[0]); ++w) {
    const std::int64_t n = widths[w];
    const std::vector<float> a = random_matrix(&rng, m, k);
    const std::vector<float> b = random_matrix(&rng, k, n);

    std::vector<float> reference;
    const char* reference_name = "";
    int count = 0;
    const llm::ops::MicroKernelChoice* table =
        llm::ops::all_micro_kernels(&count);
    for (int i = 0; i < count; ++i) {
      if (!table[i].available || !table[i].kernel.fused) {
        continue;
      }
      const ForcedKernel forced(table[i].kernel);
      std::vector<float> actual(static_cast<std::size_t>(m * n), 0.0f);
      llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n,
                     0.0f, actual.data(), n);
      if (reference.empty()) {
        reference = actual;
        reference_name = table[i].kernel.name;
        continue;
      }
      for (std::size_t j = 0; j < reference.size(); ++j) {
        LLM_CHECK_MSG(reference[j] == actual[j],
                      "ядра '" << reference_name << "' и '"
                               << table[i].kernel.name << "' при n = " << n
                               << " разошлись в элементе " << j << ": "
                               << actual[j] << " вместо " << reference[j]
                               << " — похоже, они выбрали разные пути");
      }
    }
  }
}

// --- веса половинной разрядности ---------------------------------------------
//
// Обещание gemm_half_b одно, и оно сильное: результат обязан совпасть ПОБИТОВО
// с обычным gemm, которому дали те же веса, уже прошедшие округление. Не «в
// пределах погрешности» — побитово. Из этого следует, что половинная
// разрядность не может испортить ничего, кроме самих весов, и что машины с
// разными наборами инструкций по-прежнему считают одинаково.
//
// Если бы проверка была с допуском, она пропустила бы ровно те ошибки, ради
// которых написана: перепутанный порядок накопления, потерянный краевой
// столбец, другой путь внутри gemm.

namespace {

struct HalfCase {
  std::int64_t m;
  std::int64_t n;
  std::int64_t k;
  const char* what;
};

void check_half_matches_rounded(const HalfCase& shape) {
  llm::Rng rng(20240816 + shape.n);
  std::vector<float> a = random_matrix(&rng, shape.m, shape.k);
  std::vector<float> b = random_matrix(&rng, shape.k, shape.n);

  // Веса округляются ОБЕИМ сторонам. Иначе проверка смешала бы два вопроса —
  // правильность ядра и точность половинной разрядности — и не ответила бы ни
  // на один.
  std::vector<llm::Half> packed(b.size());
  llm::floats_to_half(b.data(), packed.data(),
                      static_cast<std::int64_t>(b.size()));
  llm::half_to_floats(packed.data(), b.data(),
                      static_cast<std::int64_t>(b.size()));

  std::vector<float> reference(static_cast<std::size_t>(shape.m * shape.n),
                               0.25f);
  std::vector<float> actual = reference;

  llm::ops::gemm(false, false, shape.m, shape.n, shape.k, 1.5f, a.data(),
                 shape.k, b.data(), shape.n, 0.5f, reference.data(), shape.n);
  llm::ops::gemm_half_b(false, shape.m, shape.n, shape.k, 1.5f, a.data(),
                        shape.k, packed.data(), shape.n, 0.5f, actual.data(),
                        shape.n);

  for (std::size_t i = 0; i < reference.size(); ++i) {
    LLM_CHECK_MSG(reference[i] == actual[i],
                  shape.what << ": элемент " << i << " даёт " << actual[i]
                             << " вместо " << reference[i]);
  }
}

// Формы подобраны по порогу прямого пути: он включается, когда B укладывается в
// 256 килобайт. Средняя строка — та самая полоса, где порог, посчитанный по
// настоящим байтам половинной разрядности, пустил бы gemm_half_b прямым путём,
// а обычный gemm остался бы на блочном. Первая написанная здесь версия так и
// делала, и эта строка её уронила: пути делят глубину по-разному. Поэтому
// размер считается по четыре байта на вес независимо от хранения, а строка
// осталась сторожить это решение.
const HalfCase kHalfCases[] = {
    {1, 256, 256, "генерация, прямой путь"},
    {1, 384, 256, "генерация, полоса у порога прямого пути"},
    {1, 1024, 256, "генерация, блочный путь"},
    {7, 256, 256, "неполная плитка строк, прямой путь"},
    {7, 1024, 256, "неполная плитка строк, блочный путь"},
    {64, 256, 704, "батч, глубина больше куска"},
    {64, 1024, 256, "батч, блочный путь"},
    {33, 320, 97, "нечётные размеры, край плитки"},
};

}  // namespace

LLM_TEST(Gemm, HalfWeightsMatchRoundedExactly) {
  for (std::size_t i = 0; i < sizeof(kHalfCases) / sizeof(kHalfCases[0]); ++i) {
    check_half_matches_rounded(kHalfCases[i]);
  }
}

// То же самое на каждом собранном микроядре. Ядро прямого пути у каждого своё,
// и ошибиться в нём можно независимо от остальных.
LLM_TEST(Gemm, HalfWeightsAgreeAcrossKernels) {
  int count = 0;
  const llm::ops::MicroKernelChoice* table =
      llm::ops::all_micro_kernels(&count);
  int checked = 0;
  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      continue;
    }
    const ForcedKernel forced(table[i].kernel);
    for (std::size_t j = 0; j < sizeof(kHalfCases) / sizeof(kHalfCases[0]);
         ++j) {
      check_half_matches_rounded(kHalfCases[j]);
    }
    ++checked;
  }
  LLM_CHECK_MSG(checked > 0, "не проверено ни одного ядра");
}

// Запасной путь: процессор умеет считать, но не умеет разворачивать половинную
// разрядность одной командой. Тогда веса разворачиваются целиком заранее — и
// вот что здесь проверяется: результат обязан остаться тем же побитово, потому
// что путь внутри gemm не изменился. Если бы запасной вариант уходил на
// блочный путь, тот поделил бы глубину иначе, и такая машина считала бы не как
// все остальные.
LLM_TEST(Gemm, HalfFallbackKeepsTheSamePath) {
  int count = 0;
  const llm::ops::MicroKernelChoice* table =
      llm::ops::all_micro_kernels(&count);
  int checked = 0;
  for (int i = 0; i < count; ++i) {
    if (!table[i].available || table[i].kernel.run_rows_half == nullptr) {
      continue;
    }
    llm::ops::MicroKernel crippled = table[i].kernel;
    crippled.run_rows_half = nullptr;
    const ForcedKernel forced(crippled);
    for (std::size_t j = 0; j < sizeof(kHalfCases) / sizeof(kHalfCases[0]);
         ++j) {
      check_half_matches_rounded(kHalfCases[j]);
    }
    ++checked;
  }
  LLM_CHECK_MSG(checked > 0, "не проверено ни одного ядра");
}

// --- транспонирующая упаковка ------------------------------------------------
//
// Единственная неарифметическая примитивная операция в умножении: она только
// переставляет значения, поэтому «то же самое» здесь означает буквально то же
// самое, без всякого допуска.
//
// До сих пор она проверялась лишь косвенно — через сравнение ядер между собой,
// потому что упаковка панели A входит в любое обычное умножение. Проверка
// работала, но не пиняла контракт: раскладку панели и добивку нулями там, где
// действительных строк меньше, чем дорожек. Маскированный хвост векторного
// варианта вдобавок срабатывает только у AVX2 — у него плитка шириной шесть, а
// у AVX-512 тридцать два, то есть кратна восьми и хвоста не даёт.
LLM_TEST(Gemm, TransposePackMatchesTheScalarReference) {
  int count = 0;
  const llm::ops::MicroKernelChoice* table =
      llm::ops::all_micro_kernels(&count);
  LLM_CHECK_MSG(count > 0, "таблица ядер пуста");

  // Скалярное ядро пользуется простым вариантом упаковки, он и эталон.
  llm::ops::PackTransposeFn reference = nullptr;
  for (int i = 0; i < count; ++i) {
    if (std::string(table[i].kernel.name).find("скалярное") == 0) {
      reference = table[i].kernel.pack_transpose;
    }
  }
  LLM_CHECK_MSG(reference != nullptr, "скалярное ядро не найдено");

  llm::Rng rng(515);
  int checked = 0;
  // Ширины дорожек — все, какие встречаются у ядер проекта, плюс не кратная
  // восьми: именно на ней работает маскированный хвост.
  const std::int64_t lanes_list[] = {4, 6, 8, 12, 16, 32};
  const std::int64_t depths[] = {1, 3, 8, 9, 16, 33};

  for (std::size_t l = 0; l < sizeof(lanes_list) / sizeof(lanes_list[0]); ++l) {
    const std::int64_t lanes = lanes_list[l];
    for (std::size_t d = 0; d < sizeof(depths) / sizeof(depths[0]); ++d) {
      const std::int64_t kc = depths[d];
      // Действительных строк может быть меньше дорожек — это край матрицы.
      for (std::int64_t rows = 1; rows <= lanes; ++rows) {
        // Шаг строки нарочно больше глубины: панель берётся из середины
        // матрицы, и упаковка не вправе читать соседние столбцы.
        const std::int64_t lda = kc + 5;
        const std::vector<float> source = random_matrix(&rng, lanes, lda);

        std::vector<float> expected(static_cast<std::size_t>(kc * lanes),
                                    -7.0f);
        reference(source.data(), lda, lanes, rows, kc, expected.data());

        for (int i = 0; i < count; ++i) {
          if (!table[i].available) {
            continue;
          }
          std::vector<float> actual(static_cast<std::size_t>(kc * lanes),
                                    -7.0f);
          table[i].kernel.pack_transpose(source.data(), lda, lanes, rows, kc,
                                         actual.data());
          for (std::size_t j = 0; j < expected.size(); ++j) {
            LLM_CHECK_MSG(actual[j] == expected[j],
                          table[i].kernel.name
                              << ": дорожек " << lanes << ", строк " << rows
                              << ", глубина " << kc << ", элемент " << j
                              << " равен " << actual[j] << " вместо "
                              << expected[j]);
          }
          ++checked;
        }
      }
    }
  }
  LLM_CHECK_MSG(checked > 0, "не проверено ни одного ядра");
}

LLM_TEST(Gemm, DepthSumIsSplitIntoBlocksOfTheDeclaredLength) {
  // Порядок сложения по глубине — договор, а не деталь реализации: от него
  // зависят младшие разряды всех весов, какие проект когда-либо обучит.
  // Записан он двумя числами: длиной куска и тем, что внутри куска слагаемые
  // идут последовательно, слитным умножением с накоплением.
  //
  // Проверяется это независимым счётом, а не отпечатком: отпечаток сказал бы
  // «стало иначе», а эталон говорит, чем именно должно быть. Проверено
  // подменой дважды: при куске 96 вместо 128 расходятся 6630 элементов из
  // 8000, и при сложении внутри куска без слитного умножения — тоже. Прежде
  // ни того, ни другого не замечала ни одна проверка проекта.
  //
  // Скалярное ядро в сравнение не входит: у него нет слитного умножения, оно
  // делает умножение и сложение по отдельности, с двумя округлениями вместо
  // одного — то же исключение, что в VectorKernelsAgreeBitForBit.

  // Длина куска спрашивается у самого gemm — так эталон ниже проверяет
  // структуру суммы независимо от того, где живёт константа. Но само её
  // значение закреплено отдельной строкой: без неё тест подстроился бы под
  // любое новое значение и смены не заметил. Проверено — так и было, пока
  // этой строки не было.
  //
  // Менять 128 можно, но тогда все веса, какие проект обучил, сдвинутся в
  // младших разрядах, и числа обучения в README придётся мерить заново.
  const std::int64_t chunk = llm::ops::gemm_depth_block();
  LLM_CHECK_MSG(chunk == 128,
                "длина куска по глубине стала "
                    << chunk
                    << " вместо 128: все веса сдвинутся в младших разрядах");

  // n не кратно 32, поэтому прямой путь выключен и работает блочный — тот
  // самый, который режет глубину. Глубина взята больше трёх кусков, с
  // остатком.
  llm::Rng rng(11);
  const std::int64_t m = 40;
  const std::int64_t n = 200;
  const std::int64_t k = chunk * 3 + 17;
  const std::vector<float> a = random_matrix(&rng, m, k);
  const std::vector<float> b = random_matrix(&rng, k, n);

  std::vector<float> expected(static_cast<std::size_t>(m * n), 0.0f);
  for (std::int64_t i = 0; i < m; ++i) {
    for (std::int64_t j = 0; j < n; ++j) {
      float total = 0.0f;
      for (std::int64_t begin = 0; begin < k; begin += chunk) {
        const std::int64_t end = std::min(begin + chunk, k);
        float part = 0.0f;
        for (std::int64_t p = begin; p < end; ++p) {
          part = std::fma(a[static_cast<std::size_t>(i * k + p)],
                          b[static_cast<std::size_t>(p * n + j)], part);
        }
        total += part;
      }
      expected[static_cast<std::size_t>(i * n + j)] = total;
    }
  }

  int count = 0;
  const llm::ops::MicroKernelChoice* table =
      llm::ops::all_micro_kernels(&count);
  int checked = 0;
  for (int i = 0; i < count; ++i) {
    if (!table[i].available || !table[i].kernel.fused) {
      continue;
    }
    const ForcedKernel forced(table[i].kernel);
    std::vector<float> actual(static_cast<std::size_t>(m * n), 0.0f);
    llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                   actual.data(), n);
    for (std::size_t index = 0; index < expected.size(); ++index) {
      LLM_CHECK_MSG(
          expected[index] == actual[index],
          "ядро '" << table[i].kernel.name << "', элемент " << index << ": "
                   << actual[index] << " вместо " << expected[index]
                   << " — сумма по глубине сложена не кусками по " << chunk);
    }
    ++checked;
  }
  LLM_CHECK_MSG(checked > 0, "на этой машине нет ни одного слитного ядра");
}

LLM_TEST(Gemm, KernelsAgreeBitForBitWithScalingToo) {
  // VectorKernelsAgreeBitForBit сравнивает ядра при alpha = 1 и beta = 0, а
  // это ровно тот случай, в котором разница спрятана.
  //
  // У каждого микроядра два пути записи результата. Полная плитка пишется
  // слитным умножением с накоплением — одно округление на элемент. Неполная,
  // краевая, пишется обычным c += alpha * acc, и слитным его делает уже не
  // код, а компилятор. При alpha = 1 умножение точное, и оба пути дают одно и
  // то же независимо от того, слил компилятор или нет. При alpha, которое не
  // степень двойки, — уже нет.
  //
  // Плитки у ядер разной ширины (16 у AVX2, 32 у AVX-512), поэтому полными и
  // краевыми у них оказываются РАЗНЫЕ куски одной и той же задачи. Стоит
  // одному компилятору перестать сливать в хвосте — и ядра разойдутся, то
  // есть обучение на двух машинах даст разные веса. Проверено: сегодня и gcc,
  // и clang сливают, расхождений нет.
  llm::Rng rng(20260920);
  const std::int64_t m = 200;
  const std::int64_t n = 150;  // не кратно ни 16, ни 32: края есть у обоих
  const std::int64_t k = 300;
  const std::vector<float> a = random_matrix(&rng, m, k);
  const std::vector<float> b = random_matrix(&rng, k, n);
  const std::vector<float> c_initial = random_matrix(&rng, m, n);

  // 0.3 и -2.5 не представимы точно как степень двойки, поэтому умножение на
  // них округляет; 0.25 — представима, и на ней разницы не будет даже при
  // разном округлении. Она здесь как раз затем, чтобы это было видно.
  const float alphas[] = {0.3f, -2.5f, 0.25f};
  const float betas[] = {0.5f, 1.0f};

  int count = 0;
  const llm::ops::MicroKernelChoice* table =
      llm::ops::all_micro_kernels(&count);
  int compared = 0;

  for (std::size_t ai = 0; ai < sizeof(alphas) / sizeof(alphas[0]); ++ai) {
    for (std::size_t bi = 0; bi < sizeof(betas) / sizeof(betas[0]); ++bi) {
      std::vector<float> reference;
      const char* reference_name = "";
      for (int i = 0; i < count; ++i) {
        if (!table[i].available || !table[i].kernel.fused) {
          continue;
        }
        const ForcedKernel forced(table[i].kernel);
        std::vector<float> actual = c_initial;
        llm::ops::gemm(false, false, m, n, k, alphas[ai], a.data(), k, b.data(),
                       n, betas[bi], actual.data(), n);
        if (reference.empty()) {
          reference = actual;
          reference_name = table[i].kernel.name;
          continue;
        }
        ++compared;
        for (std::size_t index = 0; index < reference.size(); ++index) {
          LLM_CHECK_MSG(reference[index] == actual[index],
                        "alpha=" << alphas[ai] << " beta=" << betas[bi]
                                 << ": ядра '" << reference_name << "' и '"
                                 << table[i].kernel.name << "' расходятся в "
                                 << index << ": " << reference[index]
                                 << " против " << actual[index]);
        }
      }
    }
  }
  LLM_CHECK_MSG(compared > 0 || count < 3,
                "сравнить оказалось нечего: доступно ядер " << count);
}

LLM_TEST(Gemm, EveryKernelMatchesNaiveOnEveryTileRemainder) {
  // Сверка с наивным эталоном шла только для ядра, выбранного на этой машине.
  // Остальные сверялись лишь друг с другом — побитово, если слитные, и с
  // допуском, если нет. Этого мало: ошибка в обработке неполной плитки,
  // одинаковая у всех ядер, прошла бы такую проверку насквозь, потому что
  // сравнивать их было бы не с чем.
  //
  // Размеры подобраны так, чтобы дать остаток по каждой ширине плитки, какая
  // есть в проекте: 4 и 32 у скалярного, 6 и 16 у AVX2, 8 и 32 у AVX-512,
  // 12 и 8 у NEON. Глубина — единица, мелкое число, ровно один блок по
  // глубине (128) и блок с остатком.
  const std::int64_t sizes[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                                11, 12, 13, 16, 17, 24, 31, 32, 33};
  const std::int64_t depths[] = {1, 5, 128, 130};
  const std::size_t size_count = sizeof(sizes) / sizeof(sizes[0]);
  const std::size_t depth_count = sizeof(depths) / sizeof(depths[0]);

  int count = 0;
  const llm::ops::MicroKernelChoice* table =
      llm::ops::all_micro_kernels(&count);
  LLM_CHECK_GT(count, 0);

  int checked_kernels = 0;
  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      continue;
    }
    const ForcedKernel forced(table[i].kernel);
    ++checked_kernels;
    // Своё зерно на ядро: одинаковые данные у всех ядер ничего не добавляют,
    // а разные расширяют охват.
    llm::Rng rng(90000 + i);
    for (std::size_t mi = 0; mi < size_count; ++mi) {
      for (std::size_t ni = 0; ni < size_count; ++ni) {
        const std::int64_t k = depths[(mi + ni) % depth_count];
        // Масштабы чередуются: хвост ядра, где считается alpha * acc + c,
        // при alpha = 1 и beta = 0 упрощается и проверяется не весь.
        const bool scaled = ((mi + ni) % 2) != 0;
        compare_with_naive(&rng, false, false, sizes[mi], sizes[ni], k,
                           scaled ? 0.75f : 1.0f, scaled ? 0.5f : 0.0f, 0);
      }
    }
  }
  LLM_CHECK_MSG(checked_kernels > 0, "ни одного доступного ядра");
}

LLM_TEST(Gemm, KnownProductInAllFourTransposeCombinations) {
  // Транспонированные варианты сверялись только gemm против gemm_naive, то
  // есть по кругу: согласованная ошибка в понимании того, что означает
  // transpose_a, прошла бы обе реализации насквозь. Посчитанный вручную
  // пример эту петлю разрывает.
  //
  //   op(A) = [[1, 2, 3],   op(B) = [[1, 2],    op(A)*op(B) = [[22, 28],
  //            [4, 5, 6]]            [3, 4],                   [49, 64]]
  //                                  [5, 6]]
  //
  // Меняется только то, КАК эти же матрицы лежат в памяти. Шаг строки задан
  // до транспонирования, поэтому у транспонированной A он равен двум, а не
  // трём.
  const std::vector<float> a_plain = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> a_stored = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
  const std::vector<float> b_plain = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> b_stored = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f};
  const double expected[4] = {22.0, 28.0, 49.0, 64.0};

  for (int mode = 0; mode < 4; ++mode) {
    const bool transpose_a = (mode & 1) != 0;
    const bool transpose_b = (mode & 2) != 0;
    const std::vector<float>& a = transpose_a ? a_stored : a_plain;
    const std::vector<float>& b = transpose_b ? b_stored : b_plain;
    const std::int64_t lda = transpose_a ? 2 : 3;
    const std::int64_t ldb = transpose_b ? 3 : 2;

    std::vector<float> blocked(4, 0.0f);
    llm::ops::gemm(transpose_a, transpose_b, 2, 2, 3, 1.0f, a.data(), lda,
                   b.data(), ldb, 0.0f, blocked.data(), 2);
    std::vector<float> naive(4, 0.0f);
    llm::ops::gemm_naive(transpose_a, transpose_b, 2, 2, 3, 1.0f, a.data(), lda,
                         b.data(), ldb, 0.0f, naive.data(), 2);

    for (std::size_t i = 0; i < 4; ++i) {
      LLM_CHECK_MSG(std::fabs(blocked[i] - expected[i]) < 1e-5,
                    "tA=" << transpose_a << " tB=" << transpose_b
                          << ": блочный дал " << blocked[i] << " вместо "
                          << expected[i] << " в элементе " << i);
      LLM_CHECK_MSG(std::fabs(naive[i] - expected[i]) < 1e-5,
                    "tA=" << transpose_a << " tB=" << transpose_b
                          << ": наивный дал " << naive[i] << " вместо "
                          << expected[i] << " в элементе " << i);
    }
  }
}

// Учёт трафика считает все три матрицы на обоих путях.
//
// Счётчик существует затем, чтобы отвечать числом на вопрос «где лежат байты»,
// и из его показаний взяты утверждения о том, что при генерации веса дают две
// трети чтения. Проверялся он при этом ничем.
//
// Оказалось, что накопитель C учитывался только на прямом пути. На блочном не
// учитывался вовсе — то есть ровно там, где C и велик: блочный путь режет
// глубину, и каждый её кусок читает и переписывает весь прямоугольник заново.
// Наружу это выходило не нулём в отчёте, а перераспределением долей: A и B
// оставались верными, а доля весов от этого оказывалась завышенной.
//
// Ожидания посчитаны по форме задачи, а не сняты с самой реализации:
//   A упаковывается один раз на каждый кусок глубины, всего m * k значений;
//   B — столько же раз, всего n * k;
//   C читается и пишется на каждом куске глубины, то есть m * n * 2 значений
//     на кусок, а кусков — k, поделённое на длину куска с округлением вверх.
LLM_TEST(Gemm, TrafficCountsEveryMatrixOnBothPaths) {
  // Восстанавливается прежнее состояние, а не «выключено»: учёт включают
  // переменной окружения, и прогон набора с ней не должен ломаться об эту
  // проверку.
  struct Guard {
    bool was;
    Guard() : was(llm::ops::traffic().enabled) {
      llm::ops::set_traffic_enabled(true);
    }
    ~Guard() {
      llm::ops::set_traffic_enabled(was);
      llm::ops::reset_traffic();
    }
  } guard;

  // Ширина одна: при делении по столбцам каждый поток перечитывает A целиком,
  // и суммарный трафик честно зависит от числа потоков. Проверка про формулу,
  // а не про деление.
  const WidthGuard width(1);

  llm::Rng rng(20260921);
  const std::int64_t m = 64;
  // Не кратно тридцати двум — значит блочный путь на любой машине: условие
  // выбора пути спрашивает про общее число, а не про ширину плитки ядра.
  const std::int64_t n = 48;
  // Больше двух кусков по глубине, с остатком: иначе множитель у C был бы
  // единицей, и пропуск учёта на блочном пути выглядел бы как верный ответ.
  const std::int64_t k = 300;
  const std::vector<float> a = random_matrix(&rng, m, k);
  const std::vector<float> b = random_matrix(&rng, k, n);
  std::vector<float> c(static_cast<std::size_t>(m * n), 0.0f);

  llm::ops::reset_traffic();
  llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                 c.data(), n);
  const llm::ops::GemmTraffic blocked = llm::ops::traffic();

  LLM_CHECK_MSG(blocked.enabled, "учёт не включился, проверка пуста");
  const std::int64_t chunk = llm::ops::gemm_depth_block();
  const std::int64_t depth_blocks = (k + chunk - 1) / chunk;
  LLM_CHECK_MSG(depth_blocks > 1,
                "глубина уложилась в один кусок, множитель у C не проверяется");

  LLM_CHECK_EQ(blocked.a_bytes, static_cast<long long>(m * k * 4));
  LLM_CHECK_EQ(blocked.b_bytes, static_cast<long long>(n * k * 4));
  LLM_CHECK_MSG(
      blocked.c_bytes == static_cast<long long>(m * n * 4 * 2 * depth_blocks),
      "накопитель на блочном пути учтён как " << blocked.c_bytes << " вместо "
                                              << m * n * 4 * 2 * depth_blocks);

  // Прямой путь: своя формула, но накопитель обязан быть учтён и там.
  // Ширина плитки у ядер разная, поэтому A перечитывается разное число раз —
  // берём его у выбранного ядра, а не вписываем числом.
  const std::int64_t direct_m = 4;  // не больше порога по числу строк
  const std::int64_t direct_n = 64;  // кратно тридцати двум
  const std::int64_t direct_k = 32;
  const std::vector<float> da = random_matrix(&rng, direct_m, direct_k);
  const std::vector<float> db = random_matrix(&rng, direct_k, direct_n);
  std::vector<float> dc(static_cast<std::size_t>(direct_m * direct_n), 0.0f);

  llm::ops::reset_traffic();
  llm::ops::gemm(false, false, direct_m, direct_n, direct_k, 1.0f, da.data(),
                 direct_k, db.data(), direct_n, 0.0f, dc.data(), direct_n);
  const llm::ops::GemmTraffic direct = llm::ops::traffic();

  const std::int64_t nr = llm::ops::best_micro_kernel().nr;
  const std::int64_t mr = llm::ops::best_micro_kernel().mr;
  const std::int64_t column_blocks = direct_n / nr;
  const std::int64_t row_blocks = (direct_m + mr - 1) / mr;
  LLM_CHECK_EQ(direct.a_bytes,
               static_cast<long long>(direct_m * direct_k * 4 * column_blocks));
  LLM_CHECK_EQ(direct.b_bytes,
               static_cast<long long>(direct_k * direct_n * 4 * row_blocks));
  LLM_CHECK_EQ(direct.c_bytes,
               static_cast<long long>(direct_m * direct_n * 4 * 2));

  // И выключенный учёт ничего не считает: замер не вправе платить за то, чего
  // не просил.
  llm::ops::set_traffic_enabled(false);
  llm::ops::reset_traffic();
  llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                 c.data(), n);
  const llm::ops::GemmTraffic off = llm::ops::traffic();
  LLM_CHECK(!off.enabled);
  LLM_CHECK_EQ(off.a_bytes, static_cast<long long>(0));
  LLM_CHECK_EQ(off.b_bytes, static_cast<long long>(0));
  LLM_CHECK_EQ(off.c_bytes, static_cast<long long>(0));
}
