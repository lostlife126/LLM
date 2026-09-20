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

    llm::set_parallel_width(1);
    std::vector<float> serial(static_cast<std::size_t>(m * n), 0.0f);
    llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                   serial.data(), n);

    llm::set_parallel_width(0);
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
  llm::set_parallel_width(0);
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
    llm::ops::force_micro_kernel(&table[i].kernel);
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
  llm::ops::force_micro_kernel(nullptr);

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
    llm::ops::force_micro_kernel(&table[i].kernel);
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
  llm::ops::force_micro_kernel(nullptr);
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

  llm::set_parallel_width(1);
  std::vector<float> serial(static_cast<std::size_t>(m * n), 0.0f);
  llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                 serial.data(), n);

  llm::set_parallel_width(0);
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

  llm::set_parallel_width(1);
  std::vector<float> serial(static_cast<std::size_t>(m * n), 0.25f);
  llm::ops::gemm(false, false, m, n, k, 1.5f, a.data(), k, b.data(), n, 0.5f,
                 serial.data(), n);

  llm::set_parallel_width(0);
  std::vector<float> threaded(static_cast<std::size_t>(m * n), 0.25f);
  llm::ops::gemm(false, false, m, n, k, 1.5f, a.data(), k, b.data(), n, 0.5f,
                 threaded.data(), n);

  for (std::size_t index = 0; index < serial.size(); ++index) {
    LLM_CHECK_MSG(serial[index] == threaded[index],
                  "элемент " << index << ": " << threaded[index] << " вместо "
                             << serial[index]);
  }

  // beta здесь не единица нарочно. Каждый поток применяет beta к своей полосе
  // столбцов, и если бы полосы пересеклись или какая-то осталась без beta,
  // видно было бы именно по этому.
  llm::set_parallel_width(0);
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

  llm::set_parallel_width(1);
  std::vector<float> serial(static_cast<std::size_t>(m * n), 0.25f);
  llm::ops::gemm_half_b(false, m, n, k, 1.5f, a.data(), k, packed.data(), n,
                        0.5f, serial.data(), n);

  llm::set_parallel_width(0);
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
      llm::ops::force_micro_kernel(&table[i].kernel);
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
    llm::ops::force_micro_kernel(nullptr);
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
  const llm::ops::MicroKernelChoice* table = llm::ops::all_micro_kernels(&count);
  int checked = 0;
  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      continue;
    }
    llm::ops::force_micro_kernel(&table[i].kernel);
    for (std::size_t j = 0; j < sizeof(kHalfCases) / sizeof(kHalfCases[0]);
         ++j) {
      check_half_matches_rounded(kHalfCases[j]);
    }
    ++checked;
  }
  llm::ops::force_micro_kernel(nullptr);
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
  const llm::ops::MicroKernelChoice* table = llm::ops::all_micro_kernels(&count);
  int checked = 0;
  for (int i = 0; i < count; ++i) {
    if (!table[i].available || table[i].kernel.run_rows_half == nullptr) {
      continue;
    }
    llm::ops::MicroKernel crippled = table[i].kernel;
    crippled.run_rows_half = nullptr;
    llm::ops::force_micro_kernel(&crippled);
    for (std::size_t j = 0; j < sizeof(kHalfCases) / sizeof(kHalfCases[0]);
         ++j) {
      check_half_matches_rounded(kHalfCases[j]);
    }
    ++checked;
  }
  llm::ops::force_micro_kernel(nullptr);
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
  const llm::ops::MicroKernelChoice* table = llm::ops::all_micro_kernels(&count);
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

        std::vector<float> expected(
            static_cast<std::size_t>(kc * lanes), -7.0f);
        reference(source.data(), lda, lanes, rows, kc, expected.data());

        for (int i = 0; i < count; ++i) {
          if (!table[i].available) {
            continue;
          }
          std::vector<float> actual(
              static_cast<std::size_t>(kc * lanes), -7.0f);
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
  const llm::ops::MicroKernelChoice* table = llm::ops::all_micro_kernels(&count);
  int checked = 0;
  for (int i = 0; i < count; ++i) {
    if (!table[i].available || !table[i].kernel.fused) {
      continue;
    }
    llm::ops::force_micro_kernel(&table[i].kernel);
    std::vector<float> actual(static_cast<std::size_t>(m * n), 0.0f);
    llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n, 0.0f,
                   actual.data(), n);
    for (std::size_t index = 0; index < expected.size(); ++index) {
      LLM_CHECK_MSG(expected[index] == actual[index],
                    "ядро '" << table[i].kernel.name << "', элемент " << index
                             << ": " << actual[index] << " вместо "
                             << expected[index]
                             << " — сумма по глубине сложена не кусками по "
                             << chunk);
    }
    ++checked;
  }
  llm::ops::force_micro_kernel(nullptr);
  LLM_CHECK_MSG(checked > 0, "на этой машине нет ни одного слитного ядра");
}
