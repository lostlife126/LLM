#include <algorithm>
#include <cmath>
#include <cstdint>
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

LLM_TEST(Gemm, PathChoiceDoesNotDependOnTheKernel) {
  // Прямой и блочный пути делят глубину по-разному, поэтому дают чуть разные
  // младшие разряды. Это допустимо — но только если выбор пути одинаков на
  // любой машине. Иначе обучение на процессоре с AVX2 и на процессоре с
  // AVX-512 разошлось бы, и сравнивать их было бы нельзя.
  //
  // Проверяется это так: одна и та же задача считается всеми доступными
  // ядрами, и требуется, чтобы расхождение оставалось на уровне одного
  // округления. Если бы ядра выбирали разные пути, разница вышла бы заметно
  // больше — глубина здесь нарочно взята больше блока, чтобы блочный путь её
  // поделил.
  //
  // Ширина 48 — тот самый случай, ради которого условие проверяет кратность
  // общему числу, а не ширине плитки: 48 делится на 16, но не на 32.
  llm::Rng rng(1234);
  const std::int64_t widths[] = {48, 64, 96, 128};
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
      if (!table[i].available) {
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
      const double error = max_relative_error(reference, actual, m, n, n);
      LLM_CHECK_MSG(error < 1e-6,
                    "ядра '" << reference_name << "' и '"
                             << table[i].kernel.name << "' при n = " << n
                             << " разошлись на " << error
                             << " — похоже, они выбрали разные пути");
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
