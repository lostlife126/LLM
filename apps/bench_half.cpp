// Стоит ли хранить веса в половинной разрядности.
//
// Вопрос узкий и измеримый. Замер трафика GEMM говорит, что при генерации по
// одному токену чтение весов даёт две трети всех прочитанных байт, а форма
// там крайняя: m = 1, то есть на каждый прочитанный вес приходится ровно одно
// умножение с накоплением. Арифметической работы столько же, сколько чтения,
// и всё упирается в память. Если так, половинная разрядность обязана дать
// примерно вдвое — и вот это «примерно» и надо увидеть числом.
//
// Ядро здесь ни на что не влияет: библиотека его не вызывает. Оно нужно,
// чтобы сравнение было честным — обе стороны считают одной и той же формой
// цикла, с одним и тем же числом накопителей, и отличаются ровно типом
// элемента B. Сравнивать самодельное ядро с настоящим было бы сравнением двух
// разных вещей сразу.
//
// Запуск: ./bench_half
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define LLM_HALF_X86 1
#else
#define LLM_HALF_X86 0
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "args.h"
#include "core/cpu.h"
#include "core/half.h"
#include "core/random.h"
#include "core/thread_pool.h"
#include "measure.h"
#include "ops/gemm.h"

namespace {

// Ширина плитки по столбцам. Восемь накопителей, а не два: при m = 1 цепочка
// накопления по k — это зависимость длиной в k, и с двумя накопителями ядро
// упёрлось бы в задержку умножения с накоплением, а не в память. Восемь
// покрывают её с запасом, и замер меряет то, что задумано.
constexpr int64_t kTileN = 64;
// Число накопителей записано в самих ядрах поимённо: acc0..acc7. Отдельной
// константы для него нет нарочно — имена накопителей обязаны быть именами, а
// не индексами массива, иначе компилятор кладёт массив в память и ядро теряет
// ровно то, ради чего написано.

#if LLM_HALF_X86

// Одна строка результата, kTileN столбцов, B в обычной разрядности.
__attribute__((target("avx2,fma"))) void row_f32(int64_t k, const float* a,
                                                 const float* b, int64_t ldb,
                                                 float* c) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();
  __m256 acc7 = _mm256_setzero_ps();

  for (int64_t p = 0; p < k; ++p) {
    const __m256 av = _mm256_set1_ps(a[p]);
    const float* row = b + p * ldb;
    acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(row + 0), acc0);
    acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(row + 8), acc1);
    acc2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(row + 16), acc2);
    acc3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(row + 24), acc3);
    acc4 = _mm256_fmadd_ps(av, _mm256_loadu_ps(row + 32), acc4);
    acc5 = _mm256_fmadd_ps(av, _mm256_loadu_ps(row + 40), acc5);
    acc6 = _mm256_fmadd_ps(av, _mm256_loadu_ps(row + 48), acc6);
    acc7 = _mm256_fmadd_ps(av, _mm256_loadu_ps(row + 56), acc7);
  }

  _mm256_storeu_ps(c + 0, acc0);
  _mm256_storeu_ps(c + 8, acc1);
  _mm256_storeu_ps(c + 16, acc2);
  _mm256_storeu_ps(c + 24, acc3);
  _mm256_storeu_ps(c + 32, acc4);
  _mm256_storeu_ps(c + 40, acc5);
  _mm256_storeu_ps(c + 48, acc6);
  _mm256_storeu_ps(c + 56, acc7);
}

// То же самое с B в половинной разрядности. Отличие ровно одно: вместо
// загрузки восьми float идёт загрузка восьми 16-разрядных значений и их
// развёртывание командой vcvtph2ps. Развёртывание точное — каждое значение
// половинной разрядности представимо в float, — поэтому накопление идёт в той
// же разрядности и в том же порядке, что и у соседа сверху. Результат обязан
// совпасть с ним побитово, если тому дать уже округлённые веса; это
// проверяется ниже.
__attribute__((target("avx2,fma,f16c"))) void row_f16(int64_t k, const float* a,
                                                      const llm::Half* b,
                                                      int64_t ldb, float* c) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();
  __m256 acc7 = _mm256_setzero_ps();

  for (int64_t p = 0; p < k; ++p) {
    const __m256 av = _mm256_set1_ps(a[p]);
    const __m128i* row = reinterpret_cast<const __m128i*>(b + p * ldb);
    acc0 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(row + 0)), acc0);
    acc1 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(row + 1)), acc1);
    acc2 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(row + 2)), acc2);
    acc3 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(row + 3)), acc3);
    acc4 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(row + 4)), acc4);
    acc5 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(row + 5)), acc5);
    acc6 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(row + 6)), acc6);
    acc7 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(row + 7)), acc7);
  }

  _mm256_storeu_ps(c + 0, acc0);
  _mm256_storeu_ps(c + 8, acc1);
  _mm256_storeu_ps(c + 16, acc2);
  _mm256_storeu_ps(c + 24, acc3);
  _mm256_storeu_ps(c + 32, acc4);
  _mm256_storeu_ps(c + 40, acc5);
  _mm256_storeu_ps(c + 48, acc6);
  _mm256_storeu_ps(c + 56, acc7);
}

void run_f32(int64_t m, int64_t n, int64_t k, const float* a, const float* b,
             float* c) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; j += kTileN) {
      row_f32(k, a + i * k, b + j, n, c + i * n + j);
    }
  }
}

void run_f16(int64_t m, int64_t n, int64_t k, const float* a,
             const llm::Half* b, float* c) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; j += kTileN) {
      row_f16(k, a + i * k, b + j, n, c + i * n + j);
    }
  }
}

void measure(int64_t m, int64_t n, int64_t k) {
  // Ядро прототипа идёт по n шагом kTileN и хвоста не разбирает: оно написано
  // под формы модели, а те кратны. Некратная не упала бы, а прочитала и
  // записала бы за границей — замер вышел бы про соседнюю память.
  if (n % kTileN != 0) {
    std::printf("n=%lld не кратно %lld — ядро прототипа эту форму не считает\n",
                static_cast<long long>(n), static_cast<long long>(kTileN));
    return;
  }
  llm::Rng rng(1234);
  std::vector<float> a(static_cast<std::size_t>(m * k));
  std::vector<float> b(static_cast<std::size_t>(k * n));
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = rng.normal() * 0.05f;
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = rng.normal() * 0.05f;

  // Веса округляются до половинной разрядности ОБЕИМ сторонам. Иначе сравнение
  // молча смешало бы два разных вопроса — скорость и точность, — а здесь
  // спрашивается только про скорость.
  std::vector<llm::Half> bh(b.size());
  llm::floats_to_half(b.data(), bh.data(), static_cast<int64_t>(b.size()));
  llm::half_to_floats(bh.data(), b.data(), static_cast<int64_t>(b.size()));

  std::vector<float> c32(static_cast<std::size_t>(m * n), 0.0f);
  std::vector<float> c16(static_cast<std::size_t>(m * n), 0.0f);

  run_f32(m, n, k, a.data(), b.data(), c32.data());
  run_f16(m, n, k, a.data(), bh.data(), c16.data());

  int64_t mismatches = 0;
  for (std::size_t i = 0; i < c32.size(); ++i) {
    if (c32[i] != c16[i]) ++mismatches;
  }

  const double seconds32 = bench::best_seconds(
      [&]() { run_f32(m, n, k, a.data(), b.data(), c32.data()); }, 0.4);
  const double seconds16 = bench::best_seconds(
      [&]() { run_f16(m, n, k, a.data(), bh.data(), c16.data()); }, 0.4);

  const double b_bytes = static_cast<double>(k) * static_cast<double>(n);
  std::printf(
      "m=%-4lld k=%-5lld n=%-5lld  B=%6.2f МБ | fp32 %7.3f мс %6.1f ГБ/с | "
      "fp16 %7.3f мс %6.1f ГБ/с | %.2fx | расхождений %lld\n",
      static_cast<long long>(m), static_cast<long long>(k),
      static_cast<long long>(n), 4.0 * b_bytes / 1048576.0, seconds32 * 1e3,
      4.0 * b_bytes * static_cast<double>(m) / seconds32 / 1e9, seconds16 * 1e3,
      2.0 * b_bytes * static_cast<double>(m) / seconds16 / 1e9,
      seconds32 / seconds16, static_cast<long long>(mismatches));
}

#endif  // LLM_HALF_X86

// Второй замер: то же самое, но уже настоящими функциями библиотеки. Первый
// отвечал на вопрос «стоит ли это писать», второй — на вопрос «написано ли оно
// так, как задумано». Числа у них разные и сравнивать их между собой не надо:
// ядро прототипа считает плиткой 1 x 64 без упаковки, библиотека выбирает путь
// и плитку сама.
void measure_library(int64_t m, int64_t n, int64_t k) {
  llm::Rng rng(4321);
  std::vector<float> a(static_cast<std::size_t>(m * k));
  std::vector<float> b(static_cast<std::size_t>(k * n));
  for (std::size_t i = 0; i < a.size(); ++i) a[i] = rng.normal() * 0.05f;
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = rng.normal() * 0.05f;

  std::vector<llm::Half> bh(b.size());
  llm::floats_to_half(b.data(), bh.data(), static_cast<int64_t>(b.size()));
  llm::half_to_floats(bh.data(), b.data(), static_cast<int64_t>(b.size()));

  std::vector<float> c32(static_cast<std::size_t>(m * n), 0.0f);
  std::vector<float> c16(static_cast<std::size_t>(m * n), 0.0f);

  const double seconds32 = bench::best_seconds(
      [&]() {
        llm::ops::gemm(false, false, m, n, k, 1.0f, a.data(), k, b.data(), n,
                       0.0f, c32.data(), n);
      },
      0.4);
  const double seconds16 = bench::best_seconds(
      [&]() {
        llm::ops::gemm_half_b(false, m, n, k, 1.0f, a.data(), k, bh.data(), n,
                              0.0f, c16.data(), n);
      },
      0.4);

  int64_t mismatches = 0;
  for (std::size_t i = 0; i < c32.size(); ++i) {
    if (c32[i] != c16[i]) ++mismatches;
  }

  std::printf(
      "m=%-4lld k=%-5lld n=%-5lld  B=%6.2f МБ | fp32 %7.3f мс | fp16 %7.3f мс "
      "| %.2fx | расхождений %lld\n",
      static_cast<long long>(m), static_cast<long long>(k),
      static_cast<long long>(n),
      4.0 * static_cast<double>(k) * static_cast<double>(n) / 1048576.0,
      seconds32 * 1e3, seconds16 * 1e3, seconds32 / seconds16,
      static_cast<long long>(mismatches));
}

}  // namespace

int main(int argc, char**) {
  // Аргументов это приложение не принимает, и лишний обязан
  // получить отказ, а не быть отброшенным молча.
  bench::expect_at_most(argc, 0, "без аргументов");
  std::printf("процессор: %s\n", llm::cpu_features().to_string().c_str());

  // Формы взяты из настоящей модели: k — это d_model или ffn_hidden, n — их
  // же или размер словаря. m = 1 — генерация по токену, m больше единицы —
  // тот же слой при обучении, для проверки, что вывод про память относится
  // именно к генерации.
  static const int64_t shapes[][3] = {
      {1, 256, 256},  {1, 256, 704},  {1, 704, 256},   {1, 256, 4096},
      {1, 512, 4096}, {4, 256, 4096}, {16, 256, 4096}, {64, 256, 4096},
  };
  const std::size_t count = sizeof(shapes) / sizeof(shapes[0]);

  std::printf("\nбиблиотека: gemm против gemm_half_b\n");
  for (std::size_t i = 0; i < count; ++i) {
    measure_library(shapes[i][0], shapes[i][2], shapes[i][1]);
  }

#if !LLM_HALF_X86
  std::printf(
      "\nотдельное ядро прототипа написано под x86 с F16C; пропущено\n");
  return 0;
#else
  if (!llm::cpu_features().has_avx2_f16c()) {
    std::printf("\nнет F16C; ядро прототипа пропущено\n");
    return 0;
  }
  std::printf("\nпрототип: одна форма цикла, отличается только тип B\n");

  for (std::size_t i = 0; i < count; ++i) {
    measure(shapes[i][0], shapes[i][2], shapes[i][1]);
  }
  return 0;
#endif
}
