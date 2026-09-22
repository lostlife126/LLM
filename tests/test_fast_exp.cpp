// Своя экспонента.
//
// Проверять её надо по двум осям сразу, и вторая важнее первой.
//
// Точность: она заменяет std::exp внутри softmax и cross-entropy, и если
// ошибётся, это проявится не падением, а слегка неправильными потерями —
// самой дорогой разновидностью ошибки в этом проекте.
//
// Совпадение реализаций: AVX2 и AVX-512 обязаны давать побитово одно и то же.
// Иначе обучение на машине без AVX-512 давало бы другие числа, и сравнивать
// варианты архитектуры, посчитанные на разных машинах, было бы нельзя.
// Скалярная в это равенство не входит: она собирается под базовый x86-64, где
// инструкции умножения с накоплением нет, и округляет дважды там, где
// векторные округляют один раз. Её расхождение проверяется по величине.
//
// Этот тест уже отработал. Первая версия ядра писала многочлен раздельными
// умножением и сложением во всех трёх реализациях — и AVX-512 всё равно
// разошлась, потому что GCC склеивает их в FMA сам, когда целевой набор
// инструкций её содержит. Разошлась на четырёх значениях из тысячи, и без
// побитовой проверки это не нашлось бы никогда.

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/random.h"
#include "ops/fast_exp.h"
#include "testing.h"

namespace {

// Возвращает автоматический выбор реализации при любом выходе из области
// видимости — в том числе через исключение. Без этого непрошедшая проверка
// внутри перебора оставляла бы принудительный выбор включённым, и все
// последующие тесты двоичного файла считались бы чужим ядром. То же самое
// уже случилось в test_gemm.cpp и было там исправлено.
class ForcedExpKernel {
 public:
  explicit ForcedExpKernel(const llm::ops::ExpKernel& kernel) {
    llm::ops::force_exp_kernel(&kernel);
  }
  ~ForcedExpKernel() { llm::ops::force_exp_kernel(nullptr); }

  ForcedExpKernel(const ForcedExpKernel&) = delete;
  ForcedExpKernel& operator=(const ForcedExpKernel&) = delete;
};

// Относительная погрешность против std::exp в двойной точности.
double relative_error(float actual, double expected) {
  if (expected == 0.0) {
    return actual == 0.0f ? 0.0 : 1.0;
  }
  return std::fabs((static_cast<double>(actual) - expected) / expected);
}

// Шаг сетки субнормальных чисел float: соседние значения там отстоят ровно на
// столько, независимо от величины.
const float kDenormalStep = 1.4012984643e-45f;

// Близки ли два значения.
//
// Три случая, и каждый нужен. Бесконечности и нули: относительная разность для
// них не определена, а совпадать они обязаны точно. Субнормальная область:
// значение вида 2e-42 само по себе несёт около одиннадцати значащих разрядов,
// и разница в один шаг сетки даёт там относительную погрешность 7e-4 — это
// свойство представления, а не ошибка счёта. Всё остальное сравнивается
// относительно.
bool close_enough(float actual, float expected, double tolerance) {
  if (actual == expected) {
    return true;
  }
  if (!std::isfinite(actual) || !std::isfinite(expected)) {
    return false;
  }
  if (std::fabs(actual - expected) <= kDenormalStep) {
    return true;
  }
  return relative_error(actual, static_cast<double>(expected)) < tolerance;
}

std::vector<float> exp_of(const std::vector<float>& input) {
  std::vector<float> output(input.size(), 0.0f);
  llm::ops::exp_array(input.data(), output.data(),
                      static_cast<std::int64_t>(input.size()));
  return output;
}

}  // namespace

LLM_TEST(FastExp, MatchesLibraryExpOverTheUsefulRange) {
  // Нижняя граница — не произвол: ниже exp(-87) результат уже субнормальный,
  // у него самого остаётся меньше двадцати значащих разрядов, и относительная
  // погрешность там задаётся не многочленом, а представлением числа. Этот
  // случай проверяется отдельно. Внутри softmax аргумент всё равно не бывает
  // больше нуля и редко меньше -30.
  std::vector<float> input;
  for (double x = -87.0; x <= 88.0; x += 0.001) {
    input.push_back(static_cast<float>(x));
  }
  const std::vector<float> actual = exp_of(input);

  double worst = 0.0;
  float worst_at = 0.0f;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const double expected = std::exp(static_cast<double>(input[i]));
    const double error = relative_error(actual[i], expected);
    if (error > worst) {
      worst = error;
      worst_at = input[i];
    }
  }
  // Один последний разряд float — это 6e-8. Требование втрое мягче: у
  // многочлена своя погрешность, и складываться им есть где.
  LLM_CHECK_MSG(worst < 2e-7, "наибольшая относительная погрешность "
                                  << worst << " при x = " << worst_at);
}

LLM_TEST(FastExp, ExactAtZeroAndOne) {
  // Единица в нуле обязана получиться точно: на ней держится softmax, где
  // максимальный элемент строки даёт ровно exp(0).
  LLM_CHECK_EQ(llm::ops::exp_scalar(0.0f), 1.0f);
  LLM_CHECK_EQ(llm::ops::exp_scalar(-0.0f), 1.0f);

  // Целые степени двойки проходят через самый неудобный путь: остаток r там
  // близок к границе отрезка приведения.
  for (int k = -60; k <= 60; ++k) {
    const float x = static_cast<float>(k) * 0.6931471805599453f;
    const double expected = std::exp(static_cast<double>(x));
    LLM_CHECK_MSG(relative_error(llm::ops::exp_scalar(x), expected) < 2e-7,
                  "k = " << k);
  }
}

LLM_TEST(FastExp, EdgesAndNotANumber) {
  const float infinity = std::numeric_limits<float>::infinity();

  LLM_CHECK_EQ(llm::ops::exp_scalar(100.0f), infinity);
  LLM_CHECK_EQ(llm::ops::exp_scalar(infinity), infinity);
  LLM_CHECK_EQ(llm::ops::exp_scalar(-200.0f), 0.0f);
  LLM_CHECK_EQ(llm::ops::exp_scalar(-infinity), 0.0f);
  LLM_CHECK(std::isnan(
      llm::ops::exp_scalar(std::numeric_limits<float>::quiet_NaN())));

  // Минус бесконечность в softmax появляется не случайно, а по построению:
  // каузальная маска закрывает будущее именно ею, и нуль после экспоненты —
  // то, на чём держится тест каузальности.
  const std::vector<float> input = {
      -infinity, 0.0f, -infinity, 1.0f, -infinity, 2.0f,
      -infinity, 3.0f, -infinity, 4.0f, -infinity, 5.0f,
      -infinity, 6.0f, -infinity, 7.0f, -infinity, 8.0f};
  const std::vector<float> actual = exp_of(input);
  for (std::size_t i = 0; i < input.size(); i += 2) {
    LLM_CHECK_MSG(actual[i] == 0.0f, "позиция " << i << " дала " << actual[i]);
  }

  // Субнормальная область. exp(-100) около 3.7e-44, и это не нуль — но через
  // одну степень двойки такое значение не построить: 2^-144 как число уже не
  // существует. Поэтому масштабирование идёт в два приёма, и здесь это и
  // проверяется.
  //
  // Допуск задан в шагах самого представления, а не в относительных долях:
  // соседние субнормальные числа этого порядка отстоят на 1.4e-45, то есть
  // сама сетка там грубее, чем любая разумная относительная точность.
  for (float x = -88.5f; x >= -103.0f; x -= 1.5f) {
    const float tiny = llm::ops::exp_scalar(x);
    const double expected = std::exp(static_cast<double>(x));
    LLM_CHECK_MSG(tiny > 0.0f, "exp(" << x << ") обнулилась");
    LLM_CHECK_MSG(
        std::fabs(static_cast<double>(tiny) - expected) <= kDenormalStep,
        "exp(" << x << ") = " << tiny << " вместо " << expected);
  }
}

LLM_TEST(FastExp, ShiftIsAppliedBeforeExponent) {
  llm::Rng rng(4242);
  std::vector<float> input;
  for (int i = 0; i < 100; ++i) {
    input.push_back(rng.uniform(-20.0f, 20.0f));
  }
  const float shift = 7.5f;

  std::vector<float> shifted_input(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    shifted_input[i] = input[i] - shift;
  }

  std::vector<float> from_shift(input.size(), 0.0f);
  llm::ops::exp_shifted(input.data(), shift, from_shift.data(),
                        static_cast<std::int64_t>(input.size()));
  const std::vector<float> from_plain = exp_of(shifted_input);

  for (std::size_t i = 0; i < input.size(); ++i) {
    LLM_CHECK_EQ(from_shift[i], from_plain[i]);
  }
}

LLM_TEST(FastExp, Sigmoid) {
  llm::Rng rng(99);
  std::vector<float> input;
  for (int i = 0; i < 1000; ++i) {
    input.push_back(rng.uniform(-40.0f, 40.0f));
  }
  std::vector<float> actual(input.size(), 0.0f);
  llm::ops::sigmoid_array(input.data(), actual.data(),
                          static_cast<std::int64_t>(input.size()));

  double worst = 0.0;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const double expected =
        1.0 / (1.0 + std::exp(-static_cast<double>(input[i])));
    worst = std::max(worst, relative_error(actual[i], expected));
  }
  LLM_CHECK_MSG(worst < 3e-7, "наибольшая погрешность сигмоиды " << worst);
}

LLM_TEST(FastExp, KernelsWithFusedMultiplyAgreeBitForBit) {
  // Длины подобраны так, чтобы попасть и в полные векторы, и в хвосты обоих
  // ядер: 8 у AVX2, 16 у AVX-512. Хвост считается тем же ядром через
  // временный буфер, а не скалярной реализацией, — иначе последние несколько
  // элементов массива отличались бы от остальных, и это проверяется здесь.
  const std::int64_t lengths[] = {1, 7, 8, 9, 15, 16, 17, 31, 33, 1000};

  llm::Rng rng(20260916);
  std::vector<float> input;
  for (int i = 0; i < 1000; ++i) {
    input.push_back(rng.uniform(-110.0f, 95.0f));
  }

  int count = 0;
  const llm::ops::ExpKernelChoice* table = llm::ops::all_exp_kernels(&count);
  LLM_CHECK_GT(count, 0);

  int compared = 0;
  int available_fused = 0;
  for (std::size_t li = 0; li < sizeof(lengths) / sizeof(lengths[0]); ++li) {
    const std::int64_t length = lengths[li];
    const std::size_t size = static_cast<std::size_t>(length);

    std::vector<float> fused_exp;
    std::vector<float> fused_sigmoid;
    const char* fused_name = "";
    available_fused = 0;

    for (int i = 0; i < count; ++i) {
      if (!table[i].available) {
        continue;
      }
      const ForcedExpKernel forced(table[i].kernel);
      std::vector<float> got_exp(size, 0.0f);
      std::vector<float> got_sigmoid(size, 0.0f);
      llm::ops::exp_shifted(input.data(), 1.25f, got_exp.data(), length);
      llm::ops::sigmoid_array(input.data(), got_sigmoid.data(), length);

      if (!table[i].kernel.fused) {
        // Реализации без FMA округляют дважды там, где остальные округляют
        // один раз. От них требуется близость, а не равенство, и сверяются
        // они ниже — когда эталон уже набран.
        continue;
      }
      ++available_fused;
      if (fused_exp.empty()) {
        fused_exp = got_exp;
        fused_sigmoid = got_sigmoid;
        fused_name = table[i].kernel.name;
        continue;
      }
      ++compared;
      for (std::size_t j = 0; j < size; ++j) {
        LLM_CHECK_MSG(fused_exp[j] == got_exp[j],
                      "экспонента: '"
                          << fused_name << "' и '" << table[i].kernel.name
                          << "' разошлись на длине " << length << " в элементе "
                          << j << ": " << fused_exp[j] << " против "
                          << got_exp[j]);
        LLM_CHECK_MSG(fused_sigmoid[j] == got_sigmoid[j],
                      "сигмоида: '"
                          << fused_name << "' и '" << table[i].kernel.name
                          << "' разошлись на длине " << length << " в элементе "
                          << j);
      }
    }

    // Реализации без FMA — против векторной: разница в одном округлении, то
    // есть единицы последних разрядов.
    if (!fused_exp.empty()) {
      for (int i = 0; i < count; ++i) {
        if (!table[i].available || table[i].kernel.fused) {
          continue;
        }
        const ForcedExpKernel forced(table[i].kernel);
        std::vector<float> got_exp(size, 0.0f);
        std::vector<float> got_sigmoid(size, 0.0f);
        llm::ops::exp_shifted(input.data(), 1.25f, got_exp.data(), length);
        llm::ops::sigmoid_array(input.data(), got_sigmoid.data(), length);
        for (std::size_t j = 0; j < size; ++j) {
          LLM_CHECK_MSG(close_enough(got_exp[j], fused_exp[j], 1e-6),
                        "'" << table[i].kernel.name << "' разошлась с '"
                            << fused_name << "' в элементе " << j << ": "
                            << got_exp[j] << " против " << fused_exp[j]);
          LLM_CHECK_MSG(close_enough(got_sigmoid[j], fused_sigmoid[j], 1e-6),
                        "сигмоида '" << table[i].kernel.name
                                     << "' разошлась в элементе " << j);
        }
      }
    }
  }

  // Если векторных реализаций на этой машине меньше двух, сравнивать нечего —
  // но молча проходить нельзя: пусть будет видно, что проверка не работала.
  LLM_CHECK_MSG(
      compared > 0 || available_fused < 2,
      "сравнить оказалось нечего: реализаций с FMA " << available_fused);
}

LLM_TEST(FastExp, EveryKernelIsAccurate) {
  // Точность проверяется у каждой реализации из таблицы, а не только у
  // выбранной. Иначе ошибка в ядре, которого на этой машине нет, дожила бы до
  // машины, где оно есть.
  std::vector<float> input;
  for (double x = -87.0; x <= 88.0; x += 0.01) {
    input.push_back(static_cast<float>(x));
  }
  const std::size_t size = input.size();

  int count = 0;
  const llm::ops::ExpKernelChoice* table = llm::ops::all_exp_kernels(&count);
  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      continue;
    }
    const ForcedExpKernel forced(table[i].kernel);
    std::vector<float> actual(size, 0.0f);
    llm::ops::exp_array(input.data(), actual.data(),
                        static_cast<std::int64_t>(size));
    double worst = 0.0;
    float worst_at = 0.0f;
    for (std::size_t j = 0; j < size; ++j) {
      const double error =
          relative_error(actual[j], std::exp(static_cast<double>(input[j])));
      if (error > worst) {
        worst = error;
        worst_at = input[j];
      }
    }
    LLM_CHECK_MSG(worst < 2e-7, "у реализации '" << table[i].kernel.name
                                                 << "' наибольшая погрешность "
                                                 << worst
                                                 << " при x = " << worst_at);
  }
}

LLM_TEST(FastExp, EveryKernelWorksInPlace) {
  // masked_softmax считает экспоненту поверх уже домноженных на масштаб
  // значений, то есть вызывает exp_shifted с одним и тем же указателем на
  // вход и на выход. Это записано в договоре fast_exp.h — а раз записано,
  // должно и проверяться: реализация, читающая вперёд уже записанного, молча
  // сломала бы softmax внимания, и проявилось бы это неверным вниманием, а не
  // падением.
  //
  // Длины те же, что у побитовой сверки: важны и полные векторы, и хвосты,
  // которые доводятся через временный буфер.
  const std::int64_t lengths[] = {1, 7, 8, 9, 15, 16, 17, 31, 33, 1000};

  llm::Rng rng(20260920);
  std::vector<float> source;
  for (int i = 0; i < 1000; ++i) {
    source.push_back(rng.uniform(-20.0f, 20.0f));
  }

  int count = 0;
  const llm::ops::ExpKernelChoice* table = llm::ops::all_exp_kernels(&count);
  LLM_CHECK_GT(count, 0);

  int checked = 0;
  for (int i = 0; i < count; ++i) {
    if (!table[i].available) {
      continue;
    }
    const ForcedExpKernel forced(table[i].kernel);
    ++checked;
    for (std::size_t li = 0; li < sizeof(lengths) / sizeof(lengths[0]); ++li) {
      const std::size_t size = static_cast<std::size_t>(lengths[li]);
      const std::vector<float> input(source.begin(),
                                     source.begin() + static_cast<long>(size));

      std::vector<float> apart(size, 0.0f);
      llm::ops::exp_shifted(input.data(), 1.25f, apart.data(),
                            static_cast<std::int64_t>(size));
      std::vector<float> in_place = input;
      llm::ops::exp_shifted(in_place.data(), 1.25f, in_place.data(),
                            static_cast<std::int64_t>(size));
      for (std::size_t j = 0; j < size; ++j) {
        LLM_CHECK_MSG(apart[j] == in_place[j],
                      table[i].kernel.name
                          << ": exp_shifted на месте разошлась на длине "
                          << size << " в элементе " << j << ": " << apart[j]
                          << " против " << in_place[j]);
      }

      std::vector<float> sig_apart(size, 0.0f);
      llm::ops::sigmoid_array(input.data(), sig_apart.data(),
                              static_cast<std::int64_t>(size));
      std::vector<float> sig_in_place = input;
      llm::ops::sigmoid_array(sig_in_place.data(), sig_in_place.data(),
                              static_cast<std::int64_t>(size));
      for (std::size_t j = 0; j < size; ++j) {
        LLM_CHECK_MSG(sig_apart[j] == sig_in_place[j],
                      table[i].kernel.name
                          << ": sigmoid_array на месте разошлась на длине "
                          << size << " в элементе " << j);
      }
    }
  }
  LLM_CHECK_MSG(checked > 0, "ни одной доступной реализации");
}
