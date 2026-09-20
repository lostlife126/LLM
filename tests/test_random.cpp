#include <cmath>
#include <cstdint>
#include <vector>

#include "core/random.h"
#include "testing.h"

LLM_TEST(Rng, IsDeterministic) {
  // Воспроизводимость — обязательное свойство: без неё тест на обучение
  // нельзя отладить.
  llm::Rng first(42);
  llm::Rng second(42);
  for (int i = 0; i < 100; ++i) {
    LLM_CHECK_EQ(first.next_bits(), second.next_bits());
  }
}

LLM_TEST(Rng, DifferentSeedsDiffer) {
  llm::Rng first(1);
  llm::Rng second(2);
  bool any_different = false;
  for (int i = 0; i < 10; ++i) {
    if (first.next_bits() != second.next_bits()) {
      any_different = true;
    }
  }
  LLM_CHECK(any_different);
}

LLM_TEST(Rng, UniformStaysInRange) {
  llm::Rng rng(7);
  for (int i = 0; i < 10000; ++i) {
    const double value = rng.uniform();
    LLM_CHECK(value >= 0.0);
    LLM_CHECK(value < 1.0);
  }
  for (int i = 0; i < 10000; ++i) {
    const float value = rng.uniform(-2.0f, 3.0f);
    LLM_CHECK(value >= -2.0f);
    LLM_CHECK(value <= 3.0f);
  }
}

LLM_TEST(Rng, UniformMeanAndVariance) {
  llm::Rng rng(123);
  const int count = 200000;
  double sum = 0.0;
  double sum_squares = 0.0;
  for (int i = 0; i < count; ++i) {
    const double value = rng.uniform();
    sum += value;
    sum_squares += value * value;
  }
  const double mean = sum / count;
  const double variance = sum_squares / count - mean * mean;
  LLM_EXPECT_NEAR(mean, 0.5, 0.01);
  LLM_EXPECT_NEAR(variance, 1.0 / 12.0, 0.01);
}

LLM_TEST(Rng, NormalMeanAndVariance) {
  llm::Rng rng(321);
  const int count = 200000;
  double sum = 0.0;
  double sum_squares = 0.0;
  double sum_fourth = 0.0;
  int far = 0;
  int repeats = 0;
  float previous = 0.0f;
  for (int i = 0; i < count; ++i) {
    const float value = rng.normal();
    sum += value;
    sum_squares += static_cast<double>(value) * value;
    sum_fourth += static_cast<double>(value) * value * value * value;
    if (std::fabs(value) > 2.0f) {
      ++far;
    }
    if (i > 0 && value == previous) {
      ++repeats;
    }
    previous = value;
  }
  const double mean = sum / count;
  const double variance = sum_squares / count - mean * mean;
  LLM_EXPECT_NEAR(mean, 0.0, 0.02);
  LLM_EXPECT_NEAR(variance, 1.0, 0.02);

  // Двух моментов мало: равномерное распределение на отрезке той же
  // дисперсии даёт ровно те же ноль и единицу. Форму задаёт четвёртый
  // момент: у нормального он равен трём, у равномерного — 1.8. Измерено на
  // четырёх зёрнах: 3.027, 2.997, 2.986, 2.994.
  const double kurtosis = sum_fourth / count / (variance * variance);
  LLM_EXPECT_NEAR(kurtosis, 3.0, 0.15);

  // И хвост: доля значений дальше двух сигм у нормального 0.0455, у
  // равномерного нулевая. Измерено: 0.0457, 0.0463, 0.0452, 0.0462.
  LLM_EXPECT_NEAR(static_cast<double>(far) / count, 0.0455, 0.005);

  // Полярный метод считает два числа сразу и второе придерживает до
  // следующего вызова. Если придержать не то — скажем, оставить в запасе x
  // вместо y, — каждое значение выйдет дважды подряд. Проверено мутацией:
  // среднее, дисперсия, эксцесс и хвост при этом остаются в норме все
  // четыре, а повторов становится 100000 из 199999 пар. То есть без этой
  // строки тест проходит на генераторе, который выдаёт половину чисел
  // дважды, — а это половина весов модели, совпадающих попарно.
  //
  // Порог не ноль, а сотня: два независимых значения могут совпасть и
  // законно, вероятность этого около 2^-24 на пару, то есть примерно одно
  // совпадение на десять таких прогонов. Измерено: ни одного на четырёх
  // зёрнах.
  LLM_CHECK_MSG(repeats < 100,
                "подряд идущих одинаковых значений " << repeats
                    << " — похоже, запасное число берётся не то");
}

LLM_TEST(Rng, NormalWithParameters) {
  llm::Rng rng(555);
  const int count = 200000;
  double sum = 0.0;
  double sum_squares = 0.0;
  for (int i = 0; i < count; ++i) {
    const double value = rng.normal(5.0f, 2.0f);
    sum += value;
    sum_squares += value * value;
  }
  const double mean = sum / count;
  const double variance = sum_squares / count - mean * mean;
  LLM_EXPECT_NEAR(mean, 5.0, 0.05);
  LLM_EXPECT_NEAR(variance, 4.0, 0.1);
}

LLM_TEST(Rng, IndexIsInRangeAndCoversAll) {
  llm::Rng rng(9);
  const std::uint64_t bound = 7;
  std::vector<int> counts(static_cast<std::size_t>(bound), 0);
  for (int i = 0; i < 70000; ++i) {
    const std::uint64_t value = rng.index(bound);
    LLM_CHECK(value < bound);
    counts[static_cast<std::size_t>(value)] += 1;
  }
  // Отбраковка остатка должна давать равномерность: при 10000 ожидаемых на
  // корзину отклонение в 10% было бы уже подозрительным.
  for (std::size_t i = 0; i < counts.size(); ++i) {
    LLM_CHECK_GT(counts[i], 9000);
    LLM_CHECK_LT(counts[i], 11000);
  }
}

LLM_TEST(Rng, IndexOfOneIsAlwaysZero) {
  llm::Rng rng(1);
  for (int i = 0; i < 100; ++i) {
    LLM_CHECK_EQ(rng.index(1), static_cast<std::uint64_t>(0));
  }
}

LLM_TEST(Rng, IndexRejectsZeroBound) {
  // Нулевая граница — деление на ноль в первой же строке index(). Проверка
  // должна работать и в Release: иначе получается SIGFPE без имени места.
  llm::Rng rng(5);
  LLM_EXPECT_THROWS(rng.index(0));
}
