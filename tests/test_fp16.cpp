// Округление до половинной разрядности.
//
// Проверка нужна не сама по себе: на этом преобразовании строится вывод о
// том, сойдётся ли обучение в fp16. Неверное округление на краях дало бы
// неверный вывод, причём правдоподобный.

#include <cmath>
#include <cstdint>
#include <limits>

#include "core/fp16.h"
#include "testing.h"

namespace {

using llm::from_fp16;
using llm::round_to_fp16;
using llm::to_fp16;

}  // namespace

LLM_TEST(Fp16, ExactValuesSurviveUnchanged) {
  // Всё, что представимо в половинной разрядности, обязано вернуться собой.
  const float exact[] = {0.0f,   1.0f,  -1.0f, 0.5f,    2.0f,
                         -0.25f, 1024.0f, 65504.0f, -65504.0f, 6.103515625e-5f};
  for (std::size_t i = 0; i < sizeof(exact) / sizeof(exact[0]); ++i) {
    LLM_CHECK_MSG(round_to_fp16(exact[i]) == exact[i],
                  "значение " << exact[i] << " изменилось на "
                              << round_to_fp16(exact[i]));
  }
}

LLM_TEST(Fp16, RoundsToNearestEven) {
  // Шаг между соседними значениями около единицы равен 2^-10. Ровно половина
  // шага обязана округлиться к чётной мантиссе, а не всегда вверх.
  const float step = std::ldexp(1.0f, -10);
  const float half = step * 0.5f;
  // 1 + half: мантисса 0 (чётная) против 1 — остаётся 1.0.
  LLM_CHECK_MSG(round_to_fp16(1.0f + half) == 1.0f,
                "округление к чётному не сработало: "
                    << round_to_fp16(1.0f + half));
  // 1 + step + half: выбор между мантиссами 1 и 2, чётная — 2.
  LLM_CHECK_MSG(round_to_fp16(1.0f + step + half) == 1.0f + 2.0f * step,
                "округление к чётному не сработало вверх: "
                    << round_to_fp16(1.0f + step + half));
}

LLM_TEST(Fp16, OverflowBecomesInfinity) {
  // Наибольшее конечное — 65504; всё, что округляется выше, становится
  // бесконечностью. Это и есть верхняя граница, из-за которой в fp16
  // приходится масштабировать функцию потерь.
  LLM_CHECK(round_to_fp16(65505.0f) == 65504.0f);  // округляется вниз
  LLM_CHECK(std::isinf(round_to_fp16(70000.0f)));
  LLM_CHECK(round_to_fp16(70000.0f) > 0.0f);
  LLM_CHECK(std::isinf(round_to_fp16(-70000.0f)));
  LLM_CHECK(round_to_fp16(-70000.0f) < 0.0f);
}

LLM_TEST(Fp16, UnderflowLosesSmallValues) {
  // Нижняя граница — то, из-за чего в fp16 проваливаются градиенты.
  // Наименьшее субнормальное 2^-24; половина от него уходит в нуль.
  const float smallest = std::ldexp(1.0f, -24);
  LLM_CHECK_MSG(round_to_fp16(smallest) == smallest,
                "наименьшее субнормальное потерялось");
  LLM_CHECK_MSG(round_to_fp16(smallest * 0.4f) == 0.0f,
                "значение ниже половины наименьшего должно обнуляться");
  // Нормальные числа начинаются с 2^-14; между ним и 2^-24 — субнормальные,
  // у которых разрядов мантиссы тем меньше, чем меньше само число.
  const float sub = std::ldexp(1.0f, -20);
  LLM_CHECK_MSG(round_to_fp16(sub) == sub, "субнормальное 2^-20 потерялось");
}

LLM_TEST(Fp16, NanStaysNan) {
  // NaN обязан пройти насквозь: если обучение разошлось, это надо увидеть,
  // а не получить правдоподобную бесконечность.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  LLM_CHECK(std::isnan(round_to_fp16(nan)));
  LLM_CHECK(std::isinf(round_to_fp16(std::numeric_limits<float>::infinity())));
}

LLM_TEST(Fp16, RelativeErrorMatchesFormat) {
  // У половинной разрядности десять разрядов мантиссы, значит относительная
  // погрешность округления не больше 2^-11. Проверяется на всём рабочем
  // диапазоне, а не на отдельных точках.
  double worst = 0.0;
  for (int i = -14; i <= 15; ++i) {
    for (int j = 0; j < 997; ++j) {
      const float value =
          std::ldexp(1.0f + static_cast<float>(j) / 997.0f, i);
      const float rounded = round_to_fp16(value);
      const double error = std::fabs(rounded - value) / value;
      if (error > worst) {
        worst = error;
      }
    }
  }
  LLM_CHECK_MSG(worst <= std::ldexp(1.0, -11),
                "относительная погрешность " << worst << " больше 2^-11");
}

LLM_TEST(Fp16, RoundTripIsIdempotent) {
  // Повторное округление ничего не меняет: иначе значения «ползли» бы от
  // операции к операции, и замер сходимости показывал бы дрейф округления,
  // а не свойство арифметики.
  for (int i = 0; i < 5000; ++i) {
    const float value =
        std::ldexp(static_cast<float>((i * 7919 % 2003) - 1000) / 997.0f,
                   (i % 25) - 12);
    const float once = round_to_fp16(value);
    LLM_CHECK_MSG(round_to_fp16(once) == once,
                  "повторное округление изменило " << once);
  }
}

LLM_TEST(Fp16, SubnormalTiesRoundToEven) {
  // Округление к ближайшему чётному проверялось только в нормальной области.
  // В субнормальной та же развилка написана отдельно — там сдвиг переменный,
  // и половина считается как 1 << (shift - 1), — а проверена не была.
  // Проверено подменой: если заменить условие на «ничья всегда вверх», ни
  // одна проверка проекта этого не замечала.
  //
  // Шаг субнормальной сетки — 2^-24. Ровно посередине между узлами
  // округление обязано идти к чётному числу шагов.
  const float step = std::ldexp(1.0f, -24);
  LLM_CHECK_EQ(llm::to_fp16(std::ldexp(1.0f, -25)), static_cast<uint16_t>(0));
  LLM_CHECK_EQ(llm::to_fp16(1.5f * step), static_cast<uint16_t>(2));
  LLM_CHECK_EQ(llm::to_fp16(2.5f * step), static_cast<uint16_t>(2));
  LLM_CHECK_EQ(llm::to_fp16(3.5f * step), static_cast<uint16_t>(4));

  // И не ничьи по обе стороны от них — чтобы видеть, что дело именно в
  // ничьей, а не в общем сдвиге сетки.
  LLM_CHECK_EQ(llm::to_fp16(1.4f * step), static_cast<uint16_t>(1));
  LLM_CHECK_EQ(llm::to_fp16(1.6f * step), static_cast<uint16_t>(2));
  LLM_CHECK_EQ(llm::to_fp16(2.6f * step), static_cast<uint16_t>(3));

  // Знак на выбор чётного не влияет.
  LLM_CHECK_EQ(llm::to_fp16(-2.5f * step), static_cast<uint16_t>(0x8002u));
}
