// Восьмиразрядные форматы.
//
// Проверка нужна не сама по себе: на этом преобразовании будет строиться вывод
// о том, годится ли fp8 для весов. Неверное округление на краях дало бы
// неверный вывод, причём правдоподобный — именно так и выглядела бы модель,
// которую «убили восемь разрядов», если на самом деле у неё просто сломалась
// субнормальная область.

#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

#include "core/fp8.h"
#include "core/random.h"
#include "testing.h"

namespace {

using llm::from_fp8;
using llm::Fp8Format;
using llm::kFp8E4M3;
using llm::kFp8E5M2;
using llm::round_to_fp8;
using llm::to_fp8;

const Fp8Format kBoth[2] = {kFp8E4M3, kFp8E5M2};
const char* const kNames[2] = {"E4M3", "E5M2"};

}  // namespace

// Полный перебор всех 256 кодов. Он здесь возможен, а значит обязателен:
// выборочная проверка на «типичных» числах прошла бы и при сломанной
// субнормальной области, которая как раз и решает судьбу весов.
LLM_TEST(Fp8, RoundTripSurvivesEveryCode) {
  for (int which = 0; which < 2; ++which) {
    const Fp8Format& format = kBoth[which];
    int checked = 0;
    for (uint32_t code = 0; code < 256; ++code) {
      const float value = from_fp8(static_cast<uint8_t>(code), format);
      if (std::isnan(value)) {
        continue;
      }
      const uint8_t back = to_fp8(value, format);
      LLM_CHECK_MSG(back == static_cast<uint8_t>(code),
                    kNames[which] << ": код " << code << " вернулся как "
                                  << static_cast<int>(back) << " (значение "
                                  << value << ")");
      ++checked;
    }
    // Проверка самой проверки. У E4M3 NaN ровно два кода, по одному на знак;
    // у E5M2 верхнее поле порядка даёт по три NaN на знак плюс бесконечности,
    // а бесконечность круг проходит.
    const int expected = which == 0 ? 254 : 250;
    LLM_CHECK_MSG(checked == expected,
                  kNames[which] << ": прошли круг " << checked
                                << " кодов вместо " << expected);
  }
}

// Границы формата — те числа, из которых складывается вывод о применимости.
// Записаны здесь явно, чтобы ошибка в них была видна как падение, а не как
// «модель почему-то развалилась».
LLM_TEST(Fp8, BoundariesMatchTheSpecification) {
  LLM_CHECK_MSG(round_to_fp8(448.0f, kFp8E4M3) == 448.0f,
                "наибольшее конечное E4M3 должно быть 448");
  LLM_CHECK_MSG(round_to_fp8(1.0e30f, kFp8E4M3) == 448.0f,
                "переполнение E4M3 обязано насыщать, а не давать NaN");
  LLM_CHECK_MSG(round_to_fp8(std::ldexp(1.0f, -6), kFp8E4M3) ==
                    std::ldexp(1.0f, -6),
                "наименьшее нормальное E4M3 — 2^-6");
  LLM_CHECK_MSG(round_to_fp8(std::ldexp(1.0f, -9), kFp8E4M3) ==
                    std::ldexp(1.0f, -9),
                "наименьшее субнормальное E4M3 — 2^-9");
  LLM_CHECK_MSG(round_to_fp8(std::ldexp(1.0f, -12), kFp8E4M3) == 0.0f,
                "значение много меньше субнормального обязано дать нуль");

  LLM_CHECK_MSG(round_to_fp8(57344.0f, kFp8E5M2) == 57344.0f,
                "наибольшее конечное E5M2 должно быть 57344");
  LLM_CHECK_MSG(round_to_fp8(1.0e30f, kFp8E5M2) == 57344.0f,
                "переполнение E5M2 обязано насыщать");
  LLM_CHECK_MSG(round_to_fp8(std::ldexp(1.0f, -14), kFp8E5M2) ==
                    std::ldexp(1.0f, -14),
                "наименьшее нормальное E5M2 — 2^-14");
  LLM_CHECK_MSG(round_to_fp8(std::ldexp(1.0f, -16), kFp8E5M2) ==
                    std::ldexp(1.0f, -16),
                "наименьшее субнормальное E5M2 — 2^-16");
}

// Сколько всего различных значений даёт формат. Число маленькое, и это главное,
// что про восемь разрядов надо помнить: у E4M3 их около двухсот пятидесяти на
// всю числовую ось, у fp16 — шестьдесят три тысячи.
LLM_TEST(Fp8, DistinctValueCountIsSmall) {
  for (int which = 0; which < 2; ++which) {
    std::set<float> values;
    for (uint32_t code = 0; code < 256; ++code) {
      const float value = from_fp8(static_cast<uint8_t>(code), kBoth[which]);
      if (!std::isnan(value) && !std::isinf(value)) {
        values.insert(value);
      }
    }
    // Считаем явно, потому что первая написанная здесь цифра была неверной, и
    // тест это поймал.
    //
    // E4M3: код со всеми единицами порядка и мантиссы занят NaN, по одному на
    // знак, бесконечности нет. Конечных кодов 254, различных значений 253 —
    // плюс-нуль и минус-нуль в сравнении совпадают.
    //
    // E5M2: верхнее поле порядка даёт на знак одну бесконечность и три NaN, то
    // есть восемь кодов на оба знака. Конечных 248, различных 247.
    const std::size_t expected = which == 0 ? 253 : 247;
    LLM_CHECK_MSG(values.size() == expected,
                  kNames[which] << ": различных значений " << values.size()
                                << " вместо " << expected);
  }
}

// Граница погрешности, двусоставная — как и у fp16, и по той же причине.
// Первая половина и есть то число, из которого будет следовать вывод: шесть
// процентов у E4M3 против пяти сотых процента у fp16.
LLM_TEST(Fp8, ErrorStaysWithinLattice) {
  struct Expectation {
    Fp8Format format;
    double relative_bound;   // 2^-(mantissa_bits + 1)
    float min_normal;
  };
  const Expectation expectations[2] = {
      {kFp8E4M3, 1.0 / 16.0, std::ldexp(1.0f, -6)},
      {kFp8E5M2, 1.0 / 8.0, std::ldexp(1.0f, -14)},
  };

  for (int which = 0; which < 2; ++which) {
    const Expectation& e = expectations[which];
    llm::Rng rng(31 + which);
    double worst = 0.0;
    int samples = 0;
    for (int i = 0; i < 100000; ++i) {
      // Разброс взят широким нарочно: значения обязаны попадать в разные
      // бинады, иначе проверка увидела бы только одну ступеньку решётки.
      const float value =
          rng.normal() * std::ldexp(1.0f, rng.index(10) - 2);
      if (value == 0.0f || std::fabs(value) < e.min_normal ||
          std::fabs(value) > 400.0f) {
        continue;
      }
      ++samples;
      const float rounded = round_to_fp8(value, e.format);
      const double relative = std::fabs(
          (static_cast<double>(rounded) - value) / static_cast<double>(value));
      if (relative > worst) {
        worst = relative;
      }
    }
    LLM_CHECK_MSG(samples > 1000,
                  kNames[which] << ": выборка вышла пустой, " << samples
                                << " значений");
    LLM_CHECK_MSG(worst <= e.relative_bound,
                  kNames[which] << ": погрешность " << worst
                                << " больше половины шага решётки "
                                << e.relative_bound);
    // Иначе проверка ничего не ловит: при подмене формата на fp16 погрешность
    // была бы в сотни раз меньше.
    LLM_CHECK_MSG(worst > e.relative_bound / 2.0,
                  kNames[which] << ": погрешность " << worst
                                << " подозрительно мала");
  }
}

// Второе округление ничего не меняет. Нужно тому месту, где веса округляются
// повторно: иначе неподвижный вес медленно уползал бы сам по себе.
LLM_TEST(Fp8, RoundingIsIdempotent) {
  for (int which = 0; which < 2; ++which) {
    llm::Rng rng(7 + which);
    for (int i = 0; i < 20000; ++i) {
      const float value = rng.normal() * std::ldexp(1.0f, rng.index(12) - 6);
      const float once = round_to_fp8(value, kBoth[which]);
      const float twice = round_to_fp8(once, kBoth[which]);
      LLM_CHECK_MSG(once == twice, kNames[which]
                                       << ": повторное округление сдвинуло "
                                       << once << " до " << twice);
    }
  }
}
