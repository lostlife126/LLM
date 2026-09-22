// Проверка хранения в половинной разрядности.
//
// Само преобразование проверяется в test_fp16: там краевые случаи,
// субнормальные числа, округление к чётному. Здесь — то, что добавляет
// half.h: тип, пакетные функции и обещание, что через них значение проходит
// туда и обратно без потерь, если оно уже представимо.

#include <cmath>
#include <cstdint>
#include <vector>

#include "core/fp16.h"
#include "core/half.h"
#include "core/random.h"
#include "testing.h"

namespace {

using llm::float_from_half;
using llm::Half;
using llm::half_from_float;

}  // namespace

// Все 65536 значений половинной разрядности разом. Полный перебор здесь
// возможен и потому обязателен: выборочная проверка на «типичных» числах
// прошла бы и при сломанной субнормальной области.
LLM_TEST(Half, RoundTripSurvivesEveryValue) {
  int64_t checked = 0;
  for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
    Half value;
    value.bits = static_cast<uint16_t>(bits);
    const float expanded = float_from_half(value);
    if (std::isnan(expanded)) {
      continue;  // NaN не сравнивается сам с собой, и разряды его не оговорены
    }
    const Half back = half_from_float(expanded);
    LLM_CHECK_MSG(back.bits == static_cast<uint16_t>(bits),
                  "значение " << bits << " вернулось как " << back.bits);
    ++checked;
  }
  // Проверка самой проверки: NaN в половинной разрядности ровно 2046 штук, по
  // 1023 на знак. Если бы цикл молча пропускал что-то ещё, полный перебор
  // перестал бы быть полным, и об этом никто бы не узнал.
  LLM_CHECK_MSG(checked == 65536 - 2046,
                "прошли круг " << checked << " значений вместо 63490");
}

// Пакетное развёртывание считается не так, как from_fp16: там разбор случаев и
// цикл нормализации, здесь арифметика без ветвей, которую компилятор
// раскладывает по векторам. Значит нужна проверка, что это одна и та же
// функция, и проверка полным перебором — выборочная прошла бы и при сломанной
// субнормальной области, а именно ради малых значений всё и затевалось.
LLM_TEST(Half, BulkExpansionEqualsReferenceOnEveryValue) {
  std::vector<Half> all(0x10000u);
  for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
    all[bits].bits = static_cast<uint16_t>(bits);
  }
  std::vector<float> expanded(all.size());
  llm::half_to_floats(all.data(), expanded.data(),
                      static_cast<int64_t>(all.size()));

  for (uint32_t bits = 0; bits < 0x10000u; ++bits) {
    const float reference = llm::from_fp16(static_cast<uint16_t>(bits));
    if (std::isnan(reference)) {
      // NaN не сравнивается сам с собой; сверяются разряды, потому что и
      // старшая часть мантиссы обязана дойти без изменений.
      LLM_CHECK_MSG(std::isnan(expanded[bits]),
                    "значение " << bits << " перестало быть NaN");
      LLM_CHECK_MSG(
          llm::float_bits(expanded[bits]) == llm::float_bits(reference),
          "разряды NaN у значения " << bits << " разошлись");
      continue;
    }
    LLM_CHECK_MSG(expanded[bits] == reference,
                  "значение " << bits << " развернулось в " << expanded[bits]
                              << " вместо " << reference);
  }
}

LLM_TEST(Half, BulkMatchesScalar) {
  std::vector<float> source;
  for (int i = -2000; i < 2000; ++i) {
    source.push_back(static_cast<float>(i) * 0.013f);
  }
  // Хвост, ради которого всё и затевалось: значения около границы
  // субнормальной области, под ней и за верхним краем.
  source.push_back(6.0e-08f);
  source.push_back(6.0e-05f);
  source.push_back(1.0e-09f);
  source.push_back(-6.0e-05f);
  source.push_back(70000.0f);

  std::vector<Half> packed(source.size());
  llm::floats_to_half(source.data(), packed.data(),
                      static_cast<int64_t>(source.size()));
  std::vector<float> expanded(source.size());
  llm::half_to_floats(packed.data(), expanded.data(),
                      static_cast<int64_t>(source.size()));

  for (std::size_t i = 0; i < source.size(); ++i) {
    LLM_CHECK_MSG(
        packed[i].bits == half_from_float(source[i]).bits,
        "пакетное упаковало " << source[i] << " иначе, чем поштучное");
    LLM_CHECK_MSG(expanded[i] == float_from_half(packed[i]),
                  "пакетное развернуло " << source[i] << " иначе");
  }
}

// Второе округление ничего не меняет. Свойство неочевидное — округление к
// ближайшему чётному вообще-то умеет «доехать» до соседа, — но здесь значение
// уже лежит в решётке половинной разрядности, и ближайший к нему узел он сам.
// Нужно это тому месту, где веса округляются на каждом шаге оптимизатора:
// иначе неподвижный вес медленно уползал бы сам по себе.
LLM_TEST(Half, RoundingIsIdempotent) {
  std::vector<float> source;
  for (int i = 1; i < 500; ++i) {
    source.push_back(1.0f / static_cast<float>(i));
    source.push_back(static_cast<float>(i) * 3.7f);
    source.push_back(static_cast<float>(i) * 1.0e-6f);
  }
  for (std::size_t i = 0; i < source.size(); ++i) {
    const float once = float_from_half(half_from_float(source[i]));
    const float twice = float_from_half(half_from_float(once));
    LLM_CHECK_MSG(once == twice,
                  "повторное округление сдвинуло " << once << " до " << twice);
  }
}

// Граница погрешности. Она двусоставная, и это не педантизм: весь вывод о
// том, что оптимизатору нужны эталонные веса в fp32, опирается на первую
// половину, а вторая объясняет, почему малые градиенты не пропадают.
//
//   |v| >= 2^-14 — нормальная область, шаг решётки относительный, погрешность
//                  не больше 2^-11 от значения;
//   |v| <  2^-14 — субнормальная, шаг постоянный 2^-24, погрешность не больше
//                  2^-25 по модулю, а относительная растёт до половины.
//
// Первая написанная здесь проверка требовала 2^-11 от всего подряд и упала на
// выборке: примерно одно значение из двух тысяч попадает под 2^-14, и там
// относительная погрешность доходила до 0.015.
LLM_TEST(Half, ErrorStaysWithinLattice) {
  llm::Rng rng(7);
  double worst_relative = 0.0;
  double worst_absolute = 0.0;
  int normal_samples = 0;
  int subnormal_samples = 0;

  for (int i = 0; i < 200000; ++i) {
    const float value = rng.normal() * 0.1f;
    if (value == 0.0f) {
      continue;
    }
    const float rounded = float_from_half(half_from_float(value));
    const double difference = static_cast<double>(rounded) - value;

    if (std::fabs(value) >= 6.103515625e-05f) {
      ++normal_samples;
      const double relative = std::fabs(difference / value);
      if (relative > worst_relative) {
        worst_relative = relative;
      }
    } else {
      ++subnormal_samples;
      if (std::fabs(difference) > worst_absolute) {
        worst_absolute = std::fabs(difference);
      }
    }
  }

  LLM_CHECK_MSG(normal_samples > 0 && subnormal_samples > 0,
                "выборка не задела обе области: нормальных "
                    << normal_samples << ", субнормальных "
                    << subnormal_samples);

  LLM_CHECK_MSG(worst_relative <= 1.0 / 2048.0, "относительная погрешность "
                                                    << worst_relative
                                                    << " больше половины шага");
  // Иначе проверка ничего не ловит: при округлении в сторону нуля погрешность
  // была бы вдвое больше, а при подмене half на float — нулевой.
  LLM_CHECK_MSG(worst_relative > 1.0 / 4096.0, "относительная погрешность "
                                                   << worst_relative
                                                   << " подозрительно мала");

  LLM_CHECK_MSG(worst_absolute <= 1.0 / 33554432.0,
                "абсолютная погрешность в субнормальной области "
                    << worst_absolute << " больше 2^-25");
}
