// Округление тензора с масштабом.
//
// Главное, что здесь проверяется, — что масштабирование НЕ добавляет
// погрешности. Масштаб берётся степенью двойки именно для этого: иначе
// измеренная потеря качества относилась бы к арифметике вокруг формата, а не к
// самому формату, и вывод о применимости восьми разрядов был бы ни о чём.

#include <cmath>
#include <string>
#include <vector>

#include "core/quantize.h"
#include "core/random.h"
#include "testing.h"

namespace {

llm::Tensor tensor_from(const std::vector<float>& values) {
  llm::Tensor out = llm::Tensor::uninitialized(
      llm::Shape({static_cast<int64_t>(values.size())}));
  for (std::size_t i = 0; i < values.size(); ++i) {
    out.data()[i] = values[i];
  }
  return out;
}

double worst_relative_error(const std::vector<float>& before,
                            const llm::Tensor& after) {
  double worst = 0.0;
  for (std::size_t i = 0; i < before.size(); ++i) {
    if (before[i] == 0.0f) {
      continue;
    }
    const double relative =
        std::fabs((static_cast<double>(after.data()[i]) - before[i]) /
                  static_cast<double>(before[i]));
    if (relative > worst) {
      worst = relative;
    }
  }
  return worst;
}

}  // namespace

LLM_TEST(Quantize, NamesRoundTrip) {
  const char* const names[5] = {"fp32", "fp16", "e4m3", "e5m2", "e4m3-raw"};
  for (int i = 0; i < 5; ++i) {
    llm::Precision precision = llm::Precision::kFp32;
    LLM_CHECK_MSG(llm::parse_precision(names[i], &precision),
                  "имя " << names[i] << " не разобралось");
    LLM_CHECK_MSG(std::string(llm::precision_name(precision)) == names[i],
                  "имя " << names[i] << " вернулось как "
                         << llm::precision_name(precision));
  }
  llm::Precision unused = llm::Precision::kFp32;
  LLM_CHECK_MSG(!llm::parse_precision("fp6", &unused),
                "незнакомое имя должно отвергаться");
}

// Вот то, ради чего масштаб взят степенью двойки. Разброс весов этой модели —
// 0.02, то есть больше половины их лежит ниже наименьшего нормального значения
// E4M3 (2^-6). С масштабом погрешность обязана остаться на уровне решётки
// формата — 2^-4, то есть 6.25 процента, — а без масштаба она уходит в разы.
LLM_TEST(Quantize, ScaleKeepsErrorAtTheLatticeLevel) {
  llm::Rng rng(4242);
  std::vector<float> values;
  for (int i = 0; i < 20000; ++i) {
    values.push_back(rng.normal() * 0.02f);
  }

  llm::Tensor scaled = tensor_from(values);
  llm::quantize(&scaled, llm::Precision::kFp8E4M3);
  const double with_scale = worst_relative_error(values, scaled);

  llm::Tensor raw = tensor_from(values);
  llm::quantize(&raw, llm::Precision::kFp8E4M3NoScale);
  const double without_scale = worst_relative_error(values, raw);

  LLM_CHECK_MSG(with_scale <= 1.0 / 16.0 + 1e-6,
                "с масштабом погрешность " << with_scale
                                           << " больше решётки формата");
  // Без масштаба она обязана быть в разы хуже. Если бы оказалась такой же,
  // значит масштаб ни на что не влияет, и вся конструкция бессмысленна.
  LLM_CHECK_MSG(without_scale > 10.0 * with_scale,
                "без масштаба погрешность " << without_scale
                                            << " не хуже, чем с масштабом "
                                            << with_scale);
}

// Масштабирование точно. Проверяется так: значения, уже лежащие в решётке
// формата после деления на масштаб, обязаны пройти округление БЕЗ изменений.
// Если бы масштаб не был степенью двойки, умножение и деление на него вносили
// бы по округлению каждое.
LLM_TEST(Quantize, ScalingItselfIsExact) {
  // Набор строится из представимых значений формата, умноженных на степень
  // двойки: после деления на масштаб они попадут точно в узлы решётки.
  std::vector<float> values;
  for (int mantissa = 0; mantissa < 8; ++mantissa) {
    for (int exponent = -4; exponent <= 4; ++exponent) {
      const float unit = 1.0f + static_cast<float>(mantissa) / 8.0f;
      values.push_back(std::ldexp(unit, exponent) * std::ldexp(1.0f, -13));
      values.push_back(-std::ldexp(unit, exponent) * std::ldexp(1.0f, -13));
    }
  }

  llm::Tensor tensor = tensor_from(values);
  llm::quantize(&tensor, llm::Precision::kFp8E4M3);
  for (std::size_t i = 0; i < values.size(); ++i) {
    LLM_CHECK_MSG(tensor.data()[i] == values[i],
                  "значение " << values[i] << " изменилось на "
                              << tensor.data()[i]
                              << " — масштабирование неточно");
  }
}

// Показатель масштаба выбирается так, чтобы наибольший элемент не вылезал за
// верхний край формата, но и не оставлял запаса больше одной бинады.
LLM_TEST(Quantize, ScaleExponentFitsTheFormat) {
  for (int shift = -20; shift <= 10; ++shift) {
    std::vector<float> values;
    values.push_back(std::ldexp(0.7f, shift));
    values.push_back(-std::ldexp(0.3f, shift));
    const llm::Tensor tensor = tensor_from(values);

    const int exponent = llm::scale_exponent(tensor, llm::Precision::kFp8E4M3);
    const double largest = std::ldexp(0.7, shift);
    const double after = std::ldexp(largest, -exponent);
    LLM_CHECK_MSG(after <= 448.0, "после масштаба " << after
                                                    << " всё ещё больше 448");
    LLM_CHECK_MSG(after > 448.0 / 2.0,
                  "после масштаба " << after
                                    << " оставляет больше бинады запаса");
  }

  // Отдельно — случай, которого множитель 0.7 никогда не даёт: наибольшее
  // значение ровно на краю формата или на его степени двойки. Тогда отношение
  // к краю — точная степень двойки, frexp возвращает мантиссу ровно 1/2, и без
  // поправки показатель выходил на единицу больше нужного. Тензор с
  // наибольшим значением 448 масштабировался до 224, то есть целая бинада
  // пропадала впустую, а все малые элементы уезжали на бинаду ближе к
  // субнормальному дну формата.
  const float exact[] = {448.0f, 224.0f, 896.0f, std::ldexp(448.0f, -20)};
  for (std::size_t i = 0; i < sizeof(exact) / sizeof(exact[0]); ++i) {
    std::vector<float> values;
    values.push_back(exact[i]);
    values.push_back(-exact[i] / 4.0f);
    const llm::Tensor tensor = tensor_from(values);
    const int exponent = llm::scale_exponent(tensor, llm::Precision::kFp8E4M3);
    const double after = std::ldexp(static_cast<double>(exact[i]), -exponent);
    LLM_CHECK_MSG(after == 448.0,
                  "наибольшее " << exact[i] << " обязано лечь ровно на край "
                                << "формата, а легло на " << after);
  }

  // То же у E5M2, у которого другой край.
  const llm::Tensor wide = tensor_from({57344.0f, -1.0f});
  LLM_CHECK_EQ(llm::scale_exponent(wide, llm::Precision::kFp8E5M2), 0);
}

// Ноль и тензор из одних нулей не должны ломать выбор масштаба.
LLM_TEST(Quantize, ZerosSurvive) {
  std::vector<float> values(16, 0.0f);
  llm::Tensor tensor = tensor_from(values);
  llm::quantize(&tensor, llm::Precision::kFp8E4M3);
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    LLM_CHECK_MSG(tensor.data()[i] == 0.0f, "нуль перестал быть нулём");
  }
}
