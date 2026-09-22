// Годится ли пониженная разрядность для весов — по качеству, а не по скорости.
//
// Вопрос решающий: если восемь разрядов губят модель, то и мерить их скорость
// незачем. Приём тот же, которым снимался риск половинной разрядности: веса
// округляются В FLOAT, ни одного ядра писать не надо, и ответ получается за
// минуты вместо дней.
//
// Меряется на обученном чекпоинте, а не обучением с нуля, и это существенно.
// Веса в пониженной разрядности этот проект применяет только к инференсу —
// замер сходимости показал, что обучению они противопоказаны: относительный
// шаг решётки у fp16 составляет 4.9e-04, а отношение «шаг оптимизатора к весу»
// в конце обучения доходит до 2.6e-04, и шаг стирается округлением целиком. У
// восьми разрядов шаг решётки 6 процентов, то есть вопрос об обучении даже не
// ставится. Здесь проверяется другое: сколько качества теряет ГОТОВАЯ модель.
//
// Что округляется: ровно то, что округлил бы pack_half, — веса линейных слоёв
// и таблица эмбеддингов. Нормировки и смещения остаются, их суммарный объём
// ничтожен, а чувствительность к ним наибольшая.
//
// Запуск: precision_scope <чекпоинт.llmw> <словарь.bpe> <корпус.txt> [батчей]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "args.h"
#include "core/check.h"
#include "core/quantize.h"
#include "core/util.h"
#include "data/dataset.h"
#include "nn/model.h"
#include "read_file.h"
#include "serialize/checkpoint.h"
#include "tokenizer/bpe.h"
#include "train/trainer.h"

namespace {

// Веса, к которым применяется округление, и их отбор.
//
// Отбор по имени, а не по размеру: нормировки и смещения тоже «веса», но в
// пониженную разрядность их никто не переводит — объём у них ничтожный, а
// чувствительность наибольшая. Признак — наличие двух осей: у матриц их две,
// у нормировок одна.
bool is_matrix_weight(const llm::nn::NamedParameter& parameter) {
  return parameter.value->value().rank() == 2;
}

struct Result {
  llm::Precision precision;
  float loss;
  int64_t quantized_parameters;
  // Насколько сдвинулись сами веса. Без этой величины таблица потерь
  // недоказательна: пустое округление дало бы ровно те же потери, что и fp32,
  // и выглядело бы это как «восемь разрядов ничего не стоят».
  double weight_rms_change;
  // Наибольшая по модулю величина веса в тензорах и наибольший показатель
  // масштаба. По ним видно, попадают ли данные в диапазон формата или
  // упираются в его край.
  float largest_weight;
  int largest_scale_exponent;
};

}  // namespace

int main(int argc, char** argv) {
  const char* const usage =
      "<чекпоинт.llmw> <словарь.bpe> <корпус.txt> [батчей]";
  if (argc < 4) {
    std::fprintf(stderr, "использование: %s %s\n", argv[0], usage);
    return 1;
  }
  bench::expect_at_most(argc, 4, usage);
  const std::string checkpoint_path = argv[1];
  const std::string vocab_path = argv[2];
  const std::string corpus_path = argv[3];
  const int64_t batches =
      argc > 4 ? bench::parse_positive_int64(argv[4], "число батчей") : 32;

  const llm::Bpe tokenizer = llm::Bpe::load(vocab_path);
  const std::string text = bench::read_file(corpus_path);
  const llm::data::TokenDataset dataset =
      llm::data::TokenDataset::from_text(text, tokenizer, 0.1);

  const llm::nn::ModelConfig config =
      llm::serialize::read_config(checkpoint_path);

  // Батч оценки — один на всю таблицу и на проверку ниже.
  const int64_t kBatch = 8;

  // Если проверочная часть короче одного батча, evaluate честно вернёт нуль —
  // и вся таблица выйдет из нулей, а столбец «к fp32» из плюс-минус нулей.
  // Читается это как «восемь разрядов ничего не стоят», то есть ровно как
  // вывод, обратный тому, ради которого программа написана. Рядом уже стоит
  // такая же защита от пустого округления; эта — от пустой оценки.
  LLM_CHECK_MSG(
      dataset.validation_batch_count(kBatch, config.max_seq_len) > 0,
      "проверочная часть — "
          << dataset.validation_size() << " токенов, а на один батч нужно "
          << kBatch * config.max_seq_len << " (" << kBatch << " окон по "
          << config.max_seq_len << "): оценивать будет нечем");

  std::printf("модель: %s\n", config.to_string().c_str());
  std::printf("проверочная часть: %lld токенов, батчей на оценку %lld\n\n",
              static_cast<long long>(dataset.validation_size()),
              static_cast<long long>(batches));

  const llm::Precision order[5] = {
      llm::Precision::kFp32,           llm::Precision::kFp16,
      llm::Precision::kFp8E5M2,        llm::Precision::kFp8E4M3,
      llm::Precision::kFp8E4M3NoScale,
  };

  std::vector<Result> results;
  for (int which = 0; which < 5; ++which) {
    // Модель загружается заново на каждую разрядность: округление
    // необратимо, и переиспользование накапливало бы потери.
    llm::nn::Model model(config, 0);
    llm::serialize::load_checkpoint(checkpoint_path, &model);

    Result result;
    result.precision = order[which];
    result.quantized_parameters = 0;
    result.weight_rms_change = 0.0;
    result.largest_weight = 0.0f;
    result.largest_scale_exponent = -1000;

    double squared_change = 0.0;
    double squared_weight = 0.0;

    const std::vector<llm::nn::NamedParameter> parameters = model.parameters();
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      if (!is_matrix_weight(parameters[i])) {
        continue;
      }
      llm::Tensor& value = parameters[i].value->value();
      const int exponent = llm::scale_exponent(value, order[which]);
      if (exponent > result.largest_scale_exponent) {
        result.largest_scale_exponent = exponent;
      }

      // Копия до округления — только затем, чтобы измерить сдвиг.
      const llm::Tensor before = value.clone();
      llm::quantize(&value, order[which]);
      for (int64_t j = 0; j < value.numel(); ++j) {
        const double original = before.data()[j];
        const double change = static_cast<double>(value.data()[j]) - original;
        squared_change += change * change;
        squared_weight += original * original;
        const float magnitude = std::fabs(before.data()[j]);
        if (magnitude > result.largest_weight) {
          result.largest_weight = magnitude;
        }
      }
      result.quantized_parameters += value.numel();
    }
    result.weight_rms_change =
        squared_weight > 0.0 ? std::sqrt(squared_change / squared_weight) : 0.0;

    result.loss = llm::train::evaluate(&model, dataset, kBatch,
                                       config.max_seq_len, batches);
    results.push_back(result);
  }

  std::printf("%s %s %s %10s %s %s\n", llm::pad_utf8("разрядность", 10).c_str(),
              llm::pad_utf8_right("потери", 10).c_str(),
              llm::pad_utf8_right("перплекс.", 10).c_str(), "к fp32",
              llm::pad_utf8_right("сдвиг весов", 12).c_str(),
              llm::pad_utf8_right("масштаб", 8).c_str());
  std::printf(
      "-----------------------------------------------------------------\n");
  const double base = results[0].loss;
  for (std::size_t i = 0; i < results.size(); ++i) {
    std::printf(
        "%s %10.4f %10.1f %+10.4f %11.2f%% %8d\n",
        llm::pad_utf8(llm::precision_name(results[i].precision), 10).c_str(),
        results[i].loss, std::exp(static_cast<double>(results[i].loss)),
        results[i].loss - base, 100.0 * results[i].weight_rms_change,
        results[i].largest_scale_exponent);
  }
  std::printf(
      "\nсдвиг весов — среднеквадратичный, в долях от них самих. Если он "
      "нулевой,\n"
      "округление ничего не сделало, и строка потерь ничего не значит.\n"
      "масштаб — показатель степени двойки, на которую делится тензор перед\n"
      "округлением; для fp16 и e4m3-raw масштаб не применяется.\n");
  std::printf("наибольший вес в матрицах: %.4f\n", results[0].largest_weight);
  std::printf("\nокруглено весов: %lld\n",
              static_cast<long long>(results[0].quantized_parameters));
  std::printf(
      "строка e4m3-raw — тот же формат без масштаба на тензор. Она здесь\n"
      "затем, чтобы разница с e4m3 была видна числом: без масштаба диапазон\n"
      "порядка формата промахивается мимо данных, и вывод о применимости\n"
      "восьми разрядов получился бы неверным.\n");
  return 0;
}
