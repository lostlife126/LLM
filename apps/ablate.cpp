// Архитектурная лаборатория: измерение вместо пересказа.
//
// Каждый вариант отличается от базового ровно одним решением, обучается с
// одинаковым бюджетом и сравнивается по перплексии на одной и той же
// проверочной выборке.
//
// Каждый вариант прогоняется несколькими зёрнами. Без этого таблица абляций
// бессмысленна: непонятно, что считать различием, а что разбросом от
// случайной инициализации. Разброс базового варианта — и есть та планка, выше
// которой различие можно обсуждать.
//
// Запуск: ablate <корпус.txt> <словарь.bpe> [шагов] [зёрен]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/check.h"
#include "core/util.h"
#include "args.h"
#include "read_file.h"
#include "data/dataset.h"
#include "nn/model.h"
#include "tokenizer/bpe.h"
#include "train/trainer.h"

namespace {

using llm::nn::FfnKind;
using llm::nn::ModelConfig;
using llm::nn::NormKind;
using llm::nn::PositionKind;

struct Variant {
  std::string name;
  std::string question;  // на какой вопрос отвечает этот прогон
  ModelConfig config;
};

std::vector<Variant> build_variants(int64_t vocab_size) {
  ModelConfig base = ModelConfig::ablation();
  base.vocab_size = vocab_size;

  std::vector<Variant> variants;

  Variant baseline;
  baseline.name = "базовый (Llama-style)";
  baseline.question = "RMSNorm, pre-norm, RoPE, SwiGLU, GQA 4/2, связанные";
  baseline.config = base;
  variants.push_back(baseline);

  Variant layer_norm;
  layer_norm.name = "LayerNorm";
  layer_norm.question = "даёт ли что-нибудь вычитание среднего";
  layer_norm.config = base;
  layer_norm.config.norm = NormKind::kLayerNorm;
  variants.push_back(layer_norm);

  Variant post;
  post.name = "пост-нормировка";
  post.question = "мешает ли нормировка на пути остатка";
  post.config = base;
  post.config.post_norm = true;
  variants.push_back(post);

  Variant gelu;
  gelu.name = "FFN с GELU";
  gelu.question = "стоит ли вентиль SwiGLU лишней матрицы";
  gelu.config = base;
  gelu.config.ffn = FfnKind::kGeluMlp;
  gelu.config.ffn_hidden = base.ffn_hidden_matching(FfnKind::kGeluMlp);
  variants.push_back(gelu);

  Variant learned;
  learned.name = "обучаемые позиции";
  learned.question = "выигрывает ли RoPE у таблицы позиций";
  learned.config = base;
  learned.config.position = PositionKind::kLearned;
  variants.push_back(learned);

  Variant none;
  none.name = "без позиций";
  none.question = "сколько порядка даёт одна каузальная маска";
  none.config = base;
  none.config.position = PositionKind::kNone;
  variants.push_back(none);

  Variant mha;
  mha.name = "MHA (голов ключей 4)";
  mha.question = "сколько стоит экономия на головах ключей";
  mha.config = base;
  mha.config.n_kv_heads = base.n_heads;
  variants.push_back(mha);

  Variant mqa;
  mqa.name = "MQA (голова ключей одна)";
  mqa.question = "докуда можно ужимать KV-кэш";
  mqa.config = base;
  mqa.config.n_kv_heads = 1;
  variants.push_back(mqa);

  Variant untied;
  untied.name = "несвязанные эмбеддинги";
  untied.question =
      "что даёт отдельная выходная матрица (НЕ выровнено по параметрам)";
  untied.config = base;
  untied.config.tie_embeddings = false;
  variants.push_back(untied);

  Variant qk;
  qk.name = "QK-норма";
  qk.question = "мешает ли рост логитов внимания";
  qk.config = base;
  qk.config.qk_norm = true;
  variants.push_back(qk);

  Variant z;
  z.name = "Z-loss";
  z.question = "мешает ли дрейф логитов от нуля";
  z.config = base;
  z.config.z_loss_coef = 1e-4f;
  variants.push_back(z);

  // Коэффициент 1e-4 взят из PaLM, и на нашем масштабе он почти ничего не
  // делает: логиты и так не уезжают. Вариант с коэффициентом в сто раз больше
  // отличает «приём не работает» от «доза мала»: если и он ничего не меняет,
  // дело не в дозе.
  Variant z_strong;
  z_strong.name = "Z-loss x100";
  z_strong.question = "дело в приёме или в величине коэффициента";
  z_strong.config = base;
  z_strong.config.z_loss_coef = 1e-2f;
  variants.push_back(z_strong);

  Variant both;
  both.name = "QK-норма + Z-loss";
  both.question = "складываются ли два приёма";
  both.config = base;
  both.config.qk_norm = true;
  both.config.z_loss_coef = 1e-4f;
  variants.push_back(both);

  // Развязывание эмбеддингов добавляет vocab * d_model весов, поэтому строка
  // выше отвечает на вопрос «помогает ли +24% параметров», а не «помогает ли
  // развязывание». Честный ответ даёт этот вариант: те же лишние веса, но
  // потраченные на ширину FFN при связанных эмбеддингах.
  Variant wider;
  wider.name = "связанные + шире FFN";
  wider.question = "те же +24% параметров, но в FFN, а не в выходной матрице";
  wider.config = base;
  wider.config.ffn_hidden =
      base.ffn_hidden +
      (base.vocab_size * base.d_model) / (3 * base.d_model * base.n_layers);
  variants.push_back(wider);

  return variants;
}

struct Outcome {
  std::string name;
  std::string question;
  int64_t parameters = 0;
  std::vector<float> validation_loss;

  float mean() const {
    if (validation_loss.empty()) {
      return 0.0f;  // как в mean_of ниже: делить на ноль не на что
    }
    double total = 0.0;
    for (std::size_t i = 0; i < validation_loss.size(); ++i) {
      total += validation_loss[i];
    }
    return static_cast<float>(total /
                              static_cast<double>(validation_loss.size()));
  }
  float spread() const {
    if (validation_loss.size() < 2) {
      return 0.0f;
    }
    const float low =
        *std::min_element(validation_loss.begin(), validation_loss.end());
    const float high =
        *std::max_element(validation_loss.begin(), validation_loss.end());
    return high - low;
  }

  // Разность с базовым вариантом, взятая по каждому зерну отдельно.
  //
  // Сравнение здесь парное: у одинаковых зёрен совпадают и начальная
  // инициализация, и порядок данных, то есть различаются прогоны ровно
  // проверяемым решением. Поэтому разброс средних по зёрнам — неправильная
  // линейка: он меряет, насколько зёрна отличаются друг от друга, а не
  // насколько неустойчив эффект. Правильная линейка — разброс самих
  // разностей: если по каждому зерну вариант проигрывает базовому примерно
  // одинаково, эффект есть, даже когда он меньше расстояния между зёрнами.
  std::vector<float> paired_delta(const Outcome& baseline) const {
    std::vector<float> delta;
    const std::size_t count =
        validation_loss.size() < baseline.validation_loss.size()
            ? validation_loss.size()
            : baseline.validation_loss.size();
    for (std::size_t i = 0; i < count; ++i) {
      delta.push_back(validation_loss[i] - baseline.validation_loss[i]);
    }
    return delta;
  }
};

float mean_of(const std::vector<float>& values) {
  if (values.empty()) {
    return 0.0f;
  }
  double total = 0.0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    total += values[i];
  }
  return static_cast<float>(total / static_cast<double>(values.size()));
}

float spread_of(const std::vector<float>& values) {
  if (values.size() < 2) {
    return 0.0f;
  }
  return *std::max_element(values.begin(), values.end()) -
         *std::min_element(values.begin(), values.end());
}

// Совпал ли знак разности у всех зёрен. Разошедшийся знак означает, что
// вариант на одном зерне лучше базового, а на другом хуже, — то есть эффекта
// нет независимо от того, чему равно среднее.
bool sign_agrees(const std::vector<float>& delta) {
  if (delta.size() < 2) {
    return false;
  }
  for (std::size_t i = 1; i < delta.size(); ++i) {
    if (delta[i] * delta[0] <= 0.0f) {
      return false;
    }
  }
  return true;
}


}  // namespace

int main(int argc, char** argv) {
  // Фильтр вариантов раньше в подсказке не упоминался, хотя читается ниже.
  const char* const usage =
      "<корпус.txt> <словарь.bpe> [шагов] [зёрен] [фильтр,через,запятую]";
  if (argc < 3) {
    std::fprintf(stderr, "использование: %s %s\n", argv[0], usage);
    return 1;
  }
  bench::expect_at_most(argc, 5, usage);
  const std::string corpus_path = argv[1];
  const std::string vocab_path = argv[2];
  const int64_t steps =
      argc > 3 ? bench::parse_positive_int64(argv[3], "число шагов") : 1000;
  const int seeds =
      argc > 4 ? static_cast<int>(bench::parse_positive_int64(argv[4], "число зёрен"))
               : 2;
  // Необязательный фильтр: список подстрок через запятую. Позволяет догнать
  // несколько вариантов, не пересчитывая всю таблицу.
  const std::string filter = argc > 5 ? argv[5] : std::string();
  std::vector<std::string> wanted;
  if (!filter.empty()) {
    std::size_t begin = 0;
    while (begin <= filter.size()) {
      const std::size_t comma = filter.find(',', begin);
      const std::size_t end =
          comma == std::string::npos ? filter.size() : comma;
      wanted.push_back(filter.substr(begin, end - begin));
      if (comma == std::string::npos) {
        break;
      }
      begin = comma + 1;
    }
  }

  const llm::Bpe tokenizer = llm::Bpe::load(vocab_path);
  const llm::data::TokenDataset dataset = llm::data::TokenDataset::from_text(
      bench::read_file(corpus_path), tokenizer, 0.1);

  const std::vector<Variant> variants = build_variants(tokenizer.vocab_size());

  // Батч один на все прогоны и на проверку ниже.
  const int64_t kBatch = 16;

  // Вся таблица — сравнение проверочных потерь между вариантами. Если
  // проверочная часть короче одного батча, тренер не станет её считать, и
  // final_validation_loss у каждого варианта останется нулём: таблица выйдет
  // из нулей, парные разности из нулей, а столбец «знак» скажет «нет» про
  // всё сразу. Выглядело бы это как честный результат «ни одно решение
  // ничего не меняет».
  LLM_CHECK_MSG(
      dataset.validation_batch_count(kBatch, variants[0].config.max_seq_len) >
          0,
      "проверочная часть корпуса — "
          << dataset.validation_size() << " токенов, а на один батч нужно "
          << kBatch * variants[0].config.max_seq_len << " (" << kBatch
          << " окон по " << variants[0].config.max_seq_len
          << "): сравнивать варианты будет не по чему");

  std::printf("абляции: %zu вариантов по %d зёрен, %lld шагов каждый\n",
              variants.size(), seeds, static_cast<long long>(steps));
  // Сброс буфера после шапки: при выводе в файл она иначе не появится до
  // первого готового прогона, и несколько минут кажется, что ничего не идёт.
  std::printf("обучающих токенов %lld, проверочных %lld\n\n",
              static_cast<long long>(dataset.train_size()),
              static_cast<long long>(dataset.validation_size()));
  std::fflush(stdout);

  std::vector<Outcome> outcomes;
  for (std::size_t v = 0; v < variants.size(); ++v) {
    // Базовый вариант считается всегда: без него колонка разностей меряла бы
    // расстояние до случайного варианта, попавшего в выборку первым.
    if (!wanted.empty() && v != 0) {
      bool matched = false;
      for (std::size_t w = 0; w < wanted.size(); ++w) {
        if (variants[v].name.find(wanted[w]) != std::string::npos) {
          matched = true;
        }
      }
      if (!matched) {
        continue;
      }
    }
    Outcome outcome;
    outcome.name = variants[v].name;
    outcome.question = variants[v].question;
    outcome.parameters = variants[v].config.parameter_count();

    for (int seed = 0; seed < seeds; ++seed) {
      llm::nn::Model model(variants[v].config,
                           static_cast<uint64_t>(1000 + seed));

      llm::train::TrainConfig train_config;
      train_config.steps = steps;
      train_config.batch_size = kBatch;
      train_config.warmup_steps = steps / 20 + 1;
      // Порядок данных одинаков у всех прогонов: меняется только начальная
      // инициализация, поэтому разброс отражает именно её.
      train_config.seed = 777;
      train_config.log_every = 0;
      train_config.eval_batches = 16;
      train_config.verbose = false;

      const llm::train::TrainReport report =
          llm::train::train(&model, dataset, train_config);
      outcome.validation_loss.push_back(report.final_validation_loss);

      std::printf("  %s зерно %d: потери %.4f  (%.1f мин)\n",
                  llm::pad_utf8(variants[v].name, 26).c_str(), seed,
                  report.final_validation_loss, report.seconds / 60.0);
      std::fflush(stdout);
    }
    outcomes.push_back(outcome);
  }

  // Базовый вариант считается всегда, поэтому пустым список не бывает — и
  // проверка на пустоту ничего не ловила. Ловить надо другое: фильтр,
  // не выбравший ничего сверх базового. Это опечатка в подстроке, и таблица
  // из одной строки про неё не скажет ничего — колонка разностей будет
  // сравнивать базовый вариант сам с собой.
  LLM_CHECK_MSG(filter.empty() || outcomes.size() > 1,
                "фильтр '" << filter
                           << "' не выбрал ни одного варианта сверх базового");
  const Outcome& baseline = outcomes[0];

  // Итоговая таблица, отсортированная по качеству.
  std::vector<Outcome> sorted = outcomes;
  std::sort(
      sorted.begin(), sorted.end(),
      [](const Outcome& a, const Outcome& b) { return a.mean() < b.mean(); });

  std::printf("\n%s %s %s %s %s %s %s\n", llm::pad_utf8("вариант", 26).c_str(),
              llm::pad_utf8_right("параметров", 10).c_str(),
              llm::pad_utf8_right("потери", 8).c_str(),
              llm::pad_utf8_right("перплексия", 10).c_str(),
              llm::pad_utf8_right("парный Δ", 8).c_str(),
              llm::pad_utf8_right("разбр Δ", 8).c_str(),
              llm::pad_utf8_right("знак", 6).c_str());
  std::printf(
      "----------------------------------------------------------------------"
      "--------------\n");
  for (std::size_t i = 0; i < sorted.size(); ++i) {
    const std::vector<float> delta = sorted[i].paired_delta(baseline);
    const bool is_baseline = sorted[i].name == baseline.name;
    const std::string verdict =
        is_baseline ? "-" : (sign_agrees(delta) ? "да" : "нет");
    std::printf("%s %10lld %8.4f %10.1f %+8.4f %8.4f %s\n",
                llm::pad_utf8(sorted[i].name, 26).c_str(),
                static_cast<long long>(sorted[i].parameters), sorted[i].mean(),
                std::exp(sorted[i].mean()), mean_of(delta), spread_of(delta),
                llm::pad_utf8_right(verdict, 6).c_str());
  }

  std::printf("\nчто проверял каждый вариант:\n");
  for (std::size_t i = 0; i < outcomes.size(); ++i) {
    std::printf("  %s %s\n", llm::pad_utf8(outcomes[i].name, 26).c_str(),
                outcomes[i].question.c_str());
  }

  // Две линейки, и путать их нельзя.
  //
  // Разброс базового варианта по зёрнам говорит, насколько вообще шумит
  // обучение. Он велик: одно зерно обгоняет другое сильнее, чем половина
  // проверяемых решений что-либо меняет.
  //
  // Но сравниваются варианты не по средним, а по зёрнам попарно, и парная
  // разность шумит на порядок меньше — из неё общий для обоих прогонов шум
  // вычитается. Поэтому решает столбец «разбр Δ» вместе со столбцом «знак»:
  // если знак разошёлся, эффекта нет, каким бы ни было среднее.
  std::printf(
      "\nразброс базового варианта по зёрнам: %.4f — столько шумит само "
      "обучение\n",
      baseline.spread());
  std::printf(
      "сравнение парное (зёрна и порядок данных совпадают), поэтому решает не "
      "он,\nа разброс парных разностей и совпадение их знака\n");
  return 0;
}
