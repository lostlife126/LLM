// Замер шага обучения модели.
//
// Цифра нужна для честной оценки времени обучения: пропускная способность
// GEMM — это верхняя граница, а реальный шаг тратит заметную часть времени на
// нормировки, softmax и мелкие матрицы внутри внимания.
//
// Шаг оптимизатора меряется отдельной строкой, и это не педантизм. Раньше
// здесь считались только прямой и обратный проходы, а обрезка нормы и AdamW —
// два полных прохода по всем параметрам — не считались вовсе. Из этой цифры
// выводилось «сколько займут 2000 шагов», и вывод получался оптимистичным на
// то, чего в замере не было.
//
// Запуск: ./bench_model [пресет] [батч]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "args.h"
#include "measure.h"

#include "core/cpu.h"
#include "core/random.h"
#include "core/thread_pool.h"
#include "nn/model.h"
#include "train/optimizer.h"

namespace {

std::vector<int32_t> random_ids(int64_t count, int64_t vocab) {
  llm::Rng rng(1);
  std::vector<int32_t> ids;
  for (int64_t i = 0; i < count; ++i) {
    ids.push_back(
        static_cast<int32_t>(rng.index(static_cast<uint64_t>(vocab))));
  }
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  // Приложение читает ровно два аргумента. Длина окна берётся у пресета и
  // аргументом не задаётся — в scripts/pi_check.sh третьим стояло «128», и
  // оно молча отбрасывалось.
  bench::expect_at_most(argc, 2, "[пресет] [батч]");
  const std::string preset = argc > 1 ? argv[1] : "nano";
  const int64_t batch = argc > 2 ? bench::parse_positive_int64(argv[2], "батч") : 16;

  const llm::nn::ModelConfig config = llm::nn::ModelConfig::by_name(preset);
  llm::nn::Model model(config, 1234);
  const int64_t seq = config.max_seq_len;
  const std::vector<int32_t> ids = random_ids(batch * seq, config.vocab_size);

  std::printf("процессор: %s, ядер %d, потоков в работе %d\n",
              llm::cpu_features().to_string().c_str(), llm::detect_core_count(),
              llm::parallel_width());
  std::printf("пресет %s: %s\n", preset.c_str(), config.to_string().c_str());
  std::printf("батч %lld x %lld = %lld токенов на шаг\n",
              static_cast<long long>(batch), static_cast<long long>(seq),
              static_cast<long long>(batch * seq));

  // Один прогон вне замера: прогрев кэшей и аллокатора.
  model.loss(ids, batch, seq);

  {
    llm::autograd::NoGradGuard no_grad;
    double best = 0.0;
    best = bench::best_seconds([&]() { model.loss(ids, batch, seq); }, 1.0);
    std::printf("только прямой проход: %.3f с\n", best);
  }

  double step_seconds = bench::best_seconds(
      [&]() {
        llm::autograd::Var loss = model.loss(ids, batch, seq);
        loss.backward();
        const std::vector<llm::nn::NamedParameter> parameters =
            model.parameters();
        for (std::size_t i = 0; i < parameters.size(); ++i) {
          parameters[i].value->zero_grad();
        }
      },
      2.0);
  std::printf("вперёд + назад: %.3f с\n", step_seconds);

  // Шаг оптимизатора: обрезка нормы и сам AdamW. Градиенты для него нужны
  // настоящие, поэтому перед серией делается один проход вперёд-назад, а
  // замеряется только то, что идёт после.
  //
  // Одна оговорка к этому числу. Градиенты между вызовами не обновляются, а
  // clip_grad_norm масштабирует их только когда норма превысила предел:
  // первый вызов приводит её ровно к пределу, и все следующие проход
  // масштабирования пропускают. То есть замеряется обрезка без
  // масштабирования, и настоящий шаг на ранних итерациях обучения, где норма
  // действительно велика, стоит на один проход по параметрам дороже.
  //
  // Восстанавливать градиенты перед каждым вызовом смысла нет: это сам по
  // себе проход по всем параметрам, и он попал бы в замер вместо
  // пропущенного. А величина недоучёта мала: весь шаг оптимизатора у nano —
  // 0.002 с из 0.050 с, то есть 4% шага, и пропущенный проход — доля от
  // этих четырёх процентов.
  double optimizer_seconds = 0.0;
  {
    llm::train::AdamWConfig adam_config;
    llm::train::AdamW optimizer(model.trainable_parameters(), adam_config);
    llm::autograd::Var loss = model.loss(ids, batch, seq);
    loss.backward();
    optimizer_seconds = bench::best_seconds(
        [&]() {
          optimizer.clip_grad_norm(1.0f);
          optimizer.step(1e-4f);
        },
        1.0);
  }
  std::printf("шаг оптимизатора: %.3f с (%.0f%% сверх прохода)\n",
              optimizer_seconds, 100.0 * optimizer_seconds / step_seconds);
  step_seconds += optimizer_seconds;
  std::printf("шаг обучения целиком: %.3f с\n", step_seconds);
  std::printf("пропускная способность: %.0f токенов/с\n",
              static_cast<double>(batch * seq) / step_seconds);
  std::printf("2000 шагов заняли бы %.1f мин\n", 2000.0 * step_seconds / 60.0);

  // Оценка полезной арифметики: 6 * параметров * токенов — общепринятое
  // приближение для прямого и обратного прохода вместе.
  //
  // Пик умножения матриц не подставляется числом: он зависит от машины, от
  // выбранного микроядра и от числа потоков, и зашитая константа врала бы при
  // каждом из этих изменений. Считать его надо ./bench_gemm, там же, где
  // считается этот замер.
  const double flops = 6.0 * static_cast<double>(config.parameter_count()) *
                       static_cast<double>(batch * seq);
  std::printf("эффективно %.1f GFLOPS (пик умножения матриц — ./bench_gemm)\n",
              flops / step_seconds / 1e9);
  return 0;
}
