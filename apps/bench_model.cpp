// Замер прямого и обратного прохода модели.
//
// Цифра нужна для честной оценки времени обучения: пропускная способность
// GEMM — это верхняя граница, а реальный шаг тратит заметную часть времени на
// нормировки, softmax и мелкие матрицы внутри внимания.
//
// Запуск: ./bench_model [пресет] [батч]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/cpu.h"
#include "core/random.h"
#include "core/thread_pool.h"
#include "nn/model.h"

namespace {

using Clock = std::chrono::steady_clock;

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
  const std::string preset = argc > 1 ? argv[1] : "nano";
  const int64_t batch = argc > 2 ? std::atoi(argv[2]) : 16;

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
    const Clock::time_point start = Clock::now();
    int repetitions = 0;
    double elapsed = 0.0;
    do {
      model.loss(ids, batch, seq);
      ++repetitions;
      elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    } while (elapsed < 2.0);
    std::printf("только прямой проход: %.3f с\n", elapsed / repetitions);
  }

  const Clock::time_point start = Clock::now();
  int repetitions = 0;
  double elapsed = 0.0;
  do {
    llm::autograd::Var loss = model.loss(ids, batch, seq);
    loss.backward();
    const std::vector<llm::nn::NamedParameter> parameters = model.parameters();
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      parameters[i].value->zero_grad();
    }
    ++repetitions;
    elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  } while (elapsed < 4.0);

  const double step_seconds = elapsed / repetitions;
  std::printf("шаг обучения (вперёд + назад): %.3f с\n", step_seconds);
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
