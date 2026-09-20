// Замер генерации: обычная разрядность весов против половинной.
//
// Отдельно от apps/generate потому, что там нужен обученный чекпоинт, а здесь
// он не нужен вовсе: скорость генерации от значений весов не зависит, зависит
// только от размеров модели. Значит замер можно запустить на любой машине
// сразу после сборки, ничего не обучая и ничего не скачивая, — а это как раз
// то, что требуется от проверки на чужом железе.
//
// Что именно меряется. Генерация по одному токену с KV-кэшем — та форма, где
// на каждый прочитанный вес приходится ровно одно умножение с накоплением, и
// всё упирается в чтение памяти. Замер трафика показал, что при такой форме
// веса дают две трети всех прочитанных байт.
//
// Запуск: ./bench_generate [пресет] [токенов]

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/cpu.h"
#include "core/thread_pool.h"
#include "infer/generate.h"
#include "ops/gemm.h"
#include "nn/config.h"
#include "nn/model.h"

namespace {

double measure(llm::nn::Model* model, const std::vector<int32_t>& prompt,
               int64_t tokens) {
  llm::infer::GenerateConfig config;
  config.max_tokens = tokens;
  config.sampler.temperature = 0.8f;
  config.sampler.top_k = 40;
  config.sampler.seed = 2026;

  // Лучшее из трёх, а не среднее. Мешает замеру только посторонняя нагрузка, и
  // она может добавить времени, но не убавить, — значит наименьшее из
  // измерений ближе к правде, чем их среднее.
  double best = 0.0;
  for (int round = 0; round < 3; ++round) {
    const llm::infer::GenerateResult result =
        llm::infer::generate(model, prompt, config);
    const double rate =
        static_cast<double>(result.tokens.size()) / result.seconds;
    if (rate > best) {
      best = rate;
    }
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string name = argc > 1 ? argv[1] : "tiny";
  const int64_t tokens = argc > 2 ? std::atoll(argv[2]) : 100;

  const llm::nn::ModelConfig config = llm::nn::ModelConfig::by_name(name);
  llm::nn::Model model(config, 1234);

  const double weight_mb =
      static_cast<double>(model.parameter_count()) * 4.0 / 1048576.0;
  std::printf("процессор: %s, потоков %d\n",
              llm::cpu_features().to_string().c_str(), llm::parallel_width());
  std::printf("пресет %s: %s\n", name.c_str(), config.to_string().c_str());
  std::printf("веса: %.1f МБ в обычной разрядности, %.1f МБ в половинной\n",
              weight_mb, weight_mb / 2.0);

  // Затравка короткая: замеряется декодирование по токену, а не разбор
  // контекста. Длинная затравка добавила бы к результату один проход другой
  // формы и размыла бы то, что мерится.
  std::vector<int32_t> prompt;
  prompt.push_back(1);
  prompt.push_back(2);
  prompt.push_back(3);

  // Подготовка к инференсу в обычной разрядности. Без неё сравнение было бы
  // нечестным: при связанных эмбеддингах обычный путь перекладывал бы таблицу
  // на каждый токен, а половинная разрядность получает её уже переложенной и
  // выигрывала бы не тем, чем заявлено.
  model.prepare_inference();
  // Счётчик сбрасывается перед каждым из двух замеров, а не только перед
  // вторым. Иначе в первый попадает всё, что умножалось до него, и две доли,
  // которые дальше сравниваются, считались бы за разные отрезки работы.
  llm::ops::reset_traffic();
  const double plain = measure(&model, prompt, tokens);
  const llm::ops::GemmTraffic plain_traffic = llm::ops::traffic();

  // Упаковка не заменяет веса, а добавляет копию, поэтому на время замера
  // модель занимает в полтора раза больше памяти.
  model.pack_half();
  llm::ops::reset_traffic();
  const double half = measure(&model, prompt, tokens);
  const llm::ops::GemmTraffic half_traffic = llm::ops::traffic();

  std::printf("\nобычная разрядность:   %7.1f токенов/с\n", plain);
  std::printf("половинная разрядность: %7.1f токенов/с\n", half);
  std::printf("отношение: %.2fx\n", half / plain);

  if (plain_traffic.enabled) {
    // Доля весов в чтении — это потолок для любого дальнейшего сжатия. Если
    // после перехода на половинную разрядность веса дают долю p, то сжатие их
    // ещё вдвое сокращает чтение в 1 / (1 - p / 2) раза, и это ВЕРХНЯЯ оценка:
    // преобразование более узкого формата стоит дороже, а всё остальное —
    // активации, накопитель, кэш ключей — не сокращается вовсе.
    const auto report = [](const char* name, const llm::ops::GemmTraffic& t) {
      const double a = static_cast<double>(t.a_bytes);
      const double b = static_cast<double>(t.b_bytes);
      const double c = static_cast<double>(t.c_bytes);
      const double total = a + b + c;
      if (total <= 0.0) {
        return 0.0;
      }
      std::printf("%s: A %.1f МБ (%.0f%%)  B %.1f МБ (%.0f%%)  C %.1f МБ (%.0f%%)\n",
                  name, a / 1048576.0, 100 * a / total, b / 1048576.0,
                  100 * b / total, c / 1048576.0, 100 * c / total);
      return b / total;
    };
    std::printf("\nтрафик умножений за прогон\n");
    report("обычная   ", plain_traffic);
    const double share = report("половинная", half_traffic);
    if (share > 0.0) {
      std::printf(
          "\nпотолок дальнейшего сжатия весов, от половинной разрядности:\n"
          "  до 8 разрядов  не больше %.2fx\n"
          "  до 4 разрядов  не больше %.2fx\n",
          1.0 / (1.0 - share / 2.0), 1.0 / (1.0 - share * 0.75));
    }
  }
  return 0;
}
