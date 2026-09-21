#include "train/trainer.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/check.h"
#include "core/util.h"
#include "nn/dropout.h"
#include "serialize/checkpoint.h"
#include "train/resume.h"

namespace llm {
namespace train {
namespace {

using Clock = std::chrono::steady_clock;

// Номер слоя из имени параметра вида "block.3.attention.query.weight".
// Возвращает -1 для параметров вне слоёв (эмбеддинги, финальная нормировка).
int64_t layer_of(const std::string& name) {
  const std::string prefix = "block.";
  if (name.compare(0, prefix.size(), prefix) != 0) {
    return -1;
  }
  const std::size_t dot = name.find('.', prefix.size());
  if (dot == std::string::npos) {
    return -1;
  }
  // Разбор строгий, хотя строку сюда подаёт не человек, а сборка имён в
  // модели. Причина в том, что atoll на нечисловом куске возвращает нуль, и
  // параметр с именем вроде block.lora.weight молча приписался бы СЛОЮ НОЛЬ —
  // не пустой строкой в таблице, а завышенной нормой у первого слоя. Это
  // последний atoll в проекте; остальные убраны из разбора аргументов по той
  // же причине, и держать исключение ради того, что «сейчас имена всегда
  // числовые», значит ставить диагностику в зависимость от чужого соглашения
  // об именах.
  const std::string digits = name.substr(prefix.size(), dot - prefix.size());
  if (digits.empty()) {
    return -1;
  }
  int64_t layer = 0;
  for (std::size_t i = 0; i < digits.size(); ++i) {
    if (digits[i] < '0' || digits[i] > '9') {
      return -1;
    }
    layer = layer * 10 + (digits[i] - '0');
    // Номер слоя больше числа слоёв всё равно отбрасывается вызывающим;
    // предел нужен затем, чтобы длинная цифровая строка не переполнила счёт.
    if (layer > 1000000) {
      return -1;
    }
  }
  return layer;
}

// Норма градиента по каждому слою отдельно.
//
// Общая норма говорит, велик ли градиент; послойная — где именно он велик.
// Затухание к первым слоям или взрыв в последних видно только так.
//
// Считать это ОБЯЗАТЕЛЬНО до обрезки нормы, и вот почему. Обрезка умножает
// все градиенты на общий множитель, то есть послойные нормы после неё
// оказываются в другом масштабе, чем общая норма, которую печатает шапка, —
// а шапка печатает величину до обрезки. Сравнивать их было нельзя, и это не
// умозрительно: на восьмом шаге nano общая норма 2.4987, слой 0 до обрезки
// 1.3516, а в таблице стояло 0.5409.
//
// Хуже того, обрезка нормирует сумму, поэтому послойный столбец переставал
// меняться от шага к шагу (0.89, 0.88, 0.88, 0.88, 0.87, 0.88, 0.83, 0.83)
// при том, что сам градиент гулял (2.31, 2.21, 2.44, 2.34, 2.54, 2.78, 2.20,
// 2.08). То есть диагностика прятала ровно то изменение, ради которого
// заведена.
std::vector<float> grad_norm_by_layer(
    const std::vector<nn::NamedParameter>& parameters, int64_t layers) {
  std::vector<double> squares(static_cast<std::size_t>(layers), 0.0);
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    const int64_t layer = layer_of(parameters[i].name);
    if (layer < 0 || layer >= layers) {
      continue;
    }
    const Tensor& grad = parameters[i].value->grad();
    if (!grad.defined()) {
      continue;
    }
    const float* data = grad.data();
    for (int64_t element = 0; element < grad.numel(); ++element) {
      squares[static_cast<std::size_t>(layer)] +=
          static_cast<double>(data[element]) * data[element];
    }
  }
  std::vector<float> norms;
  for (std::size_t i = 0; i < squares.size(); ++i) {
    norms.push_back(static_cast<float>(std::sqrt(squares[i])));
  }
  return norms;
}

void print_progress_bar(int64_t step, int64_t total) {
  const int width = 20;
  const int filled = static_cast<int>(step * width / total);
  std::printf("[");
  for (int i = 0; i < width; ++i) {
    std::printf("%s", i < filled ? "=" : " ");
  }
  std::printf("]");
}

std::string format_duration(double seconds) {
  char buffer[64];
  if (seconds < 90.0) {
    std::snprintf(buffer, sizeof(buffer), "%.0f с", seconds);
  } else if (seconds < 5400.0) {
    std::snprintf(buffer, sizeof(buffer), "%.1f мин", seconds / 60.0);
  } else {
    std::snprintf(buffer, sizeof(buffer), "%.1f ч", seconds / 3600.0);
  }
  return std::string(buffer);
}

// Полная диагностика: что происходит внутри модели, а не только чему равны
// потери.
void print_diagnostics(const nn::ForwardStats& stats,
                       const std::vector<float>& layer_grad_norms,
                       float validation_loss, float train_loss) {
  std::printf("\n  --- диагностика ---\n");
  if (validation_loss > 0.0f) {
    std::printf(
        "  проверочные потери %.4f  перплексия %.1f  расхождение с обучающими "
        "%+.3f\n",
        validation_loss, std::exp(validation_loss),
        validation_loss - train_loss);
  }

  const std::size_t layers = stats.residual_rms.size();
  if (layers != 0) {
    // Ширины считаются в символах: в подписях есть кириллица и знак набла, а
    // printf выравнивает по байтам.
    std::printf(
        "\n  %s%s%s%s%s\n", pad_utf8("слой", 7).c_str(),
        pad_utf8("норма∇", 10).c_str(), pad_utf8("RMS остатка", 14).c_str(),
        pad_utf8("внимание", 11).c_str(), pad_utf8("вентили", 9).c_str());

    for (std::size_t layer = 0; layer < layers; ++layer) {
      std::printf("  %-7zu", layer);
      if (layer < layer_grad_norms.size()) {
        std::printf("%-10.4f", layer_grad_norms[layer]);
      } else {
        std::printf("%-10s", "-");
      }
      std::printf("%-14.4f", stats.residual_rms[layer]);
      if (layer < stats.attention_entropy.size()) {
        std::printf("%-11.3f", stats.attention_entropy[layer]);
      } else {
        std::printf("%-11s", "-");
      }
      if (layer < stats.gate_open_fraction.size()) {
        std::printf("%.0f%%", stats.gate_open_fraction[layer] * 100.0);
      } else {
        std::printf("%s", "-");
      }
      std::printf("\n");
    }
  }

  if (layers != 0) {
    // Пояснение строкой, а не подписью под столбцами: выравнивать его пришлось
    // бы вручную, а смысл от этого понятнее не становится.
    std::printf(
        "  энтропия внимания: 1 — равномерно, 0 — сосредоточено; "
        "вентили — доля открытых каналов\n");
  }

  if (stats.loss_by_quarter.size() == 4) {
    std::printf("\n  потери по четвертям окна: %.3f / %.3f / %.3f / %.3f",
                stats.loss_by_quarter[0], stats.loss_by_quarter[1],
                stats.loss_by_quarter[2], stats.loss_by_quarter[3]);
    // Конец окна должен предсказываться лучше начала: там больше контекста.
    // Положительная величина означает, что контекст помогает; ноль или
    // отрицательная — что модель им ещё не пользуется.
    const float gain = stats.loss_by_quarter[0] - stats.loss_by_quarter[3];
    std::printf("   выигрыш от контекста %+.3f\n", gain);
  }

  // log Z выходных логитов: softmax к нему нечувствителен, поэтому дрейф от
  // нуля ничего не меняет в предсказаниях и виден только здесь. Это то, что
  // штрафует Z-loss, — и по этому числу видно, есть ли что штрафовать.
  std::printf("  log Z выходных логитов %+.2f\n", stats.log_z);
  std::printf("\n");
}

}  // namespace

float evaluate(nn::Model* model, const data::TokenDataset& dataset,
               int64_t batch, int64_t seq, int64_t max_batches,
               nn::ForwardStats* stats) {
  LLM_CHECK(model != nullptr);
  // Дропаут обязан быть выключен, и проверяется это здесь, а не подразумевается
  // вызывающим.
  //
  // Сейчас инвариант держится на расстановке скобок в train(): область
  // обучения охватывает только проход вперёд-назад, а evaluate вызывается за
  // ней. Расширить эту область на шаг целиком — правка на один символ, и она
  // ничего не сломает заметно: прогон не упадёт, потери не станут NaN. Они
  // просто окажутся выше и шумнее, потому что часть каналов на проверке
  // зануляется, — а по этим числам выбирается лучший чекпоинт.
  //
  // Отключать дропаут прямо тут было бы хуже: функция чинила бы состояние,
  // заданное снаружи, и настоящая ошибка вызывающего осталась бы незамеченной.
  LLM_CHECK_MSG(!nn::training_mode(),
                "evaluate вызван внутри области обучения: дропаут включён, и "
                "проверочные потери получились бы завышенными");
  const int64_t available = dataset.validation_batch_count(batch, seq);
  const int64_t count = available < max_batches ? available : max_batches;
  // Считать нечего — это один случай, а не два. Раньше пустая проверочная
  // часть давала нуль, а max_batches == 0 доходил до деления на count и
  // возвращал NaN: цикл не выполнялся, а делить всё равно приходилось.
  // Тренер сюда с нулём не приходит — он спрашивает eval_batches > 0, — но
  // функция объявлена в заголовке, и молчаливый NaN в проверочных потерях
  // разбирать пришлось бы долго: он пошёл бы дальше в выбор лучшего снимка.
  if (count <= 0) {
    return 0.0f;
  }

  // Лента не нужна: считается только значение, и без неё не расходуется
  // память на промежуточные величины.
  autograd::NoGradGuard no_grad;

  double total = 0.0;
  double accuracy_total = 0.0;
  double entropy_total = 0.0;
  double log_z_total = 0.0;
  std::vector<double> quarter_totals(4, 0.0);
  nn::ForwardStats last_batch_stats;

  for (int64_t index = 0; index < count; ++index) {
    const std::vector<int32_t> ids =
        dataset.validation_batch(batch, seq, index);
    // Диагностика собирается всегда, даже если вызывающий её не просил: из неё
    // берётся чистая перекрёстная энтропия. Без этого проверочные потери
    // включали бы вспомогательные штрафы и были бы несравнимы между
    // конфигурациями.
    nn::ForwardStats batch_stats;
    model->loss(ids, batch, seq, &batch_stats);
    total += batch_stats.cross_entropy;

    if (stats != nullptr) {
      accuracy_total += batch_stats.top1_accuracy;
      entropy_total += batch_stats.prediction_entropy;
      log_z_total += batch_stats.log_z;
      for (std::size_t i = 0; i < batch_stats.loss_by_quarter.size() && i < 4;
           ++i) {
        quarter_totals[i] += batch_stats.loss_by_quarter[i];
      }
      last_batch_stats = batch_stats;
    }
  }

  if (stats != nullptr) {
    const double denominator = static_cast<double>(count);
    // Послойные величины берутся с последнего батча: они устойчивы и почти не
    // зависят от того, какой именно текст подан, так что усреднять их незачем.
    *stats = last_batch_stats;
    stats->top1_accuracy = static_cast<float>(accuracy_total / denominator);
    stats->prediction_entropy = static_cast<float>(entropy_total / denominator);
    stats->log_z = static_cast<float>(log_z_total / denominator);
    stats->loss_by_quarter.clear();
    for (std::size_t i = 0; i < 4; ++i) {
      stats->loss_by_quarter.push_back(
          static_cast<float>(quarter_totals[i] / denominator));
    }
  }
  return static_cast<float>(total / static_cast<double>(count));
}

TrainReport train(nn::Model* model, const data::TokenDataset& dataset,
                  const TrainConfig& config) {
  LLM_CHECK(model != nullptr);
  const int64_t seq =
      config.seq_len > 0 ? config.seq_len : model->config().max_seq_len;
  LLM_CHECK_LE(seq, model->config().max_seq_len);
  // Снимок пишется по расписанию чекпоинта и только по нему. Путь без
  // расписания — настройка, которая ничего не делает: прогон выглядел бы
  // возобновляемым, а после обрыва оказалось бы, что сохранять было нечем.
  LLM_CHECK_MSG(config.resume_path.empty() || config.checkpoint_every > 0,
                "задан путь снимка, но checkpoint_every равен "
                    << config.checkpoint_every
                    << ": снимок пишется по расписанию чекпоинта, и при нуле "
                       "не писался бы никогда");

  AdamWConfig optimizer_config;
  optimizer_config.weight_decay = config.weight_decay;
  // Обучаемые, а не все: при дообучении адаптерами базовые веса заморожены,
  // и заводить на них моменты Адама — чистая трата памяти.
  AdamW optimizer(model->trainable_parameters(), optimizer_config);

  ScheduleConfig schedule;
  schedule.max_learning_rate = config.max_learning_rate;
  schedule.min_ratio = config.min_lr_ratio;
  schedule.warmup_steps = config.warmup_steps;
  schedule.total_steps = config.steps;

  Rng rng(config.seed);
  TrainReport report;
  // Стало ли сохранение «лучшим». Пока не стало, работает периодическое: иначе
  // прогон без проверочной выборки не сохранил бы ничего.
  bool saved_best = false;
  const Clock::time_point start_time = Clock::now();

  // Возобновление. Веса и моменты берутся из снимка, а порядок батчей
  // переигрывается: генератор создан с тем же зерном, и достаточно прокрутить
  // выборку столько раз, сколько шагов уже сделано. Так порядок совпадает не
  // «примерно», а по построению, и хранить состояние генератора не нужно.
  RunShape shape;
  shape.batch_size = config.batch_size;
  shape.seq_len = seq;
  shape.seed = config.seed;

  int64_t start_step = 0;
  if (!config.resume_path.empty() && resume_exists(config.resume_path)) {
    const ResumeState state =
        load_resume(config.resume_path, model, &optimizer, shape);
    start_step = state.step;
    report.best_validation_loss = state.best_validation_loss;
    report.best_step = state.best_step;
    report.resumed_from = start_step;
    LLM_CHECK_MSG(start_step <= config.steps,
                  "снимок сделан на шаге " << start_step
                                           << ", а расписание кончается на "
                                           << config.steps);
    for (int64_t skipped = 0; skipped < start_step; ++skipped) {
      dataset.sample_batch(config.batch_size, seq, &rng, false);
    }
    if (config.verbose) {
      std::printf("возобновлено с шага %lld, лучшее пока %.4f на шаге %lld\n",
                  static_cast<long long>(start_step),
                  state.best_validation_loss,
                  static_cast<long long>(state.best_step));
    }
  }

  const int64_t stop_step =
      config.stop_at_step > 0 && config.stop_at_step < config.steps
          ? config.stop_at_step
          : config.steps;

  if (config.verbose) {
    std::printf(
        "параметров: %lld, обучаемых: %lld, из них с распадом веса: %lld\n",
        static_cast<long long>(model->parameter_count()),
        static_cast<long long>(model->trainable_parameter_count()),
        static_cast<long long>(optimizer.decayed_parameter_count()));
    std::printf("обучающих токенов: %lld, проверочных: %lld\n",
                static_cast<long long>(dataset.train_size()),
                static_cast<long long>(dataset.validation_size()));
  }

  for (int64_t step = start_step; step < stop_step; ++step) {
    const std::vector<int32_t> ids =
        dataset.sample_batch(config.batch_size, seq, &rng, false);

    // «Последний» — последний в этом запуске, а не в расписании: на нём надо
    // снять диагностику и сохраниться, даже если расписание не кончилось.
    const bool last = step + 1 == stop_step;
    const bool wants_log =
        config.log_every > 0 && ((step + 1) % config.log_every == 0 || last);
    const bool wants_diagnostics =
        config.diagnostics_every > 0 &&
        ((step + 1) % config.diagnostics_every == 0 || last);

    // Диагностика считается только когда её собираются показать: редукции по
    // всем активациям стоят заметной доли шага, а меняются эти величины
    // медленно.
    nn::ForwardStats stats;
    nn::ForwardStats* stats_pointer =
        (wants_log || wants_diagnostics) ? &stats : nullptr;

    // Режим обучения включается ровно на прямой и обратный проход шага.
    // Дальше по коду идут оценка на проверочной выборке и диагностика, и там
    // дропаут был бы прямым враньём: проверочные потери зависели бы от
    // случайной маски.
    //
    // Зерно связано с номером шага, поэтому маски у каждого шага свои, но весь
    // прогон воспроизводится в точности.
    float loss_value = 0.0f;
    autograd::Var loss;
    {
      const nn::TrainingScope training(config.seed * 1000003u +
                                       static_cast<uint64_t>(step));
      loss = model->loss(ids, config.batch_size, seq, stats_pointer);
      loss_value = *loss.value().data();
      LLM_CHECK_MSG(std::isfinite(loss_value),
                    "потери перестали быть конечными на шаге " << step);

      optimizer.zero_grad();
      loss.backward();
    }
    // Послойные нормы снимаются ДО обрезки — см. grad_norm_by_layer — и
    // только на тех шагах, где диагностика будет показана: проход по всем
    // градиентам стоит заметной доли шага.
    std::vector<float> layer_grad_norms;
    if (wants_diagnostics) {
      layer_grad_norms =
          grad_norm_by_layer(optimizer.parameters(), model->config().n_layers);
    }
    const float norm = optimizer.clip_grad_norm(config.grad_clip);
    const float learning_rate = learning_rate_at(schedule, step);
    optimizer.step(learning_rate);

    report.train_loss.push_back(loss_value);
    report.grad_norm.push_back(norm);
    report.update_ratio.push_back(optimizer.last_update_ratio());

    if (config.verbose && wants_log) {
      const double elapsed =
          std::chrono::duration<double>(Clock::now() - start_time).count();
      const double tokens =
          static_cast<double>((step + 1) * config.batch_size * seq);
      const double per_step = elapsed / static_cast<double>(step + 1);
      const double remaining =
          per_step * static_cast<double>(config.steps - step - 1);

      std::printf("шаг %6lld/%lld ", static_cast<long long>(step + 1),
                  static_cast<long long>(config.steps));
      print_progress_bar(step + 1, config.steps);
      std::printf(
          "  потери %7.4f  перпл %7.1f  точность %5.1f%%  энтропия %5.3f\n",
          loss_value, std::exp(loss_value), 100.0 * stats.top1_accuracy,
          stats.prediction_entropy);
      // Полезная арифметика: общепринятое приближение 6 * параметров * токенов
      // на прямой и обратный проход вместе. Рядом с пропускной способностью
      // GEMM видно, сколько времени уходит не на умножение матриц.
      const double flops = 6.0 * static_cast<double>(model->parameter_count()) *
                           static_cast<double>(config.batch_size * seq);
      std::printf(
          "                                 норма∇ %6.3f  шаг/вес %.1e  "
          "скорость %.2e  %.0f ток/с  %.1f GFLOPS  осталось %s\n",
          norm, static_cast<double>(optimizer.last_update_ratio()),
          static_cast<double>(learning_rate), tokens / elapsed,
          flops / per_step / 1e9, format_duration(remaining).c_str());
      std::fflush(stdout);
    }

    if (wants_diagnostics) {
      float validation = 0.0f;
      // Диагностика снимается с проверочной выборки и усредняется по всем её
      // батчам. На одном обучающем батче разбивка потерь по четвертям окна
      // тонула в шуме: она скакала от -0.15 до +0.27 без всякой системы, то
      // есть не говорила ничего.
      nn::ForwardStats shown = stats;
      // Ноль проверочных батчей бывает не только по просьбе, но и когда
      // выборка короче одного окна. Тогда evaluate вернул бы ноль, и этот
      // ноль стал бы «лучшим» результатом навсегда.
      if (config.eval_batches > 0 &&
          dataset.validation_batch_count(config.batch_size, seq) > 0) {
        validation = evaluate(model, dataset, config.batch_size, seq,
                              config.eval_batches, &shown);
        report.final_validation_loss = validation;

        if (report.best_step == 0 || validation < report.best_validation_loss) {
          report.best_validation_loss = validation;
          report.best_step = step + 1;
          // Сохраняем ровно здесь, а не в конце: к концу этот набор весов уже
          // не существует.
          if (config.keep_best && !config.checkpoint_path.empty()) {
            serialize::save_checkpoint(config.checkpoint_path, model, step + 1);
            saved_best = true;
          }
        }
      }
      report.final_accuracy = shown.top1_accuracy;
      report.final_prediction_entropy = shown.prediction_entropy;
      report.final_stats = shown;

      if (config.verbose) {
        print_diagnostics(shown, layer_grad_norms, validation, loss_value);
        std::fflush(stdout);
      }
    }
    // Периодическое сохранение остаётся для случая, когда мерить нечем: без
    // проверочной выборки «лучший» определить невозможно, и последний —
    // единственный разумный выбор.
    if (!saved_best && config.checkpoint_every > 0 &&
        !config.checkpoint_path.empty() &&
        ((step + 1) % config.checkpoint_every == 0 || last)) {
      serialize::save_checkpoint(config.checkpoint_path, model, step + 1);
    }

    // Снимок пишется по тому же расписанию, что и чекпоинт, и обязательно на
    // последнем шаге запуска: иначе прерванный прогон потеряет всё, что успел
    // после предыдущего сохранения.
    if (!config.resume_path.empty() && config.checkpoint_every > 0 &&
        ((step + 1) % config.checkpoint_every == 0 || last)) {
      ResumeState state;
      state.step = step + 1;
      state.best_validation_loss = report.best_validation_loss;
      state.best_step = report.best_step;
      save_resume(config.resume_path, model, optimizer, state, shape);
    }
  }

  report.seconds =
      std::chrono::duration<double>(Clock::now() - start_time).count();
  if (!report.train_loss.empty()) {
    report.final_train_loss = report.train_loss.back();
  }
  return report;
}

std::vector<float> overfit_batch(nn::Model* model,
                                 const std::vector<int32_t>& ids, int64_t batch,
                                 int64_t seq, int64_t steps,
                                 float learning_rate) {
  LLM_CHECK(model != nullptr);
  AdamWConfig optimizer_config;
  // Регуляризация здесь мешает: цель — именно запомнить батч.
  optimizer_config.weight_decay = 0.0f;
  // Обучаемые, а не все: при дообучении адаптерами базовые веса заморожены,
  // и заводить на них моменты Адама — чистая трата памяти.
  AdamW optimizer(model->trainable_parameters(), optimizer_config);

  std::vector<float> history;
  for (int64_t step = 0; step < steps; ++step) {
    autograd::Var loss = model->loss(ids, batch, seq);
    history.push_back(*loss.value().data());
    optimizer.zero_grad();
    loss.backward();
    optimizer.clip_grad_norm(1.0f);
    optimizer.step(learning_rate);
  }
  return history;
}

}  // namespace train
}  // namespace llm
