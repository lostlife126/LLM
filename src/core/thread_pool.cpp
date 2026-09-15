#include "core/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#include "core/cpu.h"

namespace llm {
namespace {

// Признак «этот поток сейчас внутри задачи». Ставится и у рабочих потоков, и
// у вызывающего на время его собственной порции работы: вложенный вызов
// обязан увидеть признак независимо от того, кто его сделал.
thread_local bool g_inside_region = false;

using Clock = std::chrono::steady_clock;

// Сколько ждать работу вращением, прежде чем уснуть на условной переменной.
//
// Число выбрано измерением, и без него пул был бы бесполезен. Разбудить три
// спящих потока и дождаться, пока все три отзовутся, стоило здесь 58
// микросекунд — больше, чем само умножение на формах внимания, и обучение
// стало бы медленнее, а не быстрее. Поток, который вращается на атомарном
// счётчике, подхватывает работу за доли микросекунды.
//
// Обратная сторона — ядро занято, пока поток вращается. Поэтому окно
// ограничено: за шаг обучения умножения идут одно за другим с промежутками в
// единицы микросекунд, и окна хватает, чтобы потоки не уснули между ними; а
// на длинной паузе — чтении данных, записи снимка — потоки всё-таки засыпают
// и не жгут ядра впустую.
constexpr std::chrono::microseconds kSpinWindow(200);

// Пауза внутри цикла вращения. Инструкция pause на x86 подсказывает
// процессору, что это ожидание: она снижает энергопотребление и, что важнее,
// освобождает исполнительные устройства другому потоку того же ядра.
inline void spin_pause() {
#if defined(__x86_64__) || defined(_M_X64)
  __builtin_ia32_pause();
#else
  std::this_thread::yield();
#endif
}

// Поколение области и число её участников лежат в одном атомарном слове:
// поколение в старших разрядах, число участников в младших.
//
// Раздельно читать их нельзя, и это не осторожность, а исправленная ошибка.
// Поток, проснувшийся с опозданием, прочитал бы поколение своей области, а
// число участников — уже следующей; решив, что он в неё принят, он уменьшил бы
// её счётчик незавершённых, не будучи в нём учтён, а затем, проснувшись
// законно, уменьшил бы его второй раз. Счётчик ушёл бы ниже нуля, и ожидание
// в run() не закончилось бы никогда. Атомарное слово делает пару
// согласованной по построению.
constexpr int kWorkerBits = 16;

std::uint64_t make_epoch(std::uint64_t generation, int workers) {
  return (generation << kWorkerBits) | static_cast<std::uint64_t>(workers);
}

int epoch_workers(std::uint64_t epoch) {
  return static_cast<int>(epoch & ((1u << kWorkerBits) - 1));
}

class Pool {
 public:
  explicit Pool(int extra_threads) {
    const int count = std::max(0, extra_threads);
    threads_.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      threads_.push_back(std::thread(&Pool::worker_loop, this, i));
    }
  }

  ~Pool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_.store(true, std::memory_order_relaxed);
      ++generation_;
      epoch_.store(make_epoch(generation_, 0), std::memory_order_release);
    }
    start_.notify_all();
    for (std::size_t i = 0; i < threads_.size(); ++i) {
      threads_[i].join();
    }
  }

  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  // Сколько потоков реально участвует в работе: созданные плюс вызывающий.
  int participants() const { return static_cast<int>(threads_.size()) + 1; }

  void run(int tasks, const std::function<void(int)>& task) {
    if (tasks <= 0) {
      return;
    }
    // Последовательные случаи: делить нечего, делить некем, или мы уже внутри
    // параллельной области и делить нельзя.
    if (tasks == 1 || threads_.empty() || g_inside_region) {
      for (int i = 0; i < tasks; ++i) {
        task(i);
      }
      return;
    }

    // Пул один, поэтому две параллельные области не могут идти одновременно.
    // Второй вызывающий дождётся первого, а не испортит ему состояние.
    std::lock_guard<std::mutex> serialize(job_mutex_);

    // Будим ровно столько потоков, сколько нужно: вызывающий — тоже
    // участник, и при двух задачах второй поток не нужен вовсе.
    const int workers = std::min(static_cast<int>(threads_.size()), tasks - 1);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_ = &task;
      job_tasks_ = tasks;
      next_.store(0, std::memory_order_relaxed);
      busy_.store(workers, std::memory_order_relaxed);
      failure_ = nullptr;
      // Слово области меняется под мьютексом сознательно. Вращающийся поток
      // читает его без мьютекса и увидит новое значение сразу, а вот
      // засыпающий проверяет условие, держа мьютекс, — и если бы слово
      // менялось снаружи, оповещение могло бы попасть точно между проверкой
      // условия и засыпанием и потеряться. Так пул встал бы намертво.
      ++generation_;
      epoch_.store(make_epoch(generation_, workers), std::memory_order_release);
    }
    start_.notify_all();

    // Вызывающий берёт задачи наравне с остальными.
    g_inside_region = true;
    drain();
    g_inside_region = false;

    wait_for_workers(workers);

    // Описание области намеренно не сбрасывается. Сброс был бы записью,
    // которую мог бы прочитать поток, проснувшийся с опозданием и в эту
    // область не принятый; перезапишет его следующая область, а она не
    // начнётся, пока не вернётся этот вызов.
    std::exception_ptr failure;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      failure = failure_;
    }
    if (failure) {
      std::rethrow_exception(failure);
    }
  }

 private:
  // Ждёт новую область. Возвращает false, если пора заканчивать.
  bool wait_for_work(std::uint64_t* seen) {
    const Clock::time_point deadline = Clock::now() + kSpinWindow;
    for (int spin = 0;; ++spin) {
      const std::uint64_t epoch = epoch_.load(std::memory_order_acquire);
      if (epoch != *seen) {
        *seen = epoch;
        return !stopping_.load(std::memory_order_relaxed);
      }
      spin_pause();
      // Часы читаются не каждый оборот: чтение стоит десятки наносекунд, а
      // пауза — единицы, и проверка времени стала бы главным содержанием
      // цикла.
      if ((spin & 0xFF) == 0xFF && Clock::now() >= deadline) {
        break;
      }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    start_.wait(lock, [this, seen]() {
      return epoch_.load(std::memory_order_relaxed) != *seen;
    });
    *seen = epoch_.load(std::memory_order_relaxed);
    return !stopping_.load(std::memory_order_relaxed);
  }

  // Ждёт, пока рабочие потоки закончат свои порции. Вращение здесь по той же
  // причине: без него каждая параллельная область платила бы за пробуждение
  // вызывающего.
  void wait_for_workers(int workers) {
    if (workers == 0) {
      return;
    }
    const Clock::time_point deadline = Clock::now() + kSpinWindow;
    for (int spin = 0;; ++spin) {
      if (busy_.load(std::memory_order_acquire) == 0) {
        return;
      }
      spin_pause();
      if ((spin & 0xFF) == 0xFF && Clock::now() >= deadline) {
        break;
      }
    }
    std::unique_lock<std::mutex> lock(mutex_);
    // Рабочий уменьшает busy_ до захвата мьютекса, а оповещает под ним,
    // поэтому оповещение не может проскочить между проверкой условия и
    // ожиданием: обе операции идут под тем же мьютексом.
    finish_.wait(
        lock, [this]() { return busy_.load(std::memory_order_acquire) == 0; });
  }

  void worker_loop(int index) {
    g_inside_region = true;
    std::uint64_t seen = 0;
    for (;;) {
      if (!wait_for_work(&seen)) {
        return;
      }
      // В область могло войти меньше потоков, чем создано. Лишние просто
      // возвращаются к ожиданию и счётчик завершения не трогают. Число
      // участников берётся из того же слова, что и поколение, — см. выше.
      if (index >= epoch_workers(seen)) {
        continue;
      }
      drain();
      // Уменьшаем счётчик до мьютекса, а оповещаем под ним — см.
      // wait_for_workers. Оповещает только последний: остальным нечего.
      if (busy_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lock(mutex_);
        finish_.notify_all();
      }
    }
  }

  // Разбор задач до исчерпания. Номера выдаются атомарным счётчиком: так
  // порция достаётся тому, кто освободился, без отдельной раздачи.
  void drain() {
    const int tasks = job_tasks_;
    const std::function<void(int)>& task = *job_;
    for (;;) {
      const int index = next_.fetch_add(1, std::memory_order_relaxed);
      if (index >= tasks) {
        return;
      }
      try {
        task(index);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_) {
          failure_ = std::current_exception();
        }
      }
    }
  }

  std::vector<std::thread> threads_;
  std::mutex job_mutex_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable finish_;

  // Описание текущей области. Пишется под mutex_ до записи слова области,
  // читается участниками после того, как слово увидено: выпуск и захват на
  // epoch_ и делают эту передачу безопасной без мьютекса на чтении. Читают
  // эти поля только участники, и только пока область не закончилась, — а
  // перезаписать их может лишь следующая область, которая до тех пор не
  // начнётся.
  const std::function<void(int)>* job_ = nullptr;
  int job_tasks_ = 0;

  std::atomic<std::uint64_t> epoch_{0};
  std::uint64_t generation_ = 0;  // под mutex_
  std::atomic<int> next_{0};
  std::atomic<int> busy_{0};
  std::atomic<bool> stopping_{false};
  std::exception_ptr failure_;
};

int env_thread_count() {
  const char* text = std::getenv("LLM_THREADS");
  if (text == nullptr || *text == '\0') {
    return 0;
  }
  const long value = std::strtol(text, nullptr, 10);
  if (value <= 0) {
    return 0;
  }
  return static_cast<int>(std::min<long>(value, 1024));
}

int default_width() {
  const int from_env = env_thread_count();
  if (from_env > 0) {
    return from_env;
  }
  const int cores = detect_core_count();
  return cores > 0 ? cores : 1;
}

Pool& pool() {
  // Размер пула выбирается один раз и не зависит от ширины: менять число
  // потоков на ходу — значит создавать и убивать их в середине счёта, а это
  // ровно та цена, которой пул и существует избежать. Ширина ограничивает
  // число задач, лишние потоки просто спят.
  static Pool instance(default_width() - 1);
  return instance;
}

std::atomic<int>& configured_width() {
  static std::atomic<int> width(default_width());
  return width;
}

}  // namespace

int parallel_width() {
  const int width = configured_width().load(std::memory_order_relaxed);
  const int limit = pool().participants();
  return std::max(1, std::min(width, limit));
}

void set_parallel_width(int width) {
  configured_width().store(width > 0 ? width : default_width(),
                           std::memory_order_relaxed);
}

void parallel_for(int tasks, const std::function<void(int)>& task) {
  pool().run(tasks, task);
}

void parallel_range(std::int64_t count, std::int64_t grain,
                    const std::function<void(std::int64_t, std::int64_t)>& fn) {
  if (count <= 0) {
    return;
  }
  const std::int64_t step = std::max<std::int64_t>(grain, 1);
  const int width = parallel_width();
  const int tasks = static_cast<int>(
      std::max<std::int64_t>(1, std::min<std::int64_t>(width, count / step)));
  if (tasks == 1) {
    fn(0, count);
    return;
  }
  parallel_for(tasks, [count, tasks, &fn](int index) {
    // Границы считаются от номера куска, а не накоплением: так номер
    // однозначно определяет отрезок, и куски не разъезжаются при округлении.
    const std::int64_t begin = count * index / tasks;
    const std::int64_t end = count * (index + 1) / tasks;
    if (begin < end) {
      fn(begin, end);
    }
  });
}

bool inside_parallel_region() { return g_inside_region; }

}  // namespace llm
