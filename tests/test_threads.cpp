// Пул потоков.
//
// Проверять здесь надо не «стало быстрее» — это дело замера, — а три вещи,
// каждая из которых при ошибке стоит дорого и проявляется не сразу:
//   1) каждая задача выполнена ровно один раз. Потерянная задача даёт не
//      падение, а нули в части результата;
//   2) вложенный вызов не встаёт намертво. Параллельные точки у нас вложены
//      естественно: matmul делит батч, gemm делит матрицу;
//   3) исключение из задачи доходит до вызывающего. Иначе LLM_CHECK внутри
//      параллельной области убивает процесс через std::terminate, и от
//      осмысленного сообщения не остаётся ничего.

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/check.h"
#include "core/thread_pool.h"
#include "testing.h"

namespace {

// Возвращает ширину, какой она была, чтобы один тест не влиял на другой.
class WidthGuard {
 public:
  explicit WidthGuard(int width) { llm::set_parallel_width(width); }
  ~WidthGuard() { llm::set_parallel_width(0); }

  WidthGuard(const WidthGuard&) = delete;
  WidthGuard& operator=(const WidthGuard&) = delete;
};

}  // namespace

LLM_TEST(Threads, EveryTaskRunsExactlyOnce) {
  // Задач заведомо больше, чем потоков: важно, что счётчик выдачи не
  // пропускает и не повторяет номера, а не что задачи разошлись по одной.
  const int tasks = 1000;
  std::vector<int> visits(static_cast<std::size_t>(tasks), 0);
  llm::parallel_for(
      tasks, [&](int index) { ++visits[static_cast<std::size_t>(index)]; });
  for (int i = 0; i < tasks; ++i) {
    LLM_CHECK_EQ(visits[static_cast<std::size_t>(i)], 1);
  }
}

LLM_TEST(Threads, ZeroAndOneTaskAreAllowed) {
  int calls = 0;
  llm::parallel_for(0, [&](int) { ++calls; });
  LLM_CHECK_EQ(calls, 0);

  llm::parallel_for(1, [&](int index) {
    LLM_CHECK_EQ(index, 0);
    ++calls;
  });
  LLM_CHECK_EQ(calls, 1);
}

LLM_TEST(Threads, TasksRunAtTheSameTime) {
  // Самая важная проверка и единственная, которую нельзя сделать без
  // ожидания: пул мог бы молча выполнять всё последовательно, и все остальные
  // тесты прошли бы.
  //
  // Приём такой: каждая задача отмечается и ждёт, пока отметятся все.
  // Дождаться этого можно только если задачи идут одновременно. Ожидание
  // ограничено по времени, иначе последовательное выполнение не провалило бы
  // тест, а повесило бы его.
  if (llm::parallel_width() < 2) {
    return;  // Одно ядро — проверять нечего.
  }

  const int tasks = llm::parallel_width();
  std::atomic<int> arrived(0);
  std::atomic<int> saw_everyone(0);
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);

  llm::parallel_for(tasks, [&](int) {
    arrived.fetch_add(1);
    while (arrived.load() < tasks &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    if (arrived.load() == tasks) {
      saw_everyone.fetch_add(1);
    }
  });

  LLM_CHECK_EQ(saw_everyone.load(), tasks);
}

LLM_TEST(Threads, WorkIsSpreadOverSeveralThreads) {
  if (llm::parallel_width() < 2) {
    return;
  }
  const int tasks = llm::parallel_width();
  std::mutex mutex;
  std::set<std::thread::id> threads;
  std::atomic<int> arrived(0);
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);

  llm::parallel_for(tasks, [&](int) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      threads.insert(std::this_thread::get_id());
    }
    // Держим задачу занятой, пока не подтянутся остальные: иначе вызывающий
    // поток успел бы разобрать все задачи сам, и разных потоков не вышло бы.
    arrived.fetch_add(1);
    while (arrived.load() < tasks &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
  });

  LLM_CHECK_EQ(threads.size(), static_cast<std::size_t>(tasks));
}

LLM_TEST(Threads, NestedCallRunsSerially) {
  // Вложенный вызов обязан выполнить свои задачи сам и вернуться. Проверка
  // одновременно и на отсутствие взаимной блокировки: если бы внутренний
  // вызов ждал потоков, тест не закончился бы никогда.
  LLM_CHECK(!llm::inside_parallel_region());

  std::atomic<int> inner(0);
  std::atomic<int> outer(0);
  llm::parallel_for(8, [&](int) {
    LLM_CHECK(llm::inside_parallel_region());
    outer.fetch_add(1);
    llm::parallel_for(4, [&](int) {
      LLM_CHECK(llm::inside_parallel_region());
      inner.fetch_add(1);
    });
  });

  LLM_CHECK_EQ(outer.load(), 8);
  LLM_CHECK_EQ(inner.load(), 32);
  LLM_CHECK(!llm::inside_parallel_region());
}

LLM_TEST(Threads, ExceptionFromTaskReachesTheCaller) {
  bool caught = false;
  try {
    llm::parallel_for(64, [](int index) {
      if (index == 37) {
        throw std::runtime_error("задача упала");
      }
    });
  } catch (const std::runtime_error& error) {
    caught = true;
    LLM_CHECK_EQ(std::string(error.what()), std::string("задача упала"));
  }
  LLM_CHECK(caught);

  // Пул обязан остаться рабочим: одно упавшее умножение не должно лишать
  // процесс потоков до конца работы.
  std::atomic<int> visits(0);
  llm::parallel_for(16, [&](int) { visits.fetch_add(1); });
  LLM_CHECK_EQ(visits.load(), 16);
}

LLM_TEST(Threads, CheckFailureInsideRegionIsCatchable) {
  // То же, но исключением проекта: именно так упадёт настоящая проверка
  // внутри параллельной области.
  bool caught = false;
  try {
    llm::parallel_for(8, [](int index) { LLM_CHECK_LT(index, 4); });
  } catch (const llm::CheckFailure&) {
    caught = true;
  }
  LLM_CHECK(caught);
}

LLM_TEST(Threads, WidthIsClampedAndRestorable) {
  const int automatic = llm::parallel_width();
  LLM_CHECK_GE(automatic, 1);

  {
    const WidthGuard guard(1);
    LLM_CHECK_EQ(llm::parallel_width(), 1);
    // При единичной ширине деление отключено, но задачи всё равно обязаны
    // выполниться — все.
    std::vector<int> visits(10, 0);
    llm::parallel_for(
        10, [&](int index) { ++visits[static_cast<std::size_t>(index)]; });
    for (std::size_t i = 0; i < visits.size(); ++i) {
      LLM_CHECK_EQ(visits[i], 1);
    }
  }
  LLM_CHECK_EQ(llm::parallel_width(), automatic);

  {
    // Ширина больше числа созданных потоков обрезается: участников не бывает
    // больше, чем есть.
    const WidthGuard guard(1000);
    LLM_CHECK_EQ(llm::parallel_width(), automatic);
  }
  LLM_CHECK_EQ(llm::parallel_width(), automatic);
}
