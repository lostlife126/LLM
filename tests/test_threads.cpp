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

LLM_TEST(Threads, VaryingTaskCountsDoNotLoseTasks) {
  // Число задач меняется от области к области, и это не праздный случай.
  // Именно он вскрыл гонку: пул будит ровно столько потоков, сколько нужно, и
  // поток, проснувшийся с опозданием и в свою область не принятый, читал
  // число участников уже от следующей. Проявлялось это не потерянной задачей,
  // а счётчиком незавершённых, уходящим ниже нуля, — то есть зависанием.
  //
  // Проверка поэтому не только в том, что все задачи выполнены, но и в том,
  // что тест вообще завершается.
  const int width = llm::parallel_width();
  for (int round = 0; round < 200; ++round) {
    const int tasks = 1 + (round % (width + 2));
    std::vector<int> visits(static_cast<std::size_t>(tasks), 0);
    llm::parallel_for(
        tasks, [&](int index) { ++visits[static_cast<std::size_t>(index)]; });
    for (int i = 0; i < tasks; ++i) {
      LLM_CHECK_EQ(visits[static_cast<std::size_t>(i)], 1);
    }
  }
}

LLM_TEST(Threads, WidthLimitsHowManyThreadsActuallyRun) {
  // Ширина обязана действовать на сам пул, а не только на тех вызывающих, кто
  // сам считает число задач от parallel_width(). Раньше parallel_for её не
  // замечал вовсе: при ширине 1 и четырёх ядрах шестнадцать задач расходились
  // по четырём потокам. На счёт это не влияло — все боевые вызывающие ширину
  // спрашивают, — но ручка, действующая не везде, обесценивает замер «на одном
  // ядре против четырёх», ради которого она и заведена.
  //
  // Считаются не задачи, а различные идентификаторы потока: именно это и
  // означает «сколько ядер занято».
  struct Counter {
    static int distinct(int tasks) {
      std::set<std::thread::id> seen;
      std::mutex guard;
      llm::parallel_for(tasks, [&seen, &guard](int) {
        std::lock_guard<std::mutex> lock(guard);
        seen.insert(std::this_thread::get_id());
      });
      return static_cast<int>(seen.size());
    }
  };

  const int automatic = llm::parallel_width();
  if (automatic < 2) {
    // Машина одноядерная: проверять нечего, и главное — нечем. Молчаливо
    // проходящий тест на одном ядре честнее выдуманного.
    return;
  }

  // Проверка не пуста: при автоматической ширине потоков заведомо больше
  // одного, иначе утверждение про единицу ниже ничего не значило бы.
  LLM_CHECK_MSG(Counter::distinct(64) > 1,
                "при автоматической ширине работа не разошлась по потокам");

  {
    const WidthGuard guard(1);
    LLM_CHECK_MSG(Counter::distinct(64) == 1,
                  "при ширине 1 работа обязана идти в потоке вызывающего");
  }
  {
    const WidthGuard guard(2);
    LLM_CHECK_MSG(Counter::distinct(64) <= 2,
                  "при ширине 2 участников не может быть больше двух");
  }
  LLM_CHECK_EQ(llm::parallel_width(), automatic);
}

LLM_TEST(Threads, ThreadCountFromEnvironmentIsParsedStrictly) {
  // Величина существует ровно ради замера «на одном ядре против четырёх».
  // Понятая не так, она даёт число, с виду неотличимое от измеренного, —
  // поэтому разбор строгий, как и у аргументов командной строки.
  LLM_CHECK_EQ(llm::parse_thread_count("4"), 4);
  LLM_CHECK_EQ(llm::parse_thread_count("1"), 1);

  // Не задано — ноль, то есть «по числу ядер».
  LLM_CHECK_EQ(llm::parse_thread_count(nullptr), 0);
  LLM_CHECK_EQ(llm::parse_thread_count(""), 0);

  // Ноль и отрицательное означают то же самое, что и не заданная величина:
  // так же ведёт себя set_parallel_width, и разнобой был бы неожиданностью.
  LLM_CHECK_EQ(llm::parse_thread_count("0"), 0);
  LLM_CHECK_EQ(llm::parse_thread_count("-3"), 0);

  // Вот случаи, ради которых всё это. Хвост после числа: при разборе без
  // проверки хвоста «4x» дало бы четыре. Мусор перед числом: «x4» дало бы
  // ноль, то есть молча число ядер вместо запрошенного.
  LLM_EXPECT_THROWS(llm::parse_thread_count("4x"));
  LLM_EXPECT_THROWS(llm::parse_thread_count("x4"));
  LLM_EXPECT_THROWS(llm::parse_thread_count("2 "));
  LLM_EXPECT_THROWS(llm::parse_thread_count("2.5"));
  LLM_EXPECT_THROWS(llm::parse_thread_count("2k"));
  // И величина, не помещающаяся в long: strtol выставляет ERANGE и отдаёт
  // предел, то есть без проверки вышло бы правдоподобное число.
  LLM_EXPECT_THROWS(llm::parse_thread_count("99999999999999999999"));

  // Потолок применяется, а не отвергается: больше созданных потоков всё
  // равно не бывает, и отказывать тут не за что.
  LLM_CHECK_EQ(llm::parse_thread_count("100000"), 1024);
}
