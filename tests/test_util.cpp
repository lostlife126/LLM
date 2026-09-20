#include <cstddef>
#include <string>

#include "core/cpu.h"
#include "core/util.h"
#include "testing.h"

LLM_TEST(Util, Clamp) {
  LLM_CHECK_EQ(llm::clamp(5, 0, 10), 5);
  LLM_CHECK_EQ(llm::clamp(-3, 0, 10), 0);
  LLM_CHECK_EQ(llm::clamp(99, 0, 10), 10);
  LLM_CHECK_EQ(llm::clamp(0, 0, 0), 0);
}

LLM_TEST(Util, DivUp) {
  LLM_CHECK_EQ(llm::div_up(0, 8), static_cast<std::size_t>(0));
  LLM_CHECK_EQ(llm::div_up(1, 8), static_cast<std::size_t>(1));
  LLM_CHECK_EQ(llm::div_up(8, 8), static_cast<std::size_t>(1));
  LLM_CHECK_EQ(llm::div_up(9, 8), static_cast<std::size_t>(2));
  LLM_CHECK_EQ(llm::div_up(64, 8), static_cast<std::size_t>(8));
}

LLM_TEST(Util, RoundUp) {
  LLM_CHECK_EQ(llm::round_up(0, 64), static_cast<std::size_t>(0));
  LLM_CHECK_EQ(llm::round_up(1, 64), static_cast<std::size_t>(64));
  LLM_CHECK_EQ(llm::round_up(64, 64), static_cast<std::size_t>(64));
  LLM_CHECK_EQ(llm::round_up(65, 64), static_cast<std::size_t>(128));
}

LLM_TEST(Util, IsAligned) {
  // Инициализация не нужна по смыслу теста (проверяются адреса, не данные),
  // но без неё GCC выдаёт ложное -Wmaybe-uninitialized при инлайне.
  alignas(64) char buffer[128] = {};
  LLM_CHECK(llm::is_aligned(buffer, 64));
  LLM_CHECK(llm::is_aligned(buffer + 64, 64));
  LLM_CHECK(!llm::is_aligned(buffer + 1, 64));
  LLM_CHECK(llm::is_aligned(buffer + 1, 1));
}

LLM_TEST(Util, PadUtf8CountsCharactersNotBytes) {
  // Весь смысл этих функций в том, что printf считает байты, а таблица должна
  // выравниваться по символам: в кириллице байтов вдвое больше.
  LLM_CHECK_EQ(llm::pad_utf8("ab", 5), std::string("ab   "));
  LLM_CHECK_EQ(llm::pad_utf8_right("ab", 5), std::string("   ab"));

  // «шаг» — три символа и шесть байт. Дополнение считается по символам.
  LLM_CHECK_EQ(llm::pad_utf8("шаг", 5), std::string("шаг  "));
  LLM_CHECK_EQ(llm::pad_utf8_right("шаг", 5), std::string("  шаг"));

  // Более длинный текст не обрезается: лучше поехавший столбец, чем потерянное
  // слово.
  LLM_CHECK_EQ(llm::pad_utf8("слишком", 3), std::string("слишком"));
  LLM_CHECK_EQ(llm::pad_utf8_right("слишком", 3), std::string("слишком"));

  // Ровно по ширине — без единого лишнего пробела.
  LLM_CHECK_EQ(llm::pad_utf8("Δ", 1), std::string("Δ"));
  LLM_CHECK_EQ(llm::pad_utf8_right("Δ", 1), std::string("Δ"));
}

LLM_TEST(Cpu, FeatureListIsSeparatedByExactlyOneBlank) {
  // Строка подставляется в шапки всех измерений: «процессор: %s, ядер %d».
  // Хвостовой пробел там виден глазами — «avx512vl , ядер 4». Но проверять
  // только края мало: разделитель, выброшенный целиком, даёт слипшееся
  // «avx2fmaf16c», у которого краевых пробелов тоже нет. Поэтому считаем
  // слова и сверяем с числом признаков.
  const llm::CpuFeatures& features = llm::cpu_features();
  const std::string text = features.to_string();
  LLM_CHECK(!text.empty());

  int expected = 0;
  expected += features.neon ? 1 : 0;
  expected += features.avx2 ? 1 : 0;
  expected += features.fma ? 1 : 0;
  expected += features.f16c ? 1 : 0;
  expected += features.avx512f ? 1 : 0;
  expected += features.avx512bw ? 1 : 0;
  expected += features.avx512vl ? 1 : 0;

  if (expected == 0) {
    // Запасная подпись — единственный случай, когда пробел внутри законен.
    LLM_CHECK_EQ(text, std::string("базовый x86-64"));
    return;
  }

  LLM_CHECK_NE(text.front(), ' ');
  LLM_CHECK_NE(text.back(), ' ');
  LLM_CHECK_EQ(text.find("  "), std::string::npos);

  // Слов ровно столько, сколько включённых признаков: ни слипшихся, ни лишних.
  int words = 1;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == ' ') {
      ++words;
    }
  }
  LLM_CHECK_EQ(words, expected);
}

// Переключатель из окружения разбирается строго.
//
// Проверка нужна ровно из-за одного случая: LLM_FP16=off. Прежнее правило —
// «включено, если непусто и не начинается с нуля» — на нём ВКЛЮЧАЛО имитацию
// половинной разрядности. Слово, которым режим выключают, его включало, и
// замер сходимости шёл не про то, про что заявлен, молча.
LLM_TEST(Util, EnvFlagRefusesWhatItDoesNotUnderstand) {
  LLM_CHECK(!llm::parse_env_flag("X", nullptr));
  LLM_CHECK(!llm::parse_env_flag("X", ""));

  const char* off[] = {"0", "false", "no", "off", "FALSE", "Off", "NO"};
  for (std::size_t i = 0; i < sizeof(off) / sizeof(off[0]); ++i) {
    LLM_CHECK_MSG(!llm::parse_env_flag("X", off[i]),
                  "'" << off[i] << "' должно выключать");
  }
  const char* on[] = {"1", "true", "yes", "on", "TRUE", "On", "YES"};
  for (std::size_t i = 0; i < sizeof(on) / sizeof(on[0]); ++i) {
    LLM_CHECK_MSG(llm::parse_env_flag("X", on[i]),
                  "'" << on[i] << "' должно включать");
  }

  // Всё остальное — падение, а не догадка. В том числе «01» и «2»: прежнее
  // правило видело в них истину, и «0x0» тоже считало ложью по первому байту.
  LLM_EXPECT_THROWS(llm::parse_env_flag("X", "2"));
  LLM_EXPECT_THROWS(llm::parse_env_flag("X", "01"));
  LLM_EXPECT_THROWS(llm::parse_env_flag("X", "0x0"));
  LLM_EXPECT_THROWS(llm::parse_env_flag("X", "да"));
  LLM_EXPECT_THROWS(llm::parse_env_flag("X", " 1"));
}
