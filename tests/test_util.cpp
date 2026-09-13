#include <cstddef>
#include <string>

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
