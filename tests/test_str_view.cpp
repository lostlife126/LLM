#include <sstream>
#include <string>

#include "core/str_view.h"
#include "testing.h"

LLM_TEST(StrView, FromCString) {
  llm::StrView view("hello");
  LLM_CHECK_EQ(view.size(), static_cast<std::size_t>(5));
  LLM_CHECK_EQ(view[0], 'h');
  LLM_CHECK(view == llm::StrView("hello"));
}

LLM_TEST(StrView, FromStdString) {
  const std::string owner = "tokenizer";
  llm::StrView view(owner);
  LLM_CHECK_EQ(view.size(), owner.size());
  LLM_CHECK(view.data() == owner.data());  // вид, а не копия
}

LLM_TEST(StrView, DefaultIsEmpty) {
  llm::StrView view;
  LLM_CHECK(view.empty());
  LLM_CHECK(view == llm::StrView(""));
}

LLM_TEST(StrView, HandlesEmbeddedNul) {
  // Вид задаётся длиной, а не завершающим нулём: для byte-level BPE это
  // принципиально, там встречаются любые байты.
  const char raw[] = {'a', '\0', 'b'};
  llm::StrView view(raw, 3);
  LLM_CHECK_EQ(view.size(), static_cast<std::size_t>(3));
  LLM_CHECK_EQ(view[1], '\0');
  LLM_CHECK_EQ(view[2], 'b');
}

LLM_TEST(StrView, Substr) {
  llm::StrView view("abcdef");
  LLM_CHECK(view.substr(2, 3) == llm::StrView("cde"));
  LLM_CHECK(view.substr(3) == llm::StrView("def"));
  // Слишком длинный count просто усекается до конца строки.
  LLM_CHECK(view.substr(4, 100) == llm::StrView("ef"));
  LLM_CHECK(view.substr(6).empty());
}

LLM_TEST(StrView, Prefixes) {
  llm::StrView view("merge_table");
  LLM_CHECK(view.starts_with("merge"));
  LLM_CHECK(!view.starts_with("table"));
  LLM_CHECK(view.ends_with("table"));
  LLM_CHECK(!view.ends_with("merge"));
  LLM_CHECK(view.starts_with(""));
}

LLM_TEST(StrView, Find) {
  llm::StrView view("a,b,c");
  LLM_CHECK_EQ(view.find(','), static_cast<std::size_t>(1));
  LLM_CHECK_EQ(view.find(',', 2), static_cast<std::size_t>(3));
  LLM_CHECK_EQ(view.find('z'), llm::StrView::npos);
}

LLM_TEST(StrView, Ordering) {
  LLM_CHECK(llm::StrView("abc") < llm::StrView("abd"));
  LLM_CHECK(llm::StrView("abc") < llm::StrView("abcd"));
  LLM_CHECK(!(llm::StrView("abc") < llm::StrView("abc")));
}

LLM_TEST(StrView, Streaming) {
  std::ostringstream oss;
  oss << llm::StrView("rms_norm");
  LLM_CHECK(oss.str() == "rms_norm");
}
