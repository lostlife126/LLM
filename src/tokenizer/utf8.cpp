#include "tokenizer/utf8.h"

namespace llm {
namespace {

bool is_continuation(char byte) {
  return (static_cast<unsigned char>(byte) & 0xC0u) == 0x80u;
}

uint32_t tail(char byte) {
  return static_cast<uint32_t>(static_cast<unsigned char>(byte) & 0x3Fu);
}

}  // namespace

Utf8Char decode_utf8(const char* data, std::size_t size) {
  Utf8Char out;
  if (size == 0) {
    return out;
  }
  const unsigned char lead = static_cast<unsigned char>(data[0]);

  if (lead < 0x80u) {
    out.code = lead;
    out.bytes = 1;
    out.valid = true;
    return out;
  }

  int length = 0;
  uint32_t code = 0;
  if ((lead & 0xE0u) == 0xC0u) {
    length = 2;
    code = lead & 0x1Fu;
  } else if ((lead & 0xF0u) == 0xE0u) {
    length = 3;
    code = lead & 0x0Fu;
  } else if ((lead & 0xF8u) == 0xF0u) {
    length = 4;
    code = lead & 0x07u;
  } else {
    // Либо байт продолжения не на своём месте, либо начало пятибайтовой
    // последовательности, которых в UTF-8 нет.
    out.code = lead;
    out.bytes = 1;
    return out;
  }

  if (static_cast<std::size_t>(length) > size) {
    out.code = lead;
    out.bytes = 1;
    return out;
  }
  for (int i = 1; i < length; ++i) {
    if (!is_continuation(data[i])) {
      out.code = lead;
      out.bytes = 1;
      return out;
    }
    code = (code << 6) | tail(data[i]);
  }

  // Избыточно длинная запись, суррогаты и выход за предел — всё это
  // недопустимо. Эталонные реализации такие места буквами не считают, и мы
  // тоже: возвращаем один байт, он уйдёт в прочие символы.
  const uint32_t minimum[5] = {0, 0, 0x80u, 0x800u, 0x10000u};
  if (code < minimum[length] || code > 0x10FFFFu ||
      (code >= 0xD800u && code <= 0xDFFFu)) {
    out.code = lead;
    out.bytes = 1;
    return out;
  }

  out.code = code;
  out.bytes = length;
  out.valid = true;
  return out;
}

void append_utf8(uint32_t code, std::string* out) {
  if (code < 0x80u) {
    out->push_back(static_cast<char>(code));
  } else if (code < 0x800u) {
    out->push_back(static_cast<char>(0xC0u | (code >> 6)));
    out->push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
  } else if (code < 0x10000u) {
    out->push_back(static_cast<char>(0xE0u | (code >> 12)));
    out->push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
    out->push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
  } else {
    out->push_back(static_cast<char>(0xF0u | (code >> 18)));
    out->push_back(static_cast<char>(0x80u | ((code >> 12) & 0x3Fu)));
    out->push_back(static_cast<char>(0x80u | ((code >> 6) & 0x3Fu)));
    out->push_back(static_cast<char>(0x80u | (code & 0x3Fu)));
  }
}

}  // namespace llm
