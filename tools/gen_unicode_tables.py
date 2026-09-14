#!/usr/bin/env python3
"""Генератор таблиц категорий Unicode для предтокенизации.

Зачем он есть. Предтокенизаторы чужих моделей задаются регулярным выражением
с классами \\p{L} и \\p{N} — «любая буква», «любая цифра». std::regex не
понимает их ни в одной версии стандарта (проверено вплоть до C++20: \\p{L}
отвергается как ошибка, а [[:alpha:]] берёт только ASCII). Значит нужны сами
диапазоны.

Выписывать их руками нельзя — их под восемьсот. Поэтому таблица генерируется
из базы Unicode, которая идёт в составе Python, и результат кладётся в
репозиторий. Во время сборки и работы никаких зависимостей нет: в проекте
лежит обычный C++.

Запуск: python3 tools/gen_unicode_tables.py > src/tokenizer/unicode_tables.cpp
"""

import unicodedata


def ranges(predicate):
    out = []
    start = None
    for code in range(0x110000):
        if predicate(code):
            if start is None:
                start = code
        elif start is not None:
            out.append((start, code - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def emit(name, table):
    print("const CodeRange k%s[] = {" % name)
    line = "   "
    for low, high in table:
        piece = " {0x%05X, 0x%05X}," % (low, high)
        if len(line) + len(piece) > 78:
            print(line)
            line = "   "
        line += piece
    if line.strip():
        print(line)
    print("};")
    print("")


letters = ranges(lambda c: unicodedata.category(chr(c)).startswith("L"))
numbers = ranges(lambda c: unicodedata.category(chr(c)).startswith("N"))

print("// Сгенерировано tools/gen_unicode_tables.py — руками не править.")
print("//")
print("// База Unicode %s (модуль unicodedata из Python)." % unicodedata.unidata_version)
print("// Диапазонов букв: %d, цифр: %d." % (len(letters), len(numbers)))
print("")
print('#include "tokenizer/unicode_tables.h"')
print("")
print("#include <cstddef>")
print("")
print("namespace llm {")
print("namespace {")
print("")
print("struct CodeRange {")
print("  uint32_t low;")
print("  uint32_t high;")
print("};")
print("")
emit("Letters", letters)
emit("Numbers", numbers)
print("bool in_table(const CodeRange* table, std::size_t count, uint32_t code) {")
print("  // Двоичный поиск: диапазоны не пересекаются и идут по возрастанию.")
print("  std::size_t low = 0;")
print("  std::size_t high = count;")
print("  while (low < high) {")
print("    const std::size_t middle = low + (high - low) / 2;")
print("    if (code < table[middle].low) {")
print("      high = middle;")
print("    } else if (code > table[middle].high) {")
print("      low = middle + 1;")
print("    } else {")
print("      return true;")
print("    }")
print("  }")
print("  return false;")
print("}")
print("")
print("}  // namespace")
print("")
print("bool is_unicode_letter(uint32_t code) {")
print("  return in_table(kLetters, sizeof(kLetters) / sizeof(kLetters[0]), code);")
print("}")
print("")
print("bool is_unicode_number(uint32_t code) {")
print("  return in_table(kNumbers, sizeof(kNumbers) / sizeof(kNumbers[0]), code);")
print("}")
print("")
print("}  // namespace llm")
