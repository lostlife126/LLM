// Чтение файла целиком — одно на все приложения.
//
// Было скопировано в пяти программах слово в слово.
//
// Читается сразу в строку, а не через ostringstream. Через поток содержимое
// оказывалось в памяти дважды: сначала во внутреннем буфере потока, потом в
// строке, которую возвращает str(). Измерено на файле 191 МБ: пик 384 МБ
// против 192 МБ, то есть ровно вдвое. Это та же правка, что уже сделана в
// чтении safetensors, и по той же причине — корпус бывает больше игрушечного.

#ifndef LLM_APPS_READ_FILE_H_
#define LLM_APPS_READ_FILE_H_

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <string>

namespace bench {

inline std::string read_file(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::binary);
  if (!file.good()) {
    std::fprintf(stderr, "не удалось открыть %s\n", path.c_str());
    std::exit(1);
  }
  file.seekg(0, std::ios::end);
  const std::streamoff size = file.tellg();
  // Два случая, а не один. Пустой файл — не ошибка: читать нечего, и дальше
  // об этом скажет тот, кому пустой корпус не подходит. А отрицательное
  // значение означает, что перемещение или опрос позиции НЕ УДАЛИСЬ, и
  // выдавать это за пустой файл нельзя: так выглядит, например, каталог,
  // отданный вместо файла, — ifstream его открывает, а размера у него нет.
  // Сообщение при этом получалось про пустой корпус, то есть указывало не
  // туда.
  if (size < 0) {
    std::fprintf(stderr, "не удалось узнать размер %s — это точно файл?\n",
                 path.c_str());
    std::exit(1);
  }
  if (size == 0) {
    return std::string();
  }
  file.seekg(0, std::ios::beg);
  std::string text(static_cast<std::size_t>(size), '\0');
  file.read(&text[0], size);
  if (static_cast<std::streamoff>(file.gcount()) != size) {
    std::fprintf(stderr, "%s прочитан не целиком\n", path.c_str());
    std::exit(1);
  }
  return text;
}

}  // namespace bench

#endif  // LLM_APPS_READ_FILE_H_
