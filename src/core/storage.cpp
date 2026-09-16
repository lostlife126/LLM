#include "core/storage.h"

#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace llm {
namespace {

// Пул освобождённых буферов.
//
// Зачем он нужен. Шаг обучения выделяет и освобождает четыре сотни буферов
// общим объёмом больше гигабайта у пресета tiny, и на каждом шаге это одни и
// те же размеры: форма модели не меняется. Крупные блоки аллокатор отдаёт
// ядру, а следующее выделение получает свежие страницы, которые размечаются по
// первому обращению. Замер: 13 тысяч ошибок страниц за шаг у nano и 103 тысячи
// у tiny — при стоимости около микросекунды каждая это десятая-четвёртая часть
// шага, потраченная не на счёт, а на раздачу памяти.
//
// Пул держит освобождённые блоки у себя и отдаёт обратно по точному размеру.
// Точному, а не «не меньше»: размеры повторяются от шага к шагу в точности, и
// разбиение на классы только испортило бы попадание.
//
// Условие, которое нельзя нарушать: ни один объект со статическим временем
// жизни и ни один thread_local не вправе держать Tensor. Пул у потока —
// thread_local, и разрушается он при выходе потока; освобождение буфера после
// этого обратилось бы к уже разрушенному пулу. Сейчас условие соблюдается:
// статических тензоров в проекте нет, а thread_local держат только
// std::vector<float>, которые с пулом не связаны.
//
// Пул у каждого потока свой, поэтому блокировок нет вовсе. Буфер, выделенный
// одним потоком и освобождённый другим, просто переезжает в пул второго — на
// правильность это не влияет, а на попадание влияет мало: тензоры почти всегда
// выделяются в потоке, который ведёт шаг.
class BufferPool {
 public:
  ~BufferPool() {
    for (Map::iterator it = free_.begin(); it != free_.end(); ++it) {
      for (std::size_t i = 0; i < it->second.size(); ++i) {
        delete[] it->second[i];
      }
    }
  }

  unsigned char* take(std::size_t capacity) {
    const Map::iterator it = free_.find(capacity);
    if (it == free_.end() || it->second.empty()) {
      return new unsigned char[capacity];
    }
    unsigned char* block = it->second.back();
    it->second.pop_back();
    held_ -= capacity;
    return block;
  }

  void give(unsigned char* block, std::size_t capacity) {
    if (block == nullptr) {
      return;
    }
    // Потолок нужен, чтобы пул не превращался в течь при задачах, где размеры
    // всё время разные. Сверх потолка блок отдаётся обратно аллокатору.
    if (held_ + capacity > kMaxHeldBytes) {
      delete[] block;
      return;
    }
    free_[capacity].push_back(block);
    held_ += capacity;
  }

 private:
  typedef std::map<std::size_t, std::vector<unsigned char*> > Map;

  // Сколько байт пул готов держать. Пик живой памяти у tiny — около 450 МБ,
  // и потолок выбран с запасом над ним: пул должен переживать шаг целиком,
  // иначе он не поможет там, где помощь нужна.
  static const std::size_t kMaxHeldBytes = 1024u * 1024u * 1024u;

  Map free_;
  std::size_t held_ = 0;
};

BufferPool& pool() {
  static thread_local BufferPool instance;
  return instance;
}

}  // namespace

const std::size_t Storage::kAlignment;

Storage::Storage(std::size_t nbytes)
    : raw_(nullptr), aligned_(nullptr), nbytes_(nbytes) {
  if (nbytes == 0) {
    return;
  }
  // Запас в kAlignment - 1 байт гарантирует, что внутри выделенного блока
  // найдётся выровненный адрес, с которого помещается nbytes байт.
  const std::size_t capacity = nbytes + kAlignment - 1;
  raw_ = pool().take(capacity);

  void* cursor = raw_;
  std::size_t space = capacity;
  aligned_ = std::align(kAlignment, nbytes, cursor, space);
  LLM_CHECK_MSG(
      aligned_ != nullptr,
      "std::align не нашла выровненный адрес для " << nbytes << " байт");
}

Storage::~Storage() { release(); }

Storage::Storage(Storage&& other)
    : raw_(other.raw_), aligned_(other.aligned_), nbytes_(other.nbytes_) {
  other.raw_ = nullptr;
  other.aligned_ = nullptr;
  other.nbytes_ = 0;
}

Storage& Storage::operator=(Storage&& other) {
  if (this != &other) {
    release();
    raw_ = other.raw_;
    aligned_ = other.aligned_;
    nbytes_ = other.nbytes_;
    other.raw_ = nullptr;
    other.aligned_ = nullptr;
    other.nbytes_ = 0;
  }
  return *this;
}

void Storage::zero() {
  if (nbytes_ != 0) {
    std::memset(aligned_, 0, nbytes_);
  }
}

void Storage::release() {
  if (raw_ != nullptr) {
    pool().give(raw_, nbytes_ + kAlignment - 1);
  }
  raw_ = nullptr;
  aligned_ = nullptr;
  nbytes_ = 0;
}

}  // namespace llm
