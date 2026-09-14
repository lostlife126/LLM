// Чтение config.json от HuggingFace.
//
// Файл описывает форму модели, и почти все его поля у нас есть. Опасны не они,
// а те, которых у нас нет. Незнакомое поле, пропущенное молча, даёт модель,
// которая загрузилась, запустилась и выдаёт связный мусор: скажем,
// attention_bias добавляет свободные члены к проекциям, и без них веса
// остаются правильными, а ответы — нет.
//
// Поэтому разбор устроен наоборот обычному: не «прочитать, что понимаем», а
// «убедиться, что всё остальное имеет безопасное значение». Список
// безобидных полей — тех, что не влияют на вычисление, — задан явно, и всё,
// чего нет ни в нём, ни среди читаемых, останавливает загрузку с указанием
// имени поля.

#ifndef LLM_SERIALIZE_HF_CONFIG_H_
#define LLM_SERIALIZE_HF_CONFIG_H_

#include <string>
#include <vector>

#include "nn/config.h"

namespace llm {
namespace serialize {

struct HfConfig {
  nn::ModelConfig model;

  // Что осталось в файле сверх прочитанного. Не ошибка: сюда попадают поля
  // вроде transformers_version, по которым видно, чем модель выложена.
  std::vector<std::string> ignored;

  // Связаны ли эмбеддинги с выходной проекцией. Дублирует поле в model, но
  // нужно отдельно: от него зависит, ждать ли в файле весов lm_head.
  bool tie_embeddings = true;

  // Идентификаторы служебных токенов. Для вычисления не нужны, для генерации
  // нужны: без eos нечем остановиться.
  int64_t bos_token_id = -1;
  int64_t eos_token_id = -1;

  std::string architecture;
};

// max_seq_len берётся не из файла: max_position_embeddings у современных
// моделей доходит до сотен тысяч, а у нас маска и KV-кэш растут квадратом
// длины. Ограничение задаётся снаружи и на веса не влияет — RoPE считает
// позиции на лету, поэтому окно короче объявленного совершенно законно.
HfConfig read_hf_config(const std::string& path, int64_t max_seq_len);

// Разбор из готовой строки — для тестов и для случая, когда файл уже прочитан.
HfConfig parse_hf_config(const std::string& text, const std::string& origin,
                         int64_t max_seq_len);

}  // namespace serialize
}  // namespace llm

#endif  // LLM_SERIALIZE_HF_CONFIG_H_
