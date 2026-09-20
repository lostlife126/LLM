#include "train/resume.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include "core/check.h"
#include "serialize/checkpoint.h"

namespace llm {
namespace train {
namespace {

const uint32_t kMagic =
    0x5253554du;  // "MURS" в little-endian: MU Resume Snapshot
// Версия 2: в снимок добавлены параметры прогона (батч, окно, зерно). Старые
// снимки отвергаются с внятным сообщением — это лучше, чем продолжать их
// молча не тем порядком данных.
const uint32_t kVersion = 2;

// Те же метки, что и у чекпоинта: файл, прочитанный на машине с другим
// порядком байт, должен честно отказаться, а не выдать правдоподобный мусор.
const uint32_t kEndianMarker = 0x01020304u;
const float kFloatMarker = 1.5f;

template <typename T>
void write_pod(std::ofstream* file, const T& value) {
  file->write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_pod(std::ifstream* file, const std::string& path) {
  T value;
  file->read(reinterpret_cast<char*>(&value), sizeof(T));
  LLM_CHECK_MSG(file->good(), "снимок " << path << " оборвался при чтении");
  return value;
}

void write_string(std::ofstream* file, const std::string& text) {
  write_pod<uint32_t>(file, static_cast<uint32_t>(text.size()));
  file->write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string read_string(std::ifstream* file, const std::string& path) {
  const uint32_t length = read_pod<uint32_t>(file, path);
  LLM_CHECK_MSG(length < (1u << 20), "неправдоподобная длина имени в " << path);
  std::string text(length, '\0');
  if (length != 0) {
    file->read(&text[0], static_cast<std::streamsize>(length));
  }
  LLM_CHECK_MSG(file->good(), "снимок " << path << " оборвался на имени");
  return text;
}

void write_tensor(std::ofstream* file, const Tensor& tensor) {
  const Tensor dense = tensor.contiguous();
  write_pod<uint32_t>(file, static_cast<uint32_t>(dense.rank()));
  for (int axis = 0; axis < dense.rank(); ++axis) {
    write_pod<int64_t>(file, dense.dim(axis));
  }
  file->write(reinterpret_cast<const char*>(dense.data()),
              static_cast<std::streamsize>(dense.numel() * sizeof(float)));
}

Tensor read_tensor(std::ifstream* file, const std::string& path) {
  const uint32_t rank = read_pod<uint32_t>(file, path);
  LLM_CHECK_MSG(rank <= 8, "неправдоподобный ранг " << rank << " в " << path);
  std::vector<int64_t> dims;
  for (uint32_t axis = 0; axis < rank; ++axis) {
    dims.push_back(read_pod<int64_t>(file, path));
  }
  Tensor out = Tensor::uninitialized(Shape(dims));
  if (out.numel() != 0) {
    file->read(reinterpret_cast<char*>(out.data()),
               static_cast<std::streamsize>(out.numel() * sizeof(float)));
  }
  LLM_CHECK_MSG(file->good(), "снимок " << path << " оборвался на тензоре");
  return out;
}

}  // namespace

std::string RunShape::to_string() const {
  std::ostringstream out;
  out << "батч " << batch_size << " окно " << seq_len << " зерно " << seed;
  return out.str();
}

bool resume_exists(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  std::ifstream file(path.c_str(), std::ios::binary);
  return file.good();
}

void save_resume(const std::string& path, nn::Model* model,
                 const AdamW& optimizer, const ResumeState& state,
                 const RunShape& shape) {
  LLM_CHECK(model != nullptr);
  std::ofstream file(path.c_str(), std::ios::binary);
  LLM_CHECK_MSG(file.good(), "не удалось создать " << path);

  write_pod<uint32_t>(&file, kMagic);
  write_pod<uint32_t>(&file, kVersion);
  write_pod<uint32_t>(&file, kEndianMarker);
  write_pod<float>(&file, kFloatMarker);
  write_string(&file, model->config().to_string());
  write_string(&file, shape.to_string());

  write_pod<int64_t>(&file, state.step);
  write_pod<float>(&file, state.best_validation_loss);
  write_pod<int64_t>(&file, state.best_step);

  std::vector<nn::NamedParameter> parameters = model->parameters();
  const std::vector<Tensor>& first = optimizer.first_moment();
  const std::vector<Tensor>& second = optimizer.second_moment();

  // Оптимизатор ведёт моменты только для обучаемых параметров, а модель знает
  // все. При дообучении адаптерами это разные списки, поэтому в снимок идут
  // оба: веса по именам модели, моменты по именам оптимизатора.
  write_pod<uint32_t>(&file, static_cast<uint32_t>(parameters.size()));
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    write_string(&file, parameters[i].name);
    write_tensor(&file, parameters[i].value->value());
  }

  const std::vector<nn::NamedParameter>& tracked = optimizer.parameters();
  LLM_CHECK_EQ(first.size(), tracked.size());
  LLM_CHECK_EQ(second.size(), tracked.size());
  write_pod<uint32_t>(&file, static_cast<uint32_t>(tracked.size()));
  for (std::size_t i = 0; i < tracked.size(); ++i) {
    write_string(&file, tracked[i].name);
    write_tensor(&file, first[i]);
    write_tensor(&file, second[i]);
  }
  LLM_CHECK_MSG(file.good(), "запись снимка " << path << " не удалась");
}

ResumeState load_resume(const std::string& path, nn::Model* model,
                        AdamW* optimizer, const RunShape& shape) {
  LLM_CHECK(model != nullptr);
  LLM_CHECK(optimizer != nullptr);
  std::ifstream file(path.c_str(), std::ios::binary);
  LLM_CHECK_MSG(file.good(), "не удалось открыть " << path);

  LLM_CHECK_MSG(read_pod<uint32_t>(&file, path) == kMagic,
                path << " не похож на снимок обучения");
  const uint32_t version = read_pod<uint32_t>(&file, path);
  LLM_CHECK_MSG(version == kVersion,
                "снимок версии " << version << ", а мы понимаем " << kVersion);
  LLM_CHECK_MSG(read_pod<uint32_t>(&file, path) == kEndianMarker,
                path << " записан на машине с другим порядком байт");
  LLM_CHECK_MSG(read_pod<float>(&file, path) == kFloatMarker,
                path << " записан с другим представлением float");

  const std::string described = read_string(&file, path);
  LLM_CHECK_MSG(described == model->config().to_string(),
                "снимок описывает другую модель:\n  в файле: "
                    << described
                    << "\n  в памяти: " << model->config().to_string());

  const std::string described_run = read_string(&file, path);
  LLM_CHECK_MSG(described_run == shape.to_string(),
                "снимок сделан с другими параметрами прогона, и порядок "
                "данных не совпадёт:\n  в файле: "
                    << described_run << "\n  сейчас:  " << shape.to_string());

  ResumeState state;
  state.step = read_pod<int64_t>(&file, path);
  state.best_validation_loss = read_pod<float>(&file, path);
  state.best_step = read_pod<int64_t>(&file, path);

  std::vector<nn::NamedParameter> parameters = model->parameters();
  const uint32_t weight_count = read_pod<uint32_t>(&file, path);
  LLM_CHECK_MSG(weight_count == parameters.size(),
                "в снимке " << weight_count << " весов, а модели нужно "
                            << parameters.size());
  for (uint32_t i = 0; i < weight_count; ++i) {
    const std::string name = read_string(&file, path);
    LLM_CHECK_MSG(name == parameters[i].name,
                  "порядок весов разошёлся: в снимке '"
                      << name << "', ожидался '" << parameters[i].name << "'");
    const Tensor value = read_tensor(&file, path);
    Tensor& target = parameters[i].value->value();
    LLM_CHECK_MSG(value.shape() == target.shape(),
                  "у веса " << name << " в снимке другая форма");
    std::memcpy(target.data(), value.data(),
                static_cast<std::size_t>(value.numel()) * sizeof(float));
  }

  const uint32_t moment_count = read_pod<uint32_t>(&file, path);
  const std::vector<nn::NamedParameter>& tracked = optimizer->parameters();
  LLM_CHECK_MSG(moment_count == tracked.size(),
                "в снимке " << moment_count << " моментов, а оптимизатор ведёт "
                            << tracked.size());
  std::vector<Tensor> first;
  std::vector<Tensor> second;
  for (uint32_t i = 0; i < moment_count; ++i) {
    const std::string name = read_string(&file, path);
    LLM_CHECK_MSG(name == tracked[i].name,
                  "порядок моментов разошёлся: в снимке '"
                      << name << "', ожидался '" << tracked[i].name << "'");
    first.push_back(read_tensor(&file, path));
    second.push_back(read_tensor(&file, path));
  }
  optimizer->restore(std::move(first), std::move(second), state.step);
  return state;
}

}  // namespace train
}  // namespace llm
