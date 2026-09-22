#include "autograd/node.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "core/check.h"
#include "core/fp16.h"
#include "core/util.h"
#include "ops/elementwise.h"
#include "ops/parallel.h"

namespace llm {
namespace autograd {
namespace {

bool g_grad_enabled = true;

// Переменная окружения читается один раз при первом обращении и дальше живёт
// обычной переменной: getenv на каждой операции был бы заметен.
//
// Именно при первом обращении, а не при статической инициализации. Разбор
// строгий и на непонятном значении падает, а падение до входа в main — это
// std::terminate, где от сообщения остаётся то, что успеет напечатать
// обработчик по умолчанию.
bool& fp16_flag() {
  static bool value = env_flag("LLM_FP16");
  return value;
}

bool& fp16_weight_flag() {
  static bool value = env_flag("LLM_FP16_WEIGHTS");
  return value;
}

}  // namespace

bool fp16_simulation() { return fp16_flag(); }

bool fp16_weight_simulation() { return fp16_weight_flag(); }

void set_fp16_simulation(bool enabled) { fp16_flag() = enabled; }

void apply_fp16_simulation(Tensor* tensor) {
  if (!fp16_flag() || tensor == nullptr || !tensor->defined()) {
    return;
  }
  // Только плотные. Вид на чужой буфер сюда не попадает: операции, которые
  // его возвращают, пользуются Var::from_op_view и округление пропускают —
  // см. комментарий там.
  //
  // Условие «единоличного владения» здесь стояло и оказалось неверным. Оно
  // отличало не вид от свежего результата, а «на буфер смотрит один» от «на
  // буфер смотрят двое», — а операция вправе сохранить собственный выход для
  // обратного прохода, и тогда смотрящих двое. Так под условие попали softmax
  // и masked_softmax, то есть самые горячие активации модели: имитация
  // молча переставала их округлять, и замер сходимости вышел бы завышенным.
  if (!tensor->is_contiguous()) {
    return;
  }
  float* data = tensor->data();
  const int64_t count = tensor->numel();
  for (int64_t i = 0; i < count; ++i) {
    data[i] = round_to_fp16(data[i]);
  }
}

void Node::accumulate(Tensor&& contribution) {
  // Забрать буфер можно при двух условиях сразу. Временное значение — этого
  // требует сама перегрузка. И единоличное владение: временным бывает и вид на
  // чужой буфер (reshape, transpose), а он делит хранилище с живым тензором, и
  // запись в накопленный градиент испортила бы чужие данные.
  if (!grad_.defined() && contribution.is_contiguous() &&
      contribution.owns_whole_storage()) {
    grad_ = std::move(contribution);
    apply_fp16_simulation(&grad_);
    return;
  }
  accumulate(static_cast<const Tensor&>(contribution));
}

void Node::accumulate(const Tensor& contribution) {
  if (!grad_.defined()) {
    // Первый вклад: забираем копию. Именно копию, а не вид — вкладчик может
    // передать временный тензор или чужой буфер.
    grad_ = contribution.clone();
    apply_fp16_simulation(&grad_);
    return;
  }
  LLM_CHECK_MSG(grad_.shape() == contribution.shape(),
                "вклад формы " << contribution.shape() << " в градиент формы "
                               << grad_.shape() << " (узел " << name_ << ")");
  // Прибавка идёт через ops::add_into: там обход кусками и деление по
  // потокам, а накопление градиента — одно из самых частых действий обратного
  // прохода.
  ops::add_into(contribution, &grad_);
  // Накопленный градиент — то, что в настоящем fp16 лежало бы в fp16, поэтому
  // округляется после каждого вклада, а не однажды в конце. Именно здесь и
  // проваливаются малые градиенты, ради проверки чего всё это написано.
  apply_fp16_simulation(&grad_);
}

void Node::scale_grad(float factor) {
  if (!grad_.defined()) {
    return;
  }
  float* data = grad_.data();
  // Обрезка нормы вызывает это для каждого параметра, то есть проход по всем
  // градиентам модели. Делится так же, как поэлементные операции.
  ops::for_elements(grad_.numel(),
                    [&](int64_t element) { data[element] *= factor; });
}

const Tensor& Var::grad() const {
  LLM_CHECK_MSG(node_ != nullptr,
                "запрошен градиент переменной, которая его не требует");
  return node_->grad();
}

std::vector<NodePtr> topological_order(const NodePtr& root) {
  // Обратный постпорядок обхода в глубину — это топологический порядок:
  // каждый узел оказывается раньше всех своих входов. Значит, к моменту его
  // обработки все потребители уже внесли свои вклады в его градиент, и
  // накопленный градиент полон.
  //
  // Обход итеративный, а не рекурсивный: в графе обучения цепочка узлов
  // длиной в сотни слоёв и шагов — реальный случай, а глубина стека
  // ограничена.
  std::vector<NodePtr> postorder;
  if (!root) {
    return postorder;
  }

  std::unordered_set<const Node*> visited;
  // Второй элемент пары — признак того, что входы узла уже разложены на стек
  // и узел пора записать в постпорядок.
  std::vector<std::pair<NodePtr, bool>> stack;
  stack.push_back(std::make_pair(root, false));

  while (!stack.empty()) {
    NodePtr node = stack.back().first;
    const bool expanded = stack.back().second;
    stack.pop_back();

    if (expanded) {
      postorder.push_back(node);
      continue;
    }
    if (visited.count(node.get()) != 0) {
      continue;
    }
    visited.insert(node.get());

    stack.push_back(std::make_pair(node, true));
    for (std::size_t i = 0; i < node->inputs().size(); ++i) {
      const NodePtr& input = node->inputs()[i];
      if (input && visited.count(input.get()) == 0) {
        stack.push_back(std::make_pair(input, false));
      }
    }
  }

  std::reverse(postorder.begin(), postorder.end());
  return postorder;
}

void Var::backward() {
  LLM_CHECK_MSG(node_ != nullptr,
                "backward() от переменной, которая не требует градиента");
  LLM_CHECK_MSG(value_.numel() == 1,
                "backward() допустим только от скаляра, получена форма "
                    << value_.shape());

  // Затравка: производная скаляра по самому себе равна единице.
  node_->accumulate(Tensor::full(value_.shape(), 1.0f));

  const std::vector<NodePtr> order = topological_order(node_);
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i]->run_backward();
    // Градиент промежуточного узла больше не нужен: топологический порядок
    // гарантирует, что вкладов в него уже не будет. Освобождение здесь заметно
    // экономит память на длинной ленте. У листьев градиент остаётся — за ним
    // и придёт оптимизатор.
    if (!order[i]->is_leaf()) {
      order[i]->clear_grad();
    }
  }
}

bool grad_enabled() { return g_grad_enabled; }

NoGradGuard::NoGradGuard() : saved_(g_grad_enabled) { g_grad_enabled = false; }

NoGradGuard::~NoGradGuard() { g_grad_enabled = saved_; }

}  // namespace autograd
}  // namespace llm
