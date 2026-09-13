#include "autograd/node.h"

#include <algorithm>
#include <unordered_set>

#include "core/check.h"
#include "core/iterate.h"

namespace llm {
namespace autograd {
namespace {

bool g_grad_enabled = true;

}  // namespace

void Node::accumulate(const Tensor& contribution) {
  if (!grad_.defined()) {
    // Первый вклад: забираем копию. Именно копию, а не вид — вкладчик может
    // передать временный тензор или чужой буфер.
    grad_ = contribution.clone();
    return;
  }
  LLM_CHECK_MSG(grad_.shape() == contribution.shape(),
                "вклад формы " << contribution.shape() << " в градиент формы "
                               << grad_.shape() << " (узел " << name_ << ")");
  float* target = grad_.data();
  const float* source = contribution.data();
  for_each_offset2(
      grad_.shape(), grad_.strides(), contribution.strides(),
      [&](int64_t to, int64_t from) { target[to] += source[from]; });
}

void Node::scale_grad(float factor) {
  if (!grad_.defined()) {
    return;
  }
  float* data = grad_.data();
  const int64_t count = grad_.numel();
  for (int64_t element = 0; element < count; ++element) {
    data[element] *= factor;
  }
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
