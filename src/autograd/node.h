// Лента вычислений.
//
// Устройство простое: каждая операция, кроме значения, создаёт узел Node.
// Узел помнит своих входов и хранит замыкание, которое умеет превратить
// градиент по выходу операции в градиенты по её входам. Граф строится сам по
// себе по ходу прямого прохода — отсюда название «лента».
//
// Почему замыкание, а не иерархия классов с виртуальным backward: замыкание
// само захватывает всё, что нужно обратному проходу (сохранённые входы,
// промежуточные величины вроде суммы экспонент в softmax), и не требует
// заводить класс на каждую операцию. Добавление новой операции — одна функция.
//
// Направление ссылок: узел ссылается на своих ВХОДОВ, то есть от результата к
// аргументам. Циклов не возникает, и вся лента освобождается сама, как только
// исчезает последняя ссылка на её вершину.
//
// Var — тензор плюс место в графе. Разделение сознательное: Tensor остаётся
// чистым хранилищем данных, ничего не знающим про градиенты, и его можно
// использовать в инференсе напрямую. Все операции над Var идут через
// autograd/ops.h, а ядра вычислений живут в ops/ и тестируются отдельно.

#ifndef LLM_AUTOGRAD_NODE_H_
#define LLM_AUTOGRAD_NODE_H_

#include <functional>
#include <memory>
#include <vector>

#include "core/tensor.h"

namespace llm {
namespace autograd {

class Node;
using NodePtr = std::shared_ptr<Node>;

// Замыкание обратного прохода: получает градиент по выходу операции и
// добавляет вклады в градиенты входов.
using BackwardFn = std::function<void(const Tensor& grad_output)>;

class Node {
 public:
  // Лист: параметр модели или вход, дальше которого градиент не идёт.
  explicit Node(const char* name) : name_(name) {}

  Node(const char* name, std::vector<NodePtr> inputs, BackwardFn backward)
      : name_(name),
        inputs_(std::move(inputs)),
        backward_(std::move(backward)) {}

  const char* name() const { return name_; }
  const std::vector<NodePtr>& inputs() const { return inputs_; }

  // У листа нет обратного прохода: именно в нём градиент и остаётся, чтобы его
  // забрал оптимизатор.
  bool is_leaf() const { return !backward_; }

  const Tensor& grad() const { return grad_; }
  bool has_grad() const { return grad_.defined(); }

  // Накопление, а не присваивание: если переменная использована в нескольких
  // местах графа, её градиент складывается из вкладов всех потребителей.
  void accumulate(const Tensor& contribution);

  // Перегрузка для временных значений. Обратные проходы почти всегда передают
  // сюда только что посчитанный тензор, и копировать его незачем — буфер можно
  // забрать. Отдельная перегрузка, а не проверка внутри, потому что по
  // значению этого не видно: вызывающий вправе пользоваться своим тензором и
  // после вызова, а накопленный градиент меняется на месте.
  void accumulate(Tensor&& contribution);

  void clear_grad() { grad_ = Tensor(); }

  // Умножает накопленный градиент на множитель. Нужно обрезке градиента по
  // норме: иначе пришлось бы снимать const с буфера градиента.
  void scale_grad(float factor);

  // Узел без накопленного градиента пропускается: он не участвовал в
  // вычислении величины, от которой идёт обратный проход.
  void run_backward() {
    if (backward_ && grad_.defined()) {
      backward_(grad_);
    }
  }

 private:
  const char* name_;
  std::vector<NodePtr> inputs_;
  BackwardFn backward_;
  Tensor grad_;
};

class Var {
 public:
  Var() {}

  // Значение без градиента: константа, вход инференса, буфер.
  static Var constant(Tensor value) { return Var(std::move(value), nullptr); }

  // Лист графа: параметр модели. requires_grad решает, заводить ли узел.
  static Var leaf(Tensor value, bool requires_grad, const char* name = "leaf") {
    return Var(std::move(value),
               requires_grad ? std::make_shared<Node>(name) : nullptr);
  }

  // Результат операции. Вызывается только из autograd/ops.
  static Var from_op(Tensor value, const char* name,
                     std::vector<NodePtr> inputs, BackwardFn backward) {
    return Var(std::move(value), std::make_shared<Node>(name, std::move(inputs),
                                                        std::move(backward)));
  }

  const Tensor& value() const { return value_; }
  Tensor& value() { return value_; }
  const Shape& shape() const { return value_.shape(); }
  int64_t numel() const { return value_.numel(); }

  bool requires_grad() const { return node_ != nullptr; }
  const NodePtr& node() const { return node_; }

  // Накопленный градиент. Неопределённый тензор (grad().defined() == false)
  // означает, что обратный проход сюда не дошёл — например, переменная не
  // участвовала в вычислении функции потерь.
  const Tensor& grad() const;

  void zero_grad() {
    if (node_) {
      node_->clear_grad();
    }
  }

  // Обратный проход от скаляра. Только от скаляра: градиент вектора по вектору
  // — это матрица Якоби, и «градиент» без указания, по какой свёртке, не
  // определён. В обучении корень всегда один — значение функции потерь.
  void backward();

 private:
  Var(Tensor value, NodePtr node)
      : value_(std::move(value)), node_(std::move(node)) {}

  Tensor value_;
  NodePtr node_;
};

// Порядок обработки узлов при обратном проходе: вершина первой, каждый узел —
// раньше своих входов. Вынесено в заголовок ради тестов: правильность этого
// порядка — половина правильности автограда.
std::vector<NodePtr> topological_order(const NodePtr& root);

// Построение ленты можно выключить. В инференсе граф не нужен: он только ест
// память, удерживая все промежуточные значения до конца прямого прохода.
bool grad_enabled();

class NoGradGuard {
 public:
  NoGradGuard();
  ~NoGradGuard();

  NoGradGuard(const NoGradGuard&) = delete;
  NoGradGuard& operator=(const NoGradGuard&) = delete;

 private:
  bool saved_;
};

}  // namespace autograd
}  // namespace llm

#endif  // LLM_AUTOGRAD_NODE_H_
