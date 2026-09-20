#include "ops/matmul.h"

#include <algorithm>
#include <vector>

#include "core/check.h"
#include "core/thread_pool.h"
#include "ops/gemm.h"

namespace llm {
namespace ops {
namespace {

// Описание двумерного тензора в терминах, которые понимает gemm: указатель,
// шаг строки и флаг транспонирования.
struct MatrixView {
  const float* data;
  int64_t ld;
  bool transposed;
};

// gemm умеет работать с матрицей, у которой единичный шаг по одной из двух
// осей. Это ровно два случая: плотные строки (обычная матрица) и плотные
// столбцы (транспонированный вид). Всё остальное — например, результат
// permute с перемешанными осями — приходится materializovat через
// contiguous().
bool try_as_matrix(const Tensor& tensor, MatrixView* out) {
  LLM_DCHECK_EQ(tensor.rank(), 2);
  const int64_t rows = tensor.dim(0);
  const int64_t cols = tensor.dim(1);
  int64_t row_stride = tensor.stride(0);
  int64_t col_stride = tensor.stride(1);

  // Ось размера 1 не ограничивает размещение: индекс по ней всегда 0, её шаг
  // в вычислении адреса не участвует. Подставляем значения плотной матрицы,
  // иначе такой вид отвергался бы без причины.
  if (rows == 1) {
    row_stride = cols;
  }
  if (cols == 1) {
    col_stride = 1;
  }

  if (col_stride == 1 && row_stride >= cols) {
    out->data = tensor.data();
    out->ld = row_stride;
    out->transposed = false;
    return true;
  }
  if (row_stride == 1 && col_stride >= rows) {
    out->data = tensor.data();
    out->ld = col_stride;
    out->transposed = true;
    return true;
  }
  return false;
}

// Приводит двумерный тензор к виду, пригодному для gemm, при необходимости
// делая плотную копию. Копия возвращается через fallback, чтобы вызывающий
// держал её живой до конца умножения.
MatrixView as_matrix(const Tensor& tensor, Tensor* fallback) {
  MatrixView view;
  if (try_as_matrix(tensor, &view)) {
    return view;
  }
  *fallback = tensor.contiguous();
  const bool ok = try_as_matrix(*fallback, &view);
  LLM_CHECK(ok);
  return view;
}

// Двумерный срез батча. Для матрицы без оси батча возвращает её саму — так
// один и тот же код обслуживает и общий вес, и отдельную матрицу на элемент
// батча.
Tensor batch_slice(const Tensor& tensor, int64_t index) {
  if (tensor.rank() == 2) {
    return tensor;
  }
  return tensor.select(0, tensor.dim(0) == 1 ? 0 : index);
}

// Ниже какого объёма работы делить батч не стоит — тот же порог, что у
// самого gemm, и по той же причине: вход в параллельную область стоит около
// двух микросекунд.
constexpr double kMinParallelFlops = 1.0e5;

int64_t batch_size(const Tensor& tensor) {
  return tensor.rank() == 3 ? tensor.dim(0) : 1;
}

Shape result_shape(const Tensor& a, const Tensor& b) {
  const int64_t m = a.dim(-2);
  const int64_t n = b.dim(-1);
  if (a.rank() == 2 && b.rank() == 2) {
    return Shape{m, n};
  }
  return Shape{std::max(batch_size(a), batch_size(b)), m, n};
}

void check_operands(const Tensor& a, const Tensor& b) {
  LLM_CHECK_MSG(a.rank() == 2 || a.rank() == 3,
                "matmul ожидает ранг 2 или 3, получено " << a.shape());
  LLM_CHECK_MSG(b.rank() == 2 || b.rank() == 3,
                "matmul ожидает ранг 2 или 3, получено " << b.shape());
  LLM_CHECK_MSG(a.dim(-1) == b.dim(-2), "несовместимые формы для matmul: "
                                            << a.shape() << " и " << b.shape());
  const int64_t batch_a = batch_size(a);
  const int64_t batch_b = batch_size(b);
  LLM_CHECK_MSG(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                "несовместимые батчи: " << a.shape() << " и " << b.shape());
}

}  // namespace

void matmul_into(const Tensor& a, const Tensor& b, float alpha, float beta,
                 Tensor* out) {
  LLM_CHECK(out != nullptr);
  check_operands(a, b);
  LLM_CHECK_MSG(out->shape() == result_shape(a, b),
                "out имеет форму " << out->shape() << ", ожидалась "
                                   << result_shape(a, b));
  LLM_CHECK_MSG(out->is_contiguous(), "out должен быть плотным");

  const int64_t m = a.dim(-2);
  const int64_t k = a.dim(-1);
  const int64_t n = b.dim(-1);
  const int64_t batch = std::max(batch_size(a), batch_size(b));

  // Общий вес и плотный вход — это не партия умножений, а одно умножение.
  //
  // У линейного слоя ось батча ничего не значит: строка (b, s) и строка
  // (b', s') умножаются на одну и ту же матрицу, то есть (batch, seq, k) —
  // это batch * seq строк подряд, и плотный тензор такой формы уже лежит в
  // памяти ровно так. Партией умножение становится там, где у каждого
  // элемента батча своя матрица справа: после разворота голов во внимании.
  // Ради этого случая цикл ниже и написан, и для него он и остаётся.
  //
  // Посчитано на шаге обучения пресета nano (батч 16): сюда приходит 24708
  // вызовов, в цикл — 13587, и все они с настоящим батчем матриц. Вызовов с
  // общим весом, которые не прошли бы условие ниже, не оказалось ни одного:
  // требование плотности никого не отсекает, вход линейного слоя плотный
  // всегда.
  //
  // Дробить на batch кусков там, где матрица справа одна, — не решение, а
  // след того, что ось батча пронесли через всю функцию, не спросив, значит
  // ли она что-нибудь. Стоило это трижды: вес упаковывался заново на каждый
  // элемент, блочное разбиение видело задачу в batch раз мельче настоящей, а
  // деление по потокам зависело от оси, которая ни при чём — при batch меньше
  // числа потоков работу делил gemm, при batch не меньше — сама эта функция.
  //
  // Замер на четырёх ядрах, лучшее из пяти серий: выходная проекция nano
  // (батч 16, длина 64) 1.065 -> 0.984 мс, она же у tiny (батч 16, длина 128)
  // 15.71 -> 14.87 мс, FFN вверх у tiny 2.62 -> 2.50 мс. На шаге обучения
  // целиком разницы не видно: три пары запусков вперемежку дали 0.349/0.316/
  // 0.313 против 0.312/0.309/0.341 с — выигрыш тонет во внимании, поэлементных
  // операциях и оптимизаторе. Так что дело здесь не в скорости, а в том, что
  // одно умножение и записано как одно.
  //
  // Инференс это не трогает вовсе: там батч единица, и условие ниже не
  // выполняется.
  //
  // След обучения не меняется: выбор пути в gemm зависит от m, но на
  // обучающих формах обе величины m — и seq, и batch * seq — попадают в одну
  // ветвь. Проверено побитово на пресете nano, батч 8, длина 32, двенадцать
  // шагов: история потерь совпала до последнего разряда.
  if (a.rank() == 3 && batch > 1 && batch_size(b) == 1 && a.is_contiguous()) {
    Tensor shared_copy;
    const MatrixView shared = as_matrix(batch_slice(b, 0), &shared_copy);
    gemm(false, shared.transposed, batch * m, n, k, alpha, a.data(), k,
         shared.data, shared.ld, beta, out->data(), n);
    return;
  }

  const auto multiply = [&](int64_t index) {
    const Tensor a_slice = batch_slice(a, index);
    const Tensor b_slice = batch_slice(b, index);
    Tensor a_copy;
    Tensor b_copy;
    const MatrixView a_view = as_matrix(a_slice, &a_copy);
    const MatrixView b_view = as_matrix(b_slice, &b_copy);

    Tensor out_slice = out->rank() == 3 ? out->select(0, index) : *out;
    gemm(a_view.transposed, b_view.transposed, m, n, k, alpha, a_view.data,
         a_view.ld, b_view.data, b_view.ld, beta, out_slice.data(),
         out_slice.stride(0));
  };

  // Внимание — это не одно большое умножение, а множество мелких: после
  // разворота голов батч равен batch * heads, а каждая матрица имеет форму
  // вроде 64 x 32. По замеру такие умножения составляют девятнадцать из
  // двадцати вызовов gemm за шаг обучения, и делить их внутри бесполезно —
  // каждое слишком мало, чтобы окупить вход в параллельную область.
  //
  // Поэтому делится батч. Условие batch >= width, а не batch > 1: при двух
  // элементах батча и четырёх ядрах выгоднее отдать оба умножения gemm,
  // который займёт все четыре ядра каждым по очереди, чем занять два ядра и
  // оставить два простаивать.
  const int width = parallel_width();
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k) * static_cast<double>(batch);
  if (width > 1 && batch >= width && flops >= kMinParallelFlops &&
      !inside_parallel_region()) {
    parallel_for(width, [&](int task) {
      for (int64_t index = task; index < batch; index += width) {
        multiply(index);
      }
    });
    return;
  }

  for (int64_t index = 0; index < batch; ++index) {
    multiply(index);
  }
}

Tensor matmul_half(const Tensor& a, const HalfMatrix& b) {
  LLM_CHECK_MSG(a.rank() == 2 || a.rank() == 3,
                "matmul_half ожидает ранг 2 или 3, получено " << a.shape());
  LLM_CHECK_MSG(a.dim(-1) == b.rows,
                "несовместимые формы: " << a.shape() << " и матрица "
                                        << b.rows << " x " << b.columns);

  const int64_t m = a.dim(-2);
  const int64_t k = b.rows;
  const int64_t n = b.columns;
  const int64_t batch = a.rank() == 3 ? a.dim(0) : 1;

  Tensor out = Tensor::uninitialized(a.rank() == 2 ? Shape{m, n}
                                                   : Shape{batch, m, n});

  // Здесь матрица справа общая всегда — она одна по самому виду HalfMatrix, —
  // поэтому плотный трёхмерный вход сливается в одно умножение без всяких
  // условий на неё. Причина та же, что в matmul_into выше.
  if (a.rank() == 3 && batch > 1 && a.is_contiguous()) {
    gemm_half_b(false, batch * m, n, k, 1.0f, a.data(), k, b.values, n, 0.0f,
                out.data(), n);
    return out;
  }

  // Вес общий для всего батча и транспонирования не знает: gemm_half_b его и
  // не поддерживает. Поэтому здесь нет ни выбора вида матрицы B, ни запасной
  // плотной копии — только у A.
  const auto multiply = [&](int64_t index) {
    const Tensor a_slice = batch_slice(a, index);
    Tensor a_copy;
    const MatrixView a_view = as_matrix(a_slice, &a_copy);
    Tensor out_slice = out.rank() == 3 ? out.select(0, index) : out;
    gemm_half_b(a_view.transposed, m, n, k, 1.0f, a_view.data, a_view.ld,
                b.values, n, 0.0f, out_slice.data(), out_slice.stride(0));
  };

  // Деление то же и по той же причине, что у matmul_into.
  const int width = parallel_width();
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                       static_cast<double>(k) * static_cast<double>(batch);
  if (width > 1 && batch >= width && flops >= kMinParallelFlops &&
      !inside_parallel_region()) {
    parallel_for(width, [&](int task) {
      for (int64_t index = task; index < batch; index += width) {
        multiply(index);
      }
    });
    return out;
  }

  for (int64_t index = 0; index < batch; ++index) {
    multiply(index);
  }
  return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  check_operands(a, b);
  Tensor out = Tensor::uninitialized(result_shape(a, b));
  matmul_into(a, b, 1.0f, 0.0f, &out);
  return out;
}

}  // namespace ops
}  // namespace llm
