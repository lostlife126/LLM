// Замер альтернативы для Q*K^T: ядро без упаковки.
//
// При transpose_b обе матрицы лежат построчно по k — и Q[i][p], и K[j][p]
// непрерывны. Значит ядро может читать обе напрямую, а упаковка K становится
// не нужна. Платить за это приходится горизонтальной свёрткой: плитка
// накапливается векторами частичных сумм, и в конце каждый вектор надо
// свернуть в одно число.
//
// Ядро здесь ни на что не влияет: библиотека его не вызывает, оно живёт
// только ради этого замера. Держим его в проекте, потому что вывод — «эта
// форма проигрывает втрое» — стоит того, чтобы его можно было перепроверить,
// а не только прочитать. Разбор в README, раздел «Q*K^T без упаковки».
//
// Запуск: ./bench_qk
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define LLM_QK_X86 1
#else
#define LLM_QK_X86 0
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "measure.h"

#include "core/cpu.h"
#include "core/thread_pool.h"
#include "ops/gemm.h"

#if LLM_QK_X86

namespace {
// Свёртка шестнадцати аккумуляторов в один вектор из шестнадцати сумм.
// Дерево из четырёх уровней: пары, четвёрки по 64 бита, потом 128-битные
// дорожки. Сорок пять операций на шестнадцать результатов.
__attribute__((target("avx512f")))
__m512 reduce16(const __m512* a) {
  __m512 s[8];
  for (int i = 0; i < 8; ++i) {
    s[i] = _mm512_add_ps(_mm512_unpacklo_ps(a[2 * i], a[2 * i + 1]),
                         _mm512_unpackhi_ps(a[2 * i], a[2 * i + 1]));
  }
  __m512 v[4];
  for (int i = 0; i < 4; ++i) {
    const __m512d x = _mm512_castps_pd(s[2 * i]);
    const __m512d y = _mm512_castps_pd(s[2 * i + 1]);
    v[i] = _mm512_add_ps(_mm512_castpd_ps(_mm512_unpacklo_pd(x, y)),
                         _mm512_castpd_ps(_mm512_unpackhi_pd(x, y)));
  }
  __m512 q[2];
  for (int i = 0; i < 2; ++i) {
    q[i] = _mm512_add_ps(
        _mm512_shuffle_f32x4(v[2 * i], v[2 * i + 1], 0x44),
        _mm512_shuffle_f32x4(v[2 * i], v[2 * i + 1], 0xee));
  }
  return _mm512_add_ps(_mm512_shuffle_f32x4(q[0], q[1], 0x88),
                       _mm512_shuffle_f32x4(q[0], q[1], 0xdd));
}

__attribute__((target("avx512f")))
bool reduce16_self_check() {
  __m512 acc[16];
  for (int t = 0; t < 16; ++t) {
    acc[t] = _mm512_set1_ps(static_cast<float>(t + 1));
  }
  float got[16];
  _mm512_storeu_ps(got, reduce16(acc));
  for (int t = 0; t < 16; ++t) {
    const float want = 16.0f * static_cast<float>(t + 1);
    if (got[t] != want) {
      std::printf("дерево свёртки неверно: дорожка %d дала %.1f вместо %.1f\n",
                  t, got[t], want);
      return false;
    }
  }
  std::printf("дерево свёртки: сходится\n");
  return true;
}

// C = A * B^T, обе построчно. Плитка 1 x 16: одна строка A, шестнадцать строк B.
__attribute__((target("avx512f")))
void gemm_dot(int64_t m, int64_t n, int64_t k, float alpha, const float* a,
              int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc) {
  const __m512 scale = _mm512_set1_ps(alpha);
  for (int64_t i = 0; i < m; ++i) {
    const float* arow = a + i * lda;
    for (int64_t j = 0; j < n; j += 16) {
      __m512 acc[16];
      for (int t = 0; t < 16; ++t) acc[t] = _mm512_setzero_ps();
      for (int64_t p = 0; p < k; p += 16) {
        const __m512 av = _mm512_loadu_ps(arow + p);
        for (int t = 0; t < 16; ++t) {
          acc[t] = _mm512_fmadd_ps(av, _mm512_loadu_ps(b + (j + t) * ldb + p),
                                   acc[t]);
        }
      }
      _mm512_storeu_ps(c + i * ldc + j, _mm512_mul_ps(reduce16(acc), scale));
    }
  }
}

void gemm_ref(int64_t m, int64_t n, int64_t k, float alpha, const float* a,
              int64_t lda, const float* b, int64_t ldb, float* c, int64_t ldc) {
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int64_t p = 0; p < k; ++p) sum += a[i * lda + p] * b[j * ldb + p];
      c[i * ldc + j] = static_cast<float>(alpha * sum);
    }
  }
}

struct Case {
  const char* label;
  int64_t heads;
  int64_t m;
  int64_t n;
  int64_t k;
};

// Замер — общий, см. apps/measure.h.
template <typename Fn>
double best_gflops(double flops, Fn fn) {
  return flops / bench::best_seconds(fn) / 1e9;
}

}  // namespace

#else  // LLM_QK_X86

// Ядро здесь написано под AVX-512 и никуда, кроме x86, не переносилось: оно
// существует ради одного вывода про форму вычисления, а не ради работы. На
// других архитектурах замер просто не проводится.

#endif  // LLM_QK_X86

int main() {
#if !LLM_QK_X86
  std::printf("замер написан под AVX-512 и на этой архитектуре не работает\n");
  return 0;
#else
  // Ядро написано только под AVX-512 — оно здесь не для переносимости, а для
  // одного вопроса. Без AVX-512 замер просто не проводится.
  if (!llm::cpu_features().has_avx512()) {
    std::printf("на этой машине нет AVX-512, замер пропущен\n");
    return 0;
  }
  llm::set_parallel_width(1);

  // Сверка самого дерева свёртки на известных числах: аккумулятор t заполнен
  // числом t + 1, значит сумма его шестнадцати дорожек равна 16 * (t + 1).
  if (!reduce16_self_check()) {
    return 1;
  }

  // Сверка ядра с эталоном на двойной точности. Порядок суммирования другой,
  // поэтому сравнение не побитовое; норма — масштаб всего результата, а не
  // модуль отдельного элемента: скалярные произведения проходят через ноль, и
  // около нуля относительная ошибка взрывается при любой перестановке.
  {
    const int64_t m = 17, n = 32, k = 64;
    std::vector<float> a(m * k), b(n * k), c1(m * n), c2(m * n);
    for (size_t i = 0; i < a.size(); ++i) a[i] = std::sin(0.3 * i);
    for (size_t i = 0; i < b.size(); ++i) b[i] = std::cos(0.2 * i);
    gemm_ref(m, n, k, 0.5f, a.data(), k, b.data(), k, c1.data(), n);
    gemm_dot(m, n, k, 0.5f, a.data(), k, b.data(), k, c2.data(), n);
    double scale_ref = 0.0;
    for (int64_t i = 0; i < m * n; ++i) {
      if (std::fabs(c1[i]) > scale_ref) scale_ref = std::fabs(c1[i]);
    }
    double worst = 0.0;
    for (int64_t i = 0; i < m * n; ++i) {
      const double d = std::fabs(c1[i] - c2[i]) / scale_ref;
      if (d > worst) worst = d;
    }
    std::printf("сверка с эталоном: худшее отклонение %.2e от масштаба\n\n",
                worst);
    if (worst > 1e-6) {
      std::printf("ЯДРО НЕВЕРНО, замер бессмысленен\n");
      return 1;
    }
  }

  const Case cases[] = {
      {"nano  Q*K^T", 16 * 4 * 4, 64, 64, 32},
      {"tiny  Q*K^T", 16 * 8 * 4, 128, 128, 32},
      {"small Q*K^T", 16 * 6 * 6, 256, 256, 64},
  };
  std::printf("%-12s %10s %10s %8s\n", "форма", "упаковка", "скалярн.", "отн.");
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const Case& c = cases[i];
    std::vector<float> a(c.heads * c.m * c.k, 0.5f);
    std::vector<float> b(c.heads * c.n * c.k, 0.25f);
    std::vector<float> out(c.heads * c.m * c.n, 0.0f);
    const double flops = 2.0 * c.heads * c.m * c.n * c.k;
    const double packed = best_gflops(flops, [&]() {
      for (int64_t h = 0; h < c.heads; ++h) {
        llm::ops::gemm(false, true, c.m, c.n, c.k, 1.0f,
                       a.data() + h * c.m * c.k, c.k, b.data() + h * c.n * c.k,
                       c.k, 0.0f, out.data() + h * c.m * c.n, c.n);
      }
    });
    const double dot = best_gflops(flops, [&]() {
      for (int64_t h = 0; h < c.heads; ++h) {
        gemm_dot(c.m, c.n, c.k, 1.0f, a.data() + h * c.m * c.k, c.k,
                 b.data() + h * c.n * c.k, c.k, out.data() + h * c.m * c.n,
                 c.n);
      }
    });
    std::printf("%-12s %10.1f %10.1f %7.2fx\n", c.label, packed, dot,
                dot / packed);
    std::fflush(stdout);
  }
  return 0;
#endif  // LLM_QK_X86
}
