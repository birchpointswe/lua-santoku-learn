#ifndef TK_LEARN_MATHLIBS_H
#define TK_LEARN_MATHLIBS_H

#if defined(_OPENMP) && !defined(__EMSCRIPTEN__)
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

#if !defined(__EMSCRIPTEN__)
#include <sys/mman.h>
#endif

#ifdef __EMSCRIPTEN__

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };

#define LAPACK_COL_MAJOR 102

static inline float cblas_sdot (int n, const float *x, int incx, const float *y, int incy) {
  float s = 0;
  for (int i = 0; i < n; i++)
    s += x[i * incx] * y[i * incy];
  return s;
}

static inline float cblas_snrm2 (int n, const float *x, int incx) {
  float s = 0;
  for (int i = 0; i < n; i++)
    s += x[i * incx] * x[i * incx];
  return sqrtf(s);
}

static inline double cblas_dnrm2 (int n, const double *x, int incx) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += x[i * incx] * x[i * incx];
  return sqrt(s);
}

static inline void cblas_dgemv (enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
    int m, int n, double alpha, const double *A, int lda,
    const double *x, int incx, double beta, double *y, int incy) {
  int rows = (trans == CblasNoTrans) ? m : n;
  int cols = (trans == CblasNoTrans) ? n : m;
  for (int i = 0; i < rows; i++) {
    double s = 0;
    for (int j = 0; j < cols; j++) {
      double a;
      if (order == CblasRowMajor)
        a = (trans == CblasNoTrans) ? A[i * lda + j] : A[j * lda + i];
      else
        a = (trans == CblasNoTrans) ? A[j * lda + i] : A[i * lda + j];
      s += a * x[j * incx];
    }
    y[i * incy] = alpha * s + beta * y[i * incy];
  }
}

static inline void cblas_sgemm (enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA,
    enum CBLAS_TRANSPOSE transB, int M, int N, int K,
    float alpha, const float *A, int lda, const float *B, int ldb,
    float beta, float *C, int ldc) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float s = 0;
      for (int k = 0; k < K; k++) {
        float a, b;
        if (order == CblasRowMajor) {
          a = (transA == CblasNoTrans) ? A[i * lda + k] : A[k * lda + i];
          b = (transB == CblasNoTrans) ? B[k * ldb + j] : B[j * ldb + k];
        } else {
          a = (transA == CblasNoTrans) ? A[k * lda + i] : A[i * lda + k];
          b = (transB == CblasNoTrans) ? B[j * ldb + k] : B[k * ldb + j];
        }
        s += a * b;
      }
      if (order == CblasRowMajor)
        C[i * ldc + j] = alpha * s + beta * C[i * ldc + j];
      else
        C[j * ldc + i] = alpha * s + beta * C[j * ldc + i];
    }
  }
}

static inline void cblas_dgemm (enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transA,
    enum CBLAS_TRANSPOSE transB, int M, int N, int K,
    double alpha, const double *A, int lda, const double *B, int ldb,
    double beta, double *C, int ldc) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double s = 0;
      for (int k = 0; k < K; k++) {
        double a, b;
        if (order == CblasRowMajor) {
          a = (transA == CblasNoTrans) ? A[i * lda + k] : A[k * lda + i];
          b = (transB == CblasNoTrans) ? B[k * ldb + j] : B[j * ldb + k];
        } else {
          a = (transA == CblasNoTrans) ? A[k * lda + i] : A[i * lda + k];
          b = (transB == CblasNoTrans) ? B[j * ldb + k] : B[k * ldb + j];
        }
        s += a * b;
      }
      if (order == CblasRowMajor)
        C[i * ldc + j] = alpha * s + beta * C[i * ldc + j];
      else
        C[j * ldc + i] = alpha * s + beta * C[j * ldc + i];
    }
  }
}

static inline void cblas_dsyrk (enum CBLAS_ORDER order, enum CBLAS_UPLO uplo,
    enum CBLAS_TRANSPOSE trans, int N, int K,
    double alpha, const double *A, int lda, double beta, double *C, int ldc) {
  for (int i = 0; i < N; i++) {
    for (int j = (uplo == CblasUpper ? i : 0); j < (uplo == CblasUpper ? N : i + 1); j++) {
      double s = 0;
      for (int k = 0; k < K; k++) {
        double ai, aj;
        if (order == CblasColMajor) {
          ai = (trans == CblasTrans) ? A[i * lda + k] : A[k * lda + i];
          aj = (trans == CblasTrans) ? A[j * lda + k] : A[k * lda + j];
        } else {
          ai = (trans == CblasTrans) ? A[k * lda + i] : A[i * lda + k];
          aj = (trans == CblasTrans) ? A[k * lda + j] : A[j * lda + k];
        }
        s += ai * aj;
      }
      if (order == CblasColMajor)
        C[j * ldc + i] = alpha * s + beta * C[j * ldc + i];
      else
        C[i * ldc + j] = alpha * s + beta * C[i * ldc + j];
    }
  }
}

static inline void cblas_dsyr (enum CBLAS_ORDER order, enum CBLAS_UPLO uplo,
    int N, double alpha, const double *x, int incx, double *A, int lda) {
  for (int i = 0; i < N; i++) {
    for (int j = (uplo == CblasUpper ? i : 0); j < (uplo == CblasUpper ? N : i + 1); j++) {
      if (order == CblasColMajor)
        A[j * lda + i] += alpha * x[i * incx] * x[j * incx];
      else
        A[i * lda + j] += alpha * x[i * incx] * x[j * incx];
    }
  }
}

static inline void cblas_dger (enum CBLAS_ORDER order,
    int M, int N, double alpha, const double *x, int incx,
    const double *y, int incy, double *A, int lda) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      if (order == CblasRowMajor)
        A[i * lda + j] += alpha * x[i * incx] * y[j * incy];
      else
        A[j * lda + i] += alpha * x[i * incx] * y[j * incy];
    }
}

static inline void cblas_strsm (enum CBLAS_ORDER order, enum CBLAS_SIDE side,
    enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE trans, enum CBLAS_DIAG diag,
    int M, int N, float alpha, const float *A, int lda, float *B, int ldb) {
  (void)order; (void)side; (void)uplo; (void)trans; (void)diag;
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < M; i++)
      B[i * ldb + j] *= alpha;
    if (side == CblasLeft && uplo == CblasLower && trans == CblasTrans) {
      for (int i = M - 1; i >= 0; i--) {
        if (diag == CblasNonUnit)
          B[i * ldb + j] /= A[i * lda + i];
        for (int k = 0; k < i; k++)
          B[k * ldb + j] -= A[i * lda + k] * B[i * ldb + j];
      }
    } else if (side == CblasLeft && uplo == CblasUpper && trans == CblasNoTrans) {
      for (int i = M - 1; i >= 0; i--) {
        if (diag == CblasNonUnit)
          B[i * ldb + j] /= A[i * lda + i];
        for (int k = 0; k < i; k++)
          B[k * ldb + j] -= A[k * lda + i] * B[i * ldb + j];
      }
    } else if (side == CblasLeft && uplo == CblasLower && trans == CblasNoTrans) {
      for (int i = 0; i < M; i++) {
        if (diag == CblasNonUnit)
          B[i * ldb + j] /= A[i * lda + i];
        for (int k = i + 1; k < M; k++)
          B[k * ldb + j] -= A[k * lda + i] * B[i * ldb + j];
      }
    }
  }
}

static inline void tk_jacobi_rotate (double *A, int n, int p, int q, double c, double s) {
  for (int r = 0; r < n; r++) {
    double ap = A[r * n + p], aq = A[r * n + q];
    A[r * n + p] = c * ap - s * aq;
    A[r * n + q] = s * ap + c * aq;
  }
}

static inline int LAPACKE_dsyevd (int matrix_layout, char jobz, char uplo,
    int n, double *A, int lda, double *w) {
  (void)matrix_layout; (void)uplo;
  double *V = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  if (!V) return -1;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      V[i * n + j] = (i == j) ? 1.0 : 0.0;
  double *M = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
  if (!M) { free(V); return -1; }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      M[i * n + j] = A[j * lda + i];
  for (int iter = 0; iter < 100 * n * n; iter++) {
    double off = 0;
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++)
        off += M[i * n + j] * M[i * n + j];
    if (off < 1e-30) break;
    for (int p = 0; p < n - 1; p++) {
      for (int q = p + 1; q < n; q++) {
        double apq = M[p * n + q];
        if (fabs(apq) < 1e-15) continue;
        double d = (M[q * n + q] - M[p * n + p]) / (2.0 * apq);
        double t = (d >= 0 ? 1.0 : -1.0) / (fabs(d) + sqrt(d * d + 1.0));
        double c = 1.0 / sqrt(t * t + 1.0);
        double s = t * c;
        double tau = s / (1.0 + c);
        M[p * n + p] -= t * apq;
        M[q * n + q] += t * apq;
        M[p * n + q] = 0;
        M[q * n + p] = 0;
        for (int r = 0; r < n; r++) {
          if (r == p || r == q) continue;
          double mrp = M[r * n + p], mrq = M[r * n + q];
          M[r * n + p] = mrp - s * (mrq + tau * mrp);
          M[q * n + r] = M[r * n + q] = mrq + s * (mrp - tau * mrq);
          M[p * n + r] = M[r * n + p];
        }
        tk_jacobi_rotate(V, n, p, q, c, s);
      }
    }
  }
  for (int i = 0; i < n; i++)
    w[i] = M[i * n + i];
  for (int i = 0; i < n - 1; i++) {
    int mi = i;
    for (int j = i + 1; j < n; j++)
      if (w[j] < w[mi]) mi = j;
    if (mi != i) {
      double tmp = w[i]; w[i] = w[mi]; w[mi] = tmp;
      for (int r = 0; r < n; r++) {
        tmp = V[r * n + i]; V[r * n + i] = V[r * n + mi]; V[r * n + mi] = tmp;
      }
    }
  }
  if (jobz == 'V')
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        A[j * lda + i] = V[i * n + j];
  free(V);
  free(M);
  return 0;
}

#else

#include <cblas.h>
#include <lapacke.h>

#endif

#endif
