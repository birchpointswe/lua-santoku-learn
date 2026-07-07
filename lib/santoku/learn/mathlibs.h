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

static inline double cblas_dnrm2 (int n, const double *x, int incx) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += x[i * incx] * x[i * incx];
  return sqrt(s);
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

static inline void cblas_ssyrk (enum CBLAS_ORDER order, enum CBLAS_UPLO uplo,
    enum CBLAS_TRANSPOSE trans, int N, int K,
    float alpha, const float *A, int lda, float beta, float *C, int ldc) {
  for (int i = 0; i < N; i++) {
    for (int j = (uplo == CblasUpper ? i : 0); j < (uplo == CblasUpper ? N : i + 1); j++) {
      float s = 0;
      for (int k = 0; k < K; k++) {
        float ai, aj;
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

static inline void cblas_ssyr (enum CBLAS_ORDER order, enum CBLAS_UPLO uplo,
    int N, float alpha, const float *x, int incx, float *A, int lda) {
  for (int i = 0; i < N; i++) {
    for (int j = (uplo == CblasUpper ? i : 0); j < (uplo == CblasUpper ? N : i + 1); j++) {
      if (order == CblasColMajor)
        A[j * lda + i] += alpha * x[i * incx] * x[j * incx];
      else
        A[i * lda + j] += alpha * x[i * incx] * x[j * incx];
    }
  }
}

static inline void cblas_sger (enum CBLAS_ORDER order,
    int M, int N, float alpha, const float *x, int incx,
    const float *y, int incy, float *A, int lda) {
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
  if (side == CblasRight && uplo == CblasLower && trans == CblasTrans) {
    for (int i = 0; i < M; i++) {
      float *b = B + i * ldb;
      for (int j = 0; j < N; j++) {
        float s = alpha * b[j];
        for (int k = 0; k < j; k++)
          s -= A[j * lda + k] * b[k];
        b[j] = (diag == CblasNonUnit) ? s / A[j * lda + j] : s;
      }
    }
    return;
  }
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

static inline int LAPACKE_spotrf (int matrix_layout, char uplo, int n, float *A, int lda) {
  (void)matrix_layout; (void)uplo;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i <= j; i++) {
      float s = A[i + j * lda];
      for (int k = 0; k < i; k++)
        s -= A[k + i * lda] * A[k + j * lda];
      if (i == j) {
        if (s <= 0.0f) return j + 1;
        A[j + j * lda] = sqrtf(s);
      } else {
        A[i + j * lda] = s / A[i + i * lda];
      }
    }
  }
  return 0;
}

static inline int LAPACKE_spotrs (int matrix_layout, char uplo, int n, int nrhs,
    const float *A, int lda, float *B, int ldb) {
  (void)matrix_layout; (void)uplo;
  for (int c = 0; c < nrhs; c++) {
    float *b = B + c * ldb;
    for (int i = 0; i < n; i++) {
      float s = b[i];
      for (int k = 0; k < i; k++)
        s -= A[k + i * lda] * b[k];
      b[i] = s / A[i + i * lda];
    }
    for (int i = n - 1; i >= 0; i--) {
      float s = b[i];
      for (int k = i + 1; k < n; k++)
        s -= A[i + k * lda] * b[k];
      b[i] = s / A[i + i * lda];
    }
  }
  return 0;
}

#else

#include <cblas.h>
#include <lapacke.h>

#endif

#endif
