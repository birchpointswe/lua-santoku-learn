#ifndef TK_ITQ_H
#define TK_ITQ_H

#include <santoku/iuset.h>
#include <santoku/cvec.h>
#include <santoku/dvec.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <cblas.h>
#include <omp.h>

static inline void tk_itq_threshold (
  char *out,
  double *X,
  double *thresholds,
  uint64_t N,
  uint64_t K
) {
  #pragma omp parallel for
  for (uint64_t i = 0; i < N; i ++) {
    double *row = X + i * K;
    uint8_t *out_row = (uint8_t *)(out + i * TK_CVEC_BITS_BYTES(K));
    uint64_t full_bytes = K / 8;
    for (uint64_t byte_idx = 0; byte_idx < full_bytes; byte_idx ++) {
      uint8_t byte_val = 0;
      uint64_t j_base = byte_idx * 8;
      for (uint64_t bit = 0; bit < 8; bit ++) {
        byte_val |= (row[j_base + bit] >= thresholds[j_base + bit]) << bit;
      }
      out_row[byte_idx] = byte_val;
    }
    uint64_t remaining_start = full_bytes * 8;
    if (remaining_start < K) {
      uint8_t byte_val = 0;
      for (uint64_t j = remaining_start; j < K; j ++) {
        byte_val |= (row[j] >= thresholds[j]) << (j - remaining_start);
      }
      out_row[full_bytes] = byte_val;
    }
  }
}

static inline void tk_itq_sign (
  char *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  #pragma omp parallel for
  for (uint64_t i = 0; i < N; i ++) {
    double *row = X + i * K;
    uint8_t *out_row = (uint8_t *)(out + i * TK_CVEC_BITS_BYTES(K));
    uint64_t full_bytes = K / 8;
    for (uint64_t byte_idx = 0; byte_idx < full_bytes; byte_idx ++) {
      uint8_t byte_val = 0;
      uint64_t j_base = byte_idx * 8;
      for (uint64_t bit = 0; bit < 8; bit ++) {
        byte_val |= (row[j_base + bit] >= 0.0) << bit;
      }
      out_row[byte_idx] = byte_val;
    }
    uint64_t remaining_start = full_bytes * 8;
    if (remaining_start < K) {
      uint8_t byte_val = 0;
      for (uint64_t j = remaining_start; j < K; j ++) {
        byte_val |= (row[j] >= 0.0) << (j - remaining_start);
      }
      out_row[full_bytes] = byte_val;
    }
  }
}

static int tk_itq_cmp_double (const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static inline void tk_itq_median (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_dims,
  tk_dvec_t **medians_out
) {
  const uint64_t K = n_dims;
  const size_t N = codes->n / K;
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);

  double *medians = tk_malloc(L, K * sizeof(double));
  double *col_buf = tk_malloc(L, N * sizeof(double));

  for (uint64_t k = 0; k < K; k++) {
    for (uint64_t i = 0; i < N; i++)
      col_buf[i] = codes->a[i * K + k];
    qsort(col_buf, N, sizeof(double), tk_itq_cmp_double);
    medians[k] = (N % 2 == 1) ? col_buf[N / 2] : (col_buf[N / 2 - 1] + col_buf[N / 2]) / 2.0;
  }

  free(col_buf);

  tk_itq_threshold(out->a, codes->a, medians, N, K);

  if (medians_out) {
    tk_dvec_t *med_vec = tk_dvec_create(L, K, 0, 0);
    med_vec->n = K;
    memcpy(med_vec->a, medians, K * sizeof(double));
    *medians_out = med_vec;
  }

  free(medians);
}

static inline void tk_itq_encode (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_dims,
  uint64_t max_iterations,
  double tolerance,
  int i_each,
  tk_dvec_t **means_out,
  tk_dvec_t **rotation_out
) {
  const uint64_t K = n_dims;
  const size_t N = codes->n / K;
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);

  size_t work_size = (N * K * 3 + K * K * 4 + K) * sizeof(double);
  double *mem = tk_malloc(L, work_size);
  double *X = mem;
  double *V0 = X + N * K;
  double *B = V0 + N * K;
  double *R = B + N * K;
  double *Y = R + K * K;
  double *YtY = Y + K * K;
  double *tmp = YtY + K * K;
  double *mean_buf = tmp + K * K;

  memcpy(X, codes->a, N * K * sizeof(double));

  #pragma omp parallel
  {
    #pragma omp for
    for (uint64_t k = 0; k < K; k++) {
      double sum = 0.0;
      for (uint64_t i = 0; i < N; i++)
        sum += X[i * K + k];
      double mu = sum / (double)N;
      mean_buf[k] = mu;
      for (uint64_t i = 0; i < N; i++)
        X[i * K + k] -= mu;
    }

    #pragma omp for
    for (uint64_t i = 0; i < K * K; i++)
      R[i] = 0.0;

    #pragma omp single
    for (uint64_t i = 0; i < K; i++)
      R[i * K + i] = 1.0;
  }

  double last_obj = DBL_MAX, first_obj = 0.0;
  uint64_t it = 0;
  for (it = 0; it < max_iterations; it++) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, R, K, 0.0, V0, K);
    double obj = 0.0;
    #pragma omp parallel for reduction(+:obj)
    for (size_t idx = 0; idx < N * K; idx++) {
      double v = V0[idx];
      double b = (v >= 0.0 ? 1.0 : -1.0);
      B[idx] = b;
      double d = b - v;
      obj += d * d;
    }
    if (it == 0)
      first_obj = obj;
    if (it > 0 && fabs(last_obj - obj) < tolerance * fabs(obj))
      break;
    last_obj = obj;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, N, 1.0, B, K, X, K, 0.0, Y, K);
    double frob_sq = 0.0;
    #pragma omp parallel for reduction(+:frob_sq)
    for (uint64_t i = 0; i < K * K; i++)
      frob_sq += Y[i] * Y[i];
    double scale = 1.0 / sqrt(frob_sq);
    #pragma omp parallel for
    for (uint64_t i = 0; i < K * K; i++)
      Y[i] *= scale;
    for (int ns = 0; ns < 15; ns++) {
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, K, 1.0, Y, K, Y, K, 0.0, YtY, K);
      #pragma omp parallel for collapse(2)
      for (uint64_t i = 0; i < K; i++)
        for (uint64_t j = 0; j < K; j++)
          tmp[i * K + j] = (i == j ? 3.0 : 0.0) - YtY[i * K + j];
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, K, K, 0.5, Y, K, tmp, K, 0.0, R, K);
      memcpy(Y, R, K * K * sizeof(double));
    }
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, R, K, 0.0, V0, K);
  tk_itq_sign(out->a, V0, N, K);

  if (means_out) {
    tk_dvec_t *means = tk_dvec_create(L, K, 0, 0);
    means->n = K;
    memcpy(means->a, mean_buf, K * sizeof(double));
    *means_out = means;
  }

  if (rotation_out) {
    tk_dvec_t *rotation = tk_dvec_create(L, K * K, 0, 0);
    rotation->n = K * K;
    memcpy(rotation->a, R, K * K * sizeof(double));
    *rotation_out = rotation;
  }

  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) it);
    lua_pushnumber(L, first_obj);
    lua_pushnumber(L, last_obj);
    lua_call(L, 3, 0);
  }

  free(mem);
}

#endif
