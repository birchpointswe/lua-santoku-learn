#ifndef TK_ITQ_H
#define TK_ITQ_H

#include <santoku/iuset.h>
#include <santoku/cvec.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <santoku/ivec.h>
#include <santoku/rvec.h>
#include <santoku/rvec/ext.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cblas.h>
#include <lapacke.h>
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

static inline void tk_itq_ica (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_dims,
  uint64_t max_iterations,
  double tolerance,
  int i_each,
  tk_dvec_t **means_out,
  tk_dvec_t **unmixing_out
) {
  const uint64_t K = n_dims;
  const size_t N = codes->n / K;
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);

  size_t work_size = (N * K * 2 + K * K * 4 + K * 3) * sizeof(double);
  double *mem = tk_malloc(L, work_size);
  double *X = mem;
  double *X_white = X + N * K;
  double *cov = X_white + N * K;
  double *W_white = cov + K * K;
  double *W_ica = W_white + K * K;
  double *W_full = W_ica + K * K;
  double *mean_buf = W_full + K * K;
  double *eigenvalues = mean_buf + K;
  double *w_tmp = eigenvalues + K;

  memcpy(X, codes->a, N * K * sizeof(double));

  #pragma omp parallel for
  for (uint64_t k = 0; k < K; k++) {
    double sum = 0.0;
    for (uint64_t i = 0; i < N; i++)
      sum += X[i * K + k];
    double mu = sum / (double)N;
    mean_buf[k] = mu;
    for (uint64_t i = 0; i < N; i++)
      X[i * K + k] -= mu;
  }

  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    K, K, N, 1.0 / (double)N, X, K, X, K, 0.0, cov, K);

  memcpy(W_white, cov, K * K * sizeof(double));
  int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', K, W_white, K, eigenvalues);
  if (info != 0)
    luaL_error(L, "ICA whitening eigendecomposition failed: %d", info);

  for (uint64_t i = 0; i < K; i++) {
    double scale = (eigenvalues[i] > 1e-10) ? 1.0 / sqrt(eigenvalues[i]) : 0.0;
    for (uint64_t j = 0; j < K; j++)
      W_white[j * K + i] *= scale;
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    N, K, K, 1.0, X, K, W_white, K, 0.0, X_white, K);

  for (uint64_t i = 0; i < K * K; i++)
    W_ica[i] = 0.0;
  for (uint64_t i = 0; i < K; i++)
    W_ica[i * K + i] = 1.0;

  double *wx = tk_malloc(L, N * sizeof(double));
  double *gwx = tk_malloc(L, N * sizeof(double));

  for (uint64_t comp = 0; comp < K; comp++) {
    double *w = W_ica + comp * K;

    for (uint64_t i = 0; i < K; i++)
      w[i] = (double)rand() / RAND_MAX - 0.5;
    double norm = 0.0;
    for (uint64_t i = 0; i < K; i++)
      norm += w[i] * w[i];
    norm = sqrt(norm);
    for (uint64_t i = 0; i < K; i++)
      w[i] /= norm;

    for (uint64_t it = 0; it < max_iterations; it++) {
      #pragma omp parallel for
      for (uint64_t i = 0; i < N; i++) {
        double dot = 0.0;
        for (uint64_t j = 0; j < K; j++)
          dot += X_white[i * K + j] * w[j];
        wx[i] = dot;
        gwx[i] = tanh(dot);
      }

      for (uint64_t j = 0; j < K; j++)
        w_tmp[j] = 0.0;

      double g_prime_mean = 0.0;
      for (uint64_t i = 0; i < N; i++) {
        double gp = 1.0 - gwx[i] * gwx[i];
        g_prime_mean += gp;
        for (uint64_t j = 0; j < K; j++)
          w_tmp[j] += X_white[i * K + j] * gwx[i];
      }
      g_prime_mean /= (double)N;

      for (uint64_t j = 0; j < K; j++)
        w_tmp[j] = w_tmp[j] / (double)N - g_prime_mean * w[j];

      for (uint64_t prev = 0; prev < comp; prev++) {
        double *w_prev = W_ica + prev * K;
        double dot = 0.0;
        for (uint64_t j = 0; j < K; j++)
          dot += w_tmp[j] * w_prev[j];
        for (uint64_t j = 0; j < K; j++)
          w_tmp[j] -= dot * w_prev[j];
      }

      norm = 0.0;
      for (uint64_t j = 0; j < K; j++)
        norm += w_tmp[j] * w_tmp[j];
      norm = sqrt(norm);
      if (norm < 1e-10)
        break;

      double change = 0.0;
      for (uint64_t j = 0; j < K; j++) {
        double new_val = w_tmp[j] / norm;
        change += fabs(fabs(new_val) - fabs(w[j]));
        w[j] = new_val;
      }

      if (change < tolerance)
        break;
    }
  }

  free(wx);
  free(gwx);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    K, K, K, 1.0, W_white, K, W_ica, K, 0.0, W_full, K);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    N, K, K, 1.0, X, K, W_full, K, 0.0, X_white, K);

  double *medians = tk_malloc(L, K * sizeof(double));
  double *col_buf = tk_malloc(L, N * sizeof(double));
  for (uint64_t k = 0; k < K; k++) {
    for (uint64_t i = 0; i < N; i++)
      col_buf[i] = X_white[i * K + k];
    qsort(col_buf, N, sizeof(double), tk_itq_cmp_double);
    medians[k] = (N % 2 == 1) ? col_buf[N / 2] : (col_buf[N / 2 - 1] + col_buf[N / 2]) / 2.0;
  }
  free(col_buf);

  tk_itq_threshold(out->a, X_white, medians, N, K);
  free(medians);

  if (means_out) {
    tk_dvec_t *means = tk_dvec_create(L, K, 0, 0);
    means->n = K;
    memcpy(means->a, mean_buf, K * sizeof(double));
    *means_out = means;
  }

  if (unmixing_out) {
    tk_dvec_t *unmixing = tk_dvec_create(L, K * K, 0, 0);
    unmixing->n = K * K;
    memcpy(unmixing->a, W_full, K * K * sizeof(double));
    *unmixing_out = unmixing;
  }

  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t)K);
    lua_call(L, 1, 0);
  }

  free(mem);
}

typedef struct {
  int64_t idx;
  double val;
} tk_itq_cca_pair_t;

static int tk_itq_cca_pair_cmp_desc (const void *a, const void *b) {
  double va = ((const tk_itq_cca_pair_t *)a)->val;
  double vb = ((const tk_itq_cca_pair_t *)b)->val;
  if (va > vb) return -1;
  if (va < vb) return 1;
  return 0;
}

static inline void tk_itq_cca (
  lua_State *L,
  tk_dvec_t *codes,
  tk_ivec_t *tokens,
  uint64_t n_samples,
  uint64_t n_dims,
  uint64_t n_tokens,
  uint64_t features_per_class,
  int i_each,
  tk_dvec_t **means_out,
  tk_dvec_t **rotation_out,
  tk_ivec_t **feat_offsets_out,
  tk_ivec_t **feat_ids_out,
  tk_dvec_t **feat_weights_out
) {
  const uint64_t D = n_dims;
  const uint64_t N = n_samples;
  const uint64_t V = n_tokens;
  const uint64_t K = features_per_class > 0 ? features_per_class : V;

  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(D), 0, 0);
  tk_cvec_zero(out);

  double *X = tk_malloc(L, N * D * sizeof(double));
  double *mean_buf = tk_malloc(L, D * sizeof(double));
  double *C = tk_malloc(L, D * V * sizeof(double));
  double *CCt = tk_malloc(L, D * D * sizeof(double));
  double *R = tk_malloc(L, D * D * sizeof(double));
  double *V0 = tk_malloc(L, N * D * sizeof(double));
  double *eigenvalues = tk_malloc(L, D * sizeof(double));

  memcpy(X, codes->a, N * D * sizeof(double));
  memset(C, 0, D * V * sizeof(double));

  #pragma omp parallel for schedule(static)
  for (uint64_t d = 0; d < D; d++) {
    double sum = 0.0;
    for (uint64_t i = 0; i < N; i++)
      sum += X[i * D + d];
    double mu = sum / (double)N;
    mean_buf[d] = mu;
    for (uint64_t i = 0; i < N; i++)
      X[i * D + d] -= mu;
  }

  int64_t *csr_offsets = tk_malloc(L, (N + 1) * sizeof(int64_t));
  memset(csr_offsets, 0, (N + 1) * sizeof(int64_t));
  for (uint64_t i = 0; i < tokens->n; i++) {
    int64_t v = tokens->a[i];
    if (v < 0) continue;
    uint64_t s = (uint64_t)v / V;
    if (s < N)
      csr_offsets[s + 1]++;
  }
  for (uint64_t i = 1; i <= N; i++)
    csr_offsets[i] += csr_offsets[i - 1];

  uint64_t total = (uint64_t)csr_offsets[N];
  int64_t *tok_ids = tk_malloc(L, (total > 0 ? total : 1) * sizeof(int64_t));
  int64_t *counts = tk_malloc(L, N * sizeof(int64_t));
  memset(counts, 0, N * sizeof(int64_t));
  for (uint64_t i = 0; i < tokens->n; i++) {
    int64_t v = tokens->a[i];
    if (v < 0) continue;
    uint64_t s = (uint64_t)v / V;
    uint64_t tok = (uint64_t)v % V;
    if (s < N) {
      uint64_t pos = (uint64_t)csr_offsets[s] + (uint64_t)counts[s];
      tok_ids[pos] = (int64_t)tok;
      counts[s]++;
    }
  }

  for (uint64_t i = 0; i < N; i++) {
    int64_t start = csr_offsets[i];
    int64_t end = csr_offsets[i + 1];
    double *Xi = X + i * D;
    for (int64_t k = start; k < end; k++) {
      int64_t j = tok_ids[k];
      for (uint64_t d = 0; d < D; d++)
        C[d * V + (uint64_t) j] += Xi[d];
    }
  }

  free(csr_offsets);
  free(tok_ids);
  free(counts);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    (int)D, (int)D, (int)V, 1.0, C, (int)V, C, (int)V, 0.0, CCt, (int)D);

  int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', (int)D, CCt, (int)D, eigenvalues);
  if (info != 0)
    luaL_error(L, "CCA eigendecomposition failed: %d", info);

  for (uint64_t i = 0; i < D; i++) {
    for (uint64_t j = 0; j < D; j++)
      R[i * D + j] = CCt[i * D + (D - 1 - j)];
  }

  double *Cprime = tk_malloc(L, D * V * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    (int)D, (int)V, (int)D, 1.0, R, (int)D, C, (int)V, 0.0, Cprime, (int)V);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)N, (int)D, (int)D, 1.0, X, (int)D, R, (int)D, 0.0, V0, (int)D);

  tk_itq_sign(out->a, V0, N, D);

  if (means_out) {
    tk_dvec_t *means = tk_dvec_create(L, D, 0, 0);
    means->n = D;
    memcpy(means->a, mean_buf, D * sizeof(double));
    *means_out = means;
  }

  if (rotation_out) {
    tk_dvec_t *rotation = tk_dvec_create(L, D * D, 0, 0);
    rotation->n = D * D;
    memcpy(rotation->a, R, D * D * sizeof(double));
    *rotation_out = rotation;
  }

  if (feat_offsets_out && feat_ids_out && feat_weights_out) {
    uint64_t actual_k = K < V ? K : V;
    tk_ivec_t *f_offsets = tk_ivec_create(L, D + 1, NULL, NULL);
    tk_ivec_t *f_ids = tk_ivec_create(L, D * actual_k, NULL, NULL);
    tk_dvec_t *f_weights = tk_dvec_create(L, D * actual_k, NULL, NULL);
    f_offsets->n = D + 1;
    f_ids->n = D * actual_k;
    f_weights->n = D * actual_k;

    tk_itq_cca_pair_t *pairs = tk_malloc(L, V * sizeof(tk_itq_cca_pair_t));

    for (uint64_t d = 0; d < D; d++) {
      f_offsets->a[d] = (int64_t)(d * actual_k);
      double *Cd = Cprime + d * V;
      for (uint64_t j = 0; j < V; j++) {
        pairs[j].idx = (int64_t)j;
        pairs[j].val = fabs(Cd[j]);
      }
      qsort(pairs, V, sizeof(tk_itq_cca_pair_t), tk_itq_cca_pair_cmp_desc);
      for (uint64_t i = 0; i < actual_k; i++) {
        f_ids->a[d * actual_k + i] = pairs[i].idx;
        f_weights->a[d * actual_k + i] = Cprime[d * V + (uint64_t) pairs[i].idx];
      }
    }
    f_offsets->a[D] = (int64_t)(D * actual_k);

    free(pairs);
    *feat_offsets_out = f_offsets;
    *feat_ids_out = f_ids;
    *feat_weights_out = f_weights;
  }

  if (i_each >= 0) {
    double eig_min = eigenvalues[0];
    double eig_max = eigenvalues[D - 1];
    lua_pushvalue(L, i_each);
    lua_pushnumber(L, eig_min);
    lua_pushnumber(L, eig_max);
    lua_call(L, 2, 0);
  }

  free(X);
  free(mean_buf);
  free(C);
  free(Cprime);
  free(CCt);
  free(R);
  free(V0);
  free(eigenvalues);
}

#endif
