#ifndef TK_ITQ_H
#define TK_ITQ_H

#include <santoku/iuset.h>
#include <santoku/cvec.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <santoku/rvec/ext.h>
#include <santoku/iumap.h>
#include <santoku/dumap.h>
#include <santoku/pvec.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>

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

static inline void tk_itq_median (
  lua_State *L,
  char *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  #pragma omp parallel
  {
    double *col = malloc(N * sizeof(double));
    char *local_out = calloc(N * TK_CVEC_BITS_BYTES(K), 1);

    if (col && local_out) {
      #pragma omp for
      for (uint64_t j = 0; j < K; j ++) {
        for (uint64_t i = 0; i < N; i ++)
          col[i] = X[i * K + j];

        ks_introsort(tk_dvec_asc, N, col);
        double med = (N & 1) ? col[N / 2] : 0.5 * (col[N / 2 - 1] + col[N / 2]);

        uint64_t byte_offset = TK_CVEC_BITS_BYTE(j);
        uint8_t bit_mask = (uint8_t) 1 << TK_CVEC_BITS_BIT(j);
        for (uint64_t i = 0; i < N; i ++) {
          if (X[i * K + j] >= med) {
            local_out[i * TK_CVEC_BITS_BYTES(K) + byte_offset] |= (char)bit_mask;
          }
        }
      }

      #pragma omp critical
      {
        for (uint64_t idx = 0; idx < N * TK_CVEC_BITS_BYTES(K); idx ++) {
          out[idx] |= local_out[idx];
        }
      }

      free(local_out);
      free(col);
    }
  }
}

static inline void tk_itq_otsu (
  lua_State *L,
  char *out,
  double *X,
  uint64_t N,
  uint64_t K,
  const char *metric,
  uint64_t n_bins
) {
  if (n_bins == 0)
    n_bins = 32;
  #pragma omp parallel
  {
    uint64_t *bins = (uint64_t *)calloc(n_bins, sizeof(uint64_t));
    char *local_out = calloc(N * TK_CVEC_BITS_BYTES(K), 1);
    if (!bins || !local_out) {
      #pragma omp critical
      {
        if (bins) free(bins);
        if (local_out) free(local_out);
        luaL_error(L, "Adaptive Otsu: failed to allocate buffers");
      }
    }
    bool use_entropy = (strcmp(metric, "entropy") == 0);
    #pragma omp for
    for (uint64_t j = 0; j < K; j ++) {
      double min_val = X[0 * K + j];
      double max_val = min_val;
      for (uint64_t i = 1; i < N; i ++) {
        double val = X[i * K + j];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
      }
      double range = max_val - min_val;
      if (range < 1e-10) {
        continue;
      }
      memset(bins, 0, n_bins * sizeof(uint64_t));
      for (uint64_t i = 0; i < N; i ++) {
        double val = X[i * K + j];
        double normalized = (val - min_val) / range;
        uint64_t bin_idx = (uint64_t)(normalized * (double)(n_bins - 1));
        if (bin_idx >= n_bins) bin_idx = n_bins - 1;
        bins[bin_idx]++;
      }
      double max_score = -DBL_MAX;
      double best_threshold = min_val;
      for (uint64_t b = 0; b < n_bins - 1; b ++) {
        uint64_t count_below = 0;
        uint64_t count_above = 0;
        for (uint64_t i = 0; i <= b; i ++)
          count_below += bins[i];
        for (uint64_t i = b + 1; i < n_bins; i ++)
          count_above += bins[i];
        if (count_below == 0 || count_above == 0)
          continue;
        double score = 0.0;
        if (use_entropy) {
          double H0 = 0.0;
          for (uint64_t i = 0; i <= b; i ++) {
            if (bins[i] > 0) {
              double p = (double)bins[i] / (double)count_below;
              H0 -= p * log2(p);
            }
          }
          double H1 = 0.0;
          for (uint64_t i = b + 1; i < n_bins; i ++) {
            if (bins[i] > 0) {
              double p = (double)bins[i] / (double)count_above;
              H1 -= p * log2(p);
            }
          }
          score = H0 + H1;
        } else {
          double w0 = (double)count_below / (double)N;
          double w1 = (double)count_above / (double)N;
          double sum_below = 0.0;
          double sum_above = 0.0;
          for (uint64_t i = 0; i <= b; i ++) {
            double bin_center = min_val + range * ((double)i + 0.5) / (double)n_bins;
            sum_below += (double)bins[i] * bin_center;
          }
          for (uint64_t i = b + 1; i < n_bins; i ++) {
            double bin_center = min_val + range * ((double)i + 0.5) / (double)n_bins;
            sum_above += (double)bins[i] * bin_center;
          }
          double mean0 = sum_below / (double)count_below;
          double mean1 = sum_above / (double)count_above;
          score = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);
        }
        if (score > max_score) {
          max_score = score;
          best_threshold = min_val + range * ((double)(b + 1)) / (double)n_bins;
        }
      }
      uint64_t byte_offset = TK_CVEC_BITS_BYTE(j);
      uint8_t bit_mask = (uint8_t) 1 << TK_CVEC_BITS_BIT(j);
      for (uint64_t i = 0; i < N; i ++) {
        if (X[i * K + j] >= best_threshold) {
          local_out[i * TK_CVEC_BITS_BYTES(K) + byte_offset] |= (char)bit_mask;
        }
      }
    }
    #pragma omp critical
    {
      for (uint64_t idx = 0; idx < N * TK_CVEC_BITS_BYTES(K); idx ++) {
        out[idx] |= local_out[idx];
      }
    }
    free(local_out);
    free(bins);
  }
}

static inline void tk_itq_encode (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_dims,
  uint64_t max_iterations,
  double tolerance,
  int i_each
) {
  const uint64_t K = n_dims;
  const size_t N = codes->n / K;
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);
  size_t total_size = (N * K * 3 +  K * K * 4 + K + K - 1) * sizeof(double);
  double *mem = tk_malloc(L, total_size);
  double *X = mem;
  double *V0 = X + N * K;
  double *B = V0 + N * K;
  double *R = B + N * K;
  double *BtV = R + K * K;
  double *U = BtV + K * K;
  double *VT = U + K * K;
  double *S = VT + K * K;
  double *superb = S + K;
  memcpy(X, codes->a, N * K * sizeof(double));
  tk_dvec_center(X, N, K);
  #pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < K; i ++)
    for (uint64_t j = 0; j < K; j ++)
      R[i * K + j] = (i == j ? 1.0 : 0.0);
  double last_obj = DBL_MAX, first_obj = 0.0;
  uint64_t it = 0;
  for (it = 0; it < max_iterations; it ++) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, R, K, 0.0, V0, K);
    double obj = 0.0;
    #pragma omp parallel for reduction(+:obj)
    for (size_t idx = 0; idx < N * K; idx ++) {
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
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, N, 1.0, B, K, X, K, 0.0, BtV, K);
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', K, K, BtV, K, S, U, K, VT, K, superb);
    if (info != 0) {
      free(mem);
      luaL_error(L, "ITQ SVD failed to converge (info=%d)", info);
      return;
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, K, K, K, 1.0, VT, K, U, K, 0.0, R, K);
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, R, K, 0.0, V0, K);
  tk_itq_sign(out->a, V0, N, K);
  free(mem);
  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) it);
    lua_pushnumber(L, first_obj);
    lua_pushnumber(L, last_obj);
    lua_call(L, 3, 0);
  }
}

#endif
