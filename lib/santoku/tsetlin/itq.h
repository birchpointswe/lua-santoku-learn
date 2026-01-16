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
  uint64_t K,
  tk_dvec_t *in_thresholds,
  tk_dvec_t **out_thresholds
) {
  bool use_provided = (in_thresholds != NULL);

  if (use_provided && in_thresholds->n != K) {
    luaL_error(L, "median: thresholds size mismatch (expected %llu, got %llu)",
               (unsigned long long)K, (unsigned long long)in_thresholds->n);
    return;
  }

  tk_dvec_t *thresholds = NULL;
  if (!use_provided && out_thresholds) {
    thresholds = tk_dvec_create(L, K, 0, 0);
    thresholds->n = K;
  }

  #pragma omp parallel
  {
    double *col = use_provided ? NULL : malloc(N * sizeof(double));
    char *local_out = calloc(N * TK_CVEC_BITS_BYTES(K), 1);

    if (local_out && (use_provided || col)) {
      #pragma omp for
      for (uint64_t j = 0; j < K; j++) {
        double med;
        if (use_provided) {
          med = in_thresholds->a[j];
        } else {
          for (uint64_t i = 0; i < N; i++)
            col[i] = X[i * K + j];

          ks_introsort(tk_dvec_asc, N, col);
          med = (N & 1) ? col[N / 2] : 0.5 * (col[N / 2 - 1] + col[N / 2]);

          if (thresholds)
            thresholds->a[j] = med;
        }

        uint64_t byte_offset = TK_CVEC_BITS_BYTE(j);
        uint8_t bit_mask = (uint8_t) 1 << TK_CVEC_BITS_BIT(j);
        for (uint64_t i = 0; i < N; i++) {
          if (X[i * K + j] >= med) {
            local_out[i * TK_CVEC_BITS_BYTES(K) + byte_offset] |= (char)bit_mask;
          }
        }
      }

      #pragma omp critical
      {
        for (uint64_t idx = 0; idx < N * TK_CVEC_BITS_BYTES(K); idx++) {
          out[idx] |= local_out[idx];
        }
      }

      free(local_out);
      if (col) free(col);
    }
  }

  if (out_thresholds) *out_thresholds = thresholds;
}

static inline void tk_itq_otsu (
  lua_State *L,
  char *out,
  double *X,
  uint64_t N,
  uint64_t K,
  const char *metric,
  uint64_t n_bins,
  bool minimize,
  tk_ivec_t *in_indices,
  tk_dvec_t *in_thresholds,
  tk_ivec_t *out_indices,
  tk_dvec_t *out_scores,
  tk_dvec_t **out_thresholds
) {
  bool use_provided = (in_indices != NULL && in_thresholds != NULL);

  if (use_provided) {
    if (in_indices->n != K) {
      luaL_error(L, "otsu: indices size mismatch (expected %llu, got %llu)",
                 (unsigned long long)K, (unsigned long long)in_indices->n);
      return;
    }
    if (in_thresholds->n != K) {
      luaL_error(L, "otsu: thresholds size mismatch (expected %llu, got %llu)",
                 (unsigned long long)K, (unsigned long long)in_thresholds->n);
      return;
    }

    char *temp_out = (char *)calloc(N * TK_CVEC_BITS_BYTES(K), 1);
    if (!temp_out) {
      luaL_error(L, "Otsu: failed to allocate buffer");
      return;
    }

    #pragma omp parallel for
    for (uint64_t j = 0; j < K; j++) {
      uint64_t orig_j = (uint64_t)in_indices->a[j];
      double threshold = in_thresholds->a[orig_j];
      uint64_t byte_offset = TK_CVEC_BITS_BYTE(j);
      uint8_t bit_mask = (uint8_t) 1 << TK_CVEC_BITS_BIT(j);
      for (uint64_t i = 0; i < N; i++) {
        if (X[i * K + orig_j] >= threshold) {
          ((uint8_t *)temp_out)[i * TK_CVEC_BITS_BYTES(K) + byte_offset] |= bit_mask;
        }
      }
    }

    memcpy(out, temp_out, N * TK_CVEC_BITS_BYTES(K));
    free(temp_out);
    return;
  }

  if (n_bins == 0)
    n_bins = 32;
  bool use_entropy = (strcmp(metric, "entropy") == 0);
  tk_rvec_t *ranked = tk_rvec_create(0, 0, 0, 0);
  tk_rvec_ensure(ranked, K);
  double *scores = (double *)calloc(K, sizeof(double));
  double *thresholds = (double *)calloc(K, sizeof(double));
  char *temp_out = (char *)calloc(N * TK_CVEC_BITS_BYTES(K), 1);
  if (!scores || !thresholds || !temp_out) {
    if (scores) free(scores);
    if (thresholds) free(thresholds);
    if (temp_out) free(temp_out);
    tk_rvec_destroy(ranked);
    luaL_error(L, "Otsu: failed to allocate buffers");
    return;
  }
  #pragma omp parallel
  {
    uint64_t *bins = (uint64_t *)calloc(n_bins, sizeof(uint64_t));
    if (!bins) {
      #pragma omp critical
      luaL_error(L, "Otsu: failed to allocate bin buffer");
    }
    #pragma omp for
    for (uint64_t j = 0; j < K; j++) {
      double min_val = X[0 * K + j];
      double max_val = min_val;
      for (uint64_t i = 1; i < N; i++) {
        double val = X[i * K + j];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
      }
      double range = max_val - min_val;
      if (range < 1e-10) {
        scores[j] = 0.0;
        thresholds[j] = min_val;
        continue;
      }
      memset(bins, 0, n_bins * sizeof(uint64_t));
      for (uint64_t i = 0; i < N; i++) {
        double val = X[i * K + j];
        double normalized = (val - min_val) / range;
        uint64_t bin_idx = (uint64_t)(normalized * (double)(n_bins - 1));
        if (bin_idx >= n_bins) bin_idx = n_bins - 1;
        bins[bin_idx]++;
      }
      double best_score = minimize ? DBL_MAX : -DBL_MAX;
      double best_threshold = min_val;
      for (uint64_t b = 0; b < n_bins - 1; b++) {
        uint64_t count_below = 0;
        uint64_t count_above = 0;
        for (uint64_t i = 0; i <= b; i++)
          count_below += bins[i];
        for (uint64_t i = b + 1; i < n_bins; i++)
          count_above += bins[i];
        if (count_below == 0 || count_above == 0)
          continue;
        double score = 0.0;
        if (use_entropy) {
          double H0 = 0.0;
          for (uint64_t i = 0; i <= b; i++) {
            if (bins[i] > 0) {
              double p = (double)bins[i] / (double)count_below;
              H0 -= p * log2(p);
            }
          }
          double H1 = 0.0;
          for (uint64_t i = b + 1; i < n_bins; i++) {
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
          for (uint64_t i = 0; i <= b; i++) {
            double bin_center = min_val + range * ((double)i + 0.5) / (double)n_bins;
            sum_below += (double)bins[i] * bin_center;
          }
          for (uint64_t i = b + 1; i < n_bins; i++) {
            double bin_center = min_val + range * ((double)i + 0.5) / (double)n_bins;
            sum_above += (double)bins[i] * bin_center;
          }
          double mean0 = sum_below / (double)count_below;
          double mean1 = sum_above / (double)count_above;
          score = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);
        }
        bool is_better = minimize ? (score < best_score) : (score > best_score);
        if (is_better) {
          best_score = score;
          best_threshold = min_val + range * ((double)(b + 1)) / (double)n_bins;
        }
      }
      scores[j] = best_score;
      thresholds[j] = best_threshold;
    }
    free(bins);
  }
  for (uint64_t j = 0; j < K; j++) {
    double rank_score = minimize ? -scores[j] : scores[j];
    tk_rank_t r = { (int64_t)j, rank_score };
    tk_rvec_push(ranked, r);
  }
  tk_rvec_desc(ranked, 0, ranked->n);
  for (uint64_t j = 0; j < K; j++) {
    uint64_t orig_j = (uint64_t)ranked->a[j].i;
    double threshold = thresholds[orig_j];
    uint64_t byte_offset = TK_CVEC_BITS_BYTE(j);
    uint8_t bit_mask = (uint8_t) 1 << TK_CVEC_BITS_BIT(j);
    for (uint64_t i = 0; i < N; i++) {
      if (X[i * K + orig_j] >= threshold) {
        temp_out[i * TK_CVEC_BITS_BYTES(K) + byte_offset] |= (char)bit_mask;
      }
    }
  }
  memcpy(out, temp_out, N * TK_CVEC_BITS_BYTES(K));
  if (out_indices) {
    tk_rvec_keys(L, ranked, out_indices);
  }
  if (out_scores) {
    tk_dvec_ensure(out_scores, K);
    for (uint64_t j = 0; j < K; j++) {
      uint64_t orig_j = (uint64_t)ranked->a[j].i;
      out_scores->a[j] = minimize ? -scores[orig_j] : scores[orig_j];
    }
    out_scores->n = K;
  }
  if (out_thresholds) {
    tk_dvec_t *thresh_out = tk_dvec_create(L, K, 0, 0);
    thresh_out->n = K;
    memcpy(thresh_out->a, thresholds, K * sizeof(double));
    *out_thresholds = thresh_out;
  }
  free(scores);
  free(thresholds);
  free(temp_out);
  tk_rvec_destroy(ranked);
}

static inline void tk_itq_encode (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_dims,
  uint64_t max_iterations,
  double tolerance,
  int i_each,
  tk_dvec_t *in_rotation,
  tk_dvec_t *in_mean,
  tk_dvec_t **out_rotation,
  tk_dvec_t **out_mean
) {
  const uint64_t K = n_dims;
  const size_t N = codes->n / K;
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);

  bool use_provided = (in_rotation != NULL && in_mean != NULL);
  bool learn_new = !use_provided;

  if (use_provided) {
    if (in_rotation->n != K * K)
      luaL_error(L, "ITQ: rotation matrix size mismatch (expected %llu, got %llu)",
                 (unsigned long long)(K * K), (unsigned long long)in_rotation->n);
    if (in_mean->n != K)
      luaL_error(L, "ITQ: mean vector size mismatch (expected %llu, got %llu)",
                 (unsigned long long)K, (unsigned long long)in_mean->n);
  }

  size_t work_size = N * K * 2 * sizeof(double);
  if (learn_new)
    work_size = (N * K * 3 + K * K * 4 + K + K - 1 + K) * sizeof(double);

  double *mem = tk_malloc(L, work_size);
  double *X = mem;
  double *V0 = X + N * K;

  memcpy(X, codes->a, N * K * sizeof(double));

  tk_dvec_t *rotation = NULL;
  tk_dvec_t *mean = NULL;

  if (use_provided) {
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; i++) {
      for (uint64_t k = 0; k < K; k++) {
        X[i * K + k] -= in_mean->a[k];
      }
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, in_rotation->a, K, 0.0, V0, K);
    tk_itq_sign(out->a, V0, N, K);
    free(mem);
    if (i_each >= 0) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, 0);
      lua_pushnumber(L, 0.0);
      lua_pushnumber(L, 0.0);
      lua_call(L, 3, 0);
    }
  } else {
    double *B = V0 + N * K;
    double *R = B + N * K;
    double *BtV = R + K * K;
    double *U = BtV + K * K;
    double *VT = U + K * K;
    double *S = VT + K * K;
    double *superb = S + K;

    double *mean_buf = superb + (K - 1);
    #pragma omp parallel for
    for (uint64_t k = 0; k < K; k++) {
      double sum = 0.0;
      for (uint64_t i = 0; i < N; i++) {
        sum += X[i * K + k];
      }
      double mu = sum / (double)N;
      mean_buf[k] = mu;
      for (uint64_t i = 0; i < N; i++) {
        X[i * K + k] -= mu;
      }
    }

    #pragma omp parallel for collapse(2)
    for (uint64_t i = 0; i < K; i++)
      for (uint64_t j = 0; j < K; j++)
        R[i * K + j] = (i == j ? 1.0 : 0.0);

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

    if (i_each >= 0) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t) it);
      lua_pushnumber(L, first_obj);
      lua_pushnumber(L, last_obj);
      lua_call(L, 3, 0);
    }

    rotation = tk_dvec_create(L, K * K, 0, 0);
    rotation->n = K * K;
    memcpy(rotation->a, R, K * K * sizeof(double));

    mean = tk_dvec_create(L, K, 0, 0);
    mean->n = K;
    memcpy(mean->a, mean_buf, K * sizeof(double));

    free(mem);

    if (out_rotation) *out_rotation = rotation;
    if (out_mean) *out_mean = mean;
  }
}

#endif
