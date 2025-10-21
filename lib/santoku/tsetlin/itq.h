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
#include <santoku/threads.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <lapacke.h>
#include <cblas.h>

typedef enum {
  TK_ITQ_MATVEC_XR,
  TK_ITQ_CENTER_SUM,
  TK_ITQ_CENTER_SUB,
  TK_ITQ_SIGN,
  TK_ITQ_PARTIAL_BTX,
  TK_ITQ_MATVEC_XR_FINAL,
  TK_ITQ_OBJECTIVE,
} tk_itq_stage_t;

typedef struct tk_itq_thread_s tk_itq_thread_t;
typedef struct tk_itq_data_s tk_itq_data_t;

typedef struct tk_itq_data_s {
  double *X;
  double *R;
  double *V0;
  double *V1;
  double *B;
  double *BtX_partials;
  double *obj_partials;
  double *center_sums;
  uint64_t N;
  uint64_t K;
  tk_threadpool_t *pool;
} tk_itq_data_t;

typedef struct tk_itq_thread_s {
  tk_itq_data_t *data;
  uint64_t ifirst;
  uint64_t ilast;
  unsigned int thread_idx;
} tk_itq_thread_t;

static inline void tk_itq_worker(void *dp, int sig) {
  tk_itq_stage_t stage = (tk_itq_stage_t) sig;
  tk_itq_thread_t *thread = (tk_itq_thread_t *) dp;
  tk_itq_data_t *data = thread->data;
  uint64_t ifirst = thread->ifirst;
  uint64_t ilast = thread->ilast;
  uint64_t K = data->K;
  unsigned int tid = thread->thread_idx;

  switch (stage) {
    case TK_ITQ_MATVEC_XR:
    case TK_ITQ_MATVEC_XR_FINAL: {
      double *X = data->X;
      double *R = data->R;
      double *V = (stage == TK_ITQ_MATVEC_XR) ? data->V0 : data->V1;
      for (uint64_t i = ifirst; i <= ilast; i++) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    K, K, 1.0,
                    R, K,
                    X + i * K, 1,
                    0.0,
                    V + i * K, 1);
      }
      break;
    }

    case TK_ITQ_CENTER_SUM: {
      double *V = data->V0;
      double *sums = data->center_sums + tid * K;
      for (uint64_t j = 0; j < K; j++)
        sums[j] = 0.0;
      for (uint64_t i = ifirst; i <= ilast; i++) {
        for (uint64_t j = 0; j < K; j++)
          sums[j] += V[i * K + j];
      }
      break;
    }

    case TK_ITQ_CENTER_SUB: {
      double *V = data->V0;
      double *means = data->center_sums;
      for (uint64_t i = ifirst; i <= ilast; i++) {
        for (uint64_t j = 0; j < K; j++)
          V[i * K + j] -= means[j];
      }
      break;
    }

    case TK_ITQ_SIGN: {
      double *V = data->V0;
      double *B = data->B;
      for (uint64_t i = ifirst; i <= ilast; i++) {
        for (uint64_t j = 0; j < K; j++)
          B[i * K + j] = (V[i * K + j] >= 0.0) ? 1.0 : -1.0;
      }
      break;
    }

    case TK_ITQ_PARTIAL_BTX: {
      double *B = data->B;
      double *X = data->X;
      double *partial = data->BtX_partials + tid * K * K;
      uint64_t n_rows = ilast - ifirst + 1;
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  K, K, n_rows,
                  1.0,
                  B + ifirst * K, K,
                  X + ifirst * K, K,
                  0.0,
                  partial, K);
      break;
    }

    case TK_ITQ_OBJECTIVE: {
      double *B = data->B;
      double *V = data->V1;
      double sum = 0.0;
      for (uint64_t i = ifirst; i <= ilast; i++) {
        for (uint64_t j = 0; j < K; j++) {
          double d = B[i * K + j] - V[i * K + j];
          sum += d * d;
        }
      }
      data->obj_partials[tid] = sum;
      break;
    }
  }
}

typedef struct {
  double a;
  double b;
} tk_dbq_threshold_t;

static inline tk_dbq_threshold_t tk_dbq_learn_threshold(
  double *values,
  uint64_t N
) {
  tk_dbq_threshold_t result = {0.0, 0.0};
  uint64_t n_neg = 0, n_pos = 0;
  for (uint64_t i = 0; i < N; i++) {
    if (values[i] <= 0.0) n_neg++;
    else n_pos++;
  }
  if (n_neg == 0 || n_pos == 0) {
    result.a = -1e-6;
    result.b = 1e-6;
    return result;
  }
  double *neg_sorted = malloc(n_neg * sizeof(double));
  double *pos_sorted = malloc(n_pos * sizeof(double));
  uint64_t neg_idx = 0, pos_idx = 0;
  for (uint64_t i = 0; i < N; i++) {
    if (values[i] <= 0.0) neg_sorted[neg_idx++] = values[i];
    else pos_sorted[pos_idx++] = values[i];
  }
  ks_introsort(tk_dvec_desc, n_neg, neg_sorted);
  ks_introsort(tk_dvec_asc, n_pos, pos_sorted);
  double max_F = -1e308;
  uint64_t s1_size = n_neg;
  uint64_t s3_size = n_pos;
  double sum_s2 = 0.0;
  double sum_s1 = 0.0;
  for (uint64_t i = 0; i < n_neg; i++)
    sum_s1 += neg_sorted[i];
  double sum_s3 = 0.0;
  for (uint64_t i = 0; i < n_pos; i++)
    sum_s3 += pos_sorted[i];
  uint64_t s2_neg_count = 0;
  uint64_t s2_pos_count = 0;
  while (s1_size > 0 || s3_size > 0) {
    if (sum_s2 <= 0.0 && s3_size > 0) {
      double val = pos_sorted[s2_pos_count];
      sum_s2 += val;
      sum_s3 -= val;
      s2_pos_count++;
      s3_size--;
    } else if (s1_size > 0) {
      double val = neg_sorted[n_neg - s1_size];
      sum_s2 += val;
      sum_s1 -= val;
      s2_neg_count++;
      s1_size--;
    } else {
      break;
    }
    if (s1_size > 0 && s3_size > 0) {
      double F = (sum_s1 * sum_s1) / (double)s1_size + (sum_s3 * sum_s3) / (double)s3_size;
      if (F > max_F) {
        max_F = F;
        result.a = neg_sorted[n_neg - s1_size];
        result.b = (s2_pos_count > 0) ? pos_sorted[s2_pos_count - 1] : neg_sorted[0];
      }
    }
  }
  free(neg_sorted);
  free(pos_sorted);
  return result;
}

static inline tk_dbq_threshold_t* tk_dbq_learn_thresholds(
  double *X,
  uint64_t N,
  uint64_t K_dims
) {
  tk_dbq_threshold_t *thresholds = malloc(K_dims * sizeof(tk_dbq_threshold_t));
  double *dim_values = malloc(N * sizeof(double));
  for (uint64_t j = 0; j < K_dims; j++) {
    for (uint64_t i = 0; i < N; i++) {
      dim_values[i] = X[i * K_dims + j];
    }
    thresholds[j] = tk_dbq_learn_threshold(dim_values, N);
  }
  free(dim_values);
  return thresholds;
}

static inline void tk_itq_dbq(
  char *out,
  double *X,
  uint64_t N,
  uint64_t K,
  tk_dbq_threshold_t *thresholds
) {
  uint64_t K_dims = K / 2;
  for (uint64_t j = 0; j < K_dims; j++) {
    double a = thresholds[j].a;
    double b = thresholds[j].b;
    uint64_t bit_idx_0 = j * 2;
    uint64_t bit_idx_1 = j * 2 + 1;
    for (uint64_t i = 0; i < N; i++) {
      double val = X[i * K_dims + j];
      if (val <= a) {
        out[i * TK_CVEC_BITS_BYTES(K) + TK_CVEC_BITS_BYTE(bit_idx_1)] |= ((uint8_t)1 << TK_CVEC_BITS_BIT(bit_idx_1));
      } else if (val > b) {
        out[i * TK_CVEC_BITS_BYTES(K) + TK_CVEC_BITS_BYTE(bit_idx_0)] |= ((uint8_t)1 << TK_CVEC_BITS_BIT(bit_idx_0));
      }
    }
  }
}

static inline void tk_itq_dbq_full(
  char *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  tk_dbq_threshold_t *thresholds = tk_dbq_learn_thresholds(X, N, K / 2);
  tk_itq_dbq(out, X, N, K, thresholds);
  free(thresholds);
}

static inline void tk_itq_sign (
  char *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  for (uint64_t j = 0; j < K; j ++)
    for (uint64_t i = 0; i < N; i ++)
      if (X[i * K + j] >= 0.0)
        out[i * TK_CVEC_BITS_BYTES(K) + TK_CVEC_BITS_BYTE(j)] |= ((uint8_t) 1 << TK_CVEC_BITS_BIT(j));
}

static inline void tk_itq_median (
  lua_State *L,
  char *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  double *col = tk_malloc(L, N * sizeof(double));
  for (uint64_t j = 0; j < K; j ++) {
    for (uint64_t i = 0; i < N; i ++)
      col[i] = X[i * K + j];
    ks_introsort(tk_dvec_asc, N, col);
    double med = (N & 1) ? col[N / 2] : 0.5 * (col[N / 2 - 1] + col[N / 2]);
    for (uint64_t i = 0; i < N; i ++)
      if (X[i * K + j] >= med)
        out[i * TK_CVEC_BITS_BYTES(K) + TK_CVEC_BITS_BYTE(j)] |= ((uint8_t) 1 << TK_CVEC_BITS_BIT(j));
  }
  free(col);
}

static inline void tk_itq_encode (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_dims,
  uint64_t max_iterations,
  double tolerance,
  int i_each,
  unsigned int n_threads
) {
  const uint64_t K = n_dims;
  const size_t N = codes->n / K;
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);
  double *X = tk_malloc(L, N * K * sizeof(double));
  double *R = tk_malloc(L, K*K*sizeof(double));
  double *V0 = tk_malloc(L, N * K*sizeof(double));
  double *V1 = tk_malloc(L, N * K*sizeof(double));
  double *B = tk_malloc(L, N * K*sizeof(double));
  double *BtV = tk_malloc(L, K*K*sizeof(double));
  double *U = tk_malloc(L, K*K*sizeof(double));
  double *S = tk_malloc(L, K  *sizeof(double));
  double *VT = tk_malloc(L, K*K*sizeof(double));
  double *superb= tk_malloc(L, (K-1)*sizeof(double));
  memcpy(X, codes->a, N * K * sizeof(double));
  tk_dvec_center(X, N, K);
  for (uint64_t i = 0; i < K; i ++)
    for (uint64_t j = 0; j < K; j ++)
      R[i * K + j] = (i==j? 1.0 : 0.0);

  // Setup threading
  tk_itq_data_t thread_data;
  thread_data.X = X;
  thread_data.R = R;
  thread_data.V0 = V0;
  thread_data.V1 = V1;
  thread_data.B = B;
  thread_data.N = N;
  thread_data.K = K;
  thread_data.BtX_partials = tk_malloc(L, n_threads * K * K * sizeof(double));
  thread_data.obj_partials = tk_malloc(L, n_threads * sizeof(double));
  thread_data.center_sums = tk_malloc(L, n_threads * K * sizeof(double));

  tk_itq_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_itq_thread_t));
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_itq_worker);
  thread_data.pool = pool;

  for (unsigned int i = 0; i < n_threads; i++) {
    threads[i].data = &thread_data;
    threads[i].thread_idx = i;
    pool->threads[i].data = &threads[i];
    tk_thread_range(i, n_threads, N, &threads[i].ifirst, &threads[i].ilast);
  }

  double last_obj = DBL_MAX, first_obj = 0.0;
  uint64_t it = 0;
  for (it = 0; it < max_iterations; it ++) {
    // V0 = X * R (threaded)
    tk_threads_signal(pool, TK_ITQ_MATVEC_XR, 0);

    // Center V0 (threaded two-pass)
    tk_threads_signal(pool, TK_ITQ_CENTER_SUM, 0);
    // Reduce sums from all threads
    for (uint64_t j = 0; j < K; j++) {
      double sum = 0.0;
      for (unsigned int t = 0; t < n_threads; t++)
        sum += thread_data.center_sums[t * K + j];
      thread_data.center_sums[j] = sum / (double) N;
    }
    tk_threads_signal(pool, TK_ITQ_CENTER_SUB, 0);

    // B = sign(V0) (threaded)
    tk_threads_signal(pool, TK_ITQ_SIGN, 0);

    // BtV = B^T * X (threaded with reduction)
    tk_threads_signal(pool, TK_ITQ_PARTIAL_BTX, 0);
    // Reduce partial results
    memset(BtV, 0, K * K * sizeof(double));
    for (unsigned int t = 0; t < n_threads; t++) {
      double *partial = thread_data.BtX_partials + t * K * K;
      for (uint64_t i = 0; i < K * K; i++)
        BtV[i] += partial[i];
    }
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR,'A','A',
                              K,K,BtV,K,S,U,K,VT,K,superb);
    if (info != 0) {
      tk_threads_destroy(pool);
      free(threads);
      free(thread_data.BtX_partials);
      free(thread_data.obj_partials);
      free(thread_data.center_sums);
      free(superb);
      free(VT);
      free(S);
      free(U);
      free(BtV);
      free(B);
      free(V1);
      free(V0);
      free(R);
      free(X);
      luaL_error(L, "ITQ SVD failed to converge (info=%d)", info);
    }

    // R = VT^T * U^T (K×K, sequential - too small to parallelize)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                K, K, K, 1.0, VT, K, U, K, 0.0, R, K);

    // V1 = X * R (threaded)
    tk_threads_signal(pool, TK_ITQ_MATVEC_XR_FINAL, 0);

    // Compute objective (threaded with reduction)
    tk_threads_signal(pool, TK_ITQ_OBJECTIVE, 0);
    double obj = 0.0;
    for (unsigned int t = 0; t < n_threads; t++)
      obj += thread_data.obj_partials[t];
    if (it == 0)
      first_obj = obj;
    if (it > 0 && fabs(last_obj - obj) < tolerance * fabs(obj))
      break;
    last_obj = obj;
  }

  // Final rotation and binarization
  tk_threads_signal(pool, TK_ITQ_MATVEC_XR_FINAL, 0);
  tk_dvec_center(V1, N, K);
  tk_itq_sign(out->a, V1, N, K);

  // Cleanup threading
  tk_threads_destroy(pool);
  lua_pop(L, 1); // threads
  free(threads);
  free(thread_data.BtX_partials);
  free(thread_data.obj_partials);
  free(thread_data.center_sums);

  free(superb);
  free(VT);
  free(S);
  free(U);
  free(BtV);
  free(B);
  free(V0);
  free(V1);
  free(R);
  free(X);
  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) it);
    lua_pushnumber(L, first_obj);
    lua_pushnumber(L, last_obj);
    lua_call(L, 3, 0);
  }
}

static inline void tk_sr_encode (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_dims,
  uint64_t max_iterations,
  double tolerance,
  int i_each,
  unsigned int n_threads
) {
  const uint64_t K = n_dims;
  const size_t N = codes->n / K;
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);
  double *F = tk_malloc(L, N * K * sizeof(double));
  double *Q = tk_malloc(L, K * K * sizeof(double));
  double *V0 = tk_malloc(L, N * K * sizeof(double));
  double *V1 = tk_malloc(L, N * K * sizeof(double));
  double *B = tk_malloc(L, N * K * sizeof(double));
  double *BtF = tk_malloc(L, K * K * sizeof(double));
  double *U = tk_malloc(L, K * K * sizeof(double));
  double *S = tk_malloc(L, K * sizeof(double));
  double *VT = tk_malloc(L, K * K * sizeof(double));
  double *superb = tk_malloc(L, (K-1) * sizeof(double));
  tk_rvec_t *rank_vec = tk_rvec_create(0, N, 0, 0);
  tk_rank_t *ranks = rank_vec->a;
  memcpy(F, codes->a, N * K * sizeof(double));
  tk_dvec_center(F, N, K);
  for (uint64_t i = 0; i < K; i++)
    for (uint64_t j = 0; j < K; j++)
      Q[i * K + j] = (i == j ? 1.0 : 0.0);
  double last_obj = DBL_MAX, first_obj = 0.0;
  uint64_t it = 0;
  for (it = 0; it < max_iterations; it++) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, F, K, Q, K, 0.0, V0, K);
    for (uint64_t j = 0; j < K; j++) {
      for (size_t i = 0; i < N; i++)
        ranks[i] = tk_rank((int64_t) i, V0[i * K + j]);
      ks_introsort(tk_rvec_desc, N, ranks);
      size_t half = N / 2;
      for (size_t i = 0; i < N; i++) {
        size_t idx = (size_t) ranks[i].i;
        B[idx * K + j] = (i < half) ? 1.0 : -1.0;
      }
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, N, 1.0, B, K, F, K, 0.0, BtF, K);
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', K, K, BtF, K, S, U, K, VT, K, superb);
    if (info != 0) {
      tk_rvec_destroy(rank_vec);
      free(superb);
      free(VT);
      free(S);
      free(U);
      free(BtF);
      free(B);
      free(V1);
      free(V0);
      free(Q);
      free(F);
      luaL_error(L, "SR SVD failed to converge (info=%d)", info);
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, K, K, K, 1.0, VT, K, U, K, 0.0, Q, K);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, F, K, Q, K, 0.0, V1, K);
    double obj = 0.0;
    for (size_t idx = 0; idx < N * K; idx++) {
      double d = B[idx] - V1[idx];
      obj += d * d;
    }
    if (it == 0)
      first_obj = obj;
    if (it > 0 && fabs(last_obj - obj) < tolerance * fabs(obj))
      break;
    last_obj = obj;
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, F, K, Q, K, 0.0, V1, K);
  for (uint64_t j = 0; j < K; j++) {
    for (size_t i = 0; i < N; i++)
      ranks[i] = tk_rank((int64_t) i, V1[i * K + j]);
    ks_introsort(tk_rvec_desc, N, ranks);
    size_t half = N / 2;
    for (size_t i = 0; i < half; i++) {
      size_t sample_idx = (size_t) ranks[i].i;
      size_t byte_idx = sample_idx * TK_CVEC_BITS_BYTES(K) + (j / 8);
      uint8_t bit_pos = j % 8;
      out->a[byte_idx] |= (1 << bit_pos);
    }
  }
  tk_rvec_destroy(rank_vec);
  free(superb);
  free(VT);
  free(S);
  free(U);
  free(BtF);
  free(B);
  free(V1);
  free(V0);
  free(Q);
  free(F);
  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) it);
    lua_pushnumber(L, first_obj);
    lua_pushnumber(L, last_obj);
    lua_call(L, 3, 0);
  }
}

static inline void tk_sr_ranking_encode (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_dims,
  tk_ivec_t *adj_ids,
  tk_ivec_t *adj_offsets,
  tk_ivec_t *adj_neighbors,
  tk_dvec_t *adj_weights,
  uint64_t max_iterations,
  double tolerance,
  int i_each,
  unsigned int n_threads
) {
  const uint64_t K = n_dims;
  const size_t N = codes->n / K;

  // Create output binary codes
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);

  // Allocate working memory
  double *F = tk_malloc(L, N * K * sizeof(double));
  double *Q = tk_malloc(L, K * K * sizeof(double));
  double *V0 = tk_malloc(L, N * K * sizeof(double));
  double *V1 = tk_malloc(L, N * K * sizeof(double));
  double *B = tk_malloc(L, N * K * sizeof(double));
  double *F_weighted = tk_malloc(L, N * K * sizeof(double));
  double *B_weighted = tk_malloc(L, N * K * sizeof(double));
  double *BtF = tk_malloc(L, K * K * sizeof(double));
  double *U = tk_malloc(L, K * K * sizeof(double));
  double *S = tk_malloc(L, K * sizeof(double));
  double *VT = tk_malloc(L, K * K * sizeof(double));
  double *superb = tk_malloc(L, (K-1) * sizeof(double));
  double *weights = tk_malloc(L, N * sizeof(double));

  tk_rvec_t *rank_vec = tk_rvec_create(NULL, N, 0, 0);
  if (!rank_vec) {
    free(weights);
    free(superb); free(VT); free(S); free(U); free(BtF);
    free(B_weighted); free(F_weighted); free(B); free(V1); free(V0); free(Q); free(F);
    luaL_error(L, "SR ranking: allocation failed");
  }
  tk_rank_t *ranks = rank_vec->a;

  // Build id->index map for adjacency lookup
  tk_iumap_t *id_map = tk_iumap_create(NULL, N);
  if (!id_map) {
    tk_rvec_destroy(rank_vec);
    free(weights);
    free(superb); free(VT); free(S); free(U); free(BtF);
    free(B_weighted); free(F_weighted); free(B); free(V1); free(V0); free(Q); free(F);
    luaL_error(L, "SR ranking: allocation failed");
  }
  int kha;
  for (size_t i = 0; i < N; i++) {
    uint32_t khi = tk_iumap_put(id_map, adj_ids->a[i], &kha);
    tk_iumap_setval(id_map, khi, (int64_t)i);
  }

  // Allocate temporary structures for loop (reused each iteration)
  size_t bytes_per_code = TK_CVEC_BITS_BYTES(K);
  unsigned char *B_binary = malloc(N * bytes_per_code);
  if (!B_binary) {
    tk_iumap_destroy(id_map);
    tk_rvec_destroy(rank_vec);
    free(weights);
    free(superb); free(VT); free(S); free(U); free(BtF);
    free(B_weighted); free(F_weighted); free(B); free(V1); free(V0); free(Q); free(F);
    luaL_error(L, "SR ranking: allocation failed");
  }

  tk_dumap_t *rank_buffer_b = tk_dumap_create(NULL, 0);
  tk_pvec_t *bin_ranks = tk_pvec_create(NULL, 0, 0, 0);
  tk_ivec_t *count_buffer = tk_ivec_create(NULL, K + 1, 0, 0);
  tk_dvec_t *avgrank_buffer = tk_dvec_create(NULL, K + 1, 0, 0);
  if (!rank_buffer_b || !bin_ranks || !count_buffer || !avgrank_buffer) {
    if (bin_ranks) tk_pvec_destroy(bin_ranks);
    if (rank_buffer_b) tk_dumap_destroy(rank_buffer_b);
    if (count_buffer) tk_ivec_destroy(count_buffer);
    if (avgrank_buffer) tk_dvec_destroy(avgrank_buffer);
    free(B_binary);
    tk_iumap_destroy(id_map);
    tk_rvec_destroy(rank_vec);
    free(weights);
    free(superb); free(VT); free(S); free(U); free(BtF);
    free(B_weighted); free(F_weighted); free(B); free(V1); free(V0); free(Q); free(F);
    luaL_error(L, "SR ranking: allocation failed");
  }

  // Initialize
  memcpy(F, codes->a, N * K * sizeof(double));
  tk_dvec_center(F, N, K);
  for (uint64_t i = 0; i < K; i++)
    for (uint64_t j = 0; j < K; j++)
      Q[i * K + j] = (i == j ? 1.0 : 0.0);

  // Initialize weights to uniform
  for (size_t i = 0; i < N; i++)
    weights[i] = 1.0;

  double last_obj = DBL_MAX, first_obj = 0.0;
  uint64_t it = 0;

  for (it = 0; it < max_iterations; it++) {
    // Rotate: V0 = F * Q
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, F, K, Q, K, 0.0, V0, K);

    // Rank-based quantization
    for (uint64_t j = 0; j < K; j++) {
      for (size_t i = 0; i < N; i++)
        ranks[i] = tk_rank((int64_t) i, V0[i * K + j]);
      ks_introsort(tk_rvec_desc, N, ranks);
      size_t half = N / 2;
      for (size_t i = 0; i < N; i++) {
        size_t idx = (size_t) ranks[i].i;
        B[idx * K + j] = (i < half) ? 1.0 : -1.0;
      }
    }

    // Convert B to binary codes for Hamming distance computation
    memset(B_binary, 0, N * bytes_per_code);
    for (size_t i = 0; i < N; i++) {
      for (uint64_t j = 0; j < K; j++) {
        if (B[i * K + j] > 0.0) {
          size_t byte_idx = i * bytes_per_code + (j / 8);
          uint8_t bit_pos = j % 8;
          B_binary[byte_idx] |= (1 << bit_pos);
        }
      }
    }

    // Compute per-node ranking quality (negative Spearman = worse = higher weight)
    for (size_t node_idx = 0; node_idx < (size_t)(adj_offsets->n - 1); node_idx++) {
      int64_t start = adj_offsets->a[node_idx];
      int64_t end = adj_offsets->a[node_idx + 1];

      if (end <= start) {
        weights[node_idx] = 1.0;
        continue;
      }

      // Build (neighbor_idx, hamming) pairs - reuse bin_ranks, clearing it first
      tk_pvec_clear(bin_ranks);
      unsigned char *node_code = B_binary + node_idx * bytes_per_code;

      for (int64_t j = start; j < end; j++) {
        int64_t neighbor_pos = adj_neighbors->a[j];
        uint32_t khi = tk_iumap_get(id_map, adj_ids->a[neighbor_pos]);
        if (khi == tk_iumap_end(id_map))
          continue;
        size_t neighbor_idx = (size_t)tk_iumap_val(id_map, khi);

        unsigned char *neighbor_code = B_binary + neighbor_idx * bytes_per_code;
        uint64_t hamming_dist = tk_cvec_bits_hamming(node_code, neighbor_code, K);
        tk_pvec_push(bin_ranks, tk_pair(neighbor_pos, (int64_t)hamming_dist));
      }

      // Compute Spearman correlation (counting sort inside, no pre-sort needed)
      double corr = tk_csr_spearman(adj_neighbors, adj_weights, start, end, bin_ranks,
                                    K, count_buffer, avgrank_buffer, rank_buffer_b);

      // Weight = 1 - corr (higher weight for worse ranking)
      // Clamp to [0.1, 2.0] to avoid extreme weights
      double w = 1.0 - corr;
      if (w < 0.1) w = 0.1;
      if (w > 2.0) w = 2.0;
      weights[node_idx] = w;
    }

    // Weighted Procrustes: scale rows by sqrt(weight)
    for (size_t i = 0; i < N; i++) {
      double sqrt_w = sqrt(weights[i]);
      for (uint64_t j = 0; j < K; j++) {
        F_weighted[i * K + j] = sqrt_w * F[i * K + j];
        B_weighted[i * K + j] = sqrt_w * B[i * K + j];
      }
    }

    // Compute weighted correlation: BtF = B_weighted^T * F_weighted
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, N, 1.0, B_weighted, K, F_weighted, K, 0.0, BtF, K);

    // SVD
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', K, K, BtF, K, S, U, K, VT, K, superb);
    if (info != 0) {
      tk_pvec_destroy(bin_ranks);
      tk_dumap_destroy(rank_buffer_b);
      tk_ivec_destroy(count_buffer);
      tk_dvec_destroy(avgrank_buffer);
      free(B_binary);
      tk_iumap_destroy(id_map);
      tk_rvec_destroy(rank_vec);
      free(weights);
      free(superb); free(VT); free(S); free(U); free(BtF);
      free(B_weighted); free(F_weighted); free(B); free(V1); free(V0); free(Q); free(F);
      luaL_error(L, "SR ranking SVD failed to converge (info=%d)", info);
    }

    // Update Q = VT^T * U^T
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, K, K, K, 1.0, VT, K, U, K, 0.0, Q, K);

    // Compute objective
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, F, K, Q, K, 0.0, V1, K);
    double obj = 0.0;
    for (size_t idx = 0; idx < N * K; idx++) {
      double d = B[idx] - V1[idx];
      obj += d * d;
    }

    if (it == 0)
      first_obj = obj;
    if (it > 0 && fabs(last_obj - obj) < tolerance * fabs(obj))
      break;
    last_obj = obj;
  }

  // Final binarization
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, F, K, Q, K, 0.0, V1, K);
  for (uint64_t j = 0; j < K; j++) {
    for (size_t i = 0; i < N; i++)
      ranks[i] = tk_rank((int64_t) i, V1[i * K + j]);
    ks_introsort(tk_rvec_desc, N, ranks);
    size_t half = N / 2;
    for (size_t i = 0; i < half; i++) {
      size_t sample_idx = (size_t) ranks[i].i;
      size_t byte_idx = sample_idx * TK_CVEC_BITS_BYTES(K) + (j / 8);
      uint8_t bit_pos = j % 8;
      out->a[byte_idx] |= (1 << bit_pos);
    }
  }

  // Cleanup
  tk_pvec_destroy(bin_ranks);
  tk_dumap_destroy(rank_buffer_b);
  tk_ivec_destroy(count_buffer);
  tk_dvec_destroy(avgrank_buffer);
  free(B_binary);
  tk_iumap_destroy(id_map);
  tk_rvec_destroy(rank_vec);
  free(weights);
  free(superb); free(VT); free(S); free(U); free(BtF);
  free(B_weighted); free(F_weighted); free(B); free(V1); free(V0); free(Q); free(F);

  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) it);
    lua_pushnumber(L, first_obj);
    lua_pushnumber(L, last_obj);
    lua_call(L, 3, 0);
  }
}

#endif
