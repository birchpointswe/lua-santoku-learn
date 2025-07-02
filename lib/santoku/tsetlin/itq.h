#ifndef TK_ITQ_H
#define TK_ITQ_H

#include <santoku/tsetlin/conf.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>

static inline void tk_itq_sign (
  tk_ivec_t *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  for (uint64_t j = 0; j < K; j++)
    for (uint64_t i = 0; i < N; i++)
      if (X[i*K + j] >= 0.0)
        tk_ivec_push(out, (int64_t)(i * K + j));
}

static inline void tk_itq_median (
  tk_ivec_t *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  double *col = malloc(N * sizeof(double));
  for (uint64_t j = 0; j < K; j++) {
    for (uint64_t i = 0; i < N; i++)
      col[i] = X[i*K + j];
    ks_introsort(tk_dvec_asc, N, col);
    double med = (N & 1) ? col[N/2] : 0.5 * (col[N/2 - 1] + col[N/2]);
    for (uint64_t i = 0; i < N; i++) {
      if (X[i*K + j] >= med) {
        tk_ivec_push(out, (int64_t)(i * K + j));
      }
    }
  }
  free(col);
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

  // Copy codes: TODO: parallel
  double *X = malloc(N * K * sizeof(double));
  memcpy(X, codes->a, N * K * sizeof(double));

  // Center: TODO: parallel
  for (uint64_t f = 0; f < n_dims; f ++) {
    double mu = 0.0;
    for (uint64_t i = 0; i < N; i ++)
      mu += X[i * n_dims + f];
    mu /= N;
    for (uint64_t i = 0; i < N; i ++)
      X[i * n_dims + f] -= mu;
  }

  // Normalize by variance: TODO: parallel
  for (uint64_t f = 0; f < n_dims; f++) {
    double norm = 0;
    for (uint64_t i = 0; i < N; i++)
      norm += X[i*n_dims + f] * X[i*n_dims + f];
    norm = sqrt(norm / N);
    if (norm > 0) {
      for (uint64_t i = 0; i < N; i++)
        X[i*n_dims + f] /= norm;
    }
  }

  tk_ivec_t *out = tk_ivec_create(L, 0, 0, 0); // returned via lua stack
  double *R = malloc(K*K*sizeof(double));
  double *V0 = malloc(N*K*sizeof(double));
  double *V1 = malloc(N*K*sizeof(double));
  double *B = malloc(N*K*sizeof(double));
  double *BtV = malloc(K*K*sizeof(double));
  double *U = malloc(K*K*sizeof(double));
  double *S = malloc(K  *sizeof(double));
  double *VT = malloc(K*K*sizeof(double));
  double *superb= malloc((K-1)*sizeof(double));
  for (uint64_t i = 0; i < K; i++)
    for (uint64_t j = 0; j < K; j++)
      R[i*K + j] = (i==j? 1.0 : 0.0);
  double last_obj = DBL_MAX, first_obj = 0.0;
  uint64_t it = 0;
  for (it = 0; it < max_iterations; it++) {
    // 1) project with the old rotation
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, K, K, 1.0, X, K, R, K, 0.0, V0, K);
    // 2) form sign‐matrix B
    for (size_t idx = 0; idx < N*K; idx++)
      B[idx] = (V0[idx] >= 0.0 ? 1.0 : -1.0);
    // 3) solve Procrustes: SVD of (B^T V0)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                K, K, N, 1.0, B, K, X, K, 0.0, BtV, K);
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR,'A','A',
                              K,K,BtV,K,S,U,K,VT,K,superb);
    if (info != 0)
      luaL_error(L, "ITQ SVD failed to converge (info=%d)", info);
    // R_new = V * U^T
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                K, K, K, 1.0, VT, K, U, K, 0.0, R, K);
    // 4) project with the new rotation
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, K, K, 1.0, X, K, R, K, 0.0, V1, K);
    // 5) measure objective ‖B - V1‖²
    double obj = 0.0;
    for (size_t idx = 0; idx < N*K; idx++) {
      double d = B[idx] - V1[idx];
      obj += d*d;
    }
    if (it == 0) first_obj = obj;
    if (obj > last_obj + 1e-8)
      fprintf(stderr, "ITQ warning: obj rose %.6f → %.6f at iter %llu\n",
              last_obj, obj, (unsigned long long)it);
    if (fabs(last_obj - obj) < tolerance * last_obj)
      break;
    last_obj = obj;
  }

  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) it);
    lua_pushnumber(L, first_obj);
    lua_pushnumber(L, last_obj);
    lua_call(L, 3, 0);
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, K, K, 1.0, X, K, R, K, 0.0, V1, K);
  tk_itq_sign(out, V1, N, K);

  // Cleanup
  tk_ivec_shrink(L, out);
  free(superb);
  free(R);
  free(B);
  free(BtV);
  free(U);
  free(S);
  free(VT);
  free(V0);
  free(V1);
  free(X);
}

#endif
