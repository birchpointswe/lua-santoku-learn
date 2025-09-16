#ifndef TK_ITQ_H
#define TK_ITQ_H

#include <santoku/tsetlin/conf.h>
#include <santoku/cvec.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

#if __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#else
#include <cblas.h>
#endif

#if __has_include(<openblas/lapacke.h>)
#include <openblas/lapacke.h>
#else
#include <lapacke.h>
#endif

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
  char *out,
  double *X,
  uint64_t N,
  uint64_t K
) {
  double *col = malloc(N * sizeof(double));
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

  double *X = malloc(N * K * sizeof(double));
  memcpy(X, codes->a, N * K * sizeof(double));

  tk_dvec_center(X, N, K);

  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);
  double *R = malloc(K*K*sizeof(double));
  double *V0 = malloc(N * K*sizeof(double));
  double *V1 = malloc(N * K*sizeof(double));
  double *B = malloc(N * K*sizeof(double));
  double *BtV = malloc(K*K*sizeof(double));
  double *U = malloc(K*K*sizeof(double));
  double *S = malloc(K  *sizeof(double));
  double *VT = malloc(K*K*sizeof(double));
  double *superb= malloc((K-1)*sizeof(double));
  for (uint64_t i = 0; i < K; i ++)
    for (uint64_t j = 0; j < K; j ++)
      R[i * K + j] = (i==j? 1.0 : 0.0);
  double last_obj = DBL_MAX, first_obj = 0.0;
  uint64_t it = 0;

  openblas_set_num_threads((int) n_threads);

  for (it = 0; it < max_iterations; it ++) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, K, K, 1.0, X, K, R, K, 0.0, V0, K);
    tk_dvec_center(V0, N, K);
    for (size_t idx = 0; idx < N * K; idx ++)
      B[idx] = (V0[idx] >= 0.0 ? 1.0 : -1.0);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                K, K, N, 1.0, B, K, X, K, 0.0, BtV, K);
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR,'A','A',
                              K,K,BtV,K,S,U,K,VT,K,superb);
    if (info != 0)
      luaL_error(L, "ITQ SVD failed to converge (info=%d)", info);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                K, K, K, 1.0, VT, K, U, K, 0.0, R, K);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, K, K, 1.0, X, K, R, K, 0.0, V1, K);
    double obj = 0.0;
    for (size_t idx = 0; idx < N * K; idx ++) {
      double d = B[idx] - V1[idx];
      obj += d * d;
    }
    if (it == 0)
      first_obj = obj;
    if (it > 0 && fabs(last_obj - obj) < tolerance * fabs(obj))
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

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, R, K, 0.0, V1, K);
  tk_dvec_center(V1, N, K);
  tk_itq_sign(out->a, V1, N, K);

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
