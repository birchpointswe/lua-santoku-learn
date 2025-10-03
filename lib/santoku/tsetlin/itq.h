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
#include <lapacke.h>
#include <cblas.h>

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

  // Allocate Lua-managed memory first (can longjmp)
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);

  // Now allocate malloc-based temporary buffers (tk_malloc throws on failure)
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
  double last_obj = DBL_MAX, first_obj = 0.0;
  uint64_t it = 0;

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
    if (info != 0) {
      // Cleanup before throwing
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

  // Complete final computation before callback
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, R, K, 0.0, V1, K);
  tk_dvec_center(V1, N, K);
  tk_itq_sign(out->a, V1, N, K);

  // Cleanup malloc'd buffers before any Lua API calls that can throw
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

  // Now safe to call Lua callback (no malloc leaks if it throws)
  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) it);
    lua_pushnumber(L, first_obj);
    lua_pushnumber(L, last_obj);
    lua_call(L, 3, 0);
  }

}

#endif
