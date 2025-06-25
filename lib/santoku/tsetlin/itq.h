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

static inline void tk_itq_encode (
  lua_State *L,
  tk_dvec_t *codes,
  uint64_t n_hidden,
  uint64_t iterations,
  int i_each
) {
  const size_t N = codes->n / n_hidden;

  // Data matrix
  double *X = codes->a;

  // Rotation matrix, initialized to identity
  double *R = tk_malloc(L, n_hidden * n_hidden * sizeof(double));
  for (uint64_t i = 0; i < n_hidden; i ++)
    for (uint64_t j = 0; j < n_hidden; j ++)
      R[i * n_hidden + j] = (i == j) ? 1.0 : 0.0;

  // Temp buffers
  double *V = tk_malloc(L, N * n_hidden * sizeof(double));
  double *B = tk_malloc(L, N * n_hidden * sizeof(double));
  double *BtV = tk_malloc(L, n_hidden * n_hidden * sizeof(double));
  double *U = tk_malloc(L, n_hidden * n_hidden * sizeof(double));
  double *S = tk_malloc(L, n_hidden * sizeof(double));
  double *VT = tk_malloc(L, n_hidden * n_hidden * sizeof(double));
  double *superb = tk_malloc(L, (n_hidden - 1) * sizeof(double));
  lapack_int info;

  tk_ivec_t *out = tk_ivec_create(L, 0, 0, 0);

  for (uint64_t it = 0; it < iterations; it ++)
  {
    // V = X * R  (N x n_hidden)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, n_hidden, n_hidden,
                1.0, X, n_hidden,
                R, n_hidden,
                0.0, V, n_hidden);

    // B = sign(V)
    for (size_t idx = 0; idx < N * n_hidden; idx ++)
      B[idx] = (V[idx] >= 0.0 ? 1.0 : -1.0);

    // BtV = B^T * V  (n_hidden x n_hidden)
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n_hidden, n_hidden, N,
                1.0, B, n_hidden,
                V, n_hidden,
                0.0, BtV, n_hidden);

    // Compute SVD of BtV: BtV = U * diag(S) * VT
    // Use LAPACK: gesvd
    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A',
                          n_hidden, n_hidden,
                          BtV, n_hidden,
                          S, U, n_hidden,
                          VT, n_hidden,
                          superb);
    if (info > 0) {
      // SVD did not converge
      break;
    }

    // R = V * Uᵀ  ==  VTᵀ * Uᵀ
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans,
                n_hidden, n_hidden, n_hidden,
                1.0, VT, n_hidden,
                     U,  n_hidden,
                0.0, R,  n_hidden);

    // Log and check for early exit via Lua callback
    if (i_each >= 0) {

      double obj = 0.0;
      for (size_t idx = 0; idx < N * n_hidden; ++idx) {
        double diff = B[idx] - V[idx];
        obj += diff * diff;
      }

      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t) it);
      lua_pushnumber(L, obj);
      lua_call(L, 2, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && !lua_toboolean(L, -1)) {
        lua_pop(L, 1);
        break;
      }
      lua_pop(L, 1);
    }
  }

  // Final codes: sign(X * R)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, n_hidden, n_hidden,
              1.0, X, n_hidden,
              R, n_hidden,
              0.0, V, n_hidden);

  // Emit set bits
  for (size_t i = 0; i < N; i ++)
    for (size_t j = 0; j < n_hidden; j ++)
      if (V[i * n_hidden + j] >= 0.0)
        tk_ivec_push(out, (int64_t) (i * n_hidden + j));

  // Cleanup
  // tk_ivec_shrink(L, out);
  free(superb);
  free(R);
  free(V);
  free(B);
  free(BtV);
  free(U);
  free(S);
  free(VT);
}

#endif
