#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/dvec.h>
#include <santoku/ivec.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>

static inline int tm_encode (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "chol");
  tk_dvec_t *chol = tk_dvec_peek(L, -1, "chol");

  uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "spectral", "n_samples");
  uint64_t n_landmarks = tk_lua_fcheckunsigned(L, 1, "spectral", "n_landmarks");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "spectral", "n_dims");

  if (chol->n != n_samples * n_landmarks)
    return luaL_error(L, "chol size (%llu) != n_samples * n_landmarks (%llu)",
      (unsigned long long)chol->n, (unsigned long long)(n_samples * n_landmarks));

  if (n_dims > n_landmarks)
    return luaL_error(L, "n_dims (%llu) must be <= n_landmarks (%llu)",
      (unsigned long long)n_dims, (unsigned long long)n_landmarks);

  tk_dvec_t *chol_centered = tk_dvec_create(L, n_samples * n_landmarks, 0, 0);
  chol_centered->n = n_samples * n_landmarks;
  memcpy(chol_centered->a, chol->a, n_samples * n_landmarks * sizeof(double));

  #pragma omp parallel for schedule(static)
  for (uint64_t j = 0; j < n_landmarks; j++) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n_samples; i++)
      sum += chol_centered->a[i * n_landmarks + j];
    double mu = sum / (double)n_samples;
    for (uint64_t i = 0; i < n_samples; i++)
      chol_centered->a[i * n_landmarks + j] -= mu;
  }

  tk_dvec_t *gram = tk_dvec_create(L, n_landmarks * n_landmarks, 0, 0);
  gram->n = n_landmarks * n_landmarks;

  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    (int)n_landmarks, (int)n_landmarks, (int)n_samples,
    1.0, chol_centered->a, (int)n_landmarks, chol_centered->a, (int)n_landmarks,
    0.0, gram->a, (int)n_landmarks);

  tk_dvec_t *eigenvalues_raw = tk_dvec_create(L, n_dims, 0, 0);
  double *evecs_raw = malloc(n_landmarks * n_landmarks * sizeof(double));
  int *isuppz = malloc(2 * n_dims * sizeof(int));
  int m = 0;

  int info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U',
    (int)n_landmarks, gram->a, (int)n_landmarks,
    0.0, 0.0, (int)(n_landmarks - n_dims + 1), (int)n_landmarks,
    0.0, &m, eigenvalues_raw->a, evecs_raw, (int)n_landmarks, isuppz);

  free(isuppz);

  if (info != 0) {
    free(evecs_raw);
    return luaL_error(L, "LAPACKE_dsyevr failed with info=%d", info);
  }

  tk_dvec_t *eigenvectors = tk_dvec_create(L, n_landmarks * n_dims, 0, 0);
  tk_dvec_t *eigenvalues = tk_dvec_create(L, n_dims, 0, 0);

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_landmarks; i++) {
    for (uint64_t k = 0; k < n_dims; k++) {
      uint64_t col = n_dims - 1 - k;
      eigenvectors->a[i * n_dims + k] = evecs_raw[i * n_landmarks + col];
    }
  }
  for (uint64_t k = 0; k < n_dims; k++)
    eigenvalues->a[k] = eigenvalues_raw->a[n_dims - 1 - k];

  free(evecs_raw);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = lua_gettop(L);
  }

  if (i_each != -1) {
    for (uint64_t i = 0; i < n_dims; i++) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "eig");
      lua_pushinteger(L, (int64_t)i);
      lua_pushnumber(L, eigenvalues->a[i]);
      lua_pushboolean(L, 1);
      lua_call(L, 4, 0);
    }
    lua_pushvalue(L, i_each);
    lua_pushstring(L, "done");
    lua_pushinteger(L, 0);
    lua_call(L, 2, 0);
  }

  return 2;
}

static luaL_Reg tm_fns[] =
{
  { "encode", tm_encode },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_spectral (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_fns, 0);
  return 1;
}
