#include <lua.h>
#include <lauxlib.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>

static inline uint64_t elm_splitmix64 (uint64_t *s)
{
  uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static inline double elm_rand_normal (uint64_t *s)
{
  double u1 = ((double)(elm_splitmix64(s) >> 11) + 0.5) / (double)(1ULL << 53);
  double u2 = ((double)(elm_splitmix64(s) >> 11) + 0.5) / (double)(1ULL << 53);
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static int tk_elm_project_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "csc_offsets");
  int has_sparse = !lua_isnil(L, -1);
  tk_ivec_t *csc_off = has_sparse ? tk_ivec_peek(L, -1, "csc_offsets") : NULL;
  lua_pop(L, 1);

  tk_ivec_t *csc_idx = NULL;
  int64_t n_tokens = 0;
  double *fw = NULL;
  if (has_sparse) {
    lua_getfield(L, 1, "csc_indices");
    csc_idx = tk_ivec_peek(L, -1, "csc_indices");
    lua_pop(L, 1);
    n_tokens = (int64_t)tk_lua_fcheckunsigned(L, 1, "project", "n_tokens");
    lua_getfield(L, 1, "feature_weights");
    if (!lua_isnil(L, -1))
      fw = tk_dvec_peek(L, -1, "feature_weights")->a;
    lua_pop(L, 1);
  }

  lua_getfield(L, 1, "dense_features");
  int has_dense = !lua_isnil(L, -1);
  double *df = has_dense ? tk_dvec_peek(L, -1, "dense_features")->a : NULL;
  lua_pop(L, 1);

  int64_t n_dense = 0;
  if (has_dense)
    n_dense = (int64_t)tk_lua_fcheckunsigned(L, 1, "project", "n_dense");

  if (!has_sparse && !has_dense)
    return luaL_error(L, "project: csc_offsets or dense_features required");

  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "project", "n_samples");
  int64_t n_hidden = (int64_t)tk_lua_fcheckunsigned(L, 1, "project", "n_hidden");

  lua_getfield(L, 1, "seed");
  uint64_t seed = lua_isnumber(L, -1) ? (uint64_t)lua_tointeger(L, -1) : 42;
  lua_pop(L, 1);

  int sidecar = has_sparse && has_dense;
  int64_t out_cols = sidecar ? n_hidden + n_dense : n_hidden;
  uint64_t total = (uint64_t)(n_samples * out_cols);

  tk_dvec_t *out;
  lua_getfield(L, 1, "out");
  if (!lua_isnil(L, -1)) {
    out = tk_dvec_peek(L, -1, "out");
    tk_dvec_ensure(out, total);
    out->n = total;
  } else {
    lua_pop(L, 1);
    out = tk_dvec_create(L, total, NULL, NULL);
  }

  double *inv_norm = NULL;
  if (has_sparse) {
    inv_norm = (double *)calloc((uint64_t)n_samples, sizeof(double));
    for (int64_t t = 0; t < n_tokens; t++) {
      double fw_sq = fw ? fw[t] * fw[t] : 1.0;
      int64_t lo = csc_off->a[t], hi = csc_off->a[t + 1];
      for (int64_t i = lo; i < hi; i++)
        inv_norm[csc_idx->a[i]] += fw_sq;
    }
    for (int64_t s = 0; s < n_samples; s++) {
      double n2 = inv_norm[s];
      inv_norm[s] = n2 > 0.0 ? 1.0 / sqrt(n2) : 0.0;
    }
  }

  #pragma omp parallel
  {
    double *col = (double *)malloc((uint64_t)n_samples * sizeof(double));
    #pragma omp for schedule(static)
    for (int64_t h = 0; h < n_hidden; h++) {
      memset(col, 0, (uint64_t)n_samples * sizeof(double));
      uint64_t rng = seed ^ ((uint64_t)h * 0x517cc1b727220a95ULL);
      elm_splitmix64(&rng);
      if (has_sparse) {
        for (int64_t t = 0; t < n_tokens; t++) {
          double w = elm_rand_normal(&rng);
          if (fw) w *= fw[t];
          int64_t lo = csc_off->a[t];
          int64_t hi = csc_off->a[t + 1];
          for (int64_t i = lo; i < hi; i++)
            col[csc_idx->a[i]] += w * inv_norm[csc_idx->a[i]];
        }
      }
      if (!sidecar && has_dense) {
        for (int64_t d = 0; d < n_dense; d++) {
          double w = elm_rand_normal(&rng);
          for (int64_t s = 0; s < n_samples; s++)
            col[s] += w * df[s * n_dense + d];
        }
      }
      double bias = elm_rand_normal(&rng);
      for (int64_t s = 0; s < n_samples; s++) {
        double v = col[s] + bias;
        out->a[s * out_cols + h] = v > 0.0 ? v : 0.0;
      }
    }
    free(col);
  }

  if (sidecar) {
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < n_samples; s++) {
      for (int64_t d = 0; d < n_dense; d++)
        out->a[s * out_cols + n_hidden + d] = df[s * n_dense + d];
    }
  }

  #pragma omp parallel for schedule(static)
  for (int64_t j = 0; j < out_cols; j++) {
    double sum = 0, sum2 = 0;
    for (int64_t i = 0; i < n_samples; i++) {
      double v = out->a[i * out_cols + j];
      sum += v;
      sum2 += v * v;
    }
    double mean = sum / (double)n_samples;
    double var = sum2 / (double)n_samples - mean * mean;
    double inv_std = var > 1e-24 ? 1.0 / sqrt(var) : 0.0;
    for (int64_t i = 0; i < n_samples; i++)
      out->a[i * out_cols + j] *= inv_std;
  }

  free(inv_norm);
  return 1;
}

static luaL_Reg tk_elm_fns[] = {
  { "project", tk_elm_project_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_elm (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_elm_fns, 0);
  return 1;
}
