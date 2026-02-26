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
  tk_ivec_t *csc_off = tk_ivec_peek(L, -1, "csc_offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "csc_indices");
  tk_ivec_t *csc_idx = tk_ivec_peek(L, -1, "csc_indices");
  lua_pop(L, 1);

  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "project", "n_samples");
  int64_t n_tokens = (int64_t)tk_lua_fcheckunsigned(L, 1, "project", "n_tokens");
  int64_t n_hidden = (int64_t)tk_lua_fcheckunsigned(L, 1, "project", "n_hidden");

  lua_getfield(L, 1, "seed");
  uint64_t seed = lua_isnumber(L, -1) ? (uint64_t)lua_tointeger(L, -1) : 42;
  lua_pop(L, 1);

  lua_getfield(L, 1, "feature_weights");
  double *fw = NULL;
  if (!lua_isnil(L, -1)) {
    tk_dvec_t *fw_vec = tk_dvec_peek(L, -1, "feature_weights");
    fw = fw_vec->a;
  }
  lua_pop(L, 1);

  uint64_t total = (uint64_t)(n_samples * n_hidden);
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
  memset(out->a, 0, total * sizeof(double));

  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (int64_t h = 0; h < n_hidden; h++) {
      uint64_t rng = seed ^ ((uint64_t)h * 0x517cc1b727220a95ULL);
      elm_splitmix64(&rng);
      for (int64_t t = 0; t < n_tokens; t++) {
        double w = elm_rand_normal(&rng);
        if (fw) w *= fw[t];
        int64_t lo = csc_off->a[t];
        int64_t hi = csc_off->a[t + 1];
        for (int64_t i = lo; i < hi; i++) {
          int64_t s = csc_idx->a[i];
          out->a[s * n_hidden + h] += w;
        }
      }
      double bias = elm_rand_normal(&rng);
      for (int64_t s = 0; s < n_samples; s++) {
        double v = out->a[s * n_hidden + h] + bias;
        out->a[s * n_hidden + h] = v > 0.0 ? v : 0.0;
      }
    }
  }

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
