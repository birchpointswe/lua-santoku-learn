#include <santoku/tsetlin/itq.h>

#include <float.h>
#include <lauxlib.h>
#include <lua.h>

static inline int tk_itq_encode_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  uint64_t max_iterations = tk_lua_foptunsigned(L, 1, "itq", "iterations", 1000);
  double tolerance = tk_lua_foptposdouble(L, 1, "itq", "tolerance", 1e-8);
  bool return_rotation = tk_lua_foptboolean(L, 1, "itq", "return_rotation", false);

  lua_getfield(L, 1, "rotation");
  tk_dvec_t *in_rotation = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "mean");
  tk_dvec_t *in_mean = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  if ((in_rotation && !in_mean) || (!in_rotation && in_mean))
    return luaL_error(L, "itq: must provide both rotation and mean, or neither");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_dvec_t *out_rotation = NULL;
  tk_dvec_t *out_mean = NULL;

  tk_itq_encode(L, codes, n_dims, max_iterations, tolerance, i_each,
                in_rotation, in_mean,
                return_rotation ? &out_rotation : NULL,
                return_rotation ? &out_mean : NULL);

  if (return_rotation && out_rotation && out_mean) {
    return 3;
  }
  return 1;
}

static inline int tk_itq_sign_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  tk_cvec_t *out = tk_cvec_create(L, codes->n / n_dims * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);
  tk_itq_sign(out->a, codes->a, codes->n / n_dims, n_dims);
  return 1;
}

static inline int tk_itq_median_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  bool return_thresholds = tk_lua_foptboolean(L, 1, "median", "return_thresholds", false);

  lua_getfield(L, 1, "thresholds");
  tk_dvec_t *in_thresholds = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  tk_cvec_t *out = tk_cvec_create(L, codes->n / n_dims * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);

  tk_dvec_t *out_thresholds = NULL;

  tk_itq_median(L, out->a, codes->a, codes->n / n_dims, n_dims,
                in_thresholds,
                return_thresholds ? &out_thresholds : NULL);

  if (return_thresholds && out_thresholds) {
    return 2;
  }
  return 1;
}

static inline int tk_itq_otsu_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  const char *metric = tk_lua_foptstring(L, 1, "itq", "metric", "variance");
  uint64_t n_bins = tk_lua_foptunsigned(L, 1, "itq", "n_bins", 0);
  bool minimize = tk_lua_foptboolean(L, 1, "itq", "minimize", false);
  bool return_thresholds = tk_lua_foptboolean(L, 1, "otsu", "return_thresholds", false);

  lua_getfield(L, 1, "indices");
  tk_ivec_t *in_indices = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "thresholds");
  tk_dvec_t *in_thresholds = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  if ((in_indices && !in_thresholds) || (!in_indices && in_thresholds))
    return luaL_error(L, "otsu: must provide both indices and thresholds, or neither");

  tk_cvec_t *out = tk_cvec_create(L, codes->n / n_dims * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);

  if (in_indices && in_thresholds) {
    tk_itq_otsu(L, out->a, codes->a, codes->n / n_dims, n_dims, metric, n_bins, minimize,
                in_indices, in_thresholds, NULL, NULL, NULL);
    return 1;
  }

  tk_ivec_t *out_indices = tk_ivec_create(L, 0, 0, 0);
  tk_dvec_t *out_scores = tk_dvec_create(L, 0, 0, 0);
  tk_dvec_t *out_thresholds = NULL;

  tk_itq_otsu(L, out->a, codes->a, codes->n / n_dims, n_dims, metric, n_bins, minimize,
              NULL, NULL, out_indices, out_scores, return_thresholds ? &out_thresholds : NULL);

  lua_insert(L, -2);
  lua_insert(L, -2);

  if (return_thresholds && out_thresholds) {
    return 4;
  }
  return 3;
}

static luaL_Reg tk_itq_fns[] =
{
  { "encode", tk_itq_encode_lua },
  { "itq", tk_itq_encode_lua },
  { "sign", tk_itq_sign_lua },
  { "median", tk_itq_median_lua },
  { "otsu", tk_itq_otsu_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_itq (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tk_itq_fns, 0); // t
  return 1;
}
