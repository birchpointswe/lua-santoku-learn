#define _GNU_SOURCE

#include <santoku/tsetlin/conf.h>
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
  double tolerance = tk_lua_foptposdouble(L, 1, "itq", "tolerance", 1e-4);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  // Run itq
  tk_itq_encode(L, codes, n_dims, max_iterations, tolerance, i_each);
  return 1;
}

static inline int tk_itq_sign_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  tk_ivec_t *out = tk_ivec_create(L, 0, 0, 0);
  tk_itq_sign(out, codes->a, codes->n / n_dims, n_dims);
  return 1;
}

static inline int tk_itq_median_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  tk_ivec_t *out = tk_ivec_create(L, 0, 0, 0);
  tk_itq_median(out, codes->a, codes->n / n_dims, n_dims);
  return 1;
}

static luaL_Reg tk_itq_fns[] =
{
  { "encode", tk_itq_encode_lua },
  { "sign", tk_itq_sign_lua },
  { "median", tk_itq_median_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_itq (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tk_itq_fns, 0); // t
  return 1;
}
