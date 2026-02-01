#include <santoku/tsetlin/itq.h>

#include <float.h>
#include <lauxlib.h>
#include <lua.h>
#include <lapacke.h>

static inline int tk_itq_center_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "center", "n_dims");
  tk_dvec_t *centered = NULL;
  tk_dvec_t *means = NULL;
  tk_itq_center(L, codes, n_dims, &centered, &means);
  return 2;
}

static inline int tk_itq_rotate_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "rotate", "n_dims");

  tk_dvec_t *rotation = NULL;
  if (tk_lua_ftype(L, 1, "rotation") != LUA_TNIL) {
    lua_getfield(L, 1, "rotation");
    rotation = tk_dvec_peek(L, -1, "rotation");
  }

  tk_dvec_t *rotated = NULL;
  tk_itq_rotate(L, codes, rotation, n_dims, &rotated);
  return 1;
}

static inline int tk_itq_sign_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "sign", "n_dims");
  tk_cvec_t *out = NULL;
  tk_itq_sign(L, codes, n_dims, &out);
  return 1;
}

static inline int tk_itq_median_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "median", "n_dims");
  tk_cvec_t *out = NULL;
  tk_dvec_t *medians = NULL;
  tk_itq_median(L, codes, n_dims, &out, &medians);
  return 2;
}

static inline int tk_itq_itq_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  uint64_t max_iterations = tk_lua_foptunsigned(L, 1, "itq", "iterations", 1000);
  double tolerance = tk_lua_foptposdouble(L, 1, "itq", "tolerance", 1e-8);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_dvec_t *rotation = NULL;
  tk_itq_itq(L, codes, n_dims, max_iterations, tolerance, i_each, &rotation);
  return 1;
}

static luaL_Reg tk_itq_fns[] =
{
  { "center", tk_itq_center_lua },
  { "rotate", tk_itq_rotate_lua },
  { "sign", tk_itq_sign_lua },
  { "median", tk_itq_median_lua },
  { "itq", tk_itq_itq_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_itq (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_itq_fns, 0);
  return 1;
}
