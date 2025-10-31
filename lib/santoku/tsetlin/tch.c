#include <santoku/tsetlin/tch.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/threads.h>

#include <float.h>
#include <lauxlib.h>
#include <lua.h>

static inline int tk_tch_refine_lua (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_cvec_t *codes = tk_cvec_peek(L, -1, "codes");
  int i_out = tk_lua_absindex(L, -1);

  lua_getfield(L, 1, "ids");
  tk_ivec_t *uids = tk_ivec_peek(L, -1, "ids");

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *adj_offset = tk_ivec_peek(L, -1, "offsets");

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *adj_data = tk_ivec_peek(L, -1, "neighbors");

  lua_getfield(L, 1, "weights");
  tk_dvec_t *adj_weights = tk_dvec_peek(L, -1, "weights");

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "tch", "n_dims");

  double eps = tk_lua_foptnumber(L, 1, "tch", "eps", -1.0);

  lua_getfield(L, 1, "rank_weights");
  tk_dvec_t *rank_weights = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  uint64_t knn = tk_lua_foptunsigned(L, 1, "tch", "knn", 0);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_tch_refine(L, codes, uids, adj_offset, adj_data, adj_weights, n_dims, eps, rank_weights, knn, i_each);
  lua_pushvalue(L, i_out);
  return 1;
}

static luaL_Reg tk_tch_fns[] =
{
  { "refine", tk_tch_refine_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_tch (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_tch_fns, 0);
  return 1;
}
