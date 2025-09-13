#include <santoku/lua/utils.h>
#include <santoku/tsetlin/inv.h>

static inline int tk_inv_create_lua (lua_State *L)
{
  int i_weights = -1;
  tk_dvec_t *weights = NULL;
  int i_ranks = -1;
  tk_ivec_t *ranks = NULL;
  uint64_t features = 0;
  uint64_t n_ranks = 0;
  lua_getfield(L, 1, "features");
  if (lua_type(L, -1) == LUA_TNUMBER) {
    features = tk_lua_checkunsigned(L, -1, "features");
  } else {
    weights = tk_dvec_peek(L, -1, "features");
    i_weights = tk_lua_absindex(L, -1);
    features = weights->n;
  }
  lua_getfield(L, 1, "ranks");
  if (!lua_isnil(L, -1)) {
    ranks = tk_ivec_peek(L, -1, "ranks");
    i_ranks = tk_lua_absindex(L, -1);
  }
  if (ranks != NULL) {
    n_ranks = tk_lua_fcheckunsigned(L, 1, "create", "n_ranks");
  }
  double decay = tk_lua_foptnumber(L, 1, "create", "decay", 1.0);
  uint64_t n_threads = tk_threads_getn(L, 1, "create", "threads");
  tk_inv_create(L, features, weights, n_ranks, ranks, decay, n_threads, i_weights, i_ranks);
  return 1;
}

static inline int tk_inv_load_lua (lua_State *L)
{
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  uint64_t n_threads = tk_threads_getn(L, 3, "create", NULL);
  tk_inv_load(L, fh, n_threads);
  tk_lua_fclose(L, fh);
  return 1;
}

static luaL_Reg tk_inv_fns[] =
{
  { "create", tk_inv_create_lua },
  { "load", tk_inv_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_inv (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_inv_fns, 0);
  tk_inv_hoods_create(L, 0, 0, 0);
  luaL_getmetafield(L, -1, "__index");
  luaL_register(
    L, NULL, tk_inv_hoods_lua_mt_fns); // t
  lua_pop(L, 2);
  return 1;
}
