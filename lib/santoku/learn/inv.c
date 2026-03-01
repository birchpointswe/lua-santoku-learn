#include <santoku/iuset.h>
#include <santoku/lua/utils.h>
#include <santoku/learn/inv.h>

static inline int tk_inv_create_lua (lua_State *L)
{
  tk_dvec_t *weights = NULL;
  tk_ivec_t *ranks = NULL;
  uint64_t features = 0;
  uint64_t n_ranks = 0;
  lua_getfield(L, 1, "features");
  if (lua_type(L, -1) == LUA_TNUMBER) {
    features = tk_lua_checkunsigned(L, -1, "features");
  } else {
    weights = tk_dvec_peek(L, -1, "features");
    features = weights->n;
  }
  lua_getfield(L, 1, "ranks");
  if (!lua_isnil(L, -1)) {
    ranks = tk_ivec_peek(L, -1, "ranks");
  }
  if (ranks != NULL)
    n_ranks = tk_lua_fcheckunsigned(L, 1, "create", "n_ranks");
  tk_inv_kernel_t kernel = TK_INV_COSINE;
  lua_getfield(L, 1, "kernel");
  if (lua_isstring(L, -1)) {
    const char *k = lua_tostring(L, -1);
    if (strcmp(k, "jaccard") == 0) kernel = TK_INV_JACCARD;
    else if (strcmp(k, "cosine") != 0)
      return luaL_error(L, "invalid kernel: %s", k);
  }
  lua_pop(L, 1);
  tk_inv_create(L, features, weights, n_ranks, ranks, kernel);
  return 1;
}

static inline int tk_inv_load_lua (lua_State *L)
{
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  tk_inv_load(L, fh);
  tk_lua_fclose(L, fh);
  return 1;
}

static luaL_Reg tk_inv_fns[] =
{
  { "create", tk_inv_create_lua },
  { "load", tk_inv_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_inv (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_inv_fns, 0);
  tk_inv_hoods_create(L, 0, 0, 0);
  luaL_getmetafield(L, -1, "__index");
  luaL_register(L, NULL, tk_inv_hoods_lua_mt_fns);
  lua_pop(L, 2);
  return 1;
}
