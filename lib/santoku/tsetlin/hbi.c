#define _GNU_SOURCE

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/threads.h>

static inline int tk_hbi_create_lua (lua_State *L)
{
  uint64_t features = tk_lua_fcheckunsigned(L, 1, "create", "features");
  uint64_t n_threads = tk_threads_getn(L, 1, "create", "threads");
  tk_hbi_create(L, features, n_threads);
  return 1;
}

static inline int tk_hbi_load_lua (lua_State *L)
{
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  tk_hbi_load(L, fh);
  tk_lua_fclose(L, fh);
  return 1;
}

static luaL_Reg tk_hbi_fns[] =
{
  { "create", tk_hbi_create_lua },
  { "load", tk_hbi_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_hbi (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_hbi_fns, 0);
  return 1;
}
