#include <santoku/lua/utils.h>
#include <santoku/learn/ann.h>

static inline int tk_ann_create_lua (lua_State *L)
{
  uint64_t features = tk_lua_fcheckunsigned(L, 1, "create", "features");
  lua_getfield(L, 1, "data");
  tk_cvec_t *data = tk_cvec_peek(L, -1, "data");
  int data_idx = lua_gettop(L);
  uint64_t N = data->n / TK_CVEC_BITS_BYTES(features);
  tk_ann_flat_create(L, data->a, N, features);
  int flat_idx = lua_gettop(L);
  lua_newtable(L);
  int eph = lua_gettop(L);
  lua_pushvalue(L, eph);
  lua_setfenv(L, flat_idx);
  lua_pushvalue(L, data_idx);
  lua_setfield(L, eph, "data");
  lua_getfield(L, 1, "codes");
  tk_fvec_t *codes = tk_fvec_peekopt(L, -1);
  if (codes) {
    tk_ann_flat_t *flat = tk_ann_flat_peek(L, flat_idx);
    uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "create", "n_dims");
    flat->codes = codes->a;
    flat->n_dims = n_dims;
    lua_setfield(L, eph, "codes");
  } else {
    lua_pop(L, 1);
  }
  lua_settop(L, flat_idx);
  return 1;
}

static luaL_Reg tk_ann_fns[] =
{
  { "create", tk_ann_create_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_ann (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_ann_fns, 0);
  return 1;
}
