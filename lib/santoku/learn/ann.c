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
  tk_lua_add_ephemeron(L, TK_ANN_EPH, flat_idx, data_idx);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peekopt(L, -1);
  if (codes) {
    tk_ann_flat_t *flat = tk_ann_flat_peek(L, flat_idx);
    uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "create", "n_dims");
    flat->codes = codes->a;
    flat->n_dims = n_dims;
    tk_lua_add_ephemeron(L, TK_ANN_EPH, flat_idx, lua_gettop(L));
  }
  lua_pop(L, 1);
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
