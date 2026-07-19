#include <santoku/lua/utils.h>
#include <santoku/learn/ann.h>

static inline int tk_ann_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "codes");
  tk_mtx_t *M = tk_mtx_peek(L, -1, "codes");
  int codes_idx = lua_gettop(L);
  lua_getfield(L, 1, "rerank");
  bool keep_codes = lua_isnil(L, -1) ? true : lua_toboolean(L, -1);
  lua_pop(L, 1);

  uint64_t N = M->n_rows;
  uint64_t n_dims = M->n_cols;
  uint64_t features = n_dims;
  const float *codes_a = ((tk_fvec_t *) M->v)->a;
  size_t bpv = TK_CVEC_BITS_BYTES(features);
  uint64_t m = (features + TK_ANN_SUBSTR_BITS - 1) / TK_ANN_SUBSTR_BITS;
  if (m == 0) m = 1;
  uint64_t stride = (uint64_t)TK_ANN_BUCKETS + 1;

  tk_cvec_t *bits = tk_cvec_create(L, N * bpv);
  bits->n = N * bpv;
  int bits_idx = lua_gettop(L);
  tk_ann_sign_pack(codes_a, N, n_dims, features, bits->a, bpv);

  tk_ann_flat_t *flat = tk_lua_newuserdata(L, tk_ann_flat_t, TK_ANN_MT, tk_ann_flat_mt_fns, tk_ann_flat_gc);
  int flat_idx = lua_gettop(L);
  flat->sorted_sids = NULL;
  flat->bucket_off = NULL;
  flat->data = NULL;
  flat->codes = NULL;
  flat->n_dims = 0;

  tk_ivec_t *sids = tk_ivec_create(L, m * N);
  sids->n = m * N;
  int sids_idx = lua_gettop(L);
  tk_ivec_t *buckets = tk_ivec_create(L, m * stride);
  buckets->n = m * stride;
  int buckets_idx = lua_gettop(L);

  flat->sorted_sids = sids->a;
  flat->bucket_off = buckets->a;
  tk_ann_flat_build(flat, bits->a, N, features);

  lua_newtable(L);
  int eph = lua_gettop(L);
  lua_pushvalue(L, bits_idx);
  lua_setfield(L, eph, "data");
  lua_pushvalue(L, sids_idx);
  lua_setfield(L, eph, "sids");
  lua_pushvalue(L, buckets_idx);
  lua_setfield(L, eph, "buckets");
  if (keep_codes) {
    flat->codes = codes_a;
    flat->n_dims = n_dims;
    lua_pushvalue(L, codes_idx);
    lua_setfield(L, eph, "codes");
  }
  lua_pushvalue(L, eph);
  lua_setfenv(L, flat_idx);
  lua_settop(L, flat_idx);
  return 1;
}

static luaL_Reg tk_ann_fns[] =
{
  { "create", tk_ann_create_lua },
  { "load", tk_ann_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_ann (lua_State *L)
{
  tk_lua_require_mod(L, "santoku.csr");
  tk_lua_require_mod(L, "santoku.mtx");
  tk_lua_require_mod(L, "santoku.ivec");
  tk_lua_require_mod(L, "santoku.cvec");
  lua_newtable(L);
  tk_lua_register(L, tk_ann_fns, 0);
  return 1;
}
