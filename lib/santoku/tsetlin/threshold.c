#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/pairs.h>
#include <santoku/tsetlin/threshold.h>

#include <float.h>
#include <lauxlib.h>
#include <lua.h>
#include <primme.h>

#include "khash.h"
KHASH_SET_INIT_INT64(i64)
typedef khash_t(i64) i64_hash_t;

static inline int tm_threshold_tch (lua_State *L)
{
  lua_settop(L, 1);

  uint64_t n_sentences = tk_lua_fcheckunsigned(L, 1, "threshold_tch", "n_sentences");
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "threshold_tch", "n_hidden");

  int64_t *codes_bits = NULL;
  uint64_t n_codes_bits = 0;
  lua_getfield(L, 1, "codes");
  if (lua_type(L, -1) > 0) {
    lua_pushboolean(L, true);
    tk_lua_callmod(L, 2, 4, "santoku.matrix.integer", "view");
    codes_bits = (int64_t *) tk_lua_checkustring(L, -4, NULL);
    n_codes_bits = (uint64_t) luaL_checkinteger(L, -1);
  }

  double *z = NULL;
  lua_getfield(L, 1, "z");
  if (lua_type(L, -1) > 0) {
    tk_lua_callmod(L, 1, 1, "santoku.matrix.number", "view");
    z = (double *) tk_lua_checkuserdata(L, -1, NULL);
  }

  lua_getfield(L, 1, "pos");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pos = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pos = (uint64_t) luaL_checkinteger(L, -1) / 2;

  lua_getfield(L, 1, "neg");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *neg = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_neg = (uint64_t) luaL_checkinteger(L, -1) / 2;

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  // Setup pairs & adjacency lists
  tm_pairs_t *pairs = kh_init(pairs);
  roaring64_bitmap_t **adj_pos = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  roaring64_bitmap_t **adj_neg = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  tm_pairs_init(L, pairs, pos, neg, &n_pos, &n_neg);
  tm_adj_init(pairs, adj_pos, adj_neg, n_sentences);

  unsigned int global_iter = 0;
  tm_run_tch_thresholding(L, z, &codes_bits, &n_codes_bits, adj_pos, adj_neg, n_sentences, n_hidden, i_each, &global_iter);

  // Cleanup
  kh_destroy(pairs, pairs);
  for (uint64_t u = 0; u < n_sentences; u ++) {
    roaring64_bitmap_free(adj_pos[u]);
    roaring64_bitmap_free(adj_neg[u]);
  }
  free(adj_pos);
  free(adj_neg);

  // Update codes
  lua_pushlightuserdata(L, codes_bits);
  lua_pushinteger(L, 1);
  lua_pushinteger(L, (int64_t) n_codes_bits);
  lua_getfield(L, 1, "codes");
  tk_lua_callmod(L, 4, 1, "santoku.matrix.integer", "from_view");
  return 1;
}

static inline int tm_threshold_median (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "z");
  tk_lua_callmod(L, 1, 1, "santoku.matrix.number", "view");
  double *z = (double *) tk_lua_checkuserdata(L, -1, NULL);
  uint64_t n_sentences = tk_lua_fcheckunsigned(L, 1, "threshold_median", "n_sentences");
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "threshold_median", "n_hidden");
  tm_run_median_thresholding(L, z, n_sentences, n_hidden);
  return 1;
}


static luaL_Reg tm_threshold_fns[] =
{
  { "tch", tm_threshold_tch },
  { "median", tm_threshold_median },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_threshold (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_threshold_fns, 0); // t
  return 1;
}
