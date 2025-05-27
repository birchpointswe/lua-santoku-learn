#define _GNU_SOURCE

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/threshold.h>
#include <santoku/tsetlin/graph.h>

#include <float.h>
#include <lauxlib.h>
#include <lua.h>
#include <primme.h>

static inline int tm_threshold_tch (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "graph");
  tk_graph_t *graph = tk_graph_peek(L, -1);

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

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  // Setup pairs & adjacency lists
  roaring64_bitmap_t **adj_pos = graph->adj_pos;
  roaring64_bitmap_t **adj_neg = graph->adj_neg;

  // Run tch
  tm_run_tch_thresholding(L, z, &codes_bits, &n_codes_bits, adj_pos, adj_neg, graph->n_nodes, n_hidden, i_each);

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
