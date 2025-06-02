#define _GNU_SOURCE

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/threshold.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/ivec.h>

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

  bool use_codes;
  int i_out;

  tk_ivec_t *codes_bits = NULL;
  lua_getfield(L, 1, "codes");

  if (lua_type(L, -1) > 0) {
    codes_bits = tk_ivec_peek(L, -1);
    i_out = tk_lua_absindex(L, -1);
    use_codes = true;
  } else {
    codes_bits = tk_ivec_create(L, 0, 0, 0);
    i_out = tk_lua_absindex(L, -1);
    use_codes = false;
  }

  tk_dvec_t *z = NULL;
  lua_getfield(L, 1, "z");
  if (lua_type(L, -1) > 0)
    z = tk_dvec_peek(L, -1);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  // Setup pairs & adjacency lists
  roaring64_bitmap_t **adj_pos = graph->adj_pos;
  roaring64_bitmap_t **adj_neg = graph->adj_neg;

  // Run tch
  tm_run_tch_thresholding(L, z, codes_bits, use_codes, adj_pos, adj_neg, graph->n_nodes, n_hidden, i_each);
  lua_pushvalue(L, i_out);
  return 1;
}

static inline int tm_threshold_median (lua_State *L)
{
  lua_settop(L, 1);
  tk_dvec_t *z = tk_dvec_peek(L, 1);
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
