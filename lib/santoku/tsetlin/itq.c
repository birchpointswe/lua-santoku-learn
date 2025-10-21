#include <santoku/tsetlin/itq.h>

#include <float.h>
#include <lauxlib.h>
#include <lua.h>

static inline int tk_itq_encode_lua (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  uint64_t max_iterations = tk_lua_foptunsigned(L, 1, "itq", "iterations", 1000);
  double tolerance = tk_lua_foptposdouble(L, 1, "itq", "tolerance", 1e-8);
  unsigned int n_threads = tk_threads_getn(L, 1, "itq", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_itq_encode(L, codes, n_dims, max_iterations, tolerance, i_each, n_threads);
  return 1;
}

static inline int tk_sr_encode_lua (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  uint64_t max_iterations = tk_lua_foptunsigned(L, 1, "itq", "iterations", 1000);
  double tolerance = tk_lua_foptposdouble(L, 1, "itq", "tolerance", 1e-8);
  unsigned int n_threads = tk_threads_getn(L, 1, "itq", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_sr_encode(L, codes, n_dims, max_iterations, tolerance, i_each, n_threads);
  return 1;
}

static inline int tk_itq_sign_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  tk_cvec_t *out = tk_cvec_create(L, codes->n / n_dims * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);
  tk_itq_sign(out->a, codes->a, codes->n / n_dims, n_dims);
  return 1;
}

static inline int tk_itq_dbq_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  uint64_t n_samples = codes->n / n_dims;
  uint64_t n_dims_out = n_dims * 2;
  tk_cvec_t *out = tk_cvec_create(L, n_samples * TK_CVEC_BITS_BYTES(n_dims_out), 0, 0);
  tk_cvec_zero(out);
  tk_itq_dbq_full(out->a, codes->a, n_samples, n_dims_out);
  lua_pushinteger(L, (int64_t) n_dims_out);
  return 2;
}

static inline int tk_itq_median_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  tk_cvec_t *out = tk_cvec_create(L, codes->n / n_dims * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);
  tk_itq_median(L, out->a, codes->a, codes->n / n_dims, n_dims);
  return 1;
}

static inline int tk_sr_ranking_encode_lua (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  lua_pop(L, 1);

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");

  // Get adjacency arrays
  lua_getfield(L, 1, "ids");
  tk_ivec_t *adj_ids = tk_ivec_peek(L, -1, "ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *adj_offsets = tk_ivec_peek(L, -1, "offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *adj_neighbors = tk_ivec_peek(L, -1, "neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "weights");
  tk_dvec_t *adj_weights = tk_dvec_peek(L, -1, "weights");
  lua_pop(L, 1);

  uint64_t max_iterations = tk_lua_foptunsigned(L, 1, "itq", "iterations", 1000);
  double tolerance = tk_lua_foptposdouble(L, 1, "itq", "tolerance", 1e-8);
  unsigned int n_threads = tk_threads_getn(L, 1, "itq", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_sr_ranking_encode(L, codes, n_dims, adj_ids, adj_offsets, adj_neighbors, adj_weights,
                       max_iterations, tolerance, i_each, n_threads);
  return 1;
}

static luaL_Reg tk_itq_fns[] =
{
  { "encode", tk_itq_encode_lua },
  { "sr", tk_sr_encode_lua },
  { "sr_ranking", tk_sr_ranking_encode_lua },
  { "sign", tk_itq_sign_lua },
  { "median", tk_itq_median_lua },
  { "dbq", tk_itq_dbq_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_itq (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tk_itq_fns, 0); // t
  return 1;
}
