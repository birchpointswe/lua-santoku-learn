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

typedef struct {
  roaring64_bitmap_t **adj_pos;
  roaring64_bitmap_t **adj_neg;
  double *inv_sqrt_deg;
  uint64_t n_sentences;
} laplacian_ctx_t;

static inline void laplacian_matvec (
  void *vx,
  PRIMME_INT *ldx,
  void *vy,
  PRIMME_INT *ldy,
  int *blockSize,
  struct primme_params *primme,
  int *ierr
) {
  // Get context from primme struct
  laplacian_ctx_t *data = (laplacian_ctx_t*)primme->matrix;
  const double *x = (const double*)vx;
  double *y = (double*)vy;
  uint64_t n = data->n_sentences;
  double *inv_sqrt_deg = data->inv_sqrt_deg;
  roaring64_bitmap_t **adj_pos = data->adj_pos;
  roaring64_bitmap_t **adj_neg = data->adj_neg;
  // Only support blockSize=1 for clarity
  #pragma omp parallel for schedule(dynamic)
  for (uint64_t u = 0; u < n; u ++) {
    double scale_u = inv_sqrt_deg[u];
    double sum = scale_u * x[u];
    roaring64_iterator_t it;
    // Positive neighbors
    roaring64_iterator_reinit(adj_pos[u], &it);
    while (roaring64_iterator_has_value(&it)) {
      uint64_t v = roaring64_iterator_value(&it);
      roaring64_iterator_advance(&it);
      double w = scale_u * inv_sqrt_deg[v];
      sum -= w * x[v];
    }
    // Negative neighbors
    roaring64_iterator_reinit(adj_neg[u], &it);
    while (roaring64_iterator_has_value(&it)) {
      uint64_t v = roaring64_iterator_value(&it);
      roaring64_iterator_advance(&it);
      double w = scale_u * inv_sqrt_deg[v];
      sum += w * x[v];
    }
    y[u] = sum;
  }
  *ierr = 0;
}

static inline void tm_run_spectral (
  lua_State *L,
  double *z,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  uint64_t n_sentences,
  uint64_t n_hidden,
  int i_each,
  unsigned int *global_iter
) {
  unsigned int *degree = tk_malloc(L, n_sentences * sizeof(unsigned int));
  for (uint64_t i = 0; i < n_sentences; i++)
    degree[i] = (unsigned int)roaring64_bitmap_get_cardinality(adj_pos[i]) +
      (unsigned int)roaring64_bitmap_get_cardinality(adj_neg[i]);
  double *inv_sqrt_deg = tk_malloc(L, n_sentences * sizeof(double));
  for (uint64_t i = 0; i < n_sentences; i++)
    inv_sqrt_deg[i] = degree[i] > 0 ? 1.0 / sqrt((double) degree[i]) : 0.0;
  free(degree);
  laplacian_ctx_t ctx = { adj_pos, adj_neg, inv_sqrt_deg, n_sentences };
  primme_params params;
  primme_initialize(&params);
  params.n = (int64_t) n_sentences;
  params.numEvals = n_hidden + 1;
  params.matrixMatvec = laplacian_matvec;
  params.matrix = &ctx;
  params.maxBlockSize = 1;
  params.printLevel = 0;
  params.eps = 1e-3;
  params.target = primme_smallest;
  double *evals = tk_malloc(L, (size_t) params.numEvals * sizeof(double));
  double *evecs = tk_malloc(L, (size_t) params.n * (size_t) params.numEvals * sizeof(double));
  double *resNorms = tk_malloc(L, (size_t) params.numEvals * sizeof(double));
  int ret = dprimme(evals, evecs, resNorms, &params);
  if (ret != 0)
    tk_lua_verror(L, 2, "spectral", "failure calling PRIMME (code %d)", ret);
  for (uint64_t i = 0; i < n_sentences; i++)
    for (uint64_t f = 0; f < n_hidden; f++)
      z[i * n_hidden + f] = evecs[i + (f+1) * n_sentences];
  free(inv_sqrt_deg);
  free(evals);
  free(evecs);
  free(resNorms);
  primme_free(&params);
  (*global_iter)++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushinteger(L, params.stats.numMatvecs);
    lua_call(L, 2, 0);
  }
}

static inline int tm_encode (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "pos");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pos = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pos = (uint64_t) luaL_checkinteger(L, -1) / 2;

  lua_getfield(L, 1, "neg");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *neg = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_neg = (uint64_t) luaL_checkinteger(L, -1) / 2;

  uint64_t n_sentences = tk_lua_fcheckunsigned(L, 1, "spectral", "n_sentences");
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "spectral", "n_hidden");

  if (BITS_MOD(n_hidden) != 0)
    tk_lua_verror(L, 3, "spectral", "n_hidden", "must be a multiple of " STR(BITS));

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

  // Spectral hashing
  double *z = tk_malloc(L, (size_t) n_sentences * (size_t) n_hidden * sizeof(double));
  tm_run_spectral(L, z, adj_pos, adj_neg, n_sentences, n_hidden, i_each, &global_iter);

  // Push floats
  lua_pushlightuserdata(L, z);
  lua_pushinteger(L, (int64_t) n_sentences);
  lua_pushinteger(L, (int64_t) n_hidden);
  tk_lua_callmod(L, 3, 1, "santoku.matrix.number", "from_view");

  // Cleanup
  kh_destroy(pairs, pairs);
  for (uint64_t u = 0; u < n_sentences; u ++) {
    roaring64_bitmap_free(adj_pos[u]);
    roaring64_bitmap_free(adj_neg[u]);
  }
  free(adj_pos);
  free(adj_neg);
  return 1;
}

static luaL_Reg tm_codebook_fns[] =
{
  { "encode", tm_encode },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_spectral (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_codebook_fns, 0); // t
  return 1;
}
