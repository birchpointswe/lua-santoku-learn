#define _GNU_SOURCE

#include <santoku/tsetlin/graph.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/threshold.h>
#include <santoku/threads.h>
#include <santoku/dvec.h>
#include <float.h>
#include <primme.h>

typedef enum {
  TK_SPECTRAL_INIT,
  TK_SPECTRAL_MATVEC,
  TK_SPECTRAL_FINALIZE
} tk_spectral_stage_t;

typedef struct {
  double *x, *y, *z;
  double *evals, *evecs, *resNorms;
  double *laplacian;
  roaring64_bitmap_t **adj_pos;
  roaring64_bitmap_t **adj_neg;
  uint64_t n_sentences;
  uint64_t n_hidden;
  tk_threadpool_t *pool;
} tk_spectral_t;

typedef struct {
  tk_spectral_t *spec;
  uint64_t ufirst, ulast;
} tk_spectral_thread_t;

static inline void tk_spectral_worker (void *dp, int sig)
{
  tk_spectral_stage_t stage = (tk_spectral_stage_t) sig;
  tk_spectral_thread_t *data = (tk_spectral_thread_t *) dp;
  double *x = data->spec->x;
  double *y = data->spec->y;
  double *z = data->spec->z;
  double *evecs = data->spec->evecs;
  double *laplacian = data->spec->laplacian;
  uint64_t n_sentences = data->spec->n_sentences;
  uint64_t n_hidden = data->spec->n_hidden;
  uint64_t ufirst = data->ufirst;
  uint64_t ulast = data->ulast;
  roaring64_iterator_t it;
  roaring64_bitmap_t **adj_pos = data->spec->adj_pos;
  roaring64_bitmap_t **adj_neg = data->spec->adj_neg;

  switch (stage) {

    case TK_SPECTRAL_INIT:
      for (uint64_t i = ufirst; i <= ulast; i ++) {
        laplacian[i] =
          (double) roaring64_bitmap_get_cardinality(adj_pos[i]) +
          (double) roaring64_bitmap_get_cardinality(adj_neg[i]);
        laplacian[i] =
          laplacian[i] > 0 ? 1.0 / sqrt((double) laplacian[i]) : 0.0;
      }
      break;

    case TK_SPECTRAL_MATVEC:
      for (uint64_t u = ufirst; u <= ulast; u ++) {
        double scale_u = laplacian[u];
        double sum = scale_u * x[u];
        // Positive neighbors
        roaring64_iterator_reinit(adj_pos[u], &it);
        while (roaring64_iterator_has_value(&it)) {
          uint64_t v = roaring64_iterator_value(&it);
          roaring64_iterator_advance(&it);
          double w = scale_u * laplacian[v];
          sum -= w * x[v];
        }
        // Negative neighbors
        roaring64_iterator_reinit(adj_neg[u], &it);
        while (roaring64_iterator_has_value(&it)) {
          uint64_t v = roaring64_iterator_value(&it);
          roaring64_iterator_advance(&it);
          double w = scale_u * laplacian[v];
          sum += w * x[v];
        }
        y[u] = sum;
      }
      break;

    case TK_SPECTRAL_FINALIZE:
      for (uint64_t i = ufirst; i <= ulast; i ++)
        for (uint64_t f = 0; f < n_hidden; f ++)
          z[i * n_hidden + f] = evecs[i + (f + 1) * n_sentences];
      break;

  }
}

static inline void tk_spectral_matvec (
  void *vx,
  PRIMME_INT *ldx,
  void *vy,
  PRIMME_INT *ldy,
  int *blockSize,
  struct primme_params *primme,
  int *ierr
) {
  // Get context from primme struct
  tk_spectral_t *spec = (tk_spectral_t *) primme->matrix;
  spec->x = (double *) vx;
  spec->y = (double *) vy;
  tk_threads_signal(spec->pool, TK_SPECTRAL_MATVEC);
  *ierr = 0;
}

// Consider primme params.method for perf/accuracy tradeoffs
static inline void tm_run_spectral (
  lua_State *L,
  tk_threadpool_t *pool,
  tk_dvec_t *z,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  uint64_t n_sentences,
  uint64_t n_hidden,
  int i_each
) {

  // Init
  tk_spectral_t spec;
  tk_spectral_thread_t threads[pool->n_threads];
  spec.z = z->a;
  spec.laplacian = tk_malloc(L, n_sentences * sizeof(double));
  spec.adj_pos = adj_pos;
  spec.adj_neg = adj_neg;
  spec.n_sentences = n_sentences;
  spec.n_hidden = n_hidden;
  spec.pool = pool;
  for (unsigned int i = 0; i < pool->n_threads; i ++) {
    tk_spectral_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->spec = &spec;
    tk_thread_range(i, pool->n_threads, n_sentences, &data->ufirst, &data->ulast);
  }

  // Init laplacian
  tk_threads_signal(pool, TK_SPECTRAL_INIT);

  // Run PRIMME to compute the smallest n_hidden + 1 eigenvectors of the graph
  // laplacian
  primme_params params;
  primme_initialize(&params);
  params.n = (int64_t) n_sentences;
  params.numEvals = n_hidden + 1;
  params.matrixMatvec = tk_spectral_matvec;
  params.matrix = &spec;
  params.maxBlockSize = 1;
  params.printLevel = 0;
  params.eps = 1e-3;
  params.target = primme_smallest;
  spec.evals = tk_malloc(L, (size_t) params.numEvals * sizeof(double));
  spec.evecs = tk_malloc(L, (size_t) params.n * (size_t) params.numEvals * sizeof(double));
  spec.resNorms = tk_malloc(L, (size_t) params.numEvals * sizeof(double));
  int ret = dprimme(spec.evals, spec.evecs, spec.resNorms, &params);
  if (ret != 0)
    tk_lua_verror(L, 2, "spectral", "failure calling PRIMME");

  // Copy eigenvectors into the output matrix, skipping the first
  tk_threads_signal(pool, TK_SPECTRAL_FINALIZE);

  // Cleanup
  free(spec.laplacian);
  free(spec.evals);
  free(spec.evecs);
  free(spec.resNorms);
  primme_free(&params);

  // Log
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, params.stats.numMatvecs);
    lua_call(L, 1, 0);
  }
}

static inline int tm_encode (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "graph");
  tk_graph_t *graph = tk_graph_peek(L, -1);
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "spectral", "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 1, "spectral", "threads");

  if (BITS_MOD(n_hidden) != 0)
    tk_lua_verror(L, 3, "spectral", "n_hidden", "must be a multiple of " STR(BITS));

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_spectral_worker);
  roaring64_bitmap_t **adj_pos = graph->adj_pos;
  roaring64_bitmap_t **adj_neg = graph->adj_neg;
  uint64_t n_nodes = graph->n_nodes;

  // Spectral hashing
  tk_dvec_t *z = tk_dvec_create(L, graph->n_nodes * n_hidden, 0, 0);
  tm_run_spectral(L, pool, z, adj_pos, adj_neg, n_nodes, n_hidden, i_each);

  // Cleanup
  tk_threads_destroy(pool);
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
