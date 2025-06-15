#define _GNU_SOURCE

#include <santoku/tsetlin/graph.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/dvec.h>
#include <float.h>
#include <primme.h>

typedef enum {
  TK_SPECTRAL_INIT,
  TK_SPECTRAL_MATVEC,
  TK_SPECTRAL_FINALIZE,
  TK_SPECTRAL_CENTER
} tk_spectral_stage_t;

typedef struct {
  double *x, *y, *z;
  double *evals, *evecs, *resNorms;
  double *laplacian;
  tk_graph_adj_t *adj_pos;
  tk_graph_adj_t *adj_neg;
  uint64_t n_nodes;
  uint64_t n_hidden;
  tk_threadpool_t *pool;
} tk_spectral_t;

typedef struct {
  tk_spectral_t *spec;
  uint64_t ifirst, ilast;
  uint64_t hfirst, hlast;
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
  uint64_t n_nodes = data->spec->n_nodes;
  uint64_t n_hidden = data->spec->n_hidden;
  uint64_t ifirst = data->ifirst;
  uint64_t ilast = data->ilast;
  uint64_t hfirst = data->hfirst;
  uint64_t hlast = data->hlast;
  tk_graph_adj_t *adj_pos = data->spec->adj_pos;
  tk_graph_adj_t *adj_neg = data->spec->adj_neg;
  int64_t iv;

  switch (stage) {

    case TK_SPECTRAL_INIT:
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        // NOTE: Apparently including negatives in this degree vector is common
        // in practice but not strictly mathematically justified academically.
        laplacian[i] =
          (double) tk_iuset_size(adj_pos->a[i]) +
          (double) tk_iuset_size(adj_neg->a[i]);
        laplacian[i] =
          laplacian[i] > 0 ? 1.0 / sqrt((double) laplacian[i]) : 0.0;
        if (!isfinite(laplacian[i]))
          laplacian[i] = 0.0;
      }
      break;

    case TK_SPECTRAL_MATVEC:
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        double scale_i = laplacian[i];
        double sum = x[i];
        // Positive neighbors
        tk_iuset_foreach(adj_pos->a[i], iv, ({
          double w = scale_i * laplacian[iv];
          sum -= w * x[iv];
        }))
        // Negative neighbors
        tk_iuset_foreach(adj_neg->a[i], iv, ({
          double w = scale_i * laplacian[iv];
          sum += w * x[iv];
        }))
        y[i] = sum;
      }
      break;

    case TK_SPECTRAL_FINALIZE:
      for (uint64_t i = ifirst; i <= ilast; i ++)
        for (uint64_t f = 0; f < n_hidden; f ++)
          z[i * n_hidden + f] = evecs[i + (f + 1) * n_nodes];
      break;

    case TK_SPECTRAL_CENTER:
      for (uint64_t f = hfirst; f <= hlast; f ++) {
        double mu = 0.0;
        for (uint64_t i = 0; i < n_nodes; i ++)
          mu += z[i * n_hidden + f];
        mu /= n_nodes;
        for (uint64_t i = 0; i < n_nodes; i ++)
          z[i * n_hidden + f] -= mu;
      }
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
  tk_graph_adj_t *adj_pos,
  tk_graph_adj_t *adj_neg,
  uint64_t n_nodes,
  uint64_t n_hidden,
  int i_each
) {

  // Init
  tk_spectral_t spec;
  tk_spectral_thread_t threads[pool->n_threads];
  spec.z = z->a;
  spec.laplacian = tk_malloc(L, n_nodes * sizeof(double));
  spec.adj_pos = adj_pos;
  spec.adj_neg = adj_neg;
  spec.n_nodes = n_nodes;
  spec.n_hidden = n_hidden;
  spec.pool = pool;
  for (unsigned int i = 0; i < pool->n_threads; i ++) {
    tk_spectral_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->spec = &spec;
    tk_thread_range(i, pool->n_threads, n_nodes, &data->ifirst, &data->ilast);
    tk_thread_range(i, pool->n_threads, n_hidden, &data->hfirst, &data->hlast);
  }

  // Init laplacian
  tk_threads_signal(pool, TK_SPECTRAL_INIT);

  // Run PRIMME to compute the smallest n_hidden + 1 eigenvectors of the graph
  // laplacian
  primme_params params;
  primme_initialize(&params);
  params.n = (int64_t) n_nodes;
  params.numEvals = n_hidden + 1;
  params.matrixMatvec = tk_spectral_matvec;
  params.matrix = &spec;
  params.printLevel = 0;
  params.eps = 1e-6;
  params.target = primme_closest_abs;
  params.numTargetShifts = 1;
  params.targetShifts = tk_malloc(L, (size_t) params.numTargetShifts * sizeof(double));
  params.targetShifts[0] = 0.0;
  spec.evals = tk_malloc(L, (size_t) params.numEvals * sizeof(double));
  spec.evecs = tk_malloc(L, (size_t) params.n * (size_t) params.numEvals * sizeof(double));
  spec.resNorms = tk_malloc(L, (size_t) params.numEvals * sizeof(double));
  primme_set_method(PRIMME_DEFAULT_MIN_TIME, &params);
  int ret = dprimme(spec.evals, spec.evecs, spec.resNorms, &params);
  if (ret != 0)
    tk_lua_verror(L, 2, "spectral", "failure calling PRIMME");

  // Copy eigenvectors into the output matrix, skipping the first
  tk_threads_signal(pool, TK_SPECTRAL_FINALIZE);

  // Zero-center
  tk_threads_signal(pool, TK_SPECTRAL_CENTER);

  // Cleanup
  free(spec.evals);
  free(spec.evecs);
  free(spec.resNorms);
  free(params.targetShifts);
  primme_free(&params);

  // Log
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, params.stats.numMatvecs);
    lua_call(L, 1, 0);
  }

  // Push laplacian
  tk_dvec_create(L, n_nodes, spec.laplacian, 0);
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
  // TODO: These are vectors of sets. Consider making them vectors of vectors or
  // fully flattened arrays for better matvec performance.
  tk_graph_adj_t *adj_pos = graph->adj_pos;
  tk_graph_adj_t *adj_neg = graph->adj_neg;

  // Spectral hashing
  tk_lua_get_ephemeron(L, TK_GRAPH_EPH, graph->uids);
  tk_dvec_t *z = tk_dvec_create(L, graph->uids->n * n_hidden, 0, 0);
  tm_run_spectral(L, pool, z, adj_pos, adj_neg, graph->uids->n, n_hidden, i_each);

  // Cleanup
  tk_threads_destroy(pool);
  return 3;
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
