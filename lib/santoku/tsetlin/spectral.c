#define _GNU_SOURCE

#include <santoku/tsetlin/graph.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/dvec.h>
#include <float.h>
#include <primme.h>

typedef enum {
  TK_SPECTRAL_COUNT,
  TK_SPECTRAL_SCALE,
  TK_SPECTRAL_MATVEC,
} tk_spectral_stage_t;

typedef struct {
  double *x, *y, *z;
  double *evals, *evecs, *resNorms;
  double *scale;
  double *degree;
  tk_graph_t *graph;
  tk_graph_adj_t *adj_pos;
  tk_graph_adj_t *adj_neg;
  tk_ivec_t *kept;
  bool normalized;
  uint64_t n_nodes;
  uint64_t n_hidden;
  uint64_t n_evals;
  double neg_scale;
  tk_threadpool_t *pool;
} tk_spectral_t;

typedef struct {
  tk_spectral_t *spec;
  uint64_t ifirst, ilast;
  uint64_t kfirst, klast;
} tk_spectral_thread_t;

static inline void tk_spectral_worker (void *dp, int sig)
{
  tk_spectral_stage_t stage = (tk_spectral_stage_t) sig;
  tk_spectral_thread_t *data = (tk_spectral_thread_t *) dp;
  double *x = data->spec->x;
  double *y = data->spec->y;
  double *scale = data->spec->scale;
  double *degree = data->spec->degree;
  double neg_scale = data->spec->neg_scale;
  uint64_t ifirst = data->ifirst;
  uint64_t ilast = data->ilast;
  tk_graph_adj_t *adj_pos = data->spec->adj_pos;
  tk_graph_adj_t *adj_neg = data->spec->adj_neg;
  int64_t iv;

  bool normalized = data->spec->normalized;

  double w_pos = 1.0;
  double w_neg = neg_scale;

  switch (stage) {

    case TK_SPECTRAL_COUNT:
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        double deg_pos = (double) tk_iuset_size(adj_pos->a[i]) * w_pos;
        double deg_neg = (double) tk_iuset_size(adj_neg->a[i]) * fabs(w_neg);
        degree[i] = deg_pos + deg_neg;
      }
      break;

    case TK_SPECTRAL_SCALE:
      if (normalized) {
        for (uint64_t i = ifirst; i <= ilast; i ++) {
          scale[i] = degree[i] > 0 ? 1.0 / sqrt(degree[i]) : 0.0;
          if (!isfinite(scale[i]))
            scale[i] = 0.0;
        }
      } else {
        for (uint64_t i = ifirst; i <= ilast; i ++) {
          scale[i] = 1.0;
        }
      }
      break;

    case TK_SPECTRAL_MATVEC:
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        double sum = 0.0;
        // Positive edges
        tk_iuset_foreach(adj_pos->a[i], iv, ({
          sum += x[iv] * scale[i] * scale[iv] * w_pos;
        }))
        // Negative edges
        tk_iuset_foreach(adj_neg->a[i], iv, ({
          sum += x[iv] * scale[i] * scale[iv] * w_neg;
        }))
        y[i] = x[i] - sum;
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
  tk_dvec_t *scale,
  tk_dvec_t *degree,
  tk_graph_t *graph,
  tk_graph_adj_t *adj_pos,
  tk_graph_adj_t *adj_neg,
  uint64_t n_nodes,
  uint64_t n_hidden,
  int64_t n_fixed,
  double eps_primme,
  double eps_keep,
  bool normalized,
  int i_each,
  double *neg_scalep,
  uint64_t *n_dimsp
) {
  // Init
  tk_spectral_t spec;
  tk_spectral_thread_t threads[pool->n_threads];
  spec.normalized = normalized;
  spec.scale = scale->a;
  spec.degree = degree->a;
  spec.graph = graph;
  spec.adj_pos = adj_pos;
  spec.adj_neg = adj_neg;
  spec.n_nodes = n_nodes;
  spec.n_hidden = n_hidden;
  spec.n_evals = n_hidden + 1;
  spec.neg_scale = (*neg_scalep);
  assert(spec.n_evals >= 2);
  spec.pool = pool;
  for (unsigned int i = 0; i < pool->n_threads; i ++) {
    tk_spectral_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->spec = &spec;
    tk_thread_range(i, pool->n_threads, n_nodes, &data->ifirst, &data->ilast);
    tk_thread_range(i, pool->n_threads, spec.n_evals, &data->kfirst, &data->klast);
  }

  // Init total pull
  tk_threads_signal(pool, TK_SPECTRAL_COUNT);

  // Init scaling/normalization
  tk_threads_signal(pool, TK_SPECTRAL_SCALE);

  // Run PRIMME to compute the eigenvectors closest to zero
  primme_params params;
  primme_initialize(&params);
  params.n = (int64_t) n_nodes;
  params.numEvals = spec.n_evals;
  params.matrixMatvec = tk_spectral_matvec;
  params.matrix = &spec;
  params.printLevel = 0;
  params.eps = eps_primme;
  params.target = primme_closest_abs;
  params.numTargetShifts = 1;
  params.targetShifts = (double[]) { 0.0 };
  spec.evals = tk_malloc(L, (size_t) spec.n_evals * sizeof(double));
  spec.evecs = tk_malloc(L, (size_t) params.n * (size_t) spec.n_evals * sizeof(double));
  spec.resNorms = tk_malloc(L, (size_t) spec.n_evals * sizeof(double));
  // primme_set_method(PRIMME_DEFAULT_MIN_TIME, &params);
  int ret = dprimme(spec.evals, spec.evecs, spec.resNorms, &params);
  if (ret != 0)
    tk_lua_verror(L, 2, "spectral", "failure calling PRIMME");

  // Determine vectors to keep
  spec.kept = tk_ivec_create(L, 0, 0, 0);
  uint64_t n_dims = 0;
  if (n_fixed == -1) {
    uint64_t elbow_idx = n_hidden;
    double max_jump = 0.0;
    uint64_t start = 0;
    while (start < (uint64_t) spec.n_evals && spec.evals[start] < eps_keep)
      ++ start;
    for (uint64_t i = start + 1; i < (uint64_t) spec.n_evals; i ++) {
      double eval = spec.evals[i];
      double jump = fabs(eval - spec.evals[i - 1]);
      if (jump > max_jump) {
        max_jump = jump;
        elbow_idx = i;
      }
    }
    for (uint64_t i = 0; i < spec.n_evals; i++) {
      bool keep = spec.evals[i] > eps_keep && i < elbow_idx;
      if (keep)
        tk_ivec_push(spec.kept, (int64_t) i);
      if (i_each != -1) {
        lua_pushvalue(L, i_each);
        lua_pushstring(L, "eig");
        lua_pushinteger(L, (int64_t) i);
        lua_pushnumber(L, spec.evals[i]);
        lua_pushboolean(L, keep);
        lua_call(L, 4, 0);
      }
    }
  } else {
    uint64_t target = !n_fixed ? n_hidden : (uint64_t) llabs(n_fixed);
    double cut = n_fixed < 0 ? -DBL_MAX : eps_keep;
    for (uint64_t i = 0; i < spec.n_evals; i++) {
      bool keep = fabs(spec.evals[i]) > cut && spec.kept->n < target;
      if (keep)
        tk_ivec_push(spec.kept, (int64_t) i);
      if (i_each != -1) {
        lua_pushvalue(L, i_each);
        lua_pushstring(L, "eig");
        lua_pushinteger(L, (int64_t) i);
        lua_pushnumber(L, spec.evals[i]);
        lua_pushboolean(L, keep);
        lua_call(L, 4, 0);
      }
    }
  }
  n_dims = spec.kept->n;
  // Emit to z (drop unselected)
  // TODO: Multithread this
  tk_dvec_ensure(L, z, n_nodes * n_dims);
  z->n = n_nodes * n_dims;
  for (uint64_t i = 0; i < n_nodes; i ++) {
    for (uint64_t k = 0; k < n_dims; k ++) {
      uint64_t f = (uint64_t) spec.kept->a[k];
      z->a[i * n_dims + k] = spec.evecs[ i + f * n_nodes ];
    }
  }
  lua_pop(L, 1);

  // Cleanup
  free(spec.evals);
  free(spec.evecs);
  free(spec.resNorms);
  primme_free(&params);

  // Log
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushstring(L, "done");
    lua_pushinteger(L, params.stats.numMatvecs);
    lua_call(L, 2, 0);
  }

  *n_dimsp = spec.kept->n;
  *neg_scalep = spec.neg_scale;
}

static inline int tm_encode (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "graph");
  tk_graph_t *graph = tk_graph_peek(L, -1);
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "spectral", "n_hidden");
  int64_t n_fixed = tk_lua_foptinteger(L, 1, "spectral", "n_fixed", -1);
  unsigned int n_threads = tk_threads_getn(L, 1, "spectral", "threads");
  double neg_scale = tk_lua_foptnumber(L, 1, "spectral", "negatives", -1.0);
  bool normalized = tk_lua_foptboolean(L, 1, "spectral", "normalized", false);
  double eps_primme = tk_lua_foptnumber(L, 1, "spectral", "eps_primme", 1e-4);
  double eps_keep = tk_lua_foptnumber(L, 1, "spectral", "eps_keep", 1e-4);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_spectral_worker);
  tk_graph_adj_t *adj_pos = graph->adj_pos;
  tk_graph_adj_t *adj_neg = graph->adj_neg;

  // Spectral hashing
  tk_lua_get_ephemeron(L, TK_GRAPH_EPH, graph->uids);
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);
  tk_dvec_t *z = tk_dvec_create(L, 0, 0, 0);
  tk_dvec_t *scale = tk_dvec_create(L, graph->uids->n, 0, 0);
  tk_dvec_t *degree = tk_dvec_create(L, graph->uids->n, 0, 0);
  uint64_t n_dims = 0;
  tm_run_spectral(L, pool, z, scale, degree, graph, adj_pos, adj_neg,
                  graph->uids->n, n_hidden, n_fixed, eps_primme, eps_keep,
                  normalized, i_each, &neg_scale, &n_dims);
  lua_pop(L, 1); // degrees
  lua_pushinteger(L, (int64_t) n_dims);
  lua_pushnumber(L, neg_scale);

  // Cleanup
  tk_threads_destroy(pool);
  assert(tk_ivec_peekopt(L, -5) == ids);
  assert(tk_dvec_peekopt(L, -4) == z);
  assert(tk_dvec_peekopt(L, -3) == scale);
  assert(lua_tointeger(L, -2) == (int64_t) n_dims);
  assert(lua_tonumber(L, -1) == neg_scale);
  return 5; // ids, z, scale, n_dims, neg_scale
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
