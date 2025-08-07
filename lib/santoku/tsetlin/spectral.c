#include <santoku/tsetlin/graph.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/dvec.h>
#include <openblas/cblas.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <primme.h>

typedef enum {
  TK_SPECTRAL_CSR_OFFSET_LOCAL,
  TK_SPECTRAL_CSR_OFFSET_GLOBAL,
  TK_SPECTRAL_CSR_DATA,
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
  tk_ivec_t *adj_pos_offset;
  tk_ivec_t *adj_pos_data;
  tk_dvec_t *adj_pos_weights;
  tk_ivec_t *adj_neg_offset;
  tk_ivec_t *adj_neg_data;
  tk_dvec_t *adj_neg_weights;
  double pos_scale;
  bool normalized;
  uint64_t n_nodes;
  uint64_t n_hidden;
  uint64_t n_evals;
  tk_threadpool_t *pool;
} tk_spectral_t;

typedef struct {
  tk_spectral_t *spec;
  int64_t csr_pos_total;
  int64_t csr_neg_total;
  bool has_negatives;
  uint64_t ifirst, ilast;
  unsigned int index;
} tk_spectral_thread_t;

static inline void tk_spectral_worker (void *dp, int sig)
{
  tk_spectral_stage_t stage = (tk_spectral_stage_t) sig;
  tk_spectral_thread_t *data = (tk_spectral_thread_t *) dp;

  switch (stage) {

    case TK_SPECTRAL_CSR_OFFSET_LOCAL: {
      tk_graph_adj_item_t *adj_pos = data->spec->adj_pos->a;
      tk_graph_adj_item_t *adj_neg = data->spec->adj_neg->a;
      int64_t *adj_pos_offset = data->spec->adj_pos_offset->a;
      int64_t *adj_neg_offset = data->spec->adj_neg_offset->a;
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      int64_t poffset = 0;
      int64_t noffset = 0;
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        adj_pos_offset[i] = poffset;
        adj_neg_offset[i] = noffset;
        int64_t deg_pos = tk_iuset_size(adj_pos[i]);
        int64_t deg_neg = tk_iuset_size(adj_neg[i]);
        poffset += deg_pos;
        noffset += deg_neg;
      }
      data->csr_pos_total = poffset;
      data->csr_neg_total = noffset;
      break;
    }

    case TK_SPECTRAL_CSR_OFFSET_GLOBAL: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      int64_t *adj_pos_offset = data->spec->adj_pos_offset->a;
      int64_t *adj_neg_offset = data->spec->adj_neg_offset->a;
      int64_t csr_pos_total = data->csr_pos_total;
      int64_t csr_neg_total = data->csr_neg_total;
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        adj_pos_offset[i] += csr_pos_total;
        adj_neg_offset[i] += csr_neg_total;
      }
      break;
    }

    case TK_SPECTRAL_CSR_DATA: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      double *degree = data->spec->degree;
      tk_graph_adj_item_t *adj_pos = data->spec->adj_pos->a;
      tk_graph_adj_item_t *adj_neg = data->spec->adj_neg->a;
      int64_t *adj_pos_data = data->spec->adj_pos_data->a;
      int64_t *adj_neg_data = data->spec->adj_neg_data->a;
      int64_t *adj_pos_offset = data->spec->adj_pos_offset->a;
      int64_t *adj_neg_offset = data->spec->adj_neg_offset->a;
      double *adj_pos_weights = data->spec->adj_pos_weights->a;
      double *adj_neg_weights = data->spec->adj_neg_weights->a;
      tk_graph_t *graph = data->spec->graph;
      int64_t *uids = graph->uids->a;
      data->has_negatives = false;
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        int64_t u = uids[i];
        int64_t pwrite = adj_pos_offset[i];
        int64_t nwrite = adj_neg_offset[i];
        double wsum = 0.0;
        int64_t iv, v;
        double w;
        tk_iuset_foreach(adj_pos[i], iv, ({
          v = uids[iv];
          w = tk_graph_get_weight(graph, u, v);
          adj_pos_data[pwrite] = iv;
          adj_pos_weights[pwrite] = w;
          data->has_negatives = data->has_negatives || w < 0;
          wsum += fabs(w);
          pwrite ++;
        }))
        tk_iuset_foreach(adj_neg[i], iv, ({
          v = uids[iv];
          w = tk_graph_get_weight(graph, u, v);
          adj_neg_data[nwrite] = iv;
          adj_neg_weights[nwrite] = w;
          data->has_negatives = data->has_negatives || w < 0;
          wsum += fabs(w);
          nwrite ++;
        }))
        degree[i] = wsum;
      }
      break;
    }

    case TK_SPECTRAL_SCALE: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      double *scale = data->spec->scale;
      double *degree = data->spec->degree;
      bool normalized = data->spec->normalized;
      if (normalized)
        for (uint64_t i = ifirst; i <= ilast; i ++)
          scale[i] = degree[i] > 0 ? 1.0 / sqrt(degree[i]) : 0.0;
      else
        for (uint64_t i = ifirst; i <= ilast; i ++)
          scale[i] = 1.0;
      break;
    }

    case TK_SPECTRAL_MATVEC: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      double *scale = data->spec->scale;
      double *x = data->spec->x;
      double *y = data->spec->y;
      int64_t *adj_pos_data = data->spec->adj_pos_data->a;
      int64_t *adj_neg_data = data->spec->adj_neg_data->a;
      int64_t *adj_pos_offset = data->spec->adj_pos_offset->a;
      int64_t *adj_neg_offset = data->spec->adj_neg_offset->a;
      double *adj_pos_weights = data->spec->adj_pos_weights->a;
      double *adj_neg_weights = data->spec->adj_neg_weights->a;
      int64_t iv;
      double w;
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        double sum = 0.0;
        for (int64_t j = adj_pos_offset[i]; j < adj_pos_offset[i + 1]; j ++) {
          iv = adj_pos_data[j];
          w = adj_pos_weights[j];
          sum += x[iv] * scale[i] * scale[iv] * w;
        }
        for (int64_t j = adj_neg_offset[i]; j < adj_neg_offset[i + 1]; j ++) {
          iv = adj_neg_data[j];
          w = adj_neg_weights[j];
          sum += x[iv] * scale[i] * scale[iv] * w;
        }
        y[i] = x[i] - sum;
      }
      break;
    }

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
  tk_threads_signal(spec->pool, TK_SPECTRAL_MATVEC, 0);
  *ierr = 0;
}

// Consider primme params.method for perf/accuracy tradeoffs
static inline void tm_run_spectral (
  lua_State *L,
  int Gi,
  tk_threadpool_t *pool,
  tk_dvec_t *z,
  tk_dvec_t *scale,
  tk_dvec_t *degree,
  tk_graph_t *graph,
  tk_graph_adj_t *adj_pos,
  tk_graph_adj_t *adj_neg,
  uint64_t n_nodes,
  uint64_t n_hidden,
  double eps_primme,
  bool normalized,
  int i_each
) {
  // Init
  tk_spectral_t spec;
  tk_spectral_thread_t *threads = lua_newuserdata(L, pool->n_threads * sizeof(tk_spectral_thread_t));
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  spec.normalized = normalized;
  spec.scale = scale->a;
  spec.degree = degree->a;
  spec.graph = graph;
  spec.adj_pos = adj_pos;
  spec.adj_neg = adj_neg;
  spec.adj_pos_offset = tk_ivec_create(L, n_nodes + 1, 0, 0);
  spec.adj_neg_offset = tk_ivec_create(L, n_nodes + 1, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -2);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 2);
  spec.n_nodes = n_nodes;
  spec.n_hidden = n_hidden;
  spec.n_evals = n_hidden + 1;
  assert(spec.n_evals >= 2);
  spec.pool = pool;
  for (unsigned int i = 0; i < pool->n_threads; i ++) {
    tk_spectral_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->spec = &spec;
    data->index = i;
    data->has_negatives = false;
    tk_thread_range(i, pool->n_threads, n_nodes, &data->ifirst, &data->ilast);
  }

  // Init csr for fast matvecs
  tk_threads_signal(pool, TK_SPECTRAL_CSR_OFFSET_LOCAL, 0);
  int64_t pos_total = 0;
  int64_t neg_total = 0;
  for (unsigned int i = 0; i < pool->n_threads; i ++) {
    tk_spectral_thread_t *data = threads + i;
    int64_t pos_total0 = pos_total;
    int64_t neg_total0 = neg_total;
    pos_total += data->csr_pos_total;
    neg_total += data->csr_neg_total;
    data->csr_pos_total = pos_total0;
    data->csr_neg_total = neg_total0;
  }
  tk_threads_signal(pool, TK_SPECTRAL_CSR_OFFSET_GLOBAL, 0);
  spec.adj_pos_offset->a[spec.adj_pos_offset->n - 1] = pos_total;
  spec.adj_neg_offset->a[spec.adj_neg_offset->n - 1] = neg_total;
  spec.adj_pos_data = tk_ivec_create(L, (size_t) pos_total, 0, 0);
  spec.adj_neg_data = tk_ivec_create(L, (size_t) neg_total, 0, 0);
  spec.adj_pos_weights = tk_dvec_create(L, (size_t) pos_total, 0, 0);
  spec.adj_neg_weights = tk_dvec_create(L, (size_t) neg_total, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -4);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -3);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -2);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 4);
  tk_threads_signal(pool, TK_SPECTRAL_CSR_DATA, 0);

  // Init scaling/normalization
  tk_threads_signal(pool, TK_SPECTRAL_SCALE, 0);
  bool has_negatives = false;
  for (unsigned int i = 0; !has_negatives && i < pool->n_threads; i ++) {
    tk_spectral_thread_t *data = threads + i;
    has_negatives = data->has_negatives;
  }

  // Run PRIMME to compute the eigenvectors closest to zero
  openblas_set_num_threads((int) pool->n_threads);
  primme_params params;
  primme_initialize(&params);
  params.maxBlockSize = 1;
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
  int ret = dprimme(spec.evals, spec.evecs, spec.resNorms, &params);
  if (ret != 0) {
    free(spec.evals);
    free(spec.evecs);
    free(spec.resNorms);
    primme_free(&params);
    tk_lua_verror(L, 2, "spectral", "failure calling PRIMME");
    return;
  }

  tk_dvec_ensure(z, n_nodes * n_hidden);
  z->n = n_nodes * n_hidden;
  double eps_drop = fmax(1e-8, 10.0 * eps_primme);
  uint64_t start = !has_negatives || fabs(spec.evals[0]) < eps_drop ? 1 : 0;
  for (uint64_t i = 0; i < n_nodes; i ++) {
    for (uint64_t k = 0; k < n_hidden; k ++) {
      uint64_t f = start + k;
      z->a[i * n_hidden + k] = spec.evecs[i + f * n_nodes];
    }
  }

  for (uint64_t i = 0; i < spec.n_evals; i ++) {
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "eig");
      lua_pushinteger(L, (int64_t) i);
      lua_pushnumber(L, spec.evals[i]);
      lua_pushboolean(L, i >= start);
      lua_call(L, 4, 0);
    }
  }

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
}

static inline int tm_encode (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "graph");
  tk_graph_t *graph = tk_graph_peek(L, -1);
  if (tk_dsu_components(&graph->dsu) > 1)
    tk_lua_verror(L, 2, "spectral", "graph is not fully connected");

  int Gi = tk_lua_absindex(L, -1);
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "spectral", "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 1, "spectral", "threads");
  bool normalized = tk_lua_foptboolean(L, 1, "spectral", "normalized", true);
  double eps_primme = tk_lua_foptnumber(L, 1, "spectral", "eps_primme", 1e-4);

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
  tm_run_spectral(L, Gi, pool, z, scale, degree, graph, adj_pos, adj_neg,
                  graph->uids->n, n_hidden, eps_primme, normalized, i_each);
  lua_pop(L, 2); // degree, scale

  // Cleanup
  tk_threads_destroy(pool);
  assert(tk_ivec_peekopt(L, -2) == ids);
  assert(tk_dvec_peekopt(L, -1) == z);

  return 2; // ids, z
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
