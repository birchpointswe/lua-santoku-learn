#include <santoku/tsetlin/graph.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <openblas/cblas.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <primme.h>

typedef enum {
  TK_SPECTRAL_SCALE,
  TK_SPECTRAL_MATVEC,
} tk_spectral_stage_t;

typedef struct {
  double *x, *y, *z;
  double *evals, *evecs, *resNorms;
  double *scale;
  double *degree;
  tk_graph_t *graph;
  tk_ivec_t *adj_offset;
  tk_ivec_t *adj_data;
  tk_dvec_t *adj_weights;
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

    case TK_SPECTRAL_SCALE: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      double *degree = data->spec->degree;
      double *scale = data->spec->scale;
      int64_t *adj_offset = data->spec->adj_offset->a;
      double *adj_weights = data->spec->adj_weights->a;
      bool normalized = data->spec->normalized;
      data->has_negatives = false;
      for (uint64_t i = ifirst; i <= ilast; i++) {
        double sum = 0.0;
        for (int64_t j = adj_offset[i]; j < adj_offset[i + 1]; j++) {
          double w = adj_weights[j];
          if (w < 0)
            data->has_negatives = true;
          sum += fabs(w);
        }
        degree[i] = sum;
        scale[i] = normalized ? (sum > 0.0 ? 1.0 / sqrt(sum) : 0.0) : 1.0;
      }
      break;
    }

    case TK_SPECTRAL_MATVEC: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      double *scale = data->spec->scale;
      double *x = data->spec->x;
      double *y = data->spec->y;
      int64_t *adj_data = data->spec->adj_data->a;
      int64_t *adj_offset = data->spec->adj_offset->a;
      double *adj_weights = data->spec->adj_weights->a;
      double *degree = data->spec->degree;
      bool normalized = data->spec->normalized;
      int64_t iv;
      double w;
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        double sum = 0.0;
        for (int64_t j = adj_offset[i]; j < adj_offset[i + 1]; j ++) {
          iv = adj_data[j];
          w = adj_weights[j];
          if (normalized) {
            sum += x[iv] * scale[i] * scale[iv] * w;
          } else {
            sum += x[iv] * w;
          }
        }
        y[i] = (normalized ? 1.0 : degree[i]) * x[i] - sum;
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
  tk_threadpool_t *pool,
  tk_dvec_t *z,
  tk_dvec_t *scale,
  tk_dvec_t *degree,
  tk_ivec_t *uids,
  tk_ivec_t *adj_offset,
  tk_ivec_t *adj_data,
  tk_dvec_t *adj_weights,
  uint64_t n_hidden,
  double eps_primme,
  bool normalized,
  int i_each
) {
  // Init
  tk_spectral_t spec;
  tk_spectral_thread_t *threads = tk_malloc(L, pool->n_threads * sizeof(tk_spectral_thread_t));
  spec.normalized = normalized;
  spec.scale = scale->a;
  spec.degree = degree->a;
  spec.adj_offset = adj_offset;
  spec.adj_data = adj_data;
  spec.adj_weights = adj_weights;
  spec.n_nodes = uids->n;
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
    tk_thread_range(i, pool->n_threads, uids->n, &data->ifirst, &data->ilast);
  }

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
  params.n = (int64_t) uids->n;
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
    free(threads);
    tk_lua_verror(L, 2, "spectral", "failure calling PRIMME");
    return;
  }

  tk_dvec_ensure(z, uids->n * n_hidden);
  z->n = uids->n * n_hidden;
  double eps_drop = fmax(1e-8, 10.0 * eps_primme);
  uint64_t start = !has_negatives || fabs(spec.evals[0]) < eps_drop ? 1 : 0;
  for (uint64_t i = 0; i < uids->n; i ++) {
    for (uint64_t k = 0; k < n_hidden; k ++) {
      uint64_t f = start + k;
      z->a[i * n_hidden + k] = spec.evecs[i + f * uids->n];
    }
  }
  
  // Center each eigenvector dimension to have zero mean
  tk_dvec_center(z->a, uids->n, n_hidden);

  for (uint64_t i = 0; i < spec.n_evals; i ++) {
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "eig");
      lua_pushinteger(L, (int64_t) i);
      lua_pushnumber(L, spec.evals[i]);
      lua_pushboolean(L, i >= start);
      // NOTE: this hides errors in callback
      if (lua_pcall(L, 4, 0, 0))
        lua_pop(L, 1);
    }
  }

  // Cleanup
  free(spec.evals);
  free(spec.evecs);
  free(spec.resNorms);
  primme_free(&params);
  free(threads);

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

  lua_getfield(L, 1, "ids");
  tk_ivec_t *uids = tk_ivec_peek(L, -1, "ids");
  int i_uids = tk_lua_absindex(L, -1);

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *adj_offset = tk_ivec_peek(L, -1, "offsets");

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *adj_data = tk_ivec_peek(L, -1, "neighbors");

  lua_getfield(L, 1, "weights");
  tk_dvec_t *adj_weights = tk_dvec_peek(L, -1, "weights");

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

  // Spectral hashing
  lua_pushvalue(L, i_uids); // ids
  tk_dvec_t *z = tk_dvec_create(L, 0, 0, 0); // ids, z
  tk_dvec_t *scale = tk_dvec_create(L, uids->n, 0, 0); // ids, z, scale
  tk_dvec_t *degree = tk_dvec_create(L, uids->n, 0, 0); // ids, z, scale, degree
  tm_run_spectral(L, pool, z, scale, degree, uids, adj_offset, adj_data, adj_weights, n_hidden, eps_primme, normalized, i_each);
  lua_pop(L, 2); // ids, z

  // Cleanup
  tk_threads_destroy(pool);
  assert(tk_ivec_peekopt(L, -2) == uids);
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
