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
#include <string.h>
#include <assert.h>
#include <primme.h>
#include <time.h>

#define TK_SPECTRAL_BLOCKSIZE 16

typedef enum {
  TK_SPECTRAL_SCALE,
  TK_SPECTRAL_MATVEC_EDGES,
  TK_SPECTRAL_MATVEC_REDUCE,
} tk_spectral_stage_t;

typedef enum {
  TK_LAPLACIAN_UNNORMALIZED,
  TK_LAPLACIAN_NORMALIZED,
  TK_LAPLACIAN_RANDOM
} tk_laplacian_type_t;

typedef struct {
  double *x, *y, *z;
  double *evals, *evecs, *resNorms;
  tk_dvec_t *scale;
  tk_dvec_t *degree;
  tk_graph_t *graph;
  tk_ivec_t *adj_offset;
  tk_ivec_t *adj_neighbors;
  tk_dvec_t *adj_weights;
  tk_ivec_t *adj_sources;
  tk_laplacian_type_t laplacian_type;
  uint64_t n_nodes;
  uint64_t n_hidden;
  uint64_t n_evals;
  uint64_t n_edges;
  tk_threadpool_t *pool;
  int blockSize;
  PRIMME_INT ldx;
  PRIMME_INT ldy;
  double time_edges;
  double time_reduce;
  double time_clear;
  uint64_t matvec_count;
} tk_spectral_t;

typedef struct {
  tk_spectral_t *spec;
  uint64_t ifirst, ilast;
  uint64_t edge_first, edge_last;
  double *y_local;
  unsigned int index;
} tk_spectral_thread_t;


static inline void tk_spectral_worker (void *dp, int sig)
{
  tk_spectral_stage_t stage = (tk_spectral_stage_t) sig;
  tk_spectral_thread_t *data = (tk_spectral_thread_t *) dp;

  switch (stage) {

    case TK_SPECTRAL_SCALE: {
      const uint64_t ifirst = data->ifirst;
      const uint64_t ilast = data->ilast;
      const tk_laplacian_type_t laplacian_type = data->spec->laplacian_type;
      double * restrict degree = data->spec->degree->a;
      double * restrict scale = data->spec->scale->a;
      const int64_t * restrict adj_offset = data->spec->adj_offset->a;
      const double * restrict adj_weights = data->spec->adj_weights->a;
      if (laplacian_type == TK_LAPLACIAN_UNNORMALIZED) {
        for (uint64_t i = ifirst; i <= ilast; i++) {
          double sum = 0.0;
          const int64_t j_start = adj_offset[i];
          const int64_t j_end = adj_offset[i + 1];
          for (int64_t j = j_start; j < j_end; j++)
            sum += adj_weights[j];
          degree[i] = sum;
          scale[i] = 1.0;
        }
      } else {
        for (uint64_t i = ifirst; i <= ilast; i++) {
          double sum = 0.0;
          const int64_t j_start = adj_offset[i];
          const int64_t j_end = adj_offset[i + 1];
          for (int64_t j = j_start; j < j_end; j++)
            sum += adj_weights[j];
          degree[i] = sum;
          scale[i] = sum > 0.0 ? 1.0 / sqrt(sum) : 0.0;
        }
      }
      break;
    }

    case TK_SPECTRAL_MATVEC_EDGES: {
      const uint64_t edge_first = data->edge_first;
      const uint64_t edge_last = data->edge_last;
      const uint64_t node_first = data->ifirst;
      const uint64_t node_last = data->ilast;
      const uint64_t n_nodes = data->spec->n_nodes;
      const int blockSize = data->spec->blockSize;
      const PRIMME_INT ldx = data->spec->ldx;
      const tk_laplacian_type_t laplacian_type = data->spec->laplacian_type;
      double * restrict x = data->spec->x;
      double * restrict y_local = data->y_local;
      const int64_t * restrict adj_neighbors = data->spec->adj_neighbors->a;
      const int64_t * restrict adj_sources = data->spec->adj_sources->a;
      const double * restrict adj_weights = data->spec->adj_weights->a;
      const double * restrict degree = data->spec->degree->a;
      const double * restrict scale = data->spec->scale->a;
      memset(y_local, 0, (size_t) n_nodes * (size_t) blockSize * sizeof(double));
      if (laplacian_type == TK_LAPLACIAN_UNNORMALIZED) {
        for (uint64_t e = edge_first; e <= edge_last; e++) {
          const int64_t src = adj_sources[e];
          const int64_t dst = adj_neighbors[e];
          const double weight = adj_weights[e];
          for (int b = 0; b < blockSize; b++) {
            y_local[(size_t) b * n_nodes + (size_t) src] -= weight * x[(size_t) b * (size_t) ldx + (size_t) dst];
          }
        }
        for (int b = 0; b < blockSize; b++) {
          double * restrict yb = y_local + (size_t) b * n_nodes;
          const double * restrict xb = x + (size_t) b * (size_t) ldx;
          for (uint64_t i = node_first; i <= node_last; i++)
            yb[i] += degree[i] * xb[i];
        }
      } else {
        for (uint64_t e = edge_first; e <= edge_last; e++) {
          const int64_t src = adj_sources[e];
          const int64_t dst = adj_neighbors[e];
          const double scaled_weight = scale[src] * scale[dst] * adj_weights[e];
          for (int b = 0; b < blockSize; b++) {
            y_local[(size_t) b * n_nodes + (size_t) src] -= scaled_weight * x[(size_t) b * (size_t) ldx + (size_t) dst];
          }
        }
        for (int b = 0; b < blockSize; b++) {
          double * restrict yb = y_local + (size_t) b * n_nodes;
          const double * restrict xb = x + (size_t) b * (size_t) ldx;
          for (uint64_t i = node_first; i <= node_last; i++)
            yb[i] += xb[i];
        }
      }
      break;
    }

    case TK_SPECTRAL_MATVEC_REDUCE: {
      const uint64_t node_first = data->ifirst;
      const uint64_t node_last = data->ilast;
      const uint64_t n_nodes = data->spec->n_nodes;
      const int blockSize = data->spec->blockSize;
      const PRIMME_INT ldy = data->spec->ldy;
      const unsigned int n_threads = data->spec->pool->n_threads;
      double * restrict y = data->spec->y;
      tk_spectral_thread_t *threads = (tk_spectral_thread_t *) data->spec->pool->threads[0].data;
      for (int b = 0; b < blockSize; b++) {
        double *yb = y + (size_t) b * (size_t) ldy;
        const size_t b_offset = (size_t) b * n_nodes;
        for (uint64_t i = node_first; i <= node_last; i++) {
          double sum = 0.0;
          for (unsigned int t = 0; t < n_threads; t ++)
            sum += threads[t].y_local[b_offset + i];
          yb[i] = sum;
        }
      }
      break;
    }

  }
}

static inline void tk_spectral_preconditioner (
  void *vx,
  PRIMME_INT *ldx,
  void *vy,
  PRIMME_INT *ldy,
  int *blockSize,
  struct primme_params *primme,
  int *ierr
) {
  tk_spectral_t *spec = (tk_spectral_t *) primme->matrix;
  double *xvec = (double *) vx;
  double *yvec = (double *) vy;

  if (spec->laplacian_type == TK_LAPLACIAN_UNNORMALIZED) {
    for (int b = 0; b < *blockSize; b++)
      for (uint64_t i = 0; i < spec->n_nodes; i++) {
        double d = spec->degree->a[i];
        yvec[(uint64_t) b * (size_t) (*ldy) + i] = d > 0 ? xvec[(uint64_t) b * (size_t) (*ldx) + i] / d : xvec[(uint64_t) b * (size_t) (*ldx) + i];
      }
  } else if (spec->laplacian_type == TK_LAPLACIAN_NORMALIZED) {
    for (int b = 0; b < *blockSize; b++)
      memcpy(yvec + (uint64_t) b * (size_t) (*ldy), xvec + (uint64_t) b * (size_t) (*ldx), spec->n_nodes * sizeof(double));
  } else {
    for (int b = 0; b < *blockSize; b++)
      memcpy(yvec + (uint64_t) b * (size_t) (*ldy), xvec + (uint64_t) b * (size_t) (*ldx), spec->n_nodes * sizeof(double));
  }
  *ierr = 0;
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
  tk_spectral_t *spec = (tk_spectral_t *) primme->matrix;
  double *x = (double *) vx;
  double *y = (double *) vy;
  spec->x = x;
  spec->y = y;
  spec->blockSize = *blockSize;
  spec->ldx = *ldx;
  spec->ldy = *ldy;
  struct timespec t1, t2, t3, t4;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  memset(vy, 0, (size_t) spec->n_nodes * (size_t) (*blockSize) * sizeof(double));
  clock_gettime(CLOCK_MONOTONIC, &t2);
  tk_threads_signal(spec->pool, TK_SPECTRAL_MATVEC_EDGES, 0);
  clock_gettime(CLOCK_MONOTONIC, &t3);
  tk_threads_signal(spec->pool, TK_SPECTRAL_MATVEC_REDUCE, 0);
  clock_gettime(CLOCK_MONOTONIC, &t4);
  spec->time_clear += (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) * 1e-9;
  spec->time_edges += (t3.tv_sec - t2.tv_sec) + (t3.tv_nsec - t2.tv_nsec) * 1e-9;
  spec->time_reduce += (t4.tv_sec - t3.tv_sec) + (t4.tv_nsec - t3.tv_nsec) * 1e-9;
  spec->matvec_count++;
  *ierr = 0;
}

static inline void tm_run_spectral (
  lua_State *L,
  tk_threadpool_t *pool,
  tk_dvec_t *z,
  tk_dvec_t *scale,
  tk_dvec_t *degree,
  tk_ivec_t *uids,
  tk_ivec_t *adj_offset,
  tk_ivec_t *adj_neighbors,
  tk_dvec_t *adj_weights,
  tk_ivec_t *adj_sources,
  uint64_t n_hidden,
  double eps,
  tk_laplacian_type_t laplacian_type,
  int i_each
) {
  tk_spectral_t spec;
  memset(&spec, 0, sizeof(tk_spectral_t));
  tk_spectral_thread_t *threads = tk_malloc(L, pool->n_threads * sizeof(tk_spectral_thread_t));
  spec.laplacian_type = laplacian_type;
  spec.scale = scale;
  spec.degree = degree;
  spec.adj_offset = adj_offset;
  spec.adj_neighbors = adj_neighbors;
  spec.adj_weights = adj_weights;
  spec.n_nodes = uids->n;
  spec.n_hidden = n_hidden;
  spec.n_evals = n_hidden + 1;
  assert(spec.n_evals >= 2);
  spec.pool = pool;
  spec.n_edges = (uint64_t) adj_offset->a[uids->n];
  spec.adj_sources = adj_sources;

  for (unsigned int i = 0; i < pool->n_threads; i ++) {
    tk_spectral_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->spec = &spec;
    data->index = i;
    tk_thread_range(i, pool->n_threads, uids->n, &data->ifirst, &data->ilast);
    tk_thread_range(i, pool->n_threads, spec.n_edges, &data->edge_first, &data->edge_last);
    data->y_local = calloc((size_t) uids->n * TK_SPECTRAL_BLOCKSIZE, sizeof(double));
  }

  tk_threads_signal(pool, TK_SPECTRAL_SCALE, 0);

  openblas_set_num_threads((int) pool->n_threads);
  primme_params params;
  primme_initialize(&params);
  params.n = (int64_t) uids->n;
  params.numEvals = spec.n_evals;
  params.matrixMatvec = tk_spectral_matvec;
  params.matrix = &spec;
  params.eps = eps;
  params.printLevel = 0;
  params.target = primme_smallest;
  params.maxBlockSize = TK_SPECTRAL_BLOCKSIZE;
  primme_set_method(PRIMME_DEFAULT_MIN_TIME, &params);

  params.applyPreconditioner = tk_spectral_preconditioner;
  params.correctionParams.precondition = 1;

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
  double eps_drop = fmax(1e-8, 10.0 * eps);
  uint64_t start = fabs(spec.evals[0]) < eps_drop ? 1 : 0;

  tk_dvec_t *eigenvalues = tk_dvec_create(L, n_hidden, 0, 0);
  eigenvalues->n = n_hidden;

  for (uint64_t i = 0; i < uids->n; i ++) {
    for (uint64_t k = 0; k < n_hidden; k ++) {
      uint64_t f = start + k;
      double eigval = spec.evecs[i + f * uids->n];
      if (laplacian_type == TK_LAPLACIAN_RANDOM)
        eigval = scale->a[i] > 0.0 ? eigval / scale->a[i] : eigval;
      z->a[i * n_hidden + k] = eigval;
      if (i == 0)
        eigenvalues->a[k] = spec.evals[f];
    }
  }

  tk_dvec_center(z->a, uids->n, n_hidden);
  tk_dvec_rnorml2(z->a, uids->n, n_hidden);

  for (uint64_t i = 0; i < spec.n_evals; i ++) {
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "eig");
      lua_pushinteger(L, (int64_t) i);
      lua_pushnumber(L, spec.evals[i]);
      lua_pushboolean(L, i >= start);
      lua_call(L, 4, 0);
      // TODO: Set things up such that memory is correctly freed even if this
      // throws
    }
  }

  free(spec.evals);
  free(spec.evecs);
  free(spec.resNorms);
  primme_free(&params);

  for (unsigned int i = 0; i < pool->n_threads; i++) {
    tk_spectral_thread_t *data = threads + i;
    free(data->y_local);
  }
  free(threads);

  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushstring(L, "done");
    lua_pushinteger(L, params.stats.numMatvecs);
    lua_pushnumber(L, spec.time_clear / spec.matvec_count);
    lua_pushnumber(L, spec.time_edges / spec.matvec_count);
    lua_pushnumber(L, spec.time_reduce / spec.matvec_count);
    lua_call(L, 5, 0);
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
  tk_ivec_t *adj_neighbors = tk_ivec_peek(L, -1, "neighbors");

  lua_getfield(L, 1, "weights");
  tk_dvec_t *adj_weights = tk_dvec_peek(L, -1, "weights");

  lua_getfield(L, 1, "sources");
  tk_ivec_t *adj_sources = tk_ivec_peek(L, -1, "sources");

  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "spectral", "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 1, "spectral", "threads");
  double eps = tk_lua_foptnumber(L, 1, "spectral", "eps", 1e-12);
  const char *type_str = tk_lua_foptstring(L, 1, "spectral", "type", "random");
  tk_laplacian_type_t laplacian_type = TK_LAPLACIAN_RANDOM;

  if (strcmp(type_str, "unnormalized") == 0) {
    laplacian_type = TK_LAPLACIAN_UNNORMALIZED;
  } else if (strcmp(type_str, "normalized") == 0) {
    laplacian_type = TK_LAPLACIAN_NORMALIZED;
  } else if (strcmp(type_str, "random") == 0) {
    laplacian_type = TK_LAPLACIAN_RANDOM;
  }

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_spectral_worker);

  lua_pushvalue(L, i_uids); // ids
  tk_dvec_t *z = tk_dvec_create(L, 0, 0, 0); // ids, z
  tk_dvec_t *scale = tk_dvec_create(L, uids->n, 0, 0); // ids, z, scale
  tk_dvec_t *degree = tk_dvec_create(L, uids->n, 0, 0); // ids, z, scale, degree
  tm_run_spectral(L, pool, z, scale, degree, uids, adj_offset, adj_neighbors,
                  adj_weights, adj_sources, n_hidden, eps,
                  laplacian_type, i_each);
  lua_remove(L, -2);

  tk_threads_destroy(pool);
  assert(tk_ivec_peekopt(L, -4) == uids);
  assert(tk_dvec_peekopt(L, -3) == z);
  return 4;
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
