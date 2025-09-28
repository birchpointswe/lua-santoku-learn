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

typedef enum {
  TK_SPECTRAL_SCALE,
  TK_SPECTRAL_MATVEC,
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
  tk_laplacian_type_t laplacian_type;
  uint64_t n_nodes;
  uint64_t n_hidden;
  uint64_t n_evals;
  tk_threadpool_t *pool;
  int blockSize;
  PRIMME_INT ldx;
  PRIMME_INT ldy;
} tk_spectral_t;

typedef struct {
  tk_spectral_t *spec;
  uint64_t ifirst, ilast;
  unsigned int index;
} tk_spectral_thread_t;

static inline void tk_spectral_matvec_unnormalized (
  uint64_t ifirst, uint64_t ilast,
  double *x, double *y,
  int64_t *adj_neighbors, int64_t *adj_offset, double *adj_weights,
  double *degree,
  int blockSize, PRIMME_INT ldx, PRIMME_INT ldy
) {
  for (int b = 0; b < blockSize; b++) {
    double *xb = x + b * ldx;
    double *yb = y + b * ldy;
    for (uint64_t i = ifirst; i <= ilast; i++) {
      double sum = 0.0;
      for (int64_t j = adj_offset[i]; j < adj_offset[i + 1]; j++) {
        int64_t iv = adj_neighbors[j];
        sum += xb[iv] * adj_weights[j];
      }
      yb[i] = degree[i] * xb[i] - sum;
    }
  }
}

static inline void tk_spectral_matvec_normalized (
  uint64_t ifirst, uint64_t ilast,
  double *x, double *y,
  int64_t *adj_neighbors, int64_t *adj_offset, double *adj_weights,
  double *scale,
  int blockSize, PRIMME_INT ldx, PRIMME_INT ldy
) {
  for (int b = 0; b < blockSize; b++) {
    double *xb = x + b * ldx;
    double *yb = y + b * ldy;
    for (uint64_t i = ifirst; i <= ilast; i++) {
      double sum = 0.0;
      double scale_i = scale[i];
      for (int64_t j = adj_offset[i]; j < adj_offset[i + 1]; j++) {
        int64_t iv = adj_neighbors[j];
        sum += xb[iv] * scale_i * scale[iv] * adj_weights[j];
      }
      yb[i] = 1.0 * xb[i] - sum;
    }
  }
}

static inline void tk_spectral_worker (void *dp, int sig)
{
  tk_spectral_stage_t stage = (tk_spectral_stage_t) sig;
  tk_spectral_thread_t *data = (tk_spectral_thread_t *) dp;

  switch (stage) {

    case TK_SPECTRAL_SCALE: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      double *degree = data->spec->degree->a;
      double *scale = data->spec->scale->a;
      int64_t *adj_offset = data->spec->adj_offset->a;
      double *adj_weights = data->spec->adj_weights->a;
      tk_laplacian_type_t laplacian_type = data->spec->laplacian_type;
      for (uint64_t i = ifirst; i <= ilast; i++) {
        double sum = 0.0;
        for (int64_t j = adj_offset[i]; j < adj_offset[i + 1]; j++) {
          double w = adj_weights[j];
          sum += w;
        }
        degree[i] = sum;
        if (laplacian_type == TK_LAPLACIAN_NORMALIZED) {
          scale[i] = sum > 0.0 ? 1.0 / sqrt(sum) : 0.0;
        } else if (laplacian_type == TK_LAPLACIAN_RANDOM) {
          scale[i] = sum > 0.0 ? 1.0 / sqrt(sum) : 0.0;
        } else {
          scale[i] = 1.0;
        }
      }
      break;
    }

    case TK_SPECTRAL_MATVEC: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      double *x = data->spec->x;
      double *y = data->spec->y;
      int64_t *adj_neighbors = data->spec->adj_neighbors->a;
      int64_t *adj_offset = data->spec->adj_offset->a;
      double *adj_weights = data->spec->adj_weights->a;
      int blockSize = data->spec->blockSize;
      PRIMME_INT ldx = data->spec->ldx;
      PRIMME_INT ldy = data->spec->ldy;
      switch (data->spec->laplacian_type) {
        case TK_LAPLACIAN_UNNORMALIZED:
          tk_spectral_matvec_unnormalized(
            ifirst, ilast, x, y,
            adj_neighbors, adj_offset, adj_weights,
            data->spec->degree->a,
            blockSize, ldx, ldy
          );
          break;
        case TK_LAPLACIAN_NORMALIZED:
        case TK_LAPLACIAN_RANDOM:
          tk_spectral_matvec_normalized(
            ifirst, ilast, x, y,
            adj_neighbors, adj_offset, adj_weights,
            data->spec->scale->a,
            blockSize, ldx, ldy
          );
          break;
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
  tk_spectral_t *spec = (tk_spectral_t *) primme->matrix;
  spec->x = (double *) vx;
  spec->y = (double *) vy;
  spec->blockSize = *blockSize;
  spec->ldx = *ldx;
  spec->ldy = *ldy;
  tk_threads_signal(spec->pool, TK_SPECTRAL_MATVEC, 0);
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
  uint64_t n_hidden,
  double eps_primme,
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
  for (unsigned int i = 0; i < pool->n_threads; i ++) {
    tk_spectral_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->spec = &spec;
    data->index = i;
    tk_thread_range(i, pool->n_threads, uids->n, &data->ifirst, &data->ilast);
  }

  tk_threads_signal(pool, TK_SPECTRAL_SCALE, 0);

  openblas_set_num_threads((int) pool->n_threads);
  primme_params params;
  primme_initialize(&params);
  params.n = (int64_t) uids->n;
  params.numEvals = spec.n_evals;
  params.matrixMatvec = tk_spectral_matvec;
  params.matrix = &spec;
  params.eps = eps_primme;
  params.printLevel = 0;
  params.target = primme_smallest;
  params.maxBlockSize = 16;
  primme_set_method(PRIMME_DEFAULT_MIN_TIME, &params);

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
  tk_ivec_t *adj_neighbors = tk_ivec_peek(L, -1, "neighbors");

  lua_getfield(L, 1, "weights");
  tk_dvec_t *adj_weights = tk_dvec_peek(L, -1, "weights");

  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "spectral", "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 1, "spectral", "threads");
  const char *type_str = tk_lua_foptstring(L, 1, "spectral", "type", "random");
  tk_laplacian_type_t laplacian_type = TK_LAPLACIAN_RANDOM;
  if (strcmp(type_str, "unnormalized") == 0) {
    laplacian_type = TK_LAPLACIAN_UNNORMALIZED;
  } else if (strcmp(type_str, "normalized") == 0) {
    laplacian_type = TK_LAPLACIAN_NORMALIZED;
  } else if (strcmp(type_str, "random") == 0) {
    laplacian_type = TK_LAPLACIAN_RANDOM;
  }
  double eps_primme = tk_lua_foptnumber(L, 1, "spectral", "eps_primme", 1e-12);
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
  tm_run_spectral(L, pool, z, scale, degree, uids, adj_offset, adj_neighbors,
                  adj_weights, n_hidden, eps_primme, laplacian_type, i_each);
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
