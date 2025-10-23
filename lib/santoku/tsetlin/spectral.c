#include <santoku/tsetlin/graph.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <primme.h>
#include <cblas.h>
#include <omp.h>

#define TK_SPECTRAL_BLOCKSIZE 64
#define TK_SPECTRAL_EPH "tk_spectral_eph"

typedef enum {
  TK_LAPLACIAN_UNNORMALIZED,
  TK_LAPLACIAN_NORMALIZED,
  TK_LAPLACIAN_RANDOM
} tk_laplacian_type_t;

typedef struct {
  double *x, *y;
  double *precond_x, *precond_y;
  double *evals, *evecs, *resNorms;
  tk_dvec_t *scale;
  tk_dvec_t *degree;
  tk_ivec_t *adj_offset;
  tk_ivec_t *adj_neighbors;
  tk_dvec_t *adj_weights;
  tk_laplacian_type_t laplacian_type;
  uint64_t n_nodes;
  uint64_t n_evals;
  int blockSize;
  PRIMME_INT ldx;
  PRIMME_INT ldy;
  PRIMME_INT precond_ldx;
  PRIMME_INT precond_ldy;
} tk_spectral_t;

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
  const double * restrict degree = spec->degree->a;
  #pragma omp parallel for schedule(static)
  for (int b = 0; b < *blockSize; b++) {
    const double * restrict xb = xvec + (size_t) b * (size_t) *ldx;
    double * restrict yb = yvec + (size_t) b * (size_t) *ldy;
    for (uint64_t i = 0; i < spec->n_nodes; i++)
      yb[i] = xb[i] / degree[i];
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
  const tk_laplacian_type_t laplacian_type = spec->laplacian_type;
  const int64_t * restrict adj_offset = spec->adj_offset->a;
  const int64_t * restrict adj_neighbors = spec->adj_neighbors->a;
  const double * restrict adj_weights = spec->adj_weights->a;
  const double * restrict degree = spec->degree->a;
  const double * restrict scale = spec->scale->a;
  if (laplacian_type == TK_LAPLACIAN_UNNORMALIZED) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < *blockSize; b++) {
      double * restrict yb = y + (size_t) b * (size_t) *ldy;
      const double * restrict xb = x + (size_t) b * (size_t) *ldx;
      for (uint64_t i = 0; i < spec->n_nodes; i++) {
        double accum = degree[i] * xb[i];
        const int64_t edge_start = adj_offset[i];
        const int64_t edge_end = adj_offset[i + 1];
        for (int64_t e = edge_start; e < edge_end; e++) {
          const int64_t dst = adj_neighbors[e];
          const double weight = adj_weights[e];
          accum -= weight * xb[dst];
        }
        yb[i] = accum;
      }
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < *blockSize; b++) {
      double * restrict yb = y + (size_t) b * (size_t) *ldy;
      const double * restrict xb = x + (size_t) b * (size_t) *ldx;
      for (uint64_t i = 0; i < spec->n_nodes; i++) {
        double accum = xb[i];
        const double scale_i = scale[i];
        const int64_t edge_start = adj_offset[i];
        const int64_t edge_end = adj_offset[i + 1];
        for (int64_t e = edge_start; e < edge_end; e++) {
          const int64_t dst = adj_neighbors[e];
          const double scaled_weight = scale_i * scale[dst] * adj_weights[e];
          accum -= scaled_weight * xb[dst];
        }
        yb[i] = accum;
      }
    }
  }
  *ierr = 0;
}


static inline void tm_run_spectral (
  lua_State *L,
  tk_dvec_t *z,
  tk_dvec_t *scale,
  tk_dvec_t *degree,
  tk_ivec_t *uids,
  tk_ivec_t *adj_offset,
  tk_ivec_t *adj_neighbors,
  tk_dvec_t *adj_weights,
  uint64_t n_hidden,
  double eps,
  tk_laplacian_type_t laplacian_type,
  const char *method_str,
  int use_precond,
  int i_each
) {
  tk_spectral_t spec;
  spec.laplacian_type = laplacian_type;
  spec.scale = scale;
  spec.degree = degree;
  spec.adj_offset = adj_offset;
  spec.adj_neighbors = adj_neighbors;
  spec.adj_weights = adj_weights;
  spec.n_nodes = uids->n;
  spec.n_evals = n_hidden + 1;
  assert(spec.n_evals >= 2);

  {
    const tk_laplacian_type_t lap_type = spec.laplacian_type;
    double * restrict deg = spec.degree->a;
    double * restrict sc = spec.scale->a;
    const int64_t * restrict offset = spec.adj_offset->a;
    const double * restrict weights = spec.adj_weights->a;

    if (lap_type == TK_LAPLACIAN_UNNORMALIZED) {
      #pragma omp parallel for
      for (uint64_t i = 0; i < spec.n_nodes; i++) {
        double sum = 0.0;
        const int64_t j_start = offset[i];
        const int64_t j_end = offset[i + 1];
        for (int64_t j = j_start; j < j_end; j++)
          sum += weights[j];
        deg[i] = sum;
        sc[i] = 1.0;
      }
    } else {
      #pragma omp parallel for
      for (uint64_t i = 0; i < spec.n_nodes; i++) {
        double sum = 0.0;
        const int64_t j_start = offset[i];
        const int64_t j_end = offset[i + 1];
        for (int64_t j = j_start; j < j_end; j++)
          sum += weights[j];
        deg[i] = sum;
        sc[i] = sum > 0.0 ? 1.0 / sqrt(sum) : 0.0;
      }
    }
  }

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

  primme_preset_method method = PRIMME_DEFAULT_MIN_TIME;
  if (method_str != NULL) {
    if (strcmp(method_str, "gd") == 0) {
      method = PRIMME_GD_plusK;
    } else if (strcmp(method_str, "jdqmr") == 0) {
      method = PRIMME_JDQMR;
    } else if (strcmp(method_str, "lobpcg") == 0) {
      method = PRIMME_LOBPCG_OrthoBasis_Window;
    } else if (strcmp(method_str, "jdqr") == 0) {
      method = PRIMME_JDQMR_ETol;
    }
  }
  primme_set_method(method, &params);
  if (use_precond && laplacian_type == TK_LAPLACIAN_UNNORMALIZED) {
    params.applyPreconditioner = tk_spectral_preconditioner;
    params.correctionParams.precondition = 1;
  } else {
    params.correctionParams.precondition = 0;
  }

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

  if (tk_dvec_ensure(z, uids->n * n_hidden) != 0) {
    primme_free(&params);
    free(spec.evals);
    free(spec.evecs);
    free(spec.resNorms);
    tk_lua_verror(L, 2, "spectral", "allocation failed");
    return;
  }
  z->n = uids->n * n_hidden;
  double eps_drop = fmax(1e-8, 10.0 * eps);
  uint64_t start = fabs(spec.evals[0]) < eps_drop ? 1 : 0;

  tk_dvec_t *eigenvalues = tk_dvec_create(L, n_hidden, 0, 0);
  eigenvalues->n = n_hidden;

  #pragma omp parallel for
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

  double *evals_copy = NULL;
  int64_t numMatvecs = params.stats.numMatvecs;
  if (i_each != -1) {
    evals_copy = tk_malloc(L, spec.n_evals * sizeof(double));
    memcpy(evals_copy, spec.evals, spec.n_evals * sizeof(double));
  }

  free(spec.evals);
  free(spec.evecs);
  free(spec.resNorms);
  primme_free(&params);

  if (i_each != -1) {
    for (uint64_t i = 0; i < spec.n_evals; i ++) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "eig");
      lua_pushinteger(L, (int64_t) i);
      lua_pushnumber(L, evals_copy[i]);
      lua_pushboolean(L, i >= start);
      lua_call(L, 4, 0);
    }
    lua_pushvalue(L, i_each);
    lua_pushstring(L, "done");
    lua_pushinteger(L, numMatvecs);
    lua_call(L, 2, 0);
    free(evals_copy);
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
  double eps = tk_lua_foptnumber(L, 1, "spectral", "eps", 1e-6);
  const char *type_str = tk_lua_foptstring(L, 1, "spectral", "type", "unnormalized");
  const char *method_str = tk_lua_foptstring(L, 1, "spectral", "method", NULL);

  int use_precond = tk_lua_ftype(L, 1, "precondition") != LUA_TNIL
    ? tk_lua_fcheckboolean(L, 1, "spectral", "precondition") : 1;

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

  lua_pushvalue(L, i_uids);
  tk_dvec_t *z = tk_dvec_create(L, 0, 0, 0);
  tk_dvec_t *scale = tk_dvec_create(L, uids->n, 0, 0);
  tk_dvec_t *degree = tk_dvec_create(L, uids->n, 0, 0);

  tm_run_spectral(L, z, scale, degree, uids, adj_offset, adj_neighbors, adj_weights, n_hidden, eps, laplacian_type, method_str, use_precond, i_each);
  lua_remove(L, -2);

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
  lua_newtable(L);
  tk_lua_register(L, tm_codebook_fns, 0);
  return 1;
}
