#include <santoku/tsetlin/graph.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
// #include <assert.h>
#include <primme.h>
#include <cblas.h>
#include <omp.h>

#define TK_SPECTRAL_MT "tk_spectral_t"
#define TK_SPECTRAL_EPH "tk_spectral_eph"

typedef enum {
  TK_LAPLACIAN_UNNORMALIZED,
  TK_LAPLACIAN_NORMALIZED,
  TK_LAPLACIAN_RANDOM
} tk_laplacian_type_t;

typedef enum {
  TK_PRECOND_OFF,
  TK_PRECOND_DIAG,
  TK_PRECOND_IC,
  TK_PRECOND_POLY
} tk_precond_type_t;

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
  int use_ic;
  double *ic_L;
  int64_t *ic_offset;
  int64_t *ic_neighbors;
  double *ic_work;
  int ic_max_threads;
  int use_poly;
  int poly_degree;
  double poly_lambda_min;
  double poly_lambda_max;
  double *poly_work;
  double *evals_copy;
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
  const uint64_t n = spec->n_nodes;

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n; i++) {
    const double inv_deg = 1.0 / degree[i];
    for (int b = 0; b < *blockSize; b++) {
      const double * restrict xb = xvec + (size_t) b * (size_t) *ldx;
      double * restrict yb = yvec + (size_t) b * (size_t) *ldy;
      yb[i] = xb[i] * inv_deg;
    }
  }

  *ierr = 0;
}

static inline int tk_spectral_gc(lua_State *L) {
  tk_spectral_t *spec = luaL_checkudata(L, 1, TK_SPECTRAL_MT);
  if (spec->ic_L) {
    free(spec->ic_L);
    spec->ic_L = NULL;
  }
  if (spec->ic_offset) {
    free(spec->ic_offset);
    spec->ic_offset = NULL;
  }
  if (spec->ic_neighbors) {
    free(spec->ic_neighbors);
    spec->ic_neighbors = NULL;
  }
  if (spec->ic_work) {
    free(spec->ic_work);
    spec->ic_work = NULL;
  }
  if (spec->poly_work) {
    free(spec->poly_work);
    spec->poly_work = NULL;
  }
  if (spec->evals_copy) {
    free(spec->evals_copy);
    spec->evals_copy = NULL;
  }
  if (spec->evals) {
    free(spec->evals);
    spec->evals = NULL;
  }
  if (spec->evecs) {
    free(spec->evecs);
    spec->evecs = NULL;
  }
  if (spec->resNorms) {
    free(spec->resNorms);
    spec->resNorms = NULL;
  }
  return 0;
}

static inline int tk_spectral_compute_ic (
  lua_State *L,
  tk_spectral_t *spec
) {
  const uint64_t n = spec->n_nodes;
  const int64_t * restrict adj_offset = spec->adj_offset->a;
  const int64_t * restrict adj_neighbors = spec->adj_neighbors->a;
  const double * restrict adj_weights = spec->adj_weights->a;
  const double * restrict degree = spec->degree->a;
  uint64_t nnz_lower = 0;
  for (uint64_t i = 0; i < n; i++) {
    const int64_t edge_start = adj_offset[i];
    const int64_t edge_end = adj_offset[i + 1];
    for (int64_t e = edge_start; e < edge_end; e++) {
      if (adj_neighbors[e] < (int64_t)i) {
        nnz_lower++;
      }
    }
  }
  uint64_t total_entries = nnz_lower + n;
  spec->ic_offset = tk_malloc(L, (n + 1) * sizeof(int64_t));
  spec->ic_neighbors = tk_malloc(L, total_entries * sizeof(int64_t));
  spec->ic_L = tk_malloc(L, total_entries * sizeof(double));
  uint64_t current_nnz = 0;
  for (uint64_t i = 0; i < n; i++) {
    spec->ic_offset[i] = (int64_t) current_nnz;
    const int64_t edge_start = adj_offset[i];
    const int64_t edge_end = adj_offset[i + 1];
    for (int64_t e = edge_start; e < edge_end; e++) {
      int64_t j = adj_neighbors[e];
      if (j < (int64_t)i) {
        spec->ic_neighbors[current_nnz] = j;
        spec->ic_L[current_nnz] = -adj_weights[e];
        current_nnz++;
      }
    }
    spec->ic_neighbors[current_nnz] = (int64_t) i;
    spec->ic_L[current_nnz] = degree[i];
    current_nnz++;
  }
  spec->ic_offset[n] = (int64_t) current_nnz;
  for (uint64_t j = 0; j < n; j++) {
    const int64_t j_start = spec->ic_offset[j];
    const int64_t j_end = spec->ic_offset[j + 1];
    const int64_t j_diag = j_end - 1;
    for (int64_t j_pos = j_start; j_pos < j_diag; j_pos++) {
      int64_t i = spec->ic_neighbors[j_pos];
      const int64_t i_start = spec->ic_offset[i];
      const int64_t i_end = spec->ic_offset[i + 1] - 1;
      int64_t j_k = j_start;
      int64_t i_k = i_start;
      while (j_k < j_pos && i_k < i_end) {
        int64_t j_col = spec->ic_neighbors[j_k];
        int64_t i_col = spec->ic_neighbors[i_k];
        if (j_col == i_col) {
          spec->ic_L[j_pos] -= spec->ic_L[j_k] * spec->ic_L[i_k];
          j_k++;
          i_k++;
        } else if (j_col < i_col) {
          j_k++;
        } else {
          i_k++;
        }
      }
      const int64_t i_diag = spec->ic_offset[i + 1] - 1;
      spec->ic_L[j_pos] /= spec->ic_L[i_diag];
    }
    double diag_val = spec->ic_L[j_diag];
    for (int64_t pos = j_start; pos < j_diag; pos++) {
      double val = spec->ic_L[pos];
      diag_val -= val * val;
    }
    if (diag_val <= 1e-14) {
      diag_val = 1e-8;
    }
    spec->ic_L[j_diag] = sqrt(diag_val);
  }
  int max_threads = omp_get_max_threads();
  spec->ic_max_threads = max_threads;
  spec->ic_work = tk_malloc(L, (size_t) n * (size_t) max_threads * sizeof(double));
  return 0;
}

static inline void tk_spectral_ic (
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
  const uint64_t n = spec->n_nodes;
  const int64_t * restrict ic_offset = spec->ic_offset;
  const int64_t * restrict ic_neighbors = spec->ic_neighbors;
  const double * restrict ic_L = spec->ic_L;
  double * restrict work = spec->ic_work;
  #pragma omp parallel for schedule(static)
  for (int b = 0; b < *blockSize; b++) {
    const double * restrict xb = xvec + (size_t) b * (size_t) *ldx;
    double * restrict yb = yvec + (size_t) b * (size_t) *ldy;
    int tid = omp_get_thread_num();
    double *z = work + (size_t) tid * n;
    for (uint64_t i = 0; i < n; i++) {
      const int64_t row_start = ic_offset[i];
      const int64_t row_end = ic_offset[i + 1];
      const int64_t diag_pos = row_end - 1;
      double sum = xb[i];
      for (int64_t pos = row_start; pos < diag_pos; pos++) {
        int64_t j = ic_neighbors[pos];
        sum -= ic_L[pos] * z[j];
      }
      z[i] = sum / ic_L[diag_pos];
    }
    memcpy(yb, z, n * sizeof(double));
    for (int64_t i = (int64_t)n - 1; i >= 0; i--) {
      const int64_t row_start = ic_offset[i];
      const int64_t row_end = ic_offset[i + 1];
      const int64_t diag_pos = row_end - 1;
      yb[i] /= ic_L[diag_pos];
      double yi = yb[i];
      for (int64_t pos = row_start; pos < diag_pos; pos++) {
        int64_t j = ic_neighbors[pos];
        yb[j] -= ic_L[pos] * yi;
      }
    }
  }
  *ierr = 0;
}

static inline int tk_spectral_compute_poly (
  lua_State *L,
  tk_spectral_t *spec
) {
  const uint64_t n = spec->n_nodes;
  const int64_t * restrict adj_offset = spec->adj_offset->a;
  const int64_t * restrict adj_neighbors = spec->adj_neighbors->a;
  const double * restrict adj_weights = spec->adj_weights->a;
  const double * restrict degree = spec->degree->a;
  double *v = malloc(n * sizeof(double));
  double *Av = malloc(n * sizeof(double));
  if (!v || !Av) {
    if (v) free(v);
    if (Av) free(Av);
    return -1;
  }
  for (uint64_t i = 0; i < n; i++) {
    v[i] = ((double)rand() / RAND_MAX) - 0.5;
  }
  double norm = cblas_dnrm2(n, v, 1);
  cblas_dscal(n, 1.0 / norm, v, 1);
  double lambda_max = 0.0;
  for (int iter = 0; iter < 10; iter++) {
    for (uint64_t i = 0; i < n; i++) {
      double accum = degree[i] * v[i];
      const int64_t edge_start = adj_offset[i];
      const int64_t edge_end = adj_offset[i + 1];
      for (int64_t e = edge_start; e < edge_end; e++) {
        const int64_t dst = adj_neighbors[e];
        const double weight = adj_weights[e];
        accum -= weight * v[dst];
      }
      Av[i] = accum;
    }
    lambda_max = cblas_ddot(n, v, 1, Av, 1);
    norm = cblas_dnrm2(n, Av, 1);
    if (norm > 0.0) {
      memcpy(v, Av, n * sizeof(double));
      cblas_dscal(n, 1.0 / norm, v, 1);
    }
  }
  free(v);
  free(Av);
  double lambda_min = 0.0;
  for (uint64_t i = 0; i < n; i++) {
    double row_sum = 0.0;
    const int64_t edge_start = adj_offset[i];
    const int64_t edge_end = adj_offset[i + 1];
    for (int64_t e = edge_start; e < edge_end; e++) {
      row_sum += adj_weights[e];
    }
    double center = degree[i];
    double radius = row_sum;
    double lower_bound = fmax(0.0, center - radius);
    if (i == 0 || lower_bound < lambda_min) {
      lambda_min = lower_bound;
    }
  }
  lambda_min = fmax(lambda_min, lambda_max * 1e-6);
  spec->poly_lambda_min = lambda_min;
  spec->poly_lambda_max = lambda_max;
  spec->poly_work = tk_malloc(L, (size_t) n * 2 * sizeof(double));
  return 0;
}

static inline void tk_spectral_poly (
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
  const uint64_t n = spec->n_nodes;
  const int degree = spec->poly_degree;
  const double lambda_min = spec->poly_lambda_min;
  const double lambda_max = spec->poly_lambda_max;
  const int64_t * restrict adj_offset = spec->adj_offset->a;
  const int64_t * restrict adj_neighbors = spec->adj_neighbors->a;
  const double * restrict adj_weights = spec->adj_weights->a;
  const double * restrict degree_arr = spec->degree->a;
  double * restrict work = spec->poly_work;
  double center = (lambda_max + lambda_min) / 2.0;
  double radius = (lambda_max - lambda_min) / 2.0;
  double alpha = 1.0 / radius;
  for (int b = 0; b < *blockSize; b++) {
    const double * restrict xb = xvec + (size_t) b * (size_t) *ldx;
    double * restrict yb = yvec + (size_t) b * (size_t) *ldy;
    double *w0 = work;
    double *w1 = work + n;
    memcpy(w0, xb, n * sizeof(double));
    memset(yb, 0, n * sizeof(double));
    double c0 = 1.0 / lambda_max;
    cblas_daxpy(n, c0, w0, 1, yb, 1);
    if (degree > 0) {
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n; i++) {
        double accum = degree_arr[i] * w0[i];
        const int64_t edge_start = adj_offset[i];
        const int64_t edge_end = adj_offset[i + 1];
        for (int64_t e = edge_start; e < edge_end; e++) {
          const int64_t dst = adj_neighbors[e];
          const double weight = adj_weights[e];
          accum -= weight * w0[dst];
        }
        w1[i] = (accum - center * w0[i]) * alpha;
      }
      double c1 = 2.0 / lambda_max;
      cblas_daxpy(n, c1, w1, 1, yb, 1);
      for (int k = 1; k < degree; k++) {
        double *w_prev = w0;
        double *w_curr = w1;
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n; i++) {
          double accum = degree_arr[i] * w_curr[i];
          const int64_t edge_start = adj_offset[i];
          const int64_t edge_end = adj_offset[i + 1];
          for (int64_t e = edge_start; e < edge_end; e++) {
            const int64_t dst = adj_neighbors[e];
            const double weight = adj_weights[e];
            accum -= weight * w_curr[dst];
          }
          w_prev[i] = 2.0 * alpha * (accum - center * w_curr[i]) - w_prev[i];
        }
        double ck = 2.0 / lambda_max;
        cblas_daxpy(n, ck, w_prev, 1, yb, 1);
        double *tmp = w0;
        w0 = w1;
        w1 = tmp;
      }
    }
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
    #pragma omp parallel
    {
      for (int b = 0; b < *blockSize; b++) {
        const double * restrict xb = x + (size_t) b * (size_t) *ldx;
        double * restrict yb = y + (size_t) b * (size_t) *ldy;
        #pragma omp for schedule(guided)
        for (uint64_t i = 0; i < spec->n_nodes; i++) {
          const double deg = degree[i];
          const int64_t edge_start = adj_offset[i];
          const int64_t edge_end = adj_offset[i + 1];
          double accum = deg * xb[i];
          for (int64_t e = edge_start; e < edge_end; e++) {
            const int64_t dst = adj_neighbors[e];
            const double weight = adj_weights[e];
            accum -= weight * xb[dst];
          }
          yb[i] = accum;
        }
      }
    }
  } else {
    #pragma omp parallel
    {
      for (int b = 0; b < *blockSize; b++) {
        const double * restrict xb = x + (size_t) b * (size_t) *ldx;
        double * restrict yb = y + (size_t) b * (size_t) *ldy;
        #pragma omp for schedule(guided)
        for (uint64_t i = 0; i < spec->n_nodes; i++) {
          const double scale_i = scale[i];
          const int64_t edge_start = adj_offset[i];
          const int64_t edge_end = adj_offset[i + 1];
          double accum = xb[i];
          for (int64_t e = edge_start; e < edge_end; e++) {
            const int64_t dst = adj_neighbors[e];
            const double scaled_weight = scale_i * scale[dst] * adj_weights[e];
            accum -= scaled_weight * xb[dst];
          }
          yb[i] = accum;
        }
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
  primme_preset_method method,
  tk_precond_type_t precond,
  uint64_t block_size,
  int i_each
) {
  tk_spectral_t *spec = tk_lua_newuserdata(L, tk_spectral_t, TK_SPECTRAL_MT, NULL, tk_spectral_gc);
  int spec_idx = lua_gettop(L);
  spec->laplacian_type = laplacian_type;
  spec->scale = scale;
  spec->degree = degree;
  spec->adj_offset = adj_offset;
  spec->adj_neighbors = adj_neighbors;
  spec->adj_weights = adj_weights;
  spec->n_nodes = uids->n;
  spec->n_evals = n_hidden + 1;
  spec->use_ic = 0;
  spec->ic_L = NULL;
  spec->ic_offset = NULL;
  spec->ic_neighbors = NULL;
  spec->ic_work = NULL;
  spec->ic_max_threads = 0;
  spec->use_poly = 0;
  spec->poly_degree = 5;
  spec->poly_lambda_min = 0.0;
  spec->poly_lambda_max = 0.0;
  spec->poly_work = NULL;
  spec->evals = NULL;
  spec->evecs = NULL;
  spec->resNorms = NULL;
  spec->evals_copy = NULL;
  // assert(spec->n_evals >= 2);

  {
    const tk_laplacian_type_t lap_type = spec->laplacian_type;
    double * restrict deg = spec->degree->a;
    double * restrict sc = spec->scale->a;
    const int64_t * restrict offset = spec->adj_offset->a;
    const double * restrict weights = spec->adj_weights->a;
    if (lap_type == TK_LAPLACIAN_UNNORMALIZED) {
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < spec->n_nodes; i++) {
        double sum = 0.0;
        const int64_t j_start = offset[i];
        const int64_t j_end = offset[i + 1];
        for (int64_t j = j_start; j < j_end; j++)
          sum += weights[j];
        deg[i] = sum;
        sc[i] = 1.0;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < spec->n_nodes; i++) {
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
  primme_set_method(method, &params);

  if (precond == TK_PRECOND_IC) {
    spec->use_ic = 1;
    if (tk_spectral_compute_ic(L, spec) != 0) {
      tk_lua_verror(L, 2, "spectral", "IC factorization failed");
      return;
    }
    params.applyPreconditioner = tk_spectral_ic;
    params.correctionParams.precondition = 1;
  } else if (precond == TK_PRECOND_POLY) {
    spec->use_poly = 1;
    if (tk_spectral_compute_poly(L, spec) != 0) {
      tk_lua_verror(L, 2, "spectral", "Polynomial preconditioner setup failed");
      return;
    }
    params.applyPreconditioner = tk_spectral_poly;
    params.correctionParams.precondition = 1;
  } else if (precond == TK_PRECOND_DIAG) {
    params.applyPreconditioner = tk_spectral_preconditioner;
    params.correctionParams.precondition = 1;
  } else {
    params.correctionParams.precondition = 0;
  }

  params.n = (int64_t) uids->n;
  params.numEvals = spec->n_evals;
  params.matrixMatvec = tk_spectral_matvec;
  params.matrix = spec;
  params.eps = eps;
  params.printLevel = 0;
  params.target = primme_smallest;
  params.maxBlockSize = block_size;

  spec->evals = tk_malloc(L, (size_t) spec->n_evals * sizeof(double));
  spec->evecs = tk_malloc(L, (size_t) params.n * (size_t) spec->n_evals * sizeof(double));
  spec->resNorms = tk_malloc(L, (size_t) spec->n_evals * sizeof(double));
  int ret = dprimme(spec->evals, spec->evecs, spec->resNorms, &params);
  if (ret != 0) {
    primme_free(&params);
    tk_lua_verror(L, 2, "spectral", "failure calling PRIMME");
    return;
  }

  if (tk_dvec_ensure(z, uids->n * n_hidden) != 0) {
    primme_free(&params);
    tk_lua_verror(L, 2, "spectral", "allocation failed");
    return;
  }
  z->n = uids->n * n_hidden;
  double eps_drop = fmax(1e-8, 10.0 * eps);
  uint64_t start = fabs(spec->evals[0]) < eps_drop ? 1 : 0;

  tk_dvec_t *eigenvalues = tk_dvec_create(L, n_hidden, 0, 0);
  eigenvalues->n = n_hidden;

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < uids->n; i ++) {
    for (uint64_t k = 0; k < n_hidden; k ++) {
      uint64_t f = start + k;
      double eigval = spec->evecs[i + f * uids->n];
      if (laplacian_type == TK_LAPLACIAN_RANDOM)
        eigval = scale->a[i] > 0.0 ? eigval / scale->a[i] : eigval;
      z->a[i * n_hidden + k] = eigval;
      if (i == 0)
        eigenvalues->a[k] = spec->evals[f];
    }
  }

  int64_t numMatvecs = params.stats.numMatvecs;
  if (i_each != -1) {
    spec->evals_copy = tk_malloc(L, spec->n_evals * sizeof(double));
    memcpy(spec->evals_copy, spec->evals, spec->n_evals * sizeof(double));
  }

  primme_free(&params);

  if (i_each != -1) {
    for (uint64_t i = 0; i < spec->n_evals; i ++) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "eig");
      lua_pushinteger(L, (int64_t) i);
      lua_pushnumber(L, spec->evals_copy[i]);
      lua_pushboolean(L, i >= start);
      lua_call(L, 4, 0);
    }
    lua_pushvalue(L, i_each);
    lua_pushstring(L, "done");
    lua_pushinteger(L, numMatvecs);
    lua_call(L, 2, 0);
  }

  lua_remove(L, spec_idx);
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

  uint64_t block_size = tk_lua_foptunsigned(L, 1, "spectral", "block_size", 64);
  block_size = block_size ? block_size : 1;
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "spectral", "n_hidden");
  double eps = tk_lua_foptnumber(L, 1, "spectral", "eps", 1e-12);

  const char *type_str = tk_lua_foptstring(L, 1, "spectral", "type", "unnormalized");
  tk_laplacian_type_t laplacian_type = TK_LAPLACIAN_UNNORMALIZED;
  if (strcmp(type_str, "normalized") == 0) {
    laplacian_type = TK_LAPLACIAN_NORMALIZED;
  } else if (strcmp(type_str, "random") == 0) {
    laplacian_type = TK_LAPLACIAN_RANDOM;
  }

  const char *method_str = tk_lua_foptstring(L, 1, "spectral", "method", "jdqr");
  primme_preset_method method = PRIMME_JDQMR_ETol;
  if (strcmp(method_str, "gd") == 0) {
    method = PRIMME_GD_plusK;
  } else if (strcmp(method_str, "jdqmr") == 0) {
    method = PRIMME_JDQMR;
  } else if (strcmp(method_str, "lobpcg") == 0) {
    method = PRIMME_LOBPCG_OrthoBasis_Window;
  } else if (strcmp(method_str, "jdqr") == 0) {
    method = PRIMME_JDQMR_ETol;
  }

  const char *precond_str = NULL;
  if (tk_lua_ftype(L, 1, "precondition") != LUA_TNIL) {
    if (tk_lua_ftype(L, 1, "precondition") == LUA_TBOOLEAN) {
      precond_str = tk_lua_fcheckboolean(L, 1, "spectral", "precondition") ? "diag" : "off";
    } else {
      precond_str = tk_lua_fcheckstring(L, 1, "spectral", "precondition");
    }
  } else {
    precond_str = (laplacian_type == TK_LAPLACIAN_UNNORMALIZED) ? "diag" : "off";
  }

  if (tk_lua_ftype(L, 1, "ic") != LUA_TNIL && tk_lua_fcheckboolean(L, 1, "spectral", "ic")) {
    precond_str = "ic";
  }

  tk_precond_type_t precond = TK_PRECOND_DIAG;
  if (strcmp(precond_str, "off") == 0 || strcmp(precond_str, "false") == 0) {
    precond = TK_PRECOND_OFF;
  } else if (strcmp(precond_str, "diag") == 0) {
    precond = TK_PRECOND_DIAG;
  } else if (strcmp(precond_str, "ic") == 0) {
    precond = TK_PRECOND_IC;
  } else if (strcmp(precond_str, "poly") == 0) {
    precond = TK_PRECOND_POLY;
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

  tm_run_spectral(L, z, scale, degree, uids, adj_offset, adj_neighbors, adj_weights, n_hidden, eps, laplacian_type, method, precond, block_size, i_each);
  lua_remove(L, -2);

  // assert(tk_ivec_peekopt(L, -4) == uids);
  // assert(tk_dvec_peekopt(L, -3) == z);
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
