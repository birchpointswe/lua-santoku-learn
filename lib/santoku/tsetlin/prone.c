#include <santoku/tsetlin/graph.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>

static inline double tk_bessel_iv(int n, double x) {
  double term = pow(x / 2.0, n) / tgamma(n + 1);
  double sum = term;
  double x2_4 = x * x / 4.0;
  for (int k = 1; k <= 50 && fabs(term) > 1e-15 * fabs(sum); k++) {
    term *= x2_4 / (k * (n + k));
    sum += term;
  }
  return sum;
}

static inline void tk_prone_spmm(
  const int64_t *offsets,
  const int64_t *neighbors,
  const double *weights,
  const double *inv_degree,
  uint64_t n_nodes,
  uint64_t n_cols,
  const double *X,
  double *Y,
  double scale,
  bool use_weights
) {
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_nodes; i++) {
    const int64_t start = offsets[i];
    const int64_t end = offsets[i + 1];
    const double inv_di = inv_degree[i];
    double *Yi = Y + i * n_cols;
    for (uint64_t k = 0; k < n_cols; k++)
      Yi[k] = 0.0;
    if (use_weights) {
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        const double w = weights[e] * inv_di * scale;
        const double *Xj = X + j * n_cols;
        for (uint64_t k = 0; k < n_cols; k++)
          Yi[k] += w * Xj[k];
      }
    } else {
      const double w = inv_di * scale;
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        const double *Xj = X + j * n_cols;
        for (uint64_t k = 0; k < n_cols; k++)
          Yi[k] += w * Xj[k];
      }
    }
  }
}

static inline void tk_prone_laplacian_mult(
  const int64_t *offsets,
  const int64_t *neighbors,
  const double *weights,
  const double *inv_sqrt_degree,
  uint64_t n_nodes,
  uint64_t n_cols,
  const double *X,
  double *Y,
  double mu,
  bool use_weights
) {
  const double diag_factor = 1.0 - mu;
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_nodes; i++) {
    const int64_t start = offsets[i];
    const int64_t end = offsets[i + 1];
    const double inv_sqrt_di = inv_sqrt_degree[i];
    const double *Xi = X + i * n_cols;
    double *Yi = Y + i * n_cols;
    for (uint64_t k = 0; k < n_cols; k++)
      Yi[k] = diag_factor * Xi[k];
    if (use_weights) {
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        const double w = weights[e] * inv_sqrt_di * inv_sqrt_degree[j];
        const double *Xj = X + j * n_cols;
        for (uint64_t k = 0; k < n_cols; k++)
          Yi[k] -= w * Xj[k];
      }
    } else {
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        const double w = inv_sqrt_di * inv_sqrt_degree[j];
        const double *Xj = X + j * n_cols;
        for (uint64_t k = 0; k < n_cols; k++)
          Yi[k] -= w * Xj[k];
      }
    }
  }
}

static inline void tk_prone_adjacency_mult(
  const int64_t *offsets,
  const int64_t *neighbors,
  const double *weights,
  uint64_t n_nodes,
  uint64_t n_cols,
  const double *X,
  double *Y,
  bool use_weights
) {
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_nodes; i++) {
    const int64_t start = offsets[i];
    const int64_t end = offsets[i + 1];
    double *Yi = Y + i * n_cols;
    for (uint64_t k = 0; k < n_cols; k++)
      Yi[k] = 0.0;
    if (use_weights) {
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        const double w = weights[e];
        const double *Xj = X + j * n_cols;
        for (uint64_t k = 0; k < n_cols; k++)
          Yi[k] += w * Xj[k];
      }
    } else {
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        const double *Xj = X + j * n_cols;
        for (uint64_t k = 0; k < n_cols; k++)
          Yi[k] += Xj[k];
      }
    }
  }
}

static inline void tk_prone_smf(
  lua_State *L,
  const int64_t *offsets,
  const int64_t *neighbors,
  const double *weights,
  const double *degree,
  const double *inv_degree,
  double vol,
  uint64_t n_nodes,
  uint64_t n_hidden,
  uint64_t n_iter,
  uint64_t neg_samples,
  double *embeddings,
  int i_each,
  bool use_weights
) {
  double log_neg = log((double)neg_samples);
  double log_vol = log(vol);
  double *Omega = tk_malloc(L, n_nodes * n_hidden * sizeof(double));
  double *Y = tk_malloc(L, n_nodes * n_hidden * sizeof(double));
  double *Q = tk_malloc(L, n_nodes * n_hidden * sizeof(double));
  double *B = tk_malloc(L, n_hidden * n_hidden * sizeof(double));
  double *U_B = tk_malloc(L, n_hidden * n_hidden * sizeof(double));
  double *VT = tk_malloc(L, n_hidden * n_hidden * sizeof(double));
  double *S = tk_malloc(L, n_hidden * sizeof(double));
  double *superb = tk_malloc(L, (n_hidden - 1) * sizeof(double));
  double *tau = tk_malloc(L, n_hidden * sizeof(double));
  double *log_degree = tk_malloc(L, n_nodes * sizeof(double));

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_nodes; i++)
    log_degree[i] = log(degree[i]);

  #pragma omp parallel
  {
    uint64_t seed = (uint64_t)omp_get_thread_num() * 12345 + 67890;
    #pragma omp for
    for (uint64_t i = 0; i < n_nodes * n_hidden; i++) {
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      Omega[i] = ((double)(seed >> 11) / (double)(1ULL << 53)) * 2.0 - 1.0;
    }
  }

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_nodes; i++) {
    const int64_t start = offsets[i];
    const int64_t end = offsets[i + 1];
    const double log_inv_di = -log_degree[i];
    double *Yi = Y + i * n_hidden;
    for (uint64_t k = 0; k < n_hidden; k++)
      Yi[k] = 0.0;
    if (use_weights) {
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        double f_ij = log_vol + log(weights[e]) + log_inv_di - log_degree[j] - log_neg;
        if (f_ij > 0.0) {
          const double *Oj = Omega + j * n_hidden;
          for (uint64_t k = 0; k < n_hidden; k++)
            Yi[k] += f_ij * Oj[k];
        }
      }
    } else {
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        double f_ij = log_vol + log_inv_di - log_degree[j] - log_neg;
        if (f_ij > 0.0) {
          const double *Oj = Omega + j * n_hidden;
          for (uint64_t k = 0; k < n_hidden; k++)
            Yi[k] += f_ij * Oj[k];
        }
      }
    }
  }

  double *temp = tk_malloc(L, n_nodes * n_hidden * sizeof(double));
  for (uint64_t iter = 0; iter < n_iter; iter++) {
    #pragma omp parallel for schedule(static)
    for (uint64_t j = 0; j < n_nodes; j++) {
      const int64_t start = offsets[j];
      const int64_t end = offsets[j + 1];
      const double log_inv_dj = -log_degree[j];
      double *Tj = temp + j * n_hidden;
      for (uint64_t k = 0; k < n_hidden; k++)
        Tj[k] = 0.0;
      if (use_weights) {
        for (int64_t e = start; e < end; e++) {
          const int64_t ii = neighbors[e];
          double f_ji = log_vol + log(weights[e]) + log_inv_dj - log_degree[ii] - log_neg;
          if (f_ji > 0.0) {
            const double *Yii = Y + ii * n_hidden;
            for (uint64_t k = 0; k < n_hidden; k++)
              Tj[k] += f_ji * Yii[k];
          }
        }
      } else {
        for (int64_t e = start; e < end; e++) {
          const int64_t ii = neighbors[e];
          double f_ji = log_vol + log_inv_dj - log_degree[ii] - log_neg;
          if (f_ji > 0.0) {
            const double *Yii = Y + ii * n_hidden;
            for (uint64_t k = 0; k < n_hidden; k++)
              Tj[k] += f_ji * Yii[k];
          }
        }
      }
    }

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_nodes; i++) {
      const int64_t start = offsets[i];
      const int64_t end = offsets[i + 1];
      const double log_inv_di = -log_degree[i];
      double *Yi = Y + i * n_hidden;
      for (uint64_t k = 0; k < n_hidden; k++)
        Yi[k] = 0.0;
      if (use_weights) {
        for (int64_t e = start; e < end; e++) {
          const int64_t j = neighbors[e];
          double f_ij = log_vol + log(weights[e]) + log_inv_di - log_degree[j] - log_neg;
          if (f_ij > 0.0) {
            const double *Tj = temp + j * n_hidden;
            for (uint64_t k = 0; k < n_hidden; k++)
              Yi[k] += f_ij * Tj[k];
          }
        }
      } else {
        for (int64_t e = start; e < end; e++) {
          const int64_t j = neighbors[e];
          double f_ij = log_vol + log_inv_di - log_degree[j] - log_neg;
          if (f_ij > 0.0) {
            const double *Tj = temp + j * n_hidden;
            for (uint64_t k = 0; k < n_hidden; k++)
              Yi[k] += f_ij * Tj[k];
          }
        }
      }
    }

    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "smf_iter");
      lua_pushinteger(L, (int64_t)(iter + 1));
      lua_call(L, 2, 0);
    }
  }
  free(temp);

  int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (int)n_nodes, (int)n_hidden, Y, (int)n_hidden, tau);
  if (info != 0) {
    free(Omega); free(Y); free(Q); free(B); free(U_B); free(VT); free(S); free(superb); free(tau); free(log_degree);
    tk_lua_verror(L, 2, "prone", "QR factorization failed");
    return;
  }

  info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (int)n_nodes, (int)n_hidden, (int)n_hidden, Y, (int)n_hidden, tau);
  if (info != 0) {
    free(Omega); free(Y); free(Q); free(B); free(U_B); free(VT); free(S); free(superb); free(tau); free(log_degree);
    tk_lua_verror(L, 2, "prone", "Q extraction failed");
    return;
  }
  memcpy(Q, Y, n_nodes * n_hidden * sizeof(double));

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_nodes; i++) {
    const int64_t start = offsets[i];
    const int64_t end = offsets[i + 1];
    const double log_inv_di = -log_degree[i];
    double *Yi = Y + i * n_hidden;
    for (uint64_t k = 0; k < n_hidden; k++)
      Yi[k] = 0.0;
    if (use_weights) {
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        double f_ij = log_vol + log(weights[e]) + log_inv_di - log_degree[j] - log_neg;
        if (f_ij > 0.0) {
          const double *Qj = Q + j * n_hidden;
          for (uint64_t k = 0; k < n_hidden; k++)
            Yi[k] += f_ij * Qj[k];
        }
      }
    } else {
      for (int64_t e = start; e < end; e++) {
        const int64_t j = neighbors[e];
        double f_ij = log_vol + log_inv_di - log_degree[j] - log_neg;
        if (f_ij > 0.0) {
          const double *Qj = Q + j * n_hidden;
          for (uint64_t k = 0; k < n_hidden; k++)
            Yi[k] += f_ij * Qj[k];
        }
      }
    }
  }

  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n_hidden, n_hidden, n_nodes,
              1.0, Q, n_hidden, Y, n_hidden, 0.0, B, n_hidden);

  info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', (int)n_hidden, (int)n_hidden,
                        B, (int)n_hidden, S, U_B, (int)n_hidden, VT, (int)n_hidden, superb);
  if (info != 0) {
    free(Omega); free(Y); free(Q); free(B); free(U_B); free(VT); free(S); free(superb); free(tau); free(log_degree);
    tk_lua_verror(L, 2, "prone", "SVD failed");
    return;
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_nodes, n_hidden, n_hidden,
              1.0, Q, n_hidden, U_B, n_hidden, 0.0, embeddings, n_hidden);

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_nodes; i++) {
    double norm = 0.0;
    for (uint64_t k = 0; k < n_hidden; k++) {
      double val = embeddings[i * n_hidden + k] * sqrt(fmax(S[k], 0.0));
      embeddings[i * n_hidden + k] = val;
      norm += val * val;
    }
    norm = sqrt(norm);
    if (norm > 1e-10) {
      for (uint64_t k = 0; k < n_hidden; k++)
        embeddings[i * n_hidden + k] /= norm;
    }
  }

  free(Omega);
  free(Y);
  free(Q);
  free(B);
  free(U_B);
  free(VT);
  free(S);
  free(superb);
  free(tau);
  free(log_degree);
}

static inline void tk_prone_propagate(
  lua_State *L,
  const int64_t *offsets,
  const int64_t *neighbors,
  const double *weights,
  const double *inv_sqrt_degree,
  uint64_t n_nodes,
  uint64_t n_hidden,
  uint64_t step,
  double mu,
  double theta,
  double *embeddings,
  int i_each,
  bool use_weights
) {
  double *coeffs = tk_malloc(L, (step + 1) * sizeof(double));
  double bessel_sum = 0.0;
  for (uint64_t k = 0; k <= step; k++) {
    coeffs[k] = tk_bessel_iv((int)k, theta);
    bessel_sum += (k == 0 ? 1.0 : 2.0) * coeffs[k];
  }
  for (uint64_t k = 0; k <= step; k++)
    coeffs[k] /= bessel_sum;

  double *T_prev = tk_malloc(L, n_nodes * n_hidden * sizeof(double));
  double *T_curr = tk_malloc(L, n_nodes * n_hidden * sizeof(double));
  double *T_next = tk_malloc(L, n_nodes * n_hidden * sizeof(double));
  double *conv = tk_malloc(L, n_nodes * n_hidden * sizeof(double));

  memcpy(T_prev, embeddings, n_nodes * n_hidden * sizeof(double));
  memset(conv, 0, n_nodes * n_hidden * sizeof(double));

  #pragma omp parallel for
  for (uint64_t idx = 0; idx < n_nodes * n_hidden; idx++)
    conv[idx] = coeffs[0] * T_prev[idx];

  if (step >= 1) {
    tk_prone_laplacian_mult(offsets, neighbors, weights, inv_sqrt_degree,
                            n_nodes, n_hidden, T_prev, T_curr, mu, use_weights);
    #pragma omp parallel for
    for (uint64_t idx = 0; idx < n_nodes * n_hidden; idx++)
      conv[idx] += coeffs[1] * T_curr[idx];
  }

  for (uint64_t k = 2; k <= step; k++) {
    tk_prone_laplacian_mult(offsets, neighbors, weights, inv_sqrt_degree,
                            n_nodes, n_hidden, T_curr, T_next, mu, use_weights);
    #pragma omp parallel for
    for (uint64_t idx = 0; idx < n_nodes * n_hidden; idx++) {
      T_next[idx] = 2.0 * T_next[idx] - T_prev[idx];
      conv[idx] += coeffs[k] * T_next[idx];
    }

    double *tmp = T_prev;
    T_prev = T_curr;
    T_curr = T_next;
    T_next = tmp;

    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "prop_iter");
      lua_pushinteger(L, (int64_t)k);
      lua_call(L, 2, 0);
    }
  }

  #pragma omp parallel for
  for (uint64_t idx = 0; idx < n_nodes * n_hidden; idx++)
    T_prev[idx] = embeddings[idx] - conv[idx];

  tk_prone_adjacency_mult(offsets, neighbors, weights, n_nodes, n_hidden, T_prev, embeddings, use_weights);

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_nodes; i++) {
    double norm = 0.0;
    for (uint64_t k = 0; k < n_hidden; k++) {
      double val = embeddings[i * n_hidden + k];
      norm += val * val;
    }
    norm = sqrt(norm);
    if (norm > 1e-10) {
      for (uint64_t k = 0; k < n_hidden; k++)
        embeddings[i * n_hidden + k] /= norm;
    }
  }

  free(coeffs);
  free(T_prev);
  free(T_curr);
  free(T_next);
  free(conv);
}

static inline int tm_encode(lua_State *L)
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
  bool unweighted = lua_isnil(L, -1);
  tk_dvec_t *adj_weights = NULL;
  if (!unweighted)
    adj_weights = tk_dvec_peek(L, -1, "weights");

  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "prone", "n_hidden");
  uint64_t n_iter = tk_lua_foptunsigned(L, 1, "prone", "n_iter", 5);
  uint64_t step = tk_lua_foptunsigned(L, 1, "prone", "step", 10);
  double mu = tk_lua_foptnumber(L, 1, "prone", "mu", 0.2);
  double theta = tk_lua_foptnumber(L, 1, "prone", "theta", 0.5);
  uint64_t neg_samples = tk_lua_foptunsigned(L, 1, "prone", "neg_samples", 5);
  bool propagate = tk_lua_foptboolean(L, 1, "prone", "propagate", true);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  uint64_t n_nodes = uids->n;

  tk_dvec_t *z = tk_dvec_create(L, n_nodes * n_hidden, 0, 0);
  z->n = n_nodes * n_hidden;
  int i_z = lua_gettop(L);

  double *degree = tk_malloc(L, n_nodes * sizeof(double));
  double *inv_degree = tk_malloc(L, n_nodes * sizeof(double));
  double *inv_sqrt_degree = tk_malloc(L, n_nodes * sizeof(double));

  double vol = 0.0;
  if (unweighted) {
    #pragma omp parallel for reduction(+:vol)
    for (uint64_t i = 0; i < n_nodes; i++) {
      double d = (double)(adj_offset->a[i + 1] - adj_offset->a[i]);
      degree[i] = d;
      inv_degree[i] = d > 0 ? 1.0 / d : 0.0;
      inv_sqrt_degree[i] = d > 0 ? 1.0 / sqrt(d) : 0.0;
      vol += d;
    }
  } else {
    #pragma omp parallel for reduction(+:vol)
    for (uint64_t i = 0; i < n_nodes; i++) {
      double d = 0.0;
      for (int64_t e = adj_offset->a[i]; e < adj_offset->a[i + 1]; e++)
        d += adj_weights->a[e];
      degree[i] = d;
      inv_degree[i] = d > 0 ? 1.0 / d : 0.0;
      inv_sqrt_degree[i] = d > 0 ? 1.0 / sqrt(d) : 0.0;
      vol += d;
    }
  }

  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushstring(L, "start");
    lua_pushinteger(L, (int64_t)n_nodes);
    lua_pushinteger(L, (int64_t)n_hidden);
    lua_pushnumber(L, vol);
    lua_call(L, 4, 0);
  }

  tk_prone_smf(L, adj_offset->a, adj_neighbors->a,
               unweighted ? NULL : adj_weights->a,
               degree, inv_degree, vol,
               n_nodes, n_hidden, n_iter, neg_samples,
               z->a, i_each, !unweighted);

  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushstring(L, "smf_done");
    lua_call(L, 1, 0);
  }

  if (propagate) {
    tk_prone_propagate(L, adj_offset->a, adj_neighbors->a,
                       unweighted ? NULL : adj_weights->a,
                       inv_sqrt_degree,
                       n_nodes, n_hidden, step, mu, theta,
                       z->a, i_each, !unweighted);

    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "prop_done");
      lua_call(L, 1, 0);
    }
  }

  free(degree);
  free(inv_degree);
  free(inv_sqrt_degree);

  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushstring(L, "done");
    lua_call(L, 1, 0);
  }

  lua_pushvalue(L, i_uids);
  lua_replace(L, 1);

  lua_pushvalue(L, i_z);
  lua_replace(L, 2);

  lua_settop(L, 2);
  return 2;
}

static luaL_Reg tm_prone_fns[] =
{
  { "encode", tm_encode },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_prone(lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_prone_fns, 0);
  return 1;
}
