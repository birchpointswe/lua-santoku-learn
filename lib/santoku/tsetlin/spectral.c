#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/dvec.h>
#include <santoku/ivec.h>
#include <santoku/tsetlin/inv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>

typedef struct {
  int64_t *sid_map;
  double *residual;
  double *L_mat;
  int64_t *landmark_sids;
  int64_t *landmark_idx_map;
} tk_spectral_landmarks_ctx_t;

static inline int tk_spectral_landmarks_ctx_gc (lua_State *L) {
  tk_spectral_landmarks_ctx_t *ctx = (tk_spectral_landmarks_ctx_t *)lua_touserdata(L, 1);
  if (ctx->sid_map) { free(ctx->sid_map); ctx->sid_map = NULL; }
  if (ctx->residual) { free(ctx->residual); ctx->residual = NULL; }
  if (ctx->L_mat) { free(ctx->L_mat); ctx->L_mat = NULL; }
  if (ctx->landmark_sids) { free(ctx->landmark_sids); ctx->landmark_sids = NULL; }
  if (ctx->landmark_idx_map) { free(ctx->landmark_idx_map); ctx->landmark_idx_map = NULL; }
  return 0;
}

static inline void tk_spectral_compute_degrees (
  tk_inv_t *inv,
  tk_inv_rank_weights_t *rw,
  double *degrees
) {
  uint64_t n_ranks = inv->n_ranks;
  int64_t next_sid = inv->next_sid;
  uint64_t n_features = inv->postings->n;

  double *norm_by_rank = (double *)calloc((size_t)n_ranks * (size_t)next_sid, sizeof(double));
  double *mu = (double *)malloc(n_features * sizeof(double));
  double *deg_by_rank = (double *)calloc((size_t)n_ranks * (size_t)next_sid, sizeof(double));

  for (uint64_t fid = 0; fid < n_features; fid++) {
    tk_ivec_t *post = inv->postings->a[fid];
    int64_t rank = inv->ranks->a[fid];
    double w = inv->weights->a[fid];
    for (uint64_t p = 0; p < post->n; p++) {
      int64_t sid = post->a[p];
      if (inv->sid_to_uid->a[sid] < 0) continue;
      norm_by_rank[(size_t)rank * (size_t)next_sid + (size_t)sid] += w * w;
    }
  }

  for (int64_t sid = 0; sid < next_sid; sid++) {
    if (inv->sid_to_uid->a[sid] < 0) continue;
    for (uint64_t r = 0; r < n_ranks; r++) {
      double v = norm_by_rank[r * (size_t)next_sid + (size_t)sid];
      norm_by_rank[r * (size_t)next_sid + (size_t)sid] = sqrt(v);
    }
  }

  for (uint64_t fid = 0; fid < n_features; fid++) {
    tk_ivec_t *post = inv->postings->a[fid];
    int64_t rank = inv->ranks->a[fid];
    double w = inv->weights->a[fid];
    double sum = 0.0;
    for (uint64_t p = 0; p < post->n; p++) {
      int64_t sid = post->a[p];
      if (inv->sid_to_uid->a[sid] < 0) continue;
      double nm = norm_by_rank[(size_t)rank * (size_t)next_sid + (size_t)sid];
      if (nm > 0.0) sum += w / nm;
    }
    mu[fid] = sum;
  }

  for (uint64_t fid = 0; fid < n_features; fid++) {
    tk_ivec_t *post = inv->postings->a[fid];
    int64_t rank = inv->ranks->a[fid];
    double w = inv->weights->a[fid];
    for (uint64_t p = 0; p < post->n; p++) {
      int64_t sid = post->a[p];
      if (inv->sid_to_uid->a[sid] < 0) continue;
      double nm = norm_by_rank[(size_t)rank * (size_t)next_sid + (size_t)sid];
      if (nm > 0.0)
        deg_by_rank[(size_t)rank * (size_t)next_sid + (size_t)sid] += (w / nm) * mu[fid];
    }
  }

  for (int64_t sid = 0; sid < next_sid; sid++) {
    if (inv->sid_to_uid->a[sid] < 0) continue;
    double accum = 0.0;
    for (uint64_t r = 0; r < n_ranks; r++)
      accum += rw->weights[r] * deg_by_rank[r * (size_t)next_sid + (size_t)sid];
    degrees[sid] = (rw->total > 0.0) ? accum / rw->total : 0.0;
  }

  double min_deg = DBL_MAX;
  for (int64_t sid = 0; sid < next_sid; sid++)
    if (degrees[sid] > 0.0 && degrees[sid] < min_deg)
      min_deg = degrees[sid];
  if (min_deg < DBL_MAX) {
    double floor_val = min_deg / 100.0;
    for (int64_t sid = 0; sid < next_sid; sid++)
      if (degrees[sid] > 0.0 && degrees[sid] < floor_val)
        degrees[sid] = floor_val;
  }

  free(norm_by_rank);
  free(mu);
  free(deg_by_rank);
}

static inline void tk_spectral_sample_landmarks (
  lua_State *L,
  tk_inv_t *inv,
  uint64_t n_landmarks,
  double trace_tol,
  double decay,
  double bandwidth,
  double *degrees,
  tk_ivec_t **ids_out,
  tk_dvec_t **chol_out,
  tk_dvec_t **full_chol_out,
  tk_ivec_t **full_chol_ids_out,
  uint64_t *actual_landmarks_out,
  double *trace_ratio_out
) {
  uint64_t n_docs = 0;
  for (int64_t sid = 0; sid < inv->next_sid; sid++) {
    if (inv->sid_to_uid->a[sid] >= 0)
      n_docs++;
  }

  if (n_landmarks == 0 || n_landmarks > n_docs)
    n_landmarks = n_docs;
  if (trace_tol <= 0.0)
    trace_tol = 1e-15;
  if (n_landmarks == 0) {
    *ids_out = tk_ivec_create(L, 0, 0, 0);
    *chol_out = tk_dvec_create(L, 0, 0, 0);
    *full_chol_out = tk_dvec_create(L, 0, 0, 0);
    *full_chol_ids_out = tk_ivec_create(L, 0, 0, 0);
    *actual_landmarks_out = 0;
    *trace_ratio_out = 0.0;
    return;
  }

  tk_spectral_landmarks_ctx_t *ctx = (tk_spectral_landmarks_ctx_t *)
    lua_newuserdata(L, sizeof(tk_spectral_landmarks_ctx_t));
  memset(ctx, 0, sizeof(tk_spectral_landmarks_ctx_t));
  lua_newtable(L);
  lua_pushcfunction(L, tk_spectral_landmarks_ctx_gc);
  lua_setfield(L, -2, "__gc");
  lua_setmetatable(L, -2);
  int ctx_idx = lua_gettop(L);

  ctx->sid_map = (int64_t *)malloc(n_docs * sizeof(int64_t));
  ctx->residual = (double *)malloc(n_docs * sizeof(double));
  ctx->L_mat = (double *)malloc(n_docs * n_landmarks * sizeof(double));
  ctx->landmark_sids = (int64_t *)malloc(n_landmarks * sizeof(int64_t));
  ctx->landmark_idx_map = (int64_t *)malloc(n_landmarks * sizeof(int64_t));
  if (!ctx->sid_map || !ctx->residual || !ctx->L_mat || !ctx->landmark_sids ||
      !ctx->landmark_idx_map) {
    luaL_error(L, "sample_landmarks: out of memory");
    return;
  }

  int64_t *sid_map = ctx->sid_map;
  double *residual = ctx->residual;
  double *L_mat = ctx->L_mat;
  int64_t *landmark_sids = ctx->landmark_sids;
  int64_t *landmark_idx_map = ctx->landmark_idx_map;

  uint64_t idx = 0;
  for (int64_t sid = 0; sid < inv->next_sid; sid++) {
    if (inv->sid_to_uid->a[sid] >= 0)
      sid_map[idx++] = sid;
  }

  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, inv->n_ranks, decay);

  int64_t *ranks_arr = inv->ranks->a;
  double *weights_arr = inv->weights->a;

  memset(residual, 0, n_docs * sizeof(double));
  memset(L_mat, 0, n_docs * n_landmarks * sizeof(double));
  memset(landmark_idx_map, -1, n_landmarks * sizeof(int64_t));

  uint64_t actual_landmarks = 0;
  double initial_trace = 0.0;
  double trace = 0.0;
  uint64_t pivot_idx = 0;
  double scale = 0.0;
  double *pivot_row = NULL;
  size_t p_nbits = 0;
  int64_t *p_bits = NULL;
  bool done = false;
  #pragma omp parallel
  {
    double thr_q[TK_INV_MAX_RANKS];
    double thr_e[TK_INV_MAX_RANKS];
    double thr_i[TK_INV_MAX_RANKS];

    #pragma omp for reduction(+:initial_trace)
    for (uint64_t i = 0; i < n_docs; i++) {
      size_t i_nbits;
      int64_t *i_bits = tk_inv_sget(inv, sid_map[i], &i_nbits);
      residual[i] = tk_inv_similarity_fast(ranks_arr, weights_arr, inv->n_ranks,
                                           i_bits, i_nbits, i_bits, i_nbits,
                                           bandwidth, &rw, thr_q, thr_e, thr_i);
      if (degrees && degrees[sid_map[i]] > 0)
        residual[i] /= degrees[sid_map[i]];
      initial_trace += residual[i];
    }

    for (uint64_t j = 0; j < n_landmarks && !done; j++) {

      #pragma omp single
      {
        trace = 0.0;
        for (uint64_t i = 0; i < n_docs; i++) {
          if (residual[i] > 0.0)
            trace += residual[i];
        }

        if (trace < trace_tol * initial_trace || trace < 1e-15) {
          done = true;
        } else {
          double r = tk_fast_drand() * trace;
          pivot_idx = n_docs - 1;
          double cumsum = 0.0;
          for (uint64_t i = 0; i < n_docs; i++) {
            if (residual[i] > 0.0) {
              cumsum += residual[i];
              if (cumsum >= r) {
                pivot_idx = i;
                break;
              }
            }
          }

          double pivot_residual = residual[pivot_idx];
          if (pivot_residual < 1e-15) {
            done = true;
          } else {
            landmark_sids[actual_landmarks] = sid_map[pivot_idx];
            landmark_idx_map[actual_landmarks] = (int64_t)pivot_idx;
            scale = sqrt(pivot_residual);
            actual_landmarks++;
            p_bits = tk_inv_sget(inv, sid_map[pivot_idx], &p_nbits);
            pivot_row = &L_mat[pivot_idx * n_landmarks];
          }
        }
      }

      if (done) continue;

      #pragma omp for
      for (uint64_t i = 0; i < n_docs; i++) {
        size_t i_nbits;
        int64_t *i_bits = tk_inv_sget(inv, sid_map[i], &i_nbits);
        double kip = tk_inv_similarity_fast(ranks_arr, weights_arr, inv->n_ranks,
                                            i_bits, i_nbits, p_bits, p_nbits,
                                            bandwidth, &rw, thr_q, thr_e, thr_i);
        if (degrees) {
          double dd = degrees[sid_map[i]] * degrees[sid_map[pivot_idx]];
          if (dd > 0) kip /= sqrt(dd);
        }
        double dot = (j > 0) ? cblas_ddot((int)j, &L_mat[i * n_landmarks], 1, pivot_row, 1) : 0.0;
        L_mat[i * n_landmarks + j] = (kip - dot) / scale;
      }

      #pragma omp for
      for (uint64_t i = 0; i < n_docs; i++) {
        double lij = L_mat[i * n_landmarks + j];
        residual[i] -= lij * lij;
        if (residual[i] < 0.0)
          residual[i] = 0.0;
      }

      #pragma omp single
      {
        residual[pivot_idx] = 0.0;
      }
    }
  }

  tk_ivec_t *landmark_ids = tk_ivec_create(L, actual_landmarks, 0, 0);
  for (uint64_t i = 0; i < actual_landmarks; i++)
    landmark_ids->a[i] = inv->sid_to_uid->a[landmark_sids[i]];
  landmark_ids->n = actual_landmarks;

  tk_dvec_t *chol = tk_dvec_create(L, actual_landmarks * actual_landmarks, 0, 0);
  for (uint64_t li = 0; li < actual_landmarks; li++) {
    uint64_t doc_idx = (uint64_t)landmark_idx_map[li];
    for (uint64_t jj = 0; jj < actual_landmarks; jj++)
      chol->a[li * actual_landmarks + jj] = L_mat[doc_idx * n_landmarks + jj];
  }
  chol->n = actual_landmarks * actual_landmarks;

  tk_dvec_t *full_chol = tk_dvec_create(L, n_docs * actual_landmarks, 0, 0);
  for (uint64_t i = 0; i < n_docs; i++)
    for (uint64_t j = 0; j < actual_landmarks; j++)
      full_chol->a[i * actual_landmarks + j] = L_mat[i * n_landmarks + j];
  full_chol->n = n_docs * actual_landmarks;

  tk_ivec_t *full_chol_ids = tk_ivec_create(L, n_docs, 0, 0);
  for (uint64_t i = 0; i < n_docs; i++)
    full_chol_ids->a[i] = inv->sid_to_uid->a[sid_map[i]];
  full_chol_ids->n = n_docs;

  lua_remove(L, ctx_idx);

  *ids_out = landmark_ids;
  *chol_out = chol;
  *full_chol_out = full_chol;
  *full_chol_ids_out = full_chol_ids;
  *actual_landmarks_out = actual_landmarks;
  *trace_ratio_out = (initial_trace > 0.0) ? (trace / initial_trace) : 0.0;
}

static inline int tk_spectral_sample_landmarks_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "inv");
  tk_inv_t *inv = tk_inv_peek(L, -1);
  lua_pop(L, 1);

  uint64_t n_landmarks = tk_lua_foptunsigned(L, 1, "sample_landmarks", "n_landmarks", 0);
  double decay = tk_lua_foptnumber(L, 1, "sample_landmarks", "decay", 0.0);
  double bandwidth = tk_lua_foptnumber(L, 1, "sample_landmarks", "bandwidth", -1.0);
  double trace_tol = tk_lua_foptnumber(L, 1, "sample_landmarks", "trace_tol", 1e-15);

  tk_ivec_t *landmark_ids;
  tk_dvec_t *chol;
  tk_dvec_t *full_chol;
  tk_ivec_t *full_chol_ids;
  uint64_t actual_landmarks;
  double trace_ratio;
  tk_spectral_sample_landmarks(L, inv, n_landmarks, trace_tol, decay, bandwidth,
                               NULL,
                               &landmark_ids, &chol, &full_chol, &full_chol_ids,
                               &actual_landmarks, &trace_ratio);
  lua_pushinteger(L, (int64_t) actual_landmarks);
  lua_pushnumber(L, trace_ratio);
  return 6;
}

static inline int tm_encode (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "chol");
  tk_dvec_t *chol = tk_dvec_peek(L, -1, "chol");
  lua_pop(L, 1);

  uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "spectral", "n_samples");
  uint64_t n_landmarks = tk_lua_fcheckunsigned(L, 1, "spectral", "n_landmarks");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "spectral", "n_dims");

  if (chol->n != n_samples * n_landmarks)
    return luaL_error(L, "chol size (%llu) != n_samples * n_landmarks (%llu)",
      (unsigned long long)chol->n, (unsigned long long)(n_samples * n_landmarks));

  if (n_dims > n_landmarks)
    return luaL_error(L, "n_dims (%llu) must be <= n_landmarks (%llu)",
      (unsigned long long)n_dims, (unsigned long long)n_landmarks);

  tk_dvec_t *chol_processed = tk_dvec_create(L, n_samples * n_landmarks, 0, 0);
  chol_processed->n = n_samples * n_landmarks;
  memcpy(chol_processed->a, chol->a, n_samples * n_landmarks * sizeof(double));

  tk_dvec_t *col_means = tk_dvec_create(L, n_landmarks, 0, 0);
  col_means->n = n_landmarks;

  #pragma omp parallel for schedule(static)
  for (uint64_t j = 0; j < n_landmarks; j++) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n_samples; i++)
      sum += chol_processed->a[i * n_landmarks + j];
    double mu = sum / (double)n_samples;
    col_means->a[j] = mu;
    for (uint64_t i = 0; i < n_samples; i++)
      chol_processed->a[i * n_landmarks + j] -= mu;
  }

  tk_dvec_t *gram = tk_dvec_create(L, n_landmarks * n_landmarks, 0, 0);
  gram->n = n_landmarks * n_landmarks;

  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    (int)n_landmarks, (int)n_landmarks, (int)n_samples,
    1.0, chol_processed->a, (int)n_landmarks, chol_processed->a, (int)n_landmarks,
    0.0, gram->a, (int)n_landmarks);

  tk_dvec_t *eigenvalues_raw = tk_dvec_create(L, n_dims, 0, 0);
  double *evecs_raw = malloc(n_landmarks * n_landmarks * sizeof(double));
  int *isuppz = malloc(2 * n_dims * sizeof(int));
  int m = 0;

  int info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U',
    (int)n_landmarks, gram->a, (int)n_landmarks,
    0.0, 0.0, (int)(n_landmarks - n_dims + 1), (int)n_landmarks,
    0.0, &m, eigenvalues_raw->a, evecs_raw, (int)n_landmarks, isuppz);

  free(isuppz);

  if (info != 0) {
    free(evecs_raw);
    return luaL_error(L, "LAPACKE_dsyevr failed with info=%d", info);
  }

  tk_dvec_t *eigenvectors = tk_dvec_create(L, n_landmarks * n_dims, 0, 0);
  tk_dvec_t *eigenvalues = tk_dvec_create(L, n_dims, 0, 0);
  eigenvectors->n = n_landmarks * n_dims;
  eigenvalues->n = n_dims;

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_landmarks; i++) {
    for (uint64_t k = 0; k < n_dims; k++) {
      uint64_t col = n_dims - 1 - k;
      eigenvectors->a[i * n_dims + k] = evecs_raw[i * n_landmarks + col];
    }
  }
  for (uint64_t k = 0; k < n_dims; k++)
    eigenvalues->a[k] = eigenvalues_raw->a[n_dims - 1 - k];

  free(evecs_raw);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = lua_gettop(L);
  }

  if (i_each != -1) {
    for (uint64_t i = 0; i < n_dims; i++) {
      lua_pushvalue(L, i_each);
      lua_pushstring(L, "eig");
      lua_pushinteger(L, (int64_t)i);
      lua_pushnumber(L, eigenvalues->a[i]);
      lua_pushboolean(L, 1);
      lua_call(L, 4, 0);
    }
    lua_pushvalue(L, i_each);
    lua_pushstring(L, "done");
    lua_pushinteger(L, 0);
    lua_call(L, 2, 0);
  }

  lua_pushvalue(L, 6);
  lua_pushvalue(L, 7);
  lua_pushvalue(L, 3);
  return 3;
}

typedef struct {
  int64_t *uid_to_chol;
  int64_t *lm_sids;
  int64_t *feat_sid_map;
  double *adjustment;
  double *projection;
  double *lm_degrees;
  double *feat_degrees;
} tk_encode_nystrom_ctx_t;

static inline int tk_encode_nystrom_ctx_gc (lua_State *L) {
  tk_encode_nystrom_ctx_t *c = (tk_encode_nystrom_ctx_t *)lua_touserdata(L, 1);
  free(c->uid_to_chol);
  free(c->lm_sids);
  free(c->feat_sid_map);
  free(c->adjustment);
  free(c->projection);
  free(c->lm_degrees);
  free(c->feat_degrees);
  return 0;
}

static inline int tm_encode_nystrom (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "inv");
  tk_inv_t *feat_inv = tk_inv_peek(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "landmarks_inv");
  tk_inv_t *lm_inv = tk_inv_peekopt(L, -1);
  lua_pop(L, 1);
  if (!lm_inv) lm_inv = feat_inv;

  uint64_t n_lm_req = tk_lua_foptunsigned(L, 1, "encode_nystrom", "n_landmarks", 0);
  uint64_t n_dims_req = tk_lua_fcheckunsigned(L, 1, "encode_nystrom", "n_dims");
  double decay = tk_lua_foptnumber(L, 1, "encode_nystrom", "decay", 0.0);
  double bandwidth = tk_lua_foptnumber(L, 1, "encode_nystrom", "bandwidth", -1.0);
  double trace_tol = tk_lua_foptnumber(L, 1, "encode_nystrom", "trace_tol", 1e-15);
  bool normalized = tk_lua_foptboolean(L, 1, "encode_nystrom", "normalized", false);
  bool cholesky = tk_lua_foptboolean(L, 1, "encode_nystrom", "cholesky", false);

  double *lm_degrees = NULL;
  double *feat_degrees = NULL;
  tk_inv_rank_weights_t feat_rw;
  tk_inv_precompute_rank_weights(&feat_rw, feat_inv->n_ranks, decay);

  if (normalized) {
    if (lm_inv == feat_inv) {
      lm_degrees = (double *)calloc((size_t)lm_inv->next_sid, sizeof(double));
      tk_spectral_compute_degrees(lm_inv, &feat_rw, lm_degrees);
      feat_degrees = lm_degrees;
    } else {
      tk_inv_rank_weights_t lm_rw;
      tk_inv_precompute_rank_weights(&lm_rw, lm_inv->n_ranks, decay);
      lm_degrees = (double *)calloc((size_t)lm_inv->next_sid, sizeof(double));
      tk_spectral_compute_degrees(lm_inv, &lm_rw, lm_degrees);
      feat_degrees = (double *)calloc((size_t)feat_inv->next_sid, sizeof(double));
      tk_spectral_compute_degrees(feat_inv, &feat_rw, feat_degrees);
    }
  }

  tk_ivec_t *lm_ids, *chol_ids;
  tk_dvec_t *lm_chol, *full_chol;
  uint64_t m;
  double trace_ratio;
  tk_spectral_sample_landmarks(L, lm_inv, n_lm_req, trace_tol, decay, bandwidth,
    lm_degrees,
    &lm_ids, &lm_chol, &full_chol, &chol_ids, &m, &trace_ratio);
  int lm_ids_idx = lua_gettop(L) - 3;

  uint64_t d = n_dims_req;
  if (d > m) d = m;

  if (m == 0 || d == 0) {
    if (lm_degrees && lm_degrees != feat_degrees) free(lm_degrees);
    if (feat_degrees) free(feat_degrees);
    tk_dvec_create(L, 0, 0, 0);
    tk_ivec_create(L, 0, 0, 0);
    lua_pushinteger(L, 0);
    lua_pushvalue(L, lm_ids_idx);
    lua_pushinteger(L, 0);
    lua_pushnumber(L, 0.0);
    return 6;
  }

  tk_encode_nystrom_ctx_t *ctx = (tk_encode_nystrom_ctx_t *)
    lua_newuserdata(L, sizeof(tk_encode_nystrom_ctx_t));
  memset(ctx, 0, sizeof(*ctx));
  lua_newtable(L);
  lua_pushcfunction(L, tk_encode_nystrom_ctx_gc);
  lua_setfield(L, -2, "__gc");
  lua_setmetatable(L, -2);

  ctx->lm_degrees = (lm_inv != feat_inv) ? lm_degrees : NULL;
  ctx->feat_degrees = feat_degrees;

  tk_dvec_t *cw = tk_dvec_create(L, m * m, 0, 0);
  cw->n = m * m;
  memcpy(cw->a, lm_chol->a, m * m * sizeof(double));

  tk_dvec_t *cmeans = tk_dvec_create(L, m, 0, 0);
  cmeans->n = m;

  #pragma omp parallel for schedule(static)
  for (uint64_t j = 0; j < m; j++) {
    double s = 0.0;
    for (uint64_t i = 0; i < m; i++) s += cw->a[i * m + j];
    double mu = s / (double)m;
    cmeans->a[j] = mu;
    for (uint64_t i = 0; i < m; i++) cw->a[i * m + j] -= mu;
  }

  uint64_t nc = chol_ids->n;
  tk_dvec_t *ccodes;

  if (cholesky) {
    d = m;

    ccodes = tk_dvec_create(L, nc * d, 0, 0);
    ccodes->n = nc * d;
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < nc; i++)
      for (uint64_t j = 0; j < d; j++)
        ccodes->a[i * d + j] = full_chol->a[i * m + j] - cmeans->a[j];

    ctx->adjustment = (double *)malloc(d * sizeof(double));
    memcpy(ctx->adjustment, cmeans->a, d * sizeof(double));

    ctx->projection = (double *)malloc(m * d * sizeof(double));
    memset(ctx->projection, 0, m * d * sizeof(double));
    for (uint64_t i = 0; i < m; i++)
      ctx->projection[i * d + i] = 1.0;
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
      (int)m, (int)d, 1.0, lm_chol->a, (int)m, ctx->projection, (int)d);

  } else {
    tk_dvec_t *gram = tk_dvec_create(L, m * m, 0, 0);
    gram->n = m * m;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)m, (int)m, (int)m, 1.0, cw->a, (int)m, cw->a, (int)m, 0.0, gram->a, (int)m);

    tk_dvec_t *eig_raw = tk_dvec_create(L, d, 0, 0);
    eig_raw->n = d;
    double *ev_raw = malloc(m * m * sizeof(double));
    int *isuppz = malloc(2 * d * sizeof(int));
    if (!ev_raw || !isuppz) {
      free(ev_raw); free(isuppz);
      return luaL_error(L, "encode_nystrom: out of memory");
    }
    int n_eig = 0;
    int info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U',
      (int)m, gram->a, (int)m, 0.0, 0.0, (int)(m - d + 1), (int)m,
      0.0, &n_eig, eig_raw->a, ev_raw, (int)m, isuppz);
    free(isuppz);
    if (info != 0) {
      free(ev_raw);
      return luaL_error(L, "encode_nystrom: dsyevr info=%d", info);
    }

    tk_dvec_t *eigvecs = tk_dvec_create(L, m * d, 0, 0);
    eigvecs->n = m * d;

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < m; i++)
      for (uint64_t k = 0; k < d; k++)
        eigvecs->a[i * d + k] = ev_raw[i * m + (d - 1 - k)];
    free(ev_raw);

    ccodes = tk_dvec_create(L, nc * d, 0, 0);
    ccodes->n = nc * d;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)nc, (int)d, (int)m, 1.0, full_chol->a, (int)m, eigvecs->a, (int)d,
      0.0, ccodes->a, (int)d);

    ctx->adjustment = (double *)malloc(d * sizeof(double));
    cblas_dgemv(CblasRowMajor, CblasTrans,
      (int)m, (int)d, 1.0, eigvecs->a, (int)d, cmeans->a, 1, 0.0, ctx->adjustment, 1);

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < nc; i++) {
      double *r = ccodes->a + i * d;
      for (uint64_t j = 0; j < d; j++)
        r[j] -= ctx->adjustment[j];
    }

    ctx->projection = (double *)malloc(m * d * sizeof(double));
    memcpy(ctx->projection, eigvecs->a, m * d * sizeof(double));

    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
      (int)m, (int)d, 1.0, lm_chol->a, (int)m, ctx->projection, (int)d);
  }

  int64_t max_uid = -1;
  for (uint64_t i = 0; i < nc; i++)
    if (chol_ids->a[i] > max_uid) max_uid = chol_ids->a[i];

  uint64_t nf = 0;
  for (int64_t sid = 0; sid < feat_inv->next_sid; sid++)
    if (feat_inv->sid_to_uid->a[sid] >= 0) {
      nf++;
      if (feat_inv->sid_to_uid->a[sid] > max_uid)
        max_uid = feat_inv->sid_to_uid->a[sid];
    }

  if (nf == 0) {
    tk_dvec_create(L, 0, 0, 0);
    tk_ivec_create(L, 0, 0, 0);
    lua_pushinteger(L, (lua_Integer)d);
    lua_pushvalue(L, lm_ids_idx);
    lua_pushinteger(L, (lua_Integer)m);
    lua_pushnumber(L, trace_ratio);
    return 6;
  }

  ctx->uid_to_chol = (int64_t *)malloc((max_uid + 1) * sizeof(int64_t));
  memset(ctx->uid_to_chol, -1, (max_uid + 1) * sizeof(int64_t));
  for (uint64_t i = 0; i < nc; i++)
    ctx->uid_to_chol[chol_ids->a[i]] = (int64_t)i;

  ctx->feat_sid_map = (int64_t *)malloc(nf * sizeof(int64_t));
  uint64_t fi = 0;
  for (int64_t sid = 0; sid < feat_inv->next_sid; sid++)
    if (feat_inv->sid_to_uid->a[sid] >= 0)
      ctx->feat_sid_map[fi++] = sid;

  ctx->lm_sids = (int64_t *)malloc(m * sizeof(int64_t));
  for (uint64_t j = 0; j < m; j++)
    ctx->lm_sids[j] = tk_inv_uid_sid(feat_inv, lm_ids->a[j], TK_INV_FIND);

  tk_dvec_t *raw_codes = tk_dvec_create(L, nf * d, 0, 0);
  raw_codes->n = nf * d;
  int raw_codes_idx = lua_gettop(L);

  tk_ivec_t *out_ids = tk_ivec_create(L, nf, 0, 0);
  out_ids->n = nf;
  int out_ids_idx = lua_gettop(L);

  #pragma omp parallel
  {
    double thr_q[TK_INV_MAX_RANKS];
    double thr_e[TK_INV_MAX_RANKS];
    double thr_i[TK_INV_MAX_RANKS];
    double *sims = (double *)malloc(m * sizeof(double));

    #pragma omp for schedule(dynamic, 64)
    for (uint64_t i = 0; i < nf; i++) {
      int64_t sid = ctx->feat_sid_map[i];
      int64_t uid = feat_inv->sid_to_uid->a[sid];
      out_ids->a[i] = uid;
      int64_t cr = (uid <= max_uid) ? ctx->uid_to_chol[uid] : -1;
      if (cr >= 0) {
        memcpy(raw_codes->a + i * d, ccodes->a + cr * d, d * sizeof(double));
      } else {
        size_t qn;
        int64_t *qb = tk_inv_sget(feat_inv, sid, &qn);
        for (uint64_t j = 0; j < m; j++) {
          double sim = 0.0;
          if (ctx->lm_sids[j] >= 0 && qn > 0) {
            size_t ln;
            int64_t *lb = tk_inv_sget(feat_inv, ctx->lm_sids[j], &ln);
            if (lb && ln > 0)
              sim = tk_inv_similarity_fast(feat_inv->ranks->a, feat_inv->weights->a,
                feat_inv->n_ranks, qb, qn, lb, ln, bandwidth, &feat_rw, thr_q, thr_e, thr_i);
          }
          sims[j] = sim;
        }
        if (ctx->feat_degrees) {
          double dq = ctx->feat_degrees[sid];
          for (uint64_t j = 0; j < m; j++) {
            if (ctx->lm_sids[j] >= 0) {
              double dl = ctx->feat_degrees[ctx->lm_sids[j]];
              double dd = dq * dl;
              if (dd > 0) sims[j] /= sqrt(dd);
            }
          }
        }
        double *out = raw_codes->a + i * d;
        cblas_dgemv(CblasRowMajor, CblasTrans,
          (int)m, (int)d, 1.0, ctx->projection, (int)d, sims, 1, 0.0, out, 1);
        for (uint64_t k = 0; k < d; k++) out[k] -= ctx->adjustment[k];
      }
    }

    free(sims);
  }

  lua_pushvalue(L, raw_codes_idx);
  lua_pushvalue(L, out_ids_idx);
  lua_pushinteger(L, (lua_Integer)d);
  lua_pushvalue(L, lm_ids_idx);
  lua_pushinteger(L, (lua_Integer)m);
  lua_pushnumber(L, trace_ratio);
  return 6;
}

static luaL_Reg tm_fns[] =
{
  { "encode", tm_encode },
  { "encode_nystrom", tm_encode_nystrom },
  { "sample_landmarks", tk_spectral_sample_landmarks_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_spectral (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_fns, 0);
  return 1;
}
