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
} tk_spectral_landmarks_ctx_t;

static inline int tk_spectral_landmarks_ctx_gc (lua_State *L) {
  tk_spectral_landmarks_ctx_t *ctx = (tk_spectral_landmarks_ctx_t *)lua_touserdata(L, 1);
  if (ctx->sid_map) { free(ctx->sid_map); ctx->sid_map = NULL; }
  if (ctx->residual) { free(ctx->residual); ctx->residual = NULL; }
  if (ctx->L_mat) { free(ctx->L_mat); ctx->L_mat = NULL; }
  if (ctx->landmark_sids) { free(ctx->landmark_sids); ctx->landmark_sids = NULL; }
  return 0;
}

static inline void tk_spectral_sample_landmarks (
  lua_State *L,
  tk_inv_t *inv,
  uint64_t n_landmarks,
  double trace_tol,
  tk_ivec_sim_type_t cmp,
  double cmp_alpha,
  double cmp_beta,
  double decay,
  tk_combine_type_t combine,
  tk_ivec_t **ids_out,
  tk_ivec_t **doc_ids_out,
  tk_dvec_t **chol_out,
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
    trace_tol = 1e-12;
  if (n_landmarks == 0) {
    *ids_out = tk_ivec_create(L, 0, 0, 0);
    *doc_ids_out = tk_ivec_create(L, 0, 0, 0);
    *chol_out = tk_dvec_create(L, 0, 0, 0);
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
  if (!ctx->sid_map || !ctx->residual || !ctx->L_mat || !ctx->landmark_sids) {
    luaL_error(L, "sample_landmarks: out of memory");
    return;
  }

  int64_t *sid_map = ctx->sid_map;
  double *residual = ctx->residual;
  double *L_mat = ctx->L_mat;
  int64_t *landmark_sids = ctx->landmark_sids;

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
                                           cmp, cmp_alpha, cmp_beta,
                                           combine, &rw, thr_q, thr_e, thr_i);
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
            actual_landmarks++;
            scale = sqrt(pivot_residual);
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
                                            cmp, cmp_alpha, cmp_beta,
                                            combine, &rw, thr_q, thr_e, thr_i);
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

  tk_ivec_t *all_doc_ids = tk_ivec_create(L, n_docs, 0, 0);
  for (uint64_t i = 0; i < n_docs; i++)
    all_doc_ids->a[i] = inv->sid_to_uid->a[sid_map[i]];
  all_doc_ids->n = n_docs;

  tk_dvec_t *chol = tk_dvec_create(L, n_docs * actual_landmarks, 0, 0);
  for (uint64_t i = 0; i < n_docs; i++) {
    for (uint64_t jj = 0; jj < actual_landmarks; jj++)
      chol->a[i * actual_landmarks + jj] = L_mat[i * n_landmarks + jj];
  }
  chol->n = n_docs * actual_landmarks;

  lua_remove(L, ctx_idx);

  *ids_out = landmark_ids;
  *doc_ids_out = all_doc_ids;
  *chol_out = chol;
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
  tk_ivec_sim_type_t cmp = tk_inv_parse_cmp(tk_lua_foptstring(L, 1, "sample_landmarks", "cmp", "jaccard"));
  double cmp_alpha = tk_lua_foptnumber(L, 1, "sample_landmarks", "cmp_alpha", 0.5);
  double cmp_beta = tk_lua_foptnumber(L, 1, "sample_landmarks", "cmp_beta", 0.5);
  double decay = tk_lua_foptnumber(L, 1, "sample_landmarks", "decay", 0.0);
  tk_combine_type_t combine = tk_inv_parse_combine(tk_lua_foptstring(L, 1, "sample_landmarks", "combine", "weighted_avg"));
  double trace_tol = tk_lua_foptnumber(L, 1, "sample_landmarks", "trace_tol", 1e-12);

  tk_ivec_t *landmark_ids;
  tk_ivec_t *doc_ids;
  tk_dvec_t *chol;
  uint64_t actual_landmarks;
  double trace_ratio;
  tk_spectral_sample_landmarks(L, inv, n_landmarks, trace_tol, cmp, cmp_alpha, cmp_beta, decay, combine,
                               &landmark_ids, &doc_ids, &chol, &actual_landmarks, &trace_ratio);
  lua_pushinteger(L, (int64_t) actual_landmarks);
  lua_pushnumber(L, trace_ratio);
  return 5;
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

static luaL_Reg tm_fns[] =
{
  { "encode", tm_encode },
  { "sample_landmarks", tk_spectral_sample_landmarks_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_spectral (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_fns, 0);
  return 1;
}
