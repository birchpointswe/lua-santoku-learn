#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/dvec.h>
#include <santoku/ivec.h>
#include <santoku/learn/inv.h>
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

static inline void tk_spectral_sample_landmarks (
  lua_State *L,
  tk_inv_t *inv,
  uint64_t n_landmarks,
  double trace_tol,
  double decay,
  double bandwidth,
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

  const double *weights_arr = inv->weights->a;
  const int64_t *node_bits_a = inv->node_bits->a;
  const int64_t *nro_a = inv->node_rank_offsets->a;
  const double *node_weights_a = inv->node_weights->a;
  uint64_t n_ranks = inv->n_ranks;
  uint64_t stride = n_ranks + 1;

  memset(residual, 0, n_docs * sizeof(double));
  memset(L_mat, 0, n_docs * n_landmarks * sizeof(double));
  memset(landmark_idx_map, -1, n_landmarks * sizeof(int64_t));

  uint64_t actual_landmarks = 0;
  double initial_trace = 0.0;
  double trace = 0.0;
  uint64_t pivot_idx = 0;
  double scale = 0.0;
  double *pivot_row = NULL;
  bool done = false;
  #pragma omp parallel
  {
    double thr_i[TK_INV_MAX_RANKS];

    #pragma omp for reduction(+:initial_trace)
    for (uint64_t i = 0; i < n_docs; i++) {
      const double *nw = node_weights_a + sid_map[i] * (int64_t) n_ranks;
      double accum = 0.0;
      for (uint64_t r = 0; r < n_ranks; r++)
        if (nw[r] > 0.0) accum += rw.weights[r];
      double avg_sim = (rw.total > 0.0) ? accum / rw.total : 0.0;
      residual[i] = (bandwidth >= 0.0) ? exp(-(1.0 - avg_sim) * bandwidth) : avg_sim;
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
            pivot_row = &L_mat[pivot_idx * n_landmarks];
          }
        }
      }

      if (done) continue;

      #pragma omp for schedule(guided)
      for (uint64_t i = 0; i < n_docs; i++) {
        const int64_t *i_ro = nro_a + sid_map[i] * (int64_t) stride;
        const int64_t *p_ro = nro_a + sid_map[pivot_idx] * (int64_t) stride;
        const double *i_nw = node_weights_a + sid_map[i] * (int64_t) n_ranks;
        const double *p_nw = node_weights_a + sid_map[pivot_idx] * (int64_t) n_ranks;
        double kip = tk_inv_similarity_fast_cached(weights_arr, n_ranks,
                                            node_bits_a, i_ro, node_bits_a, p_ro,
                                            bandwidth, &rw, i_nw, p_nw, thr_i);
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

typedef struct {
  int64_t *uid_to_chol;
  int64_t *lm_sids;
  int64_t *feat_sid_map;
  double *adjustment;
  double *projection;
} tk_encode_nystrom_ctx_t;

static inline int tk_encode_nystrom_ctx_gc (lua_State *L) {
  tk_encode_nystrom_ctx_t *c = (tk_encode_nystrom_ctx_t *)lua_touserdata(L, 1);
  free(c->uid_to_chol);
  free(c->lm_sids);
  free(c->feat_sid_map);
  free(c->adjustment);
  free(c->projection);
  return 0;
}

#define TK_NYSTROM_ENCODER_MT "tk_nystrom_encoder_t"

typedef struct {
  double *projection;
  double *adjustment;
  int64_t *lm_sids;
  uint64_t m;
  uint64_t d;
  double bandwidth;
  double decay;
  double trace_ratio;
  bool destroyed;
} tk_nystrom_encoder_t;

static inline tk_nystrom_encoder_t *tk_nystrom_encoder_peek (lua_State *L, int i) {
  return (tk_nystrom_encoder_t *)luaL_checkudata(L, i, TK_NYSTROM_ENCODER_MT);
}

static inline int tk_nystrom_encoder_gc (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (!enc->destroyed) {
    free(enc->projection);
    free(enc->adjustment);
    free(enc->lm_sids);
    enc->destroyed = true;
  }
  return 0;
}

static inline int tk_nystrom_encode_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  tk_ivec_t *sparse = tk_ivec_peek(L, 2, "sparse_bits");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_features = tk_lua_checkunsigned(L, 4, "n_features");
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "inv");
  tk_inv_t *inv = tk_inv_peek(L, -1);
  lua_pop(L, 2);
  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, inv->n_ranks, enc->decay);
  uint64_t d = enc->d;
  uint64_t m = enc->m;
  tk_dvec_t *out = tk_dvec_create(L, n_samples * d, 0, 0);
  out->n = n_samples * d;
  uint64_t nystrom_stride = inv->n_ranks + 1;
  const int64_t *nystrom_nro = inv->node_rank_offsets->a;
  const int64_t *nystrom_nb = inv->node_bits->a;
  #pragma omp parallel
  {
    double thr_q[TK_INV_MAX_RANKS], thr_e[TK_INV_MAX_RANKS], thr_i[TK_INV_MAX_RANKS];
    int64_t q_ro[TK_INV_MAX_RANKS + 1];
    double *sims = (double *)malloc(m * sizeof(double));
    int64_t *feat_buf = (int64_t *)malloc(n_features * sizeof(int64_t));
    #pragma omp for schedule(dynamic, 16)
    for (uint64_t i = 0; i < n_samples; i++) {
      int64_t lo = (int64_t)(i * n_features);
      int64_t hi = (int64_t)((i + 1) * n_features);
      uint64_t start, end;
      {
        uint64_t l = 0, r = sparse->n;
        while (l < r) { uint64_t mid = l + (r - l) / 2; if (sparse->a[mid] < lo) l = mid + 1; else r = mid; }
        start = l;
      }
      {
        uint64_t l = start, r = sparse->n;
        while (l < r) { uint64_t mid = l + (r - l) / 2; if (sparse->a[mid] < hi) l = mid + 1; else r = mid; }
        end = l;
      }
      uint64_t nf = end - start;
      for (uint64_t f = 0; f < nf; f++)
        feat_buf[f] = sparse->a[start + f] - lo;
      tk_inv_partition_by_rank(inv->ranks->a, inv->n_ranks, feat_buf, nf, q_ro);
      for (uint64_t j = 0; j < m; j++) {
        if (enc->lm_sids[j] >= 0 && nf > 0) {
          const int64_t *lm_ro = nystrom_nro + enc->lm_sids[j] * (int64_t) nystrom_stride;
          sims[j] = tk_inv_similarity_fast(inv->weights->a, inv->n_ranks,
            feat_buf, q_ro, nystrom_nb, lm_ro, enc->bandwidth, &rw, thr_q, thr_e, thr_i);
        } else {
          sims[j] = 0.0;
        }
      }
      double *row = out->a + i * d;
      cblas_dgemv(CblasRowMajor, CblasTrans,
        (int)m, (int)d, 1.0, enc->projection, (int)d, sims, 1, 0.0, row, 1);
      for (uint64_t k = 0; k < d; k++)
        row[k] -= enc->adjustment[k];
    }
    free(sims);
    free(feat_buf);
  }
  return 1;
}

static inline int tk_nystrom_dims_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->d);
  return 1;
}

static inline int tk_nystrom_n_landmarks_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->m);
  return 1;
}

static inline int tk_nystrom_trace_ratio_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  lua_pushnumber(L, enc->trace_ratio);
  return 1;
}

static inline int tk_nystrom_landmark_ids_lua (lua_State *L) {
  tk_nystrom_encoder_peek(L, 1);
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "landmark_ids");
  return 1;
}

static inline int tk_nystrom_encoder_persist_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (enc->destroyed)
    return luaL_error(L, "cannot persist a destroyed encoder");
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_lua_fwrite(L, "TKny", 1, 4, fh);
  uint8_t version = 1;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->m, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->d, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->bandwidth, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &enc->decay, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &enc->trace_ratio, sizeof(double), 1, fh);
  tk_lua_fwrite(L, enc->projection, sizeof(double), enc->m * enc->d, fh);
  tk_lua_fwrite(L, enc->adjustment, sizeof(double), enc->d, fh);
  tk_lua_fwrite(L, enc->lm_sids, sizeof(int64_t), enc->m, fh);
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "landmark_ids");
  tk_ivec_t *lm_ids = tk_ivec_peek(L, -1, "landmark_ids");
  tk_ivec_persist(L, lm_ids, fh);
  lua_pop(L, 2);
  if (!tostr) {
    tk_lua_fclose(L, fh);
    return 0;
  } else {
    size_t len;
    char *data = tk_lua_fslurp(L, fh, &len);
    if (data) {
      lua_pushlstring(L, data, len);
      free(data);
      tk_lua_fclose(L, fh);
      return 1;
    } else {
      tk_lua_fclose(L, fh);
      return 0;
    }
  }
}

static luaL_Reg tk_nystrom_encoder_mt_fns[] = {
  { "encode", tk_nystrom_encode_lua },
  { "dims", tk_nystrom_dims_lua },
  { "n_landmarks", tk_nystrom_n_landmarks_lua },
  { "landmark_ids", tk_nystrom_landmark_ids_lua },
  { "trace_ratio", tk_nystrom_trace_ratio_lua },
  { "persist", tk_nystrom_encoder_persist_lua },
  { NULL, NULL }
};

static inline int tm_encode (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "inv");
  tk_inv_t *feat_inv = tk_inv_peek(L, -1);
  int feat_inv_idx = lua_gettop(L);

  lua_getfield(L, 1, "landmarks_inv");
  tk_inv_t *lm_inv = tk_inv_peekopt(L, -1);
  lua_pop(L, 1);
  if (!lm_inv) lm_inv = feat_inv;

  uint64_t n_lm_req = tk_lua_foptunsigned(L, 1, "encode", "n_landmarks", 0);
  uint64_t n_dims_req = tk_lua_fcheckunsigned(L, 1, "encode", "n_dims");
  double decay = tk_lua_foptnumber(L, 1, "encode", "decay", 0.0);
  double bandwidth = tk_lua_foptnumber(L, 1, "encode", "bandwidth", -1.0);
  double trace_tol = tk_lua_foptnumber(L, 1, "encode", "trace_tol", 1e-15);

  tk_inv_rank_weights_t feat_rw;
  tk_inv_precompute_rank_weights(&feat_rw, feat_inv->n_ranks, decay);

  tk_ivec_t *lm_ids = NULL, *chol_ids = NULL;
  tk_dvec_t *lm_chol = NULL, *full_chol = NULL;
  uint64_t m;
  double trace_ratio;
  tk_spectral_sample_landmarks(L, lm_inv, n_lm_req, trace_tol, decay, bandwidth,
    &lm_ids, &lm_chol, &full_chol, &chol_ids, &m, &trace_ratio);
  int lm_ids_idx = lua_gettop(L) - 3;

  uint64_t d = n_dims_req;
  if (d > m) d = m;

  if (m == 0 || d == 0) {
    tk_dvec_create(L, 0, 0, 0);
    tk_ivec_create(L, 0, 0, 0);
    lua_pushnil(L);
    lua_pushnil(L);
    return 4;
  }

  tk_encode_nystrom_ctx_t *ctx = (tk_encode_nystrom_ctx_t *)
    lua_newuserdata(L, sizeof(tk_encode_nystrom_ctx_t));
  memset(ctx, 0, sizeof(*ctx));
  lua_newtable(L);
  lua_pushcfunction(L, tk_encode_nystrom_ctx_gc);
  lua_setfield(L, -2, "__gc");
  lua_setmetatable(L, -2);

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
    return luaL_error(L, "encode: out of memory");
  }
  int n_eig = 0;
  int info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U',
    (int)m, gram->a, (int)m, 0.0, 0.0, (int)(m - d + 1), (int)m,
    0.0, &n_eig, eig_raw->a, ev_raw, (int)m, isuppz);
  free(isuppz);
  if (info != 0) {
    free(ev_raw);
    return luaL_error(L, "encode: dsyevr info=%d", info);
  }

  for (uint64_t i = 0; i < d / 2; i++) {
    double tmp = eig_raw->a[i];
    eig_raw->a[i] = eig_raw->a[d - 1 - i];
    eig_raw->a[d - 1 - i] = tmp;
  }
  int eig_raw_idx = lua_gettop(L);

  tk_dvec_t *eigvecs = tk_dvec_create(L, m * d, 0, 0);
  eigvecs->n = m * d;

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < m; i++)
    for (uint64_t k = 0; k < d; k++)
      eigvecs->a[i * d + k] = ev_raw[i * m + (d - 1 - k)];
  free(ev_raw);

  for (uint64_t k = 0; k < d; k++) {
    double s = (eig_raw->a[k] > 1e-15) ? 1.0 / sqrt(eig_raw->a[k]) : 0.0;
    for (uint64_t i = 0; i < m; i++)
      eigvecs->a[i * d + k] *= s;
  }

  tk_dvec_t *ccodes = tk_dvec_create(L, nc * d, 0, 0);
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
    lua_pushnil(L);
    lua_pushnil(L);
    return 4;
  }

  ctx->uid_to_chol = (int64_t *)malloc((uint64_t)(max_uid + 1) * sizeof(int64_t));
  memset(ctx->uid_to_chol, -1, (uint64_t)(max_uid + 1) * sizeof(int64_t));
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

  uint64_t feat_stride = feat_inv->n_ranks + 1;
  const int64_t *feat_nro = feat_inv->node_rank_offsets->a;
  const int64_t *feat_nb = feat_inv->node_bits->a;
  const double *feat_nw = feat_inv->node_weights->a;
  uint64_t feat_n_ranks = feat_inv->n_ranks;

  #pragma omp parallel
  {
    double thr_i[TK_INV_MAX_RANKS];
    double *sims = (double *)malloc(m * sizeof(double));

    #pragma omp for schedule(dynamic, 64)
    for (uint64_t i = 0; i < nf; i++) {
      int64_t sid = ctx->feat_sid_map[i];
      int64_t uid = feat_inv->sid_to_uid->a[sid];
      out_ids->a[i] = uid;
      int64_t cr = (uid <= max_uid) ? ctx->uid_to_chol[uid] : -1;
      if (cr >= 0) {
        memcpy(raw_codes->a + i * d, ccodes->a + (uint64_t)cr * d, d * sizeof(double));
      } else {
        const int64_t *q_ro = feat_nro + sid * (int64_t) feat_stride;
        const double *q_nw = feat_nw + sid * (int64_t) feat_n_ranks;
        for (uint64_t j = 0; j < m; j++) {
          double sim = 0.0;
          if (ctx->lm_sids[j] >= 0) {
            const int64_t *lm_ro = feat_nro + ctx->lm_sids[j] * (int64_t) feat_stride;
            const double *lm_nw = feat_nw + ctx->lm_sids[j] * (int64_t) feat_n_ranks;
            sim = tk_inv_similarity_fast_cached(feat_inv->weights->a,
              feat_n_ranks, feat_nb, q_ro, feat_nb, lm_ro, bandwidth, &feat_rw, q_nw, lm_nw, thr_i);
          }
          sims[j] = sim;
        }
        double *out = raw_codes->a + i * d;
        cblas_dgemv(CblasRowMajor, CblasTrans,
          (int)m, (int)d, 1.0, ctx->projection, (int)d, sims, 1, 0.0, out, 1);
        for (uint64_t k = 0; k < d; k++) out[k] -= ctx->adjustment[k];
      }
    }

    free(sims);
  }

  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->projection = ctx->projection; ctx->projection = NULL;
  enc->adjustment = ctx->adjustment; ctx->adjustment = NULL;
  enc->lm_sids = ctx->lm_sids; ctx->lm_sids = NULL;
  enc->m = m;
  enc->d = d;
  enc->bandwidth = bandwidth;
  enc->decay = decay;
  enc->trace_ratio = trace_ratio;
  enc->destroyed = false;

  lua_newtable(L);
  lua_pushvalue(L, feat_inv_idx);
  lua_setfield(L, -2, "inv");
  lua_pushvalue(L, lm_ids_idx);
  lua_setfield(L, -2, "landmark_ids");
  lua_setfenv(L, enc_idx);

  lua_pushvalue(L, raw_codes_idx);
  lua_pushvalue(L, out_ids_idx);
  lua_pushvalue(L, enc_idx);
  lua_pushvalue(L, eig_raw_idx);
  return 4;
}

static inline int tk_nystrom_encoder_load_lua (lua_State *L) {
  lua_settop(L, 3);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  int inv_idx = 2;
  tk_inv_peek(L, inv_idx);
  bool isstr = lua_type(L, 3) == LUA_TBOOLEAN && lua_toboolean(L, 3);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKny", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid nystrom encoder file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 1) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported nystrom encoder version %d", (int)version);
  }
  uint64_t m, d;
  double bandwidth, decay, trace_ratio;
  tk_lua_fread(L, &m, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &d, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &bandwidth, sizeof(double), 1, fh);
  tk_lua_fread(L, &decay, sizeof(double), 1, fh);
  tk_lua_fread(L, &trace_ratio, sizeof(double), 1, fh);
  double *projection = (double *)malloc(m * d * sizeof(double));
  double *adjustment = (double *)malloc(d * sizeof(double));
  int64_t *lm_sids = (int64_t *)malloc(m * sizeof(int64_t));
  if (!projection || !adjustment || !lm_sids) {
    free(projection); free(adjustment); free(lm_sids);
    tk_lua_fclose(L, fh);
    return luaL_error(L, "nystrom load: out of memory");
  }
  tk_lua_fread(L, projection, sizeof(double), m * d, fh);
  tk_lua_fread(L, adjustment, sizeof(double), d, fh);
  tk_lua_fread(L, lm_sids, sizeof(int64_t), m, fh);
  tk_ivec_load(L, fh);
  int lm_ids_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);
  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->projection = projection;
  enc->adjustment = adjustment;
  enc->lm_sids = lm_sids;
  enc->m = m;
  enc->d = d;
  enc->bandwidth = bandwidth;
  enc->decay = decay;
  enc->trace_ratio = trace_ratio;
  enc->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, inv_idx);
  lua_setfield(L, -2, "inv");
  lua_pushvalue(L, lm_ids_idx);
  lua_setfield(L, -2, "landmark_ids");
  lua_setfenv(L, enc_idx);
  lua_pushvalue(L, enc_idx);
  return 1;
}

static luaL_Reg tm_fns[] =
{
  { "encode", tm_encode },
  { "load", tk_nystrom_encoder_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_spectral (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_fns, 0);
  return 1;
}
