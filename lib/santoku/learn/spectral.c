#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/learn/buf.h>
#include <santoku/learn/gram.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>

typedef enum {
  TK_SPECTRAL_LINEAR = 0,
  TK_SPECTRAL_COSINE = 1,
  TK_SPECTRAL_ARCCOS0 = 2,
  TK_SPECTRAL_ARCCOS1 = 3,
} tk_spectral_kernel_t;

static inline double tk_spectral_kernel_apply (tk_spectral_kernel_t k, double c) {
  if (k == TK_SPECTRAL_LINEAR) return c;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;
  if (k == TK_SPECTRAL_ARCCOS0)
    return 1.0 - acos(c) / M_PI;
  if (k == TK_SPECTRAL_ARCCOS1) {
    double t = acos(c);
    return (sin(t) + (M_PI - t) * c) / M_PI;
  }
  return c;
}


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

typedef struct { int64_t tok; double val; } tk_csr_sort_entry_t;
#define tk_csr_sort_entry_lt(a, b) ((a).tok < (b).tok)
KSORT_INIT(tk_csr_sort_entry, tk_csr_sort_entry_t, tk_csr_sort_entry_lt)

#define TK_CHOL_BLOCK 64

static inline void tk_spectral_sample_landmarks (
  lua_State *L,
  double *dense, int64_t d_input, uint64_t n_docs_hint,
  const uint8_t *bits_data, uint64_t bits_d, uint64_t bits_n_samples,
  const int64_t *csr_offsets, const int64_t *csr_tokens,
  const double *csr_values, const double *csr_norms, uint64_t csr_n_samples, uint64_t csr_n_tokens,
  tk_spectral_kernel_t kernel, const double *dense_norms,
  uint64_t n_landmarks,
  double trace_tol,
  tk_ivec_t **ids_out,
  tk_dvec_t **chol_out,
  double **full_chol_out,
  uint64_t *n_docs_out,
  uint64_t *actual_landmarks_out,
  double *trace_ratio_out
) {
  int is_dense = (dense != NULL && csr_offsets == NULL);
  int is_bits = (dense == NULL && bits_data != NULL && csr_offsets == NULL);
  int is_csr = (csr_offsets != NULL);
  uint64_t bits_row_bytes = is_bits ? TK_CVEC_BITS_BYTES(bits_d) : 0;
  uint64_t n_docs = 0;
  if (is_csr) {
    n_docs = csr_n_samples;
  } else if (is_dense) {
    n_docs = n_docs_hint;
  } else {
    n_docs = bits_n_samples;
  }

  if (n_landmarks == 0 || n_landmarks > n_docs)
    n_landmarks = n_docs;
  if (n_landmarks == 0) {
    *ids_out = tk_ivec_create(L, 0, 0, 0);
    *chol_out = NULL;
    *full_chol_out = NULL;
    *n_docs_out = 0;
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
  ctx->L_mat = (double *)calloc(n_landmarks * n_docs, sizeof(double));
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

  for (uint64_t i = 0; i < n_docs; i++)
    sid_map[i] = (int64_t)i;

  memset(residual, 0, n_docs * sizeof(double));
  memset(landmark_idx_map, -1, n_landmarks * sizeof(int64_t));

  uint64_t actual_landmarks = 0;
  double initial_trace = 0.0;
  double trace = 0.0;
  bool done = false;

  double *kip_block = (double *)malloc(n_docs * TK_CHOL_BLOCK * sizeof(double));
  double *cross_dots = (double *)malloc(n_docs * TK_CHOL_BLOCK * sizeof(double));
  double *pivot_prev_L = (double *)malloc(TK_CHOL_BLOCK * n_landmarks * sizeof(double));
  double *pivot_dense_rows = is_dense
    ? (double *)malloc(TK_CHOL_BLOCK * (uint64_t)d_input * sizeof(double)) : NULL;
  uint64_t blk_pivots[TK_CHOL_BLOCK];

  double *proposal = (double *)malloc(n_docs * sizeof(double));
  double *proposal_cdf = (double *)malloc(n_docs * sizeof(double));

  if (!kip_block || !cross_dots || !pivot_prev_L
      || (is_dense && !pivot_dense_rows)
      || !proposal || !proposal_cdf) {
    free(kip_block); free(cross_dots);
    free(pivot_prev_L); free(pivot_dense_rows);
    free(proposal); free(proposal_cdf);
    luaL_error(L, "sample_landmarks: out of memory (buffers)");
    return;
  }

  #pragma omp parallel for reduction(+:initial_trace)
  for (uint64_t i = 0; i < n_docs; i++) {
    residual[i] = 1.0;
    initial_trace += 1.0;
  }

  memcpy(proposal, residual, n_docs * sizeof(double));
  double proposal_total = initial_trace;
  {
    double cum = 0.0;
    for (uint64_t i = 0; i < n_docs; i++) {
      cum += proposal[i];
      proposal_cdf[i] = cum;
    }
  }

  #define SAMPLE_PROPOSAL() ({ \
    double _r = tk_fast_drand() * proposal_total; \
    uint64_t _lo = 0, _hi = n_docs; \
    while (_lo < _hi) { \
      uint64_t _mid = _lo + (_hi - _lo) / 2; \
      if (proposal_cdf[_mid] < _r) _lo = _mid + 1; else _hi = _mid; \
    } \
    _lo < n_docs ? _lo : n_docs - 1; \
  })

  uint64_t total_proposed = 0;
  uint64_t total_accepted = 0;

  while (actual_landmarks < n_landmarks && !done) {

    trace = 0.0;
    #pragma omp parallel for reduction(+:trace)
    for (uint64_t i = 0; i < n_docs; i++)
      if (residual[i] > 0.0) trace += residual[i];
    if (trace < 1e-15 ||
        (trace_tol > 0.0 && initial_trace > 0.0 &&
         trace / initial_trace < trace_tol)) {
      done = true;
      break;
    }

    if (total_proposed > 0 && total_accepted * 4 < total_proposed) {
      memcpy(proposal, residual, n_docs * sizeof(double));
      proposal_total = trace;
      double cum = 0.0;
      for (uint64_t i = 0; i < n_docs; i++) {
        cum += proposal[i];
        proposal_cdf[i] = cum;
      }
      total_proposed = 0;
      total_accepted = 0;
    }

    uint64_t max_blk = n_landmarks - actual_landmarks;
    if (max_blk > TK_CHOL_BLOCK) max_blk = TK_CHOL_BLOCK;

    uint64_t n_propose = max_blk * 2;
    if (n_propose > TK_CHOL_BLOCK) n_propose = TK_CHOL_BLOCK;
    uint64_t np = 0;
    for (uint64_t b = 0; b < n_propose; b++) {
      uint64_t pi = SAMPLE_PROPOSAL();
      if (residual[pi] < 1e-15) continue;
      int dup = 0;
      for (uint64_t k = 0; k < np; k++)
        if (blk_pivots[k] == pi) { dup = 1; break; }
      if (dup) continue;
      blk_pivots[np] = pi;
      np++;
    }
    total_proposed += n_propose;
    if (np == 0) { done = true; break; }

    if (is_csr) {
      int64_t plo_arr[TK_CHOL_BLOCK], phi_arr[TK_CHOL_BLOCK];
      for (uint64_t b = 0; b < np; b++) {
        plo_arr[b] = csr_offsets[blk_pivots[b]];
        phi_arr[b] = csr_offsets[blk_pivots[b] + 1];
      }

      {
        int64_t *piv_csc_off = (int64_t *)calloc(csr_n_tokens + 1, sizeof(int64_t));
        for (uint64_t b = 0; b < np; b++) {
          int64_t p = plo_arr[b];
          while (p < phi_arr[b]) {
            int64_t tok = csr_tokens[p];
            piv_csc_off[tok + 1]++;
            while (p < phi_arr[b] && csr_tokens[p] == tok) p++;
          }
        }
        for (uint64_t t = 0; t < csr_n_tokens; t++)
          piv_csc_off[t + 1] += piv_csc_off[t];
        uint64_t total_pnnz = (uint64_t)piv_csc_off[csr_n_tokens];
        uint8_t *piv_csc_piv = (uint8_t *)malloc(total_pnnz);
        double *piv_csc_val = (double *)malloc(total_pnnz * sizeof(double));
        int64_t *piv_csc_pos = (int64_t *)malloc(csr_n_tokens * sizeof(int64_t));
        memcpy(piv_csc_pos, piv_csc_off, csr_n_tokens * sizeof(int64_t));
        for (uint64_t b = 0; b < np; b++) {
          int64_t p = plo_arr[b];
          while (p < phi_arr[b]) {
            int64_t tok = csr_tokens[p];
            double sum = 0.0;
            while (p < phi_arr[b] && csr_tokens[p] == tok) { sum += csr_values[p]; p++; }
            int64_t pos = piv_csc_pos[tok]++;
            piv_csc_piv[pos] = (uint8_t)b;
            piv_csc_val[pos] = sum;
          }
        }
        free(piv_csc_pos);
        const int64_t *restrict pc_off = piv_csc_off;
        const uint8_t *restrict pc_piv = piv_csc_piv;
        const double *restrict pc_val = piv_csc_val;
        const int64_t *restrict c_off = csr_offsets;
        const int64_t *restrict c_tok = csr_tokens;
        const double *restrict c_val = csr_values;
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++) {
          double kip_row[TK_CHOL_BLOCK];
          memset(kip_row, 0, np * sizeof(double));
          int64_t jlo = c_off[i], jhi = c_off[i + 1];
          for (int64_t j = jlo; j < jhi; j++) {
            int64_t tok = c_tok[j];
            double val = c_val[j];
            int64_t clo = pc_off[tok], chi = pc_off[tok + 1];
            for (int64_t c = clo; c < chi; c++)
              kip_row[pc_piv[c]] += val * pc_val[c];
          }
          memcpy(kip_block + i * np, kip_row, np * sizeof(double));
        }
        free(piv_csc_off);
        free(piv_csc_piv);
        free(piv_csc_val);
      }

      {
        double pnorms[TK_CHOL_BLOCK];
        for (uint64_t b = 0; b < np; b++)
          pnorms[b] = csr_norms[blk_pivots[b]];
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++)
          for (uint64_t b = 0; b < np; b++) {
            double denom = csr_norms[i] * pnorms[b];
            kip_block[i * np + b] = tk_spectral_kernel_apply(kernel,
              denom > 1e-15 ? kip_block[i * np + b] / denom : 0.0);
          }
      }

    } else if (is_bits) {
      double inv_d = 1.0 / (double)bits_d;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_docs; i++) {
        const uint8_t *row_i = bits_data + i * bits_row_bytes;
        for (uint64_t b = 0; b < np; b++) {
          uint64_t ham = tk_cvec_bits_hamming_serial(
            row_i, bits_data + blk_pivots[b] * bits_row_bytes, bits_d);
          kip_block[i * np + b] = 1.0 - (double)ham * inv_d;
        }
      }

    } else if (is_dense) {
      for (uint64_t b = 0; b < np; b++)
        memcpy(pivot_dense_rows + b * (uint64_t)d_input,
               dense + blk_pivots[b] * (uint64_t)d_input,
               (uint64_t)d_input * sizeof(double));
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        (int)n_docs, (int)np, (int)d_input, 1.0,
        dense, (int)d_input,
        pivot_dense_rows, (int)d_input,
        0.0, kip_block, (int)np);
      {
        double pnorms[TK_CHOL_BLOCK];
        for (uint64_t b = 0; b < np; b++)
          pnorms[b] = dense_norms[blk_pivots[b]];
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++)
          for (uint64_t b = 0; b < np; b++) {
            double denom = dense_norms[i] * pnorms[b];
            kip_block[i * np + b] = tk_spectral_kernel_apply(kernel,
              denom > 1e-15 ? kip_block[i * np + b] / denom : 0.0);
          }
      }
    }

    uint64_t jb = actual_landmarks;
    if (jb > 0) {
      for (uint64_t k = 0; k < jb; k++)
        for (uint64_t b = 0; b < np; b++)
          pivot_prev_L[b * jb + k] = L_mat[k * n_docs + blk_pivots[b]];
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        (int)n_docs, (int)np, (int)jb, 1.0,
        L_mat, (int)n_docs,
        pivot_prev_L, (int)jb,
        0.0, cross_dots, (int)n_docs);
    }

    uint64_t within_accepted = 0;
    for (uint64_t b = 0; b < np && actual_landmarks < n_landmarks; b++) {
      uint64_t pi = blk_pivots[b];

      if (proposal[pi] > 1e-15) {
        double accept_prob = residual[pi] / proposal[pi];
        if (tk_fast_drand() > accept_prob) continue;
      } else {
        continue;
      }
      total_accepted++;

      uint64_t col = actual_landmarks;
      double sc_sq = residual[pi];
      if (sc_sq < 1e-15) continue;
      double sc = sqrt(sc_sq);

      landmark_sids[col] = sid_map[pi];
      landmark_idx_map[col] = (int64_t)pi;

      double within_pi[TK_CHOL_BLOCK];
      for (uint64_t k = 0; k < within_accepted; k++)
        within_pi[k] = L_mat[(jb + k) * n_docs + pi];
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_docs; i++) {
        double raw = kip_block[i * np + b];
        double cross = (jb > 0) ? cross_dots[b * n_docs + i] : 0.0;
        double within = 0.0;
        for (uint64_t k = 0; k < within_accepted; k++)
          within += L_mat[(jb + k) * n_docs + i] * within_pi[k];
        double lij = (raw - cross - within) / sc;
        L_mat[col * n_docs + i] = lij;
        residual[i] -= lij * lij;
        if (residual[i] < 0.0) residual[i] = 0.0;
      }
      residual[pi] = 0.0;

      actual_landmarks++;
      within_accepted++;
    }
  }

  free(kip_block);
  free(cross_dots);
  free(pivot_prev_L);
  free(pivot_dense_rows);
  free(proposal);
  free(proposal_cdf);

  tk_ivec_t *landmark_ids = tk_ivec_create(L, actual_landmarks, 0, 0);
  for (uint64_t i = 0; i < actual_landmarks; i++)
    landmark_ids->a[i] = landmark_sids[i];
  landmark_ids->n = actual_landmarks;

  tk_dvec_t *chol = tk_dvec_create(NULL, actual_landmarks * actual_landmarks, 0, 0);
  #pragma omp parallel for schedule(static)
  for (uint64_t li = 0; li < actual_landmarks; li++) {
    uint64_t doc_idx = (uint64_t)landmark_idx_map[li];
    for (uint64_t k = 0; k < actual_landmarks; k++)
      chol->a[li * actual_landmarks + k] = L_mat[k * n_docs + doc_idx];
  }
  chol->n = actual_landmarks * actual_landmarks;

  if (actual_landmarks > 0 && actual_landmarks < n_landmarks) {
    double *shrunk = realloc(L_mat, actual_landmarks * n_docs * sizeof(double));
    if (shrunk) L_mat = shrunk;
  }

  double *full_chol = L_mat;
  ctx->L_mat = NULL;

  lua_remove(L, ctx_idx);

  *ids_out = landmark_ids;
  *chol_out = chol;
  *full_chol_out = full_chol;
  *n_docs_out = n_docs;
  *actual_landmarks_out = actual_landmarks;
  *trace_ratio_out = (initial_trace > 0.0) ? (trace / initial_trace) : 0.0;
}

typedef struct {
  double *projection;
  tk_dvec_t *lm_chol;
  double *full_chol;
} tk_encode_nystrom_ctx_t;

static inline int tk_encode_nystrom_ctx_gc (lua_State *L) {
  tk_encode_nystrom_ctx_t *c = (tk_encode_nystrom_ctx_t *)lua_touserdata(L, 1);
  free(c->projection);
  if (c->lm_chol) tk_dvec_destroy(c->lm_chol);
  free(c->full_chol);
  return 0;
}

#define TK_NYSTROM_ENCODER_MT "tk_nystrom_encoder_t"

typedef struct {
  float *projection;
  double *lm_vecs;
  uint8_t *lm_bits;
  int64_t *lm_csr_offsets;
  int64_t *lm_csr_tokens;
  double *lm_csr_values;
  double *lm_csr_norms;
  int64_t *lm_csc_offsets;
  int64_t *lm_csc_rows;
  double *lm_csc_values;
  double *feature_weights;
  uint64_t fw_len;
  double *lm_dense_norms;
  uint64_t m;
  uint64_t d;
  int64_t d_input;
  uint64_t bits_d;
  uint64_t csr_n_tokens;
  double trace_ratio;
  uint8_t dense_mode;
  tk_spectral_kernel_t kernel;
  bool destroyed;
} tk_nystrom_encoder_t;

static inline tk_nystrom_encoder_t *tk_nystrom_encoder_peek (lua_State *L, int i) {
  return (tk_nystrom_encoder_t *)luaL_checkudata(L, i, TK_NYSTROM_ENCODER_MT);
}

static inline int tk_nystrom_encoder_gc (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (!enc->destroyed) {
    free(enc->projection);
    free(enc->lm_vecs);
    free(enc->lm_bits);
    free(enc->lm_csr_offsets);
    free(enc->lm_csr_tokens);
    free(enc->lm_csr_values);
    free(enc->lm_csr_norms);
    free(enc->lm_csc_offsets);
    free(enc->lm_csc_rows);
    free(enc->lm_csc_values);
    free(enc->feature_weights);
    free(enc->lm_dense_norms);
    enc->destroyed = true;
  }
  return 0;
}

static inline int tk_nystrom_encode_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (!enc->projection)
    return luaL_error(L, "encode: projection released");
  uint64_t d = enc->d;
  uint64_t m = enc->m;

  if (enc->dense_mode == 2) {
    tk_cvec_t *bits_cv = tk_cvec_peek(L, 2, "bits");
    uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
    uint64_t tile = (uint64_t)luaL_optinteger(L, 4, 0);
    uint64_t bd = enc->bits_d;
    uint64_t row_bytes = TK_CVEC_BITS_BYTES(bd);
    const uint8_t *bdata = (const uint8_t *)bits_cv->a;
    double inv_bd = 1.0 / (double)bd;
    TK_FVEC_BUF(out, 5, n_samples * d);
    if (tile == 0 || tile > n_samples) tile = n_samples;
    uint64_t sims_need = tile * m;
    float *sims_f = (float *)malloc(sims_need * sizeof(float));
    if (!sims_f) return luaL_error(L, "encode: out of memory");
    for (uint64_t base = 0; base < n_samples; base += tile) {
      uint64_t blk = base + tile <= n_samples ? tile : n_samples - base;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < blk; i++) {
        const uint8_t *row_i = bdata + (base + i) * row_bytes;
        float *sims_row = sims_f + i * m;
        for (uint64_t j = 0; j < m; j++) {
          uint64_t ham = tk_cvec_bits_hamming_serial(
            row_i, enc->lm_bits + j * row_bytes, bd);
          sims_row[j] = (float)(1.0 - (double)ham * inv_bd);
        }
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)blk, (int)d, (int)m, 1.0f,
        sims_f, (int)m,
        enc->projection, (int)d,
        0.0f, out->a + base * d, (int)d);
    }
    free(sims_f);
    lua_pushvalue(L, out_idx);
    return 1;
  }

  if (enc->dense_mode == 1) {
    tk_dvec_t *codes = tk_dvec_peek(L, 2, "codes");
    uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
    int64_t d_in = enc->d_input;
    double *src = codes->a;
    double *tmp_scaled = NULL;
    if (enc->feature_weights) {
      tmp_scaled = (double *)malloc(n_samples * (uint64_t)d_in * sizeof(double));
      if (!tmp_scaled) return luaL_error(L, "encode: out of memory");
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++)
        for (int64_t k = 0; k < d_in; k++)
          tmp_scaled[i * (uint64_t)d_in + (uint64_t)k] = codes->a[i * (uint64_t)d_in + (uint64_t)k] * enc->feature_weights[k];
      src = tmp_scaled;
    }
    TK_FVEC_BUF(out, 4, n_samples * d);
    uint64_t nd = n_samples * (uint64_t)d_in;
    float *src_f = (float *)malloc(nd * sizeof(float));
    if (!src_f) { free(tmp_scaled); return luaL_error(L, "encode: out of memory"); }
    for (uint64_t i = 0; i < nd; i++) src_f[i] = (float)src[i];
    free(tmp_scaled);
    float *lm_f = (float *)malloc(m * (uint64_t)d_in * sizeof(float));
    if (!lm_f) { free(src_f); return luaL_error(L, "encode: out of memory"); }
    for (uint64_t i = 0; i < m * (uint64_t)d_in; i++) lm_f[i] = (float)enc->lm_vecs[i];
    uint64_t sims_need = n_samples * m;
    float *sims_f = (float *)malloc(sims_need * sizeof(float));
    if (!sims_f) { free(lm_f); free(src_f); return luaL_error(L, "encode: out of memory"); }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      (int)n_samples, (int)m, (int)d_in, 1.0f,
      src_f, (int)d_in, lm_f, (int)d_in,
      0.0f, sims_f, (int)m);
    free(lm_f);
    {
      float *new_norms = (float *)malloc(n_samples * sizeof(float));
      if (!new_norms) { free(sims_f); free(src_f); return luaL_error(L, "encode: out of memory"); }
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++)
        new_norms[i] = cblas_snrm2((int)d_in, src_f + i * (uint64_t)d_in, 1);
      free(src_f);
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++)
        for (uint64_t j = 0; j < m; j++) {
          double denom = (double)new_norms[i] * enc->lm_dense_norms[j];
          sims_f[i * m + j] = (float)tk_spectral_kernel_apply(enc->kernel,
            denom > 1e-15 ? (double)sims_f[i * m + j] / denom : 0.0);
        }
      free(new_norms);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)n_samples, (int)d, (int)m, 1.0f,
      sims_f, (int)m,
      enc->projection, (int)d,
      0.0f, out->a, (int)d);
    free(sims_f);
    lua_pushvalue(L, out_idx);
    return 1;
  }

  if (enc->dense_mode == 3) {
    luaL_checktype(L, 2, LUA_TTABLE);
    lua_getfield(L, 2, "offsets");
    tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_getfield(L, 2, "values");
    tk_dvec_t *values = tk_dvec_peekopt(L, -1);
    lua_getfield(L, 2, "n_samples");
    uint64_t n_samples = (uint64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 4);
    lua_getfield(L, 2, "output");
    tk_fvec_t *out_fv = tk_fvec_peekopt(L, -1);
    int out_fv_idx = out_fv ? lua_gettop(L) : 0;
    if (!out_fv) lua_pop(L, 1);
    uint64_t nnz = tokens->n;
    double *sval = (double *)malloc(nnz * sizeof(double));
    double *norms = (double *)calloc(n_samples, sizeof(double));
    if (!sval || !norms) {
      free(sval); free(norms);
      return luaL_error(L, "encode: out of memory");
    }
    for (uint64_t i = 0; i < nnz; i++)
      sval[i] = values ? values->a[i] : 1.0;
    if (enc->feature_weights) {
      for (uint64_t i = 0; i < nnz; i++)
        sval[i] *= enc->feature_weights[tokens->a[i]];
    }
    {
      double *tmp = (double *)calloc(enc->csr_n_tokens, sizeof(double));
      if (!tmp) { free(sval); free(norms); return luaL_error(L, "encode: out of memory"); }
      for (uint64_t s = 0; s < n_samples; s++) {
        int64_t lo = offsets->a[s], hi = offsets->a[s + 1];
        for (int64_t j = lo; j < hi; j++)
          tmp[tokens->a[j]] += sval[j];
        double ss = 0.0;
        for (int64_t j = lo; j < hi; j++) {
          int64_t tok = tokens->a[j];
          if (tmp[tok] != 0.0) { ss += tmp[tok] * tmp[tok]; tmp[tok] = 0.0; }
        }
        norms[s] = sqrt(ss);
      }
      free(tmp);
    }
    uint64_t sims_need = n_samples * m;
    float *sims_f = (float *)malloc(sims_need * sizeof(float));
    if (!sims_f) { free(sval); free(norms); return luaL_error(L, "encode: out of memory"); }
    {
      const int64_t *restrict csc_off = enc->lm_csc_offsets;
      const int64_t *restrict csc_rows = enc->lm_csc_rows;
      const double *restrict csc_vals = enc->lm_csc_values;
      const int64_t *restrict off_a = offsets->a;
      const int64_t *restrict tok_a = tokens->a;
      const double *restrict sv = sval;
      #pragma omp parallel
      {
        double *row_buf = (double *)calloc(m, sizeof(double));
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < n_samples; i++) {
          float *restrict sims_row = sims_f + i * m;
          int64_t jlo = off_a[i], jhi = off_a[i + 1];
          for (int64_t j = jlo; j < jhi; j++) {
            int64_t tok = tok_a[j];
            double val = sv[j];
            int64_t clo = csc_off[tok], chi = csc_off[tok + 1];
            for (int64_t c = clo; c < chi; c++)
              row_buf[(uint64_t)csc_rows[c]] += val * csc_vals[c];
          }
          for (uint64_t j = 0; j < m; j++) {
            double denom = norms[i] * enc->lm_csr_norms[j];
            sims_row[j] = (float)tk_spectral_kernel_apply(enc->kernel,
              denom > 1e-15 ? row_buf[j] / denom : 0.0);
            row_buf[j] = 0.0;
          }
        }
        free(row_buf);
      }
    }
    free(sval); free(norms);
    tk_fvec_t *out;
    if (out_fv) {
      tk_fvec_ensure(out_fv, n_samples * d);
      out_fv->n = n_samples * d;
      out = out_fv;
    } else {
      out = tk_fvec_create(L, n_samples * d, 0, 0);
      out->n = n_samples * d;
      out_fv_idx = lua_gettop(L);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)n_samples, (int)d, (int)m, 1.0f,
      sims_f, (int)m,
      enc->projection, (int)d,
      0.0f, out->a, (int)d);
    free(sims_f);
    lua_pushvalue(L, out_fv_idx);
    return 1;
  }

  return luaL_error(L, "encode: unknown dense_mode %d", (int)enc->dense_mode);
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

static inline int tk_nystrom_restrict_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  tk_ivec_t *keep = tk_ivec_peek(L, 2, "keep_dims");
  uint64_t new_d = keep->n;
  if (new_d == 0) return luaL_error(L, "restrict: empty keep_dims");
  for (uint64_t i = 0; i < new_d; i++)
    if (keep->a[i] < 0 || (uint64_t)keep->a[i] >= enc->d)
      return luaL_error(L, "restrict: dim %d out of range [0,%d)", (int)keep->a[i], (int)enc->d);
  float *new_proj = (float *)malloc(enc->m * new_d * sizeof(float));
  for (uint64_t j = 0; j < enc->m; j++)
    for (uint64_t i = 0; i < new_d; i++)
      new_proj[j * new_d + i] = enc->projection[j * enc->d + (uint64_t)keep->a[i]];
  free(enc->projection);
  enc->projection = new_proj;
  enc->d = new_d;
  return 0;
}

static inline int tk_nystrom_encoder_persist_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (enc->destroyed)
    return luaL_error(L, "cannot persist a destroyed encoder");
  if (!enc->projection)
    return luaL_error(L, "cannot persist: projection released");
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
  uint8_t version = 17;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->dense_mode, sizeof(uint8_t), 1, fh);
  uint8_t kernel_byte = (uint8_t)enc->kernel;
  tk_lua_fwrite(L, &kernel_byte, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->m, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->d, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->trace_ratio, sizeof(double), 1, fh);
  tk_lua_fwrite(L, enc->projection, sizeof(float), enc->m * enc->d, fh);
  tk_lua_fwrite(L, &enc->fw_len, sizeof(uint64_t), 1, fh);
  if (enc->fw_len > 0)
    tk_lua_fwrite(L, enc->feature_weights, sizeof(double), enc->fw_len, fh);
  if (enc->dense_mode == 2) {
    tk_lua_fwrite(L, &enc->bits_d, sizeof(uint64_t), 1, fh);
    uint64_t row_bytes = TK_CVEC_BITS_BYTES(enc->bits_d);
    tk_lua_fwrite(L, enc->lm_bits, 1, enc->m * row_bytes, fh);
  } else if (enc->dense_mode == 1) {
    tk_lua_fwrite(L, &enc->d_input, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, enc->lm_vecs, sizeof(double), enc->m * (uint64_t)enc->d_input, fh);
    tk_lua_fwrite(L, enc->lm_dense_norms, sizeof(double), enc->m, fh);
  } else if (enc->dense_mode == 3) {
    tk_lua_fwrite(L, &enc->csr_n_tokens, sizeof(uint64_t), 1, fh);
    uint64_t total_csr = (uint64_t)enc->lm_csr_offsets[enc->m];
    tk_lua_fwrite(L, enc->lm_csr_offsets, sizeof(int64_t), enc->m + 1, fh);
    tk_lua_fwrite(L, enc->lm_csr_tokens, sizeof(int64_t), total_csr, fh);
    tk_lua_fwrite(L, enc->lm_csr_values, sizeof(double), total_csr, fh);
    tk_lua_fwrite(L, enc->lm_csr_norms, sizeof(double), enc->m, fh);
  }
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

static inline int tk_nystrom_shrink_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  free(enc->projection);
  enc->projection = NULL;
  return 0;
}

static luaL_Reg tk_nystrom_encoder_mt_fns[] = {
  { "encode", tk_nystrom_encode_lua },
  { "dims", tk_nystrom_dims_lua },
  { "n_landmarks", tk_nystrom_n_landmarks_lua },
  { "landmark_ids", tk_nystrom_landmark_ids_lua },
  { "trace_ratio", tk_nystrom_trace_ratio_lua },
  { "restrict", tk_nystrom_restrict_lua },
  { "persist", tk_nystrom_encoder_persist_lua },
  { "shrink", tk_nystrom_shrink_lua },
  { NULL, NULL }
};

static inline int tm_encode (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  double *dense_data = NULL;
  int64_t dense_d_input = 0;
  uint64_t dense_n_samples = 0;
  const uint8_t *bits_data = NULL;
  uint64_t bits_d = 0;
  uint64_t bits_n_samples = 0;
  int64_t *csr_offsets_a = NULL;
  int64_t *csr_tokens_a = NULL;
  double *csr_values_a = NULL;
  double *csr_norms_a = NULL;
  uint64_t csr_n_samples = 0;
  uint64_t csr_n_tokens = 0;
  int has_csr = 0;
  double *dense_norms = NULL;
  double *dense_fw_scaled = NULL;
  tk_dvec_t *fw_dv = NULL;
  {
    lua_getfield(L, 1, "offsets");
    has_csr = !lua_isnil(L, -1);
    if (has_csr) {
      tk_ivec_t *off_iv = tk_ivec_peek(L, -1, "offsets");
      lua_pop(L, 1);
      lua_getfield(L, 1, "tokens");
      tk_ivec_t *tok_iv = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      csr_n_samples = tk_lua_fcheckunsigned(L, 1, "encode", "n_samples");
      csr_n_tokens = tk_lua_fcheckunsigned(L, 1, "encode", "n_tokens");
      lua_getfield(L, 1, "values");
      tk_dvec_t *val_dv = tk_dvec_peekopt(L, -1);
      lua_pop(L, 1);
      lua_getfield(L, 1, "feature_weights");
      fw_dv = tk_dvec_peekopt(L, -1);
      lua_pop(L, 1);
      uint64_t nnz = tok_iv->n;
      csr_offsets_a = off_iv->a;
      csr_tokens_a = tok_iv->a;
      csr_values_a = (double *)malloc(nnz * sizeof(double));
      csr_norms_a = (double *)calloc(csr_n_samples, sizeof(double));
      for (uint64_t i = 0; i < nnz; i++)
        csr_values_a[i] = val_dv ? val_dv->a[i] : 1.0;
      if (fw_dv) {
        for (uint64_t i = 0; i < nnz; i++)
          csr_values_a[i] *= fw_dv->a[csr_tokens_a[i]];
      }
    } else {
      lua_pop(L, 1);
      lua_getfield(L, 1, "bits");
      int has_bits = !lua_isnil(L, -1);
      if (has_bits) {
        tk_cvec_t *bits_cv = tk_cvec_peek(L, -1, "bits");
        bits_n_samples = tk_lua_fcheckunsigned(L, 1, "encode", "n_samples");
        bits_d = tk_lua_fcheckunsigned(L, 1, "encode", "d_bits");
        bits_data = (const uint8_t *)bits_cv->a;
        lua_pop(L, 1);
      } else {
        lua_pop(L, 1);
        lua_getfield(L, 1, "codes");
        tk_dvec_t *codes_dv = tk_dvec_peek(L, -1, "codes");
        dense_n_samples = tk_lua_fcheckunsigned(L, 1, "encode", "n_samples");
        dense_d_input = (int64_t)(codes_dv->n / dense_n_samples);
        lua_pop(L, 1);
        lua_getfield(L, 1, "feature_weights");
        fw_dv = tk_dvec_peekopt(L, -1);
        lua_pop(L, 1);
        if (fw_dv) {
          dense_fw_scaled = (double *)malloc(dense_n_samples * (uint64_t)dense_d_input * sizeof(double));
          for (uint64_t i = 0; i < dense_n_samples; i++)
            for (int64_t k = 0; k < dense_d_input; k++)
              dense_fw_scaled[i * (uint64_t)dense_d_input + (uint64_t)k] =
                codes_dv->a[i * (uint64_t)dense_d_input + (uint64_t)k] * fw_dv->a[k];
          dense_data = dense_fw_scaled;
        } else {
          dense_data = codes_dv->a;
        }
      }
    }
  }

  lua_getfield(L, 1, "kernel");
  const char *kernel_str = lua_isnil(L, -1) ? "cosine" : lua_tostring(L, -1);
  lua_pop(L, 1);
  tk_spectral_kernel_t kernel = TK_SPECTRAL_COSINE;
  if (strcmp(kernel_str, "cosine") == 0) kernel = TK_SPECTRAL_COSINE;
  else if (strcmp(kernel_str, "arccos0") == 0) kernel = TK_SPECTRAL_ARCCOS0;
  else if (strcmp(kernel_str, "arccos1") == 0) kernel = TK_SPECTRAL_ARCCOS1;
  else if (strcmp(kernel_str, "linear") != 0)
    return luaL_error(L, "encode: unknown kernel '%s'", kernel_str);

  if (dense_data) {
    dense_norms = (double *)malloc(dense_n_samples * sizeof(double));
    for (uint64_t i = 0; i < dense_n_samples; i++)
      dense_norms[i] = cblas_dnrm2((int)dense_d_input, dense_data + i * (uint64_t)dense_d_input, 1);
  }

  uint64_t n_lm_req = tk_lua_foptunsigned(L, 1, "encode", "n_landmarks", 0);
  double trace_tol = tk_lua_foptnumber(L, 1, "encode", "trace_tol", 0.0);

  int64_t *sorted_csr_tokens = NULL;
  double *sorted_csr_values = NULL;
  if (has_csr) {
    uint64_t nnz = (uint64_t)csr_offsets_a[csr_n_samples];
    sorted_csr_tokens = (int64_t *)malloc(nnz * sizeof(int64_t));
    sorted_csr_values = (double *)malloc(nnz * sizeof(double));
    memcpy(sorted_csr_tokens, csr_tokens_a, nnz * sizeof(int64_t));
    memcpy(sorted_csr_values, csr_values_a, nnz * sizeof(double));
    int64_t max_row = 0;
    for (uint64_t s = 0; s < csr_n_samples; s++) {
      int64_t len = csr_offsets_a[s + 1] - csr_offsets_a[s];
      if (len > max_row) max_row = len;
    }
    #pragma omp parallel
    {
      tk_csr_sort_entry_t *buf = (tk_csr_sort_entry_t *)malloc(
        (uint64_t)max_row * sizeof(tk_csr_sort_entry_t));
      #pragma omp for schedule(dynamic)
      for (uint64_t s = 0; s < csr_n_samples; s++) {
        int64_t lo = csr_offsets_a[s], hi = csr_offsets_a[s + 1];
        int64_t len = hi - lo;
        if (len <= 1) continue;
        for (int64_t j = 0; j < len; j++) {
          buf[j].tok = sorted_csr_tokens[lo + j];
          buf[j].val = sorted_csr_values[lo + j];
        }
        ks_introsort_tk_csr_sort_entry((size_t)len, buf);
        for (int64_t j = 0; j < len; j++) {
          sorted_csr_tokens[lo + j] = buf[j].tok;
          sorted_csr_values[lo + j] = buf[j].val;
        }
      }
      free(buf);
    }
    {
      double *tmp = (double *)calloc(csr_n_tokens, sizeof(double));
      for (uint64_t s = 0; s < csr_n_samples; s++) {
        int64_t lo = csr_offsets_a[s], hi = csr_offsets_a[s + 1];
        for (int64_t j = lo; j < hi; j++)
          tmp[csr_tokens_a[j]] += csr_values_a[j];
        double ss = 0.0;
        for (int64_t j = lo; j < hi; j++) {
          int64_t tok = csr_tokens_a[j];
          if (tmp[tok] != 0.0) { ss += tmp[tok] * tmp[tok]; tmp[tok] = 0.0; }
        }
        csr_norms_a[s] = sqrt(ss);
      }
      free(tmp);
    }
  }

  tk_ivec_t *lm_ids = NULL;
  tk_dvec_t *lm_chol = NULL;
  double *full_chol = NULL;
  uint64_t nc, m;
  double trace_ratio;
  tk_spectral_sample_landmarks(L,
    dense_data, dense_d_input, dense_n_samples,
    bits_data, bits_d, bits_n_samples,
    csr_offsets_a, sorted_csr_tokens, sorted_csr_values, csr_norms_a, csr_n_samples, csr_n_tokens,
    kernel, dense_norms,
    n_lm_req, trace_tol,
    &lm_ids, &lm_chol, &full_chol, &nc, &m, &trace_ratio);
  free(sorted_csr_tokens); free(sorted_csr_values);
  int lm_ids_idx = lua_gettop(L);

  uint64_t d = m;

  if (m == 0) {
    if (lm_chol) tk_dvec_destroy(lm_chol);
    free(full_chol);
    free(csr_values_a); free(csr_norms_a);
    free(dense_fw_scaled); free(dense_norms);
    lua_pushnil(L);
    lua_pushnil(L);
    return 2;
  }

  tk_encode_nystrom_ctx_t *ctx = (tk_encode_nystrom_ctx_t *)
    lua_newuserdata(L, sizeof(tk_encode_nystrom_ctx_t));
  memset(ctx, 0, sizeof(*ctx));
  lua_newtable(L);
  lua_pushcfunction(L, tk_encode_nystrom_ctx_gc);
  lua_setfield(L, -2, "__gc");
  lua_setmetatable(L, -2);
  ctx->lm_chol = lm_chol;
  ctx->full_chol = full_chol;

  int64_t im = (int64_t)m;

  tk_fvec_t *train_codes = tk_fvec_create(L, nc * d, 0, 0);
  train_codes->n = nc * d;
  int train_codes_idx = lua_gettop(L);
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < nc; i++)
    for (uint64_t j = 0; j < d; j++)
      train_codes->a[i * d + j] = (float)full_chol[j * nc + i];

  ctx->projection = (double *)calloc(m * m, sizeof(double));
  for (uint64_t i = 0; i < m; i++) ctx->projection[i * m + i] = 1.0;
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
    (int)m, (int)m, 1.0, lm_chol->a, (int)m, ctx->projection, (int)m);
  tk_dvec_destroy(lm_chol); ctx->lm_chol = NULL;

  lua_getfield(L, 1, "label_offsets");
  int has_gram_labels = !lua_isnil(L, -1);
  tk_ivec_t *gram_lbl_off = has_gram_labels ? tk_ivec_peek(L, -1, "label_offsets") : NULL;
  lua_pop(L, 1);
  lua_getfield(L, 1, "label_neighbors");
  tk_ivec_t *gram_lbl_nbr = has_gram_labels ? tk_ivec_peek(L, -1, "label_neighbors") : NULL;
  lua_pop(L, 1);
  int64_t nl = 0;
  if (has_gram_labels)
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "encode", "n_labels");
  lua_getfield(L, 1, "targets");
  int has_gram_targets = !lua_isnil(L, -1);
  tk_dvec_t *gram_targets = has_gram_targets ? tk_dvec_peek(L, -1, "targets") : NULL;
  lua_pop(L, 1);
  if (has_gram_targets)
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "encode", "n_targets");

  int has_gram = has_gram_labels || has_gram_targets;
  int gram_idx = 0;

  if (has_gram) {
    uint64_t dd = (uint64_t)m * (uint64_t)m;
    uint64_t unl = (uint64_t)nl;
    uint64_t dnl = m * unl;
    double *XtX = (double *)calloc(dd, sizeof(double));
    double *eigenvals = (double *)malloc(m * sizeof(double));
    if (!XtX || !eigenvals) {
      free(XtX); free(eigenvals);
      return luaL_error(L, "encode: out of memory");
    }
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
      (int)m, (int)nc, 1.0, full_chol, (int)nc, 0.0, XtX, (int)m);

    double *xty = (double *)calloc(dnl, sizeof(double));
    if (!xty) { free(XtX); free(eigenvals); return luaL_error(L, "encode: out of memory"); }

    tk_dvec_t *lc = NULL;
    int lc_idx = 0;

    if (has_gram_labels) {
      int64_t nnz = (int64_t)gram_lbl_off->a[nc];
      int64_t *csc_off = (int64_t *)calloc((uint64_t)(nl + 1), sizeof(int64_t));
      int64_t *csc_rows = (int64_t *)malloc((uint64_t)nnz * sizeof(int64_t));
      if (!csc_off || !csc_rows) {
        free(csc_off); free(csc_rows); free(XtX); free(eigenvals); free(xty);
        return luaL_error(L, "encode: out of memory");
      }
      for (int64_t i = 0; i < nnz; i++)
        csc_off[gram_lbl_nbr->a[i] + 1]++;
      for (int64_t l = 0; l < nl; l++)
        csc_off[l + 1] += csc_off[l];
      int64_t *pos = (int64_t *)malloc((uint64_t)nl * sizeof(int64_t));
      memcpy(pos, csc_off, (uint64_t)nl * sizeof(int64_t));
      for (uint64_t s = 0; s < nc; s++)
        for (int64_t j = gram_lbl_off->a[s]; j < gram_lbl_off->a[s + 1]; j++) {
          int64_t l = gram_lbl_nbr->a[j];
          csc_rows[pos[l]++] = (int64_t)s;
        }
      free(pos);
      #pragma omp parallel for schedule(static)
      for (int64_t k = 0; k < im; k++) {
        double *col = full_chol + (uint64_t)k * nc;
        for (int64_t l = 0; l < nl; l++) {
          double s = 0.0;
          for (int64_t j = csc_off[l]; j < csc_off[l + 1]; j++)
            s += col[csc_rows[j]];
          xty[k * nl + l] = s;
        }
      }
      lc = tk_dvec_create(L, (uint64_t)nl, 0, 0);
      lc->n = (uint64_t)nl;
      lc_idx = lua_gettop(L);
      for (int64_t l = 0; l < nl; l++)
        lc->a[l] = (double)(csc_off[l + 1] - csc_off[l]);
      free(csc_off);
      free(csc_rows);
    } else {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)m, (int)nl, (int)nc, 1.0, full_chol, (int)nc,
        gram_targets->a, (int)nl, 0.0, xty, (int)nl);
    }

    double *col_mean = (double *)malloc(m * sizeof(double));
    double *y_mean_arr = (double *)malloc(unl * sizeof(double));
    if (!col_mean || !y_mean_arr) {
      free(col_mean); free(y_mean_arr); free(xty);
      free(XtX); free(eigenvals);
      return luaL_error(L, "encode: out of memory");
    }
    #pragma omp parallel for schedule(static)
    for (uint64_t j = 0; j < m; j++) {
      double s = 0.0;
      double *col = full_chol + j * nc;
      for (uint64_t i = 0; i < nc; i++) s += col[i];
      col_mean[j] = s / (double)nc;
    }
    free(full_chol); ctx->full_chol = NULL;

    cblas_dsyr(CblasColMajor, CblasUpper, (int)m,
      -(double)nc, col_mean, 1, XtX, (int)m);
    if (has_gram_labels) {
      for (int64_t l = 0; l < nl; l++)
        y_mean_arr[l] = lc->a[l] / (double)nc;
    } else {
      for (int64_t l = 0; l < nl; l++) {
        double s = 0.0;
        for (uint64_t i = 0; i < nc; i++)
          s += gram_targets->a[i * (uint64_t)nl + (uint64_t)l];
        y_mean_arr[l] = s / (double)nc;
      }
    }
    cblas_dger(CblasRowMajor, (int)m, (int)nl,
      -(double)nc, col_mean, 1, y_mean_arr, 1, xty, (int)nl);

    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (int)m, XtX, (int)m, eigenvals);
    double mean_eig = 0.0;
    for (uint64_t i = 0; i < m; i++)
      mean_eig += eigenvals[i];
    mean_eig /= (double)m;
    double *PQtY = (double *)malloc(dnl * sizeof(double));
    double *W_work = (double *)malloc(dnl * sizeof(double));
    if (!PQtY || !W_work) {
      free(PQtY); free(W_work); free(xty);
      free(XtX); free(eigenvals);
      free(col_mean); free(y_mean_arr);
      return luaL_error(L, "encode: out of memory");
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)m, (int)nl, (int)m, 1.0, XtX, (int)m,
      xty, (int)nl, 0.0, PQtY, (int)nl);
    free(xty);

    tk_gram_t *g = tk_lua_newuserdata(L, tk_gram_t,
      TK_GRAM_MT, tk_gram_mt_fns, tk_gram_gc);
    gram_idx = lua_gettop(L);
    g->evecs = XtX;
    g->eigenvals = eigenvals;
    g->PQtY = PQtY;
    g->label_counts = lc;
    g->W_work = W_work;
    g->W_work_f = NULL;
    g->sbuf_f = NULL;
    g->val_F_f = NULL;
    g->col_mean = col_mean;
    g->y_mean = y_mean_arr;
    g->cm_proj = NULL;
    g->intercept = NULL;
    g->mean_eig = mean_eig;
    g->n_dims = (int64_t)m;
    g->n_labels = nl;
    g->n_samples = (int64_t)nc;
    g->val_n = 0;
    g->destroyed = false;

    lua_newtable(L);
    if (lc_idx > 0) {
      lua_pushvalue(L, lc_idx);
      lua_setfield(L, -2, "label_counts");
    }
    lua_setfenv(L, gram_idx);
  } else {
    free(full_chol); ctx->full_chol = NULL;
  }

  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  {
    uint64_t proj_n = m * d;
    enc->projection = (float *)malloc(proj_n * sizeof(float));
    for (uint64_t i = 0; i < proj_n; i++)
      enc->projection[i] = (float)ctx->projection[i];
    free(ctx->projection); ctx->projection = NULL;
  }
  enc->m = m;
  enc->d = d;
  enc->trace_ratio = trace_ratio;
  enc->destroyed = false;

  enc->lm_vecs = NULL;
  enc->lm_bits = NULL;
  enc->lm_csr_offsets = NULL;
  enc->lm_csr_tokens = NULL;
  enc->lm_csr_values = NULL;
  enc->lm_csr_norms = NULL;
  enc->lm_csc_offsets = NULL;
  enc->lm_csc_rows = NULL;
  enc->lm_csc_values = NULL;
  enc->feature_weights = NULL;
  enc->fw_len = 0;
  enc->lm_dense_norms = NULL;
  enc->d_input = 0;
  enc->bits_d = 0;
  enc->csr_n_tokens = 0;
  enc->kernel = kernel;
  if (has_csr) {
    uint64_t lm_total = 0;
    for (uint64_t j = 0; j < m; j++) {
      uint64_t si = (uint64_t)lm_ids->a[j];
      lm_total += (uint64_t)(csr_offsets_a[si + 1] - csr_offsets_a[si]);
    }
    enc->lm_csr_offsets = (int64_t *)malloc((m + 1) * sizeof(int64_t));
    enc->lm_csr_tokens = (int64_t *)malloc(lm_total * sizeof(int64_t));
    enc->lm_csr_values = (double *)malloc(lm_total * sizeof(double));
    enc->lm_csr_norms = (double *)malloc(m * sizeof(double));
    enc->lm_csr_offsets[0] = 0;
    for (uint64_t j = 0; j < m; j++) {
      uint64_t si = (uint64_t)lm_ids->a[j];
      int64_t lo = csr_offsets_a[si], hi = csr_offsets_a[si + 1];
      int64_t cnt = hi - lo;
      memcpy(enc->lm_csr_tokens + enc->lm_csr_offsets[j], csr_tokens_a + lo, (uint64_t)cnt * sizeof(int64_t));
      memcpy(enc->lm_csr_values + enc->lm_csr_offsets[j], csr_values_a + lo, (uint64_t)cnt * sizeof(double));
      enc->lm_csr_norms[j] = csr_norms_a[si];
      enc->lm_csr_offsets[j + 1] = enc->lm_csr_offsets[j] + cnt;
    }
    enc->lm_csc_offsets = (int64_t *)calloc(csr_n_tokens + 1, sizeof(int64_t));
    for (uint64_t i = 0; i < lm_total; i++)
      enc->lm_csc_offsets[enc->lm_csr_tokens[i] + 1]++;
    for (uint64_t t = 0; t < csr_n_tokens; t++)
      enc->lm_csc_offsets[t + 1] += enc->lm_csc_offsets[t];
    enc->lm_csc_rows = (int64_t *)malloc(lm_total * sizeof(int64_t));
    enc->lm_csc_values = (double *)malloc(lm_total * sizeof(double));
    int64_t *csc_pos = (int64_t *)malloc((csr_n_tokens + 1) * sizeof(int64_t));
    memcpy(csc_pos, enc->lm_csc_offsets, (csr_n_tokens + 1) * sizeof(int64_t));
    for (uint64_t j = 0; j < m; j++) {
      for (int64_t a = enc->lm_csr_offsets[j]; a < enc->lm_csr_offsets[j + 1]; a++) {
        int64_t tok = enc->lm_csr_tokens[a];
        int64_t p = csc_pos[tok]++;
        enc->lm_csc_rows[p] = (int64_t)j;
        enc->lm_csc_values[p] = enc->lm_csr_values[a];
      }
    }
    free(csc_pos);
    enc->csr_n_tokens = csr_n_tokens;
    enc->dense_mode = 3;
    if (fw_dv) {
      enc->feature_weights = (double *)malloc(fw_dv->n * sizeof(double));
      memcpy(enc->feature_weights, fw_dv->a, fw_dv->n * sizeof(double));
      enc->fw_len = fw_dv->n;
    }
    free(csr_values_a); free(csr_norms_a);
  } else if (bits_data != NULL) {
    uint64_t row_bytes = TK_CVEC_BITS_BYTES(bits_d);
    uint8_t *lmb = (uint8_t *)malloc(m * row_bytes);
    for (uint64_t j = 0; j < m; j++)
      memcpy(lmb + j * row_bytes, bits_data + (uint64_t)lm_ids->a[j] * row_bytes, row_bytes);
    enc->lm_bits = lmb;
    enc->bits_d = bits_d;
    enc->dense_mode = 2;
  } else {
    double *lmv = (double *)malloc(m * (uint64_t)dense_d_input * sizeof(double));
    for (uint64_t j = 0; j < m; j++) {
      int64_t row = lm_ids->a[j];
      memcpy(lmv + j * (uint64_t)dense_d_input, dense_data + (uint64_t)row * (uint64_t)dense_d_input,
             (uint64_t)dense_d_input * sizeof(double));
    }
    enc->lm_vecs = lmv;
    enc->dense_mode = 1;
    enc->d_input = dense_d_input;
    enc->lm_dense_norms = (double *)malloc(m * sizeof(double));
    for (uint64_t j = 0; j < m; j++)
      enc->lm_dense_norms[j] = cblas_dnrm2((int)dense_d_input,
        enc->lm_vecs + j * (uint64_t)dense_d_input, 1);
    if (fw_dv) {
      enc->feature_weights = (double *)malloc(fw_dv->n * sizeof(double));
      memcpy(enc->feature_weights, fw_dv->a, fw_dv->n * sizeof(double));
      enc->fw_len = fw_dv->n;
    }
    free(dense_fw_scaled);
    free(dense_norms);
  }

  lua_newtable(L);
  lua_pushvalue(L, lm_ids_idx);
  lua_setfield(L, -2, "landmark_ids");
  lua_setfenv(L, enc_idx);

  lua_pushvalue(L, train_codes_idx);
  lua_pushvalue(L, enc_idx);
  if (has_gram) {
    lua_pushvalue(L, gram_idx);
    return 3;
  }
  return 2;
}

static inline int tk_nystrom_encoder_load_lua (lua_State *L) {
  lua_settop(L, 2);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && lua_toboolean(L, 2);
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
  if (version != 17) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported nystrom encoder version %d (expected 17)", (int)version);
  }
  uint8_t dense_mode;
  tk_lua_fread(L, &dense_mode, sizeof(uint8_t), 1, fh);
  uint8_t kernel_byte;
  tk_lua_fread(L, &kernel_byte, sizeof(uint8_t), 1, fh);
  tk_spectral_kernel_t kernel = (tk_spectral_kernel_t)kernel_byte;
  uint64_t m, d;
  double trace_ratio;
  tk_lua_fread(L, &m, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &d, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &trace_ratio, sizeof(double), 1, fh);
  float *projection = (float *)malloc(m * d * sizeof(float));
  if (!projection) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "nystrom load: out of memory");
  }
  tk_lua_fread(L, projection, sizeof(float), m * d, fh);
  uint64_t fw_len;
  double *feature_weights = NULL;
  tk_lua_fread(L, &fw_len, sizeof(uint64_t), 1, fh);
  if (fw_len > 0) {
    feature_weights = (double *)malloc(fw_len * sizeof(double));
    tk_lua_fread(L, feature_weights, sizeof(double), fw_len, fh);
  }
  double *lm_vecs = NULL;
  uint8_t *lm_bits = NULL;
  int64_t *lm_csr_offsets = NULL;
  int64_t *lm_csr_tokens = NULL;
  double *lm_csr_values = NULL;
  double *lm_csr_norms = NULL;
  double *lm_dense_norms = NULL;
  int64_t d_input = 0;
  uint64_t bits_d = 0;
  uint64_t csr_n_tokens = 0;
  if (dense_mode == 2) {
    tk_lua_fread(L, &bits_d, sizeof(uint64_t), 1, fh);
    uint64_t row_bytes = TK_CVEC_BITS_BYTES(bits_d);
    lm_bits = (uint8_t *)malloc(m * row_bytes);
    if (!lm_bits) {
      free(projection); free(feature_weights);
      tk_lua_fclose(L, fh);
      return luaL_error(L, "nystrom load: out of memory");
    }
    tk_lua_fread(L, lm_bits, 1, m * row_bytes, fh);
  } else if (dense_mode == 1) {
    tk_lua_fread(L, &d_input, sizeof(int64_t), 1, fh);
    lm_vecs = (double *)malloc(m * (uint64_t)d_input * sizeof(double));
    if (!lm_vecs) {
      free(projection); free(feature_weights);
      tk_lua_fclose(L, fh);
      return luaL_error(L, "nystrom load: out of memory");
    }
    tk_lua_fread(L, lm_vecs, sizeof(double), m * (uint64_t)d_input, fh);
    lm_dense_norms = (double *)malloc(m * sizeof(double));
    tk_lua_fread(L, lm_dense_norms, sizeof(double), m, fh);
  } else if (dense_mode == 3) {
    tk_lua_fread(L, &csr_n_tokens, sizeof(uint64_t), 1, fh);
    lm_csr_offsets = (int64_t *)malloc((m + 1) * sizeof(int64_t));
    tk_lua_fread(L, lm_csr_offsets, sizeof(int64_t), m + 1, fh);
    uint64_t total_csr = (uint64_t)lm_csr_offsets[m];
    lm_csr_tokens = (int64_t *)malloc(total_csr * sizeof(int64_t));
    lm_csr_values = (double *)malloc(total_csr * sizeof(double));
    lm_csr_norms = (double *)malloc(m * sizeof(double));
    tk_lua_fread(L, lm_csr_tokens, sizeof(int64_t), total_csr, fh);
    tk_lua_fread(L, lm_csr_values, sizeof(double), total_csr, fh);
    tk_lua_fread(L, lm_csr_norms, sizeof(double), m, fh);
  }
  tk_ivec_load(L, fh);
  int lm_ids_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);
  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->projection = projection;
  enc->lm_vecs = lm_vecs;
  enc->lm_bits = lm_bits;
  enc->lm_csr_offsets = lm_csr_offsets;
  enc->lm_csr_tokens = lm_csr_tokens;
  enc->lm_csr_values = lm_csr_values;
  enc->lm_csr_norms = lm_csr_norms;
  enc->lm_csc_offsets = NULL;
  enc->lm_csc_rows = NULL;
  enc->lm_csc_values = NULL;
  enc->feature_weights = feature_weights;
  enc->fw_len = fw_len;
  enc->lm_dense_norms = lm_dense_norms;
  enc->m = m;
  enc->d = d;
  enc->d_input = d_input;
  enc->bits_d = bits_d;
  enc->csr_n_tokens = csr_n_tokens;
  enc->trace_ratio = trace_ratio;
  enc->dense_mode = dense_mode;
  enc->kernel = kernel;
  enc->destroyed = false;
  if (dense_mode == 3 && lm_csr_offsets) {
    uint64_t lm_total = (uint64_t)lm_csr_offsets[m];
    enc->lm_csc_offsets = (int64_t *)calloc(csr_n_tokens + 1, sizeof(int64_t));
    for (uint64_t i = 0; i < lm_total; i++)
      enc->lm_csc_offsets[lm_csr_tokens[i] + 1]++;
    for (uint64_t t = 0; t < csr_n_tokens; t++)
      enc->lm_csc_offsets[t + 1] += enc->lm_csc_offsets[t];
    enc->lm_csc_rows = (int64_t *)malloc(lm_total * sizeof(int64_t));
    enc->lm_csc_values = (double *)malloc(lm_total * sizeof(double));
    int64_t *csc_pos = (int64_t *)malloc((csr_n_tokens + 1) * sizeof(int64_t));
    memcpy(csc_pos, enc->lm_csc_offsets, (csr_n_tokens + 1) * sizeof(int64_t));
    for (uint64_t j = 0; j < m; j++) {
      for (int64_t a = lm_csr_offsets[j]; a < lm_csr_offsets[j + 1]; a++) {
        int64_t tok = lm_csr_tokens[a];
        int64_t p = csc_pos[tok]++;
        enc->lm_csc_rows[p] = (int64_t)j;
        enc->lm_csc_values[p] = lm_csr_values[a];
      }
    }
    free(csc_pos);
  }
  lua_newtable(L);
  lua_pushvalue(L, lm_ids_idx);
  lua_setfield(L, -2, "landmark_ids");
  lua_setfenv(L, enc_idx);
  lua_pushvalue(L, enc_idx);
  return 1;
}

static inline int tk_spectral_gram_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 1, "gram", "n_samples");
  int64_t d = (int64_t)tk_lua_fcheckunsigned(L, 1, "gram", "n_dims");
  lua_getfield(L, 1, "codes");
  tk_fvec_t *codes = tk_fvec_peek(L, -1, "codes");
  lua_pop(L, 1);
  float *X = codes->a;

  lua_getfield(L, 1, "label_offsets");
  int has_labels = !lua_isnil(L, -1);
  tk_ivec_t *lbl_off = has_labels ? tk_ivec_peek(L, -1, "label_offsets") : NULL;
  lua_pop(L, 1);
  lua_getfield(L, 1, "label_neighbors");
  tk_ivec_t *lbl_nbr = has_labels ? tk_ivec_peek(L, -1, "label_neighbors") : NULL;
  lua_pop(L, 1);
  int64_t nl = 0;
  if (has_labels)
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "gram", "n_labels");
  lua_getfield(L, 1, "targets");
  int has_targets = !lua_isnil(L, -1);
  tk_dvec_t *targets = has_targets ? tk_dvec_peek(L, -1, "targets") : NULL;
  lua_pop(L, 1);
  if (has_targets)
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "gram", "n_targets");
  if (!has_labels && !has_targets)
    return luaL_error(L, "gram: need label_offsets/label_neighbors or targets");

  double *F = (double *)malloc((uint64_t)n * (uint64_t)d * sizeof(double));
  if (!F) return luaL_error(L, "gram: out of memory");
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < d; j++)
      F[j * n + i] = (double)X[i * d + j];

  uint64_t dd = (uint64_t)d * (uint64_t)d;
  uint64_t unl = (uint64_t)nl;
  uint64_t dnl = (uint64_t)d * unl;
  double *XtX = (double *)calloc(dd, sizeof(double));
  double *eigenvals = (double *)malloc((uint64_t)d * sizeof(double));
  if (!XtX || !eigenvals) {
    free(XtX); free(eigenvals); free(F);
    return luaL_error(L, "gram: out of memory");
  }
  cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
    (int)d, (int)n, 1.0, F, (int)n, 0.0, XtX, (int)d);

  double *xty = (double *)calloc(dnl, sizeof(double));
  if (!xty) { free(XtX); free(eigenvals); free(F); return luaL_error(L, "gram: out of memory"); }

  tk_dvec_t *lc = NULL;
  int lc_idx = 0;

  if (has_labels) {
    int64_t nnz = (int64_t)lbl_off->a[n];
    int64_t *csc_off = (int64_t *)calloc((uint64_t)(nl + 1), sizeof(int64_t));
    int64_t *csc_rows = (int64_t *)malloc((uint64_t)nnz * sizeof(int64_t));
    if (!csc_off || !csc_rows) {
      free(csc_off); free(csc_rows); free(XtX); free(eigenvals); free(xty); free(F);
      return luaL_error(L, "gram: out of memory");
    }
    for (int64_t i = 0; i < nnz; i++)
      csc_off[lbl_nbr->a[i] + 1]++;
    for (int64_t l = 0; l < nl; l++)
      csc_off[l + 1] += csc_off[l];
    int64_t *pos = (int64_t *)malloc((uint64_t)nl * sizeof(int64_t));
    memcpy(pos, csc_off, (uint64_t)nl * sizeof(int64_t));
    for (int64_t s = 0; s < n; s++)
      for (int64_t j = lbl_off->a[s]; j < lbl_off->a[s + 1]; j++) {
        int64_t l = lbl_nbr->a[j];
        csc_rows[pos[l]++] = s;
      }
    free(pos);
    lc = tk_dvec_create(L, unl, 0, 0);
    lc->n = unl;
    lc_idx = lua_gettop(L);
    for (int64_t l = 0; l < nl; l++)
      lc->a[l] = (double)(csc_off[l + 1] - csc_off[l]);
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < d; k++) {
      double *col = F + (uint64_t)k * (uint64_t)n;
      for (int64_t l = 0; l < nl; l++) {
        double s = 0.0;
        for (int64_t j = csc_off[l]; j < csc_off[l + 1]; j++)
          s += col[csc_rows[j]];
        xty[k * nl + l] = s;
      }
    }
    free(csc_off);
    free(csc_rows);
  } else {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)d, (int)nl, (int)n, 1.0, F, (int)n,
      targets->a, (int)nl, 0.0, xty, (int)nl);
  }
  double *col_mean = (double *)malloc((uint64_t)d * sizeof(double));
  double *y_mean_arr = (double *)malloc(unl * sizeof(double));
  if (!col_mean || !y_mean_arr) {
    free(col_mean); free(y_mean_arr); free(xty);
    free(XtX); free(eigenvals); free(F);
    return luaL_error(L, "gram: out of memory");
  }
  #pragma omp parallel for schedule(static)
  for (int64_t j = 0; j < d; j++) {
    double s = 0.0;
    double *col = F + (uint64_t)j * (uint64_t)n;
    for (int64_t i = 0; i < n; i++) s += col[i];
    col_mean[j] = s / (double)n;
  }
  free(F);

  cblas_dsyr(CblasColMajor, CblasUpper, (int)d,
    -(double)n, col_mean, 1, XtX, (int)d);
  if (has_labels) {
    for (int64_t l = 0; l < nl; l++)
      y_mean_arr[l] = lc->a[l] / (double)n;
  } else {
    for (int64_t l = 0; l < nl; l++) {
      double s = 0.0;
      for (int64_t i = 0; i < n; i++)
        s += targets->a[(uint64_t)i * (uint64_t)nl + (uint64_t)l];
      y_mean_arr[l] = s / (double)n;
    }
  }
  cblas_dger(CblasRowMajor, (int)d, (int)nl,
    -(double)n, col_mean, 1, y_mean_arr, 1, xty, (int)nl);

  LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (int)d, XtX, (int)d, eigenvals);
  double mean_eig = 0.0;
  for (int64_t i = 0; i < d; i++)
    mean_eig += eigenvals[i];
  mean_eig /= (double)d;
  double *PQtY = (double *)malloc(dnl * sizeof(double));
  double *W_work = (double *)malloc(dnl * sizeof(double));
  if (!PQtY || !W_work) {
    free(PQtY); free(W_work); free(xty);
    free(XtX); free(eigenvals);
    free(col_mean); free(y_mean_arr);
    return luaL_error(L, "gram: out of memory");
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)d, (int)nl, (int)d, 1.0, XtX, (int)d,
    xty, (int)nl, 0.0, PQtY, (int)nl);
  free(xty);

  tk_gram_t *g = tk_lua_newuserdata(L, tk_gram_t,
    TK_GRAM_MT, tk_gram_mt_fns, tk_gram_gc);
  int gram_idx = lua_gettop(L);
  g->evecs = XtX;
  g->eigenvals = eigenvals;
  g->PQtY = PQtY;
  g->label_counts = lc;
  g->W_work = W_work;
  g->W_work_f = NULL;
  g->sbuf_f = NULL;
  g->val_F_f = NULL;
  g->col_mean = col_mean;
  g->y_mean = y_mean_arr;
  g->cm_proj = NULL;
  g->intercept = NULL;
  g->mean_eig = mean_eig;
  g->n_dims = d;
  g->n_labels = nl;
  g->n_samples = n;
  g->val_n = 0;
  g->destroyed = false;

  lua_newtable(L);
  if (lc_idx > 0) {
    lua_pushvalue(L, lc_idx);
    lua_setfield(L, -2, "label_counts");
  }
  lua_setfenv(L, gram_idx);

  lua_pushvalue(L, gram_idx);
  return 1;
}

static luaL_Reg tm_fns[] =
{
  { "encode", tm_encode },
  { "gram", tk_spectral_gram_lua },
  { "load", tk_nystrom_encoder_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_spectral (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_fns, 0);
  return 1;
}
