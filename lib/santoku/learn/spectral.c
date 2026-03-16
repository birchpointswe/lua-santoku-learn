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
  TK_SPECTRAL_COSINE = 0,
  TK_SPECTRAL_NNGP = 1,
  TK_SPECTRAL_NTK = 2,
  TK_SPECTRAL_EXPCOS = 3,
  TK_SPECTRAL_GEOLAPLACE = 4,
} tk_spectral_kernel_t;

static inline double tk_spectral_kernel_apply (tk_spectral_kernel_t k, double raw, double denom) {
  double c = denom > 1e-15 ? raw / denom : 0.0;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;
  if (k == TK_SPECTRAL_NNGP) {
    double t = acos(c);
    return denom * (sin(t) + (M_PI - t) * c) / M_PI;
  }
  if (k == TK_SPECTRAL_NTK) {
    double t = acos(c);
    return raw * (1.0 - t / M_PI) + denom * (sin(t) + (M_PI - t) * c) / M_PI;
  }
  if (k == TK_SPECTRAL_EXPCOS)
    return exp(c - 1.0);
  if (k == TK_SPECTRAL_GEOLAPLACE) {
    double d2 = 2.0 * (1.0 - c);
    return exp(-sqrt(d2 > 0.0 ? d2 : 0.0));
  }
  return c;
}

#define TK_MOD_CSR   0
#define TK_MOD_DENSE 1
#define TK_MOD_BITS  2
#define TK_MERGE_MEAN    0
#define TK_MERGE_PRODUCT 1
#define TK_MAX_MOD 8

typedef struct {
  uint8_t type;
  const int64_t *csr_offsets;
  const int64_t *csr_tokens;
  const float *csr_values;
  const double *csr_norms;
  uint64_t csr_n_tokens;
  const double *dense;
  const double *dense_norms;
  int64_t d_input;
  const uint8_t *bits_data;
  uint64_t bits_d;
} tk_spectral_modality_t;

typedef struct {
  int64_t *sid_map;
  double *residual;
  float *L_mat;
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

#define TK_CHOL_BLOCK 64

static inline void tk_spectral_sample_landmarks (
  lua_State *L,
  tk_spectral_modality_t *mod,
  uint64_t n_samples,
  tk_spectral_kernel_t kernel,
  uint64_t n_landmarks,
  double trace_tol,
  tk_ivec_t **ids_out,
  tk_fvec_t **chol_out,
  float **full_chol_out,
  uint64_t *n_docs_out,
  uint64_t *actual_landmarks_out,
  double *trace_ratio_out
) {
  uint64_t n_docs = n_samples;

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
  ctx->L_mat = (float *)calloc(n_landmarks * n_docs, sizeof(float));
  ctx->landmark_sids = (int64_t *)malloc(n_landmarks * sizeof(int64_t));
  ctx->landmark_idx_map = (int64_t *)malloc(n_landmarks * sizeof(int64_t));
  if (!ctx->sid_map || !ctx->residual || !ctx->L_mat || !ctx->landmark_sids ||
      !ctx->landmark_idx_map) {
    luaL_error(L, "sample_landmarks: out of memory");
    return;
  }

  int64_t *sid_map = ctx->sid_map;
  double *residual = ctx->residual;
  float *L_mat = ctx->L_mat;
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

  float *kip_block = (float *)malloc(n_docs * TK_CHOL_BLOCK * sizeof(float));
  float *cross_dots = (float *)malloc(n_docs * TK_CHOL_BLOCK * sizeof(float));
  float *pivot_prev_L = (float *)malloc(TK_CHOL_BLOCK * n_landmarks * sizeof(float));
  int64_t max_d_input = 0;
  if (mod->type == TK_MOD_DENSE && mod->d_input > max_d_input)
    max_d_input = mod->d_input;
  double *pivot_dense_rows = max_d_input > 0
    ? (double *)malloc(TK_CHOL_BLOCK * (uint64_t)max_d_input * sizeof(double)) : NULL;
  uint64_t blk_pivots[TK_CHOL_BLOCK];

  double *proposal = (double *)malloc(n_docs * sizeof(double));
  double *proposal_cdf = (double *)malloc(n_docs * sizeof(double));

  int64_t *piv_csc_off = NULL;
  uint8_t *piv_csc_piv = NULL;
  float *piv_csc_val = NULL;
  int64_t *piv_csc_pos = NULL;
  uint64_t piv_csc_cap = 0;
  if (mod->type == TK_MOD_CSR) {
    uint64_t csr_n_tokens = mod->csr_n_tokens;
    piv_csc_off = (int64_t *)calloc(csr_n_tokens + 1, sizeof(int64_t));
    piv_csc_pos = (int64_t *)malloc(csr_n_tokens * sizeof(int64_t));
    uint64_t max_nnz = 0;
    for (uint64_t i = 0; i < n_docs; i++) {
      uint64_t nnz = (uint64_t)(mod->csr_offsets[i + 1] - mod->csr_offsets[i]);
      if (nnz > max_nnz) max_nnz = nnz;
    }
    piv_csc_cap = TK_CHOL_BLOCK * max_nnz;
    piv_csc_piv = (uint8_t *)malloc(piv_csc_cap);
    piv_csc_val = (float *)malloc(piv_csc_cap * sizeof(float));
  }

  double *dense_kip = NULL;
  if (mod->type == TK_MOD_DENSE)
    dense_kip = (double *)malloc(n_docs * TK_CHOL_BLOCK * sizeof(double));

  if (!kip_block || !cross_dots || !pivot_prev_L
      || (max_d_input > 0 && !pivot_dense_rows)
      || !proposal || !proposal_cdf
      || (mod->type == TK_MOD_CSR && (!piv_csc_off || !piv_csc_piv || !piv_csc_val || !piv_csc_pos))
      || (mod->type == TK_MOD_DENSE && !dense_kip)) {
    free(kip_block); free(cross_dots);
    free(pivot_prev_L);
    free(pivot_dense_rows);
    free(proposal); free(proposal_cdf);
    free(piv_csc_off); free(piv_csc_piv); free(piv_csc_val); free(piv_csc_pos);
    free(dense_kip);
    luaL_error(L, "sample_landmarks: out of memory (buffers)");
    return;
  }

  #pragma omp parallel for reduction(+:initial_trace)
  for (uint64_t i = 0; i < n_docs; i++) {
    double diag = 1.0;
    if (kernel != TK_SPECTRAL_COSINE && kernel != TK_SPECTRAL_EXPCOS && kernel != TK_SPECTRAL_GEOLAPLACE) {
      double n2 = 1.0;
      if (mod->type == TK_MOD_CSR)
        n2 = mod->csr_norms[i] * mod->csr_norms[i];
      else if (mod->type == TK_MOD_DENSE)
        n2 = mod->dense_norms[i] * mod->dense_norms[i];
      if (kernel == TK_SPECTRAL_NTK) diag = 2.0 * n2;
      else diag = n2;
    }
    residual[i] = diag;
    initial_trace += diag;
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

    memset(kip_block, 0, n_docs * np * sizeof(float));

    if (mod->type == TK_MOD_CSR) {
      const int64_t *csr_offsets = mod->csr_offsets;
      const int64_t *csr_tokens = mod->csr_tokens;
      const float *csr_values = mod->csr_values;
      const double *csr_norms = mod->csr_norms;
      uint64_t csr_n_tokens = mod->csr_n_tokens;
      int64_t plo_arr[TK_CHOL_BLOCK], phi_arr[TK_CHOL_BLOCK];
      for (uint64_t b = 0; b < np; b++) {
        plo_arr[b] = csr_offsets[blk_pivots[b]];
        phi_arr[b] = csr_offsets[blk_pivots[b] + 1];
      }
      {
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
        if (total_pnnz > piv_csc_cap) {
          piv_csc_cap = total_pnnz;
          piv_csc_piv = (uint8_t *)realloc(piv_csc_piv, piv_csc_cap);
          piv_csc_val = (float *)realloc(piv_csc_val, piv_csc_cap * sizeof(float));
        }
        memcpy(piv_csc_pos, piv_csc_off, csr_n_tokens * sizeof(int64_t));
        for (uint64_t b = 0; b < np; b++) {
          int64_t p = plo_arr[b];
          while (p < phi_arr[b]) {
            int64_t tok = csr_tokens[p];
            float sum = 0.0f;
            while (p < phi_arr[b] && csr_tokens[p] == tok) { sum += csr_values[p]; p++; }
            int64_t pos = piv_csc_pos[tok]++;
            piv_csc_piv[pos] = (uint8_t)b;
            piv_csc_val[pos] = sum;
          }
        }
        const int64_t *restrict pc_off = piv_csc_off;
        const uint8_t *restrict pc_piv = piv_csc_piv;
        const float *restrict pc_val = piv_csc_val;
        const int64_t *restrict c_off = csr_offsets;
        const int64_t *restrict c_tok = csr_tokens;
        const float *restrict c_val = csr_values;
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++) {
          double kip_row[TK_CHOL_BLOCK];
          memset(kip_row, 0, np * sizeof(double));
          int64_t jlo = c_off[i], jhi = c_off[i + 1];
          for (int64_t j = jlo; j < jhi; j++) {
            int64_t tok = c_tok[j];
            double val = (double)c_val[j];
            int64_t clo = pc_off[tok], chi = pc_off[tok + 1];
            for (int64_t c = clo; c < chi; c++)
              kip_row[pc_piv[c]] += val * (double)pc_val[c];
          }
          for (uint64_t b = 0; b < np; b++)
            kip_block[i * np + b] = (float)kip_row[b];
        }
        memset(piv_csc_off, 0, (csr_n_tokens + 1) * sizeof(int64_t));
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
              kip_block[i * np + b], denom);
          }
      }

    } else if (mod->type == TK_MOD_BITS) {
      uint64_t bd = mod->bits_d;
      uint64_t row_bytes = TK_CVEC_BITS_BYTES(bd);
      const uint8_t *bdata = mod->bits_data;
      double inv_bd = 1.0 / (double)bd;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_docs; i++) {
        const uint8_t *row_i = bdata + i * row_bytes;
        for (uint64_t b = 0; b < np; b++) {
          uint64_t ham = tk_cvec_bits_hamming_serial(
            row_i, bdata + blk_pivots[b] * row_bytes, bd);
          kip_block[i * np + b] = 1.0 - (double)ham * inv_bd;
        }
      }

    } else if (mod->type == TK_MOD_DENSE) {
      int64_t di = mod->d_input;
      const double *dense = mod->dense;
      const double *dnorms = mod->dense_norms;
      for (uint64_t b = 0; b < np; b++)
        memcpy(pivot_dense_rows + b * (uint64_t)di,
               dense + blk_pivots[b] * (uint64_t)di,
               (uint64_t)di * sizeof(double));
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        (int)n_docs, (int)np, (int)di, 1.0,
        dense, (int)di,
        pivot_dense_rows, (int)di,
        0.0, dense_kip, (int)np);
      {
        double pnorms[TK_CHOL_BLOCK];
        for (uint64_t b = 0; b < np; b++)
          pnorms[b] = dnorms[blk_pivots[b]];
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++)
          for (uint64_t b = 0; b < np; b++) {
            double denom = dnorms[i] * pnorms[b];
            kip_block[i * np + b] = (float)tk_spectral_kernel_apply(kernel,
              dense_kip[i * np + b], denom);
          }
      }
    }

    uint64_t jb = actual_landmarks;
    if (jb > 0) {
      for (uint64_t k = 0; k < jb; k++)
        for (uint64_t b = 0; b < np; b++)
          pivot_prev_L[b * jb + k] = L_mat[k * n_docs + blk_pivots[b]];
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        (int)n_docs, (int)np, (int)jb, 1.0f,
        L_mat, (int)n_docs,
        pivot_prev_L, (int)jb,
        0.0f, cross_dots, (int)n_docs);
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
  free(piv_csc_off);
  free(piv_csc_piv);
  free(piv_csc_val);
  free(piv_csc_pos);
  free(dense_kip);

  tk_ivec_t *landmark_ids = tk_ivec_create(L, actual_landmarks, 0, 0);
  for (uint64_t i = 0; i < actual_landmarks; i++)
    landmark_ids->a[i] = landmark_sids[i];
  landmark_ids->n = actual_landmarks;

  tk_fvec_t *chol = tk_fvec_create(NULL, actual_landmarks * actual_landmarks, 0, 0);
  #pragma omp parallel for schedule(static)
  for (uint64_t li = 0; li < actual_landmarks; li++) {
    uint64_t doc_idx = (uint64_t)landmark_idx_map[li];
    for (uint64_t k = 0; k < actual_landmarks; k++)
      chol->a[li * actual_landmarks + k] = L_mat[k * n_docs + doc_idx];
  }
  chol->n = actual_landmarks * actual_landmarks;

  if (actual_landmarks > 0 && actual_landmarks < n_landmarks) {
    float *shrunk = realloc(L_mat, actual_landmarks * n_docs * sizeof(float));
    if (shrunk) L_mat = shrunk;
  }

  float *full_chol = L_mat;
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
  float *projection;
  tk_fvec_t *lm_chol;
  float *full_chol;
} tk_encode_nystrom_ctx_t;

static inline int tk_encode_nystrom_ctx_gc (lua_State *L) {
  tk_encode_nystrom_ctx_t *c = (tk_encode_nystrom_ctx_t *)lua_touserdata(L, 1);
  free(c->projection);
  if (c->lm_chol) tk_fvec_destroy(c->lm_chol);
  free(c->full_chol);
  return 0;
}

#define TK_NYSTROM_ENCODER_MT "tk_nystrom_encoder_t"

typedef struct {
  float *projection;
  uint64_t m;
  uint64_t d;
  double trace_ratio;
  tk_spectral_kernel_t kernel;
  uint8_t mod_type;
  int64_t *csr_offsets;
  int64_t *csr_tokens;
  float *csr_values;
  float *csr_norms;
  int64_t *csc_offsets;
  int64_t *csc_rows;
  float *csc_values;
  uint64_t csr_n_tokens;
  float *dense_vecs;
  float *dense_norms;
  int64_t d_input;
  uint8_t *bits_data;
  uint64_t bits_d;
  bool destroyed;
} tk_nystrom_encoder_t;

static inline tk_nystrom_encoder_t *tk_nystrom_encoder_peek (lua_State *L, int i) {
  return (tk_nystrom_encoder_t *)luaL_checkudata(L, i, TK_NYSTROM_ENCODER_MT);
}

static inline int tk_nystrom_encoder_gc (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (!enc->destroyed) {
    free(enc->projection);
    if (enc->mod_type == TK_MOD_CSR) {
      free(enc->csr_offsets);
      free(enc->csr_tokens);
      free(enc->csr_values);
      free(enc->csr_norms);
      free(enc->csc_offsets);
      free(enc->csc_rows);
      free(enc->csc_values);
    } else if (enc->mod_type == TK_MOD_DENSE) {
      free(enc->dense_vecs);
      free(enc->dense_norms);
    } else if (enc->mod_type == TK_MOD_BITS) {
      free(enc->bits_data);
    }
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

  luaL_checktype(L, 2, LUA_TTABLE);
  lua_getfield(L, 2, "n_samples");
  uint64_t n_samples = (uint64_t)luaL_checkinteger(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 2, "offsets");
  tk_ivec_t *in_offsets = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 2, "tokens");
  tk_ivec_t *in_tokens = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 2, "values");
  tk_fvec_t *in_values_f = tk_fvec_peekopt(L, -1);
  tk_dvec_t *in_values_d = in_values_f ? NULL : tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 2, "codes");
  tk_dvec_t *in_codes_dv = tk_dvec_peekopt(L, -1);
  tk_fvec_t *in_codes_fv = in_codes_dv ? NULL : tk_fvec_peekopt(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 2, "d_input");
  int64_t in_d_input_scalar = lua_isnumber(L, -1) ? (int64_t)lua_tointeger(L, -1) : 0;
  lua_pop(L, 1);
  lua_getfield(L, 2, "bits");
  tk_cvec_t *in_bits = tk_cvec_peekopt(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 2, "output");
  tk_fvec_t *out_fv = tk_fvec_peekopt(L, -1);
  int out_fv_idx = out_fv ? lua_gettop(L) : 0;
  if (!out_fv) lua_pop(L, 1);

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

  uint64_t tile = 4096;
  while (tile > 1 && tile * m * sizeof(float) > 256ULL * 1024 * 1024)
    tile /= 2;
  if (tile > n_samples) tile = n_samples;
  float *sims_f = (float *)malloc(tile * m * sizeof(float));
  if (!sims_f) return luaL_error(L, "encode: out of memory");

  float *csr_sval = NULL;
  const float *csr_sv = NULL;
  float *csr_norms = NULL;
  if (enc->mod_type == TK_MOD_CSR) {
    if (!in_offsets) { free(sims_f); return luaL_error(L, "encode: CSR modality but no offsets"); }
    uint64_t nnz = in_tokens->n;
    if (!in_values_f) {
      csr_sval = (float *)malloc(nnz * sizeof(float));
      if (in_values_d)
        for (uint64_t i = 0; i < nnz; i++) csr_sval[i] = (float)in_values_d->a[i];
      else
        for (uint64_t i = 0; i < nnz; i++) csr_sval[i] = 1.0f;
      csr_sv = csr_sval;
    } else {
      csr_sv = in_values_f->a;
    }
    csr_norms = (float *)calloc(n_samples, sizeof(float));
    #pragma omp parallel for schedule(static)
    for (uint64_t s = 0; s < n_samples; s++) {
      int64_t lo = in_offsets->a[s], hi = in_offsets->a[s + 1];
      double ss = 0.0;
      for (int64_t j = lo; j < hi; j++) {
        double v = (double)csr_sv[j];
        ss += v * v;
      }
      csr_norms[s] = (float)sqrt(ss);
    }
  }

  int nt = omp_get_max_threads();
  float *row_bufs = NULL;
  if (enc->mod_type == TK_MOD_CSR) {
    row_bufs = (float *)calloc((uint64_t)nt * m, sizeof(float));
    if (!row_bufs) {
      free(sims_f); free(csr_sval); free(csr_norms);
      return luaL_error(L, "encode: out of memory (row_bufs)");
    }
  }

  for (uint64_t base = 0; base < n_samples; base += tile) {
    uint64_t blk = base + tile <= n_samples ? tile : n_samples - base;

    if (enc->mod_type == TK_MOD_CSR) {
      const int64_t *off_a = in_offsets->a;
      const int64_t *tok_a = in_tokens->a;
      const int64_t *restrict csc_off = enc->csc_offsets;
      const int64_t *restrict csc_rows_a = enc->csc_rows;
      const float *restrict csc_vals = enc->csc_values;
      #pragma omp parallel
      {
        float *row_buf = row_bufs + (uint64_t)omp_get_thread_num() * m;
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < blk; i++) {
          float *restrict sims_row = sims_f + i * m;
          uint64_t si = base + i;
          int64_t jlo = off_a[si], jhi = off_a[si + 1];
          for (int64_t j = jlo; j < jhi; j++) {
            int64_t tok = tok_a[j];
            float val = csr_sv[j];
            int64_t clo = csc_off[tok], chi = csc_off[tok + 1];
            for (int64_t c = clo; c < chi; c++)
              row_buf[(uint64_t)csc_rows_a[c]] += val * csc_vals[c];
          }
          for (uint64_t j = 0; j < m; j++) {
            double denom = (double)csr_norms[si] * (double)enc->csr_norms[j];
            sims_row[j] = (float)tk_spectral_kernel_apply(enc->kernel,
              (double)row_buf[j], denom);
            row_buf[j] = 0.0f;
          }
        }
      }

    } else if (enc->mod_type == TK_MOD_DENSE) {
      int64_t di = enc->d_input;
      float *src_f = (float *)malloc(blk * (uint64_t)di * sizeof(float));
      if (!src_f) { free(sims_f); free(csr_sval); free(csr_norms); return luaL_error(L, "encode: out of memory"); }
      uint64_t ddi = in_d_input_scalar > 0 ? (uint64_t)in_d_input_scalar : (uint64_t)di;
      uint64_t src_off_base = base * ddi;
      uint64_t cnt = blk * (uint64_t)di;
      if (in_codes_fv) {
        memcpy(src_f, in_codes_fv->a + src_off_base, cnt * sizeof(float));
      } else {
        for (uint64_t i = 0; i < cnt; i++)
          src_f[i] = (float)in_codes_dv->a[src_off_base + i];
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        (int)blk, (int)m, (int)di, 1.0f,
        src_f, (int)di, enc->dense_vecs, (int)di,
        0.0f, sims_f, (int)m);
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < blk; i++) {
        float ni = cblas_snrm2((int)di, src_f + i * (uint64_t)di, 1);
        for (uint64_t j = 0; j < m; j++) {
          double denom = (double)ni * (double)enc->dense_norms[j];
          sims_f[i * m + j] = (float)tk_spectral_kernel_apply(enc->kernel,
            (double)sims_f[i * m + j], denom);
        }
      }
      free(src_f);

    } else if (enc->mod_type == TK_MOD_BITS) {
      uint64_t bd = enc->bits_d;
      uint64_t row_bytes = TK_CVEC_BITS_BYTES(bd);
      const uint8_t *bdata = (const uint8_t *)in_bits->a;
      double inv_bd = 1.0 / (double)bd;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < blk; i++) {
        const uint8_t *row_i = bdata + (base + i) * row_bytes;
        float *sims_row = sims_f + i * m;
        for (uint64_t j = 0; j < m; j++) {
          uint64_t ham = tk_cvec_bits_hamming_serial(
            row_i, enc->bits_data + j * row_bytes, bd);
          sims_row[j] = (float)(1.0 - (double)ham * inv_bd);
        }
      }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)blk, (int)d, (int)m, 1.0f,
      sims_f, (int)m,
      enc->projection, (int)d,
      0.0f, out->a + base * d, (int)d);
  }

  free(sims_f);
  free(row_bufs);
  free(csr_sval);
  free(csr_norms);
  lua_pushvalue(L, out_fv_idx);
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
  uint8_t version = 21;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  uint8_t kernel_byte = (uint8_t)enc->kernel;
  tk_lua_fwrite(L, &kernel_byte, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->mod_type, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->m, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->d, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->trace_ratio, sizeof(double), 1, fh);
  tk_lua_fwrite(L, enc->projection, sizeof(float), enc->m * enc->d, fh);
  if (enc->mod_type == TK_MOD_CSR) {
    tk_lua_fwrite(L, &enc->csr_n_tokens, sizeof(uint64_t), 1, fh);
    uint64_t total_csr = (uint64_t)enc->csr_offsets[enc->m];
    tk_lua_fwrite(L, enc->csr_offsets, sizeof(int64_t), enc->m + 1, fh);
    tk_lua_fwrite(L, enc->csr_tokens, sizeof(int64_t), total_csr, fh);
    tk_lua_fwrite(L, enc->csr_values, sizeof(float), total_csr, fh);
    tk_lua_fwrite(L, enc->csr_norms, sizeof(float), enc->m, fh);
  } else if (enc->mod_type == TK_MOD_DENSE) {
    tk_lua_fwrite(L, &enc->d_input, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, enc->dense_vecs, sizeof(float), enc->m * (uint64_t)enc->d_input, fh);
    tk_lua_fwrite(L, enc->dense_norms, sizeof(float), enc->m, fh);
  } else if (enc->mod_type == TK_MOD_BITS) {
    tk_lua_fwrite(L, &enc->bits_d, sizeof(uint64_t), 1, fh);
    uint64_t row_bytes = TK_CVEC_BITS_BYTES(enc->bits_d);
    tk_lua_fwrite(L, enc->bits_data, 1, enc->m * row_bytes, fh);
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

  uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "encode", "n_samples");

  tk_spectral_modality_t mod;
  memset(&mod, 0, sizeof(mod));
  float *csr_values_owned = NULL;
  double *csr_norms_owned = NULL;
  double *dense_norms_owned = NULL;

  lua_getfield(L, 1, "offsets");
  int has_csr = !lua_isnil(L, -1);
  if (has_csr) {
    tk_ivec_t *off_iv = tk_ivec_peek(L, -1, "offsets");
    lua_pop(L, 1);
    lua_getfield(L, 1, "tokens");
    tk_ivec_t *tok_iv = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    lua_getfield(L, 1, "n_tokens");
    mod.csr_n_tokens = tk_lua_fcheckunsigned(L, 1, "encode", "n_tokens");
    lua_pop(L, 1);
    lua_getfield(L, 1, "values");
    tk_fvec_t *val_fv = tk_fvec_peekopt(L, -1);
    lua_pop(L, 1);
    mod.csr_offsets = off_iv->a;
    mod.csr_tokens = tok_iv->a;
    mod.csr_values = val_fv ? val_fv->a : NULL;
    mod.type = TK_MOD_CSR;
  } else {
    lua_pop(L, 1);
  }

  lua_getfield(L, 1, "codes");
  int has_dense = !lua_isnil(L, -1);
  double *dense_from_fvec = NULL;
  if (has_dense) {
    tk_dvec_t *codes_dv = tk_dvec_peekopt(L, -1);
    tk_fvec_t *codes_fv = codes_dv ? NULL : tk_fvec_peekopt(L, -1);
    if (!codes_dv && !codes_fv)
      return luaL_error(L, "codes: expected dvec or fvec");
    lua_pop(L, 1);
    uint64_t cn = codes_dv ? codes_dv->n : codes_fv->n;
    lua_getfield(L, 1, "d_input");
    int64_t di;
    if (lua_isnumber(L, -1))
      di = (int64_t)lua_tointeger(L, -1);
    else
      di = (int64_t)(cn / n_samples);
    lua_pop(L, 1);
    mod.type = TK_MOD_DENSE;
    mod.d_input = di;
    if (codes_dv) {
      mod.dense = codes_dv->a;
    } else {
      dense_from_fvec = (double *)malloc(cn * sizeof(double));
      for (uint64_t i = 0; i < cn; i++)
        dense_from_fvec[i] = (double)codes_fv->a[i];
      mod.dense = dense_from_fvec;
    }
  } else {
    lua_pop(L, 1);
  }

  lua_getfield(L, 1, "bits");
  int has_bits = !lua_isnil(L, -1);
  if (has_bits) {
    tk_cvec_t *bits_cv = tk_cvec_peek(L, -1, "bits");
    lua_pop(L, 1);
    lua_getfield(L, 1, "d_bits");
    mod.bits_d = tk_lua_fcheckunsigned(L, 1, "encode", "d_bits");
    lua_pop(L, 1);
    mod.type = TK_MOD_BITS;
    mod.bits_data = (const uint8_t *)bits_cv->a;
  } else {
    lua_pop(L, 1);
  }

  int n_provided = has_csr + has_dense + has_bits;
  if (n_provided == 0)
    return luaL_error(L, "encode: no modality provided");
  if (n_provided > 1)
    return luaL_error(L, "encode: provide exactly one modality (csr, dense, or bits)");

  lua_getfield(L, 1, "kernel");
  const char *kernel_str = lua_isnil(L, -1) ? "cosine" : lua_tostring(L, -1);
  lua_pop(L, 1);
  tk_spectral_kernel_t kernel = TK_SPECTRAL_COSINE;
  if (strcmp(kernel_str, "cosine") == 0) kernel = TK_SPECTRAL_COSINE;
  else if (strcmp(kernel_str, "nngp") == 0) kernel = TK_SPECTRAL_NNGP;
  else if (strcmp(kernel_str, "ntk") == 0) kernel = TK_SPECTRAL_NTK;
  else if (strcmp(kernel_str, "expcos") == 0) kernel = TK_SPECTRAL_EXPCOS;
  else if (strcmp(kernel_str, "geolaplace") == 0) kernel = TK_SPECTRAL_GEOLAPLACE;
  else return luaL_error(L, "encode: unknown kernel '%s'", kernel_str);

  if (has_csr) {
    uint64_t nnz = (uint64_t)(mod.csr_offsets[n_samples] - mod.csr_offsets[0]);
    if (!mod.csr_values) {
      csr_values_owned = (float *)malloc(nnz * sizeof(float));
      for (uint64_t i = 0; i < nnz; i++)
        csr_values_owned[i] = 1.0f;
      mod.csr_values = csr_values_owned;
    }
    csr_norms_owned = (double *)calloc(n_samples, sizeof(double));
    mod.csr_norms = csr_norms_owned;
    for (uint64_t s = 0; s < n_samples; s++) {
      int64_t lo = mod.csr_offsets[s], hi = mod.csr_offsets[s + 1];
      double ss = 0.0;
      for (int64_t j = lo; j < hi; j++) {
        double v = (double)mod.csr_values[j];
        ss += v * v;
      }
      csr_norms_owned[s] = sqrt(ss);
    }
  }

  if (has_dense) {
    int64_t di = mod.d_input;
    dense_norms_owned = (double *)malloc(n_samples * sizeof(double));
    for (uint64_t i = 0; i < n_samples; i++)
      dense_norms_owned[i] = cblas_dnrm2((int)di, mod.dense + i * (uint64_t)di, 1);
    mod.dense_norms = dense_norms_owned;
  }

  uint64_t n_lm_req = tk_lua_foptunsigned(L, 1, "encode", "n_landmarks", 0);
  double trace_tol = tk_lua_foptnumber(L, 1, "encode", "trace_tol", 0.0);

  lua_getfield(L, 1, "transform");
  int transform_sign = 0, transform_codes = 0;
  if (lua_isstring(L, -1) && strcmp(lua_tostring(L, -1), "sign") == 0)
    transform_sign = 1;
  else if (lua_toboolean(L, -1))
    transform_codes = 1;
  lua_pop(L, 1);

  tk_ivec_t *lm_ids = NULL;
  tk_fvec_t *lm_chol = NULL;
  float *full_chol = NULL;
  uint64_t nc, m;
  double trace_ratio;
  tk_spectral_sample_landmarks(L,
    &mod, n_samples, kernel,
    n_lm_req, trace_tol,
    &lm_ids, &lm_chol, &full_chol, &nc, &m, &trace_ratio);
  int lm_ids_idx = lua_gettop(L);

  uint64_t d = m;

  lua_getfield(L, 1, "label_offsets");
  int has_gram_labels = !lua_isnil(L, -1);
  tk_ivec_t *gram_lbl_off = has_gram_labels ? tk_ivec_peek(L, -1, "label_offsets") : NULL;
  lua_pop(L, 1);
  lua_getfield(L, 1, "label_neighbors");
  tk_ivec_t *gram_lbl_nbr = has_gram_labels ? tk_ivec_peek(L, -1, "label_neighbors") : NULL;
  lua_pop(L, 1);
  int64_t gram_nl = 0;
  if (has_gram_labels) {
    lua_getfield(L, 1, "n_labels");
    gram_nl = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
  }
  lua_getfield(L, 1, "targets");
  int has_gram_targets = !lua_isnil(L, -1);
  tk_dvec_t *gram_targets_dv = has_gram_targets ? tk_dvec_peek(L, -1, "targets") : NULL;
  lua_pop(L, 1);
  if (has_gram_targets) {
    lua_getfield(L, 1, "n_targets");
    gram_nl = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
  }
  int build_gram = has_gram_labels || has_gram_targets;

  if (m == 0) {
    if (lm_chol) tk_fvec_destroy(lm_chol);
    free(full_chol);
    free(csr_values_owned); free(csr_norms_owned);
    free(dense_norms_owned); free(dense_from_fvec);
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

  tk_fvec_t *train_codes = NULL;
  int train_codes_idx = 0;
  tk_cvec_t *sign_cvec = NULL;
  int sign_cvec_idx = 0;
  uint64_t sign_row_bytes = 0;
  uint8_t *sign_data = NULL;

  if (transform_codes) {
    train_codes = tk_fvec_create(L, nc * d, 0, 0);
    train_codes->n = nc * d;
    train_codes_idx = lua_gettop(L);
  } else if (transform_sign) {
    sign_row_bytes = TK_CVEC_BITS_BYTES(d);
    sign_cvec = tk_cvec_create(L, nc * sign_row_bytes, 0, 0);
    sign_cvec->n = nc * sign_row_bytes;
    tk_cvec_zero(sign_cvec);
    sign_cvec_idx = lua_gettop(L);
    sign_data = (uint8_t *)sign_cvec->a;
  }

  ctx->projection = (float *)calloc(m * m, sizeof(float));
  for (uint64_t i = 0; i < m; i++) ctx->projection[i * m + i] = 1.0f;
  cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
    (int)m, (int)m, 1.0f, lm_chol->a, (int)m, ctx->projection, (int)m);
  tk_fvec_destroy(lm_chol); ctx->lm_chol = NULL;

  int gram_result_idx = 0;
  if (build_gram) {
    uint64_t unl = (uint64_t)gram_nl;
    double *XtX = (double *)calloc(d * d, sizeof(double));
    double *xty = (double *)calloc(d * unl, sizeof(double));
    double *col_mean = (double *)calloc(d, sizeof(double));
    double *y_mean_arr = (double *)malloc(unl * sizeof(double));
    double *eigenvals = (double *)malloc(d * sizeof(double));
    if (!XtX || !xty || !col_mean || !y_mean_arr || !eigenvals) {
      free(XtX); free(xty); free(col_mean); free(y_mean_arr); free(eigenvals);
      return luaL_error(L, "gram fusion: out of memory");
    }
    int64_t tile_size = 1024;
    double *tile_buf = (double *)malloc((uint64_t)tile_size * d * sizeof(double));
    if (!tile_buf) {
      free(XtX); free(xty); free(col_mean); free(y_mean_arr); free(eigenvals);
      return luaL_error(L, "gram fusion: out of memory (tile_buf)");
    }
    for (int64_t base = 0; base < (int64_t)nc; base += tile_size) {
      int64_t bs = (base + tile_size <= (int64_t)nc) ? tile_size : (int64_t)nc - base;
      uint64_t ubs = (uint64_t)bs;
      if (transform_codes) {
        for (uint64_t j = 0; j < d; j++) {
          double *col = tile_buf + j * ubs;
          float *src = full_chol + j * nc + base;
          for (uint64_t i = 0; i < ubs; i++) {
            double v = (double)src[i];
            col[i] = v;
            col_mean[j] += v;
            train_codes->a[((uint64_t)base + i) * d + j] = src[i];
          }
        }
      } else if (transform_sign) {
        for (uint64_t j = 0; j < d; j++) {
          double *col = tile_buf + j * ubs;
          float *src = full_chol + j * nc + base;
          for (uint64_t i = 0; i < ubs; i++) {
            double v = (double)src[i];
            col[i] = v;
            col_mean[j] += v;
            if (v >= 0.0)
              sign_data[((uint64_t)base + i) * sign_row_bytes + j / 8] |= (1u << (j % 8));
          }
        }
      } else {
        for (uint64_t j = 0; j < d; j++) {
          double *col = tile_buf + j * ubs;
          float *src = full_chol + j * nc + base;
          for (uint64_t i = 0; i < ubs; i++) {
            double v = (double)src[i];
            col[i] = v;
            col_mean[j] += v;
          }
        }
      }
      cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
        (int)d, (int)bs, 1.0, tile_buf, (int)bs, 1.0, XtX, (int)d);
      if (has_gram_labels) {
        #pragma omp parallel for schedule(static)
        for (int64_t k = 0; k < (int64_t)d; k++) {
          double *col = tile_buf + (uint64_t)k * ubs;
          for (uint64_t i = 0; i < ubs; i++)
            for (int64_t j = gram_lbl_off->a[(uint64_t)base + i];
                 j < gram_lbl_off->a[(uint64_t)base + i + 1]; j++)
              xty[k * gram_nl + gram_lbl_nbr->a[j]] += col[i];
        }
      }
      if (has_gram_targets) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          (int)d, (int)gram_nl, (int)bs, 1.0, tile_buf, (int)bs,
          gram_targets_dv->a + (uint64_t)base * unl, (int)gram_nl,
          1.0, xty, (int)gram_nl);
      }
    }
    free(tile_buf);
    free(full_chol); ctx->full_chol = NULL;
    for (uint64_t j = 0; j < d; j++)
      col_mean[j] /= (double)nc;
    tk_dvec_t *lc = NULL;
    int lc_idx = 0;
    if (has_gram_labels) {
      lc = tk_dvec_create(L, unl, 0, 0);
      lc->n = unl;
      lc_idx = lua_gettop(L);
      memset(lc->a, 0, unl * sizeof(double));
      for (uint64_t s = 0; s < nc; s++)
        for (int64_t j = gram_lbl_off->a[s]; j < gram_lbl_off->a[s + 1]; j++)
          lc->a[gram_lbl_nbr->a[j]] += 1.0;
      for (int64_t l = 0; l < gram_nl; l++)
        y_mean_arr[l] = lc->a[l] / (double)nc;
    }
    if (has_gram_targets) {
      for (int64_t l = 0; l < gram_nl; l++) {
        double s = 0.0;
        for (uint64_t i = 0; i < nc; i++)
          s += gram_targets_dv->a[i * unl + (uint64_t)l];
        y_mean_arr[l] = s / (double)nc;
      }
    }
    tk_gram_finalize(L, XtX, xty, col_mean, y_mean_arr,
      eigenvals, lc, lc_idx, (int64_t)nc, (int64_t)d, gram_nl);
    gram_result_idx = lua_gettop(L);
  }

  if (!build_gram) {
    #define TK_TILE 32
    if (transform_codes) {
      #pragma omp parallel for schedule(static) collapse(2)
      for (uint64_t jj = 0; jj < d; jj += TK_TILE)
        for (uint64_t ii = 0; ii < nc; ii += TK_TILE) {
          uint64_t je = jj + TK_TILE < d ? jj + TK_TILE : d;
          uint64_t ie = ii + TK_TILE < nc ? ii + TK_TILE : nc;
          for (uint64_t j = jj; j < je; j++)
            for (uint64_t i = ii; i < ie; i++)
              train_codes->a[i * d + j] = full_chol[j * nc + i];
        }
    } else if (transform_sign) {
      for (uint64_t jj = 0; jj < d; jj += TK_TILE)
        for (uint64_t ii = 0; ii < nc; ii += TK_TILE) {
          uint64_t je = jj + TK_TILE < d ? jj + TK_TILE : d;
          uint64_t ie = ii + TK_TILE < nc ? ii + TK_TILE : nc;
          for (uint64_t j = jj; j < je; j++)
            for (uint64_t i = ii; i < ie; i++)
              if (full_chol[j * nc + i] >= 0.0f)
                sign_data[i * sign_row_bytes + j / 8] |= (1u << (j % 8));
        }
    }
    #undef TK_TILE
    free(full_chol); ctx->full_chol = NULL;
  }

  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->projection = ctx->projection;
  ctx->projection = NULL;
  enc->m = m;
  enc->d = d;
  enc->trace_ratio = trace_ratio;
  enc->destroyed = false;
  enc->kernel = kernel;
  enc->mod_type = mod.type;
  enc->csr_offsets = NULL;
  enc->csr_tokens = NULL;
  enc->csr_values = NULL;
  enc->csr_norms = NULL;
  enc->csc_offsets = NULL;
  enc->csc_rows = NULL;
  enc->csc_values = NULL;
  enc->csr_n_tokens = 0;
  enc->dense_vecs = NULL;
  enc->dense_norms = NULL;
  enc->d_input = 0;
  enc->bits_data = NULL;
  enc->bits_d = 0;

  if (has_csr) {
    uint64_t csr_nt = mod.csr_n_tokens;
    enc->csr_n_tokens = csr_nt;
    uint64_t lm_total = 0;
    for (uint64_t j = 0; j < m; j++) {
      uint64_t si = (uint64_t)lm_ids->a[j];
      lm_total += (uint64_t)(mod.csr_offsets[si + 1] - mod.csr_offsets[si]);
    }
    enc->csr_offsets = (int64_t *)malloc((m + 1) * sizeof(int64_t));
    enc->csr_tokens = (int64_t *)malloc(lm_total * sizeof(int64_t));
    enc->csr_values = (float *)malloc(lm_total * sizeof(float));
    enc->csr_norms = (float *)malloc(m * sizeof(float));
    enc->csr_offsets[0] = 0;
    for (uint64_t j = 0; j < m; j++) {
      uint64_t si = (uint64_t)lm_ids->a[j];
      int64_t lo = mod.csr_offsets[si], hi = mod.csr_offsets[si + 1];
      int64_t cnt = hi - lo;
      memcpy(enc->csr_tokens + enc->csr_offsets[j],
             ((const int64_t *)mod.csr_tokens) + (lo - mod.csr_offsets[0]),
             (uint64_t)cnt * sizeof(int64_t));
      memcpy(enc->csr_values + enc->csr_offsets[j],
             ((const float *)mod.csr_values) + (lo - mod.csr_offsets[0]),
             (uint64_t)cnt * sizeof(float));
      enc->csr_norms[j] = (float)csr_norms_owned[si];
      enc->csr_offsets[j + 1] = enc->csr_offsets[j] + cnt;
    }
    enc->csc_offsets = (int64_t *)calloc(csr_nt + 1, sizeof(int64_t));
    for (uint64_t i = 0; i < lm_total; i++)
      enc->csc_offsets[enc->csr_tokens[i] + 1]++;
    for (uint64_t t = 0; t < csr_nt; t++)
      enc->csc_offsets[t + 1] += enc->csc_offsets[t];
    enc->csc_rows = (int64_t *)malloc(lm_total * sizeof(int64_t));
    enc->csc_values = (float *)malloc(lm_total * sizeof(float));
    int64_t *csc_pos = (int64_t *)malloc((csr_nt + 1) * sizeof(int64_t));
    memcpy(csc_pos, enc->csc_offsets, (csr_nt + 1) * sizeof(int64_t));
    for (uint64_t j = 0; j < m; j++) {
      for (int64_t a = enc->csr_offsets[j]; a < enc->csr_offsets[j + 1]; a++) {
        int64_t tok = enc->csr_tokens[a];
        int64_t p = csc_pos[tok]++;
        enc->csc_rows[p] = (int64_t)j;
        enc->csc_values[p] = enc->csr_values[a];
      }
    }
    free(csc_pos);
    free(csr_values_owned); free(csr_norms_owned);
  }

  if (has_dense) {
    int64_t di = mod.d_input;
    enc->d_input = di;
    float *lmv = (float *)malloc(m * (uint64_t)di * sizeof(float));
    for (uint64_t j = 0; j < m; j++) {
      uint64_t off = j * (uint64_t)di;
      uint64_t src = (uint64_t)lm_ids->a[j] * (uint64_t)di;
      for (int64_t k = 0; k < di; k++)
        lmv[off + (uint64_t)k] = (float)mod.dense[src + (uint64_t)k];
    }
    enc->dense_vecs = lmv;
    enc->dense_norms = (float *)malloc(m * sizeof(float));
    for (uint64_t j = 0; j < m; j++)
      enc->dense_norms[j] = cblas_snrm2((int)di, lmv + j * (uint64_t)di, 1);
    free(dense_norms_owned); free(dense_from_fvec);
  }

  if (has_bits) {
    uint64_t bd = mod.bits_d;
    enc->bits_d = bd;
    uint64_t row_bytes = TK_CVEC_BITS_BYTES(bd);
    uint8_t *lmb = (uint8_t *)malloc(m * row_bytes);
    for (uint64_t j = 0; j < m; j++)
      memcpy(lmb + j * row_bytes, mod.bits_data + (uint64_t)lm_ids->a[j] * row_bytes, row_bytes);
    enc->bits_data = lmb;
  }

  lua_newtable(L);
  lua_pushvalue(L, lm_ids_idx);
  lua_setfield(L, -2, "landmark_ids");
  lua_setfenv(L, enc_idx);

  if (transform_codes)
    lua_pushvalue(L, train_codes_idx);
  else if (transform_sign)
    lua_pushvalue(L, sign_cvec_idx);
  else
    lua_pushnil(L);
  lua_pushvalue(L, enc_idx);
  if (gram_result_idx > 0) {
    lua_pushvalue(L, gram_result_idx);
    return 3;
  }
  return 2;
}

static inline void tk_nystrom_build_csc (tk_nystrom_encoder_t *enc) {
  uint64_t csr_nt = enc->csr_n_tokens;
  uint64_t lm_total = (uint64_t)enc->csr_offsets[enc->m];
  enc->csc_offsets = (int64_t *)calloc(csr_nt + 1, sizeof(int64_t));
  for (uint64_t i = 0; i < lm_total; i++)
    enc->csc_offsets[enc->csr_tokens[i] + 1]++;
  for (uint64_t t = 0; t < csr_nt; t++)
    enc->csc_offsets[t + 1] += enc->csc_offsets[t];
  enc->csc_rows = (int64_t *)malloc(lm_total * sizeof(int64_t));
  enc->csc_values = (float *)malloc(lm_total * sizeof(float));
  int64_t *csc_pos = (int64_t *)malloc((csr_nt + 1) * sizeof(int64_t));
  memcpy(csc_pos, enc->csc_offsets, (csr_nt + 1) * sizeof(int64_t));
  for (uint64_t j = 0; j < enc->m; j++) {
    for (int64_t a = enc->csr_offsets[j]; a < enc->csr_offsets[j + 1]; a++) {
      int64_t tok = enc->csr_tokens[a];
      int64_t p = csc_pos[tok]++;
      enc->csc_rows[p] = (int64_t)j;
      enc->csc_values[p] = enc->csr_values[a];
    }
  }
  free(csc_pos);
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
  if (version != 21) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported nystrom encoder version %d", (int)version);
  }

  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->destroyed = false;
  enc->projection = NULL;
  enc->csr_offsets = NULL; enc->csr_tokens = NULL; enc->csr_values = NULL; enc->csr_norms = NULL;
  enc->csc_offsets = NULL; enc->csc_rows = NULL; enc->csc_values = NULL;
  enc->csr_n_tokens = 0;
  enc->dense_vecs = NULL; enc->dense_norms = NULL; enc->d_input = 0;
  enc->bits_data = NULL; enc->bits_d = 0;

  uint8_t kernel_byte;
  tk_lua_fread(L, &kernel_byte, sizeof(uint8_t), 1, fh);
  enc->kernel = (tk_spectral_kernel_t)kernel_byte;
  tk_lua_fread(L, &enc->mod_type, sizeof(uint8_t), 1, fh);
  tk_lua_fread(L, &enc->m, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &enc->d, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &enc->trace_ratio, sizeof(double), 1, fh);
  enc->projection = (float *)malloc(enc->m * enc->d * sizeof(float));
  tk_lua_fread(L, enc->projection, sizeof(float), enc->m * enc->d, fh);
  if (enc->mod_type == TK_MOD_CSR) {
    tk_lua_fread(L, &enc->csr_n_tokens, sizeof(uint64_t), 1, fh);
    enc->csr_offsets = (int64_t *)malloc((enc->m + 1) * sizeof(int64_t));
    tk_lua_fread(L, enc->csr_offsets, sizeof(int64_t), enc->m + 1, fh);
    uint64_t total_csr = (uint64_t)enc->csr_offsets[enc->m];
    enc->csr_tokens = (int64_t *)malloc(total_csr * sizeof(int64_t));
    enc->csr_values = (float *)malloc(total_csr * sizeof(float));
    enc->csr_norms = (float *)malloc(enc->m * sizeof(float));
    tk_lua_fread(L, enc->csr_tokens, sizeof(int64_t), total_csr, fh);
    tk_lua_fread(L, enc->csr_values, sizeof(float), total_csr, fh);
    tk_lua_fread(L, enc->csr_norms, sizeof(float), enc->m, fh);
    tk_nystrom_build_csc(enc);
  } else if (enc->mod_type == TK_MOD_DENSE) {
    tk_lua_fread(L, &enc->d_input, sizeof(int64_t), 1, fh);
    enc->dense_vecs = (float *)malloc(enc->m * (uint64_t)enc->d_input * sizeof(float));
    tk_lua_fread(L, enc->dense_vecs, sizeof(float), enc->m * (uint64_t)enc->d_input, fh);
    enc->dense_norms = (float *)malloc(enc->m * sizeof(float));
    tk_lua_fread(L, enc->dense_norms, sizeof(float), enc->m, fh);
  } else if (enc->mod_type == TK_MOD_BITS) {
    tk_lua_fread(L, &enc->bits_d, sizeof(uint64_t), 1, fh);
    uint64_t row_bytes = TK_CVEC_BITS_BYTES(enc->bits_d);
    enc->bits_data = (uint8_t *)malloc(enc->m * row_bytes);
    tk_lua_fread(L, enc->bits_data, 1, enc->m * row_bytes, fh);
  }

  tk_ivec_load(L, fh);
  int lm_ids_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);
  lua_newtable(L);
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
