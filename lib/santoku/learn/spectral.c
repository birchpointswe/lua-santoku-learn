#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/ivec.h>
#include <santoku/svec.h>
#include <santoku/cvec.h>
#include <santoku/learn/buf.h>
#include <santoku/learn/gram.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <santoku/learn/mathlibs.h>

static inline const int32_t *tk_peek_tokens (lua_State *L, int idx, uint64_t *out_n) {
  tk_svec_t *sv = tk_svec_peekopt(L, idx);
  if (sv) { *out_n = sv->n; return sv->a; }
  tk_ivec_t *iv = tk_ivec_peekopt(L, idx);
  if (!iv) { *out_n = 0; return NULL; }
  tk_svec_t *conv = tk_svec_create(L, iv->n);
  conv->n = iv->n;
  for (uint64_t i = 0; i < iv->n; i++) conv->a[i] = (int32_t)iv->a[i];
  lua_replace(L, idx);
  *out_n = conv->n;
  return conv->a;
}

// All supported kernels are self-similarity=1: each is a function f of the cosine c only
// (scale-invariant, ignores magnitude) with f(1)=1, so k(x,x)=1 always. Inputs are unit-L2-normalized
// by the caller, so c is just the sparse/dense dot (no denom here). This keeps downstream well
// behaved: the residual/trace diagonal is a constant 1, which leaves trace_tol invariant. Any
// kernel added here MUST satisfy f(1)=1 and be PSD on the sphere. Norm-scaled kernels (e.g. nngp/ntk, where k(x,x)=||x||^2) were
// removed for breaking these guarantees. The sphere identities: d^2 = 2(1-c) is the squared
// chordal distance, so expcos = Gaussian (exp(-d^2/2)) and geolaplace = Laplacian/Matern-1/2
// (exp(-d)); matern52 continues that smoothness ladder; arccos1 is arc-cosine order 1 normalized to
// self-sim=1 (the old nngp with the ||x|| denom divided out); rq is the rational-quadratic (alpha=1)
// heavy-tailed kernel. All at the natural scale (l=1). (The full-search-dominated angular and
// matern32 were pruned: each was a never-winning interpolator between two retained kernels.)
typedef enum {
  TK_SPECTRAL_COSINE = 0,
  TK_SPECTRAL_EXPCOS = 1,
  TK_SPECTRAL_GEOLAPLACE = 2,
  TK_SPECTRAL_MATERN52 = 3,
  TK_SPECTRAL_RQ = 4,
  TK_SPECTRAL_ARCCOS1 = 5,
} tk_spectral_kernel_t;

static inline double tk_spectral_kernel_apply (tk_spectral_kernel_t k, double c) {
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;
  if (k == TK_SPECTRAL_EXPCOS)
    return exp(c - 1.0);
  if (k == TK_SPECTRAL_GEOLAPLACE) {
    double d2 = 2.0 * (1.0 - c);
    return exp(-sqrt(d2 > 0.0 ? d2 : 0.0));
  }
  if (k == TK_SPECTRAL_MATERN52) {
    double d2 = 2.0 * (1.0 - c);
    double a = sqrt(5.0 * (d2 > 0.0 ? d2 : 0.0));
    return (1.0 + a + (5.0 / 3.0) * d2) * exp(-a);
  }
  if (k == TK_SPECTRAL_RQ)
    return 1.0 / (2.0 - c);
  if (k == TK_SPECTRAL_ARCCOS1) {
    double t = acos(c);
    return (sin(t) + (M_PI - t) * c) / M_PI;
  }
  return c;
}

#define TK_MOD_CSR   0
#define TK_MOD_DENSE 1
#define TK_MERGE_MEAN    0
#define TK_MERGE_PRODUCT 1
#define TK_MAX_MOD 8

typedef struct {
  uint8_t type;
  const int64_t *csr_offsets;
  const int32_t *csr_tokens;
  const float *csr_values;
  uint64_t csr_n_tokens;
  const double *dense;
  int64_t d_input;
} tk_spectral_modality_t;

typedef struct {
  int64_t *sid_map;
  double *residual;
  float *L_mat;
  int L_mat_external;
  int64_t *landmark_sids;
  int64_t *landmark_idx_map;
} tk_spectral_landmarks_ctx_t;

static inline int tk_spectral_landmarks_ctx_gc (lua_State *L) {
  tk_spectral_landmarks_ctx_t *ctx = (tk_spectral_landmarks_ctx_t *)lua_touserdata(L, 1);
  if (ctx->sid_map) { free(ctx->sid_map); ctx->sid_map = NULL; }
  if (ctx->residual) { free(ctx->residual); ctx->residual = NULL; }
  if (ctx->L_mat && !ctx->L_mat_external) { free(ctx->L_mat); ctx->L_mat = NULL; }
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
  float *ext_chol,
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
    *ids_out = tk_ivec_create(L, 0);
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
  if (ext_chol) {
    ctx->L_mat = ext_chol;
    ctx->L_mat_external = 1;
    memset(ext_chol, 0, n_landmarks * n_docs * sizeof(float));
  } else {
    ctx->L_mat = (float *)calloc(n_landmarks * n_docs, sizeof(float));
    ctx->L_mat_external = 0;
  }
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

  #pragma omp parallel for
  for (uint64_t i = 0; i < n_docs; i++)
    residual[i] = 1.0;
  initial_trace = (double)n_docs;

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
    uint64_t n_valid = 0;
    for (uint64_t b = 0; b < n_propose && np < max_blk; b++) {
      uint64_t pi = SAMPLE_PROPOSAL();
      if (residual[pi] < 1.2e-6) continue; // 10*machine_epsilon (float32)
      int dup = 0;
      for (uint64_t k = 0; k < np; k++)
        if (blk_pivots[k] == pi) { dup = 1; break; }
      if (dup) continue;
      n_valid++;
      // Rejection sampling, moved here from the acceptance loop: only pivots that pass get the
      // expensive kip_block + cross_dots computed. Uses block-start residual (consistent with
      // the block approximation kip/cross_dots already make), so accepted pivots are the only
      // columns paid for.
      if (proposal[pi] > 1e-15) {
        double accept_prob = residual[pi] / proposal[pi];
        if (tk_fast_drand() > accept_prob) continue;
      } else {
        continue;
      }
      blk_pivots[np] = pi;
      np++;
    }
    total_proposed += n_propose;
    total_accepted += np;
    if (n_valid == 0) { done = true; break; } // no above-threshold candidate -> converged
    if (np == 0) continue;                     // all rejected this round -> resample/rebuild

    memset(kip_block, 0, n_docs * np * sizeof(float));

    if (mod->type == TK_MOD_CSR) {
      const int64_t *csr_offsets = mod->csr_offsets;
      const int32_t *csr_tokens = mod->csr_tokens;
      const float *csr_values = mod->csr_values;
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
            int32_t tok = csr_tokens[p];
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
            int32_t tok = csr_tokens[p];
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
        const int32_t *restrict c_tok = csr_tokens;
        const float *restrict c_val = csr_values;
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++) {
          double kip_row[TK_CHOL_BLOCK];
          memset(kip_row, 0, np * sizeof(double));
          int64_t jlo = c_off[i], jhi = c_off[i + 1];
          for (int64_t j = jlo; j < jhi; j++) {
            int32_t tok = c_tok[j];
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
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++)
          for (uint64_t b = 0; b < np; b++)
            kip_block[i * np + b] = (float)tk_spectral_kernel_apply(kernel,
              (double)kip_block[i * np + b]);
      }

    } else if (mod->type == TK_MOD_DENSE) {
      int64_t di = mod->d_input;
      const double *dense = mod->dense;
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
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++)
          for (uint64_t b = 0; b < np; b++)
            kip_block[i * np + b] = (float)tk_spectral_kernel_apply(kernel,
              dense_kip[i * np + b]);
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
      // Rejection sampling already applied at propose time; every blk_pivots entry is accepted.
      uint64_t col = actual_landmarks;
      double sc_sq = residual[pi];
      if (sc_sq < 1.2e-6) continue; // 10*machine_epsilon (float32)
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
        double lij = (i == pi) ? sc : (raw - cross - within) / sc;
        float fij = (float)lij;
        L_mat[col * n_docs + i] = fij;
        residual[i] -= (double)fij * (double)fij;
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

  tk_ivec_t *landmark_ids = tk_ivec_create(L, actual_landmarks);
  for (uint64_t i = 0; i < actual_landmarks; i++)
    landmark_ids->a[i] = landmark_sids[i];
  landmark_ids->n = actual_landmarks;

  tk_fvec_t *chol = tk_fvec_create(NULL, actual_landmarks * actual_landmarks);
  #pragma omp parallel for schedule(static)
  for (uint64_t li = 0; li < actual_landmarks; li++) {
    uint64_t doc_idx = (uint64_t)landmark_idx_map[li];
    for (uint64_t k = 0; k < actual_landmarks; k++)
      chol->a[li * actual_landmarks + k] = L_mat[k * n_docs + doc_idx];
  }
  chol->n = actual_landmarks * actual_landmarks;

  if (!ext_chol && actual_landmarks > 0 && actual_landmarks < n_landmarks) {
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
  int full_chol_external;
} tk_encode_nystrom_ctx_t;

static inline int tk_encode_nystrom_ctx_gc (lua_State *L) {
  tk_encode_nystrom_ctx_t *c = (tk_encode_nystrom_ctx_t *)lua_touserdata(L, 1);
  free(c->projection);
  if (c->lm_chol) tk_fvec_destroy(c->lm_chol);
  if (!c->full_chol_external) free(c->full_chol);
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
  int32_t *csr_tokens;
  float *csr_values;
  int64_t *csc_offsets;
  int64_t *csc_rows;
  float *csc_values;
  uint64_t csr_n_tokens;
  float *dense_vecs;
  int64_t d_input;
  float *sims_buf;
  float *row_bufs;
  float *dense_tile_buf;
  uint64_t tile;
  int n_threads;
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
      free(enc->csc_offsets);
      free(enc->csc_rows);
      free(enc->csc_values);
    } else if (enc->mod_type == TK_MOD_DENSE) {
      free(enc->dense_vecs);
    }
    free(enc->sims_buf);
    free(enc->row_bufs);
    free(enc->dense_tile_buf);
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
  uint64_t in_tok_n;
  const int32_t *in_tok_a = tk_peek_tokens(L, -1, &in_tok_n);
  lua_setfield(L, 2, "tokens");
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
    out = tk_fvec_create(L, n_samples * d);
    out->n = n_samples * d;
    out_fv_idx = lua_gettop(L);
  }

  if (!enc->sims_buf) {
    uint64_t tile = 4096;
    while (tile > 1 && tile * m * sizeof(float) > 256ULL * 1024 * 1024)
      tile /= 2;
    enc->tile = tile;
    enc->sims_buf = (float *)malloc(tile * m * sizeof(float));
    if (!enc->sims_buf) return luaL_error(L, "encode: out of memory");
  }
  uint64_t tile = enc->tile;
  if (tile > n_samples) tile = n_samples;
  float *sims_f = enc->sims_buf;

  float *csr_sval = NULL;
  const float *csr_sv = NULL;
  if (enc->mod_type == TK_MOD_CSR) {
    if (!in_offsets) return luaL_error(L, "encode: CSR modality but no offsets");
    uint64_t nnz = in_tok_n;
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
  }

  float *row_bufs = NULL;
  if (enc->mod_type == TK_MOD_CSR) {
    int nt = omp_get_max_threads();
    if (!enc->row_bufs || nt > enc->n_threads) {
      free(enc->row_bufs);
      enc->row_bufs = (float *)calloc((uint64_t)nt * m, sizeof(float));
      enc->n_threads = nt;
      if (!enc->row_bufs) {
        free(csr_sval);
        return luaL_error(L, "encode: out of memory (row_bufs)");
      }
    }
    row_bufs = enc->row_bufs;
  }

  for (uint64_t base = 0; base < n_samples; base += tile) {
    uint64_t blk = base + tile <= n_samples ? tile : n_samples - base;

    if (enc->mod_type == TK_MOD_CSR) {
      const int64_t *off_a = in_offsets->a;
      const int32_t *tok_a = in_tok_a;
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
            int32_t tok = tok_a[j];
            float val = csr_sv[j];
            int64_t clo = csc_off[tok], chi = csc_off[tok + 1];
            for (int64_t c = clo; c < chi; c++)
              row_buf[(uint64_t)csc_rows_a[c]] += val * csc_vals[c];
          }
          for (uint64_t j = 0; j < m; j++) {
            sims_row[j] = (float)tk_spectral_kernel_apply(enc->kernel, (double)row_buf[j]);
            row_buf[j] = 0.0f;
          }
        }
      }

    } else if (enc->mod_type == TK_MOD_DENSE) {
      int64_t di = enc->d_input;
      if (!enc->dense_tile_buf)
        enc->dense_tile_buf = (float *)malloc(enc->tile * (uint64_t)di * sizeof(float));
      float *src_f = enc->dense_tile_buf;
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
      for (uint64_t i = 0; i < blk; i++)
        for (uint64_t j = 0; j < m; j++)
          sims_f[i * m + j] = (float)tk_spectral_kernel_apply(enc->kernel,
            (double)sims_f[i * m + j]);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)blk, (int)d, (int)m, 1.0f,
      sims_f, (int)m,
      enc->projection, (int)d,
      0.0f, out->a + base * d, (int)d);
  }

  free(csr_sval);
  lua_pushvalue(L, out_fv_idx);
  return 1;
}

static inline int tk_nystrom_dims_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->d);
  return 1;
}

static inline int tk_nystrom_encoder_persist_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (enc->destroyed)
    return luaL_error(L, "cannot persist a destroyed encoder");
  if (!enc->projection)
    return luaL_error(L, "cannot persist: projection released");
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKny", 1, 4, fh);
  uint8_t version = 24;
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
    tk_lua_fwrite(L, enc->csr_tokens, sizeof(int32_t), total_csr, fh);
    tk_lua_fwrite(L, enc->csr_values, sizeof(float), total_csr, fh);
  } else if (enc->mod_type == TK_MOD_DENSE) {
    tk_lua_fwrite(L, &enc->d_input, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, enc->dense_vecs, sizeof(float), enc->m * (uint64_t)enc->d_input, fh);
  }
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "landmark_ids");
  tk_ivec_t *lm_ids = tk_ivec_peek(L, -1, "landmark_ids");
  tk_ivec_persist(L, lm_ids, fh);
  lua_pop(L, 2);
  tk_lua_fclose(L, fh);
  return 0;
}

static inline int tk_nystrom_shrink_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  free(enc->projection);
  enc->projection = NULL;
  free(enc->sims_buf); enc->sims_buf = NULL;
  free(enc->row_bufs); enc->row_bufs = NULL;
  free(enc->dense_tile_buf); enc->dense_tile_buf = NULL;
  return 0;
}

static luaL_Reg tk_nystrom_encoder_mt_fns[] = {
  { "encode", tk_nystrom_encode_lua },
  { "dims", tk_nystrom_dims_lua },
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
  lua_getfield(L, 1, "offsets");
  int has_csr = !lua_isnil(L, -1);
  if (has_csr) {
    tk_ivec_t *off_iv = tk_ivec_peek(L, -1, "offsets");
    lua_pop(L, 1);
    lua_getfield(L, 1, "tokens");
    uint64_t tok_n;
    const int32_t *tok_a = tk_peek_tokens(L, -1, &tok_n);
    if (!tok_a) return luaL_error(L, "tokens: expected svec or ivec");
    lua_setfield(L, 1, "tokens");
    lua_getfield(L, 1, "n_tokens");
    mod.csr_n_tokens = tk_lua_fcheckunsigned(L, 1, "encode", "n_tokens");
    lua_pop(L, 1);
    lua_getfield(L, 1, "values");
    tk_fvec_t *val_fv = tk_fvec_peekopt(L, -1);
    lua_pop(L, 1);
    mod.csr_offsets = off_iv->a;
    mod.csr_tokens = tok_a;
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

  int n_provided = has_csr + has_dense;
  if (n_provided == 0)
    return luaL_error(L, "encode: no modality provided");
  if (n_provided > 1)
    return luaL_error(L, "encode: provide exactly one modality (csr or dense)");

  lua_getfield(L, 1, "kernel");
  const char *kernel_str = lua_isnil(L, -1) ? "cosine" : lua_tostring(L, -1);
  lua_pop(L, 1);
  tk_spectral_kernel_t kernel = TK_SPECTRAL_COSINE;
  if (strcmp(kernel_str, "cosine") == 0) kernel = TK_SPECTRAL_COSINE;
  else if (strcmp(kernel_str, "expcos") == 0) kernel = TK_SPECTRAL_EXPCOS;
  else if (strcmp(kernel_str, "geolaplace") == 0) kernel = TK_SPECTRAL_GEOLAPLACE;
  else if (strcmp(kernel_str, "matern52") == 0) kernel = TK_SPECTRAL_MATERN52;
  else if (strcmp(kernel_str, "rq") == 0) kernel = TK_SPECTRAL_RQ;
  else if (strcmp(kernel_str, "arccos1") == 0) kernel = TK_SPECTRAL_ARCCOS1;
  else return luaL_error(L, "encode: unknown kernel '%s'", kernel_str);

  if (has_csr && !mod.csr_values) {
    uint64_t nnz = (uint64_t)(mod.csr_offsets[n_samples] - mod.csr_offsets[0]);
    csr_values_owned = (float *)malloc(nnz * sizeof(float));
    for (uint64_t i = 0; i < nnz; i++)
      csr_values_owned[i] = 1.0f;
    mod.csr_values = csr_values_owned;
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

  lua_getfield(L, 1, "chol_buf");
  tk_fvec_t *chol_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "chol_buf");
  lua_pop(L, 1);

  tk_ivec_t *lm_ids = NULL;
  tk_fvec_t *lm_chol = NULL;
  float *full_chol = NULL;
  uint64_t nc, m;
  double trace_ratio;
  tk_spectral_sample_landmarks(L,
    &mod, n_samples, kernel,
    n_lm_req, trace_tol,
    chol_buf ? chol_buf->a : NULL,
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
  int build_gram = has_gram_targets;

  int64_t tile_labels = 0;
  lua_getfield(L, 1, "tile_labels");
  if (lua_isnumber(L, -1)) tile_labels = (int64_t)lua_tointeger(L, -1);
  lua_pop(L, 1);
  if (tile_labels <= 0 && has_gram_labels && !has_gram_targets)
    tile_labels = 1024;
  int64_t sample_tile_size = 1024;
  lua_getfield(L, 1, "tile_samples");
  if (lua_isnumber(L, -1)) sample_tile_size = (int64_t)lua_tointeger(L, -1);
  lua_pop(L, 1);
  int build_gram_tiled = tile_labels > 0 && has_gram_labels && !has_gram_targets;

  // Cholesky shortcut: when lambda (and propensity, for labels) are fixed, the
  // eigendecomposition is never reused, so solve W once via Cholesky and skip dsyevd.
  lua_getfield(L, 1, "solve_lambda");
  int do_solve = lua_isnumber(L, -1);
  double solve_lambda = do_solve ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "solve_propensity_a");
  int solve_do_prop = do_solve && lua_isnumber(L, -1);
  double solve_prop_a = solve_do_prop ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "solve_propensity_b");
  double solve_prop_b = solve_do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
  lua_pop(L, 1);

  if (m == 0) {
    if (lm_chol) tk_fvec_destroy(lm_chol);
    if (!chol_buf) free(full_chol);
    free(csr_values_owned); free(dense_from_fvec);
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
  ctx->full_chol_external = (chol_buf != NULL);

  tk_fvec_t *train_codes = NULL;
  int train_codes_idx = 0;
  tk_cvec_t *sign_cvec = NULL;
  int sign_cvec_idx = 0;
  uint64_t sign_row_bytes = 0;
  uint8_t *sign_data = NULL;

  if (transform_codes) {
    train_codes = tk_fvec_create(L, nc * d);
    train_codes->n = nc * d;
    train_codes_idx = lua_gettop(L);
  } else if (transform_sign) {
    sign_row_bytes = TK_CVEC_BITS_BYTES(d);
    sign_cvec = tk_cvec_create(L, nc * sign_row_bytes);
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
  if (build_gram_tiled) {
    uint64_t unl = (uint64_t)gram_nl;
    double *XtX = (double *)calloc(d * d, sizeof(double));
    double *col_mean = (double *)calloc(d, sizeof(double));
    double *eigenvals = (double *)malloc(d * sizeof(double));
    if (!XtX || !col_mean || !eigenvals) {
      free(XtX); free(col_mean); free(eigenvals);
      return luaL_error(L, "gram tiled: out of memory");
    }
    int64_t tile_size = sample_tile_size;
    double *tile_buf = (double *)malloc((uint64_t)tile_size * d * sizeof(double));
    if (!tile_buf) {
      free(XtX); free(col_mean); free(eigenvals);
      return luaL_error(L, "gram tiled: out of memory (tile_buf)");
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
    }
    for (uint64_t j = 0; j < d; j++)
      col_mean[j] /= (double)nc;
    double *label_counts = (double *)calloc(unl, sizeof(double));
    double *y_mean_arr = (double *)calloc(unl, sizeof(double));
    if (!label_counts || !y_mean_arr) {
      free(label_counts); free(y_mean_arr);
      free(XtX); free(col_mean); free(eigenvals); free(tile_buf);
      return luaL_error(L, "gram tiled: out of memory (label stats)");
    }
    for (uint64_t s = 0; s < nc; s++)
      for (int64_t j = gram_lbl_off->a[s]; j < gram_lbl_off->a[s + 1]; j++) {
        int64_t lbl = gram_lbl_nbr->a[j];
        label_counts[lbl] += 1.0;
        y_mean_arr[lbl] += 1.0;
      }
    for (int64_t l = 0; l < gram_nl; l++)
      y_mean_arr[l] /= (double)nc;
    cblas_dsyr(CblasColMajor, CblasUpper, (int)d,
      -(double)nc, col_mean, 1, XtX, (int)d);
    if (do_solve) {
      double mean_eig = 0.0;
      for (uint64_t i = 0; i < d; i++) mean_eig += XtX[i * d + i];
      mean_eig /= (double)d;
      double mu = solve_lambda * mean_eig + mean_eig * 1e-7;
      for (uint64_t i = 0; i < d; i++) XtX[i * d + i] += mu;
      int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', (int)d, XtX, (int)d);
      if (info != 0) {
        free(label_counts); free(y_mean_arr);
        free(XtX); free(col_mean); free(eigenvals); free(tile_buf);
        return luaL_error(L, "gram tiled cholesky: dpotrf failed (info=%d)", info);
      }
      free(eigenvals);
      int64_t B = tile_labels;
      float *W_baked_f = (float *)malloc((uint64_t)d * unl * sizeof(float));
      double *intercept = (double *)malloc(unl * sizeof(double));
      double *xty_tile = (double *)malloc(d * (uint64_t)B * sizeof(double));
      double *Bcm = (double *)malloc(d * (uint64_t)B * sizeof(double));
      if (!W_baked_f || !intercept || !xty_tile || !Bcm) {
        free(W_baked_f); free(intercept); free(xty_tile); free(Bcm);
        free(label_counts); free(y_mean_arr);
        free(XtX); free(col_mean); free(tile_buf);
        return luaL_error(L, "gram tiled cholesky: out of memory");
      }
      double C = 0.0;
      if (solve_do_prop)
        C = (log((double)nc) - 1.0) * pow(solve_prop_b + 1.0, solve_prop_a);
      for (int64_t tl_start = 0; tl_start < gram_nl; tl_start += B) {
        int64_t actual_B = (tl_start + B <= gram_nl) ? B : gram_nl - tl_start;
        memset(xty_tile, 0, d * (uint64_t)actual_B * sizeof(double));
        for (int64_t base = 0; base < (int64_t)nc; base += tile_size) {
          int64_t bs = (base + tile_size <= (int64_t)nc) ? tile_size : (int64_t)nc - base;
          uint64_t ubs = (uint64_t)bs;
          for (uint64_t j = 0; j < d; j++) {
            double *col = tile_buf + j * ubs;
            float *src = full_chol + j * nc + base;
            for (uint64_t i = 0; i < ubs; i++)
              col[i] = (double)src[i];
          }
          #pragma omp parallel for schedule(static)
          for (int64_t k = 0; k < (int64_t)d; k++) {
            double *col = tile_buf + (uint64_t)k * ubs;
            for (uint64_t i = 0; i < ubs; i++) {
              uint64_t si = (uint64_t)base + i;
              for (int64_t j = gram_lbl_off->a[si]; j < gram_lbl_off->a[si + 1]; j++) {
                int64_t lbl = gram_lbl_nbr->a[j];
                if (lbl >= tl_start && lbl < tl_start + actual_B)
                  xty_tile[k * actual_B + (lbl - tl_start)] += col[i];
              }
            }
          }
        }
        cblas_dger(CblasRowMajor, (int)d, (int)actual_B,
          -(double)nc, col_mean, 1, y_mean_arr + tl_start, 1, xty_tile, (int)actual_B);
        if (solve_do_prop) {
          for (int64_t k = 0; k < (int64_t)d; k++)
            for (int64_t tl = 0; tl < actual_B; tl++)
              xty_tile[k * actual_B + tl] *=
                (1.0 + C / pow(label_counts[tl_start + tl] + solve_prop_b, solve_prop_a));
        }
        for (int64_t k = 0; k < (int64_t)d; k++)
          for (int64_t tl = 0; tl < actual_B; tl++)
            Bcm[(uint64_t)tl * d + (uint64_t)k] = xty_tile[k * actual_B + tl];
        LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', (int)d, (int)actual_B, XtX, (int)d, Bcm, (int)d);
        for (int64_t k = 0; k < (int64_t)d; k++)
          for (int64_t tl = 0; tl < actual_B; tl++)
            W_baked_f[(uint64_t)k * unl + (uint64_t)(tl_start + tl)] =
              (float)Bcm[(uint64_t)tl * d + (uint64_t)k];
        for (int64_t tl = 0; tl < actual_B; tl++) {
          double prop = solve_do_prop
            ? (1.0 + C / pow(label_counts[tl_start + tl] + solve_prop_b, solve_prop_a)) : 1.0;
          intercept[tl_start + tl] = prop * y_mean_arr[tl_start + tl]
            - cblas_ddot((int)d, Bcm + (uint64_t)tl * d, 1, col_mean, 1);
        }
      }
      free(xty_tile); free(Bcm); free(tile_buf);
      if (!ctx->full_chol_external) free(full_chol);
      ctx->full_chol = NULL;
      free(XtX);
      tk_dvec_t *lc = tk_dvec_create(L, unl);
      lc->n = unl;
      int lc_idx = lua_gettop(L);
      memcpy(lc->a, label_counts, unl * sizeof(double));
      free(label_counts);
      tk_gram_make_baked(L, W_baked_f, intercept, col_mean, y_mean_arr,
        lc, lc_idx, mean_eig, (int64_t)nc, (int64_t)d, gram_nl, tile_labels);
      gram_result_idx = lua_gettop(L);
    } else {
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (int)d, XtX, (int)d, eigenvals);

    {
      int64_t B = tile_labels;
      lua_getfield(L, 1, "pqty_buf");
      tk_fvec_t *pqty_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "pqty_buf");
      int pqty_buf_idx = pqty_buf ? lua_gettop(L) : 0;
      if (!pqty_buf) lua_pop(L, 1);
      float *pqty_f;
      if (pqty_buf) {
        tk_fvec_ensure(pqty_buf, d * unl);
        pqty_buf->n = d * unl;
        pqty_f = pqty_buf->a;
      } else {
        pqty_f = (float *)calloc(d * unl, sizeof(float));
      }
      double *xty_tile = (double *)malloc(d * (uint64_t)B * sizeof(double));
      double *pqty_tile = (double *)malloc(d * (uint64_t)B * sizeof(double));
      if (!xty_tile || !pqty_tile) {
        free(xty_tile); free(pqty_tile);
        if (!pqty_buf) free(pqty_f);
        free(label_counts); free(y_mean_arr);
        free(XtX); free(col_mean); free(eigenvals); free(tile_buf);
        return luaL_error(L, "gram tiled gram: out of memory");
      }
      for (int64_t tl_start = 0; tl_start < gram_nl; tl_start += B) {
        int64_t actual_B = (tl_start + B <= gram_nl) ? B : gram_nl - tl_start;
        memset(xty_tile, 0, d * (uint64_t)actual_B * sizeof(double));
        for (int64_t base = 0; base < (int64_t)nc; base += tile_size) {
          int64_t bs = (base + tile_size <= (int64_t)nc) ? tile_size : (int64_t)nc - base;
          uint64_t ubs = (uint64_t)bs;
          for (uint64_t j = 0; j < d; j++) {
            double *col = tile_buf + j * ubs;
            float *src = full_chol + j * nc + base;
            for (uint64_t i = 0; i < ubs; i++)
              col[i] = (double)src[i];
          }
          #pragma omp parallel for schedule(static)
          for (int64_t k = 0; k < (int64_t)d; k++) {
            double *col = tile_buf + (uint64_t)k * ubs;
            for (uint64_t i = 0; i < ubs; i++) {
              uint64_t si = (uint64_t)base + i;
              for (int64_t j = gram_lbl_off->a[si]; j < gram_lbl_off->a[si + 1]; j++) {
                int64_t lbl = gram_lbl_nbr->a[j];
                if (lbl >= tl_start && lbl < tl_start + actual_B)
                  xty_tile[k * actual_B + (lbl - tl_start)] += col[i];
              }
            }
          }
        }
        cblas_dger(CblasRowMajor, (int)d, (int)actual_B,
          -(double)nc, col_mean, 1, y_mean_arr + tl_start, 1, xty_tile, (int)actual_B);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          (int)d, (int)actual_B, (int)d, 1.0, XtX, (int)d,
          xty_tile, (int)actual_B, 0.0, pqty_tile, (int)actual_B);
        for (int64_t i = 0; i < (int64_t)d; i++)
          for (int64_t tl = 0; tl < actual_B; tl++)
            pqty_f[i * gram_nl + tl_start + tl] = (float)pqty_tile[i * actual_B + tl];
      }
      free(xty_tile); free(pqty_tile); free(tile_buf);
      if (!ctx->full_chol_external) free(full_chol);
      ctx->full_chol = NULL;
      tk_dvec_t *lc = tk_dvec_create(L, unl);
      lc->n = unl;
      int lc_idx = lua_gettop(L);
      memcpy(lc->a, label_counts, unl * sizeof(double));
      free(label_counts);
      tk_gram_finalize_tiled(L, XtX, col_mean, y_mean_arr,
        eigenvals, pqty_f, pqty_buf != NULL, lc, lc_idx,
        (int64_t)nc, (int64_t)d, gram_nl, tile_labels);
      gram_result_idx = lua_gettop(L);
      if (pqty_buf_idx > 0) {
        lua_getfenv(L, gram_result_idx);
        lua_pushvalue(L, pqty_buf_idx);
        lua_setfield(L, -2, "pqty_buf");
        lua_pop(L, 1);
      }
    }
    }
  } else if (build_gram) {
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
    int64_t tile_size = sample_tile_size;
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
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)d, (int)gram_nl, (int)bs, 1.0, tile_buf, (int)bs,
        gram_targets_dv->a + (uint64_t)base * unl, (int)gram_nl,
        1.0, xty, (int)gram_nl);
    }
    free(tile_buf);
    if (!ctx->full_chol_external) free(full_chol);
    ctx->full_chol = NULL;
    for (uint64_t j = 0; j < d; j++)
      col_mean[j] /= (double)nc;
    for (int64_t l = 0; l < gram_nl; l++) {
      double s = 0.0;
      for (uint64_t i = 0; i < nc; i++)
        s += gram_targets_dv->a[i * unl + (uint64_t)l];
      y_mean_arr[l] = s / (double)nc;
    }
    if (do_solve) {
      free(eigenvals);
      tk_gram_finalize_cholesky(L, XtX, xty, col_mean, y_mean_arr,
        NULL, 0, (int64_t)nc, (int64_t)d, gram_nl,
        solve_lambda, false, 0.0, 0.0);
    } else {
      tk_gram_finalize(L, XtX, xty, col_mean, y_mean_arr,
        eigenvals, NULL, 0, (int64_t)nc, (int64_t)d, gram_nl);
    }
    gram_result_idx = lua_gettop(L);
  }

  if (!build_gram && !build_gram_tiled) {
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
      #pragma omp parallel for schedule(static) collapse(2)
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
    if (!ctx->full_chol_external) free(full_chol);
    ctx->full_chol = NULL;
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
  enc->csc_offsets = NULL;
  enc->csc_rows = NULL;
  enc->csc_values = NULL;
  enc->csr_n_tokens = 0;
  enc->dense_vecs = NULL;
  enc->d_input = 0;
  enc->sims_buf = NULL;
  enc->row_bufs = NULL;
  enc->dense_tile_buf = NULL;
  enc->tile = 0;
  enc->n_threads = 0;

  if (has_csr) {
    uint64_t csr_nt = mod.csr_n_tokens;
    enc->csr_n_tokens = csr_nt;
    uint64_t lm_total = 0;
    for (uint64_t j = 0; j < m; j++) {
      uint64_t si = (uint64_t)lm_ids->a[j];
      lm_total += (uint64_t)(mod.csr_offsets[si + 1] - mod.csr_offsets[si]);
    }
    enc->csr_offsets = (int64_t *)malloc((m + 1) * sizeof(int64_t));
    enc->csr_tokens = (int32_t *)malloc(lm_total * sizeof(int32_t));
    enc->csr_values = (float *)malloc(lm_total * sizeof(float));
    enc->csr_offsets[0] = 0;
    for (uint64_t j = 0; j < m; j++) {
      uint64_t si = (uint64_t)lm_ids->a[j];
      int64_t lo = mod.csr_offsets[si], hi = mod.csr_offsets[si + 1];
      int64_t cnt = hi - lo;
      memcpy(enc->csr_tokens + enc->csr_offsets[j],
             mod.csr_tokens + lo,
             (uint64_t)cnt * sizeof(int32_t));
      memcpy(enc->csr_values + enc->csr_offsets[j],
             mod.csr_values + lo,
             (uint64_t)cnt * sizeof(float));
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
        int32_t tok = enc->csr_tokens[a];
        int64_t p = csc_pos[tok]++;
        enc->csc_rows[p] = (int64_t)j;
        enc->csc_values[p] = enc->csr_values[a];
      }
    }
    free(csc_pos);
    free(csr_values_owned);
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
    free(dense_from_fvec);
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
      int32_t tok = enc->csr_tokens[a];
      int64_t p = csc_pos[tok]++;
      enc->csc_rows[p] = (int64_t)j;
      enc->csc_values[p] = enc->csr_values[a];
    }
  }
  free(csc_pos);
}

static inline int tk_nystrom_encoder_load_lua (lua_State *L) {
  lua_settop(L, 1);
  const char *data = luaL_checkstring(L, 1);
  FILE *fh = tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKny", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid nystrom encoder file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 24) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported nystrom encoder version %d", (int)version);
  }

  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->destroyed = false;
  enc->projection = NULL;
  enc->csr_offsets = NULL; enc->csr_tokens = NULL; enc->csr_values = NULL;
  enc->csc_offsets = NULL; enc->csc_rows = NULL; enc->csc_values = NULL;
  enc->csr_n_tokens = 0;
  enc->dense_vecs = NULL; enc->d_input = 0;
  enc->sims_buf = NULL; enc->row_bufs = NULL; enc->dense_tile_buf = NULL;
  enc->tile = 0; enc->n_threads = 0;

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
    enc->csr_tokens = (int32_t *)malloc(total_csr * sizeof(int32_t));
    enc->csr_values = (float *)malloc(total_csr * sizeof(float));
    tk_lua_fread(L, enc->csr_tokens, sizeof(int32_t), total_csr, fh);
    tk_lua_fread(L, enc->csr_values, sizeof(float), total_csr, fh);
    tk_nystrom_build_csc(enc);
  } else if (enc->mod_type == TK_MOD_DENSE) {
    tk_lua_fread(L, &enc->d_input, sizeof(int64_t), 1, fh);
    enc->dense_vecs = (float *)malloc(enc->m * (uint64_t)enc->d_input * sizeof(float));
    tk_lua_fread(L, enc->dense_vecs, sizeof(float), enc->m * (uint64_t)enc->d_input, fh);
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
