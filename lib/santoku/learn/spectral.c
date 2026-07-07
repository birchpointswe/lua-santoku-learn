#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/ivec.h>
#include <santoku/svec.h>
#include <santoku/cvec.h>
#include <santoku/learn/gram.h>
#include <santoku/csr.h>
#include <santoku/mtx.h>
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

typedef enum {
  TK_SPECTRAL_COSINE = 0,
  TK_SPECTRAL_MATERN = 1,
  TK_SPECTRAL_ARCCOS = 2,
} tk_spectral_family_t;

typedef struct {
  uint8_t family;
  uint8_t nu;
  uint8_t order;
  uint8_t depth;
  uint8_t tangent;
  float gamma;
} tk_spectral_kernel_t;

static inline float tk_arccos_raw (int n, float c) {
  if (c > 1.0f) c = 1.0f;
  if (c < -1.0f) c = -1.0f;
  float t = acosf(c), s = sinf(t), ch = cosf(t), P = (float)M_PI - t;
  float c2 = ch * ch, c4 = c2 * c2, c6 = c4 * c2;
  float j;
  if (n <= 0)      j = P;
  else if (n == 1) j = s + P * ch;
  else if (n == 2) j = 3.0f * s * ch + P * (1.0f + 2.0f * c2);
  else if (n == 3) j = s * (4.0f + 11.0f * c2) + P * ch * (9.0f + 6.0f * c2);
  else if (n == 4) j = 5.0f * ch * s * (11.0f + 10.0f * c2) + 3.0f * P * (3.0f + 24.0f * c2 + 8.0f * c4);
  else if (n == 5) j = s * (64.0f + 607.0f * c2 + 274.0f * c4)
                       + 15.0f * P * ch * (15.0f + 40.0f * c2 + 8.0f * c4);
  else             j = 21.0f * ch * s * (99.0f + 312.0f * c2 + 84.0f * c4)
                       + 45.0f * P * (5.0f + 90.0f * c2 + 120.0f * c4 + 16.0f * c6);
  return j / (float)M_PI;
}

static inline float tk_arccos_norm (int n, float c) {
  static const float nrm[7] = { 1.0f, 1.0f, 3.0f, 15.0f, 105.0f, 945.0f, 10395.0f };
  int i = n < 0 ? 0 : (n > 6 ? 6 : n);
  return tk_arccos_raw(i, c) / nrm[i];
}

static inline float tk_arccos_deriv (int n, float c) {
  return tk_arccos_norm(n > 0 ? n - 1 : 0, c);
}

static inline float tk_arccos_apply (int order, int depth, int tangent, float c) {
  float sigma = c, ntk = c;
  for (int l = 0; l < depth; l++) {
    float dot = tk_arccos_deriv(order, sigma);
    sigma = tk_arccos_norm(order, sigma);
    ntk = sigma + ntk * dot;
  }
  return tangent ? ntk / (float)(depth + 1) : sigma;
}

static inline float tk_matern_apply (int nu, float gamma, float c) {
  if (nu == 3) return expf(-gamma * (1.0f - c));
  float d2 = 2.0f * (1.0f - c);
  if (d2 < 0.0f) d2 = 0.0f;
  float nuf = (nu == 0) ? 0.5f : (nu == 1 ? 1.5f : 2.5f);
  float arg = sqrtf(2.0f * nuf * d2 * gamma);
  if (nu == 0) return expf(-arg);
  if (nu == 1) return (1.0f + arg) * expf(-arg);
  return (1.0f + arg + arg * arg / 3.0f) * expf(-arg);
}
static inline float tk_spectral_kernel_apply (tk_spectral_kernel_t k, float c) {
  if (c > 1.0f) c = 1.0f;
  if (c < -1.0f) c = -1.0f;
  if (k.family == TK_SPECTRAL_MATERN) return tk_matern_apply(k.nu, k.gamma, c);
  if (k.family == TK_SPECTRAL_ARCCOS) return tk_arccos_apply(k.order, k.depth, k.tangent, c);
  return c;
}

#define TK_MOD_CSR   0
#define TK_MOD_DENSE 1
#define TK_MAX_MOD 16

typedef struct {
  uint8_t type;
  const int64_t *csr_offsets;
  const int32_t *csr_tokens;
  const float *csr_values;
  uint64_t csr_n_tokens;
  const float *dense;
  const float *dense_cs2;
  int64_t d_input;
  const float *rowscale;
  int n_blocks;
  const int64_t *blk_off[TK_MAX_MOD];
  const int32_t *blk_tok[TK_MAX_MOD];
  const float *blk_val[TK_MAX_MOD];
  const float *blk_cs [TK_MAX_MOD];
  int64_t blk_start[TK_MAX_MOD];
  float blk_s [TK_MAX_MOD];
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
#define TK_SIMS_CHUNK 512
#define TK_SIMS_CHUNK_MEM (256ull << 20)

static inline double tk_lm_drand (uint64_t *s) {
  uint64_t x = *s;
  unsigned int count = (unsigned int) (x >> 61);
  *s = x * tk_fast_multiplier;
  return ((double) ((uint32_t) ((x ^ x >> 22) >> (22 + count)))) / ((double) UINT32_MAX);
}

static inline int tk_spectral_pivot_sims (
  tk_spectral_modality_t *mod,
  tk_spectral_kernel_t kernel,
  uint64_t n_docs,
  const uint64_t *blk_pivots,
  uint64_t np,
  float *kip_block,
  float *pivot_dense_rows,
  int64_t *piv_csc_off,
  int64_t *piv_csc_pos,
  uint16_t **piv_csc_piv_p,
  float **piv_csc_val_p,
  uint64_t *piv_csc_cap_p
) {
  uint16_t *piv_csc_piv = *piv_csc_piv_p;
  float *piv_csc_val = *piv_csc_val_p;
  uint64_t piv_csc_cap = *piv_csc_cap_p;

  // no upfront memset: block 0 (multi-block CSR) writes through, plain-CSR assigns, dense sgemm
  // has beta=0, so kip_block is fully overwritten by every path
  if (mod->type == TK_MOD_CSR && mod->n_blocks > 0) {
    int64_t plo_arr[TK_SIMS_CHUNK], phi_arr[TK_SIMS_CHUNK];
    for (int bl = 0; bl < mod->n_blocks; bl++) {
      const int64_t *boff = mod->blk_off[bl];
      const int32_t *btok = mod->blk_tok[bl];
      const float   *bval = mod->blk_val[bl];
      const float   *bcs  = mod->blk_cs[bl];
      float s = mod->blk_s[bl];
      uint64_t bnt = (bl + 1 < mod->n_blocks)
        ? (uint64_t)(mod->blk_start[bl + 1] - mod->blk_start[bl])
        : (mod->csr_n_tokens - (uint64_t) mod->blk_start[bl]);
      for (uint64_t b = 0; b < np; b++) {
        uint64_t pv = (uint64_t) blk_pivots[b];
        plo_arr[b] = boff[pv];
        phi_arr[b] = boff[pv + 1];
      }
      for (uint64_t b = 0; b < np; b++) {
        int64_t p = plo_arr[b];
        while (p < phi_arr[b]) { int32_t tok = btok[p]; piv_csc_off[tok + 1]++;
          while (p < phi_arr[b] && btok[p] == tok) p++; }
      }
      for (uint64_t t = 0; t < bnt; t++) piv_csc_off[t + 1] += piv_csc_off[t];
      uint64_t total_pnnz = (uint64_t) piv_csc_off[bnt];
      if (total_pnnz > piv_csc_cap) {
        uint16_t *npv = (uint16_t *) realloc(piv_csc_piv, total_pnnz * sizeof(uint16_t));
        if (npv) piv_csc_piv = npv;
        float *nvl = (float *) realloc(piv_csc_val, total_pnnz * sizeof(float));
        if (nvl) piv_csc_val = nvl;
        *piv_csc_piv_p = piv_csc_piv;
        *piv_csc_val_p = piv_csc_val;
        if (!npv || !nvl) return -1;
        piv_csc_cap = total_pnnz;
        *piv_csc_cap_p = piv_csc_cap;
      }
      memcpy(piv_csc_pos, piv_csc_off, bnt * sizeof(int64_t));
      for (uint64_t b = 0; b < np; b++) {
        int64_t p = plo_arr[b];
        while (p < phi_arr[b]) {
          int32_t tok = btok[p]; float sum = 0.0f;
          while (p < phi_arr[b] && btok[p] == tok) { sum += bval[p]; p++; }
          int64_t pos = piv_csc_pos[tok]++;
          piv_csc_piv[pos] = (uint16_t) b;
          piv_csc_val[pos] = sum * (bcs ? bcs[tok] : 1.0f) * s;
        }
      }
      const int64_t *restrict pc_off = piv_csc_off;
      const uint16_t *restrict pc_piv = piv_csc_piv;
      const float *restrict pc_val = piv_csc_val;
      const int64_t *restrict bo = boff;
      const int32_t *restrict bt = btok;
      const float *restrict bv = bval;
      const float *restrict bw = bcs;
      float ss = s;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_docs; i++) {
        double kip_row[TK_SIMS_CHUNK];
        memset(kip_row, 0, np * sizeof(double));
        int64_t jlo = bo[i], jhi = bo[i + 1];
        for (int64_t j = jlo; j < jhi; j++) {
          int32_t tok = bt[j];
          double val = (double) bv[j] * (bw ? (double) bw[tok] : 1.0) * (double) ss;
          int64_t clo = pc_off[tok], chi = pc_off[tok + 1];
          for (int64_t c = clo; c < chi; c++)
            kip_row[pc_piv[c]] += val * (double) pc_val[c];
        }
        if (bl == 0)
          for (uint64_t b = 0; b < np; b++)
            kip_block[i * np + b] = (float) kip_row[b];
        else
          for (uint64_t b = 0; b < np; b++)
            kip_block[i * np + b] += (float) kip_row[b];
      }
      memset(piv_csc_off, 0, (bnt + 1) * sizeof(int64_t));
    }
    const float *restrict rs = mod->rowscale;
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_docs; i++) {
      double rsi = rs ? (double) rs[i] : 1.0;
      for (uint64_t b = 0; b < np; b++) {
        double rsp = rs ? (double) rs[(uint64_t) blk_pivots[b]] : 1.0;
        kip_block[i * np + b] = tk_spectral_kernel_apply(kernel,
          (float)((double) kip_block[i * np + b] * rsi * rsp));
      }
    }

  } else if (mod->type == TK_MOD_CSR) {
    const int64_t *csr_offsets = mod->csr_offsets;
    const int32_t *csr_tokens = mod->csr_tokens;
    const float *csr_values = mod->csr_values;
    uint64_t csr_n_tokens = mod->csr_n_tokens;
    int64_t plo_arr[TK_SIMS_CHUNK], phi_arr[TK_SIMS_CHUNK];
    float piv_rs[TK_SIMS_CHUNK];
    for (uint64_t b = 0; b < np; b++) {
      uint64_t pv = (uint64_t) blk_pivots[b];
      plo_arr[b] = csr_offsets[pv];
      phi_arr[b] = csr_offsets[pv + 1];
      piv_rs[b] = mod->rowscale ? mod->rowscale[pv] : 1.0f;
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
        uint16_t *npv = (uint16_t *)realloc(piv_csc_piv, total_pnnz * sizeof(uint16_t));
        if (npv) piv_csc_piv = npv;
        float *nvl = (float *)realloc(piv_csc_val, total_pnnz * sizeof(float));
        if (nvl) piv_csc_val = nvl;
        *piv_csc_piv_p = piv_csc_piv;
        *piv_csc_val_p = piv_csc_val;
        if (!npv || !nvl) return -1;
        piv_csc_cap = total_pnnz;
        *piv_csc_cap_p = piv_csc_cap;
      }
      memcpy(piv_csc_pos, piv_csc_off, csr_n_tokens * sizeof(int64_t));
      for (uint64_t b = 0; b < np; b++) {
        int64_t p = plo_arr[b];
        while (p < phi_arr[b]) {
          int32_t tok = csr_tokens[p];
          float sum = 0.0f;
          while (p < phi_arr[b] && csr_tokens[p] == tok) { sum += csr_values[p]; p++; }
          int64_t pos = piv_csc_pos[tok]++;
          piv_csc_piv[pos] = (uint16_t)b;
          piv_csc_val[pos] = sum * piv_rs[b];
        }
      }
      const int64_t *restrict pc_off = piv_csc_off;
      const uint16_t *restrict pc_piv = piv_csc_piv;
      const float *restrict pc_val = piv_csc_val;
      const int64_t *restrict c_off = csr_offsets;
      const int32_t *restrict c_tok = csr_tokens;
      const float *restrict c_val = csr_values;
      const float *restrict c_rs = mod->rowscale;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_docs; i++) {
        double kip_row[TK_SIMS_CHUNK];
        memset(kip_row, 0, np * sizeof(double));
        int64_t jlo = c_off[i], jhi = c_off[i + 1];
        for (int64_t j = jlo; j < jhi; j++) {
          int32_t tok = c_tok[j];
          double val = (double)c_val[j];
          int64_t clo = pc_off[tok], chi = pc_off[tok + 1];
          for (int64_t c = clo; c < chi; c++)
            kip_row[pc_piv[c]] += val * (double)pc_val[c];
        }
        double rsi = c_rs ? (double)c_rs[i] : 1.0;
        for (uint64_t b = 0; b < np; b++)
          kip_block[i * np + b] = (float)(kip_row[b] * rsi);
      }
      memset(piv_csc_off, 0, (csr_n_tokens + 1) * sizeof(int64_t));
    }
    {
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_docs; i++)
        for (uint64_t b = 0; b < np; b++)
          kip_block[i * np + b] = tk_spectral_kernel_apply(kernel,
            kip_block[i * np + b]);
    }

  } else if (mod->type == TK_MOD_DENSE) {
    int64_t di = mod->d_input;
    const float *dense = mod->dense;
    for (uint64_t b = 0; b < np; b++)
      memcpy(pivot_dense_rows + b * (uint64_t)di,
             dense + blk_pivots[b] * (uint64_t)di,
             (uint64_t)di * sizeof(float));
    if (mod->dense_cs2)
      for (uint64_t b = 0; b < np; b++)
        for (int64_t k = 0; k < di; k++)
          pivot_dense_rows[b * (uint64_t)di + (uint64_t)k] *= mod->dense_cs2[k];
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      (int)n_docs, (int)np, (int)di, 1.0f,
      dense, (int)di,
      pivot_dense_rows, (int)di,
      0.0f, kip_block, (int)np);
    {
      const float *drs = mod->rowscale;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_docs; i++)
        for (uint64_t b = 0; b < np; b++) {
          float raw = kip_block[i * np + b];
          if (drs) raw = (float)((double)raw * (double)drs[i] * (double)drs[blk_pivots[b]]);
          kip_block[i * np + b] = tk_spectral_kernel_apply(kernel, raw);
        }
    }
  }
  return 0;
}

static inline void tk_spectral_sample_landmarks (
  lua_State *L,
  tk_spectral_modality_t *mod,
  uint64_t n_samples,
  tk_spectral_kernel_t kernel,
  uint64_t n_landmarks,
  double trace_tol,
  uint64_t chol_block,
  float *ext_chol,
  // optional external buffer for the m x m Cholesky factor (proj_buf, search path only): forced
  // mode factors K_SS directly into it, so tm_encode skips the fvec alloc + m^2 memcpy. When used,
  // *chol_out is NULL and *chol_ext_out points here (caller-owned, not freed).
  float *ext_kss,
  uint64_t ext_kss_cap,
  tk_ivec_t **ids_out,
  tk_fvec_t **chol_out,
  float **chol_ext_out,
  float **full_chol_out,
  uint64_t *n_docs_out,
  uint64_t *actual_landmarks_out,
  double *trace_ratio_out,
  const int64_t *forced,
  uint64_t forced_n,
  uint64_t *backstop_out
) {
  uint64_t n_docs = n_samples;
  if (chol_block == 0) chol_block = 64;
  if (chol_block > TK_CHOL_BLOCK) chol_block = TK_CHOL_BLOCK;

  uint64_t lm_state = tk_hash_mix(0x9E3779B97F4A7C15ull);

  if (n_landmarks == 0 || n_landmarks > n_docs)
    n_landmarks = n_docs;
  if (forced && forced_n < n_landmarks)
    n_landmarks = forced_n;
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
    // forced mode fully overwrites rows [0,m0) via the sims transpose then strsm-in-place, and
    // never reads rows [m0,n_landmarks); adaptive needs the zeros (unwritten cols read as 0)
    if (!forced)
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

  uint64_t sims_chunk = chol_block;
  if (forced) {
    sims_chunk = TK_SIMS_CHUNK;
    while (sims_chunk > chol_block && n_docs * sims_chunk * sizeof(float) > TK_SIMS_CHUNK_MEM)
      sims_chunk >>= 1;
    if (sims_chunk < chol_block) sims_chunk = chol_block;
  }
  float *kip_block = (float *)malloc(n_docs * sims_chunk * sizeof(float));
  float *cross_dots = (float *)malloc(n_docs * chol_block * sizeof(float));
  float *pivot_prev_L = (float *)malloc(chol_block * n_landmarks * sizeof(float));
  int64_t max_d_input = 0;
  if (mod->type == TK_MOD_DENSE && mod->d_input > max_d_input)
    max_d_input = mod->d_input;
  float *pivot_dense_rows = max_d_input > 0
    ? (float *)malloc(sims_chunk * (uint64_t)max_d_input * sizeof(float)) : NULL;
  uint64_t blk_pivots[TK_CHOL_BLOCK];

  double *proposal = (double *)malloc(n_docs * sizeof(double));
  double *proposal_cdf = (double *)malloc(n_docs * sizeof(double));

  int64_t *piv_csc_off = NULL;
  uint16_t *piv_csc_piv = NULL;
  float *piv_csc_val = NULL;
  int64_t *piv_csc_pos = NULL;
  uint64_t piv_csc_cap = 0;
  if (mod->type == TK_MOD_CSR) {
    uint64_t csr_n_tokens = mod->csr_n_tokens;
    piv_csc_off = (int64_t *)calloc(csr_n_tokens + 1, sizeof(int64_t));
    piv_csc_pos = (int64_t *)malloc(csr_n_tokens * sizeof(int64_t));
    uint64_t max_nnz = 0;
    if (mod->n_blocks > 0) {
      for (int bl = 0; bl < mod->n_blocks; bl++) {
        const int64_t *off = mod->blk_off[bl];
        for (uint64_t i = 0; i < n_docs; i++) {
          uint64_t nnz = (uint64_t)(off[i + 1] - off[i]);
          if (nnz > max_nnz) max_nnz = nnz;
        }
      }
    } else {
      for (uint64_t i = 0; i < n_docs; i++) {
        uint64_t nnz = (uint64_t)(mod->csr_offsets[i + 1] - mod->csr_offsets[i]);
        if (nnz > max_nnz) max_nnz = nnz;
      }
    }
    piv_csc_cap = sims_chunk * max_nnz;
    piv_csc_piv = (uint16_t *)malloc(piv_csc_cap * sizeof(uint16_t));
    piv_csc_val = (float *)malloc(piv_csc_cap * sizeof(float));
  }

  if (!kip_block || !cross_dots || !pivot_prev_L
      || (max_d_input > 0 && !pivot_dense_rows)
      || !proposal || !proposal_cdf
      || (mod->type == TK_MOD_CSR && (!piv_csc_off || !piv_csc_piv || !piv_csc_val || !piv_csc_pos))) {
    free(kip_block); free(cross_dots);
    free(pivot_prev_L);
    free(pivot_dense_rows);
    free(proposal); free(proposal_cdf);
    free(piv_csc_off); free(piv_csc_piv); free(piv_csc_val); free(piv_csc_pos);
    luaL_error(L, "sample_landmarks: out of memory (buffers)");
    return;
  }

  #pragma omp parallel for
  for (uint64_t i = 0; i < n_docs; i++)
    residual[i] = 1.0;
  initial_trace = (double)n_docs;

  double proposal_total = initial_trace;
  tk_fvec_t *chol = NULL;
  float *chol_ext = NULL;   // set when the factor is written into ext_kss instead of an fvec

  if (forced) {
    uint64_t *S = (uint64_t *) malloc(n_landmarks * sizeof(uint64_t));
    if (!S) {
      free(kip_block); free(cross_dots); free(pivot_prev_L); free(pivot_dense_rows);
      free(proposal); free(proposal_cdf);
      free(piv_csc_off); free(piv_csc_piv); free(piv_csc_val); free(piv_csc_pos);
      luaL_error(L, "sample_landmarks: out of memory (forced ids)");
      return;
    }
    uint64_t m0 = 0;
    for (uint64_t i = 0; i < forced_n && m0 < n_landmarks; i++) {
      uint64_t pi = (uint64_t) forced[i];
      if (pi < n_docs) S[m0++] = pi;
    }
    for (uint64_t base = 0; base < m0; base += sims_chunk) {
      uint64_t np = m0 - base;
      if (np > sims_chunk) np = sims_chunk;
      if (tk_spectral_pivot_sims(mod, kernel, n_docs, S + base, np, kip_block,
          pivot_dense_rows, piv_csc_off, piv_csc_pos,
          &piv_csc_piv, &piv_csc_val, &piv_csc_cap) != 0) {
        free(S);
        free(kip_block); free(cross_dots); free(pivot_prev_L); free(pivot_dense_rows);
        free(proposal); free(proposal_cdf);
        free(piv_csc_off); free(piv_csc_piv); free(piv_csc_val); free(piv_csc_pos);
        luaL_error(L, "sample_landmarks: out of memory (csc)");
        return;
      }
      const float *kb = kip_block;
      float *dst = L_mat + base * n_docs;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_docs; i++)
        for (uint64_t b = 0; b < np; b++)
          dst[b * n_docs + i] = kb[i * np + b];
    }
    if (m0 > 0) {
      for (uint64_t k = 0; k < m0; k++) {
        landmark_sids[k] = sid_map[S[k]];
        landmark_idx_map[k] = (int64_t) S[k];
      }
      float *kss;
      if (ext_kss && ext_kss_cap >= m0 * m0) {
        kss = ext_kss;         // factor straight into proj_buf: no fvec alloc, no m^2 memcpy
        chol_ext = ext_kss;
      } else {
        chol = tk_fvec_create(NULL, m0 * m0);
        kss = chol->a;
      }
      double jitter = 1e-6;
      for (;;) {
        #pragma omp parallel for schedule(static)
        for (uint64_t r = 0; r < m0; r++) {
          uint64_t sr = S[r];
          for (uint64_t c = 0; c < m0; c++)
            kss[r * m0 + c] = L_mat[c * n_docs + sr];
          if (jitter > 0.0)
            kss[r * m0 + r] += (float) jitter;
        }
        if (LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', (int) m0, kss, (int) m0) == 0)
          break;
        if (jitter > 1e-1) {
          if (chol) tk_fvec_destroy(chol);
          free(S);
          free(kip_block); free(cross_dots); free(pivot_prev_L); free(pivot_dense_rows);
          free(proposal); free(proposal_cdf);
          free(piv_csc_off); free(piv_csc_piv); free(piv_csc_val); free(piv_csc_pos);
          luaL_error(L, "sample_landmarks: forced spotrf failed after jitter escalation");
          return;
        }
        jitter *= 10.0;
        if (backstop_out) (*backstop_out) ++;
      }
      #pragma omp parallel for schedule(static)
      for (uint64_t r = 0; r < m0; r++)
        for (uint64_t c = r + 1; c < m0; c++)
          kss[r * m0 + c] = 0.0f;
      if (chol) chol->n = m0 * m0;
      cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
        (int) m0, (int) n_docs, 1.0f, kss, (int) m0, L_mat, (int) n_docs);
      double covered = 0.0;
      #pragma omp parallel for reduction(+:covered)
      for (uint64_t k = 0; k < m0 * n_docs; k++)
        covered += (double) L_mat[k] * (double) L_mat[k];
      trace = (double) n_docs - covered;
      if (trace < 0.0) trace = 0.0;
      actual_landmarks = m0;
    }
    free(S);
    done = true;
  }

  if (!forced) {
    memcpy(proposal, residual, n_docs * sizeof(double));
    double cum = 0.0;
    for (uint64_t i = 0; i < n_docs; i++) {
      cum += proposal[i];
      proposal_cdf[i] = cum;
    }
  }

  #define SAMPLE_PROPOSAL() ({ \
    double _r = tk_lm_drand(&lm_state) * proposal_total; \
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
    if (max_blk > chol_block) max_blk = chol_block;

    uint64_t np = 0;
    uint64_t n_propose = max_blk * 2;
    if (n_propose > chol_block) n_propose = chol_block;
    uint64_t n_valid = 0;
    for (uint64_t b = 0; b < n_propose && np < max_blk; b++) {
      uint64_t pi = SAMPLE_PROPOSAL();
      if (residual[pi] < 1.2e-6) continue;
      int dup = 0;
      for (uint64_t k = 0; k < np; k++)
        if (blk_pivots[k] == pi) { dup = 1; break; }
      if (dup) continue;
      n_valid++;
      if (proposal[pi] > 1e-15) {
        double accept_prob = residual[pi] / proposal[pi];
        if (tk_lm_drand(&lm_state) > accept_prob) continue;
      } else {
        continue;
      }
      blk_pivots[np] = pi;
      np++;
    }
    total_proposed += n_propose;
    total_accepted += np;
    if (n_valid == 0) { done = true; break; }
    if (np == 0) continue;

    if (tk_spectral_pivot_sims(mod, kernel, n_docs, blk_pivots, np, kip_block,
        pivot_dense_rows, piv_csc_off, piv_csc_pos,
        &piv_csc_piv, &piv_csc_val, &piv_csc_cap) != 0) {
      free(kip_block); free(cross_dots); free(pivot_prev_L); free(pivot_dense_rows);
      free(proposal); free(proposal_cdf);
      free(piv_csc_off); free(piv_csc_piv); free(piv_csc_val); free(piv_csc_pos);
      luaL_error(L, "sample_landmarks: out of memory (csc)");
      return;
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
      uint64_t col = actual_landmarks;
      double sc_sq = residual[pi];
      if (sc_sq < 1.2e-6)
        continue;
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

  tk_ivec_t *landmark_ids = tk_ivec_create(L, actual_landmarks);
  for (uint64_t i = 0; i < actual_landmarks; i++)
    landmark_ids->a[i] = landmark_sids[i];
  landmark_ids->n = actual_landmarks;

  if (!chol && !chol_ext) {
    chol = tk_fvec_create(NULL, actual_landmarks * actual_landmarks);
    #pragma omp parallel for schedule(static)
    for (uint64_t li = 0; li < actual_landmarks; li++) {
      uint64_t doc_idx = (uint64_t)landmark_idx_map[li];
      for (uint64_t k = 0; k < actual_landmarks; k++)
        chol->a[li * actual_landmarks + k] = L_mat[k * n_docs + doc_idx];
    }
    chol->n = actual_landmarks * actual_landmarks;
  }

  if (!ext_chol && actual_landmarks > 0 && actual_landmarks < n_landmarks) {
    float *shrunk = realloc(L_mat, actual_landmarks * n_docs * sizeof(float));
    if (shrunk) L_mat = shrunk;
  }

  float *full_chol = L_mat;
  ctx->L_mat = NULL;

  lua_remove(L, ctx_idx);

  *ids_out = landmark_ids;
  *chol_out = chol;
  *chol_ext_out = chol_ext;
  *full_chol_out = full_chol;
  *n_docs_out = n_docs;
  *actual_landmarks_out = actual_landmarks;
  *trace_ratio_out = (initial_trace > 0.0) ? (trace / initial_trace) : 0.0;
}

typedef struct {
  tk_fvec_t *lm_chol;
  float *full_chol;
  int full_chol_external;
} tk_encode_nystrom_ctx_t;

static inline int tk_encode_nystrom_ctx_gc (lua_State *L) {
  tk_encode_nystrom_ctx_t *c = (tk_encode_nystrom_ctx_t *)lua_touserdata(L, 1);
  if (c->lm_chol) tk_fvec_destroy(c->lm_chol);
  if (!c->full_chol_external) free(c->full_chol);
  return 0;
}

#define TK_NYSTROM_ENCODER_MT "tk_nystrom_encoder_t"

typedef struct {
  float *chol;
  tk_fvec_t *chol_f;
  uint64_t m;
  uint64_t d;
  double trace_ratio;
  tk_spectral_kernel_t kernel;
  uint8_t mod_type;
  int64_t *csr_offsets;
  int32_t *csr_tokens;
  float *csr_values;
  int64_t *csc_offsets;
  int16_t *csc_rows;
  float *csc_values;
  uint64_t csr_n_tokens;
  int n_blocks;
  int64_t blk_start[TK_MAX_MOD];
  float blk_s[TK_MAX_MOD];
  float *dense_vecs;
  float *dense_cs2;
  int64_t d_input;
  float *sims_buf;
  int sims_buf_external;
  uint64_t sims_buf_cap;
  float *row_bufs;
  int row_bufs_external;
  uint64_t row_bufs_cap;
  float *dense_tile_buf;
  uint64_t tile;
  int n_threads;
  bool destroyed;
} tk_nystrom_encoder_t;

static inline tk_nystrom_encoder_t *tk_nystrom_encoder_peek (lua_State *L, int i) {
  return (tk_nystrom_encoder_t *)luaL_checkudata(L, i, TK_NYSTROM_ENCODER_MT);
}

static inline int tk_nystrom_build_csc (tk_nystrom_encoder_t *enc);

static inline int tk_nystrom_encoder_gc (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (!enc->destroyed) {
    if (enc->chol_f) tk_fvec_destroy(enc->chol_f);
    if (enc->mod_type == TK_MOD_CSR) {
      free(enc->csr_offsets);
      free(enc->csr_tokens);
      free(enc->csr_values);
      free(enc->csc_offsets);
      free(enc->csc_rows);
      free(enc->csc_values);
    } else if (enc->mod_type == TK_MOD_DENSE) {
      free(enc->dense_vecs);
      free(enc->dense_cs2);
    }
    if (!enc->sims_buf_external) free(enc->sims_buf);
    if (!enc->row_bufs_external) free(enc->row_bufs);
    free(enc->dense_tile_buf);
    enc->destroyed = true;
  }
  return 0;
}

static inline void tk_nystrom_encoder_zero (tk_nystrom_encoder_t *enc) {
  enc->chol = NULL;
  enc->chol_f = NULL;
  enc->csr_offsets = NULL; enc->csr_tokens = NULL; enc->csr_values = NULL;
  enc->csc_offsets = NULL; enc->csc_rows = NULL; enc->csc_values = NULL;
  enc->csr_n_tokens = 0;
  enc->n_blocks = 0;
  enc->dense_vecs = NULL;
  enc->dense_cs2 = NULL;
  enc->d_input = 0;
  enc->sims_buf = NULL; enc->sims_buf_external = 0; enc->sims_buf_cap = 0;
  enc->row_bufs = NULL; enc->row_bufs_external = 0; enc->row_bufs_cap = 0;
  enc->dense_tile_buf = NULL;
  enc->tile = 0;
  enc->n_threads = 0;
  enc->destroyed = false;
}

static inline int tk_spectral_block_rowscale (
  int nb, const int64_t *const *boff, const int32_t *const *btok,
  const float *const *bval, const float *const *bcs,
  const float *bs, uint64_t n_samples, float *rowscale)
{
  double *acc = (double *) calloc(n_samples, sizeof(double));
  if (!acc) return -1;
  for (int b = 0; b < nb; b++) {
    double s2 = (double) bs[b] * (double) bs[b];
    const int64_t *off = boff[b]; const float *val = bval[b];
    const int32_t *tok = btok[b]; const float *cs = bcs[b];
    for (uint64_t i = 0; i < n_samples; i++) {
      int64_t lo = off[i], hi = off[i + 1];
      double ss = 0.0;
      if (cs) for (int64_t j = lo; j < hi; j++) {
        double cv = (double) cs[tok[j]] * (val ? (double) val[j] : 1.0);
        ss += cv * cv;
      }
      else if (val) for (int64_t j = lo; j < hi; j++) ss += (double) val[j] * (double) val[j];
      else ss = (double) (hi - lo);
      acc[i] += s2 * ss;
    }
  }
  for (uint64_t i = 0; i < n_samples; i++)
    rowscale[i] = acc[i] > 0.0 ? (float) (1.0 / sqrt(acc[i])) : 0.0f;
  free(acc);
  return 0;
}

static inline int tk_nystrom_ensure_sims (tk_nystrom_encoder_t *enc, uint64_t m) {
  if (enc->tile == 0) {
    uint64_t tile = 4096;
    while (tile > 1 && tile * m * sizeof(float) > 256ULL * 1024 * 1024) tile /= 2;
    if (enc->sims_buf_external)
      while (tile > 1 && (uint64_t) tile * m > enc->sims_buf_cap) tile /= 2;
    enc->tile = tile;
  }
  if (!enc->sims_buf)
    enc->sims_buf = (float *) malloc((uint64_t) enc->tile * m * sizeof(float));
  return enc->sims_buf ? 0 : -1;
}

static inline int tk_nystrom_ensure_row_bufs (tk_nystrom_encoder_t *enc, uint64_t m) {
  int nt = omp_get_max_threads();
  if (!enc->row_bufs_external && (!enc->row_bufs || nt > enc->n_threads)) {
    free(enc->row_bufs);
    enc->row_bufs = (float *) calloc((uint64_t) nt * m, sizeof(float));
    enc->n_threads = nt;
    if (!enc->row_bufs) return -1;
  }
  if (enc->row_bufs_external)
    memset(enc->row_bufs, 0, (uint64_t) enc->n_threads * m * sizeof(float));
  return 0;
}

static inline void tk_nystrom_csr_tiles (
  tk_nystrom_encoder_t *enc, uint64_t n_samples, uint64_t d, uint64_t m,
  int nb, const int64_t *const *boff, const int32_t *const *btok,
  const float *const *bval,
  const int64_t *bstart, const float *rs, float *out_a)
{
  uint64_t tile = enc->tile;
  if (tile > n_samples) tile = n_samples;
  float *sims_f = enc->sims_buf;
  float *row_bufs = enc->row_bufs;
  const int64_t *restrict csc_off = enc->csc_offsets;
  const int16_t *restrict csc_rows_a = enc->csc_rows;
  const float *restrict csc_vals = enc->csc_values;
  for (uint64_t base = 0; base < n_samples; base += tile) {
    uint64_t blk = base + tile <= n_samples ? tile : n_samples - base;
    #pragma omp parallel
    {
      float *row_buf = row_bufs + (uint64_t) omp_get_thread_num() * m;
      #pragma omp for schedule(static)
      for (uint64_t i = 0; i < blk; i++) {
        float *restrict sims_row = sims_f + i * m;
        uint64_t si = base + i;
        for (int b = 0; b < nb; b++) {
          int64_t bs0 = bstart ? bstart[b] : 0;
          const int64_t *off = boff[b]; const int32_t *tok = btok[b];
          const float *val = bval[b];
          int64_t jlo = off[si], jhi = off[si + 1];
          for (int64_t j = jlo; j < jhi; j++) {
            int32_t gt = (int32_t) (bs0 + tok[j]);
            float v = val ? val[j] : 1.0f;
            int64_t clo = csc_off[gt], chi = csc_off[gt + 1];
            for (int64_t c = clo; c < chi; c++)
              row_buf[(uint64_t) csc_rows_a[c]] += v * csc_vals[c];
          }
        }
        double rsi = rs ? (double) rs[si] : 1.0;
        for (uint64_t j = 0; j < m; j++) {
          sims_row[j] = tk_spectral_kernel_apply(enc->kernel, (float) (row_buf[j] * rsi));
          row_buf[j] = 0.0f;
        }
      }
    }
    float *dst = out_a + base * d;
    memcpy(dst, sims_f, blk * m * sizeof(float));
    cblas_strsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
      (int) blk, (int) m, 1.0f, enc->chol, (int) m, dst, (int) d);
  }
}

static inline int tk_nystrom_encode_blocks (lua_State *L, tk_nystrom_encoder_t *enc) {
  if (!enc->chol)
    return luaL_error(L, "encode: chol released");
  uint64_t d = enc->d, m = enc->m;
  int nb = enc->n_blocks;
  tk_mtx_t *out_mtx = (lua_gettop(L) >= 3 && !lua_isnil(L, 3)) ? tk_mtx_peek(L, 3, "out") : NULL;
  luaL_checktype(L, 2, LUA_TTABLE);
  lua_getfield(L, 2, "blocks");
  int blk_idx;
  if (!lua_isnil(L, -1)) { blk_idx = lua_gettop(L); }
  else { lua_pop(L, 1); lua_pushvalue(L, 2); blk_idx = lua_gettop(L); }
  int got = (int) lua_objlen(L, blk_idx);
  if (got != nb)
    return luaL_error(L, "encode: block count mismatch (got %d, expected %d)", got, nb);
  const int64_t *boff[TK_MAX_MOD]; const int32_t *btok[TK_MAX_MOD];
  const float *bval[TK_MAX_MOD];
  const float *bcs[TK_MAX_MOD];
  uint64_t n_samples = 0;
  for (int b = 0; b < nb; b++) {
    lua_rawgeti(L, blk_idx, b + 1);
    int bt = lua_gettop(L);
    lua_getfield(L, bt, "x");
    tk_csr_t *Xb = tk_csr_peek(L, -1, "blocks[].x");
    if (Xb->ntag != TK_TAG_I32)
      return luaL_error(L, "encode: block neighbors must be i32");
    if (Xb->values && Xb->tag != TK_TAG_F32)
      return luaL_error(L, "encode: block values must be f32 (got an f64/other-typed csr; convert to_fvec)");
    if (b == 0) n_samples = tk_csr_rows(Xb);
    boff[b] = Xb->offsets->a;
    btok[b] = (const int32_t *) tk_csr_nbr_ptr(Xb);
    bval[b] = Xb->values ? ((tk_fvec_t *) Xb->values)->a : NULL;
    lua_pop(L, 1);
    lua_getfield(L, bt, "colscale");
    if (lua_isnil(L, -1)) bcs[b] = NULL;
    else {
      tk_fvec_t *cv = tk_fvec_peekopt(L, -1);
      if (!cv) return luaL_error(L, "encode: colscale must be an fvec");
      bcs[b] = cv->a;
    }
    lua_pop(L, 1);
    lua_pop(L, 1);
  }

  float *rowscale = (float *) malloc(n_samples * sizeof(float));
  if (!rowscale || tk_spectral_block_rowscale(nb, boff, btok, bval, bcs,
      enc->blk_s, n_samples, rowscale) != 0) {
    free(rowscale);
    return luaL_error(L, "encode: oom (rowscale)");
  }

  tk_mtx_t *M;
  if (out_mtx != NULL) { tk_mtx_reshape(L, out_mtx, n_samples, d); M = out_mtx; lua_pushvalue(L, 3); }
  else M = tk_mtx_push_new(L, TK_TAG_F32, n_samples, d);
  tk_fvec_t *out = (tk_fvec_t *) M->v;
  out->n = n_samples * d;
  int out_idx = lua_gettop(L);

  if (tk_nystrom_ensure_sims(enc, m) != 0) {
    free(rowscale);
    return luaL_error(L, "encode: out of memory");
  }
  if (tk_nystrom_ensure_row_bufs(enc, m) != 0) {
    free(rowscale);
    return luaL_error(L, "encode: out of memory (row_bufs)");
  }
  tk_nystrom_csr_tiles(enc, n_samples, d, m, nb, boff, btok, bval,
    enc->blk_start, rowscale, out->a);
  free(rowscale);
  lua_pushvalue(L, out_idx);
  return 1;
}

static inline int tk_nystrom_encode_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (enc->n_blocks > 0)
    return tk_nystrom_encode_blocks(L, enc);
  if (!enc->chol)
    return luaL_error(L, "encode: chol released");
  uint64_t d = enc->d;
  uint64_t m = enc->m;

  uint64_t n_samples;
  tk_ivec_t *in_offsets = NULL;
  const int32_t *in_tok_a = NULL; uint64_t in_tok_n = 0;
  tk_fvec_t *in_values_f = NULL; tk_dvec_t *in_values_d = NULL;
  tk_fvec_t *in_codes_fv = NULL; tk_dvec_t *in_codes_dv = NULL;
  int64_t in_d_input_scalar = 0;
  tk_fvec_t *out;
  int out_fv_idx;
  tk_csr_t *Xcsr = tk_csr_peekopt(L, 2);
  tk_mtx_t *Xmtx = Xcsr ? NULL : tk_mtx_peek(L, 2, "x");
  tk_mtx_t *out_mtx = NULL;
  if (lua_gettop(L) >= 3 && !lua_isnil(L, 3))
    out_mtx = tk_mtx_peek(L, 3, "out");
  if (Xcsr != NULL) {
    n_samples = tk_csr_rows(Xcsr);
    in_offsets = Xcsr->offsets;
    tk_eph_get(L, 2, Xcsr->neighbors);
    in_tok_a = tk_peek_tokens(L, -1, &in_tok_n);
    if (Xcsr->tag == TK_TAG_F32) in_values_f = (tk_fvec_t *) Xcsr->values;
    else if (Xcsr->tag == TK_TAG_F64) in_values_d = (tk_dvec_t *) Xcsr->values;
  } else {
    n_samples = Xmtx->n_rows;
    if (Xmtx->tag == TK_TAG_F32) in_codes_fv = (tk_fvec_t *) Xmtx->v;
    else if (Xmtx->tag == TK_TAG_F64) in_codes_dv = (tk_dvec_t *) Xmtx->v;
    in_d_input_scalar = (int64_t) Xmtx->n_cols;
  }
  tk_mtx_t *M;
  if (out_mtx != NULL) {
    tk_mtx_reshape(L, out_mtx, n_samples, d);
    M = out_mtx;
    lua_pushvalue(L, 3);
  } else {
    M = tk_mtx_push_new(L, TK_TAG_F32, n_samples, d);
  }
  out = (tk_fvec_t *) M->v;
  out->n = n_samples * d;
  out_fv_idx = lua_gettop(L);

  if (tk_nystrom_ensure_sims(enc, m) != 0)
    return luaL_error(L, "encode: out of memory");

  if (enc->mod_type == TK_MOD_CSR) {
    if (!in_offsets) return luaL_error(L, "encode: CSR modality but no offsets");
    float *csr_sval = NULL;
    const float *csr_sv = NULL;
    uint64_t nnz = in_tok_n;
    if (!in_values_f) {
      csr_sval = (float *)malloc(nnz * sizeof(float));
      if (!csr_sval) return luaL_error(L, "encode: out of memory (values)");
      if (in_values_d)
        for (uint64_t i = 0; i < nnz; i++) csr_sval[i] = (float)in_values_d->a[i];
      else
        for (uint64_t i = 0; i < nnz; i++) csr_sval[i] = 1.0f;
      csr_sv = csr_sval;
    } else {
      csr_sv = in_values_f->a;
    }
    if (tk_nystrom_ensure_row_bufs(enc, m) != 0) {
      free(csr_sval);
      return luaL_error(L, "encode: out of memory (row_bufs)");
    }
    const int64_t *boff1[1] = { in_offsets->a };
    const int32_t *btok1[1] = { in_tok_a };
    const float *bval1[1] = { csr_sv };
    tk_nystrom_csr_tiles(enc, n_samples, d, m, 1, boff1, btok1, bval1, NULL, NULL, out->a);
    free(csr_sval);
  } else if (enc->mod_type == TK_MOD_DENSE) {
    int64_t di = enc->d_input;
    uint64_t ddi = in_d_input_scalar > 0 ? (uint64_t)in_d_input_scalar : (uint64_t)di;
    if (ddi != (uint64_t)di)
      return luaL_error(L, "encode: input has %d cols, encoder expects %d", (int)ddi, (int)di);
    if (!enc->dense_tile_buf) {
      enc->dense_tile_buf = (float *)malloc(enc->tile * (uint64_t)di * sizeof(float));
      if (!enc->dense_tile_buf) return luaL_error(L, "encode: out of memory (dense tile)");
    }
    uint64_t tile = enc->tile;
    if (tile > n_samples) tile = n_samples;
    float *sims_f = enc->sims_buf;
    float *src_f = enc->dense_tile_buf;
    for (uint64_t base = 0; base < n_samples; base += tile) {
      uint64_t blk = base + tile <= n_samples ? tile : n_samples - base;
      uint64_t cnt = blk * (uint64_t)di;
      if (in_codes_fv) {
        memcpy(src_f, in_codes_fv->a + base * (uint64_t)di, cnt * sizeof(float));
      } else {
        for (uint64_t i = 0; i < cnt; i++)
          src_f[i] = (float)in_codes_dv->a[base * (uint64_t)di + i];
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        (int)blk, (int)m, (int)di, 1.0f,
        src_f, (int)di, enc->dense_vecs, (int)di,
        0.0f, sims_f, (int)m);
      if (enc->dense_cs2) {
        const float *cs2 = enc->dense_cs2;
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < blk; i++) {
          const float *row = src_f + i * (uint64_t)di;
          double acc = 0.0;
          for (int64_t k = 0; k < di; k++)
            acc += (double)cs2[k] * (double)row[k] * (double)row[k];
          float rsq = acc > 0.0 ? (float)(1.0 / sqrt(acc)) : 0.0f;
          for (uint64_t j = 0; j < m; j++)
            sims_f[i * m + j] = tk_spectral_kernel_apply(enc->kernel, sims_f[i * m + j] * rsq);
        }
      } else {
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < blk; i++)
          for (uint64_t j = 0; j < m; j++)
            sims_f[i * m + j] = tk_spectral_kernel_apply(enc->kernel, sims_f[i * m + j]);
      }
      float *dst = out->a + base * d;
      memcpy(dst, sims_f, blk * m * sizeof(float));
      cblas_strsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
        (int)blk, (int)m, 1.0f, enc->chol, (int)m, dst, (int)d);
    }
  }

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
  if (!enc->chol)
    return luaL_error(L, "cannot persist: chol released");
  uint8_t chol_external = 0;
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "chol");
  tk_fvec_t *cfv = tk_fvec_peekopt(L, -1);
  if (cfv && cfv->lua_managed == 2) chol_external = 1;
  lua_pop(L, 2);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKny", 1, 4, fh);
  uint8_t version = 29;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->kernel.family, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->kernel.nu, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->kernel.order, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->kernel.depth, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->kernel.tangent, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->kernel.gamma, sizeof(float), 1, fh);
  tk_lua_fwrite(L, &enc->mod_type, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->m, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->d, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->trace_ratio, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &chol_external, sizeof(uint8_t), 1, fh);
  if (chol_external) {
#if !defined(__EMSCRIPTEN__)
    msync(enc->chol, (uint64_t)enc->m * enc->m * sizeof(float), MS_SYNC);
#endif
  } else {
    tk_lua_fwrite(L, enc->chol, sizeof(float), (uint64_t)enc->m * enc->m, fh);
  }
  if (enc->mod_type == TK_MOD_CSR) {
    tk_lua_fwrite(L, &enc->csr_n_tokens, sizeof(uint64_t), 1, fh);
    uint64_t total_csr = (uint64_t)enc->csr_offsets[enc->m];
    tk_lua_fwrite(L, enc->csr_offsets, sizeof(int64_t), enc->m + 1, fh);
    tk_lua_fwrite(L, enc->csr_tokens, sizeof(int32_t), total_csr, fh);
    tk_lua_fwrite(L, enc->csr_values, sizeof(float), total_csr, fh);
    tk_lua_fwrite(L, &enc->n_blocks, sizeof(int), 1, fh);
    if (enc->n_blocks > 0) {
      tk_lua_fwrite(L, enc->blk_start, sizeof(int64_t), (size_t) enc->n_blocks, fh);
      tk_lua_fwrite(L, enc->blk_s, sizeof(float), (size_t) enc->n_blocks, fh);
    }
  } else if (enc->mod_type == TK_MOD_DENSE) {
    tk_lua_fwrite(L, &enc->d_input, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, enc->dense_vecs, sizeof(float), enc->m * (uint64_t)enc->d_input, fh);
    uint8_t has_cs = enc->dense_cs2 ? 1 : 0;
    tk_lua_fwrite(L, &has_cs, sizeof(uint8_t), 1, fh);
    if (has_cs)
      tk_lua_fwrite(L, enc->dense_cs2, sizeof(float), (uint64_t)enc->d_input, fh);
  }
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "landmark_ids");
  tk_ivec_t *lm_ids = tk_ivec_peek(L, -1, "landmark_ids");
  tk_ivec_persist(L, lm_ids, fh);
  lua_pop(L, 2);
  tk_lua_fclose(L, fh);
  return 0;
}

static luaL_Reg tk_nystrom_encoder_mt_fns[] = {
  { "encode", tk_nystrom_encode_lua },
  { "dims", tk_nystrom_dims_lua },
  { "persist", tk_nystrom_encoder_persist_lua },
  { NULL, NULL }
};

static inline int tm_encode (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  double tp_enc0 = omp_get_wtime(), tp_lm = 0.0, tp_pj = 0.0, tp_gr = 0.0;

  lua_getfield(L, 1, "x");
  if (!lua_isnil(L, -1)) {
    int xi = lua_gettop(L);
    tk_csr_t *Xc = tk_csr_peekopt(L, xi);
    if (Xc != NULL) {
      tk_eph_get(L, xi, Xc->offsets); lua_setfield(L, 1, "offsets");
      tk_eph_get(L, xi, Xc->neighbors); lua_setfield(L, 1, "tokens");
      if (Xc->values != NULL) { tk_eph_get(L, xi, Xc->values); lua_setfield(L, 1, "values"); }
      lua_getfield(L, 1, "n_tokens");
      int nt_given = !lua_isnil(L, -1);
      lua_pop(L, 1);
      if (!nt_given) { lua_pushinteger(L, (lua_Integer) Xc->n_cols); lua_setfield(L, 1, "n_tokens"); }
      lua_pushinteger(L, (lua_Integer) tk_csr_rows(Xc)); lua_setfield(L, 1, "n_samples");
    } else {
      tk_mtx_t *Xm = tk_mtx_peek(L, xi, "x");
      tk_eph_get(L, xi, Xm->v); lua_setfield(L, 1, "codes");
      lua_pushinteger(L, (lua_Integer) Xm->n_cols); lua_setfield(L, 1, "d_input");
      lua_pushinteger(L, (lua_Integer) Xm->n_rows); lua_setfield(L, 1, "n_samples");
    }
  }
  lua_pop(L, 1);
  lua_getfield(L, 1, "y");
  if (!lua_isnil(L, -1)) {
    int yi = lua_gettop(L);
    tk_csr_t *Yc = tk_csr_peek(L, yi, "y");
    tk_eph_get(L, yi, Yc->offsets); lua_setfield(L, 1, "label_offsets");
    tk_eph_get(L, yi, Yc->neighbors); lua_setfield(L, 1, "label_neighbors");
    lua_pushinteger(L, (lua_Integer) Yc->n_cols); lua_setfield(L, 1, "n_labels");
  }
  lua_pop(L, 1);

  lua_getfield(L, 1, "blocks");
  if (!lua_isnil(L, -1)) {
    lua_rawgeti(L, -1, 1);
    lua_getfield(L, -1, "x");
    lua_pushinteger(L, (lua_Integer) tk_csr_rows(tk_csr_peek(L, -1, "blocks[1].x")));
    lua_setfield(L, 1, "n_samples");
    lua_pop(L, 2);
  }
  lua_pop(L, 1);

  uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "encode", "n_samples");

  tk_spectral_modality_t mod;
  memset(&mod, 0, sizeof(mod));
  float *csr_values_owned = NULL;
  float *blk_rowscale = NULL;
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
    if (!val_fv && !lua_isnil(L, -1))
      return luaL_error(L, "encode: values must be f32 (got an f64/other-typed vec; convert to_fvec)");
    lua_pop(L, 1);
    mod.csr_offsets = off_iv->a;
    mod.csr_tokens = tok_a;
    mod.csr_values = val_fv ? val_fv->a : NULL;
    mod.type = TK_MOD_CSR;
    lua_getfield(L, 1, "rowscale");
    if (!lua_isnil(L, -1)) { tk_fvec_t *rs = tk_fvec_peekopt(L, -1); if (rs) mod.rowscale = rs->a; }
    lua_pop(L, 1);
  } else {
    lua_pop(L, 1);
  }

  int n_blocks = 0;
  lua_getfield(L, 1, "blocks");
  if (!lua_isnil(L, -1)) {
    n_blocks = (int) lua_objlen(L, -1);
    if (n_blocks > TK_MAX_MOD)
      return luaL_error(L, "encode: too many blocks (max %d)", TK_MAX_MOD);
    int bidx = lua_gettop(L);
    int64_t gtok = 0;
    for (int b = 0; b < n_blocks; b++) {
      lua_rawgeti(L, bidx, b + 1);
      int bt = lua_gettop(L);
      lua_getfield(L, bt, "x");
      tk_csr_t *Xb = tk_csr_peek(L, -1, "blocks[].x");
      if (Xb->ntag != TK_TAG_I32)
        return luaL_error(L, "encode: block neighbors must be i32");
      if (Xb->values && Xb->tag != TK_TAG_F32)
        return luaL_error(L, "encode: block values must be f32 (got an f64/other-typed csr; convert to_fvec)");
      mod.blk_off[b] = Xb->offsets->a;
      mod.blk_tok[b] = (const int32_t *) tk_csr_nbr_ptr(Xb);
      mod.blk_val[b] = Xb->values ? ((tk_fvec_t *) Xb->values)->a : NULL;
      lua_pop(L, 1);
      lua_getfield(L, bt, "colscale");
      if (lua_isnil(L, -1)) mod.blk_cs[b] = NULL;
      else {
        tk_fvec_t *cv = tk_fvec_peekopt(L, -1);
        if (!cv) return luaL_error(L, "encode: colscale must be an fvec");
        mod.blk_cs[b] = cv->a;
      }
      lua_pop(L, 1);
      mod.blk_s[b] = (float) tk_lua_foptnumber(L, bt, "encode", "scale", 1.0);
      mod.blk_start[b] = gtok;
      gtok += (int64_t) tk_lua_fcheckunsigned(L, bt, "encode", "n_tokens");
      lua_pop(L, 1);
    }
    mod.n_blocks = n_blocks;
    mod.csr_n_tokens = (uint64_t) gtok;
    mod.type = TK_MOD_CSR;
    blk_rowscale = (float *) malloc(n_samples * sizeof(float));
    if (!blk_rowscale || tk_spectral_block_rowscale(n_blocks, mod.blk_off, mod.blk_tok,
        mod.blk_val, mod.blk_cs, mod.blk_s, n_samples, blk_rowscale) != 0) {
      free(blk_rowscale);
      return luaL_error(L, "encode: oom (rowscale)");
    }
    mod.rowscale = blk_rowscale;
  }
  lua_pop(L, 1);

  lua_getfield(L, 1, "codes");
  int has_dense = !lua_isnil(L, -1);
  float *dense_owned = NULL;
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
    if (codes_fv) {
      mod.dense = codes_fv->a;
    } else {
      dense_owned = (float *)malloc(cn * sizeof(float));
      if (!dense_owned)
        return luaL_error(L, "encode: out of memory (codes)");
      for (uint64_t i = 0; i < cn; i++)
        dense_owned[i] = (float)codes_dv->a[i];
      mod.dense = dense_owned;
    }
  } else {
    lua_pop(L, 1);
  }

  float *dense_rowscale = NULL;
  float *dense_cs2 = NULL;
  if (has_dense) {
    lua_getfield(L, 1, "colscale");
    if (!lua_isnil(L, -1)) {
      tk_fvec_t *cv = tk_fvec_peekopt(L, -1);
      if (!cv || cv->n < (uint64_t) mod.d_input) {
        free(dense_owned);
        return luaL_error(L, "encode: colscale must be an fvec of length d_input");
      }
      int64_t di = mod.d_input;
      dense_cs2 = (float *)malloc((uint64_t)di * sizeof(float));
      dense_rowscale = (float *)malloc(n_samples * sizeof(float));
      if (!dense_cs2 || !dense_rowscale) {
        free(dense_cs2); free(dense_rowscale); free(dense_owned);
        return luaL_error(L, "encode: out of memory (colscale)");
      }
      for (int64_t k = 0; k < di; k++)
        dense_cs2[k] = cv->a[k] * cv->a[k];
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++) {
        const float *row = mod.dense + i * (uint64_t)di;
        double acc = 0.0;
        for (int64_t k = 0; k < di; k++)
          acc += (double)dense_cs2[k] * (double)row[k] * (double)row[k];
        dense_rowscale[i] = acc > 0.0 ? (float)(1.0 / sqrt(acc)) : 0.0f;
      }
      mod.dense_cs2 = dense_cs2;
      mod.rowscale = dense_rowscale;
    }
    lua_pop(L, 1);
  }

  int n_provided = has_csr + has_dense + (n_blocks > 0 ? 1 : 0);
  if (n_provided == 0)
    return luaL_error(L, "encode: no modality provided");
  if (n_provided > 1)
    return luaL_error(L, "encode: provide exactly one modality (csr, blocks, or dense)");

  lua_getfield(L, 1, "kernel");
  const char *kernel_str = lua_isnil(L, -1) ? "cosine" : lua_tostring(L, -1);
  lua_pop(L, 1);
  float gamma = (float)tk_lua_foptnumber(L, 1, "encode", "gamma", 1.0);
  tk_spectral_kernel_t kernel = { .family = TK_SPECTRAL_COSINE, .nu = 3,
    .order = 1, .depth = 1, .tangent = 0, .gamma = gamma };
  if (strcmp(kernel_str, "cosine") == 0) {
    kernel.family = TK_SPECTRAL_COSINE;
  } else if (strcmp(kernel_str, "matern") == 0) {
    kernel.family = TK_SPECTRAL_MATERN;
    kernel.nu = (uint8_t)tk_lua_foptunsigned(L, 1, "encode", "nu", 3);
  } else if (strcmp(kernel_str, "arccos") == 0) {
    kernel.family = TK_SPECTRAL_ARCCOS;
    kernel.order = (uint8_t)tk_lua_foptunsigned(L, 1, "encode", "order", 1);
    kernel.depth = (uint8_t)tk_lua_foptunsigned(L, 1, "encode", "depth", 1);
    kernel.tangent = (uint8_t)tk_lua_foptunsigned(L, 1, "encode", "tangent", 0);
  } else {
    return luaL_error(L, "encode: unknown kernel '%s'", kernel_str);
  }
  if (kernel.depth < 1) kernel.depth = 1;
  if (kernel.order > 6) kernel.order = 6;

  if (has_csr && !mod.csr_values) {
    uint64_t nnz = (uint64_t)(mod.csr_offsets[n_samples] - mod.csr_offsets[0]);
    csr_values_owned = (float *)malloc(nnz * sizeof(float));
    for (uint64_t i = 0; i < nnz; i++)
      csr_values_owned[i] = 1.0f;
    mod.csr_values = csr_values_owned;
  }

  uint64_t n_lm_req = tk_lua_foptunsigned(L, 1, "encode", "n_landmarks", 0);
  double trace_tol = tk_lua_foptnumber(L, 1, "encode", "trace_tol", 0.001);
  uint64_t chol_block = tk_lua_foptunsigned(L, 1, "encode", "chol_block", 64);

  lua_getfield(L, 1, "chol_buf");
  tk_fvec_t *chol_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "chol_buf");
  lua_pop(L, 1);

  lua_getfield(L, 1, "xtx_buf");
  tk_fvec_t *xtx_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "xtx_buf");
  lua_pop(L, 1);

  lua_getfield(L, 1, "xty_buf");
  tk_fvec_t *xty_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "xty_buf");
  lua_pop(L, 1);

  lua_getfield(L, 1, "w_buf");
  tk_fvec_t *w_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "w_buf");
  lua_pop(L, 1);

  lua_getfield(L, 1, "proj_buf");
  tk_fvec_t *proj_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "proj_buf");
  lua_pop(L, 1);

  lua_getfield(L, 1, "sims_buf");
  tk_fvec_t *sims_buf_arg = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "sims_buf");
  lua_pop(L, 1);

  lua_getfield(L, 1, "row_buf");
  tk_fvec_t *row_buf_arg = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "row_buf");
  lua_pop(L, 1);

  tk_ivec_t *lm_ids = NULL;
  tk_fvec_t *lm_chol = NULL;
  float *lm_chol_ext = NULL;
  float *full_chol = NULL;
  uint64_t nc, m;
  double trace_ratio;
  lua_getfield(L, 1, "landmarks");
  tk_ivec_t *lm_forced = lua_isnil(L, -1) ? NULL : tk_ivec_peek(L, -1, "landmarks");
  lua_pop(L, 1);
  uint64_t lm_backstop = 0;
  double tp_t = omp_get_wtime();
  tk_spectral_sample_landmarks(L,
    &mod, n_samples, kernel,
    n_lm_req, trace_tol, chol_block,
    chol_buf ? chol_buf->a : NULL,
    proj_buf ? proj_buf->a : NULL, proj_buf ? proj_buf->m : 0,
    &lm_ids, &lm_chol, &lm_chol_ext, &full_chol, &nc, &m, &trace_ratio,
    lm_forced ? lm_forced->a : NULL, lm_forced ? lm_forced->n : 0, &lm_backstop);
  tp_lm = omp_get_wtime() - tp_t;
  lua_pushinteger(L, (lua_Integer) lm_backstop);
  lua_setfield(L, 1, "lm_backstop");
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
  int64_t tile_labels = 0;
  lua_getfield(L, 1, "tile_labels");
  if (lua_isnumber(L, -1)) tile_labels = (int64_t)lua_tointeger(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "solve_lambda");
  int do_solve = lua_isnumber(L, -1);
  double solve_lambda = do_solve ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);

  if (m == 0) {
    if (lm_chol) tk_fvec_destroy(lm_chol);
    if (!chol_buf) free(full_chol);
    free(csr_values_owned); free(dense_owned);
    free(dense_rowscale); free(dense_cs2);
    lua_pushnil(L);
    lua_pushnil(L);
    return 2;
  }
  if (m > 32768) {
    if (lm_chol) tk_fvec_destroy(lm_chol);
    if (!chol_buf) free(full_chol);
    free(csr_values_owned); free(dense_owned);
    free(dense_rowscale); free(dense_cs2);
    return luaL_error(L, "encode: n_landmarks %d exceeds 32768 (int16 csc_rows ceiling)", (int)m);
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

  double tp_t2 = omp_get_wtime();
  float *chol_store = NULL;
  int chol_external = 0;
  if (lm_chol_ext) {
    // sampler factored K_SS straight into proj_buf; nothing to copy
    chol_store = lm_chol_ext;
    chol_external = 1;
  } else if (proj_buf && proj_buf->m >= (uint64_t)m * m) {
    memcpy(proj_buf->a, lm_chol->a, (uint64_t)m * m * sizeof(float));
    chol_store = proj_buf->a;
    chol_external = 1;
    tk_fvec_destroy(lm_chol); ctx->lm_chol = NULL; lm_chol = NULL;
  }
  tp_pj = omp_get_wtime() - tp_t2;

  double tp_t3 = omp_get_wtime();
  int gram_result_idx = 0;
  int build_prepared = has_gram_labels || has_gram_targets;
  if (build_prepared) {
    uint64_t unl = (uint64_t)gram_nl;
    uint64_t dd = (uint64_t)d * d;
    uint64_t dnl = (uint64_t)d * unl;
    bool XtX_external = xtx_buf && xtx_buf->m >= dd;
    bool xty_external = xty_buf && xty_buf->m >= dnl;
    float *XtX = XtX_external ? xtx_buf->a : (float *)malloc(dd * sizeof(float));
    float *xty = xty_external ? xty_buf->a : (float *)malloc(dnl * sizeof(float));
    float *cm_f = (float *)malloc(d * sizeof(float));
    double *y_mean = (double *)calloc(unl, sizeof(double));
    float *ym_f = (float *)malloc(unl * sizeof(float));
    if (!XtX || !xty || !cm_f || !y_mean || !ym_f) {
      if (!XtX_external) free(XtX);
      if (!xty_external) free(xty);
      free(cm_f); free(y_mean); free(ym_f);
      return luaL_error(L, "gram prepare: out of memory");
    }
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < (int64_t)d; j++) {
      double s = 0.0;
      const float *src = full_chol + (uint64_t)j * nc;
      for (uint64_t i = 0; i < nc; i++) s += (double)src[i];
      cm_f[j] = (float)(s / (double)nc);
    }
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
      (int)d, (int)nc, 1.0f, full_chol, (int)nc, 0.0f, XtX, (int)d);
    if (has_gram_targets) {
      float *Yf = (float *)malloc((uint64_t)nc * unl * sizeof(float));
      if (!Yf) {
        if (!XtX_external) free(XtX);
        if (!xty_external) free(xty);
        free(cm_f); free(y_mean); free(ym_f);
        return luaL_error(L, "gram prepare: out of memory (targets)");
      }
      for (uint64_t i = 0; i < nc * unl; i++) Yf[i] = (float)gram_targets_dv->a[i];
      for (int64_t l = 0; l < gram_nl; l++) {
        double s = 0.0;
        for (uint64_t i = 0; i < nc; i++) s += gram_targets_dv->a[i * unl + (uint64_t)l];
        y_mean[l] = s / (double)nc;
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)d, (int)gram_nl, (int)nc, 1.0f, full_chol, (int)nc,
        Yf, (int)gram_nl, 0.0f, xty, (int)gram_nl);
      free(Yf);
    } else {
      const int64_t *lbl_off = gram_lbl_off->a;
      const int64_t *lbl_nbr = gram_lbl_nbr->a;
      for (uint64_t i = 0; i < nc; i++)
        for (int64_t j = lbl_off[i]; j < lbl_off[i + 1]; j++)
          y_mean[lbl_nbr[j]] += 1.0;
      for (int64_t l = 0; l < gram_nl; l++) y_mean[l] /= (double)nc;
      memset(xty, 0, dnl * sizeof(float));
      #pragma omp parallel for schedule(static)
      for (int64_t kk = 0; kk < (int64_t)d; kk++) {
        const float *src = full_chol + (uint64_t)kk * nc;
        float *dst = xty + (uint64_t)kk * unl;
        for (uint64_t i = 0; i < nc; i++) {
          float v = src[i];
          for (int64_t j = lbl_off[i]; j < lbl_off[i + 1]; j++)
            dst[lbl_nbr[j]] += v;
        }
      }
    }
    cblas_ssyr(CblasColMajor, CblasUpper, (int)d, -(float)nc, cm_f, 1, XtX, (int)d);
    for (int64_t l = 0; l < gram_nl; l++) ym_f[l] = (float)y_mean[l];
    cblas_sger(CblasRowMajor, (int)d, (int)gram_nl, -(float)nc, cm_f, 1, ym_f, 1, xty, (int)gram_nl);
    free(ym_f);
    double mean_eig = 0.0;
    for (uint64_t i = 0; i < d; i++) mean_eig += (double)XtX[i * d + i];
    mean_eig /= (double)d;
    if (!ctx->full_chol_external) free(full_chol);
    ctx->full_chol = NULL;
    tk_gram_make_prepared(L, XtX, XtX_external, xty, xty_external, cm_f, y_mean, mean_eig,
      (int64_t)nc, (int64_t)d, gram_nl, tile_labels);
    gram_result_idx = lua_gettop(L);
    lua_getfenv(L, gram_result_idx);
    lua_getfield(L, 1, "xtx_buf");
    lua_setfield(L, -2, "xtx_buf");
    lua_getfield(L, 1, "xty_buf");
    lua_setfield(L, -2, "xty_buf");
    lua_pop(L, 1);
    if (do_solve) {
      tk_gram_t *g = tk_gram_peek(L, gram_result_idx);
      tk_gram_solve_impl(L, g, solve_lambda, w_buf);
      if (g->W_baked_external && w_buf) {
        lua_getfenv(L, gram_result_idx);
        lua_getfield(L, 1, "w_buf");
        lua_setfield(L, -2, "w_buf");
        lua_pop(L, 1);
      }
      tk_gram_release_prepared(g);
    }
  }
  if (!build_prepared) {
    if (!ctx->full_chol_external) free(full_chol);
    ctx->full_chol = NULL;
  }
  tp_gr = omp_get_wtime() - tp_t3;

  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  tk_nystrom_encoder_zero(enc);
  if (chol_external) {
    enc->chol = chol_store;
  } else {
    enc->chol_f = lm_chol;
    enc->chol = lm_chol->a;
    ctx->lm_chol = NULL;
  }
  enc->m = m;
  enc->d = d;
  enc->trace_ratio = trace_ratio;
  enc->kernel = kernel;
  enc->mod_type = mod.type;
  if (sims_buf_arg && sims_buf_arg->m >= m) {
    enc->sims_buf = sims_buf_arg->a; enc->sims_buf_external = 1; enc->sims_buf_cap = sims_buf_arg->m;
  }
  uint64_t nt_max = (uint64_t) omp_get_max_threads();
  if (row_buf_arg && row_buf_arg->m >= nt_max * m) {
    enc->row_bufs = row_buf_arg->a; enc->row_bufs_external = 1;
    enc->row_bufs_cap = row_buf_arg->m; enc->n_threads = (int) nt_max;
  }

  if (mod.n_blocks > 0) {
    enc->n_blocks = mod.n_blocks;
    uint64_t csr_nt = mod.csr_n_tokens;
    enc->csr_n_tokens = csr_nt;
    for (int b = 0; b < mod.n_blocks; b++) {
      enc->blk_start[b] = mod.blk_start[b];
      enc->blk_s[b] = mod.blk_s[b];
    }
    uint64_t lm_total = 0;
    for (uint64_t j = 0; j < m; j++) {
      uint64_t si = (uint64_t) lm_ids->a[j];
      for (int b = 0; b < mod.n_blocks; b++) {
        const int64_t *off = mod.blk_off[b];
        lm_total += (uint64_t) (off[si + 1] - off[si]);
      }
    }
    enc->csr_offsets = (int64_t *) malloc((m + 1) * sizeof(int64_t));
    enc->csr_tokens = (int32_t *) malloc(lm_total * sizeof(int32_t));
    enc->csr_values = (float *) malloc(lm_total * sizeof(float));
    if (!enc->csr_offsets || !enc->csr_tokens || !enc->csr_values)
      return luaL_error(L, "encode: out of memory (landmarks)");
    enc->csr_offsets[0] = 0;
    for (uint64_t j = 0; j < m; j++) {
      uint64_t si = (uint64_t) lm_ids->a[j];
      float rsj = mod.rowscale ? mod.rowscale[si] : 1.0f;
      int64_t dst = enc->csr_offsets[j];
      for (int b = 0; b < mod.n_blocks; b++) {
        const int64_t *off = mod.blk_off[b]; const int32_t *tok = mod.blk_tok[b];
        const float *val = mod.blk_val[b];
        const float *cs = mod.blk_cs[b];
        double s2 = (double) mod.blk_s[b] * (double) mod.blk_s[b];
        int64_t bstart = mod.blk_start[b];
        int64_t lo = off[si], hi = off[si + 1];
        for (int64_t k = lo; k < hi; k++) {
          double cs2 = cs ? (double) cs[tok[k]] * (double) cs[tok[k]] : 1.0;
          enc->csr_tokens[dst] = (int32_t) (bstart + tok[k]);
          enc->csr_values[dst] = (float) ((val ? (double) val[k] : 1.0) * cs2 * s2 * (double) rsj);
          dst++;
        }
      }
      enc->csr_offsets[j + 1] = dst;
    }
    if (tk_nystrom_build_csc(enc) != 0)
      return luaL_error(L, "encode: out of memory (csc)");
  } else if (has_csr) {
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
    if (!enc->csr_offsets || !enc->csr_tokens || !enc->csr_values) {
      free(csr_values_owned);
      return luaL_error(L, "encode: out of memory (landmarks)");
    }
    enc->csr_offsets[0] = 0;
    for (uint64_t j = 0; j < m; j++) {
      uint64_t si = (uint64_t)lm_ids->a[j];
      int64_t lo = mod.csr_offsets[si];
      int64_t cnt = mod.csr_offsets[si + 1] - lo;
      float rsj = mod.rowscale ? mod.rowscale[si] : 1.0f;
      int64_t dst = enc->csr_offsets[j];
      memcpy(enc->csr_tokens + dst, mod.csr_tokens + lo, (uint64_t)cnt * sizeof(int32_t));
      for (int64_t k = 0; k < cnt; k++)
        enc->csr_values[dst + k] = (mod.csr_values ? mod.csr_values[lo + k] : 1.0f) * rsj;
      enc->csr_offsets[j + 1] = dst + cnt;
    }
    if (tk_nystrom_build_csc(enc) != 0) {
      free(csr_values_owned);
      return luaL_error(L, "encode: out of memory (csc)");
    }
    free(csr_values_owned);
  }

  if (has_dense) {
    int64_t di = mod.d_input;
    enc->d_input = di;
    float *lmv = (float *)malloc(m * (uint64_t)di * sizeof(float));
    if (!lmv) {
      free(dense_owned); free(dense_rowscale); free(dense_cs2);
      return luaL_error(L, "encode: out of memory (landmarks)");
    }
    for (uint64_t j = 0; j < m; j++) {
      uint64_t lm = (uint64_t)lm_ids->a[j];
      memcpy(lmv + j * (uint64_t)di,
             mod.dense + lm * (uint64_t)di,
             (uint64_t)di * sizeof(float));
      if (mod.dense_cs2) {
        float rs_l = mod.rowscale[lm];
        for (int64_t k = 0; k < di; k++)
          lmv[j * (uint64_t)di + (uint64_t)k] *= mod.dense_cs2[k] * rs_l;
      }
    }
    enc->dense_vecs = lmv;
    if (dense_cs2) {
      enc->dense_cs2 = dense_cs2;
      dense_cs2 = NULL;
    }
    free(dense_owned);
  }

  free(blk_rowscale);
  free(dense_rowscale);

  {
    double tp_setup = (omp_get_wtime() - tp_enc0) - tp_lm - tp_pj - tp_gr;
    lua_newtable(L);
    lua_pushnumber(L, tp_setup > 0.0 ? tp_setup : 0.0); lua_setfield(L, -2, "setup");
    lua_pushnumber(L, tp_lm); lua_setfield(L, -2, "landmarks");
    lua_pushnumber(L, tp_pj); lua_setfield(L, -2, "project");
    lua_pushnumber(L, tp_gr); lua_setfield(L, -2, "gram");
    lua_setfield(L, 1, "enc_phases");
  }

  lua_newtable(L);
  lua_pushvalue(L, lm_ids_idx);
  lua_setfield(L, -2, "landmark_ids");
  if (chol_external) {
    lua_getfield(L, 1, "proj_buf");
    lua_setfield(L, -2, "chol");
  }
  lua_setfenv(L, enc_idx);

  lua_pushnil(L);
  lua_pushvalue(L, enc_idx);
  if (gram_result_idx > 0) {
    lua_pushvalue(L, gram_result_idx);
    return 3;
  }
  return 2;
}

static inline int tk_nystrom_build_csc (tk_nystrom_encoder_t *enc) {
  uint64_t csr_nt = enc->csr_n_tokens;
  uint64_t lm_total = (uint64_t)enc->csr_offsets[enc->m];
  enc->csc_offsets = (int64_t *)calloc(csr_nt + 1, sizeof(int64_t));
  enc->csc_rows = (int16_t *)malloc(lm_total * sizeof(int16_t));
  enc->csc_values = (float *)malloc(lm_total * sizeof(float));
  int64_t *csc_pos = (int64_t *)malloc((csr_nt + 1) * sizeof(int64_t));
  if (!enc->csc_offsets || !enc->csc_rows || !enc->csc_values || !csc_pos) {
    free(csc_pos);
    return -1;
  }
  for (uint64_t i = 0; i < lm_total; i++)
    enc->csc_offsets[enc->csr_tokens[i] + 1]++;
  for (uint64_t t = 0; t < csr_nt; t++)
    enc->csc_offsets[t + 1] += enc->csc_offsets[t];
  memcpy(csc_pos, enc->csc_offsets, (csr_nt + 1) * sizeof(int64_t));
  for (uint64_t j = 0; j < enc->m; j++) {
    for (int64_t a = enc->csr_offsets[j]; a < enc->csr_offsets[j + 1]; a++) {
      int32_t tok = enc->csr_tokens[a];
      int64_t p = csc_pos[tok]++;
      enc->csc_rows[p] = (int16_t)j;
      enc->csc_values[p] = enc->csr_values[a];
    }
  }
  free(csc_pos);
  return 0;
}

static inline int tk_nystrom_encoder_load_lua (lua_State *L) {
  lua_settop(L, 2);
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
  if (version != 29) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported nystrom encoder version %d (old layout; re-persist required)", (int)version);
  }

  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  tk_nystrom_encoder_zero(enc);

  tk_lua_fread(L, &enc->kernel.family, sizeof(uint8_t), 1, fh);
  tk_lua_fread(L, &enc->kernel.nu, sizeof(uint8_t), 1, fh);
  tk_lua_fread(L, &enc->kernel.order, sizeof(uint8_t), 1, fh);
  tk_lua_fread(L, &enc->kernel.depth, sizeof(uint8_t), 1, fh);
  tk_lua_fread(L, &enc->kernel.tangent, sizeof(uint8_t), 1, fh);
  tk_lua_fread(L, &enc->kernel.gamma, sizeof(float), 1, fh);
  tk_lua_fread(L, &enc->mod_type, sizeof(uint8_t), 1, fh);
  tk_lua_fread(L, &enc->m, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &enc->d, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &enc->trace_ratio, sizeof(double), 1, fh);
  uint8_t chol_external;
  tk_lua_fread(L, &chol_external, sizeof(uint8_t), 1, fh);
  uint64_t mm = enc->m * enc->m;
  int chol_arg_idx = 0;
  bool have_arg = !lua_isnil(L, 2);
  if (chol_external) {
    if (!have_arg) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "load: external chol requires fvec arg 2");
    }
    tk_fvec_t *cf = tk_fvec_peek(L, 2, "chol");
    if (cf->n < mm) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "load: chol buffer too small");
    }
    enc->chol = cf->a;
    chol_arg_idx = 2;
  } else {
    if (have_arg) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "load: file embeds chol; do not pass a buffer");
    }
    tk_fvec_t *cf = tk_fvec_create(NULL, mm);
    if (!cf) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "load: out of memory (chol)");
    }
    cf->n = mm;
    enc->chol_f = cf;
    enc->chol = cf->a;
    tk_lua_fread(L, cf->a, sizeof(float), mm, fh);
  }
  if (enc->mod_type == TK_MOD_CSR) {
    tk_lua_fread(L, &enc->csr_n_tokens, sizeof(uint64_t), 1, fh);
    enc->csr_offsets = (int64_t *)malloc((enc->m + 1) * sizeof(int64_t));
    if (!enc->csr_offsets) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "load: out of memory (landmarks)");
    }
    tk_lua_fread(L, enc->csr_offsets, sizeof(int64_t), enc->m + 1, fh);
    uint64_t total_csr = (uint64_t)enc->csr_offsets[enc->m];
    enc->csr_tokens = (int32_t *)malloc(total_csr * sizeof(int32_t));
    enc->csr_values = (float *)malloc(total_csr * sizeof(float));
    if (!enc->csr_tokens || !enc->csr_values) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "load: out of memory (landmarks)");
    }
    tk_lua_fread(L, enc->csr_tokens, sizeof(int32_t), total_csr, fh);
    tk_lua_fread(L, enc->csr_values, sizeof(float), total_csr, fh);
    tk_lua_fread(L, &enc->n_blocks, sizeof(int), 1, fh);
    if (enc->n_blocks > 0) {
      tk_lua_fread(L, enc->blk_start, sizeof(int64_t), (size_t) enc->n_blocks, fh);
      tk_lua_fread(L, enc->blk_s, sizeof(float), (size_t) enc->n_blocks, fh);
    }
    if (tk_nystrom_build_csc(enc) != 0) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "load: out of memory (csc)");
    }
  } else if (enc->mod_type == TK_MOD_DENSE) {
    tk_lua_fread(L, &enc->d_input, sizeof(int64_t), 1, fh);
    enc->dense_vecs = (float *)malloc(enc->m * (uint64_t)enc->d_input * sizeof(float));
    if (!enc->dense_vecs) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "load: out of memory (landmarks)");
    }
    tk_lua_fread(L, enc->dense_vecs, sizeof(float), enc->m * (uint64_t)enc->d_input, fh);
    uint8_t has_cs;
    tk_lua_fread(L, &has_cs, sizeof(uint8_t), 1, fh);
    if (has_cs) {
      enc->dense_cs2 = (float *)malloc((uint64_t)enc->d_input * sizeof(float));
      if (!enc->dense_cs2) {
        tk_lua_fclose(L, fh);
        return luaL_error(L, "load: out of memory (colscale)");
      }
      tk_lua_fread(L, enc->dense_cs2, sizeof(float), (uint64_t)enc->d_input, fh);
    }
  }

  tk_ivec_load(L, fh);
  int lm_ids_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);
  lua_newtable(L);
  lua_pushvalue(L, lm_ids_idx);
  lua_setfield(L, -2, "landmark_ids");
  if (chol_arg_idx) {
    lua_pushvalue(L, chol_arg_idx);
    lua_setfield(L, -2, "chol");
  }
  lua_setfenv(L, enc_idx);
  lua_pushvalue(L, enc_idx);
  return 1;
}

static inline uint64_t *tk_spectral_row_fps (lua_State *L, uint64_t *n_rows_out) {
  uint64_t *out = NULL;
  uint64_t n_rows = 0;
  lua_getfield(L, 1, "blocks");
  if (!lua_isnil(L, -1)) {
    int bidx = lua_gettop(L);
    int nb = (int) lua_objlen(L, bidx);
    if (nb < 1 || nb > TK_MAX_MOD) { lua_pop(L, 1); return NULL; }
    const int64_t *boff[TK_MAX_MOD]; const int32_t *btok[TK_MAX_MOD]; const float *bval[TK_MAX_MOD];
    for (int b = 0; b < nb; b ++) {
      lua_rawgeti(L, bidx, b + 1);
      lua_getfield(L, -1, "x");
      tk_csr_t *Xb = tk_csr_peek(L, -1, "blocks[].x");
      if (Xb->ntag != TK_TAG_I32) { lua_pop(L, 3); return NULL; }
      if (b == 0) n_rows = tk_csr_rows(Xb);
      boff[b] = Xb->offsets->a;
      btok[b] = (const int32_t *) tk_csr_nbr_ptr(Xb);
      bval[b] = Xb->values ? ((tk_fvec_t *) Xb->values)->a : NULL;
      lua_pop(L, 2);
    }
    lua_pop(L, 1);
    out = (uint64_t *) malloc((n_rows ? n_rows : 1) * sizeof(uint64_t));
    if (!out) return NULL;
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_rows; i ++) {
      uint64_t h = 1469598103934665603ULL;
      for (int b = 0; b < nb; b ++) {
        for (int64_t j = boff[b][i]; j < boff[b][i + 1]; j ++) {
          h ^= (uint32_t) btok[b][j]; h *= 1099511628211ULL;
          if (bval[b]) {
            union { float f; uint32_t u; } cv; cv.f = bval[b][j];
            h ^= cv.u; h *= 1099511628211ULL;
          }
        }
        h ^= 0x9E3779B97F4A7C15ULL; h *= 1099511628211ULL;
      }
      out[i] = h;
    }
    *n_rows_out = n_rows;
    return out;
  }
  lua_pop(L, 1);
  lua_getfield(L, 1, "x");
  if (lua_isnil(L, -1)) { lua_pop(L, 1); return NULL; }
  int xi = lua_gettop(L);
  tk_csr_t *Xc = tk_csr_peekopt(L, xi);
  if (Xc != NULL) {
    n_rows = tk_csr_rows(Xc);
    out = (uint64_t *) malloc((n_rows ? n_rows : 1) * sizeof(uint64_t));
    if (!out) { lua_pop(L, 1); return NULL; }
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_rows; i ++) {
      uint64_t h = 1469598103934665603ULL;
      for (int64_t j = Xc->offsets->a[i]; j < Xc->offsets->a[i + 1]; j ++) {
        h ^= (uint64_t) tk_csr_nbr(Xc, (uint64_t) j); h *= 1099511628211ULL;
        if (Xc->values) {
          union { float f; uint32_t u; } cv; cv.f = (float) tk_csr_val1(Xc, (uint64_t) j);
          h ^= cv.u; h *= 1099511628211ULL;
        }
      }
      out[i] = h;
    }
    lua_pop(L, 1);
    *n_rows_out = n_rows;
    return out;
  }
  tk_mtx_t *Xm = tk_mtx_peek(L, xi, "x");
  n_rows = (uint64_t) Xm->n_rows;
  uint64_t dc = (uint64_t) Xm->n_cols;
  tk_eph_get(L, xi, Xm->v);
  tk_dvec_t *vd = tk_dvec_peekopt(L, -1);
  tk_fvec_t *vf = vd ? NULL : tk_fvec_peekopt(L, -1);
  if (!vd && !vf) { lua_pop(L, 2); return NULL; }
  out = (uint64_t *) malloc((n_rows ? n_rows : 1) * sizeof(uint64_t));
  if (!out) { lua_pop(L, 2); return NULL; }
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_rows; i ++) {
    uint64_t h = 1469598103934665603ULL;
    for (uint64_t k = 0; k < dc; k ++) {
      if (vd) {
        union { double d; uint64_t u; } cv; cv.d = vd->a[i * dc + k];
        h ^= cv.u; h *= 1099511628211ULL;
      } else {
        union { float f; uint32_t u; } cv; cv.f = vf->a[i * dc + k];
        h ^= cv.u; h *= 1099511628211ULL;
      }
    }
    out[i] = h;
  }
  lua_pop(L, 2);
  *n_rows_out = n_rows;
  return out;
}

static inline int tm_uniform_landmarks (lua_State *L) {
  lua_settop(L, 3);
  luaL_checktype(L, 1, LUA_TTABLE);
  uint64_t m = tk_lua_checkunsigned(L, 2, "m");
  uint64_t seedv = (uint64_t) luaL_optnumber(L, 3, 1);
  uint64_t n_rows = 0;
  uint64_t *fp = tk_spectral_row_fps(L, &n_rows);
  if (!fp)
    return luaL_error(L, "uniform_landmarks: expected blocks or x (or oom)");
  if (m > n_rows) m = n_rows;
  if (n_rows == 0 || m == 0) {
    free(fp);
    tk_ivec_t *out = tk_ivec_create(L, 0);
    out->n = 0;
    return 1;
  }
  uint64_t cap = 1;
  while (cap < n_rows * 2) cap <<= 1;
  uint64_t *keys = (uint64_t *) malloc(cap * sizeof(uint64_t));
  uint8_t *used = (uint8_t *) calloc(cap, 1);
  uint8_t *canon = (uint8_t *) calloc(n_rows, 1);
  int64_t *perm = (int64_t *) malloc(n_rows * sizeof(int64_t));
  if (!keys || !used || !canon || !perm) {
    free(fp); free(keys); free(used); free(canon); free(perm);
    return luaL_error(L, "uniform_landmarks: out of memory");
  }
  for (uint64_t i = 0; i < n_rows; i ++) {
    uint64_t h = fp[i];
    uint64_t p = tk_hash_mix(h) & (cap - 1);
    for (;;) {
      if (!used[p]) { used[p] = 1; keys[p] = h; canon[i] = 1; break; }
      if (keys[p] == h) break;
      p = (p + 1) & (cap - 1);
    }
  }
  free(keys); free(used); free(fp);
  for (uint64_t i = 0; i < n_rows; i ++) perm[i] = (int64_t) i;
  uint64_t st = tk_hash_mix(seedv ? seedv : 1);
  for (uint64_t i = n_rows - 1; i > 0; i --) {
    uint64_t j = (uint64_t) (tk_lm_drand(&st) * (double) (i + 1));
    if (j > i) j = i;
    int64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  tk_ivec_t *out = tk_ivec_create(L, m);
  uint64_t taken = 0;
  for (uint64_t i = 0; i < n_rows && taken < m; i ++) {
    int64_t r = perm[i];
    if (canon[r]) out->a[taken ++] = r;
  }
  out->n = taken;
  free(perm); free(canon);
  return 1;
}

static luaL_Reg tm_fns[] =
{
  { "encode", tm_encode },
  { "uniform_landmarks", tm_uniform_landmarks },
  { "load", tk_nystrom_encoder_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_spectral (lua_State *L)
{
  tk_lua_require_mod(L, "santoku.csr");
  tk_lua_require_mod(L, "santoku.mtx");
  lua_newtable(L);
  tk_lua_register(L, tm_fns, 0);
  return 1;
}
