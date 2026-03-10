#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/dvec.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/learn/inv.h>
#include <santoku/learn/buf.h>
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

static inline double tk_spectral_pos_delta (int64_t p1, int64_t p2, int64_t row_length) {
  if (row_length > 0) {
    int64_t r1 = p1 / row_length, c1 = p1 % row_length;
    int64_t r2 = p2 / row_length, c2 = p2 % row_length;
    int64_t dr = llabs(r1 - r2), dc = llabs(c1 - c2);
    return (double)(dr > dc ? dr : dc);
  }
  return fabs((double)(p1 - p2));
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

typedef struct { int64_t tok; double val; int64_t pos; } tk_csr_sort_entry_t;
#define tk_csr_sort_entry_lt(a, b) ((a).tok < (b).tok)
KSORT_INIT(tk_csr_sort_entry, tk_csr_sort_entry_t, tk_csr_sort_entry_lt)

#define TK_CHOL_BLOCK 64

static inline void tk_spectral_sample_landmarks (
  lua_State *L,
  tk_inv_t *inv,
  double *dense, int64_t d_input, uint64_t n_docs_hint,
  const uint8_t *bits_data, uint64_t bits_d, uint64_t bits_n_samples,
  const int64_t *csr_offsets, const int64_t *csr_tokens,
  const double *csr_values, const double *csr_norms, uint64_t csr_n_samples,
  const int64_t *csr_positions, int64_t pos_window, int64_t row_length,
  tk_spectral_kernel_t kernel, const double *dense_norms,
  uint64_t n_landmarks,
  double decay,
  double trace_tol,
  tk_ivec_t **ids_out,
  tk_dvec_t **chol_out,
  double **full_chol_out,
  tk_ivec_t **full_chol_ids_out,
  uint64_t *actual_landmarks_out,
  double *trace_ratio_out
) {
  int is_dense = (inv == NULL && dense != NULL && csr_offsets == NULL);
  int is_bits = (inv == NULL && dense == NULL && bits_data != NULL && csr_offsets == NULL);
  int is_csr = (csr_offsets != NULL);
  uint64_t bits_row_bytes = is_bits ? TK_CVEC_BITS_BYTES(bits_d) : 0;
  uint64_t n_docs = 0;
  if (is_csr) {
    n_docs = csr_n_samples;
  } else if (is_dense || is_bits) {
    n_docs = is_dense ? n_docs_hint : bits_n_samples;
  } else {
    for (int64_t sid = 0; sid < inv->next_sid; sid++) {
      if (inv->sid_to_uid->a[sid] >= 0)
        n_docs++;
    }
  }

  if (n_landmarks == 0 || n_landmarks > n_docs)
    n_landmarks = n_docs;
  if (n_landmarks == 0) {
    *ids_out = tk_ivec_create(L, 0, 0, 0);
    *chol_out = NULL;
    *full_chol_out = NULL;
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

  uint64_t L_stride = n_landmarks;
  ctx->sid_map = (int64_t *)malloc(n_docs * sizeof(int64_t));
  ctx->residual = (double *)malloc(n_docs * sizeof(double));
  ctx->L_mat = (double *)calloc(n_docs * L_stride, sizeof(double));
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
  if (is_dense || is_bits || is_csr) {
    for (uint64_t i = 0; i < n_docs; i++)
      sid_map[i] = (int64_t)i;
    idx = n_docs;
  } else {
    for (int64_t sid = 0; sid < inv->next_sid; sid++) {
      if (inv->sid_to_uid->a[sid] >= 0)
        sid_map[idx++] = sid;
    }
  }

  tk_inv_rank_weights_t rw;
  const double *weights_arr = NULL;
  const int64_t *node_bits_a = NULL;
  const int64_t *nro_a = NULL;
  const double *node_weights_a = NULL;
  uint64_t n_ranks = 0;
  uint64_t stride = 0;
  if (!is_dense && !is_bits && !is_csr) {
    tk_inv_precompute_rank_weights(&rw, inv->n_ranks, decay);
    weights_arr = inv->weights->a;
    node_bits_a = inv->node_bits->a;
    nro_a = inv->node_rank_offsets->a;
    node_weights_a = inv->node_weights->a;
    n_ranks = inv->n_ranks;
    stride = n_ranks + 1;
  }

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
    if (is_bits) {
      residual[i] = 1.0;
    } else if (is_csr) {
      residual[i] = (kernel == TK_SPECTRAL_LINEAR)
        ? csr_norms[i] * csr_norms[i]
        : 1.0;
    } else if (is_dense) {
      residual[i] = (kernel == TK_SPECTRAL_LINEAR)
        ? cblas_ddot((int)d_input, dense + i * (uint64_t)d_input, 1, dense + i * (uint64_t)d_input, 1)
        : 1.0;
    } else {
      const double *nw = node_weights_a + sid_map[i] * (int64_t) n_ranks;
      double accum = 0.0;
      for (uint64_t r = 0; r < n_ranks; r++)
        if (nw[r] > 0.0) accum += rw.weights[r];
      residual[i] = (rw.total > 0.0) ? accum / rw.total : 0.0;
    }
    initial_trace += residual[i];
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

      if (pos_window > 0) {
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++) {
          int64_t ilo = csr_offsets[i], ihi = csr_offsets[i + 1];
          for (uint64_t b = 0; b < np; b++) {
            double sim = 0.0;
            int64_t ii = ilo, pp = plo_arr[b];
            while (ii < ihi && pp < phi_arr[b]) {
              if (csr_tokens[ii] == csr_tokens[pp]) {
                int64_t tok = csr_tokens[ii];
                int64_t i_start = ii, p_start = pp;
                while (ii < ihi && csr_tokens[ii] == tok) ii++;
                while (pp < phi_arr[b] && csr_tokens[pp] == tok) pp++;
                for (int64_t a = i_start; a < ii; a++)
                  for (int64_t c = p_start; c < pp; c++) {
                    double adelta = tk_spectral_pos_delta(
                      csr_positions[a], csr_positions[c], row_length);
                    if (adelta < pos_window)
                      sim += 1.0 - adelta / (double)pos_window;
                  }
              } else if (csr_tokens[ii] < csr_tokens[pp]) {
                ii++;
              } else {
                pp++;
              }
            }
            kip_block[i * np + b] = sim;
          }
        }
      } else {
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n_docs; i++) {
          int64_t ilo = csr_offsets[i], ihi = csr_offsets[i + 1];
          for (uint64_t b = 0; b < np; b++) {
            double dot = 0.0;
            int64_t ii = ilo, pp = plo_arr[b];
            while (ii < ihi && pp < phi_arr[b]) {
              if (csr_tokens[ii] == csr_tokens[pp]) {
                int64_t tok = csr_tokens[ii];
                double i_sum = 0.0;
                while (ii < ihi && csr_tokens[ii] == tok) { i_sum += csr_values[ii]; ii++; }
                double p_sum = 0.0;
                while (pp < phi_arr[b] && csr_tokens[pp] == tok) { p_sum += csr_values[pp]; pp++; }
                dot += i_sum * p_sum;
              } else if (csr_tokens[ii] < csr_tokens[pp]) {
                ii++;
              } else {
                pp++;
              }
            }
            kip_block[i * np + b] = dot;
          }
        }
      }

      if (kernel != TK_SPECTRAL_LINEAR) {
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
      if (kernel != TK_SPECTRAL_LINEAR) {
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

    } else {
      #pragma omp parallel
      {
        double thr_i[TK_INV_MAX_RANKS];
        #pragma omp for schedule(guided)
        for (uint64_t i = 0; i < n_docs; i++) {
          const int64_t *i_ro = nro_a + sid_map[i] * (int64_t)stride;
          const double *i_nw = node_weights_a + sid_map[i] * (int64_t)n_ranks;
          for (uint64_t b = 0; b < np; b++) {
            const int64_t *p_ro = nro_a + sid_map[blk_pivots[b]] * (int64_t)stride;
            const double *p_nw = node_weights_a + sid_map[blk_pivots[b]] * (int64_t)n_ranks;
            kip_block[i * np + b] = tk_inv_similarity_fast_cached(
              weights_arr, n_ranks, node_bits_a, i_ro, node_bits_a, p_ro,
              &rw, i_nw, p_nw, thr_i, inv->kernel);
          }
        }
      }
    }

    uint64_t jb = actual_landmarks;
    if (jb > 0) {
      for (uint64_t b = 0; b < np; b++)
        memcpy(pivot_prev_L + b * jb,
               L_mat + blk_pivots[b] * L_stride,
               jb * sizeof(double));
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        (int)n_docs, (int)np, (int)jb, 1.0,
        L_mat, (int)L_stride,
        pivot_prev_L, (int)jb,
        0.0, cross_dots, (int)np);
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

      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_docs; i++) {
        double raw = kip_block[i * np + b];
        double cross = (jb > 0) ? cross_dots[i * np + b] : 0.0;
        double within = 0.0;
        for (uint64_t k = 0; k < within_accepted; k++)
          within += L_mat[i * L_stride + jb + k] *
                    L_mat[pi * L_stride + jb + k];
        double lij = (raw - cross - within) / sc;
        L_mat[i * L_stride + col] = lij;
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
    landmark_ids->a[i] = (is_dense || is_bits || is_csr) ? landmark_sids[i] : inv->sid_to_uid->a[landmark_sids[i]];
  landmark_ids->n = actual_landmarks;

  tk_dvec_t *chol = tk_dvec_create(NULL, actual_landmarks * actual_landmarks, 0, 0);
  #pragma omp parallel for schedule(static)
  for (uint64_t li = 0; li < actual_landmarks; li++) {
    uint64_t doc_idx = (uint64_t)landmark_idx_map[li];
    memcpy(chol->a + li * actual_landmarks, L_mat + doc_idx * L_stride,
           actual_landmarks * sizeof(double));
  }
  chol->n = actual_landmarks * actual_landmarks;

  if (L_stride > actual_landmarks) {
    for (uint64_t i = 1; i < n_docs; i++)
      memmove(L_mat + i * actual_landmarks, L_mat + i * L_stride,
              actual_landmarks * sizeof(double));
    double *shrunk = (double *)realloc(L_mat, n_docs * actual_landmarks * sizeof(double));
    if (shrunk) L_mat = shrunk;
  }
  double *full_chol = L_mat;
  ctx->L_mat = NULL;

  tk_ivec_t *full_chol_ids = tk_ivec_create(L, n_docs, 0, 0);
  for (uint64_t i = 0; i < n_docs; i++)
    full_chol_ids->a[i] = (is_dense || is_bits || is_csr) ? (int64_t)i : inv->sid_to_uid->a[sid_map[i]];
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
  int64_t *lm_sids;
  double *adjustment;
  double *projection;
  double *inv_sqrt_eig;
  tk_dvec_t *lm_chol;
  double *full_chol;
  tk_dvec_t *gram;
  tk_dvec_t *cmeans;
  tk_dvec_t *eigvecs;
} tk_encode_nystrom_ctx_t;

static inline int tk_encode_nystrom_ctx_gc (lua_State *L) {
  tk_encode_nystrom_ctx_t *c = (tk_encode_nystrom_ctx_t *)lua_touserdata(L, 1);
  free(c->lm_sids);
  free(c->adjustment);
  free(c->projection);
  free(c->inv_sqrt_eig);
  if (c->lm_chol) tk_dvec_destroy(c->lm_chol);
  free(c->full_chol);
  if (c->gram) tk_dvec_destroy(c->gram);
  if (c->cmeans) tk_dvec_destroy(c->cmeans);
  if (c->eigvecs) tk_dvec_destroy(c->eigvecs);
  return 0;
}

#define TK_NYSTROM_ENCODER_MT "tk_nystrom_encoder_t"

typedef enum {
  TK_BINARIZE_NONE = 0,
  TK_BINARIZE_SIGN = 1,
  TK_BINARIZE_ITQ  = 2,
} tk_binarize_mode_t;

typedef struct {
  double *projection;
  double *adjustment;
  double *inv_sqrt_eig;
  int64_t *lm_sids;
  double *lm_vecs;
  uint8_t *lm_bits;
  int64_t *lm_csr_offsets;
  int64_t *lm_csr_tokens;
  double *lm_csr_values;
  double *lm_csr_norms;
  int64_t *lm_csc_offsets;
  int64_t *lm_csc_rows;
  double *lm_csc_values;
  int64_t *lm_csc_positions;
  double *feature_weights;
  uint64_t fw_len;
  double *lm_dense_norms;
  double *itq_rotation;
  double *itq_means;
  uint64_t binarize_dims;
  tk_binarize_mode_t binarize_mode;
  int64_t pos_window;
  int64_t row_length;
  uint64_t m;
  uint64_t d;
  int64_t d_input;
  uint64_t bits_d;
  uint64_t csr_n_tokens;
  double decay;
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
    free(enc->adjustment);
    free(enc->inv_sqrt_eig);
    free(enc->lm_sids);
    free(enc->lm_vecs);
    free(enc->lm_bits);
    free(enc->lm_csr_offsets);
    free(enc->lm_csr_tokens);
    free(enc->lm_csr_values);
    free(enc->lm_csr_norms);
    free(enc->lm_csc_offsets);
    free(enc->lm_csc_rows);
    free(enc->lm_csc_values);
    free(enc->lm_csc_positions);
    free(enc->feature_weights);
    free(enc->lm_dense_norms);
    free(enc->itq_rotation);
    free(enc->itq_means);
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
    tk_dvec_t *sims_dv = (lua_gettop(L) >= 5 && lua_isuserdata(L, 5))
      ? tk_dvec_peek(L, 5, "sims") : NULL;
    TK_DVEC_BUF(out, 6, n_samples * d);
    if (tile == 0 || tile > n_samples) tile = n_samples;
    uint64_t sims_need = tile * m;
    double *sims;
    if (sims_dv) {
      tk_dvec_ensure(sims_dv, sims_need);
      sims_dv->n = sims_need;
      sims = sims_dv->a;
    } else {
      sims = (double *)malloc(sims_need * sizeof(double));
      if (!sims) return luaL_error(L, "encode: out of memory");
    }
    for (uint64_t base = 0; base < n_samples; base += tile) {
      uint64_t blk = base + tile <= n_samples ? tile : n_samples - base;
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < blk; i++) {
        const uint8_t *row_i = bdata + (base + i) * row_bytes;
        double *sims_row = sims + i * m;
        for (uint64_t j = 0; j < m; j++) {
          uint64_t ham = tk_cvec_bits_hamming_serial(
            row_i, enc->lm_bits + j * row_bytes, bd);
          sims_row[j] = 1.0 - (double)ham * inv_bd;
        }
      }
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)blk, (int)d, (int)m, 1.0,
        sims, (int)m,
        enc->projection, (int)d,
        0.0, out->a + base * d, (int)d);
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < blk; i++) {
        double *row = out->a + (base + i) * d;
        for (uint64_t k = 0; k < d; k++) {
          row[k] -= enc->adjustment[k];
          row[k] *= enc->inv_sqrt_eig[k];
        }
        double nrm = cblas_dnrm2((int)d, row, 1);
        if (nrm > 1e-12) cblas_dscal((int)d, 1.0 / nrm, row, 1);
      }
    }
    if (!sims_dv) free(sims);
    lua_pushvalue(L, out_idx);
    return 1;
  }

  if (enc->dense_mode == 1) {
    tk_dvec_t *codes = tk_dvec_peek(L, 2, "codes");
    uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
    int64_t d_in = enc->d_input;
    double *src = codes->a;
    double *tmp_scaled = NULL;
    tk_dvec_t *sims_dv = (lua_gettop(L) >= 4 && lua_isuserdata(L, 4))
      ? tk_dvec_peek(L, 4, "sims") : NULL;
    if (enc->feature_weights) {
      tmp_scaled = (double *)malloc(n_samples * (uint64_t)d_in * sizeof(double));
      if (!tmp_scaled) return luaL_error(L, "encode: out of memory");
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++)
        for (int64_t k = 0; k < d_in; k++)
          tmp_scaled[i * (uint64_t)d_in + (uint64_t)k] = codes->a[i * (uint64_t)d_in + (uint64_t)k] * enc->feature_weights[k];
      src = tmp_scaled;
    }
    TK_DVEC_BUF(out, 5, n_samples * d);
    uint64_t sims_need = n_samples * m;
    double *sims;
    if (sims_dv) {
      tk_dvec_ensure(sims_dv, sims_need);
      sims_dv->n = sims_need;
      sims = sims_dv->a;
    } else {
      sims = (double *)malloc(sims_need * sizeof(double));
      if (!sims) { free(tmp_scaled); return luaL_error(L, "encode: out of memory"); }
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
      (int)n_samples, (int)m, (int)d_in, 1.0,
      src, (int)d_in,
      enc->lm_vecs, (int)d_in,
      0.0, sims, (int)m);
    if (enc->kernel != TK_SPECTRAL_LINEAR) {
      double *new_norms = (double *)malloc(n_samples * sizeof(double));
      if (!new_norms) { if (!sims_dv) free(sims); free(tmp_scaled); return luaL_error(L, "encode: out of memory"); }
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++)
        new_norms[i] = cblas_dnrm2((int)d_in, src + i * (uint64_t)d_in, 1);
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++)
        for (uint64_t j = 0; j < m; j++) {
          double denom = new_norms[i] * enc->lm_dense_norms[j];
          sims[i * m + j] = tk_spectral_kernel_apply(enc->kernel,
            denom > 1e-15 ? sims[i * m + j] / denom : 0.0);
        }
      free(new_norms);
    }
    free(tmp_scaled);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)n_samples, (int)d, (int)m, 1.0,
      sims, (int)m,
      enc->projection, (int)d,
      0.0, out->a, (int)d);
    if (!sims_dv) free(sims);
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      double *row = out->a + i * d;
      for (uint64_t k = 0; k < d; k++) {
        row[k] -= enc->adjustment[k];
        row[k] *= enc->inv_sqrt_eig[k];
      }
      double nrm = cblas_dnrm2((int)d, row, 1);
      if (nrm > 1e-12) cblas_dscal((int)d, 1.0 / nrm, row, 1);
    }
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
    lua_getfield(L, 2, "sims");
    tk_dvec_t *sims_dv = tk_dvec_peekopt(L, -1);
    lua_pop(L, 1);
    lua_getfield(L, 2, "output");
    tk_dvec_t *out_dv = tk_dvec_peekopt(L, -1);
    int out_dv_idx = out_dv ? lua_gettop(L) : 0;
    if (!out_dv) lua_pop(L, 1);
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
    if (enc->pos_window > 0) {
      for (uint64_t s = 0; s < n_samples; s++) {
        int64_t lo = offsets->a[s], hi = offsets->a[s + 1];
        double ss = 0.0;
        for (int64_t a = lo; a < hi; a++)
          for (int64_t b = a; b < hi; b++)
            if (tokens->a[a] == tokens->a[b]) {
              double adelta = tk_spectral_pos_delta(a - lo, b - lo, enc->row_length);
              if (adelta < enc->pos_window) {
                double ct = 1.0 - adelta / (double)enc->pos_window;
                ss += (a == b) ? ct : 2.0 * ct;
              }
            }
        norms[s] = sqrt(ss);
      }
    } else {
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
    double *sims;
    if (sims_dv) {
      tk_dvec_ensure(sims_dv, sims_need);
      sims_dv->n = sims_need;
      sims = sims_dv->a;
      memset(sims, 0, sims_need * sizeof(double));
    } else {
      sims = (double *)calloc(sims_need, sizeof(double));
      if (!sims) { free(sval); free(norms); return luaL_error(L, "encode: out of memory"); }
    }
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      for (int64_t j = offsets->a[i]; j < offsets->a[i + 1]; j++) {
        int64_t tok = tokens->a[j];
        if (enc->pos_window > 0) {
          int64_t input_pos = j - offsets->a[i];
          for (int64_t c = enc->lm_csc_offsets[tok]; c < enc->lm_csc_offsets[tok + 1]; c++) {
            double adelta = tk_spectral_pos_delta(input_pos, enc->lm_csc_positions[c], enc->row_length);
            if (adelta < enc->pos_window)
              sims[i * m + (uint64_t)enc->lm_csc_rows[c]] += 1.0 - adelta / (double)enc->pos_window;
          }
        } else {
          double val = sval[j];
          for (int64_t c = enc->lm_csc_offsets[tok]; c < enc->lm_csc_offsets[tok + 1]; c++)
            sims[i * m + (uint64_t)enc->lm_csc_rows[c]] += val * enc->lm_csc_values[c];
        }
      }
    }
    if (enc->kernel != TK_SPECTRAL_LINEAR) {
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++)
        for (uint64_t j = 0; j < m; j++) {
          double denom = norms[i] * enc->lm_csr_norms[j];
          sims[i * m + j] = tk_spectral_kernel_apply(enc->kernel,
            denom > 1e-15 ? sims[i * m + j] / denom : 0.0);
        }
    }
    free(sval); free(norms);
    tk_dvec_t *out;
    if (out_dv) {
      tk_dvec_ensure(out_dv, n_samples * d);
      out_dv->n = n_samples * d;
      out = out_dv;
    } else {
      out = tk_dvec_create(L, n_samples * d, 0, 0);
      out->n = n_samples * d;
      out_dv_idx = lua_gettop(L);
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)n_samples, (int)d, (int)m, 1.0,
      sims, (int)m,
      enc->projection, (int)d,
      0.0, out->a, (int)d);
    if (!sims_dv) free(sims);
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      double *row = out->a + i * d;
      for (uint64_t k = 0; k < d; k++) {
        row[k] -= enc->adjustment[k];
        row[k] *= enc->inv_sqrt_eig[k];
      }
      double nrm = cblas_dnrm2((int)d, row, 1);
      if (nrm > 1e-12) cblas_dscal((int)d, 1.0 / nrm, row, 1);
    }
    lua_pushvalue(L, out_dv_idx);
    return 1;
  }

  tk_ivec_t *sparse = tk_ivec_peek(L, 2, "sparse_bits");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_features = tk_lua_checkunsigned(L, 4, "n_features");
  tk_dvec_t *sims_dv = (lua_gettop(L) >= 5 && lua_isuserdata(L, 5))
    ? tk_dvec_peek(L, 5, "sims") : NULL;
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "inv");
  tk_inv_t *inv = tk_inv_peek(L, -1);
  lua_pop(L, 2);
  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, inv->n_ranks, enc->decay);
  TK_DVEC_BUF(out, 6, n_samples * d);
  uint64_t nystrom_stride = inv->n_ranks + 1;
  const int64_t *nystrom_nro = inv->node_rank_offsets->a;
  const int64_t *nystrom_nb = inv->node_bits->a;
  uint64_t sims_need = n_samples * m;
  double *sims_all;
  if (sims_dv) {
    tk_dvec_ensure(sims_dv, sims_need);
    sims_dv->n = sims_need;
    sims_all = sims_dv->a;
  } else {
    sims_all = (double *)malloc(sims_need * sizeof(double));
    if (!sims_all) return luaL_error(L, "encode: out of memory");
  }
  #pragma omp parallel
  {
    double thr_q[TK_INV_MAX_RANKS], thr_e[TK_INV_MAX_RANKS], thr_i[TK_INV_MAX_RANKS];
    int64_t q_ro[TK_INV_MAX_RANKS + 1];
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
      double *sims_row = sims_all + i * m;
      for (uint64_t j = 0; j < m; j++) {
        if (enc->lm_sids[j] >= 0 && nf > 0) {
          const int64_t *lm_ro = nystrom_nro + enc->lm_sids[j] * (int64_t) nystrom_stride;
          sims_row[j] = tk_inv_similarity_fast(inv->weights->a, inv->n_ranks,
            feat_buf, q_ro, nystrom_nb, lm_ro, &rw, thr_q, thr_e, thr_i, inv->kernel);
        } else {
          sims_row[j] = 0.0;
        }
      }
    }
    free(feat_buf);
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)n_samples, (int)d, (int)m, 1.0,
    sims_all, (int)m,
    enc->projection, (int)d,
    0.0, out->a, (int)d);
  if (!sims_dv) free(sims_all);
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    double *row = out->a + i * d;
    for (uint64_t k = 0; k < d; k++) {
      row[k] -= enc->adjustment[k];
      row[k] *= enc->inv_sqrt_eig[k];
    }
    double nrm = cblas_dnrm2((int)d, row, 1);
    if (nrm > 1e-12) cblas_dscal((int)d, 1.0 / nrm, row, 1);
  }
  lua_pushvalue(L, out_idx);
  return 1;
}

static inline int tk_nystrom_dims_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->d);
  return 1;
}

static inline int tk_nystrom_binarize_dims_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (enc->binarize_mode == TK_BINARIZE_NONE) {
    lua_pushnil(L);
  } else {
    lua_pushinteger(L, (lua_Integer)enc->binarize_dims);
  }
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

static inline void tk_nystrom_binarize_core (lua_State *L, tk_nystrom_encoder_t *enc, double *src, uint64_t src_stride, uint64_t n_samples) {
  uint64_t k = enc->binarize_dims;
  uint64_t n_bytes = TK_CVEC_BITS_BYTES(k);
  tk_cvec_t *out = tk_cvec_create(L, n_samples * n_bytes, NULL, NULL);
  memset(out->a, 0, out->n);
  if (enc->binarize_mode == TK_BINARIZE_ITQ) {
    double *centered = (double *)malloc(n_samples * k * sizeof(double));
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++)
      for (uint64_t j = 0; j < k; j++)
        centered[i * k + j] = src[i * src_stride + j] - enc->itq_means[j];
    double *rotated = (double *)malloc(n_samples * k * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)n_samples, (int)k, (int)k, 1.0, centered, (int)k,
      enc->itq_rotation, (int)k, 0.0, rotated, (int)k);
    free(centered);
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      double *r = rotated + i * k;
      uint8_t *dest = (uint8_t *)out->a + i * n_bytes;
      for (uint64_t j = 0; j < k; j++)
        if (r[j] > 0.0) dest[TK_CVEC_BITS_BYTE(j)] |= (1 << TK_CVEC_BITS_BIT(j));
    }
    free(rotated);
  } else {
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      const double *r = src + i * src_stride;
      uint8_t *dest = (uint8_t *)out->a + i * n_bytes;
      for (uint64_t j = 0; j < k; j++)
        if (r[j] > 0.0) dest[TK_CVEC_BITS_BYTE(j)] |= (1 << TK_CVEC_BITS_BIT(j));
    }
  }
}

static inline int tk_nystrom_binarize_raw_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (enc->binarize_mode == TK_BINARIZE_NONE) return luaL_error(L, "encoder has no binarization");
  tk_dvec_t *dv = tk_dvec_peek(L, 2, "codes");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  tk_nystrom_binarize_core(L, enc, dv->a, enc->d, n_samples);
  return 1;
}

static inline int tk_nystrom_binary_lua (lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  if (enc->binarize_mode == TK_BINARIZE_NONE) return luaL_error(L, "encoder has no binarization");
  uint64_t n_samples = 0;
  if (enc->dense_mode == 3) {
    lua_getfield(L, 2, "n_samples");
    n_samples = (uint64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
  } else
    n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  tk_nystrom_encode_lua(L);
  tk_dvec_t *dv = (tk_dvec_t *)lua_touserdata(L, -1);
  tk_nystrom_binarize_core(L, enc, dv->a, enc->d, n_samples);
  lua_remove(L, -2);
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
  double *new_proj = (double *)malloc(enc->m * new_d * sizeof(double));
  double *new_adj = (double *)malloc(new_d * sizeof(double));
  double *new_ise = (double *)malloc(new_d * sizeof(double));
  for (uint64_t j = 0; j < enc->m; j++)
    for (uint64_t i = 0; i < new_d; i++)
      new_proj[j * new_d + i] = enc->projection[j * enc->d + (uint64_t)keep->a[i]];
  for (uint64_t i = 0; i < new_d; i++) {
    new_adj[i] = enc->adjustment[(uint64_t)keep->a[i]];
    new_ise[i] = enc->inv_sqrt_eig[(uint64_t)keep->a[i]];
  }
  free(enc->projection);
  free(enc->adjustment);
  free(enc->inv_sqrt_eig);
  enc->projection = new_proj;
  enc->adjustment = new_adj;
  enc->inv_sqrt_eig = new_ise;
  enc->d = new_d;
  if (enc->binarize_mode != TK_BINARIZE_NONE && new_d < enc->binarize_dims) {
    free(enc->itq_rotation); enc->itq_rotation = NULL;
    free(enc->itq_means); enc->itq_means = NULL;
    enc->binarize_dims = 0;
    enc->binarize_mode = TK_BINARIZE_NONE;
  }
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
  uint8_t version = 12;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->dense_mode, sizeof(uint8_t), 1, fh);
  uint8_t kernel_byte = (uint8_t)enc->kernel;
  tk_lua_fwrite(L, &kernel_byte, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->m, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->d, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->decay, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &enc->trace_ratio, sizeof(double), 1, fh);
  tk_lua_fwrite(L, enc->projection, sizeof(double), enc->m * enc->d, fh);
  tk_lua_fwrite(L, enc->adjustment, sizeof(double), enc->d, fh);
  tk_lua_fwrite(L, enc->inv_sqrt_eig, sizeof(double), enc->d, fh);
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
    uint8_t has_norms = enc->lm_dense_norms ? 1 : 0;
    tk_lua_fwrite(L, &has_norms, sizeof(uint8_t), 1, fh);
    if (has_norms)
      tk_lua_fwrite(L, enc->lm_dense_norms, sizeof(double), enc->m, fh);
  } else if (enc->dense_mode == 3) {
    tk_lua_fwrite(L, &enc->csr_n_tokens, sizeof(uint64_t), 1, fh);
    uint64_t total_csr = (uint64_t)enc->lm_csr_offsets[enc->m];
    tk_lua_fwrite(L, enc->lm_csr_offsets, sizeof(int64_t), enc->m + 1, fh);
    tk_lua_fwrite(L, enc->lm_csr_tokens, sizeof(int64_t), total_csr, fh);
    tk_lua_fwrite(L, enc->lm_csr_values, sizeof(double), total_csr, fh);
    tk_lua_fwrite(L, enc->lm_csr_norms, sizeof(double), enc->m, fh);
    tk_lua_fwrite(L, &enc->pos_window, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, &enc->row_length, sizeof(int64_t), 1, fh);
  } else {
    tk_lua_fwrite(L, enc->lm_sids, sizeof(int64_t), enc->m, fh);
  }
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "landmark_ids");
  tk_ivec_t *lm_ids = tk_ivec_peek(L, -1, "landmark_ids");
  tk_ivec_persist(L, lm_ids, fh);
  lua_pop(L, 2);
  uint8_t bin_mode = (uint8_t)enc->binarize_mode;
  tk_lua_fwrite(L, &bin_mode, sizeof(uint8_t), 1, fh);
  if (bin_mode != TK_BINARIZE_NONE) {
    tk_lua_fwrite(L, &enc->binarize_dims, sizeof(uint64_t), 1, fh);
    if (bin_mode == TK_BINARIZE_ITQ) {
      tk_lua_fwrite(L, enc->itq_rotation, sizeof(double), enc->binarize_dims * enc->binarize_dims, fh);
      tk_lua_fwrite(L, enc->itq_means, sizeof(double), enc->binarize_dims, fh);
    }
  }
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
  { "continuous", tk_nystrom_encode_lua },
  { "binary", tk_nystrom_binary_lua },
  { "binarize", tk_nystrom_binarize_raw_lua },
  { "dims", tk_nystrom_dims_lua },
  { "binarize_dims", tk_nystrom_binarize_dims_lua },
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

  lua_getfield(L, 1, "inv");
  int has_inv = !lua_isnil(L, -1);
  tk_inv_t *feat_inv = has_inv ? tk_inv_peek(L, -1) : NULL;
  int feat_inv_idx = lua_gettop(L);

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
  if (!has_inv) {
    lua_pop(L, 1);
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
  const char *kernel_str = lua_isnil(L, -1) ? "linear" : lua_tostring(L, -1);
  lua_pop(L, 1);
  tk_spectral_kernel_t kernel = TK_SPECTRAL_LINEAR;
  if (strcmp(kernel_str, "cosine") == 0) kernel = TK_SPECTRAL_COSINE;
  else if (strcmp(kernel_str, "arccos0") == 0) kernel = TK_SPECTRAL_ARCCOS0;
  else if (strcmp(kernel_str, "arccos1") == 0) kernel = TK_SPECTRAL_ARCCOS1;
  else if (strcmp(kernel_str, "linear") != 0)
    return luaL_error(L, "encode: unknown kernel '%s'", kernel_str);

  int64_t pos_window = (int64_t)tk_lua_foptinteger(L, 1, "encode", "pos_window", 0);
  int64_t row_length = (int64_t)tk_lua_foptinteger(L, 1, "encode", "row_length", 0);

  if (dense_data && kernel != TK_SPECTRAL_LINEAR) {
    dense_norms = (double *)malloc(dense_n_samples * sizeof(double));
    for (uint64_t i = 0; i < dense_n_samples; i++)
      dense_norms[i] = cblas_dnrm2((int)dense_d_input, dense_data + i * (uint64_t)dense_d_input, 1);
  }

  uint64_t n_lm_req = tk_lua_foptunsigned(L, 1, "encode", "n_landmarks", 0);
  uint64_t n_dims_req = tk_lua_foptunsigned(L, 1, "encode", "n_dims", 0);
  double decay = tk_lua_foptnumber(L, 1, "encode", "decay", 0.0);
  double trace_tol = tk_lua_foptnumber(L, 1, "encode", "trace_tol", 0.0);
  int cholesky_mode = tk_lua_foptboolean(L, 1, "encode", "cholesky", 0);

  int64_t *sorted_csr_tokens = NULL;
  double *sorted_csr_values = NULL;
  int64_t *sorted_csr_positions = NULL;
  if (has_csr) {
    uint64_t nnz = (uint64_t)csr_offsets_a[csr_n_samples];
    sorted_csr_tokens = (int64_t *)malloc(nnz * sizeof(int64_t));
    sorted_csr_values = (double *)malloc(nnz * sizeof(double));
    memcpy(sorted_csr_tokens, csr_tokens_a, nnz * sizeof(int64_t));
    memcpy(sorted_csr_values, csr_values_a, nnz * sizeof(double));
    if (pos_window > 0) {
      sorted_csr_positions = (int64_t *)malloc(nnz * sizeof(int64_t));
      for (uint64_t s = 0; s < csr_n_samples; s++)
        for (int64_t j = csr_offsets_a[s]; j < csr_offsets_a[s + 1]; j++)
          sorted_csr_positions[j] = j - csr_offsets_a[s];
    }
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
          buf[j].pos = sorted_csr_positions ? sorted_csr_positions[lo + j] : 0;
        }
        ks_introsort_tk_csr_sort_entry((size_t)len, buf);
        for (int64_t j = 0; j < len; j++) {
          sorted_csr_tokens[lo + j] = buf[j].tok;
          sorted_csr_values[lo + j] = buf[j].val;
          if (sorted_csr_positions)
            sorted_csr_positions[lo + j] = buf[j].pos;
        }
      }
      free(buf);
    }
    if (pos_window > 0) {
      for (uint64_t s = 0; s < csr_n_samples; s++) {
        int64_t lo = csr_offsets_a[s], hi = csr_offsets_a[s + 1];
        double ss = 0.0;
        int64_t j = lo;
        while (j < hi) {
          int64_t tok = sorted_csr_tokens[j];
          int64_t start = j;
          while (j < hi && sorted_csr_tokens[j] == tok) j++;
          for (int64_t a = start; a < j; a++)
            for (int64_t b = start; b < j; b++) {
              double adelta = tk_spectral_pos_delta(
                sorted_csr_positions[a], sorted_csr_positions[b], row_length);
              if (adelta < pos_window)
                ss += 1.0 - adelta / (double)pos_window;
            }
        }
        csr_norms_a[s] = sqrt(ss);
      }
    } else {
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

  tk_ivec_t *lm_ids = NULL, *chol_ids = NULL;
  tk_dvec_t *lm_chol = NULL;
  double *full_chol = NULL;
  uint64_t m;
  double trace_ratio;
  tk_spectral_sample_landmarks(L, feat_inv,
    dense_data, dense_d_input, dense_n_samples,
    bits_data, bits_d, bits_n_samples,
    csr_offsets_a, sorted_csr_tokens, sorted_csr_values, csr_norms_a, csr_n_samples,
    sorted_csr_positions, pos_window, row_length,
    kernel, dense_norms,
    n_lm_req, decay, trace_tol,
    &lm_ids, &lm_chol, &full_chol, &chol_ids, &m, &trace_ratio);
  free(sorted_csr_tokens); free(sorted_csr_values); free(sorted_csr_positions);
  int lm_ids_idx = lua_gettop(L) - 1;
  int chol_ids_idx = lua_gettop(L);

  uint64_t d = (n_dims_req == 0 || n_dims_req > m) ? m : n_dims_req;

  if (m == 0 || d == 0) {
    if (lm_chol) tk_dvec_destroy(lm_chol);
    free(full_chol);
    free(csr_values_a); free(csr_norms_a);
    free(dense_fw_scaled); free(dense_norms);
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
  ctx->lm_chol = lm_chol;
  ctx->full_chol = full_chol;

  uint64_t nc = chol_ids->n;
  int eig_raw_idx;
  int raw_codes_idx;

  lua_getfield(L, 1, "output");
  tk_dvec_t *out_buf = tk_dvec_peekopt(L, -1);
  int out_buf_idx = out_buf ? lua_gettop(L) : 0;
  if (!out_buf) lua_pop(L, 1);

  if (cholesky_mode) {

    d = m;
    tk_dvec_t *eig_raw = tk_dvec_create(L, d, 0, 0);
    eig_raw->n = d;
    for (uint64_t j = 0; j < d; j++) eig_raw->a[j] = 1.0;
    eig_raw_idx = lua_gettop(L);

    tk_dvec_t *ccodes;
    if (out_buf) {
      tk_dvec_ensure(out_buf, nc * d);
      out_buf->n = nc * d;
      memcpy(out_buf->a, full_chol, nc * d * sizeof(double));
      free(full_chol);
      ctx->full_chol = NULL;
      full_chol = out_buf->a;
      ccodes = out_buf;
      lua_pushvalue(L, out_buf_idx);
    } else {
      ccodes = tk_dvec_create(L, 1, 0, 0);
      free(ccodes->a);
      ccodes->a = full_chol;
      ccodes->n = nc * d;
      ccodes->m = nc * d;
      ctx->full_chol = NULL;
    }
    raw_codes_idx = lua_gettop(L);

    int64_t im = (int64_t)m;
    double inv_n = 1.0 / (double)nc;
    double *raw_mean = (double *)malloc(m * sizeof(double));
    double *raw_istd = (double *)malloc(m * sizeof(double));
    if (!raw_mean || !raw_istd) {
      free(raw_mean); free(raw_istd);
      return luaL_error(L, "encode: out of memory");
    }
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < im; j++) {
      double s = 0.0, ss = 0.0;
      for (uint64_t i = 0; i < nc; i++) {
        double v = full_chol[i * m + (uint64_t)j];
        s += v;
        ss += v * v;
      }
      raw_mean[j] = s * inv_n;
      double var = ss * inv_n - raw_mean[j] * raw_mean[j];
      raw_istd[j] = var > 1e-12 ? 1.0 / sqrt(var) : 1.0;
    }
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < nc; i++) {
      double *row = full_chol + i * m;
      for (int64_t j = 0; j < im; j++)
        row[j] = (row[j] - raw_mean[j]) * raw_istd[j];
    }
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < nc; i++) {
      double *row = full_chol + i * m;
      double nrm = cblas_dnrm2((int)m, row, 1);
      if (nrm > 1e-12) cblas_dscal((int)m, 1.0 / nrm, row, 1);
    }

    ctx->adjustment = raw_mean;
    ctx->inv_sqrt_eig = raw_istd;

    ctx->projection = (double *)calloc(m * m, sizeof(double));
    for (uint64_t i = 0; i < m; i++) ctx->projection[i * m + i] = 1.0;
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
      (int)m, (int)m, 1.0, lm_chol->a, (int)m, ctx->projection, (int)m);
    tk_dvec_destroy(lm_chol); ctx->lm_chol = NULL;

  } else {

  tk_dvec_t *cmeans = tk_dvec_create(NULL, m, 0, 0);
  cmeans->n = m;
  ctx->cmeans = cmeans;
  #pragma omp parallel for schedule(static)
  for (uint64_t j = 0; j < m; j++) {
    double s = 0.0;
    for (uint64_t i = 0; i < nc; i++) s += full_chol[i * m + j];
    cmeans->a[j] = s / (double)nc;
  }

  tk_dvec_t *gram = tk_dvec_create(NULL, m * m, 0, 0);
  gram->n = m * m;
  ctx->gram = gram;
  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
    (int)m, (int)m, 1.0, lm_chol->a, (int)m, 0.0, gram->a, (int)m);
  {
    double *lm_colsum = (double *)malloc(m * sizeof(double));
    for (uint64_t j = 0; j < m; j++) {
      double s = 0.0;
      for (uint64_t i = 0; i < m; i++) s += lm_chol->a[i * m + j];
      lm_colsum[j] = s;
    }
    cblas_dsyr2(CblasRowMajor, CblasUpper, (int)m, -1.0,
      lm_colsum, 1, cmeans->a, 1, gram->a, (int)m);
    cblas_dsyr(CblasRowMajor, CblasUpper, (int)m, (double)m,
      cmeans->a, 1, gram->a, (int)m);
    free(lm_colsum);
  }

  tk_dvec_t *eig_raw = tk_dvec_create(L, d, 0, 0);
  eig_raw->n = d;
  double *ev_raw = malloc(m * m * sizeof(double));
  int *isuppz = malloc(2 * m * sizeof(int));
  if (!ev_raw || !isuppz) {
    free(ev_raw); free(isuppz);
    return luaL_error(L, "encode: out of memory");
  }
  int n_eig = 0;
  int info = LAPACKE_dsyevr(LAPACK_COL_MAJOR, 'V', 'I', 'U',
    (int)m, gram->a, (int)m, 0.0, 0.0, (int)(m - d + 1), (int)m,
    0.0, &n_eig, eig_raw->a, ev_raw, (int)m, isuppz);
  free(isuppz);
  tk_dvec_destroy(gram); ctx->gram = NULL;
  if (info != 0) {
    free(ev_raw);
    return luaL_error(L, "encode: dsyevr info=%d", info);
  }

  for (uint64_t i = 0; i < d / 2; i++) {
    double tmp = eig_raw->a[i];
    eig_raw->a[i] = eig_raw->a[d - 1 - i];
    eig_raw->a[d - 1 - i] = tmp;
  }
  if (d > 0 && eig_raw->a[0] > 0.0) {
    double eig_floor = eig_raw->a[0] * 1e-10;
    uint64_t d_safe = d;
    for (uint64_t j = 0; j < d; j++) {
      if (eig_raw->a[j] <= eig_floor) {
        d_safe = j;
        break;
      }
    }
    if (d_safe == 0) d_safe = 1;
    if (d_safe < d) {
      d = d_safe;
      eig_raw->n = d;
    }
  }
  eig_raw_idx = lua_gettop(L);

  tk_dvec_t *eigvecs = tk_dvec_create(NULL, m * d, 0, 0);
  eigvecs->n = m * d;
  ctx->eigvecs = eigvecs;

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < m; i++)
    for (uint64_t k = 0; k < d; k++)
      eigvecs->a[i * d + k] = ev_raw[(d - 1 - k) * m + i];
  free(ev_raw);

  tk_dvec_t *ccodes;
  if (out_buf) {
    tk_dvec_ensure(out_buf, nc * d);
    out_buf->n = nc * d;
    ccodes = out_buf;
    lua_pushvalue(L, out_buf_idx);
  } else {
    ccodes = tk_dvec_create(L, nc * d, 0, 0);
    ccodes->n = nc * d;
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)nc, (int)d, (int)m, 1.0, full_chol, (int)m, eigvecs->a, (int)d,
    0.0, ccodes->a, (int)d);
  free(full_chol); ctx->full_chol = NULL;

  ctx->adjustment = (double *)malloc(d * sizeof(double));
  cblas_dgemv(CblasRowMajor, CblasTrans,
    (int)m, (int)d, 1.0, eigvecs->a, (int)d, cmeans->a, 1, 0.0, ctx->adjustment, 1);
  tk_dvec_destroy(cmeans); ctx->cmeans = NULL;

  ctx->inv_sqrt_eig = (double *)malloc(d * sizeof(double));
  for (uint64_t j = 0; j < d; j++)
    ctx->inv_sqrt_eig[j] = 1.0 / sqrt(eig_raw->a[j]);

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < nc; i++) {
    double *r = ccodes->a + i * d;
    for (uint64_t j = 0; j < d; j++) {
      r[j] -= ctx->adjustment[j];
      r[j] *= ctx->inv_sqrt_eig[j];
    }
    double nrm = cblas_dnrm2((int)d, r, 1);
    if (nrm > 1e-12) cblas_dscal((int)d, 1.0 / nrm, r, 1);
  }

  ctx->projection = (double *)malloc(m * d * sizeof(double));
  memcpy(ctx->projection, eigvecs->a, m * d * sizeof(double));
  tk_dvec_destroy(eigvecs); ctx->eigvecs = NULL;

  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
    (int)m, (int)d, 1.0, lm_chol->a, (int)m, ctx->projection, (int)d);
  tk_dvec_destroy(lm_chol); ctx->lm_chol = NULL;

  } /* end else (eigendecomposition path) */

  if (!cholesky_mode) raw_codes_idx = lua_gettop(L);

  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->projection = ctx->projection; ctx->projection = NULL;
  enc->adjustment = ctx->adjustment; ctx->adjustment = NULL;
  enc->inv_sqrt_eig = ctx->inv_sqrt_eig; ctx->inv_sqrt_eig = NULL;
  enc->m = m;
  enc->d = d;
  enc->decay = decay;
  enc->trace_ratio = trace_ratio;
  enc->destroyed = false;

  enc->lm_sids = NULL;
  enc->lm_vecs = NULL;
  enc->lm_bits = NULL;
  enc->lm_csr_offsets = NULL;
  enc->lm_csr_tokens = NULL;
  enc->lm_csr_values = NULL;
  enc->lm_csr_norms = NULL;
  enc->lm_csc_offsets = NULL;
  enc->lm_csc_rows = NULL;
  enc->lm_csc_values = NULL;
  enc->lm_csc_positions = NULL;
  enc->pos_window = 0;
  enc->row_length = 0;
  enc->feature_weights = NULL;
  enc->fw_len = 0;
  enc->lm_dense_norms = NULL;
  enc->itq_rotation = NULL;
  enc->itq_means = NULL;
  enc->binarize_dims = 0;
  enc->binarize_mode = TK_BINARIZE_NONE;
  enc->d_input = 0;
  enc->bits_d = 0;
  enc->csr_n_tokens = 0;
  enc->kernel = kernel;
  if (has_inv) {
    ctx->lm_sids = (int64_t *)malloc(m * sizeof(int64_t));
    for (uint64_t j = 0; j < m; j++)
      ctx->lm_sids[j] = tk_inv_uid_sid(feat_inv, lm_ids->a[j], TK_INV_FIND);
    enc->lm_sids = ctx->lm_sids; ctx->lm_sids = NULL;
    enc->dense_mode = 0;
  } else if (has_csr) {
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
    enc->lm_csc_positions = (pos_window > 0)
      ? (int64_t *)malloc(lm_total * sizeof(int64_t)) : NULL;
    int64_t *csc_pos = (int64_t *)malloc((csr_n_tokens + 1) * sizeof(int64_t));
    memcpy(csc_pos, enc->lm_csc_offsets, (csr_n_tokens + 1) * sizeof(int64_t));
    for (uint64_t j = 0; j < m; j++) {
      for (int64_t a = enc->lm_csr_offsets[j]; a < enc->lm_csr_offsets[j + 1]; a++) {
        int64_t tok = enc->lm_csr_tokens[a];
        int64_t p = csc_pos[tok]++;
        enc->lm_csc_rows[p] = (int64_t)j;
        enc->lm_csc_values[p] = enc->lm_csr_values[a];
        if (enc->lm_csc_positions)
          enc->lm_csc_positions[p] = a - enc->lm_csr_offsets[j];
      }
    }
    free(csc_pos);
    enc->csr_n_tokens = csr_n_tokens;
    enc->pos_window = pos_window;
    enc->row_length = row_length;
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
    if (kernel != TK_SPECTRAL_LINEAR) {
      enc->lm_dense_norms = (double *)malloc(m * sizeof(double));
      for (uint64_t j = 0; j < m; j++)
        enc->lm_dense_norms[j] = cblas_dnrm2((int)dense_d_input,
          enc->lm_vecs + j * (uint64_t)dense_d_input, 1);
    }
    if (fw_dv) {
      enc->feature_weights = (double *)malloc(fw_dv->n * sizeof(double));
      memcpy(enc->feature_weights, fw_dv->a, fw_dv->n * sizeof(double));
      enc->fw_len = fw_dv->n;
    }
    free(dense_fw_scaled);
    free(dense_norms);
  }

  lua_getfield(L, 1, "binarize");
  const char *binarize_str = lua_isstring(L, -1) ? lua_tostring(L, -1) : NULL;
  lua_pop(L, 1);
  tk_binarize_mode_t binarize_mode = TK_BINARIZE_NONE;
  if (binarize_str) {
    if (strcmp(binarize_str, "sign") == 0) binarize_mode = TK_BINARIZE_SIGN;
    else if (strcmp(binarize_str, "itq") == 0) binarize_mode = TK_BINARIZE_ITQ;
    else return luaL_error(L, "binarize must be 'sign' or 'itq'");
  }
  if (binarize_mode != TK_BINARIZE_NONE) {
    uint64_t bin_dims = tk_lua_foptunsigned(L, 1, "encode", "binarize_dims", d);
    if (bin_dims > d) bin_dims = d;
    enc->binarize_dims = bin_dims;
    enc->binarize_mode = binarize_mode;
    if (binarize_mode == TK_BINARIZE_ITQ) {
      uint64_t itq_iters = tk_lua_foptunsigned(L, 1, "encode", "itq_iterations", 50);
      tk_dvec_t *raw_dv = (tk_dvec_t *)lua_touserdata(L, raw_codes_idx);
      double *means = (double *)malloc(bin_dims * sizeof(double));
      double *data = (double *)malloc(nc * bin_dims * sizeof(double));
      #pragma omp parallel for schedule(static)
      for (uint64_t j = 0; j < bin_dims; j++) {
        double s = 0.0;
        for (uint64_t i = 0; i < nc; i++)
          s += raw_dv->a[i * d + j];
        means[j] = s / (double)nc;
        for (uint64_t i = 0; i < nc; i++)
          data[i * bin_dims + j] = raw_dv->a[i * d + j] - means[j];
      }
      double *R = (double *)malloc(bin_dims * bin_dims * sizeof(double));
      for (uint64_t i = 0; i < bin_dims; i++)
        for (uint64_t j = 0; j < bin_dims; j++)
          R[i * bin_dims + j] = (i == j) ? 1.0 : 0.0;
      double *PB = (double *)malloc(nc * bin_dims * sizeof(double));
      double *BtV = (double *)malloc(bin_dims * bin_dims * sizeof(double));
      double *BtVtBtV = (double *)malloc(bin_dims * bin_dims * sizeof(double));
      double *tmp = (double *)malloc(bin_dims * bin_dims * sizeof(double));
      for (uint64_t iter = 0; iter < itq_iters; iter++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          (int)nc, (int)bin_dims, (int)bin_dims, 1.0, data, (int)bin_dims,
          R, (int)bin_dims, 0.0, PB, (int)bin_dims);
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < nc * bin_dims; i++)
          PB[i] = PB[i] > 0.0 ? 1.0 : -1.0;
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          (int)bin_dims, (int)bin_dims, (int)nc, 1.0, PB, (int)bin_dims,
          data, (int)bin_dims, 0.0, BtV, (int)bin_dims);
        double nrm = cblas_dnrm2((int)(bin_dims * bin_dims), BtV, 1);
        if (nrm < 1e-15) break;
        cblas_dscal((int)(bin_dims * bin_dims), 1.0 / nrm, BtV, 1);
        for (int ns = 0; ns < 15; ns++) {
          cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            (int)bin_dims, (int)bin_dims, (int)bin_dims, 1.0, BtV, (int)bin_dims,
            BtV, (int)bin_dims, 0.0, BtVtBtV, (int)bin_dims);
          memcpy(tmp, BtV, bin_dims * bin_dims * sizeof(double));
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            (int)bin_dims, (int)bin_dims, (int)bin_dims, -0.5, tmp, (int)bin_dims,
            BtVtBtV, (int)bin_dims, 1.5, BtV, (int)bin_dims);
        }
        memcpy(R, BtV, bin_dims * bin_dims * sizeof(double));
      }
      free(PB); free(BtV); free(BtVtBtV); free(tmp); free(data);
      enc->itq_rotation = R;
      enc->itq_means = means;
    }
  }

  lua_newtable(L);
  if (has_inv) {
    lua_pushvalue(L, feat_inv_idx);
    lua_setfield(L, -2, "inv");
  }
  lua_pushvalue(L, lm_ids_idx);
  lua_setfield(L, -2, "landmark_ids");
  lua_setfenv(L, enc_idx);

  lua_pushvalue(L, raw_codes_idx);
  lua_pushvalue(L, chol_ids_idx);
  lua_pushvalue(L, enc_idx);
  lua_pushvalue(L, eig_raw_idx);
  return 4;
}

static inline int tk_nystrom_encoder_load_lua (lua_State *L) {
  lua_settop(L, 3);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  int inv_idx = 2;
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
  if (version != 12) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported nystrom encoder version %d (expected 12)", (int)version);
  }
  uint8_t dense_mode;
  tk_lua_fread(L, &dense_mode, sizeof(uint8_t), 1, fh);
  uint8_t kernel_byte;
  tk_lua_fread(L, &kernel_byte, sizeof(uint8_t), 1, fh);
  tk_spectral_kernel_t kernel = (tk_spectral_kernel_t)kernel_byte;
  uint64_t m, d;
  double decay, trace_ratio;
  tk_lua_fread(L, &m, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &d, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &decay, sizeof(double), 1, fh);
  tk_lua_fread(L, &trace_ratio, sizeof(double), 1, fh);
  double *projection = (double *)malloc(m * d * sizeof(double));
  double *adjustment = (double *)malloc(d * sizeof(double));
  double *inv_sqrt_eig = (double *)malloc(d * sizeof(double));
  if (!projection || !adjustment || !inv_sqrt_eig) {
    free(projection); free(adjustment); free(inv_sqrt_eig);
    tk_lua_fclose(L, fh);
    return luaL_error(L, "nystrom load: out of memory");
  }
  tk_lua_fread(L, projection, sizeof(double), m * d, fh);
  tk_lua_fread(L, adjustment, sizeof(double), d, fh);
  tk_lua_fread(L, inv_sqrt_eig, sizeof(double), d, fh);
  uint64_t fw_len;
  double *feature_weights = NULL;
  tk_lua_fread(L, &fw_len, sizeof(uint64_t), 1, fh);
  if (fw_len > 0) {
    feature_weights = (double *)malloc(fw_len * sizeof(double));
    tk_lua_fread(L, feature_weights, sizeof(double), fw_len, fh);
  }
  int64_t *lm_sids = NULL;
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
  int64_t pos_window = 0;
  int64_t row_length = 0;
  if (dense_mode == 2) {
    tk_lua_fread(L, &bits_d, sizeof(uint64_t), 1, fh);
    uint64_t row_bytes = TK_CVEC_BITS_BYTES(bits_d);
    lm_bits = (uint8_t *)malloc(m * row_bytes);
    if (!lm_bits) {
      free(projection); free(adjustment); free(inv_sqrt_eig); free(feature_weights);
      tk_lua_fclose(L, fh);
      return luaL_error(L, "nystrom load: out of memory");
    }
    tk_lua_fread(L, lm_bits, 1, m * row_bytes, fh);
  } else if (dense_mode == 1) {
    tk_lua_fread(L, &d_input, sizeof(int64_t), 1, fh);
    lm_vecs = (double *)malloc(m * (uint64_t)d_input * sizeof(double));
    if (!lm_vecs) {
      free(projection); free(adjustment); free(inv_sqrt_eig); free(feature_weights);
      tk_lua_fclose(L, fh);
      return luaL_error(L, "nystrom load: out of memory");
    }
    tk_lua_fread(L, lm_vecs, sizeof(double), m * (uint64_t)d_input, fh);
    uint8_t has_norms;
    tk_lua_fread(L, &has_norms, sizeof(uint8_t), 1, fh);
    if (has_norms) {
      lm_dense_norms = (double *)malloc(m * sizeof(double));
      tk_lua_fread(L, lm_dense_norms, sizeof(double), m, fh);
    } else if (kernel != TK_SPECTRAL_LINEAR) {
      lm_dense_norms = (double *)malloc(m * sizeof(double));
      for (uint64_t j = 0; j < m; j++)
        lm_dense_norms[j] = cblas_dnrm2((int)d_input, lm_vecs + j * (uint64_t)d_input, 1);
    }
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
    tk_lua_fread(L, &pos_window, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, &row_length, sizeof(int64_t), 1, fh);
  } else {
    lm_sids = (int64_t *)malloc(m * sizeof(int64_t));
    if (!lm_sids) {
      free(projection); free(adjustment); free(inv_sqrt_eig); free(feature_weights);
      tk_lua_fclose(L, fh);
      return luaL_error(L, "nystrom load: out of memory");
    }
    tk_lua_fread(L, lm_sids, sizeof(int64_t), m, fh);
  }
  tk_ivec_load(L, fh);
  int lm_ids_idx = lua_gettop(L);
  double *itq_rotation = NULL;
  double *itq_means = NULL;
  uint64_t binarize_dims = 0;
  tk_binarize_mode_t binarize_mode = TK_BINARIZE_NONE;
  uint8_t bin_mode;
  tk_lua_fread(L, &bin_mode, sizeof(uint8_t), 1, fh);
  binarize_mode = (tk_binarize_mode_t)bin_mode;
  if (binarize_mode != TK_BINARIZE_NONE) {
    tk_lua_fread(L, &binarize_dims, sizeof(uint64_t), 1, fh);
    if (binarize_mode == TK_BINARIZE_ITQ) {
      itq_rotation = (double *)malloc(binarize_dims * binarize_dims * sizeof(double));
      tk_lua_fread(L, itq_rotation, sizeof(double), binarize_dims * binarize_dims, fh);
      itq_means = (double *)malloc(binarize_dims * sizeof(double));
      tk_lua_fread(L, itq_means, sizeof(double), binarize_dims, fh);
    }
  }
  tk_lua_fclose(L, fh);
  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->projection = projection;
  enc->adjustment = adjustment;
  enc->inv_sqrt_eig = inv_sqrt_eig;
  enc->lm_sids = lm_sids;
  enc->lm_vecs = lm_vecs;
  enc->lm_bits = lm_bits;
  enc->lm_csr_offsets = lm_csr_offsets;
  enc->lm_csr_tokens = lm_csr_tokens;
  enc->lm_csr_values = lm_csr_values;
  enc->lm_csr_norms = lm_csr_norms;
  enc->lm_csc_offsets = NULL;
  enc->lm_csc_rows = NULL;
  enc->lm_csc_values = NULL;
  enc->lm_csc_positions = NULL;
  enc->pos_window = pos_window;
  enc->row_length = row_length;
  enc->feature_weights = feature_weights;
  enc->fw_len = fw_len;
  enc->lm_dense_norms = lm_dense_norms;
  enc->itq_rotation = itq_rotation;
  enc->itq_means = itq_means;
  enc->binarize_dims = binarize_dims;
  enc->binarize_mode = binarize_mode;
  enc->m = m;
  enc->d = d;
  enc->d_input = d_input;
  enc->bits_d = bits_d;
  enc->csr_n_tokens = csr_n_tokens;
  enc->decay = decay;
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
    enc->lm_csc_positions = (pos_window > 0)
      ? (int64_t *)malloc(lm_total * sizeof(int64_t)) : NULL;
    int64_t *csc_pos = (int64_t *)malloc((csr_n_tokens + 1) * sizeof(int64_t));
    memcpy(csc_pos, enc->lm_csc_offsets, (csr_n_tokens + 1) * sizeof(int64_t));
    for (uint64_t j = 0; j < m; j++) {
      for (int64_t a = lm_csr_offsets[j]; a < lm_csr_offsets[j + 1]; a++) {
        int64_t tok = lm_csr_tokens[a];
        int64_t p = csc_pos[tok]++;
        enc->lm_csc_rows[p] = (int64_t)j;
        enc->lm_csc_values[p] = lm_csr_values[a];
        if (enc->lm_csc_positions)
          enc->lm_csc_positions[p] = a - lm_csr_offsets[j];
      }
    }
    free(csc_pos);
  }
  lua_newtable(L);
  if (dense_mode == 0) {
    tk_inv_peek(L, inv_idx);
    lua_pushvalue(L, inv_idx);
    lua_setfield(L, -2, "inv");
  }
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
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  lua_pop(L, 1);
  double *X = codes->a;

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

  double inv_n = 1.0 / (double)n;

  tk_dvec_t *cm = tk_dvec_create(L, (uint64_t)d, 0, 0); cm->n = (uint64_t)d;
  int cm_idx = lua_gettop(L);
  #pragma omp parallel for schedule(static)
  for (int64_t j = 0; j < d; j++) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += X[i * d + j];
    cm->a[j] = s * inv_n;
  }

  tk_dvec_t *xtx = tk_dvec_create(L, (uint64_t)(d * d), 0, 0); xtx->n = (uint64_t)(d * d);
  int xtx_idx = lua_gettop(L);
  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
    (int)d, (int)n, inv_n, X, (int)d, 0.0, xtx->a, (int)d);
  for (int64_t i = 0; i < d; i++)
    for (int64_t j = i + 1; j < d; j++)
      xtx->a[j * d + i] = xtx->a[i * d + j];

  tk_dvec_t *xty = tk_dvec_create(L, (uint64_t)(d * nl), 0, 0);
  xty->n = (uint64_t)(d * nl);
  int xty_idx = lua_gettop(L);
  memset(xty->a, 0, (uint64_t)(d * nl) * sizeof(double));

  tk_dvec_t *ym = tk_dvec_create(L, (uint64_t)nl, 0, 0);
  ym->n = (uint64_t)nl;
  int ym_idx = lua_gettop(L);
  memset(ym->a, 0, (uint64_t)nl * sizeof(double));

  int64_t batch = 1024;
  double *Y_batch = (double *)malloc((uint64_t)batch * (uint64_t)nl * sizeof(double));
  if (!Y_batch) return luaL_error(L, "gram: out of memory");
  for (int64_t base = 0; base < n; base += batch) {
    int64_t bs = (base + batch <= n) ? batch : n - base;
    memset(Y_batch, 0, (uint64_t)(bs * nl) * sizeof(double));
    if (has_labels) {
      for (int64_t i = 0; i < bs; i++) {
        int64_t sid = base + i;
        for (int64_t j = lbl_off->a[sid]; j < lbl_off->a[sid + 1]; j++)
          Y_batch[i * nl + lbl_nbr->a[j]] = 1.0;
      }
    } else {
      memcpy(Y_batch, targets->a + (uint64_t)base * (uint64_t)nl,
             (uint64_t)(bs * nl) * sizeof(double));
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)d, (int)nl, (int)bs, inv_n,
      X + (uint64_t)base * (uint64_t)d, (int)d,
      Y_batch, (int)nl, 1.0, xty->a, (int)nl);
    for (int64_t i = 0; i < bs; i++)
      for (int64_t l = 0; l < nl; l++)
        ym->a[l] += Y_batch[i * nl + l] * inv_n;
  }
  free(Y_batch);

  int lc_idx = 0;
  if (has_labels) {
    tk_dvec_t *lc = tk_dvec_create(L, (uint64_t)nl, 0, 0);
    lc->n = (uint64_t)nl;
    lc_idx = lua_gettop(L);
    for (int64_t l = 0; l < nl; l++)
      lc->a[l] = ym->a[l] * (double)n;
  }

  tk_dvec_t *pm = tk_dvec_create(L, (uint64_t)d, 0, 0); pm->n = (uint64_t)d;
  int pm_idx = lua_gettop(L);
  memcpy(pm->a, cm->a, (uint64_t)d * sizeof(double));

  tk_dvec_t *pi = tk_dvec_create(L, (uint64_t)d, 0, 0); pi->n = (uint64_t)d;
  int pi_idx = lua_gettop(L);
  for (int64_t j = 0; j < d; j++) {
    double var = xtx->a[j * d + j] - cm->a[j] * cm->a[j];
    pi->a[j] = var > 1e-12 ? 1.0 / sqrt(var) : 1.0;
  }

  lua_pushvalue(L, xtx_idx);
  lua_pushvalue(L, xty_idx);
  lua_pushvalue(L, cm_idx);
  lua_pushvalue(L, ym_idx);
  if (lc_idx) lua_pushvalue(L, lc_idx);
  else lua_pushnil(L);
  lua_pushvalue(L, pm_idx);
  lua_pushvalue(L, pi_idx);
  return 7;
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
