#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/iumap.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/dvec.h>

#define TK_SFBS_ENCODER_MT "tk_sfbs_encoder_t"

typedef struct {
  tk_ivec_t *bit_dims;
  tk_dvec_t *bit_thresholds;
  uint64_t n_dims;
  uint64_t n_bits;
  bool destroyed;
} tk_sfbs_encoder_t;

static inline tk_sfbs_encoder_t *tk_sfbs_encoder_peek(lua_State *L, int i) {
  return (tk_sfbs_encoder_t *)luaL_checkudata(L, i, TK_SFBS_ENCODER_MT);
}

static inline int tk_sfbs_encoder_gc(lua_State *L) {
  tk_sfbs_encoder_t *enc = tk_sfbs_encoder_peek(L, 1);
  enc->bit_dims = NULL;
  enc->bit_thresholds = NULL;
  enc->destroyed = true;
  return 0;
}

static inline void tk_sfbs_report(
  lua_State *L, int i_each,
  int64_t a, int64_t b, double c, double d, double e, const char *label
) {
  if (i_each < 0) return;
  lua_pushvalue(L, i_each);
  lua_pushinteger(L, a);
  lua_pushinteger(L, b);
  lua_pushnumber(L, c);
  lua_pushnumber(L, d);
  lua_pushnumber(L, e);
  lua_pushstring(L, label);
  lua_call(L, 6, 0);
}

static inline void tk_sfbs_encode_c(
  tk_sfbs_encoder_t *enc,
  double *raw_codes, uint64_t n_samples,
  uint8_t *out, uint64_t n_bytes
) {
  uint64_t n_dims = enc->n_dims;
  uint64_t n_bits = enc->n_bits;
  int64_t *dims = enc->bit_dims->a;
  double *thresholds = enc->bit_thresholds->a;
  memset(out, 0, n_samples * n_bytes);
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    double *x = raw_codes + i * n_dims;
    uint8_t *dest = out + i * n_bytes;
    for (uint64_t k = 0; k < n_bits; k++) {
      if (x[dims[k]] > thresholds[k])
        dest[TK_CVEC_BITS_BYTE(k)] |= (1 << TK_CVEC_BITS_BIT(k));
    }
  }
}

static inline int tk_sfbs_encode_lua(lua_State *L) {
  tk_sfbs_encoder_t *enc = tk_sfbs_encoder_peek(L, 1);
  uint64_t n_bits = enc->n_bits;
  uint64_t n_bytes = TK_CVEC_BITS_BYTES(n_bits);
  tk_dvec_t *dvec_in = tk_dvec_peekopt(L, 2);
  if (dvec_in) {
    uint64_t n_dims = enc->n_dims;
    uint64_t n_samples = dvec_in->n / n_dims;
    uint64_t out_size = n_samples * n_bytes;
    tk_cvec_t *out = tk_cvec_peekopt(L, 3);
    if (out) {
      tk_cvec_ensure(out, out_size);
      out->n = out_size;
      lua_pushvalue(L, 3);
    } else {
      out = tk_cvec_create(L, out_size, NULL, NULL);
      out->n = out_size;
    }
    tk_sfbs_encode_c(enc, dvec_in->a, n_samples, (uint8_t *)out->a, n_bytes);
    return 1;
  }
  tk_cvec_t *cvec_in = tk_cvec_peek(L, 2, "codes (dvec or cvec)");
  uint64_t src_bits = enc->n_dims;
  uint64_t src_bytes = TK_CVEC_BITS_BYTES(src_bits);
  uint64_t n_samples = cvec_in->n / src_bytes;
  uint64_t out_size = n_samples * n_bytes;
  tk_cvec_t *out = tk_cvec_peekopt(L, 3);
  if (out) {
    tk_cvec_ensure(out, out_size);
    out->n = out_size;
    lua_pushvalue(L, 3);
  } else {
    out = tk_cvec_create(L, out_size, NULL, NULL);
    out->n = out_size;
  }
  memset(out->a, 0, out->n);
  int64_t *dims = enc->bit_dims->a;
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    uint8_t *src = (uint8_t *)cvec_in->a + i * src_bytes;
    uint8_t *dest = (uint8_t *)out->a + i * n_bytes;
    for (uint64_t k = 0; k < n_bits; k++) {
      uint64_t sb = (uint64_t)dims[k];
      if ((src[sb >> 3] >> (sb & 7)) & 1)
        dest[TK_CVEC_BITS_BYTE(k)] |= (1 << TK_CVEC_BITS_BIT(k));
    }
  }
  return 1;
}

static inline int tk_sfbs_n_bits_lua(lua_State *L) {
  tk_sfbs_encoder_t *enc = tk_sfbs_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->n_bits);
  return 1;
}

static inline int tk_sfbs_n_dims_lua(lua_State *L) {
  tk_sfbs_encoder_t *enc = tk_sfbs_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->n_dims);
  return 1;
}

static inline int tk_sfbs_dims_lua(lua_State *L) {
  tk_sfbs_encoder_peek(L, 1);
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "dims");
  return 1;
}

static inline int tk_sfbs_thresholds_lua(lua_State *L) {
  tk_sfbs_encoder_peek(L, 1);
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "thresholds");
  return 1;
}

static inline int tk_sfbs_used_dims_lua(lua_State *L) {
  tk_sfbs_encoder_t *enc = tk_sfbs_encoder_peek(L, 1);
  tk_ivec_t *out = tk_ivec_create(L, enc->n_bits, 0, 0);
  out->n = 0;
  for (uint64_t i = 0; i < enc->n_bits; i++)
    tk_ivec_push(out, enc->bit_dims->a[i]);
  out->n = tk_ivec_uasc(out, 0, out->n);
  return 1;
}

static inline int tk_sfbs_restrict_lua(lua_State *L) {
  tk_sfbs_encoder_t *enc = tk_sfbs_encoder_peek(L, 1);
  tk_ivec_t *kept = tk_ivec_peek(L, 2, "kept_dims");
  tk_iumap_t *dim_map = tk_iumap_create(NULL, 0);
  for (uint64_t i = 0; i < kept->n; i++) {
    int kha;
    khint_t khi = tk_iumap_put(dim_map, kept->a[i], &kha);
    tk_iumap_setval(dim_map, khi, (int64_t)i);
  }
  uint64_t w = 0;
  for (uint64_t r = 0; r < enc->n_bits; r++) {
    khint_t khi = tk_iumap_get(dim_map, enc->bit_dims->a[r]);
    if (khi != tk_iumap_end(dim_map)) {
      enc->bit_dims->a[w] = tk_iumap_val(dim_map, khi);
      enc->bit_thresholds->a[w] = enc->bit_thresholds->a[r];
      w++;
    }
  }
  enc->n_bits = w;
  enc->n_dims = kept->n;
  tk_iumap_destroy(dim_map);
  return 0;
}

static inline int tk_sfbs_restrict_bits_lua(lua_State *L) {
  tk_sfbs_encoder_t *enc = tk_sfbs_encoder_peek(L, 1);
  tk_ivec_t *kept_bits = tk_ivec_peek(L, 2, "kept_bits");
  uint64_t n_new = kept_bits->n;
  int64_t *dims = enc->bit_dims->a;
  double *thresholds = enc->bit_thresholds->a;
  for (uint64_t i = 0; i < n_new; i++) {
    uint64_t src = (uint64_t)kept_bits->a[i];
    dims[i] = dims[src];
    thresholds[i] = thresholds[src];
  }
  enc->n_bits = n_new;
  return 0;
}

static inline int tk_sfbs_encoder_persist_lua(lua_State *L);

static luaL_Reg tk_sfbs_encoder_mt_fns[] = {
  { "encode", tk_sfbs_encode_lua },
  { "restrict", tk_sfbs_restrict_lua },
  { "restrict_bits", tk_sfbs_restrict_bits_lua },
  { "n_bits", tk_sfbs_n_bits_lua },
  { "n_dims", tk_sfbs_n_dims_lua },
  { "dims", tk_sfbs_dims_lua },
  { "thresholds", tk_sfbs_thresholds_lua },
  { "used_dims", tk_sfbs_used_dims_lua },
  { "persist", tk_sfbs_encoder_persist_lua },
  { NULL, NULL }
};

static inline double tk_sfbs_dcg(
  uint64_t *bc, double *bw, double *pd, uint64_t n_buckets
) {
  double dcg = 0.0;
  uint64_t pos = 0;
  for (uint64_t d = 0; d < n_buckets; d++) {
    if (bc[d] > 0)
      dcg += bw[d] * (pd[pos + bc[d]] - pd[pos]) / (double)bc[d];
    pos += bc[d];
  }
  return dcg;
}

static inline void tk_sfbs_commit(
  uint8_t *bin, int64_t *pra, int64_t *prb,
  uint16_t *distances, uint64_t n_pairs, int add
) {
  for (uint64_t j = 0; j < n_pairs; j++) {
    uint8_t ba = (bin[pra[j] >> 3] >> (pra[j] & 7)) & 1;
    uint8_t bb = (bin[prb[j] >> 3] >> (prb[j] & 7)) & 1;
    if (add) distances[j] += ba ^ bb;
    else distances[j] -= ba ^ bb;
  }
}

static inline void tk_sfbs_commit_xor(
  uint8_t *xor_codes, uint64_t chunks, uint64_t bit,
  uint16_t *distances, uint64_t n_pairs, int add
) {
  for (uint64_t j = 0; j < n_pairs; j++) {
    uint8_t xbit = (xor_codes[j * chunks + (bit >> 3)] >> (bit & 7)) & 1;
    if (add) distances[j] += xbit;
    else distances[j] -= xbit;
  }
}

static inline void tk_sfbs_spawn_children(
  double *raw_codes, uint64_t n_samples, uint64_t n_dims,
  uint64_t dim, double parent_thresh,
  uint64_t *cand_dims, double *cand_thresholds, uint8_t **cand_bins,
  uint8_t *cand_selected, uint64_t *n_candidates, uint64_t max_candidates,
  uint64_t n_sample_bytes
) {
  tk_dvec_t *vals = tk_dvec_create(NULL, n_samples, 0, 0);
  uint64_t n_lo = 0, n_hi = 0;
  for (uint64_t i = 0; i < n_samples; i++) {
    double v = raw_codes[i * n_dims + dim];
    if (v <= parent_thresh) vals->a[n_lo++] = v;
  }
  if (n_lo > 1 && *n_candidates < max_candidates) {
    tk_dvec_asc(vals, 0, n_lo);
    double lo_thresh = vals->a[n_lo / 2];
    uint64_t ci = (*n_candidates)++;
    cand_dims[ci] = dim;
    cand_thresholds[ci] = lo_thresh;
    cand_bins[ci] = (uint8_t *)calloc(n_sample_bytes, 1);
    cand_selected[ci] = 0;
    for (uint64_t i = 0; i < n_samples; i++)
      if (raw_codes[i * n_dims + dim] > lo_thresh)
        cand_bins[ci][i >> 3] |= (uint8_t)(1 << (i & 7));
  }
  n_hi = 0;
  for (uint64_t i = 0; i < n_samples; i++) {
    double v = raw_codes[i * n_dims + dim];
    if (v > parent_thresh) vals->a[n_hi++] = v;
  }
  if (n_hi > 1 && *n_candidates < max_candidates) {
    tk_dvec_asc(vals, 0, n_hi);
    double hi_thresh = vals->a[n_hi / 2];
    uint64_t ci = (*n_candidates)++;
    cand_dims[ci] = dim;
    cand_thresholds[ci] = hi_thresh;
    cand_bins[ci] = (uint8_t *)calloc(n_sample_bytes, 1);
    cand_selected[ci] = 0;
    for (uint64_t i = 0; i < n_samples; i++)
      if (raw_codes[i * n_dims + dim] > hi_thresh)
        cand_bins[ci][i >> 3] |= (uint8_t)(1 << (i & 7));
  }
  tk_dvec_destroy(vals);
}

static inline double tk_sfbs_score_node(
  uint8_t *bin, int64_t node_row,
  int64_t *prb, uint16_t *dist, double *wts,
  int64_t st, uint64_t nn, uint64_t n_buckets, int add,
  uint64_t *bc, double *bw, double *pd
) {
  uint8_t nb = (bin[node_row >> 3] >> (node_row & 7)) & 1;
  memset(bc, 0, n_buckets * sizeof(uint64_t));
  memset(bw, 0, n_buckets * sizeof(double));
  for (uint64_t i = 0; i < nn; i++) {
    int64_t nr = prb[(uint64_t)st + i];
    uint8_t xb = nb ^ ((bin[nr >> 3] >> (nr & 7)) & 1);
    uint16_t nd = add ? dist[(uint64_t)st + i] + xb : dist[(uint64_t)st + i] - xb;
    bc[nd]++;
    bw[nd] += wts[st + (int64_t)i];
  }
  return tk_sfbs_dcg(bc, bw, pd, n_buckets);
}

static uint64_t tk_sfbs_binary_select(
  lua_State *L, int i_each,
  unsigned char *codes, uint64_t n_samples, uint64_t n_bits,
  int64_t *pair_row_a, int64_t *pair_row_b,
  int64_t *offs_a, double *weights_a, double *idcg, double *prefix_discount,
  uint64_t n_nodes, uint64_t n_pairs, uint64_t max_neighbors,
  uint64_t target_bits, double tolerance,
  uint8_t *selected_out, double *out_score
) {
  uint64_t chunks = TK_CVEC_BITS_BYTES(n_bits);
  uint8_t *xor_codes = (uint8_t *)malloc(n_pairs * chunks);
  #pragma omp parallel for schedule(static)
  for (uint64_t j = 0; j < n_pairs; j++) {
    uint8_t *xa = codes + (uint64_t)pair_row_a[j] * chunks;
    uint8_t *xb = codes + (uint64_t)pair_row_b[j] * chunks;
    uint8_t *xd = xor_codes + j * chunks;
    for (uint64_t c = 0; c < chunks; c++)
      xd[c] = xa[c] ^ xb[c];
  }

  uint16_t *distances = (uint16_t *)calloc(n_pairs, sizeof(uint16_t));
  uint64_t n_buckets = target_bits + 2;
  uint64_t n_selected = 0;
  double current_score = 0.0;

  int phase = 0;
  int64_t swap_remove_bit = -1;
  double pre_swap_score = 0.0;

  while (phase < 3) {

    if (phase == 0 || phase == 2) {
      double *global_scores = (double *)calloc(n_bits, sizeof(double));
      int64_t best_bit = -1;
      double best_score = -INFINITY;

      #pragma omp parallel
      {
        double *ls = (double *)calloc(n_bits, sizeof(double));
        uint64_t *bc = (uint64_t *)calloc(n_buckets, sizeof(uint64_t));
        double *bw = (double *)calloc(n_buckets, sizeof(double));

        #pragma omp for schedule(static)
        for (uint64_t ni = 0; ni < n_nodes; ni++) {
          int64_t st = offs_a[ni];
          uint64_t nn = (uint64_t)(offs_a[ni + 1] - st);
          if (nn == 0 || idcg[ni] <= 0.0) continue;
          for (uint64_t b = 0; b < n_bits; b++) {
            if (selected_out[b]) continue;
            memset(bc, 0, n_buckets * sizeof(uint64_t));
            memset(bw, 0, n_buckets * sizeof(double));
            for (uint64_t i = 0; i < nn; i++) {
              uint64_t j = (uint64_t)st + i;
              uint8_t xbit = (xor_codes[j * chunks + (b >> 3)] >> (b & 7)) & 1;
              uint16_t d = distances[j] + xbit;
              if (d >= n_buckets) d = (uint16_t)(n_buckets - 1);
              bc[d]++;
              bw[d] += weights_a[st + (int64_t)i];
            }
            ls[b] += tk_sfbs_dcg(bc, bw, prefix_discount, n_buckets) / idcg[ni];
          }
        }

        #pragma omp critical
        for (uint64_t b = 0; b < n_bits; b++)
          global_scores[b] += ls[b];

        free(ls);
        free(bc);
        free(bw);
      }

      for (uint64_t b = 0; b < n_bits; b++) {
        if (selected_out[b]) continue;
        double sc = n_nodes > 0 ? global_scores[b] / (double)n_nodes : 0.0;
        if (sc > best_score) {
          best_score = sc;
          best_bit = (int64_t)b;
        }
      }
      free(global_scores);

      if (phase == 0) {
        bool added = false;
        if (best_bit >= 0 && n_selected < target_bits) {
          double gain = (n_selected == 0) ? best_score : (best_score - current_score);
          if (gain > tolerance) {
            selected_out[best_bit] = 1;
            n_selected++;
            tk_sfbs_commit_xor(xor_codes, chunks, (uint64_t)best_bit, distances, n_pairs, 1);
            current_score = best_score;
            added = true;
            tk_sfbs_report(L, i_each, (int64_t)best_bit, 0, 0, gain, current_score, "bit_add");
          }
        }
        if (!added) {
          if (n_selected > 0) phase = 1;
          else phase = 3;
        }
      } else {
        if (best_bit >= 0 && best_score - pre_swap_score > tolerance) {
          selected_out[best_bit] = 1;
          n_selected++;
          tk_sfbs_commit_xor(xor_codes, chunks, (uint64_t)best_bit, distances, n_pairs, 1);
          current_score = best_score;
          phase = 0;
          tk_sfbs_report(L, i_each, (int64_t)swap_remove_bit, 0, 0, 0, current_score, "bit_swap_remove");
          tk_sfbs_report(L, i_each, (int64_t)best_bit, 0, 0, best_score - pre_swap_score, current_score, "bit_swap_add");
        } else {
          selected_out[swap_remove_bit] = 1;
          n_selected++;
          tk_sfbs_commit_xor(xor_codes, chunks, (uint64_t)swap_remove_bit, distances, n_pairs, 1);
          current_score = pre_swap_score;
          phase = 3;
        }
      }

    } else {
      double *global_scores = (double *)calloc(n_bits, sizeof(double));
      int64_t best_bit = -1;
      double best_score = -INFINITY;

      #pragma omp parallel
      {
        double *ls = (double *)calloc(n_bits, sizeof(double));
        uint64_t *bc = (uint64_t *)calloc(n_buckets, sizeof(uint64_t));
        double *bw = (double *)calloc(n_buckets, sizeof(double));

        #pragma omp for schedule(static)
        for (uint64_t ni = 0; ni < n_nodes; ni++) {
          int64_t st = offs_a[ni];
          uint64_t nn = (uint64_t)(offs_a[ni + 1] - st);
          if (nn == 0 || idcg[ni] <= 0.0) continue;
          for (uint64_t b = 0; b < n_bits; b++) {
            if (!selected_out[b]) continue;
            memset(bc, 0, n_buckets * sizeof(uint64_t));
            memset(bw, 0, n_buckets * sizeof(double));
            for (uint64_t i = 0; i < nn; i++) {
              uint64_t j = (uint64_t)st + i;
              uint8_t xbit = (xor_codes[j * chunks + (b >> 3)] >> (b & 7)) & 1;
              uint16_t d = distances[j] - xbit;
              if (d >= n_buckets) d = (uint16_t)(n_buckets - 1);
              bc[d]++;
              bw[d] += weights_a[st + (int64_t)i];
            }
            ls[b] += tk_sfbs_dcg(bc, bw, prefix_discount, n_buckets) / idcg[ni];
          }
        }

        #pragma omp critical
        for (uint64_t b = 0; b < n_bits; b++)
          global_scores[b] += ls[b];

        free(ls);
        free(bc);
        free(bw);
      }

      for (uint64_t b = 0; b < n_bits; b++) {
        if (!selected_out[b]) continue;
        double sc = n_nodes > 0 ? global_scores[b] / (double)n_nodes : 0.0;
        if (sc > best_score) {
          best_score = sc;
          best_bit = (int64_t)b;
        }
      }
      free(global_scores);

      if (best_bit >= 0) {
        double gain = best_score - current_score;
        if (gain > tolerance) {
          selected_out[best_bit] = 0;
          n_selected--;
          tk_sfbs_commit_xor(xor_codes, chunks, (uint64_t)best_bit, distances, n_pairs, 0);
          current_score = best_score;
          phase = 0;
          tk_sfbs_report(L, i_each, (int64_t)best_bit, 0, 0, gain, current_score, "bit_remove");
        } else {
          swap_remove_bit = best_bit;
          pre_swap_score = current_score;
          selected_out[best_bit] = 0;
          n_selected--;
          tk_sfbs_commit_xor(xor_codes, chunks, (uint64_t)best_bit, distances, n_pairs, 0);
          current_score = best_score;
          phase = 2;
        }
      } else {
        phase = 3;
      }
    }
  }

  free(xor_codes);
  free(distances);
  if (out_score) *out_score = current_score;
  return n_selected;
}

static inline int tk_quantizer_create_lua(lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "mode");
  const char *mode = lua_isstring(L, -1) ? lua_tostring(L, -1) : NULL;
  lua_pop(L, 1);

  if (mode && strcmp(mode, "thermometer") == 0) {
    lua_getfield(L, 1, "raw_codes");
    tk_dvec_t *input_dvec = tk_dvec_peek(L, -1, "raw_codes");
    lua_pop(L, 1);
    double *inp = input_dvec->a;
    uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "create", "n_samples");
    uint64_t d_in = input_dvec->n / n_samples;
    uint64_t B = tk_lua_foptunsigned(L, 1, "create", "n_bins", 0);
    double **dt = (double **)malloc(d_in * sizeof(double *));
    uint64_t *dn = (uint64_t *)malloc(d_in * sizeof(uint64_t));
    #pragma omp parallel
    {
      double *col = (double *)malloc(n_samples * sizeof(double));
      #pragma omp for schedule(dynamic)
      for (uint64_t d = 0; d < d_in; d++) {
        for (uint64_t i = 0; i < n_samples; i++)
          col[i] = inp[i * d_in + d];
        tk_dvec_t tmp = { .a = col, .n = n_samples, .m = n_samples };
        tk_dvec_asc(&tmp, 0, n_samples);
        if (B == 0) {
          uint64_t nu = 0;
          for (uint64_t i = 0; i < n_samples; i++)
            if (i == 0 || col[i] != col[i - 1]) nu++;
          dt[d] = (double *)malloc(nu * sizeof(double));
          dn[d] = nu;
          uint64_t j = 0;
          for (uint64_t i = 0; i < n_samples; i++)
            if (i == 0 || col[i] != col[i - 1]) dt[d][j++] = col[i];
        } else {
          dt[d] = (double *)malloc(B * sizeof(double));
          dn[d] = B;
          for (uint64_t b = 0; b < B; b++) {
            uint64_t qi = (uint64_t)(((double)(b + 1) / (double)(B + 1)) * (double)n_samples);
            if (qi >= n_samples) qi = n_samples - 1;
            dt[d][b] = col[qi];
          }
        }
      }
      free(col);
    }
    uint64_t total_bits = 0;
    for (uint64_t d = 0; d < d_in; d++) total_bits += dn[d];
    tk_ivec_t *out_dims = tk_ivec_create(L, total_bits, 0, 0);
    int out_dims_idx = lua_gettop(L);
    tk_dvec_t *out_thresholds = tk_dvec_create(L, total_bits, 0, 0);
    int out_thresholds_idx = lua_gettop(L);
    out_dims->n = total_bits;
    out_thresholds->n = total_bits;
    uint64_t pos = 0;
    for (uint64_t d = 0; d < d_in; d++) {
      for (uint64_t k = 0; k < dn[d]; k++) {
        out_dims->a[pos] = (int64_t)d;
        out_thresholds->a[pos] = dt[d][k];
        pos++;
      }
      free(dt[d]);
    }
    free(dt); free(dn);
    tk_sfbs_encoder_t *enc = tk_lua_newuserdata(L, tk_sfbs_encoder_t,
      TK_SFBS_ENCODER_MT, tk_sfbs_encoder_mt_fns, tk_sfbs_encoder_gc);
    int Ei = lua_gettop(L);
    enc->bit_dims = out_dims;
    enc->bit_thresholds = out_thresholds;
    enc->n_dims = d_in;
    enc->n_bits = total_bits;
    enc->destroyed = false;
    lua_newtable(L);
    lua_pushvalue(L, out_dims_idx);
    lua_setfield(L, -2, "dims");
    lua_pushvalue(L, out_thresholds_idx);
    lua_setfield(L, -2, "thresholds");
    lua_setfenv(L, Ei);
    lua_pushvalue(L, Ei);
    return 1;
  }

  lua_getfield(L, 1, "raw_codes");
  tk_dvec_t *raw_codes_dvec = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  double *raw_codes = raw_codes_dvec ? raw_codes_dvec->a : NULL;

  lua_getfield(L, 1, "codes");
  tk_cvec_t *codes_cvec = tk_cvec_peekopt(L, -1);
  lua_pop(L, 1);

  if (!raw_codes_dvec && !codes_cvec)
    return luaL_error(L, "create: raw_codes or codes required");

  uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "create", "n_samples");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "create", "n_dims");
  uint64_t target_bits = tk_lua_foptunsigned(L, 1, "create", "target_bits", n_dims * 8);
  uint64_t max_dims = tk_lua_foptunsigned(L, 1, "create", "max_dims", 0);

  double tolerance = tk_lua_foptnumber(L, 1, "create", "tolerance", 0.0);

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peek(L, -1, "ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_ids");
  tk_ivec_t *adjacency_ids = tk_ivec_peek(L, -1, "expected_ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "expected_offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_weights");
  tk_dvec_t *weights = tk_dvec_peek(L, -1, "expected_weights");
  lua_pop(L, 1);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = lua_gettop(L);
  }

  uint64_t n_nodes = offsets->n - 1;
  uint64_t n_pairs = neighbors->n;

  int64_t max_uid = -1;
  for (uint64_t i = 0; i < ids->n; i++)
    if (ids->a[i] > max_uid) max_uid = ids->a[i];
  for (uint64_t i = 0; i < adjacency_ids->n; i++)
    if (adjacency_ids->a[i] > max_uid) max_uid = adjacency_ids->a[i];

  int64_t *uid_to_row = (int64_t *)malloc((uint64_t)(max_uid + 1) * sizeof(int64_t));
  memset(uid_to_row, -1, (uint64_t)(max_uid + 1) * sizeof(int64_t));
  for (uint64_t i = 0; i < ids->n; i++) {
    int64_t uid = ids->a[i];
    if (uid >= 0 && uid <= max_uid && uid_to_row[uid] < 0)
      uid_to_row[uid] = (int64_t)i;
  }

  int64_t sentinel_row = 0;
  int64_t *pair_row_a = (int64_t *)malloc(n_pairs * sizeof(int64_t));
  int64_t *pair_row_b = (int64_t *)malloc(n_pairs * sizeof(int64_t));
  for (uint64_t j = 0; j < n_pairs; j++) {
    pair_row_a[j] = sentinel_row;
    pair_row_b[j] = sentinel_row;
  }
  uint16_t *distances = (uint16_t *)calloc(n_pairs, sizeof(uint16_t));
  uint64_t max_neighbors = 0;

  for (uint64_t ni = 0; ni < n_nodes; ni++) {
    int64_t st = offsets->a[ni];
    int64_t en = offsets->a[ni + 1];
    uint64_t cnt = (uint64_t)(en - st);
    if (cnt > max_neighbors) max_neighbors = cnt;
    int64_t node_uid = adjacency_ids->a[ni];
    if (node_uid < 0 || node_uid > max_uid) continue;
    int64_t node_row = uid_to_row[node_uid];
    if (node_row < 0) continue;
    for (int64_t j = st; j < en; j++) {
      pair_row_a[j] = node_row;
      int64_t nbr_pos = neighbors->a[j];
      int64_t nbr_uid = adjacency_ids->a[nbr_pos];
      if (nbr_uid < 0 || nbr_uid > max_uid) continue;
      int64_t nbr_row = uid_to_row[nbr_uid];
      if (nbr_row < 0) continue;
      pair_row_b[j] = nbr_row;
    }
  }
  free(uid_to_row);

  double *prefix_discount = (double *)malloc((max_neighbors + 1) * sizeof(double));
  prefix_discount[0] = 0.0;
  for (uint64_t i = 0; i < max_neighbors; i++)
    prefix_discount[i + 1] = prefix_discount[i] + 1.0 / log2((double)(i + 2));

  double *idcg = (double *)calloc(n_nodes, sizeof(double));
  {
    double *iw = (double *)malloc(max_neighbors * sizeof(double));
    for (uint64_t ni = 0; ni < n_nodes; ni++) {
      int64_t st = offsets->a[ni];
      int64_t en = offsets->a[ni + 1];
      uint64_t nn = (uint64_t)(en - st);
      if (nn == 0) continue;
      for (uint64_t i = 0; i < nn; i++)
        iw[i] = weights->a[st + (int64_t)i];
      for (uint64_t i = 0; i < nn; i++)
        for (uint64_t k = i + 1; k < nn; k++)
          if (iw[k] > iw[i]) { double t = iw[i]; iw[i] = iw[k]; iw[k] = t; }
      for (uint64_t i = 0; i < nn; i++)
        idcg[ni] += iw[i] * (prefix_discount[i + 1] - prefix_discount[i]);
    }
    free(iw);
  }

  int64_t *offs_a = offsets->a;
  double *weights_a = weights->a;

  if (codes_cvec) {
    uint64_t src_bits = n_dims;
    uint8_t *bit_selected = (uint8_t *)calloc(src_bits, 1);
    double score = 0.0;
    uint64_t n_sel = tk_sfbs_binary_select(
      L, i_each, (unsigned char *)codes_cvec->a, n_samples, src_bits,
      pair_row_a, pair_row_b, offs_a, weights_a, idcg, prefix_discount,
      n_nodes, n_pairs, max_neighbors, target_bits, tolerance,
      bit_selected, &score);

    tk_ivec_t *out_dims = tk_ivec_create(L, n_sel, 0, 0);
    int out_dims_idx = lua_gettop(L);
    tk_dvec_t *out_thresholds = tk_dvec_create(L, n_sel, 0, 0);
    int out_thresholds_idx = lua_gettop(L);
    out_dims->n = n_sel;
    out_thresholds->n = n_sel;
    uint64_t w = 0;
    for (uint64_t b = 0; b < src_bits; b++) {
      if (bit_selected[b]) {
        out_dims->a[w] = (int64_t)b;
        out_thresholds->a[w] = 0.0;
        w++;
      }
    }
    free(bit_selected);

    tk_sfbs_encoder_t *enc = tk_lua_newuserdata(L, tk_sfbs_encoder_t,
      TK_SFBS_ENCODER_MT, tk_sfbs_encoder_mt_fns, tk_sfbs_encoder_gc);
    int Ei = lua_gettop(L);
    enc->bit_dims = out_dims;
    enc->bit_thresholds = out_thresholds;
    enc->n_dims = src_bits;
    enc->n_bits = n_sel;
    enc->destroyed = false;

    lua_newtable(L);
    lua_pushvalue(L, out_dims_idx);
    lua_setfield(L, -2, "dims");
    lua_pushvalue(L, out_thresholds_idx);
    lua_setfield(L, -2, "thresholds");
    lua_setfenv(L, Ei);

    free(pair_row_a);
    free(pair_row_b);
    free(idcg);
    free(prefix_discount);
    free(distances);
    lua_pushvalue(L, Ei);
    return 1;
  }

  double *medians = (double *)malloc(n_dims * sizeof(double));
  uint64_t n_sample_bytes = (n_samples + 7) / 8;
  uint64_t max_candidates = n_dims + 16 * target_bits;
  uint64_t n_candidates = n_dims;
  uint64_t *cand_dims = (uint64_t *)malloc(max_candidates * sizeof(uint64_t));
  double *cand_thresholds = (double *)malloc(max_candidates * sizeof(double));
  uint8_t **cand_bins = (uint8_t **)malloc(max_candidates * sizeof(uint8_t *));
  uint8_t *cand_selected = (uint8_t *)calloc(max_candidates, 1);
  #pragma omp parallel
  {
    tk_dvec_t *local_vals = tk_dvec_create(NULL, n_samples, 0, 0);
    #pragma omp for schedule(dynamic)
    for (uint64_t d = 0; d < n_dims; d++) {
      for (uint64_t i = 0; i < n_samples; i++)
        local_vals->a[i] = raw_codes[i * n_dims + d];
      tk_dvec_asc(local_vals, 0, n_samples);
      medians[d] = local_vals->a[n_samples / 2];
      cand_dims[d] = d;
      cand_thresholds[d] = medians[d];
      cand_bins[d] = (uint8_t *)calloc(n_sample_bytes, 1);
      for (uint64_t i = 0; i < n_samples; i++)
        if (raw_codes[i * n_dims + d] > medians[d])
          cand_bins[d][i >> 3] |= (uint8_t)(1 << (i & 7));
    }
    tk_dvec_destroy(local_vals);
  }
  free(medians);

  uint64_t n_selected = 0;
  uint64_t *selected_order = (uint64_t *)malloc(target_bits * sizeof(uint64_t));
  double current_score = 0.0;
  uint64_t *dim_refcount = (uint64_t *)calloc(n_dims, sizeof(uint64_t));
  uint64_t n_used_dims = 0;

  int phase = 0;
  int64_t swap_remove_cand = -1;
  double pre_swap_score = 0.0;

  double cand_score = 0.0;
  int64_t best_cand = -1;
  double best_score = -INFINITY;
  double remove_score = 0.0;

  #pragma omp parallel
  {
    uint64_t *bucket_counts = (uint64_t *)calloc(target_bits + 2, sizeof(uint64_t));
    double *bucket_wsums = (double *)calloc(target_bits + 2, sizeof(double));

    while (phase < 3) {

      if (phase == 0 || phase == 2) {

        #pragma omp single
        { best_cand = -1; best_score = -INFINITY; }

        for (uint64_t c = 0; c < n_candidates; c++) {
          if (cand_selected[c]) continue;
          if (max_dims > 0 && dim_refcount[cand_dims[c]] == 0
              && n_used_dims >= max_dims) continue;

          uint8_t *bin = cand_bins[c];
          uint64_t n_buckets = n_selected + 2;

          #pragma omp single
          { cand_score = 0.0; }

          #pragma omp for schedule(static) reduction(+:cand_score)
          for (uint64_t ni = 0; ni < n_nodes; ni++) {
            int64_t st = offs_a[ni];
            uint64_t nn = (uint64_t)(offs_a[ni + 1] - st);
            if (nn == 0 || idcg[ni] <= 0.0) continue;
            cand_score += tk_sfbs_score_node(
              bin, pair_row_a[st], pair_row_b, distances, weights_a,
              st, nn, n_buckets, 1, bucket_counts, bucket_wsums,
              prefix_discount) / idcg[ni];
          }

          #pragma omp single
          {
            double sc = n_nodes > 0 ? cand_score / (double)n_nodes : 0.0;
            if (sc > best_score) {
              best_score = sc;
              best_cand = (int64_t)c;
            }
          }
        }

        #pragma omp single
        {
          if (phase == 0) {
            bool added = false;
            if (best_cand >= 0 && n_selected < target_bits) {
              double gain = (n_selected == 0) ? best_score : (best_score - current_score);
              if (gain > tolerance) {
                tk_sfbs_commit(cand_bins[best_cand], pair_row_a, pair_row_b,
                  distances, n_pairs, 1);
                cand_selected[best_cand] = 1;
                if (dim_refcount[cand_dims[best_cand]]++ == 0) n_used_dims++;
                selected_order[n_selected] = (uint64_t)best_cand;
                n_selected++;
                current_score = best_score;
                added = true;
                tk_sfbs_spawn_children(raw_codes, n_samples, n_dims,
                  cand_dims[best_cand], cand_thresholds[best_cand],
                  cand_dims, cand_thresholds, cand_bins,
                  cand_selected, &n_candidates, max_candidates, n_sample_bytes);
                tk_sfbs_report(L, i_each, (int64_t)(n_selected - 1), (int64_t)cand_dims[best_cand],
                  cand_thresholds[best_cand], gain, current_score, "add");
              }
            }
            if (!added) {
              if (n_selected > 0) phase = 1;
              else phase = 3;
            }
          } else {
            if (best_cand >= 0 && best_score - pre_swap_score > tolerance) {
              tk_sfbs_commit(cand_bins[best_cand], pair_row_a, pair_row_b,
                distances, n_pairs, 1);
              cand_selected[best_cand] = 1;
              if (dim_refcount[cand_dims[best_cand]]++ == 0) n_used_dims++;
              selected_order[n_selected] = (uint64_t)best_cand;
              n_selected++;
              current_score = best_score;
              phase = 0;
              tk_sfbs_spawn_children(raw_codes, n_samples, n_dims,
                cand_dims[best_cand], cand_thresholds[best_cand],
                cand_dims, cand_thresholds, cand_bins,
                cand_selected, &n_candidates, max_candidates, n_sample_bytes);
              tk_sfbs_report(L, i_each, (int64_t)n_selected, (int64_t)cand_dims[swap_remove_cand],
                cand_thresholds[swap_remove_cand], 0, current_score, "swap_remove");
              tk_sfbs_report(L, i_each, (int64_t)(n_selected - 1), (int64_t)cand_dims[best_cand],
                cand_thresholds[best_cand], best_score - pre_swap_score, current_score, "swap_add");
            } else {
              tk_sfbs_commit(cand_bins[swap_remove_cand], pair_row_a, pair_row_b,
                distances, n_pairs, 1);
              cand_selected[swap_remove_cand] = 1;
              if (dim_refcount[cand_dims[swap_remove_cand]]++ == 0) n_used_dims++;
              selected_order[n_selected] = (uint64_t)swap_remove_cand;
              n_selected++;
              current_score = pre_swap_score;
              phase = 3;
            }
          }
        }

      } else {

        #pragma omp single
        { best_cand = -1; best_score = -INFINITY; }

        for (uint64_t s = 0; s < n_selected; s++) {
          uint64_t c = selected_order[s];
          uint8_t *bin = cand_bins[c];
          uint64_t n_buckets = n_selected + 2;

          #pragma omp single
          { remove_score = 0.0; }

          #pragma omp for schedule(static) reduction(+:remove_score)
          for (uint64_t ni = 0; ni < n_nodes; ni++) {
            int64_t st = offs_a[ni];
            uint64_t nn = (uint64_t)(offs_a[ni + 1] - st);
            if (nn == 0 || idcg[ni] <= 0.0) continue;
            remove_score += tk_sfbs_score_node(
              bin, pair_row_a[st], pair_row_b, distances, weights_a,
              st, nn, n_buckets, 0, bucket_counts, bucket_wsums,
              prefix_discount) / idcg[ni];
          }

          #pragma omp single
          {
            double sc = n_nodes > 0 ? remove_score / (double)n_nodes : 0.0;
            if (sc > best_score) {
              best_score = sc;
              best_cand = (int64_t)c;
            }
          }
        }

        #pragma omp single
        {
          if (best_cand >= 0) {
            double gain = best_score - current_score;
            if (gain > tolerance) {
              tk_sfbs_commit(cand_bins[best_cand], pair_row_a, pair_row_b,
                distances, n_pairs, 0);
              cand_selected[best_cand] = 0;
              if (--dim_refcount[cand_dims[best_cand]] == 0) n_used_dims--;
              for (uint64_t i = 0; i < n_selected; i++) {
                if (selected_order[i] == (uint64_t)best_cand) {
                  for (uint64_t j = i; j < n_selected - 1; j++)
                    selected_order[j] = selected_order[j + 1];
                  n_selected--;
                  break;
                }
              }
              current_score = best_score;
              phase = 0;
              tk_sfbs_report(L, i_each, (int64_t)n_selected, (int64_t)cand_dims[best_cand],
                cand_thresholds[best_cand], gain, current_score, "remove");
            } else {
              swap_remove_cand = best_cand;
              pre_swap_score = current_score;
              tk_sfbs_commit(cand_bins[best_cand], pair_row_a, pair_row_b,
                distances, n_pairs, 0);
              cand_selected[best_cand] = 0;
              if (--dim_refcount[cand_dims[best_cand]] == 0) n_used_dims--;
              for (uint64_t i = 0; i < n_selected; i++) {
                if (selected_order[i] == (uint64_t)best_cand) {
                  for (uint64_t j = i; j < n_selected - 1; j++)
                    selected_order[j] = selected_order[j + 1];
                  n_selected--;
                  break;
                }
              }
              current_score = best_score;
              phase = 2;
            }
          } else {
            phase = 3;
          }
        }

      }
    }

    free(bucket_counts);
    free(bucket_wsums);
  }

  tk_ivec_t *out_dims = tk_ivec_create(L, n_selected, 0, 0);
  int out_dims_idx = lua_gettop(L);
  tk_dvec_t *out_thresholds = tk_dvec_create(L, n_selected, 0, 0);
  int out_thresholds_idx = lua_gettop(L);
  out_dims->n = n_selected;
  out_thresholds->n = n_selected;
  for (uint64_t k = 0; k < n_selected; k++) {
    uint64_t c = selected_order[k];
    out_dims->a[k] = (int64_t)cand_dims[c];
    out_thresholds->a[k] = cand_thresholds[c];
  }

  for (uint64_t c = 0; c < n_candidates; c++) free(cand_bins[c]);
  free(cand_dims);
  free(cand_thresholds);
  free(cand_bins);
  free(cand_selected);
  free(selected_order);
  free(dim_refcount);

  tk_sfbs_encoder_t *enc = tk_lua_newuserdata(L, tk_sfbs_encoder_t,
    TK_SFBS_ENCODER_MT, tk_sfbs_encoder_mt_fns, tk_sfbs_encoder_gc);
  int Ei = lua_gettop(L);
  enc->bit_dims = out_dims;
  enc->bit_thresholds = out_thresholds;
  enc->n_dims = n_dims;
  enc->n_bits = n_selected;
  enc->destroyed = false;

  lua_newtable(L);
  lua_pushvalue(L, out_dims_idx);
  lua_setfield(L, -2, "dims");
  lua_pushvalue(L, out_thresholds_idx);
  lua_setfield(L, -2, "thresholds");
  lua_setfenv(L, Ei);

  free(distances);
  free(pair_row_a);
  free(pair_row_b);
  free(idcg);
  free(prefix_discount);

  lua_pushvalue(L, Ei);
  return 1;
}

static inline int tk_sfbs_encoder_persist_lua(lua_State *L) {
  tk_sfbs_encoder_t *enc = tk_sfbs_encoder_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_lua_fwrite(L, "TKqt", 1, 4, fh);
  uint8_t version = 1;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, (char *)&enc->n_dims, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *)&enc->n_bits, sizeof(uint64_t), 1, fh);
  tk_ivec_persist(L, enc->bit_dims, fh);
  tk_dvec_persist(L, enc->bit_thresholds, fh);
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

static inline int tk_quantizer_load_lua(lua_State *L) {
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKqt", 4) != 0)
    luaL_error(L, "invalid quantizer file (bad magic)");
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 1)
    luaL_error(L, "unsupported quantizer version %d", (int)version);
  uint64_t n_dims, n_bits;
  tk_lua_fread(L, &n_dims, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &n_bits, sizeof(uint64_t), 1, fh);
  tk_ivec_t *bit_dims = tk_ivec_load(L, fh);
  int dims_idx = lua_gettop(L);
  tk_dvec_t *bit_thresholds = tk_dvec_load(L, fh);
  int thresh_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);
  tk_sfbs_encoder_t *enc = tk_lua_newuserdata(L, tk_sfbs_encoder_t,
    TK_SFBS_ENCODER_MT, tk_sfbs_encoder_mt_fns, tk_sfbs_encoder_gc);
  int Ei = lua_gettop(L);
  enc->bit_dims = bit_dims;
  enc->bit_thresholds = bit_thresholds;
  enc->n_dims = n_dims;
  enc->n_bits = n_bits;
  enc->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, dims_idx);
  lua_setfield(L, -2, "dims");
  lua_pushvalue(L, thresh_idx);
  lua_setfield(L, -2, "thresholds");
  lua_setfenv(L, Ei);
  lua_pushvalue(L, Ei);
  return 1;
}

static luaL_Reg tk_quantizer_fns[] = {
  { "create", tk_quantizer_create_lua },
  { "load", tk_quantizer_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_quantizer(lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_quantizer_fns, 0);
  return 1;
}
