#include <santoku/learn/csr.h>
#include <santoku/learn/normalize.h>
#include <santoku/fvec.h>
#include <santoku/svec.h>
#include <santoku/cvec.h>
#include <santoku/ivec/ext.h>
#include <santoku/iumap/ext.h>
#include <santoku/learn/mathlibs.h>
#include <assert.h>
#include <math.h>

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

static int tm_csr_seq_select (lua_State *L)
{
  tk_svec_t *tokens = tk_svec_peek(L, 1, "tokens");
  tk_ivec_t *offsets = tk_ivec_peek(L, 2, "offsets");
  tk_ivec_t *keep_ids = tk_ivec_peek(L, 3, "keep_ids");
  tk_fvec_t *values_f = tk_fvec_peekopt(L, 4);
  tk_dvec_t *values_d = values_f ? NULL : tk_dvec_peekopt(L, 4);
  int has_values = values_f || values_d;
  tk_iumap_t *inverse = tk_iumap_from_ivec(L, keep_ids);
  if (!inverse) return luaL_error(L, "seq_select: allocation failed");
  uint64_t n_docs = offsets->n - 1;
  tk_svec_t *new_tok = tk_svec_create(L, tokens->n);
  tk_ivec_t *new_off = tk_ivec_create(L, n_docs + 1);
  new_off->n = n_docs + 1;
  tk_fvec_t *new_val_f = (has_values && values_f) ? tk_fvec_create(L, tokens->n) : NULL;
  tk_dvec_t *new_val_d = (has_values && values_d) ? tk_dvec_create(L, tokens->n) : NULL;
  new_off->a[0] = 0;
  uint64_t pos = 0;
  for (uint64_t d = 0; d < n_docs; d++) {
    int64_t start = offsets->a[d];
    int64_t end = offsets->a[d + 1];
    for (int64_t j = start; j < end; j++) {
      int64_t new_id = tk_iumap_get_or(inverse, (int64_t)tokens->a[j], -1);
      if (new_id < 0) continue;
      new_tok->a[pos] = (int32_t)new_id;
      if (new_val_f)
        new_val_f->a[pos] = values_f->a[j];
      else if (new_val_d)
        new_val_d->a[pos] = values_d->a[j];
      pos++;
    }
    new_off->a[d + 1] = (int64_t)pos;
  }
  new_tok->n = pos;
  if (new_val_f) new_val_f->n = pos;
  if (new_val_d) new_val_d->n = pos;
  tk_iumap_destroy(inverse);
  return has_values ? 3 : 2;
}

static int tm_csr_label_union (lua_State *L)
{
  tk_ivec_t *nn_off = tk_ivec_peek(L, 1, "nn_offsets");
  tk_ivec_t *nn_nbr = tk_ivec_peek(L, 2, "nn_neighbors");
  tk_ivec_t *hood_ids = tk_ivec_peek(L, 3, "hood_ids");
  tk_ivec_t *lab_off = tk_ivec_peek(L, 4, "label_offsets");
  tk_ivec_t *lab_nbr = tk_ivec_peek(L, 5, "label_neighbors");
  uint64_t n_labels = tk_lua_checkunsigned(L, 6, "n_labels");

  uint64_t n_queries = nn_off->n - 1;
  uint64_t bm_bytes = (n_labels + 7) / 8;

  uint64_t *counts = (uint64_t *)calloc(n_queries, sizeof(uint64_t));
  if (!counts)
    return luaL_error(L, "label_union: allocation failed");

  #pragma omp parallel
  {
    uint8_t *bm = (uint8_t *)calloc(1, bm_bytes);
    #pragma omp for schedule(dynamic, 64)
    for (uint64_t i = 0; i < n_queries; i++) {
      memset(bm, 0, bm_bytes);
      int64_t ns = nn_off->a[i], ne = nn_off->a[i + 1];
      for (int64_t j = ns; j < ne; j++) {
        int64_t uid = hood_ids->a[nn_nbr->a[j]];
        int64_t ls = lab_off->a[uid], le = lab_off->a[uid + 1];
        for (int64_t k = ls; k < le; k++) {
          uint64_t lab = (uint64_t)lab_nbr->a[k];
          bm[lab / 8] |= (uint8_t)(1 << (lab % 8));
        }
      }
      uint64_t cnt = 0;
      for (uint64_t b = 0; b < bm_bytes; b++)
        cnt += (uint64_t)__builtin_popcount((unsigned int)bm[b]);
      counts[i] = cnt;
    }
    free(bm);
  }

  uint64_t total = 0;
  for (uint64_t i = 0; i < n_queries; i++)
    total += counts[i];

  tk_ivec_t *out_off = tk_ivec_create(L, n_queries + 1);
  out_off->n = n_queries + 1;
  tk_ivec_t *out_nbr = tk_ivec_create(L, total);
  out_nbr->n = total;

  out_off->a[0] = 0;
  for (uint64_t i = 0; i < n_queries; i++)
    out_off->a[i + 1] = out_off->a[i] + (int64_t)counts[i];
  free(counts);

  #pragma omp parallel
  {
    uint8_t *bm = (uint8_t *)calloc(1, bm_bytes);
    #pragma omp for schedule(dynamic, 64)
    for (uint64_t i = 0; i < n_queries; i++) {
      memset(bm, 0, bm_bytes);
      int64_t ns = nn_off->a[i], ne = nn_off->a[i + 1];
      for (int64_t j = ns; j < ne; j++) {
        int64_t uid = hood_ids->a[nn_nbr->a[j]];
        int64_t ls = lab_off->a[uid], le = lab_off->a[uid + 1];
        for (int64_t k = ls; k < le; k++) {
          uint64_t lab = (uint64_t)lab_nbr->a[k];
          bm[lab / 8] |= (uint8_t)(1 << (lab % 8));
        }
      }
      uint64_t wp = (uint64_t)out_off->a[i];
      for (uint64_t f = 0; f < n_labels; f++) {
        if (bm[f / 8] & (1 << (f % 8)))
          out_nbr->a[wp++] = (int64_t)f;
      }
    }
    free(bm);
  }

  return 2;
}

static int tm_csr_transpose (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_svec_t *tokens = tk_svec_peek(L, 2, "tokens");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_tokens = tk_lua_checkunsigned(L, 4, "n_tokens");
  tk_fvec_t *values_f = tk_fvec_peekopt(L, 5);
  tk_dvec_t *values_d = values_f ? NULL : tk_dvec_peekopt(L, 5);
  int has_values = values_f || values_d;
  uint64_t nnz = tokens->n;
  int64_t *counts = (int64_t *)calloc(n_tokens + 1, sizeof(int64_t));
  if (!counts)
    return luaL_error(L, "transpose: allocation failed");
  for (uint64_t i = 0; i < nnz; i++)
    counts[tokens->a[i] + 1]++;
  for (uint64_t t = 0; t < n_tokens; t++)
    counts[t + 1] += counts[t];
  tk_ivec_t *csc_off = tk_ivec_create(L, n_tokens + 1);
  csc_off->n = n_tokens + 1;
  memcpy(csc_off->a, counts, (n_tokens + 1) * sizeof(int64_t));
  tk_ivec_t *csc_rows = tk_ivec_create(L, nnz);
  csc_rows->n = nnz;
  tk_fvec_t *csc_vals_f = (has_values && values_f) ? tk_fvec_create(L, nnz) : NULL;
  tk_dvec_t *csc_vals_d = (has_values && values_d) ? tk_dvec_create(L, nnz) : NULL;
  if (csc_vals_f) csc_vals_f->n = nnz;
  if (csc_vals_d) csc_vals_d->n = nnz;
  for (uint64_t s = 0; s < n_samples; s++) {
    for (int64_t j = offsets->a[s]; j < offsets->a[s + 1]; j++) {
      int32_t tok = tokens->a[j];
      int64_t pos = counts[tok]++;
      csc_rows->a[pos] = (int64_t)s;
      if (csc_vals_f) csc_vals_f->a[pos] = values_f->a[j];
      else if (csc_vals_d) csc_vals_d->a[pos] = values_d->a[j];
    }
  }
  free(counts);
  return has_values ? 3 : 2;
}

static int tm_csr_sort_csr_desc (lua_State *L)
{
  tk_ivec_t *off = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *nbr = tk_ivec_peek(L, 2, "neighbors");
  tk_fvec_t *scores_f = tk_fvec_peekopt(L, 3);
  tk_dvec_t *scores_d = scores_f ? NULL : tk_dvec_peek(L, 3, "scores");
  uint64_t n = off->n - 1;
  tk_ivec_t *out_n = tk_ivec_create(L, nbr->n);
  memcpy(out_n->a, nbr->a, nbr->n * sizeof(int64_t));
  if (scores_f) {
    tk_fvec_t *out_s = tk_fvec_create(L, scores_f->n);
    memcpy(out_s->a, scores_f->a, scores_f->n * sizeof(float));
    #pragma omp parallel for schedule(dynamic, 64)
    for (uint64_t i = 0; i < n; i++) {
      int64_t s = off->a[i], e = off->a[i + 1];
      for (int64_t j = s + 1; j < e; j++) {
        float ks = out_s->a[j];
        int64_t kn = out_n->a[j];
        int64_t p = j - 1;
        while (p >= s && out_s->a[p] < ks) {
          out_s->a[p + 1] = out_s->a[p];
          out_n->a[p + 1] = out_n->a[p];
          p--;
        }
        out_s->a[p + 1] = ks;
        out_n->a[p + 1] = kn;
      }
    }
  } else {
    tk_dvec_t *out_s = tk_dvec_create(L, scores_d->n);
    memcpy(out_s->a, scores_d->a, scores_d->n * sizeof(double));
    #pragma omp parallel for schedule(dynamic, 64)
    for (uint64_t i = 0; i < n; i++) {
      int64_t s = off->a[i], e = off->a[i + 1];
      for (int64_t j = s + 1; j < e; j++) {
        double ks = out_s->a[j];
        int64_t kn = out_n->a[j];
        int64_t p = j - 1;
        while (p >= s && out_s->a[p] < ks) {
          out_s->a[p + 1] = out_s->a[p];
          out_n->a[p + 1] = out_n->a[p];
          p--;
        }
        out_s->a[p + 1] = ks;
        out_n->a[p + 1] = kn;
      }
    }
  }
  return 2;
}

// Pack byte n-grams: exact bit-concatenation when n <= 8 (fits in 64 bits),
// else a polynomial rolling hash. Elements are always bytes.
static inline size_t tm_csr_pack_ngrams_w (
  const void *data, size_t n_elems, int n, int64_t *out)
{
  const uint8_t *d = (const uint8_t *)data;
  if (n_elems < (size_t)n) return 0;
  size_t count = n_elems - (size_t)n + 1;
  if (n <= 8) {
    uint64_t mask = (n < 8) ? ((1ULL << (n * 8)) - 1) : ~0ULL;
    uint64_t id = 0;
    for (int i = 0; i < n - 1; i++)
      id = (id << 8) | d[i];
    for (size_t i = 0; i < count; i++) {
      id = ((id << 8) | d[(size_t)(n - 1) + i]) & mask;
      out[i] = (int64_t)id;
    }
  } else {
    const uint64_t P = 0x9E3779B97F4A7C15ULL;
    uint64_t p_pow_n = 1;
    for (int j = 0; j < n - 1; j++) p_pow_n *= P;
    uint64_t h = 0;
    for (int j = 0; j < n; j++)
      h = h * P + d[j];
    out[0] = (int64_t)h;
    for (size_t i = 1; i < count; i++) {
      h = (h - d[i - 1] * p_pow_n) * P + d[i + (size_t)n - 1];
      out[i] = (int64_t)h;
    }
  }
  return count;
}

static inline size_t tm_csr_do_pack (
  const char *str, size_t len, int64_t ngram_min, int64_t ngram_max,
  bool normalize, int64_t *out, uint8_t *norm)
{
  const char *src = str;
  size_t srclen = len;
  if (normalize) {
    srclen = tk_text_normalize_buffer(str, len, norm, NULL);
    src = (const char *)norm;
  }
  size_t count = 0;
  for (int64_t ng = ngram_min; ng <= ngram_max; ng++)
    count += tm_csr_pack_ngrams_w(src, srclen, (int)ng, out + count);
  return count;
}

// Like tm_csr_do_pack, but also emits each gram's RAW byte start position into pos[]. Under normalize,
// translates the normalized buffer index through src_index back to raw input coordinates.
static inline size_t tm_csr_do_pack_pos (
  const char *str, size_t len, int64_t ngram_min, int64_t ngram_max,
  bool normalize, int64_t *out, int64_t *pos, uint8_t *norm, size_t *src_index)
{
  const char *src = str;
  size_t srclen = len;
  if (normalize) {
    srclen = tk_text_normalize_buffer(str, len, norm, src_index);
    src = (const char *)norm;
  }
  size_t count = 0;
  for (int64_t ng = ngram_min; ng <= ngram_max; ng++) {
    size_t c = tm_csr_pack_ngrams_w(src, srclen, (int)ng, out + count);
    for (size_t j = 0; j < c; j++)
      pos[count + j] = normalize ? (int64_t)src_index[j] : (int64_t)j;
    count += c;
  }
  return count;
}

static int tm_csr_tokenize_core (lua_State *L, const char **strs, size_t *lens,
    size_t max_len, int64_t n_samples, int64_t ngram_min, int64_t ngram_max,
    bool normalize, bool sequence)
{
  size_t buf_size = (size_t)(ngram_max - ngram_min + 1) * max_len;
  lua_getfield(L, 1, "ngram_map");
  bool raw_mode = lua_isboolean(L, -1) && !lua_toboolean(L, -1);
  bool have_map = !lua_isnil(L, -1) && !raw_mode;
  tk_iumap_t *ngram_map = NULL;
  int map_idx;
  int64_t next_id = 0;
  int64_t *sample_counts = (int64_t *)calloc((size_t)n_samples, sizeof(int64_t));
  if (have_map) {
    ngram_map = tk_iumap_peek(L, -1, "ngram_map");
    map_idx = lua_gettop(L);
    next_id = (int64_t)tk_iumap_size(ngram_map);
    uint32_t me = tk_iumap_end(ngram_map);
    #pragma omp parallel
    {
      int64_t *packed_buf = (int64_t *)malloc(buf_size * sizeof(int64_t));
      uint8_t *norm_buf = (uint8_t *)malloc(max_len ? max_len : 1);
      #pragma omp for schedule(dynamic)
      for (int64_t s = 0; s < n_samples; s++) {
        if (!strs[s] || !lens[s]) continue;
        size_t count = tm_csr_do_pack(strs[s], lens[s], ngram_min, ngram_max, normalize, packed_buf, norm_buf);
        int64_t nv = 0;
        for (size_t i = 0; i < count; i++) {
          uint32_t iter = tk_iumap_get(ngram_map, packed_buf[i]);
          if (iter != me) packed_buf[nv++] = tk_iumap_val(ngram_map, iter);
        }
        if (sequence) {
          sample_counts[s] = nv;
        } else {
          ks_introsort(tk_ivec_asc, (size_t)nv, packed_buf);
          int64_t unique = 0;
          for (int64_t i = 0; i < nv; i++)
            if (i == 0 || packed_buf[i] != packed_buf[i - 1]) unique++;
          sample_counts[s] = unique;
        }
      }
      free(packed_buf);
      free(norm_buf);
    }
  } else if (raw_mode) {
    lua_pop(L, 1);
    #pragma omp parallel
    {
      int64_t *packed_buf = (int64_t *)malloc(buf_size * sizeof(int64_t));
      uint8_t *norm_buf = (uint8_t *)malloc(max_len ? max_len : 1);
      #pragma omp for schedule(dynamic)
      for (int64_t s = 0; s < n_samples; s++) {
        if (!strs[s] || !lens[s]) continue;
        size_t count = tm_csr_do_pack(strs[s], lens[s], ngram_min, ngram_max, normalize, packed_buf, norm_buf);
        if (sequence) {
          sample_counts[s] = (int64_t)count;
        } else {
          ks_introsort(tk_ivec_asc, count, packed_buf);
          int64_t unique = 0;
          for (size_t i = 0; i < count; i++)
            if (i == 0 || packed_buf[i] != packed_buf[i - 1]) unique++;
          sample_counts[s] = unique;
        }
      }
      free(packed_buf);
      free(norm_buf);
    }
    lua_pushnil(L);
    map_idx = lua_gettop(L);
  } else {
    lua_pop(L, 1);
    int max_threads = omp_get_max_threads();
    tk_iumap_t **local_maps = (tk_iumap_t **)calloc((size_t)max_threads, sizeof(tk_iumap_t *));
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      tk_iumap_t *lm = tk_iumap_create(NULL, 0);
      local_maps[tid] = lm;
      int64_t *packed_buf = (int64_t *)malloc(buf_size * sizeof(int64_t));
      uint8_t *norm_buf = (uint8_t *)malloc(max_len ? max_len : 1);
      #pragma omp for schedule(dynamic)
      for (int64_t s = 0; s < n_samples; s++) {
        if (!strs[s] || !lens[s]) continue;
        size_t count = tm_csr_do_pack(strs[s], lens[s], ngram_min, ngram_max, normalize, packed_buf, norm_buf);
        for (size_t i = 0; i < count; i++) {
          int absent;
          tk_iumap_put(lm, packed_buf[i], &absent);
        }
        if (sequence) {
          sample_counts[s] = (int64_t)count;
        } else {
          ks_introsort(tk_ivec_asc, count, packed_buf);
          int64_t unique = 0;
          for (size_t i = 0; i < count; i++)
            if (i == 0 || packed_buf[i] != packed_buf[i - 1]) unique++;
          sample_counts[s] = unique;
        }
      }
      free(packed_buf);
      free(norm_buf);
    }
    uint32_t est = 0;
    for (int t = 0; t < max_threads; t++)
      if (local_maps[t] && tk_iumap_size(local_maps[t]) > est)
        est = tk_iumap_size(local_maps[t]);
    ngram_map = tk_iumap_create(L, est);
    next_id = 0;
    for (int t = 0; t < max_threads; t++) {
      if (!local_maps[t]) continue;
      int64_t k;
      tk_umap_foreach_keys(local_maps[t], k, ({
        int absent;
        uint32_t gi = tk_iumap_put(ngram_map, k, &absent);
        if (absent)
          tk_iumap_setval(ngram_map, gi, next_id++);
      }));
      tk_iumap_destroy(local_maps[t]);
    }
    free(local_maps);
    map_idx = lua_gettop(L);
  }
  uint64_t n_tokens = raw_mode ? 0 : (uint64_t)next_id;
  uint32_t map_end = ngram_map ? tk_iumap_end(ngram_map) : 0;
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n_samples + 1));
  offsets->n = (uint64_t)(n_samples + 1);
  offsets->a[0] = 0;
  int64_t total = 0;
  for (int64_t s = 0; s < n_samples; s++) {
    total += sample_counts[s];
    offsets->a[s + 1] = total;
  }
  free(sample_counts);
  tk_svec_t *stok_out = NULL;
  tk_ivec_t *itok_out = NULL;
  if (raw_mode) {
    itok_out = tk_ivec_create(L, (uint64_t)total);
    itok_out->n = (uint64_t)total;
  } else {
    stok_out = tk_svec_create(L, (uint64_t)total);
    stok_out->n = (uint64_t)total;
  }
  tk_fvec_t *val_out = NULL;
  tk_ivec_t *pos_out = NULL;
  if (sequence) {
    pos_out = tk_ivec_create(L, (uint64_t)total);
    pos_out->n = (uint64_t)total;
  } else {
    val_out = tk_fvec_create(L, (uint64_t)total);
    val_out->n = (uint64_t)total;
  }
  #pragma omp parallel
  {
    int64_t *packed_buf = (int64_t *)malloc(buf_size * sizeof(int64_t));
    uint8_t *norm_buf = (uint8_t *)malloc(max_len ? max_len : 1);
    int64_t *pos_buf = sequence ? (int64_t *)malloc(buf_size * sizeof(int64_t)) : NULL;
    size_t *src_index = (sequence && normalize) ? (size_t *)malloc((max_len ? max_len : 1) * sizeof(size_t)) : NULL;
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      if (!strs[s] || !lens[s]) continue;
      if (sequence) {
        size_t count = tm_csr_do_pack_pos(strs[s], lens[s], ngram_min, ngram_max, normalize, packed_buf, pos_buf, norm_buf, src_index);
        int64_t w = offsets->a[s];
        if (raw_mode) {
          for (size_t i = 0; i < count; i++) {
            itok_out->a[w] = packed_buf[i];
            pos_out->a[w] = pos_buf[i];
            w++;
          }
        } else {
          for (size_t i = 0; i < count; i++) {
            uint32_t iter = tk_iumap_get(ngram_map, packed_buf[i]);
            if (iter == map_end) continue;
            stok_out->a[w] = (int32_t)tk_iumap_val(ngram_map, iter);
            pos_out->a[w] = pos_buf[i];
            w++;
          }
        }
        continue;
      }
      size_t count = tm_csr_do_pack(strs[s], lens[s], ngram_min, ngram_max, normalize, packed_buf, norm_buf);
      int64_t nv;
      if (raw_mode) {
        ks_introsort(tk_ivec_asc, count, packed_buf);
        nv = (int64_t)count;
      } else {
        nv = 0;
        for (size_t i = 0; i < count; i++) {
          uint32_t iter = tk_iumap_get(ngram_map, packed_buf[i]);
          if (iter != map_end) packed_buf[nv++] = tk_iumap_val(ngram_map, iter);
        }
        ks_introsort(tk_ivec_asc, (size_t)nv, packed_buf);
      }
      int64_t pos = offsets->a[s];
      for (int64_t i = 0; i < nv; ) {
        int64_t tok = packed_buf[i];
        float cnt = 0.0f;
        while (i < nv && packed_buf[i] == tok) { cnt += 1.0f; i++; }
        if (raw_mode)
          itok_out->a[pos] = tok;
        else
          stok_out->a[pos] = (int32_t)tok;
        val_out->a[pos] = cnt;
        pos++;
      }
    }
    free(packed_buf);
    free(norm_buf);
    free(pos_buf);
    free(src_index);
  }
  lua_pushvalue(L, map_idx);
  lua_pushvalue(L, map_idx + 1);
  lua_pushvalue(L, map_idx + 2);
  lua_pushvalue(L, map_idx + 3);
  lua_pushinteger(L, (lua_Integer)n_tokens);
  return 5;
}

static int tm_csr_tokenize (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t ngram_min = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize", "ngram_min");
  int64_t ngram_max = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize", "ngram_max");
  if (ngram_min < 1 || ngram_min > ngram_max)
    return luaL_error(L, "tokenize: need 1 <= ngram_min <= ngram_max");
  bool do_normalize = tk_lua_foptboolean(L, 1, "tokenize", "normalize", false);
  bool terminals = tk_lua_foptboolean(L, 1, "tokenize", "terminals", false);
  bool sequence = tk_lua_foptboolean(L, 1, "tokenize", "sequence", false);
  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize", "n_samples");

  lua_getfield(L, 1, "sequences");
  if (!lua_isnil(L, -1)) {
    lua_getfield(L, 1, "sequence_offsets");
    tk_ivec_t *seq_off = tk_ivec_peek(L, -1, "sequence_offsets");
    lua_pop(L, 1);
    tk_cvec_t *seq_cv = tk_cvec_peek(L, -1, "sequences");
    const char *seq_data = (const char *)seq_cv->a;
    lua_pop(L, 1);
    const char **strs = (const char **)malloc((uint64_t)n_samples * sizeof(const char *));
    size_t *lens = (size_t *)malloc((uint64_t)n_samples * sizeof(size_t));
    size_t max_len = 0;
    for (int64_t s = 0; s < n_samples; s++) {
      int64_t s0 = seq_off->a[s], s1 = seq_off->a[s + 1];
      strs[s] = seq_data + s0;
      lens[s] = (size_t)(s1 - s0);
      if (lens[s] > max_len) max_len = lens[s];
    }
    int result = tm_csr_tokenize_core(L, strs, lens, max_len, n_samples, ngram_min, ngram_max, do_normalize, sequence);
    free(strs);
    free(lens);
    return result;
  }
  lua_pop(L, 1);

  lua_getfield(L, 1, "texts");
  const char **strs = (const char **)malloc((uint64_t)n_samples * sizeof(const char *));
  size_t *lens = (size_t *)malloc((uint64_t)n_samples * sizeof(size_t));
  size_t max_len = 0;
  if (lua_isfunction(L, -1)) {
    int fn_idx = lua_gettop(L);
    lua_newtable(L);
    int anchor_idx = lua_gettop(L);
    for (int64_t s = 0; s < n_samples; s++) {
      lua_pushvalue(L, fn_idx);
      lua_call(L, 0, 3);
      if (lua_isnil(L, -3)) {
        lua_pop(L, 3);
        for (int64_t r = s; r < n_samples; r++) {
          strs[r] = NULL;
          lens[r] = 0;
        }
        break;
      }
      if (!lua_isnil(L, -2) && !lua_isnil(L, -1)) {
        size_t full_len;
        const char *full_str = lua_tolstring(L, -3, &full_len);
        int64_t sub_s = lua_tointeger(L, -2);
        int64_t sub_e = lua_tointeger(L, -1);
        if (sub_s < 1) sub_s = 1;
        if ((uint64_t)sub_e > full_len) sub_e = (int64_t)full_len;
        strs[s] = full_str + (sub_s - 1);
        lens[s] = (sub_e >= sub_s) ? (size_t)(sub_e - sub_s + 1) : 0;
      } else {
        strs[s] = lua_tolstring(L, -3, &lens[s]);
      }
      lua_pushvalue(L, -3);
      lua_rawseti(L, anchor_idx, (int)(s + 1));
      lua_pop(L, 3);
      if (lens[s] > max_len) max_len = lens[s];
    }
  } else if (lua_istable(L, -1)) {
    int texts_idx = lua_gettop(L);
    for (int64_t s = 0; s < n_samples; s++) {
      lua_rawgeti(L, texts_idx, (int)(s + 1));
      strs[s] = lua_tolstring(L, -1, &lens[s]);
      lua_pop(L, 1);
      if (lens[s] > max_len) max_len = lens[s];
    }
  } else {
    free(strs);
    free(lens);
    return luaL_error(L, "tokenize: texts must be table or function");
  }
  char *term_pool = NULL;
  if (terminals) {
    size_t total_bytes = 0;
    for (int64_t s = 0; s < n_samples; s++)
      if (strs[s] && lens[s]) total_bytes += lens[s] + 2;
    term_pool = (char *)malloc(total_bytes);
    char *p = term_pool;
    for (int64_t s = 0; s < n_samples; s++) {
      if (!strs[s] || !lens[s]) continue;
      p[0] = '\x03';
      memcpy(p + 1, strs[s], lens[s]);
      p[1 + lens[s]] = '\x04';
      strs[s] = p;
      lens[s] += 2;
      p += lens[s];
    }
    max_len += 2;
  }
  int result = tm_csr_tokenize_core(L, strs, lens, max_len, n_samples, ngram_min, ngram_max, do_normalize, sequence);
  free(term_pool);
  free(strs);
  free(lens);
  return result;
}

static int tm_csr_tokenize_annotated (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t ngram_min = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize_annotated", "ngram_min");
  int64_t ngram_max = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize_annotated", "ngram_max");
  if (ngram_min < 1 || ngram_min > ngram_max)
    return luaL_error(L, "tokenize_annotated: need 1 <= ngram_min <= ngram_max");
  bool do_normalize = tk_lua_foptboolean(L, 1, "tokenize_annotated", "normalize", false);
  bool terminals = tk_lua_foptboolean(L, 1, "tokenize_annotated", "terminals", false);
  lua_getfield(L, 1, "collapse");
  const char *collapse_str = lua_tostring(L, -1);
  lua_pop(L, 1);
  enum { COL_NONE, COL_FOCUS, COL_MARK, COL_SPANS, COL_ALL, COL_CONTEXT } collapse = COL_NONE;
  if (collapse_str) {
    if (strcmp(collapse_str, "focus") == 0) collapse = COL_FOCUS;
    else if (strcmp(collapse_str, "mark") == 0) collapse = COL_MARK;
    else if (strcmp(collapse_str, "spans") == 0) collapse = COL_SPANS;
    else if (strcmp(collapse_str, "all") == 0) collapse = COL_ALL;
    else if (strcmp(collapse_str, "context") == 0) collapse = COL_CONTEXT;
  }
  lua_getfield(L, 1, "doc_span_offsets");
  tk_ivec_t *doc_span_offsets = tk_ivec_peek(L, -1, "doc_span_offsets");
  lua_pop(L, 1);
  lua_getfield(L, 1, "span_starts");
  tk_ivec_t *span_starts = tk_ivec_peek(L, -1, "span_starts");
  lua_pop(L, 1);
  lua_getfield(L, 1, "span_ends");
  tk_ivec_t *span_ends = tk_ivec_peek(L, -1, "span_ends");
  lua_pop(L, 1);
  lua_getfield(L, 1, "span_types");
  tk_ivec_t *span_types = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);
  // context spans form the collapsed backdrop in mark/spans/all; default = focus set
  lua_getfield(L, 1, "context_offsets");
  tk_ivec_t *ctx_off = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);
  tk_ivec_t *ctx_starts = span_starts, *ctx_ends = span_ends, *ctx_types = span_types;
  if (ctx_off) {
    lua_getfield(L, 1, "context_starts"); ctx_starts = tk_ivec_peek(L, -1, "context_starts"); lua_pop(L, 1);
    lua_getfield(L, 1, "context_ends"); ctx_ends = tk_ivec_peek(L, -1, "context_ends"); lua_pop(L, 1);
    lua_getfield(L, 1, "context_types"); ctx_types = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  } else {
    ctx_off = doc_span_offsets;
  }

  int64_t n_docs = (int64_t)(doc_span_offsets->n - 1);
  int64_t total_spans = doc_span_offsets->a[n_docs];
  size_t *text_lens = (size_t *)malloc((uint64_t)n_docs * sizeof(size_t));
  const char **text_ptrs = (const char **)malloc((uint64_t)n_docs * sizeof(const char *));
  lua_getfield(L, 1, "cvec");
  if (!lua_isnil(L, -1)) {
    // compact byte input: one flat buffer + sequence_offsets (kept on stack alive)
    tk_cvec_t *cv = tk_cvec_peek(L, -1, "cvec");
    lua_getfield(L, 1, "sequence_offsets");
    tk_ivec_t *so = tk_ivec_peek(L, -1, "sequence_offsets");
    for (int64_t d = 0; d < n_docs; d++) {
      text_ptrs[d] = (const char *)cv->a + so->a[d];
      text_lens[d] = (size_t)(so->a[d + 1] - so->a[d]);
    }
    lua_pop(L, 1); // pop sequence_offsets; leave cvec on stack to anchor cv->a
  } else {
    lua_pop(L, 1); // pop nil cvec
    lua_getfield(L, 1, "texts");
    luaL_checktype(L, -1, LUA_TTABLE);
    int texts_idx = lua_gettop(L);
    for (int64_t d = 0; d < n_docs; d++) {
      lua_rawgeti(L, texts_idx, (int)(d + 1));
      text_ptrs[d] = lua_tolstring(L, -1, &text_lens[d]);
      lua_pop(L, 1);
    }
  }
  static const uint8_t default_markers[] = {
    0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
    0x0E,0x0F,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
    0x7F
  };
  lua_getfield(L, 1, "markers");
  size_t mk_len = 0;
  const uint8_t *mk = (const uint8_t *)lua_tolstring(L, -1, &mk_len);
  if (!mk) { mk = default_markers; mk_len = sizeof(default_markers); }
  int64_t max_type = 0;
  if (span_types)
    for (int64_t i = 0; i < total_spans; i++)
      if (span_types->a[i] > max_type) max_type = span_types->a[i];
  if (ctx_types && ctx_types != span_types) {
    int64_t total_ctx = ctx_off->a[n_docs];
    for (int64_t i = 0; i < total_ctx; i++)
      if (ctx_types->a[i] > max_type) max_type = ctx_types->a[i];
  }
  size_t need_mk = (size_t)(3 + 2 * (max_type + 1));
  if (mk_len < need_mk) {
    free(text_ptrs); free(text_lens);
    return luaL_error(L, "tokenize_annotated: markers too short (need %d, have %d)", (int)need_mk, (int)mk_len);
  }
  uint8_t M_FOCUS = mk[0], M_START = mk[1], M_END = mk[2];
  #define M_OPEN(t)  (mk[3 + 2 * (t)])
  #define M_CLOSE(t) (mk[4 + 2 * (t)])
  const char **strs = (const char **)malloc((uint64_t)total_spans * sizeof(const char *));
  size_t *lens = (size_t *)malloc((uint64_t)total_spans * sizeof(size_t));
  size_t max_len = 0;
  int result;
  {
    size_t total_bytes = 0;
    for (int64_t d = 0; d < n_docs; d++) {
      int64_t ds = doc_span_offsets->a[d];
      int64_t de = doc_span_offsets->a[d + 1];
      int64_t nctx = ctx_off->a[d + 1] - ctx_off->a[d];
      size_t per = (size_t)(((collapse == COL_ALL || collapse == COL_CONTEXT) ? 0 : (int64_t)text_lens[d])
                            + 3 + 2 * nctx + (terminals ? 2 : 0));
      total_bytes += (size_t)(de - ds) * per;
    }
    char *pool = (char *)malloc(total_bytes ? total_bytes : 1);
    char *p = pool;
    for (int64_t d = 0; d < n_docs; d++) {
      int64_t ds = doc_span_offsets->a[d];
      int64_t de = doc_span_offsets->a[d + 1];
      int64_t c0 = ctx_off->a[d], c1 = ctx_off->a[d + 1];
      const char *text = text_ptrs[d];
      size_t tlen = text_lens[d];
      for (int64_t i = ds; i < de; i++) {
        size_t s = (size_t)span_starts->a[i];
        size_t e = (size_t)span_ends->a[i];
        int64_t ti = span_types ? span_types->a[i] : 0;
        size_t w = 0;
        if (terminals) p[w++] = (char)M_START;
        if (collapse == COL_NONE) {
          memcpy(p + w, text, s); w += s;
          p[w++] = (char)M_OPEN(ti);
          memcpy(p + w, text + s, e - s); w += e - s;
          p[w++] = (char)M_CLOSE(ti);
          memcpy(p + w, text + e, tlen - e); w += tlen - e;
        } else if (collapse == COL_FOCUS) {
          memcpy(p + w, text, s); w += s;
          p[w++] = (char)M_FOCUS;
          p[w++] = (char)M_OPEN(ti);
          memcpy(p + w, text + e, tlen - e); w += tlen - e;
        } else if (collapse == COL_CONTEXT) {
          // every context span (token) -> its type marker, including those inside the focus; the focus
          // span's tokens are bracketed by M_FOCUS. Text is dropped: the candidate becomes its tag-type
          // signature within the tagged sentence.
          bool fopen = false, fclose = false;
          for (int64_t cj = c0; cj < c1; cj++) {
            int64_t cstart = ctx_starts->a[cj];
            int64_t tc = ctx_types ? ctx_types->a[cj] : 0;
            if (!fopen && (size_t)cstart >= s) { p[w++] = (char)M_FOCUS; fopen = true; }
            if (fopen && !fclose && (size_t)cstart >= e) { p[w++] = (char)M_FOCUS; fclose = true; }
            p[w++] = (char)M_OPEN(tc);
          }
          if (fopen && !fclose) p[w++] = (char)M_FOCUS;
        } else {
          // mark / spans / all: backdrop from context spans, focus overlaid (focus wins on overlap)
          size_t pos = 0;
          bool fdone = false;
          for (int64_t cj = c0; cj <= c1; cj++) {
            int64_t cs = (cj < c1) ? ctx_starts->a[cj] : (int64_t)tlen;
            int64_t ce = (cj < c1) ? ctx_ends->a[cj] : (int64_t)tlen;
            int64_t tc = (cj < c1 && ctx_types) ? ctx_types->a[cj] : 0;
            if (!fdone && (cj == c1 || (size_t)cs >= s)) {
              if (collapse == COL_ALL) {
                p[w++] = (char)M_FOCUS; p[w++] = (char)M_OPEN(ti);
              } else {
                memcpy(p + w, text + pos, s - pos); w += s - pos;
                p[w++] = (char)M_FOCUS; p[w++] = (char)M_OPEN(ti);
                memcpy(p + w, text + s, e - s); w += e - s;
                p[w++] = (char)M_CLOSE(ti);
                pos = e;
              }
              fdone = true;
            }
            if (cj == c1) break;
            if ((size_t)cs < e && (size_t)ce > s) continue; // overlaps focus -> skip (focus wins)
            if (collapse == COL_ALL) {
              p[w++] = (char)M_OPEN(tc);
            } else if (collapse == COL_MARK) {
              if ((size_t)cs > pos) { memcpy(p + w, text + pos, (size_t)cs - pos); w += (size_t)cs - pos; }
              p[w++] = (char)M_OPEN(tc);
              memcpy(p + w, text + cs, (size_t)(ce - cs)); w += (size_t)(ce - cs);
              p[w++] = (char)M_CLOSE(tc);
              pos = (size_t)ce;
            } else { // COL_SPANS
              if ((size_t)cs > pos) { memcpy(p + w, text + pos, (size_t)cs - pos); w += (size_t)cs - pos; }
              p[w++] = (char)M_OPEN(tc);
              pos = (size_t)ce;
            }
          }
          if (collapse != COL_ALL && pos < tlen) {
            memcpy(p + w, text + pos, tlen - pos); w += tlen - pos;
          }
        }
        if (terminals) p[w++] = (char)M_END;
        strs[i] = p;
        lens[i] = w;
        if (w > max_len) max_len = w;
        p += w;
      }
    }
    free(text_ptrs);
    free(text_lens);
    result = tm_csr_tokenize_core(L, strs, lens, max_len, total_spans, ngram_min, ngram_max, do_normalize, false);
    free(pool);
  }
  #undef M_OPEN
  #undef M_CLOSE
  free(strs);
  free(lens);
  return result;
}

static int tm_csr_truncate (lua_State *L)
{
  tk_ivec_t *off = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *nbr = tk_ivec_peek(L, 2, "neighbors");
  tk_fvec_t *sco_f = tk_fvec_peekopt(L, 3);
  tk_dvec_t *sco_d = sco_f ? NULL : tk_dvec_peekopt(L, 3);
  int64_t k = (int64_t)luaL_checkinteger(L, 4);
  uint64_t ns = off->n - 1;
  tk_ivec_t *new_off = tk_ivec_create(L, ns + 1);
  new_off->n = ns + 1;
  int64_t total = 0;
  new_off->a[0] = 0;
  for (uint64_t i = 0; i < ns; i++) {
    int64_t rlen = off->a[i + 1] - off->a[i];
    if (rlen > k) rlen = k;
    total += rlen;
    new_off->a[i + 1] = total;
  }
  tk_ivec_t *new_nbr = tk_ivec_create(L, (uint64_t)total);
  new_nbr->n = (uint64_t)total;
  if (sco_f) {
    tk_fvec_t *new_sco = tk_fvec_create(L, (uint64_t)total);
    new_sco->n = (uint64_t)total;
    for (uint64_t i = 0; i < ns; i++) {
      int64_t src = off->a[i], dst = new_off->a[i];
      int64_t rlen = new_off->a[i + 1] - dst;
      memcpy(new_nbr->a + dst, nbr->a + src, (uint64_t)rlen * sizeof(int64_t));
      memcpy(new_sco->a + dst, sco_f->a + src, (uint64_t)rlen * sizeof(float));
    }
  } else if (sco_d) {
    tk_dvec_t *new_sco = tk_dvec_create(L, (uint64_t)total);
    new_sco->n = (uint64_t)total;
    for (uint64_t i = 0; i < ns; i++) {
      int64_t src = off->a[i], dst = new_off->a[i];
      int64_t rlen = new_off->a[i + 1] - dst;
      memcpy(new_nbr->a + dst, nbr->a + src, (uint64_t)rlen * sizeof(int64_t));
      memcpy(new_sco->a + dst, sco_d->a + src, (uint64_t)rlen * sizeof(double));
    }
  } else {
    for (uint64_t i = 0; i < ns; i++) {
      int64_t src = off->a[i], dst = new_off->a[i];
      int64_t rlen = new_off->a[i + 1] - dst;
      memcpy(new_nbr->a + dst, nbr->a + src, (uint64_t)rlen * sizeof(int64_t));
    }
  }
  return sco_f || sco_d ? 3 : 2;
}

static int tm_csr_sqrt (lua_State *L)
{
  tk_fvec_t *values = tk_fvec_peek(L, 1, "values");
  for (uint64_t i = 0; i < values->n; i++)
    values->a[i] = sqrtf(values->a[i]);
  lua_pushvalue(L, 1);
  return 1;
}

static inline double tm_probit (double p)
{
  if (p <= 0.0) return -1e10;
  if (p >= 1.0) return 1e10;
  static const double a[] = {
    -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
     1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
  };
  static const double b[] = {
    -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
     6.680131188771972e+01, -1.328068155288572e+01
  };
  static const double c[] = {
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
  };
  static const double d[] = {
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00
  };
  double plow = 0.02425, phigh = 1.0 - plow;
  double q, r;
  if (p < plow) {
    q = sqrt(-2.0 * log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
  } else if (p <= phigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
  } else {
    q = sqrt(-2.0 * log(1.0 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
  }
}

#define TM_SMOOTH_EPS 0.5

static inline double tm_bns (double N, double C, double P, double A)
{
  if (C <= 0 || C >= N || P <= 0 || P >= N) return 0.0;
  double tpr = (A + TM_SMOOTH_EPS) / (P + 2.0 * TM_SMOOTH_EPS);
  double fpr = (C - A + TM_SMOOTH_EPS) / (N - P + 2.0 * TM_SMOOTH_EPS);
  double bns = tm_probit(tpr) - tm_probit(fpr);
  return fabs(bns);
}

static inline void tm_csr_gather_mul (
  tk_fvec_t *values, tk_svec_t *tokens, tk_fvec_t *scores)
{
  for (uint64_t j = 0; j < values->n; j++)
    values->a[j] *= scores->a[tokens->a[j]];
}


static int tm_csr_apply_bns (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_svec_t *tokens = tk_svec_peek(L, 2, "tokens");
  tk_fvec_t *values = tk_fvec_peek(L, 3, "values");
  tk_fvec_t *scores = tk_fvec_peekopt(L, 4);
  if (scores) {
    tm_csr_gather_mul(values, tokens, scores);
    lua_pushvalue(L, 4);
    return 1;
  }
  tk_ivec_t *label_off = tk_ivec_peek(L, 5, "label_offsets");
  tk_ivec_t *label_nbr = tk_ivec_peek(L, 6, "label_neighbors");
  uint64_t n_tokens = tk_lua_checkunsigned(L, 7, "n_tokens");
  uint64_t n_labels = tk_lua_checkunsigned(L, 8, "n_labels");
  uint64_t n_samples = offsets->n - 1;
  double N = (double)n_samples;
  uint32_t *doc_freq = (uint32_t *)calloc(n_tokens, sizeof(uint32_t));
  uint32_t *label_freq = (uint32_t *)calloc(n_labels, sizeof(uint32_t));
  uint32_t *lbl_off = (uint32_t *)malloc((n_labels + 1) * sizeof(uint32_t));
  if (!doc_freq || !label_freq || !lbl_off) {
    free(doc_freq); free(label_freq); free(lbl_off);
    return luaL_error(L, "apply_bns: alloc failed");
  }
  for (uint64_t j = 0; j < tokens->n; j++)
    doc_freq[tokens->a[j]]++;
  for (uint64_t d = 0; d < n_samples; d++) {
    int64_t lo = label_off->a[d], hi = label_off->a[d + 1];
    for (int64_t j = lo; j < hi; j++) {
      uint64_t b = (uint64_t)label_nbr->a[j];
      if (b < n_labels) label_freq[b]++;
    }
  }
  lbl_off[0] = 0;
  for (uint64_t b = 0; b < n_labels; b++)
    lbl_off[b + 1] = lbl_off[b] + label_freq[b];
  uint32_t *lbl_docs = (uint32_t *)malloc((uint64_t)lbl_off[n_labels] * sizeof(uint32_t));
  uint32_t *lbl_pos = (uint32_t *)calloc(n_labels, sizeof(uint32_t));
  if (!lbl_docs || !lbl_pos) {
    free(doc_freq); free(label_freq); free(lbl_off);
    free(lbl_docs); free(lbl_pos);
    return luaL_error(L, "apply_bns: alloc failed");
  }
  for (uint64_t d = 0; d < n_samples; d++) {
    int64_t lo = label_off->a[d], hi = label_off->a[d + 1];
    for (int64_t j = lo; j < hi; j++) {
      uint64_t b = (uint64_t)label_nbr->a[j];
      if (b < n_labels) {
        lbl_docs[lbl_off[b] + lbl_pos[b]] = (uint32_t)d;
        lbl_pos[b]++;
      }
    }
  }
  free(lbl_pos);
  scores = tk_fvec_create(L, n_tokens);
  scores->n = n_tokens;
  memset(scores->a, 0, n_tokens * sizeof(float));
  float *cooc = (float *)calloc(n_tokens, sizeof(float));
  int32_t *touched = (int32_t *)malloc(n_tokens * sizeof(int32_t));
  if (!cooc || !touched) {
    free(doc_freq); free(label_freq); free(lbl_off); free(lbl_docs);
    free(cooc); free(touched);
    return luaL_error(L, "apply_bns: alloc failed");
  }
  for (uint64_t b = 0; b < n_labels; b++) {
    double P = (double)label_freq[b];
    if (P <= 0.0 || P >= N) continue;
    uint32_t n_touched = 0;
    for (uint32_t di = lbl_off[b]; di < lbl_off[b + 1]; di++) {
      uint32_t dd = lbl_docs[di];
      int64_t lo = offsets->a[dd], hi = offsets->a[dd + 1];
      for (int64_t j = lo; j < hi; j++) {
        int32_t f = (int32_t)tokens->a[j];
        if (cooc[f] == 0.0f) touched[n_touched++] = f;
        cooc[f] += 1.0f;
      }
    }
    for (uint32_t i = 0; i < n_touched; i++) {
      int32_t f = touched[i];
      float w = (float)tm_bns(N, (double)doc_freq[f], P, (double)cooc[f]);
      if (w > scores->a[f]) scores->a[f] = w;
      cooc[f] = 0.0f;
    }
  }
  free(cooc); free(touched);
  free(doc_freq); free(label_freq);
  free(lbl_off); free(lbl_docs);
  tm_csr_gather_mul(values, tokens, scores);
  return 1;
}

static int tm_csr_apply_auc (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_svec_t *tokens = tk_svec_peek(L, 2, "tokens");
  tk_fvec_t *values = tk_fvec_peek(L, 3, "values");
  tk_fvec_t *scores = tk_fvec_peekopt(L, 4);
  if (scores) {
    tm_csr_gather_mul(values, tokens, scores);
    lua_pushvalue(L, 4);
    return 1;
  }
  tk_dvec_t *targets = tk_dvec_peek(L, 5, "targets");
  uint64_t n_tokens = tk_lua_checkunsigned(L, 6, "n_tokens");
  uint64_t n_hidden = tk_lua_checkunsigned(L, 7, "n_hidden");
  uint64_t n_samples = offsets->n - 1;
  uint32_t *doc_freq = (uint32_t *)calloc(n_tokens, sizeof(uint32_t));
  if (!doc_freq) return luaL_error(L, "apply_auc: alloc failed");
  for (uint64_t j = 0; j < tokens->n; j++)
    doc_freq[tokens->a[j]]++;
  scores = tk_fvec_create(L, n_tokens);
  scores->n = n_tokens;
  memset(scores->a, 0, n_tokens * sizeof(float));
  tk_rank_t *pairs = (tk_rank_t *)malloc(n_samples * sizeof(tk_rank_t));
  double *ranks = (double *)malloc(n_samples * sizeof(double));
  double *rank_sums = (double *)calloc(n_tokens, sizeof(double));
  if (!pairs || !ranks || !rank_sums) {
    free(doc_freq); free(pairs); free(ranks); free(rank_sums);
    return luaL_error(L, "apply_auc: alloc failed");
  }
  for (uint64_t h = 0; h < n_hidden; h++) {
    for (uint64_t s = 0; s < n_samples; s++) {
      pairs[s].i = (int64_t)s;
      pairs[s].d = targets->a[s * n_hidden + h];
    }
    ks_introsort(tk_rvec_asc, (size_t)n_samples, pairs);
    uint64_t i = 0;
    while (i < n_samples) {
      uint64_t j = i + 1;
      while (j < n_samples && pairs[j].d == pairs[i].d) j++;
      double avg_rank = (double)(i + 1 + j) / 2.0;
      for (uint64_t k = i; k < j; k++)
        ranks[pairs[k].i] = avg_rank;
      i = j;
    }
    memset(rank_sums, 0, n_tokens * sizeof(double));
    for (uint64_t dd = 0; dd < n_samples; dd++) {
      double rd = ranks[dd];
      int64_t lo = offsets->a[dd], hi = offsets->a[dd + 1];
      for (int64_t j = lo; j < hi; j++)
        rank_sums[tokens->a[j]] += rd;
    }
    for (uint64_t f = 0; f < n_tokens; f++) {
      uint32_t n1 = doc_freq[f];
      if (n1 == 0) continue;
      uint32_t n0 = (uint32_t)n_samples - n1;
      if (n0 == 0) continue;
      double auc = (rank_sums[f] - (double)n1 * ((double)n1 + 1.0) / 2.0)
                 / ((double)n1 * (double)n0);
      auc = (auc + TM_SMOOTH_EPS) / (1.0 + 2.0 * TM_SMOOTH_EPS);
      float score = (float)(fabs(tm_probit(auc)) * M_SQRT2);
      if (score > scores->a[f]) scores->a[f] = score;
    }
  }
  free(pairs); free(ranks); free(rank_sums); free(doc_freq);
  tm_csr_gather_mul(values, tokens, scores);
  return 1;
}

static int tm_csr_apply_idf (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_svec_t *tokens = tk_svec_peek(L, 2, "tokens");
  tk_fvec_t *values = tk_fvec_peek(L, 3, "values");
  tk_fvec_t *scores = tk_fvec_peekopt(L, 4);
  if (scores) {
    tm_csr_gather_mul(values, tokens, scores);
    lua_pushvalue(L, 4);
    return 1;
  }
  uint64_t n_tokens = tk_lua_checkunsigned(L, 5, "n_tokens");
  uint64_t n_samples = offsets->n - 1;
  double N = (double)n_samples;
  uint32_t *doc_freq = (uint32_t *)calloc(n_tokens, sizeof(uint32_t));
  if (!doc_freq) return luaL_error(L, "apply_idf: alloc failed");
  for (uint64_t j = 0; j < tokens->n; j++)
    doc_freq[tokens->a[j]]++;
  scores = tk_fvec_create(L, n_tokens);
  scores->n = n_tokens;
  for (uint64_t t = 0; t < n_tokens; t++) {
    double df = (double)doc_freq[t];
    scores->a[t] = (float)log((N - df + 0.5) / (df + 0.5));
  }
  free(doc_freq);
  tm_csr_gather_mul(values, tokens, scores);
  return 1;
}

static int tm_csr_gather_rows (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_svec_t *tokens = tk_svec_peek(L, 2, "tokens");
  tk_fvec_t *values = tk_fvec_peek(L, 3, "values");
  tk_ivec_t *indices = tk_ivec_peek(L, 4, "indices");
  uint64_t n_out = indices->n;
  int64_t total = 0;
  for (uint64_t i = 0; i < n_out; i++) {
    int64_t idx = indices->a[i];
    total += offsets->a[idx + 1] - offsets->a[idx];
  }
  tk_ivec_t *out_off = tk_ivec_create(L, n_out + 1);
  out_off->n = n_out + 1;
  tk_svec_t *out_tok = tk_svec_create(L, (uint64_t)total);
  out_tok->n = (uint64_t)total;
  tk_fvec_t *out_val = tk_fvec_create(L, (uint64_t)total);
  out_val->n = (uint64_t)total;
  out_off->a[0] = 0;
  int64_t pos = 0;
  for (uint64_t i = 0; i < n_out; i++) {
    int64_t idx = indices->a[i];
    int64_t s = offsets->a[idx], e = offsets->a[idx + 1];
    int64_t len = e - s;
    memcpy(out_tok->a + pos, tokens->a + s, (uint64_t)len * sizeof(int32_t));
    memcpy(out_val->a + pos, values->a + s, (uint64_t)len * sizeof(float));
    pos += len;
    out_off->a[i + 1] = pos;
  }
  return 3;
}

// binary_label_csr(lab) -> off, nbr
// lab: ivec length n with 0/1 indicators. Builds the binary label CSR: off[n+1] is the
// exclusive prefix count of positives, nbr[npos] is all zeros (single label 0 per positive).
static int tm_csr_binary_label_csr (lua_State *L)
{
  tk_ivec_t *lab = tk_ivec_peek(L, 1, "lab");
  uint64_t n = lab->n;
  int64_t npos = 0;
  for (uint64_t i = 0; i < n; i++)
    if (lab->a[i] != 0) npos++;
  tk_ivec_t *off = tk_ivec_create(L, n + 1);
  off->n = n + 1;
  tk_ivec_t *nbr = tk_ivec_create(L, (uint64_t) npos);
  nbr->n = (uint64_t) npos;
  int64_t c = 0;
  off->a[0] = 0;
  for (uint64_t i = 0; i < n; i++) {
    if (lab->a[i] != 0) { nbr->a[c] = 0; c++; }
    off->a[i + 1] = c;
  }
  return 2;
}

// filter_spans(doc_off, starts, ends, types, mask) -> coff, cs, ce, cty
// Per doc d (span range [doc_off[d], doc_off[d+1])), keep spans where mask[j] != 0, emitting
// their start/end/type compacted, with per-doc offsets coff[n_docs+1]. types is optional (0 when nil).
static int tm_csr_filter_spans (lua_State *L)
{
  tk_ivec_t *doc_off = tk_ivec_peek(L, 1, "doc_offsets");
  tk_ivec_t *starts = tk_ivec_peek(L, 2, "starts");
  tk_ivec_t *ends = tk_ivec_peek(L, 3, "ends");
  tk_ivec_t *types = tk_ivec_peekopt(L, 4);
  tk_ivec_t *mask = tk_ivec_peek(L, 5, "mask");
  int64_t n_docs = (int64_t) (doc_off->n - 1);
  int64_t total = doc_off->a[n_docs];
  int64_t kept = 0;
  for (int64_t j = 0; j < total; j++)
    if (mask->a[j] != 0) kept++;
  tk_ivec_t *coff = tk_ivec_create(L, (uint64_t) (n_docs + 1));
  coff->n = (uint64_t) (n_docs + 1);
  tk_ivec_t *cs = tk_ivec_create(L, (uint64_t) kept);
  cs->n = (uint64_t) kept;
  tk_ivec_t *ce = tk_ivec_create(L, (uint64_t) kept);
  ce->n = (uint64_t) kept;
  tk_ivec_t *cty = tk_ivec_create(L, (uint64_t) kept);
  cty->n = (uint64_t) kept;
  int64_t c = 0;
  coff->a[0] = 0;
  for (int64_t d = 0; d < n_docs; d++) {
    for (int64_t j = doc_off->a[d]; j < doc_off->a[d + 1]; j++) {
      if (mask->a[j] != 0) {
        cs->a[c] = starts->a[j];
        ce->a[c] = ends->a[j];
        cty->a[c] = types ? types->a[j] : 0;
        c++;
      }
    }
    coff->a[d + 1] = c;
  }
  return 4;
}

// bio_encode(doc_off, tok_starts, tok_ends, ent_off, ent_starts, ent_ends, ent_types, n_types)
//   -> lab_off, lab_nbr
// Per token, assign a BIO x type class from its doc's entity char-spans:
//   O = 0; B-<t> = 1 + t; I-<t> = 1 + n_types + t.  (n_classes = 2*n_types + 1)
// lab_off is the stride-1 single-label CSR (one class per token). ent_types optional (0 if nil).
static int tm_csr_bio_encode (lua_State *L)
{
  tk_ivec_t *doc_off = tk_ivec_peek(L, 1, "doc_offsets");
  tk_ivec_t *tok_s = tk_ivec_peek(L, 2, "tok_starts");
  tk_ivec_t *tok_e = tk_ivec_peek(L, 3, "tok_ends");
  tk_ivec_t *ent_off = tk_ivec_peek(L, 4, "ent_offsets");
  tk_ivec_t *ent_s = tk_ivec_peek(L, 5, "ent_starts");
  tk_ivec_t *ent_e = tk_ivec_peek(L, 6, "ent_ends");
  tk_ivec_t *ent_ty = tk_ivec_peekopt(L, 7);
  int64_t n_types = tk_lua_checkinteger(L, 8, "n_types");
  int64_t n_docs = (int64_t) (doc_off->n - 1);
  int64_t n_tok = doc_off->a[n_docs];
  tk_ivec_t *lab_off = tk_ivec_create(L, (uint64_t) (n_tok + 1));
  lab_off->n = (uint64_t) (n_tok + 1);
  tk_ivec_t *lab_nbr = tk_ivec_create(L, (uint64_t) n_tok);
  lab_nbr->n = (uint64_t) n_tok;
  for (int64_t i = 0; i <= n_tok; i++) lab_off->a[i] = i;
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t e0 = ent_off->a[d], e1 = ent_off->a[d + 1];
    for (int64_t j = doc_off->a[d]; j < doc_off->a[d + 1]; j++) {
      int64_t ts = tok_s->a[j], te = tok_e->a[j];
      int64_t lab = 0;
      for (int64_t k = e0; k < e1; k++) {
        int64_t es = ent_s->a[k], ee = ent_e->a[k];
        int64_t typ = ent_ty ? ent_ty->a[k] : 0;
        if (ts == es) { lab = 1 + typ; break; }
        else if (ts > es && te <= ee) { lab = 1 + n_types + typ; break; }
      }
      lab_nbr->a[j] = lab;
    }
  }
  return 2;
}

// bio_decode(doc_off, tok_starts, tok_ends, classes, n_types) -> ent_off, ent_starts, ent_ends, ent_types
// Inverse of bio_encode: collapse per-token classes into entity spans (deterministic run-merge).
static int tm_csr_bio_decode (lua_State *L)
{
  tk_ivec_t *doc_off = tk_ivec_peek(L, 1, "doc_offsets");
  tk_ivec_t *tok_s = tk_ivec_peek(L, 2, "tok_starts");
  tk_ivec_t *tok_e = tk_ivec_peek(L, 3, "tok_ends");
  tk_ivec_t *cls = tk_ivec_peek(L, 4, "classes");
  int64_t n_types = tk_lua_checkinteger(L, 5, "n_types");
  int64_t n_docs = (int64_t) (doc_off->n - 1);
  tk_ivec_t *ent_off = tk_ivec_create(L, (uint64_t) (n_docs + 1));
  ent_off->n = (uint64_t) (n_docs + 1);
  tk_ivec_t *ent_s = tk_ivec_create(L, 0);
  tk_ivec_t *ent_e = tk_ivec_create(L, 0);
  tk_ivec_t *ent_ty = tk_ivec_create(L, 0);
  ent_off->a[0] = 0;
  int64_t ct, cs, ce;
  #define TK_BIO_FLUSH() do { \
    if (ct >= 0) { tk_ivec_push(ent_s, cs); tk_ivec_push(ent_e, ce); tk_ivec_push(ent_ty, ct); ct = -1; } \
  } while (0)
  for (int64_t d = 0; d < n_docs; d++) {
    ct = -1; cs = 0; ce = 0;
    for (int64_t j = doc_off->a[d]; j < doc_off->a[d + 1]; j++) {
      int64_t lab = cls->a[j];
      int64_t ts = tok_s->a[j], te = tok_e->a[j];
      if (lab >= 1 && lab <= n_types) {
        TK_BIO_FLUSH(); ct = lab - 1; cs = ts; ce = te;
      } else if (lab > n_types) {
        int64_t typ = lab - n_types - 1;
        if (ct == typ) { ce = te; }
        else { TK_BIO_FLUSH(); ct = typ; cs = ts; ce = te; }
      } else {
        TK_BIO_FLUSH();
      }
    }
    TK_BIO_FLUSH();
    ent_off->a[d + 1] = (int64_t) ent_s->n;
  }
  #undef TK_BIO_FLUSH
  return 4;
}

// bio_viterbi(doc_off, score_off, score_nbr, score_val, n_types) -> classes
// Hard-constraint linear-chain decode: per doc, the best VALID BIO x type label sequence maximizing
// summed emission scores subject to BIO validity (I-<t> may only follow B-<t> or I-<t>; an entity
// can't start with I). Emissions come from ridge:label(codes, n, n_classes): score_off/nbr/val is
// that top-(n_classes) CSR (sorted, scattered back to class order here). Returns one class per token.
static int tm_csr_bio_viterbi (lua_State *L)
{
  tk_ivec_t *doc_off = tk_ivec_peek(L, 1, "doc_offsets");
  tk_ivec_t *s_off = tk_ivec_peek(L, 2, "score_offsets");
  tk_ivec_t *s_nbr = tk_ivec_peek(L, 3, "score_neighbors");
  tk_fvec_t *s_val = tk_fvec_peek(L, 4, "score_values");
  int64_t n_types = tk_lua_checkinteger(L, 5, "n_types");
  int64_t nc = 2 * n_types + 1;
  int64_t n_docs = (int64_t) (doc_off->n - 1);
  int64_t n_tok = doc_off->a[n_docs];
  tk_ivec_t *out = tk_ivec_create(L, (uint64_t) n_tok);
  out->n = (uint64_t) n_tok;
  int64_t maxT = 0;
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t t = doc_off->a[d + 1] - doc_off->a[d];
    if (t > maxT) maxT = t;
  }
  if (maxT == 0) return 1;
  const double NEG = -1e30;
  double *emis = (double *) malloc((size_t) (maxT * nc) * sizeof(double));
  double *dp = (double *) malloc((size_t) (maxT * nc) * sizeof(double));
  int64_t *bp = (int64_t *) malloc((size_t) (maxT * nc) * sizeof(int64_t));
  if (!emis || !dp || !bp) {
    free(emis); free(dp); free(bp);
    return luaL_error(L, "bio_viterbi: alloc failed");
  }
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t t0 = doc_off->a[d], T = doc_off->a[d + 1] - t0;
    if (T == 0) continue;
    for (int64_t i = 0; i < T; i++) {
      int64_t j = t0 + i;
      for (int64_t c = 0; c < nc; c++) emis[i * nc + c] = NEG;
      for (int64_t k = s_off->a[j]; k < s_off->a[j + 1]; k++) {
        int64_t c = s_nbr->a[k];
        if (c >= 0 && c < nc) emis[i * nc + c] = (double) s_val->a[k];
      }
    }
    for (int64_t c = 0; c < nc; c++) {       // t = 0: from virtual O (no I start)
      dp[c] = (c > n_types) ? NEG : emis[c];
      bp[c] = -1;
    }
    for (int64_t i = 1; i < T; i++) {
      for (int64_t c = 0; c < nc; c++) {
        double best = NEG; int64_t barg = -1;
        for (int64_t a = 0; a < nc; a++) {
          if (dp[(i - 1) * nc + a] <= NEG) continue;
          if (c > n_types) {                 // c = I-tc: only B-tc or I-tc may precede
            int64_t tc = c - n_types - 1;
            if (a != 1 + tc && a != 1 + n_types + tc) continue;
          }
          double v = dp[(i - 1) * nc + a];
          if (v > best) { best = v; barg = a; }
        }
        dp[i * nc + c] = (best <= NEG) ? NEG : best + emis[i * nc + c];
        bp[i * nc + c] = barg;
      }
    }
    int64_t cur = 0; double bv = NEG;
    for (int64_t c = 0; c < nc; c++)
      if (dp[(T - 1) * nc + c] > bv) { bv = dp[(T - 1) * nc + c]; cur = c; }
    for (int64_t i = T - 1; i >= 0; i--) {
      out->a[t0 + i] = cur;
      if (i > 0) { cur = bp[i * nc + cur]; if (cur < 0) cur = 0; }
    }
  }
  free(emis); free(dp); free(bp);
  return 1;
}

// bio_band_mask(labels, lo, hi) -> mask : token-wise 0/1, 1 iff lo <= label <= hi. Used to pull the
// B-class band [1, n_types] out of a bio_encode label vector (O=0, B=1..nt, I=nt+1..2nt) for the
// segment stage's B targets.
static int tm_csr_bio_band_mask (lua_State *L)
{
  tk_ivec_t *lab = tk_ivec_peek(L, 1, "labels");
  int64_t lo = tk_lua_checkinteger(L, 2, "lo");
  int64_t hi = tk_lua_checkinteger(L, 3, "hi");
  int64_t n = (int64_t) lab->n;
  tk_ivec_t *out = tk_ivec_create(L, (uint64_t) n);
  out->n = (uint64_t) n;
  for (int64_t i = 0; i < n; i++) {
    int64_t v = lab->a[i];
    out->a[i] = (v >= lo && v <= hi) ? 1 : 0;
  }
  return 1;
}

// bio_token_type(bio_labels, n_types) -> per-token class for the tag stage: the entity type (0..n_types-1,
// dropping the B/I distinction) for an entity token, else n_types (= O / outer). bio_encode scheme is
// O=0, B-t=1..n_types, I-t=n_types+1..2*n_types.
static int tm_csr_bio_token_type (lua_State *L)
{
  tk_ivec_t *lab = tk_ivec_peek(L, 1, "bio_labels");
  int64_t nt = tk_lua_checkinteger(L, 2, "n_types");
  int64_t n = (int64_t) lab->n;
  tk_ivec_t *out = tk_ivec_create(L, (uint64_t) n);
  out->n = (uint64_t) n;
  for (int64_t i = 0; i < n; i++) {
    int64_t v = lab->a[i];
    out->a[i] = (v == 0) ? nt : (v <= nt ? v - 1 : v - nt - 1);
  }
  return 1;
}

// enumerate_subspans(doc_offsets, tok_starts, tok_ends, tok_types, max_span, outer) -> off, starts, ends:
// within each maximal run of contiguous non-outer tokens, emit every contiguous subspan of 1..max_span
// tokens (as char start..end). The tag stage over-generates candidate spans; the type stage prunes them.
static int tm_csr_enumerate_subspans (lua_State *L)
{
  tk_ivec_t *doc_off = tk_ivec_peek(L, 1, "doc_offsets");
  tk_ivec_t *ts = tk_ivec_peek(L, 2, "tok_starts");
  tk_ivec_t *te = tk_ivec_peek(L, 3, "tok_ends");
  tk_ivec_t *ty = tk_ivec_peek(L, 4, "tok_types");
  int64_t max_span = tk_lua_checkinteger(L, 5, "max_span");
  int64_t outer = tk_lua_checkinteger(L, 6, "outer");
  int64_t n_docs = (int64_t) (doc_off->n - 1);
  tk_ivec_t *coff = tk_ivec_create(L, (uint64_t) (n_docs + 1));
  coff->n = (uint64_t) (n_docs + 1);
  tk_ivec_t *cs = tk_ivec_create(L, 0);
  tk_ivec_t *ce = tk_ivec_create(L, 0);
  coff->a[0] = 0;
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t j = doc_off->a[d], end = doc_off->a[d + 1];
    while (j < end) {
      if (ty->a[j] == outer) { j++; continue; }
      int64_t r0 = j;
      while (j < end && ty->a[j] != outer) j++;
      int64_t r1 = j;
      for (int64_t i = r0; i < r1; i++)
        for (int64_t len = 1; len <= max_span && i + len <= r1; len++) {
          tk_ivec_push(cs, ts->a[i]);
          tk_ivec_push(ce, te->a[i + len - 1]);
        }
    }
    coff->a[d + 1] = (int64_t) cs->n;
  }
  return 3;
}

// type_labels(cand_off,cand_s,cand_e, gold_off,gold_s,gold_e,gold_ty, n_types) -> class id per candidate:
// the gold type if (start,end) matches a gold span in the same doc, else reject (= n_types). One label
// per untyped candidate for the (n_types+1)-class type head.
static int tm_csr_type_labels (lua_State *L)
{
  tk_ivec_t *co = tk_ivec_peek(L, 1, "cand_offsets");
  tk_ivec_t *cs = tk_ivec_peek(L, 2, "cand_starts");
  tk_ivec_t *ce = tk_ivec_peek(L, 3, "cand_ends");
  tk_ivec_t *go = tk_ivec_peek(L, 4, "gold_offsets");
  tk_ivec_t *gs = tk_ivec_peek(L, 5, "gold_starts");
  tk_ivec_t *ge = tk_ivec_peek(L, 6, "gold_ends");
  tk_ivec_t *gt = tk_ivec_peek(L, 7, "gold_types");
  int64_t n_types = tk_lua_checkinteger(L, 8, "n_types");
  int64_t n_docs = (int64_t) (co->n - 1);
  int64_t ncand = (int64_t) cs->n;
  tk_ivec_t *out = tk_ivec_create(L, (uint64_t) ncand);
  out->n = (uint64_t) ncand;
  for (int64_t d = 0; d < n_docs; d++) {
    for (int64_t c = co->a[d]; c < co->a[d + 1]; c++) {
      int64_t a = cs->a[c], b = ce->a[c], lab = n_types;
      for (int64_t g = go->a[d]; g < go->a[d + 1]; g++)
        if (gs->a[g] == a && ge->a[g] == b) { lab = gt->a[g]; break; }
      out->a[c] = lab;
    }
  }
  return 1;
}

// nms(cand_off, cand_s, cand_e, classes, scores, reject) -> keep mask (0/1): per doc, greedily keep the
// highest-scoring non-reject candidate, suppress any candidate that overlaps a kept one (type-agnostic,
// flat NER; [s1,e1) overlaps [s2,e2) iff s1<e2 && s2<e1), repeat. Reject candidates are never kept.
static int tm_csr_nms (lua_State *L)
{
  tk_ivec_t *co = tk_ivec_peek(L, 1, "cand_offsets");
  tk_ivec_t *cs = tk_ivec_peek(L, 2, "cand_starts");
  tk_ivec_t *ce = tk_ivec_peek(L, 3, "cand_ends");
  tk_ivec_t *cls = tk_ivec_peek(L, 4, "classes");
  tk_fvec_t *sco = tk_fvec_peek(L, 5, "scores");
  int64_t reject = tk_lua_checkinteger(L, 6, "reject");
  int64_t n_docs = (int64_t) (co->n - 1);
  int64_t ncand = (int64_t) cs->n;
  tk_ivec_t *keep = tk_ivec_create(L, (uint64_t) ncand);
  keep->n = (uint64_t) ncand;
  for (int64_t i = 0; i < ncand; i++) keep->a[i] = 0;
  char *done = (char *) calloc(ncand ? (size_t) ncand : 1, 1);
  if (!done) return luaL_error(L, "nms: alloc failed");
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t lo = co->a[d], hi = co->a[d + 1];
    for (int64_t i = lo; i < hi; i++) if (cls->a[i] == reject) done[i] = 1;
    for (;;) {
      int64_t best = -1; double bestsco = 0;
      for (int64_t i = lo; i < hi; i++) {
        if (done[i]) continue;
        double s = (double) sco->a[i];
        if (best < 0 || s > bestsco) { best = i; bestsco = s; }
      }
      if (best < 0) break;
      keep->a[best] = 1; done[best] = 1;
      int64_t bs = cs->a[best], be = ce->a[best];
      for (int64_t i = lo; i < hi; i++)
        if (!done[i] && cs->a[i] < be && bs < ce->a[i]) done[i] = 1;
    }
  }
  free(done);
  return 1;
}

typedef struct { int64_t s, e, oi; double w; } tk_wis_iv;
static int tk_wis_cmp (const void *a, const void *b) {
  int64_t ea = ((const tk_wis_iv *) a)->e, eb = ((const tk_wis_iv *) b)->e;
  return (ea < eb) ? -1 : (ea > eb) ? 1 : 0;
}

// nms_dp(cand_off, cand_s, cand_e, labels, scores, k, reject) -> keep mask, argmax class: labels/scores are
// ridge:label top-k output (k>=2). Per candidate the argmax class is labels[c*k] and the DP weight is the
// margin scores[c*k]-scores[c*k+1] (>=0). Per doc this returns the max-total-margin set of NON-OVERLAPPING
// non-reject candidates (weighted interval scheduling, optimal) -- unlike greedy nms it can keep two short
// correct spans over one higher-scoring wrong long one. Compatible iff end[j] <= start[i].
static int tm_csr_nms_dp (lua_State *L)
{
  tk_ivec_t *co = tk_ivec_peek(L, 1, "cand_offsets");
  tk_ivec_t *cs = tk_ivec_peek(L, 2, "cand_starts");
  tk_ivec_t *ce = tk_ivec_peek(L, 3, "cand_ends");
  tk_ivec_t *lab = tk_ivec_peek(L, 4, "labels");
  tk_fvec_t *sco = tk_fvec_peek(L, 5, "scores");
  int64_t k = tk_lua_checkinteger(L, 6, "k");
  int64_t reject = tk_lua_checkinteger(L, 7, "reject");
  int64_t n_docs = (int64_t) (co->n - 1);
  int64_t ncand = (int64_t) cs->n;
  tk_ivec_t *keep = tk_ivec_create(L, (uint64_t) ncand);
  keep->n = (uint64_t) ncand;
  tk_ivec_t *cls = tk_ivec_create(L, (uint64_t) ncand);
  cls->n = (uint64_t) ncand;
  for (int64_t c = 0; c < ncand; c++) { keep->a[c] = 0; cls->a[c] = lab->a[c * k]; }
  int64_t maxn = 0;
  for (int64_t d = 0; d < n_docs; d++) { int64_t n = co->a[d + 1] - co->a[d]; if (n > maxn) maxn = n; }
  if (maxn == 0) return 2;
  tk_wis_iv *iv = (tk_wis_iv *) malloc((size_t) maxn * sizeof(tk_wis_iv));
  double *M = (double *) malloc((size_t) (maxn + 1) * sizeof(double));
  int64_t *P = (int64_t *) malloc((size_t) (maxn + 1) * sizeof(int64_t));
  if (!iv || !M || !P) { free(iv); free(M); free(P); return luaL_error(L, "nms_dp: alloc failed"); }
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t m = 0;
    for (int64_t c = co->a[d]; c < co->a[d + 1]; c++) {
      if (lab->a[c * k] == reject) continue;
      iv[m].s = cs->a[c]; iv[m].e = ce->a[c]; iv[m].oi = c;
      iv[m].w = (double) sco->a[c * k] - (double) sco->a[c * k + 1];
      m++;
    }
    if (m == 0) continue;
    qsort(iv, (size_t) m, sizeof(tk_wis_iv), tk_wis_cmp);
    M[0] = 0.0;
    for (int64_t i = 1; i <= m; i++) {
      int64_t target = iv[i - 1].s, lo = 0, hi = i - 1;
      while (lo < hi) { int64_t mid = (lo + hi) / 2; if (iv[mid].e <= target) lo = mid + 1; else hi = mid; }
      P[i] = lo;
      double take = iv[i - 1].w + M[P[i]];
      M[i] = (take >= M[i - 1]) ? take : M[i - 1];
    }
    int64_t i = m;
    while (i >= 1) {
      double take = iv[i - 1].w + M[P[i]];
      if (take >= M[i - 1]) { keep->a[iv[i - 1].oi] = 1; i = P[i]; }
      else i = i - 1;
    }
  }
  free(iv); free(M); free(P);
  return 2;
}

// union_spans{ a_offsets,a_starts,a_ends,a_types, b_offsets,b_starts,b_ends,b_types,
//   gold_offsets,gold_starts,gold_ends,gold_types } -> off, starts, ends, types, labels
// Per doc, the deduped union of candidate sets A and B (matched on (start,end,type)), each candidate
// labeled 1 if it matches a gold span in that doc else 0. Spans/doc are small, so a nested scan is fine.
static int tm_csr_union_spans (lua_State *L)
{
  luaL_checktype(L, 1, LUA_TTABLE);
  tk_ivec_t *fields[12];
  const char *names[12] = {
    "a_offsets", "a_starts", "a_ends", "a_types",
    "b_offsets", "b_starts", "b_ends", "b_types",
    "gold_offsets", "gold_starts", "gold_ends", "gold_types" };
  for (int i = 0; i < 12; i++) {
    lua_getfield(L, 1, names[i]);
    fields[i] = tk_ivec_peek(L, -1, names[i]);
    lua_pop(L, 1);
  }
  tk_ivec_t *ao = fields[0], *as = fields[1], *ae = fields[2], *at = fields[3];
  tk_ivec_t *bo = fields[4], *bs = fields[5], *be = fields[6], *bt = fields[7];
  tk_ivec_t *go = fields[8], *gs = fields[9], *ge = fields[10], *gt = fields[11];
  int64_t n_docs = (int64_t) (ao->n - 1);
  tk_ivec_t *uoff = tk_ivec_create(L, (uint64_t) (n_docs + 1));
  uoff->n = (uint64_t) (n_docs + 1);
  tk_ivec_t *us = tk_ivec_create(L, 0);
  tk_ivec_t *ue = tk_ivec_create(L, 0);
  tk_ivec_t *ut = tk_ivec_create(L, 0);
  tk_ivec_t *ulab = tk_ivec_create(L, 0);
  uoff->a[0] = 0;
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t doc_start = (int64_t) us->n;
    for (int pass = 0; pass < 2; pass++) {
      tk_ivec_t *po = pass ? bo : ao, *ps = pass ? bs : as, *pe = pass ? be : ae, *pt = pass ? bt : at;
      for (int64_t j = po->a[d]; j < po->a[d + 1]; j++) {
        int64_t sj = ps->a[j], ej = pe->a[j], tj = pt->a[j];
        int dup = 0;
        for (int64_t k = doc_start; k < (int64_t) us->n; k++)
          if (us->a[k] == sj && ue->a[k] == ej && ut->a[k] == tj) { dup = 1; break; }
        if (dup) continue;
        int64_t lab = 0;
        for (int64_t g = go->a[d]; g < go->a[d + 1]; g++)
          if (gs->a[g] == sj && ge->a[g] == ej && gt->a[g] == tj) { lab = 1; break; }
        tk_ivec_push(us, sj); tk_ivec_push(ue, ej); tk_ivec_push(ut, tj); tk_ivec_push(ulab, lab);
      }
    }
    uoff->a[d + 1] = (int64_t) us->n;
  }
  return 5;
}

// span_miss_report{ gaz_offsets,gaz_starts,gaz_ends,gaz_types,
//   bio_offsets,bio_starts,bio_ends,bio_types, gold_offsets,gold_starts,gold_ends,gold_types, n_types }
//   -> { gold, covered, wrong_type, over, under, cross, none,
//        under_gaz, under_bio, under_both, under_by_type = { [0..n_types-1] } }
// Decomposes why each gold span is/isn't recovered by the candidate pool (gaz UNION bio). Spans are
// [start,end). Per missed gold, the highest-priority relationship to any same-doc candidate is counted:
//   covered    exact (start,end,type) match
//   wrong_type exact (start,end), type differs  -> a type-only model recovers it
//   over       gold strictly inside a candidate -> sub-span enumeration recovers it
//   under      candidate strictly inside gold   -> merge / looser emit recovers it
//   cross      overlaps, neither contains        -> boundary error
//   none       no candidate overlaps             -> detector miss, unrecoverable by decode
// For the `under` bucket: which source(s) supplied the strictly-inside candidate (gaz / bio / both),
// and the per-gold-type counts, to localize whether under-spanning is gazetteer sub-surfaces or BIO
// terminating early, and whether it concentrates in the longer entity types.
static int tm_csr_span_miss_report (lua_State *L)
{
  luaL_checktype(L, 1, LUA_TTABLE);
  const char *names[12] = {
    "gaz_offsets", "gaz_starts", "gaz_ends", "gaz_types",
    "bio_offsets", "bio_starts", "bio_ends", "bio_types",
    "gold_offsets", "gold_starts", "gold_ends", "gold_types" };
  tk_ivec_t *f[12];
  for (int i = 0; i < 12; i++) {
    lua_getfield(L, 1, names[i]);
    f[i] = tk_ivec_peek(L, -1, names[i]);
    lua_pop(L, 1);
  }
  lua_getfield(L, 1, "n_types");
  int64_t n_types = tk_lua_checkinteger(L, -1, "n_types");
  lua_pop(L, 1);
  // [0]=gaz, [1]=bio for both candidate sources
  tk_ivec_t *so[2] = { f[0], f[4] }, *ss[2] = { f[1], f[5] }, *se[2] = { f[2], f[6] }, *st[2] = { f[3], f[7] };
  tk_ivec_t *go = f[8], *gs = f[9], *ge = f[10], *gt = f[11];
  int64_t n_docs = (int64_t) (go->n - 1);
  int64_t n_gold = 0, covered = 0, wrong = 0, over = 0, under = 0, cross = 0, none = 0;
  int64_t under_gaz = 0, under_bio = 0, under_both = 0;
  int64_t *under_ty = (int64_t *) calloc((size_t) (n_types > 0 ? n_types : 1), sizeof(int64_t));
  if (!under_ty) return luaL_error(L, "span_miss_report: alloc failed");
  for (int64_t d = 0; d < n_docs; d++) {
    for (int64_t g = go->a[d]; g < go->a[d + 1]; g++) {
      int64_t a = gs->a[g], b = ge->a[g], t = gt->a[g];
      n_gold++;
      int exact = 0, fwrong = 0, fover = 0, fcross = 0;
      int funder_s[2] = { 0, 0 };
      for (int src = 0; src < 2 && !exact; src++) {
        for (int64_t c = so[src]->a[d]; c < so[src]->a[d + 1]; c++) {
          int64_t x = ss[src]->a[c], y = se[src]->a[c];
          if (x >= b || a >= y) continue;          // no overlap
          if (x == a && y == b) {
            if (st[src]->a[c] == t) { exact = 1; break; }
            fwrong = 1;
          } else if (x <= a && b <= y) {
            fover = 1;
          } else if (a <= x && y <= b) {
            funder_s[src] = 1;
          } else {
            fcross = 1;
          }
        }
      }
      int funder = funder_s[0] || funder_s[1];
      if (exact) covered++;
      else if (fwrong) wrong++;
      else if (fover) over++;
      else if (funder) {
        under++;
        if (funder_s[0] && funder_s[1]) under_both++;
        else if (funder_s[0]) under_gaz++;
        else under_bio++;
        if (t >= 0 && t < n_types) under_ty[t]++;
      }
      else if (fcross) cross++;
      else none++;
    }
  }
  lua_newtable(L);
  lua_pushinteger(L, (lua_Integer) n_gold); lua_setfield(L, -2, "gold");
  lua_pushinteger(L, (lua_Integer) covered); lua_setfield(L, -2, "covered");
  lua_pushinteger(L, (lua_Integer) wrong); lua_setfield(L, -2, "wrong_type");
  lua_pushinteger(L, (lua_Integer) over); lua_setfield(L, -2, "over");
  lua_pushinteger(L, (lua_Integer) under); lua_setfield(L, -2, "under");
  lua_pushinteger(L, (lua_Integer) cross); lua_setfield(L, -2, "cross");
  lua_pushinteger(L, (lua_Integer) none); lua_setfield(L, -2, "none");
  lua_pushinteger(L, (lua_Integer) under_gaz); lua_setfield(L, -2, "under_gaz");
  lua_pushinteger(L, (lua_Integer) under_bio); lua_setfield(L, -2, "under_bio");
  lua_pushinteger(L, (lua_Integer) under_both); lua_setfield(L, -2, "under_both");
  lua_newtable(L);
  for (int64_t i = 0; i < n_types; i++) {
    lua_pushinteger(L, (lua_Integer) under_ty[i]);
    lua_rawseti(L, -2, (int) i);
  }
  lua_setfield(L, -2, "under_by_type");
  free(under_ty);
  return 1;
}

// type_decode_report{ cand_offsets, cand_starts, cand_ends, cand_pred, pred_stride,
//   gold_offsets, gold_starts, gold_ends, gold_types, n_types }
//   -> { gold, in_pool, not_in_pool, correct, false_reject, mistype,
//        correct_by_type={[0..n_types-1]}, reject_by_type={..}, mistype_by_type={..},
//        confusion={[t*n_types+L]} }
// For each gold span, finds the candidate with the exact (start,end) and inspects the TYPE head's
// top-1 prediction (cand_pred[c*pred_stride]; class == n_types means REJECT). Splits the conversion
// loss (golds present in the pool but not emitted correctly) into false_reject vs mistype, with a
// gold->pred confusion matrix to localize confusable type pairs. Golds with no exact candidate are
// not_in_pool (the coverage miss already decomposed by span_miss_report).
static int tm_csr_type_decode_report (lua_State *L)
{
  luaL_checktype(L, 1, LUA_TTABLE);
  const char *names[8] = {
    "cand_offsets", "cand_starts", "cand_ends", "cand_pred",
    "gold_offsets", "gold_starts", "gold_ends", "gold_types" };
  tk_ivec_t *f[8];
  for (int i = 0; i < 8; i++) {
    lua_getfield(L, 1, names[i]);
    f[i] = tk_ivec_peek(L, -1, names[i]);
    lua_pop(L, 1);
  }
  lua_getfield(L, 1, "n_types");
  int64_t n_types = tk_lua_checkinteger(L, -1, "n_types");
  lua_pop(L, 1);
  lua_getfield(L, 1, "pred_stride");
  int64_t stride = tk_lua_checkinteger(L, -1, "pred_stride");
  lua_pop(L, 1);
  tk_ivec_t *co = f[0], *cs = f[1], *ce = f[2], *cp = f[3];
  tk_ivec_t *go = f[4], *gs = f[5], *ge = f[6], *gt = f[7];
  int64_t n_docs = (int64_t) (go->n - 1);
  int64_t n_gold = 0, in_pool = 0, not_in_pool = 0, correct = 0, freject = 0, mistype = 0;
  size_t nt = (size_t) (n_types > 0 ? n_types : 1);
  int64_t *corr_ty = (int64_t *) calloc(nt, sizeof(int64_t));
  int64_t *rej_ty = (int64_t *) calloc(nt, sizeof(int64_t));
  int64_t *mis_ty = (int64_t *) calloc(nt, sizeof(int64_t));
  int64_t *conf = (int64_t *) calloc(nt * nt, sizeof(int64_t));
  if (!corr_ty || !rej_ty || !mis_ty || !conf) {
    free(corr_ty); free(rej_ty); free(mis_ty); free(conf);
    return luaL_error(L, "type_decode_report: alloc failed");
  }
  for (int64_t d = 0; d < n_docs; d++) {
    for (int64_t g = go->a[d]; g < go->a[d + 1]; g++) {
      int64_t a = gs->a[g], b = ge->a[g], t = gt->a[g];
      n_gold++;
      int64_t cmatch = -1;
      for (int64_t c = co->a[d]; c < co->a[d + 1]; c++)
        if (cs->a[c] == a && ce->a[c] == b) { cmatch = c; break; }
      if (cmatch < 0) { not_in_pool++; continue; }
      in_pool++;
      int64_t pred = cp->a[cmatch * stride];
      if (pred == n_types) { freject++; if (t >= 0 && t < n_types) rej_ty[t]++; }
      else if (pred == t) { correct++; if (t >= 0 && t < n_types) corr_ty[t]++; }
      else {
        mistype++;
        if (t >= 0 && t < n_types) {
          mis_ty[t]++;
          if (pred >= 0 && pred < n_types) conf[t * n_types + pred]++;
        }
      }
    }
  }
  lua_newtable(L);
  lua_pushinteger(L, (lua_Integer) n_gold); lua_setfield(L, -2, "gold");
  lua_pushinteger(L, (lua_Integer) in_pool); lua_setfield(L, -2, "in_pool");
  lua_pushinteger(L, (lua_Integer) not_in_pool); lua_setfield(L, -2, "not_in_pool");
  lua_pushinteger(L, (lua_Integer) correct); lua_setfield(L, -2, "correct");
  lua_pushinteger(L, (lua_Integer) freject); lua_setfield(L, -2, "false_reject");
  lua_pushinteger(L, (lua_Integer) mistype); lua_setfield(L, -2, "mistype");
  lua_newtable(L);
  for (int64_t i = 0; i < n_types; i++) { lua_pushinteger(L, (lua_Integer) corr_ty[i]); lua_rawseti(L, -2, (int) i); }
  lua_setfield(L, -2, "correct_by_type");
  lua_newtable(L);
  for (int64_t i = 0; i < n_types; i++) { lua_pushinteger(L, (lua_Integer) rej_ty[i]); lua_rawseti(L, -2, (int) i); }
  lua_setfield(L, -2, "reject_by_type");
  lua_newtable(L);
  for (int64_t i = 0; i < n_types; i++) { lua_pushinteger(L, (lua_Integer) mis_ty[i]); lua_rawseti(L, -2, (int) i); }
  lua_setfield(L, -2, "mistype_by_type");
  lua_newtable(L);
  for (int64_t i = 0; i < n_types * n_types; i++) { lua_pushinteger(L, (lua_Integer) conf[i]); lua_rawseti(L, -2, (int) i); }
  lua_setfield(L, -2, "confusion");
  free(corr_ty); free(rej_ty); free(mis_ty); free(conf);
  return 1;
}

static int tm_csr_merge (lua_State *L)
{
  tk_ivec_t *off1 = tk_ivec_peek(L, 1, "off1");
  uint64_t nbr1_n;
  const int32_t *nbr1_a = tk_peek_tokens(L, 2, &nbr1_n);
  if (!nbr1_a) return luaL_error(L, "nbr1: expected svec or ivec");
  tk_fvec_t *val1_f = tk_fvec_peekopt(L, 3);
  tk_dvec_t *val1_d = val1_f ? NULL : tk_dvec_peekopt(L, 3);
  tk_ivec_t *off2 = tk_ivec_peek(L, 4, "off2");
  uint64_t nbr2_n;
  const int32_t *nbr2_a = tk_peek_tokens(L, 5, &nbr2_n);
  if (!nbr2_a) return luaL_error(L, "nbr2: expected svec or ivec");
  tk_fvec_t *val2_f = tk_fvec_peekopt(L, 6);
  tk_dvec_t *val2_d = val2_f ? NULL : tk_dvec_peekopt(L, 6);
  int64_t shift = (int64_t)tk_lua_checkunsigned(L, 7, "token_shift");
  uint64_t n = off1->n - 1;
  uint64_t total = nbr1_n + nbr2_n;
  tk_ivec_t *out_off = tk_ivec_create(L, n + 1);
  tk_svec_t *out_nbr = tk_svec_create(L, total);
  out_nbr->n = total;
  tk_fvec_t *out_val = tk_fvec_create(L, total);
  out_val->n = total;
  out_off->a[0] = 0;
  uint64_t pos = 0;
  for (uint64_t i = 0; i < n; i++) {
    int64_t s1 = off1->a[i], e1 = off1->a[i + 1];
    for (int64_t j = s1; j < e1; j++) {
      out_nbr->a[pos] = nbr1_a[j];
      out_val->a[pos] = val1_f ? val1_f->a[j] : val1_d ? (float)val1_d->a[j] : 1.0f;
      pos++;
    }
    int64_t s2 = off2->a[i], e2 = off2->a[i + 1];
    for (int64_t j = s2; j < e2; j++) {
      out_nbr->a[pos] = (int32_t)((int64_t)nbr2_a[j] + shift);
      out_val->a[pos] = val2_f ? val2_f->a[j] : val2_d ? (float)val2_d->a[j] : 1.0f;
      pos++;
    }
    out_off->a[i + 1] = (int64_t)pos;
  }
  return 3;
}

static int tm_csr_standardize (lua_State *L)
{
  tk_svec_t *tokens = tk_svec_peek(L, 2, "tokens");
  tk_fvec_t *values = tk_fvec_peek(L, 3, "values");
  tk_fvec_t *scores = tk_fvec_peekopt(L, 4);
  if (scores) {
    tm_csr_gather_mul(values, tokens, scores);
    lua_pushvalue(L, 4);
    return 1;
  }
  uint64_t n_tokens = tk_lua_checkunsigned(L, 5, "n_tokens");
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  uint64_t n_samples = offsets->n - 1;
  double *sum = (double *)calloc(n_tokens, sizeof(double));
  double *sum_sq = (double *)calloc(n_tokens, sizeof(double));
  if (!sum || !sum_sq) {
    free(sum); free(sum_sq);
    return luaL_error(L, "standardize: alloc failed");
  }
  for (uint64_t j = 0; j < tokens->n; j++) {
    double v = (double)values->a[j];
    sum[tokens->a[j]] += v;
    sum_sq[tokens->a[j]] += v * v;
  }
  scores = tk_fvec_create(L, n_tokens);
  scores->n = n_tokens;
  double n = (double)n_samples;
  for (uint64_t t = 0; t < n_tokens; t++) {
    double mean = sum[t] / n;
    double var = sum_sq[t] / n - mean * mean;
    double std = sqrt(var);
    scores->a[t] = std > 1e-10 ? (float)(1.0 / std) : 0.0f;
  }
  free(sum); free(sum_sq);
  tm_csr_gather_mul(values, tokens, scores);
  return 1;
}

// Per-row L2 normalize. normalize(offsets, [values]) -> values.
// With a values fvec: scale each row's values by 1/sqrt(sum v^2) in place. Without values (binary
// CSR, implicit 1.0): create and return an fvec[offsets[n]] holding 1/sqrt(nnz_row) per nonzero.
// Empty / zero-norm rows are left at 0. After this, the per-row L2 norm is 1, so a cosine-family
// kernel's similarity is just the sparse dot (no denom needed).
static int tm_csr_normalize (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  uint64_t n = offsets->n - 1;
  tk_fvec_t *values = tk_fvec_peekopt(L, 2);
  if (values) {
    for (uint64_t i = 0; i < n; i++) {
      int64_t lo = offsets->a[i], hi = offsets->a[i + 1];
      double ss = 0.0;
      for (int64_t j = lo; j < hi; j++) { double v = (double) values->a[j]; ss += v * v; }
      if (ss > 1e-20) {
        float inv = (float) (1.0 / sqrt(ss));
        for (int64_t j = lo; j < hi; j++) values->a[j] *= inv;
      }
    }
    lua_pushvalue(L, 2);
    return 1;
  }
  uint64_t total = (uint64_t) offsets->a[n];
  values = tk_fvec_create(L, total);
  values->n = total;
  for (uint64_t i = 0; i < n; i++) {
    int64_t lo = offsets->a[i], hi = offsets->a[i + 1];
    uint64_t cnt = (uint64_t) (hi - lo);
    float v = cnt > 0 ? (float) (1.0 / sqrt((double) cnt)) : 0.0f;
    for (int64_t j = lo; j < hi; j++) values->a[j] = v;
  }
  return 1;
}

// Sum of squared values per token-block. bounds is a table of ascending cut points
// {b0, b1, ..., bn}; returns dvec[n] with out[k] = sum of values[j]^2 for tokens in [bk, bk+1).
static int tm_csr_block_sumsq (lua_State *L)
{
  uint64_t nnz;
  const int32_t *toks = tk_peek_tokens(L, 1, &nnz);
  if (!toks)
    return luaL_error(L, "block_sumsq: tokens expected svec or ivec");
  tk_fvec_t *values_f = tk_fvec_peekopt(L, 2);
  tk_dvec_t *values_d = values_f ? NULL : tk_dvec_peekopt(L, 2);
  if (!values_f && !values_d)
    return luaL_error(L, "block_sumsq: values expected fvec or dvec");
  luaL_checktype(L, 3, LUA_TTABLE);
  int nb = (int) lua_objlen(L, 3) - 1;
  if (nb < 1)
    return luaL_error(L, "block_sumsq: bounds needs >= 2 entries");
  int64_t *bounds = (int64_t *) malloc((size_t) (nb + 1) * sizeof(int64_t));
  for (int b = 0; b <= nb; b++) {
    lua_rawgeti(L, 3, b + 1);
    bounds[b] = (int64_t) lua_tointeger(L, -1);
    lua_pop(L, 1);
  }
  tk_dvec_t *out = tk_dvec_create(L, (uint64_t) nb);
  out->n = (uint64_t) nb;
  memset(out->a, 0, (size_t) nb * sizeof(double));
  for (uint64_t j = 0; j < nnz; j++) {
    int64_t tok = (int64_t) toks[j];
    double v = values_f ? (double) values_f->a[j] : values_d->a[j];
    for (int b = 0; b < nb; b++)
      if (tok >= bounds[b] && tok < bounds[b + 1]) { out->a[b] += v * v; break; }
  }
  free(bounds);
  return 1;
}

typedef enum {
  TK_SEG_LEAD = 0, TK_SEG_INNER_FWD = 1, TK_SEG_INNER_BWD = 2, TK_SEG_TRAIL = 3
} tk_seg_t;

// Gather one focus's windowed segment into acc/touched (acc indexed by gram id, reset to 0 by caller
// via the touched list). Returns Σweight; sets *nt_out = #distinct ids touched.
static inline double tm_seg_gather_sparse (
  tk_seg_t seg, int64_t a, int64_t b, int64_t W, const float *alpha_pow,
  const int64_t *boff, const int32_t *bucket_id, int64_t maxpos,
  float *acc, int32_t *touched, int64_t *nt_out)
{
  int64_t lo, hi;
  if (seg == TK_SEG_LEAD) { lo = a - W; hi = a; }
  else if (seg == TK_SEG_TRAIL) { lo = b; hi = b + W; }
  else { lo = a; hi = b; }                 // inner_fwd / inner_bwd
  if (lo < 0) lo = 0;
  if (hi > maxpos + 1) hi = maxpos + 1;
  double z = 0.0;
  int64_t nt = 0;
  for (int64_t p = lo; p < hi; p++) {
    int64_t dist;
    if (seg == TK_SEG_LEAD) dist = (a - 1) - p;
    else if (seg == TK_SEG_TRAIL) dist = p - b;
    else if (seg == TK_SEG_INNER_FWD) dist = (b - 1) - p;
    else dist = p - a;                     // inner_bwd
    if (dist < 0 || dist >= W) continue;
    float w = alpha_pow[dist];
    int64_t s0 = boff[p], s1 = boff[p + 1];
    for (int64_t slot = s0; slot < s1; slot++) {
      int32_t id = bucket_id[slot];
      if (acc[id] == 0.0f) touched[nt++] = id;
      acc[id] += w;
    }
    z += w * (double)(s1 - s0);
  }
  *nt_out = nt;
  return z;
}

// csr.segment_pool: PCNN-segment EMA pool of one segment around each focus span.
// Sparse (stream_*): windowed decayed gram histogram. Dense (features): exact prefix/suffix EMA.
static int tm_csr_segment_pool (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "segment");
  const char *seg_str = luaL_checkstring(L, -1);
  tk_seg_t seg;
  if (!strcmp(seg_str, "lead")) seg = TK_SEG_LEAD;
  else if (!strcmp(seg_str, "inner_fwd")) seg = TK_SEG_INNER_FWD;
  else if (!strcmp(seg_str, "inner_bwd")) seg = TK_SEG_INNER_BWD;
  else if (!strcmp(seg_str, "trail")) seg = TK_SEG_TRAIL;
  else return luaL_error(L, "segment_pool: bad segment '%s'", seg_str);
  lua_pop(L, 1);
  lua_getfield(L, 1, "alpha");
  double alpha = luaL_checknumber(L, -1);
  lua_pop(L, 1);
  if (!(alpha > 0.0 && alpha < 1.0))
    return luaL_error(L, "segment_pool: alpha must be in (0,1)");
  lua_getfield(L, 1, "doc_span_offsets");
  tk_ivec_t *doc_off = tk_ivec_peek(L, -1, "doc_span_offsets"); lua_pop(L, 1);
  lua_getfield(L, 1, "span_starts");
  tk_ivec_t *span_s = tk_ivec_peek(L, -1, "span_starts"); lua_pop(L, 1);
  lua_getfield(L, 1, "span_ends");
  tk_ivec_t *span_e = tk_ivec_peek(L, -1, "span_ends"); lua_pop(L, 1);
  int64_t n_docs = (int64_t)doc_off->n - 1;
  int64_t n_focus = doc_off->a[n_docs];

  lua_getfield(L, 1, "features");
  int dense = !lua_isnil(L, -1);
  tk_fvec_t *feat_f = NULL; tk_dvec_t *feat_d = NULL;
  int64_t dcols = 0;
  tk_ivec_t *tok_off = NULL, *tok_s = NULL, *tok_e = NULL;
  int64_t W = 0; float *alpha_pow = NULL;
  tk_ivec_t *stream_off = NULL, *stream_pos = NULL; const int32_t *stream_tok = NULL;
  if (dense) {
    feat_f = tk_fvec_peekopt(L, -1);
    feat_d = feat_f ? NULL : tk_dvec_peekopt(L, -1);
    if (!feat_f && !feat_d) return luaL_error(L, "segment_pool: features expected fvec or dvec");
    lua_pop(L, 1);
    lua_getfield(L, 1, "d"); dcols = (int64_t)tk_lua_checkunsigned(L, -1, "d"); lua_pop(L, 1);
    if (dcols > 64) return luaL_error(L, "segment_pool: dense d must be <= 64");
    lua_getfield(L, 1, "token_offsets"); tok_off = tk_ivec_peek(L, -1, "token_offsets"); lua_pop(L, 1);
    lua_getfield(L, 1, "token_starts"); tok_s = tk_ivec_peek(L, -1, "token_starts"); lua_pop(L, 1);
    lua_getfield(L, 1, "token_ends"); tok_e = tk_ivec_peek(L, -1, "token_ends"); lua_pop(L, 1);
  } else {
    lua_pop(L, 1);
    lua_getfield(L, 1, "stream_offsets"); stream_off = tk_ivec_peek(L, -1, "stream_offsets"); lua_pop(L, 1);
    lua_getfield(L, 1, "stream_tokens");
    uint64_t stok_n; stream_tok = tk_peek_tokens(L, -1, &stok_n);
    if (!stream_tok) return luaL_error(L, "segment_pool: stream_tokens expected svec or ivec");
    lua_setfield(L, 1, "stream_tokens");
    lua_getfield(L, 1, "stream_positions"); stream_pos = tk_ivec_peek(L, -1, "stream_positions"); lua_pop(L, 1);
    lua_getfield(L, 1, "n_stream"); int64_t n_stream = (int64_t)tk_lua_checkunsigned(L, -1, "n_stream"); lua_pop(L, 1);
    dcols = n_stream; // not used for emission, kept for clarity
    W = (int64_t)ceil(log(1e-6) / log(alpha));
    if (W < 1) W = 1;
    if (W > 4096) W = 4096;
    alpha_pow = (float *)malloc((size_t)(W + 1) * sizeof(float));
    double ap = 1.0;
    for (int64_t k = 0; k <= W; k++) { alpha_pow[k] = (float)ap; ap *= alpha; }
  }
  int64_t U = dense ? dcols : 0; // sparse acc sized to vocab
  if (!dense) { lua_getfield(L, 1, "n_stream"); U = (int64_t)tk_lua_checkunsigned(L, -1, "n_stream"); lua_pop(L, 1); }

  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n_focus + 1));
  offsets->n = (uint64_t)(n_focus + 1);
  int off_idx = lua_gettop(L);
  int64_t *row_nnz = (int64_t *)calloc((size_t)(n_focus > 0 ? n_focus : 1), sizeof(int64_t));

  // PASS 1: row_nnz
  #pragma omp parallel
  {
    float *acc = dense ? NULL : (float *)calloc((size_t)U, sizeof(float));
    int32_t *touched = dense ? NULL : (int32_t *)malloc((size_t)U * sizeof(int32_t));
    #pragma omp for schedule(dynamic)
    for (int64_t d = 0; d < n_docs; d++) {
      int64_t f0 = doc_off->a[d], f1 = doc_off->a[d + 1];
      if (dense) {
        int64_t t0 = tok_off->a[d], t1 = tok_off->a[d + 1], nt = t1 - t0;
        for (int64_t f = f0; f < f1; f++) {
          int64_t a = span_s->a[f], b = span_e->a[f];
          int64_t ia = 0; while (ia < nt && tok_e->a[t0 + ia] <= a) ia++;
          int64_t ib = 0; while (ib < nt && tok_s->a[t0 + ib] < b) ib++;
          int ne = (seg == TK_SEG_LEAD) ? (ia > 0)
                 : (seg == TK_SEG_TRAIL) ? (ib < nt) : (ia < ib);
          row_nnz[f] = ne ? dcols : 0;
        }
      } else {
        int64_t e0 = stream_off->a[d], e1 = stream_off->a[d + 1];
        int64_t maxpos = -1;
        for (int64_t j = e0; j < e1; j++) if (stream_pos->a[j] > maxpos) maxpos = stream_pos->a[j];
        if (maxpos < 0) { for (int64_t f = f0; f < f1; f++) row_nnz[f] = 0; continue; }
        int64_t *boff = (int64_t *)calloc((size_t)(maxpos + 2), sizeof(int64_t));
        int32_t *bucket_id = (int32_t *)malloc((size_t)(e1 - e0) * sizeof(int32_t));
        for (int64_t j = e0; j < e1; j++) boff[stream_pos->a[j] + 1]++;
        for (int64_t p = 1; p <= maxpos + 1; p++) boff[p] += boff[p - 1];
        int64_t *bpos = (int64_t *)malloc((size_t)(maxpos + 2) * sizeof(int64_t));
        memcpy(bpos, boff, (size_t)(maxpos + 2) * sizeof(int64_t));
        for (int64_t j = e0; j < e1; j++) { int64_t p = stream_pos->a[j]; bucket_id[bpos[p]++] = stream_tok[j]; }
        free(bpos);
        for (int64_t f = f0; f < f1; f++) {
          int64_t nt = 0;
          tm_seg_gather_sparse(seg, span_s->a[f], span_e->a[f], W, alpha_pow,
            boff, bucket_id, maxpos, acc, touched, &nt);
          for (int64_t i = 0; i < nt; i++) acc[touched[i]] = 0.0f;
          row_nnz[f] = nt;
        }
        free(boff); free(bucket_id);
      }
    }
    free(acc); free(touched);
  }

  offsets->a[0] = 0;
  for (int64_t f = 0; f < n_focus; f++) offsets->a[f + 1] = offsets->a[f] + row_nnz[f];
  int64_t total = offsets->a[n_focus];

  tk_svec_t *tok_out = tk_svec_create(L, (uint64_t)total);
  tok_out->n = (uint64_t)total; int tok_idx = lua_gettop(L);
  tk_fvec_t *val_out = tk_fvec_create(L, (uint64_t)total);
  val_out->n = (uint64_t)total; int val_idx = lua_gettop(L);

  // PASS 2: fill
  #pragma omp parallel
  {
    float *acc = dense ? NULL : (float *)calloc((size_t)U, sizeof(float));
    int32_t *touched = dense ? NULL : (int32_t *)malloc((size_t)U * sizeof(int32_t));
    double *F = NULL, *B = NULL, *Zf = NULL, *Zb = NULL; int64_t cap_nt = 0;
    #pragma omp for schedule(dynamic)
    for (int64_t d = 0; d < n_docs; d++) {
      int64_t f0 = doc_off->a[d], f1 = doc_off->a[d + 1];
      if (dense) {
        int64_t t0 = tok_off->a[d], t1 = tok_off->a[d + 1], nt = t1 - t0;
        if (nt > cap_nt) {
          cap_nt = nt;
          F = (double *)realloc(F, (size_t)nt * (size_t)dcols * sizeof(double));
          B = (double *)realloc(B, (size_t)nt * (size_t)dcols * sizeof(double));
          Zf = (double *)realloc(Zf, (size_t)nt * sizeof(double));
          Zb = (double *)realloc(Zb, (size_t)nt * sizeof(double));
        }
        for (int64_t i = 0; i < nt; i++) {
          int64_t gi = t0 + i;
          for (int64_t k = 0; k < dcols; k++) {
            double v = feat_f ? (double)feat_f->a[gi * dcols + k] : feat_d->a[gi * dcols + k];
            F[i * dcols + k] = (i > 0 ? alpha * F[(i - 1) * dcols + k] : 0.0) + v;
          }
          Zf[i] = (i > 0 ? alpha * Zf[i - 1] : 0.0) + 1.0;
        }
        for (int64_t i = nt - 1; i >= 0; i--) {
          int64_t gi = t0 + i;
          for (int64_t k = 0; k < dcols; k++) {
            double v = feat_f ? (double)feat_f->a[gi * dcols + k] : feat_d->a[gi * dcols + k];
            B[i * dcols + k] = (i < nt - 1 ? alpha * B[(i + 1) * dcols + k] : 0.0) + v;
          }
          Zb[i] = (i < nt - 1 ? alpha * Zb[i + 1] : 0.0) + 1.0;
        }
        for (int64_t f = f0; f < f1; f++) {
          int64_t a = span_s->a[f], b = span_e->a[f];
          int64_t ia = 0; while (ia < nt && tok_e->a[t0 + ia] <= a) ia++;
          int64_t ib = 0; while (ib < nt && tok_s->a[t0 + ib] < b) ib++;
          int64_t w = offsets->a[f];
          if (seg == TK_SEG_LEAD) {
            if (ia <= 0) continue;
            double zz = Zf[ia - 1];
            for (int64_t k = 0; k < dcols; k++) { tok_out->a[w] = (int32_t)k; val_out->a[w] = (float)(F[(ia - 1) * dcols + k] / zz); w++; }
          } else if (seg == TK_SEG_TRAIL) {
            if (ib >= nt) continue;
            double zz = Zb[ib];
            for (int64_t k = 0; k < dcols; k++) { tok_out->a[w] = (int32_t)k; val_out->a[w] = (float)(B[ib * dcols + k] / zz); w++; }
          } else {
            if (ia >= ib) continue;
            // inner: running weight; fwd dominates tail (start at ib-1), bwd dominates head (start at ia)
            double zz = 0.0;
            double tmp[64];
            for (int64_t k = 0; k < dcols; k++) tmp[k] = 0.0;
            if (seg == TK_SEG_INNER_FWD) {
              double wt = 1.0;
              for (int64_t i = ib - 1; i >= ia; i--) {
                int64_t gi = t0 + i;
                for (int64_t k = 0; k < dcols; k++) {
                  double v = feat_f ? (double)feat_f->a[gi * dcols + k] : feat_d->a[gi * dcols + k];
                  tmp[k] += wt * v;
                }
                zz += wt; wt *= alpha;
              }
            } else {
              double wt = 1.0;
              for (int64_t i = ia; i < ib; i++) {
                int64_t gi = t0 + i;
                for (int64_t k = 0; k < dcols; k++) {
                  double v = feat_f ? (double)feat_f->a[gi * dcols + k] : feat_d->a[gi * dcols + k];
                  tmp[k] += wt * v;
                }
                zz += wt; wt *= alpha;
              }
            }
            for (int64_t k = 0; k < dcols; k++) { tok_out->a[w] = (int32_t)k; val_out->a[w] = (float)(tmp[k] / zz); w++; }
          }
        }
      } else {
        int64_t e0 = stream_off->a[d], e1 = stream_off->a[d + 1];
        int64_t maxpos = -1;
        for (int64_t j = e0; j < e1; j++) if (stream_pos->a[j] > maxpos) maxpos = stream_pos->a[j];
        if (maxpos < 0) continue;
        int64_t *boff = (int64_t *)calloc((size_t)(maxpos + 2), sizeof(int64_t));
        int32_t *bucket_id = (int32_t *)malloc((size_t)(e1 - e0) * sizeof(int32_t));
        for (int64_t j = e0; j < e1; j++) boff[stream_pos->a[j] + 1]++;
        for (int64_t p = 1; p <= maxpos + 1; p++) boff[p] += boff[p - 1];
        int64_t *bpos = (int64_t *)malloc((size_t)(maxpos + 2) * sizeof(int64_t));
        memcpy(bpos, boff, (size_t)(maxpos + 2) * sizeof(int64_t));
        for (int64_t j = e0; j < e1; j++) { int64_t p = stream_pos->a[j]; bucket_id[bpos[p]++] = stream_tok[j]; }
        free(bpos);
        for (int64_t f = f0; f < f1; f++) {
          int64_t nt = 0;
          double z = tm_seg_gather_sparse(seg, span_s->a[f], span_e->a[f], W, alpha_pow,
            boff, bucket_id, maxpos, acc, touched, &nt);
          int64_t w = offsets->a[f];
          if (z > 0.0) {
            for (int64_t i = 0; i < nt; i++) {
              int32_t id = touched[i];
              tok_out->a[w] = id; val_out->a[w] = (float)(acc[id] / z); w++;
            }
          }
          for (int64_t i = 0; i < nt; i++) acc[touched[i]] = 0.0f;
        }
        free(boff); free(bucket_id);
      }
    }
    free(acc); free(touched);
    free(F); free(B); free(Zf); free(Zb);
  }

  free(row_nnz);
  free(alpha_pow);
  lua_pushvalue(L, off_idx);
  lua_pushvalue(L, tok_idx);
  lua_pushvalue(L, val_idx);
  return 3;
}

static luaL_Reg tm_csr_fns[] = {
  { "segment_pool", tm_csr_segment_pool },
  { "seq_select", tm_csr_seq_select },
  { "label_union", tm_csr_label_union },
  { "sort_csr_desc", tm_csr_sort_csr_desc },
  { "truncate", tm_csr_truncate },
  { "transpose", tm_csr_transpose },
  { "tokenize", tm_csr_tokenize },
  { "tokenize_annotated", tm_csr_tokenize_annotated },
  { "sqrt", tm_csr_sqrt },
  { "apply_bns", tm_csr_apply_bns },
  { "apply_auc", tm_csr_apply_auc },
  { "apply_idf", tm_csr_apply_idf },
  { "gather_rows", tm_csr_gather_rows },
  { "binary_label_csr", tm_csr_binary_label_csr },
  { "filter_spans", tm_csr_filter_spans },
  { "bio_encode", tm_csr_bio_encode },
  { "bio_decode", tm_csr_bio_decode },
  { "bio_viterbi", tm_csr_bio_viterbi },
  { "bio_band_mask", tm_csr_bio_band_mask },
  { "bio_token_type", tm_csr_bio_token_type },
  { "enumerate_subspans", tm_csr_enumerate_subspans },
  { "type_labels", tm_csr_type_labels },
  { "nms", tm_csr_nms },
  { "nms_dp", tm_csr_nms_dp },
  { "union_spans", tm_csr_union_spans },
  { "span_miss_report", tm_csr_span_miss_report },
  { "type_decode_report", tm_csr_type_decode_report },
  { "merge", tm_csr_merge },
  { "standardize", tm_csr_standardize },
  { "normalize", tm_csr_normalize },
  { "block_sumsq", tm_csr_block_sumsq },
  { NULL, NULL }
};

int luaopen_santoku_learn_csr (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_csr_fns, 0);
  return 1;
}
