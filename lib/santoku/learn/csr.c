#include <santoku/learn/csr.h>
#include <santoku/fvec.h>
#include <santoku/ivec/ext.h>
#include <santoku/iumap/ext.h>
#include <omp.h>
#include <assert.h>
#include <math.h>

static int tm_csr_seq_select (lua_State *L)
{
  tk_ivec_t *tokens = tk_ivec_peek(L, 1, "tokens");
  tk_ivec_t *offsets = tk_ivec_peek(L, 2, "offsets");
  tk_ivec_t *keep_ids = tk_ivec_peek(L, 3, "keep_ids");
  tk_fvec_t *values_f = tk_fvec_peekopt(L, 4);
  tk_dvec_t *values_d = values_f ? NULL : tk_dvec_peekopt(L, 4);
  int has_values = values_f || values_d;
  tk_iumap_t *inverse = tk_iumap_from_ivec(L, keep_ids);
  if (!inverse) return luaL_error(L, "seq_select: allocation failed");
  uint64_t n_docs = offsets->n - 1;
  tk_ivec_t *new_tok = tk_ivec_create(L, tokens->n, 0, 0);
  tk_ivec_t *new_off = tk_ivec_create(L, n_docs + 1, 0, 0);
  new_off->n = n_docs + 1;
  tk_fvec_t *new_val_f = (has_values && values_f) ? tk_fvec_create(L, tokens->n, 0, 0) : NULL;
  tk_dvec_t *new_val_d = (has_values && values_d) ? tk_dvec_create(L, tokens->n, 0, 0) : NULL;
  new_off->a[0] = 0;
  uint64_t pos = 0;
  for (uint64_t d = 0; d < n_docs; d++) {
    int64_t start = offsets->a[d];
    int64_t end = offsets->a[d + 1];
    for (int64_t j = start; j < end; j++) {
      int64_t new_id = tk_iumap_get_or(inverse, tokens->a[j], -1);
      if (new_id < 0) continue;
      new_tok->a[pos] = new_id;
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

  tk_ivec_t *out_off = tk_ivec_create(L, n_queries + 1, 0, 0);
  out_off->n = n_queries + 1;
  tk_ivec_t *out_nbr = tk_ivec_create(L, total, 0, 0);
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
  tk_ivec_t *tokens = tk_ivec_peek(L, 2, "tokens");
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
  tk_ivec_t *csc_off = tk_ivec_create(L, n_tokens + 1, 0, 0);
  csc_off->n = n_tokens + 1;
  memcpy(csc_off->a, counts, (n_tokens + 1) * sizeof(int64_t));
  tk_ivec_t *csc_rows = tk_ivec_create(L, nnz, 0, 0);
  csc_rows->n = nnz;
  tk_fvec_t *csc_vals_f = (has_values && values_f) ? tk_fvec_create(L, nnz, 0, 0) : NULL;
  tk_dvec_t *csc_vals_d = (has_values && values_d) ? tk_dvec_create(L, nnz, 0, 0) : NULL;
  if (csc_vals_f) csc_vals_f->n = nnz;
  if (csc_vals_d) csc_vals_d->n = nnz;
  for (uint64_t s = 0; s < n_samples; s++) {
    for (int64_t j = offsets->a[s]; j < offsets->a[s + 1]; j++) {
      int64_t tok = tokens->a[j];
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
  tk_ivec_t *out_n = tk_ivec_create(L, nbr->n, NULL, NULL);
  memcpy(out_n->a, nbr->a, nbr->n * sizeof(int64_t));
  if (scores_f) {
    tk_fvec_t *out_s = tk_fvec_create(L, scores_f->n, NULL, NULL);
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
    tk_dvec_t *out_s = tk_dvec_create(L, scores_d->n, NULL, NULL);
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

static inline size_t tm_csr_pack_ngrams (
  const char *text, size_t len, int n, int64_t *out)
{
  assert(n >= 1 && n <= 8);
  if (len < (size_t)n) return 0;
  uint64_t mask = (n < 8) ? ((1ULL << (n * 8)) - 1) : ~0ULL;
  uint64_t id = 0;
  for (int i = 0; i < n - 1; i++)
    id = (id << 8) | (uint8_t)text[i];
  size_t count = len - (size_t)n + 1;
  for (size_t i = 0; i < count; i++) {
    id = ((id << 8) | (uint8_t)text[(size_t)(n - 1) + i]) & mask;
    out[i] = (int64_t)id;
  }
  return count;
}

static int tm_csr_tokenize_fn (lua_State *L, int fn_idx, int64_t ngram, int64_t n_samples)
{
  lua_getfield(L, 1, "ngram_map");
  bool have_map = !lua_isnil(L, -1);
  tk_iumap_t *ngram_map;
  int64_t next_id;
  if (have_map) {
    ngram_map = tk_iumap_peek(L, -1, "ngram_map");
    next_id = (int64_t)tk_iumap_size(ngram_map);
  } else {
    lua_pop(L, 1);
    ngram_map = tk_iumap_create(L, 0);
    next_id = 0;
  }
  int map_idx = lua_gettop(L);
  uint32_t map_end = tk_iumap_end(ngram_map);
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n_samples + 1), 0, 0);
  offsets->n = (uint64_t)(n_samples + 1);
  offsets->a[0] = 0;
  size_t tok_cap = 1024 * 1024;
  int64_t *tok_buf = (int64_t *)malloc(tok_cap * sizeof(int64_t));
  float *vbuf = (float *)malloc(tok_cap * sizeof(float));
  int64_t tok_n = 0;
  size_t pack_cap = 4096;
  int64_t *packed = (int64_t *)malloc(pack_cap * sizeof(int64_t));
  for (int64_t s = 0; s < n_samples; s++) {
    lua_pushvalue(L, fn_idx);
    lua_call(L, 0, 3);
    if (lua_isnil(L, -3)) {
      lua_pop(L, 3);
      for (int64_t r = s; r < n_samples; r++)
        offsets->a[r + 1] = tok_n;
      break;
    }
    const char *str;
    size_t len;
    if (!lua_isnil(L, -2) && !lua_isnil(L, -1)) {
      size_t full_len;
      const char *full_str = lua_tolstring(L, -3, &full_len);
      int64_t sub_s = lua_tointeger(L, -2);
      int64_t sub_e = lua_tointeger(L, -1);
      if (sub_s < 1) sub_s = 1;
      if ((uint64_t)sub_e > full_len) sub_e = (int64_t)full_len;
      str = full_str + (sub_s - 1);
      len = (sub_e >= sub_s) ? (size_t)(sub_e - sub_s + 1) : 0;
    } else {
      str = lua_tolstring(L, -3, &len);
    }
    lua_pop(L, 3);
    if (!str || len == 0) {
      offsets->a[s + 1] = tok_n;
      continue;
    }
    if (len > pack_cap) {
      pack_cap = len;
      packed = (int64_t *)realloc(packed, pack_cap * sizeof(int64_t));
    }
    size_t count = tm_csr_pack_ngrams(str, len, (int)ngram, packed);
    int64_t row_n = 0;
    for (size_t i = 0; i < count; i++) {
      if (!have_map) {
        int absent;
        uint32_t it = tk_iumap_put(ngram_map, packed[i], &absent);
        if (absent) {
          tk_iumap_setval(ngram_map, it, next_id);
          packed[row_n++] = next_id++;
        } else {
          packed[row_n++] = tk_iumap_val(ngram_map, it);
        }
      } else {
        uint32_t it = tk_iumap_get(ngram_map, packed[i]);
        if (it == map_end) continue;
        packed[row_n++] = tk_iumap_val(ngram_map, it);
      }
    }
    ks_introsort(tk_ivec_asc, (size_t)row_n, packed);
    for (int64_t i = 0; i < row_n; ) {
      int64_t tok = packed[i];
      float cnt = 0.0f;
      while (i < row_n && packed[i] == tok) { cnt += 1.0f; i++; }
      if (tok_n >= (int64_t)tok_cap) {
        tok_cap *= 2;
        tok_buf = (int64_t *)realloc(tok_buf, tok_cap * sizeof(int64_t));
        vbuf = (float *)realloc(vbuf, tok_cap * sizeof(float));
      }
      tok_buf[tok_n] = tok;
      vbuf[tok_n] = cnt;
      tok_n++;
    }
    offsets->a[s + 1] = tok_n;
  }
  free(packed);
  tk_ivec_t *tok_out = tk_ivec_create(L, (uint64_t)tok_n, 0, 0);
  tok_out->n = (uint64_t)tok_n;
  if (tok_n > 0)
    memcpy(tok_out->a, tok_buf, (size_t)tok_n * sizeof(int64_t));
  free(tok_buf);
  tk_fvec_t *val_out = tk_fvec_create(L, (uint64_t)tok_n, 0, 0);
  val_out->n = (uint64_t)tok_n;
  if (tok_n > 0)
    memcpy(val_out->a, vbuf, (size_t)tok_n * sizeof(float));
  free(vbuf);
  uint64_t n_tokens = have_map ? (uint64_t)tk_iumap_size(ngram_map) : (uint64_t)next_id;
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
  lua_getfield(L, 1, "texts");
  if (lua_isfunction(L, -1))
    return tm_csr_tokenize_fn(L, lua_gettop(L), (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize", "hdc_ngram"),
      (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize", "n_samples"));
  if (!lua_istable(L, -1))
    return luaL_error(L, "tokenize: texts must be table or function");
  int texts_idx = lua_gettop(L);
  int64_t ngram = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize", "hdc_ngram");
  if (ngram < 1 || ngram > 8)
    return luaL_error(L, "tokenize: hdc_ngram must be 1-8");
  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize", "n_samples");
  const char **strs = (const char **)malloc((uint64_t)n_samples * sizeof(const char *));
  size_t *lens = (size_t *)malloc((uint64_t)n_samples * sizeof(size_t));
  size_t max_len = 0;
  for (int64_t s = 0; s < n_samples; s++) {
    lua_rawgeti(L, texts_idx, (int)(s + 1));
    strs[s] = lua_tolstring(L, -1, &lens[s]);
    lua_pop(L, 1);
    if (lens[s] > max_len) max_len = lens[s];
  }
  lua_getfield(L, 1, "ngram_map");
  bool have_map = !lua_isnil(L, -1);
  tk_iumap_t *ngram_map;
  int map_idx;
  int64_t next_id;
  int64_t *sample_counts = (int64_t *)calloc((size_t)n_samples, sizeof(int64_t));
  if (have_map) {
    ngram_map = tk_iumap_peek(L, -1, "ngram_map");
    map_idx = lua_gettop(L);
    next_id = (int64_t)tk_iumap_size(ngram_map);
    uint32_t me = tk_iumap_end(ngram_map);
    #pragma omp parallel
    {
      int64_t *packed_buf = (int64_t *)malloc(max_len * sizeof(int64_t));
      #pragma omp for schedule(dynamic)
      for (int64_t s = 0; s < n_samples; s++) {
        size_t count = tm_csr_pack_ngrams(strs[s], lens[s], (int)ngram, packed_buf);
        int64_t nv = 0;
        for (size_t i = 0; i < count; i++) {
          uint32_t iter = tk_iumap_get(ngram_map, packed_buf[i]);
          if (iter != me) packed_buf[nv++] = tk_iumap_val(ngram_map, iter);
        }
        ks_introsort(tk_ivec_asc, (size_t)nv, packed_buf);
        int64_t unique = 0;
        for (int64_t i = 0; i < nv; i++)
          if (i == 0 || packed_buf[i] != packed_buf[i - 1]) unique++;
        sample_counts[s] = unique;
      }
      free(packed_buf);
    }
  } else {
    lua_pop(L, 1);
    int max_threads = omp_get_max_threads();
    tk_iumap_t **local_maps = (tk_iumap_t **)calloc((size_t)max_threads, sizeof(tk_iumap_t *));
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      tk_iumap_t *lm = tk_iumap_create(NULL, 0);
      local_maps[tid] = lm;
      int64_t *packed_buf = (int64_t *)malloc(max_len * sizeof(int64_t));
      #pragma omp for schedule(dynamic)
      for (int64_t s = 0; s < n_samples; s++) {
        size_t count = tm_csr_pack_ngrams(strs[s], lens[s], (int)ngram, packed_buf);
        for (size_t i = 0; i < count; i++) {
          int absent;
          tk_iumap_put(lm, packed_buf[i], &absent);
        }
        ks_introsort(tk_ivec_asc, count, packed_buf);
        int64_t unique = 0;
        for (size_t i = 0; i < count; i++)
          if (i == 0 || packed_buf[i] != packed_buf[i - 1]) unique++;
        sample_counts[s] = unique;
      }
      free(packed_buf);
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
  uint64_t n_tokens = (uint64_t)next_id;
  uint32_t map_end = tk_iumap_end(ngram_map);
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n_samples + 1), 0, 0);
  offsets->n = (uint64_t)(n_samples + 1);
  offsets->a[0] = 0;
  int64_t total = 0;
  for (int64_t s = 0; s < n_samples; s++) {
    total += sample_counts[s];
    offsets->a[s + 1] = total;
  }
  free(sample_counts);
  tk_ivec_t *tok_out = tk_ivec_create(L, (uint64_t)total, 0, 0);
  tok_out->n = (uint64_t)total;
  tk_fvec_t *val_out = tk_fvec_create(L, (uint64_t)total, 0, 0);
  val_out->n = (uint64_t)total;
  #pragma omp parallel
  {
    int64_t *packed_buf = (int64_t *)malloc(max_len * sizeof(int64_t));
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      size_t count = tm_csr_pack_ngrams(strs[s], lens[s], (int)ngram, packed_buf);
      int64_t nv = 0;
      for (size_t i = 0; i < count; i++) {
        uint32_t iter = tk_iumap_get(ngram_map, packed_buf[i]);
        if (iter != map_end) packed_buf[nv++] = tk_iumap_val(ngram_map, iter);
      }
      ks_introsort(tk_ivec_asc, (size_t)nv, packed_buf);
      int64_t pos = offsets->a[s];
      for (int64_t i = 0; i < nv; ) {
        int64_t tok = packed_buf[i];
        float cnt = 0.0f;
        while (i < nv && packed_buf[i] == tok) { cnt += 1.0f; i++; }
        tok_out->a[pos] = tok;
        val_out->a[pos] = cnt;
        pos++;
      }
    }
    free(packed_buf);
  }
  free(strs);
  free(lens);
  lua_pushvalue(L, map_idx);
  lua_pushvalue(L, map_idx + 1);
  lua_pushvalue(L, map_idx + 2);
  lua_pushvalue(L, map_idx + 3);
  lua_pushinteger(L, (lua_Integer)n_tokens);
  return 5;
}

static int tm_csr_truncate (lua_State *L)
{
  tk_ivec_t *off = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *nbr = tk_ivec_peek(L, 2, "neighbors");
  tk_fvec_t *sco_f = tk_fvec_peekopt(L, 3);
  tk_dvec_t *sco_d = sco_f ? NULL : tk_dvec_peekopt(L, 3);
  int64_t k = (int64_t)luaL_checkinteger(L, 4);
  uint64_t ns = off->n - 1;
  tk_ivec_t *new_off = tk_ivec_create(L, ns + 1, 0, 0);
  new_off->n = ns + 1;
  int64_t total = 0;
  new_off->a[0] = 0;
  for (uint64_t i = 0; i < ns; i++) {
    int64_t rlen = off->a[i + 1] - off->a[i];
    if (rlen > k) rlen = k;
    total += rlen;
    new_off->a[i + 1] = total;
  }
  tk_ivec_t *new_nbr = tk_ivec_create(L, (uint64_t)total, 0, 0);
  new_nbr->n = (uint64_t)total;
  if (sco_f) {
    tk_fvec_t *new_sco = tk_fvec_create(L, (uint64_t)total, 0, 0);
    new_sco->n = (uint64_t)total;
    for (uint64_t i = 0; i < ns; i++) {
      int64_t src = off->a[i], dst = new_off->a[i];
      int64_t rlen = new_off->a[i + 1] - dst;
      memcpy(new_nbr->a + dst, nbr->a + src, (uint64_t)rlen * sizeof(int64_t));
      memcpy(new_sco->a + dst, sco_f->a + src, (uint64_t)rlen * sizeof(float));
    }
  } else if (sco_d) {
    tk_dvec_t *new_sco = tk_dvec_create(L, (uint64_t)total, 0, 0);
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
  return fabs(tm_probit(tpr) - tm_probit(fpr));
}

static inline void tm_csr_gather_mul (
  tk_fvec_t *values, tk_ivec_t *tokens, tk_fvec_t *scores)
{
  for (uint64_t j = 0; j < values->n; j++)
    values->a[j] *= scores->a[tokens->a[j]];
}

static int tm_csr_apply_bns (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *tokens = tk_ivec_peek(L, 2, "tokens");
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
    free(doc_freq); free(label_freq); free(lbl_off); free(lbl_docs); free(lbl_pos);
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
  scores = tk_fvec_create(L, n_tokens, NULL, NULL);
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
      float bns = (float)tm_bns(N, (double)doc_freq[f], P, (double)cooc[f]);
      if (bns > scores->a[f]) scores->a[f] = bns;
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
  tk_ivec_t *tokens = tk_ivec_peek(L, 2, "tokens");
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
  scores = tk_fvec_create(L, n_tokens, NULL, NULL);
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
      float score = (float)fabs(tm_probit(auc));
      if (score > scores->a[f]) scores->a[f] = score;
    }
  }
  free(pairs); free(ranks); free(rank_sums); free(doc_freq);
  tm_csr_gather_mul(values, tokens, scores);
  return 1;
}

static int tm_csr_merge (lua_State *L)
{
  tk_ivec_t *off1 = tk_ivec_peek(L, 1, "off1");
  tk_ivec_t *nbr1 = tk_ivec_peek(L, 2, "nbr1");
  tk_fvec_t *val1_f = tk_fvec_peekopt(L, 3);
  tk_dvec_t *val1_d = val1_f ? NULL : tk_dvec_peekopt(L, 3);
  tk_ivec_t *off2 = tk_ivec_peek(L, 4, "off2");
  tk_ivec_t *nbr2 = tk_ivec_peek(L, 5, "nbr2");
  tk_fvec_t *val2_f = tk_fvec_peekopt(L, 6);
  tk_dvec_t *val2_d = val2_f ? NULL : tk_dvec_peekopt(L, 6);
  int64_t shift = (int64_t)tk_lua_checkunsigned(L, 7, "token_shift");
  uint64_t n = off1->n - 1;
  uint64_t total = nbr1->n + nbr2->n;
  tk_ivec_t *out_off = tk_ivec_create(L, n + 1, NULL, NULL);
  tk_ivec_t *out_nbr = tk_ivec_create(L, total, NULL, NULL);
  out_nbr->n = total;
  tk_fvec_t *out_val = tk_fvec_create(L, total, NULL, NULL);
  out_val->n = total;
  out_off->a[0] = 0;
  uint64_t pos = 0;
  for (uint64_t i = 0; i < n; i++) {
    int64_t s1 = off1->a[i], e1 = off1->a[i + 1];
    for (int64_t j = s1; j < e1; j++) {
      out_nbr->a[pos] = nbr1->a[j];
      out_val->a[pos] = val1_f ? val1_f->a[j] : val1_d ? (float)val1_d->a[j] : 1.0f;
      pos++;
    }
    int64_t s2 = off2->a[i], e2 = off2->a[i + 1];
    for (int64_t j = s2; j < e2; j++) {
      out_nbr->a[pos] = nbr2->a[j] + shift;
      out_val->a[pos] = val2_f ? val2_f->a[j] : val2_d ? (float)val2_d->a[j] : 1.0f;
      pos++;
    }
    out_off->a[i + 1] = (int64_t)pos;
  }
  return 3;
}

static int tm_csr_standardize (lua_State *L)
{
  tk_ivec_t *tokens = tk_ivec_peek(L, 2, "tokens");
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
  scores = tk_fvec_create(L, n_tokens, NULL, NULL);
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

static luaL_Reg tm_csr_fns[] = {
  { "seq_select", tm_csr_seq_select },
  { "label_union", tm_csr_label_union },
  { "sort_csr_desc", tm_csr_sort_csr_desc },
  { "truncate", tm_csr_truncate },
  { "transpose", tm_csr_transpose },
  { "tokenize", tm_csr_tokenize },
  { "sqrt", tm_csr_sqrt },
  { "apply_bns", tm_csr_apply_bns },
  { "apply_auc", tm_csr_apply_auc },
  { "merge", tm_csr_merge },
  { "standardize", tm_csr_standardize },
  { NULL, NULL }
};

int luaopen_santoku_learn_csr (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_csr_fns, 0);
  return 1;
}
