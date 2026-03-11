#include <santoku/learn/csr.h>
#include <santoku/fvec.h>
#include <santoku/ivec/ext.h>
#include <santoku/iumap/ext.h>
#include <omp.h>
#include <assert.h>

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

static int tm_csr_stratified_sample (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *neighbors = tk_ivec_peek(L, 2, "neighbors");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_labels = tk_lua_checkunsigned(L, 4, "n_labels");
  uint64_t n_select = tk_lua_checkunsigned(L, 5, "n_select");

  if (n_select > n_samples) n_select = n_samples;
  if (n_select == n_samples) {
    tk_ivec_t *r = tk_ivec_create(L, n_samples, 0, 0);
    r->n = n_samples;
    for (uint64_t i = 0; i < n_samples; i++) r->a[i] = (int64_t)i;
    return 1;
  }

  uint64_t *lcnt = (uint64_t *)calloc(n_labels, sizeof(uint64_t));
  if (!lcnt) return luaL_error(L, "stratified_sample: alloc failed");
  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t lo = offsets->a[i], hi = offsets->a[i + 1];
    for (int64_t j = lo; j < hi; j++) {
      uint64_t lab = (uint64_t)neighbors->a[j];
      if (lab < n_labels) lcnt[lab]++;
    }
  }

  uint64_t *loff = (uint64_t *)malloc((n_labels + 1) * sizeof(uint64_t));
  if (!loff) { free(lcnt); return luaL_error(L, "stratified_sample: alloc failed"); }
  loff[0] = 0;
  for (uint64_t l = 0; l < n_labels; l++)
    loff[l + 1] = loff[l] + lcnt[l];
  uint64_t total_entries = loff[n_labels];

  uint64_t *lsamp = (uint64_t *)malloc(total_entries * sizeof(uint64_t));
  if (!lsamp) { free(lcnt); free(loff); return luaL_error(L, "stratified_sample: alloc failed"); }
  memset(lcnt, 0, n_labels * sizeof(uint64_t));
  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t lo = offsets->a[i], hi = offsets->a[i + 1];
    for (int64_t j = lo; j < hi; j++) {
      uint64_t lab = (uint64_t)neighbors->a[j];
      if (lab < n_labels) {
        lsamp[loff[lab] + lcnt[lab]] = i;
        lcnt[lab]++;
      }
    }
  }

  for (uint64_t l = 0; l < n_labels; l++) {
    uint64_t lo = loff[l], cnt = loff[l + 1] - lo;
    for (uint64_t i = cnt; i > 1; i--) {
      uint64_t j = tk_fast_random() % i;
      uint64_t tmp = lsamp[lo + i - 1];
      lsamp[lo + i - 1] = lsamp[lo + j];
      lsamp[lo + j] = tmp;
    }
  }

  uint64_t *lorder = (uint64_t *)malloc(n_labels * sizeof(uint64_t));
  uint64_t *lsz = (uint64_t *)malloc(n_labels * sizeof(uint64_t));
  if (!lorder || !lsz) {
    free(lcnt); free(loff); free(lsamp); free(lorder); free(lsz);
    return luaL_error(L, "stratified_sample: alloc failed");
  }
  for (uint64_t l = 0; l < n_labels; l++) {
    lorder[l] = l;
    lsz[l] = loff[l + 1] - loff[l];
  }
  for (uint64_t i = 1; i < n_labels; i++) {
    uint64_t key = lorder[i], ksz = lsz[lorder[i]];
    int64_t j = (int64_t)i - 1;
    while (j >= 0 && lsz[lorder[(uint64_t)j]] > ksz) {
      lorder[(uint64_t)j + 1] = lorder[(uint64_t)j];
      j--;
    }
    lorder[(uint64_t)j + 1] = key;
  }

  uint64_t bm_bytes = (n_samples + 7) / 8;
  uint8_t *sel = (uint8_t *)calloc(bm_bytes, 1);
  uint64_t *curs = (uint64_t *)calloc(n_labels, sizeof(uint64_t));
  if (!sel || !curs) {
    free(lcnt); free(loff); free(lsamp); free(lorder); free(lsz); free(sel); free(curs);
    return luaL_error(L, "stratified_sample: alloc failed");
  }

  tk_ivec_t *result = tk_ivec_create(L, n_select, 0, 0);
  uint64_t n_sel = 0;
  bool progress = true;
  while (n_sel < n_select && progress) {
    progress = false;
    for (uint64_t li = 0; li < n_labels && n_sel < n_select; li++) {
      uint64_t l = lorder[li];
      uint64_t lo = loff[l], cnt = loff[l + 1] - lo;
      while (curs[l] < cnt) {
        uint64_t sid = lsamp[lo + curs[l]];
        curs[l]++;
        if (!(sel[sid / 8] & (1 << (sid % 8)))) {
          sel[sid / 8] |= (uint8_t)(1 << (sid % 8));
          result->a[n_sel++] = (int64_t)sid;
          progress = true;
          break;
        }
      }
    }
  }

  if (n_sel < n_select) {
    for (uint64_t i = 0; i < n_samples && n_sel < n_select; i++) {
      if (!(sel[i / 8] & (1 << (i % 8)))) {
        result->a[n_sel++] = (int64_t)i;
      }
    }
  }

  result->n = n_sel;
  free(lcnt); free(loff); free(lsamp); free(lorder); free(lsz); free(sel); free(curs);
  return 1;
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

static int tm_csr_subsample (lua_State *L)
{
  lua_settop(L, 3);
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *neighbors = tk_ivec_peek(L, 2, "neighbors");
  tk_ivec_t *sample_ids = tk_ivec_peek(L, 3, "sample_ids");
  int64_t n = (int64_t)sample_ids->n;
  tk_ivec_t *new_off = tk_ivec_create(L, (uint64_t)(n + 1), NULL, NULL);
  int64_t total = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t sid = sample_ids->a[i];
    total += offsets->a[sid + 1] - offsets->a[sid];
  }
  tk_ivec_t *new_nbr = tk_ivec_create(L, (uint64_t)total, NULL, NULL);
  int64_t pos = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t sid = sample_ids->a[i];
    int64_t lo = offsets->a[sid];
    int64_t hi = offsets->a[sid + 1];
    new_off->a[i] = pos;
    for (int64_t j = lo; j < hi; j++)
      new_nbr->a[pos++] = neighbors->a[j];
  }
  new_off->a[n] = pos;
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

static int tm_csr_to_bits (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *tokens = tk_ivec_peek(L, 2, "tokens");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_tokens = tk_lua_checkunsigned(L, 4, "n_tokens");
  uint64_t total = 0;
  for (uint64_t s = 0; s < n_samples; s++)
    total += (uint64_t)(offsets->a[s + 1] - offsets->a[s]);
  tk_ivec_t *out = tk_ivec_create(L, total, 0, 0);
  out->n = total;
  uint64_t pos = 0;
  for (uint64_t s = 0; s < n_samples; s++) {
    int64_t lo = offsets->a[s], hi = offsets->a[s + 1];
    for (int64_t j = lo; j < hi; j++)
      out->a[pos++] = (int64_t)(s * n_tokens + (uint64_t)tokens->a[j]);
  }
  return 1;
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
    for (size_t i = 0; i < count; i++) {
      int64_t token_id;
      if (!have_map) {
        int absent;
        uint32_t it = tk_iumap_put(ngram_map, packed[i], &absent);
        if (absent) {
          tk_iumap_setval(ngram_map, it, next_id);
          token_id = next_id++;
        } else {
          token_id = tk_iumap_val(ngram_map, it);
        }
      } else {
        uint32_t it = tk_iumap_get(ngram_map, packed[i]);
        if (it == map_end) continue;
        token_id = tk_iumap_val(ngram_map, it);
      }
      if (tok_n >= (int64_t)tok_cap) {
        tok_cap *= 2;
        tok_buf = (int64_t *)realloc(tok_buf, tok_cap * sizeof(int64_t));
      }
      tok_buf[tok_n++] = token_id;
    }
    offsets->a[s + 1] = tok_n;
  }
  free(packed);
  tk_ivec_t *tok_out = tk_ivec_create(L, (uint64_t)tok_n, 0, 0);
  tok_out->n = (uint64_t)tok_n;
  if (tok_n > 0)
    memcpy(tok_out->a, tok_buf, (size_t)tok_n * sizeof(int64_t));
  free(tok_buf);
  uint64_t n_tokens = have_map ? (uint64_t)tk_iumap_size(ngram_map) : (uint64_t)next_id;
  lua_pushvalue(L, map_idx);
  lua_pushvalue(L, map_idx + 1);
  lua_pushvalue(L, map_idx + 2);
  lua_pushinteger(L, (lua_Integer)n_tokens);
  return 4;
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
  if (have_map) {
    ngram_map = tk_iumap_peek(L, -1, "ngram_map");
    map_idx = lua_gettop(L);
    next_id = (int64_t)tk_iumap_size(ngram_map);
  } else {
    lua_pop(L, 1);
    ngram_map = tk_iumap_create(L, 0);
    map_idx = lua_gettop(L);
    next_id = 0;
    int64_t *packed_buf = (int64_t *)malloc(max_len * sizeof(int64_t));
    for (int64_t s = 0; s < n_samples; s++) {
      size_t count = tm_csr_pack_ngrams(strs[s], lens[s], (int)ngram, packed_buf);
      for (size_t i = 0; i < count; i++) {
        int absent;
        uint32_t iter = tk_iumap_put(ngram_map, packed_buf[i], &absent);
        if (absent)
          tk_iumap_setval(ngram_map, iter, next_id++);
      }
    }
    free(packed_buf);
  }
  uint64_t n_tokens = (uint64_t)next_id;
  uint32_t map_end = tk_iumap_end(ngram_map);
  int64_t *sample_counts = (int64_t *)calloc((size_t)n_samples, sizeof(int64_t));
  #pragma omp parallel
  {
    int64_t *packed_buf = (int64_t *)malloc(max_len * sizeof(int64_t));
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      size_t count = tm_csr_pack_ngrams(strs[s], lens[s], (int)ngram, packed_buf);
      int64_t valid = 0;
      for (size_t i = 0; i < count; i++) {
        uint32_t iter = tk_iumap_get(ngram_map, packed_buf[i]);
        if (iter != map_end) valid++;
      }
      sample_counts[s] = valid;
    }
    free(packed_buf);
  }
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
  #pragma omp parallel
  {
    int64_t *packed_buf = (int64_t *)malloc(max_len * sizeof(int64_t));
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      size_t count = tm_csr_pack_ngrams(strs[s], lens[s], (int)ngram, packed_buf);
      int64_t pos = offsets->a[s];
      for (size_t i = 0; i < count; i++) {
        uint32_t iter = tk_iumap_get(ngram_map, packed_buf[i]);
        if (iter == map_end) continue;
        tok_out->a[pos++] = tk_iumap_val(ngram_map, iter);
      }
    }
    free(packed_buf);
  }
  free(strs);
  free(lens);
  lua_pushvalue(L, map_idx);
  lua_pushvalue(L, map_idx + 1);
  lua_pushvalue(L, map_idx + 2);
  lua_pushinteger(L, (lua_Integer)n_tokens);
  return 4;
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

static luaL_Reg tm_csr_fns[] = {
  { "to_bits", tm_csr_to_bits },
  { "seq_select", tm_csr_seq_select },
  { "stratified_sample", tm_csr_stratified_sample },
  { "label_union", tm_csr_label_union },
  { "subsample", tm_csr_subsample },
  { "sort_csr_desc", tm_csr_sort_csr_desc },
  { "truncate", tm_csr_truncate },
  { "transpose", tm_csr_transpose },
  { "tokenize", tm_csr_tokenize },
  { NULL, NULL }
};

int luaopen_santoku_learn_csr (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_csr_fns, 0);
  return 1;
}
