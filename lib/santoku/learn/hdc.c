#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <cblas.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/cvec.h>
#include <santoku/iumap.h>
#include <santoku/dumap.h>

#define TK_HDC_MT "tk_hdc_t"
#define TK_HDC_ALIGN 64
#define tk_hdc_amalloc(sz) aligned_alloc(TK_HDC_ALIGN, ((sz) + (TK_HDC_ALIGN - 1)) & ~(size_t)(TK_HDC_ALIGN - 1))

typedef struct {
  tk_dumap_t *weight_map;
  uint64_t *shifted_base;
  double *feature_weights;
  int64_t d;
  int64_t d_words;
  int64_t hdc_ngram;
  uint64_t hdc_seed;
  int64_t n_tokens;
} tk_hdc_t;

static inline tk_hdc_t *tk_hdc_peek (lua_State *L, int i) {
  return (tk_hdc_t *)luaL_checkudata(L, i, TK_HDC_MT);
}

static inline int tk_hdc_gc (lua_State *L) {
  tk_hdc_t *h = tk_hdc_peek(L, 1);
  free(h->shifted_base);
  h->shifted_base = NULL;
  free(h->feature_weights);
  h->feature_weights = NULL;
  h->weight_map = NULL;
  return 0;
}

static luaL_Reg tk_hdc_mt_fns[];

static inline uint64_t hdc_splitmix64 (uint64_t *s)
{
  uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

#define TK_HDC_N_ATOMS 256

static inline int64_t tk_hdc_pack_ngram (const char *text, size_t pos, int64_t ng)
{
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int64_t k = 0; k < ng; k++) {
    h ^= (uint64_t)(unsigned char)text[pos + (size_t)k];
    h *= 0x100000001b3ULL;
  }
  return (int64_t)h;
}

static double tk_hdc_sign_table[256][8];

static void tk_hdc_init_sign_table (void)
{
  for (int i = 0; i < 256; i++)
    for (int b = 0; b < 8; b++)
      tk_hdc_sign_table[i][b] = ((i >> b) & 1) ? -1.0 : 1.0;
}

static void tk_hdc_gen_base_packed (uint64_t *base, int64_t d, int64_t d_words, uint64_t seed)
{
  for (int c = 0; c < TK_HDC_N_ATOMS; c++) {
    uint64_t rng = seed ^ ((uint64_t)(unsigned)c * 0x517cc1b727220a95ULL);
    hdc_splitmix64(&rng);
    uint64_t *row = base + c * d_words;
    memset(row, 0, (size_t)d_words * sizeof(uint64_t));
    for (int64_t j = 0; j < d; j++) {
      if (!(hdc_splitmix64(&rng) & 1))
        row[j / 64] |= (1ULL << (j % 64));
    }
  }
}

static void tk_hdc_bit_rotate (
  uint64_t *restrict dst, const uint64_t *restrict src,
  int64_t d, int64_t d_words, int64_t shift
)
{
  memset(dst, 0, (size_t)d_words * sizeof(uint64_t));
  for (int64_t j = 0; j < d; j++) {
    int64_t src_j = (j + shift) % d;
    if ((src[src_j / 64] >> (src_j % 64)) & 1)
      dst[j / 64] |= (1ULL << (j % 64));
  }
}

static void tk_hdc_project (
  double *out, const char **strs, const size_t *lens,
  int64_t n_samples, int64_t d, int64_t d_words,
  int64_t ngram, const uint64_t *shifted_base,
  tk_dumap_t *weight_map
)
{
  uint32_t wmap_end = tk_dumap_end(weight_map);
  size_t d_bytes = (size_t)d * sizeof(double);
  size_t dw_bytes = (size_t)d_words * sizeof(uint64_t);
  #pragma omp parallel
  {
    uint64_t *tmp = (uint64_t *)tk_hdc_amalloc(dw_bytes);
    double *row = (double *)tk_hdc_amalloc(d_bytes);
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      const char *text = strs[s];
      size_t len = lens[s];
      memset(row, 0, d_bytes);
      if ((int64_t)len >= ngram) {
        size_t n_pos = len - (size_t)ngram + 1;
        for (size_t i = 0; i < n_pos; i++) {
          int64_t packed = tk_hdc_pack_ngram(text, i, ngram);
          uint32_t iter = tk_dumap_get(weight_map, packed);
          if (iter == wmap_end) continue;
          double weight = tk_dumap_val(weight_map, iter);
          int ci0 = (unsigned char)text[i];
          int64_t sh0 = ngram - 1;
          const uint64_t *a0 = shifted_base + ((uint64_t)sh0 * TK_HDC_N_ATOMS + (uint64_t)ci0) * (uint64_t)d_words;
          memcpy(tmp, a0, dw_bytes);
          for (int64_t k = 1; k < ngram; k++) {
            int ck = (unsigned char)text[i + (size_t)k];
            int64_t shk = ngram - 1 - k;
            const uint64_t *ak = shifted_base + ((uint64_t)shk * TK_HDC_N_ATOMS + (uint64_t)ck) * (uint64_t)d_words;
            for (int64_t j = 0; j < d_words; j++)
              tmp[j] ^= ak[j];
          }
          const uint8_t *bytes = (const uint8_t *)tmp;
          int64_t full_bytes = d / 8;
          int64_t tail_bits = d % 8;
          for (int64_t k = 0; k < full_bytes; k++) {
            const double *signs = tk_hdc_sign_table[bytes[k]];
            double *rp = row + k * 8;
            for (int m = 0; m < 8; m++)
              rp[m] += weight * signs[m];
          }
          if (tail_bits > 0) {
            const double *signs = tk_hdc_sign_table[bytes[full_bytes]];
            double *rp = row + full_bytes * 8;
            for (int64_t b = 0; b < tail_bits; b++)
              rp[b] += weight * signs[b];
          }
        }
      }
      double norm = cblas_dnrm2((int)d, row, 1);
      if (norm > 0.0)
        cblas_dscal((int)d, 1.0 / norm, row, 1);
      memcpy(out + s * d, row, d_bytes);
    }
    free(tmp);
    free(row);
  }
}

static inline void tk_hdc_gen_rotated_atom (
  uint64_t *dst, int64_t d, int64_t d_words,
  uint64_t seed, int64_t token_id, int64_t shift
)
{
  memset(dst, 0, (size_t)d_words * sizeof(uint64_t));
  uint64_t rng = seed ^ ((uint64_t)token_id * 0x517cc1b727220a95ULL);
  hdc_splitmix64(&rng);
  for (int64_t j = 0; j < d; j++) {
    if (!(hdc_splitmix64(&rng) & 1)) {
      int64_t dst_j = (j - shift % d + d) % d;
      dst[dst_j / 64] |= (1ULL << (dst_j % 64));
    }
  }
}

static void tk_hdc_project_tokens (
  double *out, const int64_t *offsets, const int64_t *tokens,
  int64_t n_samples, int64_t d, int64_t d_words,
  int64_t ngram, uint64_t seed, const double *feature_weights
)
{
  size_t d_bytes = (size_t)d * sizeof(double);
  size_t dw_bytes = (size_t)d_words * sizeof(uint64_t);
  #pragma omp parallel
  {
    uint64_t *tmp = (uint64_t *)tk_hdc_amalloc(dw_bytes);
    uint64_t *atom = (uint64_t *)tk_hdc_amalloc(dw_bytes);
    double *row = (double *)tk_hdc_amalloc(d_bytes);
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      int64_t start = offsets[s];
      int64_t end = offsets[s + 1];
      int64_t len = end - start;
      memset(row, 0, d_bytes);
      if (len >= ngram) {
        int64_t n_pos = len - ngram + 1;
        for (int64_t i = 0; i < n_pos; i++) {
          double weight = 0.0;
          tk_hdc_gen_rotated_atom(tmp, d, d_words, seed, tokens[start + i], ngram - 1);
          if (feature_weights)
            weight += feature_weights[tokens[start + i]];
          for (int64_t k = 1; k < ngram; k++) {
            int64_t sh = ngram - 1 - k;
            tk_hdc_gen_rotated_atom(atom, d, d_words, seed, tokens[start + i + k], sh);
            for (int64_t j = 0; j < d_words; j++)
              tmp[j] ^= atom[j];
            if (feature_weights)
              weight += feature_weights[tokens[start + i + k]];
          }
          double w = feature_weights ? weight / (double)ngram : 1.0;
          const uint8_t *bytes = (const uint8_t *)tmp;
          int64_t full_bytes = d / 8;
          int64_t tail_bits = d % 8;
          for (int64_t kb = 0; kb < full_bytes; kb++) {
            const double *signs = tk_hdc_sign_table[bytes[kb]];
            double *rp = row + kb * 8;
            for (int m = 0; m < 8; m++)
              rp[m] += w * signs[m];
          }
          if (tail_bits > 0) {
            const double *signs = tk_hdc_sign_table[bytes[full_bytes]];
            double *rp = row + full_bytes * 8;
            for (int64_t b = 0; b < tail_bits; b++)
              rp[b] += w * signs[b];
          }
        }
      }
      double norm = cblas_dnrm2((int)d, row, 1);
      if (norm > 0.0)
        cblas_dscal((int)d, 1.0 / norm, row, 1);
      memcpy(out + s * d, row, d_bytes);
    }
    free(tmp);
    free(atom);
    free(row);
  }
}

static tk_dumap_t *tk_hdc_build_weight_map (
  lua_State *L, tk_iumap_t *ngram_map, tk_ivec_t *ids, tk_dvec_t *weights
)
{
  uint32_t map_size = tk_iumap_size(ngram_map);
  tk_dumap_t *dm = tk_dumap_create(L, map_size);
  if (ids) {
    double *weight_by_id = (double *)calloc(map_size, sizeof(double));
    for (size_t i = 0; i < ids->n; i++) {
      int64_t id = ids->a[i];
      if (id >= 0 && (uint32_t)id < map_size)
        weight_by_id[id] = weights->a[i];
    }
    int64_t key; int64_t val;
    tk_umap_foreach(ngram_map, key, val, ({
      if (weight_by_id[val] != 0.0) {
        int absent;
        uint32_t iter = tk_dumap_put(dm, key, &absent);
        tk_dumap_setval(dm, iter, weight_by_id[val]);
      }
    }))
    free(weight_by_id);
  } else {
    int64_t key; int64_t val;
    (void)val;
    tk_umap_foreach(ngram_map, key, val, ({
      int absent;
      uint32_t iter = tk_dumap_put(dm, key, &absent);
      tk_dumap_setval(dm, iter, 1.0);
    }))
  }
  return dm;
}

static int tk_hdc_tokenize_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "texts");
  if (!lua_istable(L, -1))
    return luaL_error(L, "tokenize: texts required");
  int texts_idx = lua_gettop(L);
  int64_t ngram = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize", "hdc_ngram");
  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize", "n_samples");
  const char **strs = (const char **)malloc((uint64_t)n_samples * sizeof(const char *));
  size_t *lens = (size_t *)malloc((uint64_t)n_samples * sizeof(size_t));
  for (int64_t s = 0; s < n_samples; s++) {
    lua_rawgeti(L, texts_idx, (int)(s + 1));
    strs[s] = lua_tolstring(L, -1, &lens[s]);
    lua_pop(L, 1);
  }
  tk_iumap_t *ngram_map = tk_iumap_create(L, 0);
  int map_idx = lua_gettop(L);
  int64_t next_id = 0;
  for (int64_t s = 0; s < n_samples; s++) {
    const char *text = strs[s];
    size_t len = lens[s];
    if ((int64_t)len >= ngram) {
      for (size_t i = 0; i <= len - (size_t)ngram; i++) {
        int64_t packed = tk_hdc_pack_ngram(text, i, ngram);
        int absent;
        uint32_t iter = tk_iumap_put(ngram_map, packed, &absent);
        if (absent)
          tk_iumap_setval(ngram_map, iter, next_id++);
      }
    }
  }
  uint64_t n_tokens = (uint64_t)next_id;
  uint64_t row_bytes = TK_CVEC_BITS_BYTES(n_tokens);
  uint8_t *tmp_row = (uint8_t *)calloc(row_bytes, 1);
  tk_ivec_t *set_bits = tk_ivec_create(L, 0, 0, 0);
  for (int64_t s = 0; s < n_samples; s++) {
    const char *text = strs[s];
    size_t len = lens[s];
    memset(tmp_row, 0, row_bytes);
    if ((int64_t)len >= ngram) {
      for (size_t i = 0; i <= len - (size_t)ngram; i++) {
        int64_t packed = tk_hdc_pack_ngram(text, i, ngram);
        uint32_t iter = tk_iumap_get(ngram_map, packed);
        int64_t tid = tk_iumap_val(ngram_map, iter);
        uint64_t byte = (uint64_t)tid / CHAR_BIT;
        uint8_t bit = (uint8_t)(1 << (tid % CHAR_BIT));
        if (!(tmp_row[byte] & bit)) {
          tmp_row[byte] |= bit;
          tk_ivec_push(set_bits, s * (int64_t)n_tokens + tid);
        }
      }
    }
  }
  free(tmp_row);
  free(strs);
  free(lens);
  lua_pushvalue(L, map_idx);
  lua_pushvalue(L, map_idx + 1);
  lua_pushinteger(L, (lua_Integer)n_tokens);
  return 3;
}

static int tk_hdc_reweight_lua (lua_State *L)
{
  tk_hdc_t *h = tk_hdc_peek(L, 1);
  if (h->n_tokens > 0) {
    if (lua_type(L, 2) == LUA_TUSERDATA) {
      tk_dvec_t *fw = tk_dvec_peek(L, 2, "feature_weights");
      free(h->feature_weights);
      h->feature_weights = (double *)malloc((size_t)h->n_tokens * sizeof(double));
      memcpy(h->feature_weights, fw->a, (size_t)h->n_tokens * sizeof(double));
    } else {
      free(h->feature_weights);
      h->feature_weights = NULL;
    }
  } else {
    luaL_checktype(L, 2, LUA_TUSERDATA);
    tk_iumap_t *ngram_map = tk_iumap_peek(L, 2, "ngram_map");
    tk_ivec_t *ids = NULL;
    tk_dvec_t *weights = NULL;
    if (lua_type(L, 3) == LUA_TUSERDATA) {
      ids = tk_ivec_peek(L, 3, "ids");
      weights = tk_dvec_peek(L, 4, "weights");
    }
    tk_dumap_t *dm = tk_hdc_build_weight_map(L, ngram_map, ids, weights);
    int dm_idx = lua_gettop(L);
    h->weight_map = dm;
    lua_getfenv(L, 1);
    lua_pushvalue(L, dm_idx);
    lua_setfield(L, -2, "weight_map");
    lua_pop(L, 1);
  }
  return 0;
}

static void tk_hdc_init_cached (tk_hdc_t *h)
{
  h->d_words = (h->d + 63) / 64;
  if (h->n_tokens > 0)
    return;
  int64_t dw = h->d_words;
  size_t base_bytes = (size_t)TK_HDC_N_ATOMS * (size_t)dw * sizeof(uint64_t);
  uint64_t *base = (uint64_t *)tk_hdc_amalloc(base_bytes);
  tk_hdc_gen_base_packed(base, h->d, dw, h->hdc_seed);
  size_t shifted_bytes = (size_t)h->hdc_ngram * (size_t)TK_HDC_N_ATOMS * (size_t)dw * sizeof(uint64_t);
  h->shifted_base = (uint64_t *)tk_hdc_amalloc(shifted_bytes);
  for (int64_t sh = 0; sh < h->hdc_ngram; sh++) {
    for (int c = 0; c < TK_HDC_N_ATOMS; c++) {
      uint64_t *src = base + c * dw;
      uint64_t *dst = h->shifted_base + (sh * TK_HDC_N_ATOMS + c) * dw;
      if (sh == 0)
        memcpy(dst, src, (size_t)dw * sizeof(uint64_t));
      else
        tk_hdc_bit_rotate(dst, src, h->d, dw, sh);
    }
  }
  free(base);
}

static int tk_hdc_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t d = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "d");
  int64_t hdc_ngram = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "hdc_ngram");
  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_samples");
  lua_getfield(L, 1, "hdc_seed");
  uint64_t hdc_seed = lua_isnumber(L, -1) ? (uint64_t)lua_tointeger(L, -1) : 42;
  lua_pop(L, 1);
  lua_getfield(L, 1, "offsets");
  bool token_mode = !lua_isnil(L, -1);
  lua_pop(L, 1);
  uint64_t total = (uint64_t)n_samples * (uint64_t)d;
  tk_dvec_t *out = tk_dvec_create(L, total, NULL, NULL);
  out->n = total;
  int out_idx = lua_gettop(L);
  tk_hdc_t *h = tk_lua_newuserdata(L, tk_hdc_t,
    TK_HDC_MT, tk_hdc_mt_fns, tk_hdc_gc);
  int Hi = lua_gettop(L);
  h->d = d;
  h->hdc_ngram = hdc_ngram;
  h->hdc_seed = hdc_seed;
  h->shifted_base = NULL;
  h->feature_weights = NULL;
  h->weight_map = NULL;
  if (token_mode) {
    lua_getfield(L, 1, "offsets");
    tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
    lua_getfield(L, 1, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    h->n_tokens = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_tokens");
    lua_getfield(L, 1, "feature_weights");
    if (!lua_isnil(L, -1)) {
      tk_dvec_t *fw = tk_dvec_peek(L, -1, "feature_weights");
      h->feature_weights = (double *)malloc((size_t)h->n_tokens * sizeof(double));
      memcpy(h->feature_weights, fw->a, (size_t)h->n_tokens * sizeof(double));
    }
    lua_pop(L, 1);
    tk_hdc_init_cached(h);
    tk_hdc_project_tokens(out->a, offsets->a, tokens->a, n_samples, d, h->d_words, hdc_ngram, hdc_seed, h->feature_weights);
    lua_newtable(L);
    lua_setfenv(L, Hi);
  } else {
    lua_getfield(L, 1, "texts");
    if (!lua_istable(L, -1))
      return luaL_error(L, "create: texts or offsets required");
    int texts_idx = lua_gettop(L);
    h->n_tokens = 0;
    lua_getfield(L, 1, "weight_map");
    if (lua_isnil(L, -1))
      return luaL_error(L, "create: weight_map required");
    tk_iumap_t *wmap = tk_iumap_peek(L, -1, "weight_map");
    tk_ivec_t *wids = NULL;
    tk_dvec_t *wts = NULL;
    lua_getfield(L, 1, "weight_ids");
    if (!lua_isnil(L, -1)) {
      wids = tk_ivec_peek(L, -1, "weight_ids");
      lua_getfield(L, 1, "weights");
      wts = tk_dvec_peek(L, -1, "weights");
    } else {
      lua_pop(L, 1);
    }
    tk_dumap_t *weight_map = tk_hdc_build_weight_map(L, wmap, wids, wts);
    h->weight_map = weight_map;
    int wmap_idx = lua_gettop(L);
    const char **strs = (const char **)malloc((uint64_t)n_samples * sizeof(const char *));
    size_t *lens = (size_t *)malloc((uint64_t)n_samples * sizeof(size_t));
    for (int64_t s = 0; s < n_samples; s++) {
      lua_rawgeti(L, texts_idx, (int)(s + 1));
      strs[s] = lua_tolstring(L, -1, &lens[s]);
      lua_pop(L, 1);
    }
    tk_hdc_init_cached(h);
    tk_hdc_project(out->a, strs, lens, n_samples, d, h->d_words, hdc_ngram, h->shifted_base, weight_map);
    free(strs);
    free(lens);
    lua_newtable(L);
    lua_pushvalue(L, wmap_idx);
    lua_setfield(L, -2, "weight_map");
    lua_setfenv(L, Hi);
  }
  lua_pushvalue(L, Hi);
  lua_pushvalue(L, out_idx);
  return 2;
}

static int tk_hdc_encode_lua (lua_State *L)
{
  tk_hdc_t *h = tk_hdc_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 2, "encode", "n_samples");
  uint64_t total = (uint64_t)n_samples * (uint64_t)h->d;
  lua_getfield(L, 2, "out");
  tk_dvec_t *out;
  if (!lua_isnil(L, -1)) {
    out = tk_dvec_peek(L, -1, "out");
    out->n = total;
  } else {
    lua_pop(L, 1);
    out = tk_dvec_create(L, total, NULL, NULL);
    out->n = total;
  }
  int out_idx = lua_gettop(L);
  if (h->n_tokens > 0) {
    lua_getfield(L, 2, "offsets");
    tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    tk_hdc_project_tokens(out->a, offsets->a, tokens->a, n_samples, h->d, h->d_words, h->hdc_ngram, h->hdc_seed, h->feature_weights);
  } else {
    lua_getfield(L, 2, "texts");
    if (!lua_istable(L, -1))
      return luaL_error(L, "encode: texts required");
    int texts_idx = lua_gettop(L);
    const char **strs = (const char **)malloc((uint64_t)n_samples * sizeof(const char *));
    size_t *lens = (size_t *)malloc((uint64_t)n_samples * sizeof(size_t));
    for (int64_t s = 0; s < n_samples; s++) {
      lua_rawgeti(L, texts_idx, (int)(s + 1));
      strs[s] = lua_tolstring(L, -1, &lens[s]);
      lua_pop(L, 1);
    }
    tk_hdc_project(out->a, strs, lens, n_samples, h->d, h->d_words, h->hdc_ngram, h->shifted_base, h->weight_map);
    free(strs);
    free(lens);
  }
  lua_pushvalue(L, out_idx);
  return 1;
}

static int tk_hdc_out_d_lua (lua_State *L) {
  tk_hdc_t *h = tk_hdc_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)h->d);
  return 1;
}

static int tk_hdc_persist_lua (lua_State *L) {
  tk_hdc_t *h = tk_hdc_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting filepath or true");
  tk_lua_fwrite(L, "TKhd", 1, 4, fh);
  uint8_t version = 7;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &h->d, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &h->hdc_ngram, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &h->hdc_seed, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &h->n_tokens, sizeof(int64_t), 1, fh);
  if (h->n_tokens > 0) {
    uint8_t has_weights = h->feature_weights != NULL;
    tk_lua_fwrite(L, &has_weights, sizeof(uint8_t), 1, fh);
    if (has_weights)
      tk_lua_fwrite(L, h->feature_weights, sizeof(double), (size_t)h->n_tokens, fh);
  } else {
    tk_dumap_persist(L, h->weight_map, fh);
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

static int tk_hdc_load_lua (lua_State *L)
{
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKhd", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid hdc file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 5 && version != 6 && version != 7) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported hdc version %d (expected 5, 6, or 7)", (int)version);
  }
  int64_t d, hdc_ngram;
  uint64_t hdc_seed;
  tk_lua_fread(L, &d, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &hdc_ngram, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &hdc_seed, sizeof(uint64_t), 1, fh);
  int64_t n_tokens = 0;
  double *feature_weights = NULL;
  tk_dumap_t *weight_map = NULL;
  int wmap_idx = 0;
  if (version == 7) {
    tk_lua_fread(L, &n_tokens, sizeof(int64_t), 1, fh);
    if (n_tokens > 0) {
      uint8_t has_weights;
      tk_lua_fread(L, &has_weights, sizeof(uint8_t), 1, fh);
      if (has_weights) {
        feature_weights = (double *)malloc((size_t)n_tokens * sizeof(double));
        tk_lua_fread(L, feature_weights, sizeof(double), (size_t)n_tokens, fh);
      }
    } else {
      weight_map = tk_dumap_load(L, fh);
      wmap_idx = lua_gettop(L);
    }
  } else {
    if (version == 5) {
      uint8_t has_weights;
      tk_lua_fread(L, &has_weights, sizeof(uint8_t), 1, fh);
      if (!has_weights) {
        tk_lua_fclose(L, fh);
        return luaL_error(L, "hdc v5 file missing weight_map");
      }
    }
    weight_map = tk_dumap_load(L, fh);
    wmap_idx = lua_gettop(L);
  }
  tk_lua_fclose(L, fh);
  tk_hdc_t *h = tk_lua_newuserdata(L, tk_hdc_t,
    TK_HDC_MT, tk_hdc_mt_fns, tk_hdc_gc);
  int Hi = lua_gettop(L);
  h->weight_map = weight_map;
  h->feature_weights = feature_weights;
  h->d = d;
  h->hdc_ngram = hdc_ngram;
  h->hdc_seed = hdc_seed;
  h->n_tokens = n_tokens;
  h->shifted_base = NULL;
  tk_hdc_init_cached(h);
  lua_newtable(L);
  if (wmap_idx > 0) {
    lua_pushvalue(L, wmap_idx);
    lua_setfield(L, -2, "weight_map");
  }
  lua_setfenv(L, Hi);
  lua_pushvalue(L, Hi);
  return 1;
}

static luaL_Reg tk_hdc_mt_fns[] = {
  { "encode", tk_hdc_encode_lua },
  { "reweight", tk_hdc_reweight_lua },
  { "persist", tk_hdc_persist_lua },
  { "out_d", tk_hdc_out_d_lua },
  { NULL, NULL }
};

static luaL_Reg tk_hdc_fns[] = {
  { "create", tk_hdc_create_lua },
  { "tokenize", tk_hdc_tokenize_lua },
  { "load", tk_hdc_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_hdc (lua_State *L)
{
  tk_hdc_init_sign_table();
  lua_newtable(L);
  tk_lua_register(L, tk_hdc_fns, 0);
  return 1;
}
