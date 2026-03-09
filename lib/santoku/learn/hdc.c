#include <lua.h>
#include <lauxlib.h>

#include <stdlib.h>
#include <string.h>
#include <omp.h>
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
  uint64_t *bsc_shifted_base;
  uint64_t *level_atoms;
  double *level_thresholds;
  double *feature_weights;
  int64_t *window_offsets;
  int64_t d;
  int64_t d_words;
  int64_t hdc_ngram;
  uint64_t hdc_seed;
  int64_t n_tokens;
  int64_t n_dims;
  int64_t row_length;
  int64_t window_size;
  int64_t bits_mode;
  int64_t n_levels;
} tk_hdc_t;

static inline tk_hdc_t *tk_hdc_peek (lua_State *L, int i) {
  return (tk_hdc_t *)luaL_checkudata(L, i, TK_HDC_MT);
}

static inline int tk_hdc_gc (lua_State *L) {
  tk_hdc_t *h = tk_hdc_peek(L, 1);
  free(h->bsc_shifted_base);
  h->bsc_shifted_base = NULL;
  free(h->level_atoms);
  h->level_atoms = NULL;
  free(h->level_thresholds);
  h->level_thresholds = NULL;
  free(h->feature_weights);
  h->feature_weights = NULL;
  free(h->window_offsets);
  h->window_offsets = NULL;
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

static inline int64_t tk_hdc_pack_window (const char *text, size_t pos, int64_t ws, const int64_t *wo)
{
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int64_t k = 0; k < ws; k++) {
    h ^= (uint64_t)(unsigned char)text[pos + (size_t)wo[k]];
    h *= 0x100000001b3ULL;
  }
  return (int64_t)h;
}

static void tk_hdc_bit_rotate (
  uint64_t * restrict dst, const uint64_t * restrict src,
  int64_t d, int64_t d_words, int64_t shift
)
{
  shift = ((shift % d) + d) % d;
  if (shift == 0) { memcpy(dst, src, (size_t)d_words * sizeof(uint64_t)); return; }
  if (d % 64 == 0) {
    int64_t ws = shift / 64;
    int64_t bs = shift % 64;
    if (bs == 0) {
      for (int64_t w = 0; w < d_words; w++)
        dst[w] = src[(w + ws) % d_words];
    } else {
      for (int64_t w = 0; w < d_words; w++) {
        int64_t s1 = (w + ws) % d_words;
        int64_t s2 = (s1 + 1) % d_words;
        dst[w] = (src[s1] >> bs) | (src[s2] << (64 - bs));
      }
    }
    return;
  }
  memset(dst, 0, (size_t)d_words * sizeof(uint64_t));
  for (int64_t j = 0; j < d; j++) {
    int64_t src_j = (j + shift) % d;
    if ((src[src_j / 64] >> (src_j % 64)) & 1)
      dst[j / 64] |= (1ULL << (j % 64));
  }
}

static void tk_hdc_gen_base_bsc (uint64_t *base, int64_t d, int64_t d_words, uint64_t seed)
{
  for (int c = 0; c < TK_HDC_N_ATOMS; c++) {
    uint64_t rng = seed ^ ((uint64_t)(unsigned)c * 0x517cc1b727220a95ULL);
    hdc_splitmix64(&rng);
    uint64_t *row = base + c * d_words;
    for (int64_t w = 0; w < d_words; w++)
      row[w] = hdc_splitmix64(&rng);
    if (d % 64 != 0)
      row[d_words - 1] &= (1ULL << (d % 64)) - 1;
  }
}

static inline void tk_hdc_bsc_waccumulate (
  float * restrict counts, const uint64_t * restrict bits,
  float weight, int64_t d, int64_t d_words
)
{
  for (int64_t w = 0; w < d_words; w++) {
    uint64_t word = bits[w];
    float *cnt = counts + w * 64;
    int64_t n = d - w * 64;
    if (n > 64) n = 64;
    for (int64_t b = 0; b < n; b++)
      cnt[b] += weight * (((int32_t)((word >> b) & 1)) * 2 - 1);
  }
}

static void tk_hdc_gen_levels (
  uint64_t *levels, int64_t n_levels, int64_t d, int64_t d_words, uint64_t seed
)
{
  uint64_t rng = seed ^ 0x4c4556454c53ULL;
  hdc_splitmix64(&rng);
  uint64_t *base = levels;
  for (int64_t w = 0; w < d_words; w++)
    base[w] = hdc_splitmix64(&rng);
  if (d % 64 != 0)
    base[d_words - 1] &= (1ULL << (d % 64)) - 1;
  if (n_levels <= 1) return;
  int64_t *perm = (int64_t *)malloc((size_t)d * sizeof(int64_t));
  for (int64_t i = 0; i < d; i++) perm[i] = i;
  uint64_t rng2 = seed ^ 0x5045524d55544ULL;
  hdc_splitmix64(&rng2);
  for (int64_t i = d - 1; i > 0; i--) {
    uint64_t r = hdc_splitmix64(&rng2);
    int64_t j = (int64_t)(r % (uint64_t)(i + 1));
    int64_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
  }
  for (int64_t lev = 1; lev < n_levels; lev++) {
    uint64_t *dst = levels + lev * d_words;
    memcpy(dst, base, (size_t)d_words * sizeof(uint64_t));
    int64_t n_flip = lev * d / (2 * (n_levels - 1));
    for (int64_t f = 0; f < n_flip; f++)
      dst[perm[f] / 64] ^= (1ULL << (perm[f] % 64));
  }
  free(perm);
}

static int tk_hdc_cmp_double (const void *a, const void *b)
{
  double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}

static void tk_hdc_compute_thresholds (
  const double *pool, int64_t pool_size,
  double *thresholds, int64_t n_levels, uint64_t seed
)
{
  int64_t cap = 1000000;
  int64_t sample_size = pool_size < cap ? pool_size : cap;
  double *sample = (double *)malloc((size_t)sample_size * sizeof(double));
  if (pool_size <= cap) {
    memcpy(sample, pool, (size_t)pool_size * sizeof(double));
  } else {
    memcpy(sample, pool, (size_t)cap * sizeof(double));
    uint64_t rng = seed ^ 0x5448524553484ULL;
    hdc_splitmix64(&rng);
    for (int64_t i = cap; i < pool_size; i++) {
      uint64_t r = hdc_splitmix64(&rng);
      int64_t j = (int64_t)(r % (uint64_t)(i + 1));
      if (j < cap) sample[j] = pool[i];
    }
  }
  qsort(sample, (size_t)sample_size, sizeof(double), tk_hdc_cmp_double);
  for (int64_t k = 0; k < n_levels - 1; k++) {
    int64_t idx = (k + 1) * sample_size / n_levels;
    if (idx >= sample_size) idx = sample_size - 1;
    thresholds[k] = sample[idx];
  }
  free(sample);
}

static inline int64_t tk_hdc_quantize (double val, const double *thresholds, int64_t n_levels)
{
  for (int64_t k = 0; k < n_levels - 1; k++)
    if (val <= thresholds[k]) return k;
  return n_levels - 1;
}

#define TK_HDC_SEED_ATOM(buf, seed, id, d, d_words, lrow) do { \
  uint64_t _rng = (seed) ^ ((uint64_t)(id) * 0x517cc1b727220a95ULL); \
  hdc_splitmix64(&_rng); \
  for (int64_t _w = 0; _w < (d_words); _w++) \
    (buf)[_w] = hdc_splitmix64(&_rng); \
  if ((d) % 64 != 0) \
    (buf)[(d_words) - 1] &= (1ULL << ((d) % 64)) - 1; \
  if (lrow) \
    for (int64_t _w = 0; _w < (d_words); _w++) \
      (buf)[_w] ^= (lrow)[_w]; \
} while (0)

#define TK_HDC_BIND_ROTATED(tmp, src, rot, d, d_words, k, ws) do { \
  tk_hdc_bit_rotate((rot), (src), (d), (d_words), (ws) - 1 - (k)); \
  if ((k) == 0) \
    memcpy((tmp), (rot), (size_t)(d_words) * sizeof(uint64_t)); \
  else \
    for (int64_t _j = 0; _j < (d_words); _j++) \
      (tmp)[_j] ^= (rot)[_j]; \
} while (0)

static void tk_hdc_project_dense (
  uint8_t * restrict out, const double * restrict codes,
  int64_t n_samples, tk_hdc_t *h, int64_t d_out
)
{
  int64_t n_dims = h->n_dims, d = h->d, d_words = h->d_words;
  int64_t out_row_bytes = (d_out + 7) / 8;
  int64_t ngram = h->hdc_ngram;
  uint64_t seed = h->hdc_seed;
  int64_t row_length = h->row_length, window_size = h->window_size;
  const int64_t *window_offsets = h->window_offsets;
  const double *fw = h->feature_weights;
  int64_t n_levels = h->n_levels;
  const uint64_t *level_atoms = h->level_atoms;
  const double *level_thresholds = h->level_thresholds;
  int64_t r_lim, c_lim;
  if (row_length > 0) {
    r_lim = n_dims / row_length - ngram + 1;
    c_lim = row_length - ngram + 1;
  } else {
    r_lim = 1;
    c_lim = n_dims - ngram + 1;
  }
  if (window_size == 1) { r_lim = 1; c_lim = n_dims; }
  if (r_lim < 1 || c_lim < 1) { r_lim = 0; c_lim = 0; }
  #pragma omp parallel
  {
    uint64_t * restrict tmp = (uint64_t *)tk_hdc_amalloc((size_t)d_words * sizeof(uint64_t));
    uint64_t * restrict scratch = (uint64_t *)tk_hdc_amalloc((size_t)d_words * sizeof(uint64_t));
    uint64_t * restrict rotated = (uint64_t *)tk_hdc_amalloc((size_t)d_words * sizeof(uint64_t));
    float * restrict counts = (float *)tk_hdc_amalloc((size_t)d * sizeof(float));
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      memset(counts, 0, (size_t)d * sizeof(float));
      if (window_size == 1) {
        for (int64_t i = 0; i < n_dims; i++) {
          double occ = codes[s * n_dims + i];
          double fwi = fw ? fw[i] : 1.0;
          const uint64_t *lrow = NULL;
          float wt;
          if (n_levels > 0) {
            lrow = level_atoms + tk_hdc_quantize(occ, level_thresholds + i * (n_levels - 1), n_levels) * d_words;
            wt = (float)fwi;
          } else {
            wt = (float)(occ * fwi);
          }
          TK_HDC_SEED_ATOM(scratch, seed, i, d, d_words, lrow);
          tk_hdc_bsc_waccumulate(counts, scratch, wt, d, d_words);
        }
      } else {
        for (int64_t ri = 0; ri < r_lim; ri++)
          for (int64_t ci = 0; ci < c_lim; ci++) {
            int64_t i = ri * row_length + ci;
            float weight = 0.0f;
            for (int64_t k = 0; k < window_size; k++) {
              int64_t dim_k = i + window_offsets[k];
              double occ = codes[s * n_dims + dim_k];
              double fwk = fw ? fw[dim_k] : 1.0;
              const uint64_t *lrow = NULL;
              if (n_levels > 0) {
                lrow = level_atoms + tk_hdc_quantize(occ, level_thresholds + dim_k * (n_levels - 1), n_levels) * d_words;
                weight += (float)fwk;
              } else {
                weight += (float)(occ * fwk);
              }
              TK_HDC_SEED_ATOM(scratch, seed, dim_k, d, d_words, lrow);
              TK_HDC_BIND_ROTATED(tmp, scratch, rotated, d, d_words, k, window_size);
            }
            tk_hdc_bsc_waccumulate(counts, tmp, weight / (float)window_size, d, d_words);
          }
      }
      {
        uint8_t *row = out + s * out_row_bytes;
        memset(row, 0, (size_t)out_row_bytes);
        for (int64_t j = 0; j < d_out; j++)
          if (counts[j] >= 0.0f)
            row[j / 8] |= (uint8_t)(1 << (j % 8));
      }
    }
    free(tmp);
    free(scratch);
    free(rotated);
    free(counts);
  }
}

static void tk_hdc_project_tokens (
  uint8_t * restrict out, const int64_t *offsets, const int64_t *tokens,
  const double * restrict values,
  int64_t n_samples, tk_hdc_t *h, int64_t d_out
)
{
  int64_t d = h->d, d_words = h->d_words;
  int64_t out_row_bytes = (d_out + 7) / 8;
  int64_t ngram = h->hdc_ngram;
  uint64_t seed = h->hdc_seed;
  int64_t row_length = h->row_length, window_size = h->window_size;
  const int64_t *window_offsets = h->window_offsets;
  const double *feature_weights = h->feature_weights;
  int64_t n_levels = h->n_levels;
  const uint64_t *level_atoms = h->level_atoms;
  const double *level_thresholds = h->level_thresholds;
  #pragma omp parallel
  {
    uint64_t * restrict tmp = (uint64_t *)tk_hdc_amalloc((size_t)d_words * sizeof(uint64_t));
    uint64_t * restrict scratch = (uint64_t *)tk_hdc_amalloc((size_t)d_words * sizeof(uint64_t));
    uint64_t * restrict rotated = (uint64_t *)tk_hdc_amalloc((size_t)d_words * sizeof(uint64_t));
    float * restrict counts = (float *)tk_hdc_amalloc((size_t)d * sizeof(float));
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      int64_t start = offsets[s];
      int64_t len = offsets[s + 1] - start;
      memset(counts, 0, (size_t)d * sizeof(float));
      if (window_size == 1) {
        for (int64_t i = 0; i < len; i++) {
          int64_t tid = tokens[start + i];
          double occ = values ? values[start + i] : 1.0;
          double fwt = feature_weights ? feature_weights[tid] : 1.0;
          const uint64_t *lrow = NULL;
          float wt;
          if (n_levels > 0) {
            lrow = level_atoms + tk_hdc_quantize(occ, level_thresholds, n_levels) * d_words;
            wt = (float)fwt;
          } else {
            wt = (float)(occ * fwt);
          }
          TK_HDC_SEED_ATOM(scratch, seed, tid, d, d_words, lrow);
          tk_hdc_bsc_waccumulate(counts, scratch, wt, d, d_words);
        }
      } else {
        int64_t r_lim, c_lim;
        if (row_length > 0) {
          r_lim = len / row_length - ngram + 1;
          c_lim = row_length - ngram + 1;
        } else {
          r_lim = 1;
          c_lim = len - ngram + 1;
        }
        if (r_lim < 1 || c_lim < 1) { r_lim = 0; c_lim = 0; }
        for (int64_t ri = 0; ri < r_lim; ri++)
          for (int64_t ci = 0; ci < c_lim; ci++) {
            int64_t i = ri * row_length + ci;
            float weight = 0.0f;
            for (int64_t k = 0; k < window_size; k++) {
              int64_t pos_k = start + i + window_offsets[k];
              int64_t tid = tokens[pos_k];
              double occ = values ? values[pos_k] : 1.0;
              double fwk = feature_weights ? feature_weights[tid] : 1.0;
              const uint64_t *lrow = NULL;
              if (n_levels > 0) {
                lrow = level_atoms + tk_hdc_quantize(occ, level_thresholds, n_levels) * d_words;
                weight += (float)fwk;
              } else {
                weight += (float)(occ * fwk);
              }
              TK_HDC_SEED_ATOM(scratch, seed, tid, d, d_words, lrow);
              TK_HDC_BIND_ROTATED(tmp, scratch, rotated, d, d_words, k, window_size);
            }
            tk_hdc_bsc_waccumulate(counts, tmp, weight / (float)window_size, d, d_words);
          }
      }
      {
        uint8_t *row = out + s * out_row_bytes;
        memset(row, 0, (size_t)out_row_bytes);
        for (int64_t j = 0; j < d_out; j++)
          if (counts[j] >= 0.0f)
            row[j / 8] |= (uint8_t)(1 << (j % 8));
      }
    }
    free(tmp);
    free(scratch);
    free(rotated);
    free(counts);
  }
}

static void tk_hdc_project_bits (
  uint8_t * restrict out, const uint8_t * restrict bits,
  int64_t n_samples, tk_hdc_t *h, int64_t d_out
)
{
  int64_t n_dims = h->n_dims, d = h->d, d_words = h->d_words;
  int64_t out_row_bytes = (d_out + 7) / 8;
  int64_t ngram = h->hdc_ngram;
  int64_t row_length = h->row_length, window_size = h->window_size;
  const int64_t *window_offsets = h->window_offsets;
  const uint64_t *bsc_base = h->bsc_shifted_base;
  const double *fw = h->feature_weights;
  int64_t in_row_bytes = (n_dims + 7) / 8;
  int64_t r_lim, c_lim;
  if (row_length > 0) {
    r_lim = n_dims / row_length - ngram + 1;
    c_lim = row_length - ngram + 1;
  } else {
    r_lim = 1;
    c_lim = n_dims - ngram + 1;
  }
  if (r_lim < 1 || c_lim < 1) { r_lim = 0; c_lim = 0; }
  #pragma omp parallel
  {
    uint64_t * restrict tmp = (uint64_t *)tk_hdc_amalloc((size_t)d_words * sizeof(uint64_t));
    uint64_t * restrict atom = (uint64_t *)tk_hdc_amalloc((size_t)d_words * sizeof(uint64_t));
    float * restrict counts = (float *)tk_hdc_amalloc((size_t)d * sizeof(float));
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      const uint8_t *sb = bits + s * in_row_bytes;
      memset(counts, 0, (size_t)d * sizeof(float));
      for (int64_t ri = 0; ri < r_lim; ri++)
        for (int64_t ci = 0; ci < c_lim; ci++) {
          int64_t i = ri * row_length + ci;
          float weight = 0.0f;
          for (int64_t k = 0; k < window_size; k++) {
            int64_t dim_k = i + window_offsets[k];
            TK_HDC_BIND_ROTATED(tmp, bsc_base + dim_k * d_words, atom, d, d_words, k, window_size);
            if ((sb[dim_k / 8] >> (dim_k % 8)) & 1) {
              float fwk = fw ? (float)fw[dim_k] : 1.0f;
              weight += fwk;
            }
          }
          tk_hdc_bsc_waccumulate(counts, tmp, weight / (float)window_size, d, d_words);
        }
      {
        uint8_t *row = out + s * out_row_bytes;
        memset(row, 0, (size_t)out_row_bytes);
        for (int64_t j = 0; j < d_out; j++)
          if (counts[j] >= 0.0f)
            row[j / 8] |= (uint8_t)(1 << (j % 8));
      }
    }
    free(tmp);
    free(atom);
    free(counts);
  }
}

static void tk_hdc_project_text (
  uint8_t * restrict out, const char **strs, const size_t *lens,
  int64_t n_samples, tk_hdc_t *h, int64_t d_out
)
{
  int64_t d = h->d, d_words = h->d_words;
  int64_t out_row_bytes = (d_out + 7) / 8;
  int64_t ngram = h->hdc_ngram;
  int64_t row_length = h->row_length, window_size = h->window_size;
  const int64_t *window_offsets = h->window_offsets;
  const uint64_t *bsc_shifted_base = h->bsc_shifted_base;
  tk_dumap_t *weight_map = h->weight_map;
  uint32_t wmap_end = tk_dumap_end(weight_map);
  #pragma omp parallel
  {
    uint64_t * restrict tmp = (uint64_t *)tk_hdc_amalloc((size_t)d_words * sizeof(uint64_t));
    float * restrict counts = (float *)tk_hdc_amalloc((size_t)d * sizeof(float));
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      const char *text = strs[s];
      int64_t len = (int64_t)lens[s];
      memset(counts, 0, (size_t)d * sizeof(float));
      int64_t r_lim, c_lim;
      if (row_length > 0) {
        r_lim = len / row_length - ngram + 1;
        c_lim = row_length - ngram + 1;
      } else {
        r_lim = 1;
        c_lim = len - ngram + 1;
      }
      if (r_lim < 1 || c_lim < 1) { r_lim = 0; c_lim = 0; }
      for (int64_t r = 0; r < r_lim; r++)
        for (int64_t c = 0; c < c_lim; c++) {
          int64_t i = r * row_length + c;
          int64_t packed = tk_hdc_pack_window(text, (size_t)i, window_size, window_offsets);
          uint32_t iter = tk_dumap_get(weight_map, packed);
          if (iter == wmap_end) continue;
          float weight = (float)tk_dumap_val(weight_map, iter);
          for (int64_t k = 0; k < window_size; k++) {
            int ck = (unsigned char)text[i + window_offsets[k]];
            int64_t shk = window_size - 1 - k;
            const uint64_t *ak = bsc_shifted_base + ((uint64_t)shk * TK_HDC_N_ATOMS + (uint64_t)ck) * (uint64_t)d_words;
            if (k == 0)
              memcpy(tmp, ak, (size_t)d_words * sizeof(uint64_t));
            else
              for (int64_t j = 0; j < d_words; j++)
                tmp[j] ^= ak[j];
          }
          tk_hdc_bsc_waccumulate(counts, tmp, weight, d, d_words);
        }
      {
        uint8_t *row = out + s * out_row_bytes;
        memset(row, 0, (size_t)out_row_bytes);
        for (int64_t j = 0; j < d_out; j++)
          if (counts[j] >= 0.0f)
            row[j / 8] |= (uint8_t)(1 << (j % 8));
      }
    }
    free(tmp);
    free(counts);
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
  int64_t *counts = (int64_t *)calloc((size_t)n_samples, sizeof(int64_t));
  #pragma omp parallel
  {
    uint8_t *my_row = (uint8_t *)calloc(row_bytes, 1);
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      const char *text = strs[s];
      size_t len = lens[s];
      memset(my_row, 0, row_bytes);
      int64_t cnt = 0;
      if ((int64_t)len >= ngram) {
        for (size_t i = 0; i <= len - (size_t)ngram; i++) {
          int64_t packed = tk_hdc_pack_ngram(text, i, ngram);
          uint32_t iter = tk_iumap_get(ngram_map, packed);
          int64_t tid = tk_iumap_val(ngram_map, iter);
          uint64_t byte = (uint64_t)tid / CHAR_BIT;
          uint8_t bit = (uint8_t)(1 << (tid % CHAR_BIT));
          if (!(my_row[byte] & bit)) {
            my_row[byte] |= bit;
            cnt++;
          }
        }
      }
      counts[s] = cnt;
    }
    free(my_row);
  }
  int64_t total = 0;
  for (int64_t s = 0; s < n_samples; s++) {
    int64_t c = counts[s];
    counts[s] = total;
    total += c;
  }
  tk_ivec_t *set_bits = tk_ivec_create(L, (uint64_t)total, 0, 0);
  set_bits->n = (uint64_t)total;
  #pragma omp parallel
  {
    uint8_t *my_row = (uint8_t *)calloc(row_bytes, 1);
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      const char *text = strs[s];
      size_t len = lens[s];
      memset(my_row, 0, row_bytes);
      int64_t pos = counts[s];
      if ((int64_t)len >= ngram) {
        for (size_t i = 0; i <= len - (size_t)ngram; i++) {
          int64_t packed = tk_hdc_pack_ngram(text, i, ngram);
          uint32_t iter = tk_iumap_get(ngram_map, packed);
          int64_t tid = tk_iumap_val(ngram_map, iter);
          uint64_t byte = (uint64_t)tid / CHAR_BIT;
          uint8_t bit = (uint8_t)(1 << (tid % CHAR_BIT));
          if (!(my_row[byte] & bit)) {
            my_row[byte] |= bit;
            set_bits->a[pos++] = s * (int64_t)n_tokens + tid;
          }
        }
      }
    }
    free(my_row);
  }
  free(counts);
  free(strs);
  free(lens);
  lua_pushvalue(L, map_idx);
  lua_pushvalue(L, map_idx + 1);
  lua_pushinteger(L, (lua_Integer)n_tokens);
  return 3;
}

static int tk_hdc_tokenize_csr_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "texts");
  if (!lua_istable(L, -1))
    return luaL_error(L, "tokenize_csr: texts required");
  int texts_idx = lua_gettop(L);
  int64_t ngram = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize_csr", "hdc_ngram");
  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "tokenize_csr", "n_samples");
  const char **strs = (const char **)malloc((uint64_t)n_samples * sizeof(const char *));
  size_t *lens = (size_t *)malloc((uint64_t)n_samples * sizeof(size_t));
  for (int64_t s = 0; s < n_samples; s++) {
    lua_rawgeti(L, texts_idx, (int)(s + 1));
    strs[s] = lua_tolstring(L, -1, &lens[s]);
    lua_pop(L, 1);
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
  }
  uint64_t n_tokens = (uint64_t)next_id;
  uint32_t map_end = tk_iumap_end(ngram_map);
  int64_t *sample_nuniq = (int64_t *)calloc((size_t)n_samples, sizeof(int64_t));
  #pragma omp parallel
  {
    int64_t *my_counts = (int64_t *)calloc((size_t)n_tokens, sizeof(int64_t));
    int64_t *my_dirty = (int64_t *)malloc((size_t)n_tokens * sizeof(int64_t));
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      const char *text = strs[s];
      size_t len = lens[s];
      int64_t n_dirty = 0;
      if ((int64_t)len >= ngram) {
        for (size_t i = 0; i <= len - (size_t)ngram; i++) {
          int64_t packed = tk_hdc_pack_ngram(text, i, ngram);
          uint32_t iter = tk_iumap_get(ngram_map, packed);
          if (iter == map_end) continue;
          int64_t tid = tk_iumap_val(ngram_map, iter);
          if (my_counts[tid] == 0)
            my_dirty[n_dirty++] = tid;
          my_counts[tid]++;
        }
      }
      sample_nuniq[s] = n_dirty;
      for (int64_t j = 0; j < n_dirty; j++)
        my_counts[my_dirty[j]] = 0;
    }
    free(my_counts);
    free(my_dirty);
  }
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n_samples + 1), 0, 0);
  offsets->n = (uint64_t)(n_samples + 1);
  offsets->a[0] = 0;
  int64_t total = 0;
  for (int64_t s = 0; s < n_samples; s++) {
    total += sample_nuniq[s];
    offsets->a[s + 1] = total;
  }
  free(sample_nuniq);
  tk_ivec_t *tok_out = tk_ivec_create(L, (uint64_t)total, 0, 0);
  tok_out->n = (uint64_t)total;
  tk_dvec_t *freq_out = tk_dvec_create(L, (uint64_t)total, NULL, NULL);
  freq_out->n = (uint64_t)total;
  #pragma omp parallel
  {
    int64_t *my_counts = (int64_t *)calloc((size_t)n_tokens, sizeof(int64_t));
    int64_t *my_dirty = (int64_t *)malloc((size_t)n_tokens * sizeof(int64_t));
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < n_samples; s++) {
      const char *text = strs[s];
      size_t len = lens[s];
      int64_t n_dirty = 0;
      int64_t pos = offsets->a[s];
      if ((int64_t)len >= ngram) {
        for (size_t i = 0; i <= len - (size_t)ngram; i++) {
          int64_t packed = tk_hdc_pack_ngram(text, i, ngram);
          uint32_t iter = tk_iumap_get(ngram_map, packed);
          if (iter == map_end) continue;
          int64_t tid = tk_iumap_val(ngram_map, iter);
          if (my_counts[tid] == 0)
            my_dirty[n_dirty++] = tid;
          my_counts[tid]++;
        }
      }
      for (int64_t j = 0; j < n_dirty; j++) {
        int64_t tid = my_dirty[j];
        tok_out->a[pos + j] = tid;
        freq_out->a[pos + j] = (double)my_counts[tid];
        my_counts[tid] = 0;
      }
    }
    free(my_counts);
    free(my_dirty);
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
  } else if (h->n_dims > 0) {
    if (lua_type(L, 2) == LUA_TUSERDATA) {
      tk_dvec_t *fw = tk_dvec_peek(L, 2, "feature_weights");
      free(h->feature_weights);
      h->feature_weights = (double *)malloc((size_t)h->n_dims * sizeof(double));
      memcpy(h->feature_weights, fw->a, (size_t)h->n_dims * sizeof(double));
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
  int64_t d = h->d;
  int64_t dw = h->d_words;
  int64_t ng = h->hdc_ngram;
  h->window_size = (h->row_length > 0) ? ng * ng : ng;
  int64_t ws = h->window_size;
  h->window_offsets = (int64_t *)malloc((size_t)ws * sizeof(int64_t));
  if (h->row_length > 0) {
    for (int64_t pr = 0; pr < ng; pr++)
      for (int64_t pc = 0; pc < ng; pc++)
        h->window_offsets[pr * ng + pc] = pr * h->row_length + pc;
  } else {
    for (int64_t k = 0; k < ng; k++)
      h->window_offsets[k] = k;
  }
  if (h->n_levels > 0) {
    h->level_atoms = (uint64_t *)tk_hdc_amalloc((size_t)h->n_levels * (size_t)dw * sizeof(uint64_t));
    tk_hdc_gen_levels(h->level_atoms, h->n_levels, d, dw, h->hdc_seed);
  }
  if (h->bits_mode) {
    size_t base_bytes = (size_t)h->n_dims * (size_t)dw * sizeof(uint64_t);
    h->bsc_shifted_base = (uint64_t *)tk_hdc_amalloc(base_bytes);
    for (int64_t dim = 0; dim < h->n_dims; dim++) {
      uint64_t rng = h->hdc_seed ^ ((uint64_t)dim * 0x517cc1b727220a95ULL);
      hdc_splitmix64(&rng);
      uint64_t *row = h->bsc_shifted_base + dim * dw;
      for (int64_t w = 0; w < dw; w++)
        row[w] = hdc_splitmix64(&rng);
      if (d % 64 != 0)
        row[dw - 1] &= (1ULL << (d % 64)) - 1;
    }
    return;
  }
  if (h->n_tokens > 0 || h->n_dims > 0)
    return;
  size_t bsc_base_bytes = (size_t)TK_HDC_N_ATOMS * (size_t)dw * sizeof(uint64_t);
  uint64_t *bsc_base = (uint64_t *)tk_hdc_amalloc(bsc_base_bytes);
  tk_hdc_gen_base_bsc(bsc_base, d, dw, h->hdc_seed);
  size_t bsc_shifted_bytes = (size_t)ws * (size_t)TK_HDC_N_ATOMS * (size_t)dw * sizeof(uint64_t);
  h->bsc_shifted_base = (uint64_t *)tk_hdc_amalloc(bsc_shifted_bytes);
  for (int64_t sh = 0; sh < ws; sh++)
    for (int c = 0; c < TK_HDC_N_ATOMS; c++)
      tk_hdc_bit_rotate(
        h->bsc_shifted_base + (sh * TK_HDC_N_ATOMS + c) * dw,
        bsc_base + c * dw, d, dw, sh);
  free(bsc_base);
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
  lua_getfield(L, 1, "row_length");
  int64_t row_length = lua_isnumber(L, -1) ? (int64_t)lua_tointeger(L, -1) : 0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "offsets");
  bool has_offsets = !lua_isnil(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 1, "codes");
  bool has_codes = !lua_isnil(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 1, "bits");
  bool has_bits = !lua_isnil(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 1, "n_levels");
  int64_t n_levels = lua_isnumber(L, -1) ? (int64_t)lua_tointeger(L, -1) : 0;
  lua_pop(L, 1);
  if (n_levels < 2) n_levels = 0;
  tk_hdc_t *h = tk_lua_newuserdata(L, tk_hdc_t,
    TK_HDC_MT, tk_hdc_mt_fns, tk_hdc_gc);
  int Hi = lua_gettop(L);
  h->d = d;
  h->hdc_ngram = hdc_ngram;
  h->hdc_seed = hdc_seed;
  h->bsc_shifted_base = NULL;
  h->level_atoms = NULL;
  h->level_thresholds = NULL;
  h->feature_weights = NULL;
  h->window_offsets = NULL;
  h->weight_map = NULL;
  h->n_dims = 0;
  h->n_tokens = 0;
  h->n_levels = n_levels;
  h->row_length = row_length;
  h->window_size = 0;
  h->bits_mode = 0;
  h->d_words = (d + 63) / 64;
  double *codes_a = NULL;
  int64_t *off_a = NULL, *tok_a = NULL;
  double *vals_a = NULL;
  uint8_t *bits_a = NULL;
  const char **strs = NULL;
  size_t *lens = NULL;
  if (has_codes) {
    lua_getfield(L, 1, "codes");
    codes_a = tk_dvec_peek(L, -1, "codes")->a;
    h->n_dims = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_dims");
    lua_getfield(L, 1, "feature_weights");
    if (!lua_isnil(L, -1)) {
      tk_dvec_t *fw = tk_dvec_peek(L, -1, "feature_weights");
      h->feature_weights = (double *)malloc((size_t)h->n_dims * sizeof(double));
      memcpy(h->feature_weights, fw->a, (size_t)h->n_dims * sizeof(double));
    }
    lua_pop(L, 1);
    tk_hdc_init_cached(h);
    if (n_levels > 1) {
      int64_t n_per = n_levels - 1;
      h->level_thresholds = (double *)malloc((size_t)(h->n_dims * n_per) * sizeof(double));
      double *pool = (double *)malloc((size_t)n_samples * sizeof(double));
      for (int64_t dim = 0; dim < h->n_dims; dim++) {
        for (int64_t s = 0; s < n_samples; s++)
          pool[s] = codes_a[s * h->n_dims + dim];
        tk_hdc_compute_thresholds(pool, n_samples, h->level_thresholds + dim * n_per, n_levels, hdc_seed ^ (uint64_t)dim);
      }
      free(pool);
    }
    lua_newtable(L);
    lua_setfenv(L, Hi);
  } else if (has_offsets) {
    lua_getfield(L, 1, "offsets");
    off_a = tk_ivec_peek(L, -1, "offsets")->a;
    lua_getfield(L, 1, "tokens");
    tok_a = tk_ivec_peek(L, -1, "tokens")->a;
    h->n_tokens = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_tokens");
    lua_getfield(L, 1, "feature_weights");
    if (!lua_isnil(L, -1)) {
      tk_dvec_t *fw = tk_dvec_peek(L, -1, "feature_weights");
      h->feature_weights = (double *)malloc((size_t)h->n_tokens * sizeof(double));
      memcpy(h->feature_weights, fw->a, (size_t)h->n_tokens * sizeof(double));
    }
    lua_pop(L, 1);
    tk_hdc_init_cached(h);
    lua_getfield(L, 1, "values");
    if (!lua_isnil(L, -1))
      vals_a = tk_dvec_peek(L, -1, "values")->a;
    lua_pop(L, 1);
    if (n_levels > 1 && !vals_a)
      return luaL_error(L, "create: levels require values for token mode");
    if (n_levels > 1 && vals_a) {
      int64_t pool_size = off_a[n_samples];
      h->level_thresholds = (double *)malloc((size_t)(n_levels - 1) * sizeof(double));
      tk_hdc_compute_thresholds(vals_a, pool_size, h->level_thresholds, n_levels, hdc_seed);
    }
    lua_newtable(L);
    lua_setfenv(L, Hi);
  } else if (has_bits) {
    lua_getfield(L, 1, "bits");
    bits_a = (uint8_t *)tk_cvec_peek(L, -1, "bits")->a;
    h->n_dims = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_dims");
    h->bits_mode = 1;
    lua_getfield(L, 1, "feature_weights");
    if (!lua_isnil(L, -1)) {
      tk_dvec_t *fw = tk_dvec_peek(L, -1, "feature_weights");
      h->feature_weights = (double *)malloc((size_t)h->n_dims * sizeof(double));
      memcpy(h->feature_weights, fw->a, (size_t)h->n_dims * sizeof(double));
    }
    lua_pop(L, 1);
    tk_hdc_init_cached(h);
    lua_newtable(L);
    lua_setfenv(L, Hi);
  } else {
    lua_getfield(L, 1, "texts");
    if (!lua_istable(L, -1))
      return luaL_error(L, "create: texts, offsets, codes, or bits required");
    int texts_idx = lua_gettop(L);
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
    strs = (const char **)malloc((uint64_t)n_samples * sizeof(const char *));
    lens = (size_t *)malloc((uint64_t)n_samples * sizeof(size_t));
    for (int64_t s = 0; s < n_samples; s++) {
      lua_rawgeti(L, texts_idx, (int)(s + 1));
      strs[s] = lua_tolstring(L, -1, &lens[s]);
      lua_pop(L, 1);
    }
    tk_hdc_init_cached(h);
    lua_newtable(L);
    lua_pushvalue(L, wmap_idx);
    lua_setfield(L, -2, "weight_map");
    lua_setfenv(L, Hi);
  }
  int64_t out_row_bytes = (d + 7) / 8;
  uint64_t total_bytes = (uint64_t)n_samples * (uint64_t)out_row_bytes;
  tk_cvec_t *cout = tk_cvec_create(L, total_bytes, NULL, NULL);
  int out_idx = lua_gettop(L);
  uint8_t *out_bits = (uint8_t *)cout->a;
  if (has_codes)
    tk_hdc_project_dense(out_bits, codes_a, n_samples, h, d);
  else if (has_offsets)
    tk_hdc_project_tokens(out_bits, off_a, tok_a, vals_a, n_samples, h, d);
  else if (has_bits)
    tk_hdc_project_bits(out_bits, bits_a, n_samples, h, d);
  else
    tk_hdc_project_text(out_bits, strs, lens, n_samples, h, d);
  free(strs);
  free(lens);
  lua_pushvalue(L, Hi);
  lua_pushvalue(L, out_idx);
  return 2;
}

static int tk_hdc_encode_lua (lua_State *L)
{
  tk_hdc_t *h = tk_hdc_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 2, "encode", "n_samples");
  int64_t d = h->d;
  lua_getfield(L, 2, "n_truncated");
  int64_t d_out = lua_isnil(L, -1) ? d : (int64_t)lua_tonumber(L, -1);
  lua_pop(L, 1);
  if (d_out <= 0 || d_out > d) d_out = d;
  int64_t out_row_bytes = (d_out + 7) / 8;
  uint64_t total_bytes = (uint64_t)n_samples * (uint64_t)out_row_bytes;
  lua_getfield(L, 2, "out");
  tk_cvec_t *cout;
  if (!lua_isnil(L, -1)) {
    cout = tk_cvec_peek(L, -1, "out");
    tk_cvec_ensure(cout, total_bytes);
    cout->n = total_bytes;
  } else {
    lua_pop(L, 1);
    cout = tk_cvec_create(L, total_bytes, NULL, NULL);
  }
  int out_idx = lua_gettop(L);
  uint8_t *out_bits = (uint8_t *)cout->a;
  if (h->n_tokens > 0) {
    lua_getfield(L, 2, "offsets");
    tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_getfield(L, 2, "values");
    tk_dvec_t *vals = lua_isnil(L, -1) ? NULL : tk_dvec_peek(L, -1, "values");
    lua_pop(L, 1);
    tk_hdc_project_tokens(out_bits, offsets->a, tokens->a, vals ? vals->a : NULL, n_samples, h, d_out);
  } else if (h->n_dims > 0 && !h->bits_mode) {
    lua_getfield(L, 2, "codes");
    tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
    tk_hdc_project_dense(out_bits, codes->a, n_samples, h, d_out);
  } else if (h->bits_mode) {
    lua_getfield(L, 2, "bits");
    tk_cvec_t *bits = tk_cvec_peek(L, -1, "bits");
    tk_hdc_project_bits(out_bits, (const uint8_t *)bits->a, n_samples, h, d_out);
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
    tk_hdc_project_text(out_bits, strs, lens, n_samples, h, d_out);
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
  uint8_t version = 15;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &h->d, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &h->hdc_ngram, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &h->hdc_seed, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &h->n_tokens, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &h->row_length, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &h->n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &h->bits_mode, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &h->n_levels, sizeof(int64_t), 1, fh);
  if (h->n_levels > 1) {
    int64_t n_thresh = (h->n_dims > 0 && !h->bits_mode) ? h->n_dims * (h->n_levels - 1) : (h->n_levels - 1);
    tk_lua_fwrite(L, h->level_thresholds, sizeof(double), (size_t)n_thresh, fh);
  }
  if (h->n_tokens > 0) {
    uint8_t has_weights = h->feature_weights != NULL;
    tk_lua_fwrite(L, &has_weights, sizeof(uint8_t), 1, fh);
    if (has_weights)
      tk_lua_fwrite(L, h->feature_weights, sizeof(double), (size_t)h->n_tokens, fh);
  } else if (h->n_dims > 0) {
    uint8_t has_weights = h->feature_weights != NULL;
    tk_lua_fwrite(L, &has_weights, sizeof(uint8_t), 1, fh);
    if (has_weights)
      tk_lua_fwrite(L, h->feature_weights, sizeof(double), (size_t)h->n_dims, fh);
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
  if (version < 12 || version > 15) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported hdc version %d (expected 12-15)", (int)version);
  }
  int64_t d, hdc_ngram;
  uint64_t hdc_seed;
  tk_lua_fread(L, &d, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &hdc_ngram, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &hdc_seed, sizeof(uint64_t), 1, fh);
  int64_t n_tokens = 0;
  int64_t n_dims = 0;
  int64_t row_length = 0;
  int64_t bits_mode = 0;
  double *feature_weights = NULL;
  tk_dumap_t *weight_map = NULL;
  int wmap_idx = 0;
  tk_lua_fread(L, &n_tokens, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &row_length, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &bits_mode, sizeof(int64_t), 1, fh);
  int64_t n_levels = 0;
  double *level_thresholds = NULL;
  if (version >= 13 && version <= 14) {
    uint8_t dummy;
    tk_lua_fread(L, &dummy, sizeof(uint8_t), 1, fh);
  }
  if (version == 14) {
    uint8_t dummy;
    tk_lua_fread(L, &dummy, sizeof(uint8_t), 1, fh);
  }
  if (version >= 15) {
    tk_lua_fread(L, &n_levels, sizeof(int64_t), 1, fh);
    if (n_levels < 2) n_levels = 0;
    if (n_levels > 1) {
      int64_t n_thresh = (n_dims > 0 && !bits_mode) ? n_dims * (n_levels - 1) : (n_levels - 1);
      level_thresholds = (double *)malloc((size_t)n_thresh * sizeof(double));
      tk_lua_fread(L, level_thresholds, sizeof(double), (size_t)n_thresh, fh);
    }
  }
  if (n_tokens > 0) {
    uint8_t has_weights;
    tk_lua_fread(L, &has_weights, sizeof(uint8_t), 1, fh);
    if (has_weights) {
      feature_weights = (double *)malloc((size_t)n_tokens * sizeof(double));
      tk_lua_fread(L, feature_weights, sizeof(double), (size_t)n_tokens, fh);
    }
  } else if (version >= 15 && n_dims > 0) {
    uint8_t has_weights;
    tk_lua_fread(L, &has_weights, sizeof(uint8_t), 1, fh);
    if (has_weights) {
      feature_weights = (double *)malloc((size_t)n_dims * sizeof(double));
      tk_lua_fread(L, feature_weights, sizeof(double), (size_t)n_dims, fh);
    }
  } else if (n_dims == 0 && !bits_mode) {
    weight_map = tk_dumap_load(L, fh);
    wmap_idx = lua_gettop(L);
  }
  tk_lua_fclose(L, fh);
  tk_hdc_t *h = tk_lua_newuserdata(L, tk_hdc_t,
    TK_HDC_MT, tk_hdc_mt_fns, tk_hdc_gc);
  int Hi = lua_gettop(L);
  h->weight_map = weight_map;
  h->feature_weights = feature_weights;
  h->level_thresholds = level_thresholds;
  h->level_atoms = NULL;
  h->d = d;
  h->hdc_ngram = hdc_ngram;
  h->hdc_seed = hdc_seed;
  h->n_tokens = n_tokens;
  h->n_dims = n_dims;
  h->n_levels = n_levels;
  h->row_length = row_length;
  h->bits_mode = bits_mode;
  h->d_words = (d + 63) / 64;
  h->bsc_shifted_base = NULL;
  h->window_offsets = NULL;
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
  { "tokenize_csr", tk_hdc_tokenize_csr_lua },
  { "load", tk_hdc_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_hdc (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_hdc_fns, 0);
  return 1;
}
