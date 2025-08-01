#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>

#include <float.h>
#include <math.h>
#include <time.h>

#define MT_COREX "santoku_corex"

typedef enum {
  TK_CMP_INIT_SEED,
  TK_CMP_INIT_ALPHA,
  TK_CMP_INIT_TCS,
  TK_CMP_INIT_PYX_UNNORM,
  TK_CMP_MARGINALS,
  TK_CMP_MAXMIS,
  TK_CMP_ALPHA,
  TK_CMP_LATENT_ALL,
  TK_CMP_LATENT_BASELINE,
  TK_CMP_LATENT_SUMS,
  TK_CMP_LATENT_PY,
  TK_CMP_LATENT_NORM,
  TK_CMP_UPDATE_TC,
} tk_corex_stage_t;

typedef struct tk_corex_s tk_corex_t;
typedef struct tk_corex_thread_s tk_corex_thread_t;

typedef struct tk_corex_thread_s {
  tk_corex_t *C;
  uint64_t hfirst;
  uint64_t hlast;
  uint64_t vfirst;
  uint64_t vlast;
  unsigned int index;
} tk_corex_thread_t;

typedef struct tk_corex_sort_s {
  uint64_t s;
  unsigned int v;
} tk_corex_sort_t;

typedef struct tk_corex_s {
  bool trained; // already trained
  bool destroyed;
  double *alpha;
  double *log_marg;
  double *log_py;
  double *log_pyx_unnorm;
  size_t maxmis_len;
  double *maxmis;
  double *mis;
  double *log_z;
  double *sums;
  double *baseline;
  tk_corex_sort_t *sort;
  uint64_t n_set_bits;
  unsigned int n_samples;
  size_t samples_len;
  uint64_t *samples;
  size_t visibles_len;
  unsigned int *visibles;
  size_t px_len;
  double *px;
  size_t entropy_len;
  double *entropy_x;
  double *pyx;
  double *counts;
  double last_tc;
  double tc_dev;
  double *tcs;
  double *tcs_temp;
  double lam;
  double spa;
  double tmin;
  double ttc;
  double anchor;
  unsigned int tile_sblock;
  unsigned int tile_vblock;
  double smoothing;
  unsigned int n_visible;
  unsigned int n_hidden;
  bool initialized_threads;
  tk_threadpool_t *pool;
} tk_corex_t;

static tk_corex_t *peek_corex (lua_State *L, int i)
{
  return (tk_corex_t *) luaL_checkudata(L, i, MT_COREX);
}

static inline void tk_corex_shrink (tk_corex_t *C)
{
  free(C->log_z); C->log_z = NULL;
  free(C->sums); C->sums = NULL;
  free(C->counts); C->counts = NULL;
  free(C->tcs); C->tcs = NULL;
  free(C->tcs_temp); C->tcs_temp = NULL;
  if (numa_available() == -1) {
    free(C->maxmis); C->maxmis = NULL;
    free(C->px); C->px = NULL;
    free(C->entropy_x); C->entropy_x = NULL;
  } else {
    numa_free(C->maxmis, C->maxmis_len); C->maxmis = NULL; C->maxmis_len = 0;
    numa_free(C->entropy_x, C->entropy_len); C->entropy_x = NULL; C->entropy_len = 0;
    numa_free(C->px, C->px_len); C->px = NULL; C->px_len = 0;
  }
}

static int tk_corex_gc (lua_State *L)
{
  lua_settop(L, 1);
  tk_corex_t *C = peek_corex(L, 1);
  if (C->destroyed)
    return 1;
  C->destroyed = true;
  tk_corex_shrink(C);
  free(C->mis); C->mis = NULL;
  free(C->alpha); C->alpha = NULL;
  free(C->log_marg); C->log_marg = NULL;
  free(C->log_py); C->log_py = NULL;
  free(C->log_pyx_unnorm); C->log_pyx_unnorm = NULL;
  free(C->pyx); C->pyx = NULL;
  free(C->baseline); C->baseline = NULL;
  free(C->sort); C->sort = NULL;
  if (numa_available() == -1) {
    free(C->samples); C->samples = NULL;
    free(C->visibles); C->visibles = NULL;
  } else {
    numa_free(C->samples, C->samples_len); C->samples = NULL;
    numa_free(C->visibles, C->visibles_len); C->visibles = NULL;
  }
  for (unsigned int i = 0; i < C->pool->n_threads; i ++)
    free(C->pool->threads[i].data);
  tk_threads_destroy(C->pool);
  return 0;
}

static inline int tk_corex_destroy (lua_State *L)
{
  lua_settop(L, 0);
  lua_pushvalue(L, lua_upvalueindex(1));
  return tk_corex_gc(L);
}

static inline int tk_corex_compress (lua_State *);

static inline void tk_corex_marginals_thread (
  uint64_t *restrict samples,
  unsigned int *restrict visibles,
  double *restrict log_py,
  double *restrict pyx,
  double *restrict counts,
  double *restrict log_marg,
  double *restrict mis,
  double *restrict px,
  double *restrict entropy_x,
  double *restrict tcs,
  double *restrict tcs_temp,
  double smoothing,
  double tmin,
  double ttc,
  double anchor,
  uint64_t n_set_bits,
  unsigned int n_samples,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict lm00 = log_marg + 0 * n_hidden * n_visible;
  double *restrict lm01 = log_marg + 1 * n_hidden * n_visible;
  double *restrict lm10 = log_marg + 2 * n_hidden * n_visible;
  double *restrict lm11 = log_marg + 3 * n_hidden * n_visible;
  double *restrict pc00 = counts + 0 * n_hidden * n_visible;
  double *restrict pc01 = counts + 1 * n_hidden * n_visible;
  double *restrict pc10 = counts + 2 * n_hidden * n_visible;
  double *restrict pc11 = counts + 3 * n_hidden * n_visible;
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double *restrict pyx0 = pyx + h * n_samples;
    double counts_0 = smoothing;
    double counts_1 = smoothing;
    for (unsigned int s = 0; s < n_samples; s ++) {
      counts_0 += pyx0[s];
      counts_1 += (1 - pyx0[s]);
    }
    double sum_counts = counts_0 + counts_1;
    log_py[h] = log(counts_0) - log(sum_counts);
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double sum_py0 = 0.0;
    double *py0_h = pyx + h * n_samples;
    for (unsigned int s = 0; s < n_samples; s ++)
      sum_py0 += py0_h[s];
    double sum_py1 = n_samples - sum_py0;
    double *pc10h = pc10 + h * n_visible;
    double *pc11h = pc11 + h * n_visible;
    memset(pc10h, 0, n_visible * sizeof(double));
    memset(pc11h, 0, n_visible * sizeof(double));
    for (uint64_t c = 0; c < n_set_bits; c ++) {
      unsigned int v = visibles[c];
      uint64_t s = samples[c];
      double py0 = py0_h[s];
      pc10h[v] += py0;
      pc11h[v] += 1.0 - py0;
    }
    double *pc00h = pc00 + h * n_visible;
    double *pc01h = pc01 + h * n_visible;
    // TODO: can we avoid the inner if statements?
    for (unsigned int v = 0; v < n_visible; v ++) {
      pc00h[v] = (sum_py0 - pc10h[v]) + smoothing;
      if (pc00h[v] < smoothing)
        pc00h[v] = smoothing;
      pc01h[v] = (sum_py1 - pc11h[v]) + smoothing;
      if (pc01h[v] < smoothing)
        pc01h[v] = smoothing;
      pc10h[v] += smoothing;
      pc11h[v] += smoothing;
    }
  }
  for (unsigned int i = hfirst * n_visible; i < (hlast + 1) * n_visible; i ++) {
    double log_total0 = log(pc00[i] + pc01[i]);
    lm00[i] = log(pc00[i]) - log_total0;
    lm01[i] = log(pc01[i]) - log_total0;
  }
  for (unsigned int i = hfirst * n_visible; i < (hlast + 1) * n_visible; i ++) {
    double log_total1 = log(pc10[i] + pc11[i]);
    lm10[i] = log(pc10[i]) - log_total1;
    lm11[i] = log(pc11[i]) - log_total1;
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double lpy0v = log_py[h];
    double p0 = exp(lpy0v);
    double p1 = 1.0 - p0;
    double lpy1v = log1p(-fmin(p0, 1.0 - DBL_EPSILON));
    double *restrict lm00a = lm00 + h * n_visible;
    double *restrict lm01a = lm01 + h * n_visible;
    double *restrict lm10a = lm10 + h * n_visible;
    double *restrict lm11a = lm11 + h * n_visible;
    double *restrict pc00a = pc00 + h * n_visible;
    double *restrict pc01a = pc01 + h * n_visible;
    double *restrict pc10a = pc10 + h * n_visible;
    double *restrict pc11a = pc11 + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++) {
      lm00a[v] -= lpy0v;
      lm01a[v] -= lpy1v;
      lm10a[v] -= lpy0v;
      lm11a[v] -= lpy1v;
      pc00a[v] = exp(lm00a[v]) * p0;
      pc01a[v] = exp(lm01a[v]) * p1;
      pc10a[v] = exp(lm10a[v]) * p0;
      pc11a[v] = exp(lm11a[v]) * p1;
    }
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double *restrict pc00a = pc00 + h * n_visible;
    double *restrict pc01a = pc01 + h * n_visible;
    double *restrict pc10a = pc10 + h * n_visible;
    double *restrict pc11a = pc11 + h * n_visible;
    double *restrict lm00a = lm00 + h * n_visible;
    double *restrict lm10a = lm10 + h * n_visible;
    double *restrict lm01a = lm01 + h * n_visible;
    double *restrict lm11a = lm11 + h * n_visible;
    double *restrict mish = mis + h * n_visible;
    unsigned int anchor_feature_idx = anchor > 0.0 ? (n_visible - n_hidden) + h : n_visible;
    for (unsigned int v = 0; v < n_visible; v ++) {
      double group0 = pc00a[v] * lm00a[v] + pc01a[v] * lm01a[v];
      double group1 = pc10a[v] * lm10a[v] + pc11a[v] * lm11a[v];
      double mutual_info = group0 * (1 - px[v]) + group1 * px[v];
      if (v == anchor_feature_idx)
        mutual_info += anchor;
      else if (anchor > 0.0 && v >= (n_visible - n_hidden))
        mutual_info = 0.0;
      mish[v] = mutual_info;
    }
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, non-affine base
    double *restrict mish = mis + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++)
      mish[v] /= fmax(entropy_x[v], 1e-4);
  }
  for (unsigned int h = hfirst; h <= hlast; h ++)
    tcs_temp[h] = fabs(tcs[h]) * ttc + tmin;
}

static inline void tk_corex_maxmis_thread (
  double *restrict mis,
  double *restrict maxmis,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int vfirst,
  unsigned int vlast
) {
  for (unsigned int v = vfirst; v <= vlast; v ++) {
    double max_val = -DBL_MAX;
    for (unsigned int h = 0; h < n_hidden; h ++) {
      double candidate = mis[h * n_visible + v];
      if (candidate > max_val)
        max_val = candidate;
    }
    maxmis[v] = max_val;
  }
}

static inline void tk_corex_alpha_thread (
  double *restrict alpha,
  double *restrict baseline,
  double *restrict log_marg,
  double *restrict tcs_temp,
  double *restrict mis,
  double *restrict maxmis,
  double lam,
  double spa,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
){
  double *restrict baseline0 = baseline + 0 * n_hidden;
  double *restrict baseline1 = baseline + 1 * n_hidden;
  double *restrict lm00 = log_marg + 0 * n_hidden * n_visible;
  double *restrict lm01 = log_marg + 1 * n_hidden * n_visible;
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double scale = tcs_temp[h] / spa;
    double *restrict alphah = alpha + h * n_visible;
    double *restrict mish = mis + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++) {
      double update = exp(scale * (mish[v] - maxmis[v]));
      alphah[v] = (1.0 - lam) * alphah[v] + lam * update;
      if (alphah[v] < 0.0) alphah[v] = 0.0;
      else if (alphah[v] > 1.0) alphah[v] = 1.0;
    }
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double s0 = 0.0, s1 = 0.0;
    double *restrict lm00h = lm00 + h * n_visible;
    double *restrict lm01h = lm01 + h * n_visible;
    double *restrict alphah = alpha + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++) {
      s0 += alphah[v] * lm00h[v];
      s1 += alphah[v] * lm01h[v];
    }
    baseline0[h] = s0;
    baseline1[h] = s1;
  }
}

static inline void tk_corex_latent_baseline_thread (
  double *restrict sums,
  double *restrict baseline,
  unsigned int n_samples,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict sums0 = sums + 0 * n_hidden * n_samples;
  double *restrict sums1 = sums + 1 * n_hidden * n_samples;
  double *restrict baseline0 = baseline + 0 * n_hidden;
  double *restrict baseline1 = baseline + 1 * n_hidden;
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, not affine
    double s0 = baseline0[h];
    double s1 = baseline1[h];
    double *restrict sums0a = sums0 + h * n_samples;
    double *restrict sums1a = sums1 + h * n_samples;
    for (unsigned int i = 0; i < n_samples; i ++) {
      sums0a[i] = s0;
      sums1a[i] = s1;
    }
  }
}

static inline void tk_corex_latent_sums_thread (
  uint64_t *restrict samples,
  unsigned int *restrict visibles,
  double *restrict alpha,
  double *restrict log_marg,
  double *restrict sums,
  uint64_t n_set_bits,
  unsigned int n_samples,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict sums0 = sums + 0 * n_hidden * n_samples;
  double *restrict sums1 = sums + 1 * n_hidden * n_samples;
  double *restrict lm00 = log_marg + 0 * n_hidden * n_visible;
  double *restrict lm01 = log_marg + 1 * n_hidden * n_visible;
  double *restrict lm10 = log_marg + 2 * n_hidden * n_visible;
  double *restrict lm11 = log_marg + 3 * n_hidden * n_visible;
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double *restrict sums0h = sums0 + h * n_samples;
    double *restrict sums1h = sums1 + h * n_samples;
    double *restrict lm00a = lm00 + h * n_visible;
    double *restrict lm01a = lm01 + h * n_visible;
    double *restrict lm10a = lm10 + h * n_visible;
    double *restrict lm11a = lm11 + h * n_visible;
    double *restrict aph = alpha + h * n_visible;
    for (unsigned int b = 0; b < n_set_bits; b ++) {
      uint64_t s = samples[b];
      unsigned int v = visibles[b];
      sums0h[s] = sums0h[s] - aph[v] * lm00a[v] + aph[v] * lm10a[v];
      sums1h[s] = sums1h[s] - aph[v] * lm01a[v] + aph[v] * lm11a[v];
    }
  }
}

static inline void tk_corex_latent_py_thread (
  double *restrict log_py,
  double *restrict log_pyx_unnorm,
  double *restrict sums,
  unsigned int n_samples,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict sums0 = sums + 0 * n_hidden * n_samples;
  double *restrict sums1 = sums + 1 * n_hidden * n_samples;
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, not affine
    double lpy0v = log_py[h];
    double p0 = exp(lpy0v);
    double lpy1v = log1p(-fmin(p0, 1.0 - DBL_EPSILON));
    double *restrict sums0h = sums0 + h * n_samples;
    double *restrict sums1h = sums1 + h * n_samples;
    double *restrict lpyx0 = log_pyx_unnorm + 0 * n_hidden * n_samples + h * n_samples;
    double *restrict lpyx1 = log_pyx_unnorm + 1 * n_hidden * n_samples + h * n_samples;
    for (unsigned int i = 0; i < n_samples; i ++) {
      lpyx0[i] = sums0h[i] + lpy0v;
      lpyx1[i] = sums1h[i] + lpy1v;
    }
  }
}

static inline void tk_corex_latent_norm_thread (
  double *restrict log_z,
  double *restrict pyx,
  double *restrict log_pyx_unnorm,
  unsigned int n_samples,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict lpyx0 = log_pyx_unnorm + 0 * n_hidden * n_samples;
  double *restrict lpyx1 = log_pyx_unnorm + 1 * n_hidden * n_samples;
  for (unsigned int i = hfirst * n_samples; i < (hlast + 1) * n_samples; i ++) {
    double a = lpyx0[i];
    double b = lpyx1[i];
    double max_ab  = (a > b) ? a : b;
    double sum_exp = exp(a - max_ab) + exp(b - max_ab);
    log_z[i] = max_ab + log(sum_exp);
  }
  for (unsigned int i = hfirst * n_samples; i < (hlast + 1) * n_samples; i ++)
    pyx[i] = exp(lpyx0[i] - log_z[i]);
}

static inline void tk_corex_update_tc_thread (
  double *restrict log_z,
  double *restrict tcs,
  unsigned int n_samples,
  unsigned int hfirst,
  unsigned int hlast
) {
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, unsupported outer form
    const double *restrict lz = log_z + h * n_samples;
    double sum = 0.0;
    for (unsigned int s = 0; s < n_samples; s ++)
      sum += lz[s];
    double tc = sum / (double) n_samples;
    tcs[h] = tc;
  }
}

static inline void tk_corex_update_last_tc (
  double *restrict tcs,
  double *last_tc,
  double *tc_dev,
  unsigned int n_hidden
) {
  double min = tcs[0], max = tcs[0], sum = 0.0, sumsq = 0.0;
  for (unsigned int h = 0; h < n_hidden; h ++) {
    double tc = tcs[h];
    sum += tc;
    sumsq += tc * tc;
    if (tc < min) min = tc;
    if (tc > max) max = tc;
  }
  double mean = sum / n_hidden;
  double stdev = sqrt(sumsq / n_hidden - mean * mean);
  *last_tc = sum;
  *tc_dev = stdev;
}

static inline int tk_corex_sort_lt (tk_corex_sort_t a, tk_corex_sort_t b)
{
  if (a.v < b.v) return 1;
  if (a.v > b.v) return 0;
  if (a.s < b.s) return 1;
  if (a.s > b.s) return 0;
  return 0;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wsign-conversion"
KSORT_INIT(pairs, tk_corex_sort_t, tk_corex_sort_lt);
#pragma GCC diagnostic pop

typedef struct {
  tk_corex_sort_t *pairs;
  size_t capacity;
  size_t size;
} tk_corex_tile_t;

static void tk_corex_tile_push (
  lua_State *L,
  tk_corex_tile_t *tile,
  tk_corex_sort_t pair
) {
  if (tile->size >= tile->capacity) {
    size_t newcap = (tile->capacity == 0) ? 1024 : (tile->capacity * 2);
    tk_corex_sort_t *newtile = tk_realloc(L, tile->pairs, newcap * sizeof(tk_corex_sort_t));
    tile->pairs = newtile;
    tile->capacity = newcap;
  }
  tile->pairs[tile->size ++] = pair;
}

static void tk_corex_tile_pairs (
  lua_State *L,
  tk_corex_sort_t *pairs,
  size_t n_pairs,
  unsigned int max_v,
  unsigned int tile_sblock,
  unsigned int tile_vblock
) {
  size_t num_s_tiles = ((n_pairs - 1) / tile_sblock) + 1;
  size_t num_v_tiles = (max_v / tile_vblock) + 1;
  size_t total_tiles = num_s_tiles * num_v_tiles;
  tk_corex_tile_t *tiles = tk_malloc(L, total_tiles * sizeof(tk_corex_tile_t));
  memset(tiles, 0, sizeof(tk_corex_tile_t) * total_tiles);
  for (size_t i = 0; i < n_pairs; i ++) {
    uint64_t s = pairs[i].s;
    unsigned int v = pairs[i].v;
    size_t tile_s = (size_t)(s / tile_sblock);
    size_t tile_v = (size_t)(v / tile_vblock);
    size_t tile_index = tile_s * num_v_tiles + tile_v;
    tk_corex_tile_push(L, &tiles[tile_index], pairs[i]);
  }
  size_t out_idx = 0;
  for (size_t t = 0; t < total_tiles; t ++) {
    tk_corex_tile_t *tile = &tiles[t];
    ks_introsort(pairs, tile->size, tile->pairs);
    for (size_t j = 0; j < tile->size; j ++) {
      pairs[out_idx ++] = tile->pairs[j];
    }
    free(tile->pairs);
  }
  free(tiles);
}

static inline uint64_t tk_corex_setup_bits (
  lua_State *L,
  tk_ivec_t *set_bits,
  tk_corex_sort_t *pairs,
  uint64_t *samples,
  unsigned int *visibles,
  unsigned int n_visible,
  bool tile,
  unsigned int tile_sblock,
  unsigned int tile_vblock
) {
  uint64_t n = 0;
  for (uint64_t i = 0; i < set_bits->n; i ++) {
    int64_t val = set_bits->a[i];
    if (val < 0)
      continue;
    pairs[n] = (tk_corex_sort_t) {
      .s = (uint64_t) val / n_visible,
      .v = (unsigned int) ((uint64_t) val % n_visible)
    };
    n ++;
  }
  if (!tile)
    ks_introsort(pairs, n, pairs);
  else
    tk_corex_tile_pairs(L, pairs, n, n_visible, tile_sblock, tile_vblock);
  for (size_t i = 0; i < n; i ++) {
    samples[i]  = pairs[i].s;
    visibles[i] = pairs[i].v;
  }
  return n;
}

static inline void tk_corex_data_stats (
  uint64_t n_set_bits,
  unsigned int *restrict visibles,
  double *restrict px,
  double *restrict entropy_x,
  unsigned int n_samples,
  unsigned int n_visible
) {
  for (unsigned int v = 0; v < n_visible; v ++)
    px[v] = 0;
  for (uint64_t c = 0; c < n_set_bits; c ++)
    px[visibles[c]] ++;
  double frac = n_samples ? 1.0 / (double) n_samples : 0.0;
  for (unsigned int v = 0; v < n_visible; v ++) {
    px[v] *= frac;
    px[v] = fmin(fmax(px[v], 0.0), 1.0);
  }
  for (unsigned int v = 0; v < n_visible; ++v) {
    double p = px[v] > 0.0 ? px[v] : DBL_EPSILON;
    double q = (1.0 - px[v]) > 0.0 ? (1.0 - px[v]) : DBL_EPSILON;
    double entropy = -p * log(p) - q * log(q);
    entropy_x[v] = entropy > 0.0 ? entropy : DBL_EPSILON;
  }
}

static inline void tk_corex_init_alpha_thread (
  double *alpha,
  unsigned int n_visible,
  unsigned int hfirst,
  unsigned int hlast
) {
  for (unsigned int i = hfirst * n_visible; i < (hlast + 1) * n_visible; i ++)
    alpha[i] = 0.5 + 0.5 * fast_drand();
}

// Doesn't really need to be threaded..'
// Consider combining with one of the other thread inits
static inline void tk_corex_init_tcs_thread (
  double *tcs,
  double *tcs_temp,
  unsigned int hfirst,
  unsigned int hlast
) {
  for (unsigned int i = hfirst; i <= hlast; i ++)
    tcs[i] = 0.0;
  for (unsigned int i = hfirst; i <= hlast; i ++)
    tcs_temp[i] = 0.0;
}

static inline void tk_corex_init_log_pyx_unnorm_thread (
  double *log_pyx_unnorm,
  unsigned int n_samples,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double log_dim_hidden = -log(2);
  double *restrict lpyx0 = log_pyx_unnorm + 0 * n_hidden * n_samples;
  double *restrict lpyx1 = log_pyx_unnorm + 1 * n_hidden * n_samples;
  for (unsigned int i = hfirst * n_samples; i < (hlast + 1) * n_samples; i ++) {
    lpyx0[i] = log_dim_hidden * (0.5 + fast_drand());
    lpyx1[i] = log_dim_hidden * (0.5 + fast_drand());
  }
}

static void tk_corex_worker (void *dp, int sig)
{
  tk_corex_stage_t stage = (tk_corex_stage_t) sig;
  tk_corex_thread_t *data = (tk_corex_thread_t *) dp;
  tk_corex_t *C = data->C;
  switch (stage) {
    case TK_CMP_INIT_SEED:
      seed_rand(data->index);
      break;
    case TK_CMP_INIT_ALPHA:
      tk_corex_init_alpha_thread(
        C->alpha,
        C->n_visible,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_INIT_TCS:
      tk_corex_init_tcs_thread(
        C->tcs,
        C->tcs_temp,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_INIT_PYX_UNNORM:
      tk_corex_init_log_pyx_unnorm_thread(
        C->log_pyx_unnorm,
        C->n_samples,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_MARGINALS:
      tk_corex_marginals_thread(
        C->samples,
        C->visibles,
        C->log_py,
        C->pyx,
        C->counts,
        C->log_marg,
        C->mis,
        C->px,
        C->entropy_x,
        C->tcs,
        C->tcs_temp,
        C->smoothing,
        C->tmin,
        C->ttc,
        C->anchor,
        C->n_set_bits,
        C->n_samples,
        C->n_visible,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_MAXMIS:
      tk_corex_maxmis_thread(
        C->mis,
        C->maxmis,
        C->n_visible,
        C->n_hidden,
        data->vfirst,
        data->vlast);
      break;
    case TK_CMP_ALPHA:
      tk_corex_alpha_thread(
        C->alpha,
        C->baseline,
        C->log_marg,
        C->tcs_temp,
        C->mis,
        C->maxmis,
        C->lam,
        C->spa,
        C->n_visible,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_LATENT_ALL:
      tk_corex_latent_baseline_thread(
        C->sums,
        C->baseline,
        C->n_samples,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      tk_corex_latent_sums_thread(
        C->samples,
        C->visibles,
        C->alpha,
        C->log_marg,
        C->sums,
        C->n_set_bits,
        C->n_samples,
        C->n_visible,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      tk_corex_latent_py_thread(
        C->log_py,
        C->log_pyx_unnorm,
        C->sums,
        C->n_samples,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      tk_corex_latent_norm_thread(
        C->log_z,
        C->pyx,
        C->log_pyx_unnorm,
        C->n_samples,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_LATENT_BASELINE:
      tk_corex_latent_baseline_thread(
        C->sums,
        C->baseline,
        C->n_samples,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_LATENT_SUMS:
      tk_corex_latent_sums_thread(
        C->samples,
        C->visibles,
        C->alpha,
        C->log_marg,
        C->sums,
        C->n_set_bits,
        C->n_samples,
        C->n_visible,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_LATENT_PY:
      tk_corex_latent_py_thread(
        C->log_py,
        C->log_pyx_unnorm,
        C->sums,
        C->n_samples,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_LATENT_NORM:
      tk_corex_latent_norm_thread(
        C->log_z,
        C->pyx,
        C->log_pyx_unnorm,
        C->n_samples,
        C->n_hidden,
        data->hfirst,
        data->hlast);
      break;
    case TK_CMP_UPDATE_TC:
      tk_corex_update_tc_thread(
        C->log_z,
        C->tcs,
        C->n_samples,
        data->hfirst,
        data->hlast);
      break;
    default:
      assert(false);
      break;
  }
}

static inline void tk_corex_setup_threads (
  lua_State *L,
  tk_corex_t *C,
  uint64_t n_set_bits,
  unsigned int n_samples
) {
  tk_threadpool_t *pool = C->pool;
  if (!C->initialized_threads) {
    for (unsigned int i = 0; i < C->pool->n_threads; i ++) {
      tk_corex_thread_t *data = tk_malloc(L, sizeof(tk_corex_thread_t));
      pool->threads[i].data = data;
      data->C = C;
      data->index = i;
      tk_thread_range(i, C->pool->n_threads, C->n_hidden, &data->hfirst, &data->hlast);
      tk_thread_range(i, C->pool->n_threads, C->n_visible, &data->vfirst, &data->vlast);
    }
    C->initialized_threads = true;
  }
}

static inline int tk_corex_top_visible (lua_State *L)
{
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  if (!C->trained)
    return tk_lua_verror(L, 1, "Can't extract features from untrained model!");
  unsigned int top_k = tk_lua_checkunsigned(L, 1, "top_k");
  tk_dvec_t scores = tk_dvec(C->mis, C->n_hidden * C->n_visible);
  tk_ivec_top_generic(L, &scores, C->n_visible, C->n_hidden, top_k, C->anchor > 0.0 ? C->n_hidden : 0);
  return 1;
}

static inline int tk_corex_compress (lua_State *L)
{
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  tk_ivec_t *set_bits = tk_ivec_peek(L, 1, "set_bits");
  unsigned int n_samples = tk_lua_optunsigned(L, 2, "n_samples", 1);

  // TODO: Expose shrink via the api, and only realloc if new size is larger than old
  C->sort = tk_realloc(L, C->sort, set_bits->n * sizeof(tk_corex_sort_t));
  C->samples = tk_ensure_interleaved(L, &C->samples_len, C->samples, set_bits->n * sizeof(uint64_t), false);
  C->visibles = tk_ensure_interleaved(L, &C->visibles_len, C->visibles, set_bits->n * sizeof(unsigned int), false);
  set_bits->n = tk_corex_setup_bits(L, set_bits, C->sort, C->samples, C->visibles, C->n_visible, false, 0, 0);
  C->n_samples = n_samples;
  C->n_set_bits = set_bits->n;
  tk_corex_setup_threads(L, C, set_bits->n, n_samples);
  C->log_z = tk_realloc(L, C->log_z, C->n_hidden * n_samples * sizeof(double));
  C->pyx = tk_realloc(L, C->pyx, C->n_hidden * n_samples * sizeof(double));
  C->log_pyx_unnorm = tk_realloc(L, C->log_pyx_unnorm, 2 * C->n_hidden * n_samples * sizeof(double));
  C->sums = tk_realloc(L, C->sums, 2 * C->n_hidden * n_samples * sizeof(double));
  tk_threads_signal(C->pool, TK_CMP_LATENT_ALL, 0);
  // TODO: Parallelize output
  tk_ivec_clear(set_bits);
  for (unsigned int h = 0; h < C->n_hidden; h ++) {
    for (unsigned int s = 0; s < n_samples; s ++) {
      double py0 = C->pyx[h * n_samples + s];
      // TODO: should this be reversed?
      if (py0 < 0.5)
        tk_ivec_push(set_bits, s * C->n_hidden + h);
    }
  }
  tk_ivec_shrink(set_bits);
  return 1;
}

static inline void _tk_corex_train (
  lua_State *L,
  tk_corex_t *C,
  tk_ivec_t *set_bits,
  unsigned int n_samples,
  unsigned int max_iter,
  int i_each
) {
  C->smoothing = 0.001;
  // C->smoothing = fmax(1e-10, 1.0 / (double) n_samples);
  C->pyx = tk_malloc(L, C->n_hidden * n_samples * sizeof(double));
  C->log_pyx_unnorm = tk_malloc(L, 2 * C->n_hidden * n_samples * sizeof(double));
  C->sums = tk_malloc(L, 2 * C->n_hidden * n_samples * sizeof(double));
  C->mis = tk_malloc(L, C->n_hidden * C->n_visible * sizeof(double));
  C->log_z = tk_malloc(L, C->n_hidden * n_samples * sizeof(double));
  C->sort = tk_malloc(L, set_bits->n * sizeof(tk_corex_sort_t));
  C->samples = tk_malloc_interleaved(L, &C->samples_len, set_bits->n * sizeof(uint64_t));
  C->visibles = tk_malloc_interleaved(L, &C->visibles_len, set_bits->n * sizeof(unsigned int));
  set_bits->n = tk_corex_setup_bits(L, set_bits, C->sort, C->samples, C->visibles, C->n_visible, true, C->tile_sblock, C->tile_vblock);
  C->n_samples = n_samples;
  C->n_set_bits = set_bits->n;
  tk_corex_data_stats(
    C->n_set_bits,
    C->visibles,
    C->px,
    C->entropy_x,
    C->n_samples,
    C->n_visible);
  tk_corex_setup_threads(L, C, C->n_set_bits, n_samples);
  tk_threads_signal(C->pool, TK_CMP_INIT_PYX_UNNORM, 0);
  tk_threads_signal(C->pool, TK_CMP_LATENT_NORM, 0);
  for (unsigned int i = 0; i < max_iter; i ++) {
    tk_threads_signal(C->pool, TK_CMP_MARGINALS, 0);
    tk_threads_signal(C->pool, TK_CMP_MAXMIS, 0);
    tk_threads_signal(C->pool, TK_CMP_ALPHA, 0);
    tk_threads_signal(C->pool, TK_CMP_LATENT_ALL, 0);
    tk_threads_signal(C->pool, TK_CMP_UPDATE_TC, 0);
    tk_corex_update_last_tc(
      C->tcs,
      &C->last_tc,
      &C->tc_dev,
      C->n_hidden);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      lua_pushnumber(L, C->last_tc);
      lua_pushnumber(L, C->tc_dev);
      lua_call(L, 3, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break;
      } else {
        lua_pop(L, 1);
      }
    }
  }
  tk_corex_shrink(C);
  C->trained = true;
}

static inline void tk_corex_init (
  lua_State *L,
  tk_corex_t *C,
  double lam,
  double spa,
  double tmin,
  double ttc,
  double anchor,
  unsigned int tile_sblock,
  unsigned int tile_vblock,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int n_threads
) {
  memset(C, 0, sizeof(tk_corex_t));
  C->n_visible = n_visible;
  C->n_hidden = n_hidden;
  C->lam = lam;
  C->spa = spa;
  C->tmin = tmin;
  C->ttc = ttc;
  C->anchor = anchor;
  C->tile_sblock = tile_sblock;
  C->tile_vblock = tile_vblock;
  C->tcs = tk_malloc(L, C->n_hidden * sizeof(double));
  C->tcs_temp = tk_malloc(L, C->n_hidden * sizeof(double));
  C->alpha = tk_malloc(L, C->n_hidden * C->n_visible * sizeof(double));
  C->log_py = tk_malloc(L, C->n_hidden * sizeof(double));
  C->log_marg = tk_malloc(L, 2 * 2 * C->n_hidden * C->n_visible * sizeof(double));
  C->counts = tk_malloc(L, 2 * 2 * C->n_hidden * C->n_visible * sizeof(double));
  C->baseline = tk_malloc(L, 2 * C->n_hidden * sizeof(double));
  C->px = tk_malloc_interleaved(L, &C->px_len, C->n_visible * sizeof(double));
  C->entropy_x = tk_malloc_interleaved(L, &C->entropy_len, C->n_visible * sizeof(double));
  C->maxmis = tk_malloc_interleaved(L, &C->maxmis_len, C->n_visible * sizeof(double));
  C->pool = tk_threads_create(L, n_threads, tk_corex_worker);
  tk_corex_setup_threads(L, C, 0, 0);
  tk_threads_signal(C->pool, TK_CMP_INIT_SEED, 0);
  tk_threads_signal(C->pool, TK_CMP_INIT_TCS, 0);
  tk_threads_signal(C->pool, TK_CMP_INIT_ALPHA, 0);
}

static inline int tk_corex_visible (lua_State *L) {
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  lua_pushinteger(L, C->n_visible);
  return 1;
}

static inline int tk_corex_hidden (lua_State *L) {
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  lua_pushinteger(L, C->n_hidden);
  return 1;
}

static inline int tk_corex_train (lua_State *L) {
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  if (C->trained)
    return tk_lua_error(L, "Already trained!\n");
  lua_getfield(L, 1, "corpus");
  tk_ivec_t *set_bits = tk_ivec_peek(L, -1, "set_bits");
  unsigned int n_samples = tk_lua_fcheckunsigned(L, 1, "train", "samples");
  unsigned int max_iter = tk_lua_fcheckunsigned(L, 1, "train", "iterations");
  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }
  _tk_corex_train(L, C, set_bits, n_samples, max_iter, i_each); // c
  return 0;
}

static inline int tk_corex_persist (lua_State *L)
{
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  if (!C->trained)
    return tk_lua_error(L, "Can't persist an untrained model\n");
  lua_settop(L, 1);
  bool tostr = lua_type(L, 1) == LUA_TNIL;
  FILE *fh = tostr ? tk_lua_tmpfile(L) : tk_lua_fopen(L, luaL_checkstring(L, 1), "w");
  tk_lua_fwrite(L, &C->trained, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, &C->n_visible, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &C->n_hidden, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &C->lam, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &C->spa, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &C->tmin, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &C->ttc, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &C->anchor, sizeof(double), 1, fh);
  tk_lua_fwrite(L, C->alpha, sizeof(double), C->n_hidden * C->n_visible, fh);
  tk_lua_fwrite(L, C->log_py, sizeof(double), C->n_hidden, fh);
  tk_lua_fwrite(L, C->log_marg, sizeof(double), 2 * 2 * C->n_hidden * C->n_visible, fh);
  tk_lua_fwrite(L, C->baseline, sizeof(double), 2 * C->n_hidden, fh);
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

static luaL_Reg mt_fns[] =
{
  { "visible", tk_corex_visible },
  { "hidden", tk_corex_hidden },
  { "compress", tk_corex_compress },
  { "persist", tk_corex_persist },
  { "train", tk_corex_train },
  { "top_visible", tk_corex_top_visible },
  { "destroy", tk_corex_destroy },
  { NULL, NULL }
};

static inline int tk_corex_load (lua_State *L)
{
  lua_settop(L, 3); // fp ts
  tk_corex_t *C = (tk_corex_t *)
    lua_newuserdata(L, sizeof(tk_corex_t)); // tp ts c
  memset(C, 0, sizeof(tk_corex_t));
  luaL_getmetatable(L, MT_COREX); // fp ts c mt
  lua_setmetatable(L, -2); // fp ts c
  lua_newtable(L); // fp ts c t
  lua_pushvalue(L, -2); // fp ts c t c
  tk_lua_register(L, mt_fns, 1); // fp ts c t
  lua_remove(L, -2); // fp ts t
  unsigned int n_threads = tk_threads_getn(L, 2, "threads", NULL);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  bool isstr = lua_type(L, 3) == LUA_TBOOLEAN && lua_toboolean(L, 3);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  tk_lua_fread(L, &C->trained, sizeof(bool), 1, fh);
  tk_lua_fread(L, &C->n_visible, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &C->n_hidden, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &C->lam, sizeof(double), 1, fh);
  tk_lua_fread(L, &C->spa, sizeof(double), 1, fh);
  tk_lua_fread(L, &C->tmin, sizeof(double), 1, fh);
  tk_lua_fread(L, &C->ttc, sizeof(double), 1, fh);
  tk_lua_fread(L, &C->anchor, sizeof(double), 1, fh);
  C->alpha = tk_malloc(L, C->n_hidden * C->n_visible * sizeof(double));
  tk_lua_fread(L, C->alpha, sizeof(double), C->n_hidden * C->n_visible, fh);
  C->log_py = tk_malloc(L, C->n_hidden * sizeof(double));
  tk_lua_fread(L, C->log_py, sizeof(double), C->n_hidden, fh);
  C->log_marg = tk_malloc(L, 2 * 2 * C->n_hidden * C->n_visible * sizeof(double));
  tk_lua_fread(L, C->log_marg, sizeof(double), 2 * 2 * C->n_hidden * C->n_visible, fh);
  C->baseline = tk_malloc(L, 2 * C->n_hidden * sizeof(double));
  tk_lua_fread(L, C->baseline, sizeof(double), 2 * C->n_hidden, fh);
  tk_lua_fclose(L, fh);
  C->pool = tk_threads_create(L, n_threads, tk_corex_worker);
  tk_corex_setup_threads(L, C, 0, 0);
  return 1;
}

static inline int tk_corex_create (lua_State *L)
{
  lua_settop(L, 1);
  unsigned int n_visible = tk_lua_fcheckunsigned(L, 1, "create", "visible");
  unsigned int n_hidden = tk_lua_fcheckunsigned(L, 1, "create", "hidden");
  double lam = tk_lua_foptnumber(L, 1, "create", "lam", 0.3);
  double spa = tk_lua_foptnumber(L, 1, "create", "spa", 10.0);
  double tmin = tk_lua_foptnumber(L, 1, "create", "tmin", 1.0);
  double ttc = tk_lua_foptnumber(L, 1, "create", "ttc", 500.0);
  double anchor = tk_lua_foptnumber(L, 1, "create", "anchor", 0.0);
  unsigned int tile_sblock = tk_lua_foptunsigned(L, 1, "create", "tile_s", 1024);
  unsigned int tile_vblock = tk_lua_foptunsigned(L, 1, "create", "tile_v", 2048);
  unsigned int n_threads = tk_threads_getn(L, 1, "create", "threads");
  tk_corex_t *C = (tk_corex_t *) lua_newuserdata(L, sizeof(tk_corex_t)); // c
  luaL_getmetatable(L, MT_COREX); // c mt
  lua_setmetatable(L, -2); // c
  tk_corex_init(L, C, lam, spa, tmin, ttc, anchor, tile_sblock, tile_vblock, n_visible, n_hidden, n_threads); // c
  lua_newtable(L); // c t
  lua_pushvalue(L, -2); // c t c
  tk_lua_register(L, mt_fns, 1); // t
  return 1;
}

static luaL_Reg fns[] =
{
  { "create", tk_corex_create },
  { "load", tk_corex_load },
  { NULL, NULL }
};

int luaopen_santoku_corex (lua_State *L)
{
  lua_newtable(L); // t
  luaL_register(L, NULL, fns); // t
  luaL_newmetatable(L, MT_COREX); // t mt
  lua_pushcfunction(L, tk_corex_gc); // t mt fn
  lua_setfield(L, -2, "__gc"); // t mt
  lua_pop(L, 1); // t
  return 1;
}
