#ifndef TK_GRAM_H
#define TK_GRAM_H

#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/rvec.h>
#include <santoku/learn/buf.h>

#define TK_GRAM_MT "tk_gram_t"

typedef struct {
  double *evecs;
  float *evecs_f;
  double *eigenvals;
  double *PQtY;
  float *PQtY_f;
  bool PQtY_f_external;
  tk_dvec_t *label_counts;
  double *W_work;
  float *W_work_f;
  float *sbuf_f;
  float *val_F_f;
  double mean_eig;
  int64_t n_dims;
  int64_t n_labels;
  int64_t n_samples;
  int64_t val_n;
  int64_t tile_labels;
  double *col_mean;
  double *y_mean;
  double *cm_proj;
  double *intercept;
  int n_threads;
  tk_rank_t **thread_heaps;
  int64_t thread_heap_cap;
  uint8_t **thread_bitmaps;
  tk_rank_t *sample_heaps;
  int64_t sample_heap_cap;
  int64_t *gfm_labels;
  double *gfm_scores;
  tk_rank_t *gfm_pool;
  int64_t gfm_cap;
  bool destroyed;
} tk_gram_t;

__attribute__((unused)) static luaL_Reg tk_gram_mt_fns[];

static inline tk_gram_t *tk_gram_peek (lua_State *L, int i) {
  return (tk_gram_t *)luaL_checkudata(L, i, TK_GRAM_MT);
}

static inline void tk_gram_free_threads (tk_gram_t *g) {
  if (g->thread_heaps) {
    for (int t = 0; t < g->n_threads; t++)
      free(g->thread_heaps[t]);
    free(g->thread_heaps);
    g->thread_heaps = NULL;
  }
  if (g->thread_bitmaps) {
    for (int t = 0; t < g->n_threads; t++)
      free(g->thread_bitmaps[t]);
    free(g->thread_bitmaps);
    g->thread_bitmaps = NULL;
  }
  g->thread_heap_cap = 0;
  g->n_threads = 0;
}

static inline void tk_gram_ensure_heaps (tk_gram_t *g, int64_t cap) {
  int nt = omp_get_max_threads();
  if (g->thread_heaps && cap <= g->thread_heap_cap && nt <= g->n_threads)
    return;
  tk_gram_free_threads(g);
  g->n_threads = nt;
  g->thread_heaps = (tk_rank_t **)calloc((uint64_t)nt, sizeof(tk_rank_t *));
  g->thread_bitmaps = (uint8_t **)calloc((uint64_t)nt, sizeof(uint8_t *));
  for (int t = 0; t < nt; t++) {
    g->thread_heaps[t] = (tk_rank_t *)malloc((uint64_t)cap * sizeof(tk_rank_t));
    g->thread_bitmaps[t] = (uint8_t *)calloc((uint64_t)g->n_labels, sizeof(uint8_t));
  }
  g->thread_heap_cap = cap;
}

static inline void tk_gram_ensure_gfm (tk_gram_t *g, int64_t cap) {
  if (cap <= g->gfm_cap) return;
  free(g->gfm_labels);
  free(g->gfm_scores);
  free(g->gfm_pool);
  g->gfm_labels = (int64_t *)malloc((uint64_t)cap * sizeof(int64_t));
  g->gfm_scores = (double *)malloc((uint64_t)cap * sizeof(double));
  g->gfm_pool = (tk_rank_t *)malloc((uint64_t)cap * sizeof(tk_rank_t));
  g->gfm_cap = cap;
}

static inline int tk_gram_gc (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  free(g->evecs);
  free(g->evecs_f);
  free(g->eigenvals);
  free(g->PQtY);
  if (!g->PQtY_f_external) free(g->PQtY_f);
  free(g->W_work);
  free(g->W_work_f);
  free(g->sbuf_f);
  free(g->val_F_f);
  free(g->col_mean);
  free(g->y_mean);
  free(g->cm_proj);
  free(g->intercept);
  tk_gram_free_threads(g);
  free(g->sample_heaps);
  free(g->gfm_labels);
  free(g->gfm_scores);
  free(g->gfm_pool);
  g->label_counts = NULL;
  g->destroyed = true;
  return 0;
}

static inline void tk_gram_add_intercept (
  double *sbuf, int64_t bs, int64_t nl, double *intercept)
{
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < bs; i++)
    for (int64_t l = 0; l < nl; l++)
      sbuf[i * nl + l] += intercept[l];
}

static inline void tk_gram_add_intercept_f (
  float *sbuf, int64_t bs, int64_t nl, double *intercept)
{
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < bs; i++)
    for (int64_t l = 0; l < nl; l++)
      sbuf[i * nl + l] += (float)intercept[l];
}

static inline int tk_gram_finalize (
  lua_State *L,
  double *XtX,
  double *xty,
  double *col_mean,
  double *y_mean_arr,
  double *eigenvals,
  tk_dvec_t *lc,
  int lc_idx,
  int64_t nc, int64_t m, int64_t nl)
{
  uint64_t um = (uint64_t)m;
  uint64_t dd = um * um;
  uint64_t dnl = um * (uint64_t)nl;
  cblas_dsyr(CblasColMajor, CblasUpper, (int)m,
    -(double)nc, col_mean, 1, XtX, (int)m);
  cblas_dger(CblasRowMajor, (int)m, (int)nl,
    -(double)nc, col_mean, 1, y_mean_arr, 1, xty, (int)nl);
  LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (int)m, XtX, (int)m, eigenvals);
  double mean_eig = 0.0;
  for (uint64_t i = 0; i < um; i++)
    mean_eig += eigenvals[i];
  mean_eig /= (double)m;
  double *PQtY = (double *)malloc(dnl * sizeof(double));
  double *W_work = (double *)malloc(dnl * sizeof(double));
  if (!PQtY || !W_work) {
    free(PQtY); free(W_work); free(xty);
    free(XtX); free(eigenvals); free(col_mean); free(y_mean_arr);
    return luaL_error(L, "gram: out of memory");
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)m, (int)nl, (int)m, 1.0, XtX, (int)m,
    xty, (int)nl, 0.0, PQtY, (int)nl);
  free(xty);
  float *evecs_f = (float *)malloc(dd * sizeof(float));
  for (uint64_t i = 0; i < dd; i++)
    evecs_f[i] = (float)XtX[i];
  tk_gram_t *g = tk_lua_newuserdata(L, tk_gram_t,
    TK_GRAM_MT, tk_gram_mt_fns, tk_gram_gc);
  int gram_idx = lua_gettop(L);
  g->evecs = XtX;
  g->evecs_f = evecs_f;
  g->eigenvals = eigenvals;
  g->PQtY = PQtY;
  g->PQtY_f = (float *)malloc(dnl * sizeof(float));
  for (uint64_t i = 0; i < dnl; i++)
    g->PQtY_f[i] = (float)PQtY[i];
  g->PQtY_f_external = false;
  g->label_counts = lc;
  g->W_work = W_work;
  g->W_work_f = NULL;
  g->sbuf_f = NULL;
  g->val_F_f = NULL;
  g->col_mean = col_mean;
  g->y_mean = y_mean_arr;
  g->cm_proj = NULL;
  g->intercept = NULL;
  g->n_threads = 0;
  g->thread_heaps = NULL;
  g->thread_heap_cap = 0;
  g->thread_bitmaps = NULL;
  g->sample_heaps = NULL;
  g->sample_heap_cap = 0;
  g->gfm_labels = NULL;
  g->gfm_scores = NULL;
  g->gfm_pool = NULL;
  g->gfm_cap = 0;
  g->mean_eig = mean_eig;
  g->n_dims = m;
  g->n_labels = nl;
  g->n_samples = nc;
  g->tile_labels = 1024;
  g->val_n = 0;
  g->destroyed = false;
  lua_newtable(L);
  if (lc_idx > 0) {
    lua_pushvalue(L, lc_idx);
    lua_setfield(L, -2, "label_counts");
  }
  lua_setfenv(L, gram_idx);
  lua_pushvalue(L, gram_idx);
  return 1;
}


static inline int tk_gram_finalize_tiled (
  lua_State *L,
  double *XtX,
  double *col_mean,
  double *y_mean_arr,
  double *eigenvals,
  float *PQtY_f,
  bool pqty_external,
  tk_dvec_t *lc,
  int lc_idx,
  int64_t nc, int64_t m, int64_t nl,
  int64_t tile_labels)
{
  uint64_t um = (uint64_t)m;
  uint64_t dd = um * um;
  double mean_eig = 0.0;
  for (uint64_t i = 0; i < um; i++)
    mean_eig += eigenvals[i];
  mean_eig /= (double)m;
  float *evecs_f = (float *)malloc(dd * sizeof(float));
  for (uint64_t i = 0; i < dd; i++)
    evecs_f[i] = (float)XtX[i];
  tk_gram_t *g = tk_lua_newuserdata(L, tk_gram_t,
    TK_GRAM_MT, tk_gram_mt_fns, tk_gram_gc);
  int gram_idx = lua_gettop(L);
  g->evecs = XtX;
  g->evecs_f = evecs_f;
  g->eigenvals = eigenvals;
  g->PQtY = NULL;
  g->PQtY_f = PQtY_f;
  g->PQtY_f_external = pqty_external;
  g->label_counts = lc;
  g->W_work = NULL;
  g->W_work_f = NULL;
  g->sbuf_f = NULL;
  g->val_F_f = NULL;
  g->col_mean = col_mean;
  g->y_mean = y_mean_arr;
  g->cm_proj = NULL;
  g->intercept = NULL;
  g->n_threads = 0;
  g->thread_heaps = NULL;
  g->thread_heap_cap = 0;
  g->thread_bitmaps = NULL;
  g->sample_heaps = NULL;
  g->sample_heap_cap = 0;
  g->gfm_labels = NULL;
  g->gfm_scores = NULL;
  g->gfm_pool = NULL;
  g->gfm_cap = 0;
  g->mean_eig = mean_eig;
  g->n_dims = m;
  g->n_labels = nl;
  g->n_samples = nc;
  g->tile_labels = tile_labels;
  g->val_n = 0;
  g->destroyed = false;
  lua_newtable(L);
  if (lc_idx > 0) {
    lua_pushvalue(L, lc_idx);
    lua_setfield(L, -2, "label_counts");
  }
  lua_setfenv(L, gram_idx);
  lua_pushvalue(L, gram_idx);
  return 1;
}

static inline void tk_gram_solve_w (
  tk_gram_t *g, double lambda_raw, bool do_prop, double prop_a, double prop_b)
{
  int64_t d = g->n_dims, nl = g->n_labels;
  uint64_t dnl = (uint64_t)d * (uint64_t)nl;
  double mu = lambda_raw * g->mean_eig + g->mean_eig * 1e-7;
  if (do_prop && g->label_counts) {
    double C = (log((double)g->n_samples) - 1.0) * pow(prop_b + 1.0, prop_a);
    double *lc = g->label_counts->a;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < d; i++) {
      double inv = 1.0 / (g->eigenvals[i] + mu);
      for (int64_t l = 0; l < nl; l++)
        g->W_work[i * nl + l] = g->PQtY[i * nl + l] *
          (1.0 + C / pow(lc[l] + prop_b, prop_a)) * inv;
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < d; i++) {
      double inv = 1.0 / (g->eigenvals[i] + mu);
      for (int64_t l = 0; l < nl; l++)
        g->W_work[i * nl + l] = g->PQtY[i * nl + l] * inv;
    }
  }
  if (!g->W_work_f)
    g->W_work_f = (float *)malloc(dnl * sizeof(float));
  for (uint64_t i = 0; i < dnl; i++)
    g->W_work_f[i] = (float)g->W_work[i];
  if (g->cm_proj) {
    if (!g->intercept)
      g->intercept = (double *)malloc((uint64_t)nl * sizeof(double));
    if (do_prop && g->label_counts) {
      double C = (log((double)g->n_samples) - 1.0) * pow(prop_b + 1.0, prop_a);
      double *lc = g->label_counts->a;
      for (int64_t l = 0; l < nl; l++)
        g->intercept[l] = (1.0 + C / pow(lc[l] + prop_b, prop_a)) * g->y_mean[l];
    } else {
      memcpy(g->intercept, g->y_mean, (uint64_t)nl * sizeof(double));
    }
    cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl,
      -1.0, g->W_work, (int)nl, g->cm_proj, 1, 1.0, g->intercept, 1);
  }
}

static inline int tk_gram_prepare_val_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  tk_fvec_t *val_fvec = tk_fvec_peek(L, 2, "val_codes");
  int64_t val_n = (int64_t)luaL_checkinteger(L, 3);
  int64_t d = g->n_dims;
  free(g->val_F_f);
  free(g->sbuf_f); g->sbuf_f = NULL;
  uint64_t nd = (uint64_t)val_n * (uint64_t)d;
  if (g->col_mean) {
    if (!g->cm_proj)
      g->cm_proj = (double *)malloc((uint64_t)d * sizeof(double));
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)d, (int)d,
      1.0, g->evecs, (int)d, g->col_mean, 1, 0.0, g->cm_proj, 1);
  }
  g->val_F_f = (float *)malloc(nd * sizeof(float));
  if (!g->val_F_f) return luaL_error(L, "prepare: out of memory");
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    (int)val_n, (int)d, (int)d, 1.0f, val_fvec->a, (int)d,
    g->evecs_f, (int)d, 0.0f, g->val_F_f, (int)d);
  g->val_n = val_n;
  return 0;
}

static inline void tk_gram_ensure_sample_heaps (tk_gram_t *g, int64_t val_n, int64_t k) {
  int64_t need = val_n * k;
  if (g->sample_heaps && need <= g->sample_heap_cap)
    return;
  free(g->sample_heaps);
  g->sample_heaps = (tk_rank_t *)malloc((uint64_t)need * sizeof(tk_rank_t));
  g->sample_heap_cap = need;
}

static inline int tk_gram_trial_label_tiled_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->val_F_f) return luaL_error(L, "label: call prepare first");
  int64_t d = g->n_dims, nl = g->n_labels, val_n = g->val_n;
  double lambda_raw = luaL_checknumber(L, 2);
  int64_t k = (int64_t)luaL_checkinteger(L, 3);
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  bool do_prop = lua_isnumber(L, 4);
  double prop_a = do_prop ? lua_tonumber(L, 4) : 0.0;
  double prop_b = do_prop ? (lua_isnumber(L, 5) ? lua_tonumber(L, 5) : 1.5) : 0.0;
  TK_IVEC_BUF(offsets, 6, val_n + 1);
  TK_IVEC_BUF(labels, 7, val_n * k);
  TK_FVEC_BUF(scores_out, 8, val_n * k);
  for (int64_t i = 0; i <= val_n; i++)
    offsets->a[i] = i * k;
  int64_t B = g->tile_labels;
  double mu = lambda_raw * g->mean_eig + g->mean_eig * 1e-7;
  double C = 0.0;
  double *lc = NULL;
  if (do_prop && g->label_counts) {
    C = (log((double)g->n_samples) - 1.0) * pow(prop_b + 1.0, prop_a);
    lc = g->label_counts->a;
  }
  tk_gram_ensure_sample_heaps(g, val_n, k);
  tk_rvec_t *sheaps = (tk_rvec_t *)malloc((uint64_t)val_n * sizeof(tk_rvec_t));
  for (int64_t i = 0; i < val_n; i++) {
    sheaps[i].n = 0;
    sheaps[i].m = (size_t)k;
    sheaps[i].lua_managed = false;
    sheaps[i].a = g->sample_heaps + i * k;
  }
  double *W_tile = (double *)malloc((uint64_t)d * (uint64_t)B * sizeof(double));
  float *W_f_tile = (float *)malloc((uint64_t)d * (uint64_t)B * sizeof(float));
  float *sbuf_tile = (float *)malloc((uint64_t)val_n * (uint64_t)B * sizeof(float));
  double *intercept_tile = (double *)malloc((uint64_t)B * sizeof(double));
  for (int64_t tl_start = 0; tl_start < nl; tl_start += B) {
    int64_t aB = (tl_start + B <= nl) ? B : nl - tl_start;
    for (int64_t i = 0; i < d; i++) {
      double inv = 1.0 / (g->eigenvals[i] + mu);
      for (int64_t l = 0; l < aB; l++) {
        double pqty = (double)g->PQtY_f[i * nl + tl_start + l];
        double prop = (lc) ? (1.0 + C / pow(lc[tl_start + l] + prop_b, prop_a)) : 1.0;
        W_tile[i * aB + l] = pqty * prop * inv;
      }
    }
    for (int64_t i = 0; i < d * aB; i++)
      W_f_tile[i] = (float)W_tile[i];
    if (g->cm_proj) {
      for (int64_t l = 0; l < aB; l++) {
        double prop = (lc) ? (1.0 + C / pow(lc[tl_start + l] + prop_b, prop_a)) : 1.0;
        intercept_tile[l] = prop * g->y_mean[tl_start + l];
      }
      cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)aB,
        -1.0, W_tile, (int)aB, g->cm_proj, 1, 1.0, intercept_tile, 1);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)val_n, (int)aB, (int)d, 1.0f, g->val_F_f, (int)d,
      W_f_tile, (int)aB, 0.0f, sbuf_tile, (int)aB);
    if (g->cm_proj)
      tk_gram_add_intercept_f(sbuf_tile, val_n, aB, intercept_tile);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < val_n; i++) {
      float *row = sbuf_tile + i * aB;
      for (int64_t l = 0; l < aB; l++)
        tk_rvec_hmin(&sheaps[i], (size_t)k, tk_rank(tl_start + l, (double)row[l]));
    }
  }
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < val_n; i++) {
    tk_rvec_desc(&sheaps[i], 0, sheaps[i].n);
    int64_t out_base = i * k;
    for (int64_t j = 0; j < (int64_t)sheaps[i].n; j++) {
      labels->a[out_base + j] = sheaps[i].a[j].i;
      scores_out->a[out_base + j] = (float)sheaps[i].a[j].d;
    }
  }
  free(W_tile); free(W_f_tile); free(sbuf_tile); free(intercept_tile); free(sheaps);
  lua_pushvalue(L, offsets_idx);
  lua_pushvalue(L, labels_idx);
  lua_pushvalue(L, scores_out_idx);
  return 3;
}

static inline int tk_gram_trial_label_lua (lua_State *L) {
  return tk_gram_trial_label_tiled_lua(L);
}

static inline int tk_gram_regress_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->val_F_f) return luaL_error(L, "regress: call prepare first");
  int64_t d = g->n_dims, nl = g->n_labels, val_n = g->val_n;
  double lambda_raw = luaL_checknumber(L, 2);
  bool do_prop = lua_isnumber(L, 3);
  double prop_a = do_prop ? lua_tonumber(L, 3) : 0.0;
  double prop_b = do_prop ? (lua_isnumber(L, 4) ? lua_tonumber(L, 4) : 1.5) : 0.0;
  TK_FVEC_BUF(out, 5, val_n * nl);
  tk_gram_solve_w(g, lambda_raw, do_prop, prop_a, prop_b);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)val_n, (int)nl, (int)d, 1.0f, g->val_F_f, (int)d,
    g->W_work_f, (int)nl, 0.0f, out->a, (int)nl);
  if (g->intercept)
    tk_gram_add_intercept_f(out->a, val_n, nl, g->intercept);
  lua_pushvalue(L, out_idx);
  return 1;
}

static inline int tk_gram_label_accuracy_tiled_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->val_F_f) return luaL_error(L, "label_accuracy: call prepare first");
  int64_t d = g->n_dims, nl = g->n_labels, val_n = g->val_n;
  double lambda_raw = luaL_checknumber(L, 2);
  int64_t topk = (int64_t)luaL_checkinteger(L, 3);
  if (topk > nl) topk = nl;
  if (topk < 1) topk = 1;
  bool do_prop = lua_isnumber(L, 4);
  double prop_a = do_prop ? lua_tonumber(L, 4) : 0.0;
  double prop_b = do_prop ? (lua_isnumber(L, 5) ? lua_tonumber(L, 5) : 1.5) : 0.0;
  tk_ivec_t *exp_off = tk_ivec_peek(L, 6, "exp_off");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, 7, "exp_nbr");
  int mode = 0;
  int64_t fixed_k = 0;
  if (lua_isnumber(L, 8)) {
    mode = 1;
    fixed_k = (int64_t)lua_tointeger(L, 8);
    if (fixed_k > topk) fixed_k = topk;
    if (fixed_k < 1) fixed_k = 1;
  } else if (lua_isstring(L, 8)) {
    mode = 2;
  }
  int64_t B = g->tile_labels;
  double mu = lambda_raw * g->mean_eig + g->mean_eig * 1e-7;
  double C = 0.0;
  double *lc = NULL;
  if (do_prop && g->label_counts) {
    C = (log((double)g->n_samples) - 1.0) * pow(prop_b + 1.0, prop_a);
    lc = g->label_counts->a;
  }
  int64_t heap_k = (mode == 1) ? fixed_k : topk;
  tk_gram_ensure_sample_heaps(g, val_n, heap_k);
  tk_rvec_t *sheaps = (tk_rvec_t *)malloc((uint64_t)val_n * sizeof(tk_rvec_t));
  for (int64_t i = 0; i < val_n; i++) {
    sheaps[i].n = 0;
    sheaps[i].m = (size_t)heap_k;
    sheaps[i].lua_managed = false;
    sheaps[i].a = g->sample_heaps + i * heap_k;
  }
  double *W_tile = (double *)malloc((uint64_t)d * (uint64_t)B * sizeof(double));
  float *W_f_tile = (float *)malloc((uint64_t)d * (uint64_t)B * sizeof(float));
  float *sbuf_tile = (float *)malloc((uint64_t)val_n * (uint64_t)B * sizeof(float));
  double *intercept_tile = (double *)malloc((uint64_t)B * sizeof(double));
  for (int64_t tl_start = 0; tl_start < nl; tl_start += B) {
    int64_t aB = (tl_start + B <= nl) ? B : nl - tl_start;
    for (int64_t i = 0; i < d; i++) {
      double inv = 1.0 / (g->eigenvals[i] + mu);
      for (int64_t l = 0; l < aB; l++) {
        double pqty = (double)g->PQtY_f[i * nl + tl_start + l];
        double prop = (lc) ? (1.0 + C / pow(lc[tl_start + l] + prop_b, prop_a)) : 1.0;
        W_tile[i * aB + l] = pqty * prop * inv;
      }
    }
    for (int64_t i = 0; i < d * aB; i++)
      W_f_tile[i] = (float)W_tile[i];
    if (g->cm_proj) {
      for (int64_t l = 0; l < aB; l++) {
        double prop = (lc) ? (1.0 + C / pow(lc[tl_start + l] + prop_b, prop_a)) : 1.0;
        intercept_tile[l] = prop * g->y_mean[tl_start + l];
      }
      cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)aB,
        -1.0, W_tile, (int)aB, g->cm_proj, 1, 1.0, intercept_tile, 1);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)val_n, (int)aB, (int)d, 1.0f, g->val_F_f, (int)d,
      W_f_tile, (int)aB, 0.0f, sbuf_tile, (int)aB);
    if (g->cm_proj)
      tk_gram_add_intercept_f(sbuf_tile, val_n, aB, intercept_tile);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < val_n; i++) {
      float *row = sbuf_tile + i * aB;
      for (int64_t l = 0; l < aB; l++)
        tk_rvec_hmin(&sheaps[i], (size_t)heap_k, tk_rank(tl_start + l, (double)row[l]));
    }
  }
  free(W_tile); free(W_f_tile); free(sbuf_tile); free(intercept_tile);
  double f1 = 0.0, prec = 0.0, rec = 0.0;
  if (mode == 1) {
    uint64_t total_tp = 0, total_pred = 0, total_exp = 0;
    tk_gram_ensure_heaps(g, topk);
    #pragma omp parallel reduction(+:total_tp,total_pred,total_exp)
    {
      int tid = omp_get_thread_num();
      uint8_t *bm = g->thread_bitmaps[tid];
      #pragma omp for schedule(static)
      for (int64_t i = 0; i < val_n; i++) {
        tk_rvec_t *h = &sheaps[i];
        int64_t es = exp_off->a[i], ee = exp_off->a[i + 1];
        total_exp += (uint64_t)(ee - es);
        total_pred += (uint64_t)h->n;
        if (ee <= es || h->n == 0) continue;
        for (int64_t j = es; j < ee; j++)
          if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 1;
        uint64_t hits = 0;
        for (size_t j = 0; j < h->n; j++)
          if (h->a[j].i >= 0 && h->a[j].i < nl && bm[h->a[j].i]) hits++;
        for (int64_t j = es; j < ee; j++)
          if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 0;
        total_tp += hits;
      }
    }
    prec = total_pred > 0 ? (double)total_tp / (double)total_pred : 0.0;
    rec = total_exp > 0 ? (double)total_tp / (double)total_exp : 0.0;
    f1 = (total_pred + total_exp) > 0
      ? 2.0 * (double)total_tp / ((double)total_pred + (double)total_exp) : 0.0;
  } else if (mode == 0) {
    uint64_t total_tp = 0, total_pred = 0, total_exp = 0;
    tk_gram_ensure_heaps(g, topk);
    #pragma omp parallel reduction(+:total_tp,total_pred,total_exp)
    {
      int tid = omp_get_thread_num();
      uint8_t *bm = g->thread_bitmaps[tid];
      #pragma omp for schedule(static)
      for (int64_t i = 0; i < val_n; i++) {
        tk_rvec_t *h = &sheaps[i];
        int64_t es = exp_off->a[i], ee = exp_off->a[i + 1];
        uint64_t n_exp = (uint64_t)(ee - es);
        total_exp += n_exp;
        if (n_exp == 0 || h->n == 0) continue;
        tk_rvec_desc(h, 0, h->n);
        for (int64_t j = es; j < ee; j++)
          if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 1;
        uint64_t running_hits = 0, best_tp_i = 0;
        int64_t best_k_i = 0;
        double best_f1_i = 0.0;
        for (size_t j = 0; j < h->n; j++) {
          if (h->a[j].i >= 0 && h->a[j].i < nl && bm[h->a[j].i]) running_hits++;
          double f1_j = 2.0 * (double)running_hits / ((double)(j + 1) + (double)n_exp);
          if (f1_j > best_f1_i) {
            best_f1_i = f1_j;
            best_tp_i = running_hits;
            best_k_i = (int64_t)(j + 1);
          }
        }
        for (int64_t j = es; j < ee; j++)
          if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 0;
        total_tp += best_tp_i;
        total_pred += (uint64_t)best_k_i;
      }
    }
    prec = total_pred > 0 ? (double)total_tp / (double)total_pred : 0.0;
    rec = total_exp > 0 ? (double)total_tp / (double)total_exp : 0.0;
    f1 = (total_pred + total_exp) > 0
      ? 2.0 * (double)total_tp / ((double)total_pred + (double)total_exp) : 0.0;
  } else {
    uint64_t total_entries = (uint64_t)val_n * (uint64_t)topk;
    uint64_t total_expected = 0;
    for (int64_t i = 0; i < val_n; i++)
      total_expected += (uint64_t)(exp_off->a[i + 1] - exp_off->a[i]);
    if (total_entries == 0 || total_expected == 0) {
      free(sheaps);
      lua_pushnumber(L, 0.0);
      lua_pushnumber(L, 0.0);
      lua_pushnumber(L, 0.0);
      return 3;
    }
    tk_gram_ensure_gfm(g, (int64_t)total_entries);
    tk_gram_ensure_heaps(g, topk);
    int64_t *topk_labels = g->gfm_labels;
    double *topk_scores = g->gfm_scores;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < val_n; i++) {
      tk_rvec_t *h = &sheaps[i];
      int64_t base = i * topk;
      tk_rvec_desc(h, 0, h->n);
      for (size_t j = 0; j < h->n; j++) {
        topk_labels[base + (int64_t)j] = h->a[j].i;
        topk_scores[base + (int64_t)j] = h->a[j].d;
      }
      for (size_t j = h->n; j < (size_t)topk; j++) {
        topk_labels[base + (int64_t)j] = -1;
        topk_scores[base + (int64_t)j] = -HUGE_VAL;
      }
    }
    uint8_t *bm = g->thread_bitmaps[0];
    tk_rvec_t pool = { .n = 0, .m = (size_t)total_entries, .lua_managed = false,
                       .a = g->gfm_pool };
    for (int64_t i = 0; i < val_n; i++) {
      for (int64_t j = exp_off->a[i]; j < exp_off->a[i + 1]; j++)
        if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 1;
      int64_t base = i * topk;
      for (int64_t j = 0; j < topk; j++) {
        int64_t lbl = topk_labels[base + j];
        if (lbl < 0) continue;
        pool.a[pool.n].i = (lbl < nl && bm[lbl]) ? 1 : 0;
        pool.a[pool.n].d = topk_scores[base + j];
        pool.n++;
      }
      for (int64_t j = exp_off->a[i]; j < exp_off->a[i + 1]; j++)
        if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 0;
    }
    tk_rvec_desc(&pool, 0, pool.n);
    uint64_t tp = 0, best_tp = 0;
    int64_t best_pred = 0;
    double best_f1 = 0.0;
    for (size_t i = 0; i < pool.n; i++) {
      tp += (uint64_t)pool.a[i].i;
      double f1_val = 2.0 * (double)tp / ((double)(i + 1) + (double)total_expected);
      if (f1_val > best_f1) {
        best_f1 = f1_val;
        best_tp = tp;
        best_pred = (int64_t)(i + 1);
      }
    }
    f1 = best_f1;
    prec = best_pred > 0 ? (double)best_tp / (double)best_pred : 0.0;
    rec = total_expected > 0 ? (double)best_tp / (double)total_expected : 0.0;
  }
  free(sheaps);
  lua_pushnumber(L, f1);
  lua_pushnumber(L, prec);
  lua_pushnumber(L, rec);
  return 3;
}

static inline int tk_gram_label_accuracy_lua (lua_State *L) {
  return tk_gram_label_accuracy_tiled_lua(L);
}

static inline int tk_gram_label_ranking_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->val_F_f) return luaL_error(L, "label_ranking: call prepare first");
  int64_t d = g->n_dims, nl = g->n_labels, val_n = g->val_n;
  double lambda_raw = luaL_checknumber(L, 2);
  int64_t topk = (int64_t)luaL_checkinteger(L, 3);
  if (topk > nl) topk = nl;
  if (topk < 1) topk = 1;
  bool do_prop = lua_isnumber(L, 4);
  double prop_a = do_prop ? lua_tonumber(L, 4) : 0.0;
  double prop_b = do_prop ? (lua_isnumber(L, 5) ? lua_tonumber(L, 5) : 1.5) : 0.0;
  tk_ivec_t *exp_off = tk_ivec_peek(L, 6, "exp_off");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, 7, "exp_nbr");
  tk_gram_solve_w(g, lambda_raw, do_prop, prop_a, prop_b);
  uint64_t need = (uint64_t)val_n * (uint64_t)nl;
  if (!g->sbuf_f) {
    g->sbuf_f = (float *)malloc(need * sizeof(float));
    if (!g->sbuf_f) return luaL_error(L, "label_ranking: malloc failed");
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)val_n, (int)nl, (int)d, 1.0f, g->val_F_f, (int)d,
    g->W_work_f, (int)nl, 0.0f, g->sbuf_f, (int)nl);
  if (g->intercept)
    tk_gram_add_intercept_f(g->sbuf_f, val_n, nl, g->intercept);
  tk_gram_ensure_heaps(g, topk);
  double sum_ndcg = 0.0;
  uint64_t n_valid = 0;
  #pragma omp parallel reduction(+:sum_ndcg,n_valid)
  {
    int tid = omp_get_thread_num();
    tk_rvec_t heap = { .n = 0, .m = (size_t)topk, .lua_managed = false,
                       .a = g->thread_heaps[tid] };
    uint8_t *bm = g->thread_bitmaps[tid];
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < val_n; i++) {
      float *row = g->sbuf_f + i * nl;
      int64_t es = exp_off->a[i], ee = exp_off->a[i + 1];
      int64_t n_exp = ee - es;
      if (n_exp == 0) continue;
      heap.n = 0;
      for (int64_t l = 0; l < nl; l++)
        tk_rvec_hmin(&heap, (size_t)topk, tk_rank(l, (double)row[l]));
      tk_rvec_desc(&heap, 0, heap.n);
      for (int64_t j = es; j < ee; j++)
        if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 1;
      double dcg = 0.0;
      for (size_t j = 0; j < heap.n; j++)
        if (heap.a[j].i >= 0 && heap.a[j].i < nl && bm[heap.a[j].i])
          dcg += 1.0 / log2((double)(j + 2));
      for (int64_t j = es; j < ee; j++)
        if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 0;
      int64_t ideal_n = n_exp < (int64_t)heap.n ? n_exp : (int64_t)heap.n;
      double idcg = 0.0;
      for (int64_t j = 0; j < ideal_n; j++)
        idcg += 1.0 / log2((double)(j + 2));
      if (idcg > 0.0)
        sum_ndcg += dcg / idcg;
      n_valid++;
    }
  }
  double ndcg = n_valid > 0 ? sum_ndcg / (double)n_valid : 0.0;
  lua_pushnumber(L, ndcg);
  return 1;
}

static inline int tk_gram_regress_accuracy_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->val_F_f) return luaL_error(L, "regress_accuracy: call prepare first");
  int64_t d = g->n_dims, nl = g->n_labels, val_n = g->val_n;
  double lambda_raw = luaL_checknumber(L, 2);
  bool do_prop = lua_isnumber(L, 3);
  double prop_a = do_prop ? lua_tonumber(L, 3) : 0.0;
  double prop_b = do_prop ? (lua_isnumber(L, 4) ? lua_tonumber(L, 4) : 1.5) : 0.0;
  tk_dvec_t *targets = tk_dvec_peek(L, 5, "targets");
  uint64_t n = (uint64_t)val_n * (uint64_t)nl;
  if (targets->n != n)
    return luaL_error(L, "regress_accuracy: targets length mismatch");
  tk_gram_solve_w(g, lambda_raw, do_prop, prop_a, prop_b);
  if (!g->sbuf_f) {
    g->sbuf_f = (float *)malloc(n * sizeof(float));
    if (!g->sbuf_f) return luaL_error(L, "regress_accuracy: malloc failed");
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)val_n, (int)nl, (int)d, 1.0f, g->val_F_f, (int)d,
    g->W_work_f, (int)nl, 0.0f, g->sbuf_f, (int)nl);
  if (g->intercept)
    tk_gram_add_intercept_f(g->sbuf_f, val_n, nl, g->intercept);
  double total_err = 0.0, sum_exp = 0.0;
  #pragma omp parallel for reduction(+:total_err,sum_exp)
  for (uint64_t i = 0; i < n; i++) {
    total_err += fabs((double)g->sbuf_f[i] - targets->a[i]);
    sum_exp += targets->a[i];
  }
  double mae = n > 0 ? total_err / (double)n : 0.0;
  double mean_exp = n > 0 ? sum_exp / (double)n : 0.0;
  double nmae = mean_exp > 0.0 ? mae / mean_exp : 0.0;
  lua_pushnumber(L, mae);
  lua_pushnumber(L, nmae);
  return 2;
}

__attribute__((unused)) static luaL_Reg tk_gram_mt_fns[] = {
  { "prepare", tk_gram_prepare_val_lua },
  { "label", tk_gram_trial_label_lua },
  { "label_accuracy", tk_gram_label_accuracy_lua },
  { "label_ranking", tk_gram_label_ranking_lua },
  { "regress", tk_gram_regress_lua },
  { "regress_accuracy", tk_gram_regress_accuracy_lua },
  { NULL, NULL }
};

#endif
