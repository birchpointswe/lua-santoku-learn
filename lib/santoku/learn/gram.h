#ifndef TK_GRAM_H
#define TK_GRAM_H

#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/rvec.h>
#include <santoku/learn/buf.h>

#define TK_GRAM_MT "tk_gram_t"

typedef struct {
  tk_dvec_t *evecs;
  tk_dvec_t *eigenvals;
  tk_dvec_t *RXtY;
  tk_dvec_t *label_counts;
  tk_dvec_t *work;
  tk_dvec_t *col_mean;
  tk_dvec_t *y_mean;
  int64_t n_dims;
  int64_t n_labels;
  int64_t n_samples;
  bool destroyed;
  double *H_val;
  double *VCM;
  double *intercept_buf;
  int64_t val_n;
} tk_gram_t;

static inline tk_gram_t *tk_gram_peek (lua_State *L, int i) {
  return (tk_gram_t *)luaL_checkudata(L, i, TK_GRAM_MT);
}

static inline int tk_gram_gc (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  free(g->H_val);
  free(g->VCM);
  free(g->intercept_buf);
  g->evecs = NULL;
  g->eigenvals = NULL;
  g->RXtY = NULL;
  g->label_counts = NULL;
  g->work = NULL;
  g->col_mean = NULL;
  g->y_mean = NULL;
  g->destroyed = true;
  return 0;
}

static inline void tk_gram_add_intercept (
  double *buf, int64_t bs, int64_t nl, double *intercept)
{
  for (int64_t i = 0; i < bs; i++) {
    double *row = buf + i * nl;
    for (int64_t l = 0; l < nl; l++)
      row[l] += intercept[l];
  }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
static inline void tk_gram_topk_block (
  double *sbuf, int64_t bs, int64_t nl, int64_t k, int64_t base,
  tk_ivec_t *offsets, tk_ivec_t *labels, tk_dvec_t *scores_out)
{
  #pragma omp parallel
  {
    tk_rvec_t heap = { .n = 0, .m = (size_t)k, .lua_managed = false,
                       .a = (tk_rank_t *)malloc((uint64_t)k * sizeof(tk_rank_t)) };
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < bs; i++) {
      double *row = sbuf + i * nl;
      int64_t out_base = (base + i) * k;
      heap.n = 0;
      for (int64_t l = 0; l < nl; l++)
        tk_rvec_hmin(&heap, (size_t)k, tk_rank(l, row[l]));
      tk_rvec_desc(&heap, 0, heap.n);
      for (int64_t j = 0; j < (int64_t)heap.n; j++) {
        labels->a[out_base + j] = heap.a[j].i;
        scores_out->a[out_base + j] = heap.a[j].d;
      }
    }
    free(heap.a);
  }
}
#pragma GCC diagnostic pop

static inline int tk_gram_prepare_val_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  int64_t val_n = (int64_t)luaL_checkinteger(L, 3);
  int64_t d = g->n_dims, nl = g->n_labels;
  tk_dvec_t *val_dvec = tk_dvec_peek(L, 2, "val_codes");
  double *val_a = val_dvec->a;
  free(g->H_val); free(g->VCM); free(g->intercept_buf);
  g->H_val = (double *)malloc((uint64_t)val_n * (uint64_t)d * sizeof(double));
  g->VCM = (double *)malloc((uint64_t)d * sizeof(double));
  g->intercept_buf = (double *)malloc((uint64_t)nl * sizeof(double));
  g->val_n = val_n;
  if (!g->H_val || !g->VCM || !g->intercept_buf) {
    free(g->H_val); free(g->VCM); free(g->intercept_buf);
    g->H_val = NULL; g->VCM = NULL; g->intercept_buf = NULL;
    return luaL_error(L, "prepare: malloc failed");
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    (int)val_n, (int)d, (int)d, 1.0, val_a, (int)d,
    g->evecs->a, (int)d, 0.0, g->H_val, (int)d);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)d, (int)d, 1.0,
    g->evecs->a, (int)d, g->col_mean->a, 1, 0.0, g->VCM, 1);
  return 0;
}

static inline int tk_gram_compute_sz (
  lua_State *L, tk_gram_t *g, double lambda_raw,
  bool do_prop, double prop_a, double prop_b)
{
  int64_t d = g->n_dims, nl = g->n_labels;
  double max_eig = d > 0 ? g->eigenvals->a[d - 1] : 1.0;
  double lambda = lambda_raw * max_eig + max_eig * 1e-7;
  double *SZ = g->work->a;
  double *intercept = g->intercept_buf;
  if (do_prop) {
    if (!g->label_counts)
      return luaL_error(L, "propensity requires label prepare");
    double C = (log((double)g->n_samples) - 1.0) * pow(prop_b + 1.0, prop_a);
    for (int64_t l = 0; l < nl; l++)
      intercept[l] = 1.0 + C / pow(g->label_counts->a[l] + prop_b, prop_a);
    for (int64_t i = 0; i < d; i++) {
      double s = 1.0 / (g->eigenvals->a[i] + lambda);
      for (int64_t l = 0; l < nl; l++)
        SZ[i * nl + l] = s * intercept[l] * g->RXtY->a[i * nl + l];
    }
    for (int64_t l = 0; l < nl; l++)
      intercept[l] *= g->y_mean->a[l];
  } else {
    for (int64_t i = 0; i < d; i++) {
      double s = 1.0 / (g->eigenvals->a[i] + lambda);
      for (int64_t l = 0; l < nl; l++)
        SZ[i * nl + l] = s * g->RXtY->a[i * nl + l];
    }
    memcpy(intercept, g->y_mean->a, (uint64_t)nl * sizeof(double));
  }
  cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl, -1.0,
    SZ, (int)nl, g->VCM, 1, 1.0, intercept, 1);
  return 0;
}

static inline void tk_gram_fill_scores (
  tk_gram_t *g, double *out, int64_t n, int64_t base)
{
  int64_t d = g->n_dims, nl = g->n_labels;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)n, (int)nl, (int)d, 1.0, g->H_val + base * d, (int)d,
    g->work->a, (int)nl, 0.0, out, (int)nl);
  tk_gram_add_intercept(out, n, nl, g->intercept_buf);
}

static inline int tk_gram_trial_label_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->H_val) return luaL_error(L, "label: call prepare first");
  int64_t nl = g->n_labels, val_n = g->val_n;
  double lambda_raw = luaL_checknumber(L, 2);
  int64_t k = (int64_t)luaL_checkinteger(L, 3);
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  bool do_prop = lua_isnumber(L, 4);
  double prop_a = do_prop ? lua_tonumber(L, 4) : 0.0;
  double prop_b = do_prop ? (lua_isnumber(L, 5) ? lua_tonumber(L, 5) : 1.5) : 0.0;
  TK_IVEC_BUF(offsets, 6, val_n + 1);
  TK_IVEC_BUF(labels, 7, val_n * k);
  TK_DVEC_BUF(scores_out, 8, val_n * k);
  for (int64_t i = 0; i <= val_n; i++)
    offsets->a[i] = i * k;
  int err = tk_gram_compute_sz(L, g, lambda_raw, do_prop, prop_a, prop_b);
  if (err) return err;
  double *sbuf = (double *)malloc((uint64_t)val_n * (uint64_t)nl * sizeof(double));
  if (!sbuf) return luaL_error(L, "label: malloc failed");
  tk_gram_fill_scores(g, sbuf, val_n, 0);
  tk_gram_topk_block(sbuf, val_n, nl, k, 0, offsets, labels, scores_out);
  free(sbuf);
  lua_pushvalue(L, offsets_idx);
  lua_pushvalue(L, labels_idx);
  lua_pushvalue(L, scores_out_idx);
  return 3;
}

static inline int tk_gram_regress_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->H_val) return luaL_error(L, "regress: call prepare first");
  int64_t nl = g->n_labels, val_n = g->val_n;
  double lambda_raw = luaL_checknumber(L, 2);
  bool do_prop = lua_isnumber(L, 3);
  double prop_a = do_prop ? lua_tonumber(L, 3) : 0.0;
  double prop_b = do_prop ? (lua_isnumber(L, 4) ? lua_tonumber(L, 4) : 1.5) : 0.0;
  TK_DVEC_BUF(out, 5, val_n * nl);
  int err = tk_gram_compute_sz(L, g, lambda_raw, do_prop, prop_a, prop_b);
  if (err) return err;
  tk_gram_fill_scores(g, out->a, val_n, 0);
  lua_pushvalue(L, out_idx);
  return 1;
}

static luaL_Reg tk_gram_mt_fns[] = {
  { "prepare", tk_gram_prepare_val_lua },
  { "label", tk_gram_trial_label_lua },
  { "regress", tk_gram_regress_lua },
  { NULL, NULL }
};

#endif
