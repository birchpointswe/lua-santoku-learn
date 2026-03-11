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
#include <santoku/rvec.h>
#include <santoku/learn/buf.h>

#define TK_GRAM_MT "tk_gram_t"

typedef struct {
  double *evecs;
  double *eigenvals;
  double *PQtY;
  tk_dvec_t *label_counts;
  double *W_work;
  double *sbuf;
  double *val_F;
  double mean_eig;
  int64_t n_dims;
  int64_t n_labels;
  int64_t n_samples;
  int64_t val_n;
  bool destroyed;
} tk_gram_t;

static inline tk_gram_t *tk_gram_peek (lua_State *L, int i) {
  return (tk_gram_t *)luaL_checkudata(L, i, TK_GRAM_MT);
}

static inline int tk_gram_gc (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  free(g->evecs);
  free(g->eigenvals);
  free(g->PQtY);
  free(g->W_work);
  free(g->sbuf);
  free(g->val_F);
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

static inline void tk_gram_solve_w (
  tk_gram_t *g, double lambda_raw, bool do_prop, double prop_a, double prop_b)
{
  int64_t d = g->n_dims, nl = g->n_labels;
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
}

static inline int tk_gram_prepare_val_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  tk_dvec_t *val_dvec = tk_dvec_peek(L, 2, "val_codes");
  int64_t val_n = (int64_t)luaL_checkinteger(L, 3);
  int64_t d = g->n_dims;
  free(g->val_F);
  free(g->sbuf); g->sbuf = NULL;
  g->val_F = (double *)malloc((uint64_t)val_n * (uint64_t)d * sizeof(double));
  if (!g->val_F) return luaL_error(L, "prepare: out of memory");
  g->val_n = val_n;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    (int)val_n, (int)d, (int)d, 1.0, val_dvec->a, (int)d,
    g->evecs, (int)d, 0.0, g->val_F, (int)d);
  return 0;
}

static inline int tk_gram_trial_label_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->val_F) return luaL_error(L, "label: call prepare first");
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
  TK_DVEC_BUF(scores_out, 8, val_n * k);
  for (int64_t i = 0; i <= val_n; i++)
    offsets->a[i] = i * k;
  tk_gram_solve_w(g, lambda_raw, do_prop, prop_a, prop_b);
  uint64_t need = (uint64_t)val_n * (uint64_t)nl;
  if (!g->sbuf) {
    g->sbuf = (double *)malloc(need * sizeof(double));
    if (!g->sbuf) return luaL_error(L, "label: malloc failed");
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)val_n, (int)nl, (int)d, 1.0, g->val_F, (int)d,
    g->W_work, (int)nl, 0.0, g->sbuf, (int)nl);
  tk_gram_topk_block(g->sbuf, val_n, nl, k, 0, offsets, labels, scores_out);
  lua_pushvalue(L, offsets_idx);
  lua_pushvalue(L, labels_idx);
  lua_pushvalue(L, scores_out_idx);
  return 3;
}

static inline int tk_gram_regress_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->val_F) return luaL_error(L, "regress: call prepare first");
  int64_t d = g->n_dims, nl = g->n_labels, val_n = g->val_n;
  double lambda_raw = luaL_checknumber(L, 2);
  bool do_prop = lua_isnumber(L, 3);
  double prop_a = do_prop ? lua_tonumber(L, 3) : 0.0;
  double prop_b = do_prop ? (lua_isnumber(L, 4) ? lua_tonumber(L, 4) : 1.5) : 0.0;
  TK_DVEC_BUF(out, 5, val_n * nl);
  tk_gram_solve_w(g, lambda_raw, do_prop, prop_a, prop_b);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)val_n, (int)nl, (int)d, 1.0, g->val_F, (int)d,
    g->W_work, (int)nl, 0.0, out->a, (int)nl);
  lua_pushvalue(L, out_idx);
  return 1;
}

__attribute__((unused)) static luaL_Reg tk_gram_mt_fns[] = {
  { "prepare", tk_gram_prepare_val_lua },
  { "label", tk_gram_trial_label_lua },
  { "regress", tk_gram_regress_lua },
  { NULL, NULL }
};

#endif
