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
  double *col_mean;
  double *y_mean;
  double *cm_proj;
  double *intercept;
  bool destroyed;
} tk_gram_t;

__attribute__((unused)) static luaL_Reg tk_gram_mt_fns[];

static inline tk_gram_t *tk_gram_peek (lua_State *L, int i) {
  return (tk_gram_t *)luaL_checkudata(L, i, TK_GRAM_MT);
}

static inline int tk_gram_gc (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  free(g->evecs);
  free(g->evecs_f);
  free(g->eigenvals);
  free(g->PQtY);
  free(g->W_work);
  free(g->W_work_f);
  free(g->sbuf_f);
  free(g->val_F_f);
  free(g->col_mean);
  free(g->y_mean);
  free(g->cm_proj);
  free(g->intercept);
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
  g->label_counts = lc;
  g->W_work = W_work;
  g->W_work_f = NULL;
  g->sbuf_f = NULL;
  g->val_F_f = NULL;
  g->col_mean = col_mean;
  g->y_mean = y_mean_arr;
  g->cm_proj = NULL;
  g->intercept = NULL;
  g->mean_eig = mean_eig;
  g->n_dims = m;
  g->n_labels = nl;
  g->n_samples = nc;
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

static inline void tk_gram_topk_block_f (
  float *sbuf, int64_t bs, int64_t nl, int64_t k, int64_t base,
  tk_ivec_t *offsets, tk_ivec_t *labels, tk_fvec_t *scores_out)
{
  #pragma omp parallel
  {
    tk_rvec_t heap = { .n = 0, .m = (size_t)k, .lua_managed = false,
                       .a = (tk_rank_t *)malloc((uint64_t)k * sizeof(tk_rank_t)) };
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < bs; i++) {
      float *row = sbuf + i * nl;
      int64_t out_base = (base + i) * k;
      heap.n = 0;
      for (int64_t l = 0; l < nl; l++)
        tk_rvec_hmin(&heap, (size_t)k, tk_rank(l, (double)row[l]));
      tk_rvec_desc(&heap, 0, heap.n);
      for (int64_t j = 0; j < (int64_t)heap.n; j++) {
        labels->a[out_base + j] = heap.a[j].i;
        scores_out->a[out_base + j] = (float)heap.a[j].d;
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

static inline int tk_gram_trial_label_lua (lua_State *L) {
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
  tk_gram_solve_w(g, lambda_raw, do_prop, prop_a, prop_b);
  uint64_t need = (uint64_t)val_n * (uint64_t)nl;
  if (!g->sbuf_f) {
    g->sbuf_f = (float *)malloc(need * sizeof(float));
    if (!g->sbuf_f) return luaL_error(L, "label: malloc failed");
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)val_n, (int)nl, (int)d, 1.0f, g->val_F_f, (int)d,
    g->W_work_f, (int)nl, 0.0f, g->sbuf_f, (int)nl);
  if (g->intercept)
    tk_gram_add_intercept_f(g->sbuf_f, val_n, nl, g->intercept);
  tk_gram_topk_block_f(g->sbuf_f, val_n, nl, k, 0, offsets, labels, scores_out);
  lua_pushvalue(L, offsets_idx);
  lua_pushvalue(L, labels_idx);
  lua_pushvalue(L, scores_out_idx);
  return 3;
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

__attribute__((unused)) static luaL_Reg tk_gram_mt_fns[] = {
  { "prepare", tk_gram_prepare_val_lua },
  { "label", tk_gram_trial_label_lua },
  { "regress", tk_gram_regress_lua },
  { NULL, NULL }
};

#endif
