#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <lapacke.h>
#include <cblas.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/rvec.h>

#define TK_RIDGE_MT "tk_ridge_t"
#define TK_RIDGE_GRAM_MT "tk_ridge_gram_t"

typedef struct {
  tk_dvec_t *W;
  tk_dvec_t *intercept;
  int64_t n_dims;
  int64_t n_labels;
  bool destroyed;
} tk_ridge_t;

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
  double *sbuf;
  double *intercept_buf;
  int64_t val_n;
} tk_ridge_gram_t;

static inline tk_ridge_t *tk_ridge_peek (lua_State *L, int i) {
  return (tk_ridge_t *)luaL_checkudata(L, i, TK_RIDGE_MT);
}

static inline tk_ridge_gram_t *tk_ridge_gram_peek (lua_State *L, int i) {
  return (tk_ridge_gram_t *)luaL_checkudata(L, i, TK_RIDGE_GRAM_MT);
}

static inline int tk_ridge_gc (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  r->W = NULL;
  r->intercept = NULL;
  r->destroyed = true;
  return 0;
}

static inline int tk_ridge_gram_gc (lua_State *L) {
  tk_ridge_gram_t *g = tk_ridge_gram_peek(L, 1);
  free(g->H_val);
  free(g->VCM);
  free(g->sbuf);
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
static inline void tk_ridge_topk_block (
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

static inline double *tk_ridge_transpose_w (tk_ridge_t *r) {
  int64_t d = r->n_dims, nl = r->n_labels;
  double *wt = (double *)malloc((uint64_t)nl * (uint64_t)d * sizeof(double));
  for (int64_t i = 0; i < d; i++)
    for (int64_t j = 0; j < nl; j++)
      wt[j * d + i] = r->W->a[i * nl + j];
  return wt;
}

static inline void tk_ridge_add_intercept (
  double *buf, int64_t bs, int64_t nl, double *intercept)
{
  for (int64_t i = 0; i < bs; i++) {
    double *row = buf + i * nl;
    for (int64_t l = 0; l < nl; l++)
      row[l] += intercept[l];
  }
}

static inline int tk_ridge_encode_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  int64_t nl = r->n_labels, d = r->n_dims;
  tk_dvec_t *codes = tk_dvec_peek(L, 2, "codes");
  int64_t n = (int64_t)luaL_checkinteger(L, 3);
  int64_t k = (int64_t)luaL_checkinteger(L, 4);
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  int orig_top = lua_gettop(L);
  tk_ivec_t *offsets;
  int off_idx;
  if (orig_top >= 5 && !lua_isnil(L, 5)) {
    offsets = tk_ivec_peek(L, 5, "offsets_buf");
    tk_ivec_ensure(offsets, (uint64_t)(n + 1));
    offsets->n = (uint64_t)(n + 1);
    off_idx = 5;
  } else {
    offsets = tk_ivec_create(L, (uint64_t)(n + 1), NULL, NULL);
    off_idx = lua_gettop(L);
  }
  tk_ivec_t *labels;
  int lab_idx;
  if (orig_top >= 6 && !lua_isnil(L, 6)) {
    labels = tk_ivec_peek(L, 6, "labels_buf");
    tk_ivec_ensure(labels, (uint64_t)(n * k));
    labels->n = (uint64_t)(n * k);
    lab_idx = 6;
  } else {
    labels = tk_ivec_create(L, (uint64_t)(n * k), NULL, NULL);
    lab_idx = lua_gettop(L);
  }
  tk_dvec_t *scores_out;
  int sco_idx;
  if (orig_top >= 7 && !lua_isnil(L, 7)) {
    scores_out = tk_dvec_peek(L, 7, "scores_buf");
    tk_dvec_ensure(scores_out, (uint64_t)(n * k));
    scores_out->n = (uint64_t)(n * k);
    sco_idx = 7;
  } else {
    scores_out = tk_dvec_create(L, (uint64_t)(n * k), NULL, NULL);
    sco_idx = lua_gettop(L);
  }
  for (int64_t i = 0; i <= n; i++)
    offsets->a[i] = i * k;
  bool sparse = orig_top >= 8 && !lua_isnil(L, 8);
  if (sparse) {
    tk_ivec_t *csr_off = tk_ivec_peek(L, 8, "csr_off");
    tk_ivec_t *csr_nbr = tk_ivec_peek(L, 9, "csr_nbr");
    double *wt = tk_ridge_transpose_w(r);
    double *intercept = r->intercept ? r->intercept->a : NULL;
    #pragma omp parallel
    {
      tk_rvec_t heap = { .n = 0, .m = (size_t)k, .lua_managed = false,
                         .a = (tk_rank_t *)malloc((uint64_t)k * sizeof(tk_rank_t)) };
      #pragma omp for schedule(dynamic, 64)
      for (int64_t i = 0; i < n; i++) {
        double *row = codes->a + i * d;
        int64_t lo = csr_off->a[i], hi = csr_off->a[i + 1];
        int64_t out_base = i * k;
        heap.n = 0;
        for (int64_t j = lo; j < hi; j++) {
          int64_t label = csr_nbr->a[j];
          double s = cblas_ddot((int)d, row, 1, wt + label * d, 1);
          if (intercept) s += intercept[label];
          tk_rvec_hmin(&heap, (size_t)k, tk_rank(label, s));
        }
        tk_rvec_desc(&heap, 0, heap.n);
        for (int64_t j = 0; j < (int64_t)heap.n; j++) {
          labels->a[out_base + j] = heap.a[j].i;
          scores_out->a[out_base + j] = heap.a[j].d;
        }
        for (int64_t j = (int64_t)heap.n; j < k; j++) {
          labels->a[out_base + j] = 0;
          scores_out->a[out_base + j] = -1e30;
        }
      }
      free(heap.a);
    }
    free(wt);
  } else {
    int64_t block = 256;
    while (block > 1 && (uint64_t)block * (uint64_t)nl * sizeof(double) > 64ULL * 1024 * 1024)
      block /= 2;
    double *sbuf = (double *)malloc((uint64_t)block * (uint64_t)nl * sizeof(double));
    for (int64_t base = 0; base < n; base += block) {
      int64_t bs = (base + block <= n) ? block : n - base;
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)bs, (int)nl, (int)d, 1.0, codes->a + base * d, (int)d,
        r->W->a, (int)nl, 0.0, sbuf, (int)nl);
      if (r->intercept)
        tk_ridge_add_intercept(sbuf, bs, nl, r->intercept->a);
      tk_ridge_topk_block(sbuf, bs, nl, k, base, offsets, labels, scores_out);
    }
    free(sbuf);
  }
  lua_pushvalue(L, off_idx);
  lua_pushvalue(L, lab_idx);
  lua_pushvalue(L, sco_idx);
  return 3;
}

static inline int tk_ridge_persist_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting filepath or true");
  tk_lua_fwrite(L, "TKri", 1, 4, fh);
  uint8_t version = 2;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &r->n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &r->n_labels, sizeof(int64_t), 1, fh);
  tk_dvec_persist(L, r->W, fh);
  uint8_t has_intercept = r->intercept ? 1 : 0;
  tk_lua_fwrite(L, &has_intercept, sizeof(uint8_t), 1, fh);
  if (r->intercept)
    tk_dvec_persist(L, r->intercept, fh);
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

static inline int tk_ridge_transform_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  tk_dvec_t *codes = tk_dvec_peek(L, 2, "codes");
  int64_t n = (int64_t)luaL_checkinteger(L, 3);
  int64_t d = r->n_dims, nl = r->n_labels;
  if (lua_gettop(L) >= 5 && !lua_isnil(L, 4)) {
    tk_ivec_t *csr_off = tk_ivec_peek(L, 4, "csr_off");
    tk_ivec_t *csr_nbr = tk_ivec_peek(L, 5, "csr_nbr");
    int64_t total = csr_off->a[n];
    tk_dvec_t *out = tk_dvec_create(L, (uint64_t)total, NULL, NULL);
    double *wt = tk_ridge_transpose_w(r);
    double *intercept = r->intercept ? r->intercept->a : NULL;
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t i = 0; i < n; i++) {
      int64_t lo = csr_off->a[i], hi = csr_off->a[i + 1];
      double *row = codes->a + i * d;
      for (int64_t j = lo; j < hi; j++) {
        int64_t label = csr_nbr->a[j];
        double s = cblas_ddot((int)d, row, 1, wt + label * d, 1);
        if (intercept) s += intercept[label];
        out->a[j] = s;
      }
    }
    free(wt);
    return 1;
  }
  tk_dvec_t *out;
  if (lua_gettop(L) >= 4 && !lua_isnil(L, 4)) {
    out = tk_dvec_peek(L, 4, "out_buf");
    tk_dvec_ensure(out, (uint64_t)(n * nl));
    out->n = (uint64_t)(n * nl);
    lua_pushvalue(L, 4);
  } else {
    out = tk_dvec_create(L, (uint64_t)(n * nl), NULL, NULL);
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)n, (int)nl, (int)d, 1.0, codes->a, (int)d,
    r->W->a, (int)nl, 0.0, out->a, (int)nl);
  if (r->intercept)
    tk_ridge_add_intercept(out->a, n, nl, r->intercept->a);
  return 1;
}

static luaL_Reg tk_ridge_mt_fns[] = {
  { "label", tk_ridge_encode_lua },
  { "persist", tk_ridge_persist_lua },
  { "regress", tk_ridge_transform_lua },
  { NULL, NULL }
};

static inline int tk_ridge_gram_prepare_val_lua (lua_State *L) {
  tk_ridge_gram_t *g = tk_ridge_gram_peek(L, 1);
  tk_dvec_t *val_codes = tk_dvec_peek(L, 2, "val_codes");
  int64_t val_n = (int64_t)luaL_checkinteger(L, 3);
  int64_t d = g->n_dims, nl = g->n_labels;
  free(g->H_val); free(g->VCM); free(g->sbuf); free(g->intercept_buf);
  g->H_val = (double *)malloc((uint64_t)val_n * (uint64_t)d * sizeof(double));
  g->VCM = (double *)malloc((uint64_t)d * sizeof(double));
  g->sbuf = (double *)malloc((uint64_t)val_n * (uint64_t)nl * sizeof(double));
  g->intercept_buf = (double *)malloc((uint64_t)nl * sizeof(double));
  g->val_n = val_n;
  if (!g->H_val || !g->VCM || !g->sbuf || !g->intercept_buf) {
    free(g->H_val); free(g->VCM); free(g->sbuf); free(g->intercept_buf);
    g->H_val = NULL; g->VCM = NULL; g->sbuf = NULL; g->intercept_buf = NULL;
    return luaL_error(L, "prepare: malloc failed");
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    (int)val_n, (int)d, (int)d, 1.0, val_codes->a, (int)d,
    g->evecs->a, (int)d, 0.0, g->H_val, (int)d);
  cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)d, (int)d, 1.0,
    g->evecs->a, (int)d, g->col_mean->a, 1, 0.0, g->VCM, 1);
  return 0;
}

static inline int tk_ridge_gram_trial_label_lua (lua_State *L) {
  tk_ridge_gram_t *g = tk_ridge_gram_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  if (!g->H_val) return luaL_error(L, "label: call prepare first");
  int64_t d = g->n_dims, nl = g->n_labels, val_n = g->val_n;
  lua_getfield(L, 2, "lambda");
  double lambda_raw = (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.0);
  lua_pop(L, 1);
  double max_eig = d > 0 ? g->eigenvals->a[d - 1] : 1.0;
  double lambda = lambda_raw * max_eig + max_eig * 1e-7;
  lua_getfield(L, 2, "propensity_a");
  bool do_prop = lua_isnumber(L, -1);
  double prop_a = do_prop ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 2, "propensity_b");
  double prop_b = do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
  lua_pop(L, 1);
  int64_t k = (int64_t)tk_lua_fcheckunsigned(L, 2, "label", "k");
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  lua_getfield(L, 2, "off_buf");
  tk_ivec_t *offsets;
  int off_idx;
  if (!lua_isnil(L, -1)) {
    offsets = tk_ivec_peek(L, -1, "off_buf");
    tk_ivec_ensure(offsets, (uint64_t)(val_n + 1));
    offsets->n = (uint64_t)(val_n + 1);
    off_idx = lua_gettop(L);
  } else {
    lua_pop(L, 1);
    offsets = tk_ivec_create(L, (uint64_t)(val_n + 1), NULL, NULL);
    off_idx = lua_gettop(L);
  }
  lua_getfield(L, 2, "nbr_buf");
  tk_ivec_t *labels;
  int lab_idx;
  if (!lua_isnil(L, -1)) {
    labels = tk_ivec_peek(L, -1, "nbr_buf");
    tk_ivec_ensure(labels, (uint64_t)(val_n * k));
    labels->n = (uint64_t)(val_n * k);
    lab_idx = lua_gettop(L);
  } else {
    lua_pop(L, 1);
    labels = tk_ivec_create(L, (uint64_t)(val_n * k), NULL, NULL);
    lab_idx = lua_gettop(L);
  }
  lua_getfield(L, 2, "sco_buf");
  tk_dvec_t *scores_out;
  int sco_idx;
  if (!lua_isnil(L, -1)) {
    scores_out = tk_dvec_peek(L, -1, "sco_buf");
    tk_dvec_ensure(scores_out, (uint64_t)(val_n * k));
    scores_out->n = (uint64_t)(val_n * k);
    sco_idx = lua_gettop(L);
  } else {
    lua_pop(L, 1);
    scores_out = tk_dvec_create(L, (uint64_t)(val_n * k), NULL, NULL);
    sco_idx = lua_gettop(L);
  }
  for (int64_t i = 0; i <= val_n; i++)
    offsets->a[i] = i * k;
  double *SZ = g->work->a;
  double *intercept = g->intercept_buf;
  if (do_prop) {
    if (!g->label_counts)
      return luaL_error(L, "label: propensity requires label prepare");
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
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)val_n, (int)nl, (int)d, 1.0, g->H_val, (int)d,
    SZ, (int)nl, 0.0, g->sbuf, (int)nl);
  tk_ridge_add_intercept(g->sbuf, val_n, nl, intercept);
  tk_ridge_topk_block(g->sbuf, val_n, nl, k, 0, offsets, labels, scores_out);
  lua_pushvalue(L, off_idx);
  lua_pushvalue(L, lab_idx);
  lua_pushvalue(L, sco_idx);
  return 3;
}

static inline int tk_ridge_gram_regress_lua (lua_State *L) {
  tk_ridge_gram_t *g = tk_ridge_gram_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  if (!g->H_val) return luaL_error(L, "regress: call prepare first");
  int64_t d = g->n_dims, nl = g->n_labels, val_n = g->val_n;
  lua_getfield(L, 2, "lambda");
  double lambda_raw = (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.0);
  lua_pop(L, 1);
  double max_eig = d > 0 ? g->eigenvals->a[d - 1] : 1.0;
  double lambda = lambda_raw * max_eig + max_eig * 1e-7;
  lua_getfield(L, 2, "propensity_a");
  bool do_prop = lua_isnumber(L, -1);
  double prop_a = do_prop ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 2, "propensity_b");
  double prop_b = do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 2, "out_buf");
  tk_dvec_t *out;
  if (!lua_isnil(L, -1)) {
    out = tk_dvec_peek(L, -1, "out_buf");
    tk_dvec_ensure(out, (uint64_t)(val_n * nl));
    out->n = (uint64_t)(val_n * nl);
  } else {
    lua_pop(L, 1);
    out = tk_dvec_create(L, (uint64_t)(val_n * nl), NULL, NULL);
  }
  int out_idx = lua_gettop(L);
  double *SZ = g->work->a;
  double *intercept = g->intercept_buf;
  if (do_prop) {
    if (!g->label_counts)
      return luaL_error(L, "regress: propensity requires label prepare");
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
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)val_n, (int)nl, (int)d, 1.0, g->H_val, (int)d,
    SZ, (int)nl, 0.0, out->a, (int)nl);
  tk_ridge_add_intercept(out->a, val_n, nl, intercept);
  lua_pushvalue(L, out_idx);
  return 1;
}

static luaL_Reg tk_ridge_gram_mt_fns[] = {
  { "prepare", tk_ridge_gram_prepare_val_lua },
  { "label", tk_ridge_gram_trial_label_lua },
  { "regress", tk_ridge_gram_regress_lua },
  { NULL, NULL }
};

static inline int tk_ridge_precompute_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 1, "prepare", "n_samples");
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  lua_pop(L, 1);
  int64_t d = (int64_t)tk_lua_fcheckunsigned(L, 1, "prepare", "n_dims");
  lua_getfield(L, 1, "targets");
  bool dense = !lua_isnil(L, -1);
  tk_dvec_t *targets = dense ? tk_dvec_peek(L, -1, "targets") : NULL;
  lua_pop(L, 1);
  int64_t nl;
  tk_ivec_t *lab_off = NULL, *lab_nbr = NULL;
  if (dense) {
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "prepare", "n_targets");
  } else {
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "prepare", "n_labels");
    lua_getfield(L, 1, "label_offsets");
    lab_off = tk_ivec_peek(L, -1, "label_offsets");
    lua_getfield(L, 1, "label_neighbors");
    lab_nbr = tk_ivec_peek(L, -1, "label_neighbors");
    lua_pop(L, 2);
  }

  double inv_n = 1.0 / (double)n;

  tk_dvec_t *cm_dvec = tk_dvec_create(L, (uint64_t)d, NULL, NULL);
  int cm_idx = lua_gettop(L);
  memset(cm_dvec->a, 0, (uint64_t)d * sizeof(double));
  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < d; j++)
      cm_dvec->a[j] += codes->a[i * d + j];
  cblas_dscal((int)d, inv_n, cm_dvec->a, 1);

  tk_dvec_t *ev_dvec = tk_dvec_create(L, (uint64_t)(d * d), NULL, NULL);
  int ev_idx = lua_gettop(L);
  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
    (int)d, (int)n, inv_n, codes->a, (int)d, 0.0, ev_dvec->a, (int)d);
  cblas_dsyr(CblasRowMajor, CblasUpper, (int)d, -1.0,
    cm_dvec->a, 1, ev_dvec->a, (int)d);
  for (int64_t i = 0; i < d; i++)
    for (int64_t j = i + 1; j < d; j++)
      ev_dvec->a[j * d + i] = ev_dvec->a[i * d + j];

  tk_dvec_t *eig_dvec = tk_dvec_create(L, (uint64_t)d, NULL, NULL);
  int eig_idx = lua_gettop(L);
  int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (int)d, ev_dvec->a, (int)d, eig_dvec->a);
  if (info != 0)
    return luaL_error(L, "prepare: eigendecomposition failed (info=%d)", info);

  tk_dvec_t *cnt_dvec = NULL;
  int cnt_idx = 0;
  if (!dense) {
    cnt_dvec = tk_dvec_create(L, (uint64_t)nl, NULL, NULL);
    cnt_idx = lua_gettop(L);
    memset(cnt_dvec->a, 0, (uint64_t)nl * sizeof(double));
    for (int64_t i = 0; i < n; i++) {
      int64_t lo = lab_off->a[i], hi = lab_off->a[i + 1];
      for (int64_t j = lo; j < hi; j++)
        cnt_dvec->a[lab_nbr->a[j]] += 1.0;
    }
  }

  tk_dvec_t *ym_dvec = tk_dvec_create(L, (uint64_t)nl, NULL, NULL);
  int ym_idx = lua_gettop(L);
  if (dense) {
    memset(ym_dvec->a, 0, (uint64_t)nl * sizeof(double));
    for (int64_t i = 0; i < n; i++)
      for (int64_t l = 0; l < nl; l++)
        ym_dvec->a[l] += targets->a[i * nl + l];
    cblas_dscal((int)nl, inv_n, ym_dvec->a, 1);
  } else {
    for (int64_t l = 0; l < nl; l++)
      ym_dvec->a[l] = cnt_dvec->a[l] * inv_n;
  }

  uint64_t dnl = (uint64_t)d * (uint64_t)nl;

  tk_dvec_t *work_dvec = tk_dvec_create(L, dnl, NULL, NULL);
  int work_idx = lua_gettop(L);
  memset(work_dvec->a, 0, dnl * sizeof(double));
  double *xty = work_dvec->a;
  if (dense) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)d, (int)nl, (int)n, inv_n, codes->a, (int)d,
      targets->a, (int)nl, 0.0, xty, (int)nl);
  } else {
    #pragma omp parallel for schedule(static)
    for (int64_t dd = 0; dd < d; dd++) {
      double *xty_row = xty + dd * nl;
      for (int64_t i = 0; i < n; i++) {
        int64_t lo = lab_off->a[i], hi = lab_off->a[i + 1];
        for (int64_t j = lo; j < hi; j++)
          xty_row[lab_nbr->a[j]] += codes->a[i * d + dd];
      }
    }
    cblas_dscal((int)(d * nl), inv_n, xty, 1);
  }
  cblas_dger(CblasRowMajor, (int)d, (int)nl, -1.0,
    cm_dvec->a, 1, ym_dvec->a, 1, xty, (int)nl);

  tk_dvec_t *rxty_dvec = tk_dvec_create(L, dnl, NULL, NULL);
  int rxty_idx = lua_gettop(L);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)d, (int)nl, (int)d, 1.0, ev_dvec->a, (int)d,
    xty, (int)nl, 0.0, rxty_dvec->a, (int)nl);

  tk_ridge_gram_t *g = tk_lua_newuserdata(L, tk_ridge_gram_t,
    TK_RIDGE_GRAM_MT, tk_ridge_gram_mt_fns, tk_ridge_gram_gc);
  int Gi = lua_gettop(L);
  g->evecs = ev_dvec;
  g->eigenvals = eig_dvec;
  g->RXtY = rxty_dvec;
  g->label_counts = cnt_dvec;
  g->work = work_dvec;
  g->col_mean = cm_dvec;
  g->y_mean = ym_dvec;
  g->n_dims = d;
  g->n_labels = nl;
  g->n_samples = n;
  g->destroyed = false;
  g->H_val = NULL;
  g->VCM = NULL;
  g->sbuf = NULL;
  g->intercept_buf = NULL;
  g->val_n = 0;

  lua_newtable(L);
  lua_pushvalue(L, ev_idx);
  lua_setfield(L, -2, "evecs");
  lua_pushvalue(L, eig_idx);
  lua_setfield(L, -2, "eigenvals");
  lua_pushvalue(L, rxty_idx);
  lua_setfield(L, -2, "RXtY");
  lua_pushvalue(L, work_idx);
  lua_setfield(L, -2, "work");
  lua_pushvalue(L, cm_idx);
  lua_setfield(L, -2, "col_mean");
  lua_pushvalue(L, ym_idx);
  lua_setfield(L, -2, "y_mean");
  if (cnt_idx > 0) {
    lua_pushvalue(L, cnt_idx);
    lua_setfield(L, -2, "label_counts");
  }
  lua_setfenv(L, Gi);
  lua_pushvalue(L, Gi);
  return 1;
}


static inline int tk_ridge_create_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "gram");
  if (!lua_isnil(L, -1)) {
    tk_ridge_gram_t *gram = tk_ridge_gram_peek(L, -1);
    int gram_lua_idx = lua_gettop(L);
    int64_t d = gram->n_dims, nl = gram->n_labels;
    uint64_t dnl = (uint64_t)d * (uint64_t)nl;
    lua_getfield(L, 1, "lambda");
    double lambda_raw = (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.0);
    lua_pop(L, 1);
    double max_eig = d > 0 ? gram->eigenvals->a[d - 1] : 1.0;
    double lambda = lambda_raw * max_eig + max_eig * 1e-7;
    lua_getfield(L, 1, "propensity_a");
    bool do_prop = lua_isnumber(L, -1);
    double prop_a = do_prop ? lua_tonumber(L, -1) : 0.0;
    lua_pop(L, 1);
    lua_getfield(L, 1, "propensity_b");
    double prop_b = do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
    lua_pop(L, 1);
    lua_getfield(L, 1, "w_buf");
    tk_dvec_t *w_buf = lua_isnil(L, -1) ? NULL : tk_dvec_peek(L, -1, "w_buf");
    int w_buf_lua_idx = w_buf ? lua_gettop(L) : 0;
    if (!w_buf) lua_pop(L, 1);
    lua_getfield(L, 1, "intercept_buf");
    tk_dvec_t *ib = lua_isnil(L, -1) ? NULL : tk_dvec_peek(L, -1, "intercept_buf");
    int ib_lua_idx = ib ? lua_gettop(L) : 0;
    if (!ib) lua_pop(L, 1);
    tk_dvec_t *W_dvec;
    int W_idx;
    if (w_buf) {
      tk_dvec_ensure(w_buf, (uint64_t)(d * nl));
      w_buf->n = (uint64_t)(d * nl);
      W_dvec = w_buf;
      W_idx = w_buf_lua_idx;
    } else {
      W_dvec = tk_dvec_create(L, (uint64_t)(d * nl), NULL, NULL);
      W_idx = lua_gettop(L);
    }
    double *Z = gram->work->a;
    memcpy(Z, gram->RXtY->a, dnl * sizeof(double));
    if (do_prop) {
      if (!gram->label_counts)
        return luaL_error(L, "ridge create: propensity requires label prepare");
      int64_t ns = gram->n_samples;
      double C = (log((double)ns) - 1.0) * pow(prop_b + 1.0, prop_a);
      for (int64_t l = 0; l < nl; l++) {
        double w = 1.0 + C / pow(gram->label_counts->a[l] + prop_b, prop_a);
        for (int64_t i = 0; i < d; i++)
          Z[i * nl + l] *= w;
      }
    }
    for (int64_t i = 0; i < d; i++) {
      double s = 1.0 / (gram->eigenvals->a[i] + lambda);
      for (int64_t j = 0; j < nl; j++)
        Z[i * nl + j] *= s;
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)d, (int)nl, (int)d, 1.0, gram->evecs->a, (int)d,
      Z, (int)nl, 0.0, W_dvec->a, (int)nl);

    tk_dvec_t *b_dvec;
    int b_idx;
    if (ib) {
      tk_dvec_ensure(ib, (uint64_t)nl);
      ib->n = (uint64_t)nl;
      b_dvec = ib;
      b_idx = ib_lua_idx;
    } else {
      b_dvec = tk_dvec_create(L, (uint64_t)nl, NULL, NULL);
      b_idx = lua_gettop(L);
    }
    if (do_prop) {
      int64_t ns = gram->n_samples;
      double C = (log((double)ns) - 1.0) * pow(prop_b + 1.0, prop_a);
      for (int64_t l = 0; l < nl; l++) {
        double w = 1.0 + C / pow(gram->label_counts->a[l] + prop_b, prop_a);
        b_dvec->a[l] = w * gram->y_mean->a[l];
      }
    } else {
      memcpy(b_dvec->a, gram->y_mean->a, (uint64_t)nl * sizeof(double));
    }
    cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl, -1.0,
      W_dvec->a, (int)nl, gram->col_mean->a, 1, 1.0, b_dvec->a, 1);

    tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
      TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
    int Ei = lua_gettop(L);
    r->W = W_dvec;
    r->intercept = b_dvec;
    r->n_dims = d;
    r->n_labels = nl;
    r->destroyed = false;
    lua_newtable(L);
    lua_pushvalue(L, W_idx);
    lua_setfield(L, -2, "W");
    lua_pushvalue(L, b_idx);
    lua_setfield(L, -2, "intercept");
    lua_pushvalue(L, gram_lua_idx);
    lua_setfield(L, -2, "gram");
    lua_setfenv(L, Ei);
    lua_pushvalue(L, Ei);
    lua_pushvalue(L, W_idx);
    lua_pushvalue(L, b_idx);
    return 3;
  }
  return luaL_error(L, "ridge create: gram required");
}

static inline int tk_ridge_load_lua (lua_State *L) {
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKri", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid ridge file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 1 && version != 2) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported ridge version %d", (int)version);
  }
  int64_t n_dims, n_labels;
  tk_lua_fread(L, &n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_labels, sizeof(int64_t), 1, fh);
  tk_dvec_t *W = tk_dvec_load(L, fh);
  int W_idx = lua_gettop(L);
  tk_dvec_t *intercept = NULL;
  int b_idx = 0;
  if (version >= 2) {
    uint8_t has_intercept;
    tk_lua_fread(L, &has_intercept, sizeof(uint8_t), 1, fh);
    if (has_intercept) {
      intercept = tk_dvec_load(L, fh);
      b_idx = lua_gettop(L);
    }
  }
  tk_lua_fclose(L, fh);
  tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
    TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
  int Ei = lua_gettop(L);
  r->W = W;
  r->intercept = intercept;
  r->n_dims = n_dims;
  r->n_labels = n_labels;
  r->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, W_idx);
  lua_setfield(L, -2, "W");
  if (b_idx > 0) {
    lua_pushvalue(L, b_idx);
    lua_setfield(L, -2, "intercept");
  }
  lua_setfenv(L, Ei);
  lua_pushvalue(L, Ei);
  return 1;
}

static inline int tk_ridge_solve_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 1, "solve", "n_samples");
  int64_t d = (int64_t)tk_lua_fcheckunsigned(L, 1, "solve", "n_dims");
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  lua_pop(L, 1);
  lua_getfield(L, 1, "targets");
  bool dense = !lua_isnil(L, -1);
  tk_dvec_t *targets = dense ? tk_dvec_peek(L, -1, "targets") : NULL;
  lua_pop(L, 1);
  int64_t nl;
  tk_ivec_t *lab_off = NULL, *lab_nbr = NULL;
  if (dense) {
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "solve", "n_targets");
  } else {
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "solve", "n_labels");
    lua_getfield(L, 1, "label_offsets");
    lab_off = tk_ivec_peek(L, -1, "label_offsets");
    lua_getfield(L, 1, "label_neighbors");
    lab_nbr = tk_ivec_peek(L, -1, "label_neighbors");
    lua_pop(L, 2);
  }
  lua_getfield(L, 1, "lambda");
  double lambda_raw = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "propensity_a");
  bool do_prop = lua_isnumber(L, -1);
  double prop_a = do_prop ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "propensity_b");
  double prop_b = do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "w_buf");
  tk_dvec_t *w_buf = lua_isnil(L, -1) ? NULL : tk_dvec_peek(L, -1, "w_buf");
  int w_buf_lua_idx = w_buf ? lua_gettop(L) : 0;
  if (!w_buf) lua_pop(L, 1);
  lua_getfield(L, 1, "intercept_buf");
  tk_dvec_t *ib = lua_isnil(L, -1) ? NULL : tk_dvec_peek(L, -1, "intercept_buf");
  int ib_lua_idx = ib ? lua_gettop(L) : 0;
  if (!ib) lua_pop(L, 1);

  double inv_n = 1.0 / (double)n;
  uint64_t dd = (uint64_t)d;
  uint64_t dnl = dd * (uint64_t)nl;

  tk_dvec_t *W_dvec;
  int W_idx;
  if (w_buf) {
    tk_dvec_ensure(w_buf, dnl);
    w_buf->n = dnl;
    W_dvec = w_buf;
    W_idx = w_buf_lua_idx;
  } else {
    W_dvec = tk_dvec_create(L, dnl, NULL, NULL);
    W_idx = lua_gettop(L);
  }
  memset(W_dvec->a, 0, dnl * sizeof(double));
  double *xty = W_dvec->a;

  tk_dvec_t *b_dvec;
  int b_idx;
  if (ib) {
    tk_dvec_ensure(ib, (uint64_t)nl);
    ib->n = (uint64_t)nl;
    b_dvec = ib;
    b_idx = ib_lua_idx;
  } else {
    b_dvec = tk_dvec_create(L, (uint64_t)nl, NULL, NULL);
    b_idx = lua_gettop(L);
  }
  double *prop_y_mean = b_dvec->a;

  double *col_mean = (double *)calloc(dd, sizeof(double));
  double *gram = (double *)calloc(dd * dd, sizeof(double));
  double *y_mean = (double *)calloc((uint64_t)nl, sizeof(double));
  double *label_counts = (!dense) ? (double *)calloc((uint64_t)nl, sizeof(double)) : NULL;
  if (!col_mean || !gram || !y_mean || (!dense && !label_counts)) {
    free(col_mean); free(gram); free(y_mean); free(label_counts);
    return luaL_error(L, "solve: malloc failed");
  }

  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < d; j++)
      col_mean[j] += codes->a[i * d + j];
  cblas_dscal((int)d, inv_n, col_mean, 1);

  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
    (int)d, (int)n, inv_n, codes->a, (int)d, 0.0, gram, (int)d);
  cblas_dsyr(CblasRowMajor, CblasUpper, (int)d, -1.0,
    col_mean, 1, gram, (int)d);

  if (!dense) {
    for (int64_t i = 0; i < n; i++) {
      int64_t lo = lab_off->a[i], hi = lab_off->a[i + 1];
      for (int64_t j = lo; j < hi; j++)
        label_counts[lab_nbr->a[j]] += 1.0;
    }
  }

  if (dense) {
    for (int64_t i = 0; i < n; i++)
      for (int64_t l = 0; l < nl; l++)
        y_mean[l] += targets->a[i * nl + l];
    cblas_dscal((int)nl, inv_n, y_mean, 1);
  } else {
    for (int64_t l = 0; l < nl; l++)
      y_mean[l] = label_counts[l] * inv_n;
  }

  if (dense) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)d, (int)nl, (int)n, inv_n, codes->a, (int)d,
      targets->a, (int)nl, 0.0, xty, (int)nl);
  } else {
    #pragma omp parallel for schedule(static)
    for (int64_t dd2 = 0; dd2 < d; dd2++) {
      double *xty_row = xty + dd2 * nl;
      for (int64_t i = 0; i < n; i++) {
        int64_t lo = lab_off->a[i], hi = lab_off->a[i + 1];
        for (int64_t j = lo; j < hi; j++)
          xty_row[lab_nbr->a[j]] += codes->a[i * d + dd2];
      }
    }
    cblas_dscal((int)(d * nl), inv_n, xty, 1);
  }
  cblas_dger(CblasRowMajor, (int)d, (int)nl, -1.0,
    col_mean, 1, y_mean, 1, xty, (int)nl);

  if (do_prop && !dense) {
    double C = (log((double)n) - 1.0) * pow(prop_b + 1.0, prop_a);
    for (int64_t l = 0; l < nl; l++) {
      double pw = 1.0 + C / pow(label_counts[l] + prop_b, prop_a);
      prop_y_mean[l] = pw * y_mean[l];
      for (int64_t i = 0; i < d; i++)
        xty[i * nl + l] *= pw;
    }
  } else {
    memcpy(prop_y_mean, y_mean, (uint64_t)nl * sizeof(double));
  }

  for (int64_t i = 0; i < d; i++)
    for (int64_t j = i + 1; j < d; j++)
      gram[j * d + i] = gram[i * d + j];

  double max_diag = 0.0;
  for (int64_t i = 0; i < d; i++)
    if (gram[i * d + i] > max_diag) max_diag = gram[i * d + i];
  if (max_diag <= 0.0) max_diag = 1.0;
  double lambda = lambda_raw * max_diag + max_diag * 1e-7;
  for (int64_t i = 0; i < d; i++)
    gram[i * d + i] += lambda;

  int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', (int)d, gram, (int)d);
  if (info != 0) {
    free(col_mean); free(gram); free(y_mean); free(label_counts);
    return luaL_error(L, "solve: Cholesky failed (info=%d)", info);
  }
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
    (int)d, (int)nl, 1.0, gram, (int)d, xty, (int)nl);
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
    (int)d, (int)nl, 1.0, gram, (int)d, xty, (int)nl);

  cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl, -1.0,
    W_dvec->a, (int)nl, col_mean, 1, 1.0, b_dvec->a, 1);

  free(col_mean); free(gram); free(y_mean); free(label_counts);

  tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
    TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
  int Ei = lua_gettop(L);
  r->W = W_dvec;
  r->intercept = b_dvec;
  r->n_dims = d;
  r->n_labels = nl;
  r->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, W_idx);
  lua_setfield(L, -2, "W");
  lua_pushvalue(L, b_idx);
  lua_setfield(L, -2, "intercept");
  lua_setfenv(L, Ei);
  lua_pushvalue(L, Ei);
  lua_pushvalue(L, W_idx);
  lua_pushvalue(L, b_idx);
  return 3;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
static inline void tk_ridge_feed_heap(
  tk_rank_t *heaps, int64_t *hcounts, int64_t orig, int64_t k,
  const double *row, int64_t l0, int64_t bw)
{
  tk_rvec_t h = { .a = heaps + orig * k, .m = (size_t)k,
                   .n = (size_t)hcounts[orig] };
  for (int64_t l = 0; l < bw; l++)
    tk_rvec_hmin(&h, (size_t)k, tk_rank(l0 + l, row[l]));
  hcounts[orig] = (int64_t)h.n;
}
#pragma GCC diagnostic pop

static inline void tk_ridge_gather_from_row(
  double *sco, const tk_ivec_t *off, const tk_ivec_t *nbr,
  int64_t orig, const double *row, int64_t l0, int64_t bw)
{
  for (int64_t j = off->a[orig]; j < off->a[orig + 1]; j++) {
    int64_t lbl = nbr->a[j];
    if (lbl >= l0 && lbl < l0 + bw)
      sco[j] = row[lbl - l0];
  }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
static void tk_ridge_predict_chunk(
  const double *codes, int64_t ns, int64_t d,
  const double *W, int64_t bw, const double *intercept,
  int64_t l0, int64_t sample_block,
  tk_rank_t *heaps, int64_t *hcounts, int64_t k,
  double *gather_sco, const tk_ivec_t *gather_off, const tk_ivec_t *gather_nbr,
  double *transform_buf,
  const int64_t *sample_ids, double *sbuf)
{
  for (int64_t s = 0; s < ns; s += sample_block) {
    int64_t cn = (s + sample_block > ns) ? (ns - s) : sample_block;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)cn, (int)bw, (int)d, 1.0, codes + s * d, (int)d,
      W, (int)bw, 0.0, sbuf, (int)bw);
    for (int64_t vi = 0; vi < cn; vi++) {
      double *row = sbuf + vi * bw;
      for (int64_t l = 0; l < bw; l++) row[l] += intercept[l];
      int64_t orig = sample_ids ? sample_ids[s + vi] : (s + vi);
      if (k > 0)
        tk_ridge_feed_heap(heaps, hcounts, orig, k, row, l0, bw);
      if (gather_sco)
        tk_ridge_gather_from_row(gather_sco, gather_off, gather_nbr, orig, row, l0, bw);
      if (transform_buf)
        memcpy(transform_buf + orig * bw, row, (uint64_t)bw * sizeof(double));
    }
  }
}
#pragma GCC diagnostic pop

static void tk_ridge_build_xty_chunk_csc(
  const double *codes, const int64_t *csc_off, const int64_t *csc_rows,
  int64_t d, int64_t l0, int64_t bw, double *out)
{
  memset(out, 0, (uint64_t)d * (uint64_t)bw * sizeof(double));
  #pragma omp parallel for schedule(dynamic, 16)
  for (int64_t l = l0; l < l0 + bw; l++) {
    int64_t lcol = l - l0;
    for (int64_t p = csc_off[l]; p < csc_off[l + 1]; p++) {
      int64_t s = csc_rows[p];
      for (int64_t dd2 = 0; dd2 < d; dd2++)
        out[dd2 * bw + lcol] += codes[s * d + dd2];
    }
  }
}

static void tk_ridge_build_xty_chunk_fold(
  const double *val_codes, const tk_ivec_t *lab_off, const tk_ivec_t *lab_nbr,
  const int64_t *perm, int64_t val_s, int64_t val_n,
  int64_t d, int64_t l0, int64_t bw, double *out)
{
  memset(out, 0, (uint64_t)d * (uint64_t)bw * sizeof(double));
  for (int64_t vi = 0; vi < val_n; vi++) {
    int64_t idx = perm[val_s + vi];
    for (int64_t j = lab_off->a[idx]; j < lab_off->a[idx + 1]; j++) {
      int64_t lbl = lab_nbr->a[j];
      if (lbl >= l0 && lbl < l0 + bw) {
        int64_t lcol = lbl - l0;
        for (int64_t dd2 = 0; dd2 < d; dd2++)
          out[dd2 * bw + lcol] += val_codes[vi * d + dd2];
      }
    }
  }
}

static void tk_ridge_solve_chunk(
  const double *L_chol, int64_t d,
  const double *full_xty, const double *fold_xty,
  double scale, const double *col_mean, const double *y_mean_slice,
  const double *label_counts_slice, bool do_prop,
  double prop_a, double prop_b, double C,
  int64_t bw, double *W_out, double *intercept_out)
{
  if (fold_xty) {
    for (int64_t i = 0; i < d * bw; i++)
      W_out[i] = (full_xty[i] - fold_xty[i]) * scale;
  } else {
    for (int64_t i = 0; i < d * bw; i++)
      W_out[i] = full_xty[i] * scale;
  }
  cblas_dger(CblasRowMajor, (int)d, (int)bw, -1.0,
    col_mean, 1, y_mean_slice, 1, W_out, (int)bw);
  if (do_prop) {
    for (int64_t l = 0; l < bw; l++) {
      double pw = 1.0 + C / pow(label_counts_slice[l] + prop_b, prop_a);
      intercept_out[l] = pw * y_mean_slice[l];
      for (int64_t dd2 = 0; dd2 < d; dd2++)
        W_out[dd2 * bw + l] *= pw;
    }
  } else {
    memcpy(intercept_out, y_mean_slice, (uint64_t)bw * sizeof(double));
  }
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
    (int)d, (int)bw, 1.0, L_chol, (int)d, W_out, (int)bw);
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
    (int)d, (int)bw, 1.0, L_chol, (int)d, W_out, (int)bw);
  cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)bw, -1.0,
    W_out, (int)bw, col_mean, 1, 1.0, intercept_out, 1);
}

static inline int tk_ridge_solve_oof_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 1, "solve_oof", "n_samples");
  int64_t d = (int64_t)tk_lua_fcheckunsigned(L, 1, "solve_oof", "n_dims");
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  lua_pop(L, 1);
  lua_getfield(L, 1, "targets");
  bool dense = !lua_isnil(L, -1);
  tk_dvec_t *targets = dense ? tk_dvec_peek(L, -1, "targets") : NULL;
  lua_pop(L, 1);
  int64_t nl;
  tk_ivec_t *lab_off = NULL, *lab_nbr = NULL;
  if (dense) {
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "solve_oof", "n_targets");
  } else {
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "solve_oof", "n_labels");
    lua_getfield(L, 1, "label_offsets");
    lab_off = tk_ivec_peek(L, -1, "label_offsets");
    lua_getfield(L, 1, "label_neighbors");
    lab_nbr = tk_ivec_peek(L, -1, "label_neighbors");
    lua_pop(L, 2);
  }
  lua_getfield(L, 1, "lambda");
  double lambda_raw = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "propensity_a");
  bool do_prop = !dense && lua_isnumber(L, -1);
  double prop_a = do_prop ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  lua_getfield(L, 1, "propensity_b");
  double prop_b = do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
  lua_pop(L, 1);
  int64_t k = 0;
  bool have_k = false;
  if (!dense) {
    lua_getfield(L, 1, "k");
    have_k = lua_isnumber(L, -1);
    if (have_k) {
      k = (int64_t)lua_tointeger(L, -1);
      if (k > nl) k = nl;
      if (k < 1) k = 1;
    }
    lua_pop(L, 1);
  }
  lua_getfield(L, 1, "n_folds");
  int64_t nf = lua_isnumber(L, -1) ? (int64_t)lua_tointeger(L, -1) : 5;
  lua_pop(L, 1);
  lua_getfield(L, 1, "transform");
  bool do_transform = !dense && lua_toboolean(L, -1);
  lua_pop(L, 1);
  tk_ivec_t *gather_off = NULL, *gather_nbr = NULL;
  bool do_gather = false;
  if (!dense) {
    lua_getfield(L, 1, "gather_offsets");
    if (!lua_isnil(L, -1)) {
      gather_off = tk_ivec_peek(L, -1, "gather_offsets");
      lua_getfield(L, 1, "gather_neighbors");
      gather_nbr = tk_ivec_peek(L, -1, "gather_neighbors");
      lua_pop(L, 2);
      do_gather = true;
    } else {
      lua_pop(L, 1);
    }
  }
  int64_t label_batch = 0;
  lua_getfield(L, 1, "label_batch");
  if (lua_isnumber(L, -1)) label_batch = (int64_t)lua_tointeger(L, -1);
  lua_pop(L, 1);
  bool do_batch = !dense && label_batch > 0 && nl > label_batch;
  if (do_gather && do_transform)
    return luaL_error(L, "oof: cannot use both transform and gather");
  if (do_batch && do_transform)
    return luaL_error(L, "oof: label_batch does not support transform");
  if (!dense && !have_k && !do_transform && !do_gather)
    return luaL_error(L, "oof: label mode requires k, transform, or gather");

  uint64_t dd = (uint64_t)d;
  uint64_t dnl = dd * (uint64_t)nl;

  int64_t *perm = (int64_t *)malloc((uint64_t)n * sizeof(int64_t));
  for (int64_t i = 0; i < n; i++) perm[i] = i;
  for (int64_t i = n - 1; i > 0; i--) {
    int64_t j = (int64_t)(tk_fast_random() % (uint32_t)(i + 1));
    int64_t tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
  }

  double *full_col_sum = (double *)calloc(dd, sizeof(double));
  double *full_XtX = (double *)calloc(dd * dd, sizeof(double));
  double *full_XtY = do_batch ? NULL : (double *)calloc(dnl, sizeof(double));
  double *full_y_sum = (double *)calloc((uint64_t)nl, sizeof(double));
  double *full_label_counts = (!dense) ? (double *)calloc((uint64_t)nl, sizeof(double)) : NULL;
  if (!perm || !full_col_sum || !full_XtX || (!do_batch && !full_XtY) ||
      !full_y_sum || (!dense && !full_label_counts)) {
    free(perm); free(full_col_sum); free(full_XtX); free(full_XtY); free(full_y_sum); free(full_label_counts);
    return luaL_error(L, "oof: malloc failed");
  }

  for (int64_t i = 0; i < n; i++)
    for (int64_t j = 0; j < d; j++)
      full_col_sum[j] += codes->a[i * d + j];

  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
    (int)d, (int)n, 1.0, codes->a, (int)d, 0.0, full_XtX, (int)d);

  if (dense) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)d, (int)nl, (int)n, 1.0, codes->a, (int)d,
      targets->a, (int)nl, 0.0, full_XtY, (int)nl);
    for (int64_t i = 0; i < n; i++)
      for (int64_t l = 0; l < nl; l++)
        full_y_sum[l] += targets->a[i * nl + l];
  } else {
    if (!do_batch) {
      #pragma omp parallel for schedule(static)
      for (int64_t dd2 = 0; dd2 < d; dd2++) {
        double *xty_row = full_XtY + dd2 * nl;
        for (int64_t i = 0; i < n; i++) {
          double xi_dd = codes->a[i * d + dd2];
          int64_t lo = lab_off->a[i], hi = lab_off->a[i + 1];
          for (int64_t j = lo; j < hi; j++)
            xty_row[lab_nbr->a[j]] += xi_dd;
        }
      }
    }
    for (int64_t i = 0; i < n; i++) {
      int64_t lo = lab_off->a[i], hi = lab_off->a[i + 1];
      for (int64_t j = lo; j < hi; j++)
        full_label_counts[lab_nbr->a[j]] += 1.0;
    }
    for (int64_t l = 0; l < nl; l++)
      full_y_sum[l] = full_label_counts[l];
  }

  tk_ivec_t *oof_off = NULL, *oof_nbr = NULL;
  tk_dvec_t *oof_sco = NULL, *oof_dense = NULL, *oof_transform = NULL, *gather_sco_dv = NULL;
  int oof_off_idx = 0, oof_nbr_idx = 0, oof_sco_idx = 0, oof_dense_idx = 0,
      oof_tr_idx = 0, gather_sco_idx = 0;
  if (!dense) {
    if (have_k) {
      oof_off = tk_ivec_create(L, (uint64_t)(n + 1), NULL, NULL);
      oof_off_idx = lua_gettop(L);
      for (int64_t i = 0; i <= n; i++) oof_off->a[i] = i * k;
      oof_nbr = tk_ivec_create(L, (uint64_t)(n * k), NULL, NULL);
      oof_nbr_idx = lua_gettop(L);
      memset(oof_nbr->a, 0, (uint64_t)(n * k) * sizeof(int64_t));
      oof_sco = tk_dvec_create(L, (uint64_t)(n * k), NULL, NULL);
      oof_sco_idx = lua_gettop(L);
      memset(oof_sco->a, 0, (uint64_t)(n * k) * sizeof(double));
    }
    if (do_transform) {
      oof_transform = tk_dvec_create(L, (uint64_t)((uint64_t)n * (uint64_t)nl), NULL, NULL);
      oof_tr_idx = lua_gettop(L);
      memset(oof_transform->a, 0, (uint64_t)n * (uint64_t)nl * sizeof(double));
    }
    if (do_gather) {
      int64_t gather_nnz = gather_off->a[n];
      gather_sco_dv = tk_dvec_create(L, (uint64_t)gather_nnz, NULL, NULL);
      gather_sco_idx = lua_gettop(L);
      memset(gather_sco_dv->a, 0, (uint64_t)gather_nnz * sizeof(double));
    }
  } else {
    oof_dense = tk_dvec_create(L, (uint64_t)((uint64_t)n * (uint64_t)nl), NULL, NULL);
    oof_dense_idx = lua_gettop(L);
    memset(oof_dense->a, 0, (uint64_t)n * (uint64_t)nl * sizeof(double));
  }

  tk_rank_t *oof_heaps = NULL;
  int64_t *oof_hcounts = NULL;
  if (have_k && !dense) {
    oof_heaps = (tk_rank_t *)malloc((uint64_t)n * (uint64_t)k * sizeof(tk_rank_t));
    oof_hcounts = (int64_t *)calloc((uint64_t)n, sizeof(int64_t));
    if (!oof_heaps || !oof_hcounts) {
      free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
      free(full_y_sum); free(full_label_counts);
      free(oof_heaps); free(oof_hcounts);
      return luaL_error(L, "oof: malloc failed");
    }
  }

  if (nf < 2) {
    double inv_n = 1.0 / (double)n;
    for (int64_t j = 0; j < d; j++) full_col_sum[j] *= inv_n;
    for (int64_t p = 0; p < d; p++)
      for (int64_t q = p; q < d; q++)
        full_XtX[p * d + q] *= inv_n;
    cblas_dsyr(CblasRowMajor, CblasUpper, (int)d, -1.0, full_col_sum, 1, full_XtX, (int)d);
    for (int64_t l = 0; l < nl; l++) full_y_sum[l] *= inv_n;
    for (int64_t i = 0; i < d; i++)
      for (int64_t j = i + 1; j < d; j++)
        full_XtX[j * d + i] = full_XtX[i * d + j];
    double max_diag = 0.0;
    for (int64_t i = 0; i < d; i++)
      if (full_XtX[i * d + i] > max_diag) max_diag = full_XtX[i * d + i];
    if (max_diag <= 0.0) max_diag = 1.0;
    double lambda = lambda_raw * max_diag + max_diag * 1e-7;
    for (int64_t i = 0; i < d; i++)
      full_XtX[i * d + i] += lambda;
    int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', (int)d, full_XtX, (int)d);
    if (info != 0) {
      free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
      free(full_y_sum); free(full_label_counts);
      free(oof_heaps); free(oof_hcounts);
      return luaL_error(L, "oof: Cholesky failed (info=%d)", info);
    }
    if (do_batch) {
      int64_t lb = label_batch;
      int64_t sample_block = 256;
      double C = do_prop ? (log((double)n) - 1.0) * pow(prop_b + 1.0, prop_a) : 0.0;
      int64_t nnz = lab_off->a[n];
      int64_t *csc_off = (int64_t *)calloc((uint64_t)(nl + 1), sizeof(int64_t));
      int64_t *csc_rows = (int64_t *)malloc((uint64_t)nnz * sizeof(int64_t));
      uint64_t ulb = (uint64_t)lb;
      double *full_xty_chunk = (double *)malloc(dd * ulb * sizeof(double));
      double *W_chunk = (double *)malloc(dd * ulb * sizeof(double));
      double *intercept_chunk = (double *)malloc(ulb * sizeof(double));
      double *sbuf_chunk = (double *)malloc((uint64_t)sample_block * ulb * sizeof(double));
      if (!csc_off || !csc_rows || !full_xty_chunk || !W_chunk || !intercept_chunk || !sbuf_chunk) {
        free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
        free(full_y_sum); free(full_label_counts);
        free(oof_heaps); free(oof_hcounts);
        free(csc_off); free(csc_rows);
        free(full_xty_chunk); free(W_chunk); free(intercept_chunk); free(sbuf_chunk);
        return luaL_error(L, "oof: malloc failed");
      }
      for (int64_t j = 0; j < nnz; j++)
        csc_off[lab_nbr->a[j] + 1]++;
      for (int64_t l = 0; l < nl; l++)
        csc_off[l + 1] += csc_off[l];
      { int64_t *csc_pos = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
        for (int64_t i = 0; i < n; i++)
          for (int64_t j = lab_off->a[i]; j < lab_off->a[i + 1]; j++) {
            int64_t l = lab_nbr->a[j];
            csc_rows[csc_off[l] + csc_pos[l]++] = i;
          }
        free(csc_pos); }
      for (int64_t l0 = 0; l0 < nl; l0 += lb) {
        int64_t bw = (l0 + lb > nl) ? (nl - l0) : lb;
        tk_ridge_build_xty_chunk_csc(codes->a, csc_off, csc_rows, d, l0, bw, full_xty_chunk);
        tk_ridge_solve_chunk(full_XtX, d, full_xty_chunk, NULL, inv_n,
          full_col_sum, full_y_sum + l0,
          full_label_counts + l0, do_prop, prop_a, prop_b, C,
          bw, W_chunk, intercept_chunk);
        tk_ridge_predict_chunk(codes->a, n, d, W_chunk, bw, intercept_chunk,
          l0, sample_block, oof_heaps, oof_hcounts, k,
          do_gather ? gather_sco_dv->a : NULL, gather_off, gather_nbr,
          NULL, NULL, sbuf_chunk);
      }
      free(csc_off); free(csc_rows);
      free(full_xty_chunk); free(W_chunk); free(intercept_chunk); free(sbuf_chunk);
    } else if (dense) {
      for (int64_t p = 0; p < d; p++)
        for (int64_t l = 0; l < nl; l++)
          full_XtY[p * nl + l] *= inv_n;
      cblas_dger(CblasRowMajor, (int)d, (int)nl, -1.0,
        full_col_sum, 1, full_y_sum, 1, full_XtY, (int)nl);
      double *intercept = (double *)malloc((uint64_t)nl * sizeof(double));
      if (!intercept) {
        free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
        free(full_y_sum); free(full_label_counts);
        free(oof_heaps); free(oof_hcounts);
        return luaL_error(L, "oof: malloc failed");
      }
      memcpy(intercept, full_y_sum, (uint64_t)nl * sizeof(double));
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
        (int)d, (int)nl, 1.0, full_XtX, (int)d, full_XtY, (int)nl);
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        (int)d, (int)nl, 1.0, full_XtX, (int)d, full_XtY, (int)nl);
      cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl, -1.0,
        full_XtY, (int)nl, full_col_sum, 1, 1.0, intercept, 1);
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)n, (int)nl, (int)d, 1.0, codes->a, (int)d,
        full_XtY, (int)nl, 0.0, oof_dense->a, (int)nl);
      tk_ridge_add_intercept(oof_dense->a, n, nl, intercept);
      free(intercept);
    } else {
      for (int64_t p = 0; p < d; p++)
        for (int64_t l = 0; l < nl; l++)
          full_XtY[p * nl + l] *= inv_n;
      cblas_dger(CblasRowMajor, (int)d, (int)nl, -1.0,
        full_col_sum, 1, full_y_sum, 1, full_XtY, (int)nl);
      double *intercept = (double *)malloc((uint64_t)nl * sizeof(double));
      if (!intercept) {
        free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
        free(full_y_sum); free(full_label_counts);
        free(oof_heaps); free(oof_hcounts);
        return luaL_error(L, "oof: malloc failed");
      }
      if (do_prop) {
        double C = (log((double)n) - 1.0) * pow(prop_b + 1.0, prop_a);
        for (int64_t l = 0; l < nl; l++) {
          double pw = 1.0 + C / pow(full_label_counts[l] + prop_b, prop_a);
          intercept[l] = pw * full_y_sum[l];
          for (int64_t i = 0; i < d; i++)
            full_XtY[i * nl + l] *= pw;
        }
      } else {
        memcpy(intercept, full_y_sum, (uint64_t)nl * sizeof(double));
      }
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
        (int)d, (int)nl, 1.0, full_XtX, (int)d, full_XtY, (int)nl);
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        (int)d, (int)nl, 1.0, full_XtX, (int)d, full_XtY, (int)nl);
      cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl, -1.0,
        full_XtY, (int)nl, full_col_sum, 1, 1.0, intercept, 1);
      int64_t sample_block = 256;
      double *sbuf2 = (double *)malloc((uint64_t)sample_block * (uint64_t)nl * sizeof(double));
      if (!sbuf2) {
        free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
        free(full_y_sum); free(full_label_counts); free(intercept);
        free(oof_heaps); free(oof_hcounts);
        return luaL_error(L, "oof: malloc failed");
      }
      tk_ridge_predict_chunk(codes->a, n, d, full_XtY, nl, intercept,
        0, sample_block, oof_heaps, oof_hcounts, k,
        do_gather ? gather_sco_dv->a : NULL, gather_off, gather_nbr,
        do_transform ? oof_transform->a : NULL, NULL, sbuf2);
      free(sbuf2); free(intercept);
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
    if (oof_heaps) {
      for (int64_t i = 0; i < n; i++) {
        int64_t dst = i * k;
        int64_t cnt = oof_hcounts[i];
        tk_rvec_t h = { .a = oof_heaps + dst, .m = (size_t)k, .n = (size_t)cnt };
        tk_rvec_desc(&h, 0, h.n);
        for (int64_t j = 0; j < cnt; j++) {
          oof_nbr->a[dst + j] = h.a[j].i;
          oof_sco->a[dst + j] = h.a[j].d;
        }
      }
    }
#pragma GCC diagnostic pop
    free(oof_heaps); free(oof_hcounts);
    free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
    free(full_y_sum); free(full_label_counts);
    goto push_results;
  }

  {
    int64_t fold_size = n / nf;
    int64_t max_val = n - (nf - 1) * fold_size;
    int64_t sample_block = 256;
    double *fold_col_sum = (double *)calloc(dd, sizeof(double));
    double *fold_XtX = (double *)calloc(dd * dd, sizeof(double));
    double *fold_y_sum = (double *)calloc((uint64_t)nl, sizeof(double));
    double *fold_label_counts = (!dense) ? (double *)calloc((uint64_t)nl, sizeof(double)) : NULL;
    double *train_gram = (double *)malloc(dd * dd * sizeof(double));
    double *train_col_mean = (double *)malloc(dd * sizeof(double));
    double *train_y_mean = (double *)malloc((uint64_t)nl * sizeof(double));
    double *val_codes_buf = (double *)malloc((uint64_t)max_val * dd * sizeof(double));

    double *fold_XtY = NULL, *prop_y_mean = NULL, *sbuf = NULL;
    int64_t *csc_off = NULL, *csc_rows = NULL;
    double *full_xty_chunk = NULL, *fold_xty_chunk = NULL;
    double *W_chunk = NULL, *intercept_chunk = NULL, *sbuf_chunk = NULL;

    if (do_batch) {
      int64_t nnz = lab_off->a[n];
      uint64_t ulb = (uint64_t)label_batch;
      csc_off = (int64_t *)calloc((uint64_t)(nl + 1), sizeof(int64_t));
      csc_rows = (int64_t *)malloc((uint64_t)nnz * sizeof(int64_t));
      full_xty_chunk = (double *)malloc(dd * ulb * sizeof(double));
      fold_xty_chunk = (double *)malloc(dd * ulb * sizeof(double));
      W_chunk = (double *)malloc(dd * ulb * sizeof(double));
      intercept_chunk = (double *)malloc(ulb * sizeof(double));
      sbuf_chunk = (double *)malloc((uint64_t)sample_block * ulb * sizeof(double));
    } else {
      fold_XtY = (double *)calloc(dnl, sizeof(double));
      prop_y_mean = (double *)malloc((uint64_t)nl * sizeof(double));
      int64_t sbuf_rows = dense ? max_val : sample_block;
      sbuf = (double *)malloc((uint64_t)sbuf_rows * (uint64_t)nl * sizeof(double));
    }

    if (!fold_col_sum || !fold_XtX || !fold_y_sum || !train_gram ||
        !train_col_mean || !train_y_mean || !val_codes_buf ||
        (!dense && !fold_label_counts) ||
        (do_batch && (!csc_off || !csc_rows || !full_xty_chunk || !fold_xty_chunk ||
                      !W_chunk || !intercept_chunk || !sbuf_chunk)) ||
        (!do_batch && (!fold_XtY || !prop_y_mean || !sbuf))) {
      free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
      free(full_y_sum); free(full_label_counts);
      free(oof_heaps); free(oof_hcounts);
      free(fold_col_sum); free(fold_XtX); free(fold_y_sum); free(fold_label_counts);
      free(train_gram); free(train_col_mean); free(train_y_mean); free(val_codes_buf);
      free(fold_XtY); free(prop_y_mean); free(sbuf);
      free(csc_off); free(csc_rows); free(full_xty_chunk); free(fold_xty_chunk);
      free(W_chunk); free(intercept_chunk); free(sbuf_chunk);
      return luaL_error(L, "oof: malloc failed");
    }

    if (do_batch) {
      int64_t nnz = lab_off->a[n];
      for (int64_t j = 0; j < nnz; j++)
        csc_off[lab_nbr->a[j] + 1]++;
      for (int64_t l = 0; l < nl; l++)
        csc_off[l + 1] += csc_off[l];
      { int64_t *csc_pos = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
        for (int64_t i = 0; i < n; i++)
          for (int64_t j = lab_off->a[i]; j < lab_off->a[i + 1]; j++) {
            int64_t l = lab_nbr->a[j];
            csc_rows[csc_off[l] + csc_pos[l]++] = i;
          }
        free(csc_pos); }
    }

    for (int64_t f = 0; f < nf; f++) {
      int64_t val_s = f * fold_size;
      int64_t val_e = (f == nf - 1) ? n : (val_s + fold_size);
      int64_t val_n = val_e - val_s;
      int64_t tr_n = n - val_n;
      double tr_inv = 1.0 / (double)tr_n;

      memset(fold_col_sum, 0, dd * sizeof(double));
      memset(fold_XtX, 0, dd * dd * sizeof(double));
      memset(fold_y_sum, 0, (uint64_t)nl * sizeof(double));
      if (!dense) memset(fold_label_counts, 0, (uint64_t)nl * sizeof(double));

      for (int64_t vi = 0; vi < val_n; vi++)
        memcpy(val_codes_buf + vi * d, codes->a + perm[val_s + vi] * d, dd * sizeof(double));
      for (int64_t vi = 0; vi < val_n; vi++) {
        const double *xi = val_codes_buf + vi * d;
        for (int64_t j = 0; j < d; j++)
          fold_col_sum[j] += xi[j];
      }
      cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
        (int)d, (int)val_n, 1.0, val_codes_buf, (int)d, 0.0, fold_XtX, (int)d);

      if (dense) {
        for (int64_t vi = 0; vi < val_n; vi++) {
          int64_t idx = perm[val_s + vi];
          const double *yi = targets->a + idx * nl;
          memcpy(sbuf + vi * nl, yi, (uint64_t)nl * sizeof(double));
          for (int64_t l = 0; l < nl; l++)
            fold_y_sum[l] += yi[l];
        }
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          (int)d, (int)nl, (int)val_n, 1.0, val_codes_buf, (int)d,
          sbuf, (int)nl, 0.0, fold_XtY, (int)nl);
      } else {
        for (int64_t vi = 0; vi < val_n; vi++) {
          int64_t idx = perm[val_s + vi];
          int64_t lo = lab_off->a[idx], hi = lab_off->a[idx + 1];
          for (int64_t j = lo; j < hi; j++)
            fold_label_counts[lab_nbr->a[j]] += 1.0;
        }
        for (int64_t l = 0; l < nl; l++)
          fold_y_sum[l] = fold_label_counts[l];
        if (!do_batch) {
          memset(fold_XtY, 0, dnl * sizeof(double));
          #pragma omp parallel for schedule(static)
          for (int64_t dd2 = 0; dd2 < d; dd2++) {
            double *xty_row = fold_XtY + dd2 * nl;
            for (int64_t vi = 0; vi < val_n; vi++) {
              int64_t idx = perm[val_s + vi];
              double xval = val_codes_buf[vi * d + dd2];
              int64_t lo = lab_off->a[idx], hi = lab_off->a[idx + 1];
              for (int64_t j = lo; j < hi; j++)
                xty_row[lab_nbr->a[j]] += xval;
            }
          }
        }
      }

      for (int64_t j = 0; j < d; j++)
        train_col_mean[j] = (full_col_sum[j] - fold_col_sum[j]) * tr_inv;
      for (int64_t p = 0; p < d; p++)
        for (int64_t q = p; q < d; q++)
          train_gram[p * d + q] = (full_XtX[p * d + q] - fold_XtX[p * d + q]) * tr_inv;
      cblas_dsyr(CblasRowMajor, CblasUpper, (int)d, -1.0,
        train_col_mean, 1, train_gram, (int)d);
      for (int64_t l = 0; l < nl; l++)
        train_y_mean[l] = (full_y_sum[l] - fold_y_sum[l]) * tr_inv;

      for (int64_t i = 0; i < d; i++)
        for (int64_t j = i + 1; j < d; j++)
          train_gram[j * d + i] = train_gram[i * d + j];
      double max_diag = 0.0;
      for (int64_t i = 0; i < d; i++)
        if (train_gram[i * d + i] > max_diag) max_diag = train_gram[i * d + i];
      if (max_diag <= 0.0) max_diag = 1.0;
      double lambda = lambda_raw * max_diag + max_diag * 1e-7;
      for (int64_t i = 0; i < d; i++)
        train_gram[i * d + i] += lambda;
      int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', (int)d, train_gram, (int)d);
      if (info != 0) {
        free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
        free(full_y_sum); free(full_label_counts);
        free(oof_heaps); free(oof_hcounts);
        free(fold_col_sum); free(fold_XtX); free(fold_y_sum); free(fold_label_counts);
        free(train_gram); free(train_col_mean); free(train_y_mean); free(val_codes_buf);
        free(fold_XtY); free(prop_y_mean); free(sbuf);
        free(csc_off); free(csc_rows); free(full_xty_chunk); free(fold_xty_chunk);
        free(W_chunk); free(intercept_chunk); free(sbuf_chunk);
        return luaL_error(L, "oof: Cholesky failed fold %d (info=%d)", (int)f, info);
      }

      if (do_batch) {
        int64_t lb = label_batch;
        for (int64_t l = 0; l < nl; l++)
          fold_label_counts[l] = full_label_counts[l] - fold_label_counts[l];
        double C = do_prop ? (log((double)tr_n) - 1.0) * pow(prop_b + 1.0, prop_a) : 0.0;
        for (int64_t l0 = 0; l0 < nl; l0 += lb) {
          int64_t bw = (l0 + lb > nl) ? (nl - l0) : lb;
          tk_ridge_build_xty_chunk_csc(codes->a, csc_off, csc_rows, d, l0, bw, full_xty_chunk);
          tk_ridge_build_xty_chunk_fold(val_codes_buf, lab_off, lab_nbr, perm,
            val_s, val_n, d, l0, bw, fold_xty_chunk);
          tk_ridge_solve_chunk(train_gram, d, full_xty_chunk, fold_xty_chunk, tr_inv,
            train_col_mean, train_y_mean + l0,
            fold_label_counts + l0, do_prop, prop_a, prop_b, C,
            bw, W_chunk, intercept_chunk);
          tk_ridge_predict_chunk(val_codes_buf, val_n, d, W_chunk, bw, intercept_chunk,
            l0, sample_block, oof_heaps, oof_hcounts, k,
            do_gather ? gather_sco_dv->a : NULL, gather_off, gather_nbr,
            NULL, perm + val_s, sbuf_chunk);
        }
      } else {
        for (int64_t i = 0; i < (int64_t)dnl; i++)
          fold_XtY[i] = (full_XtY[i] - fold_XtY[i]) * tr_inv;
        cblas_dger(CblasRowMajor, (int)d, (int)nl, -1.0,
          train_col_mean, 1, train_y_mean, 1, fold_XtY, (int)nl);
        if (do_prop) {
          for (int64_t l = 0; l < nl; l++)
            fold_label_counts[l] = full_label_counts[l] - fold_label_counts[l];
          double C = (log((double)tr_n) - 1.0) * pow(prop_b + 1.0, prop_a);
          for (int64_t l = 0; l < nl; l++) {
            double pw = 1.0 + C / pow(fold_label_counts[l] + prop_b, prop_a);
            prop_y_mean[l] = pw * train_y_mean[l];
            for (int64_t i = 0; i < d; i++)
              fold_XtY[i * nl + l] *= pw;
          }
        } else {
          memcpy(prop_y_mean, train_y_mean, (uint64_t)nl * sizeof(double));
        }
        cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
          (int)d, (int)nl, 1.0, train_gram, (int)d, fold_XtY, (int)nl);
        cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
          (int)d, (int)nl, 1.0, train_gram, (int)d, fold_XtY, (int)nl);
        double *W = fold_XtY;
        double *intercept = prop_y_mean;
        cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl, -1.0,
          W, (int)nl, train_col_mean, 1, 1.0, intercept, 1);
        if (!dense) {
          tk_ridge_predict_chunk(val_codes_buf, val_n, d, W, nl, intercept,
            0, sample_block, oof_heaps, oof_hcounts, k,
            do_gather ? gather_sco_dv->a : NULL, gather_off, gather_nbr,
            do_transform ? oof_transform->a : NULL, perm + val_s, sbuf);
        } else {
          cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            (int)val_n, (int)nl, (int)d, 1.0, val_codes_buf, (int)d,
            W, (int)nl, 0.0, sbuf, (int)nl);
          tk_ridge_add_intercept(sbuf, val_n, nl, intercept);
          for (int64_t vi = 0; vi < val_n; vi++) {
            int64_t orig = perm[val_s + vi];
            memcpy(oof_dense->a + orig * nl, sbuf + vi * nl, (uint64_t)nl * sizeof(double));
          }
        }
      }
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
    if (oof_heaps) {
      for (int64_t i = 0; i < n; i++) {
        int64_t dst = i * k;
        int64_t cnt = oof_hcounts[i];
        tk_rvec_t h = { .a = oof_heaps + dst, .m = (size_t)k, .n = (size_t)cnt };
        tk_rvec_desc(&h, 0, h.n);
        for (int64_t j = 0; j < cnt; j++) {
          oof_nbr->a[dst + j] = h.a[j].i;
          oof_sco->a[dst + j] = h.a[j].d;
        }
      }
    }
#pragma GCC diagnostic pop
    free(oof_heaps); free(oof_hcounts);
    free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
    free(full_y_sum); free(full_label_counts);
    free(fold_col_sum); free(fold_XtX); free(fold_y_sum); free(fold_label_counts);
    free(train_gram); free(train_col_mean); free(train_y_mean); free(val_codes_buf);
    free(fold_XtY); free(prop_y_mean); free(sbuf);
    free(csc_off); free(csc_rows); free(full_xty_chunk); free(fold_xty_chunk);
    free(W_chunk); free(intercept_chunk); free(sbuf_chunk);
  }

push_results:
  if (!dense) {
    if (have_k && (do_transform || do_gather)) {
      lua_pushvalue(L, oof_off_idx);
      lua_pushvalue(L, oof_nbr_idx);
      lua_pushvalue(L, oof_sco_idx);
      lua_pushvalue(L, do_gather ? gather_sco_idx : oof_tr_idx);
      return 4;
    }
    if (have_k) {
      lua_pushvalue(L, oof_off_idx);
      lua_pushvalue(L, oof_nbr_idx);
      lua_pushvalue(L, oof_sco_idx);
      return 3;
    }
    lua_pushvalue(L, do_gather ? gather_sco_idx : oof_tr_idx);
    return 1;
  } else {
    lua_pushvalue(L, oof_dense_idx);
    return 1;
  }
}

static luaL_Reg tk_ridge_fns[] = {
  { "create", tk_ridge_create_lua },
  { "prepare", tk_ridge_precompute_lua },
  { "load", tk_ridge_load_lua },
  { "solve", tk_ridge_solve_lua },
  { "oof", tk_ridge_solve_oof_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_ridge (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_ridge_fns, 0);
  return 1;
}
