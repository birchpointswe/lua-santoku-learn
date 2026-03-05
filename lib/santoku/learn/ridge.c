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
  if (!dense && !have_k && !do_transform)
    return luaL_error(L, "solve_oof: label mode requires k and/or transform");

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
  double *full_XtY = (double *)calloc(dnl, sizeof(double));
  double *full_y_sum = (double *)calloc((uint64_t)nl, sizeof(double));
  double *full_label_counts = (!dense) ? (double *)calloc((uint64_t)nl, sizeof(double)) : NULL;
  if (!perm || !full_col_sum || !full_XtX || !full_XtY || !full_y_sum || (!dense && !full_label_counts)) {
    free(perm); free(full_col_sum); free(full_XtX); free(full_XtY); free(full_y_sum); free(full_label_counts);
    return luaL_error(L, "solve_oof: malloc failed");
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
    for (int64_t i = 0; i < n; i++) {
      int64_t lo = lab_off->a[i], hi = lab_off->a[i + 1];
      for (int64_t j = lo; j < hi; j++)
        full_label_counts[lab_nbr->a[j]] += 1.0;
    }
    for (int64_t l = 0; l < nl; l++)
      full_y_sum[l] = full_label_counts[l];
  }

  tk_ivec_t *oof_off = NULL;
  tk_ivec_t *oof_nbr = NULL;
  tk_dvec_t *oof_sco = NULL;
  tk_dvec_t *oof_dense = NULL;
  tk_dvec_t *oof_transform = NULL;
  int oof_off_idx = 0, oof_nbr_idx = 0, oof_sco_idx = 0, oof_dense_idx = 0, oof_tr_idx = 0;
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
      oof_transform = tk_dvec_create(L, (uint64_t)(n * nl), NULL, NULL);
      oof_tr_idx = lua_gettop(L);
      memset(oof_transform->a, 0, (uint64_t)(n * nl) * sizeof(double));
    }
  } else {
    oof_dense = tk_dvec_create(L, (uint64_t)(n * nl), NULL, NULL);
    oof_dense_idx = lua_gettop(L);
    memset(oof_dense->a, 0, (uint64_t)(n * nl) * sizeof(double));
  }

  if (nf < 2) {
    double inv_n = 1.0 / (double)n;
    for (int64_t j = 0; j < d; j++) full_col_sum[j] *= inv_n;
    for (int64_t p = 0; p < d; p++)
      for (int64_t q = p; q < d; q++)
        full_XtX[p * d + q] *= inv_n;
    cblas_dsyr(CblasRowMajor, CblasUpper, (int)d, -1.0, full_col_sum, 1, full_XtX, (int)d);
    for (int64_t l = 0; l < nl; l++) full_y_sum[l] *= inv_n;
    for (int64_t p = 0; p < d; p++)
      for (int64_t l = 0; l < nl; l++)
        full_XtY[p * nl + l] *= inv_n;
    cblas_dger(CblasRowMajor, (int)d, (int)nl, -1.0,
      full_col_sum, 1, full_y_sum, 1, full_XtY, (int)nl);
    double *intercept = (double *)malloc((uint64_t)nl * sizeof(double));
    if (!intercept) {
      free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
      free(full_y_sum); free(full_label_counts);
      return luaL_error(L, "solve_oof: malloc failed");
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
      free(full_y_sum); free(full_label_counts); free(intercept);
      return luaL_error(L, "solve_oof: Cholesky failed (info=%d)", info);
    }
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
      (int)d, (int)nl, 1.0, full_XtX, (int)d, full_XtY, (int)nl);
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
      (int)d, (int)nl, 1.0, full_XtX, (int)d, full_XtY, (int)nl);
    double *W = full_XtY;
    cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl, -1.0,
      W, (int)nl, full_col_sum, 1, 1.0, intercept, 1);
    if (dense) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)n, (int)nl, (int)d, 1.0, codes->a, (int)d,
        W, (int)nl, 0.0, oof_dense->a, (int)nl);
      tk_ridge_add_intercept(oof_dense->a, n, nl, intercept);
    } else {
      int64_t chunk = 256;
      double *sbuf2 = (double *)malloc((uint64_t)chunk * (uint64_t)nl * sizeof(double));
      tk_rvec_t heap = { .n = 0, .m = (size_t)k, .lua_managed = false,
                         .a = (tk_rank_t *)malloc((uint64_t)k * sizeof(tk_rank_t)) };
      if (!sbuf2 || (have_k && !heap.a)) {
        free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
        free(full_y_sum); free(full_label_counts); free(intercept);
        free(sbuf2); free(heap.a);
        return luaL_error(L, "solve_oof: malloc failed");
      }
      for (int64_t s = 0; s < n; s += chunk) {
        int64_t cn = (s + chunk > n) ? (n - s) : chunk;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          (int)cn, (int)nl, (int)d, 1.0, codes->a + s * d, (int)d,
          W, (int)nl, 0.0, sbuf2, (int)nl);
        tk_ridge_add_intercept(sbuf2, cn, nl, intercept);
        for (int64_t vi = 0; vi < cn; vi++) {
          int64_t orig = s + vi;
          double *row = sbuf2 + vi * nl;
          if (have_k) {
            int64_t dst = orig * k;
            heap.n = 0;
            for (int64_t l = 0; l < nl; l++)
              tk_rvec_hmin(&heap, (size_t)k, tk_rank(l, row[l]));
            tk_rvec_desc(&heap, 0, heap.n);
            for (int64_t j = 0; j < (int64_t)heap.n; j++) {
              oof_nbr->a[dst + j] = heap.a[j].i;
              oof_sco->a[dst + j] = heap.a[j].d;
            }
          }
          if (do_transform)
            memcpy(oof_transform->a + orig * nl, row, (uint64_t)nl * sizeof(double));
        }
      }
      free(sbuf2);
      free(heap.a);
    }
    free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
    free(full_y_sum); free(full_label_counts); free(intercept);
    goto push_results;
  }

  {
    double *fold_col_sum = (double *)calloc(dd, sizeof(double));
    double *fold_XtX = (double *)calloc(dd * dd, sizeof(double));
    double *fold_XtY = (double *)calloc(dnl, sizeof(double));
    double *fold_y_sum = (double *)calloc((uint64_t)nl, sizeof(double));
    double *fold_label_counts = (!dense) ? (double *)calloc((uint64_t)nl, sizeof(double)) : NULL;
    double *train_gram = (double *)malloc(dd * dd * sizeof(double));
    double *train_xty = (double *)malloc(dnl * sizeof(double));
    double *train_col_mean = (double *)malloc(dd * sizeof(double));
    double *train_y_mean = (double *)malloc((uint64_t)nl * sizeof(double));
    double *prop_y_mean = (double *)malloc((uint64_t)nl * sizeof(double));
    int64_t fold_size = n / nf;
    int64_t max_val = n - (nf - 1) * fold_size;
    double *sbuf = (double *)malloc((uint64_t)max_val * (uint64_t)nl * sizeof(double));
    double *val_codes_buf = (double *)malloc((uint64_t)max_val * dd * sizeof(double));

    if (!fold_col_sum || !fold_XtX || !fold_XtY || !fold_y_sum || !train_gram ||
        !train_xty || !train_col_mean || !train_y_mean || !prop_y_mean ||
        !sbuf || !val_codes_buf || (!dense && !fold_label_counts)) {
      free(perm); free(full_col_sum); free(full_XtX); free(full_XtY);
      free(full_y_sum); free(full_label_counts);
      free(fold_col_sum); free(fold_XtX); free(fold_XtY); free(fold_y_sum); free(fold_label_counts);
      free(train_gram); free(train_xty); free(train_col_mean); free(train_y_mean); free(prop_y_mean);
      free(sbuf); free(val_codes_buf);
      return luaL_error(L, "solve_oof: malloc failed");
    }

    for (int64_t f = 0; f < nf; f++) {
      int64_t val_s = f * fold_size;
      int64_t val_e = (f == nf - 1) ? n : (val_s + fold_size);
      int64_t val_n = val_e - val_s;
      int64_t tr_n = n - val_n;
      double tr_inv = 1.0 / (double)tr_n;

      memset(fold_col_sum, 0, dd * sizeof(double));
      memset(fold_XtX, 0, dd * dd * sizeof(double));
      memset(fold_XtY, 0, dnl * sizeof(double));
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
        for (int64_t vi = 0; vi < val_n; vi++) {
          int64_t idx = perm[val_s + vi];
          int64_t lo = lab_off->a[idx], hi = lab_off->a[idx + 1];
          for (int64_t j = lo; j < hi; j++)
            fold_label_counts[lab_nbr->a[j]] += 1.0;
        }
        for (int64_t l = 0; l < nl; l++)
          fold_y_sum[l] = fold_label_counts[l];
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

      for (int64_t p = 0; p < d; p++)
        for (int64_t l = 0; l < nl; l++)
          train_xty[p * nl + l] = (full_XtY[p * nl + l] - fold_XtY[p * nl + l]) * tr_inv;
      cblas_dger(CblasRowMajor, (int)d, (int)nl, -1.0,
        train_col_mean, 1, train_y_mean, 1, train_xty, (int)nl);

      if (do_prop) {
        double *tr_lc = (double *)malloc((uint64_t)nl * sizeof(double));
        for (int64_t l = 0; l < nl; l++)
          tr_lc[l] = full_label_counts[l] - fold_label_counts[l];
        double C = (log((double)tr_n) - 1.0) * pow(prop_b + 1.0, prop_a);
        for (int64_t l = 0; l < nl; l++) {
          double pw = 1.0 + C / pow(tr_lc[l] + prop_b, prop_a);
          prop_y_mean[l] = pw * train_y_mean[l];
          for (int64_t i = 0; i < d; i++)
            train_xty[i * nl + l] *= pw;
        }
        free(tr_lc);
      } else {
        memcpy(prop_y_mean, train_y_mean, (uint64_t)nl * sizeof(double));
      }

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
        free(fold_col_sum); free(fold_XtX); free(fold_XtY); free(fold_y_sum); free(fold_label_counts);
        free(train_gram); free(train_xty); free(train_col_mean); free(train_y_mean); free(prop_y_mean);
        free(sbuf); free(val_codes_buf);
        return luaL_error(L, "solve_oof: Cholesky failed fold %d (info=%d)", (int)f, info);
      }
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
        (int)d, (int)nl, 1.0, train_gram, (int)d, train_xty, (int)nl);
      cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
        (int)d, (int)nl, 1.0, train_gram, (int)d, train_xty, (int)nl);

      double *W = train_xty;
      double *intercept = prop_y_mean;
      cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl, -1.0,
        W, (int)nl, train_col_mean, 1, 1.0, intercept, 1);

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)val_n, (int)nl, (int)d, 1.0, val_codes_buf, (int)d,
        W, (int)nl, 0.0, sbuf, (int)nl);
      tk_ridge_add_intercept(sbuf, val_n, nl, intercept);

      if (!dense) {
        tk_rvec_t heap = { .n = 0, .m = have_k ? (size_t)k : 0, .lua_managed = false,
                           .a = have_k ? (tk_rank_t *)malloc((uint64_t)k * sizeof(tk_rank_t)) : NULL };
        for (int64_t vi = 0; vi < val_n; vi++) {
          int64_t orig = perm[val_s + vi];
          double *row = sbuf + vi * nl;
          if (have_k) {
            int64_t dst = orig * k;
            heap.n = 0;
            for (int64_t l = 0; l < nl; l++)
              tk_rvec_hmin(&heap, (size_t)k, tk_rank(l, row[l]));
            tk_rvec_desc(&heap, 0, heap.n);
            for (int64_t j = 0; j < (int64_t)heap.n; j++) {
              oof_nbr->a[dst + j] = heap.a[j].i;
              oof_sco->a[dst + j] = heap.a[j].d;
            }
          }
          if (do_transform)
            memcpy(oof_transform->a + orig * nl, row, (uint64_t)nl * sizeof(double));
        }
        free(heap.a);
      } else {
        for (int64_t vi = 0; vi < val_n; vi++) {
          int64_t orig = perm[val_s + vi];
          memcpy(oof_dense->a + orig * nl, sbuf + vi * nl, (uint64_t)nl * sizeof(double));
        }
      }

      memcpy(prop_y_mean, train_y_mean, (uint64_t)nl * sizeof(double));
    }

    free(perm);
    free(full_col_sum); free(full_XtX); free(full_XtY); free(full_y_sum); free(full_label_counts);
    free(fold_col_sum); free(fold_XtX); free(fold_XtY); free(fold_y_sum); free(fold_label_counts);
    free(train_gram); free(train_xty); free(train_col_mean); free(train_y_mean); free(prop_y_mean);
    free(sbuf); free(val_codes_buf);
  }

push_results:
  if (!dense) {
    if (have_k && do_transform) {
      lua_pushvalue(L, oof_off_idx);
      lua_pushvalue(L, oof_nbr_idx);
      lua_pushvalue(L, oof_sco_idx);
      lua_pushvalue(L, oof_tr_idx);
      return 4;
    }
    if (have_k) {
      lua_pushvalue(L, oof_off_idx);
      lua_pushvalue(L, oof_nbr_idx);
      lua_pushvalue(L, oof_sco_idx);
      return 3;
    }
    lua_pushvalue(L, oof_tr_idx);
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
