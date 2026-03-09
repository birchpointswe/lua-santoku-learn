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
#include <santoku/learn/gram.h>

#define TK_RIDGE_MT "tk_ridge_t"

typedef struct {
  tk_dvec_t *W;
  tk_dvec_t *intercept;
  int64_t n_dims;
  int64_t n_labels;
  bool destroyed;
} tk_ridge_t;

static inline tk_ridge_t *tk_ridge_peek (lua_State *L, int i) {
  return (tk_ridge_t *)luaL_checkudata(L, i, TK_RIDGE_MT);
}

static inline int tk_ridge_gc (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  r->W = NULL;
  r->intercept = NULL;
  r->destroyed = true;
  return 0;
}

static inline double *tk_ridge_transpose_w (tk_ridge_t *r) {
  int64_t d = r->n_dims, nl = r->n_labels;
  double *wt = (double *)malloc((uint64_t)nl * (uint64_t)d * sizeof(double));
  for (int64_t i = 0; i < d; i++)
    for (int64_t j = 0; j < nl; j++)
      wt[j * d + i] = r->W->a[i * nl + j];
  return wt;
}


static inline int tk_ridge_encode_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  int64_t nl = r->n_labels, d = r->n_dims;
  int64_t n = (int64_t)luaL_checkinteger(L, 3);
  tk_dvec_t *codes_dvec = tk_dvec_peek(L, 2, "codes");
  double *codes_a = codes_dvec->a;
  int64_t k = (int64_t)luaL_checkinteger(L, 4);
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  int orig_top = lua_gettop(L);
  TK_IVEC_BUF(offsets, 5, n + 1);
  TK_IVEC_BUF(labels, 6, n * k);
  TK_DVEC_BUF(scores_out, 7, n * k);
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
        double *row = codes_a + i * d;
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
        (int)bs, (int)nl, (int)d, 1.0, codes_a + base * d, (int)d,
        r->W->a, (int)nl, 0.0, sbuf, (int)nl);
      if (r->intercept)
        tk_gram_add_intercept(sbuf, bs, nl, r->intercept->a);
      tk_gram_topk_block(sbuf, bs, nl, k, base, offsets, labels, scores_out);
    }
    free(sbuf);
  }
  lua_pushvalue(L, offsets_idx);
  lua_pushvalue(L, labels_idx);
  lua_pushvalue(L, scores_out_idx);
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
  int64_t n = (int64_t)luaL_checkinteger(L, 3);
  int64_t d = r->n_dims, nl = r->n_labels;
  tk_dvec_t *codes_dvec = tk_dvec_peek(L, 2, "codes");
  double *codes_a = codes_dvec->a;
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
      double *row = codes_a + i * d;
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
  TK_DVEC_BUF(out, 4, n * nl);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)n, (int)nl, (int)d, 1.0, codes_a, (int)d,
    r->W->a, (int)nl, 0.0, out->a, (int)nl);
  if (r->intercept)
    tk_gram_add_intercept(out->a, n, nl, r->intercept->a);
  return 1;
}

static luaL_Reg tk_ridge_mt_fns[] = {
  { "label", tk_ridge_encode_lua },
  { "persist", tk_ridge_persist_lua },
  { "regress", tk_ridge_transform_lua },
  { NULL, NULL }
};




static inline int tk_ridge_precompute_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 1, "prepare", "n_samples");
  int64_t d = (int64_t)tk_lua_fcheckunsigned(L, 1, "prepare", "n_dims");

  lua_getfield(L, 1, "XtX");
  tk_dvec_t *in_xtx = tk_dvec_peek(L, -1, "XtX");
  lua_pop(L, 1);
  lua_getfield(L, 1, "XtY");
  tk_dvec_t *in_xty = tk_dvec_peek(L, -1, "XtY");
  lua_pop(L, 1);
  lua_getfield(L, 1, "col_mean");
  tk_dvec_t *in_cm = tk_dvec_peek(L, -1, "col_mean");
  lua_pop(L, 1);
  lua_getfield(L, 1, "y_mean");
  tk_dvec_t *in_ym = tk_dvec_peek(L, -1, "y_mean");
  lua_pop(L, 1);
  lua_getfield(L, 1, "label_counts");
  bool dense = lua_isnil(L, -1);
  tk_dvec_t *in_lc = dense ? NULL : tk_dvec_peek(L, -1, "label_counts");
  lua_pop(L, 1);
  int64_t nl;
  if (dense)
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "prepare", "n_targets");
  else
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "prepare", "n_labels");

  tk_dvec_t *cm_dvec = tk_dvec_create(L, (uint64_t)d, NULL, NULL);
  int cm_idx = lua_gettop(L);
  memcpy(cm_dvec->a, in_cm->a, (uint64_t)d * sizeof(double));

  tk_dvec_t *ev_dvec = tk_dvec_create(L, (uint64_t)(d * d), NULL, NULL);
  int ev_idx = lua_gettop(L);
  memcpy(ev_dvec->a, in_xtx->a, (uint64_t)(d * d) * sizeof(double));

  uint64_t dnl = (uint64_t)d * (uint64_t)nl;
  tk_dvec_t *work_dvec = tk_dvec_create(L, dnl, NULL, NULL);
  int work_idx = lua_gettop(L);
  memcpy(work_dvec->a, in_xty->a, dnl * sizeof(double));
  double *xty = work_dvec->a;

  cblas_dsyr(CblasRowMajor, CblasUpper, (int)d, -1.0,
    cm_dvec->a, 1, ev_dvec->a, (int)d);
  for (int64_t i = 0; i < d; i++)
    for (int64_t j = i + 1; j < d; j++)
      ev_dvec->a[j * d + i] = ev_dvec->a[i * d + j];

  tk_dvec_t *eig_dvec = tk_dvec_create(L, (uint64_t)d, NULL, NULL);
  int eig_idx = lua_gettop(L);
  int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (int)d,
    ev_dvec->a, (int)d, eig_dvec->a);
  if (info != 0)
    return luaL_error(L, "prepare: eigendecomposition failed (info=%d)", info);

  tk_dvec_t *cnt_dvec = NULL;
  int cnt_idx = 0;
  if (!dense) {
    cnt_dvec = tk_dvec_create(L, (uint64_t)nl, NULL, NULL);
    cnt_idx = lua_gettop(L);
    memcpy(cnt_dvec->a, in_lc->a, (uint64_t)nl * sizeof(double));
  }

  tk_dvec_t *ym_dvec = tk_dvec_create(L, (uint64_t)nl, NULL, NULL);
  int ym_idx = lua_gettop(L);
  memcpy(ym_dvec->a, in_ym->a, (uint64_t)nl * sizeof(double));

  cblas_dger(CblasRowMajor, (int)d, (int)nl, -1.0,
    cm_dvec->a, 1, ym_dvec->a, 1, xty, (int)nl);

  tk_dvec_t *rxty_dvec = tk_dvec_create(L, dnl, NULL, NULL);
  int rxty_idx = lua_gettop(L);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)d, (int)nl, (int)d, 1.0, ev_dvec->a, (int)d,
    xty, (int)nl, 0.0, rxty_dvec->a, (int)nl);

  tk_gram_t *g = tk_lua_newuserdata(L, tk_gram_t,
    TK_GRAM_MT, tk_gram_mt_fns, tk_gram_gc);
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
    tk_gram_t *gram = tk_gram_peek(L, -1);
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

static luaL_Reg tk_ridge_fns[] = {
  { "create", tk_ridge_create_lua },
  { "prepare", tk_ridge_precompute_lua },
  { "load", tk_ridge_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_ridge (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_ridge_fns, 0);
  return 1;
}
