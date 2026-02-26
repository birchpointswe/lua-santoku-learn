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
  int64_t n_dims;
  int64_t n_labels;
  int64_t n_samples;
  bool destroyed;
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
  r->destroyed = true;
  return 0;
}

static inline int tk_ridge_gram_gc (lua_State *L) {
  tk_ridge_gram_t *g = tk_ridge_gram_peek(L, 1);
  g->evecs = NULL;
  g->eigenvals = NULL;
  g->RXtY = NULL;
  g->label_counts = NULL;
  g->work = NULL;
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

static inline int tk_ridge_encode_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  int64_t nl = r->n_labels, d = r->n_dims;
  tk_dvec_t *codes = tk_dvec_peek(L, 2, "codes");
  int64_t n = (int64_t)luaL_checkinteger(L, 3);
  int64_t k = (int64_t)luaL_checkinteger(L, 4);
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n + 1), NULL, NULL);
  int off_idx = lua_gettop(L);
  tk_ivec_t *labels = tk_ivec_create(L, (uint64_t)(n * k), NULL, NULL);
  int lab_idx = lua_gettop(L);
  tk_dvec_t *scores_out = tk_dvec_create(L, (uint64_t)(n * k), NULL, NULL);
  int sco_idx = lua_gettop(L);
  for (int64_t i = 0; i <= n; i++)
    offsets->a[i] = i * k;
  int64_t block = 256;
  while (block > 1 && (uint64_t)block * (uint64_t)nl * sizeof(double) > 64ULL * 1024 * 1024)
    block /= 2;
  double *sbuf = (double *)malloc((uint64_t)block * (uint64_t)nl * sizeof(double));
  for (int64_t base = 0; base < n; base += block) {
    int64_t bs = (base + block <= n) ? block : n - base;
    int64_t cs = r->n_dims;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)bs, (int)nl, (int)d, 1.0, codes->a + base * cs, (int)cs,
      r->W->a, (int)nl, 0.0, sbuf, (int)nl);
    tk_ridge_topk_block(sbuf, bs, nl, k, base, offsets, labels, scores_out);
  }
  free(sbuf);
  lua_pushvalue(L, off_idx);
  lua_pushvalue(L, lab_idx);
  lua_pushvalue(L, sco_idx);
  return 3;
}

static inline int tk_ridge_n_dims_lua (lua_State *L) {
  lua_pushinteger(L, (lua_Integer)tk_ridge_peek(L, 1)->n_dims);
  return 1;
}

static inline int tk_ridge_n_labels_lua (lua_State *L) {
  lua_pushinteger(L, (lua_Integer)tk_ridge_peek(L, 1)->n_labels);
  return 1;
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
  uint8_t version = 1;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &r->n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &r->n_labels, sizeof(int64_t), 1, fh);
  tk_dvec_persist(L, r->W, fh);
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

static inline int tk_ridge_dim_weights_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  int64_t d = r->n_dims, nl = r->n_labels;
  tk_dvec_t *out = tk_dvec_create(L, (uint64_t)d, NULL, NULL);
  for (int64_t i = 0; i < d; i++) {
    double s = 0.0;
    double *row = r->W->a + i * nl;
    for (int64_t j = 0; j < nl; j++)
      s += fabs(row[j]);
    out->a[i] = s / (double)nl;
  }
  return 1;
}

static inline int tk_ridge_transform_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  tk_dvec_t *codes = tk_dvec_peek(L, 2, "codes");
  int64_t n = (int64_t)luaL_checkinteger(L, 3);
  int64_t d = r->n_dims, nl = r->n_labels;
  tk_dvec_t *out = tk_dvec_create(L, (uint64_t)(n * nl), NULL, NULL);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)n, (int)nl, (int)d, 1.0, codes->a, (int)d,
    r->W->a, (int)nl, 0.0, out->a, (int)nl);
  return 1;
}

static luaL_Reg tk_ridge_mt_fns[] = {
  { "label", tk_ridge_encode_lua },
  { "n_dims", tk_ridge_n_dims_lua },
  { "n_labels", tk_ridge_n_labels_lua },
  { "dim_weights", tk_ridge_dim_weights_lua },
  { "persist", tk_ridge_persist_lua },
  { "transform", tk_ridge_transform_lua },
  { NULL, NULL }
};

static inline int tk_ridge_precompute_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 1, "precompute", "n_samples");
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  lua_pop(L, 1);
  int64_t d = (int64_t)tk_lua_fcheckunsigned(L, 1, "precompute", "n_dims");
  lua_getfield(L, 1, "targets");
  bool dense = !lua_isnil(L, -1);
  tk_dvec_t *targets = dense ? tk_dvec_peek(L, -1, "targets") : NULL;
  lua_pop(L, 1);
  int64_t nl;
  tk_ivec_t *lab_off = NULL, *lab_nbr = NULL;
  if (dense) {
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "precompute", "n_targets");
  } else {
    nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "precompute", "n_labels");
    lua_getfield(L, 1, "label_offsets");
    lab_off = tk_ivec_peek(L, -1, "label_offsets");
    lua_getfield(L, 1, "label_neighbors");
    lab_nbr = tk_ivec_peek(L, -1, "label_neighbors");
    lua_pop(L, 2);
  }
  tk_dvec_t *ev_dvec = tk_dvec_create(L, (uint64_t)(d * d), NULL, NULL);
  int ev_idx = lua_gettop(L);
  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
    (int)d, (int)n, 1.0, codes->a, (int)d, 0.0, ev_dvec->a, (int)d);
  for (int64_t i = 0; i < d; i++)
    for (int64_t j = i + 1; j < d; j++)
      ev_dvec->a[j * d + i] = ev_dvec->a[i * d + j];
  tk_dvec_t *eig_dvec = tk_dvec_create(L, (uint64_t)d, NULL, NULL);
  int eig_idx = lua_gettop(L);
  int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (int)d, ev_dvec->a, (int)d, eig_dvec->a);
  if (info != 0)
    return luaL_error(L, "precompute: eigendecomposition failed (info=%d)", info);
  uint64_t dnl = (uint64_t)d * (uint64_t)nl;
  double *xty = (double *)malloc(dnl * sizeof(double));
  if (!xty) return luaL_error(L, "precompute: malloc failed");
  memset(xty, 0, dnl * sizeof(double));
  if (dense) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)d, (int)nl, (int)n, 1.0, codes->a, (int)d,
      targets->a, (int)nl, 0.0, xty, (int)nl);
  } else {
    #pragma omp parallel for schedule(static)
    for (int64_t dd = 0; dd < d; dd++) {
      double *xty_row = xty + dd * nl;
      for (int64_t i = 0; i < n; i++) {
        double xi_dd = codes->a[i * d + dd];
        int64_t lo = lab_off->a[i], hi = lab_off->a[i + 1];
        for (int64_t j = lo; j < hi; j++)
          xty_row[lab_nbr->a[j]] += xi_dd;
      }
    }
  }
  tk_dvec_t *rxty_dvec = tk_dvec_create(L, dnl, NULL, NULL);
  int rxty_idx = lua_gettop(L);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)d, (int)nl, (int)d, 1.0, ev_dvec->a, (int)d,
    xty, (int)nl, 0.0, rxty_dvec->a, (int)nl);
  free(xty);
  tk_dvec_t *work_dvec = tk_dvec_create(L, dnl, NULL, NULL);
  int work_idx = lua_gettop(L);
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
  tk_ridge_gram_t *g = tk_lua_newuserdata(L, tk_ridge_gram_t,
    TK_RIDGE_GRAM_MT, NULL, tk_ridge_gram_gc);
  int Gi = lua_gettop(L);
  g->evecs = ev_dvec;
  g->eigenvals = eig_dvec;
  g->RXtY = rxty_dvec;
  g->label_counts = cnt_dvec;
  g->work = work_dvec;
  g->n_dims = d;
  g->n_labels = nl;
  g->n_samples = n;
  g->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, ev_idx);
  lua_setfield(L, -2, "evecs");
  lua_pushvalue(L, eig_idx);
  lua_setfield(L, -2, "eigenvals");
  lua_pushvalue(L, rxty_idx);
  lua_setfield(L, -2, "RXtY");
  lua_pushvalue(L, work_idx);
  lua_setfield(L, -2, "work");
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
    double lambda = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.0;
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
        return luaL_error(L, "ridge create: propensity requires label precompute");
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
    tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
      TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
    int Ei = lua_gettop(L);
    r->W = W_dvec;
    r->n_dims = d;
    r->n_labels = nl;
    r->destroyed = false;
    lua_newtable(L);
    lua_pushvalue(L, W_idx);
    lua_setfield(L, -2, "W");
    lua_pushvalue(L, gram_lua_idx);
    lua_setfield(L, -2, "gram");
    lua_setfenv(L, Ei);
    lua_pushvalue(L, Ei);
    lua_pushvalue(L, W_idx);
    return 2;
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
  if (version != 1) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported ridge version %d (expected 1)", (int)version);
  }
  int64_t n_dims, n_labels;
  tk_lua_fread(L, &n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_labels, sizeof(int64_t), 1, fh);
  tk_dvec_t *W = tk_dvec_load(L, fh);
  int W_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);
  tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
    TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
  int Ei = lua_gettop(L);
  r->W = W;
  r->n_dims = n_dims;
  r->n_labels = n_labels;
  r->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, W_idx);
  lua_setfield(L, -2, "W");
  lua_setfenv(L, Ei);
  lua_pushvalue(L, Ei);
  return 1;
}

static luaL_Reg tk_ridge_fns[] = {
  { "create", tk_ridge_create_lua },
  { "precompute", tk_ridge_precompute_lua },
  { "load", tk_ridge_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_ridge (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_ridge_fns, 0);
  return 1;
}
