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
#include <santoku/fvec.h>
#include <santoku/rvec.h>
#include <santoku/learn/gram.h>

#define TK_RIDGE_MT "tk_ridge_t"

typedef struct {
  tk_fvec_t *W;
  tk_dvec_t *intercept;
  int64_t n_dims;
  int64_t n_labels;
  float *Wt;
  float *sbuf;
  uint64_t sbuf_size;
  tk_rank_t *heap_buf;
  uint64_t heap_buf_size;
  bool destroyed;
} tk_ridge_t;

static inline tk_ridge_t *tk_ridge_peek (lua_State *L, int i) {
  return (tk_ridge_t *)luaL_checkudata(L, i, TK_RIDGE_MT);
}

static inline int tk_ridge_gc (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  free(r->Wt);
  free(r->sbuf);
  free(r->heap_buf);
  r->W = NULL;
  r->intercept = NULL;
  r->Wt = NULL;
  r->sbuf = NULL;
  r->heap_buf = NULL;
  r->destroyed = true;
  return 0;
}

static inline float *tk_ridge_get_wt (tk_ridge_t *r) {
  if (r->Wt) return r->Wt;
  int64_t d = r->n_dims, nl = r->n_labels;
  r->Wt = (float *)malloc((uint64_t)nl * (uint64_t)d * sizeof(float));
  for (int64_t i = 0; i < d; i++)
    for (int64_t j = 0; j < nl; j++)
      r->Wt[j * d + i] = r->W->a[i * nl + j];
  return r->Wt;
}


static inline int tk_ridge_encode_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  int64_t nl = r->n_labels, d = r->n_dims;
  int64_t n = (int64_t)luaL_checkinteger(L, 3);
  tk_fvec_t *codes_fvec = tk_fvec_peek(L, 2, "codes");
  float *codes_a = codes_fvec->a;
  int64_t k = (int64_t)luaL_checkinteger(L, 4);
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  int orig_top = lua_gettop(L);
  bool sparse = orig_top >= 5 && !lua_isnil(L, 5);
  int buf_base = sparse ? 7 : 5;
  lua_settop(L, buf_base - 1);
  TK_IVEC_BUF(offsets, buf_base, n + 1);
  TK_IVEC_BUF(labels, buf_base + 1, n * k);
  TK_FVEC_BUF(scores_out, buf_base + 2, n * k);
  for (int64_t i = 0; i <= n; i++)
    offsets->a[i] = i * k;
  {
    int nt = omp_get_max_threads();
    uint64_t heap_need = (uint64_t)nt * (uint64_t)k;
    if (!r->heap_buf || r->heap_buf_size < heap_need) {
      free(r->heap_buf);
      r->heap_buf = (tk_rank_t *)malloc(heap_need * sizeof(tk_rank_t));
      r->heap_buf_size = heap_need;
    }
  }
  if (sparse) {
    tk_ivec_t *csr_off = tk_ivec_peek(L, 5, "csr_off");
    tk_ivec_t *csr_nbr = tk_ivec_peek(L, 6, "csr_nbr");
    float *wt = tk_ridge_get_wt(r);
    double *intercept = r->intercept ? r->intercept->a : NULL;
    #pragma omp parallel
    {
      tk_rvec_t heap = { .n = 0, .m = (size_t)k, .lua_managed = false,
                         .a = r->heap_buf + (uint64_t)omp_get_thread_num() * (uint64_t)k };
      #pragma omp for schedule(dynamic, 64)
      for (int64_t i = 0; i < n; i++) {
        float *row = codes_a + i * d;
        int64_t lo = csr_off->a[i], hi = csr_off->a[i + 1];
        int64_t out_base = i * k;
        heap.n = 0;
        for (int64_t j = lo; j < hi; j++) {
          int64_t label = csr_nbr->a[j];
          double s = (double)cblas_sdot((int)d, row, 1, wt + label * d, 1);
          if (intercept) s += intercept[label];
          tk_rvec_hmin(&heap, (size_t)k, tk_rank(label, s));
        }
        tk_rvec_desc(&heap, 0, heap.n);
        for (int64_t j = 0; j < (int64_t)heap.n; j++) {
          labels->a[out_base + j] = heap.a[j].i;
          scores_out->a[out_base + j] = (float)heap.a[j].d;
        }
        for (int64_t j = (int64_t)heap.n; j < k; j++) {
          labels->a[out_base + j] = 0;
          scores_out->a[out_base + j] = -1e30f;
        }
      }
    }
  } else {
    int64_t block = 256;
    while (block > 1 && (uint64_t)block * (uint64_t)nl * sizeof(float) > 64ULL * 1024 * 1024)
      block /= 2;
    uint64_t need = (uint64_t)block * (uint64_t)nl;
    if (!r->sbuf || r->sbuf_size < need) {
      free(r->sbuf);
      r->sbuf = (float *)malloc(need * sizeof(float));
      r->sbuf_size = need;
    }
    for (int64_t base = 0; base < n; base += block) {
      int64_t bs = (base + block <= n) ? block : n - base;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)bs, (int)nl, (int)d, 1.0f, codes_a + base * d, (int)d,
        r->W->a, (int)nl, 0.0f, r->sbuf, (int)nl);
      if (r->intercept)
        tk_gram_add_intercept_f(r->sbuf, bs, nl, r->intercept->a);
      #pragma omp parallel
      {
        tk_rvec_t heap = { .n = 0, .m = (size_t)k, .lua_managed = false,
                           .a = r->heap_buf + (uint64_t)omp_get_thread_num() * (uint64_t)k };
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < bs; i++) {
          float *row = r->sbuf + i * nl;
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
      }
    }
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
  uint8_t version = 3;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &r->n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &r->n_labels, sizeof(int64_t), 1, fh);
  tk_fvec_persist(L, r->W, fh);
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
  tk_fvec_t *codes_fvec = tk_fvec_peek(L, 2, "codes");
  float *codes_a = codes_fvec->a;
  if (lua_gettop(L) >= 5 && !lua_isnil(L, 4)) {
    tk_ivec_t *csr_off = tk_ivec_peek(L, 4, "csr_off");
    tk_ivec_t *csr_nbr = tk_ivec_peek(L, 5, "csr_nbr");
    int64_t total = csr_off->a[n];
    TK_FVEC_BUF(out, 6, total);
    float *wt = tk_ridge_get_wt(r);
    double *intercept = r->intercept ? r->intercept->a : NULL;
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t i = 0; i < n; i++) {
      int64_t lo = csr_off->a[i], hi = csr_off->a[i + 1];
      float *row = codes_a + i * d;
      for (int64_t j = lo; j < hi; j++) {
        int64_t label = csr_nbr->a[j];
        double s = (double)cblas_sdot((int)d, row, 1, wt + label * d, 1);
        if (intercept) s += intercept[label];
        out->a[j] = (float)s;
      }
    }
    lua_pushvalue(L, out_idx);
    return 1;
  }
  TK_FVEC_BUF(out, 4, n * nl);
  int64_t block = 256;
  while (block > 1 && (uint64_t)block * (uint64_t)nl * sizeof(float) > 64ULL * 1024 * 1024)
    block /= 2;
  for (int64_t base = 0; base < n; base += block) {
    int64_t bs = (base + block <= n) ? block : n - base;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)bs, (int)nl, (int)d, 1.0f, codes_a + base * d, (int)d,
      r->W->a, (int)nl, 0.0f, out->a + base * nl, (int)nl);
    if (r->intercept)
      tk_gram_add_intercept_f(out->a + base * nl, bs, nl, r->intercept->a);
  }
  lua_pushvalue(L, out_idx);
  return 1;
}

static inline int tk_ridge_shrink_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  free(r->Wt); r->Wt = NULL;
  free(r->sbuf); r->sbuf = NULL; r->sbuf_size = 0;
  free(r->heap_buf); r->heap_buf = NULL; r->heap_buf_size = 0;
  return 0;
}

static luaL_Reg tk_ridge_mt_fns[] = {
  { "label", tk_ridge_encode_lua },
  { "persist", tk_ridge_persist_lua },
  { "regress", tk_ridge_transform_lua },
  { "shrink", tk_ridge_shrink_lua },
  { NULL, NULL }
};

static inline int tk_ridge_gram_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "codes");
  tk_fvec_t *codes_fv = tk_fvec_peek(L, -1, "codes");
  lua_pop(L, 1);
  lua_getfield(L, 1, "n_samples");
  int64_t nc = (int64_t)luaL_checkinteger(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 1, "n_dims");
  int64_t m = (int64_t)luaL_checkinteger(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "label_offsets");
  int has_labels = !lua_isnil(L, -1);
  tk_ivec_t *lbl_off = has_labels ? tk_ivec_peek(L, -1, "label_offsets") : NULL;
  lua_pop(L, 1);
  lua_getfield(L, 1, "label_neighbors");
  tk_ivec_t *lbl_nbr = has_labels ? tk_ivec_peek(L, -1, "label_neighbors") : NULL;
  lua_pop(L, 1);
  lua_getfield(L, 1, "label_values");
  int has_lbl_val = has_labels && !lua_isnil(L, -1);
  tk_dvec_t *lbl_val = has_lbl_val ? tk_dvec_peek(L, -1, "label_values") : NULL;
  lua_pop(L, 1);
  int64_t nl = 0;
  if (has_labels) {
    lua_getfield(L, 1, "n_labels");
    nl = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
  }
  lua_getfield(L, 1, "targets");
  int has_targets = !lua_isnil(L, -1);
  tk_dvec_t *targets_dv = has_targets ? tk_dvec_peek(L, -1, "targets") : NULL;
  lua_pop(L, 1);
  if (has_targets) {
    lua_getfield(L, 1, "n_targets");
    nl = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
  }
  if (!has_labels && !has_targets)
    return luaL_error(L, "gram: need label_offsets/label_neighbors or targets");

  uint64_t um = (uint64_t)m;
  uint64_t unc = (uint64_t)nc;
  uint64_t unl = (uint64_t)nl;

  int64_t tile_size = 1024;
  double *XtX = (double *)calloc(um * um, sizeof(double));
  double *eigenvals = (double *)malloc(um * sizeof(double));
  double *xty = (double *)calloc(um * unl, sizeof(double));
  double *col_mean = (double *)calloc(um, sizeof(double));
  double *y_mean_arr = (double *)malloc(unl * sizeof(double));
  double *tile_buf = (double *)malloc((uint64_t)tile_size * um * sizeof(double));
  if (!XtX || !eigenvals || !xty || !col_mean || !y_mean_arr || !tile_buf) {
    free(XtX); free(eigenvals); free(xty);
    free(col_mean); free(y_mean_arr); free(tile_buf);
    return luaL_error(L, "gram: out of memory");
  }

  tk_dvec_t *lc = NULL;
  int lc_idx = 0;
  if (has_labels) {
    lc = tk_dvec_create(L, unl);
    lc->n = unl;
    lc_idx = lua_gettop(L);
    memset(lc->a, 0, unl * sizeof(double));
    for (uint64_t s = 0; s < unc; s++)
      for (int64_t j = lbl_off->a[s]; j < lbl_off->a[s + 1]; j++)
        lc->a[lbl_nbr->a[j]] += has_lbl_val ? fabs(lbl_val->a[j]) : 1.0;
  }

  for (int64_t base = 0; base < nc; base += tile_size) {
    int64_t bs = (base + tile_size <= nc) ? tile_size : nc - base;
    uint64_t ubs = (uint64_t)bs;
    for (uint64_t j = 0; j < um; j++) {
      double *col = tile_buf + j * ubs;
      float *src = codes_fv->a + (uint64_t)base * um + j;
      for (uint64_t i = 0; i < ubs; i++) {
        double v = (double)src[i * um];
        col[i] = v;
        col_mean[j] += v;
      }
    }
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
      (int)m, (int)bs, 1.0, tile_buf, (int)bs,
      1.0, XtX, (int)m);
    if (has_targets)
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)m, (int)nl, (int)bs, 1.0, tile_buf, (int)bs,
        targets_dv->a + (uint64_t)base * unl, (int)nl,
        1.0, xty, (int)nl);
    if (has_labels) {
      #pragma omp parallel for schedule(static)
      for (int64_t k = 0; k < m; k++) {
        double *col = tile_buf + (uint64_t)k * ubs;
        for (uint64_t i = 0; i < ubs; i++) {
          uint64_t si = (uint64_t)base + i;
          for (int64_t j = lbl_off->a[si]; j < lbl_off->a[si + 1]; j++)
            xty[k * nl + lbl_nbr->a[j]] += has_lbl_val ? col[i] * lbl_val->a[j] : col[i];
        }
      }
    }
  }
  free(tile_buf);

  for (uint64_t j = 0; j < um; j++)
    col_mean[j] /= (double)nc;

  if (has_labels) {
    if (has_lbl_val) {
      memset(y_mean_arr, 0, unl * sizeof(double));
      for (uint64_t s = 0; s < unc; s++)
        for (int64_t j = lbl_off->a[s]; j < lbl_off->a[s + 1]; j++)
          y_mean_arr[lbl_nbr->a[j]] += lbl_val->a[j];
      for (int64_t l = 0; l < nl; l++)
        y_mean_arr[l] /= (double)nc;
    } else {
      for (int64_t l = 0; l < nl; l++)
        y_mean_arr[l] = lc->a[l] / (double)nc;
    }
  } else {
    for (int64_t l = 0; l < nl; l++) {
      double s = 0.0;
      for (uint64_t i = 0; i < unc; i++)
        s += targets_dv->a[i * unl + (uint64_t)l];
      y_mean_arr[l] = s / (double)nc;
    }
  }

  return tk_gram_finalize(L, XtX, xty, col_mean, y_mean_arr,
    eigenvals, lc, lc_idx, nc, m, nl);
}

static inline int tk_ridge_create_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "gram");
  if (!lua_isnil(L, -1)) {
    tk_gram_t *gram = tk_gram_peek(L, -1);
    int64_t d = gram->n_dims, nl = gram->n_labels;
    uint64_t dnl = (uint64_t)d * (uint64_t)nl;
    lua_getfield(L, 1, "lambda");
    double lambda_raw = (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.0);
    lua_pop(L, 1);
    lua_getfield(L, 1, "propensity_a");
    bool do_prop = lua_isnumber(L, -1);
    double prop_a = do_prop ? lua_tonumber(L, -1) : 0.0;
    lua_pop(L, 1);
    lua_getfield(L, 1, "propensity_b");
    double prop_b = do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
    lua_pop(L, 1);
    if (do_prop && !gram->label_counts)
      return luaL_error(L, "ridge create: propensity requires label_counts");
    tk_gram_solve_w(gram, lambda_raw, do_prop, prop_a, prop_b);
    double *W_d = (double *)malloc(dnl * sizeof(double));
    if (!W_d) return luaL_error(L, "ridge create: out of memory");
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)d, (int)nl, (int)d, 1.0, gram->evecs, (int)d,
      gram->W_work, (int)nl, 0.0, W_d, (int)nl);
    lua_getfield(L, 1, "w_buf");
    tk_fvec_t *w_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "w_buf");
    int w_buf_lua_idx = w_buf ? lua_gettop(L) : 0;
    if (!w_buf) lua_pop(L, 1);
    tk_fvec_t *W_fvec;
    int W_idx;
    if (w_buf) {
      tk_fvec_ensure(w_buf, dnl);
      w_buf->n = dnl;
      W_fvec = w_buf;
      W_idx = w_buf_lua_idx;
    } else {
      W_fvec = tk_fvec_create(L, dnl);
      W_idx = lua_gettop(L);
    }
    tk_dvec_t *intercept_dv = NULL;
    int intercept_idx = 0;
    if (gram->col_mean && gram->y_mean) {
      intercept_dv = tk_dvec_create(L, (uint64_t)nl);
      intercept_dv->n = (uint64_t)nl;
      intercept_idx = lua_gettop(L);
      if (do_prop && gram->label_counts) {
        double C = (log((double)gram->n_samples) - 1.0) * pow(prop_b + 1.0, prop_a);
        double *lc = gram->label_counts->a;
        for (int64_t l = 0; l < nl; l++)
          intercept_dv->a[l] = (1.0 + C / pow(lc[l] + prop_b, prop_a)) * gram->y_mean[l];
      } else {
        memcpy(intercept_dv->a, gram->y_mean, (uint64_t)nl * sizeof(double));
      }
      cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)nl,
        -1.0, W_d, (int)nl, gram->col_mean, 1, 1.0, intercept_dv->a, 1);
    }
    for (uint64_t i = 0; i < dnl; i++)
      W_fvec->a[i] = (float)W_d[i];
    free(W_d);

    tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
      TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
    int Ei = lua_gettop(L);
    r->W = W_fvec;
    r->intercept = intercept_dv;

    r->n_dims = d;
    r->n_labels = nl;
    r->Wt = NULL;
    r->sbuf = NULL;
    r->sbuf_size = 0;
    r->heap_buf = NULL;
    r->heap_buf_size = 0;
    r->destroyed = false;
    lua_newtable(L);
    lua_pushvalue(L, W_idx);
    lua_setfield(L, -2, "W");
    if (intercept_idx > 0) {
      lua_pushvalue(L, intercept_idx);
      lua_setfield(L, -2, "intercept");
    }
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
  if (version != 3) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported ridge version %d", (int)version);
  }
  int64_t n_dims, n_labels;
  tk_lua_fread(L, &n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_labels, sizeof(int64_t), 1, fh);
  tk_fvec_t *W = tk_fvec_load(L, fh);
  int W_idx = lua_gettop(L);
  tk_dvec_t *intercept = NULL;
  int b_idx = 0;
  uint8_t has_intercept;
  tk_lua_fread(L, &has_intercept, sizeof(uint8_t), 1, fh);
  if (has_intercept) {
    intercept = tk_dvec_load(L, fh);
    b_idx = lua_gettop(L);
  }
  tk_lua_fclose(L, fh);
  tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
    TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
  int Ei = lua_gettop(L);
  r->W = W;
  r->intercept = intercept;
  r->n_dims = n_dims;
  r->n_labels = n_labels;
  r->Wt = NULL;
  r->sbuf = NULL;
  r->sbuf_size = 0;
  r->heap_buf = NULL;
  r->heap_buf_size = 0;
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
  { "gram", tk_ridge_gram_lua },
  { "load", tk_ridge_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_ridge (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_ridge_fns, 0);
  return 1;
}
