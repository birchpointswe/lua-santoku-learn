#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <santoku/learn/mathlibs.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/rvec.h>
#include <santoku/csr.h>
#include <santoku/mtx.h>
#include <santoku/learn/gram.h>

#define TK_RIDGE_MT "tk_ridge_t"

typedef struct {
  tk_fvec_t *W;
  tk_dvec_t *intercept;
  int64_t n_dims;
  int64_t n_labels;
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
  free(r->sbuf);
  free(r->heap_buf);
  r->W = NULL;
  r->intercept = NULL;
  r->sbuf = NULL;
  r->heap_buf = NULL;
  r->destroyed = true;
  return 0;
}

// label(M, k) -> P csr (neighbors = top-k label ids, values = scores), dense scoring.
static inline int tk_ridge_encode_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  tk_mtx_t *Mx = tk_mtx_peek(L, 2, "codes");
  int64_t nl = r->n_labels, d = r->n_dims;
  float *codes_a = ((tk_fvec_t *) Mx->v)->a;
  int64_t n = (int64_t) Mx->n_rows;
  int64_t k = (int64_t) luaL_checkinteger(L, 3);
  if (k > nl) k = nl;
  if (k < 1) k = 1;
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n + 1)); offsets->n = (uint64_t)(n + 1);
  int offsets_idx = lua_gettop(L);
  tk_ivec_t *labels = tk_ivec_create(L, (uint64_t)(n * k)); labels->n = (uint64_t)(n * k);
  int labels_idx = lua_gettop(L);
  tk_fvec_t *scores_out = tk_fvec_create(L, (uint64_t)(n * k)); scores_out->n = (uint64_t)(n * k);
  int scores_out_idx = lua_gettop(L);
  for (int64_t i = 0; i <= n; i++) offsets->a[i] = i * k;
  int nt = omp_get_max_threads();
  uint64_t heap_need = (uint64_t)nt * (uint64_t)k;
  if (!r->heap_buf || r->heap_buf_size < heap_need) {
    free(r->heap_buf);
    r->heap_buf = (tk_rank_t *)malloc(heap_need * sizeof(tk_rank_t));
    r->heap_buf_size = heap_need;
  }
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
  tk_csr_push(L, TK_TAG_F32, TK_TAG_I64, (uint64_t) nl,
    offsets_idx, offsets, labels_idx, (void *) labels, scores_out_idx, scores_out);
  return 1;
}

static inline int tk_ridge_persist_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKri", 1, 4, fh);
  uint8_t version = 4;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &r->n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &r->n_labels, sizeof(int64_t), 1, fh);
  uint8_t w_external = (r->W->lua_managed == 2) ? 1 : 0;
  tk_lua_fwrite(L, &w_external, sizeof(uint8_t), 1, fh);
  if (w_external) {
#if !defined(__EMSCRIPTEN__)
    msync(r->W->a, r->W->n * sizeof(float), MS_SYNC);
#endif
  } else {
    tk_fvec_persist(L, r->W, fh);
  }
  uint8_t has_intercept = r->intercept ? 1 : 0;
  tk_lua_fwrite(L, &has_intercept, sizeof(uint8_t), 1, fh);
  if (r->intercept)
    tk_dvec_persist(L, r->intercept, fh);
  tk_lua_fclose(L, fh);
  return 0;
}

// regress(M[, buf]) -> fvec (n_samples x n_labels), dense scoring.
static inline int tk_ridge_transform_lua (lua_State *L) {
  tk_ridge_t *r = tk_ridge_peek(L, 1);
  int64_t d = r->n_dims, nl = r->n_labels;
  tk_mtx_t *Mx = tk_mtx_peek(L, 2, "codes");
  float *codes_a = ((tk_fvec_t *) Mx->v)->a;
  int64_t n = (int64_t) Mx->n_rows;
  tk_fvec_t *buf = (lua_gettop(L) >= 3 && !lua_isnil(L, 3)) ? tk_fvec_peek(L, 3, "buf") : NULL;
  tk_fvec_t *out;
  if (buf) { tk_fvec_ensure(buf, (uint64_t)(n * nl)); buf->n = (uint64_t)(n * nl); out = buf; lua_pushvalue(L, 3); }
  else { out = tk_fvec_create(L, (uint64_t)(n * nl)); out->n = (uint64_t)(n * nl); }
  int out_idx = lua_gettop(L);
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
  // x: dense codes mtx (F32). y: labels csr (offsets/neighbors). targets: dvec for regression.
  lua_getfield(L, 1, "x");
  tk_mtx_t *Xm = tk_mtx_peek(L, -1, "x");
  tk_fvec_t *codes_fv = (tk_fvec_t *) Xm->v;
  int64_t nc = (int64_t) Xm->n_rows;
  int64_t m = (int64_t) Xm->n_cols;
  lua_pop(L, 1);

  lua_getfield(L, 1, "y");
  int has_labels = !lua_isnil(L, -1);
  tk_ivec_t *lbl_off = NULL, *lbl_nbr = NULL;
  int64_t nl = 0;
  if (has_labels) {
    tk_csr_t *Y = tk_csr_peek(L, -1, "y");
    lbl_off = Y->offsets;
    lbl_nbr = (tk_ivec_t *) Y->neighbors;
    nl = (int64_t) Y->n_cols;
  }
  lua_pop(L, 1);
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
    return luaL_error(L, "gram: need y (labels csr) or targets");

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
        lc->a[lbl_nbr->a[j]] += 1.0;
  }

  for (int64_t base = 0; base < nc; base += tile_size) {
    int64_t bs = (base + tile_size <= nc) ? tile_size : nc - base;
    uint64_t ubs = (uint64_t)bs;
    {
      float *tile_src = codes_fv->a + (uint64_t)base * um;
      for (uint64_t i = 0; i < ubs; i++) {
        float *row = tile_src + i * um;
        for (uint64_t j = 0; j < um; j++) {
          double v = (double)row[j];
          tile_buf[j * ubs + i] = v;
          col_mean[j] += v;
        }
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
            xty[k * nl + lbl_nbr->a[j]] += col[i];
        }
      }
    }
  }
  free(tile_buf);

  for (uint64_t j = 0; j < um; j++)
    col_mean[j] /= (double)nc;

  if (has_labels) {
    for (int64_t l = 0; l < nl; l++)
      y_mean_arr[l] = lc->a[l] / (double)nc;
  } else {
    for (int64_t l = 0; l < nl; l++) {
      double s = 0.0;
      for (uint64_t i = 0; i < unc; i++)
        s += targets_dv->a[i * unl + (uint64_t)l];
      y_mean_arr[l] = s / (double)nc;
    }
  }

  lua_getfield(L, 1, "solve_lambda");
  int do_solve = lua_isnumber(L, -1);
  double solve_lambda = do_solve ? lua_tonumber(L, -1) : 0.0;
  lua_pop(L, 1);
  if (do_solve) {
    lua_getfield(L, 1, "solve_propensity_a");
    bool do_prop = lua_isnumber(L, -1);
    double prop_a = do_prop ? lua_tonumber(L, -1) : 0.0;
    lua_pop(L, 1);
    lua_getfield(L, 1, "solve_propensity_b");
    double prop_b = do_prop ? (lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 1.5) : 0.0;
    lua_pop(L, 1);
    free(eigenvals);
    return tk_gram_finalize_cholesky(L, XtX, xty, col_mean, y_mean_arr,
      lc, lc_idx, nc, m, nl, solve_lambda, do_prop, prop_a, prop_b);
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
    if (gram->baked) {
      // W already solved at the fixed lambda/propensity; ignore lambda/prop/tile_labels.
      lua_getfield(L, 1, "w_buf");
      tk_fvec_t *w_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "w_buf");
      int w_buf_idx = w_buf ? lua_gettop(L) : 0;
      if (!w_buf) lua_pop(L, 1);
      tk_fvec_t *W_fvec;
      int W_idx;
      if (w_buf) {
        tk_fvec_ensure(w_buf, dnl);
        w_buf->n = dnl;
        W_fvec = w_buf;
        W_idx = w_buf_idx;
      } else {
        W_fvec = tk_fvec_create(L, dnl);
        W_idx = lua_gettop(L);
      }
      memcpy(W_fvec->a, gram->W_baked_f, dnl * sizeof(float));
      tk_dvec_t *intercept_dv = NULL;
      int intercept_idx = 0;
      if (gram->intercept) {
        intercept_dv = tk_dvec_create(L, (uint64_t)nl);
        intercept_dv->n = (uint64_t)nl;
        memcpy(intercept_dv->a, gram->intercept, (uint64_t)nl * sizeof(double));
        intercept_idx = lua_gettop(L);
      }
      tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
        TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
      int Ei = lua_gettop(L);
      r->W = W_fvec;
      r->intercept = intercept_dv;
      r->n_dims = d;
      r->n_labels = nl;
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
      return 1;
    }
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
    lua_getfield(L, 1, "w_buf");
    tk_fvec_t *w_buf = lua_isnil(L, -1) ? NULL : tk_fvec_peek(L, -1, "w_buf");
    int w_buf_lua_idx = w_buf ? lua_gettop(L) : 0;
    if (!w_buf) lua_pop(L, 1);
    lua_getfield(L, 1, "tile_labels");
    int64_t tile_labels = lua_isnumber(L, -1) ? (int64_t)lua_tointeger(L, -1) : 0;
    lua_pop(L, 1);
    if (tile_labels <= 0) tile_labels = gram->tile_labels;
    tk_fvec_t *W_fvec;
    int W_idx;
    tk_dvec_t *intercept_dv = NULL;
    int intercept_idx = 0;
    if (tile_labels <= 0) tile_labels = 1024;
    if (w_buf) {
      tk_fvec_ensure(w_buf, dnl);
      w_buf->n = dnl;
      W_fvec = w_buf;
      W_idx = w_buf_lua_idx;
    } else {
      W_fvec = tk_fvec_create(L, dnl);
      W_idx = lua_gettop(L);
    }
    if (gram->col_mean && gram->y_mean) {
      intercept_dv = tk_dvec_create(L, (uint64_t)nl);
      intercept_dv->n = (uint64_t)nl;
      intercept_idx = lua_gettop(L);
    }
    double mu = lambda_raw * gram->mean_eig + gram->mean_eig * 1e-7;
    double C = 0.0;
    double *lc = NULL;
    if (do_prop && gram->label_counts) {
      C = (log((double)gram->n_samples) - 1.0) * pow(prop_b + 1.0, prop_a);
      lc = gram->label_counts->a;
    }
    int64_t B = tile_labels < nl ? tile_labels : nl;   // a tile is never wider than n_labels
    uint64_t tsz = (uint64_t)d * (uint64_t)B;
    // Float-eigen grams (tk_gram_finalize_f_native) store only evecs_f -- no double
    // evecs/PQtY. Two specialized paths: the double path is the spectral/M.krr original; the float path
    // keeps everything in float (sgemm/sgemv, evecs_f), writing W_fvec directly and forming the intercept
    // from a float col_mean -- no double scratch, no widen round-trips.
    bool fgram = (gram->evecs == NULL);
    double *W_tile = NULL, *W_d_tile = NULL;
    float *W_tile_f = NULL, *W_d_tile_f = NULL, *cm_f = NULL, *ic_f = NULL;
    if (fgram) {
      W_tile_f = (float *)malloc(tsz * sizeof(float));
      W_d_tile_f = (float *)malloc(tsz * sizeof(float));
      if (intercept_dv) {
        cm_f = (float *)malloc((uint64_t)d * sizeof(float));
        ic_f = (float *)malloc((uint64_t)B * sizeof(float));
        if (cm_f) for (int64_t i = 0; i < d; i++) cm_f[i] = (float)gram->col_mean[i];
      }
      if (!W_tile_f || !W_d_tile_f || (intercept_dv && (!cm_f || !ic_f))) {
        free(W_tile_f); free(W_d_tile_f); free(cm_f); free(ic_f);
        return luaL_error(L, "ridge create tiled: out of memory");
      }
    } else {
      W_tile = (double *)malloc(tsz * sizeof(double));
      W_d_tile = (double *)malloc(tsz * sizeof(double));
      if (!W_tile || !W_d_tile) {
        free(W_tile); free(W_d_tile);
        return luaL_error(L, "ridge create tiled: out of memory");
      }
    }
    for (int64_t tl_start = 0; tl_start < nl; tl_start += B) {
      int64_t aB = (tl_start + B <= nl) ? B : nl - tl_start;
      if (fgram) {
        for (int64_t i = 0; i < d; i++) {
          double inv = 1.0 / (gram->eigenvals[i] + mu);
          for (int64_t l = 0; l < aB; l++) {
            double prop = (lc) ? (1.0 + C / pow(lc[tl_start + l] + prop_b, prop_a)) : 1.0;
            W_tile_f[i * aB + l] = (float)((double)gram->PQtY_f[i * nl + tl_start + l] * prop * inv);
          }
        }
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          (int)d, (int)aB, (int)d, 1.0f, gram->evecs_f, (int)d,
          W_tile_f, (int)aB, 0.0f, W_d_tile_f, (int)aB);
        for (int64_t i = 0; i < d; i++)
          for (int64_t l = 0; l < aB; l++)
            W_fvec->a[i * nl + tl_start + l] = W_d_tile_f[i * aB + l];
        if (intercept_dv) {
          cblas_sgemv(CblasRowMajor, CblasTrans, (int)d, (int)aB,
            -1.0f, W_d_tile_f, (int)aB, cm_f, 1, 0.0f, ic_f, 1);
          for (int64_t l = 0; l < aB; l++) {
            double prop = (lc) ? (1.0 + C / pow(lc[tl_start + l] + prop_b, prop_a)) : 1.0;
            intercept_dv->a[tl_start + l] = prop * gram->y_mean[tl_start + l] + (double)ic_f[l];
          }
        }
      } else {
        for (int64_t i = 0; i < d; i++) {
          double inv = 1.0 / (gram->eigenvals[i] + mu);
          for (int64_t l = 0; l < aB; l++) {
            double pqty = (double)gram->PQtY_f[i * nl + tl_start + l];
            double prop = (lc) ? (1.0 + C / pow(lc[tl_start + l] + prop_b, prop_a)) : 1.0;
            W_tile[i * aB + l] = pqty * prop * inv;
          }
        }
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          (int)d, (int)aB, (int)d, 1.0, gram->evecs, (int)d,
          W_tile, (int)aB, 0.0, W_d_tile, (int)aB);
        for (int64_t i = 0; i < d; i++)
          for (int64_t l = 0; l < aB; l++)
            W_fvec->a[i * nl + tl_start + l] = (float)W_d_tile[i * aB + l];
        if (intercept_dv) {
          for (int64_t l = 0; l < aB; l++) {
            double prop = (lc) ? (1.0 + C / pow(lc[tl_start + l] + prop_b, prop_a)) : 1.0;
            intercept_dv->a[tl_start + l] = prop * gram->y_mean[tl_start + l];
          }
          cblas_dgemv(CblasRowMajor, CblasTrans, (int)d, (int)aB,
            -1.0, W_d_tile, (int)aB, gram->col_mean, 1, 1.0, intercept_dv->a + tl_start, 1);
        }
      }
    }
    free(W_tile); free(W_d_tile); free(W_tile_f); free(W_d_tile_f); free(cm_f); free(ic_f);

    tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
      TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
    int Ei = lua_gettop(L);
    r->W = W_fvec;
    r->intercept = intercept_dv;

    r->n_dims = d;
    r->n_labels = nl;
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
  lua_pop(L, 1);
  lua_getfield(L, 1, "W");
  if (!lua_isnil(L, -1)) {
    tk_fvec_t *W_fvec = tk_fvec_peek(L, -1, "W");
    int W_idx = lua_gettop(L);
    lua_getfield(L, 1, "n_dims");
    int64_t d = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
    lua_getfield(L, 1, "n_labels");
    int64_t nl = (int64_t)luaL_checkinteger(L, -1);
    lua_pop(L, 1);
    tk_dvec_t *intercept_dv = NULL;
    int intercept_idx = 0;
    lua_getfield(L, 1, "intercept");
    if (!lua_isnil(L, -1)) {
      intercept_dv = tk_dvec_peek(L, -1, "intercept");
      intercept_idx = lua_gettop(L);
    } else {
      lua_pop(L, 1);
    }
    tk_ridge_t *r = tk_lua_newuserdata(L, tk_ridge_t,
      TK_RIDGE_MT, tk_ridge_mt_fns, tk_ridge_gc);
    int Ei = lua_gettop(L);
    r->W = W_fvec;
    r->intercept = intercept_dv;
    r->n_dims = d;
    r->n_labels = nl;
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
    return 1;
  }
  return luaL_error(L, "ridge create: gram or W required");
}

static inline int tk_ridge_load_lua (lua_State *L) {
  const char *data = luaL_checkstring(L, 1);
  FILE *fh = tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKri", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid ridge file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 3 && version != 4) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported ridge version %d", (int)version);
  }
  int64_t n_dims, n_labels;
  tk_lua_fread(L, &n_dims, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_labels, sizeof(int64_t), 1, fh);
  tk_fvec_t *W = NULL;
  int W_idx = 0;
  bool have_arg_w = lua_gettop(L) >= 2 && !lua_isnil(L, 2);
  if (version == 4) {
    uint8_t w_external;
    tk_lua_fread(L, &w_external, sizeof(uint8_t), 1, fh);
    if (w_external) {
      if (!have_arg_w) {
        tk_lua_fclose(L, fh);
        return luaL_error(L, "ridge load: external W requires fvec arg 2");
      }
      W = tk_fvec_peek(L, 2, "W");
      W_idx = 2;
    } else {
      W = tk_fvec_load(L, fh);
      W_idx = lua_gettop(L);
    }
  } else {
    W = tk_fvec_load(L, fh);
    W_idx = lua_gettop(L);
  }
  if (have_arg_w && W_idx != 2) {
    W = tk_fvec_peek(L, 2, "W");
    W_idx = 2;
  }
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
  tk_lua_require_mod(L, "santoku.csr");   // label(M,k) -> P csr; regress/label take mtx codes
  tk_lua_require_mod(L, "santoku.mtx");
  lua_newtable(L);
  tk_lua_register(L, tk_ridge_fns, 0);
  return 1;
}
