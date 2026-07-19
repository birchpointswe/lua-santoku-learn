#ifndef TK_GRAM_H
#define TK_GRAM_H

#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <santoku/learn/mathlibs.h>
#include <santoku/lua/utils.h>
#include <santoku/fvec.h>
#include <santoku/dvec.h>

#define TK_GRAM_MT "tk_gram_t"

typedef struct {
  bool baked;
  bool prepared;
  bool W_baked_external;
  float *W_baked_f;
  double *intercept;
  float *XtX;
  bool XtX_external;
  float *xty;
  bool xty_external;
  float *factor;
  bool factor_external;
  float *factor_ext;
  uint64_t factor_ext_cap;
  float *Bcm;
  float *cm_f;
  double *y_mean;
  double mean_eig;
  int64_t n_dims;
  int64_t n_labels;
  int64_t n_samples;
  int64_t tile_labels;
  bool destroyed;
} tk_gram_t;

__attribute__((unused)) static luaL_Reg tk_gram_mt_fns[];

static inline tk_gram_t *tk_gram_peek (lua_State *L, int i) {
  return (tk_gram_t *)luaL_checkudata(L, i, TK_GRAM_MT);
}

static inline void tk_gram_release_prepared (tk_gram_t *g) {
  if (!g->XtX_external) free(g->XtX);
  g->XtX = NULL;
  if (!g->xty_external) free(g->xty);
  g->xty = NULL;
  if (!g->factor_external) free(g->factor);
  g->factor = NULL;
  g->factor_external = false;
  free(g->Bcm); g->Bcm = NULL;
  free(g->cm_f); g->cm_f = NULL;
  free(g->y_mean); g->y_mean = NULL;
  g->prepared = false;
}

static inline int tk_gram_gc (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  tk_gram_release_prepared(g);
  if (!g->W_baked_external) free(g->W_baked_f);
  g->W_baked_f = NULL;
  free(g->intercept);
  g->intercept = NULL;
  g->destroyed = true;
  return 0;
}

static inline void tk_gram_add_intercept_f (
  float *sbuf, int64_t bs, int64_t nl, double *intercept)
{
  #pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < bs; i++)
    for (int64_t l = 0; l < nl; l++)
      sbuf[i * nl + l] += (float)intercept[l];
}

static inline int tk_gram_make_prepared (
  lua_State *L,
  float *XtX, bool XtX_external,
  float *xty, bool xty_external,
  float *factor_ext, uint64_t factor_ext_cap,
  float *cm_f, double *y_mean, double mean_eig,
  int64_t nc, int64_t d, int64_t nl, int64_t tile_labels)
{
  tk_gram_t *g = tk_lua_newuserdata(L, tk_gram_t,
    TK_GRAM_MT, tk_gram_mt_fns, tk_gram_gc);
  int gram_idx = lua_gettop(L);
  g->baked = false;
  g->prepared = true;
  g->W_baked_external = false;
  g->W_baked_f = NULL;
  g->intercept = NULL;
  g->XtX = XtX;
  g->XtX_external = XtX_external;
  g->xty = xty;
  g->xty_external = xty_external;
  g->factor = NULL;
  g->factor_external = false;
  g->factor_ext = factor_ext;
  g->factor_ext_cap = factor_ext_cap;
  g->Bcm = NULL;
  g->cm_f = cm_f;
  g->y_mean = y_mean;
  g->mean_eig = mean_eig;
  g->n_dims = d;
  g->n_labels = nl;
  g->n_samples = nc;
  g->tile_labels = tile_labels > 0 ? tile_labels : 1024;
  g->destroyed = false;
  lua_newtable(L);
  lua_setfenv(L, gram_idx);
  lua_pushvalue(L, gram_idx);
  return 1;
}

static inline int tk_gram_solve_impl (
  lua_State *L, tk_gram_t *g, double lambda, tk_fvec_t *w_buf)
{
  if (!g->prepared)
    return luaL_error(L, "gram solve: prepared state released");
  uint64_t d = (uint64_t)g->n_dims;
  uint64_t unl = (uint64_t)g->n_labels;
  uint64_t dd = d * d;
  uint64_t dnl = d * unl;
  int64_t nl = g->n_labels;
  if (!g->W_baked_f) {
    if (w_buf && w_buf->m >= dnl) {
      g->W_baked_f = w_buf->a;
      g->W_baked_external = true;
    } else {
      g->W_baked_f = (float *)malloc(dnl * sizeof(float));
      g->W_baked_external = false;
      if (!g->W_baked_f)
        return luaL_error(L, "gram solve: out of memory (W)");
    }
  }
  if (!g->intercept) {
    g->intercept = (double *)malloc(unl * sizeof(double));
    if (!g->intercept)
      return luaL_error(L, "gram solve: out of memory (intercept)");
  }
  if (!g->factor) {
    if (g->factor_ext && g->factor_ext_cap >= dd) {
      g->factor = g->factor_ext;
      g->factor_external = true;
    } else {
      g->factor = (float *)malloc(dd * sizeof(float));
      g->factor_external = false;
      if (!g->factor)
        return luaL_error(L, "gram solve: out of memory (factor)");
    }
  }
  int64_t B = g->tile_labels < nl ? g->tile_labels : nl;
  if (!g->Bcm) {
    g->Bcm = (float *)malloc(d * (uint64_t)B * sizeof(float));
    if (!g->Bcm)
      return luaL_error(L, "gram solve: out of memory (Bcm)");
  }
  float *factor = g->factor;
  const float *XtX = g->XtX;
  double jit = 1e-6;
  int ok = 0;
  for (int attempt = 0; attempt < 8; attempt++) {
    memcpy(factor, XtX, dd * sizeof(float));
    double add = lambda * g->mean_eig + g->mean_eig * jit + jit;
    for (uint64_t i = 0; i < d; i++)
      factor[i * d + i] += (float)add;
    if (LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', (int)d, factor, (int)d) == 0) {
      ok = 1;
      break;
    }
    jit *= 16.0;
  }
  if (!ok)
    return luaL_error(L, "gram solve: spotrf failed (singular after jitter escalation)");
  const float *xty = g->xty;
  float *W = g->W_baked_f;
  float *Bcm = g->Bcm;
  for (int64_t tl_start = 0; tl_start < nl; tl_start += B) {
    int64_t actual_B = (tl_start + B <= nl) ? B : nl - tl_start;
    for (uint64_t k = 0; k < d; k++)
      for (int64_t tl = 0; tl < actual_B; tl++)
        Bcm[(uint64_t)tl * d + k] = xty[k * unl + (uint64_t)(tl_start + tl)];
    LAPACKE_spotrs(LAPACK_COL_MAJOR, 'U', (int)d, (int)actual_B, factor, (int)d, Bcm, (int)d);
    for (uint64_t k = 0; k < d; k++)
      for (int64_t tl = 0; tl < actual_B; tl++)
        W[k * unl + (uint64_t)(tl_start + tl)] = Bcm[(uint64_t)tl * d + k];
    for (int64_t tl = 0; tl < actual_B; tl++)
      g->intercept[tl_start + tl] = g->y_mean[tl_start + tl]
        - (double)cblas_sdot((int)d, Bcm + (uint64_t)tl * d, 1, g->cm_f, 1);
  }
  g->baked = true;
  return 0;
}


/* point this gram's factor scratch at a shared external buffer (fold grams within a
** trial solve sequentially, so one m^2 scratch serves them all) */
static inline int tk_gram_attach_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  tk_fvec_t *fv = tk_fvec_peek(L, 2, "factor");
  if (g->factor_external) {
    g->factor = NULL;
    g->factor_external = false;
  } else if (g->factor) {
    free(g->factor);
    g->factor = NULL;
  }
  g->factor_ext = fv->a;
  g->factor_ext_cap = fv->m;
  lua_getfenv(L, 1);
  lua_pushvalue(L, 2);
  lua_setfield(L, -2, "factor_ext");
  lua_pop(L, 1);
  return 0;
}

/* CV-downdate fold constructor: assemble a fold's prepared gram from the FULL prepared
** gram plus the fold's UNCENTERED val moments (accumulated by the same encode pass).
** In place into the provided slots:
**   Gxx_full = XtX_full + n*cm*cm^T            (undo the full centering)
**   XtX_fold = Gxx_full - Gxx_val - t*cm_t*cm_t^T
**   xty_fold analogous with y_mean_t; cm_t = (n*cm - sv)/t; y_mean_t = (n*ym - tv)/t
** One elementwise pass + two rank-1 updates per matrix; no re-encode, no extra syrk.
** fold(xtx_slot, xty_slot, sv_fvec, tv_dvec, val_count) -> prepared gram */
static inline int tk_gram_fold_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  tk_fvec_t *xtx_slot = tk_fvec_peek(L, 2, "xtx_slot");
  tk_fvec_t *xty_slot = tk_fvec_peek(L, 3, "xty_slot");
  tk_fvec_t *sv = tk_fvec_peek(L, 4, "sv");
  tk_dvec_t *tv = tk_dvec_peek(L, 5, "tv");
  int64_t vcount = luaL_checkinteger(L, 6);
  if (!g->prepared || !g->cm_f || !g->y_mean)
    return luaL_error(L, "gram fold: requires a prepared gram");
  uint64_t d = (uint64_t)g->n_dims;
  int64_t nl = g->n_labels;
  uint64_t unl = (uint64_t)nl;
  int64_t n = g->n_samples;
  int64_t t = n - vcount;
  if (vcount < 1 || t < 1)
    return luaL_error(L, "gram fold: degenerate fold sizes");
  if (xtx_slot->m < d * d || xty_slot->m < d * unl || sv->m < d || tv->m < unl)
    return luaL_error(L, "gram fold: slot too small");
  float *X = xtx_slot->a, *Y = xty_slot->a;
  const float *XF = g->XtX, *YF = g->xty;
  float *cm_t = (float *)malloc(d * sizeof(float));
  double *ym_t = (double *)malloc(unl * sizeof(double));
  float *ymf = (float *)malloc(unl * sizeof(float));
  float *ymtf = (float *)malloc(unl * sizeof(float));
  if (!cm_t || !ym_t || !ymf || !ymtf) {
    free(cm_t); free(ym_t); free(ymf); free(ymtf);
    return luaL_error(L, "gram fold: out of memory");
  }
  for (uint64_t j = 0; j < d; j++)
    cm_t[j] = (float)(((double)n * (double)g->cm_f[j] - (double)sv->a[j]) / (double)t);
  for (int64_t l = 0; l < nl; l++) {
    ym_t[l] = ((double)n * g->y_mean[l] - tv->a[l]) / (double)t;
    ymf[l] = (float)g->y_mean[l];
    ymtf[l] = (float)ym_t[l];
  }
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < d * d; i++)
    X[i] = XF[i] - X[i];
  cblas_ssyr(CblasColMajor, CblasUpper, (int)d, (float)n, g->cm_f, 1, X, (int)d);
  cblas_ssyr(CblasColMajor, CblasUpper, (int)d, -(float)t, cm_t, 1, X, (int)d);
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < d * unl; i++)
    Y[i] = YF[i] - Y[i];
  cblas_sger(CblasRowMajor, (int)d, (int)nl, (float)n, g->cm_f, 1, ymf, 1, Y, (int)nl);
  cblas_sger(CblasRowMajor, (int)d, (int)nl, -(float)t, cm_t, 1, ymtf, 1, Y, (int)nl);
  free(ymf); free(ymtf);
  double mean_eig = 0.0;
  for (uint64_t i = 0; i < d; i++) mean_eig += (double)X[i * d + i];
  mean_eig /= (double)d;
  tk_gram_make_prepared(L, X, true, Y, true,
    NULL, 0,
    cm_t, ym_t, mean_eig,
    t, (int64_t)d, nl, g->tile_labels);
  lua_getfenv(L, -1);
  lua_pushvalue(L, 2);
  lua_setfield(L, -2, "xtx_slot");
  lua_pushvalue(L, 3);
  lua_setfield(L, -2, "xty_slot");
  lua_pop(L, 1);
  return 1;
}

static inline int tk_gram_release_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  tk_gram_release_prepared(g);
  /* drop the m*m factor scratch child (factor_buf / kss_raw) held on the fenv: the prepared
  ** state is gone, so the borrowed factor_ext buffer no longer needs to be pinned here. The
  ** caller-pooled buffer (when reused across trials) is owned upstream; a per-trial fresh one
  ** becomes collectable. XtX/xty/w/slots are left in place (release may precede a re-solve on
  ** the SAME shell only via a fresh prepare, which repopulates the fenv). */
  g->factor_ext = NULL;
  g->factor_ext_cap = 0;
  lua_getfenv(L, 1);
  if (!lua_isnil(L, -1)) {
    lua_pushnil(L);
    lua_setfield(L, -2, "factor_buf");
  }
  lua_pop(L, 1);
  return 0;
}

/* explicit early teardown of a gram shell (search pooling: fold shells + the per-trial full
** gram are dead once eval_folds finishes). Frees every C-owned buffer O(1) -- external XtX/
** xty/factor slots are left to their owners -- and clears the fenv so no child survives to a
** GC heap walk. Idempotent via destroyed. */
static inline int tk_gram_destroy_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  if (!g->destroyed) {
    tk_gram_release_prepared(g);
    if (!g->W_baked_external) free(g->W_baked_f);
    g->W_baked_f = NULL;
    free(g->intercept);
    g->intercept = NULL;
    g->factor_ext = NULL;
    g->factor_ext_cap = 0;
    g->destroyed = true;
  }
  lua_newtable(L);
  lua_setfenv(L, 1);
  return 0;
}

static inline int tk_gram_solve_lua (lua_State *L) {
  tk_gram_t *g = tk_gram_peek(L, 1);
  double lambda = luaL_checknumber(L, 2);
  tk_fvec_t *w_buf = (lua_gettop(L) >= 3 && !lua_isnil(L, 3))
    ? tk_fvec_peek(L, 3, "w_buf") : NULL;
  bool had_w = g->W_baked_f != NULL;
  tk_gram_solve_impl(L, g, lambda, w_buf);
  if (!had_w && g->W_baked_external && w_buf) {
    lua_getfenv(L, 1);
    lua_pushvalue(L, 3);
    lua_setfield(L, -2, "w_buf");
    lua_pop(L, 1);
  }
  lua_settop(L, 1);
  return 1;
}

__attribute__((unused)) static luaL_Reg tk_gram_mt_fns[] = {
  { "solve", tk_gram_solve_lua },
  { "attach", tk_gram_attach_lua },
  { "fold", tk_gram_fold_lua },
  { "release", tk_gram_release_lua },
  { "destroy", tk_gram_destroy_lua },
  { NULL, NULL }
};

#endif
