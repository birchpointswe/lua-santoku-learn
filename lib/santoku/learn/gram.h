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
    if (LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', (int)d, factor, (int)d) == 0) { ok = 1; break; }
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
  { NULL, NULL }
};

#endif
