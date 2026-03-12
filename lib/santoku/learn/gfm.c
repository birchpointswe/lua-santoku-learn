#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <string.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/rvec.h>

#define TK_GFM_MT "tk_gfm_t"

typedef struct {
  int64_t *exp_off;
  int64_t *exp_nbr;
  int64_t ns;
  int64_t nl;
  int64_t total_relevant;
  double *thresholds;
  int64_t *pl_off;
  int64_t *pl_count;
  uint8_t *pl_hit;
  double *pl_score;
  uint64_t pl_data_cap;
  uint8_t *gt_bm;
  uint64_t gt_bm_cap;
  tk_rank_t *sort_buf;
  uint64_t sort_buf_cap;
  bool destroyed;
} tk_gfm_t;

static inline tk_gfm_t *tk_gfm_peek (lua_State *L, int i) {
  return (tk_gfm_t *)luaL_checkudata(L, i, TK_GFM_MT);
}

static inline int tk_gfm_gc (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  free(g->exp_off);
  free(g->exp_nbr);
  free(g->thresholds);
  free(g->pl_off);
  free(g->pl_count);
  free(g->pl_hit);
  free(g->pl_score);
  free(g->gt_bm);
  free(g->sort_buf);
  g->destroyed = true;
  return 0;
}

static luaL_Reg tk_gfm_mt_fns[];

static int tk_gfm_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *eo = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *en = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 2);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 1, "gfm.create", "n_samples");
  int64_t nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "gfm.create", "n_labels");
  int64_t total_relevant = eo->a[ns] - eo->a[0];
  int64_t *exp_off = (int64_t *)malloc((uint64_t)(ns + 1) * sizeof(int64_t));
  memcpy(exp_off, eo->a, (uint64_t)(ns + 1) * sizeof(int64_t));
  int64_t *exp_nbr = NULL;
  if (en->n > 0) {
    exp_nbr = (int64_t *)malloc(en->n * sizeof(int64_t));
    memcpy(exp_nbr, en->a, en->n * sizeof(int64_t));
  }
  tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
    TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
  g->exp_off = exp_off;
  g->exp_nbr = exp_nbr;
  g->ns = ns;
  g->nl = nl;
  g->total_relevant = total_relevant;
  g->thresholds = (double *)malloc((uint64_t)nl * sizeof(double));
  for (int64_t i = 0; i < nl; i++) g->thresholds[i] = 1e30;
  g->pl_off = (int64_t *)malloc((uint64_t)(nl + 1) * sizeof(int64_t));
  g->pl_count = (int64_t *)malloc((uint64_t)nl * sizeof(int64_t));
  g->pl_hit = NULL;
  g->pl_score = NULL;
  g->pl_data_cap = 0;
  g->gt_bm = NULL;
  g->gt_bm_cap = 0;
  g->sort_buf = NULL;
  g->sort_buf_cap = 0;
  g->destroyed = false;
  return 1;
}

static int tk_gfm_fit_lua (lua_State *L)
{
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  if (!g->exp_off)
    return luaL_error(L, "fit: no expected data (loaded from file?)");
  luaL_checktype(L, 2, LUA_TTABLE);
  lua_getfield(L, 2, "pred_offsets");
  tk_ivec_t *po = tk_ivec_peek(L, -1, "pred_offsets");
  lua_getfield(L, 2, "pred_neighbors");
  tk_ivec_t *pn = tk_ivec_peek(L, -1, "pred_neighbors");
  lua_getfield(L, 2, "pred_scores");
  tk_fvec_t *psf = tk_fvec_peekopt(L, -1);
  tk_dvec_t *psd = psf ? NULL : tk_dvec_peek(L, -1, "pred_scores");
  lua_pop(L, 3);
  int64_t ns = g->ns, nl = g->nl;
  int64_t *exp_off = g->exp_off, *exp_nbr = g->exp_nbr;
  int64_t total_entries = po->a[ns] - po->a[0];
  if (total_entries <= 0) {
    for (int64_t l = 0; l < nl; l++) g->thresholds[l] = 1e30;
    return 0;
  }
  uint64_t te = (uint64_t)total_entries;
  if (te > g->pl_data_cap) {
    free(g->pl_hit); free(g->pl_score);
    g->pl_hit = (uint8_t *)malloc(te * sizeof(uint8_t));
    g->pl_score = (double *)malloc(te * sizeof(double));
    g->pl_data_cap = te;
  }
  uint64_t bm_need = (uint64_t)nl;
  if (bm_need > g->gt_bm_cap) {
    free(g->gt_bm);
    g->gt_bm = (uint8_t *)calloc(bm_need, sizeof(uint8_t));
    g->gt_bm_cap = bm_need;
  } else {
    memset(g->gt_bm, 0, bm_need * sizeof(uint8_t));
  }
  int64_t *pl_off = g->pl_off, *pl_count = g->pl_count;
  uint8_t *pl_hit = g->pl_hit, *gt_bm = g->gt_bm;
  double *pl_score = g->pl_score;
  memset(pl_count, 0, (uint64_t)nl * sizeof(int64_t));
  for (int64_t i = 0; i < total_entries; i++) {
    int64_t l = pn->a[i];
    if (l >= 0 && l < nl) pl_count[l]++;
  }
  pl_off[0] = 0;
  for (int64_t l = 0; l < nl; l++)
    pl_off[l + 1] = pl_off[l] + pl_count[l];
  memset(pl_count, 0, (uint64_t)nl * sizeof(int64_t));
  for (int64_t s = 0; s < ns; s++) {
    for (int64_t j = exp_off[s]; j < exp_off[s + 1]; j++)
      if (exp_nbr[j] >= 0 && exp_nbr[j] < nl) gt_bm[exp_nbr[j]] = 1;
    for (int64_t j = po->a[s]; j < po->a[s + 1]; j++) {
      int64_t l = pn->a[j];
      if (l >= 0 && l < nl) {
        int64_t pos = pl_off[l] + pl_count[l];
        pl_score[pos] = psf ? (double)psf->a[j] : psd->a[j];
        pl_hit[pos] = gt_bm[l];
        pl_count[l]++;
      }
    }
    for (int64_t j = exp_off[s]; j < exp_off[s + 1]; j++)
      if (exp_nbr[j] >= 0 && exp_nbr[j] < nl) gt_bm[exp_nbr[j]] = 0;
  }
  int64_t max_per_label = 0;
  for (int64_t l = 0; l < nl; l++) {
    int64_t cnt = pl_off[l + 1] - pl_off[l];
    if (cnt > max_per_label) max_per_label = cnt;
  }
  if ((uint64_t)max_per_label > g->sort_buf_cap) {
    free(g->sort_buf);
    g->sort_buf = (tk_rank_t *)malloc((uint64_t)max_per_label * sizeof(tk_rank_t));
    g->sort_buf_cap = (uint64_t)max_per_label;
  }
  tk_rank_t *rbuf = g->sort_buf;
  for (int64_t l = 0; l < nl; l++) {
    int64_t lo = pl_off[l], cnt = pl_off[l + 1] - lo;
    if (cnt <= 1) continue;
    for (int64_t i = 0; i < cnt; i++)
      rbuf[i] = tk_rank(pl_hit[lo + i], pl_score[lo + i]);
    tk_rvec_t rv = { .n = (size_t)cnt, .m = (size_t)cnt, .a = rbuf, .lua_managed = false };
    tk_rvec_desc(&rv, 0, (size_t)cnt);
    for (int64_t i = 0; i < cnt; i++) {
      pl_hit[lo + i] = (uint8_t)rbuf[i].i;
      pl_score[lo + i] = rbuf[i].d;
    }
  }
  double *thresholds = g->thresholds;
  int64_t total_relevant = g->total_relevant;
  int64_t tot_tp = 0, tot_pred = 0;
  for (int64_t l = 0; l < nl; l++) thresholds[l] = 1e30;
  for (int iter = 0; iter < 10; iter++) {
    for (int64_t l = 0; l < nl; l++) {
      int64_t lo = pl_off[l], cnt = pl_off[l + 1] - lo;
      if (cnt == 0) continue;
      int64_t cur_tp = 0, cur_pred = 0;
      for (int64_t i = 0; i < cnt; i++) {
        if (pl_score[lo + i] >= thresholds[l]) {
          cur_pred++;
          if (pl_hit[lo + i]) cur_tp++;
        }
      }
      tot_tp -= cur_tp;
      tot_pred -= cur_pred;
      int64_t sw_tp = 0, sw_pred = 0;
      double best_f1 = (tot_pred + total_relevant > 0)
        ? 2.0 * (double)tot_tp / (double)(tot_pred + total_relevant) : 0.0;
      int64_t best_tp = 0, best_pred = 0;
      double best_thresh = 1e30;
      for (int64_t j = 0; j < cnt; j++) {
        sw_pred++;
        if (pl_hit[lo + j]) sw_tp++;
        if (j == cnt - 1 || pl_score[lo + j] != pl_score[lo + j + 1]) {
          double denom = (double)(tot_pred + sw_pred + total_relevant);
          double f1 = denom > 0 ? 2.0 * (double)(tot_tp + sw_tp) / denom : 0.0;
          if (f1 > best_f1) {
            best_f1 = f1;
            best_tp = sw_tp;
            best_pred = sw_pred;
            best_thresh = pl_score[lo + j];
          }
        }
      }
      thresholds[l] = best_thresh;
      tot_tp += best_tp;
      tot_pred += best_pred;
    }
  }
  return 0;
}

static int tk_gfm_predict_lua (lua_State *L)
{
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  lua_getfield(L, 2, "offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
  lua_getfield(L, 2, "neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "neighbors");
  lua_getfield(L, 2, "scores");
  tk_fvec_t *sf = tk_fvec_peekopt(L, -1);
  tk_dvec_t *sd = sf ? NULL : tk_dvec_peek(L, -1, "scores");
  lua_pop(L, 3);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "gfm.predict", "n_samples");
  double *thresholds = g->thresholds;
  int64_t nl = g->nl;
  tk_ivec_t *ks = tk_ivec_create(L, (uint64_t)ns, NULL, NULL);
  for (int64_t s = 0; s < ns; s++) {
    int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
    int64_t accepted = 0;
    for (int64_t j = ps; j < pe; j++) {
      int64_t l = neighbors->a[j];
      double sc = sf ? (double)sf->a[j] : sd->a[j];
      if (l >= 0 && l < nl && sc >= thresholds[l]) {
        neighbors->a[ps + accepted] = l;
        accepted++;
      }
    }
    ks->a[s] = accepted;
  }
  return 1;
}

static int tk_gfm_persist_lua (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting filepath or true");
  tk_lua_fwrite(L, "TKgf", 1, 4, fh);
  uint8_t version = 7;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &g->nl, sizeof(int64_t), 1, fh);
  if (g->nl > 0)
    tk_lua_fwrite(L, g->thresholds, sizeof(double), (uint64_t)g->nl, fh);
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

static int tk_gfm_load_lua (lua_State *L)
{
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKgf", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid gfm file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 7) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported gfm version %d", (int)version);
  }
  int64_t nl;
  tk_lua_fread(L, &nl, sizeof(int64_t), 1, fh);
  double *thresholds = (double *)malloc((uint64_t)nl * sizeof(double));
  if (nl > 0)
    tk_lua_fread(L, thresholds, sizeof(double), (uint64_t)nl, fh);
  tk_lua_fclose(L, fh);
  tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
    TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
  g->exp_off = NULL;
  g->exp_nbr = NULL;
  g->ns = 0;
  g->nl = nl;
  g->total_relevant = 0;
  g->thresholds = thresholds;
  g->pl_off = NULL;
  g->pl_count = NULL;
  g->pl_hit = NULL;
  g->pl_score = NULL;
  g->pl_data_cap = 0;
  g->gt_bm = NULL;
  g->gt_bm_cap = 0;
  g->sort_buf = NULL;
  g->sort_buf_cap = 0;
  g->destroyed = false;
  return 1;
}

static luaL_Reg tk_gfm_mt_fns[] = {
  { "fit", tk_gfm_fit_lua },
  { "predict", tk_gfm_predict_lua },
  { "persist", tk_gfm_persist_lua },
  { NULL, NULL }
};

static luaL_Reg tk_gfm_fns[] = {
  { "create", tk_gfm_create_lua },
  { "load", tk_gfm_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_gfm (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_gfm_fns, 0);
  return 1;
}
