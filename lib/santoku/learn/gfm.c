#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <santoku/learn/mathlibs.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/learn/gfm.h>

typedef struct { double score; uint8_t hit; } tk_entry_t;

static int tk_entry_desc (const void *a, const void *b) {
  double sa = ((const tk_entry_t *)a)->score;
  double sb = ((const tk_entry_t *)b)->score;
  return (sa > sb) ? -1 : (sa < sb) ? 1 : 0;
}

static inline int tk_gfm_gc (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  if (!g->destroyed) free(g->thresholds);
  g->thresholds = NULL;
  g->destroyed = true;
  return 0;
}

static luaL_Reg tk_gfm_mt_fns[];

static int tk_gfm_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "gfm.create", "n_labels");
  tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
    TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
  g->nl = nl;
  g->thresholds = (double *)malloc((uint64_t)nl * sizeof(double));
  for (int64_t l = 0; l < nl; l++) g->thresholds[l] = HUGE_VAL;
  g->destroyed = false;
  return 1;
}

static int tk_gfm_calibrate_lua (lua_State *L)
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
  lua_getfield(L, 2, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 2, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 5);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "gfm.calibrate", "n_samples");
  int64_t nl = g->nl;

  int64_t total_entries = offsets->a[ns] - offsets->a[0];
  uint64_t total_expected = (uint64_t)(exp_off->a[ns] - exp_off->a[0]);

  if (total_entries <= 0 || total_expected == 0) {
    for (int64_t l = 0; l < nl; l++) g->thresholds[l] = HUGE_VAL;
    lua_pushnumber(L, 0.0);
    return 1;
  }

  uint64_t te = (uint64_t)total_entries;
  tk_entry_t *pool = (tk_entry_t *)malloc(te * sizeof(tk_entry_t));
  uint8_t *bm = (uint8_t *)calloc((uint64_t)nl, sizeof(uint8_t));
  int64_t pi = 0;
  for (int64_t s = 0; s < ns; s++) {
    for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++)
      if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 1;
    for (int64_t j = offsets->a[s]; j < offsets->a[s + 1]; j++) {
      int64_t lbl = neighbors->a[j];
      pool[pi].score = sf ? (double)sf->a[j] : sd->a[j];
      pool[pi].hit = (lbl >= 0 && lbl < nl && bm[lbl]) ? 1 : 0;
      pi++;
    }
    for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++)
      if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 0;
  }
  free(bm);

  qsort(pool, te, sizeof(tk_entry_t), tk_entry_desc);
  uint64_t tp = 0, best_tp = 0;
  double best_f1 = 0.0;
  int64_t best_k = 0;
  for (int64_t i = 0; i < (int64_t)te; i++) {
    tp += pool[i].hit;
    double f1 = 2.0 * (double)tp / ((double)(i + 1) + (double)total_expected);
    if (f1 > best_f1) { best_f1 = f1; best_k = i + 1; best_tp = tp; }
  }

  double threshold;
  if (best_k == 0)
    threshold = HUGE_VAL;
  else if (best_k == (int64_t)te)
    threshold = pool[te - 1].score;
  else
    threshold = (pool[best_k - 1].score + pool[best_k].score) / 2.0;
  free(pool);

  for (int64_t l = 0; l < nl; l++) g->thresholds[l] = threshold;
  double precision = best_k > 0 ? (double)best_tp / (double)best_k : 0.0;
  double recall = total_expected > 0 ? (double)best_tp / (double)total_expected : 0.0;
  lua_pushnumber(L, best_f1);
  lua_pushnumber(L, precision);
  lua_pushnumber(L, recall);
  return 3;
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
  int64_t nl = g->nl;
  double *thresholds = g->thresholds;
  tk_ivec_t *ks = tk_ivec_create(L, (uint64_t)ns);
  #pragma omp parallel for schedule(static)
  for (int64_t s = 0; s < ns; s++) {
    int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
    int64_t k = 0;
    for (int64_t j = ps; j < pe; j++) {
      int64_t l = neighbors->a[j];
      if (l >= 0 && l < nl) {
        double sc = sf ? (double)sf->a[j] : sd->a[j];
        if (sc >= thresholds[l]) k++;
      }
    }
    ks->a[s] = k;
  }
  return 1;
}

static int tk_gfm_score_lua (lua_State *L)
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
  lua_getfield(L, 2, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 2, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 5);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "gfm.score", "n_samples");
  int64_t nl = g->nl;
  double *thresholds = g->thresholds;
  uint64_t tp = 0, predicted = 0, total_expected = 0;
  #pragma omp parallel reduction(+:tp,predicted,total_expected)
  {
    uint8_t *my_bm = (uint8_t *)calloc((uint64_t)nl, sizeof(uint8_t));
    #pragma omp for schedule(static)
    for (int64_t s = 0; s < ns; s++) {
      int64_t es = exp_off->a[s], ee = exp_off->a[s + 1];
      int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
      int64_t hood_size = pe - ps;
      if (hood_size == 0) continue;
      uint64_t n_expected = (uint64_t)(ee - es);
      total_expected += n_expected;
      for (int64_t j = es; j < ee; j++) {
        int64_t l = exp_nbr->a[j];
        if (l >= 0 && l < nl) my_bm[l] = 1;
      }
      int64_t k = 0;
      for (int64_t j = ps; j < pe; j++) {
        int64_t l = neighbors->a[j];
        if (l >= 0 && l < nl) {
          double sc = sf ? (double)sf->a[j] : sd->a[j];
          if (sc >= thresholds[l]) k++;
        }
      }
      if (k > hood_size) k = hood_size;
      for (int64_t j = ps; j < ps + k; j++) {
        predicted++;
        int64_t l = neighbors->a[j];
        if (l >= 0 && l < nl && my_bm[l]) tp++;
      }
      for (int64_t j = es; j < ee; j++) {
        int64_t l = exp_nbr->a[j];
        if (l >= 0 && l < nl) my_bm[l] = 0;
      }
    }
    free(my_bm);
  }
  double prec = predicted > 0 ? (double)tp / (double)predicted : 0.0;
  double rec = total_expected > 0 ? (double)tp / (double)total_expected : 0.0;
  double f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
  lua_pushnumber(L, f1);
  lua_newtable(L);
  lua_pushnumber(L, prec); lua_setfield(L, -2, "micro_precision");
  lua_pushnumber(L, rec); lua_setfield(L, -2, "micro_recall");
  lua_pushnumber(L, f1); lua_setfield(L, -2, "micro_f1");
  return 2;
}

static int tk_gfm_set_threshold_lua (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  double t = luaL_checknumber(L, 2);
  for (int64_t l = 0; l < g->nl; l++) g->thresholds[l] = t;
  return 0;
}

static int tk_gfm_persist_lua (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKgf", 1, 4, fh);
  uint8_t version = 13;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &g->nl, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, g->thresholds, sizeof(double), (size_t)g->nl, fh);
  tk_lua_fclose(L, fh);
  return 0;
}

static int tk_gfm_load_lua (lua_State *L)
{
  const char *data = luaL_checkstring(L, 1);
  FILE *fh = tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKgf", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid gfm file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 13) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported gfm version %d", (int)version);
  }
  int64_t nl;
  tk_lua_fread(L, &nl, sizeof(int64_t), 1, fh);
  double *thresholds = (double *)malloc((uint64_t)nl * sizeof(double));
  tk_lua_fread(L, thresholds, sizeof(double), (size_t)nl, fh);
  tk_lua_fclose(L, fh);
  tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
    TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
  g->nl = nl;
  g->thresholds = thresholds;
  g->destroyed = false;
  return 1;
}

static luaL_Reg tk_gfm_mt_fns[] = {
  { "calibrate", tk_gfm_calibrate_lua },
  { "predict", tk_gfm_predict_lua },
  { "score", tk_gfm_score_lua },
  { "set_threshold", tk_gfm_set_threshold_lua },
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
