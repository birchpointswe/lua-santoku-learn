#include <lua.h>
#include <lauxlib.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>

#define TK_GFM_MT "tk_gfm_t"

typedef struct {
  tk_dvec_t *thresholds;
  tk_dvec_t *values;
  tk_ivec_t *label_offsets;
  int64_t n_labels;
  double avg_n_true;
  bool destroyed;
} tk_gfm_t;

static inline tk_gfm_t *tk_gfm_peek (lua_State *L, int i) {
  return (tk_gfm_t *)luaL_checkudata(L, i, TK_GFM_MT);
}

static inline int tk_gfm_gc (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  g->thresholds = NULL;
  g->values = NULL;
  g->label_offsets = NULL;
  g->destroyed = true;
  return 0;
}

static luaL_Reg tk_gfm_mt_fns[];

typedef struct {
  double score;
  uint8_t positive;
} tk_iso_point_t;

static int tk_iso_point_asc (const void *a, const void *b) {
  double sa = ((const tk_iso_point_t *)a)->score;
  double sb = ((const tk_iso_point_t *)b)->score;
  return (sa > sb) - (sa < sb);
}

static int64_t isotonic_fit (
  const double *scores, const uint8_t *positive, int64_t n,
  double *out_thresh, double *out_val
) {
  if (n == 0) return 0;
  tk_iso_point_t *pts = (tk_iso_point_t *)malloc((uint64_t)n * sizeof(tk_iso_point_t));
  for (int64_t i = 0; i < n; i++) {
    pts[i].score = scores[i];
    pts[i].positive = positive[i];
  }
  qsort(pts, (size_t)n, sizeof(tk_iso_point_t), tk_iso_point_asc);
  double *bsum = (double *)malloc((uint64_t)n * sizeof(double));
  double *bwt = (double *)malloc((uint64_t)n * sizeof(double));
  double *bth = (double *)malloc((uint64_t)n * sizeof(double));
  int64_t nb = 0;
  for (int64_t i = 0; i < n; i++) {
    if (nb > 0 && pts[i].score == bth[nb - 1]) {
      bsum[nb - 1] += (double)pts[i].positive;
      bwt[nb - 1] += 1.0;
    } else {
      bsum[nb] = (double)pts[i].positive;
      bwt[nb] = 1.0;
      bth[nb] = pts[i].score;
      nb++;
    }
    while (nb >= 2 && bsum[nb - 2] * bwt[nb - 1] > bsum[nb - 1] * bwt[nb - 2]) {
      bsum[nb - 2] += bsum[nb - 1];
      bwt[nb - 2] += bwt[nb - 1];
      bth[nb - 2] = bth[nb - 1];
      nb--;
    }
  }
  for (int64_t i = 0; i < nb; i++) {
    out_thresh[i] = bth[i];
    out_val[i] = bsum[i] / bwt[i];
  }
  free(pts);
  free(bsum);
  free(bwt);
  free(bth);
  return nb;
}

static inline double isotonic_lookup (
  const double *thresholds, const double *values,
  int64_t lo, int64_t hi, double score, double default_prob
) {
  if (lo >= hi) return default_prob;
  if (score <= thresholds[lo]) return default_prob;
  if (score >= thresholds[hi - 1]) return values[hi - 1];
  int64_t a = lo, b = hi - 1;
  while (a < b) {
    int64_t mid = a + (b - a + 1) / 2;
    if (thresholds[mid] <= score) a = mid;
    else b = mid - 1;
  }
  return values[a];
}

static int tk_gfm_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "pred_offsets");
  tk_ivec_t *pred_off = tk_ivec_peek(L, -1, "pred_offsets");
  lua_getfield(L, 1, "pred_neighbors");
  tk_ivec_t *pred_nbr = tk_ivec_peek(L, -1, "pred_neighbors");
  lua_getfield(L, 1, "pred_scores");
  tk_dvec_t *pred_sc = tk_dvec_peek(L, -1, "pred_scores");
  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 5);

  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 1, "gfm.create", "n_samples");
  int64_t nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "gfm.create", "n_labels");

  double avg_n_true = 0;
  for (int64_t s = 0; s < ns; s++)
    avg_n_true += (double)(exp_off->a[s + 1] - exp_off->a[s]);
  avg_n_true /= ns;

  int64_t *lcount = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
  for (uint64_t i = 0; i < exp_nbr->n; i++)
    lcount[exp_nbr->a[i]]++;
  int64_t *lsoff = (int64_t *)calloc((uint64_t)(nl + 1), sizeof(int64_t));
  for (int64_t l = 0; l < nl; l++)
    lsoff[l + 1] = lsoff[l] + lcount[l];
  int64_t total_gt = lsoff[nl];
  int64_t *lsamp = (int64_t *)malloc((uint64_t)(total_gt > 0 ? total_gt : 1) * sizeof(int64_t));
  int64_t *lfill = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
  for (int64_t s = 0; s < ns; s++) {
    for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++) {
      int64_t l = exp_nbr->a[j];
      lsamp[lsoff[l] + lfill[l]] = s;
      lfill[l]++;
    }
  }
  free(lfill);
  free(lcount);

  int64_t *plcount = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
  for (int64_t s = 0; s < ns; s++)
    for (int64_t j = pred_off->a[s]; j < pred_off->a[s + 1]; j++)
      plcount[pred_nbr->a[j]]++;
  int64_t *plsoff = (int64_t *)calloc((uint64_t)(nl + 1), sizeof(int64_t));
  for (int64_t l = 0; l < nl; l++)
    plsoff[l + 1] = plsoff[l] + plcount[l];
  int64_t total_pred = plsoff[nl];
  int64_t *plsamp = (int64_t *)malloc((uint64_t)(total_pred > 0 ? total_pred : 1) * sizeof(int64_t));
  double *plscore = (double *)malloc((uint64_t)(total_pred > 0 ? total_pred : 1) * sizeof(double));
  int64_t *plfill = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
  for (int64_t s = 0; s < ns; s++) {
    for (int64_t j = pred_off->a[s]; j < pred_off->a[s + 1]; j++) {
      int64_t l = pred_nbr->a[j];
      int64_t pos = plsoff[l] + plfill[l];
      plsamp[pos] = s;
      plscore[pos] = pred_sc->a[j];
      plfill[l]++;
    }
  }
  free(plfill);
  free(plcount);

  double *temp_thresh = (double *)malloc((uint64_t)(total_pred > 0 ? total_pred : 1) * sizeof(double));
  double *temp_val = (double *)malloc((uint64_t)(total_pred > 0 ? total_pred : 1) * sizeof(double));
  int64_t *block_counts = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));

  #pragma omp parallel
  {
    uint8_t *pos_bm = (uint8_t *)calloc((uint64_t)ns, sizeof(uint8_t));
    uint8_t *pos_short = NULL;
    int64_t pos_short_cap = 0;
    #pragma omp for schedule(dynamic)
    for (int64_t l = 0; l < nl; l++) {
      int64_t pn = plsoff[l + 1] - plsoff[l];
      if (pn == 0) { block_counts[l] = 0; continue; }
      for (int64_t j = lsoff[l]; j < lsoff[l + 1]; j++)
        pos_bm[lsamp[j]] = 1;
      if (pn > pos_short_cap) {
        free(pos_short);
        pos_short = (uint8_t *)malloc((uint64_t)pn);
        pos_short_cap = pn;
      }
      int64_t base = plsoff[l];
      for (int64_t j = 0; j < pn; j++)
        pos_short[j] = pos_bm[plsamp[base + j]];
      block_counts[l] = isotonic_fit(
        plscore + base, pos_short, pn,
        temp_thresh + base, temp_val + base);
      for (int64_t j = lsoff[l]; j < lsoff[l + 1]; j++)
        pos_bm[lsamp[j]] = 0;
    }
    free(pos_bm);
    free(pos_short);
  }

  free(lsoff);
  free(lsamp);

  int64_t *loff_raw = (int64_t *)malloc((uint64_t)(nl + 1) * sizeof(int64_t));
  loff_raw[0] = 0;
  for (int64_t l = 0; l < nl; l++)
    loff_raw[l + 1] = loff_raw[l] + block_counts[l];
  int64_t total_blocks = loff_raw[nl];
  uint64_t tb = total_blocks > 0 ? (uint64_t)total_blocks : 1;

  double *final_thresh = (double *)malloc(tb * sizeof(double));
  double *final_val = (double *)malloc(tb * sizeof(double));
  for (int64_t l = 0; l < nl; l++) {
    int64_t cnt = block_counts[l];
    if (cnt > 0) {
      memcpy(final_thresh + loff_raw[l], temp_thresh + plsoff[l], (uint64_t)cnt * sizeof(double));
      memcpy(final_val + loff_raw[l], temp_val + plsoff[l], (uint64_t)cnt * sizeof(double));
    }
  }
  free(temp_thresh);
  free(temp_val);
  free(block_counts);
  free(plsoff);
  free(plsamp);
  free(plscore);

  tk_ivec_t *label_off_iv = tk_ivec_create(L, (uint64_t)(nl + 1), NULL, NULL);
  int loff_idx = lua_gettop(L);
  memcpy(label_off_iv->a, loff_raw, (uint64_t)(nl + 1) * sizeof(int64_t));
  free(loff_raw);

  tk_dvec_t *thresh_dv = tk_dvec_create(L, tb, NULL, NULL);
  int thresh_idx = lua_gettop(L);
  memcpy(thresh_dv->a, final_thresh, tb * sizeof(double));
  if (total_blocks == 0) thresh_dv->n = 0;
  free(final_thresh);

  tk_dvec_t *val_dv = tk_dvec_create(L, tb, NULL, NULL);
  int val_idx = lua_gettop(L);
  memcpy(val_dv->a, final_val, tb * sizeof(double));
  if (total_blocks == 0) val_dv->n = 0;
  free(final_val);

  tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
    TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
  int gi = lua_gettop(L);
  g->thresholds = thresh_dv;
  g->values = val_dv;
  g->label_offsets = label_off_iv;
  g->n_labels = nl;
  g->avg_n_true = avg_n_true;
  g->destroyed = false;

  lua_newtable(L);
  lua_pushvalue(L, loff_idx);
  lua_setfield(L, -2, "label_offsets");
  lua_pushvalue(L, thresh_idx);
  lua_setfield(L, -2, "thresholds");
  lua_pushvalue(L, val_idx);
  lua_setfield(L, -2, "values");
  lua_setfenv(L, gi);

  lua_pushvalue(L, gi);
  return 1;
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
  tk_dvec_t *scores = tk_dvec_peek(L, -1, "scores");
  lua_pop(L, 3);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "gfm.predict", "n_samples");

  double *th = g->thresholds->a;
  double *va = g->values->a;
  int64_t *loff = g->label_offsets->a;
  double avg_n = g->avg_n_true;
  double default_prob = g->n_labels > 0 ? avg_n / (double)g->n_labels : 0.0;

  tk_ivec_t *ks = tk_ivec_create(L, (uint64_t)ns, NULL, NULL);

  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < ns; s++) {
      int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
      int64_t hood = pe - ps;
      if (hood <= 0) { ks->a[s] = 0; continue; }

      double *probs = (double *)malloc((uint64_t)hood * sizeof(double));
      double psum = 0;
      for (int64_t j = 0; j < hood; j++) {
        int64_t label = neighbors->a[ps + j];
        double sc = scores->a[ps + j];
        probs[j] = isotonic_lookup(th, va, loff[label], loff[label + 1], sc, default_prob);
        psum += probs[j];
      }

      double mu_tail = avg_n - psum;
      if (mu_tail < 0) mu_tail = 0;

      int64_t m = hood;
      double *sfx = (double *)calloc((uint64_t)(m + 1) * (uint64_t)(m + 1), sizeof(double));
      #define SFX(j, t) sfx[(int64_t)(j) * (m + 1) + (t)]
      SFX(m, 0) = 1.0;
      for (int64_t j = m - 1; j >= 0; j--) {
        double pj = probs[j], qj = 1.0 - pj;
        for (int64_t t = 0; t <= m - j; t++) {
          double v = SFX(j + 1, t) * qj;
          if (t > 0) v += SFX(j + 1, t - 1) * pj;
          SFX(j, t) = v;
        }
      }

      double *pfx = (double *)calloc((uint64_t)(m + 1), sizeof(double));
      pfx[0] = 1.0;
      int64_t best_k = 1;
      double best_ef1 = 0;

      for (int64_t k = 1; k <= m; k++) {
        double pk = probs[k - 1], qk = 1.0 - pk;
        double *new_pfx = (double *)calloc((uint64_t)(k + 1), sizeof(double));
        for (int64_t tp = 0; tp <= k; tp++) {
          double v = 0;
          if (tp < k) v += pfx[tp] * qk;
          if (tp > 0) v += pfx[tp - 1] * pk;
          new_pfx[tp] = v;
        }
        free(pfx);
        pfx = new_pfx;

        double ef1 = 0;
        int64_t max_fn = m - k;
        for (int64_t tp = 1; tp <= k; tp++) {
          double p_tp = pfx[tp];
          if (p_tp < 1e-15) continue;
          for (int64_t fn = 0; fn <= max_fn; fn++) {
            double p_fn = SFX(k, fn);
            if (p_fn < 1e-15) continue;
            ef1 += p_tp * p_fn * (double)tp / ((double)k + (double)tp + (double)fn + mu_tail);
          }
        }
        ef1 *= 2.0;
        if (ef1 > best_ef1) { best_ef1 = ef1; best_k = k; }
      }

      free(pfx);
      free(sfx);
      free(probs);
      #undef SFX
      ks->a[s] = best_k;
    }
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
  uint8_t version = 2;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &g->n_labels, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &g->avg_n_true, sizeof(double), 1, fh);
  tk_ivec_persist(L, g->label_offsets, fh);
  tk_dvec_persist(L, g->thresholds, fh);
  tk_dvec_persist(L, g->values, fh);
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
  if (version != 2) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported gfm version %d", (int)version);
  }
  int64_t n_labels;
  double avg_n_true;
  tk_lua_fread(L, &n_labels, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &avg_n_true, sizeof(double), 1, fh);
  tk_ivec_t *label_off = tk_ivec_load(L, fh);
  int loff_idx = lua_gettop(L);
  tk_dvec_t *thresh = tk_dvec_load(L, fh);
  int thresh_idx = lua_gettop(L);
  tk_dvec_t *vals = tk_dvec_load(L, fh);
  int val_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);

  tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
    TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
  int gi = lua_gettop(L);
  g->thresholds = thresh;
  g->values = vals;
  g->label_offsets = label_off;
  g->n_labels = n_labels;
  g->avg_n_true = avg_n_true;
  g->destroyed = false;

  lua_newtable(L);
  lua_pushvalue(L, loff_idx);
  lua_setfield(L, -2, "label_offsets");
  lua_pushvalue(L, thresh_idx);
  lua_setfield(L, -2, "thresholds");
  lua_pushvalue(L, val_idx);
  lua_setfield(L, -2, "values");
  lua_setfenv(L, gi);

  lua_pushvalue(L, gi);
  return 1;
}

static luaL_Reg tk_gfm_mt_fns[] = {
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
