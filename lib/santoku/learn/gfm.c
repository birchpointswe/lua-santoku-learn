#include <lua.h>
#include <lauxlib.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>

#define TK_GFM_MT "tk_gfm_t"

typedef struct {
  double *label_thresholds;
  double *label_values;
  int64_t *label_offsets;
  int64_t n_labels;
  double default_prob;
  double *sy_thresholds;
  double *sy_values;
  int64_t *sy_offsets;
  int64_t pis_max_s;
  double p_sy0;
  bool destroyed;
} tk_gfm_t;

static inline tk_gfm_t *tk_gfm_peek (lua_State *L, int i) {
  return (tk_gfm_t *)luaL_checkudata(L, i, TK_GFM_MT);
}

static inline int tk_gfm_gc (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  free(g->label_thresholds);
  free(g->label_values);
  free(g->label_offsets);
  free(g->sy_thresholds);
  free(g->sy_values);
  free(g->sy_offsets);
  g->destroyed = true;
  return 0;
}

static luaL_Reg tk_gfm_mt_fns[];

static inline double isotonic_lookup (
  const double *thresholds, const double *values,
  int64_t lo, int64_t hi, double score, double default_val
) {
  if (lo >= hi) return default_val;
  if (score <= thresholds[lo]) return default_val;
  if (score >= thresholds[hi - 1]) return values[hi - 1];
  int64_t a = lo, b = hi - 1;
  while (a < b) {
    int64_t mid = a + (b - a + 1) / 2;
    if (thresholds[mid] <= score) a = mid;
    else b = mid - 1;
  }
  return values[a];
}

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

typedef struct { double score; int64_t sy; uint8_t is_hit; } tk_cand_t;

static int tk_cand_asc (const void *a, const void *b) {
  double sa = ((const tk_cand_t *)a)->score;
  double sb = ((const tk_cand_t *)b)->score;
  return (sa > sb) - (sa < sb);
}

static int64_t pav_joint (
  const tk_cand_t *cands, int64_t n, int64_t target_sy,
  double *bsum, double *bwt, double *bth,
  double *out_thresh, double *out_val
) {
  int64_t nb = 0;
  for (int64_t i = 0; i < n; i++) {
    double pos = (cands[i].is_hit && cands[i].sy == target_sy) ? 1.0 : 0.0;
    if (nb > 0 && cands[i].score == bth[nb - 1]) {
      bsum[nb - 1] += pos;
      bwt[nb - 1] += 1.0;
    } else {
      bsum[nb] = pos;
      bwt[nb] = 1.0;
      bth[nb] = cands[i].score;
      nb++;
    }
    while (nb >= 2 && bsum[nb - 2] * bwt[nb - 1] > bsum[nb - 1] * bwt[nb - 2]) {
      bsum[nb - 2] += bsum[nb - 1];
      bwt[nb - 2] += bwt[nb - 1];
      bth[nb - 2] = bth[nb - 1];
      nb--;
    }
  }
  if (out_thresh && out_val) {
    for (int64_t i = 0; i < nb; i++) {
      out_thresh[i] = bth[i];
      out_val[i] = bsum[i] / bwt[i];
    }
  }
  return nb;
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
  tk_fvec_t *pred_sco_f = tk_fvec_peekopt(L, -1);
  tk_dvec_t *pred_sco_d = pred_sco_f ? NULL : tk_dvec_peek(L, -1, "pred_scores");
  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 5);

  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 1, "gfm.create", "n_samples");
  int64_t nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "gfm.create", "n_labels");

  double avg_n_true = 0;
  int64_t max_s = 0;
  int64_t n_sy0 = 0;
  for (int64_t s = 0; s < ns; s++) {
    int64_t gt_n = exp_off->a[s + 1] - exp_off->a[s];
    avg_n_true += (double)gt_n;
    if (gt_n > max_s) max_s = gt_n;
    if (gt_n == 0) n_sy0++;
  }
  avg_n_true /= (double)ns;
  double default_prob = nl > 0 ? avg_n_true / (double)nl : 0.0;
  double p_sy0 = ns > 0 ? (double)n_sy0 / (double)ns : 0.0;

  int64_t total_cands = pred_off->a[ns] - pred_off->a[0];

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
      plscore[pos] = pred_sco_f ? (double)pred_sco_f->a[j] : pred_sco_d->a[j];
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

  int64_t *label_off_raw = (int64_t *)malloc((uint64_t)(nl + 1) * sizeof(int64_t));
  label_off_raw[0] = 0;
  for (int64_t l = 0; l < nl; l++)
    label_off_raw[l + 1] = label_off_raw[l] + block_counts[l];
  int64_t total_label_blocks = label_off_raw[nl];
  uint64_t tlb = total_label_blocks > 0 ? (uint64_t)total_label_blocks : 1;

  double *label_thresh = (double *)malloc(tlb * sizeof(double));
  double *label_val = (double *)malloc(tlb * sizeof(double));
  for (int64_t l = 0; l < nl; l++) {
    int64_t cnt = block_counts[l];
    if (cnt > 0) {
      memcpy(label_thresh + label_off_raw[l], temp_thresh + plsoff[l], (uint64_t)cnt * sizeof(double));
      memcpy(label_val + label_off_raw[l], temp_val + plsoff[l], (uint64_t)cnt * sizeof(double));
    }
  }
  free(temp_thresh);
  free(temp_val);
  free(block_counts);
  free(plsoff);
  free(plsamp);
  free(plscore);

  if (total_cands == 0 || max_s == 0) {
    tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
      TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
    g->label_thresholds = label_thresh;
    g->label_values = label_val;
    g->label_offsets = label_off_raw;
    g->n_labels = nl;
    g->default_prob = default_prob;
    g->sy_thresholds = NULL;
    g->sy_values = NULL;
    g->sy_offsets = NULL;
    g->pis_max_s = 0;
    g->p_sy0 = p_sy0;
    g->destroyed = false;
    return 1;
  }

  int64_t max_label = 0;
  for (uint64_t i = 0; i < exp_nbr->n; i++)
    if (exp_nbr->a[i] > max_label) max_label = exp_nbr->a[i];
  for (uint64_t i = 0; i < pred_nbr->n; i++)
    if (pred_nbr->a[i] > max_label) max_label = pred_nbr->a[i];

  tk_cand_t *cands = (tk_cand_t *)malloc((uint64_t)total_cands * sizeof(tk_cand_t));
  uint8_t *gt_bm = (uint8_t *)calloc((uint64_t)(max_label + 1), sizeof(uint8_t));
  int64_t ci = 0;
  for (int64_t s = 0; s < ns; s++) {
    int64_t gt_n = exp_off->a[s + 1] - exp_off->a[s];
    int64_t sy = gt_n < max_s ? gt_n : max_s;
    for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++)
      gt_bm[exp_nbr->a[j]] = 1;
    for (int64_t j = pred_off->a[s]; j < pred_off->a[s + 1]; j++) {
      int64_t label = pred_nbr->a[j];
      double raw_score = pred_sco_f ? (double)pred_sco_f->a[j] : pred_sco_d->a[j];
      double cal_p = isotonic_lookup(label_thresh, label_val,
        label_off_raw[label], label_off_raw[label + 1], raw_score, default_prob);
      cands[ci].score = cal_p;
      cands[ci].is_hit = gt_bm[label];
      cands[ci].sy = sy;
      ci++;
    }
    for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++)
      gt_bm[exp_nbr->a[j]] = 0;
  }
  free(gt_bm);

  qsort(cands, (size_t)total_cands, sizeof(tk_cand_t), tk_cand_asc);

  double *bsum = (double *)malloc((uint64_t)total_cands * sizeof(double));
  double *bwt = (double *)malloc((uint64_t)total_cands * sizeof(double));
  double *bth = (double *)malloc((uint64_t)total_cands * sizeof(double));

  int64_t *sy_block_counts = (int64_t *)calloc((uint64_t)(max_s + 1), sizeof(int64_t));
  for (int64_t sv = 1; sv <= max_s; sv++)
    sy_block_counts[sv] = pav_joint(cands, total_cands, sv, bsum, bwt, bth, NULL, NULL);

  int64_t *sy_off = (int64_t *)malloc((uint64_t)(max_s + 2) * sizeof(int64_t));
  sy_off[0] = 0;
  for (int64_t sv = 0; sv <= max_s; sv++)
    sy_off[sv + 1] = sy_off[sv] + sy_block_counts[sv];
  int64_t total_sy_blocks = sy_off[max_s + 1];
  free(sy_block_counts);

  uint64_t tsb = total_sy_blocks > 0 ? (uint64_t)total_sy_blocks : 1;
  double *sy_thresh = (double *)malloc(tsb * sizeof(double));
  double *sy_val = (double *)malloc(tsb * sizeof(double));

  for (int64_t sv = 1; sv <= max_s; sv++)
    pav_joint(cands, total_cands, sv, bsum, bwt, bth,
              sy_thresh + sy_off[sv], sy_val + sy_off[sv]);

  free(cands);
  free(bsum);
  free(bwt);
  free(bth);

  tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
    TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
  g->label_thresholds = label_thresh;
  g->label_values = label_val;
  g->label_offsets = label_off_raw;
  g->n_labels = nl;
  g->default_prob = default_prob;
  g->sy_thresholds = sy_thresh;
  g->sy_values = sy_val;
  g->sy_offsets = sy_off;
  g->pis_max_s = max_s;
  g->p_sy0 = p_sy0;
  g->destroyed = false;

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
  tk_fvec_t *scores_f = tk_fvec_peekopt(L, -1);
  tk_dvec_t *scores_d = scores_f ? NULL : tk_dvec_peek(L, -1, "scores");
  lua_pop(L, 3);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "gfm.predict", "n_samples");

  double *lth = g->label_thresholds;
  double *lva = g->label_values;
  int64_t *loff = g->label_offsets;
  double defp = g->default_prob;
  double *sth = g->sy_thresholds;
  double *sva = g->sy_values;
  int64_t *soff = g->sy_offsets;
  int64_t pis_max_s = g->pis_max_s;
  int64_t stride = pis_max_s + 1;

  tk_ivec_t *ks = tk_ivec_create(L, (uint64_t)ns, NULL, NULL);

  if (pis_max_s == 0 || sth == NULL) {
    memset(ks->a, 0, (uint64_t)ns * sizeof(int64_t));
    return 1;
  }

  #pragma omp parallel
  {
    double *prefix = (double *)calloc((uint64_t)stride, sizeof(double));

    #pragma omp for schedule(dynamic)
    for (int64_t s = 0; s < ns; s++) {
      int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
      int64_t m = pe - ps;
      if (m <= 0) { ks->a[s] = 0; continue; }

      memset(prefix, 0, (uint64_t)stride * sizeof(double));
      int64_t best_k = 0;
      double best_ef = g->p_sy0;

      for (int64_t k = 1; k <= m; k++) {
        int64_t label = neighbors->a[ps + k - 1];
        double sc = scores_f ? (double)scores_f->a[ps + k - 1] : scores_d->a[ps + k - 1];
        double cal_p = isotonic_lookup(lth, lva, loff[label], loff[label + 1], sc, defp);
        for (int64_t si = 1; si <= pis_max_s; si++)
          prefix[si] += isotonic_lookup(sth, sva, soff[si], soff[si + 1], cal_p, 0.0);
        double ef = 0;
        for (int64_t si = 1; si <= pis_max_s; si++)
          ef += prefix[si] * 2.0 / ((double)si + (double)k);
        if (ef > best_ef) { best_ef = ef; best_k = k; }
      }

      ks->a[s] = best_k;
    }
    free(prefix);
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
  uint8_t version = 5;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &g->n_labels, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &g->default_prob, sizeof(double), 1, fh);
  tk_lua_fwrite(L, g->label_offsets, sizeof(int64_t), (uint64_t)(g->n_labels + 1), fh);
  int64_t total_label_blocks = g->label_offsets[g->n_labels];
  if (total_label_blocks > 0) {
    tk_lua_fwrite(L, g->label_thresholds, sizeof(double), (uint64_t)total_label_blocks, fh);
    tk_lua_fwrite(L, g->label_values, sizeof(double), (uint64_t)total_label_blocks, fh);
  }
  tk_lua_fwrite(L, &g->pis_max_s, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &g->p_sy0, sizeof(double), 1, fh);
  if (g->pis_max_s > 0 && g->sy_offsets) {
    int64_t n_sy_off = g->pis_max_s + 2;
    tk_lua_fwrite(L, g->sy_offsets, sizeof(int64_t), (uint64_t)n_sy_off, fh);
    int64_t total_sy_blocks = g->sy_offsets[g->pis_max_s + 1];
    if (total_sy_blocks > 0) {
      tk_lua_fwrite(L, g->sy_thresholds, sizeof(double), (uint64_t)total_sy_blocks, fh);
      tk_lua_fwrite(L, g->sy_values, sizeof(double), (uint64_t)total_sy_blocks, fh);
    }
  }
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
  if (version != 5) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported gfm version %d", (int)version);
  }
  int64_t n_labels;
  double default_prob;
  tk_lua_fread(L, &n_labels, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &default_prob, sizeof(double), 1, fh);
  int64_t *label_offsets = (int64_t *)malloc((uint64_t)(n_labels + 1) * sizeof(int64_t));
  tk_lua_fread(L, label_offsets, sizeof(int64_t), (uint64_t)(n_labels + 1), fh);
  int64_t total_label_blocks = label_offsets[n_labels];
  double *label_thresholds = NULL;
  double *label_values = NULL;
  if (total_label_blocks > 0) {
    label_thresholds = (double *)malloc((uint64_t)total_label_blocks * sizeof(double));
    label_values = (double *)malloc((uint64_t)total_label_blocks * sizeof(double));
    tk_lua_fread(L, label_thresholds, sizeof(double), (uint64_t)total_label_blocks, fh);
    tk_lua_fread(L, label_values, sizeof(double), (uint64_t)total_label_blocks, fh);
  }
  int64_t pis_max_s;
  double p_sy0;
  tk_lua_fread(L, &pis_max_s, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &p_sy0, sizeof(double), 1, fh);
  int64_t *sy_offsets = NULL;
  double *sy_thresholds = NULL;
  double *sy_values = NULL;
  if (pis_max_s > 0) {
    int64_t n_sy_off = pis_max_s + 2;
    sy_offsets = (int64_t *)malloc((uint64_t)n_sy_off * sizeof(int64_t));
    tk_lua_fread(L, sy_offsets, sizeof(int64_t), (uint64_t)n_sy_off, fh);
    int64_t total_sy_blocks = sy_offsets[pis_max_s + 1];
    if (total_sy_blocks > 0) {
      sy_thresholds = (double *)malloc((uint64_t)total_sy_blocks * sizeof(double));
      sy_values = (double *)malloc((uint64_t)total_sy_blocks * sizeof(double));
      tk_lua_fread(L, sy_thresholds, sizeof(double), (uint64_t)total_sy_blocks, fh);
      tk_lua_fread(L, sy_values, sizeof(double), (uint64_t)total_sy_blocks, fh);
    }
  }
  tk_lua_fclose(L, fh);

  tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
    TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
  g->label_thresholds = label_thresholds;
  g->label_values = label_values;
  g->label_offsets = label_offsets;
  g->n_labels = n_labels;
  g->default_prob = default_prob;
  g->sy_thresholds = sy_thresholds;
  g->sy_values = sy_values;
  g->sy_offsets = sy_offsets;
  g->pis_max_s = pis_max_s;
  g->p_sy0 = p_sy0;
  g->destroyed = false;

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
