#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/rvec.h>
#include <santoku/learn/gfm.h>

static inline int tk_gfm_gc (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  free(g->platt_a);
  free(g->platt_b);
  g->destroyed = true;
  return 0;
}

static inline double tk_gfm_calibrate (tk_gfm_t *g, int64_t label, double score) {
  double f = g->platt_a[label] * score + g->platt_b[label];
  return 1.0 / (1.0 + exp(f));
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
  g->platt_a = NULL;
  g->platt_b = NULL;
  g->alpha = 1.0;
  g->destroyed = false;
  return 1;
}

static inline void tk_gfm_platt (
  double *scores, uint8_t *hits, int64_t n,
  double *out_a, double *out_b)
{
  if (n == 0) { *out_a = 0.0; *out_b = 0.0; return; }
  int64_t np = 0;
  for (int64_t i = 0; i < n; i++) np += hits[i];
  int64_t nn = n - np;
  if (np == 0) { *out_a = 0.0; *out_b = 20.0; return; }
  if (nn == 0) { *out_a = 0.0; *out_b = -20.0; return; }
  double hi = (double)(np + 1) / (double)(np + 2);
  double lo = 1.0 / (double)(nn + 2);
  double A = 0.0, B = log((double)nn / (double)np);
  double F = 0.0;
  for (int64_t i = 0; i < n; i++) {
    double t = hits[i] ? hi : lo;
    double f = A * scores[i] + B;
    F += (f >= 0) ? t * f + log(1.0 + exp(-f)) : (t - 1.0) * f + log(1.0 + exp(f));
  }
  for (int iter = 0; iter < 100; iter++) {
    double h11 = 1e-12, h22 = 1e-12, h21 = 0, g1 = 0, g2 = 0;
    for (int64_t i = 0; i < n; i++) {
      double t = hits[i] ? hi : lo;
      double f = A * scores[i] + B;
      double p, q;
      if (f >= 0) { double e = exp(-f); q = 1.0 / (1.0 + e); p = e * q; }
      else { double e = exp(f); p = 1.0 / (1.0 + e); q = e * p; }
      double d2 = p * q;
      h11 += scores[i] * scores[i] * d2;
      h22 += d2;
      h21 += scores[i] * d2;
      g1 += scores[i] * (t - p);
      g2 += (t - p);
    }
    if (fabs(g1) < 1e-5 && fabs(g2) < 1e-5) break;
    double det = h11 * h22 - h21 * h21;
    double dA = -(h22 * g1 - h21 * g2) / det;
    double dB = -(-h21 * g1 + h11 * g2) / det;
    double gd = g1 * dA + g2 * dB;
    double step = 1.0;
    while (step >= 1e-10) {
      double nA = A + step * dA, nB = B + step * dB;
      double nF = 0.0;
      for (int64_t i = 0; i < n; i++) {
        double t = hits[i] ? hi : lo;
        double f = nA * scores[i] + nB;
        nF += (f >= 0) ? t * f + log(1.0 + exp(-f)) : (t - 1.0) * f + log(1.0 + exp(f));
      }
      if (nF < F + 0.0001 * step * gd) { A = nA; B = nB; F = nF; break; }
      step *= 0.5;
    }
    if (step < 1e-10) break;
  }
  *out_a = A;
  *out_b = B;
}

static int tk_gfm_fit_lua (lua_State *L)
{
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  lua_getfield(L, 2, "pred_offsets");
  tk_ivec_t *po = tk_ivec_peek(L, -1, "pred_offsets");
  lua_getfield(L, 2, "pred_neighbors");
  tk_ivec_t *pn = tk_ivec_peek(L, -1, "pred_neighbors");
  lua_getfield(L, 2, "pred_scores");
  tk_fvec_t *psf = tk_fvec_peekopt(L, -1);
  tk_dvec_t *psd = psf ? NULL : tk_dvec_peek(L, -1, "pred_scores");
  lua_getfield(L, 2, "expected_offsets");
  tk_ivec_t *eo = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 2, "expected_neighbors");
  tk_ivec_t *en = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 5);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "gfm.fit", "n_samples");
  int64_t nl = g->nl;
  int64_t total_entries = po->a[ns] - po->a[0];
  if (!g->platt_a) {
    g->platt_a = (double *)calloc((uint64_t)nl, sizeof(double));
    g->platt_b = (double *)calloc((uint64_t)nl, sizeof(double));
  }
  if (total_entries <= 0) return 0;
  uint64_t te = (uint64_t)total_entries;
  int64_t *pl_off = (int64_t *)malloc((uint64_t)(nl + 1) * sizeof(int64_t));
  int64_t *pl_count = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
  uint8_t *pl_hit = (uint8_t *)malloc(te * sizeof(uint8_t));
  double *pl_score = (double *)malloc(te * sizeof(double));
  uint8_t *gt_bm = (uint8_t *)calloc((uint64_t)nl, sizeof(uint8_t));
  for (int64_t i = 0; i < total_entries; i++) {
    int64_t l = pn->a[i];
    if (l >= 0 && l < nl) pl_count[l]++;
  }
  pl_off[0] = 0;
  for (int64_t l = 0; l < nl; l++)
    pl_off[l + 1] = pl_off[l] + pl_count[l];
  memset(pl_count, 0, (uint64_t)nl * sizeof(int64_t));
  for (int64_t s = 0; s < ns; s++) {
    for (int64_t j = eo->a[s]; j < eo->a[s + 1]; j++)
      if (en->a[j] >= 0 && en->a[j] < nl) gt_bm[en->a[j]] = 1;
    for (int64_t j = po->a[s]; j < po->a[s + 1]; j++) {
      int64_t l = pn->a[j];
      if (l >= 0 && l < nl) {
        int64_t pos = pl_off[l] + pl_count[l];
        pl_score[pos] = psf ? (double)psf->a[j] : psd->a[j];
        pl_hit[pos] = gt_bm[l];
        pl_count[l]++;
      }
    }
    for (int64_t j = eo->a[s]; j < eo->a[s + 1]; j++)
      if (en->a[j] >= 0 && en->a[j] < nl) gt_bm[en->a[j]] = 0;
  }
  free(pl_count); free(gt_bm);
  #pragma omp parallel for schedule(dynamic)
  for (int64_t l = 0; l < nl; l++) {
    int64_t lo = pl_off[l], cnt = pl_off[l + 1] - lo;
    tk_gfm_platt(pl_score + lo, pl_hit + lo, cnt, &g->platt_a[l], &g->platt_b[l]);
  }
  free(pl_hit); free(pl_score); free(pl_off);
  return 0;
}

static int tk_gfm_predict_lua (lua_State *L)
{
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  if (!g->platt_a)
    return luaL_error(L, "predict: not fitted");
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
  double alpha = g->alpha;
  int64_t max_hood = 0;
  for (int64_t s = 0; s < ns; s++) {
    int64_t h = offsets->a[s + 1] - offsets->a[s];
    if (h > max_hood) max_hood = h;
  }
  tk_ivec_t *ks = tk_ivec_create(L, (uint64_t)ns, NULL, NULL);
  uint64_t mh = (uint64_t)max_hood;
  tk_rank_t *rbuf = (tk_rank_t *)malloc(mh * sizeof(tk_rank_t));
  for (int64_t s = 0; s < ns; s++) {
    int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
    int64_t m = pe - ps;
    if (m <= 0) { ks->a[s] = 0; continue; }
    double E_total = 0;
    for (int64_t j = 0; j < m; j++) {
      int64_t l = neighbors->a[ps + j];
      double sc = sf ? (double)sf->a[ps + j] : sd->a[ps + j];
      double c = (l >= 0 && l < nl) ? tk_gfm_calibrate(g, l, sc) : 0.0;
      rbuf[j] = tk_rank(j, c);
      E_total += c;
    }
    if (m > 0) {
      tk_rvec_t rv = { .n = (size_t)m, .m = (size_t)m, .a = rbuf, .lua_managed = false };
      tk_rvec_desc(&rv, 0, (size_t)m);
    }
    double adj_E = alpha * E_total;
    double cum = 0, best_f1 = 0;
    int64_t best_k = 0;
    if (adj_E > 0) {
      for (int64_t k = 1; k <= m; k++) {
        cum += rbuf[k - 1].d;
        double f1 = 2.0 * cum / ((double)k + adj_E);
        if (f1 > best_f1) { best_f1 = f1; best_k = k; }
      }
    }
    ks->a[s] = best_k;
  }
  free(rbuf);
  return 1;
}

static int tk_gfm_alpha_search_lua (lua_State *L)
{
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  if (!g->platt_a)
    return luaL_error(L, "alpha_search: not fitted");
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
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "gfm.alpha_search", "n_samples");
  int64_t nl = g->nl;
  int64_t total_entries = offsets->a[ns] - offsets->a[0];
  uint64_t total_expected = (uint64_t)(exp_off->a[ns] - exp_off->a[0]);
  if (total_entries <= 0 || total_expected == 0) {
    lua_pushnumber(L, 1.0);
    lua_pushnumber(L, 0.0);
    return 2;
  }
  uint64_t te = (uint64_t)total_entries;
  int64_t max_hood = 0;
  for (int64_t s = 0; s < ns; s++) {
    int64_t h = offsets->a[s + 1] - offsets->a[s];
    if (h > max_hood) max_hood = h;
  }
  double *all_cal = (double *)malloc(te * sizeof(double));
  double *sample_E = (double *)malloc((uint64_t)ns * sizeof(double));
  for (int64_t s = 0; s < ns; s++) {
    int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
    double E = 0;
    for (int64_t j = ps; j < pe; j++) {
      int64_t l = neighbors->a[j];
      double sc = sf ? (double)sf->a[j] : sd->a[j];
      all_cal[j] = (l >= 0 && l < nl) ? tk_gfm_calibrate(g, l, sc) : 0.0;
      E += all_cal[j];
    }
    sample_E[s] = E;
  }
  uint8_t *bm = (uint8_t *)calloc((uint64_t)nl, sizeof(uint8_t));
  int64_t *cum_hit = (int64_t *)malloc(te * sizeof(int64_t));
  tk_rank_t *rbuf = (tk_rank_t *)malloc((uint64_t)max_hood * sizeof(tk_rank_t));
  for (int64_t s = 0; s < ns; s++) {
    for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++)
      if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 1;
    int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
    int64_t m = pe - ps;
    for (int64_t j = 0; j < m; j++) {
      int64_t l = neighbors->a[ps + j];
      uint8_t hit = (l >= 0 && l < nl && bm[l]) ? 1 : 0;
      rbuf[j] = tk_rank(hit, all_cal[ps + j]);
    }
    if (m > 0) {
      tk_rvec_t rv = { .n = (size_t)m, .m = (size_t)m, .a = rbuf, .lua_managed = false };
      tk_rvec_desc(&rv, 0, (size_t)m);
    }
    int64_t ch = 0;
    for (int64_t j = 0; j < m; j++) {
      all_cal[ps + j] = rbuf[j].d;
      ch += rbuf[j].i;
      cum_hit[ps + j] = ch;
    }
    for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++)
      if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 0;
  }
  free(bm); free(rbuf);
  double best_f1 = 0.0, best_alpha = 1.0;
  for (int ai = 1; ai <= 1000; ai++) {
    double alpha = ai * 0.01;
    uint64_t tp = 0, predicted = 0;
    for (int64_t s = 0; s < ns; s++) {
      int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
      int64_t m = pe - ps;
      if (m <= 0) continue;
      double adj_E = alpha * sample_E[s];
      double cum = 0, bf = 0;
      int64_t bk = 0;
      if (adj_E > 0) {
        for (int64_t k = 1; k <= m; k++) {
          cum += all_cal[ps + k - 1];
          double f1 = 2.0 * cum / ((double)k + adj_E);
          if (f1 > bf) { bf = f1; bk = k; }
        }
      }
      predicted += (uint64_t)bk;
      if (bk > 0) tp += (uint64_t)cum_hit[ps + bk - 1];
    }
    double prec = predicted > 0 ? (double)tp / (double)predicted : 0.0;
    double rec = total_expected > 0 ? (double)tp / (double)total_expected : 0.0;
    double f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
    if (f1 > best_f1) {
      best_f1 = f1;
      best_alpha = alpha;
    }
  }
  free(all_cal);
  free(sample_E);
  free(cum_hit);
  lua_pushnumber(L, best_alpha);
  lua_pushnumber(L, best_f1);
  return 2;
}

static int tk_gfm_score_lua (lua_State *L)
{
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  if (!g->platt_a)
    return luaL_error(L, "score: not fitted");
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
  double alpha = g->alpha;
  int64_t max_hood = 0;
  for (int64_t s = 0; s < ns; s++) {
    int64_t h = offsets->a[s + 1] - offsets->a[s];
    if (h > max_hood) max_hood = h;
  }
  uint64_t total_expected = (uint64_t)(exp_off->a[ns] - exp_off->a[0]);
  uint64_t tp = 0, predicted = 0;
  #pragma omp parallel reduction(+:tp,predicted)
  {
    uint8_t *my_bm = (uint8_t *)calloc((uint64_t)nl, sizeof(uint8_t));
    tk_rank_t *my_rbuf = (tk_rank_t *)malloc((uint64_t)max_hood * sizeof(tk_rank_t));
    #pragma omp for schedule(static)
    for (int64_t s = 0; s < ns; s++) {
      int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
      int64_t m = pe - ps;
      for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++) {
        int64_t l = exp_nbr->a[j];
        if (l >= 0 && l < nl) my_bm[l] = 1;
      }
      double E_total = 0;
      for (int64_t j = 0; j < m; j++) {
        int64_t l = neighbors->a[ps + j];
        double sc = sf ? (double)sf->a[ps + j] : sd->a[ps + j];
        double c = (l >= 0 && l < nl) ? tk_gfm_calibrate(g, l, sc) : 0.0;
        uint8_t hit = (l >= 0 && l < nl && my_bm[l]) ? 1 : 0;
        my_rbuf[j] = tk_rank(hit, c);
        E_total += c;
      }
      if (m > 0) {
        tk_rvec_t rv = { .n = (size_t)m, .m = (size_t)m, .a = my_rbuf, .lua_managed = false };
        tk_rvec_desc(&rv, 0, (size_t)m);
      }
      double adj_E = alpha * E_total;
      double cum = 0, best_f1 = 0;
      int64_t best_k = 0;
      if (adj_E > 0) {
        for (int64_t k = 1; k <= m; k++) {
          cum += my_rbuf[k - 1].d;
          double f1 = 2.0 * cum / ((double)k + adj_E);
          if (f1 > best_f1) { best_f1 = f1; best_k = k; }
        }
      }
      predicted += (uint64_t)best_k;
      for (int64_t j = 0; j < best_k; j++)
        tp += (uint64_t)my_rbuf[j].i;
      for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++) {
        int64_t l = exp_nbr->a[j];
        if (l >= 0 && l < nl) my_bm[l] = 0;
      }
    }
    free(my_bm);
    free(my_rbuf);
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

static int tk_gfm_set_alpha_lua (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  g->alpha = luaL_checknumber(L, 2);
  return 0;
}

static int tk_gfm_persist_lua (lua_State *L) {
  tk_gfm_t *g = tk_gfm_peek(L, 1);
  if (!g->platt_a)
    return luaL_error(L, "persist: not fitted");
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
  uint8_t version = 10;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &g->nl, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &g->alpha, sizeof(double), 1, fh);
  if (g->nl > 0) {
    tk_lua_fwrite(L, g->platt_a, sizeof(double), (uint64_t)g->nl, fh);
    tk_lua_fwrite(L, g->platt_b, sizeof(double), (uint64_t)g->nl, fh);
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
  if (version != 10) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported gfm version %d", (int)version);
  }
  int64_t nl;
  tk_lua_fread(L, &nl, sizeof(int64_t), 1, fh);
  double alpha;
  tk_lua_fread(L, &alpha, sizeof(double), 1, fh);
  double *platt_a = NULL, *platt_b = NULL;
  if (nl > 0) {
    platt_a = (double *)malloc((uint64_t)nl * sizeof(double));
    platt_b = (double *)malloc((uint64_t)nl * sizeof(double));
    tk_lua_fread(L, platt_a, sizeof(double), (uint64_t)nl, fh);
    tk_lua_fread(L, platt_b, sizeof(double), (uint64_t)nl, fh);
  }
  tk_lua_fclose(L, fh);
  tk_gfm_t *g = tk_lua_newuserdata(L, tk_gfm_t,
    TK_GFM_MT, tk_gfm_mt_fns, tk_gfm_gc);
  g->nl = nl;
  g->platt_a = platt_a;
  g->platt_b = platt_b;
  g->alpha = alpha;
  g->destroyed = false;
  return 1;
}

static luaL_Reg tk_gfm_mt_fns[] = {
  { "fit", tk_gfm_fit_lua },
  { "predict", tk_gfm_predict_lua },
  { "alpha_search", tk_gfm_alpha_search_lua },
  { "score", tk_gfm_score_lua },
  { "set_alpha", tk_gfm_set_alpha_lua },
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
