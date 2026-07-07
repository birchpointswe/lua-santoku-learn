#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <santoku/learn/mathlibs.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/rvec.h>
#include <santoku/csr.h>
#include <santoku/spans.h>
#include <santoku/learn/fmeasure.h>
#include <santoku/span.h>
#define TK_DECIDE_MT "tk_decide_t"

typedef struct {
  int64_t nl;
  bool single;
  bool span;
  double threshold;
  double *offsets;
  double reject_offset;
  int64_t reject;
  bool destroyed;
} tk_decide_t;

static inline tk_decide_t *tk_decide_peek (lua_State *L, int i) {
  return (tk_decide_t *)luaL_checkudata(L, i, TK_DECIDE_MT);
}

static inline void tk_decide_read_pred (lua_State *L,
  tk_ivec_t **off, tk_ivec_t **nbr, tk_fvec_t **sf, tk_dvec_t **sd)
{
  lua_getfield(L, 2, "pred");
  tk_csr_t *P = tk_csr_peek(L, -1, "pred");
  lua_pop(L, 1);
  *off = P->offsets; *nbr = (tk_ivec_t *) P->neighbors;
  *sf = P->tag == TK_TAG_F32 ? (tk_fvec_t *) P->values : NULL;
  *sd = P->tag == TK_TAG_F64 ? (tk_dvec_t *) P->values : NULL;
}

static inline void tk_decide_read_expected (lua_State *L, tk_ivec_t **eoff, tk_ivec_t **enbr)
{
  lua_getfield(L, 2, "expected");
  tk_csr_t *E = tk_csr_peek(L, -1, "expected");
  lua_pop(L, 1);
  *eoff = E->offsets; *enbr = (tk_ivec_t *) E->neighbors;
}

static inline void tk_decide_read_cand (lua_State *L,
  tk_ivec_t **co, tk_ivec_t **cs, tk_ivec_t **ce, tk_ivec_t **cty)
{
  lua_getfield(L, 2, "cand");
  tk_spans_t *S = tk_spans_peek(L, -1, "cand");
  lua_pop(L, 1);
  int64_t it = tk_spans_colidx(S, "ty");
  *co = S->offsets;
  *cs = S->cols[tk_spans_colidx(S, "s")];
  *ce = S->cols[tk_spans_colidx(S, "e")];
  *cty = it >= 0 ? S->cols[it] : NULL;
}

static inline void tk_decide_read_gold (lua_State *L,
  tk_ivec_t **go, tk_ivec_t **gs, tk_ivec_t **ge, tk_ivec_t **gty)
{
  lua_getfield(L, 2, "gold");
  tk_spans_t *S = tk_spans_peek(L, -1, "gold");
  lua_pop(L, 1);
  int64_t it = tk_spans_colidx(S, "ty");
  *go = S->offsets;
  *gs = S->cols[tk_spans_colidx(S, "s")];
  *ge = S->cols[tk_spans_colidx(S, "e")];
  *gty = it >= 0 ? S->cols[it] : NULL;
}

static inline int64_t tk_decide_argmax (const float *row, const double *off, int64_t nl) {
  double bv = -HUGE_VAL;
  int64_t bc = 0;
  for (int64_t l = 0; l < nl; l++) {
    double vv = (double)row[l] - off[l];
    if (vv > bv) { bv = vv; bc = l; }
  }
  return bc;
}

static inline void tk_decide_span_topw (
  const float *S, int64_t ncand, int64_t nl, int64_t reject, double reject_offset,
  const int64_t *cty, int64_t *cls, double *w
) {
  if (nl == 1) {
    for (int64_t c = 0; c < ncand; c++) {
      double v = (double)S[c] + reject_offset;
      if (v > 0.0) { cls[c] = cty ? cty[c] : 0; w[c] = v; }
      else { cls[c] = reject; w[c] = 0.0; }
    }
    return;
  }
  for (int64_t c = 0; c < ncand; c++) {
    const float *row = S + c * nl;
    double b1 = -HUGE_VAL, b2 = -HUGE_VAL;
    int64_t a1 = 0;
    for (int64_t l = 0; l < nl; l++) {
      double v = (double)row[l] - (l == reject ? reject_offset : 0.0);
      if (v > b1) { b2 = b1; b1 = v; a1 = l; }
      else if (v > b2) { b2 = v; }
    }
    cls[c] = a1;
    w[c] = b1 - b2;
  }
}

static inline int tk_decide_gc (lua_State *L) {
  tk_decide_t *g = tk_decide_peek(L, 1);
  if (!g->destroyed) { free(g->offsets); }
  g->offsets = NULL;
  g->destroyed = true;
  return 0;
}

static luaL_Reg tk_decide_mt_fns[];

static int tk_decide_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t nl = (int64_t)tk_lua_fcheckunsigned(L, 1, "decide.create", "n_labels");
  lua_getfield(L, 1, "single");
  bool single = lua_toboolean(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 1, "span");
  bool span = lua_toboolean(L, -1);
  lua_pop(L, 1);
  lua_getfield(L, 1, "reject");
  int64_t reject = lua_isnumber(L, -1) ? (int64_t)lua_tointeger(L, -1) : nl - 1;
  lua_pop(L, 1);
  tk_decide_t *g = tk_lua_newuserdata(L, tk_decide_t,
    TK_DECIDE_MT, tk_decide_mt_fns, tk_decide_gc);
  g->nl = nl;
  g->single = single;
  g->span = span;
  g->threshold = HUGE_VAL;
  g->offsets = NULL;
  g->reject_offset = 0.0;
  g->reject = reject;
  lua_getfield(L, 1, "offset");
  if (lua_isnumber(L, -1)) {
    double off = lua_tonumber(L, -1);
    if (span) g->reject_offset = off; else g->threshold = off;
  }
  lua_pop(L, 1);
  if (single) {
    g->offsets = (double *)malloc((uint64_t)nl * sizeof(double));
    for (int64_t l = 0; l < nl; l++) g->offsets[l] = 0.0;
  }
  g->destroyed = false;
  return 1;
}

static int tk_decide_offset_lua (lua_State *L)
{
  tk_decide_t *g = tk_decide_peek(L, 1);
  lua_pushnumber(L, g->span ? g->reject_offset : g->threshold);
  return 1;
}

static int tk_decide_calibrate_multi (lua_State *L, tk_decide_t *g)
{
  tk_ivec_t *offsets, *neighbors, *exp_off, *exp_nbr;
  tk_fvec_t *sf; tk_dvec_t *sd;
  tk_decide_read_pred(L, &offsets, &neighbors, &sf, &sd);
  tk_decide_read_expected(L, &exp_off, &exp_nbr);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.calibrate", "n_samples");
  int64_t nl = g->nl;

  int64_t total_entries = offsets->a[ns] - offsets->a[0];
  uint64_t total_expected = (uint64_t)(exp_off->a[ns] - exp_off->a[0]);

  if (total_entries <= 0 || total_expected == 0) {
    g->threshold = HUGE_VAL;
    lua_pushnumber(L, 0.0);
    lua_pushnumber(L, 0.0);
    lua_pushnumber(L, 0.0);
    return 3;
  }

  uint64_t te = (uint64_t)total_entries;
  tk_rank_t *pool = (tk_rank_t *)malloc(te * sizeof(tk_rank_t));
  uint8_t *bm = (uint8_t *)calloc((uint64_t)nl, sizeof(uint8_t));
  int64_t pi = 0;
  for (int64_t s = 0; s < ns; s++) {
    for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++)
      if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 1;
    for (int64_t j = offsets->a[s]; j < offsets->a[s + 1]; j++) {
      int64_t lbl = neighbors->a[j];
      pool[pi].d = sf ? (double)sf->a[j] : sd->a[j];
      pool[pi].i = (lbl >= 0 && lbl < nl && bm[lbl]) ? 1 : 0;
      pi++;
    }
    for (int64_t j = exp_off->a[s]; j < exp_off->a[s + 1]; j++)
      if (exp_nbr->a[j] >= 0 && exp_nbr->a[j] < nl) bm[exp_nbr->a[j]] = 0;
  }
  free(bm);

  double f1, prec, rec, thr;
  tk_fmeasure_sweep(pool, te, total_expected, &f1, &prec, &rec, &thr);
  free(pool);

  g->threshold = thr;
  lua_pushnumber(L, f1);
  lua_pushnumber(L, prec);
  lua_pushnumber(L, rec);
  return 3;
}

static tk_fvec_t *tk_decide_read_scores (lua_State *L)
{
  lua_getfield(L, 2, "scores");
  tk_fvec_t *sf = tk_fvec_peek(L, -1, "scores");
  lua_pop(L, 1);
  return sf;
}

static int tk_decide_single_prf (
  const float *S, const double *off, const tk_ivec_t *exp_off, const tk_ivec_t *exp_nbr,
  int64_t n, int64_t nl, double *macro_out, double *acc_out)
{
  int64_t *gc = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
  int64_t *pc = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
  int64_t *tp = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
  if (!gc || !pc || !tp) {
    free(gc); free(pc); free(tp);
    return -1;
  }
  int64_t correct = 0, counted = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t gi = (exp_off->a[i] < exp_off->a[i + 1]) ? exp_nbr->a[exp_off->a[i]] : -1;
    int64_t bc = tk_decide_argmax(S + i * nl, off, nl);
    pc[bc]++;
    if (gi >= 0 && gi < nl) {
      gc[gi]++;
      counted++;
      if (gi == bc) { tp[bc]++; correct++; }
    }
  }
  double macro = 0.0;
  for (int64_t c = 0; c < nl; c++) {
    int64_t den = pc[c] + gc[c];
    macro += den > 0 ? 2.0 * (double)tp[c] / (double)den : 0.0;
  }
  *macro_out = macro / (double)nl;
  *acc_out = counted > 0 ? (double)correct / (double)counted : 0.0;
  free(gc); free(pc); free(tp);
  return 0;
}

static int tk_decide_calibrate_single (lua_State *L, tk_decide_t *g)
{
  tk_fvec_t *sf = tk_decide_read_scores(L);
  tk_ivec_t *exp_off, *exp_nbr;
  tk_decide_read_expected(L, &exp_off, &exp_nbr);
  int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.calibrate", "n_samples");
  int64_t nl = g->nl;
  double *off = g->offsets;
  for (int64_t l = 0; l < nl; l++) off[l] = 0.0;
  if (n <= 0 || nl <= 0) {
    lua_pushnumber(L, 0.0);
    lua_pushnumber(L, 0.0);
    return 2;
  }
  double macro, acc;
  if (tk_decide_single_prf(sf->a, off, exp_off, exp_nbr, n, nl, &macro, &acc) != 0)
    return tk_lua_verror(L, 2, "decide.calibrate", "alloc failed");
  lua_pushnumber(L, macro);
  lua_pushnumber(L, acc);
  return 2;
}

static int64_t tk_decide_emit_spans (
  const int64_t *co, const int64_t *cs, const int64_t *ce, const int64_t *cls,
  const int64_t *keep, int64_t n_docs,
  int64_t *p_off, int64_t *p_s, int64_t *p_e, int64_t *p_ty)
{
  int64_t np = 0;
  p_off[0] = 0;
  for (int64_t d = 0; d < n_docs; d++) {
    for (int64_t c = co[d]; c < co[d + 1]; c++)
      if (keep[c]) { p_s[np] = cs[c]; p_e[np] = ce[c]; p_ty[np] = cls[c]; np++; }
    p_off[d + 1] = np;
  }
  return np;
}

static void tk_decide_span_counts_at (
  const float *S, int64_t nl, int64_t reject, double reject_offset, const int64_t *cty,
  const int64_t *co, const int64_t *cs, const int64_t *ce, int64_t n_docs,
  const int64_t *go, const int64_t *gs, const int64_t *ge, const int64_t *gty,
  int64_t *cls, double *w, int64_t *keep, tk_span_iv *iv, double *M, int64_t *P,
  int64_t *p_off, int64_t *p_s, int64_t *p_e, int64_t *p_ty,
  int64_t *tp, int64_t *npred, int64_t *ngold
) {
  int64_t ncand = co[n_docs];
  tk_decide_span_topw(S, ncand, nl, reject, reject_offset, cty, cls, w);
  tk_span_nms_dp(co, cs, ce, n_docs, cls, w, reject, keep, iv, M, P);
  tk_decide_emit_spans(co, cs, ce, cls, keep, n_docs, p_off, p_s, p_e, p_ty);
  tk_span_counts(p_off, p_s, p_e, p_ty, go, gs, ge, gty, n_docs, tp, npred, ngold);
}

typedef struct { double dc; int64_t idx; } tk_decide_bp;
#define tk_decide_bp_lt(a, b) ((a).dc < (b).dc)
KSORT_INIT(tk_decide_bp, tk_decide_bp, tk_decide_bp_lt)

static void tk_decide_span_doc_at (
  int64_t reject, double v, double theta,
  const int64_t *top1cls, const double *top1, const double *s2, const double *sr, const double *dcv,
  const int64_t *cs, const int64_t *ce, int64_t cbeg, int64_t cend,
  const int64_t *gs, const int64_t *ge, const int64_t *gty, int64_t gbeg, int64_t gend,
  int64_t *cls, double *w, int64_t *keep, tk_span_iv *iv, double *M, int64_t *P,
  int64_t *tp_out, int64_t *pred_out
) {
  int64_t m = cend - cbeg;
  for (int64_t k = 0; k < m; k++) {
    int64_t i = cbeg + k;
    if (dcv[i] <= v) {
      double t2 = sr[i] - theta;
      if (s2[i] > t2) t2 = s2[i];
      cls[k] = top1cls[i]; w[k] = top1[i] - t2;
    } else {
      cls[k] = reject;
    }
  }
  int64_t co1[2] = { 0, m };
  tk_span_nms_dp(co1, cs + cbeg, ce + cbeg, 1, cls, w, reject, keep, iv, M, P);
  int64_t tp = 0, pred = 0;
  for (int64_t k = 0; k < m; k++) {
    if (!keep[k] || cls[k] == reject) continue;
    pred++;
    int64_t ps = cs[cbeg + k], pe = ce[cbeg + k], pty = cls[k];
    for (int64_t j = gbeg; j < gend; j++)
      if (ps == gs[j] && pe == ge[j] && (gty == NULL || pty == gty[j])) { tp++; break; }
  }
  *tp_out = tp; *pred_out = pred;
}

static int tk_decide_calibrate_span (lua_State *L, tk_decide_t *g)
{
  tk_fvec_t *sf = tk_decide_read_scores(L);
  tk_ivec_t *co, *cs, *ce, *cty, *go, *gs, *ge, *gty;
  tk_decide_read_cand(L, &co, &cs, &ce, &cty);
  tk_decide_read_gold(L, &go, &gs, &ge, &gty);
  int64_t n_docs = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.calibrate", "n_samples");
  int64_t nl = g->nl, reject = g->reject;
  float *S = sf->a;
  int64_t ncand = (int64_t)cs->n;
  const int64_t *gtyp = gty ? gty->a : NULL;
  const int64_t *ctyp = cty ? cty->a : NULL;
  if (ncand <= 0 || n_docs <= 0) {
    g->reject_offset = 0.0;
    lua_pushnumber(L, 0.0); lua_pushnumber(L, 0.0); lua_pushnumber(L, 0.0);
    return 3;
  }
  int64_t maxn = tk_span_max_doc(co->a, n_docs); if (maxn < 1) maxn = 1;
  double *top1 = (double *)malloc((uint64_t)ncand * sizeof(double));
  double *s2 = (double *)malloc((uint64_t)ncand * sizeof(double));
  double *sr = (double *)malloc((uint64_t)ncand * sizeof(double));
  double *dcv = (double *)malloc((uint64_t)ncand * sizeof(double));
  int64_t *top1cls = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
  int64_t *cdoc = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
  tk_decide_bp *bp = (tk_decide_bp *)malloc((uint64_t)ncand * sizeof(tk_decide_bp));
  for (int64_t c = 0; c < ncand; c++) {
    if (nl == 1) {
      double v = (double)S[c];
      top1[c] = v; s2[c] = -HUGE_VAL; top1cls[c] = ctyp ? ctyp[c] : 0; sr[c] = 0.0; dcv[c] = -v;
    } else {
      const float *row = S + c * nl;
      double b1 = -HUGE_VAL, b2 = -HUGE_VAL; int64_t a1 = 0;
      for (int64_t l = 0; l < nl; l++) {
        if (l == reject) continue;
        double v = (double)row[l];
        if (v > b1) { b2 = b1; b1 = v; a1 = l; } else if (v > b2) { b2 = v; }
      }
      top1[c] = b1; s2[c] = b2; top1cls[c] = a1; sr[c] = (double)row[reject]; dcv[c] = sr[c] - b1;
    }
    bp[c].dc = dcv[c]; bp[c].idx = c;
  }
  for (int64_t d = 0; d < n_docs; d++)
    for (int64_t c = co->a[d]; c < co->a[d + 1]; c++) cdoc[c] = d;
  ks_introsort(tk_decide_bp, (size_t)ncand, bp);

  int64_t *cls = (int64_t *)malloc((uint64_t)maxn * sizeof(int64_t));
  int64_t *keep = (int64_t *)malloc((uint64_t)maxn * sizeof(int64_t));
  double *w = (double *)malloc((uint64_t)maxn * sizeof(double));
  tk_span_iv *iv = (tk_span_iv *)malloc((uint64_t)maxn * sizeof(tk_span_iv));
  double *M = (double *)malloc((uint64_t)(maxn + 1) * sizeof(double));
  int64_t *P = (int64_t *)malloc((uint64_t)(maxn + 1) * sizeof(int64_t));
  int64_t *tp_d = (int64_t *)calloc((uint64_t)n_docs, sizeof(int64_t));
  int64_t *pred_d = (int64_t *)calloc((uint64_t)n_docs, sizeof(int64_t));
  int64_t gold_total = go->a[n_docs] - go->a[0];

  int64_t tp_sum = 0, pred_sum = 0;
  double best_off = bp[0].dc - 1.0, best_f1 = 0.0;
  int64_t k = 0;
  while (k < ncand) {
    double v = bp[k].dc;
    int64_t k2 = k; while (k2 < ncand && bp[k2].dc == v) k2++;
    double next = (k2 < ncand) ? bp[k2].dc : v + 1.0;
    double theta = 0.5 * (v + next);
    for (int64_t kk = k; kk < k2; kk++) {
      int64_t d = cdoc[bp[kk].idx];
      int64_t tpd, prd;
      tk_decide_span_doc_at(reject, v, theta, top1cls, top1, s2, sr, dcv,
        cs->a, ce->a, co->a[d], co->a[d + 1], gs->a, ge->a, gtyp, go->a[d], go->a[d + 1],
        cls, w, keep, iv, M, P, &tpd, &prd);
      tp_sum += tpd - tp_d[d]; pred_sum += prd - pred_d[d]; tp_d[d] = tpd; pred_d[d] = prd;
    }
    double f1 = tk_span_f1_of(tp_sum, pred_sum, gold_total);
    if (f1 > best_f1) { best_f1 = f1; best_off = theta; }
    k = k2;
  }
  g->reject_offset = best_off;
  free(top1); free(s2); free(sr); free(dcv); free(top1cls); free(cdoc); free(bp);
  free(cls); free(keep); free(w); free(iv); free(M); free(P); free(tp_d); free(pred_d);

  int64_t *fcls = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
  int64_t *fkeep = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
  int64_t *fps = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
  int64_t *fpe = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
  int64_t *fpty = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
  double *fw = (double *)malloc((uint64_t)ncand * sizeof(double));
  int64_t *fpoff = (int64_t *)malloc((uint64_t)(n_docs + 1) * sizeof(int64_t));
  tk_span_iv *fiv = (tk_span_iv *)malloc((uint64_t)maxn * sizeof(tk_span_iv));
  double *fM = (double *)malloc((uint64_t)(maxn + 1) * sizeof(double));
  int64_t *fP = (int64_t *)malloc((uint64_t)(maxn + 1) * sizeof(int64_t));
  int64_t tp, npred, ngold;
  tk_decide_span_counts_at(S, nl, reject, best_off, ctyp, co->a, cs->a, ce->a, n_docs, go->a, gs->a, ge->a, gtyp,
    fcls, fw, fkeep, fiv, fM, fP, fpoff, fps, fpe, fpty, &tp, &npred, &ngold);
  double prec = npred > 0 ? (double)tp / (double)npred : 0.0;
  double rec = ngold > 0 ? (double)tp / (double)ngold : 0.0;
  free(fcls); free(fkeep); free(fps); free(fpe); free(fpty); free(fw);
  free(fpoff); free(fiv); free(fM); free(fP);
  lua_pushnumber(L, tk_span_f1_of(tp, npred, ngold));
  lua_pushnumber(L, prec);
  lua_pushnumber(L, rec);
  return 3;
}

static int tk_decide_calibrate_lua (lua_State *L)
{
  tk_decide_t *g = tk_decide_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  if (g->span) return tk_decide_calibrate_span(L, g);
  if (g->single) return tk_decide_calibrate_single(L, g);
  return tk_decide_calibrate_multi(L, g);
}

static int tk_decide_predict_lua (lua_State *L)
{
  tk_decide_t *g = tk_decide_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  int64_t nl = g->nl;
  if (g->span) {
    tk_fvec_t *sf = tk_decide_read_scores(L);
    tk_ivec_t *co, *cs, *ce, *cty;
    tk_decide_read_cand(L, &co, &cs, &ce, &cty);
    int64_t n_docs = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.predict", "n_samples");
    int64_t reject = g->reject, ncand = (int64_t)cs->n;
    float *S = sf->a;
    const int64_t *ctyp = cty ? cty->a : NULL;
    int64_t *cls = (int64_t *)calloc((uint64_t)ncand, sizeof(int64_t));
    int64_t *keep = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
    double *w = (double *)calloc((uint64_t)ncand, sizeof(double));
    int64_t maxn = tk_span_max_doc(co->a, n_docs); if (maxn < 1) maxn = 1;
    tk_span_iv *iv = (tk_span_iv *)malloc((uint64_t)maxn * sizeof(tk_span_iv));
    double *M = (double *)malloc((uint64_t)(maxn + 1) * sizeof(double));
    int64_t *P = (int64_t *)malloc((uint64_t)(maxn + 1) * sizeof(int64_t));
    tk_decide_span_topw(S, ncand, nl, reject, g->reject_offset, ctyp, cls, w);
    tk_span_nms_dp(co->a, cs->a, ce->a, n_docs, cls, w, reject, keep, iv, M, P);
    int64_t np = 0;
    for (int64_t c = 0; c < ncand; c++) if (keep[c]) np++;
    tk_ivec_t *p_off = tk_ivec_create(L, (uint64_t)(n_docs + 1)); p_off->n = (uint64_t)(n_docs + 1);
    tk_ivec_t *p_s = tk_ivec_create(L, (uint64_t)np); p_s->n = (uint64_t)np;
    tk_ivec_t *p_e = tk_ivec_create(L, (uint64_t)np); p_e->n = (uint64_t)np;
    tk_ivec_t *p_ty = tk_ivec_create(L, (uint64_t)np); p_ty->n = (uint64_t)np;
    tk_decide_emit_spans(co->a, cs->a, ce->a, cls, keep, n_docs, p_off->a, p_s->a, p_e->a, p_ty->a);
    free(cls); free(keep); free(w); free(iv); free(M); free(P);
    return 4;
  }
  if (g->single) {
    tk_fvec_t *sf = tk_decide_read_scores(L);
    int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.predict", "n_samples");
    float *S = sf->a;
    double *off = g->offsets;
    tk_ivec_t *cls = tk_ivec_create(L, (uint64_t)n);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++)
      cls->a[i] = tk_decide_argmax(S + i * nl, off, nl);
    return 1;
  }
  tk_ivec_t *offsets, *neighbors;
  tk_fvec_t *sf; tk_dvec_t *sd;
  tk_decide_read_pred(L, &offsets, &neighbors, &sf, &sd);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.predict", "n_samples");
  double thr = g->threshold;
  tk_ivec_t *ks = tk_ivec_create(L, (uint64_t)ns);
  #pragma omp parallel for schedule(static)
  for (int64_t s = 0; s < ns; s++) {
    int64_t ps = offsets->a[s], pe = offsets->a[s + 1];
    int64_t k = 0;
    for (int64_t j = ps; j < pe; j++) {
      int64_t l = neighbors->a[j];
      if (l >= 0 && l < nl) {
        double sc = sf ? (double)sf->a[j] : sd->a[j];
        if (sc >= thr) k++;
      }
    }
    ks->a[s] = k;
  }
  return 1;
}

static int tk_decide_score_lua (lua_State *L)
{
  tk_decide_t *g = tk_decide_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  int64_t nl = g->nl;
  if (g->span) {
    tk_fvec_t *sf = tk_decide_read_scores(L);
    tk_ivec_t *co, *cs, *ce, *cty, *go, *gs, *ge, *gty;
    tk_decide_read_cand(L, &co, &cs, &ce, &cty);
    tk_decide_read_gold(L, &go, &gs, &ge, &gty);
    int64_t n_docs = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.score", "n_samples");
    int64_t reject = g->reject, ncand = (int64_t)cs->n;
    float *S = sf->a;
    const int64_t *gtyp = gty ? gty->a : NULL;
    const int64_t *ctyp = cty ? cty->a : NULL;
    int64_t *cls = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
    int64_t *keep = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
    int64_t *p_s = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
    int64_t *p_e = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
    int64_t *p_ty = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
    double *w = (double *)malloc((uint64_t)ncand * sizeof(double));
    int64_t *p_off = (int64_t *)malloc((uint64_t)(n_docs + 1) * sizeof(int64_t));
    int64_t maxn = tk_span_max_doc(co->a, n_docs); if (maxn < 1) maxn = 1;
    tk_span_iv *iv = (tk_span_iv *)malloc((uint64_t)maxn * sizeof(tk_span_iv));
    double *M = (double *)malloc((uint64_t)(maxn + 1) * sizeof(double));
    int64_t *P = (int64_t *)malloc((uint64_t)(maxn + 1) * sizeof(int64_t));
    int64_t tp, npred, ngold;
    tk_decide_span_counts_at(S, nl, reject, g->reject_offset, ctyp, co->a, cs->a, ce->a, n_docs,
      go->a, gs->a, ge->a, gtyp, cls, w, keep, iv, M, P, p_off, p_s, p_e, p_ty, &tp, &npred, &ngold);
    double prec = npred > 0 ? (double)tp / (double)npred : 0.0;
    double rec = ngold > 0 ? (double)tp / (double)ngold : 0.0;
    double f1 = tk_span_f1_of(tp, npred, ngold);
    free(cls); free(keep); free(p_s); free(p_e); free(p_ty); free(w);
    free(p_off); free(iv); free(M); free(P);
    lua_pushnumber(L, f1);
    lua_newtable(L);
    lua_pushnumber(L, prec); lua_setfield(L, -2, "precision");
    lua_pushnumber(L, rec); lua_setfield(L, -2, "recall");
    lua_pushnumber(L, f1); lua_setfield(L, -2, "span_f1");
    return 2;
  }
  if (g->single) {
    tk_fvec_t *sf = tk_decide_read_scores(L);
    tk_ivec_t *exp_off, *exp_nbr;
    tk_decide_read_expected(L, &exp_off, &exp_nbr);
    int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.score", "n_samples");
    double macro, acc;
    if (tk_decide_single_prf(sf->a, g->offsets, exp_off, exp_nbr, n, nl, &macro, &acc) != 0)
      return tk_lua_verror(L, 2, "decide.score", "alloc failed");
    lua_pushnumber(L, macro);
    lua_newtable(L);
    lua_pushnumber(L, macro); lua_setfield(L, -2, "macro_f1");
    lua_pushnumber(L, acc); lua_setfield(L, -2, "accuracy");
    return 2;
  }
  tk_ivec_t *offsets, *neighbors, *exp_off, *exp_nbr;
  tk_fvec_t *sf; tk_dvec_t *sd;
  tk_decide_read_pred(L, &offsets, &neighbors, &sf, &sd);
  tk_decide_read_expected(L, &exp_off, &exp_nbr);
  int64_t ns = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.score", "n_samples");
  double thr = g->threshold;
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
      total_expected += (uint64_t)(ee - es);
      for (int64_t j = es; j < ee; j++) {
        int64_t l = exp_nbr->a[j];
        if (l >= 0 && l < nl) my_bm[l] = 1;
      }
      int64_t k = 0;
      for (int64_t j = ps; j < pe; j++) {
        int64_t l = neighbors->a[j];
        if (l >= 0 && l < nl) {
          double sc = sf ? (double)sf->a[j] : sd->a[j];
          if (sc >= thr) k++;
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

static int tk_decide_persist_lua (lua_State *L)
{
  tk_decide_t *g = tk_decide_peek(L, 1);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKde", 1, 4, fh);
  uint8_t version = 2;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  uint8_t mode = g->span ? 2 : (g->single ? 1 : 0);
  tk_lua_fwrite(L, &mode, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &g->nl, sizeof(int64_t), 1, fh);
  if (g->span) {
    tk_lua_fwrite(L, &g->reject, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, &g->reject_offset, sizeof(double), 1, fh);
  } else if (g->single) {
    tk_lua_fwrite(L, g->offsets, sizeof(double), (size_t)g->nl, fh);
  } else {
    tk_lua_fwrite(L, &g->threshold, sizeof(double), 1, fh);
  }
  tk_lua_fclose(L, fh);
  return 0;
}

static int tk_decide_load_lua (lua_State *L)
{
  const char *data = luaL_checkstring(L, 1);
  FILE *fh = tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKde", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid decide file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 2) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported decide version %d", (int)version);
  }
  uint8_t mode;
  tk_lua_fread(L, &mode, sizeof(uint8_t), 1, fh);
  int64_t nl;
  tk_lua_fread(L, &nl, sizeof(int64_t), 1, fh);
  tk_decide_t *g = tk_lua_newuserdata(L, tk_decide_t,
    TK_DECIDE_MT, tk_decide_mt_fns, tk_decide_gc);
  g->nl = nl;
  g->single = (mode == 1);
  g->span = (mode == 2);
  g->threshold = HUGE_VAL;
  g->offsets = NULL;
  g->reject_offset = 0.0;
  g->reject = nl - 1;
  if (g->span) {
    tk_lua_fread(L, &g->reject, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, &g->reject_offset, sizeof(double), 1, fh);
  } else if (g->single) {
    g->offsets = (double *)malloc((uint64_t)nl * sizeof(double));
    tk_lua_fread(L, g->offsets, sizeof(double), (size_t)nl, fh);
  } else {
    tk_lua_fread(L, &g->threshold, sizeof(double), 1, fh);
  }
  tk_lua_fclose(L, fh);
  g->destroyed = false;
  return 1;
}

static luaL_Reg tk_decide_mt_fns[] = {
  { "calibrate", tk_decide_calibrate_lua },
  { "predict", tk_decide_predict_lua },
  { "score", tk_decide_score_lua },
  { "offset", tk_decide_offset_lua },
  { "persist", tk_decide_persist_lua },
  { NULL, NULL }
};

static luaL_Reg tk_decide_fns[] = {
  { "create", tk_decide_create_lua },
  { "load", tk_decide_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_decide (lua_State *L)
{
  tk_lua_require_mod(L, "santoku.csr");
  lua_newtable(L);
  tk_lua_register(L, tk_decide_fns, 0);
  return 1;
}
