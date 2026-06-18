#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <santoku/learn/mathlibs.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <santoku/rvec.h>
#include <santoku/learn/fmeasure.h>
#include <santoku/learn/span.h>
#include <santoku/learn/decide.h>

// single-label decode of one score row: argmax_l (score_l - offset_l), first index wins ties.
static inline int64_t tk_decide_argmax (const float *row, const double *off, int64_t nl) {
  double bv = -HUGE_VAL;
  int64_t bc = 0;
  for (int64_t l = 0; l < nl; l++) {
    double vv = (double)row[l] - off[l];
    if (vv > bv) { bv = vv; bc = l; }
  }
  return bc;
}

// span decode: per candidate, the argmax class and the top1-top2 margin of (score - reject_offset on the
// REJECT column). Feeds tk_span_nms_dp (cls=argmax, w=margin). reject_offset>0 suppresses REJECT.
static inline void tk_decide_span_topw (
  const float *S, int64_t ncand, int64_t nl, int64_t reject, double reject_offset,
  int64_t *cls, double *w
) {
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
  if (!g->destroyed) free(g->offsets);
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
  if (single) {
    g->offsets = (double *)malloc((uint64_t)nl * sizeof(double));
    for (int64_t l = 0; l < nl; l++) g->offsets[l] = 0.0;
  }
  g->destroyed = false;
  return 1;
}

// ===== multilabel: one global score threshold, micro-F1 sweep =====

static int tk_decide_calibrate_multi (lua_State *L, tk_decide_t *g)
{
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

// ===== single-label: per-label additive offsets, decode argmax(score_l - offset_l) =====
// Offsets are fit by coordinate ascent on macro-F1: holding the other offsets fixed, each label's
// offset is the cut over the sorted decision margins that maximizes macro-F1 (sweep, no black-box
// search). One global scalar would be a degenerate accuracy threshold here; macro-F1 rewards the
// rare classes the offsets exist to rescue.

static int tk_decide_calibrate_single (lua_State *L, tk_decide_t *g)
{
  lua_getfield(L, 2, "scores");
  tk_fvec_t *sf = tk_fvec_peek(L, -1, "scores");
  lua_getfield(L, 2, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 2, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 3);
  int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.calibrate", "n_samples");
  int64_t nl = g->nl;
  float *S = sf->a;
  double *off = g->offsets;
  for (int64_t l = 0; l < nl; l++) off[l] = 0.0;
  if (n <= 0 || nl <= 0) {
    lua_pushnumber(L, 0.0);
    lua_pushnumber(L, 0.0);
    return 2;
  }

  int64_t *gold = (int64_t *)malloc((uint64_t)n * sizeof(int64_t));
  int64_t *gc = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));   // gold count per class
  for (int64_t i = 0; i < n; i++) {
    int64_t gi = (exp_off->a[i] < exp_off->a[i + 1]) ? exp_nbr->a[exp_off->a[i]] : -1;
    gold[i] = gi;
    if (gi >= 0 && gi < nl) gc[gi]++;
  }
  int64_t *argother = (int64_t *)malloc((uint64_t)n * sizeof(int64_t)); // argmax over classes != l
  int64_t *pc = (int64_t *)malloc((uint64_t)nl * sizeof(int64_t));      // predicted count per class
  int64_t *tp = (int64_t *)malloc((uint64_t)nl * sizeof(int64_t));      // true positives per class
  double *f1 = (double *)malloc((uint64_t)nl * sizeof(double));
  tk_rank_t *pool = (tk_rank_t *)malloc((uint64_t)n * sizeof(tk_rank_t));
  int64_t correct = 0, counted = 0;

  double prev_macro = -1.0;
  for (int pass = 0; pass < 50; pass++) {
    for (int64_t l = 0; l < nl; l++) {
      // per-sample margin d_i = score_il - max_{j!=l}(score_ij - off_j); turning off_l below d_i
      // flips sample i's prediction from argother[i] to l.
      for (int64_t i = 0; i < n; i++) {
        const float *row = S + i * nl;
        double bo = -HUGE_VAL;
        int64_t ao = -1;
        for (int64_t j = 0; j < nl; j++) {
          if (j == l) continue;
          double vv = (double)row[j] - off[j];
          if (vv > bo) { bo = vv; ao = j; }
        }
        argother[i] = ao;
        pool[i].i = i;
        pool[i].d = (double)row[l] - bo;
      }
      // base state: off_l = +inf, l never predicted, every sample predicts argother[i]
      for (int64_t c = 0; c < nl; c++) { pc[c] = 0; tp[c] = 0; }
      for (int64_t i = 0; i < n; i++) {
        int64_t p = argother[i];
        if (p >= 0) { pc[p]++; if (gold[i] == p) tp[p]++; }
      }
      double macro_sum = 0.0;
      for (int64_t c = 0; c < nl; c++) {
        int64_t den = pc[c] + gc[c];
        f1[c] = den > 0 ? 2.0 * (double)tp[c] / (double)den : 0.0;
        macro_sum += f1[c];
      }
      tk_rvec_t v = { .n = (size_t)n, .m = (size_t)n, .lua_managed = false, .a = pool };
      tk_rvec_desc(&v, 0, (size_t)n);
      double best_macro = macro_sum;
      double best_tau = pool[0].d + 1.0;   // m = 0: offset above every margin, suppress l
      for (int64_t m = 1; m <= n; m++) {
        int64_t i = pool[m - 1].i;
        int64_t oldc = argother[i];
        macro_sum -= f1[l];
        if (oldc >= 0) macro_sum -= f1[oldc];
        if (oldc >= 0) { pc[oldc]--; if (gold[i] == oldc) tp[oldc]--; }
        pc[l]++; if (gold[i] == l) tp[l]++;
        { int64_t den = pc[l] + gc[l]; f1[l] = den > 0 ? 2.0 * (double)tp[l] / (double)den : 0.0; macro_sum += f1[l]; }
        if (oldc >= 0) { int64_t den = pc[oldc] + gc[oldc]; f1[oldc] = den > 0 ? 2.0 * (double)tp[oldc] / (double)den : 0.0; macro_sum += f1[oldc]; }
        // only score at a realizable cut: a single threshold cannot split a block of equal margins
        if (m == n || pool[m - 1].d != pool[m].d) {
          if (macro_sum > best_macro + 1e-12) {
            best_macro = macro_sum;
            best_tau = (m == n) ? pool[n - 1].d : (pool[m - 1].d + pool[m].d) / 2.0;
          }
        }
      }
      off[l] = best_tau;
    }
    // macro-F1 (+ accuracy) with the committed offsets; the last pass's values are the final metrics.
    for (int64_t c = 0; c < nl; c++) { pc[c] = 0; tp[c] = 0; }
    correct = 0; counted = 0;
    for (int64_t i = 0; i < n; i++) {
      int64_t bc = tk_decide_argmax(S + i * nl, off, nl);
      pc[bc]++;
      if (gold[i] >= 0) { counted++; if (gold[i] == bc) { tp[bc]++; correct++; } }
    }
    double macro = 0.0;
    for (int64_t c = 0; c < nl; c++) {
      int64_t den = pc[c] + gc[c];
      macro += den > 0 ? 2.0 * (double)tp[c] / (double)den : 0.0;
    }
    macro /= (double)nl;
    if (macro <= prev_macro + 1e-12) { prev_macro = macro; break; }
    prev_macro = macro;
  }
  double acc = counted > 0 ? (double)correct / (double)counted : 0.0;

  free(gold); free(gc); free(argother); free(pc); free(tp); free(f1); free(pool);
  lua_pushnumber(L, prev_macro < 0 ? 0.0 : prev_macro);
  lua_pushnumber(L, acc);
  return 2;
}

// ===== span: per-candidate argmax with a REJECT offset, resolved by nms_dp into non-overlapping spans =====
// REJECT-class decision offset is the precision/recall knob: offset>0 suppresses REJECT -> more candidates
// survive -> higher recall. Calibrated on dev by golden-section over span-F1 (unimodal in the offset).

// Decode candidates at reject_offset and count micro span PRF vs gold. All scratch is caller-owned.
static void tk_decide_span_counts_at (
  const float *S, int64_t nl, int64_t reject, double reject_offset,
  const int64_t *co, const int64_t *cs, const int64_t *ce, int64_t n_docs,
  const int64_t *go, const int64_t *gs, const int64_t *ge, const int64_t *gty,
  int64_t *cls, double *w, int64_t *keep, tk_span_iv *iv, double *M, int64_t *P,
  int64_t *p_off, int64_t *p_s, int64_t *p_e, int64_t *p_ty,
  int64_t *tp, int64_t *npred, int64_t *ngold
) {
  int64_t ncand = co[n_docs];
  tk_decide_span_topw(S, ncand, nl, reject, reject_offset, cls, w);
  tk_span_nms_dp(co, cs, ce, n_docs, cls, w, reject, keep, iv, M, P);
  int64_t np = 0;
  p_off[0] = 0;
  for (int64_t d = 0; d < n_docs; d++) {
    for (int64_t c = co[d]; c < co[d + 1]; c++)
      if (keep[c]) { p_s[np] = cs[c]; p_e[np] = ce[c]; p_ty[np] = cls[c]; np++; }
    p_off[d + 1] = np;
  }
  tk_span_counts(p_off, p_s, p_e, p_ty, go, gs, ge, gty, n_docs, tp, npred, ngold);
}

static int tk_decide_calibrate_span (lua_State *L, tk_decide_t *g)
{
  lua_getfield(L, 2, "scores"); tk_fvec_t *sf = tk_fvec_peek(L, -1, "scores");
  lua_getfield(L, 2, "cand_offsets"); tk_ivec_t *co = tk_ivec_peek(L, -1, "cand_offsets");
  lua_getfield(L, 2, "cand_starts"); tk_ivec_t *cs = tk_ivec_peek(L, -1, "cand_starts");
  lua_getfield(L, 2, "cand_ends"); tk_ivec_t *ce = tk_ivec_peek(L, -1, "cand_ends");
  lua_getfield(L, 2, "expected_offsets"); tk_ivec_t *go = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 2, "expected_starts"); tk_ivec_t *gs = tk_ivec_peek(L, -1, "expected_starts");
  lua_getfield(L, 2, "expected_ends"); tk_ivec_t *ge = tk_ivec_peek(L, -1, "expected_ends");
  lua_getfield(L, 2, "expected_types"); tk_ivec_t *gty = tk_ivec_peekopt(L, -1);
  lua_pop(L, 8);
  int64_t n_docs = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.calibrate", "n_samples");
  int64_t nl = g->nl, reject = g->reject;
  float *S = sf->a;
  int64_t ncand = (int64_t)cs->n;
  const int64_t *gtyp = gty ? gty->a : NULL;
  if (ncand <= 0 || n_docs <= 0) {
    g->reject_offset = 0.0;
    lua_pushnumber(L, 0.0); lua_pushnumber(L, 0.0); lua_pushnumber(L, 0.0);
    return 3;
  }
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

  // data-driven offset range: at offset > d_c = score[c,reject] - max_{l!=reject} score[c,l], candidate c
  // stops being REJECT. Sweeping [min d_c, max d_c] covers every flip point.
  double dmin = HUGE_VAL, dmax = -HUGE_VAL;
  for (int64_t c = 0; c < ncand; c++) {
    const float *row = S + c * nl;
    double mo = -HUGE_VAL;
    for (int64_t l = 0; l < nl; l++)
      if (l != reject && (double)row[l] > mo) mo = (double)row[l];
    double dc = (double)row[reject] - mo;
    if (dc < dmin) dmin = dc;
    if (dc > dmax) dmax = dc;
  }
  int64_t tp, npred, ngold;
  double pad = (dmax - dmin) * 0.01 + 1e-6;   // reach the keep-none / keep-all extremes past the ties
  double lo = dmin - pad, hi = dmax + pad;
  double tol = (hi - lo) * 1e-3; if (tol <= 0) tol = 1e-6;
  const double phi = 0.6180339887498949;
  double best_off = lo, best_f1 = -1.0;
  double c1 = hi - phi * (hi - lo), c2 = lo + phi * (hi - lo);
  tk_decide_span_counts_at(S, nl, reject, c1, co->a, cs->a, ce->a, n_docs, go->a, gs->a, ge->a, gtyp,
    cls, w, keep, iv, M, P, p_off, p_s, p_e, p_ty, &tp, &npred, &ngold);
  double f1c1 = tk_span_f1_of(tp, npred, ngold);
  tk_decide_span_counts_at(S, nl, reject, c2, co->a, cs->a, ce->a, n_docs, go->a, gs->a, ge->a, gtyp,
    cls, w, keep, iv, M, P, p_off, p_s, p_e, p_ty, &tp, &npred, &ngold);
  double f1c2 = tk_span_f1_of(tp, npred, ngold);
  if (f1c1 > best_f1) { best_f1 = f1c1; best_off = c1; }
  if (f1c2 > best_f1) { best_f1 = f1c2; best_off = c2; }
  while ((hi - lo) > tol) {
    if (f1c1 >= f1c2) {
      hi = c2; c2 = c1; f1c2 = f1c1;
      c1 = hi - phi * (hi - lo);
      tk_decide_span_counts_at(S, nl, reject, c1, co->a, cs->a, ce->a, n_docs, go->a, gs->a, ge->a, gtyp,
        cls, w, keep, iv, M, P, p_off, p_s, p_e, p_ty, &tp, &npred, &ngold);
      f1c1 = tk_span_f1_of(tp, npred, ngold);
      if (f1c1 > best_f1) { best_f1 = f1c1; best_off = c1; }
    } else {
      lo = c1; c1 = c2; f1c1 = f1c2;
      c2 = lo + phi * (hi - lo);
      tk_decide_span_counts_at(S, nl, reject, c2, co->a, cs->a, ce->a, n_docs, go->a, gs->a, ge->a, gtyp,
        cls, w, keep, iv, M, P, p_off, p_s, p_e, p_ty, &tp, &npred, &ngold);
      f1c2 = tk_span_f1_of(tp, npred, ngold);
      if (f1c2 > best_f1) { best_f1 = f1c2; best_off = c2; }
    }
  }
  g->reject_offset = best_off;
  tk_decide_span_counts_at(S, nl, reject, best_off, co->a, cs->a, ce->a, n_docs, go->a, gs->a, ge->a, gtyp,
    cls, w, keep, iv, M, P, p_off, p_s, p_e, p_ty, &tp, &npred, &ngold);
  double prec = npred > 0 ? (double)tp / (double)npred : 0.0;
  double rec = ngold > 0 ? (double)tp / (double)ngold : 0.0;
  free(cls); free(keep); free(p_s); free(p_e); free(p_ty); free(w);
  free(p_off); free(iv); free(M); free(P);
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
    lua_getfield(L, 2, "scores"); tk_fvec_t *sf = tk_fvec_peek(L, -1, "scores");
    lua_getfield(L, 2, "cand_offsets"); tk_ivec_t *co = tk_ivec_peek(L, -1, "cand_offsets");
    lua_getfield(L, 2, "cand_starts"); tk_ivec_t *cs = tk_ivec_peek(L, -1, "cand_starts");
    lua_getfield(L, 2, "cand_ends"); tk_ivec_t *ce = tk_ivec_peek(L, -1, "cand_ends");
    lua_pop(L, 4);
    int64_t n_docs = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.predict", "n_samples");
    int64_t reject = g->reject, ncand = (int64_t)cs->n;
    float *S = sf->a;
    int64_t *cls = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
    int64_t *keep = (int64_t *)malloc((uint64_t)ncand * sizeof(int64_t));
    double *w = (double *)malloc((uint64_t)ncand * sizeof(double));
    int64_t maxn = tk_span_max_doc(co->a, n_docs); if (maxn < 1) maxn = 1;
    tk_span_iv *iv = (tk_span_iv *)malloc((uint64_t)maxn * sizeof(tk_span_iv));
    double *M = (double *)malloc((uint64_t)(maxn + 1) * sizeof(double));
    int64_t *P = (int64_t *)malloc((uint64_t)(maxn + 1) * sizeof(int64_t));
    tk_decide_span_topw(S, ncand, nl, reject, g->reject_offset, cls, w);
    tk_span_nms_dp(co->a, cs->a, ce->a, n_docs, cls, w, reject, keep, iv, M, P);
    int64_t np = 0;
    for (int64_t c = 0; c < ncand; c++) if (keep[c]) np++;
    tk_ivec_t *p_off = tk_ivec_create(L, (uint64_t)(n_docs + 1)); p_off->n = (uint64_t)(n_docs + 1);
    tk_ivec_t *p_s = tk_ivec_create(L, (uint64_t)np); p_s->n = (uint64_t)np;
    tk_ivec_t *p_e = tk_ivec_create(L, (uint64_t)np); p_e->n = (uint64_t)np;
    tk_ivec_t *p_ty = tk_ivec_create(L, (uint64_t)np); p_ty->n = (uint64_t)np;
    int64_t j = 0;
    p_off->a[0] = 0;
    for (int64_t d = 0; d < n_docs; d++) {
      for (int64_t c = co->a[d]; c < co->a[d + 1]; c++)
        if (keep[c]) { p_s->a[j] = cs->a[c]; p_e->a[j] = ce->a[c]; p_ty->a[j] = cls[c]; j++; }
      p_off->a[d + 1] = j;
    }
    free(cls); free(keep); free(w); free(iv); free(M); free(P);
    return 4;
  }
  if (g->single) {
    lua_getfield(L, 2, "scores");
    tk_fvec_t *sf = tk_fvec_peek(L, -1, "scores");
    lua_pop(L, 1);
    int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.predict", "n_samples");
    float *S = sf->a;
    double *off = g->offsets;
    tk_ivec_t *cls = tk_ivec_create(L, (uint64_t)n);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++)
      cls->a[i] = tk_decide_argmax(S + i * nl, off, nl);
    return 1;
  }
  lua_getfield(L, 2, "offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
  lua_getfield(L, 2, "neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "neighbors");
  lua_getfield(L, 2, "scores");
  tk_fvec_t *sf = tk_fvec_peekopt(L, -1);
  tk_dvec_t *sd = sf ? NULL : tk_dvec_peek(L, -1, "scores");
  lua_pop(L, 3);
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
    lua_getfield(L, 2, "scores"); tk_fvec_t *sf = tk_fvec_peek(L, -1, "scores");
    lua_getfield(L, 2, "cand_offsets"); tk_ivec_t *co = tk_ivec_peek(L, -1, "cand_offsets");
    lua_getfield(L, 2, "cand_starts"); tk_ivec_t *cs = tk_ivec_peek(L, -1, "cand_starts");
    lua_getfield(L, 2, "cand_ends"); tk_ivec_t *ce = tk_ivec_peek(L, -1, "cand_ends");
    lua_getfield(L, 2, "expected_offsets"); tk_ivec_t *go = tk_ivec_peek(L, -1, "expected_offsets");
    lua_getfield(L, 2, "expected_starts"); tk_ivec_t *gs = tk_ivec_peek(L, -1, "expected_starts");
    lua_getfield(L, 2, "expected_ends"); tk_ivec_t *ge = tk_ivec_peek(L, -1, "expected_ends");
    lua_getfield(L, 2, "expected_types"); tk_ivec_t *gty = tk_ivec_peekopt(L, -1);
    lua_pop(L, 8);
    int64_t n_docs = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.score", "n_samples");
    int64_t reject = g->reject, ncand = (int64_t)cs->n;
    float *S = sf->a;
    const int64_t *gtyp = gty ? gty->a : NULL;
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
    tk_decide_span_counts_at(S, nl, reject, g->reject_offset, co->a, cs->a, ce->a, n_docs,
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
    lua_getfield(L, 2, "scores");
    tk_fvec_t *sf = tk_fvec_peek(L, -1, "scores");
    lua_getfield(L, 2, "expected_offsets");
    tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
    lua_getfield(L, 2, "expected_neighbors");
    tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
    lua_pop(L, 3);
    int64_t n = (int64_t)tk_lua_fcheckunsigned(L, 2, "decide.score", "n_samples");
    float *S = sf->a;
    double *off = g->offsets;
    int64_t *pc = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
    int64_t *tp = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
    int64_t *gc = (int64_t *)calloc((uint64_t)nl, sizeof(int64_t));
    int64_t correct = 0, counted = 0;
    for (int64_t i = 0; i < n; i++) {
      int64_t gi = (exp_off->a[i] < exp_off->a[i + 1]) ? exp_nbr->a[exp_off->a[i]] : -1;
      int64_t bc = tk_decide_argmax(S + i * nl, off, nl);
      pc[bc]++;
      if (gi >= 0 && gi < nl) {
        gc[gi]++;
        counted++;
        if (bc == gi) { tp[bc]++; correct++; }
      }
    }
    double macro = 0.0;
    for (int64_t c = 0; c < nl; c++) {
      int64_t den = pc[c] + gc[c];
      macro += den > 0 ? 2.0 * (double)tp[c] / (double)den : 0.0;
    }
    macro /= (double)nl;
    double acc = counted > 0 ? (double)correct / (double)counted : 0.0;
    free(pc); free(tp); free(gc);
    lua_pushnumber(L, macro);
    lua_newtable(L);
    lua_pushnumber(L, macro); lua_setfield(L, -2, "macro_f1");
    lua_pushnumber(L, acc); lua_setfield(L, -2, "accuracy");
    return 2;
  }
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
  uint8_t mode = g->span ? 2 : (g->single ? 1 : 0);   // 0=multilabel, 1=single, 2=span
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
  lua_newtable(L);
  tk_lua_register(L, tk_decide_fns, 0);
  return 1;
}
