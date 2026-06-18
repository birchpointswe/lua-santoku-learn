#include <santoku/iuset.h>
#include <santoku/learn/span.h>
#include <santoku/ivec.h>
#include <santoku/fvec.h>
#include <math.h>
#include <float.h>
#include <string.h>

static inline int tm_regress_accuracy (lua_State *L)
{
  lua_settop(L, 2);
  tk_fvec_t *predicted_f = tk_fvec_peekopt(L, 1);
  tk_dvec_t *predicted_d = predicted_f ? NULL : tk_dvec_peek(L, 1, "predicted");
  tk_dvec_t *expected_d = tk_dvec_peekopt(L, 2);
  tk_ivec_t *expected_i = expected_d ? NULL : tk_ivec_peek(L, 2, "expected");
  uint64_t n = predicted_f ? predicted_f->n : predicted_d->n;
  if ((expected_d && expected_d->n != n) || (expected_i && expected_i->n != n))
    return luaL_error(L, "predicted and expected must have same length");
  double total = 0.0, min_err = DBL_MAX, max_err = 0.0, sum_exp = 0.0;
  #pragma omp parallel for reduction(+:total,sum_exp) reduction(min:min_err) reduction(max:max_err)
  for (uint64_t i = 0; i < n; i++) {
    double exp_val = expected_d ? expected_d->a[i] : (double)expected_i->a[i];
    double pred_val = predicted_f ? (double)predicted_f->a[i] : predicted_d->a[i];
    double err = fabs(pred_val - exp_val);
    total += err;
    sum_exp += exp_val;
    if (err < min_err) min_err = err;
    if (err > max_err) max_err = err;
  }
  double mean = n > 0 ? total / n : 0.0;
  double mean_exp = n > 0 ? sum_exp / n : 0.0;
  double nmae = mean_exp > 0 ? mean / mean_exp : 0.0;
  double var = 0.0;
  #pragma omp parallel for reduction(+:var)
  for (uint64_t i = 0; i < n; i++) {
    double exp_val = expected_d ? expected_d->a[i] : (double)expected_i->a[i];
    double pred_val = predicted_f ? (double)predicted_f->a[i] : predicted_d->a[i];
    double err = fabs(pred_val - exp_val);
    var += (err - mean) * (err - mean);
  }
  double std = n > 1 ? sqrt(var / (n - 1)) : 0.0;
  lua_newtable(L);
  lua_pushnumber(L, total);
  lua_setfield(L, -2, "total");
  lua_pushnumber(L, mean);
  lua_setfield(L, -2, "mean");
  lua_pushnumber(L, n > 0 ? min_err : 0.0);
  lua_setfield(L, -2, "min");
  lua_pushnumber(L, max_err);
  lua_setfield(L, -2, "max");
  lua_pushnumber(L, std);
  lua_setfield(L, -2, "std");
  lua_pushnumber(L, nmae);
  lua_setfield(L, -2, "nmae");
  return 1;
}


// Per-sample oracle headroom: the best micro/macro F1 achievable if an oracle picked the optimal top-k
// per sample (k sweeps 1..hood maximizing that sample's F1). An upper bound for diagnostics, NOT a
// deployable decode (that is the decider's job). Returns the oracle per-sample k vector and metrics.
static inline int tm_oracle_f1 (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "pred_offsets");
  tk_ivec_t *pred_off = tk_ivec_peek(L, -1, "pred_offsets");
  lua_getfield(L, 1, "pred_neighbors");
  tk_ivec_t *pred_nbr = tk_ivec_peek(L, -1, "pred_neighbors");
  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 4);
  uint64_t n_samples = pred_off->n - 1;
  if (exp_off->n != n_samples + 1)
    return luaL_error(L, "expected_offsets length must match sample count + 1");
  tk_ivec_t *ks = tk_ivec_create(L, n_samples);
  uint64_t mi_tp = 0, mi_k = 0, mi_exp = 0, n_valid = 0;
  double ma_prec = 0, ma_rec = 0, ma_f1 = 0;
  #pragma omp parallel for reduction(+:mi_tp,mi_k,mi_exp,n_valid,ma_prec,ma_rec,ma_f1)
  for (uint64_t s = 0; s < n_samples; s++) {
    int64_t ps = pred_off->a[s], pe = pred_off->a[s + 1];
    uint64_t hood_size = (uint64_t)(pe - ps);
    int64_t es = exp_off->a[s], ee = exp_off->a[s + 1];
    uint64_t n_expected = (uint64_t)(ee - es);
    if (n_expected == 0 || hood_size == 0) { ks->a[s] = 0; continue; }
    int kha;
    tk_iuset_t *exp_set = tk_iuset_create(NULL, 0);
    for (int64_t i = es; i < ee; i++)
      tk_iuset_put(exp_set, exp_nbr->a[i], &kha);
    double best_f1 = -1.0;
    uint64_t best_k = 1, best_tp = 0, tp = 0;
    for (uint64_t k = 1; k <= hood_size; k++) {
      if (tk_iuset_contains(exp_set, pred_nbr->a[ps + (int64_t)(k - 1)]) != 0) tp++;
      double prec = (double)tp / k;
      double rec = (double)tp / n_expected;
      double f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
      if (f1 > best_f1) { best_f1 = f1; best_k = k; best_tp = tp; }
    }
    ks->a[s] = (int64_t)best_k;
    tk_iuset_destroy(exp_set);
    mi_tp += best_tp;
    mi_k += best_k;
    mi_exp += n_expected;
    ma_prec += (double)best_tp / best_k;
    ma_rec += (double)best_tp / n_expected;
    ma_f1 += best_f1;
    n_valid++;
  }
  lua_pushvalue(L, -1);
  lua_newtable(L);
  double mi_prec = mi_k > 0 ? (double)mi_tp / mi_k : 0;
  double mi_rec = mi_exp > 0 ? (double)mi_tp / mi_exp : 0;
  double mi_f1v = (mi_prec + mi_rec) > 0 ? 2.0 * mi_prec * mi_rec / (mi_prec + mi_rec) : 0;
  lua_pushnumber(L, mi_prec); lua_setfield(L, -2, "micro_precision");
  lua_pushnumber(L, mi_rec); lua_setfield(L, -2, "micro_recall");
  lua_pushnumber(L, mi_f1v); lua_setfield(L, -2, "micro_f1");
  lua_pushnumber(L, n_valid > 0 ? ma_prec / n_valid : 0); lua_setfield(L, -2, "sample_precision");
  lua_pushnumber(L, n_valid > 0 ? ma_rec / n_valid : 0); lua_setfield(L, -2, "sample_recall");
  lua_pushnumber(L, n_valid > 0 ? ma_f1 / n_valid : 0); lua_setfield(L, -2, "sample_f1");
  return 2;
}

// span_f1(pred_off, pred_s, pred_e, pred_ty, gold_off, gold_s, gold_e, gold_ty) -> f1, p, r
// Micro exact-match PRF over spans, matched per document. A predicted span counts as a hit if a
// gold span in the same doc has the same (start, end[, type]). Types optional (nil on either side
// => boundary-only match). Spans are unique per doc, so a simple nested scan suffices.
static inline int tm_span_f1 (lua_State *L)
{
  tk_ivec_t *po = tk_ivec_peek(L, 1, "pred_offsets");
  tk_ivec_t *ps = tk_ivec_peek(L, 2, "pred_starts");
  tk_ivec_t *pe = tk_ivec_peek(L, 3, "pred_ends");
  tk_ivec_t *pty = tk_ivec_peekopt(L, 4);
  tk_ivec_t *go = tk_ivec_peek(L, 5, "gold_offsets");
  tk_ivec_t *gs = tk_ivec_peek(L, 6, "gold_starts");
  tk_ivec_t *ge = tk_ivec_peek(L, 7, "gold_ends");
  tk_ivec_t *gty = tk_ivec_peekopt(L, 8);
  int64_t n_docs = (int64_t) (po->n - 1);
  if ((int64_t) (go->n - 1) != n_docs)
    return luaL_error(L, "span_f1: pred and gold doc counts differ");
  int64_t tp = 0, npred = 0, ngold = 0;
  tk_span_counts(po->a, ps->a, pe->a, pty ? pty->a : NULL,
    go->a, gs->a, ge->a, gty ? gty->a : NULL, n_docs, &tp, &npred, &ngold);
  double p = npred > 0 ? (double) tp / (double) npred : 0.0;
  double r = ngold > 0 ? (double) tp / (double) ngold : 0.0;
  double f1 = tk_span_f1_of(tp, npred, ngold);
  lua_pushnumber(L, f1);
  lua_pushnumber(L, p);
  lua_pushnumber(L, r);
  return 3;
}

static luaL_Reg tm_evaluator_fns[] =
{
  { "regress_accuracy", tm_regress_accuracy },
  { "oracle_f1", tm_oracle_f1 },
  { "span_f1", tm_span_f1 },
  { NULL, NULL }
};

int luaopen_santoku_learn_evaluator (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_evaluator_fns, 0);
  return 1;
}
