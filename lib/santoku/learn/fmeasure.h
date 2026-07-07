#ifndef TK_LEARN_FMEASURE_H
#define TK_LEARN_FMEASURE_H

#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <santoku/rvec.h>

static inline void tk_fmeasure_sweep (
  tk_rank_t *pool,
  size_t n,
  uint64_t total_expected,
  double *f1,
  double *prec,
  double *rec,
  double *threshold
) {
  tk_rvec_t v = { .n = n, .m = n, .lua_managed = false, .a = pool };
  tk_rvec_desc(&v, 0, n);
  uint64_t tp = 0, best_tp = 0;
  int64_t best_k = 0;
  double best_f1 = 0.0;
  for (size_t i = 0; i < n; i++) {
    tp += (uint64_t)pool[i].i;
    double f1_i = 2.0 * (double)tp / ((double)(i + 1) + (double)total_expected);
    if (f1_i > best_f1) { best_f1 = f1_i; best_tp = tp; best_k = (int64_t)i + 1; }
  }
  if (f1) *f1 = best_f1;
  if (prec) *prec = best_k > 0 ? (double)best_tp / (double)best_k : 0.0;
  if (rec) *rec = total_expected > 0 ? (double)best_tp / (double)total_expected : 0.0;
  if (threshold) {
    if (best_k == 0) *threshold = HUGE_VAL;
    else if (best_k == (int64_t)n) *threshold = pool[n - 1].d;
    else *threshold = (pool[(size_t)best_k - 1].d + pool[(size_t)best_k].d) / 2.0;
  }
}

#endif
