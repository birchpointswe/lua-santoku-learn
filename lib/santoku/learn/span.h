#ifndef TK_LEARN_SPAN_H
#define TK_LEARN_SPAN_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <santoku/klib.h>

// Shared span-decode cores: the weighted-interval-scheduling NMS and the micro span-PRF match, used by
// csr.nms_dp / evaluator.span_f1 and by the span decider (decide.c). Keeping one copy each.

typedef struct { int64_t s, e, oi; double w; } tk_span_iv;

#define tk_span_iv_lt(a, b) ((a).e < (b).e)
KSORT_INIT(tk_span_iv_s, tk_span_iv, tk_span_iv_lt)

static inline void tk_span_iv_suppress (void) {
  (void) ks_mergesort_tk_span_iv_s;
  (void) ks_heapmake_tk_span_iv_s;
  (void) ks_heapsort_tk_span_iv_s;
  (void) ks_ksmall_tk_span_iv_s;
  (void) ks_shuffle_tk_span_iv_s;
}

// Per-doc weighted interval scheduling over the non-reject candidates. cls[c] = candidate's predicted
// class, w[c] = its DP weight (>= 0, the top1-top2 margin). Fills keep[c] in {0,1} with the optimal
// max-total-weight set of non-overlapping (end[j] <= start[i]) non-reject candidates. iv/M/P are caller
// scratch sized >= the largest doc's candidate count (+1 for M/P).
static inline void tk_span_nms_dp (
  const int64_t *co, const int64_t *cs, const int64_t *ce, int64_t n_docs,
  const int64_t *cls, const double *w, int64_t reject,
  int64_t *keep, tk_span_iv *iv, double *M, int64_t *P
) {
  int64_t ncand = co[n_docs];
  for (int64_t c = 0; c < ncand; c++) keep[c] = 0;
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t m = 0;
    for (int64_t c = co[d]; c < co[d + 1]; c++) {
      if (cls[c] == reject) continue;
      iv[m].s = cs[c]; iv[m].e = ce[c]; iv[m].oi = c; iv[m].w = w[c];
      m++;
    }
    if (m == 0) continue;
    ks_introsort(tk_span_iv_s, (size_t) m, iv);
    M[0] = 0.0;
    for (int64_t i = 1; i <= m; i++) {
      int64_t target = iv[i - 1].s, lo = 0, hi = i - 1;
      while (lo < hi) { int64_t mid = (lo + hi) / 2; if (iv[mid].e <= target) lo = mid + 1; else hi = mid; }
      P[i] = lo;
      double take = iv[i - 1].w + M[P[i]];
      M[i] = (take >= M[i - 1]) ? take : M[i - 1];
    }
    int64_t i = m;
    while (i >= 1) {
      double take = iv[i - 1].w + M[P[i]];
      if (take >= M[i - 1]) { keep[iv[i - 1].oi] = 1; i = P[i]; }
      else i = i - 1;
    }
  }
}

// Largest doc candidate count, for sizing the nms_dp scratch.
static inline int64_t tk_span_max_doc (const int64_t *co, int64_t n_docs) {
  int64_t maxn = 0;
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t n = co[d + 1] - co[d];
    if (n > maxn) maxn = n;
  }
  return maxn;
}

// Micro exact-match span PRF, matched per doc: a predicted span is a hit if a gold span in the same doc
// has the same (start, end[, type]). pty/gty NULL => boundary-only match. Spans/doc are unique & small.
static inline void tk_span_counts (
  const int64_t *po, const int64_t *ps, const int64_t *pe, const int64_t *pty,
  const int64_t *go, const int64_t *gs, const int64_t *ge, const int64_t *gty,
  int64_t n_docs, int64_t *tp_out, int64_t *npred_out, int64_t *ngold_out
) {
  bool use_ty = pty && gty;
  int64_t tp = 0, npred = 0, ngold = 0;
  for (int64_t d = 0; d < n_docs; d++) {
    int64_t p0 = po[d], p1 = po[d + 1];
    int64_t g0 = go[d], g1 = go[d + 1];
    ngold += g1 - g0;
    for (int64_t i = p0; i < p1; i++) {
      npred++;
      for (int64_t j = g0; j < g1; j++) {
        if (ps[i] == gs[j] && pe[i] == ge[j] && (!use_ty || pty[i] == gty[j])) { tp++; break; }
      }
    }
  }
  *tp_out = tp; *npred_out = npred; *ngold_out = ngold;
}

static inline double tk_span_f1_of (int64_t tp, int64_t npred, int64_t ngold) {
  double p = npred > 0 ? (double) tp / (double) npred : 0.0;
  double r = ngold > 0 ? (double) tp / (double) ngold : 0.0;
  return (p + r > 0.0) ? 2.0 * p * r / (p + r) : 0.0;
}

#endif
