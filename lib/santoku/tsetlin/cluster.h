#ifndef TK_CLUSTER_H
#define TK_CLUSTER_H

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/dsu.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>

static inline void tk_cluster_dsu (
  tk_hbi_t *hbi,
  tk_ann_t *ann,
  tk_inv_t *inv,
  tk_rvec_t *rtmp,
  tk_pvec_t *ptmp,
  uint64_t margin,
  tk_ivec_t *ids,
  tk_ivec_t *assignments,
  tk_iumap_t *ididx,
  uint64_t *n_clustersp
) {
  if (ptmp) tk_pvec_clear(ptmp);
  if (rtmp) tk_rvec_clear(rtmp);

  tk_dsu_t dsu;
  tk_dsu_init(&dsu, ids);

  if (hbi != NULL) {
    for (uint64_t i = 0; i < ids->n; i ++) {
      int64_t uid = ids->a[i];
      tk_pvec_clear(ptmp);
      tk_hbi_neighbors_by_id(hbi, uid, 0, margin, ptmp);
      for (uint64_t j = 0; j < ptmp->n; j ++)
        if (tk_iumap_get(ididx, ptmp->a[j].i) != tk_iumap_end(ididx))
          tk_dsu_union(&dsu, uid, ptmp->a[j].i);
    }
  } else if (ann != NULL) {
    for (uint64_t i = 0; i < ids->n; i ++) {
      int64_t uid = ids->a[i];
      tk_pvec_clear(ptmp);
      tk_ann_neighbors_by_id(ann, uid, 0, margin, ptmp);
      for (uint64_t j = 0; j < ptmp->n; j ++)
        if (tk_iumap_get(ididx, ptmp->a[j].i) != tk_iumap_end(ididx))
          tk_dsu_union(&dsu, uid, ptmp->a[j].i);
    }
  } else if (inv != NULL) {
    for (uint64_t i = 0; i < ids->n; i ++) {
      int64_t uid = ids->a[i];
      tk_rvec_clear(rtmp);
      tk_inv_neighbors_by_id(inv, uid, 0, margin, rtmp, TK_INV_JACCARD);
      for (uint64_t j = 0; j < rtmp->n; j ++)
        if (tk_iumap_get(ididx, rtmp->a[j].i) != tk_iumap_end(ididx))
          tk_dsu_union(&dsu, uid, rtmp->a[j].i);
    }
  }

  tk_iumap_t *cmap = tk_iumap_create();
  int kha;
  khint_t khi;
  int64_t idx;
  int64_t next_cluster = 0;
  for (uint64_t i = 0; i < ids->n; i ++) {
    int64_t u = ids->a[i];
    int64_t c = tk_dsu_find(&dsu, u);
    khi = tk_iumap_get(ididx, u);
    assert(khi != tk_iumap_end(ididx));
    idx = tk_iumap_value(ididx, khi);
    khi = tk_iumap_put(cmap, c, &kha);
    if (kha)
      tk_iumap_value(cmap, khi) = next_cluster ++;
    c = tk_iumap_value(cmap, khi);
    assignments->a[idx] = c;
  }

  *n_clustersp = (uint64_t) next_cluster;
}

#endif
