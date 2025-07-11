#ifndef TK_CLUSTER_H
#define TK_CLUSTER_H

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/dsu.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/ivec.h>

static inline void tk_cluster_dsu (
  lua_State *L,
  tk_hbi_t *hbi,
  tk_ann_t *ann,
  tk_inv_t *inv,
  uint64_t margin,
  tk_ivec_t **idsp,
  tk_ivec_t **assignmentsp,
  uint64_t *n_clustersp
) {
  tk_pvec_t *ptmp = (hbi || ann) ? tk_pvec_create(L, 0, 0, 0) : NULL; // tmp
  tk_rvec_t *rtmp = (!ptmp && inv) ? tk_rvec_create(L, 0, 0, 0) : NULL; // tmp

  tk_ivec_t *ids = (*idsp) =
    hbi ? tk_iumap_keys(L, hbi->uid_sid) :
    ann ? tk_iumap_keys(L, ann->uid_sid) :
    inv ? tk_iumap_keys(L, inv->uid_sid) : tk_ivec_create(L, 0, 0, 0); // tmp ids
  tk_iumap_t *ididx = tk_iumap_from_ivec(ids);

  tk_dsu_t dsu;
  tk_dsu_init(L, &dsu, ids);

  if (hbi != NULL) {
    for (uint64_t i = 0; i < ids->n; i ++) {
      int64_t uid = ids->a[i];
      tk_pvec_clear(ptmp);
      tk_hbi_neighbors_by_id(L, hbi, uid, 0, margin, ptmp);
      for (uint64_t j = 0; j < ptmp->n; j ++)
        tk_dsu_union(&dsu, uid, ptmp->a[j].i);
    }
  } else if (ann != NULL) {
    for (uint64_t i = 0; i < ids->n; i ++) {
      int64_t uid = ids->a[i];
      tk_pvec_clear(ptmp);
      tk_ann_neighbors_by_id(L, ann, uid, 0, margin, ptmp);
      for (uint64_t j = 0; j < ptmp->n; j ++)
        tk_dsu_union(&dsu, uid, ptmp->a[j].i);
    }
  } else if (inv != NULL) {
    for (uint64_t i = 0; i < ids->n; i ++) {
      int64_t uid = ids->a[i];
      tk_rvec_clear(rtmp);
      tk_inv_neighbors_by_id(L, inv, uid, 0, margin, rtmp, TK_INV_JACCARD);
      for (uint64_t j = 0; j < rtmp->n; j ++)
        tk_dsu_union(&dsu, uid, rtmp->a[j].i);
    }
  }

  tk_iumap_t *cmap = tk_iumap_create();
  tk_ivec_t *assignments = (*assignmentsp) = tk_ivec_create(L, ids->n, 0, 0); // tmp ids assignments
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
  tk_iumap_destroy(ididx);
  lua_remove(L, -3); // ids assignments
}

#endif
