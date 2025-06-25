#ifndef TK_CLUSTER_H
#define TK_CLUSTER_H

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/dsu.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/ivec.h>

static inline void tk_cluster_dsu (
  lua_State *L,
  tk_hbi_t *hbi,
  uint64_t margin,
  tk_ivec_t **idsp,
  tk_ivec_t **assignmentsp,
  uint64_t *n_clustersp
) {
  tk_pvec_t *tmp = tk_pvec_create(L, 0, 0, 0); // tmp
  tk_ivec_t *ids = (*idsp) = tk_iumap_keys(L, hbi->uid_sid); // tmp ids
  tk_iumap_t *ididx = tk_iumap_from_ivec(ids);

  tk_dsu_t dsu;
  tk_dsu_init(L, &dsu, ids);

  for (uint64_t i = 0; i < ids->n; i ++) {
    int64_t uid = ids->a[i];
    tk_pvec_clear(tmp);
    tk_hbi_neighbors_by_id(L, hbi, uid, 0, margin, tmp);
    for (uint64_t j = 0; j < tmp->n; j ++)
      tk_dsu_union(&dsu, uid, tmp->a[j].i);
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
