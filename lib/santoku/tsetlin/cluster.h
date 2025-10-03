#ifndef TK_CLUSTER_H
#define TK_CLUSTER_H

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/dsu.h>
#include <santoku/tsetlin/ann.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>

static inline int tk_in_idset (tk_iumap_t *ididx, int64_t id) {
  return tk_iumap_get(ididx, id) != tk_iumap_end(ididx);
}

static inline int64_t tk_idx_of (tk_iumap_t *ididx, int64_t id) {
  khint_t khi = tk_iumap_get(ididx, id);
  return (khi == tk_iumap_end(ididx)) ? -1 : tk_iumap_val(ididx, khi);
}

#define TK_FOR_EPS_NEIGHBORS(uid, DO) \
  do { \
    if (hbi != NULL) { \
      tk_pvec_clear(ptmp); \
      tk_hbi_neighbors_by_id(hbi, (uid), 0, margin, ptmp); \
      for (uint64_t __j = 0; __j < ptmp->n; __j ++) { \
        int64_t vid = ptmp->a[__j].i; \
        DO; \
      } \
    } else if (ann != NULL) { \
      tk_pvec_clear(ptmp); \
      tk_ann_neighbors_by_id(ann, (uid), 0, margin, ptmp); \
      for (uint64_t __j = 0; __j < ptmp->n; __j ++) { \
        int64_t vid = ptmp->a[__j].i; \
        DO; \
      } \
    } \
  } while (0)

static inline void tk_cluster_dsu (
  tk_hbi_t *hbi,
  tk_ann_t *ann,
  tk_rvec_t *rtmp,
  tk_pvec_t *ptmp,
  uint64_t margin,
  uint64_t min_pts,
  bool assign_noise,
  tk_ivec_t *ids,
  tk_ivec_t *assignments,
  tk_iumap_t *ididx,
  uint64_t *n_clustersp
) {
  if (ptmp) tk_pvec_clear(ptmp);
  if (rtmp) tk_rvec_clear(rtmp);

  tk_dsu_t *dsu = tk_dsu_create(NULL, ids);

  if (min_pts <= 1) {

    for (uint64_t i = 0; i < ids->n; i ++) {
      int64_t uid = ids->a[i];
      TK_FOR_EPS_NEIGHBORS(uid, {
        if (tk_in_idset(ididx, vid))
          tk_dsu_union(dsu, uid, vid);
      });
    }

    tk_iumap_t *cmap = tk_iumap_create(0, 0);
    int kha;
    khint_t khi;
    int64_t idx;
    int64_t next_cluster = 0;
    for (uint64_t i = 0; i < ids->n; i ++) {
      int64_t u = ids->a[i];
      int64_t r = tk_dsu_find(dsu, u);
      khi = tk_iumap_get(ididx, u);
      assert(khi != tk_iumap_end(ididx));
      idx = tk_iumap_val(ididx, khi);
      khi = tk_iumap_put(cmap, r, &kha);
      if (kha) tk_iumap_setval(cmap, khi, next_cluster ++);
      assignments->a[idx] = tk_iumap_val(cmap, khi);
    }

    tk_iumap_destroy(cmap);
    tk_dsu_destroy(dsu);
    *n_clustersp = (uint64_t) next_cluster;
    return;
  }

  uint64_t n = ids->n;
  tk_ivec_t *deg = tk_ivec_create(0, n, 0, 0);
  tk_ivec_zero(deg);

  tk_ivec_t *is_core = tk_ivec_create(0, n, 0, 0);
  tk_ivec_zero(is_core);

  for (uint64_t i = 0; i < n; i ++) {
    int64_t uid = ids->a[i];
    uint64_t count = 0;
    TK_FOR_EPS_NEIGHBORS(uid, {
      if (!tk_in_idset(ididx, vid))
        continue;
      count ++;
      if (count + 1 >= min_pts)
        break;
    });
    deg->a[i] = (int64_t) count;
    is_core->a[i] = (count + 1 >= min_pts) ? 1 : 0;
  }

  for (uint64_t i = 0; i < n; i ++) {
    if (!is_core->a[i])
      continue;
    int64_t uid = ids->a[i];
    TK_FOR_EPS_NEIGHBORS(uid, {
      if (!tk_in_idset(ididx, vid))
        continue;
      int64_t k = tk_idx_of(ididx, vid);
      if (k < 0)
        continue;
      if (!is_core->a[(uint64_t) k])
        continue;
      tk_dsu_union(dsu, uid, vid);
    });
  }

  for (uint64_t i = 0; i < n; i ++) {
    int64_t u = ids->a[i];
    khint_t khi = tk_iumap_get(ididx, u);
    int64_t idx = tk_iumap_val(ididx, khi);
    assignments->a[idx] = -1;
  }

  tk_iumap_t *cmap = tk_iumap_create(0, 0);
  int kha;
  khint_t khi;
  int64_t next_cluster = 0;

  for (uint64_t i = 0; i < n; i ++) {
    if (!is_core->a[i]) continue;
    int64_t u = ids->a[i];
    int64_t r = tk_dsu_find(dsu, u);
    khi = tk_iumap_put(cmap, r, &kha);
    if (kha) tk_iumap_setval(cmap, khi, next_cluster ++);
  }

  for (uint64_t i = 0; i < n; i ++) {
    if (!is_core->a[i]) continue;
    int64_t u = ids->a[i];
    int64_t r = tk_dsu_find(dsu, u);
    khint_t kc = tk_iumap_get(cmap, r);
    int64_t cid = (kc == tk_iumap_end(cmap)) ? -1 : tk_iumap_val(cmap, kc);
    khint_t ki = tk_iumap_get(ididx, u);
    int64_t idx = tk_iumap_val(ididx, ki);
    assignments->a[idx] = cid;
  }

  if (assign_noise) {
    for (uint64_t i = 0; i < n; i ++) {
      if (is_core->a[i])
        continue;
      int64_t uid = ids->a[i];
      int64_t attach_root = -1;

      TK_FOR_EPS_NEIGHBORS(uid, {
        if (!tk_in_idset(ididx, vid))
          continue;
        int64_t k = tk_idx_of(ididx, vid);
        if (k < 0)
          continue;
        if (!is_core->a[(uint64_t) k])
          continue;
        attach_root = tk_dsu_find(dsu, vid);
        break;
      });

      khint_t ki = tk_iumap_get(ididx, uid);
      int64_t idx = tk_iumap_val(ididx, ki);

      if (attach_root >= 0) {
        khint_t kc = tk_iumap_get(cmap, attach_root);
        if (kc != tk_iumap_end(cmap))
          assignments->a[idx] = tk_iumap_val(cmap, kc);
        else
          assignments->a[idx] = -1;
      } else {
        assignments->a[idx] = -1;
      }
    }
  }

  tk_iumap_destroy(cmap);
  tk_dsu_destroy(dsu);
  tk_ivec_destroy(is_core);
  tk_ivec_destroy(deg);

  *n_clustersp = (uint64_t) next_cluster;
}

#endif
