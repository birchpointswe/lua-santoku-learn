#ifndef TK_CLUSTER_H
#define TK_CLUSTER_H

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/dsu.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/ann.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/iumap.h>

typedef struct {
  tk_inv_t *inv;
  tk_hbi_t *hbi;
  tk_ann_t *ann;
  tk_ivec_t *ids;
  tk_iumap_t *ididx;
  uint64_t margin;
  double eps;
  uint64_t min_pts;
  bool assign_noise;
  uint64_t probe_radius;
  tk_ivec_sim_type_t cmp;
  double cmp_alpha;
  double cmp_beta;
  int64_t rank_filter;
  tk_inv_hoods_t *inv_hoods;
  tk_ann_hoods_t *ann_hoods;
  tk_hbi_hoods_t *hbi_hoods;
  tk_ivec_t *uids_hoods;
  tk_iumap_t *uids_idx_hoods;
  tk_ivec_t *assignments;
  uint64_t *n_clustersp;
  tk_dsu_t *dsu_reuse;
  tk_cvec_t *is_core_reuse;
  tk_rvec_t *rtmp;
  tk_pvec_t *ptmp;
} tk_cluster_opts_t;

static inline int tk_in_idset (tk_iumap_t *ididx, int64_t id) {
  return tk_iumap_get(ididx, id) != tk_iumap_end(ididx);
}

static inline int64_t tk_idx_of (tk_iumap_t *ididx, int64_t id) {
  khint_t khi = tk_iumap_get(ididx, id);
  return (khi == tk_iumap_end(ididx)) ? -1 : tk_iumap_val(ididx, khi);
}

#define TK_FOR_EPS_NEIGHBORS(opts, uid, DO) \
  do { \
    if (opts->inv != NULL) { \
      if (opts->inv_hoods != NULL && opts->uids_hoods != NULL && opts->uids_idx_hoods != NULL) { \
        uint32_t __khi = tk_iumap_get(opts->uids_idx_hoods, (uid)); \
        if (__khi != tk_iumap_end(opts->uids_idx_hoods)) { \
          int64_t __hood_idx = tk_iumap_val(opts->uids_idx_hoods, __khi); \
          if (__hood_idx >= 0 && __hood_idx < (int64_t)opts->inv_hoods->n) { \
            tk_rvec_t *__hood = opts->inv_hoods->a[__hood_idx]; \
            for (uint64_t __j = 0; __j < __hood->n; __j ++) { \
              int64_t __neighbor_hood_idx = __hood->a[__j].i; \
              if (__neighbor_hood_idx >= 0 && __neighbor_hood_idx < (int64_t)opts->uids_hoods->n) { \
                int64_t vid = opts->uids_hoods->a[__neighbor_hood_idx]; \
                DO; \
              } \
            } \
          } \
        } \
      } else { \
        tk_rvec_clear(opts->rtmp); \
        tk_inv_neighbors_by_id(opts->inv, (uid), 0, opts->eps, opts->rtmp, \
          opts->cmp, opts->cmp_alpha, opts->cmp_beta, opts->rank_filter); \
        for (uint64_t __j = 0; __j < opts->rtmp->n; __j ++) { \
          int64_t vid = opts->rtmp->a[__j].i; \
          DO; \
        } \
      } \
    } else if (opts->hbi != NULL) { \
      if (opts->hbi_hoods != NULL && opts->uids_hoods != NULL && opts->uids_idx_hoods != NULL) { \
        uint32_t __khi = tk_iumap_get(opts->uids_idx_hoods, (uid)); \
        if (__khi != tk_iumap_end(opts->uids_idx_hoods)) { \
          int64_t __hood_idx = tk_iumap_val(opts->uids_idx_hoods, __khi); \
          if (__hood_idx >= 0 && __hood_idx < (int64_t)opts->hbi_hoods->n) { \
            tk_pvec_t *__hood = opts->hbi_hoods->a[__hood_idx]; \
            for (uint64_t __j = 0; __j < __hood->n; __j ++) { \
              int64_t __neighbor_hood_idx = __hood->a[__j].i; \
              if (__neighbor_hood_idx >= 0 && __neighbor_hood_idx < (int64_t)opts->uids_hoods->n) { \
                int64_t vid = opts->uids_hoods->a[__neighbor_hood_idx]; \
                DO; \
              } \
            } \
          } \
        } \
      } else { \
        tk_pvec_clear(opts->ptmp); \
        tk_hbi_neighbors_by_id(opts->hbi, (uid), 0, (uint64_t) opts->margin, opts->ptmp); \
        for (uint64_t __j = 0; __j < opts->ptmp->n; __j ++) { \
          int64_t vid = opts->ptmp->a[__j].i; \
          DO; \
        } \
      } \
    } else if (opts->ann != NULL) { \
      if (opts->ann_hoods != NULL && opts->uids_hoods != NULL && opts->uids_idx_hoods != NULL) { \
        uint32_t __khi = tk_iumap_get(opts->uids_idx_hoods, (uid)); \
        if (__khi != tk_iumap_end(opts->uids_idx_hoods)) { \
          int64_t __hood_idx = tk_iumap_val(opts->uids_idx_hoods, __khi); \
          if (__hood_idx >= 0 && __hood_idx < (int64_t)opts->ann_hoods->n) { \
            tk_pvec_t *__hood = opts->ann_hoods->a[__hood_idx]; \
            for (uint64_t __j = 0; __j < __hood->n; __j ++) { \
              int64_t __neighbor_hood_idx = __hood->a[__j].i; \
              if (__neighbor_hood_idx >= 0 && __neighbor_hood_idx < (int64_t)opts->uids_hoods->n) { \
                int64_t vid = opts->uids_hoods->a[__neighbor_hood_idx]; \
                DO; \
              } \
            } \
          } \
        } \
      } else { \
        tk_pvec_clear(opts->ptmp); \
        tk_ann_neighbors_by_id(opts->ann, (uid), 0, (uint64_t) opts->probe_radius, (int64_t) opts->margin, opts->ptmp); \
        for (uint64_t __j = 0; __j < opts->ptmp->n; __j ++) { \
          int64_t vid = opts->ptmp->a[__j].i; \
          DO; \
        } \
      } \
    } \
  } while (0)

static inline void tk_cluster_dsu (tk_cluster_opts_t *opts) {
  if (opts->ptmp) tk_pvec_clear(opts->ptmp);
  if (opts->rtmp) tk_rvec_clear(opts->rtmp);

  bool created_dsu = false;
  tk_dsu_t *dsu;
  if (opts->dsu_reuse != NULL) {
    dsu = opts->dsu_reuse;
  } else {
    dsu = tk_dsu_create(NULL, opts->ids);
    created_dsu = true;
  }

  if (opts->min_pts <= 1) {
    for (uint64_t i = 0; i < opts->ids->n; i ++) {
      int64_t uid = opts->ids->a[i];
      TK_FOR_EPS_NEIGHBORS(opts, uid, {
        int64_t k = tk_idx_of(opts->ididx, vid);
        if (k >= 0)
          tk_dsu_unionx(dsu, (int64_t) i, k);
      });
    }
    tk_iumap_t *cmap = tk_iumap_create(0, 0);
    int kha;
    khint_t khi;
    int64_t idx;
    int64_t next_cluster = 0;
    for (uint64_t i = 0; i < opts->ids->n; i ++) {
      int64_t u = opts->ids->a[i];
      int64_t r_idx = tk_dsu_findx(dsu, (int64_t) i);
      int64_t root_uid = opts->ids->a[r_idx];
      khi = tk_iumap_get(opts->ididx, u);
      assert(khi != tk_iumap_end(opts->ididx));
      idx = tk_iumap_val(opts->ididx, khi);
      khi = tk_iumap_put(cmap, root_uid, &kha);
      if (kha) tk_iumap_setval(cmap, khi, next_cluster ++);
      opts->assignments->a[idx] = tk_iumap_val(cmap, khi);
    }
    tk_iumap_destroy(cmap);
    if (created_dsu) tk_dsu_destroy(dsu);
    *opts->n_clustersp = (uint64_t) next_cluster;
    return;
  }

  uint64_t n = opts->ids->n;

  bool created_is_core = false;
  tk_cvec_t *is_core;
  if (opts->is_core_reuse != NULL) {
    is_core = opts->is_core_reuse;
  } else {
    is_core = tk_cvec_create(0, TK_CVEC_BITS_BYTES(n), 0, 0);
    created_is_core = true;
  }

  for (uint64_t i = 0; i < n; i ++) {
    if (is_core->a[TK_CVEC_BITS_BYTE(i)] & (1u << TK_CVEC_BITS_BIT(i)))
      continue;
    int64_t uid = opts->ids->a[i];
    uint64_t count = 0;
    TK_FOR_EPS_NEIGHBORS(opts, uid, {
      if (!tk_in_idset(opts->ididx, vid))
        continue;
      count ++;
      if (count + 1 >= opts->min_pts)
        break;
    });
    if (count + 1 >= opts->min_pts)
      is_core->a[TK_CVEC_BITS_BYTE(i)] |= (1u << TK_CVEC_BITS_BIT(i));
  }

  for (uint64_t i = 0; i < n; i ++) {
    if (is_core->a[TK_CVEC_BITS_BYTE(i)] & (1u << TK_CVEC_BITS_BIT(i))) {
      int64_t uid = opts->ids->a[i];
      TK_FOR_EPS_NEIGHBORS(opts, uid, {
        if (!tk_in_idset(opts->ididx, vid))
          continue;
        int64_t k = tk_idx_of(opts->ididx, vid);
        if (k < 0)
          continue;
        if (is_core->a[TK_CVEC_BITS_BYTE(k)] & (1u << TK_CVEC_BITS_BIT(k)))
          tk_dsu_unionx(dsu, (int64_t) i, k);
      });
    }
  }

  for (uint64_t i = 0; i < n; i ++) {
    int64_t u = opts->ids->a[i];
    khint_t khi = tk_iumap_get(opts->ididx, u);
    int64_t idx = tk_iumap_val(opts->ididx, khi);
    opts->assignments->a[idx] = -1;
  }

  tk_iumap_t *cmap = tk_iumap_create(0, 0);
  int kha;
  khint_t khi;
  int64_t next_cluster = 0;

  for (uint64_t i = 0; i < n; i ++) {
    if (is_core->a[TK_CVEC_BITS_BYTE(i)] & (1u << TK_CVEC_BITS_BIT(i))) {
      int64_t r_idx = tk_dsu_findx(dsu, (int64_t) i);
      int64_t root_uid = opts->ids->a[r_idx];
      khi = tk_iumap_put(cmap, root_uid, &kha);
      if (kha)
        tk_iumap_setval(cmap, khi, next_cluster ++);
    }
  }

  for (uint64_t i = 0; i < n; i ++) {
    if (is_core->a[TK_CVEC_BITS_BYTE(i)] & (1u << TK_CVEC_BITS_BIT(i))) {
      int64_t u = opts->ids->a[i];
      int64_t r_idx = tk_dsu_findx(dsu, (int64_t) i);
      int64_t root_uid = opts->ids->a[r_idx];
      khint_t kc = tk_iumap_get(cmap, root_uid);
      int64_t cid = (kc == tk_iumap_end(cmap)) ? -1 : tk_iumap_val(cmap, kc);
      khint_t ki = tk_iumap_get(opts->ididx, u);
      int64_t idx = tk_iumap_val(opts->ididx, ki);
      opts->assignments->a[idx] = cid;
    }
  }

  if (opts->assign_noise) {
    for (uint64_t i = 0; i < n; i ++) {
      if (is_core->a[TK_CVEC_BITS_BYTE(i)] & (1u << TK_CVEC_BITS_BIT(i))) {
        int64_t uid = opts->ids->a[i];
        int64_t attach_root = -1;
        TK_FOR_EPS_NEIGHBORS(opts, uid, {
          if (!tk_in_idset(opts->ididx, vid))
            continue;
          int64_t k = tk_idx_of(opts->ididx, vid);
          if (k < 0)
            continue;
          if (is_core->a[TK_CVEC_BITS_BYTE(k)] & (1u << TK_CVEC_BITS_BIT(k))) {
            int64_t r_idx = tk_dsu_findx(dsu, k);
            attach_root = opts->ids->a[r_idx];
            break;
          }
        });
        khint_t ki = tk_iumap_get(opts->ididx, uid);
        int64_t idx = tk_iumap_val(opts->ididx, ki);
        if (attach_root >= 0) {
          khint_t kc = tk_iumap_get(cmap, attach_root);
          if (kc != tk_iumap_end(cmap))
            opts->assignments->a[idx] = tk_iumap_val(cmap, kc);
          else
            opts->assignments->a[idx] = -1;
        } else {
          opts->assignments->a[idx] = -1;
        }
      }
    }
  }

  tk_iumap_destroy(cmap);
  if (created_dsu)
    tk_dsu_destroy(dsu);
  if (created_is_core)
    tk_cvec_destroy(is_core);
  *opts->n_clustersp = (uint64_t) next_cluster;
}

#endif
