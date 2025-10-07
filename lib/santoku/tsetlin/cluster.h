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
  uint64_t depth;  // For prefix-based clustering
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
  uint64_t *prefix_cache_reuse;
  tk_bits_t *prefix_bytes_reuse;
  uint64_t prefix_cache_depth;
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

static inline bool tk_prefix_equal_bytes(
  const tk_bits_t *a, const tk_bits_t *b, uint64_t depth
) {
  uint64_t full_bytes = depth / 8;
  uint64_t remaining_bits = depth % 8;
  if (full_bytes > 0 && memcmp(a, b, full_bytes) != 0)
    return false;
  if (remaining_bits > 0) {
    uint8_t mask = (uint8_t)((1u << remaining_bits) - 1);
    if ((a[full_bytes] & mask) != (b[full_bytes] & mask))
      return false;
  }
  return true;
}

static inline uint64_t tk_extract_prefix_u64(
  const tk_bits_t *code, uint64_t depth
) {
  uint64_t prefix = 0;
  for (uint64_t bit = 0; bit < depth && bit < 64; bit++) {
    uint64_t byte_idx = bit / 8;
    uint64_t bit_idx = bit % 8;
    if (code[byte_idx] & (1u << bit_idx))
      prefix |= (1ULL << bit);
  }
  return prefix;
}

static inline void tk_cluster_prefix (tk_cluster_opts_t *opts) {
  uint64_t n = opts->ids->n;
  uint64_t depth = opts->depth;

  uint64_t features = 0;
  if (opts->hbi != NULL) {
    features = opts->hbi->features;
  } else if (opts->ann != NULL) {
    features = opts->ann->features;
  } else {
    *opts->n_clustersp = 0;
    return;
  }

  if (depth > features) depth = features;

  if (depth == 0) {
    for (uint64_t i = 0; i < n; i++) {
      int64_t u = opts->ids->a[i];
      khint_t khi = tk_iumap_get(opts->ididx, u);
      int64_t idx = tk_iumap_val(opts->ididx, khi);
      opts->assignments->a[idx] = 0;
    }
    *opts->n_clustersp = 1;
    return;
  }

  bool use_incremental = (opts->prefix_cache_reuse != NULL &&
                          opts->prefix_cache_depth == depth + 1);
  bool use_u64 = (depth <= 64);

  if (use_incremental) {
    tk_iumap_t *prefix_to_cluster = tk_iumap_create(0, 0);
    int64_t next_cluster_id = 0;

    for (uint64_t i = 0; i < n; i++) {
      int64_t uid = opts->ids->a[i];
      uint64_t new_prefix = opts->prefix_cache_reuse[i] >> 1;
      khint_t khi = tk_iumap_get(prefix_to_cluster, (int64_t)new_prefix);
      int64_t cluster_id;
      if (khi == tk_iumap_end(prefix_to_cluster)) {
        int kha;
        khi = tk_iumap_put(prefix_to_cluster, (int64_t)new_prefix, &kha);
        tk_iumap_setval(prefix_to_cluster, khi, next_cluster_id);
        cluster_id = next_cluster_id++;
      } else {
        cluster_id = tk_iumap_val(prefix_to_cluster, khi);
      }
      opts->prefix_cache_reuse[i] = new_prefix;
      khint_t khi_idx = tk_iumap_get(opts->ididx, uid);
      int64_t idx = tk_iumap_val(opts->ididx, khi_idx);
      opts->assignments->a[idx] = cluster_id;
    }

    tk_iumap_destroy(prefix_to_cluster);
    opts->prefix_cache_depth = depth;
    *opts->n_clustersp = (uint64_t)next_cluster_id;

  } else if (use_u64) {
    tk_iumap_t *prefix_map = tk_iumap_create(0, 0);
    int64_t next_cluster_id = 0;

    for (uint64_t i = 0; i < n; i++) {
      int64_t uid = opts->ids->a[i];
      uint64_t prefix = 0;

      if (opts->hbi != NULL) {
        khint_t khi_uid = tk_iumap_get(opts->hbi->uid_sid, uid);
        if (khi_uid == tk_iumap_end(opts->hbi->uid_sid))
          continue;
        int64_t sid = tk_iumap_val(opts->hbi->uid_sid, khi_uid);
        uint32_t code = opts->hbi->codes->a[sid];
        prefix = (depth < features) ? (code >> (features - depth)) : code;
      } else if (opts->ann != NULL) {
        khint_t khi_uid = tk_iumap_get(opts->ann->uid_sid, uid);
        if (khi_uid == tk_iumap_end(opts->ann->uid_sid))
          continue;
        int64_t sid = tk_iumap_val(opts->ann->uid_sid, khi_uid);
        uint64_t chunks = TK_CVEC_BITS_BYTES(features);
        tk_bits_t *vec = (tk_bits_t *)opts->ann->vectors->a + ((uint64_t)sid * chunks);
        prefix = tk_extract_prefix_u64(vec, depth);
      }

      if (opts->prefix_cache_reuse != NULL)
        opts->prefix_cache_reuse[i] = prefix;

      khint_t khi = tk_iumap_get(prefix_map, (int64_t)prefix);
      int64_t cluster_id;
      if (khi == tk_iumap_end(prefix_map)) {
        int kha;
        khi = tk_iumap_put(prefix_map, (int64_t)prefix, &kha);
        tk_iumap_setval(prefix_map, khi, next_cluster_id);
        cluster_id = next_cluster_id++;
      } else {
        cluster_id = tk_iumap_val(prefix_map, khi);
      }

      khint_t khi_idx = tk_iumap_get(opts->ididx, uid);
      int64_t idx = tk_iumap_val(opts->ididx, khi_idx);
      opts->assignments->a[idx] = cluster_id;
    }

    if (opts->prefix_cache_reuse != NULL)
      opts->prefix_cache_depth = depth;

    tk_iumap_destroy(prefix_map);
    *opts->n_clustersp = (uint64_t)next_cluster_id;

  } else {
    uint64_t chunks = TK_CVEC_BITS_BYTES(features);
    tk_bits_t *codes_buffer = NULL;
    bool codes_cached = (opts->prefix_bytes_reuse != NULL &&
                        opts->prefix_cache_depth == depth + 1);

    if (!codes_cached) {
      codes_buffer = (tk_bits_t *)malloc(n * chunks);
      if (!codes_buffer) {
        *opts->n_clustersp = 0;
        return;
      }
      for (uint64_t i = 0; i < n; i++) {
        int64_t uid = opts->ids->a[i];
        tk_bits_t *dest = codes_buffer + i * chunks;
        if (opts->ann != NULL) {
          khint_t khi_uid = tk_iumap_get(opts->ann->uid_sid, uid);
          if (khi_uid == tk_iumap_end(opts->ann->uid_sid)) {
            memset(dest, 0, chunks);
            continue;
          }
          int64_t sid = tk_iumap_val(opts->ann->uid_sid, khi_uid);
          tk_bits_t *src = (tk_bits_t *) opts->ann->vectors->a + ((uint64_t)sid * chunks);
          memcpy(dest, src, chunks);
        }
      }
      if (opts->prefix_bytes_reuse != NULL)
        memcpy(opts->prefix_bytes_reuse, codes_buffer, n * chunks);
    } else {
      codes_buffer = opts->prefix_bytes_reuse;
    }

    uint64_t *indices = (uint64_t *)malloc(n * sizeof(uint64_t));
    if (!indices) {
      if (!codes_cached) free(codes_buffer);
      *opts->n_clustersp = 0;
      return;
    }
    for (uint64_t i = 0; i < n; i++)
      indices[i] = i;

    for (uint64_t i = 1; i < n; i++) {
      uint64_t key_idx = indices[i];
      tk_bits_t *key_code = codes_buffer + key_idx * chunks;
      int64_t j = (int64_t)i - 1;
      while (j >= 0) {
        uint64_t cmp_idx = indices[j];
        tk_bits_t *cmp_code = codes_buffer + cmp_idx * chunks;
        bool less = false;
        uint64_t full_bytes = depth / 8;
        uint64_t remaining_bits = depth % 8;
        int cmp_result = memcmp(key_code, cmp_code, full_bytes);
        if (cmp_result < 0) {
          less = true;
        } else if (cmp_result == 0 && remaining_bits > 0) {
          uint8_t mask = (uint8_t)((1u << remaining_bits) - 1);
          if ((key_code[full_bytes] & mask) < (cmp_code[full_bytes] & mask))
            less = true;
        }
        if (!less) break;
        indices[j + 1] = indices[j];
        j--;
      }
      indices[j + 1] = key_idx;
    }

    int64_t next_cluster_id = 0;
    tk_bits_t *prev_code = NULL;
    for (uint64_t i = 0; i < n; i++) {
      uint64_t idx_in_ids = indices[i];
      int64_t uid = opts->ids->a[idx_in_ids];
      tk_bits_t *curr_code = codes_buffer + idx_in_ids * chunks;
      bool new_cluster = (prev_code == NULL ||
                         !tk_prefix_equal_bytes(curr_code, prev_code, depth));
      if (new_cluster)
        next_cluster_id++;
      khint_t khi_idx = tk_iumap_get(opts->ididx, uid);
      int64_t assign_idx = tk_iumap_val(opts->ididx, khi_idx);
      opts->assignments->a[assign_idx] = next_cluster_id - 1;
      prev_code = curr_code;
    }

    free(indices);
    if (!codes_cached) free(codes_buffer);
    if (opts->prefix_bytes_reuse != NULL)
      opts->prefix_cache_depth = depth;
    *opts->n_clustersp = (uint64_t)next_cluster_id;
  }
}

#endif
