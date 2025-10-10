#ifndef TK_CLUSTER_H
#define TK_CLUSTER_H

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/dsu.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/ann.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/tsetlin/simhash.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/iumap.h>
#include <santoku/iuset.h>
#include <santoku/pvec.h>
#include <santoku/evec.h>
#include <math.h>
#include <string.h>

#define TK_AGGLO_EPH "tk_agglo_eph"

typedef struct {
  tk_inv_t *inv;
  tk_hbi_t *hbi;
  tk_ann_t *ann;
  tk_ivec_t *ids;
  tk_iumap_t *ididx;
  uint64_t margin;
  double eps;
  uint64_t depth;
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
  unsigned int n_threads;
  bool use_agglo;
  lua_State *L;
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
    is_core = tk_cvec_create(NULL, TK_CVEC_BITS_BYTES(n), 0, 0);
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

typedef enum {
  TK_AGGLO_USE_ANN,
  TK_AGGLO_USE_HBI
} tk_agglo_index_type_t;

typedef enum {
  TK_AGGLO_STAGE_FIND_MIN_EDGES,
  TK_AGGLO_STAGE_MERGE_CLUSTERS
} tk_agglo_stage_t;

typedef struct {
  int64_t cluster_id;
  tk_simhash_t *simhash;
  tk_ivec_t *members;
  uint64_t code_chunks;
  bool active;
  uint64_t size;
  bool is_userdata;
} tk_agglo_cluster_t;

typedef struct {
  tk_agglo_index_type_t index_type;
  uint64_t features;
  uint64_t state_bits;
  uint64_t probe_radius;
  tk_ann_t *ann;
  tk_hbi_t *hbi;
  tk_ivec_t *uids;
  uint64_t n_samples;
  tk_iumap_t *uid_to_vec_idx;
  tk_agglo_cluster_t **clusters;
  uint64_t n_clusters;
  uint64_t n_active_clusters;
  tk_iumap_t *uid_to_cluster;
  tk_iumap_t *cluster_id_to_idx;
  union {
    tk_ann_t *ann;
    tk_hbi_t *hbi;
  } index;
  tk_threadpool_t *pool;
  unsigned int n_threads;
  tk_pvec_t **thread_neighbors;
  double *thread_min_dist;
  tk_pvec_t **thread_min_edges;
  tk_pvec_t *candidate_edges;
  tk_pvec_t *selected_merges;
  tk_iuset_t *merged_clusters;
  tk_ivec_t *assignments;
  lua_State *L;
  bool is_userdata;
  uint64_t n_clusters_created;
  uint64_t n_threads_arrays_allocated;
} tk_agglo_state_t;

typedef struct {
  tk_agglo_state_t *state;
  uint64_t first, last;
  unsigned int index;
  union {
    struct {
      tk_pvec_t *neighbors;
      double min_dist;
      tk_pvec_t *min_edges;
    } find_edges;
    struct {
      uint64_t merge_first, merge_last;
    } merge;
  } stage_data;
} tk_agglo_thread_t;

static inline uint64_t tk_agglo_hash_code (tk_bits_t *code, uint64_t code_chunks) {
  uint64_t hash = 0;
  for (uint64_t i = 0; i < code_chunks && i < 8; i++) {
    hash ^= ((uint64_t)code[i]) << (i * 8);
  }
  return hash;
}

typedef void (*tk_agglo_callback_t)(
  void *user_data,
  uint64_t iteration,
  uint64_t n_active_clusters,
  tk_ivec_t *ids,
  tk_ivec_t *assignments
);

static inline tk_agglo_cluster_t *tk_agglo_cluster_create (
  lua_State *L,
  int state_idx,
  int64_t cluster_id,
  uint64_t features,
  uint64_t state_bits
) {
  tk_agglo_cluster_t *cluster;
  int cluster_idx = -1;
  if (L) {
    cluster = lua_newuserdata(L, sizeof(tk_agglo_cluster_t));
    cluster_idx = lua_gettop(L);
    memset(cluster, 0, sizeof(tk_agglo_cluster_t));
    cluster->is_userdata = true;
  } else {
    cluster = malloc(sizeof(tk_agglo_cluster_t));
    if (!cluster) return NULL;
    memset(cluster, 0, sizeof(tk_agglo_cluster_t));
    cluster->is_userdata = false;
  }
  cluster->cluster_id = cluster_id;
  cluster->active = true;
  cluster->size = 0;
  uint64_t code_chunks = TK_CVEC_BITS_BYTES(features);
  cluster->code_chunks = code_chunks;
  if (L) {
    cluster->members = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_AGGLO_EPH, cluster_idx, -1);
    lua_pop(L, 1);
  } else {
    cluster->members = tk_ivec_create(NULL, 0, 0, 0);
    if (!cluster->members) {
      free(cluster);
      return NULL;
    }
  }
  cluster->simhash = tk_malloc(L, sizeof(tk_simhash_t));
  tk_bits_t tail_mask = (features % TK_CVEC_BITS == 0) ?
    (tk_bits_t)~0 : (tk_bits_t)((1 << (features % TK_CVEC_BITS)) - 1);
  int init_result = tk_simhash_init(L, cluster->simhash, 1, code_chunks, state_bits, tail_mask, TK_SIMHASH_INTEGER);
  if (!L && init_result != 0) {
    free(cluster->simhash);
    tk_ivec_destroy(cluster->members);
    free(cluster);
    return NULL;
  }
  tk_simhash_setup_initial_state(cluster->simhash, 0, 0);
  if (L && state_idx > 0)
    tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, cluster_idx);
  return cluster;
}

static inline void tk_agglo_cluster_destroy (
  tk_agglo_cluster_t *cluster
) {
  if (!cluster)
    return;
  if (cluster->simhash) {
    tk_simhash_destroy(cluster->simhash);
    free(cluster->simhash);
    cluster->simhash = NULL;
  }
  if (cluster->members) {
    tk_ivec_destroy(cluster->members);
    cluster->members = NULL;
  }
  if (!cluster->is_userdata)
    free(cluster);
}

static inline void tk_agglo_cluster_add_member (
  tk_agglo_cluster_t *cluster,
  int64_t uid,
  tk_bits_t *code,
  uint64_t code_chunks
) {
  tk_ivec_push(cluster->members, uid);
  cluster->size++;
  if (cluster->size == 1) {
    // For first member: just increment set bits
    // This pushes them from initial_state to threshold, setting actions = code
    for (uint64_t chunk = 0; chunk < code_chunks; chunk++) {
      tk_simhash_inc(cluster->simhash, 0, chunk, code[chunk]);
    }
  } else {
    // For subsequent members: increment set bits, decrement unset bits
    for (uint64_t chunk = 0; chunk < code_chunks; chunk++) {
      tk_simhash_inc(cluster->simhash, 0, chunk, code[chunk]);
      tk_simhash_dec(cluster->simhash, 0, chunk, ~code[chunk]);
    }
  }
}

static inline void tk_agglo_cluster_merge (
  tk_agglo_cluster_t *to,
  tk_agglo_cluster_t *from
) {
  for (uint64_t i = 0; i < from->members->n; i++)
    tk_ivec_push(to->members, from->members->a[i]);
  to->size += from->size;
  tk_simhash_merge(to->simhash, from->simhash);
  // Actions now directly represent the merged cluster code
  from->active = false;
  from->size = 0;
  tk_ivec_clear(from->members);
}

static inline void tk_agglo_worker(void *dp, int sig) {
  tk_agglo_stage_t stage = (tk_agglo_stage_t)sig;
  tk_agglo_thread_t *data = (tk_agglo_thread_t *)dp;
  tk_agglo_state_t *state = data->state;
  switch (stage) {
    case TK_AGGLO_STAGE_FIND_MIN_EDGES: {
      tk_pvec_t *neighbors = data->stage_data.find_edges.neighbors;
      tk_pvec_t *min_edges = data->stage_data.find_edges.min_edges;
      tk_pvec_clear(neighbors);
      tk_pvec_clear(min_edges);
      double min_dist = INFINITY;
      for (uint64_t i = data->first; i <= data->last && i < state->n_clusters; i++) {
        tk_agglo_cluster_t *cluster = state->clusters[i];
        if (!cluster || !cluster->active) continue;
        tk_pvec_clear(neighbors);
        tk_bits_t *code = tk_simhash_actions(cluster->simhash, 0);
        if (state->index_type == TK_AGGLO_USE_ANN) {
          tk_ann_neighbors_by_vec(state->index.ann, (char *)code,
                                   cluster->cluster_id, 1, state->probe_radius, -1, neighbors);
        } else {
          tk_hbi_neighbors_by_vec(state->index.hbi, (char *)code,
                                   cluster->cluster_id, 1, state->probe_radius, neighbors);
        }
        if (neighbors->n > 0) {
          tk_pvec_asc(neighbors, 0, neighbors->n);
          int64_t local_min = neighbors->a[0].p;
          for (uint64_t j = 0; j < neighbors->n; j++) {
            if (neighbors->a[j].p > local_min) break;
            int64_t other_cluster_id = neighbors->a[j].i;
            khint_t khi = tk_iumap_get(state->cluster_id_to_idx, other_cluster_id);
            if (khi == tk_iumap_end(state->cluster_id_to_idx)) continue;
            uint64_t other_idx = (uint64_t)tk_iumap_val(state->cluster_id_to_idx, khi);
            if (!state->clusters[other_idx] || !state->clusters[other_idx]->active) continue;
            if (i == other_idx) continue;
            uint64_t c1 = i, c2 = other_idx;
            if (state->clusters[c1]->size > state->clusters[c2]->size ||
                (state->clusters[c1]->size == state->clusters[c2]->size && c1 > c2)) {
              uint64_t tmp = c1; c1 = c2; c2 = tmp;
            }
            double dist = (double)local_min;
            if (dist < min_dist) {
              min_dist = dist;
              tk_pvec_clear(min_edges);
            }
            if (dist == min_dist) {
              tk_pvec_push(min_edges, tk_pair((int64_t)c1, (int64_t)c2));
            }
          }
        }
      }
      data->stage_data.find_edges.min_dist = min_dist;
      break;
    }
    case TK_AGGLO_STAGE_MERGE_CLUSTERS: {
      for (uint64_t i = data->stage_data.merge.merge_first;
           i <= data->stage_data.merge.merge_last && i < state->selected_merges->n; i++) {
        tk_pair_t merge = state->selected_merges->a[i];
        uint64_t from_idx = (uint64_t)merge.i;
        uint64_t to_idx = (uint64_t)merge.p;
        tk_agglo_cluster_t *from = state->clusters[from_idx];
        tk_agglo_cluster_t *to = state->clusters[to_idx];
        if (!from || !to || !from->active || !to->active) continue;
        for (uint64_t j = 0; j < from->members->n; j++) {
          int64_t uid = from->members->a[j];
          khint_t khi = tk_iumap_get(state->uid_to_cluster, uid);
          if (khi != tk_iumap_end(state->uid_to_cluster)) {
            tk_iumap_setval(state->uid_to_cluster, khi, (int64_t)to_idx);
          }
        }
        tk_agglo_cluster_merge(to, from);
        khint_t khi = tk_iumap_get(state->cluster_id_to_idx, from->cluster_id);
        if (khi != tk_iumap_end(state->cluster_id_to_idx)) {
          tk_iumap_del(state->cluster_id_to_idx, khi);
        }
      }
      break;
    }
    default:
      break;
  }
}

static inline int tk_agglo_state_gc(lua_State *L) {
  tk_agglo_state_t *state = luaL_checkudata(L, 1, "tk_agglo_state_t");
  if (state->clusters) {
    for (uint64_t i = 0; i < state->n_clusters_created; i++) {
      if (state->clusters[i]) {
        tk_agglo_cluster_destroy(state->clusters[i]);
        state->clusters[i] = NULL;
      }
    }
    free(state->clusters);
    state->clusters = NULL;
  }
  if (state->thread_neighbors) {
    free(state->thread_neighbors);
    free(state->thread_min_dist);
    free(state->thread_min_edges);
    state->thread_neighbors = NULL;
    state->thread_min_dist = NULL;
    state->thread_min_edges = NULL;
  }
  return 0;
}

static inline int tk_agglo (
  lua_State *L,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  tk_ivec_t *uids,
  uint64_t features,
  tk_agglo_index_type_t index_type,
  uint64_t probe_radius,
  unsigned int n_threads,
  tk_ivec_t *assignments,
  tk_agglo_callback_t callback,
  void *callback_data
) {
  if (!L || !uids || !assignments)
    return -1;
  if (uids->n == 0)
    return -1;
  if (!ann && !hbi)
    return -1;
  if (features == 0)
    return -1;

  uint64_t n_samples = uids->n;
  uint64_t code_chunks = TK_CVEC_BITS_BYTES(features);
  if (code_chunks == 0)
    return -1;

  uint64_t state_bits = log2(n_samples);
  tk_agglo_state_t *state = tk_lua_newuserdata(L, tk_agglo_state_t, TK_AGGLO_EPH, NULL, tk_agglo_state_gc);
  int state_idx = lua_gettop(L);
  state->is_userdata = true;

  state->L = L;
  state->index_type = index_type;
  state->features = features;
  state->state_bits = state_bits;
  state->probe_radius = probe_radius;
  state->ann = ann;
  state->hbi = hbi;
  state->uids = uids;
  state->n_samples = n_samples;
  state->n_threads = n_threads;
  state->assignments = assignments;
  state->n_clusters_created = 0;
  state->n_threads_arrays_allocated = 0;

  state->uid_to_vec_idx = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);

  for (uint64_t i = 0; i < n_samples; i++) {
    int kha;
    khint_t khi = tk_iumap_put(state->uid_to_vec_idx, uids->a[i], &kha);
    tk_iumap_setval(state->uid_to_vec_idx, khi, (int64_t)i);
  }

  state->pool = tk_threads_create(L, n_threads, tk_agglo_worker);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);

  state->thread_neighbors = tk_malloc(L, n_threads * sizeof(tk_pvec_t *));
  state->thread_min_dist = tk_malloc(L, n_threads * sizeof(double));
  state->thread_min_edges = tk_malloc(L, n_threads * sizeof(tk_pvec_t *));

  for (unsigned int i = 0; i < n_threads; i++) {
    state->thread_neighbors[i] = tk_pvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
    lua_pop(L, 1);
    state->thread_min_edges[i] = tk_pvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
    lua_pop(L, 1);
    state->thread_min_dist[i] = INFINITY;
    state->n_threads_arrays_allocated++;
  }

  state->candidate_edges = tk_pvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);

  state->selected_merges = tk_pvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);

  state->merged_clusters = tk_iuset_create(L, 0);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);

  state->uid_to_cluster = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);

  state->cluster_id_to_idx = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);

  uint64_t max_unique_codes = (features <= 20) ? (1ULL << features) : n_samples;
  if (max_unique_codes > n_samples)
    max_unique_codes = n_samples;
  if (max_unique_codes > 1000000)
    max_unique_codes = 1000000;

  state->clusters = tk_malloc(L, max_unique_codes * sizeof(tk_agglo_cluster_t *));
  for (uint64_t i = 0; i < max_unique_codes; i ++)
    state->clusters[i] = NULL;

  tk_iumap_t *code_hash_to_cluster = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);

  state->n_clusters = 0;
  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t uid = uids->a[i];
    char *vec_ptr = ann ? tk_ann_get(ann, uid) : tk_hbi_get(hbi, uid);
    if (!vec_ptr)
      continue;
    tk_bits_t *code = (tk_bits_t *)vec_ptr;
    uint64_t code_hash = tk_agglo_hash_code(code, code_chunks);
    khint_t khi = tk_iumap_get(code_hash_to_cluster, (int64_t)code_hash);
    uint64_t cluster_idx;
    if (khi == tk_iumap_end(code_hash_to_cluster)) {
      if (state->n_clusters >= max_unique_codes)
        tk_error(L, "tk_agglo: too many unique clusters", ENOMEM);
      cluster_idx = state->n_clusters;
      tk_agglo_cluster_t *cluster = tk_agglo_cluster_create(
        L, state_idx, (int64_t)cluster_idx, features, state_bits);
      lua_pop(L, 1);
      state->clusters[cluster_idx] = cluster;
      state->n_clusters_created++;  // Track for cleanup
      int kha;
      khi = tk_iumap_put(code_hash_to_cluster, (int64_t)code_hash, &kha);
      tk_iumap_setval(code_hash_to_cluster, khi, (int64_t)cluster_idx);
      khint_t khi2 = tk_iumap_put(state->cluster_id_to_idx, cluster->cluster_id, &kha);
      tk_iumap_setval(state->cluster_id_to_idx, khi2, (int64_t)cluster_idx);
      state->n_clusters++;
    } else {
      cluster_idx = (uint64_t)tk_iumap_val(code_hash_to_cluster, khi);
    }
    tk_agglo_cluster_add_member(state->clusters[cluster_idx], uid, code, code_chunks);
    int kha;
    khint_t khi2 = tk_iumap_put(state->uid_to_cluster, uids->a[i], &kha);
    tk_iumap_setval(state->uid_to_cluster, khi2, (int64_t)cluster_idx);
  }
  state->n_active_clusters = state->n_clusters;

  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t uid = uids->a[i];
    khint_t khi = tk_iumap_get(state->uid_to_cluster, uid);
    if (khi != tk_iumap_end(state->uid_to_cluster)) {
      uint64_t cluster_idx = (uint64_t)tk_iumap_val(state->uid_to_cluster, khi);
      assignments->a[i] = state->clusters[cluster_idx]->cluster_id;
    } else {
      assignments->a[i] = -1;
    }
  }
  assignments->n = n_samples;

  tk_agglo_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_agglo_thread_t));
  for (unsigned int i = 0; i < n_threads; i++) {
    threads[i].state = state;
    threads[i].index = i;
    state->pool->threads[i].data = &threads[i];
  }

  tk_ivec_t *cluster_ids = tk_ivec_create(L, 0, 0, 0);
  tk_cvec_t *cluster_codes = tk_cvec_create(L, state->n_clusters * code_chunks, 0, 0);
  for (uint64_t i = 0; i < state->n_clusters; i++) {
    tk_ivec_push(cluster_ids, state->clusters[i]->cluster_id);
    memcpy(cluster_codes->a + i * code_chunks, tk_simhash_actions(state->clusters[i]->simhash, 0), code_chunks);
  }

  if (state->index_type == TK_AGGLO_USE_ANN) {
    state->index.ann = tk_ann_create_base(L, features, state->n_clusters);
    int Ai = tk_lua_absindex(L, -1);
    if (cluster_ids->n > 0)
      tk_ann_add(L, state->index.ann, Ai, cluster_ids, (char *)cluster_codes->a);
    lua_remove(L, -3); // pop cluster_ids
    lua_remove(L, -2); // pop cluster_codes
  } else {
    state->index.hbi = tk_hbi_create(L, features);
    int Ai = tk_lua_absindex(L, -1);
    if (cluster_ids->n > 0)
      tk_hbi_add(L, state->index.hbi, Ai, cluster_ids, (char *)cluster_codes->a);
    lua_remove(L, -3); // pop cluster_ids
    lua_remove(L, -2); // pop cluster_codes
  }

  uint64_t iteration = 0;
  if (callback)
    callback(callback_data, iteration, state->n_active_clusters, uids, assignments);
  iteration ++;

  while (state->n_active_clusters > 1) {

    for (unsigned int i = 0; i < n_threads; i++) {
      tk_thread_range(i, n_threads, state->n_clusters, &threads[i].first, &threads[i].last);
      threads[i].stage_data.find_edges.neighbors = state->thread_neighbors[i];
      threads[i].stage_data.find_edges.min_edges = state->thread_min_edges[i];
      threads[i].stage_data.find_edges.min_dist = INFINITY;
    }

    tk_threads_signal(state->pool, TK_AGGLO_STAGE_FIND_MIN_EDGES, 0);

    double global_min = INFINITY;
    for (unsigned int i = 0; i < n_threads; i++)
      if (threads[i].stage_data.find_edges.min_dist < global_min)
        global_min = threads[i].stage_data.find_edges.min_dist;
    if (global_min == INFINITY)
      break;

    tk_pvec_clear(state->candidate_edges);
    for (unsigned int i = 0; i < n_threads; i++) {
      if (threads[i].stage_data.find_edges.min_dist == global_min) {
        tk_pvec_t *edges = threads[i].stage_data.find_edges.min_edges;
        for (uint64_t j = 0; j < edges->n; j++)
          tk_pvec_push(state->candidate_edges, edges->a[j]);
      }
    }

    tk_evec_t *weighted_edges = tk_evec_create(L, 0, 0, 0);
    for (uint64_t i = 0; i < state->candidate_edges->n; i++) {
      tk_pair_t edge = state->candidate_edges->a[i];
      uint64_t c1 = (uint64_t)edge.i;
      uint64_t c2 = (uint64_t)edge.p;
      if (c1 >= state->n_clusters || c2 >= state->n_clusters) continue;
      if (!state->clusters[c1] || !state->clusters[c2]) continue;
      if (!state->clusters[c1]->active || !state->clusters[c2]->active) continue;
      uint64_t merged_size = state->clusters[c1]->size + state->clusters[c2]->size;
      tk_evec_push(weighted_edges, tk_edge((int64_t)c1, (int64_t)c2, (double)merged_size));
    }

    if (weighted_edges->n == 0) {
      lua_pop(L, 1);
      break;
    }

    tk_evec_asc(weighted_edges, 0, weighted_edges->n);
    tk_pvec_clear(state->selected_merges);
    tk_iuset_clear(state->merged_clusters);

    for (uint64_t i = 0; i < weighted_edges->n; i++) {
      tk_edge_t e = weighted_edges->a[i];
      uint64_t c1 = (uint64_t)e.u;
      uint64_t c2 = (uint64_t)e.v;
      khint_t khi1 = tk_iuset_get(state->merged_clusters, (int64_t)c1);
      khint_t khi2 = tk_iuset_get(state->merged_clusters, (int64_t)c2);
      if (khi1 == tk_iuset_end(state->merged_clusters) &&
          khi2 == tk_iuset_end(state->merged_clusters)) {
        tk_pvec_push(state->selected_merges, tk_pair((int64_t)c1, (int64_t)c2));
        int kha;
        tk_iuset_put(state->merged_clusters, (int64_t)c1, &kha);
        tk_iuset_put(state->merged_clusters, (int64_t)c2, &kha);
      }
    }

    lua_pop(L, 1); // pop weighted_edges
    if (state->selected_merges->n == 0)
      break;

    for (unsigned int i = 0; i < n_threads; i++) {
      tk_thread_range(i, n_threads, state->selected_merges->n,
        &threads[i].stage_data.merge.merge_first,
        &threads[i].stage_data.merge.merge_last);
    }

    tk_threads_signal(state->pool, TK_AGGLO_STAGE_MERGE_CLUSTERS, 0);
    int index_stack_top = tk_lua_absindex(L, -1);
    tk_ivec_t *temp_id = tk_ivec_create(L, 0, 0, 0);
    for (uint64_t i = 0; i < state->selected_merges->n; i++) {
      tk_pair_t merge = state->selected_merges->a[i];
      uint64_t from_idx = (uint64_t)merge.i;
      uint64_t to_idx = (uint64_t)merge.p;
      tk_agglo_cluster_t *from = state->clusters[from_idx];
      tk_agglo_cluster_t *to = state->clusters[to_idx];
      if (!from || !to)
        continue;
      if (state->index_type == TK_AGGLO_USE_ANN)
        tk_ann_remove(L, state->index.ann, from->cluster_id);
      else
        tk_hbi_remove(L, state->index.hbi, from->cluster_id);
    }

    for (uint64_t i = 0; i < state->selected_merges->n; i++) {
      tk_pair_t merge = state->selected_merges->a[i];
      uint64_t from_idx = (uint64_t)merge.i;
      uint64_t to_idx = (uint64_t)merge.p;
      tk_agglo_cluster_t *from = state->clusters[from_idx];
      tk_agglo_cluster_t *to = state->clusters[to_idx];
      if (!from || !to)
        continue;
      for (uint64_t j = 0; j < to->members->n; j++) {
        int64_t uid = to->members->a[j];
        khint_t khi = tk_iumap_get(state->uid_to_vec_idx, uid);
        if (khi != tk_iumap_end(state->uid_to_vec_idx)) {
          uint64_t vec_idx = (uint64_t)tk_iumap_val(state->uid_to_vec_idx, khi);
          assignments->a[vec_idx] = to->cluster_id;
        }
      }

      tk_ivec_clear(temp_id);
      tk_ivec_push(temp_id, to->cluster_id);
      tk_bits_t *to_code = tk_simhash_actions(to->simhash, 0);
      if (state->index_type == TK_AGGLO_USE_ANN) {
        tk_ann_remove(L, state->index.ann, to->cluster_id);
        tk_ann_add(L, state->index.ann, index_stack_top, temp_id, (char *)to_code);
      } else {
        tk_hbi_remove(L, state->index.hbi, to->cluster_id);
        tk_hbi_add(L, state->index.hbi, index_stack_top, temp_id, (char *)to_code);
      }

      state->n_active_clusters--;
      iteration++;
    }
    if (callback)
      callback(callback_data, iteration, state->n_active_clusters, uids, assignments);

    lua_pop(L, 1); // pop temp_id
  }

  free(threads);
  if (state->index_type == TK_AGGLO_USE_ANN && state->index.ann)
    lua_pop(L, 1);
  else if (state->index_type == TK_AGGLO_USE_HBI && state->index.hbi)
    lua_pop(L, 1);
  lua_pop(L, 1);
  return 0;
}

static inline void tk_cluster_agglo (
  tk_cluster_opts_t *opts,
  tk_agglo_callback_t callback,
  void *callback_data
) {
  if (!opts->ann && !opts->hbi) {
    *opts->n_clustersp = 0;
    return;
  }
  uint64_t features = 0;
  if (opts->hbi) {
    features = opts->hbi->features;
  } else if (opts->ann) {
    features = opts->ann->features;
  }
  if (features == 0) {
    *opts->n_clustersp = 0;
    return;
  }
  tk_agglo_index_type_t index_type = opts->hbi ? TK_AGGLO_USE_HBI : TK_AGGLO_USE_ANN;
  unsigned int n_threads = opts->n_threads > 0 ? opts->n_threads : 1;
  uint64_t probe_radius = opts->probe_radius > 0 ? opts->probe_radius : 3;
  int result = tk_agglo(
    opts->L,
    opts->ann,
    opts->hbi,
    opts->ids,
    features,
    index_type,
    probe_radius,
    n_threads,
    opts->assignments,
    callback,
    callback_data
  );
  if (result == 0) {
    tk_iuset_t *unique_clusters = tk_iuset_create(0, 0);
    for (uint64_t i = 0; i < opts->ids->n; i++) {
      if (opts->assignments->a[i] >= 0) {  // Skip UIDs without vectors
        int kha;
        tk_iuset_put(unique_clusters, opts->assignments->a[i], &kha);
      }
    }
    *opts->n_clustersp = (uint64_t)tk_iuset_size(unique_clusters);
    tk_iuset_destroy(unique_clusters);
  } else {
    *opts->n_clustersp = 0;
  }
}

static inline void tk_cluster (tk_cluster_opts_t *opts) {
  if (opts->use_agglo) {
    tk_cluster_agglo(opts, NULL, NULL);
  } else if (opts->depth > 0) {
    tk_cluster_prefix(opts);
  } else {
    tk_cluster_dsu(opts);
  }
}

#endif
