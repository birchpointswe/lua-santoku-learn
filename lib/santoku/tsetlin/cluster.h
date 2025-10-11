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

typedef enum {
  TK_AGGLO_USE_ANN,
  TK_AGGLO_USE_HBI
} tk_agglo_index_type_t;

typedef enum {
  TK_AGGLO_LINKAGE_SIMHASH,
  TK_AGGLO_LINKAGE_SINGLE
} tk_agglo_linkage_t;

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
  tk_agglo_linkage_t linkage;
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
  // Single-linkage specific fields:
  uint64_t knn;
  uint64_t min_pts;
  bool assign_noise;
  tk_inv_hoods_t *inv_hoods;
  tk_ann_hoods_t *ann_hoods;
  tk_hbi_hoods_t *hbi_hoods;
  tk_ivec_t *hoods_uids;
  tk_iumap_t *uid_to_hood_idx;
  tk_ivec_t *hood_offsets;  // Current offset into each neighborhood list
  tk_cvec_t *is_core;
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

      if (state->linkage == TK_AGGLO_LINKAGE_SINGLE) {
        // Single-linkage mode: scan neighborhoods for each cluster's members
        for (uint64_t i = data->first; i <= data->last && i < state->n_clusters; i++) {
          tk_agglo_cluster_t *cluster = state->clusters[i];
          if (!cluster || !cluster->active) continue;

          int64_t best_neighbor_cluster_idx = -1;
          int64_t best_neighbor_distance = LLONG_MAX;

          // For each member of this cluster
          for (uint64_t m = 0; m < cluster->members->n; m++) {
            int64_t member_uid = cluster->members->a[m];

            // Get member's hood index
            if (state->uid_to_hood_idx == NULL) continue;
            khint_t khi = tk_iumap_get(state->uid_to_hood_idx, member_uid);
            if (khi == tk_iumap_end(state->uid_to_hood_idx)) continue;
            int64_t hood_idx = tk_iumap_val(state->uid_to_hood_idx, khi);
            if (hood_idx < 0 || hood_idx >= (int64_t)state->hoods_uids->n) continue;

            // Get current offset for this hood
            uint64_t offset = (state->hood_offsets != NULL && hood_idx < (int64_t)state->hood_offsets->n)
                              ? (uint64_t) state->hood_offsets->a[hood_idx] : 0;

            // Scan forward from offset to find first neighbor in different cluster
            if (state->inv_hoods != NULL && hood_idx < (int64_t)state->inv_hoods->n) {
              tk_rvec_t *hood = state->inv_hoods->a[hood_idx];
              for (uint64_t j = offset; j < hood->n; j++) {
                int64_t neighbor_hood_idx = hood->a[j].i;
                if (neighbor_hood_idx < 0 || neighbor_hood_idx >= (int64_t)state->hoods_uids->n) continue;
                int64_t neighbor_uid = state->hoods_uids->a[neighbor_hood_idx];

                // If min_pts > 0, skip if neighbor is not core
                if (state->min_pts > 0 && state->is_core != NULL) {
                  if (neighbor_hood_idx >= (int64_t)(state->is_core->n * 8)) continue;
                  if (!(state->is_core->a[TK_CVEC_BITS_BYTE(neighbor_hood_idx)] & (1u << TK_CVEC_BITS_BIT(neighbor_hood_idx)))) {
                    continue;
                  }
                }

                // Check if neighbor is in a different cluster
                khint_t khi_n = tk_iumap_get(state->uid_to_cluster, neighbor_uid);
                if (khi_n == tk_iumap_end(state->uid_to_cluster)) continue;
                uint64_t neighbor_cluster_idx = (uint64_t)tk_iumap_val(state->uid_to_cluster, khi_n);
                if (neighbor_cluster_idx == i) continue;
                if (!state->clusters[neighbor_cluster_idx] || !state->clusters[neighbor_cluster_idx]->active) continue;

                // Found a valid neighbor in different cluster
                int64_t distance = hood->a[j].d;
                if (neighbor_cluster_idx != (uint64_t)best_neighbor_cluster_idx || distance < best_neighbor_distance) {
                  best_neighbor_cluster_idx = (int64_t)neighbor_cluster_idx;
                  best_neighbor_distance = distance;
                }
                break; // Move to next member after finding first valid neighbor
              }
            } else if (state->ann_hoods != NULL && hood_idx < (int64_t)state->ann_hoods->n) {
              tk_pvec_t *hood = state->ann_hoods->a[hood_idx];
              for (uint64_t j = offset; j < hood->n; j++) {
                int64_t neighbor_hood_idx = hood->a[j].i;
                if (neighbor_hood_idx < 0 || neighbor_hood_idx >= (int64_t)state->hoods_uids->n) continue;
                int64_t neighbor_uid = state->hoods_uids->a[neighbor_hood_idx];

                // If min_pts > 0, skip if neighbor is not core
                if (state->min_pts > 0 && state->is_core != NULL) {
                  if (neighbor_hood_idx >= (int64_t)(state->is_core->n * 8)) continue;
                  if (!(state->is_core->a[TK_CVEC_BITS_BYTE(neighbor_hood_idx)] & (1u << TK_CVEC_BITS_BIT(neighbor_hood_idx)))) {
                    continue;
                  }
                }

                // Check if neighbor is in a different cluster
                khint_t khi_n = tk_iumap_get(state->uid_to_cluster, neighbor_uid);
                if (khi_n == tk_iumap_end(state->uid_to_cluster)) continue;
                uint64_t neighbor_cluster_idx = (uint64_t)tk_iumap_val(state->uid_to_cluster, khi_n);
                if (neighbor_cluster_idx == i) continue;
                if (!state->clusters[neighbor_cluster_idx] || !state->clusters[neighbor_cluster_idx]->active) continue;

                // Found a valid neighbor in different cluster
                int64_t distance = hood->a[j].p;
                if (neighbor_cluster_idx != (uint64_t)best_neighbor_cluster_idx || distance < best_neighbor_distance) {
                  best_neighbor_cluster_idx = (int64_t)neighbor_cluster_idx;
                  best_neighbor_distance = distance;
                }
                break; // Move to next member after finding first valid neighbor
              }
            } else if (state->hbi_hoods != NULL && hood_idx < (int64_t)state->hbi_hoods->n) {
              tk_pvec_t *hood = state->hbi_hoods->a[hood_idx];
              for (uint64_t j = offset; j < hood->n; j++) {
                int64_t neighbor_hood_idx = hood->a[j].i;
                if (neighbor_hood_idx < 0 || neighbor_hood_idx >= (int64_t)state->hoods_uids->n) continue;
                int64_t neighbor_uid = state->hoods_uids->a[neighbor_hood_idx];

                // If min_pts > 0, skip if neighbor is not core
                if (state->min_pts > 0 && state->is_core != NULL) {
                  if (neighbor_hood_idx >= (int64_t)(state->is_core->n * 8)) continue;
                  if (!(state->is_core->a[TK_CVEC_BITS_BYTE(neighbor_hood_idx)] & (1u << TK_CVEC_BITS_BIT(neighbor_hood_idx)))) {
                    continue;
                  }
                }

                // Check if neighbor is in a different cluster
                khint_t khi_n = tk_iumap_get(state->uid_to_cluster, neighbor_uid);
                if (khi_n == tk_iumap_end(state->uid_to_cluster)) continue;
                uint64_t neighbor_cluster_idx = (uint64_t)tk_iumap_val(state->uid_to_cluster, khi_n);
                if (neighbor_cluster_idx == i) continue;
                if (!state->clusters[neighbor_cluster_idx] || !state->clusters[neighbor_cluster_idx]->active) continue;

                // Found a valid neighbor in different cluster
                int64_t distance = hood->a[j].p;
                if (neighbor_cluster_idx != (uint64_t)best_neighbor_cluster_idx || distance < best_neighbor_distance) {
                  best_neighbor_cluster_idx = (int64_t)neighbor_cluster_idx;
                  best_neighbor_distance = distance;
                }
                break; // Move to next member after finding first valid neighbor
              }
            }
          }

          // Track minimum distance edge for this cluster
          if (best_neighbor_cluster_idx >= 0) {
            uint64_t c1 = i, c2 = (uint64_t)best_neighbor_cluster_idx;
            if (state->clusters[c1]->size > state->clusters[c2]->size ||
                (state->clusters[c1]->size == state->clusters[c2]->size && c1 > c2)) {
              uint64_t tmp = c1; c1 = c2; c2 = tmp;
            }
            double dist = (double)best_neighbor_distance;
            if (dist < min_dist) {
              min_dist = dist;
              tk_pvec_clear(min_edges);
            }
            if (dist == min_dist) {
              tk_pvec_push(min_edges, tk_pair((int64_t)c1, (int64_t)c2));
            }
          }
        }
      } else {
        // Simhash linkage mode: use existing simhash-based logic
        for (uint64_t i = data->first; i <= data->last && i < state->n_clusters; i++) {
          tk_agglo_cluster_t *cluster = state->clusters[i];
          if (!cluster || !cluster->active) continue;
          tk_pvec_clear(neighbors);
          tk_bits_t *code = tk_simhash_actions(cluster->simhash, 0);
          if (state->index_type == TK_AGGLO_USE_ANN) {
            tk_ann_neighbors_by_vec(state->index.ann, (char *)code,
                                     cluster->cluster_id, 1, state->probe_radius, 0, -1, neighbors);
          } else {
            tk_hbi_neighbors_by_vec(state->index.hbi, (char *)code,
                                     cluster->cluster_id, 1, 0, state->probe_radius, neighbors);
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
  tk_agglo_linkage_t linkage,
  uint64_t probe_radius,
  uint64_t knn,
  uint64_t min_pts,
  bool assign_noise,
  tk_inv_hoods_t *inv_hoods,
  tk_ann_hoods_t *ann_hoods,
  tk_hbi_hoods_t *hbi_hoods,
  tk_ivec_t *hoods_uids,
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
  state->linkage = linkage;
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
  state->knn = knn;
  state->min_pts = min_pts;
  state->assign_noise = assign_noise;
  state->inv_hoods = inv_hoods;
  state->ann_hoods = ann_hoods;
  state->hbi_hoods = hbi_hoods;
  state->hoods_uids = hoods_uids;
  state->uid_to_hood_idx = NULL;
  state->hood_offsets = NULL;
  state->is_core = NULL;

  // Single-linkage initialization
  if (linkage == TK_AGGLO_LINKAGE_SINGLE && hoods_uids != NULL) {
    // Create uid_to_hood_idx map
    state->uid_to_hood_idx = tk_iumap_create(L, 0);
    tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
    lua_pop(L, 1);

    for (uint64_t i = 0; i < hoods_uids->n; i++) {
      int kha;
      khint_t khi = tk_iumap_put(state->uid_to_hood_idx, hoods_uids->a[i], &kha);
      tk_iumap_setval(state->uid_to_hood_idx, khi, (int64_t)i);
    }

    // Create hood_offsets ivec
    state->hood_offsets = tk_ivec_create(L, hoods_uids->n, 0, 0);
    tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
    lua_pop(L, 1);
    for (uint64_t i = 0; i < hoods_uids->n; i++) {
      state->hood_offsets->a[i] = 0;
    }
    state->hood_offsets->n = hoods_uids->n;

    // If min_pts > 0: Create is_core bitmap and scan hoods
    if (min_pts > 0) {
      uint64_t n_bits = hoods_uids->n;
      uint64_t n_bytes = TK_CVEC_BITS_BYTES(n_bits);
      state->is_core = tk_cvec_create(L, n_bytes, 0, 0);
      tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
      lua_pop(L, 1);

      // Initialize all bits to 0
      for (uint64_t i = 0; i < n_bytes; i++) {
        state->is_core->a[i] = 0;
      }
      state->is_core->n = n_bytes;

      // Scan hoods counting degree >= min_pts
      for (uint64_t i = 0; i < hoods_uids->n; i++) {
        uint64_t degree = 0;

        if (inv_hoods != NULL && i < inv_hoods->n) {
          degree = inv_hoods->a[i]->n;
        } else if (ann_hoods != NULL && i < ann_hoods->n) {
          degree = ann_hoods->a[i]->n;
        } else if (hbi_hoods != NULL && i < hbi_hoods->n) {
          degree = hbi_hoods->a[i]->n;
        }

        if (degree >= min_pts) {
          state->is_core->a[TK_CVEC_BITS_BYTE(i)] |= (1u << TK_CVEC_BITS_BIT(i));
        }
      }
    }
  }

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

    // For single-linkage with min_pts: check if this UID is core
    bool is_core_point = true;
    if (linkage == TK_AGGLO_LINKAGE_SINGLE && min_pts > 0 && state->is_core != NULL && state->uid_to_hood_idx != NULL) {
      khint_t khi_hood = tk_iumap_get(state->uid_to_hood_idx, uid);
      if (khi_hood != tk_iumap_end(state->uid_to_hood_idx)) {
        int64_t hood_idx = tk_iumap_val(state->uid_to_hood_idx, khi_hood);
        if (hood_idx >= 0 && hood_idx < (int64_t)(state->is_core->n * 8)) {
          is_core_point = (state->is_core->a[TK_CVEC_BITS_BYTE(hood_idx)] & (1u << TK_CVEC_BITS_BIT(hood_idx))) != 0;
        } else {
          is_core_point = false;
        }
      } else {
        is_core_point = false;  // Not in hoods, so not core
      }
    }

    tk_bits_t *code = (tk_bits_t *)vec_ptr;
    uint64_t code_hash = tk_agglo_hash_code(code, code_chunks);
    khint_t khi = tk_iumap_get(code_hash_to_cluster, (int64_t)code_hash);
    uint64_t cluster_idx;

    // Only merge into existing same-code cluster if both are core (in single-linkage mode)
    bool can_merge_same_code = true;
    if (linkage == TK_AGGLO_LINKAGE_SINGLE && min_pts > 0) {
      can_merge_same_code = is_core_point;
    }

    if (khi == tk_iumap_end(code_hash_to_cluster) || !can_merge_same_code) {
      // Create new cluster
      if (state->n_clusters >= max_unique_codes)
        tk_error(L, "tk_agglo: too many unique clusters", ENOMEM);
      cluster_idx = state->n_clusters;
      // Cluster IDs start after metadata + samples + hop_counts region
      // Layout: [n_samples, samples[n], hop_counts[n], clusters...]
      // So cluster IDs start at: 1 + n_samples + n_samples = 2*n_samples + 1
      int64_t cluster_id = (int64_t)(2 * n_samples + 1 + cluster_idx);
      tk_agglo_cluster_t *cluster = tk_agglo_cluster_create(
        L, state_idx, cluster_id, features, state_bits);
      lua_pop(L, 1);
      state->clusters[cluster_idx] = cluster;
      state->n_clusters_created++;  // Track for cleanup
      int kha;
      if (can_merge_same_code) {
        // Only register in hash if we're allowing same-code merges
        khi = tk_iumap_put(code_hash_to_cluster, (int64_t)code_hash, &kha);
        tk_iumap_setval(code_hash_to_cluster, khi, (int64_t)cluster_idx);
      }
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
      break;  // No valid merges possible, clustering complete

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

      // For single-linkage: increment hood_offsets for merged clusters' members
      if (state->linkage == TK_AGGLO_LINKAGE_SINGLE && state->hood_offsets != NULL && state->uid_to_hood_idx != NULL) {
        for (uint64_t j = 0; j < to->members->n; j++) {
          int64_t uid = to->members->a[j];
          khint_t khi = tk_iumap_get(state->uid_to_hood_idx, uid);
          if (khi != tk_iumap_end(state->uid_to_hood_idx)) {
            int64_t hood_idx = tk_iumap_val(state->uid_to_hood_idx, khi);
            if (hood_idx >= 0 && hood_idx < (int64_t)state->hood_offsets->n) {
              state->hood_offsets->a[hood_idx]++;
            }
          }
        }
      }

      state->n_active_clusters--;
    }

    if (state->selected_merges->n > 0) {
      // Call callback once per merge round (after all merges in this round)
      iteration++;
      if (callback)
        callback(callback_data, iteration, state->n_active_clusters, uids, assignments);
    }

    lua_pop(L, 1); // pop temp_id
  }

  // Noise assignment for single-linkage mode
  if (state->linkage == TK_AGGLO_LINKAGE_SINGLE && state->assign_noise && state->is_core != NULL) {
    for (uint64_t i = 0; i < n_samples; i++) {
      int64_t uid = uids->a[i];

      // Get hood index for this uid
      if (state->uid_to_hood_idx == NULL) continue;
      khint_t khi_hood = tk_iumap_get(state->uid_to_hood_idx, uid);
      if (khi_hood == tk_iumap_end(state->uid_to_hood_idx)) continue;
      int64_t hood_idx = tk_iumap_val(state->uid_to_hood_idx, khi_hood);
      if (hood_idx < 0 || hood_idx >= (int64_t)state->hoods_uids->n) continue;

      // Skip if this point is core
      if (hood_idx < (int64_t)(state->is_core->n * 8)) {
        if (state->is_core->a[TK_CVEC_BITS_BYTE(hood_idx)] & (1u << TK_CVEC_BITS_BIT(hood_idx))) {
          continue;
        }
      }

      // Find nearest core cluster from its hood
      int64_t nearest_core_cluster = -1;

      if (state->inv_hoods != NULL && hood_idx < (int64_t)state->inv_hoods->n) {
        tk_rvec_t *hood = state->inv_hoods->a[hood_idx];
        for (uint64_t j = 0; j < hood->n; j++) {
          int64_t neighbor_hood_idx = hood->a[j].i;
          if (neighbor_hood_idx < 0 || neighbor_hood_idx >= (int64_t)state->hoods_uids->n) continue;

          // Check if neighbor is core
          if (neighbor_hood_idx >= (int64_t)(state->is_core->n * 8)) continue;
          if (!(state->is_core->a[TK_CVEC_BITS_BYTE(neighbor_hood_idx)] & (1u << TK_CVEC_BITS_BIT(neighbor_hood_idx)))) {
            continue;
          }

          int64_t neighbor_uid = state->hoods_uids->a[neighbor_hood_idx];
          khint_t khi_n = tk_iumap_get(state->uid_to_cluster, neighbor_uid);
          if (khi_n != tk_iumap_end(state->uid_to_cluster)) {
            uint64_t neighbor_cluster_idx = (uint64_t)tk_iumap_val(state->uid_to_cluster, khi_n);
            if (state->clusters[neighbor_cluster_idx] && state->clusters[neighbor_cluster_idx]->active) {
              nearest_core_cluster = state->clusters[neighbor_cluster_idx]->cluster_id;
              break;
            }
          }
        }
      } else if (state->ann_hoods != NULL && hood_idx < (int64_t)state->ann_hoods->n) {
        tk_pvec_t *hood = state->ann_hoods->a[hood_idx];
        for (uint64_t j = 0; j < hood->n; j++) {
          int64_t neighbor_hood_idx = hood->a[j].i;
          if (neighbor_hood_idx < 0 || neighbor_hood_idx >= (int64_t)state->hoods_uids->n) continue;

          // Check if neighbor is core
          if (neighbor_hood_idx >= (int64_t)(state->is_core->n * 8)) continue;
          if (!(state->is_core->a[TK_CVEC_BITS_BYTE(neighbor_hood_idx)] & (1u << TK_CVEC_BITS_BIT(neighbor_hood_idx)))) {
            continue;
          }

          int64_t neighbor_uid = state->hoods_uids->a[neighbor_hood_idx];
          khint_t khi_n = tk_iumap_get(state->uid_to_cluster, neighbor_uid);
          if (khi_n != tk_iumap_end(state->uid_to_cluster)) {
            uint64_t neighbor_cluster_idx = (uint64_t)tk_iumap_val(state->uid_to_cluster, khi_n);
            if (state->clusters[neighbor_cluster_idx] && state->clusters[neighbor_cluster_idx]->active) {
              nearest_core_cluster = state->clusters[neighbor_cluster_idx]->cluster_id;
              break;
            }
          }
        }
      } else if (state->hbi_hoods != NULL && hood_idx < (int64_t)state->hbi_hoods->n) {
        tk_pvec_t *hood = state->hbi_hoods->a[hood_idx];
        for (uint64_t j = 0; j < hood->n; j++) {
          int64_t neighbor_hood_idx = hood->a[j].i;
          if (neighbor_hood_idx < 0 || neighbor_hood_idx >= (int64_t)state->hoods_uids->n) continue;

          // Check if neighbor is core
          if (neighbor_hood_idx >= (int64_t)(state->is_core->n * 8)) continue;
          if (!(state->is_core->a[TK_CVEC_BITS_BYTE(neighbor_hood_idx)] & (1u << TK_CVEC_BITS_BIT(neighbor_hood_idx)))) {
            continue;
          }

          int64_t neighbor_uid = state->hoods_uids->a[neighbor_hood_idx];
          khint_t khi_n = tk_iumap_get(state->uid_to_cluster, neighbor_uid);
          if (khi_n != tk_iumap_end(state->uid_to_cluster)) {
            uint64_t neighbor_cluster_idx = (uint64_t)tk_iumap_val(state->uid_to_cluster, khi_n);
            if (state->clusters[neighbor_cluster_idx] && state->clusters[neighbor_cluster_idx]->active) {
              nearest_core_cluster = state->clusters[neighbor_cluster_idx]->cluster_id;
              break;
            }
          }
        }
      }

      // Assign to nearest core cluster or mark as noise
      assignments->a[i] = nearest_core_cluster;
    }
  }

  free(threads);
  if (state->index_type == TK_AGGLO_USE_ANN && state->index.ann)
    lua_pop(L, 1);
  else if (state->index_type == TK_AGGLO_USE_HBI && state->index.hbi)
    lua_pop(L, 1);
  lua_pop(L, 1);
  return 0;
}

#endif
