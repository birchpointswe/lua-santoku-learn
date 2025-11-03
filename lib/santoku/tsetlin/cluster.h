#ifndef TK_CLUSTER_H
#define TK_CLUSTER_H

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/dsu.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/ann.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/tsetlin/centroid.h>
#include <omp.h>
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
  TK_AGGLO_LINKAGE_CENTROID,
  TK_AGGLO_LINKAGE_SINGLE
} tk_agglo_linkage_t;

typedef struct {
  int64_t cluster_id;
  tk_centroid_t *centroid;
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
  tk_evec_t *weighted_edges;
  tk_pvec_t *selected_merges;
  tk_iuset_t *merged_clusters;
  tk_ivec_t *assignments;
  lua_State *L;
  bool is_userdata;
  uint64_t n_clusters_created;
  uint64_t knn;
  uint64_t min_pts;
  bool assign_noise;
  tk_ivec_t *adj_ids;
  tk_ivec_t *adj_offsets;
  tk_ivec_t *adj_neighbors;
  tk_dvec_t *adj_weights;
  tk_iumap_t *uid_to_adj_idx;
  tk_ivec_t *adj_scan_offsets;
  tk_cvec_t *is_core;
  uint64_t n_adj;
  double current_epsilon;
} tk_agglo_state_t;

static inline uint64_t tk_agglo_hash_code (char *code, uint64_t code_chunks) {
  uint64_t hash = 0xcbf29ce484222325ULL;
  for (uint64_t i = 0; i < code_chunks; i++) {
    hash ^= (uint64_t)(uint8_t)code[i];
    hash *= 0x100000001b3ULL;
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
  uint64_t state_bits,
  tk_agglo_linkage_t linkage
) {
  tk_agglo_cluster_t *cluster;
  int cluster_idx = -1;
  if (L) {
    cluster = lua_newuserdata(L, sizeof(tk_agglo_cluster_t));
    cluster_idx = lua_gettop(L);
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
  if (linkage == TK_AGGLO_LINKAGE_CENTROID) {
    uint8_t tail_mask = (features % TK_CVEC_BITS == 0) ?
      0xFF : (uint8_t)((1 << (features % TK_CVEC_BITS)) - 1);
    cluster->centroid = tk_centroid_create(L, code_chunks, tail_mask);
    if (!cluster->centroid) {
      tk_ivec_destroy(cluster->members);
      if (!cluster->is_userdata)
        free(cluster);
      return NULL;
    }
  } else {
    cluster->centroid = NULL;
  }
  if (L && state_idx > 0)
    tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, cluster_idx);
  return cluster;
}

static inline void tk_agglo_cluster_destroy (
  tk_agglo_cluster_t *cluster
) {
  if (!cluster)
    return;
  if (cluster->centroid) {
    tk_centroid_destroy(cluster->centroid);
    cluster->centroid = NULL;
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
  char *code,
  uint64_t code_chunks
) {
  tk_ivec_push(cluster->members, uid);
  if (cluster->centroid) {
    tk_centroid_add_member(cluster->centroid, code, code_chunks);
    cluster->size = cluster->centroid->size;
  } else {
    cluster->size = cluster->members->n;
  }
}

static inline bool tk_agglo_cluster_merge (
  tk_agglo_cluster_t *to,
  tk_agglo_cluster_t *from
) {
  for (uint64_t i = 0; i < from->members->n; i++)
    tk_ivec_push(to->members, from->members->a[i]);
  bool changed = false;
  if (to->centroid && from->centroid) {
    changed = tk_centroid_merge(to->centroid, from->centroid);
    to->size = to->centroid->size;
  } else {
    to->size = to->members->n;
  }
  from->active = false;
  from->size = 0;
  tk_ivec_clear(from->members);
  return changed;
}

static inline bool tk_agglo_is_core(tk_cvec_t *is_core, int64_t hood_idx, uint64_t n_hoods) {
  if (!is_core || hood_idx < 0 || hood_idx >= (int64_t)n_hoods)
    return false;
  return ((uint8_t) is_core->a[TK_CVEC_BITS_BYTE(hood_idx)] & (1u << TK_CVEC_BITS_BIT(hood_idx))) != 0;
}

static inline int64_t tk_agglo_find_nearest_core_cluster(
  tk_agglo_state_t *state,
  int64_t adj_idx
) {
  int64_t nearest = -1;
  if (state->adj_offsets != NULL && adj_idx >= 0 && adj_idx < (int64_t)state->adj_ids->n) {
    int64_t start = state->adj_offsets->a[adj_idx];
    int64_t end = state->adj_offsets->a[adj_idx + 1];
    for (int64_t j = start; j < end; j++) {
      int64_t nh_idx = state->adj_neighbors->a[j];
      if (nh_idx < 0 || nh_idx >= (int64_t)state->adj_ids->n) continue;
      int64_t nh_uid = state->adj_ids->a[nh_idx];
      if (!tk_agglo_is_core(state->is_core, nh_idx, state->n_adj)) continue;
      khint_t khi = tk_iumap_get(state->uid_to_cluster, nh_uid);
      if (khi != tk_iumap_end(state->uid_to_cluster)) {
        uint64_t cidx = (uint64_t)tk_iumap_val(state->uid_to_cluster, khi);
        if (state->clusters[cidx] && state->clusters[cidx]->active)
          return state->clusters[cidx]->cluster_id;
      }
    }
  }
  return nearest;
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
  return 0;
}


static inline void tk_agglo_create_initial_clusters(
  lua_State *L,
  tk_agglo_state_t *state,
  int state_idx,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  tk_ivec_t *uids,
  tk_ivec_t *assignments,
  tk_agglo_linkage_t linkage,
  uint64_t min_pts,
  uint64_t n_samples,
  uint64_t code_chunks,
  uint64_t features,
  uint64_t state_bits,
  tk_iumap_t *code_hash_to_cluster
) {
  state->n_clusters = 0;
  bool adjacency_only = (ann == NULL && hbi == NULL && linkage == TK_AGGLO_LINKAGE_SINGLE && state->uid_to_adj_idx != NULL);
  static char dummy_code[64] = {0};

  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t uid = uids->a[i];
    char *vec_ptr = ann ? tk_ann_get(ann, uid) : (hbi ? tk_hbi_get(hbi, uid) : NULL);

    if (adjacency_only) {
      khint_t khi_adj = tk_iumap_get(state->uid_to_adj_idx, uid);
      if (khi_adj == tk_iumap_end(state->uid_to_adj_idx))
        continue;
    } else if (!vec_ptr) {
      continue;
    }

    bool is_core_point = true;
    if (linkage == TK_AGGLO_LINKAGE_SINGLE && min_pts > 0 && state->is_core != NULL && state->uid_to_adj_idx != NULL) {
      khint_t khi_adj = tk_iumap_get(state->uid_to_adj_idx, uid);
      if (khi_adj != tk_iumap_end(state->uid_to_adj_idx)) {
        int64_t adj_idx = tk_iumap_val(state->uid_to_adj_idx, khi_adj);
        if (adj_idx >= 0 && adj_idx < (int64_t)state->n_adj) {
          is_core_point = tk_agglo_is_core(state->is_core, adj_idx, state->n_adj);
        } else {
          is_core_point = false;
        }
      } else {
        is_core_point = false;
      }
    }

    char *code = vec_ptr ? (char *)vec_ptr : dummy_code;
    uint64_t cluster_idx;

    if (adjacency_only) {
      cluster_idx = state->n_clusters;
      int64_t cluster_id = (int64_t)(2 * n_samples + 1 + cluster_idx);
      tk_agglo_cluster_t *cluster = tk_agglo_cluster_create(L, state_idx, cluster_id, features, state_bits, linkage);
      lua_pop(L, 1);
      state->clusters[cluster_idx] = cluster;
      state->n_clusters_created++;
      int kha;
      khint_t khi2 = tk_iumap_put(state->cluster_id_to_idx, cluster->cluster_id, &kha);
      tk_iumap_setval(state->cluster_id_to_idx, khi2, (int64_t)cluster_idx);
      state->n_clusters++;
    } else {
      uint64_t code_hash = tk_agglo_hash_code(code, code_chunks);
      khint_t khi = tk_iumap_get(code_hash_to_cluster, (int64_t)code_hash);
      bool can_merge_same_code = (linkage != TK_AGGLO_LINKAGE_SINGLE) || is_core_point;
      if (khi == tk_iumap_end(code_hash_to_cluster) || !can_merge_same_code) {
        cluster_idx = state->n_clusters;
        int64_t cluster_id = (int64_t)(2 * n_samples + 1 + cluster_idx);
        tk_agglo_cluster_t *cluster = tk_agglo_cluster_create(L, state_idx, cluster_id, features, state_bits, linkage);
        lua_pop(L, 1);
        state->clusters[cluster_idx] = cluster;
        state->n_clusters_created++;
        int kha;
        if (can_merge_same_code) {
          khi = tk_iumap_put(code_hash_to_cluster, (int64_t)code_hash, &kha);
          tk_iumap_setval(code_hash_to_cluster, khi, (int64_t)cluster_idx);
        }
        khint_t khi2 = tk_iumap_put(state->cluster_id_to_idx, cluster->cluster_id, &kha);
        tk_iumap_setval(state->cluster_id_to_idx, khi2, (int64_t)cluster_idx);
        state->n_clusters++;
      } else {
        cluster_idx = (uint64_t)tk_iumap_val(code_hash_to_cluster, khi);
      }
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
}

static inline bool tk_agglo_iteration(
  lua_State *L,
  tk_agglo_state_t *state,
  tk_ivec_t *assignments,
  tk_ivec_t *uids,
  uint64_t *iteration,
  tk_ivec_t *dendro_offsets,
  tk_pvec_t *dendro_merges
) {
  double global_min = INFINITY;
  tk_evec_clear(state->weighted_edges);

  #pragma omp parallel
  {
    tk_pvec_t *neighbors = tk_pvec_create(0, 0, 0, 0);
    tk_pvec_t *min_edges = tk_pvec_create(0, 0, 0, 0);
    double min_dist = INFINITY;

    #pragma omp for schedule(static) nowait
    for (uint64_t i = 0; i < state->n_clusters; i++) {
      tk_agglo_cluster_t *cluster = state->clusters[i];
      if (!cluster || !cluster->active) continue;

      if (state->linkage == TK_AGGLO_LINKAGE_SINGLE) {
        int64_t best_neighbor_cluster_idx = -1;
        double best_neighbor_distance = INFINITY;
        for (uint64_t m = 0; m < cluster->members->n; m++) {
          int64_t member_uid = cluster->members->a[m];
          khint_t khi = tk_iumap_get(state->uid_to_adj_idx, member_uid);
          if (khi == tk_iumap_end(state->uid_to_adj_idx)) continue;
          int64_t adj_idx = tk_iumap_val(state->uid_to_adj_idx, khi);
          if (adj_idx < 0 || adj_idx >= (int64_t)state->adj_ids->n) continue;
          uint64_t offset = (state->adj_scan_offsets != NULL &&
                            adj_idx < (int64_t)state->adj_scan_offsets->n)
                           ? (uint64_t)state->adj_scan_offsets->a[adj_idx] : 0;
          int64_t start = state->adj_offsets->a[adj_idx];
          int64_t end = state->adj_offsets->a[adj_idx + 1];
          for (int64_t j = start + (int64_t)offset; j < end; j++) {
            int64_t nh_idx = state->adj_neighbors->a[j];
            if (nh_idx < 0 || nh_idx >= (int64_t)state->adj_ids->n) continue;
            int64_t nh_uid = state->adj_ids->a[nh_idx];
            double similarity = state->adj_weights->a[j];
            double distance = 1.0 - similarity;
            if (distance > state->current_epsilon) {
              break;
            }
            khint_t khi_n = tk_iumap_get(state->uid_to_cluster, nh_uid);
            if (khi_n == tk_iumap_end(state->uid_to_cluster)) continue;
            uint64_t neighbor_cluster_idx = (uint64_t)tk_iumap_val(state->uid_to_cluster, khi_n);
            if (neighbor_cluster_idx == i) continue;
            if (!state->clusters[neighbor_cluster_idx] ||
                !state->clusters[neighbor_cluster_idx]->active) continue;
            if (distance == state->current_epsilon) {
              if (neighbor_cluster_idx != (uint64_t)best_neighbor_cluster_idx ||
                  distance < best_neighbor_distance) {
                best_neighbor_cluster_idx = (int64_t)neighbor_cluster_idx;
                best_neighbor_distance = distance;
              }
            }
          }
        }
        if (best_neighbor_cluster_idx >= 0 && best_neighbor_distance == state->current_epsilon) {
          uint64_t c1 = i, c2 = (uint64_t)best_neighbor_cluster_idx;
          if (state->clusters[c1]->size > state->clusters[c2]->size ||
              (state->clusters[c1]->size == state->clusters[c2]->size && c1 > c2)) {
            uint64_t tmp = c1; c1 = c2; c2 = tmp;
          }
          double dist = best_neighbor_distance;
          if (dist < min_dist) {
            min_dist = dist;
            tk_pvec_clear(min_edges);
          }
          if (dist == min_dist)
            tk_pvec_push(min_edges, tk_pair((int64_t)c1, (int64_t)c2));
        }
      } else {
        tk_pvec_clear(neighbors);
        if (state->index_type == TK_AGGLO_USE_ANN) {
          tk_ann_neighbors_by_id(state->index.ann, cluster->cluster_id, state->knn, state->probe_radius, 0, state->index.ann->features, neighbors);
        } else {
          tk_hbi_neighbors_by_id(state->index.hbi, cluster->cluster_id, state->knn, 0, state->probe_radius, neighbors);
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
            if (dist == min_dist)
              tk_pvec_push(min_edges, tk_pair((int64_t)c1, (int64_t)c2));
          }
        }
      }
    }

    #pragma omp critical
    {
      if (min_dist < global_min) {
        global_min = min_dist;
      }
    }

    #pragma omp barrier

    if (min_dist == global_min) {
      #pragma omp critical
      {
        for (uint64_t j = 0; j < min_edges->n; j++) {
          tk_pair_t edge = min_edges->a[j];
          uint64_t c1 = (uint64_t)edge.i;
          uint64_t c2 = (uint64_t)edge.p;
          if (c1 >= state->n_clusters || c2 >= state->n_clusters) continue;
          if (!state->clusters[c1] || !state->clusters[c2]) continue;
          if (!state->clusters[c1]->active || !state->clusters[c2]->active) continue;
          uint64_t merged_size = state->clusters[c1]->size + state->clusters[c2]->size;
          tk_evec_push(state->weighted_edges, tk_edge((int64_t)c1, (int64_t)c2, (double)merged_size));
        }
      }
    }

    tk_pvec_destroy(neighbors);
    tk_pvec_destroy(min_edges);
  }

  if (global_min == INFINITY)
    return false;
  if (state->weighted_edges->n == 0)
    return false;
  tk_evec_asc(state->weighted_edges, 0, state->weighted_edges->n);
  tk_pvec_clear(state->selected_merges);
  tk_iuset_clear(state->merged_clusters);
  for (uint64_t i = 0; i < state->weighted_edges->n; i++) {
    tk_edge_t e = state->weighted_edges->a[i];
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
  if (state->selected_merges->n == 0)
    return false;

  int index_stack_top = tk_lua_absindex(L, -1);
  tk_ivec_t *temp_id = tk_ivec_create(L, 0, 0, 0);
  uint64_t merges_completed = 0;
  bool centroid_changed = false;

  for (uint64_t i = 0; i < state->selected_merges->n; i++) {
    tk_pair_t merge = state->selected_merges->a[i];
    uint64_t from_idx = (uint64_t)merge.i;
    uint64_t to_idx = (uint64_t)merge.p;
    tk_agglo_cluster_t *from = state->clusters[from_idx];
    tk_agglo_cluster_t *to = state->clusters[to_idx];
    if (!from || !to)
      continue;

    if (state->linkage == TK_AGGLO_LINKAGE_CENTROID) {
      if (state->index_type == TK_AGGLO_USE_ANN)
        tk_ann_remove(L, state->index.ann, from->cluster_id);
      else
        tk_hbi_remove(L, state->index.hbi, from->cluster_id);
    }

    for (uint64_t j = 0; j < from->members->n; j++) {
      int64_t uid = from->members->a[j];
      khint_t khi = tk_iumap_get(state->uid_to_cluster, uid);
      if (khi != tk_iumap_end(state->uid_to_cluster))
        tk_iumap_setval(state->uid_to_cluster, khi, (int64_t)to_idx);
    }

    centroid_changed = tk_agglo_cluster_merge(to, from);

    if (dendro_merges) {
      tk_pvec_push(dendro_merges, tk_pair(from->cluster_id, to->cluster_id));
    }

    khint_t khi = tk_iumap_get(state->cluster_id_to_idx, from->cluster_id);
    if (khi != tk_iumap_end(state->cluster_id_to_idx))
      tk_iumap_del(state->cluster_id_to_idx, khi);

    for (uint64_t j = 0; j < to->members->n; j++) {
      int64_t uid = to->members->a[j];
      khint_t khi2 = tk_iumap_get(state->uid_to_vec_idx, uid);
      if (khi2 != tk_iumap_end(state->uid_to_vec_idx)) {
        uint64_t vec_idx = (uint64_t)tk_iumap_val(state->uid_to_vec_idx, khi2);
        assignments->a[vec_idx] = to->cluster_id;
      }
    }

    if (state->linkage == TK_AGGLO_LINKAGE_CENTROID) {
      tk_ivec_clear(temp_id);
      tk_ivec_push(temp_id, to->cluster_id);
      char *to_code = tk_centroid_code(to->centroid);
      if (state->index_type == TK_AGGLO_USE_ANN) {
        tk_ann_remove(L, state->index.ann, to->cluster_id);
        tk_ann_add(L, state->index.ann, index_stack_top, temp_id, to_code);
      } else {
        tk_hbi_remove(L, state->index.hbi, to->cluster_id);
        tk_hbi_add(L, state->index.hbi, index_stack_top, temp_id, to_code);
      }
    }

    state->n_active_clusters--;
    merges_completed++;

    if (centroid_changed)
      break;
  }
  if (state->linkage == TK_AGGLO_LINKAGE_SINGLE && merges_completed > 0) {
    if (state->adj_scan_offsets) {
      #pragma omp parallel for schedule(static)
      for (uint64_t adj_idx = 0; adj_idx < state->adj_ids->n; adj_idx++) {
        int64_t start = state->adj_offsets->a[adj_idx];
        int64_t end = state->adj_offsets->a[adj_idx + 1];
        int64_t offset = state->adj_scan_offsets->a[adj_idx];

        for (int64_t j = start + offset; j < end; j++) {
          double similarity = state->adj_weights->a[j];
          double d = 1.0 - similarity;
          if (d > state->current_epsilon) {
            state->adj_scan_offsets->a[adj_idx] = j - start;
            break;
          }
        }
        if (start + state->adj_scan_offsets->a[adj_idx] >= end) {
          state->adj_scan_offsets->a[adj_idx] = end - start;
        }
      }
    }

    double next_epsilon = INFINITY;
    #pragma omp parallel for reduction(min:next_epsilon) schedule(dynamic, 8)
    for (uint64_t i = 0; i < state->n_clusters; i++) {
      tk_agglo_cluster_t *cluster = state->clusters[i];
      if (!cluster || !cluster->active) continue;
      for (uint64_t m = 0; m < cluster->members->n; m++) {
        int64_t member_uid = cluster->members->a[m];
        khint_t khi = tk_iumap_get(state->uid_to_adj_idx, member_uid);
        if (khi == tk_iumap_end(state->uid_to_adj_idx)) continue;
        int64_t adj_idx = tk_iumap_val(state->uid_to_adj_idx, khi);
        if (adj_idx < 0 || adj_idx >= (int64_t)state->adj_ids->n) continue;

        int64_t offset = state->adj_scan_offsets->a[adj_idx];
        int64_t start = state->adj_offsets->a[adj_idx];
        int64_t end = state->adj_offsets->a[adj_idx + 1];
        int64_t pos = start + offset;

        if (pos < end) {
          double similarity = state->adj_weights->a[pos];
          double d = 1.0 - similarity;
          if (d < next_epsilon) next_epsilon = d;
        }
      }
    }
    state->current_epsilon = next_epsilon;
  }
  if (merges_completed > 0) {
    (*iteration)++;

    if (dendro_offsets) {
      tk_ivec_push(dendro_offsets, (int64_t)dendro_merges->n);
    }
  }
  lua_pop(L, 1);
  return true;
}

static inline int tk_agglo_single_linkage_dsu (
  lua_State *L,
  tk_ivec_t *adj_ids,
  tk_ivec_t *adj_offsets,
  tk_ivec_t *adj_neighbors,
  tk_dvec_t *adj_weights,
  tk_ivec_t *assignments,
  tk_ivec_t *dendro_offsets,
  tk_pvec_t *dendro_merges
) {
  uint64_t n_entities = adj_ids->n;

  tk_evec_t *edges = tk_evec_create(NULL, 0, 0, 0);
  for (uint64_t i = 0; i < n_entities; i++) {
    int64_t start = adj_offsets->a[i];
    int64_t end = adj_offsets->a[i + 1];
    for (int64_t j = start; j < end; j++) {
      int64_t nbr_idx = adj_neighbors->a[j];
      if (nbr_idx <= (int64_t)i) continue;
      double distance = 1.0 - adj_weights->a[j];
      tk_evec_push(edges, tk_edge((int64_t)i, nbr_idx, distance));
    }
  }
  tk_evec_asc(edges, 0, edges->n);

  tk_dsu_t *dsu = tk_dsu_create(L, adj_ids);
  tk_iumap_t *idx_to_cluster_id = tk_iumap_create(NULL, 0);

  for (uint64_t i = 0; i < n_entities; i++) {
    int64_t cluster_id = (int64_t)(2 * n_entities + 1 + i);
    assignments->a[i] = cluster_id;
    int kha;
    uint32_t khi = tk_iumap_put(idx_to_cluster_id, (int64_t)i, &kha);
    tk_iumap_setval(idx_to_cluster_id, khi, cluster_id);
  }

  if (dendro_offsets) {
    tk_ivec_ensure(dendro_offsets, n_entities + 1);
    for (uint64_t i = 0; i < n_entities; i++)
      dendro_offsets->a[i] = assignments->a[i];
    dendro_offsets->a[n_entities] = 0;
    dendro_offsets->n = n_entities + 1;
  }

  uint64_t edge_idx = 0;
  while (edge_idx < edges->n && tk_dsu_components(dsu) > 1) {
    double cur_dist = edges->a[edge_idx].w;

    while (edge_idx < edges->n && edges->a[edge_idx].w == cur_dist) {
      int64_t i1 = edges->a[edge_idx].u;
      int64_t i2 = edges->a[edge_idx].v;
      edge_idx++;

      int64_t r1 = tk_dsu_findx(dsu, i1);
      int64_t r2 = tk_dsu_findx(dsu, i2);
      if (r1 == r2) continue;

      khint_t k1 = tk_iumap_get(idx_to_cluster_id, r1);
      khint_t k2 = tk_iumap_get(idx_to_cluster_id, r2);
      int64_t cid1 = tk_iumap_val(idx_to_cluster_id, k1);
      int64_t cid2 = tk_iumap_val(idx_to_cluster_id, k2);

      tk_dsu_unionx(dsu, i1, i2);
      int64_t new_root = tk_dsu_findx(dsu, i1);

      int64_t absorbed = cid1 < cid2 ? cid1 : cid2;
      int64_t surviving = cid1 < cid2 ? cid2 : cid1;
      if (dendro_merges) tk_pvec_push(dendro_merges, tk_pair(absorbed, surviving));

      // Update cluster ID mapping - the new root must map to surviving
      int kha;
      khint_t knew = tk_iumap_put(idx_to_cluster_id, new_root, &kha);
      tk_iumap_setval(idx_to_cluster_id, knew, surviving);
    }

    // Update ALL assignments after processing this distance level (O(n) once per distance)
    for (uint64_t j = 0; j < n_entities; j++) {
      int64_t root = tk_dsu_findx(dsu, (int64_t)j);
      khint_t khi = tk_iumap_get(idx_to_cluster_id, root);
      if (khi != tk_iumap_end(idx_to_cluster_id)) {
        assignments->a[j] = tk_iumap_val(idx_to_cluster_id, khi);
      }
    }

    // Record dendrogram offset
    if (dendro_offsets) tk_ivec_push(dendro_offsets, (int64_t)dendro_merges->n);
  }

  tk_evec_destroy(edges);
  tk_iumap_destroy(idx_to_cluster_id);
  lua_pop(L, 1);
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
  tk_ivec_t *adj_ids,
  tk_ivec_t *adj_offsets,
  tk_ivec_t *adj_neighbors,
  tk_dvec_t *adj_weights,
  tk_ivec_t *assignments,
  tk_ivec_t *dendro_offsets,
  tk_pvec_t *dendro_merges
) {
  if (!L || !uids || !assignments)
    return -1;
  if (uids->n == 0)
    return -1;

  if (linkage == TK_AGGLO_LINKAGE_SINGLE && adj_ids)
    return tk_agglo_single_linkage_dsu(L, adj_ids, adj_offsets, adj_neighbors, adj_weights,
                                       assignments, dendro_offsets, dendro_merges);

  if (linkage == TK_AGGLO_LINKAGE_CENTROID && !ann && !hbi)
    return -1;
  if (linkage == TK_AGGLO_LINKAGE_SINGLE && !adj_ids && !ann && !hbi)
    return -1;
  if ((ann || hbi) && features == 0)
    return -1;

  uint64_t n_samples = uids->n;
  uint64_t code_chunks = TK_CVEC_BITS_BYTES(features);
  if ((ann || hbi) && code_chunks == 0)
    return -1;

  uint64_t state_bits = log2(n_samples * 2);
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
  state->assignments = assignments;
  state->n_clusters_created = 0;
  state->knn = knn;
  state->min_pts = min_pts;
  state->assign_noise = assign_noise;
  state->adj_ids = adj_ids;
  state->adj_offsets = adj_offsets;
  state->adj_neighbors = adj_neighbors;
  state->adj_weights = adj_weights;
  state->n_adj = 0;
  state->uid_to_vec_idx = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);

  for (uint64_t i = 0; i < n_samples; i++) {
    int kha;
    khint_t khi = tk_iumap_put(state->uid_to_vec_idx, uids->a[i], &kha);
    tk_iumap_setval(state->uid_to_vec_idx, khi, (int64_t)i);
  }

  state->weighted_edges = tk_evec_create(L, 0, 0, 0);
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

  state->clusters = tk_malloc(L, n_samples * sizeof(tk_agglo_cluster_t *));
  for (uint64_t i = 0; i < n_samples; i ++)
    state->clusters[i] = NULL;

  tk_iumap_t *code_hash_to_cluster = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_AGGLO_EPH, state_idx, -1);
  lua_pop(L, 1);
  tk_agglo_create_initial_clusters(L, state, state_idx, ann, hbi, uids, assignments, linkage, min_pts,
    n_samples, code_chunks, features, state_bits, code_hash_to_cluster);

  if (linkage == TK_AGGLO_LINKAGE_CENTROID) {
    if (state->n_clusters > 0) {
      tk_ivec_t *cluster_ids = tk_ivec_create(L, 0, 0, 0);
      tk_cvec_t *cluster_codes = tk_cvec_create(L, state->n_clusters * code_chunks, 0, 0);
      for (uint64_t i = 0; i < state->n_clusters; i++) {
        tk_ivec_push(cluster_ids, state->clusters[i]->cluster_id);
        memcpy(cluster_codes->a + i * code_chunks, tk_centroid_code(state->clusters[i]->centroid), code_chunks);
      }
      if (state->index_type == TK_AGGLO_USE_ANN) {
        state->index.ann = tk_ann_create_randomized(L, features, 30, state->n_clusters);
        int Ai = tk_lua_absindex(L, -1);
        tk_ann_add(L, state->index.ann, Ai, cluster_ids, (char *)cluster_codes->a);
        lua_remove(L, -3);
        lua_remove(L, -2);
      } else {
        state->index.hbi = tk_hbi_create(L, features);
        int Ai = tk_lua_absindex(L, -1);
        tk_hbi_add(L, state->index.hbi, Ai, cluster_ids, (char *)cluster_codes->a);
        lua_remove(L, -3);
        lua_remove(L, -2);
      }
    } else {
      if (state->index_type == TK_AGGLO_USE_ANN) {
        state->index.ann = tk_ann_create_randomized(L, features, 30, 0);
      } else {
        state->index.hbi = tk_hbi_create(L, features);
      }
    }
  }

  state->current_epsilon = 0;
  uint64_t iteration = 0;

  if (dendro_offsets) {
    tk_ivec_ensure(dendro_offsets, n_samples + 1);
    for (uint64_t i = 0; i < n_samples; i++) {
      dendro_offsets->a[i] = assignments->a[i];
    }
    dendro_offsets->a[n_samples] = 0;
    dendro_offsets->n = n_samples + 1;
  }

  while (state->n_active_clusters > 1) {
    if (!tk_agglo_iteration(L, state, assignments, uids, &iteration, dendro_offsets, dendro_merges))
      break;
  }
  if (state->linkage == TK_AGGLO_LINKAGE_SINGLE && state->assign_noise && state->is_core != NULL) {
    #pragma omp parallel for schedule(static, 64)
    for (uint64_t i = 0; i < n_samples; i++) {
      int64_t uid = uids->a[i];
      if (state->uid_to_adj_idx == NULL) continue;
      khint_t khi_adj = tk_iumap_get(state->uid_to_adj_idx, uid);
      if (khi_adj == tk_iumap_end(state->uid_to_adj_idx)) continue;
      int64_t adj_idx = tk_iumap_val(state->uid_to_adj_idx, khi_adj);
      if (adj_idx < 0 || adj_idx >= (int64_t)state->adj_ids->n) continue;
      if (tk_agglo_is_core(state->is_core, adj_idx, state->n_adj)) continue;
      assignments->a[i] = tk_agglo_find_nearest_core_cluster(state, adj_idx);
    }
  }
  if (state->index_type == TK_AGGLO_USE_ANN && state->index.ann)
    lua_pop(L, 1);
  else if (state->index_type == TK_AGGLO_USE_HBI && state->index.hbi)
    lua_pop(L, 1);
  lua_pop(L, 1);
  return 0;
}

#endif
