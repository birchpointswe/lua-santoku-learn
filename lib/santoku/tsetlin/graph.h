#ifndef TK_GRAPH_H
#define TK_GRAPH_H

#include <santoku/iuset.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>
#include <santoku/euset.h>
#include <santoku/iumap.h>
#include <santoku/pumap.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/ann.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/tsetlin/dsu.h>
#include <float.h>
#include <math.h>
#include <stdatomic.h>

#define TK_GRAPH_MT "tk_graph_t"
#define TK_GRAPH_EPH "tk_graph_eph"

typedef tk_iuset_t * tk_graph_adj_item_t;
#define tk_vec_name tk_graph_adj
#define tk_vec_base tk_graph_adj_item_t
#define tk_vec_destroy_item(...) tk_iuset_destroy(__VA_ARGS__)
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef enum {
  TK_GRAPH_SIGMA,
  TK_GRAPH_REWEIGHT,
  TK_GRAPH_COMPUTE_WEIGHTS,
  TK_GRAPH_GEN_RANDOM_PAIRS,
  TK_GRAPH_GEN_MULTICLASS_POS,
  TK_GRAPH_GEN_MULTICLASS_NEG,
  TK_GRAPH_STAR_HOODS,
  TK_GRAPH_ANCHORS,
  TK_GRAPH_ADJ_HOODS,
} tk_graph_stage_t;

typedef enum {
  TK_GRAPH_WEIGHT_POOL_MIN,
  TK_GRAPH_WEIGHT_POOL_MAX
} tk_graph_weight_pooling_t;

typedef struct tk_graph_thread_s tk_graph_thread_t;

typedef struct tk_graph_s {

  tk_euset_t *pairs;
  tk_graph_adj_t *adj;

  // k-NN topology indices
  tk_inv_t *knn_inv; tk_inv_hoods_t *knn_inv_hoods;
  tk_ann_t *knn_ann; tk_ann_hoods_t *knn_ann_hoods;
  tk_hbi_t *knn_hbi; tk_hbi_hoods_t *knn_hbi_hoods;
  tk_ivec_sim_type_t knn_cmp;  // For knn_inv similarity
  double knn_cmp_alpha, knn_cmp_beta;
  int64_t knn_rank;  // Specific rank to scan when accumulating knn_inv candidates (defaults to category_ranks when knn_index == category_index)

  // Category topology index
  tk_inv_t *category_inv;
  tk_ivec_sim_type_t category_cmp;
  double category_alpha, category_beta;
  uint64_t category_anchors;
  uint64_t category_knn;
  double category_knn_decay;
  uint64_t category_negatives;

  // Weight calculation index
  tk_inv_t *weight_inv;
  tk_ann_t *weight_ann;
  tk_hbi_t *weight_hbi;
  tk_ivec_sim_type_t weight_cmp;
  double weight_alpha, weight_beta;
  tk_graph_weight_pooling_t weight_pooling;

  // Other topology generation
  uint64_t random_pairs;

  tk_ivec_t *uids;
  tk_ivec_t *uids_hoods;
  tk_iumap_t *uids_idx;
  tk_iumap_t *uids_idx_hoods;
  tk_pvec_t *edges;

  double weight_eps;
  int64_t sigma_k;
  double sigma_scale;

  uint64_t knn;
  uint64_t knn_min;
  uint64_t knn_cache;
  double knn_eps;
  bool knn_mutual;
  bool bridge;
  uint64_t probe_radius;
  int64_t category_ranks;  // Number of ranks from category_index to treat as categorical (anchor/negative topology uses ranks [0, category_ranks))

  tk_dvec_t *sigmas;
  uint64_t n_edges;
  tk_dsu_t *dsu;
  int64_t largest_component_root;
  tk_dvec_t *q_weights;
  tk_dvec_t *e_weights;
  tk_dvec_t *inter_weights;

  bool destroyed;

} tk_graph_t;

typedef struct tk_graph_thread_s {
  tk_graph_t *graph;
  uint64_t ifirst, ilast;
  unsigned int index;
  atomic_bool has_error;
  tk_dvec_t *q_weights;
  tk_dvec_t *e_weights;
  tk_dvec_t *inter_weights;
  union {
    struct {
      tk_ivec_t *adj_offset;
      tk_ivec_t *adj_data;
      tk_dvec_t *adj_weights;
      tk_ivec_t *adj_sources;
      int64_t csr_base;
    } recon;
    struct {
      tk_pvec_t *local_pairs;
      tk_ivec_t *ids;
      tk_ivec_t *labels;
      uint64_t edges_per_node;
      tk_ivec_t **ids_by_label;
      uint64_t n_labels;
    } random;
    struct {
      tk_pvec_t *local_pos;
      tk_pvec_t *local_neg;
      tk_ivec_t **class_ids;
      tk_ivec_t **anchors;
      tk_iumap_t *label_of_id;
      uint64_t n_classes;
      tk_inv_t *index;
      double eps_pos;
      double eps_neg;
      uint64_t n_anchors_neg;
      tk_ivec_sim_type_t cmp;
      double cmp_alpha;
      double cmp_beta;
    } multiclass;
    struct {
      tk_pvec_t *local_pairs;
      tk_ivec_t *ids;
      tk_inv_hoods_t *inv_hoods;
      tk_ann_hoods_t *ann_hoods;
      tk_hbi_hoods_t *hbi_hoods;
    } star;
    struct {
      tk_pvec_t *local_pairs;
      tk_inv_t *categories;
      tk_ivec_t *rank_features;
      tk_ivec_t *anchors;
      uint64_t category_anchors;
      uint64_t feature_start;
      uint64_t feature_end;
      tk_rvec_t *knn_heap;
    } anchors;
    struct {
      tk_ivec_t *ids;
      tk_inv_hoods_t *inv_hoods;
      tk_ann_hoods_t *ann_hoods;
      tk_hbi_hoods_t *hbi_hoods;
      uint64_t features;
      tk_ivec_t *offsets;
      tk_ivec_t *neighbors;
      tk_dvec_t *weights;
    } adj_hoods;
  } task;
} tk_graph_thread_t;

static inline tk_graph_t *tk_graph_peek (lua_State *L, int i)
{
  return (tk_graph_t *) luaL_checkudata(L, i, TK_GRAPH_MT);
}

// Macro to compute distance from any index type (inv/ann/hbi)
#define TK_GRAPH_INDEX_DISTANCE(idx_inv, idx_ann, idx_hbi, u, v, cmp, alpha, beta, q_w, e_w, i_w, dist_var) \
  do { \
    tk_inv_t *__idx_inv = (idx_inv); \
    tk_ann_t *__idx_ann = (idx_ann); \
    tk_hbi_t *__idx_hbi = (idx_hbi); \
    (dist_var) = DBL_MAX; \
    if (__idx_inv != NULL) { \
      size_t un, vn; \
      int64_t *uset = tk_inv_get(__idx_inv, (u), &un); \
      int64_t *vset = tk_inv_get(__idx_inv, (v), &vn); \
      if (uset && vset) { \
        double sim = tk_inv_similarity(__idx_inv, uset, un, vset, vn, (cmp), (alpha), (beta), (q_w), (e_w), (i_w)); \
        (dist_var) = 1.0 - sim; \
      } else { \
        (dist_var) = 1.0; \
      } \
    } else if (__idx_ann != NULL) { \
      char *uset = tk_ann_get(__idx_ann, (u)); \
      char *wset = tk_ann_get(__idx_ann, (v)); \
      if (uset && wset) { \
        (dist_var) = (double)tk_cvec_bits_hamming((const uint8_t *)uset, (const uint8_t *)wset, \
                                                   __idx_ann->features) / (double)__idx_ann->features; \
      } \
    } else if (__idx_hbi != NULL) { \
      char *uset = tk_hbi_get(__idx_hbi, (u)); \
      char *wset = tk_hbi_get(__idx_hbi, (v)); \
      if (uset && wset) { \
        (dist_var) = (double)tk_cvec_bits_hamming((const uint8_t *)uset, (const uint8_t *)wset, \
                                                   __idx_hbi->features) / (double)__idx_hbi->features; \
      } \
    } \
  } while(0)

static inline double tk_graph_distance (
  tk_graph_t *graph,
  int64_t u,
  int64_t v,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
) {
  double d = DBL_MAX;

  // Try weight_index first
  TK_GRAPH_INDEX_DISTANCE(graph->weight_inv, graph->weight_ann, graph->weight_hbi,
                          u, v, graph->weight_cmp, graph->weight_alpha, graph->weight_beta,
                          q_weights, e_weights, inter_weights, d);
  if (d != DBL_MAX)
    return d;

  // Fallback to category + knn blending
  bool same_index = (graph->category_inv == graph->knn_inv);
  if (same_index && graph->category_inv != NULL) {
    TK_GRAPH_INDEX_DISTANCE(graph->category_inv, NULL, NULL,
                            u, v, graph->category_cmp, graph->category_alpha, graph->category_beta,
                            q_weights, e_weights, inter_weights, d);
  } else if (graph->category_inv != NULL) {
    double obs_distance = DBL_MAX;
    TK_GRAPH_INDEX_DISTANCE(graph->knn_inv, graph->knn_ann, graph->knn_hbi,
                            u, v, graph->knn_cmp, graph->knn_cmp_alpha, graph->knn_cmp_beta,
                            q_weights, e_weights, inter_weights, obs_distance);
    d = tk_inv_distance_extend(
      graph->category_inv, u, v, obs_distance, graph->category_cmp,
      graph->category_alpha, graph->category_beta,
      q_weights, e_weights, inter_weights);
  } else {
    // Just knn_index
    TK_GRAPH_INDEX_DISTANCE(graph->knn_inv, graph->knn_ann, graph->knn_hbi,
                            u, v, graph->knn_cmp, graph->knn_cmp_alpha, graph->knn_cmp_beta,
                            q_weights, e_weights, inter_weights, d);
  }
  return d;
}

static inline double tk_graph_weight (
  const tk_graph_t *g,
  double base,
  int64_t iu,
  int64_t iv
) {
  const double eps = g->weight_eps;
  double b = base;
  if (isnan(b) || b == DBL_MAX)
    b = 1.0;
  if (b < 0.0)
    b = 0.0;
  else if (b > 1.0)
    b = 1.0;
  double sim;
  if (g->sigmas && g->sigmas->n) {
    double si = (iu >= 0 && (uint64_t) iu < g->sigmas->n) ? g->sigmas->a[iu] : eps;
    double sj = (iv >= 0 && (uint64_t) iv < g->sigmas->n) ? g->sigmas->a[iv] : eps;
    if (si <= 0.0) {
      si = eps;
    }
    if (sj <= 0.0) {
      sj = eps;
    }
    double s = sqrt(si * sj);
    if (s > 0.0) {
      double s2 = s * s;
      sim = exp(-0.5 * (b * b) / s2);
    } else {
      sim = 1.0 - b;
    }
  } else {
    sim = 1.0 - b;
  }
  if (sim < eps) {
    sim = eps;
  }
  if (sim > 1.0) {
    sim = 1.0;
  }
  return sim;
}

static inline void tk_graph_worker (void *dp, int sig)
{
  tk_graph_stage_t stage = (tk_graph_stage_t) sig;
  tk_graph_thread_t *data = (tk_graph_thread_t *) dp;

  switch (stage) {

    case TK_GRAPH_SIGMA: {
      tk_graph_t *graph = data->graph;
      tk_dvec_t *distances = tk_dvec_create(0, 0, 0, 0);
      tk_iuset_t *seen = tk_iuset_create(0, 0);
      for (uint64_t i = data->ifirst; i <= data->ilast; i++) {
        tk_dvec_clear(distances);
        tk_iuset_clear(seen);
        int64_t uid = graph->uids->a[i];
        int64_t neighbor_idx;
        tk_umap_foreach_keys(graph->adj->a[i], neighbor_idx, ({
          int64_t neighbor_uid = graph->uids->a[neighbor_idx];
          double d = tk_graph_distance(graph, uid, neighbor_uid, data->q_weights, data->e_weights, data->inter_weights);
          if (d != DBL_MAX) {
            if (tk_dvec_push(distances, d) != 0) {
              atomic_store(&data->has_error, true);
              return;
            }
            int kha;
            tk_iuset_put(seen, neighbor_idx, &kha);
          }
        }))
        if (graph->uids_idx_hoods) {
          uint32_t khi = tk_iumap_get(graph->uids_idx_hoods, uid);
          if (khi != tk_iumap_end(graph->uids_idx_hoods)) {
            int64_t hood_idx = tk_iumap_val(graph->uids_idx_hoods, khi);
            if (graph->knn_inv_hoods && hood_idx < (int64_t)graph->knn_inv_hoods->n) {
              tk_rvec_t *hood = graph->knn_inv_hoods->a[hood_idx];
              for (uint64_t j = 0; j < hood->m; j++) {
                int64_t neighbor_hood_idx = hood->a[j].i;
                if (neighbor_hood_idx >= 0 && neighbor_hood_idx < (int64_t)graph->uids_hoods->n) {
                  int64_t neighbor_uid = graph->uids_hoods->a[neighbor_hood_idx];
                  uint32_t n_khi = tk_iumap_get(graph->uids_idx, neighbor_uid);
                  if (n_khi != tk_iumap_end(graph->uids_idx)) {
                    int64_t neighbor_global_idx = tk_iumap_val(graph->uids_idx, n_khi);
                    int kha;
                    uint32_t s_khi = tk_iuset_put(seen, neighbor_global_idx, &kha);
                    (void)s_khi;
                    if (kha) {
                      double d = tk_graph_distance(graph, uid, neighbor_uid, data->q_weights, data->e_weights, data->inter_weights);
                      if (d == DBL_MAX)
                        d = hood->a[j].d;
                      if (tk_dvec_push(distances, d) != 0) {
                        atomic_store(&data->has_error, true);
                        return;
                      }
                    }
                  }
                }
              }
            } else if (graph->knn_ann_hoods && hood_idx < (int64_t)graph->knn_ann_hoods->n) {
              tk_pvec_t *hood = graph->knn_ann_hoods->a[hood_idx];
              double denom = graph->knn_ann->features ? (double)graph->knn_ann->features : 1.0;
              for (uint64_t j = 0; j < hood->m; j++) {
                int64_t neighbor_hood_idx = hood->a[j].i;
                if (neighbor_hood_idx >= 0 && neighbor_hood_idx < (int64_t)graph->uids_hoods->n) {
                  int64_t neighbor_uid = graph->uids_hoods->a[neighbor_hood_idx];
                  uint32_t n_khi = tk_iumap_get(graph->uids_idx, neighbor_uid);
                  if (n_khi != tk_iumap_end(graph->uids_idx)) {
                    int64_t neighbor_global_idx = tk_iumap_val(graph->uids_idx, n_khi);
                    int kha;
                    uint32_t s_khi = tk_iuset_put(seen, neighbor_global_idx, &kha);
                    (void)s_khi;
                    if (kha) {
                      double d = tk_graph_distance(graph, uid, neighbor_uid, data->q_weights, data->e_weights, data->inter_weights);
                      if (d == DBL_MAX)
                        d = (double)hood->a[j].p / denom;
                      if (tk_dvec_push(distances, d) != 0) {
                        atomic_store(&data->has_error, true);
                        return;
                      }
                    }
                  }
                }
              }
            } else if (graph->knn_hbi_hoods && hood_idx < (int64_t)graph->knn_hbi_hoods->n) {
              tk_pvec_t *hood = graph->knn_hbi_hoods->a[hood_idx];
              double denom = graph->knn_hbi->features ? (double)graph->knn_hbi->features : 1.0;
              for (uint64_t j = 0; j < hood->m; j++) {
                int64_t neighbor_hood_idx = hood->a[j].i;
                if (neighbor_hood_idx >= 0 && neighbor_hood_idx < (int64_t)graph->uids_hoods->n) {
                  int64_t neighbor_uid = graph->uids_hoods->a[neighbor_hood_idx];
                  uint32_t n_khi = tk_iumap_get(graph->uids_idx, neighbor_uid);
                  if (n_khi != tk_iumap_end(graph->uids_idx)) {
                    int64_t neighbor_global_idx = tk_iumap_val(graph->uids_idx, n_khi);
                    int kha;
                    uint32_t s_khi = tk_iuset_put(seen, neighbor_global_idx, &kha);
                    (void)s_khi;
                    if (kha) {
                      double d = tk_graph_distance(graph, uid, neighbor_uid, data->q_weights, data->e_weights, data->inter_weights);
                      if (d == DBL_MAX)
                        d = (double)hood->a[j].p / denom;
                      if (tk_dvec_push(distances, d) != 0) {
                        atomic_store(&data->has_error, true);
                        return;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        double sigma = 1.0;
        if (distances->n > 0) {
          tk_dvec_asc(distances, 0, distances->n);
          uint64_t k = (graph->sigma_k > 0) ? (uint64_t)graph->sigma_k : distances->n;
          if (k > distances->n)
            k = distances->n;
          sigma = distances->a[k - 1];
        }
        graph->sigmas->a[i] = sigma * graph->sigma_scale;
      }
      tk_dvec_destroy(distances);
      tk_iuset_destroy(seen);
      break;
    }

    case TK_GRAPH_REWEIGHT: {
      tk_graph_t *graph = data->graph;
      const double eps = graph->weight_eps;
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i++) {
        int64_t u = graph->uids->a[i];
        int64_t neighbor_idx;
        tk_umap_foreach_keys(graph->adj->a[i], neighbor_idx, ({
          int64_t v = graph->uids->a[neighbor_idx];
          if (u < v) {
            tk_edge_t edge_key = tk_edge(u, v, 0);
            uint32_t k = tk_euset_get(graph->pairs, edge_key);
            if (k != tk_euset_end(graph->pairs)) {
              double stored_weight = kh_key(graph->pairs, k).w;
              double d = 1.0 - stored_weight;
              if (d < 0.0) d = 0.0;
              if (d > 1.0) d = 1.0;
              double si = (i >= 0 && (uint64_t) i < graph->sigmas->n) ? graph->sigmas->a[i] : eps;
              double sj = (neighbor_idx >= 0 && (uint64_t) neighbor_idx < graph->sigmas->n) ? graph->sigmas->a[neighbor_idx] : eps;
              if (si <= 0.0) si = eps;
              if (sj <= 0.0) sj = eps;
              double s = sqrt(si * sj);
              double new_weight;
              if (s > 0.0) {
                double s2 = s * s;
                new_weight = exp(-0.5 * (d * d) / s2);
              } else {
                new_weight = 1.0 - d;
              }
              if (new_weight < eps) new_weight = eps;
              if (new_weight > 1.0) new_weight = 1.0;
              kh_key(graph->pairs, k).w = new_weight;
            }
          }
        }))
      }
      break;
    }

    case TK_GRAPH_STAR_HOODS: {
      for (uint64_t idx = data->ifirst; idx <= data->ilast; idx++) {
        int64_t id = data->task.star.ids->a[idx];
        #define _tk_graph_process_hood(hood) \
          do { \
            for (uint64_t j = 0; j < (hood)->n; j++) { \
              int64_t idxnbr = (hood)->a[j].i; \
              if (idxnbr >= 0 && idxnbr < (int64_t) data->task.star.ids->n) { \
                int64_t nbr = data->task.star.ids->a[idxnbr]; \
                if (tk_pvec_push(data->task.star.local_pairs, tk_pair(id, nbr)) != 0) { \
                  atomic_store(&data->has_error, true); \
                  return; \
                } \
              } \
            } \
          } while(0)
        if (data->task.star.inv_hoods != NULL)
          _tk_graph_process_hood(data->task.star.inv_hoods->a[idx]);
        else if (data->task.star.ann_hoods != NULL)
          _tk_graph_process_hood(data->task.star.ann_hoods->a[idx]);
        else if (data->task.star.hbi_hoods != NULL)
          _tk_graph_process_hood(data->task.star.hbi_hoods->a[idx]);
        #undef _tk_graph_process_hood
      }
      break;
    }

    case TK_GRAPH_GEN_RANDOM_PAIRS: {
      tk_ivec_t *ids = data->task.random.ids;
      uint64_t edges_per_node = data->task.random.edges_per_node;
      uint64_t n = ids->n;
      for (uint64_t i = data->ifirst; i <= data->ilast && i < n; i++) {
        int64_t id1 = ids->a[i];
        for (uint64_t e = 0; e < edges_per_node; e++) {
          int64_t id2 = -1;
          uint64_t idx2 = tk_fast_random() % n;
          if (idx2 == i)
            idx2 = (idx2 + 1) % n;
          id2 = ids->a[idx2];
          if (id2 >= 0 && id2 != id1) {
            if (tk_pvec_push(data->task.random.local_pairs, tk_pair(id1, id2)) != 0) {
              atomic_store(&data->has_error, true);
              return;
            }
          }
        }
      }
      break;
    }

    case TK_GRAPH_GEN_MULTICLASS_POS: {
      for (uint64_t class_idx = data->ifirst; class_idx <= data->ilast && class_idx < data->task.multiclass.n_classes; class_idx++) {
        tk_ivec_t *class_ids = data->task.multiclass.class_ids[class_idx];
        tk_ivec_t *anchors = data->task.multiclass.anchors[class_idx];
        if (!class_ids || !anchors)
          continue;
        for (uint64_t i = 0; i < class_ids->n; i++) {
          int64_t id = class_ids->a[i];
          for (uint64_t j = 0; j < anchors->n; j++) {
            int64_t anchor = anchors->a[j];
            if (id == anchor)
              continue;
            if (data->task.multiclass.index && data->task.multiclass.eps_pos > 0.0) {
              double dist = tk_inv_distance(
                data->task.multiclass.index, id, anchor,
                data->task.multiclass.cmp,
                data->task.multiclass.cmp_alpha,
                data->task.multiclass.cmp_beta,
                data->q_weights,
                data->e_weights,
                data->inter_weights);
              if (dist > data->task.multiclass.eps_pos)
                continue;
            }
            int64_t a = id < anchor ? id : anchor;
            int64_t b = id < anchor ? anchor : id;
            if (tk_pvec_push(data->task.multiclass.local_pos, tk_pair(a, b)) != 0) {
              atomic_store(&data->has_error, true);
              return;
            }
          }
        }
      }
      break;
    }

    case TK_GRAPH_GEN_MULTICLASS_NEG: {
      for (uint64_t class_idx = data->ifirst; class_idx <= data->ilast && class_idx < data->task.multiclass.n_classes; class_idx++) {
        tk_ivec_t *class_ids = data->task.multiclass.class_ids[class_idx];
        if (!class_ids)
          continue;
        for (uint64_t i = 0; i < class_ids->n; i++) {
          int64_t id = class_ids->a[i];
          for (uint64_t n = 0; n < data->task.multiclass.n_anchors_neg; n++) {
            uint64_t tries = 0;
            uint64_t max_tries = 100;
            int64_t other_id = -1;
            while (tries < max_tries) {
              uint64_t other_class = tk_fast_random() % data->task.multiclass.n_classes;
              if (other_class == class_idx) {
                tries++;
                continue;
              }
              tk_ivec_t *other_class_ids = data->task.multiclass.class_ids[other_class];
              if (!other_class_ids || other_class_ids->n == 0) {
                tries++;
                continue;
              }
              uint64_t other_idx = tk_fast_random() % other_class_ids->n;
              other_id = other_class_ids->a[other_idx];
              if (data->task.multiclass.index && data->task.multiclass.eps_neg > 0.0) {
                double dist = tk_inv_distance(
                  data->task.multiclass.index, id, other_id,
                  data->task.multiclass.cmp,
                  data->task.multiclass.cmp_alpha,
                  data->task.multiclass.cmp_beta,
                  data->q_weights,
                  data->e_weights,
                  data->inter_weights);
                if (dist < data->task.multiclass.eps_neg) {
                  tries++;
                  continue;
                }
              }
              break;
            }
            if (other_id >= 0 && tries < max_tries) {
              int64_t a = id < other_id ? id : other_id;
              int64_t b = id < other_id ? other_id : id;
              if (tk_pvec_push(data->task.multiclass.local_neg, tk_pair(a, b)) != 0) {
                atomic_store(&data->has_error, true);
                return;
              }
            }
          }
        }
      }
      break;
    }

    case TK_GRAPH_ANCHORS: {
      tk_inv_t *inv = data->task.anchors.categories;
      uint64_t category_anchors = data->task.anchors.category_anchors;
      uint64_t feature_start = data->task.anchors.feature_start;
      uint64_t feature_end = data->task.anchors.feature_end;
      int64_t *rank_features = data->task.anchors.rank_features ? data->task.anchors.rank_features->a : NULL;
      int64_t max_rank = (int64_t) inv->n_ranks - 1;
      tk_rvec_t *knn_heap = data->task.anchors.knn_heap;
      tk_ivec_t *anchors = data->task.anchors.anchors;
      tk_graph_t *graph = data->graph;
      double category_knn_decay = graph->category_knn_decay;
      double category_knn = graph->category_knn;
      double category_cmp = graph->category_cmp;
      double category_alpha = graph->category_alpha;
      double category_beta = graph->category_beta;
      tk_pvec_t *local_pairs = data->task.anchors.local_pairs;
      for (uint64_t fi = feature_start; fi <= feature_end; fi++) {
        int64_t f = rank_features == NULL ? (int64_t) fi : rank_features[fi];
        tk_ivec_t *postings = inv->postings->a[f];
        if (!postings || postings->n == 0)
          continue;
        int64_t rank = inv->ranks ? inv->ranks->a[f] : 0;
        int64_t effective_rank = (category_knn_decay < 0.0) ? (max_rank - rank) : rank;
        double knn_weight = exp(-(double)effective_rank * fabs(category_knn_decay));
        uint64_t n_anchors = (uint64_t) category_anchors;
        uint64_t n_wanted = category_knn > 0 ? category_knn : n_anchors;
        uint64_t k_nearest = (uint64_t) ceil((double) n_wanted * knn_weight);
        tk_ivec_clear(anchors);
        for (uint64_t i = 0; i < postings->n; i++) {
          int64_t sid = postings->a[i];
          uint32_t khi = tk_iumap_get(inv->sid_uid, sid);
          if (khi != tk_iumap_end(inv->sid_uid)) {
            int64_t uid = tk_iumap_val(inv->sid_uid, khi);
            if (tk_ivec_push(anchors, uid) != 0) {
              atomic_store(&data->has_error, true);
              return;
            }
          }
        }
        if (anchors->n == 0)
          continue;
        tk_ivec_shuffle(anchors);
        for (uint64_t i = 0; i < anchors->n; i++) {
          int64_t u = anchors->a[i];
          tk_rvec_clear(knn_heap);
          for (uint64_t j = 0; j < anchors->n && j < n_anchors; j ++) {
            if (i == j) continue;
            int64_t v = anchors->a[j];
            double dist = tk_inv_distance(inv, u, v, category_cmp, category_alpha, category_beta, data->q_weights, data->e_weights, data->inter_weights);
            tk_rvec_hmin(knn_heap, k_nearest, tk_rank(v, dist));
          }
          for (uint64_t k = 0; k < knn_heap->n; k++) {
            int64_t v = knn_heap->a[k].i;
            if (tk_pvec_push(local_pairs, tk_pair(u, v)) != 0) {
              atomic_store(&data->has_error, true);
              return;
            }
          }
        }
      }
      break;
    }

    case TK_GRAPH_COMPUTE_WEIGHTS: {
      tk_graph_t *graph = data->graph;
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i++) {
        int64_t u = graph->uids->a[i];
        int64_t neighbor_idx;
        tk_umap_foreach_keys(graph->adj->a[i], neighbor_idx, ({
          int64_t v = graph->uids->a[neighbor_idx];
          if (u < v) {
            tk_edge_t edge_key = tk_edge(u, v, 0);
            uint32_t k = tk_euset_get(graph->pairs, edge_key);
            if (k != tk_euset_end(graph->pairs)) {
              double d = tk_graph_distance(graph, u, v, data->q_weights, data->e_weights, data->inter_weights);
              double w = tk_graph_weight(graph, d, i, neighbor_idx);
              kh_key(graph->pairs, k).w = w;
            }
          }
        }))
      }
      break;
    }

    case TK_GRAPH_ADJ_HOODS: {
      for (uint64_t i = data->ifirst; i <= data->ilast && i < data->task.adj_hoods.ids->n; i++) {
        int64_t start = data->task.adj_hoods.offsets->a[i];
        int64_t end = data->task.adj_hoods.offsets->a[i + 1];
        int64_t idx = start;

        if (data->task.adj_hoods.ann_hoods && i < data->task.adj_hoods.ann_hoods->n) {
          tk_pvec_t *hood = data->task.adj_hoods.ann_hoods->a[i];
          for (uint64_t j = 0; j < hood->n && idx < end; j++) {
            data->task.adj_hoods.neighbors->a[idx] = hood->a[j].i;
            double distance = (double)hood->a[j].p / data->task.adj_hoods.features;
            data->task.adj_hoods.weights->a[idx] = 1.0 - distance;
            idx++;
          }
        } else if (data->task.adj_hoods.hbi_hoods && i < data->task.adj_hoods.hbi_hoods->n) {
          tk_pvec_t *hood = data->task.adj_hoods.hbi_hoods->a[i];
          for (uint64_t j = 0; j < hood->n && idx < end; j++) {
            data->task.adj_hoods.neighbors->a[idx] = hood->a[j].i;
            double distance = (double)hood->a[j].p / data->task.adj_hoods.features;
            data->task.adj_hoods.weights->a[idx] = 1.0 - distance;
            idx++;
          }
        } else if (data->task.adj_hoods.inv_hoods && i < data->task.adj_hoods.inv_hoods->n) {
          tk_rvec_t *hood = data->task.adj_hoods.inv_hoods->a[i];
          for (uint64_t j = 0; j < hood->n && idx < end; j++) {
            data->task.adj_hoods.neighbors->a[idx] = hood->a[j].i;
            data->task.adj_hoods.weights->a[idx] = 1.0 - hood->a[j].d;
            idx++;
          }
        }
      }
      break;
    }

  }
}

static inline double tk_graph_get_weight (
  tk_graph_t *graph,
  int64_t u,
  int64_t v
) {
  khint_t khi;
  tk_edge_t p = tk_edge(u, v, 0);
  khi = tk_euset_get(graph->pairs, p);
  if (khi == tk_euset_end(graph->pairs))
    return 0.0;
  return tk_euset_key(graph->pairs, khi).w;
}

#define tk_edge_gtu(a, b) ((a).u > (b).u)
KSORT_INIT(tk_evec_asc_u, tk_edge_t, tk_edge_gtu)

static inline int tk_graph_pairs_to_csr (
  tk_ivec_t *ids,
  tk_pvec_t *pos,
  tk_pvec_t *neg,
  tk_ivec_t **out_offsets,
  tk_ivec_t **out_neighbors,
  tk_dvec_t **out_weights
) {
  if (ids->n == 0) {
    *out_offsets = tk_ivec_create(NULL, 1, 0, 0);
    *out_neighbors = tk_ivec_create(NULL, 0, 0, 0);
    *out_weights = tk_dvec_create(NULL, 0, 0, 0);
    if (!*out_offsets || !*out_neighbors || !*out_weights) {
      if (*out_offsets) tk_ivec_destroy(*out_offsets);
      if (*out_neighbors) tk_ivec_destroy(*out_neighbors);
      if (*out_weights) tk_dvec_destroy(*out_weights);
      *out_offsets = NULL;
      *out_neighbors = NULL;
      *out_weights = NULL;
      return -1;
    }
    (*out_offsets)->a[0] = 0;
    (*out_offsets)->n = 1;
    return 0;
  }

  tk_iumap_t *ididx = tk_iumap_from_ivec(NULL, ids);
  if (!ididx)
    return -1;

  uint64_t n_nodes = ids->n;
  tk_rvec_t **hoods = calloc(n_nodes, sizeof(tk_rvec_t *));
  if (!hoods) {
    tk_iumap_destroy(ididx);
    return -1;
  }

  for (uint64_t i = 0; i < n_nodes; i++)
    hoods[i] = NULL;

  // Process positive pairs (weight 1.0)
  for (uint64_t i = 0; i < pos->n; i++) {
    int64_t a = pos->a[i].i;
    int64_t b = pos->a[i].p;
    khint_t khi_a = tk_iumap_get(ididx, a);
    khint_t khi_b = tk_iumap_get(ididx, b);
    if (khi_a == tk_iumap_end(ididx) || khi_b == tk_iumap_end(ididx))
      continue;
    int64_t aidx = tk_iumap_val(ididx, khi_a);
    int64_t bidx = tk_iumap_val(ididx, khi_b);
    if (aidx < 0 || aidx >= (int64_t)n_nodes || bidx < 0 || bidx >= (int64_t)n_nodes)
      continue;

    if (!hoods[aidx]) {
      hoods[aidx] = tk_rvec_create(NULL, 0, 0, 0);
      if (!hoods[aidx]) goto cleanup;
    }
    if (!hoods[bidx]) {
      hoods[bidx] = tk_rvec_create(NULL, 0, 0, 0);
      if (!hoods[bidx]) goto cleanup;
    }

    if (tk_rvec_push(hoods[aidx], tk_rank(bidx, 1.0)) != 0)
      goto cleanup;
    if (tk_rvec_push(hoods[bidx], tk_rank(aidx, 1.0)) != 0)
      goto cleanup;
  }

  // Process negative pairs (weight 0.0)
  for (uint64_t i = 0; i < neg->n; i++) {
    int64_t a = neg->a[i].i;
    int64_t b = neg->a[i].p;
    khint_t khi_a = tk_iumap_get(ididx, a);
    khint_t khi_b = tk_iumap_get(ididx, b);
    if (khi_a == tk_iumap_end(ididx) || khi_b == tk_iumap_end(ididx))
      continue;
    int64_t aidx = tk_iumap_val(ididx, khi_a);
    int64_t bidx = tk_iumap_val(ididx, khi_b);
    if (aidx < 0 || aidx >= (int64_t)n_nodes || bidx < 0 || bidx >= (int64_t)n_nodes)
      continue;

    if (!hoods[aidx]) {
      hoods[aidx] = tk_rvec_create(NULL, 0, 0, 0);
      if (!hoods[aidx]) goto cleanup;
    }
    if (!hoods[bidx]) {
      hoods[bidx] = tk_rvec_create(NULL, 0, 0, 0);
      if (!hoods[bidx]) goto cleanup;
    }

    if (tk_rvec_push(hoods[aidx], tk_rank(bidx, 0.0)) != 0)
      goto cleanup;
    if (tk_rvec_push(hoods[bidx], tk_rank(aidx, 0.0)) != 0)
      goto cleanup;
  }

  // Build CSR arrays
  *out_offsets = tk_ivec_create(NULL, n_nodes + 1, 0, 0);
  *out_neighbors = tk_ivec_create(NULL, 0, 0, 0);
  *out_weights = tk_dvec_create(NULL, 0, 0, 0);
  if (!*out_offsets || !*out_neighbors || !*out_weights)
    goto cleanup;

  (*out_offsets)->a[0] = 0;
  for (uint64_t idx = 0; idx < n_nodes; idx++) {
    tk_rvec_t *hood = hoods[idx];
    if (hood) {
      tk_rvec_desc(hood, 0, hood->n);
      for (uint64_t j = 0; j < hood->n; j++) {
        if (tk_ivec_push(*out_neighbors, hood->a[j].i) != 0)
          goto cleanup;
        if (tk_dvec_push(*out_weights, hood->a[j].d) != 0)
          goto cleanup;
      }
    }
    (*out_offsets)->a[idx + 1] = (int64_t) (*out_neighbors)->n;
  }
  (*out_offsets)->n = n_nodes + 1;

  for (uint64_t i = 0; i < n_nodes; i++)
    if (hoods[i]) tk_rvec_destroy(hoods[i]);
  free(hoods);
  tk_iumap_destroy(ididx);
  return 0;

cleanup:
  for (uint64_t i = 0; i < n_nodes; i++)
    if (hoods[i]) tk_rvec_destroy(hoods[i]);
  free(hoods);
  tk_iumap_destroy(ididx);
  if (*out_offsets) tk_ivec_destroy(*out_offsets);
  if (*out_neighbors) tk_ivec_destroy(*out_neighbors);
  if (*out_weights) tk_dvec_destroy(*out_weights);
  *out_offsets = NULL;
  *out_neighbors = NULL;
  *out_weights = NULL;
  return -1;
}

static inline int tk_graph_star_hoods (
  lua_State *L,
  tk_ivec_t *ids,
  tk_inv_hoods_t *inv_hoods,
  tk_ann_hoods_t *ann_hoods,
  tk_hbi_hoods_t *hbi_hoods,
  unsigned int n_threads,
  tk_pvec_t **out_pairs
) {
  if (ids->n == 0) {
    *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
    return *out_pairs ? 0 : -1;
  }

  tk_graph_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
  if (!threads)
    return -1;
  memset(threads, 0, n_threads * sizeof(tk_graph_thread_t));

  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_graph_worker);
  if (!pool) {
    free(threads);
    return -1;
  }

  for (unsigned int i = 0; i < n_threads; i++) {
    threads[i].task.star.local_pairs = tk_pvec_create(NULL, 0, 0, 0);
    if (!threads[i].task.star.local_pairs) {
      for (unsigned int j = 0; j < i; j++)
        tk_pvec_destroy(threads[j].task.star.local_pairs);
      tk_threads_destroy(pool);
      free(threads);
      return -1;
    }
    threads[i].task.star.ids = ids;
    threads[i].task.star.inv_hoods = inv_hoods;
    threads[i].task.star.ann_hoods = ann_hoods;
    threads[i].task.star.hbi_hoods = hbi_hoods;
    pool->threads[i].data = &threads[i];
    tk_thread_range(i, n_threads, ids->n, &threads[i].ifirst, &threads[i].ilast);
    atomic_init(&threads[i].has_error, false);
  }

  tk_threads_signal(pool, TK_GRAPH_STAR_HOODS, 0);

  // Check for errors
  for (unsigned int i = 0; i < n_threads; i++) {
    if (atomic_load(&threads[i].has_error)) {
      for (unsigned int j = 0; j < n_threads; j++)
        tk_pvec_destroy(threads[j].task.star.local_pairs);
      tk_threads_destroy(pool);
      free(threads);
      return -1;
    }
  }

  // Merge results
  *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
  if (!*out_pairs) {
    for (unsigned int i = 0; i < n_threads; i++)
      tk_pvec_destroy(threads[i].task.star.local_pairs);
    tk_threads_destroy(pool);
    free(threads);
    return -1;
  }

  for (unsigned int i = 0; i < n_threads; i++) {
    for (uint64_t j = 0; j < threads[i].task.star.local_pairs->n; j++) {
      if (tk_pvec_push(*out_pairs, threads[i].task.star.local_pairs->a[j]) != 0) {
        tk_pvec_destroy(*out_pairs);
        for (unsigned int k = 0; k < n_threads; k++)
          tk_pvec_destroy(threads[k].task.star.local_pairs);
        tk_threads_destroy(pool);
        free(threads);
        return -1;
      }
    }
    tk_pvec_destroy(threads[i].task.star.local_pairs);
  }

  tk_threads_destroy(pool);
  free(threads);
  return 0;
}

static inline int tk_graph_anchor_pairs (
  tk_ivec_t *ids,
  tk_ivec_t *labels,
  uint64_t n_anchors,
  tk_pvec_t **out_pairs
) {
  if (ids->n == 0 || n_anchors == 0) {
    *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
    return *out_pairs ? 0 : -1;
  }

  *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
  if (!*out_pairs)
    return -1;

  if (labels && labels->n == ids->n) {
    // Per-label anchor generation
    tk_iumap_t **ids_by_label = NULL;
    int64_t max_label = -1;

    // Find max label
    for (uint64_t i = 0; i < labels->n; i++) {
      int64_t lbl = labels->a[i];
      if (lbl > max_label)
        max_label = lbl;
    }

    if (max_label < 0)
      return 0;

    ids_by_label = calloc((uint64_t) max_label + 1, sizeof(tk_iumap_t *));
    if (!ids_by_label) {
      tk_pvec_destroy(*out_pairs);
      return -1;
    }

    // Group IDs by label
    for (uint64_t i = 0; i < ids->n; i++) {
      int64_t lbl = labels->a[i];
      if (lbl == -1)
        continue;
      if (!ids_by_label[lbl]) {
        ids_by_label[lbl] = tk_iumap_create(NULL, 0);
        if (!ids_by_label[lbl]) goto cleanup_label;
      }
      int kha;
      tk_iumap_put(ids_by_label[lbl], ids->a[i], &kha);
    }

    // For each label, select anchors and create edges
    for (int64_t lbl = 0; lbl <= max_label; lbl++) {
      if (!ids_by_label[lbl])
        continue;
      uint64_t n_in_class = tk_iumap_size(ids_by_label[lbl]);
      if (n_in_class == 0)
        continue;

      uint64_t n_label_anchors = n_anchors < n_in_class ? n_anchors : n_in_class;
      tk_iuset_t *selected_anchors = tk_iuset_create(NULL, 0);
      if (!selected_anchors) goto cleanup_label;

      // Select n_label_anchors random anchors
      while (tk_iuset_size(selected_anchors) < n_label_anchors) {
        uint32_t rand_k = tk_iumap_begin(ids_by_label[lbl]);
        uint64_t skip = tk_fast_random() % tk_iumap_size(ids_by_label[lbl]);
        for (uint64_t s = 0; s < skip && rand_k != tk_iumap_end(ids_by_label[lbl]); s++)
          rand_k++;
        if (rand_k != tk_iumap_end(ids_by_label[lbl])) {
          int64_t anchor = tk_iumap_key(ids_by_label[lbl], rand_k);
          int kha;
          tk_iuset_put(selected_anchors, anchor, &kha);
        }
      }

      // Create edges from all IDs in class to anchors
      for (uint32_t k = tk_iumap_begin(ids_by_label[lbl]); k != tk_iumap_end(ids_by_label[lbl]); k++) {
        if (!tk_iumap_exist(ids_by_label[lbl], k))
          continue;
        int64_t id = tk_iumap_key(ids_by_label[lbl], k);
        for (uint32_t ak = tk_iuset_begin(selected_anchors); ak != tk_iuset_end(selected_anchors); ak++) {
          if (!tk_iuset_exist(selected_anchors, ak))
            continue;
          int64_t anchor = tk_iuset_key(selected_anchors, ak);
          if (id != anchor) {
            if (tk_pvec_push(*out_pairs, tk_pair(id, anchor)) != 0) {
              tk_iuset_destroy(selected_anchors);
              goto cleanup_label;
            }
          }
        }
      }
      tk_iuset_destroy(selected_anchors);
    }

cleanup_label:
    for (int64_t lbl = 0; lbl <= max_label; lbl++)
      if (ids_by_label[lbl])
        tk_iumap_destroy(ids_by_label[lbl]);
    free(ids_by_label);

  } else {
    // Global anchor generation
    tk_iuset_t *selected_anchors = tk_iuset_create(NULL, 0);
    if (!selected_anchors) {
      tk_pvec_destroy(*out_pairs);
      return -1;
    }

    uint64_t n_actual = n_anchors < ids->n ? n_anchors : ids->n;
    while (tk_iuset_size(selected_anchors) < n_actual) {
      uint64_t rand_idx = tk_fast_random() % ids->n;
      int64_t anchor = ids->a[rand_idx];
      int kha;
      tk_iuset_put(selected_anchors, anchor, &kha);
    }

    // Create edges from all IDs to anchors
    for (uint64_t i = 0; i < ids->n; i++) {
      int64_t id = ids->a[i];
      for (uint32_t k = tk_iuset_begin(selected_anchors); k != tk_iuset_end(selected_anchors); k++) {
        if (!tk_iuset_exist(selected_anchors, k))
          continue;
        int64_t anchor = tk_iuset_key(selected_anchors, k);
        if (id != anchor) {
          if (tk_pvec_push(*out_pairs, tk_pair(id, anchor)) != 0) {
            tk_iuset_destroy(selected_anchors);
            tk_pvec_destroy(*out_pairs);
            return -1;
          }
        }
      }
    }
    tk_iuset_destroy(selected_anchors);
  }

  return 0;
}

static inline int tk_graph_random_pairs (
  lua_State *L,
  tk_ivec_t *ids,
  tk_ivec_t *labels,
  uint64_t edges_per_node,
  unsigned int n_threads,
  tk_pvec_t **out_pairs
) {
  *out_pairs = NULL;

  if (ids->n == 0) {
    *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
    return *out_pairs ? 0 : -1;
  }

  // Build ids_by_label if labels provided
  tk_ivec_t **ids_by_label = NULL;
  uint64_t n_labels = 0;

  if (labels && labels->n == ids->n) {
    int64_t max_label = -1;
    for (uint64_t i = 0; i < labels->n; i++)
      if (labels->a[i] > max_label)
        max_label = labels->a[i];

    if (max_label >= 0) {
      n_labels = (uint64_t) max_label + 1;
      ids_by_label = calloc(n_labels, sizeof(tk_ivec_t *));
      if (!ids_by_label)
        return -1;

      for (uint64_t i = 0; i < ids->n; i++) {
        int64_t lbl = labels->a[i];
        if (lbl == -1)
          continue;
        if (!ids_by_label[lbl]) {
          ids_by_label[lbl] = tk_ivec_create(NULL, 0, 0, 0);
          if (!ids_by_label[lbl]) goto cleanup_random;
        }
        if (tk_ivec_push(ids_by_label[lbl], ids->a[i]) != 0)
          goto cleanup_random;
      }
    }
  }

  tk_graph_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
  if (!threads) goto cleanup_random;
  memset(threads, 0, n_threads * sizeof(tk_graph_thread_t));

  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_graph_worker);
  if (!pool) {
    free(threads);
    goto cleanup_random;
  }

  for (unsigned int i = 0; i < n_threads; i++) {
    threads[i].task.random.local_pairs = tk_pvec_create(NULL, 0, 0, 0);
    if (!threads[i].task.random.local_pairs) {
      for (unsigned int j = 0; j < i; j++)
        tk_pvec_destroy(threads[j].task.random.local_pairs);
      tk_threads_destroy(pool);
      free(threads);
      goto cleanup_random;
    }
    threads[i].task.random.ids = ids;
    threads[i].task.random.labels = labels;
    threads[i].task.random.edges_per_node = edges_per_node;
    threads[i].task.random.ids_by_label = ids_by_label;
    threads[i].task.random.n_labels = n_labels;
    pool->threads[i].data = &threads[i];
    tk_thread_range(i, n_threads, ids->n, &threads[i].ifirst, &threads[i].ilast);
    atomic_init(&threads[i].has_error, false);
  }

  tk_threads_signal(pool, TK_GRAPH_GEN_RANDOM_PAIRS, 0);

  // Check for errors
  for (unsigned int i = 0; i < n_threads; i++) {
    if (atomic_load(&threads[i].has_error)) {
      for (unsigned int j = 0; j < n_threads; j++)
        tk_pvec_destroy(threads[j].task.random.local_pairs);
      tk_threads_destroy(pool);
      free(threads);
      goto cleanup_random;
    }
  }

  // Merge and deduplicate
  *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
  if (!*out_pairs) {
    for (unsigned int i = 0; i < n_threads; i++)
      tk_pvec_destroy(threads[i].task.random.local_pairs);
    tk_threads_destroy(pool);
    free(threads);
    goto cleanup_random;
  }

  for (unsigned int i = 0; i < n_threads; i++) {
    for (uint64_t j = 0; j < threads[i].task.random.local_pairs->n; j++) {
      if (tk_pvec_push(*out_pairs, threads[i].task.random.local_pairs->a[j]) != 0) {
        tk_pvec_destroy(*out_pairs);
        for (unsigned int k = 0; k < n_threads; k++)
          tk_pvec_destroy(threads[k].task.random.local_pairs);
        tk_threads_destroy(pool);
        free(threads);
        goto cleanup_random;
      }
    }
    tk_pvec_destroy(threads[i].task.random.local_pairs);
  }

  tk_threads_destroy(pool);
  free(threads);

  // Sort and unique
  tk_pvec_xasc(*out_pairs, 0, (*out_pairs)->n);

  // Cleanup
  if (ids_by_label) {
    for (uint64_t i = 0; i < n_labels; i++)
      if (ids_by_label[i])
        tk_ivec_destroy(ids_by_label[i]);
    free(ids_by_label);
  }
  return 0;

cleanup_random:
  if (ids_by_label) {
    for (uint64_t i = 0; i < n_labels; i++)
      if (ids_by_label[i])
        tk_ivec_destroy(ids_by_label[i]);
    free(ids_by_label);
  }
  if (*out_pairs)
    tk_pvec_destroy(*out_pairs);
  *out_pairs = NULL;
  return -1;
}

static inline int tk_graph_multiclass_pairs (
  lua_State *L,
  tk_ivec_t *ids,
  tk_ivec_t *labels,
  uint64_t n_anchors_pos,
  uint64_t n_anchors_neg,
  tk_inv_t *index,
  double eps_pos,
  double eps_neg,
  unsigned int n_threads,
  tk_pvec_t **out_pos,
  tk_pvec_t **out_neg
) {
  *out_pos = NULL;
  *out_neg = NULL;

  if (ids->n == 0 || !labels || labels->n != ids->n) {
    *out_pos = tk_pvec_create(NULL, 0, 0, 0);
    *out_neg = tk_pvec_create(NULL, 0, 0, 0);
    if (!*out_pos || !*out_neg) {
      if (*out_pos) tk_pvec_destroy(*out_pos);
      if (*out_neg) tk_pvec_destroy(*out_neg);
      *out_pos = NULL;
      *out_neg = NULL;
      return -1;
    }
    return 0;
  }

  // Find classes and build class_ids
  int64_t max_label = -1;
  for (uint64_t i = 0; i < labels->n; i++)
    if (labels->a[i] > max_label)
      max_label = labels->a[i];

  if (max_label < 0) {
    *out_pos = tk_pvec_create(NULL, 0, 0, 0);
    *out_neg = tk_pvec_create(NULL, 0, 0, 0);
    return (*out_pos && *out_neg) ? 0 : -1;
  }

  uint64_t n_classes = (uint64_t) max_label + 1;
  tk_ivec_t **class_ids = calloc(n_classes, sizeof(tk_ivec_t *));
  tk_ivec_t **anchors = calloc(n_classes, sizeof(tk_ivec_t *));
  if (!class_ids || !anchors) {
    if (class_ids) free(class_ids);
    if (anchors) free(anchors);
    return -1;
  }

  // Build class_ids
  for (uint64_t i = 0; i < ids->n; i++) {
    int64_t lbl = labels->a[i];
    if (lbl == -1)
      continue;
    if (!class_ids[lbl]) {
      class_ids[lbl] = tk_ivec_create(NULL, 0, 0, 0);
      if (!class_ids[lbl]) goto cleanup_multiclass;
    }
    if (tk_ivec_push(class_ids[lbl], ids->a[i]) != 0)
      goto cleanup_multiclass;
  }

  // Select anchors per class
  if (n_anchors_pos > 0) {
    for (uint64_t c = 0; c < n_classes; c++) {
      if (!class_ids[c])
        continue;
      anchors[c] = tk_ivec_create(NULL, 0, 0, 0);
      if (!anchors[c]) goto cleanup_multiclass;

      uint64_t k = n_anchors_pos < class_ids[c]->n ? n_anchors_pos : class_ids[c]->n;
      tk_ivec_t *shuffle = tk_ivec_create(NULL, class_ids[c]->n, 0, 0);
      if (!shuffle) goto cleanup_multiclass;

      tk_ivec_fill_indices(shuffle);
      tk_ivec_shuffle(shuffle);

      for (uint64_t i = 0; i < k; i++) {
        int64_t anchor = class_ids[c]->a[shuffle->a[i]];
        if (tk_ivec_push(anchors[c], anchor) != 0) {
          tk_ivec_destroy(shuffle);
          goto cleanup_multiclass;
        }
      }
      tk_ivec_destroy(shuffle);
    }
  }

  // Generate positive pairs (threaded)
  *out_pos = tk_pvec_create(NULL, 0, 0, 0);
  if (!*out_pos) goto cleanup_multiclass;

  if (n_anchors_pos > 0) {
    tk_graph_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
    if (!threads) goto cleanup_multiclass;
    memset(threads, 0, n_threads * sizeof(tk_graph_thread_t));

    tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_graph_worker);
    if (!pool) {
      free(threads);
      goto cleanup_multiclass;
    }

    bool need_buffers = index && index->ranks && index->n_ranks > 1;

    for (unsigned int i = 0; i < n_threads; i++) {
      threads[i].task.multiclass.local_pos = tk_pvec_create(NULL, 0, 0, 0);
      if (!threads[i].task.multiclass.local_pos) {
        for (unsigned int j = 0; j < i; j++)
          tk_pvec_destroy(threads[j].task.multiclass.local_pos);
        tk_threads_destroy(pool);
        free(threads);
        goto cleanup_multiclass;
      }

      // Per-thread buffers for inv distance computation
      threads[i].q_weights = NULL;
      threads[i].e_weights = NULL;
      threads[i].inter_weights = NULL;
      if (need_buffers) {
        threads[i].q_weights = tk_dvec_create(NULL, 0, 0, 0);
        threads[i].e_weights = tk_dvec_create(NULL, 0, 0, 0);
        threads[i].inter_weights = tk_dvec_create(NULL, 0, 0, 0);
      }

      threads[i].task.multiclass.class_ids = class_ids;
      threads[i].task.multiclass.anchors = anchors;
      threads[i].task.multiclass.n_classes = n_classes;
      threads[i].task.multiclass.index = index;
      threads[i].task.multiclass.eps_pos = eps_pos;
      threads[i].task.multiclass.cmp = TK_IVEC_JACCARD;
      threads[i].task.multiclass.cmp_alpha = 0.0;
      threads[i].task.multiclass.cmp_beta = 0.0;
      pool->threads[i].data = &threads[i];
      tk_thread_range(i, n_threads, n_classes, &threads[i].ifirst, &threads[i].ilast);
      atomic_init(&threads[i].has_error, false);
    }

    tk_threads_signal(pool, TK_GRAPH_GEN_MULTICLASS_POS, 0);

    for (unsigned int i = 0; i < n_threads; i++) {
      if (atomic_load(&threads[i].has_error)) {
        for (unsigned int j = 0; j < n_threads; j++)
          tk_pvec_destroy(threads[j].task.multiclass.local_pos);
        tk_threads_destroy(pool);
        free(threads);
        goto cleanup_multiclass;
      }
    }

    for (unsigned int i = 0; i < n_threads; i++) {
      for (uint64_t j = 0; j < threads[i].task.multiclass.local_pos->n; j++) {
        if (tk_pvec_push(*out_pos, threads[i].task.multiclass.local_pos->a[j]) != 0) {
          for (unsigned int k = i; k < n_threads; k++)
            tk_pvec_destroy(threads[k].task.multiclass.local_pos);
          tk_threads_destroy(pool);
          free(threads);
          goto cleanup_multiclass;
        }
      }
      tk_pvec_destroy(threads[i].task.multiclass.local_pos);
      if (threads[i].q_weights) tk_dvec_destroy(threads[i].q_weights);
      if (threads[i].e_weights) tk_dvec_destroy(threads[i].e_weights);
      if (threads[i].inter_weights) tk_dvec_destroy(threads[i].inter_weights);
    }

    tk_threads_destroy(pool);
    free(threads);
  }

  // Generate negative pairs (threaded)
  *out_neg = tk_pvec_create(NULL, 0, 0, 0);
  if (!*out_neg) goto cleanup_multiclass;

  if (n_anchors_neg > 0) {
    tk_graph_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
    if (!threads) goto cleanup_multiclass;
    memset(threads, 0, n_threads * sizeof(tk_graph_thread_t));

    tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_graph_worker);
    if (!pool) {
      free(threads);
      goto cleanup_multiclass;
    }

    bool need_buffers_neg = index && index->ranks && index->n_ranks > 1;

    for (unsigned int i = 0; i < n_threads; i++) {
      threads[i].task.multiclass.local_neg = tk_pvec_create(NULL, 0, 0, 0);
      if (!threads[i].task.multiclass.local_neg) {
        for (unsigned int j = 0; j < i; j++)
          tk_pvec_destroy(threads[j].task.multiclass.local_neg);
        tk_threads_destroy(pool);
        free(threads);
        goto cleanup_multiclass;
      }

      // Per-thread buffers for inv distance computation
      threads[i].q_weights = NULL;
      threads[i].e_weights = NULL;
      threads[i].inter_weights = NULL;
      if (need_buffers_neg) {
        threads[i].q_weights = tk_dvec_create(NULL, 0, 0, 0);
        threads[i].e_weights = tk_dvec_create(NULL, 0, 0, 0);
        threads[i].inter_weights = tk_dvec_create(NULL, 0, 0, 0);
      }

      threads[i].task.multiclass.class_ids = class_ids;
      threads[i].task.multiclass.n_classes = n_classes;
      threads[i].task.multiclass.index = index;
      threads[i].task.multiclass.eps_neg = eps_neg;
      threads[i].task.multiclass.n_anchors_neg = n_anchors_neg;
      threads[i].task.multiclass.cmp = TK_IVEC_JACCARD;
      threads[i].task.multiclass.cmp_alpha = 0.0;
      threads[i].task.multiclass.cmp_beta = 0.0;
      pool->threads[i].data = &threads[i];
      tk_thread_range(i, n_threads, n_classes, &threads[i].ifirst, &threads[i].ilast);
      atomic_init(&threads[i].has_error, false);
    }

    tk_threads_signal(pool, TK_GRAPH_GEN_MULTICLASS_NEG, 0);

    for (unsigned int i = 0; i < n_threads; i++) {
      if (atomic_load(&threads[i].has_error)) {
        for (unsigned int j = 0; j < n_threads; j++)
          tk_pvec_destroy(threads[j].task.multiclass.local_neg);
        tk_threads_destroy(pool);
        free(threads);
        goto cleanup_multiclass;
      }
    }

    for (unsigned int i = 0; i < n_threads; i++) {
      for (uint64_t j = 0; j < threads[i].task.multiclass.local_neg->n; j++) {
        if (tk_pvec_push(*out_neg, threads[i].task.multiclass.local_neg->a[j]) != 0) {
          for (unsigned int k = i; k < n_threads; k++)
            tk_pvec_destroy(threads[k].task.multiclass.local_neg);
          tk_threads_destroy(pool);
          free(threads);
          goto cleanup_multiclass;
        }
      }
      tk_pvec_destroy(threads[i].task.multiclass.local_neg);
      if (threads[i].q_weights) tk_dvec_destroy(threads[i].q_weights);
      if (threads[i].e_weights) tk_dvec_destroy(threads[i].e_weights);
      if (threads[i].inter_weights) tk_dvec_destroy(threads[i].inter_weights);
    }

    tk_threads_destroy(pool);
    free(threads);
  }

  // Sort outputs
  tk_pvec_xasc(*out_pos, 0, (*out_pos)->n);
  tk_pvec_xasc(*out_neg, 0, (*out_neg)->n);

  // Cleanup
  for (uint64_t i = 0; i < n_classes; i++) {
    if (class_ids[i])
      tk_ivec_destroy(class_ids[i]);
    if (anchors[i])
      tk_ivec_destroy(anchors[i]);
  }
  free(class_ids);
  free(anchors);
  return 0;

cleanup_multiclass:
  if (class_ids) {
    for (uint64_t i = 0; i < n_classes; i++)
      if (class_ids[i])
        tk_ivec_destroy(class_ids[i]);
    free(class_ids);
  }
  if (anchors) {
    for (uint64_t i = 0; i < n_classes; i++)
      if (anchors[i])
        tk_ivec_destroy(anchors[i]);
    free(anchors);
  }
  if (*out_pos)
    tk_pvec_destroy(*out_pos);
  if (*out_neg)
    tk_pvec_destroy(*out_neg);
  *out_pos = NULL;
  *out_neg = NULL;
  return -1;
}

static inline void tk_graph_adj_mst (
  tk_ivec_t *offset,
  tk_ivec_t *neighbors,
  tk_dvec_t *weights,
  tk_ivec_t *mst_offset,
  tk_ivec_t *mst_neighbors,
  tk_dvec_t *mst_weights,
  tk_ivec_t *mst_sources
) {
  if (!offset->n)
    return;

  int kha;
  tk_euset_t *edgeset = tk_euset_create(0, neighbors->n / 2);
  for (int64_t i = 0; i < (int64_t) offset->n - 1; i ++)
    for (int64_t j = offset->a[i]; j < offset->a[i + 1]; j ++) {
      int64_t neighbor = neighbors->a[j];
      tk_euset_put(edgeset, tk_edge(i, neighbor, weights->a[j]), &kha);
    }

  tk_evec_t *edges = tk_euset_keys(0, edgeset);
  tk_evec_desc(edges, 0, edges->n);
  tk_ivec_t *ids = tk_ivec_create(0, offset->n - 1, 0, 0);
  tk_ivec_fill_indices(ids);

  tk_dsu_t *dsu = tk_dsu_create(NULL, ids);
  tk_evec_t *mst_edges = tk_evec_create(0, 0, 0, 0);

  for (uint64_t i = 0; i < edges->n; i ++) {
    tk_edge_t e = edges->a[i];
    if (tk_dsu_find(dsu, e.u) == tk_dsu_find(dsu, e.v))
      continue;
    tk_dsu_union(dsu, e.u, e.v);
    if (tk_evec_push(mst_edges, tk_edge(e.u, e.v, e.w)) != 0)
      goto cleanup;
  }

  ks_introsort(tk_evec_asc_u, mst_edges->n, mst_edges->a);
  tk_ivec_clear(mst_offset);
  tk_ivec_clear(mst_neighbors);
  tk_dvec_clear(mst_weights);
  tk_ivec_clear(mst_sources);

  if (mst_edges->n == 0) {
    for (uint64_t i = 0; i <= offset->n - 1; i++)
      if (tk_ivec_push(mst_offset, 0) != 0)
        goto cleanup;
    tk_evec_destroy(mst_edges);
    tk_evec_destroy(edges);
    tk_euset_destroy(edgeset);
    tk_ivec_destroy(ids);
    return;
  }

  tk_ivec_t *degree = tk_ivec_create(0, offset->n - 1, 0, 0);
  if (!degree)
    goto cleanup;
  for (uint64_t i = 0; i < offset->n - 1; i++)
    degree->a[i] = 0;

  for (uint64_t i = 0; i < mst_edges->n; i++) {
    tk_edge_t e = mst_edges->a[i];
    if (e.u >= 0 && e.u < (int64_t)(offset->n - 1) &&
        e.v >= 0 && e.v < (int64_t)(offset->n - 1)) {
      degree->a[e.u]++;
      degree->a[e.v]++;
    }
  }

  if (tk_ivec_push(mst_offset, 0) != 0)
    goto cleanup;
  for (uint64_t i = 0; i < offset->n - 1; i++) {
    int64_t prev_offset = mst_offset->a[mst_offset->n - 1];
    int64_t deg = degree->a[i];
    if (deg < 0 || prev_offset < 0 || (deg > 0 && prev_offset > INT64_MAX - deg)) {
      if (tk_ivec_push(mst_offset, INT64_MAX) != 0)
        goto cleanup;
    } else {
      if (tk_ivec_push(mst_offset, prev_offset + deg) != 0)
        goto cleanup;
    }
  }

  // Add final offset entry for CSR format
  int64_t final_offset = mst_offset->a[mst_offset->n - 1];
  if (tk_ivec_push(mst_offset, final_offset) != 0)
    goto cleanup;

  tk_ivec_t **adj_lists = calloc(offset->n - 1, sizeof(tk_ivec_t*));
  tk_dvec_t **weight_lists = calloc(offset->n - 1, sizeof(tk_dvec_t*));
  if (!adj_lists || !weight_lists) {
    free(adj_lists);
    free(weight_lists);
    tk_ivec_destroy(degree);
    tk_evec_destroy(mst_edges);
    tk_evec_destroy(edges);
    tk_euset_destroy(edgeset);
    tk_ivec_destroy(ids);
    return;
  }
  for (uint64_t i = 0; i < offset->n - 1; i++) {
    adj_lists[i] = tk_ivec_create(0, 0, 0, 0);
    if (!adj_lists[i]) goto cleanup;
    weight_lists[i] = tk_dvec_create(0, 0, 0, 0);
    if (!weight_lists[i]) goto cleanup;
  }

  for (uint64_t i = 0; i < mst_edges->n; i++) {
    tk_edge_t e = mst_edges->a[i];
    if (e.u >= 0 && e.u < (int64_t)(offset->n - 1) &&
        e.v >= 0 && e.v < (int64_t)(offset->n - 1)) {
      if (tk_ivec_push(adj_lists[e.u], e.v) != 0)
        goto cleanup;
      if (tk_dvec_push(weight_lists[e.u], e.w) != 0)
        goto cleanup;
      if (tk_ivec_push(adj_lists[e.v], e.u) != 0)
        goto cleanup;
      if (tk_dvec_push(weight_lists[e.v], e.w) != 0)
        goto cleanup;
    }
  }

  // Sort each adjacency list by weight descending
  for (uint64_t i = 0; i < offset->n - 1; i++) {
    if (adj_lists[i] && adj_lists[i]->n > 0) {
      tk_rvec_t *temp = tk_rvec_create(0, 0, 0, 0);
      if (!temp) goto cleanup;
      for (uint64_t j = 0; j < adj_lists[i]->n; j++) {
        if (tk_rvec_push(temp, tk_rank(adj_lists[i]->a[j], weight_lists[i]->a[j])) != 0) {
          tk_rvec_destroy(temp);
          goto cleanup;
        }
      }
      tk_rvec_desc(temp, 0, temp->n);
      for (uint64_t j = 0; j < temp->n; j++) {
        adj_lists[i]->a[j] = temp->a[j].i;
        weight_lists[i]->a[j] = temp->a[j].d;
      }
      tk_rvec_destroy(temp);
    }
  }

  for (uint64_t i = 0; i < offset->n - 1; i++) {
    if (adj_lists[i]) {
      for (uint64_t j = 0; j < adj_lists[i]->n; j++) {
        int64_t neighbor = adj_lists[i]->a[j];
        double weight = weight_lists[i]->a[j];
        if (neighbor >= 0 && neighbor < (int64_t)(offset->n - 1)) {
          if (tk_ivec_push(mst_neighbors, neighbor) != 0)
            goto cleanup;
          if (tk_dvec_push(mst_weights, weight) != 0)
            goto cleanup;
          if (tk_ivec_push(mst_sources, (int64_t) i) != 0)
            goto cleanup;
        }
      }
    }
  }

cleanup:
  for (uint64_t i = 0; i < offset->n - 1; i++) {
    if (adj_lists && adj_lists[i]) tk_ivec_destroy(adj_lists[i]);
    if (weight_lists && weight_lists[i]) tk_dvec_destroy(weight_lists[i]);
  }
  free(adj_lists);
  free(weight_lists);
  tk_ivec_destroy(degree);
  tk_evec_destroy(mst_edges);
  tk_evec_destroy(edges);
  tk_euset_destroy(edgeset);
  tk_dsu_destroy(dsu);
  tk_ivec_destroy(ids);
}

static inline int tk_graph_adj_hoods(
  lua_State *L,
  tk_ivec_t *ids,
  tk_inv_hoods_t *inv_hoods,
  tk_ann_hoods_t *ann_hoods,
  tk_hbi_hoods_t *hbi_hoods,
  uint64_t features,
  unsigned int n_threads,
  tk_ivec_t **offsets_out,
  tk_ivec_t **neighbors_out,
  tk_dvec_t **weights_out
) {
  if (!ids || ids->n == 0) {
    *offsets_out = tk_ivec_create(NULL, 1, 0, 0);
    *neighbors_out = tk_ivec_create(NULL, 0, 0, 0);
    *weights_out = tk_dvec_create(NULL, 0, 0, 0);
    if (!*offsets_out || !*neighbors_out || !*weights_out) {
      if (*offsets_out) tk_ivec_destroy(*offsets_out);
      if (*neighbors_out) tk_ivec_destroy(*neighbors_out);
      if (*weights_out) tk_dvec_destroy(*weights_out);
      return -1;
    }
    (*offsets_out)->a[0] = 0;
    (*offsets_out)->n = 1;
    return 0;
  }

  uint64_t n_queries = ids->n;

  // Phase 1: Build offsets (sequential)
  *offsets_out = tk_ivec_create(L, n_queries + 1, 0, 0);
  if (!*offsets_out)
    return -1;

  (*offsets_out)->a[0] = 0;
  for (uint64_t i = 0; i < n_queries; i++) {
    uint64_t n_neighbors = 0;
    if (ann_hoods && i < ann_hoods->n)
      n_neighbors = ann_hoods->a[i]->n;
    else if (hbi_hoods && i < hbi_hoods->n)
      n_neighbors = hbi_hoods->a[i]->n;
    else if (inv_hoods && i < inv_hoods->n)
      n_neighbors = inv_hoods->a[i]->n;
    (*offsets_out)->a[i + 1] = (*offsets_out)->a[i] + (int64_t)n_neighbors;
  }
  (*offsets_out)->n = n_queries + 1;

  // Phase 2: Allocate neighbors and weights
  uint64_t total = (uint64_t)(*offsets_out)->a[n_queries];
  *neighbors_out = tk_ivec_create(L, total, 0, 0);
  *weights_out = tk_dvec_create(L, total, 0, 0);
  if (!*neighbors_out || !*weights_out) {
    if (*neighbors_out) tk_ivec_destroy(*neighbors_out);
    if (*weights_out) tk_dvec_destroy(*weights_out);
    tk_ivec_destroy(*offsets_out);
    return -1;
  }
  (*neighbors_out)->n = total;
  (*weights_out)->n = total;

  // Phase 3: Flatten in parallel using existing infrastructure
  tk_graph_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
  if (!threads) {
    tk_ivec_destroy(*neighbors_out);
    tk_dvec_destroy(*weights_out);
    tk_ivec_destroy(*offsets_out);
    return -1;
  }
  memset(threads, 0, n_threads * sizeof(tk_graph_thread_t));

  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_graph_worker);
  if (!pool) {
    free(threads);
    tk_ivec_destroy(*neighbors_out);
    tk_dvec_destroy(*weights_out);
    tk_ivec_destroy(*offsets_out);
    return -1;
  }

  for (unsigned int i = 0; i < n_threads; i++) {
    threads[i].task.adj_hoods.ids = ids;
    threads[i].task.adj_hoods.inv_hoods = inv_hoods;
    threads[i].task.adj_hoods.ann_hoods = ann_hoods;
    threads[i].task.adj_hoods.hbi_hoods = hbi_hoods;
    threads[i].task.adj_hoods.features = features;
    threads[i].task.adj_hoods.offsets = *offsets_out;
    threads[i].task.adj_hoods.neighbors = *neighbors_out;
    threads[i].task.adj_hoods.weights = *weights_out;
    pool->threads[i].data = &threads[i];
    tk_thread_range(i, n_threads, n_queries, &threads[i].ifirst, &threads[i].ilast);
    atomic_init(&threads[i].has_error, false);
  }

  tk_threads_signal(pool, TK_GRAPH_ADJ_HOODS, 0);

  // Check for errors
  bool has_error = false;
  for (unsigned int i = 0; i < n_threads; i++) {
    if (atomic_load(&threads[i].has_error)) {
      has_error = true;
      break;
    }
  }

  tk_threads_destroy(pool);
  free(threads);

  if (has_error) {
    tk_ivec_destroy(*neighbors_out);
    tk_dvec_destroy(*weights_out);
    tk_ivec_destroy(*offsets_out);
    return -1;
  }

  return 0;
}

#endif
