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
  TK_GRAPH_WEIGHT_POOL_MIN,
  TK_GRAPH_WEIGHT_POOL_MAX
} tk_graph_weight_pooling_t;


typedef struct tk_graph_s {

  tk_euset_t *pairs;
  tk_graph_adj_t *adj;

  tk_inv_t *knn_inv; tk_inv_hoods_t *knn_inv_hoods;
  tk_ann_t *knn_ann; tk_ann_hoods_t *knn_ann_hoods;
  tk_hbi_t *knn_hbi; tk_hbi_hoods_t *knn_hbi_hoods;
  tk_ivec_sim_type_t knn_cmp;
  double knn_cmp_alpha, knn_cmp_beta;
  int64_t knn_rank;

  tk_inv_t *category_inv;
  tk_ivec_sim_type_t category_cmp;
  double category_alpha, category_beta;
  uint64_t category_anchors;
  uint64_t category_knn;
  double category_knn_decay;
  uint64_t category_negatives;

  tk_inv_t *weight_inv;
  tk_ann_t *weight_ann;
  tk_hbi_t *weight_hbi;
  tk_ivec_sim_type_t weight_cmp;
  double weight_alpha, weight_beta;
  tk_graph_weight_pooling_t weight_pooling;

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
  int64_t category_ranks;

  tk_dvec_t *sigmas;
  uint64_t n_edges;
  tk_dsu_t *dsu;
  int64_t largest_component_root;
  tk_dvec_t *q_weights;
  tk_dvec_t *e_weights;
  tk_dvec_t *inter_weights;

  bool destroyed;

} tk_graph_t;


static inline tk_graph_t *tk_graph_peek (lua_State *L, int i)
{
  return (tk_graph_t *) luaL_checkudata(L, i, TK_GRAPH_MT);
}

#define TK_GRAPH_INDEX_DISTANCE(idx_inv, idx_ann, idx_hbi, u, v, cmp, alpha, beta, q_w, e_w, i_w, dist_var) \
  do { \
    tk_inv_t *__idx_inv = (idx_inv); \
    tk_ann_t *__idx_ann = (idx_ann); \
    tk_hbi_t *__idx_hbi = (idx_hbi); \
    (dist_var) = DBL_MAX; \
    if (__idx_inv != NULL) { \
      size_t un = 0, vn = 0; \
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
        (dist_var) = (double)tk_cvec_bits_hamming_serial((const uint8_t *)uset, (const uint8_t *)wset, \
                                                   __idx_ann->features) / (double)__idx_ann->features; \
      } \
    } else if (__idx_hbi != NULL) { \
      char *uset = tk_hbi_get(__idx_hbi, (u)); \
      char *wset = tk_hbi_get(__idx_hbi, (v)); \
      if (uset && wset) { \
        (dist_var) = (double)tk_cvec_bits_hamming_serial((const uint8_t *)uset, (const uint8_t *)wset, \
                                                   __idx_hbi->features) / (double)__idx_hbi->features; \
      } \
    } \
  } while(0)

#define TK_GRAPH_FOREACH_HOOD_NEIGHBOR(inv_hoods, ann_hoods, hbi_hoods, hood_idx, uids_hoods, neighbor_idx_var, neighbor_uid_var, body) \
  do { \
    if ((inv_hoods) != NULL && (hood_idx) < (inv_hoods)->n) { \
      tk_rvec_t *__hood = (inv_hoods)->a[hood_idx]; \
      for (uint64_t __j = 0; __j < __hood->n; __j++) { \
        tk_rank_t __r = __hood->a[__j]; \
        if (__r.i < 0 || __r.i >= (int64_t)(uids_hoods)->n) continue; \
        (neighbor_idx_var) = __r.i; \
        (neighbor_uid_var) = (uids_hoods)->a[__r.i]; \
        body \
      } \
    } else if ((ann_hoods) != NULL && (hood_idx) < (ann_hoods)->n) { \
      tk_pvec_t *__hood = (ann_hoods)->a[hood_idx]; \
      for (uint64_t __j = 0; __j < __hood->n; __j++) { \
        tk_pair_t __r = __hood->a[__j]; \
        if (__r.i < 0 || __r.i >= (int64_t)(uids_hoods)->n) continue; \
        (neighbor_idx_var) = __r.i; \
        (neighbor_uid_var) = (uids_hoods)->a[__r.i]; \
        body \
      } \
    } else if ((hbi_hoods) != NULL && (hood_idx) < (hbi_hoods)->n) { \
      tk_pvec_t *__hood = (hbi_hoods)->a[hood_idx]; \
      for (uint64_t __j = 0; __j < __hood->n; __j++) { \
        tk_pair_t __r = __hood->a[__j]; \
        if (__r.i < 0 || __r.i >= (int64_t)(uids_hoods)->n) continue; \
        (neighbor_idx_var) = __r.i; \
        (neighbor_uid_var) = (uids_hoods)->a[__r.i]; \
        body \
      } \
    } \
  } while(0)

#define TK_GRAPH_HOOD_DISTANCE(inv_hoods, ann_hoods, hbi_hoods, hood_idx, elem_idx, features_ann, features_hbi, dist_var) \
  do { \
    if ((inv_hoods) != NULL && (hood_idx) < (inv_hoods)->n) { \
      (dist_var) = (inv_hoods)->a[hood_idx]->a[elem_idx].d; \
    } else if ((ann_hoods) != NULL && (hood_idx) < (ann_hoods)->n) { \
      (dist_var) = (double)(ann_hoods)->a[hood_idx]->a[elem_idx].p / (double)(features_ann); \
    } else if ((hbi_hoods) != NULL && (hood_idx) < (hbi_hoods)->n) { \
      (dist_var) = (double)(hbi_hoods)->a[hood_idx]->a[elem_idx].p / (double)(features_hbi); \
    } \
  } while(0)

#define TK_GRAPH_HOOD_SIZE(inv_hoods, ann_hoods, hbi_hoods, idx, size_var) \
  do { \
    (size_var) = 0; \
    if ((inv_hoods) && (idx) < (inv_hoods)->n) { \
      (size_var) = (inv_hoods)->a[idx]->n; \
    } else if ((ann_hoods) && (idx) < (ann_hoods)->n) { \
      (size_var) = (ann_hoods)->a[idx]->n; \
    } else if ((hbi_hoods) && (idx) < (hbi_hoods)->n) { \
      (size_var) = (hbi_hoods)->a[idx]->n; \
    } \
  } while(0)

#define TK_GRAPH_HOOD_TO_CSR(inv_hoods, ann_hoods, hbi_hoods, idx, features, neighbors_arr, weights_arr, write_pos, end_pos) \
  do { \
    if ((inv_hoods) && (idx) < (inv_hoods)->n) { \
      tk_rvec_t *__hood = (inv_hoods)->a[idx]; \
      for (uint64_t __j = 0; __j < __hood->n && (write_pos) < (end_pos); __j++) { \
        (neighbors_arr)[(write_pos)] = __hood->a[__j].i; \
        (weights_arr)[(write_pos)] = 1.0 - __hood->a[__j].d; \
        (write_pos)++; \
      } \
    } else if ((ann_hoods) && (idx) < (ann_hoods)->n) { \
      tk_pvec_t *__hood = (ann_hoods)->a[idx]; \
      for (uint64_t __j = 0; __j < __hood->n && (write_pos) < (end_pos); __j++) { \
        (neighbors_arr)[(write_pos)] = __hood->a[__j].i; \
        double __dist = (double)__hood->a[__j].p / (double)(features); \
        (weights_arr)[(write_pos)] = 1.0 - __dist; \
        (write_pos)++; \
      } \
    } else if ((hbi_hoods) && (idx) < (hbi_hoods)->n) { \
      tk_pvec_t *__hood = (hbi_hoods)->a[idx]; \
      for (uint64_t __j = 0; __j < __hood->n && (write_pos) < (end_pos); __j++) { \
        (neighbors_arr)[(write_pos)] = __hood->a[__j].i; \
        double __dist = (double)__hood->a[__j].p / (double)(features); \
        (weights_arr)[(write_pos)] = 1.0 - __dist; \
        (write_pos)++; \
      } \
    } \
  } while(0)

#define TK_INDEX_NEIGHBORHOODS(inv, ann, hbi, L, k, probe_radius, eps_min, eps_max, min, mutual, cmp, alpha, beta, rank, inv_hoods_out, ann_hoods_out, hbi_hoods_out, uids_out) \
  do { \
    if ((inv) != NULL) { \
      tk_inv_neighborhoods((L), (inv), (k), 0.0, 1.0, (min), (cmp), (alpha), (beta), (mutual), (rank), (inv_hoods_out), (uids_out)); \
    } else if ((ann) != NULL) { \
      tk_ann_neighborhoods((L), (ann), (k), (probe_radius), (eps_min), (eps_max), (min), (mutual), (ann_hoods_out), (uids_out)); \
    } else if ((hbi) != NULL) { \
      tk_hbi_neighborhoods((L), (hbi), (k), (uint64_t)(eps_min), (uint64_t)(eps_max), (min), (mutual), (hbi_hoods_out), (uids_out)); \
    } \
  } while(0)

#define TK_INDEX_MUTUALIZE(inv, ann, hbi, L, inv_hoods, ann_hoods, hbi_hoods, uids, min, precomputed) \
  do { \
    if ((inv) != NULL) { \
      tk_inv_mutualize((L), (inv), (inv_hoods), (uids), (min), (precomputed)); \
    } else if ((ann) != NULL) { \
      tk_ann_mutualize((L), (ann), (ann_hoods), (uids), (min), (precomputed)); \
    } else if ((hbi) != NULL) { \
      tk_hbi_mutualize((L), (hbi), (hbi_hoods), (uids), (min), (precomputed)); \
    } \
  } while(0)

#define TK_GRAPH_FOREACH_HOOD_FOR_SIGMA(inv_hoods, ann_hoods, hbi_hoods, features_ann, features_hbi, hood_idx, graph, uid, uids_hoods, uids_idx, seen, distances, q_w, e_w, i_w, error_var) \
  do { \
    if ((inv_hoods) && (hood_idx) < (int64_t)(inv_hoods)->n) { \
      tk_rvec_t *__hood = (inv_hoods)->a[hood_idx]; \
      for (uint64_t __j = 0; __j < __hood->m; __j++) { \
        int64_t __nh_idx = __hood->a[__j].i; \
        if (__nh_idx >= 0 && __nh_idx < (int64_t)(uids_hoods)->n) { \
          int64_t __nh_uid = (uids_hoods)->a[__nh_idx]; \
          uint32_t __n_khi = tk_iumap_get((uids_idx), __nh_uid); \
          if (__n_khi != tk_iumap_end((uids_idx))) { \
            int64_t __nh_global_idx = tk_iumap_val((uids_idx), __n_khi); \
            int __kha; \
            tk_iuset_put((seen), __nh_global_idx, &__kha); \
            if (__kha) { \
              double __d = tk_graph_distance((graph), (uid), __nh_uid, (q_w), (e_w), (i_w)); \
              if (__d == DBL_MAX) __d = __hood->a[__j].d; \
              if (tk_dvec_push((distances), __d) != 0) { (error_var) = true; break; } \
            } \
          } \
        } \
      } \
    } else if ((ann_hoods) && (hood_idx) < (int64_t)(ann_hoods)->n) { \
      tk_pvec_t *__hood = (ann_hoods)->a[hood_idx]; \
      double __denom = (features_ann) ? (double)(features_ann) : 1.0; \
      for (uint64_t __j = 0; __j < __hood->m; __j++) { \
        int64_t __nh_idx = __hood->a[__j].i; \
        if (__nh_idx >= 0 && __nh_idx < (int64_t)(uids_hoods)->n) { \
          int64_t __nh_uid = (uids_hoods)->a[__nh_idx]; \
          uint32_t __n_khi = tk_iumap_get((uids_idx), __nh_uid); \
          if (__n_khi != tk_iumap_end((uids_idx))) { \
            int64_t __nh_global_idx = tk_iumap_val((uids_idx), __n_khi); \
            int __kha; \
            tk_iuset_put((seen), __nh_global_idx, &__kha); \
            if (__kha) { \
              double __d = tk_graph_distance((graph), (uid), __nh_uid, (q_w), (e_w), (i_w)); \
              if (__d == DBL_MAX) __d = (double)__hood->a[__j].p / __denom; \
              if (tk_dvec_push((distances), __d) != 0) { (error_var) = true; break; } \
            } \
          } \
        } \
      } \
    } else if ((hbi_hoods) && (hood_idx) < (int64_t)(hbi_hoods)->n) { \
      tk_pvec_t *__hood = (hbi_hoods)->a[hood_idx]; \
      double __denom = (features_hbi) ? (double)(features_hbi) : 1.0; \
      for (uint64_t __j = 0; __j < __hood->m; __j++) { \
        int64_t __nh_idx = __hood->a[__j].i; \
        if (__nh_idx >= 0 && __nh_idx < (int64_t)(uids_hoods)->n) { \
          int64_t __nh_uid = (uids_hoods)->a[__nh_idx]; \
          uint32_t __n_khi = tk_iumap_get((uids_idx), __nh_uid); \
          if (__n_khi != tk_iumap_end((uids_idx))) { \
            int64_t __nh_global_idx = tk_iumap_val((uids_idx), __n_khi); \
            int __kha; \
            tk_iuset_put((seen), __nh_global_idx, &__kha); \
            if (__kha) { \
              double __d = tk_graph_distance((graph), (uid), __nh_uid, (q_w), (e_w), (i_w)); \
              if (__d == DBL_MAX) __d = (double)__hood->a[__j].p / __denom; \
              if (tk_dvec_push((distances), __d) != 0) { (error_var) = true; break; } \
            } \
          } \
        } \
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

  TK_GRAPH_INDEX_DISTANCE(graph->weight_inv, graph->weight_ann, graph->weight_hbi,
                          u, v, graph->weight_cmp, graph->weight_alpha, graph->weight_beta,
                          q_weights, e_weights, inter_weights, d);
  if (d != DBL_MAX)
    return d;

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
  tk_pvec_t **out_pairs
) {
  if (ids->n == 0) {
    *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
    return *out_pairs ? 0 : -1;
  }

  *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
  if (!*out_pairs)
    return -1;

  bool has_error = false;

  #pragma omp parallel reduction(||:has_error)
  {
    tk_pvec_t *local_pairs = tk_pvec_create(NULL, 0, 0, 0);
    if (!local_pairs) {
      has_error = true;
    } else {
      #pragma omp for schedule(static)
      for (uint64_t idx = 0; idx < ids->n; idx++) {
        if (has_error) continue;

        int64_t id = ids->a[idx];
        #define _tk_graph_process_hood(hood) \
          do { \
            for (uint64_t j = 0; j < (hood)->n; j++) { \
              int64_t idxnbr = (hood)->a[j].i; \
              if (idxnbr >= 0 && idxnbr < (int64_t)ids->n) { \
                int64_t nbr = ids->a[idxnbr]; \
                if (tk_pvec_push(local_pairs, tk_pair(id, nbr)) != 0) { \
                  has_error = true; \
                  break; \
                } \
              } \
            } \
          } while(0)

        if (inv_hoods != NULL)
          _tk_graph_process_hood(inv_hoods->a[idx]);
        else if (ann_hoods != NULL)
          _tk_graph_process_hood(ann_hoods->a[idx]);
        else if (hbi_hoods != NULL)
          _tk_graph_process_hood(hbi_hoods->a[idx]);

        #undef _tk_graph_process_hood
      }

      #pragma omp critical
      {
        for (uint64_t j = 0; j < local_pairs->n; j++) {
          if (tk_pvec_push(*out_pairs, local_pairs->a[j]) != 0) {
            has_error = true;
            break;
          }
        }
      }

      tk_pvec_destroy(local_pairs);
    }
  }

  if (has_error) {
    tk_pvec_destroy(*out_pairs);
    *out_pairs = NULL;
    return -1;
  }

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
    tk_iumap_t **ids_by_label = NULL;
    int64_t max_label = -1;

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

    for (int64_t lbl = 0; lbl <= max_label; lbl++) {
      if (!ids_by_label[lbl])
        continue;
      uint64_t n_in_class = tk_iumap_size(ids_by_label[lbl]);
      if (n_in_class == 0)
        continue;

      uint64_t n_label_anchors = n_anchors < n_in_class ? n_anchors : n_in_class;
      tk_iuset_t *selected_anchors = tk_iuset_create(NULL, 0);
      if (!selected_anchors) goto cleanup_label;

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
  tk_pvec_t **out_pairs
) {
  *out_pairs = NULL;

  if (ids->n == 0) {
    *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
    return *out_pairs ? 0 : -1;
  }

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

  *out_pairs = tk_pvec_create(NULL, 0, 0, 0);
  if (!*out_pairs)
    goto cleanup_random;

  bool has_error = false;

  #pragma omp parallel reduction(||:has_error)
  {
    tk_pvec_t *local_pairs = tk_pvec_create(NULL, 0, 0, 0);
    if (!local_pairs) {
      has_error = true;
    } else {
      #pragma omp for schedule(static)
      for (uint64_t i = 0; i < ids->n; i++) {
        if (has_error) continue;

        int64_t id1 = ids->a[i];
        for (uint64_t e = 0; e < edges_per_node; e++) {
          int64_t id2 = -1;
          uint64_t idx2 = tk_fast_random() % ids->n;
          if (idx2 == i)
            idx2 = (idx2 + 1) % ids->n;
          id2 = ids->a[idx2];
          if (id2 >= 0 && id2 != id1) {
            if (tk_pvec_push(local_pairs, tk_pair(id1, id2)) != 0) {
              has_error = true;
              break;
            }
          }
        }
      }

      #pragma omp critical
      {
        for (uint64_t j = 0; j < local_pairs->n; j++) {
          if (tk_pvec_push(*out_pairs, local_pairs->a[j]) != 0) {
            has_error = true;
            break;
          }
        }
      }

      tk_pvec_destroy(local_pairs);
    }
  }

  if (has_error) {
    tk_pvec_destroy(*out_pairs);
    *out_pairs = NULL;
    goto cleanup_random;
  }

  tk_pvec_xasc(*out_pairs, 0, (*out_pairs)->n);

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

  *out_pos = tk_pvec_create(NULL, 0, 0, 0);
  if (!*out_pos) goto cleanup_multiclass;

  if (n_anchors_pos > 0) {
    bool need_buffers = index && index->ranks && index->n_ranks > 1;
    bool has_error = false;

    #pragma omp parallel reduction(||:has_error)
    {
      tk_pvec_t *local_pos = tk_pvec_create(NULL, 0, 0, 0);
      tk_dvec_t *q_weights = NULL;
      tk_dvec_t *e_weights = NULL;
      tk_dvec_t *inter_weights = NULL;

      if (!local_pos) {
        has_error = true;
      } else {
        if (need_buffers) {
          q_weights = tk_dvec_create(NULL, 0, 0, 0);
          e_weights = tk_dvec_create(NULL, 0, 0, 0);
          inter_weights = tk_dvec_create(NULL, 0, 0, 0);
          if (!q_weights || !e_weights || !inter_weights) {
            has_error = true;
          }
        }

        if (!has_error) {
          #pragma omp for schedule(static)
          for (uint64_t class_idx = 0; class_idx < n_classes; class_idx++) {
            if (has_error) continue;

            tk_ivec_t *class_ids_c = class_ids[class_idx];
            tk_ivec_t *anchors_c = anchors[class_idx];
            if (!class_ids_c || !anchors_c)
              continue;

            for (uint64_t i = 0; i < class_ids_c->n; i++) {
              int64_t id = class_ids_c->a[i];
              for (uint64_t j = 0; j < anchors_c->n; j++) {
                int64_t anchor = anchors_c->a[j];
                if (id == anchor)
                  continue;
                if (index && eps_pos > 0.0) {
                  double dist = tk_inv_distance(
                    index, id, anchor,
                    TK_IVEC_JACCARD, 0.0, 0.0,
                    q_weights, e_weights, inter_weights);
                  if (dist > eps_pos)
                    continue;
                }
                int64_t a = id < anchor ? id : anchor;
                int64_t b = id < anchor ? anchor : id;
                if (tk_pvec_push(local_pos, tk_pair(a, b)) != 0) {
                  has_error = true;
                  break;
                }
              }
              if (has_error) break;
            }
          }

          #pragma omp critical
          {
            for (uint64_t j = 0; j < local_pos->n; j++) {
              if (tk_pvec_push(*out_pos, local_pos->a[j]) != 0) {
                has_error = true;
                break;
              }
            }
          }
        }

        if (local_pos) tk_pvec_destroy(local_pos);
        if (q_weights) tk_dvec_destroy(q_weights);
        if (e_weights) tk_dvec_destroy(e_weights);
        if (inter_weights) tk_dvec_destroy(inter_weights);
      }
    }

    if (has_error)
      goto cleanup_multiclass;
  }

  *out_neg = tk_pvec_create(NULL, 0, 0, 0);
  if (!*out_neg) goto cleanup_multiclass;

  if (n_anchors_neg > 0) {
    bool need_buffers_neg = index && index->ranks && index->n_ranks > 1;
    bool has_error = false;

    #pragma omp parallel reduction(||:has_error)
    {
      tk_pvec_t *local_neg = tk_pvec_create(NULL, 0, 0, 0);
      tk_dvec_t *q_weights = NULL;
      tk_dvec_t *e_weights = NULL;
      tk_dvec_t *inter_weights = NULL;

      if (!local_neg) {
        has_error = true;
      } else {
        if (need_buffers_neg) {
          q_weights = tk_dvec_create(NULL, 0, 0, 0);
          e_weights = tk_dvec_create(NULL, 0, 0, 0);
          inter_weights = tk_dvec_create(NULL, 0, 0, 0);
          if (!q_weights || !e_weights || !inter_weights) {
            has_error = true;
          }
        }

        if (!has_error) {
          #pragma omp for schedule(static)
          for (uint64_t class_idx = 0; class_idx < n_classes; class_idx++) {
            if (has_error) continue;

            tk_ivec_t *class_ids_c = class_ids[class_idx];
            if (!class_ids_c)
              continue;

            for (uint64_t i = 0; i < class_ids_c->n; i++) {
              int64_t id = class_ids_c->a[i];
              for (uint64_t n = 0; n < n_anchors_neg; n++) {
                uint64_t tries = 0;
                uint64_t max_tries = 100;
                int64_t other_id = -1;
                while (tries < max_tries) {
                  uint64_t other_class = tk_fast_random() % n_classes;
                  if (other_class == class_idx) {
                    tries++;
                    continue;
                  }
                  tk_ivec_t *other_class_ids = class_ids[other_class];
                  if (!other_class_ids || other_class_ids->n == 0) {
                    tries++;
                    continue;
                  }
                  uint64_t other_idx = tk_fast_random() % other_class_ids->n;
                  other_id = other_class_ids->a[other_idx];
                  if (index && eps_neg > 0.0) {
                    double dist = tk_inv_distance(
                      index, id, other_id,
                      TK_IVEC_JACCARD, 0.0, 0.0,
                      q_weights, e_weights, inter_weights);
                    if (dist < eps_neg) {
                      tries++;
                      continue;
                    }
                  }
                  break;
                }
                if (other_id >= 0 && tries < max_tries) {
                  int64_t a = id < other_id ? id : other_id;
                  int64_t b = id < other_id ? other_id : id;
                  if (tk_pvec_push(local_neg, tk_pair(a, b)) != 0) {
                    has_error = true;
                    break;
                  }
                }
              }
              if (has_error) break;
            }
          }

          #pragma omp critical
          {
            for (uint64_t j = 0; j < local_neg->n; j++) {
              if (tk_pvec_push(*out_neg, local_neg->a[j]) != 0) {
                has_error = true;
                break;
              }
            }
          }
        }

        if (local_neg) tk_pvec_destroy(local_neg);
        if (q_weights) tk_dvec_destroy(q_weights);
        if (e_weights) tk_dvec_destroy(e_weights);
        if (inter_weights) tk_dvec_destroy(inter_weights);
      }
    }

    if (has_error)
      goto cleanup_multiclass;
  }

  tk_pvec_xasc(*out_pos, 0, (*out_pos)->n);
  tk_pvec_xasc(*out_neg, 0, (*out_neg)->n);

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

  *offsets_out = tk_ivec_create(L, n_queries + 1, 0, 0);
  if (!*offsets_out)
    return -1;

  (*offsets_out)->a[0] = 0;
  for (uint64_t i = 0; i < n_queries; i++) {
    uint64_t n_neighbors = 0;
    TK_GRAPH_HOOD_SIZE(inv_hoods, ann_hoods, hbi_hoods, i, n_neighbors);
    (*offsets_out)->a[i + 1] = (*offsets_out)->a[i] + (int64_t)n_neighbors;
  }
  (*offsets_out)->n = n_queries + 1;

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
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_queries; i++) {
    int64_t start = (*offsets_out)->a[i];
    int64_t end = (*offsets_out)->a[i + 1];
    int64_t idx = start;
    TK_GRAPH_HOOD_TO_CSR(inv_hoods, ann_hoods, hbi_hoods, i, features,
                         (*neighbors_out)->a, (*weights_out)->a, idx, end);
  }

  return 0;
}

#endif
