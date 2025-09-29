#ifndef TK_GRAPH_H
#define TK_GRAPH_H

#include <santoku/tsetlin/conf.h>
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

#define TK_GRAPH_MT "tk_graph_t"
#define TK_GRAPH_EPH "tk_graph_eph"

typedef tk_iuset_t * tk_graph_adj_item_t;
#define tk_vec_name tk_graph_adj
#define tk_vec_base tk_graph_adj_item_t
#define tk_vec_destroy_item(...) tk_iuset_destroy(__VA_ARGS__)
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef enum {
  TK_GRAPH_CSR_OFFSET_LOCAL,
  TK_GRAPH_CSR_OFFSET_GLOBAL,
  TK_GRAPH_CSR_DATA,
  TK_GRAPH_CSR_SOURCES,
  TK_GRAPH_SIGMA,
  TK_GRAPH_REWEIGHT
} tk_graph_stage_t;

typedef struct tk_graph_thread_s tk_graph_thread_t;

typedef struct tk_graph_s {

  tk_euset_t *pairs;
  tk_graph_adj_t *adj;
  tk_inv_t *inv; tk_inv_hoods_t *inv_hoods; tk_ivec_sim_type_t cmp; double cmp_alpha, cmp_beta;
  tk_ann_t *ann; tk_ann_hoods_t *ann_hoods;
  tk_hbi_t *hbi; tk_hbi_hoods_t *hbi_hoods;
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
  int64_t knn_rank;
  bool bridge;

  tk_dvec_t *sigmas;
  uint64_t n_edges;
  tk_dsu_t dsu;
  int64_t largest_component_root;

  tk_graph_thread_t *threads;
  tk_threadpool_t *pool;

} tk_graph_t;

typedef struct tk_graph_thread_s {
  tk_graph_t *graph;
  tk_ivec_t *adj_offset;
  tk_ivec_t *adj_data;
  tk_dvec_t *adj_weights;
  tk_ivec_t *adj_sources;
  int64_t csr_base;
  uint64_t ifirst, ilast;
  unsigned int index;
} tk_graph_thread_t;

static inline tk_graph_t *tk_graph_peek (lua_State *L, int i)
{
  return (tk_graph_t *) luaL_checkudata(L, i, TK_GRAPH_MT);
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

  tk_dsu_t dsu;
  tk_dsu_init(&dsu, ids);
  tk_evec_t *mst_edges = tk_evec_create(0, 0, 0, 0);

  for (uint64_t i = 0; i < edges->n; i ++) {
    tk_edge_t e = edges->a[i];
    if (tk_dsu_find(&dsu, e.u) == tk_dsu_find(&dsu, e.v))
      continue;
    tk_dsu_union(&dsu, e.u, e.v);
    tk_evec_push(mst_edges, tk_edge(e.u, e.v, e.w));
  }

  ks_introsort(tk_evec_asc_u, mst_edges->n, mst_edges->a);
  tk_ivec_clear(mst_offset);
  tk_ivec_clear(mst_neighbors);
  tk_dvec_clear(mst_weights);
  tk_ivec_clear(mst_sources);

  if (mst_edges->n == 0) {
    for (uint64_t i = 0; i <= offset->n - 1; i++)
      tk_ivec_push(mst_offset, 0);
    tk_evec_destroy(mst_edges);
    tk_evec_destroy(edges);
    tk_euset_destroy(edgeset);
    tk_ivec_destroy(ids);
    return;
  }

  tk_ivec_t *degree = tk_ivec_create(0, offset->n - 1, 0, 0);
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

  tk_ivec_push(mst_offset, 0);
  uint64_t zero_degree_count = 0;
  for (uint64_t i = 0; i < offset->n - 1; i++) {
    int64_t prev_offset = mst_offset->a[mst_offset->n - 1];
    int64_t deg = degree->a[i];
    if (deg == 0) zero_degree_count++;
    if (deg < 0 || prev_offset < 0 || (deg > 0 && prev_offset > INT64_MAX - deg)) {
      tk_ivec_push(mst_offset, INT64_MAX);
    } else {
      tk_ivec_push(mst_offset, prev_offset + deg);
    }
  }

  // Add final offset entry for CSR format
  int64_t final_offset = mst_offset->a[mst_offset->n - 1];
  tk_ivec_push(mst_offset, final_offset);

  tk_ivec_t **adj_lists = malloc((offset->n - 1) * sizeof(tk_ivec_t*));
  tk_dvec_t **weight_lists = malloc((offset->n - 1) * sizeof(tk_dvec_t*));
  for (uint64_t i = 0; i < offset->n - 1; i++) {
    adj_lists[i] = tk_ivec_create(0, 0, 0, 0);
    weight_lists[i] = tk_dvec_create(0, 0, 0, 0);
  }

  for (uint64_t i = 0; i < mst_edges->n; i++) {
    tk_edge_t e = mst_edges->a[i];
    if (e.u >= 0 && e.u < (int64_t)(offset->n - 1) &&
        e.v >= 0 && e.v < (int64_t)(offset->n - 1)) {
      tk_ivec_push(adj_lists[e.u], e.v);
      tk_dvec_push(weight_lists[e.u], e.w);
      tk_ivec_push(adj_lists[e.v], e.u);
      tk_dvec_push(weight_lists[e.v], e.w);
    }
  }

  for (uint64_t i = 0; i < offset->n - 1; i++) {
    for (uint64_t j = 0; j < adj_lists[i]->n; j++) {
      int64_t neighbor = adj_lists[i]->a[j];
      double weight = weight_lists[i]->a[j];
      if (neighbor >= 0 && neighbor < (int64_t)(offset->n - 1)) {
        tk_ivec_push(mst_neighbors, neighbor);
        tk_dvec_push(mst_weights, weight);
        tk_ivec_push(mst_sources, (int64_t) i);
      }
    }
    tk_ivec_destroy(adj_lists[i]);
    tk_dvec_destroy(weight_lists[i]);
  }
  free(adj_lists);
  free(weight_lists);
  tk_ivec_destroy(degree);
  tk_evec_destroy(mst_edges);
  tk_evec_destroy(edges);
  tk_euset_destroy(edgeset);
  tk_ivec_destroy(ids);
}

#endif
