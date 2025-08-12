#ifndef TK_GRAPH_H
#define TK_GRAPH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/pumap.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/ann.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/tsetlin/dsu.h>
#include <float.h>

#define TK_GRAPH_MT "tk_graph_t"
#define TK_GRAPH_EPH "tk_graph_eph"

typedef struct { int64_t tu, tv; int64_t ru, rv; double B; } tk_graph_pair_rec_t;
static inline int tk_graph_pair_rec_lt (
  const tk_graph_pair_rec_t x,
  const tk_graph_pair_rec_t y
) {
  if (x.ru != y.ru) return (x.ru < y.ru) ? true : false;
  if (x.rv != y.rv) return (x.rv < y.rv) ? true : false;
  if (x.tu != y.tu) return (x.tu < y.tu) ? true : false;
  if (x.tv != y.tv) return (x.tv < y.tv) ? true : false;
  return false;
}
static inline int tk_graph_pair_rec_gt (
  const tk_graph_pair_rec_t x,
  const tk_graph_pair_rec_t y
) {
  if (x.ru != y.ru) return (x.ru > y.ru) ? true : false;
  if (x.rv != y.rv) return (x.rv > y.rv) ? true : false;
  if (x.tu != y.tu) return (x.tu > y.tu) ? true : false;
  if (x.tv != y.tv) return (x.tv > y.tv) ? true : false;
  return false;
}

#define tk_vec_name tk_graph_pair_recs
#define tk_vec_base tk_graph_pair_rec_t
#define tk_vec_lt(a, b) tk_graph_pair_rec_lt(a, b)
#define tk_vec_gt(a, b) tk_graph_pair_rec_gt(a, b)
#define tk_vec_limited
#include <santoku/vec/tpl.h>

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
  TK_GRAPH_SIGMA
} tk_graph_stage_t;

typedef struct tk_graph_thread_s tk_graph_thread_t;

typedef struct tk_graph_s {
  tm_pairs_t *pairs;
  // tm_pairs_t *type_mass;
  // tk_ivec_t *types;
  // tk_ivec_t *type_ranks;
  // int64_t type_sigma_k;
  tk_ivec_t *uids;
  tk_ivec_t *uids_hoods;
  tk_iumap_t *uids_idx;
  tk_iumap_t *uids_hoods_idx;
  tk_graph_adj_t *adj_pos, *adj_neg;
  tk_inv_t *inv;
  tk_ann_t *ann;
  tk_hbi_t *hbi;
  tk_inv_hoods_t *inv_hoods;
  tk_ann_hoods_t *ann_hoods;
  tk_hbi_hoods_t *hbi_hoods;
  uint64_t knn_cache;
  double knn_eps;
  double pos_default;
  double neg_default;
  double pos_scale;
  double neg_scale;
  double pos_sigma_scale;
  double neg_sigma_scale;
  int64_t sigma_k;
  double weight_eps;
  double bridge_density;
  bool no_label_is_match;
  tk_dvec_t *sigmas;
  tk_pvec_t *pos, *neg;
  tk_ivec_t *labels;
  uint64_t n_pos, n_neg;
  tk_dsu_t dsu;
  tk_graph_thread_t *threads;
  tk_threadpool_t *pool;
} tk_graph_t;

typedef struct tk_graph_thread_s {
  tk_graph_t *graph;
  tk_ivec_t *adj_offset;
  tk_ivec_t *adj_data;
  tk_dvec_t *adj_weights;
  tk_dvec_t *pos_sigma_slice;
  tk_dvec_t *neg_sigma_slice;
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
  tm_pair_t p = tm_pair(u, v, 0);
  khi = kh_get(pairs, graph->pairs, p);
  if (khi == kh_end(graph->pairs))
    return 0.0;
  return kh_key(graph->pairs, khi).w;
}

#endif
