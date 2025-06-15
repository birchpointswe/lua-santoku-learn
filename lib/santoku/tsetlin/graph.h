#ifndef TK_GRAPH_H
#define TK_GRAPH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/ann.h>
#include <float.h>

#define TK_GRAPH_MT "tk_graph_t"
#define TK_GRAPH_EPH "tk_graph_eph"

KHASH_INIT(tk_dsu_members, int64_t, tk_iuset_t *, 1, kh_int64_hash_func, kh_int64_hash_equal)
typedef khash_t(tk_dsu_members) tk_dsu_members_t;

typedef tk_iuset_t * tk_graph_adj_item_t;
#define tk_vec_name tk_graph_adj
#define tk_vec_base tk_graph_adj_item_t
#define tk_vec_destroy_item(...) tk_iuset_destroy(__VA_ARGS__)
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef enum {
  TK_GRAPH_TODO
} tk_graph_stage_t;

typedef struct {
  tk_iumap_t *node_component;
  tk_iumap_t *node_parent;
  tk_iumap_t *component_rank;
  tk_dsu_members_t *component_members;
} tk_dsu_t;

typedef struct tk_graph_thread_s tk_graph_thread_t;

typedef struct tk_graph_s {
  tm_pairs_t *pairs;
  tk_ivec_t *uids;
  tk_iumap_t *uid_hood;
  tk_graph_adj_t *adj_pos, *adj_neg;
  struct {
    bool is_inv;
    union {
      tk_inv_t *inv;
      tk_ann_t *ann;
    };
    union {
      tk_inv_hoods_t *inv_hoods;
      tk_ann_hoods_t *ann_hoods;
    };
  } index;
  uint64_t knn_cache;
  double knn_eps;
  tk_ivec_t *pos, *neg;
  uint64_t n_pos, n_neg;
  tk_dsu_t dsu;
  tk_graph_thread_t *threads;
  tk_threadpool_t *pool;
} tk_graph_t;

typedef struct tk_graph_thread_s {
  tk_graph_t *graph;
} tk_graph_thread_t;

static inline tk_graph_t *tk_graph_peek (lua_State *L, int i)
{
  return (tk_graph_t *) luaL_checkudata(L, i, TK_GRAPH_MT);
}

#endif
