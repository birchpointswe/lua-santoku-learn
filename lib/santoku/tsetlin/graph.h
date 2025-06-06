#ifndef TK_GRAPH_H
#define TK_GRAPH_H

#include <santoku/tsetlin/roaring.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>
#include <float.h>

KHASH_SET_INIT_INT64(i64)
typedef khash_t(i64) i64_hash_t;

#define TK_GRAPH_MT "santoku_tsetlin_graph"

typedef enum {
  TK_GRAPH_KNN
} tk_graph_stage_t;

typedef struct {
  uint64_t n_original;
  uint64_t n_components;
  int64_t *components;
  int64_t *parent;
  int64_t *rank;
  int64_t **members;
  uint64_t *count;
  uint64_t *cap;
  roaring64_bitmap_t *unseen;
} tm_dsu_t;

typedef struct tk_graph_thread_s tk_graph_thread_t;

typedef struct tk_graph_s {
  double *x, *y, *z;
  double *evals, *evecs, *resNorms;
  double *laplacian;
  uint64_t n_nodes, n_hidden, n_features;
  tm_pairs_t *pairs;
  tm_neighbors_t *neighbors;
  tm_neighbors_t *fneighbors;
  roaring64_bitmap_t **nodes;
  tk_ivec_t *set_bits;
  uint64_t knn_cache;
  tk_ivec_t *pos, *neg;
  uint64_t n_pos, n_neg;
  roaring64_bitmap_t **adj_pos, **adj_neg, **index;
  tm_dsu_t dsu;
  tk_graph_thread_t *threads;
  tk_threadpool_t *pool;
} tk_graph_t;

typedef struct tk_graph_thread_s {
  tk_graph_t *graph;
  roaring64_bitmap_t *candidates;
  roaring64_bitmap_t *seen;
  uint64_t ufirst, ulast;
} tk_graph_thread_t;

static inline tk_graph_t *tk_graph_peek (lua_State *L, int i)
{
  return (tk_graph_t *) luaL_checkudata(L, i, TK_GRAPH_MT);
}

#endif
