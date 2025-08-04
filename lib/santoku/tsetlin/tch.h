#ifndef TK_TCH_H
#define TK_TCH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/dvec.h>

static inline void tk_tch_refine (

  lua_State *L,
  tk_ivec_t *codes,

  tk_graph_t *graph,
  uint64_t n_dims,

  double neg_scale,

  int i_each

) {
  uint64_t total_steps = 0;
  tk_graph_adj_t *adj_pos = graph->adj_pos;
  tk_graph_adj_t *adj_neg = graph->adj_neg;
  uint64_t n_nodes = graph->uids->n;

  // Prep node order shuffle
  tk_ivec_t *node_order = tk_ivec_create(L, n_nodes, 0, 0);
  tk_ivec_fill_indices(node_order);

  int *bitvecs = tk_malloc(L, n_dims * n_nodes * sizeof(int));

  for (uint64_t i = 0; i < n_dims * n_nodes; i ++)
    bitvecs[i] = -1;

  for (uint64_t i = 0; i < codes->n; i ++) {
    int64_t v = codes->a[i];
    if (v < 0)
      continue;
    uint64_t s = (uint64_t) v / n_dims;
    uint64_t f = (uint64_t) v % n_dims;
    bitvecs[f * n_nodes + s] = +1;
  }

  codes->n = 0;

  for (uint64_t f = 0; f < n_dims; f ++) {
    int *bitvec = bitvecs + f * n_nodes;

    bool updated;
    uint64_t steps = 0;
    int64_t i, j;

    tk_ivec_shuffle(node_order);

    do {
      updated = false;
      steps ++;
      total_steps ++;

      for (uint64_t si = 0; si < n_nodes; si ++) {
        i = node_order->a[si];
        double delta = 0.0;
        // Positive neighbors
        tk_iuset_foreach(adj_pos->a[i], j, ({
          delta += bitvec[i] * bitvec[j];
        }))
        // Negative neighbors
        tk_iuset_foreach(adj_neg->a[i], j, ({
          delta += bitvec[i] * bitvec[j] * neg_scale;
        }))
        // Check
        if (delta < 0.0){
          bitvec[i] = -bitvec[i];
          updated = true;
        }
      }

    } while (updated);

    for (uint64_t i = 0; i < n_nodes; i ++)
      if (bitvec[i] > 0)
        tk_ivec_push(codes, (int64_t) i * (int64_t) n_dims + (int64_t) f);
  }

  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) total_steps);
    lua_call(L, 1, 0);
  }

  free(bitvecs);
  lua_pop(L, 2); // node_order

  tk_ivec_shrink(codes);
  tk_ivec_asc(codes, 0, codes->n);
}

#endif
