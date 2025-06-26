#ifndef TK_TCH_H
#define TK_TCH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/dvec.h>

// TODO: parallelize
static inline void tk_tch_refine (
  lua_State *L,
  tk_ivec_t *codes,
  tk_dvec_t *scale,
  tk_graph_t *graph,
  uint64_t n_hidden,
  int i_each
) {
  uint64_t total_steps = 0;
  tk_graph_adj_t *adj_pos = graph->adj_pos;
  tk_graph_adj_t *adj_neg = graph->adj_neg;
  uint64_t n_nodes = graph->uids->n;

  // Prep nodes shuffle
  tk_pvec_t *node_order = tk_pvec_create(L, n_nodes, 0, 0);
  for (uint64_t i = 0; i < n_nodes; i ++)
    node_order->a[i] = (tk_pair_t) { (int64_t)  i, (int64_t) (tk_iuset_size(adj_pos->a[i]) + tk_iuset_size(adj_neg->a[i])) };
  tk_pvec_desc(node_order, 0, node_order->n);

  int *bitvecs = tk_malloc(L, n_hidden * n_nodes * sizeof(int));

  for (uint64_t i = 0; i < n_hidden * n_nodes; i ++)
    bitvecs[i] = -1;

  for (uint64_t i = 0; i < codes->n; i ++) {
    int64_t v = codes->a[i];
    if (v < 0)
      continue;
    uint64_t s = (uint64_t) v / n_hidden;
    uint64_t f = (uint64_t) v % n_hidden;
    bitvecs[f * n_nodes + s] = +1;
  }

  codes->n = 0;

  for (uint64_t f = 0; f < n_hidden; f ++) {
    int *bitvec = bitvecs + f * n_nodes;

    bool updated;
    uint64_t steps = 0;
    int64_t i, j;
    double delta;

    do {
      updated = false;
      steps ++;
      total_steps ++;

      for (uint64_t si = 0; si < n_nodes; si ++) {
        i = node_order->a[si].i;
        delta = 0.0;
        // Positive neighbors
        tk_iuset_foreach(adj_pos->a[i], j, ({
          delta += bitvec[i] * bitvec[j] * (scale == NULL ? 1.0 : scale->a[i] * scale->a[j]);
        }))
        // Negative neighbors
        tk_iuset_foreach(adj_neg->a[i], j, ({
          delta -= bitvec[i] * bitvec[j] * (scale == NULL ? 1.0 : scale->a[i] * scale->a[j]);
        }))
        // Check
        if (delta < 0){
          bitvec[i] = -bitvec[i];
          updated = true;
        }
      }

    } while (updated);

    // Write out the final bits into your packed codes
    for (uint64_t i = 0; i < n_nodes; i ++)
      if (bitvec[i] > 0)
        tk_ivec_push(codes, (int64_t) i * (int64_t) n_hidden + (int64_t) f);
  }

  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) total_steps);
    lua_call(L, 1, 0);
  }

  free(bitvecs);
  lua_pop(L, 1); // node_order

  tk_ivec_shrink(L, codes);
  tk_ivec_asc(codes, 0, codes->n);
}

#endif
