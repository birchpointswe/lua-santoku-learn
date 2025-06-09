#ifndef TK_TCH_H
#define TK_TCH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/dvec.h>

// TODO: parallelize
static inline void tk_tch_refine (
  lua_State *L,
  tk_ivec_t *codes,
  tk_graph_adj_t *adj_pos,
  tk_graph_adj_t *adj_neg,
  uint64_t n_nodes,
  uint64_t n_hidden,
  int i_each
) {
  uint64_t total_steps = 0;

  // Prep hidden shuffle (outside parallel region)
  tk_ivec_t *shuf_hidden = tk_ivec_create(L, n_hidden, 0, 0);
  tk_ivec_fill_indices(shuf_hidden);
  tk_ivec_shuffle(shuf_hidden);

  // Prep nodes shuffle
  tk_ivec_t *shuf_nodes = tk_ivec_create(L, n_nodes, 0, 0);
  tk_ivec_fill_indices(shuf_nodes);
  tk_ivec_shuffle(shuf_nodes);

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

  for (uint64_t sf = 0; sf < n_hidden; sf ++) {

    uint64_t f = (uint64_t) shuf_hidden->a[sf];
    int *bitvec = bitvecs + f * n_nodes;

    // Shuffle nodes
    tk_ivec_shuffle(shuf_nodes);

    bool updated;
    uint64_t steps = 0;
    int64_t i, j, delta;

    do {
      updated = false;
      steps ++;
      total_steps ++;

      for (uint64_t si = 0; si < n_nodes; si ++) {
        i = shuf_nodes->a[si];
        delta = 0;
        // Positive neighbors
        tk_iuset_foreach(adj_pos->a[i], j, ({
          delta += bitvec[i] * bitvec[j];
        }))
        // Negative neighbors
        tk_iuset_foreach(adj_neg->a[i], j, ({
          delta -= bitvec[i] * bitvec[j];
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
  lua_pop(L, 2); // shuf_hidden, shuf_nodes

  tk_ivec_shrink(L, codes);
  tk_ivec_asc(codes, 0, codes->n);
}

#endif
