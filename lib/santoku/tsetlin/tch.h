#ifndef TK_TCH_H
#define TK_TCH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/roaring.h>
#include <santoku/dvec.h>

// TODO: parallelize
static inline void tk_tch_refine (
  lua_State *L,
  tk_ivec_t *codes,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  uint64_t n_nodes,
  uint64_t n_hidden,
  int i_each
) {
  uint64_t total_steps = 0;

  // Prep hidden shuffle (outside parallel region)
  int64_t *shuf_hidden = tk_malloc(L, n_hidden * sizeof(int64_t));
  for (int64_t i = 0; i < (int64_t) n_hidden; i ++)
    shuf_hidden[i] = i;
  ks_shuffle(i64, n_hidden, shuf_hidden);

  // Prep nodes shuffle
  int64_t *shuf_nodes = tk_malloc(L, n_nodes * sizeof(int64_t));
  for (int64_t i = 0; i < (int64_t) n_nodes; i ++)
    shuf_nodes[i] = i;

  int *bitvecs = tk_malloc(L, n_hidden * n_nodes * sizeof(int));
  roaring64_iterator_t it;

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

    uint64_t f = (uint64_t) shuf_hidden[sf];
    int *bitvec = bitvecs + f * n_nodes;

    // Shuffle nodes
    ks_shuffle(i64, n_nodes, shuf_nodes);

    bool updated;
    uint64_t steps = 0;

    do {
      updated = false;
      steps ++;
      total_steps ++;

      for (uint64_t si = 0; si < n_nodes; si ++) {
        uint64_t i = (uint64_t) shuf_nodes[si];
        int delta = 0;
        // Positive neighbors
        roaring64_iterator_reinit(adj_pos[i], &it);
        while (roaring64_iterator_has_value(&it)) {
          uint64_t j = roaring64_iterator_value(&it);
          roaring64_iterator_advance(&it);
          delta += bitvec[i] * bitvec[j];
        }
        // Negative neighbors
        roaring64_iterator_reinit(adj_neg[i], &it);
        while (roaring64_iterator_has_value(&it)) {
          uint64_t j = roaring64_iterator_value(&it);
          roaring64_iterator_advance(&it);
          delta -= bitvec[i] * bitvec[j];
        }
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
  free(shuf_nodes);
  free(shuf_hidden);

  tk_ivec_shrink(L, codes);
  tk_ivec_asc(codes, 0, codes->n);
}

#endif
