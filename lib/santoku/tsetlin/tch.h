#ifndef TK_TCH_H
#define TK_TCH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/dvec.h>
#include <santoku/cvec.h>

static inline void tk_tch_refine (

  lua_State *L,
  tk_cvec_t *codes,

  tk_ivec_t *uids,
  tk_ivec_t *adj_offset,
  tk_ivec_t *adj_data,
  tk_dvec_t *adj_weights,
  uint64_t n_dims,

  int i_each

) {
  uint64_t total_steps = 0;

  // Prep node order shuffle
  tk_ivec_t *node_order = tk_ivec_create(L, uids->n, 0, 0);
  tk_ivec_fill_indices(node_order);

  int *bitvecs = tk_malloc(L, n_dims * uids->n * sizeof(int));

  // Convert dense binary codes (cvec) to TCH format (-1/+1)
  uint64_t bytes_per_sample = TK_CVEC_BITS_BYTES(n_dims);
  for (uint64_t s = 0; s < uids->n; s++) {
    for (uint64_t f = 0; f < n_dims; f++) {
      uint64_t byte_idx = s * bytes_per_sample + (f / 8);
      uint8_t bit_mask = 1 << (f % 8);
      bool bit_set = (codes->a[byte_idx] & bit_mask) != 0;
      bitvecs[f * uids->n + s] = bit_set ? +1 : -1;
    }
  }

  tk_cvec_zero(codes);

  for (uint64_t f = 0; f < n_dims; f ++) {
    int *bitvec = bitvecs + f * uids->n;

    bool updated;
    uint64_t steps = 0;

    tk_ivec_shuffle(node_order);

    do {
      updated = false;
      steps ++;
      total_steps ++;
      for (uint64_t si = 0; si < uids->n; si ++) {
        int64_t i = node_order->a[si];
        int64_t row_start = adj_offset->a[i];
        int64_t row_end = adj_offset->a[i + 1];
        double delta = 0.0;
        for (int64_t jj = row_start; jj < row_end; jj++) {
          int64_t j = adj_data->a[jj];
          delta += bitvec[i] * bitvec[j] * adj_weights->a[jj];
        }
        if (delta < 0.0){
          bitvec[i] = -bitvec[i];
          updated = true;
        }
      }
    } while (updated);

    // Convert back to dense binary codes
    for (uint64_t i = 0; i < uids->n; i ++) {
      uint64_t byte_idx = i * bytes_per_sample + (f / 8);
      uint8_t bit_mask = 1 << (f % 8);
      if (bitvec[i] > 0) {
        codes->a[byte_idx] |= bit_mask;  // Set bit
      } else {
        codes->a[byte_idx] &= ~bit_mask;  // Clear bit
      }
    }
  }

  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) total_steps);
    lua_call(L, 1, 0);
  }

  free(bitvecs);
  lua_pop(L, 1); // node_order
}

#endif
