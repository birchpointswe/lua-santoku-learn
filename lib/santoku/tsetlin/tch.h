#ifndef TK_TCH_H
#define TK_TCH_H

#include <santoku/tsetlin/graph.h>
#include <santoku/dvec.h>
#include <santoku/cvec.h>
#include <santoku/ivec.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>

static inline void tk_tch_refine (
  lua_State *L,
  tk_cvec_t *codes,
  tk_ivec_t *uids,
  tk_ivec_t *adj_offset,
  tk_ivec_t *adj_data,
  tk_dvec_t *adj_weights,
  uint64_t n_dims,
  double eps,
  tk_dvec_t *rank_weights,
  uint64_t knn,
  int i_each
) {
  uint64_t n_nodes = uids->n;
  uint64_t bytes_per_sample = TK_CVEC_BITS_BYTES(n_dims);
  int *bitvecs = tk_malloc(L, n_dims * n_nodes * sizeof(int));
  for (uint64_t s = 0; s < n_nodes; s++) {
    for (uint64_t f = 0; f < n_dims; f++) {
      uint64_t byte_idx = s * bytes_per_sample + (f / 8);
      uint8_t bit_mask = 1 << (f % 8);
      bool bit_set = (codes->a[byte_idx] & bit_mask) != 0;
      bitvecs[f * n_nodes + s] = bit_set ? +1 : -1;
    }
  }
  tk_cvec_zero(codes);
  uint64_t total_steps = 0;
  uint64_t n_bytes = (n_dims + 7) / 8;
  double *eps_per_bit = NULL;
  if (eps >= 0.0) {
    eps_per_bit = tk_malloc(L, n_dims * sizeof(double));
    for (uint64_t f = 0; f < n_dims; f++) {
      eps_per_bit[f] = eps;
    }
  } else if (rank_weights != NULL && rank_weights->n > 0) {
    eps_per_bit = tk_malloc(L, n_dims * sizeof(double));
    double total_weight = 0.0;
    for (uint64_t r = 0; r < rank_weights->n; r++) {
      total_weight += rank_weights->a[r];
    }
    uint64_t bit_offset = 0;
    for (uint64_t r = 0; r < rank_weights->n; r++) {
      double rank_fraction = rank_weights->a[r] / total_weight;
      uint64_t bits_for_rank = (uint64_t)(rank_fraction * (double)n_dims);
      if (rank_weights->a[r] > 0.0 && bits_for_rank == 0) {
        bits_for_rank = 1;
      }
      double threshold;
      if (r < rank_weights->n - 1) {
        threshold = sqrt(rank_weights->a[r] * rank_weights->a[r + 1]);
      } else {
        threshold = -1.0;
      }
      for (uint64_t b = 0; b < bits_for_rank && bit_offset < n_dims; b++) {
        eps_per_bit[bit_offset] = threshold;
        bit_offset++;
      }
    }
    while (bit_offset < n_dims) {
      eps_per_bit[bit_offset] = -1.0;
      bit_offset++;
    }
  }
  #pragma omp parallel reduction(+:total_steps)
  {
    tk_ivec_t *node_order = tk_ivec_create(0, n_nodes, 0, 0);
    tk_ivec_fill_indices_serial(node_order);
    tk_fast_seed((unsigned int)omp_get_thread_num() ^ (unsigned int)time(NULL));
    #pragma omp for
    for (uint64_t byte_idx = 0; byte_idx < n_bytes; byte_idx++) {
      uint64_t first_bit = byte_idx * 8;
      uint64_t last_bit = ((byte_idx + 1) * 8 - 1) < (n_dims - 1) ? ((byte_idx + 1) * 8 - 1) : (n_dims - 1);
      for (uint64_t f = first_bit; f <= last_bit; f++) {
        int *bitvec = bitvecs + f * n_nodes;
        double eps_f = eps_per_bit ? eps_per_bit[f] : -1.0;
        tk_ivec_shuffle(node_order);
        bool updated;
        do {
          updated = false;
          total_steps++;
          for (uint64_t si = 0; si < n_nodes; si++) {
            int64_t i = node_order->a[si];
            int64_t row_start = adj_offset->a[i];
            int64_t row_end = adj_offset->a[i + 1];
            double delta = 0.0;
            int64_t neighbor_limit = row_end;
            if (knn > 0) {
              int64_t k_limit = row_start + (int64_t)knn;
              if (k_limit < row_end) {
                neighbor_limit = k_limit;
              }
            }
            for (int64_t jj = row_start; jj < neighbor_limit; jj++) {
              int64_t j = adj_data->a[jj];
              double weight = adj_weights->a[jj];
              if (eps_f < 0.0) {
                delta += bitvec[i] * bitvec[j] * weight;
              } else if (weight >= eps_f) {
                delta += bitvec[i] * bitvec[j] * weight;
              }
            }
            if (delta < 0.0) {
              bitvec[i] = -bitvec[i];
              updated = true;
            }
          }
        } while (updated);
      }
    }
    tk_ivec_destroy(node_order);
  }
  #pragma omp parallel for
  for (uint64_t s = 0; s < n_nodes; s++) {
    for (uint64_t f = 0; f < n_dims; f++) {
      int *bitvec = bitvecs + f * n_nodes;
      uint64_t byte_idx = s * bytes_per_sample + (f / 8);
      uint8_t bit_mask = 1 << (f % 8);
      if (bitvec[s] > 0) {
        codes->a[byte_idx] |= bit_mask;
      } else {
        codes->a[byte_idx] &= ~bit_mask;
      }
    }
  }
  if (eps_per_bit) {
    free(eps_per_bit);
  }
  free(bitvecs);
  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) total_steps);
    lua_call(L, 1, 0);
  }
}

#endif
