#ifndef TK_TCH_H
#define TK_TCH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/threads.h>
#include <santoku/dvec.h>
#include <santoku/cvec.h>
#include <santoku/ivec.h>
#include <stdbool.h>

typedef enum {
  TK_TCH_OPTIMIZE_BITS,
  TK_TCH_CONVERT_TO_BINARY,
} tk_tch_stage_t;

typedef struct {
  tk_cvec_t *codes;
  tk_ivec_t *uids;
  tk_ivec_t *adj_offset;
  tk_ivec_t *adj_data;
  tk_dvec_t *adj_weights;
  tk_dvec_t *scale;
  uint64_t n_dims;
  uint64_t n_nodes;
  int *bitvecs;
  tk_threadpool_t *pool;
  int i_each;
  lua_State *L;
} tk_tch_state_t;

typedef struct {
  tk_tch_state_t *state;
  uint64_t byte_first, byte_last;
  uint64_t first, last;
  unsigned int index;
  uint64_t steps;
} tk_tch_thread_t;

static inline void tk_tch_worker(void *dp, int sig)
{
  tk_tch_stage_t stage = (tk_tch_stage_t) sig;
  tk_tch_thread_t *data = (tk_tch_thread_t *) dp;
  tk_tch_state_t *state = data->state;

  switch (stage) {

    case TK_TCH_OPTIMIZE_BITS: {
      uint64_t n_nodes = state->n_nodes;
      uint64_t n_dims = state->n_dims;

      tk_ivec_t *node_order = tk_ivec_create(0, n_nodes, 0, 0);
      tk_ivec_fill_indices(node_order);
      tk_fast_seed((unsigned int) pthread_self());

      for (uint64_t byte_idx = data->byte_first; byte_idx <= data->byte_last; byte_idx++) {
        uint64_t first_bit = byte_idx * 8;
        uint64_t last_bit = ((byte_idx + 1) * 8 - 1) < (n_dims - 1) ? ((byte_idx + 1) * 8 - 1) : (n_dims - 1);

        for (uint64_t f = first_bit; f <= last_bit; f++) {
          int *bitvec = state->bitvecs + f * n_nodes;
          tk_ivec_shuffle(node_order);

          bool updated;
          do {
            updated = false;
            data->steps++;

            for (uint64_t si = 0; si < n_nodes; si++) {
              int64_t i = node_order->a[si];
              int64_t row_start = state->adj_offset->a[i];
              int64_t row_end = state->adj_offset->a[i + 1];
              double delta = 0.0;

              for (int64_t jj = row_start; jj < row_end; jj++) {
                int64_t j = state->adj_data->a[jj];
                double weight = state->adj_weights->a[jj];
                if (state->scale != NULL) {
                  weight *= state->scale->a[i] * state->scale->a[j];
                }
                delta += bitvec[i] * bitvec[j] * weight;
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
      break;
    }

    case TK_TCH_CONVERT_TO_BINARY: {
      uint64_t n_dims = state->n_dims;
      uint64_t bytes_per_sample = TK_CVEC_BITS_BYTES(n_dims);

      for (uint64_t s = data->first; s <= data->last; s++) {
        for (uint64_t f = 0; f < n_dims; f++) {
          int *bitvec = state->bitvecs + f * state->n_nodes;
          uint64_t byte_idx = s * bytes_per_sample + (f / 8);
          uint8_t bit_mask = 1 << (f % 8);

          if (bitvec[s] > 0) {
            state->codes->a[byte_idx] |= bit_mask;
          } else {
            state->codes->a[byte_idx] &= ~bit_mask;
          }
        }
      }
      break;
    }
  }
}

static inline void tk_tch_refine (
  lua_State *L,
  tk_cvec_t *codes,
  tk_ivec_t *uids,
  tk_ivec_t *adj_offset,
  tk_ivec_t *adj_data,
  tk_dvec_t *adj_weights,
  tk_dvec_t *scale,
  uint64_t n_dims,
  unsigned int n_threads,
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
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_tch_worker);
  tk_tch_state_t state = {
    .codes = codes,
    .uids = uids,
    .adj_offset = adj_offset,
    .adj_data = adj_data,
    .adj_weights = adj_weights,
    .scale = scale,
    .n_dims = n_dims,
    .n_nodes = n_nodes,
    .bitvecs = bitvecs,
    .pool = pool,
    .i_each = i_each,
    .L = L
  };

  tk_tch_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_tch_thread_t));
  for (unsigned int i = 0; i < n_threads; i++) {
    threads[i].state = &state;
    threads[i].index = i;
    threads[i].steps = 0;
    pool->threads[i].data = &threads[i];
  }

  // Each thread gets complete bytes to avoid race conditions
  uint64_t n_bytes = (n_dims + 7) / 8;
  for (unsigned int i = 0; i < n_threads; i++) {
    tk_thread_range(i, n_threads, n_bytes, &threads[i].byte_first, &threads[i].byte_last);
  }
  tk_threads_signal(pool, TK_TCH_OPTIMIZE_BITS, 0);

  for (unsigned int i = 0; i < n_threads; i++) {
    tk_thread_range(i, n_threads, n_nodes, &threads[i].first, &threads[i].last);
  }
  tk_threads_signal(pool, TK_TCH_CONVERT_TO_BINARY, 0);

  if (i_each >= 0) {
    uint64_t total_steps = 0;
    for (unsigned int i = 0; i < n_threads; i++) {
      total_steps += threads[i].steps;
    }
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t) total_steps);
    lua_call(L, 1, 0);
  }

  free(bitvecs);
  free(threads);
  tk_threads_destroy(pool);
}

#endif
