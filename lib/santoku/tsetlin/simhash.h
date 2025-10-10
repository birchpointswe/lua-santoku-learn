#ifndef TK_SIMHASH_H
#define TK_SIMHASH_H

#include <santoku/tsetlin/conf.h>
#include <santoku/cvec.h>

typedef enum {
  TK_SIMHASH_PACKED,
  TK_SIMHASH_INTEGER
} tk_simhash_mode_t;

typedef struct {
  tk_simhash_mode_t mode;
  uint64_t n_items;
  uint64_t n_chunks;
  uint64_t state_bits;
  tk_bits_t *counts;
  uint8_t *counts_int;
  tk_bits_t *actions;
  tk_bits_t tail_mask;
} tk_simhash_t;

static inline tk_bits_t *tk_simhash_counts (
  tk_simhash_t *sh,
  uint64_t item,
  uint64_t chunk
) {
  return &sh->counts[item * sh->n_chunks * (sh->state_bits - 1) +
                     chunk * (sh->state_bits - 1)];
}

static inline tk_bits_t *tk_simhash_actions (
  tk_simhash_t *sh,
  uint64_t item
) {
  return &sh->actions[item * sh->n_chunks];
}

static inline uint8_t *tk_simhash_counts_int (
  tk_simhash_t *sh,
  uint64_t item,
  uint64_t chunk
) {
  return &sh->counts_int[(item * sh->n_chunks + chunk) * TK_CVEC_BITS];
}

static inline void tk_simhash_inc (
  tk_simhash_t *sh,
  uint64_t item,
  uint64_t chunk,
  tk_bits_t active
) {
  if (!active) return;
  if (sh->mode == TK_SIMHASH_INTEGER) {
    uint8_t *counts = tk_simhash_counts_int(sh, item, chunk);
    tk_bits_t *actions = tk_simhash_actions(sh, item);
    uint64_t threshold = 1 << (sh->state_bits - 2);
    uint64_t max_count = (1 << sh->state_bits) - 1;
    tk_bits_t mask = (chunk == sh->n_chunks - 1) ? sh->tail_mask : (tk_bits_t)~0;
    for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
      if (active & (1 << b)) {
        if (counts[b] < max_count) {
          counts[b]++;
          if (counts[b] == threshold)
            actions[chunk] = (actions[chunk] | (1 << b)) & mask;
        }
      }
    }
  } else {
    uint64_t m = sh->state_bits - 1;
    tk_bits_t *counts = tk_simhash_counts(sh, item, chunk);
    tk_bits_t carry, carry_next;
    carry = active;
    for (uint64_t b = 0; b < m; b ++) {
      carry_next = counts[b] & carry;
      counts[b] ^= carry;
      carry = carry_next;
    }
    tk_bits_t *actions = tk_simhash_actions(sh, item);
    tk_bits_t carry_masked = carry & ((chunk == sh->n_chunks - 1) ? sh->tail_mask : (tk_bits_t)~0);
    carry_next = actions[chunk] & carry_masked;
    actions[chunk] ^= carry_masked;
    carry = carry_next;
    for (uint64_t b = 0; b < m; b ++)
      counts[b] |= carry_masked;
    actions[chunk] |= carry_masked;
  }
}

static inline void tk_simhash_dec (
  tk_simhash_t *sh,
  uint64_t item,
  uint64_t chunk,
  tk_bits_t active
) {
  if (!active) return;
  if (sh->mode == TK_SIMHASH_INTEGER) {
    uint8_t *counts = tk_simhash_counts_int(sh, item, chunk);
    tk_bits_t *actions = tk_simhash_actions(sh, item);
    uint64_t threshold = 1 << (sh->state_bits - 2);
    tk_bits_t mask = (chunk == sh->n_chunks - 1) ? sh->tail_mask : (tk_bits_t)~0;
    for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
      if (active & (1 << b)) {
        if (counts[b] > 0) {
          counts[b]--;
          if (counts[b] < threshold)
            actions[chunk] &= ~(1 << b);
        }
      }
    }
    actions[chunk] &= mask;
  } else {
    uint64_t m = sh->state_bits - 1;
    tk_bits_t *counts = tk_simhash_counts(sh, item, chunk);
    tk_bits_t carry, carry_next;
    carry = active;
    for (uint64_t b = 0; b < m; b ++) {
      carry_next = (~counts[b]) & carry;
      counts[b] ^= carry;
      carry = carry_next;
    }
    tk_bits_t *actions = tk_simhash_actions(sh, item);
    tk_bits_t carry_masked = carry & ((chunk == sh->n_chunks - 1) ? sh->tail_mask : (tk_bits_t) ~0);
    carry_next = (~actions[chunk]) & carry_masked;
    actions[chunk] ^= carry_masked;
    carry = carry_next;
    for (uint64_t b = 0; b < m; b ++)
      counts[b] &= ~carry_masked;
    actions[chunk] &= ~carry_masked;
  }
}

static inline int tk_simhash_init (
  lua_State *L,
  tk_simhash_t *sh,
  uint64_t n_items,
  uint64_t n_chunks,
  uint64_t state_bits,
  tk_bits_t tail_mask,
  tk_simhash_mode_t mode
) {
  sh->mode = mode;
  sh->n_items = n_items;
  sh->n_chunks = n_chunks;
  sh->state_bits = state_bits;
  sh->tail_mask = tail_mask;
  sh->counts = NULL;
  sh->counts_int = NULL;
  uint64_t action_chunks = n_items * n_chunks;
  sh->actions = tk_malloc_aligned(L, sizeof(tk_bits_t) * action_chunks, TK_CVEC_BITS);
  if (!sh->actions) {
    return -1;
  }
  if (mode == TK_SIMHASH_INTEGER) {
    uint64_t int_count = n_items * n_chunks * TK_CVEC_BITS;
    sh->counts_int = tk_malloc_aligned(L, sizeof(uint8_t) * int_count, TK_CVEC_BITS);
    if (!sh->counts_int) {
      free(sh->actions);
      sh->actions = NULL;
      return -1;
    }
  } else {
    uint64_t state_chunks = n_items * n_chunks * (state_bits - 1);
    sh->counts = tk_malloc_aligned(L, sizeof(tk_bits_t) * state_chunks, TK_CVEC_BITS);
    if (!sh->counts) {
      free(sh->actions);
      sh->actions = NULL;
      return -1;
    }
  }
  return 0;
}

static inline void tk_simhash_destroy (tk_simhash_t *sh) {
  if (sh->counts) {
    free(sh->counts);
    sh->counts = NULL;
  }
  if (sh->counts_int) {
    free(sh->counts_int);
    sh->counts_int = NULL;
  }
  if (sh->actions) {
    free(sh->actions);
    sh->actions = NULL;
  }
}

static inline void tk_simhash_setup_initial_state (
  tk_simhash_t *sh,
  uint64_t item_first,
  uint64_t item_last
) {
  uint64_t m = sh->state_bits - 1;
  uint64_t include_threshold = 1 << (m - 1);
  uint64_t initial_state = include_threshold - 1;

  if (sh->mode == TK_SIMHASH_INTEGER) {
    for (uint64_t item = item_first; item <= item_last; item++) {
      for (uint64_t chunk = 0; chunk < sh->n_chunks; chunk++) {
        tk_bits_t *actions = tk_simhash_actions(sh, item);
        uint8_t *counts = tk_simhash_counts_int(sh, item, chunk);
        actions[chunk] = 0;
        for (uint64_t b = 0; b < TK_CVEC_BITS; b++)
          counts[b] = initial_state;
      }
    }
  } else {
    for (uint64_t item = item_first; item <= item_last; item++) {
      for (uint64_t chunk = 0; chunk < sh->n_chunks; chunk++) {
        tk_bits_t *actions = tk_simhash_actions(sh, item);
        tk_bits_t *counts = tk_simhash_counts(sh, item, chunk);
        actions[chunk] = 0;
        for (uint64_t b = 0; b < m; b++) {
          if (initial_state & (1 << b))
            counts[b] = TK_CVEC_ALL_MASK;
          else
            counts[b] = TK_CVEC_ZERO_MASK;
        }
      }
    }
  }
}

static inline uint64_t tk_simhash_read_count_packed (
  tk_simhash_t *sh,
  uint64_t item,
  uint64_t chunk,
  uint64_t bit
) {
  tk_bits_t *counts = tk_simhash_counts(sh, item, chunk);
  tk_bits_t *actions = tk_simhash_actions(sh, item);
  uint64_t m = sh->state_bits - 1;
  uint64_t count = 0;
  for (uint64_t b = 0; b < m; b++) {
    if (counts[b] & (1 << bit))
      count |= (1 << b);
  }
  if (actions[chunk] & (1 << bit))
    count |= (1 << m);
  return count;
}

static inline void tk_simhash_merge (
  tk_simhash_t *sh1,
  tk_simhash_t *sh2
) {
  assert(sh1->state_bits == sh2->state_bits);
  assert(sh1->n_items == sh2->n_items);
  assert(sh1->n_chunks == sh2->n_chunks);

  uint64_t threshold = 1 << (sh1->state_bits - 2);
  uint64_t max_count = (1 << sh1->state_bits) - 1;

  if (sh1->mode == TK_SIMHASH_INTEGER && sh2->mode == TK_SIMHASH_INTEGER) {
    // INTEGER + INTEGER
    for (uint64_t item = 0; item < sh1->n_items; item++) {
      tk_bits_t *actions1 = tk_simhash_actions(sh1, item);
      for (uint64_t chunk = 0; chunk < sh1->n_chunks; chunk++) {
        uint8_t *counts1 = tk_simhash_counts_int(sh1, item, chunk);
        uint8_t *counts2 = tk_simhash_counts_int(sh2, item, chunk);
        tk_bits_t mask = (chunk == sh1->n_chunks - 1) ? sh1->tail_mask : (tk_bits_t)~0;
        tk_bits_t new_actions = 0;
        for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
          uint16_t sum = (uint16_t)counts1[b] + (uint16_t)counts2[b];
          counts1[b] = (sum > max_count) ? max_count : sum;
          if (counts1[b] >= threshold)
            new_actions |= (1 << b);
        }
        actions1[chunk] = new_actions & mask;
      }
    }
  } else if (sh1->mode == TK_SIMHASH_INTEGER && sh2->mode == TK_SIMHASH_PACKED) {
    // INTEGER + PACKED: extract sh2 counts and add to sh1
    for (uint64_t item = 0; item < sh1->n_items; item++) {
      tk_bits_t *actions1 = tk_simhash_actions(sh1, item);
      for (uint64_t chunk = 0; chunk < sh1->n_chunks; chunk++) {
        uint8_t *counts1 = tk_simhash_counts_int(sh1, item, chunk);
        tk_bits_t mask = (chunk == sh1->n_chunks - 1) ? sh1->tail_mask : (tk_bits_t)~0;
        tk_bits_t new_actions = 0;
        for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
          uint64_t count2 = tk_simhash_read_count_packed(sh2, item, chunk, b);
          uint16_t sum = (uint16_t)counts1[b] + (uint16_t)count2;
          counts1[b] = (sum > max_count) ? max_count : sum;
          if (counts1[b] >= threshold)
            new_actions |= (1 << b);
        }
        actions1[chunk] = new_actions & mask;
      }
    }
  } else if (sh1->mode == TK_SIMHASH_PACKED && sh2->mode == TK_SIMHASH_INTEGER) {
    // PACKED + INTEGER: extract sh2 counts and add to sh1 using inc
    for (uint64_t item = 0; item < sh1->n_items; item++) {
      for (uint64_t chunk = 0; chunk < sh1->n_chunks; chunk++) {
        uint8_t *counts2 = tk_simhash_counts_int(sh2, item, chunk);
        for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
          uint64_t count2 = counts2[b];
          // Increment sh1 'count2' times for this bit
          for (uint64_t c = 0; c < count2; c++) {
            tk_simhash_inc(sh1, item, chunk, 1 << b);
          }
        }
      }
    }
  } else {
    // PACKED + PACKED
    uint64_t m = sh1->state_bits - 1;
    for (uint64_t item = 0; item < sh1->n_items; item++) {
      for (uint64_t chunk = 0; chunk < sh1->n_chunks; chunk++) {
        tk_bits_t *counts1 = tk_simhash_counts(sh1, item, chunk);
        tk_bits_t *counts2 = tk_simhash_counts(sh2, item, chunk);
        tk_bits_t *actions1 = tk_simhash_actions(sh1, item);
        tk_bits_t *actions2 = tk_simhash_actions(sh2, item);
        tk_bits_t carry = 0;
        for (uint64_t b = 0; b < m; b++) {
          tk_bits_t sum = counts1[b] ^ counts2[b] ^ carry;
          carry = (counts1[b] & counts2[b]) | ((counts1[b] ^ counts2[b]) & carry);
          counts1[b] = sum;
        }
        tk_bits_t mask = (chunk == sh1->n_chunks - 1) ? sh1->tail_mask : (tk_bits_t)~0;
        carry = carry & mask;
        tk_bits_t a1 = actions1[chunk] & mask;
        tk_bits_t a2 = actions2[chunk] & mask;
        tk_bits_t sum_action = a1 ^ a2 ^ carry;
        tk_bits_t carry_out = ((a1 & a2) | ((a1 ^ a2) & carry)) & mask;
        actions1[chunk] = (actions1[chunk] & ~mask) | sum_action;
        for (uint64_t b = 0; b < m; b++)
          counts1[b] |= carry_out;
        actions1[chunk] |= carry_out;
      }
    }
  }
}

#endif
