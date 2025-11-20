#ifndef TK_AUTOMATA_H
#define TK_AUTOMATA_H

#include <omp.h>
#include <santoku/cvec.h>

typedef struct {
  uint64_t n_clauses;
  uint64_t n_chunks;
  uint64_t state_bits;
  char *counts;
  char *actions;
  uint8_t tail_mask;
} tk_automata_t;

static inline char *tk_automata_actions (
  tk_automata_t *aut,
  uint64_t clause
) {
  return &aut->actions[clause * aut->n_chunks];
}

static inline char *tk_automata_counts_plane (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t plane
) {
  uint64_t plane_global_index = clause * (aut->state_bits - 1) + plane;
  return &aut->counts[plane_global_index * aut->n_chunks];
}

static inline int tk_automata_init (
  lua_State *L,
  tk_automata_t *aut,
  uint64_t n_clauses,
  uint64_t n_chunks,
  uint64_t state_bits,
  uint8_t tail_mask
) {
  aut->n_clauses = n_clauses;
  aut->n_chunks = n_chunks;
  aut->state_bits = state_bits;
  aut->tail_mask = tail_mask;
  uint64_t action_chunks = n_clauses * n_chunks;
  aut->actions = (char *)tk_malloc_aligned(L, action_chunks, TK_CVEC_BITS);
  if (!aut->actions)
    return -1;
  uint64_t state_chunks = n_clauses * n_chunks * (state_bits - 1);
  aut->counts = (char *)tk_malloc_aligned(L, state_chunks, TK_CVEC_BITS);
  if (!aut->counts) {
    free(aut->actions);
    aut->actions = NULL;
    return -1;
  }
  return 0;
}

static inline void tk_automata_destroy (tk_automata_t *aut) {
  if (aut->counts) {
    free(aut->counts);
    aut->counts = NULL;
  }
  if (aut->actions) {
    free(aut->actions);
    aut->actions = NULL;
  }
}

static inline void tk_automata_setup (
  tk_automata_t *aut,
  uint64_t clause_first,
  uint64_t clause_last
) {
  uint64_t m = aut->state_bits - 1;
  #pragma omp parallel for
  for (uint64_t clause = clause_first; clause <= clause_last; clause++) {
    for (uint64_t b = 0; b < m; b++) {
      uint8_t *counts_plane = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
      memset(counts_plane, 0xFF, aut->n_chunks);
    }
    uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);
    memset(actions, 0, aut->n_chunks);
  }
}

static inline void tk_automata_inc_byte (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk,
  uint8_t active
) {
  if (!active) return;
  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);
  uint8_t mask = (chunk == aut->n_chunks - 1) ? aut->tail_mask : 0xFF;

  uint8_t carry = active;
  for (uint64_t b = 0; b < m; b++) {
    uint8_t *counts_b_plane = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
    uint8_t carry_next = counts_b_plane[chunk] & carry;
    counts_b_plane[chunk] ^= carry;
    carry = carry_next;
  }
  uint8_t carry_masked = carry & mask;
  uint8_t carry_next = actions[chunk] & carry_masked;
  actions[chunk] ^= carry_masked;
  carry = carry_next;

  for (uint64_t b = 0; b < m; b++) {
    uint8_t *counts_b_plane = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
    counts_b_plane[chunk] |= carry;
  }
  actions[chunk] |= carry;
}

static inline void tk_automata_dec_byte (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk,
  uint8_t active
) {
  if (!active) return;

  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);
  uint8_t mask = (chunk == aut->n_chunks - 1) ? aut->tail_mask : 0xFF;

  uint8_t borrow = active;
  for (uint64_t b = 0; b < m; b++) {
    uint8_t *counts_b_plane = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
    uint8_t borrow_next = (~counts_b_plane[chunk]) & borrow;
    counts_b_plane[chunk] ^= borrow;
    borrow = borrow_next;
  }
  uint8_t borrow_masked = borrow & mask;
  uint8_t borrow_next = (~actions[chunk]) & borrow_masked;
  actions[chunk] ^= borrow_masked;
  borrow = borrow_next;
  for (uint64_t b = 0; b < m; b++) {
    uint8_t *counts_b_plane = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
    counts_b_plane[chunk] &= ~borrow;
  }
  actions[chunk] &= ~borrow;
}
static inline void tk_automata_inc (
  tk_automata_t *aut,
  uint64_t clause,
  const uint8_t *input,
  uint64_t n_chunks
) {
  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);

  uint8_t *counts_planes[m];
  for (uint64_t b = 0; b < m; b++) {
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
  }

  uint64_t k = 0;

#ifdef __SIZEOF_INT128__
  for (; k + 15 < n_chunks; k += 16) {
    __uint128_t inp_val;
    memcpy(&inp_val, &input[k], sizeof(__uint128_t));
    __uint128_t carry = inp_val;

    for (uint64_t b = 0; b < m; b++) {
      __uint128_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(__uint128_t));
      __uint128_t carry_next = counts & carry;
      __uint128_t new_counts = counts ^ carry;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(__uint128_t));
      carry = carry_next;
    }

    __uint128_t acts;
    memcpy(&acts, &actions[k], sizeof(__uint128_t));
    __uint128_t carry_next = acts & carry;
    __uint128_t new_acts = acts ^ carry;
    memcpy(&actions[k], &new_acts, sizeof(__uint128_t));
    carry = carry_next;

    for (uint64_t b = 0; b < m; b++) {
      __uint128_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(__uint128_t));
      __uint128_t new_counts = counts | carry;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(__uint128_t));
    }
    memcpy(&acts, &actions[k], sizeof(__uint128_t));
    acts = acts | carry;
    memcpy(&actions[k], &acts, sizeof(__uint128_t));
  }
#else
  for (; k + 7 < n_chunks; k += 8) {
    uint64_t inp_val;
    memcpy(&inp_val, &input[k], sizeof(uint64_t));
    uint64_t carry = inp_val;

    for (uint64_t b = 0; b < m; b++) {
      uint64_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(uint64_t));
      uint64_t carry_next = counts & carry;
      uint64_t new_counts = counts ^ carry;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(uint64_t));
      carry = carry_next;
    }

    uint64_t acts;
    memcpy(&acts, &actions[k], sizeof(uint64_t));
    uint64_t carry_next = acts & carry;
    uint64_t new_acts = acts ^ carry;
    memcpy(&actions[k], &new_acts, sizeof(uint64_t));
    carry = carry_next;

    for (uint64_t b = 0; b < m; b++) {
      uint64_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(uint64_t));
      uint64_t new_counts = counts | carry;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(uint64_t));
    }
    memcpy(&acts, &actions[k], sizeof(uint64_t));
    acts = acts | carry;
    memcpy(&actions[k], &acts, sizeof(uint64_t));
  }
#endif

  for (; k < n_chunks; k++) {
    tk_automata_inc_byte(aut, clause, k, input[k]);
  }
}

static inline void tk_automata_dec (
  tk_automata_t *aut,
  uint64_t clause,
  const uint8_t *input,
  uint64_t n_chunks
) {
  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);

  uint8_t *counts_planes[m];
  for (uint64_t b = 0; b < m; b++) {
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
  }

  uint64_t k = 0;

#ifdef __SIZEOF_INT128__
  for (; k + 15 < n_chunks; k += 16) {
    __uint128_t inp_val;
    memcpy(&inp_val, &input[k], sizeof(__uint128_t));
    __uint128_t borrow = inp_val;

    for (uint64_t b = 0; b < m; b++) {
      __uint128_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(__uint128_t));
      __uint128_t borrow_next = ~counts & borrow;
      __uint128_t new_counts = counts ^ borrow;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(__uint128_t));
      borrow = borrow_next;
    }

    __uint128_t acts;
    memcpy(&acts, &actions[k], sizeof(__uint128_t));
    __uint128_t borrow_next = ~acts & borrow;
    __uint128_t new_acts = acts ^ borrow;
    memcpy(&actions[k], &new_acts, sizeof(__uint128_t));
    borrow = borrow_next;

    __uint128_t not_borrow = ~borrow;
    for (uint64_t b = 0; b < m; b++) {
      __uint128_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(__uint128_t));
      __uint128_t new_counts = counts & not_borrow;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(__uint128_t));
    }
    memcpy(&acts, &actions[k], sizeof(__uint128_t));
    acts = acts & not_borrow;
    memcpy(&actions[k], &acts, sizeof(__uint128_t));
  }
#else
  for (; k + 7 < n_chunks; k += 8) {
    uint64_t inp_val;
    memcpy(&inp_val, &input[k], sizeof(uint64_t));
    uint64_t borrow = inp_val;

    for (uint64_t b = 0; b < m; b++) {
      uint64_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(uint64_t));
      uint64_t borrow_next = ~counts & borrow;
      uint64_t new_counts = counts ^ borrow;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(uint64_t));
      borrow = borrow_next;
    }

    uint64_t acts;
    memcpy(&acts, &actions[k], sizeof(uint64_t));
    uint64_t borrow_next = ~acts & borrow;
    uint64_t new_acts = acts ^ borrow;
    memcpy(&actions[k], &new_acts, sizeof(uint64_t));
    borrow = borrow_next;

    uint64_t not_borrow = ~borrow;
    for (uint64_t b = 0; b < m; b++) {
      uint64_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(uint64_t));
      uint64_t new_counts = counts & not_borrow;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(uint64_t));
    }
    memcpy(&acts, &actions[k], sizeof(uint64_t));
    acts = acts & not_borrow;
    memcpy(&actions[k], &acts, sizeof(uint64_t));
  }
#endif

  for (; k < n_chunks; k++) {
    tk_automata_dec_byte(aut, clause, k, input[k]);
  }
}

static inline void tk_automata_inc_not (
  tk_automata_t *aut,
  uint64_t clause,
  const uint8_t *input,
  uint64_t n_chunks
) {
  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);

  uint8_t *counts_planes[m];
  for (uint64_t b = 0; b < m; b++) {
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
  }

  uint64_t k = 0;

#ifdef __SIZEOF_INT128__
  for (; k + 15 < n_chunks; k += 16) {
    __uint128_t inp_val;
    memcpy(&inp_val, &input[k], sizeof(__uint128_t));
    __uint128_t carry = ~inp_val;

    for (uint64_t b = 0; b < m; b++) {
      __uint128_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(__uint128_t));
      __uint128_t carry_next = counts & carry;
      __uint128_t new_counts = counts ^ carry;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(__uint128_t));
      carry = carry_next;
    }

    __uint128_t acts;
    memcpy(&acts, &actions[k], sizeof(__uint128_t));
    __uint128_t carry_next = acts & carry;
    __uint128_t new_acts = acts ^ carry;
    memcpy(&actions[k], &new_acts, sizeof(__uint128_t));
    carry = carry_next;

    for (uint64_t b = 0; b < m; b++) {
      __uint128_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(__uint128_t));
      __uint128_t new_counts = counts | carry;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(__uint128_t));
    }
    memcpy(&acts, &actions[k], sizeof(__uint128_t));
    acts = acts | carry;
    memcpy(&actions[k], &acts, sizeof(__uint128_t));
  }
#else
  for (; k + 7 < n_chunks; k += 8) {
    uint64_t inp_val;
    memcpy(&inp_val, &input[k], sizeof(uint64_t));
    uint64_t carry = ~inp_val;

    for (uint64_t b = 0; b < m; b++) {
      uint64_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(uint64_t));
      uint64_t carry_next = counts & carry;
      uint64_t new_counts = counts ^ carry;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(uint64_t));
      carry = carry_next;
    }

    uint64_t acts;
    memcpy(&acts, &actions[k], sizeof(uint64_t));
    uint64_t carry_next = acts & carry;
    uint64_t new_acts = acts ^ carry;
    memcpy(&actions[k], &new_acts, sizeof(uint64_t));
    carry = carry_next;

    for (uint64_t b = 0; b < m; b++) {
      uint64_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(uint64_t));
      uint64_t new_counts = counts | carry;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(uint64_t));
    }
    memcpy(&acts, &actions[k], sizeof(uint64_t));
    acts = acts | carry;
    memcpy(&actions[k], &acts, sizeof(uint64_t));
  }
#endif

  for (; k < n_chunks; k++) {
    tk_automata_inc_byte(aut, clause, k, ~input[k]);
  }
}

static inline void tk_automata_dec_not (
  tk_automata_t *aut,
  uint64_t clause,
  const uint8_t *input,
  uint64_t n_chunks
) {
  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);

  uint8_t *counts_planes[m];
  for (uint64_t b = 0; b < m; b++) {
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
  }

  uint64_t k = 0;

#ifdef __SIZEOF_INT128__
  for (; k + 15 < n_chunks; k += 16) {
    __uint128_t inp_val;
    memcpy(&inp_val, &input[k], sizeof(__uint128_t));
    __uint128_t borrow = ~inp_val;

    for (uint64_t b = 0; b < m; b++) {
      __uint128_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(__uint128_t));
      __uint128_t borrow_next = ~counts & borrow;
      __uint128_t new_counts = counts ^ borrow;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(__uint128_t));
      borrow = borrow_next;
    }

    __uint128_t acts;
    memcpy(&acts, &actions[k], sizeof(__uint128_t));
    __uint128_t borrow_next = ~acts & borrow;
    __uint128_t new_acts = acts ^ borrow;
    memcpy(&actions[k], &new_acts, sizeof(__uint128_t));
    borrow = borrow_next;

    __uint128_t not_borrow = ~borrow;
    for (uint64_t b = 0; b < m; b++) {
      __uint128_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(__uint128_t));
      __uint128_t new_counts = counts & not_borrow;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(__uint128_t));
    }
    memcpy(&acts, &actions[k], sizeof(__uint128_t));
    acts = acts & not_borrow;
    memcpy(&actions[k], &acts, sizeof(__uint128_t));
  }
#else
  for (; k + 7 < n_chunks; k += 8) {
    uint64_t inp_val;
    memcpy(&inp_val, &input[k], sizeof(uint64_t));
    uint64_t borrow = ~inp_val;

    for (uint64_t b = 0; b < m; b++) {
      uint64_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(uint64_t));
      uint64_t borrow_next = ~counts & borrow;
      uint64_t new_counts = counts ^ borrow;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(uint64_t));
      borrow = borrow_next;
    }

    uint64_t acts;
    memcpy(&acts, &actions[k], sizeof(uint64_t));
    uint64_t borrow_next = ~acts & borrow;
    uint64_t new_acts = acts ^ borrow;
    memcpy(&actions[k], &new_acts, sizeof(uint64_t));
    borrow = borrow_next;

    uint64_t not_borrow = ~borrow;
    for (uint64_t b = 0; b < m; b++) {
      uint64_t counts;
      memcpy(&counts, &counts_planes[b][k], sizeof(uint64_t));
      uint64_t new_counts = counts & not_borrow;
      memcpy(&counts_planes[b][k], &new_counts, sizeof(uint64_t));
    }
    memcpy(&acts, &actions[k], sizeof(uint64_t));
    acts = acts & not_borrow;
    memcpy(&actions[k], &acts, sizeof(uint64_t));
  }
#endif

  for (; k < n_chunks; k++) {
    tk_automata_dec_byte(aut, clause, k, ~input[k]);
  }
}

#endif
