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

static inline char *tk_automata_counts (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk
) {
  return &aut->counts[clause * aut->n_chunks * (aut->state_bits - 1) +
                      chunk * (aut->state_bits - 1)];
}

static inline char *tk_automata_actions (
  tk_automata_t *aut,
  uint64_t clause
) {
  return &aut->actions[clause * aut->n_chunks];
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
    for (uint64_t chunk = 0; chunk < aut->n_chunks; chunk++) {
      uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);
      uint8_t *counts = (uint8_t *)tk_automata_counts(aut, clause, chunk);
      for (uint64_t b = 0; b < m; b++)
        counts[b] = 0xFF;
      actions[chunk] = 0;
    }
  }
}

static inline void tk_automata_inc (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk,
  uint8_t active
) {
  if (!active) return;

  uint64_t m = aut->state_bits - 1;
  uint8_t *counts = (uint8_t *)tk_automata_counts(aut, clause, chunk);
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);
  uint8_t mask = (chunk == aut->n_chunks - 1) ? aut->tail_mask : 0xFF;
  uint8_t carry = active;
  for (uint64_t b = 0; b < m; b++) {
    uint8_t carry_next = counts[b] & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  uint8_t carry_masked = carry & mask;
  uint8_t carry_next = actions[chunk] & carry_masked;
  actions[chunk] ^= carry_masked;
  carry = carry_next;
  for (uint64_t b = 0; b < m; b++)
    counts[b] |= carry_masked;
  actions[chunk] |= carry_masked;
}

static inline void tk_automata_dec (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk,
  uint8_t active
) {
  if (!active) return;

  uint64_t m = aut->state_bits - 1;
  uint8_t *counts = (uint8_t *)tk_automata_counts(aut, clause, chunk);
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);
  uint8_t mask = (chunk == aut->n_chunks - 1) ? aut->tail_mask : 0xFF;
  uint8_t borrow = active;
  for (uint64_t b = 0; b < m; b++) {
    uint8_t borrow_next = (~counts[b]) & borrow;
    counts[b] ^= borrow;
    borrow = borrow_next;
  }
  uint8_t borrow_masked = borrow & mask;
  uint8_t borrow_next = (~actions[chunk]) & borrow_masked;
  actions[chunk] ^= borrow_masked;
  borrow = borrow_next;
  for (uint64_t b = 0; b < m; b++)
    counts[b] &= ~borrow_masked;
  actions[chunk] &= ~borrow_masked;
}

#endif
