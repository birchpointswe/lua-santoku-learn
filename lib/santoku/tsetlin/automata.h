#ifndef TK_AUTOMATA_H
#define TK_AUTOMATA_H

#include <santoku/tsetlin/conf.h>
#include <santoku/cvec.h>

// Bit-parallel counter array for Tsetlin Machine automata
// Uses inverted two's complement: MSB=1 means count>=0, MSB=0 means count<0
// Optimized for fast parallel inc/dec operations across all bit positions
typedef struct {
  uint64_t n_clauses;     // Number of independent automata (clauses)
  uint64_t n_chunks;      // Input vector chunks
  uint64_t state_bits;    // Counter bit width (typically 8)
  tk_bits_t *counts;      // Bit planes [n_clauses × n_chunks × (state_bits-1)]
  tk_bits_t *actions;     // MSB plane (action bits) [n_clauses × n_chunks]
  tk_bits_t tail_mask;    // Mask for last chunk
} tk_automata_t;

// Get count bit planes for a specific clause and chunk
static inline tk_bits_t *tk_automata_counts (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk
) {
  return &aut->counts[clause * aut->n_chunks * (aut->state_bits - 1) +
                      chunk * (aut->state_bits - 1)];
}

// Get action bits for a specific clause
static inline tk_bits_t *tk_automata_actions (
  tk_automata_t *aut,
  uint64_t clause
) {
  return &aut->actions[clause * aut->n_chunks];
}

// Initialize automata structure
static inline int tk_automata_init (
  lua_State *L,
  tk_automata_t *aut,
  uint64_t n_clauses,
  uint64_t n_chunks,
  uint64_t state_bits,
  tk_bits_t tail_mask
) {
  aut->n_clauses = n_clauses;
  aut->n_chunks = n_chunks;
  aut->state_bits = state_bits;
  aut->tail_mask = tail_mask;

  uint64_t action_chunks = n_clauses * n_chunks;
  aut->actions = tk_malloc_aligned(L, sizeof(tk_bits_t) * action_chunks, TK_CVEC_BITS);
  if (!aut->actions)
    return -1;

  uint64_t state_chunks = n_clauses * n_chunks * (state_bits - 1);
  aut->counts = tk_malloc_aligned(L, sizeof(tk_bits_t) * state_chunks, TK_CVEC_BITS);
  if (!aut->counts) {
    free(aut->actions);
    aut->actions = NULL;
    return -1;
  }

  return 0;
}

// Destroy automata and free memory
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

// Setup initial state for a range of clauses
// Initializes all counters to -1 (inverted two's complement: lower bits all 1, MSB=0)
static inline void tk_automata_setup (
  tk_automata_t *aut,
  uint64_t clause_first,
  uint64_t clause_last
) {
  uint64_t m = aut->state_bits - 1;

  for (uint64_t clause = clause_first; clause <= clause_last; clause++) {
    for (uint64_t chunk = 0; chunk < aut->n_chunks; chunk++) {
      tk_bits_t *actions = tk_automata_actions(aut, clause);
      tk_bits_t *counts = tk_automata_counts(aut, clause, chunk);

      // Set all lower bits to 1 (representing -1)
      for (uint64_t b = 0; b < m; b++)
        counts[b] = TK_CVEC_ALL_MASK;

      // MSB = 0 for negative (inverted two's complement)
      actions[chunk] = 0;
    }
  }
}

// Increment counters for active bit positions (vote for 1)
// Uses bit-parallel ripple-carry addition across all 32 positions simultaneously
static inline void tk_automata_inc (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk,
  tk_bits_t active
) {
  if (!active) return;

  uint64_t m = aut->state_bits - 1;
  tk_bits_t *counts = tk_automata_counts(aut, clause, chunk);
  tk_bits_t *actions = tk_automata_actions(aut, clause);
  tk_bits_t mask = (chunk == aut->n_chunks - 1) ? aut->tail_mask : (tk_bits_t)~0;

  // Ripple carry addition
  tk_bits_t carry = active;
  for (uint64_t b = 0; b < m; b++) {
    tk_bits_t carry_next = counts[b] & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }

  // Final carry into MSB (inverted sign bit: flip means crossing 0)
  tk_bits_t carry_masked = carry & mask;
  tk_bits_t carry_next = actions[chunk] & carry_masked;
  actions[chunk] ^= carry_masked;
  carry = carry_next;

  // Saturation on overflow: set all bits to 1
  for (uint64_t b = 0; b < m; b++)
    counts[b] |= carry_masked;
  actions[chunk] |= carry_masked;
}

// Decrement counters for active bit positions (vote for 0)
// Uses bit-parallel ripple-borrow subtraction
static inline void tk_automata_dec (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk,
  tk_bits_t active
) {
  if (!active) return;

  uint64_t m = aut->state_bits - 1;
  tk_bits_t *counts = tk_automata_counts(aut, clause, chunk);
  tk_bits_t *actions = tk_automata_actions(aut, clause);
  tk_bits_t mask = (chunk == aut->n_chunks - 1) ? aut->tail_mask : (tk_bits_t)~0;

  // Ripple borrow subtraction
  tk_bits_t borrow = active;
  for (uint64_t b = 0; b < m; b++) {
    tk_bits_t borrow_next = (~counts[b]) & borrow;
    counts[b] ^= borrow;
    borrow = borrow_next;
  }

  // Final borrow from MSB
  tk_bits_t borrow_masked = borrow & mask;
  tk_bits_t borrow_next = (~actions[chunk]) & borrow_masked;
  actions[chunk] ^= borrow_masked;
  borrow = borrow_next;

  // Saturation on underflow: set all bits to 0
  for (uint64_t b = 0; b < m; b++)
    counts[b] &= ~borrow_masked;
  actions[chunk] &= ~borrow_masked;
}

#endif
