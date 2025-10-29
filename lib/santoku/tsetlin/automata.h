#ifndef TK_AUTOMATA_H
#define TK_AUTOMATA_H

#include <omp.h>
#include <santoku/cvec.h>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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
#if defined(__AVX2__)
  for (; k + 31 < n_chunks; k += 32) {
    __m256i v_carry = _mm256_loadu_si256((__m256i*)&input[k]);
    for (uint64_t b = 0; b < m; b++) {
      __m256i v_counts_b = _mm256_loadu_si256((__m256i*)&counts_planes[b][k]);
      __m256i v_carry_next = _mm256_and_si256(v_counts_b, v_carry);
      _mm256_storeu_si256((__m256i*)&counts_planes[b][k], _mm256_xor_si256(v_counts_b, v_carry));
      v_carry = v_carry_next;
    }
    __m256i v_actions = _mm256_loadu_si256((__m256i*)&actions[k]);
    __m256i v_carry_next = _mm256_and_si256(v_actions, v_carry);
    _mm256_storeu_si256((__m256i*)&actions[k], _mm256_xor_si256(v_actions, v_carry));
    v_carry = v_carry_next;
    for (uint64_t b = 0; b < m; b++) {
      __m256i v_counts_b = _mm256_loadu_si256((__m256i*)&counts_planes[b][k]);
      _mm256_storeu_si256((__m256i*)&counts_planes[b][k], _mm256_or_si256(v_counts_b, v_carry));
    }
    v_actions = _mm256_loadu_si256((__m256i*)&actions[k]);
    _mm256_storeu_si256((__m256i*)&actions[k], _mm256_or_si256(v_actions, v_carry));
  }
#elif defined(__ARM_NEON)
  for (; k + 15 < n_chunks; k += 16) {
    uint8x16_t v_carry = vld1q_u8(&input[k]);
    for (uint64_t b = 0; b < m; b++) {
      uint8x16_t v_counts_b = vld1q_u8(&counts_planes[b][k]);
      uint8x16_t v_carry_next = vandq_u8(v_counts_b, v_carry);
      vst1q_u8(&counts_planes[b][k], veorq_u8(v_counts_b, v_carry));
      v_carry = v_carry_next;
    }
    uint8x16_t v_actions = vld1q_u8(&actions[k]);
    uint8x16_t v_carry_next = vandq_u8(v_actions, v_carry);
    vst1q_u8(&actions[k], veorq_u8(v_actions, v_carry));
    v_carry = v_carry_next;
    for (uint64_t b = 0; b < m; b++) {
      uint8x16_t v_counts_b = vld1q_u8(&counts_planes[b][k]);
      vst1q_u8(&counts_planes[b][k], vorrq_u8(v_counts_b, v_carry));
    }
    v_actions = vld1q_u8(&actions[k]);
    vst1q_u8(&actions[k], vorrq_u8(v_actions, v_carry));
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
#if defined(__AVX2__)
  const __m256i v_all_ones = _mm256_set1_epi8(0xFF);
  for (; k + 31 < n_chunks; k += 32) {
    __m256i v_borrow = _mm256_loadu_si256((__m256i*)&input[k]);
    for (uint64_t b = 0; b < m; b++) {
      __m256i v_counts_b = _mm256_loadu_si256((__m256i*)&counts_planes[b][k]);
      __m256i v_borrow_next = _mm256_andnot_si256(v_counts_b, v_borrow);
      _mm256_storeu_si256((__m256i*)&counts_planes[b][k], _mm256_xor_si256(v_counts_b, v_borrow));
      v_borrow = v_borrow_next;
    }
    __m256i v_actions = _mm256_loadu_si256((__m256i*)&actions[k]);
    __m256i v_borrow_next = _mm256_andnot_si256(v_actions, v_borrow);
    _mm256_storeu_si256((__m256i*)&actions[k], _mm256_xor_si256(v_actions, v_borrow));
    v_borrow = v_borrow_next;
    __m256i v_not_borrow = _mm256_xor_si256(v_borrow, v_all_ones);
    for (uint64_t b = 0; b < m; b++) {
      __m256i v_counts_b = _mm256_loadu_si256((__m256i*)&counts_planes[b][k]);
      _mm256_storeu_si256((__m256i*)&counts_planes[b][k], _mm256_and_si256(v_counts_b, v_not_borrow));
    }
    v_actions = _mm256_loadu_si256((__m256i*)&actions[k]);
    _mm256_storeu_si256((__m256i*)&actions[k], _mm256_and_si256(v_actions, v_not_borrow));
  }
#elif defined(__ARM_NEON)
  for (; k + 15 < n_chunks; k += 16) {
    uint8x16_t v_borrow = vld1q_u8(&input[k]);
    for (uint64_t b = 0; b < m; b++) {
      uint8x16_t v_counts_b = vld1q_u8(&counts_planes[b][k]);
      uint8x16_t v_borrow_next = vbicq_u8(v_borrow, v_counts_b);
      vst1q_u8(&counts_planes[b][k], veorq_u8(v_counts_b, v_borrow));
      v_borrow = v_borrow_next;
    }
    uint8x16_t v_actions = vld1q_u8(&actions[k]);
    uint8x16_t v_borrow_next = vbicq_u8(v_borrow, v_actions);
    vst1q_u8(&actions[k], veorq_u8(v_actions, v_borrow));
    v_borrow = v_borrow_next;
    uint8x16_t v_not_borrow = vmvnq_u8(v_borrow);
    for (uint64_t b = 0; b < m; b++) {
      uint8x16_t v_counts_b = vld1q_u8(&counts_planes[b][k]);
      vst1q_u8(&counts_planes[b][k], vandq_u8(v_counts_b, v_not_borrow));
    }
    v_actions = vld1q_u8(&actions[k]);
    vst1q_u8(&actions[k], vandq_u8(v_actions, v_not_borrow));
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
#if defined(__AVX2__)
  const __m256i v_all_ones = _mm256_set1_epi8(0xFF);
  for (; k + 31 < n_chunks; k += 32) {
    __m256i v_input = _mm256_loadu_si256((__m256i*)&input[k]);
    __m256i v_carry = _mm256_xor_si256(v_input, v_all_ones);
    for (uint64_t b = 0; b < m; b++) {
      __m256i v_counts_b = _mm256_loadu_si256((__m256i*)&counts_planes[b][k]);
      __m256i v_carry_next = _mm256_and_si256(v_counts_b, v_carry);
      _mm256_storeu_si256((__m256i*)&counts_planes[b][k], _mm256_xor_si256(v_counts_b, v_carry));
      v_carry = v_carry_next;
    }
    __m256i v_actions = _mm256_loadu_si256((__m256i*)&actions[k]);
    __m256i v_carry_next = _mm256_and_si256(v_actions, v_carry);
    _mm256_storeu_si256((__m256i*)&actions[k], _mm256_xor_si256(v_actions, v_carry));
    v_carry = v_carry_next;
    for (uint64_t b = 0; b < m; b++) {
      __m256i v_counts_b = _mm256_loadu_si256((__m256i*)&counts_planes[b][k]);
      _mm256_storeu_si256((__m256i*)&counts_planes[b][k], _mm256_or_si256(v_counts_b, v_carry));
    }
    v_actions = _mm256_loadu_si256((__m256i*)&actions[k]);
    _mm256_storeu_si256((__m256i*)&actions[k], _mm256_or_si256(v_actions, v_carry));
  }
#elif defined(__ARM_NEON)
  for (; k + 15 < n_chunks; k += 16) {
    uint8x16_t v_input = vld1q_u8(&input[k]);
    uint8x16_t v_carry = vmvnq_u8(v_input);
    for (uint64_t b = 0; b < m; b++) {
      uint8x16_t v_counts_b = vld1q_u8(&counts_planes[b][k]);
      uint8x16_t v_carry_next = vandq_u8(v_counts_b, v_carry);
      vst1q_u8(&counts_planes[b][k], veorq_u8(v_counts_b, v_carry));
      v_carry = v_carry_next;
    }
    uint8x16_t v_actions = vld1q_u8(&actions[k]);
    uint8x16_t v_carry_next = vandq_u8(v_actions, v_carry);
    vst1q_u8(&actions[k], veorq_u8(v_actions, v_carry));
    v_carry = v_carry_next;
    for (uint64_t b = 0; b < m; b++) {
      uint8x16_t v_counts_b = vld1q_u8(&counts_planes[b][k]);
      vst1q_u8(&counts_planes[b][k], vorrq_u8(v_counts_b, v_carry));
    }
    v_actions = vld1q_u8(&actions[k]);
    vst1q_u8(&actions[k], vorrq_u8(v_actions, v_carry));
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
#if defined(__AVX2__)
  const __m256i v_all_ones = _mm256_set1_epi8(0xFF);
  for (; k + 31 < n_chunks; k += 32) {
    __m256i v_input = _mm256_loadu_si256((__m256i*)&input[k]);
    __m256i v_borrow = _mm256_xor_si256(v_input, v_all_ones);
    for (uint64_t b = 0; b < m; b++) {
      __m256i v_counts_b = _mm256_loadu_si256((__m256i*)&counts_planes[b][k]);
      __m256i v_borrow_next = _mm256_andnot_si256(v_counts_b, v_borrow);
      _mm256_storeu_si256((__m256i*)&counts_planes[b][k], _mm256_xor_si256(v_counts_b, v_borrow));
      v_borrow = v_borrow_next;
    }
    __m256i v_actions = _mm256_loadu_si256((__m256i*)&actions[k]);
    __m256i v_borrow_next = _mm256_andnot_si256(v_actions, v_borrow);
    _mm256_storeu_si256((__m256i*)&actions[k], _mm256_xor_si256(v_actions, v_borrow));
    v_borrow = v_borrow_next;
    __m256i v_not_borrow = _mm256_xor_si256(v_borrow, v_all_ones);
    for (uint64_t b = 0; b < m; b++) {
      __m256i v_counts_b = _mm256_loadu_si256((__m256i*)&counts_planes[b][k]);
      _mm256_storeu_si256((__m256i*)&counts_planes[b][k], _mm256_and_si256(v_counts_b, v_not_borrow));
    }
    v_actions = _mm256_loadu_si256((__m256i*)&actions[k]);
    _mm256_storeu_si256((__m256i*)&actions[k], _mm256_and_si256(v_actions, v_not_borrow));
  }
#elif defined(__ARM_NEON)
  for (; k + 15 < n_chunks; k += 16) {
    uint8x16_t v_input = vld1q_u8(&input[k]);
    uint8x16_t v_borrow = vmvnq_u8(v_input);
    for (uint64_t b = 0; b < m; b++) {
      uint8x16_t v_counts_b = vld1q_u8(&counts_planes[b][k]);
      uint8x16_t v_borrow_next = vbicq_u8(v_borrow, v_counts_b);
      vst1q_u8(&counts_planes[b][k], veorq_u8(v_counts_b, v_borrow));
      v_borrow = v_borrow_next;
    }
    uint8x16_t v_actions = vld1q_u8(&actions[k]);
    uint8x16_t v_borrow_next = vbicq_u8(v_borrow, v_actions);
    vst1q_u8(&actions[k], veorq_u8(v_actions, v_borrow));
    v_borrow = v_borrow_next;
    uint8x16_t v_not_borrow = vmvnq_u8(v_borrow);
    for (uint64_t b = 0; b < m; b++) {
      uint8x16_t v_counts_b = vld1q_u8(&counts_planes[b][k]);
      vst1q_u8(&counts_planes[b][k], vandq_u8(v_counts_b, v_not_borrow));
    }
    v_actions = vld1q_u8(&actions[k]);
    vst1q_u8(&actions[k], vandq_u8(v_actions, v_not_borrow));
  }
#endif
  for (; k < n_chunks; k++) {
    tk_automata_dec_byte(aut, clause, k, ~input[k]);
  }
}

#endif
