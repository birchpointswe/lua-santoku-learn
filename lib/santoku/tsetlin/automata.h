#ifndef TK_AUTOMATA_H
#define TK_AUTOMATA_H

#include <omp.h>
#include <santoku/cvec.h>
#include <santoku/lua/utils.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#define TK_AUTOMATA_CARRY_STACK 8192

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

static inline void tk_automata_setup (
  tk_automata_t *aut,
  uint64_t clause_first,
  uint64_t clause_last
) {
  uint64_t m = aut->state_bits - 1;
  for (uint64_t clause = clause_first; clause <= clause_last; clause++) {
    for (uint64_t b = 0; b < m; b++) {
      uint8_t *counts_plane = (uint8_t *)tk_automata_counts_plane(aut, clause, b);
      memset(counts_plane, 0xFF, aut->n_chunks);
    }
    uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);
    memset(actions, 0, aut->n_chunks);
  }
}

static inline void tk_automata_setup_midpoint (
  tk_automata_t *aut,
  uint64_t clause_first,
  uint64_t clause_last
) {
  tk_automata_setup(aut, clause_first, clause_last);
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

static inline uint8_t tk_automata_included_byte (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk
) {
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);
  return actions[chunk];
}

static inline void tk_automata_dec_byte_excluded (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk,
  uint8_t active
) {
  uint8_t included = tk_automata_included_byte(aut, clause, chunk);
  active &= ~included;
  if (!active) return;
  tk_automata_dec_byte(aut, clause, chunk, active);
}

static inline void tk_automata_inc_byte_excluded (
  tk_automata_t *aut,
  uint64_t clause,
  uint64_t chunk,
  uint8_t active
) {
  uint8_t included = tk_automata_included_byte(aut, clause, chunk);
  active &= ~included;
  if (!active) return;
  tk_automata_inc_byte(aut, clause, chunk, active);
}

#define TK_CARRY_ALLOC(name, n) \
  uint8_t name##_stack_[TK_AUTOMATA_CARRY_STACK]; \
  uint8_t *name##_heap_ = ((n) >= TK_AUTOMATA_CARRY_STACK) ? (uint8_t *)malloc(n) : NULL; \
  uint8_t *name = name##_heap_ ? name##_heap_ : name##_stack_

#define TK_CARRY_FREE(name) free(name##_heap_)

#ifdef __AVX512F__

static inline int tk_automata_carry_zero (const uint8_t *buf, uint64_t n) {
  __m512i acc = _mm512_setzero_si512();
  uint64_t k = 0;
  for (; k + 63 < n; k += 64)
    acc = _mm512_or_si512(acc, _mm512_loadu_si512(buf + k));
  if (_mm512_test_epi64_mask(acc, acc)) return 0;
  for (; k < n; k++)
    if (buf[k]) return 0;
  return 1;
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
  for (uint64_t b = 0; b < m; b++)
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);

  TK_CARRY_ALLOC(carry, n_chunks);
  memcpy(carry, input, n_chunks);

  uint64_t b = 0;
  for (; b + 1 < m; b += 2) {
    uint8_t *p0 = counts_planes[b];
    uint8_t *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo0 = _mm512_loadu_si512(p0 + k);
      __m512i vo1 = _mm512_loadu_si512(p1 + k);
      __m512i vc  = _mm512_loadu_si512(carry + k);
      __m512i nc  = _mm512_ternarylogic_epi64(vo1, vo0, vc, 0x80);
      _mm512_storeu_si512(p0 + k, _mm512_xor_si512(vo0, vc));
      _mm512_storeu_si512(p1 + k, _mm512_ternarylogic_epi64(vo1, vo0, vc, 0x78));
      _mm512_storeu_si512(carry + k, nc);
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = carry[k];
      p0[k] = o0 ^ c;
      p1[k] = o1 ^ (o0 & c);
      carry[k] = o1 & o0 & c;
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo = _mm512_loadu_si512(plane + k);
      __m512i vc = _mm512_loadu_si512(carry + k);
      _mm512_storeu_si512(plane + k, _mm512_xor_si512(vo, vc));
      _mm512_storeu_si512(carry + k, _mm512_and_si512(vo, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = plane[k];
      plane[k] = old ^ carry[k];
      carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i va = _mm512_loadu_si512(actions + k);
      __m512i vc = _mm512_loadu_si512(carry + k);
      _mm512_storeu_si512(actions + k, _mm512_xor_si512(va, vc));
      _mm512_storeu_si512(carry + k, _mm512_and_si512(va, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = actions[k];
      actions[k] = old ^ carry[k];
      carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64)
      _mm512_storeu_si512(plane + k, _mm512_or_si512(
        _mm512_loadu_si512(plane + k), _mm512_loadu_si512(carry + k)));
    for (; k < n_chunks; k++)
      plane[k] |= carry[k];
  }
  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64)
      _mm512_storeu_si512(actions + k, _mm512_or_si512(
        _mm512_loadu_si512(actions + k), _mm512_loadu_si512(carry + k)));
    for (; k < n_chunks; k++)
      actions[k] |= carry[k];
  }

done:
  TK_CARRY_FREE(carry);
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
  for (uint64_t b = 0; b < m; b++)
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);

  TK_CARRY_ALLOC(borrow, n_chunks);
  memcpy(borrow, input, n_chunks);

  uint64_t b = 0;
  for (; b + 1 < m; b += 2) {
    uint8_t *p0 = counts_planes[b];
    uint8_t *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo0 = _mm512_loadu_si512(p0 + k);
      __m512i vo1 = _mm512_loadu_si512(p1 + k);
      __m512i vc  = _mm512_loadu_si512(borrow + k);
      __m512i nb  = _mm512_ternarylogic_epi64(vo1, vo0, vc, 0x02);
      _mm512_storeu_si512(p0 + k, _mm512_xor_si512(vo0, vc));
      _mm512_storeu_si512(p1 + k, _mm512_ternarylogic_epi64(vo1, vo0, vc, 0xD2));
      _mm512_storeu_si512(borrow + k, nb);
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = borrow[k];
      p0[k] = o0 ^ c;
      p1[k] = o1 ^ (~o0 & c);
      borrow[k] = ~o1 & ~o0 & c;
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo = _mm512_loadu_si512(plane + k);
      __m512i vc = _mm512_loadu_si512(borrow + k);
      _mm512_storeu_si512(plane + k, _mm512_xor_si512(vo, vc));
      _mm512_storeu_si512(borrow + k, _mm512_andnot_si512(vo, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = plane[k];
      plane[k] = old ^ borrow[k];
      borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i va = _mm512_loadu_si512(actions + k);
      __m512i vc = _mm512_loadu_si512(borrow + k);
      _mm512_storeu_si512(actions + k, _mm512_xor_si512(va, vc));
      _mm512_storeu_si512(borrow + k, _mm512_andnot_si512(va, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = actions[k];
      actions[k] = old ^ borrow[k];
      borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64)
      _mm512_storeu_si512(plane + k, _mm512_andnot_si512(
        _mm512_loadu_si512(borrow + k), _mm512_loadu_si512(plane + k)));
    for (; k < n_chunks; k++)
      plane[k] &= ~borrow[k];
  }
  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64)
      _mm512_storeu_si512(actions + k, _mm512_andnot_si512(
        _mm512_loadu_si512(borrow + k), _mm512_loadu_si512(actions + k)));
    for (; k < n_chunks; k++)
      actions[k] &= ~borrow[k];
  }

done:
  TK_CARRY_FREE(borrow);
}

static inline void tk_automata_dec_not_excluded (
  tk_automata_t *aut,
  uint64_t clause,
  const uint8_t *input,
  uint64_t n_chunks
) {
  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);

  uint8_t *counts_planes[m];
  for (uint64_t b = 0; b < m; b++)
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);

  TK_CARRY_ALLOC(borrow, n_chunks);
  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vi = _mm512_loadu_si512(input + k);
      __m512i va = _mm512_loadu_si512(actions + k);
      _mm512_storeu_si512(borrow + k, _mm512_ternarylogic_epi64(vi, va, vi, 0x03));
    }
    for (; k < n_chunks; k++)
      borrow[k] = ~input[k] & ~actions[k];
  }

  uint64_t b = 0;
  for (; b + 1 < m; b += 2) {
    uint8_t *p0 = counts_planes[b];
    uint8_t *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo0 = _mm512_loadu_si512(p0 + k);
      __m512i vo1 = _mm512_loadu_si512(p1 + k);
      __m512i vc  = _mm512_loadu_si512(borrow + k);
      __m512i nb  = _mm512_ternarylogic_epi64(vo1, vo0, vc, 0x02);
      _mm512_storeu_si512(p0 + k, _mm512_xor_si512(vo0, vc));
      _mm512_storeu_si512(p1 + k, _mm512_ternarylogic_epi64(vo1, vo0, vc, 0xD2));
      _mm512_storeu_si512(borrow + k, nb);
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = borrow[k];
      p0[k] = o0 ^ c;
      p1[k] = o1 ^ (~o0 & c);
      borrow[k] = ~o1 & ~o0 & c;
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo = _mm512_loadu_si512(plane + k);
      __m512i vc = _mm512_loadu_si512(borrow + k);
      _mm512_storeu_si512(plane + k, _mm512_xor_si512(vo, vc));
      _mm512_storeu_si512(borrow + k, _mm512_andnot_si512(vo, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = plane[k];
      plane[k] = old ^ borrow[k];
      borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i va = _mm512_loadu_si512(actions + k);
      __m512i vc = _mm512_loadu_si512(borrow + k);
      _mm512_storeu_si512(actions + k, _mm512_xor_si512(va, vc));
      _mm512_storeu_si512(borrow + k, _mm512_andnot_si512(va, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = actions[k];
      actions[k] = old ^ borrow[k];
      borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64)
      _mm512_storeu_si512(plane + k, _mm512_andnot_si512(
        _mm512_loadu_si512(borrow + k), _mm512_loadu_si512(plane + k)));
    for (; k < n_chunks; k++)
      plane[k] &= ~borrow[k];
  }
  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64)
      _mm512_storeu_si512(actions + k, _mm512_andnot_si512(
        _mm512_loadu_si512(borrow + k), _mm512_loadu_si512(actions + k)));
    for (; k < n_chunks; k++)
      actions[k] &= ~borrow[k];
  }

done:
  TK_CARRY_FREE(borrow);
}

static inline void tk_automata_inc_not_excluded (
  tk_automata_t *aut,
  uint64_t clause,
  const uint8_t *input,
  uint64_t n_chunks
) {
  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);

  uint8_t *counts_planes[m];
  for (uint64_t b = 0; b < m; b++)
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);

  TK_CARRY_ALLOC(carry, n_chunks);
  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vi = _mm512_loadu_si512(input + k);
      __m512i va = _mm512_loadu_si512(actions + k);
      _mm512_storeu_si512(carry + k, _mm512_ternarylogic_epi64(vi, va, vi, 0x03));
    }
    for (; k < n_chunks; k++)
      carry[k] = ~input[k] & ~actions[k];
  }

  uint64_t b = 0;
  for (; b + 1 < m; b += 2) {
    uint8_t *p0 = counts_planes[b];
    uint8_t *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo0 = _mm512_loadu_si512(p0 + k);
      __m512i vo1 = _mm512_loadu_si512(p1 + k);
      __m512i vc  = _mm512_loadu_si512(carry + k);
      __m512i nc  = _mm512_ternarylogic_epi64(vo1, vo0, vc, 0x80);
      _mm512_storeu_si512(p0 + k, _mm512_xor_si512(vo0, vc));
      _mm512_storeu_si512(p1 + k, _mm512_ternarylogic_epi64(vo1, vo0, vc, 0x78));
      _mm512_storeu_si512(carry + k, nc);
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = carry[k];
      p0[k] = o0 ^ c;
      p1[k] = o1 ^ (o0 & c);
      carry[k] = o1 & o0 & c;
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo = _mm512_loadu_si512(plane + k);
      __m512i vc = _mm512_loadu_si512(carry + k);
      _mm512_storeu_si512(plane + k, _mm512_xor_si512(vo, vc));
      _mm512_storeu_si512(carry + k, _mm512_and_si512(vo, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = plane[k];
      plane[k] = old ^ carry[k];
      carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i va = _mm512_loadu_si512(actions + k);
      __m512i vc = _mm512_loadu_si512(carry + k);
      _mm512_storeu_si512(actions + k, _mm512_xor_si512(va, vc));
      _mm512_storeu_si512(carry + k, _mm512_and_si512(va, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = actions[k];
      actions[k] = old ^ carry[k];
      carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64)
      _mm512_storeu_si512(plane + k, _mm512_or_si512(
        _mm512_loadu_si512(plane + k), _mm512_loadu_si512(carry + k)));
    for (; k < n_chunks; k++)
      plane[k] |= carry[k];
  }
  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64)
      _mm512_storeu_si512(actions + k, _mm512_or_si512(
        _mm512_loadu_si512(actions + k), _mm512_loadu_si512(carry + k)));
    for (; k < n_chunks; k++)
      actions[k] |= carry[k];
  }

done:
  TK_CARRY_FREE(carry);
}

#else

static inline int tk_automata_carry_zero (const uint8_t *buf, uint64_t n) {
  uint64_t acc = 0;
  for (uint64_t k = 0; k < n; k++)
    acc |= buf[k];
  return !acc;
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
  for (uint64_t b = 0; b < m; b++)
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);

  TK_CARRY_ALLOC(carry, n_chunks);
  memcpy(carry, input, n_chunks);

  for (uint64_t b = 0; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    for (uint64_t k = 0; k < n_chunks; k++) {
      uint8_t old = plane[k];
      plane[k] = old ^ carry[k];
      carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  for (uint64_t k = 0; k < n_chunks; k++) {
    uint8_t old = actions[k];
    actions[k] = old ^ carry[k];
    carry[k] = old & carry[k];
  }
  if (tk_automata_carry_zero(carry, n_chunks)) goto done;

  for (uint64_t b = 0; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    for (uint64_t k = 0; k < n_chunks; k++)
      plane[k] |= carry[k];
  }
  for (uint64_t k = 0; k < n_chunks; k++)
    actions[k] |= carry[k];

done:
  TK_CARRY_FREE(carry);
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
  for (uint64_t b = 0; b < m; b++)
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);

  TK_CARRY_ALLOC(borrow, n_chunks);
  memcpy(borrow, input, n_chunks);

  for (uint64_t b = 0; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    for (uint64_t k = 0; k < n_chunks; k++) {
      uint8_t old = plane[k];
      plane[k] = old ^ borrow[k];
      borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  for (uint64_t k = 0; k < n_chunks; k++) {
    uint8_t old = actions[k];
    actions[k] = old ^ borrow[k];
    borrow[k] = ~old & borrow[k];
  }
  if (tk_automata_carry_zero(borrow, n_chunks)) goto done;

  for (uint64_t b = 0; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    for (uint64_t k = 0; k < n_chunks; k++)
      plane[k] &= ~borrow[k];
  }
  for (uint64_t k = 0; k < n_chunks; k++)
    actions[k] &= ~borrow[k];

done:
  TK_CARRY_FREE(borrow);
}

static inline void tk_automata_dec_not_excluded (
  tk_automata_t *aut,
  uint64_t clause,
  const uint8_t *input,
  uint64_t n_chunks
) {
  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);

  uint8_t *counts_planes[m];
  for (uint64_t b = 0; b < m; b++)
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);

  TK_CARRY_ALLOC(borrow, n_chunks);
  for (uint64_t k = 0; k < n_chunks; k++)
    borrow[k] = ~input[k] & ~actions[k];

  for (uint64_t b = 0; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    for (uint64_t k = 0; k < n_chunks; k++) {
      uint8_t old = plane[k];
      plane[k] = old ^ borrow[k];
      borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  for (uint64_t k = 0; k < n_chunks; k++) {
    uint8_t old = actions[k];
    actions[k] = old ^ borrow[k];
    borrow[k] = ~old & borrow[k];
  }
  if (tk_automata_carry_zero(borrow, n_chunks)) goto done;

  for (uint64_t b = 0; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    for (uint64_t k = 0; k < n_chunks; k++)
      plane[k] &= ~borrow[k];
  }
  for (uint64_t k = 0; k < n_chunks; k++)
    actions[k] &= ~borrow[k];

done:
  TK_CARRY_FREE(borrow);
}

static inline void tk_automata_inc_not_excluded (
  tk_automata_t *aut,
  uint64_t clause,
  const uint8_t *input,
  uint64_t n_chunks
) {
  uint64_t m = aut->state_bits - 1;
  uint8_t *actions = (uint8_t *)tk_automata_actions(aut, clause);

  uint8_t *counts_planes[m];
  for (uint64_t b = 0; b < m; b++)
    counts_planes[b] = (uint8_t *)tk_automata_counts_plane(aut, clause, b);

  TK_CARRY_ALLOC(carry, n_chunks);
  for (uint64_t k = 0; k < n_chunks; k++)
    carry[k] = ~input[k] & ~actions[k];

  for (uint64_t b = 0; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    for (uint64_t k = 0; k < n_chunks; k++) {
      uint8_t old = plane[k];
      plane[k] = old ^ carry[k];
      carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  for (uint64_t k = 0; k < n_chunks; k++) {
    uint8_t old = actions[k];
    actions[k] = old ^ carry[k];
    carry[k] = old & carry[k];
  }
  if (tk_automata_carry_zero(carry, n_chunks)) goto done;

  for (uint64_t b = 0; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    for (uint64_t k = 0; k < n_chunks; k++)
      plane[k] |= carry[k];
  }
  for (uint64_t k = 0; k < n_chunks; k++)
    actions[k] |= carry[k];

done:
  TK_CARRY_FREE(carry);
}

#endif

#endif
