#ifndef TK_AUTOMATA_H
#define TK_AUTOMATA_H

#include <omp.h>
#include <santoku/cvec.h>
#include <santoku/lua/utils.h>

#ifdef __AVX512F__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

typedef uint64_t tk_u64a __attribute__((__may_alias__));
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
  uint64_t name##_stack64_[TK_AUTOMATA_CARRY_STACK / 8]; \
  uint8_t *name##_heap_ = ((n) >= TK_AUTOMATA_CARRY_STACK) ? (uint8_t *)malloc(n) : NULL; \
  uint8_t *name = name##_heap_ ? name##_heap_ : (uint8_t *)name##_stack64_

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

static inline void tk_automata_inc_and_dec_not_excluded (
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
  TK_CARRY_ALLOC(borrow, n_chunks);
  memcpy(carry, input, n_chunks);
  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vi = _mm512_loadu_si512(input + k);
      __m512i va = _mm512_loadu_si512(actions + k);
      _mm512_storeu_si512(borrow + k, _mm512_ternarylogic_epi64(vi, va, vi, 0x03));
    }
    for (; k < n_chunks; k++) borrow[k] = ~input[k] & ~actions[k];
  }

  uint64_t b = 0;
  for (; b + 1 < m; b += 2) {
    uint8_t *p0 = counts_planes[b], *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo0 = _mm512_loadu_si512(p0 + k);
      __m512i vo1 = _mm512_loadu_si512(p1 + k);
      __m512i vc  = _mm512_loadu_si512(carry + k);
      __m512i vd  = _mm512_loadu_si512(borrow + k);
      __m512i ti = _mm512_and_si512(vo0, vc);
      __m512i td = _mm512_andnot_si512(vo0, vd);
      _mm512_storeu_si512(p0 + k, _mm512_ternarylogic_epi64(vo0, vc, vd, 0x96));
      _mm512_storeu_si512(p1 + k, _mm512_ternarylogic_epi64(vo1, ti, td, 0x96));
      _mm512_storeu_si512(carry + k, _mm512_and_si512(vo1, ti));
      _mm512_storeu_si512(borrow + k, _mm512_andnot_si512(vo1, td));
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = carry[k], d = borrow[k];
      uint8_t ti = o0 & c, td = ~o0 & d;
      p0[k] = o0 ^ c ^ d; p1[k] = o1 ^ ti ^ td;
      carry[k] = o1 & ti; borrow[k] = ~o1 & td;
    }
    if (tk_automata_carry_zero(carry, n_chunks) && tk_automata_carry_zero(borrow, n_chunks))
      goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vo = _mm512_loadu_si512(plane + k);
      __m512i vc = _mm512_loadu_si512(carry + k);
      __m512i vd = _mm512_loadu_si512(borrow + k);
      _mm512_storeu_si512(plane + k, _mm512_ternarylogic_epi64(vo, vc, vd, 0x96));
      _mm512_storeu_si512(carry + k, _mm512_and_si512(vo, vc));
      _mm512_storeu_si512(borrow + k, _mm512_andnot_si512(vo, vd));
    }
    for (; k < n_chunks; k++) {
      uint8_t o = plane[k], c = carry[k], d = borrow[k];
      plane[k] = o ^ c ^ d; carry[k] = o & c; borrow[k] = ~o & d;
    }
    if (tk_automata_carry_zero(carry, n_chunks) && tk_automata_carry_zero(borrow, n_chunks))
      goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i va = _mm512_loadu_si512(actions + k);
      __m512i vc = _mm512_loadu_si512(carry + k);
      __m512i vd = _mm512_loadu_si512(borrow + k);
      _mm512_storeu_si512(actions + k, _mm512_ternarylogic_epi64(va, vc, vd, 0x96));
      _mm512_storeu_si512(carry + k, _mm512_and_si512(va, vc));
      _mm512_storeu_si512(borrow + k, _mm512_andnot_si512(va, vd));
    }
    for (; k < n_chunks; k++) {
      uint8_t o = actions[k], c = carry[k], d = borrow[k];
      actions[k] = o ^ c ^ d; carry[k] = o & c; borrow[k] = ~o & d;
    }
    if (tk_automata_carry_zero(carry, n_chunks) && tk_automata_carry_zero(borrow, n_chunks))
      goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i vp = _mm512_loadu_si512(plane + k);
      vp = _mm512_or_si512(vp, _mm512_loadu_si512(carry + k));
      _mm512_storeu_si512(plane + k, _mm512_andnot_si512(_mm512_loadu_si512(borrow + k), vp));
    }
    for (; k < n_chunks; k++) plane[k] = (plane[k] | carry[k]) & ~borrow[k];
  }
  {
    uint64_t k = 0;
    for (; k + 63 < n_chunks; k += 64) {
      __m512i va = _mm512_loadu_si512(actions + k);
      va = _mm512_or_si512(va, _mm512_loadu_si512(carry + k));
      _mm512_storeu_si512(actions + k, _mm512_andnot_si512(_mm512_loadu_si512(borrow + k), va));
    }
    for (; k < n_chunks; k++) actions[k] = (actions[k] | carry[k]) & ~borrow[k];
  }

done:
  TK_CARRY_FREE(borrow);
  TK_CARRY_FREE(carry);
}

#elif defined(__aarch64__)

static inline int tk_automata_carry_zero (const uint8_t *buf, uint64_t n) {
  uint8x16_t acc = vdupq_n_u8(0);
  uint64_t k = 0;
  for (; k + 15 < n; k += 16)
    acc = vorrq_u8(acc, vld1q_u8(buf + k));
  if (vmaxvq_u8(acc)) return 0;
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
    uint8_t *p0 = counts_planes[b], *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo0 = vld1q_u8(p0 + k), vo1 = vld1q_u8(p1 + k), vc = vld1q_u8(carry + k);
      uint8x16_t t = vandq_u8(vo0, vc);
      vst1q_u8(p0 + k, veorq_u8(vo0, vc));
      vst1q_u8(p1 + k, veorq_u8(vo1, t));
      vst1q_u8(carry + k, vandq_u8(vo1, t));
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = carry[k];
      p0[k] = o0 ^ c; p1[k] = o1 ^ (o0 & c); carry[k] = o1 & o0 & c;
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo = vld1q_u8(plane + k), vc = vld1q_u8(carry + k);
      vst1q_u8(plane + k, veorq_u8(vo, vc));
      vst1q_u8(carry + k, vandq_u8(vo, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = plane[k]; plane[k] = old ^ carry[k]; carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t va = vld1q_u8(actions + k), vc = vld1q_u8(carry + k);
      vst1q_u8(actions + k, veorq_u8(va, vc));
      vst1q_u8(carry + k, vandq_u8(va, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = actions[k]; actions[k] = old ^ carry[k]; carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(plane + k, vorrq_u8(vld1q_u8(plane + k), vld1q_u8(carry + k)));
    for (; k < n_chunks; k++) plane[k] |= carry[k];
  }
  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(actions + k, vorrq_u8(vld1q_u8(actions + k), vld1q_u8(carry + k)));
    for (; k < n_chunks; k++) actions[k] |= carry[k];
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
    uint8_t *p0 = counts_planes[b], *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo0 = vld1q_u8(p0 + k), vo1 = vld1q_u8(p1 + k), vc = vld1q_u8(borrow + k);
      uint8x16_t t = vbicq_u8(vc, vo0);
      vst1q_u8(p0 + k, veorq_u8(vo0, vc));
      vst1q_u8(p1 + k, veorq_u8(vo1, t));
      vst1q_u8(borrow + k, vbicq_u8(t, vo1));
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = borrow[k];
      p0[k] = o0 ^ c; p1[k] = o1 ^ (~o0 & c); borrow[k] = ~o1 & ~o0 & c;
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo = vld1q_u8(plane + k), vc = vld1q_u8(borrow + k);
      vst1q_u8(plane + k, veorq_u8(vo, vc));
      vst1q_u8(borrow + k, vbicq_u8(vc, vo));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = plane[k]; plane[k] = old ^ borrow[k]; borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t va = vld1q_u8(actions + k), vc = vld1q_u8(borrow + k);
      vst1q_u8(actions + k, veorq_u8(va, vc));
      vst1q_u8(borrow + k, vbicq_u8(vc, va));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = actions[k]; actions[k] = old ^ borrow[k]; borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(plane + k, vbicq_u8(vld1q_u8(plane + k), vld1q_u8(borrow + k)));
    for (; k < n_chunks; k++) plane[k] &= ~borrow[k];
  }
  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(actions + k, vbicq_u8(vld1q_u8(actions + k), vld1q_u8(borrow + k)));
    for (; k < n_chunks; k++) actions[k] &= ~borrow[k];
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
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(borrow + k, vmvnq_u8(vorrq_u8(vld1q_u8(input + k), vld1q_u8(actions + k))));
    for (; k < n_chunks; k++) borrow[k] = ~input[k] & ~actions[k];
  }

  uint64_t b = 0;
  for (; b + 1 < m; b += 2) {
    uint8_t *p0 = counts_planes[b], *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo0 = vld1q_u8(p0 + k), vo1 = vld1q_u8(p1 + k), vc = vld1q_u8(borrow + k);
      uint8x16_t t = vbicq_u8(vc, vo0);
      vst1q_u8(p0 + k, veorq_u8(vo0, vc));
      vst1q_u8(p1 + k, veorq_u8(vo1, t));
      vst1q_u8(borrow + k, vbicq_u8(t, vo1));
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = borrow[k];
      p0[k] = o0 ^ c; p1[k] = o1 ^ (~o0 & c); borrow[k] = ~o1 & ~o0 & c;
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo = vld1q_u8(plane + k), vc = vld1q_u8(borrow + k);
      vst1q_u8(plane + k, veorq_u8(vo, vc));
      vst1q_u8(borrow + k, vbicq_u8(vc, vo));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = plane[k]; plane[k] = old ^ borrow[k]; borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t va = vld1q_u8(actions + k), vc = vld1q_u8(borrow + k);
      vst1q_u8(actions + k, veorq_u8(va, vc));
      vst1q_u8(borrow + k, vbicq_u8(vc, va));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = actions[k]; actions[k] = old ^ borrow[k]; borrow[k] = ~old & borrow[k];
    }
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(plane + k, vbicq_u8(vld1q_u8(plane + k), vld1q_u8(borrow + k)));
    for (; k < n_chunks; k++) plane[k] &= ~borrow[k];
  }
  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(actions + k, vbicq_u8(vld1q_u8(actions + k), vld1q_u8(borrow + k)));
    for (; k < n_chunks; k++) actions[k] &= ~borrow[k];
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
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(carry + k, vmvnq_u8(vorrq_u8(vld1q_u8(input + k), vld1q_u8(actions + k))));
    for (; k < n_chunks; k++) carry[k] = ~input[k] & ~actions[k];
  }

  uint64_t b = 0;
  for (; b + 1 < m; b += 2) {
    uint8_t *p0 = counts_planes[b], *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo0 = vld1q_u8(p0 + k), vo1 = vld1q_u8(p1 + k), vc = vld1q_u8(carry + k);
      uint8x16_t t = vandq_u8(vo0, vc);
      vst1q_u8(p0 + k, veorq_u8(vo0, vc));
      vst1q_u8(p1 + k, veorq_u8(vo1, t));
      vst1q_u8(carry + k, vandq_u8(vo1, t));
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = carry[k];
      p0[k] = o0 ^ c; p1[k] = o1 ^ (o0 & c); carry[k] = o1 & o0 & c;
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo = vld1q_u8(plane + k), vc = vld1q_u8(carry + k);
      vst1q_u8(plane + k, veorq_u8(vo, vc));
      vst1q_u8(carry + k, vandq_u8(vo, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = plane[k]; plane[k] = old ^ carry[k]; carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t va = vld1q_u8(actions + k), vc = vld1q_u8(carry + k);
      vst1q_u8(actions + k, veorq_u8(va, vc));
      vst1q_u8(carry + k, vandq_u8(va, vc));
    }
    for (; k < n_chunks; k++) {
      uint8_t old = actions[k]; actions[k] = old ^ carry[k]; carry[k] = old & carry[k];
    }
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(plane + k, vorrq_u8(vld1q_u8(plane + k), vld1q_u8(carry + k)));
    for (; k < n_chunks; k++) plane[k] |= carry[k];
  }
  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(actions + k, vorrq_u8(vld1q_u8(actions + k), vld1q_u8(carry + k)));
    for (; k < n_chunks; k++) actions[k] |= carry[k];
  }

done:
  TK_CARRY_FREE(carry);
}

static inline void tk_automata_inc_and_dec_not_excluded (
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
  TK_CARRY_ALLOC(borrow, n_chunks);
  memcpy(carry, input, n_chunks);
  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16)
      vst1q_u8(borrow + k, vmvnq_u8(vorrq_u8(vld1q_u8(input + k), vld1q_u8(actions + k))));
    for (; k < n_chunks; k++) borrow[k] = ~input[k] & ~actions[k];
  }

  uint64_t b = 0;
  for (; b + 1 < m; b += 2) {
    uint8_t *p0 = counts_planes[b], *p1 = counts_planes[b + 1];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo0 = vld1q_u8(p0 + k), vo1 = vld1q_u8(p1 + k);
      uint8x16_t vi = vld1q_u8(carry + k), vd = vld1q_u8(borrow + k);
      uint8x16_t ti = vandq_u8(vo0, vi);
      uint8x16_t td = vbicq_u8(vd, vo0);
      vst1q_u8(p0 + k, veorq_u8(veorq_u8(vo0, vi), vd));
      vst1q_u8(p1 + k, veorq_u8(veorq_u8(vo1, ti), td));
      vst1q_u8(carry + k, vandq_u8(vo1, ti));
      vst1q_u8(borrow + k, vbicq_u8(td, vo1));
    }
    for (; k < n_chunks; k++) {
      uint8_t o0 = p0[k], o1 = p1[k], c = carry[k], d = borrow[k];
      uint8_t ti = o0 & c, td = ~o0 & d;
      p0[k] = o0 ^ c ^ d; p1[k] = o1 ^ ti ^ td;
      carry[k] = o1 & ti; borrow[k] = ~o1 & td;
    }
    if (tk_automata_carry_zero(carry, n_chunks) && tk_automata_carry_zero(borrow, n_chunks))
      goto done;
  }
  for (; b < m; b++) {
    uint8_t *plane = counts_planes[b];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vo = vld1q_u8(plane + k);
      uint8x16_t vi = vld1q_u8(carry + k), vd = vld1q_u8(borrow + k);
      vst1q_u8(plane + k, veorq_u8(veorq_u8(vo, vi), vd));
      vst1q_u8(carry + k, vandq_u8(vo, vi));
      vst1q_u8(borrow + k, vbicq_u8(vd, vo));
    }
    for (; k < n_chunks; k++) {
      uint8_t o = plane[k], c = carry[k], d = borrow[k];
      plane[k] = o ^ c ^ d; carry[k] = o & c; borrow[k] = ~o & d;
    }
    if (tk_automata_carry_zero(carry, n_chunks) && tk_automata_carry_zero(borrow, n_chunks))
      goto done;
  }

  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t va = vld1q_u8(actions + k);
      uint8x16_t vi = vld1q_u8(carry + k), vd = vld1q_u8(borrow + k);
      vst1q_u8(actions + k, veorq_u8(veorq_u8(va, vi), vd));
      vst1q_u8(carry + k, vandq_u8(va, vi));
      vst1q_u8(borrow + k, vbicq_u8(vd, va));
    }
    for (; k < n_chunks; k++) {
      uint8_t o = actions[k], c = carry[k], d = borrow[k];
      actions[k] = o ^ c ^ d; carry[k] = o & c; borrow[k] = ~o & d;
    }
    if (tk_automata_carry_zero(carry, n_chunks) && tk_automata_carry_zero(borrow, n_chunks))
      goto done;
  }

  for (uint64_t b2 = 0; b2 < m; b2++) {
    uint8_t *plane = counts_planes[b2];
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t vp = vld1q_u8(plane + k);
      vst1q_u8(plane + k, vbicq_u8(vorrq_u8(vp, vld1q_u8(carry + k)), vld1q_u8(borrow + k)));
    }
    for (; k < n_chunks; k++) plane[k] = (plane[k] | carry[k]) & ~borrow[k];
  }
  {
    uint64_t k = 0;
    for (; k + 15 < n_chunks; k += 16) {
      uint8x16_t va = vld1q_u8(actions + k);
      vst1q_u8(actions + k, vbicq_u8(vorrq_u8(va, vld1q_u8(carry + k)), vld1q_u8(borrow + k)));
    }
    for (; k < n_chunks; k++) actions[k] = (actions[k] | carry[k]) & ~borrow[k];
  }

done:
  TK_CARRY_FREE(borrow);
  TK_CARRY_FREE(carry);
}

#else

static inline int tk_automata_carry_zero (const uint8_t *buf, uint64_t n) {
  const tk_u64a *p = (const tk_u64a *)buf;
  uint64_t acc = 0;
  uint64_t n8 = n / 8;
  for (uint64_t k = 0; k < n8; k++)
    acc |= p[k];
  for (uint64_t k = n8 * 8; k < n; k++)
    acc |= buf[k];
  return !acc;
}

#define TK_AUT_LOOP8_INC(plane, carry, n) do { \
  tk_u64a *p_ = (tk_u64a *)(plane); tk_u64a *c_ = (tk_u64a *)(carry); \
  uint64_t n8_ = (n) / 8; \
  for (uint64_t i_ = 0; i_ < n8_; i_++) { \
    uint64_t vo_ = p_[i_], vc_ = c_[i_]; \
    p_[i_] = vo_ ^ vc_; c_[i_] = vo_ & vc_; \
  } \
  for (uint64_t i_ = n8_ * 8; i_ < (n); i_++) { \
    uint8_t o_ = (plane)[i_]; (plane)[i_] = o_ ^ (carry)[i_]; (carry)[i_] = o_ & (carry)[i_]; \
  } \
} while(0)

#define TK_AUT_LOOP8_DEC(plane, borrow, n) do { \
  tk_u64a *p_ = (tk_u64a *)(plane); tk_u64a *c_ = (tk_u64a *)(borrow); \
  uint64_t n8_ = (n) / 8; \
  for (uint64_t i_ = 0; i_ < n8_; i_++) { \
    uint64_t vo_ = p_[i_], vc_ = c_[i_]; \
    p_[i_] = vo_ ^ vc_; c_[i_] = ~vo_ & vc_; \
  } \
  for (uint64_t i_ = n8_ * 8; i_ < (n); i_++) { \
    uint8_t o_ = (plane)[i_]; (plane)[i_] = o_ ^ (borrow)[i_]; (borrow)[i_] = ~o_ & (borrow)[i_]; \
  } \
} while(0)

#define TK_AUT_LOOP8_OR(plane, src, n) do { \
  tk_u64a *p_ = (tk_u64a *)(plane); const tk_u64a *s_ = (const tk_u64a *)(src); \
  uint64_t n8_ = (n) / 8; \
  for (uint64_t i_ = 0; i_ < n8_; i_++) p_[i_] |= s_[i_]; \
  for (uint64_t i_ = n8_ * 8; i_ < (n); i_++) (plane)[i_] |= (src)[i_]; \
} while(0)

#define TK_AUT_LOOP8_ANDNOT(plane, src, n) do { \
  tk_u64a *p_ = (tk_u64a *)(plane); const tk_u64a *s_ = (const tk_u64a *)(src); \
  uint64_t n8_ = (n) / 8; \
  for (uint64_t i_ = 0; i_ < n8_; i_++) p_[i_] &= ~s_[i_]; \
  for (uint64_t i_ = n8_ * 8; i_ < (n); i_++) (plane)[i_] &= ~(src)[i_]; \
} while(0)

#define TK_AUT_LOOP8_INIT_ANDNOT(dst, a, b, n) do { \
  tk_u64a *d_ = (tk_u64a *)(dst); \
  const tk_u64a *a_ = (const tk_u64a *)(a); const tk_u64a *b_ = (const tk_u64a *)(b); \
  uint64_t n8_ = (n) / 8; \
  for (uint64_t i_ = 0; i_ < n8_; i_++) d_[i_] = ~a_[i_] & ~b_[i_]; \
  for (uint64_t i_ = n8_ * 8; i_ < (n); i_++) (dst)[i_] = ~(a)[i_] & ~(b)[i_]; \
} while(0)

#define TK_AUT_LOOP8_FUSED(plane, carry, borrow, n) do { \
  tk_u64a *fp_ = (tk_u64a *)(plane); \
  tk_u64a *fc_ = (tk_u64a *)(carry); \
  tk_u64a *fd_ = (tk_u64a *)(borrow); \
  uint64_t fn8_ = (n) / 8; \
  for (uint64_t fi_ = 0; fi_ < fn8_; fi_++) { \
    uint64_t vo_ = fp_[fi_], vc_ = fc_[fi_], vd_ = fd_[fi_]; \
    fp_[fi_] = vo_ ^ vc_ ^ vd_; \
    fc_[fi_] = vo_ & vc_; \
    fd_[fi_] = ~vo_ & vd_; \
  } \
  for (uint64_t fi_ = fn8_ * 8; fi_ < (n); fi_++) { \
    uint8_t o_ = (plane)[fi_], c_ = (carry)[fi_], d_ = (borrow)[fi_]; \
    (plane)[fi_] = o_ ^ c_ ^ d_; \
    (carry)[fi_] = o_ & c_; \
    (borrow)[fi_] = ~o_ & d_; \
  } \
} while(0)

#define TK_AUT_LOOP8_SAT_FUSED(plane, carry, borrow, n) do { \
  tk_u64a *fp_ = (tk_u64a *)(plane); \
  const tk_u64a *fc_ = (const tk_u64a *)(carry); \
  const tk_u64a *fd_ = (const tk_u64a *)(borrow); \
  uint64_t fn8_ = (n) / 8; \
  for (uint64_t fi_ = 0; fi_ < fn8_; fi_++) fp_[fi_] = (fp_[fi_] | fc_[fi_]) & ~fd_[fi_]; \
  for (uint64_t fi_ = fn8_ * 8; fi_ < (n); fi_++) (plane)[fi_] = ((plane)[fi_] | (carry)[fi_]) & ~(borrow)[fi_]; \
} while(0)

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
    TK_AUT_LOOP8_INC(counts_planes[b], carry, n_chunks);
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }
  TK_AUT_LOOP8_INC(actions, carry, n_chunks);
  if (tk_automata_carry_zero(carry, n_chunks)) goto done;

  for (uint64_t b = 0; b < m; b++)
    TK_AUT_LOOP8_OR(counts_planes[b], carry, n_chunks);
  TK_AUT_LOOP8_OR(actions, carry, n_chunks);

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
    TK_AUT_LOOP8_DEC(counts_planes[b], borrow, n_chunks);
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }
  TK_AUT_LOOP8_DEC(actions, borrow, n_chunks);
  if (tk_automata_carry_zero(borrow, n_chunks)) goto done;

  for (uint64_t b = 0; b < m; b++)
    TK_AUT_LOOP8_ANDNOT(counts_planes[b], borrow, n_chunks);
  TK_AUT_LOOP8_ANDNOT(actions, borrow, n_chunks);

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
  TK_AUT_LOOP8_INIT_ANDNOT(borrow, input, actions, n_chunks);

  for (uint64_t b = 0; b < m; b++) {
    TK_AUT_LOOP8_DEC(counts_planes[b], borrow, n_chunks);
    if (tk_automata_carry_zero(borrow, n_chunks)) goto done;
  }
  TK_AUT_LOOP8_DEC(actions, borrow, n_chunks);
  if (tk_automata_carry_zero(borrow, n_chunks)) goto done;

  for (uint64_t b = 0; b < m; b++)
    TK_AUT_LOOP8_ANDNOT(counts_planes[b], borrow, n_chunks);
  TK_AUT_LOOP8_ANDNOT(actions, borrow, n_chunks);

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
  TK_AUT_LOOP8_INIT_ANDNOT(carry, input, actions, n_chunks);

  for (uint64_t b = 0; b < m; b++) {
    TK_AUT_LOOP8_INC(counts_planes[b], carry, n_chunks);
    if (tk_automata_carry_zero(carry, n_chunks)) goto done;
  }
  TK_AUT_LOOP8_INC(actions, carry, n_chunks);
  if (tk_automata_carry_zero(carry, n_chunks)) goto done;

  for (uint64_t b = 0; b < m; b++)
    TK_AUT_LOOP8_OR(counts_planes[b], carry, n_chunks);
  TK_AUT_LOOP8_OR(actions, carry, n_chunks);

done:
  TK_CARRY_FREE(carry);
}

static inline void tk_automata_inc_and_dec_not_excluded (
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
  TK_CARRY_ALLOC(borrow, n_chunks);
  memcpy(carry, input, n_chunks);
  TK_AUT_LOOP8_INIT_ANDNOT(borrow, input, actions, n_chunks);

  for (uint64_t b = 0; b < m; b++) {
    TK_AUT_LOOP8_FUSED(counts_planes[b], carry, borrow, n_chunks);
    if (tk_automata_carry_zero(carry, n_chunks) && tk_automata_carry_zero(borrow, n_chunks))
      goto done;
  }
  TK_AUT_LOOP8_FUSED(actions, carry, borrow, n_chunks);
  if (tk_automata_carry_zero(carry, n_chunks) && tk_automata_carry_zero(borrow, n_chunks))
    goto done;

  for (uint64_t b = 0; b < m; b++)
    TK_AUT_LOOP8_SAT_FUSED(counts_planes[b], carry, borrow, n_chunks);
  TK_AUT_LOOP8_SAT_FUSED(actions, carry, borrow, n_chunks);

done:
  TK_CARRY_FREE(borrow);
  TK_CARRY_FREE(carry);
}

#endif

#endif
