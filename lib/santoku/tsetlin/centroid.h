#ifndef TK_CENTROID_H
#define TK_CENTROID_H

#include <santoku/cvec.h>

typedef struct {
  uint64_t n_chunks;
  int32_t *votes;
  char *code;
  uint8_t tail_mask;
  uint64_t size;
} tk_centroid_t;

static inline tk_centroid_t *tk_centroid_create (
  lua_State *L,
  uint64_t n_chunks,
  uint8_t tail_mask
) {
  tk_centroid_t *centroid = tk_malloc(L, sizeof(tk_centroid_t));
  if (!centroid) return NULL;
  centroid->n_chunks = n_chunks;
  centroid->tail_mask = tail_mask;
  centroid->size = 0;
  uint64_t n_votes = n_chunks * TK_CVEC_BITS;
  centroid->votes = tk_malloc_aligned(L, sizeof(int32_t) * n_votes, TK_CVEC_BITS);
  if (!centroid->votes) {
    free(centroid);
    return NULL;
  }
  centroid->code = (char *)tk_malloc_aligned(L, n_chunks, TK_CVEC_BITS);
  if (!centroid->code) {
    free(centroid->votes);
    free(centroid);
    return NULL;
  }
  for (uint64_t i = 0; i < n_votes; i++)
    centroid->votes[i] = -1;
  for (uint64_t i = 0; i < n_chunks; i++)
    centroid->code[i] = 0;
  return centroid;
}

static inline void tk_centroid_destroy (tk_centroid_t *centroid) {
  if (!centroid) return;
  if (centroid->votes) {
    free(centroid->votes);
    centroid->votes = NULL;
  }
  if (centroid->code) {
    free(centroid->code);
    centroid->code = NULL;
  }
  free(centroid);
}

static inline void tk_centroid_add_member (
  tk_centroid_t *centroid,
  char *member_code,
  uint64_t code_chunks
) {
  centroid->size++;
  bool is_first = (centroid->size == 1);
  for (uint64_t chunk = 0; chunk < code_chunks; chunk++) {
    uint8_t mask = (chunk == code_chunks - 1) ? centroid->tail_mask : 0xFF;
    uint8_t member_bits = ((uint8_t *)member_code)[chunk];
    int32_t *votes = &centroid->votes[chunk * TK_CVEC_BITS];
    uint8_t *code_byte = &((uint8_t *)centroid->code)[chunk];
    for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
      if (member_bits & (1 << b)) {
        votes[b]++;
      } else if (!is_first) {
        votes[b]--;
      }
      if (votes[b] >= 0) {
        *code_byte |= (1 << b);
      } else {
        *code_byte &= ~(1 << b);
      }
    }
    ((uint8_t *)centroid->code)[chunk] &= mask;
  }
}

static inline bool tk_centroid_merge (
  tk_centroid_t *dst,
  tk_centroid_t *src
) {
  bool changed = false;
  for (uint64_t chunk = 0; chunk < dst->n_chunks; chunk++) {
    uint8_t mask = (chunk == dst->n_chunks - 1) ? dst->tail_mask : 0xFF;
    uint8_t old_code = ((uint8_t *)dst->code)[chunk];
    int32_t *votes_dst = &dst->votes[chunk * TK_CVEC_BITS];
    int32_t *votes_src = &src->votes[chunk * TK_CVEC_BITS];
    uint8_t *code_byte = &((uint8_t *)dst->code)[chunk];
    for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
      votes_dst[b] += votes_src[b];
      if (votes_dst[b] >= 0) {
        *code_byte |= (1 << b);
      } else {
        *code_byte &= ~(1 << b);
      }
    }
    ((uint8_t *)dst->code)[chunk] &= mask;
    if (((uint8_t *)dst->code)[chunk] != old_code)
      changed = true;
  }
  dst->size += src->size;
  return changed;
}

static inline char *tk_centroid_code (tk_centroid_t *centroid) {
  return centroid->code;
}

#endif
