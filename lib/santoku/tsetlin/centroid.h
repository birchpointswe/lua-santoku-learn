#ifndef TK_CENTROID_H
#define TK_CENTROID_H

#include <santoku/cvec.h>

typedef struct {
  uint64_t n_items;
  uint64_t n_dims;
  uint64_t n_chunks;
  int32_t *votes;
  char *code;
  uint8_t tail_mask;
  uint64_t size;
} tk_centroid_t;

static inline tk_centroid_t *tk_centroid_create_batch (
  lua_State *L,
  uint64_t n_items,
  uint64_t n_chunks,
  uint8_t tail_mask
) {
  tk_centroid_t *centroid = tk_malloc(L, sizeof(tk_centroid_t));
  if (!centroid) return NULL;

  centroid->n_items = n_items;
  centroid->n_dims = n_chunks * TK_CVEC_BITS;
  centroid->n_chunks = n_chunks;
  centroid->tail_mask = tail_mask;
  centroid->size = 0;

  uint64_t total_votes = n_items * centroid->n_dims;
  centroid->votes = tk_malloc_aligned(L, sizeof(int32_t) * total_votes, TK_CVEC_BITS);
  if (!centroid->votes) {
    free(centroid);
    return NULL;
  }

  uint64_t total_bytes = n_items * n_chunks;
  centroid->code = (char *)tk_malloc_aligned(L, total_bytes, TK_CVEC_BITS);
  if (!centroid->code) {
    free(centroid->votes);
    free(centroid);
    return NULL;
  }

  for (uint64_t i = 0; i < total_votes; i++)
    centroid->votes[i] = 0;
  for (uint64_t i = 0; i < total_bytes; i++)
    centroid->code[i] = 0;

  return centroid;
}

static inline tk_centroid_t *tk_centroid_create (
  lua_State *L,
  uint64_t n_chunks,
  uint8_t tail_mask
) {
  return tk_centroid_create_batch(L, 1, n_chunks, tail_mask);
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

static inline void tk_centroid_clear (tk_centroid_t *centroid) {
  if (!centroid) return;
  uint64_t total_votes = centroid->n_items * centroid->n_dims;
  for (uint64_t i = 0; i < total_votes; i++) {
    centroid->votes[i] = 0;
  }
  uint64_t total_bytes = centroid->n_items * centroid->n_chunks;
  for (uint64_t i = 0; i < total_bytes; i++) {
    centroid->code[i] = 0;
  }
  centroid->size = 0;
}

static inline void tk_centroid_clear_item (
  tk_centroid_t *centroid,
  uint64_t item_idx
) {
  if (!centroid || item_idx >= centroid->n_items) return;

  int32_t *item_votes = centroid->votes + item_idx * centroid->n_dims;
  for (uint64_t i = 0; i < centroid->n_dims; i++) {
    item_votes[i] = 0;
  }

  char *item_code = centroid->code + item_idx * centroid->n_chunks;
  for (uint64_t i = 0; i < centroid->n_chunks; i++) {
    item_code[i] = 0;
  }
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

static inline void tk_centroid_add_votes (
  tk_centroid_t *centroid,
  uint64_t item_idx,
  const tk_ivec_t *weights
) {
  if (item_idx >= centroid->n_items) return;

  int32_t *item_votes = centroid->votes + item_idx * centroid->n_dims;
  for (uint64_t i = 0; i < weights->n && i < centroid->n_dims; i++) {
    item_votes[i] += (int32_t)weights->a[i];
  }
}

static inline void tk_centroid_recompute (
  tk_centroid_t *centroid,
  uint64_t item_idx
) {
  if (item_idx >= centroid->n_items) return;

  int32_t *item_votes = centroid->votes + item_idx * centroid->n_dims;
  char *item_code = centroid->code + item_idx * centroid->n_chunks;

  for (uint64_t chunk = 0; chunk < centroid->n_chunks; chunk++) {
    uint8_t code_byte = 0;
    for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
      uint64_t bit = chunk * TK_CVEC_BITS + b;
      if (bit >= centroid->n_dims) break;
      if (item_votes[bit] >= 0) {
        code_byte |= (1 << b);
      }
    }
    if (chunk == centroid->n_chunks - 1) {
      code_byte &= centroid->tail_mask;
    }
    item_code[chunk] = (char)code_byte;
  }
}

static inline void tk_centroid_recompute_all (
  tk_centroid_t *centroid
) {
  for (uint64_t item = 0; item < centroid->n_items; item++) {
    tk_centroid_recompute(centroid, item);
  }
}

#endif
