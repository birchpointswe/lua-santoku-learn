#ifndef TK_CENTROID_H
#define TK_CENTROID_H

#include <santoku/cvec.h>

// Signed vote-based majority centroid for clustering binary codes
// Uses simple signed int32_t counters for efficient merging
// Action bits represent majority: bit=1 when vote_count >= 0
typedef struct {
  uint64_t n_chunks;      // Number of code chunks
  int32_t *votes;         // Signed vote counts [n_chunks × 32]
  char *code;             // Majority centroid bitmap [n_chunks]
  uint8_t tail_mask;      // Mask for last chunk
  uint64_t size;          // Number of members (for tracking first member)
} tk_centroid_t;

// Create a new centroid
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

  // Allocate vote counters
  uint64_t n_votes = n_chunks * TK_CVEC_BITS;
  centroid->votes = tk_malloc_aligned(L, sizeof(int32_t) * n_votes, TK_CVEC_BITS);
  if (!centroid->votes) {
    free(centroid);
    return NULL;
  }

  // Allocate centroid code
  centroid->code = (char *)tk_malloc_aligned(L, n_chunks, TK_CVEC_BITS);
  if (!centroid->code) {
    free(centroid->votes);
    free(centroid);
    return NULL;
  }

  // Initialize: all votes = -1 (bias toward 0), code = 0x00
  for (uint64_t i = 0; i < n_votes; i++)
    centroid->votes[i] = -1;
  for (uint64_t i = 0; i < n_chunks; i++)
    centroid->code[i] = 0;

  return centroid;
}

// Destroy centroid and free memory
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

// Add a member's code to the centroid
// First member: only vote for set bits (optimization)
// Subsequent: vote for set bits (inc) and against unset bits (dec)
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

    for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
      if (member_bits & (1 << b)) {
        // Member has this bit set - vote for 1
        votes[b]++;
      } else if (!is_first) {
        // Member has this bit unset - vote for 0 (only for subsequent members)
        votes[b]--;
      }

      // Update centroid code based on vote majority
      if (votes[b] >= 0)
        ((uint8_t *)centroid->code)[chunk] |= (1 << b);
      else
        ((uint8_t *)centroid->code)[chunk] &= ~(1 << b);
    }

    ((uint8_t *)centroid->code)[chunk] &= mask;
  }
}

// Merge source centroid into destination
// Returns true if destination centroid code changed
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

    for (uint64_t b = 0; b < TK_CVEC_BITS; b++) {
      // Simple addition - signed zero-centered representation
      votes_dst[b] += votes_src[b];

      // Update centroid code based on combined vote majority
      if (votes_dst[b] >= 0)
        ((uint8_t *)dst->code)[chunk] |= (1 << b);
      else
        ((uint8_t *)dst->code)[chunk] &= ~(1 << b);
    }

    ((uint8_t *)dst->code)[chunk] &= mask;
    if (((uint8_t *)dst->code)[chunk] != old_code)
      changed = true;
  }

  dst->size += src->size;
  return changed;
}

// Get the centroid code (parallel to tk_automata_actions)
static inline char *tk_centroid_code (tk_centroid_t *centroid) {
  return centroid->code;
}

#endif
