#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/cvec.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/centroid.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

// MurmurHash3 64-bit finalizer - excellent avalanche properties
static inline uint64_t murmur3_fmix64(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

static inline void hash_feature_to_weights(
  int64_t feature_id,
  uint64_t bit_start,
  uint64_t n_bits_for_rank,
  int32_t weight,
  tk_ivec_t *output_weights
) {
  uint64_t bits_remaining = n_bits_for_rank;
  uint64_t bit_offset = 0;
  uint64_t seed = (uint64_t)feature_id;

  while (bits_remaining > 0) {
    // Use MurmurHash3 for excellent bit distribution
    uint64_t hash64 = murmur3_fmix64(seed);
    uint64_t bits_to_use = bits_remaining < 64 ? bits_remaining : 64;

    for (uint64_t b = 0; b < bits_to_use; b++) {
      uint64_t bit_idx = bit_start + bit_offset + b;
      output_weights->a[bit_idx] = (hash64 & (1ULL << b)) ? weight : -weight;
    }

    bits_remaining -= bits_to_use;
    bit_offset += bits_to_use;
    seed = hash64;  // Chain with full 64-bit output
  }
}

static inline int tm_encode(lua_State *L) {
  tk_inv_t *inv = tk_inv_peek(L, 1);
  uint64_t n_bits = tk_lua_checkunsigned(L, 2, "n_bits");
  int64_t hashed_ranks = -1;
  if (lua_gettop(L) >= 3 && !lua_isnil(L, 3)) {
    hashed_ranks = (int64_t)tk_lua_checkunsigned(L, 3, "hashed_ranks");
  }
  double scale = 1.0;
  if (inv->rank_weights && inv->n_ranks > 0) {
    double min_rank_weight = DBL_MAX;
    #pragma omp parallel for reduction(min:min_rank_weight)
    for (uint64_t r = 0; r < inv->n_ranks; r++) {
      double w = inv->rank_weights->a[r];
      if (w > 0 && w < min_rank_weight) {
        min_rank_weight = w;
      }
    }
    if (min_rank_weight < DBL_MAX && min_rank_weight > 0) {
      scale = 1.0 / min_rank_weight;
    }
  }
  if (scale < 1.0) scale = 1.0;
  if (scale > 1e6) scale = 1e6;

  // Determine number of ranks to hash
  uint64_t n_ranks_to_hash = (hashed_ranks >= 0 && hashed_ranks < (int64_t)inv->n_ranks)
    ? (uint64_t)hashed_ranks
    : inv->n_ranks;

  // Allocate bits per rank based on rank weights
  uint64_t *rank_bit_start = malloc((n_ranks_to_hash + 1) * sizeof(uint64_t));
  uint64_t *rank_n_bits = malloc(n_ranks_to_hash * sizeof(uint64_t));

  if (inv->rank_weights && n_ranks_to_hash > 1) {
    // Compute total weight for ranks we're hashing
    double total_weight = 0.0;
    for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
      total_weight += inv->rank_weights->a[r];
    }

    // Allocate bits proportional to rank weight
    // Higher weight = more bits (more Hamming distance contribution)
    uint64_t bits_allocated = 0;
    for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
      double proportion = inv->rank_weights->a[r] / total_weight;
      uint64_t bits = (uint64_t)(proportion * n_bits);
      if (bits < 1) bits = 1;  // At least 1 bit per rank
      rank_n_bits[r] = bits;
      bits_allocated += bits;
    }

    // Adjust for rounding errors
    if (bits_allocated != n_bits) {
      int64_t diff = (int64_t)n_bits - (int64_t)bits_allocated;
      // Add/subtract from rank 0 (highest weight, gets the adjustment)
      rank_n_bits[0] = (uint64_t)((int64_t)rank_n_bits[0] + diff);
    }

    // Compute bit ranges
    rank_bit_start[0] = 0;
    for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
      rank_bit_start[r + 1] = rank_bit_start[r] + rank_n_bits[r];
    }
  } else {
    // Single rank or no weights - all bits for rank 0
    rank_bit_start[0] = 0;
    rank_bit_start[1] = n_bits;
    rank_n_bits[0] = n_bits;
  }

  uint64_t n_chunks = (n_bits + TK_CVEC_BITS - 1) / TK_CVEC_BITS;
  uint64_t tail_bits = n_bits % TK_CVEC_BITS;
  uint8_t tail_mask = tail_bits ? ((1 << tail_bits) - 1) : 0xFF;
  uint64_t max_sid = inv->node_offsets->n - 1;

  // Count valid entities
  uint64_t n_valid = 0;
  for (uint64_t sid = 0; sid < max_sid; sid++) {
    if (inv->sid_to_uid->a[sid] >= 0) {
      n_valid++;
    }
  }

  // Create mapping: sid -> output position (only for valid entities)
  int64_t *sid_to_outpos = malloc(max_sid * sizeof(int64_t));
  int64_t out_idx = 0;
  for (uint64_t sid = 0; sid < max_sid; sid++) {
    if (inv->sid_to_uid->a[sid] >= 0) {
      sid_to_outpos[sid] = out_idx++;
    } else {
      sid_to_outpos[sid] = -1;
    }
  }

  tk_centroid_t *centroid = tk_centroid_create_batch(L, n_valid, n_chunks, tail_mask);

  #pragma omp parallel
  {
    tk_ivec_t *feature_weights = tk_ivec_create(NULL, n_bits, 0, 0);
    feature_weights->n = n_bits;
    double *rank_totals = malloc(inv->n_ranks * sizeof(double));

    #pragma omp for schedule(static)
    for (uint64_t sid = 0; sid < max_sid; sid++) {
      // Skip deleted entities
      if (inv->sid_to_uid->a[sid] < 0) {
        continue;
      }

      uint64_t outpos = (uint64_t)sid_to_outpos[sid];
      int64_t start = inv->node_offsets->a[sid];
      int64_t end = inv->node_offsets->a[sid + 1];

      // Clear feature_weights buffer (set all to 0)
      for (uint64_t i = 0; i < n_bits; i++) {
        feature_weights->a[i] = 0;
      }

      for (uint64_t r = 0; r < inv->n_ranks; r++) {
        rank_totals[r] = 0.0;
      }

      for (int64_t i = start; i < end; i++) {
        int64_t feature_id = inv->node_bits->a[i];

        int64_t rank = inv->ranks ? inv->ranks->a[feature_id] : 0;

        if (hashed_ranks >= 0 && rank >= hashed_ranks) {
          continue;
        }

        double feature_weight = inv->weights ? inv->weights->a[feature_id] : 1.0;

        if (rank >= 0 && rank < (int64_t)inv->n_ranks) {
          rank_totals[rank] += feature_weight;
        }
      }

      for (int64_t i = start; i < end; i++) {
        int64_t feature_id = inv->node_bits->a[i];

        int64_t rank = inv->ranks ? inv->ranks->a[feature_id] : 0;

        if (hashed_ranks >= 0 && rank >= hashed_ranks) {
          continue;
        }

        double feature_weight = inv->weights ? inv->weights->a[feature_id] : 1.0;

        double normalized_weight = (rank_totals[rank] > 0.0)
          ? feature_weight / rank_totals[rank]
          : 0.0;

        double rank_weight = (inv->rank_weights && rank >= 0 && rank < (int64_t)inv->n_ranks)
          ? inv->rank_weights->a[rank]
          : 1.0;

        int32_t weight = (int32_t)(normalized_weight * rank_weight * scale);

        // Hash into rank-specific bit range
        uint64_t bit_start = rank_bit_start[rank];
        uint64_t n_bits_for_rank = rank_n_bits[rank];
        hash_feature_to_weights(feature_id, bit_start, n_bits_for_rank, weight, feature_weights);
      }

      // Accumulate all votes at once
      tk_centroid_add_votes(centroid, outpos, feature_weights);
      tk_centroid_recompute(centroid, outpos);
    }

    free(rank_totals);
    tk_ivec_destroy(feature_weights);
  }

  free(rank_bit_start);
  free(rank_n_bits);
  // Return IDs in output order (only valid entities)
  tk_ivec_t *ids = tk_ivec_create(L, n_valid, 0, 0);
  ids->n = n_valid;
  uint64_t out_i = 0;
  for (uint64_t sid = 0; sid < max_sid; sid++) {
    if (inv->sid_to_uid->a[sid] >= 0) {
      ids->a[out_i++] = inv->sid_to_uid->a[sid];
    }
  }

  // Return codes (already packed in centroid)
  size_t total_bytes = n_valid * n_chunks;
  tk_cvec_t *output = tk_cvec_create(L, total_bytes, 0, 0);
  output->n = total_bytes;
  memcpy(output->a, centroid->code, total_bytes);

  free(sid_to_outpos);

  return 2;
}

static luaL_Reg tm_simhash_fns[] = {
  { "encode", tm_encode },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_simhash(lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tm_simhash_fns, 0);
  return 1;
}
