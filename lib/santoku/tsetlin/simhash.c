#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/cvec.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/centroid.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

static inline void hash_feature_to_weights(
  int64_t feature_id,
  uint64_t n_bits,
  int32_t weight,
  tk_ivec_t *output_weights
) {
  output_weights->n = n_bits;
  uint64_t bits_remaining = n_bits;
  uint64_t bit_offset = 0;
  uint64_t seed = (uint64_t)feature_id;
  while (bits_remaining > 0) {
    uint32_t hash = kh_int64_hash_func(seed);
    uint64_t bits_to_use = bits_remaining < 32 ? bits_remaining : 32;
    for (uint64_t b = 0; b < bits_to_use; b++) {
      uint64_t bit_idx = bit_offset + b;
      output_weights->a[bit_idx] = (hash & (1U << b)) ? weight : -weight;
    }
    bits_remaining -= bits_to_use;
    bit_offset += bits_to_use;
    seed = hash;
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
  uint64_t out_idx = 0;
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

        hash_feature_to_weights(feature_id, n_bits, weight, feature_weights);

        tk_centroid_add_votes(centroid, outpos, feature_weights);
      }

      tk_centroid_recompute(centroid, outpos);
    }

    free(rank_totals);
    tk_ivec_destroy(feature_weights);
  }
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
