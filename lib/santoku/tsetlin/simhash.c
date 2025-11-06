#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/cvec.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/centroid.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

#define TK_SIMHASH_SCALE 1e6

static inline uint64_t murmur3_fmix64 (uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

static inline void hash_feature_to_weights (
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
    uint64_t hash64 = murmur3_fmix64(seed);
    uint64_t bits_to_use = bits_remaining < 64 ? bits_remaining : 64;
    for (uint64_t b = 0; b < bits_to_use; b++) {
      uint64_t bit_idx = bit_start + bit_offset + b;
      output_weights->a[bit_idx] += (hash64 & (1ULL << b)) ? weight : -weight;
    }
    bits_remaining -= bits_to_use;
    bit_offset += bits_to_use;
    seed = hash64;
  }
}

static inline void write_bits_to_uint64_buffer (
  uint64_t *buffer_chunks,
  uint64_t global_bit_offset,
  uint64_t n_bits,
  uint64_t value
) {
  if (n_bits == 0) return;
  uint64_t value_mask = (n_bits >= 64) ? 0xFFFFFFFFFFFFFFFFULL : (1ULL << n_bits) - 1;
  uint64_t bits_to_write = value & value_mask;
  uint64_t current_global_bit = global_bit_offset;
  uint64_t bits_written = 0;
  while (bits_written < n_bits) {
    uint64_t chunk_idx = current_global_bit / 64;
    uint64_t bit_in_chunk_start = current_global_bit % 64;
    uint64_t bits_can_write_in_chunk = 64 - bit_in_chunk_start;
    uint64_t bits_left_to_write = n_bits - bits_written;
    uint64_t bits_to_write_this_op = (bits_left_to_write < bits_can_write_in_chunk) ? bits_left_to_write : bits_can_write_in_chunk;
    uint64_t value_op_mask = (bits_to_write_this_op >= 64) ? 0xFFFFFFFFFFFFFFFFULL : (1ULL << bits_to_write_this_op) - 1;
    uint64_t value_bits = (bits_written >= 64) ? 0 : ((bits_to_write >> bits_written) & value_op_mask);
    uint64_t buffer_clear_mask = (bits_to_write_this_op >= 64) ? 0 : ~(value_op_mask << bit_in_chunk_start);
    buffer_chunks[chunk_idx] &= buffer_clear_mask;
    buffer_chunks[chunk_idx] |= (value_bits << bit_in_chunk_start);
    bits_written += bits_to_write_this_op;
    current_global_bit += bits_to_write_this_op;
  }
}

static inline int tm_minhash (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  uint64_t n_bits = tk_lua_checkunsigned(L, 2, "n_bits");
  int64_t hashed_ranks = -1;
  if (lua_gettop(L) >= 3 && !lua_isnil(L, 3))
    hashed_ranks = (int64_t)tk_lua_checkunsigned(L, 3, "hashed_ranks");
  bool weighted = tk_lua_optboolean(L, 4, false, "weighted");
  uint64_t n_ranks_to_hash = (hashed_ranks >= 0 && hashed_ranks < (int64_t)inv->n_ranks) ? (uint64_t)hashed_ranks : inv->n_ranks;
  if (n_ranks_to_hash == 0)
    return luaL_error(L, "n_ranks_to_hash must be > 0");
  uint64_t *rank_bit_start = malloc((n_ranks_to_hash + 1) * sizeof(uint64_t));
  uint64_t *rank_n_bits = malloc(n_ranks_to_hash * sizeof(uint64_t));
  if (!rank_bit_start || !rank_n_bits) {
    free(rank_bit_start);
    free(rank_n_bits);
    return luaL_error(L, "out of memory");
  }
  if (inv->rank_weights && n_ranks_to_hash > 1) {
    double total_weight = 0.0;
    for (uint64_t r = 0; r < n_ranks_to_hash; r++)
      total_weight += inv->rank_weights->a[r];
    if (total_weight > DBL_MIN) {
      uint64_t bits_allocated = 0;
      for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
        double proportion = inv->rank_weights->a[r] / total_weight;
        uint64_t bits = (uint64_t)(proportion * n_bits);
        if (bits < 1) bits = 1;
        rank_n_bits[r] = bits;
        bits_allocated += bits;
      }
      if (bits_allocated != n_bits) {
        int64_t diff = (int64_t)n_bits - (int64_t)bits_allocated;
        int64_t new_bits = (int64_t)rank_n_bits[0] + diff;
        if (new_bits < 1) new_bits = 1;
        rank_n_bits[0] = (uint64_t)new_bits;
      }
      rank_bit_start[0] = 0;
      for (uint64_t r = 0; r < n_ranks_to_hash; r++)
        rank_bit_start[r + 1] = rank_bit_start[r] + rank_n_bits[r];
    } else {
      uint64_t bits_per_rank = n_bits / n_ranks_to_hash;
      uint64_t extra_bits = n_bits % n_ranks_to_hash;
      rank_bit_start[0] = 0;
      for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
        rank_n_bits[r] = bits_per_rank + (r < extra_bits ? 1 : 0);
        rank_bit_start[r + 1] = rank_bit_start[r] + rank_n_bits[r];
      }
    }
  } else {
    uint64_t bits_per_rank = n_bits / n_ranks_to_hash;
    uint64_t extra_bits = n_bits % n_ranks_to_hash;
    rank_bit_start[0] = 0;
    for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
      rank_n_bits[r] = bits_per_rank + (r < extra_bits ? 1 : 0);
      rank_bit_start[r + 1] = rank_bit_start[r] + rank_n_bits[r];
    }
  }
  uint64_t n_uint64s = (n_bits + 63) / 64;
  uint64_t n_chunks = n_uint64s * 8;
  uint64_t tail_bits = n_bits % TK_CVEC_BITS;
  uint8_t tail_mask = tail_bits ? ((1 << tail_bits) - 1) : 0xFF;
  uint64_t max_sid = inv->node_offsets->n - 1;
  uint64_t n_valid = 0;
  for (uint64_t sid = 0; sid < max_sid; sid++)
    if (inv->sid_to_uid->a[sid] >= 0)
      n_valid++;
  int64_t *sid_to_outpos = malloc(max_sid * sizeof(int64_t));
  if (!sid_to_outpos) {
    free(rank_bit_start);
    free(rank_n_bits);
    return luaL_error(L, "out of memory");
  }
  int64_t out_idx = 0;
  for (uint64_t sid = 0; sid < max_sid; sid++)
    if (inv->sid_to_uid->a[sid] >= 0)
      sid_to_outpos[sid] = out_idx++;
    else
      sid_to_outpos[sid] = -1;
  size_t total_bytes = n_valid * n_chunks;
  tk_cvec_t *output = tk_cvec_create(L, total_bytes, 0, 0);
  output->n = total_bytes;
  memset(output->a, 0, total_bytes);
  #pragma omp parallel
  {
    double *min_scaled_hashes = malloc(n_ranks_to_hash * sizeof(double));
    double *rank_totals = malloc(inv->n_ranks * sizeof(double));
    if (min_scaled_hashes && rank_totals) {
    #pragma omp for schedule(static)
    for (uint64_t sid = 0; sid < max_sid; sid++) {
      if (inv->sid_to_uid->a[sid] < 0)
        continue;
      uint64_t outpos = (uint64_t)sid_to_outpos[sid];
      int64_t start = inv->node_offsets->a[sid];
      int64_t end = inv->node_offsets->a[sid + 1];
      for (uint64_t r = 0; r < n_ranks_to_hash; r++)
        min_scaled_hashes[r] = DBL_MAX;
      for (uint64_t r = 0; r < inv->n_ranks; r++)
        rank_totals[r] = 0.0;
      for (int64_t i = start; i < end; i++) {
        int64_t feature_id = inv->node_bits->a[i];
        int64_t rank = inv->ranks ? inv->ranks->a[feature_id] : 0;
        if (hashed_ranks >= 0 && rank >= hashed_ranks)
          continue;
        double feature_weight = (weighted && inv->weights) ? inv->weights->a[feature_id] : 1.0;
        if (rank >= 0 && rank < (int64_t)inv->n_ranks)
          rank_totals[rank] += feature_weight;
      }
      for (int64_t i = start; i < end; i++) {
        int64_t feature_id = inv->node_bits->a[i];
        int64_t rank = inv->ranks ? inv->ranks->a[feature_id] : 0;
        if (hashed_ranks >= 0 && rank >= hashed_ranks)
          continue;
        if (rank >= 0 && rank < (int64_t)n_ranks_to_hash) {
          double feature_weight = (weighted && inv->weights) ? inv->weights->a[feature_id] : 1.0;
          double normalized_weight = (rank_totals[rank] > 0.0) ? feature_weight / rank_totals[rank] : 0.0;
          double rank_weight = (inv->rank_weights && rank >= 0 && rank < (int64_t)inv->n_ranks) ? inv->rank_weights->a[rank] : 1.0;
          double final_weight = normalized_weight * rank_weight;
          uint64_t hash = murmur3_fmix64((uint64_t)feature_id);
          double scaled_hash = (double)hash / (final_weight + DBL_MIN);
          if (scaled_hash < min_scaled_hashes[rank])
            min_scaled_hashes[rank] = scaled_hash;
        }
      }
      uint64_t *doc_code_ptr = (uint64_t*)(output->a + outpos * n_chunks);
      for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
        uint64_t bit_start = rank_bit_start[r];
        uint64_t n_bits_for_rank = rank_n_bits[r];
        uint64_t min_hash_val;
        if (min_scaled_hashes[r] == DBL_MAX)
          min_hash_val = 0;
        else
          min_hash_val = (uint64_t)min_scaled_hashes[r];
        write_bits_to_uint64_buffer(doc_code_ptr, bit_start, n_bits_for_rank, min_hash_val);
      }
      if (n_chunks > 0)
        ((uint8_t*)doc_code_ptr)[n_chunks - 1] &= tail_mask;
    }
    }
    free(min_scaled_hashes);
    free(rank_totals);
  }
  free(rank_bit_start);
  free(rank_n_bits);
  tk_ivec_t *ids = tk_ivec_create(L, n_valid, 0, 0);
  ids->n = n_valid;
  uint64_t out_i = 0;
  for (uint64_t sid = 0; sid < max_sid; sid++) {
    if (inv->sid_to_uid->a[sid] >= 0) {
      ids->a[out_i++] = inv->sid_to_uid->a[sid];
    }
  }
  free(sid_to_outpos);
  lua_insert(L, -2);
  return 2;
}

static inline int tm_simhash (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  uint64_t n_bits = tk_lua_checkunsigned(L, 2, "n_bits");
  int64_t hashed_ranks = -1;
  if (lua_gettop(L) >= 3 && !lua_isnil(L, 3))
    hashed_ranks = (int64_t)tk_lua_checkunsigned(L, 3, "hashed_ranks");
  bool weighted = tk_lua_optboolean(L, 4, false, "weighted");
  uint64_t n_ranks_to_hash = (hashed_ranks >= 0 && hashed_ranks < (int64_t)inv->n_ranks) ? (uint64_t)hashed_ranks : inv->n_ranks;
  if (n_ranks_to_hash == 0)
    return luaL_error(L, "n_ranks_to_hash must be > 0");
  uint64_t *rank_bit_start = malloc((n_ranks_to_hash + 1) * sizeof(uint64_t));
  uint64_t *rank_n_bits = malloc(n_ranks_to_hash * sizeof(uint64_t));
  if (!rank_bit_start || !rank_n_bits) {
    free(rank_bit_start);
    free(rank_n_bits);
    return luaL_error(L, "out of memory");
  }
  if (inv->rank_weights && n_ranks_to_hash > 1) {
    double total_weight = 0.0;
    for (uint64_t r = 0; r < n_ranks_to_hash; r++)
      total_weight += inv->rank_weights->a[r];
    if (total_weight > DBL_MIN) {
      uint64_t bits_allocated = 0;
      for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
        double proportion = inv->rank_weights->a[r] / total_weight;
        uint64_t bits = (uint64_t)(proportion * n_bits);
        if (bits < 1) bits = 1;
        rank_n_bits[r] = bits;
        bits_allocated += bits;
      }
      if (bits_allocated != n_bits) {
        int64_t diff = (int64_t)n_bits - (int64_t)bits_allocated;
        int64_t new_bits = (int64_t)rank_n_bits[0] + diff;
        if (new_bits < 1) new_bits = 1;
        rank_n_bits[0] = (uint64_t)new_bits;
      }
      rank_bit_start[0] = 0;
      for (uint64_t r = 0; r < n_ranks_to_hash; r++)
        rank_bit_start[r + 1] = rank_bit_start[r] + rank_n_bits[r];
    } else {
      uint64_t bits_per_rank = n_bits / n_ranks_to_hash;
      uint64_t extra_bits = n_bits % n_ranks_to_hash;
      rank_bit_start[0] = 0;
      for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
        rank_n_bits[r] = bits_per_rank + (r < extra_bits ? 1 : 0);
        rank_bit_start[r + 1] = rank_bit_start[r] + rank_n_bits[r];
      }
    }
  } else {
    uint64_t bits_per_rank = n_bits / n_ranks_to_hash;
    uint64_t extra_bits = n_bits % n_ranks_to_hash;
    rank_bit_start[0] = 0;
    for (uint64_t r = 0; r < n_ranks_to_hash; r++) {
      rank_n_bits[r] = bits_per_rank + (r < extra_bits ? 1 : 0);
      rank_bit_start[r + 1] = rank_bit_start[r] + rank_n_bits[r];
    }
  }
  uint64_t n_uint64s = (n_bits + 63) / 64;
  uint64_t n_chunks = n_uint64s * 8;
  uint64_t tail_bits = n_bits % TK_CVEC_BITS;
  uint8_t tail_mask = tail_bits ? ((1 << tail_bits) - 1) : 0xFF;
  uint64_t max_sid = inv->node_offsets->n - 1;
  uint64_t n_valid = 0;
  for (uint64_t sid = 0; sid < max_sid; sid++) {
    if (inv->sid_to_uid->a[sid] >= 0) {
      n_valid++;
    }
  }
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
    if (feature_weights && rank_totals) {
    #pragma omp for schedule(static)
    for (uint64_t sid = 0; sid < max_sid; sid++) {
      if (inv->sid_to_uid->a[sid] < 0) {
        continue;
      }
      uint64_t outpos = (uint64_t)sid_to_outpos[sid];
      int64_t start = inv->node_offsets->a[sid];
      int64_t end = inv->node_offsets->a[sid + 1];
      for (uint64_t i = 0; i < n_bits; i++)
        feature_weights->a[i] = 0;
      for (uint64_t r = 0; r < inv->n_ranks; r++)
        rank_totals[r] = 0.0;
      for (int64_t i = start; i < end; i++) {
        int64_t feature_id = inv->node_bits->a[i];
        int64_t rank = inv->ranks ? inv->ranks->a[feature_id] : 0;
        if (hashed_ranks >= 0 && rank >= hashed_ranks)
          continue;
        double feature_weight = (weighted && inv->weights) ? inv->weights->a[feature_id] : 1.0;
        if (rank >= 0 && rank < (int64_t)inv->n_ranks)
          rank_totals[rank] += feature_weight;
      }
      for (int64_t i = start; i < end; i++) {
        int64_t feature_id = inv->node_bits->a[i];
        int64_t rank = inv->ranks ? inv->ranks->a[feature_id] : 0;
        if (hashed_ranks >= 0 && rank >= hashed_ranks)
          continue;
        if (rank >= 0 && rank < (int64_t)n_ranks_to_hash) {
          double feature_weight = (weighted && inv->weights) ? inv->weights->a[feature_id] : 1.0;
          double normalized_weight = (rank_totals[rank] > 0.0) ? feature_weight / rank_totals[rank] : 0.0;
          double rank_weight = (inv->rank_weights && rank >= 0 && rank < (int64_t)inv->n_ranks) ? inv->rank_weights->a[rank] : 1.0;
          int32_t weight = (int32_t)(normalized_weight * rank_weight * TK_SIMHASH_SCALE);
          uint64_t bit_start = rank_bit_start[rank];
          uint64_t n_bits_for_rank = rank_n_bits[rank];
          hash_feature_to_weights(feature_id, bit_start, n_bits_for_rank, weight, feature_weights);
        }
      }
      tk_centroid_add_votes(centroid, outpos, feature_weights);
      tk_centroid_recompute(centroid, outpos);
    }
    }
    free(rank_totals);
    tk_ivec_destroy(feature_weights);
  }
  free(rank_bit_start);
  free(rank_n_bits);
  tk_ivec_t *ids = tk_ivec_create(L, n_valid, 0, 0);
  ids->n = n_valid;
  uint64_t out_i = 0;
  for (uint64_t sid = 0; sid < max_sid; sid++)
    if (inv->sid_to_uid->a[sid] >= 0)
      ids->a[out_i++] = inv->sid_to_uid->a[sid];
  size_t total_bytes = n_valid * n_chunks;
  tk_cvec_t *output = tk_cvec_create(L, total_bytes, 0, 0);
  output->n = total_bytes;
  memcpy(output->a, centroid->code, total_bytes);
  free(sid_to_outpos);
  return 2;
}

static luaL_Reg tm_simhash_fns[] = {
  { "simhash", tm_simhash },
  { "minhash", tm_minhash },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_simhash(lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tm_simhash_fns, 0);
  return 1;
}
