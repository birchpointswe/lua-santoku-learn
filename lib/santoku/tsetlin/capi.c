/*

Copyright (C) 2025 Birch Point SWE (FuzzyPattern TM added)
Copyright (C) 2024 Birch Point SWE (Persist to and restore from disk)
Copyright (C) 2024 Birch Point SWE (Lua integration, train/evaluate)
Copyright (C) 2024 Birch Point SWE (Loss scaling, multi-threading, auto-vectorizer support)
Copyright (C) 2019 Ole-Christoffer Granmo (Original classifier C implementation)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <santoku/iuset.h>
#include <santoku/lua/utils.h>
#include <santoku/tsetlin/automata.h>
#include <santoku/cvec.h>
#include <omp.h>

#ifndef LUA_OK
#define LUA_OK 0
#endif

#define TK_TSETLIN_MT "santoku_tsetlin"

typedef enum {
  TM_CLASSIFIER,
  TM_ENCODER,
} tk_tsetlin_type_t;

typedef struct tk_tsetlin_s {
  tk_tsetlin_type_t type;
  bool has_state;
  bool trained;
  bool destroyed;
  bool reusable;
  bool individualized;
  unsigned int classes;
  unsigned int features;
  unsigned int clauses;
  unsigned int clause_tolerance;
  unsigned int clause_maximum;
  unsigned int target;
  unsigned int state_bits;
  unsigned int include_bits;
  unsigned int input_bits;
  unsigned int input_chunks;
  unsigned int clause_chunks;
  unsigned int class_chunks;
  unsigned int state_chunks;
  unsigned int action_chunks;
  uint8_t tail_mask;
  char *state;
  char *actions;
  tk_automata_t automata;
  double negative;
  double specificity;
  unsigned int *results;
  size_t results_len;
  char *encodings;
  size_t encodings_len;
  uint64_t *feat_sizes;
  uint64_t *input_chunks_ind;
  uint8_t *tail_masks_ind;
  uint64_t *state_offsets;
  uint64_t *actions_offsets;
  tk_automata_t *automata_arr;
} tk_tsetlin_t;

static inline uint8_t tk_tsetlin_calculate (
  tk_tsetlin_t *tm,
  char *input,
  unsigned int *literalsp,
  unsigned int *votesp,
  unsigned int chunk
) {
  uint8_t out = 0;
  const unsigned int input_chunks = tm->input_chunks;
  const unsigned int tolerance = tm->clause_tolerance;
  const uint8_t tail_mask = tm->tail_mask;
  const uint8_t *input_bytes = (const uint8_t *)input;
  if (input_chunks > 1) {
    const unsigned int bulk_bits = (input_chunks - 1) * 8;
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) {
      unsigned int clause = chunk * TK_CVEC_BITS + j;
      const uint8_t *actions = (const uint8_t *)tk_automata_actions(&tm->automata, clause);
      uint64_t literals_64, failed_64;
      tk_cvec_bits_popcount_andnot_serial(actions, input_bytes, bulk_bits, &literals_64, &failed_64);
      unsigned int literals = (unsigned int)literals_64;
      unsigned int failed = (unsigned int)failed_64;
      const uint8_t last_act = actions[input_chunks - 1] & tail_mask;
      const uint8_t last_in = input_bytes[input_chunks - 1];
      literals += (unsigned int)__builtin_popcount(last_act);
      failed += (unsigned int)__builtin_popcount(last_act & ~last_in);
      long int votes;
      if (literals == 0) {
        votes = (long int)tolerance;
      } else {
        votes = (literals < tolerance ? (long int)literals : (long int)tolerance) - (long int)failed;
        if (votes < 0)
          votes = 0;
      }
      if (votes > 0)
        out |= (1U << j);
      literalsp[j] = literals;
      votesp[j] = (unsigned int)votes;
    }
  } else {
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) {
      unsigned int clause = chunk * TK_CVEC_BITS + j;
      const uint8_t *actions = (const uint8_t *)tk_automata_actions(&tm->automata, clause);
      const uint8_t last_act = actions[0] & tail_mask;
      const uint8_t last_in = input_bytes[0];
      unsigned int literals = (unsigned int)__builtin_popcount(last_act);
      unsigned int failed = (unsigned int)__builtin_popcount(last_act & ~last_in);
      long int votes;
      if (literals == 0) {
        votes = (long int)tolerance;
      } else {
        votes = (literals < tolerance ? (long int)literals : (long int)tolerance) - (long int)failed;
        if (votes < 0)
          votes = 0;
      }
      if (votes > 0)
        out |= (1U << j);
      literalsp[j] = literals;
      votesp[j] = (unsigned int)votes;
    }
  }
  return out;
}

static inline long int tk_tsetlin_sums (
  tk_tsetlin_t *tm,
  uint8_t out,
  unsigned int *votes
) {
  long int sum = 0;
  for (unsigned int j = 0; j < TK_CVEC_BITS; j += 2)
    if (out & (1 << j))
      sum += (long int) votes[j];
  for (unsigned int j = 1; j < TK_CVEC_BITS; j += 2)
    if (out & (1 << j))
      sum -= (long int) votes[j];
  return sum;
}

static inline void apply_feedback (
  tk_tsetlin_t *tm,
  unsigned int clause_idx,
  unsigned int chunk,
  char *input,
  unsigned int *literals,
  unsigned int *votes,
  bool positive_feedback,
  unsigned int thread
) {
  unsigned int features = tm->features;
  unsigned int max_literals = tm->clause_maximum;
  unsigned int input_chunks = tm->input_chunks;
  double specificity = tm->specificity;
  bool output = votes[clause_idx] > 0;
  unsigned int clause_id = chunk * TK_CVEC_BITS + clause_idx;
  if (positive_feedback) {
    if (output) {
      if (literals[clause_idx] <= max_literals)
        tk_automata_inc(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
      tk_automata_dec_not_excluded(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
    } else {
      unsigned int s = (2 * features) / specificity;
      unsigned int input_bits = tm->input_bits;
      for (unsigned int r = 0; r < s; r ++) {
        unsigned int random_input_bit = tk_fast_random() % input_bits;
        unsigned int random_chunk = random_input_bit / 8;
        uint8_t random_mask = 1 << (random_input_bit % 8);
        tk_automata_dec_byte(&tm->automata, clause_id, random_chunk, random_mask);
      }
    }
  } else {
    if (output) {
      tk_automata_inc_not_excluded(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
    }
  }
}

static inline void tm_update (
  tk_tsetlin_t *tm,
  char *input,
  uint8_t out,
  unsigned int *literals,
  unsigned int *votes,
  unsigned int sample,
  unsigned int sample_class,
  unsigned int target_vote,
  unsigned int chunk_class,
  unsigned int chunk,
  unsigned int thread
) {
  double negative_sampling = tm->negative;
  if (chunk_class != sample_class && tk_fast_chance(1 - negative_sampling))
    return;
  long int vote_target = (long int) tm->target;
  long int chunk_vote = tk_tsetlin_sums(tm, out, votes);
  chunk_vote = (chunk_vote > vote_target) ? vote_target : chunk_vote;
  chunk_vote = (chunk_vote < -vote_target) ? -vote_target : chunk_vote;
  double p;
  if (chunk_class == sample_class && target_vote)
    p = (double) (vote_target - chunk_vote) / (2.0 * vote_target);
  else
    p = (double) (vote_target + chunk_vote) / (2.0 * vote_target);
  for (unsigned int j = 0; j < TK_CVEC_BITS; j += 2) {
    unsigned int pos_clause = j;
    unsigned int neg_clause = j + 1;
    if (neg_clause >= TK_CVEC_BITS)
      break;
    if (chunk_class == sample_class && target_vote) {
      if (tk_fast_chance(p))
        apply_feedback(tm, pos_clause, chunk, input, literals, votes, true, thread);
      if (tk_fast_chance(p))
        apply_feedback(tm, neg_clause, chunk, input, literals, votes, false, thread);
    } else {
      if (tk_fast_chance(p))
        apply_feedback(tm, neg_clause, chunk, input, literals, votes, true, thread);
      if (tk_fast_chance(p))
        apply_feedback(tm, pos_clause, chunk, input, literals, votes, false, thread);
    }
  }
}

static inline void tk_tsetlin_init_shuffle (
  unsigned int *shuffle,
  unsigned int n
) {
  for (unsigned int i = 0; i < n; i ++) {
    shuffle[i] = i;
    unsigned int j = i == 0 ? 0 : tk_fast_random() % (i + 1);
    unsigned int t = shuffle[i];
    shuffle[i] = shuffle[j];
    shuffle[j] = t;
  }
}

static inline uint8_t tk_tsetlin_calculate_ind (
  tk_automata_t *aut,
  char *input,
  unsigned int *literalsp,
  unsigned int *votesp,
  unsigned int chunk,
  uint64_t input_chunks,
  uint8_t tail_mask,
  unsigned int tolerance
) {
  uint8_t out = 0;
  const uint8_t *input_bytes = (const uint8_t *)input;
  if (input_chunks > 1) {
    const unsigned int bulk_bits = (unsigned int)(input_chunks - 1) * 8;
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) {
      unsigned int clause = chunk * TK_CVEC_BITS + j;
      const uint8_t *actions = (const uint8_t *)tk_automata_actions(aut, clause);
      uint64_t literals_64, failed_64;
      tk_cvec_bits_popcount_andnot_serial(actions, input_bytes, bulk_bits, &literals_64, &failed_64);
      unsigned int literals = (unsigned int)literals_64;
      unsigned int failed = (unsigned int)failed_64;
      const uint8_t last_act = actions[input_chunks - 1] & tail_mask;
      const uint8_t last_in = input_bytes[input_chunks - 1];
      literals += (unsigned int)__builtin_popcount(last_act);
      failed += (unsigned int)__builtin_popcount(last_act & ~last_in);
      long int votes;
      if (literals == 0) {
        votes = (long int)tolerance;
      } else {
        votes = (literals < tolerance ? (long int)literals : (long int)tolerance) - (long int)failed;
        if (votes < 0) votes = 0;
      }
      if (votes > 0) out |= (1U << j);
      literalsp[j] = literals;
      votesp[j] = (unsigned int)votes;
    }
  } else {
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) {
      unsigned int clause = chunk * TK_CVEC_BITS + j;
      const uint8_t *actions = (const uint8_t *)tk_automata_actions(aut, clause);
      const uint8_t last_act = actions[0] & tail_mask;
      const uint8_t last_in = input_bytes[0];
      unsigned int literals = (unsigned int)__builtin_popcount(last_act);
      unsigned int failed = (unsigned int)__builtin_popcount(last_act & ~last_in);
      long int votes;
      if (literals == 0) {
        votes = (long int)tolerance;
      } else {
        votes = (literals < tolerance ? (long int)literals : (long int)tolerance) - (long int)failed;
        if (votes < 0) votes = 0;
      }
      if (votes > 0) out |= (1U << j);
      literalsp[j] = literals;
      votesp[j] = (unsigned int)votes;
    }
  }
  return out;
}

static inline long int tk_tsetlin_sums_ind (uint8_t out, unsigned int *votes) {
  long int sum = 0;
  for (unsigned int j = 0; j < TK_CVEC_BITS; j += 2)
    if (out & (1 << j)) sum += (long int) votes[j];
  for (unsigned int j = 1; j < TK_CVEC_BITS; j += 2)
    if (out & (1 << j)) sum -= (long int) votes[j];
  return sum;
}

static inline void apply_feedback_ind (
  tk_automata_t *aut,
  unsigned int clause_idx,
  unsigned int chunk,
  char *input,
  unsigned int *literals,
  unsigned int *votes,
  bool positive_feedback,
  uint64_t features,
  uint64_t input_chunks,
  double specificity,
  unsigned int max_literals
) {
  bool output = votes[clause_idx] > 0;
  unsigned int clause_id = chunk * TK_CVEC_BITS + clause_idx;
  if (positive_feedback) {
    if (output) {
      if (literals[clause_idx] <= max_literals)
        tk_automata_inc(aut, clause_id, (uint8_t*)input, input_chunks);
      tk_automata_dec_not_excluded(aut, clause_id, (uint8_t*)input, input_chunks);
    } else {
      uint64_t input_bits = 2 * features;
      unsigned int s = (unsigned int)((double)(2 * features) / specificity);
      for (unsigned int r = 0; r < s; r++) {
        unsigned int random_input_bit = tk_fast_random() % input_bits;
        unsigned int random_chunk = random_input_bit / 8;
        uint8_t random_mask = 1 << (random_input_bit % 8);
        tk_automata_dec_byte(aut, clause_id, random_chunk, random_mask);
      }
    }
  } else {
    if (output) {
      tk_automata_inc_not_excluded(aut, clause_id, (uint8_t*)input, input_chunks);
    }
  }
}

static inline void tm_update_ind (
  tk_automata_t *aut,
  char *input,
  uint8_t out,
  unsigned int *literals,
  unsigned int *votes,
  bool target_vote,
  unsigned int chunk,
  long int vote_target,
  double negative_sampling,
  uint64_t features,
  uint64_t input_chunks,
  double specificity,
  unsigned int max_literals
) {
  long int chunk_vote = tk_tsetlin_sums_ind(out, votes);
  chunk_vote = (chunk_vote > vote_target) ? vote_target : chunk_vote;
  chunk_vote = (chunk_vote < -vote_target) ? -vote_target : chunk_vote;
  double p;
  if (target_vote)
    p = (double)(vote_target - chunk_vote) / (2.0 * vote_target);
  else
    p = (double)(vote_target + chunk_vote) / (2.0 * vote_target);
  for (unsigned int j = 0; j < TK_CVEC_BITS; j += 2) {
    unsigned int pos_clause = j;
    unsigned int neg_clause = j + 1;
    if (neg_clause >= TK_CVEC_BITS) break;
    if (target_vote) {
      if (tk_fast_chance(p))
        apply_feedback_ind(aut, pos_clause, chunk, input, literals, votes, true, features, input_chunks, specificity, max_literals);
      if (tk_fast_chance(p))
        apply_feedback_ind(aut, neg_clause, chunk, input, literals, votes, false, features, input_chunks, specificity, max_literals);
    } else {
      if (tk_fast_chance(p))
        apply_feedback_ind(aut, neg_clause, chunk, input, literals, votes, true, features, input_chunks, specificity, max_literals);
      if (tk_fast_chance(p))
        apply_feedback_ind(aut, pos_clause, chunk, input, literals, votes, false, features, input_chunks, specificity, max_literals);
    }
  }
}

tk_tsetlin_t *tk_tsetlin_peek (lua_State *L, int i)
{
  return (tk_tsetlin_t *) luaL_checkudata(L, i, TK_TSETLIN_MT);
}

static inline int tk_tsetlin_train (lua_State *);
static inline int tk_tsetlin_predict (lua_State *);
static inline int tk_tsetlin_destroy (lua_State *L);
static inline int tk_tsetlin_persist (lua_State *);
static inline int tk_tsetlin_checkpoint (lua_State *);
static inline int tk_tsetlin_restore (lua_State *);
static inline int tk_tsetlin_reconfigure (lua_State *);
static inline int tk_tsetlin_type (lua_State *);

static luaL_Reg tk_tsetlin_mt_fns[] =
{
  { "train", tk_tsetlin_train },
  { "predict", tk_tsetlin_predict },
  { "destroy", tk_tsetlin_destroy },
  { "persist", tk_tsetlin_persist },
  { "checkpoint", tk_tsetlin_checkpoint },
  { "restore", tk_tsetlin_restore },
  { "reconfigure", tk_tsetlin_reconfigure },
  { "type", tk_tsetlin_type },
  { NULL, NULL }
};

static inline tk_tsetlin_t *tk_tsetlin_alloc_classifier (lua_State *L, bool has_state)
{
  tk_tsetlin_t *tm = tk_lua_newuserdata(L, tk_tsetlin_t, TK_TSETLIN_MT, tk_tsetlin_mt_fns, tk_tsetlin_destroy);
  tm->type = TM_CLASSIFIER;
  tm->has_state = has_state;
  return tm;
}

static inline tk_tsetlin_t *tk_tsetlin_alloc_encoder (lua_State *L, bool has_state)
{
  tk_tsetlin_t *tm = tk_lua_newuserdata(L, tk_tsetlin_t, TK_TSETLIN_MT, tk_tsetlin_mt_fns, tk_tsetlin_destroy);
  tm->type = TM_ENCODER;
  tm->has_state = has_state;
  return tm;
}

static inline void tk_tsetlin_init_classifier (
  lua_State *L,
  tk_tsetlin_t *tm,
  unsigned int classes,
  unsigned int features,
  unsigned int clauses,
  unsigned int clause_tolerance,
  unsigned int clause_maximum,
  unsigned int state_bits,
  unsigned int include_bits,
  double targetf,
  double negative,
  double specificity
) {
  if (!classes)
    tk_lua_verror(L, 3, "create classifier", "classes", "must be greater than 1");
  if (!clauses)
    tk_lua_verror(L, 3, "create classifier", "clauses", "must be greater than 0");
  if (!clause_tolerance)
    tk_lua_verror(L, 3, "create classifier", "clause_tolerance", "must be greater than 0");
  if (!clause_maximum)
    tk_lua_verror(L, 3, "create classifier", "clause_maximum", "must be greater than 0");
  if (state_bits < 2)
    tk_lua_verror(L, 3, "create classifier", "bits", "must be greater than 1");
  tm->reusable = false;
  tm->negative = negative < 0 ? 1.0 / (double) classes : negative;
  tm->classes = classes;
  tm->class_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  tm->clauses = TK_CVEC_BITS_BYTES(clauses) * TK_CVEC_BITS;
  tm->clause_tolerance = clause_tolerance;
  tm->clause_maximum = clause_maximum;
  tm->target =
    targetf < 0
      ? sqrt((double) tm->clauses / 2.0) * (double) clause_tolerance
      : ceil(targetf >= 1 ? targetf : fmaxf(1.0, (double) tm->clauses * targetf));
  tm->features = features;
  tm->state_bits = state_bits;
  tm->include_bits = include_bits ? include_bits : 1;
  tm->input_bits = 2 * tm->features;
  uint64_t tail_bits = tm->input_bits & (TK_CVEC_BITS - 1);
  tm->tail_mask = tail_bits ? (uint8_t)((1u << tail_bits) - 1) : 0xFF;
  tm->input_chunks = TK_CVEC_BITS_BYTES(tm->input_bits);
  tm->clause_chunks = TK_CVEC_BITS_BYTES(tm->clauses);
  tm->state_chunks = tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tm->action_chunks = tm->classes * tm->clauses * tm->input_chunks;
  tm->state = (char *)tk_malloc_aligned(L, tm->state_chunks, TK_CVEC_BITS);
  tm->actions = (char *)tk_malloc_aligned(L, tm->action_chunks, TK_CVEC_BITS);
  tm->specificity = specificity;
  if (!(tm->state && tm->actions))
    luaL_error(L, "error in malloc during creation of classifier");
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->automata.n_chunks = tm->input_chunks;
  tm->automata.state_bits = tm->state_bits;
  tm->automata.include_bits = tm->include_bits;
  tm->automata.tail_mask = tm->tail_mask;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;
}

static inline int tk_tsetlin_init_encoder (
  lua_State *L,
  tk_tsetlin_t *tm,
  unsigned int encoding_bits,
  unsigned int features,
  unsigned int clauses,
  unsigned int clause_tolerance,
  unsigned int clause_maximum,
  unsigned int state_bits,
  unsigned int include_bits,
  double targetf,
  double negative,
  double specificity
) {
  tk_tsetlin_init_classifier(
    L, tm, encoding_bits, features, clauses, clause_tolerance, clause_maximum,
    state_bits, include_bits, targetf, negative, specificity);
  return 0;
}

static inline void tk_tsetlin_init_classifier_ind (
  lua_State *L,
  tk_tsetlin_t *tm,
  unsigned int classes,
  tk_ivec_t *feat_offsets,
  unsigned int clauses,
  unsigned int clause_tolerance,
  unsigned int clause_maximum,
  unsigned int state_bits,
  unsigned int include_bits,
  double targetf,
  double negative,
  double specificity
) {
  if (!classes)
    tk_lua_verror(L, 3, "create classifier ind", "classes", "must be greater than 1");
  if (!clauses)
    tk_lua_verror(L, 3, "create classifier ind", "clauses", "must be greater than 0");
  if (!clause_tolerance)
    tk_lua_verror(L, 3, "create classifier ind", "clause_tolerance", "must be greater than 0");
  if (!clause_maximum)
    tk_lua_verror(L, 3, "create classifier ind", "clause_maximum", "must be greater than 0");
  if (state_bits < 2)
    tk_lua_verror(L, 3, "create classifier ind", "bits", "must be greater than 1");
  if ((uint64_t)feat_offsets->n != classes + 1)
    tk_lua_verror(L, 3, "create classifier ind", "feat_offsets", "must have classes + 1 elements");
  tm->individualized = true;
  tm->reusable = false;
  tm->negative = negative < 0 ? 1.0 / (double) classes : negative;
  tm->classes = classes;
  tm->class_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  tm->clauses = TK_CVEC_BITS_BYTES(clauses) * TK_CVEC_BITS;
  tm->clause_chunks = TK_CVEC_BITS_BYTES(tm->clauses);
  tm->clause_tolerance = clause_tolerance;
  tm->clause_maximum = clause_maximum;
  tm->target =
    targetf < 0
      ? sqrt((double) tm->clauses / 2.0) * (double) clause_tolerance
      : ceil(targetf >= 1 ? targetf : fmaxf(1.0, (double) tm->clauses * targetf));
  tm->state_bits = state_bits;
  tm->include_bits = include_bits ? include_bits : 1;
  tm->specificity = specificity;
  tm->feat_sizes = (uint64_t *)tk_malloc(L, classes * sizeof(uint64_t));
  tm->input_chunks_ind = (uint64_t *)tk_malloc(L, classes * sizeof(uint64_t));
  tm->tail_masks_ind = (uint8_t *)tk_malloc(L, classes * sizeof(uint8_t));
  tm->state_offsets = (uint64_t *)tk_malloc(L, classes * sizeof(uint64_t));
  tm->actions_offsets = (uint64_t *)tk_malloc(L, classes * sizeof(uint64_t));
  tm->automata_arr = (tk_automata_t *)tk_malloc(L, classes * sizeof(tk_automata_t));
  uint64_t total_state_bytes = 0;
  uint64_t total_action_bytes = 0;
  for (unsigned int h = 0; h < classes; h++) {
    uint64_t k_h = (uint64_t)(feat_offsets->a[h + 1] - feat_offsets->a[h]);
    tm->feat_sizes[h] = k_h;
    uint64_t input_bits_h = 2 * k_h;
    tm->input_chunks_ind[h] = TK_CVEC_BITS_BYTES(input_bits_h);
    uint64_t tail_bits_h = input_bits_h & (TK_CVEC_BITS - 1);
    tm->tail_masks_ind[h] = tail_bits_h ? (uint8_t)((1u << tail_bits_h) - 1) : 0xFF;
    tm->state_offsets[h] = total_state_bytes;
    tm->actions_offsets[h] = total_action_bytes;
    total_state_bytes += tm->clauses * tm->input_chunks_ind[h] * (state_bits - 1);
    total_action_bytes += tm->clauses * tm->input_chunks_ind[h];
  }
  tm->state_chunks = (unsigned int)total_state_bytes;
  tm->action_chunks = (unsigned int)total_action_bytes;
  tm->state = (char *)tk_malloc_aligned(L, total_state_bytes, TK_CVEC_BITS);
  tm->actions = (char *)tk_malloc_aligned(L, total_action_bytes, TK_CVEC_BITS);
  if (!(tm->state && tm->actions))
    luaL_error(L, "error in malloc during creation of individualized classifier");
  for (unsigned int h = 0; h < classes; h++) {
    tk_automata_t *a = &tm->automata_arr[h];
    a->n_clauses = tm->clauses;
    a->n_chunks = tm->input_chunks_ind[h];
    a->state_bits = tm->state_bits;
    a->include_bits = tm->include_bits;
    a->tail_mask = tm->tail_masks_ind[h];
    a->counts = tm->state + tm->state_offsets[h];
    a->actions = tm->actions + tm->actions_offsets[h];
  }
}

static inline void tk_tsetlin_init_encoder_ind (
  lua_State *L,
  tk_tsetlin_t *tm,
  unsigned int encoding_bits,
  tk_ivec_t *feat_offsets,
  unsigned int clauses,
  unsigned int clause_tolerance,
  unsigned int clause_maximum,
  unsigned int state_bits,
  unsigned int include_bits,
  double targetf,
  double negative,
  double specificity
) {
  tk_tsetlin_init_classifier_ind(
    L, tm, encoding_bits, feat_offsets, clauses, clause_tolerance, clause_maximum,
    state_bits, include_bits, targetf, negative, specificity);
}

static inline void tk_tsetlin_create_classifier (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, true);
  lua_insert(L, 1);
  bool individualized = tk_lua_foptboolean(L, 2, "create classifier", "individualized", false);
  if (individualized) {
    lua_getfield(L, 2, "feat_offsets");
    tk_ivec_t *feat_offsets = tk_ivec_peek(L, -1, "feat_offsets");
    lua_pop(L, 1);
    tk_tsetlin_init_classifier_ind(L, tm,
        tk_lua_fcheckunsigned(L, 2, "create classifier", "classes"),
        feat_offsets,
        tk_lua_fcheckunsigned(L, 2, "create classifier", "clauses"),
        tk_lua_fcheckunsigned(L, 2, "create classifier", "clause_tolerance"),
        tk_lua_fcheckunsigned(L, 2, "create classifier", "clause_maximum"),
        tk_lua_foptunsigned(L, 2, "create classifier", "state", 8),
        tk_lua_foptunsigned(L, 2, "create classifier", "include_bits", 1),
        tk_lua_foptposdouble(L, 2, "create classifier", "target", -1.0),
        tk_lua_foptposdouble(L, 2, "create classifier", "negative", -1.0),
        tk_lua_fcheckposdouble(L, 2, "create classifier", "specificity"));
  } else {
    tk_tsetlin_init_classifier(L, tm,
        tk_lua_fcheckunsigned(L, 2, "create classifier", "classes"),
        tk_lua_fcheckunsigned(L, 2, "create classifier", "features"),
        tk_lua_fcheckunsigned(L, 2, "create classifier", "clauses"),
        tk_lua_fcheckunsigned(L, 2, "create classifier", "clause_tolerance"),
        tk_lua_fcheckunsigned(L, 2, "create classifier", "clause_maximum"),
        tk_lua_foptunsigned(L, 2, "create classifier", "state", 8),
        tk_lua_foptunsigned(L, 2, "create classifier", "include_bits", 1),
        tk_lua_foptposdouble(L, 2, "create classifier", "target", -1.0),
        tk_lua_foptposdouble(L, 2, "create classifier", "negative", -1.0),
        tk_lua_fcheckposdouble(L, 2, "create classifier", "specificity"));
  }
  tm->reusable = tk_lua_foptboolean(L, 2, "create classifier", "reusable", false);
  lua_settop(L, 1);
}

static inline void tk_tsetlin_create_encoder (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, true);
  lua_insert(L, 1);
  bool individualized = tk_lua_foptboolean(L, 2, "create encoder", "individualized", false);
  if (individualized) {
    lua_getfield(L, 2, "feat_offsets");
    tk_ivec_t *feat_offsets = tk_ivec_peek(L, -1, "feat_offsets");
    lua_pop(L, 1);
    tk_tsetlin_init_encoder_ind(L, tm,
        tk_lua_fcheckunsigned(L, 2, "create encoder", "hidden"),
        feat_offsets,
        tk_lua_fcheckunsigned(L, 2, "create encoder", "clauses"),
        tk_lua_fcheckunsigned(L, 2, "create encoder", "clause_tolerance"),
        tk_lua_fcheckunsigned(L, 2, "create encoder", "clause_maximum"),
        tk_lua_foptunsigned(L, 2, "create encoder", "state", 8),
        tk_lua_foptunsigned(L, 2, "create encoder", "include_bits", 1),
        tk_lua_foptposdouble(L, 2, "create encoder", "target", -1.0),
        tk_lua_foptposdouble(L, 2, "create encoder", "negative", -1.0),
        tk_lua_fcheckposdouble(L, 2, "create encoder", "specificity"));
  } else {
    tk_tsetlin_init_encoder(L, tm,
        tk_lua_fcheckunsigned(L, 2, "create encoder", "hidden"),
        tk_lua_fcheckunsigned(L, 2, "create encoder", "visible"),
        tk_lua_fcheckunsigned(L, 2, "create encoder", "clauses"),
        tk_lua_fcheckunsigned(L, 2, "create encoder", "clause_tolerance"),
        tk_lua_fcheckunsigned(L, 2, "create encoder", "clause_maximum"),
        tk_lua_foptunsigned(L, 2, "create encoder", "state", 8),
        tk_lua_foptunsigned(L, 2, "create encoder", "include_bits", 1),
        tk_lua_foptposdouble(L, 2, "create encoder", "target", -1.0),
        tk_lua_foptposdouble(L, 2, "create encoder", "negative", -1.0),
        tk_lua_fcheckposdouble(L, 2, "create encoder", "specificity"));
  }
  tm->reusable = tk_lua_foptboolean(L, 2, "create encoder", "reusable", false);
  lua_settop(L, 1);
}

static inline int tk_tsetlin_create (lua_State *L)
{
  const char *type = luaL_checkstring(L, 1);
  if (!strcmp(type, "classifier")) {
    lua_remove(L, 1);
    tk_tsetlin_create_classifier(L);
    return 1;
  } else if (!strcmp(type, "encoder")) {
    lua_remove(L, 1);
    tk_tsetlin_create_encoder(L);
    return 1;
  } else {
    luaL_error(L, "unexpected tsetlin machine type in create");
    return 0;
  }
}

static inline void tk_tsetlin_shrink (tk_tsetlin_t *tm)
{
  if (tm == NULL) return;
  tm->reusable = false;
  free(tm->state); tm->state = NULL;
}

static inline void _tk_tsetlin_destroy (tk_tsetlin_t *tm)
{
  if (tm == NULL) return;
  if (tm->destroyed) return;
  tm->destroyed = true;
  tk_tsetlin_shrink(tm);
  free(tm->actions); tm->actions = NULL;
  free(tm->results); tm->results = NULL;
  free(tm->encodings); tm->encodings = NULL;
  if (tm->individualized) {
    free(tm->feat_sizes); tm->feat_sizes = NULL;
    free(tm->input_chunks_ind); tm->input_chunks_ind = NULL;
    free(tm->tail_masks_ind); tm->tail_masks_ind = NULL;
    free(tm->state_offsets); tm->state_offsets = NULL;
    free(tm->actions_offsets); tm->actions_offsets = NULL;
    free(tm->automata_arr); tm->automata_arr = NULL;
  }
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  _tk_tsetlin_destroy(tm);
  return 0;
}

static inline int tk_tsetlin_predict_classifier (
  lua_State *L,
  tk_tsetlin_t *tm
) {
  lua_settop(L, 3);
  tk_cvec_t *ps = tk_cvec_peek(L, 2, "problems");
  unsigned int n = tk_lua_checkunsigned(L, 3, "argument 2 is not an integer n_samples");
  size_t needed = n * sizeof(unsigned int);
  if (needed > tm->results_len) {
    tm->results = tk_realloc(L, tm->results, needed);
    tm->results_len = needed;
  }
  const unsigned int total_chunks = tm->clause_chunks * tm->classes;
  const unsigned int input_chunks = tm->input_chunks;
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int classes = tm->classes;
  unsigned int *results = tm->results;
  #pragma omp parallel for schedule(static)
  for (unsigned int s = 0; s < n; s++) {
    long int sums[classes];
    for (unsigned int i = 0; i < classes; i++)
      sums[i] = 0;
    unsigned int literals[TK_CVEC_BITS];
    unsigned int votes[TK_CVEC_BITS];
    char *input = ps->a + s * input_chunks;
    for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
      uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk);
      long int score = tk_tsetlin_sums(tm, out, votes);
      unsigned int class = chunk / clause_chunks;
      sums[class] += score;
    }
    long int maxval = -INT64_MAX;
    unsigned int maxclass = 0;
    for (unsigned int class = 0; class < classes; class++) {
      if (sums[class] > maxval) {
        maxval = sums[class];
        maxclass = class;
      }
    }
    results[s] = maxclass;
  }
  tk_ivec_t *out = tk_ivec_create(L, n, 0, 0);
  for (uint64_t i = 0; i < n; i ++)
    out->a[i] = tm->results[i];
  return 1;
}

static inline int tk_tsetlin_predict_encoder (
  lua_State *L,
  tk_tsetlin_t *tm
) {
  lua_settop(L, 3);
  tk_cvec_t *ps = tk_cvec_peek(L, 2, "problems");
  unsigned int n = tk_lua_checkunsigned(L, 3, "argument 2 is not an integer n_samples");
  size_t needed = n * tm->class_chunks;
  if (needed > tm->encodings_len) {
    tm->encodings = (char *)tk_realloc(L, tm->encodings, needed);
    tm->encodings_len = needed;
  }
  const unsigned int total_chunks = tm->clause_chunks * tm->classes;
  const unsigned int input_chunks = tm->input_chunks;
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int classes = tm->classes;
  const unsigned int class_chunks = tm->class_chunks;
  char *encodings = tm->encodings;
  #pragma omp parallel for schedule(static)
  for (unsigned int s = 0; s < n; s++) {
    long int votes[classes];
    for (unsigned int i = 0; i < classes; i++)
      votes[i] = 0;
    unsigned int literals[TK_CVEC_BITS];
    unsigned int votes_buf[TK_CVEC_BITS];
    char *input = ps->a + s * input_chunks;
    for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
      uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes_buf, chunk);
      long int score = tk_tsetlin_sums(tm, out, votes_buf);
      unsigned int class = chunk / clause_chunks;
      votes[class] += score;
    }
    uint8_t *e = (uint8_t *)(encodings + s * class_chunks);
    for (unsigned int class = 0; class < classes; class++) {
      unsigned int chunk = TK_CVEC_BITS_BYTE(class);
      unsigned int pos = TK_CVEC_BITS_BIT(class);
      if (votes[class] > 0)
        e[chunk] |= (1 << pos);
      else
        e[chunk] &= ~(1 << pos);
    }
  }
  tk_cvec_t *out = tk_cvec_create(L, n * tm->class_chunks, 0, 0);
  memcpy(out->a, tm->encodings, n * tm->class_chunks);
  return 1;
}

static inline int tk_tsetlin_predict_classifier_ind (lua_State *L, tk_tsetlin_t *tm) {
  lua_settop(L, 4);
  tk_cvec_t *ps = tk_cvec_peek(L, 2, "problems");
  tk_ivec_t *dim_offsets = tk_ivec_peek(L, 3, "dim_offsets");
  unsigned int n = tk_lua_checkunsigned(L, 4, "n_samples");
  size_t needed = n * sizeof(unsigned int);
  if (needed > tm->results_len) {
    tm->results = tk_realloc(L, tm->results, needed);
    tm->results_len = needed;
  }
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int classes = tm->classes;
  const unsigned int tolerance = tm->clause_tolerance;
  unsigned int *results = tm->results;
  #pragma omp parallel for schedule(static)
  for (unsigned int s = 0; s < n; s++) {
    long int sums[classes];
    for (unsigned int i = 0; i < classes; i++)
      sums[i] = 0;
    unsigned int literals[TK_CVEC_BITS];
    unsigned int votes[TK_CVEC_BITS];
    for (unsigned int h = 0; h < classes; h++) {
      tk_automata_t *aut = &tm->automata_arr[h];
      uint64_t input_chunks_h = tm->input_chunks_ind[h];
      uint8_t tail_mask_h = tm->tail_masks_ind[h];
      uint64_t byte_offset_h = (uint64_t)dim_offsets->a[h];
      char *input = ps->a + byte_offset_h + s * input_chunks_h;
      for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
        uint8_t out = tk_tsetlin_calculate_ind(aut, input, literals, votes, chunk, input_chunks_h, tail_mask_h, tolerance);
        sums[h] += tk_tsetlin_sums_ind(out, votes);
      }
    }
    long int maxval = -INT64_MAX;
    unsigned int maxclass = 0;
    for (unsigned int h = 0; h < classes; h++) {
      if (sums[h] > maxval) {
        maxval = sums[h];
        maxclass = h;
      }
    }
    results[s] = maxclass;
  }
  tk_ivec_t *out = tk_ivec_create(L, n, 0, 0);
  for (uint64_t i = 0; i < n; i++)
    out->a[i] = tm->results[i];
  return 1;
}

static inline int tk_tsetlin_predict_encoder_ind (lua_State *L, tk_tsetlin_t *tm) {
  lua_settop(L, 4);
  tk_cvec_t *ps = tk_cvec_peek(L, 2, "sentences");
  tk_ivec_t *dim_offsets = tk_ivec_peek(L, 3, "dim_offsets");
  unsigned int n = tk_lua_checkunsigned(L, 4, "n_samples");
  size_t needed = n * tm->class_chunks;
  if (needed > tm->encodings_len) {
    tm->encodings = (char *)tk_realloc(L, tm->encodings, needed);
    tm->encodings_len = needed;
  }
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int classes = tm->classes;
  const unsigned int class_chunks = tm->class_chunks;
  const unsigned int tolerance = tm->clause_tolerance;
  char *encodings = tm->encodings;
  memset(encodings, 0, n * class_chunks);
  #pragma omp parallel for schedule(static)
  for (unsigned int s = 0; s < n; s++) {
    unsigned int literals[TK_CVEC_BITS];
    unsigned int votes_buf[TK_CVEC_BITS];
    uint8_t *e = (uint8_t *)(encodings + s * class_chunks);
    for (unsigned int h = 0; h < classes; h++) {
      tk_automata_t *aut = &tm->automata_arr[h];
      uint64_t input_chunks_h = tm->input_chunks_ind[h];
      uint8_t tail_mask_h = tm->tail_masks_ind[h];
      uint64_t byte_offset_h = (uint64_t)dim_offsets->a[h];
      char *input = ps->a + byte_offset_h + s * input_chunks_h;
      long int vote_sum = 0;
      for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
        uint8_t out = tk_tsetlin_calculate_ind(aut, input, literals, votes_buf, chunk, input_chunks_h, tail_mask_h, tolerance);
        vote_sum += tk_tsetlin_sums_ind(out, votes_buf);
      }
      unsigned int enc_chunk = TK_CVEC_BITS_BYTE(h);
      unsigned int enc_bit = TK_CVEC_BITS_BIT(h);
      if (vote_sum > 0)
        e[enc_chunk] |= (1 << enc_bit);
    }
  }
  tk_cvec_t *out = tk_cvec_create(L, n * class_chunks, 0, 0);
  memcpy(out->a, encodings, n * class_chunks);
  return 1;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  if (tm->individualized) {
    switch (tm->type) {
      case TM_CLASSIFIER:
        return tk_tsetlin_predict_classifier_ind(L, tm);
      case TM_ENCODER:
        return tk_tsetlin_predict_encoder_ind(L, tm);
      default:
        return luaL_error(L, "unexpected tsetlin machine type in predict");
    }
  }
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_predict_classifier(L, tm);
    case TM_ENCODER:
      return tk_tsetlin_predict_encoder(L, tm);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in predict");
  }
  return 0;
}

static inline int tk_tsetlin_train_classifier (
  lua_State *L,
  tk_tsetlin_t *tm
) {
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  lua_getfield(L, 2, "problems");
  tk_cvec_t *ps = tk_cvec_peek(L, -1, "problems");
  lua_getfield(L, 2, "solutions");
  tk_ivec_t *ss = tk_ivec_peek(L, -1, "solutions");
  unsigned int max_iter = tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }
  unsigned int total_chunks = tm->clause_chunks * tm->classes;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  int64_t *lbls = ss->a;
  bool break_flag = false;
  int max_threads = omp_get_max_threads();
  unsigned int **shuffles = (unsigned int **)tk_malloc(L, (size_t)max_threads * sizeof(unsigned int *));
  for (int t = 0; t < max_threads; t++)
    shuffles[t] = (unsigned int *)tk_malloc(L, n * sizeof(unsigned int));
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    tk_fast_seed((uint64_t)tid);
    #pragma omp for schedule(static)
    for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
      uint64_t first_clause = chunk * TK_CVEC_BITS;
      uint64_t last_clause = chunk * TK_CVEC_BITS + TK_CVEC_BITS - 1;
      tk_automata_setup(&tm->automata, first_clause, last_clause);
    }
  }
  for (unsigned int iter = 0; iter < max_iter; iter++) {
    if (break_flag) break;
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int *shuffle = shuffles[tid];
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes[TK_CVEC_BITS];
      tk_tsetlin_init_shuffle(shuffle, n);
      #pragma omp for schedule(static)
      for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
        unsigned int chunk_class = chunk / clause_chunks;
        for (unsigned int i = 0; i < n; i++) {
          unsigned int sample = shuffle[i];
          char *input = ps->a + sample * input_chunks;
          uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk);
          unsigned int sample_class = lbls[sample];
          tm_update(tm, input, out, literals, votes, sample, sample_class, 1, chunk_class, chunk, (unsigned int)tid);
        }
      }
    }
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, iter + 1);
      int status = lua_pcall(L, 1, 1, 0);
      if (status != LUA_OK) {
        fprintf(stderr, "Error in Lua callback: %s\n", lua_tostring(L, -1));
        lua_pop(L, 1);
        break_flag = true;
      } else if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break_flag = true;
      } else {
        lua_pop(L, 1);
      }
    }
  }
  for (int t = 0; t < max_threads; t++)
    free(shuffles[t]);
  free(shuffles);
  if (!tm->reusable)
    tk_tsetlin_shrink(tm);
  tm->trained = true;
  return 0;
}

static inline int tk_tsetlin_train_encoder (
  lua_State *L,
  tk_tsetlin_t *tm
) {
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  lua_getfield(L, 2, "sentences");
  tk_cvec_t *ps = tk_cvec_peek(L, -1, "sentences");
  lua_pop(L, 1);
  lua_getfield(L, 2, "codes");
  tk_cvec_t *ls = tk_cvec_peek(L, -1, "codes");
  lua_pop(L, 1);
  unsigned int max_iter = tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  size_t needed = n * tm->class_chunks;
  if (needed > tm->encodings_len) {
    tm->encodings = (char *)tk_realloc(L, tm->encodings, needed);
    tm->encodings_len = needed;
  }
  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }
  unsigned int total_chunks = tm->clause_chunks * tm->classes;
  unsigned int class_chunks = tm->class_chunks;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  char *lbls = ls->a;
  bool break_flag = false;
  int max_threads = omp_get_max_threads();
  unsigned int **shuffles = (unsigned int **)tk_malloc(L, (size_t)max_threads * sizeof(unsigned int *));
  for (int t = 0; t < max_threads; t++)
    shuffles[t] = (unsigned int *)tk_malloc(L, n * sizeof(unsigned int));
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    tk_fast_seed((uint64_t)tid);
    #pragma omp for schedule(static)
    for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
      uint64_t first_clause = chunk * TK_CVEC_BITS;
      uint64_t last_clause = chunk * TK_CVEC_BITS + TK_CVEC_BITS - 1;
      tk_automata_setup(&tm->automata, first_clause, last_clause);
    }
  }
  for (unsigned int iter = 0; iter < max_iter; iter++) {
    if (break_flag) break;
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int *shuffle = shuffles[tid];
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes[TK_CVEC_BITS];
      tk_tsetlin_init_shuffle(shuffle, n);
      #pragma omp for schedule(static)
      for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
        unsigned int chunk_class = chunk / clause_chunks;
        unsigned int enc_chunk = TK_CVEC_BITS_BYTE(chunk_class);
        unsigned int enc_bit = TK_CVEC_BITS_BIT(chunk_class);
        for (unsigned int i = 0; i < n; i++) {
          unsigned int sample = shuffle[i];
          char *input = ps->a + sample * input_chunks;
          uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk);
          bool target_vote = (((uint8_t *)lbls)[sample * class_chunks + enc_chunk] & (1 << enc_bit)) > 0;
          tm_update(tm, input, out, literals, votes, sample, chunk_class, target_vote, chunk_class, chunk, (unsigned int)tid);
        }
      }
    }
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, iter + 1);
      int status = lua_pcall(L, 1, 1, 0);
      if (status != LUA_OK) {
        fprintf(stderr, "Error in Lua callback: %s\n", lua_tostring(L, -1));
        lua_pop(L, 1);
        break_flag = true;
      } else if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break_flag = true;
      } else {
        lua_pop(L, 1);
      }
    }
  }
  for (int t = 0; t < max_threads; t++)
    free(shuffles[t]);
  free(shuffles);
  if (!tm->reusable)
    tk_tsetlin_shrink(tm);
  tm->trained = true;
  return 0;
}

static inline int tk_tsetlin_train_encoder_ind (lua_State *L, tk_tsetlin_t *tm) {
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  lua_getfield(L, 2, "sentences");
  tk_cvec_t *ps = tk_cvec_peek(L, -1, "sentences");
  lua_pop(L, 1);
  lua_getfield(L, 2, "codes");
  tk_cvec_t *ls = tk_cvec_peek(L, -1, "codes");
  lua_pop(L, 1);
  lua_getfield(L, 2, "dim_offsets");
  tk_ivec_t *dim_offsets = tk_ivec_peek(L, -1, "dim_offsets");
  lua_pop(L, 1);
  unsigned int max_iter = tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  size_t needed = n * tm->class_chunks;
  if (needed > tm->encodings_len) {
    tm->encodings = (char *)tk_realloc(L, tm->encodings, needed);
    tm->encodings_len = needed;
  }
  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }
  unsigned int classes = tm->classes;
  unsigned int class_chunks = tm->class_chunks;
  unsigned int clause_chunks = tm->clause_chunks;
  char *lbls = ls->a;
  long int vote_target = (long int)tm->target;
  double specificity = tm->specificity;
  unsigned int max_literals = tm->clause_maximum;
  unsigned int tolerance = tm->clause_tolerance;
  bool break_flag = false;
  int max_threads = omp_get_max_threads();
  unsigned int **shuffles = (unsigned int **)tk_malloc(L, (size_t)max_threads * sizeof(unsigned int *));
  for (int t = 0; t < max_threads; t++)
    shuffles[t] = (unsigned int *)tk_malloc(L, n * sizeof(unsigned int));
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    tk_fast_seed((uint64_t)tid);
    #pragma omp for schedule(static)
    for (unsigned int h = 0; h < classes; h++) {
      tk_automata_t *aut = &tm->automata_arr[h];
      for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
        uint64_t first_clause = chunk * TK_CVEC_BITS;
        uint64_t last_clause = chunk * TK_CVEC_BITS + TK_CVEC_BITS - 1;
        tk_automata_setup(aut, first_clause, last_clause);
      }
    }
  }
  for (unsigned int iter = 0; iter < max_iter; iter++) {
    if (break_flag) break;
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int *shuffle = shuffles[tid];
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes[TK_CVEC_BITS];
      tk_tsetlin_init_shuffle(shuffle, n);
      #pragma omp for schedule(static)
      for (unsigned int h = 0; h < classes; h++) {
        tk_automata_t *aut = &tm->automata_arr[h];
        uint64_t input_chunks_h = tm->input_chunks_ind[h];
        uint8_t tail_mask_h = tm->tail_masks_ind[h];
        uint64_t feat_size_h = tm->feat_sizes[h];
        uint64_t byte_offset_h = (uint64_t)dim_offsets->a[h];
        unsigned int enc_chunk = TK_CVEC_BITS_BYTE(h);
        unsigned int enc_bit = TK_CVEC_BITS_BIT(h);
        for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
          for (unsigned int i = 0; i < n; i++) {
            unsigned int sample = shuffle[i];
            char *input = ps->a + byte_offset_h + sample * input_chunks_h;
            uint8_t out = tk_tsetlin_calculate_ind(aut, input, literals, votes, chunk, input_chunks_h, tail_mask_h, tolerance);
            bool target_vote = (((uint8_t *)lbls)[sample * class_chunks + enc_chunk] & (1 << enc_bit)) > 0;
            tm_update_ind(aut, input, out, literals, votes, target_vote, chunk, vote_target, 0.0, feat_size_h, input_chunks_h, specificity, max_literals);
          }
        }
      }
    }
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, iter + 1);
      int status = lua_pcall(L, 1, 1, 0);
      if (status != LUA_OK) {
        fprintf(stderr, "Error in Lua callback: %s\n", lua_tostring(L, -1));
        lua_pop(L, 1);
        break_flag = true;
      } else if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break_flag = true;
      } else {
        lua_pop(L, 1);
      }
    }
  }
  for (int t = 0; t < max_threads; t++)
    free(shuffles[t]);
  free(shuffles);
  if (!tm->reusable)
    tk_tsetlin_shrink(tm);
  tm->trained = true;
  return 0;
}

static inline int tk_tsetlin_train_classifier_ind (lua_State *L, tk_tsetlin_t *tm) {
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  lua_getfield(L, 2, "problems");
  tk_cvec_t *ps = tk_cvec_peek(L, -1, "problems");
  lua_getfield(L, 2, "solutions");
  tk_ivec_t *ss = tk_ivec_peek(L, -1, "solutions");
  lua_getfield(L, 2, "dim_offsets");
  tk_ivec_t *dim_offsets = tk_ivec_peek(L, -1, "dim_offsets");
  lua_pop(L, 3);
  unsigned int max_iter = tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }
  unsigned int classes = tm->classes;
  unsigned int clause_chunks = tm->clause_chunks;
  int64_t *lbls = ss->a;
  long int vote_target = (long int)tm->target;
  double specificity = tm->specificity;
  double negative_sampling = tm->negative;
  unsigned int max_literals = tm->clause_maximum;
  unsigned int tolerance = tm->clause_tolerance;
  bool break_flag = false;
  int max_threads = omp_get_max_threads();
  unsigned int **shuffles = (unsigned int **)tk_malloc(L, (size_t)max_threads * sizeof(unsigned int *));
  for (int t = 0; t < max_threads; t++)
    shuffles[t] = (unsigned int *)tk_malloc(L, n * sizeof(unsigned int));
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    tk_fast_seed((uint64_t)tid);
    #pragma omp for schedule(static)
    for (unsigned int h = 0; h < classes; h++) {
      tk_automata_t *aut = &tm->automata_arr[h];
      for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
        uint64_t first_clause = chunk * TK_CVEC_BITS;
        uint64_t last_clause = chunk * TK_CVEC_BITS + TK_CVEC_BITS - 1;
        tk_automata_setup(aut, first_clause, last_clause);
      }
    }
  }
  for (unsigned int iter = 0; iter < max_iter; iter++) {
    if (break_flag) break;
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int *shuffle = shuffles[tid];
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes[TK_CVEC_BITS];
      tk_tsetlin_init_shuffle(shuffle, n);
      #pragma omp for schedule(static)
      for (unsigned int h = 0; h < classes; h++) {
        tk_automata_t *aut = &tm->automata_arr[h];
        uint64_t input_chunks_h = tm->input_chunks_ind[h];
        uint8_t tail_mask_h = tm->tail_masks_ind[h];
        uint64_t feat_size_h = tm->feat_sizes[h];
        uint64_t byte_offset_h = (uint64_t)dim_offsets->a[h];
        for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
          for (unsigned int i = 0; i < n; i++) {
            unsigned int sample = shuffle[i];
            unsigned int sample_class = (unsigned int)lbls[sample];
            if (h != sample_class && tk_fast_chance(1 - negative_sampling))
              continue;
            char *input = ps->a + byte_offset_h + sample * input_chunks_h;
            uint8_t out = tk_tsetlin_calculate_ind(aut, input, literals, votes, chunk, input_chunks_h, tail_mask_h, tolerance);
            bool target_vote = (h == sample_class);
            tm_update_ind(aut, input, out, literals, votes, target_vote, chunk, vote_target, negative_sampling, feat_size_h, input_chunks_h, specificity, max_literals);
          }
        }
      }
    }
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, iter + 1);
      int status = lua_pcall(L, 1, 1, 0);
      if (status != LUA_OK) {
        fprintf(stderr, "Error in Lua callback: %s\n", lua_tostring(L, -1));
        lua_pop(L, 1);
        break_flag = true;
      } else if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break_flag = true;
      } else {
        lua_pop(L, 1);
      }
    }
  }
  for (int t = 0; t < max_threads; t++)
    free(shuffles[t]);
  free(shuffles);
  if (!tm->reusable)
    tk_tsetlin_shrink(tm);
  tm->trained = true;
  return 0;
}

static inline int tk_tsetlin_train (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  if (!tm->has_state)
    luaL_error(L, "can't train a model loaded without state");
  if (tm->individualized) {
    switch (tm->type) {
      case TM_CLASSIFIER:
        return tk_tsetlin_train_classifier_ind(L, tm);
      case TM_ENCODER:
        return tk_tsetlin_train_encoder_ind(L, tm);
      default:
        return luaL_error(L, "unexpected tsetlin machine type in train");
    }
  }
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_train_classifier(L, tm);
    case TM_ENCODER:
      return tk_tsetlin_train_encoder(L, tm);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in train");
  }
}

static inline void _tk_tsetlin_persist_classifier (lua_State *L, tk_tsetlin_t *tm, FILE *fh)
{
  tk_lua_fwrite(L, &tm->classes, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->features, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clauses, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clause_tolerance, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clause_maximum, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->target, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->state_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->include_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->input_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->input_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clause_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->state_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->action_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &tm->tail_mask, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, tm->actions, 1, tm->action_chunks, fh);
  tk_lua_fwrite(L, &tm->individualized, sizeof(bool), 1, fh);
  if (tm->individualized) {
    tk_lua_fwrite(L, tm->feat_sizes, sizeof(uint64_t), tm->classes, fh);
  }
}

static inline void tk_tsetlin_persist_classifier (lua_State *L, tk_tsetlin_t *tm, FILE *fh)
{
  _tk_tsetlin_persist_classifier(L, tm, fh);
}

static inline int tk_tsetlin_persist (lua_State *L)
{
  lua_settop(L, 2);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  bool tostr = lua_type(L, 2) == LUA_TNIL;
  FILE *fh = tostr ? tk_lua_tmpfile(L) : tk_lua_fopen(L, tk_lua_checkstring(L, 2, "persist path"), "w");
  tk_lua_fwrite(L, &tm->type, sizeof(tk_tsetlin_type_t), 1, fh);
  tk_tsetlin_persist_classifier(L, tm, fh);
  if (!tostr) {
    tk_lua_fclose(L, fh);
    return 0;
  } else {
    size_t len;
    char *data = tk_lua_fslurp(L, fh, &len);
    if (data) {
      lua_pushlstring(L, data, len);
      free(data);
      tk_lua_fclose(L, fh);
      return 1;
    } else {
      tk_lua_fclose(L, fh);
      return 0;
    }
  }
}

static inline int tk_tsetlin_checkpoint (lua_State *L)
{
  lua_settop(L, 2);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  tk_cvec_t *checkpoint = tk_cvec_peek(L, 2, "checkpoint");

  size_t size = tm->action_chunks;
  if (tk_cvec_ensure(checkpoint, size) != 0)
    luaL_error(L, "failed to resize checkpoint buffer");
  checkpoint->n = size;
  memcpy(checkpoint->a, tm->actions, size);
  return 0;
}

static inline int tk_tsetlin_restore (lua_State *L)
{
  lua_settop(L, 2);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  tk_cvec_t *checkpoint = tk_cvec_peek(L, 2, "checkpoint");
  size_t expected_size = tm->action_chunks;
  if (checkpoint->n != expected_size)
    luaL_error(L, "checkpoint size mismatch: expected %zu bytes, got %zu", expected_size, checkpoint->n);
  memcpy(tm->actions, checkpoint->a, expected_size);
  return 0;
}

static inline int tk_tsetlin_reconfigure (lua_State *L)
{
  lua_settop(L, 2);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);

  if (!tm->reusable)
    luaL_error(L, "reconfigure requires reusable=true at creation time");

  unsigned int new_clauses = tk_lua_fcheckunsigned(L, 2, "reconfigure", "clauses");
  unsigned int new_tolerance = tk_lua_fcheckunsigned(L, 2, "reconfigure", "clause_tolerance");
  unsigned int new_maximum = tk_lua_fcheckunsigned(L, 2, "reconfigure", "clause_maximum");
  unsigned int new_include_bits = tk_lua_foptunsigned(L, 2, "reconfigure", "include_bits", tm->include_bits);
  double new_target = tk_lua_foptposdouble(L, 2, "reconfigure", "target", -1.0);
  double new_specificity = tk_lua_fcheckposdouble(L, 2, "reconfigure", "specificity");

  new_clauses = TK_CVEC_BITS_BYTES(new_clauses) * TK_CVEC_BITS;
  unsigned int new_clause_chunks = TK_CVEC_BITS_BYTES(new_clauses);

  unsigned int new_action_chunks, new_state_chunks;
  if (tm->individualized) {
    uint64_t total_action = 0, total_state = 0;
    for (unsigned int h = 0; h < tm->classes; h++) {
      total_action += new_clauses * tm->input_chunks_ind[h];
      total_state += new_clauses * tm->input_chunks_ind[h] * (tm->state_bits - 1);
    }
    new_action_chunks = (unsigned int)total_action;
    new_state_chunks = (unsigned int)total_state;
  } else {
    new_action_chunks = tm->classes * new_clauses * tm->input_chunks;
    new_state_chunks = tm->classes * new_clauses * (tm->state_bits - 1) * tm->input_chunks;
  }

  if (new_action_chunks > tm->action_chunks) {
    free(tm->actions);
    tm->actions = (char *)tk_malloc_aligned(L, new_action_chunks, TK_CVEC_BITS);
    if (!tm->actions)
      luaL_error(L, "failed to allocate actions in reconfigure");
  }

  memset(tm->actions, 0, new_action_chunks);

  if (new_state_chunks > tm->state_chunks || !tm->state) {
    if (tm->state) free(tm->state);
    tm->state = (char *)tk_malloc_aligned(L, new_state_chunks, TK_CVEC_BITS);
    if (!tm->state)
      luaL_error(L, "failed to allocate state in reconfigure");
  }

  tm->clauses = new_clauses;
  tm->clause_chunks = new_clause_chunks;
  tm->clause_tolerance = new_tolerance;
  tm->clause_maximum = new_maximum;
  tm->include_bits = new_include_bits ? new_include_bits : 1;
  tm->action_chunks = new_action_chunks;
  tm->state_chunks = new_state_chunks;
  tm->specificity = new_specificity;
  tm->target = new_target < 0
    ? sqrt((double) tm->clauses / 2.0) * (double) new_tolerance
    : ceil(new_target >= 1 ? new_target : fmaxf(1.0, (double) tm->clauses * new_target));

  if (tm->individualized) {
    uint64_t action_offset = 0, state_offset = 0;
    for (unsigned int h = 0; h < tm->classes; h++) {
      tm->actions_offsets[h] = action_offset;
      tm->state_offsets[h] = state_offset;
      action_offset += new_clauses * tm->input_chunks_ind[h];
      state_offset += new_clauses * tm->input_chunks_ind[h] * (tm->state_bits - 1);
      tk_automata_t *a = &tm->automata_arr[h];
      a->n_clauses = new_clauses;
      a->include_bits = tm->include_bits;
      a->counts = tm->state + tm->state_offsets[h];
      a->actions = tm->actions + tm->actions_offsets[h];
    }
  } else {
    tm->automata.n_clauses = tm->classes * tm->clauses;
    tm->automata.include_bits = tm->include_bits;
    tm->automata.counts = tm->state;
    tm->automata.actions = tm->actions;
  }

  tm->trained = false;

  return 0;
}

static inline void _tk_tsetlin_load_classifier (lua_State *L, tk_tsetlin_t *tm, FILE *fh)
{
  tk_lua_fread(L, &tm->classes, sizeof(unsigned int), 1, fh);
  tm->class_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  tk_lua_fread(L, &tm->features, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clauses, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clause_tolerance, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clause_maximum, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->target, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->state_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->include_bits, sizeof(unsigned int), 1, fh);
  if (!tm->include_bits) tm->include_bits = 1;
  tk_lua_fread(L, &tm->input_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->input_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clause_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->state_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->action_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fread(L, &tm->tail_mask, sizeof(uint8_t), 1, fh);
  tm->actions = (char *)tk_malloc_aligned(L, tm->action_chunks, TK_CVEC_BITS);
  tk_lua_fread(L, tm->actions, 1, tm->action_chunks, fh);
  tm->state = NULL;
  tk_lua_fread(L, &tm->individualized, sizeof(bool), 1, fh);
  if (tm->individualized) {
    unsigned int classes = tm->classes;
    tm->feat_sizes = (uint64_t *)tk_malloc(L, classes * sizeof(uint64_t));
    tk_lua_fread(L, tm->feat_sizes, sizeof(uint64_t), classes, fh);
    tm->input_chunks_ind = (uint64_t *)tk_malloc(L, classes * sizeof(uint64_t));
    tm->tail_masks_ind = (uint8_t *)tk_malloc(L, classes * sizeof(uint8_t));
    tm->state_offsets = (uint64_t *)tk_malloc(L, classes * sizeof(uint64_t));
    tm->actions_offsets = (uint64_t *)tk_malloc(L, classes * sizeof(uint64_t));
    tm->automata_arr = (tk_automata_t *)tk_malloc(L, classes * sizeof(tk_automata_t));
    uint64_t total_action_bytes = 0;
    for (unsigned int h = 0; h < classes; h++) {
      uint64_t k_h = tm->feat_sizes[h];
      uint64_t input_bits_h = 2 * k_h;
      tm->input_chunks_ind[h] = TK_CVEC_BITS_BYTES(input_bits_h);
      uint64_t tail_bits_h = input_bits_h & (TK_CVEC_BITS - 1);
      tm->tail_masks_ind[h] = tail_bits_h ? (uint8_t)((1u << tail_bits_h) - 1) : 0xFF;
      tm->actions_offsets[h] = total_action_bytes;
      tm->state_offsets[h] = 0;
      total_action_bytes += tm->clauses * tm->input_chunks_ind[h];
    }
    for (unsigned int h = 0; h < classes; h++) {
      tk_automata_t *a = &tm->automata_arr[h];
      a->n_clauses = tm->clauses;
      a->n_chunks = tm->input_chunks_ind[h];
      a->state_bits = tm->state_bits;
      a->include_bits = tm->include_bits;
      a->tail_mask = tm->tail_masks_ind[h];
      a->counts = NULL;
      a->actions = tm->actions + tm->actions_offsets[h];
    }
  } else {
    tm->automata.n_clauses = tm->classes * tm->clauses;
    tm->automata.n_chunks = tm->input_chunks;
    tm->automata.state_bits = tm->state_bits;
    tm->automata.include_bits = tm->include_bits;
    tm->automata.tail_mask = tm->tail_mask;
    tm->automata.counts = tm->state;
    tm->automata.actions = tm->actions;
  }
}

static inline void tk_tsetlin_load_classifier (lua_State *L, FILE *fh)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, true);
  _tk_tsetlin_load_classifier(L, tm, fh);
}

static inline void tk_tsetlin_load_encoder (lua_State *L, FILE *fh)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, true);
  _tk_tsetlin_load_classifier(L, tm, fh);
}

static inline int tk_tsetlin_load (lua_State *L)
{
  lua_settop(L, 2);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && lua_toboolean(L, 2);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  tk_tsetlin_type_t type;
  tk_lua_fread(L, &type, sizeof(type), 1, fh);
  switch (type) {
    case TM_CLASSIFIER:
      tk_tsetlin_load_classifier(L, fh);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_ENCODER:
      tk_tsetlin_load_encoder(L, fh);
      tk_lua_fclose(L, fh);
      return 1;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in load");
  }
}

static inline int tk_tsetlin_type (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      lua_pushstring(L, "classifier");
      break;
    case TM_ENCODER:
      lua_pushstring(L, "encoder");
      break;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in type");
  }
  return 1;
}

static luaL_Reg tk_tsetlin_fns[] =
{
  { "create", tk_tsetlin_create },
  { "load", tk_tsetlin_load },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_capi (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_tsetlin_fns, 0);
  lua_pushinteger(L, TK_CVEC_BITS);
  lua_setfield(L, -2, "align");
  return 1;
}
