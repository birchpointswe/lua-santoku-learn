/*

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

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

typedef enum {
  TM_CLASSIFIER,
  TM_ENCODER,
} tk_tsetlin_type_t;

typedef enum {
  TM_SEED,
  TM_CLASSIFIER_SETUP,
  TM_CLASSIFIER_TRAIN,
  TM_CLASSIFIER_PREDICT,
  TM_CLASSIFIER_PREDICT_REDUCE,
  TM_ENCODER_SETUP,
  TM_ENCODER_PREDICT,
  TM_ENCODER_PREDICT_REDUCE,
  TM_ENCODER_TRAIN,
} tk_tsetlin_stage_t;

typedef struct tk_tsetlin_s tk_tsetlin_t;
typedef struct tk_tsetlin_thread_s tk_tsetlin_thread_t;

typedef struct tk_tsetlin_thread_s {
  tk_tsetlin_t *tm;
  uint64_t sfirst;
  uint64_t slast;
  uint64_t cfirst;
  uint64_t clast;
  unsigned int index;
  struct {
    unsigned int n;
    tk_bits_t *ps;
    tk_bits_t *ls;
    unsigned int *ss;
  } train;
  struct {
    unsigned int n;
    tk_bits_t *ps;
  } predict;
  unsigned int *shuffle;
  long int *scores;
} tk_tsetlin_thread_t;

typedef struct tk_tsetlin_s {
  tk_tsetlin_type_t type;
  bool has_state;
  bool trained;
  bool destroyed;
  unsigned int classes;
  unsigned int features;
  unsigned int clauses;
  unsigned int clause_tolerance;
  unsigned int clause_maximum;
  unsigned int target;
  unsigned int state_bits;
  unsigned int input_bits;
  unsigned int input_chunks;
  unsigned int clause_chunks;
  unsigned int class_chunks;
  unsigned int state_chunks;
  unsigned int action_chunks;
  tk_bits_t tail_mask;
  tk_bits_t *state;
  tk_bits_t *actions;
  double negative;
  double specificity;
  size_t results_len;
  unsigned int *results;
  size_t encodings_len;
  tk_bits_t *encodings;
  tk_threadpool_t *pool;
  bool initialized_threads;
} tk_tsetlin_t;

#define tm_state_shuffle(tm, thread) \
  (((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->shuffle)

#define tm_state_scores(tm, thread, chunk, sample) \
  (((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->scores[ \
    ((chunk) - ((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->cfirst) * ((tk_tsetlin_thread_t *) (tm)->pool->threads[(thread)].data)->predict.n + \
    (sample)])

#define tm_state_clause_chunks(tm, thread) \
  (((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->clast - ((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->cfirst + 1)

#define tm_state_counts(tm, clause, input_chunk) \
  (&(tm)->state[(clause) * (tm)->input_chunks * ((tm)->state_bits - 1) + \
                (input_chunk) * ((tm)->state_bits - 1)])

#define tm_state_actions(tm, clause) \
  (&(tm)->actions[(clause) * (tm)->input_chunks])

static inline void tm_inc (
  tk_tsetlin_t *tm,
  unsigned int clause,
  unsigned int chunk,
  tk_bits_t active
) {
  if (!active) return;
  unsigned int m = tm->state_bits - 1;
  tk_bits_t *counts = tm_state_counts(tm, clause, chunk);
  tk_bits_t carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = counts[b] & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  tk_bits_t *actions = tm_state_actions(tm, clause);
  tk_bits_t carry_masked = carry & ((chunk == tm->input_chunks - 1) ? tm->tail_mask : (tk_bits_t)~0);
  carry_next = actions[chunk] & carry_masked;
  actions[chunk] ^= carry_masked;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] |= carry_masked;
  actions[chunk] |= carry_masked;
}

static inline void tm_dec (
  tk_tsetlin_t *tm,
  unsigned int clause,
  unsigned int chunk,
  tk_bits_t active
) {
  if (!active) return;
  unsigned int m = tm->state_bits - 1;
  tk_bits_t *counts = tm_state_counts(tm, clause, chunk);
  tk_bits_t carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = (~counts[b]) & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  tk_bits_t *actions = tm_state_actions(tm, clause);
  tk_bits_t carry_masked = carry & ((chunk == tm->input_chunks - 1) ? tm->tail_mask : (tk_bits_t) ~0);
  carry_next = (~actions[chunk]) & carry_masked;
  actions[chunk] ^= carry_masked;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] &= ~carry_masked;
  actions[chunk] &= ~carry_masked;
}

static inline tk_bits_t tk_tsetlin_calculate (
  tk_tsetlin_t *tm,
  tk_bits_t *input,
  unsigned int *literalsp,
  unsigned int *votesp,
  unsigned int chunk
) {
  tk_bits_t out = (tk_bits_t) 0;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int tolerance = tm->clause_tolerance;
  for (unsigned int j = 0; j < BITS; j ++) {
    unsigned int clause = chunk * BITS + j;
    tk_bits_t *actions = tm_state_actions(tm, clause);
    unsigned int failed = 0;
    unsigned int literals = 0;
    for (unsigned int k = 0; k < input_chunks - 1; k ++) {
      tk_bits_t inp = input[k];
      tk_bits_t act = actions[k];
      literals += popcount(act);
      failed += popcount(act & (~inp));
    }
    tk_bits_t last_act = actions[input_chunks - 1] & tm->tail_mask;
    literals += popcount(last_act);
    failed += popcount(last_act & (~input[input_chunks - 1]) & tm->tail_mask);
    long int votes;
    if (literals == 0) {
      votes = (long int) tolerance;
    } else {
      votes = (literals < tolerance ? (long int) literals : (long int) tolerance) - (long int) failed;
      if (votes < 0)
        votes = 0;
    }
    if (votes > 0)
      out |= ((tk_bits_t) 1 << j);
    literalsp[j] = literals;
    votesp[j] = (unsigned int) votes;
  }
  return out;
}

static inline long int tk_tsetlin_sums (
  tk_tsetlin_t *tm,
  tk_bits_t out,
  unsigned int *votes
) {
  long int sum = 0;
  for (unsigned int j = 0; j < BITS; j += 2)
    if (out & ((tk_bits_t) 1 << j))
      sum += (long int) votes[j];
  for (unsigned int j = 1; j < BITS; j += 2)
    if (out & ((tk_bits_t) 1 << j))
      sum -= (long int) votes[j];
  return sum;
}

static inline void apply_feedback (
  tk_tsetlin_t *tm,
  unsigned int clause_idx,
  unsigned int chunk,
  tk_bits_t *input,
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
  if (positive_feedback) {
    // Positive feedback
    if (output) {
      // Type Ia: clause voted
      if (literals[clause_idx] < max_literals)
        for (unsigned int k = 0; k < input_chunks; k ++)
          tm_inc(tm, chunk * BITS + clause_idx, k, input[k]);
      // Type Ib: deterministic negative feedback
      for (unsigned int k = 0; k < input_chunks; k ++)
        tm_dec(tm, chunk * BITS + clause_idx, k, ~input[k]);
    } else {
      // Clause didn't vote: random penalty
      unsigned int s = (2 * features) / specificity;
      for (unsigned int r = 0; r < s; r ++) {
        unsigned int random_chunk = fast_rand() % input_chunks;
        unsigned int random_bit = fast_rand() % BITS;
        tk_bits_t random_mask = (tk_bits_t)1 << random_bit;
        tm_dec(tm, chunk * BITS + clause_idx, random_chunk, random_mask);
      }
    }
  } else {
    // Negative feedback
    if (output)
      // Type II feedback: include false input literals
      for (unsigned int k = 0; k < input_chunks; k ++)
        tm_inc(tm, chunk * BITS + clause_idx, k, ~input[k]);
  }
}

static inline void tm_update (
  tk_tsetlin_t *tm,
  tk_bits_t *input,
  tk_bits_t out,
  unsigned int *literals,
  unsigned int *votes,
  unsigned int sample, // sample number

  // the class of this sample (e.g. the MNIST digit number or class id in
  // classifier mode, or the index of the bit we're looking at in encoder mode)
  unsigned int sample_class,

  // the correct vote for this class (in classifier mode, this is always 1
  // because we want to vote 1 for the sample's class, however in encoder mode,
  // since the sample_class is actually the index of the bit in the sample's
  // code we're trying to learn, this defines whether that bit is 0 or 1))
  unsigned int target_vote,

  // the class to which this chunk is assigned (this chunk might be part of the
  // quorum for class 3 (in classifier mode) or bit 3 (in encoder mode) (aka bit
  // 6 in encoder mode); this is of course different than the class label for
  // the sample.
  unsigned int chunk_class,

  // the chunk number (0 to (clauses * classes))
  unsigned int chunk,

  unsigned int thread
) {
  // If you're not confused enough yet, some more detail:
  //
  // Each class has `tm->clauses` clauses, which are grouped into chunks.
  // This routine is processing feedback for one of those chunks.
  // In order to determine the feedback to give, we need to know what class this
  // chunk maps to, and what the class or the bit value is for the sample we're
  // looking at.
  // Sample_class tells you the class of the sample (classifier mode) or the
  // current bit position (encoder mode).
  // Target_vote tells you what the right vote is for this class or bit
  // position (always 1 in classifier mode, since the correct vote for the
  // correct class is 1, but can be 0 or 1 in encoder mode depending on what the
  // value of this bit position is.)
  // Chunk_class tells you the class to which this chunk corresponds.
  // When in classifier mode and chunk_class == sample_class, we're looking at a
  // chunk that maps to the correct class so we want to encourage 1 votes.
  // When in classifier mode and chunk_class != sample_class, we're looking at a
  // chunk that maps to a different class than the class of the sample we're
  // looking at, and therefore we want to encourge 0 votes.
  // When in encoder mode, chunk_class will always equal sample class, since
  // we're only concerned with the bit position to which this chunk corresponds,
  // and the target_vote tells us if, for this sample, the chunk should output 0
  // or 1.


  // If we're looking at a chunk for a different class, we're going to encourage
  // voting 0, but we do that gated by probability so that negatives don't
  // overwhelm positives. Typically if we have 10 classes, we'd set the negative
  // sampling to be 1/10th since this chunk is going to see every single sample
  // and we assume that 9/10ths of them will be for other classes. In encoder
  // mode, chunk_class always equals sample_class so the routine runs always.
  double negative_sampling = tm->negative;
  if (chunk_class != sample_class && fast_chance(1 - negative_sampling))
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

  for (unsigned int j = 0; j < BITS; j += 2) {
    unsigned int pos_clause = j;
    unsigned int neg_clause = j + 1;
    if (neg_clause >= BITS)
      break;
    if (chunk_class == sample_class && target_vote) {
      // Strengthen positive clauses
      if (fast_chance(p))
        apply_feedback(tm, pos_clause, chunk, input, literals, votes, true, thread);
      // Weaken negative clauses
      if (fast_chance(p))
        apply_feedback(tm, neg_clause, chunk, input, literals, votes, false, thread);
    } else { // expected vote is 0
      // Strengthen negative clauses
      if (fast_chance(p))
        apply_feedback(tm, neg_clause, chunk, input, literals, votes, true, thread);
      // Weaken positive clauses
      if (fast_chance(p))
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
    unsigned int j = i == 0 ? 0 : fast_rand() % (i + 1);
    unsigned int t = shuffle[i];
    shuffle[i] = shuffle[j];
    shuffle[j] = t;
  }
}

tk_tsetlin_t *tk_tsetlin_peek (lua_State *L, int i)
{
  return (tk_tsetlin_t *) luaL_checkudata(L, i, TK_TSETLIN_MT);
}

static inline tk_tsetlin_t *tk_tsetlin_alloc (lua_State *L)
{
  tk_tsetlin_t *tm = lua_newuserdata(L, sizeof(tk_tsetlin_t)); // ud
  if (!tm) luaL_error(L, "error in malloc during creation");
  memset(tm, 0, sizeof(tk_tsetlin_t));
  luaL_getmetatable(L, TK_TSETLIN_MT); // ud mt
  lua_setmetatable(L, -2); // ud
  return tm;
}

static inline tk_tsetlin_t *tk_tsetlin_alloc_classifier (lua_State *L, bool has_state)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_CLASSIFIER;
  tm->has_state = has_state;
  return tm;
}

static inline tk_tsetlin_t *tk_tsetlin_alloc_encoder (lua_State *L, bool has_state)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_ENCODER;
  tm->has_state = has_state;
  return tm;
}

static inline void tk_tsetlin_setup_threads (lua_State *L, tk_tsetlin_t *tm, unsigned int n_threads);

static void tk_classifier_predict_reduce_thread (
  tk_tsetlin_t *tm,
  unsigned int sfirst,
  unsigned int slast
) {
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int *results = tm->results;
  long int sums[tm->classes];
  for (unsigned int s = sfirst; s <= slast; s ++) {
    for (unsigned int i = 0; i < tm->classes; i ++)
      sums[i] = 0;
    for (unsigned int t = 0; t < tm->pool->n_threads; t ++) {
      tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[t].data;
      for (unsigned int chunk = data->cfirst; chunk <= data->clast; chunk ++) {
        unsigned int class = chunk / clause_chunks;
        sums[class] += tm_state_scores(tm, t, chunk, s);
      }
    }
    long int maxval = -INT64_MAX;
    unsigned int maxclass = 0;
    for (unsigned int class = 0; class < tm->classes; class ++) {
      if (sums[class] > maxval) {
        maxval = sums[class];
        maxclass = class;
      }
    }
    results[s] = maxclass;
  }
}

static void tk_classifier_predict_thread (
  tk_tsetlin_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  unsigned int input_chunks = tm->input_chunks;
  unsigned int literals[BITS];
  unsigned int votes[BITS];
  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {
    for (unsigned int s = 0; s < n; s ++) {
      tk_bits_t *input = ps + s * input_chunks;
      tk_bits_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk);
      long int score = tk_tsetlin_sums(tm, out, votes);
      tm_state_scores(tm, thread, chunk, s) = score;
    }
  }
}

static void tk_encoder_predict_thread (
  tk_tsetlin_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  return tk_classifier_predict_thread(tm, n, ps, cfirst, clast, thread);
}

static void tk_classifier_setup_thread (
  tk_tsetlin_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  unsigned int m = tm->state_bits - 1;
  unsigned int include_threshold = 1 << (m - 1);
  unsigned int initial_state = include_threshold - 1;
  for (unsigned int clause_chunk = cfirst; clause_chunk <= clast; clause_chunk ++) {
    for (unsigned int clause_chunk_pos = 0; clause_chunk_pos < BITS; clause_chunk_pos ++) {
      unsigned int clause = clause_chunk * BITS + clause_chunk_pos;
      for (unsigned int input_chunk = 0; input_chunk < tm->input_chunks; input_chunk ++) {
        tk_bits_t *actions = tm_state_actions(tm, clause);
        tk_bits_t *counts = tm_state_counts(tm, clause, input_chunk);
        actions[input_chunk] = 0;
        for (unsigned int b = 0; b < m; b ++) {
          if (initial_state & (1 << b))
            counts[b] = ALL_MASK;
          else
            counts[b] = ZERO_MASK;
        }
      }
    }
  }
}

static void tk_encoder_setup_thread (
  tk_tsetlin_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_classifier_setup_thread(tm, n, ps, cfirst, clast, thread);
}

static void tk_classifier_train_thread (
  tk_tsetlin_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int *ss,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_tsetlin_init_shuffle(tm_state_shuffle(tm, thread), n);
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int *shuffle = tm_state_shuffle(tm, thread);
  unsigned int literals[BITS];
  unsigned int votes[BITS];
  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {
    unsigned int chunk_class = chunk / clause_chunks;
    for (unsigned int i = 0; i < n; i ++) {
      unsigned int sample = shuffle[i];
      tk_bits_t *input = ps + sample * input_chunks;
      tk_bits_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk);
      unsigned int sample_class = ss[sample];
      tm_update(tm, input, out, literals, votes, sample, sample_class, 1, chunk_class, chunk, thread);
    }
  }
}

static void tk_encoder_train_thread (
  tk_tsetlin_t *tm,
  tk_bits_t *ps,
  tk_bits_t *ls,
  unsigned int n,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_tsetlin_init_shuffle(tm_state_shuffle(tm, thread), n);
  unsigned int class_chunks = tm->class_chunks;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int *shuffle = tm_state_shuffle(tm, thread);
  unsigned int literals[BITS];
  unsigned int votes[BITS];
  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {
    unsigned int chunk_class = chunk / clause_chunks;
    unsigned int enc_chunk = BITS_BYTE(chunk_class);
    unsigned int enc_bit = BITS_BIT(chunk_class);
    for (unsigned int i = 0; i < n; i ++) {
      unsigned int sample = shuffle[i];
      tk_bits_t *input = ps + sample * input_chunks;
      tk_bits_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk);
      bool target_vote = (ls[sample * class_chunks + enc_chunk] & ((tk_bits_t) 1 << enc_bit)) > 0;
      tm_update(tm, input, out, literals, votes, sample, chunk_class, target_vote, chunk_class, chunk, thread);
    }
  }
}

static void tk_encoder_predict_reduce_thread (
  tk_tsetlin_t *tm,
  unsigned int sfirst,
  unsigned int slast
) {
  tk_bits_t *encodings = tm->encodings;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int classes = tm->classes;
  unsigned int class_chunks = tm->class_chunks;
  long int votes[classes];
  for (unsigned int s = sfirst; s <= slast; s ++) {
    for (unsigned int i = 0; i < classes; i ++)
      votes[i] = 0;
    for (unsigned int t = 0; t < tm->pool->n_threads; t ++) {
      tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[t].data;
      for (unsigned int chunk = data->cfirst; chunk <= data->clast; chunk ++) {
        unsigned int class = chunk / clause_chunks;
        votes[class] += tm_state_scores(tm, t, chunk, s);
      }
    }
    tk_bits_t *e = encodings + s * class_chunks;
    for (unsigned int class = 0; class < tm->classes; class ++) {
      unsigned int chunk = BITS_BYTE(class);
      unsigned int pos = BITS_BIT(class);
      if (votes[class] > 0)
        e[chunk] |= ((tk_bits_t)1 << pos);
      else
        e[chunk] &= ~((tk_bits_t)1 << pos);
    }
  }
}

static void tk_tsetlin_worker (void *dp, int sig)
{
  tk_tsetlin_stage_t stage = (tk_tsetlin_stage_t) sig;
  tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) dp;
  switch (stage) {
    case TM_SEED:
      seed_rand(data->index);
      break;
    case TM_CLASSIFIER_SETUP:
      tk_classifier_setup_thread(
        data->tm,
        data->train.n,
        data->train.ps,
        data->cfirst,
        data->clast,
        data->index);
      break;
    case TM_CLASSIFIER_TRAIN:
      tk_classifier_train_thread(
        data->tm,
        data->train.n,
        data->train.ps,
        data->train.ss,
        data->cfirst,
        data->clast,
        data->index);
      break;
    case TM_CLASSIFIER_PREDICT:
      tk_classifier_predict_thread(
        data->tm,
        data->predict.n,
        data->predict.ps,
        data->cfirst,
        data->clast,
        data->index);
      break;
    case TM_CLASSIFIER_PREDICT_REDUCE:
      tk_classifier_predict_reduce_thread(
        data->tm,
        data->sfirst,
        data->slast);
      break;
    case TM_ENCODER_SETUP:
      tk_encoder_setup_thread(
        data->tm,
        data->train.n,
        data->train.ps,
        data->cfirst,
        data->clast,
        data->index);
      break;
    case TM_ENCODER_PREDICT:
      tk_encoder_predict_thread(
        data->tm,
        data->predict.n,
        data->predict.ps,
        data->cfirst,
        data->clast,
        data->index);
      break;
    case TM_ENCODER_PREDICT_REDUCE:
      tk_encoder_predict_reduce_thread(
        data->tm,
        data->sfirst,
        data->slast);
      break;
    case TM_ENCODER_TRAIN:
      tk_encoder_train_thread(
        data->tm,
        data->train.ps,
        data->train.ls,
        data->train.n,
        data->cfirst,
        data->clast,
        data->index);
      break;
    default:
      assert(false);
      break;
  }
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
  double targetf,
  double negative,
  double specificity,
  unsigned int n_threads
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
  tm->negative = negative < 0 ? 1.0 / (double) classes : negative; // Note: unused in encoder
  tm->classes = classes;
  tm->class_chunks = BITS_BYTES(tm->classes);
  tm->clauses = BITS_BYTES(clauses) * BITS;
  tm->clause_tolerance = clause_tolerance;
  tm->clause_maximum = clause_maximum;
  tm->target =
    targetf < 0
      ? sqrt((double) tm->clauses / 2.0) * (double) clause_tolerance
      : ceil(targetf >= 1 ? targetf : fmaxf(1.0, (double) tm->clauses * targetf));
  tm->features = features;
  tm->state_bits = state_bits;
  tm->input_bits = 2 * tm->features;
  uint64_t tail_bits = tm->input_bits & (BITS - 1);
  tm->tail_mask = tail_bits ? (tk_bits_t)((1u << tail_bits) - 1) : (tk_bits_t) ~0;
  tm->input_chunks = BITS_BYTES(tm->input_bits);
  tm->clause_chunks = BITS_BYTES(tm->clauses);
  tm->state_chunks = tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tm->action_chunks = tm->classes * tm->clauses * tm->input_chunks;
  tm->state = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->state_chunks, BITS);
  tm->actions = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->action_chunks, BITS);
  tm->specificity = specificity;
  if (!(tm->state && tm->actions))
    luaL_error(L, "error in malloc during creation of classifier");
  tm->pool = tk_threads_create(L, n_threads, tk_tsetlin_worker);
  tk_tsetlin_setup_threads(L, tm, n_threads);
  tk_threads_signal(tm->pool, TM_SEED, 0);
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
  double targetf,
  double negative,
  double specificity,
  unsigned int n_threads
) {
  tk_tsetlin_init_classifier(
    L, tm, encoding_bits, features, clauses, clause_tolerance, clause_maximum,
    state_bits, targetf, negative, specificity,
    n_threads);
  return 0;
}

static inline void tk_tsetlin_create_classifier (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, true);
  lua_insert(L, 1);
  tk_tsetlin_init_classifier(L, tm,
      tk_lua_fcheckunsigned(L, 2, "create classifier", "classes"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "features"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "clauses"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "clause_tolerance"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "clause_maximum"),
      tk_lua_foptunsigned(L, 2, "create classifier", "state", 8),
      tk_lua_foptposdouble(L, 2, "create classifier", "target", -1.0),
      tk_lua_foptposdouble(L, 2, "create classifier", "negative", -1.0),
      tk_lua_fcheckposdouble(L, 2, "create classifier", "specificity"),
      tk_threads_getn(L, 2, "create classifier", "threads"));
  lua_settop(L, 1);
}

static inline void tk_tsetlin_create_encoder (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, true);
  lua_insert(L, 1);
  tk_tsetlin_init_encoder(L, tm,
      tk_lua_fcheckunsigned(L, 2, "create encoder", "hidden"),
      tk_lua_fcheckunsigned(L, 2, "create encoder", "visible"),
      tk_lua_fcheckunsigned(L, 2, "create encoder", "clauses"),
      tk_lua_fcheckunsigned(L, 2, "create encoder", "clause_tolerance"),
      tk_lua_fcheckunsigned(L, 2, "create encoder", "clause_maximum"),
      tk_lua_foptunsigned(L, 2, "create encoder", "state", 8),
      tk_lua_foptposdouble(L, 2, "create encoder", "target", -1.0),
      tk_lua_foptposdouble(L, 2, "create encoder", "negative", -1.0), // Note: unused in encoder
      tk_lua_fcheckposdouble(L, 2, "create encoder", "specificity"),
      tk_threads_getn(L, 2, "create encoder", "threads"));
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

// TODO: expose via the api
static inline void tk_tsetlin_shrink (tk_tsetlin_t *tm)
{
  if (tm == NULL) return;
  free(tm->state); tm->state = NULL;
  if (tm->initialized_threads)
    for (unsigned int i = 0; i < tm->pool->n_threads; i ++) {
      tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[i].data;
      free(data->shuffle); data->shuffle = NULL;
      free(data->scores); data->scores = NULL;
    }
}

static inline void _tk_tsetlin_destroy (tk_tsetlin_t *tm)
{
  if (tm == NULL) return;
  if (tm->destroyed) return;
  tm->destroyed = true;
  tk_tsetlin_shrink(tm);
  free(tm->actions); tm->actions = NULL;
  if (numa_available() == -1) {
    free(tm->results); tm->results = NULL;
    free(tm->encodings); tm->encodings = NULL;
  } else {
    numa_free(tm->results, tm->results_len); tm->results = NULL; tm->results_len = 0;
    numa_free(tm->encodings, tm->encodings_len); tm->encodings = NULL; tm->encodings_len = 0;
  }
  for (unsigned int i = 0; i < tm->pool->n_threads; i ++)
    free(tm->pool->threads[i].data);
  tk_threads_destroy(tm->pool);
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  _tk_tsetlin_destroy(tm);
  return 0;
}

static inline void tk_tsetlin_setup_thread_samples (
  tk_tsetlin_t *tm,
  unsigned int n
) {
  for (unsigned int i = 0; i < tm->pool->n_threads; i ++) {
    tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[i].data;
    tk_thread_range(i, tm->pool->n_threads, n, &data->sfirst, &data->slast);
  }
}

static inline int tk_tsetlin_predict_classifier (
  lua_State *L,
  tk_tsetlin_t *tm
) {
  lua_settop(L, 3);
  tk_bits_t *ps = (tk_bits_t *) tk_lua_checkstring(L, 2, "argument 1 is not a raw bit-matrix of samples");
  unsigned int n = tk_lua_checkunsigned(L, 3, "argument 2 is not an integer n_samples");
  tm->results = tk_ensure_interleaved(L, &tm->results_len, tm->results, n * sizeof(unsigned int), false);
  for (unsigned int i = 0; i < tm->pool->n_threads; i ++) {
    unsigned int chunks = tm_state_clause_chunks(tm, i);
    tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[i].data;
    data->predict.n = n;
    data->predict.ps = ps;
    data->scores = tk_realloc(L, data->scores, chunks * n * sizeof(long int));
  }
  tk_tsetlin_setup_thread_samples(tm, n);
  tk_threads_signal(tm->pool, TM_CLASSIFIER_PREDICT, 0);
  tk_threads_signal(tm->pool, TM_CLASSIFIER_PREDICT_REDUCE, 0);
  lua_pushlstring(L, (char *) tm->results, n * sizeof(unsigned int));
  return 1;
}

static inline int tk_tsetlin_predict_encoder (
  lua_State *L,
  tk_tsetlin_t *tm
) {
  lua_settop(L, 3);
  tk_bits_t *ps = (tk_bits_t *) tk_lua_checkstring(L, 2, "argument 1 is not a raw bit-matrix of samples");
  unsigned int n = tk_lua_checkunsigned(L, 3, "argument 2 is not an integer n_samples");
  tm->encodings = tk_ensure_interleaved(L, &tm->encodings_len, tm->encodings, n * tm->class_chunks * sizeof(tk_bits_t), false);
  for (unsigned int i = 0; i < tm->pool->n_threads; i ++) {
    tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[i].data;
    unsigned int chunks = tm_state_clause_chunks(tm, i);
    data->predict.n = n;
    data->predict.ps = ps;
    data->scores = tk_realloc(L, data->scores, chunks * n * sizeof(long int));
  }
  tk_tsetlin_setup_thread_samples(tm, n);
  tk_threads_signal(tm->pool, TM_ENCODER_PREDICT, 0);
  tk_threads_signal(tm->pool, TM_ENCODER_PREDICT_REDUCE, 0);
  lua_pushlstring(L, (char *) tm->encodings, n * tm->class_chunks * sizeof(tk_bits_t));
  return 1;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
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
  tk_bits_t *ps = (tk_bits_t *) tk_lua_fcheckstring(L, 2, "train", "problems");
  unsigned int *ss = (unsigned int *) tk_lua_fcheckstring(L, 2, "train", "solutions");
  unsigned int max_iter =  tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }
  for (unsigned int i = 0; i < tm->pool->n_threads; i ++) {
    tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[i].data;
    data->train.n = n;
    data->train.ps = ps;
    data->train.ss = ss;
    data->shuffle = tk_realloc(L, data->shuffle, n * sizeof(unsigned int));
  }
  tk_tsetlin_setup_thread_samples(tm, n);
  tk_threads_signal(tm->pool, TM_CLASSIFIER_SETUP, 0);
  for (unsigned int i = 0; i < max_iter; i ++) {
    tk_threads_signal(tm->pool, TM_CLASSIFIER_TRAIN, 0);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      lua_call(L, 1, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break;
      }
      lua_pop(L, 1);
    }
  }
  tk_tsetlin_shrink(tm);
  tm->trained = true;
  return 0;
}

static inline int tk_tsetlin_train_encoder (
  lua_State *L,
  tk_tsetlin_t *tm
) {
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  tk_bits_t *ps = (tk_bits_t *) tk_lua_fcheckustring(L, 2, "train", "sentences");
  tk_bits_t *ls = (tk_bits_t *) tk_lua_fcheckustring(L, 2, "train", "codes");
  unsigned int max_iter =  tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  tm->encodings = tk_ensure_interleaved(L, &tm->encodings_len, tm->encodings, n * tm->class_chunks * sizeof(tk_bits_t), false);
  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }
  for (unsigned int i = 0; i < tm->pool->n_threads; i ++) {
    tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[i].data;
    unsigned int chunks = tm_state_clause_chunks(tm, i);
    data->predict.n = n;
    data->predict.ps = ps;
    data->scores = tk_realloc(L, data->scores, chunks * n * sizeof(long int));
    data->train.n = n;
    data->train.ps = ps;
    data->train.ls = ls;
    data->shuffle = tk_realloc(L, data->shuffle, n * sizeof(unsigned int));
  }
  tk_tsetlin_setup_thread_samples(tm, n);
  tk_threads_signal(tm->pool, TM_ENCODER_SETUP, 0);
  for (unsigned int i = 0; i < max_iter; i ++) {
    tk_threads_signal(tm->pool, TM_ENCODER_TRAIN, 0);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      lua_call(L, 1, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break;
      }
      lua_pop(L, 1);
    }
  }
  tk_tsetlin_shrink(tm);
  tm->trained = true;
  return 0;
}

static inline int tk_tsetlin_train (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  if (!tm->has_state)
    luaL_error(L, "can't train a model loaded without state");
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_train_classifier(L, tm);
    case TM_ENCODER:
      return tk_tsetlin_train_encoder(L, tm);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in train");
  }
}

static inline void tk_tsetlin_setup_threads (
  lua_State *L,
  tk_tsetlin_t *tm,
  unsigned int n_threads
) {
  tk_threadpool_t *pool = tm->pool;
  if (!tm->initialized_threads) {
    for (unsigned int i = 0; i < tm->pool->n_threads; i ++) {
      tk_tsetlin_thread_t *data = tk_malloc(L, sizeof(tk_tsetlin_thread_t));
      memset(data, 0, sizeof(tk_tsetlin_thread_t));
      pool->threads[i].data = data;
      data->tm = tm;
      data->index = i;
      tk_thread_range(i, tm->pool->n_threads, tm->clause_chunks * tm->classes, &data->cfirst, &data->clast);
    }
    tm->initialized_threads = true;
  }
}

static inline void _tk_tsetlin_persist_classifier (lua_State *L, tk_tsetlin_t *tm, FILE *fh, bool persist_state)
{
  tk_lua_fwrite(L, &tm->classes, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->features, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clauses, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clause_tolerance, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clause_maximum, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->target, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->state_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->input_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->input_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clause_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->state_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->action_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &tm->tail_mask, sizeof(tk_bits_t), 1, fh);
  tk_lua_fwrite(L, tm->actions, sizeof(tk_bits_t), tm->action_chunks, fh);
}

static inline void tk_tsetlin_persist_classifier (lua_State *L, tk_tsetlin_t *tm, FILE *fh, bool persist_state)
{
  _tk_tsetlin_persist_classifier(L, tm, fh, persist_state);
}

static inline int tk_tsetlin_persist (lua_State *L)
{
  lua_settop(L, 3);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  bool tostr = lua_type(L, 2) == LUA_TNIL;
  FILE *fh = tostr ? tk_lua_tmpfile(L) : tk_lua_fopen(L, tk_lua_checkstring(L, 2, "persist path"), "w");
  bool persist_state = lua_toboolean(L, 3);
  if (persist_state && !tm->has_state)
    luaL_error(L, "can't persist the state of a model loaded without state");
  tk_lua_fwrite(L, &tm->type, sizeof(tk_tsetlin_type_t), 1, fh);
  tk_lua_fwrite(L, &persist_state, sizeof(bool), 1, fh);
  tk_tsetlin_persist_classifier(L, tm, fh, persist_state);
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

static inline void _tk_tsetlin_load_classifier (lua_State *L, tk_tsetlin_t *tm, FILE *fh, bool read_state, bool has_state, unsigned int n_threads)
{
  tk_lua_fread(L, &tm->classes, sizeof(unsigned int), 1, fh);
  tm->class_chunks = BITS_BYTES(tm->classes);
  tk_lua_fread(L, &tm->features, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clauses, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clause_tolerance, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clause_maximum, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->target, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->state_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->input_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->input_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clause_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->state_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->action_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fread(L, &tm->tail_mask, sizeof(tk_bits_t), 1, fh);
  tm->actions = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->action_chunks, BITS);
  tk_lua_fread(L, tm->actions, sizeof(tk_bits_t), tm->action_chunks, fh);
  tm->pool = tk_threads_create(L, n_threads, tk_tsetlin_worker);
  tk_tsetlin_setup_threads(L, tm, n_threads);
  tk_threads_signal(tm->pool, TM_SEED, 0);
}

static inline void tk_tsetlin_load_classifier (lua_State *L, FILE *fh, bool read_state, bool has_state)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, read_state);
  unsigned int n_threads = tk_threads_getn(L, 2, "threads", NULL);
  _tk_tsetlin_load_classifier(L, tm, fh, read_state, has_state, n_threads);
}

static inline void tk_tsetlin_load_encoder (lua_State *L, FILE *fh, bool read_state, bool has_state)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, read_state);
  unsigned int n_threads = tk_threads_getn(L, 2, "threads", NULL);
  _tk_tsetlin_load_classifier(L, tm, fh, read_state, has_state, n_threads);
}

// TODO: Merge malloc/assignment logic from load_* and create_* to reduce
// chances for coding errors
static inline int tk_tsetlin_load (lua_State *L)
{
  lua_settop(L, 3);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && lua_toboolean(L, 2);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  bool read_state = lua_toboolean(L, 3);
  tk_tsetlin_type_t type;
  bool has_state;
  tk_lua_fread(L, &type, sizeof(type), 1, fh);
  tk_lua_fread(L, &has_state, sizeof(bool), 1, fh);
  if (read_state && !has_state)
    luaL_error(L, "read_state is true but state not persisted");
  switch (type) {
    case TM_CLASSIFIER:
      tk_tsetlin_load_classifier(L, fh, read_state, has_state);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_ENCODER:
      tk_tsetlin_load_encoder(L, fh, read_state, has_state);
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
  { "train", tk_tsetlin_train },
  { "predict", tk_tsetlin_predict },
  { "destroy", tk_tsetlin_destroy },
  { "persist", tk_tsetlin_persist },
  { "type", tk_tsetlin_type },
  { "create", tk_tsetlin_create },
  { "load", tk_tsetlin_load },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tk_tsetlin_fns, 0); // t
  lua_pushinteger(L, BITS); // b
  lua_setfield(L, -2, "align"); // t
  luaL_newmetatable(L, TK_TSETLIN_MT); // t mt
  lua_pushcfunction(L, tk_tsetlin_destroy); // t mt fn
  lua_setfield(L, -2, "__gc"); // t mt
  lua_pop(L, 1); // t
  return 1;
}
