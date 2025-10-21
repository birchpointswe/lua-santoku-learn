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
#include <santoku/threads.h>
#include <santoku/cvec.h>

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
  tk_threadpool_t *pool;
  uint64_t sfirst;
  uint64_t slast;
  uint64_t cfirst;
  uint64_t clast;
  unsigned int index;
  struct {
    unsigned int n;
    tk_cvec_t *ps;
    tk_cvec_t *ls;
    tk_ivec_t *ss;
  } train;
  struct {
    unsigned int n;
    tk_cvec_t *ps;
  } predict;
  unsigned int *shuffle;
  long int *scores;
} tk_tsetlin_thread_t;

typedef struct tk_tsetlin_s {
  tk_tsetlin_type_t type;
  bool has_state;
  bool trained;
  bool destroyed;
  bool reusable;
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
} tk_tsetlin_t;

#define tm_thread_data(pool, thread) \
  ((tk_tsetlin_thread_t *) (pool)->threads[thread].data)

#define tm_state_shuffle(pool, thread) \
  (tm_thread_data(pool, thread)->shuffle)

#define tm_state_scores(pool, thread, chunk, sample) \
  (tm_thread_data(pool, thread)->scores[ \
    ((chunk) - tm_thread_data(pool, thread)->cfirst) * tm_thread_data(pool, thread)->predict.n + \
    (sample)])

#define tm_state_clause_chunks(pool, thread) \
  (tm_thread_data(pool, thread)->clast - tm_thread_data(pool, thread)->cfirst + 1)

static inline uint8_t tk_tsetlin_calculate (
  tk_tsetlin_t *tm,
  char *input,
  unsigned int *literalsp,
  unsigned int *votesp,
  unsigned int chunk
) {
  uint8_t out = 0;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int tolerance = tm->clause_tolerance;
  for (unsigned int j = 0; j < TK_CVEC_BITS; j ++) {
    unsigned int clause = chunk * TK_CVEC_BITS + j;
    char *actions = tk_automata_actions(&tm->automata, clause);
    unsigned int failed = 0;
    unsigned int literals = 0;
    for (unsigned int k = 0; k < input_chunks - 1; k ++) {
      uint8_t inp = ((uint8_t *)input)[k];
      uint8_t act = ((uint8_t *)actions)[k];
      literals += (unsigned int) tk_cvec_byte_popcount(act);
      failed += (unsigned int) tk_cvec_byte_popcount(act & (~inp));
    }
    uint8_t last_act = ((uint8_t *)actions)[input_chunks - 1] & tm->tail_mask;
    literals += (unsigned int) tk_cvec_byte_popcount(last_act);
    failed += (unsigned int) tk_cvec_byte_popcount(last_act & (~((uint8_t *)input)[input_chunks - 1]) & tm->tail_mask);
    long int votes;
    if (literals == 0) {
      votes = (long int) tolerance;
    } else {
      votes = (literals < tolerance ? (long int) literals : (long int) tolerance) - (long int) failed;
      if (votes < 0)
        votes = 0;
    }
    if (votes > 0)
      out |= (1 << j);
    literalsp[j] = literals;
    votesp[j] = (unsigned int) votes;
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
  if (positive_feedback) {
    // Positive feedback
    if (output) {
      // Type Ia: clause voted
      if (literals[clause_idx] < max_literals)
        for (unsigned int k = 0; k < input_chunks; k ++)
          tk_automata_inc(&tm->automata, chunk * TK_CVEC_BITS + clause_idx, k, ((uint8_t *)input)[k]);
      // Type Ib: deterministic negative feedback
      for (unsigned int k = 0; k < input_chunks; k ++)
        tk_automata_dec(&tm->automata, chunk * TK_CVEC_BITS + clause_idx, k, ~((uint8_t *)input)[k]);
    } else {
      // Clause didn't vote: random penalty
      unsigned int s = (2 * features) / specificity;
      for (unsigned int r = 0; r < s; r ++) {
        unsigned int random_chunk = tk_fast_random() % input_chunks;
        unsigned int random_bit = tk_fast_random() % TK_CVEC_BITS;
        uint8_t random_mask = 1 << random_bit;
        tk_automata_dec(&tm->automata, chunk * TK_CVEC_BITS + clause_idx, random_chunk, random_mask);
      }
    }
  } else {
    // Negative feedback
    if (output)
      // Type II feedback: include false input literals
      for (unsigned int k = 0; k < input_chunks; k ++)
        tk_automata_inc(&tm->automata, chunk * TK_CVEC_BITS + clause_idx, k, ~((uint8_t *)input)[k]);
  }
}

static inline void tm_update (
  tk_tsetlin_t *tm,
  char *input,
  uint8_t out,
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
      // Strengthen positive clauses
      if (tk_fast_chance(p))
        apply_feedback(tm, pos_clause, chunk, input, literals, votes, true, thread);
      // Weaken negative clauses
      if (tk_fast_chance(p))
        apply_feedback(tm, neg_clause, chunk, input, literals, votes, false, thread);
    } else { // expected vote is 0
      // Strengthen negative clauses
      if (tk_fast_chance(p))
        apply_feedback(tm, neg_clause, chunk, input, literals, votes, true, thread);
      // Weaken positive clauses
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


static void tk_classifier_predict_reduce_thread (
  tk_tsetlin_t *tm,
  tk_threadpool_t *pool,
  unsigned int sfirst,
  unsigned int slast
) {
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int *results = tm->results;
  long int sums[tm->classes];
  for (unsigned int s = sfirst; s <= slast; s ++) {
    for (unsigned int i = 0; i < tm->classes; i ++)
      sums[i] = 0;
    for (unsigned int t = 0; t < pool->n_threads; t ++) {
      tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) pool->threads[t].data;
      for (unsigned int chunk = data->cfirst; chunk <= data->clast; chunk ++) {
        unsigned int class = chunk / clause_chunks;
        sums[class] += tm_state_scores(pool, t, chunk, s);
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
  tk_threadpool_t *pool,
  unsigned int n,
  tk_cvec_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  unsigned int input_chunks = tm->input_chunks;
  unsigned int literals[TK_CVEC_BITS];
  unsigned int votes[TK_CVEC_BITS];
  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {
    for (unsigned int s = 0; s < n; s ++) {
      char *input = ps->a + s * input_chunks;
      uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk);
      long int score = tk_tsetlin_sums(tm, out, votes);
      tm_state_scores(pool, thread, chunk, s) = score;
    }
  }
}

static void tk_encoder_predict_thread (
  tk_tsetlin_t *tm,
  tk_threadpool_t *pool,
  unsigned int n,
  tk_cvec_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  return tk_classifier_predict_thread(tm, pool, n, ps, cfirst, clast, thread);
}

static void tk_classifier_setup_thread (
  tk_tsetlin_t *tm,
  unsigned int n,
  tk_cvec_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  uint64_t first_clause = cfirst * TK_CVEC_BITS;
  uint64_t last_clause = clast * TK_CVEC_BITS + TK_CVEC_BITS - 1;
  tk_automata_setup(&tm->automata, first_clause, last_clause);
}

static void tk_encoder_setup_thread (
  tk_tsetlin_t *tm,
  unsigned int n,
  tk_cvec_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_classifier_setup_thread(tm, n, ps, cfirst, clast, thread);
}

static void tk_classifier_train_thread (
  tk_tsetlin_t *tm,
  tk_threadpool_t *pool,
  unsigned int n,
  tk_cvec_t *ps,
  tk_ivec_t *ss,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_tsetlin_init_shuffle(tm_state_shuffle(pool, thread), n);
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int *shuffle = tm_state_shuffle(pool, thread);
  unsigned int literals[TK_CVEC_BITS];
  unsigned int votes[TK_CVEC_BITS];
  int64_t *lbls = ss->a;
  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {
    unsigned int chunk_class = chunk / clause_chunks;
    for (unsigned int i = 0; i < n; i ++) {
      unsigned int sample = shuffle[i];
      char *input = ps->a + sample * input_chunks;
      uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk);
      unsigned int sample_class = lbls[sample];
      tm_update(tm, input, out, literals, votes, sample, sample_class, 1, chunk_class, chunk, thread);
    }
  }
}

static void tk_encoder_train_thread (
  tk_tsetlin_t *tm,
  tk_threadpool_t *pool,
  tk_cvec_t *ps,
  tk_cvec_t *ls,
  unsigned int n,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_tsetlin_init_shuffle(tm_state_shuffle(pool, thread), n);
  unsigned int class_chunks = tm->class_chunks;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int *shuffle = tm_state_shuffle(pool, thread);
  unsigned int literals[TK_CVEC_BITS];
  unsigned int votes[TK_CVEC_BITS];
  char *lbls = ls->a;
  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {
    unsigned int chunk_class = chunk / clause_chunks;
    unsigned int enc_chunk = TK_CVEC_BITS_BYTE(chunk_class);
    unsigned int enc_bit = TK_CVEC_BITS_BIT(chunk_class);
    for (unsigned int i = 0; i < n; i ++) {
      unsigned int sample = shuffle[i];
      char *input = ps->a + sample * input_chunks;
      uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk);
      bool target_vote = (((uint8_t *)lbls)[sample * class_chunks + enc_chunk] & (1 << enc_bit)) > 0;
      tm_update(tm, input, out, literals, votes, sample, chunk_class, target_vote, chunk_class, chunk, thread);
    }
  }
}

static void tk_encoder_predict_reduce_thread (
  tk_tsetlin_t *tm,
  tk_threadpool_t *pool,
  unsigned int sfirst,
  unsigned int slast
) {
  char *encodings = tm->encodings;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int classes = tm->classes;
  unsigned int class_chunks = tm->class_chunks;
  long int votes[classes];
  for (unsigned int s = sfirst; s <= slast; s ++) {
    for (unsigned int i = 0; i < classes; i ++)
      votes[i] = 0;
    for (unsigned int t = 0; t < pool->n_threads; t ++) {
      tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) pool->threads[t].data;
      for (unsigned int chunk = data->cfirst; chunk <= data->clast; chunk ++) {
        unsigned int class = chunk / clause_chunks;
        votes[class] += tm_state_scores(pool, t, chunk, s);
      }
    }
    uint8_t *e = (uint8_t *)(encodings + s * class_chunks);
    for (unsigned int class = 0; class < tm->classes; class ++) {
      unsigned int chunk = TK_CVEC_BITS_BYTE(class);
      unsigned int pos = TK_CVEC_BITS_BIT(class);
      if (votes[class] > 0)
        e[chunk] |= (1 << pos);
      else
        e[chunk] &= ~(1 << pos);
    }
  }
}

static void tk_tsetlin_worker (void *dp, int sig)
{
  tk_tsetlin_stage_t stage = (tk_tsetlin_stage_t) sig;
  tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) dp;
  switch (stage) {
    case TM_SEED:
      tk_fast_seed(data->index);
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
        data->pool,
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
        data->pool,
        data->predict.n,
        data->predict.ps,
        data->cfirst,
        data->clast,
        data->index);
      break;
    case TM_CLASSIFIER_PREDICT_REDUCE:
      tk_classifier_predict_reduce_thread(
        data->tm,
        data->pool,
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
        data->pool,
        data->predict.n,
        data->predict.ps,
        data->cfirst,
        data->clast,
        data->index);
      break;
    case TM_ENCODER_PREDICT_REDUCE:
      tk_encoder_predict_reduce_thread(
        data->tm,
        data->pool,
        data->sfirst,
        data->slast);
      break;
    case TM_ENCODER_TRAIN:
      tk_encoder_train_thread(
        data->tm,
        data->pool,
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
  tm->negative = negative < 0 ? 1.0 / (double) classes : negative; // Note: unused in encoder
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
  double targetf,
  double negative,
  double specificity
) {
  tk_tsetlin_init_classifier(
    L, tm, encoding_bits, features, clauses, clause_tolerance, clause_maximum,
    state_bits, targetf, negative, specificity);
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
      tk_lua_fcheckposdouble(L, 2, "create classifier", "specificity"));
  tm->reusable = tk_lua_foptboolean(L, 2, "create classifier", "reusable", false);
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
      tk_lua_fcheckposdouble(L, 2, "create encoder", "specificity"));
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

// TODO: expose via the api
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
  lua_settop(L, 4);
  tk_cvec_t *ps = tk_cvec_peek(L, 2, "problems");
  unsigned int n = tk_lua_checkunsigned(L, 3, "argument 2 is not an integer n_samples");
  unsigned int n_threads = tk_threads_getn(L, 4, "predict", NULL);

  // Only reallocate if needed
  size_t needed = n * sizeof(unsigned int);
  if (needed > tm->results_len) {
    tm->results = tk_realloc(L, tm->results, needed);
    tm->results_len = needed;
  }

  tk_tsetlin_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_tsetlin_thread_t));
  memset(threads, 0, n_threads * sizeof(tk_tsetlin_thread_t));
  tk_threadpool_t *pool = tk_threads_create(0, n_threads, tk_tsetlin_worker);

  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_tsetlin_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->tm = tm;
    data->pool = pool;
    data->index = i;
    tk_thread_range(i, n_threads, tm->clause_chunks * tm->classes, &data->cfirst, &data->clast);
    tk_thread_range(i, n_threads, n, &data->sfirst, &data->slast);
    unsigned int chunks = tm_state_clause_chunks(pool, i);
    data->predict.n = n;
    data->predict.ps = ps;
    data->scores = tk_malloc(L, chunks * n * sizeof(long int));
  }

  tk_threads_signal(pool, TM_CLASSIFIER_PREDICT, 0);
  tk_threads_signal(pool, TM_CLASSIFIER_PREDICT_REDUCE, 0);

  for (unsigned int i = 0; i < n_threads; i ++) {
    free(threads[i].scores);
  }
  tk_threads_destroy(pool);
  free(threads);

  tk_ivec_t *out = tk_ivec_create(L, n, 0, 0);
  for (uint64_t i = 0; i < n; i ++)
    out->a[i] = tm->results[i];
  return 1;
}

static inline int tk_tsetlin_predict_encoder (
  lua_State *L,
  tk_tsetlin_t *tm
) {
  lua_settop(L, 4);
  tk_cvec_t *ps = tk_cvec_peek(L, 2, "problems");
  unsigned int n = tk_lua_checkunsigned(L, 3, "argument 2 is not an integer n_samples");
  unsigned int n_threads = tk_threads_getn(L, 4, "predict", NULL);

  // Only reallocate if needed
  size_t needed = n * tm->class_chunks;
  if (needed > tm->encodings_len) {
    tm->encodings = (char *)tk_realloc(L, tm->encodings, needed);
    tm->encodings_len = needed;
  }

  tk_tsetlin_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_tsetlin_thread_t));
  memset(threads, 0, n_threads * sizeof(tk_tsetlin_thread_t));
  tk_threadpool_t *pool = tk_threads_create(0, n_threads, tk_tsetlin_worker);

  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_tsetlin_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->tm = tm;
    data->pool = pool;
    data->index = i;
    tk_thread_range(i, n_threads, tm->clause_chunks * tm->classes, &data->cfirst, &data->clast);
    tk_thread_range(i, n_threads, n, &data->sfirst, &data->slast);
    unsigned int chunks = tm_state_clause_chunks(pool, i);
    data->predict.n = n;
    data->predict.ps = ps;
    data->scores = tk_malloc(L, chunks * n * sizeof(long int));
  }

  tk_threads_signal(pool, TM_ENCODER_PREDICT, 0);
  tk_threads_signal(pool, TM_ENCODER_PREDICT_REDUCE, 0);

  for (unsigned int i = 0; i < n_threads; i ++) {
    free(threads[i].scores);
  }
  tk_threads_destroy(pool);
  free(threads);

  tk_cvec_t *out = tk_cvec_create(L, n * tm->class_chunks, 0, 0);
  memcpy(out->a, tm->encodings, n * tm->class_chunks);
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
  lua_getfield(L, 2, "problems");
  tk_cvec_t *ps = tk_cvec_peek(L, -1, "problems");
  lua_getfield(L, 2, "solutions");
  tk_ivec_t *ss = tk_ivec_peek(L, -1, "solutions");
  unsigned int max_iter =  tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  unsigned int n_threads = tk_threads_getn(L, 2, "train", "threads");
  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_tsetlin_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_tsetlin_thread_t));
  memset(threads, 0, n_threads * sizeof(tk_tsetlin_thread_t));
  tk_threadpool_t *pool = tk_threads_create(0, n_threads, tk_tsetlin_worker);

  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_tsetlin_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->tm = tm;
    data->pool = pool;
    data->index = i;
    tk_thread_range(i, n_threads, tm->clause_chunks * tm->classes, &data->cfirst, &data->clast);
    tk_thread_range(i, n_threads, n, &data->sfirst, &data->slast);
    data->train.n = n;
    data->train.ps = ps;
    data->train.ss = ss;
    data->shuffle = tk_malloc(L, n * sizeof(unsigned int));
  }

  tk_threads_signal(pool, TM_SEED, 0);
  tk_threads_signal(pool, TM_CLASSIFIER_SETUP, 0);

  for (unsigned int i = 0; i < max_iter; i ++) {
    tk_threads_signal(pool, TM_CLASSIFIER_TRAIN, 0);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      int status = lua_pcall(L, 1, 1, 0);
      if (status != LUA_OK) {
        // Callback threw - clean up before re-throwing
        for (unsigned int j = 0; j < n_threads; j ++) {
          free(threads[j].shuffle);
          free(threads[j].scores);
        }
        tk_threads_destroy(pool);
        free(threads);
        lua_error(L);
      }
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break;
      }
      lua_pop(L, 1);
    }
  }

  for (unsigned int i = 0; i < n_threads; i ++) {
    free(threads[i].shuffle);
    free(threads[i].scores);
  }
  tk_threads_destroy(pool);
  free(threads);

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
  unsigned int max_iter =  tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  unsigned int n_threads = tk_threads_getn(L, 2, "train", "threads");

  // Only reallocate if needed
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

  tk_tsetlin_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_tsetlin_thread_t));
  memset(threads, 0, n_threads * sizeof(tk_tsetlin_thread_t));
  tk_threadpool_t *pool = tk_threads_create(0, n_threads, tk_tsetlin_worker);

  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_tsetlin_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->tm = tm;
    data->pool = pool;
    data->index = i;
    tk_thread_range(i, n_threads, tm->clause_chunks * tm->classes, &data->cfirst, &data->clast);
    tk_thread_range(i, n_threads, n, &data->sfirst, &data->slast);
    unsigned int chunks = tm_state_clause_chunks(pool, i);
    data->predict.n = n;
    data->predict.ps = ps;
    data->scores = tk_malloc(L, chunks * n * sizeof(long int));
    data->train.n = n;
    data->train.ps = ps;
    data->train.ls = ls;
    data->shuffle = tk_malloc(L, n * sizeof(unsigned int));
  }

  // Seed RNG
  tk_threads_signal(pool, TM_SEED, 0);

  // Setup initial state
  tk_threads_signal(pool, TM_ENCODER_SETUP, 0);

  // Training iterations
  for (unsigned int i = 0; i < max_iter; i ++) {
    tk_threads_signal(pool, TM_ENCODER_TRAIN, 0);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      int status = lua_pcall(L, 1, 1, 0);
      if (status != LUA_OK) {
        // Callback threw - clean up before re-throwing
        for (unsigned int j = 0; j < n_threads; j ++) {
          free(threads[j].shuffle);
          free(threads[j].scores);
        }
        tk_threads_destroy(pool);
        free(threads);
        lua_error(L);
      }
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break;
      }
      lua_pop(L, 1);
    }
  }

  for (unsigned int i = 0; i < n_threads; i ++) {
    free(threads[i].shuffle);
    free(threads[i].scores);
  }
  tk_threads_destroy(pool);
  free(threads);

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
  tk_lua_fwrite(L, &tm->input_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->input_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clause_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->state_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->action_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &tm->tail_mask, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, tm->actions, 1, tm->action_chunks, fh);
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

  // Copy actions into the cvec
  memcpy(checkpoint->a, tm->actions, size);

  return 0;
}

static inline int tk_tsetlin_restore (lua_State *L)
{
  lua_settop(L, 2);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  tk_cvec_t *checkpoint = tk_cvec_peek(L, 2, "checkpoint");

  // Validate size matches
  size_t expected_size = tm->action_chunks;
  if (checkpoint->n != expected_size)
    luaL_error(L, "checkpoint size mismatch: expected %zu bytes, got %zu",
               expected_size, checkpoint->n);

  // Restore actions from checkpoint
  memcpy(tm->actions, checkpoint->a, expected_size);

  return 0;
}

static inline int tk_tsetlin_reconfigure (lua_State *L)
{
  lua_settop(L, 2);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);

  if (!tm->reusable)
    luaL_error(L, "reconfigure requires reusable=true at creation time");

  // Parse new params from table
  unsigned int new_clauses = tk_lua_fcheckunsigned(L, 2, "reconfigure", "clauses");
  unsigned int new_tolerance = tk_lua_fcheckunsigned(L, 2, "reconfigure", "clause_tolerance");
  unsigned int new_maximum = tk_lua_fcheckunsigned(L, 2, "reconfigure", "clause_maximum");
  double new_target = tk_lua_foptposdouble(L, 2, "reconfigure", "target", -1.0);
  double new_specificity = tk_lua_fcheckposdouble(L, 2, "reconfigure", "specificity");

  // Normalize clauses to chunk boundary
  new_clauses = TK_CVEC_BITS_BYTES(new_clauses) * TK_CVEC_BITS;
  unsigned int new_clause_chunks = TK_CVEC_BITS_BYTES(new_clauses);

  // Calculate new sizes
  unsigned int new_action_chunks = tm->classes * new_clauses * tm->input_chunks;
  unsigned int new_state_chunks = tm->classes * new_clauses * (tm->state_bits - 1) * tm->input_chunks;

  // Reallocate actions if needed
  if (new_action_chunks > tm->action_chunks) {
    free(tm->actions);
    tm->actions = (char *)tk_malloc_aligned(L, new_action_chunks, TK_CVEC_BITS);
    if (!tm->actions)
      luaL_error(L, "failed to allocate actions in reconfigure");
  }

  // Zero actions (fresh start)
  memset(tm->actions, 0, new_action_chunks);

  // Reallocate state if needed
  if (new_state_chunks > tm->state_chunks || !tm->state) {
    if (tm->state) free(tm->state);
    tm->state = (char *)tk_malloc_aligned(L, new_state_chunks, TK_CVEC_BITS);
    if (!tm->state)
      luaL_error(L, "failed to allocate state in reconfigure");
  }

  // Update metadata
  tm->clauses = new_clauses;
  tm->clause_chunks = new_clause_chunks;
  tm->clause_tolerance = new_tolerance;
  tm->clause_maximum = new_maximum;
  tm->action_chunks = new_action_chunks;
  tm->state_chunks = new_state_chunks;
  tm->specificity = new_specificity;
  tm->target = new_target < 0
    ? sqrt((double) tm->clauses / 2.0) * (double) new_tolerance
    : ceil(new_target >= 1 ? new_target : fmaxf(1.0, (double) tm->clauses * new_target));

  // Update automata pointers
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;

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
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->automata.n_chunks = tm->input_chunks;
  tm->automata.state_bits = tm->state_bits;
  tm->automata.tail_mask = tm->tail_mask;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;
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

// TODO: Merge malloc/assignment logic from load_* and create_* to reduce
// chances for coding errors
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

// Module-level functions
static luaL_Reg tk_tsetlin_fns[] =
{
  { "create", tk_tsetlin_create },
  { "load", tk_tsetlin_load },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tk_tsetlin_fns, 0); // t - registers create, load
  lua_pushinteger(L, TK_CVEC_BITS); // t b
  lua_setfield(L, -2, "align"); // t
  // Metatable methods and __gc are registered by tk_lua_newuserdata
  return 1;
}
