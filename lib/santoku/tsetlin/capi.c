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

#define _GNU_SOURCE

#include <santoku/lua/utils.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

typedef enum {
  TM_CLASSIFIER,
  TM_ENCODER,
} tk_tsetlin_type_t;

typedef enum {
  TM_CLASSIFIER_PRIME,
  TM_CLASSIFIER_SETUP,
  TM_CLASSIFIER_TRAIN,
  TM_CLASSIFIER_PREDICT,
  TM_CLASSIFIER_PREDICT_REDUCE,
  TM_ENCODER_SETUP,
  TM_ENCODER_PRIME,
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

  long int *sums_old; // classes x samples
  long int *sums_local; // classes x samples
  unsigned int *shuffle; // samples
  long int *scores; // classes x samples

} tk_tsetlin_thread_t;

typedef struct tk_tsetlin_s {

  tk_tsetlin_type_t type;
  bool has_state;

  bool trained;
  bool destroyed;

  unsigned int classes;
  unsigned int features;
  unsigned int clauses;
  unsigned int threshold;
  unsigned int state_bits;
  bool boost_true_positive;
  unsigned int input_bits;
  unsigned int input_chunks;
  unsigned int clause_chunks;
  unsigned int class_chunks;
  unsigned int state_chunks;
  unsigned int action_chunks;
  tk_bits_t *state; // class clause bit chunk
  tk_bits_t *actions; // class clause chunk

  double negative;
  double specificity;

  size_t results_len;
  unsigned int *results;

  size_t encodings_len;
  tk_bits_t *encodings;

  size_t sums_len;
  atomic_long *sums; // samples x classes

  tk_threadpool_t *pool;
  bool initialized_threads;

} tk_tsetlin_t;

#define tm_state_old_sum(tm, thread, chunk, sample) \
  (((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->sums_old[ \
    (chunk) * ((tk_tsetlin_thread_t *) (tm)->pool->threads[(thread)].data)->train.n + \
    (sample)])

#define tm_state_sum_local(tm, thread, class, sample) \
  (((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->sums_local[ \
    (class) * ((tk_tsetlin_thread_t *) (tm)->pool->threads[(thread)].data)->predict.n + \
    (sample)])

#define tm_state_shuffle(tm, thread) \
  (((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->shuffle)

#define tm_state_scores(tm, thread, chunk, sample) \
  (((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->scores[ \
    ((chunk) - ((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->cfirst) * ((tk_tsetlin_thread_t *) (tm)->pool->threads[(thread)].data)->predict.n + \
    (sample)])

#define tm_state_clause_chunks(tm, thread) \
  (((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->clast - ((tk_tsetlin_thread_t *) (tm)->pool->threads[thread].data)->cfirst + 1)

#define tm_state_sum(tm, class, sample) \
  (&(tm)->sums[(sample) * (tm)->classes + \
               (class)])

#define tm_state_counts(tm, clause, input_chunk) \
  (&(tm)->state[(clause) * (tm)->input_chunks * ((tm)->state_bits - 1) + \
                (input_chunk) * ((tm)->state_bits - 1)])

#define tm_state_actions(tm, clause) \
  (&(tm)->actions[(clause) * (tm)->input_chunks])

static inline void tk_tsetlin_init_streams (
  tk_bits_t *mask,
  unsigned int n,
  double specificity
) {
  const unsigned int n_chunks = BITS_DIV(n);
  for (unsigned int i = 0; i < n_chunks; i ++)
    mask[i] = (tk_bits_t)0;
  const double p = 1.0 / specificity;
  long int active = fast_norm(n * p, n * p * (1 - p));
  if (active >= n)
    active = n;
  if (active < 0)
    active = 0;
  for (unsigned int i = 0; i < active; i ++) {
    unsigned int r = fast_rand();
    unsigned int f = ((uint64_t)r * n) >> 32;
    unsigned int chunk = BITS_DIV(f);
    unsigned int bit = BITS_MOD(f);
    mask[chunk] |= ((tk_bits_t)1 << bit);
  }
}

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
  carry_next = actions[chunk] & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] |= carry;
  actions[chunk] |= carry;
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
  carry_next = (~actions[chunk]) & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] &= ~carry;
  actions[chunk] &= ~carry;
}

static inline void tk_tsetlin_calculate (
  tk_tsetlin_t *tm,
  tk_bits_t *input,
  bool predict,
  tk_bits_t *out,
  unsigned int cfirst,
  unsigned int clast
) {
  unsigned int input_chunks = tm->input_chunks;
  for (unsigned int clause_chunk = cfirst; clause_chunk <= clast; clause_chunk ++) {
    out[clause_chunk - cfirst] = (tk_bits_t)0;
    for (unsigned int clause_chunk_pos = 0; clause_chunk_pos < BITS; clause_chunk_pos ++) {
      tk_bits_t output = 0;
      tk_bits_t all_exclude = 0;
      unsigned int clause = clause_chunk * BITS + clause_chunk_pos;
      tk_bits_t *actions = tm_state_actions(tm, clause);
      for (unsigned int k = 0; k < input_chunks - 1; k ++) {
        tk_bits_t inp = input[k];
        tk_bits_t act = actions[k];
        output |= ((act & inp) ^ act);
        all_exclude |= act;
      }
      output |=
        (actions[input_chunks - 1] & input[input_chunks - 1]) ^
        (actions[input_chunks - 1]);
      all_exclude |= ((actions[input_chunks - 1]) ^ 0);
      output = !output && !(predict && !all_exclude);
      if (output)
        out[clause_chunk - cfirst] |= ((tk_bits_t)1 << clause_chunk_pos);
    }
  }
}

static inline long int tk_tsetlin_sums (
  tk_tsetlin_t *tm,
  tk_bits_t *out,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int offset
) {
  long int sum = 0;
  for (unsigned int i = cfirst; i <= clast; i ++) {
    sum += (long int) popcount(out[i - offset] & POS_MASK); // 0101
    sum -= (long int) popcount(out[i - offset] & NEG_MASK); // 1010
  }
  return sum;
}

static inline void tm_update (
  tk_tsetlin_t *tm,
  tk_bits_t *input,
  tk_bits_t out,
  unsigned int s,
  unsigned int target_class,
  unsigned int target_vote,
  unsigned int class,
  unsigned int chunk,
  unsigned int thread
) {
  double negative_sampling = tm->negative;
  if (class != target_class && fast_chance(1 - negative_sampling))
    return;
  unsigned int features = tm->features;
  long int threshold = (long int) tm->threshold;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int boost_true_positive = tm->boost_true_positive;
  double specificity = tm->specificity;
  // NOTE: This is much faster given the stack allocation of feedback arrays.
  // Can we somehow achieve the same with heap?
  tk_bits_t feedback_to_la[input_chunks];
  unsigned int target = class == target_class ? target_vote : !target_vote;
  long int tgt = (long int) target;
  atomic_long *class_sump = tm_state_sum(tm, class, s);
  long int old_sum = tm_state_old_sum(tm, thread, chunk, s);
  long int new_sum = tk_tsetlin_sums(tm, &out, chunk, chunk, chunk);
  long int diff = new_sum - old_sum;
  long int class_sum;
  if (diff != 0) {
    tm_state_old_sum(tm, thread, chunk, s) = new_sum;
    class_sum = tm_state_sum_local(tm, thread, class, s) = atomic_fetch_add(class_sump, diff) + diff;
  } else {
    class_sum = tm_state_sum_local(tm, thread, class, s);
  }
  class_sum = (class_sum > threshold) ? threshold : class_sum;
  class_sum = (class_sum < -threshold) ? -threshold : class_sum;
  long int err = (target == 1) ? (threshold - class_sum) : (threshold + class_sum);
  double p = (double) err / (2.0 * threshold);
  for (unsigned int j = 0; j < BITS; j ++) {
    long int jl = (long int) j;
    tk_bits_t *actions = tm_state_actions(tm, chunk * BITS + j);
    bool output = (out & ((tk_bits_t)1 << j)) > 0;
    if (fast_chance(p)) {
      if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == -1) {
        // Type II feedback
        if (output)
          for (unsigned int k = 0; k < input_chunks; k ++) {
            tk_bits_t updated = (~input[k]) & (~actions[k]);
            tm_inc(tm, chunk * BITS + j, k, updated);
          }
      } else if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == 1) {
        // Type I Feedback
        tk_tsetlin_init_streams(feedback_to_la, 2 * features, specificity);
        if (output) {
          if (boost_true_positive)
            for (unsigned int k = 0; k < input_chunks; k ++) {
              tk_bits_t ichunk = input[k];
              tm_inc(tm, chunk * BITS + j, k, ichunk);
            }
          else
            for (unsigned int k = 0; k < input_chunks; k ++) {
              tk_bits_t fb = input[k] & (~feedback_to_la[k]);
              tm_inc(tm, chunk * BITS + j, k, fb);
            }
          for (unsigned int k = 0; k < input_chunks; k ++) {
            tk_bits_t fb = (~input[k]) & feedback_to_la[k];
            tm_dec(tm, chunk * BITS + j, k, fb);
          }
        } else {
          for (unsigned int k = 0; k < input_chunks; k ++)
            tm_dec(tm, chunk * BITS + j, k, feedback_to_la[k]);
        }
      }
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
  tk_bits_t out;
  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {
    for (unsigned int s = 0; s < n; s ++) {
      tk_tsetlin_calculate(tm, ps + s * input_chunks, true, &out, chunk, chunk);
      long int score = tk_tsetlin_sums(tm, &out, chunk, chunk, chunk);
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
  for (unsigned int clause_chunk = cfirst; clause_chunk <= clast; clause_chunk ++) {
    for (unsigned int clause_chunk_pos = 0; clause_chunk_pos < BITS; clause_chunk_pos ++) {
      unsigned int clause = clause_chunk * BITS + clause_chunk_pos;
      for (unsigned int input_chunk = 0; input_chunk < tm->input_chunks; input_chunk ++) {
        unsigned int m = tm->state_bits - 1;
        tk_bits_t *actions = tm_state_actions(tm, clause);
        tk_bits_t *counts = tm_state_counts(tm, clause, input_chunk);
        actions[input_chunk] = 0;
        for (unsigned int b = 0; b < m; b ++)
          counts[b] = ~((tk_bits_t)0);
      }
    }
  }
  unsigned int clause_chunks = tm->clause_chunks;
  for (unsigned int s = 0; s < n; s ++) {
    tk_bits_t *input = ps + s * tm->input_chunks;
    tk_bits_t out;
    for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {
      tk_tsetlin_calculate(tm, input, false, &out, chunk, chunk);
      unsigned int class = chunk / clause_chunks;
      atomic_long *class_sump = tm_state_sum(tm, class, s);
      long int sum = tk_tsetlin_sums(tm, &out, chunk, chunk, chunk);
      atomic_fetch_add(class_sump, sum);
      tm_state_old_sum(tm, thread, chunk, s) = sum;
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

static void tk_classifier_prime_thread (
  tk_tsetlin_t *tm,
  unsigned int n,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[thread].data;
  unsigned int clause_chunks = tm->clause_chunks;
  memset(data->sums_local, 0, tm->classes * n * sizeof(long int));
  for (unsigned int s = 0; s < n; s ++) {
    for (unsigned int class = cfirst / clause_chunks; class <= clast / clause_chunks; class ++) {
      atomic_long *class_sump = tm_state_sum(tm, class, s);
      tm_state_sum_local(tm, thread, class, s) = atomic_load(class_sump);
    }
  }
}

static void tk_encoder_prime_thread (
  tk_tsetlin_t *tm,
  unsigned int n,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_classifier_prime_thread(tm, n, cfirst, clast, thread);
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
  seed_rand(thread);
  tk_tsetlin_init_shuffle(tm_state_shuffle(tm, thread), n);

  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int *shuffle = tm_state_shuffle(tm, thread);
  tk_bits_t out;

  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {

    // TODO: vectorized pre-calculation of class?
    unsigned int class = chunk / clause_chunks;

    for (unsigned int i = 0; i < n; i ++) {
      unsigned int s = shuffle[i];
      tk_bits_t *input = ps + s * input_chunks;
      tk_tsetlin_calculate(tm, input, false, &out, chunk, chunk);
      unsigned int target_class = ss[s];
      tm_update(tm, input, out, s, target_class, 1, class, chunk, thread);
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
  seed_rand(thread);
  tk_tsetlin_init_shuffle(tm_state_shuffle(tm, thread), n);

  unsigned int class_chunks = tm->class_chunks;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int *shuffle = tm_state_shuffle(tm, thread);
  tk_bits_t out;

  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {

    unsigned int class = chunk / clause_chunks;
    unsigned int enc_chunk = BITS_DIV(class);
    unsigned int enc_bit = BITS_MOD(class);

    for (unsigned int i = 0; i < n; i ++) {
      unsigned int s = shuffle[i];
      tk_bits_t *input = ps + s * input_chunks;
      tk_tsetlin_calculate(tm, input, false, &out, chunk, chunk);
      bool target = (ls[s * class_chunks + enc_chunk] & ((tk_bits_t) 1 << enc_bit)) > 0;
      tm_update(tm, input, out, s, class, target, class, chunk, thread);
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
  long int sums[classes];
  for (unsigned int s = sfirst; s <= slast; s ++) {
    for (unsigned int i = 0; i < classes; i ++)
      sums[i] = 0;
    for (unsigned int t = 0; t < tm->pool->n_threads; t ++) {
      tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[t].data;
      for (unsigned int chunk = data->cfirst; chunk <= data->clast; chunk ++) {
        unsigned int class = chunk / clause_chunks;
        sums[class] += tm_state_scores(tm, t, chunk, s);
      }
    }
    tk_bits_t *e = encodings + s * class_chunks;
    for (unsigned int class = 0; class < tm->classes; class ++) {
      unsigned int chunk = BITS_DIV(class);
      unsigned int pos = BITS_MOD(class);
      if (sums[class] > 0)
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
    case TM_CLASSIFIER_SETUP:
      tk_classifier_setup_thread(
        data->tm,
        data->train.n,
        data->train.ps,
        data->cfirst,
        data->clast,
        data->index);
      break;
    case TM_CLASSIFIER_PRIME:
      tk_classifier_prime_thread(
        data->tm,
        data->train.n,
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
    case TM_ENCODER_PRIME:
      tk_encoder_prime_thread(
        data->tm,
        data->train.n,
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
  unsigned int state_bits,
  double thresholdf,
  bool boost_true_positive,
  double negative,
  double specificity,
  unsigned int n_threads
) {
  if (!classes)
    tk_lua_verror(L, 3, "create classifier", "classes", "must be greater than 1");
  if (!clauses)
    tk_lua_verror(L, 3, "create classifier", "clauses", "must be greater than 0");
  if (BITS_MOD(clauses))
    tk_lua_verror(L, 3, "create classifier", "clauses", "must be a multiple of " STR(BITS));
  if (BITS_MOD(features))
    tk_lua_verror(L, 3, "create classifier", "features", "must be a multiple of " STR(BITS));
  if (state_bits < 2)
    tk_lua_verror(L, 3, "create classifier", "bits", "must be greater than 1");
  if (thresholdf <= 0)
    tk_lua_verror(L, 3, "create classifier", "target", "must be greater than 0");
  tm->negative = negative; // Note: unused in encoder
  tm->classes = classes;
  tm->class_chunks = BITS_DIV((tm->classes - 1)) + 1;
  tm->clauses = clauses;
  tm->threshold = ceil(thresholdf >= 1 ? thresholdf : fmaxf(1.0, (double) clauses * thresholdf));
  tm->features = features;
  tm->state_bits = state_bits;
  tm->boost_true_positive = boost_true_positive;
  tm->input_bits = 2 * tm->features;
  tm->input_chunks = BITS_DIV((tm->input_bits - 1)) + 1;
  tm->clause_chunks = BITS_DIV((tm->clauses - 1)) + 1;
  tm->state_chunks = tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tm->action_chunks = tm->classes * tm->clauses * tm->input_chunks;
  tm->state = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->state_chunks, BITS);
  tm->actions = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->action_chunks, BITS);
  tm->specificity = specificity;
  if (!(tm->state && tm->actions))
    luaL_error(L, "error in malloc during creation of classifier");
  tm->pool = tk_threads_create(L, n_threads, tk_tsetlin_worker);
  tk_tsetlin_setup_threads(L, tm, n_threads);
}

static inline int tk_tsetlin_init_encoder (
  lua_State *L,
  tk_tsetlin_t *tm,
  unsigned int encoding_bits,
  unsigned int features,
  unsigned int clauses,
  unsigned int state_bits,
  double thresholdf,
  bool boost_true_positive,
  double negative,
  double specificity,
  unsigned int n_threads
) {
  if (BITS_MOD(encoding_bits))
    tk_lua_verror(L, 3, "create encoder", "hidden", "must be a multiple of " STR(BITS));
  tk_tsetlin_init_classifier(L, tm,
      encoding_bits, features, clauses, state_bits, thresholdf,
      boost_true_positive, negative, specificity, n_threads);
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
      tk_lua_foptunsigned(L, 2, "create classifier", "state", 8),
      tk_lua_fcheckposdouble(L, 2, "create classifier", "target"),
      tk_lua_foptboolean(L, 2, "create classifier", "boost", true),
      tk_lua_fcheckposdouble(L, 2, "create classifier", "negative"),
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
      tk_lua_foptunsigned(L, 2, "create encoder", "state", 8),
      tk_lua_fcheckposdouble(L, 2, "create encoder", "target"),
      tk_lua_foptboolean(L, 2, "create encoder", "boost", true),
      tk_lua_foptposdouble(L, 2, "create encoder", "negative", 1.0), // note used in encoder
      tk_lua_fcheckposdouble(L, 2, "create encoder", "specificity"),
      tk_threads_getn(L, 2, "create encoder", "threads"));

  lua_settop(L, 1);
}

// TODO: Instead of the user passing in a string name for the type, pass in the
// enum value (TM_CLASSIFIER, TM_ENCODER, etc.). Expose these enum values via
// the library table.
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
  if (numa_available() == -1) {
    free(tm->sums); tm->sums = NULL;
  } else {
    numa_free(tm->sums, tm->sums_len); tm->sums = NULL; tm->sums_len = 0;
  }
  if (tm->initialized_threads)
    for (unsigned int i = 0; i < tm->pool->n_threads; i ++) {
      tk_tsetlin_thread_t *data = (tk_tsetlin_thread_t *) tm->pool->threads[i].data;
      free(data->shuffle); data->shuffle = NULL;
      free(data->sums_old); data->sums_old = NULL;
      free(data->sums_local); data->sums_local = NULL;
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
  tk_threads_signal(tm->pool, TM_CLASSIFIER_PREDICT);
  tk_threads_signal(tm->pool, TM_CLASSIFIER_PREDICT_REDUCE);
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

  tk_threads_signal(tm->pool, TM_ENCODER_PREDICT);
  tk_threads_signal(tm->pool, TM_ENCODER_PREDICT_REDUCE);

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

static inline void en_tm_populate (
  tk_tsetlin_t *tm,
  unsigned int s,
  tk_bits_t *encoding,
  unsigned int t
) {
  unsigned int classes = tm->classes;
  for (unsigned int i = 0; i < classes; i ++) {
    unsigned int chunk = BITS_DIV(i);
    unsigned int pos = BITS_MOD(i);
    if (tm_state_sum_local(tm, t, i, s) > 0)
      encoding[chunk] |= ((tk_bits_t)1 << pos);
    else
      encoding[chunk] &= ~((tk_bits_t)1 << pos);
  }
}

static inline int tk_tsetlin_train_classifier (
  lua_State *L,
  tk_tsetlin_t *tm
) {
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  tk_bits_t *ps = (tk_bits_t *) tk_lua_fcheckustring(L, 2, "train", "problems");
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
    data->sums_old = tk_realloc(L, data->sums_old, tm->classes * tm->clause_chunks * n * sizeof(long int));
    data->sums_local = tk_realloc(L, data->sums_local, tm->classes * n * sizeof(long int));
  }

  tm->sums = tk_ensure_interleaved(L, &tm->sums_len, tm->sums, n * tm->classes * sizeof(atomic_long), false);
  for (unsigned int c = 0; c < tm->classes; c ++)
    for (unsigned int s = 0; s < n; s ++)
      atomic_init(tm_state_sum(tm, c, s), 0);

  tk_tsetlin_setup_thread_samples(tm, n);

  tk_threads_signal(tm->pool, TM_CLASSIFIER_SETUP);
  tk_threads_signal(tm->pool, TM_CLASSIFIER_PRIME);

  for (unsigned int i = 0; i < max_iter; i ++) {

    tk_threads_signal(tm->pool, TM_CLASSIFIER_TRAIN);

    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      lua_call(L, 1, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0)
        break;
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
    data->sums_old = tk_realloc(L, data->sums_old, tm->classes * tm->clause_chunks * n * sizeof(long int));
    data->sums_local = tk_realloc(L, data->sums_local, tm->classes * n * sizeof(long int));
  }

  tm->sums = tk_ensure_interleaved(L, &tm->sums_len, tm->sums, n * tm->classes * sizeof(atomic_long), false);
  for (unsigned int c = 0; c < tm->classes; c ++)
    for (unsigned int s = 0; s < n; s ++)
      atomic_init(tm_state_sum(tm, c, s), 0);

  tk_tsetlin_setup_thread_samples(tm, n);

  tk_threads_signal(tm->pool, TM_ENCODER_SETUP);
  tk_threads_signal(tm->pool, TM_ENCODER_PRIME);

  for (unsigned int i = 0; i < max_iter; i ++) {

    tk_threads_signal(tm->pool, TM_ENCODER_TRAIN);

    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      lua_call(L, 1, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0)
        break;
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
  tk_lua_fwrite(L, &tm->threshold, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->state_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->boost_true_positive, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, &tm->input_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->input_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->clause_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->state_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->action_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->specificity, sizeof(double), 1, fh);
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
  tm->class_chunks = BITS_DIV((tm->classes - 1)) + 1;
  tk_lua_fread(L, &tm->features, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clauses, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->threshold, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->state_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->boost_true_positive, sizeof(bool), 1, fh);
  tk_lua_fread(L, &tm->input_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->input_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->clause_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->state_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->action_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &tm->specificity, sizeof(double), 1, fh);
  tm->actions = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->action_chunks, BITS);
  tk_lua_fread(L, tm->actions, sizeof(tk_bits_t), tm->action_chunks, fh);
  tm->pool = tk_threads_create(L, n_threads, tk_tsetlin_worker);
  tk_tsetlin_setup_threads(L, tm, n_threads);
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
// changes for coding errors
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
