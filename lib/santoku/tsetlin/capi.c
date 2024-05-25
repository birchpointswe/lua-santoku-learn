/*

Copyright (C) 2024 Birch Point SWE (Lua integration, train/evaluate, active clause, autoencoder, regressor)
Copyright (C) 2024 Birch Point SWE (recurrence, loss scaling, multi-threading, auto-vectorizer support)
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

#include "lua.h"
#include "lauxlib.h"

#include <limits.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <errno.h>
#include <stdbool.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

#define BITS (sizeof(unsigned int) * CHAR_BIT)

typedef enum {
  TM_CLASSIFIER,
  TM_RECURRENT_CLASSIFIER,
  TM_ENCODER,
  TM_RECURRENT_ENCODER,
  TM_AUTO_ENCODER,
  TM_REGRESSOR,
  TM_RECURRENT_REGRESSOR,
} tsetlin_type_t;

typedef struct {
  unsigned int classes;
  unsigned int features;
  unsigned int clauses;
  unsigned int threshold;
  unsigned int state_bits;
  bool boost_true_positive;
  unsigned int input_bits;
  unsigned int input_chunks;
  unsigned int clause_chunks;
  unsigned int state_chunks;
  unsigned int action_chunks;
  unsigned int filter;
  unsigned int *state; // class clause bit chunk
  unsigned int *actions; // class clause chunk
  unsigned int *active_clause; // class clause chunk
  pthread_mutex_t *locks;
} tsetlin_classifier_t;

typedef struct {
  tsetlin_classifier_t classifier;
} tsetlin_recurrent_classifier_t;

typedef struct {
  unsigned int encoding_bits;
  unsigned int encoding_chunks;
  unsigned int encoding_filter;
  tsetlin_classifier_t encoder;
} tsetlin_encoder_t;

typedef struct {
  unsigned int token_bits;
  unsigned int token_chunks;
  tsetlin_encoder_t encoder;
} tsetlin_recurrent_encoder_t;

typedef struct {
  tsetlin_encoder_t encoder;
  tsetlin_encoder_t decoder;
} tsetlin_auto_encoder_t;

typedef struct {
  tsetlin_classifier_t classifier;
} tsetlin_regressor_t;

typedef struct {
  tsetlin_classifier_t classifier;
} tsetlin_recurrent_regressor_t;

typedef struct {
  tsetlin_type_t type;
  union {
    tsetlin_classifier_t *classifier;
    tsetlin_recurrent_classifier_t *recurrent_classifier;
    tsetlin_encoder_t *encoder;
    tsetlin_recurrent_encoder_t *recurrent_encoder;
    tsetlin_auto_encoder_t *auto_encoder;
    tsetlin_regressor_t *regressor;
    tsetlin_recurrent_regressor_t *recurrent_regressor;
  };
} tsetlin_t;

#define tm_state_lock(tm, class, clause, input_chunk) \
  (&(tm)->locks[(class) * (tm)->clauses * (tm)->input_chunks + \
                (clause) * (tm)->input_chunks + \
                (input_chunk)])

#define tm_state_counts(tm, class, clause, input_chunk) \
  (&(tm)->state[(class) * (tm)->clauses * (tm)->input_chunks * ((tm)->state_bits - 1) + \
                (clause) * (tm)->input_chunks * ((tm)->state_bits - 1) + \
                (input_chunk) * ((tm)->state_bits - 1)])

#define tm_state_actions(tm, class, clause) \
  (&(tm)->actions[(class) * (tm)->clauses * (tm)->input_chunks + \
                  (clause) * (tm)->input_chunks])

static uint64_t const multiplier = 6364136223846793005u;
__thread uint64_t mcg_state = 0xcafef00dd15ea5e5u;

static inline uint32_t fast_rand ()
{
  uint64_t x = mcg_state;
  unsigned int count = (unsigned int) (x >> 61);
  mcg_state = x * multiplier;
  return (uint32_t) ((x ^ x >> 22) >> (22 + count));
}

static inline double fast_drand ()
{
  return ((double)fast_rand()) / ((double)UINT32_MAX);
}

static inline bool fast_chance (double p)
{
  return fast_drand() <= p;
}

static inline int normal (double mean, double variance)
{
  double u1 = (double) (fast_rand() + 1) / ((double) UINT32_MAX + 1);
  double u2 = (double) fast_rand() / UINT32_MAX;
  double n1 = sqrt(-2 * log(u1)) * sin(8 * atan(1) * u2);
  return (int) round(mean + sqrt(variance) * n1);
}

static inline unsigned int popcount (
  unsigned int x
) {
  x = x - ((x >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  x = (x + (x >> 4)) & 0x0F0F0F0F;
  x = x + (x >> 8);
  x = x + (x >> 16);
  return x & 0x0000003F;
}

static inline void init_shuffle (
  unsigned int *shuffle,
  unsigned int n
) {
  // TODO: Can we do this without two loop? Initialize and shuffle at the same
  // time?
  for (unsigned int i = 0; i < n; i ++)
    shuffle[i] = i;
  for (unsigned i = 0; i < n - 1; i ++) {
    unsigned int j = i + fast_rand() / (UINT32_MAX / (n - i) + 1);
    unsigned int t = shuffle[j];
    shuffle[j] = shuffle[i];
    shuffle[i] = t;
  }
}

static inline unsigned int hamming (
  unsigned int *a,
  unsigned int *b,
  unsigned int bits
) {
  unsigned int chunks = (bits - 1) / BITS + 1;
  unsigned int distance = 0;
  for (unsigned int i = 0; i < chunks; i ++) {
    unsigned int diff = a[i] ^ b[i];
    distance += (unsigned int) popcount(diff);
  }
  return distance;
}

static inline double hamming_loss (
  unsigned int *a,
  unsigned int *b,
  unsigned int bits,
  double alpha
) {
  double loss = hamming(a, b, bits);
  return pow(loss / (double) bits, alpha);
}

static inline double triplet_loss_hamming (
  unsigned int *a,
  unsigned int *n,
  unsigned int *p,
  unsigned int bits,
  double margin,
  double alpha
) {
  unsigned int dist_an = hamming(a, n, bits);
  unsigned int dist_ap = hamming(a, p, bits);
  return fminf(1.0f, fmaxf(0.0f, (double) dist_ap - dist_an + margin) * alpha / bits);
}

static inline void flip_bits (
  unsigned int *a,
  unsigned int n
) {
  for (unsigned int i = 0; i < n; i ++)
    a[i] = ~a[i];
}

static inline void tk_lua_callmod (
  lua_State *L,
  int nargs,
  int nret,
  const char *smod,
  const char *sfn
) {
  lua_getglobal(L, "require"); // arg req
  lua_pushstring(L, smod); // arg req smod
  lua_call(L, 1, 1); // arg mod
  lua_pushstring(L, sfn); // args mod sfn
  lua_gettable(L, -2); // args mod fn
  lua_remove(L, -2); // args fn
  lua_insert(L, - nargs - 1); // fn args
  lua_call(L, nargs, nret); // results
}

static inline int tk_error (
  lua_State *L,
  const char *label,
  int err
) {
  lua_pushstring(L, label);
  lua_pushstring(L, strerror(err));
  tk_lua_callmod(L, 2, 0, "santoku.error", "error");
  return 1;
}

static inline void tm_initialize_random_streams (
  tsetlin_classifier_t *tm,
  unsigned int *feedback_to_la,
  double specificity
) {
  unsigned int features = tm->features;
  unsigned int input_chunks = tm->input_chunks;
  memset(feedback_to_la, 0, input_chunks * sizeof(unsigned int));
  long int n = 2 * features;
  double p = 1.0 / specificity;
  long int active = normal(n * p, n * p * (1 - p));
  active = active >= n ? n : active;
  active = active < 0 ? 0 : active;
  for (unsigned int i = 0; i < active; i ++) {
    unsigned int f = fast_rand() % (2 * features);
    while (feedback_to_la[f / BITS] & (1U << (f % BITS)))
      f = fast_rand() % (2 * features);
    feedback_to_la[f / BITS] |= 1U << (f % BITS);
  }
}

static inline void tm_inc (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int clause,
  unsigned int chunk,
  unsigned int active
) {
  pthread_mutex_t *lock = tm_state_lock(tm, class, clause, chunk);
  pthread_mutex_lock(lock);

  unsigned int m = tm->state_bits - 1;
  unsigned int *counts = tm_state_counts(tm, class, clause, chunk);
  unsigned int carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = counts[b] & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  unsigned int *actions = tm_state_actions(tm, class, clause);
  carry_next = actions[chunk] & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] |= carry;
  actions[chunk] |= carry;

  pthread_mutex_unlock(lock);
}

static inline void tm_dec (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int clause,
  unsigned int chunk,
  unsigned int active
) {
  pthread_mutex_t *lock = tm_state_lock(tm, class, clause, chunk);
  pthread_mutex_lock(lock);

  unsigned int m = tm->state_bits - 1;
  unsigned int *counts = tm_state_counts(tm, class, clause, chunk);
  unsigned int carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = (~counts[b]) & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  unsigned int *actions = tm_state_actions(tm, class, clause);
  carry_next = (~actions[chunk]) & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] &= ~carry;
  actions[chunk] &= ~carry;

  pthread_mutex_unlock(lock);
}

static inline long int sum_up_class_votes (
  tsetlin_classifier_t *tm,
  unsigned int *clause_output,
  bool predict
) {
  long int class_sum = 0;
  unsigned int *active = tm->active_clause;
  unsigned int clause_chunks = tm->clause_chunks;
  if (predict) {
    for (unsigned int i = 0; i < clause_chunks; i ++) {
      class_sum += popcount(clause_output[i] & 0x55555555); // 0101
      class_sum -= popcount(clause_output[i] & 0xaaaaaaaa); // 1010
    }
  } else {
    for (unsigned int i = 0; i < clause_chunks; i ++) {
      class_sum += popcount(clause_output[i] & 0x55555555 & active[i]); // 0101
      class_sum -= popcount(clause_output[i] & 0xaaaaaaaa & active[i]); // 1010
    }
  }
  long int threshold = tm->threshold;
  class_sum = (class_sum > threshold) ? threshold : class_sum;
  class_sum = (class_sum < -threshold) ? -threshold : class_sum;
  return class_sum;
}

static inline void tm_calculate_clause_output (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int *input,
  unsigned int *clause_output,
  bool predict
) {
	memset(clause_output, 0, tm->clause_chunks * sizeof(unsigned int));
  unsigned int clauses = tm->clauses;
  unsigned int filter = tm->filter;
  for (unsigned int j = 0; j < clauses; j ++) {
    unsigned int output = 0;
    unsigned int all_exclude = 0;
    unsigned int input_chunks = tm->input_chunks;
    unsigned int clause_chunk = j / BITS;
    unsigned int clause_chunk_pos = j % BITS;
    unsigned int *actions = tm_state_actions(tm, class, j);
    for (unsigned int k = 0; k < input_chunks - 1; k ++) {
      output |= ((actions[k] & input[k]) ^ actions[k]);
      all_exclude |= actions[k];
    }
    output |=
      (actions[input_chunks - 1] & input[input_chunks - 1] & filter) ^
      (actions[input_chunks - 1] & filter);
    all_exclude |= ((actions[input_chunks - 1] & filter) ^ 0);
    output = !output && !(predict && !all_exclude);
    if (output)
      clause_output[clause_chunk] |= (1U << clause_chunk_pos);
  }
}

static inline void tm_update (
  tsetlin_classifier_t *tm,
  unsigned int class,
  unsigned int *input,
  unsigned int target,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la,
  double specificity
) {
  tm_calculate_clause_output(tm, class, input, clause_output, false);
  long int class_sum = sum_up_class_votes(tm, clause_output, false);
  long int tgt = target ? 1 : 0;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int boost_true_positive = tm->boost_true_positive;
  unsigned int clauses = tm->clauses;
  unsigned int threshold = tm->threshold;
  unsigned int *active_clause = tm->active_clause;
  float p = (1.0 / (threshold * 2)) * (threshold + (1 - 2 * tgt) * class_sum);
  memset(feedback_to_clauses, 0, clause_chunks * sizeof(unsigned int));
  for (unsigned int i = 0; i < clauses; i ++) {
    unsigned int clause_chunk = i / BITS;
    unsigned int clause_chunk_pos = i % BITS;
    feedback_to_clauses[clause_chunk] |= ((unsigned int) fast_chance(p) << clause_chunk_pos);
  }
  for (unsigned int i = 0; i < clause_chunks; i ++)
    feedback_to_clauses[i] &= active_clause[i];
  for (unsigned int j = 0; j < clauses; j ++) {
    long int jl = (long int) j;
    unsigned int clause_chunk = j / BITS;
    unsigned int clause_chunk_pos = j % BITS;
    unsigned int *actions = tm_state_actions(tm, class, j);
    if (!(feedback_to_clauses[clause_chunk] & (1U << clause_chunk_pos)))
      continue;
    if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == -1) {
      // Type II feedback
      if ((clause_output[clause_chunk] & (1U << clause_chunk_pos)) > 0)
        for (unsigned int k = 0; k < input_chunks; k ++) {
          unsigned int active = (~input[k]) & (~actions[k]);
          tm_inc(tm, class, j, k, active);
        }
    } else if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == 1) {
      // Type I Feedback
      tm_initialize_random_streams(tm, feedback_to_la, specificity);
      if ((clause_output[clause_chunk] & (1U << clause_chunk_pos)) > 0) {
        if (boost_true_positive)
          for (unsigned int k = 0; k < input_chunks; k ++) {
            unsigned int chunk = input[k];
            tm_inc(tm, class, j, k, chunk);
          }
        else
          for (unsigned int k = 0; k < input_chunks; k ++) {
            unsigned int fb = input[k] & (~feedback_to_la[k]);
            tm_inc(tm, class, j, k, fb);
          }
        for (unsigned int k = 0; k < input_chunks; k ++) {
          unsigned int fb = (~input[k]) & feedback_to_la[k];
          tm_dec(tm, class, j, k, fb);
        }
      } else {
        for (unsigned int k = 0; k < input_chunks; k ++)
          tm_dec(tm, class, j, k, feedback_to_la[k]);
      }
    }
  }
}

static inline unsigned int tm_score_max (
  tsetlin_classifier_t *tm,
  long int *scores
) {
  unsigned int n_classes = tm->classes;
  unsigned int max_class = 0;
  long int max_score = scores[0];
  for (unsigned int class = 1; class < n_classes; class ++) {
    if (scores[class] > max_score) {
      max_class = class;
      max_score = scores[class];
    }
  }
  return max_class;
}

static inline void tm_score (
  tsetlin_classifier_t *tm,
  unsigned int *input,
  unsigned int *clause_output,
  long int *scores
) {
  unsigned int n_classes = tm->classes;
  for (unsigned int class = 0; class < n_classes; class ++) {
    tm_calculate_clause_output(tm, class, input, clause_output, true);
    scores[class] = sum_up_class_votes(tm, clause_output, true);
  }
}

static inline unsigned int mc_tm_predict (
  tsetlin_classifier_t *tm,
  unsigned int *input
) {
  unsigned int clause_output[tm->clause_chunks];
  long int scores[tm->classes];
  tm_score(tm, input, clause_output, scores);
  return tm_score_max(tm, scores);
}

static inline void mc_tm_update (
  tsetlin_classifier_t *tm,
  unsigned int *input,
  unsigned int class,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la,
  double specificity
) {
  tm_update(tm, class, input, 1, clause_output, feedback_to_clauses, feedback_to_la, specificity);
  // TODO: Is there a faster way to do negative class selection? Is negative
  // class selection even worth it?
  unsigned int negative_class = (unsigned int) fast_rand() % tm->classes;
  while (negative_class == class)
    negative_class = (unsigned int) fast_rand() % tm->classes;
  tm_update(tm, negative_class, input, 0, clause_output, feedback_to_clauses, feedback_to_la, specificity);
}

static inline void ae_tm_decode (
  tsetlin_auto_encoder_t *tm,
  unsigned int *input,
  unsigned int *decoding,
  long int *scores
) {
  tsetlin_classifier_t *decoder = &tm->decoder.encoder;
  unsigned int decoder_classes = decoder->classes;
  unsigned int clause_output[decoder->clause_chunks];
  tm_score(decoder, input, clause_output, scores);

  for (unsigned int i = 0; i < decoder_classes; i ++)
  {
    unsigned int chunk = i / BITS;
    unsigned int pos = i % BITS;
    if (scores[i] > 0)
      decoding[chunk] |= (1U << pos);
    else
      decoding[chunk] &= ~(1U << pos);
  }
  decoding[tm->decoder.encoding_chunks - 1] &= decoder->filter;
}

static inline void ae_tm_encode (
  tsetlin_auto_encoder_t *tm,
  unsigned int *input,
  unsigned int *encoding,
  long int *scores
) {
  tsetlin_classifier_t *encoder = &tm->encoder.encoder;
  unsigned int encoder_classes = encoder->classes;
  unsigned int clause_output[encoder->clause_chunks];
  tm_score(encoder, input, clause_output, scores);

  for (unsigned int i = 0; i < encoder_classes; i ++)
  {
    unsigned int chunk = i / BITS;
    unsigned int pos = i % BITS;
    if (scores[i] > 0)
      encoding[chunk] |= (1U << pos);
    else
      encoding[chunk] &= ~(1U << pos);
  }
  encoding[tm->encoder.encoding_chunks - 1] &= encoder->filter;
}

static inline void en_tm_encode (
  tsetlin_encoder_t *tm,
  unsigned int *input,
  unsigned int *encoding,
  long int *scores
) {
  unsigned int encoder_classes = tm->encoder.classes;
  unsigned int clause_output[tm->encoder.clause_chunks];
  tm_score(&tm->encoder, input, clause_output, scores);
  for (unsigned int i = 0; i < encoder_classes; i ++)
  {
    unsigned int chunk = i / BITS;
    unsigned int pos = i % BITS;
    if (scores[i] > 0)
      encoding[chunk] |= (1U << pos);
    else
      encoding[chunk] &= ~(1U << pos);
  }
  encoding[tm->encoding_chunks - 1] &= tm->encoder.filter;
}

static inline void re_tm_encode (
  tsetlin_recurrent_encoder_t *tm,
  unsigned int x_first,
  unsigned int x_len,
  unsigned int *x_data,
  unsigned int **states,
  unsigned int *states_size,
  unsigned int *states_max,
  unsigned int *input,
  long int *scores
) {
  if (x_first > x_len)
    return;

  tsetlin_encoder_t *encoder = &tm->encoder;

  unsigned int token_chunks = tm->token_chunks;
  unsigned int encoding_chunks = encoder->encoding_chunks;

  *states_size = x_len + 1;

  if (*states_size > *states_max) {
    *states_max = *states_size;
    *states = realloc(*states, encoding_chunks * (*states_size) * sizeof(unsigned int));
  }

  if (x_first == 1)
    memset((*states), 0, encoding_chunks * sizeof(unsigned int));

  unsigned int off_token0 = 0;
  unsigned int off_state0 = off_token0 + token_chunks;
  unsigned int off_flipped = off_state0 + encoding_chunks;

  for (unsigned int i = x_first; i <= x_len; i ++) {
    unsigned int *x = x_data + token_chunks * (i - 1);
    memcpy(input + off_token0, x, token_chunks * sizeof(unsigned int));
    memcpy(input + off_state0, (*states) + (encoding_chunks * (i - 1)), encoding_chunks * sizeof(unsigned int));
    memcpy(input + off_token0, input + off_flipped, token_chunks * sizeof(unsigned int) + encoding_chunks * sizeof(unsigned int));
    flip_bits(input + off_flipped, token_chunks + encoding_chunks);
    en_tm_encode(encoder, input, (*states) + (encoding_chunks * i), scores);
  }
}

static inline void ae_tm_update (
  tsetlin_auto_encoder_t *tm,
  unsigned int *input,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la_e,
  unsigned int *feedback_to_la_d,
  long int *scores_e,
  long int *scores_d,
  double specificity,
  double loss_alpha
) {
  unsigned int encoding[tm->encoder.encoding_chunks * 2]; // encoding + flipped bits for input to decoder
  unsigned int decoding[tm->decoder.encoding_chunks * 2]; // decoding + flipped bits to compare to input

  // encode input, copy, flip bits
  ae_tm_encode(tm, input, encoding, scores_e);
  memcpy(encoding + tm->encoder.encoding_chunks, encoding, tm->encoder.encoding_chunks * sizeof(unsigned int));
  flip_bits(encoding + tm->encoder.encoding_chunks, tm->encoder.encoding_chunks);

  // decode encoding, copy, flip bits
  ae_tm_decode(tm, encoding, decoding, scores_d);
  memcpy(decoding + tm->decoder.encoding_chunks, decoding, tm->decoder.encoding_chunks * sizeof(unsigned int));
  flip_bits(decoding + tm->decoder.encoding_chunks, tm->decoder.encoding_chunks);

  // compare input to decoding
  double loss = hamming_loss(input, decoding, tm->encoder.encoder.input_bits, loss_alpha);

  for (unsigned int bit = 0; bit < tm->encoder.encoding_bits; bit ++)
    if (fast_chance(loss)) {
      unsigned int chunk0 = bit / BITS;
      unsigned int chunk1 = chunk0 + tm->encoder.encoding_chunks;
      unsigned int pos = bit % BITS;
      unsigned int bit_x = encoding[chunk0] & (1U << pos);
      unsigned int bit_x_flipped = !bit_x;
      encoding[chunk0] ^= (1U << pos);
      encoding[chunk1] ^= (1U << pos);
      ae_tm_decode(tm, encoding, decoding, scores_d);
      memcpy(decoding + tm->decoder.encoding_chunks, decoding, tm->decoder.encoding_chunks * sizeof(unsigned int));
      flip_bits(decoding + tm->decoder.encoding_chunks, tm->decoder.encoding_chunks);
      double loss0 = hamming_loss(input, decoding, tm->encoder.encoder.input_bits, loss_alpha);
      encoding[chunk0] ^= (1U << pos);
      encoding[chunk1] ^= (1U << pos);
      if (loss0 < loss)
        tm_update(&tm->encoder.encoder, bit, input, bit_x_flipped, clause_output, feedback_to_clauses, feedback_to_la_e, specificity);
      else if (loss0 > loss)
        tm_update(&tm->encoder.encoder, bit, input, bit_x, clause_output, feedback_to_clauses, feedback_to_la_e, specificity);
    }

  for (unsigned int bit = 0; bit < tm->encoder.encoder.input_bits; bit ++)
    if (fast_chance(loss)) {
      unsigned int chunk = bit / BITS;
      unsigned int pos = bit % BITS;
      unsigned int bit_i = input[chunk] & (1U << pos);
      tm_update(&tm->decoder.encoder, bit, encoding, bit_i, clause_output, feedback_to_clauses, feedback_to_la_d, specificity);
    }
}

static inline void en_tm_update (
  tsetlin_encoder_t *tm,
  unsigned int *a,
  unsigned int *n,
  unsigned int *p,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la,
  long int *scores,
  double specificity,
  double margin,
  double loss_alpha
) {

  tsetlin_classifier_t *encoder = &tm->encoder;
  unsigned int classes = encoder->classes;
  unsigned int encoding_chunks = tm->encoding_chunks;
  unsigned int encoding_bits = tm->encoding_bits;

  unsigned int encoding_a[encoding_chunks];
  unsigned int encoding_n[encoding_chunks];
  unsigned int encoding_p[encoding_chunks];

  en_tm_encode(tm, a, encoding_a, scores);
  en_tm_encode(tm, n, encoding_n, scores);
  en_tm_encode(tm, p, encoding_p, scores);

  double loss = triplet_loss_hamming(encoding_a, encoding_n, encoding_p, encoding_bits, margin, loss_alpha);

  for (unsigned int i = 0; i < classes; i ++) {
    if (fast_chance(loss)) {
      unsigned int chunk = i / BITS;
      unsigned int pos = i % BITS;
      unsigned int bit_a = encoding_a[chunk] & (1U << pos);
      unsigned int bit_n = encoding_n[chunk] & (1U << pos);
      unsigned int bit_p = encoding_p[chunk] & (1U << pos);
      if ((bit_a && bit_n && bit_p) || (!bit_a && !bit_n && !bit_p)) {
        // flip n, keep a and p
        tm_update(encoder, i, a, bit_a, clause_output, feedback_to_clauses, feedback_to_la, specificity);
        tm_update(encoder, i, n, !bit_n, clause_output, feedback_to_clauses, feedback_to_la, specificity);
        tm_update(encoder, i, p, bit_p, clause_output, feedback_to_clauses, feedback_to_la, specificity);
      } else if ((bit_a && bit_n && !bit_p) || (!bit_a && !bit_n && bit_p)) {
        // flip a, keep n and p
        tm_update(encoder, i, a, !bit_a, clause_output, feedback_to_clauses, feedback_to_la, specificity);
        tm_update(encoder, i, n, bit_n, clause_output, feedback_to_clauses, feedback_to_la, specificity);
        tm_update(encoder, i, p, bit_p, clause_output, feedback_to_clauses, feedback_to_la, specificity);
      } else if ((bit_a && !bit_n && bit_p) || (!bit_a && bit_n && !bit_p)) {
        // keep all
        tm_update(encoder, i, a, bit_a, clause_output, feedback_to_clauses, feedback_to_la, specificity);
        tm_update(encoder, i, n, bit_n, clause_output, feedback_to_clauses, feedback_to_la, specificity);
        tm_update(encoder, i, p, bit_p, clause_output, feedback_to_clauses, feedback_to_la, specificity);
      } else if ((bit_a && !bit_n && !bit_p) || (!bit_a && bit_n && bit_p)) {
        // flip p, keep a and n
        tm_update(encoder, i, a, bit_a, clause_output, feedback_to_clauses, feedback_to_la, specificity);
        tm_update(encoder, i, n, bit_n, clause_output, feedback_to_clauses, feedback_to_la, specificity);
        tm_update(encoder, i, p, !bit_p, clause_output, feedback_to_clauses, feedback_to_la, specificity);
      }
    }
  }

}

static inline void re_tm_update_recompute (
  tsetlin_recurrent_encoder_t *tm,
  tsetlin_classifier_t *encoder,
  unsigned int x_len,
  unsigned int *x_data,
  unsigned int **state_x,
  unsigned int *state_x_size,
  unsigned int *state_x_max,
  unsigned int *input_x,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la,
  long int *scores,
  unsigned int encoding_bits,
  unsigned int encoding_chunks,
  double specificity,
  double margin,
  double loss,
  double loss_alpha,
  unsigned int *encoding_a,
  unsigned int *encoding_n,
  unsigned int *encoding_p
) {
  for (unsigned int word = 1; word <= x_len; word ++) {
    for (unsigned int bit = 0; bit < encoding_bits; bit ++) {
      if (fast_chance(loss)) {
        unsigned int chunk = bit / BITS;
        unsigned int pos = bit % BITS;
        unsigned chunk0 = word * encoding_chunks + chunk;
        unsigned int bit_x = (*state_x)[chunk0] & (1U << pos);
        unsigned int bit_x_flipped = !bit_x;
        (*state_x)[chunk0] ^= (1U << pos);
        re_tm_encode(tm, word + 1, x_len, x_data, state_x, state_x_size, state_x_max, input_x, scores);
        (*state_x)[chunk0] ^= (1U << pos);
        unsigned int *encoding_x = (*state_x) + (*state_x_size - 1) * encoding_chunks;
        double loss0 = triplet_loss_hamming(
          encoding_a ? encoding_a : encoding_x,
          encoding_n ? encoding_n : encoding_x,
          encoding_p ? encoding_p : encoding_x,
          encoding_bits, margin, loss_alpha);
        if (loss0 < loss)
          tm_update(encoder, bit, input_x, bit_x_flipped,
              clause_output, feedback_to_clauses, feedback_to_la, specificity);
        else if (loss0 >= loss)
          tm_update(encoder, bit, input_x, bit_x,
              clause_output, feedback_to_clauses, feedback_to_la, specificity);
      }
    }
  }
}

static inline void re_tm_update (
  tsetlin_recurrent_encoder_t *tm,
  unsigned int a_len,
  unsigned int *a_data,
  unsigned int n_len,
  unsigned int *n_data,
  unsigned int p_len,
  unsigned int *p_data,
  unsigned int *input_a,
  unsigned int *input_n,
  unsigned int *input_p,
  unsigned int **state_a,
  unsigned int *state_a_size,
  unsigned int *state_a_max,
  unsigned int **state_n,
  unsigned int *state_n_size,
  unsigned int *state_n_max,
  unsigned int **state_p,
  unsigned int *state_p_size,
  unsigned int *state_p_max,
  unsigned int *clause_output,
  unsigned int *feedback_to_clauses,
  unsigned int *feedback_to_la,
  long int *scores,
  double specificity,
  double margin,
  double loss_alpha
) {
  tsetlin_classifier_t *encoder = &tm->encoder.encoder;
  unsigned int encoding_bits = tm->encoder.encoding_bits;
  unsigned int encoding_chunks = tm->encoder.encoding_chunks;

  re_tm_encode(tm, 1, a_len, a_data, state_a, state_a_size, state_a_max, input_a, scores);
  re_tm_encode(tm, 1, n_len, n_data, state_n, state_n_size, state_n_max, input_n, scores);
  re_tm_encode(tm, 1, p_len, p_data, state_p, state_p_size, state_p_max, input_p, scores);

  unsigned int *encoding_a = *state_a + ((*state_a_size - 1) * encoding_chunks);
  unsigned int *encoding_n = *state_n + ((*state_n_size - 1) * encoding_chunks);
  unsigned int *encoding_p = *state_p + ((*state_p_size - 1) * encoding_chunks);

  double loss = triplet_loss_hamming(encoding_a, encoding_n, encoding_p, encoding_bits, margin, loss_alpha);

  unsigned int r = fast_rand() % 3;

  if (r == 1)
    re_tm_update_recompute(tm, encoder, a_len, a_data, state_a, state_a_size, state_a_max, input_a, clause_output, feedback_to_clauses, feedback_to_la, scores, encoding_bits, encoding_chunks, specificity, margin, loss, loss_alpha, NULL, encoding_n, encoding_p);

  else if (r == 2)
    re_tm_update_recompute(tm, encoder, n_len, n_data, state_n, state_n_size, state_n_max, input_n, clause_output, feedback_to_clauses, feedback_to_la, scores, encoding_bits, encoding_chunks, specificity, margin, loss, loss_alpha, encoding_a, NULL, encoding_p);

  else if (r == 3)
    re_tm_update_recompute(tm, encoder, p_len, p_data, state_p, state_p_size, state_p_max, input_p, clause_output, feedback_to_clauses, feedback_to_la, scores, encoding_bits, encoding_chunks, specificity, margin, loss, loss_alpha, encoding_a, encoding_n, NULL);
}

static inline void mc_tm_initialize_active_clause (
  tsetlin_classifier_t *tm,
  double active_clause
) {
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int *active_clauses = tm->active_clause;
  memset(active_clauses, 0xFF, clause_chunks * sizeof(unsigned int));
  for (unsigned int i = 0; i < clause_chunks; i ++)
    for (unsigned int j = 0; j < BITS; j ++)
      if (!fast_chance(active_clause))
        active_clauses[i] &= ~(1U << j);
}

tsetlin_t *tk_tsetlin_peek (lua_State *L, int i)
{
  return (tsetlin_t *) luaL_checkudata(L, i, TK_TSETLIN_MT);
}

static inline unsigned int tk_tsetlin_checkunsigned (lua_State *L, int i)
{
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  if (l > UINT_MAX)
    luaL_error(L, "value is too large");
  return (unsigned int) l;
}

static inline double tk_tsetlin_checkposdouble (lua_State *L, int i)
{
  lua_Number l = luaL_checknumber(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  return (double) l;
}

static inline bool tk_tsetlin_checkboolean (lua_State *L, int i)
{
  if (lua_type(L, i) == LUA_TNIL)
    return false;
  luaL_checktype(L, i, LUA_TBOOLEAN);
  return lua_toboolean(L, i);
}

static inline void tk_tsetlin_register (lua_State *L, luaL_Reg *regs, int nup)
{
  while (true) {
    if ((*regs).name == NULL)
      break;
    for (int i = 0; i < nup; i ++)
      lua_pushvalue(L, -nup); // t upsa upsb
    lua_pushcclosure(L, (*regs).func, nup); // t upsa fn
    lua_setfield(L, -nup - 2, (*regs).name); // t
    regs ++;
  }
  lua_pop(L, nup);
}

static inline void tk_tsetlin_init_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm,
  unsigned int classes,
  unsigned int features,
  unsigned int clauses,
  unsigned int state_bits,
  unsigned int threshold,
  bool boost_true_positive
) {
  tm->classes = classes;
  tm->features = features;
  tm->clauses = clauses;
  tm->state_bits = state_bits;
  tm->threshold = threshold;
  tm->boost_true_positive = boost_true_positive;
  tm->input_bits = 2 * tm->features;
  tm->input_chunks = (tm->input_bits - 1) / BITS + 1;
  tm->clause_chunks = (tm->clauses - 1) / BITS + 1;
  tm->state_chunks = tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tm->action_chunks = tm->classes * tm->clauses * tm->input_chunks;
  tm->filter = tm->input_bits % BITS != 0
    ? ~(((unsigned int) ~0) << (tm->input_bits % BITS))
    : (unsigned int) ~0;
  tm->state = malloc(sizeof(unsigned int) * tm->state_chunks);
  tm->actions = malloc(sizeof(unsigned int) * tm->action_chunks);
  tm->active_clause = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->locks = malloc(sizeof(pthread_mutex_t) * tm->action_chunks);
  for (unsigned int i = 0; i < tm->action_chunks; i ++)
    pthread_mutex_init(&tm->locks[i], NULL);
  if (!(tm->active_clause && tm->state && tm->actions))
    luaL_error(L, "error in malloc during creation of classifier");
  for (unsigned int i = 0; i < tm->classes; i ++)
    for (unsigned int j = 0; j < tm->clauses; j ++)
      for (unsigned int k = 0; k < tm->input_chunks; k ++) {
        unsigned int m = tm->state_bits - 1;
        unsigned int *actions = tm_state_actions(tm, i, j);
        unsigned int *counts = tm_state_counts(tm, i, j, k);
        actions[k] = 0U;
        for (unsigned int b = 0; b < m; b ++)
          counts[b] = ~0U;
      }
}

static inline void tk_tsetlin_create_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  tk_tsetlin_init_classifier(L, tm,
      tk_tsetlin_checkunsigned(L, 1),
      tk_tsetlin_checkunsigned(L, 2),
      tk_tsetlin_checkunsigned(L, 3),
      tk_tsetlin_checkunsigned(L, 4),
      tk_tsetlin_checkunsigned(L, 5),
      tk_tsetlin_checkboolean(L, 6));
}

static inline int tk_tsetlin_create_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *tm)
{
  luaL_error(L, "unimplemented: create recurrent classifier");
  return 0;
}

static inline int tk_tsetlin_init_encoder (
  lua_State *L,
  tsetlin_encoder_t *tm,
  unsigned int encoding_bits,
  unsigned int features,
  unsigned int clauses,
  unsigned int state_bits,
  unsigned int threshold,
  bool boost_true_positive
) {
  tk_tsetlin_init_classifier(L, &tm->encoder,
      encoding_bits, features, clauses, state_bits, threshold, boost_true_positive);
  tm->encoding_bits = encoding_bits;
  tm->encoding_chunks = (encoding_bits - 1) / BITS + 1;
  tm->encoding_filter = encoding_bits % BITS != 0
    ? ~(((unsigned int) ~0) << (encoding_bits % BITS))
    : (unsigned int) ~0;
  return 0;
}

static inline int tk_tsetlin_create_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  return tk_tsetlin_init_encoder(L, tm,
      tk_tsetlin_checkunsigned(L, 1),
      tk_tsetlin_checkunsigned(L, 2),
      tk_tsetlin_checkunsigned(L, 3),
      tk_tsetlin_checkunsigned(L, 4),
      tk_tsetlin_checkunsigned(L, 5),
      tk_tsetlin_checkboolean(L, 6));
}

static inline int tk_tsetlin_create_recurrent_encoder (lua_State *L, tsetlin_recurrent_encoder_t *tm)
{
  unsigned int output_bits = tk_tsetlin_checkunsigned(L, 1);
  unsigned int token_bits = tk_tsetlin_checkunsigned(L, 2);
  unsigned int clauses = tk_tsetlin_checkunsigned(L, 3);
  unsigned int state_bits = tk_tsetlin_checkunsigned(L, 4);
  unsigned int threshold = tk_tsetlin_checkunsigned(L, 5);
  unsigned int boost_true_positive = tk_tsetlin_checkboolean(L, 6);
  tk_tsetlin_init_encoder(L, &tm->encoder,
    output_bits,
    (token_bits + output_bits),
    clauses,
    state_bits,
    threshold,
    boost_true_positive);
  tm->token_bits = token_bits;
  tm->token_chunks = (tm->token_bits - 1) / BITS + 1;
  return 0;
}

static inline int tk_tsetlin_create_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  unsigned int encoding_bits = tk_tsetlin_checkunsigned(L, 1);
  unsigned int features = tk_tsetlin_checkunsigned(L, 2);
  unsigned int clauses = tk_tsetlin_checkunsigned(L, 3);
  unsigned int state_bits = tk_tsetlin_checkunsigned(L, 4);
  unsigned int threshold = tk_tsetlin_checkunsigned(L, 5);
  bool boost_true_positive = tk_tsetlin_checkboolean(L, 6);
  tk_tsetlin_init_encoder(L, &tm->encoder,
      encoding_bits, features, clauses, state_bits, threshold, boost_true_positive);
  tk_tsetlin_init_encoder(L, &tm->decoder,
      tm->encoder.encoder.input_bits, encoding_bits, clauses, state_bits, threshold, boost_true_positive);
  return 0;
}

static inline int tk_tsetlin_create_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  luaL_error(L, "unimplemented: create regressor");
  return 0;
}

static inline int tk_tsetlin_create_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *tm)
{
  luaL_error(L, "unimplemented: create recurrent regressor");
  return 0;
}

static inline tsetlin_t *tk_tsetlin_alloc (lua_State *L)
{
  tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t));
  if (!tm) luaL_error(L, "error in malloc during creation");
  memset(tm, 0, sizeof(tsetlin_t));
  luaL_getmetatable(L, TK_TSETLIN_MT);
  lua_setmetatable(L, -2);
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_classifier (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_CLASSIFIER;
  tm->classifier = malloc(sizeof(tsetlin_classifier_t));
  if (!tm->classifier) luaL_error(L, "error in malloc during creation");
  memset(tm->classifier, 0, sizeof(tsetlin_classifier_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_recurrent_classifier (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_RECURRENT_CLASSIFIER;
  tm->recurrent_classifier = malloc(sizeof(tsetlin_recurrent_classifier_t));
  if (!tm->recurrent_classifier) luaL_error(L, "error in malloc during creation");
  memset(tm->recurrent_classifier, 0, sizeof(tsetlin_recurrent_classifier_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_encoder (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_ENCODER;
  tm->encoder = malloc(sizeof(tsetlin_encoder_t));
  if (!tm->encoder) luaL_error(L, "error in malloc during creation");
  memset(tm->encoder, 0, sizeof(tsetlin_encoder_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_recurrent_encoder (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_RECURRENT_ENCODER;
  tm->recurrent_encoder = malloc(sizeof(tsetlin_recurrent_encoder_t));
  if (!tm->recurrent_encoder) luaL_error(L, "error in malloc during creation");
  memset(tm->recurrent_encoder, 0, sizeof(tsetlin_recurrent_encoder_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_auto_encoder (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_AUTO_ENCODER;
  tm->auto_encoder = malloc(sizeof(tsetlin_auto_encoder_t));
  if (!tm->auto_encoder) luaL_error(L, "error in malloc during creation");
  memset(tm->auto_encoder, 0, sizeof(tsetlin_auto_encoder_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_regressor (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_REGRESSOR;
  tm->regressor = malloc(sizeof(tsetlin_regressor_t));
  if (!tm->regressor) luaL_error(L, "error in malloc during creation");
  memset(tm->regressor, 0, sizeof(tsetlin_regressor_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_recurrent_regressor (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_RECURRENT_REGRESSOR;
  tm->recurrent_regressor = malloc(sizeof(tsetlin_recurrent_regressor_t));
  if (!tm->recurrent_regressor) luaL_error(L, "error in malloc during creation");
  memset(tm->recurrent_regressor, 0, sizeof(tsetlin_recurrent_regressor_t));
  return tm;
}

// TODO: Instead of the user passing in a string name for the type, pass in the
// enum value (TM_CLASSIFIER, TM_ENCODER, etc.). Expose these enum values via
// the library table.
static inline int tk_tsetlin_create (lua_State *L)
{
  const char *type = luaL_checkstring(L, 1);
  if (!strcmp(type, "classifier")) {

    lua_remove(L, 1);
    tsetlin_t *tm = tk_tsetlin_alloc_classifier(L);
    tk_tsetlin_create_classifier(L, tm->classifier);
    return 1;

  } else if (!strcmp(type, "recurrent_classifier")) {

    lua_remove(L, 1);
    tsetlin_t *tm = tk_tsetlin_alloc_recurrent_classifier(L);
    tk_tsetlin_create_recurrent_classifier(L, tm->recurrent_classifier);
    return 1;

  } else if (!strcmp(type, "encoder")) {

    lua_remove(L, 1);
    tsetlin_t *tm = tk_tsetlin_alloc_encoder(L);
    tk_tsetlin_create_encoder(L, tm->encoder);
    return 1;

  } else if (!strcmp(type, "recurrent_encoder")) {

    lua_remove(L, 1);
    tsetlin_t *tm = tk_tsetlin_alloc_recurrent_encoder(L);
    tk_tsetlin_create_recurrent_encoder(L, tm->recurrent_encoder);
    return 1;

  } else if (!strcmp(type, "auto_encoder")) {

    lua_remove(L, 1);
    tsetlin_t *tm = tk_tsetlin_alloc_auto_encoder(L);
    tk_tsetlin_create_auto_encoder(L, tm->auto_encoder);
    return 1;

  } else if (!strcmp(type, "regressor")) {

    lua_remove(L, 1);
    tsetlin_t *tm = tk_tsetlin_alloc_regressor(L);
    tk_tsetlin_create_regressor(L, tm->regressor);
    return 1;

  } else if (!strcmp(type, "recurrent_regressor")) {

    lua_remove(L, 1);
    tsetlin_t *tm = tk_tsetlin_alloc_recurrent_regressor(L);
    tk_tsetlin_create_recurrent_regressor(L, tm->recurrent_regressor);
    return 1;

  } else {

    luaL_error(L, "unexpected tsetlin machine type in create");
    return 0;

  }
}

static inline void tk_tsetlin_destroy_classifier (tsetlin_classifier_t *tm)
{
  if (tm == NULL)
    return;
  free(tm->state);
  free(tm->actions);
  free(tm->active_clause);
  for (unsigned int i = 0; i < tm->action_chunks; i ++)
    pthread_mutex_destroy(&tm->locks[i]);
  free(tm->locks);
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      tk_tsetlin_destroy_classifier(tm->classifier);
      free(tm->classifier);
      break;
    case TM_RECURRENT_CLASSIFIER:
      luaL_error(L, "unimplemented: destroy recurrent classifier");
      break;
    case TM_ENCODER:
      tk_tsetlin_destroy_classifier(&tm->encoder->encoder);
      free(tm->encoder);
      break;
    case TM_RECURRENT_ENCODER:
      tk_tsetlin_destroy_classifier(&tm->recurrent_encoder->encoder.encoder);
      free(tm->recurrent_encoder);
      break;
    case TM_AUTO_ENCODER:
      tk_tsetlin_destroy_classifier(&tm->auto_encoder->encoder.encoder);
      tk_tsetlin_destroy_classifier(&tm->auto_encoder->decoder.encoder);
      free(tm->auto_encoder);
      break;
    case TM_REGRESSOR:
      luaL_error(L, "unimplemented: destroy regressor");
      break;
    case TM_RECURRENT_REGRESSOR:
      luaL_error(L, "unimplemented: destroy recurrent regressor");
      break;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in destroy");
  }
  return 0;
}

static inline int tk_tsetlin_predict_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  lua_pushinteger(L, mc_tm_predict(tm, bm));
  return 1;
}

static inline int tk_tsetlin_predict_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *tm)
{
  luaL_error(L, "unimplemented: predict recurrent classifier");
  return 0;
}

static inline int tk_tsetlin_predict_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  unsigned int encoding_a[tm->encoding_chunks];
  long int scores[tm->encoder.classes];
  en_tm_encode(tm, bm, encoding_a, scores);
  lua_pushlstring(L, (char *) encoding_a, sizeof(unsigned int) * tm->encoding_chunks);
  return 1;
}

static inline int tk_tsetlin_predict_recurrent_encoder (lua_State *L, tsetlin_recurrent_encoder_t *tm)
{
  lua_settop(L, 3);
  unsigned int a_len = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *a_data = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int encoding_chunks = tm->encoder.encoding_chunks;
  unsigned int input_chunks = tm->encoder.encoder.input_chunks;
  unsigned int *state_a = NULL;
  unsigned int state_a_size = 0;
  unsigned int state_a_max = 0;
  unsigned int input_a[input_chunks];
  long int scores[tm->encoder.encoder.classes];
  re_tm_encode(tm, 1, a_len, a_data, &state_a, &state_a_size, &state_a_max, input_a, scores);
  unsigned int *encoding_a = state_a + ((state_a_size - 1) * encoding_chunks);
  lua_pushlstring(L, (char *) encoding_a, sizeof(unsigned int) * encoding_chunks);
  free(state_a);
  return 1;
}

static inline int tk_tsetlin_predict_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  unsigned int encoding_chunks = tm->encoder.encoding_chunks;
  unsigned int encoding[encoding_chunks];
  long int scores[tm->encoder.encoder.classes];
  ae_tm_encode(tm, bm, encoding, scores);
  lua_pushlstring(L, (char *) encoding, sizeof(unsigned int) * encoding_chunks);
  return 1;
}

static inline int tk_tsetlin_predict_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  luaL_error(L, "unimplemented: predict regressor");
  return 0;
}

static inline int tk_tsetlin_predict_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *tm)
{
  luaL_error(L, "unimplemented: predict recurrent regressor");
  return 0;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_predict_classifier(L, tm->classifier);
    case TM_RECURRENT_CLASSIFIER:
      return tk_tsetlin_predict_recurrent_classifier(L, tm->recurrent_classifier);
    case TM_ENCODER:
      return tk_tsetlin_predict_encoder(L, tm->encoder);
    case TM_RECURRENT_ENCODER:
      return tk_tsetlin_predict_recurrent_encoder(L, tm->recurrent_encoder);
    case TM_AUTO_ENCODER:
      return tk_tsetlin_predict_auto_encoder(L, tm->auto_encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_predict_regressor(L, tm->regressor);
    case TM_RECURRENT_REGRESSOR:
      return tk_tsetlin_predict_recurrent_regressor(L, tm->recurrent_regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in predict");
  }
  return 0;
}

static inline int tk_tsetlin_update_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  lua_settop(L, 5);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  lua_Integer tgt = luaL_checkinteger(L, 3);
  if (tgt < 0)
    luaL_error(L, "target class must be greater than zero");
  double specificity = tk_tsetlin_checkposdouble(L, 4);
  double active_clause = tk_tsetlin_checkposdouble(L, 5);
  mc_tm_initialize_active_clause(tm, active_clause);
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int clause_output[clause_chunks];
  unsigned int feedback_to_clauses[clause_chunks];
  unsigned int feedback_to_la[input_chunks];
  mc_tm_update(tm, bm, tgt, clause_output, feedback_to_clauses, feedback_to_la, specificity);
  return 0;
}

static inline int tk_tsetlin_update_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *tm)
{
  luaL_error(L, "unimplemented: update recurrent classifier");
  return 0;
}

static inline int tk_tsetlin_update_encoder (
  lua_State *L,
  tsetlin_encoder_t *tm
) {
  lua_settop(L, 8);
  unsigned int *a = (unsigned int *) luaL_checkstring(L, 2);
  unsigned int *n = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *p = (unsigned int *) luaL_checkstring(L, 4);
  double specificity = tk_tsetlin_checkposdouble(L, 5);
  double active_clause = tk_tsetlin_checkposdouble(L, 6);
  double margin = tk_tsetlin_checkposdouble(L, 7);
  double loss_alpha = tk_tsetlin_checkposdouble(L, 8);
  mc_tm_initialize_active_clause(&tm->encoder, active_clause);
  unsigned int clause_output[tm->encoder.clause_chunks];
  unsigned int feedback_to_clauses[tm->encoder.clause_chunks];
  unsigned int feedback_to_la[tm->encoder.input_chunks];
  long int scores[tm->encoder.classes];
  en_tm_update(tm, a, n, p, clause_output, feedback_to_clauses, feedback_to_la, scores, specificity, margin, loss_alpha);
  return 0;
}

static inline int tk_tsetlin_update_recurrent_encoder (lua_State *L, tsetlin_recurrent_encoder_t *tm)
{
  luaL_error(L, "unimplemented: update recurrent encoder");
  return 0;
}

static inline int tk_tsetlin_update_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 5);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  double specificity = tk_tsetlin_checkposdouble(L, 3);
  double active_clause = tk_tsetlin_checkposdouble(L, 4);
  double loss_alpha = tk_tsetlin_checkposdouble(L, 5);
  unsigned int clause_output[tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_clauses[tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_la_e[tm->encoder.encoder.input_chunks];
  unsigned int feedback_to_la_d[tm->encoder.encoding_chunks * 2];
  long int scores_e[tm->encoder.encoder.classes];
  long int scores_d[tm->decoder.encoder.classes];
  mc_tm_initialize_active_clause(&tm->encoder.encoder, active_clause);
  mc_tm_initialize_active_clause(&tm->decoder.encoder, active_clause);
  ae_tm_update(tm, bm,
      clause_output, feedback_to_clauses, feedback_to_la_e, feedback_to_la_d, scores_e, scores_d,
      specificity, loss_alpha);
  return 0;
}

static inline int tk_tsetlin_update_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  luaL_error(L, "unimplemented: update regressor");
  return 0;
}

static inline int tk_tsetlin_update_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *tm)
{
  luaL_error(L, "unimplemented: update recurrent regressor");
  return 0;
}

static inline int tk_tsetlin_update (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_update_classifier(L, tm->classifier);
    case TM_RECURRENT_CLASSIFIER:
      return tk_tsetlin_update_recurrent_classifier(L, tm->recurrent_classifier);
    case TM_ENCODER:
      return tk_tsetlin_update_encoder(L, tm->encoder);
    case TM_RECURRENT_ENCODER:
      return tk_tsetlin_update_recurrent_encoder(L, tm->recurrent_encoder);
    case TM_AUTO_ENCODER:
      return tk_tsetlin_update_auto_encoder(L, tm->auto_encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_update_regressor(L, tm->regressor);
    case TM_RECURRENT_REGRESSOR:
      return tk_tsetlin_update_recurrent_regressor(L, tm->recurrent_regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in update");
  }
  return 0;
}

#include <pthread.h>
#include <lua.h>
#include <lauxlib.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  tsetlin_classifier_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *shuffle;
  unsigned int *ps;
  unsigned int *ss;
  double specificity;
  pthread_mutex_t *qlock;
} train_classifier_thread_data_t;

static void *train_classifier_thread (void *arg)
{
  train_classifier_thread_data_t *data = (train_classifier_thread_data_t *) arg;
  unsigned int clause_chunks = data->tm->clause_chunks;
  unsigned int input_chunks = data->tm->input_chunks;
  unsigned int clause_output[clause_chunks];
  unsigned int feedback_to_clauses[clause_chunks];
  unsigned int feedback_to_la[input_chunks];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int idx = data->shuffle[next];
    mc_tm_update(data->tm, data->ps + idx * input_chunks, data->ss[idx],
        clause_output, feedback_to_clauses, feedback_to_la, data->specificity);
  }
}

static inline int tk_tsetlin_train_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  lua_settop(L, 6);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *)luaL_checkstring(L, 3);
  unsigned int *ss = (unsigned int *)luaL_checkstring(L, 4);
  double specificity = tk_tsetlin_checkposdouble(L, 5);
  double active_clause = tk_tsetlin_checkposdouble(L, 6);
  mc_tm_initialize_active_clause(tm, active_clause);

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores <= 0)
    return tk_error(L, "sysconf", errno);

  unsigned int next = 0;
  unsigned int shuffle[n];
  init_shuffle(shuffle, n);

  pthread_t threads[cores];
  pthread_mutex_t qlock;
  pthread_mutex_init(&qlock, NULL);
  train_classifier_thread_data_t thread_data[cores];

  for (unsigned int i = 0; i < cores; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].shuffle = shuffle;
    thread_data[i].ps = ps;
    thread_data[i].ss = ss;
    thread_data[i].specificity = specificity;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, train_classifier_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < cores; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&qlock);

  return 0;
}

static inline int tk_tsetlin_train_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *tm)
{
  luaL_error(L, "unimplemented: train recurrent classifier");
  return 0;
}

typedef struct {
  tsetlin_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *shuffle;
  unsigned int *tokens;
  double specificity;
  double margin;
  double loss_alpha;
  pthread_mutex_t *qlock;
} train_encoder_thread_data_t;

static void *train_encoder_thread (void *arg)
{
  train_encoder_thread_data_t *data = (train_encoder_thread_data_t *) arg;
  unsigned int clause_chunks = data->tm->encoder.clause_chunks;
  unsigned int encoder_classes = data->tm->encoder.classes;
  unsigned int input_chunks = data->tm->encoder.input_chunks;
  unsigned int clause_output[clause_chunks];
  unsigned int feedback_to_clauses[clause_chunks];
  unsigned int feedback_to_la[input_chunks];
  long int scores[encoder_classes];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int idx = data->shuffle[next];
    unsigned int *a = data->tokens + ((idx * 3 + 0) * input_chunks);
    unsigned int *n = data->tokens + ((idx * 3 + 1) * input_chunks);
    unsigned int *p = data->tokens + ((idx * 3 + 2) * input_chunks);
    en_tm_update(data->tm, a, n, p,
        clause_output, feedback_to_clauses, feedback_to_la, scores,
        data->specificity, data->margin, data->loss_alpha);
  }
  return NULL;
}

static inline int tk_tsetlin_train_encoder (
  lua_State *L,
  tsetlin_encoder_t *tm
) {
  lua_settop(L, 7);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *tokens = (unsigned int *) luaL_checkstring(L, 3);
  double specificity = tk_tsetlin_checkposdouble(L, 4);
  double active_clause = tk_tsetlin_checkposdouble(L, 5);
  double margin = tk_tsetlin_checkposdouble(L, 6);
  double loss_alpha = tk_tsetlin_checkposdouble(L, 7);
  mc_tm_initialize_active_clause(&tm->encoder, active_clause);

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores <= 0)
    return tk_error(L, "sysconf", errno);

  pthread_t threads[cores];
  pthread_mutex_t qlock;
  pthread_mutex_init(&qlock, NULL);
  train_encoder_thread_data_t thread_data[cores];

  unsigned int next = 0;
  unsigned int shuffle[n];
  init_shuffle(shuffle, n);

  for (unsigned int i = 0; i < cores; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].shuffle = shuffle;
    thread_data[i].tokens = tokens;
    thread_data[i].specificity = specificity;
    thread_data[i].margin = margin;
    thread_data[i].loss_alpha = loss_alpha;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, train_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < cores; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&qlock);

  return 0;
}

typedef struct {
  tsetlin_recurrent_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *shuffle;
  unsigned int *indices;
  unsigned int *tokens;
  double specificity;
  double margin;
  double loss_alpha;
  pthread_mutex_t *qlock;
} train_recurrent_encoder_thread_data_t;

static void *train_recurrent_encoder_thread (void *arg)
{
  train_recurrent_encoder_thread_data_t *data = (train_recurrent_encoder_thread_data_t *) arg;
  unsigned int input_a[data->tm->encoder.encoder.input_chunks];
  unsigned int input_n[data->tm->encoder.encoder.input_chunks];
  unsigned int input_p[data->tm->encoder.encoder.input_chunks];
  long int scores[data->tm->encoder.encoder.classes];
  unsigned int clause_output[data->tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_clauses[data->tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_la[data->tm->encoder.encoder.input_chunks];
  unsigned int *state_a = NULL;
  unsigned int *state_n = NULL;
  unsigned int *state_p = NULL;
  unsigned int state_a_size = 0;
  unsigned int state_n_size = 0;
  unsigned int state_p_size = 0;
  unsigned int state_a_max = 0;
  unsigned int state_n_max = 0;
  unsigned int state_p_max = 0;
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
    {
      free(state_a);
      free(state_n);
      free(state_p);
      return NULL;
    }
    unsigned int idx = data->shuffle[next];
    unsigned int a_offset = data->indices[idx * 6 + 0] * data->tm->token_chunks;
    unsigned int a_len = data->indices[idx * 6 + 1];
    unsigned int *a_data = data->tokens + a_offset;
    unsigned int n_offset = data->indices[idx * 6 + 2] * data->tm->token_chunks;
    unsigned int n_len = data->indices[idx * 6 + 3];
    unsigned int *n_data = data->tokens + n_offset;
    unsigned int p_offset = data->indices[idx * 6 + 4] * data->tm->token_chunks;
    unsigned int p_len = data->indices[idx * 6 + 5];
    unsigned int *p_data = data->tokens + p_offset;
    re_tm_update(data->tm, a_len, a_data, n_len, n_data, p_len, p_data,
        input_a, input_n, input_p,
        &state_a, &state_a_size, &state_a_max,
        &state_n, &state_n_size, &state_n_max,
        &state_p, &state_p_size, &state_p_max,
        clause_output, feedback_to_clauses, feedback_to_la, scores,
        data->specificity, data->margin, data->loss_alpha);
  }
}

static inline int tk_tsetlin_train_recurrent_encoder (
  lua_State *L,
  tsetlin_recurrent_encoder_t *tm
) {
  lua_settop(L, 8);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *indices = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *tokens = (unsigned int *) luaL_checkstring(L, 4);
  double specificity = tk_tsetlin_checkposdouble(L, 5);
  double active_clause = tk_tsetlin_checkposdouble(L, 6);
  double margin = tk_tsetlin_checkposdouble(L, 7);
  double loss_alpha = tk_tsetlin_checkposdouble(L, 8);
  mc_tm_initialize_active_clause(&tm->encoder.encoder, active_clause);

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores <= 0)
    return tk_error(L, "sysconf", errno);

  pthread_t threads[cores];
  pthread_mutex_t qlock;
  pthread_mutex_init(&qlock, NULL);
  train_recurrent_encoder_thread_data_t thread_data[cores];

  unsigned int next = 0;
  unsigned int shuffle[n];
  init_shuffle(shuffle, n);

  for (unsigned int i = 0; i < cores; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].shuffle = shuffle;
    thread_data[i].indices = indices;
    thread_data[i].tokens = tokens;
    thread_data[i].specificity = specificity;
    thread_data[i].margin = margin;
    thread_data[i].loss_alpha = loss_alpha;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, train_recurrent_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < cores; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&qlock);

  return 0;
}

typedef struct {
  tsetlin_auto_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *shuffle;
  unsigned int *ps;
  double specificity;
  double loss_alpha;
  pthread_mutex_t *qlock;
} train_auto_encoder_thread_data_t;

static void *train_auto_encoder_thread (void *arg)
{
  train_auto_encoder_thread_data_t *data = (train_auto_encoder_thread_data_t *) arg;
  unsigned int clause_output[data->tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_clauses[data->tm->encoder.encoder.clause_chunks];
  unsigned int feedback_to_la_e[data->tm->encoder.encoder.input_chunks];
  unsigned int feedback_to_la_d[data->tm->encoder.encoding_chunks * 2];
  long int scores_e[data->tm->encoder.encoder.classes];
  long int scores_d[data->tm->decoder.encoder.classes];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int idx = data->shuffle[next];
    ae_tm_update(data->tm, data->ps + idx * data->tm->encoder.encoder.input_chunks,
        clause_output, feedback_to_clauses, feedback_to_la_e, feedback_to_la_d, scores_e, scores_d,
        data->specificity, data->loss_alpha);
  }
  return NULL;
}

static inline int tk_tsetlin_train_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 6);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  double specificity = tk_tsetlin_checkposdouble(L, 4);
  double active_clause = tk_tsetlin_checkposdouble(L, 5);
  double loss_alpha = tk_tsetlin_checkposdouble(L, 6);

  // TODO: Should the active clause be shared? Does that make more sense for an
  // auto_encoder?
  mc_tm_initialize_active_clause(&tm->encoder.encoder, active_clause);
  mc_tm_initialize_active_clause(&tm->decoder.encoder, active_clause);

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores <= 0)
    return tk_error(L, "sysconf", errno);

  pthread_t threads[cores];
  pthread_mutex_t qlock;
  pthread_mutex_init(&qlock, NULL);
  train_auto_encoder_thread_data_t thread_data[cores];

  unsigned int next = 0;
  unsigned int shuffle[n];
  init_shuffle(shuffle, n);

  for (unsigned int i = 0; i < cores; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].shuffle = shuffle;
    thread_data[i].ps = ps;
    thread_data[i].specificity = specificity;
    thread_data[i].loss_alpha = loss_alpha;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, train_auto_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < cores; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  return 0;
}

static inline int tk_tsetlin_train_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  luaL_error(L, "unimplemented: train regressor");
  return 0;
}

static inline int tk_tsetlin_train_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *tm)
{
  luaL_error(L, "unimplemented: train recurrent regressor");
  return 0;
}

static inline int tk_tsetlin_train (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_train_classifier(L, tm->classifier);
    case TM_RECURRENT_CLASSIFIER:
      return tk_tsetlin_train_recurrent_classifier(L, tm->recurrent_classifier);
    case TM_ENCODER:
      return tk_tsetlin_train_encoder(L, tm->encoder);
    case TM_RECURRENT_ENCODER:
      return tk_tsetlin_train_recurrent_encoder(L, tm->recurrent_encoder);
    case TM_AUTO_ENCODER:
      return tk_tsetlin_train_auto_encoder(L, tm->auto_encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_train_regressor(L, tm->regressor);
    case TM_RECURRENT_REGRESSOR:
      return tk_tsetlin_train_recurrent_regressor(L, tm->recurrent_regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in train");
  }
}

typedef struct {
  tsetlin_classifier_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *ps;
  unsigned int *ss;
  bool track_stats;
  unsigned int *correct;
  unsigned int **observations;
  unsigned int **predictions;
  unsigned int **confusion;
  pthread_mutex_t *lock;
  pthread_mutex_t *qlock;
} evaluate_classifier_thread_data_t;

static void *evaluate_classifier_thread (void *arg)
{
  evaluate_classifier_thread_data_t *data = (evaluate_classifier_thread_data_t *) arg;
  unsigned int classes = data->tm->classes;
  unsigned int input_chunks = data->tm->input_chunks;
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int expected = data->ss[next];
    unsigned int predicted = mc_tm_predict(data->tm, data->ps + next * input_chunks);
    pthread_mutex_lock(data->lock);
    if (expected == predicted)
      (*data->correct) += 1;
    if (data->track_stats) {
      (*data->observations)[expected] ++;
      (*data->predictions)[predicted] ++;
      if (expected != predicted)
        (*data->confusion)[expected * classes + predicted] ++;
    }
    pthread_mutex_unlock(data->lock);
  }
}

static inline int tk_tsetlin_evaluate_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm
) {

  lua_settop(L, 5);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *ss = (unsigned int *) luaL_checkstring(L, 4);
  bool track_stats = tk_tsetlin_checkboolean(L, 5);
  unsigned int correct = 0;
  unsigned int *confusion = NULL;
  unsigned int *predictions = NULL;
  unsigned int *observations = NULL;
  unsigned int classes = tm->classes;

  if (track_stats) {
    confusion = malloc(sizeof(unsigned int) * classes * classes);
    predictions = malloc(sizeof(unsigned int) * classes);
    observations = malloc(sizeof(unsigned int) * classes);
    if (!(confusion && predictions && observations))
      luaL_error(L, "error in malloc during evaluation");
    memset(confusion, 0, sizeof(unsigned int) * classes * classes);
    memset(predictions, 0, sizeof(unsigned int) * classes);
    memset(observations, 0, sizeof(unsigned int) * classes);
  }

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores <= 0)
    return tk_error(L, "sysconf", errno);

  pthread_t threads[cores];
  pthread_mutex_t lock;
  pthread_mutex_t qlock;
  pthread_mutex_init(&lock, NULL);
  pthread_mutex_init(&qlock, NULL);
  evaluate_classifier_thread_data_t thread_data[cores];

  unsigned int next = 0;

  for (unsigned int i = 0; i < cores; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].ps = ps;
    thread_data[i].ss = ss;
    thread_data[i].track_stats = track_stats;
    thread_data[i].correct = &correct;
    thread_data[i].observations = &observations;
    thread_data[i].predictions = &predictions;
    thread_data[i].confusion = &confusion;
    thread_data[i].lock = &lock;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, evaluate_classifier_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < cores; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&lock);
  pthread_mutex_destroy(&qlock);

  lua_pushnumber(L, correct);
  if (track_stats) {
    lua_newtable(L); // ct
    for (unsigned int i = 0; i < classes; i ++) {
      for (unsigned int j = 0; j < classes; j ++) {
        unsigned int c = confusion[i * classes + j];
        if (c > 0) {
          lua_pushinteger(L, i); // ct i
          lua_gettable(L, -2); // ct t
          if (lua_type(L, -1) == LUA_TNIL) {
            lua_pop(L, 1); // ct
            lua_newtable(L); // ct t
            lua_pushinteger(L, i); // ct t i
            lua_pushvalue(L, -2); // ct t i t
            lua_settable(L, -4); // ct t
          }
          lua_pushinteger(L, j); // ct t j
          lua_pushinteger(L, c); // ct t j c
          lua_settable(L, -3); // ct t
          lua_pop(L, 1);
        }
      }
    }

    lua_newtable(L); // pt
    for (unsigned int i = 0; i < classes; i ++) {
      lua_pushinteger(L, i); // pt i
      lua_pushinteger(L, predictions[i]); // pt i p
      lua_settable(L, -3); // pt
    }

    lua_newtable(L); // ot
    for (unsigned int i = 0; i < classes; i ++) {
      lua_pushinteger(L, i); // ot i
      lua_pushinteger(L, observations[i]); // ot i o
      lua_settable(L, -3); // ot
    }

    free(confusion);
    free(predictions);
    free(observations);
    return 4;

  } else {
    return 1;
  }
}

static inline int tk_tsetlin_evaluate_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *tm)
{
  luaL_error(L, "unimplemented: evaluate recurrent classifier");
  return 0;
}

typedef struct {
  tsetlin_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *tokens;
  double margin;
  unsigned int *correct;
  pthread_mutex_t *lock;
  pthread_mutex_t *qlock;
} evaluate_encoder_thread_data_t;

static void *evaluate_encoder_thread (void *arg)
{
  evaluate_encoder_thread_data_t *data = (evaluate_encoder_thread_data_t *) arg;
  unsigned int encoding_a[data->tm->encoding_chunks];
  unsigned int encoding_n[data->tm->encoding_chunks];
  unsigned int encoding_p[data->tm->encoding_chunks];
  long int scores[data->tm->encoder.classes];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int *a = data->tokens + ((next * 3 + 0) * data->tm->encoder.input_chunks);
    unsigned int *n = data->tokens + ((next * 3 + 1) * data->tm->encoder.input_chunks);
    unsigned int *p = data->tokens + ((next * 3 + 2) * data->tm->encoder.input_chunks);
    en_tm_encode(data->tm, a, encoding_a, scores);
    en_tm_encode(data->tm, n, encoding_n, scores);
    en_tm_encode(data->tm, p, encoding_p, scores);
    unsigned int dist_an = hamming(encoding_a, encoding_n, data->tm->encoding_bits);
    unsigned int dist_ap = hamming(encoding_a, encoding_p, data->tm->encoding_bits);
    if (dist_ap < dist_an) {
      pthread_mutex_lock(data->lock);
      (*data->correct) += 1;
      pthread_mutex_unlock(data->lock);
    }
  }
  return NULL;
}

static inline int tk_tsetlin_evaluate_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 3);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *tokens = (unsigned int *) luaL_checkstring(L, 3);

  unsigned int correct = 0;

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores <= 0)
    return tk_error(L, "sysconf", errno);

  pthread_t threads[cores];
  pthread_mutex_t lock;
  pthread_mutex_t qlock;
  pthread_mutex_init(&lock, NULL);
  pthread_mutex_init(&qlock, NULL);
  evaluate_encoder_thread_data_t thread_data[cores];

  unsigned int next = 0;

  for (unsigned int i = 0; i < cores; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].tokens = tokens;
    thread_data[i].correct = &correct;
    thread_data[i].lock = &lock;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, evaluate_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < cores; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&lock);
  pthread_mutex_destroy(&qlock);

  lua_pushnumber(L, (double) correct / n);
  return 1;
}

typedef struct {
  tsetlin_recurrent_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *indices;
  unsigned int *tokens;
  unsigned int *correct;
  pthread_mutex_t *lock;
  pthread_mutex_t *qlock;
} evaluate_recurrent_encoder_thread_data_t;

static void *evaluate_recurrent_encoder_thread (void *arg)
{
  evaluate_recurrent_encoder_thread_data_t *data = (evaluate_recurrent_encoder_thread_data_t *) arg;
  unsigned int input_a[data->tm->encoder.encoder.input_chunks];
  unsigned int input_n[data->tm->encoder.encoder.input_chunks];
  unsigned int input_p[data->tm->encoder.encoder.input_chunks];
  long int scores[data->tm->encoder.encoder.classes];
  unsigned int *state_a = NULL;
  unsigned int *state_n = NULL;
  unsigned int *state_p = NULL;
  unsigned int state_a_size = 0;
  unsigned int state_n_size = 0;
  unsigned int state_p_size = 0;
  unsigned int state_a_max = 0;
  unsigned int state_n_max = 0;
  unsigned int state_p_max = 0;
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n) {
      free(state_a);
      free(state_n);
      free(state_p);
      return NULL;
    }
    unsigned int a_offset = data->indices[next * 6 + 0] * data->tm->token_chunks;
    unsigned int a_len = data->indices[next * 6 + 1];
    unsigned int *a_data = data->tokens + a_offset;
    unsigned int n_offset = data->indices[next * 6 + 2] * data->tm->token_chunks;
    unsigned int n_len = data->indices[next * 6 + 3];
    unsigned int *n_data = data->tokens + n_offset;
    unsigned int p_offset = data->indices[next * 6 + 4] * data->tm->token_chunks;
    unsigned int p_len = data->indices[next * 6 + 5];
    unsigned int *p_data = data->tokens + p_offset;
    re_tm_encode(data->tm, 1, a_len, a_data, &state_a, &state_a_size, &state_a_max, input_a, scores);
    re_tm_encode(data->tm, 1, n_len, n_data, &state_n, &state_n_size, &state_n_max, input_n, scores);
    re_tm_encode(data->tm, 1, p_len, p_data, &state_p, &state_p_size, &state_p_max, input_p, scores);
    unsigned int encoding_bits = data->tm->encoder.encoding_bits;
    unsigned int encoding_chunks = data->tm->encoder.encoding_chunks;
    unsigned int *encoding_a = state_a + ((state_a_size - 1) * encoding_chunks);
    unsigned int *encoding_n = state_n + ((state_n_size - 1) * encoding_chunks);
    unsigned int *encoding_p = state_p + ((state_p_size - 1) * encoding_chunks);
    unsigned int dist_an = hamming(encoding_a, encoding_n, encoding_bits);
    unsigned int dist_ap = hamming(encoding_a, encoding_p, encoding_bits);
    if (dist_ap < dist_an)
    {
      pthread_mutex_lock(data->lock);
      (*data->correct) += 1;
      pthread_mutex_unlock(data->lock);
    }
  }
}

static inline int tk_tsetlin_evaluate_recurrent_encoder (
  lua_State *L,
  tsetlin_recurrent_encoder_t *tm
) {
  lua_settop(L, 4);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *indices = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *tokens = (unsigned int *) luaL_checkstring(L, 4);
  unsigned int correct = 0;

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores <= 0)
    return tk_error(L, "sysconf", errno);

  pthread_t threads[cores];
  pthread_mutex_t lock;
  pthread_mutex_t qlock;
  pthread_mutex_init(&lock, NULL);
  pthread_mutex_init(&qlock, NULL);
  evaluate_recurrent_encoder_thread_data_t thread_data[cores];

  unsigned int next = 0;

  for (unsigned int i = 0; i < cores; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].indices = indices;
    thread_data[i].tokens = tokens;
    thread_data[i].correct = &correct;
    thread_data[i].lock = &lock;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, evaluate_recurrent_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < cores; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&lock);
  pthread_mutex_destroy(&qlock);

  lua_pushnumber(L, (double) correct / n);
  return 1;
}

typedef struct {
  tsetlin_auto_encoder_t *tm;
  unsigned int n;
  unsigned int *next;
  unsigned int *ps;
  unsigned int *total_diff;
  pthread_mutex_t *lock;
  pthread_mutex_t *qlock;
} evaluate_auto_encoder_thread_data_t;

static void *evaluate_auto_encoder_thread (void *arg)
{
  evaluate_auto_encoder_thread_data_t *data = (evaluate_auto_encoder_thread_data_t *) arg;
  unsigned int encoding[data->tm->encoder.encoding_chunks * 2];
  unsigned int decoding[data->tm->decoder.encoding_chunks * 2];
  long int scores_e[data->tm->encoder.encoder.classes];
  long int scores_d[data->tm->decoder.encoder.classes];
  while (1) {
    pthread_mutex_lock(data->qlock);
    unsigned int next = *data->next;
    (*data->next) += 1;
    pthread_mutex_unlock(data->qlock);
    if (next >= data->n)
      return NULL;
    unsigned int *input = data->ps + next * data->tm->encoder.encoder.input_chunks;
    ae_tm_encode(data->tm, input, encoding, scores_e);
    memcpy(encoding + data->tm->encoder.encoding_chunks, encoding, data->tm->encoder.encoding_chunks * sizeof(unsigned int));
    flip_bits(encoding + data->tm->encoder.encoding_chunks, data->tm->encoder.encoding_chunks);
    ae_tm_decode(data->tm, encoding, decoding, scores_d);
    memcpy(decoding + data->tm->decoder.encoding_chunks, decoding, data->tm->decoder.encoding_chunks * sizeof(unsigned int));
    flip_bits(decoding + data->tm->decoder.encoding_chunks, data->tm->decoder.encoding_chunks);
    pthread_mutex_lock(data->lock);
    (*data->total_diff) += hamming(input, decoding, data->tm->encoder.encoder.input_bits);
    pthread_mutex_unlock(data->lock);
  }
}

static inline int tk_tsetlin_evaluate_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 3);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int input_bits = tm->encoder.encoder.input_bits;

  unsigned int total_bits = n * input_bits;
  unsigned int total_diff = 0;

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores <= 0)
    return tk_error(L, "sysconf", errno);

  pthread_t threads[cores];
  pthread_mutex_t lock;
  pthread_mutex_t qlock;
  pthread_mutex_init(&lock, NULL);
  pthread_mutex_init(&qlock, NULL);
  evaluate_auto_encoder_thread_data_t thread_data[cores];

  unsigned int next = 0;

  for (unsigned int i = 0; i < cores; i++) {
    thread_data[i].tm = tm;
    thread_data[i].n = n;
    thread_data[i].next = &next;
    thread_data[i].ps = ps;
    thread_data[i].total_diff = &total_diff;
    thread_data[i].lock = &lock;
    thread_data[i].qlock = &qlock;
    if (pthread_create(&threads[i], NULL, evaluate_auto_encoder_thread, &thread_data[i]) != 0)
      return tk_error(L, "pthread_create", errno);
  }

  // TODO: Ensure these get freed on error above
  for (unsigned int i = 0; i < cores; i++)
    if (pthread_join(threads[i], NULL) != 0)
      return tk_error(L, "pthread_join", errno);

  pthread_mutex_destroy(&lock);
  pthread_mutex_destroy(&qlock);

  lua_pushnumber(L, 1 - (double) total_diff / (double) total_bits);
  return 1;
}

static inline int tk_tsetlin_evaluate_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  luaL_error(L, "unimplemented: evaluate regressor");
  return 0;
}

static inline int tk_tsetlin_evaluate_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *tm)
{
  luaL_error(L, "unimplemented: evaluate recurrent regressor");
  return 0;
}

static inline int tk_tsetlin_evaluate (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_evaluate_classifier(L, tm->classifier);
    case TM_RECURRENT_CLASSIFIER:
      return tk_tsetlin_evaluate_recurrent_classifier(L, tm->recurrent_classifier);
    case TM_ENCODER:
      return tk_tsetlin_evaluate_encoder(L, tm->encoder);
    case TM_RECURRENT_ENCODER:
      return tk_tsetlin_evaluate_recurrent_encoder(L, tm->recurrent_encoder);
    case TM_AUTO_ENCODER:
      return tk_tsetlin_evaluate_auto_encoder(L, tm->auto_encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_evaluate_regressor(L, tm->regressor);
    case TM_RECURRENT_REGRESSOR:
      return tk_tsetlin_evaluate_recurrent_regressor(L, tm->recurrent_regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in evaluate");
  }
}

static inline FILE *tk_lua_fopen (lua_State *L, const char *fp, const char *flag)
{
  FILE *fh = fopen(fp, flag);
  if (fh) return fh;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error opening file");
  lua_pushstring(L, fp);
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 4, 0, "santoku.error", "error");
  return NULL;
}

static inline void tk_lua_fclose (lua_State *L, FILE *fh)
{
  if (!fclose(fh)) return;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error closing file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
}

static inline void tk_lua_fwrite (lua_State *L, void *data, size_t size, size_t memb, FILE *fh)
{
  fwrite(data, size, memb, fh);
  if (!ferror(fh)) return;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error writing to file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
}

static inline void tk_lua_fread (lua_State *L, void *data, size_t size, size_t memb, FILE *fh)
{
  size_t r = fread(data, size, memb, fh);
  if (!ferror(fh) || !r) return;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error reading from file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
}

static inline void _tk_tsetlin_persist_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh)
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
  tk_lua_fwrite(L, &tm->filter, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, tm->state, sizeof(unsigned int), tm->state_chunks, fh);
  tk_lua_fwrite(L, tm->actions, sizeof(unsigned int), tm->action_chunks, fh);
}

static inline void tk_tsetlin_persist_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh)
{
  _tk_tsetlin_persist_classifier(L, tm, fh);
}

static inline void tk_tsetlin_persist_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *tm, FILE *fh)
{
  _tk_tsetlin_persist_classifier(L, &tm->classifier, fh);
}

static inline void tk_tsetlin_persist_encoder (lua_State *L, tsetlin_encoder_t *tm, FILE *fh)
{
  tk_lua_fwrite(L, &tm->encoding_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->encoding_chunks, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->encoding_filter, sizeof(unsigned int), 1, fh);
  _tk_tsetlin_persist_classifier(L, &tm->encoder, fh);
}

static inline void tk_tsetlin_persist_recurrent_encoder (lua_State *L, tsetlin_recurrent_encoder_t *tm, FILE *fh)
{
  tk_lua_fwrite(L, &tm->token_bits, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &tm->token_chunks, sizeof(unsigned int), 1, fh);
  tk_tsetlin_persist_encoder(L, &tm->encoder, fh);
}

static inline void tk_tsetlin_persist_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm, FILE *fh)
{
  tk_tsetlin_persist_encoder(L, &tm->encoder, fh);
  tk_tsetlin_persist_encoder(L, &tm->decoder, fh);
}

static inline void tk_tsetlin_persist_regressor (lua_State *L, tsetlin_regressor_t *tm, FILE *fh)
{
  _tk_tsetlin_persist_classifier(L, &tm->classifier, fh);
}

static inline void tk_tsetlin_persist_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *tm, FILE *fh)
{
  _tk_tsetlin_persist_classifier(L, &tm->classifier, fh);
}

static inline int tk_tsetlin_persist (lua_State *L)
{
  lua_settop(L, 2);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  const char *fp = luaL_checkstring(L, 2);
  FILE *fh = tk_lua_fopen(L, fp, "w");
  tk_lua_fwrite(L, &tm->type, sizeof(tsetlin_type_t), 1, fh);
  switch (tm->type) {
    case TM_CLASSIFIER:
      tk_tsetlin_persist_classifier(L, tm->classifier, fh);
      break;
    case TM_RECURRENT_CLASSIFIER:
      tk_tsetlin_persist_recurrent_classifier(L, tm->recurrent_classifier, fh);
      break;
    case TM_ENCODER:
      tk_tsetlin_persist_encoder(L, tm->encoder, fh);
      break;
    case TM_RECURRENT_ENCODER:
      tk_tsetlin_persist_recurrent_encoder(L, tm->recurrent_encoder, fh);
      break;
    case TM_AUTO_ENCODER:
      tk_tsetlin_persist_auto_encoder(L, tm->auto_encoder, fh);
      break;
    case TM_REGRESSOR:
      tk_tsetlin_persist_regressor(L, tm->regressor, fh);
      break;
    case TM_RECURRENT_REGRESSOR:
      tk_tsetlin_persist_recurrent_regressor(L, tm->recurrent_regressor, fh);
      break;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in persist");
  }
  tk_lua_fclose(L, fh);
  return 0;
}

static inline void _tk_tsetlin_load_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh)
{
  tk_lua_fread(L, &tm->classes, sizeof(unsigned int), 1, fh);
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
  tk_lua_fread(L, &tm->filter, sizeof(unsigned int), 1, fh);
  tm->state = malloc(sizeof(unsigned int) * tm->state_chunks);
  tm->actions = malloc(sizeof(unsigned int) * tm->action_chunks);
  tm->active_clause = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->locks = malloc(sizeof(pthread_mutex_t) * tm->action_chunks);
  for (unsigned int i = 0; i < tm->action_chunks; i ++)
    pthread_mutex_init(&tm->locks[i], NULL);
  tk_lua_fread(L, tm->state, sizeof(unsigned int), tm->state_chunks, fh);
  tk_lua_fread(L, tm->actions, sizeof(unsigned int), tm->action_chunks, fh);
}

static inline void tk_tsetlin_load_classifier (lua_State *L, FILE *fh)
{
  tsetlin_t *tm = tk_tsetlin_alloc_classifier(L);
  _tk_tsetlin_load_classifier(L, tm->classifier, fh);
}

static inline void tk_tsetlin_load_recurrent_classifier (lua_State *L, FILE *fh)
{
  luaL_error(L, "unimplemented: load recurrent classifier");
}

static inline void _tk_tsetlin_load_encoder (lua_State *L, tsetlin_encoder_t *en, FILE *fh)
{
  tk_lua_fread(L, &en->encoding_bits, sizeof(en->encoding_bits), 1, fh);
  tk_lua_fread(L, &en->encoding_chunks, sizeof(en->encoding_chunks), 1, fh);
  tk_lua_fread(L, &en->encoding_filter, sizeof(en->encoding_filter), 1, fh);
  _tk_tsetlin_load_classifier(L, &en->encoder, fh);
}

static inline void tk_tsetlin_load_encoder (lua_State *L, FILE *fh)
{
  tsetlin_t *tm = tk_tsetlin_alloc_encoder(L);
  tsetlin_encoder_t *en = tm->encoder;
  _tk_tsetlin_load_encoder(L, en, fh);
}

static inline void tk_tsetlin_load_recurrent_encoder (lua_State *L, FILE *fh)
{
  tsetlin_t *tm = tk_tsetlin_alloc_recurrent_encoder(L);
  tsetlin_recurrent_encoder_t *en = tm->recurrent_encoder;
  tk_lua_fread(L, &en->token_bits, sizeof(en->token_bits), 1, fh);
  tk_lua_fread(L, &en->token_chunks, sizeof(en->token_chunks), 1, fh);
  _tk_tsetlin_load_encoder(L, &en->encoder, fh);
}

static inline void tk_tsetlin_load_auto_encoder (lua_State *L, FILE *fh)
{
  tsetlin_t *tm = tk_tsetlin_alloc_auto_encoder(L);
  tsetlin_auto_encoder_t *ae = tm->auto_encoder;
  _tk_tsetlin_load_encoder(L, &ae->encoder, fh);
  _tk_tsetlin_load_encoder(L, &ae->decoder, fh);
}

static inline void tk_tsetlin_load_regressor (lua_State *L, FILE *fh)
{
  tsetlin_t *tm = tk_tsetlin_alloc_regressor(L);
  tsetlin_regressor_t *rg = tm->regressor;
  _tk_tsetlin_load_classifier(L, &rg->classifier, fh);
}

static inline void tk_tsetlin_load_recurrent_regressor (lua_State *L, FILE *fh)
{
  luaL_error(L, "unimplemented: load recurrent regressor");
}

// TODO: Merge malloc/assignment logic from load_* and create_* to reduce
// changes for coding errors
static inline int tk_tsetlin_load (lua_State *L)
{
  lua_settop(L, 1);
  const char *fp = luaL_checkstring(L, 1);
  lua_pop(L, 1);
  FILE *fh = tk_lua_fopen(L, fp, "r");
  tsetlin_type_t type;
  tk_lua_fread(L, &type, sizeof(type), 1, fh);
  switch (type) {
    case TM_CLASSIFIER:
      tk_tsetlin_load_classifier(L, fh);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_RECURRENT_CLASSIFIER:
      tk_tsetlin_load_recurrent_classifier(L, fh);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_ENCODER:
      tk_tsetlin_load_encoder(L, fh);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_RECURRENT_ENCODER:
      tk_tsetlin_load_recurrent_encoder(L, fh);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_AUTO_ENCODER:
      tk_tsetlin_load_auto_encoder(L, fh);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_REGRESSOR:
      tk_tsetlin_load_regressor(L, fh);
      tk_lua_fclose(L, fh);
      return 1;
    case TM_RECURRENT_REGRESSOR:
      tk_tsetlin_load_recurrent_regressor(L, fh);
      tk_lua_fclose(L, fh);
      return 1;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in load");
  }
}

static inline int tk_tsetlin_type (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      lua_pushstring(L, "classifier");
      break;
    case TM_RECURRENT_CLASSIFIER:
      lua_pushstring(L, "recurrent_classifier");
      break;
    case TM_ENCODER:
      lua_pushstring(L, "encoder");
      break;
    case TM_RECURRENT_ENCODER:
      lua_pushstring(L, "recurrent_encoder");
      break;
    case TM_AUTO_ENCODER:
      lua_pushstring(L, "auto_encoder");
      break;
    case TM_REGRESSOR:
      lua_pushstring(L, "regressor");
      break;
    case TM_RECURRENT_REGRESSOR:
      lua_pushstring(L, "recurrent_regressor");
      break;
    default:
      return luaL_error(L, "unexpected tsetlin machine type in type");
  }
  return 1;
}

static luaL_Reg tk_tsetlin_fns[] =
{
  { "create", tk_tsetlin_create },
  { "destroy", tk_tsetlin_destroy },
  { "update", tk_tsetlin_update },
  { "predict", tk_tsetlin_predict },
  { "train", tk_tsetlin_train },
  { "evaluate", tk_tsetlin_evaluate },
  { "persist", tk_tsetlin_persist },
  { "load", tk_tsetlin_load },
  { "type", tk_tsetlin_type },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_capi (lua_State *L)
{
  srand(0);
  lua_newtable(L); // t
  tk_tsetlin_register(L, tk_tsetlin_fns, 0); // t
  luaL_newmetatable(L, TK_TSETLIN_MT); // t mt
  lua_pushcfunction(L, tk_tsetlin_destroy); // t mt fn
  lua_setfield(L, -2, "__gc"); // t mt
  lua_pop(L, 1); // t
  return 1;
}
