/*

Copyright (C) 2024 Birch Point SWE (Lua integration, train/evaluate, drop clause, autoencoder, regressor)
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
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

typedef enum {
  TM_CLASSIFIER,
  TM_AUTOENCODER,
  TM_REGRESSOR,
  TM_ENCODER,
} tsetlin_type_t;

typedef struct {
  unsigned int classes;
  unsigned int features;
  unsigned int clauses;
  unsigned int threshold;
  unsigned int state_bits;
  bool boost_true_positive;
  unsigned int input_chunks;
  unsigned int clause_chunks;
  unsigned int filter;
  unsigned int *state; // class clause bit chunk
  unsigned int *actions; // class clause chunk
  unsigned int *clause_output;
  unsigned int *feedback_to_la;
  unsigned int *feedback_to_clauses;
  unsigned int *drop_clause;
} tsetlin_classifier_t;

typedef struct {
  unsigned int *encoding;
  unsigned int *decoding;
  tsetlin_classifier_t encoder;
  tsetlin_classifier_t decoder;
} tsetlin_autoencoder_t;

typedef struct {
  unsigned int encoding_bits;
  unsigned int encoding_chunks;
  unsigned int encoding_filter;
  unsigned int *encoding_a;
  unsigned int *encoding_b;
  tsetlin_classifier_t encoder;
} tsetlin_encoder_t;

typedef struct {
  tsetlin_classifier_t classifier;
} tsetlin_regressor_t;

typedef struct {
  tsetlin_type_t type;
  union {
    tsetlin_classifier_t *classifier;
    tsetlin_autoencoder_t *autoencoder;
    tsetlin_encoder_t *encoder;
    tsetlin_regressor_t *regressor;
  };
} tsetlin_t;

#define tm_state_idx_counts(tm, class, clause, input_chunk) \
  (&(tm)->state[(class) * (tm)->clauses * (tm)->input_chunks * ((tm)->state_bits - 1) + \
                (clause) * (tm)->input_chunks * ((tm)->state_bits - 1) + \
                (input_chunk) * ((tm)->state_bits - 1)])

#define tm_state_idx_actions(tm, class, clause) \
  (&(tm)->actions[(class) * (tm)->clauses *  (tm)->input_chunks + \
                  (clause) * (tm)->input_chunks])

static uint64_t const multiplier = 6364136223846793005u;
static uint64_t mcg_state = 0xcafef00dd15ea5e5u;

static inline uint32_t fast_rand ()
{
  uint64_t x = mcg_state;
  unsigned int count = (unsigned int) (x >> 61);
  mcg_state = x * multiplier;
  return (uint32_t) ((x ^ x >> 22) >> (22 + count));
}

static inline int normal (double mean, double variance)
{
  double u1 = (double) (fast_rand() + 1) / ((double) UINT32_MAX + 1);
  double u2 = (double) fast_rand() / UINT32_MAX;
  double n1 = sqrt(-2 * log(u1)) * sin(8 * atan(1) * u2);
  return (int) round(mean + sqrt(variance) * n1);
}

double jaccard (unsigned int *a, unsigned int *b, unsigned int n) {
  int n_intersection = 0;
  int n_union = 0;
  for (unsigned int i = 0; i < n; i++) {
    n_intersection += __builtin_popcount(a[i] & b[i]);
    n_union += __builtin_popcount(a[i] | b[i]);
  }
  if (n_union == 0)
    return 0.0;
  else
    return (double) n_intersection / n_union;
}

static inline unsigned int hamming (unsigned int *a, unsigned int *b, unsigned int n) {
  unsigned int distance = 0;
  for (unsigned int i = 0; i < n; i ++) {
    unsigned int diff = a[i] ^ b[i];
    while (diff) {
      distance += diff & 1;
      diff >>= 1;
    }
  }
  return distance;
}

static inline void tm_initialize_random_streams (tsetlin_classifier_t *tm, double specificity)
{
  memset((*tm).feedback_to_la, 0, tm->input_chunks * sizeof(unsigned int));
  long int n = 2 * tm->features;
  double p = 1.0 / specificity;
  long int active = normal(n * p, n * p * (1 - p));
  active = active >= n ? n : active;
  active = active < 0 ? 0 : active;
  while (active--) {
    unsigned int f = fast_rand() % (2 * tm->features);
    while ((*tm).feedback_to_la[f / (sizeof(unsigned int) * CHAR_BIT)] & (1 << (f % (sizeof(unsigned int) * CHAR_BIT))))
      f = fast_rand() % (2 * tm->features);
    (*tm).feedback_to_la[f / (sizeof(unsigned int) * CHAR_BIT)] |= 1 << (f % (sizeof(unsigned int) * CHAR_BIT));
  }
}

static inline void tm_inc (tsetlin_classifier_t *tm, unsigned int class, unsigned int clause, unsigned int chunk, unsigned int active)
{
  unsigned int m = tm->state_bits - 1;
  unsigned int *counts = tm_state_idx_counts(tm, class, clause, chunk);
  unsigned int carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = counts[b] & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  unsigned int *actions = tm_state_idx_actions(tm, class, clause);
  carry_next = actions[chunk] & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] |= carry;
  actions[chunk] |= carry;
}

static inline void tm_dec (tsetlin_classifier_t *tm, unsigned int class, unsigned int clause, unsigned int chunk, unsigned int active)
{
  unsigned int m = tm->state_bits - 1;
  unsigned int *counts = tm_state_idx_counts(tm, class, clause, chunk);
  unsigned int carry, carry_next;
  carry = active;
  for (unsigned int b = 0; b < m; b ++) {
    carry_next = (~counts[b]) & carry;
    counts[b] ^= carry;
    carry = carry_next;
  }
  unsigned int *actions = tm_state_idx_actions(tm, class, clause);
  carry_next = (~actions[chunk]) & carry;
  actions[chunk] ^= carry;
  carry = carry_next;
  for (unsigned int b = 0; b < m; b ++)
    counts[b] &= ~carry;
  actions[chunk] &= ~carry;
}

static inline long int sum_up_class_votes (tsetlin_classifier_t *tm, bool predict)
{
  long int class_sum = 0;
  unsigned int *output = tm->clause_output;
  unsigned int *drop = tm->drop_clause;
  unsigned int clause_chunks = tm->clause_chunks;
  if (predict) {
    for (unsigned int i = 0; i < clause_chunks * 32; i ++) {
      class_sum += (output[i / 32] & 0x55555555) << (31 - (i % 32)) >> 31; // 0101
      class_sum -= (output[i / 32] & 0xaaaaaaaa) << (31 - (i % 32)) >> 31; // 1010
    }
  } else {
    for (unsigned int i = 0; i < clause_chunks * 32; i ++) {
      class_sum += (output[i / 32] & drop[i / 32] & 0x55555555) << (31 - (i % 32)) >> 31; // 0101
      class_sum -= (output[i / 32] & drop[i / 32] & 0xaaaaaaaa) << (31 - (i % 32)) >> 31; // 1010
    }
  }
  long int threshold = tm->threshold;
  class_sum = (class_sum > threshold) ? threshold : class_sum;
  class_sum = (class_sum < -threshold) ? -threshold : class_sum;
  return class_sum;
}

static inline void tm_calculate_clause_output (tsetlin_classifier_t *tm, unsigned int class, unsigned int *Xi, bool predict)
{
  unsigned int clauses = tm->clauses;
  unsigned int filter = tm->filter;
  unsigned int *clause_output = tm->clause_output;
  for (unsigned int j = 0; j < clauses; j ++) {
    unsigned int output = 0;
    unsigned int all_exclude = 0;
    unsigned int input_chunks = tm->input_chunks;
    unsigned int clause_chunk = j / (sizeof(unsigned int) * CHAR_BIT);
    unsigned int clause_chunk_pos = j % (sizeof(unsigned int) * CHAR_BIT);
    unsigned int *actions = tm_state_idx_actions(tm, class, j);
    for (unsigned int k = 0; k < input_chunks - 1; k ++) {
      output |= ((actions[k] & Xi[k]) ^ actions[k]);
      all_exclude |= actions[k];
    }
    output |=
      (actions[input_chunks - 1] & Xi[input_chunks - 1] & filter) ^
      (actions[input_chunks - 1] & filter);
    all_exclude |= ((actions[input_chunks - 1] & filter) ^ 0);
    output = !output && !(predict && !all_exclude);
    if (output)
      clause_output[clause_chunk] |= (1U << clause_chunk_pos);
    else
      clause_output[clause_chunk] &= ~(1U << clause_chunk_pos);
  }
}

static inline void tm_update (tsetlin_classifier_t *tm, unsigned int class, unsigned int *Xi, unsigned int target, double specificity)
{
  tm_calculate_clause_output(tm, class, Xi, false);
  long int tgt = target ? 1 : 0;
  long int class_sum = sum_up_class_votes(tm, false);
  unsigned int input_chunks = tm->input_chunks;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int *clause_output = tm->clause_output;
  unsigned int *drop_clause = tm->drop_clause;
  unsigned int *feedback_to_la = tm->feedback_to_la;
  unsigned int *feedback_to_clauses = tm->feedback_to_clauses;
  float p = (1.0 / (tm->threshold * 2)) * (tm->threshold + (1 - 2 * tgt) * class_sum);
  memset(feedback_to_clauses, 0, clause_chunks * sizeof(unsigned int));
  for (unsigned int i = 0; i < clause_chunks; i ++)
    for (unsigned int j = 0; j < sizeof(unsigned int) * CHAR_BIT; j ++)
      feedback_to_clauses[i] |= (unsigned int)
        (((float) fast_rand()) / ((float) UINT32_MAX) <= p) << j;
  for (unsigned int i = 0; i < clause_chunks; i ++)
    feedback_to_clauses[i] &= drop_clause[i];
  for (unsigned int j = 0; j < tm->clauses; j ++) {
    long int jl = (long int) j;
    unsigned int clause_chunk = j / (sizeof(unsigned int) * CHAR_BIT);
    unsigned int clause_chunk_pos = j % (sizeof(unsigned int) * CHAR_BIT);
    unsigned int *actions = tm_state_idx_actions(tm, class, j);
    if (!(feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos)))
      continue;
    if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == -1) {
      // Type II feedback
      if ((clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0)
        for (unsigned int k = 0; k < input_chunks; k ++)
          tm_inc(tm, class, j, k, (~Xi[k]) & (~actions[k]));
    } else if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == 1) {
      // Type I Feedback
      tm_initialize_random_streams(tm, specificity);
      if ((clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
        if (tm->boost_true_positive)
          for (unsigned int k = 0; k < input_chunks; k ++)
            tm_inc(tm, class, j, k, Xi[k]);
        else
          for (unsigned int k = 0; k < input_chunks; k ++)
            tm_inc(tm, class, j, k, Xi[k] & (~feedback_to_la[k]));
        for (unsigned int k = 0; k < input_chunks; k ++)
          tm_dec(tm, class, j, k, (~Xi[k]) & feedback_to_la[k]);
      } else {
        for (unsigned int k = 0; k < input_chunks; k ++)
          tm_dec(tm, class, j, k, feedback_to_la[k]);
      }
    }
  }
}

static inline int tm_score (tsetlin_classifier_t *tm, unsigned int class, unsigned int *Xi) {
  tm_calculate_clause_output(tm, class, Xi, true);
  int r = sum_up_class_votes(tm, true);
  return r;
}

static inline unsigned int mc_tm_predict (tsetlin_classifier_t *tm, unsigned int *X)
{
  unsigned int m = tm->classes;
  unsigned int max_class = 0;
  long int max_class_sum = tm_score(tm, 0, X);
  for (long int i = 1; i < m; i ++) {
    long int class_sum = tm_score(tm, i, X);
    if (max_class_sum < class_sum) {
      max_class_sum = class_sum;
      max_class = i;
    }
  }
  return max_class;
}

static inline void mc_tm_update (tsetlin_classifier_t *tm, unsigned int *Xi, unsigned int target_class, double specificity)
{
  tm_update(tm, target_class, Xi, 1, specificity);
  unsigned int negative_target_class = (unsigned int) fast_rand() % tm->classes;
  while (negative_target_class == target_class)
    negative_target_class = (unsigned int) fast_rand() % tm->classes;
  tm_update(tm, negative_target_class, Xi, 0, specificity);
}

static inline void ae_tm_decode (tsetlin_autoencoder_t *tm)
{
  unsigned int decoder_classes = tm->decoder.classes;
  unsigned int *encoding = tm->encoding;
  unsigned int *decoding = tm->decoding;

  for (unsigned int i = 0; i < decoder_classes; i ++)
  {
    unsigned int chunk = i / (sizeof(unsigned int) * CHAR_BIT);
    unsigned int pos = i % (sizeof(unsigned int) * CHAR_BIT);

    if (tm_score(&tm->decoder, i, encoding) > 0)
      decoding[chunk] |= (1U << pos);
    else
      decoding[chunk] &= ~(1U << pos);
  }

  decoding[decoder_classes / (sizeof(unsigned int) * CHAR_BIT)] &= tm->encoder.filter;
}

static inline void ae_tm_encode (tsetlin_autoencoder_t *tm, unsigned int *input)
{
  unsigned int encoder_classes = tm->encoder.classes;
  unsigned int *encoding = tm->encoding;

  for (unsigned int i = 0; i < encoder_classes; i ++)
  {
    unsigned int chunk = i / (sizeof(unsigned int) * CHAR_BIT);
    unsigned int pos = i % (sizeof(unsigned int) * CHAR_BIT);

    if (tm_score(&tm->encoder, i, input) > 0)
      encoding[chunk] |= (1U << pos);
    else
      encoding[chunk] &= ~(1U << pos);
  }

  encoding[encoder_classes / (sizeof(unsigned int) * CHAR_BIT)] &= tm->decoder.filter;
}

static inline void en_tm_encode (tsetlin_encoder_t *tm, unsigned int *input, unsigned int *encoding)
{
  unsigned int encoder_classes = tm->encoder.classes;

  for (unsigned int i = 0; i < encoder_classes; i ++)
  {
    unsigned int chunk = i / (sizeof(unsigned int) * CHAR_BIT);
    unsigned int pos = i % (sizeof(unsigned int) * CHAR_BIT);

    if (tm_score(&tm->encoder, i, input) > 0)
      encoding[chunk] |= (1U << pos);
    else
      encoding[chunk] &= ~(1U << pos);
  }

  encoding[encoder_classes / (sizeof(unsigned int) * CHAR_BIT)] &= tm->encoding_filter;
}

static inline void ae_tm_update (tsetlin_autoencoder_t *tm, unsigned int *input, double specificity)
{
  ae_tm_encode(tm, input);
  unsigned int decoder_classes = tm->decoder.classes;
  unsigned int encoder_classes = tm->encoder.classes;
  unsigned int *encoding = tm->encoding;
  tsetlin_classifier_t *decoder = &tm->decoder;
  tsetlin_classifier_t *encoder = &tm->encoder;

  for (unsigned int i = 0; i < decoder_classes; i ++)
  {
    unsigned int chunk = i / (sizeof(unsigned int) * CHAR_BIT);
    unsigned int pos = i % (sizeof(unsigned int) * CHAR_BIT);

    unsigned int expected = input[chunk] & (1U << pos);
    tm_update(decoder, i, encoding, expected, specificity);

    // TODO: Is this right? What should the "expected" class be for the
    // encoding? This just uses the expected output.
    // TODO: Should we re-use the existing calculated clause output? This
    // currently re-encodes the input.
    for (unsigned int j = 0; j < encoder_classes; j ++)
      tm_update(encoder, j, input, expected, specificity);
  }
}

static inline void en_tm_update (tsetlin_encoder_t *tm, unsigned int *a, unsigned int *b, double similarity, double specificity, double update_pm)
{
  tsetlin_classifier_t *encoder = &tm->encoder;
  unsigned int classes = encoder->classes;
  unsigned int encoding_chunks = tm->encoding_chunks;

  unsigned int *encoding_a = tm->encoding_a;
  unsigned int *encoding_b = tm->encoding_b;

  en_tm_encode(tm, a, encoding_a);
  en_tm_encode(tm, b, encoding_b);

  // double similarity0 = hamming(encoding_a, encoding_b, encoding_chunks) / (double) tm->encoding_bits;
  double similarity0 = jaccard(encoding_a, encoding_b, encoding_chunks);
  double update_p = fabs(similarity - similarity0) * update_pm;

  if (similarity0 < similarity) {

    for (unsigned int i = 0; i < classes; i ++) {
      unsigned int chunk = i / (sizeof(unsigned int) * CHAR_BIT);
      unsigned int pos = i % (sizeof(unsigned int) * CHAR_BIT);
      unsigned int bit_a = encoding_a[chunk] & (1 << pos);
      unsigned int bit_b = encoding_b[chunk] & (1 << pos);
      if (!bit_a && !bit_b && (((float) fast_rand()) / ((float) UINT32_MAX) <= update_p)) {
        tm_update(encoder, i, a, 1, specificity);
        tm_update(encoder, i, b, 1, specificity);
      }
    }

  } else if (similarity0 > similarity) {

    for (unsigned int i = 0; i < classes; i ++) {
      unsigned int chunk = i / (sizeof(unsigned int) * CHAR_BIT);
      unsigned int pos = i % (sizeof(unsigned int) * CHAR_BIT);
      unsigned int bit_a = encoding_a[chunk] & (1 << pos);
      unsigned int bit_b = encoding_b[chunk] & (1 << pos);
      if (bit_a && bit_b && (((float) fast_rand()) / ((float) UINT32_MAX) <= update_p)) {
        tm_update(encoder, i, a, 0, specificity);
        tm_update(encoder, i, b, 0, specificity);
      }
    }

  }
}

static inline void mc_tm_initialize_drop_clause (tsetlin_classifier_t *tm, double drop_clause)
{
  memset(tm->drop_clause, 0, sizeof(unsigned int) * tm->clause_chunks);
  for (unsigned int i = 0; i < tm->clause_chunks; i ++)
    for (unsigned int j = 0; j < sizeof(unsigned int) * CHAR_BIT; j ++)
      if (((float)fast_rand())/((float)UINT32_MAX) <= drop_clause)
        tm->drop_clause[i] |= (1 << j);
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
  tm->input_chunks = (2 * tm->features - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  tm->clause_chunks = (tm->clauses - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  tm->filter = (tm->features * 2) % (sizeof(unsigned int) * CHAR_BIT) != 0
    ? ~(((unsigned int) ~0) << ((tm->features * 2) % (sizeof(unsigned int) * CHAR_BIT)))
    : (unsigned int) ~0;
  tm->drop_clause = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->state = malloc(sizeof(unsigned int) * tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks);
  tm->actions = malloc(sizeof(unsigned int) * tm->classes * tm->clauses * tm->input_chunks);
  tm->clause_output = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->feedback_to_la = malloc(sizeof(unsigned int) * tm->input_chunks);
  tm->feedback_to_clauses = malloc(sizeof(unsigned int) * tm->clause_chunks);
  if (!(tm->drop_clause && tm->state && tm->clause_output && tm->feedback_to_la && tm->feedback_to_clauses))
    luaL_error(L, "error in malloc during creation of classifier");
  for (unsigned int i = 0; i < tm->classes; i ++)
    for (unsigned int j = 0; j < tm->clauses; j ++)
      for (unsigned int k = 0; k < tm->input_chunks; k ++) {
        unsigned int m = tm->state_bits - 1;
        unsigned int *actions = tm_state_idx_actions(tm, i, j);
        unsigned int *counts = tm_state_idx_counts(tm, i, j, k);
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

static inline int tk_tsetlin_create_autoencoder (lua_State *L, tsetlin_autoencoder_t *tm)
{
  unsigned int encoding_bits = tk_tsetlin_checkunsigned(L, 1);
  unsigned int features = tk_tsetlin_checkunsigned(L, 2);
  unsigned int clauses = tk_tsetlin_checkunsigned(L, 3);
  unsigned int state_bits = tk_tsetlin_checkunsigned(L, 4);
  unsigned int threshold = tk_tsetlin_checkunsigned(L, 5);
  bool boost_true_positive = tk_tsetlin_checkboolean(L, 6);
  tk_tsetlin_init_classifier(L, &tm->encoder,
      encoding_bits, features, clauses, state_bits, threshold, boost_true_positive);
  tk_tsetlin_init_classifier(L, &tm->decoder,
      features, encoding_bits, clauses, state_bits, threshold, boost_true_positive);
  tm->encoding = malloc(sizeof(unsigned int) * tm->decoder.input_chunks);
  tm->decoding = malloc(sizeof(unsigned int) * tm->encoder.input_chunks);
  if (!tm->encoding)
    luaL_error(L, "error in malloc during creation of autoencoder");
  if (!tm->decoding)
    luaL_error(L, "error in malloc during creation of autoencoder");
  return 0;
}

static inline int tk_tsetlin_create_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  unsigned int encoding_bits = tk_tsetlin_checkunsigned(L, 1);
  unsigned int features = tk_tsetlin_checkunsigned(L, 2);
  unsigned int clauses = tk_tsetlin_checkunsigned(L, 3);
  unsigned int state_bits = tk_tsetlin_checkunsigned(L, 4);
  unsigned int threshold = tk_tsetlin_checkunsigned(L, 5);
  bool boost_true_positive = tk_tsetlin_checkboolean(L, 6);
  tk_tsetlin_init_classifier(L, &tm->encoder,
      encoding_bits, features, clauses, state_bits, threshold, boost_true_positive);
  tm->encoding_bits = encoding_bits;
  tm->encoding_chunks = (encoding_bits - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  tm->encoding_a = malloc(tm->encoding_chunks * sizeof(unsigned int) * CHAR_BIT);
  tm->encoding_b = malloc(tm->encoding_chunks * sizeof(unsigned int) * CHAR_BIT);
  tm->encoding_filter = encoding_bits % (sizeof(unsigned int) * CHAR_BIT) != 0
    ? ~(((unsigned int) ~0) << (encoding_bits % (sizeof(unsigned int) * CHAR_BIT)))
    : (unsigned int) ~0;
  if (!(tm->encoding_a && tm->encoding_b))
    luaL_error(L, "error in malloc during creation of encoder");
  return 0;
}

static inline int tk_tsetlin_create_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: create regressor");
  return 0;
}

static inline int tk_tsetlin_create (lua_State *L)
{
  const char *type = luaL_checkstring(L, 1);
  if (!strcmp(type, "classifier")) {

    lua_remove(L, 1);
    tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t));
    if (!tm) goto err_mem;
    luaL_getmetatable(L, TK_TSETLIN_MT);
    lua_setmetatable(L, -2);
    tm->type = TM_CLASSIFIER;
    tm->classifier = malloc(sizeof(tsetlin_classifier_t));
    if (!tm->classifier) goto err_mem;
    tk_tsetlin_create_classifier(L, tm->classifier);
    return 1;

  } else if (!strcmp(type, "autoencoder")) {

    lua_remove(L, 1);
    tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t));
    if (!tm) goto err_mem;
    luaL_getmetatable(L, TK_TSETLIN_MT);
    lua_setmetatable(L, -2);
    tm->type = TM_AUTOENCODER;
    tm->autoencoder = malloc(sizeof(tsetlin_autoencoder_t));
    if (!tm->autoencoder) goto err_mem;
    tk_tsetlin_create_autoencoder(L, tm->autoencoder);
    return 1;

  } else if (!strcmp(type, "encoder")) {

    lua_remove(L, 1);
    tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t));
    if (!tm) goto err_mem;
    luaL_getmetatable(L, TK_TSETLIN_MT);
    lua_setmetatable(L, -2);
    tm->type = TM_ENCODER;
    tm->encoder = malloc(sizeof(tsetlin_encoder_t));
    if (!tm->encoder) goto err_mem;
    tk_tsetlin_create_encoder(L, tm->encoder);
    return 1;

  } else if (!strcmp(type, "regressor")) {

    lua_remove(L, 1);
    tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t));
    if (!tm) goto err_mem;
    luaL_getmetatable(L, TK_TSETLIN_MT);
    lua_setmetatable(L, -2);
    tm->type = TM_REGRESSOR;
    tm->regressor = malloc(sizeof(tsetlin_regressor_t));
    if (!tm->regressor) goto err_mem;
    tk_tsetlin_create_regressor(L, tm->regressor);
    return 1;

  } else {

    luaL_error(L, "unexpected tsetlin machine type in create");
    return 0;

  }

err_mem:
  return luaL_error(L, "error in malloc during creation");
}

static inline void tk_tsetlin_destroy_classifier (tsetlin_classifier_t *tm)
{
  if (tm == NULL)
    return;
  free(tm->state);
  free(tm->actions);
  free(tm->clause_output);
  free(tm->drop_clause);
  free(tm->feedback_to_la);
  free(tm->feedback_to_clauses);
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tsetlin_t *tm = (tsetlin_t *) tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      tk_tsetlin_destroy_classifier(tm->classifier);
      free(tm->classifier);
      break;
    case TM_AUTOENCODER:
      tk_tsetlin_destroy_classifier(&tm->autoencoder->encoder);
      tk_tsetlin_destroy_classifier(&tm->autoencoder->decoder);
      free(tm->autoencoder->encoding);
      free(tm->autoencoder->decoding);
      free(tm->autoencoder);
      break;
    case TM_ENCODER:
      tk_tsetlin_destroy_classifier(&tm->encoder->encoder);
      free(tm->encoder->encoding_a);
      free(tm->encoder->encoding_b);
      free(tm->encoder);
      break;
    case TM_REGRESSOR:
      tk_tsetlin_destroy_classifier(&tm->regressor->classifier);
      free(tm->regressor);
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
  unsigned int class = mc_tm_predict(tm, bm);
  lua_pushinteger(L, class);
  return 1;
}

static inline int tk_tsetlin_predict_autoencoder (lua_State *L, tsetlin_autoencoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  ae_tm_encode(tm, bm);
  lua_pushlstring(L, (char *) tm->encoding, sizeof(unsigned int) * tm->decoder.input_chunks);
  return 1;
}

static inline int tk_tsetlin_predict_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  en_tm_encode(tm, bm, tm->encoding_a);
  lua_pushlstring(L, (char *) tm->encoding_a, sizeof(unsigned int) * tm->encoder.input_chunks);
  return 1;
}

static inline int tk_tsetlin_predict_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: predict regressor");
  return 0;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  tsetlin_t *tm = (tsetlin_t *) tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_predict_classifier(L, tm->classifier);
    case TM_AUTOENCODER:
      return tk_tsetlin_predict_autoencoder(L, tm->autoencoder);
    case TM_ENCODER:
      return tk_tsetlin_predict_encoder(L, tm->encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_predict_regressor(L, tm->regressor);
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
  double specificity = luaL_checknumber(L, 4);
  double drop_clause = luaL_optnumber(L, 5, 1);
  mc_tm_initialize_drop_clause(tm, drop_clause);
  mc_tm_update(tm, bm, tgt, specificity);
  return 0;
}

static inline int tk_tsetlin_update_autoencoder (lua_State *L, tsetlin_autoencoder_t *tm)
{
  lua_settop(L, 4);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  double specificity = luaL_checknumber(L, 3);
  double drop_clause = luaL_optnumber(L, 4, 1);
  mc_tm_initialize_drop_clause(&tm->encoder, drop_clause);
  mc_tm_initialize_drop_clause(&tm->decoder, drop_clause);
  ae_tm_update(tm, bm, specificity);
  return 0;
}

static inline int tk_tsetlin_update_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 7);
  unsigned int *a = (unsigned int *) luaL_checkstring(L, 2);
  unsigned int *b = (unsigned int *) luaL_checkstring(L, 3);
  double expected = luaL_checknumber(L, 4);
  double specificity = luaL_checknumber(L, 5);
  double update_pm = luaL_checknumber(L, 6);
  double drop_clause = luaL_optnumber(L, 7, 1);
  mc_tm_initialize_drop_clause(&tm->encoder, drop_clause);
  en_tm_update(tm, a, b, expected, specificity, update_pm);
  return 0;
}

static inline int tk_tsetlin_update_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: update regressor");
  return 0;
}

static inline int tk_tsetlin_update (lua_State *L)
{
  tsetlin_t *tm = (tsetlin_t *) tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_update_classifier(L, tm->classifier);
    case TM_AUTOENCODER:
      return tk_tsetlin_update_autoencoder(L, tm->autoencoder);
    case TM_ENCODER:
      return tk_tsetlin_update_encoder(L, tm->encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_update_regressor(L, tm->regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in update");
  }
  return 0;
}

static inline int tk_tsetlin_train_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  lua_settop(L, 6);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *ss = (unsigned int *) luaL_checkstring(L, 4);
  double specificity = luaL_checknumber(L, 5);
  double drop_clause = luaL_optnumber(L, 6, 1);
  mc_tm_initialize_drop_clause(tm, drop_clause);
  for (unsigned int i = 0; i < n; i ++)
    mc_tm_update(tm, &ps[i * tm->input_chunks], ss[i], specificity);
  return 0;
}

static inline int tk_tsetlin_train_autoencoder (lua_State *L, tsetlin_autoencoder_t *tm)
{
  lua_settop(L, 5);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  double specificity = luaL_checknumber(L, 4);
  double drop_clause = luaL_optnumber(L, 5, 1);
  // TODO: Should the drop clause be shared? Does that make more sense for an
  // autoencoder?
  mc_tm_initialize_drop_clause(&tm->encoder, drop_clause);
  mc_tm_initialize_drop_clause(&tm->decoder, drop_clause);
  for (unsigned int i = 0; i < n; i ++)
    ae_tm_update(tm, &ps[i * tm->encoder.input_chunks], specificity);
  return 0;
}

static inline int tk_tsetlin_train_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 8);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *as = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *bs = (unsigned int *) luaL_checkstring(L, 4);
  double *expected = (double *) luaL_checkstring(L, 5);
  double specificity = luaL_checknumber(L, 6);
  double update_pm = luaL_checknumber(L, 7);
  double drop_clause = luaL_optnumber(L, 8, 1);
  unsigned int input_chunks = tm->encoder.input_chunks;
  mc_tm_initialize_drop_clause(&tm->encoder, drop_clause);
  for (unsigned int i = 0; i < n; i ++) {
    unsigned int *a = &as[i * input_chunks];
    unsigned int *b = &bs[i * input_chunks];
    en_tm_update(tm, a, b, expected[i], specificity, update_pm);
  }
  return 0;
}

static inline int tk_tsetlin_train_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: train regressor");
  return 0;
}

static inline int tk_tsetlin_train (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_train_classifier(L, tm->classifier);
    case TM_AUTOENCODER:
      return tk_tsetlin_train_autoencoder(L, tm->autoencoder);
    case TM_ENCODER:
      return tk_tsetlin_train_encoder(L, tm->encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_train_regressor(L, tm->regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in train");
  }
}

static inline int tk_tsetlin_evaluate_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  lua_settop(L, 5);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *ss = (unsigned int *) luaL_checkstring(L, 4);
  bool track_stats = tk_tsetlin_checkboolean(L, 5);
  unsigned int correct = 0;
  unsigned int *confusion;
  unsigned int *predictions;
  unsigned int *observations;
  if (track_stats) {
    confusion = malloc(sizeof(unsigned int) * tm->classes * tm->classes);
    predictions = malloc(sizeof(unsigned int) * tm->classes);
    observations = malloc(sizeof(unsigned int) * tm->classes);
    if (!(confusion && predictions && observations))
      luaL_error(L, "error in malloc during evaluation");
    memset(confusion, 0, sizeof(unsigned int) * tm->classes * tm->classes);
    memset(predictions, 0, sizeof(unsigned int) * tm->classes);
    memset(observations, 0, sizeof(unsigned int) * tm->classes);
  }
  for (unsigned int i = 0; i < n; i ++) {
    unsigned int expected = ss[i];
    unsigned int predicted = mc_tm_predict(tm, &ps[i * tm->input_chunks]);
    if (expected == predicted)
      correct ++;
    if (track_stats) {
      observations[expected] ++;
      predictions[predicted] ++;
      if (expected != predicted)
        confusion[expected * tm->classes + predicted] ++;
    }
  }
  lua_pushnumber(L, correct);
  if (track_stats) {
    lua_newtable(L); // ct
    for (unsigned int i = 0; i < tm->classes; i ++) {
      for (unsigned int j = 0; j < tm->classes; j ++) {
        unsigned int c = confusion[i * tm->classes + j];
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
    for (unsigned int i = 0; i < tm->classes; i ++) {
      lua_pushinteger(L, i); // pt i
      lua_pushinteger(L, predictions[i]); // pt i p
      lua_settable(L, -3); // pt
    }
    lua_newtable(L); // ot
    for (unsigned int i = 0; i < tm->classes; i ++) {
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

static inline int tk_tsetlin_evaluate_autoencoder (lua_State *L, tsetlin_autoencoder_t *tm)
{
  lua_settop(L, 3);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  tsetlin_classifier_t *encoder = &tm->encoder;
  unsigned int *decoding = tm->decoding;
  unsigned int input_chunks = encoder->input_chunks;
  unsigned int features = encoder->features;

  long unsigned int total_bits = n * features * 2 * CHAR_BIT;
  long unsigned int total_diff = 0;

  for (unsigned int i = 0; i < n; i ++)
  {
    unsigned int *input = &ps[i * input_chunks];
    ae_tm_encode(tm, input);
    ae_tm_decode(tm);
    total_diff += hamming(input, decoding, input_chunks);
  }

  lua_pushnumber(L, (double) (total_bits - total_diff) / total_bits);
  return 1;
}

static inline int tk_tsetlin_evaluate_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 5);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *as = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *bs = (unsigned int *) luaL_checkstring(L, 4);
  double *scores = (double *) luaL_checkstring(L, 5);
  tsetlin_classifier_t *encoder = &tm->encoder;
  unsigned int encoding_chunks = tm->encoding_chunks;
  unsigned int input_chunks = encoder->input_chunks;

  unsigned int n_high = 0;
  unsigned int n_low = 0;

  double total_diff = 0;

  unsigned int *encoding_a = tm->encoding_a;
  unsigned int *encoding_b = tm->encoding_b;

  for (unsigned int i = 0; i < n; i ++)
  {
    unsigned int *a = &as[i * input_chunks];
    en_tm_encode(tm, a, encoding_a);

    unsigned int *b = &bs[i * input_chunks];
    en_tm_encode(tm, b, encoding_b);

    double similarity = scores[i];
    double similarity0 = jaccard(encoding_a, encoding_b, encoding_chunks);

    total_diff += fabs(similarity - similarity0);

    if (similarity0 > similarity)
      n_high ++;
    else if (similarity0 < similarity)
      n_low ++;
  }

  lua_pushnumber(L, (double) (n - total_diff) / n);
  lua_pushinteger(L, n_high);
  lua_pushinteger(L, n_low);
  return 3;
}

static inline int tk_tsetlin_evaluate_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: evaluate regressor");
  return 0;
}

static inline int tk_tsetlin_evaluate (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_evaluate_classifier(L, tm->classifier);
    case TM_AUTOENCODER:
      return tk_tsetlin_evaluate_autoencoder(L, tm->autoencoder);
    case TM_ENCODER:
      return tk_tsetlin_evaluate_encoder(L, tm->encoder);
    case TM_REGRESSOR:
      return tk_tsetlin_evaluate_regressor(L, tm->regressor);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in evaluate");
  }
}

static inline int tk_tsetlin_type (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      lua_pushstring(L, "classifier");
      break;
    case TM_AUTOENCODER:
      lua_pushstring(L, "autoencoder");
      break;
    case TM_ENCODER:
      lua_pushstring(L, "encoder");
      break;
    case TM_REGRESSOR:
      lua_pushstring(L, "regressor");
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
  { "type", tk_tsetlin_type },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_tsetlin_register(L, tk_tsetlin_fns, 0); // t
  luaL_newmetatable(L, TK_TSETLIN_MT); // t mt
  lua_pushcfunction(L, tk_tsetlin_destroy); // t mt fn
  lua_setfield(L, -2, "__gc"); // t mt
  lua_pop(L, 1); // t
  return 1;
}
