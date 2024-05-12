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
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

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
  unsigned int *clause_output;
  unsigned int *feedback_to_la;
  unsigned int *feedback_to_clauses;
  unsigned int *drop_clause;
} tsetlin_classifier_t;

typedef struct {
  tsetlin_classifier_t classifier;
} tsetlin_recurrent_classifier_t;

typedef struct {
  unsigned int encoding_bits;
  unsigned int encoding_chunks;
  unsigned int encoding_filter;
  unsigned int *encoding_a;
  unsigned int *encoding_n;
  unsigned int *encoding_p;
  tsetlin_classifier_t encoder;
} tsetlin_encoder_t;

typedef struct {
  unsigned int *input_a;
  unsigned int *input_n;
  unsigned int *input_p;
  unsigned int *state;
  unsigned int token_bits;
  unsigned int token_chunks;
  tsetlin_encoder_t encoder;
} tsetlin_recurrent_encoder_t;

typedef struct {
  unsigned int *encoding;
  unsigned int *decoding;
  tsetlin_classifier_t encoder;
  tsetlin_classifier_t decoder;
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

static inline unsigned int hamming (unsigned int *a, unsigned int *b, unsigned int chunks) {
  unsigned int distance = 0;
  for (unsigned int i = 0; i < chunks; i ++) {
    unsigned int diff = a[i] ^ b[i];
    while (diff) {
      distance += diff & 1;
      diff >>= 1;
    }
  }
  return distance;
}

static inline void flip_bits (unsigned int *a, unsigned int n) {
  for (unsigned int i = 0; i < n; i ++)
    a[i] = ~a[i];
}

static inline double triplet_loss (unsigned int *a, unsigned int *n, unsigned int *p, unsigned int bits, unsigned int chunks, double margin)
{
  double dist_an = (double) hamming(a, n, chunks) / bits;
  double dist_ap = (double) hamming(a, p, chunks) / bits;
  return fmax(0.0, dist_ap - dist_an + margin);
}

static inline void tk_lua_callmod (lua_State *L, int nargs, int nret, const char *smod, const char *sfn)
{
  lua_getglobal(L, "require"); // arg req
  lua_pushstring(L, smod); // arg req smod
  lua_call(L, 1, 1); // arg mod
  lua_pushstring(L, sfn); // args mod sfn
  lua_gettable(L, -2); // args mod fn
  lua_remove(L, -2); // args fn
  lua_insert(L, - nargs - 1); // fn args
  lua_call(L, nargs, nret); // results
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

static inline unsigned int mc_tm_predict (tsetlin_classifier_t *tm, unsigned int *X, long int *scores)
{
  unsigned int m = tm->classes;
  unsigned int max_class = 0;
  long int max_class_sum = tm_score(tm, 0, X);
  if (scores)
    scores[0] = max_class_sum;
  for (long int i = 1; i < m; i ++) {
    long int class_sum = tm_score(tm, i, X);
    if (scores)
      scores[i] = class_sum;
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

static inline void ae_tm_decode (tsetlin_auto_encoder_t *tm)
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

static inline void ae_tm_encode (tsetlin_auto_encoder_t *tm, unsigned int *input)
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

static inline void re_tm_encode (lua_State *L, tsetlin_recurrent_encoder_t *tm, int i_xs, unsigned int *input, unsigned int *xo)
{
  lua_pushvalue(L, i_xs); // t
  memset(tm->state, 0, tm->encoder.encoding_chunks * sizeof(unsigned int));
  unsigned int off_token0 = 0;
  unsigned int off_token1 = off_token0 + tm->token_chunks;
  unsigned int off_state0 = off_token1 + tm->token_chunks;
  unsigned int off_state1 = off_state0 + tm->encoder.encoding_chunks;
  unsigned int e = lua_objlen(L, -1);
  for (unsigned int i = 1; i <= e; i ++) {
    lua_pushinteger(L, i); // t i
    lua_gettable(L, -2); // t v
    unsigned int *x = (unsigned int *) luaL_checkstring(L, -1);
    lua_pop(L, 1); //
    memcpy(input + off_token0, x, tm->token_chunks * sizeof(unsigned int));
    memcpy(input + off_token1, x, tm->token_chunks * sizeof(unsigned int));
    flip_bits(input + off_token1, tm->token_chunks);
    memcpy(input + off_state0, tm->state, tm->encoder.encoding_chunks * sizeof(unsigned int));
    memcpy(input + off_state1, tm->state, tm->encoder.encoding_chunks * sizeof(unsigned int));
    flip_bits(input + off_state1, tm->encoder.encoding_chunks);
    en_tm_encode(&tm->encoder, input,
      i == e ? xo : tm->state);
  }
  lua_pop(L, 1); //
}

static inline void ae_tm_update (tsetlin_auto_encoder_t *tm, unsigned int *input, double specificity)
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

static inline void en_tm_update (tsetlin_encoder_t *tm, unsigned int *a, unsigned int *n, unsigned int *p, double specificity, double margin, double loss_scale)
{
  tsetlin_classifier_t *encoder = &tm->encoder;
  unsigned int classes = encoder->classes;
  unsigned int encoding_bits = tm->encoding_bits;
  unsigned int encoding_chunks = tm->encoding_chunks;

  unsigned int *encoding_a = tm->encoding_a;
  unsigned int *encoding_n = tm->encoding_n;
  unsigned int *encoding_p = tm->encoding_p;

  en_tm_encode(tm, a, encoding_a);
  en_tm_encode(tm, n, encoding_n);
  en_tm_encode(tm, p, encoding_p);

  double loss = triplet_loss(encoding_a, encoding_n, encoding_p, encoding_bits, encoding_chunks, margin);
  double loss_p = pow(loss, loss_scale);

  if (loss > 0) {
    for (unsigned int i = 0; i < classes; i ++) {
      unsigned int chunk = i / (sizeof(unsigned int) * CHAR_BIT);
      unsigned int pos = i % (sizeof(unsigned int) * CHAR_BIT);
      unsigned int bit_n = encoding_n[chunk] & (1 << pos);
      unsigned int bit_p = encoding_p[chunk] & (1 << pos);
      if (((float) fast_rand()) / ((float) UINT32_MAX) < loss_p) {
        if ((!bit_n && !bit_p) || (bit_n && bit_p)) {
          tm_update(encoder, i, n, !bit_n, specificity);
          tm_update(encoder, i, p, !bit_p, specificity);
        } else {
          tm_update(encoder, i, n, bit_n, specificity);
          tm_update(encoder, i, p, bit_p, specificity);
        }
      }
    }
  }
}

static inline void re_tm_update (lua_State *L, tsetlin_recurrent_encoder_t *tm, int i_as, int i_ns, int i_ps, double specificity, double margin, double loss_scale)
{
  tsetlin_classifier_t *encoder = &tm->encoder.encoder;
  unsigned int classes = encoder->classes;
  unsigned int encoding_bits = tm->encoder.encoding_bits;
  unsigned int encoding_chunks = tm->encoder.encoding_chunks;

  unsigned int *encoding_a = tm->encoder.encoding_a;
  unsigned int *encoding_n = tm->encoder.encoding_n;
  unsigned int *encoding_p = tm->encoder.encoding_p;

  re_tm_encode(L, tm, i_as, tm->input_a, encoding_a);
  re_tm_encode(L, tm, i_ns, tm->input_n, encoding_n);
  re_tm_encode(L, tm, i_ps, tm->input_p, encoding_p);

  double loss = triplet_loss(encoding_a, encoding_n, encoding_p, encoding_bits, encoding_chunks, margin);
  double loss_p = pow(loss, loss_scale);

  if (loss > 0) {
    for (unsigned int i = 0; i < classes; i ++) {
      unsigned int chunk = i / (sizeof(unsigned int) * CHAR_BIT);
      unsigned int pos = i % (sizeof(unsigned int) * CHAR_BIT);
      unsigned int bit_n = encoding_n[chunk] & (1 << pos);
      unsigned int bit_p = encoding_p[chunk] & (1 << pos);
      if (((float) fast_rand()) / ((float) UINT32_MAX) < loss_p) {
        if ((!bit_n && !bit_p) || (bit_n && bit_p)) {
          tm_update(encoder, i, tm->input_n, !bit_n, specificity);
          tm_update(encoder, i, tm->input_p, !bit_p, specificity);
        } else {
          tm_update(encoder, i, tm->input_n, bit_n, specificity);
          tm_update(encoder, i, tm->input_p, bit_p, specificity);
        }
      }
    }
  }
}

static inline void mc_tm_initialize_drop_clause (tsetlin_classifier_t *tm, double drop_clause)
{
  memset(tm->drop_clause, 0, tm->clause_chunks * sizeof(unsigned int));
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
  tm->input_bits = 2 * tm->features;
  tm->input_chunks = (tm->input_bits - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  tm->clause_chunks = (tm->clauses - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  tm->state_chunks = tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tm->action_chunks = tm->classes * tm->clauses * tm->input_chunks;
  tm->filter = tm->input_bits % (sizeof(unsigned int) * CHAR_BIT) != 0
    ? ~(((unsigned int) ~0) << (tm->input_bits % (sizeof(unsigned int) * CHAR_BIT)))
    : (unsigned int) ~0;
  tm->state = malloc(sizeof(unsigned int) * tm->state_chunks);
  tm->actions = malloc(sizeof(unsigned int) * tm->action_chunks);
  tm->clause_output = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->feedback_to_la = malloc(sizeof(unsigned int) * tm->input_chunks);
  tm->feedback_to_clauses = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->drop_clause = malloc(sizeof(unsigned int) * tm->clause_chunks);
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

static inline int tk_tsetlin_create_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *)
{
  // TODO
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
  tm->encoding_chunks = (encoding_bits - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  tm->encoding_a = malloc(tm->encoding_chunks * sizeof(unsigned int) * CHAR_BIT);
  tm->encoding_n = malloc(tm->encoding_chunks * sizeof(unsigned int) * CHAR_BIT);
  tm->encoding_p = malloc(tm->encoding_chunks * sizeof(unsigned int) * CHAR_BIT);
  tm->encoding_filter = encoding_bits % (sizeof(unsigned int) * CHAR_BIT) != 0
    ? ~(((unsigned int) ~0) << (encoding_bits % (sizeof(unsigned int) * CHAR_BIT)))
    : (unsigned int) ~0;
  if (!(tm->encoding_a && tm->encoding_n && tm->encoding_p))
    luaL_error(L, "error in malloc during creation of encoder");
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
    (token_bits + output_bits) * 2,
    clauses,
    state_bits,
    threshold,
    boost_true_positive);
  tm->input_a = malloc(sizeof(unsigned int) * tm->encoder.encoder.input_chunks);
  tm->input_n = malloc(sizeof(unsigned int) * tm->encoder.encoder.input_chunks);
  tm->input_p = malloc(sizeof(unsigned int) * tm->encoder.encoder.input_chunks);
  tm->state = malloc(sizeof(unsigned int) * tm->encoder.encoding_chunks);
  tm->token_bits = token_bits;
  tm->token_chunks = (tm->token_bits - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  if (!(tm->input_a && tm->input_n && tm->input_p && tm->state))
    luaL_error(L, "error in malloc during creation of recurrent encoder");
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
  tk_tsetlin_init_classifier(L, &tm->encoder,
      encoding_bits, features, clauses, state_bits, threshold, boost_true_positive);
  tk_tsetlin_init_classifier(L, &tm->decoder,
      features, encoding_bits, clauses, state_bits, threshold, boost_true_positive);
  tm->encoding = malloc(sizeof(unsigned int) * tm->decoder.input_chunks);
  tm->decoding = malloc(sizeof(unsigned int) * tm->encoder.input_chunks);
  if (!tm->encoding)
    luaL_error(L, "error in malloc during creation of auto_encoder");
  if (!tm->decoding)
    luaL_error(L, "error in malloc during creation of auto_encoder");
  return 0;
}

static inline int tk_tsetlin_create_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: create regressor");
  return 0;
}

static inline int tk_tsetlin_create_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *)
{
  // TODO
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
    case TM_RECURRENT_CLASSIFIER:
      luaL_error(L, "unimplemented: destroy recurrent classifier");
      break;
    case TM_ENCODER:
      tk_tsetlin_destroy_classifier(&tm->encoder->encoder);
      free(tm->encoder->encoding_a);
      free(tm->encoder->encoding_n);
      free(tm->encoder->encoding_p);
      free(tm->encoder);
      break;
    case TM_RECURRENT_ENCODER:
      tk_tsetlin_destroy_classifier(&tm->recurrent_encoder->encoder.encoder);
      free(tm->recurrent_encoder->input_a);
      free(tm->recurrent_encoder->input_n);
      free(tm->recurrent_encoder->input_p);
      free(tm->recurrent_encoder->state);
      free(tm->recurrent_encoder);
      break;
    case TM_AUTO_ENCODER:
      tk_tsetlin_destroy_classifier(&tm->auto_encoder->encoder);
      tk_tsetlin_destroy_classifier(&tm->auto_encoder->decoder);
      free(tm->auto_encoder->encoding);
      free(tm->auto_encoder->decoding);
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
  lua_settop(L, 3);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  bool all_scores = lua_toboolean(L, 3);
  if (!all_scores) {
    lua_pushinteger(L, mc_tm_predict(tm, bm, NULL));
  } else {
    long int scores[tm->classes];
    mc_tm_predict(tm, bm, scores);
    lua_newtable(L);
    for (unsigned int i = 0; i < tm->classes; i ++) {
      lua_pushinteger(L, i + 1);
      lua_pushinteger(L, scores[i]);
      lua_settable(L, -3);
    }
  }
  return 1;
}

static inline int tk_tsetlin_predict_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *)
{
  // TODO
  luaL_error(L, "unimplemented: predict recurrent classifier");
  return 0;
}

static inline int tk_tsetlin_predict_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  en_tm_encode(tm, bm, tm->encoding_a);
  lua_pushlstring(L, (char *) tm->encoding_a, sizeof(unsigned int) * tm->encoder.input_chunks);
  return 1;
}

static inline int tk_tsetlin_predict_recurrent_encoder (lua_State *L, tsetlin_recurrent_encoder_t *tm)
{
  lua_settop(L, 2);
  luaL_checktype(L, 2, LUA_TTABLE);
  re_tm_encode(L, tm, 2, tm->input_a, tm->state);
  lua_pushlstring(L, (char *) tm->state, sizeof(unsigned int) * tm->encoder.encoder.input_chunks);
  return 1;
}

static inline int tk_tsetlin_predict_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 2);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  ae_tm_encode(tm, bm);
  lua_pushlstring(L, (char *) tm->encoding, sizeof(unsigned int) * tm->decoder.input_chunks);
  return 1;
}

static inline int tk_tsetlin_predict_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: predict regressor");
  return 0;
}

static inline int tk_tsetlin_predict_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: predict recurrent regressor");
  return 0;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  tsetlin_t *tm = (tsetlin_t *) tk_tsetlin_peek(L, 1);
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
  double specificity = luaL_checknumber(L, 4);
  double drop_clause = luaL_checknumber(L, 5);
  mc_tm_initialize_drop_clause(tm, drop_clause);
  mc_tm_update(tm, bm, tgt, specificity);
  return 0;
}

static inline int tk_tsetlin_update_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *)
{
  // TODO
  luaL_error(L, "unimplemented: update recurrent classifier");
  return 0;
}

static inline int tk_tsetlin_update_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 8);
  unsigned int *a = (unsigned int *) luaL_checkstring(L, 2);
  unsigned int *n = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *p = (unsigned int *) luaL_checkstring(L, 4);
  double specificity = luaL_checknumber(L, 5);
  double drop_clause = luaL_checknumber(L, 6);
  double margin = luaL_checknumber(L, 7);
  double loss_scale = luaL_checknumber(L, 8);
  mc_tm_initialize_drop_clause(&tm->encoder, drop_clause);
  en_tm_update(tm, a, n, p, specificity, margin, loss_scale);
  return 0;
}

static inline int tk_tsetlin_update_recurrent_encoder (lua_State *L, tsetlin_recurrent_encoder_t *)
{
  // TODO
  luaL_error(L, "unimplemented: update recurrent encoder");
  return 0;
}

static inline int tk_tsetlin_update_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 4);
  unsigned int *bm = (unsigned int *) luaL_checkstring(L, 2);
  double specificity = luaL_checknumber(L, 3);
  double drop_clause = luaL_checknumber(L, 4);
  mc_tm_initialize_drop_clause(&tm->encoder, drop_clause);
  mc_tm_initialize_drop_clause(&tm->decoder, drop_clause);
  ae_tm_update(tm, bm, specificity);
  return 0;
}

static inline int tk_tsetlin_update_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: update regressor");
  return 0;
}

static inline int tk_tsetlin_update_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: update recurrent regressor");
  return 0;
}

static inline int tk_tsetlin_update (lua_State *L)
{
  tsetlin_t *tm = (tsetlin_t *) tk_tsetlin_peek(L, 1);
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

static inline int tk_tsetlin_train_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  lua_settop(L, 6);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *ss = (unsigned int *) luaL_checkstring(L, 4);
  double specificity = luaL_checknumber(L, 5);
  double drop_clause = luaL_checknumber(L, 6);
  mc_tm_initialize_drop_clause(tm, drop_clause);
  for (unsigned int i = 0; i < n; i ++)
    mc_tm_update(tm, &ps[i * tm->input_chunks], ss[i], specificity);
  return 0;
}

static inline int tk_tsetlin_train_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *)
{
  // TODO
  luaL_error(L, "unimplemented: train recurrent classifier");
  return 0;
}

static inline int tk_tsetlin_train_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 9);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *as = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *ns = (unsigned int *) luaL_checkstring(L, 4);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 5);
  double specificity = luaL_checknumber(L, 6);
  double drop_clause = luaL_checknumber(L, 7);
  double margin = luaL_checknumber(L, 8);
  double loss_scale = luaL_checknumber(L, 9);
  unsigned int input_chunks = tm->encoder.input_chunks;
  mc_tm_initialize_drop_clause(&tm->encoder, drop_clause);
  for (unsigned int i = 0; i < n; i ++) {
    unsigned int *a = &as[i * input_chunks];
    unsigned int *n = &ns[i * input_chunks];
    unsigned int *p = &ps[i * input_chunks];
    en_tm_update(tm, a, n, p, specificity, margin, loss_scale);
  }
  return 0;
}

static inline int tk_tsetlin_train_recurrent_encoder (lua_State *L, tsetlin_recurrent_encoder_t *tm)
{
  lua_settop(L, 8);
  int i_aas = 2;
  int i_nns = 3;
  int i_pps = 4;
  luaL_checktype(L, i_aas, LUA_TTABLE);
  luaL_checktype(L, i_nns, LUA_TTABLE);
  luaL_checktype(L, i_pps, LUA_TTABLE);
  double specificity = luaL_checknumber(L, 5);
  double drop_clause = luaL_checknumber(L, 6);
  double margin = luaL_checknumber(L, 7);
  double loss_scale = luaL_checknumber(L, 8);
  mc_tm_initialize_drop_clause(&tm->encoder.encoder, drop_clause);
  for (unsigned int i = 1; i <= lua_objlen(L, i_aas); i ++) {
    lua_pushinteger(L, i); lua_gettable(L, i_aas);
    lua_pushinteger(L, i); lua_gettable(L, i_nns);
    lua_pushinteger(L, i); lua_gettable(L, i_pps);
    re_tm_update(L, tm, -3, -2, -1, specificity, margin, loss_scale);
    lua_pop(L, 3);
  }
  return 0;
}

static inline int tk_tsetlin_train_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 5);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  double specificity = luaL_checknumber(L, 4);
  double drop_clause = luaL_checknumber(L, 5);
  // TODO: Should the drop clause be shared? Does that make more sense for an
  // auto_encoder?
  mc_tm_initialize_drop_clause(&tm->encoder, drop_clause);
  mc_tm_initialize_drop_clause(&tm->decoder, drop_clause);
  for (unsigned int i = 0; i < n; i ++)
    ae_tm_update(tm, &ps[i * tm->encoder.input_chunks], specificity);
  return 0;
}

static inline int tk_tsetlin_train_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: train regressor");
  return 0;
}

static inline int tk_tsetlin_train_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *)
{
  // TODO
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
    unsigned int predicted = mc_tm_predict(tm, &ps[i * tm->input_chunks], NULL);
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

static inline int tk_tsetlin_evaluate_recurrent_classifier (lua_State *L, tsetlin_recurrent_classifier_t *)
{
  // TODO
  luaL_error(L, "unimplemented: evaluate recurrent classifier");
  return 0;
}

static inline int tk_tsetlin_evaluate_encoder (lua_State *L, tsetlin_encoder_t *tm)
{
  lua_settop(L, 6);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *as = (unsigned int *) luaL_checkstring(L, 3);
  unsigned int *ns = (unsigned int *) luaL_checkstring(L, 4);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 5);
  double margin = luaL_checknumber(L, 6);
  tsetlin_classifier_t *encoder = &tm->encoder;
  unsigned int encoding_bits = tm->encoding_bits;
  unsigned int encoding_chunks = tm->encoding_chunks;
  unsigned int input_chunks = encoder->input_chunks;

  unsigned int *encoding_a = tm->encoding_a;
  unsigned int *encoding_n = tm->encoding_n;
  unsigned int *encoding_p = tm->encoding_p;

  unsigned int correct = 0;

  for (unsigned int i = 0; i < n; i ++)
  {
    unsigned int *a = &as[i * input_chunks];
    unsigned int *n = &ns[i * input_chunks];
    unsigned int *p = &ps[i * input_chunks];
    en_tm_encode(tm, a, encoding_a);
    en_tm_encode(tm, n, encoding_n);
    en_tm_encode(tm, p, encoding_p);
    double loss = triplet_loss(encoding_a, encoding_n, encoding_p, encoding_bits, encoding_chunks, margin);
    if (loss == 0)
      correct ++;
  }

  lua_pushnumber(L, (double) correct / n);
  return 1;
}

static inline int tk_tsetlin_evaluate_recurrent_encoder (lua_State *L, tsetlin_recurrent_encoder_t *tm)
{
  lua_settop(L, 5);
  int i_aas = 2;
  int i_nns = 3;
  int i_pps = 4;
  luaL_checktype(L, i_aas, LUA_TTABLE);
  luaL_checktype(L, i_nns, LUA_TTABLE);
  luaL_checktype(L, i_pps, LUA_TTABLE);
  double margin = luaL_checknumber(L, 5);

  unsigned int encoding_bits = tm->encoder.encoding_bits;
  unsigned int encoding_chunks = tm->encoder.encoding_chunks;
  unsigned int *encoding_a = tm->encoder.encoding_a;
  unsigned int *encoding_n = tm->encoder.encoding_n;
  unsigned int *encoding_p = tm->encoder.encoding_p;

  unsigned int correct = 0;
  unsigned int n = lua_objlen(L, i_aas);

  for (unsigned int i = 1; i <= n; i ++)
  {
    lua_pushinteger(L, i); lua_gettable(L, i_aas);
    lua_pushinteger(L, i); lua_gettable(L, i_nns);
    lua_pushinteger(L, i); lua_gettable(L, i_pps);
    re_tm_encode(L, tm, -3, tm->input_a, encoding_a);
    re_tm_encode(L, tm, -2, tm->input_n, encoding_n);
    re_tm_encode(L, tm, -1, tm->input_p, encoding_p);
    lua_pop(L, 3);
    double loss = triplet_loss(encoding_a, encoding_n, encoding_p, encoding_bits, encoding_chunks, margin);
    if (loss == 0)
      correct ++;
  }

  lua_pushnumber(L, (double) correct / n);
  return 1;
}

static inline int tk_tsetlin_evaluate_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm)
{
  lua_settop(L, 3);
  unsigned int n = tk_tsetlin_checkunsigned(L, 2);
  unsigned int *ps = (unsigned int *) luaL_checkstring(L, 3);
  tsetlin_classifier_t *encoder = &tm->encoder;
  unsigned int *decoding = tm->decoding;
  unsigned int input_chunks = encoder->input_chunks;
  unsigned int input_bits = encoder->input_bits;

  unsigned int total_bits = n * input_bits;
  unsigned int total_correct = total_bits;

  for (unsigned int i = 0; i < n; i ++)
  {
    unsigned int *input = &ps[i * input_chunks];
    ae_tm_encode(tm, input);
    ae_tm_decode(tm);
    total_correct -= hamming(input, decoding, input_chunks);
  }

  lua_pushnumber(L, (double) total_correct / (double) total_bits);
  return 1;
}

static inline int tk_tsetlin_evaluate_regressor (lua_State *L, tsetlin_regressor_t *)
{
  // TODO
  luaL_error(L, "unimplemented: evaluate regressor");
  return 0;
}

static inline int tk_tsetlin_evaluate_recurrent_regressor (lua_State *L, tsetlin_recurrent_regressor_t *)
{
  // TODO
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
  (void) fread(data, size, memb, fh);
  if (!ferror(fh)) return;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error reading from file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
}

static inline void _tk_tsetlin_persist_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh)
{
  tk_lua_fwrite(L, &tm->classes, sizeof(tm->classes), 1, fh);
  tk_lua_fwrite(L, &tm->features, sizeof(tm->features), 1, fh);
  tk_lua_fwrite(L, &tm->clauses, sizeof(tm->clauses), 1, fh);
  tk_lua_fwrite(L, &tm->threshold, sizeof(tm->threshold), 1, fh);
  tk_lua_fwrite(L, &tm->state_bits, sizeof(tm->state_bits), 1, fh);
  tk_lua_fwrite(L, &tm->boost_true_positive, sizeof(tm->boost_true_positive), 1, fh);
  tk_lua_fwrite(L, &tm->input_bits, sizeof(tm->input_bits), 1, fh);
  tk_lua_fwrite(L, &tm->input_chunks, sizeof(tm->input_chunks), 1, fh);
  tk_lua_fwrite(L, &tm->clause_chunks, sizeof(tm->clause_chunks), 1, fh);
  tk_lua_fwrite(L, &tm->state_chunks, sizeof(tm->state_chunks), 1, fh);
  tk_lua_fwrite(L, &tm->action_chunks, sizeof(tm->action_chunks), 1, fh);
  tk_lua_fwrite(L, &tm->filter, sizeof(tm->filter), 1, fh);
  tk_lua_fwrite(L, tm->state, sizeof(*tm->state), tm->state_chunks, fh);
  tk_lua_fwrite(L, tm->actions, sizeof(*tm->actions), tm->action_chunks, fh);
  tk_lua_fwrite(L, tm->clause_output, sizeof(*tm->clause_output), tm->clause_chunks, fh);
  tk_lua_fwrite(L, tm->feedback_to_la, sizeof(*tm->feedback_to_la), tm->input_chunks, fh);
  tk_lua_fwrite(L, tm->feedback_to_clauses, sizeof(*tm->feedback_to_clauses), tm->clause_chunks, fh);
  tk_lua_fwrite(L, tm->drop_clause, sizeof(*tm->drop_clause), tm->clause_chunks, fh);
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
  tk_lua_fwrite(L, &tm->encoding_bits, sizeof(tm->encoding_bits), 1, fh);
  tk_lua_fwrite(L, &tm->encoding_chunks, sizeof(tm->encoding_chunks), 1, fh);
  tk_lua_fwrite(L, &tm->encoding_filter, sizeof(tm->encoding_filter), 1, fh);
  _tk_tsetlin_persist_classifier(L, &tm->encoder, fh);
}

static inline void tk_tsetlin_persist_recurrent_encoder (lua_State *L, tsetlin_recurrent_encoder_t *tm, FILE *fh)
{
  tk_lua_fwrite(L, &tm->token_bits, sizeof(tm->token_bits), 1, fh);
  tk_lua_fwrite(L, &tm->token_chunks, sizeof(tm->token_chunks), 1, fh);
  tk_tsetlin_persist_encoder(L, &tm->encoder, fh);
}

static inline void tk_tsetlin_persist_auto_encoder (lua_State *L, tsetlin_auto_encoder_t *tm, FILE *fh)
{
  _tk_tsetlin_persist_classifier(L, &tm->encoder, fh);
  _tk_tsetlin_persist_classifier(L, &tm->decoder, fh);
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
  tk_lua_fwrite(L, &tm->type, sizeof(tm->type), 1, fh);
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
  tk_lua_fread(L, &tm->classes, sizeof(tm->classes), 1, fh);
  tk_lua_fread(L, &tm->features, sizeof(tm->features), 1, fh);
  tk_lua_fread(L, &tm->clauses, sizeof(tm->clauses), 1, fh);
  tk_lua_fread(L, &tm->threshold, sizeof(tm->threshold), 1, fh);
  tk_lua_fread(L, &tm->state_bits, sizeof(tm->state_bits), 1, fh);
  tk_lua_fread(L, &tm->boost_true_positive, sizeof(tm->boost_true_positive), 1, fh);
  tk_lua_fread(L, &tm->input_bits, sizeof(tm->input_bits), 1, fh);
  tk_lua_fread(L, &tm->input_chunks, sizeof(tm->input_chunks), 1, fh);
  tk_lua_fread(L, &tm->clause_chunks, sizeof(tm->clause_chunks), 1, fh);
  tk_lua_fread(L, &tm->state_chunks, sizeof(tm->state_chunks), 1, fh);
  tk_lua_fread(L, &tm->action_chunks, sizeof(tm->action_chunks), 1, fh);
  tk_lua_fread(L, &tm->filter, sizeof(tm->filter), 1, fh);
  tm->state = malloc(sizeof(*tm->state) * tm->state_chunks);
  tk_lua_fread(L, tm->state, sizeof(*tm->state), tm->state_chunks, fh);
  tm->actions = malloc(sizeof(*tm->actions) * tm->action_chunks);
  tk_lua_fread(L, tm->actions, sizeof(*tm->actions), tm->action_chunks, fh);
  tm->clause_output = malloc(sizeof(*tm->clause_output) * tm->clause_chunks);
  tk_lua_fread(L, tm->clause_output, sizeof(*tm->clause_output), tm->clause_chunks, fh);
  tm->feedback_to_la = malloc(sizeof(*tm->feedback_to_la) * tm->input_chunks);
  tk_lua_fread(L, tm->feedback_to_la, sizeof(*tm->feedback_to_la), tm->input_chunks, fh);
  tm->feedback_to_clauses = malloc(sizeof(*tm->feedback_to_clauses) * tm->clause_chunks);
  tk_lua_fread(L, tm->feedback_to_clauses, sizeof(*tm->feedback_to_clauses), tm->clause_chunks, fh);
  tm->drop_clause = malloc(sizeof(*tm->drop_clause) * tm->clause_chunks);
  tk_lua_fread(L, tm->drop_clause, sizeof(*tm->drop_clause), tm->clause_chunks, fh);
}

static inline void tk_tsetlin_load_classifier (lua_State *L, FILE *fh)
{
  tsetlin_t *tm = tk_tsetlin_alloc_classifier(L);
  _tk_tsetlin_load_classifier(L, tm->classifier, fh);
}

static inline void tk_tsetlin_load_recurrent_classifier (lua_State *L, FILE *)
{
  // TODO
  luaL_error(L, "unimplemented: load recurrent classifier");
}

static inline void _tk_tsetlin_load_encoder (lua_State *L, tsetlin_encoder_t *en, FILE *fh)
{
  tk_lua_fread(L, &en->encoding_bits, sizeof(en->encoding_bits), 1, fh);
  tk_lua_fread(L, &en->encoding_chunks, sizeof(en->encoding_chunks), 1, fh);
  tk_lua_fread(L, &en->encoding_filter, sizeof(en->encoding_filter), 1, fh);
  en->encoding_a = malloc(en->encoding_chunks * sizeof(*en->encoding_a) * CHAR_BIT);
  en->encoding_n = malloc(en->encoding_chunks * sizeof(*en->encoding_n) * CHAR_BIT);
  en->encoding_p = malloc(en->encoding_chunks * sizeof(*en->encoding_p) * CHAR_BIT);
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
  en->input_a = malloc(sizeof(unsigned int) * en->encoder.encoder.input_chunks);
  en->input_n = malloc(sizeof(unsigned int) * en->encoder.encoder.input_chunks);
  en->input_p = malloc(sizeof(unsigned int) * en->encoder.encoder.input_chunks);
  en->state = malloc(sizeof(unsigned int) * en->encoder.encoding_chunks);
  _tk_tsetlin_load_encoder(L, &en->encoder, fh);
}

static inline void tk_tsetlin_load_auto_encoder (lua_State *L, FILE *fh)
{
  tsetlin_t *tm = tk_tsetlin_alloc_auto_encoder(L);
  tsetlin_auto_encoder_t *ae = tm->auto_encoder;
  ae->encoding = malloc(sizeof(*ae->encoding) * ae->decoder.input_chunks);
  ae->decoding = malloc(sizeof(*ae->decoding) * ae->encoder.input_chunks);
  _tk_tsetlin_load_classifier(L, &ae->encoder, fh);
  _tk_tsetlin_load_classifier(L, &ae->decoder, fh);
}

static inline void tk_tsetlin_load_regressor (lua_State *L, FILE *fh)
{
  tsetlin_t *tm = tk_tsetlin_alloc_regressor(L);
  tsetlin_regressor_t *rg = tm->regressor;
  _tk_tsetlin_load_classifier(L, &rg->classifier, fh);
}

static inline void tk_tsetlin_load_recurrent_regressor (lua_State *L, FILE *)
{
  // TODO
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
  lua_newtable(L); // t
  tk_tsetlin_register(L, tk_tsetlin_fns, 0); // t
  luaL_newmetatable(L, TK_TSETLIN_MT); // t mt
  lua_pushcfunction(L, tk_tsetlin_destroy); // t mt fn
  lua_setfield(L, -2, "__gc"); // t mt
  lua_pop(L, 1); // t
  return 1;
}
