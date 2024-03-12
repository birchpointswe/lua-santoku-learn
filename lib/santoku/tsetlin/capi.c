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
  TM_REGRESSOR
} tsetlin_type_t;

typedef struct {
  unsigned int classes;
  unsigned int features;
  unsigned int clauses;
  unsigned int threshold;
  unsigned int state_bits;
  bool boost_true_positive;
  unsigned int la_chunks;
  unsigned int clause_chunks;
  unsigned int filter;
  unsigned int *ta_state; // class clause bit chunk
  unsigned int *actions; // class clause chunk
  unsigned int *clause_output;
  unsigned int *feedback_to_la;
  unsigned int *feedback_to_clauses;
  unsigned int *drop_clause;
} tsetlin_classifier_t;

typedef struct {
  tsetlin_classifier_t encoder;
  tsetlin_classifier_t decoder;
} tsetlin_autoencoder_t;

typedef struct {
  tsetlin_classifier_t classifier;
} tsetlin_regressor_t;

typedef struct {
  tsetlin_type_t type;
  union {
    tsetlin_classifier_t *classifier;
    tsetlin_autoencoder_t *autoencoder;
    tsetlin_regressor_t *regressor;
  };
} tsetlin_t;

#define tm_state_idx_counts(tm, class, clause, la_chunk) \
  (&(tm)->ta_state[(class) * (tm)->clauses * (tm)->la_chunks * ((tm)->state_bits - 1) + \
                   (clause) * (tm)->la_chunks * ((tm)->state_bits - 1) + \
                   (la_chunk) * ((tm)->state_bits - 1)])

#define tm_state_idx_actions(tm, class, clause) \
  (&(tm)->actions[(class) * (tm)->clauses *  (tm)->la_chunks + \
                  (clause) * (tm)->la_chunks])

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

static inline void tm_initialize_random_streams (tsetlin_classifier_t *tm, double specificity)
{
  memset((*tm).feedback_to_la, 0, tm->la_chunks*sizeof(unsigned int));
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
    unsigned int la_chunks = tm->la_chunks;
    unsigned int clause_chunk = j / (sizeof(unsigned int) * CHAR_BIT);
    unsigned int clause_chunk_pos = j % (sizeof(unsigned int) * CHAR_BIT);
    unsigned int *actions = tm_state_idx_actions(tm, class, j);
    for (unsigned int k = 0; k < la_chunks - 1; k ++) {
      output |= ((actions[k] & Xi[k]) ^ actions[k]);
      all_exclude |= actions[k];
    }
    output |=
      (actions[la_chunks - 1] & Xi[la_chunks - 1] & filter) ^
      (actions[la_chunks - 1] & filter);
    all_exclude |= ((actions[la_chunks - 1] & filter) ^ 0);
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
  long int tgt = target;
  long int class_sum = sum_up_class_votes(tm, false);
  unsigned int la_chunks = tm->la_chunks;
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
        for (unsigned int k = 0; k < la_chunks; k ++)
          tm_inc(tm, class, j, k, (~Xi[k]) & (~actions[k]));
    } else if ((2 * tgt - 1) * (1 - 2 * (jl & 1)) == 1) {
      // Type I Feedback
      tm_initialize_random_streams(tm, specificity);
      if ((clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
        if (tm->boost_true_positive)
          for (unsigned int k = 0; k < la_chunks; k ++)
            tm_inc(tm, class, j, k, Xi[k]);
        else
          for (unsigned int k = 0; k < la_chunks; k ++)
            tm_inc(tm, class, j, k, Xi[k] & (~feedback_to_la[k]));
        for (unsigned int k = 0; k < la_chunks; k ++)
          tm_dec(tm, class, j, k, (~Xi[k]) & feedback_to_la[k]);
      } else {
        for (unsigned int k = 0; k < la_chunks; k ++)
          tm_dec(tm, class, j, k, feedback_to_la[k]);
      }
    }
  }
}

static inline int tm_score (tsetlin_classifier_t *tm, unsigned int class, unsigned int *Xi) {
  tm_calculate_clause_output(tm, class, Xi, true);
  return sum_up_class_votes(tm, true);
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

static inline int tk_tsetlin_create_classifier (lua_State *L, tsetlin_classifier_t *tm)
{
  tm->classes = tk_tsetlin_checkunsigned(L, 1);
  tm->features = tk_tsetlin_checkunsigned(L, 2);
  tm->clauses = tk_tsetlin_checkunsigned(L, 3);
  tm->state_bits = tk_tsetlin_checkunsigned(L, 4);
  tm->threshold = tk_tsetlin_checkunsigned(L, 5);
  luaL_checktype(L, 6, LUA_TBOOLEAN);
  tm->boost_true_positive = lua_toboolean(L, 6);
  tm->la_chunks = (2 * tm->features - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  tm->clause_chunks = (tm->clauses - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
  tm->filter = (tm->features * 2) % (sizeof(unsigned int) * CHAR_BIT) != 0
    ? ~(((unsigned int) ~0) << ((tm->features * 2) % (sizeof(unsigned int) * CHAR_BIT)))
    : (unsigned int) ~0;
  tm->drop_clause = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->ta_state = malloc(sizeof(unsigned int) * tm->classes * tm->clauses * (tm->state_bits - 1) * tm->la_chunks);
  tm->actions = malloc(sizeof(unsigned int) * tm->classes * tm->clauses * tm->la_chunks);
  tm->clause_output = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->feedback_to_la = malloc(sizeof(unsigned int) * tm->la_chunks);
  tm->feedback_to_clauses = malloc(sizeof(unsigned int) * tm->clause_chunks);
  if (!(tm->drop_clause && tm->ta_state && tm->clause_output && tm->feedback_to_la && tm->feedback_to_clauses))
    luaL_error(L, "error in malloc during creation");
  for (unsigned int i = 0; i < tm->classes; i ++)
    for (unsigned int j = 0; j < tm->clauses; j ++)
      for (unsigned int k = 0; k < tm->la_chunks; k ++) {
        unsigned int m = tm->state_bits - 1;
        unsigned int *actions = tm_state_idx_actions(tm, i, j);
        unsigned int *counts = tm_state_idx_counts(tm, i, j, k);
        actions[k] = 0U;
        for (unsigned int b = 0; b < m; b ++)
          counts[b] = ~0U;
      }
  return 0;
}

static inline int tk_tsetlin_create_autoencoder (lua_State *L, tsetlin_autoencoder_t *tm)
{
  // TODO
  return 0;
}

static inline int tk_tsetlin_create_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  // TODO
  return 0;
}

static inline int tk_tsetlin_create (lua_State *L)
{
  const char *type = luaL_checkstring(L, 1);
  if (!strcmp(type, "classifier")) {

    lua_remove(L, 1);
    tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t *));
    if (!tm) goto err_mem;
    luaL_getmetatable(L, TK_TSETLIN_MT);
    lua_setmetatable(L, -2);
    tm->type = TM_CLASSIFIER;
    tm->classifier = (void *)malloc(sizeof(tsetlin_classifier_t));
    if (!tm->classifier) goto err_mem;
    tk_tsetlin_create_classifier(L, tm->classifier);
    return 1;

  } else if (!strcmp(type, "autoencoder")) {

    lua_remove(L, 1);
    tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t *));
    if (!tm) goto err_mem;
    luaL_getmetatable(L, TK_TSETLIN_MT);
    lua_setmetatable(L, -2);
    tm->type = TM_AUTOENCODER;
    tm->autoencoder = (void *)malloc(sizeof(tsetlin_autoencoder_t));
    if (!tm->autoencoder) goto err_mem;
    tk_tsetlin_create_autoencoder(L, tm->autoencoder);
    return 1;

  } else if (!strcmp(type, "regressor")) {

    lua_remove(L, 1);
    tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t *));
    if (!tm) goto err_mem;
    luaL_getmetatable(L, TK_TSETLIN_MT);
    lua_setmetatable(L, -2);
    tm->type = TM_REGRESSOR;
    tm->regressor = (void *)malloc(sizeof(tsetlin_regressor_t));
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
  free(tm->ta_state);
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
      free(tm->autoencoder);
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
  const char *bm = luaL_checkstring(L, 2);
  unsigned int class = mc_tm_predict(tm, (unsigned int *) bm);
  lua_pushinteger(L, class);
  return 1;
}

static inline int tk_tsetlin_predict_autoencoder (lua_State *L, tsetlin_autoencoder_t *tm)
{
  // TODO
  return 0;
}

static inline int tk_tsetlin_predict_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  // TODO
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
  const char *bm = luaL_checkstring(L, 2);
  lua_Integer tgt = luaL_checkinteger(L, 3);
  if (tgt < 0)
    luaL_error(L, "target class must be greater than zero");
  double specificity = luaL_checknumber(L, 4);
  double drop_clause = luaL_optnumber(L, 5, 1);
  mc_tm_initialize_drop_clause(tm, drop_clause);
  mc_tm_update(tm, (unsigned int *) bm, tgt, specificity);
  return 0;
}

static inline int tk_tsetlin_update_autoencoder (lua_State *L, tsetlin_autoencoder_t *tm)
{
  // TODO
  return 0;
}

static inline int tk_tsetlin_update_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  // TODO
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
    mc_tm_update(tm, &ps[i * tm->la_chunks], ss[i], specificity);
  return 0;
}

static inline int tk_tsetlin_train_autoencoder (lua_State *L, tsetlin_autoencoder_t *tm)
{
  // TODO
  return 0;
}

static inline int tk_tsetlin_train_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  // TODO
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
  bool track_stats = lua_toboolean(L, 5);
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
    unsigned int predicted = mc_tm_predict(tm, &ps[i * tm->la_chunks]);
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
  // TODO
  return 0;
}

static inline int tk_tsetlin_evaluate_regressor (lua_State *L, tsetlin_regressor_t *tm)
{
  // TODO
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
