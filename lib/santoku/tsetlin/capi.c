/*

Copyright (C) 2024 Birch Point SWE (Lua integration, train/evaluate stats, drop clause)
Copyright (C) 2019 Ole-Christoffer Granmo (Original C implementation)

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
	unsigned int *ta_state;
	unsigned int *clause_output;
	unsigned int *feedback_to_la;
	unsigned int *feedback_to_clauses;
  unsigned int *drop_clause;
} tsetlin_t;

#define tm_state_idx(tm, class, clause, la_chunk, bit) \
  ((tm)->ta_state[(class) * (tm)->clauses * (tm)->la_chunks * (tm)->state_bits + \
                  (clause) * (tm)->la_chunks * (tm)->state_bits + \
                  (la_chunk) * (tm)->state_bits + \
                  (bit)])

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

static inline void tm_initialize_random_streams (tsetlin_t *tm, double specificity)
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

static inline void tm_inc (tsetlin_t *tm, unsigned int class, unsigned int clause, unsigned int chunk, unsigned int active)
{
	unsigned int carry, carry_next;
	carry = active;
	for (unsigned int b = 0; b < tm->state_bits; b ++) {
		if (carry == 0)
			break;
		carry_next = tm_state_idx(tm, class, clause, chunk, b) & carry;
		tm_state_idx(tm, class, clause, chunk, b) = tm_state_idx(tm, class, clause, chunk, b) ^ carry;
		carry = carry_next;
	}
	if (carry > 0)
		for (unsigned int b = 0; b < tm->state_bits; b ++)
			tm_state_idx(tm, class, clause, chunk, b) |= carry;
}

static inline void tm_dec (tsetlin_t *tm, unsigned int class, unsigned int clause, unsigned int chunk, unsigned int active)
{
	unsigned int carry, carry_next;
	carry = active;
	for (unsigned int b = 0; b < tm->state_bits; b ++) {
		if (carry == 0)
			break;
		carry_next = (~tm_state_idx(tm, class, clause, chunk, b)) & carry; // Sets carry bits (overflow) passing on to next bit
		tm_state_idx(tm, class, clause, chunk, b) = tm_state_idx(tm, class, clause, chunk, b) ^ carry; // Performs increments with XOR
		carry = carry_next;
	}
	if (carry > 0)
		for (unsigned int b = 0; b < tm->state_bits; b ++)
			tm_state_idx(tm, class, clause, chunk, b) &= ~carry;
}

static inline long int sum_up_class_votes (tsetlin_t *tm, bool predict)
{
	long int class_sum = 0;
  if (predict) {
    for (unsigned int j = 0; j < tm->clause_chunks; j ++) {
      class_sum += __builtin_popcount((*tm).clause_output[j] & 0x55555555); // 0101
      class_sum -= __builtin_popcount((*tm).clause_output[j] & 0xaaaaaaaa); // 1010
    }
  } else {
    for (unsigned int j = 0; j < tm->clause_chunks; j ++) {
      class_sum += __builtin_popcount((*tm).clause_output[j] & (*tm).drop_clause[j] & 0x55555555); // 0101
      class_sum -= __builtin_popcount((*tm).clause_output[j] & (*tm).drop_clause[j] & 0xaaaaaaaa); // 1010
    }
  }
  long int threshold = tm->threshold;
	class_sum = (class_sum > threshold) ? threshold : class_sum;
	class_sum = (class_sum < -threshold) ? -threshold : class_sum;
	return class_sum;
}

static inline void tm_calculate_clause_output (tsetlin_t *tm, unsigned int class, unsigned int *Xi, bool predict)
{
  memset((*tm).clause_output, 0, tm->clause_chunks * sizeof(unsigned int));
  for (unsigned int j = 0; j < tm->clauses; j ++) {
    unsigned int output = 1;
    unsigned int all_exclude = 1;
    for (unsigned int k = 0; k < tm->la_chunks - 1; k ++) {
      output = output && (tm_state_idx(tm, class, j, k, tm->state_bits-1) & Xi[k]) == tm_state_idx(tm, class, j, k, tm->state_bits-1);
      if (!output)
        break;
      all_exclude = all_exclude && (tm_state_idx(tm, class, j, k, tm->state_bits-1) == 0);
    }
		output = output &&
			(tm_state_idx(tm, class, j, tm->la_chunks-1, tm->state_bits-1) & Xi[tm->la_chunks-1] & tm->filter) ==
			(tm_state_idx(tm, class, j, tm->la_chunks-1, tm->state_bits-1) & tm->filter);
		all_exclude = all_exclude && ((tm_state_idx(tm, class, j, tm->la_chunks-1, tm->state_bits-1) & tm->filter) == 0);
		output = output && !(predict && all_exclude == 1);
		if (output) {
			unsigned int clause_chunk = j / (sizeof(unsigned int) * CHAR_BIT);
			unsigned int clause_chunk_pos = j % (sizeof(unsigned int) * CHAR_BIT);
 			(*tm).clause_output[clause_chunk] |= (1 << clause_chunk_pos);
 		}
 	}
}

static inline void tm_update (tsetlin_t *tm, unsigned int class, unsigned int *Xi, unsigned int target, double specificity)
{
  tm_calculate_clause_output(tm, class, Xi, false);
  long int tgt = target;
  long int class_sum = sum_up_class_votes(tm, false);
  float p = (1.0/(tm->threshold*2))*(tm->threshold + (1 - 2*tgt)*class_sum);
  memset((*tm).feedback_to_clauses, 0, tm->clause_chunks * sizeof(unsigned int));
  for (unsigned int i = 0; i < tm->clause_chunks; i ++)
  {
    for (unsigned int j = 0; j < sizeof(unsigned int) * CHAR_BIT; j ++)
      (*tm).feedback_to_clauses[i] |= (unsigned int)
        (((float) fast_rand()) / ((float) UINT32_MAX) <= p) << j;
    (*tm).feedback_to_clauses[i] &= tm->drop_clause[i];
  }
	for (unsigned int j = 0; j < tm->clauses; j ++) {
    long int jl = (long int) j;
		unsigned int clause_chunk = j / (sizeof(unsigned int) * CHAR_BIT);
		unsigned int clause_chunk_pos = j % (sizeof(unsigned int) * CHAR_BIT);
		if (!((*tm).feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos)))
			continue;
		if ((2*tgt-1) * (1 - 2 * (jl & 1)) == -1) {
      // Type II feedback
			if (((*tm).clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0)
				for (unsigned int k = 0; k < tm->la_chunks; k ++)
					tm_inc(tm, class, j, k, (~Xi[k]) & (~tm_state_idx(tm, class, j, k, tm->state_bits-1)));
		} else if ((2*tgt-1) * (1 - 2 * (jl & 1)) == 1) {
			// Type I Feedback
			tm_initialize_random_streams(tm, specificity);
			if (((*tm).clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				for (unsigned int k = 0; k < tm->la_chunks; k ++) {
          if (tm->boost_true_positive)
            tm_inc(tm, class, j, k, Xi[k]);
          else
            tm_inc(tm, class, j, k, Xi[k] & (~tm->feedback_to_la[k]));
		 			tm_dec(tm, class, j, k, (~Xi[k]) & tm->feedback_to_la[k]);
				}
			} else {
				for (unsigned int k = 0; k < tm->la_chunks; k ++) {
					tm_dec(tm, class, j, k, tm->feedback_to_la[k]);
				}
			}
		}
	}
}

static inline int tm_score (tsetlin_t *tm, unsigned int class, unsigned int *Xi) {
	tm_calculate_clause_output(tm, class, Xi, true);
	return sum_up_class_votes(tm, true);
}

static inline unsigned int mc_tm_predict (tsetlin_t *tm, unsigned int *X)
{
	long int max_class = 0;
	long int max_class_sum = tm_score(tm, 0, X);
  for (long int i = 1; i < tm->classes; i ++) {
    long int class_sum = tm_score(tm, i, X);
    if (max_class_sum < class_sum) {
      max_class_sum = class_sum;
      max_class = i;
    }
  }
	return max_class;
}

static inline void mc_tm_update (tsetlin_t *tm, unsigned int *Xi, unsigned int target_class, double specificity)
{
	tm_update(tm, target_class, Xi, 1, specificity);
	unsigned int negative_target_class = (unsigned int) fast_rand() % tm->classes;
	while (negative_target_class == target_class)
		negative_target_class = (unsigned int) fast_rand() % tm->classes;
	tm_update(tm, negative_target_class, Xi, 0, specificity);
}

static inline void mc_tm_initialize_drop_clause (tsetlin_t *tm, double drop_clause)
{
  memset(tm->drop_clause, 0, sizeof(unsigned int) * tm->clause_chunks);
  for (unsigned int i = 0; i < tm->clause_chunks; i ++)
    for (unsigned int j = 0; j < sizeof(unsigned int) * CHAR_BIT; j ++)
      if (((float)fast_rand())/((float)UINT32_MAX) <= drop_clause)
        tm->drop_clause[i] |= (1 << j);
}

tsetlin_t **tk_tsetlin_peekp (lua_State *L, int i)
{
  return (tsetlin_t **) luaL_checkudata(L, i, TK_TSETLIN_MT);
}

tsetlin_t *tk_tsetlin_peek (lua_State *L, int i)
{
  return *tk_tsetlin_peekp(L, i);
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

static inline int tk_tsetlin_create (lua_State *L)
{
  lua_settop(L, 6);
	tsetlin_t *tm;
	tm = (void *)malloc(sizeof(tsetlin_t));
  if (!tm)
    luaL_error(L, "error in malloc during creation");
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
  tm->ta_state = malloc(sizeof(unsigned int) * tm->classes * tm->clauses * tm->la_chunks * tm->state_bits);
  tm->clause_output = malloc(sizeof(unsigned int) * tm->clause_chunks);
  tm->feedback_to_la = malloc(sizeof(unsigned int) * tm->la_chunks);
  tm->feedback_to_clauses = malloc(sizeof(unsigned int) * tm->clause_chunks);
  if (!(tm->drop_clause && tm->ta_state && tm->clause_output && tm->feedback_to_la && tm->feedback_to_clauses))
    luaL_error(L, "error in malloc during creation");
	for (unsigned int i = 0; i < tm->classes; i ++)
    for (unsigned int j = 0; j < tm->clauses; j ++) {
      for (unsigned int k = 0; k < tm->la_chunks; k ++) {
        for (unsigned int b = 0; b < tm->state_bits - 1; b ++)
          tm_state_idx(tm, i, j, k, b) = ~((unsigned int) 0);
        tm_state_idx(tm, i, j, k, tm->state_bits - 1) = 0;
      }
    }
  tsetlin_t **tmp = (tsetlin_t **)
    lua_newuserdata(L, sizeof(tsetlin_t *));
  *tmp = tm;
  luaL_getmetatable(L, TK_TSETLIN_MT);
  lua_setmetatable(L, -2);
  return 1;
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tsetlin_t **tmp = tk_tsetlin_peekp(L, 1);
  tsetlin_t *tm = *tmp;
  if (tm == NULL)
    return 0;
  free(tm->ta_state);
  free(tm->clause_output);
  free(tm->drop_clause);
  free(tm->feedback_to_la);
  free(tm->feedback_to_clauses);
  free(tm);
  *tmp = NULL;
  return 0;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  lua_settop(L, 3);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  const char *bm = luaL_checkstring(L, 2);
  unsigned int class = mc_tm_predict(tm, (unsigned int *) bm);
  lua_pushinteger(L, class);
  return 1;
}

static inline int tk_tsetlin_update (lua_State *L)
{
  lua_settop(L, 5);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
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

static inline int tk_tsetlin_train (lua_State *L)
{
  lua_settop(L, 6);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
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

static inline int tk_tsetlin_evaluate (lua_State *L)
{
  lua_settop(L, 5);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
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

static luaL_Reg tk_tsetlin_fns[] =
{
  { "create", tk_tsetlin_create },
  { "destroy", tk_tsetlin_destroy },
  { "update", tk_tsetlin_update },
  { "predict", tk_tsetlin_predict },
  { "train", tk_tsetlin_train },
  { "evaluate", tk_tsetlin_evaluate },
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
