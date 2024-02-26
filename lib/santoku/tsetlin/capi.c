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

struct TsetlinMachine {
  unsigned int features;
  unsigned int threshold;
  bool boost_true_positive;
  unsigned int clauses;
  unsigned int state_bits;
  unsigned int la_chunks;
  unsigned int clause_chunks;
  unsigned int filter;
	unsigned int *ta_state;
	unsigned int *clause_output;
	unsigned int *feedback_to_la;
	int *feedback_to_clauses;
};

#define tm_state_idx(tm, clause, la_chunk, bit) \
  ((tm)->ta_state[(clause) * ((tm)->la_chunks * (tm)->state_bits) + \
                  (la_chunk) * (tm)->state_bits + \
                  (bit)])

static uint64_t const multiplier = 6364136223846793005u;
static uint64_t mcg_state = 0xcafef00dd15ea5e5u;

static uint32_t fast_rand () {
  uint64_t x = mcg_state;
  unsigned int count = (unsigned int) (x >> 61);
  mcg_state = x * multiplier;
  return (uint32_t) ((x ^ x >> 22) >> (22 + count));
}

static int normal (double mean, double variance) {
  double u1 = (double) (fast_rand() + 1) / ((double) UINT32_MAX + 1);
  double u2 = (double) fast_rand() / UINT32_MAX;
  double n1 = sqrt(-2 * log(u1)) * sin(8 * atan(1) * u2);
  return (int) round(mean + sqrt(variance) * n1);
}

static void tm_initialize_random_streams (struct TsetlinMachine *tm, double specificity)
{
  memset((*tm).feedback_to_la, 0, tm->la_chunks*sizeof(unsigned int));
  int n = 2 * tm->features;
  double p = 1.0 / specificity;
  int active = normal(n * p, n * p * (1 - p));
  active = active >= n ? n : active;
  active = active < 0 ? 0 : active;
  while (active--) {
    int f = fast_rand() % (2 * tm->features);
    while ((*tm).feedback_to_la[f / (sizeof(unsigned int) * CHAR_BIT)] & (1 << (f % (sizeof(unsigned int) * CHAR_BIT))))
      f = fast_rand() % (2 * tm->features);
    (*tm).feedback_to_la[f / (sizeof(unsigned int) * CHAR_BIT)] |= 1 << (f % (sizeof(unsigned int) * CHAR_BIT));
  }
}

static inline void tm_inc (struct TsetlinMachine *tm, int clause, int chunk, unsigned int active)
{
	unsigned int carry, carry_next;
	carry = active;
	for (long int b = 0; b < tm->state_bits; ++b) {
		if (carry == 0)
			break;
		carry_next = tm_state_idx(tm, clause, chunk, b) & carry;
		tm_state_idx(tm, clause, chunk, b) = tm_state_idx(tm, clause, chunk, b) ^ carry;
		carry = carry_next;
	}
	if (carry > 0)
		for (long int b = 0; b < tm->state_bits; ++b)
			tm_state_idx(tm, clause, chunk, b) |= carry;
}

static inline void tm_dec (struct TsetlinMachine *tm, int clause, int chunk, unsigned int active)
{
	unsigned int carry, carry_next;
	carry = active;
	for (long int b = 0; b < tm->state_bits; ++b) {
		if (carry == 0)
			break;
		carry_next = (~tm_state_idx(tm, clause, chunk, b)) & carry; // Sets carry bits (overflow) passing on to next bit
		tm_state_idx(tm, clause, chunk, b) = tm_state_idx(tm, clause, chunk, b) ^ carry; // Performs increments with XOR
		carry = carry_next;
	}
	if (carry > 0)
		for (long int b = 0; b < tm->state_bits; ++b)
			tm_state_idx(tm, clause, chunk, b) &= ~carry;
}

static inline int sum_up_class_votes (struct TsetlinMachine *tm)
{
	int class_sum = 0;
	for (long int j = 0; j < tm->clause_chunks; j++) {
		class_sum += __builtin_popcount((*tm).clause_output[j] & 0x55555555); // 0101
		class_sum -= __builtin_popcount((*tm).clause_output[j] & 0xaaaaaaaa); // 1010
	}
  long int threshold = tm->threshold;
	class_sum = (class_sum > threshold) ? threshold : class_sum;
	class_sum = (class_sum < -threshold) ? -threshold : class_sum;
	return class_sum;
}

static inline void tm_calculate_clause_output(struct TsetlinMachine *tm, unsigned int *Xi, bool predict)
{
  memset((*tm).clause_output, 0, tm->clause_chunks * sizeof(unsigned int));
  for (long int j = 0; j < tm->clauses; j++) {
    unsigned int output = 1;
    unsigned int all_exclude = 1;
    for (long int k = 0; k < tm->la_chunks - 1; k++) {
      output = output && (tm_state_idx(tm, j, k, tm->state_bits-1) & Xi[k]) == tm_state_idx(tm, j, k, tm->state_bits-1);
      if (!output)
        break;
      all_exclude = all_exclude && (tm_state_idx(tm, j, k, tm->state_bits-1) == 0);
    }
		output = output &&
			(tm_state_idx(tm, j, tm->la_chunks-1, tm->state_bits-1) & Xi[tm->la_chunks-1] & tm->filter) ==
			(tm_state_idx(tm, j, tm->la_chunks-1, tm->state_bits-1) & tm->filter);
		all_exclude = all_exclude && ((tm_state_idx(tm, j, tm->la_chunks-1, tm->state_bits-1) & tm->filter) == 0);
		output = output && !(predict && all_exclude == 1);
		if (output) {
			unsigned int clause_chunk = j / (sizeof(unsigned int) * CHAR_BIT);
			unsigned int clause_chunk_pos = j % (sizeof(unsigned int) * CHAR_BIT);
 			(*tm).clause_output[clause_chunk] |= (1 << clause_chunk_pos);
 		}
 	}
}

static void tm_update (struct TsetlinMachine *tm, unsigned int *Xi, unsigned int target, double specificity)
{
  tm_calculate_clause_output(tm, Xi, false);
  long int tgt = target;
  int class_sum = sum_up_class_votes(tm);
  float p = (1.0/(tm->threshold*2))*(tm->threshold + (1 - 2*tgt)*class_sum);
  memset((*tm).feedback_to_clauses, 0, tm->clause_chunks * sizeof(unsigned int));
  for (long int j = 0; j < tm->clauses; j++) {
    unsigned int clause_chunk = j / (sizeof(unsigned int) * CHAR_BIT);
    unsigned int clause_chunk_pos = j % (sizeof(unsigned int) * CHAR_BIT);
    (*tm).feedback_to_clauses[clause_chunk] |= (((float)fast_rand())/((float)UINT32_MAX) <= p) << clause_chunk_pos;
  }
	for (long int j = 0; j < tm->clauses; j++) {
		unsigned int clause_chunk = j / (sizeof(unsigned int) * CHAR_BIT);
		unsigned int clause_chunk_pos = j % (sizeof(unsigned int) * CHAR_BIT);
		if (!((*tm).feedback_to_clauses[clause_chunk] & (1 << clause_chunk_pos)))
			continue;
		if ((2*tgt-1) * (1 - 2 * (j & 1)) == -1) {
      // Type II feedback
			if (((*tm).clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0)
				for (long int k = 0; k < tm->la_chunks; ++k)
					tm_inc(tm, j, k, (~Xi[k]) & (~tm_state_idx(tm, j, k, tm->state_bits-1)));
		} else if ((2*tgt-1) * (1 - 2 * (j & 1)) == 1) {
			// Type I Feedback
			tm_initialize_random_streams(tm, specificity);
			if (((*tm).clause_output[clause_chunk] & (1 << clause_chunk_pos)) > 0) {
				for (long int k = 0; k < tm->la_chunks; ++k) {
					#ifdef _BOOST_TRUE_POSITIVE_FEEDBACK
          tm_inc(tm, j, k, Xi[k]);
					#else
          tm_inc(tm, j, k, Xi[k] & (~tm->feedback_to_la[k]));
					#endif
		 			tm_dec(tm, j, k, (~Xi[k]) & tm->feedback_to_la[k]);
				}
			} else {
				for (long int k = 0; k < tm->la_chunks; ++k) {
					tm_dec(tm, j, k, tm->feedback_to_la[k]);
				}
			}
		}
	}
}

static int tm_score (struct TsetlinMachine *tm, unsigned int *Xi) {
	tm_calculate_clause_output(tm, Xi, true);
	return sum_up_class_votes(tm);
}

struct MultiClassTsetlinMachine {
  unsigned int classes;
	struct TsetlinMachine **tsetlin_machines;
};

static void mc_tm_initialize (struct MultiClassTsetlinMachine *mc_tm)
{
	for (long int i = 0; i < mc_tm->classes; i++)
  {
    struct TsetlinMachine *tm = mc_tm->tsetlin_machines[i];
    for (long int j = 0; j < tm->clauses; ++j) {
      for (long int k = 0; k < tm->la_chunks; ++k) {
        for (long int b = 0; b < tm->state_bits-1; ++b)
          tm_state_idx(tm, j, k, b) = ~0;
        tm_state_idx(tm, j, k, tm->state_bits-1) = 0;
      }
    }
  }
}

static unsigned int mc_tm_predict (struct MultiClassTsetlinMachine *tm, unsigned int *X)
{
	long int max_class = 0;
	long int max_class_sum = tm_score(tm->tsetlin_machines[0], X);
  for (long int i = 1; i < tm->classes; i++) {
    int class_sum = tm_score(tm->tsetlin_machines[i], X);
    if (max_class_sum < class_sum) {
      max_class_sum = class_sum;
      max_class = i;
    }
  }
	return max_class;
}

static void mc_tm_update (struct MultiClassTsetlinMachine *tm, unsigned int *Xi, unsigned int target_class, double specificity)
{
	tm_update(tm->tsetlin_machines[target_class], Xi, 1, specificity);
	unsigned int negative_target_class =
    ((unsigned int) tm->classes * 1.0 * rand()) /
    ((unsigned int) RAND_MAX + 1);
	while (negative_target_class == target_class)
		negative_target_class = (unsigned int)tm->classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	tm_update(tm->tsetlin_machines[negative_target_class], Xi, 0, specificity);
}

struct MultiClassTsetlinMachine **tk_tsetlin_peekp (lua_State *L, int i)
{
  return (struct MultiClassTsetlinMachine **) luaL_checkudata(L, i, TK_TSETLIN_MT);
}

struct MultiClassTsetlinMachine *tk_tsetlin_peek (lua_State *L, int i)
{
  return *tk_tsetlin_peekp(L, i);
}

static unsigned int tk_tsetlin_checkunsigned (lua_State *L, int i)
{
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  if (l > UINT_MAX)
    luaL_error(L, "value is too large");
  return (unsigned int) l;
}

static void tk_tsetlin_register (lua_State *L, luaL_Reg *regs, int nup)
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

static int tk_tsetlin_create (lua_State *L)
{
  lua_settop(L, 6);
	struct MultiClassTsetlinMachine *mc_tm;
	mc_tm = (void *)malloc(sizeof(struct MultiClassTsetlinMachine));
  mc_tm->classes = tk_tsetlin_checkunsigned(L, 1);
  mc_tm->tsetlin_machines = malloc(sizeof(struct TsetlinMachine *) * mc_tm->classes);
	for (long int i = 0; i < mc_tm->classes; i++)
  {
    struct TsetlinMachine *tm = (void *)malloc(sizeof(struct TsetlinMachine));
		mc_tm->tsetlin_machines[i] = tm;
    tm->features = tk_tsetlin_checkunsigned(L, 2);
    tm->clauses = tk_tsetlin_checkunsigned(L, 3);
    tm->state_bits = tk_tsetlin_checkunsigned(L, 4);
    tm->threshold = tk_tsetlin_checkunsigned(L, 5);
    luaL_checktype(L, 6, LUA_TBOOLEAN);
    tm->boost_true_positive = lua_toboolean(L, 6);
    tm->la_chunks = (2 * tm->features - 1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
    tm->clause_chunks = (tm->clauses-  1) / (sizeof(unsigned int) * CHAR_BIT) + 1;
    tm->filter = (tm->features * 2) % (sizeof(unsigned int) * CHAR_BIT) != 0
      ? ~(((unsigned int) ~0) << ((tm->features * 2) % (sizeof(unsigned int) * CHAR_BIT)))
      : (unsigned int) ~0;
    tm->ta_state = malloc(sizeof(unsigned int) * tm->clauses * tm->la_chunks * tm->state_bits);
    tm->clause_output = malloc(sizeof(unsigned int) * tm->clause_chunks);
    tm->feedback_to_la = malloc(sizeof(unsigned int) * tm->la_chunks);
    tm->feedback_to_clauses = malloc(sizeof(int) * tm->clause_chunks);
  }
  mc_tm_initialize(mc_tm);
  struct MultiClassTsetlinMachine **mc_tmp = (struct MultiClassTsetlinMachine **)
    lua_newuserdata(L, sizeof(struct MultiClassTsetlinMachine *));
  *mc_tmp = mc_tm;
  luaL_getmetatable(L, TK_TSETLIN_MT);
  lua_setmetatable(L, -2);
  return 1;
}

static int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  struct MultiClassTsetlinMachine **mc_tmp = tk_tsetlin_peekp(L, 1);
  struct MultiClassTsetlinMachine *mc_tm = *mc_tmp;
  if (mc_tm == NULL)
    return 0;
	for (long int i = 0; i < mc_tm->classes; i++)
  {
    free(mc_tm->tsetlin_machines[i]->ta_state);
    free(mc_tm->tsetlin_machines[i]->clause_output);
    free(mc_tm->tsetlin_machines[i]->feedback_to_la);
    free(mc_tm->tsetlin_machines[i]->feedback_to_clauses);
    free(mc_tm->tsetlin_machines[i]);
  }
  free(mc_tm->tsetlin_machines);
  free(mc_tm);
  *mc_tmp = NULL;
  return 0;
}

static int tk_tsetlin_predict (lua_State *L)
{
  lua_settop(L, 3);
  struct MultiClassTsetlinMachine *tm = tk_tsetlin_peek(L, 1);
  const char *bm = luaL_checkstring(L, 2);
  unsigned int class = mc_tm_predict(tm, (unsigned int *) bm);
  lua_pushinteger(L, class);
  return 1;
}

static int tk_tsetlin_update (lua_State *L)
{
  lua_settop(L, 4);
  struct MultiClassTsetlinMachine *tm = tk_tsetlin_peek(L, 1);
  const char *bm = luaL_checkstring(L, 2);
  lua_Integer tgt = luaL_checkinteger(L, 3);
  if (tgt < 0)
    luaL_error(L, "target class must be greater than zero");
  mc_tm_update(tm, (unsigned int *) bm, tgt, luaL_checknumber(L, 4));
  return 0;
}

static luaL_Reg tk_tsetlin_fns[] =
{
  { "create", tk_tsetlin_create },
  { "destroy", tk_tsetlin_destroy },
  { "update", tk_tsetlin_update },
  { "predict", tk_tsetlin_predict },
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
