/*

Copyright (C) 2025 Birch Point SWE (FuzzyPattern TM added)
Copyright (C) 2025 Birch Point SWE (Regression and Encoder)
Copyright (C) 2025 Birch Point SWE (Grouped mode)
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
#include <santoku/dvec.h>
#include <santoku/rvec.h>
#include <omp.h>
#include <math.h>
#include <limits.h>

#ifndef LUA_OK
#define LUA_OK 0
#endif

#define TK_TSETLIN_MT "santoku_tsetlin"

typedef struct tk_tsetlin_s {
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
  size_t state_chunks;
  size_t action_chunks;
  uint8_t tail_mask;
  char *state;
  char *actions;
  tk_automata_t automata;
  double specificity;
  unsigned int specificity_threshold;
  unsigned int specificity_uint;
  unsigned int *results;
  size_t results_len;
  char *encodings;
  size_t encodings_len;
  double *regression_out;
  size_t regression_out_len;
  double *y_min;
  double *y_max;
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
        votes = 0;
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
        votes = 0;
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

static inline void apply_feedback (
  tk_tsetlin_t *tm,
  unsigned int clause_idx,
  unsigned int chunk,
  char *input,
  unsigned int *literals,
  unsigned int *votes,
  bool positive_feedback
) {
  unsigned int max_literals = tm->clause_maximum;
  unsigned int input_chunks = tm->input_chunks;
  bool output = votes[clause_idx] > 0;
  unsigned int clause_id = chunk * TK_CVEC_BITS + clause_idx;
  if (positive_feedback) {
    if (output) {
      if (literals[clause_idx] < max_literals)
        tk_automata_inc(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
      tk_automata_dec_not_excluded(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
    } else {
      if (literals[clause_idx] == 0) {
        tk_automata_inc(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
      } else {
        unsigned int s = tm->specificity_threshold;
        unsigned int input_bits = tm->input_bits;
        for (unsigned int r = 0; r < s; r ++) {
          unsigned int random_input_bit = tk_fast_random() % input_bits;
          unsigned int random_chunk = random_input_bit / 8;
          uint8_t random_mask = 1 << (random_input_bit % 8);
          tk_automata_dec_byte(&tm->automata, clause_id, random_chunk, random_mask);
        }
      }
    }
  } else {
    if (output) {
      tk_automata_inc_not_excluded(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
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
static inline int tk_tsetlin_classify (lua_State *);
static inline int tk_tsetlin_regress (lua_State *);
static inline int tk_tsetlin_encode (lua_State *);
static inline int tk_tsetlin_destroy (lua_State *L);
static inline int tk_tsetlin_persist (lua_State *);
static inline int tk_tsetlin_checkpoint (lua_State *);
static inline int tk_tsetlin_restore (lua_State *);
static inline int tk_tsetlin_reconfigure (lua_State *);
static inline int tk_tsetlin_restrict (lua_State *);

static luaL_Reg tk_tsetlin_mt_fns[] =
{
  { "train", tk_tsetlin_train },
  { "classify", tk_tsetlin_classify },
  { "regress", tk_tsetlin_regress },
  { "encode", tk_tsetlin_encode },
  { "destroy", tk_tsetlin_destroy },
  { "persist", tk_tsetlin_persist },
  { "checkpoint", tk_tsetlin_checkpoint },
  { "restore", tk_tsetlin_restore },
  { "reconfigure", tk_tsetlin_reconfigure },
  { "restrict", tk_tsetlin_restrict },
  { NULL, NULL }
};

static inline tk_tsetlin_t *tk_tsetlin_alloc (lua_State *L, bool has_state)
{
  tk_tsetlin_t *tm = tk_lua_newuserdata(L, tk_tsetlin_t, TK_TSETLIN_MT, tk_tsetlin_mt_fns, tk_tsetlin_destroy);
  tm->has_state = has_state;
  return tm;
}

static inline void tk_tsetlin_init (
  lua_State *L,
  tk_tsetlin_t *tm,
  unsigned int outputs,
  unsigned int features,
  unsigned int clauses,
  unsigned int clause_tolerance,
  unsigned int clause_maximum,
  unsigned int state_bits,
  unsigned int target,
  double specificity
) {
  if (!outputs)
    tk_lua_verror(L, 3, "create", "outputs", "must be greater than 0");
  if (!clauses)
    tk_lua_verror(L, 3, "create", "clauses", "must be greater than 0");
  if (!clause_tolerance)
    tk_lua_verror(L, 3, "create", "clause_tolerance", "must be greater than 0");
  if (!clause_maximum)
    tk_lua_verror(L, 3, "create", "clause_maximum", "must be greater than 0");
  if (clause_tolerance > clause_maximum) {
    unsigned int tmp = clause_tolerance;
    clause_tolerance = clause_maximum;
    clause_maximum = tmp;
  }
  if (!target)
    tk_lua_verror(L, 3, "create", "target", "must be greater than 0");
  if (state_bits < 2)
    tk_lua_verror(L, 3, "create", "bits", "must be greater than 1");
  tm->reusable = false;
  tm->classes = outputs;
  tm->class_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  tm->clauses = TK_CVEC_BITS_BYTES(clauses) * TK_CVEC_BITS;
  tm->clause_tolerance = clause_tolerance;
  tm->clause_maximum = clause_maximum;
  tm->target = target;
  tm->features = features;
  tm->state_bits = state_bits;
  tm->input_bits = 2 * tm->features;
  uint64_t tail_bits = tm->input_bits & (TK_CVEC_BITS - 1);
  tm->tail_mask = tail_bits ? (uint8_t)((1u << tail_bits) - 1) : 0xFF;
  tm->input_chunks = TK_CVEC_BITS_BYTES(tm->input_bits);
  tm->clause_chunks = TK_CVEC_BITS_BYTES(tm->clauses);
  tm->state_chunks = (size_t)tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tm->action_chunks = (size_t)tm->classes * tm->clauses * tm->input_chunks;
  tm->state = (char *)tk_malloc_aligned(L, tm->state_chunks, TK_CVEC_BITS);
  tm->actions = (char *)tk_malloc_aligned(L, tm->action_chunks, TK_CVEC_BITS);
  tm->specificity = specificity;
  tm->specificity_uint = (unsigned int)specificity;
  tm->specificity_threshold = (2 * tm->features) / tm->specificity_uint;
  tm->y_min = (double *)tk_malloc(L, tm->classes * sizeof(double));
  tm->y_max = (double *)tk_malloc(L, tm->classes * sizeof(double));
  for (unsigned int c = 0; c < tm->classes; c++) {
    tm->y_min[c] = 0.0;
    tm->y_max[c] = 1.0;
  }
  if (!(tm->state && tm->actions && tm->y_min && tm->y_max))
    luaL_error(L, "error in malloc during creation");
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->automata.n_chunks = tm->input_chunks;
  tm->automata.state_bits = tm->state_bits;
  tm->automata.tail_mask = tm->tail_mask;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;
}

static inline void tk_tsetlin_create_impl (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_alloc(L, true);
  lua_insert(L, 1);
  tk_tsetlin_init(L, tm,
      tk_lua_foptunsigned(L, 2, "create", "outputs", 1),
      tk_lua_fcheckunsigned(L, 2, "create", "features"),
      tk_lua_fcheckunsigned(L, 2, "create", "clauses"),
      tk_lua_fcheckunsigned(L, 2, "create", "clause_tolerance"),
      tk_lua_fcheckunsigned(L, 2, "create", "clause_maximum"),
      tk_lua_foptunsigned(L, 2, "create", "state", 8),
      tk_lua_fcheckunsigned(L, 2, "create", "target"),
      tk_lua_fcheckposdouble(L, 2, "create", "specificity"));
  tm->reusable = tk_lua_foptboolean(L, 2, "create", "reusable", false);
  lua_settop(L, 1);
}

static inline int tk_tsetlin_create (lua_State *L)
{
  tk_tsetlin_create_impl(L);
  return 1;
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
  free(tm->regression_out); tm->regression_out = NULL;
  free(tm->y_min); tm->y_min = NULL;
  free(tm->y_max); tm->y_max = NULL;
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  _tk_tsetlin_destroy(tm);
  return 0;
}

static inline int tk_tsetlin_predict_regressor (lua_State *L, tk_tsetlin_t *tm) {
  lua_settop(L, 5);
  tk_cvec_t *ps = tk_cvec_peek(L, 2, "problems");
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  bool grouped = lua_toboolean(L, 4);
  tk_dvec_t *out_buf = lua_isnil(L, 5) ? NULL : tk_dvec_peek(L, 5, "output");
  const unsigned int classes = tm->classes;
  size_t needed = n * classes * sizeof(double);
  if (needed > tm->regression_out_len) {
    tm->regression_out = (double *)tk_realloc(L, tm->regression_out, needed);
    tm->regression_out_len = needed;
  }
  const unsigned int total_chunks = tm->clause_chunks * classes;
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int input_chunks = tm->input_chunks;
  const unsigned int bytes_per_class = grouped ? TK_CVEC_BITS_BYTES(tm->features * 2) : 0;
  const unsigned int input_stride = grouped ? classes * bytes_per_class : 0;
  const unsigned int target = tm->target;
  const double *y_min = tm->y_min;
  const double *y_max = tm->y_max;
  double *regression_out = tm->regression_out;
  const unsigned int neg_chunks = clause_chunks / 2;
  const double neg_offset = (double)(neg_chunks * target);
  const double max_possible = (double)(clause_chunks * target);
  #pragma omp parallel for schedule(static)
  for (unsigned int s = 0; s < n; s++) {
    long int votes_per_class[classes];
    for (unsigned int c = 0; c < classes; c++)
      votes_per_class[c] = 0;
    unsigned int literals[TK_CVEC_BITS];
    unsigned int votes_buf[TK_CVEC_BITS];
    for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
      unsigned int c = chunk / clause_chunks;
      char *input = grouped
        ? ps->a + s * input_stride + c * bytes_per_class
        : ps->a + s * input_chunks;
      uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes_buf, chunk);
      long int chunk_vote = 0;
      for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
        if (out & (1 << j))
          chunk_vote += votes_buf[j];
      if (chunk_vote > (long int)target) chunk_vote = (long int)target;
      bool is_negative = (chunk % clause_chunks) >= clause_chunks - neg_chunks;
      if (is_negative)
        votes_per_class[c] -= chunk_vote;
      else
        votes_per_class[c] += chunk_vote;
    }
    for (unsigned int c = 0; c < classes; c++) {
      double y_range = y_max[c] - y_min[c];
      regression_out[s * classes + c] = ((double)votes_per_class[c] + neg_offset) / max_possible * y_range + y_min[c];
    }
  }
  tk_dvec_t *out;
  if (out_buf) {
    tk_dvec_ensure(out_buf, n * classes);
    out_buf->n = n * classes;
    memcpy(out_buf->a, tm->regression_out, n * classes * sizeof(double));
    lua_pushvalue(L, 5);
    out = out_buf;
  } else {
    out = tk_dvec_create(L, n * classes, 0, 0);
    memcpy(out->a, tm->regression_out, n * classes * sizeof(double));
  }
  (void)out;
  return 1;
}

static inline int tk_tsetlin_classify (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  lua_settop(L, 5);
  tk_cvec_t *ps = tk_cvec_peek(L, 2, "problems");
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  bool grouped = lua_toboolean(L, 4);
  tk_ivec_t *out_buf = lua_isnil(L, 5) ? NULL : tk_ivec_peek(L, 5, "output");
  const unsigned int classes = tm->classes;
  const unsigned int total_chunks = tm->clause_chunks * classes;
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int input_chunks = tm->input_chunks;
  const unsigned int bytes_per_class = grouped ? TK_CVEC_BITS_BYTES(tm->features * 2) : 0;
  const unsigned int input_stride = grouped ? classes * bytes_per_class : 0;
  size_t needed = n * sizeof(unsigned int);
  if (needed > tm->results_len) {
    tm->results = tk_realloc(L, tm->results, needed);
    tm->results_len = needed;
  }
  unsigned int *results = tm->results;
  #pragma omp parallel for schedule(static)
  for (unsigned int s = 0; s < n; s++) {
    long int votes_per_class[classes];
    for (unsigned int c = 0; c < classes; c++)
      votes_per_class[c] = 0;
    unsigned int literals[TK_CVEC_BITS];
    unsigned int votes_buf[TK_CVEC_BITS];
    for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
      unsigned int c = chunk / clause_chunks;
      char *input = grouped
        ? ps->a + s * input_stride + c * bytes_per_class
        : ps->a + s * input_chunks;
      uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes_buf, chunk);
      long int chunk_vote = 0;
      for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
        if (out & (1 << j))
          chunk_vote += votes_buf[j];
      bool is_negative = (chunk % clause_chunks) >= clause_chunks - clause_chunks / 2;
      if (is_negative)
        votes_per_class[c] -= chunk_vote;
      else
        votes_per_class[c] += chunk_vote;
    }
    long int maxval = LONG_MIN;
    unsigned int maxclass = 0;
    for (unsigned int c = 0; c < classes; c++) {
      if (votes_per_class[c] > maxval) {
        maxval = votes_per_class[c];
        maxclass = c;
      }
    }
    results[s] = maxclass;
  }
  tk_ivec_t *out;
  if (out_buf) {
    tk_ivec_ensure(out_buf, n);
    out_buf->n = n;
    for (unsigned int i = 0; i < n; i++)
      out_buf->a[i] = results[i];
    lua_pushvalue(L, 5);
    out = out_buf;
  } else {
    out = tk_ivec_create(L, n, 0, 0);
    for (unsigned int i = 0; i < n; i++)
      out->a[i] = results[i];
  }
  (void)out;
  return 1;
}

static inline int tk_tsetlin_regress (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  return tk_tsetlin_predict_regressor(L, tm);
}

static inline int tk_tsetlin_encode (lua_State *L)
{
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  lua_settop(L, 5);
  tk_cvec_t *ps = tk_cvec_peek(L, 2, "problems");
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  bool grouped = lua_toboolean(L, 4);
  tk_cvec_t *out_buf = lua_isnil(L, 5) ? NULL : tk_cvec_peek(L, 5, "output");
  const unsigned int classes = tm->classes;
  const unsigned int total_chunks = tm->clause_chunks * classes;
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int neg_chunks = clause_chunks / 2;
  const long int encode_mid = (long int)((clause_chunks - 2 * neg_chunks) * tm->target) / 2;
  const unsigned int input_chunks = tm->input_chunks;
  const unsigned int bytes_per_class = grouped ? TK_CVEC_BITS_BYTES(tm->features * 2) : 0;
  const unsigned int input_stride = grouped ? classes * bytes_per_class : 0;
  const unsigned int out_bytes = TK_CVEC_BITS_BYTES(classes);
  size_t needed = n * out_bytes;
  if (needed > tm->encodings_len) {
    tm->encodings = tk_realloc(L, tm->encodings, needed);
    tm->encodings_len = needed;
  }
  char *encodings = tm->encodings;
  memset(encodings, 0, needed);
  #pragma omp parallel for schedule(static)
  for (unsigned int s = 0; s < n; s++) {
    long int votes_per_class[classes];
    for (unsigned int c = 0; c < classes; c++)
      votes_per_class[c] = 0;
    unsigned int literals[TK_CVEC_BITS];
    unsigned int votes_buf[TK_CVEC_BITS];
    for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
      unsigned int c = chunk / clause_chunks;
      char *input = grouped
        ? ps->a + s * input_stride + c * bytes_per_class
        : ps->a + s * input_chunks;
      uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes_buf, chunk);
      long int chunk_vote = 0;
      for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
        if (out & (1 << j))
          chunk_vote += votes_buf[j];
      bool is_negative = (chunk % clause_chunks) >= clause_chunks - neg_chunks;
      if (is_negative)
        votes_per_class[c] -= chunk_vote;
      else
        votes_per_class[c] += chunk_vote;
    }
    char *out_row = encodings + s * out_bytes;
    for (unsigned int c = 0; c < classes; c++) {
      if (votes_per_class[c] > encode_mid) {
        unsigned int byte_idx = c / 8;
        unsigned int bit_idx = c % 8;
        out_row[byte_idx] |= (1 << bit_idx);
      }
    }
  }
  if (out_buf) {
    if (tk_cvec_ensure(out_buf, needed) != 0)
      luaL_error(L, "failed to resize output buffer");
    out_buf->n = needed;
    memcpy(out_buf->a, encodings, needed);
    lua_pushvalue(L, 5);
  } else {
    tk_cvec_t *out = tk_cvec_create(L, needed, 0, 0);
    memcpy(out->a, encodings, needed);
  }
  return 1;
}

typedef enum {
  TM_TARGET_IVEC,
  TM_TARGET_CVEC,
  TM_TARGET_DVEC,
} tm_target_mode_t;

#define TM_REGRESSION_INNER_LOOP(GET_TARGET, SKIP_LOGIC) \
  for (unsigned int i = 0; i < n; i++) { \
    unsigned int sample = shuffle[i]; \
    SKIP_LOGIC \
    char *input = grouped \
      ? ps->a + sample * input_stride + chunk_class * bytes_per_class \
      : ps->a + sample * input_chunks; \
    double y_target = GET_TARGET; \
    uint8_t out = tk_tsetlin_calculate(tm, input, literals, votes, chunk); \
    long int chunk_vote = 0; \
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) \
      if (out & (1 << j)) \
        chunk_vote += votes[j]; \
    if (chunk_vote > vote_target) chunk_vote = vote_target; \
    double class_y_min = y_min[chunk_class]; \
    double class_y_range = y_max[chunk_class] - class_y_min; \
    double target_ratio = (y_target - class_y_min) / class_y_range; \
    if (is_negative) target_ratio = 1.0 - target_ratio; \
    long int ideal_chunk_vote = (long int)(target_ratio * (double)vote_target); \
    if (ideal_chunk_vote < 0) ideal_chunk_vote = 0; \
    if (ideal_chunk_vote > vote_target) ideal_chunk_vote = vote_target; \
    bool want_more = (chunk_vote < ideal_chunk_vote); \
    double error_ratio = (double)(chunk_vote - ideal_chunk_vote) / (double)vote_target; \
    double probability = error_ratio * error_ratio; \
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) { \
      if ((double)tk_fast_random() / 4294967295.0 < probability) \
        apply_feedback(tm, j, chunk, input, literals, votes, want_more); \
    } \
  }

static inline int tk_tsetlin_train_regressor (lua_State *L, tk_tsetlin_t *tm)
{
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  lua_getfield(L, 2, "problems");
  tk_cvec_t *ps = tk_cvec_peek(L, -1, "problems");
  lua_pop(L, 1);

  tm_target_mode_t target_mode;
  tk_ivec_t *solutions = NULL;
  tk_cvec_t *codes = NULL;
  tk_dvec_t *targets_dvec = NULL;
  double *targets = NULL;
  int64_t *labels = NULL;
  uint8_t *code_bytes = NULL;
  unsigned int code_chunks = 0;

  if (tk_lua_ftype(L, 2, "solutions") != LUA_TNIL) {
    target_mode = TM_TARGET_IVEC;
    lua_getfield(L, 2, "solutions");
    solutions = tk_ivec_peek(L, -1, "solutions");
    lua_pop(L, 1);
    labels = solutions->a;
  } else if (tk_lua_ftype(L, 2, "codes") != LUA_TNIL) {
    target_mode = TM_TARGET_CVEC;
    lua_getfield(L, 2, "codes");
    codes = tk_cvec_peek(L, -1, "codes");
    lua_pop(L, 1);
    code_bytes = (uint8_t *)codes->a;
    code_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  } else if (tk_lua_ftype(L, 2, "targets") != LUA_TNIL) {
    target_mode = TM_TARGET_DVEC;
    lua_getfield(L, 2, "targets");
    targets_dvec = tk_dvec_peek(L, -1, "targets");
    lua_pop(L, 1);
    targets = targets_dvec->a;
  } else {
    return luaL_error(L, "regressor train requires solutions (ivec), codes (cvec), or targets (dvec)");
  }

  unsigned int max_iter = tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  unsigned int classes = tm->classes;
  double *y_min = tm->y_min;
  double *y_max = tm->y_max;
  if (target_mode == TM_TARGET_DVEC) {
    for (unsigned int c = 0; c < classes; c++) {
      y_min[c] = DBL_MAX;
      y_max[c] = -DBL_MAX;
    }
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int c = 0; c < classes; c++) {
        double v = targets[i * classes + c];
        if (v > y_max[c]) y_max[c] = v;
        if (v < y_min[c]) y_min[c] = v;
      }
    }
    for (unsigned int c = 0; c < classes; c++) {
      if (y_max[c] <= y_min[c]) { y_min[c] = 0.0; y_max[c] = 1.0; }
    }
  } else {
    for (unsigned int c = 0; c < classes; c++) {
      y_min[c] = -1.0;
      y_max[c] = 1.0;
    }
  }

  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int total_chunks = clause_chunks * classes;
  unsigned int input_chunks = tm->input_chunks;
  bool grouped = tk_lua_foptboolean(L, 2, "train", "grouped", false);
  unsigned int bytes_per_class = grouped ? TK_CVEC_BITS_BYTES(tm->features * 2) : 0;
  unsigned int input_stride = grouped ? classes * bytes_per_class : 0;
  long int vote_target = (long int)tm->target;

  uint32_t *skip_thresholds = NULL;
  if (target_mode == TM_TARGET_IVEC) {
    uint64_t *class_counts = (uint64_t *)calloc(classes, sizeof(uint64_t));
    for (unsigned int i = 0; i < n; i++) {
      unsigned int c = (unsigned int)labels[i];
      if (c < classes) class_counts[c]++;
    }
    skip_thresholds = (uint32_t *)tk_malloc(L, classes * sizeof(uint32_t));
    for (unsigned int c = 0; c < classes; c++) {
      uint64_t pos = class_counts[c];
      uint64_t neg = n - pos;
      if (pos == 0) {
        skip_thresholds[c] = UINT32_MAX;
      } else if (pos >= neg) {
        skip_thresholds[c] = 0;
      } else {
        double skip_prob = 1.0 - (double)pos / (double)neg;
        skip_thresholds[c] = (uint32_t)(skip_prob * (double)UINT32_MAX);
      }
    }
    free(class_counts);
  } else if (target_mode == TM_TARGET_CVEC) {
    uint64_t *bit_counts = (uint64_t *)calloc(classes, sizeof(uint64_t));
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int c = 0; c < classes; c++) {
        if (code_bytes[i * code_chunks + c / 8] & (1 << (c % 8)))
          bit_counts[c]++;
      }
    }
    skip_thresholds = (uint32_t *)tk_malloc(L, classes * sizeof(uint32_t));
    for (unsigned int c = 0; c < classes; c++) {
      uint64_t pos = bit_counts[c];
      uint64_t neg = n - pos;
      if (pos == 0) {
        skip_thresholds[c] = UINT32_MAX;
      } else if (pos >= neg) {
        skip_thresholds[c] = 0;
      } else {
        double skip_prob = 1.0 - (double)pos / (double)neg;
        skip_thresholds[c] = (uint32_t)(skip_prob * (double)UINT32_MAX);
      }
    }
    free(bit_counts);
  }

  bool break_flag = false;
  int max_threads = omp_get_max_threads();
  unsigned int **shuffles = (unsigned int **)tk_malloc(L, (size_t)max_threads * sizeof(unsigned int *));
  for (int t = 0; t < max_threads; t++)
    shuffles[t] = (unsigned int *)tk_malloc(L, n * sizeof(unsigned int));

  if (!tm->trained) {
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      tk_fast_seed((uint64_t)tid);
      #pragma omp for schedule(static)
      for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
        uint64_t first_clause = chunk * TK_CVEC_BITS;
        uint64_t last_clause = chunk * TK_CVEC_BITS + TK_CVEC_BITS - 1;
        if (last_clause >= tm->automata.n_clauses)
          last_clause = tm->automata.n_clauses - 1;
        if (first_clause < tm->automata.n_clauses)
          tk_automata_setup_midpoint(&tm->automata, first_clause, last_clause);
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
      for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
        unsigned int chunk_class = chunk / clause_chunks;
        bool is_negative = (chunk % clause_chunks) >= clause_chunks - clause_chunks / 2;
        switch (target_mode) {
          case TM_TARGET_IVEC:
            TM_REGRESSION_INNER_LOOP(
              (labels[sample] == chunk_class) ? 1.0 : -1.0,
              if (labels[sample] != (int64_t)chunk_class && tk_fast_random() < skip_thresholds[chunk_class]) continue;
            )
            break;
          case TM_TARGET_CVEC:
            TM_REGRESSION_INNER_LOOP(
              (code_bytes[sample * code_chunks + chunk_class / 8] & (1 << (chunk_class % 8))) ? 1.0 : -1.0,
              if (!(code_bytes[sample * code_chunks + chunk_class / 8] & (1 << (chunk_class % 8))) && tk_fast_random() < skip_thresholds[chunk_class]) continue;
            )
            break;
          case TM_TARGET_DVEC:
            TM_REGRESSION_INNER_LOOP(
              targets[sample * classes + chunk_class],
              /* no skip */
            )
            break;
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
  if (skip_thresholds)
    free(skip_thresholds);
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
  return tk_tsetlin_train_regressor(L, tm);
}

static inline void _tk_tsetlin_persist (lua_State *L, tk_tsetlin_t *tm, FILE *fh)
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
  tk_lua_fwrite(L, &tm->state_chunks, sizeof(size_t), 1, fh);
  tk_lua_fwrite(L, &tm->action_chunks, sizeof(size_t), 1, fh);
  tk_lua_fwrite(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &tm->tail_mask, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, tm->actions, 1, tm->action_chunks, fh);
  tk_lua_fwrite(L, tm->y_min, sizeof(double), tm->classes, fh);
  tk_lua_fwrite(L, tm->y_max, sizeof(double), tm->classes, fh);
}

static inline int tk_tsetlin_persist (lua_State *L)
{
  lua_settop(L, 2);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  bool tostr = lua_type(L, 2) == LUA_TNIL;
  FILE *fh = tostr ? tk_lua_tmpfile(L) : tk_lua_fopen(L, tk_lua_checkstring(L, 2, "persist path"), "w");
  _tk_tsetlin_persist(L, tm, fh);
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
  if (new_tolerance > new_maximum) {
    unsigned int tmp = new_tolerance;
    new_tolerance = new_maximum;
    new_maximum = tmp;
  }
  unsigned int new_target = tk_lua_fcheckunsigned(L, 2, "reconfigure", "target");
  double new_specificity = tk_lua_fcheckposdouble(L, 2, "reconfigure", "specificity");
  new_clauses = TK_CVEC_BITS_BYTES(new_clauses) * TK_CVEC_BITS;
  unsigned int new_clause_chunks = TK_CVEC_BITS_BYTES(new_clauses);
  size_t new_action_chunks = (size_t)tm->classes * new_clauses * tm->input_chunks;
  size_t new_state_chunks = (size_t)tm->classes * new_clauses * (tm->state_bits - 1) * tm->input_chunks;
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
  tm->action_chunks = new_action_chunks;
  tm->state_chunks = new_state_chunks;
  tm->specificity = new_specificity;
  tm->specificity_uint = (unsigned int)new_specificity;
  tm->target = new_target;
  tm->specificity_threshold = (2 * tm->features) / tm->specificity_uint;
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;
  tm->trained = false;
  return 0;
}

static inline void tk_tsetlin_restrict_buffer (
  char *buf,
  size_t bytes_per_class,
  unsigned int old_classes,
  unsigned int new_classes,
  int64_t *keep
) {
  (void)old_classes;
  char *temp = malloc(new_classes * bytes_per_class);
  for (unsigned int i = 0; i < new_classes; i++) {
    unsigned int src = (unsigned int)keep[i];
    memcpy(temp + i * bytes_per_class, buf + src * bytes_per_class, bytes_per_class);
  }
  memcpy(buf, temp, new_classes * bytes_per_class);
  free(temp);
}

static inline int tk_tsetlin_restrict (lua_State *L)
{
  lua_settop(L, 2);
  tk_tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  tk_ivec_t *keep = tk_ivec_peek(L, 2, "classes");
  if (tm->destroyed)
    return luaL_error(L, "cannot restrict a destroyed model");
  if (keep->n == 0)
    return luaL_error(L, "restrict requires at least one class");
  tk_iuset_t *seen = tk_iuset_create(NULL, 0);
  int kha;
  for (uint64_t i = 0; i < keep->n; i++) {
    if (keep->a[i] < 0 || (unsigned int)keep->a[i] >= tm->classes) {
      tk_iuset_destroy(seen);
      return luaL_error(L, "class index %d out of range [0, %d)", (int)keep->a[i], tm->classes);
    }
    if (tk_iuset_contains(seen, keep->a[i])) {
      tk_iuset_destroy(seen);
      return luaL_error(L, "duplicate class index %d", (int)keep->a[i]);
    }
    tk_iuset_put(seen, keep->a[i], &kha);
  }
  tk_iuset_destroy(seen);
  unsigned int new_classes = (unsigned int)keep->n;
  size_t bytes_per_class_actions = (size_t)tm->clauses * tm->input_chunks;
  size_t bytes_per_class_state = (size_t)tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tk_tsetlin_restrict_buffer(tm->actions, bytes_per_class_actions, tm->classes, new_classes, keep->a);
  tm->action_chunks = new_classes * bytes_per_class_actions;
  if (tm->state) {
    tk_tsetlin_restrict_buffer(tm->state, bytes_per_class_state, tm->classes, new_classes, keep->a);
    tm->state_chunks = new_classes * bytes_per_class_state;
  }
  tm->classes = new_classes;
  tm->class_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  tm->automata.n_clauses = tm->classes * tm->clauses;
  return 0;
}

static inline void _tk_tsetlin_load (lua_State *L, tk_tsetlin_t *tm, FILE *fh)
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
  tk_lua_fread(L, &tm->state_chunks, sizeof(size_t), 1, fh);
  tk_lua_fread(L, &tm->action_chunks, sizeof(size_t), 1, fh);
  tk_lua_fread(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fread(L, &tm->tail_mask, sizeof(uint8_t), 1, fh);
  tm->actions = (char *)tk_malloc_aligned(L, tm->action_chunks, TK_CVEC_BITS);
  tk_lua_fread(L, tm->actions, 1, tm->action_chunks, fh);
  tm->state = NULL;
  tm->y_min = (double *)tk_malloc(L, tm->classes * sizeof(double));
  tm->y_max = (double *)tk_malloc(L, tm->classes * sizeof(double));
  tk_lua_fread(L, tm->y_min, sizeof(double), tm->classes, fh);
  tk_lua_fread(L, tm->y_max, sizeof(double), tm->classes, fh);
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->automata.n_chunks = tm->input_chunks;
  tm->automata.state_bits = tm->state_bits;
  tm->automata.tail_mask = tm->tail_mask;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;
}

static inline int tk_tsetlin_load (lua_State *L)
{
  lua_settop(L, 2);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && lua_toboolean(L, 2);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  tk_tsetlin_t *tm = tk_tsetlin_alloc(L, false);
  _tk_tsetlin_load(L, tm, fh);
  tk_lua_fclose(L, fh);
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
