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
#include <santoku/learn/automata.h>
#include <santoku/cvec.h>
#include <santoku/dvec.h>
#include <santoku/ivec.h>
#include <santoku/pvec.h>
#include <santoku/rvec.h>
#include <omp.h>
#include <math.h>
#include <limits.h>
#include <string.h>

#ifndef LUA_OK
#define LUA_OK 0
#endif

#define TK_LEARN_MT "santoku_learn"

typedef struct tk_learn_s {
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
  unsigned int *results;
  size_t results_len;
  char *encodings;
  size_t encodings_len;
  double *regression_out;
  size_t regression_out_len;
  double *y_min;
  double *y_max;

  unsigned int *clause_tolerances;
  unsigned int *clause_maximums;
  unsigned int *specificity_thresholds;
  unsigned int *targets;

  bool sparse_mode;
  unsigned int n_tokens;
  unsigned int absorb_interval;

  int64_t *csc_offsets;
  int64_t *csc_indices;

  int64_t *mapping;
  uint8_t *active;
  unsigned int active_bytes;

  char *managed_dense;
  unsigned int dense_n_samples;
  unsigned int dense_bpc;
  unsigned int dense_bps;

  uint8_t *absorb_scratch;
  int absorb_threads;
  unsigned int absorb_threshold;
  unsigned int absorb_maximum;
  unsigned int absorb_insert;

  int64_t *absorb_ranking;
  unsigned int absorb_ranking_n;
  int64_t *absorb_ranking_offsets;
  int64_t *absorb_ranking_global;
  unsigned int absorb_ranking_global_n;
  unsigned int *absorb_cursor;
} tk_learn_t;

static inline uint8_t tk_learn_calculate (
  tk_learn_t *tm,
  char *input,
  unsigned int *literalsp,
  unsigned int *votesp,
  unsigned int chunk,
  unsigned int tolerance,
  unsigned int empty_vote
) {
  uint8_t out = 0;
  const unsigned int input_chunks = tm->input_chunks;
  const uint8_t tail_mask = tm->tail_mask;
  const uint8_t *input_bytes = (const uint8_t *)input;
  if (input_chunks > 1) {
    const unsigned int all_bits = input_chunks * 8;
    const uint8_t garbage_mask = (uint8_t)~tail_mask;
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) {
      unsigned int clause = chunk * TK_CVEC_BITS + j;
      const uint8_t *actions = (const uint8_t *)tk_automata_actions(&tm->automata, clause);
      uint64_t literals_64, failed_64;
      tk_cvec_bits_popcount_andnot_serial(actions, input_bytes, all_bits, &literals_64, &failed_64);
      const uint8_t garbage_act = actions[input_chunks - 1] & garbage_mask;
      unsigned int literals = (unsigned int)literals_64 - (unsigned int)__builtin_popcount(garbage_act);
      unsigned int failed = (unsigned int)failed_64 - (unsigned int)__builtin_popcount(garbage_act & ~input_bytes[input_chunks - 1]);
      long int votes;
      if (literals == 0) {
        votes = empty_vote;
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
        votes = empty_vote;
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
  tk_learn_t *tm,
  unsigned int clause_idx,
  unsigned int chunk,
  char *input,
  unsigned int *literals,
  unsigned int *votes,
  bool positive_feedback,
  unsigned int max_literals,
  unsigned int specificity_thresh
) {
  unsigned int input_chunks = tm->input_chunks;
  bool output = votes[clause_idx] > 0;
  unsigned int clause_id = chunk * TK_CVEC_BITS + clause_idx;
  if (positive_feedback) {
    if (output) {
      if (literals[clause_idx] < max_literals)
        tk_automata_inc(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
      tk_automata_dec_not_excluded(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
    } else {
      unsigned int s = specificity_thresh;
      unsigned int input_bits = tm->input_bits;
      for (unsigned int r = 0; r < s; r ++) {
        unsigned int random_input_bit = tk_fast_random() % input_bits;
        unsigned int random_chunk = random_input_bit / 8;
        uint8_t random_mask = 1 << (random_input_bit % 8);
        tk_automata_dec_byte(&tm->automata, clause_id, random_chunk, random_mask);
      }
    }
  } else {
    if (output) {
      tk_automata_inc_not_excluded(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
    }
  }
}

static inline void tk_learn_init_shuffle (
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

tk_learn_t *tk_learn_peek (lua_State *L, int i)
{
  return (tk_learn_t *) luaL_checkudata(L, i, TK_LEARN_MT);
}

static inline int tk_learn_train (lua_State *);
static inline int tk_learn_classify (lua_State *);
static inline int tk_learn_regress (lua_State *);
static inline int tk_learn_encode (lua_State *);
static inline int tk_learn_destroy (lua_State *L);
static inline int tk_learn_persist (lua_State *);
static inline int tk_learn_checkpoint (lua_State *);
static inline int tk_learn_restore (lua_State *);
static inline int tk_learn_reconfigure (lua_State *);
static inline int tk_learn_restrict (lua_State *);
static inline int tk_learn_active_features (lua_State *);

static luaL_Reg tk_learn_mt_fns[] =
{
  { "train", tk_learn_train },
  { "classify", tk_learn_classify },
  { "regress", tk_learn_regress },
  { "encode", tk_learn_encode },
  { "destroy", tk_learn_destroy },
  { "persist", tk_learn_persist },
  { "checkpoint", tk_learn_checkpoint },
  { "restore", tk_learn_restore },
  { "reconfigure", tk_learn_reconfigure },
  { "restrict", tk_learn_restrict },
  { "active_features", tk_learn_active_features },
  { NULL, NULL }
};

static inline tk_learn_t *tk_learn_alloc (lua_State *L, bool has_state)
{
  tk_learn_t *tm = tk_lua_newuserdata(L, tk_learn_t, TK_LEARN_MT, tk_learn_mt_fns, tk_learn_destroy);
  tm->has_state = has_state;
  tm->sparse_mode = false;
  tm->n_tokens = 0;
  tm->absorb_interval = 10;
  tm->csc_offsets = NULL;
  tm->csc_indices = NULL;
  tm->mapping = NULL;
  tm->active = NULL;
  tm->active_bytes = 0;
  tm->managed_dense = NULL;
  tm->dense_n_samples = 0;
  tm->dense_bpc = 0;
  tm->dense_bps = 0;
  tm->absorb_scratch = NULL;
  tm->absorb_threads = 0;
  tm->absorb_threshold = 0;
  tm->absorb_maximum = 0;
  tm->absorb_insert = 1;
  tm->absorb_ranking = NULL;
  tm->absorb_ranking_n = 0;
  tm->absorb_ranking_offsets = NULL;
  tm->absorb_ranking_global = NULL;
  tm->absorb_ranking_global_n = 0;
  tm->absorb_cursor = NULL;
  tm->clause_tolerances = NULL;
  tm->clause_maximums = NULL;
  tm->specificity_thresholds = NULL;
  tm->targets = NULL;
  return tm;
}

static inline void tk_learn_init (
  lua_State *L,
  tk_learn_t *tm,
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
  if (clause_tolerance > clause_maximum)
    tk_lua_verror(L, 3, "create", "clause_tolerance", "must be <= clause_maximum");
  if (!target)
    tk_lua_verror(L, 3, "create", "target", "must be greater than 0");
  if (state_bits < 2)
    tk_lua_verror(L, 3, "create", "bits", "must be greater than 1");
  tm->reusable = false;
  tm->classes = outputs;
  tm->class_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  tm->clause_chunks = clauses * 2;
  tm->clauses = tm->clause_chunks * TK_CVEC_BITS;
  tm->clause_tolerance = clause_tolerance;
  tm->clause_maximum = clause_maximum;
  tm->target = target;
  tm->features = features;
  tm->state_bits = state_bits;
  tm->input_bits = 2 * tm->features;
  uint64_t tail_bits = tm->input_bits & (TK_CVEC_BITS - 1);
  tm->tail_mask = tail_bits ? (uint8_t)((1u << tail_bits) - 1) : 0xFF;
  tm->input_chunks = TK_CVEC_BITS_BYTES(tm->input_bits);
  tm->state_chunks = (size_t)tm->classes * tm->clauses * (tm->state_bits - 1) * tm->input_chunks;
  tm->action_chunks = (size_t)tm->classes * tm->clauses * tm->input_chunks;
  tm->state = (char *)tk_malloc_aligned(L, tm->state_chunks, TK_CVEC_BITS);
  tm->actions = (char *)tk_malloc_aligned(L, tm->action_chunks, TK_CVEC_BITS);
  tm->specificity = specificity;
  tm->specificity_threshold = (unsigned int)((2.0 * (double)tm->features) / specificity);
  tm->y_min = (double *)tk_malloc(L, tm->classes * sizeof(double));
  tm->y_max = (double *)tk_malloc(L, tm->classes * sizeof(double));
  for (unsigned int c = 0; c < tm->classes; c++) {
    tm->y_min[c] = 0.0;
    tm->y_max[c] = 1.0;
  }
  tm->clause_tolerances = (unsigned int *)tk_malloc(L, outputs * sizeof(unsigned int));
  tm->clause_maximums = (unsigned int *)tk_malloc(L, outputs * sizeof(unsigned int));
  tm->specificity_thresholds = (unsigned int *)tk_malloc(L, outputs * sizeof(unsigned int));
  tm->targets = (unsigned int *)tk_malloc(L, outputs * sizeof(unsigned int));
  for (unsigned int c = 0; c < outputs; c++) {
    tm->clause_tolerances[c] = clause_tolerance;
    tm->clause_maximums[c] = clause_maximum;
    tm->specificity_thresholds[c] = tm->specificity_threshold;
    tm->targets[c] = target;
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

static inline void tk_learn_create_impl (lua_State *L)
{
  tk_learn_t *tm = tk_learn_alloc(L, true);
  lua_insert(L, 1);
  tk_learn_init(L, tm,
      tk_lua_foptunsigned(L, 2, "create", "outputs", 1),
      tk_lua_fcheckunsigned(L, 2, "create", "features"),
      tk_lua_fcheckunsigned(L, 2, "create", "clauses"),
      tk_lua_fcheckunsigned(L, 2, "create", "clause_tolerance"),
      tk_lua_fcheckunsigned(L, 2, "create", "clause_maximum"),
      tk_lua_foptunsigned(L, 2, "create", "state", 8),
      tk_lua_fcheckunsigned(L, 2, "create", "target"),
      tk_lua_fcheckposdouble(L, 2, "create", "specificity"));
  tm->reusable = tk_lua_foptboolean(L, 2, "create", "reusable", false);
  if (tk_lua_ftype(L, 2, "per_class_tolerances") != LUA_TNIL) {
    lua_getfield(L, 2, "per_class_tolerances");
    tk_ivec_t *v = tk_ivec_peek(L, -1, "per_class_tolerances");
    for (unsigned int c = 0; c < tm->classes; c++)
      tm->clause_tolerances[c] = (unsigned int)v->a[c];
    lua_pop(L, 1);
  }
  if (tk_lua_ftype(L, 2, "per_class_maximums") != LUA_TNIL) {
    lua_getfield(L, 2, "per_class_maximums");
    tk_ivec_t *v = tk_ivec_peek(L, -1, "per_class_maximums");
    for (unsigned int c = 0; c < tm->classes; c++)
      tm->clause_maximums[c] = (unsigned int)v->a[c];
    lua_pop(L, 1);
  }
  if (tk_lua_ftype(L, 2, "per_class_spec_thresholds") != LUA_TNIL) {
    lua_getfield(L, 2, "per_class_spec_thresholds");
    tk_ivec_t *v = tk_ivec_peek(L, -1, "per_class_spec_thresholds");
    for (unsigned int c = 0; c < tm->classes; c++)
      tm->specificity_thresholds[c] = (unsigned int)v->a[c];
    lua_pop(L, 1);
  }
  if (tk_lua_ftype(L, 2, "per_class_targets") != LUA_TNIL) {
    lua_getfield(L, 2, "per_class_targets");
    tk_ivec_t *v = tk_ivec_peek(L, -1, "per_class_targets");
    for (unsigned int c = 0; c < tm->classes; c++)
      tm->targets[c] = (unsigned int)v->a[c];
    lua_pop(L, 1);
  }
  if (tk_lua_ftype(L, 2, "n_tokens") != LUA_TNIL) {
    tm->sparse_mode = true;
    tm->n_tokens = tk_lua_fcheckunsigned(L, 2, "create", "n_tokens");
    tm->absorb_interval = tk_lua_foptunsigned(L, 2, "create", "absorb_interval", 10);
    tm->absorb_threshold = tk_lua_foptunsigned(L, 2, "create", "absorb_threshold", 0);
    tm->absorb_maximum = tk_lua_foptunsigned(L, 2, "create", "absorb_maximum", 0);
    tm->absorb_insert = tk_lua_foptunsigned(L, 2, "create", "absorb_insert", tm->absorb_threshold + 1);
    tm->active_bytes = TK_CVEC_BITS_BYTES(tm->n_tokens);
    tm->active = (uint8_t *)calloc((uint64_t)tm->classes * tm->active_bytes, 1);
    tm->mapping = (int64_t *)malloc((uint64_t)tm->classes * tm->features * sizeof(int64_t));
    tm->absorb_threads = omp_get_max_threads();
    tm->absorb_scratch = (uint8_t *)malloc((uint64_t)tm->absorb_threads * tm->features * 2 * sizeof(unsigned int));
  }
  lua_settop(L, 1);
}

static inline int tk_learn_create (lua_State *L)
{
  tk_learn_create_impl(L);
  return 1;
}

static inline void tk_learn_shrink (tk_learn_t *tm)
{
  if (tm == NULL) return;
  tm->reusable = false;
  free(tm->state); tm->state = NULL;
}

static inline void _tk_learn_destroy (tk_learn_t *tm)
{
  if (tm == NULL) return;
  if (tm->destroyed) return;
  tm->destroyed = true;
  tk_learn_shrink(tm);
  free(tm->actions); tm->actions = NULL;
  free(tm->results); tm->results = NULL;
  free(tm->encodings); tm->encodings = NULL;
  free(tm->regression_out); tm->regression_out = NULL;
  free(tm->y_min); tm->y_min = NULL;
  free(tm->y_max); tm->y_max = NULL;
  free(tm->clause_tolerances); tm->clause_tolerances = NULL;
  free(tm->clause_maximums); tm->clause_maximums = NULL;
  free(tm->specificity_thresholds); tm->specificity_thresholds = NULL;
  free(tm->targets); tm->targets = NULL;
  tm->csc_offsets = NULL;
  tm->csc_indices = NULL;
  free(tm->mapping); tm->mapping = NULL;
  free(tm->active); tm->active = NULL;
  free(tm->managed_dense); tm->managed_dense = NULL;
  free(tm->absorb_scratch); tm->absorb_scratch = NULL;
  free(tm->absorb_cursor); tm->absorb_cursor = NULL;
  free(tm->absorb_ranking_offsets); tm->absorb_ranking_offsets = NULL;
  if (tm->absorb_ranking_global != tm->absorb_ranking)
    free(tm->absorb_ranking_global);
  tm->absorb_ranking_global = NULL;
  free(tm->absorb_ranking); tm->absorb_ranking = NULL;
}

static inline int tk_learn_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tk_learn_t *tm = tk_learn_peek(L, 1);
  _tk_learn_destroy(tm);
  return 0;
}

static inline bool sparse_is_active (tk_learn_t *tm, unsigned int c, unsigned int tok) {
  uint64_t idx = (uint64_t)c * tm->active_bytes + tok / 8;
  return (tm->active[idx] >> (tok % 8)) & 1;
}

static inline void sparse_set_active (tk_learn_t *tm, unsigned int c, unsigned int tok) {
  uint64_t idx = (uint64_t)c * tm->active_bytes + tok / 8;
  tm->active[idx] |= (uint8_t)(1 << (tok % 8));
}

static inline void sparse_clear_active (tk_learn_t *tm, unsigned int c, unsigned int tok) {
  uint64_t idx = (uint64_t)c * tm->active_bytes + tok / 8;
  tm->active[idx] &= (uint8_t)~(1 << (tok % 8));
}


static inline void tk_learn_init_mapping (tk_learn_t *tm, unsigned int c,
    int64_t *init_ids, unsigned int n_init) {
  unsigned int features = tm->features;
  unsigned int n_tokens = tm->n_tokens;
  unsigned int filled = 0;
  for (unsigned int i = 0; i < n_init && filled < features; i++) {
    unsigned int tok = (unsigned int)init_ids[i];
    if (tok >= n_tokens || sparse_is_active(tm, c, tok)) continue;
    tm->mapping[(uint64_t)c * features + filled] = (int64_t)tok;
    sparse_set_active(tm, c, tok);
    filled++;
  }
  while (filled < features) {
    unsigned int tok = tk_fast_random() % n_tokens;
    if (sparse_is_active(tm, c, tok)) continue;
    tm->mapping[(uint64_t)c * features + filled] = (int64_t)tok;
    sparse_set_active(tm, c, tok);
    filled++;
  }
}

static inline void tk_learn_densify_slot (tk_learn_t *tm, unsigned int c, unsigned int slot, unsigned int new_tok, int64_t old_tok) {
  unsigned int bpc = tm->dense_bpc;
  unsigned int bps = tm->dense_bps;
  uint8_t *out = (uint8_t *)tm->managed_dense;
  unsigned int bit_pos = (2 * slot) % 8;
  uint8_t pair_mask = (uint8_t)(0x3 << bit_pos);
  uint8_t absent_bits = (uint8_t)(0x2 << bit_pos);
  uint8_t present_bits = (uint8_t)(0x1 << bit_pos);
  unsigned int local_byte = (2 * slot) / 8;

  if (old_tok >= 0) {
    int64_t csc_start = tm->csc_offsets[old_tok];
    int64_t csc_end = tm->csc_offsets[old_tok + 1];
    for (int64_t i = csc_start; i < csc_end; i++) {
      uint32_t s = (uint32_t)tm->csc_indices[i];
      uint8_t *byte = out + (uint64_t)s * bps + (uint64_t)c * bpc + local_byte;
      *byte = (*byte & ~pair_mask) | absent_bits;
    }
  }

  int64_t csc_start = tm->csc_offsets[new_tok];
  int64_t csc_end = tm->csc_offsets[new_tok + 1];
  for (int64_t i = csc_start; i < csc_end; i++) {
    uint32_t s = (uint32_t)tm->csc_indices[i];
    uint8_t *byte = out + (uint64_t)s * bps + (uint64_t)c * bpc + local_byte;
    *byte = (*byte & ~pair_mask) | present_bits;
  }
}

static inline void tk_learn_densify_all (tk_learn_t *tm) {
  unsigned int classes = tm->classes;
  unsigned int features = tm->features;
  unsigned int bps = tm->dense_bps;
  unsigned int n_samples = tm->dense_n_samples;
  uint8_t *out = (uint8_t *)tm->managed_dense;

  memset(out, 0xAA, (uint64_t)n_samples * bps);

  #pragma omp parallel for schedule(dynamic)
  for (unsigned int c = 0; c < classes; c++) {
    for (unsigned int k = 0; k < features; k++) {
      unsigned int tok = (unsigned int)tm->mapping[(uint64_t)c * features + k];
      tk_learn_densify_slot(tm, c, k, tok, -1);
    }
  }
}

static inline void tk_learn_reset_slot_automata (tk_learn_t *tm, unsigned int c, unsigned int slot) {
  unsigned int clauses = tm->clauses;
  unsigned int state_bits = tm->state_bits;
  unsigned int m = state_bits - 1;
  unsigned int clause_base = c * clauses;
  unsigned int byte_idx = (2 * slot) / 8;
  uint8_t mask = (uint8_t)(0x3 << ((2 * slot) % 8));
  unsigned int insert_val = tm->absorb_insert;
  unsigned int max_excl = (1u << m) - 1;
  if (insert_val <= tm->absorb_threshold) insert_val = tm->absorb_threshold + 1;
  if (insert_val > max_excl) insert_val = max_excl;
  for (unsigned int ci = 0; ci < clauses; ci++) {
    unsigned int clause_id = clause_base + ci;
    for (unsigned int b = 0; b < m; b++) {
      uint8_t *plane = (uint8_t *)tk_automata_counts_plane(&tm->automata, clause_id, b);
      if (insert_val & (1u << b))
        plane[byte_idx] |= mask;
      else
        plane[byte_idx] &= ~mask;
    }
    uint8_t *actions = (uint8_t *)tk_automata_actions(&tm->automata, clause_id);
    actions[byte_idx] &= ~mask;
  }
}

static inline unsigned int tk_learn_absorb_class (tk_learn_t *tm, unsigned int c) {
  unsigned int clauses = tm->clauses;
  unsigned int features = tm->features;
  unsigned int n_tokens = tm->n_tokens;
  unsigned int clause_base = c * clauses;
  unsigned int threshold = tm->absorb_threshold;
  unsigned int m = tm->state_bits - 1;
  unsigned int max_counter = (1u << m) - 1;

  unsigned int *scratch = (unsigned int *)(tm->absorb_scratch +
    (uint64_t)omp_get_thread_num() * features * 2 * sizeof(unsigned int));
  unsigned int *max_states = scratch;
  unsigned int *eligible = scratch + features;
  memset(max_states, 0, features * sizeof(unsigned int));

  for (unsigned int ci = 0; ci < clauses; ci++) {
    unsigned int clause_id = clause_base + ci;
    uint8_t *action = (uint8_t *)tk_automata_actions(&tm->automata, clause_id);
    for (unsigned int k = 0; k < features; k++) {
      if (max_states[k] > threshold) continue;
      for (unsigned int lit = 0; lit < 2; lit++) {
        unsigned int bit = 2 * k + lit;
        unsigned int byte_idx = bit / 8;
        uint8_t mask = 1 << (bit % 8);
        unsigned int act = (action[byte_idx] & mask) ? 1 : 0;
        unsigned int counter = 0;
        for (unsigned int b = 0; b < m; b++) {
          uint8_t *plane = (uint8_t *)tk_automata_counts_plane(&tm->automata, clause_id, b);
          if (plane[byte_idx] & mask) counter |= (1u << b);
        }
        unsigned int full_state = act ? (max_counter + 1 + counter) : counter;
        if (full_state > max_states[k])
          max_states[k] = full_state;
      }
    }
  }

  unsigned int n_eligible = 0;
  for (unsigned int k = 0; k < features; k++)
    if (max_states[k] <= threshold)
      eligible[n_eligible++] = k;

  if (n_eligible == 0) return 0;

  for (unsigned int i = 1; i < n_eligible; i++) {
    unsigned int key = eligible[i];
    unsigned int key_val = max_states[key];
    int j = (int)i - 1;
    while (j >= 0 && max_states[eligible[j]] > key_val) {
      eligible[j + 1] = eligible[j];
      j--;
    }
    eligible[j + 1] = key;
  }

  unsigned int max_replace = tm->absorb_maximum;
  if (max_replace == 0 || max_replace > n_eligible)
    max_replace = n_eligible;

  uint8_t *active_base = tm->active + (uint64_t)c * tm->active_bytes;
  unsigned int active_pop = (unsigned int)tk_cvec_bits_popcount_serial(active_base, n_tokens);
  bool vocab_exhausted = active_pop >= n_tokens;

  int64_t *rank_base = tm->absorb_ranking_global;
  unsigned int rank_len = tm->absorb_ranking_global_n;

  unsigned int n_replaced = 0;
  for (unsigned int i = 0; i < max_replace; i++) {
    if (vocab_exhausted) break;
    unsigned int k = eligible[i];
    int64_t old_tok = tm->mapping[(uint64_t)c * features + k];
    if (old_tok >= 0)
      sparse_clear_active(tm, c, (unsigned int)old_tok);
    unsigned int new_tok;
    unsigned int *cursor = &tm->absorb_cursor[c];
    unsigned int start = *cursor;
    do {
      new_tok = (unsigned int)rank_base[*cursor];
      *cursor = (*cursor + 1) % rank_len;
    } while (sparse_is_active(tm, c, new_tok) && *cursor != start);
    if (sparse_is_active(tm, c, new_tok)) { continue; }
    tm->mapping[(uint64_t)c * features + k] = (int64_t)new_tok;
    sparse_set_active(tm, c, new_tok);
    tk_learn_densify_slot(tm, c, k, new_tok, old_tok);
    tk_learn_reset_slot_automata(tm, c, k);
    n_replaced++;
  }

  return n_replaced;
}

static inline char *tk_learn_densify_tokens (tk_learn_t *tm, tk_ivec_t *tokens, unsigned int n_samples) {
  unsigned int classes = tm->classes;
  unsigned int features = tm->features;
  unsigned int n_tokens = tm->n_tokens;
  unsigned int bpc = TK_CVEC_BITS_BYTES(features * 2);
  unsigned int bps = classes * bpc;
  uint64_t out_size = (uint64_t)n_samples * bps;
  char *out = (char *)malloc(out_size);
  memset(out, 0xAA, out_size);
  uint8_t *data = (uint8_t *)out;

  uint64_t *counts = (uint64_t *)calloc(n_samples, sizeof(uint64_t));
  for (uint64_t i = 0; i < tokens->n; i++) {
    int64_t v = tokens->a[i];
    if (v < 0) continue;
    uint64_t s = (uint64_t)v / n_tokens;
    if (s < n_samples) counts[s]++;
  }
  uint64_t *sample_offsets = (uint64_t *)malloc((n_samples + 1) * sizeof(uint64_t));
  sample_offsets[0] = 0;
  for (unsigned int s = 0; s < n_samples; s++)
    sample_offsets[s + 1] = sample_offsets[s] + counts[s];
  uint32_t *binned = (uint32_t *)malloc(sample_offsets[n_samples] * sizeof(uint32_t));
  memset(counts, 0, n_samples * sizeof(uint64_t));
  for (uint64_t i = 0; i < tokens->n; i++) {
    int64_t v = tokens->a[i];
    if (v < 0) continue;
    uint64_t s = (uint64_t)v / n_tokens;
    if (s >= n_samples) continue;
    binned[sample_offsets[s] + counts[s]] = (uint32_t)((uint64_t)v % n_tokens);
    counts[s]++;
  }
  free(counts);

  int64_t *reverse = (int64_t *)malloc((uint64_t)n_tokens * sizeof(int64_t));
  memset(reverse, 0xFF, (uint64_t)n_tokens * sizeof(int64_t));

  for (unsigned int c = 0; c < classes; c++) {
    int64_t *cm = tm->mapping + (uint64_t)c * features;
    for (unsigned int k = 0; k < features; k++)
      if (cm[k] >= 0) reverse[cm[k]] = (int64_t)k;

    #pragma omp parallel for schedule(guided)
    for (unsigned int s = 0; s < n_samples; s++) {
      uint8_t *base = data + (uint64_t)s * bps + (uint64_t)c * bpc;
      for (uint64_t bi = sample_offsets[s]; bi < sample_offsets[s + 1]; bi++) {
        uint32_t tok = binned[bi];
        int64_t slot = reverse[tok];
        if (slot < 0) continue;
        unsigned int k = (unsigned int)slot;
        unsigned int bit_pos = (2 * k) % 8;
        uint8_t pair_mask = (uint8_t)(0x3 << bit_pos);
        uint8_t present_bits = (uint8_t)(0x1 << bit_pos);
        unsigned int local_byte = (2 * k) / 8;
        base[local_byte] = (base[local_byte] & ~pair_mask) | present_bits;
      }
    }

    for (unsigned int k = 0; k < features; k++)
      if (cm[k] >= 0) reverse[cm[k]] = -1;
  }

  free(reverse);
  free(binned);
  free(sample_offsets);
  return out;
}

static inline int tk_learn_predict_regressor (lua_State *L, tk_learn_t *tm) {
  lua_settop(L, 5);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  bool grouped = lua_toboolean(L, 4);
  tk_dvec_t *out_buf = lua_isnil(L, 5) ? NULL : tk_dvec_peek(L, 5, "output");

  tk_cvec_t *ps = NULL;
  char *temp_dense = NULL;
  if (tm->sparse_mode && lua_type(L, 2) != LUA_TUSERDATA) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = lua_type(L, 2) == LUA_TTABLE ? tk_lua_fcheckunsigned(L, 2, "regress", "n_samples") : n;
    temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
    grouped = true;
  } else {
    ps = tk_cvec_peek(L, 2, "problems");
  }

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
  const unsigned int *per_class_targets = tm->targets;
  const double *y_min = tm->y_min;
  const double *y_max = tm->y_max;
  double *regression_out = tm->regression_out;
  char *base_ptr = temp_dense ? temp_dense : ps->a;
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
        ? base_ptr + s * input_stride + c * bytes_per_class
        : base_ptr + s * input_chunks;
      uint8_t out = tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
      long int chunk_vote = 0;
      for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
        if (out & (1 << j))
          chunk_vote += votes_buf[j];
      if (chunk_vote > (long int)tm->targets[c]) chunk_vote = (long int)tm->targets[c];
      if ((chunk % clause_chunks) >= clause_chunks / 2) chunk_vote = -chunk_vote;
      votes_per_class[c] += chunk_vote;
    }
    for (unsigned int c = 0; c < classes; c++) {
      double y_range = y_max[c] - y_min[c];
      regression_out[s * classes + c] = ((double)votes_per_class[c] / ((double)clause_chunks * per_class_targets[c]) + 0.5) * y_range + y_min[c];
    }
  }
  free(temp_dense);
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

static inline int tk_learn_classify (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  lua_settop(L, 5);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  bool grouped = lua_toboolean(L, 4);
  tk_ivec_t *out_buf = lua_isnil(L, 5) ? NULL : tk_ivec_peek(L, 5, "output");

  tk_cvec_t *ps = NULL;
  char *temp_dense = NULL;
  if (tm->sparse_mode && lua_type(L, 2) != LUA_TUSERDATA) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = lua_type(L, 2) == LUA_TTABLE ? tk_lua_fcheckunsigned(L, 2, "classify", "n_samples") : n;
    temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
    grouped = true;
  } else {
    ps = tk_cvec_peek(L, 2, "problems");
  }

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
  char *base_ptr = temp_dense ? temp_dense : ps->a;
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
        ? base_ptr + s * input_stride + c * bytes_per_class
        : base_ptr + s * input_chunks;
      uint8_t out = tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
      long int chunk_vote = 0;
      for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
        if (out & (1 << j))
          chunk_vote += votes_buf[j];
      if ((chunk % clause_chunks) >= clause_chunks / 2) chunk_vote = -chunk_vote;
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
  free(temp_dense);
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

static inline int tk_learn_regress (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  return tk_learn_predict_regressor(L, tm);
}

static inline int tk_learn_encode (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  lua_settop(L, 5);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  bool grouped = lua_toboolean(L, 4);
  tk_cvec_t *out_buf = lua_isnil(L, 5) ? NULL : tk_cvec_peek(L, 5, "output");

  tk_cvec_t *ps = NULL;
  char *temp_dense = NULL;
  if (tm->sparse_mode && lua_type(L, 2) != LUA_TUSERDATA) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = lua_type(L, 2) == LUA_TTABLE ? tk_lua_fcheckunsigned(L, 2, "encode", "n_samples") : n;
    temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
    grouped = true;
  } else {
    ps = tk_cvec_peek(L, 2, "problems");
  }

  const unsigned int classes = tm->classes;
  const unsigned int total_chunks = tm->clause_chunks * classes;
  const unsigned int clause_chunks = tm->clause_chunks;
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
  char *base_ptr = temp_dense ? temp_dense : ps->a;
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
        ? base_ptr + s * input_stride + c * bytes_per_class
        : base_ptr + s * input_chunks;
      uint8_t out = tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
      long int chunk_vote = 0;
      for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
        if (out & (1 << j))
          chunk_vote += votes_buf[j];
      if ((chunk % clause_chunks) >= clause_chunks / 2) chunk_vote = -chunk_vote;
      votes_per_class[c] += chunk_vote;
    }
    char *out_row = encodings + s * out_bytes;
    for (unsigned int c = 0; c < classes; c++) {
      if (votes_per_class[c] > 0) {
        unsigned int byte_idx = c / 8;
        unsigned int bit_idx = c % 8;
        out_row[byte_idx] |= (1 << bit_idx);
      }
    }
  }
  free(temp_dense);
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
  { \
  unsigned int chunk_tolerance = tm->clause_tolerances[chunk_class]; \
  unsigned int chunk_maximum = tm->clause_maximums[chunk_class]; \
  unsigned int chunk_spec_thresh = tm->specificity_thresholds[chunk_class]; \
  long int vote_target = (long int)tm->targets[chunk_class]; \
  for (unsigned int i = 0; i < n; i++) { \
    unsigned int sample = shuffle[i]; \
    SKIP_LOGIC \
    char *input = grouped \
      ? ps->a + sample * input_stride + chunk_class * bytes_per_class \
      : ps->a + sample * input_chunks; \
    double y_target = GET_TARGET; \
    uint8_t out = tk_learn_calculate(tm, input, literals, votes, chunk, chunk_tolerance, 1); \
    long int chunk_vote = 0; \
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) \
      if (out & (1 << j)) \
        chunk_vote += votes[j]; \
    bool chunk_neg = (chunk % clause_chunks) >= clause_chunks / 2; \
    double class_y_min = y_min[chunk_class]; \
    double class_y_range = y_max[chunk_class] - class_y_min; \
    double target_ratio = (y_target - class_y_min) / class_y_range; \
    if (chunk_neg) target_ratio = 1.0 - target_ratio; \
    double ideal_chunk_vote = target_ratio * (double)vote_target; \
    if (ideal_chunk_vote < 0.0) ideal_chunk_vote = 0.0; \
    if (ideal_chunk_vote > (double)vote_target) ideal_chunk_vote = (double)vote_target; \
    bool want_more = ((double)chunk_vote < ideal_chunk_vote); \
    double error_ratio = ((double)chunk_vote - ideal_chunk_vote) / (double)vote_target; \
    double probability = fabs(error_ratio); \
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) { \
      if ((double)tk_fast_random() / 4294967295.0 < probability) { \
        apply_feedback(tm, j, chunk, input, literals, votes, want_more, chunk_maximum, chunk_spec_thresh); \
      } \
    } \
  } \
  }

static inline int tk_learn_train_regressor (lua_State *L, tk_learn_t *tm)
{
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");

  tk_cvec_t *ps = NULL;
  bool sparse_train = false;

  if (tm->sparse_mode && tk_lua_ftype(L, 2, "csc_offsets") != LUA_TNIL) {
    sparse_train = true;

    if (!tm->managed_dense) {
      lua_getfield(L, 2, "csc_offsets");
      tk_ivec_t *csc_off = tk_ivec_peek(L, -1, "csc_offsets");
      lua_pop(L, 1);
      lua_getfield(L, 2, "csc_indices");
      tk_ivec_t *csc_idx = tk_ivec_peek(L, -1, "csc_indices");
      lua_pop(L, 1);
      tm->csc_offsets = csc_off->a;
      tm->csc_indices = csc_idx->a;
      if (!tm->absorb_ranking) {
        lua_getfield(L, 2, "absorb_ranking");
        if (!lua_isnil(L, -1)) {
          tk_ivec_t *rank_ivec = tk_ivec_peek(L, -1, "absorb_ranking");
          tm->absorb_ranking_n = (unsigned int)rank_ivec->n;
          tm->absorb_ranking = (int64_t *)malloc(rank_ivec->n * sizeof(int64_t));
          memcpy(tm->absorb_ranking, rank_ivec->a, rank_ivec->n * sizeof(int64_t));
          lua_pop(L, 1);
          lua_getfield(L, 2, "absorb_ranking_offsets");
          if (!lua_isnil(L, -1)) {
            tk_ivec_t *off_ivec = tk_ivec_peek(L, -1, "absorb_ranking_offsets");
            tm->absorb_ranking_offsets = (int64_t *)malloc(off_ivec->n * sizeof(int64_t));
            memcpy(tm->absorb_ranking_offsets, off_ivec->a, off_ivec->n * sizeof(int64_t));
          }
          lua_pop(L, 1);
        } else {
          lua_pop(L, 1);
          tm->absorb_ranking_n = tm->n_tokens;
          tm->absorb_ranking = (int64_t *)malloc((uint64_t)tm->n_tokens * sizeof(int64_t));
          for (unsigned int i = 0; i < tm->n_tokens; i++)
            tm->absorb_ranking[i] = (int64_t)i;
          for (unsigned int i = tm->n_tokens - 1; i > 0; i--) {
            unsigned int j = tk_fast_random() % (i + 1);
            int64_t tmp = tm->absorb_ranking[i];
            tm->absorb_ranking[i] = tm->absorb_ranking[j];
            tm->absorb_ranking[j] = tmp;
          }
        }
      }
      if (!tm->absorb_ranking_global) {
        lua_getfield(L, 2, "absorb_ranking_global");
        if (!lua_isnil(L, -1)) {
          tk_ivec_t *g = tk_ivec_peek(L, -1, "absorb_ranking_global");
          tm->absorb_ranking_global_n = (unsigned int)g->n;
          tm->absorb_ranking_global = (int64_t *)malloc(g->n * sizeof(int64_t));
          memcpy(tm->absorb_ranking_global, g->a, g->n * sizeof(int64_t));
        } else if (!tm->absorb_ranking_offsets) {
          tm->absorb_ranking_global = tm->absorb_ranking;
          tm->absorb_ranking_global_n = tm->absorb_ranking_n;
        } else {
          tm->absorb_ranking_global_n = tm->absorb_ranking_n;
          tm->absorb_ranking_global = (int64_t *)malloc(tm->absorb_ranking_n * sizeof(int64_t));
          memcpy(tm->absorb_ranking_global, tm->absorb_ranking, tm->absorb_ranking_n * sizeof(int64_t));
        }
        lua_pop(L, 1);
      }
      if (!tm->absorb_cursor)
        tm->absorb_cursor = (unsigned int *)calloc(tm->classes, sizeof(unsigned int));
      memset(tm->absorb_cursor, 0, tm->classes * sizeof(unsigned int));
      tk_fast_seed(42);
      for (unsigned int c = 0; c < tm->classes; c++) {
        int64_t *init_ids = NULL;
        unsigned int n_init = 0;
        if (tm->absorb_ranking_offsets) {
          init_ids = tm->absorb_ranking + tm->absorb_ranking_offsets[c];
          n_init = (unsigned int)(tm->absorb_ranking_offsets[c + 1] - tm->absorb_ranking_offsets[c]);
        } else {
          init_ids = tm->absorb_ranking;
          n_init = tm->absorb_ranking_n;
        }
        tk_learn_init_mapping(tm, c, init_ids, n_init);
      }
      tm->dense_bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
      tm->dense_bps = tm->classes * tm->dense_bpc;
      tm->dense_n_samples = n;
      tm->managed_dense = (char *)malloc((uint64_t)n * tm->dense_bps);
      tk_learn_densify_all(tm);
    }

    ps = (tk_cvec_t *)malloc(sizeof(tk_cvec_t));
    ps->a = tm->managed_dense;
    ps->n = (uint64_t)n * tm->dense_bps;
  } else {
    lua_getfield(L, 2, "problems");
    ps = tk_cvec_peek(L, -1, "problems");
    lua_pop(L, 1);
  }

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
    if (sparse_train) free(ps);
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
  bool grouped = sparse_train ? true : tk_lua_foptboolean(L, 2, "train", "grouped", false);
  unsigned int bytes_per_class = grouped ? TK_CVEC_BITS_BYTES(tm->features * 2) : 0;
  unsigned int input_stride = grouped ? classes * bytes_per_class : 0;

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
  unsigned int shuffle_n = n;
  unsigned int **shuffles = (unsigned int **)tk_malloc(L, (size_t)max_threads * sizeof(unsigned int *));
  for (int t = 0; t < max_threads; t++)
    shuffles[t] = (unsigned int *)tk_malloc(L, shuffle_n * sizeof(unsigned int));

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

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    unsigned int *shuffle = shuffles[tid];
    unsigned int literals[TK_CVEC_BITS];
    unsigned int votes[TK_CVEC_BITS];
    for (unsigned int iter = 0; iter < max_iter; iter++) {
      if (break_flag) break;
      tk_learn_init_shuffle(shuffle, shuffle_n);
      #pragma omp for schedule(guided)
      for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
        unsigned int chunk_class = chunk / clause_chunks;
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
      if (tm->sparse_mode && tm->absorb_interval > 0 && ((iter + 1) % tm->absorb_interval == 0)) {
        #pragma omp for schedule(dynamic)
        for (unsigned int c = 0; c < tm->classes; c++)
          tk_learn_absorb_class(tm, c);
      }
      #pragma omp single
      {
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
    }
  }

  for (int t = 0; t < max_threads; t++)
    free(shuffles[t]);
  free(shuffles);
  if (skip_thresholds)
    free(skip_thresholds);
  if (sparse_train)
    free(ps);
  if (!tm->reusable)
    tk_learn_shrink(tm);
  tm->trained = true;

  return 0;
}

static inline int tk_learn_train (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  if (!tm->has_state)
    luaL_error(L, "can't train a model loaded without state");
  return tk_learn_train_regressor(L, tm);
}

static inline void _tk_learn_persist (lua_State *L, tk_learn_t *tm, FILE *fh)
{
  tk_lua_fwrite(L, "TKtm", 1, 4, fh);
  uint8_t version = 2;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  uint32_t u32;
  u32 = (uint32_t)tm->classes; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->features; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->clauses; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->clause_tolerance; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->clause_maximum; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->target; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->state_bits; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->input_bits; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->input_chunks; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->clause_chunks; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  uint64_t u64;
  u64 = (uint64_t)tm->state_chunks; tk_lua_fwrite(L, &u64, sizeof(uint64_t), 1, fh);
  u64 = (uint64_t)tm->action_chunks; tk_lua_fwrite(L, &u64, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &tm->tail_mask, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, tm->actions, 1, tm->action_chunks, fh);
  tk_lua_fwrite(L, tm->y_min, sizeof(double), tm->classes, fh);
  tk_lua_fwrite(L, tm->y_max, sizeof(double), tm->classes, fh);
  tk_lua_fwrite(L, tm->clause_tolerances, sizeof(unsigned int), tm->classes, fh);
  tk_lua_fwrite(L, tm->targets, sizeof(unsigned int), tm->classes, fh);
}

static inline int tk_learn_persist (lua_State *L)
{
  lua_settop(L, 2);
  tk_learn_t *tm = tk_learn_peek(L, 1);
  bool tostr = lua_type(L, 2) == LUA_TNIL;
  FILE *fh = tostr ? tk_lua_tmpfile(L) : tk_lua_fopen(L, tk_lua_checkstring(L, 2, "persist path"), "w");
  _tk_learn_persist(L, tm, fh);
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

static inline int tk_learn_checkpoint (lua_State *L)
{
  lua_settop(L, 2);
  tk_learn_t *tm = tk_learn_peek(L, 1);
  tk_cvec_t *checkpoint = tk_cvec_peek(L, 2, "checkpoint");
  size_t mapping_size = tm->sparse_mode ? (size_t)tm->classes * tm->features * sizeof(int64_t) : 0;
  size_t size = tm->action_chunks + mapping_size;
  if (tk_cvec_ensure(checkpoint, size) != 0)
    luaL_error(L, "failed to resize checkpoint buffer");
  checkpoint->n = size;
  char *p = checkpoint->a;
  memcpy(p, tm->actions, tm->action_chunks); p += tm->action_chunks;
  if (mapping_size > 0) { memcpy(p, tm->mapping, mapping_size); }
  return 0;
}

static inline int tk_learn_restore (lua_State *L)
{
  lua_settop(L, 2);
  tk_learn_t *tm = tk_learn_peek(L, 1);
  tk_cvec_t *checkpoint = tk_cvec_peek(L, 2, "checkpoint");
  size_t mapping_size = tm->sparse_mode ? (size_t)tm->classes * tm->features * sizeof(int64_t) : 0;
  size_t expected = tm->action_chunks + mapping_size;
  if (checkpoint->n != expected)
    luaL_error(L, "checkpoint size mismatch: expected %zu bytes, got %zu", expected, checkpoint->n);
  const char *p = checkpoint->a;
  memcpy(tm->actions, p, tm->action_chunks); p += tm->action_chunks;
  if (mapping_size > 0) {
    memcpy(tm->mapping, p, mapping_size);
    memset(tm->active, 0, (uint64_t)tm->classes * tm->active_bytes);
    for (unsigned int c = 0; c < tm->classes; c++) {
      for (unsigned int k = 0; k < tm->features; k++) {
        int64_t tok = tm->mapping[(uint64_t)c * tm->features + k];
        if (tok >= 0)
          sparse_set_active(tm, c, (unsigned int)tok);
      }
    }
    if (tm->managed_dense)
      tk_learn_densify_all(tm);
  }
  return 0;
}

static inline int tk_learn_reconfigure (lua_State *L)
{
  lua_settop(L, 2);
  tk_learn_t *tm = tk_learn_peek(L, 1);
  if (!tm->reusable)
    luaL_error(L, "reconfigure requires reusable=true at creation time");
  unsigned int new_clauses = tk_lua_fcheckunsigned(L, 2, "reconfigure", "clauses");
  unsigned int new_tolerance = tk_lua_fcheckunsigned(L, 2, "reconfigure", "clause_tolerance");
  unsigned int new_maximum = tk_lua_fcheckunsigned(L, 2, "reconfigure", "clause_maximum");
  if (new_tolerance > new_maximum)
    tk_lua_verror(L, 2, "reconfigure", "clause_tolerance", "must be <= clause_maximum");
  unsigned int new_target = tk_lua_fcheckunsigned(L, 2, "reconfigure", "target");
  double new_specificity = tk_lua_fcheckposdouble(L, 2, "reconfigure", "specificity");
  unsigned int new_features = tk_lua_foptunsigned(L, 2, "reconfigure", "features", tm->features);
  if (new_features != tm->features) {
    tm->features = new_features;
    tm->input_bits = 2 * tm->features;
    uint64_t tail_bits = tm->input_bits & (TK_CVEC_BITS - 1);
    tm->tail_mask = tail_bits ? (uint8_t)((1u << tail_bits) - 1) : 0xFF;
    tm->input_chunks = TK_CVEC_BITS_BYTES(tm->input_bits);
    if (tm->sparse_mode) {
      free(tm->mapping);
      tm->mapping = (int64_t *)malloc((uint64_t)tm->classes * tm->features * sizeof(int64_t));
      free(tm->absorb_scratch);
      tm->absorb_scratch = (uint8_t *)malloc((uint64_t)tm->absorb_threads * tm->features * 2 * sizeof(unsigned int));
    }
  }
  unsigned int new_clause_chunks = new_clauses * 2;
  new_clauses = new_clause_chunks * TK_CVEC_BITS;
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
  memset(tm->state, 0, new_state_chunks);
  tm->clauses = new_clauses;
  tm->clause_chunks = new_clause_chunks;
  tm->clause_tolerance = new_tolerance;
  tm->clause_maximum = new_maximum;
  tm->action_chunks = new_action_chunks;
  tm->state_chunks = new_state_chunks;
  tm->specificity = new_specificity;
  tm->target = new_target;
  tm->specificity_threshold = (unsigned int)((2.0 * (double)tm->features) / new_specificity);
  for (unsigned int c = 0; c < tm->classes; c++) {
    tm->clause_tolerances[c] = new_tolerance;
    tm->clause_maximums[c] = new_maximum;
    tm->specificity_thresholds[c] = tm->specificity_threshold;
    tm->targets[c] = new_target;
  }
  if (tk_lua_ftype(L, 2, "per_class_tolerances") != LUA_TNIL) {
    lua_getfield(L, 2, "per_class_tolerances");
    tk_ivec_t *v = tk_ivec_peek(L, -1, "per_class_tolerances");
    for (unsigned int c = 0; c < tm->classes; c++)
      tm->clause_tolerances[c] = (unsigned int)v->a[c];
    lua_pop(L, 1);
  }
  if (tk_lua_ftype(L, 2, "per_class_maximums") != LUA_TNIL) {
    lua_getfield(L, 2, "per_class_maximums");
    tk_ivec_t *v = tk_ivec_peek(L, -1, "per_class_maximums");
    for (unsigned int c = 0; c < tm->classes; c++)
      tm->clause_maximums[c] = (unsigned int)v->a[c];
    lua_pop(L, 1);
  }
  if (tk_lua_ftype(L, 2, "per_class_spec_thresholds") != LUA_TNIL) {
    lua_getfield(L, 2, "per_class_spec_thresholds");
    tk_ivec_t *v = tk_ivec_peek(L, -1, "per_class_spec_thresholds");
    for (unsigned int c = 0; c < tm->classes; c++)
      tm->specificity_thresholds[c] = (unsigned int)v->a[c];
    lua_pop(L, 1);
  }
  if (tk_lua_ftype(L, 2, "per_class_targets") != LUA_TNIL) {
    lua_getfield(L, 2, "per_class_targets");
    tk_ivec_t *v = tk_ivec_peek(L, -1, "per_class_targets");
    for (unsigned int c = 0; c < tm->classes; c++)
      tm->targets[c] = (unsigned int)v->a[c];
    lua_pop(L, 1);
  }
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->automata.n_chunks = tm->input_chunks;
  tm->automata.tail_mask = tm->tail_mask;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;
  tm->trained = false;
  if (tm->sparse_mode) {
    tm->absorb_interval = tk_lua_foptunsigned(L, 2, "reconfigure", "absorb_interval", tm->absorb_interval);
    tm->absorb_threshold = tk_lua_foptunsigned(L, 2, "reconfigure", "absorb_threshold", tm->absorb_threshold);
    tm->absorb_maximum = tk_lua_foptunsigned(L, 2, "reconfigure", "absorb_maximum", tm->absorb_maximum);
    tm->absorb_insert = tk_lua_foptunsigned(L, 2, "reconfigure", "absorb_insert", tm->absorb_insert);
    free(tm->managed_dense);
    tm->managed_dense = NULL;
    memset(tm->active, 0, (uint64_t)tm->classes * tm->active_bytes);
    if (tm->absorb_cursor)
      memset(tm->absorb_cursor, 0, tm->classes * sizeof(unsigned int));
  }
  return 0;
}

static inline void tk_learn_restrict_buffer (
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

static inline int tk_learn_restrict (lua_State *L)
{
  lua_settop(L, 2);
  tk_learn_t *tm = tk_learn_peek(L, 1);
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
  tk_learn_restrict_buffer(tm->actions, bytes_per_class_actions, tm->classes, new_classes, keep->a);
  tm->action_chunks = new_classes * bytes_per_class_actions;
  if (tm->state) {
    tk_learn_restrict_buffer(tm->state, bytes_per_class_state, tm->classes, new_classes, keep->a);
    tm->state_chunks = new_classes * bytes_per_class_state;
  }
  for (unsigned int i = 0; i < new_classes; i++) {
    unsigned int src = (unsigned int)keep->a[i];
    tm->clause_tolerances[i] = tm->clause_tolerances[src];
    tm->clause_maximums[i] = tm->clause_maximums[src];
    tm->specificity_thresholds[i] = tm->specificity_thresholds[src];
    tm->targets[i] = tm->targets[src];
  }
  tm->classes = new_classes;
  tm->class_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  tm->automata.n_clauses = tm->classes * tm->clauses;
  return 0;
}

static inline int tk_learn_active_features (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  if (!tm->sparse_mode)
    return luaL_error(L, "active_features requires sparse mode");
  unsigned int classes = tm->classes;
  unsigned int features = tm->features;
  tk_ivec_t *offsets = tk_ivec_create(L, classes + 1, 0, 0);
  for (unsigned int c = 0; c <= classes; c++)
    offsets->a[c] = (int64_t)c * features;
  tk_ivec_t *feat_ids = tk_ivec_create(L, (uint64_t)classes * features, 0, 0);
  for (unsigned int c = 0; c < classes; c++)
    for (unsigned int k = 0; k < features; k++)
      feat_ids->a[(uint64_t)c * features + k] = tm->mapping[(uint64_t)c * features + k];
  return 2;
}

static inline void _tk_learn_load (lua_State *L, tk_learn_t *tm, FILE *fh)
{
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKtm", 4) != 0)
    luaL_error(L, "invalid TM file (bad magic)");
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 1 && version != 2)
    luaL_error(L, "unsupported TM version %d", (int)version);
  uint32_t u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->classes = u32;
  tm->class_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->features = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->clauses = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->clause_tolerance = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->clause_maximum = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->target = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->state_bits = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->input_bits = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->input_chunks = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->clause_chunks = u32;
  uint64_t u64;
  tk_lua_fread(L, &u64, sizeof(uint64_t), 1, fh); tm->state_chunks = (size_t)u64;
  tk_lua_fread(L, &u64, sizeof(uint64_t), 1, fh); tm->action_chunks = (size_t)u64;
  tk_lua_fread(L, &tm->specificity, sizeof(double), 1, fh);
  tk_lua_fread(L, &tm->tail_mask, sizeof(uint8_t), 1, fh);
  tm->actions = (char *)tk_malloc_aligned(L, tm->action_chunks, TK_CVEC_BITS);
  tk_lua_fread(L, tm->actions, 1, tm->action_chunks, fh);
  tm->state = NULL;
  tm->y_min = (double *)tk_malloc(L, tm->classes * sizeof(double));
  tm->y_max = (double *)tk_malloc(L, tm->classes * sizeof(double));
  tk_lua_fread(L, tm->y_min, sizeof(double), tm->classes, fh);
  tk_lua_fread(L, tm->y_max, sizeof(double), tm->classes, fh);
  tm->specificity_threshold = (unsigned int)((2.0 * (double)tm->features) / tm->specificity);
  tm->clause_tolerances = (unsigned int *)tk_malloc(L, tm->classes * sizeof(unsigned int));
  tm->clause_maximums = (unsigned int *)tk_malloc(L, tm->classes * sizeof(unsigned int));
  tm->specificity_thresholds = (unsigned int *)tk_malloc(L, tm->classes * sizeof(unsigned int));
  tm->targets = (unsigned int *)tk_malloc(L, tm->classes * sizeof(unsigned int));
  if (version >= 2) {
    tk_lua_fread(L, tm->clause_tolerances, sizeof(unsigned int), tm->classes, fh);
    tk_lua_fread(L, tm->targets, sizeof(unsigned int), tm->classes, fh);
    for (unsigned int c = 0; c < tm->classes; c++) {
      tm->clause_maximums[c] = tm->clause_maximum;
      tm->specificity_thresholds[c] = tm->specificity_threshold;
    }
  } else {
    for (unsigned int c = 0; c < tm->classes; c++) {
      tm->clause_tolerances[c] = tm->clause_tolerance;
      tm->clause_maximums[c] = tm->clause_maximum;
      tm->specificity_thresholds[c] = tm->specificity_threshold;
      tm->targets[c] = tm->target;
    }
  }
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->automata.n_chunks = tm->input_chunks;
  tm->automata.state_bits = tm->state_bits;
  tm->automata.tail_mask = tm->tail_mask;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;
}

static inline int tk_learn_load (lua_State *L)
{
  lua_settop(L, 2);
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && lua_toboolean(L, 2);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  tk_learn_t *tm = tk_learn_alloc(L, false);
  _tk_learn_load(L, tm, fh);
  tk_lua_fclose(L, fh);
  return 1;
}

static luaL_Reg tk_learn_fns[] =
{
  { "create", tk_learn_create },
  { "load", tk_learn_load },
  { NULL, NULL }
};

int luaopen_santoku_learn_regressor (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_learn_fns, 0);
  lua_pushinteger(L, TK_CVEC_BITS);
  lua_setfield(L, -2, "align");
  return 1;
}
