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
#include <santoku/rvec.h>
#include <omp.h>
#include <math.h>
#include <limits.h>
#include <string.h>

#ifndef LUA_OK
#define LUA_OK 0
#endif

#define TK_LEARN_MT "santoku_learn"

static inline unsigned int tk_next_pow2 (unsigned int v) {
  if (v == 0) return 1;
  v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

typedef struct {
  uint64_t *sample_offsets;
  uint32_t *binned;
  int64_t *reverse;
  unsigned int n_samples;
} tk_token_bin_t;

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
  uint8_t *specificity_pts;

  unsigned int n_tokens;

  int64_t *csc_offsets;
  int64_t *csc_indices;

  int64_t *mapping;
  uint64_t mapping_alloc;
  uint8_t *active;
  unsigned int active_bytes;

  char *managed_dense;
  uint64_t dense_alloc;
  unsigned int dense_n_samples;
  unsigned int dense_bpc;
  unsigned int dense_bps;
  unsigned int dense_batch_size;
  unsigned int dense_batch_start;

  uint8_t *absorb_scratch;
  uint64_t scratch_alloc;
  int absorb_threads;
  unsigned int absorb_threshold;
  unsigned int absorb_maximum;
  unsigned int absorb_insert;

  int64_t *absorb_ranking;
  unsigned int absorb_ranking_n;
  int64_t *absorb_ranking_offsets;
  int64_t *absorb_ranking_global;
  unsigned int absorb_ranking_global_n;
  unsigned int absorb_ranking_limit;
  unsigned int *absorb_cursor;

  bool dense_stale;

  tk_token_bin_t cached_bin;
  const int64_t *cached_bin_tokens;
  bool has_cached_bin;

  unsigned int n_orig_dims;
  unsigned int n_code_bits;
  uint8_t *dim_prefixes;
  unsigned int code_prefix_bytes;
  bool flat_evict;
  uint32_t flat_skip_thresh;

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
      unsigned int failed = (unsigned int)failed_64 - (unsigned int)__builtin_popcount((unsigned)(garbage_act & ~input_bytes[input_chunks - 1]));
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
      unsigned int failed = (unsigned int)__builtin_popcount((unsigned)(last_act & ~last_in));
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

static inline void tk_learn_calculate_split (
  tk_learn_t *tm,
  const uint8_t *prefix,
  unsigned int prefix_bytes,
  const uint8_t *base_input,
  unsigned int *literalsp,
  unsigned int *votesp,
  unsigned int chunk,
  unsigned int tolerance,
  unsigned int empty_vote
) {
  const unsigned int input_chunks = tm->input_chunks;
  const uint8_t tail_mask = tm->tail_mask;
  const uint8_t garbage_mask = (uint8_t)~tail_mask;
  unsigned int base_bytes = input_chunks - prefix_bytes;
  for (unsigned int j = 0; j < TK_CVEC_BITS; j++) {
    unsigned int clause = chunk * TK_CVEC_BITS + j;
    const uint8_t *actions = (const uint8_t *)tk_automata_actions(&tm->automata, clause);
    uint64_t pfx_lit, pfx_fail, base_lit, base_fail;
    tk_cvec_bits_popcount_andnot_serial(actions, prefix, prefix_bytes * 8, &pfx_lit, &pfx_fail);
    tk_cvec_bits_popcount_andnot_serial(actions + prefix_bytes, base_input + prefix_bytes, base_bytes * 8, &base_lit, &base_fail);
    const uint8_t garbage_act = actions[input_chunks - 1] & garbage_mask;
    unsigned int literals = (unsigned int)(pfx_lit + base_lit) - (unsigned int)__builtin_popcount(garbage_act);
    unsigned int failed = (unsigned int)(pfx_fail + base_fail) - (unsigned int)__builtin_popcount((unsigned)(garbage_act & ~base_input[input_chunks - 1]));
    long int votes;
    if (literals == 0) {
      votes = empty_vote;
    } else {
      votes = (literals < tolerance ? (long int)literals : (long int)tolerance) - (long int)failed;
      if (votes < 0)
        votes = 0;
    }
    literalsp[j] = literals;
    votesp[j] = (unsigned int)votes;
  }
}

typedef struct {
  uint8_t *buf;
  unsigned int size;
  unsigned int cursor;
  uint8_t sp;
} tk_spec_buf_t;

static inline void tk_spec_fill (tk_spec_buf_t *sb) {
  uint8_t sp = sb->sp;
  unsigned int sz = sb->size;
  unsigned int k = 0;
#ifdef __AVX512BW__
  {
    __m512i sp64 = _mm512_set1_epi8((char)sp);
    for (; k + 8 <= sz; k += 8) {
      uint32_t r[16];
      for (int q = 0; q < 16; q++) r[q] = tk_fast_random();
      __m512i rv = _mm512_loadu_si512(r);
      uint64_t m = _mm512_cmplt_epu8_mask(rv, sp64);
      memcpy(sb->buf + k, &m, 8);
    }
  }
#elif defined(__aarch64__)
  {
    uint8x16_t sp16 = vdupq_n_u8(sp);
    static const uint8_t shift_vals[16] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    uint8x16_t shifts = vld1q_u8(shift_vals);
    for (; k + 2 <= sz; k += 2) {
      uint32_t r[4] = { tk_fast_random(), tk_fast_random(), tk_fast_random(), tk_fast_random() };
      uint8x16_t rv = vld1q_u8((const uint8_t *)r);
      uint8x16_t cmp = vcltq_u8(rv, sp16);
      uint8x16_t masked = vandq_u8(cmp, shifts);
      sb->buf[k]     = vaddv_u8(vget_low_u8(masked));
      sb->buf[k + 1] = vaddv_u8(vget_high_u8(masked));
    }
  }
#endif
  for (; k < sz; k++) {
    uint32_t r1 = tk_fast_random(), r2 = tk_fast_random();
    sb->buf[k] =
      (((uint8_t)(r1      ) < sp)     ) |
      (((uint8_t)(r1 >>  8) < sp) << 1) |
      (((uint8_t)(r1 >> 16) < sp) << 2) |
      (((uint8_t)(r1 >> 24) < sp) << 3) |
      (((uint8_t)(r2      ) < sp) << 4) |
      (((uint8_t)(r2 >>  8) < sp) << 5) |
      (((uint8_t)(r2 >> 16) < sp) << 6) |
      (((uint8_t)(r2 >> 24) < sp) << 7);
  }
  sb->cursor = 0;
}

static inline uint8_t *tk_spec_get (tk_spec_buf_t *sb, unsigned int n) {
  if (sb->cursor + n > sb->size)
    tk_spec_fill(sb);
  uint8_t *ptr = sb->buf + sb->cursor;
  sb->cursor += n;
  return ptr;
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
  unsigned int specificity_thresh,
  tk_spec_buf_t *spec_buf
) {
  unsigned int input_chunks = tm->input_chunks;
  bool output = votes[clause_idx] > 0;
  unsigned int clause_id = chunk * TK_CVEC_BITS + clause_idx;
  if (positive_feedback) {
    if (output) {
      if (literals[clause_idx] < max_literals)
        tk_automata_inc_and_dec_not_excluded(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
      else
        tk_automata_dec_not_excluded(&tm->automata, clause_id, (uint8_t*)input, input_chunks);
    } else {
      uint8_t *bmap = tk_spec_get(spec_buf, input_chunks);
      if (input_chunks > 0) bmap[input_chunks - 1] &= tm->tail_mask;
      tk_automata_dec(&tm->automata, clause_id, bmap, input_chunks);
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
static inline int tk_learn_regress (lua_State *);
static inline int tk_learn_regress_mae (lua_State *);
static inline int tk_learn_regress_nmae (lua_State *);
static inline int tk_learn_encode (lua_State *);
static inline int tk_learn_encode_hamming (lua_State *);
static inline int tk_learn_destroy (lua_State *L);
static inline int tk_learn_persist (lua_State *);
static inline int tk_learn_checkpoint (lua_State *);
static inline int tk_learn_restore (lua_State *);
static inline int tk_learn_reconfigure (lua_State *);
static inline int tk_learn_restrict (lua_State *);
static inline int tk_learn_label (lua_State *);
static inline int tk_learn_label_f1 (lua_State *);

static luaL_Reg tk_learn_mt_fns[] =
{
  { "train", tk_learn_train },
  { "regress", tk_learn_regress },
  { "regress_mae", tk_learn_regress_mae },
  { "regress_nmae", tk_learn_regress_nmae },
  { "encode", tk_learn_encode },
  { "encode_hamming", tk_learn_encode_hamming },
  { "label", tk_learn_label },
  { "label_f1", tk_learn_label_f1 },
  { "destroy", tk_learn_destroy },
  { "persist", tk_learn_persist },
  { "checkpoint", tk_learn_checkpoint },
  { "restore", tk_learn_restore },
  { "reconfigure", tk_learn_reconfigure },
  { "restrict", tk_learn_restrict },
  { NULL, NULL }
};

static inline tk_learn_t *tk_learn_alloc (lua_State *L, bool has_state)
{
  tk_learn_t *tm = tk_lua_newuserdata(L, tk_learn_t, TK_LEARN_MT, tk_learn_mt_fns, tk_learn_destroy);
  tm->has_state = has_state;
  tm->n_tokens = 0;
  tm->csc_offsets = NULL;
  tm->csc_indices = NULL;
  tm->mapping = NULL;
  tm->mapping_alloc = 0;
  tm->active = NULL;
  tm->active_bytes = 0;
  tm->managed_dense = NULL;
  tm->dense_alloc = 0;
  tm->dense_stale = false;
  tm->dense_n_samples = 0;
  tm->dense_bpc = 0;
  tm->dense_bps = 0;
  tm->dense_batch_size = 0;
  tm->dense_batch_start = 0;
  tm->absorb_scratch = NULL;
  tm->scratch_alloc = 0;
  tm->absorb_threads = 0;
  tm->absorb_threshold = 0;
  tm->absorb_maximum = 0;
  tm->absorb_insert = 1;
  tm->absorb_ranking = NULL;
  tm->absorb_ranking_n = 0;
  tm->absorb_ranking_offsets = NULL;
  tm->absorb_ranking_global = NULL;
  tm->absorb_ranking_global_n = 0;
  tm->absorb_ranking_limit = 0;
  tm->absorb_cursor = NULL;
  tm->has_cached_bin = false;
  tm->cached_bin_tokens = NULL;
  tm->n_orig_dims = 0;
  tm->n_code_bits = 0;
  tm->dim_prefixes = NULL;
  tm->code_prefix_bytes = 0;
  tm->flat_evict = false;
  tm->flat_skip_thresh = 0;
  tm->clause_tolerances = NULL;
  tm->clause_maximums = NULL;
  tm->specificity_thresholds = NULL;
  tm->targets = NULL;
  tm->specificity_pts = NULL;
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
  tm->specificity_pts = (uint8_t *)tk_malloc(L, outputs * sizeof(uint8_t));
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

static inline void tk_learn_recompute_specificity_pts (tk_learn_t *tm) {
  for (unsigned int c = 0; c < tm->classes; c++) {
    double p = 1.0 - exp(-(double)tm->specificity_thresholds[c] / (double)tm->input_bits);
    tm->specificity_pts[c] = (uint8_t)(p * 255.0);
  }
}

static void tk_learn_apply_fractions (tk_learn_t *tm, lua_State *L, int arg_idx, const char *fn_name)
{
  double cmf = tk_lua_fcheckdouble(L, arg_idx, (char *)fn_name, "clause_maximum_fraction");
  double ctf = tk_lua_fcheckdouble(L, arg_idx, (char *)fn_name, "clause_tolerance_fraction");
  double tf = tk_lua_fcheckdouble(L, arg_idx, (char *)fn_name, "target_fraction");
  double sf = tk_lua_fcheckdouble(L, arg_idx, (char *)fn_name, "specificity_fraction");
  double amf = tk_lua_foptnumber(L, arg_idx, (char *)fn_name, "absorb_maximum_fraction", 0.0);
  double arf = tk_lua_foptnumber(L, arg_idx, (char *)fn_name, "absorb_ranking_fraction", 0.0);

  unsigned int features = tm->features;
  unsigned int input_bits = tm->input_bits;
  unsigned int n_tokens = tm->n_tokens;

  unsigned int cm = (unsigned int)fmax(1.0, floor(cmf * input_bits + 0.5));
  unsigned int ct = (unsigned int)fmax(1.0, floor(ctf * cm + 0.5));
  unsigned int tgt = (unsigned int)fmax(1.0, floor(tf * 8.0 * ct + 0.5));
  double spec = fmax(1.0, floor(1.0 / sf + 0.5));
  unsigned int spec_thresh = (unsigned int)floor(2.0 * features / spec);
  unsigned int am = amf > 0 ? (unsigned int)fmax(1.0, floor(amf * features + 0.5)) : 0;
  unsigned int arl = arf > 0 ? (unsigned int)fmax(1.0, floor(arf * n_tokens + 0.5)) : 0;

  tm->clause_maximum = cm;
  tm->clause_tolerance = ct;
  tm->target = tgt;
  tm->specificity = spec;
  tm->specificity_threshold = spec_thresh;
  tm->absorb_maximum = am;
  tm->absorb_ranking_limit = arl;

  double alpha_tol = tk_lua_foptnumber(L, arg_idx, (char *)fn_name, "alpha_tolerance", 0.0);
  double alpha_max = tk_lua_foptnumber(L, arg_idx, (char *)fn_name, "alpha_maximum", 0.0);
  double alpha_spec = tk_lua_foptnumber(L, arg_idx, (char *)fn_name, "alpha_specificity", 0.0);
  double alpha_tgt = tk_lua_foptnumber(L, arg_idx, (char *)fn_name, "alpha_target", 0.0);

  bool has_alpha = alpha_tol != 0.0 || alpha_max != 0.0 || alpha_spec != 0.0 || alpha_tgt != 0.0;
  bool has_weights = tk_lua_ftype(L, arg_idx, "output_weights") != LUA_TNIL;

  if (has_alpha && has_weights) {
    lua_getfield(L, arg_idx, "output_weights");
    tk_dvec_t *weights = tk_dvec_peek(L, -1, "output_weights");
    double w_min = weights->a[0], w_max = weights->a[0];
    for (unsigned int c = 1; c < tm->classes; c++) {
      if (weights->a[c] < w_min) w_min = weights->a[c];
      if (weights->a[c] > w_max) w_max = weights->a[c];
    }
    double w_range = w_max - w_min;
    for (unsigned int c = 0; c < tm->classes; c++) {
      double w_norm = w_range > 0 ? (weights->a[c] - w_min) / w_range : 0.5;
      double factor = w_norm - 0.5;
      unsigned int ct_c = (unsigned int)fmax(1.0, floor(ct * exp(alpha_tol * factor) + 0.5));
      unsigned int cm_c = (unsigned int)fmax(1.0, floor(cm * exp(alpha_max * factor) + 0.5));
      unsigned int tgt_c = (unsigned int)fmax(1.0, floor(tgt * exp(alpha_tgt * factor) + 0.5));
      double spec_c = fmax(1.0, spec * exp(alpha_spec * factor));
      unsigned int st_c = (unsigned int)floor(2.0 * features / spec_c);
      if (ct_c > cm_c) ct_c = cm_c;
      if (tgt_c > 8 * ct_c) tgt_c = 8 * ct_c;
      tm->clause_tolerances[c] = ct_c;
      tm->clause_maximums[c] = cm_c;
      tm->specificity_thresholds[c] = st_c;
      tm->targets[c] = tgt_c;
    }
    lua_pop(L, 1);
  } else {
    for (unsigned int c = 0; c < tm->classes; c++) {
      tm->clause_tolerances[c] = ct;
      tm->clause_maximums[c] = cm;
      tm->specificity_thresholds[c] = spec_thresh;
      tm->targets[c] = tgt;
    }
  }

  tk_learn_recompute_specificity_pts(tm);
}

static inline void tk_learn_create_impl (lua_State *L)
{
  tk_learn_t *tm = tk_learn_alloc(L, true);
  lua_insert(L, 1);
  unsigned int raw_outputs = tk_lua_foptunsigned(L, 2, "create", "outputs", 1);
  bool flat = tk_lua_foptboolean(L, 2, "create", "flat", false);
  unsigned int init_outputs = flat ? 1 : raw_outputs;
  unsigned int user_features = tk_lua_fcheckunsigned(L, 2, "create", "features");
  unsigned int n_code_bits = 0;
  const char *flat_enc = flat ? tk_lua_foptstring(L, 2, "create", "flat_encoding", "hadamard") : NULL;
  tk_ivec_t *flat_enc_cvec = NULL;
  if (flat) {
    if (strcmp(flat_enc, "hadamard") == 0) {
      n_code_bits = tk_next_pow2(raw_outputs);
      if (n_code_bits < 2) n_code_bits = 2;
    } else if (strcmp(flat_enc, "thermometer") == 0) {
      n_code_bits = raw_outputs > 1 ? raw_outputs - 1 : 1;
    } else if (strcmp(flat_enc, "cvec") == 0) {
      n_code_bits = tk_lua_fcheckunsigned(L, 2, "create", "flat_encoding_dims");
      lua_getfield(L, 2, "flat_encoding_cvec");
      flat_enc_cvec = tk_ivec_peek(L, -1, "flat_encoding_cvec");
      lua_pop(L, 1);
    } else {
      luaL_error(L, "unknown flat_encoding: %s", flat_enc);
    }
  }
  tk_learn_init(L, tm,
      init_outputs,
      user_features + n_code_bits,
      tk_lua_fcheckunsigned(L, 2, "create", "clauses"),
      1, 1,
      tk_lua_foptunsigned(L, 2, "create", "state", 8),
      1, 1.0);
  if (flat) {
    tm->flat_evict = tk_lua_foptboolean(L, 2, "create", "flat_evict", false);
    double flat_skip = tk_lua_foptnumber(L, 2, "create", "flat_skip", 0.0);
    tm->flat_skip_thresh = (uint32_t)(flat_skip * 4294967295.0);
    tm->n_orig_dims = raw_outputs;
    tm->n_code_bits = n_code_bits;
    tm->code_prefix_bytes = TK_CVEC_BITS_BYTES(n_code_bits * 2);
    tm->dim_prefixes = (uint8_t *)malloc((uint64_t)tm->n_orig_dims * tm->code_prefix_bytes);
    unsigned int src_row_bytes = TK_CVEC_BITS_BYTES(n_code_bits);
    for (unsigned int d = 0; d < tm->n_orig_dims; d++) {
      uint8_t *prefix = tm->dim_prefixes + (uint64_t)d * tm->code_prefix_bytes;
      memset(prefix, 0xAA, tm->code_prefix_bytes);
      if (strcmp(flat_enc, "hadamard") == 0) {
        for (unsigned int k = 0; k < n_code_bits; k++) {
          if (__builtin_popcount(d & k) % 2 == 0) {
            unsigned int byte_idx = (2 * k) / 8;
            unsigned int bit_pos = (2 * k) % 8;
            prefix[byte_idx] = (prefix[byte_idx] & ~(0x3 << bit_pos)) | (uint8_t)(0x1 << bit_pos);
          }
        }
      } else if (strcmp(flat_enc, "thermometer") == 0) {
        for (unsigned int k = 0; k < n_code_bits; k++) {
          if (k < d) {
            unsigned int byte_idx = (2 * k) / 8;
            unsigned int bit_pos = (2 * k) % 8;
            prefix[byte_idx] = (prefix[byte_idx] & ~(0x3 << bit_pos)) | (uint8_t)(0x1 << bit_pos);
          }
        }
      } else {
        const uint8_t *src = (const uint8_t *)flat_enc_cvec->a + (uint64_t)d * src_row_bytes;
        for (unsigned int k = 0; k < n_code_bits; k++) {
          if (src[k / 8] & (1 << (k % 8))) {
            unsigned int byte_idx = (2 * k) / 8;
            unsigned int bit_pos = (2 * k) % 8;
            prefix[byte_idx] = (prefix[byte_idx] & ~(0x3 << bit_pos)) | (uint8_t)(0x1 << bit_pos);
          }
        }
      }
    }
  }
  tm->reusable = tk_lua_foptboolean(L, 2, "create", "reusable", false);
  tm->dense_batch_size = tk_lua_foptunsigned(L, 2, "create", "class_batch", 0);
  tm->n_tokens = tk_lua_fcheckunsigned(L, 2, "create", "n_tokens");
  tm->absorb_threshold = tk_lua_foptunsigned(L, 2, "create", "absorb_threshold", 0);
  tm->absorb_insert = tk_lua_foptunsigned(L, 2, "create", "absorb_insert", tm->absorb_threshold + 1);
  tk_learn_apply_fractions(tm, L, 2, "create");
  tm->active_bytes = TK_CVEC_BITS_BYTES(tm->n_tokens + tm->n_code_bits);
  tm->active = (uint8_t *)calloc((uint64_t)tm->classes * tm->active_bytes, 1);
  tm->mapping_alloc = (uint64_t)tm->classes * tm->features * sizeof(int64_t);
  tm->mapping = (int64_t *)malloc(tm->mapping_alloc);
  tm->absorb_threads = omp_get_max_threads();
  tm->scratch_alloc = (uint64_t)tm->absorb_threads * tm->features * 2 * sizeof(unsigned int);
  tm->absorb_scratch = (uint8_t *)malloc(tm->scratch_alloc);
  tk_learn_recompute_specificity_pts(tm);
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

static inline void tk_learn_bin_free (tk_token_bin_t *bin) {
  free(bin->reverse);
  free(bin->binned);
  free(bin->sample_offsets);
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
  free(tm->specificity_pts); tm->specificity_pts = NULL;
  tm->csc_offsets = NULL;
  tm->csc_indices = NULL;
  free(tm->mapping); tm->mapping = NULL;
  free(tm->active); tm->active = NULL;
  free(tm->dim_prefixes); tm->dim_prefixes = NULL;
  free(tm->managed_dense); tm->managed_dense = NULL;
  free(tm->absorb_scratch); tm->absorb_scratch = NULL;
  free(tm->absorb_cursor); tm->absorb_cursor = NULL;
  free(tm->absorb_ranking_offsets); tm->absorb_ranking_offsets = NULL;
  if (tm->absorb_ranking_global != tm->absorb_ranking)
    free(tm->absorb_ranking_global);
  tm->absorb_ranking_global = NULL;
  free(tm->absorb_ranking); tm->absorb_ranking = NULL;
  if (tm->has_cached_bin) {
    tk_learn_bin_free(&tm->cached_bin);
    tm->has_cached_bin = false;
  }
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
  for (unsigned int k = 0; k < tm->n_code_bits && filled < features; k++) {
    tm->mapping[(uint64_t)c * features + filled] = (int64_t)(n_tokens + k);
    sparse_set_active(tm, c, n_tokens + k);
    filled++;
  }
  for (unsigned int i = 0; i < n_init && filled < features; i++) {
    unsigned int tok = (unsigned int)init_ids[i];
    if (tok >= n_tokens || sparse_is_active(tm, c, tok)) continue;
    tm->mapping[(uint64_t)c * features + filled] = (int64_t)tok;
    sparse_set_active(tm, c, tok);
    filled++;
  }
  uint8_t *active_base = tm->active + (uint64_t)c * tm->active_bytes;
  unsigned int active_pop = (unsigned int)tk_cvec_bits_popcount_serial(active_base, n_tokens);
  while (filled < features && active_pop < n_tokens) {
    unsigned int tok = tk_fast_random() % n_tokens;
    if (sparse_is_active(tm, c, tok)) continue;
    tm->mapping[(uint64_t)c * features + filled] = (int64_t)tok;
    sparse_set_active(tm, c, tok);
    filled++;
    active_pop++;
  }
  for (unsigned int k = filled; k < features; k++)
    tm->mapping[(uint64_t)c * features + k] = -1;
}

static inline void tk_learn_densify_slot (tk_learn_t *tm, unsigned int c, unsigned int slot, unsigned int new_tok, int64_t old_tok) {
  unsigned int bpc = tm->dense_bpc;
  unsigned int n_samples = tm->dense_n_samples;
  uint8_t *out = (uint8_t *)tm->managed_dense;
  unsigned int bit_pos = (2 * slot) % 8;
  uint8_t pair_mask = (uint8_t)(0x3 << bit_pos);
  uint8_t absent_bits = (uint8_t)(0x2 << bit_pos);
  uint8_t present_bits = (uint8_t)(0x1 << bit_pos);
  unsigned int local_byte = (2 * slot) / 8;
  unsigned int dense_c = (tm->dense_batch_size > 0) ? (c - tm->dense_batch_start) : c;
  uint64_t class_base = (uint64_t)dense_c * n_samples * bpc;

  if (old_tok >= 0) {
    int64_t csc_start = tm->csc_offsets[old_tok];
    int64_t csc_end = tm->csc_offsets[old_tok + 1];
    for (int64_t i = csc_start; i < csc_end; i++) {
      uint32_t s = (uint32_t)tm->csc_indices[i];
      uint8_t *byte = out + class_base + (uint64_t)s * bpc + local_byte;
      *byte = (*byte & ~pair_mask) | absent_bits;
    }
  }

  int64_t csc_start = tm->csc_offsets[new_tok];
  int64_t csc_end = tm->csc_offsets[new_tok + 1];
  for (int64_t i = csc_start; i < csc_end; i++) {
    uint32_t s = (uint32_t)tm->csc_indices[i];
    uint8_t *byte = out + class_base + (uint64_t)s * bpc + local_byte;
    *byte = (*byte & ~pair_mask) | present_bits;
  }
}

static inline void tk_learn_densify_batch (tk_learn_t *tm, unsigned int batch_start, unsigned int batch_size) {
  unsigned int features = tm->features;
  unsigned int bpc = tm->dense_bpc;
  unsigned int n_samples = tm->dense_n_samples;
  uint8_t *out = (uint8_t *)tm->managed_dense;
  uint64_t class_stride = (uint64_t)n_samples * bpc;

  tm->dense_batch_start = batch_start;

  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (unsigned int ci = 0; ci < batch_size; ci++)
      memset(out + (uint64_t)ci * class_stride, 0xAA, class_stride);
    #pragma omp for schedule(dynamic)
    for (unsigned int c = batch_start; c < batch_start + batch_size; c++) {
      for (unsigned int k = 0; k < features; k++) {
        int64_t mtok = tm->mapping[(uint64_t)c * features + k];
        if (mtok < 0 || (unsigned int)mtok >= tm->n_tokens) continue;
        tk_learn_densify_slot(tm, c, k, (unsigned int)mtok, -1);
      }
    }
  }
}

static inline void tk_learn_densify_all (tk_learn_t *tm) {
  tm->dense_batch_start = 0;
  tk_learn_densify_batch(tm, 0, tm->classes);
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
  if (features >= n_tokens) return 0;
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
  unsigned int code_slots = tm->flat_evict ? 0 : tm->n_code_bits;
  for (unsigned int k = 0; k < features; k++) {
    if (k < code_slots) continue;
    if (max_states[k] <= threshold)
      eligible[n_eligible++] = k;
  }

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
  if (tm->absorb_ranking_limit > 0 && rank_len > tm->absorb_ranking_limit)
    rank_len = tm->absorb_ranking_limit;

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
    if (!(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes))
      tk_learn_densify_slot(tm, c, k, new_tok, old_tok);
    tk_learn_reset_slot_automata(tm, c, k);
    n_replaced++;
  }

  return n_replaced;
}

static inline tk_token_bin_t tk_learn_bin_tokens (tk_learn_t *tm, tk_ivec_t *tokens, unsigned int n_samples) {
  unsigned int n_tokens = tm->n_tokens;
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
  return (tk_token_bin_t){ sample_offsets, binned, reverse, n_samples };
}

static inline tk_token_bin_t *tk_learn_get_bin (tk_learn_t *tm, tk_ivec_t *tokens, unsigned int n_samples) {
  if (tm->has_cached_bin && tm->cached_bin_tokens == tokens->a && tm->cached_bin.n_samples == n_samples)
    return &tm->cached_bin;
  if (tm->has_cached_bin)
    tk_learn_bin_free(&tm->cached_bin);
  tm->cached_bin = tk_learn_bin_tokens(tm, tokens, n_samples);
  tm->cached_bin_tokens = tokens->a;
  tm->has_cached_bin = true;
  return &tm->cached_bin;
}

static inline void tk_learn_densify_tokens_range (
  tk_learn_t *tm, tk_token_bin_t *bin,
  char *out, unsigned int batch_start, unsigned int batch_size
) {
  unsigned int features = tm->features;
  unsigned int n_tokens = tm->n_tokens;
  unsigned int bpc = TK_CVEC_BITS_BYTES(features * 2);
  unsigned int n_samples = bin->n_samples;
  uint64_t class_stride = (uint64_t)n_samples * bpc;
  uint8_t *data = (uint8_t *)out;
  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (unsigned int ci = 0; ci < batch_size; ci++)
      memset(data + ci * class_stride, 0xAA, class_stride);
    int64_t *local_reverse = (int64_t *)malloc((uint64_t)n_tokens * sizeof(int64_t));
    memset(local_reverse, 0xFF, (uint64_t)n_tokens * sizeof(int64_t));
    #pragma omp for schedule(dynamic)
    for (unsigned int ci = 0; ci < batch_size; ci++) {
      unsigned int c = batch_start + ci;
      int64_t *cm = tm->mapping + (uint64_t)c * features;
      for (unsigned int k = 0; k < features; k++)
        if (cm[k] >= 0 && cm[k] < (int64_t)n_tokens) local_reverse[cm[k]] = (int64_t)k;
      for (unsigned int s = 0; s < n_samples; s++) {
        uint8_t *base = data + ci * class_stride + (uint64_t)s * bpc;
        for (uint64_t bi = bin->sample_offsets[s]; bi < bin->sample_offsets[s + 1]; bi++) {
          uint32_t tok = bin->binned[bi];
          int64_t slot = local_reverse[tok];
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
        if (cm[k] >= 0 && cm[k] < (int64_t)n_tokens) local_reverse[cm[k]] = -1;
    }
    free(local_reverse);
  }
}

static inline char *tk_learn_densify_tokens (tk_learn_t *tm, tk_ivec_t *tokens, unsigned int n_samples) {
  unsigned int classes = tm->classes;
  tk_token_bin_t bin = tk_learn_bin_tokens(tm, tokens, n_samples);
  unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
  char *out = (char *)malloc((uint64_t)n_samples * classes * bpc);
  tk_learn_densify_tokens_range(tm, &bin, out, 0, classes);
  tk_learn_bin_free(&bin);
  return out;
}

static inline int tk_learn_predict_regressor (lua_State *L, tk_learn_t *tm) {
  lua_settop(L, 5);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  tk_dvec_t *out_buf = lua_isnil(L, 5) ? NULL : tk_dvec_peek(L, 5, "output");

  const unsigned int classes = tm->classes;
  unsigned int out_cols = tm->n_orig_dims > 0 ? tm->n_orig_dims : classes;
  size_t n_elems = (size_t)n * out_cols;
  double *regression_out;
  if (out_buf) {
    tk_dvec_ensure(out_buf, n_elems);
    out_buf->n = n_elems;
    regression_out = out_buf->a;
  } else {
    size_t needed = n_elems * sizeof(double);
    if (needed > tm->regression_out_len) {
      tm->regression_out = (double *)tk_realloc(L, tm->regression_out, needed);
      tm->regression_out_len = needed;
    }
    regression_out = tm->regression_out;
  }
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int *per_class_targets = tm->targets;
  const double *y_min = tm->y_min;
  const double *y_max = tm->y_max;

  bool use_batching = tm->dense_batch_size > 0 && tm->dense_batch_size < classes;

  if (tm->n_orig_dims > 0) {
    unsigned int n_dims = tm->n_orig_dims;
    unsigned int cpb = tm->code_prefix_bytes;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    char *base_ptr;
    char *temp_dense = NULL;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      base_ptr = tm->managed_dense;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "regress", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      base_ptr = temp_dense;
    }
    double y_range = y_max[0] - y_min[0];
    double scale = 1.0 / ((double)clause_chunks * per_class_targets[0]);
    #pragma omp parallel for schedule(static)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      const uint8_t *base_input = (const uint8_t *)(base_ptr + (uint64_t)s * bpc);
      for (unsigned int d = 0; d < n_dims; d++) {
        const uint8_t *prefix = tm->dim_prefixes + (uint64_t)d * cpb;
        long int total_vote = 0;
        for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
          tk_learn_calculate_split(tm, prefix, cpb, base_input, literals, votes_buf, chunk, tm->clause_tolerances[0], 0);
          long int cv = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++) cv += votes_buf[j];
          if (cv > (long int)tm->targets[0]) cv = (long int)tm->targets[0];
          if ((chunk % clause_chunks) >= clause_chunks / 2) cv = -cv;
          total_vote += cv;
        }
        regression_out[s * n_dims + d] = ((double)total_vote * scale + 0.5) * y_range + y_min[0];
      }
    }
    free(temp_dense);
  } else if (use_batching) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "regress", "n_samples");
    unsigned int B = tm->dense_batch_size;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    tk_token_bin_t *bin = tk_learn_get_bin(tm, tokens, n_tok);
    bool own_buf = !tm->managed_dense || tm->dense_n_samples < n_tok;
    char *buf = own_buf ? (char *)malloc((uint64_t)n_tok * B * bpc) : tm->managed_dense;
    for (unsigned int batch_start = 0; batch_start < classes; batch_start += B) {
      unsigned int actual_batch = (batch_start + B <= classes) ? B : (classes - batch_start);
      uint64_t batch_class_stride = (uint64_t)n_tok * bpc;
      tk_learn_densify_tokens_range(tm, bin, buf, batch_start, actual_batch);
      #pragma omp parallel for schedule(static)
      for (unsigned int s = 0; s < n; s++) {
        unsigned int literals[TK_CVEC_BITS];
        unsigned int votes_buf[TK_CVEC_BITS];
        for (unsigned int ci = 0; ci < actual_batch; ci++) {
          unsigned int c = batch_start + ci;
          char *input = buf + ci * batch_class_stride + s * bpc;
          double y_range = y_max[c] - y_min[c];
          double scale = 1.0 / ((double)clause_chunks * per_class_targets[c]);
          long int class_vote = 0;
          for (unsigned int cc = 0; cc < clause_chunks; cc++) {
            unsigned int chunk = c * clause_chunks + cc;
            tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
            long int chunk_vote = 0;
            for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
                chunk_vote += votes_buf[j];
            if (chunk_vote > (long int)tm->targets[c]) chunk_vote = (long int)tm->targets[c];
            if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
            class_vote += chunk_vote;
          }
          regression_out[s * classes + c] = ((double)class_vote * scale + 0.5) * y_range + y_min[c];
        }
      }
    }
    if (own_buf) free(buf);
  } else {
    char *temp_dense = NULL;
    unsigned int class_n = 0;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      class_n = n;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "regress", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      class_n = n_tok;
    }
    const unsigned int bytes_per_class = TK_CVEC_BITS_BYTES(tm->features * 2);
    const uint64_t class_stride = (uint64_t)class_n * bytes_per_class;
    char *base_ptr = reuse_dense ? tm->managed_dense : temp_dense;
    #pragma omp parallel for schedule(static)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      for (unsigned int c = 0; c < classes; c++) {
        char *input = base_ptr + c * class_stride + s * bytes_per_class;
        double y_range = y_max[c] - y_min[c];
        double scale = 1.0 / ((double)clause_chunks * per_class_targets[c]);
        long int class_vote = 0;
        for (unsigned int cc = 0; cc < clause_chunks; cc++) {
          unsigned int chunk = c * clause_chunks + cc;
          tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
          long int chunk_vote = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
              chunk_vote += votes_buf[j];
          if (chunk_vote > (long int)tm->targets[c]) chunk_vote = (long int)tm->targets[c];
          if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
          class_vote += chunk_vote;
        }
        regression_out[s * classes + c] = ((double)class_vote * scale + 0.5) * y_range + y_min[c];
      }
    }
    free(temp_dense);
  }

  if (out_buf) {
    lua_pushvalue(L, 5);
  } else {
    tk_dvec_t *out = tk_dvec_create(L, n_elems, 0, 0);
    memcpy(out->a, regression_out, n_elems * sizeof(double));
  }
  return 1;
}

static inline int tk_learn_label (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  lua_settop(L, 4);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int k = tk_lua_checkunsigned(L, 4, "k");
  const unsigned int classes = tm->classes;
  const unsigned int clause_chunks = tm->clause_chunks;
  unsigned int out_classes = tm->n_orig_dims > 0 ? tm->n_orig_dims : classes;
  if (k > out_classes) k = out_classes;
  if (k < 1) k = 1;
  tk_ivec_t *offsets = tk_ivec_create(L, (uint64_t)(n + 1), NULL, NULL);
  int off_idx = lua_gettop(L);
  tk_ivec_t *labels = tk_ivec_create(L, (uint64_t)n * k, NULL, NULL);
  int lab_idx = lua_gettop(L);
  tk_dvec_t *scores = tk_dvec_create(L, (uint64_t)n * k, NULL, NULL);
  int sco_idx = lua_gettop(L);
  for (unsigned int i = 0; i <= n; i++)
    offsets->a[i] = (int64_t)i * k;
  bool use_batching = tm->dense_batch_size > 0 && tm->dense_batch_size < classes;
  if (tm->n_orig_dims > 0) {
    unsigned int n_dims = tm->n_orig_dims;
    unsigned int cpb = tm->code_prefix_bytes;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    char *base_ptr;
    char *temp_dense_buf = NULL;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      base_ptr = tm->managed_dense;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "label", "n_samples");
      temp_dense_buf = tk_learn_densify_tokens(tm, tokens, n_tok);
      base_ptr = temp_dense_buf;
    }
    #pragma omp parallel
    {
      tk_rvec_t heap = { .n = 0, .m = k, .lua_managed = false,
                         .a = (tk_rank_t *)malloc(k * sizeof(tk_rank_t)) };
      #pragma omp for schedule(static)
      for (unsigned int s = 0; s < n; s++) {
        unsigned int literals[TK_CVEC_BITS];
        unsigned int votes_buf[TK_CVEC_BITS];
        const uint8_t *base_input = (const uint8_t *)(base_ptr + (uint64_t)s * bpc);
        heap.n = 0;
        for (unsigned int d = 0; d < n_dims; d++) {
          const uint8_t *prefix = tm->dim_prefixes + (uint64_t)d * cpb;
          long int total_vote = 0;
          for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
            tk_learn_calculate_split(tm, prefix, cpb, base_input, literals, votes_buf, chunk, tm->clause_tolerances[0], 0);
            long int cv = 0;
            for (unsigned int j = 0; j < TK_CVEC_BITS; j++) cv += votes_buf[j];
            if ((chunk % clause_chunks) >= clause_chunks / 2) cv = -cv;
            total_vote += cv;
          }
          tk_rvec_hmin(&heap, k, tk_rank((int64_t)d, (double)total_vote));
        }
        tk_rvec_desc(&heap, 0, heap.n);
        uint64_t base = (uint64_t)s * k;
        for (uint64_t j = 0; j < heap.n; j++) {
          labels->a[base + j] = heap.a[j].i;
          scores->a[base + j] = heap.a[j].d;
        }
      }
      free(heap.a);
    }
    free(temp_dense_buf);
  } else if (use_batching) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "label", "n_samples");
    unsigned int B = tm->dense_batch_size;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    tk_token_bin_t *bin = tk_learn_get_bin(tm, tokens, n_tok);
    bool own_buf = !tm->managed_dense || tm->dense_n_samples < n_tok;
    char *buf = own_buf ? (char *)malloc((uint64_t)n_tok * B * bpc) : tm->managed_dense;
    long int *all_votes = (long int *)calloc((uint64_t)n * classes, sizeof(long int));
    for (unsigned int batch_start = 0; batch_start < classes; batch_start += B) {
      unsigned int actual_batch = (batch_start + B <= classes) ? B : (classes - batch_start);
      uint64_t batch_class_stride = (uint64_t)n_tok * bpc;
      tk_learn_densify_tokens_range(tm, bin, buf, batch_start, actual_batch);
      #pragma omp parallel for schedule(static)
      for (unsigned int s = 0; s < n; s++) {
        unsigned int literals[TK_CVEC_BITS];
        unsigned int votes_buf[TK_CVEC_BITS];
        for (unsigned int ci = 0; ci < actual_batch; ci++) {
          unsigned int c = batch_start + ci;
          char *input = buf + ci * batch_class_stride + s * bpc;
          long int class_vote = 0;
          for (unsigned int cc = 0; cc < clause_chunks; cc++) {
            unsigned int chunk = c * clause_chunks + cc;
            tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
            long int chunk_vote = 0;
            for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
              chunk_vote += votes_buf[j];
            if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
            class_vote += chunk_vote;
          }
          all_votes[(uint64_t)s * classes + c] = class_vote;
        }
      }
    }
    #pragma omp parallel
    {
      tk_rvec_t heap = { .n = 0, .m = k, .lua_managed = false,
                         .a = (tk_rank_t *)malloc(k * sizeof(tk_rank_t)) };
      #pragma omp for schedule(static)
      for (unsigned int s = 0; s < n; s++) {
        heap.n = 0;
        for (unsigned int c = 0; c < classes; c++)
          tk_rvec_hmin(&heap, k, tk_rank((int64_t)c, (double)all_votes[(uint64_t)s * classes + c]));
        tk_rvec_desc(&heap, 0, heap.n);
        uint64_t base = (uint64_t)s * k;
        for (uint64_t j = 0; j < heap.n; j++) {
          labels->a[base + j] = heap.a[j].i;
          scores->a[base + j] = heap.a[j].d;
        }
      }
      free(heap.a);
    }
    free(all_votes);
    if (own_buf) free(buf);
  } else {
    char *temp_dense = NULL;
    unsigned int class_n = 0;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      class_n = n;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "label", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      class_n = n_tok;
    }
    const unsigned int bytes_per_class = TK_CVEC_BITS_BYTES(tm->features * 2);
    const uint64_t class_stride = (uint64_t)class_n * bytes_per_class;
    char *base_ptr = reuse_dense ? tm->managed_dense : temp_dense;
    #pragma omp parallel
    {
      tk_rvec_t heap = { .n = 0, .m = k, .lua_managed = false,
                         .a = (tk_rank_t *)malloc(k * sizeof(tk_rank_t)) };
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      #pragma omp for schedule(static)
      for (unsigned int s = 0; s < n; s++) {
        heap.n = 0;
        for (unsigned int c = 0; c < classes; c++) {
          char *input = base_ptr + c * class_stride + s * bytes_per_class;
          long int class_vote = 0;
          for (unsigned int cc = 0; cc < clause_chunks; cc++) {
            unsigned int chunk = c * clause_chunks + cc;
            tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
            long int chunk_vote = 0;
            for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
              chunk_vote += votes_buf[j];
            if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
            class_vote += chunk_vote;
          }
          tk_rvec_hmin(&heap, k, tk_rank((int64_t)c, (double)class_vote));
        }
        tk_rvec_desc(&heap, 0, heap.n);
        uint64_t base = (uint64_t)s * k;
        for (uint64_t j = 0; j < heap.n; j++) {
          labels->a[base + j] = heap.a[j].i;
          scores->a[base + j] = heap.a[j].d;
        }
      }
      free(heap.a);
    }
    free(temp_dense);
  }
  lua_pushvalue(L, off_idx);
  lua_pushvalue(L, lab_idx);
  lua_pushvalue(L, sco_idx);
  return 3;
}

static inline int tk_learn_label_f1 (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  lua_settop(L, 5);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  tk_ivec_t *exp_off = tk_ivec_peek(L, 4, "expected_offsets");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, 5, "expected_neighbors");
  const unsigned int classes = tm->classes;
  const unsigned int clause_chunks = tm->clause_chunks;
  unsigned int out_classes = tm->n_orig_dims > 0 ? tm->n_orig_dims : classes;
  bool use_batching = tm->dense_batch_size > 0 && tm->dense_batch_size < classes;
  long int *all_votes = NULL;
  if (tm->n_orig_dims > 0) {
    unsigned int n_dims = tm->n_orig_dims;
    unsigned int cpb = tm->code_prefix_bytes;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    char *base_ptr;
    char *temp_dense_buf = NULL;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      base_ptr = tm->managed_dense;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "label_f1", "n_samples");
      temp_dense_buf = tk_learn_densify_tokens(tm, tokens, n_tok);
      base_ptr = temp_dense_buf;
    }
    all_votes = (long int *)calloc((uint64_t)n * n_dims, sizeof(long int));
    #pragma omp parallel for schedule(static)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      const uint8_t *base_input = (const uint8_t *)(base_ptr + (uint64_t)s * bpc);
      for (unsigned int d = 0; d < n_dims; d++) {
        const uint8_t *prefix = tm->dim_prefixes + (uint64_t)d * cpb;
        long int total_vote = 0;
        for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
          tk_learn_calculate_split(tm, prefix, cpb, base_input, literals, votes_buf, chunk, tm->clause_tolerances[0], 0);
          long int cv = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++) cv += votes_buf[j];
          if ((chunk % clause_chunks) >= clause_chunks / 2) cv = -cv;
          total_vote += cv;
        }
        all_votes[(uint64_t)s * n_dims + d] = total_vote;
      }
    }
    free(temp_dense_buf);
  } else if (use_batching) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "label_f1", "n_samples");
    unsigned int B = tm->dense_batch_size;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    tk_token_bin_t *bin = tk_learn_get_bin(tm, tokens, n_tok);
    bool own_buf = !tm->managed_dense || tm->dense_n_samples < n_tok;
    char *buf = own_buf ? (char *)malloc((uint64_t)n_tok * B * bpc) : tm->managed_dense;
    all_votes = (long int *)calloc((uint64_t)n * classes, sizeof(long int));
    for (unsigned int batch_start = 0; batch_start < classes; batch_start += B) {
      unsigned int actual_batch = (batch_start + B <= classes) ? B : (classes - batch_start);
      uint64_t batch_class_stride = (uint64_t)n_tok * bpc;
      tk_learn_densify_tokens_range(tm, bin, buf, batch_start, actual_batch);
      #pragma omp parallel for schedule(static)
      for (unsigned int s = 0; s < n; s++) {
        unsigned int literals[TK_CVEC_BITS];
        unsigned int votes_buf[TK_CVEC_BITS];
        for (unsigned int ci = 0; ci < actual_batch; ci++) {
          unsigned int c = batch_start + ci;
          char *input = buf + ci * batch_class_stride + s * bpc;
          long int class_vote = 0;
          for (unsigned int cc = 0; cc < clause_chunks; cc++) {
            unsigned int chunk = c * clause_chunks + cc;
            tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
            long int chunk_vote = 0;
            for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
              chunk_vote += votes_buf[j];
            if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
            class_vote += chunk_vote;
          }
          all_votes[(uint64_t)s * classes + c] = class_vote;
        }
      }
    }
    if (own_buf) free(buf);
  } else {
    char *temp_dense = NULL;
    unsigned int class_n = 0;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      class_n = n;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "label_f1", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      class_n = n_tok;
    }
    const unsigned int bytes_per_class = TK_CVEC_BITS_BYTES(tm->features * 2);
    const uint64_t class_stride = (uint64_t)class_n * bytes_per_class;
    char *base_ptr = reuse_dense ? tm->managed_dense : temp_dense;
    all_votes = (long int *)calloc((uint64_t)n * classes, sizeof(long int));
    #pragma omp parallel for schedule(static)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      for (unsigned int c = 0; c < classes; c++) {
        char *input = base_ptr + c * class_stride + s * bytes_per_class;
        long int class_vote = 0;
        for (unsigned int cc = 0; cc < clause_chunks; cc++) {
          unsigned int chunk = c * clause_chunks + cc;
          tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
          long int chunk_vote = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
            chunk_vote += votes_buf[j];
          if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
          class_vote += chunk_vote;
        }
        all_votes[(uint64_t)s * classes + c] = class_vote;
      }
    }
    free(temp_dense);
  }
  uint64_t mi_tp = 0, mi_k = 0, mi_exp = 0, n_valid = 0;
  double ma_f1 = 0;
  #pragma omp parallel for reduction(+:mi_tp,mi_k,mi_exp,n_valid,ma_f1)
  for (unsigned int s = 0; s < n; s++) {
    int64_t es = exp_off->a[s], ee = exp_off->a[s + 1];
    uint64_t n_expected = (uint64_t)(ee - es);
    if (n_expected == 0) continue;
    int kha;
    tk_iuset_t *exp_set = tk_iuset_create(NULL, 0);
    for (int64_t i = es; i < ee; i++)
      tk_iuset_put(exp_set, exp_nbr->a[i], &kha);
    tk_rvec_t sorted_rv = { .n = out_classes, .m = out_classes, .lua_managed = false,
                            .a = (tk_rank_t *)malloc(out_classes * sizeof(tk_rank_t)) };
    for (unsigned int c = 0; c < out_classes; c++)
      sorted_rv.a[c] = tk_rank((int64_t)c, (double)all_votes[(uint64_t)s * out_classes + c]);
    tk_rvec_desc(&sorted_rv, 0, sorted_rv.n);
    tk_rank_t *sorted = sorted_rv.a;
    double best_f1 = 0;
    uint64_t best_k = 1, best_tp = 0, tp = 0;
    for (unsigned int ki = 1; ki <= out_classes; ki++) {
      if (tk_iuset_contains(exp_set, sorted[ki - 1].i)) tp++;
      double prec = (double)tp / ki;
      double rec = (double)tp / n_expected;
      double f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
      if (f1 > best_f1) {
        best_f1 = f1;
        best_k = ki;
        best_tp = tp;
      }
    }
    tk_iuset_destroy(exp_set);
    free(sorted_rv.a);
    mi_tp += best_tp;
    mi_k += best_k;
    mi_exp += n_expected;
    ma_f1 += best_f1;
    n_valid++;
  }
  free(all_votes);
  double mi_prec = mi_k > 0 ? (double)mi_tp / mi_k : 0;
  double mi_rec = mi_exp > 0 ? (double)mi_tp / mi_exp : 0;
  double micro_f1 = (mi_prec + mi_rec) > 0 ? 2.0 * mi_prec * mi_rec / (mi_prec + mi_rec) : 0;
  double macro_f1 = n_valid > 0 ? ma_f1 / n_valid : 0;
  lua_pushnumber(L, micro_f1);
  lua_pushnumber(L, macro_f1);
  return 2;
}

static inline int tk_learn_regress (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  return tk_learn_predict_regressor(L, tm);
}

static inline int tk_learn_regress_mae (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  lua_settop(L, 4);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  tk_dvec_t *targets = tk_dvec_peek(L, 4, "targets");

  const unsigned int classes = tm->classes;
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int *per_class_targets = tm->targets;
  const double *y_min = tm->y_min;
  const double *y_max = tm->y_max;
  const double *tgt = targets->a;
  unsigned int out_cols = tm->n_orig_dims > 0 ? tm->n_orig_dims : classes;
  size_t n_elems = (size_t)n * out_cols;
  if (targets->n < n_elems)
    return luaL_error(L, "targets length %d < n_samples * outputs %d", (int)targets->n, (int)n_elems);

  double total_err = 0.0;

  bool use_batching = tm->dense_batch_size > 0 && tm->dense_batch_size < classes;

  if (tm->n_orig_dims > 0) {
    unsigned int n_dims = tm->n_orig_dims;
    unsigned int cpb = tm->code_prefix_bytes;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    char *base_ptr;
    char *temp_dense = NULL;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      base_ptr = tm->managed_dense;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "regress_mae", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      base_ptr = temp_dense;
    }
    double y_range = y_max[0] - y_min[0];
    double scale = 1.0 / ((double)clause_chunks * per_class_targets[0]);
    #pragma omp parallel for schedule(static) reduction(+:total_err)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      const uint8_t *base_input = (const uint8_t *)(base_ptr + (uint64_t)s * bpc);
      for (unsigned int d = 0; d < n_dims; d++) {
        const uint8_t *prefix = tm->dim_prefixes + (uint64_t)d * cpb;
        long int total_vote = 0;
        for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
          tk_learn_calculate_split(tm, prefix, cpb, base_input, literals, votes_buf, chunk, tm->clause_tolerances[0], 0);
          long int cv = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++) cv += votes_buf[j];
          if (cv > (long int)tm->targets[0]) cv = (long int)tm->targets[0];
          if ((chunk % clause_chunks) >= clause_chunks / 2) cv = -cv;
          total_vote += cv;
        }
        double pred = ((double)total_vote * scale + 0.5) * y_range + y_min[0];
        total_err += fabs(pred - tgt[s * n_dims + d]);
      }
    }
    free(temp_dense);
  } else if (use_batching) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "regress_mae", "n_samples");
    unsigned int B = tm->dense_batch_size;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    tk_token_bin_t *bin = tk_learn_get_bin(tm, tokens, n_tok);
    bool own_buf = !tm->managed_dense || tm->dense_n_samples < n_tok;
    char *buf = own_buf ? (char *)malloc((uint64_t)n_tok * B * bpc) : tm->managed_dense;
    for (unsigned int batch_start = 0; batch_start < classes; batch_start += B) {
      unsigned int actual_batch = (batch_start + B <= classes) ? B : (classes - batch_start);
      uint64_t batch_class_stride = (uint64_t)n_tok * bpc;
      tk_learn_densify_tokens_range(tm, bin, buf, batch_start, actual_batch);
      double batch_err = 0.0;
      #pragma omp parallel for schedule(static) reduction(+:batch_err)
      for (unsigned int s = 0; s < n; s++) {
        unsigned int literals[TK_CVEC_BITS];
        unsigned int votes_buf[TK_CVEC_BITS];
        for (unsigned int ci = 0; ci < actual_batch; ci++) {
          unsigned int c = batch_start + ci;
          char *input = buf + ci * batch_class_stride + s * bpc;
          double y_range = y_max[c] - y_min[c];
          double scale = 1.0 / ((double)clause_chunks * per_class_targets[c]);
          long int class_vote = 0;
          for (unsigned int cc = 0; cc < clause_chunks; cc++) {
            unsigned int chunk = c * clause_chunks + cc;
            tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
            long int chunk_vote = 0;
            for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
                chunk_vote += votes_buf[j];
            if (chunk_vote > (long int)tm->targets[c]) chunk_vote = (long int)tm->targets[c];
            if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
            class_vote += chunk_vote;
          }
          double pred = ((double)class_vote * scale + 0.5) * y_range + y_min[c];
          batch_err += fabs(pred - tgt[s * classes + c]);
        }
      }
      total_err += batch_err;
    }
    if (own_buf) free(buf);
  } else {
    char *temp_dense = NULL;
    unsigned int class_n = 0;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      class_n = n;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "regress_mae", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      class_n = n_tok;
    }
    const unsigned int bytes_per_class = TK_CVEC_BITS_BYTES(tm->features * 2);
    const uint64_t class_stride = (uint64_t)class_n * bytes_per_class;
    char *base_ptr = reuse_dense ? tm->managed_dense : temp_dense;
    #pragma omp parallel for schedule(static) reduction(+:total_err)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      for (unsigned int c = 0; c < classes; c++) {
        char *input = base_ptr + c * class_stride + s * bytes_per_class;
        double y_range = y_max[c] - y_min[c];
        double scale = 1.0 / ((double)clause_chunks * per_class_targets[c]);
        long int class_vote = 0;
        for (unsigned int cc = 0; cc < clause_chunks; cc++) {
          unsigned int chunk = c * clause_chunks + cc;
          tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
          long int chunk_vote = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
              chunk_vote += votes_buf[j];
          if (chunk_vote > (long int)tm->targets[c]) chunk_vote = (long int)tm->targets[c];
          if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
          class_vote += chunk_vote;
        }
        double pred = ((double)class_vote * scale + 0.5) * y_range + y_min[c];
        total_err += fabs(pred - tgt[s * classes + c]);
      }
    }
    free(temp_dense);
  }

  double mae = n_elems > 0 ? total_err / (double)n_elems : 0.0;
  lua_pushnumber(L, -mae);
  lua_pushnumber(L, mae);
  return 2;
}

static inline int tk_learn_regress_nmae (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  lua_settop(L, 4);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  tk_dvec_t *targets = tk_dvec_peek(L, 4, "targets");

  const unsigned int classes = tm->classes;
  const unsigned int clause_chunks = tm->clause_chunks;
  const unsigned int *per_class_targets = tm->targets;
  const double *y_min = tm->y_min;
  const double *y_max = tm->y_max;
  const double *tgt = targets->a;
  unsigned int out_cols = tm->n_orig_dims > 0 ? tm->n_orig_dims : classes;
  size_t n_elems = (size_t)n * out_cols;
  if (targets->n < n_elems)
    return luaL_error(L, "targets length %d < n_samples * outputs %d", (int)targets->n, (int)n_elems);

  double total_err = 0.0;
  double sum_exp = 0.0;

  bool use_batching = tm->dense_batch_size > 0 && tm->dense_batch_size < classes;

  if (tm->n_orig_dims > 0) {
    unsigned int n_dims = tm->n_orig_dims;
    unsigned int cpb = tm->code_prefix_bytes;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    char *base_ptr;
    char *temp_dense = NULL;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      base_ptr = tm->managed_dense;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "regress_nmae", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      base_ptr = temp_dense;
    }
    double y_range = y_max[0] - y_min[0];
    double scale = 1.0 / ((double)clause_chunks * per_class_targets[0]);
    #pragma omp parallel for schedule(static) reduction(+:total_err,sum_exp)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      const uint8_t *base_input = (const uint8_t *)(base_ptr + (uint64_t)s * bpc);
      for (unsigned int d = 0; d < n_dims; d++) {
        const uint8_t *prefix = tm->dim_prefixes + (uint64_t)d * cpb;
        long int total_vote = 0;
        for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
          tk_learn_calculate_split(tm, prefix, cpb, base_input, literals, votes_buf, chunk, tm->clause_tolerances[0], 0);
          long int cv = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++) cv += votes_buf[j];
          if (cv > (long int)tm->targets[0]) cv = (long int)tm->targets[0];
          if ((chunk % clause_chunks) >= clause_chunks / 2) cv = -cv;
          total_vote += cv;
        }
        double pred = ((double)total_vote * scale + 0.5) * y_range + y_min[0];
        double exp_val = tgt[s * n_dims + d];
        total_err += fabs(pred - exp_val);
        sum_exp += exp_val;
      }
    }
    free(temp_dense);
  } else if (use_batching) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "regress_nmae", "n_samples");
    unsigned int B = tm->dense_batch_size;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    tk_token_bin_t *bin = tk_learn_get_bin(tm, tokens, n_tok);
    bool own_buf = !tm->managed_dense || tm->dense_n_samples < n_tok;
    char *buf = own_buf ? (char *)malloc((uint64_t)n_tok * B * bpc) : tm->managed_dense;
    for (unsigned int batch_start = 0; batch_start < classes; batch_start += B) {
      unsigned int actual_batch = (batch_start + B <= classes) ? B : (classes - batch_start);
      uint64_t batch_class_stride = (uint64_t)n_tok * bpc;
      tk_learn_densify_tokens_range(tm, bin, buf, batch_start, actual_batch);
      double batch_err = 0.0;
      double batch_exp = 0.0;
      #pragma omp parallel for schedule(static) reduction(+:batch_err,batch_exp)
      for (unsigned int s = 0; s < n; s++) {
        unsigned int literals[TK_CVEC_BITS];
        unsigned int votes_buf[TK_CVEC_BITS];
        for (unsigned int ci = 0; ci < actual_batch; ci++) {
          unsigned int c = batch_start + ci;
          char *input = buf + ci * batch_class_stride + s * bpc;
          double y_range = y_max[c] - y_min[c];
          double scale = 1.0 / ((double)clause_chunks * per_class_targets[c]);
          long int class_vote = 0;
          for (unsigned int cc = 0; cc < clause_chunks; cc++) {
            unsigned int chunk = c * clause_chunks + cc;
            tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
            long int chunk_vote = 0;
            for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
                chunk_vote += votes_buf[j];
            if (chunk_vote > (long int)tm->targets[c]) chunk_vote = (long int)tm->targets[c];
            if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
            class_vote += chunk_vote;
          }
          double pred = ((double)class_vote * scale + 0.5) * y_range + y_min[c];
          double exp_val = tgt[s * classes + c];
          batch_err += fabs(pred - exp_val);
          batch_exp += exp_val;
        }
      }
      total_err += batch_err;
      sum_exp += batch_exp;
    }
    if (own_buf) free(buf);
  } else {
    char *temp_dense = NULL;
    unsigned int class_n = 0;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      class_n = n;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "regress_nmae", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      class_n = n_tok;
    }
    const unsigned int bytes_per_class = TK_CVEC_BITS_BYTES(tm->features * 2);
    const uint64_t class_stride = (uint64_t)class_n * bytes_per_class;
    char *base_ptr = reuse_dense ? tm->managed_dense : temp_dense;
    #pragma omp parallel for schedule(static) reduction(+:total_err,sum_exp)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      for (unsigned int c = 0; c < classes; c++) {
        char *input = base_ptr + c * class_stride + s * bytes_per_class;
        double y_range = y_max[c] - y_min[c];
        double scale = 1.0 / ((double)clause_chunks * per_class_targets[c]);
        long int class_vote = 0;
        for (unsigned int cc = 0; cc < clause_chunks; cc++) {
          unsigned int chunk = c * clause_chunks + cc;
          tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
          long int chunk_vote = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
              chunk_vote += votes_buf[j];
          if (chunk_vote > (long int)tm->targets[c]) chunk_vote = (long int)tm->targets[c];
          if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
          class_vote += chunk_vote;
        }
        double pred = ((double)class_vote * scale + 0.5) * y_range + y_min[c];
        double exp_val = tgt[s * classes + c];
        total_err += fabs(pred - exp_val);
        sum_exp += exp_val;
      }
    }
    free(temp_dense);
  }

  double mean_err = n_elems > 0 ? total_err / (double)n_elems : 0.0;
  double mean_exp = n_elems > 0 ? sum_exp / (double)n_elems : 0.0;
  double nmae = mean_exp > 0.0 ? mean_err / mean_exp : 0.0;
  lua_pushnumber(L, -nmae);
  lua_pushnumber(L, nmae);
  return 2;
}

static inline int tk_learn_encode (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  lua_settop(L, 5);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  tk_cvec_t *out_buf = lua_isnil(L, 5) ? NULL : tk_cvec_peek(L, 5, "output");

  const unsigned int classes = tm->classes;
  const unsigned int clause_chunks = tm->clause_chunks;
  unsigned int out_classes = tm->n_orig_dims > 0 ? tm->n_orig_dims : classes;
  const unsigned int out_bytes = TK_CVEC_BITS_BYTES(out_classes);
  size_t needed = n * out_bytes;
  if (needed > tm->encodings_len) {
    tm->encodings = tk_realloc(L, tm->encodings, needed);
    tm->encodings_len = needed;
  }
  char *encodings = tm->encodings;
  memset(encodings, 0, needed);

  bool use_batching = tm->dense_batch_size > 0 && tm->dense_batch_size < classes;

  if (tm->n_orig_dims > 0) {
    unsigned int n_dims = tm->n_orig_dims;
    unsigned int cpb = tm->code_prefix_bytes;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    char *base_ptr;
    char *temp_dense_buf = NULL;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      base_ptr = tm->managed_dense;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "encode", "n_samples");
      temp_dense_buf = tk_learn_densify_tokens(tm, tokens, n_tok);
      base_ptr = temp_dense_buf;
    }
    #pragma omp parallel for schedule(static)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      const uint8_t *base_input = (const uint8_t *)(base_ptr + (uint64_t)s * bpc);
      char *out_row = encodings + s * out_bytes;
      for (unsigned int d = 0; d < n_dims; d++) {
        const uint8_t *prefix = tm->dim_prefixes + (uint64_t)d * cpb;
        long int total_vote = 0;
        for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
          tk_learn_calculate_split(tm, prefix, cpb, base_input, literals, votes_buf, chunk, tm->clause_tolerances[0], 0);
          long int cv = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++) cv += votes_buf[j];
          if ((chunk % clause_chunks) >= clause_chunks / 2) cv = -cv;
          total_vote += cv;
        }
        if (total_vote > 0)
          out_row[d / 8] |= (1 << (d % 8));
      }
    }
    free(temp_dense_buf);
  } else if (use_batching) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "encode", "n_samples");
    unsigned int B = tm->dense_batch_size;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    tk_token_bin_t *bin = tk_learn_get_bin(tm, tokens, n_tok);
    bool own_buf = !tm->managed_dense || tm->dense_n_samples < n_tok;
    char *buf = own_buf ? (char *)malloc((uint64_t)n_tok * B * bpc) : tm->managed_dense;
    for (unsigned int batch_start = 0; batch_start < classes; batch_start += B) {
      unsigned int actual_batch = (batch_start + B <= classes) ? B : (classes - batch_start);
      uint64_t batch_class_stride = (uint64_t)n_tok * bpc;
      tk_learn_densify_tokens_range(tm, bin, buf, batch_start, actual_batch);
      #pragma omp parallel for schedule(static)
      for (unsigned int s = 0; s < n; s++) {
        unsigned int literals[TK_CVEC_BITS];
        unsigned int votes_buf[TK_CVEC_BITS];
        char *out_row = encodings + s * out_bytes;
        for (unsigned int ci = 0; ci < actual_batch; ci++) {
          unsigned int c = batch_start + ci;
          char *input = buf + ci * batch_class_stride + s * bpc;
          long int class_vote = 0;
          for (unsigned int cc = 0; cc < clause_chunks; cc++) {
            unsigned int chunk = c * clause_chunks + cc;
            tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
            long int chunk_vote = 0;
            for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
                chunk_vote += votes_buf[j];
            if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
            class_vote += chunk_vote;
          }
          if (class_vote > 0)
            out_row[c / 8] |= (1 << (c % 8));
        }
      }
    }
    if (own_buf) free(buf);
  } else {
    char *temp_dense = NULL;
    unsigned int class_n = 0;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      class_n = n;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "encode", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      class_n = n_tok;
    }
    const unsigned int bytes_per_class = TK_CVEC_BITS_BYTES(tm->features * 2);
    const uint64_t class_stride = (uint64_t)class_n * bytes_per_class;
    char *base_ptr = reuse_dense ? tm->managed_dense : temp_dense;
    #pragma omp parallel for schedule(static)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      char *out_row = encodings + s * out_bytes;
      for (unsigned int c = 0; c < classes; c++) {
        char *input = base_ptr + c * class_stride + s * bytes_per_class;
        long int class_vote = 0;
        for (unsigned int cc = 0; cc < clause_chunks; cc++) {
          unsigned int chunk = c * clause_chunks + cc;
          tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
          long int chunk_vote = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
              chunk_vote += votes_buf[j];
          if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
          class_vote += chunk_vote;
        }
        if (class_vote > 0)
          out_row[c / 8] |= (1 << (c % 8));
      }
    }
    free(temp_dense);
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

static inline int tk_learn_encode_hamming (lua_State *L)
{
  tk_learn_t *tm = tk_learn_peek(L, 1);
  lua_settop(L, 4);
  unsigned int n = tk_lua_checkunsigned(L, 3, "n_samples");
  tk_cvec_t *expected = tk_cvec_peek(L, 4, "expected");

  const unsigned int classes = tm->classes;
  const unsigned int clause_chunks = tm->clause_chunks;
  unsigned int out_classes = tm->n_orig_dims > 0 ? tm->n_orig_dims : classes;
  const unsigned int out_bytes = TK_CVEC_BITS_BYTES(out_classes);
  const uint8_t *exp_data = (const uint8_t *)expected->a;

  uint64_t total_match = 0;
  uint64_t total_bits = (uint64_t)n * out_classes;

  bool use_batching = tm->dense_batch_size > 0 && tm->dense_batch_size < classes;

  if (tm->n_orig_dims > 0) {
    unsigned int n_dims = tm->n_orig_dims;
    unsigned int cpb = tm->code_prefix_bytes;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    char *base_ptr;
    char *temp_dense_buf = NULL;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      base_ptr = tm->managed_dense;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "encode_hamming", "n_samples");
      temp_dense_buf = tk_learn_densify_tokens(tm, tokens, n_tok);
      base_ptr = temp_dense_buf;
    }
    #pragma omp parallel for schedule(static) reduction(+:total_match)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      const uint8_t *base_input = (const uint8_t *)(base_ptr + (uint64_t)s * bpc);
      for (unsigned int d = 0; d < n_dims; d++) {
        const uint8_t *prefix = tm->dim_prefixes + (uint64_t)d * cpb;
        long int total_vote = 0;
        for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
          tk_learn_calculate_split(tm, prefix, cpb, base_input, literals, votes_buf, chunk, tm->clause_tolerances[0], 0);
          long int cv = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++) cv += votes_buf[j];
          if ((chunk % clause_chunks) >= clause_chunks / 2) cv = -cv;
          total_vote += cv;
        }
        unsigned int pred_bit = total_vote > 0 ? 1 : 0;
        unsigned int exp_bit = (exp_data[s * out_bytes + d / 8] >> (d % 8)) & 1;
        if (pred_bit == exp_bit) total_match++;
      }
    }
    free(temp_dense_buf);
  } else if (use_batching) {
    lua_getfield(L, 2, "tokens");
    tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
    lua_pop(L, 1);
    unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "encode_hamming", "n_samples");
    unsigned int B = tm->dense_batch_size;
    unsigned int bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
    tk_token_bin_t *bin = tk_learn_get_bin(tm, tokens, n_tok);
    bool own_buf = !tm->managed_dense || tm->dense_n_samples < n_tok;
    char *buf = own_buf ? (char *)malloc((uint64_t)n_tok * B * bpc) : tm->managed_dense;
    for (unsigned int batch_start = 0; batch_start < classes; batch_start += B) {
      unsigned int actual_batch = (batch_start + B <= classes) ? B : (classes - batch_start);
      uint64_t batch_class_stride = (uint64_t)n_tok * bpc;
      tk_learn_densify_tokens_range(tm, bin, buf, batch_start, actual_batch);
      uint64_t batch_match = 0;
      #pragma omp parallel for schedule(static) reduction(+:batch_match)
      for (unsigned int s = 0; s < n; s++) {
        unsigned int literals[TK_CVEC_BITS];
        unsigned int votes_buf[TK_CVEC_BITS];
        for (unsigned int ci = 0; ci < actual_batch; ci++) {
          unsigned int c = batch_start + ci;
          char *input = buf + ci * batch_class_stride + s * bpc;
          long int class_vote = 0;
          for (unsigned int cc = 0; cc < clause_chunks; cc++) {
            unsigned int chunk = c * clause_chunks + cc;
            tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
            long int chunk_vote = 0;
            for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
                chunk_vote += votes_buf[j];
            if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
            class_vote += chunk_vote;
          }
          unsigned int pred_bit = class_vote > 0 ? 1 : 0;
          unsigned int exp_bit = (exp_data[s * out_bytes + c / 8] >> (c % 8)) & 1;
          if (pred_bit == exp_bit) batch_match++;
        }
      }
      total_match += batch_match;
    }
    if (own_buf) free(buf);
  } else {
    char *temp_dense = NULL;
    unsigned int class_n = 0;
    bool reuse_dense = tm->managed_dense && tm->dense_n_samples == n
      && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes);
    if (reuse_dense) {
      class_n = n;
    } else {
      lua_getfield(L, 2, "tokens");
      tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
      lua_pop(L, 1);
      unsigned int n_tok = tk_lua_fcheckunsigned(L, 2, "encode_hamming", "n_samples");
      temp_dense = tk_learn_densify_tokens(tm, tokens, n_tok);
      class_n = n_tok;
    }
    const unsigned int bytes_per_class = TK_CVEC_BITS_BYTES(tm->features * 2);
    const uint64_t class_stride = (uint64_t)class_n * bytes_per_class;
    char *base_ptr = reuse_dense ? tm->managed_dense : temp_dense;
    #pragma omp parallel for schedule(static) reduction(+:total_match)
    for (unsigned int s = 0; s < n; s++) {
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      for (unsigned int c = 0; c < classes; c++) {
        char *input = base_ptr + c * class_stride + s * bytes_per_class;
        long int class_vote = 0;
        for (unsigned int cc = 0; cc < clause_chunks; cc++) {
          unsigned int chunk = c * clause_chunks + cc;
          tk_learn_calculate(tm, input, literals, votes_buf, chunk, tm->clause_tolerances[c], 0);
          long int chunk_vote = 0;
          for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
              chunk_vote += votes_buf[j];
          if (cc >= clause_chunks / 2) chunk_vote = -chunk_vote;
          class_vote += chunk_vote;
        }
        unsigned int pred_bit = class_vote > 0 ? 1 : 0;
        unsigned int exp_bit = (exp_data[s * out_bytes + c / 8] >> (c % 8)) & 1;
        if (pred_bit == exp_bit) total_match++;
      }
    }
    free(temp_dense);
  }

  double accuracy = total_bits > 0 ? (double)total_match / (double)total_bits : 0.0;
  lua_pushnumber(L, accuracy);
  lua_pushnumber(L, accuracy);
  return 2;
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
      ? ps->a + local_class * class_stride + sample * bytes_per_class \
      : ps->a + sample * input_chunks; \
    double y_target = GET_TARGET; \
    tk_learn_calculate(tm, input, literals, votes, chunk, chunk_tolerance, 1); \
    long int chunk_vote = 0; \
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) \
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
    uint32_t prob_thresh = (uint32_t)(probability * 4294967295.0); \
    for (unsigned int j = 0; j < TK_CVEC_BITS; j++) { \
      if (tk_fast_random() < prob_thresh) { \
        apply_feedback(tm, j, chunk, input, literals, votes, want_more, chunk_maximum, chunk_spec_thresh, &spec_bufs[chunk_class]); \
      } \
    } \
  } \
  }

static inline int tk_learn_train_regressor (lua_State *L, tk_learn_t *tm)
{
  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");

  tk_cvec_t *ps = NULL;

  {
    if (!tm->managed_dense || tm->dense_stale) {
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
      if (tm->flat_evict && tm->n_code_bits > 0) {
        unsigned int ncb = tm->n_code_bits;
        unsigned int old_n = tm->absorb_ranking_global_n;
        unsigned int new_n = ncb + old_n;
        int64_t *old_ptr = tm->absorb_ranking_global;
        bool was_alias = (old_ptr == tm->absorb_ranking);
        int64_t *new_ptr = (int64_t *)malloc(new_n * sizeof(int64_t));
        for (unsigned int k = 0; k < ncb; k++)
          new_ptr[k] = (int64_t)(tm->n_tokens + k);
        if (old_n > 0)
          memcpy(new_ptr + ncb, old_ptr, old_n * sizeof(int64_t));
        if (!was_alias)
          free(old_ptr);
        tm->absorb_ranking_global = new_ptr;
        tm->absorb_ranking_global_n = new_n;
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
        if (tm->absorb_ranking_limit > 0 && n_init > tm->absorb_ranking_limit)
          n_init = tm->absorb_ranking_limit;
        tk_learn_init_mapping(tm, c, init_ids, n_init);
      }
      tm->dense_bpc = TK_CVEC_BITS_BYTES(tm->features * 2);
      unsigned int alloc_classes;
      if (tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes)
        alloc_classes = tm->dense_batch_size;
      else
        alloc_classes = tm->classes;
      tm->dense_bps = alloc_classes * tm->dense_bpc;
      tm->dense_n_samples = n;
      uint64_t dense_needed = (uint64_t)n * tm->dense_bps;
      if (!tm->managed_dense || dense_needed > tm->dense_alloc) {
        free(tm->managed_dense);
        tm->managed_dense = (char *)malloc(dense_needed);
        tm->dense_alloc = dense_needed;
      }
      if (!(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes))
        tk_learn_densify_all(tm);
      tm->dense_stale = false;
    }

    ps = (tk_cvec_t *)malloc(sizeof(tk_cvec_t));
    ps->a = tm->managed_dense;
    ps->n = (uint64_t)n * tm->dense_bps;
  }

  tm_target_mode_t target_mode;
  tk_cvec_t *codes = NULL;
  tk_dvec_t *targets_dvec = NULL;
  double *targets = NULL;
  int64_t *sol_offsets = NULL;
  int64_t *sol_neighbors = NULL;
  size_t sol_neighbors_n = 0;
  uint8_t *code_bytes = NULL;
  unsigned int code_chunks = 0;

  if (tk_lua_ftype(L, 2, "sol_offsets") != LUA_TNIL) {
    target_mode = TM_TARGET_IVEC;
    lua_getfield(L, 2, "sol_offsets");
    tk_ivec_t *soff = tk_ivec_peek(L, -1, "sol_offsets");
    lua_pop(L, 1);
    lua_getfield(L, 2, "sol_neighbors");
    tk_ivec_t *snbr = tk_ivec_peek(L, -1, "sol_neighbors");
    lua_pop(L, 1);
    sol_offsets = soff->a;
    sol_neighbors = snbr->a;
    sol_neighbors_n = snbr->n;
  } else if (tk_lua_ftype(L, 2, "codes") != LUA_TNIL) {
    target_mode = TM_TARGET_CVEC;
    lua_getfield(L, 2, "codes");
    codes = tk_cvec_peek(L, -1, "codes");
    lua_pop(L, 1);
    code_bytes = (uint8_t *)codes->a;
    code_chunks = TK_CVEC_BITS_BYTES(tm->n_orig_dims > 0 ? tm->n_orig_dims : tm->classes);
  } else if (tk_lua_ftype(L, 2, "targets") != LUA_TNIL) {
    target_mode = TM_TARGET_DVEC;
    lua_getfield(L, 2, "targets");
    targets_dvec = tk_dvec_peek(L, -1, "targets");
    lua_pop(L, 1);
    targets = targets_dvec->a;
  } else {
    free(ps);
    return luaL_error(L, "regressor train requires sol_offsets+sol_neighbors (CSR), codes (cvec), or targets (dvec)");
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
  if (target_mode == TM_TARGET_DVEC && tm->n_orig_dims > 0) {
    unsigned int n_dims = tm->n_orig_dims;
    y_min[0] = DBL_MAX;
    y_max[0] = -DBL_MAX;
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int d = 0; d < n_dims; d++) {
        double v = targets[i * n_dims + d];
        if (v > y_max[0]) y_max[0] = v;
        if (v < y_min[0]) y_min[0] = v;
      }
    }
    if (y_max[0] <= y_min[0]) { y_min[0] = 0.0; y_max[0] = 1.0; }
  } else if (target_mode == TM_TARGET_DVEC) {
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
  bool grouped = true;
  unsigned int bytes_per_class = TK_CVEC_BITS_BYTES(tm->features * 2);
  bool use_batching = tm->dense_batch_size > 0 && tm->dense_batch_size < classes;
  uint64_t class_stride = (uint64_t)n * bytes_per_class;

  unsigned int skip_n = tm->n_orig_dims > 0 ? tm->n_orig_dims : classes;
  uint32_t *skip_thresholds = NULL;
  if (target_mode == TM_TARGET_IVEC) {
    uint64_t *class_counts = (uint64_t *)calloc(skip_n, sizeof(uint64_t));
    for (size_t j = 0; j < sol_neighbors_n; j++) {
      unsigned int c = (unsigned int)sol_neighbors[j];
      if (c < skip_n) class_counts[c]++;
    }
    skip_thresholds = (uint32_t *)tk_malloc(L, skip_n * sizeof(uint32_t));
    for (unsigned int c = 0; c < skip_n; c++) {
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
    uint64_t *bit_counts = (uint64_t *)calloc(skip_n, sizeof(uint64_t));
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int c = 0; c < skip_n; c++) {
        if (code_bytes[i * code_chunks + c / 8] & (1 << (c % 8)))
          bit_counts[c]++;
      }
    }
    skip_thresholds = (uint32_t *)tk_malloc(L, skip_n * sizeof(uint32_t));
    for (unsigned int c = 0; c < skip_n; c++) {
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

  unsigned int spec_per_class = 65536;
  {
    uint64_t total = (uint64_t)classes * spec_per_class;
    if (total > 8u * 1024 * 1024) {
      spec_per_class = (8u * 1024 * 1024) / classes;
      if (spec_per_class < tm->input_chunks)
        spec_per_class = tm->input_chunks;
    }
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

  if (tm->n_orig_dims > 0) {
    unsigned int n_dims = tm->n_orig_dims;
    unsigned int cpb = tm->code_prefix_bytes;
    unsigned int bpc = bytes_per_class;
    char *base_ptr = ps->a;
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int *shuffle = shuffles[tid];
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes_buf[TK_CVEC_BITS];
      char *temp_row = (char *)malloc(bpc);
      uint8_t *spec_data = (uint8_t *)malloc(spec_per_class);
      tk_spec_buf_t spec_bufs_flat;
      spec_bufs_flat.buf = spec_data;
      spec_bufs_flat.size = spec_per_class;
      spec_bufs_flat.cursor = spec_per_class;
      spec_bufs_flat.sp = tm->specificity_pts[0];
      tk_spec_buf_t *spec_bufs = &spec_bufs_flat;
      for (unsigned int iter = 0; iter < max_iter; iter++) {
        if (break_flag) break;
        tk_learn_init_shuffle(shuffle, shuffle_n);
        #pragma omp for schedule(dynamic)
        for (unsigned int chunk = 0; chunk < clause_chunks; chunk++) {
          unsigned int chunk_tolerance = tm->clause_tolerances[0];
          unsigned int chunk_maximum = tm->clause_maximums[0];
          unsigned int chunk_spec_thresh = tm->specificity_thresholds[0];
          long int vote_target = (long int)tm->targets[0];
          for (unsigned int i = 0; i < n; i++) {
            unsigned int sample = shuffle[i];
            const uint8_t *base_input = (const uint8_t *)(base_ptr + (uint64_t)sample * bpc);
            bool temp_row_dirty = true;
            for (unsigned int d = 0; d < n_dims; d++) {
              if (tm->flat_skip_thresh > 0 && tk_fast_random() < tm->flat_skip_thresh) continue;
              double y_target = 0.0;
              switch (target_mode) {
                case TM_TARGET_IVEC: {
                  bool is_pos = false;
                  for (int64_t _k = sol_offsets[sample]; _k < sol_offsets[sample + 1]; _k++)
                    if ((unsigned int)sol_neighbors[_k] == d) { is_pos = true; break; }
                  if (!is_pos && tk_fast_random() < skip_thresholds[d]) continue;
                  y_target = is_pos ? 1.0 : -1.0;
                  break;
                }
                case TM_TARGET_CVEC: {
                  bool bit_set = code_bytes[sample * code_chunks + d / 8] & (1 << (d % 8));
                  if (!bit_set && tk_fast_random() < skip_thresholds[d]) continue;
                  y_target = bit_set ? 1.0 : -1.0;
                  break;
                }
                case TM_TARGET_DVEC:
                  y_target = targets[sample * n_dims + d];
                  break;
              }
              const uint8_t *prefix = tm->dim_prefixes + (uint64_t)d * cpb;
              tk_learn_calculate_split(tm, prefix, cpb, base_input, literals, votes_buf, chunk, chunk_tolerance, 1);
              long int chunk_vote = 0;
              for (unsigned int j = 0; j < TK_CVEC_BITS; j++)
                chunk_vote += votes_buf[j];
              bool chunk_neg = (chunk % clause_chunks) >= clause_chunks / 2;
              double class_y_min = y_min[0];
              double class_y_range = y_max[0] - class_y_min;
              double target_ratio = (y_target - class_y_min) / class_y_range;
              if (chunk_neg) target_ratio = 1.0 - target_ratio;
              double ideal_chunk_vote = target_ratio * (double)vote_target;
              if (ideal_chunk_vote < 0.0) ideal_chunk_vote = 0.0;
              if (ideal_chunk_vote > (double)vote_target) ideal_chunk_vote = (double)vote_target;
              bool want_more = ((double)chunk_vote < ideal_chunk_vote);
              double error_ratio = ((double)chunk_vote - ideal_chunk_vote) / (double)vote_target;
              double probability = fabs(error_ratio);
              uint32_t prob_thresh = (uint32_t)(probability * 4294967295.0);
              if (prob_thresh > 0) {
                if (temp_row_dirty) {
                  memcpy(temp_row, base_input, bpc);
                  temp_row_dirty = false;
                }
                memcpy(temp_row, prefix, cpb);
                for (unsigned int j = 0; j < TK_CVEC_BITS; j++) {
                  if (tk_fast_random() < prob_thresh)
                    apply_feedback(tm, j, chunk, temp_row, literals, votes_buf, want_more, chunk_maximum, chunk_spec_thresh, &spec_bufs[0]);
                }
              }
            }
          }
        }
        #pragma omp for schedule(dynamic)
        for (unsigned int c = 0; c < 1; c++)
          tk_learn_absorb_class(tm, c);
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
      free(spec_data);
      free(temp_row);
    }
  } else if (use_batching) {
    unsigned int B = tm->dense_batch_size;
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int *shuffle = shuffles[tid];
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes[TK_CVEC_BITS];
      uint8_t *spec_data = (uint8_t *)malloc((size_t)classes * spec_per_class);
      tk_spec_buf_t *spec_bufs = (tk_spec_buf_t *)malloc(classes * sizeof(tk_spec_buf_t));
      for (unsigned int c = 0; c < classes; c++) {
        spec_bufs[c].buf = spec_data + (size_t)c * spec_per_class;
        spec_bufs[c].size = spec_per_class;
        spec_bufs[c].cursor = spec_per_class;
        spec_bufs[c].sp = tm->specificity_pts[c];
      }
      for (unsigned int iter = 0; iter < max_iter; iter++) {
        if (break_flag) break;
        tk_learn_init_shuffle(shuffle, shuffle_n);
        for (unsigned int batch_start = 0; batch_start < classes; batch_start += B) {
          unsigned int actual_batch = (batch_start + B <= classes) ? B : (classes - batch_start);
          #pragma omp single
          {
            tm->dense_batch_start = batch_start;
          }
          #pragma omp for schedule(static)
          for (unsigned int ci = 0; ci < actual_batch; ci++)
            memset((uint8_t *)tm->managed_dense + (uint64_t)ci * n * tm->dense_bpc, 0xAA, (uint64_t)n * tm->dense_bpc);
          #pragma omp for schedule(dynamic)
          for (unsigned int dc = batch_start; dc < batch_start + actual_batch; dc++) {
            for (unsigned int dk = 0; dk < tm->features; dk++) {
              int64_t dmtok = tm->mapping[(uint64_t)dc * tm->features + dk];
              if (dmtok < 0 || (unsigned int)dmtok >= tm->n_tokens) continue;
              tk_learn_densify_slot(tm, dc, dk, (unsigned int)dmtok, -1);
            }
          }
          unsigned int batch_chunks = clause_chunks * actual_batch;
          unsigned int chunk_offset = batch_start * clause_chunks;
          #pragma omp for schedule(dynamic)
          for (unsigned int local_chunk = 0; local_chunk < batch_chunks; local_chunk++) {
            unsigned int chunk = chunk_offset + local_chunk;
            unsigned int chunk_class = batch_start + local_chunk / clause_chunks;
            unsigned int local_class = local_chunk / clause_chunks;
            switch (target_mode) {
              case TM_TARGET_IVEC:
                TM_REGRESSION_INNER_LOOP(
                  _sample_is_pos ? 1.0 : -1.0,
                  bool _sample_is_pos = false;
                  for (int64_t _k = sol_offsets[sample]; _k < sol_offsets[sample + 1]; _k++)
                    if ((unsigned int)sol_neighbors[_k] == chunk_class) { _sample_is_pos = true; break; }
                  if (!_sample_is_pos && tk_fast_random() < skip_thresholds[chunk_class]) continue;
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
        #pragma omp for schedule(dynamic)
        for (unsigned int c = 0; c < tm->classes; c++)
          tk_learn_absorb_class(tm, c);
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
      free(spec_bufs);
      free(spec_data);
    }
  } else {
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int *shuffle = shuffles[tid];
      unsigned int literals[TK_CVEC_BITS];
      unsigned int votes[TK_CVEC_BITS];
      uint8_t *spec_data = (uint8_t *)malloc((size_t)classes * spec_per_class);
      tk_spec_buf_t *spec_bufs = (tk_spec_buf_t *)malloc(classes * sizeof(tk_spec_buf_t));
      for (unsigned int c = 0; c < classes; c++) {
        spec_bufs[c].buf = spec_data + (size_t)c * spec_per_class;
        spec_bufs[c].size = spec_per_class;
        spec_bufs[c].cursor = spec_per_class;
        spec_bufs[c].sp = tm->specificity_pts[c];
      }
      for (unsigned int iter = 0; iter < max_iter; iter++) {
        if (break_flag) break;
        tk_learn_init_shuffle(shuffle, shuffle_n);
        #pragma omp for schedule(dynamic)
        for (unsigned int chunk = 0; chunk < total_chunks; chunk++) {
          unsigned int chunk_class = chunk / clause_chunks;
          unsigned int local_class = chunk_class;
          switch (target_mode) {
            case TM_TARGET_IVEC:
              TM_REGRESSION_INNER_LOOP(
                _sample_is_pos ? 1.0 : -1.0,
                bool _sample_is_pos = false;
                for (int64_t _k = sol_offsets[sample]; _k < sol_offsets[sample + 1]; _k++)
                  if ((unsigned int)sol_neighbors[_k] == chunk_class) { _sample_is_pos = true; break; }
                if (!_sample_is_pos && tk_fast_random() < skip_thresholds[chunk_class]) continue;
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
        #pragma omp for schedule(dynamic)
        for (unsigned int c = 0; c < tm->classes; c++)
          tk_learn_absorb_class(tm, c);
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
      free(spec_bufs);
      free(spec_data);
    }
  }

  for (int t = 0; t < max_threads; t++)
    free(shuffles[t]);
  free(shuffles);
  if (skip_thresholds)
    free(skip_thresholds);
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
  uint8_t version = 3;
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
  u32 = (uint32_t)tm->n_tokens; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->n_code_bits; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  u32 = (uint32_t)tm->n_orig_dims; tk_lua_fwrite(L, &u32, sizeof(uint32_t), 1, fh);
  if (tm->n_orig_dims > 0) {
    unsigned int cpb = TK_CVEC_BITS_BYTES(tm->n_code_bits * 2);
    tk_lua_fwrite(L, tm->dim_prefixes, 1, (size_t)tm->n_orig_dims * cpb, fh);
    uint8_t fe = tm->flat_evict ? 1 : 0;
    tk_lua_fwrite(L, &fe, sizeof(uint8_t), 1, fh);
    tk_lua_fwrite(L, &tm->flat_skip_thresh, sizeof(uint32_t), 1, fh);
  }
  size_t mapping_size = (size_t)tm->classes * tm->features * sizeof(int64_t);
  tk_lua_fwrite(L, tm->mapping, 1, mapping_size, fh);
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
  size_t mapping_size = (size_t)tm->classes * tm->features * sizeof(int64_t);
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
  size_t mapping_size = (size_t)tm->classes * tm->features * sizeof(int64_t);
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
    if (tm->managed_dense && !(tm->dense_batch_size > 0 && tm->dense_batch_size < tm->classes))
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
  unsigned int new_features = tk_lua_foptunsigned(L, 2, "reconfigure", "features", tm->features);
  if (new_features != tm->features && tm->n_code_bits > 0)
    new_features += tm->n_code_bits;
  bool features_changed = (new_features != tm->features);
  if (features_changed) {
    tm->features = new_features;
    tm->input_bits = 2 * tm->features;
    uint64_t tail_bits = tm->input_bits & (TK_CVEC_BITS - 1);
    tm->tail_mask = tail_bits ? (uint8_t)((1u << tail_bits) - 1) : 0xFF;
    tm->input_chunks = TK_CVEC_BITS_BYTES(tm->input_bits);
    uint64_t mapping_needed = (uint64_t)tm->classes * tm->features * sizeof(int64_t);
    if (mapping_needed > tm->mapping_alloc) {
      free(tm->mapping);
      tm->mapping = (int64_t *)malloc(mapping_needed);
      tm->mapping_alloc = mapping_needed;
    }
    unsigned int scratch_threads = (unsigned int)tm->absorb_threads;
    uint64_t scratch_needed = (uint64_t)scratch_threads * tm->features * 2 * sizeof(unsigned int);
    if (scratch_needed > tm->scratch_alloc) {
      free(tm->absorb_scratch);
      tm->absorb_scratch = (uint8_t *)malloc(scratch_needed);
      tm->scratch_alloc = scratch_needed;
    }
  }
  unsigned int new_clause_chunks = new_clauses * 2;
  new_clauses = new_clause_chunks * TK_CVEC_BITS;
  unsigned int clause_mult = tm->classes;
  size_t new_action_chunks = (size_t)clause_mult * new_clauses * tm->input_chunks;
  size_t new_state_chunks = (size_t)clause_mult * new_clauses * (tm->state_bits - 1) * tm->input_chunks;
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
  tm->action_chunks = new_action_chunks;
  tm->state_chunks = new_state_chunks;
  tk_learn_apply_fractions(tm, L, 2, "reconfigure");
  tm->automata.n_clauses = clause_mult * tm->clauses;
  tm->automata.n_chunks = tm->input_chunks;
  tm->automata.tail_mask = tm->tail_mask;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;
  tm->trained = false;
  tm->dense_batch_size = tk_lua_foptunsigned(L, 2, "reconfigure", "class_batch", tm->dense_batch_size);
  tm->absorb_threshold = tk_lua_foptunsigned(L, 2, "reconfigure", "absorb_threshold", tm->absorb_threshold);
  tm->absorb_insert = tk_lua_foptunsigned(L, 2, "reconfigure", "absorb_insert", tm->absorb_insert);
  if (tm->n_orig_dims > 0) {
    double fs = tk_lua_foptnumber(L, 2, "reconfigure", "flat_skip", (double)tm->flat_skip_thresh / 4294967295.0);
    tm->flat_skip_thresh = (uint32_t)(fs * 4294967295.0);
  }
  tm->dense_stale = true;
  memset(tm->active, 0, (uint64_t)tm->classes * tm->active_bytes);
  if (tm->absorb_cursor)
    memset(tm->absorb_cursor, 0, tm->classes * sizeof(unsigned int));
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
  if (tm->mapping) {
    size_t bytes_per_class_mapping = (size_t)tm->features * sizeof(int64_t);
    tk_learn_restrict_buffer((char *)tm->mapping, bytes_per_class_mapping, tm->classes, new_classes, keep->a);
  }
  if (tm->active) {
    tk_learn_restrict_buffer((char *)tm->active, tm->active_bytes, tm->classes, new_classes, keep->a);
  }
  if (tm->absorb_cursor) {
    for (unsigned int i = 0; i < new_classes; i++)
      tm->absorb_cursor[i] = tm->absorb_cursor[(unsigned int)keep->a[i]];
  }
  for (unsigned int i = 0; i < new_classes; i++) {
    unsigned int src = (unsigned int)keep->a[i];
    tm->clause_tolerances[i] = tm->clause_tolerances[src];
    tm->clause_maximums[i] = tm->clause_maximums[src];
    tm->specificity_thresholds[i] = tm->specificity_thresholds[src];
    tm->targets[i] = tm->targets[src];
    tm->specificity_pts[i] = tm->specificity_pts[src];
    tm->y_min[i] = tm->y_min[src];
    tm->y_max[i] = tm->y_max[src];
  }
  tm->classes = new_classes;
  tm->class_chunks = TK_CVEC_BITS_BYTES(tm->classes);
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->dense_stale = true;
  return 0;
}

static inline void _tk_learn_load (lua_State *L, tk_learn_t *tm, FILE *fh)
{
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKtm", 4) != 0)
    luaL_error(L, "invalid TM file (bad magic)");
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 3)
    luaL_error(L, "unsupported TM version %d (expected 3)", (int)version);
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
  tm->specificity_pts = (uint8_t *)tk_malloc(L, tm->classes * sizeof(uint8_t));
  tk_lua_fread(L, tm->clause_tolerances, sizeof(unsigned int), tm->classes, fh);
  tk_lua_fread(L, tm->targets, sizeof(unsigned int), tm->classes, fh);
  for (unsigned int c = 0; c < tm->classes; c++) {
    tm->clause_maximums[c] = tm->clause_maximum;
    tm->specificity_thresholds[c] = tm->specificity_threshold;
  }
  tm->automata.n_clauses = tm->classes * tm->clauses;
  tm->automata.n_chunks = tm->input_chunks;
  tm->automata.state_bits = tm->state_bits;
  tm->automata.tail_mask = tm->tail_mask;
  tm->automata.counts = tm->state;
  tm->automata.actions = tm->actions;
  tk_learn_recompute_specificity_pts(tm);
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->n_tokens = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->n_code_bits = u32;
  tk_lua_fread(L, &u32, sizeof(uint32_t), 1, fh); tm->n_orig_dims = u32;
  if (tm->n_orig_dims > 0) {
    tm->code_prefix_bytes = TK_CVEC_BITS_BYTES(tm->n_code_bits * 2);
    tm->dim_prefixes = (uint8_t *)malloc((uint64_t)tm->n_orig_dims * tm->code_prefix_bytes);
    tk_lua_fread(L, tm->dim_prefixes, 1, (size_t)tm->n_orig_dims * tm->code_prefix_bytes, fh);
    uint8_t fe = 0;
    tk_lua_fread(L, &fe, sizeof(uint8_t), 1, fh);
    tm->flat_evict = (fe != 0);
    tk_lua_fread(L, &tm->flat_skip_thresh, sizeof(uint32_t), 1, fh);
  }
  tm->active_bytes = TK_CVEC_BITS_BYTES(tm->n_tokens + tm->n_code_bits);
  tm->active = (uint8_t *)calloc((uint64_t)tm->classes * tm->active_bytes, 1);
  tm->mapping_alloc = (uint64_t)tm->classes * tm->features * sizeof(int64_t);
  tm->mapping = (int64_t *)malloc(tm->mapping_alloc);
  tk_lua_fread(L, tm->mapping, 1, tm->mapping_alloc, fh);
  for (unsigned int c = 0; c < tm->classes; c++) {
    for (unsigned int k = 0; k < tm->features; k++) {
      int64_t tok = tm->mapping[(uint64_t)c * tm->features + k];
      if (tok >= 0)
        sparse_set_active(tm, c, (unsigned int)tok);
    }
  }
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
