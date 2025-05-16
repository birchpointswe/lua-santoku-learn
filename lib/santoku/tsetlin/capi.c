/*

Copyright (C) 2024 Birch Point SWE (Persist to and restore from disk)
Copyright (C) 2024 Birch Point SWE (Lua integration, train/evaluate, active clause)
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

#include "lua.h"
#include "lauxlib.h"
#include "conf.h"

#define TK_TSETLIN_MT "santoku_tsetlin"

typedef enum {
  TM_CLASSIFIER,
  TM_ENCODER,
} tsetlin_type_t;

typedef enum {
  TM_INIT,
  TM_DONE,
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
} tsetlin_classifier_stage_t;

struct tsetlin_classifier_s;
typedef struct tsetlin_classifier_s tsetlin_classifier_t;

typedef struct {

  tsetlin_classifier_t *tm;

  unsigned int sfirst;
  unsigned int slast;

  unsigned int cfirst;
  unsigned int clast;

  unsigned int class_first;
  unsigned int class_last;

  unsigned int index;
  unsigned int sigid;

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
  // tk_bits_t *active_clause; // clause_chunks*
  long int *scores; // classes x samples

} tsetlin_classifier_thread_t;

typedef struct tsetlin_classifier_s {

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

  // double active;
  double negative;
  double *specificity;

  bool created_threads;

  pthread_mutex_t mutex;
  pthread_cond_t cond_stage;
  pthread_cond_t cond_done;
  unsigned int n_threads;
  unsigned int n_threads_done;
  pthread_t *threads;
  tsetlin_classifier_stage_t stage;
  tsetlin_classifier_thread_t *thread_data;

  size_t results_len;
  unsigned int *results;

  size_t encodings_len;
  tk_bits_t *encodings;

  size_t sums_len;
  atomic_long *sums; // samples x classes
  unsigned int sigid;

} tsetlin_classifier_t;

typedef struct {
  tsetlin_type_t type;
  bool has_state;
  tsetlin_classifier_t *classifier;
} tsetlin_t;

#define tm_state_old_sum(tm, thread, chunk, sample) \
  ((tm)->thread_data[thread].sums_old[ \
    (chunk) * (tm)->thread_data[(thread)].train.n + \
    (sample)])

#define tm_state_sum_local(tm, thread, class, sample) \
  ((tm)->thread_data[thread].sums_local[ \
    (class) * (tm)->thread_data[(thread)].predict.n + \
    (sample)])

// #define tm_state_active_clause(tm, thread) \
//   ((tm)->thread_data[thread].active_clause)

#define tm_state_shuffle(tm, thread) \
  ((tm)->thread_data[thread].shuffle)

#define tm_state_scores(tm, thread, chunk, sample) \
  ((tm)->thread_data[thread].scores[ \
    ((chunk) - (tm)->thread_data[thread].cfirst) * (tm)->thread_data[(thread)].predict.n + \
    (sample)])

#define tm_state_clause_chunks(tm, thread) \
  ((tm)->thread_data[thread].clast - (tm)->thread_data[thread].cfirst + 1)

#define tm_state_sum(tm, class, sample) \
  (&(tm)->sums[(sample) * (tm)->classes + \
               (class)])

#define tm_state_counts(tm, clause, input_chunk) \
  (&(tm)->state[(clause) * (tm)->input_chunks * ((tm)->state_bits - 1) + \
                (input_chunk) * ((tm)->state_bits - 1)])

#define tm_state_actions(tm, clause) \
  (&(tm)->actions[(clause) * (tm)->input_chunks])

static inline unsigned int contrastive_loss (
  tk_bits_t *enc_a,
  tk_bits_t *enc_b,
  unsigned int margin,
  unsigned int chunks,
  bool label
) {
  unsigned int dist = hamming(enc_a, enc_b, chunks);
  if (label) {
    if (dist < margin)
      return 0;
    return margin - dist;
  } else {
    if (dist > margin)
      return 0;
    return dist - margin;
  }
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

static inline int tk_lua_absindex (lua_State *L, int i)
{
  if (i < 0 && i > LUA_REGISTRYINDEX)
    i += lua_gettop(L) + 1;
  return i;
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

static inline void *tk_realloc (
  lua_State *L,
  void *p,
  size_t s
) {
  p = realloc(p, s);
  if (!p) {
    tk_error(L, "realloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline void *tk_malloc_interleaved (
  lua_State *L,
  size_t *sp,
  size_t s
) {
  void *p = (numa_available() == -1) ? malloc(s) : numa_alloc_interleaved(s);
  if (!p) {
    tk_error(L, "malloc failed", ENOMEM);
    return NULL;
  } else {
    *sp = s;
    return p;
  }
}

static inline void *tk_ensure_interleaved (
  lua_State *L,
  size_t *s1p,
  void *p0,
  size_t s1,
  bool copy
) {
  size_t s0 = *s1p;
  if (s1 <= s0)
    return p0;
  void *p1 = tk_malloc_interleaved(L, s1p, s1);
  if (!p1) {
    tk_error(L, "realloc failed", ENOMEM);
    return NULL;
  } else {
    if (copy)
      memcpy(p1, p0, s0);
    numa_free(p0, s0);
    return p1;
  }
}

static inline void *tk_malloc_aligned (
  lua_State *L,
  size_t s,
  size_t a
) {
  void *p = NULL;
  if (posix_memalign((void **)&p, a, s) != 0)
    tk_error(L, "malloc failed", ENOMEM);
  return p;
}

static inline void *tk_malloc (
  lua_State *L,
  size_t s
) {
  void *p = malloc(s);
  if (!p) {
    tk_error(L, "malloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline int tk_lua_verror (lua_State *L, int n, ...) {
  va_list args;
  va_start(args, n);
  for (int i = 0; i < n; i ++) {
    const char *str = va_arg(args, const char *);
    lua_pushstring(L, str);
  }
  va_end(args);
  tk_lua_callmod(L, n, 0, "santoku.error", "error");
  return 0;
}

static inline int tk_lua_error (lua_State *L, const char *err)
{
  lua_pushstring(L, err);
  tk_lua_callmod(L, 1, 0, "santoku.error", "error");
  return 0;
}

// TODO: include the field name in error
static inline lua_Integer tk_lua_ftype (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  int t = lua_type(L, -1);
  lua_pop(L, 1);
  return t;
}

static inline const char *tk_lua_checkstring (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checkstring(L, i);
}

static inline const char *tk_lua_fchecklstring (lua_State *L, int i, size_t *lp, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, field, "field is not a string");
  const char *s = luaL_checklstring(L, -1, lp);
  lua_pop(L, 1);
  return s;
}

static inline const char *tk_lua_fcheckstring (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, field, "field is not a string");
  const char *s = luaL_checkstring(L, -1);
  lua_pop(L, 1);
  return s;
}

static inline double tk_lua_checkposdouble (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive number");
  lua_Number l = luaL_checknumber(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive number");
  return (double) l;
}

static inline double tk_lua_optposdouble (lua_State *L, int i, double def, char *name)
{
  if (lua_type(L, i) < 1)
    return def;
  lua_Number l = luaL_checknumber(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  return (double) l;
}

static inline bool tk_lua_checkboolean (lua_State *L, int i)
{
  if (lua_type(L, i) == LUA_TNIL)
    return false;
  luaL_checktype(L, i, LUA_TBOOLEAN);
  return lua_toboolean(L, i);
}

static inline double tk_lua_fcheckposdouble (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_Number l = luaL_checknumber(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_pop(L, 1);
  return l;
}

static inline double tk_lua_foptposdouble (lua_State *L, int i, double def, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) == LUA_TNIL) {
    lua_pop(L, 1);
    return def;
  }
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_Number l = luaL_checknumber(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_pop(L, 1);
  return l;
}

static inline unsigned int tk_lua_checkunsigned (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  if (l > UINT_MAX)
    luaL_error(L, "value is too large");
  return (unsigned int) l;
}

static inline unsigned int tk_lua_fcheckunsigned (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_pop(L, 1);
  return l;
}

static inline bool tk_lua_fcheckboolean (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TBOOLEAN)
    tk_lua_verror(L, 3, name, field, "field is not a boolean");
  luaL_checktype(L, -1, LUA_TBOOLEAN);
  bool n = lua_toboolean(L, -1);
  lua_pop(L, 1);
  return n;
}

static inline unsigned int tk_lua_foptunsigned (lua_State *L, int i, bool def, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) == LUA_TNIL) {
    lua_pop(L, 1);
    return def;
  }
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_pop(L, 1);
  return l;
}

static inline unsigned int tk_lua_optunsigned (lua_State *L, int i, unsigned int def, char *name)
{
  if (lua_type(L, i) == LUA_TNIL)
    return def;
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_pop(L, 1);
  return l;
}

static inline bool tk_lua_optboolean (lua_State *L, int i, bool def, char *name)
{
  if (lua_type(L, i) == LUA_TNIL)
    return def;
  if (lua_type(L, i) != LUA_TBOOLEAN)
    tk_lua_verror(L, 3, name, "value is not a boolean");
  return lua_toboolean(L, i);
}

static inline bool tk_lua_foptboolean (lua_State *L, int i, bool def, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) == LUA_TNIL) {
    lua_pop(L, 1);
    return def;
  }
  if (lua_type(L, -1) != LUA_TBOOLEAN)
    tk_lua_verror(L, 3, name, field, "field is not a boolean or nil");
  bool b = lua_toboolean(L, -1);
  lua_pop(L, 1);
  return b;
}

static inline int tk_lua_errmalloc (lua_State *L)
{
  lua_pushstring(L, "Error in malloc");
  tk_lua_callmod(L, 1, 0, "santoku.error", "error");
  return 0;
}

static inline int tk_lua_errno (lua_State *L, int err)
{
  lua_pushstring(L, strerror(errno));
  lua_pushinteger(L, err);
  tk_lua_callmod(L, 2, 0, "santoku.error", "error");
  return 0;
}

static inline FILE *tk_lua_tmpfile (lua_State *L)
{
  FILE *fh = tmpfile();
  if (fh) return fh;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error opening tmpfile");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
  return NULL;
}

static inline FILE *tk_lua_fmemopen (lua_State *L, char *data, size_t size, const char *flag)
{
  FILE *fh = fmemopen(data, size, flag);
  if (fh) return fh;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error opening string as file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
  return NULL;
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

static inline void tk_lua_fseek (lua_State *L, size_t size, size_t memb, FILE *fh)
{
  int r = fseek(fh, (long) (size * memb), SEEK_CUR);
  if (!ferror(fh) || !r) return;
  int e = errno;
  lua_settop(L, 0);
  lua_pushstring(L, "Error reading from file");
  lua_pushstring(L, strerror(e));
  lua_pushinteger(L, e);
  tk_lua_callmod(L, 3, 0, "santoku.error", "error");
}

static inline char *tk_lua_fslurp (lua_State *L, FILE *fh, size_t *len)
{
  if (fseek(fh, 0, SEEK_END) != 0) {
    tk_lua_errno(L, errno);
    return NULL;
  }
  long size = ftell(fh);
  if (size < 0) {
    tk_lua_errno(L, errno);
    return NULL;
  }
  if (fseek(fh, 0, SEEK_SET) != 0) {
    tk_lua_errno(L, errno);
    return NULL;
  }
  char *buffer = malloc((size_t) size);
  if (!buffer) {
    tk_lua_errmalloc(L);
    return NULL;
  }
  if (fread(buffer, 1, (size_t) size, fh) != (size_t) size) {
    free(buffer);
    tk_lua_errno(L, errno);
    return NULL;
  }
  *len = (size_t) size;
  return buffer;
}

static inline void tk_lua_register (lua_State *L, luaL_Reg *regs, int nup)
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
  if (active >= n) active = n;
  if (active < 0) active = 0;
  for (unsigned int i = 0; i < active; i ++) {
    unsigned int r = fast_rand();
    unsigned int f = ((uint64_t)r * n) >> 32;
    unsigned int chunk = BITS_DIV(f);
    unsigned int bit = BITS_MOD(f);
    // Note: This doesn't seem to add much except extra runtime..
    // while (mask[chunk] & ((tk_bits_t)1 << bit)) {
    //   r = fast_rand();
    //   f = ((uint64_t)r * n) >> 32;
    //   chunk = BITS_DIV(f);
    //   bit = BITS_MOD(f);
    // }
    mask[chunk] |= ((tk_bits_t)1 << bit);
  }
}

static inline void tm_inc (
  tsetlin_classifier_t *tm,
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
  tsetlin_classifier_t *tm,
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
  tsetlin_classifier_t *tm,
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
  tsetlin_classifier_t *tm,
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
  tsetlin_classifier_t *tm,
  tk_bits_t *input,
  tk_bits_t out,
  unsigned int s,
  unsigned int target_class,
  unsigned int target_vote,
  // tk_bits_t active,
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
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int boost_true_positive = tm->boost_true_positive;
  double *specificity = tm->specificity + (chunk % clause_chunks) * BITS;
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
    // if (fast_chance(p) && (active & ((tk_bits_t)1 << j))) {
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
        tk_tsetlin_init_streams(feedback_to_la, 2 * features, specificity[j]);
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

// static inline void tk_tsetlin_init_active (
//   tsetlin_classifier_t *tm,
//   unsigned int thread
// ) {
//   double active = tm->active;
//   unsigned int clause_chunks = tm_state_clause_chunks(tm, thread);
//   tk_bits_t *active_clause = tm_state_active_clause(tm, thread);
//   for (unsigned int i = 0; i < clause_chunks; i ++)
//     active_clause[i] = ALL_MASK;
//   for (unsigned int i = 0; i < clause_chunks; i ++)
//     for (unsigned int j = 0; j < BITS; j ++)
//       if (!fast_chance(active))
//         active_clause[i] &= ~((tk_bits_t)1 << j);
// }

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

// static inline void tk_tsetlin_re_shuffle (
//   unsigned int *shuffle,
//   unsigned int n
// ) {
//   for (unsigned int i = n - 1; i > 0; i --) {
//     unsigned int j = fast_rand() % (i + 1);
//     unsigned int tmp = shuffle[i];
//     shuffle[i] = shuffle[j];
//     shuffle[j] = tmp;
//   }
// }

tsetlin_t *tk_tsetlin_peek (lua_State *L, int i)
{
  return (tsetlin_t *) luaL_checkudata(L, i, TK_TSETLIN_MT);
}

static inline tsetlin_t *tk_tsetlin_alloc (lua_State *L)
{
  tsetlin_t *tm = lua_newuserdata(L, sizeof(tsetlin_t)); // ud
  if (!tm) luaL_error(L, "error in malloc during creation");
  memset(tm, 0, sizeof(tsetlin_t));
  luaL_getmetatable(L, TK_TSETLIN_MT); // ud mt
  lua_setmetatable(L, -2); // ud
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_classifier (lua_State *L, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_CLASSIFIER;
  tm->has_state = has_state;
  tm->classifier = tk_malloc(L, sizeof(tsetlin_classifier_t));
  memset(tm->classifier, 0, sizeof(tsetlin_classifier_t));
  return tm;
}

static inline tsetlin_t *tk_tsetlin_alloc_encoder (lua_State *L, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc(L);
  tm->type = TM_ENCODER;
  tm->has_state = has_state;
  tm->classifier = tk_malloc(L, sizeof(tsetlin_classifier_t));
  memset(tm->classifier, 0, sizeof(tsetlin_classifier_t));
  return tm;
}

static inline void tk_tsetlin_wait_for_threads (
  pthread_mutex_t *mutex,
  pthread_cond_t *cond_done,
  unsigned int *n_threads_done,
  unsigned int n_threads
) {
  pthread_mutex_lock(mutex);
  while ((*n_threads_done) < n_threads)
    pthread_cond_wait(cond_done, mutex);
  pthread_mutex_unlock(mutex);
}

static inline void tk_tsetlin_signal (
  int stage,
  unsigned int *sigid,
  int *stagep,
  pthread_mutex_t *mutex,
  pthread_cond_t *cond_stage,
  pthread_cond_t *cond_done,
  unsigned int *n_threads_done,
  unsigned int n_threads
) {
  pthread_mutex_lock(mutex);
  (*sigid) ++;
  (*stagep) = stage;
  (*n_threads_done) = 0;
  pthread_cond_broadcast(cond_stage);
  pthread_mutex_unlock(mutex);
  tk_tsetlin_wait_for_threads(mutex, cond_done, n_threads_done, n_threads);
  pthread_cond_broadcast(cond_stage);
}

static inline void tk_tsetlin_setup_threads (lua_State *L, tsetlin_classifier_t *tm, unsigned int n_threads);

static inline void tk_tsetlin_init_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm,
  unsigned int classes,
  unsigned int features,
  unsigned int clauses,
  unsigned int state_bits,
  double thresholdf,
  bool boost_true_positive,
  double specificity_high,
  double specificity_low,
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
  tm->classes = classes;
  tm->class_chunks = BITS_DIV((tm->classes - 1)) + 1;
  tm->clauses = clauses;
  tm->threshold = ceil(thresholdf >= 1 ? thresholdf : fmaxf(1.0, (double) clauses / thresholdf));
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
  tm->sigid = 0;
  tm->specificity = tk_malloc(L, tm->clauses * sizeof(double));
  for (size_t i = 0; i < tm->clauses; i ++)
    tm->specificity[i] = (1.0 * i / tm->clauses) * (specificity_high - specificity_low) + specificity_low;
  if (!(tm->state && tm->actions))
    luaL_error(L, "error in malloc during creation of classifier");
  tk_tsetlin_setup_threads(L, tm, n_threads);
}

static inline int tk_tsetlin_init_encoder (
  lua_State *L,
  tsetlin_classifier_t *tm,
  unsigned int encoding_bits,
  unsigned int features,
  unsigned int clauses,
  unsigned int state_bits,
  double thresholdf,
  bool boost_true_positive,
  double specificity_high,
  double specificity_low,
  unsigned int n_threads
) {
  if (BITS_MOD(encoding_bits))
    tk_lua_verror(L, 3, "create encoder", "hidden", "must be a multiple of " STR(BITS));
  tk_tsetlin_init_classifier(L, tm,
      encoding_bits, features, clauses, state_bits, thresholdf,
      boost_true_positive, specificity_high, specificity_low, n_threads);
  return 0;
}

static inline unsigned int tk_tsetlin_get_nthreads (
  lua_State *L, int i, char *name, char *field
) {
  long ts;
  unsigned int n_threads;
  if (field != NULL)
    n_threads = tk_lua_foptunsigned(L, i, 0, name, field);
  else
    n_threads = tk_lua_optunsigned(L, i, 0, name);
  if (n_threads)
    return n_threads;
  ts = sysconf(_SC_NPROCESSORS_ONLN) - 1;
  if (ts <= 0)
    return (unsigned int) tk_lua_verror(L, 3, name, "sysconf", errno);
  lua_pushinteger(L, ts);
  n_threads = tk_lua_checkunsigned(L, -1, "sysconf");
  lua_pop(L, 1);
  return n_threads;
}

static inline void tk_tsetlin_create_classifier (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, true);
  lua_insert(L, 1);

  tk_tsetlin_init_classifier(L, tm->classifier,
      tk_lua_fcheckunsigned(L, 2, "create classifier", "classes"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "features"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "clauses"),
      tk_lua_fcheckunsigned(L, 2, "create classifier", "state"),
      tk_lua_fcheckposdouble(L, 2, "create classifier", "target"),
      tk_lua_fcheckboolean(L, 2, "create classifier", "boost"),
      tk_lua_foptposdouble(L, 2, 2.0, "create encoder", "specificity_high"),
      tk_lua_foptposdouble(L, 2, 200.0, "create encoder", "specificity_low"),
      tk_tsetlin_get_nthreads(L, 2, "create classifier", "threads"));

  lua_settop(L, 1);
}

static inline void tk_tsetlin_create_encoder (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, true);
  lua_insert(L, 1);

  double specificity_high = tk_lua_foptposdouble(L, 2, 2.0, "create encoder", "specificity_high");
  double specificity_low = tk_lua_foptposdouble(L, 2, 200.0, "create encoder", "specificity_low");

  tk_tsetlin_init_encoder(L, tm->classifier,
      tk_lua_fcheckunsigned(L, 2, "create encoder", "hidden"),
      tk_lua_fcheckunsigned(L, 2, "create encoder", "visible"),
      tk_lua_fcheckunsigned(L, 2, "create encoder", "clauses"),
      tk_lua_fcheckunsigned(L, 2, "create encoder", "state"),
      tk_lua_fcheckposdouble(L, 2, "create encoder", "target"),
      tk_lua_fcheckboolean(L, 2, "create encoder", "boost"),
      specificity_high,
      specificity_low,
      tk_tsetlin_get_nthreads(L, 2, "create encoder", "threads"));

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
static inline void tk_classifier_shrink (tsetlin_classifier_t *tm)
{
  if (tm == NULL) return;
  free(tm->state); tm->state = NULL;
  if (numa_available() == -1) {
    free(tm->sums); tm->sums = NULL;
  } else {
    numa_free(tm->sums, tm->sums_len); tm->sums = NULL; tm->sums_len = 0;
  }
  if (tm->created_threads)
    for (unsigned int i = 0; i < tm->n_threads; i ++) {
      free(tm->thread_data[i].shuffle); tm->thread_data[i].shuffle = NULL;
      free(tm->thread_data[i].sums_old); tm->thread_data[i].sums_old = NULL;
      free(tm->thread_data[i].sums_local); tm->thread_data[i].sums_local = NULL;
    }
}

static inline void tk_classifier_destroy (tsetlin_classifier_t *tm)
{
  if (tm == NULL) return;
  if (tm->destroyed) return;
  tm->destroyed = true;
  tk_classifier_shrink(tm);
  free(tm->actions); tm->actions = NULL;
  if (numa_available() == -1) {
    free(tm->results); tm->results = NULL;
    free(tm->encodings); tm->encodings = NULL;
  } else {
    numa_free(tm->results, tm->results_len); tm->results = NULL; tm->results_len = 0;
    numa_free(tm->encodings, tm->encodings_len); tm->encodings = NULL; tm->encodings_len = 0;
  }
  tk_tsetlin_signal(
    (int) TM_DONE, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);
  // TODO: What is the right way to deal with potential thread errors (or other
  // errors, for that matter) during the finalizer?
  if (tm->created_threads)
    for (unsigned int i = 0; i < tm->n_threads; i ++) {
      free(tm->thread_data[i].scores); tm->thread_data[i].scores = NULL;
      pthread_join(tm->threads[i], NULL);
      // if (pthread_join(tm->threads[i], NULL) != 0)
      //   tk_error(L, "pthread_join", errno);
    }
  pthread_mutex_destroy(&tm->mutex);
  pthread_cond_destroy(&tm->cond_stage);
  pthread_cond_destroy(&tm->cond_done);
  free(tm->threads); tm->threads = NULL;
  free(tm->thread_data); tm->thread_data = NULL;
}

static inline int tk_tsetlin_destroy (lua_State *L)
{
  lua_settop(L, 1);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  tk_classifier_destroy(tm->classifier);
  free(tm->classifier);
  return 0;
}

static void tk_classifier_predict_reduce_thread (
  tsetlin_classifier_t *tm,
  unsigned int sfirst,
  unsigned int slast
) {
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int *results = tm->results;
  long int sums[tm->classes];
  for (unsigned int s = sfirst; s <= slast; s ++) {
    for (unsigned int i = 0; i < tm->classes; i ++)
      sums[i] = 0;
    for (unsigned int t = 0; t < tm->n_threads; t ++)
      for (unsigned int chunk = tm->thread_data[t].cfirst; chunk <= tm->thread_data[t].clast; chunk ++) {
        unsigned int class = chunk / clause_chunks;
        sums[class] += tm_state_scores(tm, t, chunk, s);
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
  tsetlin_classifier_t *tm,
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
  tsetlin_classifier_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  return tk_classifier_predict_thread(tm, n, ps, cfirst, clast, thread);
}

static inline void tk_tsetlin_setup_thread_samples (
  tsetlin_classifier_t *tm,
  unsigned int n
) {
  unsigned int sslice = n / tm->n_threads;
  unsigned int sremaining = n % tm->n_threads;
  unsigned int sfirst = 0;
  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    unsigned int extra = sremaining ? 1 : 0;
    unsigned int count = sslice + extra;
    if (sremaining) sremaining--;
    tm->thread_data[i].sfirst = sfirst;
    tm->thread_data[i].slast = sfirst + count - 1;
    sfirst += count;
  }
}

static inline int tk_tsetlin_predict_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm
) {
  lua_settop(L, 3);
  tk_bits_t *ps = (tk_bits_t *) tk_lua_checkstring(L, 2, "argument 1 is not a raw bit-matrix of samples");
  unsigned int n = tk_lua_checkunsigned(L, 3, "argument 2 is not an integer n_samples");

  tm->results = tk_ensure_interleaved(L, &tm->results_len, tm->results, n * sizeof(unsigned int), false);

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    unsigned int chunks = tm_state_clause_chunks(tm, i);
    tm->thread_data[i].predict.n = n;
    tm->thread_data[i].predict.ps = ps;
    tm->thread_data[i].scores = tk_realloc(L, tm->thread_data[i].scores, chunks * n * sizeof(long int));
  }

  tk_tsetlin_setup_thread_samples(tm, n);

  tk_tsetlin_signal(
    (int) TM_CLASSIFIER_PREDICT, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  tk_tsetlin_signal(
    (int) TM_CLASSIFIER_PREDICT_REDUCE, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  lua_pushlstring(L, (char *) tm->results, n * sizeof(unsigned int));
  return 1;
}

static inline int tk_tsetlin_predict_encoder (
  lua_State *L,
  tsetlin_classifier_t *tm
) {

  lua_settop(L, 3);
  tk_bits_t *ps = (tk_bits_t *) tk_lua_checkstring(L, 2, "argument 1 is not a raw bit-matrix of samples");
  unsigned int n = tk_lua_checkunsigned(L, 3, "argument 2 is not an integer n_samples");

  tm->encodings = tk_ensure_interleaved(L, &tm->encodings_len, tm->encodings, n * tm->class_chunks * sizeof(tk_bits_t), false);

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    unsigned int chunks = tm_state_clause_chunks(tm, i);
    tm->thread_data[i].predict.n = n;
    tm->thread_data[i].predict.ps = ps;
    tm->thread_data[i].scores = tk_realloc(L, tm->thread_data[i].scores, chunks * n * sizeof(long int));
  }

  tk_tsetlin_setup_thread_samples(tm, n);

  tk_tsetlin_signal(
    (int) TM_ENCODER_PREDICT, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  tk_tsetlin_signal(
    (int) TM_ENCODER_PREDICT_REDUCE, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  lua_pushlstring(L, (char *) tm->encodings, n * tm->class_chunks * sizeof(tk_bits_t));
  return 1;
}

static inline int tk_tsetlin_predict (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_predict_classifier(L, tm->classifier);
    case TM_ENCODER:
      return tk_tsetlin_predict_encoder(L, tm->classifier);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in predict");
  }
  return 0;
}

static void tk_classifier_setup_thread (
  tsetlin_classifier_t *tm,
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
  tsetlin_classifier_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_classifier_setup_thread(tm, n, ps, cfirst, clast, thread);
}

static void tk_classifier_prime_thread (
  tsetlin_classifier_t *tm,
  unsigned int n,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  unsigned int clause_chunks = tm->clause_chunks;
  for (unsigned int s = 0; s < n; s ++) {
    for (unsigned int class = cfirst / clause_chunks; class <= clast / clause_chunks; class ++) {
      atomic_long *class_sump = tm_state_sum(tm, class, s);
      tm_state_sum_local(tm, thread, class, s) = atomic_load(class_sump);
    }
  }
}

static void tk_encoder_prime_thread (
  tsetlin_classifier_t *tm,
  unsigned int n,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  tk_classifier_prime_thread(tm, n, cfirst, clast, thread);
}

static void tk_classifier_train_thread (
  tsetlin_classifier_t *tm,
  unsigned int n,
  tk_bits_t *ps,
  unsigned int *ss,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  seed_rand(thread);
  // tk_tsetlin_init_active(tm, thread);
  tk_tsetlin_init_shuffle(tm_state_shuffle(tm, thread), n);

  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int *shuffle = tm_state_shuffle(tm, thread);
  // tk_bits_t *active = tm_state_active_clause(tm, thread);
  tk_bits_t out;

  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {

    // TODO: vectorized pre-calculation of class?
    unsigned int class = chunk / clause_chunks;
    // tk_bits_t act = active[chunk - cfirst];

    for (unsigned int i = 0; i < n; i ++) {
      unsigned int s = shuffle[i];
      // unsigned int s = i; //shuffle[i];
      tk_bits_t *input = ps + s * input_chunks;
      tk_tsetlin_calculate(tm, input, false, &out, chunk, chunk);
      unsigned int target_class = ss[s];
      tm_update(tm, input, out, s, target_class, 1, /*act, */class, chunk, thread);
    }
  }
}

static inline void en_tm_populate (
  tsetlin_classifier_t *tm,
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

static void tk_encoder_train_thread (
  tsetlin_classifier_t *tm,
  tk_bits_t *ps,
  tk_bits_t *ls,
  unsigned int n,
  unsigned int cfirst,
  unsigned int clast,
  unsigned int thread
) {
  seed_rand(thread);
  // tk_tsetlin_init_active(tm, thread);
  tk_tsetlin_init_shuffle(tm_state_shuffle(tm, thread), n);

  unsigned int class_chunks = tm->class_chunks;
  unsigned int clause_chunks = tm->clause_chunks;
  unsigned int input_chunks = tm->input_chunks;
  unsigned int *shuffle = tm_state_shuffle(tm, thread);
  // tk_bits_t *active = tm_state_active_clause(tm, thread);
  tk_bits_t out;

  for (unsigned int chunk = cfirst; chunk <= clast; chunk ++) {

    unsigned int class = chunk / clause_chunks;
    unsigned int enc_chunk = BITS_DIV(class);
    unsigned int enc_bit = BITS_MOD(class);

    // tk_bits_t act = active[chunk - cfirst];

    for (unsigned int i = 0; i < n; i ++) {
      unsigned int s = shuffle[i];
      // unsigned int s = i; //shuffle[i];
      tk_bits_t *input = ps + s * input_chunks;
      tk_tsetlin_calculate(tm, input, false, &out, chunk, chunk);
      bool target = (ls[s * class_chunks + enc_chunk] & ((tk_bits_t) 1 << enc_bit)) > 0;
      tm_update(tm, input, out, s, class, target, /*act, */class, chunk, thread);
    }
  }
}

static void tk_encoder_predict_reduce_thread (
  tsetlin_classifier_t *tm,
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
    for (unsigned int t = 0; t < tm->n_threads; t ++)
      for (unsigned int chunk = tm->thread_data[t].cfirst; chunk <= tm->thread_data[t].clast; chunk ++) {
        unsigned int class = chunk / clause_chunks;
        sums[class] += tm_state_scores(tm, t, chunk, s);
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

static inline int tk_tsetlin_train_classifier (
  lua_State *L,
  tsetlin_classifier_t *tm
) {

  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  tk_bits_t *ps = (tk_bits_t *) tk_lua_fcheckstring(L, 2, "train", "problems");
  unsigned int *ss = (unsigned int *) tk_lua_fcheckstring(L, 2, "train", "solutions");
  unsigned int max_iter =  tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  // tm->active = tk_lua_fcheckposdouble(L, 2, "train", "active");
  tm->negative = tk_lua_fcheckposdouble(L, 2, "train", "negative");

  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    tm->thread_data[i].train.n = n;
    tm->thread_data[i].train.ps = ps;
    tm->thread_data[i].train.ss = ss;
    tm->thread_data[i].shuffle = tk_realloc(L, tm->thread_data[i].shuffle, n * sizeof(unsigned int));
    tm->thread_data[i].sums_old = tk_realloc(L, tm->thread_data[i].sums_old, tm->classes * tm->clause_chunks * n * sizeof(long int));
    tm->thread_data[i].sums_local = tk_realloc(L, tm->thread_data[i].sums_local, tm->classes * n * sizeof(long int));
  }

  tm->sums = tk_ensure_interleaved(L, &tm->sums_len, tm->sums, n * tm->classes * sizeof(atomic_long), false);
  for (unsigned int c = 0; c < tm->classes; c ++)
    for (unsigned int s = 0; s < n; s ++)
      atomic_init(tm_state_sum(tm, c, s), 0);

  tk_tsetlin_setup_thread_samples(tm, n);

  tk_tsetlin_signal(
    (int) TM_CLASSIFIER_SETUP, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  tk_tsetlin_signal(
    (int) TM_CLASSIFIER_PRIME, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  for (unsigned int i = 0; i < max_iter; i ++) {

    tk_tsetlin_signal(
      (int) TM_CLASSIFIER_TRAIN, &tm->sigid,
      (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
      &tm->n_threads_done, tm->n_threads);

    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      lua_call(L, 1, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0)
        break;
      lua_pop(L, 1);
    }

  }

  tk_classifier_shrink(tm);

  tm->trained = true;
  return 0;
}

static inline int tk_tsetlin_train_encoder (
  lua_State *L,
  tsetlin_classifier_t *tm
) {

  unsigned int n = tk_lua_fcheckunsigned(L, 2, "train", "samples");
  size_t ps_len;
  tk_bits_t *ps = (tk_bits_t *) tk_lua_fchecklstring(L, 2, &ps_len, "train", "sentences");
  size_t ls_len;
  tk_bits_t *ls = (tk_bits_t *) tk_lua_fchecklstring(L, 2, &ls_len, "train", "codes");
  unsigned int max_iter =  tk_lua_fcheckunsigned(L, 2, "train", "iterations");
  // tm->active = tk_lua_fcheckposdouble(L, 2, "train", "active");
  tm->negative = 1.0; // Note: unused in encoder
  tm->encodings = tk_ensure_interleaved(L, &tm->encodings_len, tm->encodings, n * tm->class_chunks * sizeof(tk_bits_t), false);

  int i_each = -1;
  if (tk_lua_ftype(L, 2, "each") != LUA_TNIL) {
    lua_getfield(L, 2, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    unsigned int chunks = tm_state_clause_chunks(tm, i);
    tm->thread_data[i].predict.n = n;
    tm->thread_data[i].predict.ps = ps;
    tm->thread_data[i].scores = tk_realloc(L, tm->thread_data[i].scores, chunks * n * sizeof(long int));
    tm->thread_data[i].train.n = n;
    tm->thread_data[i].train.ps = ps;
    tm->thread_data[i].train.ls = ls;
    tm->thread_data[i].shuffle = tk_realloc(L, tm->thread_data[i].shuffle, n * sizeof(unsigned int));
    tm->thread_data[i].sums_old = tk_realloc(L, tm->thread_data[i].sums_old, tm->classes * tm->clause_chunks * n * sizeof(long int));
    tm->thread_data[i].sums_local = tk_realloc(L, tm->thread_data[i].sums_local, tm->classes * n * sizeof(long int));
  }

  tm->sums = tk_ensure_interleaved(L, &tm->sums_len, tm->sums, n * tm->classes * sizeof(atomic_long), false);
  for (unsigned int c = 0; c < tm->classes; c ++)
    for (unsigned int s = 0; s < n; s ++)
      atomic_init(tm_state_sum(tm, c, s), 0);

  tk_tsetlin_setup_thread_samples(tm, n);

  tk_tsetlin_signal(
    (int) TM_ENCODER_SETUP, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  tk_tsetlin_signal(
    (int) TM_ENCODER_PRIME, &tm->sigid,
    (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
    &tm->n_threads_done, tm->n_threads);

  for (unsigned int i = 0; i < max_iter; i ++) {

    tk_tsetlin_signal(
      (int) TM_ENCODER_TRAIN, &tm->sigid,
      (int *) &tm->stage, &tm->mutex, &tm->cond_stage, &tm->cond_done,
      &tm->n_threads_done, tm->n_threads);

    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      lua_call(L, 1, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0)
        break;
      lua_pop(L, 1);
    }

  }

  tk_classifier_shrink(tm);

  tm->trained = true;
  return 0;
}

static inline int tk_tsetlin_train (lua_State *L)
{
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  if (!tm->has_state)
    luaL_error(L, "can't train a model loaded without state");
  switch (tm->type) {
    case TM_CLASSIFIER:
      return tk_tsetlin_train_classifier(L, tm->classifier);
    case TM_ENCODER:
      return tk_tsetlin_train_encoder(L, tm->classifier);
    default:
      return luaL_error(L, "unexpected tsetlin machine type in train");
  }
}

static void *tk_tsetlin_classifier_worker (void *datap)
{
  tsetlin_classifier_thread_t *data =
    (tsetlin_classifier_thread_t *) datap;
  pthread_mutex_lock(&data->tm->mutex);
  data->tm->n_threads_done ++;
  if (data->tm->n_threads_done == data->tm->n_threads)
    pthread_cond_signal(&data->tm->cond_done);
  pthread_mutex_unlock(&data->tm->mutex);
  while (1) {
    pthread_mutex_lock(&data->tm->mutex);
    while (data->sigid == data->tm->sigid)
      pthread_cond_wait(&data->tm->cond_stage, &data->tm->mutex);
    data->sigid = data->tm->sigid;
    tsetlin_classifier_stage_t stage = data->tm->stage;
    pthread_mutex_unlock(&data->tm->mutex);
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
      case TM_DONE:
        break;
      default:
        assert(false);
        break;
    }
    pthread_mutex_lock(&data->tm->mutex);
    data->tm->n_threads_done ++;
    if (data->tm->n_threads_done == data->tm->n_threads)
      pthread_cond_signal(&data->tm->cond_done);
    pthread_mutex_unlock(&data->tm->mutex);
    if (stage == TM_DONE)
      break;
  }
  return NULL;
}

static inline void tk_pin_thread_to_cpu (
  unsigned int thread_index,
  unsigned int n_threads
) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  unsigned int n_nodes = (unsigned int) numa_max_node() + 1;
  unsigned int threads_per_node = n_threads / n_nodes;
  if (threads_per_node == 0) threads_per_node = 1;
  unsigned int node = thread_index / threads_per_node;
  if (node >= n_nodes) node = n_nodes - 1;
  struct bitmask *cpus = numa_allocate_cpumask();
  if (numa_node_to_cpus((int) node, cpus) == 0) {
    unsigned int count = 0;
    for (unsigned int i = 0; i < cpus->size; i ++) {
      if (numa_bitmask_isbitset(cpus, i)) {
        count ++;
      }
    }
    if (count > 0) {
      unsigned int local_index =
        (thread_index - node * threads_per_node) % count;
      unsigned int found = 0;
      for (unsigned int i = 0; i < cpus->size; i ++) {
        if (numa_bitmask_isbitset(cpus, i)) {
          if (found == local_index) {
            CPU_SET(i, &cpuset);
            break;
          }
          found ++;
        }
      }
    }
  }
  numa_free_cpumask(cpus);
  pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

static void *tk_tsetlin_classifier_worker_wrapper (void *arg) {
  tsetlin_classifier_thread_t *td = (tsetlin_classifier_thread_t *)arg;
  if (numa_available() != -1 && numa_max_node() > 0)
    tk_pin_thread_to_cpu(td->index, td->tm->n_threads);
  return tk_tsetlin_classifier_worker(arg);
}

static inline void tk_tsetlin_setup_threads (
  lua_State *L,
  tsetlin_classifier_t *tm,
  unsigned int n_threads
) {
  tm->n_threads = n_threads;
  tm->n_threads_done = 0;
  tm->stage = TM_INIT;
  tm->threads = tk_malloc(L, tm->n_threads * sizeof(pthread_t));
  tm->thread_data = tk_malloc(L, tm->n_threads * sizeof(tsetlin_classifier_thread_t));
  memset(tm->thread_data, 0, tm->n_threads * sizeof(tsetlin_classifier_thread_t));

  // TODO: check errors
  pthread_mutex_init(&tm->mutex, NULL);
  pthread_cond_init(&tm->cond_stage, NULL);
  pthread_cond_init(&tm->cond_done, NULL);

  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    tm->thread_data[i].tm = tm;
    tm->thread_data[i].sigid = 0;
    tm->thread_data[i].index = i;
    if (!tm->created_threads && pthread_create(&tm->threads[i], NULL, tk_tsetlin_classifier_worker_wrapper, &tm->thread_data[i]) != 0)
      tk_error(L, "pthread_create", errno);
  }

  unsigned int total_chunks = tm->clause_chunks * tm->classes;
  unsigned int cslice = total_chunks / tm->n_threads;
  unsigned int cremaining = total_chunks % tm->n_threads;
  unsigned int cfirst = 0;
  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    unsigned int extra = cremaining ? 1 : 0;
    unsigned int count = cslice + extra;
    if (cremaining) cremaining --;
    tm->thread_data[i].cfirst = cfirst;
    tm->thread_data[i].clast = cfirst + count - 1;
    cfirst += count;
  }

  unsigned int clslice = tm->classes / tm->n_threads;
  unsigned int clremaining = tm->classes % tm->n_threads;
  unsigned int clfirst = 0;
  for (unsigned int i = 0; i < tm->n_threads; i ++) {
    unsigned int extra = clremaining ? 1 : 0;
    unsigned int count = clslice + extra;
    if (clremaining) clremaining --;
    tm->thread_data[i].class_first = clfirst;
    tm->thread_data[i].class_last = clfirst + count - 1;
    clfirst += count;
  }

  // for (unsigned int i = 0; i < tm->n_threads; i ++) {
  //   unsigned int clause_chunks = tm_state_clause_chunks(tm, i);
  //   tm->thread_data[i].active_clause = tk_malloc_aligned(L, sizeof(tk_bits_t) * clause_chunks, BITS);
  // }

  tm->created_threads = true;
  tk_tsetlin_wait_for_threads(&tm->mutex, &tm->cond_done, &tm->n_threads_done, tm->n_threads);
}

static inline void _tk_tsetlin_persist_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh, bool persist_state)
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
  tk_lua_fwrite(L, tm->specificity + 0, sizeof(double), 1, fh);
  tk_lua_fwrite(L, tm->specificity + (tm->clauses - 1), sizeof(double), 1, fh);
  tk_lua_fwrite(L, tm->actions, sizeof(tk_bits_t), tm->action_chunks, fh);
}

static inline void tk_tsetlin_persist_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh, bool persist_state)
{
  _tk_tsetlin_persist_classifier(L, tm, fh, persist_state);
}

static inline int tk_tsetlin_persist (lua_State *L)
{
  lua_settop(L, 3);
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
  bool tostr = lua_type(L, 2) == LUA_TNIL;
  FILE *fh = tostr ? tk_lua_tmpfile(L) : tk_lua_fopen(L, tk_lua_checkstring(L, 2, "persist path"), "w");
  bool persist_state = lua_toboolean(L, 3);
  if (persist_state && !tm->has_state)
    luaL_error(L, "can't persist the state of a model loaded without state");
  tk_lua_fwrite(L, &tm->type, sizeof(tsetlin_type_t), 1, fh);
  tk_lua_fwrite(L, &persist_state, sizeof(bool), 1, fh);
  tk_tsetlin_persist_classifier(L, tm->classifier, fh, persist_state);
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

static inline void _tk_tsetlin_load_classifier (lua_State *L, tsetlin_classifier_t *tm, FILE *fh, bool read_state, bool has_state, unsigned int n_threads)
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
  double specificity_high, specificity_low;
  tk_lua_fread(L, &specificity_high, sizeof(double), 1, fh);
  tk_lua_fread(L, &specificity_low, sizeof(double), 1, fh);
  tm->specificity = tk_malloc(L, tm->clauses * sizeof(double));
  for (size_t i = 0; i < tm->clauses; i ++)
    tm->specificity[i] = (1.0 * i / tm->clauses) * (specificity_high - specificity_low) + specificity_low;
  tk_lua_fread(L, tm->specificity, sizeof(double), tm->clauses, fh);
  tm->actions = tk_malloc_aligned(L, sizeof(tk_bits_t) * tm->action_chunks, BITS);
  tk_lua_fread(L, tm->actions, sizeof(tk_bits_t), tm->action_chunks, fh);
  tk_tsetlin_setup_threads(L, tm, n_threads);
}

static inline void tk_tsetlin_load_classifier (lua_State *L, FILE *fh, bool read_state, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc_classifier(L, read_state);
  unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, "threads", NULL);
  _tk_tsetlin_load_classifier(L, tm->classifier, fh, read_state, has_state, n_threads);
}

static inline void tk_tsetlin_load_encoder (lua_State *L, FILE *fh, bool read_state, bool has_state)
{
  tsetlin_t *tm = tk_tsetlin_alloc_encoder(L, read_state);
  unsigned int n_threads = tk_tsetlin_get_nthreads(L, 2, "threads", NULL);
  _tk_tsetlin_load_classifier(L, tm->classifier, fh, read_state, has_state, n_threads);
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
  tsetlin_type_t type;
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
  tsetlin_t *tm = tk_tsetlin_peek(L, 1);
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
