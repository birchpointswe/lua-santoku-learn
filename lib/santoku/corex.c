#define _GNU_SOURCE

#include "lua.h"
#include "lauxlib.h"
#include "roaring.h"
#include "roaring.c"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-function"
#include "kvec.h"
#pragma GCC diagnostic pop

#define MT_COREX "santoku_corex"

typedef enum {
  TK_CMP_INIT,
  TK_CMP_INIT_ALPHA,
  TK_CMP_INIT_TCS,
  TK_CMP_INIT_PYX_UNNORM,
  TK_CMP_MARGINALS,
  TK_CMP_MAXMIS,
  TK_CMP_ALPHA,
  TK_CMP_LATENT_ALL,
  TK_CMP_LATENT_BASELINE,
  TK_CMP_LATENT_SUMS,
  TK_CMP_LATENT_PY,
  TK_CMP_LATENT_NORM,
  TK_CMP_UPDATE_TC,
  TK_CMP_DONE
} tk_corex_stage_t;

struct tk_corex_s;
typedef struct tk_corex_s tk_corex_t;

typedef struct {
  tk_corex_t *C;
  uint64_t n_set_bits;
  unsigned int n_samples;
  unsigned int hfirst;
  unsigned int hlast;
  unsigned int vfirst;
  unsigned int vlast;
  unsigned int index;
  unsigned int sigid;
} tk_corex_thread_data_t;

typedef struct {
  uint64_t s;
  unsigned int v;
} tk_corex_sort_t;

typedef struct tk_corex_s {
  bool trained; // already trained
  bool destroyed;
  double *alpha;
  double *log_marg;
  double *log_py;
  double *log_pyx_unnorm;
  size_t maxmis_len;
  double *maxmis;
  double *mis;
  double *sums;
  double *baseline;
  tk_corex_sort_t *sort;
  size_t samples_len;
  uint64_t *samples;
  size_t visibles_len;
  unsigned int *visibles;
  size_t px_len;
  double *px;
  size_t entropy_len;
  double *entropy_x;
  double *pyx;
  double *counts;
  double last_tc;
  double tc_dev;
  double *tcs;
  double lam;
  double spa;
  double tmin;
  double ttc;
  double smoothing;
  unsigned int n_visible;
  unsigned int n_hidden;
  pthread_mutex_t mutex;
  pthread_cond_t cond_stage;
  pthread_cond_t cond_done;
  unsigned int n_threads;
  unsigned int n_threads_done;
  tk_corex_stage_t stage;
  pthread_t *threads;
  tk_corex_thread_data_t *thread_data;
  bool created_threads;
  unsigned int sigid;
} tk_corex_t;

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

static inline void *tk_lua_checkuserdata (lua_State *L, int i, char *mt)
{
  if (mt == NULL && (lua_islightuserdata(L, i) || lua_isuserdata(L, i)))
    return lua_touserdata(L, i);
  void *p = luaL_checkudata(L, -1, mt);
  lua_pop(L, 1);
  return p;
}

static inline int tk_lua_error (lua_State *L, const char *err)
{
  lua_pushstring(L, err);
  tk_lua_callmod(L, 1, 0, "santoku.error", "error");
  return 0;
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

static inline int tk_lua_errno (lua_State *L, int err)
{
  lua_pushstring(L, strerror(errno));
  lua_pushinteger(L, err);
  tk_lua_callmod(L, 2, 0, "santoku.error", "error");
  return 0;
}

static inline int tk_lua_errmalloc (lua_State *L)
{
  lua_pushstring(L, "Error in malloc");
  tk_lua_callmod(L, 1, 0, "santoku.error", "error");
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

static inline const char *tk_lua_fchecklstring (lua_State *L, int i, char *field, size_t *len)
{
  lua_getfield(L, i, field);
  const char *s = luaL_checklstring(L, -1, len);
  lua_pop(L, 1);
  return s;
}

static inline unsigned int tk_lua_checkunsigned (lua_State *L, int i)
{
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  if (l > UINT_MAX)
    luaL_error(L, "value is too large");
  return (unsigned int) l;
}

static inline lua_Number tk_lua_foptnumber (lua_State *L, int i, char *field, double d)
{
  lua_getfield(L, i, field);
  lua_Number n = luaL_optnumber(L, -1, d);
  lua_pop(L, 1);
  return n;
}

static inline double tk_lua_checkposdouble (lua_State *L, int i)
{
  lua_Number l = luaL_checknumber(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  return (double) l;
}

static inline double tk_lua_fcheckposdouble (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  double n = tk_lua_checkposdouble(L, -1);
  lua_pop(L, 1);
  return n;
}

static inline lua_Integer tk_lua_fcheckunsigned (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  lua_Integer n = tk_lua_checkunsigned(L, -1);
  lua_pop(L, 1);
  return n;
}

static inline unsigned int tk_lua_optunsigned (lua_State *L, int i, unsigned int def)
{
  if (lua_type(L, i) < 1)
    return def;
  return tk_lua_checkunsigned(L, i);
}

static inline lua_Integer tk_lua_ftype (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  int t = lua_type(L, -1);
  lua_pop(L, 1);
  return t;
}

static inline void *tk_lua_fcheckuserdata (lua_State *L, int i, char *field, char *mt)
{
  lua_getfield(L, i, field);
  void *p = luaL_checkudata(L, -1, mt);
  lua_pop(L, 1);
  return p;
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
    if (numa_available() == -1)
      free(p0);
    else
      numa_free(p0, s0);
    return p1;
  }
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

static inline int tk_lua_absindex (lua_State *L, int i)
{
  if (i < 0 && i > LUA_REGISTRYINDEX)
    i += lua_gettop(L) + 1;
  return i;
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

static tk_corex_t *peek_corex (lua_State *L, int i)
{
  return (tk_corex_t *) luaL_checkudata(L, i, MT_COREX);
}

static inline void tk_corex_shrink (tk_corex_t *C)
{
  free(C->mis); C->mis = NULL;
  free(C->sums); C->sums = NULL;
  free(C->counts); C->counts = NULL;
  free(C->tcs); C->tcs = NULL;
  if (numa_available() == -1) {
    free(C->maxmis); C->maxmis = NULL;
    free(C->px); C->px = NULL;
    free(C->entropy_x); C->entropy_x = NULL;
  } else {
    numa_free(C->maxmis, C->maxmis_len); C->maxmis = NULL; C->maxmis_len = 0;
    numa_free(C->entropy_x, C->entropy_len); C->entropy_x = NULL; C->entropy_len = 0;
    numa_free(C->px, C->px_len); C->px = NULL; C->px_len = 0;
  }
}

static inline void tk_corex_wait_for_threads (
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

static inline void tk_corex_signal (
  tk_corex_stage_t stage,
  unsigned int *sigid,
  tk_corex_stage_t *stagep,
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
  tk_corex_wait_for_threads(mutex, cond_done, n_threads_done, n_threads);
  pthread_cond_broadcast(cond_stage);
}

static int tk_corex_gc (lua_State *L)
{
  lua_settop(L, 1);
  tk_corex_t *C = peek_corex(L, 1);
  if (C->destroyed)
    return 1;
  C->destroyed = true;
  tk_corex_shrink(C);
  free(C->alpha); C->alpha = NULL;
  free(C->log_marg); C->log_marg = NULL;
  free(C->log_py); C->log_py = NULL;
  free(C->log_pyx_unnorm); C->log_pyx_unnorm = NULL;
  free(C->pyx); C->pyx = NULL;
  free(C->baseline); C->baseline = NULL;
  free(C->sort); C->sort = NULL;
  if (numa_available() == -1) {
    free(C->samples); C->samples = NULL;
    free(C->visibles); C->visibles = NULL;
  } else {
    numa_free(C->samples, C->samples_len); C->samples = NULL;
    numa_free(C->visibles, C->visibles_len); C->visibles = NULL;
  }
  tk_corex_signal(
    TK_CMP_DONE, &C->sigid,
    &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
    &C->n_threads_done, C->n_threads);
  // TODO: What is the right way to deal with potential thread errors (or other
  // errors, for that matter) during the finalizer?
  if (C->created_threads)
    for (unsigned int i = 0; i < C->n_threads; i ++)
      pthread_join(C->threads[i], NULL);
    // if (pthread_join(C->threads[i], NULL) != 0)
    //   tk_error(L, "pthread_join", errno);
  pthread_mutex_destroy(&C->mutex);
  pthread_cond_destroy(&C->cond_stage);
  pthread_cond_destroy(&C->cond_done);
  free(C->threads); C->threads = NULL;
  free(C->thread_data); C->thread_data = NULL;
  return 0;
}

static inline int tk_corex_destroy (lua_State *L)
{
  lua_settop(L, 0);
  lua_pushvalue(L, lua_upvalueindex(1));
  return tk_corex_gc(L);
}

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

static inline void seed_rand ()
{
  mcg_state = (uint64_t) pthread_self() ^ (uint64_t) time(NULL);
}

static inline int tk_corex_compress (lua_State *);

static inline void tk_corex_marginals_thread (
  uint64_t n_set_bits,
  uint64_t *restrict samples,
  unsigned int *restrict visibles,
  double *restrict log_py,
  double *restrict pyx,
  double *restrict counts,
  double *restrict log_marg,
  double *restrict mis,
  double *restrict px,
  double *restrict entropy_x,
  double *restrict tcs,
  double smoothing,
  double tmin,
  double ttc,
  unsigned int n_samples,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict lm00 = log_marg + 0 * n_hidden * n_visible;
  double *restrict lm01 = log_marg + 1 * n_hidden * n_visible;
  double *restrict lm10 = log_marg + 2 * n_hidden * n_visible;
  double *restrict lm11 = log_marg + 3 * n_hidden * n_visible;
  double *restrict pc00 = counts + 0 * n_hidden * n_visible;
  double *restrict pc01 = counts + 1 * n_hidden * n_visible;
  double *restrict pc10 = counts + 2 * n_hidden * n_visible;
  double *restrict pc11 = counts + 3 * n_hidden * n_visible;
  double *restrict tmp00 = lm00 + hfirst * n_visible;
  double *restrict tmp01 = lm01 + hfirst * n_visible;
  double *restrict tmp10 = lm10 + hfirst * n_visible;
  double *restrict tmp11 = lm11 + hfirst * n_visible;
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double *restrict pyx0 = pyx + h * n_samples;
    double counts_0 = smoothing;
    double counts_1 = smoothing;
    for (unsigned int s = 0; s < n_samples; s ++) {
      counts_0 += pyx0[s];
      counts_1 += (1 - pyx0[s]);
    }
    double sum_counts = counts_0 + counts_1;
    log_py[h] = log(counts_0) - log(sum_counts);
  }
  for (unsigned int i = hfirst * n_visible; i < (hlast + 1) * n_visible; i ++)
    counts[i] = 0;
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double sum_py0 = 0.0;
    double *restrict hpy0s = pyx + h * n_samples;
    for (unsigned int s = 0; s < n_samples; s ++)
      sum_py0 += hpy0s[s];
    double sum_py1 = n_samples - sum_py0;
    double *restrict pc00a = pc00 + h * n_visible;
    double *restrict pc01a = pc01 + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++) {
      pc00a[v] += sum_py0;
      pc01a[v] += sum_py1;
    }
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    for (unsigned int v = 0; v < n_visible; v ++) {
      tmp10[v] = 0;
      tmp11[v] = 0;
      tmp00[v] = 0;
      tmp01[v] = 0;
    }
    double *restrict py0h = pyx + h * n_samples;
    for (uint64_t c = 0; c < n_set_bits; c ++) { // not vectorzed, gather scatter
      uint64_t s = samples[c];
      unsigned int v = visibles[c];
      double py0 = py0h[s];
      double py1 = 1 - py0;
      tmp10[v] += py0;
      tmp11[v] += py1;
      tmp00[v] -= py0;
      tmp01[v] -= py1;
    }
    double *restrict pc00a = pc00 + h * n_visible;
    double *restrict pc01a = pc01 + h * n_visible;
    double *restrict pc10a = pc10 + h * n_visible;
    double *restrict pc11a = pc11 + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++) {
      pc10a[v] += tmp10[v];
      pc11a[v] += tmp11[v];
      pc00a[v] += tmp00[v];
      pc01a[v] += tmp01[v];
    }
  }
  for (unsigned int i = hfirst * n_visible; i < (hlast + 1) * n_visible; i ++) {
    pc00[i] += smoothing;
    pc01[i] += smoothing;
    pc10[i] += smoothing;
    pc11[i] += smoothing;
  }
  for (unsigned int i = hfirst * n_visible; i < (hlast + 1) * n_visible; i ++) {
    double log_total0 = log(pc00[i] + pc01[i]);
    lm00[i] = log(pc00[i]) - log_total0;
    lm01[i] = log(pc01[i]) - log_total0;
  }
  for (unsigned int i = hfirst * n_visible; i < (hlast + 1) * n_visible; i ++) {
    double log_total1 = log(pc10[i] + pc11[i]);
    lm10[i] = log(pc10[i]) - log_total1;
    lm11[i] = log(pc11[i]) - log_total1;
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double *restrict lm00a = lm00 + h * n_visible;
    double *restrict lm01a = lm01 + h * n_visible;
    double *restrict lm10a = lm10 + h * n_visible;
    double *restrict lm11a = lm11 + h * n_visible;
    double lpy0v = log_py[h];
    double lpy1v = log1p(-exp(lpy0v));
    for (unsigned int v = 0; v < n_visible; v ++) {
      lm00a[v] -= lpy0v;
      lm01a[v] -= lpy1v;
      lm10a[v] -= lpy0v;
      lm11a[v] -= lpy1v;
    }
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, non-affine base
    double *restrict lm00a = lm00 + h * n_visible;
    double *restrict lm01a = lm01 + h * n_visible;
    double *restrict lm10a = lm10 + h * n_visible;
    double *restrict lm11a = lm11 + h * n_visible;
    double *restrict pc00a = pc00 + h * n_visible;
    double *restrict pc01a = pc01 + h * n_visible;
    double *restrict pc10a = pc10 + h * n_visible;
    double *restrict pc11a = pc11 + h * n_visible;
    double lpy0v = log_py[h];
    double lpy1v = log1p(-exp(lpy0v));
    for (unsigned int v = 0; v < n_visible; v ++) {
      pc00a[v] = exp(lm00a[v] + lpy0v);
      pc01a[v] = exp(lm01a[v] + lpy1v);
      pc10a[v] = exp(lm10a[v] + lpy0v);
      pc11a[v] = exp(lm11a[v] + lpy1v);
    }
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double *restrict pc00a = pc00 + h * n_visible;
    double *restrict pc01a = pc01 + h * n_visible;
    double *restrict pc10a = pc10 + h * n_visible;
    double *restrict pc11a = pc11 + h * n_visible;
    double *restrict lm00a = lm00 + h * n_visible;
    double *restrict lm10a = lm10 + h * n_visible;
    double *restrict lm01a = lm01 + h * n_visible;
    double *restrict lm11a = lm11 + h * n_visible;
    double *restrict mish = mis + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++) {
      double group0 = pc00a[v] * lm00a[v] + pc01a[v] * lm01a[v];
      double group1 = pc10a[v] * lm10a[v] + pc11a[v] * lm11a[v];
      mish[v] = group0 * (1 - px[v]) + group1 * px[v];
    }
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, non-affine base
    double *restrict mish = mis + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++)
      mish[v] /= entropy_x[v];
  }
  for (unsigned int h = hfirst; h <= hlast; h ++)
    tcs[h] = fabs(tcs[h]) * ttc + tmin;
}

static inline void tk_corex_maxmis_thread (
  double *restrict mis,
  double *restrict maxmis,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int vfirst,
  unsigned int vlast
) {
  for (unsigned int v = vfirst; v <= vlast; v ++) {
    double max_val = 0.0;
    for (unsigned int h = 0; h < n_hidden; h ++) {
      double candidate = mis[h * n_visible + v];
      if (candidate > max_val)
        max_val = candidate;
    }
    maxmis[v] = max_val;
  }
}

static inline void tk_corex_alpha_thread (
  double *restrict alpha,
  double *restrict baseline,
  double *restrict log_marg,
  double *restrict tcs,
  double *restrict mis,
  double *restrict maxmis,
  double lam,
  double spa,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict baseline0 = baseline + 0 * n_hidden;
  double *restrict baseline1 = baseline + 1 * n_hidden;
  double *restrict lm00 = log_marg + 0 * n_hidden * n_visible;
  double *restrict lm01 = log_marg + 1 * n_hidden * n_visible;
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, non-affine
    double *restrict alphah = alpha + h * n_visible;
    double *restrict mish = mis + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++)
      alphah[v] = (1.0 - lam) * alphah[v] + lam * exp(tcs[h] * (mish[v] - maxmis[v]) / spa);
  }
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, unsupported outer form
    double s0 = 0.0, s1 = 0.0;
    double *restrict lm00a = lm00 + h * n_visible;
    double *restrict lm01a = lm01 + h * n_visible;
    double *restrict aph = alpha + h * n_visible;
    for (unsigned int v = 0; v < n_visible; v ++) {
      s0 += aph[v] * lm00a[v];
      s1 += aph[v] * lm01a[v];
    }
    baseline0[h] = s0;
    baseline1[h] = s1;
  }
}

static inline void tk_corex_latent_baseline_thread (
  double *restrict sums,
  double *restrict baseline,
  unsigned int n_samples,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict sums0 = sums + 0 * n_hidden * n_samples;
  double *restrict sums1 = sums + 1 * n_hidden * n_samples;
  double *restrict baseline0 = baseline + 0 * n_hidden;
  double *restrict baseline1 = baseline + 1 * n_hidden;
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, not affine
    double s0 = baseline0[h];
    double s1 = baseline1[h];
    double *restrict sums0a = sums0 + h * n_samples;
    double *restrict sums1a = sums1 + h * n_samples;
    for (unsigned int i = 0; i < n_samples; i ++) {
      sums0a[i] = s0;
      sums1a[i] = s1;
    }
  }
}

static inline void tk_corex_latent_sums_thread (
  uint64_t *restrict samples,
  unsigned int *restrict visibles,
  double *restrict alpha,
  double *restrict log_marg,
  double *restrict sums,
  unsigned int n_samples,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast,
  uint64_t n_set_bits
) {
  double *restrict sums0 = sums + 0 * n_hidden * n_samples;
  double *restrict sums1 = sums + 1 * n_hidden * n_samples;
  double *restrict lm00 = log_marg + 0 * n_hidden * n_visible;
  double *restrict lm01 = log_marg + 1 * n_hidden * n_visible;
  double *restrict lm10 = log_marg + 2 * n_hidden * n_visible;
  double *restrict lm11 = log_marg + 3 * n_hidden * n_visible;
  for (unsigned int h = hfirst; h <= hlast; h ++) {
    double *restrict sums0h = sums0 + h * n_samples;
    double *restrict sums1h = sums1 + h * n_samples;
    double *restrict lm00a = lm00 + h * n_visible;
    double *restrict lm01a = lm01 + h * n_visible;
    double *restrict lm10a = lm10 + h * n_visible;
    double *restrict lm11a = lm11 + h * n_visible;
    double *restrict aph = alpha + h * n_visible;
    for (unsigned int b = 0; b < n_set_bits; b ++) {
      uint64_t s = samples[b];
      unsigned int v = visibles[b];
      sums0h[s] = sums0h[s] - aph[v] * lm00a[v] + aph[v] * lm10a[v];
      sums1h[s] = sums1h[s] - aph[v] * lm01a[v] + aph[v] * lm11a[v];
    }
  }
}

static inline void tk_corex_latent_py_thread (
  double *restrict log_py,
  double *restrict log_pyx_unnorm,
  double *restrict sums,
  unsigned int n_samples,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict sums0 = sums + 0 * n_hidden * n_samples;
  double *restrict sums1 = sums + 1 * n_hidden * n_samples;
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, not affine
    double lpy0v = log_py[h];
    double lpy1v = log1p(-exp(lpy0v));
    double *restrict sums0h = sums0 + h * n_samples;
    double *restrict sums1h = sums1 + h * n_samples;
    double *restrict lpyx0 = log_pyx_unnorm + 0 * n_hidden * n_samples + h * n_samples;
    double *restrict lpyx1 = log_pyx_unnorm + 1 * n_hidden * n_samples + h * n_samples;
    for (unsigned int i = 0; i < n_samples; i ++) {
      lpyx0[i] = sums0h[i] + lpy0v;
      lpyx1[i] = sums1h[i] + lpy1v;
    }
  }
}

static inline void tk_corex_latent_norm_thread (
  double *restrict log_z, // mis
  double *restrict pyx,
  double *restrict log_pyx_unnorm,
  unsigned int n_samples,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double *restrict lpyx0 = log_pyx_unnorm + 0 * n_hidden * n_samples;
  double *restrict lpyx1 = log_pyx_unnorm + 1 * n_hidden * n_samples;
  for (unsigned int i = hfirst * n_samples; i < (hlast + 1) * n_samples; i ++) {
    double a = lpyx0[i];
    double b = lpyx1[i];
    double max_ab  = (a > b) ? a : b;
    double sum_exp = exp(a - max_ab) + exp(b - max_ab);
    log_z[i] = max_ab + log(sum_exp);
  }
  for (unsigned int i = hfirst * n_samples; i < (hlast + 1) * n_samples; i ++)
    pyx[i] = exp(lpyx0[i] - log_z[i]);
}

static inline void tk_corex_update_tc_thread (
  double *restrict log_z, // mis
  double *restrict tcs,
  unsigned int n_samples,
  unsigned int hfirst,
  unsigned int hlast
) {
  for (unsigned int h = hfirst; h <= hlast; h ++) { // not vectorized, unsupported outer form
    const double *restrict lz = log_z + h * n_samples;
    double sum = 0.0;
    for (unsigned int s = 0; s < n_samples; s ++)
      sum += lz[s];
    double tc = sum / (double) n_samples;
    tcs[h] = tc;
  }
}

static inline void tk_corex_update_last_tc (
  double *restrict tcs,
  double *last_tc,
  double *tc_dev,
  unsigned int n_hidden
) {
  double min = tcs[0], max = tcs[0], sum = 0.0, sumsq = 0.0;
  for (unsigned int h = 0; h < n_hidden; h ++) {
    double tc = tcs[h];
    sum += tc;
    sumsq += tc * tc;
    if (tc < min) min = tc;
    if (tc > max) max = tc;
  }
  double mean = sum / n_hidden;
  double stdev = sqrt(sumsq / n_hidden - mean * mean);
  *last_tc = sum;
  *tc_dev = stdev;
}

static inline int tk_corex_sort_lt (tk_corex_sort_t a, tk_corex_sort_t b)
{
  if (a.v < b.v) return 1;
  if (a.v > b.v) return 0;
  if (a.s < b.s) return 1;
  if (a.s > b.s) return 0;
  return 0;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include "ksort.h"
KSORT_INIT(pairs, tk_corex_sort_t, tk_corex_sort_lt);
#pragma GCC diagnostic pop

// TODO: Consider making this configurable
#define S_BLOCK 1024
#define V_BLOCK 2048

typedef struct {
  tk_corex_sort_t *pairs;
  size_t capacity;
  size_t size;
} tk_corex_tile_t;

static void tk_corex_tile_push (
  lua_State *L,
  tk_corex_tile_t *tile,
  tk_corex_sort_t pair
) {
  if (tile->size >= tile->capacity) {
    size_t newcap = (tile->capacity == 0) ? 1024 : (tile->capacity * 2);
    tk_corex_sort_t *newtile = tk_realloc(L, tile->pairs, newcap * sizeof(tk_corex_sort_t));
    tile->pairs = newtile;
    tile->capacity = newcap;
  }
  tile->pairs[tile->size ++] = pair;
}

static void tk_corex_tile_pairs (
  lua_State *L,
  tk_corex_sort_t *pairs,
  size_t n_pairs,
  unsigned int max_v
) {
  size_t num_s_tiles = ((n_pairs - 1) / S_BLOCK) + 1;
  size_t num_v_tiles = (max_v / V_BLOCK) + 1;
  size_t total_tiles = num_s_tiles * num_v_tiles;
  tk_corex_tile_t *tiles = tk_malloc(L, total_tiles * sizeof(tk_corex_tile_t));
  memset(tiles, 0, sizeof(tk_corex_tile_t) * total_tiles);
  for (size_t i = 0; i < n_pairs; i ++) {
    uint64_t s = pairs[i].s;
    unsigned int v = pairs[i].v;
    size_t tile_s = (size_t)(s / S_BLOCK);
    size_t tile_v = (size_t)(v / V_BLOCK);
    size_t tile_index = tile_s * num_v_tiles + tile_v;
    tk_corex_tile_push(L, &tiles[tile_index], pairs[i]);
  }
  size_t out_idx = 0;
  for (size_t t = 0; t < total_tiles; t ++) {
    tk_corex_tile_t *tile = &tiles[t];
    ks_introsort(pairs, tile->size, tile->pairs);
    for (size_t j = 0; j < tile->size; j ++) {
      pairs[out_idx ++] = tile->pairs[j];
    }
    free(tile->pairs);
  }
  free(tiles);
}

static inline uint64_t tk_corex_setup_bits (
  lua_State *L,
  int64_t *set_bits,
  uint64_t n_set_bits,
  tk_corex_sort_t *pairs,
  uint64_t *samples,
  unsigned int *visibles,
  unsigned int n_visible,
  bool tile
) {
  uint64_t n = 0;
  for (uint64_t i = 0; i < n_set_bits; i ++) {
    int64_t val = set_bits[i];
    if (val < 0)
      continue;
    pairs[n] = (tk_corex_sort_t) {
      .s = (uint64_t) val / n_visible,
      .v = (unsigned int) ((uint64_t) val % n_visible)
    };
    n ++;
  }
  if (!tile)
    ks_introsort(pairs, n, pairs);
  else
    tk_corex_tile_pairs(L, pairs, n, n_visible);
  for (size_t i = 0; i < n; i ++) {
    samples[i]  = pairs[i].s;
    visibles[i] = pairs[i].v;
  }
  return n;
}

static inline void tk_corex_data_stats (
  uint64_t n_set_bits,
  unsigned int *restrict visibles,
  double *restrict px,
  double *restrict entropy_x,
  unsigned int n_samples,
  unsigned int n_visible
) {
  for (unsigned int v = 0; v < n_visible; v ++)
    px[v] = 0;
  for (uint64_t c = 0; c < n_set_bits; c ++)
    px[visibles[c]] ++;
  for (unsigned int v = 0; v < n_visible; v ++)
    px[v] /= (double) n_samples;
  for (unsigned int v = 0; v < n_visible; v ++) {
    double entropy = 0;
    entropy -= px[v] * log(px[v]);
    entropy -= (1 - px[v]) * log(1 - px[v]);
    entropy_x[v] = entropy > 0 ? entropy : 1e-10;
  }
}

static inline void tk_corex_init_alpha_thread (
  double *alpha,
  unsigned int n_visible,
  unsigned int hfirst,
  unsigned int hlast
) {
  for (unsigned int i = hfirst * n_visible; i < (hlast + 1) * n_visible; i ++)
    alpha[i] = 0.5 + 0.5 * fast_drand();
}

// Doesn't really need to be threaded..'
// Consider combining with one of the other thread inits
static inline void tk_corex_init_tcs_thread (
  double *tcs,
  unsigned int hfirst,
  unsigned int hlast
) {
  for (unsigned int i = hfirst; i <= hlast; i ++)
    tcs[i] = 0.0;
}

static inline void tk_corex_init_log_pyx_unnorm_thread (
  double *log_pyx_unnorm,
  unsigned int n_samples,
  unsigned int n_hidden,
  unsigned int hfirst,
  unsigned int hlast
) {
  double log_dim_hidden = -log(2);
  double *restrict lpyx0 = log_pyx_unnorm + 0 * n_hidden * n_samples;
  double *restrict lpyx1 = log_pyx_unnorm + 1 * n_hidden * n_samples;
  for (unsigned int i = hfirst * n_samples; i < (hlast + 1) * n_samples; i ++) {
    lpyx0[i] = log_dim_hidden * (0.5 + fast_drand());
    lpyx1[i] = log_dim_hidden * (0.5 + fast_drand());
  }
}

static void *tk_corex_worker (void *datap)
{
  seed_rand();
  tk_corex_thread_data_t *data =
    (tk_corex_thread_data_t *) datap;
  pthread_mutex_lock(&data->C->mutex);
  data->C->n_threads_done ++;
  if (data->C->n_threads_done == data->C->n_threads)
    pthread_cond_signal(&data->C->cond_done);
  pthread_mutex_unlock(&data->C->mutex);
  while (1) {
    pthread_mutex_lock(&data->C->mutex);
    while (data->sigid == data->C->sigid)
      pthread_cond_wait(&data->C->cond_stage, &data->C->mutex);
    data->sigid = data->C->sigid;
    tk_corex_stage_t stage = data->C->stage;
    pthread_mutex_unlock(&data->C->mutex);
    switch (stage) {
      case TK_CMP_INIT_ALPHA:
        tk_corex_init_alpha_thread(
          data->C->alpha,
          data->C->n_visible,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_INIT_TCS:
        tk_corex_init_tcs_thread(
          data->C->tcs,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_INIT_PYX_UNNORM:
        tk_corex_init_log_pyx_unnorm_thread(
          data->C->log_pyx_unnorm,
          data->n_samples,
          data->C->n_hidden,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_MARGINALS:
        tk_corex_marginals_thread(
          data->n_set_bits,
          data->C->samples,
          data->C->visibles,
          data->C->log_py,
          data->C->pyx,
          data->C->counts,
          data->C->log_marg,
          data->C->mis,
          data->C->px,
          data->C->entropy_x,
          data->C->tcs,
          data->C->smoothing,
          data->C->tmin,
          data->C->ttc,
          data->n_samples,
          data->C->n_visible,
          data->C->n_hidden,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_MAXMIS:
        tk_corex_maxmis_thread(
          data->C->mis,
          data->C->maxmis,
          data->C->n_visible,
          data->C->n_hidden,
          data->vfirst,
          data->vlast);
        break;
      case TK_CMP_ALPHA:
        tk_corex_alpha_thread(
          data->C->alpha,
          data->C->baseline,
          data->C->log_marg,
          data->C->tcs,
          data->C->mis,
          data->C->maxmis,
          data->C->lam,
          data->C->spa,
          data->C->n_visible,
          data->C->n_hidden,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_LATENT_ALL:
        tk_corex_latent_baseline_thread(
          data->C->sums,
          data->C->baseline,
          data->n_samples,
          data->C->n_hidden,
          data->hfirst,
          data->hlast);
        tk_corex_latent_sums_thread(
          data->C->samples,
          data->C->visibles,
          data->C->alpha,
          data->C->log_marg,
          data->C->sums,
          data->n_samples,
          data->C->n_visible,
          data->C->n_hidden,
          data->hfirst,
          data->hlast,
          data->n_set_bits);
        tk_corex_latent_py_thread(
          data->C->log_py,
          data->C->log_pyx_unnorm,
          data->C->sums,
          data->n_samples,
          data->C->n_hidden,
          data->hfirst,
          data->hlast);
        tk_corex_latent_norm_thread(
          data->C->mis,
          data->C->pyx,
          data->C->log_pyx_unnorm,
          data->n_samples,
          data->C->n_hidden,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_LATENT_BASELINE:
        tk_corex_latent_baseline_thread(
          data->C->sums,
          data->C->baseline,
          data->n_samples,
          data->C->n_hidden,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_LATENT_SUMS:
        tk_corex_latent_sums_thread(
          data->C->samples,
          data->C->visibles,
          data->C->alpha,
          data->C->log_marg,
          data->C->sums,
          data->n_samples,
          data->C->n_visible,
          data->C->n_hidden,
          data->hfirst,
          data->hlast,
          data->n_set_bits);
        break;
      case TK_CMP_LATENT_PY:
        tk_corex_latent_py_thread(
          data->C->log_py,
          data->C->log_pyx_unnorm,
          data->C->sums,
          data->n_samples,
          data->C->n_hidden,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_LATENT_NORM:
        tk_corex_latent_norm_thread(
          data->C->mis,
          data->C->pyx,
          data->C->log_pyx_unnorm,
          data->n_samples,
          data->C->n_hidden,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_UPDATE_TC:
        tk_corex_update_tc_thread(
          data->C->mis,
          data->C->tcs,
          data->n_samples,
          data->hfirst,
          data->hlast);
        break;
      case TK_CMP_DONE:
        break;
      default:
        assert(false);
    }
    pthread_mutex_lock(&data->C->mutex);
    data->C->n_threads_done ++;
    if (data->C->n_threads_done == data->C->n_threads)
      pthread_cond_signal(&data->C->cond_done);
    pthread_mutex_unlock(&data->C->mutex);
    if (stage == TK_CMP_DONE)
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
    for (unsigned int i = 0; i < cpus->size; ++i) {
      if (numa_bitmask_isbitset(cpus, i)) {
        count ++;
      }
    }
    if (count > 0) {
      unsigned int local_index =
        (thread_index - node * threads_per_node) % count;
      unsigned int found = 0;
      for (unsigned int i = 0; i < cpus->size; ++i) {
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

static void *tk_corex_worker_wrapper (void *arg) {
  tk_corex_thread_data_t *td = (tk_corex_thread_data_t *)arg;
  if (numa_available() != -1 && numa_max_node() > 0)
    tk_pin_thread_to_cpu(td->index, td->C->n_threads);
  return tk_corex_worker(arg);
}

static inline void tk_corex_setup_threads (
  lua_State *L,
  tk_corex_t *C,
  uint64_t n_set_bits,
  unsigned int n_samples
) {
  if (!C->created_threads) {
    C->sigid = 0;
    unsigned int hslice = C->n_hidden / C->n_threads;
    unsigned int hremaining = C->n_hidden % C->n_threads;
    unsigned int hfirst = 0;
    unsigned int vslice = C->n_visible / C->n_threads;
    unsigned int vremaining = C->n_visible % C->n_threads;
    unsigned int vfirst = 0;
    for (unsigned int i = 0; i < C->n_threads; i ++) {
      C->thread_data[i].C = C;
      C->thread_data[i].sigid = 0;
      C->thread_data[i].index = i;
      C->thread_data[i].hfirst = hfirst;
      C->thread_data[i].hlast = hfirst + hslice - 1;
      if (hremaining) {
        C->thread_data[i].hlast ++;
        hremaining --;
      }
      hfirst = C->thread_data[i].hlast + 1;
      C->thread_data[i].vfirst = vfirst;
      C->thread_data[i].vlast = vfirst + vslice - 1;
      if (vremaining) {
        C->thread_data[i].vlast ++;
        vremaining --;
      }
      vfirst = C->thread_data[i].vlast + 1;
    }
  }
  for (unsigned int i = 0; i < C->n_threads; i ++) {
    C->thread_data[i].n_set_bits = n_set_bits;
    C->thread_data[i].n_samples = n_samples;
  }
  if (!C->created_threads) {
    for (unsigned int i = 0; i < C->n_threads; i ++)
      if (pthread_create(&C->threads[i], NULL, tk_corex_worker_wrapper, &C->thread_data[i]) != 0)
        tk_error(L, "pthread_create", errno);
    tk_corex_wait_for_threads(
      &C->mutex,
      &C->cond_done,
      &C->n_threads_done,
      C->n_threads);
    C->created_threads = true;
  }
}

static inline int tk_corex_compress (lua_State *L)
{
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  lua_pushvalue(L, 1);
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  int64_t *set_bits = (int64_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_set_bits = (uint64_t) luaL_checkinteger(L, -1);
  unsigned int n_samples = tk_lua_optunsigned(L, 2, 1);

  // TODO: Expose shrink via the api, and only realloc if new size is larger than old
  C->sort = tk_realloc(L, C->sort, n_set_bits * sizeof(tk_corex_sort_t));
  C->samples = tk_ensure_interleaved(L, &C->samples_len, C->samples, n_set_bits * sizeof(uint64_t), false);
  C->visibles = tk_ensure_interleaved(L, &C->visibles_len, C->visibles, n_set_bits * sizeof(unsigned int), false);
  n_set_bits = tk_corex_setup_bits(L, set_bits, n_set_bits, C->sort, C->samples, C->visibles, C->n_visible, false);
  tk_corex_setup_threads(L, C, n_set_bits, n_samples);
  C->mis = tk_realloc(L, C->mis, C->n_hidden * n_samples * sizeof(double));
  C->pyx = tk_realloc(L, C->pyx, C->n_hidden * n_samples * sizeof(double));
  C->log_pyx_unnorm = tk_realloc(L, C->log_pyx_unnorm, 2 * C->n_hidden * n_samples * sizeof(double));
  C->sums = tk_realloc(L, C->sums, 2 * C->n_hidden * n_samples * sizeof(double));
  tk_corex_signal(
    TK_CMP_LATENT_ALL, &C->sigid,
    &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
    &C->n_threads_done, C->n_threads);
  // TODO: Parallelize output
  size_t write = 0;
  for (unsigned int h = 0; h < C->n_hidden; h ++) {
    for (unsigned int s = 0; s < n_samples; s ++) {
      double py0 = C->pyx[h * n_samples + s];
      // TODO: should this be reversed?
      if (py0 < 0.5)
        set_bits[write ++] = s * C->n_hidden + h;
    }
  }
  lua_pushlightuserdata(L, set_bits);
  lua_pushinteger(L, 1);
  lua_pushinteger(L, (int64_t) write);
  lua_pushvalue(L, 1);
  tk_lua_callmod(L, 4, 1, "santoku.matrix.integer", "from_view");
  return 0;
}

static inline void _tk_corex_train (
  lua_State *L,
  tk_corex_t *C,
  int64_t *set_bits,
  uint64_t n_set_bits,
  unsigned int n_samples,
  unsigned int max_iter,
  int i_each
) {
  C->smoothing = fmax(1e-10, 1.0 / (double) n_samples);
  C->pyx = tk_malloc(L, C->n_hidden * n_samples * sizeof(double));
  C->log_pyx_unnorm = tk_malloc(L, 2 * C->n_hidden * n_samples * sizeof(double));
  C->sums = tk_malloc(L, 2 * C->n_hidden * n_samples * sizeof(double));
  unsigned int len_mis = C->n_hidden * C->n_visible;
  if (len_mis < (C->n_hidden * n_samples))
    len_mis = C->n_hidden * n_samples;
  C->mis = tk_malloc(L, len_mis * sizeof(double));
  C->sort = tk_malloc(L, n_set_bits * sizeof(tk_corex_sort_t));
  C->samples = tk_malloc_interleaved(L, &C->samples_len, n_set_bits * sizeof(uint64_t));
  C->visibles = tk_malloc_interleaved(L, &C->visibles_len, n_set_bits * sizeof(unsigned int));
  n_set_bits = tk_corex_setup_bits(L, set_bits, n_set_bits, C->sort, C->samples, C->visibles, C->n_visible, true);
  tk_corex_data_stats(
    n_set_bits,
    C->visibles,
    C->px,
    C->entropy_x,
    n_samples,
    C->n_visible);
  tk_corex_setup_threads(L, C, n_set_bits, n_samples);
  tk_corex_signal(
    TK_CMP_INIT_PYX_UNNORM, &C->sigid,
    &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
    &C->n_threads_done, C->n_threads);
  tk_corex_signal(
    TK_CMP_LATENT_NORM, &C->sigid,
    &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
    &C->n_threads_done, C->n_threads);
  for (unsigned int i = 0; i < max_iter; i ++) {
    tk_corex_signal(
      TK_CMP_MARGINALS, &C->sigid,
      &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
      &C->n_threads_done, C->n_threads);
    tk_corex_signal(
      TK_CMP_MAXMIS, &C->sigid,
      &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
      &C->n_threads_done, C->n_threads);
    tk_corex_signal(
      TK_CMP_ALPHA, &C->sigid,
      &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
      &C->n_threads_done, C->n_threads);
    tk_corex_signal(
      TK_CMP_LATENT_ALL, &C->sigid,
      &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
      &C->n_threads_done, C->n_threads);
    tk_corex_signal(
      TK_CMP_UPDATE_TC, &C->sigid,
      &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
      &C->n_threads_done, C->n_threads);
    tk_corex_update_last_tc(
      C->tcs,
      &C->last_tc,
      &C->tc_dev,
      C->n_hidden);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, i + 1);
      lua_pushnumber(L, C->last_tc);
      lua_pushnumber(L, C->tc_dev);
      lua_call(L, 3, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0)
        break;
      lua_pop(L, 1);
    }
  }
  tk_corex_shrink(C);
  C->trained = true;
}

static inline void tk_corex_init (
  lua_State *L,
  tk_corex_t *C,
  double lam,
  double spa,
  double tmin,
  double ttc,
  unsigned int n_visible,
  unsigned int n_hidden,
  unsigned int n_threads
) {
  memset(C, 0, sizeof(tk_corex_t));
  C->n_visible = n_visible;
  C->n_hidden = n_hidden;
  C->lam = lam;
  C->spa = spa;
  C->tmin = tmin;
  C->ttc = ttc;
  C->tcs = tk_malloc(L, C->n_hidden * sizeof(double));
  C->alpha = tk_malloc(L, C->n_hidden * C->n_visible * sizeof(double));
  C->log_py = tk_malloc(L, C->n_hidden * sizeof(double));
  C->log_marg = tk_malloc(L, 2 * 2 * C->n_hidden * C->n_visible * sizeof(double));
  C->counts = tk_malloc(L, 2 * 2 * C->n_hidden * C->n_visible * sizeof(double));
  C->baseline = tk_malloc(L, 2 * C->n_hidden * sizeof(double));
  C->px = tk_malloc_interleaved(L, &C->px_len, C->n_visible * sizeof(double));
  C->entropy_x = tk_malloc_interleaved(L, &C->entropy_len, C->n_visible * sizeof(double));
  C->maxmis = tk_malloc_interleaved(L, &C->maxmis_len, C->n_visible * sizeof(double));
  C->n_threads = n_threads;
  C->n_threads_done = 0;
  C->stage = TK_CMP_INIT;
  C->threads = tk_malloc(L, C->n_threads * sizeof(pthread_t));
  C->thread_data = tk_malloc(L, C->n_threads * sizeof(tk_corex_thread_data_t));
  // TODO: check errors
  pthread_mutex_init(&C->mutex, NULL);
  pthread_cond_init(&C->cond_stage, NULL);
  pthread_cond_init(&C->cond_done, NULL);
  tk_corex_setup_threads(L, C, 0, 0);
  tk_corex_signal(
    TK_CMP_INIT_TCS, &C->sigid,
    &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
    &C->n_threads_done, C->n_threads);
  tk_corex_signal(
    TK_CMP_INIT_ALPHA, &C->sigid,
    &C->stage, &C->mutex, &C->cond_stage, &C->cond_done,
    &C->n_threads_done, C->n_threads);
}

static inline int tk_corex_visible (lua_State *L) {
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  lua_pushinteger(L, C->n_visible);
  return 1;
}

static inline int tk_corex_hidden (lua_State *L) {
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  lua_pushinteger(L, C->n_hidden);
  return 1;
}

static inline int tk_corex_train (lua_State *L) {
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  if (C->trained)
    return tk_lua_error(L, "Already trained!\n");
  lua_getfield(L, 1, "corpus");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  int64_t *set_bits = (int64_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_set_bits = (uint64_t) luaL_checkinteger(L, -1);
  unsigned int n_samples = tk_lua_fcheckunsigned(L, 1, "samples");
  unsigned int max_iter = tk_lua_fcheckunsigned(L, 1, "iterations");
  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }
  _tk_corex_train(L, C, set_bits, n_set_bits, n_samples, max_iter, i_each); // c
  return 0;
}

static inline int tk_corex_persist (lua_State *L)
{
  tk_corex_t *C = peek_corex(L, lua_upvalueindex(1));
  if (!C->trained)
    return tk_lua_error(L, "Can't persist an untrained model\n");
  lua_settop(L, 1);
  bool tostr = lua_type(L, 1) == LUA_TNIL;
  FILE *fh = tostr ? tk_lua_tmpfile(L) : tk_lua_fopen(L, luaL_checkstring(L, 1), "w");
  tk_lua_fwrite(L, &C->trained, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, &C->n_visible, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &C->n_hidden, sizeof(unsigned int), 1, fh);
  tk_lua_fwrite(L, &C->lam, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &C->spa, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &C->tmin, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &C->ttc, sizeof(double), 1, fh);
  tk_lua_fwrite(L, C->alpha, sizeof(double), C->n_hidden * C->n_visible, fh);
  tk_lua_fwrite(L, C->log_py, sizeof(double), C->n_hidden, fh);
  tk_lua_fwrite(L, C->log_marg, sizeof(double), 2 * 2 * C->n_hidden * C->n_visible, fh);
  tk_lua_fwrite(L, C->baseline, sizeof(double), 2 * C->n_hidden, fh);
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

static luaL_Reg mt_fns[] =
{
  { "visible", tk_corex_visible },
  { "hidden", tk_corex_hidden },
  { "compress", tk_corex_compress },
  { "persist", tk_corex_persist },
  { "train", tk_corex_train },
  { "destroy", tk_corex_destroy },
  { NULL, NULL }
};

static inline int tk_corex_load (lua_State *L)
{
  lua_settop(L, 3); // fp ts
  tk_corex_t *C = (tk_corex_t *)
    lua_newuserdata(L, sizeof(tk_corex_t)); // tp ts c
  memset(C, 0, sizeof(tk_corex_t));
  luaL_getmetatable(L, MT_COREX); // fp ts c mt
  lua_setmetatable(L, -2); // fp ts c
  lua_newtable(L); // fp ts c t
  lua_pushvalue(L, -2); // fp ts c t c
  tk_lua_register(L, mt_fns, 1); // fp ts c t
  lua_remove(L, -2); // fp ts t
  unsigned int n_threads;
  if (lua_type(L, 2) != LUA_TNIL) {
    // TODO: allow passing 0 to run everything on the main thread.
    n_threads = tk_lua_checkunsigned(L, 2);
    if (!n_threads)
      return tk_lua_error(L, "threads must be at least 1\n");
  } else {
    long ts = sysconf(_SC_NPROCESSORS_ONLN) - 1;
    if (ts <= 0)
      return tk_error(L, "sysconf", errno);
    lua_pushinteger(L, ts);
    n_threads = tk_lua_checkunsigned(L, -1);
    lua_pop(L, 1);
  }
  size_t len;
  const char *data = luaL_checklstring(L, 1, &len);
  bool isstr = lua_type(L, 3) == LUA_TBOOLEAN && lua_toboolean(L, 3);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  tk_lua_fread(L, &C->trained, sizeof(bool), 1, fh);
  tk_lua_fread(L, &C->n_visible, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &C->n_hidden, sizeof(unsigned int), 1, fh);
  tk_lua_fread(L, &C->lam, sizeof(double), 1, fh);
  tk_lua_fread(L, &C->spa, sizeof(double), 1, fh);
  tk_lua_fread(L, &C->tmin, sizeof(double), 1, fh);
  tk_lua_fread(L, &C->ttc, sizeof(double), 1, fh);
  C->alpha = tk_malloc(L, C->n_hidden * C->n_visible * sizeof(double));
  tk_lua_fread(L, C->alpha, sizeof(double), C->n_hidden * C->n_visible, fh);
  C->log_py = tk_malloc(L, C->n_hidden * sizeof(double));
  tk_lua_fread(L, C->log_py, sizeof(double), C->n_hidden, fh);
  C->log_marg = tk_malloc(L, 2 * 2 * C->n_hidden * C->n_visible * sizeof(double));
  tk_lua_fread(L, C->log_marg, sizeof(double), 2 * 2 * C->n_hidden * C->n_visible, fh);
  C->baseline = tk_malloc(L, 2 * C->n_hidden * sizeof(double));
  tk_lua_fread(L, C->baseline, sizeof(double), 2 * C->n_hidden, fh);
  C->n_threads = n_threads;
  C->n_threads_done = 0;
  C->stage = TK_CMP_INIT;
  C->threads = tk_malloc(L, C->n_threads * sizeof(pthread_t));
  C->thread_data = tk_malloc(L, C->n_threads * sizeof(tk_corex_thread_data_t));
  tk_lua_fclose(L, fh);
  pthread_mutex_init(&C->mutex, NULL);
  pthread_cond_init(&C->cond_stage, NULL);
  pthread_cond_init(&C->cond_done, NULL);
  tk_corex_setup_threads(L, C, 0, 0);
  return 1;
}

static inline int tk_corex_create (lua_State *L)
{
  lua_settop(L, 1);
  unsigned int n_visible = tk_lua_fcheckunsigned(L, 1, "visible");
  unsigned int n_hidden = tk_lua_fcheckunsigned(L, 1, "hidden");
  double lam = tk_lua_foptnumber(L, 1, "lam", 0.3);
  double spa = tk_lua_foptnumber(L, 1, "spa", 10.0);
  double tmin = tk_lua_foptnumber(L, 1, "tmin", 1.0);
  double ttc = tk_lua_foptnumber(L, 1, "ttc", 500.0);
  unsigned int n_threads;
  if (tk_lua_ftype(L, 1, "threads") != LUA_TNIL) {
    // TODO: allow passing 0 to run everything on the main thread.
    n_threads = tk_lua_fcheckunsigned(L, 1, "threads");
    if (!n_threads)
      return tk_lua_error(L, "threads must be at least 1\n");
  } else {
    long ts = sysconf(_SC_NPROCESSORS_ONLN) - 1;
    if (ts <= 0)
      return tk_error(L, "sysconf", errno);
    lua_pushinteger(L, ts);
    n_threads = tk_lua_checkunsigned(L, -1);
    lua_pop(L, 1);
  }
  tk_corex_t *C = (tk_corex_t *)
    lua_newuserdata(L, sizeof(tk_corex_t)); // c
  luaL_getmetatable(L, MT_COREX); // c mt
  lua_setmetatable(L, -2); // c
  tk_corex_init(L, C, lam, spa, tmin, ttc, n_visible, n_hidden, n_threads); // c
  lua_newtable(L); // c t
  lua_pushvalue(L, -2); // c t c
  tk_lua_register(L, mt_fns, 1); // t
  return 1;
}

static luaL_Reg fns[] =
{
  { "create", tk_corex_create },
  { "load", tk_corex_load },
  { NULL, NULL }
};

int luaopen_santoku_corex (lua_State *L)
{
  lua_newtable(L); // t
  luaL_register(L, NULL, fns); // t
  luaL_newmetatable(L, MT_COREX); // t mt
  lua_pushcfunction(L, tk_corex_gc); // t mt fn
  lua_setfield(L, -2, "__gc"); // t mt
  lua_pop(L, 1); // t
  return 1;
}
