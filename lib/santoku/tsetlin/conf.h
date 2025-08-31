#ifndef TK_CONF_H
#define TK_CONF_H

#include <santoku/lua/utils.h>
#include <santoku/threads.h>
#include <santoku/klib.h>

#include <assert.h>
#include <errno.h>
#include <lauxlib.h>
#include <limits.h>
#include <lua.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TK_TSETLIN_MT "santoku_tsetlin"

#define _STR(x) #x
#define STR(x) _STR(x)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

typedef uint8_t tk_bits_t;
#define BITS 8
#define BYTES (BITS / CHAR_BIT)

static inline uint64_t mix64 (uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static inline uint64_t hash128 (
  uint64_t lo,
  uint64_t hi
) {
  uint64_t x = lo ^ (hi + 0x9e3779b97f4a7c15ULL + (lo << 6) + (lo >> 2));
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

typedef struct { int64_t u, v; double w; } tm_pair_t;
#define tm_pair(u, v, w) (((u) < (v)) ? ((tm_pair_t) { (u), (v), (w) }) : ((tm_pair_t) { (v), (u), (w) }))
#define tm_pair_lt(a, b) ((a).u < (b).u || ((a).u == (b).u && (a).v < (b).v))
#define tm_pair_eq(a, b) ((a).u == (b).u && ((a).v == (b).v))
#define tm_pair_hash(a) ((khint_t) hash128((uint64_t) kh_int64_hash_func((a).u), (uint64_t) kh_int64_hash_func((a).v)))
KHASH_INIT(pairs, tm_pair_t, char, 0, tm_pair_hash, tm_pair_eq)
typedef khash_t(pairs) tm_pairs_t;
KSORT_INIT(pair_asc, tm_pair_t, tm_pair_lt)
typedef struct { int64_t sim; bool label; } tm_dl_t;
#define tm_dl_lt(a, b) ((a).sim < (b).sim)
KSORT_INIT(dl, tm_dl_t, tm_dl_lt)
typedef kvec_t(tm_dl_t) tm_dls_t;
typedef struct { int64_t u, v; double d; } tm_candidate_t;
#define tm_candidate_lt(a, b) ((a).d < (b).d)
#define tm_candidate_gt(a, b) ((a).d > (b).d)
#define tm_candidate(u, v, d) ((tm_candidate_t) { (u), (v), (d) })
KSORT_INIT(candidates_asc, tm_candidate_t, tm_candidate_lt)
KSORT_INIT(candidates_desc, tm_candidate_t, tm_candidate_gt)
typedef kvec_t(tm_candidate_t) tm_candidates_t;

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

#endif
