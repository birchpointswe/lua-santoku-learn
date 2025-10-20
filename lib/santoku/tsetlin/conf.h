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

typedef struct { int64_t sim; bool label; } tm_dl_t;
#define tm_dl_lt(a, b) ((a).sim < (b).sim)
KSORT_INIT(dl, tm_dl_t, tm_dl_lt)
typedef kvec_t(tm_dl_t) tm_dls_t;

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
    if (copy && p0 && s0 > 0)
      memcpy(p1, p0, s0);
    if (p0) {
      if (numa_available() == -1)
        free(p0);
      else
        numa_free(p0, s0);
    }
    return p1;
  }
}

#endif
