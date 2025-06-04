#ifndef TK_INV_H
#define TK_INV_H

#include <assert.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/threads.h>

typedef enum {
  TK_INV_KNN
} tk_inv_stage_t;

typedef struct tk_inv_thread_s tk_inv_thread_t;

typedef struct tk_inv_s {
  bool destroyed;
  uint64_t size;
  uint64_t features;
  tk_inv_thread_t *threads;
  tk_threadpool_t *pool;
} tk_inv_t;

typedef struct tk_inv_thread_s {
  tk_inv_t *A;
} tk_inv_thread_t;

#define TK_INV_MT "tk_inv_t"
#define TK_INV_EPH "tk_inv_eph"

static inline void tk_inv_worker (void *dp, int sig)
{
  tk_inv_stage_t stage = (tk_inv_stage_t) sig;
  tk_inv_thread_t *data = (tk_inv_thread_t *) dp;
  switch (stage) {
    case TK_INV_KNN:
      // TODO
      break;
  }
}

static inline tk_inv_t *tk_inv_peek (lua_State *L, int i)
{
  return (tk_inv_t *) luaL_checkudata(L, i, TK_INV_MT);
}

static inline void tk_inv_destroy (
  tk_inv_t *A
) {
  if (A->destroyed)
    return;
  tk_threads_destroy(A->pool);
  free(A->threads);
  #warning todo
  A->destroyed = true;
  A->size = 0;
}

static inline uint64_t tk_inv_size (
  tk_inv_t *A
) {
  return A->size;
}

static inline void tk_inv_persist (
  lua_State *L,
  tk_inv_t *A,
  FILE *fh
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't persist a destroyed index");
    return;
  }
  #warning todo
}

static inline int tk_inv_gc_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  tk_inv_destroy(A);
  return 0;
}

static inline int tk_inv_add_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_remove_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_get_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_neighborhoods_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_neighbors_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_centroids_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_import_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_size_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) tk_inv_size(A));
  return 1;
}

static inline int tk_inv_threads_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) A->pool->n_threads);
  #warning todo
  return 0;
}

static inline int tk_inv_features_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) A->features);
  return 0;
}

static inline int tk_inv_persist_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_inv_persist(L, A, fh);
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

static inline int tk_inv_destroy_lua (lua_State *L)
{
  tk_inv_t *A = tk_inv_peek(L, 1);
  tk_inv_destroy(A);
  return 0;
}

static luaL_Reg tk_inv_lua_mt_fns[] =
{
  { "add", tk_inv_add_lua },
  { "remove", tk_inv_remove_lua },
  { "import", tk_inv_import_lua },
  { "get", tk_inv_get_lua },
  { "neighborhoods", tk_inv_neighborhoods_lua },
  { "neighbors", tk_inv_neighbors_lua },
  { "centroids", tk_inv_centroids_lua },
  { "import", tk_inv_import_lua },
  { "size", tk_inv_size_lua },
  { "threads", tk_inv_threads_lua },
  { "features", tk_inv_features_lua },
  { "persist", tk_inv_persist_lua },
  { "destroy", tk_inv_destroy_lua },
  { NULL, NULL }
};

static inline void tk_inv_suppress_unused_lua_mt_fns (void)
  { (void) tk_inv_lua_mt_fns; }

static inline tk_inv_t *tk_inv_create (
  lua_State *L,
  uint64_t features,
  uint64_t n_threads
) {
  tk_inv_t *A = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  A->threads = tk_malloc(L, n_threads * sizeof(tk_inv_thread_t));
  memset(A->threads, 0, n_threads * sizeof(tk_inv_thread_t));
  A->pool = tk_threads_create(L, n_threads, tk_inv_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_inv_thread_t *data = A->threads + i;
    A->pool->threads[i].data = data;
    data->A = A;
  }
  A->destroyed = false;
  A->size = 0;
  #warning todo
  return A;
}

static inline tk_inv_t *tk_inv_load (
  lua_State *L,
  FILE *fh
) {
  tk_inv_t *A = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  #warning todo
  return A;
}

#endif
