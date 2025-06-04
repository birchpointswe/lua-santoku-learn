#ifndef TK_ANN_H
#define TK_ANN_H

#include <assert.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/threads.h>

typedef enum {
  TK_ANN_KNN
} tk_ann_stage_t;

typedef struct tk_ann_thread_s tk_ann_thread_t;

typedef struct tk_ann_s {
  bool destroyed;
  uint64_t size;
  uint64_t bucket_target;
  uint64_t features;
  uint64_t probe_radius;
  tk_ivec_t *hash_bits;
  tk_ann_thread_t *threads;
  tk_threadpool_t *pool;
} tk_ann_t;

typedef struct tk_ann_thread_s {
  tk_ann_t *A;
} tk_ann_thread_t;

#define TK_ANN_MT "tk_ann_t"
#define TK_ANN_EPH "tk_ann_eph"

static inline void tk_ann_worker (void *dp, int sig)
{
  tk_ann_stage_t stage = (tk_ann_stage_t) sig;
  tk_ann_thread_t *data = (tk_ann_thread_t *) dp;
  switch (stage) {
    case TK_ANN_KNN:
      // TODO
      break;
  }
}

static inline tk_ann_t *tk_ann_peek (lua_State *L, int i)
{
  return (tk_ann_t *) luaL_checkudata(L, i, TK_ANN_MT);
}

static inline void tk_ann_destroy (
  tk_ann_t *A
) {
  if (A->destroyed)
    return;
  tk_threads_destroy(A->pool);
  free(A->threads);
  #warning todo
  A->destroyed = true;
  A->size = 0;
}

static inline uint64_t tk_ann_size (
  tk_ann_t *A
) {
  return A->size;
}

static inline void tk_ann_persist (
  lua_State *L,
  tk_ann_t *A,
  FILE *fh
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't persist a destroyed index");
    return;
  }
  #warning todo
}

static inline int tk_ann_gc_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_ann_destroy(A);
  return 0;
}

static inline int tk_ann_add_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_ann_remove_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_ann_get_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_ann_neighborhoods_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_ann_neighbors_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_ann_centroids_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_ann_import_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_ann_size_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  lua_pushinteger(L, (int64_t) tk_ann_size(A));
  return 1;
}

static inline int tk_ann_threads_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  lua_pushinteger(L, (int64_t) A->pool->n_threads);
  #warning todo
  return 0;
}

static inline int tk_ann_features_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  lua_pushinteger(L, (int64_t) A->features);
  return 0;
}

static inline int tk_ann_persist_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_ann_persist(L, A, fh);
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

static inline int tk_ann_destroy_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_ann_destroy(A);
  return 0;
}

static luaL_Reg tk_ann_lua_mt_fns[] =
{
  { "add", tk_ann_add_lua },
  { "remove", tk_ann_remove_lua },
  { "import", tk_ann_import_lua },
  { "get", tk_ann_get_lua },
  { "neighborhoods", tk_ann_neighborhoods_lua },
  { "neighbors", tk_ann_neighbors_lua },
  { "centroids", tk_ann_centroids_lua },
  { "import", tk_ann_import_lua },
  { "size", tk_ann_size_lua },
  { "threads", tk_ann_threads_lua },
  { "features", tk_ann_features_lua },
  { "persist", tk_ann_persist_lua },
  { "destroy", tk_ann_destroy_lua },
  { NULL, NULL }
};

static inline void tk_ann_suppress_unused_lua_mt_fns (void)
  { (void) tk_ann_lua_mt_fns; }

static inline void tk_ann_setup_hash_bits (
  lua_State *L,
  tk_ann_t *A,
  int Ai,
  tk_ivec_t *guidance,
  uint64_t n_threads
) {
  tk_ivec_asc(guidance, 0, guidance->n);
  uint64_t n_samples = guidance->a[guidance->n] / (int64_t) A->features + 1;
  uint64_t n_hash_bits = log(n_samples / A->bucket_target);
  // A->hash_bits = tk_ivec_top_entropy(guidance, n_samples, A->features, n_hash_bits, n_threads);
  // tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  // lua_pop(L, 1);
}

static inline tk_ann_t *tk_ann_create (
  lua_State *L,
  uint64_t features,
  uint64_t bucket_target,
  uint64_t probe_radius,
  tk_ivec_t *guidance,
  uint64_t n_threads
) {
  tk_ann_t *A = tk_lua_newuserdata(L, tk_ann_t, TK_ANN_MT, tk_ann_lua_mt_fns, tk_ann_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  A->threads = tk_malloc(L, n_threads * sizeof(tk_ann_thread_t));
  memset(A->threads, 0, n_threads * sizeof(tk_ann_thread_t));
  A->pool = tk_threads_create(L, n_threads, tk_ann_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_ann_thread_t *data = A->threads + i;
    A->pool->threads[i].data = data;
    data->A = A;
  }
  A->destroyed = false;
  A->size = 0;
  A->probe_radius = probe_radius;
  A->bucket_target = bucket_target;
  tk_ann_setup_hash_bits(L, A, Ai, guidance, n_threads);
  #warning todo
  return A;
}

static inline tk_ann_t *tk_ann_load (
  lua_State *L,
  FILE *fh
) {
  tk_ann_t *A = tk_lua_newuserdata(L, tk_ann_t, TK_ANN_MT, tk_ann_lua_mt_fns, tk_ann_gc_lua);
  #warning todo
  return A;
}

#endif
