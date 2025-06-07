#ifndef TK_INV_H
#define TK_INV_H

#include <assert.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/threads.h>
#include <santoku/klib.h>

KHASH_INIT(tk_inv_nodes, int64_t, tk_iuset_t *, 1, kh_int64_hash_func, kh_int64_hash_equal)
typedef khash_t(tk_inv_nodes) tk_inv_nodes_t;
KHASH_INIT(tk_inv_postings, int64_t, tk_iuset_t *, 1, kh_int64_hash_func, kh_int64_hash_equal)
typedef khash_t(tk_inv_postings) tk_inv_postings_t;

#define TK_INV_MT "tk_inv_t"
#define TK_INV_EPH "tk_inv_eph"

typedef tk_rvec_t * tk_inv_hood_t;
#define tk_vec_name tk_inv_hoods
#define tk_vec_base tk_inv_hood_t
#define tk_vec_pushbase(L, x) tk_lua_get_ephemeron(L, TK_INV_EPH, x)
#define tk_vec_peekbase(L, i) tk_rvec_peek(L, i, "hood")
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef enum {
  TK_INV_NEIGHBORHOODS
} tk_inv_stage_t;

typedef struct tk_inv_thread_s tk_inv_thread_t;

typedef struct tk_inv_s {
  bool destroyed;
  uint64_t features;
  tk_inv_nodes_t *nodes;
  tk_inv_postings_t *postings;
  tk_inv_thread_t *threads;
  tk_threadpool_t *pool;
} tk_inv_t;

typedef struct tk_inv_thread_s {
  tk_inv_t *I;
  tk_inv_hoods_t *hoods;
  tk_iuset_t *candidates;
  tk_iuset_t *seen;
  uint64_t knn;
  uint64_t ufirst, ulast;
} tk_inv_thread_t;

#define TK_INV_MT "tk_inv_t"
#define TK_INV_EPH "tk_inv_eph"

static inline void tk_inv_worker (void *dp, int sig)
{
  tk_inv_stage_t stage = (tk_inv_stage_t) sig;
  tk_inv_thread_t *data = (tk_inv_thread_t *) dp;
  khint_t khi;
  int64_t v, f;
  switch (stage) {
    case TK_INV_NEIGHBORHOODS:
      for (int64_t u = (int64_t) data->ufirst; u <= (int64_t) data->ulast; u ++) {
        tk_rvec_t *nbrs = data->hoods->a[u];
        // Populate candidate set (those sharing at least one feature)
        tk_iuset_clear(data->candidates);
        khi = kh_get(tk_inv_nodes, data->I->nodes, u);
        if (khi == kh_end(data->I->nodes))
          continue;
        tk_iuset_t *ufs = kh_value(data->I->nodes, khi);
        tk_iuset_foreach(ufs, f, ({
          khi = kh_get(tk_inv_postings, data->I->postings, f);
          if (khi != kh_end(data->I->postings))
            tk_iuset_union(data->candidates, kh_value(data->I->postings, khi));
        }));
        // Get a sorted list of neighbors by distance
        tk_iuset_foreach(data->candidates, v, ({
          if (u == v)
            continue;
          khi = kh_get(tk_inv_nodes, data->I->nodes, v);
          if (khi == kh_end(data->I->nodes))
            continue;
          tk_iuset_t *vfs = kh_value(data->I->nodes, khi);
          double dist = 1.0 - tk_iuset_jaccard(ufs, vfs);
          tk_rvec_hasc(nbrs, tk_rank(v, dist));
        }))
        ks_heapsort(tk_rvec_asc, nbrs->n, nbrs->a);
      }
      break;
  }
}

static inline tk_inv_t *tk_inv_peek (lua_State *L, int i)
{
  return (tk_inv_t *) luaL_checkudata(L, i, TK_INV_MT);
}

static inline void tk_inv_destroy (
  tk_inv_t *I
) {
  if (I->destroyed)
    return;
  I->destroyed = true;
  tk_threads_destroy(I->pool);
  free(I->threads);
  int64_t i;
  tk_iuset_t *s;
  kh_foreach(I->nodes, i, s, ({
    tk_iuset_destroy(s);
  }));
  kh_destroy(tk_inv_nodes, I->nodes);
  #warning todo
}

static inline uint64_t tk_inv_size (
  tk_inv_t *I
) {
  return kh_size(I->nodes);
}

static inline tk_inv_hoods_t *tk_inv_neighborhoods (
  lua_State *L,
  tk_inv_t *I,
  uint64_t knn
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't query a destroyed index");
    return NULL;
  }
  tk_ivec_t *ids = tk_ivec_create(L, kh_size(I->nodes), 0, 0);
  int64_t id;
  tk_iuset_t *tmp;
  ids->n = 0;
  kh_foreach(I->nodes, id, tmp, ({
    ids->a[ids->n ++] = id;
  }));
  tk_ivec_asc(ids, 0, ids->n); // sort for cache locality
  tk_inv_hoods_t *hoods = tk_inv_hoods_create(L, ids->n, 0, 0);
  for (uint64_t i = 0; i < ids->n; i ++) {
    hoods->a[i] = tk_rvec_create(L, knn, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
    lua_pop(L, 1);
  }
  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    data->hoods = hoods;
    tk_thread_range(i, I->pool->n_threads, ids->n, &data->ufirst, &data->ulast);
  }
  tk_threads_signal(I->pool, TK_INV_NEIGHBORHOODS);
  lua_remove(L, -2); // pop ids_storage
  return hoods;
}

static inline tk_iuset_t *tk_inv_get (
  tk_inv_t *I,
  int64_t id
) {
  if (I->destroyed)
    return NULL;
  khint_t khi = kh_get(tk_inv_nodes, I->nodes, id);
  return khi == kh_end(I->nodes) ? NULL : kh_value(I->nodes, khi);
}

static inline void tk_inv_add (
  lua_State *L,
  tk_inv_t *I,
  int64_t id,
  tk_ivec_t *set_bits
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't add to a destroyed index");
    return;
  }
  int kha;
  khint_t khi;
  for (uint64_t i = 0; i < set_bits->n; i ++) {
    uint64_t b = set_bits->a[i];
    uint64_t s = b / I->features;
    uint64_t f = b % I->features;
    khi = kh_put(tk_inv_nodes, I->nodes, (int64_t) s + id, &kha);
    if (kha) kh_value(I->nodes, khi) = tk_iuset_create();
    tk_iuset_put(kh_value(I->nodes, khi), (int64_t) f, &kha);
    khi = kh_put(tk_inv_postings, I->postings, (int64_t) f, &kha);
    if (kha) kh_value(I->postings, khi) = tk_iuset_create();
    tk_iuset_put(kh_value(I->postings, khi), (int64_t) s + id, &kha);
  }
}

static inline void tk_inv_persist (
  lua_State *L,
  tk_inv_t *I,
  FILE *fh
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't persist a destroyed index");
    return;
  }
  #warning todo
}

static inline int tk_inv_gc_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_inv_destroy(I);
  return 0;
}

static inline int tk_inv_add_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  int64_t id = tk_lua_checkinteger(L, 2, "id");
  tk_ivec_t *set_bits = tk_ivec_peek(L, 3, "set_bits");
  tk_inv_add(L, I, id, set_bits);
  return 0;
}

static inline int tk_inv_remove_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_get_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_neighborhoods_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_neighbors_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  #warning todo
  return 0;
}

static inline int tk_inv_size_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) tk_inv_size(I));
  return 1;
}

static inline int tk_inv_threads_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) I->pool->n_threads);
  #warning todo
  return 0;
}

static inline int tk_inv_features_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) I->features);
  return 0;
}

static inline int tk_inv_persist_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_inv_persist(L, I, fh);
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
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_inv_destroy(I);
  return 0;
}

static luaL_Reg tk_inv_lua_mt_fns[] =
{
  { "add", tk_inv_add_lua },
  { "remove", tk_inv_remove_lua },
  { "get", tk_inv_get_lua },
  { "neighborhoods", tk_inv_neighborhoods_lua },
  { "neighbors", tk_inv_neighbors_lua },
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
  tk_inv_t *I = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  int Ii = tk_lua_absindex(L, -1);
  I->destroyed = false;
  I->features = features;
  I->nodes = kh_init(tk_inv_nodes);
  I->postings = kh_init(tk_inv_postings);
  I->threads = tk_malloc(L, n_threads * sizeof(tk_inv_thread_t));
  memset(I->threads, 0, n_threads * sizeof(tk_inv_thread_t));
  I->pool = tk_threads_create(L, n_threads, tk_inv_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    I->pool->threads[i].data = data;
    data->I = I;
    data->candidates = tk_iuset_create();
    data->seen = tk_iuset_create();
  }
  #warning todo
  return I;
}

static inline tk_inv_t *tk_inv_load (
  lua_State *L,
  FILE *fh
) {
  tk_inv_t *I = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  #warning todo
  return I;
}

#endif
