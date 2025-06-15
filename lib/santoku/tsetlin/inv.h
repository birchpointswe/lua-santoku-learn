#ifndef TK_INV_H
#define TK_INV_H

#include <assert.h>
#include <santoku/lua/utils.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/threads.h>

#define TK_INV_MT "tk_inv_t"
#define TK_INV_EPH "tk_inv_eph"

typedef tk_ivec_t * tk_inv_posting_t;
#define tk_vec_name tk_inv_postings
#define tk_vec_base tk_inv_posting_t
#define tk_vec_pushbase(L, x) tk_lua_get_ephemeron(L, TK_INV_EPH, x)
#define tk_vec_peekbase(L, i) tk_ivec_peek(L, i, "posting")
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef tk_rvec_t * tk_inv_hood_t;
#define tk_vec_name tk_inv_hoods
#define tk_vec_base tk_inv_hood_t
#define tk_vec_pushbase(L, x) tk_lua_get_ephemeron(L, TK_INV_EPH, x)
#define tk_vec_peekbase(L, i) tk_rvec_peek(L, i, "hood")
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef enum {
  TK_INV_NEIGHBORHOODS,
  TK_INV_NEIGHBORS
} tk_inv_stage_t;

typedef struct tk_inv_thread_s tk_inv_thread_t;

typedef struct tk_inv_s {
  bool destroyed;
  int64_t next_sid;
  uint64_t features;
  tk_iumap_t *uid_sid;
  tk_iumap_t *sid_uid;
  tk_ivec_t *node_offsets;
  tk_ivec_t *node_bits;
  tk_inv_postings_t *postings;
  tk_inv_thread_t *threads;
  tk_threadpool_t *pool;
} tk_inv_t;

typedef struct tk_inv_thread_s {
  tk_inv_t *I;
  tk_inv_hoods_t *hoods;
  tk_iuset_t *seen;
  tk_iumap_t *sid_idx;
  tk_ivec_t *uids;
  tk_ivec_t *sids;
  uint64_t ifirst, ilast;
  double eps;
  uint64_t k;
} tk_inv_thread_t;

#define TK_INV_MT "tk_inv_t"
#define TK_INV_EPH "tk_inv_eph"

static inline tk_inv_t *tk_inv_peek (lua_State *L, int i)
{
  return (tk_inv_t *) luaL_checkudata(L, i, TK_INV_MT);
}

static inline tk_inv_t *tk_inv_peekopt (lua_State *L, int i)
{
  return (tk_inv_t *) tk_lua_testuserdata(L, i, TK_INV_MT);
}

static inline void tk_inv_shrink (
  tk_inv_t *I
) {
  if (I->destroyed)
    return;
  #warning todo
}

static inline void tk_inv_destroy (
  tk_inv_t *I
) {
  if (I->destroyed)
    return;
  I->destroyed = true;
  tk_iumap_destroy(I->uid_sid);
  tk_iumap_destroy(I->sid_uid);
  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    tk_iuset_destroy(data->seen);
  }
  tk_threads_destroy(I->pool);
  free(I->threads);
}

static inline double tk_inv_jaccard (
  int64_t *abits,
  size_t asize,
  int64_t *bbits,
  size_t bsize
) {
  size_t i = 0, j = 0;
  size_t intersection = 0;
  size_t union_count = 0;
  while (i < asize && j < bsize) {
    if (abits[i] == bbits[j]) {
      intersection ++;
      union_count ++;
      i ++;
      j ++;
    } else if (abits[i] < bbits[j]) {
      union_count ++;
      i ++;
    } else {
      union_count ++;
      j ++;
    }
  }

  // Remaining elements
  union_count += (asize - i) + (bsize - j);

  if (union_count == 0) // both empty → define Jaccard as 1.0
    return 1.0;

  return (double) intersection / (double) union_count;
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

static inline uint64_t tk_inv_size (
  tk_inv_t *I
) {
  return tk_iumap_size(I->uid_sid);
}

static inline void tk_inv_uid_remove (
  tk_inv_t *I,
  int64_t uid
) {
  khint_t khi;
  khi = tk_iumap_get(I->uid_sid, uid);
  if (khi == kh_end(I->uid_sid))
    return;
  tk_iumap_del(I->uid_sid, khi);
  khi = tk_iumap_get(I->sid_uid, uid);
  if (khi == kh_end(I->sid_uid))
    return;
  tk_iumap_del(I->sid_uid, khi);
}

static inline int64_t tk_inv_uid_sid (
  tk_inv_t *I,
  int64_t uid,
  bool create
) {
  int kha;
  khint_t khi;
  if (create) {
    int64_t sid = (int64_t) (I->next_sid ++);
    khi = tk_iumap_put(I->uid_sid, uid, &kha);
    if (!kha) {
      int64_t sid0 = tk_iumap_value(I->uid_sid, khi);
      khi = tk_iumap_get(I->sid_uid, sid0);
      if (khi != tk_iumap_end(I->sid_uid))
        tk_iumap_del(I->sid_uid, khi);
    }
    tk_iumap_value(I->uid_sid, khi) = sid;
    khi = tk_iumap_put(I->sid_uid, sid, &kha);
    tk_iumap_value(I->sid_uid, khi) = uid;
    return sid;
  } else {
    khi = tk_iumap_get(I->uid_sid, uid);
    if (khi == kh_end(I->uid_sid))
      return -1;
    else
      return tk_iumap_value(I->uid_sid, khi);
  }
}

static inline int64_t tk_inv_sid_uid (
  tk_inv_t *I,
  int64_t sid
) {
  khint_t khi = tk_iumap_get(I->sid_uid, sid);
  if (khi == kh_end(I->sid_uid))
    return -1;
  else
    return tk_iumap_value(I->sid_uid, khi);
}

static inline int64_t *tk_inv_sget (
  tk_inv_t *I,
  int64_t sid,
  size_t *np
) {
  int64_t start = I->node_offsets->a[sid];
  int64_t end = sid + 1 == (int64_t) I->node_offsets->n ? (int64_t) I->node_bits->n : I->node_offsets->a[sid + 1];
  *np = (size_t) (end - start);
  return I->node_bits->a + start;
}

static inline int64_t *tk_inv_get (
  tk_inv_t *I,
  int64_t uid,
  size_t *np
) {
  int64_t sid = tk_inv_uid_sid(I, uid, false);
  if (sid < 0)
    return NULL;
  return tk_inv_sget(I, sid, np);
}

static inline void tk_inv_add (
  lua_State *L,
  tk_inv_t *I,
  int Ii,
  tk_ivec_t *ids,
  tk_ivec_t *node_bits
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "add", "can't add to a destroyed index");
    return;
  }
  if (node_bits->n == 0)
    return;
  int kha;
  khint_t khi;
  tk_ivec_asc(node_bits, 0, node_bits->n);
  // tk_ivec_dedupe(node_bits, 0, node_bits->n);
  tk_iuset_t *seen = tk_iuset_create();
  for (uint64_t i = 0; i < node_bits->n; i ++) {
    int64_t b = node_bits->a[i];
    if (b < 0)
      continue;
    int64_t s = b / (int64_t) I->features;
    if (s > (int64_t) ids->n)
      continue;
    int64_t fid = b % (int64_t) I->features;
    int64_t uid = ids->a[s];
    khi = tk_iuset_put(seen, uid, &kha);
    int64_t sid = tk_inv_uid_sid(I, uid, kha);
    if (kha)
      tk_ivec_push(I->node_offsets, (int64_t) I->node_bits->n);
    tk_ivec_push(I->postings->a[fid], sid);
    tk_ivec_push(I->node_bits, fid);
  }
  tk_iuset_destroy(seen);
}

static inline void tk_inv_remove (
  lua_State *L,
  tk_inv_t *I,
  int64_t uid
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "remove", "can't remove from a destroyed index");
    return;
  }
  tk_inv_uid_remove(I, uid);
}

static inline void tk_inv_neighborhoods (
  lua_State *L,
  tk_inv_t *I,
  uint64_t k,
  uint64_t eps,
  tk_inv_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "neighborhoods", "can't query a destroyed index");
    return;
  }
  int kha;
  khint_t khi;
  tk_ivec_t *sids = tk_iumap_values(L, I->uid_sid);
  tk_ivec_asc(sids, 0, sids->n); // sort for cache locality
  tk_iumap_t *sid_idx = tk_iumap_create();
  for (uint64_t i = 0; i < sids->n; i ++) {
    khi = tk_iumap_put(sid_idx, sids->a[i], &kha);
    tk_iumap_value(sid_idx, khi) = (int64_t) i;
  }
  tk_ivec_t *uids = tk_ivec_create(L, sids->n, 0, 0);
  for (uint64_t i = 0; i < sids->n; i ++)
    uids->a[i] = tk_inv_sid_uid(I, sids->a[i]);
  tk_inv_hoods_t *hoods = tk_inv_hoods_create(L, uids->n, 0, 0);
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_rvec_create(L, k, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
    lua_pop(L, 1);
  }
  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    data->uids = uids;
    data->sids = sids;
    data->hoods = hoods;
    data->sid_idx = sid_idx;
    data->eps = eps;
    data->k = k;
    tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }
  tk_threads_signal(I->pool, TK_INV_NEIGHBORHOODS);
  tk_iumap_destroy(sid_idx);
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
  lua_remove(L, -3); // sids
}

static inline tk_pvec_t *tk_inv_neighbors (
  lua_State *L,
  tk_inv_t *I,
  char *vec
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "neighbors", "can't query a destroyed index");
    return NULL;
  }
  #warning todo
  lua_pushnil(L);
  return NULL;
}

static inline int tk_inv_gc_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_inv_destroy(I);
  return 0;
}

static inline int tk_inv_add_lua (lua_State *L)
{
  int Ii = 1;
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_ivec_t *node_bits = tk_ivec_peek(L, 2, "node_bits");
  if (lua_type(L, 3) == LUA_TNUMBER) {
    int64_t s = (int64_t) tk_lua_checkunsigned(L, 3, "base_id");
    uint64_t n = tk_lua_checkunsigned(L, 4, "n_nodes");
    tk_ivec_t *ids = tk_ivec_create(L, n, 0, 0);
    tk_ivec_fill_indices(ids);
    tk_ivec_add(ids, s, 0, ids->n);
    tk_inv_add(L, I, Ii, ids, node_bits);
    lua_pop(L, 1);
  } else {
    tk_inv_add(L, I, Ii, tk_ivec_peek(L, 3, "ids"), node_bits);
  }
  return 0;
}

static inline int tk_inv_remove_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  int64_t id = tk_lua_checkinteger(L, 2, "id");
  tk_inv_remove(L, I, id);
  return 0;
}

static inline int tk_inv_get_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  int64_t id = tk_lua_checkinteger(L, 2, "id");
  size_t n = 0;
  int64_t *data = tk_inv_get(I, id, &n);
  int64_t *dup = n ? tk_malloc(L, n * sizeof(int64_t)) : NULL;
  memcpy(dup, data, n * sizeof(int64_t));
  tk_ivec_create(L, n, dup, 0);
  return 1;
}

static inline int tk_inv_neighborhoods_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  uint64_t k = tk_lua_checkunsigned(L, 2, "k");
  double epsf = tk_lua_checkposdouble(L, 3, "eps");
  uint64_t eps = (uint64_t) (epsf < 1.0 ? (double) I->features * epsf : epsf);
  tk_inv_neighborhoods(L, I, k, eps, 0, 0);
  return 2;
}

static inline int tk_inv_neighbors_lua (lua_State *L)
{
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

static inline int tk_inv_shrink_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_inv_shrink(I);
  return 0;
}

static inline void tk_inv_worker (void *dp, int sig)
{

  tk_inv_stage_t stage = (tk_inv_stage_t) sig;
  tk_inv_thread_t *data = (tk_inv_thread_t *) dp;
  tk_inv_t *I = data->I;
  tk_iuset_t *seen = data->seen;
  tk_inv_hoods_t *hoods = data->hoods;
  tk_ivec_t *sids = data->sids;
  tk_iumap_t *sid_idx = data->sid_idx;
  double eps = data->eps;
  int kha;
  khint_t khi;
  double dist;
  int64_t usid, vsid, fid;
  int64_t start, end;
  int64_t *ubits, *vbits;
  size_t nubits, nvbits;
  tk_rvec_t *uhood;
  tk_ivec_t *vsids;

  switch (stage) {

    case TK_INV_NEIGHBORHOODS:
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {
        usid = sids->a[i];
        if (tk_iumap_get(I->sid_uid, usid) == tk_iumap_end(I->sid_uid))
          continue;
        uhood = hoods->a[i];
        ubits = tk_inv_sget(I, usid, &nubits);
        start = I->node_offsets->a[usid];
        end = usid + 1 == (int64_t) I->node_offsets->n ? (int64_t) I->node_bits->n : I->node_offsets->a[usid + 1];
        tk_iuset_clear(seen);
        for (int64_t j = start; j < end; j ++) {
          fid = I->node_bits->a[j];
          vsids = I->postings->a[fid];
          for (uint64_t k = 0; k < vsids->n; k ++) {
            vsid = vsids->a[k];
            khi = tk_iumap_get(sid_idx, vsid);
            if (khi == tk_iumap_end(sid_idx))
              continue;
            int64_t iv = tk_iumap_value(sid_idx, khi);
            if (usid == vsid)
              continue;
            if (tk_iumap_get(I->sid_uid, vsid) == tk_iumap_end(I->sid_uid))
              continue;
            khi = tk_iuset_put(seen, vsid, &kha);
            if (!kha)
              continue;
            vbits = tk_inv_sget(I, vsid, &nvbits);
            dist = 1.0 - tk_inv_jaccard(ubits, nubits, vbits, nvbits);
            if (dist < eps)
              tk_rvec_hasc(uhood, tk_rank(iv, dist));
          }
        }
        tk_rvec_asc(uhood, 0, uhood->n);
      }
      break;

    case TK_INV_NEIGHBORS:
      #warning todo
      break;
  }
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
  { "shrink", tk_inv_shrink_lua },
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
  I->destroyed = false;
  I->next_sid = 0;
  I->features = features;
  I->uid_sid = tk_iumap_create();
  I->sid_uid = tk_iumap_create();
  I->node_offsets = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
  lua_pop(L, 1);
  I->node_bits = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
  lua_pop(L, 1);
  I->postings = tk_inv_postings_create(L, features, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
  for (uint64_t i = 0; i < features; i ++) {
    I->postings->a[i] = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
    lua_pop(L, 1);
  }
  lua_pop(L, 1);
  I->threads = tk_malloc(L, n_threads * sizeof(tk_inv_thread_t));
  memset(I->threads, 0, n_threads * sizeof(tk_inv_thread_t));
  I->pool = tk_threads_create(L, n_threads, tk_inv_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    I->pool->threads[i].data = data;
    data->I = I;
    data->seen = tk_iuset_create();
  }
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
