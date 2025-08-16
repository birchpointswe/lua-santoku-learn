#ifndef TK_HBI_H
#define TK_HBI_H

#include <assert.h>
#include <santoku/lua/utils.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/threads.h>

#define TK_HBI_BITS 32
#define tk_hbi_code_t uint32_t

KHASH_INIT(tk_hbi_buckets, tk_hbi_code_t, tk_ivec_t *, 1, kh_int_hash_func, kh_int_hash_equal)
typedef khash_t(tk_hbi_buckets) tk_hbi_buckets_t;

#define TK_HBI_MT "tk_hbi_t"
#define TK_HBI_EPH "tk_hbi_eph"

typedef tk_pvec_t * tk_hbi_hood_t;
#define tk_vec_name tk_hbi_hoods
#define tk_vec_base tk_hbi_hood_t
#define tk_vec_pushbase(L, x) tk_lua_get_ephemeron(L, TK_HBI_EPH, x)
#define tk_vec_peekbase(L, i) tk_pvec_peek(L, i, "hood")
#define tk_vec_limited
#include <santoku/vec/tpl.h>

#define tk_vec_name tk_hbi_codes
#define tk_vec_base tk_hbi_code_t
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef enum {
  TK_HBI_NEIGHBORHOODS,
  TK_HBI_MUTUAL,
} tk_hbi_stage_t;

typedef struct tk_hbi_thread_s tk_hbi_thread_t;

typedef struct tk_hbi_s {
  bool destroyed;
  uint64_t next_sid;
  uint64_t features;
  tk_hbi_buckets_t *buckets;
  tk_iumap_t *uid_sid;
  tk_iumap_t *sid_uid;
  tk_hbi_codes_t *codes;
  tk_hbi_thread_t *threads;
  tk_threadpool_t *pool;
} tk_hbi_t;

typedef struct tk_hbi_thread_s {
  tk_hbi_t *A;
  tk_hbi_hoods_t *hoods;
  tk_iumap_t *sid_idx;
  tk_ivec_t *uids;
  tk_ivec_t *sids;
  uint64_t ifirst, ilast;
  uint64_t k;
  uint64_t eps;
} tk_hbi_thread_t;

static inline tk_hbi_t *tk_hbi_peek (lua_State *L, int i)
{
  return (tk_hbi_t *) luaL_checkudata(L, i, TK_HBI_MT);
}

static inline tk_hbi_t *tk_hbi_peekopt (lua_State *L, int i)
{
  return (tk_hbi_t *) tk_lua_testuserdata(L, i, TK_HBI_MT);
}

static inline void tk_hbi_shrink (
  tk_hbi_t *A
) {
  if (A->destroyed)
    return;
  #warning todo
}

static inline void tk_hbi_destroy (
  tk_hbi_t *A
) {
  if (A->destroyed)
    return;
  A->destroyed = true;
  tk_iumap_destroy(A->uid_sid);
  tk_iumap_destroy(A->sid_uid);
  kh_destroy(tk_hbi_buckets, A->buckets);
  tk_threads_destroy(A->pool);
  free(A->threads);
}


static inline tk_hbi_code_t tk_hbi_pack (const char *V, uint64_t features)
{
  tk_hbi_code_t h = 0;
  size_t nbytes = (features + CHAR_BIT - 1) / CHAR_BIT;
  memcpy(&h, V, nbytes);
  if (features < 32) {
    tk_hbi_code_t mask = (features == 32) ? 0xFFFFFFFFu
      : ((features == 0) ? 0u : (((tk_hbi_code_t)1u << features) - 1u));
    h &= mask;
  }
  return h;
}

static inline tk_ivec_t *tk_hbi_ids (lua_State *L, tk_hbi_t *A)
{
  return tk_iumap_keys(L, A->uid_sid);
}

static inline int tk_hbi_ids_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  tk_iumap_keys(L, A->uid_sid);
  return 1;
}

static inline void tk_hbi_persist (
  lua_State *L,
  tk_hbi_t *A,
  FILE *fh
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't persist a destroyed index");
    return;
  }
  // core scalars
  tk_lua_fwrite(L, (char *) &A->destroyed, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &A->next_sid, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &A->features, sizeof(uint64_t), 1, fh);
  // buckets map   code → posting list
  khint_t nb = A->buckets ? kh_size(A->buckets) : 0;
  tk_lua_fwrite(L, (char *) &nb, sizeof(khint_t), 1, fh);
  for (khint_t i = kh_begin(A->buckets); i < kh_end(A->buckets); i ++)
    if (kh_exist(A->buckets, i)) {
      tk_hbi_code_t code = kh_key(A->buckets, i);
      tk_ivec_t *list   = kh_val(A->buckets, i);
      tk_lua_fwrite(L, (char *) &code, sizeof(tk_hbi_code_t), 1, fh);
      bool has = list && list->n;
      tk_lua_fwrite(L, (char *) &has, sizeof(bool), 1, fh);
      if (has) {
        uint64_t len = list->n;
        tk_lua_fwrite(L, (char *) &len, sizeof(uint64_t), 1, fh);
        tk_lua_fwrite(L, (char *) list->a, sizeof(int64_t), len, fh);
      }
    }
  // uid → sid map
  khint_t nkeys = A->uid_sid ? tk_iumap_size(A->uid_sid) : 0;
  tk_lua_fwrite(L, (char *) &nkeys, sizeof(khint_t), 1, fh);
  for (khint_t i = tk_iumap_begin(A->uid_sid); i < tk_iumap_end(A->uid_sid); i ++)
    if (tk_iumap_exist(A->uid_sid, i)) {
      int64_t k = (int64_t) tk_iumap_key(A->uid_sid, i);
      int64_t v = (int64_t) tk_iumap_value(A->uid_sid, i);
      tk_lua_fwrite(L, (char *) &k, sizeof(int64_t), 1, fh);
      tk_lua_fwrite(L, (char *) &v, sizeof(int64_t), 1, fh);
    }
  // sid → uid map
  nkeys = A->sid_uid ? tk_iumap_size(A->sid_uid) : 0;
  tk_lua_fwrite(L, (char *) &nkeys, sizeof(khint_t), 1, fh);
  for (khint_t i = tk_iumap_begin(A->sid_uid); i < tk_iumap_end(A->sid_uid); i ++)
    if (tk_iumap_exist(A->sid_uid, i)) {
      int64_t k = (int64_t) tk_iumap_key(A->sid_uid, i);
      int64_t v = (int64_t) tk_iumap_value(A->sid_uid, i);
      tk_lua_fwrite(L, (char *) &k, sizeof(int64_t), 1, fh);
      tk_lua_fwrite(L, (char *) &v, sizeof(int64_t), 1, fh);
    }
  // codes vector
  uint64_t cnum = A->codes ? A->codes->n : 0;
  tk_lua_fwrite(L, (char *) &cnum, sizeof(uint64_t), 1, fh);
  if (cnum)
    tk_lua_fwrite(L, (char *) A->codes->a, sizeof(tk_hbi_code_t), cnum, fh);
}

static inline uint64_t tk_hbi_size (
  tk_hbi_t *A
) {
  return tk_iumap_size(A->uid_sid);
}

static inline void tk_hbi_uid_remove (
  tk_hbi_t *A,
  int64_t uid
) {
  khint_t khi;
  khi = tk_iumap_get(A->uid_sid, uid);
  if (khi == tk_iumap_end(A->uid_sid))
    return;
  int64_t sid = tk_iumap_value(A->uid_sid, khi);
  tk_iumap_del(A->uid_sid, khi);
  khi = tk_iumap_get(A->sid_uid, sid);
  if (khi == tk_iumap_end(A->sid_uid))
    return;
  tk_iumap_del(A->sid_uid, khi);
}

static inline int64_t tk_hbi_uid_sid (
  tk_hbi_t *A,
  int64_t uid,
  bool create
) {
  int kha;
  khint_t khi;
  if (create) {
    int64_t sid = (int64_t) (A->next_sid ++);
    khi = tk_iumap_put(A->uid_sid, uid, &kha);
    if (!kha) {
      int64_t sid0 = tk_iumap_value(A->uid_sid, khi);
      khi = tk_iumap_get(A->sid_uid, sid0);
      if (khi != tk_iumap_end(A->sid_uid))
        tk_iumap_del(A->sid_uid, khi);
    }
    tk_iumap_value(A->uid_sid, khi) = sid;
    khi = tk_iumap_put(A->sid_uid, sid, &kha);
    tk_iumap_value(A->sid_uid, khi) = uid;
    return sid;
  } else {
    khi = tk_iumap_get(A->uid_sid, uid);
    if (khi == tk_iumap_end(A->uid_sid))
      return -1;
    else
      return tk_iumap_value(A->uid_sid, khi);
  }
}

static inline int64_t tk_hbi_sid_uid (
  tk_hbi_t *A,
  int64_t sid
) {
  khint_t khi = tk_iumap_get(A->sid_uid, sid);
  if (khi == tk_iumap_end(A->sid_uid))
    return -1;
  else
    return tk_iumap_value(A->sid_uid, khi);
}

static inline char *tk_hbi_sget (
  tk_hbi_t *A,
  int64_t sid
) {
  return (char *) (A->codes->a + (uint64_t) sid);
}

static inline char *tk_hbi_get (
  tk_hbi_t *A,
  int64_t uid
) {
  int64_t sid = tk_hbi_uid_sid(A, uid, false);
  if (sid < 0)
    return NULL;
  return tk_hbi_sget(A, sid);
}

static inline void tk_hbi_add (
  lua_State *L,
  tk_hbi_t *A,
  int Ai,
  tk_ivec_t *ids,
  char *data
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "add", "can't add to a destroyed index");
    return;
  }
  if (ids->n == 0)
    return;
  int kha;
  khint_t khi;
  for (uint64_t i = 0; i < ids->n; i ++) {
    int64_t sid = tk_hbi_uid_sid(A, ids->a[i], true);
    tk_ivec_t *bucket;
    tk_hbi_code_t h = tk_hbi_pack(data + i * ((A->features + CHAR_BIT - 1) / CHAR_BIT), A->features);
    khi = kh_put(tk_hbi_buckets, A->buckets, h, &kha);
    if (kha) {
      bucket = kh_value(A->buckets, khi) = tk_ivec_create(L, 0, 0, 0);
      tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
      lua_pop(L, 1);
    } else {
      bucket = kh_value(A->buckets, khi);
    }
    tk_ivec_push(bucket, sid);
    tk_hbi_codes_push(A->codes, h);
  }
}

static inline void tk_hbi_remove (
  lua_State *L,
  tk_hbi_t *A,
  int64_t uid
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "remove", "can't remove from a destroyed index");
    return;
  }
  tk_hbi_uid_remove(A, uid);
}

static inline void tk_hbi_extend_neighborhood (
  tk_hbi_t *A,
  uint64_t i,
  int64_t sid,
  char *V,
  tk_hbi_code_t h,
  tk_pvec_t *hood,
  tk_iumap_t *sid_idx,
  uint64_t k,
  int r
) {
  khint_t khi = kh_get(tk_hbi_buckets, A->buckets, h);
  if (khi != kh_end(A->buckets)) {
    tk_ivec_t *bucket = kh_value(A->buckets, khi);
    for (uint64_t i = 0; i < bucket->n; i ++) {
      int64_t sid1 = bucket->a[i];
      if (sid == sid1)
        continue;
      int64_t uid1 = tk_hbi_sid_uid(A, sid1);
      if (uid1 < 0)
        continue;
      khi = tk_iumap_get(sid_idx, sid1);
      if (khi == tk_iumap_end(sid_idx))
        continue;
      int64_t idx = tk_iumap_value(sid_idx, khi);
      tk_pvec_push(hood, tk_pair(idx, (int64_t) r));
    }
  }
}

static inline void tk_hbi_populate_neighborhood (
  tk_hbi_t *A,
  uint64_t i,
  int64_t sid,
  char *V,
  tk_pvec_t *hood,
  tk_iumap_t *sid_idx,
  uint64_t k,
  uint64_t eps
) {
  tk_hbi_code_t h = tk_hbi_pack(V, A->features);
  tk_hbi_extend_neighborhood(A, i, sid, V, h, hood, sid_idx, k, 0);
  if (k && hood->n >= k)
    return;
  int pos[TK_HBI_BITS];
  int nbits = (int) A->features;
  for (int r = 1; r <= (int) eps && r <= nbits; r ++) {
    for (int i = 0; i < r; i ++)
      pos[i] = i;
    while (true) {
      tk_hbi_code_t mask = 0;
      for (int i = 0; i < r; i ++)
        mask |= (1U << pos[i]);
      tk_hbi_extend_neighborhood(A, i, sid, V, h ^ mask, hood, sid_idx, k, r);
      if (k && hood->n >= k)
        return;
      int i;
      for (i = r - 1; i >= 0; i--) {
        if (pos[i] != i + nbits - r) {
          pos[i] ++;
          for (int j = i + 1; j < r; j ++)
            pos[j] = pos[j - 1] + 1;
          break;
        }
      }
      if (i < 0)
        break;
    }
  }
}

static inline void tk_hbi_mutualize (
  lua_State *L,
  tk_hbi_t *A,
  tk_hbi_hoods_t *hoods,
  tk_ivec_t *uids
) {
  if (A->destroyed)
    return;
  #warning "todo: mutualize"
  assert(false);
}

static inline void tk_hbi_neighborhoods (
  lua_State *L,
  tk_hbi_t *A,
  uint64_t k,
  uint64_t eps,
  uint64_t min,
  bool mutual,
  tk_hbi_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "neighborhoods", "can't query a destroyed index");
    return;
  }
  int kha;
  khint_t khi;
  tk_ivec_t *sids, *uids;
  if (uidsp && *uidsp) {
    // TODO: can we avoid copy? Just use uids directly? Need a way to push to
    // stack
    sids = tk_ivec_create(L, (*uidsp)->n, 0, 0);
    uids = tk_ivec_create(L, (*uidsp)->n, 0, 0);
    tk_ivec_copy(uids, *uidsp, 0, (int64_t) (*uidsp)->n, 0);
    for (uint64_t i = 0; i < uids->n; i ++)
      sids->a[i] = tk_hbi_uid_sid(A, uids->a[i], false);
  } else {
    sids = tk_iumap_values(L, A->uid_sid);
    tk_ivec_asc(sids, 0, sids->n); // sort for cache locality
    uids = tk_ivec_create(L, sids->n, 0, 0);
    for (uint64_t i = 0; i < sids->n; i ++)
      uids->a[i] = tk_hbi_sid_uid(A, sids->a[i]);
  }
  tk_iumap_t *sid_idx = tk_iumap_create();
  for (uint64_t i = 0; i < sids->n; i ++) {
    khi = tk_iumap_put(sid_idx, sids->a[i], &kha);
    tk_iumap_value(sid_idx, khi) = (int64_t) i;
  }
  tk_hbi_hoods_t *hoods = tk_hbi_hoods_create(L, uids->n, 0, 0);
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_pvec_create(L, k, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_HBI_EPH, -2, -1);
    lua_pop(L, 1);
  }
  for (uint64_t i = 0; i < A->pool->n_threads; i ++) {
    tk_hbi_thread_t *data = A->threads + i;
    data->uids = uids;
    data->sids = sids;
    data->hoods = hoods;
    data->sid_idx = sid_idx;
    data->k = k;
    data->eps = eps;
    tk_thread_range(i, A->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }
  tk_threads_signal(A->pool, TK_HBI_NEIGHBORHOODS, 0);
  if (mutual && k)
    tk_threads_signal(A->pool, TK_HBI_MUTUAL, 0);
  tk_iumap_destroy(sid_idx);
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
  lua_remove(L, -3); // sids
}

static inline tk_pvec_t *tk_hbi_neighbors_by_vec (
  tk_hbi_t *A,
  char *vec,
  int64_t sid0,
  uint64_t knn,
  uint64_t eps,
  tk_pvec_t *out
) {
  if (A->destroyed)
    return NULL;
  tk_pvec_clear(out);
  if (knn) {
    tk_pvec_ensure(out, knn);
    out->m = knn;
  }

  tk_hbi_code_t h0 = tk_hbi_pack(vec, A->features);
  int pos[TK_HBI_BITS];
  int nbits = (int) A->features;
  khint_t khi = kh_get(tk_hbi_buckets, A->buckets, h0);
  if (khi != kh_end(A->buckets)) {
    tk_ivec_t *bucket = kh_value(A->buckets, khi);
    for (uint64_t bi = 0; bi < bucket->n && (!knn || out->n < knn); bi ++) {
      int64_t sid1 = bucket->a[bi];
      if (sid1 == sid0) continue;
      int64_t uid1 = tk_hbi_sid_uid(A, sid1);
      if (uid1 < 0) continue;
      tk_pvec_push(out, tk_pair(uid1, 0));
    }
  }
  if (knn && out->n >= knn) {
    tk_pvec_asc(out, 0, out->n);
    return out;
  }
  for (int r = 1; r <= (int) eps && r <= nbits; r ++) {
    for (int i = 0; i < r; i ++)
      pos[i] = i;
    while (true) {
      tk_hbi_code_t mask = 0;
      for (int i = 0; i < r; i ++)
        mask |= (1U << pos[i]);
      khint_t khi = kh_get(tk_hbi_buckets, A->buckets, h0 ^ mask);
      if (khi != kh_end(A->buckets)) {
        tk_ivec_t *bucket = kh_value(A->buckets, khi);
        for (uint64_t bi = 0; bi < bucket->n && (!knn || out->n < knn); bi ++) {
          int64_t sid1 = bucket->a[bi];
          if (sid1 == sid0)
            continue;
          int64_t uid1 = tk_hbi_sid_uid(A, sid1);
          if (uid1 < 0)
            continue;
          tk_pvec_push(out, tk_pair(uid1, (int64_t) r));
        }
      }
      if (knn && out->n >= knn) {
        tk_pvec_asc(out, 0, out->n);
        return out;
      }
      int j;
      for (j = r - 1; j >= 0; j --) {
        if (pos[j] != j + nbits - r) {
          pos[j] ++;
          for (int k2 = j + 1; k2 < r; k2 ++)
            pos[k2] = pos[k2 - 1] + 1;
          break;
        }
      }
      if (j < 0)
        break;
    }
  }
  return out;
}

static inline tk_pvec_t *tk_hbi_neighbors_by_id (
  tk_hbi_t *A,
  int64_t uid,
  uint64_t knn,
  uint64_t eps,
  tk_pvec_t *out
) {
  int64_t sid0 = tk_hbi_uid_sid(A, uid, false);
  if (sid0 < 0) {
    tk_pvec_clear(out);
    return out;
  }
  return tk_hbi_neighbors_by_vec(A, tk_hbi_get(A, uid), sid0, knn, eps, out);
}

static inline int tk_hbi_gc_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  tk_hbi_destroy(A);
  return 0;
}

static inline int tk_hbi_add_lua (lua_State *L)
{
  int Ai = 1;
  tk_hbi_t *A = tk_hbi_peek(L, Ai);
  const char *data = tk_lua_checkustring(L, 2, "data");
  if (lua_type(L, 3) == LUA_TNUMBER) {
    int64_t s = (int64_t) tk_lua_checkunsigned(L, 3, "base_id");
    uint64_t n = tk_lua_optunsigned(L, 4, "n_nodes", 1);
    tk_ivec_t *ids = tk_ivec_create(L, n, 0, 0);
    tk_ivec_fill_indices(ids);
    tk_ivec_add(ids, s, 0, ids->n);
    tk_hbi_add(L, A, Ai, ids, (char *) data);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 3, "ids");
    tk_hbi_add(L, A, Ai, ids, (char *) data);
  }
  return 0;
}

static inline int tk_hbi_remove_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  int64_t id = tk_lua_checkinteger(L, 2, "id");
  tk_hbi_remove(L, A, id);
  return 0;
}

static inline int tk_hbi_get_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  int64_t id = tk_lua_checkinteger(L, 2, "id");
  char *data = tk_hbi_get(A, id);
  if (data == NULL)
    return 0;
  lua_pushlstring(L, (char *) data, ((A->features + CHAR_BIT - 1) / CHAR_BIT));
  return 1;
}

static inline int tk_hbi_neighborhoods_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  uint64_t k = tk_lua_optunsigned(L, 2, "k", 0);
  uint64_t eps = tk_lua_optunsigned(L, 3, "eps", 3); // NOTE: intentionally not A->features
  uint64_t min = tk_lua_optunsigned(L, 4, "min", 0);
  bool mutual = tk_lua_optboolean(L, 5, "mutual", false);
  tk_hbi_neighborhoods(L, A, k, eps, min, mutual, 0, 0);
  return 2;
}

static inline int tk_hbi_neighbors_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  uint64_t knn = tk_lua_optunsigned(L, 3, "knn", 0);
  uint64_t eps = tk_lua_optunsigned(L, 4, "eps", 0);
  tk_pvec_t *out = tk_pvec_peek(L, 5,  "out");
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_hbi_neighbors_by_id(A, uid, knn, eps, out);
  } else {
    char *vec = (char *) tk_lua_checkustring(L, 2, "vector");
    tk_hbi_neighbors_by_vec(A, vec, -1, knn, eps, out);
  }
  return 0;
}

static inline int tk_hbi_size_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  lua_pushinteger(L, (int64_t) tk_hbi_size(A));
  return 1;
}

static inline int tk_hbi_threads_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  lua_pushinteger(L, (int64_t) A->pool->n_threads);
  return 1;
}

static inline int tk_hbi_features_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  lua_pushinteger(L, (int64_t) A->features);
  return 1;
}

static inline int tk_hbi_persist_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_hbi_persist(L, A, fh);
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

static inline int tk_hbi_destroy_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  tk_hbi_destroy(A);
  return 0;
}

static inline int tk_hbi_shrink_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  tk_hbi_shrink(A);
  return 0;
}

static inline void tk_hbi_worker (void *dp, int sig)
{
  tk_hbi_stage_t stage = (tk_hbi_stage_t) sig;
  tk_hbi_thread_t *data = (tk_hbi_thread_t *) dp;

  switch (stage) {

    case TK_HBI_NEIGHBORHOODS:
      for (uint64_t i = data->ifirst; i <= data->ilast; i ++) {
        tk_pvec_t *hood = data->hoods->a[i];
        int64_t sid = data->sids->a[i];
        tk_hbi_populate_neighborhood(data->A, i, sid, tk_hbi_sget(data->A, sid), hood, data->sid_idx, data->k, data->eps);
      }
      break;

    case TK_HBI_MUTUAL: {
      #warning "todo: mutual"
      assert(false);
    }

  }
}

static luaL_Reg tk_hbi_lua_mt_fns[] =
{
  { "add", tk_hbi_add_lua },
  { "remove", tk_hbi_remove_lua },
  { "get", tk_hbi_get_lua },
  { "neighborhoods", tk_hbi_neighborhoods_lua },
  { "neighbors", tk_hbi_neighbors_lua },
  { "size", tk_hbi_size_lua },
  { "threads", tk_hbi_threads_lua },
  { "features", tk_hbi_features_lua },
  { "persist", tk_hbi_persist_lua },
  { "destroy", tk_hbi_destroy_lua },
  { "shrink", tk_hbi_shrink_lua },
  { "ids", tk_hbi_ids_lua },
  { NULL, NULL }
};

static inline void tk_hbi_suppress_unused_lua_mt_fns (void)
  { (void) tk_hbi_lua_mt_fns; }

static inline tk_hbi_t *tk_hbi_create (
  lua_State *L,
  uint64_t features,
  uint64_t n_threads
) {
  if (features > TK_HBI_BITS)
    tk_lua_verror(L, 3, "create", "features", "must be <= " STR(TK_HBI_BITS));
  tk_hbi_t *A = tk_lua_newuserdata(L, tk_hbi_t, TK_HBI_MT, tk_hbi_lua_mt_fns, tk_hbi_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  A->threads = tk_malloc(L, n_threads * sizeof(tk_hbi_thread_t));
  memset(A->threads, 0, n_threads * sizeof(tk_hbi_thread_t));
  A->pool = tk_threads_create(L, n_threads, tk_hbi_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_hbi_thread_t *data = A->threads + i;
    A->pool->threads[i].data = data;
    data->A = A;
  }
  A->features = features;
  A->buckets = kh_init(tk_hbi_buckets);
  A->uid_sid = tk_iumap_create();
  A->sid_uid = tk_iumap_create();
  A->codes = tk_hbi_codes_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
  A->destroyed = false;
  A->next_sid = 0;
  return A;
}

static inline tk_hbi_t *tk_hbi_load (
  lua_State *L,
  FILE *fh,
  uint64_t n_threads
) {
  // userdata + metatable
  tk_hbi_t *A = tk_lua_newuserdata(L, tk_hbi_t, TK_HBI_MT, tk_hbi_lua_mt_fns, tk_hbi_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  memset(A, 0, sizeof(tk_hbi_t));
  // core scalars
  tk_lua_fread(L, &A->destroyed, sizeof(bool), 1, fh);
  if (A->destroyed)
    tk_lua_verror(L, 2, "load", "index was destroyed when saved");
  tk_lua_fread(L, &A->next_sid,  sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &A->features,  sizeof(uint64_t), 1, fh);
  // buckets map   code → posting list
  A->buckets = kh_init(tk_hbi_buckets);
  khint_t nb = 0, k; int absent;
  tk_lua_fread(L, &nb, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nb; i ++) {
    tk_hbi_code_t code;
    bool has;
    tk_lua_fread(L, &code, sizeof(tk_hbi_code_t), 1, fh);
    tk_lua_fread(L, &has, sizeof(bool), 1, fh);
    k = kh_put(tk_hbi_buckets, A->buckets, code, &absent);
    if (has) {
      uint64_t len;
      tk_lua_fread(L, &len, sizeof(uint64_t), 1, fh);
      tk_ivec_t *list = tk_ivec_create(L, len, 0, 0);
      list->n = len;
      tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
      if (len)
        tk_lua_fread(L, list->a, sizeof(int64_t), len, fh);
      lua_pop(L, 1);
      kh_val(A->buckets, k) = list;
    } else {
      kh_val(A->buckets, k) = NULL;
    }
  }
  // uid → sid map
  A->uid_sid = tk_iumap_create();
  khint_t nkeys = 0; int64_t ikey, ival;
  tk_lua_fread(L, &nkeys, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nkeys; i ++) {
    tk_lua_fread(L, &ikey, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, &ival, sizeof(int64_t), 1, fh);
    k = tk_iumap_put(A->uid_sid, ikey, &absent);
    tk_iumap_value(A->uid_sid, k) = ival;
  }
  // sid → uid map
  A->sid_uid = tk_iumap_create();
  tk_lua_fread(L, &nkeys, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nkeys; i ++) {
    tk_lua_fread(L, &ikey, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, &ival, sizeof(int64_t), 1, fh);
    k = tk_iumap_put(A->sid_uid, ikey, &absent);
    tk_iumap_value(A->sid_uid, k) = ival;
  }
  // codes vector
  uint64_t cnum = 0;
  tk_lua_fread(L, &cnum, sizeof(uint64_t), 1, fh);
  A->codes = tk_hbi_codes_create(L, cnum, 0, 0);
  A->codes->n = cnum;
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  if (cnum)
    tk_lua_fread(L, A->codes->a, sizeof(tk_hbi_code_t), cnum, fh);
  lua_pop(L, 1);
  // thread pool
  A->threads = tk_malloc(L, n_threads * sizeof(tk_hbi_thread_t));
  memset(A->threads, 0, n_threads * sizeof(tk_hbi_thread_t));
  A->pool = tk_threads_create(L, n_threads, tk_hbi_worker);
  for (unsigned int t = 0; t < n_threads; t ++) {
    tk_hbi_thread_t *th = A->threads + t;
    A->pool->threads[t].data = th;
    th->A = A;
  }
  return A;
}

#endif
