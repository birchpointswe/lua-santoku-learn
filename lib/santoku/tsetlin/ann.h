#ifndef TK_ANN_H
#define TK_ANN_H

#include <assert.h>
#include <santoku/lua/utils.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/threads.h>

#define TK_ANN_BITS 32
#define tk_ann_hash_t uint32_t

KHASH_INIT(tk_ann_buckets, tk_ann_hash_t, tk_ivec_t *, 1, kh_int_hash_func, kh_int_hash_equal)
typedef khash_t(tk_ann_buckets) tk_ann_buckets_t;

#define TK_ANN_MT "tk_ann_t"
#define TK_ANN_EPH "tk_ann_eph"

typedef tk_pvec_t * tk_ann_hood_t;
#define tk_vec_name tk_ann_hoods
#define tk_vec_base tk_ann_hood_t
#define tk_vec_pushbase(L, x) tk_lua_get_ephemeron(L, TK_ANN_EPH, x)
#define tk_vec_peekbase(L, i) tk_pvec_peek(L, i, "hood")
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef enum {
  TK_ANN_NEIGHBORHOODS,
} tk_ann_stage_t;

typedef struct tk_ann_thread_s tk_ann_thread_t;

typedef struct tk_ann_s {
  bool destroyed;
  uint64_t next_sid;
  uint64_t bucket_target;
  uint64_t features;
  uint64_t probe_radius;
  tk_ivec_t *hash_bits;
  tk_ann_buckets_t *buckets;
  tk_iumap_t *uid_sid;
  tk_iumap_t *sid_uid;
  tk_cvec_t *vectors;
  tk_ann_thread_t *threads;
  tk_threadpool_t *pool;
} tk_ann_t;

typedef struct tk_ann_thread_s {
  tk_ann_t *A;
  tk_ann_hoods_t *hoods;
  tk_iumap_t *sid_idx;
  tk_ivec_t *uids;
  tk_ivec_t *sids;
  uint64_t ifirst, ilast;
  uint64_t eps;
  uint64_t k;
} tk_ann_thread_t;

static inline tk_ann_t *tk_ann_peek (lua_State *L, int i)
{
  return (tk_ann_t *) luaL_checkudata(L, i, TK_ANN_MT);
}

static inline tk_ann_t *tk_ann_peekopt (lua_State *L, int i)
{
  return (tk_ann_t *) tk_lua_testuserdata(L, i, TK_ANN_MT);
}

static inline void tk_ann_shrink (
  tk_ann_t *A
) {
  if (A->destroyed)
    return;
  #warning todo
}

static inline void tk_ann_destroy (
  tk_ann_t *A
) {
  if (A->destroyed)
    return;
  A->destroyed = true;
  tk_iumap_destroy(A->uid_sid);
  tk_iumap_destroy(A->sid_uid);
  kh_destroy(tk_ann_buckets, A->buckets);
  tk_threads_destroy(A->pool);
  free(A->threads);
}

static inline uint64_t tk_ann_hamming_mask (
  const unsigned char *a,
  const unsigned char *b,
  const unsigned char *mask,
  uint64_t n_dims
) {
  uint64_t full_bytes = BITS_BYTES(n_dims);
  uint64_t t = 0;
  for (uint64_t i = 0; i < full_bytes; i ++)
    t += (uint64_t) popcount((a[i] ^ b[i]) & mask[i]);
  return t;
}

static inline uint64_t tk_ann_hamming (
  const unsigned char *a,
  const unsigned char *b,
  uint64_t n_dims
) {
  uint64_t full_bytes = BITS_BYTES(n_dims);
  uint64_t rem_bits = BITS_BIT(n_dims);
  uint64_t dist = 0;
  for (uint64_t i = 0; i < full_bytes - (rem_bits > 0); i ++)
    dist += popcount(a[i] ^ b[i]);
  if (rem_bits > 0) {
    unsigned char x = a[full_bytes - 1] ^ b[full_bytes - 1];
    unsigned char mask = (1U << rem_bits) - 1;
    dist += popcount(x & mask);
  }
  return dist;
}

static inline tk_ann_hash_t tk_ann_hash (
  tk_ann_t *A,
  char *data
) {
  tk_ann_hash_t h = 0;
  if (A->hash_bits->n == 0)
    return h;
  for (uint64_t i = 0; i < A->hash_bits->n; i ++) {
    int64_t b = A->hash_bits->a[i];
    if (b < 0)
      continue;
    uint64_t chunk = (uint64_t) b / CHAR_BIT;
    uint64_t pos = (uint64_t) b % CHAR_BIT;
    uint64_t n_chunks = (A->features + CHAR_BIT - 1) / CHAR_BIT;
    if (chunk >= n_chunks) {
      fprintf(stderr, "BUG: chunk=%llu, n_chunks=%llu, b=%lld, n_features=%llu\n",
              (unsigned long long)chunk, (unsigned long long)n_chunks, (long long)b, (unsigned long long)A->features);
      abort();
    }
    if (data[chunk] & (1 << pos))
      h |= ((uint32_t) 1 << i);
  }
  return h;
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
  // core scalars
  tk_lua_fwrite(L, (char *) &A->destroyed, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &A->next_sid, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &A->bucket_target, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &A->features, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &A->probe_radius, sizeof(uint64_t), 1, fh);
  // hash_bits vector
  uint64_t n_hash = A->hash_bits ? A->hash_bits->n : 0;
  tk_lua_fwrite(L, (char *) &n_hash, sizeof(uint64_t), 1, fh);
  if (n_hash)
    tk_lua_fwrite(L, (char *) A->hash_bits->a, sizeof(int64_t),  n_hash, fh);
  // buckets map  (hash → posting list)
  khint_t nb = A->buckets ? kh_size(A->buckets) : 0;
  tk_lua_fwrite(L, (char *) &nb, sizeof(khint_t), 1, fh);
  for (khint_t i = kh_begin(A->buckets); i < kh_end(A->buckets); i ++)
    if (kh_exist(A->buckets, i)) {
      tk_ann_hash_t hkey = kh_key(A->buckets, i);
      tk_ivec_t *plist  = kh_val(A->buckets, i);
      tk_lua_fwrite(L, (char *) &hkey, sizeof(tk_ann_hash_t), 1, fh);
      bool has = plist && plist->n;
      tk_lua_fwrite(L, (char *) &has, sizeof(bool),           1, fh);
      if (has) {
        uint64_t plen = plist->n;
        tk_lua_fwrite(L, (char *) &plen, sizeof(uint64_t),    1, fh);
        tk_lua_fwrite(L, (char *) plist->a, sizeof(int64_t),  plen, fh);
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
  for (khint_t i = kh_begin(A->sid_uid); i < kh_end(A->sid_uid); i ++)
    if (kh_exist(A->sid_uid, i)) {
      int64_t k = (int64_t) tk_iumap_key(A->sid_uid, i);
      int64_t v = (int64_t) tk_iumap_value(A->sid_uid, i);
      tk_lua_fwrite(L, (char *) &k, sizeof(int64_t), 1, fh);
      tk_lua_fwrite(L, (char *) &v, sizeof(int64_t), 1, fh);
    }
  // vectors (contiguous float / double array)
  uint64_t vcount = A->vectors ? A->vectors->n : 0;
  tk_lua_fwrite(L, (char *) &vcount, sizeof(uint64_t), 1, fh);
  if (vcount)
    tk_lua_fwrite(L, (char *) A->vectors->a, sizeof(double), vcount, fh);
}

static inline uint64_t tk_ann_size (
  tk_ann_t *A
) {
  return tk_iumap_size(A->uid_sid);
}

static inline void tk_ann_uid_remove (
  tk_ann_t *A,
  int64_t uid
) {
  khint_t khi;
  khi = tk_iumap_get(A->uid_sid, uid);
  if (khi == tk_iumap_end(A->uid_sid))
    return;
  int64_t sid = tk_iumap_value(A->uid_sid, khi);
  tk_iumap_del(A->uid_sid, khi);
  khi = tk_iumap_get(A->sid_uid, sid);
  if (khi == kh_end(A->sid_uid))
    return;
  tk_iumap_del(A->sid_uid, khi);
}

static inline int64_t tk_ann_uid_sid (
  tk_ann_t *A,
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

static inline int64_t tk_ann_sid_uid (
  tk_ann_t *A,
  int64_t sid
) {
  khint_t khi = tk_iumap_get(A->sid_uid, sid);
  if (khi == kh_end(A->sid_uid))
    return -1;
  else
    return tk_iumap_value(A->sid_uid, khi);
}

static inline char *tk_ann_sget (
  tk_ann_t *A,
  int64_t sid
) {
  return A->vectors->a + (uint64_t) sid * ((A->features + CHAR_BIT - 1) / CHAR_BIT);
}

static inline char *tk_ann_get (
  tk_ann_t *A,
  int64_t uid
) {
  int64_t sid = tk_ann_uid_sid(A, uid, false);
  if (sid < 0)
    return NULL;
  return tk_ann_sget(A, sid);
}

static inline void tk_ann_add (
  lua_State *L,
  tk_ann_t *A,
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
    int64_t sid = tk_ann_uid_sid(A, ids->a[i], true);
    tk_ivec_t *bucket;
    tk_ann_hash_t h = tk_ann_hash(A, data + i * ((A->features + CHAR_BIT - 1) / CHAR_BIT));
    khi = kh_put(tk_ann_buckets, A->buckets, h, &kha);
    if (kha) {
      bucket = kh_value(A->buckets, khi) = tk_ivec_create(L, 0, 0, 0);
      tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
      lua_pop(L, 1);
    } else {
      bucket = kh_value(A->buckets, khi);
    }
    tk_ivec_push(bucket, sid);
    tk_cvec_t datavec = {
      .n = (A->features + CHAR_BIT - 1) / CHAR_BIT,
      .m = (A->features + CHAR_BIT - 1) / CHAR_BIT,
      .a = (char *) data + i * ((A->features + CHAR_BIT - 1) / CHAR_BIT) };
    tk_cvec_copy(L, A->vectors, &datavec, 0, (int64_t) datavec.n, sid * (int64_t) ((A->features + CHAR_BIT - 1) / CHAR_BIT));
  }
}

static inline void tk_ann_remove (
  lua_State *L,
  tk_ann_t *A,
  int64_t uid
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "remove", "can't remove from a destroyed index");
    return;
  }
  tk_ann_uid_remove(A, uid);
}

static inline void tk_ann_extend_neighborhood (
  tk_ann_t *A,
  uint64_t i,
  int64_t sid,
  char *V,
  tk_ann_hash_t h,
  tk_pvec_t *hood,
  tk_iumap_t *sid_idx,
  uint64_t eps
) {
  khint_t khi = kh_get(tk_ann_buckets, A->buckets, h);
  if (khi != kh_end(A->buckets)) {
    tk_ivec_t *bucket = kh_value(A->buckets, khi);
    for (uint64_t i = 0; i < bucket->n; i ++) {
      int64_t sid1 = bucket->a[i];
      if (sid == sid1)
        continue;
      int64_t uid1 = tk_ann_sid_uid(A, sid1);
      if (uid1 < 0)
        continue;
      khi = tk_iumap_get(sid_idx, sid1);
      if (khi == tk_iumap_end(sid_idx))
        continue;
      int64_t idx = tk_iumap_value(sid_idx, khi);
      const char *V1 = tk_ann_sget(A, sid1);
      uint64_t m = tk_ann_hamming((const unsigned char *) V, (const unsigned char *) V1, A->features);
      if (m <= eps)
        tk_pvec_hasc(hood, tk_pair(idx, (int64_t) m));
    }
    tk_pvec_asc(hood, 0, hood->n);
  }
}

static inline void tk_ann_populate_neighborhood (
  tk_ann_t *A,
  uint64_t i,
  int64_t sid,
  char *V,
  tk_pvec_t *hood,
  tk_iumap_t *sid_idx,
  uint64_t eps
) {
  tk_ann_hash_t h = tk_ann_hash(A, V);
  tk_ann_extend_neighborhood(A, i, sid, V, h, hood, sid_idx, eps);
  int pos[TK_ANN_BITS];
  for (int r = 1; r <= (int) A->probe_radius && r <= TK_ANN_BITS; r ++) {
    for (int i = 0; i < r; i ++)
      pos[i] = i;
    while (true) {
      tk_ann_hash_t mask = 0;
      for (int i = 0; i < r; i ++)
        mask |= (1U << pos[i]);
      tk_ann_extend_neighborhood(A, i, sid, V, h ^ mask, hood, sid_idx, eps);
      int i;
      for (i = r - 1; i >= 0; i--) {
        if (pos[i] != i + TK_ANN_BITS - r) {
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

static inline void tk_ann_neighborhoods (
  lua_State *L,
  tk_ann_t *A,
  uint64_t k,
  uint64_t eps,
  tk_ann_hoods_t **hoodsp,
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
    tk_ivec_copy(L, uids, *uidsp, 0, (int64_t) (*uidsp)->n, 0);
    for (uint64_t i = 0; i < uids->n; i ++)
      sids->a[i] = tk_ann_uid_sid(A, uids->a[i], false);
  } else {
    sids = tk_iumap_values(L, A->uid_sid);
    tk_ivec_asc(sids, 0, sids->n); // sort for cache locality
    uids = tk_ivec_create(L, sids->n, 0, 0);
    for (uint64_t i = 0; i < sids->n; i ++)
      uids->a[i] = tk_ann_sid_uid(A, sids->a[i]);
  }
  tk_iumap_t *sid_idx = tk_iumap_create();
  for (uint64_t i = 0; i < sids->n; i ++) {
    khi = tk_iumap_put(sid_idx, sids->a[i], &kha);
    tk_iumap_value(sid_idx, khi) = (int64_t) i;
  }
  tk_ann_hoods_t *hoods = tk_ann_hoods_create(L, uids->n, 0, 0);
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_pvec_create(L, k, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_ANN_EPH, -2, -1);
    lua_pop(L, 1);
  }
  for (uint64_t i = 0; i < A->pool->n_threads; i ++) {
    tk_ann_thread_t *data = A->threads + i;
    data->uids = uids;
    data->sids = sids;
    data->hoods = hoods;
    data->sid_idx = sid_idx;
    data->eps = eps;
    data->k = k;
    tk_thread_range(i, A->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }
  tk_threads_signal(A->pool, TK_ANN_NEIGHBORHOODS);
  tk_iumap_destroy(sid_idx);
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
  lua_remove(L, -3); // sids
}

static inline tk_pvec_t *tk_ann_neighbors_by_vec (
  lua_State *L,
  tk_ann_t *A,
  char *vec,
  int64_t sid0,
  uint64_t knn,
  uint64_t eps,
  tk_pvec_t *out
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "neighbors", "can't query a destroyed index");
    return NULL;
  }
  if (!out)
    out = tk_pvec_create(L, knn, 0, 0);
  out->n = 0;
  const tk_ann_hash_t h0 = tk_ann_hash(A, vec);
  int pos[TK_ANN_BITS];
  for (int r = 0; r <= (int) A->probe_radius && r <= TK_ANN_BITS; r ++) {
    for (int i = 0; i < r; i ++)
      pos[i] = i;
    while (true) {
      tk_ann_hash_t mask = 0;
      for (int i = 0; i < r; i ++)
        mask |= (1U << pos[i]);
      khint_t khi = kh_get(tk_ann_buckets, A->buckets, h0 ^ mask);
      if (khi != kh_end(A->buckets)) {
        tk_ivec_t *bucket = kh_value(A->buckets, khi);
        for (uint64_t bi = 0; bi < bucket->n; bi ++) {
          int64_t sid1 = bucket->a[bi];
          if (sid1 == sid0)
            continue;
          int64_t uid1 = tk_ann_sid_uid(A, sid1);
          if (uid1 < 0)
            continue;
          uint64_t d = tk_ann_hamming((const unsigned char *) vec, (const unsigned char *) tk_ann_sget(A, sid1), A->features);
          if (d <= eps) {
            if (knn > 0)
              tk_pvec_hasc(out, tk_pair(uid1, (int64_t) d));
            else
              tk_pvec_push(out, tk_pair(uid1, (int64_t) d));
          }
        }
      }
      int j;
      for (j = r - 1; j >= 0; j --) {
        if (pos[j] != j + TK_ANN_BITS - r) {
          pos[j] ++;
          for (int k = j + 1; k < r; k ++)
            pos[k] = pos[k - 1] + 1;
          break;
        }
      }
      if (j < 0)
        break;
    }
  }
  tk_pvec_asc(out, 0, out->n);
  return out;
}

static inline tk_pvec_t *tk_ann_neighbors_by_id (
  lua_State *L,
  tk_ann_t *A,
  int64_t uid,
  uint64_t knn,
  uint64_t eps,
  tk_pvec_t *out
) {
  int64_t sid0 = tk_ann_uid_sid(A, uid, false);
  if (sid0 < 0)
    return out ? (out->n = 0, out) : tk_pvec_create(L, 0, 0, 0);
  return tk_ann_neighbors_by_vec(L, A, tk_ann_get(A, uid), sid0, knn, eps, out);
}

static inline int tk_ann_gc_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_ann_destroy(A);
  return 0;
}

static inline int tk_ann_add_lua (lua_State *L)
{
  int Ai = 1;
  tk_ann_t *A = tk_ann_peek(L, Ai);
  const char *data = tk_lua_checkustring(L, 2, "data");
  if (lua_type(L, 3) == LUA_TNUMBER) {
    int64_t s = (int64_t) tk_lua_checkunsigned(L, 3, "base_id");
    uint64_t n = tk_lua_optunsigned(L, 4, "n_nodes", 1);
    tk_ivec_t *ids = tk_ivec_create(L, n, 0, 0);
    tk_ivec_fill_indices(ids);
    tk_ivec_add(ids, s, 0, ids->n);
    tk_ann_add(L, A, Ai, ids, (char *) data);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 3, "ids");
    tk_ann_add(L, A, Ai, ids, (char *) data);
  }
  return 0;
}

static inline int tk_ann_remove_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  int64_t id = tk_lua_checkinteger(L, 2, "id");
  tk_ann_remove(L, A, id);
  return 0;
}

static inline int tk_ann_get_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  int64_t id = tk_lua_checkinteger(L, 2, "id");
  char *data = tk_ann_get(A, id);
  if (data == NULL)
    return 0;
  lua_pushlstring(L, (char *) data, ((A->features + CHAR_BIT - 1) / CHAR_BIT));
  return 1;
}

static inline int tk_ann_neighborhoods_lua (lua_State *L)
{
  lua_settop(L, 3);
  tk_ann_t *A = tk_ann_peek(L, 1);
  uint64_t k = tk_lua_optunsigned(L, 2, "k", 0);
  uint64_t eps = tk_lua_optunsigned(L, 3, "eps", A->features);
  tk_ann_neighborhoods(L, A, k, eps, 0, 0);
  return 2;
}

static inline int tk_ann_neighbors_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_ann_t *A = tk_ann_peek(L, 1);
  uint64_t knn = tk_lua_optunsigned(L, 3, "knn", 0);
  uint64_t eps = tk_lua_optunsigned(L, 4, "eps", A->features);
  tk_pvec_t *out = tk_pvec_peekopt(L, 5);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_ann_neighbors_by_id(L, A, uid, knn, eps, out);
  } else {
    char *vec = (char *) tk_lua_checkustring(L, 2, "vector");
    tk_ann_neighbors_by_vec(L, A, vec, -1, knn, eps, out);
  }
  return out == NULL ? 1 : 0;
}

static inline int tk_ann_ids_lua (lua_State *L)
{
  lua_settop(L, 1);
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_iumap_keys(L, A->uid_sid);
  return 1;
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
  return 1;
}

static inline int tk_ann_features_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  lua_pushinteger(L, (int64_t) A->features);
  return 1;
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

static inline int tk_ann_shrink_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_ann_shrink(A);
  return 0;
}

static inline void tk_ann_worker (void *dp, int sig)
{
  tk_ann_stage_t stage = (tk_ann_stage_t) sig;
  tk_ann_thread_t *data = (tk_ann_thread_t *) dp;

  switch (stage) {

    case TK_ANN_NEIGHBORHOODS:
      if (data->k == 0)
        return;
      for (uint64_t i = data->ifirst; i <= data->ilast; i ++) {
        tk_pvec_t *hood = data->hoods->a[i];
        int64_t sid = data->sids->a[i];
        tk_ann_populate_neighborhood(data->A, i, sid, tk_ann_sget(data->A, sid), hood, data->sid_idx, data->eps);
        tk_pvec_asc(hood, 0, hood->n);
      }
      break;

  }
}

static luaL_Reg tk_ann_lua_mt_fns[] =
{
  { "add", tk_ann_add_lua },
  { "remove", tk_ann_remove_lua },
  { "get", tk_ann_get_lua },
  { "neighborhoods", tk_ann_neighborhoods_lua },
  { "neighbors", tk_ann_neighbors_lua },
  { "ids", tk_ann_ids_lua },
  { "size", tk_ann_size_lua },
  { "threads", tk_ann_threads_lua },
  { "features", tk_ann_features_lua },
  { "persist", tk_ann_persist_lua },
  { "destroy", tk_ann_destroy_lua },
  { "shrink", tk_ann_shrink_lua },
  { NULL, NULL }
};

static inline void tk_ann_suppress_unused_lua_mt_fns (void)
  { (void) tk_ann_lua_mt_fns; }

static inline void tk_ann_setup_hash_bits_random (
  lua_State *L,
  tk_ann_t *A,
  int Ai,
  uint64_t expected
) {
  uint64_t n_hash_bits = (uint64_t) ceil(log2((double) expected / (double) A->bucket_target));
  n_hash_bits = n_hash_bits > TK_ANN_BITS ? TK_ANN_BITS : n_hash_bits;
  A->hash_bits = tk_ivec_create(L, A->features, 0, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  tk_ivec_fill_indices(A->hash_bits);
  tk_ivec_shuffle(A->hash_bits);
  if (A->hash_bits->n > n_hash_bits)
    A->hash_bits->n = n_hash_bits;
  tk_ivec_shrink(L, A->hash_bits);
}

static inline tk_ann_t *tk_ann_create_base (
  lua_State *L,
  uint64_t features,
  uint64_t bucket_target,
  uint64_t probe_radius,
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
  A->features = features;
  A->buckets = kh_init(tk_ann_buckets);
  A->uid_sid = tk_iumap_create();
  A->sid_uid = tk_iumap_create();
  A->vectors = tk_cvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  A->destroyed = false;
  A->next_sid = 0;
  A->probe_radius = probe_radius;
  A->bucket_target = bucket_target;
  return A;
}

static inline tk_ann_t *tk_ann_create_randomized (
  lua_State *L,
  uint64_t features,
  uint64_t bucket_target,
  uint64_t probe_radius,
  uint64_t expected,
  uint64_t n_threads
) {
  tk_ann_t *A = tk_ann_create_base(L, features, bucket_target, probe_radius, n_threads);
  tk_ann_setup_hash_bits_random(L, A, tk_lua_absindex(L, -1), expected);
  return A;
}

static inline tk_ann_t *tk_ann_load (
  lua_State *L,
  FILE *fh,
  uint64_t n_threads
) {
  // userdata + metatable
  tk_ann_t *A = tk_lua_newuserdata(L, tk_ann_t, TK_ANN_MT, tk_ann_lua_mt_fns, tk_ann_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  memset(A, 0, sizeof(tk_ann_t));
  // core scalars
  tk_lua_fread(L, &A->destroyed, sizeof(bool), 1, fh);
  if (A->destroyed)
    tk_lua_verror(L, 2, "load", "index was destroyed when saved");
  tk_lua_fread(L, &A->next_sid, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &A->bucket_target, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &A->features, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &A->probe_radius, sizeof(uint64_t), 1, fh);
  // hash_bits vector
  uint64_t n_hash = 0;
  tk_lua_fread(L, &n_hash, sizeof(uint64_t), 1, fh);
  A->hash_bits = tk_ivec_create(L, n_hash, 0, 0);
  A->hash_bits->n = n_hash;
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  if (n_hash)
    tk_lua_fread(L, A->hash_bits->a, sizeof(int64_t), n_hash, fh);
  lua_pop(L, 1);
  // buckets map
  A->buckets = kh_init(tk_ann_buckets);
  khint_t nb = 0, k; int absent;
  tk_lua_fread(L, &nb, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nb; i ++) {
    tk_ann_hash_t hkey;
    bool has;
    tk_lua_fread(L, &hkey, sizeof(tk_ann_hash_t), 1, fh);
    tk_lua_fread(L, &has, sizeof(bool),           1, fh);
    k = kh_put(tk_ann_buckets, A->buckets, hkey, &absent);
    if (has) {
      uint64_t plen;
      tk_lua_fread(L, &plen, sizeof(uint64_t), 1, fh);
      tk_ivec_t *plist = tk_ivec_create(L, plen, 0, 0);
      plist->n = plen;
      tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
      if (plen)
        tk_lua_fread(L, plist->a, sizeof(int64_t), plen, fh);
      lua_pop(L, 1);
      kh_val(A->buckets, k) = plist;
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
  // vectors array
  uint64_t vcount = 0;
  tk_lua_fread(L, &vcount, sizeof(uint64_t), 1, fh);
  A->vectors = tk_cvec_create(L, vcount, 0, 0);
  A->vectors->n = vcount;
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  if (vcount)
    tk_lua_fread(L, A->vectors->a, sizeof(double), vcount, fh);
  lua_pop(L, 1);
  // thread pool
  A->threads = tk_malloc(L, n_threads * sizeof(tk_ann_thread_t));
  memset(A->threads, 0, n_threads * sizeof(tk_ann_thread_t));
  A->pool = tk_threads_create(L, n_threads, tk_ann_worker);
  for (unsigned int t = 0; t < n_threads; t ++) {
    tk_ann_thread_t *th = A->threads + t;
    A->pool->threads[t].data = th;
    th->A = A;
  }
  return A;
}


#endif
