#ifndef TK_HBI_H
#define TK_HBI_H

#include <assert.h>
#include <santoku/lua/utils.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/iuset.h>
#include <santoku/threads.h>
#include <santoku/ivec/ext.h>
#include <santoku/cvec/ext.h>

#define TK_HBI_BITS 32
#define tk_hbi_code_t uint32_t

#define TK_HBI_MT "tk_hbi_t"
#define TK_HBI_EPH "tk_hbi_eph"

#define tk_umap_name tk_hbi_buckets
#define tk_umap_key tk_hbi_code_t
#define tk_umap_value tk_ivec_t *
#define tk_umap_peekkey(...) tk_lua_checkunsigned(__VA_ARGS__)
#define tk_umap_peekvalue(...) tk_ivec_peek(__VA_ARGS__)
#define tk_umap_pushkey(...) lua_pushinteger(__VA_ARGS__)
#define tk_umap_pushvalue(L, x) tk_lua_get_ephemeron(L, TK_HBI_EPH, x)
#define tk_umap_eq(a, b) (kh_int_hash_equal(a, b))
#define tk_umap_hash(a) (kh_int_hash_func(a))
#include <santoku/umap/tpl.h>

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
  TK_HBI_MUTUAL_INIT,
  TK_HBI_MUTUAL_FILTER,
  TK_HBI_MIN_REMAP,
  TK_HBI_COLLECT_UIDS,
  TK_HBI_REMAP_UIDS,
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
  tk_iumap_t **hoods_sets;
  tk_ivec_t *uids;
  tk_ivec_t *sids;
  tk_cvec_t *query_vecs;
  uint64_t ifirst, ilast;
  uint64_t k;
  uint64_t eps;
  uint64_t min;
  int64_t *old_to_new;
  tk_iuset_t *local_uids;
  tk_iumap_t *uid_to_idx;
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
  lua_State *L,
  tk_hbi_t *A
) {
  if (A->destroyed)
    return;

  int Ai = 1; // Object is at position 1 when called from Lua
  int64_t *old_to_new = tk_malloc(L, A->next_sid * sizeof(int64_t));
  for (uint64_t i = 0; i < A->next_sid; i ++)
    old_to_new[i] = -1;
  uint64_t new_sid = 0;
  int64_t old_sid;
  tk_umap_foreach_keys(A->sid_uid, old_sid, ({
    old_to_new[old_sid] = (int64_t) new_sid ++;
  }))
  if (new_sid == A->next_sid) {
    free(old_to_new);
    tk_hbi_codes_shrink(A->codes);
    return;
  }
  tk_hbi_code_t *old_codes = A->codes->a;
  tk_hbi_code_t *new_codes = A->codes->a;
  tk_umap_foreach_keys(A->sid_uid, old_sid, ({
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val != old_sid)
      new_codes[new_sid_val] = old_codes[old_sid];
  }))
  A->codes->n = new_sid;
  for (khint_t k = kh_begin(A->buckets); k != kh_end(A->buckets); k ++) {
    if (!kh_exist(A->buckets, k))
      continue;

    tk_ivec_t *posting = tk_hbi_buckets_val(A->buckets, k);
    if (!posting)
      continue;
    for (uint64_t i = 0; i < posting->n; i ++) {
      int64_t old_sid = posting->a[i];
      int64_t new_sid_val = old_to_new[old_sid];
      if (new_sid_val >= 0)
        posting->a[i] = new_sid_val;
    }
    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < posting->n; i ++) {
      if (old_to_new[posting->a[i]] >= 0)
        posting->a[write_pos ++] = posting->a[i];
    }
    posting->n = write_pos;
    tk_ivec_shrink(posting);
  }
  tk_iumap_t *new_uid_sid = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
  tk_iumap_t *new_sid_uid = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);

  int64_t uid;
  tk_umap_foreach(A->uid_sid, uid, old_sid, ({
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val >= 0) {
      int is_new;
      khint_t khi = tk_iumap_put(new_uid_sid, uid, &is_new);
      tk_iumap_setval(new_uid_sid, khi, new_sid_val);
      khi = tk_iumap_put(new_sid_uid, new_sid_val, &is_new);
      tk_iumap_setval(new_sid_uid, khi, uid);
    }
  }))

  tk_lua_del_ephemeron(L, TK_HBI_EPH, Ai, A->uid_sid);
  tk_lua_del_ephemeron(L, TK_HBI_EPH, Ai, A->sid_uid);
  tk_iumap_destroy(A->uid_sid);
  tk_iumap_destroy(A->sid_uid);
  A->uid_sid = new_uid_sid;
  A->sid_uid = new_sid_uid;
  A->next_sid = new_sid;
  tk_hbi_codes_shrink(A->codes);

  free(old_to_new);
}

static inline void tk_hbi_destroy (
  tk_hbi_t *A
) {
  if (A->destroyed)
    return;
  A->destroyed = true;
  free(A->threads);
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
      int64_t sid0 = tk_iumap_val(A->uid_sid, khi);
      khi = tk_iumap_get(A->sid_uid, sid0);
      if (khi != tk_iumap_end(A->sid_uid))
        tk_iumap_del(A->sid_uid, khi);
    }
    tk_iumap_setval(A->uid_sid, khi, sid);
    khi = tk_iumap_put(A->sid_uid, sid, &kha);
    tk_iumap_setval(A->sid_uid, khi, uid);
    return sid;
  } else {
    khi = tk_iumap_get(A->uid_sid, uid);
    if (khi == tk_iumap_end(A->uid_sid))
      return -1;
    else
      return tk_iumap_val(A->uid_sid, khi);
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
    return tk_iumap_val(A->sid_uid, khi);
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

static inline double tk_hbi_similarity (
  tk_hbi_t *A,
  int64_t uid0,
  int64_t uid1
) {
  char *v0 = tk_hbi_get(A, uid0);
  char *v1 = tk_hbi_get(A, uid1);
  if (!v0 || !v1)
    return 0.0;
  uint64_t hamming_dist = tk_cvec_bits_hamming((const uint8_t *)v0, (const uint8_t *)v1, A->features);
  return 1.0 - ((double)hamming_dist / (double)A->features);
}

static inline double tk_hbi_distance (
  tk_hbi_t *A,
  int64_t uid0,
  int64_t uid1
) {
  char *v0 = tk_hbi_get(A, uid0);
  char *v1 = tk_hbi_get(A, uid1);
  if (!v0 || !v1)
    return 1.0;
  uint64_t hamming_dist = tk_cvec_bits_hamming((const uint8_t *)v0, (const uint8_t *)v1, A->features);
  return (double)hamming_dist / (double)A->features;
}

static inline tk_hbi_code_t tk_hbi_pack (const char *V, uint64_t features)
{
  tk_hbi_code_t h = 0;
  size_t nbytes = TK_CVEC_BITS_BYTES(features);
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
  tk_lua_fwrite(L, (char *) &A->destroyed, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &A->next_sid, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &A->features, sizeof(uint64_t), 1, fh);
  khint_t nb = A->buckets ? kh_size(A->buckets) : 0;
  tk_lua_fwrite(L, (char *) &nb, sizeof(khint_t), 1, fh);
  tk_hbi_code_t code;
  tk_ivec_t *list;
  tk_umap_foreach(A->buckets, code, list, ({
    tk_lua_fwrite(L, (char *) &code, sizeof(tk_hbi_code_t), 1, fh);
    bool has = list && list->n;
    tk_lua_fwrite(L, (char *) &has, sizeof(bool), 1, fh);
    if (has) {
      uint64_t len = list->n;
      tk_lua_fwrite(L, (char *) &len, sizeof(uint64_t), 1, fh);
      tk_lua_fwrite(L, (char *) list->a, sizeof(int64_t), len, fh);
    }
  }))
  tk_iumap_persist(L, A->uid_sid, fh);
  tk_iumap_persist(L, A->sid_uid, fh);
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
  int64_t sid = tk_iumap_val(A->uid_sid, khi);
  tk_iumap_del(A->uid_sid, khi);
  khi = tk_iumap_get(A->sid_uid, sid);
  if (khi == tk_iumap_end(A->sid_uid))
    return;
  tk_iumap_del(A->sid_uid, khi);
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
    tk_hbi_code_t h = tk_hbi_pack(data + i * TK_CVEC_BITS_BYTES(A->features), A->features);
    khi = tk_hbi_buckets_put(A->buckets, h, &kha);
    if (kha) {
      bucket = tk_ivec_create(L, 0, 0, 0);
      tk_hbi_buckets_setval(A->buckets, khi, bucket);
      tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
      lua_pop(L, 1);
    } else {
      bucket = tk_hbi_buckets_val(A->buckets, khi);
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

static inline void tk_hbi_keep (
  lua_State *L,
  tk_hbi_t *A,
  tk_ivec_t *ids
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "keep", "can't keep in a destroyed index");
    return;
  }

  tk_iuset_t *keep_set = tk_iuset_from_ivec(0, ids);

  tk_iuset_t *to_remove_set = tk_iuset_create(0, 0);
  tk_iuset_union_iumap(to_remove_set, A->uid_sid);
  tk_iuset_subtract(to_remove_set, keep_set);

  int64_t uid;
  tk_umap_foreach_keys(to_remove_set, uid, ({
    tk_hbi_uid_remove(A, uid);
  }));
  tk_iuset_destroy(keep_set);
  tk_iuset_destroy(to_remove_set);
}

static inline bool tk_hbi_probe_bucket (
  tk_hbi_t *A,
  tk_hbi_code_t h,
  tk_pvec_t *out,
  uint64_t knn,
  int64_t sid0,
  int r
) {
  khint_t khi = tk_hbi_buckets_get(A->buckets, h);
  if (khi != tk_hbi_buckets_end(A->buckets)) {
    tk_ivec_t *bucket = tk_hbi_buckets_val(A->buckets, khi);
    for (uint64_t bi = 0; bi < bucket->n; bi ++) {
      int64_t sid1 = bucket->a[bi];
      if (sid1 == sid0)
        continue;
      int64_t uid1 = tk_hbi_sid_uid(A, sid1);
      if (uid1 < 0)
        continue;
      tk_pvec_push(out, tk_pair(uid1, (int64_t) r));
      if (out->n >= knn)
        return true;
    }
  }
  return false;
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
  khint_t khi = tk_hbi_buckets_get(A->buckets, h);
  if (khi != tk_hbi_buckets_end(A->buckets)) {
    tk_ivec_t *bucket = tk_hbi_buckets_val(A->buckets, khi);
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
      int64_t idx = tk_iumap_val(sid_idx, khi);
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
  tk_ivec_t *uids,
  uint64_t min,
  int64_t **old_to_newp
) {
  if (A->destroyed)
    return;

  tk_ivec_t *sids = tk_ivec_create(L, uids->n, 0, 0);
  for (uint64_t i = 0; i < uids->n; i ++)
    sids->a[i] = tk_hbi_uid_sid(A, uids->a[i], false);
  tk_iumap_t *sid_idx = tk_iumap_from_ivec(0, sids);
  tk_iumap_t **hoods_sets = tk_malloc(L, uids->n * sizeof(tk_iumap_t *));
  for (uint64_t i = 0; i < uids->n; i ++)
    hoods_sets[i] = NULL;
  for (uint64_t i = 0; i < uids->n; i ++)
    hoods_sets[i] = tk_iumap_create(0, 0);
  for (uint64_t i = 0; i < A->pool->n_threads; i ++) {
    tk_hbi_thread_t *data = A->threads + i;
    data->uids = uids;
    data->sids = sids;
    data->hoods = hoods;
    data->hoods_sets = hoods_sets;
    data->sid_idx = sid_idx;
    tk_thread_range(i, A->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }
  tk_threads_signal(A->pool, TK_HBI_MUTUAL_INIT, 0);
  tk_threads_signal(A->pool, TK_HBI_MUTUAL_FILTER, 0);
  if (min > 0) {
    int64_t *old_to_new = tk_malloc(L, uids->n * sizeof(int64_t));
    int64_t keeper_count = 0;
    for (uint64_t i = 0; i < uids->n; i ++) {
      if (hoods->a[i]->n >= min) {
        old_to_new[i] = keeper_count ++;
      } else {
        old_to_new[i] = -1;
      }
    }
    if (keeper_count < (int64_t) uids->n) {
      for (uint64_t i = 0; i < A->pool->n_threads; i ++) {
        tk_hbi_thread_t *data = A->threads + i;
        data->old_to_new = old_to_new;
        data->min = min;
        tk_thread_range(i, A->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
      }
      tk_threads_signal(A->pool, TK_HBI_MIN_REMAP, 0);
      tk_ivec_t *new_uids = tk_ivec_create(L, (uint64_t) keeper_count, 0, 0);
      tk_hbi_hoods_t *new_hoods = tk_hbi_hoods_create(L, (uint64_t) keeper_count, 0, 0);
      new_hoods->n = (uint64_t) keeper_count;
      for (uint64_t i = 0; i < uids->n; i ++) {
        if (old_to_new[i] >= 0) {
          new_uids->a[old_to_new[i]] = uids->a[i];
          new_hoods->a[old_to_new[i]] = hoods->a[i];
        }
      }
      int64_t *old_uids_data = uids->a;
      tk_hbi_hood_t *old_hoods_data = hoods->a;
      uids->a = new_uids->a;
      uids->n = (uint64_t) keeper_count;
      uids->m = (uint64_t) keeper_count;
      hoods->a = new_hoods->a;
      hoods->n = (uint64_t) keeper_count;
      hoods->m = (uint64_t) keeper_count;
      new_uids->a = old_uids_data;
      new_hoods->a = old_hoods_data;

      lua_remove(L, -2);
      lua_remove(L, -1);
    }

    if (old_to_newp) {
      *old_to_newp = old_to_new;
    } else {
      free(old_to_new);
    }
  }
  tk_iumap_destroy(sid_idx);
  for (uint64_t i = 0; i < uids->n; i ++)
    if (hoods_sets[i])
      tk_iumap_destroy(hoods_sets[i]);
  free(hoods_sets);
  lua_remove(L, -1);
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
  tk_ivec_t *sids = tk_iumap_values(L, A->uid_sid);
  tk_ivec_asc(sids, 0, sids->n);
  tk_ivec_t *uids = tk_ivec_create(L, sids->n, 0, 0);
  for (uint64_t i = 0; i < sids->n; i ++)
    uids->a[i] = tk_hbi_sid_uid(A, sids->a[i]);

  tk_iumap_t *sid_idx = tk_iumap_create(0, 0);
  for (uint64_t i = 0; i < sids->n; i ++) {
    khi = tk_iumap_put(sid_idx, sids->a[i], &kha);
    tk_iumap_setval(sid_idx, khi, (int64_t) i);
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
    data->query_vecs = NULL;
    data->hoods = hoods;
    data->sid_idx = sid_idx;
    data->k = k;
    data->eps = eps;
    tk_thread_range(i, A->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }
  tk_threads_signal(A->pool, TK_HBI_NEIGHBORHOODS, 0);
  if (mutual && k)
    tk_threads_signal(A->pool, TK_HBI_MUTUAL, 0);
  if (sid_idx) tk_iumap_destroy(sid_idx);

  // Min filtering
  if (min > 0) {
    int64_t keeper_count = 0;
    for (uint64_t i = 0; i < uids->n; i ++)
      if (hoods->a[i]->n >= min)
        keeper_count ++;
    if (keeper_count == (int64_t) uids->n)
      goto cleanup;
    int64_t *old_to_new = tk_malloc(L, uids->n * sizeof(int64_t));
    int64_t new_idx = 0;
    for (uint64_t i = 0; i < uids->n; i ++) {
      if (hoods->a[i]->n >= min) {
        old_to_new[i] = new_idx ++;
      } else {
        old_to_new[i] = -1;
      }
    }
    for (uint64_t i = 0; i < A->pool->n_threads; i ++) {
      tk_hbi_thread_t *data = A->threads + i;
      data->old_to_new = old_to_new;
      data->min = min;
      tk_thread_range(i, A->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
    }
    tk_threads_signal(A->pool, TK_HBI_MIN_REMAP, 0);
    tk_ivec_t *new_uids = tk_ivec_create(L, (uint64_t) keeper_count, 0, 0);
    tk_hbi_hoods_t *new_hoods = tk_hbi_hoods_create(L, (uint64_t) keeper_count, 0, 0);
    new_hoods->n = (uint64_t) keeper_count;
    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < uids->n; i ++) {
      if (hoods->a[i]->n >= min) {
        new_uids->a[write_pos] = uids->a[i];
        new_hoods->a[write_pos] = hoods->a[i];
        write_pos ++;
      }
    }
    int64_t *old_uids_data = uids->a;
    tk_hbi_hood_t *old_hoods_data = hoods->a;
    uids->a = new_uids->a;
    uids->n = (uint64_t) keeper_count;
    uids->m = (uint64_t) keeper_count;
    hoods->a = new_hoods->a;
    hoods->n = (uint64_t) keeper_count;
    hoods->m = (uint64_t) keeper_count;
    new_uids->a = old_uids_data;
    new_hoods->a = old_hoods_data;
    lua_remove(L, -2); // remove new_uids
    lua_remove(L, -1); // remove new_hoods
    free(old_to_new);
  }

cleanup:
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
  if (sids) lua_remove(L, -3); // sids
}

static inline void tk_hbi_neighborhoods_by_ids (
  lua_State *L,
  tk_hbi_t *A,
  tk_ivec_t *query_ids,
  uint64_t k,
  uint64_t eps,
  uint64_t min,
  bool mutual,
  tk_hbi_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "neighborhoods_by_ids", "can't query a destroyed index");
    return;
  }

  tk_ivec_t *sids = tk_ivec_create(L, query_ids->n, 0, 0);
  tk_ivec_t *uids = tk_ivec_create(L, query_ids->n, 0, 0);
  tk_ivec_copy(uids, query_ids, 0, (int64_t) query_ids->n, 0);
  for (uint64_t i = 0; i < uids->n; i ++)
    sids->a[i] = tk_hbi_uid_sid(A, uids->a[i], false);

  int kha;
  khint_t khi;
  tk_iumap_t *sid_idx = tk_iumap_create(0, 0);
  for (uint64_t i = 0; i < sids->n; i ++) {
    khi = tk_iumap_put(sid_idx, sids->a[i], &kha);
    tk_iumap_setval(sid_idx, khi, (int64_t) i);
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
    data->query_vecs = NULL;
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

  // Min filtering
  if (min > 0) {
    int64_t keeper_count = 0;
    for (uint64_t i = 0; i < uids->n; i ++)
      if (hoods->a[i]->n >= min)
        keeper_count ++;
    if (keeper_count == (int64_t) uids->n)
      goto cleanup;
    int64_t *old_to_new = tk_malloc(L, uids->n * sizeof(int64_t));
    int64_t new_idx = 0;
    for (uint64_t i = 0; i < uids->n; i ++)
      if (hoods->a[i]->n >= min)
        old_to_new[i] = new_idx ++;
      else
        old_to_new[i] = -1;
    for (uint64_t i = 0; i < A->pool->n_threads; i ++) {
      tk_hbi_thread_t *data = A->threads + i;
      data->old_to_new = old_to_new;
      data->min = min;
      tk_thread_range(i, A->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
    }
    tk_threads_signal(A->pool, TK_HBI_MIN_REMAP, 0);
    tk_ivec_t *new_uids = tk_ivec_create(L, (uint64_t) keeper_count, 0, 0); // sids uids hoods nuids
    tk_hbi_hoods_t *new_hoods = tk_hbi_hoods_create(L, (uint64_t) keeper_count, 0, 0); // sids uids hoods nuids nhoods
    new_hoods->n = (uint64_t) keeper_count;
    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < uids->n; i ++) {
      if (hoods->a[i]->n >= min) {
        new_uids->a[write_pos] = uids->a[i];
        new_hoods->a[write_pos] = hoods->a[i];
        write_pos ++;
      }
    }
    int64_t *old_uids_data = uids->a;
    tk_hbi_hood_t *old_hoods_data = hoods->a;
    uids->a = new_uids->a;
    uids->n = (uint64_t) keeper_count;
    uids->m = (uint64_t) keeper_count;
    hoods->a = new_hoods->a;
    hoods->n = (uint64_t) keeper_count;
    hoods->m = (uint64_t) keeper_count;
    new_uids->a = old_uids_data;
    new_hoods->a = old_hoods_data;
    lua_pop(L, 2); // sids uids hoods
    free(old_to_new);
  }

cleanup:
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
  lua_remove(L, -3); // sids
}

static inline void tk_hbi_neighborhoods_by_vecs (
  lua_State *L,
  tk_hbi_t *A,
  tk_cvec_t *query_vecs,
  uint64_t k,
  uint64_t eps,
  uint64_t min,
  tk_hbi_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "neighborhoods_by_vecs", "can't query a destroyed index");
    return;
  }
  uint64_t vec_bytes = TK_CVEC_BITS_BYTES(A->features);
  uint64_t n_queries = query_vecs->n / vec_bytes;

  tk_hbi_hoods_t *hoods = tk_hbi_hoods_create(L, n_queries, 0, 0); // hoods
  hoods->n = n_queries;
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_pvec_create(L, k, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_HBI_EPH, -1, -1);
    lua_pop(L, 1);
  }

  // Pass NULL for sid_idx to make search store UIDs directly
  for (uint64_t i = 0; i < A->pool->n_threads; i ++) {
    tk_hbi_thread_t *data = A->threads + i;
    data->uids = NULL;
    data->sids = NULL;
    data->query_vecs = query_vecs;
    data->hoods = hoods;
    data->sid_idx = NULL;
    data->k = k;
    data->eps = eps;
    tk_thread_range(i, A->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }

  tk_threads_signal(A->pool, TK_HBI_NEIGHBORHOODS, 0);
  for (uint64_t i = 0; i < A->pool->n_threads; i ++) {
    A->threads[i].local_uids = tk_iuset_create(0, 0);
    tk_thread_range(i, A->pool->n_threads, hoods->n, &A->threads[i].ifirst, &A->threads[i].ilast);
  }
  tk_threads_signal(A->pool, TK_HBI_COLLECT_UIDS, 0);
  tk_iumap_t *uid_to_idx = tk_iumap_create(0, 0);
  int64_t next_idx = 0;
  int ret;
  for (uint64_t t = 0; t < A->pool->n_threads; t ++) {
    tk_iuset_t *local = A->threads[t].local_uids;
    int64_t uid;
    tk_umap_foreach_keys(local, uid, ({
      khint_t k = tk_iumap_put(uid_to_idx, uid, &ret);
      if (ret)
        tk_iumap_setval(uid_to_idx, k, next_idx++);
    }));
    tk_iuset_destroy(local);
  }
  tk_ivec_t *uids = tk_ivec_create(L, (uint64_t)next_idx, 0, 0); // hoods uids
  uids->n = (uint64_t)next_idx;
  for (khint_t k = tk_iumap_begin(uid_to_idx); k != tk_iumap_end(uid_to_idx); k++) {
    if (tk_iumap_exist(uid_to_idx, k)) {
      int64_t uid = tk_iumap_key(uid_to_idx, k);
      int64_t idx = tk_iumap_val(uid_to_idx, k);
      uids->a[idx] = uid;
    }
  }
  lua_insert(L, -2); // uids hoods

  for (uint64_t i = 0; i < A->pool->n_threads; i ++) {
    A->threads[i].uid_to_idx = uid_to_idx;
    tk_thread_range(i, A->pool->n_threads, hoods->n, &A->threads[i].ifirst, &A->threads[i].ilast);
  }
  tk_threads_signal(A->pool, TK_HBI_REMAP_UIDS, 0);
  tk_iumap_destroy(uid_to_idx);

  // Min filtering
  if (min > 0) {
    int64_t keeper_count = 0;
    for (uint64_t i = 0; i < hoods->n; i ++)
      if (hoods->a[i]->n >= min)
        keeper_count ++;
    if (keeper_count == (int64_t) hoods->n)
      goto cleanup;
    int kha;
    tk_iuset_t *kept_uids = tk_iuset_create(0, 0);
    for (uint64_t i = 0; i < hoods->n; i ++) {
      if (hoods->a[i]->n >= min) {
        tk_pvec_t *hood = hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j ++) {
          int64_t idx = hood->a[j].i;
          int64_t uid = uids->a[idx];
          tk_iuset_put(kept_uids, uid, &kha);
        }
      }
    }
    tk_iumap_t *idx_remap = tk_iumap_create(0, 0);
    tk_ivec_t *new_uids = tk_ivec_create(L, (uint64_t) tk_iuset_size(kept_uids), 0, 0); // uids hoods new_uids
    int64_t new_idx = 0;
    int64_t uid;
    tk_umap_foreach_keys(kept_uids, uid, ({
      new_uids->a[new_idx] = uid;
      for (uint64_t old_idx = 0; old_idx < uids->n; old_idx ++) {
        if (uids->a[old_idx] == uid) {
          int ret;
          khint_t k = tk_iumap_put(idx_remap, (int64_t) old_idx, &ret);
          tk_iumap_setval(idx_remap, k, new_idx);
          break;
        }
      }
      new_idx ++;
    }));
    new_uids->n = (uint64_t) new_idx;
    tk_iuset_destroy(kept_uids);
    tk_hbi_hoods_t *new_hoods = tk_hbi_hoods_create(L, (uint64_t) keeper_count, 0, 0); // uids hoods new_uids new_hoods
    new_hoods->n = (uint64_t) keeper_count;
    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < hoods->n; i ++) {
      if (hoods->a[i]->n >= min) {
        tk_pvec_t *hood = hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j ++) {
          int64_t old_idx = hood->a[j].i;
          khint_t k = tk_iumap_get(idx_remap, old_idx);
          if (k != tk_iumap_end(idx_remap)) {
            hood->a[j].i = tk_iumap_val(idx_remap, k);
          }
        }
        new_hoods->a[write_pos ++] = hood;
      }
    }
    int64_t *old_uids_data = uids->a;
    tk_hbi_hood_t *old_hoods_data = hoods->a;
    uids->a = new_uids->a;
    uids->n = new_uids->n;
    uids->m = new_uids->n;
    hoods->a = new_hoods->a;
    hoods->n = (uint64_t) keeper_count;
    hoods->m = (uint64_t) keeper_count;
    new_uids->a = old_uids_data;
    new_hoods->a = old_hoods_data;
    lua_pop(L, 2); // uids hoods
    tk_iumap_destroy(idx_remap);
  }

cleanup:
  if (hoodsp)
    *hoodsp = hoods;
  if (uidsp)
    *uidsp = uids;
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
  tk_hbi_code_t h0 = tk_hbi_pack(vec, A->features);
  int pos[TK_HBI_BITS];
  int nbits = (int) A->features;
  if (tk_hbi_probe_bucket(A, h0, out, knn, sid0, 0))
    return out;
  for (int r = 1; r <= (int) eps && r <= nbits; r ++) {
    for (int i = 0; i < r; i ++)
      pos[i] = i;
    while (true) {
      tk_hbi_code_t mask = 0;
      for (int i = 0; i < r; i ++)
        mask |= (1U << pos[i]);
      if (tk_hbi_probe_bucket(A, h0 ^ mask, out, knn, sid0, r))
        return out;
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
  tk_pvec_asc(out, 0, out->n);
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

  char *data;
  tk_cvec_t *data_cvec = tk_cvec_peekopt(L, 2);
  if (data_cvec) {
    data = data_cvec->a;
  } else {
    data = (char *) tk_lua_checkustring(L, 2, "data");
  }

  if (lua_type(L, 3) == LUA_TNUMBER) {
    int64_t s = (int64_t) tk_lua_checkunsigned(L, 3, "base_id");
    uint64_t n = tk_lua_optunsigned(L, 4, "n_nodes", 1);
    tk_ivec_t *ids = tk_ivec_create(L, n, 0, 0);
    tk_ivec_fill_indices(ids);
    tk_ivec_add(ids, s, 0, ids->n);
    tk_hbi_add(L, A, Ai, ids, data);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 3, "ids");
    tk_hbi_add(L, A, Ai, ids, data);
  }
  return 0;
}

static inline int tk_hbi_remove_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t id = tk_lua_checkinteger(L, 2, "id");
    tk_hbi_remove(L, A, id);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
    for (uint64_t i = 0; i < ids->n; i ++) {
      tk_hbi_uid_remove(A, ids->a[i]);
    }
  }
  return 0;
}

static inline int tk_hbi_keep_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t id = tk_lua_checkinteger(L, 2, "id");
    tk_ivec_t *ids = tk_ivec_create(L, 1, 0, 0);
    ids->a[0] = id;
    ids->n = 1;
    tk_hbi_keep(L, A, ids);
    lua_pop(L, 1);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
    tk_hbi_keep(L, A, ids);
  }
  return 0;
}

static inline int tk_hbi_get_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  size_t bytes_per_vec = TK_CVEC_BITS_BYTES(A->features);
  int64_t uid = -1;
  tk_ivec_t *uids = NULL;
  tk_cvec_t *out = tk_cvec_peekopt(L, 3);
  out = out == NULL ? tk_cvec_create(L, 0, 0, 0) : out; // out
  uint64_t dest_sample = tk_lua_optunsigned(L, 4, "dest_sample", 0);
  uint64_t dest_stride = tk_lua_optunsigned(L, 5, "dest_stride", 0);

  uint64_t n_samples = 0;
  if (lua_type(L, 2) == LUA_TNUMBER) {
    n_samples = 1;
    uid = tk_lua_checkinteger(L, 2, "id");
  } else {
    uids = tk_ivec_peek(L, 2, "uids");
    n_samples = uids->n;
  }

  uint64_t row_stride_bits;
  bool use_packed;

  if (dest_stride > 0) {
    use_packed = true;
    row_stride_bits = dest_stride;
  } else {
    use_packed = false;
    row_stride_bits = bytes_per_vec * CHAR_BIT;
  }

  uint64_t total_bytes;
  if (use_packed) {
    uint64_t total_bits = dest_sample * row_stride_bits + n_samples * A->features;
    if (dest_stride > 0 && n_samples > 0) {
      total_bits = (dest_sample + n_samples) * row_stride_bits;
    }
    total_bytes = TK_CVEC_BITS_BYTES(total_bits);
  } else {
    total_bytes = (dest_sample + n_samples) * bytes_per_vec;
  }

  tk_cvec_ensure(out, total_bytes);
  uint8_t *dest_data = (uint8_t *)out->a;
  if (dest_sample == 0) {
    out->n = total_bytes;
    memset(dest_data, 0, total_bytes);
  } else {
    uint64_t old_size = out->n;
    out->n = total_bytes;
    if (total_bytes > old_size) {
      memset(dest_data + old_size, 0, total_bytes - old_size);
    }
  }

  // Process samples
  if (lua_type(L, 2) == LUA_TNUMBER) {
    char *data = tk_hbi_get(A, uid);
    if (data != NULL) {
      if (use_packed) {
        uint64_t bit_offset = dest_sample * row_stride_bits;
        uint64_t byte_offset = bit_offset / CHAR_BIT;
        uint8_t bit_shift = bit_offset % CHAR_BIT;
        if (bit_shift == 0) {
          memcpy(dest_data + byte_offset, data, bytes_per_vec);
        } else {
          uint8_t *src = (uint8_t *)data;
          for (uint64_t i = 0; i < bytes_per_vec; i++) {
            uint8_t byte = src[i];
            dest_data[byte_offset + i] |= byte << bit_shift;
            if (byte_offset + i + 1 < total_bytes) {
              dest_data[byte_offset + i + 1] |= byte >> (CHAR_BIT - bit_shift);
            }
          }
        }
      } else {
        memcpy(dest_data + dest_sample * bytes_per_vec, data, bytes_per_vec);
      }
    }
  } else {
    for (uint64_t i = 0; i < uids->n; i++) {
      uid = uids->a[i];
      char *data = tk_hbi_get(A, uid);
      if (use_packed) {
        uint64_t bit_offset = (dest_sample + i) * row_stride_bits;
        uint64_t byte_offset = bit_offset / CHAR_BIT;
        uint8_t bit_shift = bit_offset % CHAR_BIT;
        if (data != NULL) {
          if (bit_shift == 0) {
            memcpy(dest_data + byte_offset, data, bytes_per_vec);
          } else {
            uint8_t *src = (uint8_t *)data;
            for (uint64_t j = 0; j < bytes_per_vec; j++) {
              uint8_t byte = src[j];
              dest_data[byte_offset + j] |= byte << bit_shift;
              if (byte_offset + j + 1 < total_bytes) {
                dest_data[byte_offset + j + 1] |= byte >> (CHAR_BIT - bit_shift);
              }
            }
          }
        }
      } else {
        uint64_t offset = (dest_sample + i) * bytes_per_vec;
        if (data != NULL)
          memcpy(dest_data + offset, data, bytes_per_vec);
      }
    }
  }

  return 1;
}

static inline int tk_hbi_neighborhoods_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  uint64_t k = tk_lua_optunsigned(L, 2, "k", 0);
  uint64_t eps = tk_lua_optunsigned(L, 3, "eps", 3);
  uint64_t min = tk_lua_optunsigned(L, 4, "min", 0);
  bool mutual = tk_lua_optboolean(L, 5, "mutual", false);
  tk_hbi_neighborhoods(L, A, k, eps, min, mutual, NULL, NULL);
  return 2;
}

static inline int tk_hbi_neighborhoods_by_ids_lua (lua_State *L)
{
  lua_settop(L, 6);
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  tk_ivec_t *query_ids = tk_ivec_peek(L, 2, "ids");
  uint64_t k = tk_lua_optunsigned(L, 3, "k", 0);
  uint64_t eps = tk_lua_optunsigned(L, 4, "eps", 3);
  uint64_t min = tk_lua_optunsigned(L, 5, "min", 0);
  bool mutual = tk_lua_optboolean(L, 6, "mutual", false);
  int64_t write_pos = 0;
  for (int64_t i = 0; i < (int64_t) query_ids->n; i ++) {
    int64_t uid = query_ids->a[i];
    khint_t kh = tk_iumap_get(A->uid_sid, uid);
    if (kh != tk_iumap_end(A->uid_sid)) {
      query_ids->a[write_pos ++] = uid;
    }
  }
  query_ids->n = (uint64_t) write_pos;
  tk_hbi_hoods_t *hoods;
  tk_hbi_neighborhoods_by_ids(L, A, query_ids, k, eps, min, mutual, &hoods, &query_ids);
  return 2;
}

static inline int tk_hbi_neighborhoods_by_vecs_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  tk_cvec_t *query_vecs = tk_cvec_peek(L, 2, "vectors");
  uint64_t k = tk_lua_optunsigned(L, 3, "k", 0);
  uint64_t eps = tk_lua_optunsigned(L, 4, "eps", 3);
  uint64_t min = tk_lua_optunsigned(L, 5, "min", 0);

  tk_hbi_neighborhoods_by_vecs(L, A, query_vecs, k, eps, min, NULL, NULL);
  return 2;
}

static inline int tk_hbi_similarity_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  int64_t uid0 = tk_lua_checkinteger(L, 2, "uid0");
  int64_t uid1 = tk_lua_checkinteger(L, 3, "uid1");
  lua_pushnumber(L, tk_hbi_similarity(A, uid0, uid1));
  return 1;
}

static inline int tk_hbi_distance_lua (lua_State *L)
{
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  int64_t uid0 = tk_lua_checkinteger(L, 2, "uid0");
  int64_t uid1 = tk_lua_checkinteger(L, 3, "uid1");
  lua_pushnumber(L, tk_hbi_distance(A, uid0, uid1));
  return 1;
}

static inline int tk_hbi_neighbors_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_hbi_t *A = tk_hbi_peek(L, 1);
  uint64_t knn = tk_lua_checkunsigned(L, 3, "knn");
  uint64_t eps = tk_lua_optunsigned(L, 4, "eps", 0);
  tk_pvec_t *out = tk_pvec_peek(L, 5,  "out");
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_hbi_neighbors_by_id(A, uid, knn, eps, out);
  } else {
    char *vec;
    tk_cvec_t *vec_cvec = tk_cvec_peekopt(L, 2);
    if (vec_cvec) {
      vec = vec_cvec->a;
    } else {
      vec = (char *) tk_lua_checkustring(L, 2, "vector");
    }
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
  tk_hbi_shrink(L, A);
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
        if (data->query_vecs) {
          uint64_t vec_bytes = TK_CVEC_BITS_BYTES(data->A->features);
          char *vec = (char *)(data->query_vecs->a + i * vec_bytes);
          tk_hbi_populate_neighborhood(data->A, i, -1, vec, hood, data->sid_idx, data->k, data->eps);
        } else {
          int64_t sid = data->sids->a[i];
          tk_hbi_populate_neighborhood(data->A, i, sid, tk_hbi_sget(data->A, sid), hood, data->sid_idx, data->k, data->eps);
        }
      }
      break;

    case TK_HBI_MUTUAL: {
      assert(false);
    }

    case TK_HBI_MUTUAL_INIT: {
      int kha;
      khint_t khi;
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {
        tk_pvec_t *uhood = data->hoods->a[i];
        tk_iumap_t *uset = data->hoods_sets[i];
        for (uint64_t j = 0; j < uhood->n; j ++) {
          khi = tk_iumap_put(uset, uhood->a[j].i, &kha);
          tk_iumap_setval(uset, khi, uhood->a[j].p);
        }
      }
      break;
    }

    case TK_HBI_MUTUAL_FILTER: {
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {
        tk_pvec_t *uhood = data->hoods->a[i];
        uint64_t orig_n = uhood->n;
        assert(uhood->m >= orig_n);
        if (orig_n == 0) {
          uhood->n = 0;
          uhood->m = 0;
          continue;
        }
        uint64_t left = 0;
        uint64_t right = orig_n - 1;
        khint_t khi;
        while (left <= right) {
          int64_t iv = uhood->a[left].i;
          int64_t d = uhood->a[left].p;
          assert(iv >= 0 && (uint64_t) iv < data->hoods->n);
          tk_iumap_t *vset = data->hoods_sets[iv];
          khi = tk_iumap_get(vset, i);
          if (khi != tk_iumap_end(vset)) {
            // Edge is mutual - keep it and update distance to minimum
            int64_t d0 = tk_iumap_val(vset, khi);
            if (d0 < d)
              uhood->a[left].p = d0;
            left ++;
          } else {
            // Edge is not mutual - move to end
            if (left != right) {
              tk_pair_t tmp = uhood->a[left];
              uhood->a[left] = uhood->a[right];
              uhood->a[right] = tmp;
            }
            if (right == 0)
              break;
            right--;
          }
        }
        uhood->n = left;
        uhood->m = orig_n;
        assert(uhood->n <= uhood->m);

        for (uint64_t qi = uhood->n; qi < uhood->m; qi ++) {
          int64_t iv = uhood->a[qi].i;
          int64_t d_forward = uhood->a[qi].p;
          int64_t d_reverse = d_forward;  // fallback
          tk_iumap_t *vset = data->hoods_sets[iv];
          khi = tk_iumap_get(vset, i);
          if (khi != tk_iumap_end(vset)) {
            d_reverse = tk_iumap_val(vset, khi);
          } else {
            int64_t usid = data->sids->a[i];
            int64_t vsid = data->sids->a[iv];
            tk_hbi_code_t ucode = data->A->codes->a[usid];
            tk_hbi_code_t vcode = data->A->codes->a[vsid];
            tk_hbi_code_t xor_result = ucode ^ vcode;
            d_reverse = (int64_t) __builtin_popcount(xor_result);
          }
          uhood->a[qi].p = (d_forward < d_reverse) ? d_forward : d_reverse;
        }
        tk_pvec_asc(uhood, 0, uhood->n);
        tk_pvec_asc(uhood, uhood->n, uhood->m);
      }
      break;
    }

    case TK_HBI_MIN_REMAP: {
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {
        if (data->hoods->a[i]->n >= data->min) {
          tk_pvec_t *hood = data->hoods->a[i];
          uint64_t mutual_write_pos = 0;
          uint64_t non_mutual_write_pos = 0;
          for (uint64_t j = 0; j < hood->n; j ++) {
            int64_t old_neighbor_idx = hood->a[j].i;
            int64_t new_neighbor_idx = data->old_to_new[old_neighbor_idx];
            if (new_neighbor_idx >= 0)
              hood->a[mutual_write_pos ++] = tk_pair(new_neighbor_idx, hood->a[j].p);
          }
          for (uint64_t j = hood->n; j < hood->m; j ++) {
            int64_t old_neighbor_idx = hood->a[j].i;
            int64_t new_neighbor_idx = data->old_to_new[old_neighbor_idx];
            if (new_neighbor_idx >= 0)
              hood->a[mutual_write_pos + non_mutual_write_pos ++] = tk_pair(new_neighbor_idx, hood->a[j].p);
          }
          hood->n = mutual_write_pos;
          hood->m = mutual_write_pos + non_mutual_write_pos;
        }
      }
      break;
    }

    case TK_HBI_COLLECT_UIDS: {
      int kha;
      for (uint64_t i = data->ifirst; i <= data->ilast; i ++) {
        tk_pvec_t *hood = data->hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j ++) {
          int64_t uid = hood->a[j].i;
          tk_iuset_put(data->local_uids, uid, &kha);
        }
      }
      break;
    }

    case TK_HBI_REMAP_UIDS: {
      for (uint64_t i = data->ifirst; i <= data->ilast; i ++) {
        tk_pvec_t *hood = data->hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j ++) {
          int64_t uid = hood->a[j].i;
          khint_t k = tk_iumap_get(data->uid_to_idx, uid);
          if (k != tk_iumap_end(data->uid_to_idx)) {
            hood->a[j].i = tk_iumap_val(data->uid_to_idx, k);
          }
        }
      }
      break;
    }

  }
}

static luaL_Reg tk_hbi_lua_mt_fns[] =
{
  { "add", tk_hbi_add_lua },
  { "remove", tk_hbi_remove_lua },
  { "keep", tk_hbi_keep_lua },
  { "get", tk_hbi_get_lua },
  { "neighborhoods", tk_hbi_neighborhoods_lua },
  { "neighborhoods_by_ids", tk_hbi_neighborhoods_by_ids_lua },
  { "neighborhoods_by_vecs", tk_hbi_neighborhoods_by_vecs_lua },
  { "neighbors", tk_hbi_neighbors_lua },
  { "similarity", tk_hbi_similarity_lua },
  { "distance", tk_hbi_distance_lua },
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
    tk_lua_verror(L, 3, "create", "features", "must be <= " tk_pp_str(TK_HBI_BITS));
  tk_hbi_t *A = tk_lua_newuserdata(L, tk_hbi_t, TK_HBI_MT, tk_hbi_lua_mt_fns, tk_hbi_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  A->threads = tk_malloc(L, n_threads * sizeof(tk_hbi_thread_t));
  memset(A->threads, 0, n_threads * sizeof(tk_hbi_thread_t));
  A->pool = tk_threads_create(L, n_threads, tk_hbi_worker);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_hbi_thread_t *data = A->threads + i;
    A->pool->threads[i].data = data;
    data->A = A;
  }
  A->features = features;
  A->buckets = tk_hbi_buckets_create(L, 0);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
  A->uid_sid = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
  A->sid_uid = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
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
  tk_hbi_t *A = tk_lua_newuserdata(L, tk_hbi_t, TK_HBI_MT, tk_hbi_lua_mt_fns, tk_hbi_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  memset(A, 0, sizeof(tk_hbi_t));
  tk_lua_fread(L, &A->destroyed, sizeof(bool), 1, fh);
  if (A->destroyed)
    tk_lua_verror(L, 2, "load", "index was destroyed when saved");
  tk_lua_fread(L, &A->next_sid,  sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &A->features,  sizeof(uint64_t), 1, fh);
  A->buckets = tk_hbi_buckets_create(L, 0);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
  khint_t nb = 0, k; int absent;
  tk_lua_fread(L, &nb, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nb; i ++) {
    tk_hbi_code_t code;
    bool has;
    tk_lua_fread(L, &code, sizeof(tk_hbi_code_t), 1, fh);
    tk_lua_fread(L, &has, sizeof(bool), 1, fh);
    k = tk_hbi_buckets_put(A->buckets, code, &absent);
    if (has) {
      uint64_t len;
      tk_lua_fread(L, &len, sizeof(uint64_t), 1, fh);
      tk_ivec_t *list = tk_ivec_create(L, len, 0, 0);
      list->n = len;
      tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
      if (len)
        tk_lua_fread(L, list->a, sizeof(int64_t), len, fh);
      lua_pop(L, 1);
      tk_hbi_buckets_setval(A->buckets, k, list);
    } else {
      tk_hbi_buckets_setval(A->buckets, k, NULL);
    }
  }
  A->uid_sid = tk_iumap_load(L, fh);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
  A->sid_uid = tk_iumap_load(L, fh);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
  uint64_t cnum = 0;
  tk_lua_fread(L, &cnum, sizeof(uint64_t), 1, fh);
  A->codes = tk_hbi_codes_create(L, cnum, 0, 0);
  A->codes->n = cnum;
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  if (cnum)
    tk_lua_fread(L, A->codes->a, sizeof(tk_hbi_code_t), cnum, fh);
  lua_pop(L, 1);
  A->threads = tk_malloc(L, n_threads * sizeof(tk_hbi_thread_t));
  memset(A->threads, 0, n_threads * sizeof(tk_hbi_thread_t));
  A->pool = tk_threads_create(L, n_threads, tk_hbi_worker);
  tk_lua_add_ephemeron(L, TK_HBI_EPH, Ai, -1);
  lua_pop(L, 1);
  for (unsigned int t = 0; t < n_threads; t ++) {
    tk_hbi_thread_t *th = A->threads + t;
    A->pool->threads[t].data = th;
    th->A = A;
  }
  return A;
}

#endif
