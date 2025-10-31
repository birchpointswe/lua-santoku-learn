#ifndef TK_ANN_H
#define TK_ANN_H

#include <santoku/iuset.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/pvec.h>
#include <santoku/iumap.h>
#include <santoku/ivec/ext.h>
#include <santoku/cvec/ext.h>
#include <omp.h>

#define TK_ANN_BITS 32
#define tk_ann_hash_t uint32_t

#define TK_ANN_MT "tk_ann_t"
#define TK_ANN_EPH "tk_ann_eph"

#define tk_umap_name tk_ann_buckets
#define tk_umap_key tk_ann_hash_t
#define tk_umap_value tk_ivec_t *
#define tk_umap_peekkey(...) tk_lua_checkunsigned(__VA_ARGS__)
#define tk_umap_peekvalue(...) tk_ivec_peek(__VA_ARGS__)
#define tk_umap_pushkey(...) lua_pushinteger(__VA_ARGS__)
#define tk_umap_pushvalue(L, x) tk_lua_get_ephemeron(L, TK_ANN_EPH, x)
#define tk_umap_eq(a, b) (kh_int_hash_equal(a, b))
#define tk_umap_hash(a) (kh_int_hash_func(a))
#include <santoku/umap/tpl.h>

typedef tk_pvec_t * tk_ann_hood_t;
#define tk_vec_name tk_ann_hoods
#define tk_vec_base tk_ann_hood_t
#define tk_vec_pushbase(L, x) tk_lua_get_ephemeron(L, TK_ANN_EPH, x)
#define tk_vec_peekbase(L, i) tk_pvec_peek(L, i, "hood")
#define tk_vec_limited
#include <santoku/vec/tpl.h>

typedef enum {
  TK_ANN_FIND,
  TK_ANN_REPLACE
} tk_ann_uid_mode_t;

typedef struct tk_ann_s {
  bool destroyed;
  uint64_t next_sid;
  uint64_t bucket_target;
  uint64_t features;
  tk_ivec_t *hash_bits;
  tk_ann_buckets_t *buckets;
  tk_iumap_t *uid_sid;
  tk_ivec_t *sid_to_uid;
  tk_cvec_t *vectors;
} tk_ann_t;

static inline tk_ann_t *tk_ann_peek (lua_State *L, int i)
{
  return (tk_ann_t *) luaL_checkudata(L, i, TK_ANN_MT);
}

static inline tk_ann_t *tk_ann_peekopt (lua_State *L, int i)
{
  return (tk_ann_t *) tk_lua_testuserdata(L, i, TK_ANN_MT);
}

static inline void tk_ann_shrink (
  lua_State *L,
  tk_ann_t *A,
  int Ai
) {
  if (A->destroyed)
    return;
  if (A->next_sid > SIZE_MAX / sizeof(int64_t))
    tk_error(L, "ann_shrink: allocation size overflow", ENOMEM);
  int64_t *old_to_new = tk_malloc(L, A->next_sid * sizeof(int64_t));
  for (uint64_t i = 0; i < A->next_sid; i ++)
    old_to_new[i] = -1;
  int64_t old_sid;
  uint64_t new_sid = 0;
  for (int64_t s = 0; s < (int64_t)A->next_sid; s++) {
    if (A->sid_to_uid->a[s] >= 0) {
      old_to_new[s] = (int64_t) new_sid++;
    }
  }
  if (new_sid == A->next_sid) {
    free(old_to_new);
    tk_cvec_shrink(A->vectors);
    if (A->hash_bits) tk_ivec_shrink(A->hash_bits);
    return;
  }
  size_t bytes_per_vec = TK_CVEC_BITS_BYTES(A->features);
  char *old_vectors = A->vectors->a;
  char *new_vectors = A->vectors->a;
  for (int64_t old_sid = 0; old_sid < (int64_t)A->next_sid; old_sid++) {
    if (A->sid_to_uid->a[old_sid] < 0) continue;
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val != old_sid) {
      memmove(new_vectors + (uint64_t) new_sid_val * bytes_per_vec,
              old_vectors + (uint64_t) old_sid * bytes_per_vec,
              bytes_per_vec);
    }
  }
  A->vectors->n = new_sid * bytes_per_vec;
  tk_ivec_t *posting;
  tk_umap_foreach_values(A->buckets, posting, ({
    if (!posting) continue;
    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < posting->n; i ++) {
      int64_t old_sid = posting->a[i];
      int64_t new_sid_val = old_to_new[old_sid];
      if (new_sid_val >= 0) {
        posting->a[write_pos ++] = new_sid_val;
      }
    }
    posting->n = write_pos;
    tk_ivec_shrink(posting);
  }))
  tk_iumap_t *new_uid_sid = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  int64_t uid;
  tk_umap_foreach(A->uid_sid, uid, old_sid, ({
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val >= 0) {
      int is_new;
      uint32_t i = tk_iumap_put(new_uid_sid, uid, &is_new);
      tk_iumap_setval(new_uid_sid, i, new_sid_val);
    }
  }))
  tk_lua_del_ephemeron(L, TK_ANN_EPH, Ai, A->uid_sid);
  tk_iumap_destroy(A->uid_sid);
  A->uid_sid = new_uid_sid;
  tk_ivec_t *new_sid_to_uid = tk_ivec_create(L, new_sid, 0, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  new_sid_to_uid->n = new_sid;
  for (int64_t old_sid = 0; old_sid < (int64_t)A->next_sid; old_sid++) {
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val >= 0) {
      new_sid_to_uid->a[new_sid_val] = A->sid_to_uid->a[old_sid];
    }
  }
  tk_lua_del_ephemeron(L, TK_ANN_EPH, Ai, A->sid_to_uid);
  A->sid_to_uid = new_sid_to_uid;
  A->next_sid = new_sid;
  tk_cvec_shrink(A->vectors);
  if (A->hash_bits)
    tk_ivec_shrink(A->hash_bits);
  free(old_to_new);
}

static inline void tk_ann_destroy (
  tk_ann_t *A
) {
  if (A->destroyed)
    return;
  A->destroyed = true;
}

static inline tk_ivec_t *tk_ann_ids (lua_State *L, tk_ann_t *A)
{
  return tk_iumap_keys(L, A->uid_sid);
}

static inline int tk_ann_ids_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_iumap_keys(L, A->uid_sid);
  return 1;
}

static inline int64_t tk_ann_uid_sid (
  tk_ann_t *A,
  int64_t uid,
  tk_ann_uid_mode_t mode
) {
  int kha;
  khint_t khi;
  if (mode == TK_ANN_FIND) {
    khi = tk_iumap_get(A->uid_sid, uid);
    if (khi == tk_iumap_end(A->uid_sid))
      return -1;
    else
      return tk_iumap_val(A->uid_sid, khi);
  } else {
    khi = tk_iumap_get(A->uid_sid, uid);
    if (khi != tk_iumap_end(A->uid_sid)) {
      int64_t old_sid = tk_iumap_val(A->uid_sid, khi);
      tk_iumap_del(A->uid_sid, khi);
      if (old_sid >= 0 && old_sid < (int64_t)A->sid_to_uid->n)
        A->sid_to_uid->a[old_sid] = -1;
    }
    int64_t sid = (int64_t) (A->next_sid ++);
    khi = tk_iumap_put(A->uid_sid, uid, &kha);
    tk_iumap_setval(A->uid_sid, khi, sid);

    tk_ivec_ensure(A->sid_to_uid, A->next_sid);
    if (A->sid_to_uid->n < A->next_sid) {
      for (uint64_t i = A->sid_to_uid->n; i < A->next_sid; i++)
        A->sid_to_uid->a[i] = -1;
      A->sid_to_uid->n = A->next_sid;
    }
    A->sid_to_uid->a[sid] = uid;
    return sid;
  }
}

static inline char *tk_ann_sget (
  tk_ann_t *A,
  int64_t sid
) {
  return A->vectors->a + (uint64_t) sid * TK_CVEC_BITS_BYTES(A->features);
}

static inline char *tk_ann_get (
  tk_ann_t *A,
  int64_t uid
) {
  int64_t sid = tk_ann_uid_sid(A, uid, TK_ANN_FIND);
  if (sid < 0)
    return NULL;
  return tk_ann_sget(A, sid);
}

static inline double tk_ann_similarity (
  tk_ann_t *A,
  int64_t uid0,
  int64_t uid1
) {
  char *v0 = tk_ann_get(A, uid0);
  char *v1 = tk_ann_get(A, uid1);
  if (!v0 || !v1)
    return 0.0;
  uint64_t hamming_dist = tk_cvec_bits_hamming_serial((const uint8_t *)v0, (const uint8_t *)v1, A->features);
  return 1.0 - ((double)hamming_dist / (double)A->features);
}

static inline double tk_ann_distance (
  tk_ann_t *A,
  int64_t uid0,
  int64_t uid1
) {
  char *v0 = tk_ann_get(A, uid0);
  char *v1 = tk_ann_get(A, uid1);
  if (!v0 || !v1)
    return 1.0;
  uint64_t hamming_dist = tk_cvec_bits_hamming_serial((const uint8_t *)v0, (const uint8_t *)v1, A->features);
  return (double)hamming_dist / (double)A->features;
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
    uint64_t chunk = TK_CVEC_BITS_BYTE((uint64_t) b);
    uint64_t pos = TK_CVEC_BITS_BIT((uint64_t) b);
    const unsigned char *ud = (const unsigned char *) data;
    if (ud[chunk] & (unsigned char)(1u << pos))
      h |= ((uint32_t)1 << i);
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
  tk_lua_fwrite(L, (char *) &A->destroyed, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &A->next_sid, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &A->bucket_target, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &A->features, sizeof(uint64_t), 1, fh);
  tk_ivec_persist(L, A->hash_bits, fh);
  khint_t nb = A->buckets ? tk_ann_buckets_size(A->buckets) : 0;
  tk_lua_fwrite(L, (char *) &nb, sizeof(khint_t), 1, fh);
  tk_ann_hash_t hkey;
  tk_ivec_t *plist;
  tk_umap_foreach(A->buckets, hkey, plist, ({
    tk_lua_fwrite(L, (char *) &hkey, sizeof(tk_ann_hash_t), 1, fh);
    bool has = plist && plist->n;
    tk_lua_fwrite(L, (char *) &has, sizeof(bool), 1, fh);
    if (has) {
      uint64_t plen = plist->n;
      tk_lua_fwrite(L, (char *) &plen, sizeof(uint64_t), 1, fh);
      tk_lua_fwrite(L, (char *) plist->a, sizeof(int64_t), plen, fh);
    }
  }))
  tk_iumap_persist(L, A->uid_sid, fh);
  tk_ivec_persist(L, A->sid_to_uid, fh);
  tk_cvec_persist(L, A->vectors, fh);
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
  int64_t sid = tk_iumap_val(A->uid_sid, khi);
  tk_iumap_del(A->uid_sid, khi);

  if (sid >= 0 && sid < (int64_t)A->sid_to_uid->n)
    A->sid_to_uid->a[sid] = -1;
}

static inline int64_t tk_ann_sid_uid (
  tk_ann_t *A,
  int64_t sid
) {
  if (sid < 0 || sid >= (int64_t)A->sid_to_uid->n)
    return -1;
  return A->sid_to_uid->a[sid];
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
  size_t bytes_per_vec = TK_CVEC_BITS_BYTES(A->features);
  tk_cvec_ensure(A->vectors, A->vectors->n + ids->n * bytes_per_vec);
  int kha;
  khint_t khi;
  for (uint64_t i = 0; i < ids->n; i ++) {
    int64_t sid = tk_ann_uid_sid(A, ids->a[i], TK_ANN_REPLACE);
    tk_ivec_t *bucket;
    tk_ann_hash_t h = tk_ann_hash(A, data + i * TK_CVEC_BITS_BYTES(A->features));
    khi = kh_put(tk_ann_buckets, A->buckets, h, &kha);
    if (kha) {
      bucket = kh_value(A->buckets, khi) = tk_ivec_create(L, 0, 0, 0);
      tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
      lua_pop(L, 1);
    } else {
      bucket = kh_value(A->buckets, khi);
    }
    if (tk_ivec_push(bucket, sid) != 0) {
      tk_lua_verror(L, 2, "add", "allocation failed during indexing");
      return;
    }
    tk_cvec_t datavec = {
      .n = TK_CVEC_BITS_BYTES(A->features),
      .m = TK_CVEC_BITS_BYTES(A->features),
      .a = (char *) data + i * TK_CVEC_BITS_BYTES(A->features) };
    tk_cvec_copy(A->vectors, &datavec, 0, (int64_t) datavec.n, sid * (int64_t) TK_CVEC_BITS_BYTES(A->features));
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

static inline void tk_ann_keep (
  lua_State *L,
  tk_ann_t *A,
  tk_ivec_t *ids
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "keep", "can't keep in a destroyed index");
    return;
  }

  tk_iuset_t *keep_set = tk_iuset_from_ivec(0, ids);
  if (!keep_set) {
    tk_lua_verror(L, 2, "keep", "allocation failed");
    return;
  }

  tk_iuset_t *to_remove_set = tk_iuset_create(0, tk_iumap_size(A->uid_sid));
  tk_iuset_union_iumap(to_remove_set, A->uid_sid);
  tk_iuset_subtract(to_remove_set, keep_set);

  int64_t uid;
  tk_umap_foreach_keys(to_remove_set, uid, ({
    tk_ann_uid_remove(A, uid);
  }));

  tk_iuset_destroy(keep_set);
  tk_iuset_destroy(to_remove_set);
}

static inline void tk_ann_probe_bucket_for_uid (
  tk_ann_t *A,
  tk_ann_hash_t h,
  const unsigned char *v,
  uint64_t ftr,
  int64_t skip_sid,
  uint64_t k,
  uint64_t eps_min,
  uint64_t eps_max,
  int r,
  tk_pvec_t *out
) {
  khint_t khi = kh_get(tk_ann_buckets, A->buckets, h);
  if (khi == kh_end(A->buckets))
    return;
  tk_ivec_t *bucket = kh_value(A->buckets, khi);
  for (uint64_t bi = 0; bi < bucket->n; bi ++) {
    int64_t sid1 = bucket->a[bi];
    if (sid1 == skip_sid)
      continue;
    int64_t uid1 = tk_ann_sid_uid(A, sid1);
    if (uid1 < 0)
      continue;
    const unsigned char *p1 = (const unsigned char *) tk_ann_sget(A, sid1);
    uint64_t dist = tk_cvec_bits_hamming_serial((const uint8_t *)v, (const uint8_t *)p1, ftr);
    if (dist < eps_min || dist > eps_max)
      continue;
    if (k)
      tk_pvec_hmax(out, k, tk_pair(uid1, (int64_t) dist));
    else
      tk_pvec_push(out, tk_pair(uid1, (int64_t) dist));
  }
}

static inline void tk_ann_probe_bucket (
  tk_ann_t *A,
  tk_ivec_t *sid_to_pos,
  tk_ann_hash_t h,
  const unsigned char *v,
  uint64_t ftr,
  int64_t skip_sid,
  uint64_t k,
  uint64_t eps_min,
  uint64_t eps_max,
  int r,
  tk_pvec_t *out
) {
  khint_t khi = kh_get(tk_ann_buckets, A->buckets, h);
  if (khi == kh_end(A->buckets))
    return;
  tk_ivec_t *bucket = kh_value(A->buckets, khi);
  for (uint64_t bi = 0; bi < bucket->n; bi ++) {
    int64_t sid1 = bucket->a[bi];
    if (sid1 == skip_sid)
      continue;

    if (sid1 < 0 || sid1 >= (int64_t)sid_to_pos->n)
      continue;
    int64_t id = sid_to_pos->a[sid1];
    if (id < 0)
      continue;

    const unsigned char *p1 = (const unsigned char *) tk_ann_sget(A, sid1);
    uint64_t dist = tk_cvec_bits_hamming_serial((const uint8_t *)v, (const uint8_t *)p1, ftr);
    if (dist < eps_min || dist > eps_max)
      continue;

    if (k)
      tk_pvec_hmax(out, k, tk_pair(id, (int64_t) dist));
    else
      tk_pvec_push(out, tk_pair(id, (int64_t) dist));
  }
}

static inline void tk_ann_populate_neighborhood (
  tk_ann_t *A,
  tk_ivec_t *sid_to_pos,
  uint64_t i,
  int64_t sid,
  char *V,
  tk_pvec_t *hood,
  uint64_t knn,
  uint64_t probe_radius,
  uint64_t eps_min,
  uint64_t eps_max
) {
  const unsigned char *v_uc = (const unsigned char *) V;
  tk_ann_hash_t h = tk_ann_hash(A, V);
  tk_ann_probe_bucket(A, sid_to_pos, h, v_uc, A->features, sid, knn, eps_min, eps_max, 0, hood);
  int pos[TK_ANN_BITS];
  for (int r = 1; r <= (int) probe_radius && r <= TK_ANN_BITS; r ++) {
    for (int i = 0; i < r; i ++)
      pos[i] = i;
    while (true) {
      tk_ann_hash_t mask = 0;
      for (int j = 0; j < r; j ++)
        mask |= (1U << pos[j]);
      tk_ann_probe_bucket(A, sid_to_pos, h ^ mask, v_uc, A->features, sid, knn, eps_min, eps_max, r, hood);
      int k;
      for (k = r - 1; k >= 0; k--) {
        if (pos[k] != k + TK_ANN_BITS - r) {
          pos[k]++;
          for (int j = k + 1; j < r; j ++)
            pos[j] = pos[j - 1] + 1;
          break;
        }
      }
      if (k < 0)
        break;
    }
  }
}

static inline void tk_ann_prepare_universe_map (
  lua_State *L,
  tk_ann_t *A,
  tk_ivec_t **uids_out,
  tk_ivec_t **sid_to_pos_out
) {
  tk_ivec_t *uids = tk_ivec_create(L, 0, 0, 0);
  tk_ivec_t *sid_to_pos = tk_ivec_create(NULL, A->next_sid, 0, 0);
  sid_to_pos->n = A->next_sid;
  uint64_t active_idx = 0;
  for (uint64_t sid = 0; sid < A->next_sid; sid++) {
    int64_t uid = A->sid_to_uid->a[sid];
    if (uid >= 0) {
      sid_to_pos->a[sid] = (int64_t)active_idx;
      tk_ivec_push(uids, uid);
      active_idx++;
    } else {
      sid_to_pos->a[sid] = -1;
    }
  }
  *uids_out = uids;
  *sid_to_pos_out = sid_to_pos;
}

static inline void tk_ann_neighborhoods (
  lua_State *L,
  tk_ann_t *A,
  uint64_t k,
  uint64_t probe_radius,
  uint64_t eps_min,
  uint64_t eps_max,
  tk_ann_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "neighborhoods", "can't query a destroyed index");
    return;
  }

  tk_ivec_t *uids, *sid_to_pos;
  tk_ann_prepare_universe_map(L, A, &uids, &sid_to_pos);

  tk_ann_hoods_t *hoods = tk_ann_hoods_create(L, uids->n, 0, 0);
  int hoods_stack_idx = lua_gettop(L);
  hoods->n = uids->n;
  for (uint64_t i = 0; i < hoods->n; i ++) {
    tk_pvec_t *hood = tk_pvec_create(L, k, 0, 0);
    hood->n = 0;
    tk_lua_add_ephemeron(L, TK_ANN_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
    hoods->a[i] = hood;
  }

  #pragma omp parallel for
  for (uint64_t i = 0; i < hoods->n; i ++) {
    tk_pvec_t *hood = hoods->a[i];
    tk_pvec_clear(hood);
    int64_t uid = uids->a[i];
    int64_t sid = tk_ann_uid_sid(A, uid, TK_ANN_FIND);
    char *vec = tk_ann_sget(A, sid);
    tk_ann_populate_neighborhood(A, sid_to_pos, i, sid, vec, hood, k, probe_radius, eps_min, eps_max);
    tk_pvec_asc(hood, 0, hood->n);
  }

  tk_ivec_destroy(sid_to_pos);

  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
}

static inline void tk_ann_neighborhoods_by_ids (
  lua_State *L,
  tk_ann_t *A,
  tk_ivec_t *query_ids,
  uint64_t k,
  uint64_t probe_radius,
  uint64_t eps_min,
  uint64_t eps_max,
  tk_ann_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (A->destroyed) {
    tk_lua_verror(L, 2, "neighborhoods_by_ids", "can't query a destroyed index");
    return;
  }

  tk_ivec_t *all_uids, *sid_to_pos;
  tk_ann_prepare_universe_map(L, A, &all_uids, &sid_to_pos);

  tk_ann_hoods_t *hoods = tk_ann_hoods_create(L, query_ids->n, 0, 0);
  int hoods_stack_idx = lua_gettop(L);
  hoods->n = query_ids->n;
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_pvec_create(L, k, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_ANN_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
  }

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < hoods->n; i ++) {
    tk_pvec_t *hood = hoods->a[i];
    int64_t uid = query_ids->a[i];
    int64_t sid = tk_ann_uid_sid(A, uid, TK_ANN_FIND);
    if (sid < 0)
      continue;
    tk_ann_populate_neighborhood(A, sid_to_pos, i, sid, tk_ann_sget(A, sid), hood, k, probe_radius, eps_min, eps_max);
    tk_pvec_asc(hood, 0, hood->n);
  }

  tk_ivec_destroy(sid_to_pos);

  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = all_uids;
  lua_remove(L, -2);
}

static inline void tk_ann_neighborhoods_by_vecs (
  lua_State *L,
  tk_ann_t *A,
  tk_cvec_t *query_vecs,
  uint64_t k,
  uint64_t probe_radius,
  uint64_t eps_min,
  uint64_t eps_max,
  tk_ann_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (A->destroyed)
    return;

  tk_ivec_t *all_uids, *sid_to_pos;
  tk_ann_prepare_universe_map(L, A, &all_uids, &sid_to_pos);

  uint64_t vec_bytes = TK_CVEC_BITS_BYTES(A->features);
  uint64_t n_queries = query_vecs->n / vec_bytes;
  tk_ann_hoods_t *hoods = tk_ann_hoods_create(L, n_queries, 0, 0);
  int hoods_stack_idx = lua_gettop(L);
  hoods->n = n_queries;
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_pvec_create(L, k, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_ANN_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
  }

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < hoods->n; i ++) {
    tk_pvec_t *hood = hoods->a[i];
    char *vec = query_vecs->a + i * vec_bytes;
    int64_t sid = -1;
    tk_ann_populate_neighborhood(A, sid_to_pos, i, sid, vec, hood, k, probe_radius, eps_min, eps_max);
    tk_pvec_asc(hood, 0, hood->n);
  }

  tk_ivec_destroy(sid_to_pos);

  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = all_uids;
}

static inline tk_pvec_t *tk_ann_neighbors_by_vec (
  tk_ann_t *A,
  char *vec,
  int64_t sid0,
  uint64_t knn,
  uint64_t probe_radius,
  uint64_t eps_min,
  uint64_t eps_max,
  tk_pvec_t *out
) {
  if (A->destroyed)
    return NULL;
  tk_pvec_clear(out);

  const tk_ann_hash_t h0 = tk_ann_hash(A, vec);
  const unsigned char *v = (const unsigned char *) vec;
  const uint64_t ftr = A->features;
  int pos[TK_ANN_BITS];
  for (int r = 0; r <= (int) probe_radius && r <= TK_ANN_BITS; r ++) {
    for (int i = 0; i < r; i ++)
      pos[i] = i;
    while (true) {
      tk_ann_hash_t mask = 0;
      for (int i = 0; i < r; i ++)
        mask |= (1U << pos[i]);
      tk_ann_probe_bucket_for_uid(A, h0 ^ mask, v, ftr, sid0, knn, eps_min, eps_max, r, out);
      int j;
      for (j = r - 1; j >= 0; j--) {
        if (pos[j] != j + TK_ANN_BITS - r) {
          pos[j]++;
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
  tk_ann_t *A,
  int64_t uid,
  uint64_t knn,
  uint64_t probe_radius,
  uint64_t eps_min,
  uint64_t eps_max,
  tk_pvec_t *out
) {
  int64_t sid0 = tk_ann_uid_sid(A, uid, TK_ANN_FIND);
  if (sid0 < 0) {
    tk_pvec_clear(out);
    return out;
  }
  return tk_ann_neighbors_by_vec(A, tk_ann_get(A, uid), sid0, knn, probe_radius, eps_min, eps_max, out);
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
    tk_ann_add(L, A, Ai, ids, data);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 3, "ids");
    tk_ann_add(L, A, Ai, ids, data);
  }
  return 0;
}

static inline int tk_ann_remove_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t id = tk_lua_checkinteger(L, 2, "id");
    tk_ann_remove(L, A, id);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
    for (uint64_t i = 0; i < ids->n; i ++) {
      tk_ann_uid_remove(A, ids->a[i]);
    }
  }
  return 0;
}

static inline int tk_ann_keep_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t id = tk_lua_checkinteger(L, 2, "id");
    tk_ivec_t *ids = tk_ivec_create(L, 1, 0, 0);
    ids->a[0] = id;
    ids->n = 1;
    tk_ann_keep(L, A, ids);
    lua_pop(L, 1);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
    tk_ann_keep(L, A, ids);
  }

  return 0;
}

static inline tk_cvec_t* tk_ann_get_many (lua_State *L, tk_ann_t *A, tk_ivec_t *uids)
{
  size_t bytes_per_vec = TK_CVEC_BITS_BYTES(A->features);
  tk_cvec_t *out = tk_cvec_create(L, 0, 0, 0);
  uint64_t total_bytes = uids->n * bytes_per_vec;
  tk_cvec_ensure(out, total_bytes);
  out->n = total_bytes;
  memset(out->a, 0, total_bytes);

  for (uint64_t i = 0; i < uids->n; i++) {
    char *data = tk_ann_get(A, uids->a[i]);
    if (data != NULL)
      memcpy((uint8_t*)out->a + i * bytes_per_vec, data, bytes_per_vec);
  }

  return out;
}

static inline int tk_ann_get_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_ann_t *A = tk_ann_peek(L, 1);
  size_t bytes_per_vec = TK_CVEC_BITS_BYTES(A->features);
  int64_t uid = -1;
  tk_ivec_t *uids = NULL;
  tk_cvec_t *out = tk_cvec_peekopt(L, 3);
  out = out == NULL ? tk_cvec_create(L, 0, 0, 0) : out;
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
    if (dest_stride > 0 && n_samples > 0)
      total_bits = (dest_sample + n_samples) * row_stride_bits;
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

  if (lua_type(L, 2) == LUA_TNUMBER) {
    char *data = tk_ann_get(A, uid);
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
      char *data = tk_ann_get(A, uid);

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

static inline int tk_ann_neighborhoods_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_ann_t *A = tk_ann_peek(L, 1);
  uint64_t k = tk_lua_optunsigned(L, 2, "k", 0);
  uint64_t probe_radius = tk_lua_optunsigned(L, 3, "probe_radius", 2);
  uint64_t eps_min = tk_lua_optunsigned(L, 4, "eps_min", 0);
  uint64_t eps_max = tk_lua_optunsigned(L, 5, "eps_max", (int64_t)A->features);
  tk_ann_neighborhoods(L, A, k, probe_radius, eps_min, eps_max, 0, 0);
  return 2;
}

static inline int tk_ann_neighborhoods_by_ids_lua (lua_State *L)
{
  lua_settop(L, 6);
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_ivec_t *query_ids = tk_ivec_peek(L, 2, "ids");
  uint64_t k = tk_lua_optunsigned(L, 3, "k", 0);
  uint64_t probe_radius = tk_lua_optunsigned(L, 4, "probe_radius", 3);
  uint64_t eps_min = tk_lua_optunsigned(L, 5, "eps_min", 0);
  uint64_t eps_max = tk_lua_optunsigned(L, 6, "eps_max", A->features);
  int64_t write_pos = 0;
  for (int64_t i = 0; i < (int64_t) query_ids->n; i ++) {
    int64_t uid = query_ids->a[i];
    khint_t kh = tk_iumap_get(A->uid_sid, uid);
    if (kh != tk_iumap_end(A->uid_sid)) {
      query_ids->a[write_pos ++] = uid;
    }
  }
  query_ids->n = (uint64_t) write_pos;
  tk_ann_hoods_t *hoods;
  tk_ann_neighborhoods_by_ids(L, A, query_ids, k, probe_radius, eps_min, eps_max, &hoods, &query_ids);
  return 2;
}

static inline int tk_ann_neighborhoods_by_vecs_lua (lua_State *L)
{
  lua_settop(L, 6);
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_cvec_t *query_vecs = tk_cvec_peek(L, 2, "vectors");
  uint64_t k = tk_lua_optunsigned(L, 3, "k", 0);
  uint64_t probe_radius = tk_lua_optunsigned(L, 4, "probe_radius", 3);
  uint64_t eps_min = tk_lua_optunsigned(L, 5, "eps_min", 0);
  uint64_t eps_max = tk_lua_optunsigned(L, 6, "eps_max", A->features);
  tk_ann_neighborhoods_by_vecs(L, A, query_vecs, k, probe_radius, eps_min, eps_max, NULL, NULL);
  return 2;
}

static inline int tk_ann_similarity_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  int64_t uid0 = tk_lua_checkinteger(L, 2, "uid0");
  int64_t uid1 = tk_lua_checkinteger(L, 3, "uid1");
  lua_pushnumber(L, tk_ann_similarity(A, uid0, uid1));
  return 1;
}

static inline int tk_ann_distance_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  int64_t uid0 = tk_lua_checkinteger(L, 2, "uid0");
  int64_t uid1 = tk_lua_checkinteger(L, 3, "uid1");
  lua_pushnumber(L, tk_ann_distance(A, uid0, uid1));
  return 1;
}

static inline int tk_ann_neighbors_lua (lua_State *L)
{
  lua_settop(L, 7);
  tk_ann_t *A = tk_ann_peek(L, 1);
  uint64_t knn = tk_lua_checkunsigned(L, 3, "knn");
  uint64_t probe_radius = tk_lua_optunsigned(L, 4, "probe_radius", 3);
  uint64_t eps_min = tk_lua_optunsigned(L, 5, "eps_min", 0);
  uint64_t eps_max = tk_lua_optunsigned(L, 6, "eps_max", (int64_t)A->features);
  tk_pvec_t *out = tk_pvec_peek(L, 7, "out");
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_ann_neighbors_by_id(A, uid, knn, probe_radius, eps_min, eps_max, out);
  } else {
    char *vec;
    tk_cvec_t *vec_cvec = tk_cvec_peekopt(L, 2);
    if (vec_cvec) {
      vec = vec_cvec->a;
    } else {
      vec = (char *) tk_lua_checkustring(L, 2, "vector");
    }
    tk_ann_neighbors_by_vec(A, vec, -1, knn, probe_radius, eps_min, eps_max, out);
  }
  return 0;
}

static inline int tk_ann_size_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  lua_pushinteger(L, (int64_t) tk_ann_size(A));
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
  tk_ann_shrink(L, A, 1);
  return 0;
}

static luaL_Reg tk_ann_lua_mt_fns[] =
{
  { "add", tk_ann_add_lua },
  { "remove", tk_ann_remove_lua },
  { "keep", tk_ann_keep_lua },
  { "get", tk_ann_get_lua },
  { "neighborhoods", tk_ann_neighborhoods_lua },
  { "neighborhoods_by_ids", tk_ann_neighborhoods_by_ids_lua },
  { "neighborhoods_by_vecs", tk_ann_neighborhoods_by_vecs_lua },
  { "neighbors", tk_ann_neighbors_lua },
  { "similarity", tk_ann_similarity_lua },
  { "distance", tk_ann_distance_lua },
  { "size", tk_ann_size_lua },
  { "features", tk_ann_features_lua },
  { "persist", tk_ann_persist_lua },
  { "destroy", tk_ann_destroy_lua },
  { "shrink", tk_ann_shrink_lua },
  { "ids", tk_ann_ids_lua },
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
  if (expected < A->bucket_target)
    expected = A->bucket_target;
  uint64_t n_hash_bits = (uint64_t) ceil(log2((double) expected / (double) A->bucket_target));
  n_hash_bits = n_hash_bits > TK_ANN_BITS ? TK_ANN_BITS : n_hash_bits;
  A->hash_bits = tk_ivec_create(L, A->features, 0, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  tk_ivec_fill_indices(A->hash_bits);
  tk_ivec_shuffle(A->hash_bits);
  if (A->hash_bits->n > n_hash_bits)
    A->hash_bits->n = n_hash_bits;
  tk_ivec_shrink(A->hash_bits);
  size_t expected_buckets = expected / A->bucket_target * 2;
  if (expected_buckets > 0)
    kh_resize(tk_ann_buckets, A->buckets, expected_buckets);
}

static inline tk_ann_t *tk_ann_create_base (
  lua_State *L,
  uint64_t features,
  uint64_t bucket_target
) {
  tk_ann_t *A = tk_lua_newuserdata(L, tk_ann_t, TK_ANN_MT, tk_ann_lua_mt_fns, tk_ann_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  A->features = features;
  A->buckets = tk_ann_buckets_create(L, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  A->uid_sid = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  A->sid_to_uid = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  A->vectors = tk_cvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  A->destroyed = false;
  A->next_sid = 0;
  A->bucket_target = bucket_target;
  return A;
}

static inline tk_ann_t *tk_ann_create_randomized (
  lua_State *L,
  uint64_t features,
  uint64_t bucket_target,
  uint64_t expected
) {
  tk_ann_t *A = tk_ann_create_base(L, features, bucket_target);
  tk_ann_setup_hash_bits_random(L, A, tk_lua_absindex(L, -1), expected);
  return A;
}

static inline tk_ann_t *tk_ann_load (
  lua_State *L,
  FILE *fh
) {
  tk_ann_t *A = tk_lua_newuserdata(L, tk_ann_t, TK_ANN_MT, tk_ann_lua_mt_fns, tk_ann_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  memset(A, 0, sizeof(tk_ann_t));
  tk_lua_fread(L, &A->destroyed, sizeof(bool), 1, fh);
  if (A->destroyed)
    tk_lua_verror(L, 2, "load", "index was destroyed when saved");
  tk_lua_fread(L, &A->next_sid, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &A->bucket_target, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &A->features, sizeof(uint64_t), 1, fh);
  uint64_t n_hash = 0;
  tk_lua_fread(L, &n_hash, sizeof(uint64_t), 1, fh);
  A->hash_bits = tk_ivec_create(L, n_hash, 0, 0);
  A->hash_bits->n = n_hash;
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  if (n_hash)
    tk_lua_fread(L, A->hash_bits->a, sizeof(int64_t), n_hash, fh);
  lua_pop(L, 1);
  A->buckets = tk_ann_buckets_create(L, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  khint_t nb = 0, k; int absent;
  tk_lua_fread(L, &nb, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nb; i ++) {
    tk_ann_hash_t hkey;
    bool has;
    tk_lua_fread(L, &hkey, sizeof(tk_ann_hash_t), 1, fh);
    tk_lua_fread(L, &has, sizeof(bool),           1, fh);
    k = tk_ann_buckets_put(A->buckets, hkey, &absent);
    if (has) {
      uint64_t plen;
      tk_lua_fread(L, &plen, sizeof(uint64_t), 1, fh);
      tk_ivec_t *plist = tk_ivec_create(L, plen, 0, 0);
      plist->n = plen;
      tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
      if (plen)
        tk_lua_fread(L, plist->a, sizeof(int64_t), plen, fh);
      lua_pop(L, 1);
      tk_ann_buckets_setval(A->buckets, k, plist);
    } else {
      tk_ann_buckets_setval(A->buckets, k, NULL);
    }
  }
  A->uid_sid = tk_iumap_load(L, fh);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  A->sid_to_uid = tk_ivec_load(L, fh);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  uint64_t vcount = 0;
  tk_lua_fread(L, &vcount, sizeof(uint64_t), 1, fh);
  A->vectors = tk_cvec_create(L, vcount, 0, 0);
  A->vectors->n = vcount;
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  if (vcount)
    tk_lua_fread(L, A->vectors->a, 1, vcount, fh);
  lua_pop(L, 1);
  return A;
}

#endif
