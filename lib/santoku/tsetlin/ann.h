#ifndef TK_ANN_H
#define TK_ANN_H

#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/pvec.h>
#include <santoku/iumap.h>
#include <santoku/ivec/ext.h>
#include <santoku/cvec/ext.h>
#include <omp.h>

#define TK_ANN_SUBSTR_BITS 32
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
  uint64_t features;
  uint64_t m;
  tk_ann_buckets_t **tables;
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

static inline void tk_ann_shrink (lua_State *L, tk_ann_t *A, int Ai)
{
  if (A->destroyed)
    return;
  if (A->next_sid > SIZE_MAX / sizeof(int64_t))
    tk_error(L, "ann_shrink: allocation size overflow", ENOMEM);
  int64_t *old_to_new = tk_malloc(L, A->next_sid * sizeof(int64_t));
  for (uint64_t i = 0; i < A->next_sid; i++)
    old_to_new[i] = -1;
  uint64_t new_sid = 0;
  for (uint64_t s = 0; s < A->next_sid; s++) {
    if (A->sid_to_uid->a[s] >= 0)
      old_to_new[s] = (int64_t) new_sid++;
  }
  if (new_sid == A->next_sid) {
    free(old_to_new);
    tk_cvec_shrink(A->vectors);
    return;
  }
  size_t bytes_per_vec = TK_CVEC_BITS_BYTES(A->features);
  char *vecs = A->vectors->a;
  for (uint64_t old_sid = 0; old_sid < A->next_sid; old_sid++) {
    if (A->sid_to_uid->a[old_sid] < 0) continue;
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val != (int64_t)old_sid)
      memmove(vecs + (uint64_t)new_sid_val * bytes_per_vec,
              vecs + old_sid * bytes_per_vec, bytes_per_vec);
  }
  A->vectors->n = new_sid * bytes_per_vec;
  for (uint64_t ti = 0; ti < A->m; ti++) {
    tk_ivec_t *posting;
    tk_umap_foreach_values(A->tables[ti], posting, ({
      if (!posting) continue;
      uint64_t write_pos = 0;
      for (uint64_t i = 0; i < posting->n; i++) {
        int64_t new_sid_val = old_to_new[posting->a[i]];
        if (new_sid_val >= 0)
          posting->a[write_pos++] = new_sid_val;
      }
      posting->n = write_pos;
      tk_ivec_shrink(posting);
    }))
  }
  tk_iumap_t *new_uid_sid = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  int64_t uid, old_sid;
  tk_umap_foreach(A->uid_sid, uid, old_sid, ({
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val >= 0) {
      int is_new;
      khint_t khi = tk_iumap_put(new_uid_sid, uid, &is_new);
      tk_iumap_setval(new_uid_sid, khi, new_sid_val);
    }
  }))
  tk_lua_del_ephemeron(L, TK_ANN_EPH, Ai, A->uid_sid);
  tk_iumap_destroy(A->uid_sid);
  A->uid_sid = new_uid_sid;
  tk_ivec_t *new_sid_to_uid = tk_ivec_create(L, new_sid, 0, 0);
  tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
  lua_pop(L, 1);
  new_sid_to_uid->n = new_sid;
  for (uint64_t old_sid = 0; old_sid < A->next_sid; old_sid++) {
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val >= 0)
      new_sid_to_uid->a[new_sid_val] = A->sid_to_uid->a[old_sid];
  }
  tk_lua_del_ephemeron(L, TK_ANN_EPH, Ai, A->sid_to_uid);
  A->sid_to_uid = new_sid_to_uid;
  A->next_sid = new_sid;
  tk_cvec_shrink(A->vectors);
  free(old_to_new);
}

static inline void tk_ann_destroy (tk_ann_t *A)
{
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

static inline int64_t tk_ann_uid_sid (tk_ann_t *A, int64_t uid, tk_ann_uid_mode_t mode)
{
  int kha;
  khint_t khi;
  if (mode == TK_ANN_FIND) {
    khi = tk_iumap_get(A->uid_sid, uid);
    if (khi == tk_iumap_end(A->uid_sid))
      return -1;
    return tk_iumap_val(A->uid_sid, khi);
  } else {
    khi = tk_iumap_get(A->uid_sid, uid);
    if (khi != tk_iumap_end(A->uid_sid)) {
      int64_t old_sid = tk_iumap_val(A->uid_sid, khi);
      tk_iumap_del(A->uid_sid, khi);
      if (old_sid >= 0 && old_sid < (int64_t)A->sid_to_uid->n)
        A->sid_to_uid->a[old_sid] = -1;
    }
    int64_t sid = (int64_t)(A->next_sid++);
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

static inline char *tk_ann_sget (tk_ann_t *A, int64_t sid)
{
  return A->vectors->a + (uint64_t)sid * TK_CVEC_BITS_BYTES(A->features);
}

static inline char *tk_ann_get (tk_ann_t *A, int64_t uid)
{
  int64_t sid = tk_ann_uid_sid(A, uid, TK_ANN_FIND);
  if (sid < 0)
    return NULL;
  return tk_ann_sget(A, sid);
}

static inline double tk_ann_similarity (tk_ann_t *A, int64_t uid0, int64_t uid1)
{
  char *v0 = tk_ann_get(A, uid0);
  char *v1 = tk_ann_get(A, uid1);
  if (!v0 || !v1)
    return 0.0;
  uint64_t dist = tk_cvec_bits_hamming_serial((const uint8_t *)v0, (const uint8_t *)v1, A->features);
  return 1.0 - ((double)dist / (double)A->features);
}

static inline double tk_ann_distance (tk_ann_t *A, int64_t uid0, int64_t uid1)
{
  char *v0 = tk_ann_get(A, uid0);
  char *v1 = tk_ann_get(A, uid1);
  if (!v0 || !v1)
    return 1.0;
  uint64_t dist = tk_cvec_bits_hamming_serial((const uint8_t *)v0, (const uint8_t *)v1, A->features);
  return (double)dist / (double)A->features;
}

static inline tk_ann_hash_t tk_ann_substring (tk_ann_t *A, const char *data, uint64_t ti)
{
  uint64_t bit_offset = ti * TK_ANN_SUBSTR_BITS;
  uint64_t byte_offset = bit_offset / 8;
  tk_ann_hash_t h = 0;
  uint64_t remaining = A->features - bit_offset;
  uint64_t bits_to_copy = remaining < TK_ANN_SUBSTR_BITS ? remaining : TK_ANN_SUBSTR_BITS;
  uint64_t bytes_to_copy = (bits_to_copy + 7) / 8;
  memcpy(&h, data + byte_offset, bytes_to_copy);
  if (bits_to_copy < 32)
    h &= (1u << bits_to_copy) - 1;
  return h;
}

static inline void tk_ann_persist (lua_State *L, tk_ann_t *A, FILE *fh)
{
  if (A->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't persist a destroyed index");
    return;
  }
  tk_lua_fwrite(L, (char *)&A->destroyed, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *)&A->next_sid, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *)&A->features, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *)&A->m, sizeof(uint64_t), 1, fh);
  for (uint64_t ti = 0; ti < A->m; ti++) {
    khint_t nb = A->tables[ti] ? tk_ann_buckets_size(A->tables[ti]) : 0;
    tk_lua_fwrite(L, (char *)&nb, sizeof(khint_t), 1, fh);
    tk_ann_hash_t hkey;
    tk_ivec_t *plist;
    tk_umap_foreach(A->tables[ti], hkey, plist, ({
      tk_lua_fwrite(L, (char *)&hkey, sizeof(tk_ann_hash_t), 1, fh);
      bool has = plist && plist->n;
      tk_lua_fwrite(L, (char *)&has, sizeof(bool), 1, fh);
      if (has) {
        uint64_t plen = plist->n;
        tk_lua_fwrite(L, (char *)&plen, sizeof(uint64_t), 1, fh);
        tk_lua_fwrite(L, (char *)plist->a, sizeof(int64_t), plen, fh);
      }
    }))
  }
  tk_iumap_persist(L, A->uid_sid, fh);
  tk_ivec_persist(L, A->sid_to_uid, fh);
  uint64_t vcount = A->vectors->n;
  tk_lua_fwrite(L, (char *)&vcount, sizeof(uint64_t), 1, fh);
  if (vcount)
    tk_lua_fwrite(L, A->vectors->a, 1, vcount, fh);
}

static inline uint64_t tk_ann_size (tk_ann_t *A)
{
  return tk_iumap_size(A->uid_sid);
}

static inline void tk_ann_uid_remove (tk_ann_t *A, int64_t uid)
{
  khint_t khi = tk_iumap_get(A->uid_sid, uid);
  if (khi == tk_iumap_end(A->uid_sid))
    return;
  int64_t sid = tk_iumap_val(A->uid_sid, khi);
  tk_iumap_del(A->uid_sid, khi);
  if (sid >= 0 && sid < (int64_t)A->sid_to_uid->n)
    A->sid_to_uid->a[sid] = -1;
}

static inline int64_t tk_ann_sid_uid (tk_ann_t *A, int64_t sid)
{
  if (sid < 0 || sid >= (int64_t)A->sid_to_uid->n)
    return -1;
  return A->sid_to_uid->a[sid];
}

static inline void tk_ann_add (lua_State *L, tk_ann_t *A, int Ai, tk_ivec_t *ids, char *data)
{
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
  for (uint64_t i = 0; i < ids->n; i++) {
    int64_t sid = tk_ann_uid_sid(A, ids->a[i], TK_ANN_REPLACE);
    char *vec = data + i * bytes_per_vec;
    for (uint64_t ti = 0; ti < A->m; ti++) {
      tk_ann_hash_t h = tk_ann_substring(A, vec, ti);
      khi = kh_put(tk_ann_buckets, A->tables[ti], h, &kha);
      if (kha) {
        kh_value(A->tables[ti], khi) = tk_ivec_create(L, 0, 0, 0);
        tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
        lua_pop(L, 1);
      }
      tk_ivec_t *bucket = kh_value(A->tables[ti], khi);
      if (tk_ivec_push(bucket, sid) != 0) {
        tk_lua_verror(L, 2, "add", "allocation failed during indexing");
        return;
      }
    }
    tk_cvec_t datavec = {
      .n = bytes_per_vec,
      .m = bytes_per_vec,
      .a = vec
    };
    tk_cvec_copy(A->vectors, &datavec, 0, (int64_t)datavec.n, sid * (int64_t)bytes_per_vec);
  }
}

static inline void tk_ann_remove (lua_State *L, tk_ann_t *A, int64_t uid)
{
  if (A->destroyed) {
    tk_lua_verror(L, 2, "remove", "can't remove from a destroyed index");
    return;
  }
  tk_ann_uid_remove(A, uid);
}

static inline void tk_ann_probe_table_at_radius (
  tk_ann_t *A,
  uint64_t ti,
  tk_ann_hash_t h,
  int r,
  int64_t skip_sid,
  tk_iumap_t *seen,
  const unsigned char *query,
  uint64_t k,
  tk_pvec_t *out
) {
  uint64_t sub_bits = (ti < A->m - 1) ? TK_ANN_SUBSTR_BITS :
    (A->features - ti * TK_ANN_SUBSTR_BITS);
  int nbits = (int)sub_bits;
  if (r > nbits)
    return;
  int pos[TK_ANN_SUBSTR_BITS];
  for (int i = 0; i < r; i++)
    pos[i] = i;
  while (true) {
    tk_ann_hash_t mask = 0;
    for (int i = 0; i < r; i++)
      mask |= (1U << pos[i]);
    tk_ann_hash_t probe_h = h ^ mask;
    khint_t khi = kh_get(tk_ann_buckets, A->tables[ti], probe_h);
    if (khi != kh_end(A->tables[ti])) {
      tk_ivec_t *bucket = kh_value(A->tables[ti], khi);
      for (uint64_t bi = 0; bi < bucket->n; bi++) {
        int64_t sid = bucket->a[bi];
        if (sid == skip_sid)
          continue;
        khint_t sh = tk_iumap_get(seen, sid);
        if (sh != tk_iumap_end(seen))
          continue;
        int64_t uid = tk_ann_sid_uid(A, sid);
        if (uid < 0)
          continue;
        int dummy;
        sh = tk_iumap_put(seen, sid, &dummy);
        tk_iumap_setval(seen, sh, 1);
        const unsigned char *vec = (const unsigned char *)tk_ann_sget(A, sid);
        uint64_t dist = tk_cvec_bits_hamming_serial(query, vec, A->features);
        if (k)
          tk_pvec_hmax(out, k, tk_pair(uid, (int64_t)dist));
        else
          tk_pvec_push(out, tk_pair(uid, (int64_t)dist));
      }
    }
    if (r == 0)
      break;
    int j;
    for (j = r - 1; j >= 0; j--) {
      if (pos[j] != j + nbits - r) {
        pos[j]++;
        for (int l = j + 1; l < r; l++)
          pos[l] = pos[l - 1] + 1;
        break;
      }
    }
    if (j < 0)
      break;
  }
}

static inline void tk_ann_query_mih (
  tk_ann_t *A,
  const char *query,
  int64_t skip_sid,
  uint64_t max_probe_radius,
  uint64_t k,
  tk_pvec_t *out
) {
  tk_pvec_clear(out);
  tk_iumap_t *seen = tk_iumap_create(NULL, 0);
  const unsigned char *q = (const unsigned char *)query;
  tk_ann_hash_t hs[16];
  for (uint64_t ti = 0; ti < A->m && ti < 16; ti++)
    hs[ti] = tk_ann_substring(A, query, ti);
  for (int r = 0; r <= (int)max_probe_radius; r++) {
    for (uint64_t ti = 0; ti < A->m; ti++)
      tk_ann_probe_table_at_radius(A, ti, hs[ti], r, skip_sid, seen, q, k, out);
    if (k && out->n >= k)
      break;
  }
  tk_iumap_destroy(seen);
  tk_pvec_asc(out, 0, out->n);
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

static inline void tk_ann_probe_table_pos_at_radius (
  tk_ann_t *A,
  tk_ivec_t *sid_to_pos,
  uint64_t ti,
  tk_ann_hash_t h,
  int r,
  int64_t skip_sid,
  tk_iumap_t *seen,
  const unsigned char *query,
  uint64_t k,
  tk_pvec_t *out
) {
  uint64_t sub_bits = (ti < A->m - 1) ? TK_ANN_SUBSTR_BITS :
    (A->features - ti * TK_ANN_SUBSTR_BITS);
  int nbits = (int)sub_bits;
  if (r > nbits)
    return;
  int pos[TK_ANN_SUBSTR_BITS];
  for (int i = 0; i < r; i++)
    pos[i] = i;
  while (true) {
    tk_ann_hash_t mask = 0;
    for (int i = 0; i < r; i++)
      mask |= (1U << pos[i]);
    tk_ann_hash_t probe_h = h ^ mask;
    khint_t khi = kh_get(tk_ann_buckets, A->tables[ti], probe_h);
    if (khi != kh_end(A->tables[ti])) {
      tk_ivec_t *bucket = kh_value(A->tables[ti], khi);
      for (uint64_t bi = 0; bi < bucket->n; bi++) {
        int64_t sid = bucket->a[bi];
        if (sid == skip_sid)
          continue;
        khint_t sh = tk_iumap_get(seen, sid);
        if (sh != tk_iumap_end(seen))
          continue;
        if (sid < 0 || sid >= (int64_t)sid_to_pos->n)
          continue;
        int64_t id = sid_to_pos->a[sid];
        if (id < 0)
          continue;
        int dummy;
        sh = tk_iumap_put(seen, sid, &dummy);
        tk_iumap_setval(seen, sh, 1);
        const unsigned char *vec = (const unsigned char *)tk_ann_sget(A, sid);
        uint64_t dist = tk_cvec_bits_hamming_serial(query, vec, A->features);
        if (k)
          tk_pvec_hmax(out, k, tk_pair(id, (int64_t)dist));
        else
          tk_pvec_push(out, tk_pair(id, (int64_t)dist));
      }
    }
    if (r == 0)
      break;
    int j;
    for (j = r - 1; j >= 0; j--) {
      if (pos[j] != j + nbits - r) {
        pos[j]++;
        for (int l = j + 1; l < r; l++)
          pos[l] = pos[l - 1] + 1;
        break;
      }
    }
    if (j < 0)
      break;
  }
}

static inline void tk_ann_neighborhoods (
  lua_State *L,
  tk_ann_t *A,
  uint64_t k,
  uint64_t max_probe_radius,
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
  for (uint64_t i = 0; i < hoods->n; i++) {
    tk_pvec_t *hood = tk_pvec_create(L, k ? k : 16, 0, 0);
    hood->n = 0;
    tk_lua_add_ephemeron(L, TK_ANN_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
    hoods->a[i] = hood;
  }
  #pragma omp parallel
  {
    tk_iumap_t *seen = tk_iumap_create(NULL, 0);
    tk_ann_hash_t hs[16];
    #pragma omp for schedule(static)
    for (uint64_t i = 0; i < hoods->n; i++) {
      tk_pvec_t *hood = hoods->a[i];
      tk_pvec_clear(hood);
      tk_iumap_clear(seen);
      int64_t uid = uids->a[i];
      int64_t sid = tk_ann_uid_sid(A, uid, TK_ANN_FIND);
      const char *vec = tk_ann_sget(A, sid);
      const unsigned char *q = (const unsigned char *)vec;
      for (uint64_t ti = 0; ti < A->m && ti < 16; ti++)
        hs[ti] = tk_ann_substring(A, vec, ti);
      for (int r = 0; r <= (int)max_probe_radius; r++) {
        for (uint64_t ti = 0; ti < A->m; ti++)
          tk_ann_probe_table_pos_at_radius(A, sid_to_pos, ti, hs[ti], r, sid, seen, q, k, hood);
        if (k && hood->n >= k)
          break;
      }
      tk_pvec_asc(hood, 0, hood->n);
    }
    tk_iumap_destroy(seen);
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
  uint64_t max_probe_radius,
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
  for (uint64_t i = 0; i < hoods->n; i++) {
    hoods->a[i] = tk_pvec_create(L, k ? k : 16, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_ANN_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
  }
  #pragma omp parallel
  {
    tk_iumap_t *seen = tk_iumap_create(NULL, 0);
    tk_ann_hash_t hs[16];
    #pragma omp for schedule(static)
    for (uint64_t i = 0; i < hoods->n; i++) {
      tk_pvec_t *hood = hoods->a[i];
      tk_iumap_clear(seen);
      int64_t uid = query_ids->a[i];
      int64_t sid = tk_ann_uid_sid(A, uid, TK_ANN_FIND);
      if (sid < 0)
        continue;
      const char *vec = tk_ann_sget(A, sid);
      const unsigned char *q = (const unsigned char *)vec;
      for (uint64_t ti = 0; ti < A->m && ti < 16; ti++)
        hs[ti] = tk_ann_substring(A, vec, ti);
      for (int r = 0; r <= (int)max_probe_radius; r++) {
        for (uint64_t ti = 0; ti < A->m; ti++)
          tk_ann_probe_table_pos_at_radius(A, sid_to_pos, ti, hs[ti], r, sid, seen, q, k, hood);
        if (k && hood->n >= k)
          break;
      }
      tk_pvec_asc(hood, 0, hood->n);
    }
    tk_iumap_destroy(seen);
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
  uint64_t max_probe_radius,
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
  for (uint64_t i = 0; i < hoods->n; i++) {
    hoods->a[i] = tk_pvec_create(L, k ? k : 16, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_ANN_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
  }
  #pragma omp parallel
  {
    tk_iumap_t *seen = tk_iumap_create(NULL, 0);
    tk_ann_hash_t hs[16];
    #pragma omp for schedule(static)
    for (uint64_t i = 0; i < hoods->n; i++) {
      tk_pvec_t *hood = hoods->a[i];
      tk_iumap_clear(seen);
      const char *vec = query_vecs->a + i * vec_bytes;
      const unsigned char *q = (const unsigned char *)vec;
      for (uint64_t ti = 0; ti < A->m && ti < 16; ti++)
        hs[ti] = tk_ann_substring(A, vec, ti);
      for (int r = 0; r <= (int)max_probe_radius; r++) {
        for (uint64_t ti = 0; ti < A->m; ti++)
          tk_ann_probe_table_pos_at_radius(A, sid_to_pos, ti, hs[ti], r, -1, seen, q, k, hood);
        if (k && hood->n >= k)
          break;
      }
      tk_pvec_asc(hood, 0, hood->n);
    }
    tk_iumap_destroy(seen);
  }
  tk_ivec_destroy(sid_to_pos);
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = all_uids;
}

static inline tk_pvec_t *tk_ann_neighbors_by_vec (
  tk_ann_t *A,
  char *vec,
  int64_t skip_sid,
  uint64_t k,
  uint64_t radius,
  tk_pvec_t *out
) {
  if (A->destroyed)
    return NULL;
  tk_ann_query_mih(A, vec, skip_sid, radius, k, out);
  return out;
}

static inline tk_pvec_t *tk_ann_neighbors_by_id (
  tk_ann_t *A,
  int64_t uid,
  uint64_t k,
  uint64_t radius,
  tk_pvec_t *out
) {
  int64_t sid = tk_ann_uid_sid(A, uid, TK_ANN_FIND);
  if (sid < 0) {
    tk_pvec_clear(out);
    return out;
  }
  return tk_ann_neighbors_by_vec(A, tk_ann_get(A, uid), sid, k, radius, out);
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
  if (data_cvec)
    data = data_cvec->a;
  else
    data = (char *)tk_lua_checkustring(L, 2, "data");
  if (lua_type(L, 3) == LUA_TNUMBER) {
    int64_t s = (int64_t)tk_lua_checkunsigned(L, 3, "base_id");
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
    for (uint64_t i = 0; i < ids->n; i++)
      tk_ann_uid_remove(A, ids->a[i]);
  }
  return 0;
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
    uids = lua_isnil(L, 2) ? tk_iumap_keys(L, A->uid_sid) : tk_ivec_peek(L, 2, "uids");
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
    if (total_bytes > old_size)
      memset(dest_data + old_size, 0, total_bytes - old_size);
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
            if (byte_offset + i + 1 < total_bytes)
              dest_data[byte_offset + i + 1] |= byte >> (CHAR_BIT - bit_shift);
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
              if (byte_offset + j + 1 < total_bytes)
                dest_data[byte_offset + j + 1] |= byte >> (CHAR_BIT - bit_shift);
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
  lua_settop(L, 3);
  tk_ann_t *A = tk_ann_peek(L, 1);
  uint64_t k = tk_lua_optunsigned(L, 2, "k", 0);
  uint64_t max_probe_radius = tk_lua_optunsigned(L, 3, "radius", 3);
  tk_ann_neighborhoods(L, A, k, max_probe_radius, NULL, NULL);
  return 2;
}

static inline int tk_ann_neighborhoods_by_ids_lua (lua_State *L)
{
  lua_settop(L, 4);
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_ivec_t *query_ids = tk_ivec_peek(L, 2, "ids");
  uint64_t k = tk_lua_optunsigned(L, 3, "k", 0);
  uint64_t max_probe_radius = tk_lua_optunsigned(L, 4, "radius", 3);
  int64_t write_pos = 0;
  for (int64_t i = 0; i < (int64_t)query_ids->n; i++) {
    int64_t uid = query_ids->a[i];
    khint_t kh = tk_iumap_get(A->uid_sid, uid);
    if (kh != tk_iumap_end(A->uid_sid))
      query_ids->a[write_pos++] = uid;
  }
  query_ids->n = (uint64_t)write_pos;
  tk_ann_hoods_t *hoods;
  tk_ann_neighborhoods_by_ids(L, A, query_ids, k, max_probe_radius, &hoods, &query_ids);
  return 2;
}

static inline int tk_ann_neighborhoods_by_vecs_lua (lua_State *L)
{
  lua_settop(L, 4);
  tk_ann_t *A = tk_ann_peek(L, 1);
  tk_cvec_t *query_vecs = tk_cvec_peek(L, 2, "vectors");
  uint64_t k = tk_lua_optunsigned(L, 3, "k", 0);
  uint64_t max_probe_radius = tk_lua_optunsigned(L, 4, "radius", 3);
  tk_ann_neighborhoods_by_vecs(L, A, query_vecs, k, max_probe_radius, NULL, NULL);
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
  lua_settop(L, 5);
  tk_ann_t *A = tk_ann_peek(L, 1);
  uint64_t k = tk_lua_checkunsigned(L, 3, "k");
  uint64_t max_probe_radius = tk_lua_optunsigned(L, 4, "radius", 3);
  tk_pvec_t *out = tk_pvec_peek(L, 5, "out");
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_ann_neighbors_by_id(A, uid, k, max_probe_radius, out);
  } else {
    char *vec;
    tk_cvec_t *vec_cvec = tk_cvec_peekopt(L, 2);
    if (vec_cvec)
      vec = vec_cvec->a;
    else
      vec = (char *)tk_lua_checkustring(L, 2, "vector");
    tk_ann_neighbors_by_vec(A, vec, -1, k, max_probe_radius, out);
  }
  return 0;
}

static inline int tk_ann_size_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  lua_pushinteger(L, (int64_t)tk_ann_size(A));
  return 1;
}

static inline int tk_ann_features_lua (lua_State *L)
{
  tk_ann_t *A = tk_ann_peek(L, 1);
  lua_pushinteger(L, (int64_t)A->features);
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

static inline tk_ann_t *tk_ann_create (lua_State *L, uint64_t features)
{
  tk_ann_t *A = tk_lua_newuserdata(L, tk_ann_t, TK_ANN_MT, tk_ann_lua_mt_fns, tk_ann_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  A->features = features;
  A->m = (features + TK_ANN_SUBSTR_BITS - 1) / TK_ANN_SUBSTR_BITS;
  if (A->m == 0) A->m = 1;
  A->tables = tk_malloc(L, A->m * sizeof(tk_ann_buckets_t *));
  for (uint64_t ti = 0; ti < A->m; ti++) {
    A->tables[ti] = tk_ann_buckets_create(L, 0);
    tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
    lua_pop(L, 1);
  }
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
  return A;
}

static inline tk_ann_t *tk_ann_load (lua_State *L, FILE *fh)
{
  tk_ann_t *A = tk_lua_newuserdata(L, tk_ann_t, TK_ANN_MT, tk_ann_lua_mt_fns, tk_ann_gc_lua);
  int Ai = tk_lua_absindex(L, -1);
  memset(A, 0, sizeof(tk_ann_t));
  tk_lua_fread(L, &A->destroyed, sizeof(bool), 1, fh);
  if (A->destroyed)
    tk_lua_verror(L, 2, "load", "index was destroyed when saved");
  tk_lua_fread(L, &A->next_sid, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &A->features, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &A->m, sizeof(uint64_t), 1, fh);
  A->tables = tk_malloc(L, A->m * sizeof(tk_ann_buckets_t *));
  for (uint64_t ti = 0; ti < A->m; ti++) {
    A->tables[ti] = tk_ann_buckets_create(L, 0);
    tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
    lua_pop(L, 1);
    khint_t nb = 0;
    tk_lua_fread(L, &nb, sizeof(khint_t), 1, fh);
    for (khint_t i = 0; i < nb; i++) {
      tk_ann_hash_t hkey;
      bool has;
      int absent;
      tk_lua_fread(L, &hkey, sizeof(tk_ann_hash_t), 1, fh);
      tk_lua_fread(L, &has, sizeof(bool), 1, fh);
      khint_t k = tk_ann_buckets_put(A->tables[ti], hkey, &absent);
      if (has) {
        uint64_t plen;
        tk_lua_fread(L, &plen, sizeof(uint64_t), 1, fh);
        tk_ivec_t *plist = tk_ivec_create(L, plen, 0, 0);
        plist->n = plen;
        tk_lua_add_ephemeron(L, TK_ANN_EPH, Ai, -1);
        if (plen)
          tk_lua_fread(L, plist->a, sizeof(int64_t), plen, fh);
        lua_pop(L, 1);
        tk_ann_buckets_setval(A->tables[ti], k, plist);
      } else {
        tk_ann_buckets_setval(A->tables[ti], k, NULL);
      }
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
