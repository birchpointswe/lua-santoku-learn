#ifndef TK_INV_H
#define TK_INV_H

#include <assert.h>
#include <math.h>
#include <string.h>
#include <santoku/lua/utils.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/iuset.h>
#include <santoku/dumap.h>
#include <santoku/threads.h>
#include <santoku/ivec/ext.h>

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
  TK_INV_MUTUAL_INIT,
  TK_INV_MUTUAL_FILTER,
  TK_INV_MIN_REMAP,
  TK_INV_COLLECT_UIDS,
  TK_INV_REMAP_UIDS,
} tk_inv_stage_t;

typedef struct tk_inv_thread_s tk_inv_thread_t;

typedef struct tk_inv_s {
  bool destroyed;
  int64_t next_sid;
  uint64_t features;
  uint64_t n_ranks;
  tk_dvec_t *weights;
  tk_ivec_t *ranks;
  double decay;
  double total_rank_weight;
  tk_dvec_t *rank_weights;
  tk_iumap_t *uid_sid;
  tk_iumap_t *sid_uid;
  tk_ivec_t *node_offsets;
  tk_ivec_t *node_bits;
  tk_inv_postings_t *postings;
  tk_dvec_t *wacc;
  tk_ivec_t *touched;
  tk_inv_thread_t *threads;
  tk_threadpool_t *pool;
  tk_ivec_t *tmp_query_offsets;
  tk_ivec_t *tmp_query_features;
  tk_dvec_t *tmp_q_weights;
  tk_dvec_t *tmp_e_weights;
  tk_dvec_t *tmp_inter_weights;
} tk_inv_t;

typedef struct tk_inv_thread_s {
  tk_inv_t *I;
  tk_inv_hoods_t *hoods;
  tk_dumap_t **hoods_sets;
  tk_iuset_t *seen;
  tk_ivec_t *touched;
  tk_iumap_t *sid_idx;
  tk_ivec_t *uids;
  tk_ivec_t *sids;
  tk_ivec_t *query_offsets;
  tk_ivec_t *query_features;
  tk_dvec_t *wacc;
  tk_dvec_t *q_weights_buf;
  tk_dvec_t *e_weights_buf;
  tk_dvec_t *inter_weights_buf;
  uint64_t ifirst, ilast;
  double eps;
  uint64_t knn;
  uint64_t min;
  bool mutual;
  tk_ivec_sim_type_t cmp;
  double tversky_alpha;
  double tversky_beta;
  int64_t *old_to_new;
  tk_iuset_t *local_uids;
  tk_iumap_t *uid_to_idx;
  int64_t rank_filter;
} tk_inv_thread_t;

static inline tk_inv_t *tk_inv_peek (lua_State *L, int i)
{
  return (tk_inv_t *) luaL_checkudata(L, i, TK_INV_MT);
}

static inline tk_inv_t *tk_inv_peekopt (lua_State *L, int i)
{
  return (tk_inv_t *) tk_lua_testuserdata(L, i, TK_INV_MT);
}

static inline void tk_inv_shrink (
  lua_State *L,
  tk_inv_t *I
) {
  if (I->destroyed)
    return;

  int64_t *old_to_new = tk_malloc(L, (size_t) I->next_sid * sizeof(int64_t));
  for (int64_t i = 0; i < I->next_sid; i ++)
    old_to_new[i] = -1;
  int64_t new_sid = 0;
  for (khint_t k = kh_begin(I->sid_uid); k != kh_end(I->sid_uid); k ++) {
    if (!kh_exist(I->sid_uid, k))
      continue;
    int64_t old_sid = kh_key(I->sid_uid, k);
    old_to_new[old_sid] = new_sid ++;
  }
  if (new_sid == I->next_sid) {
    free(old_to_new);
    tk_inv_postings_shrink(I->postings);
    for (uint64_t i = 0; i < I->postings->n; i ++)
      tk_ivec_shrink(I->postings->a[i]);
    tk_ivec_shrink(I->node_offsets);
    tk_ivec_shrink(I->node_bits);
    tk_dvec_shrink(I->wacc);
    tk_ivec_shrink(I->touched);
    if (I->weights) tk_dvec_shrink(I->weights);
    if (I->ranks) tk_ivec_shrink(I->ranks);
    if (I->tmp_query_offsets) tk_ivec_shrink(I->tmp_query_offsets);
    if (I->tmp_query_features) tk_ivec_shrink(I->tmp_query_features);
    if (I->tmp_q_weights) tk_dvec_shrink(I->tmp_q_weights);
    if (I->tmp_e_weights) tk_dvec_shrink(I->tmp_e_weights);
    if (I->tmp_inter_weights) tk_dvec_shrink(I->tmp_inter_weights);
    return;
  }
  tk_ivec_t *new_node_offsets = tk_ivec_create(L, (size_t) new_sid + 1, 0, 0);
  tk_ivec_t *new_node_bits = tk_ivec_create(L, 0, 0, 0);

  new_node_offsets->n = 0;
  for (khint_t k = kh_begin(I->sid_uid); k != kh_end(I->sid_uid); k ++) {
    if (!kh_exist(I->sid_uid, k))
      continue;
    int64_t old_sid = kh_key(I->sid_uid, k);
    int64_t start = I->node_offsets->a[old_sid];
    int64_t end = I->node_offsets->a[old_sid + 1];
    tk_ivec_push(new_node_offsets, (int64_t) new_node_bits->n);
    for (int64_t i = start; i < end; i ++)
      tk_ivec_push(new_node_bits, I->node_bits->a[i]);
  }
  tk_ivec_push(new_node_offsets, (int64_t) new_node_bits->n);
  tk_ivec_destroy(I->node_offsets);
  tk_ivec_destroy(I->node_bits);
  I->node_offsets = new_node_offsets;
  I->node_bits = new_node_bits;
  for (uint64_t fid = 0; fid < I->postings->n; fid ++) {
    tk_ivec_t *post = I->postings->a[fid];
    for (uint64_t i = 0; i < post->n; i ++) {
      int64_t old_sid = post->a[i];
      int64_t new_sid_val = old_to_new[old_sid];
      if (new_sid_val >= 0)
        post->a[i] = new_sid_val;
    }    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < post->n; i ++) {
      if (old_to_new[post->a[i]] >= 0)
        post->a[write_pos ++] = post->a[i];
    }
    post->n = write_pos;
    tk_ivec_shrink(post);
  }
  tk_iumap_t *new_uid_sid = tk_iumap_create();
  tk_iumap_t *new_sid_uid = tk_iumap_create();

  for (khint_t k = kh_begin(I->uid_sid); k != kh_end(I->uid_sid); k ++) {
    if (!kh_exist(I->uid_sid, k))
      continue;
    int64_t uid = kh_key(I->uid_sid, k);
    int64_t old_sid = kh_value(I->uid_sid, k);
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val >= 0) {
      int is_new;
      khint_t khi = tk_iumap_put(new_uid_sid, uid, &is_new);
      tk_iumap_value(new_uid_sid, khi) = new_sid_val;
      khi = tk_iumap_put(new_sid_uid, new_sid_val, &is_new);
      tk_iumap_value(new_sid_uid, khi) = uid;
    }
  }

  tk_iumap_destroy(I->uid_sid);
  tk_iumap_destroy(I->sid_uid);
  I->uid_sid = new_uid_sid;
  I->sid_uid = new_sid_uid;
  I->next_sid = new_sid;

  tk_inv_postings_shrink(I->postings);
  tk_ivec_shrink(I->node_offsets);
  tk_ivec_shrink(I->node_bits);
  tk_dvec_shrink(I->wacc);
  tk_ivec_shrink(I->touched);
  if (I->weights) tk_dvec_shrink(I->weights);
  if (I->ranks) tk_ivec_shrink(I->ranks);
  if (I->tmp_query_offsets) tk_ivec_shrink(I->tmp_query_offsets);
  if (I->tmp_query_features) tk_ivec_shrink(I->tmp_query_features);
  if (I->tmp_q_weights) tk_dvec_shrink(I->tmp_q_weights);
  if (I->tmp_e_weights) tk_dvec_shrink(I->tmp_e_weights);
  if (I->tmp_inter_weights) tk_dvec_shrink(I->tmp_inter_weights);

  free(old_to_new);
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

static inline void tk_inv_persist (
  lua_State *L,
  tk_inv_t *I,
  FILE *fh
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't persist a destroyed index");
    return;
  }
  tk_lua_fwrite(L, (char *) &I->destroyed, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &I->next_sid, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &I->features, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &I->n_ranks, sizeof(uint64_t), 1, fh);
  khint_t m = I->uid_sid ? tk_iumap_size(I->uid_sid) : 0;
  tk_lua_fwrite(L, (char *) &m, sizeof(khint_t), 1, fh);
  if (m)
    for (khint_t i = tk_iumap_begin(I->uid_sid); i < tk_iumap_end(I->uid_sid); i ++) {
      if (tk_iumap_exist(I->uid_sid, i)) {
        int64_t k = (int64_t) tk_iumap_key(I->uid_sid, i);
        int64_t v = (int64_t) tk_iumap_value(I->uid_sid, i);
        tk_lua_fwrite(L, (char *) &k, sizeof(int64_t), 1, fh);
        tk_lua_fwrite(L, (char *) &v, sizeof(int64_t), 1, fh);
      }
    }
  m = I->sid_uid ? tk_iumap_size(I->sid_uid) : 0;
  tk_lua_fwrite(L, (char *) &m, sizeof(khint_t), 1, fh);
  if (m)
    for (khint_t i = tk_iumap_begin(I->sid_uid); i < tk_iumap_end(I->sid_uid); i ++) {
      if (tk_iumap_exist(I->sid_uid, i)) {
        int64_t k = (int64_t) tk_iumap_key(I->sid_uid, i);
        int64_t v = (int64_t) tk_iumap_value(I->sid_uid, i);
        tk_lua_fwrite(L, (char *) &k, sizeof(int64_t), 1, fh);
        tk_lua_fwrite(L, (char *) &v, sizeof(int64_t), 1, fh);
      }
    }
  uint64_t n = I->node_offsets ? I->node_offsets->n : 0;
  tk_lua_fwrite(L, (char *) &n, sizeof(uint64_t), 1, fh);
  if (n)
    tk_lua_fwrite(L, (char *) I->node_offsets->a, sizeof(int64_t), n, fh);
  n = I->node_bits ? I->node_bits->n : 0;
  tk_lua_fwrite(L, (char *) &n, sizeof(uint64_t), 1, fh);
  if (n)
    tk_lua_fwrite(L, (char *) I->node_bits->a, sizeof(int64_t), n, fh);
  uint64_t pcount = I->postings ? I->postings->n : 0;
  tk_lua_fwrite(L, (char *) &pcount, sizeof(uint64_t), 1, fh);
  for (uint64_t i = 0; i < pcount; i ++) {
    tk_inv_posting_t P = I->postings->a[i];
    tk_lua_fwrite(L, (char *) &P->n, sizeof(uint64_t), 1, fh);
    tk_lua_fwrite(L, (char *) P->a, sizeof(int64_t), P->n, fh);
  }
  uint64_t wn = I->weights ? I->weights->n : 0;
  tk_lua_fwrite(L, (char *) &wn, sizeof(uint64_t), 1, fh);
  if (wn)
    tk_lua_fwrite(L, (char *) I->weights->a, sizeof(double), wn, fh);
  uint64_t rn = I->ranks ? I->ranks->n : 0;
  tk_lua_fwrite(L, (char *) &rn, sizeof(uint64_t), 1, fh);
  if (rn)
    tk_lua_fwrite(L, (char *) I->ranks->a, sizeof(int64_t), rn, fh);
  tk_lua_fwrite(L, (char *) &I->decay, sizeof(double), 1, fh);
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
  if (khi == tk_iumap_end(I->uid_sid))
    return;
  int64_t sid = tk_iumap_value(I->uid_sid, khi);
  tk_iumap_del(I->uid_sid, khi);
  khi = tk_iumap_get(I->sid_uid, sid);
  if (khi == tk_iumap_end(I->sid_uid))
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
    if (khi == tk_iumap_end(I->uid_sid))
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
  if (khi == tk_iumap_end(I->sid_uid))
    return -1;
  else
    return tk_iumap_value(I->sid_uid, khi);
}

static inline int64_t *tk_inv_sget (
  tk_inv_t *I,
  int64_t sid,
  size_t *np
) {
  if (sid < 0 || sid + 1 > (int64_t) I->node_offsets->n) {
    *np = 0;
    return NULL;
  }
  int64_t start = I->node_offsets->a[sid];
  int64_t end;
  if (sid + 1 == (int64_t) I->node_offsets->n) {
    end = (int64_t) I->node_bits->n;
  } else {
    end = I->node_offsets->a[sid + 1];
  }
  if (start < 0 || end < start || end > (int64_t) I->node_bits->n) {
    *np = 0;
    return NULL;
  }
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
  tk_ivec_uasc(node_bits, 0, node_bits->n);
  size_t nb = node_bits->n;
  size_t nsamples = ids->n;
  size_t i = 0;
  for (size_t s = 0; s < nsamples; s ++) {
    int64_t uid = ids->a[s];
    int64_t sid = tk_inv_uid_sid(I, uid, true);
    tk_ivec_push(I->node_offsets, (int64_t) I->node_bits->n);
    while (i < nb) {
      int64_t b = node_bits->a[i];
      if (b < 0) {
        i ++;
        continue;
      }
      size_t sample_idx = (size_t) b / (size_t) I->features;
      if (sample_idx != s)
        break;
      int64_t fid = b % (int64_t) I->features;
      assert(fid >= 0 && fid < (int64_t)I->features && "fid out of range");
      tk_ivec_t *post = I->postings->a[fid];
      bool found = false;
      for (size_t j = 0; j < post->n; j ++)
        if (post->a[j] == sid) {
          found = true;
          break;
        }
      if (!found)
        tk_ivec_push(post, sid);
      tk_ivec_push(I->node_bits, fid);
      i ++;
    }
  }
  tk_ivec_push(I->node_offsets, (int64_t) I->node_bits->n);
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

static inline void tk_inv_keep (
  lua_State *L,
  tk_inv_t *I,
  tk_ivec_t *ids
) {
  if (I->destroyed) {
    tk_lua_verror(L, 2, "keep", "can't keep in a destroyed index");
    return;
  }

  tk_iuset_t *keep_set = tk_iuset_from_ivec(ids);
  tk_iuset_t *to_remove_set = tk_iuset_create();
  tk_iuset_union_iumap(to_remove_set, I->uid_sid);
  tk_iuset_difference(to_remove_set, keep_set);
  int64_t uid;
  tk_iuset_foreach(to_remove_set, uid, ({
    tk_inv_uid_remove(I, uid);
  }));
  tk_iuset_destroy(keep_set);
  tk_iuset_destroy(to_remove_set);
}

static inline void tk_inv_mutualize (
  lua_State *L,
  tk_inv_t *I,
  tk_inv_hoods_t *hoods,
  tk_ivec_t *uids,
  uint64_t min,
  int64_t **old_to_newp
) {
  if (I->destroyed)
    return;
  tk_ivec_t *sids = tk_ivec_create(L, uids->n, 0, 0); // sids
  for (uint64_t i = 0; i < uids->n; i ++)
    sids->a[i] = tk_inv_uid_sid(I, uids->a[i], false);
  tk_iumap_t *sid_idx = tk_iumap_from_ivec(sids);
  tk_dumap_t **hoods_sets = tk_malloc(L, uids->n * sizeof(tk_dumap_t *));
  for (uint64_t i = 0; i < uids->n; i ++)
    hoods_sets[i] = tk_dumap_create();
  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    data->uids = uids;
    data->sids = sids;
    data->hoods = hoods;
    data->hoods_sets = hoods_sets;
    data->sid_idx = sid_idx;
    tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }
  tk_threads_signal(I->pool, TK_INV_MUTUAL_INIT, 0);
  tk_threads_signal(I->pool, TK_INV_MUTUAL_FILTER, 0);
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
      for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
        tk_inv_thread_t *data = I->threads + i;
        data->old_to_new = old_to_new;
        data->min = min;
        tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
      }
      tk_threads_signal(I->pool, TK_INV_MIN_REMAP, 0);
      tk_ivec_t *new_uids = tk_ivec_create(L, (uint64_t) keeper_count, 0, 0);
      tk_inv_hoods_t *new_hoods = tk_inv_hoods_create(L, (uint64_t) keeper_count, 0, 0);
      new_hoods->n = (uint64_t) keeper_count;
      for (uint64_t i = 0; i < uids->n; i ++) {
        if (old_to_new[i] >= 0) {
          new_uids->a[old_to_new[i]] = uids->a[i];
          new_hoods->a[old_to_new[i]] = hoods->a[i];
        }
      }
      int64_t *old_uids_data = uids->a;
      tk_inv_hood_t *old_hoods_data = hoods->a;
      uids->a = new_uids->a;
      uids->n = (uint64_t) keeper_count;
      uids->m = (uint64_t) keeper_count;
      hoods->a = new_hoods->a;
      hoods->n = (uint64_t) keeper_count;
      hoods->m = (uint64_t) keeper_count;
      new_uids->a = old_uids_data;
      new_hoods->a = old_hoods_data;

      lua_remove(L, -2); // remove new_uids (frees old uids data)
      lua_remove(L, -1); // remove new_hoods (frees old hoods data)
    }

    if (old_to_newp) {
      *old_to_newp = old_to_new;
    } else {
      free(old_to_new);
    }
  }

  tk_iumap_destroy(sid_idx);
  for (uint64_t i = 0; i < uids->n; i ++)
    tk_dumap_destroy(hoods_sets[i]);
  free(hoods_sets);
  lua_pop(L, 1); // sids
}

static inline void tk_inv_neighborhoods (
  lua_State *L,
  tk_inv_t *I,
  uint64_t knn,
  double eps,
  uint64_t min,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  bool mutual,
  int64_t rank_filter,
  tk_inv_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (I->destroyed)
    return;
  tk_ivec_t *sids = tk_iumap_values(L, I->uid_sid);
  tk_ivec_asc(sids, 0, sids->n); // sort for cache locality
  tk_ivec_t *uids = tk_ivec_create(L, sids->n, 0, 0);
  for (uint64_t i = 0; i < sids->n; i ++)
    uids->a[i] = tk_inv_sid_uid(I, sids->a[i]);

  tk_iumap_t *sid_idx = tk_iumap_from_ivec(sids);
  tk_inv_hoods_t *hoods = tk_inv_hoods_create(L, uids->n, 0, 0);
  tk_dumap_t **hoods_sets = NULL;
  if (mutual && knn) {
    hoods_sets = tk_malloc(L, uids->n * sizeof(tk_dumap_t *));
    for (uint64_t i = 0; i < uids->n; i ++)
      hoods_sets[i] = tk_dumap_create();
  }

  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_rvec_create(L, knn, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
    lua_pop(L, 1);
  }

  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    data->uids = uids;
    data->sids = sids;
    data->query_offsets = NULL;
    data->query_features = NULL;
    tk_dvec_ensure(data->wacc, sids->n * I->n_ranks);
    data->hoods = hoods;
    data->hoods_sets = hoods_sets;
    data->sid_idx = sid_idx;
    data->eps = eps;
    data->knn = knn;
    data->mutual = mutual;
    data->cmp = cmp;
    data->tversky_alpha = tversky_alpha;
    data->tversky_beta = tversky_beta;
    data->rank_filter = rank_filter;
    tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }

  tk_threads_signal(I->pool, TK_INV_NEIGHBORHOODS, 0);
  if (mutual && knn) {
    tk_threads_signal(I->pool, TK_INV_MUTUAL_INIT, 0);
    tk_threads_signal(I->pool, TK_INV_MUTUAL_FILTER, 0);
  }
  tk_iumap_destroy(sid_idx);
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

    // Set up thread data for min filtering
    for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
      tk_inv_thread_t *data = I->threads + i;
      data->old_to_new = old_to_new;
      data->min = min;
      tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
    }

    // Remap neighborhood indices in parallel
    tk_threads_signal(I->pool, TK_INV_MIN_REMAP, 0);

    // Create new compacted arrays
    tk_ivec_t *new_uids = tk_ivec_create(L, (uint64_t) keeper_count, 0, 0);
    tk_inv_hoods_t *new_hoods = tk_inv_hoods_create(L, (uint64_t) keeper_count, 0, 0);
    new_hoods->n = (uint64_t) keeper_count;
    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < uids->n; i ++) {
      if (hoods->a[i]->n >= min) {
        new_uids->a[write_pos] = uids->a[i];
        new_hoods->a[write_pos] = hoods->a[i];
        write_pos ++;
      }
    }

    // Update original arrays in-place to point to new data
    // Save old data pointers for cleanup
    int64_t *old_uids_data = uids->a;
    tk_inv_hood_t *old_hoods_data = hoods->a;

    // Swap in new data
    uids->a = new_uids->a;
    uids->n = (uint64_t) keeper_count;
    uids->m = (uint64_t) keeper_count;
    hoods->a = new_hoods->a;
    hoods->n = (uint64_t) keeper_count;
    hoods->m = (uint64_t) keeper_count;

    // Prevent double-free by clearing new array pointers
    new_uids->a = old_uids_data;
    new_hoods->a = old_hoods_data;

    // Clean up temporary arrays (will free old data)
    lua_remove(L, -2); // remove new_uids (frees old uids data)
    lua_remove(L, -1); // remove new_hoods (frees old hoods data)

    free(old_to_new);
  }

cleanup:
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
  if (hoods_sets) {
    for (uint64_t i = 0; i < uids->n; i ++)
      tk_dumap_destroy(hoods_sets[i]);
    free(hoods_sets);
  }
  if (sids) lua_remove(L, -3); // sids
}

static inline void tk_inv_neighborhoods_by_ids (
  lua_State *L,
  tk_inv_t *I,
  tk_ivec_t *query_ids,
  uint64_t knn,
  double eps,
  uint64_t min,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  bool mutual,
  int64_t rank_filter,
  tk_inv_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (I->destroyed)
    return;
  tk_ivec_t *sids = tk_ivec_create(L, query_ids->n, 0, 0);
  tk_ivec_t *uids = tk_ivec_create(L, query_ids->n, 0, 0);
  tk_ivec_copy(uids, query_ids, 0, (int64_t) query_ids->n, 0);
  for (uint64_t i = 0; i < uids->n; i ++)
    sids->a[i] = tk_inv_uid_sid(I, uids->a[i], false);

  tk_iumap_t *sid_idx = tk_iumap_from_ivec(sids);
  tk_inv_hoods_t *hoods = tk_inv_hoods_create(L, uids->n, 0, 0);
  tk_dumap_t **hoods_sets = NULL;
  if (mutual && knn) {
    hoods_sets = tk_malloc(L, uids->n * sizeof(tk_dumap_t *));
    for (uint64_t i = 0; i < uids->n; i ++)
      hoods_sets[i] = tk_dumap_create();
  }

  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_rvec_create(L, knn, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, -2, -1);
    lua_pop(L, 1);
  }

  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    data->uids = uids;
    data->sids = sids;
    data->query_offsets = NULL;
    data->query_features = NULL;
    tk_dvec_ensure(data->wacc, sids->n * I->n_ranks);
    data->hoods = hoods;
    data->hoods_sets = hoods_sets;
    data->sid_idx = sid_idx;
    data->eps = eps;
    data->knn = knn;
    data->mutual = mutual;
    data->cmp = cmp;
    data->tversky_alpha = tversky_alpha;
    data->tversky_beta = tversky_beta;
    data->rank_filter = rank_filter;
    tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }

  tk_threads_signal(I->pool, TK_INV_NEIGHBORHOODS, 0);
  if (mutual && knn) {
    tk_threads_signal(I->pool, TK_INV_MUTUAL_INIT, 0);
    tk_threads_signal(I->pool, TK_INV_MUTUAL_FILTER, 0);
  }
  tk_iumap_destroy(sid_idx);
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

    for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
      tk_inv_thread_t *data = I->threads + i;
      data->old_to_new = old_to_new;
      data->min = min;
      tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
    }

    tk_threads_signal(I->pool, TK_INV_MIN_REMAP, 0);

    tk_ivec_t *new_uids = tk_ivec_create(L, (uint64_t) keeper_count, 0, 0);
    tk_inv_hoods_t *new_hoods = tk_inv_hoods_create(L, (uint64_t) keeper_count, 0, 0);
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
    tk_inv_hood_t *old_hoods_data = hoods->a;
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
    free(old_to_new);
  }

cleanup:
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
  if (hoods_sets) {
    for (uint64_t i = 0; i < uids->n; i ++)
      tk_dumap_destroy(hoods_sets[i]);
    free(hoods_sets);
  }
  lua_remove(L, -3); // sids
}

static inline void tk_inv_neighborhoods_by_vecs (
  lua_State *L,
  tk_inv_t *I,
  tk_ivec_t *query_vecs,
  uint64_t knn,
  double eps,
  uint64_t min,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter,
  tk_inv_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (I->destroyed)
    return;
  uint64_t n_queries = 0;
  for (uint64_t i = 0; i < query_vecs->n; i ++) {
    int64_t encoded = query_vecs->a[i];
    if (encoded >= 0) {
      uint64_t sample_idx = (uint64_t) encoded / I->features;
      if (sample_idx >= n_queries) n_queries = sample_idx + 1;
    }
  }
  tk_ivec_ensure(I->tmp_query_offsets, n_queries + 1);
  tk_ivec_ensure(I->tmp_query_features, query_vecs->n);
  I->tmp_query_offsets->n = n_queries + 1;
  I->tmp_query_features->n = 0;
  for (uint64_t i = 0; i <= n_queries; i ++)
    I->tmp_query_offsets->a[i] = 0;
  for (uint64_t i = 0; i < query_vecs->n; i ++) {
    int64_t encoded = query_vecs->a[i];
    if (encoded >= 0) {
      uint64_t sample_idx = (uint64_t) encoded / I->features;
      I->tmp_query_offsets->a[sample_idx + 1] ++;
    }
  }
  for (uint64_t i = 1; i <= n_queries; i ++)
    I->tmp_query_offsets->a[i] += I->tmp_query_offsets->a[i - 1];
  tk_ivec_t *write_offsets = tk_ivec_create(L, n_queries, 0, 0); // offsets
  tk_ivec_copy(write_offsets, I->tmp_query_offsets, 0, (int64_t) n_queries, 0);
  for (uint64_t i = 0; i < query_vecs->n; i ++) {
    int64_t encoded = query_vecs->a[i];
    if (encoded >= 0) {
      uint64_t sample_idx = (uint64_t) encoded / I->features;
      int64_t fid = encoded % (int64_t) I->features;
      int64_t write_pos = write_offsets->a[sample_idx]++;
      I->tmp_query_features->a[write_pos] = fid;
    }
  }
  I->tmp_query_features->n = (size_t) I->tmp_query_offsets->a[n_queries];
  lua_pop(L, 1); //
  tk_inv_hoods_t *hoods = tk_inv_hoods_create(L, n_queries, 0, 0); // hoods
  hoods->n = n_queries;
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_rvec_create(L, knn, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, -1, -1);
    lua_pop(L, 1);
  }
  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    data->uids = NULL;  // No pre-built UIDs array
    data->sids = NULL;
    data->query_offsets = I->tmp_query_offsets;
    data->query_features = I->tmp_query_features;
    tk_dvec_ensure(data->wacc, tk_iumap_size(I->uid_sid) * I->n_ranks);
    data->hoods = hoods;
    data->hoods_sets = NULL;
    data->sid_idx = NULL;
    data->eps = eps;
    data->knn = knn;
    data->mutual = false;
    data->cmp = cmp;
    data->tversky_alpha = tversky_alpha;
    data->tversky_beta = tversky_beta;
    data->rank_filter = rank_filter;
    tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }
  tk_threads_signal(I->pool, TK_INV_NEIGHBORHOODS, 0);
  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    I->threads[i].local_uids = tk_iuset_create();
    tk_thread_range(i, I->pool->n_threads, hoods->n, &I->threads[i].ifirst, &I->threads[i].ilast);
  }
  tk_threads_signal(I->pool, TK_INV_COLLECT_UIDS, 0);
  tk_iumap_t *uid_to_idx = tk_iumap_create();
  int64_t next_idx = 0;
  int ret;
  for (uint64_t t = 0; t < I->pool->n_threads; t ++) {
    tk_iuset_t *local = I->threads[t].local_uids;
    int64_t uid;
    tk_iuset_foreach(local, uid, ({
      khint_t k = tk_iumap_put(uid_to_idx, uid, &ret);
      if (ret) // New UID
        tk_iumap_value(uid_to_idx, k) = next_idx++;
    }));
    tk_iuset_destroy(local);
  }
  tk_ivec_t *uids = tk_ivec_create(L, (uint64_t)next_idx, 0, 0); // hoods uids
  uids->n = (uint64_t)next_idx;
  for (khint_t k = tk_iumap_begin(uid_to_idx); k != tk_iumap_end(uid_to_idx); k++) {
    if (tk_iumap_exist(uid_to_idx, k)) {
      int64_t uid = tk_iumap_key(uid_to_idx, k);
      int64_t idx = tk_iumap_value(uid_to_idx, k);
      uids->a[idx] = uid;
    }
  }
  lua_insert(L, -2); // uids hoods
  for (uint64_t i = 0; i < I->pool->n_threads; i ++) {
    I->threads[i].uid_to_idx = uid_to_idx;
    tk_thread_range(i, I->pool->n_threads, hoods->n, &I->threads[i].ifirst, &I->threads[i].ilast);
  }
  tk_threads_signal(I->pool, TK_INV_REMAP_UIDS, 0);
  tk_iumap_destroy(uid_to_idx);
  if (min > 0) {
    int64_t keeper_count = 0;
    for (uint64_t i = 0; i < hoods->n; i ++)
      if (hoods->a[i]->n >= min)
        keeper_count ++;
    if (keeper_count == (int64_t) hoods->n)
      goto cleanup;
    int kha;
    tk_iuset_t *kept_uids = tk_iuset_create();
    for (uint64_t i = 0; i < hoods->n; i ++) {
      if (hoods->a[i]->n >= min) {
        tk_rvec_t *hood = hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j ++) {
          int64_t idx = hood->a[j].i;
          int64_t uid = uids->a[idx];
          tk_iuset_put(kept_uids, uid, &kha);
        }
      }
    }
    tk_iumap_t *idx_remap = tk_iumap_create();
    tk_ivec_t *new_uids = tk_ivec_create(L, (uint64_t) tk_iuset_size(kept_uids), 0, 0); // uids hoods new_uids
    int64_t new_idx = 0;
    int64_t uid;
    tk_iuset_foreach(kept_uids, uid, ({
      new_uids->a[new_idx] = uid;
      for (uint64_t old_idx = 0; old_idx < uids->n; old_idx ++) {
        if (uids->a[old_idx] == uid) {
          int ret;
          khint_t k = tk_iumap_put(idx_remap, (int64_t) old_idx, &ret);
          tk_iumap_value(idx_remap, k) = new_idx;
          break;
        }
      }
      new_idx ++;
    }));
    new_uids->n = (uint64_t) new_idx;
    tk_iuset_destroy(kept_uids);
    tk_inv_hoods_t *new_hoods = tk_inv_hoods_create(L, (uint64_t) keeper_count, 0, 0); // uids hoods new_uids new_hoods
    new_hoods->n = (uint64_t) keeper_count;
    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < hoods->n; i ++) {
      if (hoods->a[i]->n >= min) {
        tk_rvec_t *hood = hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j ++) {
          int64_t old_idx = hood->a[j].i;
          khint_t k = tk_iumap_get(idx_remap, old_idx);
          if (k != tk_iumap_end(idx_remap)) {
            hood->a[j].i = tk_iumap_value(idx_remap, k);
          }
        }
        new_hoods->a[write_pos ++] = hood;
      }
    }
    int64_t *old_uids_data = uids->a;
    tk_inv_hood_t *old_hoods_data = hoods->a;
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

static inline double tk_inv_w (
  tk_dvec_t *W,
  int64_t fid
) {
  if (W == NULL)
    return 1.0;
  assert(fid >= 0 && fid < (int64_t) W->n);
  return W->a[fid];
}

static inline void tk_inv_stats (
  tk_inv_t *I,
  int64_t *a, size_t alen,
  int64_t *b, size_t blen,
  double *inter_w,
  double *sum_a,
  double *sum_b
) {
  size_t i = 0, j = 0;
  double inter = 0.0, sa = 0.0, sb = 0.0;
  while (i < alen && j < blen) {
    int64_t ai = a[i], bj = b[j];
    if (ai == bj) {
      double w = tk_inv_w(I->weights, ai);
      inter += w;
      sa += w;
      sb += w;
      i ++;
      j ++;
    } else if (ai < bj) {
      sa += tk_inv_w(I->weights, ai);
      i ++;
    } else {
      sb += tk_inv_w(I->weights, bj);
      j ++;
    }
  }
  while (i < alen)
    sa += tk_inv_w(I->weights, a[i ++]);
  while (j < blen)
    sb += tk_inv_w(I->weights, b[j ++]);
  *inter_w = inter;
  *sum_a = sa;
  *sum_b = sb;
}

static inline double tk_inv_tversky_from_stats (double inter_w, double sa, double sb, double alpha, double beta)
{
  double a_only = sa - inter_w;
  double b_only = sb - inter_w;
  if (a_only < 0.0) a_only = 0.0;
  if (b_only < 0.0) b_only = 0.0;
  double denom = inter_w + alpha * a_only + beta * b_only;
  return (denom == 0.0) ? 0.0 : inter_w / denom;
}

#define tk_inv_similarity_partial tk_ivec_set_similarity_from_partial

static inline double tk_inv_similarity_with_buffers (
  tk_inv_t *I,
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
) {
  if (I->ranks && I->n_ranks > 1 && q_weights && e_weights && inter_weights) {
    tk_dvec_ensure(q_weights, I->n_ranks);
    tk_dvec_ensure(e_weights, I->n_ranks);
    tk_dvec_ensure(inter_weights, I->n_ranks);

    for (uint64_t r = 0; r < I->n_ranks; r ++) {
      q_weights->a[r] = 0.0;
      e_weights->a[r] = 0.0;
      inter_weights->a[r] = 0.0;
    }
    size_t i = 0, j = 0;
    while (i < asize && j < bsize) {
      if (abits[i] == bbits[j]) {
        int64_t fid = abits[i];
        int64_t rank = (fid >= 0 && fid < (int64_t)I->features && I->ranks) ? I->ranks->a[fid] : 0;
        if (rank >= 0 && rank < (int64_t)I->n_ranks) {
          double w = tk_inv_w(I->weights, fid);
          inter_weights->a[rank] += w;
          q_weights->a[rank] += w;
          e_weights->a[rank] += w;
        }
        i ++; j ++;
      } else if (abits[i] < bbits[j]) {
        int64_t fid = abits[i];
        int64_t rank = (fid >= 0 && fid < (int64_t)I->features && I->ranks) ? I->ranks->a[fid] : 0;
        if (rank >= 0 && rank < (int64_t)I->n_ranks) {
          q_weights->a[rank] += tk_inv_w(I->weights, fid);
        }
        i ++;
      } else {
        int64_t fid = bbits[j];
        int64_t rank = (fid >= 0 && fid < (int64_t)I->features && I->ranks) ? I->ranks->a[fid] : 0;
        if (rank >= 0 && rank < (int64_t)I->n_ranks) {
          e_weights->a[rank] += tk_inv_w(I->weights, fid);
        }
        j ++;
      }
    }
    while (i < asize) {
      int64_t fid = abits[i];
      int64_t rank = (fid >= 0 && fid < (int64_t)I->features && I->ranks) ? I->ranks->a[fid] : 0;
      if (rank >= 0 && rank < (int64_t)I->n_ranks) {
        q_weights->a[rank] += tk_inv_w(I->weights, fid);
      }
      i ++;
    }
    while (j < bsize) {
      int64_t fid = bbits[j];
      int64_t rank = (fid >= 0 && fid < (int64_t)I->features && I->ranks) ? I->ranks->a[fid] : 0;
      if (rank >= 0 && rank < (int64_t)I->n_ranks) {
        e_weights->a[rank] += tk_inv_w(I->weights, fid);
      }
      j ++;
    }
    double total_weighted_sim = 0.0;
    for (uint64_t rank = 0; rank < I->n_ranks; rank ++) {
      double rank_weight = I->rank_weights->a[rank];
      double rank_sim = tk_inv_similarity_partial(
        inter_weights->a[rank],
        q_weights->a[rank],
        e_weights->a[rank],
        cmp, tversky_alpha, tversky_beta);
      total_weighted_sim += rank_sim * rank_weight;
    }

    return (I->total_rank_weight > 0.0) ? total_weighted_sim / I->total_rank_weight : 0.0;
  }
  double inter_w = 0.0, sa = 0.0, sb = 0.0;
  tk_inv_stats(I, abits, asize, bbits, bsize, &inter_w, &sa, &sb);
  switch (cmp) {
    case TK_IVEC_JACCARD: {
      double u = sa + sb - inter_w;
      return (u == 0.0) ? 0.0 : inter_w / u;
    }
    case TK_IVEC_OVERLAP: {
      double m = (sa < sb) ? sa : sb;
      return (m == 0.0) ? 0.0 : inter_w / m;
    }
    case TK_IVEC_TVERSKY: {
      return tk_inv_tversky_from_stats(inter_w, sa, sb, tversky_alpha, tversky_beta);
    }
    case TK_IVEC_DICE: {
      double denom = sa + sb;
      return (denom == 0.0) ? 0.0 : (2.0 * inter_w) / denom;
    }
    default: { // fallback to Jaccard
      double u = sa + sb - inter_w;
      return (u == 0.0) ? 0.0 : inter_w / u;
    }
  }
}

static inline double tk_inv_similarity (
  tk_inv_t *I,
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta
) {
  return tk_inv_similarity_with_buffers(
    I, abits, asize, bbits, bsize, cmp, tversky_alpha, tversky_beta,
    I->tmp_q_weights, I->tmp_e_weights, I->tmp_inter_weights);
}

static inline void tk_inv_compute_query_weights_by_rank (
  tk_inv_t *I,
  int64_t *data,
  size_t datalen,
  double *q_weights_by_rank
) {
  for (uint64_t r = 0; r < I->n_ranks; r ++)
    q_weights_by_rank[r] = 0.0;
  for (size_t i = 0; i < datalen; i ++) {
    int64_t fid = data[i];
    if (fid >= 0 && fid < (int64_t) I->features) {
      int64_t rank = I->ranks ? I->ranks->a[fid] : 0;
      if (rank >= 0 && rank < (int64_t) I->n_ranks) {
        q_weights_by_rank[rank] += tk_inv_w(I->weights, fid);
      }
    }
  }
}

static inline void tk_inv_compute_candidate_weights_by_rank (
  tk_inv_t *I,
  int64_t *features,
  size_t nfeatures,
  double *e_weights_by_rank
) {
  for (uint64_t r = 0; r < I->n_ranks; r ++)
    e_weights_by_rank[r] = 0.0;
  for (size_t i = 0; i < nfeatures; i ++) {
    int64_t fid = features[i];
    if (fid >= 0 && fid < (int64_t) I->features) {
      int64_t rank = I->ranks ? I->ranks->a[fid] : 0;
      if (rank >= 0 && rank < (int64_t) I->n_ranks) {
        e_weights_by_rank[rank] += tk_inv_w(I->weights, fid);
      }
    }
  }
}

static inline double tk_inv_similarity_by_rank (
  tk_inv_t *I,
  tk_dvec_t *wacc,
  int64_t vsid,
  double *q_weights_by_rank,
  double *e_weights_by_rank,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta
) {
  double total_weighted_sim = 0.0;
  for (uint64_t rank = 0; rank < I->n_ranks; rank ++) {
    double rank_weight = I->rank_weights->a[rank];
    double inter_w = wacc->a[(int64_t) I->n_ranks * vsid + (int64_t) rank];
    double q_w = q_weights_by_rank[rank];
    double e_w = e_weights_by_rank[rank];
    double rank_sim;
    if (q_w > 0.0 || e_w > 0.0) {
      rank_sim = tk_inv_similarity_partial(inter_w, q_w, e_w, cmp, tversky_alpha, tversky_beta);
    } else {
      rank_sim = 0.0;
    }
    total_weighted_sim += rank_sim * rank_weight;
  }
  return (I->total_rank_weight > 0.0) ? total_weighted_sim / I->total_rank_weight : 0.0;
}

static inline double tk_inv_similarity_rank_filtered (
  tk_inv_t *I,
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter
) {
  if (rank_filter < 0)
    return tk_inv_similarity(I, abits, asize, bbits, bsize, cmp, tversky_alpha, tversky_beta);
  double filtered_inter_w = 0.0, filtered_sa = 0.0, filtered_sb = 0.0;
  size_t i = 0, j = 0;
  while (i < asize && j < bsize) {
    if (abits[i] == bbits[j]) {
      int64_t fid = abits[i];
      int64_t rank = (I->ranks && fid >= 0 && fid < (int64_t)I->features) ? I->ranks->a[fid] : 0;
      if (rank == rank_filter) {
        double w = tk_inv_w(I->weights, fid);
        filtered_inter_w += w;
        filtered_sa += w;
        filtered_sb += w;
      }
      i++; j++;
    } else if (abits[i] < bbits[j]) {
      int64_t fid = abits[i];
      int64_t rank = (I->ranks && fid >= 0 && fid < (int64_t)I->features) ? I->ranks->a[fid] : 0;
      if (rank == rank_filter)
        filtered_sa += tk_inv_w(I->weights, fid);
      i++;
    } else {
      int64_t fid = bbits[j];
      int64_t rank = (I->ranks && fid >= 0 && fid < (int64_t)I->features) ? I->ranks->a[fid] : 0;
      if (rank == rank_filter)
        filtered_sb += tk_inv_w(I->weights, fid);
      j++;
    }
  }
  while (i < asize) {
    int64_t fid = abits[i];
    int64_t rank = (I->ranks && fid >= 0 && fid < (int64_t)I->features) ? I->ranks->a[fid] : 0;
    if (rank == rank_filter)
      filtered_sa += tk_inv_w(I->weights, fid);
    i++;
  }
  while (j < bsize) {
    int64_t fid = bbits[j];
    int64_t rank = (I->ranks && fid >= 0 && fid < (int64_t)I->features) ? I->ranks->a[fid] : 0;
    if (rank == rank_filter)
      filtered_sb += tk_inv_w(I->weights, fid);
    j++;
  }
  double rank_sim;
  switch (cmp) {
    case TK_IVEC_JACCARD: {
      double u = filtered_sa + filtered_sb - filtered_inter_w;
      rank_sim = (u == 0.0) ? 0.0 : filtered_inter_w / u;
      break;
    }
    case TK_IVEC_OVERLAP: {
      double m = (filtered_sa < filtered_sb) ? filtered_sa : filtered_sb;
      rank_sim = (m == 0.0) ? 0.0 : filtered_inter_w / m;
      break;
    }
    case TK_IVEC_TVERSKY: {
      rank_sim = tk_inv_tversky_from_stats(filtered_inter_w, filtered_sa, filtered_sb, tversky_alpha, tversky_beta);
      break;
    }
    case TK_IVEC_DICE: {
      double denom = filtered_sa + filtered_sb;
      rank_sim = (denom == 0.0) ? 0.0 : (2.0 * filtered_inter_w) / denom;
      break;
    }
    default: { // fallback to Jaccard
      double u = filtered_sa + filtered_sb - filtered_inter_w;
      rank_sim = (u == 0.0) ? 0.0 : filtered_inter_w / u;
      break;
    }
  }

  double rank_weight = I->rank_weights->a[rank_filter];
  return (I->total_rank_weight > 0.0) ? (rank_sim * rank_weight) / I->total_rank_weight : 0.0;
}

static inline double tk_inv_distance (
  tk_inv_t *I,
  int64_t uid0,
  int64_t uid1,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta
) {
  size_t n0 = 0, n1 = 0;
  int64_t *v0 = tk_inv_get(I, uid0, &n0);
  if (v0 == NULL)
    return 1.0;
  int64_t *v1 = tk_inv_get(I, uid1, &n1);
  if (v1 == NULL)
    return 1.0;
  double sim = tk_inv_similarity(I, v0, n0, v1, n1, cmp, tversky_alpha, tversky_beta);
  return 1.0 - sim;
}

static inline tk_rvec_t *tk_inv_neighbors_by_vec (
  tk_inv_t *I,
  int64_t *data,
  size_t datalen,
  int64_t sid0,
  uint64_t knn,
  double eps,
  tk_rvec_t *out,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter
) {
  if (datalen == 0) {
    tk_rvec_clear(out);
    return out;
  }

  tk_rvec_clear(out);
  size_t n_sids = I->node_offsets->n;
  double *q_weights_by_rank = I->tmp_q_weights->a;
  tk_inv_compute_query_weights_by_rank(I, data, datalen, q_weights_by_rank);

  tk_dvec_ensure(I->wacc, n_sids * I->n_ranks);  tk_ivec_clear(I->touched);

  for (size_t i = 0; i < datalen; i ++) {
    int64_t fid = data[i];
    if (fid < 0 || fid >= (int64_t) I->postings->n)
      continue;

    int64_t rank = I->ranks ? I->ranks->a[fid] : 0;

    if (rank_filter >= 0 && rank != rank_filter)
      continue;
    double wf = tk_inv_w(I->weights, fid);
    tk_ivec_t *vsids = I->postings->a[fid];
    for (uint64_t j = 0; j < vsids->n; j ++) {
      int64_t vsid = vsids->a[j];
      if (vsid == sid0)
        continue;
      if (I->wacc->a[(int64_t) I->n_ranks * vsid + rank] == 0.0)
        tk_ivec_push(I->touched, vsid);
      I->wacc->a[(int64_t) I->n_ranks * vsid + rank] += wf;
    }
  }
  double *e_weights_by_rank = I->tmp_e_weights->a;

  for (uint64_t i = 0; i < I->touched->n; i ++) {
    int64_t vsid = I->touched->a[i];

    if (knn && out->n >= knn) {
      double max_sim = 0.0;
      for (uint64_t r = 0; r < I->n_ranks; r++) {
        double inter = I->wacc->a[(int64_t) I->n_ranks * vsid + (int64_t) r];
        if (inter > 0.0)
          max_sim += inter * I->rank_weights->a[r];
      }
      max_sim = (I->total_rank_weight > 0.0) ? max_sim / I->total_rank_weight : 0.0;

      if (1.0 - max_sim > out->a[0].d) {
        for (uint64_t r = 0; r < I->n_ranks; r ++)
          I->wacc->a[(int64_t) I->n_ranks * vsid + (int64_t) r] = 0.0;
        continue;
      }
    }

    size_t elen = 0;
    int64_t *ev = tk_inv_sget(I, vsid, &elen);
    tk_inv_compute_candidate_weights_by_rank(I, ev, elen, e_weights_by_rank);
    double sim;
    if (rank_filter >= 0) {
      sim = tk_inv_similarity_rank_filtered(I, data, datalen, ev, elen, cmp, tversky_alpha, tversky_beta, rank_filter);
    } else {
      sim = tk_inv_similarity_by_rank(I, I->wacc, vsid, q_weights_by_rank, e_weights_by_rank, cmp, tversky_alpha, tversky_beta);
    }
    double dist = 1.0 - sim;
    double current_cutoff = (knn && out->n >= knn) ? out->a[0].d : eps;
    if (dist <= current_cutoff) {
      int64_t vuid = tk_inv_sid_uid(I, vsid);
      if (vuid >= 0) {
        if (knn)
          tk_rvec_hmax(out, knn, tk_rank(vuid, dist));
        else
          tk_rvec_push(out, tk_rank(vuid, dist));
      }
    }    for (uint64_t r = 0; r < I->n_ranks; r ++)
      I->wacc->a[(int64_t) I->n_ranks * vsid + (int64_t) r] = 0.0;
  }

  tk_rvec_asc(out, 0, out->n);

  return out;
}

static inline tk_rvec_t *tk_inv_neighbors_by_id (
  tk_inv_t *I,
  int64_t uid,
  uint64_t knn,
  double eps,
  tk_rvec_t *out,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter
) {
  int64_t sid0 = tk_inv_uid_sid(I, uid, false);
  if (sid0 < 0) {
    tk_rvec_clear(out);
    return out;
  }
  size_t len = 0;
  int64_t *data = tk_inv_get(I, uid, &len);
  return tk_inv_neighbors_by_vec(I, data, len, sid0, knn, eps, out, cmp, tversky_alpha, tversky_beta, rank_filter);
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
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t id = tk_lua_checkinteger(L, 2, "id");
    tk_inv_remove(L, I, id);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
    for (uint64_t i = 0; i < ids->n; i ++) {
      tk_inv_uid_remove(I, ids->a[i]);
    }
  }
  return 0;
}

static inline int tk_inv_keep_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t id = tk_lua_checkinteger(L, 2, "id");
    tk_ivec_t *ids = tk_ivec_create(L, 1, 0, 0);
    ids->a[0] = id;
    ids->n = 1;
    tk_inv_keep(L, I, ids);
    lua_pop(L, 1);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
    tk_inv_keep(L, I, ids);
  }
  return 0;
}

static inline int tk_inv_get_lua (lua_State *L)
{
  lua_settop(L, 4);
  tk_inv_t *I = tk_inv_peek(L, 1);
  int64_t uid = -1;
  tk_ivec_t *uids = NULL;
  tk_ivec_t *out = tk_ivec_peekopt(L, 3);
  out = out == NULL ? tk_ivec_create(L, 0, 0, 0) : out; // out
  bool append = tk_lua_optboolean(L, 4, "append", false);
  if (!append)
    tk_ivec_clear(out);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    uid = tk_lua_checkinteger(L, 2, "id");
    size_t n = 0;
    int64_t *data = tk_inv_get(I, uid, &n);
    if (!n)
      return 1;
    tk_ivec_ensure(out, n);
    memcpy(out->a, data, n * sizeof(int64_t));
    out->n = n;
  } else {
    uids = tk_ivec_peek(L, 2, "uids");
    size_t total_size = 0;
    for (uint64_t i = 0; i < uids->n; i ++) {
      uid = uids->a[i];
      size_t n = 0;
      tk_inv_get(I, uid, &n);
      total_size += n;
    }    if (total_size > 0) {
      tk_ivec_ensure(out, out->n + total_size);
    }    for (uint64_t i = 0; i < uids->n; i ++) {
      uid = uids->a[i];
      size_t n = 0;
      int64_t *data = tk_inv_get(I, uid, &n);
      if (!n)
        continue;  // No features for this sample - empty set represented by no indices
      for (size_t j = 0; j < n; j ++)
        out->a[out->n ++] = data[j] + (int64_t) (i * I->features);
    }
  }
  return 1;
}

static inline int tk_inv_neighborhoods_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  uint64_t knn = tk_lua_checkunsigned(L, 2, "knn");
  double eps = tk_lua_optposdouble(L, 3, "eps", 1.0);
  uint64_t min = tk_lua_optunsigned(L, 4, "min", 0);
  const char *typ = tk_lua_optstring(L, 5, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 6, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 7, "beta", 0.1);
  bool mutual = tk_lua_optboolean(L, 8, "mutual", false);
  int64_t rank_filter = lua_isnil(L, 9) ? -1 : (int64_t) tk_lua_checkunsigned(L, 9, "rank_filter");

  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_IVEC_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_IVEC_TVERSKY;
  else
    tk_lua_verror(L, 3, "neighborhoods", "invalid comparator specified", typ);

  tk_inv_neighborhoods(L, I, knn, eps, min, cmp, tversky_alpha, tversky_beta, mutual, rank_filter, NULL, NULL);
  return 2;
}

static inline int tk_inv_neighborhoods_by_ids_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_ivec_t *query_ids = tk_ivec_peek(L, 2, "ids");
  uint64_t knn = tk_lua_checkunsigned(L, 3, "knn");
  double eps = tk_lua_optposdouble(L, 4, "eps", 1.0);
  uint64_t min = tk_lua_optunsigned(L, 5, "min", 0);
  const char *typ = tk_lua_optstring(L, 6, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 7, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 8, "beta", 0.1);
  bool mutual = tk_lua_optboolean(L, 9, "mutual", false);
  int64_t rank_filter = lua_isnil(L, 10) ? -1 : (int64_t) tk_lua_checkunsigned(L, 10, "rank_filter");

  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_IVEC_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_IVEC_TVERSKY;
  else
    tk_lua_verror(L, 3, "neighborhoods_by_ids", "invalid comparator specified", typ);
  int64_t write_pos = 0;
  for (int64_t i = 0; i < (int64_t) query_ids->n; i ++) {
    int64_t uid = query_ids->a[i];
    khint_t k = tk_iumap_get(I->uid_sid, uid);
    if (k != tk_iumap_end(I->uid_sid)) {
      query_ids->a[write_pos ++] = uid;
    }
  }
  query_ids->n = (uint64_t) write_pos;

  tk_inv_hoods_t *hoods;
  tk_inv_neighborhoods_by_ids(L, I, query_ids, knn, eps, min, cmp, tversky_alpha, tversky_beta, mutual, rank_filter, &hoods, &query_ids);
  return 2;
}

static inline int tk_inv_neighborhoods_by_vecs_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_ivec_t *query_vecs = tk_ivec_peek(L, 2, "vectors");
  uint64_t knn = tk_lua_checkunsigned(L, 3, "knn");
  double eps = tk_lua_optposdouble(L, 4, "eps", 1.0);
  uint64_t min = tk_lua_optunsigned(L, 5, "min", 0);
  const char *typ = tk_lua_optstring(L, 6, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 7, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 8, "beta", 0.1);
  int64_t rank_filter = lua_isnil(L, 9) ? -1 : (int64_t) tk_lua_checkunsigned(L, 9, "rank_filter");

  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_IVEC_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_IVEC_TVERSKY;
  else
    tk_lua_verror(L, 3, "neighborhoods_by_vecs", "invalid comparator specified", typ);

  tk_inv_neighborhoods_by_vecs(L, I, query_vecs, knn, eps, min, cmp, tversky_alpha, tversky_beta, rank_filter, NULL, NULL);
  return 2;
}

static inline int tk_inv_similarity_lua (lua_State *L)
{
  lua_settop(L, 6);
  tk_inv_t *I = tk_inv_peek(L, 1);
  const char *typ = tk_lua_optstring(L, 4, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 5, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 6, "beta", 0.1);
  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_IVEC_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_IVEC_TVERSKY;
  else
    tk_lua_verror(L, 3, "similarity", "invalid comparator specified", typ);
  int64_t uid0 = tk_lua_checkinteger(L, 2, "uid0");
  int64_t uid1 = tk_lua_checkinteger(L, 3, "uid1");
  lua_pushnumber(L, 1.0 - tk_inv_distance(I, uid0, uid1, cmp, tversky_alpha, tversky_beta));
  return 1;
}

static inline int tk_inv_distance_lua (lua_State *L)
{
  lua_settop(L, 6);
  tk_inv_t *I = tk_inv_peek(L, 1);
  const char *typ = tk_lua_optstring(L, 4, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 5, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 6, "beta", 0.1);
  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_IVEC_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_IVEC_TVERSKY;
  else
    tk_lua_verror(L, 3, "distance", "invalid comparator specified", typ);
  int64_t uid0 = tk_lua_checkinteger(L, 2, "uid0");
  int64_t uid1 = tk_lua_checkinteger(L, 3, "uid1");
  lua_pushnumber(L, tk_inv_distance(I, uid0, uid1, cmp, tversky_alpha, tversky_beta));
  return 1;
}

static inline int tk_inv_neighbors_lua (lua_State *L)
{
  lua_settop(L, 9);
  tk_inv_t *I = tk_inv_peek(L, 1);
  uint64_t knn = tk_lua_optunsigned(L, 3, "knn", 0);
  double eps = tk_lua_optposdouble(L, 4, "eps", 1.0);
  tk_rvec_t *out = tk_rvec_peek(L, 5, "out");
  const char *typ = tk_lua_optstring(L, 6, "comparator", "jaccard");
  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  double tversky_alpha = tk_lua_optnumber(L, 7, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 8, "beta", 0.1);
  int64_t rank_filter = lua_isnil(L, 9) ? -1 : (int64_t) tk_lua_checkunsigned(L, 9, "rank_filter");
  if (!strcmp(typ, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_IVEC_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_IVEC_TVERSKY;
  else
    tk_lua_verror(L, 3, "neighbors", "invalid comparator specified", typ);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_inv_neighbors_by_id(I, uid, knn, eps, out, cmp, tversky_alpha, tversky_beta, rank_filter);
  } else {
    tk_ivec_t *vec = tk_ivec_peek(L, 2, "vector");
    tk_inv_neighbors_by_vec(I, vec->a, vec->n, -1, knn, eps, out, cmp, tversky_alpha, tversky_beta, rank_filter);
  }
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
  return 1;
}

static inline int tk_inv_features_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) I->features);
  return 1;
}

static inline int tk_inv_weights_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_lua_get_ephemeron(L, TK_INV_EPH, I->weights);
  return 1;
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
  tk_inv_shrink(L, I);
  return 0;
}

static inline tk_ivec_t *tk_inv_ids (lua_State *L, tk_inv_t *I)
{
  return tk_iumap_keys(L, I->uid_sid);
}

static inline int tk_inv_weight_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  uint64_t fid = tk_lua_checkunsigned(L, 2, "fid");
  lua_pushnumber(L, tk_inv_w(I->weights, (int64_t) fid));
  return 1;
}

static inline int tk_inv_ids_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  tk_iumap_keys(L, I->uid_sid);
  return 1;
}

static inline void tk_inv_worker (void *dp, int sig)
{
  tk_inv_stage_t stage = (tk_inv_stage_t) sig;
  tk_inv_thread_t *data = (tk_inv_thread_t *) dp;
  tk_inv_t *I = data->I;
  double tversky_alpha = data->tversky_alpha;
  double tversky_beta = data->tversky_beta;
  tk_ivec_t *touched = data->touched;
  tk_inv_hoods_t *hoods = data->hoods;
  tk_dumap_t **hoods_sets = data->hoods_sets;
  tk_ivec_t *sids = data->sids;
  tk_iumap_t *sid_idx = data->sid_idx;
  tk_dvec_t *wacc = data->wacc;
  double eps = data->eps;
  uint64_t knn = data->knn;
  tk_ivec_sim_type_t cmp = data->cmp;
  khint_t khi;
  int64_t usid, vsid, fid, iv;
  int64_t *ubits, *vbits;
  size_t nubits, nvbits;
  tk_rvec_t *uhood;
  tk_dumap_t *uset;
  tk_dumap_t *vset;
  tk_ivec_t *vsids;
  switch (stage) {

    case TK_INV_NEIGHBORHOODS:
      touched->n = 0;
      // wacc size already ensured during setup - no need to check again
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {
        uhood = hoods->a[i];
        tk_rvec_clear(uhood);

        // Get features either from query vectors or from stored data
        if (data->query_offsets) {
          // Using query vectors - get features from pre-processed arrays
          int64_t start = data->query_offsets->a[i];
          int64_t end = (i + 1 < (int64_t) data->query_offsets->n)
                        ? data->query_offsets->a[i + 1]
                        : (int64_t) data->query_features->n;
          ubits = data->query_features->a + start;
          nubits = (size_t)(end - start);
          usid = -1;  // No SID for query vectors
        } else {
          // Using stored data
          usid = sids->a[i];
          if (tk_iumap_get(I->sid_uid, usid) == tk_iumap_end(I->sid_uid))
            continue;
          ubits = tk_inv_sget(I, usid, &nubits);
        }

        // Removed dead code - start/end were computed but never used

        // Use thread-local buffer for query weights (already sized correctly)
        double *q_weights_by_rank = data->q_weights_buf->a;
        tk_inv_compute_query_weights_by_rank(I, ubits, nubits, q_weights_by_rank);

        // Total query weight computation removed - was unused
        for (size_t k = 0; k < nubits; k ++) {
          fid = ubits[k];
          int64_t rank = I->ranks ? I->ranks->a[fid] : 0;
                if (data->rank_filter >= 0 && rank != data->rank_filter)
            continue;
          double wf = tk_inv_w(I->weights, fid);
          // Bounds check removed - ensured at insertion time
          vsids = I->postings->a[fid];
          for (uint64_t l = 0; l < vsids->n; l ++) {
            vsid = vsids->a[l];
            if (vsid == usid)
              continue;
            khi = tk_iumap_get(sid_idx, vsid);
            if (khi == tk_iumap_end(sid_idx))
              continue;
            iv = tk_iumap_value(sid_idx, khi);
            // Bounds check removed - sid_idx guarantees valid indices
            if (wacc->a[(int64_t) I->n_ranks * iv + rank] == 0.0)
              tk_ivec_push(touched, iv);
            wacc->a[(int64_t) I->n_ranks * iv + rank] += wf;
          }
        }
        // Use thread-local buffer for candidate weights (already sized correctly)
        double *e_weights_by_rank = data->e_weights_buf->a;

        // Get cutoff distance for early termination if using KNN
        double cutoff = (knn && uhood->n >= knn) ? uhood->a[0].d : eps;

        for (uint64_t ti = 0; ti < touched->n; ti ++) {
          iv = touched->a[ti];

          // Early termination: check if intersection alone can beat cutoff
          if (knn && uhood->n >= knn) {
            // Compute upper bound on similarity from intersection weights
            double max_possible_sim = 0.0;
            for (uint64_t r = 0; r < I->n_ranks; r++) {
              double inter = wacc->a[(int64_t) I->n_ranks * iv + (int64_t) r];
              if (inter > 0.0) {
                // Best case: all intersection, no difference
                max_possible_sim += inter * I->rank_weights->a[r];
              }
            }
            max_possible_sim = (I->total_rank_weight > 0.0) ? max_possible_sim / I->total_rank_weight : 0.0;

            // Skip if even perfect similarity can't beat cutoff
            if (1.0 - max_possible_sim > cutoff)
              continue;
          }

          vsid = sids->a[iv];
          vbits = tk_inv_sget(I, vsid, &nvbits);

          // Compute similarity (with optional rank filtering)
          double sim;
          if (data->rank_filter >= 0) {
            sim = tk_inv_similarity_rank_filtered(I, ubits, nubits, vbits, nvbits, cmp, tversky_alpha, tversky_beta, data->rank_filter);
          } else {
            // Compute candidate weights by rank
            tk_inv_compute_candidate_weights_by_rank(I, vbits, nvbits, e_weights_by_rank);
            sim = tk_inv_similarity_by_rank(I, wacc, iv, q_weights_by_rank, e_weights_by_rank, cmp, tversky_alpha, tversky_beta);
          }
          double dist = 1.0 - sim;
          if (dist <= cutoff) {
            if (knn) {
              tk_rvec_hmax(uhood, knn, tk_rank(iv, dist));
              // Update cutoff if heap is full
              if (uhood->n >= knn)
                cutoff = uhood->a[0].d;
            } else {
              tk_rvec_push(uhood, tk_rank(iv, dist));
            }
          }
        }
        // Clear only the accumulator entries we touched
        for (uint64_t ti = 0; ti < touched->n; ti ++)
          for (uint64_t r = 0; r < I->n_ranks; r ++)
            wacc->a[(int64_t) I->n_ranks * touched->a[ti] + (int64_t) r] = 0.0;
        tk_rvec_asc(uhood, 0, uhood->n);
        // Don't shrink - causes unnecessary reallocation
        touched->n = 0;

        // No cleanup needed - we're using reusable buffers
      }
      break;

    // Build hood sets for mutualization
    case TK_INV_MUTUAL_INIT: {
      int kha;
      khint_t khi;
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {
        uhood = hoods->a[i];
        uset = hoods_sets[i];
        for (uint64_t j = 0; j < uhood->n; j ++) {
          khi = tk_dumap_put(uset, uhood->a[j].i, &kha);
          tk_dumap_value(uset, khi) = uhood->a[j].d;
        }
      }
      break;
    }

    // NOTE: Move non-mutual neighbors to tail of list [hood->n, hood->m - 1]
    case TK_INV_MUTUAL_FILTER: {

      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {

        uhood = hoods->a[i];
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
          double d = uhood->a[left].d;
          assert(iv >= 0 && (uint64_t) iv < hoods->n);
          vset = hoods_sets[iv];
          khi = tk_dumap_get(vset, i);
          if (khi != tk_dumap_end(vset)) {
            double d0 = tk_dumap_value(vset, khi);
            if (d0 < d)
              uhood->a[left].d = d0;
            left ++;
          } else {
            if (left != right) {
              tk_rank_t tmp = uhood->a[left];
              uhood->a[left] = uhood->a[right];
              uhood->a[right] = tmp;
            }
            if (right == 0)
              break;
            right --;
          }
        }

        uhood->n = left;
        uhood->m = orig_n;
        assert(uhood->n <= uhood->m);

        // Make non-mutual distances symmetric
        for (uint64_t qi = uhood->n; qi < uhood->m; qi ++) {
          int64_t iv = uhood->a[qi].i;
          double d_forward = uhood->a[qi].d;
          double d_reverse = d_forward;  // fallback

          // Try to get reverse distance from neighbor's hood set, if not found,
          // fall back to full distance computation
          vset = hoods_sets[iv];
          khi = tk_dumap_get(vset, i);
          if (khi != tk_dumap_end(vset)) {
            d_reverse = tk_dumap_value(vset, khi);
          } else {
            int64_t usid = sids->a[i];
            int64_t vsid = sids->a[iv];
            size_t ulen = 0, vlen = 0;
            int64_t *ubits = tk_inv_sget(I, usid, &ulen);
            int64_t *vbits = tk_inv_sget(I, vsid, &vlen);
            if (ubits && vbits) {
              // Use thread-local buffers for thread safety
              double sim = tk_inv_similarity_with_buffers(I, vbits, vlen, ubits, ulen, cmp,
                                                           tversky_alpha, tversky_beta,
                                                           data->q_weights_buf, data->e_weights_buf, data->inter_weights_buf);
              d_reverse = 1.0 - sim;
            }
          }

          // Store minimum distance
          uhood->a[qi].d = (d_forward < d_reverse) ? d_forward : d_reverse;
        }

        // Re-sort both sections (mutual distances may have been updated)
        tk_rvec_asc(uhood, 0, uhood->n);
        tk_rvec_asc(uhood, uhood->n, uhood->m);
      }
      break;
    }

    case TK_INV_MIN_REMAP: {
      // Remap neighborhood indices for keeper UIDs
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {
        if (hoods->a[i]->n >= data->min) {
          tk_rvec_t *hood = hoods->a[i];
          uint64_t mutual_write_pos = 0;
          uint64_t non_mutual_write_pos = 0;

          // Process mutual neighbors (0 to n-1)
          for (uint64_t j = 0; j < hood->n; j ++) {
            int64_t old_neighbor_idx = hood->a[j].i;
            int64_t new_neighbor_idx = data->old_to_new[old_neighbor_idx];

            if (new_neighbor_idx >= 0) {
              // Keep this mutual neighbor, update index
              hood->a[mutual_write_pos ++] = tk_rank(new_neighbor_idx, hood->a[j].d);
            }
            // Otherwise skip (neighbor was filtered out)
          }

          // Process non-mutual neighbors (n to m-1)
          for (uint64_t j = hood->n; j < hood->m; j ++) {
            int64_t old_neighbor_idx = hood->a[j].i;
            int64_t new_neighbor_idx = data->old_to_new[old_neighbor_idx];

            if (new_neighbor_idx >= 0) {
              // Keep this non-mutual neighbor, update index
              hood->a[mutual_write_pos + non_mutual_write_pos ++] = tk_rank(new_neighbor_idx, hood->a[j].d);
            }
            // Otherwise skip (neighbor was filtered out)
          }

          hood->n = mutual_write_pos;  // Update mutual count after filtering
          hood->m = mutual_write_pos + non_mutual_write_pos;  // Update total count after filtering
        }
      }
      break;
    }

    case TK_INV_COLLECT_UIDS: {
      int kha;
      for (uint64_t i = data->ifirst; i <= data->ilast; i++) {
        tk_rvec_t *hood = data->hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j++) {
          int64_t uid = hood->a[j].i;
          tk_iuset_put(data->local_uids, uid, &kha);
        }
      }
      break;
    }

    case TK_INV_REMAP_UIDS: {
      for (uint64_t i = data->ifirst; i <= data->ilast; i++) {
        tk_rvec_t *hood = data->hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j++) {
          int64_t uid = hood->a[j].i;
          khint_t k = tk_iumap_get(data->uid_to_idx, uid);
          assert(k != tk_iumap_end(data->uid_to_idx));
          int64_t idx = tk_iumap_value(data->uid_to_idx, k);
          hood->a[j].i = idx;
        }
      }
      break;
    }

  }
}

static luaL_Reg tk_inv_lua_mt_fns[] =
{
  { "add", tk_inv_add_lua },
  { "remove", tk_inv_remove_lua },
  { "keep", tk_inv_keep_lua },
  { "get", tk_inv_get_lua },
  { "neighborhoods", tk_inv_neighborhoods_lua },
  { "neighborhoods_by_ids", tk_inv_neighborhoods_by_ids_lua },
  { "neighborhoods_by_vecs", tk_inv_neighborhoods_by_vecs_lua },
  { "neighbors", tk_inv_neighbors_lua },
  { "distance", tk_inv_distance_lua },
  { "similarity", tk_inv_similarity_lua },
  { "size", tk_inv_size_lua },
  { "threads", tk_inv_threads_lua },
  { "features", tk_inv_features_lua },
  { "weights", tk_inv_weights_lua },
  { "persist", tk_inv_persist_lua },
  { "destroy", tk_inv_destroy_lua },
  { "shrink", tk_inv_shrink_lua },
  { "ids", tk_inv_ids_lua },
  { "weight", tk_inv_weight_lua },
  { NULL, NULL }
};

static inline void tk_inv_suppress_unused_lua_mt_fns (void)
  { (void) tk_inv_lua_mt_fns; }

static inline tk_inv_t *tk_inv_create (
  lua_State *L,
  uint64_t features,
  tk_dvec_t *weights,
  uint64_t n_ranks,
  tk_ivec_t *ranks,
  double decay,
  uint64_t n_threads,
  int i_weights,
  int i_ranks
) {
  if (!features)
    tk_lua_verror(L, 2, "create", "features must be > 0");
  tk_inv_t *I = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  int Ii = tk_lua_absindex(L, -1);
  I->destroyed = false;
  I->next_sid = 0;
  I->features = features;
  I->n_ranks = n_ranks >= 1 ? n_ranks : 1;
  I->weights = weights;
  if (weights)
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, i_weights);
  I->ranks = ranks;
  if (ranks)
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, i_ranks);
  I->decay = decay;
  // Precompute rank weights and total to avoid repeated computation
  I->rank_weights = tk_dvec_create(L, I->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->total_rank_weight = 0.0;
  for (uint64_t r = 0; r < I->n_ranks; r++) {
    double weight = exp(-(double)r * decay);
    I->rank_weights->a[r] = weight;
    I->total_rank_weight += weight;
  }
  I->uid_sid = tk_iumap_create();
  I->sid_uid = tk_iumap_create();
  I->node_offsets = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->node_bits = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->touched = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->wacc = tk_dvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->postings = tk_inv_postings_create(L, features, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  for (uint64_t i = 0; i < features; i ++) {
    I->postings->a[i] = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
  }
  lua_pop(L, 1);
  // Initialize temporary vectors for query processing
  I->tmp_query_offsets = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->tmp_query_features = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  // Initialize temporary buffers for rank-aware similarity
  I->tmp_q_weights = tk_dvec_create(L, I->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->tmp_e_weights = tk_dvec_create(L, I->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->tmp_inter_weights = tk_dvec_create(L, I->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->threads = tk_malloc(L, n_threads * sizeof(tk_inv_thread_t));
  memset(I->threads, 0, n_threads * sizeof(tk_inv_thread_t));
  I->pool = tk_threads_create(L, n_threads, tk_inv_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_inv_thread_t *data = I->threads + i;
    I->pool->threads[i].data = data;
    data->I = I;
    data->seen = tk_iuset_create();
    data->touched = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    data->wacc = tk_dvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    // Create thread-local buffers for rank-aware similarity
    data->q_weights_buf = tk_dvec_create(L, I->n_ranks, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    data->e_weights_buf = tk_dvec_create(L, I->n_ranks, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    data->inter_weights_buf = tk_dvec_create(L, I->n_ranks, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
  }
  return I;
}

static inline tk_inv_t *tk_inv_load (
  lua_State *L,
  FILE *fh,
  uint64_t n_threads
) {
  tk_inv_t *I = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  int Ii = tk_lua_absindex(L, -1);
  memset(I, 0, sizeof(tk_inv_t));
  tk_lua_fread(L, &I->destroyed, sizeof(bool), 1, fh);
  if (I->destroyed)
    tk_lua_verror(L, 2, "load", "index was destroyed when saved");
  tk_lua_fread(L, &I->next_sid, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &I->features, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &I->n_ranks, sizeof(uint64_t), 1, fh);
  I->uid_sid = tk_iumap_create();
  khint_t nkeys, k; int absent;
  tk_lua_fread(L, &nkeys, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nkeys; i ++) {
    int64_t key, val;
    tk_lua_fread(L, &key, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, &val, sizeof(int64_t), 1, fh);
    k = tk_iumap_put(I->uid_sid, key, &absent);
    tk_iumap_value(I->uid_sid, k) = val;
  }
  I->sid_uid = tk_iumap_create();
  tk_lua_fread(L, &nkeys, sizeof(khint_t), 1, fh);
  for (khint_t i = 0; i < nkeys; i ++) {
    int64_t key, val;
    tk_lua_fread(L, &key, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, &val, sizeof(int64_t), 1, fh);
    k = tk_iumap_put(I->sid_uid, key, &absent);
    tk_iumap_value(I->sid_uid, k) = val;
  }
  uint64_t n = 0;
  tk_lua_fread(L, &n, sizeof(uint64_t), 1, fh);
  I->node_offsets = tk_ivec_create(L, n, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  if (n)
    tk_lua_fread(L, I->node_offsets->a, sizeof(int64_t), n, fh);
  lua_pop(L, 1);
  tk_lua_fread(L, &n, sizeof(uint64_t), 1, fh);
  I->node_bits = tk_ivec_create(L, n, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  if (n)
    tk_lua_fread(L, I->node_bits->a, sizeof(int64_t), n, fh);
  lua_pop(L, 1);
  uint64_t pcount = 0;
  tk_lua_fread(L, &pcount, sizeof(uint64_t), 1, fh);
  I->postings = tk_inv_postings_create(L, pcount, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  for (uint64_t i = 0; i < pcount; i ++) {
    uint64_t plen;
    tk_lua_fread(L, &plen, sizeof(uint64_t), 1, fh);
    tk_inv_posting_t P = tk_ivec_create(L, plen, 0, 0);
    if (plen)
      tk_lua_fread(L, P->a, sizeof(int64_t), plen, fh);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    I->postings->a[i] = P;
  }
  lua_pop(L, 1);
  uint64_t wn = 0;
  tk_lua_fread(L, &wn, sizeof(uint64_t), 1, fh);
  if (wn) {
    I->weights = tk_dvec_create(L, wn, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    tk_lua_fread(L, I->weights->a, sizeof(double), wn, fh);
    lua_pop(L, 1);
  } else {
    I->weights = NULL;
  }
  uint64_t rn = 0;
  tk_lua_fread(L, &rn, sizeof(uint64_t), 1, fh);
  if (rn) {
    I->ranks = tk_ivec_create(L, rn, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    tk_lua_fread(L, I->ranks->a, sizeof(int64_t), rn, fh);
    lua_pop(L, 1);
  } else {
    I->ranks = NULL;
  }
  tk_lua_fread(L, &I->decay, sizeof(double), 1, fh);
  // Precompute rank weights and total
  I->rank_weights = tk_dvec_create(L, I->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->total_rank_weight = 0.0;
  for (uint64_t r = 0; r < I->n_ranks; r++) {
    double weight = exp(-(double)r * I->decay);
    I->rank_weights->a[r] = weight;
    I->total_rank_weight += weight;
  }
  I->threads = tk_malloc(L, n_threads * sizeof(tk_inv_thread_t));
  memset(I->threads, 0, n_threads * sizeof(tk_inv_thread_t));
  I->pool = tk_threads_create(L, n_threads, tk_inv_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_inv_thread_t *th = I->threads + i;
    I->pool->threads[i].data = th;
    th->I = I;
    th->seen = tk_iuset_create();
    th->touched = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    th->wacc = tk_dvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    // Create thread-local buffers for rank-aware similarity
    th->q_weights_buf = tk_dvec_create(L, I->n_ranks, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    th->e_weights_buf = tk_dvec_create(L, I->n_ranks, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    th->inter_weights_buf = tk_dvec_create(L, I->n_ranks, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
  }
  I->touched = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->wacc = tk_dvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  // Initialize temporary vectors for query processing
  I->tmp_query_offsets = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->tmp_query_features = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  // Initialize temporary buffers for rank-aware similarity
  I->tmp_q_weights = tk_dvec_create(L, I->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->tmp_e_weights = tk_dvec_create(L, I->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->tmp_inter_weights = tk_dvec_create(L, I->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  return I;
}

#endif
