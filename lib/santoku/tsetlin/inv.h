#ifndef TK_INV_H
#define TK_INV_H

#include <assert.h>
#include <math.h>
#include <string.h>
#include <santoku/lua/utils.h>
#include <santoku/tsetlin/conf.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/dumap.h>
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
  TK_INV_JACCARD,
  TK_INV_OVERLAP,
  TK_INV_DICE,
  TK_INV_TVERSKY,
} tk_inv_cmp_type_t;

typedef enum {
  TK_INV_NEIGHBORHOODS,
  TK_INV_MUTUAL_INIT,
  TK_INV_MUTUAL_FILTER,
  TK_INV_MIN_REMAP,
} tk_inv_stage_t;

typedef struct tk_inv_thread_s tk_inv_thread_t;

typedef struct tk_inv_s {
  bool destroyed;
  int64_t next_sid;
  uint64_t features;
  uint64_t n_ranks;
  tk_dvec_t *weights;
  tk_ivec_t *ranks;
  int64_t rank_decay_window;
  double rank_decay_sigma;
  double rank_decay_floor;
  tk_iumap_t *uid_sid;
  tk_iumap_t *sid_uid;
  tk_ivec_t *node_offsets;
  tk_ivec_t *node_bits;
  tk_inv_postings_t *postings;
  tk_dvec_t *wacc;
  tk_ivec_t *touched;
  tk_inv_thread_t *threads;
  tk_threadpool_t *pool;
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
  tk_dvec_t *wacc;
  uint64_t ifirst, ilast;
  double eps;
  uint64_t knn;
  uint64_t min;
  bool mutual;
  tk_inv_cmp_type_t cmp;
  double tversky_alpha;
  double tversky_beta;
  int64_t *old_to_new;
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
  tk_lua_fwrite(L, (char *) &I->rank_decay_window, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &I->rank_decay_sigma, sizeof(double), 1, fh);
  tk_lua_fwrite(L, (char *) &I->rank_decay_floor, sizeof(double), 1, fh);
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

  // Apply min neighbor filtering if requested
  if (min > 0) {
    // Build old_to_new index mapping
    int64_t *old_to_new = tk_malloc(L, uids->n * sizeof(int64_t));
    int64_t keeper_count = 0;
    for (uint64_t i = 0; i < uids->n; i++) {
      if (hoods->a[i]->n >= min) {
        old_to_new[i] = keeper_count ++;
      } else {
        old_to_new[i] = -1;
      }
    }

    // Only proceed if filtering is needed
    if (keeper_count < (int64_t) uids->n) {
      // Set up thread data for min filtering
      for (uint64_t i = 0; i < I->pool->n_threads; i++) {
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

      // Compact keeper data into new arrays using old_to_new mapping
      for (uint64_t i = 0; i < uids->n; i++) {
        if (old_to_new[i] >= 0) {
          new_uids->a[old_to_new[i]] = uids->a[i];
          new_hoods->a[old_to_new[i]] = hoods->a[i];
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
  tk_inv_cmp_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  bool mutual,
  tk_inv_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (I->destroyed)
    return;

  tk_ivec_t *sids, *uids;
  if (uidsp && *uidsp) {
    sids = tk_ivec_create(L, (*uidsp)->n, 0, 0);
    uids = tk_ivec_create(L, (*uidsp)->n, 0, 0);
    tk_ivec_copy(uids, *uidsp, 0, (int64_t) (*uidsp)->n, 0);
    for (uint64_t i = 0; i < uids->n; i ++)
      sids->a[i] = tk_inv_uid_sid(I, uids->a[i], false);
  } else {
    sids = tk_iumap_values(L, I->uid_sid);
    tk_ivec_asc(sids, 0, sids->n); // sort for cache locality
    uids = tk_ivec_create(L, sids->n, 0, 0);
    for (uint64_t i = 0; i < sids->n; i ++)
      uids->a[i] = tk_inv_sid_uid(I, sids->a[i]);
  }

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
    tk_dvec_ensure(data->wacc, sids->n * I->n_ranks);
    tk_dvec_zero(data->wacc);
    data->hoods = hoods;
    data->hoods_sets = hoods_sets;
    data->sid_idx = sid_idx;
    data->eps = eps; //+ TK_INV_LENGTH_EPS;
    data->knn = knn;
    data->mutual = mutual;
    data->cmp = cmp;
    data->tversky_alpha = tversky_alpha;
    data->tversky_beta = tversky_beta;
    tk_thread_range(i, I->pool->n_threads, hoods->n, &data->ifirst, &data->ilast);
  }

  tk_threads_signal(I->pool, TK_INV_NEIGHBORHOODS, 0);
  if (mutual && knn) {
    tk_threads_signal(I->pool, TK_INV_MUTUAL_INIT, 0);
    tk_threads_signal(I->pool, TK_INV_MUTUAL_FILTER, 0);
  }
  tk_iumap_destroy(sid_idx);

  // Apply min neighbor filtering if requested
  if (min > 0) {
    // Count how many UIDs will survive the min filter
    int64_t keeper_count = 0;
    for (uint64_t i = 0; i < uids->n; i++)
      if (hoods->a[i]->n >= min)
        keeper_count ++;

    // Early exit if no filtering needed
    if (keeper_count == (int64_t) uids->n)
      goto cleanup;

    // Build old_to_new index mapping
    int64_t *old_to_new = tk_malloc(L, uids->n * sizeof(int64_t));
    int64_t new_idx = 0;
    for (uint64_t i = 0; i < uids->n; i++) {
      if (hoods->a[i]->n >= min) {
        old_to_new[i] = new_idx++;
      } else {
        old_to_new[i] = -1;
      }
    }

    // Set up thread data for min filtering
    for (uint64_t i = 0; i < I->pool->n_threads; i++) {
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

    // Compact keeper data into new arrays
    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < uids->n; i++) {
      if (hoods->a[i]->n >= min) {
        new_uids->a[write_pos] = uids->a[i];
        new_hoods->a[write_pos] = hoods->a[i];
        write_pos++;
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
  lua_remove(L, -3); // sids
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

static inline double tk_inv_similarity (
  tk_inv_t *I,
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize,
  tk_inv_cmp_type_t cmp,
  double tversky_alpha,
  double tversky_beta
) {
  double inter_w = 0.0, sa = 0.0, sb = 0.0;
  tk_inv_stats(I, abits, asize, bbits, bsize, &inter_w, &sa, &sb);
  switch (cmp) {
    case TK_INV_JACCARD: {
      double u = sa + sb - inter_w;
      return (u == 0.0) ? 0.0 : inter_w / u;
    }
    case TK_INV_OVERLAP: {
      double m = (sa < sb) ? sa : sb;
      return (m == 0.0) ? 0.0 : inter_w / m;
    }
    case TK_INV_TVERSKY: {
      return tk_inv_tversky_from_stats(inter_w, sa, sb, tversky_alpha, tversky_beta);
    }
    case TK_INV_DICE: {
      double denom = sa + sb;
      return (denom == 0.0) ? 0.0 : (2.0 * inter_w) / denom;
    }
    default: { // fallback to Jaccard
      double u = sa + sb - inter_w;
      return (u == 0.0) ? 0.0 : inter_w / u;
    }
  }
}

static inline double tk_inv_similarity_partial (
  double inter_w,
  double q_w,
  double e_w,
  tk_inv_cmp_type_t cmp,
  double tversky_alpha,
  double tversky_beta
) {
  switch (cmp) {
    case TK_INV_JACCARD: {
      double uni_w = q_w + e_w - inter_w;
      return (uni_w == 0.0) ? 0.0 : inter_w / uni_w;
    }
    case TK_INV_OVERLAP: {
      double min_w = (q_w < e_w) ? q_w : e_w;
      return (min_w == 0.0) ? 0.0 : inter_w / min_w;
    }
    case TK_INV_TVERSKY: {
      double a_only = q_w - inter_w;
      double b_only = e_w - inter_w;
      if (a_only < 0.0) a_only = 0.0;
      if (b_only < 0.0) b_only = 0.0;
      double denom = inter_w + tversky_alpha * a_only + tversky_beta * b_only;
      return (denom == 0.0) ? 0.0 : inter_w / denom;
    }
    case TK_INV_DICE: {
      double denom = q_w + e_w;
      return (denom == 0.0) ? 0.0 : (2.0 * inter_w) / denom;
    }
    default: { // fallback to Jaccard
      double uni_w = q_w + e_w - inter_w;
      return (uni_w == 0.0) ? 0.0 : inter_w / uni_w;
    }
  }
}

static inline void tk_inv_compute_query_weights_by_rank (
  tk_inv_t *I,
  int64_t *data,
  size_t datalen,
  double *q_weights_by_rank  // pre-allocated array of size I->n_ranks
) {
  // Initialize all ranks to 0
  for (uint64_t r = 0; r < I->n_ranks; r ++)
    q_weights_by_rank[r] = 0.0;

  // Accumulate weights by rank
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
  double *e_weights_by_rank  // pre-allocated array of size I->n_ranks
) {
  // Initialize all ranks to 0
  for (uint64_t r = 0; r < I->n_ranks; r ++)
    e_weights_by_rank[r] = 0.0;

  // Accumulate weights by rank
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
  tk_dvec_t *wacc,           // 2D accumulator [sid][rank]
  int64_t vsid,              // candidate document sid
  double *q_weights_by_rank, // query weights by rank
  double *e_weights_by_rank, // candidate weights by rank
  tk_inv_cmp_type_t cmp,
  double tversky_alpha,
  double tversky_beta
) {
  // Check if both documents are completely empty
  double q_total = 0.0, e_total = 0.0;
  for (uint64_t r = 0; r < I->n_ranks; r ++) {
    q_total += q_weights_by_rank[r];
    e_total += e_weights_by_rank[r];
  }

  // If both documents are completely empty, similarity is 0
  if (q_total == 0.0 && e_total == 0.0) {
    return 0.0;
  }

  double total_weighted_sim = 0.0;
  double total_rank_weight = 0.0;

  for (uint64_t rank = 0; rank < I->n_ranks; rank ++) {
    double rank_weight = 1.0;
    if (I->rank_decay_window >= 0) {
      if (rank == 0) {
        rank_weight = 1.0;
      } else if (rank >= (uint64_t)I->rank_decay_window) {
        rank_weight = I->rank_decay_floor;
      } else {
        // Sigmoid ramp from 1.0 to rank_decay_floor over rank_decay_window ranks
        double t = (double)rank / (double)I->rank_decay_window;  // 0 to 1
        // Sigmoid curve: flat at start, steep in middle
        // sigma controls the steepness: higher sigma = steeper transition
        double sigmoid_arg = I->rank_decay_sigma * (t - 0.5);  // center around 0.5
        double sigmoid_val = 1.0 / (1.0 + exp(sigmoid_arg));   // sigmoid from 1 to 0
        rank_weight = I->rank_decay_floor + (1.0 - I->rank_decay_floor) * sigmoid_val;
      }
    }

    double inter_w = wacc->a[(int64_t) I->n_ranks * vsid + (int64_t) rank];
    double q_w = q_weights_by_rank[rank];
    double e_w = e_weights_by_rank[rank];

    double rank_sim;
    if (q_w > 0.0 || e_w > 0.0) {
      // Compute similarity if either document has features at this rank
      rank_sim = tk_inv_similarity_partial(inter_w, q_w, e_w, cmp, tversky_alpha, tversky_beta);
    } else {
      // If neither document has features at this rank, they're equally empty -> max similarity
      rank_sim = 0.0;
    }

    total_weighted_sim += rank_sim * rank_weight;
    total_rank_weight += rank_weight;
  }

  return (total_rank_weight > 0.0) ? total_weighted_sim / total_rank_weight : 0.0;
}

static inline double tk_inv_distance (
  tk_inv_t *I,
  int64_t uid0,
  int64_t uid1,
  tk_inv_cmp_type_t cmp,
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
  tk_inv_cmp_type_t cmp,
  double tversky_alpha,
  double tversky_beta
) {
  if (datalen == 0) {
    tk_rvec_clear(out);
    return out;
  }

  // eps += TK_INV_LENGTH_EPS;
  tk_rvec_clear(out);
  size_t n_sids = I->node_offsets->n;

  // Compute query weights by rank
  double *q_weights_by_rank = tk_malloc(NULL, I->n_ranks * sizeof(double));
  tk_inv_compute_query_weights_by_rank(I, data, datalen, q_weights_by_rank);

  // Also compute total query weight for legacy compatibility
  double q_w = 0.0;
  for (uint64_t r = 0; r < I->n_ranks; r ++)
    q_w += q_weights_by_rank[r];

  tk_dvec_ensure(I->wacc, n_sids * I->n_ranks);
  tk_dvec_zero(I->wacc);
  tk_ivec_clear(I->touched);

  for (size_t i = 0; i < datalen; i ++) {
    int64_t fid = data[i];
    int64_t rank = I->ranks ? I->ranks->a[fid] : 0;
    if (fid < 0 || fid >= (int64_t) I->postings->n)
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

  // Allocate candidate weights array
  double *e_weights_by_rank = tk_malloc(NULL, I->n_ranks * sizeof(double));

  for (uint64_t i = 0; i < I->touched->n; i ++) {
    int64_t vsid = I->touched->a[i];
    size_t elen = 0;
    int64_t *ev = tk_inv_sget(I, vsid, &elen);

    // Compute candidate weights by rank
    tk_inv_compute_candidate_weights_by_rank(I, ev, elen, e_weights_by_rank);

    // Compute rank-weighted similarity
    double sim = tk_inv_similarity_by_rank(I, I->wacc, vsid, q_weights_by_rank, e_weights_by_rank, cmp, tversky_alpha, tversky_beta);
    double dist = 1.0 - sim;
    if (dist <= eps) {
      int64_t vuid = tk_inv_sid_uid(I, vsid);
      if (vuid >= 0) {
        if (knn)
          tk_rvec_hmax(out, knn, tk_rank(vuid, dist));
        else
          tk_rvec_push(out, tk_rank(vuid, dist));
      }
    }
    for (uint64_t i = 0; i < I->n_ranks; i ++)
      I->wacc->a[(int64_t) I->n_ranks * vsid + (int64_t) i] = 0.0;
  }

  tk_rvec_asc(out, 0, out->n);

  // Cleanup allocated arrays
  free(q_weights_by_rank);
  free(e_weights_by_rank);

  return out;
}

static inline tk_rvec_t *tk_inv_neighbors_by_id (
  tk_inv_t *I,
  int64_t uid,
  uint64_t knn,
  double eps,
  tk_rvec_t *out,
  tk_inv_cmp_type_t cmp,
  double tversky_alpha,
  double tversky_beta
) {
  int64_t sid0 = tk_inv_uid_sid(I, uid, false);
  if (sid0 < 0) {
    tk_rvec_clear(out);
    return out;
  }
  size_t len = 0;
  int64_t *data = tk_inv_get(I, uid, &len);
  return tk_inv_neighbors_by_vec(I, data, len, sid0, knn, eps, out, cmp, tversky_alpha, tversky_beta);
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
  lua_settop(L, 3);
  tk_inv_t *I = tk_inv_peek(L, 1);
  int64_t uid = -1;
  tk_ivec_t *uids = NULL;
  tk_ivec_t *out = tk_ivec_peekopt(L, 3);
  out = out == NULL ? tk_ivec_create(L, 0, 0, 0) : out; // out
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
    for (uint64_t i = 0; i < uids->n; i ++) {
      uid = uids->a[i];
      size_t n = 0;
      int64_t *data = tk_inv_get(I, uid, &n);
      if (!n)
        continue;
      tk_ivec_ensure(out, out->n + n);
      memcpy(out->a + out->n, data, n * sizeof(int64_t));
      out->n += n;
    }
  }
  return 1;
}

static inline int tk_inv_neighborhoods_lua (lua_State *L)
{
  tk_inv_t *I = tk_inv_peek(L, 1);
  uint64_t knn = tk_lua_checkunsigned(L, 2, "knn");
  double eps = tk_lua_optposdouble(L, 3, "min", 1.0);
  uint64_t min = tk_lua_optposdouble(L, 4, "eps", 1.0);
  const char *typ = tk_lua_optstring(L, 5, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 6, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 7, "beta", 0.1);
  bool mutual = tk_lua_optboolean(L, 8, "mutual", false);
  tk_inv_cmp_type_t cmp = TK_INV_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_INV_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_INV_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_INV_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_INV_TVERSKY;
  else
    tk_lua_verror(L, 3, "neighbors", "invalid comparator specified", typ);
  tk_inv_neighborhoods(L, I, knn, eps, min, cmp, tversky_alpha, tversky_beta, mutual, 0, 0);
  return 2;
}

static inline int tk_inv_similarity_lua (lua_State *L)
{
  lua_settop(L, 6);
  tk_inv_t *I = tk_inv_peek(L, 1);
  const char *typ = tk_lua_optstring(L, 4, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 5, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 6, "beta", 0.1);
  tk_inv_cmp_type_t cmp = TK_INV_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_INV_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_INV_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_INV_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_INV_TVERSKY;
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
  tk_inv_cmp_type_t cmp = TK_INV_JACCARD;
  if (!strcmp(typ, "jaccard"))
    cmp = TK_INV_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_INV_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_INV_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_INV_TVERSKY;
  else
    tk_lua_verror(L, 3, "distance", "invalid comparator specified", typ);
  int64_t uid0 = tk_lua_checkinteger(L, 2, "uid0");
  int64_t uid1 = tk_lua_checkinteger(L, 3, "uid1");
  lua_pushnumber(L, tk_inv_distance(I, uid0, uid1, cmp, tversky_alpha, tversky_beta));
  return 1;
}

static inline int tk_inv_neighbors_lua (lua_State *L)
{
  lua_settop(L, 8);
  tk_inv_t *I = tk_inv_peek(L, 1);
  uint64_t knn = tk_lua_optunsigned(L, 3, "knn", 0);
  double eps = tk_lua_optposdouble(L, 4, "eps", 1.0);
  tk_rvec_t *out = tk_rvec_peek(L, 5, "out");
  const char *typ = tk_lua_optstring(L, 6, "comparator", "jaccard");
  tk_inv_cmp_type_t cmp = TK_INV_JACCARD;
  double tversky_alpha = tk_lua_optnumber(L, 7, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 8, "beta", 0.1);
  if (!strcmp(typ, "jaccard"))
    cmp = TK_INV_JACCARD;
  else if (!strcmp(typ, "overlap"))
    cmp = TK_INV_OVERLAP;
  else if (!strcmp(typ, "dice"))
    cmp = TK_INV_DICE;
  else if (!strcmp(typ, "tversky"))
    cmp = TK_INV_TVERSKY;
  else
    tk_lua_verror(L, 3, "neighbors", "invalid comparator specified", typ);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_inv_neighbors_by_id(I, uid, knn, eps, out, cmp, tversky_alpha, tversky_beta);
  } else {
    tk_ivec_t *vec = tk_ivec_peek(L, 2, "vector");
    tk_inv_neighbors_by_vec(I, vec->a, vec->n, -1, knn, eps, out, cmp, tversky_alpha, tversky_beta);
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
  tk_inv_cmp_type_t cmp = data->cmp;
  khint_t khi;
  int64_t usid, vsid, fid, iv;
  int64_t start, end;
  int64_t *ubits, *vbits;
  size_t nubits, nvbits;
  tk_rvec_t *uhood;
  tk_dumap_t *uset;
  tk_dumap_t *vset;
  tk_ivec_t *vsids;
  switch (stage) {

    case TK_INV_NEIGHBORHOODS:
      touched->n = 0;
      if (wacc->n < sids->n * I->n_ranks) {
        tk_dvec_ensure(wacc, sids->n * I->n_ranks);
        wacc->n = sids->n * I->n_ranks;
      }
      tk_dvec_zero(wacc);
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i ++) {
        usid = sids->a[i];
        if (tk_iumap_get(I->sid_uid, usid) == tk_iumap_end(I->sid_uid))
          continue;
        ubits = tk_inv_sget(I, usid, &nubits);
        uhood = hoods->a[i];
        tk_rvec_clear(uhood);
        start = I->node_offsets->a[usid];
        end = (usid + 1 == (int64_t) I->node_offsets->n)
          ? (int64_t) I->node_bits->n
          : I->node_offsets->a[usid + 1];
        if (wacc->n < sids->n * I->n_ranks) {
          tk_dvec_ensure(wacc, sids->n * I->n_ranks);
          wacc->n = sids->n * I->n_ranks;
        }

        // Compute query weights by rank
        double *q_weights_by_rank = tk_malloc(NULL, I->n_ranks * sizeof(double));
        tk_inv_compute_query_weights_by_rank(I, ubits, nubits, q_weights_by_rank);

        // Also compute total query weight for legacy compatibility
        double q_w = 0.0;
        for (uint64_t r = 0; r < I->n_ranks; r++)
          q_w += q_weights_by_rank[r];
        for (size_t k = 0; k < nubits; k ++) {
          fid = ubits[k];
          int64_t rank = I->ranks ? I->ranks->a[fid] : 0;
          double wf = tk_inv_w(I->weights, fid);
          assert(fid >= 0 && fid < (int64_t) I->postings->n);
          vsids = I->postings->a[fid];
          for (uint64_t l = 0; l < vsids->n; l ++) {
            vsid = vsids->a[l];
            if (vsid == usid)
              continue;
            khi = tk_iumap_get(sid_idx, vsid);
            if (khi == tk_iumap_end(sid_idx))
              continue;
            iv = tk_iumap_value(sid_idx, khi);
            assert(iv >= 0 && iv < (int64_t) sids->n);
            if (wacc->a[(int64_t) I->n_ranks * iv + rank] == 0.0)
              tk_ivec_push(touched, iv);
            wacc->a[(int64_t) I->n_ranks * iv + rank] += wf;
          }
        }
        // Allocate candidate weights array
        double *e_weights_by_rank = tk_malloc(NULL, I->n_ranks * sizeof(double));

        for (uint64_t ti = 0; ti < touched->n; ti ++) {
          iv = touched->a[ti];
          vsid = sids->a[iv];
          vbits = tk_inv_sget(I, vsid, &nvbits);

          // Compute candidate weights by rank
          tk_inv_compute_candidate_weights_by_rank(I, vbits, nvbits, e_weights_by_rank);

          // Compute rank-weighted similarity
          double sim = tk_inv_similarity_by_rank(I, wacc, iv, q_weights_by_rank, e_weights_by_rank, cmp, tversky_alpha, tversky_beta);
          double dist = 1.0 - sim;
          if (dist <= eps) {
            if (knn)
              tk_rvec_hmax(uhood, knn, tk_rank(iv, dist));
            else
              tk_rvec_push(uhood, tk_rank(iv, dist));
          }
        }
        for (uint64_t ti = 0; ti < touched->n; ti ++)
          for (uint64_t r = 0; r < I->n_ranks; r ++)
            wacc->a[(int64_t) I->n_ranks * touched->a[ti] + (int64_t) r] = 0.0;
        tk_rvec_asc(uhood, 0, uhood->n);
        tk_rvec_shrink(uhood);
        touched->n = 0;

        // Cleanup allocated arrays
        free(q_weights_by_rank);
        free(e_weights_by_rank);
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
        for (uint64_t qi = uhood->n; qi < uhood->m; qi++) {
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
              double sim = tk_inv_similarity(I, vbits, vlen, ubits, ulen, cmp, tversky_alpha, tversky_beta);
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
      for (int64_t i = (int64_t) data->ifirst; i <= (int64_t) data->ilast; i++) {
        if (hoods->a[i]->n >= data->min) {
          tk_rvec_t *hood = hoods->a[i];
          uint64_t mutual_write_pos = 0;
          uint64_t non_mutual_write_pos = 0;

          // Process mutual neighbors (0 to n-1)
          for (uint64_t j = 0; j < hood->n; j++) {
            int64_t old_neighbor_idx = hood->a[j].i;
            int64_t new_neighbor_idx = data->old_to_new[old_neighbor_idx];

            if (new_neighbor_idx >= 0) {
              // Keep this mutual neighbor, update index
              hood->a[mutual_write_pos++] = tk_rank(new_neighbor_idx, hood->a[j].d);
            }
            // Otherwise skip (neighbor was filtered out)
          }

          // Process non-mutual neighbors (n to m-1)
          for (uint64_t j = hood->n; j < hood->m; j++) {
            int64_t old_neighbor_idx = hood->a[j].i;
            int64_t new_neighbor_idx = data->old_to_new[old_neighbor_idx];

            if (new_neighbor_idx >= 0) {
              // Keep this non-mutual neighbor, update index
              hood->a[mutual_write_pos + non_mutual_write_pos++] = tk_rank(new_neighbor_idx, hood->a[j].d);
            }
            // Otherwise skip (neighbor was filtered out)
          }

          hood->n = mutual_write_pos;  // Update mutual count after filtering
          hood->m = mutual_write_pos + non_mutual_write_pos;  // Update total count after filtering
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
  { "get", tk_inv_get_lua },
  { "neighborhoods", tk_inv_neighborhoods_lua },
  { "neighbors", tk_inv_neighbors_lua },
  { "distance", tk_inv_distance_lua },
  { "similarity", tk_inv_similarity_lua },
  { "size", tk_inv_size_lua },
  { "threads", tk_inv_threads_lua },
  { "features", tk_inv_features_lua },
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
  int64_t rank_decay_window,
  double rank_decay_sigma,
  double rank_decay_floor,
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
  I->rank_decay_window = rank_decay_window;
  I->rank_decay_sigma = rank_decay_sigma;
  I->rank_decay_floor = rank_decay_floor;
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
  tk_lua_fread(L, &I->rank_decay_window, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &I->rank_decay_sigma, sizeof(double), 1, fh);
  tk_lua_fread(L, &I->rank_decay_floor, sizeof(double), 1, fh);
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
  }
  I->touched = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  I->wacc = tk_dvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  return I;
}

#endif
