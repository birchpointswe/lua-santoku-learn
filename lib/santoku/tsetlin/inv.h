#ifndef TK_INV_H
#define TK_INV_H

#include <math.h>
#include <float.h>
#include <string.h>
#include <omp.h>
#include <cblas.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/iumap.h>
#include <santoku/dumap.h>
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
  TK_INV_FIND,
  TK_INV_REPLACE
} tk_inv_uid_mode_t;

typedef struct tk_inv_s {
  bool destroyed;
  int64_t next_sid;
  uint64_t features;
  uint64_t n_ranks;
  tk_dvec_t *weights;
  tk_ivec_t *ranks;
  tk_ivec_t *rank_sizes;
  tk_iumap_t *uid_sid;
  tk_ivec_t *sid_to_uid;
  tk_ivec_t *node_offsets;
  tk_ivec_t *node_bits;
  tk_inv_postings_t *postings;
} tk_inv_t;


static inline double tk_inv_w (tk_dvec_t *W, int64_t fid);

#define TK_INV_MAX_RANKS 5

typedef struct {
  double weights[TK_INV_MAX_RANKS];
  double total;
  uint64_t n_ranks;
} tk_inv_rank_weights_t;

static inline void tk_inv_precompute_rank_weights (
  tk_inv_rank_weights_t *rw,
  uint64_t n_ranks,
  double decay
);
static inline double tk_inv_similarity (
  tk_inv_t *inv,
  int64_t *a, size_t na,
  int64_t *b, size_t nb,
  double decay,
  double bandwidth
);
static inline double tk_inv_similarity_fast (
  int64_t *ranks_arr,
  double *weights_arr,
  uint64_t n_ranks,
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize,
  double bandwidth,
  tk_inv_rank_weights_t *rw,
  double *q_arr,
  double *e_arr,
  double *i_arr
);
static inline void tk_inv_compute_query_weights_by_rank (
  tk_inv_t *inv,
  int64_t *bits,
  size_t nbits,
  double *q_weights_by_rank
);
static inline void tk_inv_compute_candidate_weights_by_rank (
  tk_inv_t *inv,
  int64_t *bits,
  size_t nbits,
  double *e_weights_by_rank
);
static inline double tk_inv_similarity_by_rank_fast (
  uint64_t n_ranks,
  double *wacc_arr,
  int64_t vsid,
  double *q_weights_by_rank,
  double *e_weights_by_rank,
  double bandwidth,
  double cutoff,
  tk_inv_rank_weights_t *rw
);

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
  tk_inv_t *inv
) {
  if (inv->destroyed)
    return;
  int Ii = 1;
  if (inv->next_sid > (int64_t) (SIZE_MAX / sizeof(int64_t)))
    tk_error(L, "inv_shrink: allocation size overflow", ENOMEM);
  int64_t *old_to_new = tk_malloc(L, (size_t) inv->next_sid * sizeof(int64_t));
  for (int64_t i = 0; i < inv->next_sid; i ++)
    old_to_new[i] = -1;
  int64_t new_sid = 0;
  for (int64_t s = 0; s < inv->next_sid; s++) {
    if (inv->sid_to_uid->a[s] >= 0) {
      old_to_new[s] = new_sid++;
    }
  }
  if (new_sid == inv->next_sid) {
    free(old_to_new);
    tk_inv_postings_shrink(inv->postings);
    for (uint64_t i = 0; i < inv->postings->n; i ++)
      tk_ivec_shrink(inv->postings->a[i]);
    tk_ivec_shrink(inv->node_offsets);
    tk_ivec_shrink(inv->node_bits);
    tk_dvec_shrink(inv->weights);
    tk_ivec_shrink(inv->ranks);
    return;
  }
  tk_ivec_t *new_node_offsets = tk_ivec_create(L, (size_t) new_sid + 1, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  tk_ivec_t *new_node_bits = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  new_node_offsets->n = 0;
  for (int64_t old_sid = 0; old_sid < inv->next_sid; old_sid++) {
    if (inv->sid_to_uid->a[old_sid] < 0)
      continue;
    int64_t start = inv->node_offsets->a[old_sid];
    int64_t end = inv->node_offsets->a[old_sid + 1];
    if (tk_ivec_push(new_node_offsets, (int64_t) new_node_bits->n) != 0) {
      tk_lua_verror(L, 2, "compact", "allocation failed");
      return;
    }
    for (int64_t i = start; i < end; i ++)
      if (tk_ivec_push(new_node_bits, inv->node_bits->a[i]) != 0) {
        tk_lua_verror(L, 2, "compact", "allocation failed");
        return;
      }
  }
  if (tk_ivec_push(new_node_offsets, (int64_t) new_node_bits->n) != 0) {
    tk_lua_verror(L, 2, "compact", "allocation failed");
    return;
  }
  tk_lua_del_ephemeron(L, TK_INV_EPH, Ii, inv->node_offsets);
  tk_lua_del_ephemeron(L, TK_INV_EPH, Ii, inv->node_bits);
  tk_ivec_destroy(inv->node_offsets);
  tk_ivec_destroy(inv->node_bits);
  inv->node_offsets = new_node_offsets;
  inv->node_bits = new_node_bits;
  for (uint64_t fid = 0; fid < inv->postings->n; fid ++) {
    tk_ivec_t *post = inv->postings->a[fid];
    uint64_t write_pos = 0;
    for (uint64_t i = 0; i < post->n; i ++) {
      int64_t old_sid = post->a[i];
      int64_t new_sid_val = old_to_new[old_sid];
      if (new_sid_val >= 0) {
        post->a[write_pos ++] = new_sid_val;
      }
    }
    post->n = write_pos;
    tk_ivec_shrink(post);
  }
  tk_iumap_t *new_uid_sid = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  for (khint_t k = kh_begin(inv->uid_sid); k != kh_end(inv->uid_sid); k ++) {
    if (!kh_exist(inv->uid_sid, k))
      continue;
    int64_t uid = kh_key(inv->uid_sid, k);
    int64_t old_sid = kh_value(inv->uid_sid, k);
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val >= 0) {
      int is_new;
      khint_t khi = tk_iumap_put(new_uid_sid, uid, &is_new);
      tk_iumap_setval(new_uid_sid, khi, new_sid_val);
    }
  }
  tk_lua_del_ephemeron(L, TK_INV_EPH, Ii, inv->uid_sid);
  tk_iumap_destroy(inv->uid_sid);
  inv->uid_sid = new_uid_sid;
  tk_ivec_t *new_sid_to_uid = tk_ivec_create(L, (uint64_t)new_sid, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  new_sid_to_uid->n = (uint64_t)new_sid;
  for (int64_t old_sid = 0; old_sid < inv->next_sid; old_sid++) {
    int64_t new_sid_val = old_to_new[old_sid];
    if (new_sid_val >= 0) {
      new_sid_to_uid->a[new_sid_val] = inv->sid_to_uid->a[old_sid];
    }
  }
  tk_lua_del_ephemeron(L, TK_INV_EPH, Ii, inv->sid_to_uid);
  inv->sid_to_uid = new_sid_to_uid;
  inv->next_sid = new_sid;
  tk_inv_postings_shrink(inv->postings);
  tk_ivec_shrink(inv->node_offsets);
  tk_ivec_shrink(inv->node_bits);
  tk_dvec_shrink(inv->weights);
  tk_ivec_shrink(inv->ranks);
  free(old_to_new);
}

static inline void tk_inv_destroy (
  tk_inv_t *inv
) {
  if (inv->destroyed)
    return;
  inv->destroyed = true;
}

static inline void tk_inv_persist (
  lua_State *L,
  tk_inv_t *inv,
  FILE *fh
) {
  if (inv->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't persist a destroyed index");
    return;
  }
  tk_lua_fwrite(L, (char *) &inv->destroyed, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &inv->next_sid, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &inv->features, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &inv->n_ranks, sizeof(uint64_t), 1, fh);
  tk_iumap_persist(L, inv->uid_sid, fh);
  tk_ivec_persist(L, inv->sid_to_uid, fh);
  tk_ivec_persist(L, inv->node_offsets, fh);
  tk_ivec_persist(L, inv->node_bits, fh);
  uint64_t pcount = inv->postings ? inv->postings->n : 0;
  tk_lua_fwrite(L, (char *) &pcount, sizeof(uint64_t), 1, fh);
  for (uint64_t i = 0; i < pcount; i ++) {
    tk_inv_posting_t P = inv->postings->a[i];
    tk_lua_fwrite(L, (char *) &P->n, sizeof(uint64_t), 1, fh);
    tk_lua_fwrite(L, (char *) P->a, sizeof(int64_t), P->n, fh);
  }
  tk_dvec_persist(L, inv->weights, fh);
  tk_ivec_persist(L, inv->ranks, fh);
}

static inline uint64_t tk_inv_size (
  tk_inv_t *inv
) {
  return tk_iumap_size(inv->uid_sid);
}

static inline void tk_inv_uid_remove (
  tk_inv_t *inv,
  int64_t uid
) {
  khint_t khi;
  khi = tk_iumap_get(inv->uid_sid, uid);
  if (khi == tk_iumap_end(inv->uid_sid))
    return;
  int64_t sid = tk_iumap_val(inv->uid_sid, khi);
  tk_iumap_del(inv->uid_sid, khi);

  if (sid >= 0 && sid < (int64_t)inv->sid_to_uid->n)
    inv->sid_to_uid->a[sid] = -1;
}

static inline int64_t tk_inv_uid_sid (
  tk_inv_t *inv,
  int64_t uid,
  tk_inv_uid_mode_t mode
) {
  int kha;
  khint_t khi;
  if (mode == TK_INV_FIND) {
    khi = tk_iumap_get(inv->uid_sid, uid);
    if (khi == tk_iumap_end(inv->uid_sid))
      return -1;
    else
      return tk_iumap_val(inv->uid_sid, khi);
  } else {
    khi = tk_iumap_get(inv->uid_sid, uid);
    if (khi != tk_iumap_end(inv->uid_sid)) {
      int64_t old_sid = tk_iumap_val(inv->uid_sid, khi);
      tk_iumap_del(inv->uid_sid, khi);
      if (old_sid >= 0 && old_sid < inv->next_sid)
        inv->sid_to_uid->a[old_sid] = -1;
    }
    int64_t sid = (int64_t) (inv->next_sid ++);
    khi = tk_iumap_put(inv->uid_sid, uid, &kha);
    tk_iumap_setval(inv->uid_sid, khi, sid);

    tk_ivec_ensure(inv->sid_to_uid, (uint64_t)inv->next_sid);
    if (inv->sid_to_uid->n < (uint64_t)inv->next_sid) {
      for (uint64_t i = inv->sid_to_uid->n; i < (uint64_t)inv->next_sid; i++)
        inv->sid_to_uid->a[i] = -1;
      inv->sid_to_uid->n = (uint64_t)inv->next_sid;
    }
    inv->sid_to_uid->a[sid] = uid;
    return sid;
  }
}

static inline int64_t tk_inv_sid_uid (
  tk_inv_t *inv,
  int64_t sid
) {
  if (sid < 0 || sid >= (int64_t)inv->sid_to_uid->n)
    return -1;
  return inv->sid_to_uid->a[sid];
}

static inline int64_t *tk_inv_sget (
  tk_inv_t *inv,
  int64_t sid,
  size_t *np
) {
  if (sid < 0 || sid + 1 > (int64_t) inv->node_offsets->n) {
    *np = 0;
    return NULL;
  }
  int64_t start = inv->node_offsets->a[sid];
  int64_t end;
  if (sid + 1 == (int64_t) inv->node_offsets->n) {
    end = (int64_t) inv->node_bits->n;
  } else {
    end = inv->node_offsets->a[sid + 1];
  }
  if (start < 0 || end < start || end > (int64_t) inv->node_bits->n) {
    *np = 0;
    return NULL;
  }
  *np = (size_t) (end - start);
  return inv->node_bits->a + start;
}

static inline int64_t *tk_inv_get (
  tk_inv_t *inv,
  int64_t uid,
  size_t *np
) {
  int64_t sid = tk_inv_uid_sid(inv, uid, TK_INV_FIND);
  if (sid < 0)
    return NULL;
  return tk_inv_sget(inv, sid, np);
}

static inline void tk_inv_add (
  lua_State *L,
  tk_inv_t *inv,
  int Ii,
  tk_ivec_t *ids,
  tk_ivec_t *node_bits
) {
  if (inv->destroyed) {
    tk_lua_verror(L, 2, "add", "can't add to a destroyed index");
    return;
  }
  node_bits->n = tk_ivec_uasc(node_bits, 0, node_bits->n);
  size_t nb = node_bits->n;
  size_t nsamples = ids->n;
  size_t i = 0;
  for (size_t s = 0; s < nsamples; s ++) {
    int64_t uid = ids->a[s];
    int64_t sid = tk_inv_uid_sid(inv, uid, TK_INV_REPLACE);
    uint64_t required_size = (uint64_t)(sid + 2);
    if (tk_ivec_ensure(inv->node_offsets, required_size) != 0) {
      tk_lua_verror(L, 2, "add", "allocation failed during indexing");
      return;
    }
    if (inv->node_offsets->n < required_size)
      inv->node_offsets->n = required_size;
    inv->node_offsets->a[sid] = (int64_t) inv->node_bits->n;
    while (i < nb) {
      int64_t b = node_bits->a[i];
      if (b < 0) {
        i ++;
        continue;
      }
      size_t sample_idx = (size_t) b / (size_t) inv->features;
      if (sample_idx != s)
        break;
      int64_t fid = b % (int64_t) inv->features;
      tk_ivec_t *post = inv->postings->a[fid];
      if (tk_ivec_push(post, sid) != 0) {
        tk_lua_verror(L, 2, "add", "allocation failed during indexing");
        return;
      }
      if (tk_ivec_push(inv->node_bits, fid) != 0) {
        tk_lua_verror(L, 2, "add", "allocation failed during indexing");
        return;
      }
      i ++;
    }
    inv->node_offsets->a[sid + 1] = (int64_t) inv->node_bits->n;
  }
}

static inline void tk_inv_remove (
  lua_State *L,
  tk_inv_t *inv,
  int64_t uid
) {
  if (inv->destroyed) {
    tk_lua_verror(L, 2, "remove", "can't remove from a destroyed index");
    return;
  }
  tk_inv_uid_remove(inv, uid);
}

static inline void tk_inv_keep (
  lua_State *L,
  tk_inv_t *inv,
  tk_ivec_t *ids
) {
  if (inv->destroyed) {
    tk_lua_verror(L, 2, "keep", "can't keep in a destroyed index");
    return;
  }

  tk_iuset_t *keep_set = tk_iuset_from_ivec(0, ids);
  if (!keep_set) {
    tk_lua_verror(L, 2, "keep", "allocation failed");
    return;
  }
  tk_iuset_t *to_remove_set = tk_iuset_create(0, 0);
  tk_iuset_union_iumap(to_remove_set, inv->uid_sid);
  tk_iuset_subtract(to_remove_set, keep_set);
  int64_t uid;
  tk_umap_foreach_keys(to_remove_set, uid, ({
    tk_inv_uid_remove(inv, uid);
  }));
  tk_iuset_destroy(keep_set);
  tk_iuset_destroy(to_remove_set);
}

static inline void tk_inv_prepare_universe_map (
  lua_State *L,
  tk_inv_t *A,
  tk_ivec_t **uids_out,
  tk_ivec_t **sid_to_pos_out
) {
  tk_ivec_t *uids = tk_ivec_create(L, 0, 0, 0);
  tk_ivec_t *sid_to_pos = tk_ivec_create(NULL, (uint64_t)A->next_sid, 0, 0);
  sid_to_pos->n = (uint64_t)A->next_sid;
  uint64_t active_idx = 0;
  for (int64_t sid = 0; sid < A->next_sid; sid++) {
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

static inline void tk_inv_neighborhoods (
  lua_State *L,
  tk_inv_t *inv,
  uint64_t knn,
  double decay,
  double bandwidth,
  tk_inv_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (inv->destroyed)
    return;

  tk_ivec_t *uids = tk_ivec_create(L, 0, 0, 0);
  tk_ivec_t *sid_to_pos = tk_ivec_create(L, (uint64_t)inv->next_sid, 0, 0);
  sid_to_pos->n = (uint64_t)inv->next_sid;

  tk_inv_hoods_t *hoods = tk_inv_hoods_create(L, 0, 0, 0);
  int hoods_stack_idx = lua_gettop(L);

  uint64_t active_idx = 0;
  for (int64_t sid = 0; sid < inv->next_sid; sid++) {
    int64_t uid = inv->sid_to_uid->a[sid];
    if (uid >= 0) {
      sid_to_pos->a[sid] = (int64_t)active_idx;
      tk_ivec_push(uids, uid);
      tk_rvec_t *hood = tk_rvec_create(L, knn, 0, 0);
      hood->n = 0;
      tk_lua_add_ephemeron(L, TK_INV_EPH, hoods_stack_idx, -1);
      lua_pop(L, 1);
      tk_inv_hoods_push(hoods, hood);
      active_idx++;
    } else {
      sid_to_pos->a[sid] = -1;
    }
  }

  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, inv->n_ranks, decay);
  uint64_t n_ranks = inv->n_ranks;

  #pragma omp parallel
  {
    tk_dvec_t *wacc = tk_dvec_create(NULL, uids->n * n_ranks, 0, 0);
    memset(wacc->a, 0, sizeof(double) * uids->n * n_ranks);
    tk_ivec_t *touched = tk_ivec_create(NULL, 0, 0, 0);
    double q_weights_buf[TK_INV_MAX_RANKS];
    double e_weights_buf[TK_INV_MAX_RANKS];
    #pragma omp for schedule(static) nowait
    for (int64_t i = 0; i < (int64_t) hoods->n; i ++) {
      tk_rvec_t *uhood = hoods->a[i];
      tk_rvec_clear(uhood);
      int64_t uid = uids->a[i];
      int64_t usid = tk_inv_uid_sid(inv, uid, TK_INV_FIND);
      if (usid < 0 || usid >= (int64_t)inv->sid_to_uid->n || inv->sid_to_uid->a[usid] < 0)
        continue;
      size_t nubits;
      int64_t *ubits = tk_inv_sget(inv, usid, &nubits);
      double *q_weights_by_rank = q_weights_buf;
      tk_inv_compute_query_weights_by_rank(inv, ubits, nubits, q_weights_by_rank);
      touched->n = 0;
      double cutoff = 1.0;
      double min_required_sim = 0.0;
      double gate = 1.0;
      for (uint64_t current_rank = 0; current_rank < n_ranks; current_rank ++) {
        bool can_prune_new = gate < min_required_sim;
        for (size_t k = 0; k < nubits; k ++) {
          int64_t fid = ubits[k];
          int64_t fid_rank = inv->ranks->a[fid];
          if ((uint64_t)fid_rank != current_rank)
            continue;
          double wf = tk_inv_w(inv->weights, fid);
          tk_ivec_t *vsids = inv->postings->a[fid];
          for (uint64_t l = 0; l < vsids->n; l ++) {
            int64_t vsid = vsids->a[l];
            if (vsid == usid)
              continue;
            if (vsid < 0 || vsid >= inv->next_sid)
              continue;
            int64_t iv = sid_to_pos->a[vsid];
            if (iv < 0)
              continue;
            double *wacc_base = &wacc->a[(int64_t) n_ranks * iv];
            if (wacc_base[0] < 0)
              continue;
            bool first_touch = true;
            for (uint64_t r = 0; r < n_ranks; r ++) {
              if (wacc_base[r] != 0.0) {
                first_touch = false;
                break;
              }
            }
            if (first_touch) {
              if (can_prune_new) {
                wacc_base[0] = -1.0;
                tk_ivec_push(touched, vsid);
                continue;
              }
              tk_ivec_push(touched, vsid);
            }
            wacc_base[current_rank] += wf;
          }
        }
        gate *= decay;
      }
      touched->n = tk_ivec_uasc(touched, 0, touched->n);
      double *e_weights_by_rank = e_weights_buf;
      cutoff = (knn && uhood->n >= knn) ? uhood->a[0].d : 1.0;
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        double *wacc_base = &wacc->a[(int64_t) n_ranks * iv];
        if (wacc_base[0] < 0)
          continue;
        size_t nvbits;
        int64_t *vbits = tk_inv_sget(inv, vsid, &nvbits);
        tk_inv_compute_candidate_weights_by_rank(inv, vbits, nvbits, e_weights_by_rank);
        double sim = tk_inv_similarity_by_rank_fast(n_ranks, wacc->a, iv, q_weights_by_rank, e_weights_by_rank, bandwidth, cutoff, &rw);
        double dist = 1.0 - sim;
        if (dist <= cutoff) {
          if (knn)
            tk_rvec_hmax(uhood, knn, tk_rank(iv, dist));
          else
            tk_rvec_push(uhood, tk_rank(iv, dist));
          if (knn && uhood->n >= knn)
            cutoff = uhood->a[0].d;
        }
      }
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        for (uint64_t r = 0; r < n_ranks; r ++)
          wacc->a[(int64_t) n_ranks * iv + (int64_t) r] = 0.0;
      }
      tk_rvec_asc(uhood, 0, uhood->n);
      touched->n = 0;
    }

    tk_dvec_destroy(wacc);
    tk_ivec_destroy(touched);
  }

  tk_ivec_destroy(sid_to_pos);

  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = uids;
}

static inline void tk_inv_neighborhoods_by_ids (
  lua_State *L,
  tk_inv_t *inv,
  tk_ivec_t *query_ids,
  uint64_t knn,
  double decay,
  double bandwidth,
  tk_inv_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (inv->destroyed)
    return;

  tk_ivec_t *all_uids, *sid_to_pos;
  tk_inv_prepare_universe_map(L, inv, &all_uids, &sid_to_pos);

  tk_inv_hoods_t *hoods = tk_inv_hoods_create(L, query_ids->n, 0, 0);
  int hoods_stack_idx = lua_gettop(L);
  hoods->n = query_ids->n;
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_rvec_create(L, knn ? knn : 16, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
  }

  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, inv->n_ranks, decay);
  uint64_t n_ranks = inv->n_ranks;

  #pragma omp parallel
  {
    tk_dvec_t *wacc = tk_dvec_create(NULL, all_uids->n * n_ranks, 0, 0);
    memset(wacc->a, 0, sizeof(double) * all_uids->n * n_ranks);
    tk_ivec_t *touched = tk_ivec_create(NULL, 0, 0, 0);
    double q_weights_buf[TK_INV_MAX_RANKS];
    double e_weights_buf[TK_INV_MAX_RANKS];
    #pragma omp for schedule(static) nowait
    for (int64_t i = 0; i < (int64_t) hoods->n; i ++) {
      tk_rvec_t *uhood = hoods->a[i];
      tk_rvec_clear(uhood);
      int64_t uid = query_ids->a[i];
      int64_t usid = tk_inv_uid_sid(inv, uid, TK_INV_FIND);
      if (usid < 0 || usid >= (int64_t)inv->sid_to_uid->n || inv->sid_to_uid->a[usid] < 0)
        continue;
      size_t nubits;
      int64_t *ubits = tk_inv_sget(inv, usid, &nubits);
      double *q_weights_by_rank = q_weights_buf;
      tk_inv_compute_query_weights_by_rank(inv, ubits, nubits, q_weights_by_rank);
      touched->n = 0;
      double cutoff = 1.0;
      double min_required_sim = 0.0;
      double gate = 1.0;
      for (uint64_t current_rank = 0; current_rank < n_ranks; current_rank ++) {
        bool can_prune_new = gate < min_required_sim;
        for (size_t k = 0; k < nubits; k ++) {
          int64_t fid = ubits[k];
          int64_t fid_rank = inv->ranks->a[fid];
          if ((uint64_t)fid_rank != current_rank)
            continue;
          double wf = tk_inv_w(inv->weights, fid);
          tk_ivec_t *vsids = inv->postings->a[fid];
          for (uint64_t l = 0; l < vsids->n; l ++) {
            int64_t vsid = vsids->a[l];
            if (vsid == usid)
              continue;
            if (vsid < 0 || vsid >= inv->next_sid)
              continue;
            int64_t iv = sid_to_pos->a[vsid];
            if (iv < 0)
              continue;
            double *wacc_base = &wacc->a[(int64_t) n_ranks * iv];
            if (wacc_base[0] < 0)
              continue;
            bool first_touch = true;
            for (uint64_t r = 0; r < n_ranks; r ++) {
              if (wacc_base[r] != 0.0) {
                first_touch = false;
                break;
              }
            }
            if (first_touch) {
              if (can_prune_new) {
                wacc_base[0] = -1.0;
                tk_ivec_push(touched, vsid);
                continue;
              }
              tk_ivec_push(touched, vsid);
            }
            wacc_base[current_rank] += wf;
          }
        }
        gate *= decay;
      }
      touched->n = tk_ivec_uasc(touched, 0, touched->n);
      double *e_weights_by_rank = e_weights_buf;
      cutoff = (knn && uhood->n >= knn) ? uhood->a[0].d : 1.0;
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        double *wacc_base = &wacc->a[(int64_t) n_ranks * iv];
        if (wacc_base[0] < 0)
          continue;
        size_t nvbits;
        int64_t *vbits = tk_inv_sget(inv, vsid, &nvbits);
        tk_inv_compute_candidate_weights_by_rank(inv, vbits, nvbits, e_weights_by_rank);
        double sim = tk_inv_similarity_by_rank_fast(n_ranks, wacc->a, iv, q_weights_by_rank, e_weights_by_rank, bandwidth, cutoff, &rw);
        double dist = 1.0 - sim;
        if (dist <= cutoff) {
          if (knn)
            tk_rvec_hmax(uhood, knn, tk_rank(iv, dist));
          else
            tk_rvec_push(uhood, tk_rank(iv, dist));
          if (knn && uhood->n >= knn)
            cutoff = uhood->a[0].d;
        }
      }
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        for (uint64_t r = 0; r < n_ranks; r ++)
          wacc->a[(int64_t) n_ranks * iv + (int64_t) r] = 0.0;
      }
      tk_rvec_asc(uhood, 0, uhood->n);
      touched->n = 0;
    }

    tk_dvec_destroy(wacc);
    tk_ivec_destroy(touched);
  }

  tk_ivec_destroy(sid_to_pos);

  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = all_uids;
}

static inline void tk_inv_neighborhoods_by_vecs (
  lua_State *L,
  tk_inv_t *inv,
  tk_ivec_t *query_vecs,
  uint64_t knn,
  double decay,
  double bandwidth,
  tk_inv_hoods_t **hoodsp,
  tk_ivec_t **uidsp
) {
  if (inv->destroyed)
    return;

  uint64_t n_queries = 0;
  for (uint64_t i = 0; i < query_vecs->n; i ++) {
    int64_t encoded = query_vecs->a[i];
    if (encoded >= 0) {
      uint64_t sample_idx = (uint64_t) encoded / inv->features;
      if (sample_idx >= n_queries) n_queries = sample_idx + 1;
    }
  }

  tk_ivec_t *query_offsets = tk_ivec_create(L, n_queries + 1, 0, 0);
  tk_ivec_t *query_features = tk_ivec_create(L, query_vecs->n, 0, 0);
  query_offsets->n = n_queries + 1;
  query_features->n = 0;

  for (uint64_t i = 0; i <= n_queries; i ++)
    query_offsets->a[i] = 0;
  for (uint64_t i = 0; i < query_vecs->n; i ++) {
    int64_t encoded = query_vecs->a[i];
    if (encoded >= 0) {
      uint64_t sample_idx = (uint64_t) encoded / inv->features;
      query_offsets->a[sample_idx + 1] ++;
    }
  }
  for (uint64_t i = 1; i <= n_queries; i ++)
    query_offsets->a[i] += query_offsets->a[i - 1];

  tk_ivec_t *write_offsets = tk_ivec_create(L, n_queries, 0, 0);
  tk_ivec_copy(write_offsets, query_offsets, 0, (int64_t) n_queries, 0);

  for (uint64_t i = 0; i < query_vecs->n; i ++) {
    int64_t encoded = query_vecs->a[i];
    if (encoded >= 0) {
      uint64_t sample_idx = (uint64_t) encoded / inv->features;
      int64_t fid = encoded % (int64_t) inv->features;
      int64_t write_pos = write_offsets->a[sample_idx]++;
      query_features->a[write_pos] = fid;
    }
  }
  query_features->n = (size_t) query_offsets->a[n_queries];

  tk_inv_hoods_t *hoods = tk_inv_hoods_create(L, n_queries, 0, 0);
  int hoods_stack_idx = lua_gettop(L);
  hoods->n = n_queries;
  for (uint64_t i = 0; i < hoods->n; i ++) {
    hoods->a[i] = tk_rvec_create(L, knn ? knn : 16, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
  }

  tk_ivec_t *all_uids, *sid_to_pos;
  tk_inv_prepare_universe_map(L, inv, &all_uids, &sid_to_pos);

  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, inv->n_ranks, decay);
  uint64_t n_ranks = inv->n_ranks;

  #pragma omp parallel
  {
    tk_dvec_t *wacc = tk_dvec_create(NULL, all_uids->n * n_ranks, 0, 0);
    memset(wacc->a, 0, sizeof(double) * all_uids->n * n_ranks);
    tk_ivec_t *touched = tk_ivec_create(NULL, 0, 0, 0);
    double q_weights_buf[TK_INV_MAX_RANKS];
    double e_weights_buf[TK_INV_MAX_RANKS];

    #pragma omp for schedule(static) nowait
    for (int64_t i = 0; i < (int64_t) hoods->n; i ++) {
      tk_rvec_t *uhood = hoods->a[i];
      tk_rvec_clear(uhood);
      int64_t start = query_offsets->a[i];
      int64_t end = (i + 1 < (int64_t) query_offsets->n) ? query_offsets->a[i + 1] : (int64_t) query_features->n;
      int64_t *ubits = query_features->a + start;
      size_t nubits = (size_t)(end - start);
      double *q_weights_by_rank = q_weights_buf;
      tk_inv_compute_query_weights_by_rank(inv, ubits, nubits, q_weights_by_rank);
      touched->n = 0;
      double cutoff = 1.0;
      double min_required_sim = 0.0;
      double gate = 1.0;
      for (uint64_t current_rank = 0; current_rank < n_ranks; current_rank ++) {
        bool can_prune_new = gate < min_required_sim;
        for (size_t k = 0; k < nubits; k ++) {
          int64_t fid = ubits[k];
          int64_t fid_rank = inv->ranks->a[fid];
          if ((uint64_t)fid_rank != current_rank)
            continue;
          double wf = tk_inv_w(inv->weights, fid);
          tk_ivec_t *vsids = inv->postings->a[fid];
          for (uint64_t l = 0; l < vsids->n; l ++) {
            int64_t vsid = vsids->a[l];
            if (vsid < 0 || vsid >= inv->next_sid)
              continue;
            int64_t iv = sid_to_pos->a[vsid];
            if (iv < 0)
              continue;
            double *wacc_base = &wacc->a[(int64_t) n_ranks * iv];
            if (wacc_base[0] < 0)
              continue;
            bool first_touch = true;
            for (uint64_t r = 0; r < n_ranks; r ++) {
              if (wacc_base[r] != 0.0) {
                first_touch = false;
                break;
              }
            }
            if (first_touch) {
              if (can_prune_new) {
                wacc_base[0] = -1.0;
                tk_ivec_push(touched, vsid);
                continue;
              }
              tk_ivec_push(touched, vsid);
            }
            wacc_base[current_rank] += wf;
          }
        }
        gate *= decay;
      }
      touched->n = tk_ivec_uasc(touched, 0, touched->n);
      double *e_weights_by_rank = e_weights_buf;
      cutoff = (knn && uhood->n >= knn) ? uhood->a[0].d : 1.0;
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        double *wacc_base = &wacc->a[(int64_t) n_ranks * iv];
        if (wacc_base[0] < 0)
          continue;
        size_t nvbits;
        int64_t *vbits = tk_inv_sget(inv, vsid, &nvbits);
        tk_inv_compute_candidate_weights_by_rank(inv, vbits, nvbits, e_weights_by_rank);
        double sim = tk_inv_similarity_by_rank_fast(n_ranks, wacc->a, iv, q_weights_by_rank, e_weights_by_rank, bandwidth, cutoff, &rw);
        double dist = 1.0 - sim;
        if (dist <= cutoff) {
          if (knn)
            tk_rvec_hmax(uhood, knn, tk_rank(iv, dist));
          else
            tk_rvec_push(uhood, tk_rank(iv, dist));
          if (knn && uhood->n >= knn)
            cutoff = uhood->a[0].d;
        }
      }
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        for (uint64_t r = 0; r < n_ranks; r ++)
          wacc->a[(int64_t) n_ranks * iv + (int64_t) r] = 0.0;
      }
      tk_rvec_asc(uhood, 0, uhood->n);
      touched->n = 0;
    }

    tk_dvec_destroy(wacc);
    tk_ivec_destroy(touched);
  }

  tk_ivec_destroy(sid_to_pos);
  lua_pop(L, 2);

  tk_lua_get_ephemeron(L, TK_INV_EPH, all_uids);
  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = all_uids;
}

static inline double tk_inv_w (
  tk_dvec_t *W,
  int64_t fid
) {
  return W->a[fid];
}

static inline void tk_inv_precompute_rank_weights (
  tk_inv_rank_weights_t *rw,
  uint64_t n_ranks,
  double decay
) {
  double abs_decay = fabs(decay);
  rw->n_ranks = n_ranks;
  rw->total = 0.0;
  for (uint64_t r = 0; r < n_ranks && r < TK_INV_MAX_RANKS; r++) {
    double eff_r = (decay >= 0.0) ? (double)r : (double)(n_ranks - 1 - r);
    rw->weights[r] = exp(-eff_r * abs_decay);
    rw->total += rw->weights[r];
  }
}

static inline double tk_inv_combine_ranks (
  double *i_arr, double *q_arr, double *e_arr,
  uint64_t n_ranks,
  double bandwidth,
  tk_inv_rank_weights_t *rw
) {
  double accum = 0.0;
  for (uint64_t r = 0; r < n_ranks; r++) {
    double denom = sqrt(q_arr[r] * e_arr[r]);
    double s = (denom > 0.0) ? i_arr[r] / denom : 0.0;
    accum += rw->weights[r] * s;
  }
  double avg_sim = (rw->total > 0.0) ? accum / rw->total : 0.0;
  if (bandwidth >= 0.0) {
    double distance = 1.0 - avg_sim;
    return exp(-distance * bandwidth);
  }
  return avg_sim;
}

static inline double tk_inv_similarity_fast (
  int64_t *ranks_arr,
  double *weights_arr,
  uint64_t n_ranks,
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize,
  double bandwidth,
  tk_inv_rank_weights_t *rw,
  double *q_arr,
  double *e_arr,
  double *i_arr
) {
  for (uint64_t r = 0; r < n_ranks; r++) {
    q_arr[r] = 0.0;
    e_arr[r] = 0.0;
    i_arr[r] = 0.0;
  }
  size_t i = 0, j = 0;
  while (i < asize && j < bsize) {
    if (abits[i] == bbits[j]) {
      int64_t fid = abits[i];
      int64_t rank = ranks_arr[fid];
      double w = weights_arr[fid];
      i_arr[rank] += w;
      q_arr[rank] += w;
      e_arr[rank] += w;
      i++; j++;
    } else if (abits[i] < bbits[j]) {
      int64_t fid = abits[i];
      int64_t rank = ranks_arr[fid];
      q_arr[rank] += weights_arr[fid];
      i++;
    } else {
      int64_t fid = bbits[j];
      int64_t rank = ranks_arr[fid];
      e_arr[rank] += weights_arr[fid];
      j++;
    }
  }
  while (i < asize) {
    int64_t fid = abits[i];
    int64_t rank = ranks_arr[fid];
    q_arr[rank] += weights_arr[fid];
    i++;
  }
  while (j < bsize) {
    int64_t fid = bbits[j];
    int64_t rank = ranks_arr[fid];
    e_arr[rank] += weights_arr[fid];
    j++;
  }
  return tk_inv_combine_ranks(i_arr, q_arr, e_arr, n_ranks, bandwidth, rw);
}

static inline double tk_inv_similarity (
  tk_inv_t *inv,
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize,
  double decay,
  double bandwidth
) {
  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, inv->n_ranks, decay);
  double q_arr[TK_INV_MAX_RANKS] = {0};
  double e_arr[TK_INV_MAX_RANKS] = {0};
  double i_arr[TK_INV_MAX_RANKS] = {0};
  return tk_inv_similarity_fast(inv->ranks->a, inv->weights->a, inv->n_ranks,
    abits, asize, bbits, bsize, bandwidth, &rw, q_arr, e_arr, i_arr);
}

static inline double tk_inv_distance (
  tk_inv_t *inv,
  int64_t uid0,
  int64_t uid1,
  double decay,
  double bandwidth
) {
  size_t n0 = 0, n1 = 0;
  int64_t *v0 = tk_inv_get(inv, uid0, &n0);
  if (v0 == NULL)
    return 1.0;
  int64_t *v1 = tk_inv_get(inv, uid1, &n1);
  if (v1 == NULL)
    return 1.0;
  return 1.0 - tk_inv_similarity(inv, v0, n0, v1, n1, decay, bandwidth);
}

static inline void tk_inv_compute_query_weights_by_rank (
  tk_inv_t *inv,
  int64_t *data,
  size_t datalen,
  double *q_weights_by_rank
) {
  for (uint64_t r = 0; r < inv->n_ranks; r ++)
    q_weights_by_rank[r] = 0.0;
  for (size_t i = 0; i < datalen; i ++) {
    int64_t fid = data[i];
    if (fid >= 0 && fid < (int64_t) inv->features) {
      int64_t rank = inv->ranks->a[fid];
      if (rank >= 0 && rank < (int64_t) inv->n_ranks) {
        q_weights_by_rank[rank] += tk_inv_w(inv->weights, fid);
      }
    }
  }
}

static inline void tk_inv_compute_candidate_weights_by_rank (
  tk_inv_t *inv,
  int64_t *features,
  size_t nfeatures,
  double *e_weights_by_rank
) {
  for (uint64_t r = 0; r < inv->n_ranks; r ++)
    e_weights_by_rank[r] = 0.0;
  for (size_t i = 0; i < nfeatures; i ++) {
    int64_t fid = features[i];
    if (fid >= 0 && fid < (int64_t) inv->features) {
      int64_t rank = inv->ranks->a[fid];
      if (rank >= 0 && rank < (int64_t) inv->n_ranks) {
        e_weights_by_rank[rank] += tk_inv_w(inv->weights, fid);
      }
    }
  }
}

static inline double tk_inv_similarity_by_rank_fast (
  uint64_t n_ranks,
  double *wacc_arr,
  int64_t vsid,
  double *q_weights_by_rank,
  double *e_weights_by_rank,
  double bandwidth,
  double cutoff,
  tk_inv_rank_weights_t *rw
) {
  double accum = 0.0;
  double total_weight = 0.0;
  double min_required_sim = 1.0 - cutoff;
  for (uint64_t rank = 0; rank < n_ranks; rank++) {
    double weight = rw->weights[rank];
    double inter_w = wacc_arr[(int64_t)n_ranks * vsid + (int64_t)rank];
    double q_w = q_weights_by_rank[rank];
    double e_w = e_weights_by_rank[rank];
    double denom = sqrt(q_w * e_w);
    double rank_sim = (denom > 0.0) ? inter_w / denom : 0.0;
    accum += weight * rank_sim;
    total_weight += weight;
    if (cutoff < 1.0 && rw->total > 0.0) {
      double remaining_weight = rw->total - total_weight;
      double max_possible_sim = (accum + remaining_weight) / rw->total;
      if (max_possible_sim < min_required_sim)
        return 0.0;
    }
  }
  double avg_sim = (rw->total > 0.0) ? accum / rw->total : 0.0;
  if (bandwidth >= 0.0) {
    double distance = 1.0 - avg_sim;
    return exp(-distance * bandwidth);
  }
  return avg_sim;
}

static inline double tk_inv_similarity_by_rank (
  tk_inv_t *inv,
  tk_dvec_t *wacc,
  int64_t vsid,
  double *q_weights_by_rank,
  double *e_weights_by_rank,
  double decay,
  double bandwidth,
  double cutoff
) {
  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, inv->n_ranks, decay);
  return tk_inv_similarity_by_rank_fast(
    inv->n_ranks, wacc->a, vsid, q_weights_by_rank, e_weights_by_rank,
    bandwidth, cutoff, &rw);
}

static inline tk_rvec_t *tk_inv_neighbors_by_vec (
  tk_inv_t *inv,
  int64_t *data,
  size_t datalen,
  int64_t sid0,
  uint64_t knn,
  tk_rvec_t *out,
  double decay,
  double bandwidth
) {
  if (datalen == 0) {
    tk_rvec_clear(out);
    return out;
  }

  tk_rvec_clear(out);
  size_t n_sids = inv->node_offsets->n;
  uint64_t n_ranks = inv->n_ranks;

  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, n_ranks, decay);

  double tmp_q_weights[TK_INV_MAX_RANKS];
  double tmp_e_weights[TK_INV_MAX_RANKS];
  tk_dvec_t *wacc = tk_dvec_create(NULL, n_sids * n_ranks, 0, 0);
  tk_ivec_t *touched = tk_ivec_create(NULL, 0, 0, 0);

  double *q_weights_by_rank = tmp_q_weights;
  tk_inv_compute_query_weights_by_rank(inv, data, datalen, q_weights_by_rank);

  wacc->n = n_sids * n_ranks;
  for (uint64_t i = 0; i < wacc->n; i++)
    wacc->a[i] = 0.0;

  double min_required_sim = 0.0;
  double gate = 1.0;

  for (uint64_t current_rank = 0; current_rank < n_ranks; current_rank ++) {
    bool can_prune_new = gate < min_required_sim;
    for (size_t i = 0; i < datalen; i ++) {
      int64_t fid = data[i];
      if (fid < 0 || fid >= (int64_t) inv->postings->n)
        continue;
      int64_t fid_rank = inv->ranks->a[fid];
      if ((uint64_t)fid_rank != current_rank)
        continue;
      double wf = tk_inv_w(inv->weights, fid);
      tk_ivec_t *vsids = inv->postings->a[fid];
      for (uint64_t j = 0; j < vsids->n; j ++) {
        int64_t vsid = vsids->a[j];
        if (vsid == sid0)
          continue;
        double *wacc_base = &wacc->a[(int64_t) n_ranks * vsid];
        if (wacc_base[0] < 0)
          continue;
        bool first_touch = true;
        for (uint64_t r = 0; r < n_ranks; r ++) {
          if (wacc_base[r] != 0.0) {
            first_touch = false;
            break;
          }
        }
        if (first_touch) {
          if (can_prune_new) {
            wacc_base[0] = -1.0;
            tk_ivec_push(touched, vsid);
            continue;
          }
          tk_ivec_push(touched, vsid);
        }
        wacc_base[current_rank] += wf;
      }
    }
    gate *= decay;
  }

  touched->n = tk_ivec_uasc(touched, 0, touched->n);
  double *e_weights_by_rank = tmp_e_weights;

  for (uint64_t i = 0; i < touched->n; i ++) {
    int64_t vsid = touched->a[i];
    double *wacc_base = &wacc->a[(int64_t) n_ranks * vsid];
    if (wacc_base[0] < 0) {
      for (uint64_t r = 0; r < n_ranks; r ++)
        wacc_base[r] = 0.0;
      continue;
    }

    size_t elen = 0;
    int64_t *ev = tk_inv_sget(inv, vsid, &elen);
    tk_inv_compute_candidate_weights_by_rank(inv, ev, elen, e_weights_by_rank);
    double current_cutoff = (knn && out->n >= knn) ? out->a[0].d : 1.0;
    double sim = tk_inv_similarity_by_rank_fast(n_ranks, wacc->a, vsid, q_weights_by_rank, e_weights_by_rank, bandwidth, current_cutoff, &rw);
    double dist = 1.0 - sim;
    if (dist <= current_cutoff) {
      int64_t vuid = tk_inv_sid_uid(inv, vsid);
      if (vuid >= 0) {
        if (knn)
          tk_rvec_hmax(out, knn, tk_rank(vuid, dist));
        else
          tk_rvec_push(out, tk_rank(vuid, dist));
      }
    }
    for (uint64_t r = 0; r < n_ranks; r ++)
      wacc_base[r] = 0.0;
  }

  tk_rvec_asc(out, 0, out->n);

  tk_dvec_destroy(wacc);
  tk_ivec_destroy(touched);

  return out;
}

static inline tk_rvec_t *tk_inv_neighbors_by_id (
  tk_inv_t *inv,
  int64_t uid,
  uint64_t knn,
  tk_rvec_t *out,
  double decay,
  double bandwidth
) {
  int64_t sid0 = tk_inv_uid_sid(inv, uid, false);
  if (sid0 < 0) {
    tk_rvec_clear(out);
    return out;
  }
  size_t len = 0;
  int64_t *data = tk_inv_get(inv, uid, &len);
  return tk_inv_neighbors_by_vec(inv, data, len, sid0, knn, out, decay, bandwidth);
}

static inline int tk_inv_gc_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_inv_destroy( inv );
  return 0;
}

static inline int tk_inv_add_lua (lua_State *L)
{
  int Ii = 1;
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_ivec_t *node_bits = tk_ivec_peek(L, 2, "node_bits");
  if (lua_type(L, 3) == LUA_TNUMBER) {
    int64_t s = (int64_t) tk_lua_checkunsigned(L, 3, "base_id");
    uint64_t n = tk_lua_checkunsigned(L, 4, "n_nodes");
    tk_ivec_t *ids = tk_ivec_create(L, n, 0, 0);
    tk_ivec_fill_indices(ids);
    tk_ivec_add(ids, s, 0, ids->n);
    tk_inv_add(L, inv, Ii, ids, node_bits);
    lua_pop(L, 1);
  } else {
    tk_inv_add(L, inv, Ii, tk_ivec_peek(L, 3, "ids"), node_bits);
  }
  return 0;
}

static inline int tk_inv_remove_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t id = tk_lua_checkinteger(L, 2, "id");
    tk_inv_remove(L, inv, id);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
    for (uint64_t i = 0; i < ids->n; i ++) {
      tk_inv_uid_remove(inv, ids->a[i]);
    }
  }
  return 0;
}

static inline int tk_inv_keep_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t id = tk_lua_checkinteger(L, 2, "id");
    tk_ivec_t *ids = tk_ivec_create(L, 1, 0, 0);
    ids->a[0] = id;
    ids->n = 1;
    tk_inv_keep(L, inv, ids);
    lua_pop(L, 1);
  } else {
    tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
    tk_inv_keep(L, inv, ids);
  }
  return 0;
}

static inline int tk_inv_get_lua (lua_State *L)
{
  lua_settop(L, 4);
  tk_inv_t *inv = tk_inv_peek(L, 1);
  int64_t uid = -1;
  tk_ivec_t *uids = NULL;
  tk_ivec_t *out = tk_ivec_peekopt(L, 3);
  out = out == NULL ? tk_ivec_create(L, 0, 0, 0) : out;
  bool append = tk_lua_optboolean(L, 4, "append", false);
  if (!append)
    tk_ivec_clear(out);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    uid = tk_lua_checkinteger(L, 2, "id");
    size_t n = 0;
    int64_t *data = tk_inv_get(inv, uid, &n);
    if (!n)
      return 1;
    tk_ivec_ensure(out, n);
    memcpy(out->a, data, n * sizeof(int64_t));
    out->n = n;
  } else {
    uids = lua_isnil(L, 2) ? tk_iumap_keys(L, inv->uid_sid) : tk_ivec_peek(L, 2, "uids");
    size_t total_size = 0;
    for (uint64_t i = 0; i < uids->n; i ++) {
      uid = uids->a[i];
      size_t n = 0;
      tk_inv_get(inv, uid, &n);
      total_size += n;
    }    if (total_size > 0) {
      tk_ivec_ensure(out, out->n + total_size);
    }    for (uint64_t i = 0; i < uids->n; i ++) {
      uid = uids->a[i];
      size_t n = 0;
      int64_t *data = tk_inv_get(inv, uid, &n);
      if (!n)
        continue;
      for (size_t j = 0; j < n; j ++)
        out->a[out->n ++] = data[j] + (int64_t) (i * inv->features);
    }
  }
  return 1;
}

static inline int tk_inv_neighborhoods_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  uint64_t knn = tk_lua_checkunsigned(L, 2, "knn");
  double decay = tk_lua_optnumber(L, 3, "decay", 0.0);
  double bandwidth = tk_lua_optnumber(L, 4, "bandwidth", -1.0);
  tk_inv_neighborhoods(L, inv, knn, decay, bandwidth, NULL, NULL);
  return 2;
}

static inline int tk_inv_neighborhoods_by_ids_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_ivec_t *query_ids = tk_ivec_peek(L, 2, "ids");
  uint64_t knn = tk_lua_checkunsigned(L, 3, "knn");
  double decay = tk_lua_optnumber(L, 4, "decay", 0.0);
  double bandwidth = tk_lua_optnumber(L, 5, "bandwidth", -1.0);
  int64_t write_pos = 0;
  for (int64_t i = 0; i < (int64_t) query_ids->n; i ++) {
    int64_t uid = query_ids->a[i];
    khint_t k = tk_iumap_get(inv->uid_sid, uid);
    if (k != tk_iumap_end(inv->uid_sid)) {
      query_ids->a[write_pos ++] = uid;
    }
  }
  query_ids->n = (uint64_t) write_pos;

  tk_inv_hoods_t *hoods;
  tk_inv_neighborhoods_by_ids(L, inv, query_ids, knn, decay, bandwidth, &hoods, &query_ids);
  return 2;
}

static inline int tk_inv_neighborhoods_by_vecs_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_ivec_t *query_vecs = tk_ivec_peek(L, 2, "vectors");
  uint64_t knn = tk_lua_checkunsigned(L, 3, "knn");
  double decay = tk_lua_optnumber(L, 4, "decay", 0.0);
  double bandwidth = tk_lua_optnumber(L, 5, "bandwidth", -1.0);
  tk_inv_neighborhoods_by_vecs(L, inv, query_vecs, knn, decay, bandwidth, NULL, NULL);
  return 2;
}

static inline int tk_inv_similarity_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  int64_t uid0 = tk_lua_checkinteger(L, 2, "uid0");
  int64_t uid1 = tk_lua_checkinteger(L, 3, "uid1");
  double decay = tk_lua_optnumber(L, 4, "decay", 0.0);
  double bandwidth = tk_lua_optnumber(L, 5, "bandwidth", -1.0);
  lua_pushnumber(L, 1.0 - tk_inv_distance(inv, uid0, uid1, decay, bandwidth));
  return 1;
}

static inline int tk_inv_distance_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  int64_t uid0 = tk_lua_checkinteger(L, 2, "uid0");
  int64_t uid1 = tk_lua_checkinteger(L, 3, "uid1");
  double decay = tk_lua_optnumber(L, 4, "decay", 0.0);
  double bandwidth = tk_lua_optnumber(L, 5, "bandwidth", -1.0);
  lua_pushnumber(L, tk_inv_distance(inv, uid0, uid1, decay, bandwidth));
  return 1;
}

static inline int tk_inv_neighbors_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  uint64_t knn = tk_lua_optunsigned(L, 3, "knn", 0);
  tk_rvec_t *out = tk_rvec_peek(L, 4, "out");
  double decay = tk_lua_optnumber(L, 5, "decay", 0.0);
  double bandwidth = tk_lua_optnumber(L, 6, "bandwidth", -1.0);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_inv_neighbors_by_id(inv, uid, knn, out, decay, bandwidth);
  } else {
    tk_ivec_t *vec = tk_ivec_peek(L, 2, "vector");
    tk_inv_neighbors_by_vec(inv, vec->a, vec->n, -1, knn, out, decay, bandwidth);
  }
  return 0;
}

static inline int tk_inv_size_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) tk_inv_size( inv ));
  return 1;
}

static inline int tk_inv_features_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  lua_pushinteger(L, (int64_t) inv->features);
  return 1;
}

static inline int tk_inv_weights_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_lua_get_ephemeron(L, TK_INV_EPH, inv->weights);
  return 1;
}

static inline int tk_inv_ranks_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_lua_get_ephemeron(L, TK_INV_EPH, inv->ranks);
  return 1;
}

static inline int tk_inv_rank_sizes_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_lua_get_ephemeron(L, TK_INV_EPH, inv->rank_sizes);
  return 1;
}

static inline int tk_inv_persist_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_inv_persist(L, inv, fh);
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
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_inv_destroy( inv );
  return 0;
}

static inline int tk_inv_shrink_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_inv_shrink(L, inv);
  return 0;
}

static inline tk_ivec_t *tk_inv_ids (lua_State *L, tk_inv_t *inv)
{
  return tk_iumap_keys(L, inv->uid_sid);
}

static inline int tk_inv_weight_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  uint64_t fid = tk_lua_checkunsigned(L, 2, "fid");
  lua_pushnumber(L, tk_inv_w(inv->weights, (int64_t) fid));
  return 1;
}

static inline int tk_inv_ids_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_iumap_keys(L, inv->uid_sid);
  return 1;
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
  { "features", tk_inv_features_lua },
  { "weights", tk_inv_weights_lua },
  { "ranks", tk_inv_ranks_lua },
  { "rank_sizes", tk_inv_rank_sizes_lua },
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
  int i_weights,
  int i_ranks
) {
  if (!features)
    tk_lua_verror(L, 2, "create", "features must be > 0");
  if (n_ranks > TK_INV_MAX_RANKS)
    return luaL_error(L, "n_ranks (%llu) exceeds TK_INV_MAX_RANKS (%d)",
      (unsigned long long)n_ranks, TK_INV_MAX_RANKS), NULL;
  tk_inv_t *inv = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  int Ii = tk_lua_absindex(L, -1);
  inv->destroyed = false;
  inv->next_sid = 0;
  inv->features = features;
  inv->n_ranks = n_ranks >= 1 ? n_ranks : 1;
  if (weights) {
    inv->weights = weights;
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, i_weights);
  } else {
    inv->weights = tk_dvec_create(L, features, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    inv->weights->n = features;
    for (uint64_t f = 0; f < features; f++)
      inv->weights->a[f] = 1.0;
  }
  if (ranks) {
    inv->ranks = ranks;
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, i_ranks);
  } else {
    inv->ranks = tk_ivec_create(L, features, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    inv->ranks->n = features;
    for (uint64_t f = 0; f < features; f++)
      inv->ranks->a[f] = 0;
  }
  inv->rank_sizes = tk_ivec_create(L, inv->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  for (uint64_t r = 0; r < inv->n_ranks; r++)
    inv->rank_sizes->a[r] = 0;
  for (uint64_t f = 0; f < features; f++) {
    int64_t rank = inv->ranks->a[f];
    if (rank >= 0 && rank < (int64_t)inv->n_ranks)
      inv->rank_sizes->a[rank]++;
  }
  inv->uid_sid = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  inv->sid_to_uid = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  inv->node_offsets = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  inv->node_bits = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  inv->postings = tk_inv_postings_create(L, features, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  for (uint64_t i = 0; i < features; i ++) {
    inv->postings->a[i] = tk_ivec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
  }
  lua_pop(L, 1);
  return inv;
}

static inline tk_inv_t *tk_inv_load (
  lua_State *L,
  FILE *fh
) {
  tk_inv_t *inv = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  int Ii = tk_lua_absindex(L, -1);
  memset(inv, 0, sizeof(tk_inv_t));
  tk_lua_fread(L, &inv->destroyed, sizeof(bool), 1, fh);
  if (inv->destroyed)
    tk_lua_verror(L, 2, "load", "index was destroyed when saved");
  tk_lua_fread(L, &inv->next_sid, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &inv->features, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &inv->n_ranks, sizeof(uint64_t), 1, fh);
  if (inv->n_ranks > TK_INV_MAX_RANKS)
    tk_lua_verror(L, 2, "load", "n_ranks in file exceeds TK_INV_MAX_RANKS");
  inv->uid_sid = tk_iumap_load(L, fh);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  inv->sid_to_uid = tk_ivec_load(L, fh);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  inv->node_offsets = tk_ivec_load(L, fh);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  inv->node_bits = tk_ivec_load(L, fh);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  uint64_t pcount = 0;
  tk_lua_fread(L, &pcount, sizeof(uint64_t), 1, fh);
  inv->postings = tk_inv_postings_create(L, pcount, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  for (uint64_t i = 0; i < pcount; i ++) {
    uint64_t plen;
    tk_lua_fread(L, &plen, sizeof(uint64_t), 1, fh);
    tk_inv_posting_t P = tk_ivec_create(L, plen, 0, 0);
    if (plen)
      tk_lua_fread(L, P->a, sizeof(int64_t), plen, fh);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    inv->postings->a[i] = P;
  }
  lua_pop(L, 1);
  size_t wn = 0;
  tk_lua_fread(L, &wn, sizeof(size_t), 1, fh);
  if (wn) {
    tk_lua_fseek(L, -sizeof(size_t), 1, fh);
    inv->weights = tk_dvec_load(L, fh);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
  } else {
    inv->weights = tk_dvec_create(L, inv->features, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    inv->weights->n = inv->features;
    for (uint64_t f = 0; f < inv->features; f++)
      inv->weights->a[f] = 1.0;
  }
  size_t rn = 0;
  tk_lua_fread(L, &rn, sizeof(size_t), 1, fh);
  if (rn) {
    tk_lua_fseek(L, -sizeof(size_t), 1, fh);
    inv->ranks = tk_ivec_load(L, fh);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
  } else {
    inv->ranks = tk_ivec_create(L, inv->features, 0, 0);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
    inv->ranks->n = inv->features;
    for (uint64_t f = 0; f < inv->features; f++)
      inv->ranks->a[f] = 0;
  }

  inv->rank_sizes = tk_ivec_create(L, inv->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  for (uint64_t r = 0; r < inv->n_ranks; r++)
    inv->rank_sizes->a[r] = 0;
  for (uint64_t f = 0; f < inv->features; f++) {
    int64_t rank = inv->ranks->a[f];
    if (rank >= 0 && rank < (int64_t)inv->n_ranks)
      inv->rank_sizes->a[rank]++;
  }

  return inv;
}

#endif
