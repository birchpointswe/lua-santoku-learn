#ifndef TK_INV_H
#define TK_INV_H

#include <math.h>
#include <float.h>
#include <string.h>
#include <omp.h>
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
  double decay;
  double total_rank_weight;
  tk_dvec_t *rank_weights;
  tk_ivec_t *rank_sizes;
  tk_iumap_t *uid_sid;
  tk_ivec_t *sid_to_uid;
  tk_ivec_t *node_offsets;
  tk_ivec_t *node_bits;
  tk_inv_postings_t *postings;
} tk_inv_t;

static inline double tk_inv_w (tk_dvec_t *W, int64_t fid);
static inline double tk_inv_similarity (
  tk_inv_t *inv,
  int64_t *a, size_t na,
  int64_t *b, size_t nb,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
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
static inline double tk_inv_similarity_by_rank (
  tk_inv_t *inv,
  tk_dvec_t *wacc,
  int64_t v,
  double *q_weights_by_rank,
  double *e_weights_by_rank,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  tk_dvec_t *q_weights_buf,
  tk_dvec_t *e_weights_buf,
  tk_dvec_t *inter_weights_buf
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
    if (inv->weights) tk_dvec_shrink(inv->weights);
    if (inv->ranks) tk_ivec_shrink(inv->ranks);
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
  if (inv->weights) tk_dvec_shrink(inv->weights);
  if (inv->ranks) tk_ivec_shrink(inv->ranks);
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
  size_t wn = inv->weights ? inv->weights->n : 0;
  if (wn)
    tk_dvec_persist(L, inv->weights, fh);
  else
    tk_lua_fwrite(L, (char *) &wn, sizeof(size_t), 1, fh);
  size_t rn = inv->ranks ? inv->ranks->n : 0;
  if (rn)
    tk_ivec_persist(L, inv->ranks, fh);
  else
    tk_lua_fwrite(L, (char *) &rn, sizeof(size_t), 1, fh);
  tk_lua_fwrite(L, (char *) &inv->decay, sizeof(double), 1, fh);
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
    if (tk_ivec_push(inv->node_offsets, (int64_t) inv->node_bits->n) != 0) {
      tk_lua_verror(L, 2, "add", "allocation failed during indexing");
      return;
    }
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
  }
  if (tk_ivec_push(inv->node_offsets, (int64_t) inv->node_bits->n) != 0) {
    tk_lua_verror(L, 2, "add", "allocation failed during indexing");
    return;
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
  double eps_min,
  double eps_max,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter,
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

  #pragma omp parallel
  {
    tk_dvec_t *wacc = tk_dvec_create(NULL, uids->n * inv->n_ranks, 0, 0);
    tk_ivec_t *touched = tk_ivec_create(NULL, 0, 0, 0);
    tk_dvec_t *q_weights_buf = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
    tk_dvec_t *e_weights_buf = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
    tk_dvec_t *inter_weights_buf = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
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
      double *q_weights_by_rank = q_weights_buf->a;
      tk_inv_compute_query_weights_by_rank(inv, ubits, nubits, q_weights_by_rank);
      touched->n = 0;
      for (size_t k = 0; k < nubits; k ++) {
        int64_t fid = ubits[k];
        int64_t rank = inv->ranks ? inv->ranks->a[fid] : 0;
        if (rank_filter >= 0 && rank != rank_filter)
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
          if (wacc->a[(int64_t) inv->n_ranks * iv + rank] == 0.0)
            tk_ivec_push(touched, vsid);
          wacc->a[(int64_t) inv->n_ranks * iv + rank] += wf;
        }
      }
      touched->n = tk_ivec_uasc(touched, 0, touched->n);
      double *e_weights_by_rank = e_weights_buf->a;
      double cutoff = (uhood->n >= knn) ? uhood->a[0].d : eps_max;
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        if (uhood->n >= knn) {
          double inter_weight = 0.0;
          for (uint64_t r = 0; r < inv->n_ranks; r++) {
            double inter = wacc->a[(int64_t) inv->n_ranks * iv + (int64_t) r];
            if (inter > 0.0) {
              inter_weight += inter * inv->rank_weights->a[r];
            }
          }
          double query_weight = 0.0;
          for (uint64_t r = 0; r < inv->n_ranks; r++)
            query_weight += q_weights_by_rank[r] * inv->rank_weights->a[r];
          double max_possible_sim = (query_weight > 0.0) ? (inter_weight / query_weight) : 0.0;
          if (1.0 - max_possible_sim > cutoff)
            continue;
        }
        size_t nvbits;
        int64_t *vbits = tk_inv_sget(inv, vsid, &nvbits);
        tk_inv_compute_candidate_weights_by_rank(inv, vbits, nvbits, e_weights_by_rank);
        double sim = tk_inv_similarity_by_rank(inv, wacc, iv, q_weights_by_rank, e_weights_by_rank, cmp, tversky_alpha, tversky_beta, q_weights_buf, e_weights_buf, inter_weights_buf);
        double dist = 1.0 - sim;
        if (dist >= eps_min && dist <= cutoff) {
          tk_rvec_hmax(uhood, knn, tk_rank(iv, dist));
          if (uhood->n >= knn)
            cutoff = uhood->a[0].d;
        }
      }
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        for (uint64_t r = 0; r < inv->n_ranks; r ++)
          wacc->a[(int64_t) inv->n_ranks * iv + (int64_t) r] = 0.0;
      }
      tk_rvec_asc(uhood, 0, uhood->n);
      touched->n = 0;
    }

    tk_dvec_destroy(wacc);
    tk_ivec_destroy(touched);
    tk_dvec_destroy(q_weights_buf);
    tk_dvec_destroy(e_weights_buf);
    tk_dvec_destroy(inter_weights_buf);
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
  double eps_min,
  double eps_max,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter,
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
    hoods->a[i] = tk_rvec_create(L, knn, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
  }

  #pragma omp parallel
  {
    tk_dvec_t *wacc = tk_dvec_create(NULL, all_uids->n * inv->n_ranks, 0, 0);
    tk_ivec_t *touched = tk_ivec_create(NULL, 0, 0, 0);
    tk_dvec_t *q_weights_buf = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
    tk_dvec_t *e_weights_buf = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
    tk_dvec_t *inter_weights_buf = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
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
      double *q_weights_by_rank = q_weights_buf->a;
      tk_inv_compute_query_weights_by_rank(inv, ubits, nubits, q_weights_by_rank);
      touched->n = 0;
      for (size_t k = 0; k < nubits; k ++) {
        int64_t fid = ubits[k];
        int64_t rank = inv->ranks ? inv->ranks->a[fid] : 0;
        if (rank_filter >= 0 && rank != rank_filter)
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
          if (wacc->a[(int64_t) inv->n_ranks * iv + rank] == 0.0)
            tk_ivec_push(touched, vsid);
          wacc->a[(int64_t) inv->n_ranks * iv + rank] += wf;
        }
      }
      touched->n = tk_ivec_uasc(touched, 0, touched->n);
      double *e_weights_by_rank = e_weights_buf->a;
      double cutoff = (uhood->n >= knn) ? uhood->a[0].d : eps_max;
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        if (uhood->n >= knn) {
          double inter_weight = 0.0;
          for (uint64_t r = 0; r < inv->n_ranks; r++) {
            double inter = wacc->a[(int64_t) inv->n_ranks * iv + (int64_t) r];
            if (inter > 0.0) {
              inter_weight += inter * inv->rank_weights->a[r];
            }
          }
          double query_weight = 0.0;
          for (uint64_t r = 0; r < inv->n_ranks; r++)
            query_weight += q_weights_by_rank[r] * inv->rank_weights->a[r];
          double max_possible_sim = (query_weight > 0.0) ? (inter_weight / query_weight) : 0.0;
          if (1.0 - max_possible_sim > cutoff)
            continue;
        }
        size_t nvbits;
        int64_t *vbits = tk_inv_sget(inv, vsid, &nvbits);
        tk_inv_compute_candidate_weights_by_rank(inv, vbits, nvbits, e_weights_by_rank);
        double sim = tk_inv_similarity_by_rank(inv, wacc, iv, q_weights_by_rank, e_weights_by_rank, cmp, tversky_alpha, tversky_beta, q_weights_buf, e_weights_buf, inter_weights_buf);
        double dist = 1.0 - sim;
        if (dist >= eps_min && dist <= cutoff) {
          tk_rvec_hmax(uhood, knn, tk_rank(iv, dist));
          if (uhood->n >= knn)
            cutoff = uhood->a[0].d;
        }
      }
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        for (uint64_t r = 0; r < inv->n_ranks; r ++)
          wacc->a[(int64_t) inv->n_ranks * iv + (int64_t) r] = 0.0;
      }
      tk_rvec_asc(uhood, 0, uhood->n);
      touched->n = 0;
    }

    tk_dvec_destroy(wacc);
    tk_ivec_destroy(touched);
    tk_dvec_destroy(q_weights_buf);
    tk_dvec_destroy(e_weights_buf);
    tk_dvec_destroy(inter_weights_buf);
  }

  tk_ivec_destroy(sid_to_pos);

  if (hoodsp) *hoodsp = hoods;
  if (uidsp) *uidsp = all_uids;
  lua_remove(L, -2);
}

static inline void tk_inv_neighborhoods_by_vecs (
  lua_State *L,
  tk_inv_t *inv,
  tk_ivec_t *query_vecs,
  uint64_t knn,
  double eps_min,
  double eps_max,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter,
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
    hoods->a[i] = tk_rvec_create(L, knn, 0, 0);
    hoods->a[i]->n = 0;
    tk_lua_add_ephemeron(L, TK_INV_EPH, hoods_stack_idx, -1);
    lua_pop(L, 1);
  }

  tk_ivec_t *all_uids, *sid_to_pos;
  tk_inv_prepare_universe_map(L, inv, &all_uids, &sid_to_pos);

  #pragma omp parallel
  {
    tk_dvec_t *wacc = tk_dvec_create(NULL, all_uids->n * inv->n_ranks, 0, 0);
    tk_ivec_t *touched = tk_ivec_create(NULL, 0, 0, 0);
    tk_dvec_t *q_weights_buf = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
    tk_dvec_t *e_weights_buf = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
    tk_dvec_t *inter_weights_buf = tk_dvec_create(NULL, inv->n_ranks, 0, 0);

    #pragma omp for schedule(static) nowait
    for (int64_t i = 0; i < (int64_t) hoods->n; i ++) {
      tk_rvec_t *uhood = hoods->a[i];
      tk_rvec_clear(uhood);
      int64_t start = query_offsets->a[i];
      int64_t end = (i + 1 < (int64_t) query_offsets->n) ? query_offsets->a[i + 1] : (int64_t) query_features->n;
      int64_t *ubits = query_features->a + start;
      size_t nubits = (size_t)(end - start);
      double *q_weights_by_rank = q_weights_buf->a;
      tk_inv_compute_query_weights_by_rank(inv, ubits, nubits, q_weights_by_rank);
      touched->n = 0;
      for (size_t k = 0; k < nubits; k ++) {
        int64_t fid = ubits[k];
        int64_t rank = inv->ranks ? inv->ranks->a[fid] : 0;
        if (rank_filter >= 0 && rank != rank_filter)
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
          if (wacc->a[(int64_t) inv->n_ranks * iv + rank] == 0.0)
            tk_ivec_push(touched, vsid);
          wacc->a[(int64_t) inv->n_ranks * iv + rank] += wf;
        }
      }
      touched->n = tk_ivec_uasc(touched, 0, touched->n);
      double *e_weights_by_rank = e_weights_buf->a;
      double cutoff = (uhood->n >= knn) ? uhood->a[0].d : eps_max;
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        if (uhood->n >= knn) {
          double inter_weight = 0.0;
          for (uint64_t r = 0; r < inv->n_ranks; r++) {
            double inter = wacc->a[(int64_t) inv->n_ranks * iv + (int64_t) r];
            if (inter > 0.0) {
              inter_weight += inter * inv->rank_weights->a[r];
            }
          }
          double query_weight = 0.0;
          for (uint64_t r = 0; r < inv->n_ranks; r++)
            query_weight += q_weights_by_rank[r] * inv->rank_weights->a[r];
          double max_possible_sim = (query_weight > 0.0) ? (inter_weight / query_weight) : 0.0;
          if (1.0 - max_possible_sim > cutoff)
            continue;
        }
        size_t nvbits;
        int64_t *vbits = tk_inv_sget(inv, vsid, &nvbits);
        tk_inv_compute_candidate_weights_by_rank(inv, vbits, nvbits, e_weights_by_rank);
        double sim = tk_inv_similarity_by_rank(inv, wacc, iv, q_weights_by_rank, e_weights_by_rank, cmp, tversky_alpha, tversky_beta, q_weights_buf, e_weights_buf, inter_weights_buf);
        double dist = 1.0 - sim;
        if (dist >= eps_min && dist <= cutoff) {
          tk_rvec_hmax(uhood, knn, tk_rank(iv, dist));
          if (uhood->n >= knn)
            cutoff = uhood->a[0].d;
        }
      }
      for (uint64_t ti = 0; ti < touched->n; ti ++) {
        int64_t vsid = touched->a[ti];
        int64_t iv = sid_to_pos->a[vsid];
        for (uint64_t r = 0; r < inv->n_ranks; r ++)
          wacc->a[(int64_t) inv->n_ranks * iv + (int64_t) r] = 0.0;
      }
      tk_rvec_asc(uhood, 0, uhood->n);
      touched->n = 0;
    }

    tk_dvec_destroy(wacc);
    tk_ivec_destroy(touched);
    tk_dvec_destroy(q_weights_buf);
    tk_dvec_destroy(e_weights_buf);
    tk_dvec_destroy(inter_weights_buf);
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
  if (W == NULL)
    return 1.0;
  return W->a[fid];
}

static inline void tk_inv_stats (
  tk_inv_t *inv,
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
      double w = tk_inv_w(inv->weights, ai);
      inter += w;
      sa += w;
      sb += w;
      i ++;
      j ++;
    } else if (ai < bj) {
      sa += tk_inv_w(inv->weights, ai);
      i ++;
    } else {
      sb += tk_inv_w(inv->weights, bj);
      j ++;
    }
  }
  while (i < alen)
    sa += tk_inv_w(inv->weights, a[i ++]);
  while (j < blen)
    sb += tk_inv_w(inv->weights, b[j ++]);
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

static inline double tk_inv_similarity (
  tk_inv_t *inv,
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
) {
  if (inv->ranks && inv->n_ranks > 1 && q_weights && e_weights && inter_weights) {
    tk_dvec_ensure(q_weights, inv->n_ranks);
    tk_dvec_ensure(e_weights, inv->n_ranks);
    tk_dvec_ensure(inter_weights, inv->n_ranks);

    for (uint64_t r = 0; r < inv->n_ranks; r ++) {
      q_weights->a[r] = 0.0;
      e_weights->a[r] = 0.0;
      inter_weights->a[r] = 0.0;
    }
    size_t i = 0, j = 0;
    while (i < asize && j < bsize) {
      if (abits[i] == bbits[j]) {
        int64_t fid = abits[i];
        int64_t rank = (fid >= 0 && fid < (int64_t)inv->features && inv->ranks) ? inv->ranks->a[fid] : 0;
        if (rank >= 0 && rank < (int64_t)inv->n_ranks) {
          double w = tk_inv_w(inv->weights, fid);
          inter_weights->a[rank] += w;
          q_weights->a[rank] += w;
          e_weights->a[rank] += w;
        }
        i ++; j ++;
      } else if (abits[i] < bbits[j]) {
        int64_t fid = abits[i];
        int64_t rank = (fid >= 0 && fid < (int64_t)inv->features && inv->ranks) ? inv->ranks->a[fid] : 0;
        if (rank >= 0 && rank < (int64_t)inv->n_ranks) {
          q_weights->a[rank] += tk_inv_w(inv->weights, fid);
        }
        i ++;
      } else {
        int64_t fid = bbits[j];
        int64_t rank = (fid >= 0 && fid < (int64_t)inv->features && inv->ranks) ? inv->ranks->a[fid] : 0;
        if (rank >= 0 && rank < (int64_t)inv->n_ranks) {
          e_weights->a[rank] += tk_inv_w(inv->weights, fid);
        }
        j ++;
      }
    }
    while (i < asize) {
      int64_t fid = abits[i];
      int64_t rank = (fid >= 0 && fid < (int64_t)inv->features && inv->ranks) ? inv->ranks->a[fid] : 0;
      if (rank >= 0 && rank < (int64_t)inv->n_ranks) {
        q_weights->a[rank] += tk_inv_w(inv->weights, fid);
      }
      i ++;
    }
    while (j < bsize) {
      int64_t fid = bbits[j];
      int64_t rank = (fid >= 0 && fid < (int64_t)inv->features && inv->ranks) ? inv->ranks->a[fid] : 0;
      if (rank >= 0 && rank < (int64_t)inv->n_ranks) {
        e_weights->a[rank] += tk_inv_w(inv->weights, fid);
      }
      j ++;
    }
    double total_weighted_sim = 0.0;
    for (uint64_t rank = 0; rank < inv->n_ranks; rank ++) {
      double rank_weight = inv->rank_weights->a[rank];
      double rank_sim = tk_inv_similarity_partial(
        inter_weights->a[rank],
        q_weights->a[rank],
        e_weights->a[rank],
        cmp, tversky_alpha, tversky_beta);
      total_weighted_sim += rank_sim * rank_weight;
    }

    return (inv->total_rank_weight > 0.0) ? total_weighted_sim / inv->total_rank_weight : 0.0;
  }
  double inter_w = 0.0, sa = 0.0, sb = 0.0;
  tk_inv_stats(inv, abits, asize, bbits, bsize, &inter_w, &sa, &sb);
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
    default: {
      double u = sa + sb - inter_w;
      return (u == 0.0) ? 0.0 : inter_w / u;
    }
  }
}

static inline double tk_inv_distance (
  tk_inv_t *inv,
  int64_t uid0,
  int64_t uid1,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
) {
  size_t n0 = 0, n1 = 0;
  int64_t *v0 = tk_inv_get(inv, uid0, &n0);
  if (v0 == NULL)
    return 1.0;
  int64_t *v1 = tk_inv_get(inv, uid1, &n1);
  if (v1 == NULL)
    return 1.0;
  double sim = tk_inv_similarity(inv, v0, n0, v1, n1, cmp, tversky_alpha, tversky_beta, q_weights, e_weights, inter_weights);
  return 1.0 - sim;
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
      int64_t rank = inv->ranks ? inv->ranks->a[fid] : 0;
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
      int64_t rank = inv->ranks ? inv->ranks->a[fid] : 0;
      if (rank >= 0 && rank < (int64_t) inv->n_ranks) {
        e_weights_by_rank[rank] += tk_inv_w(inv->weights, fid);
      }
    }
  }
}

static inline double tk_inv_similarity_by_rank (
  tk_inv_t *inv,
  tk_dvec_t *wacc,
  int64_t vsid,
  double *q_weights_by_rank,
  double *e_weights_by_rank,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
) {
  double total_weighted_sim = 0.0;
  for (uint64_t rank = 0; rank < inv->n_ranks; rank ++) {
    double rank_weight = inv->rank_weights->a[rank];
    double inter_w = wacc->a[(int64_t) inv->n_ranks * vsid + (int64_t) rank];
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
  return (inv->total_rank_weight > 0.0) ? total_weighted_sim / inv->total_rank_weight : 0.0;
}

static inline double tk_inv_similarity_rank_filtered (
  tk_inv_t *inv,
  int64_t *abits, size_t asize,
  int64_t *bbits, size_t bsize,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
) {
  if (rank_filter < 0)
    return tk_inv_similarity(inv, abits, asize, bbits, bsize, cmp, tversky_alpha, tversky_beta, q_weights, e_weights, inter_weights);
  double filtered_inter_w = 0.0, filtered_sa = 0.0, filtered_sb = 0.0;
  size_t i = 0, j = 0;
  while (i < asize && j < bsize) {
    if (abits[i] == bbits[j]) {
      int64_t fid = abits[i];
      int64_t rank = (inv->ranks && fid >= 0 && fid < (int64_t)inv->features) ? inv->ranks->a[fid] : 0;
      if (rank == rank_filter) {
        double w = tk_inv_w(inv->weights, fid);
        filtered_inter_w += w;
        filtered_sa += w;
        filtered_sb += w;
      }
      i++; j++;
    } else if (abits[i] < bbits[j]) {
      int64_t fid = abits[i];
      int64_t rank = (inv->ranks && fid >= 0 && fid < (int64_t)inv->features) ? inv->ranks->a[fid] : 0;
      if (rank == rank_filter)
        filtered_sa += tk_inv_w(inv->weights, fid);
      i++;
    } else {
      int64_t fid = bbits[j];
      int64_t rank = (inv->ranks && fid >= 0 && fid < (int64_t)inv->features) ? inv->ranks->a[fid] : 0;
      if (rank == rank_filter)
        filtered_sb += tk_inv_w(inv->weights, fid);
      j++;
    }
  }
  while (i < asize) {
    int64_t fid = abits[i];
    int64_t rank = (inv->ranks && fid >= 0 && fid < (int64_t)inv->features) ? inv->ranks->a[fid] : 0;
    if (rank == rank_filter)
      filtered_sa += tk_inv_w(inv->weights, fid);
    i++;
  }
  while (j < bsize) {
    int64_t fid = bbits[j];
    int64_t rank = (inv->ranks && fid >= 0 && fid < (int64_t)inv->features) ? inv->ranks->a[fid] : 0;
    if (rank == rank_filter)
      filtered_sb += tk_inv_w(inv->weights, fid);
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
    default: {
      double u = filtered_sa + filtered_sb - filtered_inter_w;
      rank_sim = (u == 0.0) ? 0.0 : filtered_inter_w / u;
      break;
    }
  }

  double rank_weight = inv->rank_weights->a[rank_filter];
  return (inv->total_rank_weight > 0.0) ? (rank_sim * rank_weight) / inv->total_rank_weight : 0.0;
}

static inline double tk_inv_distance_extend (
  tk_inv_t *categories,
  int64_t uid0,
  int64_t uid1,
  double observable_distance,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
) {
  if (!categories || !categories->rank_weights || categories->n_ranks == 0)
    return observable_distance;
  if (observable_distance == DBL_MAX || isnan(observable_distance))
    return tk_inv_distance(categories, uid0, uid1, cmp, tversky_alpha, tversky_beta, q_weights, e_weights, inter_weights);
  size_t n0 = 0, n1 = 0;
  int64_t *v0 = tk_inv_get(categories, uid0, &n0);
  int64_t *v1 = tk_inv_get(categories, uid1, &n1);
  if (v0 == NULL || v1 == NULL)
    return observable_distance;
  double hier_sim = tk_inv_similarity(categories, v0, n0, v1, n1, cmp, tversky_alpha, tversky_beta, q_weights, e_weights, inter_weights);
  double obs_sim = 1.0 - observable_distance;
  if (obs_sim < 0.0) obs_sim = 0.0;
  if (obs_sim > 1.0) obs_sim = 1.0;
  double obs_rank_weight = 0.0;
  if (categories->n_ranks > 0 && categories->rank_weights) {
    double last_rank_weight = categories->rank_weights->a[categories->n_ranks - 1];
    if (categories->n_ranks > 1) {
      double second_last = categories->rank_weights->a[categories->n_ranks - 2];
      if (second_last > 0.0) {
        double decay_ratio = last_rank_weight / second_last;
        obs_rank_weight = last_rank_weight * decay_ratio;
      } else {
        obs_rank_weight = last_rank_weight * 0.5;
      }
    } else {
      obs_rank_weight = last_rank_weight * 0.5;
    }
  }
  double total_weight = categories->total_rank_weight + obs_rank_weight;
  if (total_weight <= 0.0)
    return observable_distance;
  double blended_sim = (hier_sim * categories->total_rank_weight + obs_sim * obs_rank_weight) / total_weight;
  return 1.0 - blended_sim;
}

static inline tk_rvec_t *tk_inv_neighbors_by_vec (
  tk_inv_t *inv,
  int64_t *data,
  size_t datalen,
  int64_t sid0,
  uint64_t knn,
  double eps_min,
  double eps_max,
  tk_rvec_t *out,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
) {
  if (datalen == 0) {
    tk_rvec_clear(out);
    return out;
  }

  tk_rvec_clear(out);
  size_t n_sids = inv->node_offsets->n;

  tk_dvec_t *tmp_q_weights = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
  tk_dvec_t *tmp_e_weights = tk_dvec_create(NULL, inv->n_ranks, 0, 0);
  tk_dvec_t *wacc = tk_dvec_create(NULL, n_sids * inv->n_ranks, 0, 0);
  tk_ivec_t *touched = tk_ivec_create(NULL, 0, 0, 0);

  double *q_weights_by_rank = tmp_q_weights->a;
  tk_inv_compute_query_weights_by_rank(inv, data, datalen, q_weights_by_rank);

  wacc->n = n_sids * inv->n_ranks;
  for (uint64_t i = 0; i < wacc->n; i++)
    wacc->a[i] = 0.0;

  for (size_t i = 0; i < datalen; i ++) {
    int64_t fid = data[i];
    if (fid < 0 || fid >= (int64_t) inv->postings->n)
      continue;

    int64_t rank = inv->ranks ? inv->ranks->a[fid] : 0;

    if (rank_filter >= 0 && rank != rank_filter)
      continue;
    double wf = tk_inv_w(inv->weights, fid);
    tk_ivec_t *vsids = inv->postings->a[fid];
    for (uint64_t j = 0; j < vsids->n; j ++) {
      int64_t vsid = vsids->a[j];
      if (vsid == sid0)
        continue;
      if (wacc->a[(int64_t) inv->n_ranks * vsid + rank] == 0.0)
        tk_ivec_push(touched, vsid);
      wacc->a[(int64_t) inv->n_ranks * vsid + rank] += wf;
    }
  }
  touched->n = tk_ivec_uasc(touched, 0, touched->n);
  double *e_weights_by_rank = tmp_e_weights->a;

  for (uint64_t i = 0; i < touched->n; i ++) {
    int64_t vsid = touched->a[i];

    if (knn && out->n >= knn) {
      double max_sim = 0.0;
      for (uint64_t r = 0; r < inv->n_ranks; r++) {
        double inter = wacc->a[(int64_t) inv->n_ranks * vsid + (int64_t) r];
        if (inter > 0.0)
          max_sim += inter * inv->rank_weights->a[r];
      }
      max_sim = (inv->total_rank_weight > 0.0) ? max_sim / inv->total_rank_weight : 0.0;

      if (1.0 - max_sim > out->a[0].d) {
        for (uint64_t r = 0; r < inv->n_ranks; r ++)
          wacc->a[(int64_t) inv->n_ranks * vsid + (int64_t) r] = 0.0;
        continue;
      }
    }

    size_t elen = 0;
    int64_t *ev = tk_inv_sget(inv, vsid, &elen);
    tk_inv_compute_candidate_weights_by_rank(inv, ev, elen, e_weights_by_rank);
    double sim = tk_inv_similarity_by_rank(inv, wacc, vsid, q_weights_by_rank, e_weights_by_rank, cmp, tversky_alpha, tversky_beta, q_weights, e_weights, inter_weights);
    double dist = 1.0 - sim;
    double current_cutoff = (knn && out->n >= knn) ? out->a[0].d : eps_max;
    if (dist >= eps_min && dist <= current_cutoff) {
      int64_t vuid = tk_inv_sid_uid(inv, vsid);
      if (vuid >= 0) {
        if (knn)
          tk_rvec_hmax(out, knn, tk_rank(vuid, dist));
        else
          tk_rvec_push(out, tk_rank(vuid, dist));
      }
    }
    for (uint64_t r = 0; r < inv->n_ranks; r ++)
      wacc->a[(int64_t) inv->n_ranks * vsid + (int64_t) r] = 0.0;
  }

  tk_rvec_asc(out, 0, out->n);

  tk_dvec_destroy(tmp_q_weights);
  tk_dvec_destroy(tmp_e_weights);
  tk_dvec_destroy(wacc);
  tk_ivec_destroy(touched);

  return out;
}

static inline tk_rvec_t *tk_inv_neighbors_by_id (
  tk_inv_t *inv,
  int64_t uid,
  uint64_t knn,
  double eps_min,
  double eps_max,
  tk_rvec_t *out,
  tk_ivec_sim_type_t cmp,
  double tversky_alpha,
  double tversky_beta,
  int64_t rank_filter,
  tk_dvec_t *q_weights,
  tk_dvec_t *e_weights,
  tk_dvec_t *inter_weights
) {
  int64_t sid0 = tk_inv_uid_sid(inv, uid, false);
  if (sid0 < 0) {
    tk_rvec_clear(out);
    return out;
  }
  size_t len = 0;
  int64_t *data = tk_inv_get(inv, uid, &len);
  return tk_inv_neighbors_by_vec(inv, data, len, sid0, knn, eps_min, eps_max, out, cmp, tversky_alpha, tversky_beta, rank_filter, q_weights, e_weights, inter_weights);
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
  double eps_min = tk_lua_optposdouble(L, 3, "eps_min", 0.0);
  double eps_max = tk_lua_optposdouble(L, 4, "eps_max", 1.0);
  const char *typ = tk_lua_optstring(L, 5, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 6, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 7, "beta", 0.1);
  int64_t rank_filter = lua_isnil(L, 8) ? -1 : (int64_t) tk_lua_checkunsigned(L, 8, "rank_filter");

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

  tk_inv_neighborhoods(L, inv, knn, eps_min, eps_max, cmp, tversky_alpha, tversky_beta, rank_filter, NULL, NULL);
  return 2;
}

static inline int tk_inv_neighborhoods_by_ids_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_ivec_t *query_ids = tk_ivec_peek(L, 2, "ids");
  uint64_t knn = tk_lua_checkunsigned(L, 3, "knn");
  double eps_min = tk_lua_optposdouble(L, 4, "eps_min", 0.0);
  double eps_max = tk_lua_optposdouble(L, 5, "eps_max", 1.0);
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
    tk_lua_verror(L, 3, "neighborhoods_by_ids", "invalid comparator specified", typ);
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
  tk_inv_neighborhoods_by_ids(L, inv, query_ids, knn, eps_min, eps_max, cmp, tversky_alpha, tversky_beta, rank_filter, &hoods, &query_ids);
  return 2;
}

static inline int tk_inv_neighborhoods_by_vecs_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_ivec_t *query_vecs = tk_ivec_peek(L, 2, "vectors");
  uint64_t knn = tk_lua_checkunsigned(L, 3, "knn");
  double eps_min = tk_lua_optposdouble(L, 4, "eps_min", 0.0);
  double eps_max = tk_lua_optposdouble(L, 5, "eps_max", 1.0);
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

  tk_inv_neighborhoods_by_vecs(L, inv, query_vecs, knn, eps_min, eps_max, cmp, tversky_alpha, tversky_beta, rank_filter, NULL, NULL);
  return 2;
}

static inline int tk_inv_similarity_lua (lua_State *L)
{
  lua_settop(L, 9);
  tk_inv_t *inv = tk_inv_peek(L, 1);
  const char *typ = tk_lua_optstring(L, 4, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 5, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 6, "beta", 0.1);
  tk_dvec_t *q_weights = tk_dvec_peekopt(L, 7);
  tk_dvec_t *e_weights = tk_dvec_peekopt(L, 8);
  tk_dvec_t *inter_weights = tk_dvec_peekopt(L, 9);
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
  if (!q_weights) q_weights = tk_dvec_create(L, 0, 0, 0);
  if (!e_weights) e_weights = tk_dvec_create(L, 0, 0, 0);
  if (!inter_weights) inter_weights = tk_dvec_create(L, 0, 0, 0);
  lua_pushnumber(L, 1.0 - tk_inv_distance(inv, uid0, uid1, cmp, tversky_alpha, tversky_beta, q_weights, e_weights, inter_weights));
  return 1;
}

static inline int tk_inv_distance_lua (lua_State *L)
{
  lua_settop(L, 9);
  tk_inv_t *inv = tk_inv_peek(L, 1);
  const char *typ = tk_lua_optstring(L, 4, "comparator", "jaccard");
  double tversky_alpha = tk_lua_optnumber(L, 5, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 6, "beta", 0.1);
  tk_dvec_t *q_weights = tk_dvec_peekopt(L, 7);
  tk_dvec_t *e_weights = tk_dvec_peekopt(L, 8);
  tk_dvec_t *inter_weights = tk_dvec_peekopt(L, 9);
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
  if (!q_weights) q_weights = tk_dvec_create(L, 0, 0, 0);
  if (!e_weights) e_weights = tk_dvec_create(L, 0, 0, 0);
  if (!inter_weights) inter_weights = tk_dvec_create(L, 0, 0, 0);
  lua_pushnumber(L, tk_inv_distance(inv, uid0, uid1, cmp, tversky_alpha, tversky_beta, q_weights, e_weights, inter_weights));
  return 1;
}

static inline int tk_inv_neighbors_lua (lua_State *L)
{
  lua_settop(L, 13);
  tk_inv_t *inv = tk_inv_peek(L, 1);
  uint64_t knn = tk_lua_optunsigned(L, 3, "knn", 0);
  double eps_min = tk_lua_optposdouble(L, 4, "eps_min", 0.0);
  double eps_max = tk_lua_optposdouble(L, 5, "eps_max", 1.0);
  tk_rvec_t *out = tk_rvec_peek(L, 6, "out");
  const char *typ = tk_lua_optstring(L, 7, "comparator", "jaccard");
  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  double tversky_alpha = tk_lua_optnumber(L, 8, "alpha", 1.0);
  double tversky_beta = tk_lua_optnumber(L, 9, "beta", 0.1);
  int64_t rank_filter = lua_isnil(L, 10) ? -1 : (int64_t) tk_lua_checkunsigned(L, 10, "rank_filter");
  tk_dvec_t *q_weights = tk_dvec_peekopt(L, 11);
  tk_dvec_t *e_weights = tk_dvec_peekopt(L, 12);
  tk_dvec_t *inter_weights = tk_dvec_peekopt(L, 13);
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
  if (!q_weights) q_weights = tk_dvec_create(L, 0, 0, 0);
  if (!e_weights) e_weights = tk_dvec_create(L, 0, 0, 0);
  if (!inter_weights) inter_weights = tk_dvec_create(L, 0, 0, 0);
  if (lua_type(L, 2) == LUA_TNUMBER) {
    int64_t uid = tk_lua_checkinteger(L, 2, "id");
    tk_inv_neighbors_by_id(inv, uid, knn, eps_min, eps_max, out, cmp, tversky_alpha, tversky_beta, rank_filter, q_weights, e_weights, inter_weights);
  } else {
    tk_ivec_t *vec = tk_ivec_peek(L, 2, "vector");
    tk_inv_neighbors_by_vec(inv, vec->a, vec->n, -1, knn, eps_min, eps_max, out, cmp, tversky_alpha, tversky_beta, rank_filter, q_weights, e_weights, inter_weights);
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

static inline int tk_inv_rank_weights_lua (lua_State *L)
{
  tk_inv_t *inv = tk_inv_peek(L, 1);
  tk_lua_get_ephemeron(L, TK_INV_EPH, inv->rank_weights);
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
  { "rank_weights", tk_inv_rank_weights_lua },
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
  double decay,
  int i_weights,
  int i_ranks
) {
  if (!features)
    tk_lua_verror(L, 2, "create", "features must be > 0");
  tk_inv_t *inv = tk_lua_newuserdata(L, tk_inv_t, TK_INV_MT, tk_inv_lua_mt_fns, tk_inv_gc_lua);
  int Ii = tk_lua_absindex(L, -1);
  inv->destroyed = false;
  inv->next_sid = 0;
  inv->features = features;
  inv->n_ranks = n_ranks >= 1 ? n_ranks : 1;
  inv->weights = weights;
  if (weights)
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, i_weights);
  inv->ranks = ranks;
  if (ranks)
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, i_ranks);
  inv->decay = decay;
  inv->rank_weights = tk_dvec_create(L, inv->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  inv->total_rank_weight = 0.0;
  for (uint64_t r = 0; r < inv->n_ranks; r++) {
    double weight;
    if (decay < 0) {
      uint64_t flipped_r = inv->n_ranks - 1 - r;
      weight = exp((double)flipped_r * decay);
    } else {
      weight = exp(-(double)r * decay);
    }
    inv->rank_weights->a[r] = weight;
    inv->total_rank_weight += weight;
  }
  inv->rank_sizes = tk_ivec_create(L, inv->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  for (uint64_t r = 0; r < inv->n_ranks; r++) {
    inv->rank_sizes->a[r] = 0;
  }
  if (ranks) {
    for (uint64_t f = 0; f < features; f++) {
      if (f < (uint64_t)ranks->n) {
        int64_t rank = ranks->a[f];
        if (rank >= 0 && rank < (int64_t)inv->n_ranks) {
          inv->rank_sizes->a[rank]++;
        }
      }
    }
  } else {
    inv->rank_sizes->a[0] = (int64_t) features;
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
    inv->weights = NULL;
  }
  size_t rn = 0;
  tk_lua_fread(L, &rn, sizeof(size_t), 1, fh);
  if (rn) {
    tk_lua_fseek(L, -sizeof(size_t), 1, fh);
    inv->ranks = tk_ivec_load(L, fh);
    tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
    lua_pop(L, 1);
  } else {
    inv->ranks = NULL;
  }
  tk_lua_fread(L, &inv->decay, sizeof(double), 1, fh);

  inv->rank_sizes = tk_ivec_create(L, inv->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  for (uint64_t r = 0; r < inv->n_ranks; r++)
    inv->rank_sizes->a[r] = 0;
  if (inv->ranks) {
    for (uint64_t f = 0; f < inv->features; f++) {
      if (f < (uint64_t)inv->ranks->n) {
        int64_t rank = inv->ranks->a[f];
        if (rank >= 0 && rank < (int64_t)inv->n_ranks) {
          inv->rank_sizes->a[rank]++;
        }
      }
    }
  }

  inv->rank_weights = tk_dvec_create(L, inv->n_ranks, 0, 0);
  tk_lua_add_ephemeron(L, TK_INV_EPH, Ii, -1);
  lua_pop(L, 1);
  inv->total_rank_weight = 0.0;
  for (uint64_t r = 0; r < inv->n_ranks; r++) {
    double weight;
    if (inv->decay < 0) {
      uint64_t flipped_r = inv->n_ranks - 1 - r;
      weight = exp((double)flipped_r * inv->decay);
    } else {
      weight = exp(-(double)r * inv->decay);
    }
    inv->rank_weights->a[r] = weight;
    inv->total_rank_weight += weight;
  }
  return inv;
}

#endif
