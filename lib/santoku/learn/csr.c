#include <santoku/learn/csr.h>
#include <santoku/cvec/ext.h>
#include <santoku/ivec/ext.h>
#include <santoku/iumap/ext.h>
#include <omp.h>
#include <float.h>
#include <math.h>

static int tm_inv_hoods_to_csr (lua_State *L)
{
  tk_inv_hoods_t *hoods = tk_inv_hoods_peek(L, 1, "hoods");
  tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
  uint64_t n = hoods->n;

  tk_iumap_t *uid_to_pos = tk_iumap_create(0, 0);
  if (!uid_to_pos)
    return luaL_error(L, "inv_hoods_to_csr: allocation failed");
  for (uint64_t i = 0; i < ids->n; i++) {
    int absent;
    uint32_t k = tk_iumap_put(uid_to_pos, ids->a[i], &absent);
    tk_iumap_setval(uid_to_pos, k, (int64_t)i);
  }

  tk_ivec_t *off = tk_ivec_create(L, n + 1, 0, 0);
  off->a[0] = 0;
  for (uint64_t i = 0; i < n; i++)
    off->a[i + 1] = off->a[i] + (int64_t)hoods->a[i]->n;
  off->n = n + 1;

  uint64_t total = (uint64_t)off->a[n];
  tk_ivec_t *nbr = tk_ivec_create(L, total, 0, 0);
  tk_dvec_t *w = tk_dvec_create(L, total, 0, 0);
  nbr->n = total;
  w->n = total;

  for (uint64_t i = 0; i < n; i++) {
    tk_rvec_t *hood = hoods->a[i];
    int64_t pos = off->a[i];
    for (uint64_t j = 0; j < hood->n; j++) {
      uint32_t ki = tk_iumap_get(uid_to_pos, hood->a[j].i);
      nbr->a[pos + (int64_t)j] = (ki != tk_iumap_end(uid_to_pos))
        ? tk_iumap_val(uid_to_pos, ki) : -1;
      w->a[pos + (int64_t)j] = 1.0 - hood->a[j].d;
    }
  }

  tk_iumap_destroy(uid_to_pos);
  return 3;
}

static int tm_ann_hoods_to_csr (lua_State *L)
{
  tk_ann_hoods_t *hoods = tk_ann_hoods_peek(L, 1, "hoods");
  uint64_t features = tk_lua_checkunsigned(L, 2, "features");
  uint64_t n = hoods->n;

  tk_ivec_t *off = tk_ivec_create(L, n + 1, 0, 0);
  off->a[0] = 0;
  for (uint64_t i = 0; i < n; i++)
    off->a[i + 1] = off->a[i] + (int64_t)hoods->a[i]->n;
  off->n = n + 1;

  uint64_t total = (uint64_t)off->a[n];
  tk_ivec_t *nbr = tk_ivec_create(L, total, 0, 0);
  tk_dvec_t *w = tk_dvec_create(L, total, 0, 0);
  nbr->n = total;
  w->n = total;

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n; i++) {
    tk_pvec_t *hood = hoods->a[i];
    int64_t pos = off->a[i];
    for (uint64_t j = 0; j < hood->n; j++) {
      nbr->a[pos + (int64_t)j] = hood->a[j].i;
      double dist = (double)hood->a[j].p / (double)features;
      w->a[pos + (int64_t)j] = 1.0 - dist;
    }
  }

  return 3;
}

static int tm_csr_bipartite (lua_State *L)
{
  tk_ivec_t *gt_off = tk_ivec_peek(L, 1, "gt_offsets");
  tk_ivec_t *gt_nbr = tk_ivec_peek(L, 2, "gt_neighbors");
  tk_ivec_t *row_ids = tk_ivec_peek(L, 3, "row_ids");
  tk_ivec_t *col_ids = tk_ivec_peek(L, 4, "col_ids");

  uint64_t n_rows = row_ids->n;
  uint64_t n_cols = col_ids->n;
  uint64_t n_total = n_rows + n_cols;

  tk_ivec_t *ids = tk_ivec_create(L, n_total, 0, 0);
  ids->n = n_total;
  for (uint64_t i = 0; i < n_rows; i++)
    ids->a[i] = row_ids->a[i];
  for (uint64_t i = 0; i < n_cols; i++)
    ids->a[n_rows + i] = col_ids->a[i];

  tk_ivec_t *off = tk_ivec_create(L, n_total + 1, 0, 0);
  tk_ivec_t *nbr = tk_ivec_create(L, 0, 0, 0);
  tk_dvec_t *w = tk_dvec_create(L, 0, 0, 0);

  off->a[0] = 0;
  for (uint64_t d = 0; d < n_rows; d++) {
    int64_t gt_start = gt_off->a[d];
    int64_t gt_end = gt_off->a[d + 1];
    for (int64_t j = gt_start; j < gt_end; j++) {
      int64_t li = gt_nbr->a[j];
      tk_ivec_push(nbr, (int64_t)(n_rows + (uint64_t)li));
      tk_dvec_push(w, 1.0);
    }
    off->a[d + 1] = (int64_t)nbr->n;
  }

  for (uint64_t i = 0; i < n_cols; i++)
    off->a[n_rows + i + 1] = (int64_t)nbr->n;

  off->n = n_total + 1;

  tk_lua_replace(L, 1, 4);
  lua_settop(L, 4);
  return 4;
}

static int tm_csr_bipartite_neg (lua_State *L)
{
  tk_ivec_t *gt_off = tk_ivec_peek(L, 1, "gt_offsets");
  tk_ivec_t *gt_nbr = tk_ivec_peek(L, 2, "gt_neighbors");
  tk_ivec_t *row_ids = tk_ivec_peek(L, 3, "row_ids");
  tk_ivec_t *col_ids = tk_ivec_peek(L, 4, "col_ids");
  uint64_t n_neg = tk_lua_checkunsigned(L, 5, "n_neg");
  bool has_index = !lua_isnoneornil(L, 6) && !lua_isnoneornil(L, 7);
  tk_inv_t *source = has_index ? tk_inv_peek(L, 6) : NULL;
  tk_inv_t *search = has_index ? tk_inv_peek(L, 7) : NULL;
  double decay = luaL_optnumber(L, 8, 0.0);

  uint64_t n_rows = row_ids->n;
  uint64_t n_cols = col_ids->n;
  uint64_t n_total = n_rows + n_cols;

  tk_ivec_t *ids = tk_ivec_create(L, n_total, 0, 0);
  ids->n = n_total;
  for (uint64_t i = 0; i < n_rows; i++)
    ids->a[i] = row_ids->a[i];
  for (uint64_t i = 0; i < n_cols; i++)
    ids->a[n_rows + i] = col_ids->a[i];

  tk_ivec_t *off = tk_ivec_create(L, n_total + 1, 0, 0);
  tk_ivec_t *nbr = tk_ivec_create(L, 0, 0, 0);
  tk_dvec_t *w = tk_dvec_create(L, 0, 0, 0);

  off->a[0] = 0;
  for (uint64_t d = 0; d < n_rows; d++) {
    int64_t gt_start = gt_off->a[d];
    int64_t gt_end = gt_off->a[d + 1];
    for (int64_t j = gt_start; j < gt_end; j++) {
      tk_ivec_push(nbr, (int64_t)(n_rows + (uint64_t)gt_nbr->a[j]));
      tk_dvec_push(w, 1.0);
    }
    off->a[d + 1] = (int64_t)nbr->n;
  }
  for (uint64_t i = 0; i < n_cols; i++)
    off->a[n_rows + i + 1] = (int64_t)nbr->n;
  off->n = n_total + 1;

  tk_iumap_t *col_uid_to_idx = NULL;
  if (has_index) {
    col_uid_to_idx = tk_iumap_create(0, 0);
    if (!col_uid_to_idx)
      return luaL_error(L, "bipartite_neg: allocation failed");
    for (uint64_t i = 0; i < n_cols; i++) {
      int absent;
      uint32_t k = tk_iumap_put(col_uid_to_idx, col_ids->a[i], &absent);
      tk_iumap_setval(col_uid_to_idx, k, (int64_t)i);
    }
  }

  uint64_t max_new = n_rows * n_neg;
  int64_t *tmp_nbr = (int64_t *)malloc(max_new * sizeof(int64_t));
  int64_t *tmp_off = (int64_t *)malloc((n_total + 1) * sizeof(int64_t));
  int64_t *tmp_cnt = (int64_t *)calloc(n_rows, sizeof(int64_t));
  if (!tmp_nbr || !tmp_off || !tmp_cnt) {
    free(tmp_nbr);
    free(tmp_off);
    free(tmp_cnt);
    if (col_uid_to_idx) tk_iumap_destroy(col_uid_to_idx);
    return luaL_error(L, "bipartite_neg: allocation failed");
  }

  #pragma omp parallel
  {
    tk_rvec_t *my_hood = has_index ? tk_rvec_create(NULL, 0, 0, 0) : NULL;

    #pragma omp for schedule(dynamic, 64)
    for (uint64_t d = 0; d < n_rows; d++) {
      int64_t gt_start = gt_off->a[d];
      int64_t gt_end = gt_off->a[d + 1];
      uint64_t n_gt = (uint64_t)(gt_end - gt_start);
      uint64_t added = 0;
      int64_t *row_out = tmp_nbr + d * n_neg;

      if (has_index && my_hood) {
        int64_t uid = row_ids->a[d];
        size_t feat_n = 0;
        int64_t *feat_data = tk_inv_get(source, uid, &feat_n);
        if (feat_data && feat_n > 0) {
          int64_t *query_bits = (int64_t *)malloc(feat_n * sizeof(int64_t));
          if (query_bits) {
            memcpy(query_bits, feat_data, feat_n * sizeof(int64_t));
            int64_t q_ro[TK_INV_MAX_RANKS + 1];
            tk_inv_partition_by_rank(source->ranks->a, source->n_ranks,
              query_bits, feat_n, q_ro);
            uint64_t request_k = n_neg + n_gt;
            if (request_k < n_neg * 2) request_k = n_neg * 2;
            tk_inv_neighbors_by_vec(search, query_bits, feat_n, q_ro,
              -1, request_k, my_hood, decay);
            free(query_bits);
            for (uint64_t h = 0; h < my_hood->n && added < n_neg; h++) {
              int64_t nuid = my_hood->a[h].i;
              uint32_t ki = tk_iumap_get(col_uid_to_idx, nuid);
              if (ki == tk_iumap_end(col_uid_to_idx))
                continue;
              int64_t col_idx = tk_iumap_val(col_uid_to_idx, ki);
              bool is_gt = false;
              for (int64_t j = gt_start; j < gt_end; j++) {
                if (gt_nbr->a[j] == col_idx) { is_gt = true; break; }
              }
              if (!is_gt) {
                row_out[added++] = (int64_t)(n_rows + (uint64_t)col_idx);
              }
            }
          }
        }
      }

      if (added < n_neg) {
        uint64_t attempts = 0;
        uint64_t max_attempts = (n_neg - added) * 10;
        while (added < n_neg && attempts < max_attempts) {
          attempts++;
          int64_t li = (int64_t)(tk_fast_random() % n_cols);
          bool is_gt = false;
          for (int64_t j = gt_start; j < gt_end; j++) {
            if (gt_nbr->a[j] == li) { is_gt = true; break; }
          }
          if (!is_gt) {
            row_out[added++] = (int64_t)(n_rows + (uint64_t)li);
          }
        }
      }

      tmp_cnt[d] = (int64_t)added;
    }

    if (my_hood) tk_rvec_destroy(my_hood);
  }

  tmp_off[0] = 0;
  for (uint64_t d = 0; d < n_rows; d++)
    tmp_off[d + 1] = tmp_off[d] + tmp_cnt[d];
  for (uint64_t i = n_rows; i < n_total; i++)
    tmp_off[i + 1] = tmp_off[n_rows];
  uint64_t tmp_pos = (uint64_t)tmp_off[n_rows];

  for (uint64_t d = 0; d < n_rows; d++) {
    if (tmp_cnt[d] > 0 && tmp_off[d] != (int64_t)(d * n_neg))
      memmove(tmp_nbr + tmp_off[d], tmp_nbr + d * n_neg,
        (size_t)tmp_cnt[d] * sizeof(int64_t));
  }
  free(tmp_cnt);

  uint64_t old_total = nbr->n;
  uint64_t new_total = old_total + tmp_pos;
  tk_ivec_ensure(nbr, new_total);
  tk_dvec_ensure(w, new_total);
  nbr->n = new_total;
  w->n = new_total;

  int64_t wp = (int64_t)new_total;
  for (int64_t i = (int64_t)n_total - 1; i >= 0; i--) {
    int64_t ts = tmp_off[i];
    int64_t tc = tmp_off[i + 1] - ts;
    wp -= tc;
    if (tc > 0) {
      memcpy(nbr->a + wp, tmp_nbr + ts, (size_t)tc * sizeof(int64_t));
      for (int64_t j = 0; j < tc; j++)
        w->a[wp + j] = 0.0;
    }
    int64_t os = off->a[i];
    int64_t oc = off->a[i + 1] - os;
    wp -= oc;
    if (oc > 0 && wp != os) {
      memmove(nbr->a + wp, nbr->a + os, (size_t)oc * sizeof(int64_t));
      memmove(w->a + wp, w->a + os, (size_t)oc * sizeof(double));
    }
    off->a[i + 1] = wp + oc + tc;
  }

  free(tmp_nbr);
  free(tmp_off);
  if (col_uid_to_idx) tk_iumap_destroy(col_uid_to_idx);

  tk_lua_replace(L, 1, 4);
  lua_settop(L, 4);
  return 4;
}

static int tm_csr_random_pairs (lua_State *L)
{
  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  uint64_t n_per_node = tk_lua_checkunsigned(L, 2, "n_per_node");
  uint64_t n = ids->n;

  uint64_t total = n * n_per_node;
  tk_ivec_t *off = tk_ivec_create(L, n + 1, 0, 0);
  tk_ivec_t *nbr = tk_ivec_create(L, total, 0, 0);
  tk_dvec_t *w = tk_dvec_create(L, total, 0, 0);

  off->n = n + 1;
  nbr->n = total;
  w->n = total;

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n; i++) {
    off->a[i] = (int64_t)(i * n_per_node);
    int64_t base = (int64_t)(i * n_per_node);
    for (uint64_t e = 0; e < n_per_node; e++) {
      uint64_t idx2 = tk_fast_random() % n;
      if (idx2 == i)
        idx2 = (idx2 + 1) % n;
      nbr->a[base + (int64_t)e] = (int64_t)idx2;
      w->a[base + (int64_t)e] = 0.0;
    }
  }
  off->a[n] = (int64_t)total;

  return 3;
}

static int tm_csr_weight_from_index (lua_State *L)
{
  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  tk_ivec_t *off = tk_ivec_peek(L, 2, "offsets");
  tk_ivec_t *nbr = tk_ivec_peek(L, 3, "neighbors");
  tk_dvec_t *w = tk_dvec_peek(L, 4, "weights");

  tk_inv_t *idx_inv = tk_inv_peekopt(L, 5);
  tk_ann_t *idx_ann = tk_ann_peekopt(L, 5);
  if (!idx_inv && !idx_ann)
    return luaL_error(L, "weight_from_index: arg 5 must be inv or ann index");

  double decay = luaL_checknumber(L, 6);
  double eps = 1e-8;
  uint64_t n = ids->n;

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n; i++) {
    int64_t u = ids->a[i];
    int64_t start = off->a[i];
    int64_t end = off->a[i + 1];
    for (int64_t j = start; j < end; j++) {
      int64_t v = ids->a[nbr->a[j]];
      double dist = 1.0;
      if (idx_inv) {
        int64_t usid = tk_inv_uid_sid(idx_inv, u, TK_INV_FIND);
        int64_t vsid = tk_inv_uid_sid(idx_inv, v, TK_INV_FIND);
        if (usid >= 0 && vsid >= 0) {
          uint64_t stride = idx_inv->n_ranks + 1;
          const int64_t *u_ro = idx_inv->node_rank_offsets->a + usid * (int64_t) stride;
          const int64_t *v_ro = idx_inv->node_rank_offsets->a + vsid * (int64_t) stride;
          dist = 1.0 - tk_inv_similarity(idx_inv, idx_inv->node_bits->a, u_ro,
            idx_inv->node_bits->a, v_ro, decay);
        } else {
          dist = 1.0;
        }
      } else {
        char *uset = tk_ann_get(idx_ann, u);
        char *wset = tk_ann_get(idx_ann, v);
        if (uset && wset)
          dist = (double)tk_cvec_bits_hamming_serial((const uint8_t *)uset, (const uint8_t *)wset,
            idx_ann->features) / (double)idx_ann->features;
        else
          dist = 1.0;
      }
      double sim = 1.0 - dist;
      if (sim < eps) sim = eps;
      if (sim > 1.0) sim = 1.0;
      w->a[j] = sim;
    }
  }

  return 0;
}

static int tm_csr_merge (lua_State *L)
{
  tk_ivec_t *off1 = tk_ivec_peek(L, 1, "offsets1");
  tk_ivec_t *nbr1 = tk_ivec_peek(L, 2, "neighbors1");
  tk_dvec_t *w1 = tk_dvec_peek(L, 3, "weights1");
  tk_ivec_t *off2 = tk_ivec_peek(L, 4, "offsets2");
  tk_ivec_t *nbr2 = tk_ivec_peek(L, 5, "neighbors2");
  tk_dvec_t *w2 = tk_dvec_peek(L, 6, "weights2");

  uint64_t n = off1->n - 1;
  uint64_t add = nbr2->n;
  uint64_t total = nbr1->n + add;

  tk_ivec_ensure(nbr1, total);
  tk_dvec_ensure(w1, total);
  nbr1->n = total;
  w1->n = total;

  int64_t wp = (int64_t)total;
  for (int64_t i = (int64_t)n - 1; i >= 0; i--) {
    int64_t s2 = off2->a[i];
    int64_t c2 = off2->a[i + 1] - s2;
    wp -= c2;
    if (c2 > 0) {
      memcpy(nbr1->a + wp, nbr2->a + s2, (size_t)c2 * sizeof(int64_t));
      memcpy(w1->a + wp, w2->a + s2, (size_t)c2 * sizeof(double));
    }
    int64_t s1 = off1->a[i];
    int64_t c1 = off1->a[i + 1] - s1;
    wp -= c1;
    if (c1 > 0 && wp != s1) {
      memmove(nbr1->a + wp, nbr1->a + s1, (size_t)c1 * sizeof(int64_t));
      memmove(w1->a + wp, w1->a + s1, (size_t)c1 * sizeof(double));
    }
    off1->a[i + 1] = wp + c1 + c2;
  }

  return 0;
}

static int tm_csr_symmetrize (lua_State *L)
{
  tk_ivec_t *off_in = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *nbr_in = tk_ivec_peek(L, 2, "neighbors");
  tk_dvec_t *w_in = tk_dvec_peek(L, 3, "weights");
  uint64_t n = tk_lua_checkunsigned(L, 4, "n_nodes");

  tk_rvec_t **hoods = calloc(n, sizeof(tk_rvec_t *));
  if (!hoods)
    return luaL_error(L, "symmetrize: allocation failed");

  for (uint64_t i = 0; i < n; i++) {
    hoods[i] = tk_rvec_create(NULL, 0, 0, 0);
    if (!hoods[i]) goto fail;
  }

  for (uint64_t i = 0; i < n; i++) {
    int64_t start = off_in->a[i];
    int64_t end = off_in->a[i + 1];
    for (int64_t j = start; j < end; j++) {
      int64_t v = nbr_in->a[j];
      double wt = w_in->a[j];
      if (v < 0 || v >= (int64_t)n) continue;
      tk_rvec_push(hoods[i], tk_rank(v, wt));
      tk_rvec_push(hoods[(uint64_t)v], tk_rank((int64_t)i, wt));
    }
  }

  for (uint64_t i = 0; i < n; i++) {
    tk_rvec_t *h = hoods[i];
    if (h->n <= 1) continue;
    uint64_t ne = tk_rvec_xasc(h, 0, h->n);
    uint64_t write = 0;
    for (uint64_t j = 0; j < ne; j++) {
      if (write > 0 && h->a[write - 1].i == h->a[j].i) {
        h->a[write - 1].d = h->a[j].d;
      } else {
        h->a[write++] = h->a[j];
      }
    }
    h->n = write;
    tk_rvec_desc(h, 0, h->n);
  }

  uint64_t total = 0;
  for (uint64_t i = 0; i < n; i++)
    total += hoods[i]->n;

  tk_ivec_ensure(nbr_in, total);
  tk_dvec_ensure(w_in, total);
  nbr_in->n = total;
  w_in->n = total;

  off_in->a[0] = 0;
  for (uint64_t i = 0; i < n; i++)
    off_in->a[i + 1] = off_in->a[i] + (int64_t)hoods[i]->n;

  for (uint64_t i = 0; i < n; i++) {
    int64_t pos = off_in->a[i];
    tk_rvec_t *h = hoods[i];
    for (uint64_t j = 0; j < h->n; j++) {
      nbr_in->a[pos + (int64_t)j] = h->a[j].i;
      w_in->a[pos + (int64_t)j] = h->a[j].d;
    }
  }

  for (uint64_t i = 0; i < n; i++)
    tk_rvec_destroy(hoods[i]);
  free(hoods);

  return 0;

fail:
  for (uint64_t i = 0; i < n; i++)
    if (hoods[i]) tk_rvec_destroy(hoods[i]);
  free(hoods);
  return luaL_error(L, "symmetrize: allocation failed");
}

static int tm_csr_bits_to_hv_encoder_fn (lua_State *L)
{
  tk_ivec_t *features = tk_ivec_peek(L, 1, "features");
  tk_ivec_t *offsets = tk_ivec_peek(L, 2, "offsets");
  tk_cvec_t *tokens = tk_cvec_peek(L, lua_upvalueindex(1), "tokens");
  uint64_t n_tokens = (uint64_t)lua_tointeger(L, lua_upvalueindex(2));
  uint64_t hv_size = (uint64_t)lua_tointeger(L, lua_upvalueindex(3));
  uint64_t shift_stride = (uint64_t)lua_tointeger(L, lua_upvalueindex(4));
  tk_ivec_bits_to_hv(L, offsets, features, tokens, n_tokens, hv_size, shift_stride);
  return 1;
}

static int tm_csr_bits_to_hv (lua_State *L)
{
  int nargs = lua_gettop(L);
  tk_ivec_t *features = tk_ivec_peek(L, 1, "features");
  tk_ivec_t *offsets = tk_ivec_peek(L, 2, "offsets");
  uint64_t hv_size = tk_lua_checkunsigned(L, 3, "hv_size");
  uint64_t n_bits = tk_lua_checkunsigned(L, 4, "n_bits");
  uint64_t shift_stride = (nargs >= 5 && !lua_isnil(L, 5)) ? tk_lua_checkunsigned(L, 5, "shift_stride") : 0;
  if (hv_size == 0) return luaL_error(L, "hv_size must be > 0");
  if (n_bits == 0) return luaL_error(L, "n_bits must be > 0");
  if (n_bits > hv_size) return luaL_error(L, "n_bits must be <= hv_size");
  uint64_t n_tokens = 0;
  tk_hv_generate_tokens(L, features, hv_size, n_bits, &n_tokens);
  int tok_idx = lua_gettop(L);
  tk_cvec_t *tokens = tk_cvec_peek(L, tok_idx, "tokens");
  lua_pushvalue(L, tok_idx);
  lua_pushinteger(L, (lua_Integer)n_tokens);
  lua_pushinteger(L, (lua_Integer)hv_size);
  lua_pushinteger(L, (lua_Integer)shift_stride);
  lua_pushcclosure(L, tm_csr_bits_to_hv_encoder_fn, 4);
  int enc_idx = lua_gettop(L);
  tk_ivec_bits_to_hv(L, offsets, features, tokens, n_tokens, hv_size, shift_stride);
  lua_pushvalue(L, enc_idx);
  return 2;
}


static int tm_csr_to_csc (lua_State *L)
{
  tk_ivec_t *tokens = tk_ivec_peek(L, 1, "tokens");
  uint64_t n_samples = tk_lua_checkunsigned(L, 2, "n_samples");
  uint64_t n_tokens = tk_lua_checkunsigned(L, 3, "n_tokens");

  uint64_t *counts = (uint64_t *)calloc(n_tokens, sizeof(uint64_t));
  if (!counts)
    return luaL_error(L, "to_csc: allocation failed");

  for (uint64_t i = 0; i < tokens->n; i++) {
    int64_t v = tokens->a[i];
    if (v < 0) continue;
    uint64_t tok = (uint64_t)v % n_tokens;
    uint64_t s = (uint64_t)v / n_tokens;
    if (s < n_samples)
      counts[tok]++;
  }

  tk_ivec_t *off = tk_ivec_create(L, n_tokens + 1, 0, 0);
  off->n = n_tokens + 1;
  off->a[0] = 0;
  for (uint64_t t = 0; t < n_tokens; t++)
    off->a[t + 1] = off->a[t] + (int64_t)counts[t];

  uint64_t total = (uint64_t)off->a[n_tokens];
  tk_ivec_t *idx = tk_ivec_create(L, total, 0, 0);
  idx->n = total;

  memset(counts, 0, n_tokens * sizeof(uint64_t));
  for (uint64_t i = 0; i < tokens->n; i++) {
    int64_t v = tokens->a[i];
    if (v < 0) continue;
    uint64_t tok = (uint64_t)v % n_tokens;
    uint64_t s = (uint64_t)v / n_tokens;
    if (s >= n_samples) continue;
    idx->a[(uint64_t)off->a[tok] + counts[tok]] = (int64_t)s;
    counts[tok]++;
  }

  free(counts);
  return 2;
}

static int tm_csr_bits_select (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *feats = tk_ivec_peek(L, 2, "feats");
  tk_ivec_t *remap_ids = tk_ivec_peek(L, 3, "remap_ids");
  tk_iumap_t *inverse = tk_iumap_from_ivec(L, remap_ids);
  if (!inverse) return luaL_error(L, "bits_select: allocation failed");
  uint64_t n_classes = offsets->n - 1;
  tk_ivec_t *new_off = tk_ivec_create(L, n_classes + 1, 0, 0);
  new_off->n = n_classes + 1;
  tk_ivec_t *new_feats = tk_ivec_create(L, feats->n, 0, 0);
  new_off->a[0] = 0;
  uint64_t pos = 0;
  for (uint64_t c = 0; c < n_classes; c++) {
    int64_t start = offsets->a[c];
    int64_t end = offsets->a[c + 1];
    for (int64_t j = start; j < end; j++) {
      int64_t new_id = tk_iumap_get_or(inverse, feats->a[j], -1);
      if (new_id < 0) continue;
      new_feats->a[pos++] = new_id;
    }
    new_off->a[c + 1] = (int64_t)pos;
  }
  new_feats->n = pos;
  tk_iumap_destroy(inverse);
  return 2;
}

static int tm_csr_propagate (lua_State *L)
{
  tk_ivec_t *bits = tk_ivec_peek(L, 1, "bits");
  uint64_t n_docs = tk_lua_checkunsigned(L, 2, "n_docs");
  uint64_t n_feats = tk_lua_checkunsigned(L, 3, "n_feats");

  uint64_t *doc_off = (uint64_t *)calloc(n_docs + 1, sizeof(uint64_t));
  uint64_t *feat_off = (uint64_t *)calloc(n_feats + 1, sizeof(uint64_t));
  if (!doc_off || !feat_off) {
    free(doc_off); free(feat_off);
    return luaL_error(L, "propagate: allocation failed");
  }

  for (uint64_t i = 0; i < bits->n; i++) {
    uint64_t v = (uint64_t)bits->a[i];
    uint64_t d = v / n_feats, f = v % n_feats;
    if (d < n_docs) { doc_off[d + 1]++; feat_off[f + 1]++; }
  }
  for (uint64_t i = 0; i < n_docs; i++) doc_off[i + 1] += doc_off[i];
  for (uint64_t i = 0; i < n_feats; i++) feat_off[i + 1] += feat_off[i];

  uint64_t *doc_feats = (uint64_t *)malloc(bits->n * sizeof(uint64_t));
  uint64_t *feat_docs = (uint64_t *)malloc(bits->n * sizeof(uint64_t));
  uint64_t *dc = (uint64_t *)calloc(n_docs, sizeof(uint64_t));
  uint64_t *fc = (uint64_t *)calloc(n_feats, sizeof(uint64_t));
  if (!doc_feats || !feat_docs || !dc || !fc) {
    free(doc_off); free(feat_off); free(doc_feats); free(feat_docs); free(dc); free(fc);
    return luaL_error(L, "propagate: allocation failed");
  }

  for (uint64_t i = 0; i < bits->n; i++) {
    uint64_t v = (uint64_t)bits->a[i];
    uint64_t d = v / n_feats, f = v % n_feats;
    if (d >= n_docs) continue;
    doc_feats[doc_off[d] + dc[d]++] = f;
    feat_docs[feat_off[f] + fc[f]++] = d;
  }
  free(dc); free(fc);

  uint64_t bm_bytes = (n_feats + 7) / 8;
  uint64_t total = 0;

  #pragma omp parallel
  {
    uint8_t *bm = (uint8_t *)calloc(1, bm_bytes);
    uint64_t local_total = 0;

    #pragma omp for schedule(dynamic, 64)
    for (uint64_t d = 0; d < n_docs; d++) {
      memset(bm, 0, bm_bytes);
      for (uint64_t i = doc_off[d]; i < doc_off[d + 1]; i++) {
        uint64_t f = doc_feats[i];
        for (uint64_t j = feat_off[f]; j < feat_off[f + 1]; j++) {
          uint64_t d2 = feat_docs[j];
          for (uint64_t k = doc_off[d2]; k < doc_off[d2 + 1]; k++) {
            uint64_t f2 = doc_feats[k];
            bm[f2 / 8] |= (uint8_t)(1 << (f2 % 8));
          }
        }
      }
      uint64_t cnt = 0;
      for (uint64_t b = 0; b < bm_bytes; b++)
        cnt += (uint64_t)__builtin_popcount((unsigned int)bm[b]);
      local_total += cnt;
    }

    #pragma omp atomic
    total += local_total;

    free(bm);
  }

  tk_ivec_t *out = tk_ivec_create(L, total, 0, 0);

  uint64_t *offsets = (uint64_t *)malloc((n_docs + 1) * sizeof(uint64_t));
  if (!offsets) {
    free(doc_off); free(feat_off); free(doc_feats); free(feat_docs);
    return luaL_error(L, "propagate: allocation failed");
  }

  #pragma omp parallel
  {
    uint8_t *bm = (uint8_t *)calloc(1, bm_bytes);

    #pragma omp for schedule(dynamic, 64)
    for (uint64_t d = 0; d < n_docs; d++) {
      memset(bm, 0, bm_bytes);
      for (uint64_t i = doc_off[d]; i < doc_off[d + 1]; i++) {
        uint64_t f = doc_feats[i];
        for (uint64_t j = feat_off[f]; j < feat_off[f + 1]; j++) {
          uint64_t d2 = feat_docs[j];
          for (uint64_t k = doc_off[d2]; k < doc_off[d2 + 1]; k++) {
            uint64_t f2 = doc_feats[k];
            bm[f2 / 8] |= (uint8_t)(1 << (f2 % 8));
          }
        }
      }
      uint64_t cnt = 0;
      for (uint64_t b = 0; b < bm_bytes; b++)
        cnt += (uint64_t)__builtin_popcount((unsigned int)bm[b]);
      offsets[d] = cnt;
    }

    free(bm);
  }

  uint64_t prefix = 0;
  for (uint64_t d = 0; d < n_docs; d++) {
    uint64_t cnt = offsets[d];
    offsets[d] = prefix;
    prefix += cnt;
  }
  offsets[n_docs] = prefix;

  #pragma omp parallel
  {
    uint8_t *bm = (uint8_t *)calloc(1, bm_bytes);

    #pragma omp for schedule(dynamic, 64)
    for (uint64_t d = 0; d < n_docs; d++) {
      memset(bm, 0, bm_bytes);
      for (uint64_t i = doc_off[d]; i < doc_off[d + 1]; i++) {
        uint64_t f = doc_feats[i];
        for (uint64_t j = feat_off[f]; j < feat_off[f + 1]; j++) {
          uint64_t d2 = feat_docs[j];
          for (uint64_t k = doc_off[d2]; k < doc_off[d2 + 1]; k++) {
            uint64_t f2 = doc_feats[k];
            bm[f2 / 8] |= (uint8_t)(1 << (f2 % 8));
          }
        }
      }
      uint64_t wp = offsets[d];
      for (uint64_t f = 0; f < n_feats; f++) {
        if (bm[f / 8] & (1 << (f % 8)))
          out->a[wp++] = (int64_t)(d * n_feats + f);
      }
    }

    free(bm);
  }

  out->n = total;
  free(offsets);
  free(doc_off); free(feat_off); free(doc_feats); free(feat_docs);
  return 1;
}

static int tm_csr_stratified_sample (lua_State *L)
{
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *neighbors = tk_ivec_peek(L, 2, "neighbors");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_labels = tk_lua_checkunsigned(L, 4, "n_labels");
  uint64_t n_select = tk_lua_checkunsigned(L, 5, "n_select");

  if (n_select > n_samples) n_select = n_samples;
  if (n_select == n_samples) {
    tk_ivec_t *r = tk_ivec_create(L, n_samples, 0, 0);
    r->n = n_samples;
    for (uint64_t i = 0; i < n_samples; i++) r->a[i] = (int64_t)i;
    return 1;
  }

  uint64_t *lcnt = (uint64_t *)calloc(n_labels, sizeof(uint64_t));
  if (!lcnt) return luaL_error(L, "stratified_sample: alloc failed");
  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t lo = offsets->a[i], hi = offsets->a[i + 1];
    for (int64_t j = lo; j < hi; j++) {
      uint64_t lab = (uint64_t)neighbors->a[j];
      if (lab < n_labels) lcnt[lab]++;
    }
  }

  uint64_t *loff = (uint64_t *)malloc((n_labels + 1) * sizeof(uint64_t));
  if (!loff) { free(lcnt); return luaL_error(L, "stratified_sample: alloc failed"); }
  loff[0] = 0;
  for (uint64_t l = 0; l < n_labels; l++)
    loff[l + 1] = loff[l] + lcnt[l];
  uint64_t total_entries = loff[n_labels];

  uint64_t *lsamp = (uint64_t *)malloc(total_entries * sizeof(uint64_t));
  if (!lsamp) { free(lcnt); free(loff); return luaL_error(L, "stratified_sample: alloc failed"); }
  memset(lcnt, 0, n_labels * sizeof(uint64_t));
  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t lo = offsets->a[i], hi = offsets->a[i + 1];
    for (int64_t j = lo; j < hi; j++) {
      uint64_t lab = (uint64_t)neighbors->a[j];
      if (lab < n_labels) {
        lsamp[loff[lab] + lcnt[lab]] = i;
        lcnt[lab]++;
      }
    }
  }

  for (uint64_t l = 0; l < n_labels; l++) {
    uint64_t lo = loff[l], cnt = loff[l + 1] - lo;
    for (uint64_t i = cnt; i > 1; i--) {
      uint64_t j = tk_fast_random() % i;
      uint64_t tmp = lsamp[lo + i - 1];
      lsamp[lo + i - 1] = lsamp[lo + j];
      lsamp[lo + j] = tmp;
    }
  }

  uint64_t *lorder = (uint64_t *)malloc(n_labels * sizeof(uint64_t));
  uint64_t *lsz = (uint64_t *)malloc(n_labels * sizeof(uint64_t));
  if (!lorder || !lsz) {
    free(lcnt); free(loff); free(lsamp); free(lorder); free(lsz);
    return luaL_error(L, "stratified_sample: alloc failed");
  }
  for (uint64_t l = 0; l < n_labels; l++) {
    lorder[l] = l;
    lsz[l] = loff[l + 1] - loff[l];
  }
  for (uint64_t i = 1; i < n_labels; i++) {
    uint64_t key = lorder[i], ksz = lsz[lorder[i]];
    int64_t j = (int64_t)i - 1;
    while (j >= 0 && lsz[lorder[(uint64_t)j]] > ksz) {
      lorder[(uint64_t)j + 1] = lorder[(uint64_t)j];
      j--;
    }
    lorder[(uint64_t)j + 1] = key;
  }

  uint64_t bm_bytes = (n_samples + 7) / 8;
  uint8_t *sel = (uint8_t *)calloc(bm_bytes, 1);
  uint64_t *curs = (uint64_t *)calloc(n_labels, sizeof(uint64_t));
  if (!sel || !curs) {
    free(lcnt); free(loff); free(lsamp); free(lorder); free(lsz); free(sel); free(curs);
    return luaL_error(L, "stratified_sample: alloc failed");
  }

  tk_ivec_t *result = tk_ivec_create(L, n_select, 0, 0);
  uint64_t n_sel = 0;
  bool progress = true;
  while (n_sel < n_select && progress) {
    progress = false;
    for (uint64_t li = 0; li < n_labels && n_sel < n_select; li++) {
      uint64_t l = lorder[li];
      uint64_t lo = loff[l], cnt = loff[l + 1] - lo;
      while (curs[l] < cnt) {
        uint64_t sid = lsamp[lo + curs[l]];
        curs[l]++;
        if (!(sel[sid / 8] & (1 << (sid % 8)))) {
          sel[sid / 8] |= (uint8_t)(1 << (sid % 8));
          result->a[n_sel++] = (int64_t)sid;
          progress = true;
          break;
        }
      }
    }
  }

  if (n_sel < n_select) {
    for (uint64_t i = 0; i < n_samples && n_sel < n_select; i++) {
      if (!(sel[i / 8] & (1 << (i % 8)))) {
        result->a[n_sel++] = (int64_t)i;
      }
    }
  }

  result->n = n_sel;
  free(lcnt); free(loff); free(lsamp); free(lorder); free(lsz); free(sel); free(curs);
  return 1;
}

static int tm_csr_label_union (lua_State *L)
{
  tk_ivec_t *nn_off = tk_ivec_peek(L, 1, "nn_offsets");
  tk_ivec_t *nn_nbr = tk_ivec_peek(L, 2, "nn_neighbors");
  tk_ivec_t *hood_ids = tk_ivec_peek(L, 3, "hood_ids");
  tk_ivec_t *lab_off = tk_ivec_peek(L, 4, "label_offsets");
  tk_ivec_t *lab_nbr = tk_ivec_peek(L, 5, "label_neighbors");
  uint64_t n_labels = tk_lua_checkunsigned(L, 6, "n_labels");

  uint64_t n_queries = nn_off->n - 1;
  uint64_t bm_bytes = (n_labels + 7) / 8;

  uint64_t *counts = (uint64_t *)calloc(n_queries, sizeof(uint64_t));
  if (!counts)
    return luaL_error(L, "label_union: allocation failed");

  #pragma omp parallel
  {
    uint8_t *bm = (uint8_t *)calloc(1, bm_bytes);
    #pragma omp for schedule(dynamic, 64)
    for (uint64_t i = 0; i < n_queries; i++) {
      memset(bm, 0, bm_bytes);
      int64_t ns = nn_off->a[i], ne = nn_off->a[i + 1];
      for (int64_t j = ns; j < ne; j++) {
        int64_t uid = hood_ids->a[nn_nbr->a[j]];
        int64_t ls = lab_off->a[uid], le = lab_off->a[uid + 1];
        for (int64_t k = ls; k < le; k++) {
          uint64_t lab = (uint64_t)lab_nbr->a[k];
          bm[lab / 8] |= (uint8_t)(1 << (lab % 8));
        }
      }
      uint64_t cnt = 0;
      for (uint64_t b = 0; b < bm_bytes; b++)
        cnt += (uint64_t)__builtin_popcount((unsigned int)bm[b]);
      counts[i] = cnt;
    }
    free(bm);
  }

  uint64_t total = 0;
  for (uint64_t i = 0; i < n_queries; i++)
    total += counts[i];

  tk_ivec_t *out = tk_ivec_create(L, total, 0, 0);
  out->n = total;

  uint64_t *prefix = (uint64_t *)malloc((n_queries + 1) * sizeof(uint64_t));
  if (!prefix) {
    free(counts);
    return luaL_error(L, "label_union: allocation failed");
  }
  prefix[0] = 0;
  for (uint64_t i = 0; i < n_queries; i++)
    prefix[i + 1] = prefix[i] + counts[i];
  free(counts);

  #pragma omp parallel
  {
    uint8_t *bm = (uint8_t *)calloc(1, bm_bytes);
    #pragma omp for schedule(dynamic, 64)
    for (uint64_t i = 0; i < n_queries; i++) {
      memset(bm, 0, bm_bytes);
      int64_t ns = nn_off->a[i], ne = nn_off->a[i + 1];
      for (int64_t j = ns; j < ne; j++) {
        int64_t uid = hood_ids->a[nn_nbr->a[j]];
        int64_t ls = lab_off->a[uid], le = lab_off->a[uid + 1];
        for (int64_t k = ls; k < le; k++) {
          uint64_t lab = (uint64_t)lab_nbr->a[k];
          bm[lab / 8] |= (uint8_t)(1 << (lab % 8));
        }
      }
      uint64_t wp = prefix[i];
      uint64_t base = i * n_labels;
      for (uint64_t f = 0; f < n_labels; f++) {
        if (bm[f / 8] & (1 << (f % 8)))
          out->a[wp++] = (int64_t)(base + f);
      }
    }
    free(bm);
  }

  free(prefix);
  return 1;
}

static int tm_csr_neighbor_average (lua_State *L)
{
  tk_ivec_t *nn_off = tk_ivec_peek(L, 1, "nn_offsets");
  tk_ivec_t *nn_nbr = tk_ivec_peek(L, 2, "nn_neighbors");
  tk_dvec_t *nn_w = tk_dvec_peek(L, 3, "nn_weights");
  tk_ivec_t *hood_ids = tk_ivec_peek(L, 4, "hood_ids");
  tk_dvec_t *raw_codes = tk_dvec_peek(L, 5, "raw_codes");
  tk_ivec_t *code_ids = tk_ivec_peek(L, 6, "code_ids");
  uint64_t d = tk_lua_checkunsigned(L, 7, "n_dims");

  tk_iumap_t *uid_to_pos = tk_iumap_create(0, 0);
  if (!uid_to_pos)
    return luaL_error(L, "neighbor_average: allocation failed");
  for (uint64_t i = 0; i < code_ids->n; i++) {
    int absent;
    uint32_t k = tk_iumap_put(uid_to_pos, code_ids->a[i], &absent);
    tk_iumap_setval(uid_to_pos, k, (int64_t)i);
  }

  uint64_t n_queries = nn_off->n - 1;
  tk_dvec_t *out = tk_dvec_create(L, n_queries * d, 0, 0);
  out->n = n_queries * d;
  memset(out->a, 0, n_queries * d * sizeof(double));

  #pragma omp parallel for schedule(dynamic, 64)
  for (uint64_t i = 0; i < n_queries; i++) {
    int64_t ns = nn_off->a[i], ne = nn_off->a[i + 1];
    double wsum = 0.0;
    double *row = out->a + i * d;
    for (int64_t j = ns; j < ne; j++) {
      int64_t uid = hood_ids->a[nn_nbr->a[j]];
      uint32_t ki = tk_iumap_get(uid_to_pos, uid);
      if (ki == tk_iumap_end(uid_to_pos))
        continue;
      int64_t pos = tk_iumap_val(uid_to_pos, ki);
      double w = nn_w->a[j];
      wsum += w;
      const double *src = raw_codes->a + pos * (int64_t)d;
      for (uint64_t k = 0; k < d; k++)
        row[k] += w * src[k];
    }
    if (wsum > 0.0)
      for (uint64_t k = 0; k < d; k++)
        row[k] /= wsum;
  }

  tk_iumap_destroy(uid_to_pos);
  return 1;
}

static int tm_csr_ivec_complement (lua_State *L)
{
  tk_ivec_t *subset = tk_ivec_peek(L, 1, "subset");
  uint64_t n = tk_lua_checkunsigned(L, 2, "n");
  uint64_t cn = n - subset->n;
  tk_ivec_t *out = tk_ivec_create(L, cn, 0, 0);
  out->n = cn;
  uint64_t si = 0, oi = 0;
  for (uint64_t i = 0; i < n; i++) {
    if (si < subset->n && (uint64_t)subset->a[si] == i)
      si++;
    else
      out->a[oi++] = (int64_t)i;
  }
  return 1;
}

static int tm_csr_scatter_fixed_k (lua_State *L)
{
  tk_ivec_t *dst_nbr = tk_ivec_peek(L, 1, "dst_neighbors");
  tk_dvec_t *dst_sco = tk_dvec_peek(L, 2, "dst_scores");
  tk_ivec_t *src_off = tk_ivec_peek(L, 3, "src_offsets");
  tk_ivec_t *src_nbr = tk_ivec_peek(L, 4, "src_neighbors");
  tk_dvec_t *src_sco = tk_dvec_peek(L, 5, "src_scores");
  tk_ivec_t *val_ids = tk_ivec_peek(L, 6, "val_ids");
  uint64_t k = tk_lua_checkunsigned(L, 7, "k");
  uint64_t val_n = val_ids->n;
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < val_n; i++) {
    int64_t orig = val_ids->a[i];
    int64_t dst = orig * (int64_t)k;
    int64_t src = src_off->a[i];
    for (uint64_t j = 0; j < k; j++) {
      dst_nbr->a[dst + (int64_t)j] = src_nbr->a[src + (int64_t)j];
      dst_sco->a[dst + (int64_t)j] = src_sco->a[src + (int64_t)j];
    }
  }
  return 0;
}

static int tm_csr_scatter_rows (lua_State *L)
{
  tk_dvec_t *dst = tk_dvec_peek(L, 1, "dst");
  tk_dvec_t *src = tk_dvec_peek(L, 2, "src");
  tk_ivec_t *val_ids = tk_ivec_peek(L, 3, "val_ids");
  uint64_t stride = tk_lua_checkunsigned(L, 4, "stride");
  uint64_t val_n = val_ids->n;
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < val_n; i++) {
    int64_t orig = val_ids->a[i];
    memcpy(dst->a + orig * (int64_t)stride,
           src->a + (int64_t)(i * stride),
           stride * sizeof(double));
  }
  return 0;
}

static int tm_csr_subsample (lua_State *L)
{
  lua_settop(L, 3);
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *neighbors = tk_ivec_peek(L, 2, "neighbors");
  tk_ivec_t *sample_ids = tk_ivec_peek(L, 3, "sample_ids");
  int64_t n = (int64_t)sample_ids->n;
  tk_ivec_t *new_off = tk_ivec_create(L, (uint64_t)(n + 1), NULL, NULL);
  int64_t total = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t sid = sample_ids->a[i];
    total += offsets->a[sid + 1] - offsets->a[sid];
  }
  tk_ivec_t *new_nbr = tk_ivec_create(L, (uint64_t)total, NULL, NULL);
  int64_t pos = 0;
  for (int64_t i = 0; i < n; i++) {
    int64_t sid = sample_ids->a[i];
    int64_t lo = offsets->a[sid];
    int64_t hi = offsets->a[sid + 1];
    new_off->a[i] = pos;
    for (int64_t j = lo; j < hi; j++)
      new_nbr->a[pos++] = neighbors->a[j];
  }
  new_off->a[n] = pos;
  return 2;
}

static int tm_csr_sort_csr_desc (lua_State *L)
{
  tk_ivec_t *off = tk_ivec_peek(L, 1, "offsets");
  tk_ivec_t *nbr = tk_ivec_peek(L, 2, "neighbors");
  tk_dvec_t *scores = tk_dvec_peek(L, 3, "scores");
  uint64_t n = off->n - 1;
  tk_ivec_t *out_n = tk_ivec_create(L, nbr->n, NULL, NULL);
  tk_dvec_t *out_s = tk_dvec_create(L, scores->n, NULL, NULL);
  memcpy(out_n->a, nbr->a, nbr->n * sizeof(int64_t));
  memcpy(out_s->a, scores->a, scores->n * sizeof(double));
  #pragma omp parallel for schedule(dynamic, 64)
  for (uint64_t i = 0; i < n; i++) {
    int64_t s = off->a[i], e = off->a[i + 1];
    for (int64_t j = s + 1; j < e; j++) {
      double ks = out_s->a[j];
      int64_t kn = out_n->a[j];
      int64_t p = j - 1;
      while (p >= s && out_s->a[p] < ks) {
        out_s->a[p + 1] = out_s->a[p];
        out_n->a[p + 1] = out_n->a[p];
        p--;
      }
      out_s->a[p + 1] = ks;
      out_n->a[p + 1] = kn;
    }
  }
  return 2;
}

static luaL_Reg tm_csr_fns[] = {
  { "to_csc", tm_csr_to_csc },
  { "to_hypervector", tm_csr_bits_to_hv },
  { "bipartite", tm_csr_bipartite },
  { "bipartite_neg", tm_csr_bipartite_neg },
  { "random_pairs", tm_csr_random_pairs },
  { "weight_from_index", tm_csr_weight_from_index },
  { "merge", tm_csr_merge },
  { "symmetrize", tm_csr_symmetrize },
  { "bits_select", tm_csr_bits_select },
  { "propagate", tm_csr_propagate },
  { "stratified_sample", tm_csr_stratified_sample },
  { "label_union", tm_csr_label_union },
  { "neighbor_average", tm_csr_neighbor_average },
  { "ivec_complement", tm_csr_ivec_complement },
  { "scatter_fixed_k", tm_csr_scatter_fixed_k },
  { "scatter_rows", tm_csr_scatter_rows },
  { "subsample", tm_csr_subsample },
  { "sort_csr_desc", tm_csr_sort_csr_desc },
  { NULL, NULL }
};

int luaopen_santoku_learn_csr (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_csr_fns, 0);

  tk_inv_hoods_create(L, 0, 0, 0);
  luaL_getmetafield(L, -1, "__index");
  lua_pushcfunction(L, tm_inv_hoods_to_csr);
  lua_setfield(L, -2, "to_csr");
  lua_pop(L, 2);

  tk_ann_hoods_create(L, 0, 0, 0);
  luaL_getmetafield(L, -1, "__index");
  lua_pushcfunction(L, tm_ann_hoods_to_csr);
  lua_setfield(L, -2, "to_csr");
  lua_pop(L, 2);

  return 1;
}
