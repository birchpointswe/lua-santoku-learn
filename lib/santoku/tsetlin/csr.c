#include <santoku/tsetlin/csr.h>
#include <santoku/cvec/ext.h>
#include <santoku/ivec/ext.h>
#include <omp.h>
#include <float.h>
#include <math.h>

static int tm_inv_hoods_to_csr (lua_State *L)
{
  tk_inv_hoods_t *hoods = tk_inv_hoods_peek(L, 1, "hoods");
  tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
  uint64_t n = ids->n;
  if (hoods->n != n)
    return luaL_error(L, "hoods size %d != ids size %d", (int)hoods->n, (int)n);

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
    tk_rvec_t *hood = hoods->a[i];
    int64_t pos = off->a[i];
    for (uint64_t j = 0; j < hood->n; j++) {
      nbr->a[pos + (int64_t)j] = hood->a[j].i;
      w->a[pos + (int64_t)j] = 1.0 - hood->a[j].d;
    }
  }

  return 3;
}

static int tm_ann_hoods_to_csr (lua_State *L)
{
  tk_ann_hoods_t *hoods = tk_ann_hoods_peek(L, 1, "hoods");
  tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
  uint64_t features = tk_lua_checkunsigned(L, 3, "features");
  uint64_t n = ids->n;
  if (hoods->n != n)
    return luaL_error(L, "hoods size %d != ids size %d", (int)hoods->n, (int)n);

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

static int tm_csr_bipartite_neg (lua_State *L)
{
  tk_ivec_t *gt_off = tk_ivec_peek(L, 1, "gt_offsets");
  tk_ivec_t *gt_nbr = tk_ivec_peek(L, 2, "gt_neighbors");
  tk_ivec_t *row_ids = tk_ivec_peek(L, 3, "row_ids");
  tk_ivec_t *col_ids = tk_ivec_peek(L, 4, "col_ids");
  uint64_t n_neg = tk_lua_checkunsigned(L, 5, "n_neg");

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

    uint64_t added = 0;
    uint64_t attempts = 0;
    uint64_t max_attempts = n_neg * 10;
    while (added < n_neg && attempts < max_attempts) {
      attempts++;
      int64_t li = (int64_t)(tk_fast_random() % n_cols);
      bool is_gt = false;
      for (int64_t j = gt_start; j < gt_end; j++) {
        if (gt_nbr->a[j] == li) { is_gt = true; break; }
      }
      if (!is_gt) {
        tk_ivec_push(nbr, (int64_t)(n_rows + (uint64_t)li));
        tk_dvec_push(w, 0.0);
        added++;
      }
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
  double bandwidth = luaL_checknumber(L, 7);
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
        size_t un = 0, vn = 0;
        int64_t *uset = tk_inv_get(idx_inv, u, &un);
        int64_t *vset = tk_inv_get(idx_inv, v, &vn);
        if (uset && vset)
          dist = 1.0 - tk_inv_similarity(idx_inv, uset, un, vset, vn, decay, bandwidth);
        else
          dist = 1.0;
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
  uint64_t total = nbr1->n + nbr2->n;

  tk_ivec_t *off = tk_ivec_create(L, n + 1, 0, 0);
  tk_ivec_t *nbr = tk_ivec_create(L, total, 0, 0);
  tk_dvec_t *w = tk_dvec_create(L, total, 0, 0);
  off->n = n + 1;
  nbr->n = total;
  w->n = total;

  off->a[0] = 0;
  int64_t pos = 0;
  for (uint64_t i = 0; i < n; i++) {
    for (int64_t j = off1->a[i]; j < off1->a[i + 1]; j++) {
      nbr->a[pos] = nbr1->a[j];
      w->a[pos] = w1->a[j];
      pos++;
    }
    for (int64_t j = off2->a[i]; j < off2->a[i + 1]; j++) {
      nbr->a[pos] = nbr2->a[j];
      w->a[pos] = w2->a[j];
      pos++;
    }
    off->a[i + 1] = pos;
  }

  return 3;
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

  tk_ivec_t *off = tk_ivec_create(L, n + 1, 0, 0);
  off->a[0] = 0;
  for (uint64_t i = 0; i < n; i++)
    off->a[i + 1] = off->a[i] + (int64_t)hoods[i]->n;
  off->n = n + 1;

  uint64_t total = (uint64_t)off->a[n];
  tk_ivec_t *nbr = tk_ivec_create(L, total, 0, 0);
  tk_dvec_t *w = tk_dvec_create(L, total, 0, 0);
  nbr->n = total;
  w->n = total;

  for (uint64_t i = 0; i < n; i++) {
    int64_t pos = off->a[i];
    tk_rvec_t *h = hoods[i];
    for (uint64_t j = 0; j < h->n; j++) {
      nbr->a[pos + (int64_t)j] = h->a[j].i;
      w->a[pos + (int64_t)j] = h->a[j].d;
    }
  }

  for (uint64_t i = 0; i < n; i++)
    tk_rvec_destroy(hoods[i]);
  free(hoods);

  return 3;

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

static luaL_Reg tm_csr_fns[] = {
  { "to_hypervector", tm_csr_bits_to_hv },
  { "bipartite_neg", tm_csr_bipartite_neg },
  { "random_pairs", tm_csr_random_pairs },
  { "weight_from_index", tm_csr_weight_from_index },
  { "merge", tm_csr_merge },
  { "symmetrize", tm_csr_symmetrize },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_csr (lua_State *L)
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
