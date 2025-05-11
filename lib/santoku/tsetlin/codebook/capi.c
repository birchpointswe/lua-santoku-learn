#include "lua.h"
#include "lauxlib.h"
#include "lbfgs.h"
#include "../conf.h"
#include <arpack/arpack.h>
#include "roaring.h"
#include "roaring.c"

static inline void tk_lua_callmod (
  lua_State *L,
  int nargs,
  int nret,
  const char *smod,
  const char *sfn
) {
  lua_getglobal(L, "require"); // arg req
  lua_pushstring(L, smod); // arg req smod
  lua_call(L, 1, 1); // arg mod
  lua_pushstring(L, sfn); // args mod sfn
  lua_gettable(L, -2); // args mod fn
  lua_remove(L, -2); // args fn
  lua_insert(L, - nargs - 1); // fn args
  lua_call(L, nargs, nret); // results
}

static inline void *tk_lua_checkuserdata (lua_State *L, int i, char *mt)
{
  if (mt == NULL && (lua_islightuserdata(L, i) || lua_isuserdata(L, i)))
    return lua_touserdata(L, i);
  void *p = luaL_checkudata(L, -1, mt);
  lua_pop(L, 1);
  return p;
}

static inline int tk_lua_absindex (lua_State *L, int i)
{
  if (i < 0 && i > LUA_REGISTRYINDEX)
    i += lua_gettop(L) + 1;
  return i;
}

static inline int tk_error (
  lua_State *L,
  const char *label,
  int err
) {
  lua_pushstring(L, label);
  lua_pushstring(L, strerror(err));
  tk_lua_callmod(L, 2, 0, "santoku.error", "error");
  return 1;
}

static inline void *tk_realloc (
  lua_State *L,
  void *p,
  size_t s
) {
  p = realloc(p, s);
  if (!p) {
    tk_error(L, "realloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline void *tk_malloc (
  lua_State *L,
  size_t s
) {
  void *p = malloc(s);
  if (!p) {
    tk_error(L, "malloc failed", ENOMEM);
    return NULL;
  } else {
    memset(p, 0, s);
    return p;
  }
}

static inline int tk_lua_verror (lua_State *L, int n, ...) {
  va_list args;
  va_start(args, n);
  for (int i = 0; i < n; i ++) {
    const char *str = va_arg(args, const char *);
    lua_pushstring(L, str);
  }
  va_end(args);
  tk_lua_callmod(L, n, 0, "santoku.error", "error");
  return 0;
}

static inline const char *tk_lua_checkstring (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checkstring(L, i);
}

static inline const char *tk_lua_checklstring (lua_State *L, int i, size_t *lenp, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checklstring(L, i, lenp);
}

static inline const char *tk_lua_fchecklstring (lua_State *L, int i, size_t *lp, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, field, "field is not a string");
  const char *s = luaL_checklstring(L, -1, lp);
  lua_pop(L, 1);
  return s;
}

static inline lua_Integer tk_lua_ftype (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  int t = lua_type(L, -1);
  lua_pop(L, 1);
  return t;
}

static inline uint64_t tk_lua_checkunsigned (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  return (uint64_t) l;
}

static inline double tk_lua_checkposdouble (lua_State *L, int i)
{
  lua_Number l = luaL_checknumber(L, i);
  if (l < 0)
    luaL_error(L, "value can't be negative");
  return (double) l;
}

static inline double tk_lua_fcheckposdouble (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_Number l = luaL_checknumber(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive number");
  lua_pop(L, 1);
  return (double) l;
}

static inline uint64_t tk_lua_fcheckunsigned (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_pop(L, 1);
  return (uint64_t) l;
}

static inline void tk_lua_register (lua_State *L, luaL_Reg *regs, int nup)
{
  while (true) {
    if ((*regs).name == NULL)
      break;
    for (int i = 0; i < nup; i ++)
      lua_pushvalue(L, -nup); // t upsa upsb
    lua_pushcclosure(L, (*regs).func, nup); // t upsa fn
    lua_setfield(L, -nup - 2, (*regs).name); // t
    regs ++;
  }
  lua_pop(L, nup);
}

typedef struct {
  tm_pair_t *pos;
  tm_pair_t *neg;
  uint64_t n_pos;
  uint64_t n_neg;
  uint64_t n_sentences;
  uint64_t n_hidden;
  lua_State *L;
  int i_each;
  unsigned int *global_iter;
} tm_ctx_t;

static double lbfgs_evaluate (
  void *instance,
  const double *x,
  double *g,
  const int n,
  const double step
) {
  tm_ctx_t *ctx = (tm_ctx_t *) instance;
  double loss = 0.0;
  for (int idx = 0; idx < n; idx ++)
    g[idx] = 0.0;
  for (unsigned int k = 0; k < ctx->n_pos + ctx->n_neg; k ++) {
    tm_pair_t *pairs = k < ctx->n_pos ? ctx->pos : ctx->neg;
    uint64_t offset = k < ctx->n_pos ? 0 : ctx->n_pos;
    int S = k < ctx->n_pos ? +1 : -1;
    tm_pair_t p = pairs[k - offset];
    double dot = 0.0;
    double *xi = (double *) (x + ((uint64_t) p.u) * ctx->n_hidden);
    double *xj = (double *) (x + ((uint64_t) p.v) * ctx->n_hidden);
    for (unsigned int f = 0; f < ctx->n_hidden; f ++)
      dot += xi[f] * xj[f];
    double target = (double) ctx->n_hidden * (double) S;
    double diff = dot - target;
    loss += diff * diff;
    double c = 2.0 * diff;
    for (unsigned int f = 0; f < ctx->n_hidden; f ++) {
      double xv = xi[f];
      double yv = xj[f];
      g[((uint32_t) p.u) * ctx->n_hidden + f] += c * yv;
      g[((uint32_t) p.v) * ctx->n_hidden + f] += c * xv;
    }
  }
  return loss;
}


// static double lbfgs_evaluate (
//   void *instance,
//   const double *x,
//   double *g,
//   const int n,
//   const double step
// ) {
//   tm_ctx_t *ctx = (tm_ctx_t *) instance;
//   const int H = ctx->n_hidden;
//   const double tau_p = (double) H * 0.9;
//   const double tau_n = (double) H * 0.3;
//   const double lambda_p = 1.0;
//   const double lambda_n = 5.0;
//   double loss = 0.0;
//   for (int i = 0; i < n; i ++)
//     g[i] = 0.0;
//   for (uint64_t k = 0; k < ctx->n_pos + ctx->n_neg; k ++) {
//     bool is_pos = (k < ctx->n_pos);
//     tm_pair_t p = is_pos
//       ? ctx->pos[k]
//       : ctx->neg[k - ctx->n_pos];
//     int u = p.u;
//     int v = p.v;
//     const double *zi = x + (size_t) u * (size_t) H;
//     const double *zj = x + (size_t) v * (size_t) H;
//     double dot = 0.0;
//     for (int f = 0; f < H; f ++)
//       dot += zi[f] * zj[f];
//     if (is_pos) {
//       double d = tau_p - dot;
//       if (d > 0.0) {
//         loss += lambda_p * d * d;
//         double c = -2.0 * lambda_p * d;
//         for (int f = 0; f < H; f ++) {
//           g[u * H + f] += c * zj[f];
//           g[v * H + f] += c * zi[f];
//         }
//       }
//     } else {
//       double d = dot - tau_n;
//       if (d > 0.0) {
//         loss += lambda_n * d * d;
//         double c = 2.0 * lambda_n * d;
//         for (int f = 0; f < H; f ++) {
//           g[u * H + f] += c * zj[f];
//           g[v * H + f] += c * zi[f];
//         }
//       }
//     }
//   }
//   return loss;
// }

static int lbfgs_progress (
  void *instance,
  const double *x,
  const double *g,
  double fx,
  double xnorm,
  double gnorm,
  double step,
  int n,
  int k,
  int ls
) {
  tm_ctx_t *ctx = (tm_ctx_t*) instance;
  (*ctx->global_iter) ++;
  if (ctx->i_each != -1) {
    lua_pushvalue(ctx->L, ctx->i_each);
    lua_pushinteger(ctx->L, *ctx->global_iter);
    lua_pushstring(ctx->L, "lbfgs");
    lua_pushnumber(ctx->L, fx);
    lua_call(ctx->L, 3, 1);
    bool stop =
      lua_type(ctx->L, -1) == LUA_TBOOLEAN &&
      lua_toboolean(ctx->L, -1) == 0;
    lua_pop(ctx->L, 1);
    if (stop)
      return 1;
  }
  return 0;
}

typedef struct {
  uint64_t n_original;
  uint64_t n_components;
  int64_t *components;
  int64_t *parent;
  int64_t *rank;
  int64_t **members;
  uint64_t *count;
  uint64_t *cap;
} tm_dsu_t;

static inline void tm_dsu_free (tm_dsu_t *dsu)
{
  for (uint64_t i = 0; i < dsu->n_original; i ++)
    if (dsu->members[i])
      free(dsu->members[i]);
  free(dsu->members);
  free(dsu->count);
  free(dsu->cap);
  free(dsu->parent);
  free(dsu->rank);
  free(dsu->components);
}

static inline int64_t tm_dsu_find (tm_dsu_t *dsu, int64_t x)
{
  if (dsu->parent[x] != x)
    dsu->parent[x] = tm_dsu_find(dsu, dsu->parent[x]);
  return dsu->parent[x];
}

static inline void tm_dsu_union (lua_State *L, tm_dsu_t *dsu, int64_t x, int64_t y)
{
  int64_t xr = tm_dsu_find(dsu, x);
  int64_t yr = tm_dsu_find(dsu, y);
  if (xr == yr)
    return;
  if (dsu->rank[xr] == dsu->rank[yr])
    dsu->rank[xr] ++;
  int64_t big = xr, small = yr;
  if (dsu->rank[xr] < dsu->rank[yr]) {
    big = yr;
    small = xr;
  }
  dsu->parent[small] = big;
  uint64_t new_count = dsu->count[big] + dsu->count[small];
  if (dsu->cap[big] < new_count) {
    dsu->cap[big] = new_count;
    dsu->members[big] = tk_realloc(L, dsu->members[big], dsu->cap[big] * sizeof(int64_t));
  }
  memcpy(
    dsu->members[big] + dsu->count[big],
    dsu->members[small],
    dsu->count[small] * sizeof(int64_t));
  dsu->count[big] = new_count;
  free(dsu->members[small]);
  dsu->members[small] = NULL;
  dsu->count[small] = 0;
  dsu->cap[small] = 0;
  uint64_t old_n = dsu->n_components;
  dsu->n_components --;
  for (uint64_t i = 0; i < old_n; i ++) {
    if (dsu->components[i] == small) {
      dsu->components[i] = dsu->components[dsu->n_components];
      break;
    }
  }
}

static inline void tm_dsu_init (
  lua_State *L,
  tm_dsu_t *dsu,
  tm_pairs_t *pairs,
  uint64_t n
) {
  dsu->n_original = n;
  dsu->n_components = n;
  dsu->components = tk_malloc(L, (size_t) n * sizeof(int64_t));
  dsu->parent = tk_malloc(L, (size_t) n * sizeof(int64_t));
  dsu->rank = tk_malloc(L, (size_t) n * sizeof(int64_t));
  dsu->members = tk_malloc(L, (size_t) n * sizeof(int64_t *));
  dsu->count = tk_malloc(L, (size_t) n * sizeof(uint64_t));
  dsu->cap = tk_malloc(L, (size_t) n * sizeof(uint64_t));
  for (uint64_t i = 0; i < n; i ++) {
    dsu->components[i] = (int64_t) i;
    dsu->parent[i] = (int64_t) i;
    dsu->rank[i] = 0;
    dsu->members[i] = tk_malloc(L, sizeof(int64_t));
    dsu->members[i][0] = (int64_t) i;
    dsu->count[i] = 1;
    dsu->cap[i] = 1;
  }
  bool l;
  tm_pair_t p;
  kh_foreach(pairs, p, l, ({
    if (l)
      tm_dsu_union(L, dsu, p.u, p.v);
  }))
}

static inline void tm_render_pairs (
  lua_State *L,
  tm_pairs_t *pairs,
  tm_pair_t *pos,
  tm_pair_t *neg,
  uint64_t *n_pos,
  uint64_t *n_neg
) {
  bool l;
  tm_pair_t p;
  uint64_t wp = 0, wn = 0;
  kh_foreach(pairs, p,  l, ({
    if (l) pos[wp ++] = p;
    else neg[wn ++] = p;
  }))
  *n_pos = wp;
  *n_neg = wn;
  ks_introsort(pair_asc, wp, pos);
  ks_introsort(pair_asc, wn, neg);
}

static inline void tm_pairs_init (
  lua_State *L,
  tm_pairs_t *pairs,
  tm_pair_t *pos,
  tm_pair_t *neg,
  uint64_t *n_pos,
  uint64_t *n_neg
) {
  int kha;
  khint_t khi;

  uint64_t n_pos_old = *n_pos;
  uint64_t n_pos_new = 0;

  for (uint64_t i = 0; i < n_pos_old; i ++) {
    tm_pair_t p = pos[i];
    if (p.v < p.u)
      pos[i] = p = tm_pair(p.v, p.u);
    khi = kh_put(pairs, pairs, p, &kha);
    if (!kha)
      continue;
    kh_value(pairs, khi) = true;
    n_pos_new ++;
  }

  uint64_t n_neg_old = *n_neg;
  uint64_t n_neg_new = 0;

  for (uint64_t i = 0; i < n_neg_old; i ++) {
    tm_pair_t p = neg[i];
    if (p.v < p.u)
      neg[i] = p = tm_pair(p.v, p.u);
    khi = kh_put(pairs, pairs, p, &kha);
    if (!kha)
      continue;
    kh_value(pairs, khi) = false;
    n_neg_new ++;
  }

  *n_pos = n_pos_new;
  *n_neg = n_neg_new;

}

static inline void tm_index_init (
  lua_State *L,
  roaring64_bitmap_t **index,
  int64_t *set_bits,
  uint64_t n_set_bits,
  uint64_t n_features
) {
  for (uint64_t i = 0; i < n_features; i ++)
    index[i] = roaring64_bitmap_create();
  for (uint64_t i = 0; i < n_set_bits; i ++) {
    int64_t id = set_bits[i];
    int64_t sid = id / (int64_t) n_features;
    int64_t fid = id % (int64_t) n_features;
    roaring64_bitmap_add(index[fid], (uint64_t) sid);
  }
}

static inline void tm_neighbors_init (
  lua_State *L,
  tm_dsu_t *dsu,
  tm_neighbors_t *neighbors,
  roaring64_bitmap_t **index,
  roaring64_bitmap_t **sentences,
  uint64_t n_sentences,
  uint64_t n_features,
  uint64_t knn
) {
  roaring64_bitmap_t *candidates = roaring64_bitmap_create();

  for (int64_t u = 0; u < (int64_t) n_sentences; u ++) {

    tm_neighbors_t nbrs;
    kv_init(nbrs);
    int64_t cu = tm_dsu_find(dsu, u);

    // Populate candidate set (those sharing at least one feature)
    roaring64_bitmap_clear(candidates);
    roaring64_iterator_t *it0 = roaring64_iterator_create(sentences[u]);
    while (roaring64_iterator_has_value(it0)) {
      uint64_t f = roaring64_iterator_value(it0);
      roaring64_bitmap_or_inplace(candidates, index[f]);
      roaring64_iterator_advance(it0);
    }
    roaring64_iterator_free(it0);

    // Get a sorted list of neighbors by distance
    roaring64_iterator_t *it1 = roaring64_iterator_create(candidates);
    while (roaring64_iterator_has_value(it1)) {
      int64_t v = (int64_t) roaring64_iterator_value(it1);
      roaring64_iterator_advance(it1);
      if (u == v || cu == tm_dsu_find(dsu, v))
        continue;
      double dist = 1 - roaring64_bitmap_jaccard_index(sentences[u], sentences[v]);
      kv_push(tm_neighbor_t, nbrs, ((tm_neighbor_t) { .v = v, .d = dist }));
    }
    roaring64_iterator_free(it1);
    if (nbrs.n <= knn)
      ks_introsort(neighbors_asc, nbrs.n, nbrs.a);
    else
      ks_ksmall(neighbors_asc, nbrs.n, nbrs.a, knn);

    // Truncate neighbors
    if (nbrs.n > knn)
      nbrs.n = knn;
    kv_resize(tm_neighbor_t, nbrs, nbrs.n);

    // Update refs
    neighbors[u] = nbrs;
  }

  // Cleanup
  roaring64_bitmap_free(candidates);
}

static inline void tm_add_backbone (
  lua_State *L,
  tm_dsu_t *dsu,
  tm_pairs_t *pairs,
  tm_neighbors_t *neighbors,
  roaring64_bitmap_t **sentences,
  uint64_t *n_pos,
  uint64_t *n_neg,
  uint64_t n_sentences,
  uint64_t n_features,
  int i_each,
  unsigned int *global_iter
) {
  int kha;
  khint_t khi;

  // Log initial density
  (*global_iter)++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushstring(L, "densify");
    lua_pushinteger(L, (int64_t) dsu->n_components);
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_call(L, 5, 0);
  }

  // Gather all inter-component edges
  tm_candidates_t all_candidates;
  kv_init(all_candidates);
  for (int64_t u = 0; u < (int64_t) dsu->n_original; u++) {
    int64_t cu = tm_dsu_find(dsu, u);
    tm_neighbors_t nbrs = neighbors[u];
    for (khint_t i = 0; i < nbrs.n; i ++) {
      tm_neighbor_t v = nbrs.a[i];
      if (cu == tm_dsu_find(dsu, v.v))
        continue;
      tm_pair_t e = tm_pair(u, v.v);
      khi = kh_get(pairs, pairs, e);
      if (khi != kh_end(pairs))
        continue;
      kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v));
    }
  }

  // Sort all by distance ascending (nearest in feature space)
  ks_introsort(candidates_asc, all_candidates.n, all_candidates.a);

  // Loop through candidates in order and add if they connect components
  for (khint_t i = 0; i < all_candidates.n && dsu->n_components > 1; i ++) {
    tm_candidate_t c = all_candidates.a[i];
    int64_t cu = tm_dsu_find(dsu, c.u);
    int64_t cv = tm_dsu_find(dsu, c.v.v);
    if (cu == cv)
      continue;
    tm_pair_t e = tm_pair(c.u, c.v.v);
    khi = kh_put(pairs, pairs, e, &kha);
    if (!kha)
      continue;
    kh_value(pairs, khi) = true;
    tm_dsu_union(L, dsu, c.u, c.v.v);
    (*n_pos)++;
  }

  // Log final density
  (*global_iter)++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushstring(L, "densify");
    lua_pushinteger(L, (int64_t) dsu->n_components);
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_call(L, 5, 0);
  }

  // Cleanup
  kv_destroy(all_candidates);
}

static inline void tm_adj_init (
  tm_pairs_t *pairs,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  uint64_t n_sentences
) {
  khint_t khi;

  // Init
  for (uint64_t s = 0; s < n_sentences; s ++) {
    adj_pos[s] = roaring64_bitmap_create();
    adj_neg[s] = roaring64_bitmap_create();
  }

  // Populate adj lists
  for (khi = kh_begin(pairs); khi < kh_end(pairs); khi ++) {
    if (!kh_exist(pairs, khi))
      continue;
    tm_pair_t p = kh_key(pairs, khi);
    bool l = kh_value(pairs, khi);
    if (l) {
      roaring64_bitmap_add(adj_pos[p.u], (uint64_t) p.v);
      roaring64_bitmap_add(adj_pos[p.v], (uint64_t) p.u);
    } else {
      roaring64_bitmap_add(adj_neg[p.u], (uint64_t) p.v);
      roaring64_bitmap_add(adj_neg[p.v], (uint64_t) p.u);
    }
  }
}

static inline void tm_add_transatives (
  lua_State *L,
  tm_pairs_t *pairs,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  roaring64_bitmap_t **sentences,
  uint64_t *n_pos,
  uint64_t *n_neg,
  uint64_t n_sentences,
  uint64_t n_hops_pos,
  uint64_t n_hops_neg,
  uint64_t n_grow_pos,
  uint64_t n_grow_neg,
  int i_each,
  unsigned int *global_iter
) {
  int kha;
  khint_t khi;

  // Add transative positives
  tm_candidates_t candidates;
  kv_init(candidates);
  for (uint64_t layer = 0; layer < n_hops_pos; layer ++) {
    bool added_any = false;
    for (uint32_t u = 0; u < n_sentences; ++u) {
      kv_size(candidates) = 0;
      roaring64_iterator_t *it1 = roaring64_iterator_create(adj_pos[u]);
      while (roaring64_iterator_has_value(it1)) {
        uint32_t v = roaring64_iterator_value(it1);
        roaring64_iterator_advance(it1);
        roaring64_iterator_t *it2 = roaring64_iterator_create(adj_pos[v]);
        while (roaring64_iterator_has_value(it2)) {
          uint32_t w = roaring64_iterator_value(it2);
          roaring64_iterator_advance(it2);
          if (w == u || roaring64_bitmap_contains(adj_pos[u], w))
            continue;
          double dist = 1 - roaring64_bitmap_jaccard_index(sentences[u], sentences[w]);
          tm_neighbor_t vw = (tm_neighbor_t) { w, dist };
          kv_push(tm_candidate_t, candidates, tm_candidate(u, vw));
        }
        roaring64_iterator_free(it2);
      }
      roaring64_iterator_free(it1);
      // Prefer nearest in feature space first
      ks_introsort(candidates_asc, candidates.n, candidates.a);
      for (uint64_t i = 0; i < candidates.n && i < n_grow_pos; i ++) {
        tm_candidate_t c = candidates.a[i];
        tm_pair_t e = tm_pair(u, c.v.v);
        khi = kh_put(pairs, pairs, e, &kha);
        if (!kha)
          continue;
        kh_value(pairs, khi) = true;
        roaring64_bitmap_add(adj_pos[u], (uint64_t) c.v.v);
        roaring64_bitmap_add(adj_pos[c.v.v], u);
        (*n_pos) ++;
        added_any = true;
      }
    }
    if (!added_any)
      break;
  }

  // Add transative negatives
  for (uint64_t layer = 0; layer < n_hops_neg; layer ++) {
    bool added_any = false;
    for (int64_t u = 0; u < (int64_t) n_sentences; u ++) {
      kv_size(candidates) = 0;
      roaring64_iterator_t *it1 = roaring64_iterator_create(adj_pos[u]);
      while (roaring64_iterator_has_value(it1)) {
        uint64_t v = roaring64_iterator_value(it1);
        roaring64_iterator_advance(it1);
        roaring64_iterator_t *it2 = roaring64_iterator_create(adj_neg[v]);
        while (roaring64_iterator_has_value(it2)) {
          int64_t w = (int64_t) roaring64_iterator_value(it2);
          roaring64_iterator_advance(it2);
          if (w == u)
            continue;
          double dist = 1.0 - roaring64_bitmap_jaccard_index(sentences[u], sentences[w]);
          tm_neighbor_t vw = (tm_neighbor_t) { w, dist };
          kv_push(tm_candidate_t, candidates, tm_candidate(u, vw));
        }
        roaring64_iterator_free(it2);
      }
      roaring64_iterator_free(it1);
      // Prefer nearest in feature space first
      ks_introsort(candidates_asc, candidates.n, candidates.a);
      for (uint64_t i = 0; i < candidates.n && i < n_grow_neg; i ++) {
        tm_candidate_t c = candidates.a[i];
        tm_pair_t e = tm_pair(u, c.v.v);
        khi = kh_put(pairs, pairs, e, &kha);
        if (!kha)
          continue;
        kh_value(pairs, khi) = false;
        roaring64_bitmap_add(adj_neg[u], (uint64_t) c.v.v);
        roaring64_bitmap_add(adj_neg[c.v.v], (uint64_t) u);
        (*n_neg) ++;
        added_any = true;
      }
    }
    if (!added_any)
      break;
  }

  // Cleanup
  kv_destroy(candidates);

  // Progress
  (*global_iter)++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushstring(L, "densify");
    lua_pushinteger(L, (int64_t) 1);
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_call(L, 5, 0);
  }

}

static inline void tm_sentences_init (
  roaring64_bitmap_t **sentences,
  int64_t *set_bits,
  uint64_t n_set_bits,
  uint64_t n_features,
  uint64_t n_sentences
) {
  for (unsigned int i = 0; i < n_sentences; i ++)
    sentences[i] = roaring64_bitmap_create();
  for (uint64_t b = 0; b < n_set_bits; b ++) {
    int64_t v = set_bits[b];
    int64_t s = v / (int64_t) n_features;
    int64_t f = v % (int64_t) n_features;
    roaring64_bitmap_add(sentences[s], (uint64_t) f);
  }
}

static inline void tm_run_tch (
  lua_State *L,
  double *z,
  tk_bits_t *codes,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  uint64_t n_sentences,
  uint64_t n_hidden,
  int i_each,
  unsigned int *global_iter
) {
  int *bitvec = tk_malloc(L, n_sentences * sizeof(int));
  double *col = malloc(n_sentences * sizeof(double));

  for (uint32_t f = 0; f < n_hidden; f ++) {

    // Find the median
    for (uint32_t i = 0; i < n_sentences; i ++)
      col[i] = z[i * n_hidden + f];
    uint64_t mid = n_sentences / 2;
    ks_ksmall(f64, n_sentences, col, mid);
    double med = col[mid];

    // Threshold around the median
    for (uint32_t i = 0; i < n_sentences; i++)
      bitvec[i] = (z[i * n_hidden + f] > med) ? +1 : -1;

    bool updated;
    uint32_t steps = 0;

    do {
      updated = false;
      steps ++;

      for (uint32_t i = 0; i < n_sentences; i ++) {
        int delta = 0;
        // Positive neighbors
        roaring64_iterator_t *it0 = roaring64_iterator_create(adj_pos[i]);
        while (roaring64_iterator_has_value(it0)) {
          uint32_t j = roaring64_iterator_value(it0);
          roaring64_iterator_advance(it0);
          delta += bitvec[i] * bitvec[j];
        }
        roaring64_iterator_free(it0);
        // Negative neighbors
        roaring64_iterator_t *it1 = roaring64_iterator_create(adj_neg[i]);
        while (roaring64_iterator_has_value(it1)) {
          uint32_t j = roaring64_iterator_value(it1);
          roaring64_iterator_advance(it1);
          delta -= bitvec[i] * bitvec[j];
        }
        roaring64_iterator_free(it1);
        // Check
        if (delta < 0){
          bitvec[i] = -bitvec[i];
          updated = true;
        }
      }

      (*global_iter) ++;
      if (i_each >= 0) {
        lua_pushvalue(L, i_each);
        lua_pushinteger(L, *global_iter);
        lua_pushstring(L, "tch");
        lua_pushinteger(L, f);
        lua_pushinteger(L, steps);
        lua_call(L, 4, 0);
      }

    } while (updated);

    // Write out the final bits into your packed codes
    for (uint32_t i = 0; i < n_sentences; i ++) {
      if (bitvec[i] > 0){
        uint32_t chunk = f / BITS;
        uint32_t b = f % BITS;
        codes[i * BITS_DIV(n_hidden) + chunk] |= ((tk_bits_t) 1 << b);
      }
    }
  }

  // Cleanup
  free(bitvec);
  free(col);
}

static inline void tm_run_lbfgs (
  lua_State *L,
  double *z,
  tm_pair_t *pos,
  tm_pair_t *neg,
  uint64_t n_pos,
  uint64_t n_neg,
  uint64_t n_sentences,
  uint64_t n_hidden,
  int i_each,
  unsigned int *global_iter,
  uint64_t lbfgs_iterations
) {
  // Prepare context for LBFGS
  tm_ctx_t ctx = {
    pos, neg, n_pos, n_neg, n_sentences, n_hidden,
    L, i_each, global_iter
  };

  // LBFGS refine
  int n_vars = n_sentences * n_hidden;
  if (lbfgs_iterations > 0) {
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.max_iterations = lbfgs_iterations;
    double fx = 0.0;
    lbfgs(n_vars, z, &fx, lbfgs_evaluate, lbfgs_progress, &ctx, &param);
  }
}

static inline void tm_run_spectral (
  lua_State *L,
  double *z,
  tm_pair_t *pos,
  tm_pair_t *neg,
  uint64_t n_pos,
  uint64_t n_neg,
  uint64_t n_sentences,
  uint64_t n_hidden,
  int i_each,
  unsigned int *global_iter
) {
  // Build degree vector for Laplacian
  unsigned int *degree = tk_malloc(L, n_sentences * sizeof(unsigned int));
  memset(degree, 0, (size_t) n_sentences * sizeof(unsigned int));
  for (unsigned int k = 0; k < n_pos; k ++) {
    tm_pair_t p = pos[k];
    degree[p.u] ++;
    degree[p.v] ++;
  }
  for (unsigned int k = 0; k < n_neg; k ++) {
    tm_pair_t p = neg[k];
    degree[p.u] ++;
    degree[p.v] ++;
  }

  // Build inv_sqrt vector
  double *inv_sqrt_deg = tk_malloc(L, n_sentences * sizeof(double));
  for (uint64_t i = 0; i < n_sentences; i++)
    inv_sqrt_deg[i] = degree[i] > 0 ? 1.0 / sqrt((double) degree[i]) : 0.0;
  free(degree);

  // ARPACK setup
  a_int ido = 0, info = 0;
  char bmat[] = "I", which[] = "SM";
  a_int nev = (a_int) n_hidden + 1;
  a_int ncv = (2 * nev + 1 < (a_int) n_sentences) ? 2 * nev + 1 : (a_int) n_sentences;
  double tol = 1e-3;
  double *resid = tk_malloc(L, (size_t) n_sentences * sizeof(double));
  double *workd = tk_malloc(L, 3 * (size_t) n_sentences * sizeof(double));
  a_int iparam[11] = {1,0,3*n_hidden,0,0,0,1,0,0,0,0};
  a_int ipntr[14] = {0};
  int lworkl = ncv * (ncv + 8);
  double *workl = tk_malloc(L, (size_t) lworkl * sizeof(double));
  double *v = tk_malloc(L, (size_t) n_sentences * (size_t) ncv * sizeof(double));

  // Reverse-communication to build Lanczos basis v
  do {
    dsaupd_c(
      &ido, bmat, n_sentences, which, nev, tol, resid, ncv, v, n_sentences,
      iparam, ipntr, workd, workl, (a_int) lworkl, &info);
    if (info != 0)
      tk_lua_verror(L, 2, "codeify", "failure calling arpack for spectral refinement");
    if (ido == -1 || ido == 1) {
      double *in  = workd + ipntr[0] - 1;
      double *out = workd + ipntr[1] - 1;
      memcpy(out, in, n_sentences * sizeof(double));
      for (unsigned int e = 0; e < n_pos; e ++) {
        tm_pair_t p = pos[e];
        double w = inv_sqrt_deg[p.u] * inv_sqrt_deg[p.v];
        out[p.u] -= w * in[p.v];
        out[p.v] -= w * in[p.u];
      }
      for (unsigned int e = 0; e < n_neg; e ++) {
        tm_pair_t p = neg[e];
        double w = inv_sqrt_deg[p.u] * inv_sqrt_deg[p.v];
        out[p.u] += w * in[p.v];
        out[p.v] += w * in[p.u];
      }
    }
  } while (ido == -1 || ido == 1);

  // Prepare containers
  a_int rvec = 1;
  char howmny[] = "A";
  a_int select[ncv];
  double d[n_hidden];
  double sigma = 0.0;
  double *zall = tk_malloc(L, (size_t) n_sentences * (size_t) nev * sizeof(double));

  // Dump modes
  dseupd_c(
    rvec, howmny, select, d, zall,
    (a_int) n_sentences, sigma,
    bmat, (a_int) n_sentences, which, nev,
    tol, resid, (a_int) ncv, v, (a_int) n_sentences,
    iparam, ipntr, workd, workl, (a_int) lworkl, &info);
  if (info != 0)
    tk_lua_verror(L, 2, "codeify", "failure calling arpack for spectral finalization");

  // Copy modes to z, skipping the first
  for (uint64_t i = 0; i < n_sentences; i ++)
    for (uint64_t f = 0; f < n_hidden; f ++)
      z[i * n_hidden + f] = zall[i * (uint64_t) nev + (f + 1)];

  // Cleanup
  free(inv_sqrt_deg);
  free(zall);
  free(resid);
  free(workd);
  free(workl);
  free(v);

  (*global_iter) ++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushstring(L, "spectral");
    lua_pushinteger(L, iparam[4]);
    lua_call(L, 3, 0);
  }
}

static inline int tm_codeify (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "pos");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pos_seed = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pos_seed = (uint64_t) luaL_checkinteger(L, -1) / 2;
  tm_pair_t *pos = tk_malloc(L, n_pos_seed * sizeof(tm_pair_t));
  uint64_t n_pos = n_pos_seed;
  memcpy(pos, pos_seed, n_pos_seed * sizeof(tm_pair_t));

  lua_getfield(L, 1, "neg");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *neg_seed = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_neg_seed = (uint64_t) luaL_checkinteger(L, -1) / 2;
  tm_pair_t *neg = tk_malloc(L, n_neg_seed * sizeof(tm_pair_t));
  uint64_t n_neg = n_neg_seed;
  memcpy(neg, neg_seed, n_neg_seed * sizeof(tm_pair_t));

  lua_getfield(L, 1, "sentences");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  int64_t *set_bits = (int64_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_set_bits = (uint64_t) luaL_checkinteger(L, -1);
  uint64_t n_features = tk_lua_fcheckunsigned(L, 1, "codeify", "n_features");
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "codeify", "n_hidden");
  uint64_t n_sentences = tk_lua_fcheckunsigned(L, 1, "codeify", "n_sentences");
  uint64_t n_hops_pos = tk_lua_fcheckunsigned(L, 1, "codeify", "n_hops_pos");
  uint64_t n_hops_neg = tk_lua_fcheckunsigned(L, 1, "codeify", "n_hops_neg");
  uint64_t n_grow_pos = tk_lua_fcheckunsigned(L, 1, "codeify", "n_grow_pos");
  uint64_t n_grow_neg = tk_lua_fcheckunsigned(L, 1, "codeify", "n_grow_neg");
  uint64_t knn = tk_lua_fcheckunsigned(L, 1, "codeify", "knn");
  uint64_t lbfgs_iterations = tk_lua_fcheckunsigned(L, 1, "codeify", "lbfgs");

  if (BITS_MOD(n_hidden) != 0)
    tk_lua_verror(L, 3, "codify", "n_hidden", "must be a multiple of " STR(BITS));

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tm_pairs_t *pairs = kh_init(pairs);
  tm_pairs_init(L, pairs, pos, neg, &n_pos, &n_neg);

  tm_dsu_t dsu;
  tm_dsu_init(L, &dsu, pairs, n_sentences);

  roaring64_bitmap_t **sentences = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  tm_sentences_init(sentences, set_bits, n_set_bits, n_features, n_sentences);

  roaring64_bitmap_t **index = tk_malloc(L, n_features * sizeof(roaring64_bitmap_t *));
  tm_index_init(L, index, set_bits, n_set_bits, n_features);

  tm_neighbors_t *neighbors = tk_malloc(L, n_sentences * sizeof(tm_neighbors_t));
  tm_neighbors_init(L, &dsu, neighbors, index, sentences, n_sentences, n_features, knn);

  unsigned int global_iter = 0;

  // Densify graph
  tm_add_backbone(L, &dsu, pairs, neighbors, sentences, &n_pos, &n_neg, n_sentences, n_features, i_each, &global_iter);

  // Cleanup
  tm_dsu_free(&dsu);
  for (uint64_t f = 0; f < n_features; f ++)
    roaring64_bitmap_free(index[f]);
  free(index);

  // Flush to pair lists
  pos = tk_realloc(L, pos, n_pos * sizeof(tm_pair_t));
  neg = tk_realloc(L, neg, n_neg * sizeof(tm_pair_t));
  tm_render_pairs(L, pairs, pos, neg, &n_pos, &n_neg);

  // Spectral hashing
  double *z = tk_malloc(L, (size_t) n_sentences * (size_t) n_hidden * sizeof(double));
  tm_run_spectral(L, z, pos, neg, n_pos, n_neg, n_sentences, n_hidden, i_each, &global_iter);

  // Setup adjacency list
  roaring64_bitmap_t **adj_pos = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  roaring64_bitmap_t **adj_neg = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  tm_adj_init(pairs, adj_pos, adj_neg, n_sentences);

  // Densify graph, adding transatives
  tm_add_transatives(L, pairs, adj_pos, adj_neg, sentences, &n_pos, &n_neg, n_sentences, n_hops_pos, n_hops_neg, n_grow_pos, n_grow_neg, i_each, &global_iter);

  // Cleanup
  for (int64_t s = 0; s < (int64_t) n_sentences; s ++)
    roaring64_bitmap_free(sentences[s]);
  free(sentences);

  // Flush to pair lists
  pos = tk_realloc(L, pos, n_pos * sizeof(tm_pair_t));
  neg = tk_realloc(L, neg, n_neg * sizeof(tm_pair_t));
  tm_render_pairs(L, pairs, pos, neg, &n_pos, &n_neg);

  // Done with pair lists
  kh_destroy(pairs, pairs);

  // L-BFGS optimization using KSH loss
  tm_run_lbfgs(L, z, pos, neg, n_pos, n_neg, n_sentences, n_hidden, i_each, &global_iter, lbfgs_iterations);

  // TCH discretization
  tk_bits_t *codes = tk_malloc(L, (size_t) n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t));
  memset(codes, 0, (size_t) n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t));
  tm_run_tch(L, z, codes, adj_pos, adj_neg, n_sentences, n_hidden, i_each, &global_iter);

  // Copy codes to Lualand
  lua_pushlstring(L, (char *) codes, n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t));

  // Cleanup
  for (uint32_t u = 0; u < n_sentences; u++) {
    roaring64_bitmap_free(adj_pos[u]);
    roaring64_bitmap_free(adj_neg[u]);
  }
  free(adj_pos);
  free(adj_neg);
  free(z);
  free(pos);
  free(neg);
  free(codes);
  return 1;
}

static luaL_Reg tm_codebook_fns[] =
{
  { "codeify", tm_codeify },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_codebook_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_codebook_fns, 0); // t
  return 1;
}
