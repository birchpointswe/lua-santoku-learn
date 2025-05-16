#include "lua.h"
#include "lauxlib.h"
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

static inline bool tk_lua_optboolean (lua_State *L, int i, bool def, char *name)
{
  if (lua_type(L, i) == LUA_TNIL)
    return def;
  if (lua_type(L, i) != LUA_TBOOLEAN)
    tk_lua_verror(L, 3, name, "value is not a boolean");
  return lua_toboolean(L, i);
}

static inline bool tk_lua_foptboolean (lua_State *L, int i, bool def, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) == LUA_TNIL) {
    lua_pop(L, 1);
    return def;
  }
  if (lua_type(L, -1) != LUA_TBOOLEAN)
    tk_lua_verror(L, 3, name, field, "field is not a boolean or nil");
  bool b = lua_toboolean(L, -1);
  lua_pop(L, 1);
  return b;
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
  uint64_t n_original;
  uint64_t n_components;
  int64_t *components;
  int64_t *parent;
  int64_t *rank;
  int64_t **members;
  uint64_t *count;
  uint64_t *cap;
  roaring64_bitmap_t *unseen;
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
  roaring64_bitmap_free(dsu->unseen);
}

static inline int64_t tm_dsu_find (tm_dsu_t *dsu, int64_t x)
{
  if (dsu->parent[x] != x)
    dsu->parent[x] = tm_dsu_find(dsu, dsu->parent[x]);
  return dsu->parent[x];
}

static inline int64_t tm_dsu_components (tm_dsu_t *dsu)
{
  return (int64_t) dsu->n_components + (int64_t) roaring64_bitmap_get_cardinality(dsu->unseen);
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
  roaring64_bitmap_remove(dsu->unseen, (uint64_t) x);
  roaring64_bitmap_remove(dsu->unseen, (uint64_t) y);
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
  dsu->unseen = roaring64_bitmap_create();
  roaring64_bitmap_add_range(dsu->unseen, 0, n);
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
  tm_pairs_t *pairs,
  tm_neighbors_t *neighbors,
  roaring64_bitmap_t **index,
  roaring64_bitmap_t **sentences,
  uint64_t n_sentences,
  uint64_t n_features,
  uint64_t knn
) {
  roaring64_iterator_t it;
  roaring64_bitmap_t *candidates = roaring64_bitmap_create();

  for (int64_t u = 0; u < (int64_t) n_sentences; u ++) {

    tm_neighbors_t nbrs;
    kv_init(nbrs);

    // Populate candidate set (those sharing at least one feature)
    roaring64_bitmap_clear(candidates);
    roaring64_iterator_reinit(sentences[u], &it);
    while (roaring64_iterator_has_value(&it)) {
      uint64_t f = roaring64_iterator_value(&it);
      roaring64_bitmap_or_inplace(candidates, index[f]);
      roaring64_iterator_advance(&it);
    }

    // Get a sorted list of neighbors by distance
    roaring64_iterator_reinit(candidates, &it);
    while (roaring64_iterator_has_value(&it)) {
      int64_t v = (int64_t) roaring64_iterator_value(&it);
      roaring64_iterator_advance(&it);
      if (u == v)
        continue;
      tm_pair_t p = tm_pair(u, v);
      if (kh_get(pairs, pairs, p) != kh_end(pairs))
        continue;
      double dist = (double) roaring64_bitmap_xor_cardinality(sentences[u], sentences[v]) / (double) n_features;
      kv_push(tm_neighbor_t, nbrs, ((tm_neighbor_t) { .v = v, .d = dist }));
    }
    if (nbrs.n <= knn)
      ks_introsort(neighbors_asc, nbrs.n, nbrs.a);
    else {
      ks_ksmall(neighbors_asc, nbrs.n, nbrs.a, knn);
      ks_introsort(neighbors_asc, knn, nbrs.a);
      nbrs.n = knn;
      kv_resize(tm_neighbor_t, nbrs, nbrs.n);
    }

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
  (*global_iter) ++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushstring(L, "densify");
    lua_pushinteger(L, tm_dsu_components(dsu));
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_pushstring(L, "initial");
    lua_call(L, 6, 0);
  }

  // Prep shuffle
  int64_t *shuf = tk_malloc(L, dsu->n_original * sizeof(int64_t));
  for (int64_t i = 0; i < (int64_t) dsu->n_original; i ++)
    shuf[i] = i;
  ks_shuffle(i64, dsu->n_original, shuf);

  // Gather all inter-component edges
  tm_candidates_t all_candidates;
  kv_init(all_candidates);
  for (int64_t su = 0; su < (int64_t) dsu->n_original; su ++) {
    int64_t u = shuf[su];
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

  // Cleanup
  free(shuf);

  // Sort all by distance ascending (nearest in feature space)
  ks_introsort(candidates_asc, all_candidates.n, all_candidates.a);

  // Loop through candidates in order and add if they connect components
  for (uint64_t i = 0; i < all_candidates.n && dsu->n_components > 1; i ++) {
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
  (*global_iter) ++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushstring(L, "densify");
    lua_pushinteger(L, tm_dsu_components(dsu));
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_pushstring(L, "kruskal");
    lua_call(L, 6, 0);
  }

  // Connect unseen by neighbor first, if possible
  roaring64_iterator_t it;
  roaring64_iterator_reinit(dsu->unseen, &it);
  while (roaring64_iterator_has_value(&it)) {
    int64_t u = (int64_t) roaring64_iterator_value(&it);
    tm_neighbors_t nbrs = neighbors[u];
    bool added = false;
    for (uint64_t i = 0; i < nbrs.n; i ++) {
      tm_neighbor_t n = nbrs.a[i];
      if (u == n.v)
        continue;
      if (roaring64_bitmap_contains(dsu->unseen, (uint64_t) n.v))
        continue;
      tm_pair_t e = tm_pair(u, n.v);
      khi = kh_put(pairs, pairs, e, &kha);
      if (!kha)
        continue;
      kh_value(pairs, khi) = true;
      tm_dsu_union(L, dsu, u, n.v);
      (*n_pos) ++;
      added = true;
      break;
    }
    roaring64_iterator_advance(&it);
  }

  // If still unseen, connect via shuffle
  if (roaring64_bitmap_get_cardinality(dsu->unseen) > 0) {
    int64_t *shuf = tk_malloc(L, n_sentences * sizeof(int64_t));
    for (int64_t i = 0; i < (int64_t) n_sentences; i ++)
      shuf[i] = i;
    ks_shuffle(i64, n_sentences, shuf);
    // Select first suitable node
    uint64_t next = 0;
    uint64_t n_unseen = roaring64_bitmap_get_cardinality(dsu->unseen);
    uint64_t *unseen_array = tk_malloc(L, n_unseen * sizeof(uint64_t));
    roaring64_bitmap_to_uint64_array(dsu->unseen, unseen_array);
    for (uint64_t i = 0; i < n_unseen && next < n_sentences; i++) {
      int64_t u = (int64_t) unseen_array[i];
      for (; next < n_sentences; next ++) {
        int64_t v = shuf[next];
        if (u == v || roaring64_bitmap_contains(dsu->unseen, (uint64_t) v))
          continue;
        tm_pair_t e = tm_pair(u, v);
        khi = kh_put(pairs, pairs, e, &kha);
        if (!kha)
          continue;
        kh_value(pairs, khi) = true;
        tm_dsu_union(L, dsu, u, v);
        (*n_pos) ++;
        break;
      }
    }
    free(unseen_array);
    free(shuf);
  }

  // Log final density
  (*global_iter) ++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushstring(L, "densify");
    lua_pushinteger(L, tm_dsu_components(dsu));
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_pushstring(L, "fallback");
    lua_call(L, 6, 0);
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
  tm_dsu_t *dsu,
  tm_pairs_t *pairs,
  tm_neighbors_t *neighbors,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  roaring64_bitmap_t **sentences,
  uint64_t *n_pos,
  uint64_t *n_neg,
  uint64_t n_sentences,
  uint64_t n_features,
  uint64_t n_hops,
  uint64_t n_grow_pos,
  uint64_t n_grow_neg,
  int i_each,
  unsigned int *global_iter
) {
  int kha;
  khint_t khi;
  roaring64_iterator_t it0;
  roaring64_iterator_t it1;

  // Transitive positive expansion
  tm_candidates_t candidates;
  kv_init(candidates);
  for (int64_t u = 0; u < (int64_t) n_sentences; u ++) {
    // Init closure
    roaring64_bitmap_t *reachable = roaring64_bitmap_create();
    roaring64_bitmap_t *frontier = roaring64_bitmap_create();
    roaring64_bitmap_t *next = roaring64_bitmap_create();
    // Start from direct neighbors
    roaring64_bitmap_or_inplace(frontier, adj_pos[u]);
    roaring64_bitmap_or_inplace(reachable, frontier);
    for (uint64_t hop = 1; hop < n_hops; hop ++) {
      roaring64_bitmap_clear(next);
      roaring64_iterator_reinit(frontier, &it0);
      while (roaring64_iterator_has_value(&it0)) {
        uint64_t v = roaring64_iterator_value(&it0);
        roaring64_bitmap_or_inplace(next, adj_pos[v]);
        roaring64_iterator_advance(&it0);
      }
      roaring64_bitmap_or_inplace(reachable, next);
      roaring64_bitmap_clear(frontier); // advance
      roaring64_bitmap_or_inplace(frontier, next); // advance
    }
    // Filter out self, known links, and unseen
    roaring64_bitmap_remove(reachable, (uint64_t) u);
    roaring64_iterator_reinit(reachable, &it0);
    kv_size(candidates) = 0;
    while (roaring64_iterator_has_value(&it0)) {
      int64_t w = (int64_t) roaring64_iterator_value(&it0);
      roaring64_iterator_advance(&it0);
      if (roaring64_bitmap_contains(adj_pos[u], (uint64_t) w))
        continue;
      if (roaring64_bitmap_contains(adj_neg[u], (uint64_t) w))
        continue;
      double dist = (double) roaring64_bitmap_xor_cardinality(sentences[u], sentences[w]) / (double) n_features;
      tm_neighbor_t vw = (tm_neighbor_t){w, dist};
      kv_push(tm_candidate_t, candidates, tm_candidate(u, vw));
    }
    // Sort and add top-k
    ks_introsort(candidates_asc, candidates.n, candidates.a);
    for (uint64_t i = 0; i < candidates.n && i < n_grow_pos; i++) {
      tm_candidate_t c = candidates.a[i];
      tm_pair_t e = tm_pair(u, c.v.v);
      khi = kh_put(pairs, pairs, e, &kha);
      if (!kha)
        continue;
      kh_value(pairs, khi) = true;
      tm_dsu_union(L, dsu, u, c.v.v);
      roaring64_bitmap_add(adj_pos[u], (uint64_t)c.v.v);
      roaring64_bitmap_add(adj_pos[c.v.v], (uint64_t)u);
      (*n_pos)++;
    }
    // Cleanup
    roaring64_bitmap_free(reachable);
    roaring64_bitmap_free(frontier);
    roaring64_bitmap_free(next);
  }

  // Contrastive negative expansion
  for (int64_t u = 0; u < (int64_t) n_sentences; u ++) {
    roaring64_bitmap_t *seen = roaring64_bitmap_create();
    kv_size(candidates) = 0;
    // One-hop contrastive expansion from adj_pos[u] to adj_neg[v]
    roaring64_iterator_reinit(adj_pos[u], &it0);
    while (roaring64_iterator_has_value(&it0)) {
      int64_t v = (int64_t) roaring64_iterator_value(&it0);
      roaring64_iterator_advance(&it0);
      roaring64_iterator_reinit(adj_neg[v], &it1);
      while (roaring64_iterator_has_value(&it1)) {
        int64_t w = (int64_t) roaring64_iterator_value(&it1);
        roaring64_iterator_advance(&it1);
        if (w == u)
          continue;
        if (roaring64_bitmap_contains(adj_pos[u], (uint64_t) w))
          continue;
        if (roaring64_bitmap_contains(adj_neg[u], (uint64_t) w))
          continue;
        if (roaring64_bitmap_contains(seen, (uint64_t) w))
          continue;
        roaring64_bitmap_add(seen, (uint64_t) w);
        double dist = (double) roaring64_bitmap_xor_cardinality(sentences[u], sentences[w]) / (double) n_features;
        tm_neighbor_t nw = (tm_neighbor_t) { w, dist };
        kv_push(tm_candidate_t, candidates, tm_candidate(u, nw));
      }
    }
    // Select furthest in feature space
    ks_introsort(candidates_desc, candidates.n, candidates.a);  // sort by descending distance
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
    }
    roaring64_bitmap_free(seen);
  }

  // Cleanup
  kv_destroy(candidates);

  // Progress
  (*global_iter)++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushstring(L, "densify");
    lua_pushinteger(L, tm_dsu_components(dsu));
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_pushstring(L, "trans");
    lua_call(L, 6, 0);
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

static inline void tm_run_median_thresholding (
  lua_State *L,
  double *z,
  tk_bits_t *codes,
  uint64_t n_sentences,
  uint64_t n_hidden
) {
  double *col = tk_malloc(L, n_sentences * sizeof(double));

  for (uint64_t f = 0; f < n_hidden; f ++) {

    // Find the median
    for (uint64_t i = 0; i < n_sentences; i ++)
      col[i] = z[i * n_hidden + f];
    uint64_t mid = n_sentences / 2;
    ks_ksmall(f64, n_sentences, col, mid);
    double med = col[mid];

    // Apply
    for (uint64_t i = 0; i < n_sentences; i ++)
      if (z[i * n_hidden + f ] > med) {
        uint64_t chunk = BITS_DIV(f);
        uint64_t b = BITS_MOD(f);
        codes[i * BITS_DIV(n_hidden) + chunk] |= ((tk_bits_t) 1 << b);
      }

  }

  free(col);
}

static inline void tm_run_tch_thresholding (
  lua_State *L,
  double *z,
  double learnability,
  tk_bits_t *codes,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  uint64_t n_sentences,
  uint64_t n_hidden,
  int i_each,
  unsigned int *global_iter
) {
  int *bitvec = tk_malloc(L, n_sentences * sizeof(int));
  double *col = tk_malloc(L, n_sentences * sizeof(double));
  uint64_t total_steps = 0;

  // Prep hidden shuffle
  int64_t *shuf_hidden = tk_malloc(L, n_hidden * sizeof(int64_t));
  for (int64_t i = 0; i < (int64_t) n_hidden; i ++)
    shuf_hidden[i] = i;
  ks_shuffle(i64, n_hidden, shuf_hidden);

  // Prep sentences shuffle
  int64_t *shuf_sentences = tk_malloc(L, n_sentences * sizeof(int64_t));
  for (int64_t i = 0; i < (int64_t) n_sentences; i ++)
    shuf_sentences[i] = i;
  ks_shuffle(i64, n_sentences, shuf_sentences);

  roaring64_iterator_t it;
  for (uint64_t sf = 0; sf < n_hidden; sf ++) {
    uint64_t f = (uint64_t) shuf_hidden[sf];

    // Find the median
    for (uint64_t i = 0; i < n_sentences; i ++)
      col[i] = z[i * n_hidden + f];
    uint64_t mid = n_sentences / 2;
    ks_ksmall(f64, n_sentences, col, mid);
    double med = col[mid];

    // Threshold around the median
    for (uint64_t i = 0; i < n_sentences; i++)
      bitvec[i] = (z[i * n_hidden + f] > med) ? +1 : -1;

    bool updated;
    uint64_t steps = 0;

    ks_shuffle(i64, n_sentences, shuf_sentences);

    do {
      updated = false;
      steps ++;
      total_steps ++;

      for (uint64_t si = 0; si < n_sentences; si ++) {
        uint64_t i = (uint64_t) shuf_sentences[si];
        int delta = 0;
        // Positive neighbors
        roaring64_iterator_reinit(adj_pos[i], &it);
        while (roaring64_iterator_has_value(&it)) {
          uint64_t j = roaring64_iterator_value(&it);
          roaring64_iterator_advance(&it);
          delta += bitvec[i] * bitvec[j];
        }
        // Negative neighbors
        roaring64_iterator_reinit(adj_neg[i], &it);
        while (roaring64_iterator_has_value(&it)) {
          uint64_t j = roaring64_iterator_value(&it);
          roaring64_iterator_advance(&it);
          delta -= bitvec[i] * bitvec[j];
        }
        // Check
        if (delta < 0){
          bitvec[i] = -bitvec[i];
          updated = true;
        }
      }

    } while (updated);

    // Write out the final bits into your packed codes
    for (uint64_t i = 0; i < n_sentences; i ++)
      if (bitvec[i] > 0){
        uint64_t chunk = BITS_DIV(f);
        uint64_t b = BITS_MOD(f);
        codes[i * BITS_DIV(n_hidden) + chunk] |= ((tk_bits_t) 1 << b);
      }
  }

  (*global_iter) ++;
  if (i_each >= 0) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushstring(L, "tch");
    lua_pushinteger(L, (int64_t) total_steps);
    lua_call(L, 3, 0);
  }

  // Cleanup
  free(bitvec);
  free(col);
}

static inline void tm_run_spectral (
  lua_State *L,
  tm_dsu_t *dsu,
  double *z,
  roaring64_bitmap_t **adj_pos,
  roaring64_bitmap_t **adj_neg,
  uint64_t n_sentences,
  uint64_t n_hidden,
  int i_each,
  unsigned int *global_iter
) {
  // Build degree vector for Laplacian
  unsigned int *degree = tk_malloc(L, n_sentences * sizeof(unsigned int));
  for (uint64_t i = 0; i < n_sentences; i ++) {
    degree[i] =
      (unsigned int) roaring64_bitmap_get_cardinality(adj_pos[i]) +
      (unsigned int) roaring64_bitmap_get_cardinality(adj_neg[i]);
  }

  // Build inv_sqrt vector
  double *inv_sqrt_deg = tk_malloc(L, n_sentences * sizeof(double));
  for (uint64_t i = 0; i < n_sentences; i ++)
    inv_sqrt_deg[i] = degree[i] > 0 ? 1.0 / sqrt((double) degree[i]) : 0.0;
  free(degree);

  // ARPACK setup
  a_int ido = 0, info = 0;
  char bmat[] = "I", which[] = "SM";
  a_int nev = n_hidden + 1;
  a_int ncv = (4 * nev < (a_int) n_sentences) ? 4 * nev : (a_int) n_sentences;
  double tol = 1e-3;
  double *resid = tk_malloc(L, (size_t) n_sentences * sizeof(double));
  double *workd = tk_malloc(L, 3 * (size_t) n_sentences * sizeof(double));
  a_int iparam[11] = {1,0,999999,0,0,0,1,0,0,0,0};
  a_int ipntr[14] = {0};
  int lworkl = ncv * (ncv + 8);
  double *workl = tk_malloc(L, (size_t) lworkl * sizeof(double));
  double *v = tk_malloc(L, (size_t) n_sentences * (size_t) ncv * sizeof(double));

  // Reverse-communication to build Lanczos basis v
  roaring64_iterator_t it;
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
      for (uint64_t u = 0; u < n_sentences; u ++) {
        double scale_u = inv_sqrt_deg[u];
        // Positive neighbors
        roaring64_iterator_reinit(adj_pos[u], &it);
        while (roaring64_iterator_has_value(&it)) {
          uint64_t v = roaring64_iterator_value(&it);
          roaring64_iterator_advance(&it);
          double w = scale_u * inv_sqrt_deg[v];
          out[u] -= w * in[v];
          out[v] -= w * in[u];
        }
        // Negative neighbors
        roaring64_iterator_reinit(adj_neg[u], &it);
        while (roaring64_iterator_has_value(&it)) {
          uint64_t v = roaring64_iterator_value(&it);
          roaring64_iterator_advance(&it);
          double w = scale_u * inv_sqrt_deg[v];
          out[u] += w * in[v];
          out[v] += w * in[u];
        }
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

static inline void tm_init_random_z (
  lua_State *L,
  double *z,
  uint64_t n_sentences,
  uint64_t n_hidden
) {
  for (uint64_t i = 0; i < n_sentences * n_hidden; i ++)
    z[i] = fast_norm(0, 1);
}

static inline int tm_codeify (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "pos");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pos_seed = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pos_seed = (uint64_t) luaL_checkinteger(L, -1) / 2;
  uint64_t n_pos = n_pos_seed;

  lua_getfield(L, 1, "neg");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *neg_seed = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_neg_seed = (uint64_t) luaL_checkinteger(L, -1) / 2;
  uint64_t n_neg = n_neg_seed;

  lua_getfield(L, 1, "sentences");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  int64_t *set_bits = (int64_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_set_bits = (uint64_t) luaL_checkinteger(L, -1);

  uint64_t n_sentences = tk_lua_fcheckunsigned(L, 1, "codeify", "n_sentences");
  uint64_t n_features = tk_lua_fcheckunsigned(L, 1, "codeify", "n_features");
  uint64_t n_hidden = tk_lua_fcheckunsigned(L, 1, "codeify", "n_hidden");

  uint64_t n_hops = tk_lua_fcheckunsigned(L, 1, "codeify", "n_hops");
  uint64_t n_grow_pos = tk_lua_fcheckunsigned(L, 1, "codeify", "n_grow_pos");
  uint64_t n_grow_neg = tk_lua_fcheckunsigned(L, 1, "codeify", "n_grow_neg");

  uint64_t knn = tk_lua_fcheckunsigned(L, 1, "codeify", "knn");
  double learnability = tk_lua_fcheckposdouble(L, 1, "codeify", "learnability");

  bool do_spectral = tk_lua_foptboolean(L, 1, true, "codeify", "spectral");
  bool do_tch = tk_lua_foptboolean(L, 1, true, "codeify", "tch");

  if (BITS_MOD(n_hidden) != 0)
    tk_lua_verror(L, 3, "codify", "n_hidden", "must be a multiple of " STR(BITS));

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tm_pairs_t *pairs = kh_init(pairs);
  tm_pairs_init(L, pairs, pos_seed, neg_seed, &n_pos, &n_neg);

  roaring64_bitmap_t **sentences = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  tm_sentences_init(sentences, set_bits, n_set_bits, n_features, n_sentences);

  roaring64_bitmap_t **index = tk_malloc(L, n_features * sizeof(roaring64_bitmap_t *));
  tm_index_init(L, index, set_bits, n_set_bits, n_features);

  tm_neighbors_t *neighbors = tk_malloc(L, n_sentences * sizeof(tm_neighbors_t));
  tm_neighbors_init(L, pairs, neighbors, index, sentences, n_sentences, n_features, knn);

  // Cleanup
  for (uint64_t f = 0; f < n_features; f ++)
    roaring64_bitmap_free(index[f]);
  free(index);

  tm_dsu_t dsu;
  tm_dsu_init(L, &dsu, pairs, n_sentences);

  unsigned int global_iter = 0;

  // Densify graph
  tm_add_backbone(L, &dsu, pairs, neighbors, sentences, &n_pos, &n_neg, n_sentences, n_features, i_each, &global_iter);

  // Setup adjacency list
  roaring64_bitmap_t **adj_pos = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  roaring64_bitmap_t **adj_neg = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  tm_adj_init(pairs, adj_pos, adj_neg, n_sentences);

  // Densify graph, adding transatives
  tm_add_transatives(L, &dsu, pairs, neighbors, adj_pos, adj_neg, sentences, &n_pos, &n_neg, n_sentences, n_features, n_hops, n_grow_pos, n_grow_neg, i_each, &global_iter);

  // Cleanup
  for (int64_t s = 0; s < (int64_t) n_sentences; s ++)
    roaring64_bitmap_free(sentences[s]);
  free(sentences);

  // Spectral hashing
  double *z = tk_malloc(L, (size_t) n_sentences * (size_t) n_hidden * sizeof(double));
  if (do_spectral)
    tm_run_spectral(L, &dsu, z, adj_pos, adj_neg, n_sentences, n_hidden, i_each, &global_iter);
  else
    tm_init_random_z(L, z, n_sentences, n_hidden);

  // Cleanup
  tm_dsu_free(&dsu);

  // Done with pair lists
  kh_destroy(pairs, pairs);

  // TCH discretization
  tk_bits_t *codes = tk_malloc(L, (size_t) n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t));
  memset(codes, 0, (size_t) n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t));
  if (do_tch)
    tm_run_tch_thresholding(L, z, learnability, codes, adj_pos, adj_neg, n_sentences, n_hidden, i_each, &global_iter);
  else
    tm_run_median_thresholding(L, z, codes, n_sentences, n_hidden);

  // Copy codes to Lualand
  lua_pushlstring(L, (char *) codes, n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t));

  // Cleanup
  for (uint64_t u = 0; u < n_sentences; u++) {
    roaring64_bitmap_free(adj_pos[u]);
    roaring64_bitmap_free(adj_neg[u]);
  }
  free(adj_pos);
  free(adj_neg);
  free(z);
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
