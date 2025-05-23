#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/pairs.h>

#include <float.h>
#include <lauxlib.h>
#include <lua.h>

#include "khash.h"
KHASH_SET_INIT_INT64(i64)
typedef khash_t(i64) i64_hash_t;

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
  #pragma omp parallel
  {
    // Each thread gets its own bitmap and iterator
    roaring64_iterator_t it;
    roaring64_bitmap_t *candidates = roaring64_bitmap_create();
    #pragma omp for schedule(dynamic)
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
    // Cleanup per-thread
    roaring64_bitmap_free(candidates);
  }
}

static inline void tm_add_knn (
  lua_State *L,
  tm_dsu_t *dsu,
  tm_pairs_t *pairs,
  tm_neighbors_t *neighbors,
  roaring64_bitmap_t **sentences,
  uint64_t *n_pos,
  uint64_t *n_neg,
  uint64_t n_sentences,
  uint64_t n_features,
  uint64_t knn,
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
    lua_pushinteger(L, tm_dsu_components(dsu));
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_pushstring(L, "initial");
    lua_call(L, 5, 0);
  }

  // Prep shuffle
  int64_t *shuf = tk_malloc(L, (uint64_t) n_sentences * sizeof(int64_t));
  for (int64_t i = 0; i < (int64_t) n_sentences; i ++)
    shuf[i] = i;
  ks_shuffle(i64, n_sentences, shuf);

  // Gather all inter-component edges
  for (int64_t su = 0; su < (int64_t) n_sentences; su ++) {
    int64_t u = shuf[su];
    tm_neighbors_t nbrs = neighbors[u];
    for (khint_t i = 0; i < nbrs.n && i < knn; i ++) {
      tm_neighbor_t v = nbrs.a[i];
      tm_pair_t e = tm_pair(u, v.v);
      khi = kh_put(pairs, pairs, e, &kha);
      if (!kha)
        continue;
      kh_value(pairs, khi) = true;
      tm_dsu_union(L, dsu, u, v.v);
      (*n_pos)++;
    }
  }

  // Cleanup
  free(shuf);

  // Log final density
  (*global_iter) ++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, *global_iter);
    lua_pushinteger(L, tm_dsu_components(dsu));
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_pushstring(L, "knn");
    lua_call(L, 5, 0);
  }
}

static inline void tm_add_mst (
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
    lua_pushinteger(L, tm_dsu_components(dsu));
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_pushstring(L, "initial");
    lua_call(L, 5, 0);
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
    lua_pushinteger(L, tm_dsu_components(dsu));
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_pushstring(L, "kruskal");
    lua_call(L, 5, 0);
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
    lua_pushinteger(L, tm_dsu_components(dsu));
    lua_pushinteger(L, (int64_t) *n_pos);
    lua_pushinteger(L, (int64_t) *n_neg);
    lua_pushstring(L, "fallback");
    lua_call(L, 5, 0);
  }

  // Cleanup
  kv_destroy(all_candidates);
}

// static inline void tm_add_transatives (
//   lua_State *L,
//   tm_dsu_t *dsu,
//   tm_pairs_t *pairs,
//   tm_neighbors_t *neighbors,
//   roaring64_bitmap_t **adj_pos,
//   roaring64_bitmap_t **adj_neg,
//   roaring64_bitmap_t **sentences,
//   uint64_t *n_pos,
//   uint64_t *n_neg,
//   uint64_t n_sentences,
//   uint64_t n_features,
//   uint64_t n_hops,
//   uint64_t n_grow_pos,
//   uint64_t n_grow_neg,
//   int i_each,
//   unsigned int *global_iter
// ) {
//   int kha;
//   khint_t khi;
//   roaring64_iterator_t it0;
//   roaring64_iterator_t it1;

//   // Transitive positive expansion
//   tm_candidates_t candidates;
//   kv_init(candidates);
//   for (int64_t u = 0; u < (int64_t) n_sentences; u ++) {
//     // Init closure
//     roaring64_bitmap_t *reachable = roaring64_bitmap_create();
//     roaring64_bitmap_t *frontier = roaring64_bitmap_create();
//     roaring64_bitmap_t *next = roaring64_bitmap_create();
//     // Start from direct neighbors
//     roaring64_bitmap_or_inplace(frontier, adj_pos[u]);
//     roaring64_bitmap_or_inplace(reachable, frontier);
//     for (uint64_t hop = 1; hop < n_hops; hop ++) {
//       roaring64_bitmap_clear(next);
//       roaring64_iterator_reinit(frontier, &it0);
//       while (roaring64_iterator_has_value(&it0)) {
//         uint64_t v = roaring64_iterator_value(&it0);
//         roaring64_bitmap_or_inplace(next, adj_pos[v]);
//         roaring64_iterator_advance(&it0);
//       }
//       roaring64_bitmap_or_inplace(reachable, next);
//       roaring64_bitmap_clear(frontier); // advance
//       roaring64_bitmap_or_inplace(frontier, next); // advance
//     }
//     // Filter out self, known links, and unseen
//     roaring64_bitmap_remove(reachable, (uint64_t) u);
//     roaring64_iterator_reinit(reachable, &it0);
//     kv_size(candidates) = 0;
//     while (roaring64_iterator_has_value(&it0)) {
//       int64_t w = (int64_t) roaring64_iterator_value(&it0);
//       roaring64_iterator_advance(&it0);
//       if (roaring64_bitmap_contains(adj_pos[u], (uint64_t) w))
//         continue;
//       if (roaring64_bitmap_contains(adj_neg[u], (uint64_t) w))
//         continue;
//       double dist = (double) roaring64_bitmap_xor_cardinality(sentences[u], sentences[w]) / (double) n_features;
//       tm_neighbor_t vw = (tm_neighbor_t){w, dist};
//       kv_push(tm_candidate_t, candidates, tm_candidate(u, vw));
//     }
//     // Sort and add top-k
//     ks_introsort(candidates_asc, candidates.n, candidates.a);
//     for (uint64_t i = 0; i < candidates.n && i < n_grow_pos; i++) {
//       tm_candidate_t c = candidates.a[i];
//       tm_pair_t e = tm_pair(u, c.v.v);
//       khi = kh_put(pairs, pairs, e, &kha);
//       if (!kha)
//         continue;
//       kh_value(pairs, khi) = true;
//       tm_dsu_union(L, dsu, u, c.v.v);
//       roaring64_bitmap_add(adj_pos[u], (uint64_t)c.v.v);
//       roaring64_bitmap_add(adj_pos[c.v.v], (uint64_t)u);
//       (*n_pos)++;
//     }
//     // Cleanup
//     roaring64_bitmap_free(reachable);
//     roaring64_bitmap_free(frontier);
//     roaring64_bitmap_free(next);
//   }

//   // Contrastive negative expansion
//   for (int64_t u = 0; u < (int64_t) n_sentences; u ++) {
//     roaring64_bitmap_t *seen = roaring64_bitmap_create();
//     kv_size(candidates) = 0;
//     // One-hop contrastive expansion from adj_pos[u] to adj_neg[v]
//     roaring64_iterator_reinit(adj_pos[u], &it0);
//     while (roaring64_iterator_has_value(&it0)) {
//       int64_t v = (int64_t) roaring64_iterator_value(&it0);
//       roaring64_iterator_advance(&it0);
//       roaring64_iterator_reinit(adj_neg[v], &it1);
//       while (roaring64_iterator_has_value(&it1)) {
//         int64_t w = (int64_t) roaring64_iterator_value(&it1);
//         roaring64_iterator_advance(&it1);
//         if (w == u)
//           continue;
//         if (roaring64_bitmap_contains(adj_pos[u], (uint64_t) w))
//           continue;
//         if (roaring64_bitmap_contains(adj_neg[u], (uint64_t) w))
//           continue;
//         if (roaring64_bitmap_contains(seen, (uint64_t) w))
//           continue;
//         roaring64_bitmap_add(seen, (uint64_t) w);
//         double dist = (double) roaring64_bitmap_xor_cardinality(sentences[u], sentences[w]) / (double) n_features;
//         tm_neighbor_t nw = (tm_neighbor_t) { w, dist };
//         kv_push(tm_candidate_t, candidates, tm_candidate(u, nw));
//       }
//     }
//     // Select furthest in feature space
//     ks_introsort(candidates_desc, candidates.n, candidates.a);  // sort by descending distance
//     for (uint64_t i = 0; i < candidates.n && i < n_grow_neg; i ++) {
//       tm_candidate_t c = candidates.a[i];
//       tm_pair_t e = tm_pair(u, c.v.v);
//       khi = kh_put(pairs, pairs, e, &kha);
//       if (!kha)
//         continue;
//       kh_value(pairs, khi) = false;
//       roaring64_bitmap_add(adj_neg[u], (uint64_t) c.v.v);
//       roaring64_bitmap_add(adj_neg[c.v.v], (uint64_t) u);
//       (*n_neg) ++;
//     }
//     roaring64_bitmap_free(seen);
//   }

//   // Cleanup
//   kv_destroy(candidates);

//   // Progress
//   (*global_iter)++;
//   if (i_each != -1) {
//     lua_pushvalue(L, i_each);
//     lua_pushinteger(L, *global_iter);
//     lua_pushinteger(L, tm_dsu_components(dsu));
//     lua_pushinteger(L, (int64_t) *n_pos);
//     lua_pushinteger(L, (int64_t) *n_neg);
//     lua_pushstring(L, "transitives");
//     lua_call(L, 5, 0);
//   }
// }

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

static inline int tm_render (lua_State *L)
{
  lua_settop(L, 2);
  lua_pushvalue(L, 1);
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pairs = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pairs = (uint64_t) luaL_checkinteger(L, -1) / 2;
  uint64_t n_sentences = tk_lua_checkunsigned(L, 2, "n_sentences");

  kvec_t(int64_t) out;
  kv_init(out);

  for (uint64_t i = 0; i < n_pairs; i ++) {
    tm_pair_t p = pairs[i];
    int64_t u = p.u, v = p.v;
    kv_push(int64_t, out, u * (int64_t) n_sentences + v);
    kv_push(int64_t, out, v * (int64_t) n_sentences + u);
  }

  kv_resize(int64_t, out, out.n);
  lua_pushlightuserdata(L, out.a);
  lua_pushinteger(L, 1);
  lua_pushinteger(L, (int64_t) out.n);
  tk_lua_callmod(L, 3, 1, "santoku.matrix.integer", "from_view");
  return 1;
}

static inline int tm_enrich (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "pos");
  lua_pushboolean(L, true);
  tk_lua_callmod(L, 2, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pos_seed = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pos_seed = (uint64_t) luaL_checkinteger(L, -1) / 2;
  uint64_t n_pos = n_pos_seed;

  lua_getfield(L, 1, "neg");
  lua_pushboolean(L, true);
  tk_lua_callmod(L, 2, 4, "santoku.matrix.integer", "view");
  tm_pair_t *neg_seed = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_neg_seed = (uint64_t) luaL_checkinteger(L, -1) / 2;
  uint64_t n_neg = n_neg_seed;

  lua_getfield(L, 1, "sentences");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  int64_t *set_bits = (int64_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_set_bits = (uint64_t) luaL_checkinteger(L, -1);

  uint64_t n_sentences = tk_lua_fcheckunsigned(L, 1, "enrich", "n_sentences");
  uint64_t n_features = tk_lua_fcheckunsigned(L, 1, "enrich", "n_features");
  uint64_t knn_cache = tk_lua_foptunsigned(L, 1, "enrich", "knn_cache", 32);
  uint64_t knn = tk_lua_foptunsigned(L, 1, "enrich", "knn", 0);
  bool do_mst = tk_lua_foptboolean(L, 1, "enrich", "mst", true);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  // Setup pairs & adjacency lists
  tm_pairs_t *pairs = kh_init(pairs);
  roaring64_bitmap_t **adj_pos = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  roaring64_bitmap_t **adj_neg = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  tm_pairs_init(L, pairs, pos_seed, neg_seed, &n_pos, &n_neg);
  tm_adj_init(pairs, adj_pos, adj_neg, n_sentences);

  roaring64_bitmap_t **sentences = tk_malloc(L, n_sentences * sizeof(roaring64_bitmap_t *));
  tm_sentences_init(sentences, set_bits, n_set_bits, n_features, n_sentences);

  roaring64_bitmap_t **index = tk_malloc(L, n_features * sizeof(roaring64_bitmap_t *));
  tm_index_init(L, index, set_bits, n_set_bits, n_features);

  tm_neighbors_t *neighbors = tk_malloc(L, n_sentences * sizeof(tm_neighbors_t));
  tm_neighbors_init(L, pairs, neighbors, index, sentences, n_sentences, n_features, knn_cache);

  // Cleanup
  for (uint64_t f = 0; f < n_features; f ++)
    roaring64_bitmap_free(index[f]);
  free(index);

  tm_dsu_t dsu;
  tm_dsu_init(L, &dsu, pairs, n_sentences);

  unsigned int global_iter = 0;

  // Add knn
  if (knn > 0)
    tm_add_knn(L, &dsu, pairs, neighbors, sentences, &n_pos, &n_neg, n_sentences, n_features, knn, i_each, &global_iter);

  // Add mst
  if (do_mst)
    tm_add_mst(L, &dsu, pairs, neighbors, sentences, &n_pos, &n_neg, n_sentences, n_features, i_each, &global_iter);

  // // Densify graph, adding transatives
  // if (do_transitives)
  //   tm_add_transatives(L, &dsu, pairs, neighbors, adj_pos, adj_neg, sentences, &n_pos, &n_neg, n_sentences, n_features, n_hops, n_grow_pos, n_grow_neg, i_each, &global_iter);

  // Cleanup
  for (int64_t s = 0; s < (int64_t) n_sentences; s ++)
    roaring64_bitmap_free(sentences[s]);
  free(sentences);

  // Cleanup
  tm_dsu_free(&dsu);

  // Render pairs
  pos_seed = tk_realloc(L, pos_seed, n_pos * sizeof(tm_pair_t));
  neg_seed = tk_realloc(L, neg_seed, n_neg * sizeof(tm_pair_t));
  tm_render_pairs(L, pairs, pos_seed, neg_seed, &n_pos, &n_neg);

  // Push updated positives
  lua_pushlightuserdata(L, pos_seed);
  lua_pushinteger(L, (int64_t) n_pos);
  lua_pushinteger(L, 2);
  lua_getfield(L, 1, "pos");
  tk_lua_callmod(L, 4, 1, "santoku.matrix.integer", "from_view");

  // Push updated negatives
  lua_pushlightuserdata(L, neg_seed);
  lua_pushinteger(L, (int64_t) n_neg);
  lua_pushinteger(L, 2);
  lua_getfield(L, 1, "neg");
  tk_lua_callmod(L, 4, 1, "santoku.matrix.integer", "from_view");

  // Cleanup
  kh_destroy(pairs, pairs);
  for (uint64_t u = 0; u < n_sentences; u ++) {
    roaring64_bitmap_free(adj_pos[u]);
    roaring64_bitmap_free(adj_neg[u]);
  }
  free(adj_pos);
  free(adj_neg);

  // Update pairs
  return 2;
}

static luaL_Reg tm_codebook_fns[] =
{
  { "enrich", tm_enrich },
  { "render", tm_render },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_graph (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_codebook_fns, 0); // t
  return 1;
}
