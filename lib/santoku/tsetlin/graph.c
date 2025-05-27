#define _GNU_SOURCE

#include <santoku/tsetlin/graph.h>

static inline void tk_graph_worker (void *dp, int sig)
{
  tk_graph_stage_t stage = (tk_graph_stage_t) sig;
  tk_graph_thread_t *data = (tk_graph_thread_t *) dp;
  roaring64_bitmap_t *candidates = data->candidates;
  roaring64_bitmap_t **nodes = data->graph->nodes;
  tm_pairs_t *pairs = data->graph->pairs;
  tm_neighbors_t *neighbors = data->graph->neighbors;
  roaring64_bitmap_t **index = data->graph->index;
  uint64_t n_features = data->graph->n_features;
  uint64_t knn_cache = data->graph->knn_cache;
  roaring64_iterator_t it;
  switch (stage) {
    case TK_GRAPH_KNN:
      for (int64_t u = (int64_t) data->ufirst; u <= (int64_t) data->ulast; u ++) {
        tm_neighbors_t nbrs;
        kv_init(nbrs);
        // Populate candidate set (those sharing at least one feature)
        roaring64_bitmap_clear(candidates);
        roaring64_iterator_reinit(nodes[u], &it);
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
          double dist = (double) roaring64_bitmap_xor_cardinality(nodes[u], nodes[v]) / (double) n_features;
          kv_push(tm_neighbor_t, nbrs, ((tm_neighbor_t) { .v = v, .d = dist }));
        }
        if (nbrs.n <= knn_cache)
          ks_introsort(neighbors_asc, nbrs.n, nbrs.a);
        else {
          ks_ksmall(neighbors_asc, nbrs.n, nbrs.a, knn_cache);
          ks_introsort(neighbors_asc, knn_cache, nbrs.a);
          nbrs.n = knn_cache;
          kv_resize(tm_neighbor_t, nbrs, nbrs.n);
        }
        // Update refs
        neighbors[u] = nbrs;
      }
      break;
  }
}

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
  tk_graph_t *graph,
  tm_pair_t *pos,
  tm_pair_t *neg,
  uint64_t *n_pos,
  uint64_t *n_neg
) {
  bool l;
  tm_pair_t p;
  uint64_t wp = 0, wn = 0;
  kh_foreach(graph->pairs, p,  l, ({
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

static inline void tm_nodes_init (
  roaring64_bitmap_t **nodes,
  int64_t *set_bits,
  uint64_t n_set_bits,
  uint64_t n_features,
  uint64_t n_nodes
) {
  for (uint64_t b = 0; b < n_set_bits; b ++) {
    int64_t v = set_bits[b];
    int64_t s = v / (int64_t) n_features;
    int64_t f = v % (int64_t) n_features;
    roaring64_bitmap_add(nodes[s], (uint64_t) f);
  }
}

static inline void tm_index_neighbors (
  lua_State *L,
  tk_graph_t *graph
) {
  graph->index = tk_malloc(L, graph->n_features * sizeof(roaring64_bitmap_t *));
  tm_nodes_init(graph->nodes, graph->set_bits, graph->n_set_bits, graph->n_features, graph->n_nodes);
  tm_index_init(L, graph->index, graph->set_bits, graph->n_set_bits, graph->n_features);
  for (unsigned int i = 0; i < graph->pool->n_threads; i ++) {
    tk_graph_thread_t *data = (tk_graph_thread_t *) graph->pool->threads[i].data;
    tk_thread_range(i, graph->pool->n_threads, graph->n_nodes, &data->ufirst, &data->ulast);
  }

  // Compute nearest neighbors
  tk_threads_signal(graph->pool, TK_GRAPH_KNN);

  // Cleanup
  for (uint64_t f = 0; f < graph->n_features; f ++)
    roaring64_bitmap_free(graph->index[f]);
  free(graph->index);
}

static inline void tm_add_knn (
  lua_State *L,
  tk_graph_t *graph,
  uint64_t knn
) {
  int kha;
  khint_t khi;

  // Prep shuffle
  int64_t *shuf = tk_malloc(L, (uint64_t) graph->n_nodes * sizeof(int64_t));
  for (int64_t i = 0; i < (int64_t) graph->n_nodes; i ++)
    shuf[i] = i;
  ks_shuffle(i64, graph->n_nodes, shuf);

  // Add neighbors
  for (int64_t su = 0; su < (int64_t) graph->n_nodes; su ++) {
    int64_t u = shuf[su];
    tm_neighbors_t nbrs = graph->neighbors[u];
    uint64_t added = 0;
    for (khint_t i = 0; i < nbrs.n && added < knn; i ++) {
      tm_neighbor_t v = nbrs.a[i];
      tm_pair_t e = tm_pair(u, v.v);
      khi = kh_put(pairs, graph->pairs, e, &kha);
      if (!kha)
        continue;
      added ++;
      kh_value(graph->pairs, khi) = true;
      roaring64_bitmap_add(graph->adj_pos[u], (uint64_t) v.v);
      roaring64_bitmap_add(graph->adj_pos[v.v], (uint64_t) u);
      tm_dsu_union(L, &graph->dsu, u, v.v);
      graph->n_pos ++;
    }
  }

  // Cleanup
  free(shuf);
}

static inline void tm_add_mst (
  lua_State *L,
  tk_graph_t *graph
) {
  int kha;
  khint_t khi;

  // Prep shuffle
  int64_t *shuf = tk_malloc(L, graph->dsu.n_original * sizeof(int64_t));
  for (int64_t i = 0; i < (int64_t) graph->dsu.n_original; i ++)
    shuf[i] = i;
  ks_shuffle(i64, graph->dsu.n_original, shuf);

  // Gather all inter-component edges
  tm_candidates_t all_candidates;
  kv_init(all_candidates);
  for (int64_t su = 0; su < (int64_t) graph->dsu.n_original; su ++) {
    int64_t u = shuf[su];
    int64_t cu = tm_dsu_find(&graph->dsu, u);
    tm_neighbors_t nbrs = graph->neighbors[u];
    for (khint_t i = 0; i < nbrs.n; i ++) {
      tm_neighbor_t v = nbrs.a[i];
      if (cu == tm_dsu_find(&graph->dsu, v.v))
        continue;
      tm_pair_t e = tm_pair(u, v.v);
      khi = kh_get(pairs, graph->pairs, e);
      if (khi != kh_end(graph->pairs))
        continue;
      kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v));
    }
  }

  // Cleanup
  free(shuf);

  // Sort all by distance ascending (nearest in feature space)
  ks_introsort(candidates_asc, all_candidates.n, all_candidates.a);

  // Loop through candidates in order and add if they connect components
  for (uint64_t i = 0; i < all_candidates.n && graph->dsu.n_components > 1; i ++) {
    tm_candidate_t c = all_candidates.a[i];
    int64_t cu = tm_dsu_find(&graph->dsu, c.u);
    int64_t cv = tm_dsu_find(&graph->dsu, c.v.v);
    if (cu == cv)
      continue;
    tm_pair_t e = tm_pair(c.u, c.v.v);
    khi = kh_put(pairs, graph->pairs, e, &kha);
    if (!kha)
      continue;
    kh_value(graph->pairs, khi) = true;
    roaring64_bitmap_add(graph->adj_pos[c.u], (uint64_t) c.v.v);
    roaring64_bitmap_add(graph->adj_pos[c.v.v], (uint64_t) c.u);
    tm_dsu_union(L, &graph->dsu, c.u, c.v.v);
    graph->n_pos ++;
  }

  // Connect unseen by neighbor first, if possible
  roaring64_iterator_t it;
  roaring64_iterator_reinit(graph->dsu.unseen, &it);
  while (roaring64_iterator_has_value(&it)) {
    int64_t u = (int64_t) roaring64_iterator_value(&it);
    tm_neighbors_t nbrs = graph->neighbors[u];
    bool added = false;
    for (uint64_t i = 0; i < nbrs.n; i ++) {
      tm_neighbor_t n = nbrs.a[i];
      if (u == n.v)
        continue;
      if (roaring64_bitmap_contains(graph->dsu.unseen, (uint64_t) n.v))
        continue;
      tm_pair_t e = tm_pair(u, n.v);
      khi = kh_put(pairs, graph->pairs, e, &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = true;
      roaring64_bitmap_add(graph->adj_pos[u], (uint64_t) n.v);
      roaring64_bitmap_add(graph->adj_pos[n.v], (uint64_t) u);
      tm_dsu_union(L, &graph->dsu, u, n.v);
      graph->n_pos ++;
      added = true;
      break;
    }
    roaring64_iterator_advance(&it);
  }

  // If still unseen, connect via shuffle
  if (roaring64_bitmap_get_cardinality(graph->dsu.unseen) > 0) {
    int64_t *shuf = tk_malloc(L, graph->n_nodes * sizeof(int64_t));
    for (int64_t i = 0; i < (int64_t) graph->n_nodes; i ++)
      shuf[i] = i;
    ks_shuffle(i64, graph->n_nodes, shuf);
    // Select first suitable node
    uint64_t next = 0;
    uint64_t n_unseen = roaring64_bitmap_get_cardinality(graph->dsu.unseen);
    uint64_t *unseen_array = tk_malloc(L, n_unseen * sizeof(uint64_t));
    roaring64_bitmap_to_uint64_array(graph->dsu.unseen, unseen_array);
    for (uint64_t i = 0; i < n_unseen && next < graph->n_nodes; i++) {
      int64_t u = (int64_t) unseen_array[i];
      for (; next < graph->n_nodes; next ++) {
        int64_t v = shuf[next];
        if (u == v || roaring64_bitmap_contains(graph->dsu.unseen, (uint64_t) v))
          continue;
        tm_pair_t e = tm_pair(u, v);
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        kh_value(graph->pairs, khi) = true;
        roaring64_bitmap_add(graph->adj_pos[u], (uint64_t) v);
        roaring64_bitmap_add(graph->adj_pos[v], (uint64_t) u);
        tm_dsu_union(L, &graph->dsu, u, v);
        graph->n_pos ++;
        break;
      }
    }
    free(unseen_array);
    free(shuf);
  }

  // Cleanup
  kv_destroy(all_candidates);
}

// TODO: Parallelize
static inline void tm_add_transatives (
  lua_State *L,
  tk_graph_t *graph,
  uint64_t n_hops,
  uint64_t n_grow_pos,
  uint64_t n_grow_neg
) {
  int kha;
  khint_t khi;
  roaring64_iterator_t it0;
  roaring64_iterator_t it1;

  kvec_t(tm_pair_t) new_pos;
  kv_init(new_pos);

  kvec_t(tm_pair_t) new_neg;
  kv_init(new_neg);

  // Transitive positive expansion
  tm_candidates_t candidates;
  kv_init(candidates);
  for (int64_t u = 0; u < (int64_t) graph->n_nodes; u ++) {
    // Init closure
    roaring64_bitmap_t *reachable = roaring64_bitmap_create();
    roaring64_bitmap_t *frontier = roaring64_bitmap_create();
    roaring64_bitmap_t *next = roaring64_bitmap_create();
    // Start from direct neighbors
    roaring64_bitmap_or_inplace(frontier, graph->adj_pos[u]);
    roaring64_bitmap_or_inplace(reachable, frontier);
    for (uint64_t hop = 1; hop < n_hops; hop ++) {
      roaring64_bitmap_clear(next);
      roaring64_iterator_reinit(frontier, &it0);
      while (roaring64_iterator_has_value(&it0)) {
        uint64_t v = roaring64_iterator_value(&it0);
        roaring64_bitmap_or_inplace(next, graph->adj_pos[v]);
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
      if (roaring64_bitmap_contains(graph->adj_pos[u], (uint64_t) w))
        continue;
      if (roaring64_bitmap_contains(graph->adj_neg[u], (uint64_t) w))
        continue;
      double dist = (double) roaring64_bitmap_xor_cardinality(graph->nodes[u], graph->nodes[w]) / (double) graph->n_features;
      tm_neighbor_t vw = (tm_neighbor_t) { w, dist };
      kv_push(tm_candidate_t, candidates, tm_candidate(u, vw));
    }
    // Sort and add top-k
    ks_introsort(candidates_asc, candidates.n, candidates.a);
    for (uint64_t i = 0; i < candidates.n && i < n_grow_pos; i ++) {
      tm_candidate_t c = candidates.a[i];
      tm_pair_t e = tm_pair(u, c.v.v);
      khi = kh_put(pairs, graph->pairs, e, &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = true;
      kv_push(tm_pair_t, new_pos, tm_pair(u, c.v.v));
      tm_dsu_union(L, &graph->dsu, u, c.v.v);
      graph->n_pos ++;
    }

    // Cleanup
    roaring64_bitmap_free(reachable);
    roaring64_bitmap_free(frontier);
    roaring64_bitmap_free(next);
  }

  // Contrastive negative expansion
  for (int64_t u = 0; u < (int64_t) graph->n_nodes; u ++) {
    roaring64_bitmap_t *seen = roaring64_bitmap_create();
    kv_size(candidates) = 0;
    // One-hop contrastive expansion from adj_pos[u] to adj_neg[v]
    roaring64_iterator_reinit(graph->adj_pos[u], &it0);
    while (roaring64_iterator_has_value(&it0)) {
      int64_t v = (int64_t) roaring64_iterator_value(&it0);
      roaring64_iterator_advance(&it0);
      roaring64_iterator_reinit(graph->adj_neg[v], &it1);
      while (roaring64_iterator_has_value(&it1)) {
        int64_t w = (int64_t) roaring64_iterator_value(&it1);
        roaring64_iterator_advance(&it1);
        if (w == u)
          continue;
        if (roaring64_bitmap_contains(graph->adj_pos[u], (uint64_t) w))
          continue;
        if (roaring64_bitmap_contains(graph->adj_neg[u], (uint64_t) w))
          continue;
        if (roaring64_bitmap_contains(seen, (uint64_t) w))
          continue;
        roaring64_bitmap_add(seen, (uint64_t) w);
        double dist = (double) roaring64_bitmap_xor_cardinality(graph->nodes[u], graph->nodes[w]) / (double) graph->n_features;
        tm_neighbor_t nw = (tm_neighbor_t) { w, dist };
        kv_push(tm_candidate_t, candidates, tm_candidate(u, nw));
      }
    }
    // Select furthest in feature space
    ks_introsort(candidates_desc, candidates.n, candidates.a);  // sort by descending distance
    for (uint64_t i = 0; i < candidates.n && i < n_grow_neg; i ++) {
      tm_candidate_t c = candidates.a[i];
      tm_pair_t e = tm_pair(u, c.v.v);
      khi = kh_put(pairs, graph->pairs, e, &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = false;
      kv_push(tm_pair_t, new_neg, tm_pair(u, c.v.v));
      graph->n_neg ++;
    }
    roaring64_bitmap_free(seen);
  }

  // Update adjacency
  for (size_t i = 0; i < kv_size(new_pos); i ++) {
    tm_pair_t e = kv_A(new_pos, i);
    roaring64_bitmap_add(graph->adj_pos[e.u], (uint64_t) e.v);
    roaring64_bitmap_add(graph->adj_pos[e.v], (uint64_t) e.u);
  }
  for (size_t i = 0; i < kv_size(new_neg); i ++) {
    tm_pair_t e = kv_A(new_neg, i);
    roaring64_bitmap_add(graph->adj_neg[e.u], (uint64_t) e.v);
    roaring64_bitmap_add(graph->adj_neg[e.v], (uint64_t) e.u);
  }

  // Cleanup
  kv_destroy(new_pos);
  kv_destroy(new_neg);
  kv_destroy(candidates);
}

static inline void tm_adj_init (
  tk_graph_t *graph
) {
  khint_t khi;

  // Init
  // TODO: parallelize
  for (uint64_t s = 0; s < graph->n_nodes; s ++) {
    roaring64_bitmap_clear(graph->adj_pos[s]);
    roaring64_bitmap_clear(graph->adj_neg[s]);
  }

  // Populate adj lists
  // TODO: parallelize?
  for (khi = kh_begin(graph->pairs); khi < kh_end(graph->pairs); khi ++) {
    if (!kh_exist(graph->pairs, khi))
      continue;
    tm_pair_t p = kh_key(graph->pairs, khi);
    bool l = kh_value(graph->pairs, khi);
    if (l) {
      roaring64_bitmap_add(graph->adj_pos[p.u], (uint64_t) p.v);
      roaring64_bitmap_add(graph->adj_pos[p.v], (uint64_t) p.u);
    } else {
      roaring64_bitmap_add(graph->adj_neg[p.u], (uint64_t) p.v);
      roaring64_bitmap_add(graph->adj_neg[p.v], (uint64_t) p.u);
    }
  }
}

static inline void tm_add_pairs (
  lua_State *L,
  tk_graph_t *graph
) {
  int kha;
  khint_t khi;

  uint64_t n_pos_old = graph->n_pos;
  uint64_t n_pos_new = 0;

  for (uint64_t i = 0; i < n_pos_old; i ++) {
    tm_pair_t p = graph->pos[i];
    if (p.v < p.u)
      graph->pos[i] = p = tm_pair(p.v, p.u);
    khi = kh_put(pairs, graph->pairs, p, &kha);
    if (!kha)
      continue;
    kh_value(graph->pairs, khi) = true;
    n_pos_new ++;
  }

  uint64_t n_neg_old = graph->n_neg;
  uint64_t n_neg_new = 0;

  for (uint64_t i = 0; i < n_neg_old; i ++) {
    tm_pair_t p = graph->neg[i];
    if (p.v < p.u)
      graph->neg[i] = p = tm_pair(p.v, p.u);
    khi = kh_put(pairs, graph->pairs, p, &kha);
    if (!kha)
      continue;
    kh_value(graph->pairs, khi) = false;
    n_neg_new ++;
  }

  graph->n_pos = n_pos_new;
  graph->n_neg = n_neg_new;

  tm_adj_init(graph);
  tm_dsu_init(L, &graph->dsu, graph->pairs, graph->n_nodes);
}

static inline int tm_to_bits (lua_State *L)
{
  lua_settop(L, 2);
  lua_pushvalue(L, 1);
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pairs = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pairs = (uint64_t) luaL_checkinteger(L, -1) / 2;
  uint64_t n_nodes = tk_lua_checkunsigned(L, 2, "n_nodes");

  kvec_t(int64_t) out;
  kv_init(out);

  for (uint64_t i = 0; i < n_pairs; i ++) {
    tm_pair_t p = pairs[i];
    int64_t u = p.u, v = p.v;
    kv_push(int64_t, out, u * (int64_t) n_nodes + v);
    kv_push(int64_t, out, v * (int64_t) n_nodes + u);
  }

  kv_resize(int64_t, out, out.n);
  lua_pushlightuserdata(L, out.a);
  lua_pushinteger(L, 1);
  lua_pushinteger(L, (int64_t) out.n);
  tk_lua_callmod(L, 3, 1, "santoku.matrix.integer", "from_view");
  return 1;
}

static tk_graph_t *tm_graph_create (
  lua_State *L,
  tm_pair_t *pos,
  tm_pair_t *neg,
  uint64_t n_pos,
  uint64_t n_neg,
  int64_t *set_bits,
  uint64_t n_set_bits,
  uint64_t n_nodes,
  uint64_t n_features,
  uint64_t knn_cache,
  unsigned int n_threads
) {
  tk_graph_t *graph = (tk_graph_t *) lua_newuserdata(L, sizeof(tk_graph_t)); // ud
  luaL_getmetatable(L, TK_MT_GRAPH); // ud mt
  lua_setmetatable(L, -2); // ud
  memset(graph, 0, sizeof(tk_graph_t));
  graph->threads = tk_malloc(L, n_threads * sizeof(tk_graph_t));
  memset(graph->threads, 0, n_threads * sizeof(tk_graph_thread_t));
  graph->pool = tk_threads_create(L, n_threads, tk_graph_worker);
  graph->neighbors = tk_malloc(L, n_nodes * sizeof(tm_neighbors_t));
  graph->nodes = tk_malloc(L, n_nodes * sizeof(roaring64_bitmap_t *));
  graph->set_bits = set_bits;
  graph->n_set_bits = n_set_bits;
  graph->n_nodes = n_nodes;
  graph->n_features = n_features;
  graph->knn_cache = knn_cache;
  graph->pairs = kh_init(pairs);
  graph->pos = pos;
  graph->neg = neg;
  graph->n_pos = n_pos;
  graph->n_neg = n_neg;
  graph->adj_pos = tk_malloc(L, n_nodes * sizeof(roaring64_bitmap_t *));
  graph->adj_neg = tk_malloc(L, n_nodes * sizeof(roaring64_bitmap_t *));
  for (uint64_t s = 0; s < graph->n_nodes; s ++) {
    graph->nodes[s] = roaring64_bitmap_create();
    graph->adj_pos[s] = roaring64_bitmap_create();
    graph->adj_neg[s] = roaring64_bitmap_create();
  }
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_graph_thread_t *data = graph->threads + i;
    graph->pool->threads[i].data = data;
    data->graph = graph;
    data->candidates = roaring64_bitmap_create();
    tk_thread_range(i, n_threads, n_nodes, &data->ufirst, &data->ulast);
  }
  return graph;
}

static void tm_graph_destroy (tk_graph_t *graph)
{
  for (int64_t s = 0; s < (int64_t) graph->n_nodes; s ++)
    roaring64_bitmap_free(graph->nodes[s]);
  free(graph->nodes);
  for (int64_t s = 0; s < (int64_t) graph->n_nodes; s ++)
    kv_destroy(graph->neighbors[s]);
  free(graph->neighbors);
  tm_dsu_free(&graph->dsu);
  for (unsigned int i = 0; i < graph->pool->n_threads; i ++) {
    tk_graph_thread_t *data = (tk_graph_thread_t *) graph->pool->threads[i].data;
    roaring64_bitmap_free(data->candidates);
  }
  tk_threads_destroy(graph->pool);
  kh_destroy(pairs, graph->pairs);
  for (uint64_t u = 0; u < graph->n_nodes; u ++) {
    roaring64_bitmap_free(graph->adj_pos[u]);
    roaring64_bitmap_free(graph->adj_neg[u]);
  }
  free(graph->adj_pos);
  free(graph->adj_neg);
  free(graph->threads);
}

static inline int tm_graph_gc (lua_State *L)
{
  tk_graph_t *graph = tk_graph_peek(L, 1);
  tm_graph_destroy(graph);
  return 0;
}

static inline int tm_create (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "pos");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pos = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pos = (uint64_t) luaL_checkinteger(L, -1) / 2;

  lua_getfield(L, 1, "neg");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *neg = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_neg = (uint64_t) luaL_checkinteger(L, -1) / 2;

  lua_getfield(L, 1, "nodes");
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  int64_t *set_bits = (int64_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_set_bits = (uint64_t) luaL_checkinteger(L, -1);

  uint64_t n_nodes = tk_lua_fcheckunsigned(L, 1, "enrich", "n_nodes");
  uint64_t n_features = tk_lua_fcheckunsigned(L, 1, "enrich", "n_features");
  uint64_t knn_cache = tk_lua_foptunsigned(L, 1, "enrich", "knn_cache", 32);
  uint64_t knn = tk_lua_foptunsigned(L, 1, "enrich", "knn", 0);
  uint64_t n_hops = tk_lua_foptunsigned(L, 1, "enrich", "trans_hops", 0);
  uint64_t n_grow_pos = tk_lua_foptunsigned(L, 1, "enrich", "trans_pos", 0);
  uint64_t n_grow_neg = tk_lua_foptunsigned(L, 1, "enrich", "trans_neg", 0);
  bool do_mst = tk_lua_foptboolean(L, 1, "enrich", "mst", true);

  unsigned int n_threads = tk_threads_getn(L, 1, "enrich", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  // Init graph
  tk_graph_t *graph = tm_graph_create(L, pos, neg, n_pos, n_neg, set_bits, n_set_bits, n_nodes, n_features, knn_cache, n_threads);

  // Add base pairs
  tm_add_pairs(L, graph);

  // Prep for densify
  tm_index_neighbors(L, graph);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tm_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_pos);
    lua_pushinteger(L, (int64_t) graph->n_neg);
    lua_pushstring(L, "seed");
    lua_call(L, 4, 0);
  }

  // Add knn
  if (knn > 0)
    tm_add_knn(L, graph, knn);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tm_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_pos);
    lua_pushinteger(L, (int64_t) graph->n_neg);
    lua_pushstring(L, "knn");
    lua_call(L, 4, 0);
  }

  // Add mst
  if (do_mst)
    tm_add_mst(L, graph);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tm_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_pos);
    lua_pushinteger(L, (int64_t) graph->n_neg);
    lua_pushstring(L, "kruskal");
    lua_call(L, 4, 0);
  }

  // Densify graph, adding transatives
  if (n_hops > 0 && (n_grow_pos > 0 || n_grow_neg > 0))
    tm_add_transatives(L, graph, n_hops, n_grow_pos, n_grow_neg);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tm_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_pos);
    lua_pushinteger(L, (int64_t) graph->n_neg);
    lua_pushstring(L, "transitives");
    lua_call(L, 4, 0);
  }

  // Return graph
  return 1;
}

static inline int tm_graph_pairs (lua_State *L)
{
  lua_settop(L, 1);
  tk_graph_t *graph = tk_graph_peek(L, 1);

  // Render pairs
  tm_pair_t *pos = tk_malloc(L, graph->n_pos * sizeof(tm_pair_t));
  tm_pair_t *neg = tk_malloc(L, graph->n_neg * sizeof(tm_pair_t));
  uint64_t n_pos, n_neg;
  tm_render_pairs(L, graph, pos, neg, &n_pos, &n_neg);

  // Push positives
  lua_pushlightuserdata(L, pos);
  lua_pushinteger(L, (int64_t) n_pos);
  lua_pushinteger(L, 2);
  tk_lua_callmod(L, 3, 1, "santoku.matrix.integer", "from_view");

  // Push updated negatives
  lua_pushlightuserdata(L, neg);
  lua_pushinteger(L, (int64_t) n_neg);
  lua_pushinteger(L, 2);
  tk_lua_callmod(L, 3, 1, "santoku.matrix.integer", "from_view");

  // Return new pos/neg pair lists
  return 2;
}

static luaL_Reg tm_graph_fns[] =
{
  { "pairs", tm_graph_pairs },
  { "create", tm_create },
  { "to_bits", tm_to_bits },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_graph (lua_State *L)
{
  lua_settop(L, 0);
  lua_newtable(L); // t
  tk_lua_register(L, tm_graph_fns, 0); // t
  luaL_newmetatable(L, TK_MT_GRAPH); // t mt
  lua_pushcfunction(L, tm_graph_gc); // t mt gc
  lua_setfield(L, -2, "__gc"); // t mt
  lua_pop(L, 1); // t
  return 1;
}
