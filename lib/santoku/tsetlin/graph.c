#define _GNU_SOURCE

#include <santoku/tsetlin/graph.h>

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  tk_pvec_t *pos,
  tk_pvec_t *neg,
  tk_ivec_t *labels,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  uint64_t knn_cache,
  double knn_eps,
  unsigned int n_threads
);

static inline void tk_graph_worker (void *dp, int sig)
{
  tk_graph_stage_t stage = (tk_graph_stage_t) sig;
  // tk_graph_thread_t *data = (tk_graph_thread_t *) dp;
  switch (stage) {
    case TK_GRAPH_TODO:
      // TODO
      break;
  }
}

static inline void tm_render_pairs (
  lua_State *L,
  tk_graph_t *graph,
  tk_pvec_t *pos,
  tk_pvec_t *neg
) {
  bool l;
  tm_pair_t p;
  kh_foreach(graph->pairs, p,  l, ({
    tk_pvec_push(l ? pos : neg, tk_pair(p.u, p.v));
  }))
  tk_pvec_asc(pos, 0, pos->n);
  tk_pvec_asc(neg, 0, neg->n);
}

static inline tk_iuset_t *tk_graph_get_adj (
  tk_graph_t *graph,
  int64_t u,
  bool pos
) {
  tk_graph_adj_t *adj = pos ? graph->adj_pos : graph->adj_neg;
  khint_t khi = tk_iumap_get(graph->uid_hood, u);
  if (khi == tk_iumap_end(graph->uid_hood))
    return NULL;
  int64_t iu = tk_iumap_value(graph->uid_hood, khi);
  return adj->a[iu];
}

static inline void tk_graph_add_adj (
  tk_graph_t *graph,
  int64_t u,
  int64_t v,
  bool pos
) {
  int kha;
  khint_t khi;
  tk_graph_adj_t *adj = pos ? graph->adj_pos : graph->adj_neg;
  khi = tk_iumap_get(graph->uid_hood, u);
  if (khi == tk_iumap_end(graph->uid_hood))
    return;
  int64_t iu = tk_iumap_value(graph->uid_hood, khi);
  khi = tk_iumap_get(graph->uid_hood, v);
  if (khi == tk_iumap_end(graph->uid_hood))
    return;
  int64_t iv = tk_iumap_value(graph->uid_hood, khi);
  tk_iuset_put(adj->a[iu], iv, &kha);
  tk_iuset_put(adj->a[iv], iu, &kha);
}

static inline bool tk_graph_same_label (
  tk_graph_t *graph,
  int64_t u,
  int64_t v
) {
  return graph->labels != NULL
      && u >= 0 && u < (int64_t) graph->labels->n
      && v >= 0 && v < (int64_t) graph->labels->n
      && graph->labels->a[u] == graph->labels->a[v];
}

static inline void tm_add_knn (
  lua_State *L,
  tk_graph_t *graph,
  uint64_t knn
) {
  int kha;
  khint_t khi;

  // Prep shuffle
  tk_ivec_t *shuf = tk_ivec_create(L, graph->uids->n, 0, 0);
  tk_ivec_fill_indices(shuf);
  tk_ivec_shuffle(shuf);

  // Add neighbors
  if (graph->inv != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t i = shuf->a[su];
      int64_t u = graph->uids->a[i];
      tk_rvec_t *ns = graph->inv_hoods->a[i];
      uint64_t added = 0;
      for (khint_t j = 0; j < ns->n && added < knn; j ++) {
        tk_rank_t r = ns->a[j];
        if (r.i > (int64_t) graph->uids->n || r.i < 0)
          continue;
        int64_t v = graph->uids->a[r.i];
        tm_pair_t e = tm_pair(u, v);
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        added ++;
        bool w = graph->labels == NULL || tk_graph_same_label(graph, u, v);
        kh_value(graph->pairs, khi) = w;
        tk_graph_add_adj(graph, u, v, w);
        tk_dsu_union(&graph->dsu, u, v);
        if (w)
          graph->n_pos ++;
        else
          graph->n_neg ++;
      }
    }

  } else if (graph->ann != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t i = shuf->a[su];
      int64_t u = graph->uids->a[i];
      tk_pvec_t *ns = graph->ann_hoods->a[i];
      uint64_t added = 0;
      for (khint_t j = 0; j < ns->n && added < knn; j ++) {
        tk_pair_t r = ns->a[j];
        if (r.i > (int64_t) graph->uids->n || r.i < 0)
          continue;
        int64_t v = graph->uids->a[r.i];
        tm_pair_t e = tm_pair(u, v);
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        added ++;
        bool w = graph->labels == NULL || tk_graph_same_label(graph, u, v);
        kh_value(graph->pairs, khi) = w;
        tk_graph_add_adj(graph, u, v, w);
        tk_dsu_union(&graph->dsu, u, v);
        if (w)
          graph->n_pos ++;
        else
          graph->n_neg ++;
      }
    }

  } else if (graph->hbi != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t i = shuf->a[su];
      int64_t u = graph->uids->a[i];
      tk_pvec_t *ns = graph->hbi_hoods->a[i];
      uint64_t added = 0;
      for (khint_t j = 0; j < ns->n && added < knn; j ++) {
        tk_pair_t r = ns->a[j];
        if (r.i > (int64_t) graph->uids->n || r.i < 0)
          continue;
        int64_t v = graph->uids->a[r.i];
        tm_pair_t e = tm_pair(u, v);
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        added ++;
        bool w = graph->labels == NULL || tk_graph_same_label(graph, u, v);
        kh_value(graph->pairs, khi) = true;
        tk_graph_add_adj(graph, u, v, true);
        tk_dsu_union(&graph->dsu, u, v);
        if (w)
          graph->n_pos ++;
        else
          graph->n_neg ++;
      }
    }

  } else {
    assert(false);
  }

  // Cleanup
  lua_pop(L, 1);
}

static inline void tm_add_mst (
  lua_State *L,
  tk_graph_t *graph
) {
  int kha;
  khint_t khi;

  // Prep shuffle
  tk_ivec_t *shuf = tk_ivec_create(L, graph->uids->n, 0, 0);
  tk_ivec_fill_indices(shuf);
  tk_ivec_shuffle(shuf);

  // Gather all inter-component edges
  // TODO: Unified/generalized INV/ANN wrapper
  tm_candidates_t all_candidates;
  kv_init(all_candidates);

  if (graph->inv != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t u = shuf->a[su];
      khi = tk_iumap_get(graph->uid_hood, u);
      if (khi == tk_iumap_end(graph->uid_hood))
        continue;
      int64_t i = tk_iumap_value(graph->uid_hood,  khi);
      tk_rvec_t *ns = graph->inv_hoods->a[i];
      int64_t cu = tk_dsu_find(&graph->dsu, u);
      for (khint_t j = 0; j < ns->n; j ++) {
        tk_rank_t r = ns->a[j];
        if (r.i > (int64_t) graph->uids->n || r.i < 0)
          continue;
        int64_t v = graph->uids->a[r.i];
        if (graph->labels != NULL && !tk_graph_same_label(graph, u, v))
          continue;
        if (cu == tk_dsu_find(&graph->dsu, v))
          continue;
        tm_pair_t e = tm_pair(u, v);
        khi = kh_get(pairs, graph->pairs, e);
        if (khi != kh_end(graph->pairs))
          continue;
        // TODO: Can we use heap?
        kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v, r.d));
      }
    }

  } else if (graph->ann != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t u = shuf->a[su];
      khi = tk_iumap_get(graph->uid_hood, u);
      if (khi == tk_iumap_end(graph->uid_hood))
        continue;
      int64_t i = tk_iumap_value(graph->uid_hood,  khi);
      tk_pvec_t *ns = graph->ann_hoods->a[i];
      int64_t cu = tk_dsu_find(&graph->dsu, u);
      for (khint_t j = 0; j < ns->n; j ++) {
        tk_pair_t r = ns->a[j];
        if (r.i > (int64_t) graph->uids->n || r.i < 0)
          continue;
        int64_t v = graph->uids->a[r.i];
        if (graph->labels != NULL && !tk_graph_same_label(graph, u, v))
          continue;
        if (cu == tk_dsu_find(&graph->dsu, v))
          continue;
        tm_pair_t e = tm_pair(u, v);
        khi = kh_get(pairs, graph->pairs, e);
        if (khi != kh_end(graph->pairs))
          continue;
        // TODO: Can we use heap?
        kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v, (double) r.p));
      }
    }

  } else if (graph->hbi != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t u = shuf->a[su];
      khi = tk_iumap_get(graph->uid_hood, u);
      if (khi == tk_iumap_end(graph->uid_hood))
        continue;
      int64_t i = tk_iumap_value(graph->uid_hood,  khi);
      tk_pvec_t *ns = graph->hbi_hoods->a[i];
      int64_t cu = tk_dsu_find(&graph->dsu, u);
      for (khint_t j = 0; j < ns->n; j ++) {
        tk_pair_t r = ns->a[j];
        if (r.i > (int64_t) graph->uids->n || r.i < 0)
          continue;
        int64_t v = graph->uids->a[r.i];
        if (graph->labels != NULL && !tk_graph_same_label(graph, u, v))
          continue;
        if (cu == tk_dsu_find(&graph->dsu, v))
          continue;
        tm_pair_t e = tm_pair(u, v);
        khi = kh_get(pairs, graph->pairs, e);
        if (khi != kh_end(graph->pairs))
          continue;
        // TODO: Can we use heap?
        kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v, (double) r.p));
      }
    }

  } else {
    assert(false);
  }

  // Sort all by distance ascending (nearest in feature space)
  ks_introsort(candidates_asc, all_candidates.n, all_candidates.a);

  // Loop through candidates in order and add if they connect components
  for (uint64_t i = 0; i < all_candidates.n && tk_dsu_components(&graph->dsu) > 1; i ++) {
    tm_candidate_t c = all_candidates.a[i];
    int64_t cu = tk_dsu_find(&graph->dsu, c.u);
    int64_t cv = tk_dsu_find(&graph->dsu, c.v);
    if (cu == cv)
      continue;
    tm_pair_t e = tm_pair(c.u, c.v);
    khi = kh_put(pairs, graph->pairs, e, &kha);
    if (!kha)
      continue;
    kh_value(graph->pairs, khi) = true;
    tk_graph_add_adj(graph, c.u, c.v, true);
    tk_dsu_union(&graph->dsu, c.u, c.v);
    graph->n_pos ++;
  }

  // Cleanup
  kv_destroy(all_candidates);
  lua_pop(L, 1);
}

static inline double tm_dist (tk_graph_t *graph, int64_t u, int64_t v)
{
  if (graph->inv != NULL) {

    size_t un;
    int64_t *uset = tk_inv_get(graph->inv, u, &un);
    if (uset == NULL)
      return -1;
    size_t wn;
    int64_t *wset = tk_inv_get(graph->inv, v, &wn);
    if (wset == NULL)
      return -1;
    return 1.0 - tk_inv_jaccard(uset, un, wset, wn);

  } else if (graph->ann != NULL) {

    char *uset = tk_ann_get(graph->ann, u);
    if (uset == NULL)
      return -1;
    char *wset = tk_ann_get(graph->ann, v);
    if (wset == NULL)
      return -1;
    return (double) tk_ann_hamming(uset, wset, graph->ann->features) / (double) graph->ann->features;

  } else if (graph->hbi != NULL) {

    char *uset = tk_hbi_get(graph->hbi, u);
    if (uset == NULL)
      return -1;
    char *wset = tk_hbi_get(graph->hbi, v);
    if (wset == NULL)
      return -1;
    return (double) tk_ann_hamming(uset, wset, graph->hbi->features) / (double) graph->hbi->features;

  } else {
    assert(false);
    return 0;
  }
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

  kvec_t(tm_pair_t) new_pos;
  kv_init(new_pos);

  kvec_t(tm_pair_t) new_neg;
  kv_init(new_neg);

  // Transitive positive expansion
  tm_candidates_t candidates;
  kv_init(candidates);
  int64_t u, w, iu, iv, iw;
  tk_iuset_t *upos, *uneg;
  tk_iuset_t *reachable = tk_iuset_create();
  tk_iuset_t *frontier = tk_iuset_create();
  tk_iuset_t *next = tk_iuset_create();
  for (iu = 0; iu < (int64_t) graph->adj_pos->n; iu ++) {
    upos = graph->adj_pos->a[iu];
    uneg = graph->adj_neg->a[iu];
    u = graph->uids->a[iu];
    // Init closure
    tk_iuset_clear(reachable);
    tk_iuset_clear(frontier);
    tk_iuset_clear(next);
    // Start from direct neighbors
    tk_iuset_union(frontier, upos);
    tk_iuset_union(reachable, frontier);
    for (uint64_t hop = 1; hop < n_hops; hop ++) {
      tk_iuset_clear(next);
      tk_iuset_foreach(frontier, iv, ({
        tk_iuset_t *vpos = graph->adj_pos->a[iv];
        if (vpos != NULL)
          tk_iuset_union(next, vpos);
      }))
      tk_iuset_union(reachable, next);
      tk_iuset_clear(frontier); // advance
      tk_iuset_union(frontier, next); // advance
    }
    // Filter out reachable
    khi = tk_iuset_get(reachable, iu);
    if (khi != tk_iuset_end(reachable))
      tk_iuset_del(reachable, khi);
    kv_size(candidates) = 0;
    tk_iuset_foreach(reachable, iw, ({
      if (upos != NULL && tk_iuset_contains(upos, iw))
        continue;
      if (uneg != NULL && tk_iuset_contains(uneg, iw))
        continue;
      w = graph->uids->a[iw];
      double dist = tm_dist(graph, u, w);
      if (dist < 0)
        continue;
      // TODO: Can we use heap?
      kv_push(tm_candidate_t, candidates, tm_candidate(u, w, dist));
    }))
    // Sort and add top-k
    ks_introsort(candidates_asc, candidates.n, candidates.a);
    for (uint64_t i = 0; i < candidates.n && i < n_grow_pos; i ++) {
      tm_candidate_t c = candidates.a[i];
      if (graph->labels != NULL && !tk_graph_same_label(graph, u, c.v))
        continue;
      tm_pair_t e = tm_pair(u, c.v);
      khi = kh_put(pairs, graph->pairs, e, &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = true;
      kv_push(tm_pair_t, new_pos, tm_pair(u, c.v));
      tk_dsu_union(&graph->dsu, u, c.v);
      graph->n_pos ++;
    }
  }

  // Cleanup
  tk_iuset_destroy(reachable);
  tk_iuset_destroy(frontier);
  tk_iuset_destroy(next);

  // Contrastive negative expansion
  tk_iuset_t *seen = tk_iuset_create();
  for (iu = 0; iu < (int64_t) graph->adj_pos->n; iu ++) {
    upos = graph->adj_pos->a[iu];
    uneg = graph->adj_neg->a[iu];
    u = graph->uids->a[iu];
    tk_iuset_clear(seen);
    kv_size(candidates) = 0;
    // One-hop contrastive expansion from adj_pos[u] to adj_neg[v]
    if (upos == NULL)
      continue;
    tk_iuset_foreach(upos, iv, ({
      tk_iuset_t *vneg = graph->adj_neg->a[iv];
      if (vneg == NULL)
        continue;
      tk_iuset_foreach(vneg, iw, ({
        if (iw == iu)
          continue;
        if (upos != NULL && tk_iuset_contains(upos, iw))
          continue;
        if (uneg != NULL && tk_iuset_contains(uneg, iw))
          continue;
        if (tk_iuset_contains(seen, iw))
          continue;
        tk_iuset_put(seen, iw, &kha);
        w = graph->uids->a[iw];
        double dist = tm_dist(graph, u, w);
        if (dist < 0)
          continue;
        // TODO: Can we use heap?
        kv_push(tm_candidate_t, candidates, tm_candidate(u, w, dist));
      }))
    }))
    // Select furthest in feature space
    ks_introsort(candidates_desc, candidates.n, candidates.a);  // sort by descending distance
    for (uint64_t i = 0; i < candidates.n && i < n_grow_neg; i ++) {
      tm_candidate_t c = candidates.a[i];
      if (graph->labels != NULL && tk_graph_same_label(graph, u, c.v))
        continue;
      tm_pair_t e = tm_pair(u, c.v);
      khi = kh_put(pairs, graph->pairs, e, &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = false;
      kv_push(tm_pair_t, new_neg, tm_pair(u, c.v));
      tk_dsu_union(&graph->dsu, u, c.v);
      graph->n_neg ++;
    }
  }
  tk_iuset_destroy(seen);

  // Update adjacency
  for (size_t i = 0; i < kv_size(new_pos); i ++) {
    tm_pair_t e = kv_A(new_pos, i);
    tk_graph_add_adj(graph, e.u, e.v, true);
  }
  for (size_t i = 0; i < kv_size(new_neg); i ++) {
    tm_pair_t e = kv_A(new_neg, i);
    tk_graph_add_adj(graph, e.u, e.v, false);
  }

  // Cleanup
  kv_destroy(new_pos);
  kv_destroy(new_neg);
  kv_destroy(candidates);
}

static inline void tm_adj_init (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  graph->adj_pos = tk_graph_adj_create(L, graph->uids->n, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);

  graph->adj_neg = tk_graph_adj_create(L, graph->uids->n, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);

  for (uint64_t i = 0; i < graph->uids->n; i ++) {
    graph->adj_pos->a[i] = tk_iuset_create();
    graph->adj_neg->a[i] = tk_iuset_create();
  }

  // Populate adj lists
  bool l;
  tm_pair_t p;
  kh_foreach(graph->pairs, p, l, ({
    tk_graph_add_adj(graph, p.u, p.v, l);
  }))
}

static inline void tm_add_pairs (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  int kha;
  khint_t khi;

  if (graph->pos != NULL) {
    uint64_t n_pos_old = graph->pos->n;
    uint64_t n_pos_new = 0;
    for (uint64_t i = 0; i < n_pos_old; i ++) {
      int64_t u = graph->pos->a[i].i;
      int64_t v = graph->pos->a[i].p;
      if (graph->labels != NULL && !tk_graph_same_label(graph, u, v))
        continue;
      khi = kh_put(pairs, graph->pairs, tm_pair(u, v), &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = true;
      tk_dsu_union(&graph->dsu, u, v);
      n_pos_new ++;
    }
    graph->n_pos = n_pos_new;
  }

  if (graph->neg != NULL) {
    uint64_t n_neg_old = graph->neg->n;
    uint64_t n_neg_new = 0;
    for (uint64_t i = 0; i < n_neg_old; i ++) {
      int64_t u = graph->neg->a[i].i;
      int64_t v = graph->neg->a[i].p;
      if (graph->labels != NULL && tk_graph_same_label(graph, u, v))
        continue;
      khi = kh_put(pairs, graph->pairs, tm_pair(u, v), &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = false;
      tk_dsu_union(&graph->dsu, u, v);
      n_neg_new ++;
    }
    graph->n_neg = n_neg_new;
  }

  tm_adj_init(L, Gi, graph);
}

static inline int tm_to_bits (lua_State *L)
{
  lua_settop(L, 2);
  tk_ivec_t *pairs = tk_ivec_peek(L, 1, "pairs");
  uint64_t n_nodes = tk_lua_checkunsigned(L, 2, "n_nodes");
  tk_ivec_t *out = tk_ivec_create(L, 0, 0, 0);
  for (uint64_t i = 0; i < pairs->n; i += 2) {
    int64_t u = pairs->a[i];
    int64_t v = pairs->a[i + 1];
    tk_ivec_push(out, u * (int64_t) n_nodes + v);
    tk_ivec_push(out, v * (int64_t) n_nodes + u);
  }
  tk_ivec_shrink(L, out);
  return 1;
}

static void tm_graph_destroy (tk_graph_t *graph)
{
  tk_dsu_free(&graph->dsu);
  tk_threads_destroy(graph->pool);
  if (graph->threads)
    free(graph->threads);
  if (graph->pairs)
    kh_destroy(pairs, graph->pairs);
  if (graph->uid_hood)
    tk_iumap_destroy(graph->uid_hood);
}

static inline int tm_graph_gc (lua_State *L)
{
  tk_graph_t *graph = tk_graph_peek(L, 1);
  tm_graph_destroy(graph);
  return 0;
}

static inline void tm_setup_hoods (lua_State *L, int Gi, tk_graph_t *graph)
{
  if (graph->inv != NULL)
    tk_inv_neighborhoods(L, graph->inv, graph->knn_cache, graph->knn_eps, &graph->inv_hoods, &graph->uids);
  else if (graph->ann != NULL)
    tk_ann_neighborhoods(L, graph->ann, graph->knn_cache, graph->ann->features * graph->knn_eps, &graph->ann_hoods, &graph->uids);
  else if (graph->hbi != NULL)
    tk_hbi_neighborhoods(L, graph->hbi, graph->knn_cache, graph->hbi->features * graph->knn_eps, &graph->hbi_hoods, &graph->uids);
  else
    assert(false);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -2); // uids
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1); // hoods
  lua_pop(L, 2);
}

static inline void tm_bridge_islands (tk_graph_t *graph)
{
  tk_pumap_t *info = tk_pumap_create();
  int kha;
  khint_t h;
  for (uint64_t i = 0; i < graph->uids->n; ++i) {
    int64_t uid = graph->uids->a[i];
    int64_t root = tk_dsu_find(&graph->dsu, uid);
    h = tk_pumap_put(info, root, &kha);
    tk_pair_t *iv = &tk_pumap_value(info, h);
    if (kha) {
      iv->i = uid;
      iv->p = 1;
    } else {
      iv->p ++;
    }
  }
  int64_t root_big = -1;
  int64_t big_size = -1;
  int64_t big_uid = -1;
  int64_t root;
  tk_pair_t iv;
  tk_pumap_foreach(info, root, iv, ({
    if (iv.p > big_size) {
      big_size = iv.p;
      root_big = root;
      big_uid = iv.i;
    }
  }))
  if (root_big == -1 || tk_pumap_size(info) == 1) {
    tk_pumap_destroy(info);
    return;
  }
  for (h = tk_pumap_begin(info); h != tk_pumap_end(info); ++h) {
    if (!tk_pumap_exist(info, h))
      continue;
    root = tk_pumap_key(info, h);
    if (root == root_big)
      continue;
    iv = tk_pumap_value(info, h);
    int64_t u = iv.i;
    int64_t v = big_uid;
    tm_pair_t e = tm_pair(u, v);
    khint_t khi = kh_put(pairs, graph->pairs, e, &kha);
    if (kha) {
      if (tk_graph_same_label(graph, u, v)) {
        kh_value(graph->pairs, khi) = true;
        tk_graph_add_adj(graph, u, v, true);
      } else {
        kh_value(graph->pairs, khi) = false;
        tk_graph_add_adj(graph, u, v, false);
      }
      graph->n_neg ++;
    }
    tk_dsu_union(&graph->dsu, u, v);
  }
  tk_pumap_destroy(info);
}

static inline int tm_create (lua_State *L)
{
  lua_settop(L, 1);
  int kha;
  khint_t khi;

  lua_getfield(L, 1, "pos");
  tk_pvec_t *pos = tk_pvec_peekopt(L, -1);

  lua_getfield(L, 1, "neg");
  tk_pvec_t *neg = tk_pvec_peekopt(L, -1);

  lua_getfield(L, 1, "labels");
  tk_ivec_t *labels = tk_ivec_peekopt(L, -1);

  lua_getfield(L, 1, "index");
  tk_inv_t *inv = tk_inv_peekopt(L, -1);
  tk_ann_t *ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *hbi = tk_hbi_peekopt(L, -1);
  if (inv == NULL && ann == NULL && hbi == NULL)
    tk_lua_verror(L, 3, "graph", "index", "either tk_ann_t, tk_inv_t, or tk_hbi_t must be provided");

  uint64_t knn = tk_lua_foptunsigned(L, 1, "graph", "knn", 0);
  uint64_t knn_cache = tk_lua_foptunsigned(L, 1, "graph", "knn_cache", 4);
  double knn_eps = tk_lua_foptunsigned(L, 1, "graph", "knn_eps", 1.0);
  if (knn > knn_cache)
    knn_cache = knn;
  uint64_t n_hops = tk_lua_foptunsigned(L, 1, "graph", "trans_hops", 0);
  uint64_t n_grow_pos = tk_lua_foptunsigned(L, 1, "graph", "trans_pos", 0);
  uint64_t n_grow_neg = tk_lua_foptunsigned(L, 1, "graph", "trans_neg", 0);
  bool do_mst = tk_lua_foptboolean(L, 1, "graph", "mst", true);

  unsigned int n_threads = tk_threads_getn(L, 1, "graph", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  // Init graph
  tk_graph_t *graph = tm_graph_create(L, pos, neg, labels, inv, ann, hbi, knn_cache, knn_eps, n_threads);
  int Gi = tk_lua_absindex(L, -1);

  // Query hoods from inv index
  tm_setup_hoods(L, Gi, graph);
  graph->uid_hood = tk_iumap_create();
  for (uint64_t i = 0; i < graph->uids->n; i ++) {
    khi = tk_iumap_put(graph->uid_hood, graph->uids->a[i], &kha);
    tk_iumap_value(graph->uid_hood, khi) = (int64_t) i;
  }

  tk_dsu_init(L, &graph->dsu, graph->uids);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_pos);
    lua_pushinteger(L, (int64_t) graph->n_neg);
    lua_pushstring(L, "init");
    lua_call(L, 4, 0);
  }

  // Add base pairs, setup dsu
  tm_add_pairs(L, Gi, graph);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
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
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
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
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
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
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_pos);
    lua_pushinteger(L, (int64_t) graph->n_neg);
    lua_pushstring(L, "transitives");
    lua_call(L, 4, 0);
  }

  if (graph->dsu.components > 1)
    tm_bridge_islands(graph);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_pos);
    lua_pushinteger(L, (int64_t) graph->n_neg);
    lua_pushstring(L, "bridge");
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
  tk_pvec_t *pos = tk_pvec_create(L, 0, 0, 0);
  tk_pvec_t *neg = tk_pvec_create(L, 0, 0, 0);
  tm_render_pairs(L, graph, pos, neg);
  return 2;
}

static luaL_Reg tm_graph_mt_fns[] =
{
  { "pairs", tm_graph_pairs },
  { NULL, NULL }
};

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  tk_pvec_t *pos,
  tk_pvec_t *neg,
  tk_ivec_t *labels,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  uint64_t knn_cache,
  double knn_eps,
  unsigned int n_threads
) {
  tk_graph_t *graph = tk_lua_newuserdata(L, tk_graph_t, TK_GRAPH_MT, tm_graph_mt_fns, tm_graph_gc); // ud
  graph->threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
  memset(graph->threads, 0, n_threads * sizeof(tk_graph_thread_t));
  graph->pool = tk_threads_create(L, n_threads, tk_graph_worker);
  graph->knn_cache = knn_cache;
  graph->knn_eps = knn_eps;
  graph->pairs = kh_init(pairs);
  if (inv != NULL)
    graph->inv = inv;
  else if (ann != NULL)
    graph->ann = ann;
  else if (hbi != NULL)
    graph->hbi = hbi;
  else
    assert(false);
  graph->pos = pos;
  graph->neg = neg;
  graph->labels = labels;
  graph->n_pos = pos == NULL ? 0 : pos->n;
  graph->n_neg = neg == NULL ? 0 : neg->n;
  for (unsigned int i = 0; i < n_threads; i ++) {
    tk_graph_thread_t *data = graph->threads + i;
    graph->pool->threads[i].data = data;
    data->graph = graph;
  }
  return graph;
}

static luaL_Reg tm_graph_fns[] =
{
  { "create", tm_create },
  { "to_bits", tm_to_bits },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_graph (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_graph_fns, 0);
  return 1;
}
