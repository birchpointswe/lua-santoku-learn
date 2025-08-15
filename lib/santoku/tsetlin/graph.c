#include <santoku/tsetlin/graph.h>

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  tk_pvec_t *edges,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  double weight_eps,
  double flip_at,
  double neg_scale,
  int64_t sigma_k,
  double sigma_scale,
  uint64_t knn,
  uint64_t knn_cache,
  double knn_eps,
  tk_inv_cmp_type_t cmp,
  double cmp_alpha,
  double cmp_beta,
  unsigned int n_threads
);

static inline void tk_graph_worker (void *dp, int sig)
{
  tk_graph_stage_t stage = (tk_graph_stage_t) sig;
  tk_graph_thread_t *data = (tk_graph_thread_t *) dp;

  switch (stage) {

    case TK_GRAPH_CSR_OFFSET_LOCAL: {
      tk_graph_adj_item_t *adj = data->graph->adj->a;
      int64_t *adj_offset = data->adj_offset->a;
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      int64_t offset = 0;
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        adj_offset[i] = offset;
        offset += tk_iuset_size(adj[i]);
      }
      data->csr_base = offset;
      break;
    }

    case TK_GRAPH_CSR_OFFSET_GLOBAL: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      int64_t *adj_offset = data->adj_offset->a;
      int64_t csr_base = data->csr_base;
      for (uint64_t i = ifirst; i <= ilast; i ++)
        adj_offset[i] += csr_base;
      break;
    }

    case TK_GRAPH_CSR_DATA: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      tk_graph_adj_item_t *adj = data->graph->adj->a;
      int64_t *adj_data = data->adj_data->a;
      int64_t *adj_offset = data->adj_offset->a;
      double *adj_weights = data->adj_weights->a;
      tk_graph_t *graph = data->graph;
      int64_t *uids = graph->uids->a;
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        int64_t u = uids[i];
        int64_t write = adj_offset[i];
        int64_t iv, v;
        double w;
        tk_iuset_foreach(adj[i], iv, ({
          v = uids[iv];
          w = tk_graph_get_weight(graph, u, v);
          adj_data[write] = iv;
          adj_weights[write] = w;
          write ++;
        }))
      }
      break;
    }

    case TK_GRAPH_SIGMA: {
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      tk_graph_t *graph = data->graph;
      tk_dvec_t *sigma = graph->sigmas;
      int64_t sigma_k = graph->sigma_k;
      double sigma_scale = graph->sigma_scale;
      tk_ann_t *ann = graph->ann;
      tk_hbi_t *hbi = graph->hbi;
      tk_inv_hoods_t *inv_hoods = graph->inv_hoods;
      tk_ann_hoods_t *ann_hoods = graph->ann_hoods;
      tk_hbi_hoods_t *hbi_hoods = graph->hbi_hoods;
      uint64_t need_k = (sigma_k > 0) ? (uint64_t) sigma_k : 0;
      for (uint64_t i = ifirst; i <= ilast; i++) {
        double s = graph->weight_eps;
        if (inv_hoods && inv_hoods->a[i]->n > 0) {
          uint64_t seen = 0;
          double last = graph->weight_eps;
          for (uint64_t j = 0; j < inv_hoods->a[i]->n; j ++) {
            last = inv_hoods->a[i]->a[j].d;
            seen ++;
            if (need_k && seen == need_k) {
              s = last;
              break;
            }
          }
          if (seen && (!need_k || seen < need_k))
            s = last;
        } else if (ann_hoods && ann_hoods->a[i]->n > 0) {
          uint64_t seen = 0;
          double last = graph->weight_eps;
          double denom = (ann && ann->features) ? (double) ann->features : 1.0;
          for (uint64_t j = 0; j < ann_hoods->a[i]->n; j ++) {
            last = (double) ann_hoods->a[i]->a[j].p / denom;
            seen ++;
            if (need_k && seen == need_k) {
              s = last;
              break;
            }
          }
          if (seen && (!need_k || seen < need_k))
            s = last;
        } else if (hbi_hoods && hbi_hoods->a[i]->n > 0) {
          uint64_t seen = 0;
          double last = graph->weight_eps;
          double denom = (hbi && hbi->features) ? (double) hbi->features : 1.0;
          for (uint64_t j = 0; j < hbi_hoods->a[i]->n; j ++) {
            last = (double) hbi_hoods->a[i]->a[j].p / denom;
            seen ++;
            if (need_k && seen == need_k) {
              s = last;
              break;
            }
          }
          if (seen && (!need_k || seen < need_k))
            s = last;
        }
        sigma->a[i] = s * sigma_scale;
      }
      break;
    }

  }
}

static inline void tm_render_pairs (
  lua_State *L,
  tk_graph_t *graph,
  tk_pvec_t *edges,
  tk_dvec_t *weights
) {
  tm_pair_t p;
  char c;
  kh_foreach(graph->pairs, p,  c, ({
    tk_pvec_push(edges, tk_pair(p.u, p.v));
    tk_dvec_push(weights, p.w);
  }))
}

static inline void tk_graph_add_adj (
  tk_graph_t *graph,
  int64_t u,
  int64_t v
) {
  int kha;
  khint_t khi;
  khi = tk_iumap_get(graph->uids_idx, u);
  if (khi == tk_iumap_end(graph->uids_idx))
    return;
  int64_t iu = tk_iumap_value(graph->uids_idx, khi);
  khi = tk_iumap_get(graph->uids_idx, v);
  if (khi == tk_iumap_end(graph->uids_idx))
    return;
  int64_t iv = tk_iumap_value(graph->uids_idx, khi);
  tk_iuset_put(graph->adj->a[iu], iv, &kha);
  tk_iuset_put(graph->adj->a[iv], iu, &kha);
}

static inline double tk_graph_weight (
  const tk_graph_t *g,
  double base,
  int64_t iu,
  int64_t iv
) {
  const double eps = g->weight_eps;
  const double flip_at = g->flip_at;
  const double neg_scale = fabs(g->neg_scale);
  double b = base;
  if (isnan(b) || b == DBL_MAX) {
    b = 1.0;
  }
  if (b < 0.0) {
    b = 0.0;
  } else if (b > 1.0) {
    b = 1.0;
  }
  double sim;
  if (g->sigmas && g->sigmas->n) {
    double si = (iu >= 0 && (uint64_t) iu < g->sigmas->n) ? g->sigmas->a[iu] : eps;
    double sj = (iv >= 0 && (uint64_t) iv < g->sigmas->n) ? g->sigmas->a[iv] : eps;
    if (si <= 0.0) {
      si = eps;
    }
    if (sj <= 0.0) {
      sj = eps;
    }
    double s = sqrt(si * sj);
    if (g->sigma_scale > 0.0) {
      s = s * g->sigma_scale;
    }
    if (s > 0.0) {
      double s2 = s * s;
      sim = exp(-0.5 * (b * b) / s2);
    } else {
      sim = 1.0 - b;
    }
  } else {
    sim = 1.0 - b;
  }
  if (flip_at < 0.0 || b < flip_at) {
    if (sim < eps) {
      sim = eps;
    }
    if (sim > 1.0) {
      sim = 1.0;
    }
    return sim;
  }
  double fa = flip_at;
  if (fa > 1.0) {
    fa = 1.0;
  }
  const double denom = 1.0 - fa;
  double t = (denom > 0.0) ? (b - fa) / denom : 1.0;
  if (t < 0.0) {
    t = 0.0;
  } else if (t > 1.0) {
    t = 1.0;
  }
  const double ramp = 0.5 * (1.0 - cos(M_PI * t));
  double mag = neg_scale * ramp;
  if (eps > 0.0 && mag < eps) {
    mag = eps;
  }
  return -mag;
}

static inline double tk_graph_distance (
  tk_graph_t *graph,
  int64_t u,
  int64_t v
) {
  if (graph->inv != NULL) {

    size_t un;
    int64_t *uset = tk_inv_get(graph->inv, u, &un);
    if (uset == NULL)
      return DBL_MAX;
    size_t wn;
    int64_t *wset = tk_inv_get(graph->inv, v, &wn);
    if (wset == NULL)
      return DBL_MAX;
    return 1.0 - tk_inv_similarity(graph->inv, uset, un, wset, wn, graph->cmp, graph->cmp_alpha, graph->cmp_beta);

  } else if (graph->ann != NULL) {

    char *uset = tk_ann_get(graph->ann, u);
    if (uset == NULL)
      return DBL_MAX;
    char *wset = tk_ann_get(graph->ann, v);
    if (wset == NULL)
      return DBL_MAX;
    return (double) tk_ann_hamming((const unsigned char *) uset, (const unsigned char *) wset, graph->ann->features) / (double) graph->ann->features;

  } else if (graph->hbi != NULL) {

    char *uset = tk_hbi_get(graph->hbi, u);
    if (uset == NULL)
      return DBL_MAX;
    char *wset = tk_hbi_get(graph->hbi, v);
    if (wset == NULL)
      return DBL_MAX;
    return (double) tk_ann_hamming((const unsigned char *) uset, (const unsigned char *) wset, graph->hbi->features) / (double) graph->hbi->features;

  } else {
    return DBL_MAX;
  }
}

static inline void tm_add_knn (
  lua_State *L,
  tk_graph_t *graph
) {
  int kha;
  khint_t khi;
  uint64_t knn = graph->knn;

  // Add neighbors
  if (graph->inv != NULL && graph->inv_hoods != NULL) {

    for (uint64_t i = 0; i < graph->uids->n; i ++) {
      int64_t u = graph->uids->a[i];
      tk_rvec_t *ns = graph->inv_hoods->a[i];
      uint64_t rem = knn;
      for (khint_t j = 0; j < ns->n && rem; j ++) {
        tk_rank_t r = ns->a[j];
        int64_t v = graph->uids->a[r.i];
        tm_pair_t e = tm_pair(u, v, tk_graph_weight(graph, r.d, (int64_t) i, r.i));
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        tk_graph_add_adj(graph, u, v);
        tk_dsu_union(&graph->dsu, u, v);
        graph->n_edges ++;
        rem --;
      }
    }

  } else if (graph->ann != NULL && graph->ann_hoods != NULL) {

    for (uint64_t i = 0; i < graph->uids->n; i ++) {
      int64_t u = graph->uids->a[i];
      tk_pvec_t *ns = graph->ann_hoods->a[i];
      uint64_t rem = knn;
      for (khint_t j = 0; j < ns->n && rem; j ++) {
        tk_pair_t r = ns->a[j];
        int64_t v = graph->uids->a[r.i];
        tm_pair_t e = tm_pair(u, v, tk_graph_weight(graph, (double) r.p / graph->ann->features, (int64_t) i, r.i));
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        tk_graph_add_adj(graph, u, v);
        tk_dsu_union(&graph->dsu, u, v);
        graph->n_edges ++;
        rem --;
      }
    }

  } else if (graph->hbi != NULL && graph->hbi_hoods != NULL) {

    for (uint64_t i = 0; i < graph->uids->n; i ++) {
      int64_t u = graph->uids->a[i];
      tk_pvec_t *ns = graph->hbi_hoods->a[i];
      uint64_t rem = knn;
      for (khint_t j = 0; j < ns->n && rem; j ++) {
        tk_pair_t r = ns->a[j];
        int64_t v = graph->uids->a[r.i];
        tm_pair_t e = tm_pair(u, v, tk_graph_weight(graph, (double) r.p / graph->hbi->features, (int64_t) i, r.i));
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        tk_graph_add_adj(graph, u, v);
        tk_dsu_union(&graph->dsu, u, v);
        graph->n_edges ++;
        rem --;
      }
    }
  }
}

static inline tm_candidates_t tm_mst_knn_candidates (
  lua_State *L,
  tk_graph_t *graph
) {
  tm_candidates_t all_candidates;
  kv_init(all_candidates);
  if (graph->inv == NULL && graph->ann == NULL && graph->hbi == NULL)
    return all_candidates;

  khint_t khi;
  if (graph->inv != NULL) {

    for (uint64_t i = 0; i < graph->uids->n; i ++) {
      int64_t u = graph->uids->a[i];
      tk_rvec_t *ns = graph->inv_hoods->a[i];
      int64_t cu = tk_dsu_find(&graph->dsu, u);
      for (khint_t j = 0; j < ns->m; j ++) { // NOTE: ns->m here since our index stores non-mutuals in the back of the array
        tk_rank_t r = ns->a[j];
        int64_t v = graph->uids->a[r.i];
        if (cu == tk_dsu_find(&graph->dsu, v))
          continue;
        tm_pair_t e = tm_pair(u, v, 0); // 0 weight, since not actually stored
        khi = kh_get(pairs, graph->pairs, e);
        if (khi != kh_end(graph->pairs))
          continue;
        kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v, r.d));
      }
    }

  } else if (graph->ann != NULL) {

    for (uint64_t i = 0; i < graph->uids->n; i ++) {
      int64_t u = graph->uids->a[i];
      tk_pvec_t *ns = graph->ann_hoods->a[i];
      int64_t cu = tk_dsu_find(&graph->dsu, u);
      for (khint_t j = 0; j < ns->m; j ++) { // NOTE: ns->m here since our index stores non-mutuals in the back of the array
        tk_pair_t r = ns->a[j];
        int64_t v = graph->uids->a[r.i];
        if (cu == tk_dsu_find(&graph->dsu, v))
          continue;
        tm_pair_t e = tm_pair(u, v, 0); // 0 weight, since not actually stored
        khi = kh_get(pairs, graph->pairs, e);
        if (khi != kh_end(graph->pairs))
          continue;
        kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v, (double) r.p / (double) graph->ann->features));
      }
    }

  } else if (graph->hbi != NULL) {

    for (uint64_t i = 0; i < graph->uids->n; i ++) {
      int64_t u = graph->uids->a[i];
      tk_pvec_t *ns = graph->hbi_hoods->a[i];
      int64_t cu = tk_dsu_find(&graph->dsu, u);
      for (khint_t j = 0; j < ns->m; j ++) { // NOTE: ns->m here since our index stores non-mutuals in the back of the array
        tk_pair_t r = ns->a[j];
        int64_t v = graph->uids->a[r.i];
        if (cu == tk_dsu_find(&graph->dsu, v))
          continue;
        tm_pair_t e = tm_pair(u, v, 0); // 0 weight, since not actually stored
        khi = kh_get(pairs, graph->pairs, e);
        if (khi != kh_end(graph->pairs))
          continue;
        kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v, (double) r.p / (double) graph->hbi->features));
      }
    }
  }

  // Sort all by distance ascending (nearest in feature space)
  ks_introsort(candidates_asc, all_candidates.n, all_candidates.a);
  return all_candidates;
}

static inline void tm_add_mst (
  lua_State *L,
  tk_graph_t *graph,
  tm_candidates_t *candidatesp
) {
  if (candidatesp != NULL) {

    tm_candidates_t candidates = *candidatesp;

    // Kruskal over candidates
    int kha;
    khint_t khi;
    for (uint64_t i = 0; i < candidates.n && tk_dsu_components(&graph->dsu) > 1; i ++) {
      tm_candidate_t c = candidates.a[i];
      int64_t cu = tk_dsu_find(&graph->dsu, c.u);
      int64_t cv = tk_dsu_find(&graph->dsu, c.v);
      if (cu == cv)
        continue;
      int64_t iu = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, c.u));
      int64_t iv = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, c.v));
      tm_pair_t e = tm_pair(c.u, c.v, tk_graph_weight(graph, c.d, iu, iv));
      khi = kh_put(pairs, graph->pairs, e, &kha);
      if (!kha)
        continue;
      tk_graph_add_adj(graph, c.u, c.v);
      tk_dsu_union(&graph->dsu, c.u, c.v);
      graph->n_edges ++;
    }

  } else {

    // Find lowest degree component members
    tk_pumap_t *reps_comp = tk_pumap_create();
    for (int64_t idx = 0; idx < (int64_t) graph->uids->n; idx ++) {
      int64_t u = graph->uids->a[idx];
      int64_t comp = tk_dsu_find(&graph->dsu, u);
      khint_t kc;
      int is_new;
      int64_t deg = tk_iuset_size(graph->adj->a[idx]);
      kc = tk_pumap_put(reps_comp, comp, &is_new);
      if (is_new || deg > tk_pumap_value(reps_comp, kc).p)
        tk_pumap_value(reps_comp, kc) = tk_pair(idx, deg);
    }

    // Shuffled list of found representatives
    tk_pvec_t *centers = tk_pumap_values(L, reps_comp);
    assert(centers->n > 1);
    tk_pvec_shuffle(centers);
    tk_pumap_destroy(reps_comp);

    // Connect in a ring
    int kha;
    for (int64_t i = 0; i < (int64_t) centers->n; i ++) {
      int64_t j = i == ((int64_t) centers->n - 1) ? 0 : i + 1;
      int64_t iu = centers->a[i].i;
      int64_t iv = centers->a[j].i;
      int64_t u = graph->uids->a[iu];
      int64_t v = graph->uids->a[iv];
      double d = tk_graph_distance(graph, u, v);
      assert(d != DBL_MAX);
      tm_pair_t e = tm_pair(u, v, tk_graph_weight(graph, d, iu, iv));
      kh_put(pairs, graph->pairs, e, &kha);
      assert(kha);
      tk_graph_add_adj(graph, u, v);
      tk_dsu_union(&graph->dsu, u, v);
      graph->n_edges ++;
    }

    lua_pop(L, 1); // centers
  }
}

static inline void tm_adj_init (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  graph->adj = tk_graph_adj_create(L, graph->uids->n, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  for (uint64_t i = 0; i < graph->uids->n; i ++)
    graph->adj->a[i] = tk_iuset_create();
}

static inline void tm_add_pairs (
  lua_State *L,
  tk_graph_t *graph
) {
  int kha;
  khint_t khi;
  if (graph->edges != NULL) {
    uint64_t n_edges_old = graph->edges->n;
    uint64_t n_edges_new = 0;
    for (uint64_t i = 0; i < n_edges_old; i ++) {
      int64_t u = graph->edges->a[i].i;
      int64_t v = graph->edges->a[i].p;
      double d = tk_graph_distance(graph, u, v);
      if (d == DBL_MAX)
        continue;
      int64_t iu = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, u));
      int64_t iv = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, v));
      double w = tk_graph_weight(graph, d, iu, iv);
      khi = kh_put(pairs, graph->pairs, tm_pair(u, v, w), &kha);
      if (!kha)
        continue;
      tk_dsu_union(&graph->dsu, u, v);
      tk_graph_add_adj(graph, u, v);
      n_edges_new ++;
    }
    graph->n_edges = n_edges_new;
  }
}

static void tm_graph_destroy (tk_graph_t *graph)
{
  tk_dsu_free(&graph->dsu);
  tk_threads_destroy(graph->pool);
  if (graph->threads)
    free(graph->threads);
  if (graph->pairs)
    kh_destroy(pairs, graph->pairs);
  if (graph->uids_idx)
    tk_iumap_destroy(graph->uids_idx);
}

static inline int tm_graph_gc (lua_State *L)
{
  tk_graph_t *graph = tk_graph_peek(L, 1);
  tm_graph_destroy(graph);
  return 0;
}

static inline void tm_setup_hoods (lua_State *L, int Gi, tk_graph_t *graph)
{
  bool have_cache = graph->knn_cache > 0;
  bool want_sigma = graph->sigma_k > 0 && graph->sigma_scale > 0.0;
  if (!have_cache) {
    if (graph->inv != NULL)
      graph->uids = tk_inv_ids(L, graph->inv);
    else if (graph->ann != NULL)
      graph->uids = tk_ann_ids(L, graph->ann);
    else if (graph->hbi != NULL)
      graph->uids = tk_hbi_ids(L, graph->hbi);
    else {
      assert(false);
      return;
    }
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
  } else {
    if (graph->inv != NULL)
      tk_inv_neighborhoods(L, graph->inv, graph->knn_cache, graph->knn_eps, graph->cmp, graph->cmp_alpha, graph->cmp_beta, false, &graph->inv_hoods, &graph->uids);
    else if (graph->ann != NULL)
      tk_ann_neighborhoods(L, graph->ann, graph->knn_cache, graph->ann->features * graph->knn_eps, false, &graph->ann_hoods, &graph->uids);
    else if (graph->hbi != NULL)
      tk_hbi_neighborhoods(L, graph->hbi, graph->knn_cache, graph->hbi->features * graph->knn_eps, false, &graph->hbi_hoods, &graph->uids);
    else {
      assert(false);
      return;
    }
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -2);
    lua_pop(L, 2);
  }
  graph->uids_idx = tk_iumap_from_ivec(graph->uids);

  if (want_sigma) {
    graph->sigmas = tk_dvec_create(L, graph->uids->n, 0, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (unsigned int i = 0; i < graph->pool->n_threads; i ++) {
      tk_graph_thread_t *data = graph->threads + i;
      tk_thread_range(i, graph->pool->n_threads, graph->uids->n, &data->ifirst, &data->ilast);
    }
    tk_threads_signal(graph->pool, TK_GRAPH_SIGMA, 0);
  }
  if (have_cache) {
    if (graph->inv != NULL)
      tk_inv_mutualize(L, graph->inv, graph->inv_hoods, graph->uids);
    else if (graph->ann != NULL)
      tk_ann_mutualize(L, graph->ann, graph->ann_hoods, graph->uids);
    else if (graph->hbi != NULL)
      tk_hbi_mutualize(L, graph->hbi, graph->hbi_hoods, graph->uids);
    else {
      assert(false);
      return;
    }
  }
}

static inline int tm_create (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "edges");
  tk_pvec_t *edges = tk_pvec_peekopt(L, -1);

  lua_getfield(L, 1, "index");
  tk_inv_t *inv = tk_inv_peekopt(L, -1);
  tk_ann_t *ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *hbi = tk_hbi_peekopt(L, -1);
  bool have_index = inv != NULL || ann != NULL || hbi != NULL;
  if (!have_index)
    tk_lua_verror(L, 2, "index", "a tk_inv_t, tk_ann_t, or tk_hbi_t index must be provided");

  const char *cmpstr = tk_lua_foptstring(L, 1, "graph", "cmp", "jaccard");
  double cmp_alpha = tk_lua_foptnumber(L, 1, "graph", "cmp_alpha", 1.0);
  double cmp_beta = tk_lua_foptnumber(L, 1, "graph", "cmp_beta", 0.1);
  tk_inv_cmp_type_t cmp = TK_INV_JACCARD;
  if (!strcmp(cmpstr, "jaccard"))
    cmp = TK_INV_JACCARD;
  else if (!strcmp(cmpstr, "overlap"))
    cmp = TK_INV_OVERLAP;
  else if (!strcmp(cmpstr, "tversky"))
    cmp = TK_INV_TVERSKY;
  else if (!strcmp(cmpstr, "dice"))
    cmp = TK_INV_DICE;
  else
    tk_lua_verror(L, 3, "graph", "invalid comparator specified", cmpstr);

  double weight_eps = tk_lua_foptnumber(L, 1, "graph", "weight_eps", 1e-8);
  double flip_at = tk_lua_foptnumber(L, 1, "graph", "flip_at", -1.0);
  double neg_scale = tk_lua_foptnumber(L, 1, "graph", "neg_scale", 1.0);
  int64_t sigma_k = tk_lua_foptinteger(L, 1, "graph", "sigma_k", 0);
  double sigma_scale = tk_lua_foptnumber(L, 1, "graph", "sigma_scale", 1.0);

  uint64_t knn = tk_lua_foptunsigned(L, 1, "graph", "knn", 0);
  uint64_t knn_cache = tk_lua_foptunsigned(L, 1, "graph", "knn_cache", 0);
  double knn_eps = tk_lua_foptposdouble(L, 1, "graph", "knn_eps", 1.0);
  if (knn > knn_cache)
    knn_cache = knn;

  unsigned int n_threads = tk_threads_getn(L, 1, "graph", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_graph_t *graph = tm_graph_create(
    L, edges, inv, ann, hbi, weight_eps, flip_at, neg_scale, sigma_k,
    sigma_scale, knn, knn_cache, knn_eps, cmp, cmp_alpha, cmp_beta, n_threads);
  int Gi = tk_lua_absindex(L, -1);
  tm_setup_hoods(L, Gi, graph);

  // Setup DSU based on uids
  tk_dsu_init(&graph->dsu, graph->uids);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, 0);
    lua_pushinteger(L, (int64_t) graph->n_edges);
    lua_pushstring(L, "init");
    lua_call(L, 3, 0);
  }

  // Setup adjacency lists & add seed pairs
  tm_adj_init(L, Gi, graph);
  tm_add_pairs(L, graph);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_edges);
    lua_pushstring(L, "seed");
    lua_call(L, 3, 0);
  }

  // Add knn
  tm_add_knn(L, graph);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_edges);
    lua_pushstring(L, "knn");
    lua_call(L, 3, 0);
  }

  // Add mst
  if (graph->knn_cache && graph->dsu.components > 1) {
    tm_candidates_t cs = tm_mst_knn_candidates(L, graph);
    tm_add_mst(L, graph, &cs);
    kv_destroy(cs);
  }

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_edges);
    lua_pushstring(L, "kruskal");
    lua_call(L, 3, 0);
  }

  if (graph->dsu.components > 1)
    tm_add_mst(L, graph, NULL);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_edges);
    lua_pushstring(L, "bridge");
    lua_call(L, 3, 0);
  }

  // Return graph
  lua_pushvalue(L, Gi);
  return 1;
}

static inline int tm_graph_adjacency (lua_State *L)
{
  lua_settop(L, 1);
  tk_graph_t *graph = tk_graph_peek(L, 1);

  uint64_t n_nodes = graph->uids->n;
  tk_lua_get_ephemeron(L, TK_GRAPH_EPH, graph->uids); // uids

  // Setup threads
  tk_ivec_t *adj_offset = tk_ivec_create(L, n_nodes + 1, 0, 0); // uids, off
  tk_ivec_t *adj_data = tk_ivec_create(L, 0, 0, 0); // uids, off, data
  tk_dvec_t *adj_weights = tk_dvec_create(L, 0, 0, 0); // uids, off, data, weight
  for (unsigned int i = 0; i < graph->pool->n_threads; i ++) {
    tk_graph_thread_t *data = graph->threads + i;
    data->adj_offset = adj_offset;
    data->adj_data = adj_data;
    data->adj_weights = adj_weights;
    tk_thread_range(i, graph->pool->n_threads, n_nodes, &data->ifirst, &data->ilast);
  }

  // Populate thread-local offsets
  tk_threads_signal(graph->pool, TK_GRAPH_CSR_OFFSET_LOCAL, 0);

  // Push base offsets through thread range
  int64_t total = 0;
  for (unsigned int i = 0; i < graph->pool->n_threads; i ++) {
    tk_graph_thread_t *data = graph->threads + i;
    int64_t tmp = data->csr_base;
    data->csr_base = total;
    total += tmp;
  }
  adj_offset->a[adj_offset->n - 1] = total;

  // Make local offsets global
  tk_threads_signal(graph->pool, TK_GRAPH_CSR_OFFSET_GLOBAL, 0);
  tk_ivec_resize(adj_data, (size_t) total, true);
  tk_dvec_resize(adj_weights, (size_t) total, true);

  // Populate csr data
  tk_threads_signal(graph->pool, TK_GRAPH_CSR_DATA, 0);

  return 4; // uids, offset, data, weight
}

static inline int tm_graph_pairs (lua_State *L)
{
  lua_settop(L, 1);
  tk_graph_t *graph = tk_graph_peek(L, 1);
  tk_pvec_t *edges = tk_pvec_create(L, 0, 0, 0);
  tk_dvec_t *weights = tk_dvec_create(L, 0, 0, 0);
  tm_render_pairs(L, graph, edges, weights);
  return 2;
}

static luaL_Reg tm_graph_mt_fns[] =
{
  { "pairs", tm_graph_pairs },
  { "adjacency", tm_graph_adjacency },
  { NULL, NULL }
};

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  tk_pvec_t *edges,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  double weight_eps,
  double flip_at,
  double neg_scale,
  int64_t sigma_k,
  double sigma_scale,
  uint64_t knn,
  uint64_t knn_cache,
  double knn_eps,
  tk_inv_cmp_type_t cmp,
  double cmp_alpha,
  double cmp_beta,
  unsigned int n_threads
) {
  tk_graph_t *graph = tk_lua_newuserdata(L, tk_graph_t, TK_GRAPH_MT, tm_graph_mt_fns, tm_graph_gc); // ud
  graph->threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
  memset(graph->threads, 0, n_threads * sizeof(tk_graph_thread_t));
  graph->pool = tk_threads_create(L, n_threads, tk_graph_worker);
  graph->edges = edges;
  graph->inv = inv;
  graph->cmp = cmp;
  graph->cmp_alpha = cmp_alpha;
  graph->cmp_beta = cmp_beta;
  graph->ann = ann;
  graph->hbi = hbi;
  graph->weight_eps = weight_eps;
  graph->flip_at = flip_at;
  graph->neg_scale = neg_scale;
  graph->sigma_k = sigma_k;
  graph->sigma_scale = sigma_scale;
  graph->knn = knn;
  graph->knn_cache = knn_cache;
  graph->knn_eps = knn_eps;
  graph->pairs = kh_init(pairs);
  graph->n_edges = edges == NULL ? 0 : edges->n;
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
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_graph (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_graph_fns, 0);
  return 1;
}
