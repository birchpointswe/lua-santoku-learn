#include <santoku/tsetlin/graph.h>
#include <santoku/cvec.h>

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  tk_pvec_t *edges,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  double weight_eps,
  int64_t sigma_k,
  double sigma_scale,
  uint64_t knn,
  uint64_t knn_min,
  uint64_t knn_cache,
  double knn_eps,
  bool knn_mutual,
  int64_t knn_rank,
  bool bridge,
  uint64_t probe_radius,
  tk_ivec_sim_type_t cmp,
  double cmp_alpha,
  double cmp_beta
);

static inline void tk_graph_add_adj (
  tk_graph_t *graph,
  int64_t u,
  int64_t v
) {
  int kha;
  uint32_t khi;
  khi = tk_iumap_get(graph->uids_idx, u);
  if (khi == tk_iumap_end(graph->uids_idx))
    return;
  int64_t iu = tk_iumap_val(graph->uids_idx, khi);
  khi = tk_iumap_get(graph->uids_idx, v);
  if (khi == tk_iumap_end(graph->uids_idx))
    return;
  int64_t iv = tk_iumap_val(graph->uids_idx, khi);
  tk_iuset_put(graph->adj->a[iu], iv, &kha);
  tk_iuset_put(graph->adj->a[iv], iu, &kha);
}

static inline void tm_add_knn (
  lua_State *L,
  tk_graph_t *graph
) {
  int kha;
  uint32_t khi;
  uint64_t knn = graph->knn;

  if (!graph->uids_hoods)
    return;

  if (graph->inv != NULL && graph->inv_hoods != NULL) {

    for (uint64_t hood_idx = 0; hood_idx < graph->uids_hoods->n && hood_idx < graph->inv_hoods->n; hood_idx++) {
      int64_t u = graph->uids_hoods->a[hood_idx];

      uint32_t u_khi = tk_iumap_get(graph->uids_idx, u);
      if (u_khi == tk_iumap_end(graph->uids_idx))
        continue;
      int64_t u_global = tk_iumap_val(graph->uids_idx, u_khi);

      tk_rvec_t *ns = graph->inv_hoods->a[hood_idx];
      uint64_t rem = knn;

      for (uint32_t j = 0; j < ns->n && rem; j++) {
        tk_rank_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        uint32_t v_khi = tk_iumap_get(graph->uids_idx, v);
        if (v_khi == tk_iumap_end(graph->uids_idx))
          continue;
        int64_t v_global = tk_iumap_val(graph->uids_idx, v_khi);
        tk_edge_t e = tk_edge(u, v, tk_graph_weight(graph, r.d, u_global, v_global));
        khi = tk_euset_put(graph->pairs, e, &kha);
        if (!kha)
          continue;
        tk_graph_add_adj(graph, u, v);
        tk_dsu_union(graph->dsu, u, v);
        graph->n_edges++;
        rem--;
      }
    }

  } else if (graph->ann != NULL && graph->ann_hoods != NULL) {

    for (uint64_t hood_idx = 0; hood_idx < graph->uids_hoods->n && hood_idx < graph->ann_hoods->n; hood_idx++) {
      int64_t u = graph->uids_hoods->a[hood_idx];

      uint32_t u_khi = tk_iumap_get(graph->uids_idx, u);
      if (u_khi == tk_iumap_end(graph->uids_idx))
        continue;
      int64_t u_global = tk_iumap_val(graph->uids_idx, u_khi);

      tk_pvec_t *ns = graph->ann_hoods->a[hood_idx];
      uint64_t rem = knn;

      for (uint32_t j = 0; j < ns->n && rem; j++) {
        tk_pair_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        uint32_t v_khi = tk_iumap_get(graph->uids_idx, v);
        if (v_khi == tk_iumap_end(graph->uids_idx))
          continue;
        int64_t v_global = tk_iumap_val(graph->uids_idx, v_khi);
        tk_edge_t e = tk_edge(u, v, tk_graph_weight(graph, (double) r.p / graph->ann->features, u_global, v_global));
        khi = tk_euset_put(graph->pairs, e, &kha);
        if (!kha)
          continue;
        tk_graph_add_adj(graph, u, v);
        tk_dsu_union(graph->dsu, u, v);
        graph->n_edges++;
        rem--;
      }
    }

  } else if (graph->hbi != NULL && graph->hbi_hoods != NULL) {

    for (uint64_t hood_idx = 0; hood_idx < graph->uids_hoods->n && hood_idx < graph->hbi_hoods->n; hood_idx++) {
      int64_t u = graph->uids_hoods->a[hood_idx];

      uint32_t u_khi = tk_iumap_get(graph->uids_idx, u);
      if (u_khi == tk_iumap_end(graph->uids_idx))
        continue;
      int64_t u_global = tk_iumap_val(graph->uids_idx, u_khi);

      tk_pvec_t *ns = graph->hbi_hoods->a[hood_idx];
      uint64_t rem = knn;

      for (uint32_t j = 0; j < ns->n && rem; j++) {
        tk_pair_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        uint32_t v_khi = tk_iumap_get(graph->uids_idx, v);
        if (v_khi == tk_iumap_end(graph->uids_idx))
          continue;
        int64_t v_global = tk_iumap_val(graph->uids_idx, v_khi);
        tk_edge_t e = tk_edge(u, v, tk_graph_weight(graph, (double) r.p / graph->hbi->features, u_global, v_global));
        khi = tk_euset_put(graph->pairs, e, &kha);
        if (!kha)
          continue;
        tk_graph_add_adj(graph, u, v);
        tk_dsu_union(graph->dsu, u, v);
        graph->n_edges++;
        rem--;
      }
    }
  }
}

static inline tk_evec_t *tm_mst_knn_candidates (
  lua_State *L,
  tk_graph_t *graph
) {
  tk_evec_t *all_candidates = tk_evec_create(0, 0, 0, 0);
  if (!graph->uids_hoods)
    return all_candidates;
  if (graph->inv == NULL && graph->ann == NULL && graph->hbi == NULL)
    return all_candidates;
  uint32_t khi;
  if (graph->inv != NULL && graph->inv_hoods != NULL) {

    for (uint64_t hood_idx = 0; hood_idx < graph->uids_hoods->n && hood_idx < graph->inv_hoods->n; hood_idx++) {
      int64_t u = graph->uids_hoods->a[hood_idx];
      tk_rvec_t *ns = graph->inv_hoods->a[hood_idx];
      int64_t cu = tk_dsu_find(graph->dsu, u);
      for (uint32_t j = 0; j < ns->m; j ++) { // NOTE: ns->m here since our index stores non-mutuals in the back of the array
        tk_rank_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        if (cu == tk_dsu_find(graph->dsu, v))
          continue;
        tk_edge_t e = tk_edge(u, v, 0); // 0 weight, since not actually stored
        khi = tk_euset_get(graph->pairs, e);
        if (khi != tk_euset_end(graph->pairs))
          continue;
        if (tk_evec_push(all_candidates, tk_edge(u, v, r.d)) != 0) {
          tk_evec_destroy(all_candidates);
          return NULL;
        }
      }
    }

  } else if (graph->ann != NULL && graph->ann_hoods != NULL) {

    for (uint64_t hood_idx = 0; hood_idx < graph->uids_hoods->n && hood_idx < graph->ann_hoods->n; hood_idx++) {
      int64_t u = graph->uids_hoods->a[hood_idx];
      tk_pvec_t *ns = graph->ann_hoods->a[hood_idx];
      int64_t cu = tk_dsu_find(graph->dsu, u);
      for (uint32_t j = 0; j < ns->m; j ++) { // NOTE: ns->m here since our index stores non-mutuals in the back of the array
        tk_pair_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        if (cu == tk_dsu_find(graph->dsu, v))
          continue;
        tk_edge_t e = tk_edge(u, v, 0); // 0 weight, since not actually stored
        khi = tk_euset_get(graph->pairs, e);
        if (khi != tk_euset_end(graph->pairs))
          continue;
        if (tk_evec_push(all_candidates, tk_edge(u, v, (double) r.p / (double) graph->ann->features)) != 0) {
          tk_evec_destroy(all_candidates);
          return NULL;
        }
      }
    }

  } else if (graph->hbi != NULL && graph->hbi_hoods != NULL) {

    for (uint64_t hood_idx = 0; hood_idx < graph->uids_hoods->n && hood_idx < graph->hbi_hoods->n; hood_idx++) {
      int64_t u = graph->uids_hoods->a[hood_idx];
      tk_pvec_t *ns = graph->hbi_hoods->a[hood_idx];
      int64_t cu = tk_dsu_find(graph->dsu, u);
      for (uint32_t j = 0; j < ns->m; j ++) { // NOTE: ns->m here since our index stores non-mutuals in the back of the array
        tk_pair_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        if (cu == tk_dsu_find(graph->dsu, v))
          continue;
        tk_edge_t e = tk_edge(u, v, 0); // 0 weight, since not actually stored
        khi = tk_euset_get(graph->pairs, e);
        if (khi != tk_euset_end(graph->pairs))
          continue;
        if (tk_evec_push(all_candidates, tk_edge(u, v, (double) r.p / (double) graph->hbi->features)) != 0) {
          tk_evec_destroy(all_candidates);
          return NULL;
        }
      }
    }
  }

  // Sort all by distance ascending (nearest in feature space)
  tk_evec_asc(all_candidates, 0, all_candidates->n);
  return all_candidates;
}

static inline void tm_add_mst (
  lua_State *L,
  int Gi,
  tk_graph_t *graph,
  tk_evec_t *candidates
) {
  if (candidates != NULL) {

    // Kruskal over candidates
    int kha;
    uint32_t khi;
    for (uint64_t i = 0; i < candidates->n && tk_dsu_components(graph->dsu) > 1; i ++) {
      tk_edge_t c = candidates->a[i];
      int64_t cu = tk_dsu_find(graph->dsu, c.u);
      int64_t cv = tk_dsu_find(graph->dsu, c.v);
      if (cu == cv)
        continue;
      int64_t iu = tk_iumap_val(graph->uids_idx, tk_iumap_get(graph->uids_idx, c.u));
      int64_t iv = tk_iumap_val(graph->uids_idx, tk_iumap_get(graph->uids_idx, c.v));
      tk_edge_t e = tk_edge(c.u, c.v, tk_graph_weight(graph, c.w, iu, iv));
      khi = tk_euset_put(graph->pairs, e, &kha);
      if (!kha)
        continue;
      tk_graph_add_adj(graph, c.u, c.v);
      tk_dsu_union(graph->dsu, c.u, c.v);
      graph->n_edges ++;
    }

  } else if (graph->bridge) {

    // Find highest degree component members
    tk_pumap_t *reps_comp = tk_pumap_create(L, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (int64_t idx = 0; idx < (int64_t) graph->uids->n; idx ++) {
      int64_t u = graph->uids->a[idx];
      int64_t comp = tk_dsu_find(graph->dsu, u);
      uint32_t kc;
      int is_new;
      int64_t deg = tk_iuset_size(graph->adj->a[idx]);
      kc = tk_pumap_put(reps_comp, comp, &is_new);
      if (is_new || deg > tk_pumap_val(reps_comp, kc).p)
        tk_pumap_setval(reps_comp, kc, tk_pair(idx, deg));
    }

    // Shuffled list of found representatives
    tk_pvec_t *centers = tk_pumap_values(L, reps_comp);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    assert(centers->n > 1);
    tk_pvec_shuffle(centers);

    // Connect in a ring
    int kha;
    for (int64_t i = centers->n > 2 ? 0 : 1; i < (int64_t) centers->n; i ++) {
      int64_t j = i == ((int64_t) centers->n - 1) ? 0 : i + 1;
      int64_t iu = centers->a[i].i;
      int64_t iv = centers->a[j].i;
      int64_t u = graph->uids->a[iu];
      int64_t v = graph->uids->a[iv];
      double d = tk_graph_distance(graph, u, v);
      if (d == DBL_MAX) {
        // Skip this edge if distance cannot be computed
        continue;
      }
      tk_edge_t e = tk_edge(u, v, tk_graph_weight(graph, d, iu, iv));
      tk_euset_put(graph->pairs, e, &kha);
      if (kha) {
        tk_graph_add_adj(graph, u, v);
        tk_dsu_union(graph->dsu, u, v);
        graph->n_edges ++;
      }
    }

  }
}

static inline void tm_init_uids (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  graph->uids = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  graph->uids_idx = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
}

static inline void tm_process_seed_edges (
  lua_State *L,
  tk_graph_t *graph
) {
  if (graph->edges == NULL || graph->edges->n == 0)
    return;
  int kha;
  for (uint64_t i = 0; i < graph->edges->n; i++) {
    int64_t u = graph->edges->a[i].i;
    int64_t v = graph->edges->a[i].p;
    uint32_t khi = tk_iumap_put(graph->uids_idx, u, &kha);
    if (kha) {
      tk_iumap_setval(graph->uids_idx, khi, (int64_t) graph->uids->n);
      if (tk_ivec_push(graph->uids, u) != 0) {
        tk_lua_verror(L, 2, "process_seed_edges", "allocation failed");
        return;
      }
    }
    khi = tk_iumap_put(graph->uids_idx, v, &kha);
    if (kha) {
      tk_iumap_setval(graph->uids_idx, khi, (int64_t) graph->uids->n);
      if (tk_ivec_push(graph->uids, v) != 0) {
        tk_lua_verror(L, 2, "process_seed_edges", "allocation failed");
        return;
      }
    }
  }

}

static inline void tm_add_seed_edges_immediate(
  lua_State *L,
  tk_graph_t *graph
) {
  if (graph->edges == NULL || graph->edges->n == 0)
    return;

  int kha;
  uint32_t khi;

  for (uint64_t i = 0; i < graph->edges->n; i++) {
    int64_t u = graph->edges->a[i].i;
    int64_t v = graph->edges->a[i].p;

    uint32_t khi_u = tk_iumap_get(graph->uids_idx, u);
    uint32_t khi_v = tk_iumap_get(graph->uids_idx, v);
    if (khi_u == tk_iumap_end(graph->uids_idx) ||
        khi_v == tk_iumap_end(graph->uids_idx)) {
      continue;
    }

    int64_t iu = tk_iumap_val(graph->uids_idx, khi_u);
    int64_t iv = tk_iumap_val(graph->uids_idx, khi_v);

    double d = tk_graph_distance(graph, u, v);
    if (d == DBL_MAX) {
      continue;
    }

    double w = tk_graph_weight(graph, d, iu, iv);

    tk_edge_t e = tk_edge(u, v, w);
    khi = tk_euset_put(graph->pairs, e, &kha);
    if (!kha) {
      continue;
    }

    tk_graph_add_adj(graph, u, v);
    tk_dsu_union(graph->dsu, u, v);
    graph->n_edges++;
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
  for (uint64_t i = 0; i < graph->uids->n; i ++) {
    graph->adj->a[i] = tk_iuset_create(L, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
  }
}

static inline void tm_adj_resize (
  lua_State *L,
  int Gi,
  tk_graph_t *graph
) {
  uint64_t old_size = graph->adj->n;
  uint64_t new_size = graph->uids->n;
  if (new_size <= old_size)
    return;
  tk_graph_adj_resize(graph->adj, new_size, true);
  for (uint64_t i = old_size; i < new_size; i++) {
    graph->adj->a[i] = tk_iuset_create(L, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
  }
}


static inline void tm_run_knn_queries (
  lua_State *L,
  int Gi,
  tk_graph_t *graph,
  uint64_t n_threads
) {
  bool have_index = graph->inv != NULL || graph->ann != NULL || graph->hbi != NULL;
  if (!graph->knn || !graph->knn_cache || !have_index)
    return;
  if (graph->inv != NULL) {
    tk_inv_neighborhoods(
      L, graph->inv, graph->knn_cache, 0.0, graph->knn_eps, 0, graph->cmp,
      graph->cmp_alpha, graph->cmp_beta, false, graph->knn_rank, n_threads,
      &graph->inv_hoods, &graph->uids_hoods);
  } else if (graph->ann != NULL) {
    tk_ann_neighborhoods(
      L, graph->ann, graph->knn_cache, graph->probe_radius, 0, (int64_t)(graph->ann->features * graph->knn_eps), 0,
      false, n_threads, &graph->ann_hoods, &graph->uids_hoods);
  } else if (graph->hbi != NULL) {
    tk_hbi_neighborhoods(
      L, graph->hbi, graph->knn_cache, 0, (uint64_t)(graph->hbi->features * graph->knn_eps), 0,
      false, n_threads, &graph->hbi_hoods, &graph->uids_hoods);
  }

  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1); // hoods
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -2); // uids_hoods
  lua_pop(L, 2);

  if (graph->knn_mutual) {
    if (graph->inv != NULL)
      tk_inv_mutualize(L, graph->inv, graph->inv_hoods, graph->uids_hoods, graph->knn_min, n_threads, NULL);
    else if (graph->ann != NULL)
      tk_ann_mutualize(L, graph->ann, graph->ann_hoods, graph->uids_hoods, graph->knn_min, n_threads, NULL);
    else if (graph->hbi != NULL)
      tk_hbi_mutualize(L, graph->hbi, graph->hbi_hoods, graph->uids_hoods, graph->knn_min, n_threads, NULL);
  }

  if (graph->uids_hoods) {
    graph->uids_idx_hoods = tk_iumap_from_ivec(L, graph->uids_hoods);
    if (!graph->uids_idx_hoods)
      tk_error(L, "graph_create: iumap_from_ivec failed", ENOMEM);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    int kha;
    for (uint64_t i = 0; i < graph->uids_hoods->n; i++) {
      int64_t uid = graph->uids_hoods->a[i];
      uint32_t khi = tk_iumap_put(graph->uids_idx, uid, &kha);
      if (kha) {
        tk_iumap_setval(graph->uids_idx, khi, (int64_t) graph->uids->n);
        if (tk_ivec_push(graph->uids, uid) != 0) {
          tk_lua_verror(L, 2, "graph_adj", "allocation failed");
          return;
        }
      }
    }
  }

}

static inline void tm_compute_sigma (
  lua_State *L,
  int Gi,
  tk_graph_t *graph,
  uint64_t n_threads
) {
  if (!graph->sigma_k || !graph->sigma_scale || graph->sigma_scale <= 0.0)
    return;

  tk_graph_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
  memset(threads, 0, n_threads * sizeof(tk_graph_thread_t));
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_graph_worker);
  for (unsigned int i = 0; i < n_threads; i++) {
    tk_graph_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->graph = graph;
  }

  graph->sigmas = tk_dvec_create(L, graph->uids->n, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  for (uint64_t i = 0; i < graph->uids->n; i++)
    graph->sigmas->a[i] = 1.0;
  for (unsigned int i = 0; i < n_threads; i++) {
    tk_graph_thread_t *data = threads + i;
    tk_thread_range(i, n_threads, graph->uids->n, &data->ifirst, &data->ilast);
    atomic_init(&data->has_error, false);
  }
  tk_threads_signal(pool, TK_GRAPH_SIGMA, 0);
  for (unsigned int i = 0; i < n_threads; i++) {
    if (atomic_load(&threads[i].has_error)) {
      tk_lua_verror(L, 2, "compute_sigma", "worker allocation failed");
      return;
    }
  }

  tk_threads_destroy(pool);
  free(threads);
}

static inline void tm_reweight_all_edges (
  lua_State *L,
  tk_graph_t *graph,
  uint64_t n_threads
) {
  if (!graph->sigmas || graph->sigmas->n == 0)
    return;

  // Create local threadpool and threads
  tk_graph_thread_t *threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
  memset(threads, 0, n_threads * sizeof(tk_graph_thread_t));
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_graph_worker);
  for (unsigned int i = 0; i < n_threads; i++) {
    tk_graph_thread_t *data = threads + i;
    pool->threads[i].data = data;
    data->graph = graph;
    tk_thread_range(i, n_threads, graph->uids->n, &data->ifirst, &data->ilast);
  }
  tk_threads_signal(pool, TK_GRAPH_REWEIGHT, 0);

  tk_threads_destroy(pool);
  free(threads);
}

static void tm_graph_destroy_internal (tk_graph_t *graph)
{
  if (graph->destroyed)
    return;
  graph->destroyed = true;
  tk_dsu_destroy(graph->dsu);
}

static inline int tm_graph_gc (lua_State *L)
{
  tk_graph_t *graph = tk_graph_peek(L, 1);
  tm_graph_destroy_internal(graph);
  return 0;
}

static inline int tm_adjacency (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

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
  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(cmpstr, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(cmpstr, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(cmpstr, "tversky"))
    cmp = TK_IVEC_TVERSKY;
  else if (!strcmp(cmpstr, "dice"))
    cmp = TK_IVEC_DICE;
  else
    tk_lua_verror(L, 3, "graph", "invalid comparator specified", cmpstr);

  double weight_eps = tk_lua_foptnumber(L, 1, "graph", "weight_eps", 1e-8);
  int64_t sigma_k = tk_lua_foptinteger(L, 1, "graph", "sigma_k", 0);
  double sigma_scale = tk_lua_foptnumber(L, 1, "graph", "sigma_scale", 1.0);

  uint64_t knn = tk_lua_foptunsigned(L, 1, "graph", "knn", 0);
  uint64_t knn_min = tk_lua_foptunsigned(L, 1, "graph", "knn_min", 0);
  uint64_t knn_cache = tk_lua_foptunsigned(L, 1, "graph", "knn_cache", 0);
  double knn_eps = tk_lua_foptposdouble(L, 1, "graph", "knn_eps", 1.0);
  bool knn_mutual = tk_lua_foptboolean(L, 1, "graph", "knn_mutual", false);
  int64_t knn_rank = tk_lua_foptinteger(L, 1, "graph", "knn_rank", -1);
  bool bridge = tk_lua_foptboolean(L, 1, "graph", "bridge", false);
  uint64_t probe_radius = tk_lua_foptunsigned(L, 1, "graph", "probe_radius", 3);
  if (knn > knn_cache)
    knn_cache = knn;

  unsigned int n_threads = tk_threads_getn(L, 1, "graph", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_graph_t *graph = tm_graph_create(
    L, edges, inv, ann, hbi, weight_eps, sigma_k, sigma_scale, knn, knn_min,
    knn_cache, knn_eps, knn_mutual, knn_rank, bridge, probe_radius, cmp, cmp_alpha, cmp_beta);
  int Gi = tk_lua_absindex(L, -1);

  tm_init_uids(L, Gi, graph);
  tm_process_seed_edges(L, graph);
  graph->dsu = tk_dsu_create(L, graph->uids);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);

  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t)graph->uids->n);
    lua_pushinteger(L, tk_dsu_components(graph->dsu));
    lua_pushinteger(L, 0);
    lua_pushstring(L, "init");
    lua_call(L, 4, 0);
  }

  tm_adj_init(L, Gi, graph);
  tm_add_seed_edges_immediate(L, graph);

  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, (int64_t)graph->uids->n);
    lua_pushinteger(L, tk_dsu_components(graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_edges);
    lua_pushstring(L, "seed");
    lua_call(L, 4, 0);
  }

  if (graph->knn) {
    uint64_t old_uid_count = graph->uids->n;
    tm_run_knn_queries(L, Gi, graph, n_threads);
    if (graph->uids->n > old_uid_count) {
      tk_dsu_destroy(graph->dsu);
      graph->dsu = tk_dsu_create(L, graph->uids);
      tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
      lua_pop(L, 1);
      tm_adj_resize(L, Gi, graph);
      tk_edge_t p;
      char c;
      kh_foreach(graph->pairs, p, c, ({
        tk_dsu_union(graph->dsu, p.u, p.v);
      }))
    }
    tm_add_knn(L, graph);
    tm_compute_sigma(L, Gi, graph, n_threads);
    tm_reweight_all_edges(L, graph, n_threads);
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t)graph->uids->n);
      lua_pushinteger(L, tk_dsu_components(graph->dsu));
      lua_pushinteger(L, (int64_t) graph->n_edges);
      lua_pushstring(L, "knn");
      lua_call(L, 4, 0);
    }
  }

  if (!graph->bridge && graph->dsu->components > 1) {

    tk_iumap_t *comp_sizes = tk_iumap_create(L, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (uint64_t i = 0; i < graph->uids->n; i++) {
      int64_t root = tk_dsu_findx(graph->dsu, (int64_t)i);
      tk_iumap_inc(comp_sizes, root);
    }
    int64_t max_size = 0;
    int64_t largest_root = -1;
    for (uint32_t k = tk_iumap_begin(comp_sizes); k != tk_iumap_end(comp_sizes); ++k) {
      if (!tk_iumap_exist(comp_sizes, k)) continue;
      int64_t size = tk_iumap_val(comp_sizes, k);
      if (size > max_size) {
        max_size = size;
        largest_root = tk_iumap_key(comp_sizes, k);
      }
    }
    graph->largest_component_root = largest_root;
    int64_t component_edges = 0;
    tk_edge_t p;
    char c;
    kh_foreach(graph->pairs, p, c, ({
      uint32_t ukhi = tk_iumap_get(graph->uids_idx, p.u);
      uint32_t vkhi = tk_iumap_get(graph->uids_idx, p.v);
      if (ukhi != tk_iumap_end(graph->uids_idx) && vkhi != tk_iumap_end(graph->uids_idx)) {
        int64_t uidx = tk_iumap_val(graph->uids_idx, ukhi);
        int64_t vidx = tk_iumap_val(graph->uids_idx, vkhi);
        if (tk_dsu_findx(graph->dsu, uidx) == largest_root &&
            tk_dsu_findx(graph->dsu, vidx) == largest_root) {
          component_edges++;
        }
      }
    }))
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, max_size);
      lua_pushinteger(L, 1);
      lua_pushinteger(L, component_edges);
      lua_pushstring(L, "largest");
      lua_call(L, 4, 0);
    }

  } else if (graph->bridge && graph->knn_cache && graph->dsu->components > 1) {

    tk_evec_t *cs = tm_mst_knn_candidates(L, graph);
    if (cs == NULL)
      tk_lua_verror(L, 2, "graph_adj", "allocation failed during candidate collection");
    tm_add_mst(L, Gi, graph, cs);
    tk_evec_destroy(cs);
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t)graph->uids->n);
      lua_pushinteger(L, tk_dsu_components(graph->dsu));
      lua_pushinteger(L, (int64_t) graph->n_edges);
      lua_pushstring(L, "kruskal");
      lua_call(L, 4, 0);
    }

  }

  if (graph->bridge && graph->dsu->components > 1) {
    tm_add_mst(L, Gi, graph, NULL);
    if (i_each != -1) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (int64_t)graph->uids->n);
      lua_pushinteger(L, tk_dsu_components(graph->dsu));
      lua_pushinteger(L, (int64_t) graph->n_edges);
      lua_pushstring(L, "bridge");
      lua_call(L, 4, 0);
    }
  }
  tk_ivec_t *selected_nodes = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  tk_iumap_t *selected_set = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  int kha;
  uint32_t khi;

  if (graph->largest_component_root != -1) {
    for (uint64_t old_idx = 0; old_idx < graph->uids->n; old_idx++) {
      if (tk_dsu_findx(graph->dsu, (int64_t)old_idx) == graph->largest_component_root) {
        khi = tk_iumap_put(selected_set, (int64_t)old_idx, &kha);
        tk_iumap_setval(selected_set, khi, (int64_t)selected_nodes->n);
        if (tk_ivec_push(selected_nodes, (int64_t)old_idx) != 0) {
          tk_lua_verror(L, 2, "graph_adj", "allocation failed");
          return 0;
        }
      }
    }
  } else {
    for (uint64_t old_idx = 0; old_idx < graph->uids->n; old_idx++) {
      khi = tk_iumap_put(selected_set, (int64_t)old_idx, &kha);
      tk_iumap_setval(selected_set, khi, (int64_t)selected_nodes->n);
      if (tk_ivec_push(selected_nodes, (int64_t)old_idx) != 0) {
        tk_lua_verror(L, 2, "graph_adj", "allocation failed");
        return 0;
      }
    }
  }

  uint64_t n_nodes = selected_nodes->n;
  tk_ivec_t *tmp_offset = tk_ivec_create(L, n_nodes + 1, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  tk_ivec_t *tmp_data = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  tk_dvec_t *tmp_weights = tk_dvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  tmp_offset->a[0] = 0;
  for (uint64_t i = 0; i < n_nodes; i++) {
    int64_t old_idx = selected_nodes->a[i];
    int64_t u_uid = graph->uids->a[old_idx];
    tk_ivec_t *neighbors = tk_iuset_keys(L, graph->adj->a[old_idx]);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    tk_ivec_asc(neighbors, 0, neighbors->n);
    for (uint64_t j = 0; j < neighbors->n; j++) {
      int64_t neighbor_old_idx = neighbors->a[j];
      uint32_t khi = tk_iumap_get(selected_set, neighbor_old_idx);
      if (khi != tk_iumap_end(selected_set)) {
        int64_t neighbor_new_idx = tk_iumap_val(selected_set, khi);
        int64_t v_uid = graph->uids->a[neighbor_old_idx];
        double w = tk_graph_get_weight(graph, u_uid, v_uid);
        if (tk_ivec_push(tmp_data, neighbor_new_idx) != 0) {
          tk_lua_verror(L, 2, "graph_adj", "allocation failed");
          return 0;
        }
        if (tk_dvec_push(tmp_weights, w) != 0) {
          tk_lua_verror(L, 2, "graph_adj", "allocation failed");
          return 0;
        }
      }
    }
    tmp_offset->a[i + 1] = (int64_t) tmp_data->n;
  }

  tmp_offset->n = n_nodes + 1;
  tk_ivec_t *old_to_new = tk_ivec_create(L, n_nodes, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  tk_ivec_t *new_to_old = tk_ivec_create(L, n_nodes, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  for (uint64_t i = 0; i < n_nodes; i++)
    old_to_new->a[i] = -1;
  int64_t min_deg = INT64_MAX;
  int64_t start = 0;
  for (uint64_t i = 0; i < n_nodes; i++) {
    int64_t deg = tmp_offset->a[i + 1] - tmp_offset->a[i];
    if (deg < min_deg && deg > 0) {
      min_deg = deg;
      start = (int64_t) i;
    }
  }
  tk_ivec_t *queue = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
  lua_pop(L, 1);
  uint64_t new_idx = 0;

  if (tk_ivec_push(queue, start) != 0) {
    tk_lua_verror(L, 2, "graph_adj", "allocation failed");
    return 0;
  }
  old_to_new->a[start] = (int64_t) new_idx;
  new_to_old->a[new_idx] = start;
  new_idx++;

  for (uint64_t q_idx = 0; q_idx < queue->n; q_idx++) {
    int64_t u = queue->a[q_idx];
    int64_t edge_start = tmp_offset->a[u];
    int64_t edge_end = tmp_offset->a[u + 1];
    tk_pvec_t *neighbors = tk_pvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (int64_t e = edge_start; e < edge_end; e++) {
      int64_t v = tmp_data->a[e];
      if (old_to_new->a[v] == -1) {
        int64_t deg = tmp_offset->a[v + 1] - tmp_offset->a[v];
        if (tk_pvec_push(neighbors, tk_pair(v, deg)) != 0) {
          tk_lua_verror(L, 2, "graph_adj", "allocation failed");
          return 0;
        }
      }
    }
    tk_pvec_asc(neighbors, 0, neighbors->n);
    for (uint64_t i = 0; i < neighbors->n; i++) {
      int64_t v = neighbors->a[i].i;
      old_to_new->a[v] = (int64_t) new_idx;
      new_to_old->a[new_idx] = v;
      new_idx++;
      if (tk_ivec_push(queue, v) != 0) {
        tk_lua_verror(L, 2, "graph_adj", "allocation failed");
        return 0;
      }
    }
  }

  for (uint64_t i = 0; i < n_nodes; i++) {
    if (old_to_new->a[i] == -1) {
      old_to_new->a[i] = (int64_t) new_idx;
      new_to_old->a[new_idx] = (int64_t) i;
      new_idx++;
    }
  }
  for (uint64_t i = 0; i < n_nodes / 2; i++) {
    int64_t tmp = new_to_old->a[i];
    new_to_old->a[i] = new_to_old->a[n_nodes - 1 - i];
    new_to_old->a[n_nodes - 1 - i] = tmp;
  }
  for (uint64_t new_i = 0; new_i < n_nodes; new_i++) {
    int64_t old_i = new_to_old->a[new_i];
    old_to_new->a[old_i] = (int64_t) new_i;
  }
  tk_ivec_t *final_uids = tk_ivec_create(L, n_nodes, 0, 0); // uids
  tk_ivec_t *final_offset = tk_ivec_create(L, n_nodes + 1, 0, 0); // uids offset
  tk_ivec_t *final_data = tk_ivec_create(L, tmp_data->n, 0, 0);   // uids offset data
  tk_dvec_t *final_weights = tk_dvec_create(L, tmp_weights->n, 0, 0); // uids offset data weights

  final_offset->a[0] = 0;
  int64_t write = 0;
  for (uint64_t new_i = 0; new_i < n_nodes; new_i++) {
    int64_t old_i = new_to_old->a[new_i];
    int64_t original_idx = selected_nodes->a[old_i];
    final_uids->a[new_i] = graph->uids->a[original_idx];
    int64_t edge_start = tmp_offset->a[old_i];
    int64_t edge_end = tmp_offset->a[old_i + 1];
    for (int64_t e = edge_start; e < edge_end; e++) {
      int64_t old_neighbor = tmp_data->a[e];
      int64_t new_neighbor = old_to_new->a[old_neighbor];
      final_data->a[write] = new_neighbor;
      final_weights->a[write] = tmp_weights->a[e];
      write++;
    }
    final_offset->a[new_i + 1] = write;
  }

  final_uids->n = n_nodes;
  final_offset->n = n_nodes + 1;
  final_data->n = tmp_data->n;
  final_weights->n = tmp_weights->n;
  tm_graph_destroy_internal(graph);
  tk_lua_replace(L, 1, 4);
  lua_settop(L, 4);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 4;
}

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  tk_pvec_t *edges,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  double weight_eps,
  int64_t sigma_k,
  double sigma_scale,
  uint64_t knn,
  uint64_t knn_min,
  uint64_t knn_cache,
  double knn_eps,
  bool knn_mutual,
  int64_t knn_rank,
  bool bridge,
  uint64_t probe_radius,
  tk_ivec_sim_type_t cmp,
  double cmp_alpha,
  double cmp_beta
) {
  tk_graph_t *graph = tk_lua_newuserdata(L, tk_graph_t, TK_GRAPH_MT, NULL, tm_graph_gc);
  graph->edges = edges;
  graph->inv = inv;
  graph->cmp = cmp;
  graph->cmp_alpha = cmp_alpha;
  graph->cmp_beta = cmp_beta;
  graph->ann = ann;
  graph->hbi = hbi;
  graph->weight_eps = weight_eps;
  graph->sigma_k = sigma_k;
  graph->sigma_scale = sigma_scale;
  graph->knn = knn;
  graph->knn_min = knn_min;
  graph->knn_cache = knn_cache;
  graph->knn_eps = knn_eps;
  graph->knn_mutual = knn_mutual;
  graph->knn_rank = knn_rank;
  graph->bridge = bridge;
  graph->probe_radius = probe_radius;
  graph->largest_component_root = -1;
  graph->pairs = tk_euset_create(L, 0);
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, lua_gettop(L), -1);
  lua_pop(L, 1);
  graph->n_edges = 0;
  graph->uids_hoods = NULL;
  graph->uids_idx_hoods = NULL;
  graph->destroyed = false;
  return graph;
}

static inline int tm_adj_pairs (lua_State *L)
{
  lua_settop(L, 3);
  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  tk_pvec_t *pos = tk_pvec_peek(L, 2, "pos");
  tk_pvec_t *neg = tk_pvec_peek(L, 3, "neg");

  tk_ivec_t *offsets = NULL;
  tk_ivec_t *neighbors = NULL;
  tk_dvec_t *weights = NULL;

  int result = tk_graph_pairs_to_csr(ids, pos, neg, &offsets, &neighbors, &weights);
  if (result != 0)
    tk_lua_verror(L, 2, "adj_pairs", "failed to build CSR adjacency");

  lua_settop(L, 1);
  tk_ivec_register(L, offsets);
  tk_ivec_register(L, neighbors);
  tk_dvec_register(L, weights);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 4;
}

static inline int tm_star_hoods (lua_State *L)
{
  lua_settop(L, 3);  // ids, hoods, {threads}

  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  tk_inv_hoods_t *inv_hoods = tk_inv_hoods_peekopt(L, 2);
  tk_ann_hoods_t *ann_hoods = tk_ann_hoods_peekopt(L, 2);
  tk_hbi_hoods_t *hbi_hoods = tk_hbi_hoods_peekopt(L, 2);
  if (!inv_hoods && !ann_hoods && !hbi_hoods)
    tk_lua_verror(L, 2, "star_hoods", "hoods must be tk_inv_hoods_t, tk_ann_hoods_t, or tk_hbi_hoods_t");
  uint64_t n_hoods = inv_hoods ? inv_hoods->n : ann_hoods ? ann_hoods->n : hbi_hoods->n;
  if (n_hoods != ids->n)
    tk_lua_verror(L, 2, "star_hoods", "hoods size must match ids size");
  unsigned int n_threads = tk_threads_getn(L, 3, "threads", NULL);
  tk_pvec_t *pairs = NULL;
  if (tk_graph_star_hoods(L, ids, inv_hoods, ann_hoods, hbi_hoods, n_threads, &pairs) != 0)
    tk_lua_verror(L, 2, "star_hoods", "failed to convert hoods");
  lua_settop(L, 0);
  tk_pvec_register(L, pairs);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tm_anchor_pairs (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peek(L, -1, "ids");

  lua_getfield(L, 1, "labels");
  tk_ivec_t *labels = tk_ivec_peekopt(L, -1);

  uint64_t n_anchors = tk_lua_foptunsigned(L, 1, "anchor_pairs", "n_anchors", 1);

  tk_pvec_t *pairs = NULL;
  int result = tk_graph_anchor_pairs(ids, labels, n_anchors, &pairs);

  if (result != 0)
    tk_lua_verror(L, 2, "anchor_pairs", "failed to generate anchor pairs");

  lua_settop(L, 0);
  tk_pvec_register(L, pairs);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tm_random_pairs (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peek(L, -1, "ids");

  lua_getfield(L, 1, "labels");
  tk_ivec_t *labels = tk_ivec_peekopt(L, -1);

  uint64_t edges_per_node = tk_lua_foptunsigned(L, 1, "random_pairs", "edges_per_node", 3);
  unsigned int n_threads = tk_threads_getn(L, 1, "random_pairs", "threads");

  tk_pvec_t *pairs = NULL;
  int result = tk_graph_random_pairs(L, ids, labels, edges_per_node, n_threads, &pairs);

  if (result != 0)
    tk_lua_verror(L, 2, "random_pairs", "failed to generate random pairs");

  lua_settop(L, 0);
  tk_pvec_register(L, pairs);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tm_multiclass_pairs (lua_State *L)
{
  lua_settop(L, 7);

  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  tk_ivec_t *labels = tk_ivec_peek(L, 2, "labels");
  uint64_t n_anchors_pos = tk_lua_optunsigned(L, 3, "n_anchors_pos", 0);
  uint64_t n_anchors_neg = tk_lua_optunsigned(L, 4, "n_anchors_neg", 0);
  tk_inv_t *index = tk_inv_peekopt(L, 5);
  double eps_pos = tk_lua_optnumber(L, 6, "eps_pos", 0.0);
  double eps_neg = tk_lua_optnumber(L, 7, "eps_neg", 0.0);
  unsigned int n_threads = tk_threads_getn(L, 1, "multiclass_pairs", "threads");

  tk_pvec_t *pos = NULL;
  tk_pvec_t *neg = NULL;
  int result = tk_graph_multiclass_pairs(
    L, ids, labels, n_anchors_pos, n_anchors_neg, index, eps_pos, eps_neg,
    n_threads, &pos, &neg);

  if (result != 0)
    tk_lua_verror(L, 2, "multiclass_pairs", "failed to generate multiclass pairs");

  lua_settop(L, 1);
  tk_pvec_register(L, pos);
  tk_pvec_register(L, neg);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 3;
}

static luaL_Reg tm_graph_fns[] =
{
  { "adjacency", tm_adjacency },
  { "adj_pairs", tm_adj_pairs },
  { "star_hoods", tm_star_hoods },
  { "anchor_pairs", tm_anchor_pairs },
  { "random_pairs", tm_random_pairs },
  { "multiclass_pairs", tm_multiclass_pairs },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_graph (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_graph_fns, 0);
  return 1;
}
