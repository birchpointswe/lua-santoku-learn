#include <santoku/tsetlin/graph.h>

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  int i_ids, tk_ivec_t *ids,
  tk_pvec_t *pos,
  tk_pvec_t *neg,
  tk_ivec_t *labels,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  uint64_t knn_cache,
  double knn_eps,
  double pos_scale,
  double neg_scale,
  double pos_default,
  double neg_default,
  double pos_sigma_scale,
  double neg_sigma_scale,
  int64_t sigma_k,
  // int i_types, tk_ivec_t *types,
  // int i_type_ranks, tk_ivec_t *type_ranks,
  // int64_t type_sigma_k,
  double bridge_density,
  double weight_eps,
  bool no_label_is_match,
  unsigned int n_threads
);

static inline bool tk_graph_same_label (tk_graph_t *graph, int64_t u, int64_t v) {
  if (!graph->labels)
    return false;
  khint_t ku = tk_iumap_get(graph->uids_idx, u);
  khint_t kv = tk_iumap_get(graph->uids_idx, v);
  if (ku == tk_iumap_end(graph->uids_idx) || kv == tk_iumap_end(graph->uids_idx))
    return false;
  int64_t iu = tk_iumap_value(graph->uids_idx, ku);
  int64_t iv = tk_iumap_value(graph->uids_idx, kv);
  if (iu < 0 || iv < 0)
    return false;
  if (iu >= (int64_t)graph->labels->n || iv >= (int64_t)graph->labels->n)
    return false;
  int64_t lu = graph->labels->a[iu];
  int64_t lv = graph->labels->a[iv];
  if (lu == -1 || lv == -1)
    return graph->no_label_is_match;
  return lu == lv;
}

static inline void tk_graph_worker (void *dp, int sig)
{
  tk_graph_stage_t stage = (tk_graph_stage_t) sig;
  tk_graph_thread_t *data = (tk_graph_thread_t *) dp;

  switch (stage) {

    case TK_GRAPH_CSR_OFFSET_LOCAL: {
      tk_graph_adj_item_t *adj_pos = data->graph->adj_pos->a;
      tk_graph_adj_item_t *adj_neg = data->graph->adj_neg->a;
      int64_t *adj_offset = data->adj_offset->a;
      uint64_t ifirst = data->ifirst;
      uint64_t ilast = data->ilast;
      int64_t offset = 0;
      for (uint64_t i = ifirst; i <= ilast; i ++) {
        adj_offset[i] = offset;
        int64_t deg_pos = tk_iuset_size(adj_pos[i]);
        int64_t deg_neg = tk_iuset_size(adj_neg[i]);
        offset += deg_pos + deg_neg;
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
      tk_graph_adj_item_t *adj_pos = data->graph->adj_pos->a;
      tk_graph_adj_item_t *adj_neg = data->graph->adj_neg->a;
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
        tk_iuset_foreach(adj_pos[i], iv, ({
          v = uids[iv];
          w = tk_graph_get_weight(graph, u, v);
          adj_data[write] = iv;
          adj_weights[write] = w;
          write ++;
        }))
        tk_iuset_foreach(adj_neg[i], iv, ({
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
      tk_ivec_t *uids_hoods = graph->uids_hoods;
      tk_dvec_t *sigma = graph->sigmas;
      int64_t sigma_k = graph->sigma_k;
      tk_ann_t *ann = graph->ann;
      tk_hbi_t *hbi = graph->hbi;
      tk_inv_hoods_t *inv_hoods = graph->inv_hoods;
      tk_ann_hoods_t *ann_hoods = graph->ann_hoods;
      tk_hbi_hoods_t *hbi_hoods = graph->hbi_hoods;
      uint64_t need_k = (sigma_k > 0) ? (uint64_t) sigma_k : 0;
      for (uint64_t i = ifirst; i <= ilast; i++) {
        int64_t u = uids_hoods->a[i];
        if (tk_iumap_get(graph->uids_idx, u) == tk_iumap_end(graph->uids_idx))
          continue;
        double s = graph->weight_eps;
        if (inv_hoods && inv_hoods->a[i]->n > 0) {
          uint64_t seen = 0;
          double last = graph->weight_eps;
          for (uint64_t j = 0; j < inv_hoods->a[i]->n; j ++) {
            int64_t v = uids_hoods->a[inv_hoods->a[i]->a[j].i];
            if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
              continue;
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
            int64_t v = uids_hoods->a[ann_hoods->a[i]->a[j].i];
            if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
              continue;
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
            int64_t v = uids_hoods->a[hbi_hoods->a[i]->a[j].i];
            if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
              continue;
            last = (double)hbi_hoods->a[i]->a[j].p / denom;
            seen ++;
            if (need_k && seen == need_k) {
              s = last;
              break;
            }
          }
          if (seen && (!need_k || seen < need_k))
            s = last;
        }
        sigma->a[i] = s;
      }
      break;
    }

  }
}

static inline void tm_render_pairs (
  lua_State *L,
  tk_graph_t *graph,
  tk_pvec_t *pos,
  tk_pvec_t *neg,
  tk_dvec_t *wpos,
  tk_dvec_t *wneg
) {
  bool l;
  tm_pair_t p;
  kh_foreach(graph->pairs, p,  l, ({
    tk_pvec_push(l ? pos : neg, tk_pair(p.u, p.v));
    tk_dvec_push(l ? wpos : wneg, p.w);
  }))
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
  khi = tk_iumap_get(graph->uids_idx, u);
  if (khi == tk_iumap_end(graph->uids_idx))
    return;
  int64_t iu = tk_iumap_value(graph->uids_idx, khi);
  khi = tk_iumap_get(graph->uids_idx, v);
  if (khi == tk_iumap_end(graph->uids_idx))
    return;
  int64_t iv = tk_iumap_value(graph->uids_idx, khi);
  tk_iuset_put(adj->a[iu], iv, &kha);
  tk_iuset_put(adj->a[iv], iu, &kha);
}

static inline double tk_graph_weight(
  const tk_graph_t *g,
  double base,
  bool is_pos,
  int64_t src_idx,
  int64_t nbr_idx
) {
  const double def_base = is_pos ? g->pos_default : g->neg_default;
  const double scale = is_pos ? g->pos_scale : g->neg_scale;
  const double sigma_scale = is_pos ? g->pos_sigma_scale : g->neg_sigma_scale;
  const double eps = g->weight_eps;
  double cap = fabs(scale);
  if (cap == 0.0) {
    return 0.0;
  }
  double b;
  if (base == DBL_MAX || isnan(base)) {
    b = def_base;
  } else if (base < 0.0) {
    b = eps;  /* synthetic fallback sentinel */
  } else {
    b = base;
  }
  if (b < 0.0) {
    b = 0.0;
  } else if (b > 1.0) {
    b = 1.0;
  }
  const double sign = (scale >= 0.0) ? 1.0 : -1.0;
  if (sigma_scale < 0.0) {
    double w = sign * cap;
    if (eps > 0.0 && fabs(w) < eps) {
      w = sign * fmin(eps, cap);
    }
    return w;
  }
  double sim;
  const bool have_sigmas = (g->sigmas && g->sigmas->n > 0);
  if (sigma_scale == 0.0 || !have_sigmas) {
    sim = 1.0 - b;
  } else {
    double si = (src_idx >= 0 && (uint64_t)src_idx < g->sigmas->n) ? g->sigmas->a[src_idx] : eps;
    double sj = (nbr_idx >= 0 && (uint64_t)nbr_idx < g->sigmas->n) ? g->sigmas->a[nbr_idx] : eps;
    if (si <= 0.0) {
      si = eps;
    }
    if (sj <= 0.0) {
      sj = eps;
    }
    double spair = sqrt(si * sj) * sigma_scale;
    if (spair > 0.0) {
      double s2 = spair * spair;
      double e1 = exp( -0.5 / s2 );
      double eb = exp( -0.5 * (b * b) / s2 );
      double denom = 1.0 - e1;
      if (denom > 0.0) {
        sim = (eb - e1) / denom;
      } else {
        sim = 1.0 - b;
      }
    } else {
      sim = 1.0 - b;
    }
  }
  double mag = (scale >= 0.0) ? sim : (1.0 - sim);
  double w = sign * mag * cap;
  if (eps > 0.0 && fabs(w) < eps) {
    w = sign * fmin(eps, cap);
  }
  if (w > cap) {
    w = cap;
  } else if (w < -cap) {
    w = -cap;
  }
  return w;
}

static inline double tk_graph_distance (
  tk_graph_t *graph,
  int64_t u,
  int64_t v,
  bool is_fallback
) {
  if (graph->inv != NULL) {

    size_t un;
    int64_t *uset = tk_inv_get(graph->inv, u, &un);
    if (uset == NULL)
      return is_fallback ? -1 : DBL_MAX;
    size_t wn;
    int64_t *wset = tk_inv_get(graph->inv, v, &wn);
    if (wset == NULL)
      return is_fallback ? -1 : DBL_MAX;
    return 1.0 - tk_inv_jaccard(uset, un, wset, wn);

  } else if (graph->ann != NULL) {

    char *uset = tk_ann_get(graph->ann, u);
    if (uset == NULL)
      return is_fallback ? -1 : DBL_MAX;
    char *wset = tk_ann_get(graph->ann, v);
    if (wset == NULL)
      return is_fallback ? -1 : DBL_MAX;
    return (double) tk_ann_hamming((const unsigned char *) uset, (const unsigned char *) wset, graph->ann->features) / (double) graph->ann->features;

  } else if (graph->hbi != NULL) {

    char *uset = tk_hbi_get(graph->hbi, u);
    if (uset == NULL)
      return is_fallback ? -1 : DBL_MAX;
    char *wset = tk_hbi_get(graph->hbi, v);
    if (wset == NULL)
      return is_fallback ? -1 : DBL_MAX;
    return (double) tk_ann_hamming((const unsigned char *) uset, (const unsigned char *) wset, graph->hbi->features) / (double) graph->hbi->features;

  } else {
    return is_fallback ? -1 : DBL_MAX;
  }
}

// static inline void tk_graph_reweight_types (
//   tk_graph_t *graph
// ) {
//   if (!graph->types || !graph->type_ranks || !graph->type_mass || graph->type_sigma_k < 1)
//     return;
//   int64_t max_rank = (int64_t) graph->type_ranks->n;
//   // maps rank to index
//   tk_iumap_t *rank = tk_iumap_from_ivec(graph->type_ranks);

//   khash_t(pairs) *agg = kh_init(pairs);
//   int kha;
//   khiter_t khi;

//   // Re-canonicalize by rank
//   for (khi = kh_begin(graph->type_mass); khi != kh_end(graph->type_mass); khi ++) {
//     if (!kh_exist(graph->type_mass, khi))
//       continue;
//     tm_pair_t k = kh_key(graph->type_mass, khi);
//     int64_t tu = k.u;
//     int64_t tv = k.v;
//     double B = k.w;
//     khint_t ruk = tk_iumap_get(rank, tu);
//     khint_t rvk = tk_iumap_get(rank, tv);
//     int64_t ru = ruk != tk_iumap_end(rank) ? tk_iumap_value(rank, ruk) : max_rank;
//     int64_t rv = rvk != tk_iumap_end(rank) ? tk_iumap_value(rank, rvk) : max_rank;
//     if (ru > rv || (ru == rv && tu > tv)) {
//       int64_t tmp_t = tu; tu = tv; tv = tmp_t;
//       int64_t tmp_r = ru; ru = rv; rv = tmp_r;
//     }
//     // Note, tm_pair(a, b, w) already canonicalizes such that pair.u is min(a,
//     // b). Are we doing something different with this?
//     tm_pair_t cp = (tm_pair_t) { tu, tv, 0.0 };
//     khint_t hc = kh_put(pairs, agg, cp, &kha);
//     if (kha)
//       kh_key(agg, hc).w = 0.0;
//     kh_key(agg, hc).w += B;
//   }

//   uint64_t m = kh_size(agg);

//   if (m < 2) {
//     tk_iumap_destroy(rank);
//     kh_destroy(pairs, agg);
//     return;
//   }

//   tk_graph_pair_recs_t *recs = tk_graph_pair_recs_create(0, m, 0, 0);
//   uint64_t idx = 0;
//   for (khi = kh_begin(agg); khi != kh_end(agg); khi ++) {
//     if (!kh_exist(agg, khi))
//       continue;
//     tm_pair_t k = kh_key(agg, khi);
//     int64_t tu = k.u;
//     int64_t tv = k.v;
//     khint_t ruk = tk_iumap_get(rank, tu);
//     khint_t rvk = tk_iumap_get(rank, tv);
//     int64_t ru = ruk != tk_iumap_end(rank) ? tk_iumap_value(rank, ruk) : max_rank;
//     int64_t rv = rvk != tk_iumap_end(rank) ? tk_iumap_value(rank, rvk) : max_rank;
//     recs->a[idx].tu = tu;
//     recs->a[idx].tv = tv;
//     recs->a[idx].ru = ru;
//     recs->a[idx].rv = rv;
//     recs->a[idx].B  = fabs(k.w);
//     idx ++;
//   }

//   tk_graph_pair_recs_asc(recs, 0, recs->n);

//   /* compute Gaussian targets with half-life K */
//   uint64_t K = (uint64_t) graph->type_sigma_k;
//   if (K < 1) {
//     K = 1;
//   }
//   if (K > m) {
//     K = m;
//   }
//   double sigma = (K > 1) ? ( (double)(K - 1) / sqrt(2.0 * log(2.0)) ) : 1e-12;

//   double *q = (double *) malloc(sizeof(double) * m);
//   double sum_q = 0.0;
//   for (uint64_t i = 0; i < m; i ++) {
//     double k0 = (double) i;
//     double qi = exp( - (k0 * k0) / (2.0 * sigma * sigma) );
//     q[i] = qi;
//     sum_q += qi;
//   }
//   double *p = (double *) malloc(sizeof(double) * m);
//   for (uint64_t i = 0; i < m; i ++)
//     p[i] = (sum_q > 0.0) ? (q[i] / sum_q) : (1.0 / (double) m);

//   khash_t(pairs) *betas = kh_init(pairs);
//   const double eps = 1e-12;
//   for (uint64_t i = 0; i < m; i ++) {
//     double B = recs->a[i].B;
//     double beta = p[i] / ((B > eps) ? B : eps);
//     tm_pair_t key = tm_pair(recs->a[i].tu, recs->a[i].tv, 0.0);
//     khiter_t hb = kh_put(pairs, betas, key, &kha);
//     if (kha) {
//       kh_key(betas, hb).w = 0.0;
//     }
//     kh_key(betas, hb).w = beta;
//   }

//   for (khi = kh_begin(graph->pairs); khi != kh_end(graph->pairs); khi ++) {
//     if (!kh_exist(graph->pairs, khi)) {
//       continue;
//     }
//     tm_pair_t *e = &kh_key(graph->pairs, khi);
//     bool is_pos = kh_value(graph->pairs, khi);

//     // map endpoints to types via uids_idx -> index into types
//     khiter_t hu = tk_iumap_get(graph->uids_idx, e->u);
//     khiter_t hv = tk_iumap_get(graph->uids_idx, e->v);
//     if (hu == tk_iumap_end(graph->uids_idx) || hv == tk_iumap_end(graph->uids_idx)) {
//       continue;
//     }
//     int64_t iu = tk_iumap_value(graph->uids_idx, hu);
//     int64_t iv = tk_iumap_value(graph->uids_idx, hv);

//     int64_t tu = graph->types->a[iu];
//     int64_t tv = graph->types->a[iv];
//     khint_t ruk = tk_iumap_get(rank, tu);
//     khint_t rvk = tk_iumap_get(rank, tv);
//     int64_t ru = ruk != tk_iumap_end(rank) ? tk_iumap_value(rank, ruk) : max_rank;
//     int64_t rv = rvk != tk_iumap_end(rank) ? tk_iumap_value(rank, rvk) : max_rank;

//     // canonicalize type pair for β lookup
//     int64_t clu = tu, clv = tv;
//     if (ru > rv || (ru == rv && tu > tv)) {
//       clu = tv;
//       clv = tu;
//     }

//     khiter_t hb = kh_get(pairs, betas, tm_pair(clu, clv, 0.0));
//     if (hb == kh_end(betas))
//       continue;
//     double beta = kh_key(betas, hb).w;

//     // scale existing absolute weight by β, clamp by family cap, keep sign
//     double cap = fabs(is_pos ? graph->pos_scale : graph->neg_scale);
//     if (cap == 0.0) {
//       e->w = 0.0;
//       continue;
//     }

//     double sign = (is_pos ? 1.0 : -1.0);
//     double mag  = fabs(e->w) * beta;
//     if (mag > cap) {
//       mag = cap;
//     }
//     e->w = sign * mag;
//   }

//   free(recs);
//   free(q);
//   free(p);
//   kh_destroy(pairs, betas);
//   kh_destroy(pairs, agg);
//   tk_iumap_destroy(rank);
// }

// static inline void tk_graph_add_type_mass (
//   tk_graph_t *graph,
//   int64_t iu,
//   int64_t iv,
//   double base_distance,
//   double def
// ) {
//   if (!graph->type_mass)
//     return;
//   if (base_distance == DBL_MAX)
//     base_distance = def;
//   else if (base_distance > 1.0 || base_distance < 0.0)
//     base_distance = 1.0;
//   double weight = 1.0 - base_distance;
//   khint_t khi;
//   int kha;
//   tm_pair_t p = tm_pair(graph->types->a[iu], graph->types->a[iv], 0);
//   khi = kh_put(pairs, graph->type_mass, p, &kha);
//   kh_key(graph->type_mass, khi).w += fabs(weight);
// }

static inline void tm_add_knn (
  lua_State *L,
  tk_graph_t *graph,
  uint64_t knn_pos,
  uint64_t knn_neg
) {
  int kha;
  khint_t khi;

  if (graph->inv == NULL && graph->ann == NULL && graph->hbi == NULL)
    return;

  // Prep shuffle
  tk_ivec_t *shuf = tk_ivec_create(L, graph->uids_hoods->n, 0, 0);
  tk_ivec_fill_indices(shuf);
  tk_ivec_shuffle(shuf);

  // Add neighbors
  if (graph->inv != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t i = shuf->a[su];
      int64_t u = graph->uids_hoods->a[i];
      if (tk_iumap_get(graph->uids_idx, u) == tk_iumap_end(graph->uids_idx))
        continue;
      tk_rvec_t *ns = graph->inv_hoods->a[i];
      uint64_t rem_pos = knn_pos;
      uint64_t rem_neg = knn_neg;
      for (khint_t j = 0; j < ns->n && (rem_pos || rem_neg); j ++) {
        tk_rank_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n || r.i < 0)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
          continue;
        bool w = graph->labels == NULL || tk_graph_same_label(graph, u, v);
        if (w && !rem_pos) continue;
        if (!w && !rem_neg) continue;
        // int64_t iu = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, u));
        // int64_t iv = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, v));
        tm_pair_t e = tm_pair(u, v, tk_graph_weight(graph, r.d, w, i, r.i));
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        kh_value(graph->pairs, khi) = w;
        // tk_graph_add_type_mass(graph, iu, iv, r.d, w ? graph->pos_default : graph->neg_default);
        tk_graph_add_adj(graph, u, v, w);
        if (w) {
          tk_dsu_union(&graph->dsu, u, v);
          graph->n_pos ++;
          rem_pos --;
        } else {
          tk_dsu_union(&graph->dsu, u, v);
          graph->n_neg ++;
          rem_neg --;
        }
      }
    }

  } else if (graph->ann != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t i = shuf->a[su];
      int64_t u = graph->uids_hoods->a[i];
      if (tk_iumap_get(graph->uids_idx, u) == tk_iumap_end(graph->uids_idx))
        continue;
      tk_pvec_t *ns = graph->ann_hoods->a[i];
      uint64_t rem_pos = knn_pos;
      uint64_t rem_neg = knn_neg;
      for (khint_t j = 0; j < ns->n && (rem_pos || rem_neg); j ++) {
        tk_pair_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n || r.i < 0)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
          continue;
        bool w = graph->labels == NULL || tk_graph_same_label(graph, u, v);
        if (w && !rem_pos) continue;
        if (!w && !rem_neg) continue;
        double d = (double) r.p / (double) graph->ann->features;
        double weight = tk_graph_weight(graph, d, w, i, r.i);
        // int64_t iu = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, u));
        // int64_t iv = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, v));
        tm_pair_t e = tm_pair(u, v, weight);
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        kh_value(graph->pairs, khi) = w;
        // tk_graph_add_type_mass(graph, iu, iv, d, w ? graph->pos_default : graph->neg_default);
        tk_graph_add_adj(graph, u, v, w);
        if (w) {
          tk_dsu_union(&graph->dsu, u, v);
          graph->n_pos ++;
          rem_pos --;
        } else {
          tk_dsu_union(&graph->dsu, u, v);
          graph->n_neg ++;
          rem_neg --;
        }
      }
    }

  } else if (graph->hbi != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t i = shuf->a[su];
      int64_t u = graph->uids_hoods->a[i];
      if (tk_iumap_get(graph->uids_idx, u) == tk_iumap_end(graph->uids_idx))
        continue;
      tk_pvec_t *ns = graph->hbi_hoods->a[i];
      uint64_t rem_pos = knn_pos;
      uint64_t rem_neg = knn_neg;
      for (khint_t j = 0; j < ns->n && (rem_pos || rem_neg); j ++) {
        tk_pair_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n || r.i < 0)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
          continue;
        bool w = graph->labels == NULL || tk_graph_same_label(graph, u, v);
        if (w && !rem_pos) continue;
        if (!w && !rem_neg) continue;
        double d = (double) r.p / (double) graph->hbi->features;
        double weight = tk_graph_weight(graph, d, w, i, r.i);
        // int64_t iu = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, u));
        // int64_t iv = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, v));
        tm_pair_t e = tm_pair(u, v, weight);
        khi = kh_put(pairs, graph->pairs, e, &kha);
        if (!kha)
          continue;
        kh_value(graph->pairs, khi) = w;
        // tk_graph_add_type_mass(graph, iu, iv, d, w ? graph->pos_default : graph->neg_default);
        tk_graph_add_adj(graph, u, v, w);
        if (w) {
          tk_dsu_union(&graph->dsu, u, v);
          graph->n_pos ++;
          rem_pos --;
        } else {
          tk_dsu_union(&graph->dsu, u, v);
          graph->n_neg ++;
          rem_neg --;
        }
      }
    }

  }

  // Cleanup
  lua_pop(L, 1);
}

static inline tm_candidates_t tm_mst_knn_candidates (
  lua_State *L,
  tk_graph_t *graph
) {
  tm_candidates_t all_candidates;
  kv_init(all_candidates);
  if (graph->inv == NULL && graph->ann == NULL && graph->hbi == NULL)
    return all_candidates;

  // Prep shuffle
  tk_ivec_t *shuf = tk_ivec_create(L, graph->uids_hoods->n, 0, 0);
  tk_ivec_fill_indices(shuf);
  tk_ivec_shuffle(shuf);

  khint_t khi;
  if (graph->inv != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t iu = shuf->a[su];
      int64_t u = graph->uids_hoods->a[iu];
      if (tk_iumap_get(graph->uids_idx, u) == tk_iumap_end(graph->uids_idx))
        continue;
      tk_rvec_t *ns = graph->inv_hoods->a[iu];
      int64_t cu = tk_dsu_find(&graph->dsu, u);
      for (khint_t j = 0; j < ns->n; j ++) {
        tk_rank_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n || r.i < 0)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
          continue;
        if (cu == tk_dsu_find(&graph->dsu, v))
          continue;
        tm_pair_t e = tm_pair(u, v, 0); // 0 weight, since not actually stored
        khi = kh_get(pairs, graph->pairs, e);
        if (khi != kh_end(graph->pairs))
          continue;
        // TODO: Can we use heap?
        kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v, r.d));
      }
    }

  } else if (graph->ann != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t iu = shuf->a[su];
      int64_t u = graph->uids_hoods->a[iu];
      if (tk_iumap_get(graph->uids_idx, u) == tk_iumap_end(graph->uids_idx))
        continue;
      tk_pvec_t *ns = graph->ann_hoods->a[iu];
      int64_t cu = tk_dsu_find(&graph->dsu, u);
      for (khint_t j = 0; j < ns->n; j ++) {
        tk_pair_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n || r.i < 0)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
          continue;
        if (cu == tk_dsu_find(&graph->dsu, v))
          continue;
        tm_pair_t e = tm_pair(u, v, 0); // 0 weight, since not actually stored
        khi = kh_get(pairs, graph->pairs, e);
        if (khi != kh_end(graph->pairs))
          continue;
        // TODO: Can we use heap?
        kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v, (double) r.p / (double) graph->ann->features));
      }
    }

  } else if (graph->hbi != NULL) {

    for (uint64_t su = 0; su < shuf->n; su ++) {
      int64_t iu = shuf->a[su];
      int64_t u = graph->uids_hoods->a[iu];
      if (tk_iumap_get(graph->uids_idx, u) == tk_iumap_end(graph->uids_idx))
        continue;
      tk_pvec_t *ns = graph->hbi_hoods->a[iu];
      int64_t cu = tk_dsu_find(&graph->dsu, u);
      for (khint_t j = 0; j < ns->n; j ++) {
        tk_pair_t r = ns->a[j];
        if (r.i >= (int64_t) graph->uids_hoods->n || r.i < 0)
          continue;
        int64_t v = graph->uids_hoods->a[r.i];
        if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
          continue;
        if (cu == tk_dsu_find(&graph->dsu, v))
          continue;
        tm_pair_t e = tm_pair(u, v, 0); // 0 weight, since not actually stored
        khi = kh_get(pairs, graph->pairs, e);
        if (khi != kh_end(graph->pairs))
          continue;
        // TODO: Can we use heap?
        kv_push(tm_candidate_t, all_candidates, tm_candidate(u, v, (double) r.p / (double) graph->hbi->features));
      }
    }

  }

  // Sort all by distance ascending (nearest in feature space)
  ks_introsort(candidates_asc, all_candidates.n, all_candidates.a);

  lua_pop(L, 1);
  return all_candidates;
}

static inline void tm_add_mst (
  lua_State *L,
  tk_graph_t *graph,
  tm_candidates_t *candidatesp
) {
  if (candidatesp != NULL) {
    tm_candidates_t candidates = *candidatesp;
    khint_t khi; int kha;
    for (uint64_t i = 0; i < candidates.n && tk_dsu_components(&graph->dsu) > 1; i ++) {
      tm_candidate_t c = candidates.a[i];
      int64_t cu = tk_dsu_find(&graph->dsu, c.u);
      int64_t cv = tk_dsu_find(&graph->dsu, c.v);
      if (cu == cv)
        continue;
      bool w = graph->labels == NULL || tk_graph_same_label(graph, c.u, c.v);
      int64_t iu = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, c.u));
      int64_t iv = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, c.v));
      double weight = tk_graph_weight(graph, c.d, w, iu, iv);
      tm_pair_t e = tm_pair(c.u, c.v, weight);
      khi = kh_put(pairs, graph->pairs, e, &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = w;
      // tk_graph_add_type_mass(graph, iu, iv, c.d, w ? graph->pos_default : graph->neg_default);
      tk_graph_add_adj(graph, c.u, c.v, w);
      tk_dsu_union(&graph->dsu, c.u, c.v);
      if (w)
        graph->n_pos++;
      else
        graph->n_neg++;
    }
  } else {
    /* ---------- reps_comp: lowest-degree representative per component ---------- */
    tk_pumap_t *reps_comp = tk_pumap_create();
    for (int64_t idx = 0; idx < (int64_t)graph->uids->n; idx++) {
      int64_t u = graph->uids->a[idx];
      int64_t comp = tk_dsu_find(&graph->dsu, u);
      int64_t deg = tk_iuset_size(graph->adj_pos->a[idx]) + tk_iuset_size(graph->adj_neg->a[idx]);
      int is_new; khint_t kc = tk_pumap_put(reps_comp, comp, &is_new);
      if (is_new || deg < tk_pumap_value(reps_comp, kc).p)
        tk_pumap_value(reps_comp, kc) = tk_pair(idx, deg);
    }
    tk_pvec_t *centers = tk_pumap_values(L, reps_comp);
    tk_pvec_asc(centers, 0, centers->n);
    tk_pumap_destroy(reps_comp);
    tk_pvec_shuffle(centers);

    /* ---------- ring + chords INSIDE each non-(-1) label (positive edges) ---------- */
    if (graph->labels && centers->n >= 2) {
      /* Count reps per label */
      tk_iumap_t *cnt = tk_iumap_create(); /* lbl -> count */
      for (uint64_t i = 0; i < centers->n; i++) {
        int64_t idx_i = centers->a[i].i;
        int64_t lbl   = graph->labels->a[idx_i];
        if (lbl < 0) continue; /* skip unlabeled in this phase */
        int is_new; khint_t kc = tk_iumap_put(cnt, lbl, &is_new);
        tk_iumap_value(cnt, kc) = is_new ? 1 : (tk_iumap_value(cnt, kc) + 1);
      }

      /* Build offsets for stable per-label slices */
      tk_iumap_t *off = tk_iumap_create();  /* lbl -> start offset */
      uint64_t total = centers->n;
      int64_t *buf = (int64_t *) malloc(total * sizeof(int64_t));
      uint64_t cur = 0;
      for (khint_t it = 0; it < kh_end(cnt); it++) {
        if (!kh_exist(cnt, it)) continue;
        int64_t lbl = kh_key(cnt, it);
        khint_t ko = tk_iumap_put(off, lbl, &(int){0});
        tk_iumap_value(off, ko) = (int64_t)cur;
        cur += (uint64_t)kh_val(cnt, it);
      }
      /* Scatter centers positions into per-label slices */
      tk_iumap_t *curm = tk_iumap_create(); /* lbl -> next write */
      for (khint_t it = 0; it < kh_end(off); it++) {
        if (!kh_exist(off, it)) continue;
        int64_t lbl = kh_key(off, it);
        khint_t kw = tk_iumap_put(curm, lbl, &(int){0});
        tk_iumap_value(curm, kw) = tk_iumap_value(off, it);
      }
      for (uint64_t i = 0; i < centers->n; i++) {
        int64_t idx_i = centers->a[i].i;
        int64_t lbl   = (graph->labels ? graph->labels->a[idx_i] : -1);
        if (lbl < 0) continue;
        khint_t kw = tk_iumap_get(curm, lbl);
        int64_t wpos = tk_iumap_value(curm, kw);
        buf[wpos] = (int64_t)i;
        tk_iumap_value(curm, kw) = wpos + 1;
      }
      /* For each label slice: ring + chords (POSITIVE) */
      for (khint_t it = 0; it < kh_end(cnt); it++) {
        if (!kh_exist(cnt, it)) continue;
        int64_t lbl = kh_key(cnt, it);
        int64_t n_l = kh_val(cnt, it);
        if (n_l <= 0) continue;
        int64_t base = tk_iumap_value(off, tk_iumap_get(off, lbl));

        if (n_l >= 2) {
          for (int64_t t = 0; t < n_l; t++) {
            int64_t pos_i = buf[base + t];
            int64_t pos_j = buf[base + ((t + 1) % n_l)];
            int64_t iu_uids = centers->a[pos_i].i;
            int64_t iv_uids = centers->a[pos_j].i;
            int64_t u = graph->uids->a[iu_uids];
            int64_t v = graph->uids->a[iv_uids];
            if (tk_dsu_find(&graph->dsu, u) != tk_dsu_find(&graph->dsu, v)) {
              double d = tk_graph_distance(graph, u, v, true);
              tm_pair_t e = tm_pair(u, v, tk_graph_weight(graph, d, true, -1, -1));
              int kha; khint_t kp = kh_put(pairs, graph->pairs, e, &kha);
              if (kha) {
                kh_value(graph->pairs, kp) = true;
                tk_graph_add_adj(graph, u, v, true);
                tk_dsu_union(&graph->dsu, u, v);
                graph->n_pos++;
              }
            }
          }
        }
        if (n_l >= 3) {
          uint64_t C = (uint64_t) llround(graph->bridge_density * (double)n_l);
          if (C > 0) {
            uint64_t step = (n_l / (int64_t)C) > 0 ? (uint64_t)(n_l / (int64_t)C) : 1;
            uint64_t jump = (uint64_t) floor(sqrt((double)n_l));
            if (jump < 2) jump = 2;
            for (uint64_t t = 0; t < C; t++) {
              uint64_t s  = (t * step) % (uint64_t)n_l;
              uint64_t s2 = (s + jump) % (uint64_t)n_l;
              int64_t pos_i = buf[base + (int64_t)s];
              int64_t pos_j = buf[base + (int64_t)s2];
              int64_t iu_uids = centers->a[pos_i].i;
              int64_t iv_uids = centers->a[pos_j].i;
              int64_t u = graph->uids->a[iu_uids];
              int64_t v = graph->uids->a[iv_uids];
              double d = tk_graph_distance(graph, u, v, true);
              tm_pair_t e = tm_pair(u, v, tk_graph_weight(graph, d, true, -1, -1));
              int kha; khint_t kp = kh_put(pairs, graph->pairs, e, &kha);
              if (kha) {
                kh_value(graph->pairs, kp) = true;
                tk_graph_add_adj(graph, u, v, true);
                tk_dsu_union(&graph->dsu, u, v);
                graph->n_pos++;
              }
            }
          }
        }
      }
      tk_iumap_destroy(curm);
      tk_iumap_destroy(off);
      tk_iumap_destroy(cnt);
      free(buf);
    }

    /* ---------- REBUILD reps after intra-label unions ---------- */
    tk_pumap_t *reps2 = tk_pumap_create();
    for (int64_t idx = 0; idx < (int64_t)graph->uids->n; idx++) {
      int64_t u = graph->uids->a[idx];
      int64_t comp = tk_dsu_find(&graph->dsu, u);
      int64_t deg = tk_iuset_size(graph->adj_pos->a[idx]) + tk_iuset_size(graph->adj_neg->a[idx]);
      int is_new; khint_t kc = tk_pumap_put(reps2, comp, &is_new);
      if (is_new || deg < tk_pumap_value(reps2, kc).p)
        tk_pumap_value(reps2, kc) = tk_pair(idx, deg);
    }
    tk_pvec_t *centers2 = tk_pumap_values(L, reps2);
    tk_pvec_asc(centers2, 0, centers2->n);
    tk_pumap_destroy(reps2);
    tk_pvec_shuffle(centers2);

    /* ---------- ring + chords over WHATEVER'S LEFT (label-agnostic, NEGATIVE) ---------- */
    if (centers2->n >= 2) {
      for (uint64_t i = 0; i < centers2->n; i++) {
        uint64_t j = (i + 1) % centers2->n;
        int64_t iu_uids = centers2->a[i].i;
        int64_t iv_uids = centers2->a[j].i;
        int64_t u = graph->uids->a[iu_uids];
        int64_t v = graph->uids->a[iv_uids];
        if (tk_dsu_find(&graph->dsu, u) != tk_dsu_find(&graph->dsu, v)) {
          double d = tk_graph_distance(graph, u, v, true);
          tm_pair_t e = tm_pair(u, v, tk_graph_weight(graph, d, false, -1, -1));
          int kha; khint_t kp = kh_put(pairs, graph->pairs, e, &kha);
          if (kha) {
            kh_value(graph->pairs, kp) = false;
            tk_graph_add_adj(graph, u, v, false);
            tk_dsu_union(&graph->dsu, u, v);
            graph->n_neg++;
          }
        }
      }
    }
    if (centers2->n >= 3) {
      uint64_t n = centers2->n;
      uint64_t C = (uint64_t) llround(graph->bridge_density * (double)n);
      if (C > 0) {
        uint64_t step = (n / C) > 0 ? (n / C) : 1;
        uint64_t jump = (uint64_t) floor(sqrt((double)n));
        if (jump < 2) jump = 2;
        for (uint64_t t = 0; t < C; t++) {
          uint64_t i = (t * step) % n;
          uint64_t j = (i + jump) % n;
          int64_t iu_uids = centers2->a[i].i;
          int64_t iv_uids = centers2->a[j].i;
          int64_t u = graph->uids->a[iu_uids];
          int64_t v = graph->uids->a[iv_uids];
          double d = tk_graph_distance(graph, u, v, true);
          tm_pair_t e = tm_pair(u, v, tk_graph_weight(graph, d, false, -1, -1));
          int kha; khint_t kp = kh_put(pairs, graph->pairs, e, &kha);
          if (kha) {
            kh_value(graph->pairs, kp) = false;
            tk_graph_add_adj(graph, u, v, false);
            tk_dsu_union(&graph->dsu, u, v);
            graph->n_neg++;
          }
        }
      }
    }
    lua_pop(L, 1); /* centers2 */
    lua_pop(L, 1); /* centers  */
  }
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
      if (tk_iumap_get(graph->uids_idx, u) == tk_iumap_end(graph->uids_idx))
        continue;
      if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
        continue;
      double d = tk_graph_distance(graph, u, v, false);
      int64_t iu = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, u));
      int64_t iv = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, v));
      double w = tk_graph_weight(graph, d, true, iu, iv);
      khi = kh_put(pairs, graph->pairs, tm_pair(u, v, w), &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = true;
      // tk_graph_add_type_mass(graph, iu, iv, d, graph->pos_default);
      tk_dsu_union(&graph->dsu, u, v);
      tk_graph_add_adj(graph, u, v, true);
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
      if (tk_iumap_get(graph->uids_idx, u) == tk_iumap_end(graph->uids_idx))
        continue;
      if (tk_iumap_get(graph->uids_idx, v) == tk_iumap_end(graph->uids_idx))
        continue;
      double d = tk_graph_distance(graph, u, v, false);
      int64_t iu = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, u));
      int64_t iv = tk_iumap_value(graph->uids_idx, tk_iumap_get(graph->uids_idx, v));
      double w = tk_graph_weight(graph, d, false, iu, iv);
      // tk_graph_add_type_mass(graph, iu, iv, d, graph->neg_default);
      khi = kh_put(pairs, graph->pairs, tm_pair(u, v, w), &kha);
      if (!kha)
        continue;
      kh_value(graph->pairs, khi) = false;
      tk_dsu_union(&graph->dsu, u, v);
      tk_graph_add_adj(graph, u, v, false);
      n_neg_new ++;
    }
    graph->n_neg = n_neg_new;
  }
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
  tk_ivec_shrink(out);
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
  if (graph->uids_idx)
    tk_iumap_destroy(graph->uids_idx);
  if (graph->uids_hoods_idx)
    tk_iumap_destroy(graph->uids_hoods_idx);
}

static inline int tm_graph_gc (lua_State *L)
{
  tk_graph_t *graph = tk_graph_peek(L, 1);
  tm_graph_destroy(graph);
  return 0;
}

static inline void tm_setup_hoods (lua_State *L, int Gi, tk_graph_t *graph)
{
  bool find_sigma = graph->sigma_k > 0;
  // NOTE: Get neighborhoods. If we need to find local sigmas, omit mutual
  // filter when getting knns, then post-filter.
  if (!graph->knn_cache)
    return;
  else if (graph->inv != NULL)
    tk_inv_neighborhoods(L, graph->inv, graph->knn_cache, graph->knn_eps, TK_INV_JACCARD, !find_sigma, &graph->inv_hoods, &graph->uids_hoods);
  else if (graph->ann != NULL)
    tk_ann_neighborhoods(L, graph->ann, graph->knn_cache, graph->ann->features * graph->knn_eps, !find_sigma, &graph->ann_hoods, &graph->uids_hoods);
  else if (graph->hbi != NULL)
    tk_hbi_neighborhoods(L, graph->hbi, graph->knn_cache, graph->hbi->features * graph->knn_eps, !find_sigma, &graph->hbi_hoods, &graph->uids_hoods);
  else {
    return;
  }
  if (find_sigma) {
    graph->sigmas = tk_dvec_create(L, graph->uids_hoods->n, 0, 0);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
    for (unsigned int i = 0; i < graph->pool->n_threads; i ++) {
      tk_graph_thread_t *data = graph->threads + i;
      tk_thread_range(i, graph->pool->n_threads, graph->uids_hoods->n, &data->ifirst, &data->ilast);
    }
    tk_threads_signal(graph->pool, TK_GRAPH_SIGMA, 0);
    if (graph->inv != NULL)
      tk_inv_mutualize(L, graph->inv, graph->inv_hoods, graph->uids_hoods);
    else if (graph->ann != NULL)
      tk_ann_mutualize(L, graph->ann, graph->ann_hoods, graph->uids_hoods);
    else if (graph->hbi != NULL)
      tk_hbi_mutualize(L, graph->hbi, graph->hbi_hoods, graph->uids_hoods);
  }
  if (graph->uids == NULL) {
    graph->uids = graph->uids_hoods;
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -2); // uids
  }
  tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1); // hoods
  lua_pop(L, 2);
}

static inline int tm_create (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "ids");
  int i_ids = tk_lua_absindex(L, -1);
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);

  lua_getfield(L, 1, "pos");
  tk_pvec_t *pos = tk_pvec_peekopt(L, -1);

  lua_getfield(L, 1, "neg");
  tk_pvec_t *neg = tk_pvec_peekopt(L, -1);

  lua_getfield(L, 1, "labels");
  int i_labels = tk_lua_absindex(L, -1);
  tk_ivec_t *labels = tk_ivec_peekopt(L, -1);

  // lua_getfield(L, 1, "types");
  // int i_types = tk_lua_absindex(L, -1);
  // tk_ivec_t *types = tk_ivec_peekopt(L, -1);

  // lua_getfield(L, 1, "type_ranks");
  // int i_type_ranks = tk_lua_absindex(L, -1);
  // tk_ivec_t *type_ranks = tk_ivec_peekopt(L, -1);

  // int64_t type_sigma_k = tk_lua_foptinteger(L, 1, "graph", "type_sigma_k", -1);

  if (labels && !ids)
    tk_lua_verror(L, 1, "labels require ids (labels[i] corresponds to ids[i])");

  if (labels && ids && labels->n != ids->n)
    tk_lua_verror(L, 1, "labels length must equal ids length");

  // if (types && (!type_ranks || type_sigma_k < 1))
  //   tk_lua_verror(L, 1, "type ranks and type_sigma_k must be provided with types");

  // if (types && ids && types->n != ids->n)
  //   tk_lua_verror(L, 1, "types length must equal ids length");

  lua_getfield(L, 1, "index");
  tk_inv_t *inv = tk_inv_peekopt(L, -1);
  tk_ann_t *ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *hbi = tk_hbi_peekopt(L, -1);

  uint64_t knn = tk_lua_foptunsigned(L, 1, "graph", "knn", 0);
  uint64_t knn_pos = tk_lua_foptunsigned(L, 1, "graph", "knn_pos", knn);
  uint64_t knn_neg = tk_lua_foptunsigned(L, 1, "graph", "knn_neg", knn);
  uint64_t knn_cache = tk_lua_foptunsigned(L, 1, "graph", "knn_cache", 0);
  double knn_eps = tk_lua_foptposdouble(L, 1, "graph", "knn_eps", 1.0);
  if ((knn_pos + knn_neg) > knn_cache)
    knn_cache = knn_pos + knn_neg;
  double pos_default = tk_lua_foptnumber(L, 1, "graph", "pos_default", 1.0);
  double neg_default = tk_lua_foptnumber(L, 1, "graph", "neg_default", 1.0);
  double pos_scale = tk_lua_foptnumber(L, 1, "graph", "pos_scale", 1.0);
  double neg_scale = tk_lua_foptnumber(L, 1, "graph", "neg_scale", -1.0);
  double pos_sigma_scale = tk_lua_foptnumber(L, 1, "graph", "pos_sigma_scale", 1.0);
  double neg_sigma_scale = tk_lua_foptnumber(L, 1, "graph", "neg_sigma_scale", 1.0);
  int64_t sigma_k = tk_lua_foptinteger(L, 1, "graph", "sigma_k", -1);
  bool no_label_is_match = tk_lua_foptboolean(L, 1, "graph", "no_label_is_match", true);
  double bridge_density = tk_lua_foptnumber(L, 1, "graph", "bridge_density", 0.02);
  double weight_eps = tk_lua_foptnumber(L, 1, "graph", "weight_eps", 1e-6);
  bool do_mst = tk_lua_foptboolean(L, 1, "graph", "mst", true);
  bool do_bridge = tk_lua_foptboolean(L, 1, "graph", "bridge", true);

  unsigned int n_threads = tk_threads_getn(L, 1, "graph", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_graph_t *graph = tm_graph_create(
    L, i_ids, ids, pos, neg, labels, inv, ann, hbi, knn_cache, knn_eps,
    pos_scale, neg_scale, pos_default, neg_default, pos_sigma_scale,
    neg_sigma_scale, sigma_k,
    /*i_types, types, i_type_ranks, type_ranks, type_sigma_k,*/
    bridge_density, weight_eps, no_label_is_match, n_threads);
  int Gi = tk_lua_absindex(L, -1);

  tm_setup_hoods(L, Gi, graph);
  if (graph->uids_hoods_idx == NULL)
    graph->uids_hoods_idx = graph->uids_hoods == NULL ? tk_iumap_create() : tk_iumap_from_ivec(graph->uids_hoods);

  if (graph->uids == NULL && (pos != NULL || neg != NULL)) {
    int kha;
    tk_iuset_t *uids = tk_iuset_create();
    if (pos) {
      for (uint64_t i = 0; i < pos->n; i ++) {
        tk_pair_t p = pos->a[i];
        tk_iuset_put(uids, p.i, &kha);
        tk_iuset_put(uids, p.p, &kha);
      }
    }
    if (neg) {
      for (uint64_t i = 0; i < neg->n; i ++) {
        tk_pair_t p = neg->a[i];
        tk_iuset_put(uids, p.i, &kha);
        tk_iuset_put(uids, p.p, &kha);
      }
    }
    graph->uids = tk_iuset_keys(L, uids);
    tk_iuset_destroy(uids);
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, -1);
    lua_pop(L, 1);
  }
  if (graph->uids == NULL)
    tk_lua_verror(L, 2, "graph", "either ids, index, pos, and or neg");
  if (graph->uids_idx == NULL)
    graph->uids_idx = graph->uids == NULL ? tk_iumap_create() : tk_iumap_from_ivec(graph->uids);
  if (labels)
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, i_labels);

  // Setup DSU based on uids
  tk_dsu_init(&graph->dsu, graph->uids);

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, 0);
    lua_pushinteger(L, (int64_t) graph->n_pos);
    lua_pushinteger(L, (int64_t) graph->n_neg);
    lua_pushstring(L, "init");
    lua_call(L, 4, 0);
  }

  // Setup adjacency lists & add seed pairs
  tm_adj_init(L, Gi, graph);
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
  if (knn_pos > 0 || knn_neg > 0)
    tm_add_knn(L, graph, knn_pos, knn_neg);

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
  if (do_mst && graph->knn_cache && graph->dsu.components > 1) {
    tm_candidates_t cs = tm_mst_knn_candidates(L, graph);
    tm_add_mst(L, graph, &cs);
    kv_destroy(cs);
  }

  // Log density
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, tk_dsu_components(&graph->dsu));
    lua_pushinteger(L, (int64_t) graph->n_pos);
    lua_pushinteger(L, (int64_t) graph->n_neg);
    lua_pushstring(L, "kruskal");
    lua_call(L, 4, 0);
  }

  // if (graph->types)
  //   tk_graph_reweight_types(graph);

  if (do_bridge && graph->dsu.components > 1)
    tm_add_mst(L, graph, NULL);

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

  // Render pairs
  tk_pvec_t *pos = tk_pvec_create(L, 0, 0, 0);
  tk_pvec_t *neg = tk_pvec_create(L, 0, 0, 0);
  tk_dvec_t *wpos = tk_dvec_create(L, 0, 0, 0);
  tk_dvec_t *wneg = tk_dvec_create(L, 0, 0, 0);
  tm_render_pairs(L, graph, pos, neg, wpos, wneg);
  return 4;
}

static luaL_Reg tm_graph_mt_fns[] =
{
  { "pairs", tm_graph_pairs },
  { "adjacency", tm_graph_adjacency },
  { NULL, NULL }
};

static inline tk_graph_t *tm_graph_create (
  lua_State *L,
  int i_ids, tk_ivec_t *ids,
  tk_pvec_t *pos,
  tk_pvec_t *neg,
  tk_ivec_t *labels,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  uint64_t knn_cache,
  double knn_eps,
  double pos_scale,
  double neg_scale,
  double pos_default,
  double neg_default,
  double pos_sigma_scale,
  double neg_sigma_scale,
  int64_t sigma_k,
  // int i_types, tk_ivec_t *types,
  // int i_type_ranks, tk_ivec_t *type_ranks,
  // int64_t type_sigma_k,
  double bridge_density,
  double weight_eps,
  bool no_label_is_match,
  unsigned int n_threads
) {
  tk_graph_t *graph = tk_lua_newuserdata(L, tk_graph_t, TK_GRAPH_MT, tm_graph_mt_fns, tm_graph_gc); // ud
  int Gi = tk_lua_absindex(L, -1);
  graph->threads = tk_malloc(L, n_threads * sizeof(tk_graph_thread_t));
  memset(graph->threads, 0, n_threads * sizeof(tk_graph_thread_t));
  graph->pool = tk_threads_create(L, n_threads, tk_graph_worker);
  graph->knn_cache = knn_cache;
  graph->knn_eps = knn_eps;
  graph->pos_scale = pos_scale;
  graph->neg_scale = neg_scale;
  graph->pos_default = pos_default;
  graph->neg_default = neg_default;
  graph->pos_sigma_scale = pos_sigma_scale;
  graph->neg_sigma_scale = neg_sigma_scale;
  graph->sigma_k = sigma_k;
  graph->no_label_is_match = no_label_is_match;
  graph->bridge_density = bridge_density;
  graph->weight_eps = weight_eps;
  graph->pairs = kh_init(pairs);
  graph->inv = inv;
  graph->ann = ann;
  graph->hbi = hbi;
  // graph->types = types;
  // graph->type_ranks = type_ranks;
  // graph->type_sigma_k = type_sigma_k;
  // if (graph->types) {
  //   tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, i_types);
  //   tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, i_type_ranks);
  //   graph->type_mass = kh_init(pairs);
  // }
  if (ids != NULL) {
    graph->uids = ids;
    tk_lua_add_ephemeron(L, TK_GRAPH_EPH, Gi, i_ids);
    if (graph->uids_idx == NULL)
      graph->uids_idx = tk_iumap_from_ivec(graph->uids);
  }
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
