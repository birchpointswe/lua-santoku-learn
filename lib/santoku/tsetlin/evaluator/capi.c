#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/tsetlin/cluster.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/rvec.h>
#include <santoku/pvec.h>
#include <math.h>

#define TK_EVAL_EPH "tk_eval_eph"

typedef enum {
  TK_EVAL_CLASS_ACCURACY,
  TK_EVAL_CLUSTERING_ACCURACY,
  TK_EVAL_OPTIMIZE_CLUSTERING,
  TK_EVAL_ENCODING_ACCURACY,
  TK_EVAL_ENCODING_SIMILARITY,
  TK_EVAL_ENCODING_AUC,
  TK_EVAL_GRAPH_RECONSTRUCTION_INIT,
  TK_EVAL_GRAPH_RECONSTRUCTION,
  TK_EVAL_GRAPH_RECONSTRUCTION_DESTROY,
} tk_eval_stage_t;

typedef enum {
  TK_CLUSTER_METHOD_GRAPH,
  TK_CLUSTER_METHOD_PREFIX,
} tk_cluster_method_t;

typedef struct {
  tm_dl_t *pl;
  atomic_ulong *ERR, *TP, *FP, *FN, *hist_pos, *hist_neg;
  atomic_ulong *valid_pos, *valid_neg;
  tk_ivec_t *ids;
  tk_iumap_t *ididx;
  tk_inv_t *inv;
  tk_ann_t *ann;
  tk_hbi_t *hbi;
  uint64_t min_pts;
  bool assign_noise;
  tk_cluster_method_t cluster_method;
  uint64_t min_margin, max_margin;
  tk_dvec_t *inv_thresholds;  // Pre-computed distance thresholds for INV
  uint64_t probe_radius;
  tk_ivec_sim_type_t cmp;
  double cmp_alpha, cmp_beta;
  int64_t rank_filter;
  tk_pvec_t *pos, *neg;
  uint64_t n_pos, n_neg;
  double *f1, *precision, *recall;
  int64_t *predicted, *expected;
  tk_bits_t *codes, *codes_predicted, *codes_expected, *mask;
  uint64_t mask_popcount;
  double *dcodes;
  tk_iumap_t *id_code, *id_assignment;
  unsigned int n_visible, n_dims, chunks;
  tk_ivec_t *assignments;
  tk_graph_t *graph;
  tk_ivec_t *offsets;
  tk_ivec_t *neighbors;
  tk_dvec_t *weights;
  uint64_t optimal_k;
  int64_t start_prefix;
  double tolerance;
  lua_State *L;
  tk_inv_hoods_t *inv_hoods;
  tk_ann_hoods_t *ann_hoods;
  tk_hbi_hoods_t *hbi_hoods;
  tk_ivec_t *uids_hoods;
  tk_iumap_t *uids_idx_hoods;
  int correlation_metric;
} tk_eval_t;

typedef struct {
  double tpr;
  double tnr;
  double bacc;
} tk_accuracy_t;

typedef struct {
  tk_thread_t *self;
  tk_eval_t *state;
  atomic_bool has_error;
  uint64_t diff;
  uint64_t *bdiff;
  uint64_t sfirst, slast;
  uint64_t pfirst, plast;
  uint64_t dfirst, dlast;
  uint64_t mfirst, mlast;
  tk_accuracy_t best, next;
  uint64_t next_n;
  uint64_t best_n;
  uint64_t next_m;
  uint64_t best_m;
  tk_ivec_t *next_assignments;
  tk_ivec_t *best_assignments;
  tk_rvec_t *rtmp;
  uint64_t wfirst, wlast;
  tk_pvec_t *ptmp;
  tk_rvec_t **adj_ranks;
  tk_dumap_t **adj_ranks_idx;
  tk_pvec_t *bin_ranks;
  tk_dumap_t *bin_ranks_idx;
  double recon_score;
  uint64_t nodes_processed;
  tk_dsu_t *dsu;
  tk_cvec_t *is_core;
  uint64_t *prefix_cache;
  tk_bits_t *prefix_bytes;
  uint64_t prefix_cache_depth;
} tk_eval_thread_t;

static inline tk_accuracy_t _tm_clustering_accuracy (
  tk_ivec_t *ids,
  tk_ivec_t *assignments,
  tk_pvec_t *pos,
  tk_pvec_t *neg
);

static void tk_eval_worker (void *dp, int sig)
{
  tk_eval_thread_t *data = (tk_eval_thread_t *) dp;
  tk_eval_t *state = data->state;
  khint_t khi;
  switch ((tk_eval_stage_t) sig) {

    case TK_EVAL_CLASS_ACCURACY:
      for (unsigned int i = data->sfirst; i <= data->slast; i ++) {
        unsigned int y_pred = state->predicted[i];
        unsigned int y_true = state->expected[i];
        if (y_pred >= state->n_dims || y_true >= state->n_dims)
          continue;
        if (y_pred == y_true)
          atomic_fetch_add(state->TP + y_true, 1);
        else {
          atomic_fetch_add(state->FP + y_pred, 1);
          atomic_fetch_add(state->FN + y_true, 1);
        }
      }
      break;

    case TK_EVAL_OPTIMIZE_CLUSTERING:
      if (state->cluster_method == TK_CLUSTER_METHOD_GRAPH) {
        if (data->dsu == NULL) {
          data->dsu = tk_dsu_create(NULL, state->ids);
          if (data->dsu == NULL) {
            atomic_store(&data->has_error, true);
            return;
          }
        }
        if (data->is_core == NULL) {
          data->is_core = tk_cvec_create(0, TK_CVEC_BITS_BYTES(state->ids->n), 0, 0);
          tk_cvec_zero(data->is_core);
          if (data->is_core == NULL) {
            atomic_store(&data->has_error, true);
            return;
          }
        }
      } else if (state->cluster_method == TK_CLUSTER_METHOD_PREFIX) {
        uint64_t max_features = state->hbi ? state->hbi->features :
                                state->ann ? state->ann->features : 0;
        if (max_features <= 64) {
          if (data->prefix_cache == NULL) {
            data->prefix_cache = (uint64_t *)malloc(state->ids->n * sizeof(uint64_t));
            if (data->prefix_cache == NULL) {
              atomic_store(&data->has_error, true);
              return;
            }
            memset(data->prefix_cache, 0, state->ids->n * sizeof(uint64_t));
            data->prefix_cache_depth = 0;
          }
        } else {
          if (data->prefix_bytes == NULL) {
            uint64_t chunks = TK_CVEC_BITS_BYTES(max_features);
            data->prefix_bytes = (tk_bits_t *)malloc(state->ids->n * chunks);
            if (data->prefix_bytes == NULL) {
              atomic_store(&data->has_error, true);
              return;
            }
            memset(data->prefix_bytes, 0, state->ids->n * chunks);
            data->prefix_cache_depth = 0;
          }
        }
      }

      uint64_t m_start = state->cluster_method == TK_CLUSTER_METHOD_PREFIX ? data->mlast : data->mfirst;
      uint64_t m_end = state->cluster_method == TK_CLUSTER_METHOD_PREFIX ? data->mfirst : data->mlast;
      int64_t m_step = state->cluster_method == TK_CLUSTER_METHOD_PREFIX ? -1 : 1;

      for (int64_t m0_signed = (int64_t)m_start;
           m_step > 0 ? (m0_signed <= (int64_t)m_end) : (m0_signed >= (int64_t)m_end);
           m0_signed += m_step) {
        uint64_t m0 = (uint64_t)m0_signed;
        tk_ivec_clear(data->next_assignments);

        tk_cluster_opts_t cluster_opts = {
          .inv = state->inv,
          .hbi = state->hbi,
          .ann = state->ann,
          .ids = state->ids,
          .ididx = state->ididx,
          .min_pts = state->min_pts,
          .assign_noise = state->assign_noise,
          .probe_radius = state->probe_radius,
          .cmp = state->cmp,
          .cmp_alpha = state->cmp_alpha,
          .cmp_beta = state->cmp_beta,
          .rank_filter = state->rank_filter,
          .inv_hoods = state->inv_hoods,
          .ann_hoods = state->ann_hoods,
          .hbi_hoods = state->hbi_hoods,
          .uids_hoods = state->uids_hoods,
          .uids_idx_hoods = state->uids_idx_hoods,
          .assignments = data->next_assignments,
          .n_clustersp = &data->next_n,
          .dsu_reuse = data->dsu,
          .is_core_reuse = data->is_core,
          .rtmp = data->rtmp,
          .ptmp = data->ptmp,
          .prefix_cache_reuse = data->prefix_cache,
          .prefix_bytes_reuse = data->prefix_bytes,
          .prefix_cache_depth = data->prefix_cache_depth
        };

        if (state->cluster_method == TK_CLUSTER_METHOD_PREFIX) {
          uint64_t depth = m0 + state->min_margin;
          cluster_opts.depth = depth;
          data->next_m = depth;
          tk_cluster_prefix(&cluster_opts);
          data->prefix_cache_depth = cluster_opts.prefix_cache_depth;
        } else {
          if (state->inv != NULL) {
            uint64_t bin_idx = m0 + state->min_margin;
            double eps = state->inv_thresholds->a[bin_idx];
            cluster_opts.eps = eps;
            data->next_m = (uint64_t)(eps * 1000);
          } else {
            uint64_t m = m0 + state->min_margin;
            cluster_opts.margin = m;
            data->next_m = m;
          }
          tk_cluster_dsu(&cluster_opts);
        }

        tk_accuracy_t result = _tm_clustering_accuracy(state->ids, data->next_assignments, state->pos, state->neg);
        data->next = result;
        tk_threads_notify_parent(data->self);

        if (result.bacc > data->best.bacc) {
          data->best = result;
          data->best_n = data->next_n;
          data->best_m = data->next_m;
          tk_ivec_t *tmp = data->best_assignments;
          data->best_assignments = data->next_assignments;
          data->next_assignments = tmp;
        }

        if (data->next_n <= 1)
          break;
      }

      if (data->dsu != NULL) {
        tk_dsu_destroy(data->dsu);
        data->dsu = NULL;
      }
      if (data->is_core != NULL) {
        tk_cvec_destroy(data->is_core);
        data->is_core = NULL;
      }
      if (data->prefix_cache != NULL) {
        free(data->prefix_cache);
        data->prefix_cache = NULL;
      }
      if (data->prefix_bytes != NULL) {
        free(data->prefix_bytes);
        data->prefix_bytes = NULL;
      }
      break;

    case TK_EVAL_CLUSTERING_ACCURACY:
      for (unsigned int i = data->pfirst; i <= data->plast; i ++) {
        tk_pvec_t *pairs = i < state->n_pos ? state->pos : state->neg;
        uint64_t offset = i < state->n_pos ? 0 : state->n_pos;
        int64_t u = pairs->a[(i - offset)].i;
        int64_t v = pairs->a[(i - offset)].p;
        khi = tk_iumap_get(state->id_assignment, u);
        if (khi == tk_iumap_end(state->id_assignment))
          continue;
        int64_t iu = tk_iumap_val(state->id_assignment, khi);
        khi = tk_iumap_get(state->id_assignment, v);
        if (khi == tk_iumap_end(state->id_assignment))
          continue;
        int64_t iv = tk_iumap_val(state->id_assignment, khi);
        int64_t cu = state->assignments->a[iu];
        int64_t cv = state->assignments->a[iv];
        if (i < state->n_pos) {
          atomic_fetch_add(state->valid_pos, 1);
          if (cu == cv)
            atomic_fetch_add(state->TP, 1);
          else
            atomic_fetch_add(state->FN, 1);
        } else {
          atomic_fetch_add(state->valid_neg, 1);
          if (cu == cv)
            atomic_fetch_add(state->FP, 1);
        }
      }
      break;

    case TK_EVAL_ENCODING_ACCURACY:
      for (uint64_t i = data->sfirst; i <= data->slast; i ++) {
        for (uint64_t j = 0; j < state->n_dims; j ++) {
          uint64_t word = TK_CVEC_BITS_BYTE(j);
          uint64_t bit = TK_CVEC_BITS_BIT(j);
          bool y =
            (state->codes_expected[i * state->chunks + word] & ((tk_bits_t)1 << bit)) ==
            (state->codes_predicted[i * state->chunks + word] & ((tk_bits_t)1 << bit));
          if (y)
            continue;
          data->diff ++;
          data->bdiff[j] ++;
        }
      }
      break;

    case TK_EVAL_ENCODING_SIMILARITY:
      for (uint64_t k = data->pfirst; k <= data->plast; k ++) {
        if (state->pl[k].sim == -1)
          continue;
        int64_t s = state->n_dims - state->pl[k].sim;
        if (s < 0) s = 0;
        if (s > state->n_dims) s = state->n_dims;
        if (state->pl[k].label)
          atomic_fetch_add(state->hist_pos + s, 1);
        else
          atomic_fetch_add(state->hist_neg + s, 1);
      }
      break;

    case TK_EVAL_ENCODING_AUC:
      for (uint64_t k = data->pfirst; k <= data->plast; k ++) {
        tk_pvec_t *pairs = k < state->n_pos ? state->pos : state->neg;
        uint64_t offset = k < state->n_pos ? 0 : state->n_pos;
        int64_t u = pairs->a[(k - offset)].i;
        int64_t v = pairs->a[(k - offset)].p;
        int64_t iu, iv;
        if (state->id_code == NULL) {
          iu = u;
          iv = v;
        } else {
          khi = tk_iumap_get(state->id_code, u);
          if (khi == tk_iumap_end(state->id_code)) {
            state->pl[k].sim = -1;
            continue;
          }
          iu = tk_iumap_val(state->id_code, khi);
          if (iu == -1)
            continue;
          khi = tk_iumap_get(state->id_code, v);
          if (khi == tk_iumap_end(state->id_code)) {
            state->pl[k].sim = -1;
            continue;
          }
          iv = tk_iumap_val(state->id_code, khi);
          if (iv == -1)
            continue;
        }
        if (state->dcodes != NULL) {
          double *X = state->dcodes;
          uint64_t K = state->n_dims;
          double dot = 0.0;
          double *xi = X + (uint64_t)iu * K;
          double *xj = X + (uint64_t)iv * K;
          for (uint64_t d = 0; d < K; d ++)
            dot += xi[d] * xj[d];
          state->pl[k].sim = (int64_t)(dot * 1e3);
        } else if (state->codes != NULL) {
          if (state->mask != NULL) {
            state->pl[k].sim = (int64_t) (state->mask_popcount - tk_cvec_bits_hamming_mask(
              (const unsigned char *) state->codes + (uint64_t) iu * state->chunks,
              (const unsigned char *) state->codes + (uint64_t) iv * state->chunks, (const unsigned char *) state->mask, state->n_dims));
          } else {
            state->pl[k].sim = (int64_t) (state->n_dims - tk_cvec_bits_hamming(
              (const unsigned char *) state->codes + (uint64_t) iu * state->chunks,
              (const unsigned char *) state->codes + (uint64_t) iv * state->chunks, state->n_dims));
          }
        } else {
          state->pl[k].sim = -1;
        }
        state->pl[k].label = k < state->n_pos ? 1 : 0;
      }
      break;

    case TK_EVAL_GRAPH_RECONSTRUCTION_INIT: {
      uint64_t n_alloc = data->wlast - data->wfirst + 1;
      data->adj_ranks = malloc(n_alloc * sizeof(tk_rvec_t *));
      data->adj_ranks_idx = malloc(n_alloc * sizeof(tk_dumap_t *));
      if (data->adj_ranks == NULL || data->adj_ranks_idx == NULL) {
        atomic_store(&data->has_error, true);
        if (data->adj_ranks) {
          free(data->adj_ranks);
          data->adj_ranks = NULL;
        }
        if (data->adj_ranks_idx) {
          free(data->adj_ranks_idx);
          data->adj_ranks_idx = NULL;
        }
        return;
      }
      for (uint64_t i = 0; i < n_alloc; i ++) {
        data->adj_ranks[i] = NULL;
        data->adj_ranks_idx[i] = NULL;
      }

      data->bin_ranks = tk_pvec_create(0, 0, 0, 0);
      data->bin_ranks_idx = tk_dumap_create(0, 0);
      for (uint64_t i = 0; i < n_alloc; i ++) {
        uint64_t node_idx = data->wfirst + i;
        assert(node_idx + 1 < state->offsets->n);
        data->adj_ranks[i] = tk_rvec_create(0, 0, 0, 0);
        data->adj_ranks_idx[i] = tk_dumap_create(0, 0);
        int64_t start = state->offsets->a[node_idx];
        int64_t end = state->offsets->a[node_idx + 1];
        for (int64_t j = start; j < end; j ++) {
          int64_t neighbor = state->neighbors->a[j];
          double weight = state->weights->a[j];
          if (tk_rvec_push(data->adj_ranks[i], tk_rank(neighbor, weight)) != 0) {
            atomic_store(&data->has_error, true);
            return;
          }
        }
        tk_rvec_desc(data->adj_ranks[i], 0, data->adj_ranks[i]->n);
        if (state->correlation_metric == 0) {
          tk_rvec_ranks(data->adj_ranks[i], data->adj_ranks_idx[i]);
        }
      }
      break;
    }

    case TK_EVAL_GRAPH_RECONSTRUCTION: {
      data->recon_score = 0.0;
      data->nodes_processed = 0;
      for (uint64_t i = 0; i < data->wlast - data->wfirst + 1; i ++) {
        tk_pvec_clear(data->bin_ranks);
        uint64_t node_idx = data->wfirst + i;
        assert(node_idx + 1 < state->offsets->n);
        int64_t start = state->offsets->a[node_idx];
        int64_t end = state->offsets->a[node_idx + 1];
        for (int64_t j = start; j < end; j ++) {
          int64_t neighbor = state->neighbors->a[j];
          uint64_t hamming_dist = tk_cvec_bits_hamming_mask(
            (const unsigned char *) state->codes + node_idx * state->chunks,
            (const unsigned char *) state->codes + neighbor * state->chunks,
            (const unsigned char *) state->mask,
            state->n_dims);
          if (tk_pvec_push(data->bin_ranks, tk_pair(neighbor, (int64_t) hamming_dist)) != 0) {
            atomic_store(&data->has_error, true);
            return;
          }
        }
        tk_pvec_asc(data->bin_ranks, 0, data->bin_ranks->n);

        double corr;
        switch (state->correlation_metric) {
          case 0:
            tk_pvec_ranks(data->bin_ranks, data->bin_ranks_idx);
            corr = tk_dumap_correlation(data->adj_ranks_idx[i], data->bin_ranks_idx);
            break;
          case 1:
            corr = tk_rvec_kendall_pvec(data->adj_ranks[i], data->bin_ranks);
            break;
          case 2:
            corr = tk_rvec_spearman_weighted_pvec(data->adj_ranks[i], data->bin_ranks);
            break;
          case 3:
            corr = tk_rvec_kendall_weighted_pvec(data->adj_ranks[i], data->bin_ranks);
            break;
          default:
            corr = 0.0;
            break;
        }

        data->recon_score += corr;
        data->nodes_processed++;
      }
      break;
    }

    case TK_EVAL_GRAPH_RECONSTRUCTION_DESTROY: {
      uint64_t n_alloc = data->wlast - data->wfirst + 1;
      for (uint64_t i = 0; i < n_alloc; i ++) {
        if (data->adj_ranks[i] != NULL)
          tk_rvec_destroy(data->adj_ranks[i]);
        if (data->adj_ranks_idx[i] != NULL)
          tk_dumap_destroy(data->adj_ranks_idx[i]);
      }
      free(data->adj_ranks);
      free(data->adj_ranks_idx);
      tk_pvec_destroy(data->bin_ranks);
      tk_dumap_destroy(data->bin_ranks_idx);
      break;
    }

    default:
      assert(false);

  }
}

static inline int tm_class_accuracy (lua_State *L)
{
  lua_settop(L, 5);
  tk_ivec_t *predicted = tk_ivec_peek(L, 1, "predicted");
  tk_ivec_t *expected = tk_ivec_peek(L, 2, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int n_dims = tk_lua_checkunsigned(L, 4, "n_classes");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);

  if (n_dims == 0)
    tk_lua_verror(L, 3, "class_accuracy", "n_classes", "must be > 0");

  tk_eval_t state;
  state.n_dims = n_dims;
  state.expected = expected->a;
  state.predicted = predicted->a;
  state.TP = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  state.FP = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  state.FN = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  state.precision = tk_malloc(L, n_dims * sizeof(double));
  state.recall = tk_malloc(L, n_dims * sizeof(double));
  state.f1 = tk_malloc(L, n_dims * sizeof(double));
  for (uint64_t i = 0; i < n_dims; i ++) {
    atomic_init(state.TP + i, 0);
    atomic_init(state.FP + i, 0);
    atomic_init(state.FN + i, 0);
  }

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    atomic_init(&data[i].has_error, false);
    tk_thread_range(i, n_threads, n_samples, &data[i].sfirst, &data[i].slast);
  }

  tk_threads_signal(pool, TK_EVAL_CLASS_ACCURACY, 0);

  // Reduce
  double precision_avg = 0.0, recall_avg = 0.0, f1_avg = 0.0;
  for (unsigned int c = 0; c < n_dims; c ++) {
    uint64_t tp = state.TP[c], fp = state.FP[c], fn = state.FN[c];
    state.precision[c] = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
    state.recall[c] = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
    state.f1[c] = (state.precision[c] + state.recall[c]) > 0 ?
      2.0 * state.precision[c] * state.recall[c] / (state.precision[c] + state.recall[c]) : 0.0;
    precision_avg += state.precision[c];
    recall_avg += state.recall[c];
    f1_avg += state.f1[c];
  }

  // Cleanup
  free(state.TP);
  free(state.FP);
  free(state.FN);

  // Lua output
  precision_avg /= n_dims;
  recall_avg /= n_dims;
  f1_avg /= n_dims;
  lua_newtable(L);
  lua_newtable(L);
  for (uint64_t c = 0; c < n_dims; c ++) {
    lua_pushinteger(L, (int64_t) c + 1);
    lua_newtable(L);
    lua_pushnumber(L, state.precision[c]);
    lua_setfield(L, -2, "precision");
    lua_pushnumber(L, state.recall[c]);
    lua_setfield(L, -2, "recall");
    lua_pushnumber(L, state.f1[c]);
    lua_setfield(L, -2, "f1");
    lua_settable(L, -3);
  }
  lua_setfield(L, -2, "classes");
  lua_pushnumber(L, precision_avg);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, recall_avg);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, f1_avg);
  lua_setfield(L, -2, "f1");

  // Cleanup
  free(state.precision);
  free(state.recall);
  free(state.f1);
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline tk_accuracy_t _tm_clustering_accuracy (
  tk_ivec_t *ids,
  tk_ivec_t *assignments,
  tk_pvec_t *pos,
  tk_pvec_t *neg
) {
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;
  if (n_pos + n_neg == 0)
    return (tk_accuracy_t) { 0 };

  tk_eval_t state;
  atomic_ulong TP, FP, FN, valid_pos, valid_neg;
  state.assignments = assignments;
  state.id_assignment = tk_iumap_from_ivec(0, ids);
  if (!state.id_assignment)
    return (tk_accuracy_t) { 0 };
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;
  state.TP = &TP;
  state.FP = &FP;
  state.FN = &FN;
  state.valid_pos = &valid_pos;
  state.valid_neg = &valid_neg;
  atomic_init(&TP, 0);
  atomic_init(&FP, 0);
  atomic_init(&FN, 0);
  atomic_init(&valid_pos, 0);
  atomic_init(&valid_neg, 0);

  tk_eval_thread_t data;
  data.state = &state;
  data.pfirst = 0;
  data.plast = n_pos + n_neg - 1;

  tk_eval_worker(&data, TK_EVAL_CLUSTERING_ACCURACY);

  uint64_t vpos = atomic_load(state.valid_pos);
  uint64_t vneg = atomic_load(state.valid_neg);
  uint64_t tp = atomic_load(state.TP);
  uint64_t fp = atomic_load(state.FP);
  uint64_t tn = vneg > fp ? (vneg - fp) : 0;
  double tpr = vpos > 0 ? (double) tp / vpos : 0.0;
  double tnr = vneg > 0 ? (double) tn / vneg : 0.0;
  double bacc = 0.5 * (tpr + tnr); // true balanced accuracy
  tk_accuracy_t acc;
  acc.tpr = tpr;
  acc.tnr = tnr;
  acc.bacc = bacc;

  tk_iumap_destroy(state.id_assignment);
  return acc;
}

static inline tk_accuracy_t tm_clustering_accuracy (
  lua_State *L,
  tk_ivec_t *ids,
  tk_ivec_t *assignments,
  tk_pvec_t *pos,
  tk_pvec_t *neg,
  unsigned int n_threads
) {
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_eval_t state;
  atomic_ulong TP, FP, FN, valid_pos, valid_neg;
  state.assignments = assignments;
  state.id_assignment = tk_iumap_from_ivec(0, ids);
  if (!state.id_assignment)
    tk_error(L, "tm_clustering_accuracy: iumap_from_ivec failed", ENOMEM);
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;
  state.TP = &TP;
  state.FP = &FP;
  state.FN = &FN;
  state.valid_pos = &valid_pos;
  state.valid_neg = &valid_neg;
  atomic_init(&TP, 0);
  atomic_init(&FP, 0);
  atomic_init(&FN, 0);
  atomic_init(&valid_pos, 0);
  atomic_init(&valid_neg, 0);

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    atomic_init(&data[i].has_error, false);
    tk_thread_range(i, n_threads, n_pos + n_neg, &data[i].pfirst, &data[i].plast);
  }

  tk_threads_signal(pool, TK_EVAL_CLUSTERING_ACCURACY, 0);

  uint64_t vpos = atomic_load(state.valid_pos);
  uint64_t vneg = atomic_load(state.valid_neg);
  uint64_t tp = atomic_load(state.TP);
  uint64_t fp = atomic_load(state.FP);
  uint64_t tn = vneg > fp ? (vneg - fp) : 0;
  double tpr = vpos > 0 ? (double) tp / vpos : 0.0;
  double tnr = vneg > 0 ? (double) tn / vneg : 0.0;
  double bacc = 0.5 * (tpr + tnr); // true balanced accuracy
  tk_accuracy_t acc;
  acc.tpr = tpr;
  acc.tnr = tnr;
  acc.bacc = bacc;

  tk_iumap_destroy(state.id_assignment);
  return acc;
}

static inline int tm_clustering_accuracy_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_ivec_t *assignments = tk_ivec_peek(L, 1, "assignments");
  tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
  tk_pvec_t *pos = tk_pvec_peek(L, 3, "pos");
  tk_pvec_t *neg = tk_pvec_peek(L, 4, "neg");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);

  tk_accuracy_t acc = tm_clustering_accuracy(L, ids, assignments, pos, neg, n_threads);

  // Lua output
  lua_newtable(L);
  lua_pushnumber(L, acc.bacc);
  lua_setfield(L, -2, "bacc");
  lua_pushnumber(L, acc.tpr);
  lua_setfield(L, -2, "tpr");
  lua_pushnumber(L, acc.tnr);
  lua_setfield(L, -2, "tnr");
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tm_encoding_accuracy (lua_State *L)
{
  lua_settop(L, 5);
  tk_bits_t *codes_predicted, *codes_expected;
  tk_cvec_t *pvec = tk_cvec_peekopt(L, 1);
  tk_cvec_t *evec = tk_cvec_peekopt(L, 2);
  codes_predicted = pvec != NULL ? (tk_bits_t *) pvec->a : (tk_bits_t *) tk_lua_checkustring(L, 1, "predicted");
  codes_expected = evec != NULL ? (tk_bits_t *) evec->a : (tk_bits_t *) tk_lua_checkustring(L, 2, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int n_dims = tk_lua_checkunsigned(L, 4, "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);
  uint64_t chunks = TK_CVEC_BITS_BYTES(n_dims);

  tk_eval_t state;
  state.n_dims = n_dims;
  state.chunks = chunks;
  state.codes_expected = codes_expected;
  state.codes_predicted = codes_predicted;

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    atomic_init(&data[i].has_error, false);
    data[i].diff = 0;
    data[i].bdiff = tk_malloc(L, n_dims * sizeof(uint64_t));
    memset(data[i].bdiff, 0, sizeof(uint64_t) * n_dims);
    tk_thread_range(i, n_threads, n_samples, &data[i].sfirst, &data[i].slast);
  }

  tk_threads_signal(pool, TK_EVAL_ENCODING_ACCURACY, 0);

  // Reduce
  uint64_t diff_total = 0;
  uint64_t *bdiff_total = tk_malloc(L, n_dims * sizeof(uint64_t));
  memset(bdiff_total, 0, n_dims * sizeof(uint64_t));
  for (unsigned int i = 0; i < n_threads; i ++) {
    diff_total += data[i].diff;
    for (uint64_t j = 0; j < n_dims; j ++)
      bdiff_total[j] += data[i].bdiff[j];
  }

  // Lua output
  lua_newtable(L);
  lua_newtable(L); // dims
  double min_bdiff = 1.0, max_bdiff = 0.0;
  for (uint64_t j = 0; j < n_dims; j ++) {
    double t = (double) bdiff_total[j] / (double) n_samples;
    if (t < min_bdiff) min_bdiff = t;
    if (t > max_bdiff) max_bdiff = t;
    lua_pushinteger(L, (int64_t) j + 1);
    lua_pushnumber(L, t);
    lua_settable(L, -3);
  }

  double mean_bdiff = (double) diff_total / (n_samples * n_dims);
  double var = 0.0;
  for (uint64_t j = 0; j < n_dims; ++j) {
    double ber = (double) bdiff_total[j] / n_samples;
    var += (ber - mean_bdiff) * (ber - mean_bdiff);
  }
  double std_bdiff = n_dims > 1 ? sqrt(var / (n_dims - 1)) : 0.0;

  lua_setfield(L, -2, "bits");
  lua_pushnumber(L, 1.0 - mean_bdiff);
  lua_setfield(L, -2, "mean_hamming");
  lua_pushnumber(L, 1.0 - min_bdiff);
  lua_setfield(L, -2, "ber_min");
  lua_pushnumber(L, 1.0 - max_bdiff);
  lua_setfield(L, -2, "ber_max");
  lua_pushnumber(L, std_bdiff);
  lua_setfield(L, -2, "ber_std");

  free(bdiff_total);
  for (unsigned int i = 0; i < n_threads; i ++)
    free(data[i].bdiff);

  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline double _tm_auc (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool
) {
  if (state->mask != NULL && state->mask_popcount == 0)
    return 0.0;

  tk_threads_signal(pool, TK_EVAL_ENCODING_AUC, 0);
  ks_introsort(dl, state->n_pos + state->n_neg, state->pl);
  double sum_ranks = 0.0;
  unsigned int rank = 1;
  uint64_t n_pos_valid = 0, n_neg_valid = 0;
  for (uint64_t k = 0; k < state->n_pos + state->n_neg; k ++) {
    if (state->pl[k].sim == -1)
      continue;
    if (state->pl[k].label)
      sum_ranks += rank, n_pos_valid ++;
    else
      n_neg_valid ++;
    rank ++;
  }
  if (n_pos_valid == 0 || n_neg_valid == 0)
    return 0.0;
  double auc = (sum_ranks - ((double) n_pos_valid * (n_pos_valid + 1) / 2)) / ((double) n_pos_valid * n_neg_valid);
  return auc;
}

static inline int tm_auc (lua_State *L)
{
  lua_settop(L, 7);

  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  tk_dvec_t *dcodes = tk_dvec_peekopt(L, 2);
  tk_cvec_t *ccodes = dcodes == NULL ? tk_cvec_peekopt(L, 2) : NULL;
  tk_bits_t *codes = dcodes != NULL ? NULL : (ccodes != NULL ? (tk_bits_t *) ccodes->a : (tk_bits_t *) tk_lua_checkstring(L, 2, "codes"));
  if (!(dcodes != NULL || codes != NULL))
    tk_lua_verror(L, 3, "auc", "codes", "must be either a string, tk_cvec_t, or tk_dvec_t");

  tk_pvec_t *pos = tk_pvec_peek(L, 3, "pos");
  tk_pvec_t *neg = tk_pvec_peek(L, 4, "neg");
  uint64_t n_dims = tk_lua_checkunsigned(L, 5, "n_hidden");
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_bits_t *mask;
  tk_cvec_t *mvec = tk_cvec_peekopt(L, 6);
  mask = mvec != NULL ? (tk_bits_t *) mvec->a :
    (lua_type(L, 6) == LUA_TSTRING ? (tk_bits_t *) luaL_checkstring(L, 6) :
     lua_type(L, 6) == LUA_TLIGHTUSERDATA ? (tk_bits_t *) lua_touserdata(L, 6) : NULL);

  unsigned int n_threads = tk_threads_getn(L, 7, "n_threads", NULL);

  tk_eval_t state;
  memset(&state, 0, sizeof(tk_eval_t));
  state.n_dims = n_dims;
  state.chunks = TK_CVEC_BITS_BYTES(n_dims);
  state.pl = tk_malloc(L, (n_pos + n_neg) * sizeof(tm_dl_t));
  state.mask = mask;
  state.id_code = tk_iumap_from_ivec(0, ids);
  if (!state.id_code)
    tk_error(L, "tm_auc: iumap_from_ivec failed", ENOMEM);
  state.codes = codes;
  state.dcodes = dcodes != NULL ? dcodes->a : NULL;
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    atomic_init(&data[i].has_error, false);
    tk_thread_range(i, n_threads, n_pos + n_neg, &data[i].pfirst, &data[i].plast);
  }

  double auc = _tm_auc(L, &state, pool);

  free(state.pl);
  tk_iumap_destroy(state.id_code);

  lua_pushnumber(L, auc);
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tm_optimize_clustering (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "index");
  int i_index = tk_lua_absindex(L, -1);
  tk_inv_t *inv = tk_inv_peekopt(L, i_index);
  tk_ann_t *ann = tk_ann_peekopt(L, i_index);
  tk_hbi_t *hbi = tk_hbi_peekopt(L, i_index);

  if (inv == NULL && ann == NULL && hbi == NULL)
    tk_lua_verror(L, 3, "optimize_clustering", "index", "tk_inv_t, tk_ann_t, or tk_hbi_t must be provided");

  bool use_inv = (inv != NULL);
  uint64_t n_features = 0;
  if (!use_inv)
    n_features = ann != NULL ? ann->features : hbi->features;

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);
  int i_ids = ids == NULL ? -1 : tk_lua_absindex(L, -1);

  lua_getfield(L, 1, "pos");
  tk_pvec_t *pos = tk_pvec_peek(L, -1, "pos");

  lua_getfield(L, 1, "neg");
  tk_pvec_t *neg = tk_pvec_peek(L, -1, "neg");

  uint64_t min_pts = tk_lua_foptunsigned(L, 1, "optimize clustering", "min_pts", 0);
  bool assign_noise = tk_lua_foptboolean(L, 1, "optimize clustering", "assign_noise", true);
  uint64_t probe_radius = tk_lua_foptunsigned(L, 1, "optimize clustering", "probe_radius", 3);

  const char *methodstr = tk_lua_foptstring(L, 1, "optimize clustering", "method", "graph");
  tk_cluster_method_t method = TK_CLUSTER_METHOD_GRAPH;
  if (!strcmp(methodstr, "graph"))
    method = TK_CLUSTER_METHOD_GRAPH;
  else if (!strcmp(methodstr, "prefix"))
    method = TK_CLUSTER_METHOD_PREFIX;

  if (method == TK_CLUSTER_METHOD_PREFIX && (hbi == NULL && ann == NULL))
    tk_lua_verror(L, 3, "optimize_clustering", "method", "prefix method requires hbi or ann index");

  uint64_t min_margin = tk_lua_foptunsigned(L, 1, "optimize clustering", "min_margin", 0);
  uint64_t max_margin = 0;

  if (method == TK_CLUSTER_METHOD_PREFIX) {
    max_margin = tk_lua_foptunsigned(L, 1, "optimize clustering", "max_margin", n_features);
    if (max_margin > n_features) max_margin = n_features;
    if (min_margin > n_features) min_margin = n_features;
    if (max_margin < min_margin)
      max_margin = min_margin;
  } else if (!use_inv) {
    max_margin = tk_lua_foptunsigned(L, 1, "optimize clustering", "max_margin", n_features);
    uint64_t fs = ann != NULL ? ann->features : hbi != NULL ? hbi->features : 0;
    if (max_margin > fs) max_margin = fs;
    if (min_margin > fs) min_margin = fs;
    if (max_margin < min_margin)
      max_margin = min_margin;
  } else {
    max_margin = tk_lua_foptunsigned(L, 1, "optimize clustering", "max_margin", 100);
    if (min_margin >= max_margin)
      min_margin = 0;
  }

  const char *cmpstr = tk_lua_foptstring(L, 1, "optimize clustering", "cmp", "jaccard");
  double cmp_alpha = tk_lua_foptnumber(L, 1, "optimize clustering", "cmp_alpha", 1.0);
  double cmp_beta = tk_lua_foptnumber(L, 1, "optimize clustering", "cmp_beta", 0.1);
  int64_t rank_filter = tk_lua_foptinteger(L, 1, "optimize clustering", "rank_filter", -1);
  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(cmpstr, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(cmpstr, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(cmpstr, "tversky"))
    cmp = TK_IVEC_TVERSKY;
  else if (!strcmp(cmpstr, "dice"))
    cmp = TK_IVEC_DICE;

  unsigned int n_threads = tk_threads_getn(L, 1, "optimize clustering", "threads");

  uint64_t knn = tk_lua_foptunsigned(L, 1, "optimize clustering", "knn", 0);
  uint64_t knn_cache = tk_lua_foptunsigned(L, 1, "optimize clustering", "knn_cache", 0);
  if (knn > knn_cache)
    knn_cache = knn;
  uint64_t knn_min = tk_lua_foptunsigned(L, 1, "optimize clustering", "knn_min", 0);
  bool knn_mutual = tk_lua_foptboolean(L, 1, "optimize clustering", "knn_mutual", false);
  double knn_eps = tk_lua_foptposdouble(L, 1, "optimize clustering", "knn_eps", 1.0);
  int64_t knn_rank = tk_lua_foptinteger(L, 1, "optimize clustering", "knn_rank", -1);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_dvec_t *inv_thresholds = NULL;
  if (use_inv) {
    inv_thresholds = tk_dvec_create(0, max_margin, 0, 0);

    for (uint64_t i = 0; i < max_margin; i++) {
      double t = (double)i / (max_margin - 1);
      double log_min = log(1e-9);
      double log_max = log(1.0 + 1e-9);
      double log_eps = log_min + t * (log_max - log_min);
      inv_thresholds->a[i] = exp(log_eps);
    }
    inv_thresholds->n = max_margin;
  }

  tk_eval_t state;
  state.inv = inv;
  state.ann = ann;
  state.hbi = hbi;
  state.pos = pos;
  state.neg = neg;
  state.assign_noise = assign_noise;
  state.min_pts = min_pts;
  state.cluster_method = method;
  state.min_margin = min_margin;
  state.max_margin = max_margin;
  state.inv_thresholds = inv_thresholds;
  state.probe_radius = probe_radius;
  state.cmp = cmp;
  state.cmp_alpha = cmp_alpha;
  state.cmp_beta = cmp_beta;
  state.rank_filter = rank_filter;
  lua_newtable(L); // t -- to track gc for vectors
  int i_eph = tk_lua_absindex(L, -1);
  if (ids != NULL)
    lua_pushvalue(L, i_ids); // t ids
  state.ids =
    ids != NULL ? ids :
    inv ? tk_iumap_keys(L, inv->uid_sid) :
    hbi ? tk_iumap_keys(L, hbi->uid_sid) :
    ann ? tk_iumap_keys(L, ann->uid_sid) : tk_ivec_create(L, 0, 0, 0); // t ids
  state.ididx =
    tk_iumap_from_ivec(L, state.ids);
  if (!state.ididx)
    tk_error(L, "eval_create: iumap_from_ivec failed", ENOMEM);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  // Pre-compute neighborhoods if knn_cache is provided
  state.inv_hoods = NULL;
  state.ann_hoods = NULL;
  state.hbi_hoods = NULL;
  state.uids_hoods = NULL;
  state.uids_idx_hoods = NULL;

  if (knn_cache > 0) {
    if (inv != NULL) {
      tk_inv_neighborhoods(
        L, inv, knn_cache, knn_eps, knn_min, cmp,
        cmp_alpha, cmp_beta, knn_mutual, knn_rank, n_threads,
        &state.inv_hoods, &state.uids_hoods);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -2); // hoods
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1); // uids_hoods
      lua_pop(L, 2);
    } else if (ann != NULL) {
      tk_ann_neighborhoods(
        L, ann, knn_cache, probe_radius, ann->features * knn_eps, knn_min,
        knn_mutual, n_threads, &state.ann_hoods, &state.uids_hoods);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -2); // hoods
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1); // uids_hoods
      lua_pop(L, 2);
    } else if (hbi != NULL) {
      tk_hbi_neighborhoods(
        L, hbi, knn_cache, hbi->features * knn_eps, knn_min,
        knn_mutual, n_threads, &state.hbi_hoods, &state.uids_hoods);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -2); // hoods
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1); // uids_hoods
      lua_pop(L, 2);
    }

    if (state.uids_hoods != NULL) {
      state.uids_idx_hoods = tk_iumap_from_ivec(L, state.uids_hoods);
      if (!state.uids_idx_hoods)
        tk_error(L, "eval_create: iumap_from_ivec failed", ENOMEM);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
      lua_pop(L, 1);
    }
  }

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].self = pool->threads + i;
    data[i].state = &state;
    atomic_init(&data[i].has_error, false);
    data[i].best = (tk_accuracy_t) { .bacc = -1.0, .tpr = 0.0, .tnr = 0.0 };
    data[i].next_n = 0;
    data[i].best_n = 0;
    data[i].best_m = 0;
    data[i].next_assignments = tk_ivec_create(L, state.ids->n, 0, 0);
    data[i].best_assignments = tk_ivec_create(L, state.ids->n, 0, 0);
    data[i].ptmp = tk_pvec_create(L, 0, 0, 0);
    data[i].rtmp = tk_rvec_create(L, 0, 0, 0);
    data[i].dsu = NULL;
    data[i].is_core = NULL;
    data[i].prefix_cache = NULL;
    data[i].prefix_bytes = NULL;
    data[i].prefix_cache_depth = 0;
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -4);
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -3);
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -2);
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
    lua_pop(L, 4);
    uint64_t n_steps = use_inv ? (max_margin - min_margin) : (max_margin - min_margin + 1);
    tk_thread_range(i, n_threads, n_steps, &data[i].mfirst, &data[i].mlast);
  }

  unsigned int child;
  while (tk_threads_signal(pool, TK_EVAL_OPTIMIZE_CLUSTERING, &child)) {
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushnumber(L, data[child].next.bacc);
      lua_pushnumber(L, data[child].next.tpr);
      lua_pushnumber(L, data[child].next.tnr);
      lua_pushinteger(L, (lua_Integer) data[child].next_m);
      lua_pushinteger(L, (lua_Integer) data[child].next_n);
      lua_pushvalue(L, i_ids);
      tk_lua_get_ephemeron(L, TK_EVAL_EPH, data[child].next_assignments);
      lua_call(L, 7, 0);
      // TODO: Set things up such that memory is correctly freed even if this
      // throws
    }
    tk_threads_acknowledge_child(pool, child);
  }

  unsigned int i_best = 0;
  for (unsigned int i = 1; i < n_threads; i ++)
    if (data[i].best.bacc > data[i_best].best.bacc)
      i_best = i;

  tk_accuracy_t best = data[i_best].best;
  uint64_t best_m = data[i_best].best_m;

  if (inv_thresholds != NULL)
    tk_dvec_destroy(inv_thresholds);
  uint64_t best_n = data[i_best].best_n;
  tk_ivec_t *assignments = data[i_best].best_assignments;
  tk_lua_get_ephemeron(L, TK_EVAL_EPH, assignments); // t ids as

  tk_lua_del_ephemeron(L, TK_EVAL_EPH, i_eph, state.ididx);
  tk_iumap_destroy(state.ididx);

  lua_newtable(L); // t ids as score
  lua_pushinteger(L, (lua_Integer) best_m);
  if (method == TK_CLUSTER_METHOD_PREFIX) {
    lua_setfield(L, -2, "depth");
    lua_pushinteger(L, (lua_Integer) best_m);
    lua_setfield(L, -2, "margin");
  } else {
    lua_setfield(L, -2, "margin");
    lua_pushinteger(L, (lua_Integer) best_m);
    lua_setfield(L, -2, "depth");
  }
  lua_pushinteger(L, (lua_Integer) best_n);
  lua_setfield(L, -2, "n_clusters");
  lua_pushnumber(L, best.bacc);
  lua_setfield(L, -2, "bacc");
  lua_pushnumber(L, best.tpr);
  lua_setfield(L, -2, "tpr");
  lua_pushnumber(L, best.tnr);
  lua_setfield(L, -2, "tnr");
  lua_insert(L, -3); // t score ids as
  lua_pushinteger(L, (int64_t) best_n); // t score ids as n
  tk_lua_replace(L, 1, 4);
  lua_settop(L, 4);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 4;
}

static inline int tm_optimize_retrieval (lua_State *L)
{
  lua_settop(L, 1);

  tk_bits_t *codes;
  lua_getfield(L, 1, "codes");
  tk_cvec_t *cvec = tk_cvec_peekopt(L, -1);
  codes = cvec != NULL
    ? (tk_bits_t *) cvec->a
    : (tk_bits_t *) tk_lua_checkustring(L, -1, "codes");

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);

  lua_getfield(L, 1, "pos");
  tk_pvec_t *pos = tk_pvec_peek(L, -1, "pos");

  lua_getfield(L, 1, "neg");
  tk_pvec_t *neg = tk_pvec_peek(L, -1, "neg");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "optimize_retrieval", "n_dims");
  unsigned int n_threads = tk_threads_getn(L, 1, "optimize_retrieval", "threads");
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_eval_t state;
  state.n_dims = n_dims;
  state.chunks = TK_CVEC_BITS_BYTES(n_dims);
  state.pl = tk_malloc(L, (n_pos + n_neg) * sizeof(tm_dl_t));
  state.mask = NULL;
  state.codes = codes;
  state.dcodes = NULL;
  // TODO: Pass id_code (aka uid_hood) in instead of recomputing
  state.id_code = ids == NULL ? NULL : tk_iumap_from_ivec(0, ids);
  if (ids != NULL && !state.id_code)
    return 0;
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;
  state.hist_pos = tk_malloc(L, (n_dims + 1) * sizeof(atomic_ulong));
  state.hist_neg = tk_malloc(L, (n_dims + 1) * sizeof(atomic_ulong));
  for (uint64_t i = 0; i < n_dims + 1; i ++) {
    atomic_init(state.hist_pos + i, 0);
    atomic_init(state.hist_neg + i, 0);
  }

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    atomic_init(&data[i].has_error, false);
    tk_thread_range(i, n_threads, n_pos + n_neg, &data[i].pfirst, &data[i].plast);
  }

  double auc = _tm_auc(L, &state, pool);

  tk_threads_signal(pool, TK_EVAL_ENCODING_SIMILARITY, 0);

  free(state.pl);
  state.pl = NULL;

  uint64_t pref_tp[n_dims + 1], pref_fp[n_dims + 1];
  uint64_t running_tp = 0, running_fp = 0;
  for (uint64_t d = 0; d <= n_dims; d ++) {
    running_tp += atomic_load(state.hist_pos + d);
    running_fp += atomic_load(state.hist_neg + d);
    pref_tp[d] = running_tp;
    pref_fp[d] = running_fp;
  }

  free(state.hist_pos);
  free(state.hist_neg);
  state.hist_pos = NULL;
  state.hist_neg = NULL;

  double best_bacc = -1.0;
  double best_tpr = -1.0;
  double best_tnr = -1.0;
  uint64_t best_margin = 0;
  for (uint64_t m = 0; m <= n_dims; m ++) {
    uint64_t tp = pref_tp[m];
    uint64_t fp = pref_fp[m];
    uint64_t tn = n_neg - fp;
    double tpr = n_pos > 0 ? (double)tp / (double)n_pos : 0;
    double tnr = n_neg > 0 ? (double)tn / (double)n_neg : 0;
    double bacc = 0.5 * (tpr + tnr);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushnumber(L, bacc);
      lua_pushnumber(L, tpr);
      lua_pushnumber(L, tnr);
      lua_pushinteger(L, (lua_Integer)m);
      lua_pushnumber(L, auc);
      lua_call(L, 5, 1);
      if (lua_isboolean(L, -1) && !lua_toboolean(L, -1)) {
        lua_pop(L, 1);
        break;
      }
      lua_pop(L, 1);
    }
    if (bacc > best_bacc || (bacc == best_bacc && tpr > best_tpr)) {
      best_bacc = bacc;
      best_tpr = tpr;
      best_tnr = tnr;
      best_margin = m;
    }
  }

  if (state.id_code != NULL)
    tk_iumap_destroy(state.id_code);

  lua_newtable(L);
  lua_pushinteger(L, (int64_t) best_margin);
  lua_setfield(L, -2, "margin");
  lua_pushnumber(L, best_bacc);
  lua_setfield(L, -2, "bacc");
  lua_pushnumber(L, best_tpr);
  lua_setfield(L, -2, "tpr");
  lua_pushnumber(L, best_tnr);
  lua_setfield(L, -2, "tnr");
  lua_pushnumber(L, auc);
  lua_setfield(L, -2, "auc");
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static double tk_compute_reconstruction (
  tk_eval_t *state,
  tk_bits_t *mask,
  uint64_t k,
  tk_threadpool_t *pool
) {
  if (k == 0)
    return -INFINITY;
  state->mask = mask;
  state->mask_popcount = k;
  tk_eval_thread_t *data = (tk_eval_thread_t *)pool->threads[0].data;
  unsigned int n_threads = pool->n_threads;
  for (unsigned int i = 0; i < n_threads; i ++) {
    data[i].recon_score = 0.0;
    data[i].nodes_processed = 0;
  }
  tk_threads_signal(pool, TK_EVAL_GRAPH_RECONSTRUCTION, 0);
  double total = 0.0;
  uint64_t total_nodes_processed = 0;
  for (unsigned int i = 0; i < n_threads; i ++) {
    total += data[i].recon_score;
    total_nodes_processed += data[i].nodes_processed;
  }
  return total_nodes_processed > 0 ? total / (double) total_nodes_processed : 0.0;
}

static void tm_optimize_bits_prefix_greedy (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool,
  int i_each
) {
  uint64_t n_dims = state->n_dims;
  uint64_t keep_prefix = state->optimal_k;
  int64_t start_prefix = state->start_prefix;
  double tolerance = state->tolerance;
  uint64_t bytes_per_mask = TK_CVEC_BITS_BYTES(n_dims);
  tk_cvec_t *mask_cvec = tk_cvec_create(L, bytes_per_mask, 0, 0);
  tk_cvec_t *candidate_cvec = tk_cvec_create(L, bytes_per_mask, 0, 0);
  tk_bits_t *mask = (tk_bits_t *)mask_cvec->a;
  tk_bits_t *candidate = (tk_bits_t *)candidate_cvec->a;

  uint64_t best_prefix;
  double best_prefix_score;

  if (start_prefix < 0) {
    uint64_t min_prefix = keep_prefix > 0 ? keep_prefix : 1;
    best_prefix = min_prefix;
    best_prefix_score = -INFINITY;
    for (uint64_t k = min_prefix; k <= n_dims; k ++) {
      memset(mask, 0, bytes_per_mask);
      for (uint64_t b = 0; b < k; b ++)
        mask[TK_CVEC_BITS_BYTE(b)] |= (1 << TK_CVEC_BITS_BIT(b));
      double score = tk_compute_reconstruction(state, mask, k, pool);
      double gain = best_prefix_score == -INFINITY ? 0 : (score - best_prefix_score);
      if (i_each >= 0) {
        lua_pushvalue(L, i_each);
        lua_pushinteger(L, (lua_Integer) k);
        lua_pushnumber(L, gain);
        lua_pushnumber(L, score);
        lua_pushliteral(L, "scan");
        lua_call(L, 4, 0);
      }
      if (score > best_prefix_score + 1e-12) {
        best_prefix_score = score;
        best_prefix = k;
      }
    }
    if (i_each >= 0) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (lua_Integer) best_prefix);
      lua_pushnumber(L, 0.0);
      lua_pushnumber(L, best_prefix_score);
      lua_pushliteral(L, "init");
      lua_call(L, 4, 0);
    }
  } else if (start_prefix == 0) {
    best_prefix = 1;
    best_prefix_score = -INFINITY;
    uint64_t best_bit = 0;
    for (uint64_t b = 0; b < n_dims; b ++) {
      memset(mask, 0, bytes_per_mask);
      mask[TK_CVEC_BITS_BYTE(b)] |= (1 << TK_CVEC_BITS_BIT(b));
      double score = tk_compute_reconstruction(state, mask, 1, pool);
      if (score > best_prefix_score + 1e-12) {
        best_prefix_score = score;
        best_bit = b;
      }
    }
    memset(mask, 0, bytes_per_mask);
    mask[TK_CVEC_BITS_BYTE(best_bit)] |= (1 << TK_CVEC_BITS_BIT(best_bit));
    best_prefix = 1;
    if (i_each >= 0) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (lua_Integer) best_bit);
      lua_pushnumber(L, 0.0);
      lua_pushnumber(L, best_prefix_score);
      lua_pushliteral(L, "init");
      lua_call(L, 4, 0);
    }
  } else {
    best_prefix = (uint64_t) start_prefix;
    if (best_prefix > n_dims)
      best_prefix = n_dims;
    memset(mask, 0, bytes_per_mask);
    for (uint64_t b = 0; b < best_prefix; b ++)
      mask[TK_CVEC_BITS_BYTE(b)] |= (1 << TK_CVEC_BITS_BIT(b));
    best_prefix_score = tk_compute_reconstruction(state, mask, best_prefix, pool);
    if (i_each >= 0) {
      lua_pushvalue(L, i_each);
      lua_pushinteger(L, (lua_Integer) best_prefix);
      lua_pushnumber(L, 0.0);
      lua_pushnumber(L, best_prefix_score);
      lua_pushliteral(L, "scan");
      lua_call(L, 4, 0);
    }
  }

  tk_ivec_t *active = tk_ivec_create(L, best_prefix, 0, 0);
  if (start_prefix == 0) {
    for (uint64_t b = 0; b < n_dims; b ++) {
      if (mask[TK_CVEC_BITS_BYTE(b)] & (1 << TK_CVEC_BITS_BIT(b))) {
        active->a[0] = (int64_t)b;
        active->n = 1;
        break;
      }
    }
  } else {
    for (uint64_t i = 0; i < best_prefix; i ++)
      active->a[i] = (int64_t) i;
    active->n = best_prefix;
  }

  double current_score = best_prefix_score;

  bool improved = true;
  while (improved && active->n > 0) {
    improved = false;

    if (active->n > 1)

      // Try removing
      for (uint64_t i = 0; i < active->n; i ++) {
        int64_t bit = active->a[i];
        if (keep_prefix > 0 && bit < (int64_t)keep_prefix)
          continue;
        memcpy(candidate, mask, bytes_per_mask);
        candidate[TK_CVEC_BITS_BYTE(bit)] &= ~(1 << TK_CVEC_BITS_BIT(bit));
        double score = tk_compute_reconstruction(state, candidate, active->n - 1, pool);
        if (score > current_score + 1e-12) {
          mask[TK_CVEC_BITS_BYTE(bit)] &= ~(1 << TK_CVEC_BITS_BIT(bit));
          active->a[i] = active->a[active->n - 1];
          active->n --;
          double gain = score - current_score;
          current_score = score;
          improved = true;
          if (i_each >= 0) {
            lua_pushvalue(L, i_each);
            lua_pushinteger(L, bit);
            lua_pushnumber(L, gain);
            lua_pushnumber(L, current_score);
            lua_pushliteral(L, "remove");
            lua_call(L, 4, 0);
          }
          break;
        }
      }

    // Try adding
    if (!improved && active->n < n_dims) {
      for (int64_t bit_add = 0; bit_add < (int64_t) n_dims; bit_add ++) {
        bool is_active = false;
        for (uint64_t j = 0; j < active->n; j ++) {
          if (active->a[j] == bit_add) {
            is_active = true;
            break;
          }
        }
        if (is_active)
          continue;
        memcpy(candidate, mask, bytes_per_mask);
        candidate[TK_CVEC_BITS_BYTE(bit_add)] |= (1 << TK_CVEC_BITS_BIT(bit_add));
        double score = tk_compute_reconstruction(state, candidate, active->n + 1, pool);
        if (score >= current_score - tolerance) {
          mask[TK_CVEC_BITS_BYTE(bit_add)] |= (1 << TK_CVEC_BITS_BIT(bit_add));
          if (tk_ivec_push(active, bit_add) != 0)
            tk_lua_verror(L, 2, "optimize_bits", "allocation failed");
          double gain = current_score == -INFINITY ? 0 : (score - current_score);
          current_score = score;
          improved = true;
          if (i_each >= 0) {
            lua_pushvalue(L, i_each);
            lua_pushinteger(L, bit_add);
            lua_pushnumber(L, gain);
            lua_pushnumber(L, current_score);
            lua_pushliteral(L, "add");
            lua_call(L, 4, 0);
          }
          break;
        }
      }
    }

    // Try swapping
    if (!improved && active->n > 0) {
      for (uint64_t i = 0; i < active->n && !improved; i ++) {
        int64_t bit_out = active->a[i];
        if (keep_prefix > 0 && bit_out < (int64_t)keep_prefix)
          continue;
        for (int64_t bit_in = 0; bit_in < (int64_t) n_dims; bit_in ++) {
          bool is_active = false;
          for (uint64_t j = 0; j < active->n; j ++) {
            if (active->a[j] == bit_in) {
              is_active = true;
              break;
            }
          }
          if (is_active)
            continue;
          memcpy(candidate, mask, bytes_per_mask);
          candidate[TK_CVEC_BITS_BYTE(bit_out)] &= ~(1 << TK_CVEC_BITS_BIT(bit_out));
          candidate[TK_CVEC_BITS_BYTE(bit_in)] |= (1 << TK_CVEC_BITS_BIT(bit_in));
          double score = tk_compute_reconstruction(state, candidate, active->n, pool);
          if (score > current_score + 1e-12) {
            mask[TK_CVEC_BITS_BYTE(bit_out)] &= ~(1 << TK_CVEC_BITS_BIT(bit_out));
            mask[TK_CVEC_BITS_BYTE(bit_in)] |= (1 << TK_CVEC_BITS_BIT(bit_in));
            active->a[i] = bit_in;
            double gain = score - current_score;
            current_score = score;
            improved = true;
            if (i_each >= 0) {
              lua_pushvalue(L, i_each);
              lua_pushinteger(L, bit_in);
              lua_pushnumber(L, gain);
              lua_pushnumber(L, current_score);
              lua_pushliteral(L, "swap");
              lua_call(L, 4, 0);
            }
            break;
          }
        }
      }
    }

  }

  tk_ivec_asc(active, 0, active->n);
}

static inline int tm_optimize_bits (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_cvec_t *cvec = tk_cvec_peek(L, -1, "codes");

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "neighbors");

  lua_getfield(L, 1, "weights");
  tk_dvec_t *weights = tk_dvec_peek(L, -1, "weights");

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "optimize_bits", "n_dims");
  unsigned int n_threads = tk_threads_getn(L, 1, "optimize_bits", "threads");
  uint64_t keep_prefix = tk_lua_foptunsigned(L, 1, "optimize_bits", "keep_prefix", 0);
  int64_t start_prefix = tk_lua_foptinteger(L, 1, "optimize_bits", "start_prefix", 0);
  double tolerance = tk_lua_foptnumber(L, 1, "optimize_bits", "tolerance", -1e12);

  const char *metric_str = tk_lua_foptstring(L, 1, "optimize_bits", "correlation", "spearman");
  int correlation_metric = 0;
  if (!strcmp(metric_str, "spearman"))
    correlation_metric = 0;
  else if (!strcmp(metric_str, "kendall"))
    correlation_metric = 1;
  else if (!strcmp(metric_str, "spearman-weighted"))
    correlation_metric = 2;
  else if (!strcmp(metric_str, "kendall-weighted"))
    correlation_metric = 3;
  else
    tk_lua_verror(L, 3, "optimize_bits", "correlation", "must be 'spearman', 'kendall', 'spearman-weighted', or 'kendall-weighted'");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_eval_t state;
  memset(&state, 0, sizeof(tk_eval_t));
  state.n_dims = n_dims;
  state.chunks = TK_CVEC_BITS_BYTES(n_dims);
  state.codes = (tk_bits_t *) cvec->a;
  state.offsets = offsets;
  state.neighbors = neighbors;
  state.weights = weights;
  state.optimal_k = keep_prefix;
  state.start_prefix = start_prefix;
  state.tolerance = tolerance;
  state.correlation_metric = correlation_metric;
  state.L = L;

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  uint64_t n_nodes = offsets->n - 1; // Number of nodes
  for (unsigned int i = 0; i < n_threads; i++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    atomic_init(&data[i].has_error, false);
    tk_thread_range(i, n_threads, n_nodes, &data[i].wfirst, &data[i].wlast);
    tk_thread_range(i, n_threads, n_nodes, &data[i].sfirst, &data[i].slast);
  }

  tk_threads_signal(pool, TK_EVAL_GRAPH_RECONSTRUCTION_INIT, 0);

  // Check for allocation errors in worker threads
  for (unsigned int i = 0; i < n_threads; i++) {
    if (atomic_load(&data[i].has_error)) {
      tk_threads_signal(pool, TK_EVAL_GRAPH_RECONSTRUCTION_DESTROY, 0);
      tk_lua_verror(L, 2, "optimize_bits", "worker thread allocation failed");
      return 0;
    }
  }
  tm_optimize_bits_prefix_greedy(L, &state, pool, i_each); // result
  tk_threads_signal(pool, TK_EVAL_GRAPH_RECONSTRUCTION_DESTROY, 0);
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tm_entropy_stats (lua_State *L)
{
  lua_settop(L, 3);
  unsigned int n_samples = tk_lua_checkunsigned(L, 2, "n_samples");
  unsigned int n_dims = tk_lua_checkunsigned(L, 3, "n_hidden");
  tk_cvec_t *cvec = tk_cvec_peekopt(L, 1);
  tk_ivec_t *ivec = NULL;
  tk_ivec_t *ids = NULL;
  tk_dvec_t *entropies = NULL;
  if (cvec == NULL) {
    ivec = tk_ivec_peekopt(L, 1);
    tk_ivec_bits_top_entropy(L, ivec, n_samples, n_dims, n_dims);
    ids = tk_ivec_peek(L, -2, "ids");
    entropies = tk_dvec_peek(L, -1, "entropies");
  } else {
    tk_cvec_bits_top_entropy(L, cvec, n_samples, n_dims, n_dims);
    ids = tk_ivec_peek(L, -2, "ids");
    entropies = tk_dvec_peek(L, -1, "entropies");
  }

  double min_entropy = 1.0, max_entropy = 0.0, sum_entropy = 0.0;
  lua_newtable(L); // result
  lua_newtable(L); // per-bit entropy table
  for (uint64_t j = 0; j < n_dims; j ++) {
    double entropy = entropies->a[j];
    lua_pushinteger(L, (int64_t) ids->a[j] + 1);
    lua_pushnumber(L, entropy);
    lua_settable(L, -3);
    if (entropy < min_entropy)
      min_entropy = entropy;
    if (entropy > max_entropy)
      max_entropy = entropy;
    sum_entropy += entropy;
  }
  lua_setfield(L, -2, "bits");

  double mean = sum_entropy / n_dims;
  double variance = 0.0;
  for (uint64_t j = 0; j < n_dims; j ++) {
    double entropy = entropies->a[j];
    variance += (entropy - mean) * (entropy - mean);
  }

  variance /= n_dims;
  lua_pushnumber(L, mean);
  lua_setfield(L, -2, "mean");
  lua_pushnumber(L, min_entropy);
  lua_setfield(L, -2, "min");
  lua_pushnumber(L, max_entropy);
  lua_setfield(L, -2, "max");
  lua_pushnumber(L, sqrt(variance));
  lua_setfield(L, -2, "std");
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static luaL_Reg tm_evaluator_fns[] =
{
  { "class_accuracy", tm_class_accuracy },
  { "encoding_accuracy", tm_encoding_accuracy },
  { "clustering_accuracy", tm_clustering_accuracy_lua },
  { "optimize_bits", tm_optimize_bits },
  { "optimize_retrieval", tm_optimize_retrieval },
  { "optimize_clustering", tm_optimize_clustering },
  { "entropy_stats", tm_entropy_stats },
  { "auc", tm_auc },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_evaluator_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_evaluator_fns, 0); // t
  return 1;
}
