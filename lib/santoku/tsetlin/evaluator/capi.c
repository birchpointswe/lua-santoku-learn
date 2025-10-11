#include <santoku/iuset.h>
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
  TK_EVAL_ENCODING_ACCURACY,
  TK_EVAL_ENCODING_AUC,
  TK_EVAL_GRAPH_RECONSTRUCTION_INIT,
  TK_EVAL_GRAPH_RECONSTRUCTION,
  TK_EVAL_GRAPH_RECONSTRUCTION_DESTROY,
  TK_EVAL_CLUSTERING_QUALITY,
  TK_EVAL_RETRIEVAL_QUALITY,
} tk_eval_stage_t;

typedef enum {
  TK_QUALITY_NONE,
  TK_QUALITY_BISERIAL,
  TK_QUALITY_VARIANCE,
} tk_quality_metric_t;

typedef enum {
  TK_CORRELATION_NONE,
  TK_CORRELATION_SPEARMAN,
  TK_CORRELATION_KENDALL,
  TK_CORRELATION_VARIANCE,
} tk_correlation_metric_t;

typedef struct {
  tm_dl_t *pl;
  atomic_ulong *TP, *FP, *FN;
  tk_ivec_t *ids;
  tk_inv_t *inv;
  tk_ann_t *ann;
  tk_hbi_t *hbi;
  uint64_t min_pts;
  bool assign_noise;
  tk_dvec_t *inv_thresholds;
  uint64_t probe_radius;
  tk_ivec_sim_type_t cmp;
  double cmp_alpha, cmp_beta;
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
  tk_quality_metric_t quality_metric;
  tk_correlation_metric_t correlation_metric;
  unsigned int agglo_n_threads;
  uint64_t margin;
} tk_eval_t;

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
  tk_rvec_t *rtmp;
  tk_iuset_t *itmp;
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
  double corr_score;
  uint64_t corr_nodes_processed;
  double diagnostic_p_sum;
  double diagnostic_std_sum;
} tk_eval_thread_t;

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
      data->itmp = tk_iuset_create(NULL, 0);
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
        if (state->correlation_metric == TK_CORRELATION_SPEARMAN)
          tk_rvec_ranks(data->adj_ranks[i], data->adj_ranks_idx[i]);
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
          case TK_CORRELATION_SPEARMAN:
            tk_pvec_ranks(data->bin_ranks, data->bin_ranks_idx);
            corr = tk_dumap_correlation(data->adj_ranks_idx[i], data->bin_ranks_idx);
            break;
          case TK_CORRELATION_KENDALL:
            corr = tk_rvec_kendall_pvec(data->adj_ranks[i], data->bin_ranks);
            break;
          case TK_CORRELATION_VARIANCE:
            tk_iuset_clear(data->itmp);
            for (uint64_t j = 0; j < data->adj_ranks[i]->n / 2; j++) {
              int kha;
              tk_iuset_put(data->itmp, data->adj_ranks[i]->a[j].i, &kha);
            }
            corr = tk_pvec_variance_ratio_binary(data->bin_ranks, data->itmp, NULL, NULL);
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
      if (data->adj_ranks != NULL) {
        uint64_t n_alloc = data->wlast - data->wfirst + 1;
        for (uint64_t i = 0; i < n_alloc; i ++) {
          if (data->adj_ranks[i] != NULL)
            tk_rvec_destroy(data->adj_ranks[i]);
          if (data->adj_ranks_idx[i] != NULL)
            tk_dumap_destroy(data->adj_ranks_idx[i]);
        }
        free(data->adj_ranks);
        free(data->adj_ranks_idx);
      }
      if (data->bin_ranks != NULL)
        tk_pvec_destroy(data->bin_ranks);
      if (data->bin_ranks_idx != NULL)
        tk_dumap_destroy(data->bin_ranks_idx);
      if (data->itmp != NULL)
        tk_iuset_destroy(data->itmp);
      break;
    }

    case TK_EVAL_CLUSTERING_QUALITY: {
      data->corr_score = 0.0;
      data->corr_nodes_processed = 0;
      data->diagnostic_p_sum = 0.0;
      data->diagnostic_std_sum = 0.0;

      // For each node, compute quality score (silhouette or biserial)
      for (uint64_t node_idx = data->wfirst; node_idx <= data->wlast; node_idx++) {
        int64_t start = state->offsets->a[node_idx];
        int64_t end = state->offsets->a[node_idx + 1];
        uint64_t n_neighbors = (uint64_t)(end - start);
        if (n_neighbors == 0)
          continue;

        int64_t my_cluster = state->assignments->a[node_idx];

        // Reuse thread-local structures
        tk_rvec_clear(data->rtmp);
        tk_iuset_clear(data->itmp);

        // Build neighbor weight vector and same-cluster set
        for (int64_t idx = start; idx < end; idx++) {
          int64_t neighbor = state->neighbors->a[idx];
          double weight = state->weights->a[idx];
          int64_t neighbor_cluster = state->assignments->a[neighbor];

          if (tk_rvec_push(data->rtmp, tk_rank(neighbor, weight)) != 0) {
            atomic_store(&data->has_error, true);
            return;
          }

          if (my_cluster == neighbor_cluster) {
            int kha;
            tk_iuset_put(data->itmp, neighbor, &kha);
          }
        }

        tk_rvec_desc(data->rtmp, 0, data->rtmp->n);

        double node_score = 0.0;

        if (state->quality_metric == TK_QUALITY_BISERIAL) {
          double mean_rank_0 = 0.0, mean_rank_1 = 0.0;
          node_score = tk_rvec_rank_biserial_binary(data->rtmp, data->itmp,
                                                     &mean_rank_0, &mean_rank_1);
          // Diagnostic output
          data->diagnostic_p_sum += mean_rank_1;    // same-cluster mean rank
          data->diagnostic_std_sum += mean_rank_0;  // different-cluster mean rank
        } else if (state->quality_metric == TK_QUALITY_VARIANCE) {
          double var_0, var_1;
          node_score = tk_rvec_variance_ratio_binary(data->rtmp, data->itmp,
                                                      &var_0, &var_1);
          // Diagnostic output
          data->diagnostic_p_sum += var_1;   // same-cluster variance
          data->diagnostic_std_sum += var_0; // different-cluster variance
        }

        data->corr_score += node_score;
        data->corr_nodes_processed++;
      }
      break;
    }

    case TK_EVAL_RETRIEVAL_QUALITY: {
      data->corr_score = 0.0;
      data->corr_nodes_processed = 0;

      // For each node, compute quality score (biserial or variance)
      for (uint64_t node_idx = data->wfirst; node_idx <= data->wlast; node_idx++) {
        int64_t start = state->offsets->a[node_idx];
        int64_t end = state->offsets->a[node_idx + 1];
        uint64_t n_neighbors = (uint64_t)(end - start);
        if (n_neighbors == 0)
          continue;

        // Reuse thread-local structures
        tk_rvec_clear(data->rtmp);
        tk_iuset_clear(data->itmp);

        // Build neighbor weight vector and within-margin set
        for (int64_t idx = start; idx < end; idx++) {
          int64_t neighbor = state->neighbors->a[idx];
          double weight = state->weights->a[idx];

          if (tk_rvec_push(data->rtmp, tk_rank(neighbor, weight)) != 0) {
            atomic_store(&data->has_error, true);
            return;
          }

          uint64_t hamming_dist = tk_cvec_bits_hamming(
            (const unsigned char *) state->codes + node_idx * state->chunks,
            (const unsigned char *) state->codes + neighbor * state->chunks,
            state->n_dims);

          if (hamming_dist <= state->margin) {
            int kha;
            tk_iuset_put(data->itmp, neighbor, &kha);
          }
        }

        tk_rvec_desc(data->rtmp, 0, data->rtmp->n);

        double node_score = 0.0;

        if (state->quality_metric == TK_QUALITY_BISERIAL) {
          double mean_rank_0 = 0.0, mean_rank_1 = 0.0;
          node_score = tk_rvec_rank_biserial_binary(data->rtmp, data->itmp,
                                                     &mean_rank_0, &mean_rank_1);
          // Diagnostic output
          data->diagnostic_p_sum += mean_rank_1;    // within-margin mean rank
          data->diagnostic_std_sum += mean_rank_0;  // outside-margin mean rank
        } else if (state->quality_metric == TK_QUALITY_VARIANCE) {
          double var_0, var_1;
          node_score = tk_rvec_variance_ratio_binary(data->rtmp, data->itmp,
                                                      &var_0, &var_1);
          // Diagnostic output
          data->diagnostic_p_sum += var_1;   // within-margin variance
          data->diagnostic_std_sum += var_0; // outside-margin variance
        }

        data->corr_score += node_score;
        data->corr_nodes_processed++;
      }
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

static inline void tm_clustering_accuracy (
  lua_State *L,
  tk_ivec_t *assignments,
  tk_ivec_t *offsets,
  tk_ivec_t *neighbors,
  tk_dvec_t *weights,
  unsigned int n_threads,
  tk_quality_metric_t quality,
  double *out_score
) {

  if (offsets->n == 0) {
    *out_score = 0.0;
    return;
  }

  uint64_t n_nodes = offsets->n - 1;
  if (n_nodes == 0) {
    *out_score = 0.0;
    return;
  }

  tk_eval_t state;
  state.assignments = assignments;
  state.offsets = offsets;
  state.neighbors = neighbors;
  state.weights = weights;
  state.quality_metric = quality;

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);

  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    atomic_init(&data[i].has_error, false);
    data[i].corr_score = 0.0;
    data[i].corr_nodes_processed = 0;
    data[i].diagnostic_p_sum = 0.0;
    data[i].diagnostic_std_sum = 0.0;
    data[i].rtmp = tk_rvec_create(NULL, 0, 0, 0);
    data[i].itmp = tk_iuset_create(NULL, 0);
    tk_thread_range(i, n_threads, n_nodes, &data[i].wfirst, &data[i].wlast);
  }

  tk_threads_signal(pool, TK_EVAL_CLUSTERING_QUALITY, 0);

  // Cleanup reusable structures
  for (unsigned int i = 0; i < n_threads; i++) {
    if (data[i].rtmp) tk_rvec_destroy(data[i].rtmp);
    if (data[i].itmp) tk_iuset_destroy(data[i].itmp);
  }

  // Aggregate across threads
  double total_corr_score = 0.0;
  uint64_t total_nodes_processed = 0;
  double total_p_sum = 0.0;
  double total_std_sum = 0.0;
  for (unsigned int i = 0; i < n_threads; i++) {
    total_corr_score += data[i].corr_score;
    total_nodes_processed += data[i].corr_nodes_processed;
    total_p_sum += data[i].diagnostic_p_sum;
    total_std_sum += data[i].diagnostic_std_sum;
  }

  *out_score = total_nodes_processed > 0 ? total_corr_score / total_nodes_processed : 0.0;
}

static inline int tm_clustering_accuracy_lua (lua_State *L)
{
  lua_settop(L, 1);

  // Parse parameters
  lua_getfield(L, 1, "assignments");
  tk_ivec_t *assignments = tk_ivec_peek(L, -1, "assignments");
  lua_pop(L, 1);

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "weights");
  tk_dvec_t *weights = tk_dvec_peek(L, -1, "weights");
  lua_pop(L, 1);

  unsigned int n_threads = tk_threads_getn(L, 1, "clustering_accuracy", "threads");

  char *metric_str = tk_lua_foptstring(L, 1, "clustering_accuracy", "metric", "correlation");
  tk_quality_metric_t quality = TK_QUALITY_NONE;
  if (!strcmp(metric_str, "correlation") || !strcmp(metric_str, "biserial"))
    quality = TK_QUALITY_BISERIAL;
  else if (!strcmp(metric_str, "variance"))
    quality = TK_QUALITY_VARIANCE;
  else
    tk_lua_verror(L, 3, "clustering_accuracy", "metric", "must be 'correlation', 'variance', or 'biserial'");

  double score;
  tm_clustering_accuracy(L, assignments, offsets, neighbors, weights, n_threads, quality, &score);

  lua_newtable(L);
  lua_pushnumber(L, score);
  lua_setfield(L, -2, "score");
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline void tm_retrieval_accuracy (
  lua_State *L,
  tk_bits_t *codes,
  uint64_t n_dims,
  uint64_t chunks,
  tk_ivec_t *offsets,
  tk_ivec_t *neighbors,
  tk_dvec_t *weights,
  uint64_t margin,
  unsigned int n_threads,
  tk_quality_metric_t quality,
  double *out_score
) {
  if (offsets->n == 0) {
    *out_score = 0.0;
    return;
  }

  uint64_t n_nodes = offsets->n - 1;
  if (n_nodes == 0) {
    *out_score = 0.0;
    return;
  }

  tk_eval_t state;
  state.codes = codes;
  state.n_dims = n_dims;
  state.chunks = chunks;
  state.offsets = offsets;
  state.neighbors = neighbors;
  state.weights = weights;
  state.margin = margin;
  state.quality_metric = quality;

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);

  for (unsigned int i = 0; i < n_threads; i++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    atomic_init(&data[i].has_error, false);
    data[i].corr_score = 0.0;
    data[i].corr_nodes_processed = 0;
    data[i].diagnostic_p_sum = 0.0;
    data[i].diagnostic_std_sum = 0.0;
    data[i].rtmp = tk_rvec_create(NULL, 0, 0, 0);
    data[i].itmp = tk_iuset_create(NULL, 0);
    tk_thread_range(i, n_threads, n_nodes, &data[i].wfirst, &data[i].wlast);
  }

  tk_threads_signal(pool, TK_EVAL_RETRIEVAL_QUALITY, 0);

  // Cleanup reusable structures
  for (unsigned int i = 0; i < n_threads; i++) {
    if (data[i].rtmp) tk_rvec_destroy(data[i].rtmp);
    if (data[i].itmp) tk_iuset_destroy(data[i].itmp);
  }

  // Aggregate quality scores across threads
  double total_quality_score = 0.0;
  uint64_t total_nodes_processed = 0;
  double total_p_sum = 0.0;
  double total_std_sum = 0.0;
  for (unsigned int i = 0; i < n_threads; i++) {
    total_quality_score += data[i].corr_score;
    total_nodes_processed += data[i].corr_nodes_processed;
    total_p_sum += data[i].diagnostic_p_sum;
    total_std_sum += data[i].diagnostic_std_sum;
  }

  if (total_nodes_processed == 0) {
    *out_score = 0.0;
    return;
  }

  *out_score = total_quality_score / total_nodes_processed;
}

static inline int tm_retrieval_accuracy_lua (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_cvec_t *cvec = tk_cvec_peek(L, -1, "codes");
  tk_bits_t *codes = (tk_bits_t *) cvec->a;
  lua_pop(L, 1);

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "weights");
  tk_dvec_t *weights = tk_dvec_peek(L, -1, "weights");
  lua_pop(L, 1);

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "retrieval_accuracy", "n_dims");
  uint64_t margin = tk_lua_fcheckunsigned(L, 1, "retrieval_accuracy", "margin");
  unsigned int n_threads = tk_threads_getn(L, 1, "retrieval_accuracy", "threads");

  char *metric_str = tk_lua_foptstring(L, 1, "retrieval_accuracy", "metric", "correlation");
  tk_quality_metric_t quality = TK_QUALITY_NONE;
  if (!strcmp(metric_str, "correlation") || !strcmp(metric_str, "biserial"))
    quality = TK_QUALITY_BISERIAL;
  else if (!strcmp(metric_str, "variance"))
    quality = TK_QUALITY_VARIANCE;
  else
    tk_lua_verror(L, 3, "retrieval_accuracy", "metric", "must be 'correlation', 'variance', or 'biserial'");

  uint64_t chunks = TK_CVEC_BITS_BYTES(n_dims);

  double score;
  tm_retrieval_accuracy(L, codes, n_dims, chunks, offsets, neighbors, weights,
                        margin, n_threads, quality, &score);

  lua_newtable(L);
  lua_pushnumber(L, score);
  lua_setfield(L, -2, "score");
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

typedef struct {
  lua_State *L;
  int i_each;
  int i_ids;
  int i_eph;
  tk_ivec_t *offsets;
  tk_ivec_t *neighbors;
  tk_dvec_t *weights;
  unsigned int n_threads;
  tk_quality_metric_t quality_metric;
  double tolerance;
  tk_ivec_t *dendro_offsets;
  tk_pvec_t *dendro_merges;
  tk_dvec_t *scores;
  tk_ivec_t *n_clusters;
  uint64_t n_samples;
} agglo_callback_data_t;

static inline tk_pvec_t *update_parent_from_diff(
  lua_State *L,
  const tk_ivec_t *prev_assignments,
  const tk_ivec_t *curr_assignments,
  uint64_t n_samples
) {
  // Collect merge pairs: (absorbed_cluster, surviving_cluster)
  tk_pvec_t *merges = tk_pvec_create(L, 0, 0, 0);
  if (!merges) return NULL;

  // Track which old clusters merged into which surviving clusters
  tk_iuset_t *surviving_ids = tk_iuset_create(NULL, 0);
  if (!surviving_ids) {
    tk_pvec_destroy(merges);
    return NULL;
  }

  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t surviving_id = curr_assignments->a[i];
    if (surviving_id >= 0) {
      int absent;
      tk_iuset_put(surviving_ids, surviving_id, &absent);
    }
  }

  int64_t surviving_cluster;
  tk_umap_foreach_keys(surviving_ids, surviving_cluster, ({
    // Collect all old cluster IDs that merged into this surviving cluster
    tk_iuset_t *old_ids = tk_iuset_create(NULL, 0);
    if (!old_ids) continue;

    for (uint64_t i = 0; i < n_samples; i++) {
      if (curr_assignments->a[i] == surviving_cluster && prev_assignments->a[i] >= 0) {
        int absent;
        tk_iuset_put(old_ids, prev_assignments->a[i], &absent);
      }
    }

    // If multiple old clusters merged
    if (tk_iuset_size(old_ids) > 1) {
      // Add merge pairs: all absorbed clusters point to surviving cluster
      int64_t old_id;
      tk_umap_foreach_keys(old_ids, old_id, ({
        if (old_id != surviving_cluster && old_id >= 0) {
          tk_pvec_push(merges, tk_pair(old_id, surviving_cluster));
        }
      }));
    }

    tk_iuset_destroy(old_ids);
  }));
  tk_iuset_destroy(surviving_ids);
  return merges;
}

static inline tk_ivec_t *tk_pvec_dendro_cut(
  lua_State *L,
  tk_ivec_t *offsets,
  tk_pvec_t *merges,
  uint64_t step,
  tk_ivec_t *assignments
) {
  // Find n_samples: scan offsets for first i where offsets[i]==0 after initial values
  uint64_t n_samples = 0;
  for (uint64_t i = 0; i < offsets->n; i++) {
    if (offsets->a[i] == 0 && i > 0) {
      n_samples = i;
      break;
    }
  }

  if (n_samples == 0 || n_samples > offsets->n) {
    tk_error(L, "tk_pvec_dendro_cut: invalid dendro_offsets structure", EINVAL);
  }

  // Ensure assignments has correct size
  if (assignments->m < n_samples) {
    tk_ivec_ensure(assignments, n_samples);
  }

  // Copy initial assignments
  for (uint64_t i = 0; i < n_samples; i++) {
    assignments->a[i] = offsets->a[i];
  }
  assignments->n = n_samples;

  // Build absorption map from merges in steps 0..step-1
  tk_iumap_t *absorbed_to_surviving = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, -1, -1);
  lua_pop(L, 1);

  for (uint64_t s = 0; s < step && s + n_samples < offsets->n; s++) {
    int64_t start = offsets->a[n_samples + s];
    int64_t end = (s + n_samples + 1 < offsets->n) ? offsets->a[n_samples + s + 1] : (int64_t)merges->n;

    for (int64_t idx = start; idx < end && idx < (int64_t)merges->n; idx++) {
      tk_pair_t merge = merges->a[idx];
      int64_t absorbed = merge.i;
      int64_t surviving = merge.p;

      int kha;
      khint_t khi = tk_iumap_put(absorbed_to_surviving, absorbed, &kha);
      tk_iumap_setval(absorbed_to_surviving, khi, surviving);
    }
  }

  // Resolve chains: follow absorption map to find final surviving cluster
  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t cluster = assignments->a[i];
    uint64_t chain_limit = 10000;  // Prevent infinite loops
    uint64_t chain_count = 0;

    while (chain_count < chain_limit) {
      khint_t khi = tk_iumap_get(absorbed_to_surviving, cluster);
      if (khi == tk_iumap_end(absorbed_to_surviving)) {
        break;  // Not absorbed, this is final cluster
      }
      cluster = tk_iumap_val(absorbed_to_surviving, khi);
      chain_count++;
    }

    assignments->a[i] = cluster;
  }

  // Remap to consecutive cluster IDs
  tk_iumap_t *cluster_remap = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, -1, -1);
  lua_pop(L, 1);

  int64_t next_id = 0;
  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t cluster = assignments->a[i];
    khint_t khi = tk_iumap_get(cluster_remap, cluster);

    if (khi == tk_iumap_end(cluster_remap)) {
      int kha;
      khi = tk_iumap_put(cluster_remap, cluster, &kha);
      tk_iumap_setval(cluster_remap, khi, next_id);
      assignments->a[i] = next_id;
      next_id++;
    } else {
      assignments->a[i] = tk_iumap_val(cluster_remap, khi);
    }
  }

  tk_iumap_destroy(absorbed_to_surviving);
  tk_iumap_destroy(cluster_remap);

  return assignments;
}

static inline void agglo_snapshot_callback (
  void *user_data,
  uint64_t iteration,
  uint64_t n_active_clusters,
  tk_ivec_t *ids,
  tk_ivec_t *snapshot_assignments
) {
  assert(ids->n == snapshot_assignments->n);
  agglo_callback_data_t *data = (agglo_callback_data_t *)user_data;

  if (iteration == 0) {
    // Initialize dendrogram: store initial assignments in offsets[0..n-1], set offsets[n]=0
    for (uint64_t i = 0; i < data->n_samples; i++) {
      data->dendro_offsets->a[i] = snapshot_assignments->a[i];
    }
    data->dendro_offsets->a[data->n_samples] = 0;  // First merge index starts at 0
    data->dendro_offsets->n = data->n_samples + 1;

    // Also need prev_assignments for next iteration
    tk_ivec_t *prev_assignments = tk_ivec_create(data->L, data->n_samples, 0, 0);
    tk_ivec_copy(prev_assignments, snapshot_assignments, 0, (int64_t)snapshot_assignments->n, 0);
    tk_lua_add_ephemeron(data->L, TK_EVAL_EPH, data->i_eph, -1);
    lua_pop(data->L, 1);
    // Store prev_assignments in a temporary location (we'll use i_eph ephemeron table)
    lua_pushlightuserdata(data->L, prev_assignments);
    lua_setfield(data->L, data->i_eph, "__prev_assignments");
  } else {
    // Retrieve prev_assignments
    lua_getfield(data->L, data->i_eph, "__prev_assignments");
    tk_ivec_t *prev_assignments = (tk_ivec_t *)lua_touserdata(data->L, -1);
    lua_pop(data->L, 1);

    // Get merge pairs from diff
    tk_pvec_t *step_merges = update_parent_from_diff(data->L, prev_assignments, snapshot_assignments, data->n_samples);

    // Append merges to dendro_merges
    if (step_merges) {
      for (uint64_t i = 0; i < step_merges->n; i++) {
        tk_pvec_push(data->dendro_merges, step_merges->a[i]);
      }
      tk_pvec_destroy(step_merges);
    }

    // Push new offset = current dendro_merges->n
    tk_ivec_push(data->dendro_offsets, (int64_t)data->dendro_merges->n);

    // Update prev_assignments for next iteration
    tk_ivec_copy(prev_assignments, snapshot_assignments, 0, (int64_t)snapshot_assignments->n, 0);
  }

  double score = 0.0;
  if (data->quality_metric != TK_QUALITY_NONE && n_active_clusters > 1) {
    tm_clustering_accuracy(
      data->L, snapshot_assignments, data->offsets, data->neighbors,
      data->weights, data->n_threads, data->quality_metric, &score);
    tk_dvec_push(data->scores, score);
  }

  // Always track cluster count
  tk_ivec_push(data->n_clusters, (int64_t)n_active_clusters);

  // User callback if provided
  if (data->i_each > -1) {
    lua_pushvalue(data->L, data->i_each);
    lua_newtable(data->L);
    lua_pushnumber(data->L, score);
    lua_setfield(data->L, -2, "score");
    lua_pushinteger(data->L, (lua_Integer) iteration);
    lua_setfield(data->L, -2, "step");
    lua_pushinteger(data->L, (lua_Integer) n_active_clusters);
    lua_setfield(data->L, -2, "n_clusters");
    lua_pushvalue(data->L, data->i_ids);
    lua_setfield(data->L, -2, "ids");
    tk_lua_get_ephemeron(data->L, TK_EVAL_EPH, snapshot_assignments);
    lua_setfield(data->L, -2, "assignments");
    lua_call(data->L, 1, 0);
  }
}

static inline int tm_setup_clustering_ids (
  lua_State *L,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  tk_ivec_t *ids,
  int i_ids,
  tk_ivec_t **out_ids,
  tk_iumap_t **out_ididx,
  int *out_i_eph
) {
  lua_newtable(L);
  int i_eph = tk_lua_absindex(L, -1);
  if (ids != NULL)
    lua_pushvalue(L, i_ids);

  *out_ids = ids != NULL ? ids :
    inv ? tk_iumap_keys(L, inv->uid_sid) :
    hbi ? tk_iumap_keys(L, hbi->uid_sid) :
    ann ? tk_iumap_keys(L, ann->uid_sid) : tk_ivec_create(L, 0, 0, 0);

  *out_ididx = tk_iumap_from_ivec(L, *out_ids);
  if (!*out_ididx)
    tk_error(L, "tm_setup_clustering_ids: iumap_from_ivec failed", ENOMEM);

  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  *out_i_eph = i_eph;
  return 0;
}

typedef struct {
  tk_ivec_t *dendro_offsets;  // [0..n-1]=initial_assignments, [n+i]=merge offset for step i
  tk_pvec_t *dendro_merges;   // merge pairs (absorbed, surviving)
  tk_dvec_t *scores;          // scores->a[step] = quality score
  tk_ivec_t *n_clusters;      // n_clusters->a[step] = cluster count at step
  uint64_t n_steps;
} tm_optimize_result_t;

static inline tm_optimize_result_t tm_optimize_clustering_agglo (
  lua_State *L,
  tk_inv_t *inv,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  tk_ivec_t *ids,
  int i_ids,
  int i_eph,
  tk_ivec_t *offsets,
  tk_ivec_t *neighbors,
  tk_dvec_t *weights,
  unsigned int n_threads,
  uint64_t probe_radius,
  tk_agglo_linkage_t linkage,
  uint64_t knn,
  uint64_t knn_min,
  bool knn_mutual,
  tk_ivec_sim_type_t cmp,
  double cmp_alpha,
  double cmp_beta,
  int64_t rank_filter,
  uint64_t min_pts,
  bool assign_noise,
  tk_quality_metric_t quality,
  double tolerance,
  int i_each
) {
  tm_optimize_result_t result;

  // Initialize dendrogram structures
  // Estimated size: n_samples for initial + estimated steps (assume n_samples/10)
  uint64_t estimated_steps = ids->n / 10 + 10;
  result.dendro_offsets = tk_ivec_create(L, ids->n + estimated_steps, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.dendro_merges = tk_pvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  // Only create scores vector if quality tracking is enabled
  if (quality != TK_QUALITY_NONE) {
    result.scores = tk_dvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
    lua_pop(L, 1);
  } else {
    result.scores = NULL;
  }

  result.n_clusters = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  tk_ivec_t *working_assignments = tk_ivec_create(L, ids->n, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  // Pre-compute neighborhoods for single-linkage mode
  tk_inv_hoods_t *inv_hoods = NULL;
  tk_ann_hoods_t *ann_hoods = NULL;
  tk_hbi_hoods_t *hbi_hoods = NULL;
  tk_ivec_t *hoods_uids = NULL;

  if (linkage == TK_AGGLO_LINKAGE_SINGLE && knn > 0) {
    // Need to pre-compute KNN neighborhoods
    if (inv != NULL) {
      tk_inv_neighborhoods(L, inv, knn, 0.0, 1.0, knn_min, cmp, cmp_alpha, cmp_beta,
                           knn_mutual, rank_filter, n_threads, &inv_hoods, &hoods_uids);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -2);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
      lua_pop(L, 2);
    } else if (ann != NULL) {
      tk_ann_neighborhoods(L, ann, knn, probe_radius, 0, -1, knn_min, knn_mutual, n_threads, &ann_hoods, &hoods_uids);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -2);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
      lua_pop(L, 2);
    } else if (hbi != NULL) {
      tk_hbi_neighborhoods(L, hbi, knn, 0, UINT64_MAX, knn_min, knn_mutual, n_threads, &hbi_hoods, &hoods_uids);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -2);
      tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
      lua_pop(L, 2);
    }
  }

  // Initialize callback data
  agglo_callback_data_t cb_data = {
    .L = L,
    .i_each = i_each,
    .i_ids = i_ids,
    .i_eph = i_eph,
    .offsets = offsets,
    .neighbors = neighbors,
    .weights = weights,
    .n_threads = n_threads,
    .quality_metric = quality,
    .tolerance = tolerance,
    .dendro_offsets = result.dendro_offsets,
    .dendro_merges = result.dendro_merges,
    .scores = result.scores,
    .n_clusters = result.n_clusters,
    .n_samples = ids->n
  };

  uint64_t features = ann != NULL ? ann->features :
                      hbi != NULL ? hbi->features :
                      inv->features;
  tk_agglo_index_type_t index_type = hbi != NULL ? TK_AGGLO_USE_HBI : TK_AGGLO_USE_ANN;

  tk_agglo(L, ann, hbi, ids, features, index_type, linkage, probe_radius, knn, min_pts, assign_noise,
           inv_hoods, ann_hoods, hbi_hoods, hoods_uids, n_threads, working_assignments,
           agglo_snapshot_callback, &cb_data);

  tk_ivec_destroy(working_assignments);

  result.n_steps = result.n_clusters->n;

  return result;
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

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);
  int i_ids = ids == NULL ? -1 : tk_lua_absindex(L, -1);

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "weights");
  tk_dvec_t *weights = tk_dvec_peek(L, -1, "weights");
  lua_pop(L, 1);

  // Only agglo method is supported
  if (inv == NULL && hbi == NULL && ann == NULL)
    tk_lua_verror(L, 3, "optimize_clustering", "index", "agglo method requires inv, hbi, or ann index");

  unsigned int n_threads = tk_threads_getn(L, 1, "optimize_clustering", "threads");

  // Parse linkage mode
  const char *linkage_str = tk_lua_foptstring(L, 1, "optimize_clustering", "linkage", "simhash");
  tk_agglo_linkage_t linkage = TK_AGGLO_LINKAGE_SIMHASH;
  if (!strcmp(linkage_str, "simhash"))
    linkage = TK_AGGLO_LINKAGE_SIMHASH;
  else if (!strcmp(linkage_str, "single"))
    linkage = TK_AGGLO_LINKAGE_SINGLE;

  // Parse single-linkage parameters
  uint64_t knn = tk_lua_foptunsigned(L, 1, "optimize_clustering", "knn", 0);
  uint64_t knn_min = tk_lua_foptunsigned(L, 1, "optimize_clustering", "knn_min", 0);
  bool knn_mutual = tk_lua_foptboolean(L, 1, "optimize_clustering", "knn_mutual", true);
  uint64_t min_pts = tk_lua_foptunsigned(L, 1, "optimize_clustering", "min_pts", 0);
  bool assign_noise = tk_lua_foptboolean(L, 1, "optimize_clustering", "assign_noise", false);

  // Parse inv-specific parameters
  const char *cmpstr = tk_lua_foptstring(L, 1, "optimize_clustering", "cmp", "jaccard");
  double cmp_alpha = tk_lua_foptnumber(L, 1, "optimize_clustering", "cmp_alpha", 1.0);
  double cmp_beta = tk_lua_foptnumber(L, 1, "optimize_clustering", "cmp_beta", 0.1);
  int64_t rank_filter = tk_lua_foptinteger(L, 1, "optimize_clustering", "rank_filter", -1);

  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(cmpstr, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(cmpstr, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(cmpstr, "tversky"))
    cmp = TK_IVEC_TVERSKY;
  else if (!strcmp(cmpstr, "dice"))
    cmp = TK_IVEC_DICE;

  // Validate single-linkage requirements
  if (linkage == TK_AGGLO_LINKAGE_SINGLE && knn == 0)
    tk_lua_verror(L, 3, "optimize_clustering", "knn", "knn parameter required for linkage=single");

  double tolerance = tk_lua_foptnumber(L, 1, "optimize_clustering", "tolerance", 1e-12);
  char *metric_str = tk_lua_foptstring(L, 1, "optimize_clustering", "metric", "correlation");
  tk_quality_metric_t quality = TK_QUALITY_NONE;
  if (!strcmp(metric_str, "correlation") || !strcmp(metric_str, "biserial"))
    quality = TK_QUALITY_BISERIAL;
  else if (!strcmp(metric_str, "variance"))
    quality = TK_QUALITY_VARIANCE;
  else
    tk_lua_verror(L, 3, "optimize_clustering", "metric", "must be 'correlation', 'variance', or 'biserial'");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  // Setup IDs and ephemeron tracking
  tk_ivec_t *state_ids;
  tk_iumap_t *state_ididx;
  int i_eph;
  tm_setup_clustering_ids(L, inv, ann, hbi, ids, i_ids, &state_ids, &state_ididx, &i_eph);

  // Run agglo optimization
  uint64_t probe_radius = tk_lua_foptunsigned(L, 1, "optimize_clustering", "probe_radius", 3);
  tm_optimize_result_t result = tm_optimize_clustering_agglo(L, inv, ann, hbi, state_ids, i_ids, i_eph,
                                        offsets, neighbors, weights, n_threads,
                                        probe_radius, linkage, knn, knn_min, knn_mutual,
                                        cmp, cmp_alpha, cmp_beta, rank_filter,
                                        min_pts, assign_noise,
                                        quality, tolerance, i_each);

  // Cleanup
  tk_lua_del_ephemeron(L, TK_EVAL_EPH, i_eph, state_ididx);
  tk_iumap_destroy(state_ididx);

  // Build result table
  lua_newtable(L);

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.dendro_offsets);
  lua_setfield(L, -2, "offsets");

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.dendro_merges);
  lua_setfield(L, -2, "merges");

  // Only include scores if quality tracking was enabled
  if (result.scores != NULL) {
    tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.scores);
    lua_setfield(L, -2, "scores");
  }

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.n_clusters);
  lua_setfield(L, -2, "n_clusters");

  lua_pushinteger(L, (lua_Integer)result.n_steps);
  lua_setfield(L, -2, "n_steps");

  if (ids != NULL)
    lua_pushvalue(L, i_ids);
  else
    tk_lua_get_ephemeron(L, TK_EVAL_EPH, state_ids);
  lua_setfield(L, -2, "ids");

  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline int tm_optimize_retrieval (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_cvec_t *cvec = tk_cvec_peek(L, -1, "codes");
  tk_bits_t *codes = (tk_bits_t *) cvec->a;
  lua_pop(L, 1);

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "weights");
  tk_dvec_t *weights = tk_dvec_peek(L, -1, "weights");
  lua_pop(L, 1);

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "optimize_retrieval", "n_dims");
  unsigned int n_threads = tk_threads_getn(L, 1, "optimize_retrieval", "threads");
  uint64_t min_margin = tk_lua_foptunsigned(L, 1, "optimize_retrieval", "min_margin", 0);
  uint64_t max_margin = tk_lua_foptunsigned(L, 1, "optimize_retrieval", "max_margin", n_dims);

  char *metric_str = tk_lua_foptstring(L, 1, "optimize_retrieval", "metric", "correlation");
  tk_quality_metric_t quality = TK_QUALITY_NONE;
  if (!strcmp(metric_str, "correlation") || !strcmp(metric_str, "biserial"))
    quality = TK_QUALITY_BISERIAL;
  else if (!strcmp(metric_str, "variance"))
    quality = TK_QUALITY_VARIANCE;
  else
    tk_lua_verror(L, 3, "optimize_retrieval", "metric", "must be 'correlation', 'variance', or 'biserial'");

  if (max_margin > n_dims)
    max_margin = n_dims;
  if (min_margin > max_margin)
    min_margin = 0;

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  uint64_t chunks = TK_CVEC_BITS_BYTES(n_dims);

  // Create vector to store score trajectory
  tk_dvec_t *scores = tk_dvec_create(L, 0, 0, 0);
  int i_scores = tk_lua_absindex(L, -1);

  for (uint64_t m = min_margin; m <= max_margin; m++) {
    double score;
    tm_retrieval_accuracy(L, codes, n_dims, chunks, offsets, neighbors, weights,
                          m, n_threads, quality, &score);

    tk_dvec_push(scores, score);

    // User callback if provided
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_newtable(L);
      lua_pushnumber(L, score);
      lua_setfield(L, -2, "score");
      lua_pushinteger(L, (lua_Integer)m);
      lua_setfield(L, -2, "margin");
      lua_call(L, 1, 0);
    }
  }

  lua_pushvalue(L, i_scores);
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
        if (score > current_score + tolerance) {
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
        if (score >= current_score + tolerance) {
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
          if (score > current_score + tolerance) {
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
  double tolerance = tk_lua_foptnumber(L, 1, "optimize_bits", "tolerance", 1e-12);

  char *metric_str = tk_lua_foptstring(L, 1, "optimize_bits", "metric", "correlation");
  tk_correlation_metric_t metric = TK_CORRELATION_NONE;
  if (!strcmp(metric_str, "correlation") || !strcmp(metric_str, "spearman"))
    metric = TK_CORRELATION_SPEARMAN;
  else if (!strcmp(metric_str, "kendall"))
    metric = TK_CORRELATION_KENDALL;
  else if (!strcmp(metric_str, "variance"))
    metric = TK_CORRELATION_VARIANCE;
  else
    tk_lua_verror(L, 3, "optimize_retrieval", "metric", "must be 'correlation', 'variance', or 'biserial'");

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
  state.correlation_metric = metric;
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

static inline int tk_pvec_dendro_cut_lua (lua_State *L)
{
  lua_settop(L, 4);
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_pvec_t *merges = tk_pvec_peek(L, 2, "merges");
  uint64_t step = tk_lua_checkunsigned(L, 3, "step");
  tk_ivec_t *assignments = tk_ivec_peekopt(L, 4);
  int i_assignments = -1;
  if (assignments == NULL) {
    assignments = tk_ivec_create(L, 0, 0, 0);
    i_assignments = tk_lua_absindex(L, -1);
  } else {
    i_assignments = tk_lua_absindex(L, -1);
  }
  tk_pvec_dendro_cut(L, offsets, merges, step, assignments);
  lua_pushvalue(L, i_assignments);
  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

typedef struct {
  tk_ivec_t *offsets;
  tk_pvec_t *merges;
  tk_ivec_t *ids;
  tk_ivec_t *assignments;
  tk_ivec_t *raw_assignments;
  tk_iumap_t *absorbed_to_surviving;
  uint64_t n_samples;
  uint64_t n_steps;
  uint64_t current_step;
} tk_dendro_iter_t;

static inline int tk_dendro_iter_gc(lua_State *L) {
  tk_dendro_iter_t *iter = luaL_checkudata(L, 1, "tk_dendro_iter_t");
  if (iter->absorbed_to_surviving) {
    tk_iumap_destroy(iter->absorbed_to_surviving);
    iter->absorbed_to_surviving = NULL;
  }
  return 0;
}

static inline int tk_dendro_iter_next(lua_State *L) {
  tk_dendro_iter_t *iter = lua_touserdata(L, lua_upvalueindex(1));
  if (iter->current_step >= iter->n_steps) {
    lua_pushnil(L);
    return 1;
  }
  uint64_t step = iter->current_step;
  if (step == 0) {
    for (uint64_t i = 0; i < iter->n_samples; i++)
      iter->raw_assignments->a[i] = iter->offsets->a[i];
    iter->raw_assignments->n = iter->n_samples;
    for (uint64_t i = 0; i < iter->n_samples; i++)
      iter->ids->a[i] = (int64_t)i;
    iter->ids->n = iter->n_samples;
  } else {
    int64_t start = iter->offsets->a[iter->n_samples + step - 1];
    int64_t end = (step + iter->n_samples < iter->offsets->n) ?
      iter->offsets->a[iter->n_samples + step] : (int64_t)iter->merges->n;
    for (int64_t idx = start; idx < end && idx < (int64_t)iter->merges->n; idx++) {
      tk_pair_t merge = iter->merges->a[idx];
      int64_t absorbed = merge.i;
      int64_t surviving = merge.p;
      int kha;
      khint_t khi = tk_iumap_put(iter->absorbed_to_surviving, absorbed, &kha);
      tk_iumap_setval(iter->absorbed_to_surviving, khi, surviving);
    }
  }
  for (uint64_t i = 0; i < iter->n_samples; i++) {
    int64_t cluster = iter->raw_assignments->a[i];
    uint64_t chain_limit = 10000;
    uint64_t chain_count = 0;
    while (chain_count < chain_limit) {
      khint_t khi = tk_iumap_get(iter->absorbed_to_surviving, cluster);
      if (khi == tk_iumap_end(iter->absorbed_to_surviving))
        break;
      cluster = tk_iumap_val(iter->absorbed_to_surviving, khi);
      chain_count++;
    }
    iter->assignments->a[i] = cluster;
  }
  iter->assignments->n = iter->n_samples;
  tk_iumap_t *cluster_remap = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, -1, -1);
  lua_pop(L, 1);
  int64_t next_id = 0;
  for (uint64_t i = 0; i < iter->n_samples; i++) {
    int64_t cluster = iter->assignments->a[i];
    khint_t khi = tk_iumap_get(cluster_remap, cluster);
    if (khi == tk_iumap_end(cluster_remap)) {
      int kha;
      khi = tk_iumap_put(cluster_remap, cluster, &kha);
      tk_iumap_setval(cluster_remap, khi, next_id);
      iter->assignments->a[i] = next_id;
      next_id++;
    } else {
      iter->assignments->a[i] = tk_iumap_val(cluster_remap, khi);
    }
  }
  tk_iumap_destroy(cluster_remap);
  iter->current_step++;
  lua_pushinteger(L, (lua_Integer)step);
  tk_lua_get_ephemeron(L, TK_EVAL_EPH, iter->ids);
  tk_lua_get_ephemeron(L, TK_EVAL_EPH, iter->assignments);
  return 3;
}

static inline int tk_dendro_iter_lua(lua_State *L) {
  lua_settop(L, 2);
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_pvec_t *merges = tk_pvec_peek(L, 2, "merges");
  uint64_t n_samples = 0;
  for (uint64_t i = 0; i < offsets->n; i++) {
    if (offsets->a[i] == 0 && i > 0) {
      n_samples = i;
      break;
    }
  }
  if (n_samples == 0 || n_samples > offsets->n)
    tk_error(L, "tk_dendro_iter: invalid dendro_offsets structure", EINVAL);
  uint64_t n_steps = offsets->n > n_samples ? offsets->n - n_samples : 0;
  tk_dendro_iter_t *iter = tk_lua_newuserdata(L, tk_dendro_iter_t, TK_EVAL_EPH, NULL, tk_dendro_iter_gc);
  int iter_idx = lua_gettop(L);
  iter->offsets = offsets;
  iter->merges = merges;
  iter->n_samples = n_samples;
  iter->n_steps = n_steps;
  iter->current_step = 0;
  iter->ids = tk_ivec_create(L, n_samples, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, iter_idx, -1);
  lua_pop(L, 1);
  iter->raw_assignments = tk_ivec_create(L, n_samples, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, iter_idx, -1);
  lua_pop(L, 1);
  iter->assignments = tk_ivec_create(L, n_samples, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, iter_idx, -1);
  lua_pop(L, 1);
  iter->absorbed_to_surviving = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, iter_idx, -1);
  lua_pop(L, 1);
  lua_pushvalue(L, iter_idx);
  lua_pushcclosure(L, tk_dendro_iter_next, 1);
  return 1;
}

static luaL_Reg tm_evaluator_fns[] =
{
  { "class_accuracy", tm_class_accuracy },
  { "encoding_accuracy", tm_encoding_accuracy },
  { "clustering_accuracy", tm_clustering_accuracy_lua },
  { "retrieval_accuracy", tm_retrieval_accuracy_lua },
  { "optimize_bits", tm_optimize_bits },
  { "optimize_retrieval", tm_optimize_retrieval },
  { "optimize_clustering", tm_optimize_clustering },
  { "entropy_stats", tm_entropy_stats },
  { "auc", tm_auc },
  { "dendro_cut", tk_pvec_dendro_cut_lua },
  { "dendro_each", tk_dendro_iter_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_evaluator_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_evaluator_fns, 0); // t
  return 1;
}
