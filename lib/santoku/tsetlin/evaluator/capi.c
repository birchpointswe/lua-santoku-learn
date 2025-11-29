#include <santoku/iuset.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/tsetlin/centroid.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/rvec.h>
#include <santoku/pvec.h>
#include <santoku/evec.h>
#include <santoku/euset.h>
#include <santoku/iumap.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define TK_EVAL_EPH "tk_eval_eph"

typedef enum {
  TK_EVAL_METRIC_NONE,
  TK_EVAL_METRIC_PEARSON,
  TK_EVAL_METRIC_SPEARMAN,
  TK_EVAL_METRIC_VARIANCE,
  TK_EVAL_METRIC_MEAN,
  TK_EVAL_METRIC_MIN,
  TK_EVAL_METRIC_NDCG,
} tk_eval_metric_t;

static inline tk_eval_metric_t tk_eval_parse_metric (const char *metric_str) {
  if (!strcmp(metric_str, "pearson"))
    return TK_EVAL_METRIC_PEARSON;
  if (!strcmp(metric_str, "spearman"))
    return TK_EVAL_METRIC_SPEARMAN;
  if (!strcmp(metric_str, "variance"))
    return TK_EVAL_METRIC_VARIANCE;
  if (!strcmp(metric_str, "mean"))
    return TK_EVAL_METRIC_MEAN;
  if (!strcmp(metric_str, "min"))
    return TK_EVAL_METRIC_MIN;
  if (!strcmp(metric_str, "ndcg"))
    return TK_EVAL_METRIC_NDCG;
  return TK_EVAL_METRIC_NONE;
}

typedef enum {
  TK_EVAL_ELBOW_NONE,
  TK_EVAL_ELBOW_LMETHOD,
  TK_EVAL_ELBOW_MAX_GAP,
  TK_EVAL_ELBOW_MAX_DROP,
  TK_EVAL_ELBOW_PLATEAU,
  TK_EVAL_ELBOW_KNEEDLE,
  TK_EVAL_ELBOW_MAX_CURVATURE,
  TK_EVAL_ELBOW_MAX_ACCELERATION,
  TK_EVAL_ELBOW_OTSU,
  TK_EVAL_ELBOW_FIRST_GAP,
} tk_eval_elbow_t;

static inline tk_eval_elbow_t tk_eval_parse_elbow (const char *elbow_str) {
  if (!strcmp(elbow_str, "lmethod"))
    return TK_EVAL_ELBOW_LMETHOD;
  if (!strcmp(elbow_str, "max_gap"))
    return TK_EVAL_ELBOW_MAX_GAP;
  if (!strcmp(elbow_str, "max_drop"))
    return TK_EVAL_ELBOW_MAX_DROP;
  if (!strcmp(elbow_str, "plateau"))
    return TK_EVAL_ELBOW_PLATEAU;
  if (!strcmp(elbow_str, "kneedle"))
    return TK_EVAL_ELBOW_KNEEDLE;
  if (!strcmp(elbow_str, "max_curvature"))
    return TK_EVAL_ELBOW_MAX_CURVATURE;
  if (!strcmp(elbow_str, "max_acceleration"))
    return TK_EVAL_ELBOW_MAX_ACCELERATION;
  if (!strcmp(elbow_str, "otsu"))
    return TK_EVAL_ELBOW_OTSU;
  if (!strcmp(elbow_str, "first_gap"))
    return TK_EVAL_ELBOW_FIRST_GAP;
  return TK_EVAL_ELBOW_NONE;
}

typedef struct {
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
  char *codes, *codes_predicted, *codes_expected, *mask;
  uint64_t mask_popcount;
  double *dcodes;
  tk_iumap_t *id_assignment;
  tk_ivec_t *adjacency_ids;
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
  tk_eval_metric_t eval_metric;
  uint64_t margin;
} tk_eval_t;

static inline int tm_class_accuracy (lua_State *L)
{
  lua_settop(L, 5);
  tk_ivec_t *predicted = tk_ivec_peek(L, 1, "predicted");
  tk_ivec_t *expected = tk_ivec_peek(L, 2, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int n_dims = tk_lua_checkunsigned(L, 4, "n_classes");
  if (n_dims == 0)
    tk_lua_verror(L, 3, "class_accuracy", "n_classes", "must be > 0");

  atomic_ulong *TP = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  atomic_ulong *FP = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  atomic_ulong *FN = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  double *precision = tk_malloc(L, n_dims * sizeof(double));
  double *recall = tk_malloc(L, n_dims * sizeof(double));
  double *f1 = tk_malloc(L, n_dims * sizeof(double));

  for (uint64_t i = 0; i < n_dims; i ++) {
    atomic_init(TP + i, 0);
    atomic_init(FP + i, 0);
    atomic_init(FN + i, 0);
  }

  #pragma omp parallel for schedule(static)
  for (unsigned int i = 0; i < n_samples; i ++) {
    unsigned int y_pred = predicted->a[i];
    unsigned int y_true = expected->a[i];
    if (y_pred >= n_dims || y_true >= n_dims)
      continue;
    if (y_pred == y_true)
      atomic_fetch_add(TP + y_true, 1);
    else {
      atomic_fetch_add(FP + y_pred, 1);
      atomic_fetch_add(FN + y_true, 1);
    }
  }

  double precision_avg = 0.0, recall_avg = 0.0, f1_avg = 0.0;
  for (unsigned int c = 0; c < n_dims; c ++) {
    uint64_t tp = TP[c], fp = FP[c], fn = FN[c];
    precision[c] = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
    recall[c] = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
    f1[c] = (precision[c] + recall[c]) > 0 ?
      2.0 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0.0;
    precision_avg += precision[c];
    recall_avg += recall[c];
    f1_avg += f1[c];
  }

  free(TP);
  free(FP);
  free(FN);

  precision_avg /= n_dims;
  recall_avg /= n_dims;
  f1_avg /= n_dims;
  lua_newtable(L);
  lua_newtable(L);
  for (uint64_t c = 0; c < n_dims; c ++) {
    lua_pushinteger(L, (int64_t) c + 1);
    lua_newtable(L);
    lua_pushnumber(L, precision[c]);
    lua_setfield(L, -2, "precision");
    lua_pushnumber(L, recall[c]);
    lua_setfield(L, -2, "recall");
    lua_pushnumber(L, f1[c]);
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

  free(precision);
  free(recall);
  free(f1);
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
  tk_eval_metric_t metric,
  double *out_score
) {

  if (!offsets || offsets->n == 0) {
    *out_score = 0.0;
    return;
  }

  uint64_t n_nodes = offsets->n - 1;
  if (n_nodes == 0) {
    *out_score = 0.0;
    return;
  }

  double total_corr_score = 0.0;
  uint64_t total_nodes_processed = 0;

  #pragma omp parallel reduction(+:total_corr_score) reduction(+:total_nodes_processed)
  {
    tk_iuset_t *itmp = tk_iuset_create(NULL, 0);
    tk_dumap_t *rank_buffer_b = tk_dumap_create(NULL, 0);

    if (!itmp || !rank_buffer_b) {
      if (itmp) tk_iuset_destroy(itmp);
      if (rank_buffer_b) tk_dumap_destroy(rank_buffer_b);
    } else {
      #pragma omp for schedule(static)
      for (uint64_t node_idx = 0; node_idx < n_nodes; node_idx++) {
        int64_t start = offsets->a[node_idx];
        int64_t end = offsets->a[node_idx + 1];
        uint64_t n_neighbors = (uint64_t)(end - start);
        if (n_neighbors == 0)
          continue;
        int64_t my_cluster = assignments->a[node_idx];
        tk_iuset_clear(itmp);
        for (int64_t idx = start; idx < end; idx++) {
          int64_t neighbor = neighbors->a[idx];
          int64_t neighbor_cluster = assignments->a[neighbor];
          if (my_cluster == neighbor_cluster) {
            int kha;
            tk_iuset_put(itmp, neighbor, &kha);
          }
        }

        if (tk_iuset_size(itmp) > 0) {
          double node_score = 0.0;
          switch (metric) {
          case TK_EVAL_METRIC_MEAN:
            node_score = tk_csr_mean(neighbors, weights, start, end, itmp);
            break;
          case TK_EVAL_METRIC_MIN:
            node_score = tk_csr_min(neighbors, weights, start, end, itmp);
            break;
          case TK_EVAL_METRIC_VARIANCE:
            node_score = tk_csr_variance_ratio(neighbors, weights, start, end, itmp, rank_buffer_b);
            break;
          default:
            node_score = 0.0;
            break;
          }
          total_corr_score += node_score;
          total_nodes_processed++;
        }
      }

      tk_iuset_destroy(itmp);
      tk_dumap_destroy(rank_buffer_b);
    }
  }

  *out_score = total_nodes_processed > 0 ? total_corr_score / total_nodes_processed : 0.0;
}

static inline int tm_clustering_accuracy_lua (lua_State *L)
{
  lua_settop(L, 1);

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

  char *metric_str = tk_lua_foptstring(L, 1, "clustering_accuracy", "metric", "mean");
  tk_eval_metric_t metric = tk_eval_parse_metric(metric_str);
  if (metric == TK_EVAL_METRIC_NONE)
    tk_lua_verror(L, 3, "clustering_accuracy", "metric", "unknown metric");

  double score;
  tm_clustering_accuracy(L, assignments, offsets, neighbors, weights, metric, &score);
  lua_pushnumber(L, score);
  lua_replace(L, 1);
  lua_settop(L, 1);
  return 1;
}

static inline void tm_retrieval_accuracy (
  lua_State *L,
  char *codes,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  tk_ivec_t *adjacency_ids,
  uint64_t n_dims,
  tk_ivec_t *offsets,
  tk_ivec_t *neighbors,
  tk_dvec_t *weights,
  uint64_t margin,
  tk_eval_metric_t metric,
  double *out_score
) {
  if (!offsets || offsets->n == 0) {
    *out_score = 0.0;
    return;
  }

  uint64_t n_nodes = offsets->n - 1;
  if (n_nodes == 0) {
    *out_score = 0.0;
    return;
  }

  uint64_t chunks = TK_CVEC_BITS_BYTES(n_dims);
  double total_quality_score = 0.0;
  uint64_t total_nodes_processed = 0;

  #pragma omp parallel reduction(+:total_quality_score) reduction(+:total_nodes_processed)
  {
    tk_iuset_t *itmp = tk_iuset_create(NULL, 0);
    tk_dumap_t *rank_buffer_b = tk_dumap_create(NULL, 0);

    if (!itmp || !rank_buffer_b) {
      if (itmp) tk_iuset_destroy(itmp);
      if (rank_buffer_b) tk_dumap_destroy(rank_buffer_b);
    } else {
      #pragma omp for schedule(static)
      for (uint64_t node_idx = 0; node_idx < n_nodes; node_idx++) {
        int64_t start = offsets->a[node_idx];
        int64_t end = offsets->a[node_idx + 1];
        uint64_t n_neighbors = (uint64_t)(end - start);
        if (n_neighbors == 0)
          continue;

        char *node_code = NULL;
        if (codes) {
          node_code = (char *)(codes + node_idx * chunks);
        } else if (adjacency_ids) {
          int64_t node_id = adjacency_ids->a[node_idx];
          node_code = ann ? tk_ann_get(ann, node_id)
                                 : tk_hbi_get(hbi, node_id);
        }
        if (!node_code)
          continue;

        tk_iuset_clear(itmp);

        for (int64_t idx = start; idx < end; idx++) {
          int64_t neighbor_pos = neighbors->a[idx];

          char *neighbor_code = NULL;
          if (codes) {
            neighbor_code = (char *)(codes + (uint64_t) neighbor_pos * chunks);
          } else if (adjacency_ids) {
            int64_t neighbor_id = adjacency_ids->a[neighbor_pos];
            neighbor_code = ann ? tk_ann_get(ann, neighbor_id)
                                       : tk_hbi_get(hbi, neighbor_id);
          }
          if (!neighbor_code)
            continue;

          uint64_t hamming_dist = tk_cvec_bits_hamming_serial(
            (const unsigned char*)node_code,
            (const unsigned char*)neighbor_code,
            n_dims);

          if (hamming_dist <= margin) {
            int kha;
            tk_iuset_put(itmp, neighbor_pos, &kha);
          }
        }

        uint64_t neighbors_within = tk_iuset_size(itmp);

        if (neighbors_within > 0) {
          double node_score = 0.0;
          switch (metric) {
            case TK_EVAL_METRIC_PEARSON:
              node_score = tk_csr_pearson(neighbors, weights, start, end, itmp);
              break;
            case TK_EVAL_METRIC_MEAN:
              node_score = tk_csr_mean(neighbors, weights, start, end, itmp);
              break;
            case TK_EVAL_METRIC_MIN:
              node_score = tk_csr_min(neighbors, weights, start, end, itmp);
              break;
            case TK_EVAL_METRIC_VARIANCE:
              node_score = tk_csr_variance_ratio(neighbors, weights, start, end, itmp, rank_buffer_b);
              break;
            default:
              node_score = 0.0;
              break;
          }
          total_quality_score += node_score;
          total_nodes_processed++;
        }
      }

      tk_iuset_destroy(itmp);
      tk_dumap_destroy(rank_buffer_b);
    }
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
  tk_cvec_t *cvec = tk_cvec_peekopt(L, -1);
  char *codes = cvec ? cvec->a : NULL;
  lua_pop(L, 1);

  lua_getfield(L, 1, "index");
  tk_ann_t *ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *hbi = tk_hbi_peekopt(L, -1);
  lua_pop(L, 1);

  if (!codes && !ann && !hbi)
    tk_lua_verror(L, 3, "retrieval_accuracy", "codes or index", "must provide either codes or index (tk_ann_t/tk_hbi_t)");

  lua_getfield(L, 1, "ids");
  tk_ivec_t *adjacency_ids = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);

  if (!codes && !adjacency_ids)
    tk_lua_verror(L, 3, "retrieval_accuracy", "ids", "required when using index");

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "weights");
  tk_dvec_t *weights = tk_dvec_peek(L, -1, "weights");
  lua_pop(L, 1);

  uint64_t n_dims = tk_lua_foptunsigned(L, 1, "retrieval_accuracy", "n_dims", 0);
  if (n_dims == 0) {
    if (ann)
      n_dims = ann->features;
    else if (hbi)
      n_dims = hbi->features;
    else if (codes)
      tk_lua_verror(L, 3, "retrieval_accuracy", "n_dims", "required when using codes without index");
  }

  uint64_t margin = tk_lua_fcheckunsigned(L, 1, "retrieval_accuracy", "margin");

  char *metric_str = tk_lua_foptstring(L, 1, "retrieval_accuracy", "metric", "mean");
  tk_eval_metric_t metric = tk_eval_parse_metric(metric_str);
  if (metric == TK_EVAL_METRIC_NONE)
    tk_lua_verror(L, 3, "retrieval_accuracy", "metric", "unknown metric");

  double score;
  tm_retrieval_accuracy(L, codes, ann, hbi, adjacency_ids, n_dims, offsets, neighbors, weights,
                        margin, metric, &score);

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
  char *codes_predicted, *codes_expected;
  tk_cvec_t *pvec = tk_cvec_peekopt(L, 1);
  tk_cvec_t *evec = tk_cvec_peekopt(L, 2);
  codes_predicted = pvec != NULL ? pvec->a : (char *)tk_lua_checkustring(L, 1, "predicted");
  codes_expected = evec != NULL ? evec->a : (char *)tk_lua_checkustring(L, 2, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int n_dims = tk_lua_checkunsigned(L, 4, "n_hidden");
  uint64_t chunks = TK_CVEC_BITS_BYTES(n_dims);

  uint64_t diff_total = 0;
  uint64_t *bdiff_total = lua_newuserdata(L, n_dims * sizeof(uint64_t));
  memset(bdiff_total, 0, n_dims * sizeof(uint64_t));

  #pragma omp parallel reduction(+:diff_total)
  {
    uint64_t *bdiff_local = calloc(n_dims, sizeof(uint64_t));

    #pragma omp for schedule(static)
    for (uint64_t i = 0; i < n_samples; i ++) {
      for (uint64_t j = 0; j < n_dims; j ++) {
        uint64_t word = TK_CVEC_BITS_BYTE(j);
        uint64_t bit = TK_CVEC_BITS_BIT(j);
        bool y =
          (((uint8_t *)codes_expected)[i * chunks + word] & (1 << bit)) ==
          (((uint8_t *)codes_predicted)[i * chunks + word] & (1 << bit));
        if (y)
          continue;
        diff_total ++;
        bdiff_local[j] ++;
      }
    }

    #pragma omp critical
    {
      for (uint64_t j = 0; j < n_dims; j ++)
        bdiff_total[j] += bdiff_local[j];
    }

    free(bdiff_local);
  }

  lua_newtable(L);
  lua_newtable(L);
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

  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline tk_ivec_t *tk_pvec_dendro_cut(
  lua_State *L,
  tk_ivec_t *offsets,
  tk_pvec_t *merges,
  uint64_t step,
  tk_ivec_t *assignments
) {
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

  if (assignments->m < n_samples) {
    tk_ivec_ensure(assignments, n_samples);
  }

  for (uint64_t i = 0; i < n_samples; i++) {
    assignments->a[i] = offsets->a[i];
  }
  assignments->n = n_samples;

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

  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t cluster = assignments->a[i];
    uint64_t chain_limit = 10000;
    uint64_t chain_count = 0;

    while (chain_count < chain_limit) {
      khint_t khi = tk_iumap_get(absorbed_to_surviving, cluster);
      if (khi == tk_iumap_end(absorbed_to_surviving)) {
        break;
      }
      cluster = tk_iumap_val(absorbed_to_surviving, khi);
      chain_count++;
    }

    assignments->a[i] = cluster;
  }

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

typedef struct {
  tk_ivec_t *dendro_offsets;
  tk_pvec_t *dendro_merges;
  uint64_t n_steps;
} tm_cluster_result_t;

typedef struct {
  int64_t cluster_id;
  tk_ivec_t *members;
  tk_centroid_t *centroid;
  bool active;
  tk_iuset_t *neighbor_ids;
  int64_t next_in_hash_chain;
} tk_cluster_t;

static inline uint64_t tk_cluster_complete_linkage_distance(
  tk_cvec_t *codes,
  uint64_t n_chunks,
  uint64_t n_bits,
  tk_cluster_t *cluster_i,
  tk_cluster_t *cluster_j,
  char *centroid_i,
  char *centroid_j,
  tk_pumap_t *distance_cache,
  uint64_t early_exit_threshold
) {
  int64_t cache_key;
  if (cluster_i->cluster_id < cluster_j->cluster_id) {
    cache_key = ((int64_t)cluster_i->cluster_id << 32) | cluster_j->cluster_id;
  } else {
    cache_key = ((int64_t)cluster_j->cluster_id << 32) | cluster_i->cluster_id;
  }

  if (distance_cache) {
    khint_t khi = tk_pumap_get(distance_cache, cache_key);
    if (khi != tk_pumap_end(distance_cache)) {
      return (uint64_t)tk_pumap_val(distance_cache, khi).p;
    }
  }

  uint64_t centroid_dist = tk_cvec_bits_hamming_serial(
    (const uint8_t*)centroid_i,
    (const uint8_t*)centroid_j,
    n_bits
  );

  if (early_exit_threshold > 0 && centroid_dist >= early_exit_threshold) {
    return centroid_dist;
  }

  uint64_t max_dist = centroid_dist;
  uint64_t total_pairs = cluster_i->members->n * cluster_j->members->n;

  if (total_pairs > 100) {
    #pragma omp parallel for reduction(max:max_dist) schedule(static)
    for (uint64_t mi = 0; mi < cluster_i->members->n; mi++) {
      int64_t member_i = cluster_i->members->a[mi];
      char *code_i = codes->a + (uint64_t)member_i * n_chunks;

      for (uint64_t mj = 0; mj < cluster_j->members->n; mj++) {
        int64_t member_j = cluster_j->members->a[mj];
        char *code_j = codes->a + (uint64_t)member_j * n_chunks;

        uint64_t dist = tk_cvec_bits_hamming_serial(
          (const uint8_t*)code_i,
          (const uint8_t*)code_j,
          n_bits
        );

        if (dist > max_dist) {
          max_dist = dist;
        }
      }
    }
  } else {
    for (uint64_t mi = 0; mi < cluster_i->members->n; mi++) {
      int64_t member_i = cluster_i->members->a[mi];
      char *code_i = codes->a + (uint64_t)member_i * n_chunks;

      for (uint64_t mj = 0; mj < cluster_j->members->n; mj++) {
        int64_t member_j = cluster_j->members->a[mj];
        char *code_j = codes->a + (uint64_t)member_j * n_chunks;

        uint64_t dist = tk_cvec_bits_hamming_serial(
          (const uint8_t*)code_i,
          (const uint8_t*)code_j,
          n_bits
        );

        if (dist > max_dist) {
          max_dist = dist;
        }
      }
    }
  }

  if (distance_cache) {
    int kha;
    khint_t khi = tk_pumap_put(distance_cache, cache_key, &kha);
    tk_pumap_setval(distance_cache, khi, tk_pair(0, (double)max_dist));
  }

  return max_dist;
}

static inline int tk_cluster_centroid(
  lua_State *L,
  tk_cvec_t *codes,
  tk_ivec_t *adj_ids,
  tk_ivec_t *adj_offsets,
  tk_ivec_t *adj_neighbors,
  uint64_t n_bits,
  tk_ivec_t *dendro_offsets,
  tk_pvec_t *dendro_merges
) {
  if ((dendro_offsets && !dendro_merges) || (!dendro_offsets && dendro_merges)) {
    return -1;
  }
  uint64_t n_nodes = adj_ids->n;

  if (n_nodes == 0) {
    return -1;
  }

  if (n_bits == 0) {
    return -1;
  }

  uint64_t n_chunks = codes->n / n_nodes;
  uint8_t tail_mask = (n_bits % 8 == 0) ? 0xFF : ((1 << (n_bits % 8)) - 1);

  uint64_t expected_code_size = n_nodes * n_chunks;
  if (codes->n != expected_code_size) {
    return -1;
  }

  if (adj_offsets->n < n_nodes + 1) {
    return -1;
  }

  tk_cluster_t **clusters = malloc(n_nodes * sizeof(tk_cluster_t*));
  if (!clusters) return -1;

  uint64_t n_clusters = 0;
  uint64_t n_active = 0;

  tk_ivec_t *entity_to_cluster = tk_ivec_create(NULL, n_nodes, 0, 0);
  if (!entity_to_cluster) {
    free(clusters);
    return -1;
  }
  entity_to_cluster->n = n_nodes;

  tk_evec_t *edge_heap = tk_evec_create(NULL, 0, 0, 0);
  if (!edge_heap) {
    tk_ivec_destroy(entity_to_cluster);
    free(clusters);
    return -1;
  }

  tk_iumap_t *code_to_cluster = tk_iumap_create(NULL, 0);
  if (!code_to_cluster) {
    tk_evec_destroy(edge_heap);
    tk_ivec_destroy(entity_to_cluster);
    free(clusters);
    return -1;
  }

  for (uint64_t i = 0; i < n_nodes; i++) {
    char *code = codes->a + i * n_chunks;

    uint64_t hash = 0xcbf29ce484222325ULL;
    for (uint64_t b = 0; b < n_chunks; b++) {
      hash ^= (uint64_t)(uint8_t)code[b];
      hash *= 0x100000001b3ULL;
    }

    int kha;
    khint_t khi = tk_iumap_put(code_to_cluster, (int64_t) hash, &kha);

    int64_t cluster_idx = -1;
    bool need_new_cluster = false;

    if (kha) {
      need_new_cluster = true;
    } else {
      int64_t chain_idx = tk_iumap_val(code_to_cluster, khi);
      int64_t matching_idx = -1;

      while (chain_idx != -1) {
        tk_cluster_t *existing = clusters[chain_idx];

        if (existing->members->n > 0) {
          char *existing_code = codes->a + (uint64_t)existing->members->a[0] * n_chunks;

          bool match = true;
          if (n_chunks > 1) {
            match = (memcmp(code, existing_code, n_chunks - 1) == 0);
          }
          if (match && n_chunks > 0) {
            uint8_t masked_new = ((uint8_t*)code)[n_chunks - 1] & tail_mask;
            uint8_t masked_existing = ((uint8_t*)existing_code)[n_chunks - 1] & tail_mask;
            match = (masked_new == masked_existing);
          }

          if (match) {
            matching_idx = chain_idx;
            break;
          }
        }

        chain_idx = existing->next_in_hash_chain;
      }

      if (matching_idx != -1) {
        cluster_idx = matching_idx;
      } else {
        need_new_cluster = true;
      }
    }

    if (need_new_cluster) {
      tk_cluster_t *cluster = malloc(sizeof(tk_cluster_t));
      if (!cluster) {
        for (uint64_t c = 0; c < n_clusters; c++) {
          if (clusters[c]) {
            if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
            if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
            if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
            free(clusters[c]);
          }
        }
        tk_iumap_destroy(code_to_cluster);
        tk_evec_destroy(edge_heap);
        tk_ivec_destroy(entity_to_cluster);
        free(clusters);
        return -1;
      }

      cluster->cluster_id = (int64_t)(2 * n_nodes + 1 + n_clusters);
      cluster->members = tk_ivec_create(NULL, 0, 0, 0);
      cluster->centroid = tk_centroid_create(NULL, n_chunks, tail_mask);
      cluster->active = true;
      cluster->neighbor_ids = tk_iuset_create(NULL, 0);

      if (!cluster->members || !cluster->centroid || !cluster->neighbor_ids) {
        if (cluster->members) tk_ivec_destroy(cluster->members);
        if (cluster->centroid) tk_centroid_destroy(cluster->centroid);
        if (cluster->neighbor_ids) tk_iuset_destroy(cluster->neighbor_ids);
        free(cluster);

        for (uint64_t c = 0; c < n_clusters; c++) {
          if (clusters[c]) {
            if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
            if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
            if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
            free(clusters[c]);
          }
        }
        tk_iumap_destroy(code_to_cluster);
        tk_evec_destroy(edge_heap);
        tk_ivec_destroy(entity_to_cluster);
        free(clusters);
        return -1;
      }

      if (kha) {
        cluster->next_in_hash_chain = -1;
        tk_iumap_setval(code_to_cluster, khi, (int64_t)n_clusters);
      } else {
        int64_t old_head = tk_iumap_val(code_to_cluster, khi);
        cluster->next_in_hash_chain = old_head;
        tk_iumap_setval(code_to_cluster, khi, (int64_t)n_clusters);
      }

      clusters[n_clusters] = cluster;
      cluster_idx = (int64_t)n_clusters;
      n_clusters++;
    }

    tk_cluster_t *cluster = clusters[cluster_idx];

    if (cluster->members->m < cluster->members->n + 1) {
      if (tk_ivec_ensure(cluster->members, cluster->members->n + 1) != 0) {
        for (uint64_t c = 0; c < n_clusters; c++) {
          if (clusters[c]) {
            if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
            if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
            if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
            free(clusters[c]);
          }
        }
        tk_iumap_destroy(code_to_cluster);
        tk_evec_destroy(edge_heap);
        tk_ivec_destroy(entity_to_cluster);
        free(clusters);
        return -1;
      }
    }

    cluster->members->a[cluster->members->n++] = (int64_t)i;
    entity_to_cluster->a[i] = cluster_idx;
    tk_centroid_add_member(cluster->centroid, code, n_chunks);
  }

  tk_iumap_destroy(code_to_cluster);
  n_active = n_clusters;

  tk_pumap_t *distance_cache = tk_pumap_create(NULL, 0);

  tk_euset_t *seen_edges = tk_euset_create(NULL, 0);
  if (!seen_edges || !distance_cache) {
    if (distance_cache) tk_pumap_destroy(distance_cache);
    if (seen_edges) tk_euset_destroy(seen_edges);
    for (uint64_t c = 0; c < n_clusters; c++) {
      if (clusters[c]) {
        if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
        if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
        if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
        free(clusters[c]);
      }
    }
    tk_evec_destroy(edge_heap);
    tk_ivec_destroy(entity_to_cluster);
    free(clusters);
    return -1;
  }

  for (uint64_t i = 0; i < n_nodes; i++) {
    int64_t cluster_i_idx = entity_to_cluster->a[i];

    int64_t start = adj_offsets->a[i];
    int64_t end = adj_offsets->a[i + 1];

    for (int64_t j = start; j < end; j++) {
      int64_t neighbor_idx = adj_neighbors->a[j];

      if (neighbor_idx < 0 || neighbor_idx >= (int64_t)n_nodes) {
        tk_euset_destroy(seen_edges);
        for (uint64_t c = 0; c < n_clusters; c++) {
          if (clusters[c]) {
            if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
            if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
            if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
            free(clusters[c]);
          }
        }
        tk_evec_destroy(edge_heap);
        tk_ivec_destroy(entity_to_cluster);
        free(clusters);
        return -1;
      }

      int64_t cluster_j_idx = entity_to_cluster->a[neighbor_idx];

      if (cluster_i_idx == cluster_j_idx)
        continue;

      tk_edge_t edge = tk_edge(cluster_i_idx, cluster_j_idx, 0.0);

      int edge_kha;
      tk_euset_put(seen_edges, edge, &edge_kha);
      if (!edge_kha)
        continue;

      int neighbor_kha;
      tk_iuset_put(clusters[edge.u]->neighbor_ids, edge.v, &neighbor_kha);
      tk_iuset_put(clusters[edge.v]->neighbor_ids, edge.u, &neighbor_kha);

      char *code_u = tk_centroid_code(clusters[edge.u]->centroid);
      char *code_v = tk_centroid_code(clusters[edge.v]->centroid);

      uint64_t complete_dist = tk_cluster_complete_linkage_distance(
        codes, n_chunks, n_bits,
        clusters[edge.u], clusters[edge.v],
        code_u, code_v,
        distance_cache,
        0
      );

      double dist = (double)complete_dist;
      tk_evec_push(edge_heap, tk_edge(edge.u, edge.v, dist));
    }
  }

  tk_euset_destroy(seen_edges);
  tk_evec_hmin_init(edge_heap);

  if (dendro_offsets) {
    if (tk_ivec_ensure(dendro_offsets, n_nodes + 1) != 0) {
      for (uint64_t c = 0; c < n_clusters; c++) {
        if (clusters[c]) {
          if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
          if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
          if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
          free(clusters[c]);
        }
      }
      tk_evec_destroy(edge_heap);
      tk_ivec_destroy(entity_to_cluster);
      free(clusters);
      return -1;
    }

    for (uint64_t i = 0; i < n_nodes; i++) {
      int64_t cluster_idx = entity_to_cluster->a[i];
      dendro_offsets->a[i] = clusters[cluster_idx]->cluster_id;
    }
    dendro_offsets->a[n_nodes] = 0;
    dendro_offsets->n = n_nodes + 1;
  }

  while (edge_heap->n > 0 && n_active > 1) {
    tk_edge_t min_edge = tk_evec_hmin_pop(edge_heap);
    double min_dist = min_edge.w;

    tk_evec_t *distance_level_edges = tk_evec_create(NULL, 0, 0, 0);
    tk_evec_push(distance_level_edges, min_edge);

    while (edge_heap->n > 0) {
      if (edge_heap->a[0].w > min_dist) break;
      tk_edge_t edge = tk_evec_hmin_pop(edge_heap);
      tk_evec_push(distance_level_edges, edge);
    }
    for (uint64_t i = 0; i < distance_level_edges->n; i++) {
      tk_edge_t *e = &distance_level_edges->a[i];
      uint64_t deg_u = clusters[e->u]->active ? tk_iuset_size(clusters[e->u]->neighbor_ids) : 0;
      uint64_t deg_v = clusters[e->v]->active ? tk_iuset_size(clusters[e->v]->neighbor_ids) : 0;
      uint64_t min_deg = deg_u < deg_v ? deg_u : deg_v;
      e->w = (double)min_deg;
    }
    tk_evec_asc(distance_level_edges, 0, distance_level_edges->n);


    uint64_t batch_start = 0;
    while (batch_start < distance_level_edges->n) {
      double current_min_degree = distance_level_edges->a[batch_start].w;


      uint64_t batch_end = batch_start + 1;
      while (batch_end < distance_level_edges->n &&
             distance_level_edges->a[batch_end].w == current_min_degree) {
        batch_end++;
      }


      tk_iuset_t *merged_this_batch = tk_iuset_create(NULL, 0);
      uint64_t merges_in_batch = 0;


      for (uint64_t edge_idx = batch_start; edge_idx < batch_end; edge_idx++) {
        tk_edge_t edge = distance_level_edges->a[edge_idx];
        int64_t ci_idx = edge.u;
        int64_t cj_idx = edge.v;

        tk_cluster_t *ci = clusters[ci_idx];
        tk_cluster_t *cj = clusters[cj_idx];

        if (!ci->active || !cj->active)
          continue;


        khint_t khi_u = tk_iuset_get(merged_this_batch, ci_idx);
        khint_t khi_v = tk_iuset_get(merged_this_batch, cj_idx);
        if (khi_u != tk_iuset_end(merged_this_batch) ||
            khi_v != tk_iuset_end(merged_this_batch)) {
          continue;
        }


        int kha;
        tk_iuset_put(merged_this_batch, ci_idx, &kha);
        tk_iuset_put(merged_this_batch, cj_idx, &kha);
        merges_in_batch++;

        if (ci->centroid->size > cj->centroid->size) {
          tk_cluster_t *tmp = ci;
          ci = cj;
          cj = tmp;
          int64_t tmp_idx = ci_idx;
          ci_idx = cj_idx;
          cj_idx = tmp_idx;
        }

        if (dendro_merges) {
      if (dendro_merges->m < dendro_merges->n + 1) {
        if (tk_pvec_ensure(dendro_merges, dendro_merges->n + 1) != 0) {
          for (uint64_t c = 0; c < n_clusters; c++) {
            if (clusters[c]) {
              if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
              if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
              if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
              free(clusters[c]);
            }
          }
          tk_evec_destroy(edge_heap);
          tk_ivec_destroy(entity_to_cluster);
          free(clusters);
          return -1;
        }
      }
      dendro_merges->a[dendro_merges->n++] = tk_pair(ci->cluster_id, cj->cluster_id);
    }

    uint64_t total_members = cj->members->n + ci->members->n;
    if (cj->members->m < total_members) {
      if (tk_ivec_ensure(cj->members, total_members) != 0) {
        for (uint64_t c = 0; c < n_clusters; c++) {
          if (clusters[c]) {
            if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
            if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
            if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
            free(clusters[c]);
          }
        }
        tk_evec_destroy(edge_heap);
        tk_ivec_destroy(entity_to_cluster);
        free(clusters);
        return -1;
      }
    }

    for (uint64_t m = 0; m < ci->members->n; m++) {
      int64_t member_idx = ci->members->a[m];
      cj->members->a[cj->members->n++] = member_idx;
      entity_to_cluster->a[member_idx] = cj_idx;
    }

    tk_centroid_merge(cj->centroid, ci->centroid);

    for (khint_t k = tk_iuset_begin(ci->neighbor_ids);
         k != tk_iuset_end(ci->neighbor_ids); ++k) {
      if (!tk_iuset_exist(ci->neighbor_ids, k))
        continue;

      int64_t neighbor_idx = tk_iuset_key(ci->neighbor_ids, k);

      if (neighbor_idx == cj_idx || !clusters[neighbor_idx]->active)
        continue;

      int neighbor_kha;
      tk_iuset_put(cj->neighbor_ids, neighbor_idx, &neighbor_kha);
    }


    khint_t ci_in_cj = tk_iuset_get(cj->neighbor_ids, ci_idx);
    if (ci_in_cj != tk_iuset_end(cj->neighbor_ids)) {
      tk_iuset_del(cj->neighbor_ids, ci_in_cj);
    }


    for (khint_t k = tk_iuset_begin(ci->neighbor_ids);
         k != tk_iuset_end(ci->neighbor_ids); ++k) {
      if (!tk_iuset_exist(ci->neighbor_ids, k))
        continue;

      int64_t neighbor_idx = tk_iuset_key(ci->neighbor_ids, k);

      if (neighbor_idx == cj_idx || !clusters[neighbor_idx]->active)
        continue;

      tk_cluster_t *neighbor = clusters[neighbor_idx];


      khint_t ci_in_neighbor = tk_iuset_get(neighbor->neighbor_ids, ci_idx);
      if (ci_in_neighbor != tk_iuset_end(neighbor->neighbor_ids)) {
        tk_iuset_del(neighbor->neighbor_ids, ci_in_neighbor);
      }


      int neighbor_kha;
      tk_iuset_put(neighbor->neighbor_ids, cj_idx, &neighbor_kha);
    }

    char *cj_code = tk_centroid_code(cj->centroid);


    uint64_t heap_min = (edge_heap->n > 0) ? (uint64_t)edge_heap->a[0].w : 0;

    for (khint_t k = tk_iuset_begin(cj->neighbor_ids);
         k != tk_iuset_end(cj->neighbor_ids); ++k) {
      if (!tk_iuset_exist(cj->neighbor_ids, k))
        continue;

      int64_t neighbor_idx = tk_iuset_key(cj->neighbor_ids, k);

      if (!clusters[neighbor_idx]->active)
        continue;

      tk_cluster_t *neighbor = clusters[neighbor_idx];
      char *neighbor_code = tk_centroid_code(neighbor->centroid);

      uint64_t complete_dist = tk_cluster_complete_linkage_distance(
        codes, n_chunks, n_bits,
        cj, neighbor,
        cj_code, neighbor_code,
        distance_cache,
        heap_min
      );

      double new_dist = (double)complete_dist;

      if (edge_heap->m < edge_heap->n + 1) {
        if (tk_evec_ensure(edge_heap, edge_heap->n + 100) != 0) {
          for (uint64_t c = 0; c < n_clusters; c++) {
            if (clusters[c]) {
              if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
              if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
              if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
              free(clusters[c]);
            }
          }
          tk_evec_destroy(edge_heap);
          tk_ivec_destroy(entity_to_cluster);
          free(clusters);
          return -1;
        }
      }

      edge_heap->a[edge_heap->n] = tk_edge(cj_idx, neighbor_idx, new_dist);
      edge_heap->n++;
      size_t idx = edge_heap->n - 1;
      while (idx > 0) {
        size_t parent = (idx - 1) >> 1;
        if (edge_heap->a[idx].w >= edge_heap->a[parent].w) break;
        tk_edge_t tmp = edge_heap->a[idx];
        edge_heap->a[idx] = edge_heap->a[parent];
        edge_heap->a[parent] = tmp;
        idx = parent;
      }
    }

        ci->active = false;
        n_active--;
      }


      tk_iuset_destroy(merged_this_batch);


      if (merges_in_batch > 0 && dendro_offsets && dendro_merges) {
        if (dendro_offsets->m < dendro_offsets->n + 1) {
          if (tk_ivec_ensure(dendro_offsets, dendro_offsets->n + 100) != 0) {
            tk_evec_destroy(distance_level_edges);
            for (uint64_t c = 0; c < n_clusters; c++) {
              if (clusters[c]) {
                if (clusters[c]->members) tk_ivec_destroy(clusters[c]->members);
                if (clusters[c]->centroid) tk_centroid_destroy(clusters[c]->centroid);
                if (clusters[c]->neighbor_ids) tk_iuset_destroy(clusters[c]->neighbor_ids);
                free(clusters[c]);
              }
            }
            tk_evec_destroy(edge_heap);
            tk_ivec_destroy(entity_to_cluster);
            free(clusters);
            return -1;
          }
        }
        dendro_offsets->a[dendro_offsets->n++] = (int64_t)dendro_merges->n;
      }

      batch_start = batch_end;
    }

    tk_evec_destroy(distance_level_edges);
  }

  for (uint64_t i = 0; i < n_clusters; i++) {
    tk_cluster_t *cluster = clusters[i];
    if (cluster) {
      if (cluster->members) tk_ivec_destroy(cluster->members);
      if (cluster->centroid) tk_centroid_destroy(cluster->centroid);
      if (cluster->neighbor_ids) tk_iuset_destroy(cluster->neighbor_ids);
      free(cluster);
    }
  }
  free(clusters);
  tk_ivec_destroy(entity_to_cluster);
  tk_evec_destroy(edge_heap);
  if (distance_cache) tk_pumap_destroy(distance_cache);

  return 0;
}

static inline tm_cluster_result_t tm_cluster_agglo (
  lua_State *L,
  tk_cvec_t *codes,
  tk_ivec_t *adj_ids,
  tk_ivec_t *adj_offsets,
  tk_ivec_t *adj_neighbors,
  uint64_t n_bits,
  int i_eph
) {
  tm_cluster_result_t result;

  result.dendro_offsets = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.dendro_merges = tk_pvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  tk_cluster_centroid(L, codes, adj_ids, adj_offsets, adj_neighbors,
                     n_bits, result.dendro_offsets, result.dendro_merges);

  result.n_steps = result.dendro_offsets->n > adj_ids->n + 1 ?
                   result.dendro_offsets->n - adj_ids->n - 1 : 0;

  return result;
}

static inline size_t tk_eval_apply_elbow_dvec (
  tk_dvec_t *v,
  tk_eval_elbow_t elbow,
  double alpha,
  double *out_val
);

static inline int tm_score_clustering (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "ids");
  tk_ivec_t *dendro_ids = tk_ivec_peek(L, -1, "ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *dendro_offsets = tk_ivec_peek(L, -1, "offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "merges");
  tk_pvec_t *dendro_merges = tk_pvec_peek(L, -1, "merges");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_ids");
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    lua_getfield(L, 1, "eval_ids");
  }
  tk_ivec_t *eval_ids = tk_ivec_peek(L, -1, "expected_ids or eval_ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_offsets");
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    lua_getfield(L, 1, "eval_offsets");
  }
  tk_ivec_t *eval_offsets = tk_ivec_peek(L, -1, "expected_offsets or eval_offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_neighbors");
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    lua_getfield(L, 1, "eval_neighbors");
  }
  tk_ivec_t *eval_neighbors = tk_ivec_peek(L, -1, "expected_neighbors or eval_neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_weights");
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    lua_getfield(L, 1, "eval_weights");
  }
  tk_dvec_t *eval_weights = tk_dvec_peek(L, -1, "expected_weights or eval_weights");
  lua_pop(L, 1);

  const char *metric_str = tk_lua_foptstring(L, 1, "score_clustering", "metric", "min");
  tk_eval_metric_t metric = tk_eval_parse_metric(metric_str);
  if (metric != TK_EVAL_METRIC_MIN &&
      metric != TK_EVAL_METRIC_MEAN &&
      metric != TK_EVAL_METRIC_VARIANCE)
    tk_lua_verror(L, 1, "score_clustering", "metric", "must be min, mean, or variance");

  const char *elbow_str = tk_lua_fcheckstring(L, 1, "score_clustering", "elbow");
  tk_eval_elbow_t elbow = tk_eval_parse_elbow(elbow_str);
  if (elbow == TK_EVAL_ELBOW_NONE)
    tk_lua_verror(L, 1, "score_clustering", "elbow", "unknown elbow method");

  const char *elbow_target_str = tk_lua_foptstring(L, 1, "score_clustering", "elbow_target", "f1");
  bool elbow_on_f1 = strcmp(elbow_target_str, "quality") != 0;

  double elbow_alpha = tk_lua_foptnumber(L, 1, "score_clustering", "elbow_alpha", 1e-4);

  bool all_merges = tk_lua_foptboolean(L, 1, "score_clustering", "all_merges", false);

  uint64_t n_samples = 0;
  for (uint64_t i = 0; i < dendro_offsets->n; i++) {
    if (dendro_offsets->a[i] == 0 && i > 0) {
      n_samples = i;
      break;
    }
  }
  if (n_samples == 0 || n_samples > dendro_offsets->n)
    tk_lua_verror(L, 2, "score_clustering", "invalid dendro_offsets structure");

  uint64_t n_steps = all_merges ? dendro_merges->n : (dendro_offsets->n > n_samples + 1 ?
                     dendro_offsets->n - n_samples - 1 : 0);

  tk_iumap_t *eval_id_to_idx = tk_iumap_from_ivec(NULL, eval_ids);
  if (!eval_id_to_idx)
    tk_error(L, "score_clustering: failed to create eval_id mapping", ENOMEM);

  int64_t *dendro_to_eval = tk_malloc(L, n_samples * sizeof(int64_t));
  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t dendro_id = dendro_ids->a[i];
    khint_t khi = tk_iumap_get(eval_id_to_idx, dendro_id);
    if (khi != tk_iumap_end(eval_id_to_idx)) {
      dendro_to_eval[i] = tk_iumap_val(eval_id_to_idx, khi);
    } else {
      dendro_to_eval[i] = -1;
    }
  }

  tk_dvec_t *scores = tk_dvec_create(L, 0, 0, 0);
  int i_scores = tk_lua_absindex(L, -1);
  tk_dvec_t *recall_scores = tk_dvec_create(L, 0, 0, 0);
  int i_recall_scores = tk_lua_absindex(L, -1);
  tk_dvec_t *f1_scores = tk_dvec_create(L, 0, 0, 0);
  int i_f1_scores = tk_lua_absindex(L, -1);
  tk_ivec_t *n_clusters_per_step = tk_ivec_create(L, 0, 0, 0);
  int i_n_clusters_per_step = tk_lua_absindex(L, -1);

  tk_ivec_t *working_assignments = tk_ivec_create(NULL, n_samples, 0, 0);
  tk_ivec_t *eval_ordered_assignments = tk_ivec_create(NULL, eval_ids->n, 0, 0);
  eval_ordered_assignments->n = eval_ids->n;
  for (uint64_t i = 0; i < eval_ids->n; i++) {
    eval_ordered_assignments->a[i] = -1;
  }

  tk_iumap_t *cluster_sizes = NULL;
  tk_iumap_t *absorbed_to_surviving = NULL;
  int64_t current_merge_idx = 0;

  if (all_merges) {
    cluster_sizes = tk_iumap_create(NULL, 0);
    absorbed_to_surviving = tk_iumap_create(NULL, 0);
  }

  tk_iumap_t *cluster_set = tk_iumap_create(NULL, 0);

  for (uint64_t step = 0; step <= n_steps; step++) {
    if (all_merges) {
      if (current_merge_idx >= (int64_t)dendro_merges->n)
        break;
      if (step == 0) {
        for (uint64_t i = 0; i < n_samples; i++) {
          int64_t cluster_id = dendro_offsets->a[i];
          working_assignments->a[i] = cluster_id;
          int kha;
          khint_t khi = tk_iumap_put(cluster_sizes, cluster_id, &kha);
          tk_iumap_setval(cluster_sizes, khi, 1);
        }
        working_assignments->n = n_samples;
      } else {
        int64_t start = current_merge_idx;
        if (start >= (int64_t)dendro_merges->n)
          break;

        int64_t distance_end = (int64_t)dendro_merges->n;
        for (uint64_t d = 0; d < dendro_offsets->n - n_samples; d++) {
          int64_t dist_start = dendro_offsets->a[n_samples + d];
          int64_t dist_next = (n_samples + d + 1 < dendro_offsets->n) ?
                              dendro_offsets->a[n_samples + d + 1] : (int64_t)dendro_merges->n;
          if (start >= dist_start && start < dist_next) {
            distance_end = dist_next;
            break;
          }
        }

        tk_pair_t first_merge = dendro_merges->a[start];
        khint_t khi_i = tk_iumap_get(cluster_sizes, first_merge.i);
        khint_t khi_p = tk_iumap_get(cluster_sizes, first_merge.p);
        int64_t size_i = (khi_i != tk_iumap_end(cluster_sizes)) ? tk_iumap_val(cluster_sizes, khi_i) : 1;
        int64_t size_p = (khi_p != tk_iumap_end(cluster_sizes)) ? tk_iumap_val(cluster_sizes, khi_p) : 1;
        int64_t target_size = size_i + size_p;
        int64_t end = start + 1;
        while (end < distance_end && end < (int64_t)dendro_merges->n) {
          tk_pair_t next_merge = dendro_merges->a[end];
          khi_i = tk_iumap_get(cluster_sizes, next_merge.i);
          khi_p = tk_iumap_get(cluster_sizes, next_merge.p);
          int64_t next_i = (khi_i != tk_iumap_end(cluster_sizes)) ? tk_iumap_val(cluster_sizes, khi_i) : 1;
          int64_t next_p = (khi_p != tk_iumap_end(cluster_sizes)) ? tk_iumap_val(cluster_sizes, khi_p) : 1;
          int64_t next_size = next_i + next_p;
          if (next_size != target_size)
            break;
          end++;
        }
        for (int64_t idx = start; idx < end; idx++) {
          tk_pair_t merge = dendro_merges->a[idx];
          int64_t absorbed = merge.i;
          int64_t surviving = merge.p;
          khi_i = tk_iumap_get(cluster_sizes, absorbed);
          khi_p = tk_iumap_get(cluster_sizes, surviving);
          size_i = (khi_i != tk_iumap_end(cluster_sizes)) ? tk_iumap_val(cluster_sizes, khi_i) : 1;
          size_p = (khi_p != tk_iumap_end(cluster_sizes)) ? tk_iumap_val(cluster_sizes, khi_p) : 1;
          int64_t new_size = size_i + size_p;
          int kha;
          khint_t khi = tk_iumap_put(cluster_sizes, surviving, &kha);
          tk_iumap_setval(cluster_sizes, khi, new_size);
          khi = tk_iumap_put(absorbed_to_surviving, absorbed, &kha);
          tk_iumap_setval(absorbed_to_surviving, khi, surviving);
        }
        current_merge_idx = end;
      }
      for (uint64_t i = 0; i < n_samples; i++) {
        int64_t cluster = working_assignments->a[i];
        uint64_t chain_limit = 10000;
        uint64_t chain_count = 0;
        while (chain_count < chain_limit) {
          khint_t khi = tk_iumap_get(absorbed_to_surviving, cluster);
          if (khi == tk_iumap_end(absorbed_to_surviving))
            break;
          cluster = tk_iumap_val(absorbed_to_surviving, khi);
          chain_count++;
        }
        working_assignments->a[i] = cluster;
      }
    } else {
      tk_pvec_dendro_cut(L, dendro_offsets, dendro_merges, step, working_assignments);
    }

    for (uint64_t i = 0; i < eval_ids->n; i++) {
      eval_ordered_assignments->a[i] = -1;
    }

    for (uint64_t i = 0; i < n_samples; i++) {
      int64_t eval_idx = dendro_to_eval[i];
      if (eval_idx >= 0) {
        eval_ordered_assignments->a[eval_idx] = working_assignments->a[i];
      }
    }


    tk_iumap_clear(cluster_set);
    for (uint64_t i = 0; i < n_samples; i++) {
      if (working_assignments->a[i] >= 0) {
        int kha;
        tk_iumap_put(cluster_set, working_assignments->a[i], &kha);
      }
    }
    uint64_t n_active_clusters = tk_iumap_size(cluster_set);

    double score = 0.0;
    double recall = 0.0;
    if (metric != TK_EVAL_METRIC_NONE && n_active_clusters >= 1) {
      tm_clustering_accuracy(L, eval_ordered_assignments, eval_offsets, eval_neighbors,
                            eval_weights, metric, &score);

      double recall_numerator = 0.0;
      double recall_denominator = 0.0;
      for (uint64_t i = 0; i < eval_offsets->n - 1; i++) {
        int64_t start = eval_offsets->a[i];
        int64_t end = eval_offsets->a[i + 1];
        int64_t cluster_i = eval_ordered_assignments->a[i];
        if (cluster_i < 0)
          continue;
        for (int64_t j = start; j < end; j++) {
          int64_t neighbor_idx = eval_neighbors->a[j];
          double weight = eval_weights->a[j];
          recall_denominator += weight;
          if (neighbor_idx >= 0 && neighbor_idx < (int64_t)eval_ordered_assignments->n) {
            int64_t cluster_j = eval_ordered_assignments->a[neighbor_idx];
            if (cluster_i == cluster_j && cluster_j >= 0) {
              recall_numerator += weight;
            }
          }
        }
      }
      recall = recall_denominator > 0.0 ? recall_numerator / recall_denominator : 0.0;
    }

    double f1 = (score > 0.0 && recall > 0.0) ?
      (2.0 * score * recall) / (score + recall) : 0.0;
    tk_dvec_push(scores, score);
    tk_dvec_push(recall_scores, recall);
    tk_dvec_push(f1_scores, f1);
    tk_ivec_push(n_clusters_per_step, (int64_t)n_active_clusters);
  }

  uint64_t actual_steps = scores->n > 0 ? scores->n - 1 : 0;

  tk_ivec_destroy(working_assignments);
  tk_ivec_destroy(eval_ordered_assignments);
  free(dendro_to_eval);
  tk_iumap_destroy(eval_id_to_idx);
  tk_iumap_destroy(cluster_set);
  if (cluster_sizes)
    tk_iumap_destroy(cluster_sizes);
  if (absorbed_to_surviving)
    tk_iumap_destroy(absorbed_to_surviving);

  tk_dvec_t *target_curve = elbow_on_f1 ? f1_scores : scores;
  double elbow_val;
  size_t best_step = tk_eval_apply_elbow_dvec(target_curve, elbow, elbow_alpha, &elbow_val);

  double best_quality = (best_step < scores->n) ? scores->a[best_step] : 0.0;
  double best_recall = (best_step < recall_scores->n) ? recall_scores->a[best_step] : 0.0;
  double best_f1 = (best_step < f1_scores->n) ? f1_scores->a[best_step] : 0.0;
  int64_t best_n_clusters = (best_step < n_clusters_per_step->n) ? n_clusters_per_step->a[best_step] : 0;

  lua_newtable(L);

  lua_pushnumber(L, best_quality);
  lua_setfield(L, -2, "quality");
  lua_pushnumber(L, best_recall);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, best_f1);
  lua_setfield(L, -2, "f1");
  lua_pushinteger(L, (lua_Integer)best_n_clusters);
  lua_setfield(L, -2, "n_clusters");
  lua_pushinteger(L, (lua_Integer)best_step);
  lua_setfield(L, -2, "best_step");

  lua_pushvalue(L, i_scores);
  lua_setfield(L, -2, "quality_curve");
  lua_pushvalue(L, i_recall_scores);
  lua_setfield(L, -2, "recall_curve");
  lua_pushvalue(L, i_f1_scores);
  lua_setfield(L, -2, "f1_curve");
  lua_pushvalue(L, i_n_clusters_per_step);
  lua_setfield(L, -2, "n_clusters_curve");
  lua_pushinteger(L, (lua_Integer)actual_steps);
  lua_setfield(L, -2, "n_steps");

  return 1;
}

static inline int tm_cluster (lua_State *L)
{
  lua_settop(L, 1);
  lua_newtable(L);
  int i_eph = tk_lua_absindex(L, -1);

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);
  if (!lua_isnil(L, -1))
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "offsets");
  tk_ivec_t *offsets = tk_ivec_peekopt(L, -1);
  if (!lua_isnil(L, -1))
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "neighbors");
  tk_ivec_t *neighbors = tk_ivec_peekopt(L, -1);
  if (!lua_isnil(L, -1))
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_cvec_t *codes = tk_cvec_peekopt(L, -1);
  if (!codes)
    tk_lua_verror(L, 3, "cluster", "codes", "required (binary codes)");
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  if (!ids)
    tk_lua_verror(L, 3, "cluster", "ids", "required");
  if (!offsets)
    tk_lua_verror(L, 3, "cluster", "offsets", "required");
  if (!neighbors)
    tk_lua_verror(L, 3, "cluster", "neighbors", "required");

  uint64_t n_bits = tk_lua_fcheckunsigned(L, 1, "cluster", "n_dims");

  tm_cluster_result_t result = tm_cluster_agglo(
    L, codes, ids, offsets, neighbors, n_bits, i_eph);

  lua_newtable(L);

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.dendro_offsets);
  lua_setfield(L, -2, "offsets");

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.dendro_merges);
  lua_setfield(L, -2, "merges");

  lua_pushinteger(L, (lua_Integer)result.n_steps);
  lua_setfield(L, -2, "n_steps");

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, ids);
  lua_setfield(L, -2, "ids");

  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static inline size_t tk_eval_apply_elbow_pvec (
  tk_pvec_t *v,
  tk_eval_elbow_t elbow,
  double alpha,
  int64_t *out_val
) {
  int64_t int_tolerance = (int64_t)alpha;
  switch (elbow) {
    case TK_EVAL_ELBOW_LMETHOD:
      return tk_pvec_scores_lmethod(v, out_val);
    case TK_EVAL_ELBOW_MAX_GAP:
      return tk_pvec_scores_max_gap(v, out_val);
    case TK_EVAL_ELBOW_MAX_DROP:
      return tk_pvec_scores_max_drop(v, out_val);
    case TK_EVAL_ELBOW_PLATEAU:
      return tk_pvec_scores_plateau(v, int_tolerance, out_val);
    case TK_EVAL_ELBOW_KNEEDLE:
      return tk_pvec_scores_kneedle(v, alpha, out_val);
    case TK_EVAL_ELBOW_MAX_CURVATURE:
      return tk_pvec_scores_max_curvature(v, out_val);
    case TK_EVAL_ELBOW_MAX_ACCELERATION:
      return tk_pvec_scores_max_acceleration(v, out_val);
    case TK_EVAL_ELBOW_OTSU:
      return tk_pvec_scores_otsu(v, out_val);
    case TK_EVAL_ELBOW_FIRST_GAP:
      return tk_pvec_scores_first_gap(v, (int64_t)alpha, out_val);
    default:
      if (out_val) *out_val = (v->n > 0) ? v->a[v->n - 1].p : 0;
      return v->n > 0 ? v->n - 1 : 0;
  }
}

static inline size_t tk_eval_apply_elbow_dvec (
  tk_dvec_t *v,
  tk_eval_elbow_t elbow,
  double alpha,
  double *out_val
) {
  switch (elbow) {
    case TK_EVAL_ELBOW_LMETHOD:
      return tk_dvec_scores_lmethod(v->a, v->n, out_val);
    case TK_EVAL_ELBOW_MAX_GAP:
      return tk_dvec_scores_max_gap(v->a, v->n, out_val);
    case TK_EVAL_ELBOW_MAX_DROP:
      return tk_dvec_scores_max_drop(v->a, v->n, out_val);
    case TK_EVAL_ELBOW_PLATEAU:
      return tk_dvec_scores_plateau(v->a, v->n, alpha, out_val);
    case TK_EVAL_ELBOW_KNEEDLE:
      return tk_dvec_scores_kneedle(v->a, v->n, alpha, out_val);
    case TK_EVAL_ELBOW_MAX_CURVATURE:
      return tk_dvec_scores_max_curvature(v->a, v->n, out_val);
    case TK_EVAL_ELBOW_MAX_ACCELERATION:
      return tk_dvec_scores_max_acceleration(v->a, v->n, out_val);
    case TK_EVAL_ELBOW_OTSU:
      return tk_dvec_scores_otsu(v->a, v->n, out_val);
    case TK_EVAL_ELBOW_FIRST_GAP:
      return tk_dvec_scores_first_gap(v->a, v->n, alpha, out_val);
    default:
      if (out_val) *out_val = (v->n > 0) ? v->a[v->n - 1] : 0.0;
      return v->n > 0 ? v->n - 1 : 0;
  }
}

static inline int tm_score_retrieval (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "retrieved_ids");
  tk_ivec_t *retrieved_ids = tk_ivec_peek(L, -1, "retrieved_ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "retrieved_offsets");
  tk_ivec_t *retrieved_offsets = tk_ivec_peek(L, -1, "retrieved_offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "retrieved_neighbors");
  tk_ivec_t *retrieved_neighbors = tk_ivec_peek(L, -1, "retrieved_neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "retrieved_weights");
  tk_dvec_t *retrieved_weights = tk_dvec_peek(L, -1, "retrieved_weights");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_ids");
  tk_ivec_t *expected_ids = tk_ivec_peek(L, -1, "expected_ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *expected_offsets = tk_ivec_peek(L, -1, "expected_offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *expected_neighbors = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_weights");
  tk_dvec_t *expected_weights = tk_dvec_peek(L, -1, "expected_weights");
  lua_pop(L, 1);

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "score_retrieval", "n_dims");

  const char *ranking_str = tk_lua_fcheckstring(L, 1, "score_retrieval", "ranking");
  tk_eval_metric_t ranking = tk_eval_parse_metric(ranking_str);
  if (ranking != TK_EVAL_METRIC_NDCG &&
      ranking != TK_EVAL_METRIC_SPEARMAN &&
      ranking != TK_EVAL_METRIC_PEARSON)
    tk_lua_verror(L, 1, "score_retrieval", "ranking", "must be ndcg, spearman, or pearson");

  const char *metric_str = tk_lua_foptstring(L, 1, "score_retrieval", "metric", "min");
  tk_eval_metric_t metric = tk_eval_parse_metric(metric_str);
  if (metric != TK_EVAL_METRIC_MIN &&
      metric != TK_EVAL_METRIC_MEAN &&
      metric != TK_EVAL_METRIC_VARIANCE)
    tk_lua_verror(L, 1, "score_retrieval", "metric", "must be min, mean, or variance");

  const char *elbow_str = tk_lua_fcheckstring(L, 1, "score_retrieval", "elbow");
  tk_eval_elbow_t elbow = tk_eval_parse_elbow(elbow_str);
  if (elbow == TK_EVAL_ELBOW_NONE)
    tk_lua_verror(L, 1, "score_retrieval", "elbow", "unknown elbow method");

  double elbow_alpha = tk_lua_foptnumber(L, 1, "score_retrieval", "elbow_alpha", 1e-4);

  tk_iumap_t *expected_id_to_idx = tk_iumap_from_ivec(NULL, expected_ids);
  if (!expected_id_to_idx)
    tk_error(L, "score_retrieval: failed to create expected ID mapping", ENOMEM);

  double total_score = 0.0;
  double total_quality = 0.0;
  double total_recall = 0.0;
  uint64_t n_queries = 0;

  #pragma omp parallel reduction(+:total_score) reduction(+:total_quality) reduction(+:total_recall) reduction(+:n_queries)
  {
    tk_pvec_t *query_neighbors = tk_pvec_create(NULL, 0, 0, 0);
    tk_iuset_t *cutoff_ids = tk_iuset_create(NULL, 0);
    tk_iuset_t *expected_neighbor_set = tk_iuset_create(NULL, 0);
    tk_dumap_t *weight_map = tk_dumap_create(NULL, 0);
    tk_pvec_t *weight_ranks_buffer = tk_pvec_create(NULL, 0, 0, 0);
    tk_dumap_t *weight_rank_map = tk_dumap_create(NULL, 0);

    if (!query_neighbors || !cutoff_ids || !expected_neighbor_set ||
        !weight_map || !weight_ranks_buffer || !weight_rank_map) {
      if (query_neighbors) tk_pvec_destroy(query_neighbors);
      if (cutoff_ids) tk_iuset_destroy(cutoff_ids);
      if (expected_neighbor_set) tk_iuset_destroy(expected_neighbor_set);
      if (weight_map) tk_dumap_destroy(weight_map);
      if (weight_ranks_buffer) tk_pvec_destroy(weight_ranks_buffer);
      if (weight_rank_map) tk_dumap_destroy(weight_rank_map);
    } else {
      #pragma omp for schedule(static)
      for (uint64_t query_idx = 0; query_idx < retrieved_offsets->n - 1; query_idx++) {
        int64_t query_id = retrieved_ids->a[query_idx];

        khint_t exp_khi = tk_iumap_get(expected_id_to_idx, query_id);
        if (exp_khi == tk_iumap_end(expected_id_to_idx))
          continue;
        int64_t expected_query_idx = tk_iumap_val(expected_id_to_idx, exp_khi);

        int64_t exp_start = expected_offsets->a[expected_query_idx];
        int64_t exp_end = expected_offsets->a[expected_query_idx + 1];
        if (exp_end == exp_start)
          continue;

        tk_pvec_clear(query_neighbors);
        int64_t ret_start = retrieved_offsets->a[query_idx];
        int64_t ret_end = retrieved_offsets->a[query_idx + 1];

        for (int64_t i = ret_start; i < ret_end; i++) {
          double hamming_sim = retrieved_weights->a[i];
          int64_t hamming_dist = (int64_t)round((1.0 - hamming_sim) * (double)n_dims);
          int64_t neighbor_idx = retrieved_neighbors->a[i];
          tk_pvec_push(query_neighbors, tk_pair(neighbor_idx, hamming_dist));
        }

        if (query_neighbors->n == 0)
          continue;

        tk_pvec_asc(query_neighbors, 0, query_neighbors->n);

        int64_t cutoff_dist;
        size_t cutoff_idx = tk_eval_apply_elbow_pvec(query_neighbors, elbow, elbow_alpha, &cutoff_dist);

        tk_iuset_clear(cutoff_ids);
        for (size_t i = 0; i <= cutoff_idx && i < query_neighbors->n; i++) {
          int64_t neighbor_idx = query_neighbors->a[i].i;
          int64_t neighbor_id = retrieved_ids->a[neighbor_idx];
          int kha;
          tk_iuset_put(cutoff_ids, neighbor_id, &kha);
        }

        double ranking_score = 0.0;
        switch (ranking) {
          case TK_EVAL_METRIC_NDCG:
            ranking_score = tk_csr_ndcg_distance(expected_neighbors, expected_weights,
              exp_start, exp_end, query_neighbors, weight_map);
            break;
          case TK_EVAL_METRIC_SPEARMAN:
            ranking_score = tk_csr_spearman_distance(expected_neighbors, expected_weights,
              exp_start, exp_end, query_neighbors, weight_ranks_buffer, weight_rank_map);
            break;
          case TK_EVAL_METRIC_PEARSON:
            ranking_score = tk_csr_pearson_distance(expected_neighbors, expected_weights,
              exp_start, exp_end, query_neighbors, weight_map);
            break;
          default:
            break;
        }

        tk_iuset_clear(expected_neighbor_set);
        for (int64_t i = exp_start; i < exp_end; i++) {
          int64_t neighbor_idx = expected_neighbors->a[i];
          int64_t neighbor_id = expected_ids->a[neighbor_idx];
          if (tk_iuset_get(cutoff_ids, neighbor_id) != tk_iuset_end(cutoff_ids)) {
            int kha;
            tk_iuset_put(expected_neighbor_set, neighbor_idx, &kha);
          }
        }

        double query_quality = 0.0;
        if (tk_iuset_size(expected_neighbor_set) > 0) {
          switch (metric) {
            case TK_EVAL_METRIC_MIN:
              query_quality = tk_csr_min(expected_neighbors, expected_weights,
                exp_start, exp_end, expected_neighbor_set);
              break;
            case TK_EVAL_METRIC_MEAN:
              query_quality = tk_csr_mean(expected_neighbors, expected_weights,
                exp_start, exp_end, expected_neighbor_set);
              break;
            case TK_EVAL_METRIC_VARIANCE:
              query_quality = tk_csr_variance_ratio(expected_neighbors, expected_weights,
                exp_start, exp_end, expected_neighbor_set, weight_map);
              break;
            default:
              break;
          }
        }

        double query_recall_num = 0.0;
        double query_recall_denom = 0.0;
        for (int64_t i = exp_start; i < exp_end; i++) {
          double weight = expected_weights->a[i];
          query_recall_denom += weight;
          int64_t neighbor_idx = expected_neighbors->a[i];
          if (tk_iuset_get(expected_neighbor_set, neighbor_idx) != tk_iuset_end(expected_neighbor_set)) {
            query_recall_num += weight;
          }
        }
        double query_recall = query_recall_denom > 0.0 ? query_recall_num / query_recall_denom : 0.0;

        total_score += ranking_score;
        total_quality += query_quality;
        total_recall += query_recall;
        n_queries++;
      }

      tk_pvec_destroy(query_neighbors);
      tk_iuset_destroy(cutoff_ids);
      tk_iuset_destroy(expected_neighbor_set);
      tk_dumap_destroy(weight_map);
      tk_pvec_destroy(weight_ranks_buffer);
      tk_dumap_destroy(weight_rank_map);
    }
  }

  tk_iumap_destroy(expected_id_to_idx);

  double avg_score = n_queries > 0 ? total_score / (double)n_queries : 0.0;
  double avg_quality = n_queries > 0 ? total_quality / (double)n_queries : 0.0;
  double avg_recall = n_queries > 0 ? total_recall / (double)n_queries : 0.0;
  double f1 = (avg_quality > 0.0 && avg_recall > 0.0) ?
    (2.0 * avg_quality * avg_recall) / (avg_quality + avg_recall) : 0.0;

  lua_newtable(L);
  lua_pushnumber(L, avg_score);
  lua_setfield(L, -2, "score");
  lua_pushnumber(L, avg_quality);
  lua_setfield(L, -2, "quality");
  lua_pushnumber(L, avg_recall);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, f1);
  lua_setfield(L, -2, "f1");

  lua_replace(L, 1);
  lua_settop(L, 1);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 1;
}

static double tk_compute_reconstruction (
  char *codes,
  tk_ann_t *ann,
  tk_hbi_t *hbi,
  tk_ivec_t *adjacency_ids,
  uint64_t n_dims,
  tk_ivec_t *offsets,
  tk_ivec_t *neighbors,
  tk_dvec_t *weights,
  tk_eval_metric_t eval_metric,
  char *mask,
  uint64_t mask_popcount
) {
  if (mask_popcount == 0)
    return -INFINITY;

  uint64_t chunks = TK_CVEC_BITS_BYTES(n_dims);
  uint64_t n_nodes = offsets->n - 1;
  double total = 0.0;
  uint64_t total_nodes_processed = 0;

  #pragma omp parallel reduction(+:total) reduction(+:total_nodes_processed)
  {
    tk_pvec_t *bin_ranks = tk_pvec_create(NULL, 0, 0, 0);
    tk_pvec_t *weight_ranks_buffer = tk_pvec_create(NULL, 0, 0, 0);
    tk_dumap_t *rank_buffer_b = tk_dumap_create(NULL, 0);
    tk_dumap_t *weight_rank_map = tk_dumap_create(NULL, 0);
    uint64_t max_hamming = n_dims;
    tk_ivec_t *count_buffer = tk_ivec_create(NULL, max_hamming + 1, 0, 0);
    tk_dvec_t *avgrank_buffer = tk_dvec_create(NULL, max_hamming + 1, 0, 0);
    tk_iuset_t *itmp = tk_iuset_create(NULL, 0);

    if (!bin_ranks || !weight_ranks_buffer || !rank_buffer_b || !weight_rank_map || !count_buffer || !avgrank_buffer || !itmp) {
      if (bin_ranks) tk_pvec_destroy(bin_ranks);
      if (weight_ranks_buffer) tk_pvec_destroy(weight_ranks_buffer);
      if (rank_buffer_b) tk_dumap_destroy(rank_buffer_b);
      if (weight_rank_map) tk_dumap_destroy(weight_rank_map);
      if (count_buffer) tk_ivec_destroy(count_buffer);
      if (avgrank_buffer) tk_dvec_destroy(avgrank_buffer);
      if (itmp) tk_iuset_destroy(itmp);
    } else {
      #pragma omp for schedule(static)
      for (uint64_t node_idx = 0; node_idx < n_nodes; node_idx++) {
        tk_pvec_clear(bin_ranks);
        int64_t start = offsets->a[node_idx];
        int64_t end = offsets->a[node_idx + 1];

        char *node_code = NULL;
        if (codes) {
          node_code = (char *)(codes + node_idx * chunks);
        } else if (adjacency_ids) {
          int64_t node_id = adjacency_ids->a[node_idx];
          node_code = ann ? tk_ann_get(ann, node_id)
                                 : tk_hbi_get(hbi, node_id);
        }
        if (!node_code)
          continue;

        for (int64_t j = start; j < end; j ++) {
          int64_t neighbor_pos = neighbors->a[j];

          char *neighbor_code = NULL;
          if (codes) {
            neighbor_code = (char *)(codes + (uint64_t) neighbor_pos * chunks);
          } else if (adjacency_ids) {
            int64_t neighbor_id = adjacency_ids->a[neighbor_pos];
            neighbor_code = ann ? tk_ann_get(ann, neighbor_id)
                                       : tk_hbi_get(hbi, neighbor_id);
          }
          if (!neighbor_code)
            continue;

          uint64_t hamming_dist = mask
            ? tk_cvec_bits_hamming_mask_serial(
                (const unsigned char*)node_code,
                (const unsigned char*)neighbor_code,
                (const unsigned char*)mask,
                n_dims)
            : tk_cvec_bits_hamming_serial(
                (const unsigned char*)node_code,
                (const unsigned char*)neighbor_code,
                n_dims);

          tk_pvec_push(bin_ranks, tk_pair(neighbor_pos, (int64_t) hamming_dist));
        }

        double corr;
        switch (eval_metric) {
          case TK_EVAL_METRIC_PEARSON:
            corr = tk_csr_pearson_distance(neighbors, weights, start, end, bin_ranks, rank_buffer_b);
            break;
          case TK_EVAL_METRIC_SPEARMAN:
            tk_pvec_desc(bin_ranks, 0, bin_ranks->n);
            corr = tk_csr_spearman_distance(neighbors, weights, start, end, bin_ranks, weight_ranks_buffer, weight_rank_map);
            break;
          case TK_EVAL_METRIC_NDCG:
            tk_pvec_asc(bin_ranks, 0, bin_ranks->n);
            corr = tk_csr_ndcg_distance(neighbors, weights, start, end, bin_ranks, rank_buffer_b);
            break;
          default:
            corr = 0.0;
            break;
        }
        total += corr;
        total_nodes_processed++;
      }

      tk_pvec_destroy(bin_ranks);
      tk_pvec_destroy(weight_ranks_buffer);
      tk_dumap_destroy(rank_buffer_b);
      tk_dumap_destroy(weight_rank_map);
      tk_ivec_destroy(count_buffer);
      tk_dvec_destroy(avgrank_buffer);
      tk_iuset_destroy(itmp);
    }
  }

  return total_nodes_processed > 0 ? total / (double) total_nodes_processed : 0.0;
}

static void tm_optimize_bits_prefix_greedy (
  lua_State *L,
  tk_eval_t *state,
  int i_each
) {
  uint64_t n_dims = state->n_dims;
  uint64_t keep_prefix = state->optimal_k;
  int64_t start_prefix = state->start_prefix;
  double tolerance = state->tolerance;
  uint64_t bytes_per_mask = TK_CVEC_BITS_BYTES(n_dims);
  tk_cvec_t *mask_cvec = tk_cvec_create(L, bytes_per_mask, 0, 0);
  tk_cvec_t *candidate_cvec = tk_cvec_create(L, bytes_per_mask, 0, 0);
  char *mask = mask_cvec->a;
  char *candidate = candidate_cvec->a;

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
      double score = tk_compute_reconstruction(state->codes, state->ann, state->hbi, state->adjacency_ids,
        state->n_dims, state->offsets, state->neighbors, state->weights, state->eval_metric, mask, k);
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
      double score = tk_compute_reconstruction(state->codes, state->ann, state->hbi, state->adjacency_ids,
        state->n_dims, state->offsets, state->neighbors, state->weights, state->eval_metric, mask, 1);
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
    best_prefix_score = tk_compute_reconstruction(state->codes, state->ann, state->hbi, state->adjacency_ids,
      state->n_dims, state->offsets, state->neighbors, state->weights, state->eval_metric, mask, best_prefix);
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

      for (uint64_t i = 0; i < active->n; i ++) {
        int64_t bit = active->a[i];
        if (keep_prefix > 0 && bit < (int64_t)keep_prefix)
          continue;
        memcpy(candidate, mask, bytes_per_mask);
        candidate[TK_CVEC_BITS_BYTE(bit)] &= ~(1 << TK_CVEC_BITS_BIT(bit));
        double score = tk_compute_reconstruction(state->codes, state->ann, state->hbi, state->adjacency_ids,
          state->n_dims, state->offsets, state->neighbors, state->weights, state->eval_metric, candidate, active->n - 1);
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
        double score = tk_compute_reconstruction(state->codes, state->ann, state->hbi, state->adjacency_ids,
          state->n_dims, state->offsets, state->neighbors, state->weights, state->eval_metric, candidate, active->n + 1);
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

  }

  tk_ivec_asc(active, 0, active->n);
}

static inline int tm_optimize_bits (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_cvec_t *cvec = tk_cvec_peek(L, -1, "codes");
  char *codes = cvec->a;
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "expected_offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_weights");
  tk_dvec_t *weights = tk_dvec_peek(L, -1, "expected_weights");
  lua_pop(L, 1);

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "optimize_bits", "n_dims");

  uint64_t keep_prefix = tk_lua_foptunsigned(L, 1, "optimize_bits", "keep_prefix", 0);
  int64_t start_prefix = tk_lua_foptinteger(L, 1, "optimize_bits", "start_prefix", 0);
  double tolerance = tk_lua_foptnumber(L, 1, "optimize_bits", "tolerance", 1e-12);

  char *metric_str = tk_lua_foptstring(L, 1, "optimize_bits", "metric", "spearman");
  tk_eval_metric_t metric = tk_eval_parse_metric(metric_str);
  if (metric == TK_EVAL_METRIC_NONE)
    tk_lua_verror(L, 3, "optimize_bits", "metric", "unknown metric");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_eval_t state;
  memset(&state, 0, sizeof(tk_eval_t));
  state.codes = codes;
  state.ann = NULL;
  state.hbi = NULL;
  state.adjacency_ids = NULL;
  state.n_dims = n_dims;
  state.chunks = TK_CVEC_BITS_BYTES(n_dims);
  state.offsets = offsets;
  state.neighbors = neighbors;
  state.weights = weights;
  state.optimal_k = keep_prefix;
  state.start_prefix = start_prefix;
  state.tolerance = tolerance;
  state.eval_metric = metric;
  state.L = L;

  tm_optimize_bits_prefix_greedy(L, &state, i_each);
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
  lua_newtable(L);
  lua_newtable(L);
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
  tk_ivec_t *dendro_offsets = tk_ivec_peek(L, 1, "offsets");
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
  tk_pvec_dendro_cut(L, dendro_offsets, merges, step, assignments);

  uint64_t n_samples = assignments->n;

  int64_t n_clusters = 0;
  for (uint64_t i = 0; i < n_samples; i++)
    if (assignments->a[i] >= n_clusters)
      n_clusters = assignments->a[i] + 1;

  tk_ivec_t *cluster_offsets = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, -1, -1);
  int i_offsets = tk_lua_absindex(L, -1);
  tk_ivec_ensure(cluster_offsets, (uint64_t)(n_clusters + 1));
  cluster_offsets->n = (uint64_t)(n_clusters + 1);

  for (int64_t i = 0; i < n_clusters + 1; i++) {
    cluster_offsets->a[i] = 0;
  }
  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t cluster = assignments->a[i];
    cluster_offsets->a[cluster + 1]++;
  }

  for (int64_t i = 1; i < n_clusters + 1; i++) {
    cluster_offsets->a[i] += cluster_offsets->a[i - 1];
  }

  tk_ivec_t *cluster_members = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, -1, -1);
  int i_members = tk_lua_absindex(L, -1);
  tk_ivec_ensure(cluster_members, n_samples);
  cluster_members->n = n_samples;

  tk_ivec_t *cluster_positions = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, -1, -1);
  tk_ivec_ensure(cluster_positions, (uint64_t)n_clusters);
  cluster_positions->n = (uint64_t)n_clusters;
  for (int64_t i = 0; i < n_clusters; i++) {
    cluster_positions->a[i] = cluster_offsets->a[i];
  }

  for (uint64_t i = 0; i < n_samples; i++) {
    int64_t cluster = assignments->a[i];
    int64_t pos = cluster_positions->a[cluster];
    cluster_members->a[pos] = (int64_t)i;
    cluster_positions->a[cluster]++;
  }

  tk_ivec_destroy(cluster_positions);
  lua_pop(L, 1);

  lua_pushvalue(L, i_members);
  lua_pushvalue(L, i_offsets);
  lua_pushvalue(L, i_assignments);
  lua_replace(L, 1);
  lua_replace(L, 2);
  lua_replace(L, 3);
  lua_settop(L, 3);
  lua_gc(L, LUA_GCCOLLECT, 0);
  return 3;
}

typedef struct {
  tk_ivec_t *offsets;
  tk_pvec_t *merges;
  tk_ivec_t *ids;
  tk_ivec_t *assignments;
  tk_ivec_t *raw_assignments;
  tk_iumap_t *absorbed_to_surviving;
  tk_iumap_t *cluster_sizes;
  uint64_t n_samples;
  uint64_t n_steps;
  uint64_t current_step;
  int64_t current_merge_idx;
  bool all_merges;
} tk_dendro_iter_t;

static inline int tk_dendro_iter_gc(lua_State *L) {
  tk_dendro_iter_t *iter = luaL_checkudata(L, 1, "tk_dendro_iter_t");
  if (iter->absorbed_to_surviving) {
    tk_iumap_destroy(iter->absorbed_to_surviving);
    iter->absorbed_to_surviving = NULL;
  }
  if (iter->cluster_sizes) {
    tk_iumap_destroy(iter->cluster_sizes);
    iter->cluster_sizes = NULL;
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
    for (uint64_t i = 0; i < iter->n_samples; i++) {
      int64_t cluster_id = iter->offsets->a[i];
      iter->raw_assignments->a[i] = cluster_id;
      if (iter->all_merges) {
        int kha;
        khint_t khi = tk_iumap_put(iter->cluster_sizes, cluster_id, &kha);
        tk_iumap_setval(iter->cluster_sizes, khi, 1);
      }
    }
    iter->raw_assignments->n = iter->n_samples;
    for (uint64_t i = 0; i < iter->n_samples; i++)
      iter->ids->a[i] = (int64_t)i;
    iter->ids->n = iter->n_samples;
  } else {
    int64_t start, end;
    if (iter->all_merges) {
      if (iter->current_merge_idx >= (int64_t)iter->merges->n) {
        lua_pushnil(L);
        return 1;
      }
      start = iter->current_merge_idx;

      int64_t distance_end = (int64_t)iter->merges->n;
      for (uint64_t d = 0; d < iter->offsets->n - iter->n_samples; d++) {
        int64_t dist_start = iter->offsets->a[iter->n_samples + d];
        int64_t dist_next = (iter->n_samples + d + 1 < iter->offsets->n) ?
                            iter->offsets->a[iter->n_samples + d + 1] : (int64_t)iter->merges->n;
        if (start >= dist_start && start < dist_next) {
          distance_end = dist_next;
          break;
        }
      }

      tk_pair_t first_merge = iter->merges->a[start];
      khint_t khi_i = tk_iumap_get(iter->cluster_sizes, first_merge.i);
      khint_t khi_p = tk_iumap_get(iter->cluster_sizes, first_merge.p);
      int64_t size_i = (khi_i != tk_iumap_end(iter->cluster_sizes)) ? tk_iumap_val(iter->cluster_sizes, khi_i) : 1;
      int64_t size_p = (khi_p != tk_iumap_end(iter->cluster_sizes)) ? tk_iumap_val(iter->cluster_sizes, khi_p) : 1;
      int64_t target_size = size_i + size_p;
      end = start + 1;
      while (end < distance_end && end < (int64_t)iter->merges->n) {
        tk_pair_t next_merge = iter->merges->a[end];
        khi_i = tk_iumap_get(iter->cluster_sizes, next_merge.i);
        khi_p = tk_iumap_get(iter->cluster_sizes, next_merge.p);
        int64_t next_i = (khi_i != tk_iumap_end(iter->cluster_sizes)) ? tk_iumap_val(iter->cluster_sizes, khi_i) : 1;
        int64_t next_p = (khi_p != tk_iumap_end(iter->cluster_sizes)) ? tk_iumap_val(iter->cluster_sizes, khi_p) : 1;
        int64_t next_size = next_i + next_p;
        if (next_size != target_size)
          break;
        end++;
      }
      iter->current_merge_idx = end;
    } else {
      start = iter->offsets->a[iter->n_samples + step - 1];
      end = (step + iter->n_samples < iter->offsets->n) ?
        iter->offsets->a[iter->n_samples + step] : (int64_t)iter->merges->n;
    }
    for (int64_t idx = start; idx < end && idx < (int64_t)iter->merges->n; idx++) {
      tk_pair_t merge = iter->merges->a[idx];
      int64_t absorbed = merge.i;
      int64_t surviving = merge.p;
      if (iter->all_merges) {
        khint_t khi_i = tk_iumap_get(iter->cluster_sizes, absorbed);
        khint_t khi_p = tk_iumap_get(iter->cluster_sizes, surviving);
        int64_t size_i = (khi_i != tk_iumap_end(iter->cluster_sizes)) ? tk_iumap_val(iter->cluster_sizes, khi_i) : 1;
        int64_t size_p = (khi_p != tk_iumap_end(iter->cluster_sizes)) ? tk_iumap_val(iter->cluster_sizes, khi_p) : 1;
        int64_t new_size = size_i + size_p;
        int kha;
        khint_t khi = tk_iumap_put(iter->cluster_sizes, surviving, &kha);
        tk_iumap_setval(iter->cluster_sizes, khi, new_size);
      }
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
  lua_settop(L, 3);
  tk_ivec_t *offsets = tk_ivec_peek(L, 1, "offsets");
  tk_pvec_t *merges = tk_pvec_peek(L, 2, "merges");
  bool all_merges = tk_lua_optboolean(L, 3, false, "all_merges");
  uint64_t n_samples = 0;
  for (uint64_t i = 0; i < offsets->n; i++) {
    if (offsets->a[i] == 0 && i > 0) {
      n_samples = i;
      break;
    }
  }
  if (n_samples == 0 || n_samples > offsets->n)
    tk_error(L, "tk_dendro_iter: invalid dendro_offsets structure", EINVAL);
  uint64_t n_steps = all_merges ? merges->n : (offsets->n > n_samples ? offsets->n - n_samples : 0);
  tk_dendro_iter_t *iter = tk_lua_newuserdata(L, tk_dendro_iter_t, TK_EVAL_EPH, NULL, tk_dendro_iter_gc);
  int iter_idx = lua_gettop(L);
  iter->offsets = offsets;
  iter->merges = merges;
  iter->n_samples = n_samples;
  iter->n_steps = n_steps;
  iter->current_step = 0;
  iter->current_merge_idx = 0;
  iter->all_merges = all_merges;
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
  iter->cluster_sizes = tk_iumap_create(L, 0);
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
  { "score_retrieval", tm_score_retrieval },
  { "cluster", tm_cluster },
  { "score_clustering", tm_score_clustering },
  { "entropy_stats", tm_entropy_stats },
  { "dendro_cut", tk_pvec_dendro_cut_lua },
  { "dendro_each", tk_dendro_iter_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_evaluator_capi (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_evaluator_fns, 0);
  return 1;
}
