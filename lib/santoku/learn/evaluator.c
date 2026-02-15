#include <santoku/iuset.h>
#include <santoku/pumap.h>
#include <santoku/learn/inv.h>
#include <santoku/learn/ann.h>
#include <santoku/learn/centroid.h>
#include <santoku/learn/csr.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/rvec.h>
#include <santoku/pvec.h>
#include <santoku/evec.h>
#include <santoku/euset.h>
#include <santoku/iumap.h>
#include <math.h>
#include <float.h>
#include <omp.h>
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

typedef struct {
  atomic_ulong *TP, *FP, *FN;
  tk_ivec_t *ids;
  tk_inv_t *inv;
  tk_ann_t *ann;
  uint64_t min_pts;
  bool assign_noise;
  tk_dvec_t *inv_thresholds;
  uint64_t probe_radius;
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
  tk_ivec_t *offsets;
  tk_ivec_t *neighbors;
  tk_dvec_t *weights;
  lua_State *L;
  tk_inv_hoods_t *inv_hoods;
  tk_ann_hoods_t *ann_hoods;
  tk_ivec_t *uids_hoods;
  uint64_t margin;
  uint64_t target_dims;
  uint64_t min_dims;
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
  return 1;
}

static inline int tm_regression_accuracy (lua_State *L)
{
  lua_settop(L, 2);
  tk_dvec_t *predicted = tk_dvec_peek(L, 1, "predicted");
  tk_dvec_t *expected_d = tk_dvec_peekopt(L, 2);
  tk_ivec_t *expected_i = expected_d ? NULL : tk_ivec_peek(L, 2, "expected");
  uint64_t n = predicted->n;
  if ((expected_d && expected_d->n != n) || (expected_i && expected_i->n != n))
    return luaL_error(L, "predicted and expected must have same length");
  double total = 0.0, min_err = DBL_MAX, max_err = 0.0, sum_exp = 0.0;
  #pragma omp parallel for reduction(+:total,sum_exp) reduction(min:min_err) reduction(max:max_err)
  for (uint64_t i = 0; i < n; i++) {
    double exp_val = expected_d ? expected_d->a[i] : (double)expected_i->a[i];
    double err = fabs(predicted->a[i] - exp_val);
    total += err;
    sum_exp += exp_val;
    if (err < min_err) min_err = err;
    if (err > max_err) max_err = err;
  }
  double mean = n > 0 ? total / n : 0.0;
  double mean_exp = n > 0 ? sum_exp / n : 0.0;
  double nmae = mean_exp > 0 ? mean / mean_exp : 0.0;
  double var = 0.0;
  #pragma omp parallel for reduction(+:var)
  for (uint64_t i = 0; i < n; i++) {
    double exp_val = expected_d ? expected_d->a[i] : (double)expected_i->a[i];
    double err = fabs(predicted->a[i] - exp_val);
    var += (err - mean) * (err - mean);
  }
  double std = n > 1 ? sqrt(var / (n - 1)) : 0.0;
  lua_newtable(L);
  lua_pushnumber(L, total);
  lua_setfield(L, -2, "total");
  lua_pushnumber(L, mean);
  lua_setfield(L, -2, "mean");
  lua_pushnumber(L, n > 0 ? min_err : 0.0);
  lua_setfield(L, -2, "min");
  lua_pushnumber(L, max_err);
  lua_setfield(L, -2, "max");
  lua_pushnumber(L, std);
  lua_setfield(L, -2, "std");
  lua_pushnumber(L, nmae);
  lua_setfield(L, -2, "nmae");
  return 1;
}

static inline int tm_multilabel_retrieval_accuracy (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "hoods");
  tk_ann_hoods_t *ann_hoods = tk_ann_hoods_peekopt(L, -1);
  tk_inv_hoods_t *inv_hoods = ann_hoods ? NULL : tk_inv_hoods_peekopt(L, -1);
  if (!ann_hoods && !inv_hoods)
    return luaL_error(L, "hoods must be ann_hoods or inv_hoods");
  lua_getfield(L, 1, "hood_ids");
  tk_ivec_t *hood_ids = tk_ivec_peek(L, -1, "hood_ids");
  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 4);
  uint64_t n = ann_hoods ? ann_hoods->n : inv_hoods->n;
  if (exp_off->n != n + 1)
    return luaL_error(L, "expected_offsets length must be hoods count + 1");
  uint64_t total_tp = 0, total_pred = 0, total_exp = 0;
  double sum_f1 = 0.0, sum_p = 0.0, sum_r = 0.0;
  uint64_t valid = 0;
  #pragma omp parallel for reduction(+:total_tp,total_pred,total_exp,sum_f1,sum_p,sum_r,valid)
  for (uint64_t s = 0; s < n; s++) {
    uint64_t k = ann_hoods ? ann_hoods->a[s]->n : inv_hoods->a[s]->n;
    int64_t es = exp_off->a[s], ee = exp_off->a[s + 1];
    uint64_t n_exp = (uint64_t)(ee - es);
    int kha;
    tk_iuset_t *exp_set = tk_iuset_create(NULL, 0);
    for (int64_t i = es; i < ee; i++)
      tk_iuset_put(exp_set, exp_nbr->a[i], &kha);
    uint64_t tp = 0;
    for (uint64_t j = 0; j < k; j++) {
      int64_t pos = ann_hoods ? ann_hoods->a[s]->a[j].i : inv_hoods->a[s]->a[j].i;
      if (tk_iuset_contains(exp_set, hood_ids->a[pos])) tp++;
    }
    tk_iuset_destroy(exp_set);
    total_tp += tp; total_pred += k; total_exp += n_exp;
    if (n_exp > 0 || k > 0) {
      double p = k > 0 ? (double)tp / k : 0.0;
      double r = n_exp > 0 ? (double)tp / n_exp : 0.0;
      double f1 = (p + r) > 0 ? 2.0 * p * r / (p + r) : 0.0;
      sum_f1 += f1; sum_p += p; sum_r += r;
      valid++;
    }
  }
  lua_newtable(L);
  double mp = total_pred > 0 ? (double)total_tp / total_pred : 0.0;
  double mr = total_exp > 0 ? (double)total_tp / total_exp : 0.0;
  double mf = (mp + mr) > 0 ? 2.0 * mp * mr / (mp + mr) : 0.0;
  lua_pushnumber(L, mp); lua_setfield(L, -2, "micro_precision");
  lua_pushnumber(L, mr); lua_setfield(L, -2, "micro_recall");
  lua_pushnumber(L, mf); lua_setfield(L, -2, "micro_f1");
  lua_pushnumber(L, valid > 0 ? sum_p / valid : 0); lua_setfield(L, -2, "macro_precision");
  lua_pushnumber(L, valid > 0 ? sum_r / valid : 0); lua_setfield(L, -2, "macro_recall");
  lua_pushnumber(L, valid > 0 ? sum_f1 / valid : 0); lua_setfield(L, -2, "macro_f1");
  return 1;
}

static inline tk_ivec_t *tk_pvec_dendro_cut(lua_State *L, tk_ivec_t *offsets, tk_pvec_t *merges, uint64_t step, tk_ivec_t *assignments);

static inline int tm_retrieval_ks (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "hoods");
  tk_inv_hoods_t *inv_hoods = tk_inv_hoods_peekopt(L, -1);
  tk_ann_hoods_t *ann_hoods = inv_hoods ? NULL : tk_ann_hoods_peekopt(L, -1);
  if (!inv_hoods && !ann_hoods)
    return luaL_error(L, "hoods must be inv_hoods or ann_hoods");
  lua_getfield(L, 1, "hood_ids");
  tk_ivec_t *hood_ids = tk_ivec_peek(L, -1, "hood_ids");
  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 4);
  uint64_t n_samples = inv_hoods ? inv_hoods->n : ann_hoods->n;
  if (exp_off->n != n_samples + 1)
    return luaL_error(L, "expected_offsets length must match hoods count + 1");
  tk_ivec_t *ks = tk_ivec_create(L, n_samples, NULL, NULL);
  uint64_t mi_tp = 0, mi_k = 0, mi_exp = 0, n_valid = 0;
  double ma_prec = 0, ma_rec = 0, ma_f1 = 0;
  #pragma omp parallel for reduction(+:mi_tp,mi_k,mi_exp,n_valid,ma_prec,ma_rec,ma_f1)
  for (uint64_t s = 0; s < n_samples; s++) {
    uint64_t hood_size = 0;
    if (inv_hoods) hood_size = inv_hoods->a[s]->n;
    else hood_size = ann_hoods->a[s]->n;
    int64_t es = exp_off->a[s], ee = exp_off->a[s + 1];
    uint64_t n_expected = (uint64_t)(ee - es);
    if (n_expected == 0 || hood_size == 0) {
      ks->a[s] = 0;
      continue;
    }
    int kha;
    tk_iuset_t *exp_set = tk_iuset_create(NULL, 0);
    for (int64_t i = es; i < ee; i++)
      tk_iuset_put(exp_set, exp_nbr->a[i], &kha);
    double best_f1 = -1.0;
    uint64_t best_k = 1, best_tp = 0;
    uint64_t tp = 0;
    for (uint64_t k = 1; k <= hood_size; k++) {
      int64_t nbr_pos;
      if (inv_hoods) nbr_pos = inv_hoods->a[s]->a[k-1].i;
      else nbr_pos = ann_hoods->a[s]->a[k-1].i;
      int64_t nbr_id = hood_ids->a[nbr_pos];
      if (tk_iuset_contains(exp_set, nbr_id)) tp++;
      double prec = (double)tp / k;
      double rec = (double)tp / n_expected;
      double f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
      if (f1 > best_f1) {
        best_f1 = f1;
        best_k = k;
        best_tp = tp;
      }
    }
    tk_iuset_destroy(exp_set);
    ks->a[s] = (int64_t)best_k;
    double s_prec = (double)best_tp / best_k;
    double s_rec = (double)best_tp / n_expected;
    mi_tp += best_tp;
    mi_k += best_k;
    mi_exp += n_expected;
    ma_prec += s_prec;
    ma_rec += s_rec;
    ma_f1 += best_f1;
    n_valid++;
  }
  lua_newtable(L);
  double mi_prec = mi_k > 0 ? (double)mi_tp / mi_k : 0;
  double mi_rec = mi_exp > 0 ? (double)mi_tp / mi_exp : 0;
  double mi_f1v = (mi_prec + mi_rec) > 0 ? 2.0 * mi_prec * mi_rec / (mi_prec + mi_rec) : 0;
  lua_pushnumber(L, mi_prec); lua_setfield(L, -2, "micro_precision");
  lua_pushnumber(L, mi_rec); lua_setfield(L, -2, "micro_recall");
  lua_pushnumber(L, mi_f1v); lua_setfield(L, -2, "micro_f1");
  lua_pushnumber(L, n_valid > 0 ? ma_prec / n_valid : 0); lua_setfield(L, -2, "macro_precision");
  lua_pushnumber(L, n_valid > 0 ? ma_rec / n_valid : 0); lua_setfield(L, -2, "macro_recall");
  lua_pushnumber(L, n_valid > 0 ? ma_f1 / n_valid : 0); lua_setfield(L, -2, "macro_f1");
  return 2;
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
  tk_dvec_t *quality_curve;
  tk_dvec_t *auc_curve;
  tk_ivec_t *n_clusters_curve;
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

static inline double tk_cluster_compute_quality(
  tk_cluster_t **clusters,
  uint64_t n_clusters,
  tk_cvec_t *codes,
  uint64_t n_chunks,
  uint64_t n_bits,
  uint64_t n_nodes
) {
  double total_similarity = 0.0;
  uint64_t total_members = 0;

  for (uint64_t c = 0; c < n_clusters; c++) {
    tk_cluster_t *cluster = clusters[c];
    if (!cluster || !cluster->active || cluster->members->n < 1)
      continue;

    char *centroid_code = tk_centroid_code(cluster->centroid);
    double cluster_similarity = 0.0;

    for (uint64_t i = 0; i < cluster->members->n; i++) {
      int64_t member = cluster->members->a[i];
      uint8_t *code = (uint8_t *)(codes->a + (uint64_t)member * n_chunks);
      uint64_t hamming_dist = tk_cvec_bits_hamming_serial(code, (uint8_t *)centroid_code, n_bits);
      double similarity = 1.0 - (double)hamming_dist / (double)n_bits;
      cluster_similarity += similarity;
    }

    total_similarity += cluster_similarity;
    total_members += cluster->members->n;
  }

  return total_members > 0 ? total_similarity / (double)total_members : 0.0;
}

static inline double tk_cluster_compute_auc(
  tk_ivec_t *entity_to_cluster,
  tk_ivec_t *exp_off,
  tk_ivec_t *exp_nbr,
  tk_dvec_t *exp_wgt,
  uint64_t n_samples
) {
  uint64_t concordant = 0, total_pairs = 0;
  for (uint64_t s = 0; s < n_samples; s++) {
    int64_t es = exp_off->a[s], ee = exp_off->a[s + 1];
    for (int64_t i = es; i < ee; i++) {
      int64_t ni = exp_nbr->a[i];
      if (ni < 0 || (uint64_t)ni >= n_samples) continue;
      bool same_i = entity_to_cluster->a[s] == entity_to_cluster->a[ni];
      double wi = exp_wgt->a[i];
      for (int64_t j = i + 1; j < ee; j++) {
        int64_t nj = exp_nbr->a[j];
        if (nj < 0 || (uint64_t)nj >= n_samples) continue;
        bool same_j = entity_to_cluster->a[s] == entity_to_cluster->a[nj];
        if (same_i == same_j) continue;
        double wj = exp_wgt->a[j];
        total_pairs++;
        if ((same_i && wi > wj) || (same_j && wj > wi)) concordant++;
        else if (wi == wj) concordant++;
      }
    }
  }
  return total_pairs > 0 ? (double)concordant / total_pairs : 0.0;
}

static inline int tk_cluster_centroid(
  lua_State *L,
  tk_cvec_t *codes,
  tk_ivec_t *adj_ids,
  tk_ivec_t *adj_offsets,
  tk_ivec_t *adj_neighbors,
  uint64_t n_bits,
  tk_ivec_t *dendro_offsets,
  tk_pvec_t *dendro_merges,
  tk_dvec_t *quality_curve,
  tk_dvec_t *auc_curve,
  tk_ivec_t *exp_off,
  tk_ivec_t *exp_nbr,
  tk_dvec_t *exp_wgt,
  tk_ivec_t *n_clusters_curve,
  bool early_exit
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

  double initial_quality = tk_cluster_compute_quality(clusters, n_clusters, codes, n_chunks, n_bits, n_nodes);
  tk_dvec_push(quality_curve, initial_quality);
  tk_ivec_push(n_clusters_curve, (int64_t)n_active);
  if (auc_curve) {
    double initial_auc = tk_cluster_compute_auc(entity_to_cluster, exp_off, exp_nbr, exp_wgt, n_nodes);
    tk_dvec_push(auc_curve, initial_auc);
  }

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


    uint64_t heap_min = early_exit && edge_heap->n > 0 ? (uint64_t)edge_heap->a[0].w : 0;

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

        double step_quality = tk_cluster_compute_quality(clusters, n_clusters, codes, n_chunks, n_bits, n_nodes);
        tk_dvec_push(quality_curve, step_quality);
        tk_ivec_push(n_clusters_curve, (int64_t)n_active);
        if (auc_curve) {
          double step_auc = tk_cluster_compute_auc(entity_to_cluster, exp_off, exp_nbr, exp_wgt, n_nodes);
          tk_dvec_push(auc_curve, step_auc);
        }
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
  int i_eph,
  bool early_exit,
  tk_ivec_t *exp_off,
  tk_ivec_t *exp_nbr,
  tk_dvec_t *exp_wgt
) {
  tm_cluster_result_t result;

  result.dendro_offsets = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.dendro_merges = tk_pvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.quality_curve = tk_dvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.n_clusters_curve = tk_ivec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.auc_curve = NULL;
  if (exp_off && exp_nbr && exp_wgt) {
    result.auc_curve = tk_dvec_create(L, 0, 0, 0);
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
    lua_pop(L, 1);
  }

  tk_cluster_centroid(L, codes, adj_ids, adj_offsets, adj_neighbors,
                     n_bits, result.dendro_offsets, result.dendro_merges,
                     result.quality_curve, result.auc_curve, exp_off, exp_nbr, exp_wgt,
                     result.n_clusters_curve, early_exit);

  result.n_steps = result.dendro_offsets->n > adj_ids->n + 1 ?
                   result.dendro_offsets->n - adj_ids->n - 1 : 0;

  return result;
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

  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peekopt(L, -1);
  if (!lua_isnil(L, -1))
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peekopt(L, -1);
  if (!lua_isnil(L, -1))
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_weights");
  tk_dvec_t *exp_wgt = tk_dvec_peekopt(L, -1);
  if (!lua_isnil(L, -1))
    tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  uint64_t n_bits = tk_lua_fcheckunsigned(L, 1, "cluster", "n_dims");
  bool early_exit = tk_lua_foptboolean(L, 1, "cluster", "early_exit", true);

  tm_cluster_result_t result = tm_cluster_agglo(
    L, codes, ids, offsets, neighbors, n_bits, i_eph, early_exit,
    exp_off, exp_nbr, exp_wgt);

  lua_newtable(L);

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.dendro_offsets);
  lua_setfield(L, -2, "offsets");

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.dendro_merges);
  lua_setfield(L, -2, "merges");

  lua_pushinteger(L, (lua_Integer)result.n_steps);
  lua_setfield(L, -2, "n_steps");

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, ids);
  lua_setfield(L, -2, "ids");

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.quality_curve);
  lua_setfield(L, -2, "radius_curve");

  tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.n_clusters_curve);
  lua_setfield(L, -2, "n_clusters_curve");

  if (result.auc_curve) {
    tk_lua_get_ephemeron(L, TK_EVAL_EPH, result.auc_curve);
    lua_setfield(L, -2, "auc_curve");
  }

  lua_replace(L, 1);
  lua_settop(L, 1);
  return 1;
}

static inline int tm_score_retrieval (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "codes");
  tk_cvec_t *codes_cvec = tk_cvec_peekopt(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "raw_codes");
  tk_dvec_t *raw_codes_dvec = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "kernel_index");
  tk_inv_t *kernel_index = tk_inv_peekopt(L, -1);
  lua_pop(L, 1);

  double kernel_decay = 0.0;
  double kernel_bandwidth = -1.0;

  if (kernel_index) {
    kernel_decay = tk_lua_foptnumber(L, 1, "score_retrieval", "kernel_decay", 0.0);
    kernel_bandwidth = tk_lua_foptnumber(L, 1, "score_retrieval", "kernel_bandwidth", -1.0);
  }

  tk_ann_t *ann = NULL;
  char *codes = NULL;
  double *raw_codes = NULL;

  if (kernel_index) {
    // kernel mode - no codes needed
  } else if (raw_codes_dvec) {
    raw_codes = raw_codes_dvec->a;
  } else if (codes_cvec) {
    codes = codes_cvec->a;
  } else {
    lua_getfield(L, 1, "index");
    ann = tk_ann_peekopt(L, -1);
    lua_pop(L, 1);
    if (!ann)
      tk_lua_verror(L, 3, "score_retrieval", "codes/index/raw_codes/kernel_index", "codes, raw_codes, index, or kernel_index required");
  }

  tk_ivec_t *code_ids = NULL;
  if (!kernel_index) {
    lua_getfield(L, 1, "ids");
    code_ids = tk_ivec_peek(L, -1, "ids");
    lua_pop(L, 1);
  }

  lua_getfield(L, 1, "eval_ids");
  tk_ivec_t *eval_ids = tk_ivec_peek(L, -1, "eval_ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "eval_offsets");
  tk_ivec_t *eval_offsets = tk_ivec_peek(L, -1, "eval_offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "eval_neighbors");
  tk_ivec_t *eval_neighbors = tk_ivec_peek(L, -1, "eval_neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "eval_weights");
  tk_dvec_t *eval_weights = tk_dvec_peek(L, -1, "eval_weights");
  lua_pop(L, 1);

  uint64_t n_dims = 0;
  uint64_t chunks = 0;
  if (!kernel_index) {
    n_dims = tk_lua_fcheckunsigned(L, 1, "score_retrieval", "n_dims");
    chunks = TK_CVEC_BITS_BYTES(n_dims);
  }

  const char *ranking_str = tk_lua_foptstring(L, 1, "score_retrieval", "ranking", "ndcg");
  tk_eval_metric_t ranking = tk_eval_parse_metric(ranking_str);
  if (ranking != TK_EVAL_METRIC_NDCG &&
      ranking != TK_EVAL_METRIC_SPEARMAN &&
      ranking != TK_EVAL_METRIC_PEARSON)
    tk_lua_verror(L, 1, "score_retrieval", "ranking", "must be ndcg, spearman, or pearson");

  tk_iumap_t *code_id_to_idx = NULL;
  if (!kernel_index) {
    code_id_to_idx = tk_iumap_from_ivec(NULL, code_ids);
    if (!code_id_to_idx)
      tk_error(L, "score_retrieval: failed to create code ID mapping", ENOMEM);
  }

  double total_score = 0.0;
  uint64_t total_queries = 0;

  #pragma omp parallel reduction(+:total_score) reduction(+:total_queries)
  {
    tk_pvec_t *query_neighbors = tk_pvec_create(NULL, 0, 0, 0);
    tk_rvec_t *query_neighbors_raw = tk_rvec_create(NULL, 0, 0, 0);
    tk_dumap_t *weight_map = tk_dumap_create(NULL, 0);
    tk_rvec_t *weight_ranks_buffer = tk_rvec_create(NULL, 0, 0, 0);
    tk_dumap_t *weight_rank_map = tk_dumap_create(NULL, 0);

    if (!query_neighbors || !query_neighbors_raw || !weight_map || !weight_ranks_buffer || !weight_rank_map) {
      if (query_neighbors) tk_pvec_destroy(query_neighbors);
      if (query_neighbors_raw) tk_rvec_destroy(query_neighbors_raw);
      if (weight_map) tk_dumap_destroy(weight_map);
      if (weight_ranks_buffer) tk_rvec_destroy(weight_ranks_buffer);
      if (weight_rank_map) tk_dumap_destroy(weight_rank_map);
    } else {
      #pragma omp for schedule(static)
      for (uint64_t query_idx = 0; query_idx < eval_offsets->n - 1; query_idx++) {
        int64_t query_id = eval_ids->a[query_idx];

        int64_t query_code_idx = -1;
        if (!kernel_index) {
          khint_t code_khi = tk_iumap_get(code_id_to_idx, query_id);
          if (code_khi == tk_iumap_end(code_id_to_idx))
            continue;
          query_code_idx = tk_iumap_val(code_id_to_idx, code_khi);
        }

        int64_t eval_start = eval_offsets->a[query_idx];
        int64_t eval_end = eval_offsets->a[query_idx + 1];
        if (eval_end == eval_start)
          continue;

        char *query_code = NULL;
        double *query_raw = NULL;
        if (kernel_index) {
          // kernel mode - no codes needed, use query_id directly
        } else if (raw_codes) {
          query_raw = raw_codes + (uint64_t)query_code_idx * n_dims;
        } else if (codes) {
          query_code = codes + (uint64_t)query_code_idx * chunks;
        } else {
          query_code = tk_ann_get(ann, query_id);
        }
        if (!kernel_index && !query_code && !query_raw)
          continue;

        tk_pvec_clear(query_neighbors);
        tk_rvec_clear(query_neighbors_raw);

        for (int64_t i = eval_start; i < eval_end; i++) {
          int64_t neighbor_eval_idx = eval_neighbors->a[i];
          int64_t neighbor_id = eval_ids->a[neighbor_eval_idx];

          if (kernel_index) {
            double kernel_dist = tk_inv_distance(kernel_index, query_id, neighbor_id,
              kernel_decay, kernel_bandwidth);
            tk_rvec_push(query_neighbors_raw, tk_rank(neighbor_eval_idx, kernel_dist));
          } else if (raw_codes) {
            khint_t nbr_khi = tk_iumap_get(code_id_to_idx, neighbor_id);
            if (nbr_khi == tk_iumap_end(code_id_to_idx))
              continue;
            int64_t neighbor_code_idx = tk_iumap_val(code_id_to_idx, nbr_khi);
            double *neighbor_raw = raw_codes + (uint64_t)neighbor_code_idx * n_dims;
            double dot = 0.0, norm_q = 0.0, norm_n = 0.0;
            for (uint64_t d = 0; d < n_dims; d++) {
              dot += query_raw[d] * neighbor_raw[d];
              norm_q += query_raw[d] * query_raw[d];
              norm_n += neighbor_raw[d] * neighbor_raw[d];
            }
            double denom = sqrt(norm_q) * sqrt(norm_n);
            double dist = 1.0 - ((denom > 1e-12) ? dot / denom : 0.0);
            tk_rvec_push(query_neighbors_raw, tk_rank(neighbor_eval_idx, dist));
          } else {
            char *neighbor_code = codes
              ? NULL
              : tk_ann_get(ann, neighbor_id);

            if (codes) {
              khint_t nbr_khi = tk_iumap_get(code_id_to_idx, neighbor_id);
              if (nbr_khi == tk_iumap_end(code_id_to_idx))
                continue;
              int64_t neighbor_code_idx = tk_iumap_val(code_id_to_idx, nbr_khi);
              neighbor_code = codes + (uint64_t)neighbor_code_idx * chunks;
            } else if (!neighbor_code) {
              continue;
            }

            uint64_t hamming_dist = tk_cvec_bits_hamming_serial(
              (const unsigned char*)query_code,
              (const unsigned char*)neighbor_code,
              n_dims);

            tk_pvec_push(query_neighbors, tk_pair(neighbor_eval_idx, (int64_t)hamming_dist));
          }
        }

        if (kernel_index || raw_codes) {
          if (query_neighbors_raw->n == 0) {
            total_queries++;
            continue;
          }
          tk_rvec_asc(query_neighbors_raw, 0, query_neighbors_raw->n);
          tk_pvec_clear(query_neighbors);
          for (uint64_t j = 0; j < query_neighbors_raw->n; j++)
            tk_pvec_push(query_neighbors, tk_pair(query_neighbors_raw->a[j].i, (int64_t)j));
        } else {
          if (query_neighbors->n == 0) {
            total_queries++;
            continue;
          }
          tk_pvec_asc(query_neighbors, 0, query_neighbors->n);
        }

        double ranking_score = 0.0;
        switch (ranking) {
          case TK_EVAL_METRIC_NDCG:
            ranking_score = tk_csr_ndcg_distance(eval_ids, eval_neighbors, eval_weights,
              eval_start, eval_end, eval_ids, query_neighbors, weight_map);
            break;
          case TK_EVAL_METRIC_SPEARMAN:
            ranking_score = tk_csr_spearman_distance(eval_ids, eval_neighbors, eval_weights,
              eval_start, eval_end, eval_ids, query_neighbors, weight_ranks_buffer, weight_rank_map);
            break;
          case TK_EVAL_METRIC_PEARSON:
            ranking_score = tk_csr_pearson_distance(eval_ids, eval_neighbors, eval_weights,
              eval_start, eval_end, eval_ids, query_neighbors, weight_map);
            break;
          default:
            break;
        }

        total_queries++;
        total_score += ranking_score;
      }

      tk_pvec_destroy(query_neighbors);
      tk_rvec_destroy(query_neighbors_raw);
      tk_dumap_destroy(weight_map);
      tk_rvec_destroy(weight_ranks_buffer);
      tk_dumap_destroy(weight_rank_map);
    }
  }

  if (code_id_to_idx)
    tk_iumap_destroy(code_id_to_idx);

  double avg_score = total_queries > 0 ? total_score / (double)total_queries : 0.0;

  lua_newtable(L);
  lua_pushnumber(L, avg_score);
  lua_setfield(L, -2, "score");
  lua_pushinteger(L, (lua_Integer)total_queries);
  lua_setfield(L, -2, "total_queries");

  lua_replace(L, 1);
  lua_settop(L, 1);
  return 1;
}

typedef struct {
  uint8_t *xor_codes;
  uint16_t *distances;
  uint64_t n_pairs;
  uint64_t n_dims;
  uint64_t chunks;
  tk_ivec_t *offsets;
  tk_ivec_t *neighbors;
  tk_dvec_t *weights;
} tk_incr_state_t;

static double tk_incr_compute_ndcg (
  tk_incr_state_t *state,
  uint64_t bit_to_add,
  bool commit,
  uint64_t n_selected_bits
) {
  uint64_t n_nodes = state->offsets->n - 1;
  double total_ndcg = 0.0;
  uint64_t byte_idx = TK_CVEC_BITS_BYTE(bit_to_add);
  uint8_t bit_mask = 1 << TK_CVEC_BITS_BIT(bit_to_add);
  uint64_t n_buckets = n_selected_bits + 2;

  uint64_t max_neighbors = 0;
  for (uint64_t i = 0; i < n_nodes; i++) {
    uint64_t cnt = (uint64_t)(state->offsets->a[i + 1] - state->offsets->a[i]);
    if (cnt > max_neighbors) max_neighbors = cnt;
  }

  #pragma omp parallel reduction(+:total_ndcg)
  {
    uint64_t *buckets = malloc(n_buckets * sizeof(uint64_t));
    uint64_t *bucket_counts = calloc(n_buckets, sizeof(uint64_t));
    int64_t *temp_indices = malloc(max_neighbors * sizeof(int64_t));
    int64_t *sorted = malloc(max_neighbors * sizeof(int64_t));
    double *sorted_weights = malloc(max_neighbors * sizeof(double));
    double *bucket_avg_discounts = malloc(n_buckets * sizeof(double));

    #pragma omp for schedule(static)
    for (uint64_t node_idx = 0; node_idx < n_nodes; node_idx++) {
      int64_t start = state->offsets->a[node_idx];
      int64_t end = state->offsets->a[node_idx + 1];
      uint64_t n_neighbors = (uint64_t)(end - start);
      if (n_neighbors == 0)
        continue;

      memset(bucket_counts, 0, n_buckets * sizeof(int64_t));

      for (int64_t j = start; j < end; j++) {
        uint16_t dist = state->distances[j];
        if (bit_to_add < state->n_dims) {
          unsigned char *xor_code = (unsigned char *)(state->xor_codes + (uint64_t)j * state->chunks);
          if (xor_code[byte_idx] & bit_mask)
            dist++;
        }
        bucket_counts[dist]++;
        temp_indices[j - start] = ((int64_t)dist << 48) | j;
        sorted_weights[j - start] = state->weights->a[j];
      }

      for (uint64_t i = 0; i < n_neighbors; i++) {
        for (uint64_t k = i + 1; k < n_neighbors; k++) {
          if (sorted_weights[k] > sorted_weights[i]) {
            double tmp = sorted_weights[i];
            sorted_weights[i] = sorted_weights[k];
            sorted_weights[k] = tmp;
          }
        }
      }
      double idcg = 0.0;
      for (uint64_t i = 0; i < n_neighbors; i++)
        idcg += sorted_weights[i] / log2((double)(i + 2));

      if (idcg <= 0.0)
        continue;

      buckets[0] = 0;
      for (uint64_t d = 1; d < n_buckets; d++)
        buckets[d] = buckets[d-1] + bucket_counts[d-1];

      uint64_t pos = 0;
      for (uint64_t d = 0; d < n_buckets; d++) {
        if (bucket_counts[d] == 0) {
          bucket_avg_discounts[d] = 1.0;
          continue;
        }
        double discount_sum = 0.0;
        for (uint64_t p = pos; p < pos + bucket_counts[d]; p++)
          discount_sum += log2((double)(p + 2));
        bucket_avg_discounts[d] = discount_sum / (double)bucket_counts[d];
        pos += bucket_counts[d];
      }

      for (uint64_t i = 0; i < n_neighbors; i++) {
        int64_t packed = temp_indices[i];
        uint16_t dist = (uint16_t)(packed >> 48);
        int64_t j = packed & 0xFFFFFFFFFFFFLL;
        sorted[buckets[dist]++] = j;
      }

      double dcg = 0.0;
      pos = 0;
      for (uint64_t d = 0; d < n_buckets; d++) {
        for (uint64_t i = 0; i < bucket_counts[d]; i++) {
          int64_t j = sorted[pos + i];
          double w = state->weights->a[j];
          dcg += w / bucket_avg_discounts[d];
        }
        pos += bucket_counts[d];
      }

      total_ndcg += dcg / idcg;
    }

    free(buckets);
    free(bucket_counts);
    free(temp_indices);
    free(sorted);
    free(sorted_weights);
    free(bucket_avg_discounts);
  }

  if (commit && bit_to_add < state->n_dims) {
    #pragma omp parallel for schedule(static)
    for (uint64_t j = 0; j < state->n_pairs; j++) {
      unsigned char *xor_code = (unsigned char *)(state->xor_codes + j * state->chunks);
      if (xor_code[byte_idx] & bit_mask)
        state->distances[j]++;
    }
  }

  return n_nodes > 0 ? total_ndcg / (double) n_nodes : 0.0;
}

static double tk_incr_compute_ndcg_sub (
  tk_incr_state_t *state,
  uint64_t bit_to_remove,
  bool commit,
  uint64_t n_selected_bits
) {
  uint64_t n_nodes = state->offsets->n - 1;
  double total_ndcg = 0.0;
  uint64_t byte_idx = TK_CVEC_BITS_BYTE(bit_to_remove);
  uint8_t bit_mask = 1 << TK_CVEC_BITS_BIT(bit_to_remove);
  uint64_t n_buckets = n_selected_bits + 2;

  uint64_t max_neighbors = 0;
  for (uint64_t i = 0; i < n_nodes; i++) {
    uint64_t cnt = (uint64_t)(state->offsets->a[i + 1] - state->offsets->a[i]);
    if (cnt > max_neighbors) max_neighbors = cnt;
  }

  #pragma omp parallel reduction(+:total_ndcg)
  {
    uint64_t *buckets = malloc(n_buckets * sizeof(uint64_t));
    uint64_t *bucket_counts = calloc(n_buckets, sizeof(uint64_t));
    int64_t *temp_indices = malloc(max_neighbors * sizeof(int64_t));
    int64_t *sorted = malloc(max_neighbors * sizeof(int64_t));
    double *sorted_weights = malloc(max_neighbors * sizeof(double));
    double *bucket_avg_discounts = malloc(n_buckets * sizeof(double));

    #pragma omp for schedule(static)
    for (uint64_t node_idx = 0; node_idx < n_nodes; node_idx++) {
      int64_t start = state->offsets->a[node_idx];
      int64_t end = state->offsets->a[node_idx + 1];
      uint64_t n_neighbors = (uint64_t)(end - start);
      if (n_neighbors == 0)
        continue;

      memset(bucket_counts, 0, n_buckets * sizeof(int64_t));

      for (int64_t j = start; j < end; j++) {
        uint16_t dist = state->distances[j];
        if (bit_to_remove < state->n_dims) {
          unsigned char *xor_code = (unsigned char *)(state->xor_codes + (uint64_t)j * state->chunks);
          if (xor_code[byte_idx] & bit_mask)
            dist--;
        }
        bucket_counts[dist]++;
        temp_indices[j - start] = ((int64_t)dist << 48) | j;
        sorted_weights[j - start] = state->weights->a[j];
      }

      for (uint64_t i = 0; i < n_neighbors; i++) {
        for (uint64_t k = i + 1; k < n_neighbors; k++) {
          if (sorted_weights[k] > sorted_weights[i]) {
            double tmp = sorted_weights[i];
            sorted_weights[i] = sorted_weights[k];
            sorted_weights[k] = tmp;
          }
        }
      }
      double idcg = 0.0;
      for (uint64_t i = 0; i < n_neighbors; i++)
        idcg += sorted_weights[i] / log2((double)(i + 2));

      if (idcg <= 0.0)
        continue;

      buckets[0] = 0;
      for (uint64_t d = 1; d < n_buckets; d++)
        buckets[d] = buckets[d-1] + bucket_counts[d-1];

      uint64_t pos = 0;
      for (uint64_t d = 0; d < n_buckets; d++) {
        if (bucket_counts[d] == 0) {
          bucket_avg_discounts[d] = 1.0;
          continue;
        }
        double discount_sum = 0.0;
        for (uint64_t p = pos; p < pos + bucket_counts[d]; p++)
          discount_sum += log2((double)(p + 2));
        bucket_avg_discounts[d] = discount_sum / (double)bucket_counts[d];
        pos += bucket_counts[d];
      }

      for (uint64_t i = 0; i < n_neighbors; i++) {
        int64_t packed = temp_indices[i];
        uint16_t dist = (uint16_t)(packed >> 48);
        int64_t j = packed & 0xFFFFFFFFFFFFLL;
        sorted[buckets[dist]++] = j;
      }

      double dcg = 0.0;
      pos = 0;
      for (uint64_t d = 0; d < n_buckets; d++) {
        for (uint64_t i = 0; i < bucket_counts[d]; i++) {
          int64_t j = sorted[pos + i];
          double w = state->weights->a[j];
          dcg += w / bucket_avg_discounts[d];
        }
        pos += bucket_counts[d];
      }

      total_ndcg += dcg / idcg;
    }

    free(buckets);
    free(bucket_counts);
    free(temp_indices);
    free(sorted);
    free(sorted_weights);
    free(bucket_avg_discounts);
  }

  if (commit && bit_to_remove < state->n_dims) {
    #pragma omp parallel for schedule(static)
    for (uint64_t j = 0; j < state->n_pairs; j++) {
      unsigned char *xor_code = (unsigned char *)(state->xor_codes + j * state->chunks);
      if (xor_code[byte_idx] & bit_mask)
        state->distances[j]--;
    }
  }

  return n_nodes > 0 ? total_ndcg / (double) n_nodes : 0.0;
}

static void tm_optimize_bits_sfbs (
  lua_State *L,
  tk_eval_t *state,
  int i_each
) {
  uint64_t n_dims = state->n_dims;
  uint64_t target_dims = state->target_dims;
  uint64_t min_dims = state->min_dims > 0 ? state->min_dims : 0;
  uint64_t chunks = TK_CVEC_BITS_BYTES(n_dims);
  uint64_t n_nodes = state->offsets->n - 1;
  uint64_t n_pairs = state->neighbors->n;

  uint8_t *xor_codes = tk_malloc(L, n_pairs * chunks);
  memset(xor_codes, 0, n_pairs * chunks);
  uint16_t *distances = tk_malloc(L, n_pairs * sizeof(uint16_t));
  memset(distances, 0, n_pairs * sizeof(uint16_t));

  uint64_t pairs_computed = 0;
  for (uint64_t node_idx = 0; node_idx < n_nodes; node_idx++) {
    int64_t start = state->offsets->a[node_idx];
    int64_t end = state->offsets->a[node_idx + 1];

    uint8_t *node_code = NULL;
    if (state->codes) {
      node_code = (uint8_t *)state->codes + node_idx * chunks;
    } else if (state->ann && state->adjacency_ids) {
      int64_t node_id = state->adjacency_ids->a[node_idx];
      node_code = (uint8_t *)tk_ann_get(state->ann, node_id);
    }
    if (!node_code)
      continue;

    for (int64_t j = start; j < end; j++) {
      int64_t neighbor_pos = state->neighbors->a[j];
      uint8_t *neighbor_code = NULL;
      if (state->codes) {
        neighbor_code = (uint8_t *)state->codes + (uint64_t)neighbor_pos * chunks;
      } else if (state->ann && state->adjacency_ids) {
        int64_t neighbor_id = state->adjacency_ids->a[neighbor_pos];
        neighbor_code = (uint8_t *)tk_ann_get(state->ann, neighbor_id);
      }
      if (!neighbor_code)
        continue;

      uint8_t *xor_dest = xor_codes + (uint64_t)j * chunks;
      for (uint64_t c = 0; c < chunks; c++) {
        xor_dest[c] = node_code[c] ^ neighbor_code[c];
      }
      pairs_computed++;
    }
  }

  if (pairs_computed == 0)
    luaL_error(L, "optimize_bits: no pairs computed (ann lookup failed?)");

  uint64_t *bit_counts = tk_malloc(L, n_dims * sizeof(uint64_t));
  memset(bit_counts, 0, n_dims * sizeof(uint64_t));
  for (uint64_t j = 0; j < n_pairs; j++) {
    uint8_t *xor_code = xor_codes + j * chunks;
    for (uint64_t b = 0; b < n_dims; b++) {
      uint64_t byte_idx = TK_CVEC_BITS_BYTE(b);
      uint8_t bit_mask = (uint8_t)(1 << TK_CVEC_BITS_BIT(b));
      if (xor_code[byte_idx] & bit_mask)
        bit_counts[b]++;
    }
  }

  uint8_t *discriminative = tk_malloc(L, n_dims);
  for (uint64_t b = 0; b < n_dims; b++)
    discriminative[b] = (bit_counts[b] > 0 && bit_counts[b] < n_pairs) ? 1 : 0;
  free(bit_counts);

  tk_incr_state_t incr_state = {
    .xor_codes = xor_codes,
    .distances = distances,
    .n_pairs = n_pairs,
    .n_dims = n_dims,
    .chunks = chunks,
    .offsets = state->offsets,
    .neighbors = state->neighbors,
    .weights = state->weights,
  };

  tk_ivec_t *active = tk_ivec_create(L, n_dims, 0, 0);
  active->n = 0;
  uint8_t *selected = tk_malloc(L, n_dims);
  memset(selected, 0, n_dims);

  double current_score = 0.0;
  uint64_t max_bits = target_dims > 0 ? target_dims : n_dims;

  while (1) {
    bool changed = false;

    if (active->n < max_bits) {
      int64_t best_add_bit = -1;
      double best_add_score = -INFINITY;

      #pragma omp parallel
      {
        int64_t local_best_bit = -1;
        double local_best_score = -INFINITY;

        #pragma omp for schedule(dynamic)
        for (uint64_t b = 0; b < n_dims; b++) {
          if (selected[b] || !discriminative[b])
            continue;
          double score = tk_incr_compute_ndcg(&incr_state, b, false, active->n);
          if (score > local_best_score) {
            local_best_score = score;
            local_best_bit = (int64_t)b;
          }
        }

        #pragma omp critical
        {
          if (local_best_score > best_add_score) {
            best_add_score = local_best_score;
            best_add_bit = local_best_bit;
          }
        }
      }

      if (best_add_bit >= 0) {
        double gain = (active->n == 0) ? best_add_score : (best_add_score - current_score);
        bool should_add = (gain > 0.0) || (active->n < min_dims);

        if (should_add) {
          tk_incr_compute_ndcg(&incr_state, (uint64_t)best_add_bit, true, active->n);
          selected[best_add_bit] = 1;
          if (tk_ivec_push(active, best_add_bit) != 0)
            luaL_error(L, "optimize_bits: allocation failed");
          current_score = best_add_score;
          changed = true;

          if (i_each >= 0) {
            lua_pushvalue(L, i_each);
            lua_pushinteger(L, best_add_bit);
            lua_pushnumber(L, gain);
            lua_pushnumber(L, current_score);
            lua_pushliteral(L, "add");
            lua_call(L, 4, 0);
          }
        }
      }
    }

    if (active->n > 1 && active->n > min_dims) {
      int64_t best_remove_bit = -1;
      double best_remove_score = -INFINITY;

      #pragma omp parallel
      {
        int64_t local_best_bit = -1;
        double local_best_score = -INFINITY;

        #pragma omp for schedule(dynamic)
        for (uint64_t b = 0; b < n_dims; b++) {
          if (!selected[b])
            continue;
          double score = tk_incr_compute_ndcg_sub(&incr_state, b, false, active->n);
          if (score > local_best_score) {
            local_best_score = score;
            local_best_bit = (int64_t)b;
          }
        }

        #pragma omp critical
        {
          if (local_best_score > best_remove_score) {
            best_remove_score = local_best_score;
            best_remove_bit = local_best_bit;
          }
        }
      }

      if (best_remove_bit >= 0 && best_remove_score > current_score) {
        double gain = best_remove_score - current_score;

        tk_incr_compute_ndcg_sub(&incr_state, (uint64_t)best_remove_bit, true, active->n);
        selected[best_remove_bit] = 0;
        current_score = best_remove_score;
        changed = true;

        if (i_each >= 0) {
          lua_pushvalue(L, i_each);
          lua_pushinteger(L, best_remove_bit);
          lua_pushnumber(L, gain);
          lua_pushnumber(L, current_score);
          lua_pushliteral(L, "remove");
          lua_call(L, 4, 0);
        }
      }
    }

    if (!changed)
      break;
  }

  active->n = 0;
  for (uint64_t b = 0; b < n_dims; b++) {
    if (selected[b]) {
      if (tk_ivec_push(active, (int64_t)b) != 0)
        luaL_error(L, "optimize_bits: allocation failed");
    }
  }

  free(xor_codes);
  free(distances);
  free(selected);
  free(discriminative);

  tk_ivec_asc(active, 0, active->n);
}

static void tk_incr_commit_cont (
  double *raw_codes,
  int64_t *pra,
  int64_t *prb,
  double *distances,
  uint64_t n_pairs,
  uint64_t n_dims,
  uint64_t dim,
  double sign
) {
  for (uint64_t j = 0; j < n_pairs; j++) {
    double da = raw_codes[(uint64_t)pra[j] * n_dims + dim]
              - raw_codes[(uint64_t)prb[j] * n_dims + dim];
    distances[j] += sign * da * da;
  }
}

static inline double tk_rank_dcg (
  const double *restrict nd,
  uint64_t nn,
  const uint64_t *restrict pos_idx,
  uint64_t n_pos,
  const double *restrict weights,
  const double *restrict discount
) {
  double dcg = 0.0;
  for (uint64_t pi = 0; pi < n_pos; pi++) {
    double dp = nd[pos_idx[pi]];
    double rank_d = 0.0;
    #pragma omp simd reduction(+:rank_d)
    for (uint64_t q = 0; q < nn; q++)
      rank_d += (nd[q] < dp) ? 1.0 : 0.0;
    dcg += weights[pos_idx[pi]] * discount[(uint64_t)rank_d];
  }
  return dcg;
}

static void tm_optimize_bits_sfbs_cont (
  lua_State *L,
  double *raw_codes,
  tk_ivec_t *code_ids,
  tk_ivec_t *adjacency_ids,
  tk_ivec_t *offsets,
  tk_ivec_t *neighbors,
  tk_dvec_t *weights,
  uint64_t n_dims,
  uint64_t target_dims,
  uint64_t min_dims,
  int i_each
) {
  uint64_t n_nodes = offsets->n - 1;
  uint64_t n_pairs = neighbors->n;

  int64_t max_uid = -1;
  for (uint64_t i = 0; i < code_ids->n; i++)
    if (code_ids->a[i] > max_uid) max_uid = code_ids->a[i];

  int64_t *uid_to_row = (int64_t *)malloc((uint64_t)(max_uid + 1) * sizeof(int64_t));
  memset(uid_to_row, -1, (uint64_t)(max_uid + 1) * sizeof(int64_t));
  for (uint64_t i = 0; i < code_ids->n; i++)
    uid_to_row[code_ids->a[i]] = (int64_t)i;

  int64_t sentinel_row = 0;
  int64_t *pair_row_a = (int64_t *)malloc(n_pairs * sizeof(int64_t));
  int64_t *pair_row_b = (int64_t *)malloc(n_pairs * sizeof(int64_t));
  for (uint64_t j = 0; j < n_pairs; j++) {
    pair_row_a[j] = sentinel_row;
    pair_row_b[j] = sentinel_row;
  }
  double *distances = (double *)calloc(n_pairs, sizeof(double));
  uint64_t max_neighbors = 0;

  for (uint64_t ni = 0; ni < n_nodes; ni++) {
    int64_t st = offsets->a[ni];
    int64_t en = offsets->a[ni + 1];
    uint64_t cnt = (uint64_t)(en - st);
    if (cnt > max_neighbors) max_neighbors = cnt;

    int64_t node_uid = adjacency_ids->a[ni];
    if (node_uid < 0 || node_uid > max_uid) continue;
    int64_t node_row = uid_to_row[node_uid];
    if (node_row < 0) continue;

    for (int64_t j = st; j < en; j++) {
      pair_row_a[j] = node_row;
      int64_t nbr_pos = neighbors->a[j];
      int64_t nbr_uid = adjacency_ids->a[nbr_pos];
      if (nbr_uid < 0 || nbr_uid > max_uid) continue;
      int64_t nbr_row = uid_to_row[nbr_uid];
      if (nbr_row < 0) continue;
      pair_row_b[j] = nbr_row;
    }
  }

  free(uid_to_row);

  double *discount = (double *)malloc(max_neighbors * sizeof(double));
  for (uint64_t i = 0; i < max_neighbors; i++)
    discount[i] = 1.0 / log2((double)(i + 2));

  double *idcg = (double *)calloc(n_nodes, sizeof(double));
  {
    double *iw = (double *)malloc(max_neighbors * sizeof(double));
    for (uint64_t ni = 0; ni < n_nodes; ni++) {
      int64_t st = offsets->a[ni];
      int64_t en = offsets->a[ni + 1];
      uint64_t nn = (uint64_t)(en - st);
      if (nn == 0) continue;
      for (uint64_t i = 0; i < nn; i++)
        iw[i] = weights->a[st + (int64_t)i];
      for (uint64_t i = 0; i < nn; i++)
        for (uint64_t k = i + 1; k < nn; k++)
          if (iw[k] > iw[i]) { double t = iw[i]; iw[i] = iw[k]; iw[k] = t; }
      for (uint64_t i = 0; i < nn; i++)
        idcg[ni] += iw[i] * discount[i];
    }
    free(iw);
  }

  double *scores = (double *)calloc(n_dims, sizeof(double));
  int64_t *offs_a = offsets->a;
  double *weights_a = weights->a;

  tk_ivec_t *active = tk_ivec_create(L, n_dims, 0, 0);
  active->n = 0;
  uint8_t *selected = (uint8_t *)calloc(n_dims, 1);

  double current_score = 0.0;
  uint64_t max_bits = target_dims > 0 ? target_dims : n_dims;

  int phase = 0;
  double eval_sign = 1.0;
  int64_t swap_remove_bit = -1;
  double pre_swap_score = 0.0;

  int n_threads;
  double **all_local_scores;
  #pragma omp parallel
  {
    #pragma omp single
    {
      n_threads = omp_get_num_threads();
      all_local_scores = (double **)malloc((size_t)n_threads * sizeof(double *));
    }
    int tid = omp_get_thread_num();
    double *local_scores = (double *)calloc(n_dims, sizeof(double));
    all_local_scores[tid] = local_scores;
    double *nd = (double *)malloc(max_neighbors * sizeof(double));
    uint64_t *pos_idx = (uint64_t *)malloc(max_neighbors * sizeof(uint64_t));
    double **av_ptrs = (double **)malloc(max_neighbors * sizeof(double *));
    double **bv_ptrs = (double **)malloc(max_neighbors * sizeof(double *));

    while (phase < 3) {
      memset(local_scores, 0, n_dims * sizeof(double));
      #pragma omp barrier

      #pragma omp for schedule(dynamic) nowait
      for (uint64_t ni = 0; ni < n_nodes; ni++) {
        int64_t st = offs_a[ni];
        int64_t en = offs_a[ni + 1];
        uint64_t nn = (uint64_t)(en - st);
        if (nn == 0 || idcg[ni] <= 0.0) continue;

        for (uint64_t i = 0; i < nn; i++) {
          uint64_t j = (uint64_t)st + i;
          av_ptrs[i] = raw_codes + (uint64_t)pair_row_a[j] * n_dims;
          bv_ptrs[i] = raw_codes + (uint64_t)pair_row_b[j] * n_dims;
        }

        double inv_idcg = 1.0 / idcg[ni];

        uint64_t n_pos = 0;
        for (uint64_t i = 0; i < nn; i++)
          if (weights_a[st + (int64_t)i] > 0.0)
            pos_idx[n_pos++] = i;

        for (uint64_t b = 0; b < n_dims; b++) {
          if (phase == 1 ? !selected[b] : selected[b]) continue;

          for (uint64_t i = 0; i < nn; i++) {
            double d = av_ptrs[i][b] - bv_ptrs[i][b];
            nd[i] = distances[(uint64_t)st + i] + eval_sign * d * d;
          }

          double dcg = tk_rank_dcg(nd, nn, pos_idx, n_pos,
            weights_a + st, discount);

          local_scores[b] += dcg * inv_idcg;
        }
      }

      #pragma omp barrier

      #pragma omp for schedule(static)
      for (uint64_t b = 0; b < n_dims; b++) {
        double s = 0.0;
        for (int t = 0; t < n_threads; t++)
          s += all_local_scores[t][b];
        scores[b] = s;
      }

      #pragma omp single
      {
        int64_t best_bit = -1;
        double best_score = -INFINITY;
        double tie_eps = (phase == 1) ? -1e-8 : 1e-8;
        for (uint64_t b = 0; b < n_dims; b++) {
          if (phase == 1 ? !selected[b] : selected[b]) continue;
          if (scores[b] > best_score + tie_eps) { best_score = scores[b]; best_bit = (int64_t)b; }
        }
        best_score = n_nodes > 0 ? best_score / (double)n_nodes : 0.0;

        switch (phase) {
        case 0: {
          bool added = false;
          if (best_bit >= 0 && active->n < max_bits) {
            double gain = (active->n == 0) ? best_score : (best_score - current_score);
            if (gain > 1e-8 || active->n < min_dims) {
              tk_incr_commit_cont(raw_codes, pair_row_a, pair_row_b,
                distances, n_pairs, n_dims, (uint64_t)best_bit, 1.0);
              selected[best_bit] = 1;
              tk_ivec_push(active, best_bit);
              current_score = best_score;
              added = true;
              if (i_each >= 0) {
                lua_pushvalue(L, i_each);
                lua_pushinteger(L, best_bit);
                lua_pushnumber(L, gain);
                lua_pushnumber(L, current_score);
                lua_pushliteral(L, "add");
                lua_call(L, 4, 0);
              }
            }
          }
          if (!added) {
            if (active->n > min_dims) {
              phase = 1;
              eval_sign = -1.0;
            } else {
              phase = 3;
            }
          }
          break;
        }
        case 1: {
          if (best_bit >= 0) {
            double gain = best_score - current_score;
            if (gain > -1e-8) {
              tk_incr_commit_cont(raw_codes, pair_row_a, pair_row_b,
                distances, n_pairs, n_dims, (uint64_t)best_bit, -1.0);
              selected[best_bit] = 0;
              for (uint64_t i = 0; i < (uint64_t)active->n; i++) {
                if (active->a[i] == best_bit) {
                  active->a[i] = active->a[active->n - 1];
                  active->n--;
                  break;
                }
              }
              current_score = best_score;
              phase = 0;
              eval_sign = 1.0;
              if (i_each >= 0) {
                lua_pushvalue(L, i_each);
                lua_pushinteger(L, best_bit);
                lua_pushnumber(L, gain);
                lua_pushnumber(L, current_score);
                lua_pushliteral(L, "remove");
                lua_call(L, 4, 0);
              }
            } else {
              swap_remove_bit = best_bit;
              pre_swap_score = current_score;
              tk_incr_commit_cont(raw_codes, pair_row_a, pair_row_b,
                distances, n_pairs, n_dims, (uint64_t)best_bit, -1.0);
              selected[best_bit] = 0;
              for (uint64_t i = 0; i < (uint64_t)active->n; i++) {
                if (active->a[i] == best_bit) {
                  active->a[i] = active->a[active->n - 1];
                  active->n--;
                  break;
                }
              }
              current_score = best_score;
              phase = 2;
              eval_sign = 1.0;
            }
          } else {
            phase = 3;
          }
          break;
        }
        case 2: {
          if (best_bit >= 0 && best_score > pre_swap_score + 1e-8) {
            tk_incr_commit_cont(raw_codes, pair_row_a, pair_row_b,
              distances, n_pairs, n_dims, (uint64_t)best_bit, 1.0);
            selected[best_bit] = 1;
            tk_ivec_push(active, best_bit);
            current_score = best_score;
            phase = 0;
            eval_sign = 1.0;
            if (i_each >= 0) {
              lua_pushvalue(L, i_each);
              lua_pushinteger(L, swap_remove_bit);
              lua_pushnumber(L, 0);
              lua_pushnumber(L, current_score);
              lua_pushliteral(L, "swap_remove");
              lua_call(L, 4, 0);
              lua_pushvalue(L, i_each);
              lua_pushinteger(L, best_bit);
              lua_pushnumber(L, best_score - pre_swap_score);
              lua_pushnumber(L, current_score);
              lua_pushliteral(L, "swap_add");
              lua_call(L, 4, 0);
            }
          } else {
            tk_incr_commit_cont(raw_codes, pair_row_a, pair_row_b,
              distances, n_pairs, n_dims, (uint64_t)swap_remove_bit, 1.0);
            selected[swap_remove_bit] = 1;
            tk_ivec_push(active, swap_remove_bit);
            current_score = pre_swap_score;
            phase = 3;
          }
          break;
        }
        }
      }
    }

    free(local_scores);
    free(nd);
    free(pos_idx);
    free(av_ptrs);
    free(bv_ptrs);
  }

  free(all_local_scores);

  active->n = 0;
  for (uint64_t b = 0; b < n_dims; b++)
    if (selected[b])
      tk_ivec_push(active, (int64_t)b);

  free(scores);
  free(pair_row_a);
  free(pair_row_b);
  free(distances);
  free(selected);
  free(discount);
  free(idcg);

  tk_ivec_asc(active, 0, active->n);
}

static inline int tm_optimize_bits (lua_State *L)
{
  lua_settop(L, 1);

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "optimize_bits", "n_dims");
  uint64_t target_dims = tk_lua_foptunsigned(L, 1, "optimize_bits", "target_dims", 0);
  uint64_t min_dims = tk_lua_foptunsigned(L, 1, "optimize_bits", "min_dims", 0);

  lua_getfield(L, 1, "expected_ids");
  tk_ivec_t *expected_ids = tk_ivec_peek(L, -1, "expected_ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *offsets = tk_ivec_peek(L, -1, "expected_offsets");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *neighbors = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 1);

  lua_getfield(L, 1, "expected_weights");
  tk_dvec_t *weights = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  if (!weights)
    tk_lua_verror(L, 1, "optimize_bits", "expected_weights", "required for bit optimization");
  if (weights->n != neighbors->n)
    luaL_error(L, "optimize_bits: expected_weights length (%d) must match expected_neighbors length (%d)",
      (int)weights->n, (int)neighbors->n);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  lua_getfield(L, 1, "raw_codes");
  tk_dvec_t *raw_codes_dvec = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  if (raw_codes_dvec) {
    lua_getfield(L, 1, "ids");
    tk_ivec_t *code_ids = tk_ivec_peek(L, -1, "ids");
    lua_pop(L, 1);
    tm_optimize_bits_sfbs_cont(L, raw_codes_dvec->a, code_ids,
      expected_ids, offsets, neighbors, weights,
      n_dims, target_dims, min_dims, i_each);
  } else {
    char *codes = NULL;
    tk_ann_t *ann = NULL;

    lua_getfield(L, 1, "codes");
    tk_cvec_t *cvec = tk_cvec_peekopt(L, -1);
    lua_pop(L, 1);
    if (cvec) {
      codes = cvec->a;
    } else {
      lua_getfield(L, 1, "index");
      ann = tk_ann_peekopt(L, -1);
      lua_pop(L, 1);
      if (!ann)
        tk_lua_verror(L, 1, "optimize_bits", "codes/index/raw_codes", "codes, index, or raw_codes required");
    }

    tk_eval_t state;
    memset(&state, 0, sizeof(tk_eval_t));
    state.codes = codes;
    state.ann = ann;
    state.adjacency_ids = expected_ids;
    state.n_dims = n_dims;
    state.chunks = TK_CVEC_BITS_BYTES(n_dims);
    state.offsets = offsets;
    state.neighbors = neighbors;
    state.weights = weights;
    state.target_dims = target_dims;
    state.min_dims = min_dims;
    state.L = L;

    tm_optimize_bits_sfbs(L, &state, i_each);
  }

  lua_replace(L, 1);
  lua_settop(L, 1);
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

static int tm_regression_per_dim (lua_State *L)
{
  lua_settop(L, 4);
  tk_dvec_t *predicted = tk_dvec_peek(L, 1, "predicted");
  tk_dvec_t *expected = tk_dvec_peek(L, 2, "expected");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_dims = tk_lua_checkunsigned(L, 4, "n_dims");
  if (predicted->n != n_samples * n_dims || expected->n != n_samples * n_dims)
    return luaL_error(L, "size mismatch: predicted=%llu expected=%llu, want %llu*%llu=%llu",
      (unsigned long long)predicted->n, (unsigned long long)expected->n,
      (unsigned long long)n_samples, (unsigned long long)n_dims,
      (unsigned long long)(n_samples * n_dims));
  tk_dvec_t *mae = tk_dvec_create(L, n_dims, 0, 0);
  tk_dvec_t *corr = tk_dvec_create(L, n_dims, 0, 0);
  tk_dvec_t *var_ratio = tk_dvec_create(L, n_dims, 0, 0);
  const double *p = predicted->a;
  const double *e = expected->a;
  double inv_n = 1.0 / (double)n_samples;
  #pragma omp parallel for schedule(static)
  for (uint64_t d = 0; d < n_dims; d++) {
    double sum_ae = 0, sum_p = 0, sum_e = 0;
    double sum_pp = 0, sum_ee = 0, sum_pe = 0;
    for (uint64_t i = 0; i < n_samples; i++) {
      double pv = p[i * n_dims + d];
      double ev = e[i * n_dims + d];
      sum_ae += fabs(pv - ev);
      sum_p += pv;
      sum_e += ev;
      sum_pp += pv * pv;
      sum_ee += ev * ev;
      sum_pe += pv * ev;
    }
    mae->a[d] = sum_ae * inv_n;
    double mp = sum_p * inv_n, me = sum_e * inv_n;
    double vp = sum_pp * inv_n - mp * mp;
    double ve = sum_ee * inv_n - me * me;
    double cov = sum_pe * inv_n - mp * me;
    corr->a[d] = (vp > 1e-30 && ve > 1e-30) ? cov / sqrt(vp * ve) : 0.0;
    var_ratio->a[d] = ve > 1e-30 ? vp / ve : 0.0;
  }
  int mae_idx = lua_gettop(L) - 2;
  lua_newtable(L);
  lua_pushvalue(L, mae_idx);
  lua_setfield(L, -2, "mae");
  lua_pushvalue(L, mae_idx + 1);
  lua_setfield(L, -2, "corr");
  lua_pushvalue(L, mae_idx + 2);
  lua_setfield(L, -2, "var_ratio");
  return 1;
}

static luaL_Reg tm_evaluator_fns[] =
{
  { "class_accuracy", tm_class_accuracy },
  { "encoding_accuracy", tm_encoding_accuracy },
  { "regression_accuracy", tm_regression_accuracy },
  { "regression_per_dim", tm_regression_per_dim },
  { "retrieval_accuracy", tm_multilabel_retrieval_accuracy },
  { "retrieval_ks", tm_retrieval_ks },
  { "optimize_bits", tm_optimize_bits },
  { "ranking_accuracy", tm_score_retrieval },
  { "cluster", tm_cluster },
  { "entropy_stats", tm_entropy_stats },
  { "dendro_cut", tk_pvec_dendro_cut_lua },
  { "dendro_each", tk_dendro_iter_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_evaluator (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_evaluator_fns, 0);
  return 1;
}
