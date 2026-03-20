#include <santoku/iuset.h>
#include <santoku/pumap.h>
#include <santoku/learn/centroid.h>
#include <santoku/learn/csr.h>
#include <santoku/ivec.h>
#include <santoku/fvec.h>
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

static inline int tm_regress_accuracy (lua_State *L)
{
  lua_settop(L, 2);
  tk_fvec_t *predicted_f = tk_fvec_peekopt(L, 1);
  tk_dvec_t *predicted_d = predicted_f ? NULL : tk_dvec_peek(L, 1, "predicted");
  tk_dvec_t *expected_d = tk_dvec_peekopt(L, 2);
  tk_ivec_t *expected_i = expected_d ? NULL : tk_ivec_peek(L, 2, "expected");
  uint64_t n = predicted_f ? predicted_f->n : predicted_d->n;
  if ((expected_d && expected_d->n != n) || (expected_i && expected_i->n != n))
    return luaL_error(L, "predicted and expected must have same length");
  double total = 0.0, min_err = DBL_MAX, max_err = 0.0, sum_exp = 0.0;
  #pragma omp parallel for reduction(+:total,sum_exp) reduction(min:min_err) reduction(max:max_err)
  for (uint64_t i = 0; i < n; i++) {
    double exp_val = expected_d ? expected_d->a[i] : (double)expected_i->a[i];
    double pred_val = predicted_f ? (double)predicted_f->a[i] : predicted_d->a[i];
    double err = fabs(pred_val - exp_val);
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
    double pred_val = predicted_f ? (double)predicted_f->a[i] : predicted_d->a[i];
    double err = fabs(pred_val - exp_val);
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

static inline tk_ivec_t *tk_pvec_dendro_cut(lua_State *L, tk_ivec_t *offsets, tk_pvec_t *merges, uint64_t step, tk_ivec_t *assignments);

static inline int tm_label_accuracy (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "pred_offsets");
  tk_ivec_t *pred_off = tk_ivec_peek(L, -1, "pred_offsets");
  lua_getfield(L, 1, "pred_neighbors");
  tk_ivec_t *pred_nbr = tk_ivec_peek(L, -1, "pred_neighbors");
  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_getfield(L, 1, "ks");
  bool have_ks = !lua_isnil(L, -1);
  bool scalar_ks = have_ks && lua_isnumber(L, -1);
  int64_t scalar_k_val = scalar_ks ? (int64_t)lua_tointeger(L, -1) : 0;
  tk_ivec_t *ks = (have_ks && !scalar_ks) ? tk_ivec_peek(L, -1, "ks") : NULL;
  lua_pop(L, 5);
  uint64_t n_samples = pred_off->n - 1;
  if (exp_off->n != n_samples + 1)
    return luaL_error(L, "expected_offsets length must match sample count + 1");
  if (scalar_ks) {
    ks = tk_ivec_create(L, n_samples);
    for (uint64_t i = 0; i < n_samples; i++) ks->a[i] = scalar_k_val;
    have_ks = true;
  } else if (!have_ks) {
    ks = tk_ivec_create(L, n_samples);
  }
  uint64_t mi_tp = 0, mi_k = 0, mi_exp = 0, n_valid = 0;
  double ma_prec = 0, ma_rec = 0, ma_f1 = 0;
  #pragma omp parallel for reduction(+:mi_tp,mi_k,mi_exp,n_valid,ma_prec,ma_rec,ma_f1)
  for (uint64_t s = 0; s < n_samples; s++) {
    int64_t ps = pred_off->a[s], pe = pred_off->a[s + 1];
    uint64_t hood_size = (uint64_t)(pe - ps);
    int64_t es = exp_off->a[s], ee = exp_off->a[s + 1];
    uint64_t n_expected = (uint64_t)(ee - es);
    if (n_expected == 0 || hood_size == 0) {
      if (!have_ks) ks->a[s] = 0;
      continue;
    }
    int kha;
    tk_iuset_t *exp_set = tk_iuset_create(NULL, 0);
    for (int64_t i = es; i < ee; i++)
      tk_iuset_put(exp_set, exp_nbr->a[i], &kha);
    uint64_t best_tp, best_k;
    double best_f1;
    if (have_ks) {
      int64_t sk = ks->a[s];
      if (sk < 1) sk = 1;
      best_k = (uint64_t)sk;
      if (best_k > hood_size) best_k = hood_size;
      best_tp = 0;
      for (uint64_t j = 0; j < best_k; j++)
        if (tk_iuset_contains(exp_set, pred_nbr->a[ps + (int64_t)j]) != 0) best_tp++;
      double prec = (double)best_tp / best_k;
      double rec = (double)best_tp / n_expected;
      best_f1 = (prec + rec > 0) ? 2.0 * prec * rec / (prec + rec) : 0;
    } else {
      double best_score = -1.0;
      best_f1 = -1.0;
      best_k = 1;
      best_tp = 0;
      uint64_t tp = 0;
      for (uint64_t k = 1; k <= hood_size; k++) {
        if (tk_iuset_contains(exp_set, pred_nbr->a[ps + (int64_t)(k - 1)]) != 0) tp++;
        double prec = (double)tp / k;
        double rec = (double)tp / n_expected;
        double f1 = (prec + rec) > 0 ? 2.0 * prec * rec / (prec + rec) : 0.0;
        double score = f1;
        if (score > best_score) {
          best_score = score;
          best_f1 = f1;
          best_k = k;
          best_tp = tp;
        }
      }
      ks->a[s] = (int64_t)best_k;
    }
    tk_iuset_destroy(exp_set);
    mi_tp += best_tp;
    mi_k += best_k;
    mi_exp += n_expected;
    ma_prec += (double)best_tp / best_k;
    ma_rec += (double)best_tp / n_expected;
    ma_f1 += best_f1;
    n_valid++;
  }
  if (!have_ks)
    lua_pushvalue(L, -1);
  else
    lua_pushnil(L);
  lua_newtable(L);
  double mi_prec = mi_k > 0 ? (double)mi_tp / mi_k : 0;
  double mi_rec = mi_exp > 0 ? (double)mi_tp / mi_exp : 0;
  double mi_f1v = (mi_prec + mi_rec) > 0 ? 2.0 * mi_prec * mi_rec / (mi_prec + mi_rec) : 0;
  lua_pushnumber(L, mi_prec); lua_setfield(L, -2, "micro_precision");
  lua_pushnumber(L, mi_rec); lua_setfield(L, -2, "micro_recall");
  lua_pushnumber(L, mi_f1v); lua_setfield(L, -2, "micro_f1");
  lua_pushnumber(L, n_valid > 0 ? ma_prec / n_valid : 0); lua_setfield(L, -2, "sample_precision");
  lua_pushnumber(L, n_valid > 0 ? ma_rec / n_valid : 0); lua_setfield(L, -2, "sample_recall");
  lua_pushnumber(L, n_valid > 0 ? ma_f1 / n_valid : 0); lua_setfield(L, -2, "sample_f1");
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

  tk_iumap_t *absorbed_to_surviving = tk_iumap_create(NULL, 0);

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

  tk_iumap_t *cluster_remap = tk_iumap_create(NULL, 0);

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

  tk_ivec_t *entity_to_cluster = tk_ivec_create(NULL, n_nodes);
  if (!entity_to_cluster) {
    free(clusters);
    return -1;
  }
  entity_to_cluster->n = n_nodes;

  tk_evec_t *edge_heap = tk_evec_create(NULL, 0);
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
      cluster->members = tk_ivec_create(NULL, 0);
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

    tk_evec_t *distance_level_edges = tk_evec_create(NULL, 0);
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

  result.dendro_offsets = tk_ivec_create(L, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.dendro_merges = tk_pvec_create(L, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.quality_curve = tk_dvec_create(L, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.n_clusters_curve = tk_ivec_create(L, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, i_eph, -1);
  lua_pop(L, 1);

  result.auc_curve = NULL;
  if (exp_off && exp_nbr && exp_wgt) {
    result.auc_curve = tk_dvec_create(L, 0);
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

static inline int tm_label_ranking (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "pred_offsets");
  tk_ivec_t *pred_off = tk_ivec_peek(L, -1, "pred_offsets");
  lua_getfield(L, 1, "pred_neighbors");
  tk_ivec_t *pred_nbr = tk_ivec_peek(L, -1, "pred_neighbors");
  lua_getfield(L, 1, "pred_scores");
  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_pop(L, 5);
  uint64_t n_samples = pred_off->n - 1;
  if (exp_off->n != n_samples + 1)
    return luaL_error(L, "expected_offsets length must match sample count + 1");
  double sum_ndcg = 0.0;
  uint64_t n_valid = 0;
  #pragma omp parallel for reduction(+:sum_ndcg,n_valid)
  for (uint64_t s = 0; s < n_samples; s++) {
    int64_t ps = pred_off->a[s], pe = pred_off->a[s + 1];
    int64_t pred_k = pe - ps;
    int64_t es = exp_off->a[s], ee = exp_off->a[s + 1];
    int64_t n_exp = ee - es;
    if (n_exp == 0 || pred_k == 0) continue;
    int kha;
    tk_iuset_t *exp_set = tk_iuset_create(NULL, 0);
    for (int64_t j = es; j < ee; j++)
      tk_iuset_put(exp_set, exp_nbr->a[j], &kha);
    double dcg = 0.0;
    for (int64_t j = 0; j < pred_k; j++)
      if (tk_iuset_contains(exp_set, pred_nbr->a[ps + j]))
        dcg += 1.0 / log2((double)(j + 2));
    tk_iuset_destroy(exp_set);
    int64_t ideal_n = n_exp < pred_k ? n_exp : pred_k;
    double idcg = 0.0;
    for (int64_t j = 0; j < ideal_n; j++)
      idcg += 1.0 / log2((double)(j + 2));
    if (idcg > 0.0)
      sum_ndcg += dcg / idcg;
    n_valid++;
  }
  lua_newtable(L);
  lua_pushnumber(L, n_valid > 0 ? sum_ndcg / (double)n_valid : 0.0);
  lua_setfield(L, -2, "ndcg");
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
    assignments = tk_ivec_create(L, 0);
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

  tk_ivec_t *cluster_offsets = tk_ivec_create(L, 0);
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

  tk_ivec_t *cluster_members = tk_ivec_create(L, 0);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, -1, -1);
  int i_members = tk_lua_absindex(L, -1);
  tk_ivec_ensure(cluster_members, n_samples);
  cluster_members->n = n_samples;

  tk_ivec_t *cluster_positions = tk_ivec_create(NULL, (uint64_t)n_clusters);
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
  tk_iumap_t *cluster_remap = tk_iumap_create(NULL, 0);
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
  iter->ids = tk_ivec_create(L, n_samples);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, iter_idx, -1);
  lua_pop(L, 1);
  iter->raw_assignments = tk_ivec_create(L, n_samples);
  tk_lua_add_ephemeron(L, TK_EVAL_EPH, iter_idx, -1);
  lua_pop(L, 1);
  iter->assignments = tk_ivec_create(L, n_samples);
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

static inline int tm_rp_at_k (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "pred_offsets");
  tk_ivec_t *pred_off = tk_ivec_peek(L, -1, "pred_offsets");
  lua_getfield(L, 1, "pred_neighbors");
  tk_ivec_t *pred_nbr = tk_ivec_peek(L, -1, "pred_neighbors");
  lua_getfield(L, 1, "expected_offsets");
  tk_ivec_t *exp_off = tk_ivec_peek(L, -1, "expected_offsets");
  lua_getfield(L, 1, "expected_neighbors");
  tk_ivec_t *exp_nbr = tk_ivec_peek(L, -1, "expected_neighbors");
  lua_getfield(L, 1, "max_k");
  int64_t max_k = luaL_optinteger(L, -1, 20);
  lua_pop(L, 5);
  uint64_t n_samples = pred_off->n - 1;
  double *sums = calloc((uint64_t)max_k, sizeof(double));
  uint64_t *counts = calloc((uint64_t)max_k, sizeof(uint64_t));
  #pragma omp parallel
  {
    double *ls = calloc((uint64_t)max_k, sizeof(double));
    uint64_t *lc = calloc((uint64_t)max_k, sizeof(uint64_t));
    #pragma omp for
    for (uint64_t s = 0; s < n_samples; s++) {
      int64_t ps = pred_off->a[s], pe = pred_off->a[s + 1];
      uint64_t hood = (uint64_t)(pe - ps);
      int64_t es = exp_off->a[s], ee = exp_off->a[s + 1];
      uint64_t n_exp = (uint64_t)(ee - es);
      if (n_exp == 0 || hood == 0) continue;
      int kha;
      tk_iuset_t *exp_set = tk_iuset_create(NULL, 0);
      for (int64_t i = es; i < ee; i++)
        tk_iuset_put(exp_set, exp_nbr->a[i], &kha);
      uint64_t tp = 0, tp_frozen = 0;
      for (int64_t ki = 0; ki < max_k; ki++) {
        uint64_t K = (uint64_t)(ki + 1);
        if (K > hood) break;
        if (tk_iuset_contains(exp_set, pred_nbr->a[ps + ki]) != 0)
          tp++;
        if (K <= n_exp) tp_frozen = tp;
        uint64_t eff_k = K < n_exp ? K : n_exp;
        ls[ki] += (double)(K <= n_exp ? tp : tp_frozen) / eff_k;
        lc[ki]++;
      }
      tk_iuset_destroy(exp_set);
    }
    #pragma omp critical
    for (int64_t ki = 0; ki < max_k; ki++) {
      sums[ki] += ls[ki];
      counts[ki] += lc[ki];
    }
    free(ls);
    free(lc);
  }
  lua_newtable(L);
  for (int64_t ki = 0; ki < max_k; ki++) {
    lua_pushnumber(L, counts[ki] > 0 ? sums[ki] / counts[ki] : 0);
    lua_rawseti(L, -2, ki + 1);
  }
  free(sums);
  free(counts);
  return 1;
}

static luaL_Reg tm_evaluator_fns[] =
{
  { "regress_accuracy", tm_regress_accuracy },
  { "label_accuracy", tm_label_accuracy },
  { "label_ranking", tm_label_ranking },
  { "rp_at_k", tm_rp_at_k },
  { "cluster", tm_cluster },
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
