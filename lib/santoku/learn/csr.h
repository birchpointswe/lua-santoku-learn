#ifndef TK_CSR_H
#define TK_CSR_H

#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/pumap.h>
#include <santoku/dumap.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/rvec.h>
#include <santoku/pvec.h>

static inline double tk_csr_pearson_distance(
  tk_ivec_t *expected_ids,
  tk_ivec_t *expected_neighbors,
  tk_dvec_t *expected_weights,
  int64_t exp_start,
  int64_t exp_end,
  tk_ivec_t *retrieved_ids,
  tk_pvec_t *bin_ranks,
  tk_dumap_t *rank_buffer_b
) {
  uint64_t m = (uint64_t)(exp_end - exp_start);
  if (m == 0 || !bin_ranks || bin_ranks->n == 0)
    return 0.0;
  tk_dumap_clear(rank_buffer_b);
  int kha;
  for (uint64_t i = 0; i < bin_ranks->n; i++) {
    int64_t neighbor_idx = bin_ranks->a[i].i;
    int64_t neighbor_id = retrieved_ids->a[neighbor_idx];
    double hamming = (double)bin_ranks->a[i].p;
    uint32_t khi = tk_dumap_put(rank_buffer_b, neighbor_id, &kha);
    tk_dumap_setval(rank_buffer_b, khi, hamming);
  }
  double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
  uint64_t n = 0;
  for (int64_t j = exp_start; j < exp_end; j++) {
    int64_t neighbor_idx = expected_neighbors->a[j];
    int64_t neighbor_id = expected_ids->a[neighbor_idx];
    uint32_t khi = tk_dumap_get(rank_buffer_b, neighbor_id);
    if (khi != tk_dumap_end(rank_buffer_b)) {
      double x = expected_weights->a[j];
      double y = tk_dumap_val(rank_buffer_b, khi);
      sum_x += x;
      sum_y += y;
      sum_xy += x * y;
      sum_x2 += x * x;
      sum_y2 += y * y;
      n++;
    }
  }
  if (n < 2)
    return 0.0;
  double n_d = (double)n;
  double numerator = n_d * sum_xy - sum_x * sum_y;
  double denom_x = n_d * sum_x2 - sum_x * sum_x;
  double denom_y = n_d * sum_y2 - sum_y * sum_y;
  if (denom_x < 1e-10 || denom_y < 1e-10)
    return 0.0;
  return numerator / sqrt(denom_x * denom_y);
}

static inline double tk_csr_spearman_distance(
  tk_ivec_t *expected_ids,
  tk_ivec_t *expected_neighbors,
  tk_dvec_t *expected_weights,
  int64_t exp_start,
  int64_t exp_end,
  tk_ivec_t *retrieved_ids,
  tk_pvec_t *sorted_bin_ranks,
  tk_rvec_t *weight_ranks_buffer,
  tk_dumap_t *weight_rank_map
) {
  uint64_t m = (uint64_t)(exp_end - exp_start);
  if (m == 0 || !sorted_bin_ranks || sorted_bin_ranks->n == 0)
    return 0.0;
  if (sorted_bin_ranks->n < 2)
    return 0.0;
  if (tk_rvec_ensure(weight_ranks_buffer, m) != 0)
    return 0.0;
  weight_ranks_buffer->n = m;
  uint64_t idx = 0;
  for (int64_t j = exp_start; j < exp_end; j++) {
    int64_t neighbor_idx = expected_neighbors->a[j];
    int64_t neighbor_id = expected_ids->a[neighbor_idx];
    double weight = expected_weights->a[j];
    weight_ranks_buffer->a[idx++] = (tk_rank_t) { neighbor_id, weight };
  }
  tk_rvec_desc(weight_ranks_buffer, 0, weight_ranks_buffer->n);
  tk_dumap_clear(weight_rank_map);
  for (uint64_t i = 0; i < weight_ranks_buffer->n; ) {
    double val = weight_ranks_buffer->a[i].d;
    uint64_t j = i;
    while (j < weight_ranks_buffer->n && weight_ranks_buffer->a[j].d == val)
      j++;
    double avg_rank = (double)(i + j - 1) / 2.0;
    for (uint64_t k = i; k < j; k++) {
      int64_t neighbor_id = weight_ranks_buffer->a[k].i;
      int kha;
      uint32_t khi = tk_dumap_put(weight_rank_map, neighbor_id, &kha);
      tk_dumap_setval(weight_rank_map, khi, avg_rank);
    }
    i = j;
  }
  double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
  uint64_t n = 0;
  for (uint64_t i = 0; i < sorted_bin_ranks->n; i++) {
    int64_t neighbor_idx = sorted_bin_ranks->a[i].i;
    int64_t neighbor_id = retrieved_ids->a[neighbor_idx];
    uint32_t khi = tk_dumap_get(weight_rank_map, neighbor_id);
    if (khi != tk_dumap_end(weight_rank_map)) {
      double x_rank = tk_dumap_val(weight_rank_map, khi);
      double y_rank = (double)i;
      sum_x += x_rank;
      sum_y += y_rank;
      sum_xy += x_rank * y_rank;
      sum_x2 += x_rank * x_rank;
      sum_y2 += y_rank * y_rank;
      n++;
    }
  }
  if (n < 2)
    return 0.0;
  double n_d = (double)n;
  double numerator = n_d * sum_xy - sum_x * sum_y;
  double denom_x = n_d * sum_x2 - sum_x * sum_x;
  double denom_y = n_d * sum_y2 - sum_y * sum_y;
  if (denom_x < 1e-10 || denom_y < 1e-10)
    return 0.0;
  return numerator / sqrt(denom_x * denom_y);
}

static inline double tk_csr_ndcg_distance(
  tk_ivec_t *expected_ids,
  tk_ivec_t *expected_neighbors,
  tk_dvec_t *expected_weights,
  int64_t exp_start,
  int64_t exp_end,
  tk_ivec_t *retrieved_ids,
  tk_pvec_t *sorted_bin_ranks,
  tk_dumap_t *weight_map
) {
  uint64_t m = (uint64_t)(exp_end - exp_start);
  if (m == 0 || !sorted_bin_ranks || sorted_bin_ranks->n == 0)
    return 0.0;

  tk_dumap_clear(weight_map);
  int kha;
  for (int64_t j = exp_start; j < exp_end; j++) {
    int64_t neighbor_idx = expected_neighbors->a[j];
    int64_t neighbor_id = expected_ids->a[neighbor_idx];
    double weight = expected_weights->a[j];
    uint32_t khi = tk_dumap_put(weight_map, neighbor_id, &kha);
    tk_dumap_setval(weight_map, khi, weight);
  }

  double dcg = 0.0;
  uint64_t n = sorted_bin_ranks->n;
  uint64_t i = 0;
  while (i < n) {
    int64_t current_hamming = sorted_bin_ranks->a[i].p;
    uint64_t tie_start = i;
    while (i < n && sorted_bin_ranks->a[i].p == current_hamming)
      i++;
    uint64_t tie_end = i;
    uint64_t tie_count = tie_end - tie_start;
    double discount_sum = 0.0;
    for (uint64_t pos = tie_start; pos < tie_end; pos++)
      discount_sum += log2((double)(pos + 2));
    double avg_discount = discount_sum / (double)tie_count;
    for (uint64_t j = tie_start; j < tie_end; j++) {
      int64_t neighbor_idx = sorted_bin_ranks->a[j].i;
      int64_t neighbor_id = retrieved_ids->a[neighbor_idx];
      uint32_t khi = tk_dumap_get(weight_map, neighbor_id);
      if (khi != tk_dumap_end(weight_map)) {
        double relevance = tk_dumap_val(weight_map, khi);
        dcg += relevance / avg_discount;
      }
    }
  }

  uint64_t k = n;
  tk_dvec_t *sorted_weights = tk_dvec_create(NULL, m);
  if (!sorted_weights)
    return 0.0;
  for (i = 0; i < m; i++)
    sorted_weights->a[i] = expected_weights->a[(uint64_t)exp_start + i];
  tk_dvec_desc(sorted_weights, 0, m);

  double idcg = 0.0;
  uint64_t idcg_count = (m < k) ? m : k;
  for (i = 0; i < idcg_count; i++) {
    double relevance = sorted_weights->a[i];
    double discount = log2((double)(i + 2));
    idcg += relevance / discount;
  }

  tk_dvec_destroy(sorted_weights);

  if (idcg < 1e-10)
    return 0.0;

  return dcg / idcg;
}

#endif
