#include <santoku/lua/utils.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/iumap.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include <omp.h>

#define TK_ITQ_ENCODER_MT "tk_itq_encoder_t"

typedef struct {
  tk_dvec_t *eigenvalues;
  tk_dvec_t *dim_scores;
  tk_ivec_t *selected_dims;
  tk_dvec_t *rotation;
  tk_dvec_t *means;
  uint64_t n_dims;
  uint64_t k;
  double alpha, beta;
  bool destroyed;
} tk_itq_encoder_t;

static inline tk_itq_encoder_t *tk_itq_encoder_peek (lua_State *L, int i) {
  return (tk_itq_encoder_t *)luaL_checkudata(L, i, TK_ITQ_ENCODER_MT);
}

static inline int tk_itq_encoder_gc (lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  enc->eigenvalues = NULL;
  enc->dim_scores = NULL;
  enc->selected_dims = NULL;
  enc->rotation = NULL;
  enc->means = NULL;
  enc->destroyed = true;
  return 0;
}

static inline void tk_itq_select_top_k (
  const double *eigenvalues, const double *dim_scores,
  uint64_t n_dims, uint64_t k, double alpha, double beta,
  int64_t *selected, double *weights
) {
  double *scores = (double *)malloc(n_dims * sizeof(double));
  int64_t *indices = (int64_t *)malloc(n_dims * sizeof(int64_t));
  for (uint64_t d = 0; d < n_dims; d++) {
    double s = 1.0;
    if (alpha != 0.0 && eigenvalues[d] > 0.0)
      s *= pow(eigenvalues[d], alpha);
    if (beta != 0.0 && dim_scores[d] > 0.0)
      s *= pow(dim_scores[d], beta);
    scores[d] = s;
    indices[d] = (int64_t)d;
  }
  for (uint64_t i = 0; i < k; i++) {
    uint64_t best = i;
    for (uint64_t j = i + 1; j < n_dims; j++) {
      if (scores[indices[j]] > scores[indices[best]])
        best = j;
    }
    if (best != i) {
      int64_t tmp = indices[i];
      indices[i] = indices[best];
      indices[best] = tmp;
    }
    selected[i] = indices[i];
    weights[i] = scores[indices[i]];
  }
  free(scores);
  free(indices);
}

static inline double *tk_itq_row_inv_norms (const double *data, uint64_t n_samples, uint64_t n_dims) {
  double *inv_norms = (double *)malloc(n_samples * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    const double *row = data + i * n_dims;
    double sum = 0.0;
    for (uint64_t j = 0; j < n_dims; j++)
      sum += row[j] * row[j];
    inv_norms[i] = (sum > 1e-30) ? 1.0 / sqrt(sum) : 0.0;
  }
  return inv_norms;
}

static inline void tk_itq_prepare_c (
  const double *raw, const double *inv_norms,
  uint64_t n_samples, uint64_t n_dims,
  const int64_t *selected, const double *weights, uint64_t k,
  double *out
) {
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    const double *src = raw + i * n_dims;
    double *dst = out + i * k;
    double s = inv_norms[i];
    for (uint64_t j = 0; j < k; j++)
      dst[j] = src[selected[j]] * s * weights[j];
  }
}

static inline void tk_itq_encode_c (
  const double *raw, const double *inv_norms,
  uint64_t n_samples, uint64_t n_dims,
  const int64_t *selected, const double *weights, uint64_t k,
  const double *rotation, const double *means,
  uint8_t *out, uint64_t n_bytes
) {
  double *prepared = (double *)malloc(n_samples * k * sizeof(double));
  tk_itq_prepare_c(raw, inv_norms, n_samples, n_dims, selected, weights, k, prepared);
  double *rotated = (double *)malloc(n_samples * k * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    double *p = prepared + i * k;
    for (uint64_t j = 0; j < k; j++)
      p[j] -= means[j];
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)n_samples, (int)k, (int)k, 1.0, prepared, (int)k,
    rotation, (int)k, 0.0, rotated, (int)k);
  memset(out, 0, n_samples * n_bytes);
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    double *r = rotated + i * k;
    uint8_t *dest = out + i * n_bytes;
    for (uint64_t j = 0; j < k; j++) {
      if (r[j] > 0.0)
        dest[TK_CVEC_BITS_BYTE(j)] |= (1 << TK_CVEC_BITS_BIT(j));
    }
  }
  free(prepared);
  free(rotated);
}

static int tk_itq_dim_scores_lua (lua_State *L) {
  lua_settop(L, 8);
  tk_dvec_t *raw_codes = tk_dvec_peek(L, 1, "raw_codes");
  tk_ivec_t *code_ids = tk_ivec_peek(L, 2, "ids");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_dims = tk_lua_checkunsigned(L, 4, "n_dims");
  tk_ivec_t *eval_ids = tk_ivec_peek(L, 5, "eval_ids");
  tk_ivec_t *eval_offsets = tk_ivec_peek(L, 6, "eval_offsets");
  tk_ivec_t *eval_neighbors = tk_ivec_peek(L, 7, "eval_neighbors");
  tk_dvec_t *eval_weights = tk_dvec_peek(L, 8, "eval_weights");

  tk_iumap_t *id_to_idx = tk_iumap_create(NULL, 0);
  for (uint64_t i = 0; i < n_samples; i++) {
    int kha;
    khint_t khi = tk_iumap_put(id_to_idx, code_ids->a[i], &kha);
    tk_iumap_setval(id_to_idx, khi, (int64_t)i);
  }

  double *inv_norms = tk_itq_row_inv_norms(raw_codes->a, n_samples, n_dims);

  uint64_t n_queries = eval_offsets->n - 1;
  tk_dvec_t *scores = tk_dvec_create(L, n_dims, 0, 0);
  memset(scores->a, 0, n_dims * sizeof(double));

  #pragma omp parallel
  {
    double *local_scores = (double *)calloc(n_dims, sizeof(double));
    double *sorted_weights = NULL;
    double *abs_diffs = NULL;
    int64_t *rank_indices = NULL;
    uint64_t buf_cap = 0;

    #pragma omp for schedule(dynamic, 4)
    for (uint64_t qi = 0; qi < n_queries; qi++) {
      int64_t q_id = eval_ids->a[qi];
      khint_t q_khi = tk_iumap_get(id_to_idx, q_id);
      if (q_khi == tk_iumap_end(id_to_idx))
        continue;
      int64_t q_idx = tk_iumap_val(id_to_idx, q_khi);

      int64_t e_start = eval_offsets->a[qi];
      int64_t e_end = eval_offsets->a[qi + 1];
      uint64_t m = (uint64_t)(e_end - e_start);
      if (m == 0) continue;

      if (m > buf_cap) {
        free(sorted_weights);
        free(abs_diffs);
        free(rank_indices);
        buf_cap = m;
        sorted_weights = (double *)malloc(m * sizeof(double));
        abs_diffs = (double *)malloc(m * sizeof(double));
        rank_indices = (int64_t *)malloc(m * sizeof(int64_t));
      }

      for (uint64_t j = 0; j < m; j++)
        sorted_weights[j] = eval_weights->a[(uint64_t)e_start + j];
      for (uint64_t i = 0; i < m; i++) {
        for (uint64_t j = i + 1; j < m; j++) {
          if (sorted_weights[j] > sorted_weights[i]) {
            double tmp = sorted_weights[i];
            sorted_weights[i] = sorted_weights[j];
            sorted_weights[j] = tmp;
          }
        }
      }
      double idcg = 0.0;
      for (uint64_t j = 0; j < m; j++)
        idcg += sorted_weights[j] / log2((double)(j + 2));
      if (idcg < 1e-10) continue;

      for (uint64_t d = 0; d < n_dims; d++) {
        double q_val = raw_codes->a[(uint64_t)q_idx * n_dims + d] * inv_norms[q_idx];
        for (uint64_t j = 0; j < m; j++) {
          int64_t n_eval_idx = eval_neighbors->a[(uint64_t)e_start + j];
          int64_t n_id = eval_ids->a[n_eval_idx];
          khint_t n_khi = tk_iumap_get(id_to_idx, n_id);
          if (n_khi == tk_iumap_end(id_to_idx)) {
            abs_diffs[j] = 1e30;
            rank_indices[j] = (int64_t)j;
            continue;
          }
          int64_t n_idx = tk_iumap_val(id_to_idx, n_khi);
          double n_val = raw_codes->a[(uint64_t)n_idx * n_dims + d] * inv_norms[n_idx];
          abs_diffs[j] = fabs(q_val - n_val);
          rank_indices[j] = (int64_t)j;
        }
        for (uint64_t i = 0; i < m; i++) {
          for (uint64_t j = i + 1; j < m; j++) {
            if (abs_diffs[rank_indices[j]] < abs_diffs[rank_indices[i]]) {
              int64_t tmp = rank_indices[i];
              rank_indices[i] = rank_indices[j];
              rank_indices[j] = tmp;
            }
          }
        }
        double dcg = 0.0;
        uint64_t pos = 0;
        uint64_t ri = 0;
        while (ri < m) {
          double cur_dist = abs_diffs[rank_indices[ri]];
          uint64_t tie_start = ri;
          while (ri < m && abs_diffs[rank_indices[ri]] == cur_dist)
            ri++;
          uint64_t tie_count = ri - tie_start;
          double discount_sum = 0.0;
          for (uint64_t t = 0; t < tie_count; t++)
            discount_sum += log2((double)(pos + t + 2));
          double avg_discount = discount_sum / (double)tie_count;
          for (uint64_t t = tie_start; t < ri; t++) {
            uint64_t orig_j = (uint64_t)rank_indices[t];
            double w = eval_weights->a[(uint64_t)e_start + orig_j];
            dcg += w / avg_discount;
          }
          pos += tie_count;
        }
        local_scores[d] += dcg / idcg;
      }
    }

    #pragma omp critical
    for (uint64_t d = 0; d < n_dims; d++)
      scores->a[d] += local_scores[d];

    free(local_scores);
    free(sorted_weights);
    free(abs_diffs);
    free(rank_indices);
  }

  for (uint64_t d = 0; d < n_dims; d++)
    scores->a[d] = n_queries > 0 ? scores->a[d] / (double)n_queries : 0.0;

  free(inv_norms);
  tk_iumap_destroy(id_to_idx);
  return 1;
}

static int tk_itq_prepare_lua (lua_State *L) {
  lua_settop(L, 8);
  tk_dvec_t *raw_codes = tk_dvec_peek(L, 1, "raw_codes");
  tk_dvec_t *eigenvalues = tk_dvec_peek(L, 2, "eigenvalues");
  tk_dvec_t *dim_scores = tk_dvec_peek(L, 3, "dim_scores");
  uint64_t n_samples = tk_lua_checkunsigned(L, 4, "n_samples");
  uint64_t n_dims = tk_lua_checkunsigned(L, 5, "n_dims");
  uint64_t k = tk_lua_checkunsigned(L, 6, "k");
  double alpha = luaL_checknumber(L, 7);
  double beta = luaL_checknumber(L, 8);
  if (k > n_dims) k = n_dims;
  int64_t *selected = (int64_t *)malloc(k * sizeof(int64_t));
  double *weights = (double *)malloc(k * sizeof(double));
  tk_itq_select_top_k(eigenvalues->a, dim_scores->a, n_dims, k, alpha, beta,
    selected, weights);
  double *inv_norms = tk_itq_row_inv_norms(raw_codes->a, n_samples, n_dims);
  tk_dvec_t *out = tk_dvec_create(L, n_samples * k, 0, 0);
  tk_itq_prepare_c(raw_codes->a, inv_norms, n_samples, n_dims, selected, weights, k, out->a);
  free(inv_norms);
  tk_ivec_t *sel_out = tk_ivec_create(L, k, 0, 0);
  memcpy(sel_out->a, selected, k * sizeof(int64_t));
  free(selected);
  free(weights);
  return 2;
}

static int tk_itq_train_lua (lua_State *L) {
  tk_dvec_t *weighted = tk_dvec_peek(L, 1, "weighted_codes");
  uint64_t n_samples = tk_lua_checkunsigned(L, 2, "n_samples");
  uint64_t k = tk_lua_checkunsigned(L, 3, "k");
  uint64_t iterations = lua_gettop(L) >= 4 ? tk_lua_checkunsigned(L, 4, "iterations") : 50;

  double *data = (double *)malloc(n_samples * k * sizeof(double));
  memcpy(data, weighted->a, n_samples * k * sizeof(double));

  double *means = (double *)malloc(k * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (uint64_t j = 0; j < k; j++) {
    double s = 0.0;
    for (uint64_t i = 0; i < n_samples; i++)
      s += data[i * k + j];
    means[j] = s / (double)n_samples;
    for (uint64_t i = 0; i < n_samples; i++)
      data[i * k + j] -= means[j];
  }

  double *W_half = (double *)malloc(k * k * sizeof(double));
  {
    double *cov = (double *)malloc(k * k * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)k, (int)k, (int)n_samples, 1.0 / (double)n_samples,
      data, (int)k, data, (int)k, 0.0, cov, (int)k);
    double *eig_vals = (double *)malloc(k * sizeof(double));
    int winfo = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', (int)k, cov, (int)k, eig_vals);
    if (winfo == 0) {
      for (uint64_t j = 0; j < k; j++) {
        double s = (eig_vals[j] > 1e-10) ? 1.0 / sqrt(eig_vals[j]) : 0.0;
        for (uint64_t i = 0; i < k; i++)
          cov[i * k + j] *= s;
      }
      memcpy(W_half, cov, k * k * sizeof(double));
      double *whitened = (double *)malloc(n_samples * k * sizeof(double));
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)n_samples, (int)k, (int)k, 1.0,
        data, (int)k, W_half, (int)k, 0.0, whitened, (int)k);
      memcpy(data, whitened, n_samples * k * sizeof(double));
      free(whitened);
    } else {
      for (uint64_t i = 0; i < k; i++)
        for (uint64_t j = 0; j < k; j++)
          W_half[i * k + j] = (i == j) ? 1.0 : 0.0;
    }
    free(eig_vals);
    free(cov);
  }

  double *R = (double *)malloc(k * k * sizeof(double));
  for (uint64_t i = 0; i < k; i++)
    for (uint64_t j = 0; j < k; j++)
      R[i * k + j] = (i == j) ? 1.0 : 0.0;

  double *projected = (double *)malloc(n_samples * k * sizeof(double));
  double *B = (double *)malloc(n_samples * k * sizeof(double));
  double *BtV = (double *)malloc(k * k * sizeof(double));
  double *U_svd = (double *)malloc(k * k * sizeof(double));
  double *S_svd = (double *)malloc(k * sizeof(double));
  double *Vt_svd = (double *)malloc(k * k * sizeof(double));
  double *superb = (double *)malloc(k * sizeof(double));

  for (uint64_t iter = 0; iter < iterations; iter++) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      (int)n_samples, (int)k, (int)k, 1.0, data, (int)k,
      R, (int)k, 0.0, projected, (int)k);

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples * k; i++)
      B[i] = projected[i] > 0.0 ? 1.0 : -1.0;

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)k, (int)k, (int)n_samples, 1.0, B, (int)k,
      data, (int)k, 0.0, BtV, (int)k);

    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A',
      (int)k, (int)k, BtV, (int)k, S_svd, U_svd, (int)k, Vt_svd, (int)k, superb);
    if (info != 0) break;

    double *newR = (double *)malloc(k * k * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
      (int)k, (int)k, (int)k, 1.0, Vt_svd, (int)k,
      U_svd, (int)k, 0.0, newR, (int)k);
    memcpy(R, newR, k * k * sizeof(double));
    free(newR);
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)n_samples, (int)k, (int)k, 1.0, data, (int)k,
    R, (int)k, 0.0, projected, (int)k);

  uint64_t n_bytes = TK_CVEC_BITS_BYTES(k);
  tk_cvec_t *codes_out = tk_cvec_create(L, n_samples * n_bytes, NULL, NULL);
  memset(codes_out->a, 0, codes_out->n);
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    double *r = projected + i * k;
    uint8_t *dest = (uint8_t *)codes_out->a + i * n_bytes;
    for (uint64_t j = 0; j < k; j++) {
      if (r[j] > 0.0)
        dest[TK_CVEC_BITS_BYTE(j)] |= (1 << TK_CVEC_BITS_BIT(j));
    }
  }

  double *R_combined = (double *)malloc(k * k * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)k, (int)k, (int)k, 1.0,
    W_half, (int)k, R, (int)k, 0.0, R_combined, (int)k);

  tk_dvec_t *rot_out = tk_dvec_create(L, k * k, 0, 0);
  memcpy(rot_out->a, R_combined, k * k * sizeof(double));

  tk_dvec_t *means_out = tk_dvec_create(L, k, 0, 0);
  memcpy(means_out->a, means, k * sizeof(double));

  free(R_combined);
  free(W_half);
  free(data);
  free(means);
  free(R);
  free(projected);
  free(B);
  free(BtV);
  free(U_svd);
  free(S_svd);
  free(Vt_svd);
  free(superb);

  return 3;
}

static int tk_itq_encode_lua (lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  tk_dvec_t *raw = tk_dvec_peek(L, 2, "raw_codes");
  uint64_t n_dims = enc->n_dims;
  uint64_t n_samples = raw->n / n_dims;
  uint64_t k = enc->k;
  uint64_t n_bytes = TK_CVEC_BITS_BYTES(k);
  double *weights = (double *)malloc(k * sizeof(double));
  for (uint64_t j = 0; j < k; j++) {
    double w = 1.0;
    int64_t d = enc->selected_dims->a[j];
    if (enc->alpha != 0.0 && enc->eigenvalues->a[d] > 0.0)
      w *= pow(enc->eigenvalues->a[d], enc->alpha);
    if (enc->beta != 0.0 && enc->dim_scores->a[d] > 0.0)
      w *= pow(enc->dim_scores->a[d], enc->beta);
    weights[j] = w;
  }
  double *inv_norms = tk_itq_row_inv_norms(raw->a, n_samples, n_dims);
  tk_cvec_t *out = tk_cvec_create(L, n_samples * n_bytes, NULL, NULL);
  tk_itq_encode_c(raw->a, inv_norms, n_samples, n_dims,
    enc->selected_dims->a, weights, k,
    enc->rotation->a, enc->means->a,
    (uint8_t *)out->a, n_bytes);
  free(inv_norms);
  free(weights);
  return 1;
}

static int tk_itq_n_bits_lua (lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->k);
  return 1;
}

static int tk_itq_n_dims_lua (lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->n_dims);
  return 1;
}

static int tk_itq_used_dims_lua (lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  tk_ivec_t *out = tk_ivec_create(L, enc->k, 0, 0);
  memcpy(out->a, enc->selected_dims->a, enc->k * sizeof(int64_t));
  return 1;
}

static int tk_itq_encoder_persist_lua (lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  if (enc->destroyed)
    return luaL_error(L, "cannot persist a destroyed encoder");
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_lua_fwrite(L, "TKiq", 1, 4, fh);
  uint8_t version = 2;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->n_dims, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->k, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, &enc->alpha, sizeof(double), 1, fh);
  tk_lua_fwrite(L, &enc->beta, sizeof(double), 1, fh);
  tk_dvec_persist(L, enc->eigenvalues, fh);
  tk_dvec_persist(L, enc->dim_scores, fh);
  tk_ivec_persist(L, enc->selected_dims, fh);
  tk_dvec_persist(L, enc->rotation, fh);
  tk_dvec_persist(L, enc->means, fh);
  if (!tostr) {
    tk_lua_fclose(L, fh);
    return 0;
  } else {
    size_t len;
    char *data = tk_lua_fslurp(L, fh, &len);
    if (data) {
      lua_pushlstring(L, data, len);
      free(data);
      tk_lua_fclose(L, fh);
      return 1;
    } else {
      tk_lua_fclose(L, fh);
      return 0;
    }
  }
}

static luaL_Reg tk_itq_encoder_mt_fns[] = {
  { "encode", tk_itq_encode_lua },
  { "n_bits", tk_itq_n_bits_lua },
  { "n_dims", tk_itq_n_dims_lua },
  { "used_dims", tk_itq_used_dims_lua },
  { "persist", tk_itq_encoder_persist_lua },
  { NULL, NULL }
};

static int tk_itq_create_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "eigenvalues");
  tk_dvec_t *eigenvalues = tk_dvec_peek(L, -1, "eigenvalues");
  int eig_idx = lua_gettop(L);

  lua_getfield(L, 1, "dim_scores");
  tk_dvec_t *dim_scores = tk_dvec_peek(L, -1, "dim_scores");
  int ds_idx = lua_gettop(L);

  lua_getfield(L, 1, "selected_dims");
  tk_ivec_t *selected_dims = tk_ivec_peek(L, -1, "selected_dims");
  int sel_idx = lua_gettop(L);

  lua_getfield(L, 1, "rotation");
  tk_dvec_t *rotation = tk_dvec_peek(L, -1, "rotation");
  int rot_idx = lua_gettop(L);

  lua_getfield(L, 1, "means");
  tk_dvec_t *means = tk_dvec_peek(L, -1, "means");
  int means_idx = lua_gettop(L);

  lua_getfield(L, 1, "n_dims");
  uint64_t n_dims = tk_lua_checkunsigned(L, -1, "n_dims");
  lua_pop(L, 1);

  lua_getfield(L, 1, "k");
  uint64_t k = tk_lua_checkunsigned(L, -1, "k");
  lua_pop(L, 1);

  lua_getfield(L, 1, "alpha");
  double alpha = luaL_checknumber(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "beta");
  double beta = luaL_checknumber(L, -1);
  lua_pop(L, 1);

  tk_itq_encoder_t *enc = (tk_itq_encoder_t *)tk_lua_newuserdata(L, tk_itq_encoder_t,
    TK_ITQ_ENCODER_MT, tk_itq_encoder_mt_fns, tk_itq_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->eigenvalues = eigenvalues;
  enc->dim_scores = dim_scores;
  enc->selected_dims = selected_dims;
  enc->rotation = rotation;
  enc->means = means;
  enc->n_dims = n_dims;
  enc->k = k;
  enc->alpha = alpha;
  enc->beta = beta;
  enc->destroyed = false;

  lua_newtable(L);
  lua_pushvalue(L, eig_idx);
  lua_setfield(L, -2, "eigenvalues");
  lua_pushvalue(L, ds_idx);
  lua_setfield(L, -2, "dim_scores");
  lua_pushvalue(L, sel_idx);
  lua_setfield(L, -2, "selected_dims");
  lua_pushvalue(L, rot_idx);
  lua_setfield(L, -2, "rotation");
  lua_pushvalue(L, means_idx);
  lua_setfield(L, -2, "means");
  lua_setfenv(L, enc_idx);

  lua_pushvalue(L, enc_idx);
  return 1;
}

static int tk_itq_load_lua (lua_State *L) {
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKiq", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid ITQ encoder file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 2) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported ITQ encoder version %d", (int)version);
  }
  uint64_t n_dims, k;
  double alpha, beta;
  tk_lua_fread(L, &n_dims, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &k, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &alpha, sizeof(double), 1, fh);
  tk_lua_fread(L, &beta, sizeof(double), 1, fh);
  tk_dvec_t *eigenvalues = tk_dvec_load(L, fh);
  int eig_idx = lua_gettop(L);
  tk_dvec_t *dim_scores = tk_dvec_load(L, fh);
  int ds_idx = lua_gettop(L);
  tk_ivec_t *selected_dims = tk_ivec_load(L, fh);
  int sel_idx = lua_gettop(L);
  tk_dvec_t *rotation = tk_dvec_load(L, fh);
  int rot_idx = lua_gettop(L);
  tk_dvec_t *means = tk_dvec_load(L, fh);
  int means_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);

  tk_itq_encoder_t *enc = (tk_itq_encoder_t *)tk_lua_newuserdata(L, tk_itq_encoder_t,
    TK_ITQ_ENCODER_MT, tk_itq_encoder_mt_fns, tk_itq_encoder_gc);
  int enc_idx = lua_gettop(L);
  enc->eigenvalues = eigenvalues;
  enc->dim_scores = dim_scores;
  enc->selected_dims = selected_dims;
  enc->rotation = rotation;
  enc->means = means;
  enc->n_dims = n_dims;
  enc->k = k;
  enc->alpha = alpha;
  enc->beta = beta;
  enc->destroyed = false;

  lua_newtable(L);
  lua_pushvalue(L, eig_idx);
  lua_setfield(L, -2, "eigenvalues");
  lua_pushvalue(L, ds_idx);
  lua_setfield(L, -2, "dim_scores");
  lua_pushvalue(L, sel_idx);
  lua_setfield(L, -2, "selected_dims");
  lua_pushvalue(L, rot_idx);
  lua_setfield(L, -2, "rotation");
  lua_pushvalue(L, means_idx);
  lua_setfield(L, -2, "means");
  lua_setfenv(L, enc_idx);

  lua_pushvalue(L, enc_idx);
  return 1;
}

static luaL_Reg tk_itq_fns[] = {
  { "dim_scores", tk_itq_dim_scores_lua },
  { "prepare", tk_itq_prepare_lua },
  { "train", tk_itq_train_lua },
  { "create", tk_itq_create_lua },
  { "load", tk_itq_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_itq (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_itq_fns, 0);
  return 1;
}
