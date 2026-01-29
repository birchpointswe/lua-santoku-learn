#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <cblas.h>
#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/dvec.h>
#include <santoku/pvec.h>
#include <santoku/rvec.h>
#include <santoku/iumap.h>
#include <santoku/iumap/ext.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/ann.h>
#include <santoku/tsetlin/graph.h>

#define TK_HLTH_ENCODER_MT "tk_hlth_encoder_t"
#define TK_HLTH_ENCODER_EPH "tk_hlth_encoder_eph"
#define TK_NYSTROM_ENCODER_MT "tk_nystrom_encoder_t"
#define TK_NYSTROM_ENCODER_EPH "tk_nystrom_encoder_eph"

typedef enum {
  TK_HLTH_IDX_INV,
  TK_HLTH_IDX_ANN
} tk_hlth_index_type_t;

typedef enum {
  TK_HLTH_MODE_CONCAT,
  TK_HLTH_MODE_FREQUENCY,
  TK_HLTH_MODE_FREQUENCY_WEIGHTED,
  TK_HLTH_MODE_CENTROID,
  TK_HLTH_MODE_CENTROID_WEIGHTED,
  TK_HLTH_MODE_SIMILARITIES
} tk_hlth_mode_t;

typedef struct {
  tk_hlth_index_type_t feat_idx_type;
  tk_hlth_index_type_t code_idx_type;
  void *feat_idx;
  void *code_idx;
  uint64_t n_landmarks;
  uint64_t n_hidden;
  uint64_t probe_radius;
  tk_ivec_sim_type_t cmp;
  double cmp_alpha;
  double cmp_beta;
  tk_hlth_mode_t mode;
  uint64_t n_thresholds;
  uint64_t n_bins;
  bool quantile;
  bool concat_query;
  bool destroyed;
} tk_hlth_encoder_t;

typedef enum {
  TK_BINARIZE_ITQ,
  TK_BINARIZE_SIGN,
  TK_BINARIZE_MEDIAN
} tk_binarize_type_t;

typedef struct {
  tk_hlth_index_type_t feat_idx_type;
  void *feat_idx;
  tk_dvec_t *eigenvectors;
  tk_dvec_t *eigenvalues;
  tk_dvec_t *col_means;
  tk_dvec_t *itq_means;
  tk_dvec_t *itq_rotation;
  tk_dvec_t *median_thresholds;
  tk_ivec_t *landmark_ids;
  uint64_t n_dims;
  uint64_t n_landmarks;
  tk_ivec_sim_type_t cmp;
  double cmp_alpha;
  double cmp_beta;
  double decay;
  tk_binarize_type_t binarize;
  bool destroyed;
} tk_nystrom_encoder_t;

static inline tk_hlth_encoder_t *tk_hlth_encoder_peek(lua_State *L, int i) {
  return (tk_hlth_encoder_t *)luaL_checkudata(L, i, TK_HLTH_ENCODER_MT);
}

static inline int tk_hlth_encoder_gc(lua_State *L) {
  tk_hlth_encoder_t *enc = tk_hlth_encoder_peek(L, 1);
  enc->feat_idx = NULL;
  enc->code_idx = NULL;
  enc->destroyed = true;
  return 0;
}

static inline tk_nystrom_encoder_t *tk_nystrom_encoder_peek(lua_State *L, int i) {
  return (tk_nystrom_encoder_t *)luaL_checkudata(L, i, TK_NYSTROM_ENCODER_MT);
}

static inline int tk_nystrom_encoder_gc(lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  enc->feat_idx = NULL;
  enc->eigenvectors = NULL;
  enc->eigenvalues = NULL;
  enc->col_means = NULL;
  enc->itq_means = NULL;
  enc->itq_rotation = NULL;
  enc->median_thresholds = NULL;
  enc->landmark_ids = NULL;
  enc->destroyed = true;
  return 0;
}

static inline int tk_hlth_encode_lua(lua_State *L) {
  tk_hlth_encoder_t *enc = (tk_hlth_encoder_t *)lua_touserdata(L, lua_upvalueindex(1));

  if (enc->destroyed)
    return luaL_error(L, "encode: encoder has been destroyed");

  if (enc->mode == TK_HLTH_MODE_SIMILARITIES) {
    tk_cvec_t *query_cvec = tk_cvec_peekopt(L, 1);
    if (!query_cvec)
      return luaL_error(L, "encode: similarities mode requires cvec query");
    uint64_t n_samples = tk_lua_checkunsigned(L, 2, "n_samples");

    tk_ann_t *feat_ann = enc->feat_idx_type == TK_HLTH_IDX_ANN ? (tk_ann_t *)enc->feat_idx : NULL;

    tk_ann_hoods_t *ann_hoods = NULL;
    tk_ivec_t *nbr_ids = NULL;

    if (feat_ann) {
      tk_ann_neighborhoods_by_vecs(L, feat_ann, query_cvec, enc->n_landmarks, enc->probe_radius,
                                   &ann_hoods, &nbr_ids);
    } else {
      return luaL_error(L, "encode: similarities mode requires ann landmarks_index");
    }

    uint64_t hood_size = enc->n_landmarks;
    uint64_t n_bins = enc->n_bins;
    uint64_t query_bits = enc->concat_query ? enc->n_hidden : 0;
    uint64_t n_latent_bits = query_bits + hood_size * n_bins;
    uint64_t n_latent_bytes = TK_CVEC_BITS_BYTES(n_latent_bits);
    uint64_t query_bytes = TK_CVEC_BITS_BYTES(query_bits);

    int stack_before_out = lua_gettop(L);
    tk_cvec_t *out = tk_cvec_create(L, n_samples * n_latent_bytes, NULL, NULL);
    out->n = n_samples * n_latent_bytes;
    memset(out->a, 0, out->n);

    if (enc->concat_query) {
      uint64_t src_bytes_per_sample = TK_CVEC_BITS_BYTES(enc->n_hidden);
      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++) {
        uint8_t *src = (uint8_t *)query_cvec->a + i * src_bytes_per_sample;
        uint8_t *dst = (uint8_t *)out->a + i * n_latent_bytes;
        memcpy(dst, src, query_bytes);
      }
    }

    double *all_sims = malloc(n_samples * hood_size * sizeof(double));
    double *thresholds = malloc(hood_size * (n_bins - 1) * sizeof(double));
    for (uint64_t k = 0; k < hood_size; k++) {
      for (uint64_t i = 0; i < n_samples; i++) {
        double sim = 0.0;
        uint64_t cur_size = ann_hoods->a[i]->n;
        if (k < cur_size) {
          sim = (double)ann_hoods->a[i]->a[k].p;
        }
        all_sims[i] = sim;
      }
      tk_dvec_t *sorted = tk_dvec_create(NULL, n_samples, 0, 0);
      memcpy(sorted->a, all_sims, n_samples * sizeof(double));
      tk_dvec_asc(sorted, 0, n_samples);
      for (uint64_t b = 0; b < n_bins - 1; b++) {
        uint64_t idx = (b + 1) * n_samples / n_bins;
        if (idx >= n_samples) idx = n_samples - 1;
        thresholds[k * (n_bins - 1) + b] = sorted->a[idx];
      }
      tk_dvec_destroy(sorted);
      for (uint64_t i = 0; i < n_samples; i++) {
        double sim = all_sims[i];
        uint64_t bin = 0;
        for (uint64_t b = 0; b < n_bins - 1; b++) {
          if (sim > thresholds[k * (n_bins - 1) + b]) bin = b + 1;
        }
        uint64_t bit_idx = query_bits + k * n_bins + bin;
        uint8_t *sample_dest = (uint8_t *)out->a + i * n_latent_bytes;
        sample_dest[TK_CVEC_BITS_BYTE(bit_idx)] |= (1 << TK_CVEC_BITS_BIT(bit_idx));
      }
    }
    free(all_sims);
    free(thresholds);

    lua_replace(L, stack_before_out);
    lua_settop(L, stack_before_out);
    return 1;
  }

  tk_ivec_t *query_ivec = tk_ivec_peekopt(L, 1);
  tk_cvec_t *query_cvec = query_ivec ? NULL : tk_cvec_peekopt(L, 1);

  if (!query_ivec && !query_cvec)
    return luaL_error(L, "encode: expected ivec or cvec query");

  uint64_t n_samples = tk_lua_checkunsigned(L, 2, "n_samples");

  tk_inv_t *feat_inv = enc->feat_idx_type == TK_HLTH_IDX_INV ? (tk_inv_t *)enc->feat_idx : NULL;
  tk_ann_t *feat_ann = enc->feat_idx_type == TK_HLTH_IDX_ANN ? (tk_ann_t *)enc->feat_idx : NULL;

  tk_ann_t *code_ann = enc->code_idx_type == TK_HLTH_IDX_ANN ? (tk_ann_t *)enc->code_idx : NULL;

  tk_ann_hoods_t *ann_hoods = NULL;
  tk_inv_hoods_t *inv_hoods = NULL;
  tk_ivec_t *nbr_ids = NULL;

  if (feat_inv && query_ivec) {
    tk_inv_neighborhoods_by_vecs(L, feat_inv, query_ivec, enc->n_landmarks,
                                 enc->cmp, enc->cmp_alpha, enc->cmp_beta,
                                 0.0, &inv_hoods, &nbr_ids);
  } else if (feat_ann && query_cvec) {
    tk_ann_neighborhoods_by_vecs(L, feat_ann, query_cvec, enc->n_landmarks, enc->probe_radius,
                                 &ann_hoods, &nbr_ids);
  } else {
    return luaL_error(L, "encode: index/query type mismatch");
  }

  int stack_before_out = lua_gettop(L);

  uint64_t n_latent_bits;
  if (enc->mode == TK_HLTH_MODE_FREQUENCY || enc->mode == TK_HLTH_MODE_FREQUENCY_WEIGHTED)
    n_latent_bits = enc->n_hidden * enc->n_thresholds;
  else if (enc->mode == TK_HLTH_MODE_CENTROID || enc->mode == TK_HLTH_MODE_CENTROID_WEIGHTED)
    n_latent_bits = enc->n_hidden;
  else
    n_latent_bits = enc->n_hidden * enc->n_landmarks;

  uint64_t n_latent_bytes = TK_CVEC_BITS_BYTES(n_latent_bits);
  tk_cvec_t *out = tk_cvec_create(L, n_samples * n_latent_bytes, NULL, NULL);
  out->n = n_samples * n_latent_bytes;
  memset(out->a, 0, out->n);

  if (enc->mode == TK_HLTH_MODE_FREQUENCY) {
    double *all_freqs = NULL;
    double *quantile_thresholds = NULL;

    if (enc->quantile) {
      all_freqs = malloc(n_samples * enc->n_hidden * sizeof(double));
      quantile_thresholds = malloc(enc->n_hidden * enc->n_thresholds * sizeof(double));
    }

    #pragma omp parallel
    {
      uint64_t *bit_counts = calloc(enc->n_hidden, sizeof(uint64_t));
      tk_ivec_t *tmp = tk_ivec_create(NULL, 0, 0, 0);
      #pragma omp for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++) {
        memset(bit_counts, 0, enc->n_hidden * sizeof(uint64_t));
        tk_ivec_clear(tmp);
        int64_t nbr_idx, nbr_uid;
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, inv_hoods, ann_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
          tk_ivec_push(tmp, nbr_uid);
        });
        uint64_t n_found = 0;
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = tk_ann_get(code_ann, tmp->a[j]);
          if (code_data != NULL) {
            for (uint64_t b = 0; b < enc->n_hidden; b++) {
              if (code_data[TK_CVEC_BITS_BYTE(b)] & (1 << TK_CVEC_BITS_BIT(b)))
                bit_counts[b]++;
            }
            n_found++;
          }
        }

        if (enc->quantile) {
          for (uint64_t b = 0; b < enc->n_hidden; b++) {
            double freq = (n_found > 0) ? (double)bit_counts[b] / (double)n_found : 0.0;
            all_freqs[i * enc->n_hidden + b] = freq;
          }
        } else {
          uint8_t *sample_dest = (uint8_t *)out->a + i * n_latent_bytes;
          for (uint64_t b = 0; b < enc->n_hidden; b++) {
            double freq = (n_found > 0) ? (double)bit_counts[b] / (double)n_found : 0.0;
            for (uint64_t t = 0; t < enc->n_thresholds; t++) {
              double threshold = (double)(t + 1) / (double)(enc->n_thresholds + 1);
              if (freq > threshold) {
                uint64_t out_bit = b * enc->n_thresholds + t;
                sample_dest[TK_CVEC_BITS_BYTE(out_bit)] |= (1 << TK_CVEC_BITS_BIT(out_bit));
              }
            }
          }
        }
      }
      tk_ivec_destroy(tmp);
      free(bit_counts);
    }

    if (enc->quantile) {
      tk_dvec_t *sorted_freqs = tk_dvec_create(NULL, n_samples, 0, 0);
      for (uint64_t b = 0; b < enc->n_hidden; b++) {
        for (uint64_t i = 0; i < n_samples; i++) {
          sorted_freqs->a[i] = all_freqs[i * enc->n_hidden + b];
        }
        tk_dvec_asc(sorted_freqs, 0, n_samples);
        for (uint64_t t = 0; t < enc->n_thresholds; t++) {
          double quantile_pos = (double)(t + 1) / (double)(enc->n_thresholds + 1);
          uint64_t idx = (uint64_t)(quantile_pos * (double)(n_samples - 1));
          quantile_thresholds[b * enc->n_thresholds + t] = sorted_freqs->a[idx];
        }
      }
      tk_dvec_destroy(sorted_freqs);

      #pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++) {
        uint8_t *sample_dest = (uint8_t *)out->a + i * n_latent_bytes;
        for (uint64_t b = 0; b < enc->n_hidden; b++) {
          double freq = all_freqs[i * enc->n_hidden + b];
          for (uint64_t t = 0; t < enc->n_thresholds; t++) {
            double threshold = quantile_thresholds[b * enc->n_thresholds + t];
            if (freq > threshold) {
              uint64_t out_bit = b * enc->n_thresholds + t;
              sample_dest[TK_CVEC_BITS_BYTE(out_bit)] |= (1 << TK_CVEC_BITS_BIT(out_bit));
            }
          }
        }
      }

      free(all_freqs);
      free(quantile_thresholds);
    }
  } else if (enc->mode == TK_HLTH_MODE_FREQUENCY_WEIGHTED) {
    uint64_t feat_ann_features = feat_ann ? feat_ann->features : 0;
    #pragma omp parallel
    {
      double *bit_weights = calloc(enc->n_hidden, sizeof(double));
      tk_ivec_t *tmp = tk_ivec_create(NULL, 0, 0, 0);
      tk_dvec_t *sims = tk_dvec_create(NULL, 0, 0, 0);
      #pragma omp for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++) {
        memset(bit_weights, 0, enc->n_hidden * sizeof(double));
        tk_ivec_clear(tmp);
        tk_dvec_clear(sims);
        int64_t nbr_idx, nbr_uid;
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, inv_hoods, ann_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
          double sim = 1.0;
          if (inv_hoods) {
            tk_rvec_t *hood = inv_hoods->a[i];
            for (uint64_t j = 0; j < hood->n; j++) {
              if (hood->a[j].i == nbr_idx) {
                sim = 1.0 - hood->a[j].d;
                break;
              }
            }
          } else if (ann_hoods && feat_ann_features > 0) {
            tk_pvec_t *hood = ann_hoods->a[i];
            for (uint64_t j = 0; j < hood->n; j++) {
              if (hood->a[j].i == nbr_idx) {
                sim = 1.0 - (double)hood->a[j].p / (double)feat_ann_features;
                break;
              }
            }
          }
          tk_ivec_push(tmp, nbr_uid);
          tk_dvec_push(sims, sim);
        });
        double total_weight = 0.0;
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = tk_ann_get(code_ann, tmp->a[j]);
          if (code_data != NULL) {
            double w = sims->a[j];
            total_weight += w;
            for (uint64_t b = 0; b < enc->n_hidden; b++) {
              if (code_data[TK_CVEC_BITS_BYTE(b)] & (1 << TK_CVEC_BITS_BIT(b)))
                bit_weights[b] += w;
            }
          }
        }
        uint8_t *sample_dest = (uint8_t *)out->a + i * n_latent_bytes;
        for (uint64_t b = 0; b < enc->n_hidden; b++) {
          double freq = (total_weight > 0.0) ? bit_weights[b] / total_weight : 0.0;
          for (uint64_t t = 0; t < enc->n_thresholds; t++) {
            double threshold = (double)(t + 1) / (double)(enc->n_thresholds + 1);
            if (freq > threshold) {
              uint64_t out_bit = b * enc->n_thresholds + t;
              sample_dest[TK_CVEC_BITS_BYTE(out_bit)] |= (1 << TK_CVEC_BITS_BIT(out_bit));
            }
          }
        }
      }
      tk_ivec_destroy(tmp);
      tk_dvec_destroy(sims);
      free(bit_weights);
    }
  } else if (enc->mode == TK_HLTH_MODE_CENTROID) {
    #pragma omp parallel
    {
      int64_t *bit_votes = calloc(enc->n_hidden, sizeof(int64_t));
      tk_ivec_t *tmp = tk_ivec_create(NULL, 0, 0, 0);
      #pragma omp for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++) {
        memset(bit_votes, 0, enc->n_hidden * sizeof(int64_t));
        tk_ivec_clear(tmp);
        int64_t nbr_idx, nbr_uid;
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, inv_hoods, ann_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
          tk_ivec_push(tmp, nbr_uid);
        });
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = tk_ann_get(code_ann, tmp->a[j]);
          if (code_data != NULL) {
            for (uint64_t b = 0; b < enc->n_hidden; b++) {
              if (code_data[TK_CVEC_BITS_BYTE(b)] & (1 << TK_CVEC_BITS_BIT(b)))
                bit_votes[b]++;
              else
                bit_votes[b]--;
            }
          }
        }
        uint8_t *sample_dest = (uint8_t *)out->a + i * n_latent_bytes;
        for (uint64_t b = 0; b < enc->n_hidden; b++) {
          if (bit_votes[b] >= 0) {
            sample_dest[TK_CVEC_BITS_BYTE(b)] |= (1 << TK_CVEC_BITS_BIT(b));
          }
        }
      }
      tk_ivec_destroy(tmp);
      free(bit_votes);
    }
  } else if (enc->mode == TK_HLTH_MODE_CENTROID_WEIGHTED) {
    uint64_t feat_ann_features = feat_ann ? feat_ann->features : 0;
    #pragma omp parallel
    {
      double *bit_weights = calloc(enc->n_hidden, sizeof(double));
      tk_ivec_t *tmp = tk_ivec_create(NULL, 0, 0, 0);
      tk_dvec_t *sims = tk_dvec_create(NULL, 0, 0, 0);
      #pragma omp for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++) {
        memset(bit_weights, 0, enc->n_hidden * sizeof(double));
        tk_ivec_clear(tmp);
        tk_dvec_clear(sims);
        int64_t nbr_idx, nbr_uid;
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, inv_hoods, ann_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
          double sim = 1.0;
          if (inv_hoods) {
            tk_rvec_t *hood = inv_hoods->a[i];
            for (uint64_t j = 0; j < hood->n; j++) {
              if (hood->a[j].i == nbr_idx) {
                sim = 1.0 - hood->a[j].d;
                break;
              }
            }
          } else if (ann_hoods && feat_ann_features > 0) {
            tk_pvec_t *hood = ann_hoods->a[i];
            for (uint64_t j = 0; j < hood->n; j++) {
              if (hood->a[j].i == nbr_idx) {
                sim = 1.0 - (double)hood->a[j].p / (double)feat_ann_features;
                break;
              }
            }
          }
          tk_ivec_push(tmp, nbr_uid);
          tk_dvec_push(sims, sim);
        });
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = tk_ann_get(code_ann, tmp->a[j]);
          if (code_data != NULL) {
            double w = sims->a[j];
            for (uint64_t b = 0; b < enc->n_hidden; b++) {
              if (code_data[TK_CVEC_BITS_BYTE(b)] & (1 << TK_CVEC_BITS_BIT(b)))
                bit_weights[b] += w;
              else
                bit_weights[b] -= w;
            }
          }
        }
        uint8_t *sample_dest = (uint8_t *)out->a + i * n_latent_bytes;
        for (uint64_t b = 0; b < enc->n_hidden; b++) {
          if (bit_weights[b] >= 0.0) {
            sample_dest[TK_CVEC_BITS_BYTE(b)] |= (1 << TK_CVEC_BITS_BIT(b));
          }
        }
      }
      tk_ivec_destroy(tmp);
      tk_dvec_destroy(sims);
      free(bit_weights);
    }
  } else {
    #pragma omp parallel
    {
      tk_ivec_t *tmp = tk_ivec_create(NULL, 0, 0, 0);
      #pragma omp for schedule(static)
      for (uint64_t i = 0; i < n_samples; i++) {
        tk_ivec_clear(tmp);
        int64_t nbr_idx, nbr_uid;
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, inv_hoods, ann_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
          tk_ivec_push(tmp, nbr_uid);
        });
        uint8_t *sample_dest = (uint8_t *)out->a + i * n_latent_bytes;
        uint64_t bit_offset = 0;
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = tk_ann_get(code_ann, tmp->a[j]);
          if (code_data != NULL) {
            for (uint64_t b = 0; b < enc->n_hidden; b++) {
              if (code_data[TK_CVEC_BITS_BYTE(b)] & (1 << TK_CVEC_BITS_BIT(b))) {
                uint64_t dst_bit = bit_offset + b;
                sample_dest[TK_CVEC_BITS_BYTE(dst_bit)] |= (1 << TK_CVEC_BITS_BIT(dst_bit));
              }
            }
          }
          bit_offset += enc->n_hidden;
        }
      }
      tk_ivec_destroy(tmp);
    }
  }

  lua_replace(L, stack_before_out);
  lua_settop(L, stack_before_out);

  return 1;
}

static luaL_Reg tk_hlth_encoder_mt_fns[] = {
  { NULL, NULL }
};

static inline int tk_hlth_landmark_encoder_lua(lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  const char *mode_str = tk_lua_foptstring(L, 1, "landmark_encoder", "mode", "concat");
  bool is_similarities = !strcmp(mode_str, "similarities");

  tk_inv_t *feat_inv = NULL;
  tk_ann_t *feat_ann = NULL;
  tk_ann_t *code_ann = NULL;

  lua_getfield(L, 1, "landmarks_index");
  feat_inv = is_similarities ? NULL : tk_inv_peekopt(L, -1);
  feat_ann = tk_ann_peekopt(L, -1);
  lua_pop(L, 1);

  if (!feat_inv && !feat_ann)
    return luaL_error(L, "landmark_encoder: landmarks_index must be %s", is_similarities ? "ann" : "inv or ann");

  if (!is_similarities) {
    lua_getfield(L, 1, "codes_index");
    code_ann = tk_ann_peekopt(L, -1);
    lua_pop(L, 1);

    if (!code_ann)
      return luaL_error(L, "landmark_encoder: codes_index must be ann");
  }

  bool concat_query = tk_lua_foptboolean(L, 1, "landmark_encoder", "concat_query", true);

  uint64_t n_landmarks = tk_lua_foptunsigned(L, 1, "landmark_encoder", "n_landmarks", 24);

  const char *cmp_str = tk_lua_foptstring(L, 1, "landmark_encoder", "cmp", "jaccard");
  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(cmp_str, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(cmp_str, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(cmp_str, "dice"))
    cmp = TK_IVEC_DICE;
  else if (!strcmp(cmp_str, "tversky"))
    cmp = TK_IVEC_TVERSKY;

  double cmp_alpha = tk_lua_foptnumber(L, 1, "landmark_encoder", "cmp_alpha", 0.5);
  double cmp_beta = tk_lua_foptnumber(L, 1, "landmark_encoder", "cmp_beta", 0.5);

  uint64_t probe_radius = tk_lua_foptunsigned(L, 1, "landmark_encoder", "probe_radius", 2);

  tk_hlth_mode_t mode = TK_HLTH_MODE_CONCAT;
  if (!strcmp(mode_str, "frequency"))
    mode = TK_HLTH_MODE_FREQUENCY;
  else if (!strcmp(mode_str, "frequency_weighted"))
    mode = TK_HLTH_MODE_FREQUENCY_WEIGHTED;
  else if (!strcmp(mode_str, "centroid"))
    mode = TK_HLTH_MODE_CENTROID;
  else if (!strcmp(mode_str, "centroid_weighted"))
    mode = TK_HLTH_MODE_CENTROID_WEIGHTED;
  else if (!strcmp(mode_str, "similarities"))
    mode = TK_HLTH_MODE_SIMILARITIES;

  uint64_t n_thresholds = tk_lua_foptunsigned(L, 1, "landmark_encoder", "n_thresholds", 1);
  uint64_t n_bins = tk_lua_foptunsigned(L, 1, "landmark_encoder", "n_bins", 8);
  bool quantile = tk_lua_foptboolean(L, 1, "landmark_encoder", "quantile", false);

  uint64_t n_hidden = 0;
  if (!is_similarities)
    n_hidden = code_ann->features;
  else if (concat_query)
    n_hidden = feat_ann->features;

  tk_hlth_encoder_t *enc = tk_lua_newuserdata(L, tk_hlth_encoder_t, TK_HLTH_ENCODER_MT, tk_hlth_encoder_mt_fns, tk_hlth_encoder_gc);
  int Ei = lua_gettop(L);
  if (feat_inv) {
    enc->feat_idx = feat_inv;
    enc->feat_idx_type = TK_HLTH_IDX_INV;
  } else if (feat_ann) {
    enc->feat_idx = feat_ann;
    enc->feat_idx_type = TK_HLTH_IDX_ANN;
  } else {
    enc->feat_idx = NULL;
    enc->feat_idx_type = TK_HLTH_IDX_INV;
  }

  if (code_ann) {
    enc->code_idx = code_ann;
    enc->code_idx_type = TK_HLTH_IDX_ANN;
  } else {
    enc->code_idx = NULL;
    enc->code_idx_type = TK_HLTH_IDX_ANN;
  }

  enc->n_landmarks = n_landmarks;
  enc->n_hidden = n_hidden;
  enc->probe_radius = probe_radius;
  enc->cmp = cmp;
  enc->cmp_alpha = cmp_alpha;
  enc->cmp_beta = cmp_beta;
  enc->mode = mode;
  enc->n_thresholds = n_thresholds;
  enc->n_bins = n_bins;
  enc->quantile = quantile;
  enc->concat_query = concat_query;

  lua_getfield(L, 1, "landmarks_index");
  tk_lua_add_ephemeron(L, TK_HLTH_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  if (!is_similarities) {
    lua_getfield(L, 1, "codes_index");
    tk_lua_add_ephemeron(L, TK_HLTH_ENCODER_EPH, Ei, -1);
    lua_pop(L, 1);
  }

  lua_pushcclosure(L, tk_hlth_encode_lua, 1);

  int64_t n_latent;
  if (mode == TK_HLTH_MODE_FREQUENCY || mode == TK_HLTH_MODE_FREQUENCY_WEIGHTED)
    n_latent = (int64_t) n_hidden * (int64_t) n_thresholds;
  else if (mode == TK_HLTH_MODE_CENTROID || mode == TK_HLTH_MODE_CENTROID_WEIGHTED)
    n_latent = (int64_t) n_hidden;
  else if (mode == TK_HLTH_MODE_SIMILARITIES)
    n_latent = (int64_t) n_landmarks * (int64_t) n_bins + (concat_query ? (int64_t) n_hidden : 0);
  else
    n_latent = (int64_t) n_landmarks * (int64_t) n_hidden;
  lua_pushinteger(L, n_latent);

  return 2;
}

static luaL_Reg tk_nystrom_encoder_mt_fns[] = {
  { NULL, NULL }
};

static inline int tk_nystrom_encode_lua(lua_State *L) {
  tk_nystrom_encoder_t *enc = (tk_nystrom_encoder_t *)lua_touserdata(L, lua_upvalueindex(1));

  if (enc->destroyed)
    return luaL_error(L, "nystrom_encode: encoder has been destroyed");

  tk_ivec_t *query_vecs = tk_ivec_peekopt(L, 1);

  if (!query_vecs)
    return luaL_error(L, "nystrom_encode: expected ivec query");

  tk_inv_t *feat_inv = enc->feat_idx_type == TK_HLTH_IDX_INV ? (tk_inv_t *)enc->feat_idx : NULL;

  if (!feat_inv)
    return luaL_error(L, "nystrom_encode: features_index must be inv");

  uint64_t n_features = feat_inv->features;
  uint64_t n_landmarks = enc->n_landmarks;
  uint64_t n_dims = enc->n_dims;

  uint64_t n_samples = 0;
  for (uint64_t i = 0; i < query_vecs->n; i++) {
    int64_t encoded = query_vecs->a[i];
    if (encoded >= 0) {
      uint64_t sample_idx = (uint64_t)encoded / n_features;
      if (sample_idx >= n_samples) n_samples = sample_idx + 1;
    }
  }

  tk_ivec_t *query_offsets = tk_ivec_create(L, n_samples + 1, 0, 0);
  tk_ivec_t *query_features = tk_ivec_create(L, query_vecs->n, 0, 0);
  query_offsets->n = n_samples + 1;
  query_features->n = 0;

  for (uint64_t i = 0; i <= n_samples; i++)
    query_offsets->a[i] = 0;
  for (uint64_t i = 0; i < query_vecs->n; i++) {
    int64_t encoded = query_vecs->a[i];
    if (encoded >= 0) {
      uint64_t sample_idx = (uint64_t)encoded / n_features;
      query_offsets->a[sample_idx + 1]++;
    }
  }
  for (uint64_t i = 1; i <= n_samples; i++)
    query_offsets->a[i] += query_offsets->a[i - 1];

  tk_ivec_t *write_offsets = tk_ivec_create(L, n_samples, 0, 0);
  tk_ivec_copy(write_offsets, query_offsets, 0, (int64_t)n_samples, 0);

  for (uint64_t i = 0; i < query_vecs->n; i++) {
    int64_t encoded = query_vecs->a[i];
    if (encoded >= 0) {
      uint64_t sample_idx = (uint64_t)encoded / n_features;
      int64_t fid = encoded % (int64_t)n_features;
      int64_t write_pos = write_offsets->a[sample_idx]++;
      query_features->a[write_pos] = fid;
    }
  }
  query_features->n = (size_t)query_offsets->a[n_samples];

  int stack_before_out = lua_gettop(L);

  tk_dvec_t *raw_codes = tk_dvec_create(L, n_samples * n_dims, 0, 0);
  raw_codes->n = n_samples * n_dims;
  memset(raw_codes->a, 0, raw_codes->n * sizeof(double));

  tk_inv_rank_weights_t rw;
  tk_inv_precompute_rank_weights(&rw, feat_inv->n_ranks, enc->decay);
  int64_t *ranks_arr = feat_inv->ranks->a;
  double *weights_arr = feat_inv->weights->a;
  uint64_t n_ranks = feat_inv->n_ranks;

  #pragma omp parallel
  {
    tk_dvec_t *q_weights = tk_dvec_create(NULL, n_ranks, 0, 0);
    tk_dvec_t *e_weights = tk_dvec_create(NULL, n_ranks, 0, 0);
    tk_dvec_t *inter_weights = tk_dvec_create(NULL, n_ranks, 0, 0);
    tk_dvec_t *sims = tk_dvec_create(NULL, n_landmarks, 0, 0);
    sims->n = n_landmarks;

    #pragma omp for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      int64_t q_start = query_offsets->a[i];
      int64_t q_end = query_offsets->a[i + 1];
      int64_t *q_bits = query_features->a + q_start;
      size_t q_len = (size_t)(q_end - q_start);

      for (uint64_t j = 0; j < n_landmarks; j++) {
        int64_t landmark_uid = enc->landmark_ids->a[j];
        int64_t landmark_sid = tk_inv_uid_sid(feat_inv, landmark_uid, TK_INV_FIND);

        double sim = 0.0;
        if (landmark_sid >= 0 && q_len > 0) {
          size_t l_nbits;
          int64_t *l_bits = tk_inv_sget(feat_inv, landmark_sid, &l_nbits);
          if (l_bits && l_nbits > 0) {
            sim = tk_inv_similarity_fast(ranks_arr, weights_arr, n_ranks,
              q_bits, q_len, l_bits, l_nbits,
              enc->cmp, enc->cmp_alpha, enc->cmp_beta, TK_COMBINE_WEIGHTED_AVG, &rw,
              q_weights->a, e_weights->a, inter_weights->a);
          }
        }
        sims->a[j] = sim;
      }

      double *sample_out = raw_codes->a + i * n_dims;
      for (uint64_t d = 0; d < n_dims; d++) {
        double sum = 0.0;
        for (uint64_t j = 0; j < n_landmarks; j++) {
          double centered_sim = sims->a[j] - enc->col_means->a[j];
          double evec_jd = enc->eigenvectors->a[j * n_dims + d];
          sum += centered_sim * evec_jd;
        }
        sample_out[d] = sum;
      }
    }

    tk_dvec_destroy(q_weights);
    tk_dvec_destroy(e_weights);
    tk_dvec_destroy(inter_weights);
    tk_dvec_destroy(sims);
  }

  if (enc->binarize == TK_BINARIZE_ITQ && enc->itq_means && enc->itq_rotation) {
    tk_cvec_t *binary_codes = tk_cvec_create(L, n_samples * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
    binary_codes->n = n_samples * TK_CVEC_BITS_BYTES(n_dims);
    memset(binary_codes->a, 0, binary_codes->n);

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      for (uint64_t d = 0; d < n_dims; d++) {
        raw_codes->a[i * n_dims + d] -= enc->itq_means->a[d];
      }
    }

    double *rotated = malloc(n_samples * n_dims * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      n_samples, n_dims, n_dims,
      1.0, raw_codes->a, n_dims,
      enc->itq_rotation->a, n_dims,
      0.0, rotated, n_dims);

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      uint8_t *sample_out = (uint8_t *)binary_codes->a + i * TK_CVEC_BITS_BYTES(n_dims);
      for (uint64_t d = 0; d < n_dims; d++) {
        if (rotated[i * n_dims + d] >= 0.0) {
          sample_out[TK_CVEC_BITS_BYTE(d)] |= (1 << TK_CVEC_BITS_BIT(d));
        }
      }
    }
    free(rotated);

    lua_pushvalue(L, lua_gettop(L));
    lua_pushinteger(L, (lua_Integer)n_samples);
    return 2;
  }

  if (enc->binarize == TK_BINARIZE_SIGN) {
    tk_cvec_t *binary_codes = tk_cvec_create(L, n_samples * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
    binary_codes->n = n_samples * TK_CVEC_BITS_BYTES(n_dims);
    memset(binary_codes->a, 0, binary_codes->n);

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      uint8_t *sample_out = (uint8_t *)binary_codes->a + i * TK_CVEC_BITS_BYTES(n_dims);
      for (uint64_t d = 0; d < n_dims; d++) {
        if (raw_codes->a[i * n_dims + d] >= 0.0) {
          sample_out[TK_CVEC_BITS_BYTE(d)] |= (1 << TK_CVEC_BITS_BIT(d));
        }
      }
    }

    lua_pushvalue(L, lua_gettop(L));
    lua_pushinteger(L, (lua_Integer)n_samples);
    return 2;
  }

  if (enc->binarize == TK_BINARIZE_MEDIAN && enc->median_thresholds) {
    tk_cvec_t *binary_codes = tk_cvec_create(L, n_samples * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
    binary_codes->n = n_samples * TK_CVEC_BITS_BYTES(n_dims);
    memset(binary_codes->a, 0, binary_codes->n);

    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      uint8_t *sample_out = (uint8_t *)binary_codes->a + i * TK_CVEC_BITS_BYTES(n_dims);
      for (uint64_t d = 0; d < n_dims; d++) {
        if (raw_codes->a[i * n_dims + d] >= enc->median_thresholds->a[d]) {
          sample_out[TK_CVEC_BITS_BYTE(d)] |= (1 << TK_CVEC_BITS_BIT(d));
        }
      }
    }

    lua_pushvalue(L, lua_gettop(L));
    lua_pushinteger(L, (lua_Integer)n_samples);
    return 2;
  }

  lua_pushvalue(L, stack_before_out + 1);
  lua_pushinteger(L, (lua_Integer)n_samples);

  return 2;
}

static inline int tk_hlth_nystrom_encoder_lua(lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "features_index");
  tk_inv_t *feat_inv = tk_inv_peekopt(L, -1);
  lua_pop(L, 1);

  if (!feat_inv)
    return luaL_error(L, "nystrom_encoder: features_index must be inv");

  lua_getfield(L, 1, "eigenvectors");
  tk_dvec_t *eigenvectors = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  if (!eigenvectors)
    return luaL_error(L, "nystrom_encoder: eigenvectors required");

  lua_getfield(L, 1, "eigenvalues");
  tk_dvec_t *eigenvalues = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  if (!eigenvalues)
    return luaL_error(L, "nystrom_encoder: eigenvalues required");

  lua_getfield(L, 1, "landmark_ids");
  tk_ivec_t *landmark_ids = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);
  if (!landmark_ids)
    return luaL_error(L, "nystrom_encoder: landmark_ids required");

  lua_getfield(L, 1, "col_means");
  tk_dvec_t *col_means = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  if (!col_means)
    return luaL_error(L, "nystrom_encoder: col_means required");

  lua_getfield(L, 1, "itq_means");
  tk_dvec_t *itq_means = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "itq_rotation");
  tk_dvec_t *itq_rotation = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "median_thresholds");
  tk_dvec_t *median_thresholds = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);

  const char *binarize_str = tk_lua_foptstring(L, 1, "nystrom_encoder", "binarize", "itq");
  tk_binarize_type_t binarize = TK_BINARIZE_ITQ;
  if (!strcmp(binarize_str, "itq"))
    binarize = TK_BINARIZE_ITQ;
  else if (!strcmp(binarize_str, "sign"))
    binarize = TK_BINARIZE_SIGN;
  else if (!strcmp(binarize_str, "median"))
    binarize = TK_BINARIZE_MEDIAN;

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "nystrom_encoder", "n_dims");

  if (n_dims == 0)
    return luaL_error(L, "nystrom_encoder: n_dims must be > 0");
  if (eigenvalues->n != n_dims)
    return luaL_error(L, "nystrom_encoder: eigenvalues size (%llu) != n_dims (%llu)",
                      (unsigned long long)eigenvalues->n, (unsigned long long)n_dims);
  if (eigenvectors->n % n_dims != 0)
    return luaL_error(L, "nystrom_encoder: eigenvectors size (%llu) not divisible by n_dims (%llu)",
                      (unsigned long long)eigenvectors->n, (unsigned long long)n_dims);

  uint64_t n_landmarks = eigenvectors->n / n_dims;
  if (landmark_ids->n != n_landmarks)
    return luaL_error(L, "nystrom_encoder: landmark_ids size (%llu) != eigenvector rows (%llu)",
                      (unsigned long long)landmark_ids->n, (unsigned long long)n_landmarks);

  const char *cmp_str = tk_lua_foptstring(L, 1, "nystrom_encoder", "cmp", "jaccard");
  tk_ivec_sim_type_t cmp = TK_IVEC_JACCARD;
  if (!strcmp(cmp_str, "jaccard"))
    cmp = TK_IVEC_JACCARD;
  else if (!strcmp(cmp_str, "overlap"))
    cmp = TK_IVEC_OVERLAP;
  else if (!strcmp(cmp_str, "dice"))
    cmp = TK_IVEC_DICE;
  else if (!strcmp(cmp_str, "tversky"))
    cmp = TK_IVEC_TVERSKY;

  double cmp_alpha = tk_lua_foptnumber(L, 1, "nystrom_encoder", "cmp_alpha", 0.5);
  double cmp_beta = tk_lua_foptnumber(L, 1, "nystrom_encoder", "cmp_beta", 0.5);
  double decay = tk_lua_foptnumber(L, 1, "nystrom_encoder", "decay", 0.0);

  tk_nystrom_encoder_t *enc = tk_lua_newuserdata(L, tk_nystrom_encoder_t, TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int Ei = lua_gettop(L);

  enc->feat_idx = feat_inv;
  enc->feat_idx_type = TK_HLTH_IDX_INV;
  enc->eigenvectors = eigenvectors;
  enc->eigenvalues = eigenvalues;
  enc->col_means = col_means;
  enc->itq_means = itq_means;
  enc->itq_rotation = itq_rotation;
  enc->median_thresholds = median_thresholds;
  enc->landmark_ids = landmark_ids;
  enc->n_dims = n_dims;
  enc->n_landmarks = n_landmarks;
  enc->cmp = cmp;
  enc->cmp_alpha = cmp_alpha;
  enc->cmp_beta = cmp_beta;
  enc->decay = decay;
  enc->binarize = binarize;
  enc->destroyed = false;

  lua_getfield(L, 1, "features_index");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "eigenvectors");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "eigenvalues");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "landmark_ids");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "col_means");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  if (itq_means) {
    lua_getfield(L, 1, "itq_means");
    tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
    lua_pop(L, 1);
  }

  if (itq_rotation) {
    lua_getfield(L, 1, "itq_rotation");
    tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
    lua_pop(L, 1);
  }

  if (median_thresholds) {
    lua_getfield(L, 1, "median_thresholds");
    tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
    lua_pop(L, 1);
  }

  lua_pushcclosure(L, tk_nystrom_encode_lua, 1);
  lua_pushinteger(L, (lua_Integer)n_dims);

  return 2;
}

static inline int tk_hlth_nystrom_lift_lua(lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "chol");
  tk_dvec_t *chol = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  if (!chol)
    return luaL_error(L, "nystrom_lift: chol (Cholesky factor L) required");

  lua_getfield(L, 1, "eigenvectors");
  tk_dvec_t *eigenvectors = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  if (!eigenvectors)
    return luaL_error(L, "nystrom_lift: eigenvectors required");

  lua_getfield(L, 1, "eigenvalues");
  tk_dvec_t *eigenvalues = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  if (!eigenvalues)
    return luaL_error(L, "nystrom_lift: eigenvalues required");

  uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "nystrom_lift", "n_samples");
  uint64_t n_landmarks = tk_lua_fcheckunsigned(L, 1, "nystrom_lift", "n_landmarks");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "nystrom_lift", "n_dims");

  if (chol->n != n_samples * n_landmarks)
    return luaL_error(L, "nystrom_lift: chol size (%llu) != n_samples * n_landmarks (%llu)",
      (unsigned long long)chol->n, (unsigned long long)(n_samples * n_landmarks));

  if (eigenvectors->n != n_landmarks * n_dims)
    return luaL_error(L, "nystrom_lift: eigenvectors size (%llu) != n_landmarks * n_dims (%llu)",
      (unsigned long long)eigenvectors->n, (unsigned long long)(n_landmarks * n_dims));

  if (eigenvalues->n != n_dims)
    return luaL_error(L, "nystrom_lift: eigenvalues size (%llu) != n_dims (%llu)",
      (unsigned long long)eigenvalues->n, (unsigned long long)n_dims);

  tk_dvec_t *col_means = tk_dvec_create(L, n_landmarks, 0, 0);
  col_means->n = n_landmarks;

  tk_dvec_t *out = tk_dvec_create(L, n_samples * n_dims, 0, 0);
  out->n = n_samples * n_dims;

  double *adjustment = (double *)malloc(n_dims * sizeof(double));

  #pragma omp parallel
  {
    #pragma omp for schedule(static)
    for (uint64_t j = 0; j < n_landmarks; j++) {
      double sum = 0.0;
      for (uint64_t i = 0; i < n_samples; i++)
        sum += chol->a[i * n_landmarks + j];
      col_means->a[j] = sum / (double)n_samples;
    }

    #pragma omp single
    {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        (int)n_samples, (int)n_dims, (int)n_landmarks,
        1.0, chol->a, (int)n_landmarks, eigenvectors->a, (int)n_dims,
        0.0, out->a, (int)n_dims);

      cblas_dgemv(CblasRowMajor, CblasTrans,
        (int)n_landmarks, (int)n_dims,
        1.0, eigenvectors->a, (int)n_dims, col_means->a, 1,
        0.0, adjustment, 1);
    }

    #pragma omp for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      double *U_row = out->a + i * n_dims;
      for (uint64_t d = 0; d < n_dims; d++)
        U_row[d] -= adjustment[d];
    }
  }

  free(adjustment);

  return 2;
}

static inline int tk_hlth_cca_linear_predict_lua(lua_State *L) {
  lua_settop(L, 1);

  lua_getfield(L, 1, "tokens");
  tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
  lua_pop(L, 1);

  lua_getfield(L, 1, "feat_ids");
  tk_ivec_t *feat_ids = tk_ivec_peek(L, -1, "feat_ids");
  lua_pop(L, 1);

  lua_getfield(L, 1, "feat_weights");
  tk_dvec_t *feat_weights = tk_dvec_peek(L, -1, "feat_weights");
  lua_pop(L, 1);

  uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "cca_linear_predict", "n_samples");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "cca_linear_predict", "n_dims");
  uint64_t n_tokens = tk_lua_fcheckunsigned(L, 1, "cca_linear_predict", "n_tokens");
  uint64_t features_per_class = tk_lua_fcheckunsigned(L, 1, "cca_linear_predict", "features_per_class");

  uint64_t code_bytes = TK_CVEC_BITS_BYTES(n_dims);
  tk_cvec_t *out = tk_cvec_create(L, n_samples * code_bytes, NULL, NULL);
  out->n = n_samples * code_bytes;
  memset(out->a, 0, out->n);

  int64_t *csr_offsets = tk_malloc(L, (n_samples + 1) * sizeof(int64_t));
  memset(csr_offsets, 0, (n_samples + 1) * sizeof(int64_t));
  for (uint64_t i = 0; i < tokens->n; i++) {
    int64_t v = tokens->a[i];
    if (v < 0) continue;
    uint64_t s = (uint64_t)v / n_tokens;
    if (s < n_samples)
      csr_offsets[s + 1]++;
  }
  for (uint64_t i = 1; i <= n_samples; i++)
    csr_offsets[i] += csr_offsets[i - 1];

  uint64_t total = (uint64_t)csr_offsets[n_samples];
  int64_t *tok_ids = tk_malloc(L, (total > 0 ? total : 1) * sizeof(int64_t));
  int64_t *counts = tk_malloc(L, n_samples * sizeof(int64_t));
  memset(counts, 0, n_samples * sizeof(int64_t));
  for (uint64_t i = 0; i < tokens->n; i++) {
    int64_t v = tokens->a[i];
    if (v < 0) continue;
    uint64_t s = (uint64_t)v / n_tokens;
    uint64_t tok = (uint64_t)v % n_tokens;
    if (s < n_samples) {
      uint64_t pos = (uint64_t)csr_offsets[s] + (uint64_t)counts[s];
      tok_ids[pos] = (int64_t)tok;
      counts[s]++;
    }
  }
  free(counts);

  tk_iuset_t **sample_sets = tk_malloc(L, n_samples * sizeof(tk_iuset_t *));
  for (uint64_t i = 0; i < n_samples; i++) {
    sample_sets[i] = tk_iuset_create(NULL, 0);
    int64_t start = csr_offsets[i];
    int64_t end = csr_offsets[i + 1];
    for (int64_t j = start; j < end; j++) {
      int kha;
      tk_iuset_put(sample_sets[i], tok_ids[j], &kha);
    }
  }
  free(csr_offsets);
  free(tok_ids);

  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    tk_iuset_t *sample_toks = sample_sets[i];
    uint8_t *sample_out = (uint8_t *)out->a + i * code_bytes;

    for (uint64_t d = 0; d < n_dims; d++) {
      double score = 0.0;
      for (uint64_t f = 0; f < features_per_class; f++) {
        int64_t fid = feat_ids->a[d * features_per_class + f];
        if (tk_iuset_contains(sample_toks, fid)) {
          score += feat_weights->a[d * features_per_class + f];
        }
      }
      if (score > 0.0) {
        sample_out[TK_CVEC_BITS_BYTE(d)] |= (1 << TK_CVEC_BITS_BIT(d));
      }
    }
  }

  for (uint64_t i = 0; i < n_samples; i++) {
    tk_iuset_destroy(sample_sets[i]);
  }
  free(sample_sets);

  return 1;
}

static luaL_Reg tk_hlth_fns[] = {
  { "landmark_encoder", tk_hlth_landmark_encoder_lua },
  { "nystrom_encoder", tk_hlth_nystrom_encoder_lua },
  { "nystrom_lift", tk_hlth_nystrom_lift_lua },
  { "cca_linear_predict", tk_hlth_cca_linear_predict_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_hlth(lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_hlth_fns, 0);
  return 1;
}
