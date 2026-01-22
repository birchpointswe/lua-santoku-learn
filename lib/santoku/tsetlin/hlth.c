#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
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
#include <santoku/tsetlin/hbi.h>
#include <santoku/tsetlin/graph.h>

#define TK_HLTH_ENCODER_MT "tk_hlth_encoder_t"
#define TK_HLTH_ENCODER_EPH "tk_hlth_encoder_eph"
#define TK_NYSTROM_ENCODER_MT "tk_nystrom_encoder_t"
#define TK_NYSTROM_ENCODER_EPH "tk_nystrom_encoder_eph"

typedef enum {
  TK_HLTH_IDX_INV,
  TK_HLTH_IDX_ANN,
  TK_HLTH_IDX_HBI
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
  double tversky_alpha;
  double tversky_beta;
  int64_t rank_filter;
  tk_hlth_mode_t mode;
  uint64_t n_thresholds;
  uint64_t n_bins;
  bool quantile;
  bool destroyed;
} tk_hlth_encoder_t;

typedef struct {
  tk_hlth_index_type_t feat_idx_type;
  void *feat_idx;
  tk_dvec_t *eigenvectors;
  tk_dvec_t *eigenvalues;
  tk_ivec_t *ids;
  tk_iumap_t *id_to_idx;
  uint64_t n_dims;
  uint64_t n_training_samples;
  uint64_t k_neighbors;
  tk_ivec_sim_type_t cmp;
  double cmp_alpha;
  double cmp_beta;
  uint64_t probe_radius;
  int64_t rank_filter;
  int threshold_fn_ref;
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
  if (enc->id_to_idx) {
    tk_iumap_destroy(enc->id_to_idx);
    enc->id_to_idx = NULL;
  }
  if (enc->threshold_fn_ref != LUA_NOREF) {
    luaL_unref(L, LUA_REGISTRYINDEX, enc->threshold_fn_ref);
    enc->threshold_fn_ref = LUA_NOREF;
  }
  enc->feat_idx = NULL;
  enc->eigenvectors = NULL;
  enc->eigenvalues = NULL;
  enc->ids = NULL;
  enc->destroyed = true;
  return 0;
}

static inline int tk_hlth_encode_lua(lua_State *L) {
  tk_hlth_encoder_t *enc = (tk_hlth_encoder_t *)lua_touserdata(L, lua_upvalueindex(1));

  if (enc->destroyed)
    return luaL_error(L, "encode: encoder has been destroyed");

  if (enc->mode == TK_HLTH_MODE_SIMILARITIES) {
    tk_inv_hoods_t *inv_hoods = tk_inv_hoods_peekopt(L, 1);
    tk_ann_hoods_t *ann_hoods = inv_hoods ? NULL : tk_ann_hoods_peekopt(L, 1);
    tk_hbi_hoods_t *hbi_hoods = (inv_hoods || ann_hoods) ? NULL : tk_hbi_hoods_peekopt(L, 1);
    if (!inv_hoods && !ann_hoods && !hbi_hoods)
      return luaL_error(L, "encode: similarities mode requires hoods (inv, ann, or hbi)");
    uint64_t n_samples = inv_hoods ? inv_hoods->n : (ann_hoods ? ann_hoods->n : hbi_hoods->n);
    uint64_t hood_size = enc->n_landmarks;
    uint64_t n_bins = enc->n_bins;
    uint64_t n_latent_bits = hood_size * n_bins;
    uint64_t n_latent_bytes = TK_CVEC_BITS_BYTES(n_latent_bits);
    tk_cvec_t *out = tk_cvec_create(L, n_samples * n_latent_bytes, NULL, NULL);
    out->n = n_samples * n_latent_bytes;
    memset(out->a, 0, out->n);
    double *all_sims = malloc(n_samples * hood_size * sizeof(double));
    double *thresholds = malloc(hood_size * (n_bins - 1) * sizeof(double));
    for (uint64_t k = 0; k < hood_size; k++) {
      for (uint64_t i = 0; i < n_samples; i++) {
        double sim = 0.0;
        uint64_t cur_size = inv_hoods ? inv_hoods->a[i]->n : (ann_hoods ? ann_hoods->a[i]->n : hbi_hoods->a[i]->n);
        if (k < cur_size) {
          if (inv_hoods) sim = inv_hoods->a[i]->a[k].d;
          else if (ann_hoods) sim = (double)ann_hoods->a[i]->a[k].p;
          else sim = (double)hbi_hoods->a[i]->a[k].p;
        }
        all_sims[i] = sim;
      }
      double *sorted = malloc(n_samples * sizeof(double));
      memcpy(sorted, all_sims, n_samples * sizeof(double));
      for (uint64_t i = 0; i < n_samples; i++) {
        for (uint64_t j = i + 1; j < n_samples; j++) {
          if (sorted[i] > sorted[j]) {
            double tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
          }
        }
      }
      for (uint64_t b = 0; b < n_bins - 1; b++) {
        uint64_t idx = (b + 1) * n_samples / n_bins;
        if (idx >= n_samples) idx = n_samples - 1;
        thresholds[k * (n_bins - 1) + b] = sorted[idx];
      }
      free(sorted);
      for (uint64_t i = 0; i < n_samples; i++) {
        double sim = all_sims[i];
        uint64_t bin = 0;
        for (uint64_t b = 0; b < n_bins - 1; b++) {
          if (sim > thresholds[k * (n_bins - 1) + b]) bin = b + 1;
        }
        uint64_t bit_idx = k * n_bins + bin;
        uint8_t *sample_dest = (uint8_t *)out->a + i * n_latent_bytes;
        sample_dest[TK_CVEC_BITS_BYTE(bit_idx)] |= (1 << TK_CVEC_BITS_BIT(bit_idx));
      }
    }
    free(all_sims);
    free(thresholds);
    return 1;
  }

  tk_ivec_t *query_ivec = tk_ivec_peekopt(L, 1);
  tk_cvec_t *query_cvec = query_ivec ? NULL : tk_cvec_peekopt(L, 1);

  if (!query_ivec && !query_cvec)
    return luaL_error(L, "encode: expected ivec or cvec query");

  uint64_t n_samples = tk_lua_checkunsigned(L, 2, "n_samples");

  tk_inv_t *feat_inv = enc->feat_idx_type == TK_HLTH_IDX_INV ? (tk_inv_t *)enc->feat_idx : NULL;
  tk_ann_t *feat_ann = enc->feat_idx_type == TK_HLTH_IDX_ANN ? (tk_ann_t *)enc->feat_idx : NULL;
  tk_hbi_t *feat_hbi = enc->feat_idx_type == TK_HLTH_IDX_HBI ? (tk_hbi_t *)enc->feat_idx : NULL;

  tk_ann_t *code_ann = enc->code_idx_type == TK_HLTH_IDX_ANN ? (tk_ann_t *)enc->code_idx : NULL;
  tk_hbi_t *code_hbi = enc->code_idx_type == TK_HLTH_IDX_HBI ? (tk_hbi_t *)enc->code_idx : NULL;

  tk_ann_hoods_t *ann_hoods = NULL;
  tk_hbi_hoods_t *hbi_hoods = NULL;
  tk_inv_hoods_t *inv_hoods = NULL;
  tk_ivec_t *nbr_ids = NULL;

  if (feat_inv && query_ivec) {
    tk_inv_neighborhoods_by_vecs(L, feat_inv, query_ivec, enc->n_landmarks, 0.0, 1.0,
                                 enc->cmp, enc->tversky_alpha, enc->tversky_beta,
                                 0.0, enc->rank_filter, &inv_hoods, &nbr_ids);
  } else if (feat_ann && query_cvec) {
    tk_ann_neighborhoods_by_vecs(L, feat_ann, query_cvec, enc->n_landmarks, enc->probe_radius,
                                 0, ~0ULL, &ann_hoods, &nbr_ids);
  } else if (feat_hbi && query_cvec) {
    tk_hbi_neighborhoods_by_vecs(L, feat_hbi, query_cvec, enc->n_landmarks, 0, ~0ULL, &hbi_hoods, &nbr_ids);
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
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, feat_hbi, inv_hoods, ann_hoods, hbi_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
          tk_ivec_push(tmp, nbr_uid);
        });
        uint64_t n_found = 0;
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = code_ann ? tk_ann_get(code_ann, tmp->a[j]) : tk_hbi_get(code_hbi, tmp->a[j]);
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
      double *sorted_freqs = malloc(n_samples * sizeof(double));
      for (uint64_t b = 0; b < enc->n_hidden; b++) {
        for (uint64_t i = 0; i < n_samples; i++) {
          sorted_freqs[i] = all_freqs[i * enc->n_hidden + b];
        }
        for (uint64_t i = 0; i < n_samples - 1; i++) {
          for (uint64_t j = i + 1; j < n_samples; j++) {
            if (sorted_freqs[j] < sorted_freqs[i]) {
              double tmp = sorted_freqs[i];
              sorted_freqs[i] = sorted_freqs[j];
              sorted_freqs[j] = tmp;
            }
          }
        }
        for (uint64_t t = 0; t < enc->n_thresholds; t++) {
          double quantile_pos = (double)(t + 1) / (double)(enc->n_thresholds + 1);
          uint64_t idx = (uint64_t)(quantile_pos * (double)(n_samples - 1));
          quantile_thresholds[b * enc->n_thresholds + t] = sorted_freqs[idx];
        }
      }
      free(sorted_freqs);

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
    uint64_t feat_hbi_features = feat_hbi ? feat_hbi->features : 0;
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
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, feat_hbi, inv_hoods, ann_hoods, hbi_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
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
          } else if (hbi_hoods && feat_hbi_features > 0) {
            tk_pvec_t *hood = hbi_hoods->a[i];
            for (uint64_t j = 0; j < hood->n; j++) {
              if (hood->a[j].i == nbr_idx) {
                sim = 1.0 - (double)hood->a[j].p / (double)feat_hbi_features;
                break;
              }
            }
          }
          tk_ivec_push(tmp, nbr_uid);
          tk_dvec_push(sims, sim);
        });
        double total_weight = 0.0;
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = code_ann ? tk_ann_get(code_ann, tmp->a[j]) : tk_hbi_get(code_hbi, tmp->a[j]);
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
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, feat_hbi, inv_hoods, ann_hoods, hbi_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
          tk_ivec_push(tmp, nbr_uid);
        });
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = code_ann ? tk_ann_get(code_ann, tmp->a[j]) : tk_hbi_get(code_hbi, tmp->a[j]);
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
    uint64_t feat_hbi_features = feat_hbi ? feat_hbi->features : 0;
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
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, feat_hbi, inv_hoods, ann_hoods, hbi_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
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
          } else if (hbi_hoods && feat_hbi_features > 0) {
            tk_pvec_t *hood = hbi_hoods->a[i];
            for (uint64_t j = 0; j < hood->n; j++) {
              if (hood->a[j].i == nbr_idx) {
                sim = 1.0 - (double)hood->a[j].p / (double)feat_hbi_features;
                break;
              }
            }
          }
          tk_ivec_push(tmp, nbr_uid);
          tk_dvec_push(sims, sim);
        });
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = code_ann ? tk_ann_get(code_ann, tmp->a[j]) : tk_hbi_get(code_hbi, tmp->a[j]);
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
        TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, feat_hbi, inv_hoods, ann_hoods, hbi_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
          tk_ivec_push(tmp, nbr_uid);
        });
        uint8_t *sample_dest = (uint8_t *)out->a + i * n_latent_bytes;
        uint64_t bit_offset = 0;
        for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
          char *code_data = code_ann ? tk_ann_get(code_ann, tmp->a[j]) : tk_hbi_get(code_hbi, tmp->a[j]);
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
  tk_hbi_t *feat_hbi = NULL;
  tk_ann_t *code_ann = NULL;
  tk_hbi_t *code_hbi = NULL;

  if (!is_similarities) {
    lua_getfield(L, 1, "landmarks_index");
    feat_inv = tk_inv_peekopt(L, -1);
    feat_ann = tk_ann_peekopt(L, -1);
    feat_hbi = tk_hbi_peekopt(L, -1);
    lua_pop(L, 1);

    if (!feat_inv && !feat_ann && !feat_hbi)
      return luaL_error(L, "landmark_encoder: landmark_index must be inv, ann, or hbi");

    lua_getfield(L, 1, "codes_index");
    code_ann = tk_ann_peekopt(L, -1);
    code_hbi = tk_hbi_peekopt(L, -1);
    lua_pop(L, 1);

    if (!code_ann && !code_hbi)
      return luaL_error(L, "landmark_encoder: code_index must be ann or hbi");
  }

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

  double tversky_alpha = tk_lua_foptnumber(L, 1, "landmark_encoder", "tversky_alpha", 0.5);
  double tversky_beta = tk_lua_foptnumber(L, 1, "landmark_encoder", "tversky_beta", 0.5);
  int64_t rank_filter = tk_lua_foptinteger(L, 1, "landmark_encoder", "rank_filter", -1);

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
    n_hidden = code_ann ? code_ann->features : code_hbi->features;

  tk_hlth_encoder_t *enc = tk_lua_newuserdata(L, tk_hlth_encoder_t, TK_HLTH_ENCODER_MT, tk_hlth_encoder_mt_fns, tk_hlth_encoder_gc);
  int Ei = lua_gettop(L);
  if (feat_inv) {
    enc->feat_idx = feat_inv;
    enc->feat_idx_type = TK_HLTH_IDX_INV;
  } else if (feat_ann) {
    enc->feat_idx = feat_ann;
    enc->feat_idx_type = TK_HLTH_IDX_ANN;
  } else if (feat_hbi) {
    enc->feat_idx = feat_hbi;
    enc->feat_idx_type = TK_HLTH_IDX_HBI;
  } else {
    enc->feat_idx = NULL;
    enc->feat_idx_type = TK_HLTH_IDX_INV;
  }

  if (code_ann) {
    enc->code_idx = code_ann;
    enc->code_idx_type = TK_HLTH_IDX_ANN;
  } else if (code_hbi) {
    enc->code_idx = code_hbi;
    enc->code_idx_type = TK_HLTH_IDX_HBI;
  } else {
    enc->code_idx = NULL;
    enc->code_idx_type = TK_HLTH_IDX_ANN;
  }

  enc->n_landmarks = n_landmarks;
  enc->n_hidden = n_hidden;
  enc->probe_radius = probe_radius;
  enc->cmp = cmp;
  enc->tversky_alpha = tversky_alpha;
  enc->tversky_beta = tversky_beta;
  enc->rank_filter = rank_filter;
  enc->mode = mode;
  enc->n_thresholds = n_thresholds;
  enc->n_bins = n_bins;
  enc->quantile = quantile;

  if (!is_similarities) {
    lua_getfield(L, 1, "landmarks_index");
    tk_lua_add_ephemeron(L, TK_HLTH_ENCODER_EPH, Ei, -1);
    lua_pop(L, 1);

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
    n_latent = (int64_t) n_landmarks * (int64_t) n_bins;
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

  tk_ivec_t *query_ivec = tk_ivec_peekopt(L, 1);
  tk_cvec_t *query_cvec = query_ivec ? NULL : tk_cvec_peekopt(L, 1);

  if (!query_ivec && !query_cvec)
    return luaL_error(L, "nystrom_encode: expected ivec or cvec query");

  uint64_t n_samples = tk_lua_checkunsigned(L, 2, "n_samples");

  tk_inv_t *feat_inv = enc->feat_idx_type == TK_HLTH_IDX_INV ? (tk_inv_t *)enc->feat_idx : NULL;
  tk_ann_t *feat_ann = enc->feat_idx_type == TK_HLTH_IDX_ANN ? (tk_ann_t *)enc->feat_idx : NULL;
  tk_hbi_t *feat_hbi = enc->feat_idx_type == TK_HLTH_IDX_HBI ? (tk_hbi_t *)enc->feat_idx : NULL;

  tk_ann_hoods_t *ann_hoods = NULL;
  tk_hbi_hoods_t *hbi_hoods = NULL;
  tk_inv_hoods_t *inv_hoods = NULL;
  tk_ivec_t *nbr_ids = NULL;

  uint64_t k = enc->k_neighbors > 0 ? enc->k_neighbors : enc->n_training_samples;

  if (feat_inv && query_ivec) {
    tk_inv_neighborhoods_by_vecs(L, feat_inv, query_ivec, k, 0.0, 1.0,
                                 enc->cmp, enc->cmp_alpha, enc->cmp_beta,
                                 0.0, enc->rank_filter, &inv_hoods, &nbr_ids);
  } else if (feat_ann && query_cvec) {
    tk_ann_neighborhoods_by_vecs(L, feat_ann, query_cvec, k, enc->probe_radius,
                                 0, ~0ULL, &ann_hoods, &nbr_ids);
  } else if (feat_hbi && query_cvec) {
    tk_hbi_neighborhoods_by_vecs(L, feat_hbi, query_cvec, k, 0, ~0ULL, &hbi_hoods, &nbr_ids);
  } else {
    return luaL_error(L, "nystrom_encode: index/query type mismatch");
  }

  int stack_before_out = lua_gettop(L);

  tk_dvec_t *raw_codes = tk_dvec_create(L, n_samples * enc->n_dims, 0, 0);
  raw_codes->n = n_samples * enc->n_dims;
  memset(raw_codes->a, 0, raw_codes->n * sizeof(double));

  uint64_t feat_ann_features = feat_ann ? feat_ann->features : 0;
  uint64_t feat_hbi_features = feat_hbi ? feat_hbi->features : 0;

  #pragma omp parallel
  {
    tk_ivec_t *tmp = tk_ivec_create(NULL, 0, 0, 0);
    tk_dvec_t *sims = tk_dvec_create(NULL, 0, 0, 0);

    #pragma omp for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      tk_ivec_clear(tmp);
      tk_dvec_clear(sims);

      int64_t nbr_idx, nbr_uid;
      TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, feat_hbi, inv_hoods, ann_hoods, hbi_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
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
        } else if (hbi_hoods && feat_hbi_features > 0) {
          tk_pvec_t *hood = hbi_hoods->a[i];
          for (uint64_t j = 0; j < hood->n; j++) {
            if (hood->a[j].i == nbr_idx) {
              sim = 1.0 - (double)hood->a[j].p / (double)feat_hbi_features;
              break;
            }
          }
        }
        tk_ivec_push(tmp, nbr_uid);
        tk_dvec_push(sims, sim);
      });

      double *sample_out = raw_codes->a + i * enc->n_dims;

      double total_w = 0.0;
      for (uint64_t j = 0; j < sims->n; j++)
        total_w += sims->a[j];

      for (uint64_t j = 0; j < tmp->n; j++) {
        int64_t uid = tmp->a[j];
        double w = sims->a[j];

        uint32_t khi = tk_iumap_get(enc->id_to_idx, uid);
        if (khi == tk_iumap_end(enc->id_to_idx))
          continue;
        int64_t train_idx = tk_iumap_val(enc->id_to_idx, khi);
        if (train_idx < 0 || (uint64_t)train_idx >= enc->n_training_samples)
          continue;

        double *train_evec = enc->eigenvectors->a + (uint64_t)train_idx * enc->n_dims;
        for (uint64_t d = 0; d < enc->n_dims; d++) {
          double lambda = enc->eigenvalues->a[d];
          double denom = total_w - lambda;
          if (fabs(denom) > 1e-10) {
            sample_out[d] += w * train_evec[d] / denom;
          }
        }
      }
    }

    tk_ivec_destroy(tmp);
    tk_dvec_destroy(sims);
  }

  lua_rawgeti(L, LUA_REGISTRYINDEX, enc->threshold_fn_ref);
  lua_pushvalue(L, stack_before_out + 1);
  lua_pushinteger(L, (lua_Integer)enc->n_dims);
  lua_call(L, 2, 1);

  lua_replace(L, stack_before_out);
  lua_settop(L, stack_before_out);

  return 1;
}

static inline int tk_hlth_nystrom_encoder_lua(lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "features_index");
  tk_inv_t *feat_inv = tk_inv_peekopt(L, -1);
  tk_ann_t *feat_ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *feat_hbi = tk_hbi_peekopt(L, -1);
  lua_pop(L, 1);

  if (!feat_inv && !feat_ann && !feat_hbi)
    return luaL_error(L, "nystrom_encoder: features_index must be inv, ann, or hbi");

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

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);
  lua_pop(L, 1);
  if (!ids)
    return luaL_error(L, "nystrom_encoder: ids required");

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "nystrom_encoder", "n_dims");

  if (n_dims == 0)
    return luaL_error(L, "nystrom_encoder: n_dims must be > 0");
  if (eigenvalues->n != n_dims)
    return luaL_error(L, "nystrom_encoder: eigenvalues size (%llu) != n_dims (%llu)",
                      (unsigned long long)eigenvalues->n, (unsigned long long)n_dims);
  if (eigenvectors->n % n_dims != 0)
    return luaL_error(L, "nystrom_encoder: eigenvectors size (%llu) not divisible by n_dims (%llu)",
                      (unsigned long long)eigenvectors->n, (unsigned long long)n_dims);

  uint64_t n_training_samples = eigenvectors->n / n_dims;
  if (ids->n != n_training_samples)
    return luaL_error(L, "nystrom_encoder: ids size (%llu) != eigenvector rows (%llu)",
                      (unsigned long long)ids->n, (unsigned long long)n_training_samples);

  lua_getfield(L, 1, "threshold");
  if (!lua_isfunction(L, -1))
    return luaL_error(L, "nystrom_encoder: threshold must be a function");
  int threshold_ref = luaL_ref(L, LUA_REGISTRYINDEX);

  uint64_t k_neighbors = tk_lua_foptunsigned(L, 1, "nystrom_encoder", "k_neighbors", 0);
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
  uint64_t probe_radius = tk_lua_foptunsigned(L, 1, "nystrom_encoder", "probe_radius", 2);
  int64_t rank_filter = tk_lua_foptinteger(L, 1, "nystrom_encoder", "rank_filter", -1);

  tk_iumap_t *id_to_idx = tk_iumap_from_ivec(NULL, ids);
  if (!id_to_idx) {
    luaL_unref(L, LUA_REGISTRYINDEX, threshold_ref);
    return luaL_error(L, "nystrom_encoder: failed to create id_to_idx map");
  }

  tk_nystrom_encoder_t *enc = tk_lua_newuserdata(L, tk_nystrom_encoder_t, TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int Ei = lua_gettop(L);

  if (feat_inv) {
    enc->feat_idx = feat_inv;
    enc->feat_idx_type = TK_HLTH_IDX_INV;
  } else if (feat_ann) {
    enc->feat_idx = feat_ann;
    enc->feat_idx_type = TK_HLTH_IDX_ANN;
  } else {
    enc->feat_idx = feat_hbi;
    enc->feat_idx_type = TK_HLTH_IDX_HBI;
  }

  enc->eigenvectors = eigenvectors;
  enc->eigenvalues = eigenvalues;
  enc->ids = ids;
  enc->id_to_idx = id_to_idx;
  enc->n_dims = n_dims;
  enc->n_training_samples = n_training_samples;
  enc->k_neighbors = k_neighbors;
  enc->cmp = cmp;
  enc->cmp_alpha = cmp_alpha;
  enc->cmp_beta = cmp_beta;
  enc->probe_radius = probe_radius;
  enc->rank_filter = rank_filter;
  enc->threshold_fn_ref = threshold_ref;
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

  lua_getfield(L, 1, "ids");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "threshold");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_pushcclosure(L, tk_nystrom_encode_lua, 1);
  lua_pushinteger(L, (lua_Integer)n_dims);

  return 2;
}

static luaL_Reg tk_hlth_fns[] = {
  { "landmark_encoder", tk_hlth_landmark_encoder_lua },
  { "nystrom_encoder", tk_hlth_nystrom_encoder_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_hlth(lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_hlth_fns, 0);
  return 1;
}
