#include <assert.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/ivec.h>
#include <santoku/duset.h>
#include <santoku/cuset.h>
#include <santoku/iumap.h>
#include <santoku/zumap.h>

#define tk_umap_name tk_observed_doubles
#define tk_umap_key int64_t
#define tk_umap_value tk_duset_t *
#define tk_umap_eq(a, b) ((a) == (b))
#define tk_umap_hash(a) (kh_int64_hash_func(a))
#include <santoku/umap/tpl.h>

#define tk_umap_name tk_observed_strings
#define tk_umap_key int64_t
#define tk_umap_value tk_cuset_t *
#define tk_umap_eq(a, b) ((a) == (b))
#define tk_umap_hash(a) (kh_int64_hash_func(a))
#include <santoku/umap/tpl.h>

#define tk_umap_name tk_cont_thresholds
#define tk_umap_key int64_t
#define tk_umap_value tk_dvec_t *
#define tk_umap_eq(a, b) ((a) == (b))
#define tk_umap_hash(a) (kh_int64_hash_func(a))
#include <santoku/umap/tpl.h>

typedef struct { int64_t f; char *v; } tk_cat_bit_string_t;
typedef struct { int64_t f; double v; } tk_cat_bit_double_t;
#define tk_cat_bit_string_hash(k) (tk_hash_128(tk_hash_integer((k).f), tk_hash_string((k).v)))
#define tk_cat_bit_double_hash(k) (tk_hash_128(tk_hash_integer((k).f), tk_hash_double((k).v)))
#define tk_cat_bit_string_equal(a, b) ((a).f == (b).f && !strcmp((a).v, (b).v))
#define tk_cat_bit_double_equal(a, b) ((a).f == (b).f && (a).v == (b).v)

#define tk_umap_name tk_cat_bits_string
#define tk_umap_key tk_cat_bit_string_t
#define tk_umap_value int64_t
#define tk_umap_eq(a, b) (tk_cat_bit_string_equal(a, b))
#define tk_umap_hash(a) (tk_cat_bit_string_hash(a))
#include <santoku/umap/tpl.h>

#define tk_umap_name tk_cat_bits_double
#define tk_umap_key tk_cat_bit_double_t
#define tk_umap_value int64_t
#define tk_umap_eq(a, b) (tk_cat_bit_double_equal(a, b))
#define tk_umap_hash(a) (tk_cat_bit_double_hash(a))
#include <santoku/umap/tpl.h>

typedef struct {

  bool finalized;
  bool destroyed;
  uint64_t n_thresholds;
  uint64_t next_attr;
  uint64_t next_feature;

  tk_iuset_t *continuous;
  tk_iuset_t *categorical;

  tk_iumap_t *integer_features;
  tk_zumap_t *string_features;
  tk_observed_doubles_t *observed_doubles;
  tk_observed_strings_t *observed_strings;

  tk_cat_bits_string_t *cat_bits_string;
  tk_cat_bits_double_t *cat_bits_double;
  tk_iumap_t *cont_bits;
  tk_cont_thresholds_t *cont_thresholds;

} tk_booleanizer_t;

#define TK_BOOLEANIZER_MT "tk_booleanizer_t"
#define TK_BOOLEANIZER_EPH "tk_booleanizer_eph"

static inline tk_booleanizer_t *tk_booleanizer_peek (lua_State *L, int i)
{
  return (tk_booleanizer_t *) luaL_checkudata(L, i, TK_BOOLEANIZER_MT);
}

static inline bool tk_booleanizer_is_categorical (
  tk_booleanizer_t *B,
  int64_t id_feature
) {
  if (tk_iuset_contains(B->categorical, id_feature))
    return true;
  if (tk_iuset_contains(B->continuous, id_feature))
    return false;
  if (!B->finalized && B->observed_strings != NULL)
    return tk_observed_strings_get(B->observed_strings, id_feature) != tk_observed_strings_end(B->observed_strings);
  return false;
}

static inline int tk_booleanizer_encode_string (
  lua_State *L,
  tk_booleanizer_t *B,
  uint64_t id_sample,
  int64_t id_feature,
  const char *value,
  tk_ivec_t *out
) {
  if (!B->finalized) {
    tk_lua_verror(L, 2, "encode", "finalize must be called before encode");
    return -1;
  }
  tk_cat_bit_string_t key = { .f = id_feature, .v = (char *) value };
  khint_t k = tk_cat_bits_string_get(B->cat_bits_string, key);
  if (k != tk_cat_bits_string_end(B->cat_bits_string)) {
    int64_t bit_id = tk_cat_bits_string_val(B->cat_bits_string, k);
    if (tk_ivec_push(out, (int64_t) id_sample * (int64_t) B->next_feature + bit_id) != 0)
      return -1;
  }
  return 0;
}

static inline int tk_booleanizer_encode_double (
  lua_State *L,
  tk_booleanizer_t *B,
  uint64_t id_sample,
  int64_t id_feature,
  double value,
  tk_ivec_t *out
) {
  if (!B->finalized) {
    tk_lua_verror(L, 2, "encode", "finalize must be called before encode");
    return -1;
  }
  if (tk_booleanizer_is_categorical(B, id_feature)) {
    tk_cat_bit_double_t key = { .f = id_feature, .v = value };
    khint_t k = tk_cat_bits_double_get(B->cat_bits_double, key);
    if (k != tk_cat_bits_double_end(B->cat_bits_double)) {
      int64_t bit_id = tk_cat_bits_double_val(B->cat_bits_double, k);
      if (tk_ivec_push(out, (int64_t) id_sample * (int64_t) B->next_feature + bit_id) != 0)
        return -1;
    }
  } else {
    khint_t k_thresh = tk_cont_thresholds_get(B->cont_thresholds, id_feature);
    khint_t k_bits = tk_iumap_get(B->cont_bits, id_feature);
    if (k_thresh == tk_cont_thresholds_end(B->cont_thresholds) || k_bits == tk_iumap_end(B->cont_bits))
      return 0;
    tk_dvec_t *thresholds = tk_cont_thresholds_val(B->cont_thresholds, k_thresh);
    int64_t bit_base = tk_iumap_val(B->cont_bits, k_bits);
    for (uint64_t t = 0; t < thresholds->n; t ++)
      if (value > thresholds->a[t])
        if (tk_ivec_push(out, (int64_t) id_sample * (int64_t) B->next_feature + bit_base + (int64_t) t) != 0)
          return -1;
  }
  return 0;
}

static inline int tk_booleanizer_encode_integer (
  lua_State *L,
  tk_booleanizer_t *B,
  uint64_t id_sample,
  int64_t id_feature,
  int64_t value,
  tk_ivec_t *out
) {
  return tk_booleanizer_encode_double(L, B, id_sample, id_feature, (double) value, out);
}

static inline int tk_booleanizer_encode_dvec (
  lua_State *L,
  tk_booleanizer_t *B,
  tk_dvec_t *D,
  uint64_t n_dims,
  uint64_t id_dim0,
  tk_ivec_t *out
) {
  if (!B->finalized) {
    tk_lua_verror(L, 2, "encode", "finalize must be called before encode");
    return -1;
  }
  uint64_t n_samples = D->n / n_dims;
  for (uint64_t i = 0; i < n_samples; i ++) {
    for (uint64_t j = 0; j < n_dims; j ++) {
      double value = D->a[i * n_dims + j];
      int64_t id_feature = (int64_t) id_dim0 + (int64_t) j;
      if (tk_booleanizer_is_categorical(B, id_feature)) {
        tk_cat_bit_double_t key = { .f = id_feature, .v = value };
        khint_t k = tk_cat_bits_double_get(B->cat_bits_double, key);
        if (k != tk_cat_bits_double_end(B->cat_bits_double)) {
          int64_t bit_id = tk_cat_bits_double_val(B->cat_bits_double, k);
          if (tk_ivec_push(out, bit_id) != 0)
            return -1;
        }
      } else {
        khint_t k_thresh = tk_cont_thresholds_get(B->cont_thresholds, id_feature);
        khint_t k_bits = tk_iumap_get(B->cont_bits, id_feature);
        if (k_thresh == tk_cont_thresholds_end(B->cont_thresholds) || k_bits == tk_iumap_end(B->cont_bits))
          continue;
        tk_dvec_t *thresholds = tk_cont_thresholds_val(B->cont_thresholds, k_thresh);
        int64_t bit_base = tk_iumap_val(B->cont_bits, k_bits);
        for (uint64_t t = 0; t < thresholds->n; t ++)
          if (value >= thresholds->a[t])
            if (tk_ivec_push(out, (int64_t) i * (int64_t) B->next_feature + bit_base + (int64_t) t) != 0)
              return -1;
      }
    }
  }
  return 0;
}

static inline int64_t tk_booleanizer_bit_integer (
  lua_State *L,
  tk_booleanizer_t *B,
  int64_t feature,
  bool train
) {
  int kha;
  khint_t khi;
  khi = tk_iumap_get(B->integer_features, feature);
  if (khi == tk_iumap_end(B->integer_features) && !train) {
    return -1;
  } else if (train) {
    khi = tk_iumap_put(B->integer_features, feature, &kha);
    if (kha) {
      int64_t id_attr = (int64_t) B->next_attr ++;
      tk_iumap_setval(B->integer_features, khi, id_attr);
      return id_attr;
    } else {
      return tk_iumap_val(B->integer_features, khi);
    }
  } else {
    return tk_iumap_val(B->integer_features, khi);
  }
}

static inline int64_t tk_booleanizer_bit_string (
  lua_State *L,
  tk_booleanizer_t *B,
  const char *feature,
  bool train
) {
  int kha;
  khint_t khi;
  khi = tk_zumap_get(B->string_features, feature);
  if (khi == tk_zumap_end(B->string_features) && !train) {
    return -1;
  } else if (train) {
    khi = tk_zumap_put(B->string_features, feature, &kha);
    if (kha) {
      size_t len = strlen(feature);
      char *z = (char *) lua_newuserdata(L, len + 1);
      memcpy(z, feature, len + 1);
      tk_zumap_setkey(B->string_features, khi, z);
      tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, 1, -1);
      lua_pop(L, 1);
      int64_t id_attr = (int64_t) B->next_attr ++;
      tk_zumap_setval(B->string_features, khi, id_attr);
      return id_attr;
    } else {
      return tk_zumap_val(B->string_features, khi);
    }
  } else {
    return tk_zumap_val(B->string_features, khi);
  }
}

static inline void tk_booleanizer_observe_double (
  lua_State *L,
  tk_booleanizer_t *B,
  int64_t id_feature,
  double value
) {
  if (B->finalized) {
    tk_lua_verror(L, 2, "observe", "can't observe a new feature value for a finalized booleanizer");
    return;
  }
  if (id_feature >= (int64_t) B->next_attr)
    B->next_attr = (uint64_t) id_feature + 1;
  int kha;
  khint_t khi;
  tk_duset_t *obs;
  khi = tk_observed_doubles_put(B->observed_doubles, id_feature, &kha);
  if (kha) {
    obs = tk_duset_create(0, 0);
    tk_observed_doubles_setval(B->observed_doubles, khi, obs);
  } else
    obs = tk_observed_doubles_val(B->observed_doubles, khi);
  tk_duset_put(obs, value, &kha);
}

static inline void tk_booleanizer_observe_string (
  lua_State *L,
  tk_booleanizer_t *B,
  int64_t id_feature,
  const char *value
) {
  if (B->finalized) {
    tk_lua_verror(L, 2, "observe", "can't observe a new feature value for a finalized booleanizer");
    return;
  }
  if (id_feature >= (int64_t) B->next_attr)
    B->next_attr = (uint64_t) id_feature + 1;
  int kha;
  khint_t khi;
  tk_cuset_t *obs;
  khi = tk_observed_strings_put(B->observed_strings, id_feature, &kha);
  if (kha) {
    obs = tk_cuset_create(0, 0);
    tk_observed_strings_setval(B->observed_strings, khi, obs);
  } else
    obs = tk_observed_strings_val(B->observed_strings, khi);
  khi = tk_cuset_put(obs, value, &kha);
  if (kha) {
    size_t len = strlen(value);
    char *v = (char *) lua_newuserdata(L, len + 1);
    memcpy(v, value, len + 1);
    tk_cuset_setkey(obs, khi, v);
    tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, 1, -1);
    lua_pop(L, 1);
  }
}

static inline void tk_booleanizer_observe_integer (
  lua_State *L,
  tk_booleanizer_t *B,
  int64_t id_feature,
  int64_t value
) {
  if (B->finalized) {
    tk_lua_verror(L, 2, "observe", "can't observe a new feature value for a finalized booleanizer");
    return;
  }
  tk_booleanizer_observe_double(L, B, id_feature, (double) value);
}

static inline void tk_booleanizer_observe_dvec (
  lua_State *L,
  tk_booleanizer_t *B,
  tk_dvec_t *D,
  uint64_t n_dims,
  uint64_t id_dim0
) {
  uint64_t n = D->n / n_dims;
  for (uint64_t s = 0; s < n; s ++)
    for (uint64_t c = 0; c < n_dims; c ++)
      tk_booleanizer_observe_double(L, B, (int64_t) id_dim0 + (int64_t) c, D->a[s * n_dims + c]);
}

static inline void tk_booleanizer_add_thresholds (
  lua_State *L,
  tk_booleanizer_t *B,
  int Bi,
  int64_t id_feature,
  tk_dvec_t *value_vec
) {
  int kha;
  khint_t khi;
  uint64_t n = value_vec->n;
  tk_dvec_t *thresholds = tk_dvec_create(L, 0, 0, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, tk_lua_absindex(L, -1));
  lua_pop(L, 1);
  if (n == 0) {
  } else if (B->n_thresholds == 0) {
    double best_gap = -1.0, thr = 0.0;
    uint64_t best_idx = 0;
    #pragma omp parallel
    {
      double local_best_gap = -1.0;
      uint64_t local_best_idx = 0;
      #pragma omp for
      for (uint64_t i = 1; i < value_vec->n; i ++) {
        double gap = value_vec->a[i] - value_vec->a[i - 1];
        if (gap > local_best_gap) {
          local_best_gap = gap;
          local_best_idx = i;
        }
      }
      #pragma omp critical
      {
        if (local_best_gap > best_gap) {
          best_gap = local_best_gap;
          best_idx = local_best_idx;
        }
      }
    }
    if (best_gap > 0.0) {
      thr = 0.5 * (value_vec->a[best_idx] + value_vec->a[best_idx - 1]);
    } else {
      double sum = 0.0;
      for (uint64_t i = 0; i < value_vec->n; i ++)
        sum += value_vec->a[i];
      thr = sum / (double) value_vec->n;
    }
    if (tk_dvec_push(thresholds, thr) != 0) {
      tk_dvec_destroy(thresholds);
      tk_lua_verror(L, 2, "add_thresholds", "allocation failed");
      return;
    }
  } else if (n <= B->n_thresholds) {
    if (tk_dvec_copy(thresholds, value_vec, 0, (int64_t) value_vec->n, 0) != 0) {
      tk_dvec_destroy(thresholds);
      tk_lua_verror(L, 2, "add_thresholds", "allocation failed");
      return;
    }
  } else {
    for (uint64_t i = 0; i < B->n_thresholds; i ++) {
      double q = (double) (i + 1) / (double) (B->n_thresholds + 1);
      double idx = q * (n - 1);
      size_t left = (size_t) idx;
      size_t right = left + 1;
      double frac = idx - left;
      if (tk_dvec_push(thresholds, (right < n)
        ? value_vec->a[left] * (1.0 - frac) + value_vec->a[right] * frac
        : value_vec->a[left]) != 0) {
        tk_dvec_destroy(thresholds);
        tk_lua_verror(L, 2, "add_thresholds", "allocation failed");
        return;
      }
    }
  }
  tk_dvec_shrink(thresholds);
  khi = tk_cont_thresholds_put(B->cont_thresholds, id_feature, &kha);
  tk_cont_thresholds_setval(B->cont_thresholds, khi, thresholds);
  khi = tk_iumap_put(B->cont_bits, id_feature, &kha);
  tk_iumap_setval(B->cont_bits, khi, (int64_t) B->next_feature);
  B->next_feature += (uint64_t) thresholds->n;
}

static inline void tk_booleanizer_shrink (
  tk_booleanizer_t *B
) {
  if (B->destroyed)
    return;
  tk_duset_t *du;
  tk_cuset_t *cu;
  if (B->observed_doubles != NULL) {
    tk_umap_foreach_values(B->observed_doubles, du, ({
      tk_duset_destroy(du);
    }));
    tk_observed_doubles_destroy(B->observed_doubles);
    B->observed_doubles = NULL;
  }
  if (B->observed_strings != NULL) {
    tk_umap_foreach_values(B->observed_strings, cu, ({
      tk_cuset_destroy(cu);
    }))
    tk_observed_strings_destroy(B->observed_strings);
    B->observed_strings = NULL;
  }
}

static inline void tk_booleanizer_finalize (
  lua_State *L,
  tk_booleanizer_t *B,
  int Bi
) {
  if (B->finalized)
    return;
  if (B->destroyed) {
    tk_lua_verror(L, 2, "finalize", "can't finalize a destroyed booleanizer");
    return;
  }
  int kha;
  khint_t kho;
  tk_iuset_t *seen_features = tk_iuset_create(0, 0);
  tk_dvec_t *value_vec = tk_dvec_create(L, 0, 0, 0);
  int i_value_vec = tk_lua_absindex(L, -1);
  int64_t id_feature;
  tk_umap_foreach_keys(B->observed_doubles, id_feature, ({
    tk_iuset_put(seen_features, id_feature, &kha);
  }));
  tk_umap_foreach_keys(B->observed_strings, id_feature, ({
    tk_iuset_put(seen_features, id_feature, &kha);
  }));
  tk_umap_foreach_keys(seen_features, id_feature, ({
    if (tk_booleanizer_is_categorical(B, id_feature)) {
      tk_iuset_put(B->categorical, id_feature, &kha);
      kho = tk_observed_strings_get(B->observed_strings, id_feature);
      if (kho != tk_observed_strings_end(B->observed_strings)) {
        tk_cuset_t *values = tk_observed_strings_val(B->observed_strings, kho);
        const char *value;
        tk_umap_foreach_keys(values, value, ({
          tk_cat_bit_string_t key = { .f = id_feature, .v = (char *) value };
          khint_t outk = tk_cat_bits_string_put(B->cat_bits_string, key, &kha);
          tk_cat_bits_string_setval(B->cat_bits_string, outk, (int64_t) B->next_feature ++);
        }));
      }
      kho = tk_observed_doubles_get(B->observed_doubles, id_feature);
      if (kho != tk_observed_doubles_end(B->observed_doubles)) {
        tk_duset_t *values = tk_observed_doubles_val(B->observed_doubles, kho);
        double value;
        tk_umap_foreach_keys(values, value, ({
          tk_cat_bit_double_t key = { .f = id_feature, .v = value };
          khint_t outk = tk_cat_bits_double_put(B->cat_bits_double, key, &kha);
          tk_cat_bits_double_setval(B->cat_bits_double, outk, (int64_t) B->next_feature ++);
        }));
      }
    } else {
      tk_iuset_put(B->continuous, id_feature, &kha);
      kho = tk_observed_doubles_get(B->observed_doubles, id_feature);
      if (kho == tk_observed_doubles_end(B->observed_doubles))
        continue;
      tk_duset_t *value_set = tk_observed_doubles_val(B->observed_doubles, kho);
      tk_dvec_clear(value_vec);
      tk_duset_dump(value_set, value_vec);
      tk_dvec_asc(value_vec, 0, value_vec->n);
      tk_booleanizer_add_thresholds(L, B, Bi, id_feature, value_vec);
    }
  }));
  tk_booleanizer_shrink(B);
  B->finalized = true;
  lua_remove(L, i_value_vec);
  tk_iuset_destroy(seen_features);
}

static inline int64_t tk_booleanizer_attributes (
  lua_State *L,
  tk_booleanizer_t *B
) {
  if (B->destroyed) {
    tk_lua_verror(L, 2, "attributes", "can't query a destroyed booleanizer");
    return -1;
  }
  if (!B->finalized) {
    tk_lua_verror(L, 2, "attributes", "finalize must be called before attributes");
    return -1;
  }
  return (int64_t) B->next_attr;
}

static inline int64_t tk_booleanizer_features (
  lua_State *L,
  tk_booleanizer_t *B
) {
  if (B->destroyed) {
    tk_lua_verror(L, 2, "features", "can't query a destroyed booleanizer");
    return -1;
  }
  if (!B->finalized) {
    tk_lua_verror(L, 2, "features", "finalize must be called before features");
    return -1;
  }
  return (int64_t) B->next_feature;
}

static inline void tk_booleanizer_restrict (
  lua_State *L,
  tk_booleanizer_t *B,
  tk_ivec_t *ids,
  int Bi
) {
  if (B->destroyed) {
    tk_lua_verror(L, 2, "restrict", "can't restrict a destroyed booleanizer");
    return;
  }
  if (!B->finalized) {
    tk_lua_verror(L, 2, "restrict", "finalize must be called before restrict");
    return;
  }
  tk_iumap_t *feature_id_map = tk_iumap_create(0, 0);
  khint_t khi;
  int kha;
  for (int64_t i = 0; i < (int64_t) ids->n; i ++) {
    int64_t old_id = ids->a[i];
    khi = tk_iumap_put(feature_id_map, old_id, &kha);
    if (kha)
      tk_iumap_setval(feature_id_map, khi, i);
  }
  tk_iuset_t *new_continuous   = tk_iuset_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_iuset_t *new_categorical  = tk_iuset_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_iumap_t *new_integer_features = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_zumap_t *new_string_features  = tk_zumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_cat_bits_string_t *new_cat_bits_string = tk_cat_bits_string_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_cat_bits_double_t *new_cat_bits_double = tk_cat_bits_double_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_iumap_t *new_cont_bits = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_cont_thresholds_t *new_cont_thresholds = tk_cont_thresholds_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  int64_t next_bit = 0;
  for (int64_t new_f = 0; new_f < (int64_t) ids->n; new_f ++) {
    int64_t old_f = ids->a[new_f];
    int kha;
    if (tk_iuset_contains(B->continuous, old_f))
      tk_iuset_put(new_continuous, new_f, &kha);
    if (tk_iuset_contains(B->categorical, old_f))
      tk_iuset_put(new_categorical, new_f, &kha);
    khint_t k = tk_iumap_get(B->integer_features, old_f);
    if (k != tk_iumap_end(B->integer_features)) {
      khi = tk_iumap_put(new_integer_features, new_f, &kha);
      if (kha)
        tk_iumap_setval(new_integer_features, khi, tk_iumap_val(B->integer_features, k));
    }
    k = tk_zumap_end(B->string_features);
    const char *z;
    int64_t v;
    tk_umap_foreach(B->string_features, z, v, ({
      if (v == old_f)
        k = tk_zumap_get(B->string_features, z);
    }));
    if (k != tk_zumap_end(B->string_features)) {
      const char *old_z = tk_zumap_key(B->string_features, k);
      size_t len = strlen(old_z);
      char *z = (char *) lua_newuserdata(L, len + 1);
      memcpy(z, old_z, len + 1);
      khi = tk_zumap_put(new_string_features, z, &kha);
      if (kha)
        tk_zumap_setval(new_string_features, khi, new_f);
      tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
      lua_pop(L, 1);
    }
    tk_cat_bit_string_t cbs;
    int64_t new_bit;
    tk_umap_foreach(B->cat_bits_string, cbs, v, ({
      if (cbs.f == old_f) {
        size_t len = strlen(cbs.v);
        char *new_v = (char *) lua_newuserdata(L, len + 1);
        memcpy(new_v, cbs.v, len + 1);
        tk_cat_bit_string_t new_key = { .f = new_f, .v = new_v };
        int kha;
        khint_t nk = tk_cat_bits_string_put(new_cat_bits_string, new_key, &kha);
        new_bit = next_bit ++;
        tk_cat_bits_string_setval(new_cat_bits_string, nk, new_bit);
        tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
        lua_pop(L, 1);
      }
    }));
    tk_cat_bit_double_t cbd;
    tk_umap_foreach(B->cat_bits_double, cbd, v, ({
      if (cbd.f == old_f) {
        tk_cat_bit_double_t new_key = { .f = new_f, .v = cbd.v };
        int kha;
        khint_t nk = tk_cat_bits_double_put(new_cat_bits_double, new_key, &kha);
        new_bit = next_bit ++;
        tk_cat_bits_double_setval(new_cat_bits_double, nk, new_bit);
      }
    }));
    k = tk_iumap_get(B->cont_bits, old_f);
    if (k != tk_iumap_end(B->cont_bits)) {
      khi = tk_iumap_put(new_cont_bits, new_f, &kha);
      if (kha)
        tk_iumap_setval(new_cont_bits, khi, next_bit);
      khint_t kth = tk_cont_thresholds_get(B->cont_thresholds, old_f);
      if (kth != tk_cont_thresholds_end(B->cont_thresholds)) {
        tk_dvec_t *old_vec = tk_cont_thresholds_val(B->cont_thresholds, kth);
        tk_dvec_t *new_vec = tk_dvec_create(L, old_vec->n, 0, 0);
        tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
        lua_pop(L, 1);
        memcpy(new_vec->a, old_vec->a, sizeof(double) * old_vec->n);
        new_vec->n = old_vec->n;
        int kha;
        khint_t nk = tk_cont_thresholds_put(new_cont_thresholds, new_f, &kha);
        tk_cont_thresholds_setval(new_cont_thresholds, nk, new_vec);
      }
      next_bit += (int64_t) B->n_thresholds;
    }
  }
  tk_lua_del_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, B->continuous);
  tk_iuset_destroy(B->continuous);
  tk_lua_del_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, B->categorical);
  tk_iuset_destroy(B->categorical);
  tk_lua_del_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, B->integer_features);
  tk_iumap_destroy(B->integer_features);
  tk_lua_del_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, B->string_features);
  tk_zumap_destroy(B->string_features);
  tk_lua_del_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, B->cat_bits_string);
  tk_cat_bits_string_destroy(B->cat_bits_string);
  tk_lua_del_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, B->cat_bits_double);
  tk_cat_bits_double_destroy(B->cat_bits_double);
  tk_lua_del_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, B->cont_bits);
  tk_iumap_destroy(B->cont_bits);
  tk_lua_del_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, B->cont_thresholds);
  tk_cont_thresholds_destroy(B->cont_thresholds);
  tk_iumap_destroy(feature_id_map);
  B->continuous = new_continuous;
  B->categorical = new_categorical;
  B->integer_features = new_integer_features;
  B->string_features = new_string_features;
  B->cat_bits_string = new_cat_bits_string;
  B->cat_bits_double = new_cat_bits_double;
  B->cont_bits = new_cont_bits;
  B->cont_thresholds = new_cont_thresholds;
  B->next_attr = ids->n;
  B->next_feature = (uint64_t) next_bit;
}

static inline void tk_booleanizer_destroy (
  tk_booleanizer_t *B
) {
  if (B->destroyed)
    return;
  tk_booleanizer_shrink(B);
  memset(B, 0, sizeof(*B));
  B->finalized = false;
  B->destroyed = true;
}

static inline void tk_booleanizer_persist (
  lua_State *L,
  tk_booleanizer_t *B,
  FILE *fh
) {
  if (B->destroyed) {
    tk_lua_verror(L, 2, "persist", "can't persist a destroyed booleanizer");
    return;
  }
  if (!B->finalized) {
    tk_lua_verror(L, 2, "persist", "finalize must be called before persist");
    return;
  }
  tk_lua_fwrite(L, (char *) &B->finalized, sizeof(bool), 1, fh);
  tk_lua_fwrite(L, (char *) &B->n_thresholds, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &B->next_attr, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &B->next_feature, sizeof(uint64_t), 1, fh);
  khint_t sz;
  int64_t f;
  tk_iuset_persist(L, B->continuous, fh);
  tk_iuset_persist(L, B->categorical, fh);
  tk_iumap_persist(L, B->integer_features, fh);
  sz = tk_zumap_size(B->string_features);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  const char *z;
  tk_umap_foreach(B->string_features, z, f, ({
    size_t len = strlen(z);
    tk_lua_fwrite(L, (char *) &len, sizeof(size_t), 1, fh);
    tk_lua_fwrite(L, (char *) z, len, 1, fh);
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
  }));
  sz = tk_cat_bits_string_size(B->cat_bits_string);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_cat_bit_string_t cbs;
  tk_umap_foreach(B->cat_bits_string, cbs, f, ({
    tk_lua_fwrite(L, (char *) &cbs.f, sizeof(int64_t), 1, fh);
    size_t len = strlen(cbs.v);
    tk_lua_fwrite(L, (char *) &len, sizeof(size_t), 1, fh);
    tk_lua_fwrite(L, cbs.v, len, 1, fh);
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
  }));
  sz = tk_cat_bits_double_size(B->cat_bits_double);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_cat_bit_double_t cbd;
  tk_umap_foreach(B->cat_bits_double, cbd, f, ({
    tk_lua_fwrite(L, (char *) &cbd.f, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, (char *) &cbd.v, sizeof(double),  1, fh);
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
  }));
  tk_iumap_persist(L, B->cont_bits, fh);
  sz = tk_cont_thresholds_size(B->cont_thresholds);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_dvec_t *thresholds;
  tk_umap_foreach(B->cont_thresholds, f, thresholds, ({
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, (char *) &thresholds->n, sizeof(size_t), 1, fh);
    tk_lua_fwrite(L, (char *) thresholds->a, sizeof(double), thresholds->n, fh);
  }));
}

static inline int tk_booleanizer_gc_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  tk_booleanizer_destroy(B);
  return 0;
}

static inline int tk_booleanizer_bit_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  bool train = lua_isboolean(L, 3) ? lua_toboolean(L, 3) : false;
  int64_t id;
  int t = lua_type(L, 2);
  switch (t) {
    case LUA_TNUMBER:
      id = tk_booleanizer_bit_integer(L, B, lua_tointeger(L, 2), train);
      if (id < 0)
        return 0;
      lua_pushinteger(L, id);
      return 1;
    case LUA_TSTRING:
      id = tk_booleanizer_bit_string(L, B, lua_tostring(L, 2), train);
      if (id < 0)
        return 0;
      lua_pushinteger(L, id);
      return 1;
    default:
      tk_lua_verror(L, 3, "bit", "unexpected type", lua_typename(L, t));
      return 0;
  }
}

static inline int tk_booleanizer_encode_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  tk_dvec_t *D = tk_dvec_peekopt(L, 2);
  tk_ivec_t *out = NULL;
  if (D != NULL) {
    uint64_t n_dims = tk_lua_checkunsigned(L, 3, "n_dims");
    uint64_t id_dim0 = tk_lua_optunsigned(L, 4, "id_dim0", 0);
    out = tk_ivec_peekopt(L, 5);
    if (out == NULL)
      out = tk_ivec_create(L, 0, 0, 0);
    else
      lua_pushvalue(L, 5);
    if (tk_booleanizer_encode_dvec(L, B, D, n_dims, id_dim0, out) != 0)
      return 0;
    return 1;
  } else {
    uint64_t id_sample = tk_lua_checkunsigned(L, 2, "id_sample");
    lua_pushcfunction(L, tk_booleanizer_bit_lua);
    lua_pushvalue(L, 1);
    lua_pushvalue(L, 3);
    lua_call(L, 2, 1);
    if (lua_isnoneornil(L, -1)) {
      lua_pop(L, 1);
      out = tk_ivec_peekopt(L, 5);
      if (out == NULL)
        out = tk_ivec_create(L, 0, 0, 0);
      else
        lua_pushvalue(L, 5);
      return 1;
    }
    int64_t id_feature = (int64_t) tk_lua_checkunsigned(L, -1, "id_feature");
    out = tk_ivec_peekopt(L, 5);
    if (out == NULL)
      out = tk_ivec_create(L, 0, 0, 0);
    else
      lua_pushvalue(L, 5);
    int t = lua_type(L, 4);
    int rc = 0;
    switch (t) {
      case LUA_TSTRING:
        rc = tk_booleanizer_encode_string(L, B, id_sample, id_feature, luaL_checkstring(L, 4), out);
        break;
      case LUA_TNUMBER:
        rc = tk_booleanizer_encode_double(L, B, id_sample, id_feature, luaL_checknumber(L, 4), out);
        break;
      case LUA_TBOOLEAN:
        rc = tk_booleanizer_encode_integer(L, B, id_sample, id_feature, lua_toboolean(L, 4), out);
        break;
      case LUA_TNIL:
        break;
      default:
        tk_lua_verror(L, 2, "encode", "unexpected type passed to encode", lua_typename(L, t));
        break;
    }
    if (rc != 0)
      return 0;
    return 1;
  }
}

static inline int tk_booleanizer_observe_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  tk_dvec_t *D = tk_dvec_peekopt(L, 2);
  if (D != NULL) {
    uint64_t n_dims = tk_lua_checkunsigned(L, 3, "n_dims");
    uint64_t id_dim0 = tk_lua_optunsigned(L, 4, "id_dim0", 0);
    tk_booleanizer_observe_dvec(L, B, D, n_dims, id_dim0);
    lua_pushinteger(L, (int64_t) id_dim0);
    lua_pushinteger(L, (int64_t) id_dim0 + (int64_t) n_dims - 1);
    return 2;
  } else {
    int nargs = lua_gettop(L);
    int feature_arg, value_arg;
    if (nargs >= 3 && !lua_isnil(L, 3)) {
      feature_arg = 2;
      value_arg = 3;
    } else {
      feature_arg = 2;
      value_arg = 2;
    }
    lua_pushcfunction(L, tk_booleanizer_bit_lua);
    lua_pushvalue(L, 1);
    lua_pushvalue(L, feature_arg);
    lua_pushboolean(L, true);
    lua_call(L, 3, 1);
    int64_t id_feature = (int64_t) tk_lua_checkunsigned(L, -1, "id_feature");
    int t = lua_type(L, value_arg);
    switch (t) {
      case LUA_TSTRING:
        tk_booleanizer_observe_string(L, B, id_feature, luaL_checkstring(L, value_arg));
        break;
      case LUA_TNUMBER:
        tk_booleanizer_observe_double(L, B, id_feature, luaL_checknumber(L, value_arg));
        break;
      case LUA_TBOOLEAN:
        tk_booleanizer_observe_integer(L, B, id_feature, lua_toboolean(L, value_arg));
        break;
      case LUA_TNIL:
        break;
      default:
        tk_lua_verror(L, 2, "observe", "unexpected type passed to observe", lua_typename(L, t));
        break;
    }
    return 1;
  }
}

static inline int tk_booleanizer_finalize_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  tk_booleanizer_finalize(L, B, 1);
  return 0;
}

static inline int tk_booleanizer_attributes_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  lua_pushinteger(L, tk_booleanizer_attributes(L, B));
  return 1;
}

static inline int tk_booleanizer_features_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  lua_pushinteger(L, tk_booleanizer_features(L, B));
  return 1;
}

static inline int tk_booleanizer_restrict_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  tk_ivec_t *ids = tk_ivec_peek(L, 2, "top_v");
  tk_booleanizer_restrict(L, B, ids, 1);
  return 0;
}

static inline int tk_booleanizer_persist_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_booleanizer_persist(L, B, fh);
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

static inline int tk_booleanizer_destroy_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  tk_booleanizer_destroy(B);
  return 0;
}

static luaL_Reg tk_booleanizer_mt_fns[] =
{
  { "observe", tk_booleanizer_observe_lua },
  { "encode", tk_booleanizer_encode_lua },
  { "attributes", tk_booleanizer_attributes_lua },
  { "features", tk_booleanizer_features_lua },
  { "bit", tk_booleanizer_bit_lua },
  { "finalize", tk_booleanizer_finalize_lua },
  { "restrict", tk_booleanizer_restrict_lua },
  { "persist", tk_booleanizer_persist_lua },
  { "destroy", tk_booleanizer_destroy_lua },
  { NULL, NULL }
};

static inline tk_booleanizer_t *tk_booleanizer_create (
  lua_State *L,
  uint64_t n_thresholds,
  tk_ivec_t *continuous,
  tk_ivec_t *categorical
) {
  tk_booleanizer_t *B = tk_lua_newuserdata(L, tk_booleanizer_t, TK_BOOLEANIZER_MT, tk_booleanizer_mt_fns, tk_booleanizer_gc_lua);
  int Bi = lua_gettop(L);
  B->continuous = (continuous != NULL) ? tk_iuset_from_ivec(L, continuous) : tk_iuset_create(L, 0);
  if (!B->continuous)
    tk_error(L, "booleanizer: iuset_from_ivec failed", ENOMEM);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->categorical = (categorical != NULL) ? tk_iuset_from_ivec(L, categorical) : tk_iuset_create(L, 0);
  if (!B->categorical)
    tk_error(L, "booleanizer: iuset_from_ivec failed", ENOMEM);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->n_thresholds = n_thresholds;
  B->integer_features = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->string_features = tk_zumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->observed_doubles = tk_observed_doubles_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->observed_strings = tk_observed_strings_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->cat_bits_string = tk_cat_bits_string_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->cat_bits_double = tk_cat_bits_double_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->cont_bits = tk_iumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->cont_thresholds = tk_cont_thresholds_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->next_attr = 0;
  B->next_feature = 0;
  B->finalized = false;
  B->destroyed = false;
  return B;
}

static inline tk_booleanizer_t *tk_booleanizer_load (
  lua_State *L,
  FILE *fh
) {
  tk_booleanizer_t *B = tk_lua_newuserdata(L, tk_booleanizer_t, TK_BOOLEANIZER_MT, tk_booleanizer_mt_fns, tk_booleanizer_gc_lua);
  int Bi = lua_gettop(L);
  memset(B, 0, sizeof(*B));
  tk_lua_fread(L, (char *) &B->finalized, sizeof(bool), 1, fh);
  tk_lua_fread(L, (char *) &B->n_thresholds, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, (char *) &B->next_attr, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, (char *) &B->next_feature, sizeof(uint64_t), 1, fh);
  khint_t khi, sz;
  int kha;
  int64_t f;
  B->continuous = tk_iuset_load(L, fh);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->categorical = tk_iuset_load(L, fh);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->integer_features = tk_iumap_load(L, fh);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->string_features = tk_zumap_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  size_t len;
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    tk_lua_fread(L, (char *) &len, sizeof(size_t), 1, fh);
    char *z = (char *) lua_newuserdata(L, len + 1);
    tk_lua_fread(L, z, len, 1, fh);
    z[len] = '\0';
    tk_lua_fread(L, (char *) &f, sizeof(int64_t), 1, fh);
    khi = tk_zumap_put(B->string_features, z, &kha);
    if (kha)
      tk_zumap_setval(B->string_features, khi, f);
    tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
    lua_pop(L, 1);
  }
  B->cat_bits_string = tk_cat_bits_string_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    int64_t feature_id;
    tk_lua_fread(L, (char *) &feature_id, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, (char *) &len, sizeof(size_t), 1, fh);
    char *value = (char *) lua_newuserdata(L, len + 1);
    tk_lua_fread(L, value, len, 1, fh);
    value[len] = '\0';
    int64_t bit_id;
    tk_lua_fread(L, (char *) &bit_id, sizeof(int64_t), 1, fh);
    tk_cat_bit_string_t key = { .f = feature_id, .v = value };
    khint_t k = tk_cat_bits_string_put(B->cat_bits_string, key, &kha);
    tk_cat_bits_string_setval(B->cat_bits_string, k, bit_id);
    tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
    lua_pop(L, 1);
  }
  B->cat_bits_double = tk_cat_bits_double_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    int64_t feature_id;
    double dval;
    int64_t bit_id;
    tk_lua_fread(L, (char *) &feature_id, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, (char *) &dval, sizeof(double), 1, fh);
    tk_lua_fread(L, (char *) &bit_id, sizeof(int64_t), 1, fh);
    tk_cat_bit_double_t key = { .f = feature_id, .v = dval };
    khint_t k = tk_cat_bits_double_put(B->cat_bits_double, key, &kha);
    tk_cat_bits_double_setval(B->cat_bits_double, k, bit_id);
  }
  B->cont_bits = tk_iumap_load(L, fh);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  B->cont_thresholds = tk_cont_thresholds_create(L, 0);
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
  lua_pop(L, 1);
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    int64_t feature_id;
    tk_lua_fread(L, (char *) &feature_id, sizeof(int64_t), 1, fh);
    size_t n_thresh;
    tk_lua_fread(L, (char *) &n_thresh, sizeof(size_t), 1, fh);
    tk_dvec_t *thresh_vec = tk_dvec_create(L, n_thresh, 0, 0);
    tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPH, Bi, -1);
    lua_pop(L, 1);
    tk_lua_fread(L, (char *) thresh_vec->a, sizeof(double), n_thresh, fh);
    thresh_vec->n = n_thresh;
    khint_t k = tk_cont_thresholds_put(B->cont_thresholds, feature_id, &kha);
    tk_cont_thresholds_setval(B->cont_thresholds, k, thresh_vec);
  }
  B->destroyed = false;
  return B;
}

static inline int tk_booleanizer_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  if (lua_isnil(L, 1))
    lua_newtable(L);
  uint64_t n_thresholds = tk_lua_foptunsigned(L, 1, "create", "n_thresholds", 0);
  lua_getfield(L, 1, "continuous");
  tk_ivec_t *continuous = lua_isnil(L, -1) ? NULL : tk_ivec_peekopt(L, -1);
  lua_getfield(L, 1, "categorical");
  tk_ivec_t *categorical = lua_isnil(L, -1) ? NULL : tk_ivec_peekopt(L, -1);
  tk_booleanizer_create(L, n_thresholds, continuous, categorical);
  return 1;
}

static inline int tk_booleanizer_load_lua (lua_State *L)
{
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr ? tk_lua_fmemopen(L, (char *) data, len, "r") : tk_lua_fopen(L, data, "r");
  tk_booleanizer_load(L, fh);
  tk_lua_fclose(L, fh);
  return 1;
}

static luaL_Reg tk_booleanizer_fns[] =
{
  { "create", tk_booleanizer_create_lua },
  { "load", tk_booleanizer_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_booleanizer (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_booleanizer_fns, 0);
  return 1;
}
