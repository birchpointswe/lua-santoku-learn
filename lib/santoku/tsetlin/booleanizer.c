#include <assert.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/duset.h>
#include <santoku/cuset.h>
#include <santoku/iumap.h>
#include <santoku/zumap.h>

KHASH_INIT(tk_observed_doubles, int64_t, tk_duset_t *, 1, kh_int64_hash_func, kh_int64_hash_equal)
typedef khash_t(tk_observed_doubles) tk_observed_doubles_t;

KHASH_INIT(tk_observed_strings, int64_t, tk_cuset_t *, 1, kh_int64_hash_func, kh_int64_hash_equal)
typedef khash_t(tk_observed_strings) tk_observed_strings_t;

KHASH_INIT(tk_cont_thresholds, int64_t, tk_dvec_t *, 1, kh_int64_hash_func, kh_int64_hash_equal)
typedef khash_t(tk_cont_thresholds) tk_cont_thresholds_t;

typedef struct { int64_t f; char *v; } tk_cat_bit_string_t;
typedef struct { int64_t f; double v; } tk_cat_bit_double_t;
#define tk_cat_bit_string_hash(k) (tk_hash_128(tk_hash_integer((k).f), tk_hash_string((k).v)))
#define tk_cat_bit_double_hash(k) (tk_hash_128(tk_hash_integer((k).f), tk_hash_double((k).v)))
#define tk_cat_bit_string_equal(a, b) ((a).f == (b).f && !strcmp((a).v, (b).v))
#define tk_cat_bit_double_equal(a, b) ((a).f == (b).f && (a).v == (b).v)
KHASH_INIT(tk_cat_bits_string, tk_cat_bit_string_t, int64_t, 1, tk_cat_bit_string_hash, tk_cat_bit_string_equal)
KHASH_INIT(tk_cat_bits_double, tk_cat_bit_double_t, int64_t, 1, tk_cat_bit_double_hash, tk_cat_bit_double_equal)
typedef khash_t(tk_cat_bits_string) tk_cat_bits_string_t;
typedef khash_t(tk_cat_bits_double) tk_cat_bits_double_t;

typedef struct {

  bool finalized;
  bool destroyed;
  uint64_t n_thresholds;
  uint64_t next_feature;
  uint64_t next_bit;

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
#define TK_BOOLEANIZER_EPHEMERON "tk_booleanizer_ephemeron"

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
    return kh_get(tk_observed_strings, B->observed_strings, id_feature) != kh_end(B->observed_strings);
  return false;
}

static inline void tk_booleanizer_encode_string (
  lua_State *L,
  tk_booleanizer_t *B,
  uint64_t id_sample,
  int64_t id_feature,
  const char *value,
  tk_ivec_t *out
) {
  if (!B->finalized) {
    tk_lua_verror(L, 2, "encode", "finalize must be called before encode");
    return;
  }
  tk_cat_bit_string_t key = { .f = id_feature, .v = (char *) value };
  khint_t k = kh_get(tk_cat_bits_string, B->cat_bits_string, key);
  if (k != kh_end(B->cat_bits_string)) {
    int64_t bit_id = kh_value(B->cat_bits_string, k);
    tk_ivec_push(out, (int64_t) id_sample * (int64_t) B->next_bit + bit_id);
  }
}

static inline void tk_booleanizer_encode_double (
  lua_State *L,
  tk_booleanizer_t *B,
  uint64_t id_sample,
  int64_t id_feature,
  double value,
  tk_ivec_t *out
) {
  if (!B->finalized) {
    tk_lua_verror(L, 2, "encode", "finalize must be called before encode");
    return;
  }
  if (tk_booleanizer_is_categorical(B, id_feature)) {
    tk_cat_bit_double_t key = { .f = id_feature, .v = value };
    khint_t k = kh_get(tk_cat_bits_double, B->cat_bits_double, key);
    if (k != kh_end(B->cat_bits_double)) {
      int64_t bit_id = kh_value(B->cat_bits_double, k);
      tk_ivec_push(out, (int64_t) id_sample * (int64_t) B->next_bit + bit_id);
    }
  } else {
    khint_t k_thresh = kh_get(tk_cont_thresholds, B->cont_thresholds, id_feature);
    khint_t k_bits = kh_get(tk_iumap, B->cont_bits, id_feature);
    if (k_thresh == kh_end(B->cont_thresholds) || k_bits == kh_end(B->cont_bits))
      return; // no thresholds/bit mapping for this feature
    tk_dvec_t *thresholds = kh_value(B->cont_thresholds, k_thresh);
    int64_t bit_base = kh_value(B->cont_bits, k_bits);
    for (uint64_t t = 0; t < thresholds->n; t ++)
      if (value > thresholds->a[t])
        tk_ivec_push(out, (int64_t) id_sample * (int64_t) B->next_bit + bit_base + (int64_t) t);
  }
}

static inline void tk_booleanizer_encode_integer (
  lua_State *L,
  tk_booleanizer_t *B,
  uint64_t id_sample,
  int64_t id_feature,
  int64_t value,
  tk_ivec_t *out
) {
  tk_booleanizer_encode_double(L, B, id_sample, id_feature, (double) value, out);
}

static inline void tk_booleanizer_encode_dvec (
  lua_State *L,
  tk_booleanizer_t *B,
  tk_dvec_t *D,
  uint64_t n_dims,
  uint64_t id_dim0,
  tk_ivec_t *out
) {
  if (!B->finalized) {
    tk_lua_verror(L, 2, "encode", "finalize must be called before encode");
    return;
  }
  uint64_t n_samples = D->n / n_dims;
  for (uint64_t i = 0; i < n_samples; i ++) {
    for (uint64_t j = 0; j < n_dims; j ++) {
      double value = D->a[i * n_dims + j];
      int64_t id_feature = (int64_t) id_dim0 + (int64_t) j;
      if (tk_booleanizer_is_categorical(B, id_feature)) {
        tk_cat_bit_double_t key = { .f = id_feature, .v = value };
        khint_t k = kh_get(tk_cat_bits_double, B->cat_bits_double, key);
        if (k != kh_end(B->cat_bits_double)) {
          int64_t bit_id = kh_value(B->cat_bits_double, k);
          tk_ivec_push(out, bit_id);
        }
      } else {
        khint_t k_thresh = kh_get(tk_cont_thresholds, B->cont_thresholds, id_feature);
        khint_t k_bits = kh_get(tk_iumap, B->cont_bits, id_feature);
        if (k_thresh == kh_end(B->cont_thresholds) || k_bits == kh_end(B->cont_bits))
          continue; // should not happen
        tk_dvec_t *thresholds = kh_value(B->cont_thresholds, k_thresh);
        int64_t bit_base = kh_value(B->cont_bits, k_bits);
        for (uint64_t t = 0; t < thresholds->n; t ++)
          if (value >= thresholds->a[t])
            tk_ivec_push(out, (int64_t) i * (int64_t) B->next_bit + bit_base + (int64_t) t);
      }
    }
  }
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
  if (khi == kh_end(B->integer_features) && !train) {
    return -1;
  } else if (train) {
    khi = tk_iumap_put(B->integer_features, feature, &kha);
    int64_t id_feature = (int64_t) B->next_feature ++;
    tk_iumap_value(B->integer_features, khi) = id_feature;
    return id_feature;
  } else {
    return tk_iumap_value(B->integer_features, khi);
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
  if (khi == kh_end(B->string_features) && !train) {
    return -1;
  } else if (train) {
    khi = tk_zumap_put(B->string_features, feature, &kha);
    if (kha)
      tk_zumap_key(B->string_features, khi) = strdup(feature);
    int64_t id_feature = (int64_t) B->next_feature ++;
    tk_zumap_value(B->string_features, khi) = id_feature;
    return id_feature;
  } else {
    return tk_zumap_value(B->string_features, khi);
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
  if (id_feature >= (int64_t) B->next_feature)
    B->next_feature = (uint64_t) id_feature + 1;
  int kha;
  khint_t khi;
  tk_duset_t *obs;
  khi = kh_put(tk_observed_doubles, B->observed_doubles, id_feature, &kha);
  if (kha)
    obs = kh_value(B->observed_doubles, khi) = tk_duset_create();
  else
    obs = kh_value(B->observed_doubles, khi);
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
  if (id_feature >= (int64_t) B->next_feature)
    B->next_feature = (uint64_t) id_feature + 1;
  int kha;
  khint_t khi;
  tk_cuset_t *obs;
  khi = kh_put(tk_observed_strings, B->observed_strings, id_feature, &kha);
  if (kha)
    obs = kh_value(B->observed_strings, khi) = tk_cuset_create();
  else
    obs = kh_value(B->observed_strings, khi);
  khi = tk_cuset_put(obs, value, &kha);
  if (kha)
    tk_cuset_key(obs, khi) = strdup(value);
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
  tk_lua_add_ephemeron(L, TK_BOOLEANIZER_EPHEMERON, Bi, tk_lua_absindex(L, -1));
  lua_pop(L, 1);
  if (n == 0) {
    // Nothing to do
  } else if (B->n_thresholds == 0) {
    // Otso
    double best_gap = -1.0, thr = 0.0;
    for (uint64_t i = 1; i < value_vec->n; i ++) {
      double gap = value_vec->a[i] - value_vec->a[i - 1];
      if (gap > best_gap) {
        best_gap = gap;
        thr = 0.5 * (value_vec->a[i] + value_vec->a[i - 1]);
      }
    }
    if (best_gap <= 0.0) {
      double sum = 0.0;
      for (uint64_t i = 0; i < value_vec->n; i ++)
        sum += value_vec->a[i];
      thr = sum / (double) value_vec->n;
    }
    tk_dvec_push(thresholds, thr);
  } else if (n <= B->n_thresholds) {
    tk_dvec_copy(thresholds, value_vec, 0, (int64_t) value_vec->n, 0);
  } else {
    for (uint64_t i = 0; i < B->n_thresholds; i ++) {
      double q = (double) (i + 1) / (double) (B->n_thresholds + 1);
      double idx = q * (n - 1);
      size_t left = (size_t) idx;
      size_t right = left + 1;
      double frac = idx - left;
      tk_dvec_push(thresholds, (right < n)
        ? value_vec->a[left] * (1.0 - frac) + value_vec->a[right] * frac
        : value_vec->a[left]);
    }
  }
  tk_dvec_shrink(thresholds);
  khi = kh_put(tk_cont_thresholds, B->cont_thresholds, id_feature, &kha);
  kh_value(B->cont_thresholds, khi) = thresholds;
  khi = kh_put(tk_iumap, B->cont_bits, id_feature, &kha);
  kh_value(B->cont_bits, khi) = (int64_t) B->next_bit;
  B->next_bit += (uint64_t) thresholds->n;
}

static inline void tk_booleanizer_shrink (
  tk_booleanizer_t *B
) {
  if (B->destroyed)
    return;
  tk_duset_t *du;
  tk_cuset_t *cu;
  int64_t f;
  if (B->observed_doubles != NULL) {
    kh_foreach(B->observed_doubles, f, du, ({
      tk_duset_destroy(du);
    }));
    kh_destroy(tk_observed_doubles, B->observed_doubles);
    B->observed_doubles = NULL;
  }
  if (B->observed_strings != NULL) {
    kh_foreach(B->observed_strings, f, cu, ({
      tk_cuset_destroy(cu);
    }))
    kh_destroy(tk_observed_strings, B->observed_strings);
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
  khint_t khi, kho, vi;
  tk_iuset_t *seen_features = tk_iuset_create();
  tk_dvec_t *value_vec = tk_dvec_create(L, 0, 0, 0);
  int i_value_vec = tk_lua_absindex(L, -1);
  int64_t id_feature;
  for (khi = kh_begin(B->observed_doubles); khi != kh_end(B->observed_doubles); khi ++) {
    if (!kh_exist(B->observed_doubles, khi))
      continue;
    id_feature = kh_key(B->observed_doubles, khi);
    tk_iuset_put(seen_features, id_feature, &kha);
  }
  for (khi = kh_begin(B->observed_strings); khi != kh_end(B->observed_strings); khi ++) {
    if (!kh_exist(B->observed_strings, khi))
      continue;
    id_feature = kh_key(B->observed_strings, khi);
    tk_iuset_put(seen_features, id_feature, &kha);
  }
  for (khi = tk_iuset_begin(seen_features); khi != tk_iuset_end(seen_features); khi ++) {
    if (!kh_exist(seen_features, khi))
      continue;
    id_feature = tk_iuset_key(seen_features, khi);
    if (tk_booleanizer_is_categorical(B, id_feature)) {
      tk_iuset_put(B->categorical, id_feature, &kha);
      kho = kh_get(tk_observed_strings, B->observed_strings, id_feature);
      if (kho != kh_end(B->observed_strings)) {
        tk_cuset_t *values = kh_value(B->observed_strings, kho);
        for (vi = kh_begin(values); vi != kh_end(values); vi ++) {
          if (!kh_exist(values, vi))
            continue;
          const char *value = kh_key(values, vi);
          tk_cat_bit_string_t key = { .f = id_feature, .v = (char *) value };
          khint_t outk = kh_put(tk_cat_bits_string, B->cat_bits_string, key, &kha);
          kh_value(B->cat_bits_string, outk) = (int64_t) B->next_bit ++;
        }
      }
      kho = kh_get(tk_observed_doubles, B->observed_doubles, id_feature);
      if (kho != kh_end(B->observed_doubles)) {
        tk_duset_t *values = kh_value(B->observed_doubles, kho);
        for (vi = kh_begin(values); vi != kh_end(values); vi ++) {
          if (!kh_exist(values, vi))
            continue;
          double value = kh_key(values, vi);
          tk_cat_bit_double_t key = { .f = id_feature, .v = value };
          khint_t outk = kh_put(tk_cat_bits_double, B->cat_bits_double, key, &kha);
          kh_value(B->cat_bits_double, outk) = (int64_t) B->next_bit ++;
        }
      }
    } else {
      tk_iuset_put(B->continuous, id_feature, &kha);
      kho = kh_get(tk_observed_doubles, B->observed_doubles, id_feature);
      if (kho == kh_end(B->observed_doubles))
        continue;
      tk_duset_t *value_set = kh_value(B->observed_doubles, kho);
      tk_dvec_clear(value_vec);
      tk_duset_dump(value_set, value_vec);
      tk_dvec_asc(value_vec, 0, value_vec->n);
      tk_booleanizer_add_thresholds(L, B, Bi, id_feature, value_vec);
    }
  }
  tk_booleanizer_shrink(B);
  B->finalized = true;
  lua_remove(L, i_value_vec);
  tk_iuset_destroy(seen_features);
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

static inline int64_t tk_booleanizer_bits (
  lua_State *L,
  tk_booleanizer_t *B
) {
  if (B->destroyed) {
    tk_lua_verror(L, 2, "bits", "can't query a destroyed booleanizer");
    return -1;
  }
  if (!B->finalized) {
    tk_lua_verror(L, 2, "bits", "finalize must be called before bits");
    return -1;
  }
  return (int64_t) B->next_bit;
}

static inline void tk_booleanizer_restrict (
  lua_State *L,
  tk_booleanizer_t *B,
  tk_ivec_t *ids
) {
  if (B->destroyed) {
    tk_lua_verror(L, 2, "restrict", "can't restrict a destroyed booleanizer");
    return;
  }
  if (!B->finalized) {
    tk_lua_verror(L, 2, "restrict", "finalize must be called before restrict");
    return;
  }
  tk_iumap_t *feature_id_map = tk_iumap_create();
  khint_t khi;
  int kha;
  for (int64_t i = 0; i < (int64_t) ids->n; i ++) {
    int64_t old_id = ids->a[i];
    khi = tk_iumap_put(feature_id_map, old_id, &kha);
    if (kha)
      tk_iumap_value(feature_id_map, khi) = i;
  }
  tk_iuset_t *new_continuous   = tk_iuset_create();
  tk_iuset_t *new_categorical  = tk_iuset_create();
  tk_iumap_t *new_integer_features = tk_iumap_create();
  tk_zumap_t *new_string_features  = tk_zumap_create();
  khash_t(tk_cat_bits_string) *new_cat_bits_string = kh_init(tk_cat_bits_string);
  khash_t(tk_cat_bits_double) *new_cat_bits_double = kh_init(tk_cat_bits_double);
  tk_iumap_t *new_cont_bits = tk_iumap_create();
  khash_t(tk_cont_thresholds) *new_cont_thresholds = kh_init(tk_cont_thresholds);
  int64_t next_bit = 0;
  for (int64_t new_f = 0; new_f < (int64_t) ids->n; new_f ++) {
    int64_t old_f = ids->a[new_f];
    if (tk_iuset_contains(B->continuous, old_f))
      tk_iuset_put(new_continuous, new_f, NULL);
    if (tk_iuset_contains(B->categorical, old_f))
      tk_iuset_put(new_categorical, new_f, NULL);
    khint_t k = tk_iumap_get(B->integer_features, old_f);
    if (k != kh_end(B->integer_features)) {
      khi = tk_iumap_put(new_integer_features, new_f, &kha);
      if (kha)
        tk_iumap_value(new_integer_features, khi) = tk_iumap_value(B->integer_features, k);
    }
    k = kh_end(B->string_features);
    const char *z;
    int64_t v;
    tk_zumap_foreach(B->string_features, z, v, ({
      if (v == old_f)
        k = tk_zumap_get(B->string_features, z);
    }));
    if (k != kh_end(B->string_features)) {
      z = strdup(kh_key(B->string_features, k));
      khi = tk_zumap_put(new_string_features, z, &kha);
      if (kha)
        tk_zumap_value(new_string_features, khi) = new_f;
    }
    tk_cat_bit_string_t cbs;
    int64_t new_bit;
    kh_foreach(B->cat_bits_string, cbs, v, ({
      if (cbs.f == old_f) {
        tk_cat_bit_string_t new_key = { .f = new_f, .v = strdup(cbs.v) };
        khint_t nk = kh_put(tk_cat_bits_string, new_cat_bits_string, new_key, NULL);
        new_bit = next_bit ++;
        kh_value(new_cat_bits_string, nk) = new_bit;
      }
    }));
    tk_cat_bit_double_t cbd;
    kh_foreach(B->cat_bits_double, cbd, v, ({
      if (cbd.f == old_f) {
        tk_cat_bit_double_t new_key = { .f = new_f, .v = cbd.v };
        khint_t nk = kh_put(tk_cat_bits_double, new_cat_bits_double, new_key, NULL);
        new_bit = next_bit ++;
        kh_value(new_cat_bits_double, nk) = new_bit;
      }
    }));
    k = tk_iumap_get(B->cont_bits, old_f);
    if (k != kh_end(B->cont_bits)) {
      khi = tk_iumap_put(new_cont_bits, new_f, &kha);
      if (kha)
        tk_iumap_value(new_cont_bits, khi) = next_bit;
      khint_t kth = kh_get(tk_cont_thresholds, B->cont_thresholds, old_f);
      if (kth != kh_end(B->cont_thresholds)) {
        tk_dvec_t *old_vec = kh_value(B->cont_thresholds, kth);
        tk_dvec_t *new_vec = tk_dvec_create(L, old_vec->n, 0, 0);
        memcpy(new_vec->a, old_vec->a, sizeof(double) * old_vec->n);
        new_vec->n = old_vec->n;
        khint_t nk = kh_put(tk_cont_thresholds, new_cont_thresholds, new_f, NULL);
        kh_value(new_cont_thresholds, nk) = new_vec;
      }
      next_bit += (int64_t) B->n_thresholds;
    }
  }
  tk_iuset_destroy(B->continuous);
  tk_iuset_destroy(B->categorical);
  tk_iumap_destroy(B->integer_features);
  const char *z;
  int64_t v;
  tk_zumap_foreach(B->string_features, z, v, ({
    free((char *) z);
  }))
  tk_zumap_destroy(B->string_features);
  kh_destroy(tk_cat_bits_string, B->cat_bits_string);
  kh_destroy(tk_cat_bits_double, B->cat_bits_double);
  tk_iumap_destroy(B->cont_bits);
  kh_destroy(tk_cont_thresholds, B->cont_thresholds);
  tk_iumap_destroy(feature_id_map);
  B->continuous = new_continuous;
  B->categorical = new_categorical;
  B->integer_features = new_integer_features;
  B->string_features = new_string_features;
  B->cat_bits_string = new_cat_bits_string;
  B->cat_bits_double = new_cat_bits_double;
  B->cont_bits = new_cont_bits;
  B->cont_thresholds = new_cont_thresholds;
  B->next_feature = ids->n;
  B->next_bit = (uint64_t) next_bit;
}

static inline void tk_booleanizer_destroy (
  tk_booleanizer_t *B
) {
  if (B->destroyed)
    return;
  tk_booleanizer_shrink(B);
  tk_iuset_destroy(B->continuous);
  tk_iuset_destroy(B->categorical);
  tk_iumap_destroy(B->integer_features);
  const char *z; int64_t f;
  tk_zumap_foreach(B->string_features, z, f, ({
    free((char *) z);
  }));
  tk_zumap_destroy(B->string_features);
  kh_destroy(tk_observed_strings, B->observed_strings);
  kh_destroy(tk_cont_thresholds, B->cont_thresholds);
  tk_cat_bit_string_t cbs;
  kh_foreach(B->cat_bits_string, cbs, f, ({
    free(cbs.v);
  }));
  kh_destroy(tk_cat_bits_string, B->cat_bits_string);
  kh_destroy(tk_cat_bits_double, B->cat_bits_double);
  tk_iumap_destroy(B->cont_bits);
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
  tk_lua_fwrite(L, (char *) &B->next_feature, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *) &B->next_bit, sizeof(uint64_t),1, fh);
  khint_t sz;
  int64_t f, v;
  sz = tk_iuset_size(B->continuous);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_iuset_foreach(B->continuous, f, ({
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
  }));
  sz = tk_iuset_size(B->categorical);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_iuset_foreach(B->categorical, f, ({
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
  }));
  sz = tk_iumap_size(B->integer_features);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_iumap_foreach(B->integer_features, f, v, ({
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, (char *) &v, sizeof(int64_t), 1, fh);
  }));
  sz = tk_zumap_size(B->string_features);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  const char *z;
  tk_zumap_foreach(B->string_features, z, f, ({
    size_t len = strlen(z);
    tk_lua_fwrite(L, (char *) &len, sizeof(size_t), 1, fh);
    tk_lua_fwrite(L, (char *) z, len, 1, fh);
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
  }));
  sz = kh_size(B->cat_bits_string);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_cat_bit_string_t cbs;
  kh_foreach(B->cat_bits_string, cbs, f, ({
    tk_lua_fwrite(L, (char *) &cbs.f, sizeof(int64_t), 1, fh);
    size_t len = strlen(cbs.v);
    tk_lua_fwrite(L, (char *) &len, sizeof(size_t), 1, fh);
    tk_lua_fwrite(L, cbs.v, len, 1, fh);
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
  }));
  sz = kh_size(B->cat_bits_double);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_cat_bit_double_t cbd;
  kh_foreach(B->cat_bits_double, cbd, f, ({
    tk_lua_fwrite(L, (char *) &cbd.f, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, (char *) &cbd.v, sizeof(double),  1, fh);
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
  }));
  sz = tk_iumap_size(B->cont_bits);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_iumap_foreach(B->cont_bits, f, v, ({
    tk_lua_fwrite(L, (char *) &f, sizeof(int64_t), 1, fh);
    tk_lua_fwrite(L, (char *) &v, sizeof(int64_t), 1, fh);
  }));
  sz = kh_size(B->cont_thresholds);
  tk_lua_fwrite(L, (char *) &sz, sizeof(sz), 1, fh);
  tk_dvec_t *thresholds;
  kh_foreach(B->cont_thresholds, f, thresholds, ({
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
  int64_t id;
  int t = lua_type(L, 2);
  switch (t) {
    case LUA_TNUMBER:
      id = tk_booleanizer_bit_integer(L, B, lua_tointeger(L, 2), false);
      if (id < 0)
        return 0;
      lua_pushinteger(L, id);
      return 1;
    case LUA_TSTRING:
      id = tk_booleanizer_bit_string(L, B, lua_tostring(L, 2), false);
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
    tk_booleanizer_encode_dvec(L, B, D, n_dims, id_dim0, out);
    return 1;
  } else {
    uint64_t id_sample = tk_lua_checkunsigned(L, 2, "id_sample");
    lua_pushcfunction(L, tk_booleanizer_bit_lua); // fn
    lua_pushvalue(L, 1); // fn B
    lua_pushvalue(L, 2); // fn B l
    lua_call(L, 2, 1); // id
    int64_t id_feature = (int64_t) tk_lua_checkunsigned(L, -1, "id_feature");
    out = tk_ivec_peekopt(L, 4);
    if (out == NULL)
      out = tk_ivec_create(L, 0, 0, 0);
    else
      lua_pushvalue(L, 4);
    int t = lua_type(L, 3);
    switch (t) {
      case LUA_TSTRING:
        tk_booleanizer_encode_string(L, B, id_sample, id_feature, luaL_checkstring(L, 3), out);
        break;
      case LUA_TNUMBER:
        tk_booleanizer_encode_double(L, B, id_sample, id_feature, luaL_checknumber(L, 3), out);
        break;
      case LUA_TBOOLEAN:
        tk_booleanizer_encode_integer(L, B, id_sample, id_feature, lua_toboolean(L, 3), out);
        break;
      default:
        tk_lua_verror(L, 2, "encode", "unexpected type passed to encode", lua_typename(L, t));
        break;
    }
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
    lua_pushcfunction(L, tk_booleanizer_bit_lua); // fn
    lua_pushvalue(L, 1); // fn B
    lua_pushvalue(L, 2); // fn B l
    lua_pushboolean(L, true); // fn B l train
    lua_call(L, 3, 1); // id
    int64_t id_feature = (int64_t) tk_lua_checkunsigned(L, -1, "id_feature");
    int t = lua_type(L, 2);
    switch (t) {
      case LUA_TSTRING:
        tk_booleanizer_observe_string(L, B, id_feature, luaL_checkstring(L, 2));
        break;
      case LUA_TNUMBER:
        tk_booleanizer_observe_double(L, B, id_feature, luaL_checknumber(L, 2));
        break;
      case LUA_TBOOLEAN:
        tk_booleanizer_observe_integer(L, B, id_feature, lua_toboolean(L, 2));
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

static inline int tk_booleanizer_features_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  lua_pushinteger(L, tk_booleanizer_features(L, B));
  return 1;
}

static inline int tk_booleanizer_bits_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  lua_pushinteger(L, tk_booleanizer_bits(L, B));
  return 1;
}

static inline int tk_booleanizer_restrict_lua (lua_State *L)
{
  tk_booleanizer_t *B = tk_booleanizer_peek(L, 1);
  tk_ivec_t *ids = tk_ivec_peek(L, 2, "top_v");
  tk_booleanizer_restrict(L, B, ids);
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
  { "features", tk_booleanizer_features_lua },
  { "bits", tk_booleanizer_bits_lua },
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
  B->continuous = (continuous != NULL) ? tk_iuset_from_ivec(continuous) : tk_iuset_create();
  B->categorical = (categorical != NULL) ? tk_iuset_from_ivec(categorical) : tk_iuset_create();
  B->n_thresholds = n_thresholds;
  B->integer_features = tk_iumap_create();
  B->string_features = tk_zumap_create();
  B->observed_doubles = kh_init(tk_observed_doubles);
  B->observed_strings = kh_init(tk_observed_strings);
  B->cat_bits_string = kh_init(tk_cat_bits_string);
  B->cat_bits_double = kh_init(tk_cat_bits_double);
  B->cont_bits = tk_iumap_create();
  B->cont_thresholds = kh_init(tk_cont_thresholds);
  B->next_feature = 0;
  B->next_bit = 0;
  B->finalized = false;
  B->destroyed = false;
  return B;
}

static inline tk_booleanizer_t *tk_booleanizer_load (
  lua_State *L,
  FILE *fh
) {
  tk_booleanizer_t *B = tk_lua_newuserdata(L, tk_booleanizer_t, TK_BOOLEANIZER_MT, tk_booleanizer_mt_fns, tk_booleanizer_gc_lua);
  memset(B, 0, sizeof(*B));
  tk_lua_fread(L, (char *) &B->finalized, sizeof(bool), 1, fh);
  tk_lua_fread(L, (char *) &B->n_thresholds, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, (char *) &B->next_feature, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, (char *) &B->next_bit, sizeof(uint64_t), 1, fh);
  khint_t khi, sz;
  int kha;
  int64_t f;
  B->continuous = tk_iuset_create();
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    tk_lua_fread(L, (char *) &f, sizeof(int64_t), 1, fh);
    tk_iuset_put(B->continuous, f, &kha);
  }
  B->categorical = tk_iuset_create();
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    tk_lua_fread(L, (char *) &f, sizeof(int64_t), 1, fh);
    tk_iuset_put(B->categorical, f, &kha);
  }
  int64_t key, val;
  B->integer_features = tk_iumap_create();
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    tk_lua_fread(L, (char *) &key, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, (char *) &val, sizeof(int64_t), 1, fh);
    khi = tk_iumap_put(B->integer_features, key, &kha);
    if (kha)
      tk_iumap_value(B->integer_features, khi) = val;
  }
  B->string_features = tk_zumap_create();
  size_t len;
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    tk_lua_fread(L, (char *) &len, sizeof(size_t), 1, fh);
    char *z = (char *) malloc(len + 1);
    tk_lua_fread(L, z, len, 1, fh);
    z[len] = '\0';
    tk_lua_fread(L, (char *) &f, sizeof(int64_t), 1, fh);
    khi = tk_zumap_put(B->string_features, z, &kha);
    if (kha)
      tk_zumap_value(B->string_features, khi) = f;
  }
  B->cat_bits_string = kh_init(tk_cat_bits_string);
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    int64_t feature_id;
    tk_lua_fread(L, (char *) &feature_id, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, (char *) &len, sizeof(size_t), 1, fh);
    char *value = (char *) malloc(len + 1);
    tk_lua_fread(L, value, len, 1, fh);
    value[len] = '\0';
    int64_t bit_id;
    tk_lua_fread(L, (char *) &bit_id, sizeof(int64_t), 1, fh);
    tk_cat_bit_string_t key = { .f = feature_id, .v = value };
    khint_t k = kh_put(tk_cat_bits_string, B->cat_bits_string, key, &kha);
    kh_value(B->cat_bits_string, k) = bit_id;
  }
  B->cat_bits_double = kh_init(tk_cat_bits_double);
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    int64_t feature_id;
    double dval;
    int64_t bit_id;
    tk_lua_fread(L, (char *) &feature_id, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, (char *) &dval, sizeof(double), 1, fh);
    tk_lua_fread(L, (char *) &bit_id, sizeof(int64_t), 1, fh);
    tk_cat_bit_double_t key = { .f = feature_id, .v = dval };
    khint_t k = kh_put(tk_cat_bits_double, B->cat_bits_double, key, &kha);
    kh_value(B->cat_bits_double, k) = bit_id;
  }
  B->cont_bits = tk_iumap_create();
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i++) {
    tk_lua_fread(L, (char *) &key, sizeof(int64_t), 1, fh);
    tk_lua_fread(L, (char *) &val, sizeof(int64_t), 1, fh);
    khi = tk_iumap_put(B->cont_bits, key, &kha);
    if (kha)
      tk_iumap_value(B->cont_bits, khi) = val;
  }
  B->cont_thresholds = kh_init(tk_cont_thresholds);
  tk_lua_fread(L, (char *) &sz, sizeof(sz), 1, fh);
  for (khint_t i = 0; i < sz; i ++) {
    int64_t feature_id;
    tk_lua_fread(L, (char *) &feature_id, sizeof(int64_t), 1, fh);
    size_t n_thresh;
    tk_lua_fread(L, (char *) &n_thresh, sizeof(size_t), 1, fh);
    tk_dvec_t *thresh_vec = tk_dvec_create(L, n_thresh, 0, 0);
    tk_lua_fread(L, (char *) thresh_vec->a, sizeof(double), n_thresh, fh);
    thresh_vec->n = n_thresh;
    khint_t k = kh_put(tk_cont_thresholds, B->cont_thresholds, feature_id, &kha);
    kh_value(B->cont_thresholds, k) = thresh_vec;
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
