#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <limits.h>
#include <santoku/lua/utils.h>
#include <santoku/iuset.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/pvec.h>
#include <santoku/rvec.h>
#include <santoku/tsetlin/inv.h>
#include <santoku/tsetlin/ann.h>
#include <santoku/tsetlin/hbi.h>
#include <santoku/tsetlin/graph.h>

#define TK_HLTH_ENCODER_MT "tk_hlth_encoder_t"
#define TK_HLTH_ENCODER_EPH "tk_hlth_encoder_eph"

typedef enum {
  TK_HLTH_IDX_INV,
  TK_HLTH_IDX_ANN,
  TK_HLTH_IDX_HBI
} tk_hlth_index_type_t;

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
  bool destroyed;
} tk_hlth_encoder_t;

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

static inline int tk_hlth_encode_lua(lua_State *L) {
  tk_hlth_encoder_t *enc = (tk_hlth_encoder_t *)lua_touserdata(L, lua_upvalueindex(1));

  if (enc->destroyed)
    return luaL_error(L, "encode: encoder has been destroyed");

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
  uint64_t n_latent_bits = enc->n_hidden * enc->n_landmarks;
  uint64_t n_latent_bytes = TK_CVEC_BITS_BYTES(n_latent_bits);
  tk_cvec_t *out = tk_cvec_create(L, n_samples * n_latent_bytes, NULL, NULL);
  out->n = n_samples * n_latent_bytes;
  memset(out->a, 0, out->n);

  if (n_samples == 1) {
    tk_ivec_t *tmp = tk_ivec_create(NULL, 0, 0, 0);
    int64_t nbr_idx, nbr_uid;
    TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, feat_hbi, inv_hoods, ann_hoods, hbi_hoods, 0, 1.0, nbr_ids, nbr_idx, nbr_uid, {
      tk_ivec_push(tmp, nbr_uid);
    });
    uint64_t bit_offset = 0;
    for (uint64_t j = 0; j < tmp->n && j < enc->n_landmarks; j++) {
      char *code_data = code_ann ? tk_ann_get(code_ann, tmp->a[j]) : tk_hbi_get(code_hbi, tmp->a[j]);
      if (code_data != NULL) {
        for (uint64_t b = 0; b < enc->n_hidden; b++) {
          if (code_data[TK_CVEC_BITS_BYTE(b)] & (1 << TK_CVEC_BITS_BIT(b))) {
            uint64_t dst_bit = bit_offset + b;
            ((uint8_t *)out->a)[TK_CVEC_BITS_BYTE(dst_bit)] |= (1 << TK_CVEC_BITS_BIT(dst_bit));
          }
        }
      }
      bit_offset += enc->n_hidden;
    }
    tk_ivec_destroy(tmp);
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

  lua_getfield(L, 1, "landmarks_index");
  tk_inv_t *feat_inv = tk_inv_peekopt(L, -1);
  tk_ann_t *feat_ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *feat_hbi = tk_hbi_peekopt(L, -1);
  lua_pop(L, 1);

  if (!feat_inv && !feat_ann && !feat_hbi)
    return luaL_error(L, "landmark_encoder: landmark_index must be inv, ann, or hbi");

  lua_getfield(L, 1, "codes_index");
  tk_ann_t *code_ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *code_hbi = tk_hbi_peekopt(L, -1);
  lua_pop(L, 1);

  if (!code_ann && !code_hbi)
    return luaL_error(L, "landmark_encoder: code_index must be ann or hbi");

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

  uint64_t n_hidden = code_ann ? code_ann->features : code_hbi->features;

  tk_hlth_encoder_t *enc = tk_lua_newuserdata(L, tk_hlth_encoder_t, TK_HLTH_ENCODER_MT, tk_hlth_encoder_mt_fns, tk_hlth_encoder_gc);
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

  if (code_ann) {
    enc->code_idx = code_ann;
    enc->code_idx_type = TK_HLTH_IDX_ANN;
  } else {
    enc->code_idx = code_hbi;
    enc->code_idx_type = TK_HLTH_IDX_HBI;
  }

  enc->n_landmarks = n_landmarks;
  enc->n_hidden = n_hidden;
  enc->probe_radius = probe_radius;
  enc->cmp = cmp;
  enc->tversky_alpha = tversky_alpha;
  enc->tversky_beta = tversky_beta;
  enc->rank_filter = rank_filter;

  lua_getfield(L, 1, "landmarks_index");
  tk_lua_add_ephemeron(L, TK_HLTH_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "codes_index");
  tk_lua_add_ephemeron(L, TK_HLTH_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_pushcclosure(L, tk_hlth_encode_lua, 1);
  lua_pushinteger(L, (int64_t) n_landmarks * (int64_t) n_hidden);

  return 2;
}

static luaL_Reg tk_hlth_fns[] = {
  { "landmark_encoder", tk_hlth_landmark_encoder_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_hlth(lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_hlth_fns, 0);
  return 1;
}
