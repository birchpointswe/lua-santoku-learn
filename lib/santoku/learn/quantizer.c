#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <lapacke.h>
#include <cblas.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/cvec.h>
#include <santoku/dvec.h>

#define TK_THRESH_ENCODER_MT "tk_thresh_encoder_t"

typedef struct {
  tk_ivec_t *bit_dims;
  tk_dvec_t *bit_thresholds;
  uint64_t n_dims;
  uint64_t n_bits;
  bool destroyed;
} tk_thresh_encoder_t;

static inline tk_thresh_encoder_t *tk_thresh_encoder_peek(lua_State *L, int i) {
  return (tk_thresh_encoder_t *)luaL_checkudata(L, i, TK_THRESH_ENCODER_MT);
}

static inline int tk_thresh_encoder_gc(lua_State *L) {
  tk_thresh_encoder_t *enc = tk_thresh_encoder_peek(L, 1);
  enc->bit_dims = NULL;
  enc->bit_thresholds = NULL;
  enc->destroyed = true;
  return 0;
}

static inline int tk_thresh_encode_lua(lua_State *L) {
  tk_thresh_encoder_t *enc = tk_thresh_encoder_peek(L, 1);
  uint64_t n_bits = enc->n_bits;
  uint64_t n_bytes = TK_CVEC_BITS_BYTES(n_bits);
  tk_dvec_t *dvec_in = tk_dvec_peek(L, 2, "raw_codes");
  uint64_t n_dims = enc->n_dims;
  uint64_t n_samples = dvec_in->n / n_dims;
  int64_t *dims = enc->bit_dims->a;
  double *thresholds = enc->bit_thresholds->a;
  tk_cvec_t *out = tk_cvec_create(L, n_samples * n_bytes, NULL, NULL);
  memset(out->a, 0, out->n);
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    double *x = dvec_in->a + i * n_dims;
    uint8_t *dest = (uint8_t *)out->a + i * n_bytes;
    for (uint64_t k = 0; k < n_bits; k++) {
      if (x[dims[k]] > thresholds[k])
        dest[TK_CVEC_BITS_BYTE(k)] |= (1 << TK_CVEC_BITS_BIT(k));
    }
  }
  return 1;
}

static inline int tk_thresh_n_bits_lua(lua_State *L) {
  tk_thresh_encoder_t *enc = tk_thresh_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->n_bits);
  return 1;
}

static inline int tk_thresh_n_dims_lua(lua_State *L) {
  tk_thresh_encoder_t *enc = tk_thresh_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->n_dims);
  return 1;
}

static inline int tk_thresh_dims_lua(lua_State *L) {
  tk_thresh_encoder_peek(L, 1);
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "dims");
  return 1;
}

static inline int tk_thresh_thresholds_lua(lua_State *L) {
  tk_thresh_encoder_peek(L, 1);
  lua_getfenv(L, 1);
  lua_getfield(L, -1, "thresholds");
  return 1;
}

static inline int tk_thresh_encoder_persist_lua(lua_State *L) {
  tk_thresh_encoder_t *enc = tk_thresh_encoder_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting filepath or true");
  tk_lua_fwrite(L, "TKqt", 1, 4, fh);
  uint8_t version = 1;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, (char *)&enc->n_dims, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *)&enc->n_bits, sizeof(uint64_t), 1, fh);
  tk_ivec_persist(L, enc->bit_dims, fh);
  tk_dvec_persist(L, enc->bit_thresholds, fh);
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

static luaL_Reg tk_thresh_encoder_mt_fns[] = {
  { "encode", tk_thresh_encode_lua },
  { "n_bits", tk_thresh_n_bits_lua },
  { "n_dims", tk_thresh_n_dims_lua },
  { "dims", tk_thresh_dims_lua },
  { "thresholds", tk_thresh_thresholds_lua },
  { "persist", tk_thresh_encoder_persist_lua },
  { NULL, NULL }
};

#define TK_ITQ_ENCODER_MT "tk_itq_encoder_t"

typedef struct {
  tk_dvec_t *rotation;
  tk_dvec_t *means;
  uint64_t k;
  bool destroyed;
} tk_itq_encoder_t;

static inline tk_itq_encoder_t *tk_itq_encoder_peek(lua_State *L, int i) {
  return (tk_itq_encoder_t *)luaL_checkudata(L, i, TK_ITQ_ENCODER_MT);
}

static inline int tk_itq_encoder_gc(lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  enc->rotation = NULL;
  enc->means = NULL;
  enc->destroyed = true;
  return 0;
}

static inline int tk_itq_encode_lua(lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  tk_dvec_t *raw = tk_dvec_peek(L, 2, "raw_codes");
  uint64_t k = enc->k;
  uint64_t n_samples = raw->n / k;
  uint64_t n_bytes = TK_CVEC_BITS_BYTES(k);
  double *centered = (double *)malloc(n_samples * k * sizeof(double));
  memcpy(centered, raw->a, n_samples * k * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    double *p = centered + i * k;
    for (uint64_t j = 0; j < k; j++)
      p[j] -= enc->means->a[j];
  }
  double *rotated = (double *)malloc(n_samples * k * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    (int)n_samples, (int)k, (int)k, 1.0, centered, (int)k,
    enc->rotation->a, (int)k, 0.0, rotated, (int)k);
  tk_cvec_t *out = tk_cvec_create(L, n_samples * n_bytes, NULL, NULL);
  memset(out->a, 0, out->n);
  #pragma omp parallel for schedule(static)
  for (uint64_t i = 0; i < n_samples; i++) {
    double *r = rotated + i * k;
    uint8_t *dest = (uint8_t *)out->a + i * n_bytes;
    for (uint64_t j = 0; j < k; j++) {
      if (r[j] > 0.0)
        dest[TK_CVEC_BITS_BYTE(j)] |= (1 << TK_CVEC_BITS_BIT(j));
    }
  }
  free(centered);
  free(rotated);
  return 1;
}

static inline int tk_itq_n_bits_lua(lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)enc->k);
  return 1;
}

static inline int tk_itq_encoder_persist_lua(lua_State *L) {
  tk_itq_encoder_t *enc = tk_itq_encoder_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting filepath or true");
  tk_lua_fwrite(L, "TKqi", 1, 4, fh);
  uint8_t version = 1;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &enc->k, sizeof(uint64_t), 1, fh);
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
  { "persist", tk_itq_encoder_persist_lua },
  { NULL, NULL }
};

static inline int tk_quantizer_create_lua(lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "mode");
  const char *mode = lua_isstring(L, -1) ? lua_tostring(L, -1) : NULL;
  lua_pop(L, 1);

  if (mode && strcmp(mode, "thermometer") == 0) {
    lua_getfield(L, 1, "raw_codes");
    tk_dvec_t *input_dvec = tk_dvec_peek(L, -1, "raw_codes");
    lua_pop(L, 1);
    double *inp = input_dvec->a;
    uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "create", "n_samples");
    uint64_t d_in = input_dvec->n / n_samples;
    uint64_t B = tk_lua_foptunsigned(L, 1, "create", "n_bins", 0);
    double **dt = (double **)malloc(d_in * sizeof(double *));
    uint64_t *dn = (uint64_t *)malloc(d_in * sizeof(uint64_t));
    #pragma omp parallel
    {
      double *col = (double *)malloc(n_samples * sizeof(double));
      #pragma omp for schedule(dynamic)
      for (uint64_t d = 0; d < d_in; d++) {
        for (uint64_t i = 0; i < n_samples; i++)
          col[i] = inp[i * d_in + d];
        tk_dvec_t tmp = { .a = col, .n = n_samples, .m = n_samples };
        tk_dvec_asc(&tmp, 0, n_samples);
        if (B == 0) {
          uint64_t nu = 0;
          for (uint64_t i = 0; i < n_samples; i++)
            if (i == 0 || col[i] != col[i - 1]) nu++;
          dt[d] = (double *)malloc(nu * sizeof(double));
          dn[d] = nu;
          uint64_t j = 0;
          for (uint64_t i = 0; i < n_samples; i++)
            if (i == 0 || col[i] != col[i - 1]) dt[d][j++] = col[i];
        } else {
          dt[d] = (double *)malloc(B * sizeof(double));
          dn[d] = B;
          for (uint64_t b = 0; b < B; b++) {
            uint64_t qi = (uint64_t)(((double)(b + 1) / (double)(B + 1)) * (double)n_samples);
            if (qi >= n_samples) qi = n_samples - 1;
            dt[d][b] = col[qi];
          }
        }
      }
      free(col);
    }
    uint64_t total_bits = 0;
    for (uint64_t d = 0; d < d_in; d++) total_bits += dn[d];
    tk_ivec_t *out_dims = tk_ivec_create(L, total_bits, 0, 0);
    int out_dims_idx = lua_gettop(L);
    tk_dvec_t *out_thresholds = tk_dvec_create(L, total_bits, 0, 0);
    int out_thresholds_idx = lua_gettop(L);
    out_dims->n = total_bits;
    out_thresholds->n = total_bits;
    uint64_t pos = 0;
    for (uint64_t d = 0; d < d_in; d++) {
      for (uint64_t k = 0; k < dn[d]; k++) {
        out_dims->a[pos] = (int64_t)d;
        out_thresholds->a[pos] = dt[d][k];
        pos++;
      }
      free(dt[d]);
    }
    free(dt); free(dn);
    tk_thresh_encoder_t *enc = tk_lua_newuserdata(L, tk_thresh_encoder_t,
      TK_THRESH_ENCODER_MT, tk_thresh_encoder_mt_fns, tk_thresh_encoder_gc);
    int Ei = lua_gettop(L);
    enc->bit_dims = out_dims;
    enc->bit_thresholds = out_thresholds;
    enc->n_dims = d_in;
    enc->n_bits = total_bits;
    enc->destroyed = false;
    lua_newtable(L);
    lua_pushvalue(L, out_dims_idx);
    lua_setfield(L, -2, "dims");
    lua_pushvalue(L, out_thresholds_idx);
    lua_setfield(L, -2, "thresholds");
    lua_setfenv(L, Ei);
    lua_pushvalue(L, Ei);
    return 1;
  }

  if (mode && strcmp(mode, "itq") == 0) {
    lua_getfield(L, 1, "raw_codes");
    tk_dvec_t *raw = tk_dvec_peek(L, -1, "raw_codes");
    lua_pop(L, 1);
    uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "create", "n_samples");
    uint64_t k = raw->n / n_samples;
    uint64_t iterations = tk_lua_foptunsigned(L, 1, "create", "iterations", 50);
    double *data = (double *)malloc(n_samples * k * sizeof(double));
    memcpy(data, raw->a, n_samples * k * sizeof(double));
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
    double *R = (double *)malloc(k * k * sizeof(double));
    for (uint64_t i = 0; i < k; i++)
      for (uint64_t j = 0; j < k; j++)
        R[i * k + j] = (i == j) ? 1.0 : 0.0;
    double *projected = (double *)malloc(n_samples * k * sizeof(double));
    double *B = (double *)malloc(n_samples * k * sizeof(double));
    double *BtV = (double *)malloc(k * k * sizeof(double));
    double *XtX = (double *)malloc(k * k * sizeof(double));
    double *tmp = (double *)malloc(k * k * sizeof(double));
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
      double nrm = cblas_dnrm2((int)(k * k), BtV, 1);
      if (nrm < 1e-15) break;
      cblas_dscal((int)(k * k), 1.0 / nrm, BtV, 1);
      for (int ns = 0; ns < 15; ns++) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
          (int)k, (int)k, (int)k, 1.0, BtV, (int)k,
          BtV, (int)k, 0.0, XtX, (int)k);
        memcpy(tmp, BtV, k * k * sizeof(double));
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          (int)k, (int)k, (int)k, -0.5, tmp, (int)k,
          XtX, (int)k, 1.5, BtV, (int)k);
      }
      memcpy(R, BtV, k * k * sizeof(double));
    }
    free(projected); free(B); free(BtV);
    free(XtX); free(tmp);
    free(data);
    tk_dvec_t *rot_out = tk_dvec_create(L, k * k, 0, 0);
    int rot_idx = lua_gettop(L);
    memcpy(rot_out->a, R, k * k * sizeof(double));
    free(R);
    tk_dvec_t *means_out = tk_dvec_create(L, k, 0, 0);
    int means_idx = lua_gettop(L);
    memcpy(means_out->a, means, k * sizeof(double));
    free(means);
    tk_itq_encoder_t *enc = tk_lua_newuserdata(L, tk_itq_encoder_t,
      TK_ITQ_ENCODER_MT, tk_itq_encoder_mt_fns, tk_itq_encoder_gc);
    int Ei = lua_gettop(L);
    enc->rotation = rot_out;
    enc->means = means_out;
    enc->k = k;
    enc->destroyed = false;
    lua_newtable(L);
    lua_pushvalue(L, rot_idx);
    lua_setfield(L, -2, "rotation");
    lua_pushvalue(L, means_idx);
    lua_setfield(L, -2, "means");
    lua_setfenv(L, Ei);
    lua_pushvalue(L, Ei);
    return 1;
  }

  return luaL_error(L, "quantizer.create: mode must be 'thermometer' or 'itq'");
}

static inline int tk_quantizer_load_lua(lua_State *L) {
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKqi", 4) == 0) {
    uint8_t version;
    tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
    if (version != 1) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "unsupported ITQ encoder version %d", (int)version);
    }
    uint64_t k;
    tk_lua_fread(L, &k, sizeof(uint64_t), 1, fh);
    tk_dvec_t *rotation = tk_dvec_load(L, fh);
    int rot_idx = lua_gettop(L);
    tk_dvec_t *means = tk_dvec_load(L, fh);
    int means_idx = lua_gettop(L);
    tk_lua_fclose(L, fh);
    tk_itq_encoder_t *enc = tk_lua_newuserdata(L, tk_itq_encoder_t,
      TK_ITQ_ENCODER_MT, tk_itq_encoder_mt_fns, tk_itq_encoder_gc);
    int Ei = lua_gettop(L);
    enc->rotation = rotation;
    enc->means = means;
    enc->k = k;
    enc->destroyed = false;
    lua_newtable(L);
    lua_pushvalue(L, rot_idx);
    lua_setfield(L, -2, "rotation");
    lua_pushvalue(L, means_idx);
    lua_setfield(L, -2, "means");
    lua_setfenv(L, Ei);
    lua_pushvalue(L, Ei);
    return 1;
  }
  if (memcmp(magic, "TKqt", 4) == 0) {
    uint8_t version;
    tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
    if (version != 1) {
      tk_lua_fclose(L, fh);
      return luaL_error(L, "unsupported threshold encoder version %d", (int)version);
    }
    uint64_t n_dims, n_bits;
    tk_lua_fread(L, &n_dims, sizeof(uint64_t), 1, fh);
    tk_lua_fread(L, &n_bits, sizeof(uint64_t), 1, fh);
    tk_ivec_t *bit_dims = tk_ivec_load(L, fh);
    int dims_idx = lua_gettop(L);
    tk_dvec_t *bit_thresholds = tk_dvec_load(L, fh);
    int thresh_idx = lua_gettop(L);
    tk_lua_fclose(L, fh);
    tk_thresh_encoder_t *enc = tk_lua_newuserdata(L, tk_thresh_encoder_t,
      TK_THRESH_ENCODER_MT, tk_thresh_encoder_mt_fns, tk_thresh_encoder_gc);
    int Ei = lua_gettop(L);
    enc->bit_dims = bit_dims;
    enc->bit_thresholds = bit_thresholds;
    enc->n_dims = n_dims;
    enc->n_bits = n_bits;
    enc->destroyed = false;
    lua_newtable(L);
    lua_pushvalue(L, dims_idx);
    lua_setfield(L, -2, "dims");
    lua_pushvalue(L, thresh_idx);
    lua_setfield(L, -2, "thresholds");
    lua_setfenv(L, Ei);
    lua_pushvalue(L, Ei);
    return 1;
  }
  tk_lua_fclose(L, fh);
  return luaL_error(L, "invalid quantizer file (bad magic)");
}

static luaL_Reg tk_quantizer_fns[] = {
  { "create", tk_quantizer_create_lua },
  { "load", tk_quantizer_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_quantizer(lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_quantizer_fns, 0);
  return 1;
}
