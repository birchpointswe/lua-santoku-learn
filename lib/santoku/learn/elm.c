#include <lua.h>
#include <lauxlib.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>

#define TK_ELM_MT "tk_elm_t"

typedef struct {
  tk_dvec_t *col_mean;
  tk_dvec_t *col_inv_std;
  tk_dvec_t *dense_mean;
  tk_dvec_t *dense_inv_std;
  tk_dvec_t *fw;
  int64_t n_hidden;
  int64_t n_tokens;
  int64_t n_dense;
  uint64_t seed;
  bool destroyed;
} tk_elm_t;

static inline tk_elm_t *tk_elm_peek (lua_State *L, int i) {
  return (tk_elm_t *)luaL_checkudata(L, i, TK_ELM_MT);
}

static inline int tk_elm_gc (lua_State *L) {
  tk_elm_t *e = tk_elm_peek(L, 1);
  e->col_mean = NULL;
  e->col_inv_std = NULL;
  e->dense_mean = NULL;
  e->dense_inv_std = NULL;
  e->fw = NULL;
  e->destroyed = true;
  return 0;
}

static luaL_Reg tk_elm_mt_fns[];

static inline uint64_t elm_splitmix64 (uint64_t *s)
{
  uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static inline double elm_rand_normal (uint64_t *s)
{
  double u1 = ((double)(elm_splitmix64(s) >> 11) + 0.5) / (double)(1ULL << 53);
  double u2 = ((double)(elm_splitmix64(s) >> 11) + 0.5) / (double)(1ULL << 53);
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static void tk_elm_project_core (
  double *out, int64_t n_samples, int64_t n_hidden,
  int64_t n_tokens, int64_t n_dense, uint64_t seed,
  tk_ivec_t *csc_off, tk_ivec_t *csc_idx, double *fw,
  double *df,
  double *dense_mean, double *dense_inv_std,
  double *col_mean, double *col_inv_std,
  double *save_col_mean, double *save_col_inv_std,
  double *save_dense_mean, double *save_dense_inv_std
)
{
  int has_sparse = csc_off != NULL;
  int has_dense = df != NULL;
  int sidecar = has_sparse && has_dense;
  int64_t out_cols = sidecar ? n_hidden + n_dense : n_hidden;

  double *inv_norm = NULL;
  if (has_sparse) {
    inv_norm = (double *)calloc((uint64_t)n_samples, sizeof(double));
    for (int64_t t = 0; t < n_tokens; t++) {
      double fw_sq = fw ? fw[t] * fw[t] : 1.0;
      int64_t lo = csc_off->a[t], hi = csc_off->a[t + 1];
      for (int64_t i = lo; i < hi; i++)
        inv_norm[csc_idx->a[i]] += fw_sq;
    }
    for (int64_t s = 0; s < n_samples; s++) {
      double n2 = inv_norm[s];
      inv_norm[s] = n2 > 0.0 ? 1.0 / sqrt(n2) : 0.0;
    }
  }

  double *df_work = NULL;
  if (has_dense) {
    uint64_t df_sz = (uint64_t)n_samples * (uint64_t)n_dense;
    df_work = (double *)malloc(df_sz * sizeof(double));
    memcpy(df_work, df, df_sz * sizeof(double));
    if (save_dense_mean) {
      for (int64_t d = 0; d < n_dense; d++) {
        double sum = 0, sum2 = 0;
        for (int64_t s = 0; s < n_samples; s++) {
          double v = df_work[s * n_dense + d];
          sum += v;
          sum2 += v * v;
        }
        double mean = sum / (double)n_samples;
        double var = sum2 / (double)n_samples - mean * mean;
        double istd = var > 1e-24 ? 1.0 / sqrt(var) : 0.0;
        save_dense_mean[d] = mean;
        save_dense_inv_std[d] = istd;
        for (int64_t s = 0; s < n_samples; s++)
          df_work[s * n_dense + d] = (df_work[s * n_dense + d] - mean) * istd;
      }
    } else if (dense_mean) {
      for (int64_t d = 0; d < n_dense; d++) {
        double mean = dense_mean[d];
        double istd = dense_inv_std[d];
        for (int64_t s = 0; s < n_samples; s++)
          df_work[s * n_dense + d] = (df_work[s * n_dense + d] - mean) * istd;
      }
    }
  }

  #pragma omp parallel
  {
    double *col = (double *)malloc((uint64_t)n_samples * sizeof(double));
    #pragma omp for schedule(static)
    for (int64_t h = 0; h < n_hidden; h++) {
      memset(col, 0, (uint64_t)n_samples * sizeof(double));
      uint64_t rng = seed ^ ((uint64_t)h * 0x517cc1b727220a95ULL);
      elm_splitmix64(&rng);
      if (has_sparse) {
        for (int64_t t = 0; t < n_tokens; t++) {
          double w = elm_rand_normal(&rng);
          if (fw) w *= fw[t];
          int64_t lo = csc_off->a[t];
          int64_t hi = csc_off->a[t + 1];
          for (int64_t i = lo; i < hi; i++)
            col[csc_idx->a[i]] += w * inv_norm[csc_idx->a[i]];
        }
      }
      if (has_dense && !sidecar) {
        for (int64_t d = 0; d < n_dense; d++) {
          double w = elm_rand_normal(&rng);
          for (int64_t s = 0; s < n_samples; s++)
            col[s] += w * df_work[s * n_dense + d];
        }
      }
      {
        double bias = elm_rand_normal(&rng);
        for (int64_t s = 0; s < n_samples; s++) {
          double v = col[s] + bias;
          out[s * out_cols + h] = v > 0.0 ? v : 0.0;
        }
      }
    }
    free(col);
  }

  free(inv_norm);

  if (save_col_mean) {
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < n_hidden; j++) {
      double sum = 0, sum2 = 0;
      for (int64_t i = 0; i < n_samples; i++) {
        double v = out[i * out_cols + j];
        sum += v;
        sum2 += v * v;
      }
      double mean = sum / (double)n_samples;
      double var = sum2 / (double)n_samples - mean * mean;
      double istd = var > 1e-24 ? 1.0 / sqrt(var) : 0.0;
      save_col_mean[j] = mean;
      save_col_inv_std[j] = istd;
      for (int64_t i = 0; i < n_samples; i++)
        out[i * out_cols + j] = (out[i * out_cols + j] - mean) * istd;
    }
  } else if (col_mean) {
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < n_hidden; j++) {
      double mean = col_mean[j];
      double istd = col_inv_std[j];
      for (int64_t i = 0; i < n_samples; i++)
        out[i * out_cols + j] = (out[i * out_cols + j] - mean) * istd;
    }
  }

  if (sidecar && df_work) {
    double ds = (n_dense > 0) ? sqrt((double)n_hidden / (double)n_dense) : 1.0;
    #pragma omp parallel for schedule(static)
    for (int64_t s = 0; s < n_samples; s++)
      for (int64_t d = 0; d < n_dense; d++)
        out[s * out_cols + n_hidden + d] = ds * df_work[s * n_dense + d];
  }
  free(df_work);
}

static inline void tk_elm_parse_sparse (
  lua_State *L, int ti,
  int *has_sparse, tk_ivec_t **csc_off, tk_ivec_t **csc_idx,
  int64_t *n_tokens, double **fw
)
{
  lua_getfield(L, ti, "csc_offsets");
  *has_sparse = !lua_isnil(L, -1);
  *csc_off = *has_sparse ? tk_ivec_peek(L, -1, "csc_offsets") : NULL;
  lua_pop(L, 1);
  *csc_idx = NULL;
  *n_tokens = 0;
  *fw = NULL;
  if (*has_sparse) {
    lua_getfield(L, ti, "csc_indices");
    *csc_idx = tk_ivec_peek(L, -1, "csc_indices");
    lua_pop(L, 1);
    *n_tokens = (int64_t)tk_lua_fcheckunsigned(L, ti, "project", "n_tokens");
    lua_getfield(L, ti, "feature_weights");
    if (!lua_isnil(L, -1))
      *fw = tk_dvec_peek(L, -1, "feature_weights")->a;
    lua_pop(L, 1);
  }
}

static inline void tk_elm_parse_dense (
  lua_State *L, int ti,
  int *has_dense, double **df, int64_t *n_dense
)
{
  lua_getfield(L, ti, "dense_features");
  *has_dense = !lua_isnil(L, -1);
  *df = *has_dense ? tk_dvec_peek(L, -1, "dense_features")->a : NULL;
  lua_pop(L, 1);
  *n_dense = 0;
  if (*has_dense)
    *n_dense = (int64_t)tk_lua_fcheckunsigned(L, ti, "project", "n_dense");
}

static int tk_elm_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  int has_sparse;
  tk_ivec_t *csc_off, *csc_idx;
  int64_t n_tokens;
  double *fw;
  tk_elm_parse_sparse(L, 1, &has_sparse, &csc_off, &csc_idx, &n_tokens, &fw);

  int has_dense;
  double *df;
  int64_t n_dense;
  tk_elm_parse_dense(L, 1, &has_dense, &df, &n_dense);

  if (!has_sparse && !has_dense)
    return luaL_error(L, "create: csc_offsets or dense_features required");

  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_samples");
  int64_t n_hidden = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_hidden");

  lua_getfield(L, 1, "seed");
  uint64_t seed = lua_isnumber(L, -1) ? (uint64_t)lua_tointeger(L, -1) : 42;
  lua_pop(L, 1);

  int sidecar = has_sparse && has_dense && n_dense > 0;
  int64_t out_cols = sidecar ? n_hidden + n_dense : n_hidden;
  uint64_t total = (uint64_t)(n_samples * out_cols);
  tk_dvec_t *out = tk_dvec_create(L, total, NULL, NULL);
  int out_idx = lua_gettop(L);

  tk_dvec_t *cm = tk_dvec_create(L, (uint64_t)n_hidden, NULL, NULL);
  int cm_idx = lua_gettop(L);
  tk_dvec_t *cis = tk_dvec_create(L, (uint64_t)n_hidden, NULL, NULL);
  int cis_idx = lua_gettop(L);

  tk_dvec_t *dm = NULL, *dis = NULL;
  int dm_idx = 0, dis_idx = 0;
  if (has_dense && n_dense > 0) {
    dm = tk_dvec_create(L, (uint64_t)n_dense, NULL, NULL);
    dm_idx = lua_gettop(L);
    dis = tk_dvec_create(L, (uint64_t)n_dense, NULL, NULL);
    dis_idx = lua_gettop(L);
  }

  tk_dvec_t *fw_dvec = NULL;
  int fw_idx = 0;
  if (has_sparse) {
    lua_getfield(L, 1, "feature_weights");
    if (!lua_isnil(L, -1)) {
      fw_dvec = tk_dvec_peek(L, -1, "feature_weights");
      fw_idx = lua_gettop(L);
    } else {
      lua_pop(L, 1);
    }
  }

  tk_elm_project_core(
    out->a, n_samples, n_hidden,
    n_tokens, n_dense, seed,
    csc_off, csc_idx, fw,
    df,
    NULL, NULL,
    NULL, NULL,
    cm->a, cis->a,
    dm ? dm->a : NULL, dis ? dis->a : NULL
  );

  tk_elm_t *e = tk_lua_newuserdata(L, tk_elm_t,
    TK_ELM_MT, tk_elm_mt_fns, tk_elm_gc);
  int Ei = lua_gettop(L);
  e->col_mean = cm;
  e->col_inv_std = cis;
  e->dense_mean = dm;
  e->dense_inv_std = dis;
  e->fw = fw_dvec;
  e->n_hidden = n_hidden;
  e->n_tokens = n_tokens;
  e->n_dense = n_dense;
  e->seed = seed;
  e->destroyed = false;

  lua_newtable(L);
  lua_pushvalue(L, cm_idx);
  lua_setfield(L, -2, "col_mean");
  lua_pushvalue(L, cis_idx);
  lua_setfield(L, -2, "col_inv_std");
  if (dm_idx > 0) {
    lua_pushvalue(L, dm_idx);
    lua_setfield(L, -2, "dense_mean");
    lua_pushvalue(L, dis_idx);
    lua_setfield(L, -2, "dense_inv_std");
  }
  if (fw_idx > 0) {
    lua_pushvalue(L, fw_idx);
    lua_setfield(L, -2, "fw");
  }
  lua_setfenv(L, Ei);

  lua_pushvalue(L, Ei);
  lua_pushvalue(L, out_idx);
  return 2;
}

static int tk_elm_encode_lua (lua_State *L)
{
  tk_elm_t *e = tk_elm_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);

  int has_sparse;
  tk_ivec_t *csc_off, *csc_idx;
  int64_t n_tokens_parsed;
  double *fw_parsed;
  lua_getfield(L, 2, "csc_offsets");
  has_sparse = !lua_isnil(L, -1);
  csc_off = has_sparse ? tk_ivec_peek(L, -1, "csc_offsets") : NULL;
  lua_pop(L, 1);
  csc_idx = NULL;
  if (has_sparse) {
    lua_getfield(L, 2, "csc_indices");
    csc_idx = tk_ivec_peek(L, -1, "csc_indices");
    lua_pop(L, 1);
  }
  n_tokens_parsed = e->n_tokens;
  fw_parsed = e->fw ? e->fw->a : NULL;

  int has_dense;
  double *df;
  lua_getfield(L, 2, "dense_features");
  has_dense = !lua_isnil(L, -1);
  df = has_dense ? tk_dvec_peek(L, -1, "dense_features")->a : NULL;
  lua_pop(L, 1);

  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 2, "encode", "n_samples");

  int sidecar = has_sparse && has_dense && e->n_dense > 0;
  int64_t out_cols = sidecar ? e->n_hidden + e->n_dense : e->n_hidden;
  uint64_t total = (uint64_t)(n_samples * out_cols);
  tk_dvec_t *out = tk_dvec_create(L, total, NULL, NULL);

  tk_elm_project_core(
    out->a, n_samples, e->n_hidden,
    n_tokens_parsed, e->n_dense, e->seed,
    csc_off, csc_idx, fw_parsed,
    df,
    e->dense_mean ? e->dense_mean->a : NULL,
    e->dense_inv_std ? e->dense_inv_std->a : NULL,
    e->col_mean->a, e->col_inv_std->a,
    NULL, NULL,
    NULL, NULL
  );

  return 1;
}

static int tk_elm_persist_lua (lua_State *L) {
  tk_elm_t *e = tk_elm_peek(L, 1);
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting filepath or true");
  tk_lua_fwrite(L, "TKel", 1, 4, fh);
  uint8_t version = 2;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &e->n_hidden, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &e->n_tokens, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &e->n_dense, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &e->seed, sizeof(uint64_t), 1, fh);
  uint8_t has_fw = e->fw ? 1 : 0;
  tk_lua_fwrite(L, &has_fw, sizeof(uint8_t), 1, fh);
  uint8_t mode_byte = 0;
  tk_lua_fwrite(L, &mode_byte, sizeof(uint8_t), 1, fh);
  tk_dvec_persist(L, e->col_mean, fh);
  tk_dvec_persist(L, e->col_inv_std, fh);
  if (e->n_dense > 0) {
    tk_dvec_persist(L, e->dense_mean, fh);
    tk_dvec_persist(L, e->dense_inv_std, fh);
  }
  if (e->fw)
    tk_dvec_persist(L, e->fw, fh);
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

static int tk_elm_load_lua (lua_State *L)
{
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKel", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid elm file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 1 && version != 2) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported elm version %d", (int)version);
  }
  int64_t n_hidden, n_tokens, n_dense;
  uint64_t seed;
  uint8_t has_fw;
  tk_lua_fread(L, &n_hidden, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_tokens, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_dense, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &seed, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &has_fw, sizeof(uint8_t), 1, fh);
  if (version >= 2) {
    uint8_t mode_byte;
    tk_lua_fread(L, &mode_byte, sizeof(uint8_t), 1, fh);
  }
  tk_dvec_t *cm = tk_dvec_load(L, fh);
  int cm_idx = lua_gettop(L);
  tk_dvec_t *cis = tk_dvec_load(L, fh);
  int cis_idx = lua_gettop(L);
  tk_dvec_t *dm = NULL, *dis = NULL;
  int dm_idx = 0, dis_idx = 0;
  if (n_dense > 0) {
    dm = tk_dvec_load(L, fh);
    dm_idx = lua_gettop(L);
    dis = tk_dvec_load(L, fh);
    dis_idx = lua_gettop(L);
  }
  tk_dvec_t *fw = NULL;
  int fw_idx = 0;
  if (has_fw) {
    fw = tk_dvec_load(L, fh);
    fw_idx = lua_gettop(L);
  }
  tk_lua_fclose(L, fh);

  tk_elm_t *e = tk_lua_newuserdata(L, tk_elm_t,
    TK_ELM_MT, tk_elm_mt_fns, tk_elm_gc);
  int Ei = lua_gettop(L);
  e->col_mean = cm;
  e->col_inv_std = cis;
  e->dense_mean = dm;
  e->dense_inv_std = dis;
  e->fw = fw;
  e->n_hidden = n_hidden;
  e->n_tokens = n_tokens;
  e->n_dense = n_dense;
  e->seed = seed;
  e->destroyed = false;

  lua_newtable(L);
  lua_pushvalue(L, cm_idx);
  lua_setfield(L, -2, "col_mean");
  lua_pushvalue(L, cis_idx);
  lua_setfield(L, -2, "col_inv_std");
  if (dm_idx > 0) {
    lua_pushvalue(L, dm_idx);
    lua_setfield(L, -2, "dense_mean");
    lua_pushvalue(L, dis_idx);
    lua_setfield(L, -2, "dense_inv_std");
  }
  if (fw_idx > 0) {
    lua_pushvalue(L, fw_idx);
    lua_setfield(L, -2, "fw");
  }
  lua_setfenv(L, Ei);

  lua_pushvalue(L, Ei);
  return 1;
}

static luaL_Reg tk_elm_mt_fns[] = {
  { "encode", tk_elm_encode_lua },
  { "persist", tk_elm_persist_lua },
  { NULL, NULL }
};

#define TK_ELM_OBJ_MT "tk_elm_obj_t"

typedef struct {
  int64_t k;
  bool destroyed;
} tk_elm_obj_t;

static inline tk_elm_obj_t *tk_elm_obj_peek (lua_State *L, int i) {
  return (tk_elm_obj_t *)luaL_checkudata(L, i, TK_ELM_OBJ_MT);
}

static inline int tk_elm_obj_gc (lua_State *L) {
  tk_elm_obj_t *o = tk_elm_obj_peek(L, 1);
  o->destroyed = true;
  return 0;
}

static int tk_elm_obj_index_lua (lua_State *L) {
  const char *key = lua_tostring(L, 2);
  if (key && (strcmp(key, "ridge") == 0 || strcmp(key, "encoder") == 0)) {
    lua_getfenv(L, 1);
    lua_pushvalue(L, 2);
    lua_gettable(L, -2);
    return 1;
  }
  lua_pushvalue(L, 2);
  lua_gettable(L, lua_upvalueindex(1));
  return 1;
}

static void tk_elm_obj_do_project (
  lua_State *L, int fenv_idx,
  int off_arg, int idx_arg, int n_arg, int df_arg
) {
  lua_getfield(L, fenv_idx, "encoder");
  int enc = lua_gettop(L);
  lua_getfield(L, enc, "encode");
  lua_pushvalue(L, enc);
  lua_newtable(L);
  if (!lua_isnil(L, off_arg)) {
    lua_pushvalue(L, off_arg); lua_setfield(L, -2, "csc_offsets");
    lua_pushvalue(L, idx_arg); lua_setfield(L, -2, "csc_indices");
  }
  lua_pushvalue(L, n_arg); lua_setfield(L, -2, "n_samples");
  if (df_arg > 0 && !lua_isnil(L, df_arg)) {
    lua_pushvalue(L, df_arg); lua_setfield(L, -2, "dense_features");
  }
  lua_call(L, 2, 1);
}

static int tk_elm_obj_wrap_lua (lua_State *L) {
  luaL_checkudata(L, 1, TK_ELM_MT);
  luaL_checktype(L, 2, LUA_TUSERDATA);
  int64_t k = lua_isnil(L, 3) ? 0 : (int64_t)luaL_checkinteger(L, 3);
  tk_elm_obj_t *o = tk_lua_newuserdata(L, tk_elm_obj_t,
    TK_ELM_OBJ_MT, NULL, tk_elm_obj_gc);
  int oi = lua_gettop(L);
  o->k = k;
  o->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, 1);
  lua_setfield(L, -2, "encoder");
  lua_pushvalue(L, 2);
  lua_setfield(L, -2, "ridge");
  lua_setfenv(L, oi);
  lua_pushvalue(L, oi);
  return 1;
}

static int tk_elm_obj_project_lua (lua_State *L) {
  tk_elm_obj_peek(L, 1);
  lua_settop(L, 5);
  lua_getfenv(L, 1);
  tk_elm_obj_do_project(L, 6, 2, 3, 4, 5);
  return 1;
}

static int tk_elm_obj_transform_lua (lua_State *L) {
  tk_elm_obj_peek(L, 1);
  lua_settop(L, 5);
  lua_getfenv(L, 1);
  tk_elm_obj_do_project(L, 6, 2, 3, 4, 5);
  int h = lua_gettop(L);
  lua_getfield(L, 6, "ridge");
  lua_getfield(L, -1, "transform");
  lua_pushvalue(L, -2);
  lua_pushvalue(L, h);
  lua_pushvalue(L, 4);
  lua_call(L, 3, 1);
  return 1;
}

static int tk_elm_obj_label_lua (lua_State *L) {
  tk_elm_obj_t *o = tk_elm_obj_peek(L, 1);
  lua_settop(L, 6);
  int64_t lk = lua_isnil(L, 5) ? o->k : (int64_t)luaL_checkinteger(L, 5);
  lua_getfenv(L, 1);
  tk_elm_obj_do_project(L, 7, 2, 3, 4, 6);
  int h = lua_gettop(L);
  lua_getfield(L, 7, "ridge");
  lua_getfield(L, -1, "label");
  lua_pushvalue(L, -2);
  lua_pushvalue(L, h);
  lua_pushvalue(L, 4);
  lua_pushinteger(L, (lua_Integer)lk);
  lua_call(L, 4, 3);
  return 3;
}

static int tk_elm_obj_persist_lua (lua_State *L) {
  tk_elm_obj_t *o = tk_elm_obj_peek(L, 1);
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  FILE *fh;
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting filepath or true");
  lua_getfenv(L, 1);
  int fi = lua_gettop(L);
  lua_getfield(L, fi, "encoder");
  lua_getfield(L, -1, "persist");
  lua_pushvalue(L, -2);
  lua_pushboolean(L, 1);
  lua_call(L, 2, 1);
  size_t enc_len;
  const char *enc_data = lua_tolstring(L, -1, &enc_len);
  lua_getfield(L, fi, "ridge");
  lua_getfield(L, -1, "persist");
  lua_pushvalue(L, -2);
  lua_pushboolean(L, 1);
  lua_call(L, 2, 1);
  size_t ridge_len;
  const char *ridge_data = lua_tolstring(L, -1, &ridge_len);
  tk_lua_fwrite(L, "TKelm", 1, 5, fh);
  uint8_t version = 4;
  tk_lua_fwrite(L, (char *)&version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, (char *)&o->k, sizeof(int64_t), 1, fh);
  uint64_t enc_len64 = (uint64_t)enc_len;
  tk_lua_fwrite(L, (char *)&enc_len64, sizeof(uint64_t), 1, fh);
  tk_lua_fwrite(L, (char *)enc_data, 1, enc_len, fh);
  tk_lua_fwrite(L, (char *)ridge_data, 1, ridge_len, fh);
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

static int tk_elm_obj_load_lua (lua_State *L) {
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[5];
  tk_lua_fread(L, magic, 1, 5, fh);
  if (memcmp(magic, "TKelm", 5) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid elm obj file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 4) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported elm obj version %d", (int)version);
  }
  int64_t k;
  tk_lua_fread(L, &k, sizeof(int64_t), 1, fh);
  uint64_t enc_len;
  tk_lua_fread(L, &enc_len, sizeof(uint64_t), 1, fh);
  char *enc_buf = (char *)malloc(enc_len > 0 ? (size_t)enc_len : 1);
  if (enc_len > 0)
    tk_lua_fread(L, enc_buf, 1, (size_t)enc_len, fh);
  size_t header_used = 5 + 1 + 8 + 8 + (size_t)enc_len;
  size_t ridge_len;
  if (isstr) {
    ridge_len = len - header_used;
  } else {
    long cur = ftell(fh);
    fseek(fh, 0, SEEK_END);
    ridge_len = (size_t)(ftell(fh) - cur);
    fseek(fh, cur, SEEK_SET);
  }
  char *ridge_buf = (char *)malloc(ridge_len > 0 ? ridge_len : 1);
  if (ridge_len > 0)
    tk_lua_fread(L, ridge_buf, 1, ridge_len, fh);
  tk_lua_fclose(L, fh);
  lua_getglobal(L, "require");
  lua_pushliteral(L, "santoku.learn.elm");
  lua_call(L, 1, 1);
  lua_getfield(L, -1, "load");
  lua_remove(L, -2);
  lua_pushlstring(L, enc_buf, (size_t)enc_len);
  lua_pushboolean(L, 1);
  lua_call(L, 2, 1);
  int enc_idx = lua_gettop(L);
  free(enc_buf);
  lua_getglobal(L, "require");
  lua_pushliteral(L, "santoku.learn.ridge");
  lua_call(L, 1, 1);
  lua_getfield(L, -1, "load");
  lua_remove(L, -2);
  lua_pushlstring(L, ridge_buf, ridge_len);
  lua_pushboolean(L, 1);
  lua_call(L, 2, 1);
  int ri_idx = lua_gettop(L);
  free(ridge_buf);
  tk_elm_obj_t *o = tk_lua_newuserdata(L, tk_elm_obj_t,
    TK_ELM_OBJ_MT, NULL, tk_elm_obj_gc);
  int oi = lua_gettop(L);
  o->k = k;
  o->destroyed = false;
  lua_newtable(L);
  lua_pushvalue(L, enc_idx);
  lua_setfield(L, -2, "encoder");
  lua_pushvalue(L, ri_idx);
  lua_setfield(L, -2, "ridge");
  lua_setfenv(L, oi);
  lua_pushvalue(L, oi);
  return 1;
}

static luaL_Reg tk_elm_obj_mt_fns[] = {
  { "project", tk_elm_obj_project_lua },
  { "transform", tk_elm_obj_transform_lua },
  { "label", tk_elm_obj_label_lua },
  { "persist", tk_elm_obj_persist_lua },
  { NULL, NULL }
};

static luaL_Reg tk_elm_fns[] = {
  { "create", tk_elm_create_lua },
  { "load", tk_elm_load_lua },
  { "wrap", tk_elm_obj_wrap_lua },
  { "obj_load", tk_elm_obj_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_elm (lua_State *L)
{
  luaL_newmetatable(L, TK_ELM_OBJ_MT);
  lua_pushstring(L, TK_ELM_OBJ_MT);
  lua_setfield(L, -2, "__name");
  lua_pushcfunction(L, tk_elm_obj_gc);
  lua_setfield(L, -2, "__gc");
  lua_newtable(L);
  tk_lua_register(L, tk_elm_obj_mt_fns, 0);
  lua_pushcclosure(L, tk_elm_obj_index_lua, 1);
  lua_setfield(L, -2, "__index");
  lua_pop(L, 1);
  lua_newtable(L);
  tk_lua_register(L, tk_elm_fns, 0);
  return 1;
}
