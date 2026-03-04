#include <lua.h>
#include <lauxlib.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/cvec.h>

#define TK_ELM_MT "tk_elm_t"

typedef struct {
  tk_dvec_t *fw;
  int64_t n_hidden;
  int64_t n_tokens;
  uint64_t seed;
  uint8_t mode;
  uint8_t norm;
  uint8_t dense;
  bool destroyed;
} tk_elm_t;

static inline tk_elm_t *tk_elm_peek (lua_State *L, int i) {
  return (tk_elm_t *)luaL_checkudata(L, i, TK_ELM_MT);
}

static inline int tk_elm_gc (lua_State *L) {
  tk_elm_t *e = tk_elm_peek(L, 1);
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
  int64_t n_tokens, uint64_t seed,
  uint8_t mode, uint8_t norm,
  tk_ivec_t *csc_off, tk_ivec_t *csc_idx, double *fw,
  double *dense_in, int64_t n_input_dims
)
{
  int has_csc = csc_off != NULL;
  int has_dense = dense_in != NULL;

  double *inv_norm = NULL;
  if (has_dense && norm != 1) {
    inv_norm = (double *)calloc((uint64_t)n_samples, sizeof(double));
    for (int64_t s = 0; s < n_samples; s++) {
      double acc = 0.0;
      for (int64_t d = 0; d < n_input_dims; d++) {
        double v = dense_in[s * n_input_dims + d];
        if (norm == 0) acc += v * v;
        else acc += fabs(v);
      }
      inv_norm[s] = acc > 0.0 ? (norm == 0 ? 1.0 / sqrt(acc) : 1.0 / acc) : 0.0;
    }
  } else if (has_csc && norm != 1) {
    inv_norm = (double *)calloc((uint64_t)n_samples, sizeof(double));
    for (int64_t t = 0; t < n_tokens; t++) {
      double fw_sq = fw ? fw[t] * fw[t] : 1.0;
      int64_t lo = csc_off->a[t], hi = csc_off->a[t + 1];
      for (int64_t i = lo; i < hi; i++)
        inv_norm[csc_idx->a[i]] += fw_sq;
    }
    if (norm == 0) {
      for (int64_t s = 0; s < n_samples; s++) {
        double n2 = inv_norm[s];
        inv_norm[s] = n2 > 0.0 ? 1.0 / sqrt(n2) : 0.0;
      }
    } else {
      for (int64_t s = 0; s < n_samples; s++) {
        double n1 = inv_norm[s];
        inv_norm[s] = n1 > 0.0 ? 1.0 / n1 : 0.0;
      }
    }
  }

  if (has_dense) {
    double *normed = dense_in;
    if (inv_norm) {
      normed = (double *)malloc((uint64_t)n_samples * (uint64_t)n_input_dims * sizeof(double));
      #pragma omp parallel for schedule(static)
      for (int64_t s = 0; s < n_samples; s++) {
        double nv = inv_norm[s];
        const double *src = dense_in + s * n_input_dims;
        double *dst = normed + s * n_input_dims;
        for (int64_t d = 0; d < n_input_dims; d++)
          dst[d] = src[d] * nv;
      }
    }
    int64_t bh = n_hidden;
    {
      int64_t max_bh = (16 * 1024 * 1024 / (int64_t)sizeof(double)) / (n_input_dims > 0 ? n_input_dims : 1);
      if (max_bh < 1) max_bh = 1;
      if (bh > max_bh) bh = max_bh;
    }
    double *Wbuf = (double *)malloc((uint64_t)bh * (uint64_t)n_input_dims * sizeof(double));
    double *biases = (double *)malloc((uint64_t)bh * sizeof(double));
    for (int64_t h0 = 0; h0 < n_hidden; h0 += bh) {
      int64_t cur = (h0 + bh <= n_hidden) ? bh : (n_hidden - h0);
      for (int64_t j = 0; j < cur; j++) {
        int64_t ah = h0 + j;
        uint64_t rng_h = (mode == 10) ? (uint64_t)(ah / 2) : (uint64_t)ah;
        uint64_t rng = seed ^ (rng_h * (uint64_t)0x517cc1b727220a95);
        elm_splitmix64(&rng);
        double *Wr = Wbuf + j * n_input_dims;
        for (int64_t d = 0; d < n_input_dims; d++)
          Wr[d] = elm_rand_normal(&rng);
        biases[j] = elm_rand_normal(&rng);
      }
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        (int)n_samples, (int)cur, (int)n_input_dims,
        1.0, normed, (int)n_input_dims,
        Wbuf, (int)n_input_dims,
        0.0, out + h0, (int)n_hidden);
      #pragma omp parallel for schedule(static)
      for (int64_t s = 0; s < n_samples; s++) {
        double *row = out + s * n_hidden + h0;
        for (int64_t j = 0; j < cur; j++) {
          double v = row[j] + biases[j];
          double a;
          switch (mode) {
            case 1: a = 1.0 / (1.0 + exp(-v)); break;
            case 2: a = tanh(v); break;
            case 3: { double t = 0.7978845608 * (v + 0.044715 * v * v * v);
                      a = 0.5 * v * (1.0 + tanh(t)); break; }
            case 4: a = log(1.0 + exp(v)); break;
            case 5: a = v > 0.0 ? v : exp(v) - 1.0; break;
            case 6: a = sin(v); break;
            case 7: a = v; break;
            case 8: a = v > 0.0 ? 1.0507009873554805 * v
                      : 1.0507009873554805 * 1.6732632423543773 * (exp(v) - 1.0); break;
            case 9: a = v * tanh(log(1.0 + exp(v))); break;
            case 10: a = ((h0 + j) % 2 == 0) ? sin(v) : cos(v); break;
            case 11: a = exp(-v * v); break;
            case 12: a = 1.0 - exp(-v * v * 0.5); break;
            default: a = v > 0.0 ? v : 0.0; break;
          }
          row[j] = a;
        }
      }
    }
    free(Wbuf);
    free(biases);
    if (normed != dense_in) free(normed);
    free(inv_norm);
    return;
  }

  #pragma omp parallel
  {
    double *col = (double *)malloc((uint64_t)n_samples * sizeof(double));
    #pragma omp for schedule(static)
    for (int64_t h = 0; h < n_hidden; h++) {
      memset(col, 0, (uint64_t)n_samples * sizeof(double));
      uint64_t rng_h = (mode == 10) ? (uint64_t)(h / 2) : (uint64_t)h;
      uint64_t rng = seed ^ (rng_h * (uint64_t)0x517cc1b727220a95);
      elm_splitmix64(&rng);
      for (int64_t t = 0; t < n_tokens; t++) {
        double w = elm_rand_normal(&rng);
        if (fw) w *= fw[t];
        int64_t lo = csc_off->a[t];
        int64_t hi = csc_off->a[t + 1];
        if (inv_norm)
          for (int64_t i = lo; i < hi; i++)
            col[csc_idx->a[i]] += w * inv_norm[csc_idx->a[i]];
        else
          for (int64_t i = lo; i < hi; i++)
            col[csc_idx->a[i]] += w;
      }
      double bias = elm_rand_normal(&rng);
      for (int64_t s = 0; s < n_samples; s++) {
        double v = col[s] + bias;
        double a;
        switch (mode) {
          case 1: a = 1.0 / (1.0 + exp(-v)); break;
          case 2: a = tanh(v); break;
          case 3: { double t = 0.7978845608 * (v + 0.044715 * v * v * v);
                    a = 0.5 * v * (1.0 + tanh(t)); break; }
          case 4: a = log(1.0 + exp(v)); break;
          case 5: a = v > 0.0 ? v : exp(v) - 1.0; break;
          case 6: a = sin(v); break;
          case 7: a = v; break;
          case 8: a = v > 0.0 ? 1.0507009873554805 * v
                    : 1.0507009873554805 * 1.6732632423543773 * (exp(v) - 1.0); break;
          case 9: a = v * tanh(log(1.0 + exp(v))); break;
          case 10: a = (h % 2 == 0) ? sin(v) : cos(v); break;
          case 11: a = exp(-v * v); break;
          case 12: a = 1.0 - exp(-v * v * 0.5); break;
          default: a = v > 0.0 ? v : 0.0; break;
        }
        out[s * n_hidden + h] = a;
      }
    }
    free(col);
  }
  free(inv_norm);
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

static int tk_elm_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  int has_dense_in = 0;
  tk_dvec_t *codes_dvec = NULL;
  int64_t n_input_dims = 0;
  int has_sparse = 0;
  tk_ivec_t *csc_off = NULL, *csc_idx = NULL;
  int64_t n_tokens = 0;
  double *fw = NULL;

  lua_getfield(L, 1, "codes");
  has_dense_in = !lua_isnil(L, -1);
  if (has_dense_in) {
    codes_dvec = tk_dvec_peek(L, -1, "codes");
    n_input_dims = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_input_dims");
  }
  lua_pop(L, 1);
  if (!has_dense_in)
    tk_elm_parse_sparse(L, 1, &has_sparse, &csc_off, &csc_idx, &n_tokens, &fw);

  if (!has_sparse && !has_dense_in)
    return luaL_error(L, "create: codes or csc_offsets required");

  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_samples");
  int64_t n_hidden = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_hidden");
  uint64_t seed = ((uint64_t)tk_fast_random() << 32) | tk_fast_random();

  uint8_t mode = 1;
  lua_getfield(L, 1, "mode");
  if (lua_isstring(L, -1)) {
    const char *ms = lua_tostring(L, -1);
    if (strcmp(ms, "relu") == 0) mode = 0;
    else if (strcmp(ms, "sigmoid") == 0) mode = 1;
    else if (strcmp(ms, "tanh") == 0) mode = 2;
    else if (strcmp(ms, "gelu") == 0) mode = 3;
    else if (strcmp(ms, "softplus") == 0) mode = 4;
    else if (strcmp(ms, "elu") == 0) mode = 5;
    else if (strcmp(ms, "sin") == 0) mode = 6;
    else if (strcmp(ms, "linear") == 0) mode = 7;
    else if (strcmp(ms, "selu") == 0) mode = 8;
    else if (strcmp(ms, "mish") == 0) mode = 9;
    else if (strcmp(ms, "rff") == 0) mode = 10;
    else if (strcmp(ms, "gaussian") == 0) mode = 11;
    else if (strcmp(ms, "welsch") == 0) mode = 12;
  }
  lua_pop(L, 1);

  uint8_t norm = 0;
  lua_getfield(L, 1, "norm");
  if (lua_isstring(L, -1)) {
    const char *ns = lua_tostring(L, -1);
    if (strcmp(ns, "l2") == 0) norm = 0;
    else if (strcmp(ns, "none") == 0) norm = 1;
    else if (strcmp(ns, "l1") == 0) norm = 2;
  }
  lua_pop(L, 1);

  uint64_t total = (uint64_t)(n_samples * n_hidden);
  tk_dvec_t *out = tk_dvec_create(L, total, NULL, NULL);
  out->n = total;
  int out_idx = lua_gettop(L);

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
    has_dense_in ? n_input_dims : n_tokens, seed, mode, norm,
    csc_off, csc_idx, fw,
    has_dense_in ? codes_dvec->a : NULL, n_input_dims
  );

  tk_elm_t *e = tk_lua_newuserdata(L, tk_elm_t,
    TK_ELM_MT, tk_elm_mt_fns, tk_elm_gc);
  int Ei = lua_gettop(L);
  e->fw = fw_dvec;
  e->n_hidden = n_hidden;
  e->n_tokens = has_dense_in ? n_input_dims : n_tokens;
  e->seed = seed;
  e->mode = mode;
  e->norm = norm;
  e->dense = has_dense_in ? 1 : 0;
  e->destroyed = false;

  lua_newtable(L);
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
  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 2, "encode", "n_samples");

  double *dense_in = NULL;
  tk_ivec_t *csc_off = NULL, *csc_idx = NULL;
  double *fw_parsed = NULL;

  if (e->dense) {
    lua_getfield(L, 2, "codes");
    if (lua_isnil(L, -1))
      return luaL_error(L, "encode: dense encoder requires codes");
    tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
    dense_in = codes->a;
    lua_pop(L, 1);
  } else {
    lua_getfield(L, 2, "csc_offsets");
    int has_sparse = !lua_isnil(L, -1);
    csc_off = has_sparse ? tk_ivec_peek(L, -1, "csc_offsets") : NULL;
    lua_pop(L, 1);
    if (has_sparse) {
      lua_getfield(L, 2, "csc_indices");
      csc_idx = tk_ivec_peek(L, -1, "csc_indices");
      lua_pop(L, 1);
    }
    fw_parsed = e->fw ? e->fw->a : NULL;
  }

  uint64_t total = (uint64_t)(n_samples * e->n_hidden);
  tk_dvec_t *out = tk_dvec_create(L, total, NULL, NULL);
  out->n = total;

  tk_elm_project_core(
    out->a, n_samples, e->n_hidden,
    e->n_tokens, e->seed, e->mode, e->norm,
    csc_off, csc_idx, fw_parsed,
    dense_in, e->n_tokens
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
  uint8_t version = 11;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &e->n_hidden, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &e->n_tokens, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &e->seed, sizeof(uint64_t), 1, fh);
  uint8_t has_fw = e->fw ? 1 : 0;
  tk_lua_fwrite(L, &has_fw, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &e->mode, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &e->norm, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &e->dense, sizeof(uint8_t), 1, fh);
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
  if (version != 11) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported elm version %d", (int)version);
  }
  int64_t n_hidden, n_tokens;
  uint64_t seed;
  uint8_t has_fw;
  tk_lua_fread(L, &n_hidden, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &n_tokens, sizeof(int64_t), 1, fh);
  tk_lua_fread(L, &seed, sizeof(uint64_t), 1, fh);
  tk_lua_fread(L, &has_fw, sizeof(uint8_t), 1, fh);
  uint8_t mode_byte;
  tk_lua_fread(L, &mode_byte, sizeof(uint8_t), 1, fh);
  uint8_t norm_byte;
  tk_lua_fread(L, &norm_byte, sizeof(uint8_t), 1, fh);
  uint8_t dense_byte;
  tk_lua_fread(L, &dense_byte, sizeof(uint8_t), 1, fh);
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
  e->fw = fw;
  e->n_hidden = n_hidden;
  e->n_tokens = n_tokens;
  e->seed = seed;
  e->mode = mode_byte;
  e->norm = norm_byte;
  e->dense = dense_byte;
  e->destroyed = false;

  lua_newtable(L);
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

static luaL_Reg tk_elm_fns[] = {
  { "create", tk_elm_create_lua },
  { "load", tk_elm_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_elm (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_elm_fns, 0);
  return 1;
}
