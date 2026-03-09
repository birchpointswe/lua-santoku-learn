#include <lua.h>
#include <lauxlib.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <lapacke.h>
#include <cblas.h>
#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/cvec.h>
#include <santoku/rvec.h>
#include <santoku/learn/gram.h>
#include <santoku/learn/activation.h>

#define TK_ELM_MT "tk_elm_t"

typedef struct {
  tk_dvec_t *fw;
  int64_t n_hidden;
  int64_t n_tokens;
  uint64_t seed;
  uint8_t mode;
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

static uint8_t tk_elm_mode_from_str (const char *ms)
{
  uint8_t m = tk_activation_mode_from_str(ms);
  return m ? m : 2;
}

static void tk_elm_project_core (
  double *out, int64_t n_samples, int64_t n_hidden,
  int64_t n_input_dims, uint64_t seed, uint8_t mode,
  double *dense_in
)
{
  double *normed = (double *)malloc((uint64_t)n_samples * (uint64_t)n_input_dims * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (int64_t s = 0; s < n_samples; s++) {
    const double *src = dense_in + s * n_input_dims;
    double *dst = normed + s * n_input_dims;
    double acc = 0.0;
    for (int64_t d = 0; d < n_input_dims; d++)
      acc += src[d] * src[d];
    double inv = acc > 0.0 ? 1.0 / sqrt(acc) : 0.0;
    for (int64_t d = 0; d < n_input_dims; d++)
      dst[d] = src[d] * inv;
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
      uint64_t rng_h = (mode == 11) ? (uint64_t)(ah / 2) : (uint64_t)ah;
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
        row[j] = tk_activate(row[j] + biases[j], mode, h0 + j);
      }
    }
  }
  free(Wbuf);
  free(biases);
  free(normed);
}

static int tk_elm_create_lua (lua_State *L)
{
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);
  int64_t n_input_dims = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_input_dims");
  int64_t n_hidden = (int64_t)tk_lua_fcheckunsigned(L, 1, "create", "n_hidden");
  uint64_t seed = ((uint64_t)tk_fast_random() << 32) | tk_fast_random();
  uint8_t mode = 2;
  lua_getfield(L, 1, "mode");
  if (lua_isstring(L, -1))
    mode = tk_elm_mode_from_str(lua_tostring(L, -1));
  lua_pop(L, 1);
  tk_elm_t *e = tk_lua_newuserdata(L, tk_elm_t,
    TK_ELM_MT, tk_elm_mt_fns, tk_elm_gc);
  int Ei = lua_gettop(L);
  e->fw = NULL;
  e->n_hidden = n_hidden;
  e->n_tokens = n_input_dims;
  e->seed = seed;
  e->mode = mode;
  e->dense = 1;
  e->destroyed = false;
  lua_newtable(L);
  lua_setfenv(L, Ei);
  lua_pushvalue(L, Ei);
  return 1;
}

static void tk_elm_project_binary (
  double *out, int64_t n_samples, int64_t n_hidden,
  int64_t n_input_dims, uint64_t seed, uint8_t mode,
  const uint8_t *packed)
{
  int64_t d = n_input_dims;
  int64_t row_bytes = (int64_t)TK_CVEC_BITS_BYTES((uint64_t)d);
  double inv_norm = 1.0 / sqrt((double)d);
  int64_t bh = n_hidden;
  {
    int64_t max_bh = (16 * 1024 * 1024 / (int64_t)sizeof(double)) / (d > 0 ? d : 1);
    if (max_bh < 1) max_bh = 1;
    if (bh > max_bh) bh = max_bh;
  }
  int64_t bs = 256;
  {
    int64_t max_bs = (16 * 1024 * 1024 / (int64_t)sizeof(double)) / (d > 0 ? d : 1);
    if (max_bs < 1) max_bs = 1;
    if (bs > max_bs) bs = max_bs;
  }
  double *Wbuf = (double *)malloc((uint64_t)bh * (uint64_t)d * sizeof(double));
  double *biases = (double *)malloc((uint64_t)bh * sizeof(double));
  double *Xbuf = (double *)malloc((uint64_t)bs * (uint64_t)d * sizeof(double));
  for (int64_t s0 = 0; s0 < n_samples; s0 += bs) {
    int64_t cur_s = (s0 + bs <= n_samples) ? bs : (n_samples - s0);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < cur_s; i++) {
      const uint8_t *row = packed + (s0 + i) * row_bytes;
      double *dst = Xbuf + i * d;
      for (int64_t j = 0; j < d; j++)
        dst[j] = ((row[j / 8] >> (j % 8)) & 1) ? inv_norm : -inv_norm;
    }
    for (int64_t h0 = 0; h0 < n_hidden; h0 += bh) {
      int64_t cur_h = (h0 + bh <= n_hidden) ? bh : (n_hidden - h0);
      for (int64_t j = 0; j < cur_h; j++) {
        int64_t ah = h0 + j;
        uint64_t rng_h = (mode == 11) ? (uint64_t)(ah / 2) : (uint64_t)ah;
        uint64_t rng = seed ^ (rng_h * (uint64_t)0x517cc1b727220a95);
        elm_splitmix64(&rng);
        double *Wr = Wbuf + j * d;
        for (int64_t dd = 0; dd < d; dd++)
          Wr[dd] = elm_rand_normal(&rng);
        biases[j] = elm_rand_normal(&rng);
      }
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        (int)cur_s, (int)cur_h, (int)d,
        1.0, Xbuf, (int)d, Wbuf, (int)d,
        0.0, out + s0 * n_hidden + h0, (int)n_hidden);
      #pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < cur_s; i++) {
        double *hr = out + (s0 + i) * n_hidden + h0;
        for (int64_t j = 0; j < cur_h; j++) {
          hr[j] = tk_activate(hr[j] + biases[j], mode, h0 + j);
        }
      }
    }
  }
  free(Wbuf);
  free(biases);
  free(Xbuf);
}

static int tk_elm_encode_lua (lua_State *L)
{
  tk_elm_t *e = tk_elm_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  int64_t n_samples = (int64_t)tk_lua_fcheckunsigned(L, 2, "encode", "n_samples");
  lua_getfield(L, 2, "codes");
  if (lua_isnil(L, -1))
    return luaL_error(L, "encode: codes required");
  tk_dvec_t *codes_dvec = tk_dvec_peekopt(L, -1);
  lua_pop(L, 1);
  uint64_t total = (uint64_t)(n_samples * e->n_hidden);
  tk_dvec_t *out = tk_dvec_create(L, total, NULL, NULL);
  out->n = total;
  if (codes_dvec) {
    tk_elm_project_core(
      out->a, n_samples, e->n_hidden,
      e->n_tokens, e->seed, e->mode,
      codes_dvec->a);
  } else {
    lua_getfield(L, 2, "codes");
    tk_cvec_t *codes_cvec = tk_cvec_peek(L, -1, "codes");
    lua_pop(L, 1);
    tk_elm_project_binary(
      out->a, n_samples, e->n_hidden,
      e->n_tokens, e->seed, e->mode,
      (const uint8_t *)codes_cvec->a);
  }
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
  uint8_t version = 13;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &e->n_hidden, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &e->n_tokens, sizeof(int64_t), 1, fh);
  tk_lua_fwrite(L, &e->seed, sizeof(uint64_t), 1, fh);
  uint8_t has_fw = e->fw ? 1 : 0;
  tk_lua_fwrite(L, &has_fw, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &e->mode, sizeof(uint8_t), 1, fh);
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
  if (version != 12 && version != 13) {
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
  if (version == 12) mode_byte = (uint8_t)(mode_byte + 1);
  uint8_t dense_byte;
  tk_lua_fread(L, &dense_byte, sizeof(uint8_t), 1, fh);
  if (dense_byte != 1) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported elm format (CSC models no longer supported)");
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
  e->fw = fw;
  e->n_hidden = n_hidden;
  e->n_tokens = n_tokens;
  e->seed = seed;
  e->mode = mode_byte;
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

static int tk_elm_activate_lua (lua_State *L)
{
  tk_dvec_t *raw = tk_dvec_peek(L, 1, "raw");
  int64_t n_samples = (int64_t)luaL_checkinteger(L, 2);
  int64_t n_hidden = (int64_t)luaL_checkinteger(L, 3);
  uint8_t mode = tk_elm_mode_from_str(luaL_checkstring(L, 4));
  int64_t total = n_samples * n_hidden;
  tk_dvec_t *out = tk_dvec_peekopt(L, 5);
  if (out) {
    tk_dvec_ensure(out, (uint64_t)total);
    out->n = (uint64_t)total;
    lua_pushvalue(L, 5);
  } else {
    out = tk_dvec_create(L, (uint64_t)total, NULL, NULL);
    out->n = (uint64_t)total;
  }
  #pragma omp parallel for schedule(static)
  for (int64_t s = 0; s < n_samples; s++) {
    double *src = raw->a + s * n_hidden;
    double *dst = out->a + s * n_hidden;
    for (int64_t h = 0; h < n_hidden; h++)
      dst[h] = tk_activate(src[h], mode, h);
  }
  return 1;
}

static int tk_elm_set_mode_lua (lua_State *L)
{
  tk_elm_t *e = tk_elm_peek(L, 1);
  uint8_t m = tk_activation_mode_from_str(luaL_checkstring(L, 2));
  e->mode = m ? m : 2;
  return 0;
}

static int tk_elm_out_d_lua (lua_State *L)
{
  tk_elm_t *e = tk_elm_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)e->n_hidden);
  return 1;
}

static luaL_Reg tk_elm_mt_fns[] = {
  { "encode", tk_elm_encode_lua },
  { "persist", tk_elm_persist_lua },
  { "set_mode", tk_elm_set_mode_lua },
  { "out_d", tk_elm_out_d_lua },
  { NULL, NULL }
};

static luaL_Reg tk_elm_fns[] = {
  { "create", tk_elm_create_lua },
  { "load", tk_elm_load_lua },
  { "activate", tk_elm_activate_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_elm (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_elm_fns, 0);
  return 1;
}
