#include <santoku/lua/utils.h>
#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#define TK_NORMALIZER_MT "tk_normalizer_t"

typedef struct {
  tk_dvec_t *src_mean;
  tk_dvec_t *src_std;
  tk_dvec_t *dst_mean;
  tk_dvec_t *dst_std;
  uint64_t n_dims;
  bool destroyed;
} tk_normalizer_t;

static inline tk_normalizer_t *tk_normalizer_peek (lua_State *L, int i) {
  return (tk_normalizer_t *)luaL_checkudata(L, i, TK_NORMALIZER_MT);
}

static inline int tk_normalizer_gc (lua_State *L) {
  tk_normalizer_t *n = tk_normalizer_peek(L, 1);
  n->src_mean = NULL;
  n->src_std = NULL;
  n->dst_mean = NULL;
  n->dst_std = NULL;
  n->destroyed = true;
  return 0;
}

static inline void tk_normalizer_compute_stats (
  const double *data, uint64_t n_samples, uint64_t n_dims,
  double *mean_out, double *std_out
) {
  for (uint64_t d = 0; d < n_dims; d++) {
    double sum = 0.0;
    for (uint64_t i = 0; i < n_samples; i++)
      sum += data[i * n_dims + d];
    double m = sum / (double)n_samples;
    mean_out[d] = m;
    double var = 0.0;
    for (uint64_t i = 0; i < n_samples; i++) {
      double diff = data[i * n_dims + d] - m;
      var += diff * diff;
    }
    double s = sqrt(var / (double)n_samples);
    std_out[d] = s > 1e-15 ? s : 1e-15;
  }
}

static int tk_normalizer_encode_lua (lua_State *L) {
  tk_normalizer_t *n = tk_normalizer_peek(L, 1);
  tk_dvec_t *input = tk_dvec_peek(L, 2, "input");
  uint64_t nd = n->n_dims;
  uint64_t ns = input->n / nd;
  tk_dvec_t *out = tk_dvec_create(L, ns * nd, 0, 0);
  const double *sm = n->src_mean->a;
  const double *ss = n->src_std->a;
  const double *dm = n->dst_mean->a;
  const double *ds = n->dst_std->a;
  for (uint64_t i = 0; i < ns; i++) {
    for (uint64_t d = 0; d < nd; d++) {
      uint64_t idx = i * nd + d;
      out->a[idx] = (input->a[idx] - sm[d]) / ss[d] * ds[d] + dm[d];
    }
  }
  return 1;
}

static int tk_normalizer_n_dims_lua (lua_State *L) {
  tk_normalizer_t *n = tk_normalizer_peek(L, 1);
  lua_pushinteger(L, (lua_Integer)n->n_dims);
  return 1;
}

static int tk_normalizer_persist_lua (lua_State *L) {
  tk_normalizer_t *n = tk_normalizer_peek(L, 1);
  if (n->destroyed)
    return luaL_error(L, "cannot persist a destroyed normalizer");
  FILE *fh;
  int t = lua_type(L, 2);
  bool tostr = t == LUA_TBOOLEAN && lua_toboolean(L, 2);
  if (t == LUA_TSTRING)
    fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  else if (tostr)
    fh = tk_lua_tmpfile(L);
  else
    return tk_lua_verror(L, 2, "persist", "expecting either a filepath or true (for string serialization)");
  tk_lua_fwrite(L, "TKnm", 1, 4, fh);
  uint8_t version = 1;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  tk_lua_fwrite(L, &n->n_dims, sizeof(uint64_t), 1, fh);
  tk_dvec_persist(L, n->src_mean, fh);
  tk_dvec_persist(L, n->src_std, fh);
  tk_dvec_persist(L, n->dst_mean, fh);
  tk_dvec_persist(L, n->dst_std, fh);
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

static luaL_Reg tk_normalizer_mt_fns[] = {
  { "encode", tk_normalizer_encode_lua },
  { "n_dims", tk_normalizer_n_dims_lua },
  { "persist", tk_normalizer_persist_lua },
  { NULL, NULL }
};

static inline void tk_normalizer_init_fenv (
  lua_State *L, int enc_idx,
  int sm_idx, int ss_idx, int dm_idx, int ds_idx
) {
  lua_newtable(L);
  lua_pushvalue(L, sm_idx);
  lua_setfield(L, -2, "src_mean");
  lua_pushvalue(L, ss_idx);
  lua_setfield(L, -2, "src_std");
  lua_pushvalue(L, dm_idx);
  lua_setfield(L, -2, "dst_mean");
  lua_pushvalue(L, ds_idx);
  lua_setfield(L, -2, "dst_std");
  lua_setfenv(L, enc_idx);
}

static int tk_normalizer_create_lua (lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  lua_getfield(L, 1, "source");
  tk_dvec_t *source = tk_dvec_peek(L, -1, "source");
  lua_pop(L, 1);

  lua_getfield(L, 1, "target");
  tk_dvec_t *target = tk_dvec_peek(L, -1, "target");
  lua_pop(L, 1);

  lua_getfield(L, 1, "n_samples_source");
  uint64_t n_src = tk_lua_checkunsigned(L, -1, "n_samples_source");
  lua_pop(L, 1);

  lua_getfield(L, 1, "n_samples_target");
  uint64_t n_dst = tk_lua_checkunsigned(L, -1, "n_samples_target");
  lua_pop(L, 1);

  lua_getfield(L, 1, "n_dims");
  uint64_t n_dims = tk_lua_checkunsigned(L, -1, "n_dims");
  lua_pop(L, 1);

  tk_dvec_t *src_mean = tk_dvec_create(L, n_dims, 0, 0);
  int sm_idx = lua_gettop(L);
  tk_dvec_t *src_std = tk_dvec_create(L, n_dims, 0, 0);
  int ss_idx = lua_gettop(L);
  tk_dvec_t *dst_mean = tk_dvec_create(L, n_dims, 0, 0);
  int dm_idx = lua_gettop(L);
  tk_dvec_t *dst_std = tk_dvec_create(L, n_dims, 0, 0);
  int ds_idx = lua_gettop(L);

  tk_normalizer_compute_stats(source->a, n_src, n_dims, src_mean->a, src_std->a);
  tk_normalizer_compute_stats(target->a, n_dst, n_dims, dst_mean->a, dst_std->a);

  tk_normalizer_t *n = (tk_normalizer_t *)tk_lua_newuserdata(L, tk_normalizer_t,
    TK_NORMALIZER_MT, tk_normalizer_mt_fns, tk_normalizer_gc);
  int enc_idx = lua_gettop(L);
  n->src_mean = src_mean;
  n->src_std = src_std;
  n->dst_mean = dst_mean;
  n->dst_std = dst_std;
  n->n_dims = n_dims;
  n->destroyed = false;

  tk_normalizer_init_fenv(L, enc_idx, sm_idx, ss_idx, dm_idx, ds_idx);

  lua_pushvalue(L, enc_idx);
  return 1;
}

static int tk_normalizer_load_lua (lua_State *L) {
  size_t len;
  const char *data = tk_lua_checklstring(L, 1, &len, "data");
  bool isstr = lua_type(L, 2) == LUA_TBOOLEAN && tk_lua_checkboolean(L, 2);
  FILE *fh = isstr
    ? tk_lua_fmemopen(L, (char *)data, len, "r")
    : tk_lua_fopen(L, data, "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKnm", 4) != 0) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "invalid normalizer file (bad magic)");
  }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 1) {
    tk_lua_fclose(L, fh);
    return luaL_error(L, "unsupported normalizer version %d", (int)version);
  }
  uint64_t n_dims;
  tk_lua_fread(L, &n_dims, sizeof(uint64_t), 1, fh);
  tk_dvec_t *src_mean = tk_dvec_load(L, fh);
  int sm_idx = lua_gettop(L);
  tk_dvec_t *src_std = tk_dvec_load(L, fh);
  int ss_idx = lua_gettop(L);
  tk_dvec_t *dst_mean = tk_dvec_load(L, fh);
  int dm_idx = lua_gettop(L);
  tk_dvec_t *dst_std = tk_dvec_load(L, fh);
  int ds_idx = lua_gettop(L);
  tk_lua_fclose(L, fh);

  tk_normalizer_t *n = (tk_normalizer_t *)tk_lua_newuserdata(L, tk_normalizer_t,
    TK_NORMALIZER_MT, tk_normalizer_mt_fns, tk_normalizer_gc);
  int enc_idx = lua_gettop(L);
  n->src_mean = src_mean;
  n->src_std = src_std;
  n->dst_mean = dst_mean;
  n->dst_std = dst_std;
  n->n_dims = n_dims;
  n->destroyed = false;

  tk_normalizer_init_fenv(L, enc_idx, sm_idx, ss_idx, dm_idx, ds_idx);

  lua_pushvalue(L, enc_idx);
  return 1;
}

static luaL_Reg tk_normalizer_fns[] = {
  { "create", tk_normalizer_create_lua },
  { "load", tk_normalizer_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_normalizer (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_normalizer_fns, 0);
  return 1;
}
