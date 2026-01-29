#include <santoku/tsetlin/itq.h>

#include <float.h>
#include <lauxlib.h>
#include <lua.h>
#include <lapacke.h>

static inline int tk_itq_sign_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "sign", "n_dims");
  const uint64_t N = codes->n / n_dims;
  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_cvec_zero(out);
  tk_itq_sign(out->a, codes->a, N, n_dims);
  return 1;
}

static inline int tk_itq_median_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "median", "n_dims");
  tk_dvec_t *medians = NULL;
  tk_itq_median(L, codes, n_dims, &medians);
  return 2;
}

static inline int tk_itq_encode_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "itq", "n_dims");
  uint64_t max_iterations = tk_lua_foptunsigned(L, 1, "itq", "iterations", 1000);
  double tolerance = tk_lua_foptposdouble(L, 1, "itq", "tolerance", 1e-8);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_dvec_t *means = NULL;
  tk_dvec_t *rotation = NULL;
  tk_itq_encode(L, codes, n_dims, max_iterations, tolerance, i_each, &means, &rotation);
  return 3;
}

static inline int tk_itq_apply_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  lua_getfield(L, 1, "means");
  tk_dvec_t *means = tk_dvec_peek(L, -1, "means");
  lua_getfield(L, 1, "rotation");
  tk_dvec_t *rotation = tk_dvec_peek(L, -1, "rotation");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "apply", "n_dims");

  const uint64_t K = n_dims;
  const uint64_t N = codes->n / K;

  double *X = tk_malloc(L, N * K * sizeof(double));
  memcpy(X, codes->a, N * K * sizeof(double));

  #pragma omp parallel for
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t d = 0; d < K; d++) {
      X[i * K + d] -= means->a[d];
    }
  }

  double *rotated = tk_malloc(L, N * K * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    N, K, K,
    1.0, X, K,
    rotation->a, K,
    0.0, rotated, K);

  tk_cvec_t *out = tk_cvec_create(L, N * TK_CVEC_BITS_BYTES(K), 0, 0);
  tk_cvec_zero(out);
  tk_itq_sign(out->a, rotated, N, K);

  free(X);
  free(rotated);
  return 1;
}

static inline int tk_itq_ica_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "ica", "n_dims");
  uint64_t max_iterations = tk_lua_foptunsigned(L, 1, "ica", "iterations", 200);
  double tolerance = tk_lua_foptposdouble(L, 1, "ica", "tolerance", 1e-4);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_dvec_t *means = NULL;
  tk_dvec_t *unmixing = NULL;
  tk_itq_ica(L, codes, n_dims, max_iterations, tolerance, i_each, &means, &unmixing);
  return 3;
}

static inline int tk_itq_cca_lua (lua_State *L)
{
  lua_settop(L, 1);
  lua_getfield(L, 1, "codes");
  tk_dvec_t *codes = tk_dvec_peek(L, -1, "codes");
  lua_getfield(L, 1, "tokens");
  tk_ivec_t *tokens = tk_ivec_peek(L, -1, "tokens");
  uint64_t n_samples = tk_lua_fcheckunsigned(L, 1, "cca", "n_samples");
  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "cca", "n_dims");
  uint64_t n_tokens = tk_lua_fcheckunsigned(L, 1, "cca", "n_tokens");
  uint64_t features_per_class = tk_lua_foptunsigned(L, 1, "cca", "features_per_class", 0);

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_dvec_t *means = NULL;
  tk_dvec_t *rotation = NULL;
  tk_ivec_t *feat_offsets = NULL;
  tk_ivec_t *feat_ids = NULL;
  tk_dvec_t *feat_weights = NULL;

  tk_itq_cca(L, codes, tokens, n_samples, n_dims, n_tokens, features_per_class,
      i_each, &means, &rotation, &feat_offsets, &feat_ids, &feat_weights);

  return 6;
}

static luaL_Reg tk_itq_fns[] =
{
  { "encode", tk_itq_encode_lua },
  { "itq", tk_itq_encode_lua },
  { "apply", tk_itq_apply_lua },
  { "sign", tk_itq_sign_lua },
  { "median", tk_itq_median_lua },
  { "ica", tk_itq_ica_lua },
  { "cca", tk_itq_cca_lua },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_itq (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tk_itq_fns, 0); // t
  return 1;
}
