#include "lua.h"
#include "lauxlib.h"
#include "lbfgs.h"
#include "../conf.h"
#include <arpack/arpack.h>
#include "khash.h"
#include "kvec.h"

static inline uint64_t mix64 (uint64_t x, uint64_t y) {
  uint64_t z = x ^ (y + 0x9e3779b97f4a7c15ULL);
  z ^= z >> 33;
  z *= 0xff51afd7ed558ccdULL;
  z ^= z >> 33;
  z *= 0xc4ceb9fe1a85ec53ULL;
  z ^= z >> 33;
  return z;
}

typedef struct { int64_t u, v; bool l; } tk_edge_t;
#define tk_edge_hash(e) (mix64(kh_int64_hash_func((e).u), kh_int64_hash_func((e).v)))
#define tk_edge_equal(a, b) ((a).u == (b).u && (a).v == (b).v)
KHASH_INIT(edges, tk_edge_t, char, 0, tk_edge_hash, tk_edge_equal);
KHASH_INIT(nodes, int64_t, char, 0, kh_int64_hash_func, kh_int64_hash_equal);
typedef khash_t(edges) tk_edges_t;
typedef khash_t(nodes) tk_nodes_t;
typedef kvec_t(int64_t) tk_neighbors_t;

static inline void tk_lua_callmod (
  lua_State *L,
  int nargs,
  int nret,
  const char *smod,
  const char *sfn
) {
  lua_getglobal(L, "require"); // arg req
  lua_pushstring(L, smod); // arg req smod
  lua_call(L, 1, 1); // arg mod
  lua_pushstring(L, sfn); // args mod sfn
  lua_gettable(L, -2); // args mod fn
  lua_remove(L, -2); // args fn
  lua_insert(L, - nargs - 1); // fn args
  lua_call(L, nargs, nret); // results
}

// TODO: include the field name in error
static inline void *tk_lua_checkuserdata (lua_State *L, int i, char *mt)
{
  if (mt == NULL && (lua_islightuserdata(L, i) || lua_isuserdata(L, i)))
    return lua_touserdata(L, i);
  void *p = luaL_checkudata(L, -1, mt);
  lua_pop(L, 1);
  return p;
}

static inline int tk_lua_absindex (lua_State *L, int i)
{
  if (i < 0 && i > LUA_REGISTRYINDEX)
    i += lua_gettop(L) + 1;
  return i;
}

static inline int tk_error (
  lua_State *L,
  const char *label,
  int err
) {
  lua_pushstring(L, label);
  lua_pushstring(L, strerror(err));
  tk_lua_callmod(L, 2, 0, "santoku.error", "error");
  return 1;
}

static inline void *tk_realloc (
  lua_State *L,
  void *p,
  size_t s
) {
  p = realloc(p, s);
  if (!p) {
    tk_error(L, "realloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline void *tk_malloc (
  lua_State *L,
  size_t s
) {
  void *p = malloc(s);
  if (!p) {
    tk_error(L, "malloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline int tk_lua_verror (lua_State *L, int n, ...) {
  va_list args;
  va_start(args, n);
  for (int i = 0; i < n; i ++) {
    const char *str = va_arg(args, const char *);
    lua_pushstring(L, str);
  }
  va_end(args);
  tk_lua_callmod(L, n, 0, "santoku.error", "error");
  return 0;
}

static inline const char *tk_lua_checkstring (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checkstring(L, i);
}

static inline const char *tk_lua_checklstring (lua_State *L, int i, size_t *lenp, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checklstring(L, i, lenp);
}

static inline const char *tk_lua_fchecklstring (lua_State *L, int i, size_t *lp, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, field, "field is not a string");
  const char *s = luaL_checklstring(L, -1, lp);
  lua_pop(L, 1);
  return s;
}

// TODO: include the field name in error
static inline lua_Integer tk_lua_ftype (lua_State *L, int i, char *field)
{
  lua_getfield(L, i, field);
  int t = lua_type(L, -1);
  lua_pop(L, 1);
  return t;
}

static inline uint64_t tk_lua_checkunsigned (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  return (uint64_t) l;
}

static inline uint64_t tk_lua_fcheckunsigned (lua_State *L, int i, char *name, char *field)
{
  lua_getfield(L, i, field);
  if (lua_type(L, -1) != LUA_TNUMBER)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, -1);
  if (l < 0)
    tk_lua_verror(L, 3, name, field, "field is not a positive integer");
  lua_pop(L, 1);
  return (uint64_t) l;
}

static inline void tk_lua_register (lua_State *L, luaL_Reg *regs, int nup)
{
  while (true) {
    if ((*regs).name == NULL)
      break;
    for (int i = 0; i < nup; i ++)
      lua_pushvalue(L, -nup); // t upsa upsb
    lua_pushcclosure(L, (*regs).func, nup); // t upsa fn
    lua_setfield(L, -nup - 2, (*regs).name); // t
    regs ++;
  }
  lua_pop(L, nup);
}

static inline int tm_class_accuracy (lua_State *L)
{
  size_t predicted_len, expected_len;
  unsigned int *predicted = (unsigned int *) tk_lua_checklstring(L, 1, &predicted_len, "predicted");
  unsigned int *expected = (unsigned int *) tk_lua_checklstring(L, 2, &expected_len, "expected");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_classes = tk_lua_checkunsigned(L, 3, "n_classes");

  if (predicted_len < n_samples * sizeof(unsigned int))
    tk_lua_verror(L, 4, "class_accuracy", "1", "predicted", "invalid data length");

  if (expected_len != predicted_len)
    tk_lua_verror(L, 4, "class_accuracy", "2", "expected", "invalid data length");

  if (n_classes)
    tk_lua_verror(L, 4, "class_accuracy", "4", "n_classes", "number of classes must be greater than 0");

  double f1[n_classes];
  double precision[n_classes];
  double recall[n_classes];

  double f1_avg;
  double precision_avg;
  double recall_avg;

  // TODO: calculate per-class and overall precision, recall, and f1

  lua_newtable(L);

  lua_newtable(L);
  for (unsigned int i = 0; i < n_classes; i ++) {
    lua_pushinteger(L, i + 1);
    lua_pushnumber(L, f1[i]);
    lua_setfield(L, -3, "f1");
    lua_pushinteger(L, i + 1);
    lua_pushnumber(L, precision[i]);
    lua_setfield(L, -4, "precision");
    lua_pushinteger(L, i + 1);
    lua_pushnumber(L, recall[i]);
    lua_setfield(L, -4, "recall");
  }
  lua_setfield(L, -2, "classes");

  lua_pushnumber(L, precision_avg);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, recall_avg);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, f1_avg);
  lua_setfield(L, -2, "f1");

  return 1;
}

static inline int tm_encoding_accuracy (lua_State *L)
{
  size_t predicted_len, expected_len;
  tk_bits_t *codes_predicted = (tk_bits_t *) tk_lua_checklstring(L, 1, &predicted_len, "predicted");
  tk_bits_t *codes_expected = (tk_bits_t *) tk_lua_checklstring(L, 2, &expected_len, "expected");
  uint64_t n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  uint64_t n_features = tk_lua_checkunsigned(L, 4, "n_features");

  if (predicted_len < n_samples * BITS_DIV(n_features) * sizeof(tk_bits_t))
    tk_lua_verror(L, 4, "encoding_accuracy", "1", "predicted", "invalid data length");

  if (expected_len < n_samples * BITS_DIV(n_features) * sizeof(tk_bits_t))
    tk_lua_verror(L, 4, "encoding_accuracy", "2", "expected", "invalid data length");

  double f1[n_features];
  double precision[n_features];
  double recall[n_features];

  double f1_avg;
  double precision_avg;
  double recall_avg;

  // TODO: calculate per-bit and overall precision, recall, and f1

  lua_newtable(L);

  lua_newtable(L);
  for (unsigned int i = 0; i < n_features; i ++) {
    lua_pushinteger(L, i + 1);
    lua_pushnumber(L, f1[i]);
    lua_setfield(L, -3, "f1");
    lua_pushinteger(L, i + 1);
    lua_pushnumber(L, precision[i]);
    lua_setfield(L, -4, "precision");
    lua_pushinteger(L, i + 1);
    lua_pushnumber(L, recall[i]);
    lua_setfield(L, -4, "recall");
  }
  lua_setfield(L, -2, "classes");

  lua_pushnumber(L, precision_avg);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, recall_avg);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, f1_avg);
  lua_setfield(L, -2, "f1");

  return 1;
}

// Context shared by evaluate & progress callbacks
typedef struct {
  unsigned int *pairs;
  tk_bits_t *labels;
  unsigned int n_pairs, n_codes, n_features;
  lua_State *L;
  int i_each;
  unsigned int *global_iter;
  tk_bits_t *codes;
} eval_ctx_t;

// libLBFGS evaluate callback: compute KSH loss & gradient over x[]
static double lbfgs_evaluate (
  void *instance,
  const double*x,
  double *g,
  const int n,
  const double step
) {
  eval_ctx_t *ctx = (eval_ctx_t*)instance;
  const unsigned int n_pairs = ctx->n_pairs;
  const unsigned int n_codes = ctx->n_codes;
  const unsigned int n_features = ctx->n_features;
  const unsigned int *pairs = ctx->pairs;
  const tk_bits_t *labels = ctx->labels;
  // initialize
  double loss = 0.0;
  for(int idx = 0; idx < n; idx ++)
    g[idx] = 0.0;
  // for each (i,j) pair
  for(unsigned int k = 0; k < n_pairs; k ++) {
    unsigned int i = pairs[2 * k + 0];
    unsigned int j = pairs[2 * k + 1];
    // label S_ij = +1 if similar bit is set, else -1
    int S = (labels[BITS_DIV(k)] & ((tk_bits_t)1 << BITS_MOD(k))) ? +1 : -1;
    // compute dot = x_i · x_j
    double dot = 0.0;
    double *xi = (double*)(x + i * n_features);
    double *xj = (double*)(x + j * n_features);
    for(unsigned int f = 0; f < n_features; f ++)
      dot += xi[f] * xj[f];
    // KSH target inner product = b * S, where b = n_features
    double target = (double)n_features * (double)S;
    double diff   = dot - target;
    // accumulate squared error
    loss += diff * diff;
    // gradient w.r.t x_i and x_j
    double c = 2.0 * diff;
    for(unsigned int f = 0; f < n_features; f ++) {
      double xv = xi[f];
      double yv = xj[f];
      g[i*n_features + f] += c * yv;
      g[j*n_features + f] += c * xv;
    }
  }
  return loss;
}

// libLBFGS progress callback: threshold x→codes, invoke Lua each()
// in your code, replacing the old progress callback:
static int lbfgs_progress (
  void *instance, // user data
  const double *x, // current point
  const double *g, // current gradient
  double fx, // current function value
  double xnorm, // ||x||
  double gnorm, // ||g||
  double step, // step size used
  int n, // number of variables
  int k, // iteration count
  int ls // line-search iterations
) {
  eval_ctx_t *ctx = (eval_ctx_t*)instance;
  double *z = (double *)x;
  for (unsigned int i = 0; i < ctx->n_codes; i++) {
    double norm = 0.0;
    for (unsigned int f = 0; f < ctx->n_features; f++)
      norm += z[i*ctx->n_features + f] * z[i*ctx->n_features + f];
    norm = sqrt(norm) + 1e-8;
    for (unsigned int f = 0; f < ctx->n_features; f++)
      z[i*ctx->n_features + f] /= norm;
  }
  // Re-threshold x → codes, invoke Lua callback as before…
  (*ctx->global_iter) ++;
  if (ctx->i_each != -1) {
    lua_pushvalue(ctx->L, ctx->i_each);
    lua_pushinteger(ctx->L, *ctx->global_iter);
    lua_pushnumber(ctx->L, fx);
    lua_pushstring(ctx->L, "lbfgs");
    lua_call(ctx->L, 3, 1);
    bool stop =
      lua_type(ctx->L, -1) == LUA_TBOOLEAN &&
      lua_toboolean(ctx->L, -1) == 0;
    lua_pop(ctx->L, 1);
    if (stop)
      return 1;
  }
  return 0;
}

static inline int tm_codeify (lua_State *L)
{
  size_t pairs_len, labels_len;
  unsigned int *pairs = (unsigned int *) tk_lua_fchecklstring(L, 1, &pairs_len, "codeify", "pairs");
  tk_bits_t *labels = (tk_bits_t *) tk_lua_fchecklstring(L, 1, &labels_len, "codeify", "labels");
  uint64_t n_pairs = tk_lua_fcheckunsigned(L, 1, "codeify", "n_pairs");
  uint64_t n_features = tk_lua_fcheckunsigned(L, 1, "codeify", "n_features");
  uint64_t n_codes = tk_lua_fcheckunsigned(L, 1, "codeify", "n_codes");
  uint64_t spectral_iterations = tk_lua_fcheckunsigned(L, 1, "codeify", "spectral_iterations");
  uint64_t lbfgs_iterations = tk_lua_fcheckunsigned(L, 1, "codeify", "lbfgs_iterations");

  if (pairs_len != n_pairs * 2 * sizeof(unsigned int))
    tk_lua_verror(L, 3, "codify", "pairs", "data length too short");
  if (labels_len < BYTES_DIV(n_pairs))
    tk_lua_verror(L, 3, "codify", "labels", "data length too short");
  if (BITS_MOD(labels_len * CHAR_BIT) != 0)
    tk_lua_verror(L, 3, "codify", "labels", "must be a multiple of " STR(BITS));
  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  tk_bits_t *codes = tk_malloc(L, (size_t) n_codes * BITS_DIV(n_features) * sizeof(tk_bits_t));

  unsigned int global_iter = 0;
  // Build degree vector for Laplacian
  unsigned int *degree = tk_malloc(L, n_codes * sizeof(unsigned int));
  memset(degree, 0, (size_t) n_codes * sizeof(unsigned int));
  for (unsigned int k = 0; k < n_pairs; k ++) {
    if (labels[BITS_DIV(k)] & ((tk_bits_t)1 << BITS_MOD(k))) {
      unsigned int i = pairs[2 * k], j = pairs[2 * k + 1];
      degree[i] ++; degree[j] ++;
    }
  }
  // ARPACK setup
  a_int ido = 0, info = 0;
  char  bmat[] = "I", which[] = "SM";
  a_int n = (a_int)n_codes;
  a_int nev = (a_int)n_features;
  a_int ncv = 4 * nev + 1;
  double tol = 1e-6;
  double *resid = tk_malloc(L, (size_t) n * sizeof(double));
  double *workd = tk_malloc(L, 3 * (size_t) n * sizeof(double));
  a_int iparam[11] = {1,0,spectral_iterations,0,0,0,1,0,0,0,0};
  a_int ipntr[14] = {0};
  int lworkl = ncv*(ncv + 8);
  double *workl = tk_malloc(L, (size_t) lworkl * sizeof(double));
  double *v = tk_malloc(L, (size_t) n * (size_t) ncv * sizeof(double));
  // Reverse-communication to build Lanczos basis v
  do {
    dsaupd_c(
      &ido,
      bmat,
      n,
      which,
      nev,
      tol,
      resid,
      ncv,
      v,
      n,
      iparam,
      ipntr,
      workd,
      workl,
      (a_int) lworkl,
      &info);
    if (ido == -1 || ido == 1) {
      double *in  = workd + ipntr[0] - 1;
      double *out = workd + ipntr[1] - 1;
      memset(out, 0, (size_t) n * sizeof(double));
      for (unsigned int e = 0; e < n_pairs; e ++) {
        if (labels[BITS_DIV(e)] & ((tk_bits_t)1 << BITS_MOD(e))) {
          unsigned int i = pairs[2 * e], j = pairs[2 * e + 1];
          out[i] -= in[j];
          out[j] -= in[i];
        }
      }
      for (a_int i = 0; i < n; i ++)
        out[i] += degree[i] * in[i];
    }
  } while (ido == -1 || ido == 1);
  // Prepare containers
  a_int rvec = 1;
  char howmny[] = "A";
  a_int select[ncv];
  double d[nev];
  double *z = tk_malloc(L, (size_t) n * (size_t) nev * sizeof(double));  /* 5: eigenvectors */
  double sigma = 0.0;
  dseupd_c(
    rvec,
    howmny,
    select,
    d,
    z,
    (a_int) n, // ldz
    sigma,
    bmat,
    (a_int) n,
    which,
    (a_int) nev,
    tol,
    resid,
    (a_int) ncv,
    v,
    (a_int) n, // ldv
    iparam,
    ipntr,
    workd,
    workl,
    (a_int) lworkl,
    &info
  );
  global_iter ++;
  if (i_each != -1) {
    lua_pushvalue(L, i_each);
    lua_pushinteger(L, global_iter);
    lua_pushinteger(L, iparam[4]);
    lua_pushstring(L, "spectral");
    lua_call(L, 3, 0);
  }
  // Prepare context for LBFGS
  eval_ctx_t ctx = {
    pairs, labels, n_pairs, n_codes, n_features,
    L, i_each, &global_iter, codes
  };
  // Flatten codes → real-valued x
  int n_vars = n_codes * n_features;
  // LBFGS refine
  if (lbfgs_iterations > 0) {
    lbfgs_parameter_t param;
    lbfgs_parameter_init(&param);
    param.max_iterations = lbfgs_iterations;
    double fx = 0.0;
    lbfgs(n_vars, z, &fx,
          lbfgs_evaluate, lbfgs_progress,
          &ctx, &param);
  }
  memset(codes, 0, (size_t) n_codes * BITS_DIV(n_features) * sizeof(tk_bits_t));
  for (unsigned int i = 0; i < n_codes; i ++) {
    for (unsigned int f = 0; f < n_features; f ++) {
      if (z[i * n_features + f] > 0) {
        unsigned int chunk = f >> 6;
        unsigned int pos = f & (BITS - 1);
        codes[i * BITS_DIV(n_features) + chunk] |= (tk_bits_t)1 << pos;
      }
    }
  }
  // Cleanup helpers
  // Cleanup
  free(resid);
  free(workd);
  free(workl);
  free(v);
  free(z);
  free(degree);

  lua_pushlstring(L, (char *) codes, n_codes * BITS_DIV(n_features) * sizeof(tk_bits_t));
  free(codes);
  return 1;
}

typedef struct {
  int64_t *parent;
  int64_t *rank;
} dsu_t;

static inline void dsu_init (dsu_t *dsu, int64_t n)
{
  dsu->parent = malloc((size_t) n * sizeof(int64_t));
  dsu->rank = malloc((size_t) n * sizeof(int64_t));
  for (int i = 0; i < n; i++) {
    dsu->parent[i] = i;
    dsu->rank[i] = 0;
  }
}

static inline int64_t dsu_find (dsu_t *dsu, int64_t x)
{
  if (dsu->parent[x] != x)
    dsu->parent[x] = dsu_find(dsu, dsu->parent[x]);
  return dsu->parent[x];
}

static inline void dsu_union (dsu_t *dsu, int64_t x, int64_t y)
{
  int64_t xr = dsu_find(dsu, x);
  int64_t yr = dsu_find(dsu, y);
  if (xr == yr) {
    return;
  } else if (dsu->rank[xr] < dsu->rank[yr]) {
    dsu->parent[xr] = yr;
  } else if (dsu->rank[xr] > dsu->rank[yr]) {
    dsu->parent[yr] = xr;
  } else {
    dsu->parent[yr] = xr;
    dsu->rank[xr]++;
  }
}

static inline int tm_components (lua_State *L)
{
  lua_settop(L, 2); // ps n
  int64_t n = (int64_t) tk_lua_checkunsigned(L, 2, "n_codes");
  lua_pop(L, 1); // ps
  tk_lua_callmod(L, 1, 3, "santoku.matrix.integer", "view"); // v r c
  int64_t *data = tk_lua_checkuserdata(L, 1, NULL);
  int64_t rows = luaL_checkinteger(L, 2);
  int64_t cols = luaL_checkinteger(L, 3);
  if (cols != 2)
    return luaL_error(L, "expected 2 columns");
  dsu_t dsu;
  dsu_init(&dsu, n);  // n = total number of distinct node IDs
  for (int64_t i = 0; i < rows; i++)
    dsu_union(&dsu, data[i*2], data[i*2+1]);
  tk_nodes_t *nodes = kh_init(nodes);
  int absent;
  for (int64_t i = 0; i < n; i++) {
    int64_t root = dsu_find(&dsu, i);
    kh_put(nodes, nodes, root, &absent);
  }
  lua_pushinteger(L, kh_size(nodes));
  free(dsu.parent);
  free(dsu.rank);
  kh_destroy(nodes, nodes);
  return 1;
}

static inline int tm_densify (lua_State *L)
{
  lua_settop(L, 3); // m l nts
  lua_pop(L, 1);
  lua_pushvalue(L, 1); // m l m
  tk_lua_callmod(L, 1, 3, "santoku.matrix.integer", "view"); // m l v r c
  int64_t *data = tk_lua_checkuserdata(L, 3, NULL);
  int64_t rows = luaL_checkinteger(L, 4);
  int64_t cols = luaL_checkinteger(L, 5);
  if (cols != 2)
    return luaL_error(L, "expected 2 columns");

  lua_pushvalue(L, 2); // m l v r c l
  lua_pushinteger(L, rows); // m l v r c l r
  lua_pushinteger(L, BITS);
  tk_lua_callmod(L, 2, 1, "santoku.num", "round"); // m l v r c raw
  tk_lua_callmod(L, 2, 1, "santoku.bitmap", "raw"); // m l v r c raw
  tk_bits_t *labels = (tk_bits_t *)luaL_checkstring(L, -1);
  lua_pop(L, 1);

  // 2) init the edge‐set and load raw pairs
  tk_edges_t *E = kh_init(edges);
  tk_edge_t e, e1, e2, rev;
  char _;
  int absent;
  for (int64_t i = 0; i < rows; ++i) {
    bool label = (labels[BITS_DIV(i)] & (1 << BITS_MOD(i))) > 0;
    e = (tk_edge_t) { .u = data[2 * i], .v = data[2 * i + 1], .l = label };
    kh_put(edges, E, e, &absent);
  }

  /* 3) enforce symmetry once (keep label) */
  kh_foreach(E, e, _, ({
    rev = (tk_edge_t){ .u = e.v, .v = e.u, .l = e.l };
    kh_put(edges, E, rev, &absent);
  }));

  /* 4) iterative closure to fixed‑point */
  bool changed;
  tk_edges_t *snapshot = kh_init(edges);
  tk_neighbors_t nbr_sim;
  kv_init(nbr_sim);   /* similar neighbours   */
  tk_neighbors_t nbr_dis;
  kv_init(nbr_dis);   /* dissimilar neighbours*/

  do {
    changed = false;
    kh_clear(edges, snapshot);
    kh_foreach(E, e, _, ({
      kh_put(edges, snapshot, e, &absent);
    }));
    // (i) clique and contradiction inference
    kh_foreach(snapshot, e1, _, ({
      int64_t u = e1.u;
      kv_size(nbr_sim) = 0;
      kv_size(nbr_dis) = 0;
      kh_foreach(snapshot, e2, _, ({
        if (e2.u != u) continue;
        if (e2.l) kv_push(int64_t, nbr_sim, e2.v);
        else kv_push(int64_t, nbr_dis, e2.v);
      }));
      // clique
      for (khint_t a = 0; a < kv_size(nbr_sim); ++a)
        for (khint_t b = a + 1; b < kv_size(nbr_sim); ++b) {
          e = (tk_edge_t){ .u = kv_A(nbr_sim,a), .v = kv_A(nbr_sim,b), .l = 1 };
          kh_put(edges, E, e, &absent); if (absent) changed = true;
          rev = (tk_edge_t){ .u = e.v, .v = e.u, .l = 1 };
          kh_put(edges, E, rev, &absent); if (absent) changed = true;
        }
      // contradiction
      for (khint_t a = 0; a < kv_size(nbr_sim); ++a)
        for (khint_t b = 0; b < kv_size(nbr_dis); ++b) {
          e = (tk_edge_t){ .u = kv_A(nbr_sim,a), .v = kv_A(nbr_dis,b), .l = 0 };
          kh_put(edges, E, e, &absent); if (absent) changed = true;
          rev = (tk_edge_t){ .u = e.v, .v = e.u, .l = 0 };
          kh_put(edges, E, rev, &absent); if (absent) changed = true;
        }
    }));
    // (ii) chaining inference
    kh_foreach(snapshot, e1, _, ({
      kh_foreach(snapshot, e2, _, ({
        if (e1.v != e2.u) continue;
        tk_edge_t add = {
          .u = e1.u,
          .v = e2.v,
          .l = (e1.l && e2.l) ? 1 : 0  // exhaustive: infer 0 if either is 0
        };
        kh_put(edges, E, add, &absent); if (absent) changed = true;
        rev = (tk_edge_t) { .u = add.v, .v = add.u, .l = add.l };
        kh_put(edges, E, rev, &absent); if (absent) changed = true;
      }));
    }));
  } while (changed);

  int64_t nout = 0;
  int64_t npair = 0;
  int64_t *out = malloc(kh_size(E) * 2 * sizeof(int64_t));
  kh_foreach(E, e, _, ({
    out[nout ++] = e.u;
    out[nout ++] = e.v;
    lua_pushvalue(L, 2);
    lua_pushinteger(L, ++ npair);
    if (e.l)
      tk_lua_callmod(L, 2, 0, "santoku.bitmap", "set");
    else
      tk_lua_callmod(L, 2, 0, "santoku.bitmap", "unset");
  }));
  int64_t outrows = nout / 2;
  int64_t outcolumns = 2;
  lua_pushlightuserdata(L, out);
  lua_pushinteger(L, outrows);
  lua_pushinteger(L, outcolumns);
  lua_pushvalue(L, 1);
  // NOTE: this takes ownership of malloc'd data. No need to free.
  tk_lua_callmod(L, 4, 1, "santoku.matrix.integer", "from_view");
  // TODO: free on error
  kh_destroy(edges, E);
  kh_destroy(edges, snapshot);
  kv_destroy(nbr_sim);
  kv_destroy(nbr_dis);
  return 1;
}

static luaL_Reg tm_codebook_fns[] =
{
  { "components", tm_components },
  { "codeify", tm_codeify },
  { "densify", tm_densify },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_codebook_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_codebook_fns, 0); // t
  return 1;
}
