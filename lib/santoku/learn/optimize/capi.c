#include <lua.h>
#include <lauxlib.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <santoku/learn/mathlibs.h>
#include <santoku/lua/utils.h>

#define TK_CMA_MT "tk_cma_t"

typedef struct {
  int n;
  int lambda;
  int mu;
  double *w;         // [mu]  recombination weights (normalized)
  double mu_eff;
  double c_sig, d_sig, c_c, c_1, c_mu, chiN;
  double sigma;
  double *m;         // [n]   mean
  double *C;         // [n*n] covariance (row-major)
  double *B;         // [n*n] eigenvectors (col j at B[i*n+j])
  double *ev;        // [n]   eigenvalues
  double *D;         // [n]   sqrt(max(ev, floor))
  double *p_sig;     // [n]
  double *p_c;       // [n]
  double *Y;         // [lambda*n] offspring y-vectors, row k
  int    *idx;       // [lambda]   sort permutation
  double *fs;        // [lambda]   f-values from tell
  double *viols;     // [lambda]
  int    *feas;      // [lambda]
  double *z, *dz, *y_w, *bt, *cinv; // [n] scratch
  double *best_hist; // [hist_len] ring
  int hist_len;
  int hist_count;
  int hist_head;
  int g;
  int safety_cap;
  bool destroyed;
} tk_cma_t;

// Symmetric eigendecomposition of an n*n row-major symmetric matrix A.
// On return A holds the orthonormal eigenvectors (component i of eigenvector j
// at A[i*n+j]) and w[j] the corresponding eigenvalue. Native path uses LAPACK;
// the emscripten build has no LAPACK, so it falls back to a cyclic Jacobi sweep.
static inline int tk_cma_eigh (double *A, int n, double *w)
{
#if !defined(__EMSCRIPTEN__)
  return LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, n, w);
#else
  double *a = (double *) malloc((size_t) n * (size_t) n * sizeof(double));
  if (!a) return -1;
  memcpy(a, A, (size_t) n * (size_t) n * sizeof(double));
  double *V = A;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      V[i * n + j] = (i == j) ? 1.0 : 0.0;
  double frob = 0.0;
  for (int i = 0; i < n * n; i++) frob += a[i] * a[i];
  frob = sqrt(frob);
  double tol = 1e-12 * frob;
  if (tol <= 0.0) tol = 1e-300;
  for (int sweep = 0; sweep < 100; sweep++) {
    double off = 0.0;
    for (int p = 0; p < n - 1; p++)
      for (int q = p + 1; q < n; q++)
        off += a[p * n + q] * a[p * n + q];
    if (sqrt(2.0 * off) < tol) break;
    for (int p = 0; p < n - 1; p++) {
      for (int q = p + 1; q < n; q++) {
        double apq = a[p * n + q];
        if (apq == 0.0) continue;
        double app = a[p * n + p];
        double aqq = a[q * n + q];
        double theta = (aqq - app) / (2.0 * apq);
        double t;
        if (theta >= 0.0)
          t = 1.0 / (theta + sqrt(theta * theta + 1.0));
        else
          t = -1.0 / (-theta + sqrt(theta * theta + 1.0));
        double c = 1.0 / sqrt(t * t + 1.0);
        double s = t * c;
        for (int k = 0; k < n; k++) {
          double akp = a[k * n + p];
          double akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; k++) {
          double apk = a[p * n + k];
          double aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; k++) {
          double vkp = V[k * n + p];
          double vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  for (int j = 0; j < n; j++) w[j] = a[j * n + j];
  free(a);
  return 0;
#endif
}

static inline tk_cma_t *tk_cma_peek (lua_State *L, int i)
{
  return (tk_cma_t *) luaL_checkudata(L, i, TK_CMA_MT);
}

static inline int tk_cma_gc (lua_State *L)
{
  tk_cma_t *c = tk_cma_peek(L, 1);
  if (c->destroyed) return 0;
  free(c->w); free(c->m); free(c->C); free(c->B); free(c->ev); free(c->D);
  free(c->p_sig); free(c->p_c); free(c->Y); free(c->idx);
  free(c->fs); free(c->viols); free(c->feas);
  free(c->z); free(c->dz); free(c->y_w); free(c->bt); free(c->cinv);
  free(c->best_hist);
  memset(c, 0, sizeof(*c));
  c->destroyed = true;
  return 0;
}

static inline int tk_cma_ask_lua (lua_State *L);
static inline int tk_cma_tell_lua (lua_State *L);

static luaL_Reg tk_cma_mt_fns[] = {
  { "ask", tk_cma_ask_lua },
  { "tell", tk_cma_tell_lua },
  { NULL, NULL }
};

// capi.cma(n, lambda, sigma, mean) -> cma userdata.
static inline int tk_cma_create_lua (lua_State *L)
{
  int n = (int) luaL_checkinteger(L, 1);
  int lambda = (int) luaL_checkinteger(L, 2);
  double sigma = luaL_checknumber(L, 3);
  luaL_checktype(L, 4, LUA_TTABLE);
  if (n < 1 || lambda < 2)
    return luaL_error(L, "cma: require n>=1 and lambda>=2");

  int mu = lambda / 2;
  if (mu < 1) mu = 1;

  tk_cma_t *c = tk_lua_newuserdata(L, tk_cma_t, TK_CMA_MT, tk_cma_mt_fns, tk_cma_gc);
  c->n = n; c->lambda = lambda; c->mu = mu; c->sigma = sigma;
  c->w = (double *) malloc((size_t) mu * sizeof(double));
  c->m = (double *) malloc((size_t) n * sizeof(double));
  c->C = (double *) malloc((size_t) n * (size_t) n * sizeof(double));
  c->B = (double *) malloc((size_t) n * (size_t) n * sizeof(double));
  c->ev = (double *) malloc((size_t) n * sizeof(double));
  c->D = (double *) malloc((size_t) n * sizeof(double));
  c->p_sig = (double *) malloc((size_t) n * sizeof(double));
  c->p_c = (double *) malloc((size_t) n * sizeof(double));
  c->Y = (double *) malloc((size_t) lambda * (size_t) n * sizeof(double));
  c->idx = (int *) malloc((size_t) lambda * sizeof(int));
  c->fs = (double *) malloc((size_t) lambda * sizeof(double));
  c->viols = (double *) malloc((size_t) lambda * sizeof(double));
  c->feas = (int *) malloc((size_t) lambda * sizeof(int));
  c->z = (double *) malloc((size_t) n * sizeof(double));
  c->dz = (double *) malloc((size_t) n * sizeof(double));
  c->y_w = (double *) malloc((size_t) n * sizeof(double));
  c->bt = (double *) malloc((size_t) n * sizeof(double));
  c->cinv = (double *) malloc((size_t) n * sizeof(double));

  // recombination weights (positive, sum to 1)
  double wsum = 0.0;
  double lhalf = log((double) lambda / 2.0 + 0.5);
  for (int i = 0; i < mu; i++) {
    double wi = lhalf - log((double) (i + 1));
    c->w[i] = wi;
    wsum += wi;
  }
  double w2 = 0.0;
  for (int i = 0; i < mu; i++) { c->w[i] /= wsum; w2 += c->w[i] * c->w[i]; }
  c->mu_eff = 1.0 / w2;

  double dn = (double) n, me = c->mu_eff;
  c->c_sig = (me + 2.0) / (dn + me + 5.0);
  c->d_sig = 1.0 + 2.0 * fmax(0.0, sqrt((me - 1.0) / (dn + 1.0)) - 1.0) + c->c_sig;
  c->c_c = (4.0 + me / dn) / (dn + 4.0 + 2.0 * me / dn);
  c->c_1 = 2.0 / ((dn + 1.3) * (dn + 1.3) + me);
  c->c_mu = fmin(1.0 - c->c_1, 2.0 * (me - 2.0 + 1.0 / me) / ((dn + 2.0) * (dn + 2.0) + me));
  c->chiN = sqrt(dn) * (1.0 - 1.0 / (4.0 * dn) + 1.0 / (21.0 * dn * dn));

  // mean
  for (int i = 0; i < n; i++) {
    lua_rawgeti(L, 4, i + 1);
    c->m[i] = lua_tonumber(L, -1);
    lua_pop(L, 1);
  }
  // C = I, paths = 0
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) c->C[i * n + j] = (i == j) ? 1.0 : 0.0;
    c->p_sig[i] = 0.0;
    c->p_c[i] = 0.0;
  }

  c->hist_len = 10 + (int) floor(30.0 * dn / (double) lambda);
  if (c->hist_len < 10) c->hist_len = 10;
  c->best_hist = (double *) malloc((size_t) c->hist_len * sizeof(double));
  c->hist_count = 0;
  c->hist_head = 0;
  c->g = 0;
  c->safety_cap = 100 + (int) floor(50.0 * dn * dn / (double) lambda);
  c->destroyed = false;

  lua_pushvalue(L, -1);
  return 1;
}

// cma:ask() -> { {x_1..x_n}, ... } (lambda candidate points in normalized space).
static inline int tk_cma_ask_lua (lua_State *L)
{
  tk_cma_t *c = tk_cma_peek(L, 1);
  int n = c->n, lambda = c->lambda;

  // symmetrize C into B, then eigendecompose (B <- eigenvectors, ev <- eigenvalues)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double avg = 0.5 * (c->C[i * n + j] + c->C[j * n + i]);
      c->B[i * n + j] = avg;
    }
  }
  if (tk_cma_eigh(c->B, n, c->ev) != 0)
    return luaL_error(L, "cma: eigendecomposition failed");
  for (int j = 0; j < n; j++) {
    double e = c->ev[j];
    if (e < 1e-30) e = 1e-30;
    c->D[j] = sqrt(e);
  }

  lua_createtable(L, lambda, 0);
  for (int k = 0; k < lambda; k++) {
    for (int j = 0; j < n; j++) {
      c->z[j] = tk_fast_normal(0.0, 1.0);
      c->dz[j] = c->D[j] * c->z[j];
    }
    lua_createtable(L, n, 0);
    for (int i = 0; i < n; i++) {
      double acc = 0.0;
      for (int j = 0; j < n; j++) acc += c->B[i * n + j] * c->dz[j];
      c->Y[k * n + i] = acc;                 // y
      lua_pushnumber(L, c->m[i] + c->sigma * acc); // x = m + sigma*y
      lua_rawseti(L, -2, i + 1);
    }
    lua_rawseti(L, -2, k + 1);
  }
  return 1;
}

static inline int tk_cma_less (tk_cma_t *c, int a, int b)
{
  int fa = c->feas[a], fb = c->feas[b];
  if (fa != fb) return fa;                    // feasible sorts first
  if (fa) return c->fs[a] < c->fs[b];
  if (c->viols[a] != c->viols[b]) return c->viols[a] < c->viols[b];
  return c->fs[a] < c->fs[b];
}

// cma:tell(fs, feas, viols) -> stop(boolean). fs/feas/viols are 1-based arrays of
// length lambda, matching the candidate order returned by the preceding ask().
static inline int tk_cma_tell_lua (lua_State *L)
{
  tk_cma_t *c = tk_cma_peek(L, 1);
  int n = c->n, lambda = c->lambda, mu = c->mu;
  luaL_checktype(L, 2, LUA_TTABLE);
  luaL_checktype(L, 3, LUA_TTABLE);
  luaL_checktype(L, 4, LUA_TTABLE);
  for (int k = 0; k < lambda; k++) {
    lua_rawgeti(L, 2, k + 1); c->fs[k] = lua_tonumber(L, -1); lua_pop(L, 1);
    lua_rawgeti(L, 3, k + 1); c->feas[k] = lua_toboolean(L, -1); lua_pop(L, 1);
    lua_rawgeti(L, 4, k + 1); c->viols[k] = lua_tonumber(L, -1); lua_pop(L, 1);
  }

  // feasible-first rank sort (insertion sort; lambda is small)
  for (int k = 0; k < lambda; k++) c->idx[k] = k;
  for (int i = 1; i < lambda; i++) {
    int key = c->idx[i], j = i - 1;
    while (j >= 0 && tk_cma_less(c, key, c->idx[j])) { c->idx[j + 1] = c->idx[j]; j--; }
    c->idx[j + 1] = key;
  }

  // per-gen best f into the ring history
  double gen_best_f = c->fs[c->idx[0]];
  if (c->hist_count < c->hist_len) {
    c->best_hist[(c->hist_head + c->hist_count) % c->hist_len] = gen_best_f;
    c->hist_count++;
  } else {
    c->best_hist[c->hist_head] = gen_best_f;
    c->hist_head = (c->hist_head + 1) % c->hist_len;
  }

  // y_w = sum_r w[r] * y_{idx[r]}
  for (int i = 0; i < n; i++) c->y_w[i] = 0.0;
  for (int r = 0; r < mu; r++) {
    double *yk = c->Y + c->idx[r] * n;
    double wr = c->w[r];
    for (int i = 0; i < n; i++) c->y_w[i] += wr * yk[i];
  }

  // p_sig update: C^{-1/2}(y_w) = B ((1/D) .* (B^T y_w))
  for (int j = 0; j < n; j++) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += c->B[i * n + j] * c->y_w[i];
    c->bt[j] = acc / c->D[j];
  }
  for (int i = 0; i < n; i++) {
    double acc = 0.0;
    for (int j = 0; j < n; j++) acc += c->B[i * n + j] * c->bt[j];
    c->cinv[i] = acc;
  }
  double ps_coef = sqrt(c->c_sig * (2.0 - c->c_sig) * c->mu_eff);
  for (int i = 0; i < n; i++)
    c->p_sig[i] = (1.0 - c->c_sig) * c->p_sig[i] + ps_coef * c->cinv[i];

  double ps_norm = 0.0;
  for (int i = 0; i < n; i++) ps_norm += c->p_sig[i] * c->p_sig[i];
  ps_norm = sqrt(ps_norm);

  double denom = sqrt(1.0 - pow(1.0 - c->c_sig, 2.0 * (c->g + 1)));
  int h_sig = ((ps_norm / denom) < (1.4 + 2.0 / (n + 1)) * c->chiN) ? 1 : 0;

  double pc_coef = h_sig * sqrt(c->c_c * (2.0 - c->c_c) * c->mu_eff);
  for (int i = 0; i < n; i++)
    c->p_c[i] = (1.0 - c->c_c) * c->p_c[i] + pc_coef * c->y_w[i];

  double delta = (1 - h_sig) * c->c_c * (2.0 - c->c_c);
  double c_scale = 1.0 - c->c_1 - c->c_mu;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double rank1 = c->p_c[i] * c->p_c[j] + delta * c->C[i * n + j];
      double rankmu = 0.0;
      for (int r = 0; r < mu; r++) {
        double *yk = c->Y + c->idx[r] * n;
        rankmu += c->w[r] * yk[i] * yk[j];
      }
      c->C[i * n + j] = c_scale * c->C[i * n + j] + c->c_1 * rank1 + c->c_mu * rankmu;
    }
  }

  double sigma_pre = c->sigma;
  c->sigma = c->sigma * exp((c->c_sig / c->d_sig) * (ps_norm / c->chiN - 1.0));
  for (int i = 0; i < n; i++) c->m[i] += sigma_pre * c->y_w[i];
  c->g += 1;

  // stopping criteria (budget/eval-count checks stay with the orchestrator)
  int stop = 0;
  double maxD = c->D[0];
  for (int j = 1; j < n; j++) if (c->D[j] > maxD) maxD = c->D[j];
  if (c->sigma * maxD < 1e-11) stop = 1;
  double maxD2 = c->D[0] * c->D[0], minD2 = c->D[0] * c->D[0];
  for (int j = 1; j < n; j++) {
    double d2 = c->D[j] * c->D[j];
    if (d2 > maxD2) maxD2 = d2;
    if (d2 < minD2) minD2 = d2;
  }
  if (minD2 > 0.0 && (maxD2 / minD2) > 1e14) stop = 1;
  if (c->hist_count >= c->hist_len) {
    double lo = c->best_hist[0], hi = c->best_hist[0];
    for (int t = 1; t < c->hist_len; t++) {
      double v = c->best_hist[t];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if ((hi - lo) < 1e-12) stop = 1;
  }
  if (c->g >= c->safety_cap) stop = 1;

  lua_pushboolean(L, stop);
  return 1;
}

static inline int tk_cma_seed_lua (lua_State *L)
{
  tk_fast_seed((uint64_t) luaL_checknumber(L, 1));
  return 0;
}

static inline int tk_cma_random_lua (lua_State *L)
{
  lua_pushinteger(L, (lua_Integer) tk_fast_random());
  return 1;
}

static inline int tk_cma_uniform_lua (lua_State *L)
{
  lua_pushnumber(L, (double) tk_fast_random() / ((double) UINT32_MAX + 1.0));
  return 1;
}

static inline int tk_cma_normal_lua (lua_State *L)
{
  double mean = luaL_checknumber(L, 1);
  double variance = luaL_checknumber(L, 2);
  lua_pushnumber(L, tk_fast_normal(mean, variance));
  return 1;
}

static luaL_Reg tk_cma_fns[] = {
  { "cma", tk_cma_create_lua },
  { "seed", tk_cma_seed_lua },
  { "random", tk_cma_random_lua },
  { "uniform", tk_cma_uniform_lua },
  { "normal", tk_cma_normal_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_optimize_capi (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tk_cma_fns, 0);
  lua_pushinteger(L, (lua_Integer) UINT32_MAX);
  lua_setfield(L, -2, "fast_max");
  return 1;
}
