#include <santoku/dvec.h>
#include <santoku/dvec/ext.h>
#include <santoku/lua/utils.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GP_SQRT5 2.2360679774997896

static uint64_t gp_rng;

static inline double gp_rand01 (void)
{
  gp_rng ^= gp_rng >> 12;
  gp_rng ^= gp_rng << 25;
  gp_rng ^= gp_rng >> 27;
  return (double)((gp_rng * 0x2545F4914F6CDD1DULL) >> 11) / (double)(1ULL << 53);
}

static inline double gp_matern52 (
  const double *x1, const double *x2, uint64_t d,
  const double *ls, double sf2)
{
  double r2 = 0.0;
  for (uint64_t i = 0; i < d; i++) {
    double diff = (x1[i] - x2[i]) / ls[i];
    r2 += diff * diff;
  }
  double r = sqrt(r2);
  double s5r = GP_SQRT5 * r;
  return sf2 * (1.0 + s5r + (5.0 / 3.0) * r2) * exp(-s5r);
}

static int gp_cholesky (double *A, uint64_t n)
{
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j <= i; j++) {
      double s = A[i * n + j];
      for (uint64_t k = 0; k < j; k++)
        s -= A[i * n + k] * A[j * n + k];
      if (i == j) {
        if (s <= 1e-10) return -1;
        A[i * n + i] = sqrt(s);
      } else {
        A[i * n + j] = s / A[j * n + j];
      }
    }
  }
  for (uint64_t i = 0; i < n; i++)
    for (uint64_t j = i + 1; j < n; j++)
      A[i * n + j] = 0.0;
  return 0;
}

static void gp_fwd (const double *L, const double *b, double *x, uint64_t n)
{
  for (uint64_t i = 0; i < n; i++) {
    double s = b[i];
    for (uint64_t j = 0; j < i; j++)
      s -= L[i * n + j] * x[j];
    x[i] = s / L[i * n + i];
  }
}

static void gp_bwd (const double *L, const double *b, double *x, uint64_t n)
{
  for (int64_t i = (int64_t)n - 1; i >= 0; i--) {
    double s = b[i];
    for (uint64_t j = (uint64_t)i + 1; j < n; j++)
      s -= L[j * n + (uint64_t)i] * x[j];
    x[(uint64_t)i] = s / L[(uint64_t)i * n + (uint64_t)i];
  }
}

static inline double gp_npdf (double x)
{
  return 0.3989422804014327 * exp(-0.5 * x * x);
}

static inline double gp_ncdf (double x)
{
  return 0.5 * erfc(-x * 0.7071067811865476);
}

static double gp_lml (
  const double *Y, const double *alpha, const double *L, uint64_t n)
{
  double v = 0.0;
  for (uint64_t i = 0; i < n; i++)
    v -= 0.5 * Y[i] * alpha[i];
  for (uint64_t i = 0; i < n; i++)
    v -= log(L[i * n + i]);
  v -= 0.5 * (double)n * 1.8378770664093453;
  return v;
}

static int gp_build (
  const double *X, const double *Y, uint64_t n, uint64_t d,
  const double *ls, double sf2, double sn2,
  double *K, double *alpha, double *tmp)
{
  for (uint64_t i = 0; i < n; i++) {
    K[i * n + i] = sf2 + sn2;
    for (uint64_t j = 0; j < i; j++) {
      double k = gp_matern52(X + i * d, X + j * d, d, ls, sf2);
      K[i * n + j] = k;
      K[j * n + i] = k;
    }
  }
  double jitter = 1e-8;
  for (int a = 0; a < 6; a++) {
    if (a > 0) {
      for (uint64_t i = 0; i < n; i++) {
        K[i * n + i] = sf2 + sn2 + jitter;
        for (uint64_t j = 0; j < i; j++) {
          double k = gp_matern52(X + i * d, X + j * d, d, ls, sf2);
          K[i * n + j] = k;
          K[j * n + i] = k;
        }
      }
    }
    if (gp_cholesky(K, n) == 0) {
      gp_fwd(K, Y, tmp, n);
      gp_bwd(K, tmp, alpha, n);
      return 0;
    }
    jitter *= 10.0;
  }
  return -1;
}

static int tm_gp_suggest (lua_State *L)
{
  tk_dvec_t *Xd = tk_dvec_peek(L, 1, "X");
  tk_dvec_t *Yd = tk_dvec_peek(L, 2, "Y");
  uint64_t d = tk_lua_checkunsigned(L, 3, "n_dims");
  tk_dvec_t *cd = tk_dvec_peek(L, 4, "candidates");
  uint64_t nc = tk_lua_checkunsigned(L, 5, "n_candidates");
  uint64_t nr = lua_gettop(L) >= 6 ? tk_lua_checkunsigned(L, 6, "n_restarts") : 10;

  uint64_t n = Yd->n;
  if (n == 0 || d == 0)
    return luaL_error(L, "need observations and dims");
  if (Xd->n != n * d)
    return luaL_error(L, "X size mismatch");
  if (cd->n != nc * d)
    return luaL_error(L, "candidates size mismatch");

  double *X = Xd->a;
  double *Yr = Yd->a;
  double *cand = cd->a;

  double ym = 0.0;
  for (uint64_t i = 0; i < n; i++) ym += Yr[i];
  ym /= (double)n;
  double yv = 0.0;
  for (uint64_t i = 0; i < n; i++) {
    double t = Yr[i] - ym;
    yv += t * t;
  }
  yv /= (double)n;
  double ysd = sqrt(yv);
  if (ysd < 1e-12) ysd = 1.0;

  double *Ys = malloc(n * sizeof(double));
  if (!Ys) return luaL_error(L, "alloc");
  for (uint64_t i = 0; i < n; i++)
    Ys[i] = (Yr[i] - ym) / ysd;

  double ybest = Ys[0];
  for (uint64_t i = 1; i < n; i++)
    if (Ys[i] > ybest) ybest = Ys[i];

  double *xsd = malloc(d * sizeof(double));
  for (uint64_t j = 0; j < d; j++) {
    double m = 0.0;
    for (uint64_t i = 0; i < n; i++) m += X[i * d + j];
    m /= (double)n;
    double v = 0.0;
    for (uint64_t i = 0; i < n; i++) {
      double t = X[i * d + j] - m;
      v += t * t;
    }
    xsd[j] = sqrt(v / (double)n);
    if (xsd[j] < 0.01) xsd[j] = 0.2;
  }

  double *K = malloc(n * n * sizeof(double));
  double *alpha = malloc(n * sizeof(double));
  double *tmp = malloc(n * sizeof(double));
  double *ls = malloc(d * sizeof(double));
  double *bls = malloc(d * sizeof(double));
  double bsf = 1.0, bsn = 0.01, blml = -1e30;

  gp_rng = 0;
  for (uint64_t i = 0; i < n; i++)
    gp_rng ^= (uint64_t)(Yr[i] * 1e8 + 0.5);
  gp_rng ^= n * 2654435761ULL;
  if (gp_rng == 0) gp_rng = 1;

  for (uint64_t j = 0; j < d; j++) bls[j] = xsd[j];

  for (uint64_t r = 0; r < nr; r++) {
    double sf2, sn2;
    if (r == 0) {
      for (uint64_t j = 0; j < d; j++) ls[j] = xsd[j];
      sf2 = 1.0;
      sn2 = 0.01;
    } else {
      for (uint64_t j = 0; j < d; j++) {
        ls[j] = exp(log(xsd[j]) + (gp_rand01() - 0.5) * 4.0);
        if (ls[j] < 0.01) ls[j] = 0.01;
        if (ls[j] > 10.0) ls[j] = 10.0;
      }
      sf2 = exp((gp_rand01() - 0.5) * 6.0);
      if (sf2 < 0.001) sf2 = 0.001;
      if (sf2 > 100.0) sf2 = 100.0;
      sn2 = exp(-1.0 - gp_rand01() * 8.0);
    }
    if (gp_build(X, Ys, n, d, ls, sf2, sn2, K, alpha, tmp) != 0)
      continue;
    double lml = gp_lml(Ys, alpha, K, n);
    if (lml > blml) {
      blml = lml;
      memcpy(bls, ls, d * sizeof(double));
      bsf = sf2;
      bsn = sn2;
    }
  }

  if (gp_build(X, Ys, n, d, bls, bsf, bsn, K, alpha, tmp) != 0) {
    for (uint64_t j = 0; j < d; j++) bls[j] = xsd[j];
    bsf = 1.0;
    bsn = 0.1;
    gp_build(X, Ys, n, d, bls, bsf, bsn, K, alpha, tmp);
  }

  tk_dvec_t *ei = tk_dvec_create(L, nc, 0, 0);
  ei->n = nc;

  double *v = malloc(n * sizeof(double));

  for (uint64_t c = 0; c < nc; c++) {
    const double *xc = cand + c * d;
    for (uint64_t i = 0; i < n; i++)
      tmp[i] = gp_matern52(xc, X + i * d, d, bls, bsf);
    double mu = 0.0;
    for (uint64_t i = 0; i < n; i++)
      mu += tmp[i] * alpha[i];
    gp_fwd(K, tmp, v, n);
    double vtv = 0.0;
    for (uint64_t i = 0; i < n; i++)
      vtv += v[i] * v[i];
    double s2 = bsf - vtv;
    if (s2 < 1e-10) s2 = 1e-10;
    double s = sqrt(s2);
    double z = (mu - ybest) / s;
    double val = (mu - ybest) * gp_ncdf(z) + s * gp_npdf(z);
    ei->a[c] = val > 0.0 ? val : 0.0;
  }

  free(v);
  free(K);
  free(alpha);
  free(tmp);
  free(ls);
  free(bls);
  free(Ys);
  free(xsd);

  return 1;
}

static luaL_Reg tm_gp_fns[] = {
  { "suggest", tm_gp_suggest },
  { NULL, NULL }
};

int luaopen_santoku_learn_gp (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_gp_fns, 0);
  return 1;
}
