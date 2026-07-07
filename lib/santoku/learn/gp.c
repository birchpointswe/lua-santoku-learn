#include <santoku/dvec.h>
#include <santoku/lua/utils.h>
#include <santoku/learn/mathlibs.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GP_SQRT5 2.2360679774997896
#define GP_LOG2PI 1.8378770664093453
#define GP_HALF_LOG2PI 0.9189385332046727
#define GP_LS_LOC 1.4142135623730951
#define GP_LS_SCALE 1.7320508075688772
#define GP_SIG_SF 2.0
#define GP_SIG_SN 2.0
#define GP_NUTS_WARMUP 150
#define GP_NUTS_WARMUP_WARM 50
#define GP_NUTS_WARMUP_EPS 15
#define GP_NUTS_THIN 1
#define GP_NUTS_MAXTREE 10
#define GP_NUTS_DELTAMAX 1000.0
#define GP_NUTS_DELTA 0.8
#define GP_NUTS_ALPHA 0.1
#define GP_NUTS_NFRAMES (GP_NUTS_MAXTREE + 2)

static const double GP_MU_SN = -4.605170185988091;

static uint64_t gp_rng;

static inline double gp_rand01 (void)
{
  gp_rng ^= gp_rng >> 12;
  gp_rng ^= gp_rng << 25;
  gp_rng ^= gp_rng >> 27;
  return (double)((gp_rng * 0x2545F4914F6CDD1DULL) >> 11) / (double)(1ULL << 53);
}

static inline double gp_clamp (double x, double lo, double hi)
{
  return x < lo ? lo : (x > hi ? hi : x);
}

static inline double gp_randn (void)
{
  double u1 = gp_rand01(); if (u1 < 1e-300) u1 = 1e-300;
  double u2 = gp_rand01();
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static inline double gp_kinetic (const double *pv, const double *Minv, uint64_t M)
{
  double k = 0.0;
  for (uint64_t i = 0; i < M; i++) k += pv[i] * pv[i] * Minv[i];
  return 0.5 * k;
}

static inline double gp_logsumexp2 (double a, double b)
{
  if (a == -HUGE_VAL) return b;
  if (b == -HUGE_VAL) return a;
  double m = a > b ? a : b;
  return m + log(exp(a - m) + exp(b - m));
}

static inline double gp_kern (
  const double *x1, const double *x2, uint64_t d,
  const double *rho, double sf2)
{
  double r2 = 0.0;
  for (uint64_t i = 0; i < d; i++) {
    double dx = x1[i] - x2[i];
    r2 += rho[i] * dx * dx;
  }
  double r = sqrt(r2);
  double s5r = GP_SQRT5 * r;
  return sf2 * (1.0 + s5r + (5.0 / 3.0) * r2) * exp(-s5r);
}

static inline double gp_kern2 (
  const double *x1, const double *x2, uint64_t d,
  const double *rho, double sf2, double *dkdr2)
{
  double r2 = 0.0;
  for (uint64_t i = 0; i < d; i++) {
    double dx = x1[i] - x2[i];
    r2 += rho[i] * dx * dx;
  }
  double s5r = GP_SQRT5 * sqrt(r2);
  double h = exp(-s5r);
  *dkdr2 = -(5.0 / 6.0) * sf2 * (1.0 + s5r) * h;
  return sf2 * (1.0 + s5r + (5.0 / 3.0) * r2) * h;
}

static int gp_cholesky (double *A, uint64_t n)
{
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j <= i; j++) {
      double s = A[i * n + j];
      for (uint64_t k = 0; k < j; k++)
        s -= A[i * n + k] * A[j * n + k];
      if (i == j) {
        if (s <= 1e-12) return -1;
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

static void gp_kinv (
  const double *L, double *Kinv, uint64_t n, double *Li)
{
  for (uint64_t c = 0; c < n; c++) {
    for (uint64_t i = 0; i < c; i++) Li[i * n + c] = 0.0;
    Li[c * n + c] = 1.0 / L[c * n + c];
    for (uint64_t i = c + 1; i < n; i++) {
      double s = 0.0;
      for (uint64_t j = c; j < i; j++) s -= L[i * n + j] * Li[j * n + c];
      Li[i * n + c] = s / L[i * n + i];
    }
  }
  for (uint64_t i = 0; i < n; i++)
    for (uint64_t k = i; k < n; k++) {
      double s = 0.0;
      for (uint64_t j = k; j < n; j++)
        s += Li[j * n + i] * Li[j * n + k];
      Kinv[i * n + k] = s;
      Kinv[k * n + i] = s;
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

static inline double gp_logh (double z)
{
  if (z > -6.0) {
    double h = z * gp_ncdf(z) + gp_npdf(z);
    if (h < 1e-300) h = 1e-300;
    return log(h);
  } else {
    double logphi = -0.5 * z * z - GP_HALF_LOG2PI;
    return logphi - 2.0 * log(-z) + log1p(-3.0 / (z * z));
  }
}

static int gp_build (
  const double *X, const double *Y, uint64_t n, uint64_t d,
  const double *rho, double sf2, double sn2, const double *noise,
  double *Kbase, double *chol, double *alpha, double *tmp,
  double *dK)
{
  for (uint64_t i = 0; i < n; i++) {
    Kbase[i * n + i] = sf2 + sn2 + (noise ? noise[i] : 0.0);
    for (uint64_t j = 0; j < i; j++) {
      double k;
      if (dK) {
        double dkdr2;
        k = gp_kern2(X + i * d, X + j * d, d, rho, sf2, &dkdr2);
        dK[i * n + j] = dkdr2;
        dK[j * n + i] = dkdr2;
      } else {
        k = gp_kern(X + i * d, X + j * d, d, rho, sf2);
      }
      Kbase[i * n + j] = k;
      Kbase[j * n + i] = k;
    }
  }
  double jit = 0.0;
  for (int a = 0; a < 7; a++) {
    memcpy(chol, Kbase, n * n * sizeof(double));
    if (jit > 0.0)
      for (uint64_t i = 0; i < n; i++)
        chol[i * n + i] += jit;
    if (gp_cholesky(chol, n) == 0) {
      gp_fwd(chol, Y, tmp, n);
      gp_bwd(chol, tmp, alpha, n);
      return 0;
    }
    jit = (jit == 0.0) ? 1e-8 : jit * 10.0;
  }
  return -1;
}

static int gp_objgrad (
  const double *X, const double *Y, uint64_t n, uint64_t d,
  const double *theta, double *grad, double *F,
  double *rho, const double *noise, double *Kbase, double *chol, double *Kinv,
  double *alpha, double *tmp, double *Li, double *dK)
{
  uint64_t P = d + 2;
  for (uint64_t j = 0; j < d; j++)
    rho[j] = exp(gp_clamp(theta[j], -18.0, 18.0));
  double p = gp_clamp(theta[d], -18.0, 18.0);
  double q = gp_clamp(theta[d + 1], -30.0, 6.0);
  double sf2 = exp(p);
  double sn2 = exp(q);

  if (gp_build(X, Y, n, d, rho, sf2, sn2, noise, Kbase, chol, alpha, tmp, dK) != 0)
    return -1;

  double quad = 0.0, logdet = 0.0;
  for (uint64_t i = 0; i < n; i++) {
    quad += Y[i] * alpha[i];
    logdet += log(chol[i * n + i]);
  }
  double lml = -0.5 * quad - logdet - 0.5 * (double)n * GP_LOG2PI;

  gp_kinv(chol, Kinv, n, Li);

  for (uint64_t j = 0; j < P; j++) grad[j] = 0.0;
  double gp_lik = 0.0, gq_lik = 0.0;

  for (uint64_t i = 0; i < n; i++) {
    const double *xi = X + i * d;
    for (uint64_t k = 0; k < n; k++) {
      double A = alpha[i] * alpha[k] - Kinv[i * n + k];
      if (i == k) {
        gp_lik += 0.5 * A * sf2;
        gq_lik += 0.5 * A * sn2;
        continue;
      }
      const double *xk = X + k * d;
      double c = 0.5 * A * dK[i * n + k];
      for (uint64_t j = 0; j < d; j++) {
        double dx = xi[j] - xk[j];
        grad[j] += c * dx * dx;
      }
      gp_lik += 0.5 * A * Kbase[i * n + k];
    }
  }

  double prior = 0.0;
  for (uint64_t j = 0; j < d; j++)
    grad[j] = grad[j] * rho[j];
  prior += -0.5 * (p * p) / (GP_SIG_SF * GP_SIG_SF);
  prior += -0.5 * ((q - GP_MU_SN) * (q - GP_MU_SN)) / (GP_SIG_SN * GP_SIG_SN);

  grad[d] = gp_lik - p / (GP_SIG_SF * GP_SIG_SF);
  grad[d + 1] = gq_lik - (q - GP_MU_SN) / (GP_SIG_SN * GP_SIG_SN);

  *F = lml + prior;
  return 0;
}

typedef struct {
  double *qm, *pm, *gm, *qp, *pp, *gpg, *qprop;
  double logw, sum_alpha;
  int nleaf, ndiv, stop;
} gp_tree;

typedef struct {
  const double *X, *Y;
  uint64_t n, d, M;
  double mu0, s0, alpha, eps;
  const double *noise, *Minv;
  double *theta, *grad, *rho, *Kbase, *chol, *Kinv, *alpha_b, *tmp, *Li, *dK;
  double *pv, *gcur, *sq, *sp, *sg;
  gp_tree *frames;
} gp_nuts_ctx;

static int gp_potential (gp_nuts_ctx *C, const double *qn, double *Uout, double *gradq)
{
  uint64_t d = C->d;
  double xi = qn[d];
  double tau = exp(gp_clamp(xi, -40.0, 40.0));
  for (uint64_t j = 0; j < d; j++) C->theta[j] = C->mu0 + xi + C->s0 * qn[j];
  C->theta[d] = qn[d + 1];
  C->theta[d + 1] = qn[d + 2];
  double F;
  if (gp_objgrad(C->X, C->Y, C->n, d, C->theta, C->grad, &F, C->rho, C->noise,
                 C->Kbase, C->chol, C->Kinv, C->alpha_b, C->tmp, C->Li, C->dK) != 0)
    return -1;
  if (!isfinite(F)) return -1;
  double sum_gu = 0.0;
  for (uint64_t j = 0; j < d; j++) {
    double gu = -C->grad[j];
    sum_gu += gu;
    gradq[j] = C->s0 * gu + qn[j];
  }
  double a2t2 = C->alpha * C->alpha + tau * tau;
  gradq[d] = sum_gu + (2.0 * tau * tau / a2t2 - 1.0);
  gradq[d + 1] = -C->grad[d];
  gradq[d + 2] = -C->grad[d + 1];
  double slab = 0.0;
  for (uint64_t j = 0; j < d; j++) slab += qn[j] * qn[j];
  *Uout = -F + 0.5 * slab + log(a2t2) - xi;
  return 0;
}

static int gp_leapfrog (gp_nuts_ctx *C, double *qv, double *pv, const double *grad_in,
                        double signed_eps, double *Unew, double *grad_out)
{
  uint64_t M = C->M;
  for (uint64_t i = 0; i < M; i++) pv[i] -= 0.5 * signed_eps * grad_in[i];
  for (uint64_t i = 0; i < M; i++) qv[i] += signed_eps * C->Minv[i] * pv[i];
  if (gp_potential(C, qv, Unew, grad_out) != 0 || !isfinite(*Unew)) return -1;
  for (uint64_t i = 0; i < M; i++) pv[i] -= 0.5 * signed_eps * grad_out[i];
  return 0;
}

static int gp_uturn (const double *qm, const double *pm, const double *qp, const double *pp,
                     const double *Minv, uint64_t M)
{
  double a = 0.0, b = 0.0;
  for (uint64_t i = 0; i < M; i++) {
    double dq = qp[i] - qm[i];
    a += dq * Minv[i] * pm[i];
    b += dq * Minv[i] * pp[i];
  }
  return (a < 0.0) || (b < 0.0);
}

static void gp_base_leaf (gp_nuts_ctx *C, const double *in_q, const double *in_p,
                          const double *in_g, int v, double H0, gp_tree *out)
{
  uint64_t M = C->M;
  size_t sz = M * sizeof(double);
  memcpy(out->qp, in_q, sz);
  memcpy(out->pp, in_p, sz);
  double U;
  int div = gp_leapfrog(C, out->qp, out->pp, in_g, (double) v * C->eps, &U, out->gpg);
  double H = div ? HUGE_VAL : U + gp_kinetic(out->pp, C->Minv, M);
  memcpy(out->qm, out->qp, sz);
  memcpy(out->pm, out->pp, sz);
  memcpy(out->gm, out->gpg, sz);
  memcpy(out->qprop, out->qp, sz);
  int diverged = div || (H - H0 > GP_NUTS_DELTAMAX) || !isfinite(H);
  double dH = H0 - H;
  out->logw = diverged ? -HUGE_VAL : dH;
  out->ndiv = diverged ? 1 : 0;
  out->stop = diverged;
  double a = dH; if (a > 0.0) a = 0.0;
  out->sum_alpha = diverged ? 0.0 : exp(a);
  out->nleaf = 1;
}

static void gp_build_tree (gp_nuts_ctx *C, const double *in_q, const double *in_p,
                           const double *in_g, int depth, int v, double H0, gp_tree *out)
{
  uint64_t M = C->M;
  size_t sz = M * sizeof(double);
  if (depth == 0) { gp_base_leaf(C, in_q, in_p, in_g, v, H0, out); return; }
  gp_build_tree(C, in_q, in_p, in_g, depth - 1, v, H0, out);
  if (out->stop) return;
  gp_tree *rt = &C->frames[depth];
  if (v == -1)
    gp_build_tree(C, out->qm, out->pm, out->gm, depth - 1, v, H0, rt);
  else
    gp_build_tree(C, out->qp, out->pp, out->gpg, depth - 1, v, H0, rt);
  double ltot = gp_logsumexp2(out->logw, rt->logw);
  if (rt->ndiv == 0 && rt->logw > -HUGE_VAL) {
    double lp = rt->logw - ltot;
    if (log(gp_rand01() + 1e-300) < lp) memcpy(out->qprop, rt->qprop, sz);
  }
  if (v == -1) {
    memcpy(out->qm, rt->qm, sz); memcpy(out->pm, rt->pm, sz); memcpy(out->gm, rt->gm, sz);
  } else {
    memcpy(out->qp, rt->qp, sz); memcpy(out->pp, rt->pp, sz); memcpy(out->gpg, rt->gpg, sz);
  }
  out->logw = ltot;
  out->ndiv += rt->ndiv;
  out->sum_alpha += rt->sum_alpha;
  out->nleaf += rt->nleaf;
  out->stop = out->stop || rt->stop || rt->ndiv > 0 ||
              gp_uturn(out->qm, out->pm, out->qp, out->pp, C->Minv, M);
}

static double gp_reasonable_eps (gp_nuts_ctx *C, const double *qv0, const double *g0, double U0)
{
  uint64_t M = C->M;
  size_t sz = M * sizeof(double);
  double eps = 1.0;
  for (uint64_t i = 0; i < M; i++) C->pv[i] = gp_randn() * sqrt(1.0 / C->Minv[i]);
  double H0 = U0 + gp_kinetic(C->pv, C->Minv, M);
  memcpy(C->sq, qv0, sz); memcpy(C->sp, C->pv, sz);
  double U1; int div = gp_leapfrog(C, C->sq, C->sp, g0, eps, &U1, C->sg);
  double H1 = div ? HUGE_VAL : U1 + gp_kinetic(C->sp, C->Minv, M);
  double logr = H0 - H1;
  double a = (logr > -0.6931471805599453) ? 1.0 : -1.0;
  int guard = 0;
  while (a * logr > -a * 0.6931471805599453 && guard++ < 60) {
    eps *= (a > 0.0) ? 2.0 : 0.5;
    memcpy(C->sq, qv0, sz); memcpy(C->sp, C->pv, sz);
    div = gp_leapfrog(C, C->sq, C->sp, g0, eps, &U1, C->sg);
    H1 = div ? HUGE_VAL : U1 + gp_kinetic(C->sp, C->Minv, M);
    logr = H0 - H1;
  }
  return eps;
}

static void gp_nuts_step (gp_nuts_ctx *C, double *qv, double *out_sa, int *out_nl)
{
  uint64_t M = C->M;
  size_t sz = M * sizeof(double);
  gp_tree *run = &C->frames[GP_NUTS_MAXTREE + 1];
  gp_tree *nt = &C->frames[GP_NUTS_MAXTREE];
  double U0;
  if (gp_potential(C, qv, &U0, C->gcur) != 0) { *out_sa = 0.0; *out_nl = 1; return; }
  for (uint64_t i = 0; i < M; i++) C->pv[i] = gp_randn() * sqrt(1.0 / C->Minv[i]);
  double H0 = U0 + gp_kinetic(C->pv, C->Minv, M);
  memcpy(run->qm, qv, sz); memcpy(run->pm, C->pv, sz); memcpy(run->gm, C->gcur, sz);
  memcpy(run->qp, qv, sz); memcpy(run->pp, C->pv, sz); memcpy(run->gpg, C->gcur, sz);
  memcpy(run->qprop, qv, sz);
  run->logw = 0.0; run->ndiv = 0; run->sum_alpha = 0.0; run->nleaf = 0; run->stop = 0;
  for (int depth = 0; depth < GP_NUTS_MAXTREE; depth++) {
    int v = (gp_rand01() < 0.5) ? 1 : -1;
    if (v == -1)
      gp_build_tree(C, run->qm, run->pm, run->gm, depth, -1, H0, nt);
    else
      gp_build_tree(C, run->qp, run->pp, run->gpg, depth, 1, H0, nt);
    if (nt->ndiv == 0 && nt->logw > -HUGE_VAL) {
      double lp = nt->logw - run->logw;
      if (lp > 0.0) lp = 0.0;
      if (log(gp_rand01() + 1e-300) < lp) memcpy(run->qprop, nt->qprop, sz);
    }
    if (v == -1) {
      memcpy(run->qm, nt->qm, sz); memcpy(run->pm, nt->pm, sz); memcpy(run->gm, nt->gm, sz);
    } else {
      memcpy(run->qp, nt->qp, sz); memcpy(run->pp, nt->pp, sz); memcpy(run->gpg, nt->gpg, sz);
    }
    run->logw = gp_logsumexp2(run->logw, nt->logw);
    run->sum_alpha += nt->sum_alpha;
    run->nleaf += nt->nleaf;
    run->ndiv += nt->ndiv;
    if (nt->stop || nt->ndiv > 0 ||
        gp_uturn(run->qm, run->pm, run->qp, run->pp, C->Minv, M)) break;
  }
  memcpy(qv, run->qprop, sz);
  *out_sa = run->sum_alpha;
  *out_nl = run->nleaf > 0 ? run->nleaf : 1;
}

static int tm_gp_suggest (lua_State *L)
{
  tk_dvec_t *Xd = tk_dvec_peek(L, 1, "X");
  tk_dvec_t *Yd = tk_dvec_peek(L, 2, "Y");
  uint64_t d = tk_lua_checkunsigned(L, 3, "n_dims");
  tk_dvec_t *cd = tk_dvec_peek(L, 4, "candidates");
  uint64_t nc = tk_lua_checkunsigned(L, 5, "n_candidates");
  uint64_t nr = lua_gettop(L) >= 6 ? tk_lua_checkunsigned(L, 6, "n_samples") : 10;
  if (nr < 1) nr = 1;
  tk_dvec_t *state = (lua_gettop(L) >= 8 && !lua_isnil(L, 8))
    ? tk_dvec_peek(L, 8, "state") : NULL;
  tk_dvec_t *lsb = (lua_gettop(L) >= 9 && !lua_isnil(L, 9))
    ? tk_dvec_peek(L, 9, "ls_buf") : NULL;
  tk_dvec_t *nzd = (lua_gettop(L) >= 10 && !lua_isnil(L, 10))
    ? tk_dvec_peek(L, 10, "noise") : NULL;

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
  uint64_t P = d + 2;
  uint64_t M = d + 3;

  double ym = 0.0;
  for (uint64_t i = 0; i < n; i++) ym += Yr[i];
  ym /= (double)n;
  double yv = 0.0;
  for (uint64_t i = 0; i < n; i++) {
    double t = Yr[i] - ym;
    yv += t * t;
  }
  double ysd = sqrt(yv / (double)n);
  if (ysd < 1e-12) ysd = 1.0;

  int have_noise = (nzd != NULL && nzd->n >= n);

  double *Ys = malloc(n * sizeof(double));
  double *xsd = malloc(d * sizeof(double));
  double *Xs = malloc(n * d * sizeof(double));
  double *cs = malloc(nc * d * sizeof(double));
  double *Kbase = malloc(n * n * sizeof(double));
  double *chol = malloc(n * n * sizeof(double));
  double *Kinv = malloc(n * n * sizeof(double));
  double *alpha = malloc(n * sizeof(double));
  double *tmp = malloc(n * sizeof(double));
  double *Li = malloc(n * n * sizeof(double));
  double *dK = malloc(n * n * sizeof(double));
  uint64_t nthr = (uint64_t) omp_get_max_threads();
  double *cand_scratch = malloc(nthr * 2 * n * sizeof(double));
  double *rho = malloc(d * sizeof(double));
  double *theta = malloc(P * sizeof(double));
  double *grad = malloc(P * sizeof(double));
  double *Minv = malloc(M * sizeof(double));
  double *qv = malloc(M * sizeof(double));
  double *pv = malloc(M * sizeof(double));
  double *gcur = malloc(M * sizeof(double));
  double *sq = malloc(M * sizeof(double));
  double *sp = malloc(M * sizeof(double));
  double *sg = malloc(M * sizeof(double));
  double *tree_store = malloc(GP_NUTS_NFRAMES * 7 * M * sizeof(double));
  gp_tree *frames = malloc(GP_NUTS_NFRAMES * sizeof(gp_tree));
  double *smp = malloc(nr * P * sizeof(double));
  double *Pbuf = malloc((nc ? nc : 1) * sizeof(double));
  double *lsacc = malloc(d * sizeof(double));
  double *nz = have_noise ? malloc(n * sizeof(double)) : NULL;

  void *allocs[] = { Ys, xsd, Xs, cs, Kbase, chol, Kinv, alpha, tmp, Li, dK,
    cand_scratch, rho, theta, grad, Minv, qv, pv, gcur, sq, sp, sg, tree_store, frames,
    smp, Pbuf, lsacc };
  int n_allocs = (int)(sizeof(allocs) / sizeof(allocs[0]));
  int alloc_fail = (have_noise && !nz) ? 1 : 0;
  for (int i = 0; i < n_allocs; i++)
    if (!allocs[i]) alloc_fail = 1;
  if (alloc_fail) {
    for (int i = 0; i < n_allocs; i++) free(allocs[i]);
    free(nz);
    return luaL_error(L, "alloc");
  }

  for (uint64_t i = 0; i < n; i++)
    Ys[i] = (Yr[i] - ym) / ysd;
  double ybest = Ys[0];
  for (uint64_t i = 1; i < n; i++)
    if (Ys[i] > ybest) ybest = Ys[i];

  if (nz != NULL) {
    double iv = 1.0 / (ysd * ysd);
    for (uint64_t i = 0; i < n; i++) {
      double v = nzd->a[i] * iv;
      nz[i] = v > 0.0 ? v : 0.0;
    }
  }

  for (uint64_t j = 0; j < d; j++) {
    double m = 0.0;
    for (uint64_t i = 0; i < n; i++) m += X[i * d + j];
    m /= (double)n;
    double v = 0.0;
    for (uint64_t i = 0; i < n; i++) {
      double t = X[i * d + j] - m;
      v += t * t;
    }
    double s = sqrt(v / (double)n);
    if (s < 1e-9) s = 1.0;
    xsd[j] = s;
  }
  for (uint64_t i = 0; i < n; i++)
    for (uint64_t j = 0; j < d; j++)
      Xs[i * d + j] = X[i * d + j] / xsd[j];
  for (uint64_t c = 0; c < nc; c++)
    for (uint64_t j = 0; j < d; j++)
      cs[c * d + j] = cand[c * d + j] / xsd[j];

  gp_rng = 0;
  for (uint64_t i = 0; i < n; i++)
    gp_rng ^= (uint64_t)(int64_t)llround(Yr[i] * 1e8);
  gp_rng ^= n * 2654435761ULL;
  if (gp_rng == 0) gp_rng = 1;

  double ls_mu = -2.0 * GP_LS_LOC - log((double) d);

  int warm = (state != NULL && state->n >= P);
  int warm_eps = (state != NULL && state->n >= P + 1);

  for (uint64_t f = 0; f < GP_NUTS_NFRAMES; f++) {
    double *b = tree_store + f * 7 * M;
    frames[f].qm = b; frames[f].pm = b + M; frames[f].gm = b + 2 * M;
    frames[f].qp = b + 3 * M; frames[f].pp = b + 4 * M; frames[f].gpg = b + 5 * M;
    frames[f].qprop = b + 6 * M;
  }
  for (uint64_t j = 0; j < d; j++) Minv[j] = 1.0;
  Minv[d] = 1.0;
  Minv[d + 1] = GP_SIG_SF * GP_SIG_SF;
  Minv[d + 2] = GP_SIG_SN * GP_SIG_SN;

  gp_nuts_ctx C;
  C.X = Xs; C.Y = Ys; C.n = n; C.d = d; C.M = M;
  C.mu0 = ls_mu; C.s0 = GP_LS_SCALE; C.alpha = GP_NUTS_ALPHA; C.eps = 1.0;
  C.noise = nz; C.Minv = Minv;
  C.theta = theta; C.grad = grad; C.rho = rho;
  C.Kbase = Kbase; C.chol = chol; C.Kinv = Kinv; C.alpha_b = alpha;
  C.tmp = tmp; C.Li = Li; C.dK = dK;
  C.pv = pv; C.gcur = gcur; C.sq = sq; C.sp = sp; C.sg = sg;
  C.frames = frames;

  if (warm) {
    double sxi = 0.0;
    for (uint64_t j = 0; j < d; j++) sxi += state->a[j] - ls_mu;
    double xi0 = sxi / (double) d;
    for (uint64_t j = 0; j < d; j++) qv[j] = (state->a[j] - ls_mu - xi0) / GP_LS_SCALE;
    qv[d] = xi0;
    qv[d + 1] = state->a[d];
    qv[d + 2] = state->a[d + 1];
  } else {
    for (uint64_t j = 0; j < d; j++) qv[j] = 0.0;
    qv[d] = log(GP_NUTS_ALPHA);
    qv[d + 1] = 0.0;
    qv[d + 2] = GP_MU_SN;
  }

  uint64_t nvalid = 0;
  double U0;
  double logeps_bar = 0.0;
  int eps_valid = 0;
  if (gp_potential(&C, qv, &U0, gcur) == 0) {
    double eps;
    uint64_t Wr;
    if (warm_eps) {
      logeps_bar = state->a[P];
      eps = exp(logeps_bar);
      Wr = GP_NUTS_WARMUP_EPS;
    } else {
      eps = gp_reasonable_eps(&C, qv, gcur, U0);
      Wr = warm ? GP_NUTS_WARMUP_WARM : GP_NUTS_WARMUP;
    }
    double mu_da = warm_eps ? logeps_bar : log(10.0 * eps);
    double Hbar = 0.0;
    C.eps = eps;
    for (uint64_t m = 1; m <= Wr; m++) {
      double sa; int nl;
      gp_nuts_step(&C, qv, &sa, &nl);
      double abar = sa / (double) nl;
      double w = 1.0 / ((double) m + 10.0);
      Hbar = (1.0 - w) * Hbar + w * (GP_NUTS_DELTA - abar);
      double logeps = mu_da - sqrt((double) m) / 0.05 * Hbar;
      double eta = pow((double) m, -0.75);
      logeps_bar = eta * logeps + (1.0 - eta) * logeps_bar;
      C.eps = exp(logeps);
    }
    C.eps = exp(logeps_bar);
    eps_valid = 1;
    for (uint64_t s = 0; s < nr; s++) {
      for (int t = 0; t < GP_NUTS_THIN; t++) {
        double sa; int nl;
        gp_nuts_step(&C, qv, &sa, &nl);
      }
      double *th = smp + nvalid * P;
      for (uint64_t j = 0; j < d; j++) th[j] = ls_mu + qv[d] + GP_LS_SCALE * qv[j];
      th[d] = qv[d + 1];
      th[d + 1] = qv[d + 2];
      nvalid++;
    }
  }

  if (nvalid == 0) {
    for (uint64_t j = 0; j < d; j++) smp[j] = ls_mu;
    smp[d] = 0.0;
    smp[d + 1] = log(0.1);
    nvalid = 1;
  }

  tk_dvec_t *ei;
  if (lua_gettop(L) >= 7 && !lua_isnil(L, 7)) {
    ei = tk_dvec_peek(L, 7, "ei_buf");
    tk_dvec_ensure(ei, nc);
    ei->n = nc;
    lua_pushvalue(L, 7);
  } else {
    ei = tk_dvec_create(L, nc);
    ei->n = nc;
  }

  for (uint64_t c = 0; c < nc; c++) { ei->a[c] = -1e300; Pbuf[c] = 0.0; }
  for (uint64_t j = 0; j < d; j++) lsacc[j] = 0.0;
  double lswsum = 0.0;

  for (uint64_t s = 0; s < nvalid; s++) {
    const double *th = smp + s * P;
    for (uint64_t j = 0; j < d; j++)
      rho[j] = exp(gp_clamp(th[j], -18.0, 18.0));
    double sf2 = exp(gp_clamp(th[d], -18.0, 18.0));
    double sn2 = exp(gp_clamp(th[d + 1], -30.0, 6.0));
    if (gp_build(Xs, Ys, n, d, rho, sf2, sn2, nz, Kbase, chol, alpha, tmp, NULL) != 0)
      continue;
    lswsum += 1.0;
    for (uint64_t j = 0; j < d; j++) lsacc[j] += rho[j];
    #pragma omp parallel
    {
      double *tloc = cand_scratch + (uint64_t) omp_get_thread_num() * 2 * n;
      double *vloc = tloc + n;
      #pragma omp for schedule(static)
      for (uint64_t c = 0; c < nc; c++) {
        const double *xc = cs + c * d;
        for (uint64_t i = 0; i < n; i++)
          tloc[i] = gp_kern(xc, Xs + i * d, d, rho, sf2);
        double mu = 0.0;
        for (uint64_t i = 0; i < n; i++)
          mu += tloc[i] * alpha[i];
        gp_fwd(chol, tloc, vloc, n);
        double vtv = 0.0;
        for (uint64_t i = 0; i < n; i++)
          vtv += vloc[i] * vloc[i];
        double s2 = sf2 - vtv;
        if (s2 < 1e-12) s2 = 1e-12;
        double sd = sqrt(s2);
        double z = (mu - ybest) / sd;
        double logei = 0.5 * log(s2) + gp_logh(z);
        double x = logei;
        double rcur = ei->a[c];
        if (x > rcur) {
          Pbuf[c] = Pbuf[c] * exp(rcur - x) + 1.0;
          ei->a[c] = x;
        } else {
          Pbuf[c] += exp(x - rcur);
        }
      }
    }
  }

  for (uint64_t c = 0; c < nc; c++) {
    if (Pbuf[c] > 0.0)
      ei->a[c] = ei->a[c] + log(Pbuf[c]) - log((double)nvalid);
    else
      ei->a[c] = -1e300;
  }

  if (lsb != NULL) {
    tk_dvec_ensure(lsb, d);
    lsb->n = d;
    for (uint64_t j = 0; j < d; j++) {
      double rb = lswsum > 0.0 ? lsacc[j] / lswsum : 1.0;
      if (rb < 1e-12) rb = 1e-12;
      lsb->a[j] = xsd[j] / sqrt(rb);
    }
  }

  if (state != NULL && nvalid > 0) {
    tk_dvec_ensure(state, P + 1);
    memcpy(state->a, smp + (nvalid - 1) * P, P * sizeof(double));
    if (eps_valid) {
      state->a[P] = logeps_bar;
      state->n = P + 1;
    } else {
      state->n = P;
    }
  }

  for (int i = 0; i < n_allocs; i++) free(allocs[i]);
  free(nz);

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
