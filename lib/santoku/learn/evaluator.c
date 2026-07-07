#include <santoku/lua/utils.h>
#include <santoku/ivec.h>
#include <santoku/dvec.h>
#include <santoku/fvec.h>
#include <math.h>
#include <float.h>

static inline double tm_eval_err (
  tk_fvec_t *pf, tk_dvec_t *pd, tk_dvec_t *ed, tk_ivec_t *ei, uint64_t i, double *exp_out
) {
  double exp_val = ed ? ed->a[i] : (double) ei->a[i];
  double pred_val = pf ? (double) pf->a[i] : pd->a[i];
  if (exp_out) *exp_out = exp_val;
  return fabs(pred_val - exp_val);
}

static inline int tm_regress_accuracy (lua_State *L)
{
  lua_settop(L, 2);
  tk_fvec_t *predicted_f = tk_fvec_peekopt(L, 1);
  tk_dvec_t *predicted_d = predicted_f ? NULL : tk_dvec_peek(L, 1, "predicted");
  tk_dvec_t *expected_d = tk_dvec_peekopt(L, 2);
  tk_ivec_t *expected_i = expected_d ? NULL : tk_ivec_peek(L, 2, "expected");
  uint64_t n = predicted_f ? predicted_f->n : predicted_d->n;
  if ((expected_d && expected_d->n != n) || (expected_i && expected_i->n != n))
    return luaL_error(L, "predicted and expected must have same length");
  double total = 0.0, min_err = DBL_MAX, max_err = 0.0, sum_exp = 0.0;
  #pragma omp parallel for reduction(+:total,sum_exp) reduction(min:min_err) reduction(max:max_err)
  for (uint64_t i = 0; i < n; i++) {
    double exp_val;
    double err = tm_eval_err(predicted_f, predicted_d, expected_d, expected_i, i, &exp_val);
    total += err;
    sum_exp += exp_val;
    if (err < min_err) min_err = err;
    if (err > max_err) max_err = err;
  }
  double mean = n > 0 ? total / n : 0.0;
  double mean_exp = n > 0 ? sum_exp / n : 0.0;
  double nmae = mean_exp > 0 ? mean / mean_exp : 0.0;
  double var = 0.0;
  #pragma omp parallel for reduction(+:var)
  for (uint64_t i = 0; i < n; i++) {
    double err = tm_eval_err(predicted_f, predicted_d, expected_d, expected_i, i, NULL);
    var += (err - mean) * (err - mean);
  }
  double std = n > 1 ? sqrt(var / (n - 1)) : 0.0;
  lua_newtable(L);
  lua_pushnumber(L, total);
  lua_setfield(L, -2, "total");
  lua_pushnumber(L, mean);
  lua_setfield(L, -2, "mean");
  lua_pushnumber(L, n > 0 ? min_err : 0.0);
  lua_setfield(L, -2, "min");
  lua_pushnumber(L, max_err);
  lua_setfield(L, -2, "max");
  lua_pushnumber(L, std);
  lua_setfield(L, -2, "std");
  lua_pushnumber(L, nmae);
  lua_setfield(L, -2, "nmae");
  return 1;
}

static luaL_Reg tm_evaluator_fns[] =
{
  { "regress_accuracy", tm_regress_accuracy },
  { NULL, NULL }
};

int luaopen_santoku_learn_evaluator (lua_State *L)
{
  lua_newtable(L);
  tk_lua_register(L, tm_evaluator_fns, 0);
  return 1;
}
