# Hyperparameter Search

## Overview

`optimize.lua` provides a GP-BO (Gaussian Process Bayesian Optimization)
search engine and a ridge regression wrapper. Both share the same
`M.search` core and sampler infrastructure.

LHS initialization fills the space before GP-BO takes over. Ridge
precomputes the gram matrix once to avoid per-trial eigendecomposition.

## Sampler System

`M.build_sampler(spec, global_dev)` constructs a sampler from a parameter
specification. Three types:

### Fixed

Scalar, boolean, or string. Always returns the same value. Indicates the
parameter is not searchable.

    clauses = 256  -- fixed at 256

### Range

Table with `min`, `max`, and optional `def`, `log`, `int`, `pow2`,
`round`, `dev` --
    lambda = { min = 0, max = 4, def = 1.0 }
    propensity_a = { min = 0, max = 4.0, def = 0.5 }

Sampling behavior:
- **LHS initialization**: The first `2 * n_dims + 1` trials use Latin
  Hypercube Sampling for space-filling coverage.
- **GP-guided**: After LHS, `gp.suggest` generates LHS candidate points,
  evaluates Expected Improvement via GP posterior, and selects the best.
- **Log scale**: Sampling operates in log space. Non-positive `min`
  values are shifted so the shifted minimum is 1 before taking logs.
- **Boundary reflection**: Samples outside `[min, max]` are reflected
  inward (mirrored at the boundary), then clamped as a final guard.
- **Rounding**: `int=true` rounds to nearest integer. `pow2=true` rounds
  to nearest power of 2. `round=N` rounds to nearest multiple of N.
- **Per-param deviation**: `dev` overrides the global `search_dev` for
  this parameter's jitter as a fraction of the span.

Each range sampler provides `normalize(x) -> [0,1]` and
`denormalize(u) -> original scale` functions for GP-BO normalization.

### List

Array of discrete values. Uniform random selection.

    loss = { "squared", "absolute" }

## Search Engine

`M.search` is the generic search loop. It accepts:

| Field | Purpose |
|---|---|
| `param_names` | Ordered list of parameter names to search over |
| `samplers` | Table of name -> sampler objects |
| `trial_fn(params, info)` | Evaluate a configuration; returns (score, metrics, result) |
| `trials` | Total number of configurations to evaluate (default 120) |
| `constrain(params)` | In-place constraint enforcement before each trial |
| `cost_fn(params)` | Returns a cost scalar for cost-cooled EI |
| `cost_beta` | Exponent for cost cooling (default 0.0 = disabled) |
| `skip_final` | If true, return best params without re-running the final trial |
| `each(event)` | Callback for logging/monitoring |
| `n_candidates` | LHS candidate pool size for GP-BO (default 500) |
| `n_hyper_restarts` | GP hyperparameter optimization restarts (default 20) |

### Search structure

1. Identify searchable dimensions (range samplers with normalize/
   denormalize).
2. Generate `2 * n_dims + 1` LHS points for initial space-filling
   exploration.
3. For trials 1..n_initial: sample from LHS points. Fixed and list
   params are sampled independently.
4. For trials n_initial+1..trials: generate `n_candidates` LHS
   candidate points, evaluate Expected Improvement via
   `gp.suggest(X_obs, Y_obs, n_dims, candidates, n_candidates,
   n_hyper_restarts)`, select the candidate with highest EI.
5. Evaluate each via `trial_fn`.
6. Record normalized observation in `X_obs`/`Y_obs` for GP model.

After all trials, optionally re-run the best configuration as a "final"
trial (signaled via `info.is_final = true` to the trial function).

### GP-BO Module (`gp.c`)

Single function: `gp.suggest(X_dvec, Y_dvec, n_dims, candidates_dvec,
n_candidates, n_restarts)` -> `ei_dvec`.

- X_dvec: flat `n_obs * n_dims` (normalized [0,1])
- Y_dvec: `n_obs` scores
- Matern 5/2 kernel with ARD length scales
- Cholesky-based inference with auto-jitter on failure
- Y standardized internally (zero mean, unit variance)
- Hyperparameter optimization: random restarts maximizing log marginal
  likelihood
- Returns Expected Improvement scores for each candidate

### Cost-cooled EI

When `cost_fn` and `cost_beta > 0` are provided, EI is modulated:

    acquisition = EI * (cost / max_cost) ^ (-cost_beta)

This biases search toward cheaper configurations at equal EI.

### All-fixed fast path

If every sampler is fixed (no searchable parameters), the search phase is
skipped entirely. The fixed configuration is assembled once and either
returned as-is (`skip_final`) or passed directly to `trial_fn` with
`is_final = true`.

### Callback events

`each` receives structured event tables:

    { event = "trial", trial, trials, params, score, metrics,
      global_best_score, is_new_best, phase }

Where `phase` is `"lhs"` for initial Latin Hypercube trials or `"gp"`
for GP-guided trials.

## Ridge Regression Search

`M.ridge(args)` searches over ridge regression classifier
hyperparameters using the same `M.search` infrastructure.

### Searched parameters

- `lambda` -- L2 regularization strength
- `propensity_a` -- propensity weighting exponent for tail labels
- `propensity_b` -- propensity weighting offset

Each can be fixed or a range spec.

### Precomputed gram matrix

Before the search loop, `ridge.gram(data)` eigendecomposes
`X'X = Q Λ Q'` once and stores `Q`, eigenvalues, `Q'X'Y`, and per-label
counts. This is the dominant cost (d² eigendecomposition).

Per-trial `ridge.create({gram=gram, lambda=, propensity_a=,
propensity_b=})` computes `W = Q * diag(1/(λ_i + λ)) * Q'X'Y_prop`
via a single dgemm.

### Input modes

**Dense mode** (when `val_targets` provided): Searches `lambda` only.
Trial function uses `gram:regress` + `eval.regression_accuracy`.
Score: negated MAE.

**Label mode** (when `val_expected_offsets` provided): Searches
`lambda`, `propensity_a`, `propensity_b`. Trial function uses
`gram:label` + `eval.retrieval_ks`. Score: micro F1.

### Return values

Returns 2 values:
1. Best ridge model (userdata)
2. Best parameters table `{ lambda [, propensity_a, propensity_b] }`
