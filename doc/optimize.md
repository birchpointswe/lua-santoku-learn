# Hyperparameter Search

## Overview

`optimize.lua` provides a GP-BO (Gaussian Process Bayesian Optimization)
search engine and three domain-specific wrappers: TM regression
hyperparameters, Nystrom spectral embedding parameters, and ridge
regression classifier parameters. All wrappers share the same `M.search`
core and sampler infrastructure.

A single TM object is reused across all search trials via `reconfigure`.
LHS initialization fills the space before GP-BO takes over. Duplicate
configurations (by `make_key`) are skipped. At equal accuracy, smaller
models are preferred. Training data can be subsampled during search.
Final training uses batched early stopping on the full dataset. Ridge
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
    clauses = { min = 64, max = 512, def = 256, int = true }
    specificity = { min = 1, max = 2000, log = true, def = 100 }
    features = { def = 4096, min = 512, max = 8192, pow2 = true }

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
| `make_key(params)` | Deduplication key function; trials with seen keys are skipped |
| `constrain(params)` | In-place constraint enforcement before each trial |
| `size_fn(params)` | Returns a cost vector for size preference |
| `preference_tolerance` | Score tolerance for considering two configs equivalent |
| `cleanup(result)` | Called when a result is superseded by a better one |
| `skip_final` | If true, return best params without re-running the final trial |
| `rerun_final` | If false, keep the best trial's result as-is (default true) |
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
5. Skip duplicates (via `make_key`).
6. Evaluate each non-duplicate via `trial_fn`.
7. Record normalized observation in `X_obs`/`Y_obs` for GP model.

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

### All-fixed fast path

If every sampler is fixed (no searchable parameters), the search phase is
skipped entirely. The fixed configuration is assembled once and either
returned as-is (`skip_final`) or passed directly to `trial_fn` with
`is_final = true`.

### Preference function

A new configuration is preferred over the current best when:
- Its score exceeds the best by more than `preference_tolerance`, OR
- Its score is within `tolerance` of the best AND its cost vector is
  lexicographically smaller.

Cost comparison is element-wise left-to-right. For the TM wrapper, cost
is `{ clauses, features }`, so at equal accuracy, fewer clauses wins,
then fewer features. For spectral, cost is `{ n_dims, n_landmarks,
|decay| }`.

### Deduplication

The `make_key` function maps parameters to a string. Seen keys are
skipped. The TM wrapper's key rounds specificity to `.2f` precision
and other parameters to nearest integer:

    key = "4096|4|64|128|32|500.00"  -- features|clauses|tol|max|target|spec
    key += "|10|35|64|36"            -- with absorb params
    key += "|1.20|-0.30|0.70|2.10"   -- with alpha params

### Callback events

`each` receives structured event tables:

    { event = "trial", trial, trials, params, score, metrics,
      global_best_score, phase }

Where `phase` is `"lhs"` for initial Latin Hypercube trials or `"gp"`
for GP-guided trials.

## TM Hyperparameter Search

`M.regressor(args)` delegates to `optimize_tm`, which wraps `M.search`
with TM-specific logic.

### Searched parameters

Core parameters (always present):
- `features` -- input feature slots per class (searchable with `pow2`)
- `clauses` -- per-class clause count (chunks per polarity, N -> N*16)
- `clause_maximum` -- literal budget for Type I reward
- `clause_tolerance_fraction` -- fraction in [0.01, 1.0]; tolerance =
  `round(fraction * clause_maximum)`
- `target_fraction` -- fraction in [0.01, 2.0]; target =
  `round(fraction * 8 * clause_tolerance)`
- `specificity` -- pattern granularity

Sparse mode (when `n_tokens` is set):
- `absorb_interval` -- iterations between absorption passes
- `absorb_threshold` -- full-state threshold for eligibility
- `absorb_maximum` -- max slot replacements per class per pass
- `absorb_insert_offset` -- offset above threshold for new TA state;
  `absorb_insert = absorb_threshold + absorb_insert_offset`

Per-output modulation (when `output_weights` provided):
- `alpha_specificity` -- modulates specificity per output dim
- `alpha_target` -- modulates target per output dim
- `alpha_tolerance` -- modulates tolerance per output dim
- `alpha_maximum` -- modulates maximum per output dim

Each is specified as a fixed value or a range spec in `args`. Default
alpha spec when `output_weights` is provided: `{ min = -3, max = 3,
def = 0 }` (no modulation at default).

### Per-output modulation

When `output_weights` (dvec, typically eigenvalues) is provided, each
output dimension gets its own tolerance, maximum, specificity threshold,
and target via:

    value_for_dim_i = base * exp(alpha * (w_norm_i - 0.5))

where `w_norm_i = (weight_i - min) / (max - min)`. GP-BO can then
find that leading eigenvalue dimensions need different hyperparameters
than trailing ones.

### Spec capping

Before building samplers, range specs are capped to enforce physical
limits:
- `clause_maximum` capped at `2 * features` (total input bits).
- `specificity` capped at `2 * features`.
- `absorb_maximum` capped at `features` (total slots).
- `absorb_threshold` capped at `2^(state_bits-1) - 2`.
- `absorb_insert_offset` capped at `2^(state_bits-1) - 1`.

### Constraint enforcement

`constrain_tm_params` runs before every trial and before final training:

1. `clause_maximum <= 2 * features`.
2. `clause_tolerance = round(clause_tolerance_fraction * clause_maximum)`,
   clamped to `[1, clause_maximum]`.
3. `target = round(target_fraction * 8 * clause_tolerance)`, clamped to
   `[1, 8 * clause_tolerance]`.
4. `specificity` clamped to `[1, 2 * features]`.
5. `absorb_maximum <= features`.
6. `absorb_threshold < 2^(state_bits-1) - 1`.
7. `absorb_insert = min(absorb_threshold + absorb_insert_offset,
   2^(state_bits-1) - 1)`.
8. Per-class ivecs computed from alphas + output_weights if applicable.
   Per-class tolerance/maximum ordering and target capping enforced.

### Target subsampling

`search_subsample_targets` reduces the number of output dimensions
during search. When set (e.g., 8), a uniformly-spaced subset of dims
is selected. Rankings are subselected via `select_ranking_segments`.
`output_weights` are subselected to match. The full set is used for
final training.

### Search phase

1. **Sample subsampling** (`search_subsample_samples`): When set
   (e.g., 0.2), a random subset of training samples is selected.
   For sparse mode: tokens are subselected via `bits_select` and a
   new CSC index is built. Rankings are shared (not subselected).

2. **Reusable search TM**: A single TM object is created with minimal
   placeholder parameters and `reusable=true`. Each trial calls
   `reconfigure` with the sampled parameters, then `train_tm_simple`
   for `search_iterations` epochs (default 40).

3. **Evaluation**: `search_metric(tm, data)` is called after training.
   Returns `(score, metrics)`. Caller-provided, typically negated MAE
   for regression or macro F1 for classification.

4. **No result retention**: The search phase passes `skip_final=true`
   to `M.search`. Only the best parameters survive.

5. **Cleanup**: The search TM is destroyed after all trials complete.

### Final training

With the best parameters from the search:

1. A fresh TM is created via `create_final_tm` with the discovered
   parameters (not reusable placeholders). For sparse mode, absorption
   parameters are set at creation time.

2. `train_tm_batched` trains on the full dataset with:
   - `final_iterations` total epochs (default 400).
   - `final_batch` epochs per batch (default 10).
   - `final_patience` batches without improvement before stopping
     (default 4).
   - `early_tolerance` minimum improvement to count as progress.

3. Batched early stopping:
   - After each batch, `metric_fn` evaluates the current model.
   - If the score improves by more than `tolerance`, checkpoint the model
     (actions + mapping saved to a cvec) and reset the patience counter.
   - If patience is exhausted, stop training and restore to the last
     checkpoint.
   - The `each` callback receives progress after each batch and can
     return `false` to abort training early.

The final TM, its metrics, and the best parameters are returned to the
caller.

## Spectral Embedding Search

`M.spectral(args)` builds a Nystrom spectral embedding. Currently
operates as a direct builder (no multi-trial search over spectral
parameters).

### Parameters

- `n_landmarks` -- number of landmark points for Nystrom approximation
- `n_dims` -- number of spectral dimensions to retain
- `decay` -- kernel decay parameter for the inverted index similarity

Each can be fixed or a range spec.

### Constraint

`n_dims <= n_landmarks` is enforced before each trial.

### Builder

`M.build_spectral_nystrom` calls `spectral.encode` with the inv index
and parameters. Returns a model table with `raw_codes`, `ids`, `dims`,
`encoder`, `eigenvalues`, `landmark_ids`, `n_landmarks`, `decay`.

### Evaluation (optional)

When `expected_ids`, `expected_offsets`, `expected_neighbors`,
`expected_weights`, and `ranking` are provided, `M.score_spectral_eval`
evaluates ranking accuracy via NDCG against ground-truth neighbor lists.
Both raw_codes (cosine) and optionally kernel_index scores are computed.

### Cleanup

`M.destroy_spectral` frees `raw_codes`, `ids`, `landmark_ids`,
`eigenvectors`, `eigenvalues`, `landmark_chol`, and `col_means`.

## Ridge Regression Search

`M.ridge(args)` searches over ridge regression classifier
hyperparameters using the same `M.search` infrastructure.

### Searched parameters

- `lambda` -- L2 regularization strength
- `propensity_a` -- propensity weighting exponent for tail labels
- `propensity_b` -- propensity weighting offset

Each can be fixed or a range spec.

### Precomputed gram matrix

Before the search loop, `ridge.precompute(data)` eigendecomposes
`X'X = Q Λ Q'` once and stores `Q`, eigenvalues, `Q'X'Y`, and per-label
counts. This is the dominant cost (d² eigendecomposition).

Per-trial `ridge.create({gram=gram, lambda=, propensity_a=,
propensity_b=})` computes `W = Q * diag(1/(λ_i + λ)) * Q'X'Y_prop`
via a single dgemm. This is ~2.2x faster than per-trial Cholesky+solve
for d=4096, 30-trial search.

### Input modes

**Dense mode**: `codes` (dvec) + `n_dims`. XtX via `cblas_dsyrk`.

**Sparse mode**: `feature_offsets` + `feature_indices` + `n_features`
(+ optional `feature_weights`). XtX via outer product accumulation.
Encode uses `cblas_daxpy` for SIMD inner loop.

### Trial function

Each trial:
1. `ridge.create({gram=gram, lambda=, propensity_a=, propensity_b=})`
2. `r:encode(codes, n_samples, k)` -> `pred_offsets`, `pred_neighbors`,
   `pred_scores` (CSR, top-k labels per sample sorted by score desc)
3. `evaluator.retrieval_ks({pred_offsets, pred_neighbors, pred_scores,
   expected_offsets, expected_neighbors})` -> `ks`, `oracle`, `thresh`
4. Return `{oracle=oracle, thresh=thresh}` as metrics

### Score function

Default: `score_fn(ret) = ret.thresh.macro_f1` -- optimizes the
achievable threshold-based macro F1 (not oracle). Can be overridden via
`args.score_fn`.

### Return values

Returns 3 values (from `M.search`):
1. Best ridge model (userdata) or nil
2. Best parameters table `{ lambda, propensity_a, propensity_b }`
3. Best metrics table `{ oracle = {...}, thresh = {...} }`

## Callback Protocol

TM search events:
- `each(ev)` where `ev = { tm, is_final, metrics, params, epoch,
  trial, trials, global_best_score, best_epoch_score, phase }`.
  Called after each training batch (search and final).
- Return `false` to abort the current training run.

Ridge search events flow through `M.search` --- `{ event = "trial", trial, trials, params, score, metrics,
  global_best_score, phase }`

Spectral events:
- `{ event = "spectral_result", n_dims, n_landmarks, trace_ratio }`
- `{ event = "eval", score, metrics, ... }` (when evaluation data
  provided)
- `{ event = "done", best_params, best_score, best_metrics }`
