# Hyperparameter Search

## Overview

`optimize.lua` provides a generic round-based adaptive search engine and
two domain-specific wrappers: one for TM regression hyperparameters, one
for Nystrom spectral embedding parameters. Both wrappers share the same
`M.search` core and sampler infrastructure.

The design prioritizes:
- Reuse of a single TM object across all search trials via `reconfigure`.
- Adaptive jitter narrowing via the 1/5th success rule.
- Deduplication to avoid re-evaluating equivalent configurations.
- Size preference at equal accuracy to bias toward smaller models.
- Optional subsampling of training data during the search phase.
- Final training with batched early stopping on the full dataset.

## Sampler System

`M.build_sampler(spec, global_dev)` constructs a sampler from a parameter
specification. Three types:

### Fixed

Scalar, boolean, or string. Always returns the same value. Indicates the
parameter is not searchable.

    clauses = 256  -- fixed at 256

### Range

Table with `min`, `max`, and optional `def`, `log`, `int`, `pow2`,
`round`, `dev`:

    clauses = { min = 64, max = 512, def = 256, int = true }
    specificity = { min = 1, max = 2000, log = true, def = 100 }

Sampling behavior:
- **Round 1**: Uniform random over the range (or log-uniform if `log=true`).
- **Rounds 2+**: Normal distribution centered on the current best value,
  with variance controlled by the jitter parameter.
- **Log scale**: Sampling and jitter operate in log space. Non-positive
  `min` values are shifted so the shifted minimum is 1 before taking logs.
- **Boundary reflection**: Samples outside `[min, max]` are reflected
  inward (mirrored at the boundary), then clamped as a final guard.
- **Rounding**: `int=true` rounds to nearest integer. `pow2=true` rounds
  to nearest power of 2. `round=N` rounds to nearest multiple of N.
  Rounding is applied after reflection but before final clamping.
- **Per-param deviation**: `dev` overrides the global `search_dev` for
  this parameter's initial jitter as a fraction of the span.

The jitter starts at `(dev or global_dev or 1.0) * span` where span is
`log(max) - log(min)` for log-scale or `max - min` otherwise. Jitter is
the standard deviation of the normal distribution used for centered
sampling.

### List

Array of discrete values. Uniform random selection, no jitter adaptation.

    loss = { "squared", "absolute" }

## Search Engine

`M.search` is the generic search loop. It accepts:

| Field | Purpose |
|---|---|
| `param_names` | Ordered list of parameter names to search over |
| `samplers` | Table of name -> sampler objects |
| `trial_fn(params, info)` | Evaluate a configuration; returns (score, metrics, result) |
| `rounds` | Number of narrowing rounds (default 6) |
| `trials` | Configurations per round (default 20) |
| `make_key(params)` | Deduplication key function; trials with seen keys are skipped |
| `constrain(params)` | In-place constraint enforcement before each trial |
| `size_fn(params)` | Returns a cost vector for size preference |
| `preference_tolerance` | Score tolerance for considering two configs equivalent |
| `cleanup(result)` | Called when a result is superseded by a better one |
| `skip_final` | If true, return best params without re-running the final trial |
| `rerun_final` | If false, keep the best trial's result as-is (default true) |
| `each(event)` | Callback for logging/monitoring |

### Round structure

Each round:
1. Sample `trials` configurations. Round 1 samples uniformly; rounds 2+
   sample centered on the global best.
2. Skip duplicates (via `make_key`).
3. Evaluate each non-duplicate via `trial_fn`.
4. Track per-round and global bests.
5. Recenter all range samplers on the global best parameters.
6. Adapt jitter based on the round's success rate.

After all rounds, optionally re-run the best configuration as a "final"
trial (signaled via `info.is_final = true` to the trial function).

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
is `{ clauses }`, so at equal accuracy, fewer clauses wins. For spectral,
cost is `{ n_dims, n_landmarks, |decay|, |bandwidth| }`.

This biases toward simpler models without sacrificing meaningful accuracy
gains.

### Deduplication

The `make_key` function maps parameters to a string. Seen keys are
skipped within a round (counted toward `skip_rate`) and across all
rounds. The TM wrapper's key rounds `clauses` to the nearest multiple of
8 and all other parameters to the nearest integer:

    key = "256|64|128|32|500"  -- clauses|tol|max|target|spec
    key = "256|64|128|32|500|10|35|64|36"  -- with absorb params

This prevents wasting trials on configurations that differ only by
insignificant amounts after rounding.

### Jitter adaptation

After each round, the success rate (fraction of non-duplicate trials that
improved the global best) determines a multiplicative factor applied to
every range sampler's jitter:

| Success rate | Factor | Interpretation |
|---|---|---|
| > 25% | 1.2x | Too conservative, widen |
| 15-25% | 1.0x | Sweet spot |
| 5-15% | 0.85x | Moderate shrink |
| < 5% | 0.7x | Aggressive shrink |

If > 80% of trials were duplicates (high skip rate), the jitter is
forcibly expanded (1.3x) regardless of success rate, since the search
space is exhausted at the current resolution. If fewer than 3 non-
duplicate samples ran, jitter is left unchanged (insufficient signal).

Jitter is clamped to `[0.1x, 2.0x]` of the initial value to prevent
runaway collapse or explosion.

This is the 1/5th success rule from evolutionary strategies, adapted with
four bands instead of the classic binary split.

## TM Hyperparameter Search

`M.regressor(args)` delegates to `optimize_tm`, which wraps `M.search`
with TM-specific logic.

### Searched parameters

Always searched:
- `clauses` — per-class clause count
- `clause_tolerance` — max vote contribution per clause
- `clause_maximum` — literal budget for Type I reward
- `target` — vote target for regression scaling
- `specificity` — pattern granularity

Additionally when `n_tokens` is set (sparse mode):
- `absorb_interval` — iterations between absorption passes
- `absorb_threshold` — full-state threshold for eligibility
- `absorb_maximum` — max slot replacements per class per pass
- `absorb_insert` — initial state for newly streamed TAs

Each parameter is specified as a fixed value or a range spec in `args`.

### Spec capping

Before building samplers, range specs are capped to enforce physical
limits:
- `clause_tolerance`, `clause_maximum` capped at `2 * features` (total
  input bits).
- `specificity` capped at `2 * features`.
- `absorb_maximum` capped at `features` (total slots).
- `absorb_threshold` capped at `2^(state_bits-1) - 2`.
- `absorb_insert` capped at `2^(state_bits-1) - 1`.

`cap_spec_max` adjusts the `min`, `max`, and `def` fields of range specs
in-place to not exceed the cap.

### Constraint enforcement

`constrain_tm_params` runs before every trial and before final training.
Invariants enforced in order:

1. `clause_tolerance <= 2 * features`, `clause_maximum <= 2 * features`.
2. `clause_tolerance <= clause_maximum` (swapped if inverted).
3. `target <= 8 * clause_tolerance`, `target >= 1`.
4. `specificity <= 2 * features`, `specificity >= 1`.
5. `absorb_maximum <= features`.
6. `absorb_threshold < 2^(state_bits-1) - 1`.
7. `absorb_insert > absorb_threshold`, `absorb_insert <= 2^(state_bits-1) - 1`.

These ensure sampled configurations are physically valid without
discarding them.

### Search phase

1. **Subsampling** (`search_subsample`): When set (e.g., 0.2), a random
   subset of training samples is selected via `fill_indices:shuffle:setn`.
   For sparse mode: the token CSR is subselected via `bits_select` and a
   new CSC index is built. For dense mode: the problems cvec is
   subselected via `bits_select`. Targets/solutions/codes are subselected
   accordingly. Rankings are shared (not subselected) — they reference
   the token vocabulary, not sample indices.

2. **Reusable search TM**: A single TM object is created with minimal
   placeholder parameters (`clauses=8`, etc.) and `reusable=true`. Each
   trial calls `reconfigure` with the sampled parameters, then
   `train_tm_simple` for `search_iterations` epochs (default 40). This
   avoids repeated allocation/deallocation across hundreds of trials.

3. **Evaluation**: `search_metric(tm, data)` is called after training.
   Returns `(score, metrics)`. The metric function is caller-provided,
   typically NDCG, P@k, or MSE on a held-out set.

4. **No result retention**: The search phase passes `skip_final=true` to
   `M.search` and returns `nil` as the result from `trial_fn`. The search
   TM's state is overwritten each trial. Only the best parameters survive.

5. **Cleanup**: The search TM is destroyed after all rounds complete.

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

`M.spectral(args)` searches over Nystrom spectral embedding parameters
using the same `M.search` infrastructure.

### Searched parameters

- `n_landmarks` — number of landmark points for Nystrom approximation
- `n_dims` — number of spectral dimensions to retain
- `decay` — kernel decay parameter for the inverted index similarity
- `bandwidth` — kernel bandwidth parameter

Each can be fixed or a range spec.

### Constraint

`n_dims <= n_landmarks` is enforced before each trial. The number of
spectral dimensions cannot exceed the landmark count since the rank of
the Nystrom approximation is bounded by the number of landmarks.

### Trial function

Each trial builds a full spectral embedding via
`M.build_spectral_nystrom`, then evaluates ranking accuracy via
`M.score_spectral_eval` (NDCG against ground-truth neighbor lists).
Requires `expected_ids`, `expected_offsets`, `expected_neighbors`,
`expected_weights` in `args` to have something to evaluate against.

### Size preference

Cost vector is `{ n_dims, n_landmarks, |decay|, |bandwidth| }`. At equal
NDCG: fewer dimensions, then fewer landmarks, then smaller kernel
parameters.

### No-search fast path

If no parameter is a range spec, or if no evaluation data is provided, or
`rounds=0`: builds the embedding once with the fixed/default values and
returns it directly. Optionally evaluates if expected data is available.

### Cleanup

Superseded spectral models are destroyed via `M.destroy_spectral`, which
frees `raw_codes`, `ids`, `landmark_ids`, `eigenvectors`, `eigenvalues`,
`landmark_chol`, and `col_means`.

The final model is returned with `rerun_final=false` — the best trial's
model is kept as-is since spectral embedding is deterministic for given
parameters.

## Callback Protocol

Both wrappers accept an `each` callback that receives structured event
tables.

TM search events flow through to the callback directly:
- `each(tm, is_final, metrics, params, epochs, round, trial, rounds, trials, global_best, local_best)` — called after each batch of training.
- Return `false` to abort the current training run.

Spectral search events are wrapped:
- `{ event = "sample", round, trial, ... }` — before each trial.
- `{ event = "eval", round, trial, score, metrics, ... }` — after each trial.
- `{ event = "round_end", round, ... }` — after each round with
  adaptation stats.
- `{ event = "done", best_params, best_score, best_metrics }` — search
  complete.

The generic `M.search` also emits `{ event = "trial" }` and
`{ event = "round" }` events to its own `each` callback, which the
spectral wrapper translates into the domain-specific format above.
