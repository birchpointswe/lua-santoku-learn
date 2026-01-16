# Elbow Refactor Plan: NDCG-Focused Optimization

## Current State

Two different "elbow" concepts exist:

1. **`select_elbow`** - Eigenvalue-based dimension selection
   - Determines how many spectral dimensions to keep
   - Uses eigenvalue curve analysis (first_gap, plateau, etc.)
   - **KEEP THIS** - it's about code dimensionality, not neighbor cutoff

2. **`elbow` (in eval_params)** - Neighbor cutoff for retrieval
   - Determines where to cut off kNN list for quality computation
   - Used to compute `quality` metric (precision at cutoff)
   - Combined with `score` (NDCG) into `combined` metric
   - **DECOUPLE THIS** - move to separate phase

## What Changes

### Phase 1: Simplify optimize.lua

**Remove from spectral/encoder optimization loop:**
- `eval_cfg.elbow` / `eval_params.elbow` - neighbor cutoff elbow
- `eval_cfg.elbow_alpha` / `eval_params.elbow_alpha`
- `quality` metric computation during optimization
- `combined` metric as optimization target
- `sample_elbow()` calls for eval elbow (keep for select_elbow)

**Keep:**
- `select_elbow` / `select_elbow_alpha` - dimension selection
- `score` (NDCG) as primary optimization metric
- Clustering/dendrogram evaluation

**Simplify eval_cfg:**
```lua
-- Before
eval = {
  elbow = { def = "first_gap", "first_gap", "plateau" },
  elbow_alpha = { ... },
  select_elbow = { ... },
  ranking = "ndcg",
  target = "combined",
}

-- After
eval = {
  ranking = "ndcg",  -- or "spearman", "pearson"
  -- select_elbow stays if using dimension selection
}
```

### Phase 2: Simplify evaluator/capi.c

**score_retrieval changes:**
- Make elbow parameter optional (default: no cutoff)
- When no elbow: compute NDCG on full ranking, skip quality
- Return `{ score = ndcg_value }` when no elbow
- Keep quality/combined computation available for elbow tuning phase

### Phase 3: Update test files

**newsgroups.lua and others:**
- Remove elbow options from eval config
- Use `target = "score"` or just remove target (default to score)
- Simplify config significantly

### Phase 4: Elbow Tuning (Separate Function)

**New function: `optimize.elbow()`**

```lua
M.elbow = function (opts)
  -- opts.index: ANN index with codes
  -- opts.ids: sample IDs
  -- opts.expected_ids/offsets/neighbors/weights: ground truth adjacency
  -- opts.elbow_methods: { "first_gap", "plateau", "lmethod" }
  -- opts.alpha_ranges: per-method alpha ranges
  -- opts.n_dims: for hamming distance computation

  -- For each elbow/alpha combo:
  --   1. Retrieve neighbors using index
  --   2. Apply elbow cutoff
  --   3. Compute quality (precision at cutoff)
  --   4. Track best

  -- Return: best_elbow, best_alpha, best_quality
end
```

**When to call:**
- After encoder training
- On predicted codes (not ground truth)

**What data to use:**
- Validation set might be too small
- Options:
  a. Train set predictions (risk: overfitting elbow to train)
  b. Cross-validation on train
  c. Larger holdout from train
  d. Use train but regularize (prefer simpler elbows)

### Files to Modify

1. **lib/santoku/tsetlin/optimize.lua**
   - Remove elbow sampling from optimization loops
   - Remove quality/combined from metrics
   - Simplify eval_cfg handling
   - Add new `M.elbow()` function

2. **lib/santoku/tsetlin/evaluator/capi.c**
   - Make elbow optional in score_retrieval
   - Skip quality when no elbow provided

3. **test/spec/santoku/tsetlin/encoder/newsgroups.lua**
   - Simplify eval config
   - Add elbow tuning phase after encoder

4. **test/spec/santoku/tsetlin/classify/*.lua**
   - Update if they use elbow in eval

## Tentative Elbow Tuning Strategy

**Problem:** Validation set (~1000 samples) may be too small for robust elbow tuning.

**Options:**

1. **Use train predictions**
   - Pro: Large sample size
   - Con: May overfit elbow to train distribution
   - Mitigation: Use regularization (prefer smaller alpha values)

2. **K-fold cross-validation on train**
   - Pro: Uses all train data, reduces overfitting
   - Con: Requires multiple encoder predictions, slow

3. **Bootstrap on train predictions**
   - Pro: Uncertainty estimation
   - Con: Still based on train distribution

4. **Use train + validation combined**
   - Pro: Larger sample, includes validation distribution
   - Con: "Peeking" at validation

**Recommended approach:**
- Use train predictions for elbow tuning
- Apply regularization: prefer elbows with smaller cutoffs (less aggressive)
- Report confidence intervals via bootstrap if needed

## Migration Path

1. Add feature flag: `use_legacy_elbow_optimization = true`
2. When false, use new NDCG-only optimization
3. Deprecate and remove legacy path later

## Open Questions

1. Should `select_elbow` (dimension selection) also be decoupled?
   - Currently it affects code quality during spectral
   - Might want to keep it coupled

2. How to handle downstream users who expect `combined` metric?
   - Keep computing it for reporting, just don't optimize for it?

3. Elbow tuning on which ground truth?
   - Category-based (same class = relevant)?
   - Or use actual retrieval quality?
