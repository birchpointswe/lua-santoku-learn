# Architecture and Usage Patterns

## Module Overview

| Module | Role | Reference |
|---|---|---|
| `ridge.c` | Ridge regression classifier with propensity weighting | -- |
| `optimize.lua` | GP-BO hyperparameter search for ridge | [optimize.md](optimize.md) |
| `spectral.c` | Kernel spectral embedding via RP-Cholesky | [spectral.md](spectral.md) |
| `evaluator.c` | Classification, regression, ranking, retrieval metrics; clustering | [evaluation.md](evaluation.md) |
| `ann.h` | Multi-index hashing ANN with optional float reranking | [ann.md](ann.md) |
| `csr.c` | CSR/CSC utilities: tokenization, BNS/AUC feature weighting, merge, standardize | -- |
| `gp.c` | Gaussian Process Bayesian Optimization | [optimize.md](optimize.md) |
| `gfm.c` | General F-Measure optimization (per-label thresholds) | -- |
| `dataset.lua` | Data loaders for standard benchmarks | -- |

## Pipeline Patterns

### Dense Regression (Spectral + Ridge)

Scalar regression from mixed categorical/continuous features.

1. Load and binarize features. Merge categorical and continuous CSR.
2. `csr.standardize` for z-score normalization.
3. `spectral.encode` with CSR input -> fvec train_codes + encoder.
4. Encode validation/test splits via `encoder:encode`.
5. `optimize.ridge` with `targets` (dvec), `val_targets` for dense mode.
   Searches `lambda` via GP-BO, optimizing negated MAE.
6. `ridge_obj:regress` on each split, `eval.regression_accuracy` for
   MAE/NMAE.

Reference: `test/spec/santoku/learn/regress/housing-elm.lua`

### XMLC Pipeline (Spectral + Ridge + GFM)

End-to-end extreme multi-label classification. Embeds documents via
kernel spectral embedding, trains a ridge classifier to map embeddings
to label predictions.

**Tokenization and feature weighting:**

1. `csr.tokenize` with ngrams. Apply `csr.apply_bns` for BNS feature
   weighting using label structure.

**Spectral embedding:**

2. `spectral.encode` with CSR tokens/values -> fvec train_codes +
   encoder. RP-Cholesky selects landmarks.
3. Encode dev/test via `encoder:encode` after applying BNS scores.

**Optional ANN shortlisting:**

4. `ann.create` with sign-quantized codes for multi-index hashing.
5. `mih:neighborhoods_by_vecs` for approximate kNN retrieval.
6. `csr.label_union` to build per-query candidate label sets from
   neighbor label assignments.

**Ridge classifier (spectral codes -> labels):**

7. `optimize.ridge` with spectral codes, label CSR as ground truth,
   and searchable `lambda`, `propensity_a`, `propensity_b`.
8. `ridge.gram` eigendecomposes `X'X` once. Per-trial `ridge.create`
   computes W via fast dgemm path.
9. `ridge_obj:label(codes, n, k)` returns CSR predictions: offsets,
   labels, scores (top-k per sample sorted by score descending).
10. `eval.retrieval_ks` computes oracle F1 (per-sample optimal k).

**GFM thresholding:**

11. `gfm.create` with expected labels, `gfm_obj:fit` with ridge
    predictions to learn per-label score thresholds.
12. `gfm_obj:predict` produces per-sample k values for final
    label assignment.

Reference: `test/spec/santoku/learn/regress/eurlex-hvelm.lua`

## Supporting Patterns

### Feature Weighting

**`csr.apply_bns`**: Bi-Normal Separation scoring. Computes per-token
BNS scores from label structure, multiplies into CSR values in-place.

**`csr.apply_auc`**: AUC-probit scoring for continuous targets. Computes
per-token AUC against each target dimension, applies probit transform.

**`csr.standardize`**: Z-score standardization. Computes per-token
mean/variance, scales values to unit variance.
