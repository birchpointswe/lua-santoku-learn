# Architecture and Usage Patterns

## Module Overview

| Module | Role | Reference |
|---|---|---|
| `regressor.c` | Regression TM with sparse absorption | [regressor.md](regressor.md) |
| `ridge.c` | Ridge regression classifier with propensity weighting | -- |
| `optimize.lua` | GP-BO hyperparameter search for TM, spectral, and ridge | [optimize.md](optimize.md) |
| `spectral.c` | Nystrom spectral embedding | [pca.md](pca.md) |
| `quantizer.c` | Iterative quantization and thermometer encoding | [ann.md](ann.md) |
| `evaluator.c` | Ranking, retrieval, classification, regression metrics; bit selection; clustering | [evaluation.md](evaluation.md) |
| `inv.h` | Rank-weighted inverted index with cosine similarity kernel | [pca.md](pca.md) |
| `ann.h` | Multi-index hashing ANN for Hamming-distance search | [ann.md](ann.md) |
| `csr.c` | CSR/CSC utilities: bipartite graphs, negative sampling, symmetrization | -- |
| `gp.c` | Gaussian Process Bayesian Optimization | [optimize.md](optimize.md) |
| `dataset.lua` | Data loaders for standard benchmarks | -- |

## Pipeline Patterns

### Dense Classification

Single-label classification from pre-binarized features. No sparse mode,
no spectral embedding. All features fit in a single cvec input vector.

1. Load and binarize features -> ivec of set-bit indices per sample.
2. `bits_select` to extract per-split subsets by sample ID.
3. `bits_to_cvec` to produce packed Boolean input.
4. `optimize.regressor` with `solutions` (ivec of class labels). The TM
   internally converts labels to per-class +1/-1 regression targets.
5. `search_metric` calls `tm:classify` + `eval.class_accuracy` on the
   validation split, optimizing macro F1.
6. Persist, reload, evaluate per-class P/R/F1.

Dense mode: `features` equals the input dimensionality. No `n_tokens`,
no CSC index, no absorption parameters.

Reference: `test/spec/santoku/learn/regress/mnist.lua`

### Continuous Regression

Scalar or multi-output regression from continuous features via
thermometer encoding.

1. Load continuous features and targets. Apply thermometer encoding
   (`quantizer.create` with `mode="thermometer"`, `n_bins` bins per
   feature) to produce dense binary features.
2. `bits_to_cvec` on the thermometer-encoded features.
3. `optimize.regressor` with `targets` (dvec). The optimizer detects
   dvec input and routes to `regressor.c`.
4. `search_metric` calls `tm:regress` + `eval.regression_accuracy`,
   optimizing negated NMAE.
5. Persist, reload, report NMAE and absolute error statistics.

Reference: `test/spec/santoku/learn/regress/housing.lua`

### Sparse Text Classification

Text classification with vocabularies too large for dense representation.
Per-class token mapping with absorption-driven feature streaming.

**Tokenization and feature selection:**

1. Build tokenizer (ngrams, character grams, skips). Train on the
   training corpus. Tokenize all splits.
2. Chi-squared feature selection via `bits_top_chi2(solutions, n,
   n_tokens, n_classes, per_k, union_k, pool)`. Returns five values:
   `union_ids`, `union_scores`, `class_offsets`, `class_feat_ids`,
   `class_scores`. The per-class outputs are token IDs sorted by
   chi-squared score descending within each class segment.
3. `bits_select(union_ids)` remaps all token ivecs to the reduced
   vocabulary. `csr.bits_select(class_offsets, class_feat_ids,
   union_ids)` remaps the per-class rankings to match.
4. `csr.to_csc(tokens, n_samples, n_tokens)` builds the inverted
   token-to-sample index for densification during training.

**Ranking construction (classification pattern):**

- `absorb_ranking = class_feat_ids`,
  `absorb_ranking_offsets = class_offsets`: Per-class chi-squared
  rankings. Each class initializes its `features` slots from the top of
  its chi-squared segment.
- `absorb_ranking_global`: Global streaming pool for absorption
  replacement. Typically `ivec.create(n_tokens):fill_indices()` (simple
  vocabulary order) or a global chi-squared ordering.

**Training:**

5. `optimize.regressor` with `solutions` (ivec labels), `n_tokens`,
   `features` (slots per class), `csc_offsets`, `csc_indices`, and all
   three ranking parameters.
6. `search_metric` calls `tm:classify` with table input
   `{ tokens = ..., n_samples = ... }` + `eval.class_accuracy`.
7. Final evaluation with per-class P/R/F1 breakdown.

The `features` parameter (slots per class) is typically much smaller
than `n_tokens`. E.g., 4096 slots from a 65K vocabulary. Absorption
cycles tokens through each class's slot mapping during training,
guided by the rankings.

Token-based inference at test time: pass `{ tokens, n_samples }` instead
of a pre-densified cvec. The TM builds a temporary dense buffer per
inference call, mapping each sample's tokens through each class's
current mapping.

Reference: `test/spec/santoku/learn/regress/imdb.lua` (binary
sentiment), `test/spec/santoku/learn/regress/newsgroups.lua`
(20-class)

### Spectral Embedding from Text

Learn continuous embeddings that preserve graph-kernel similarity, then
train a sparse TM to predict those embeddings from bag-of-words. Used
when the goal is a learned representation rather than direct
classification.

**Building the kernel index:**

1. Tokenize text. Apply BNS (Bi-Normal Separation) feature selection via
   `bits_top_bns`.
2. Build an inverted index with BNS scores as feature weights:
   `inv.create({ features = bns_scores })`.
3. The inv kernel computes cosine similarity between documents weighted
   by feature importance.

**Evaluation adjacency construction** (see [below](#evaluation-adjacency-construction)):

4. `index:neighborhoods(k, decay)` -> all-vs-all kNN.
5. Convert to CSR, generate random pairs, weight from kernel, merge,
   symmetrize.

This weighted symmetric adjacency is the ground truth for NDCG
evaluation throughout the pipeline.

**Spectral embedding:**

6. `optimize.spectral` with the inv index, evaluation adjacency, and
   optionally searchable `n_landmarks`, `n_dims`, `decay`.
7. Returns model with `raw_codes` (dvec, n * d), `ids` (ivec),
   `encoder`, and `eigenvalues` (dvec, descending order).

**Feature selection (regression pattern):**

8. Extract the training subset of spectral coordinates via
   `dvec:mtx_extend`.
9. `bits_top_reg_f(raw_codes, n, n_tokens, n_dims, per_k)`: per-
   dimension F-statistic rankings. For each spectral dimension, tokens
   are ranked by their predictive F-statistic. Returns per-class offsets
   + feature IDs (same 5-value pattern as chi-squared but driven by
   continuous targets).
10. `bits_select`, build CSC, construct rankings.

Unlike classification, the regression pattern uses per-class F-score
rankings for `absorb_ranking` (+ offsets). `absorb_ranking_global` is
typically `fill_indices` (vocabulary order).

**Training and evaluation:**

11. `optimize.regressor` with `targets` (dvec spectral coordinates) in
    sparse mode.
12. Predict embeddings on all splits. Evaluate via `ranking_accuracy`
    with `raw_codes` mode: cosine distance between predicted embeddings
    vs. ground-truth neighbor weights from the evaluation adjacency.

Reference: `test/spec/santoku/learn/regress/newsgroups_embedding.lua`

### Full XMLC Pipeline

End-to-end extreme multi-label classification. Embeds documents via
spectral decomposition of a label similarity kernel, trains a sparse TM
to predict document embeddings from bag-of-words, then uses a ridge
regression classifier to map embeddings to label predictions.

**Tokenization and feature selection:**

1. Tokenize text (ngrams, character grams). Apply BNS feature selection
   via `bits_top_bns`.

**Label similarity index:**

2. Build label CSR from training label assignments via `bits_to_csr`.
3. Build a doc-doc inverted index using label co-occurrence as features.
   Each label becomes a feature with IDF weight
   `log(n_docs / doc_count_per_label)`. Single-rank structure (all
   labels at rank 0).
4. `inv.create({ features = label_idf, ranks = label_ranks, n_ranks = 1 })`
   followed by `index:add(solutions, ids)` indexes all training
   documents.

**Spectral embedding:**

5. `optimize.spectral` with the label index. RPCholesky selects
   landmarks from the document population. Documents receive
   d-dimensional coordinates reflecting label structure.
6. Extract training spectral codes via `dvec:mtx_extend`. The model
   also returns `eigenvalues` (dvec) used for per-output hyperparameter
   modulation.

**Feature selection for regressor (F-score):**

7. `bits_top_reg_f` with spectral codes as continuous targets. Per-dim
   F-score rankings for absorption initialization.
8. `bits_select` + `csr.bits_select` to remap vocabulary and rankings.
9. Build CSC index for sparse TM training.

**TM regression (text -> spectral codes):**

10. `optimize.regressor` with `targets` (dvec spectral codes),
    `output_weights` (eigenvalues for per-dim alpha modulation),
    `n_tokens`, `features` (searchable), and absorption parameters.
11. GP-BO searches over `features`, `clauses`, `specificity`,
    `clause_tolerance_fraction`, `target_fraction`, `alpha_*` params,
    and absorption parameters.
12. `search_metric` calls `tm:regress` + `eval.regression_accuracy`,
    optimizing negated MAE.
13. Final batched training with early stopping on the full dataset.

**Ridge classifier (spectral codes -> labels):**

14. `optimize.ridge` with spectral codes (or TM-predicted codes),
    label CSR as ground truth, and searchable `lambda`, `propensity_a`,
    `propensity_b`.
15. `ridge.precompute` eigendecomposes `X'X` once. Per-trial
    `ridge.create({gram=...})` computes W via fast dgemm path.
16. `r:encode(codes, n, k)` returns CSR predictions: `offsets`, `labels`,
    `scores` (top-k labels per sample sorted by score descending).
17. `evaluator.retrieval_ks` with `pred_offsets`, `pred_neighbors`,
    `pred_scores` computes oracle F1 (per-sample optimal k) and
    threshold-based F1 (global score cutoff maximizing micro F1).
18. GP-BO optimizes threshold macro F1 over lambda and propensity.

**Evaluation:**

19. Run ridge on both spectral codes (ceiling) and TM-predicted codes
    (pipeline) to quantify the TM regression bottleneck.
20. Report per-dimension regression analysis: MAE, Pearson r, variance
    ratio by eigenvalue band.
21. Comparison table: micro/macro F1, oracle micro/macro F1, threshold,
    time for each approach.

At inference for a new document: tokenize -> BNS select -> sparse TM
regress -> ridge encode -> threshold score cutoff -> label IDs.

Reference: `test/spec/santoku/learn/regress/eurlex.lua`

## Supporting Patterns

### Feature Selection

Two feature ranking functions serve different pipeline stages:

**`bits_top_chi2`** (classification): Chi-squared independence test
between each token and each class label. Returns per-class ranked
feature lists (sorted by chi-squared score descending) and their union.
The `pool` parameter (`"sum"`, `"max"`) controls how per-class scores
combine into the union ranking.

**`bits_top_reg_f`** (regression): F-statistic regression test between
each token and each continuous target dimension. Same 5-value return
pattern.

Both return: `union_ids` (ivec), `union_scores` (dvec),
`class_offsets` (ivec), `class_feat_ids` (ivec), `class_scores` (dvec).
The per-class outputs (`class_feat_ids` + `class_offsets`) feed directly
into `absorb_ranking` + `absorb_ranking_offsets`. The union output feeds
into `bits_select` for vocabulary reduction.

**`bits_top_bns`**: Bi-Normal Separation score per token per class.
Used for feature weighting (inv index construction) rather than sparse
absorption rankings. Same return pattern.

### Ranking Architecture

The ranking system interfaces feature selection with sparse absorption:

**`absorb_ranking` + `absorb_ranking_offsets`**: Per-class initialization
priority. Segment `[offsets[c], offsets[c+1])` of the concatenated array
lists feature IDs for class `c` in priority order. Each class fills its
`features` slots from the top of its segment.

**`absorb_ranking_global`**: Streaming pool for absorption replacement.
A single ordered list of feature IDs shared across all classes. When
absorption evicts a token from a class, the per-class cursor walks this
list to find the next unmapped token.

When offsets are absent, both rankings alias the same array.

Classification pattern: per-class chi-squared for initialization,
vocabulary-order indices (or global chi-squared) for absorption.

Regression pattern: per-class F-score for initialization,
vocabulary-order indices for absorption.

### Evaluation Adjacency Construction

The kNN + random pairs pattern appears in embedding pipelines that
evaluate with NDCG ranking accuracy:

1. Compute kNN neighborhoods from an inv kernel.
2. `hoods:to_csr(uids)` -> CSR adjacency with kernel-derived weights.
3. `csr.random_pairs(uids, n_per_node)` -> uniform random node pairs.
4. `csr.weight_from_index(uids, off, nbr, w, index, decay, bw)` ->
   fill random-pair weights from the kernel.
5. `csr.merge(off, nbr, w, rp_off, rp_nbr, rp_w)` -> in-place merge.
6. `csr.symmetrize(off, nbr, w, n)` -> add reverse edges, deduplicate,
   sort.

Random pairs prevent evaluation metrics from being blind to
relationships between distant nodes that kNN retrieval would never
surface. Kernel-weighting ensures distant pairs carry appropriately low
weights rather than binary presence/absence.
