# Architecture and Usage Patterns

## Module Overview

| Module | Role | Reference |
|---|---|---|
| `regressor.c` | Regression TM with sparse absorption | [regressor.md](regressor.md) |
| `optimize.lua` | Adaptive hyperparameter search | [optimize.md](optimize.md) |
| `spectral.c` | Nystrom spectral embedding | [pca.md](pca.md) |
| `quantizer.c` | SFBS binary quantization and thermometer encoding | [ann.md](ann.md) |
| `evaluator.c` | Ranking, retrieval, classification, regression metrics; bit selection; clustering | [evaluation.md](evaluation.md) |
| `inv.h` | Rank-weighted inverted index with cosine similarity kernel | [pca.md](pca.md) |
| `ann.h` | Multi-index hashing ANN for Hamming-distance search | [ann.md](ann.md) |
| `csr.c` | CSR/CSC utilities: bipartite graphs, negative sampling, symmetrization | — |
| `dataset.lua` | Data loaders for standard benchmarks | — |

## Pipeline Patterns

Five pipeline patterns with increasing complexity. Each successive
pattern incorporates the machinery of the preceding ones.

### Dense Classification

Single-label classification from pre-binarized features. No sparse mode,
no spectral embedding. All features fit in a single cvec input vector.

1. Load and binarize features → ivec of set-bit indices per sample.
2. `bits_select` to extract per-split subsets by sample ID.
3. `bits_to_cvec` to produce packed Boolean input.
4. `optimize.regressor` with `solutions` (ivec of class labels). The TM
   internally converts labels to per-class ±1 regression targets.
5. `search_metric` calls `tm:classify` + `eval.class_accuracy` on the
   validation split, optimizing macro F1.
6. Persist, reload, evaluate per-class P/R/F1.

Dense mode: `features` equals the input dimensionality. No `n_tokens`,
no CSC index, no absorption parameters.

Reference: `test/spec/santoku/tsetlin/regress/mnist.lua`

### Continuous Regression

Scalar or multi-output regression from continuous features. The SSL
regressor path is auto-selected when `targets` is a dvec and neither
`solutions` nor `codes` is provided.

1. Load continuous features and targets. Apply quantile thresholding
   (`n_thresholds` bins per feature) to produce binary features.
2. `bits_to_cvec` on the thresholded features.
3. `optimize.regressor` with `targets` (dvec). The optimizer detects
   dvec input and routes to `regressor.c` (SSL TM with 2 input bits per
   feature: `x > lower` and `x <= upper`, thresholds learned via
   Stochastic Searching on the Line).
4. `search_metric` calls `tm:regress` + `eval.regression_accuracy`,
   optimizing negated NMAE.
5. Persist, reload, report NMAE and absolute error statistics.

Reference: `test/spec/santoku/tsetlin/regress/housing.lua`

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

Reference: `test/spec/santoku/tsetlin/regress/imdb.lua` (binary
sentiment), `test/spec/santoku/tsetlin/regress/newsgroups.lua`
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

4. `index:neighborhoods(k, decay, bandwidth)` → all-vs-all kNN.
5. Convert to CSR, generate random pairs, weight from kernel, merge,
   symmetrize.

This weighted symmetric adjacency serves as ground truth for NDCG
evaluation throughout the pipeline.

**Spectral embedding:**

6. `optimize.spectral` with the inv index, evaluation adjacency, and
   optionally searchable `n_landmarks`, `n_dims`, `decay`, `bandwidth`.
7. Returns `raw_codes` (dvec, n × d), `ids` (ivec), and an encoder
   for out-of-sample projection.

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

Reference: `test/spec/santoku/tsetlin/regress/newsgroups_embedding.lua`

### Full XMLC Pipeline

End-to-end extreme multi-label classification. Embeds documents and
labels jointly via spectral decomposition of a bipartite graph kernel,
trains a sparse TM to predict document coordinates from bag-of-words,
then binary-quantizes for Hamming-distance ANN retrieval against label
codes.

**Bipartite graph construction** (see [below](#bipartite-graph-construction)):

1. Create a joint node space: documents as nodes 0..N-1, labels as
   nodes N..N+L-1.
2. `bits_bipartite("adjacency")` on the label assignment matrix →
   per-node feature sets encoding connection patterns.
3. IDF weighting via `bits_top_df`. Assign 2-rank structure: label-side
   features at rank 0, document-side features at rank 1. With `decay >
   0`, label-side features contribute more to similarity.
4. Build three inv indexes from the same features: `graph_index`
   (all nodes), `labels_index` (label nodes only), `docs_index`
   (doc nodes only).

**Hard negative index:**

5. `bits_bipartite("inherit")` creates a second feature set where nodes
   inherit neighbors' features. Build a separate inv index on inherited
   features for hard negative mining during SFBS quantization.

**Evaluation adjacency:**

6. `graph_index:neighborhoods(k, decay, bandwidth)` → kNN from the
   bipartite graph kernel over all nodes.
7. Merge with random pairs, symmetrize.

**Spectral embedding (asymmetric mode):**

8. `optimize.spectral` with `landmarks_index = labels_index`. RPCholesky
   selects landmarks from the label population. The eigenspace is defined
   by the label-label kernel. Documents project into this space via
   cross-similarity to landmarks.
9. Both documents and labels receive d-dimensional coordinates in the
   same space. The coordinate system is defined by label structure;
   document positions reflect relationships to labels.

**First SFBS quantization (pre-regressor):**

10. `csr.bipartite_neg` constructs doc→label evaluation edges: GT label
    edges plus hard negatives from the inherited-features index.
11. `quantizer.create` on spectral coordinates with the bipartite
    evaluation adjacency. Selects bits maximizing NDCG for
    distinguishing GT labels from hard negatives.
12. Encode label spectral coordinates → binary label codes.
13. `encoder:used_dims()` identifies referenced spectral dimensions.
    `mtx_select` trims all embedding matrices to those dimensions.

This first quantization serves two purposes: producing binary label
codes for the ANN index, and identifying the relevant spectral subspace
to reduce the regressor's output dimensionality.

**Feature selection and TM regression:**

14. Tokenize documents (ngrams, character grams).
15. `bits_top_reg_f` with trimmed spectral coordinates as targets.
16. Build CSC index, construct per-class F-score rankings.
17. `optimize.regressor` with `targets` (trimmed spectral coordinates)
    in sparse mode.

**Second SFBS quantization (post-regressor):**

18. Predict raw coordinates for training documents.
19. Concatenate predicted doc codes + spectral label codes into a single
    embedding matrix with corresponding IDs.
20. `quantizer.create` on the combined matrix with the same bipartite
    evaluation adjacency. This second pass operates on the TM's actual
    output distribution rather than the oracle spectral coordinates,
    adapting bit selection to the regressor's approximation quality.
21. Re-encode label codes through the post-regressor encoder.

**Retrieval and evaluation:**

22. Build a labels-only ANN index on the final binary label codes.
23. For each split: predict raw → encode to binary → retrieve from
    label ANN via `neighborhoods_by_vecs`.
24. `retrieval_ks` finds per-sample optimal k and computes micro/macro
    P/R/F1.
25. `ranking_accuracy` with both `raw_codes` (cosine) and `codes`
    (Hamming) modes against bipartite evaluation adjacency.

At inference for a new document: tokenize → sparse TM regress →
SFBS encode → Hamming ANN lookup against label index → retrieve
label IDs.

Reference: `test/spec/santoku/tsetlin/regress/eurlex.lua`

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

The kNN + random pairs pattern appears in every embedding pipeline:

1. Compute kNN neighborhoods from an inv kernel.
2. `hoods:to_csr(uids)` → CSR adjacency with kernel-derived weights.
3. `csr.random_pairs(uids, n_per_node)` → uniform random node pairs.
4. `csr.weight_from_index(uids, off, nbr, w, index, decay, bw)` →
   fill random-pair weights from the kernel.
5. `csr.merge(off, nbr, w, rp_off, rp_nbr, rp_w)` → in-place merge.
6. `csr.symmetrize(off, nbr, w, n)` → add reverse edges, deduplicate,
   sort.

Random pairs prevent evaluation metrics from being blind to
relationships between distant nodes that kNN retrieval would never
surface. Kernel-weighting ensures distant pairs carry appropriately low
weights rather than binary presence/absence.

For XMLC evaluation of specific relationships (e.g., doc→label ranking
quality), `csr.bipartite_neg` constructs directed evaluation edges:
GT bipartite edges plus a specified number of random or hard negatives
per node. This is used for SFBS quantization evaluation and for
doc→label ranking accuracy measurement.

### Bipartite Graph Construction

For XMLC, documents and labels are joint nodes in a single graph:

- `bits_bipartite("adjacency")`: Edges from the label assignment matrix
  become shared features. Each node's feature set encodes its connection
  pattern.
- `bits_bipartite("inherit", source, n_source)`: Nodes inherit features
  from bipartite neighbors. Documents acquire label features, labels
  acquire document features. Creates a richer similarity signal for hard
  negative mining.
- Two-rank inv index: one rank for document-side features, another for
  label-side features. With `decay > 0`, the rank weighting controls
  relative contribution of each node type to similarity.
- Separate per-type indexes (docs-only, labels-only) enable directional
  queries: a document's features queried against the labels-only index
  retrieves candidate labels without document-document interference.

### Asymmetric Spectral Embedding

When `landmarks_index` differs from `index` in `optimize.spectral`,
RPCholesky selects landmarks from the landmark population (e.g., labels)
and builds the eigenspace from that population's kernel. Documents
project into this space via cross-similarity to landmarks. Both node
types receive coordinates in the same d-dimensional space.

For XMLC: the label population is typically much smaller than the
document population (e.g., 4K labels vs. 45K documents). Landmark
selection from labels produces a coordinate system where label structure
dominates. Documents are positioned relative to labels, which is the
geometry needed for retrieval.

### Two-Round SFBS Quantization

The XMLC pipeline applies SFBS quantization twice:

**Pre-regressor** (on spectral coordinates): (1) produces binary label
codes for the ANN index, (2) identifies which spectral dimensions the
selected bits reference, enabling dimension trimming before regression.

**Post-regressor** (on TM-predicted coordinates + spectral label codes):
Adapts the final bit selection to the regressor's actual output
distribution, which may differ from the spectral oracle due to
approximation error.

Both rounds use the same bipartite doc→label evaluation adjacency
(GT + hard negatives). SFBS optimizes NDCG for placing GT labels closer
in Hamming distance than negatives for each document.
