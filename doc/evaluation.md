# Evaluation and Clustering

## Overview

`evaluator.c` provides metrics for classification, regression, ranking,
and retrieval; top-k feature extraction; and graph-constrained
agglomerative clustering with dendrogram traversal.

## Classification Accuracy

`class_accuracy(predicted, expected_offsets, expected_neighbors, n_samples, n_classes)`

Single-label classification. Computes per-class and macro-averaged
precision, recall, and F1. Ground truth extracted from CSR: first
neighbor per sample is the true class. TP/FP/FN via atomic increments
under OpenMP.

- `predicted`: ivec of predicted class indices.
- `expected_offsets`, `expected_neighbors`: CSR ground truth labels.
- `n_samples`, `n_classes`: counts.

Returns `{ precision, recall, f1, classes = { [c] = { precision, recall, f1 } } }`.

## Regression Accuracy

`regression_accuracy(predicted, expected)`

Absolute error statistics.

- `predicted`: fvec or dvec of predictions.
- `expected`: dvec or ivec of targets.

Returns `{ total, mean, min, max, std, nmae }` where
`nmae = mean / mean(expected)`.

## Retrieval with Optimal k

`retrieval_ks({table})`

Finds per-sample optimal k maximizing F1, or evaluates at supplied k
values.

### Input

- `pred_offsets` (ivec), `pred_neighbors` (ivec): CSR predictions
  (e.g. from `ridge_t:label`).
- `expected_offsets` (ivec), `expected_neighbors` (ivec): CSR ground
  truth.
- `ks` (ivec, optional): if provided, evaluate at these k values
  instead of searching for optimal k.

### Algorithm

When `ks` is not provided: for each sample, walks predictions from
position 1 to hood_size, tracking cumulative TP. Records the k with
highest F1 per sample. When `ks` is provided: evaluates at the given
k per sample.

### Return values

Returns 2 values:

1. `ks` (ivec or nil): per-sample optimal k values. Nil if input `ks`
   was provided.
2. Metrics table: `{ micro_precision, micro_recall, micro_f1,
   sample_precision, sample_recall, sample_f1 }`.

## Ranking Accuracy

`ranking_accuracy({table})`

Measures how well predicted representations preserve ground-truth
neighbor rankings.

### Input

- `codes` (cvec) or `raw_codes` (dvec): predicted representations.
- `ids` (ivec): sample IDs for the code matrix.
- `eval_ids` (ivec): IDs of evaluation queries.
- `eval_offsets`, `eval_neighbors`, `eval_weights`: CSR ground-truth
  neighbor structure with relevance weights.
- `n_dims`: number of bits (binary) or dimensions (continuous).
- `ranking`: `"ndcg"`, `"spearman"`, or `"pearson"` (default `"ndcg"`).

### Distance computation

Two modes:

| Mode | Input | Distance |
|---|---|---|
| `raw_codes` | dvec | cosine distance `1 - cos(a,b)` |
| `codes` | cvec | Hamming distance |

For each eval query, distances to its expected neighbors are computed.
Binary codes use bucket-based NDCG with tied-rank discount averaging.
Continuous codes are sorted by cosine distance and converted to
positional ranks.

### Metrics

- **NDCG**: Neighbors sorted by predicted distance. Ties share average
  discount. DCG sums `relevance / avg_discount`. IDCG sorts weights
  descending with standard `1/log2(i+2)` discount.
- **Spearman**: Rank correlation between expected weight ranks and
  retrieved distance ranks.
- **Pearson**: Linear correlation between expected weights and retrieved
  distances.

Returns `{ score, total_queries }` where score is the mean.

## Top-k Features

`topk_features({table})`

Extracts per-sample score-based features from ridge prediction scores.

- `offsets` (ivec): CSR offsets.
- `scores` (fvec): prediction scores sorted descending per sample.
- `k`: number of top predictions to featurize.

For each sample's top-k predictions, computes 3 features per position:
ratio to top score, gap to next score (normalized), and log ratio.

Returns `(fvec features, int n_dims)` where features is
`n_samples × (k × 3)`.

## Agglomerative Clustering

`cluster({table})`

Graph-constrained agglomerative clustering on binary codes with
complete-linkage distance and majority-vote centroids.

### Input

- `codes` (cvec): binary codes.
- `ids` (ivec): sample IDs.
- `offsets`, `neighbors`: CSR adjacency graph.
- `n_dims`: bits per code.
- `early_exit`: centroid distance threshold for heuristic skip.
- `expected_offsets`, `expected_neighbors`, `expected_weights`
  (optional): for AUC quality metric.

### Algorithm

1. Group samples with identical binary codes (FNV-1a hash + exact
   compare) into initial clusters.
2. Build complete-linkage edges between adjacent clusters. Push onto
   min-heap.
3. Repeat: pop minimum-distance edges, process at same distance level.
   Merge smaller cluster into larger (centroid merge via vote-count
   sum). Recompute distances to neighbors. Record dendrogram merges.
4. After each distance level: compute quality and optional AUC metrics.

### Centroid

Per-bit integer vote counts. Bit is 1 iff `votes[bit] >= 0`. Merging
sums vote arrays. O(1) centroid update per merge.

### Output

Returns table with:

| Field | Content |
|---|---|
| `offsets` | dendrogram offsets (ivec) |
| `merges` | dendrogram merges (pvec of absorbed/surviving pairs) |
| `n_steps` | number of merge steps |
| `ids` | input IDs (ivec) |
| `radius_curve` | quality at each step (dvec) |
| `n_clusters_curve` | active cluster count at each step (ivec) |
| `auc_curve` | AUC at each step (dvec, if expected data provided) |

## Dendrogram Cut

`dendro_cut(offsets, merges, step [, assignments])`

Materializes cluster assignments at a given dendrogram step.

Returns `(cluster_members, cluster_offsets, assignments)` in CSR format.

## Dendrogram Iterator

`dendro_each(offsets, merges [, all_merges])`

Returns a Lua iterator yielding `(step, ids, assignments)` per step.

- `all_merges = false` (default): one step per distance level.
- `all_merges = true`: finer granularity — merges at the same distance
  producing different-sized clusters are yielded as separate steps.
