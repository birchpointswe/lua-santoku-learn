# Evaluation, Bit Selection, and Agglomerative Clustering

## Overview

`evaluator.c` provides evaluation metrics for classification,
regression, encoding, and retrieval; greedy bit/dimension selection
optimizing ranking quality; and graph-constrained agglomerative
clustering with dendrogram output.

## Classification Accuracy

`class_accuracy(predicted, expected, n_samples, n_classes)` computes
per-class and macro-averaged precision, recall, and F1 for single-label
classification. TP/FP/FN counts use atomic increments under OpenMP
parallelization.

Returns `{ precision, recall, f1, classes = { [c] = { precision, recall,
f1 } } }`.

## Encoding Accuracy

`encoding_accuracy(predicted, expected, n_samples, n_dims)` compares two
binary code matrices bit by bit. Per-bit error rate (BER) is the fraction
of samples that disagree on that bit.

Returns `{ mean_hamming = 1 - mean_BER, ber_min, ber_max, ber_std,
bits = { [j] = BER_j } }`.

## Regression Accuracy

`regression_accuracy(predicted, expected)` computes absolute error
statistics over a dvec of predictions against dvec or ivec targets.

Returns `{ total, mean, min, max, std, nmae }` where `nmae = mean / mean(expected)`.

## Regression Per-Dimension

`regression_per_dim(predicted, expected, n_samples, n_dims)` computes
per-dimension regression statistics for multi-output models. For each
dimension independently:

- **MAE**: mean absolute error
- **Pearson r**: linear correlation between predicted and expected
- **Variance ratio**: `var(predicted) / var(expected)`

Parallelized across dimensions with `omp parallel for`.

Returns `{ mae = dvec, corr = dvec, var_ratio = dvec }` where each dvec
has `n_dims` elements.

## Retrieval Accuracy

`retrieval_accuracy({ hoods, hood_ids, expected_offsets, expected_neighbors })`
evaluates retrieval quality by comparing predicted neighborhoods against
ground truth. Accepts either `ann_hoods` or `inv_hoods`.

For each query sample: the predicted neighborhood (from hoods) is
intersected with the expected neighbor set (from CSR offsets/neighbors).
TP = size of intersection, predicted = hood size, expected = GT size.

Returns micro and macro precision, recall, and F1.

## Retrieval with Optimal k

`retrieval_ks(args)` finds per-sample optimal k and optionally computes
a global score threshold maximizing micro F1.

### Input modes

Three input modes:

| Mode | Fields | Source |
|---|---|---|
| `inv_hoods` | `hoods` (inv_hoods), `hood_ids` (ivec) | Inverted index neighborhoods |
| `ann_hoods` | `hoods` (ann_hoods), `hood_ids` (ivec) | ANN neighborhoods |
| CSR predictions | `pred_offsets` (ivec), `pred_neighbors` (ivec), optional `pred_scores` (dvec) | Ridge classifier output |

Ground truth is always `expected_offsets` (ivec) + `expected_neighbors`
(ivec) in CSR format.

### Oracle metrics (always computed)

For each sample, walks predictions from position 1 to hood_size,
tracking cumulative TP. Records the k with the highest F1 per sample.
Aggregates micro and macro precision/recall/F1 at each sample's
optimal k.

### Threshold metrics (when `pred_scores` provided)

Finds the global score threshold maximizing micro F1 across all samples.
Algorithm:

1. Build an `is_tp` bitmap during the oracle walk marking each
   prediction as true positive or false positive.
2. Construct two dvecs: `all_scores` (all prediction scores) and
   `tp_scores` (scores of true positives only).
3. Sort both descending via `tk_dvec_desc`.
4. Two-pointer group walk: process score groups (ties) from highest to
   lowest. At each score level, count how many total predictions and how
   many TPs have scores >= threshold. Compute micro F1. Track the best.
5. At the best threshold: parallel loop over samples computing
   per-sample precision/recall/F1 for predictions with score >=
   threshold.

### Return values

Returns 3 values:

1. `ks` (ivec): per-sample optimal k values
2. Oracle metrics table: `{ micro_precision, micro_recall, micro_f1,
   macro_precision, macro_recall, macro_f1 }`
3. Threshold metrics table (or nil if no `pred_scores`):
   `{ threshold, micro_precision, micro_recall, micro_f1,
   macro_precision, macro_recall, macro_f1 }`

## Ranking Accuracy

`ranking_accuracy({ codes|raw_codes|index|kernel_index, ids, eval_ids,
eval_offsets, eval_neighbors, eval_weights, n_dims, ranking })` measures
how well the predicted representation preserves ground-truth neighbor
rankings.

### Distance computation

Four input modes determine how pairwise distances are computed:

| Mode | Input | Distance |
|---|---|---|
| `kernel_index` | inv index | `inv:distance(a, b, decay, bw)` |
| `raw_codes` | dvec | cosine distance `1 - cos(a, b)` |
| `codes` | cvec | Hamming distance |
| `index` | ann index | Hamming distance via `ann:get` |

For each eval query, distances to its expected neighbors are computed and
sorted. For kernel and raw_codes modes, continuous distances are converted
to positional ranks before scoring.

### Ranking metrics

**NDCG** (`tk_csr_ndcg_distance`): Neighbors sorted by predicted
distance. Ties (same Hamming distance) share the average discount of
their rank span: `avg_discount = mean(log2(pos+2))` over tied positions.
DCG sums `relevance / avg_discount` per neighbor. IDCG sorts expected
weights descending and applies the standard `1/log2(i+2)` discount.

**Spearman** (`tk_csr_spearman_distance`): Expected weights are converted
to fractional ranks (ties get average rank). Retrieved distances provide
the other rank sequence. Pearson correlation on the rank pairs.

**Pearson** (`tk_csr_pearson_distance`): Linear correlation between
expected weights and retrieved Hamming distances for matched neighbors.

Returns `{ score, total_queries }` where score is the mean across queries.

## Entropy Statistics

`entropy_stats(codes, n_samples, n_dims)` computes per-bit binary
entropy (via `tk_cvec_bits_top_entropy` or `tk_ivec_bits_top_entropy`).

Returns `{ mean, min, max, std, bits = { [j] = entropy_j } }`.

## Bit/Dimension Selection

`optimize_bits({ codes|raw_codes|index, n_dims, target_dims, min_dims,
expected_ids, expected_offsets, expected_neighbors, expected_weights, ids,
each })` selects a subset of bits (binary) or dimensions (continuous) that
maximize mean NDCG against a ground-truth neighbor structure.

### Binary mode

`tm_optimize_bits_sfbs` operates on binary codes (cvec or ann index).
Precomputes XOR codes for all neighbor pairs. Non-discriminative bits
(constant across all pairs) are filtered out.

Iterates add/remove cycles:

**Add phase**: For each unselected bit, tentatively increment pairwise
distances where the XOR bit is 1, and compute mean NDCG via
`tk_incr_compute_ndcg`. Select the best bit if it improves the score (or
if `active.n < min_dims`). Commit by updating the running distance array.

**Remove phase**: For each selected bit, tentatively decrement distances
and compute NDCG via `tk_incr_compute_ndcg_sub`. If removing any bit
improves the score, remove it and return to add phase.

Terminates when neither adding nor removing improves the score.

NDCG computation uses bucket-based aggregation: neighbors are bucketed by
their current Hamming distance. Within each bucket, ties share the
average discount of their rank span. IDCG is computed per-node by sorting
neighbor weights descending.

### Continuous mode

`tm_optimize_bits_sfbs_cont` operates on continuous embeddings (dvec).
Distance is squared Euclidean in the selected subspace:
`d(a,b) = sum_{dim in selected} (a[dim] - b[dim])^2`.

Three-phase selection (add/remove/swap) identical to the SFBS pattern
elsewhere:

- Phase 0 (add): Score each unselected dimension by tentatively adding
  its squared differences to pairwise distances. Scoring uses
  `tk_rank_dcg` which computes per-neighbor rank by counting how many
  other neighbors have smaller distance, then applies weighted discount.
  Select best if gain > 1e-8 or `active.n < min_dims`.

- Phase 1 (remove): Score each selected dimension by tentatively
  subtracting. Remove if score improves. Otherwise tentatively remove
  the weakest for a swap attempt.

- Phase 2 (swap-add): Score unselected dimensions as replacements. Accept
  if score exceeds pre-swap score by > 1e-8. Otherwise restore and
  terminate.

All scoring is parallelized: per-thread local score arrays accumulated
per node, then reduced across threads.

Returns an ivec of selected bit/dimension indices (sorted ascending).

## Agglomerative Clustering

`cluster({ codes, ids, offsets, neighbors, n_dims, early_exit,
expected_offsets, expected_neighbors, expected_weights })` performs
graph-constrained agglomerative clustering on binary codes with
complete-linkage distance and majority-vote centroids.

### Initialization

1. **Identical-code grouping**: Nodes with identical binary codes are
   placed in the same initial cluster. Codes are hashed (FNV-1a on the
   raw bytes) into a hash map with chained collision resolution. Exact
   byte comparison (with tail-byte masking for non-byte-aligned bit
   counts) resolves collisions.

2. Each initial cluster gets:
   - A unique `cluster_id` starting at `2 * n_nodes + 1`.
   - A `members` ivec of node indices.
   - A `centroid` (`tk_centroid_t`) built by adding each member's code.
   - A `neighbor_ids` set (iuset) of adjacent cluster indices.

3. **Edge construction**: The input adjacency graph (CSR format:
   `ids`, `offsets`, `neighbors`) defines which clusters are adjacent.
   For each pair of adjacent clusters, the complete-linkage distance is
   computed and pushed onto a min-heap.

### Centroid

`tk_centroid_t` maintains per-bit integer vote counts. Adding a member
increments votes for 1-bits and decrements for 0-bits. The centroid code
byte is the majority vote: bit is 1 iff `votes[bit] >= 0`.

Merging two centroids sums their vote arrays and recomputes the code.
This gives O(1) centroid update per merge (no member re-scan).

### Complete-Linkage Distance

`tk_cluster_complete_linkage_distance` computes the maximum Hamming
distance across all member pairs between two clusters:

1. Check a pairwise distance cache (key = ordered pair of cluster IDs).
2. Compute centroid-to-centroid Hamming distance.
3. If `early_exit_threshold > 0` and centroid distance exceeds it, return
   the centroid distance as a lower bound (heuristic skip).
4. Otherwise enumerate all member pairs. For >100 total pairs, uses
   `omp parallel for reduction(max:...)`.
5. Cache the result.

### Main Loop

Repeat while the heap is non-empty and more than one cluster remains:

1. Pop the minimum-distance edge. Collect all edges at the same distance
   (same distance level).

2. Re-sort edges within the level by minimum degree of their endpoints
   (ascending). This makes merge order deterministic and prioritizes
   merging low-degree clusters first.

3. Process in sub-batches of equal minimum degree:
   - Skip edges where either cluster is inactive or already merged in
     this batch (tracked via `merged_this_batch` set).
   - The smaller cluster (by centroid size) is absorbed into the larger.
   - Record the merge as `(absorbed_cluster_id, surviving_cluster_id)` in
     `dendro_merges`.
   - Transfer members, merge centroids, union neighbor sets.
   - Remove the absorbed cluster from the surviving cluster's neighbor
     set. Update all of the absorbed cluster's neighbors to point to the
     surviving cluster.
   - Recompute complete-linkage distances from the surviving cluster to
     all its neighbors. Push new edges onto the heap.
   - Deactivate the absorbed cluster.

4. After each distance level with at least one merge: append a step
   boundary to `dendro_offsets`, compute quality and AUC metrics.

### Quality Metrics

**Quality** (`tk_cluster_compute_quality`): Mean Hamming similarity of
each member to its cluster's centroid code:
`mean(1 - hamming(member, centroid) / n_bits)`.

**AUC** (`tk_cluster_compute_auc`, optional -- requires expected
offsets/neighbors/weights): For each sample's expected neighbor list,
examines all pairs of neighbors where one is in the same cluster and the
other is not. AUC = fraction of such pairs where the same-cluster
neighbor has higher expected weight (concordance).

### Output

Returns a table with:

| Field | Content |
|---|---|
| `offsets` | Dendrogram offsets (ivec) |
| `merges` | Dendrogram merges (pvec of absorbed/surviving pairs) |
| `n_steps` | Number of merge steps |
| `ids` | Input IDs (ivec) |
| `radius_curve` | Quality at each step (dvec) |
| `n_clusters_curve` | Active cluster count at each step (ivec) |
| `auc_curve` | AUC at each step (dvec, if expected data provided) |

### Dendrogram Layout

`dendro_offsets` encodes both initial assignments and step boundaries:

    offsets[0..n_samples-1]  = initial cluster_id for each node
    offsets[n_samples]       = 0 (sentinel separating init from steps)
    offsets[n_samples+1..]   = index into dendro_merges where each step starts

`dendro_merges` stores `(absorbed_cluster_id, surviving_cluster_id)` pairs
in merge order. Steps correspond to distance levels: all merges at the
same complete-linkage distance are grouped into one step.

## Dendrogram Cut

`dendro_cut(offsets, merges, step [, assignments])` materializes cluster
assignments at a given step of the dendrogram.

1. Initialize assignments from `offsets[0..n_samples-1]` (initial
   cluster IDs).
2. Apply merges from steps 0 through `step-1`: build an
   `absorbed -> surviving` map.
3. For each sample, follow the merge chain to find its final cluster
   (chain limit 10,000 hops).
4. Remap final cluster IDs to dense `0..k-1`.

Returns `(cluster_members, cluster_offsets, assignments)` -- the members
are in CSR format indexed by cluster ID.

## Dendrogram Iterator

`dendro_each(offsets, merges [, all_merges])` returns a Lua iterator
function that yields `(step, ids, assignments)` per step.

**`all_merges = false`** (default): Each step corresponds to a distance
level as encoded in `dendro_offsets`. The iterator replays merges level
by level.

**`all_merges = true`**: Each step is a single merge or a batch of merges
at the same distance producing clusters of the same size. Within a
distance level, merges that produce different-sized clusters are yielded
as separate steps, giving finer granularity than distance-level steps.

The iterator maintains incremental state: an `absorbed_to_surviving` map
and per-sample raw assignments. At each step, new merges are applied
incrementally. Final assignments are produced by following merge chains
and remapping to dense IDs.
