# Spectral Graph Hashing

A framework for learning binary codes from graph structure via spectral
decomposition. Binary codes enable fast Hamming-based similarity search and
clustering, while landmark-based encoding extends retrieval to unseen queries.

## Overview

Spectral Graph Hashing learns binary embeddings through a three-tier
optimization pipeline:

1. **Adjacency**: Construct weighted k-NN graph from ranked features
2. **Spectral**: Extract eigenvectors from graph Laplacian
3. **Evaluation**: Score embeddings via retrieval or clustering metrics

Each tier is independently configurable, enabling systematic exploration of the
parameter space through adaptive multi-round optimization.

## Use Cases

- Fast similarity search via Hamming distance operations
- Clustering with hierarchical relationships
- Out-of-sample extension using observable features
- Interpretable binary representations preserving graph structure

## Background

Spectral hashing obtains binary embeddings where Hamming distances preserve
neighborhood structure. Self-Taught Hashing extends this to unseen data by
training binary classifiers to predict hash bits from features.

This framework builds on these foundations with:

- Hierarchical edge weighting via ranked feature similarity
- Multiple Laplacian variants (unnormalized, normalized, random walk)
- Flexible thresholding strategies (ITQ, Otsu, median, sign)
- Elbow-based dimension selection using statistical metrics
- Separate evaluation frameworks for retrieval and clustering

## Feature Extraction and Ranking

Features are partitioned into **ranks** representing similarity hierarchy
levels:

| Rank | Signal Strength | Examples |
|------|-----------------|----------|
| 0 | Strong (exact match) | Categories, tags, document labels |
| 1 | Moderate | Structured metadata, entity types |
| 2+ | Weak | Text tokens, n-grams |

Each feature has optional IDF-style weights within its rank. The inverted index
stores features with their ranks and weights for similarity computation.

## Hierarchical Edge Weighting

Graph topology is determined by k-NN search on observable features, with edge
weights incorporating metadata through ranked similarity functions.

Edge weights combine per-rank similarities with exponential decay:

    sim(i,j) = Σ[sim_r(i,j) * exp(-r * decay)] / Σ[exp(-r * decay)]

Where:
- sim_r(i,j) computes similarity using only features at rank r
- decay > 0 prioritizes lower ranks (categories matter most)
- decay < 0 prioritizes higher ranks (fine-grained features matter most)
- decay = 0 applies equal weighting across ranks

## Three-Tier Pipeline

### Tier 1: Adjacency Construction

Builds weighted k-NN graph from feature indices.

#### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| knn | int | - | Number of nearest neighbors |
| knn_alpha | float | 1.0 | Scaling for adaptive k modes |
| knn_mode | string | nil | Adaptive mode: cknn, sigma, or nil |
| knn_mutual | bool | false | Require bidirectional edges |
| knn_min | int | 0 | Minimum neighbors per node |
| knn_eps | float | 1.0 | Distance threshold |
| knn_cache | int | auto | Cache size for neighborhood queries |

#### Weight Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| weight_decay | float | 0.0 | Exponential decay for ranked features |
| weight_pooling | string | min | Asymmetric weight aggregation: min or max |
| weight_eps | float | 1e-8 | Numerical stability epsilon |

#### Similarity Metrics

| Metric | Description | Parameters |
|--------|-------------|------------|
| jaccard | intersection / union | - |
| overlap | intersection / min(A, B) | - |
| dice | 2 * intersection / (A + B) | - |
| tversky | Generalized asymmetric similarity | alpha, beta |

The cmp parameter selects the metric. For Tversky, alpha and beta control
asymmetry (alpha=beta=0.5 equals Dice; alpha=beta=1.0 equals Jaccard).

#### Adaptive k-NN Modes

**CKNN (Continuous k-NN)**: Adapts edge threshold based on estimated manifold
dimension using Levina-Bickel estimation. Edge exists if distance is less than
δ√(ρ_u · ρ_v), where ρ represents local density. Threshold scales inversely
with dataset size.

**Sigma Mode**: Computes Gaussian kernel bandwidth from k-th neighbor distance.
Edge weights use geometric mean of endpoint bandwidths:
w = exp(-d² / (2 · √(σ_i · σ_j)²))

#### Bridge Strategies

| Strategy | Description |
|----------|-------------|
| mst | Minimum spanning tree connects components via k-NN candidates |
| largest | Keep only largest connected component |
| none | Allow disconnected components |

#### Dual-Index Pattern

Separate indices can control topology and weights independently:

- knn_index determines graph structure (which nodes connect)
- weight_index determines edge weights (connection strength)

This enables building graphs where topology comes from one feature space (e.g.,
text similarity) while weights come from another (e.g., category overlap).

#### Adjacency Sources

Multiple edge sources can be combined:

**Bipartite Edges**: Connect samples to features they contain, creating a
bipartite graph structure useful for recommendation scenarios.

**Seed Edges**: Pre-computed edges providing prior knowledge or known
relationships.

**Category Anchors**: Sample anchor nodes per category and connect to their
k-nearest neighbors within that category. Parameters:
- category_anchors: Number of anchors per category
- category_knn: Neighbors per anchor
- category_knn_decay: Rank-based decay for anchor edges

**Random Pairs**: Add random edges for regularization or connectivity, useful
when graph is too sparse.

### Tier 2: Spectral Decomposition

Extracts eigenvectors from graph Laplacian using PRIMME solver.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_dims | int | - | Number of eigenvectors to extract |
| laplacian | string | unnormalized | Laplacian type |
| eps | float | 1e-8 | Convergence tolerance |
| threshold | function | - | Binarization function |
| method | string | jdqr | Eigensolver method |
| precondition | string | auto | Preconditioner type |
| block_size | int | 64 | PRIMME block size |

#### Laplacian Types

| Type | Formula | Use Case |
|------|---------|----------|
| unnormalized | L = D - W | Standard spectral clustering |
| normalized | L = I - D^(-1/2) W D^(-1/2) | Better scaling properties |
| random | L = I - D^(-1) W | Random walk perspective |

#### Eigensolver Methods

| Method | Description |
|--------|-------------|
| jdqr | Jacobi-Davidson QMR (default, robust) |
| lobpcg | Locally Optimal Block Preconditioned CG |
| jdqmr | Jacobi-Davidson QMR variant |
| gd | Gradient Descent with restart |

#### Preconditioners

| Type | Description | Best For |
|------|-------------|----------|
| diag | Diagonal (D^(-1)) | General purpose, fast |
| ic | Incomplete Cholesky | Ill-conditioned problems |
| poly | Chebyshev polynomial | Balance of speed/quality |
| off | None | Well-conditioned problems |

Default: diag for unnormalized Laplacian, off for normalized.

#### Output

Spectral decomposition returns:
- ids: Node identifiers
- raw_codes: Continuous embeddings (n_samples × n_dims)
- scale: Normalization factors per node
- eigenvalues: Corresponding eigenvalues (smallest first)

### Tier 3: Evaluation

Scores spectral embeddings via retrieval or clustering metrics.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| ranking | string | - | Ranking metric for retrieval |
| metric | string | - | Quality aggregation metric |
| elbow | string | - | Elbow detection method |
| elbow_alpha | float | varies | Elbow sensitivity parameter |
| select_metric | string | nil | Dimension selection metric |
| select_elbow | string | nil | Dimension selection elbow method |
| target | string | combined | Optimization target |
| knn | int | - | k for retrieval evaluation |

## Thresholding Methods

Multiple strategies convert continuous eigenvectors to binary codes:

### ITQ (Iterative Quantization)

Learns optimal rotation matrix R minimizing quantization error. Uses alternating
optimization between quantization (sign threshold) and rotation (SVD-based
update).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| iterations | int | 1000 | Maximum optimization iterations |
| tolerance | float | 1e-8 | Convergence threshold |

Best for: High-quality codes when computational cost acceptable.

### Otsu's Method

Finds optimal per-dimension thresholds maximizing between-class variance or
minimizing entropy. Optionally ranks dimensions by quality score.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| metric | string | variance | Optimization metric: variance or entropy |
| n_bins | int | 32 | Histogram granularity |
| minimize | bool | false | Invert optimization direction |

Returns codes with dimensions reordered by quality score.

Best for: Quality-aware dimension selection, feature ranking.

### Median Thresholding

Per-dimension median threshold producing balanced codes (~50% ones per
dimension).

Best for: Balanced output, general purpose, fast.

### Sign Thresholding

Simple zero-threshold binarization.

Best for: Pre-centered embeddings, baseline comparison.

## Dimension Selection

Optional post-processing selects optimal dimension subset using statistical
metrics and elbow detection.

### Selection Metrics

| Metric | Description |
|--------|-------------|
| entropy | Shannon entropy per dimension (lower = more discriminative) |
| variance | Statistical variance |
| skewness | Distribution asymmetry |
| bimodality | Bimodal coefficient |
| dip | Hartigan's dip test for multimodality |
| eigs | Eigenvalue magnitudes |

### Elbow Methods

| Method | Alpha Range | Description |
|--------|-------------|-------------|
| lmethod | - | L-method (linear fit corner detection) |
| max_gap | - | Largest gap between consecutive values |
| max_curvature | - | Maximum curvature point |
| otsu | - | Otsu's threshold on score distribution |
| plateau | 0.001-0.1 | Plateau detection with relative tolerance |
| first_gap | 1-10 | First gap exceeding alpha × median gap |
| kneedle | 0.1-10 (log) | Kneedle algorithm |

## Ground Truth Construction

Evaluation requires expected adjacency representing ideal similarity structure.

### For Retrieval Evaluation

Query nodes retrieve from a target set. Build ground truth by:
1. Create index over target nodes (e.g., categories/tags)
2. Query with source node features
3. Weight edges by target similarity

Typical pattern: documents query against category nodes, weighted by category
overlap.

### For Clustering Evaluation

All nodes are both queries and targets. Build ground truth by:
1. Create index over all nodes using supervision signal (e.g., tags)
2. Build k-NN adjacency within this space
3. Weight edges by supervision similarity

Typical pattern: document-to-document adjacency weighted by shared tags.

### Query-Target Separation

The knn_query_ids and knn_query_codes parameters enable asymmetric evaluation
where queries come from a different set than indexed nodes.

## Retrieval Evaluation

Evaluates how well binary codes preserve expected neighborhood structure.

### Process

1. For each query, retrieve neighbors by Hamming distance
2. Apply elbow detection to distance curve for cutoff
3. Compare retrieved set against expected neighbors
4. Compute ranking and quality scores

### Ranking Metrics

| Metric | Description |
|--------|-------------|
| ndcg | Normalized Discounted Cumulative Gain |
| spearman | Spearman rank correlation |
| pearson | Pearson correlation |

### Quality Metrics

| Metric | Description |
|--------|-------------|
| min | Minimum weight of matched items |
| mean | Average weight of matched items |
| variance | Variance ratio of weights |

### Combined Score

Harmonic mean of ranking score and quality:

    combined = 2 * score * quality / (score + quality)

Returns zero if either component is zero.

### Optimization Targets

- score: Optimize ranking metric only
- quality: Optimize quality metric only
- combined: Optimize harmonic mean of both

## Clustering Evaluation

Evaluates hierarchical clustering quality using dendrogram analysis.

### Process

1. Build agglomerative clustering on binary codes (complete linkage)
2. Evaluate each dendrogram cut via clustering accuracy
3. Apply elbow detection to quality curve
4. Return optimal cut point and cluster count

### Clustering Algorithm

- Initialize clusters from unique binary codes
- Merge clusters with minimum centroid Hamming distance
- Centroids computed via majority vote (ties favor 0)
- Continue until single cluster remains

### Clustering Accuracy

For each node, measures alignment between cluster membership and expected
neighbor weights:

- min: Minimum weight among cluster members in expected neighbors
- avg: Average weight among cluster members in expected neighbors

### Output

- quality: Best quality score at optimal cut
- n_clusters: Optimal cluster count
- best_step: Optimal dendrogram cut index
- quality_curve: Quality at each step
- n_clusters_curve: Cluster count at each step

## Retrieval vs Clustering Comparison

| Aspect | Retrieval | Clustering |
|--------|-----------|------------|
| Input | Retrieved neighbors by Hamming | Dendrogram from agglomerative clustering |
| Ground Truth | Expected neighbor weights | Expected neighbor weights |
| Optimization | Per-query cutoff via elbow | Global cut via elbow on quality curve |
| Metrics | NDCG, Spearman, Pearson | Min/avg weight of cluster members |
| Output | Score, quality, combined | Quality, n_clusters, best_step |
| Use Case | Similarity search tuning | Cluster discovery |

## Landmark-Based Encoding

Extends binary embeddings to unseen data via landmark aggregation.

### Process

1. Index landmark nodes with both observable features and binary codes
2. For new query: retrieve k-nearest landmarks by observable features
3. Aggregate landmark codes using selected mode (concat or frequency)
4. Use aggregated codes directly or train classifier for prediction

### Aggregation Modes

#### Concat Mode

Directly concatenates k landmark codes into a single feature vector:

- Output dimension: n_landmarks × n_hidden bits
- Position-sensitive: landmark order matters
- Direct representation of neighborhood
- Risk of overfitting: encodes exact landmark identities rather than aggregate patterns

Example: 4 landmarks × 32-bit codes = 128-bit feature vector

#### Frequency Mode

Computes bit frequency statistics across landmarks with multi-threshold encoding:

1. For each bit position, count frequency across k landmarks
2. Apply multiple threshold levels to each frequency
3. Create binary feature indicating "bit appears in ≥ threshold% of landmarks"

- Output dimension: n_hidden × n_thresholds bits
- Position-invariant: landmark order doesn't matter
- Robust to noise via consensus encoding
- Better generalization: aggregates patterns rather than memorizing landmarks
- Preferred for out-of-sample extension where train/test gap is a concern

Example: 32 bits × 7 thresholds = 224-bit feature vector

Thresholds typically span [1/8, 2/8, ..., 7/8], encoding gradient of agreement
from "bit rare in neighborhood" to "bit ubiquitous in neighborhood".

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mode | string | concat | Aggregation mode: "concat" or "frequency" |
| n_landmarks | int | 24 | Number of landmarks to retrieve |
| n_thresholds | int | 7 | Threshold levels for frequency mode |
| cmp | string | jaccard | Similarity metric for landmark retrieval |
| tversky_alpha | float | 0.5 | Tversky alpha parameter |
| tversky_beta | float | 0.5 | Tversky beta parameter |
| probe_radius | int | 2 | Search radius for ANN index |
| rank_filter | int | -1 | Limit to top-N ranks (-1 for all) |

### Output

Returns encode function and latent dimension:
- Concat: n_landmarks × n_hidden bits
- Frequency: n_hidden × n_thresholds bits

### Tsetlin Machine Integration

Optionally train a Tsetlin machine encoder to predict binary codes from
landmark features, enabling learned hash functions rather than direct
concatenation.

## Adaptive Optimization

Multi-round optimization with adaptive parameter sampling.

### Search Strategy

1. Sample parameters from configured ranges
2. Evaluate via three-tier pipeline
3. Track best configuration per round
4. Recenter samplers toward best parameters
5. Adapt jitter based on success rate

### Success Rate Adaptation (1/5-rule)

| Success Rate | Adaptation Factor | Behavior |
|--------------|-------------------|----------|
| > 20% | 1.0 | Maintain exploration |
| 5-20% | 0.85 | Moderate focus |
| < 5% | 0.5 | Aggressive convergence |

### Parameter Specification Types

**Fixed value**: Single value, no sampling.

**Range specification**: Center value with min/max bounds. Supports logarithmic
scale, integer rounding, and power-of-2 rounding.

**List specification**: Random selection from discrete options.

### Range Options

| Option | Type | Description |
|--------|------|-------------|
| def | number | Default/starting value |
| min | number | Lower bound |
| max | number | Upper bound |
| log | bool | Logarithmic scale |
| int | bool | Round to integer |
| pow2 | bool | Round to power of 2 |
| dev | float | Initial jitter (0-1) |

### Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| rounds | int | 1 | Optimization rounds with recentering |
| adjacency_samples | int | 1 | Adjacency configs per round |
| spectral_samples | int | 1 | Spectral configs per adjacency |
| eval_samples | int | 1 | Eval configs per spectral |

## Reference Pipeline

### Core Hash Learning

1. Build k-NN graph from observable features with ranked similarity
2. Compute hierarchical edge weights using exponential decay
3. Extract eigenvectors from chosen Laplacian variant
4. Apply threshold function (ITQ, Otsu, median, or sign) to produce binary codes
5. Optionally select dimension subset via statistical metrics and elbow detection
6. Index binary codes for fast retrieval

### Evaluation

7. Score retrieval quality via ranking metrics (NDCG, Spearman, Pearson)
8. Score clustering quality via dendrogram analysis with elbow detection

### Out-of-Sample Extension

9. Index landmark features and codes
10. Encode new queries via landmark aggregation
11. Optionally train Tsetlin machine encoder from landmark features to codes

## Module Reference

| Module | Purpose |
|--------|---------|
| inv | Inverted index with ranked features and IDF weights |
| ann | Approximate nearest neighbor index for binary codes |
| hbi | Hamming ball index for exact Hamming distance search |
| graph | Build k-NN/CkNN adjacency graphs with bridging |
| spectral | Laplacian eigenvector computation via PRIMME |
| itq | Binarization: ITQ rotation, Otsu, median, sign |
| hlth | Landmark-based hash encoding |
| eval | Retrieval and clustering evaluation |
| optimize | Hyperparameter search with adaptive sampling |
| simhash | LSH-based alternative to spectral embedding |

## Future Directions

**Supervision & Graph Construction:**
- Adaptive decay parameter learning from labeled data
- End-to-end optimization through embedding quality

**Decomposition & Selection:**
- Multi-view spectral decomposition for separate feature modalities
- Alternative bit selection strategies

**Hash Learning:**
- Attention-based landmark aggregation
- Graph neural network encoders
