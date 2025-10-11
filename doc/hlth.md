# Hierarchical Landmark-Taught Hashing (HLTH)

An approach to supervised spectral hashing that encodes hierarchical similarity
through edge weighting on a k-NN graph built from observable features. This is
achieved through Laplacian eigenvector oversampling, bit subset selection
maximizing rank correlation between edge-weight and Hamming-distance rankings of
neighbors from the graph adjacency, and landmark-based hash learning for
out-of-sample extension.

## Use Cases

HLTH targets applications requiring fast similarity search and clustering over
data with hierarchical relationships. Binary codes enable efficient
Hamming-based operations on in-sample data, while landmark-based hash learning
extends retrieval to unseen queries using observable features (e.g., text
queries against document embeddings).

## Background

[Spectral hashing][1] provides an efficient method to obtain binary embeddings
for graph nodes where Hamming distances preserve neighborhood structure.
[Self-Taught Hashing (STH)][2] extends this to unseen data by training binary
classifiers to predict hash bits from node features, using in-sample binary
codes as targets.

## Hierarchical Edge-Weight Supervision

Supervision is encoded through hierarchical edge weighting on a k-NN graph.
Graph topology is determined by k-NN search on observable features, but edge
weights incorporate metadata through a multi-level similarity function.

The feature space is partitioned into ranked subspaces, each representing a
level of the similarity hierarchy. Lower ranks encode stronger similarity
signals (e.g., exact matches or high-level groupings), while higher ranks encode
weaker signals (e.g., general feature overlap).

Edge weights between nodes i and j are computed as:

w(i, j) = Σ[sim_r(i, j) * exp(-r * decay)] / Σ[exp(-r * decay)]

where `sim_r(i, j)` computes similarity between nodes considering only features
at rank `r`. Exponential decay biases the spectral decomposition toward
hierarchy-aligned partitions, transforming observable k-NN proximity into a
structure encoding globally consistent hierarchical rankings.

Spectral decomposition provides a pool of binary graph cuts progressing from
coarse to fine-grained partitions of the weighted structure.

## Eigenvector Oversampling & Subset Selection

The weighted adjacency matrix encodes hierarchical edge weights and is retained
for bit selection and evaluation. The unnormalized Laplacian L = D - W is
computed for spectral decomposition, where the smallest-k eigenvectors are
extracted with k exceeding the estimated hash length, creating an oversampled
pool of candidate vectors.

Each eigenvector is median-thresholded to produce a binary feature, then refined
via coordinate descent: bit values are iteratively flipped when doing so
increases weighted neighbor agreement.

Bit subset selection identifies which bits best preserve neighborhood rankings.
A greedy search process iteratively adds, removes, and swaps bits to maximize
Spearman rank correlation between edge-weight and Hamming-distance rankings of
graph neighbors, automatically eliminating redundant combinations.

## Landmark-Based Hash Learning

Landmark-based hash learning extends binary embeddings to unseen data. Landmarks
are in-sample nodes retaining both observable features and binary codes. Rather
than learning direct mappings from features to bits, binary classifiers learn
from concatenated landmark codes: unseen samples retrieve k-nearest landmark
neighbors by observable features, concatenate their codes, and predict target
bits from this aggregated representation. This approach naturally handles
hierarchical structure since landmark codes encode hierarchy through the
spectral decomposition process.

## Reference Pipeline

### Core Hash Learning

1. Build k-NN graph from observable features
2. Compute hierarchical edge weights using ranked similarity with exponential
   decay
3. Extract oversampled eigenvectors from unnormalized Laplacian L = D - W
4. Median-threshold eigenvectors and refine via coordinate descent bit-flipping
5. Select bit subset via SFFS starting from empty set, maximizing Spearman rank
   correlation
6. Index landmark observable features for efficient retrieval
7. Train binary classifiers from concatenated landmark codes to target bits

### Evaluation & Calibration

8. Scan Hamming distance thresholds to find optimal margin via rank-biserial
   correlation
9. Perform agglomerative clustering on binary codes and evaluate dendrogram cuts
   via rank-biserial correlation

Rank-biserial maxima identify high-quality partitions, but plateaus are common
and often more informative for practical use. The L-method effectively
identifies stable threshold ranges where performance remains consistent.

## Centroid-Linkage Agglomerative Clustering

Agglomerative clustering operates directly on binary codes using Hamming
distance. Cluster centroids are computed via majority vote across bit positions
(ties favor 0).

1. Initialize clusters from unique binary codes (samples with identical codes
   share a cluster)
2. Find the pair of clusters with minimum centroid Hamming distance
3. Merge the pair and recompute the centroid via majority vote
4. Repeat until a single cluster remains

The resulting dendrogram has merge heights corresponding to centroid Hamming
distances at merge time.

## Clustering and Margin Evaluation

Dendrogram cuts and Hamming distance thresholds partition samples for clustering
and retrieval use cases, respectively. Each requires evaluation to determine the
best cut or threshold for downstream usage.

For dendrogram cuts, rank-biserial correlation measures association between
cluster membership and edge weight rankings from the graph adjacency. Higher
values indicate within-cluster edges rank systematically higher than
between-cluster edges.

For Hamming margins, each threshold partitions adjacency pairs into similar
(within threshold) and dissimilar (outside threshold). Rank-biserial correlation
evaluates alignment with edge weight rankings, identifying optimal thresholds
for retrieval.

## Future Directions

Supervision & Graph Construction:
- Adaptive decay parameter learning from labeled data
- End-to-end optimization of graph construction through embedding quality

Decomposition & Selection:
- Normalized Laplacian variants for stability-focused applications
- Multi-view spectral decomposition for separate feature modalities
- Alternative metrics for bit selection, retrieval, and clustering evaluations

Hash Learning:
- Attention-based landmark aggregation replacing concatenation
- Graph neural network encoders incorporating hierarchical structure

## References

[1]: https://people.csail.mit.edu/torralba/publications/spectralhashing.pdf
[2]: https://arxiv.org/pdf/1004.5370
