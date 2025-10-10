# Hierarchical Landmark-Taught Hashing (HLTH)

An approach to supervised spectral hashing that encodes hierarchical similarity
relationships through edge weighting rather than topology modification, combined
with Spearman rank correlation-based optimization to select optimal bit subsets,
and landmark-based hash function learning for out-of-sample extension.

## Use Cases

This pipeline addresses scenarios requiring efficient binary embedding-based
retrieval or clustering with hierarchical levels of similarity. The in-sample
binary codes enable fast similarity search and clustering directly via Hamming
distance operations. The landmark-based hash function learning extends this
capability to unseen data, allowing users to query the embedding space using
visible features (e.g., text queries against a document corpus).

The approach is particularly suited for domains with multiple levels of
similarity relationships. From a representation learning perspective, the pipeline
projects samples into a binary coordinate space where Hamming distances preserve
hierarchical relationships encoded through edge weighting.

## Background

[Spectral hashing](https://people.csail.mit.edu/torralba/publications/spectralhashing.pdf)
provides an efficient method to obtain binary embeddings for nodes in a graph
where Hamming distances among embeddings approximate graph distances. The process
of generating embeddings for unseen nodes is called out-of-sample extension. One
method for enabling out-of-sample extension described in
[Self-Taught Hashing (STH)](https://arxiv.org/pdf/1004.5370) involves training k
binary classifiers to predict individual bits from arbitrary node features, using
the binary codes obtained for in-sample nodes as classifier targets.

## Hierarchical Edge-Weight Supervision

Rather than modifying graph topology with virtual nodes, supervision is encoded
through a hierarchical edge weighting scheme that preserves the natural sparse
k-NN structure while encoding multiple levels of similarity. The graph topology
is determined by k-NN search on visible features (e.g., text), but edge weights
are computed using a multi-level similarity function that incorporates metadata.

The feature space is partitioned into ranked subspaces, each representing a
different level of the similarity hierarchy. Lower ranks encode stronger
similarity signals (e.g., exact matches or high-level groupings), while higher
ranks encode weaker signals (e.g., general feature overlap).

Edge weights between nodes i and j are computed as:

    w(i, j) = Σ[sim_r(i, j) * exp(-r * decay)] / Σ[exp(-r * decay)]

where `sim_r(i, j)` computes similarity between nodes considering only features
at rank `r`. This exponential decay ensures that metadata overlap (lower ranks)
contributes more strongly to edge weights than visible feature overlap alone.
Critically, this weighting is applied AFTER topology discovery via k-NN,
separating the concerns of graph sparsity (topology) from similarity encoding
(weights).

This creates a graph where spectral decomposition naturally discovers
hierarchical structure: the first few eigenvectors capture broad clusters
corresponding to strong metadata relationships, while later eigenvectors capture
finer distinctions based on feature similarity.

## Eigenvector Oversampling & Subset Selection

Once the graph is constructed, it is converted to an adjacency matrix that will
be used in eigendecomposition.

During eigendecomposition, a pool of the smallest-k eigenvectors is computed
such that k is larger than is estimated to be necessary, as resources allow.
These eigenvectors are median thresholded and subsequently refined via a
coordinate descent bit flipping routine that maximizes weighted adjacency
agreement per binarized eigenvector. The unnormalized Laplacian `L = D - W` is
specifically selected to ensure that eigenvectors faithfully represent the
weight hierarchy without introducing any normalization-related artifacts.

The subset selection process finds the subset of binarized eigenvectors that
best preserves the neighborhood structure defined by the graph. While
eigenvectors are orthogonal by construction, as noted in the
[spectral hashing paper](https://people.csail.mit.edu/torralba/publications/spectralhashing.pdf),
certain combinations of bits become redundant when used together in binary codes.
Specifically, when bits are derived from eigenfunctions that factor as outer
products, their signs satisfy sign(Φᵢ(x₁)Φⱼ(x₂)) = sign(Φᵢ(x₁))·sign(Φⱼ(x₂)),
creating deterministic dependencies that provide no additional information.

A greedy sequential floating forwards selection (SFFS) process iteratively adds,
removes, and swaps bits to improve the score until convergence, automatically
identifying and removing such redundant bits. The scoring function maximizes
Spearman rank correlation (or optionally Kendall's tau-b) between the adjacency
ranking as-is and the adjacency ranking re-ranked by Hamming similarities. This
preserves the hierarchical neighborhood structure encoded in the weighted graph
by ensuring that pairs of nodes with strong edge weights remain close in Hamming
space.

## Landmark-Augmented Hash Learning

Landmark-augmented hash learning extends the binary embeddings to unseen data.
Landmarks are in-sample nodes retaining both visible features and binary codes.
Rather than learning direct mappings from visible features to bits, the method
learns from concatenated landmark codes:

`visible features → landmark neighbors → [concatenated codes → binary codes]`

The landmark retrieval step uses efficient indexing on visible features.
Retrieved landmark codes are concatenated into a fixed-size input for binary
classifiers. This triangulation approach naturally handles the hierarchical
structure: unseen samples find landmarks with similar features, and the landmark
codes encode the full hierarchical context.

## Reference Pipeline

1. **Build sparse k-NN graph**: Use visible features to find k nearest neighbors
   for each sample, creating sparse topology.

2. **Compute hierarchical edge weights**: For each edge in the k-NN graph,
   compute weights using ranked feature spaces with exponential decay, encoding
   the similarity hierarchy.

3. **Perform spectral decomposition**: Apply eigendecomposition with oversampling
   using the unnormalized Laplacian `L = D - W`.

4. **Apply median thresholding**: Binarize eigenvectors using median split.

5. **Refine with coordinate descent**: Apply iterative bit flipping to maximize
   weighted adjacency agreement.

6. **Select bits**: Use prefix testing with Spearman rank correlation to select
   bits that best preserve hierarchical neighbor orderings.

7. **Prune redundant bits**: Apply SFBS with swap operations for bit removal.

8. **Index landmarks**: Build efficient retrieval structure for landmark visible
   features.

9. **Learn hash functions**: Train binary classifiers mapping from concatenated
   landmark codes to target bits.

## Centroid-Linkage Agglomerative Clustering

Once binary codes are obtained, centroid-linkage agglomerative clustering provides
a hierarchical clustering structure over the embedded samples. The clustering
operates directly on the binary codes using Hamming distance, progressively
merging clusters based on centroid proximity.

Cluster centroids are computed using majority vote across each bit position: for
each bit in the centroid code, the value is set to 1 if the majority of cluster
members have that bit set to 1, otherwise 0. Ties favor 0.

The agglomerative process:

1. Initialize each sample as its own singleton cluster with its binary code as
   the centroid.

2. At each iteration, find the pair of clusters with minimum Hamming distance
   between their centroids.

3. Merge the pair into a new cluster and compute the new centroid using majority
   vote.

4. Repeat until a single cluster remains.

The resulting dendrogram captures the hierarchical clustering structure, with
merge heights corresponding to the Hamming distance at which clusters were
combined.

## Clustering and Margin Evaluation

Different cuts through the dendrogram produce different flat clusterings, while
Hamming distance thresholds provide continuous similarity judgments. Both
scenarios require evaluating how well binary partitions preserve the original
weighted graph structure.

For dendrogram cuts, rank-biserial correlation (or optionally variance) measures
the association between cluster membership (binary) and edge weights (continuous
ranks). Higher rank-biserial values indicate within-cluster edges have
systematically higher weights than between-cluster edges, enabling selection of
an optimal number of clusters.

For Hamming margins, each candidate threshold m creates a binary classification
where pairs within the margin are similar and pairs outside are dissimilar.
Rank-biserial correlation evaluates how well this partition aligns with edge
weights. The optimal margin maximizes rank-biserial correlation, providing a
global threshold for similarity/dissimilarity decisions and enabling binary
code-based retrieval.

## Future Directions

- Adaptive decay learning from labeled data
- Multi-view spectral decomposition for handling separate feature modalities
- Normalized Laplacian variants for applications prioritizing stability over
  fine-grain detail preservation
- Attention-based landmark aggregation instead of concatenation
- Graph neural network encoders aware of hierarchical structure
- End-to-end optimization of graph construction through embedding quality
