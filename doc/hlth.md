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
best preserves the neighborhood structure defined by the graph. All prefixes are
scored, and then beginning with the best scoring prefix, a greedy sequential
floating backwards selection (SFBS) process iteratively removes, adds, and swaps
bits to improve the score until convergence. The scoring function is Spearman
rank correlation between two ranked nearest neighbors lists derived from the
same adjacency lists used in the Laplacian. The first version is the ranked list
of neighbors using the adjacency weights, and the second version is the same
list of neighbors sorted by Hamming similarity using the current bit subset.

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

## Future Directions

- Adaptive decay learning from labeled data
- Multi-view spectral decomposition for handling separate feature modalities
- Normalized Laplacian variants for applications prioritizing stability over
  fine-grain detail preservation
- Attention-based landmark aggregation instead of concatenation
- Graph neural network encoders aware of hierarchical structure
- End-to-end optimization of graph construction through embedding quality
