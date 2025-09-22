# Graph-Supervised Self-Taught Hashing with Landmark Extension

An approach to supervised spectral hashing involving the use of virtual nodes to
encode supervision, reconstruction-error-based optimization to select the
optimal subset of bits, and landmark-based hash function learning to enable
out-of-sample extension and cross-domain inference.

## Use Cases

This pipeline addresses scenarios requiring efficient binary embedding-based
retrieval or clustering with varying levels of supervision. The in-sample binary
codes enable fast similarity search and clustering directly via Hamming distance
operations. The landmark-based hash function learning extends this capability to
unseen data, allowing users to query the embedding space using visible features
(e.g., text queries against a document corpus).

The approach is particularly suited for resource-constrained environments where
efficient storage and computation are critical. From a representation learning
perspective, the pipeline projects samples into a latent coordinate space
(Hamming codes) such that samples with similar supervisory relationships occupy
nearby coordinates, with similarity measured by Hamming distance. The virtual
node architecture allows arbitrarily complex supervisory structures to shape
this embedding space during the spectral decomposition process.

## Background

[Spectral hashing](https://people.csail.mit.edu/torralba/publications/spectralhashing.pdf)
provides an efficient method to obtain binary embeddings for nodes in a graph
where Hamming distances among embeddings approximate graph distance among nodes
in the graph. The process of generating embeddings for unseen nodes is called
out-of-sample extension. One method for enabling out-of-sample extension
described in [Self-Taught Hashing (STH)](https://arxiv.org/pdf/1004.5370)
involves training k binary classifiers to predict individual bits from arbitrary
node features, using the binary codes obtained for in-sample nodes as classifier
targets.

## Graph-Centric Supervision

In this approach to spectral hashing, supervision is injected during graph
construction through a virtual node architecture, where virtual nodes
representing latent concepts, classes, or other supervisory metadata are added
to the graph and connected to all data nodes that are associated with them. For
example, category and author supervision can be injected into a graph of text
documents by adding virtual nodes for each category and author and connecting
them to all relevant data nodes.

The feature space is formed from the union of multiple feature subspaces, each
assigned to a specific rank. In the category, author, and document example,
category IDs are assigned rank 0, author IDs rank 1, and visible (e.g. text
ngram) features rank 2. Virtual nodes contain features only at their designated
rank (category nodes have only category ID features, author nodes have only
author ID features). Data nodes contain features at all ranks: the IDs of
connected virtual nodes plus their *visible* features (e.g. text ngram features
that must be available at inference time). This rank assignment enables
similarity computation as:

    w(i, j) = Σ[sim_r(i, j) * exp(-r * decay)] / Σ[exp(-r * decay)]

...where `sim_r(i, j)` computes similarity between nodes `i` and `j` considering
only features at rank `r`. Feature overlap in lower ranks (e.g. between
documents and their associated author/category nodes or between documents
sharing authors/categories themselves) results in exponentially higher
similarity scores than feature overlap strictly in higher ranks (e.g. between
documents sharing only visible (e.g. text ngram) features). This exponential
decay pairs
naturally with the log-space mapping in reconstruction error (described below),
ensuring supervision shapes the manifold structure hierarchically.

Jaccard, Dice, or any other weighted or unweighted set similarity can be used to
compute per-rank similarity.

## The Random Walk Laplacian

The random walk normalized Laplacian `L_rw = I - D^(-1)W` is essential for
preserving this rank hierarchy in heterogeneous graphs. Virtual nodes naturally
have significantly higher degrees than data nodes, and this imbalance causes
both unnormalized and standard normalized Laplacians to produce embeddings that
either over or under-emphasize the supervision injected into the graph. It is
therefore necessary to use the random walk normalized Laplacian, which preserves
the rank-based importance ratios regardless of node degrees and enables edge
weights to be specifically designed as transition probabilities. In the
category, author, and document graph example, a document transitions to its
category with probability proportional to `exp(0)`, to its author with
probability proportional to `exp(-1 · decay)`, and to its strictly visible-feature-similar
neighbors with probability proportional to `exp(-2 · decay)`.

## Reconstruction-Optimized Bit Selection

Various eigenvector selection approaches exist (e.g. smallest-k, eigengap-based,
redundancy handling). This approach attempts to remain aligned with foundational
spectral hashing literature while addressing practical considerations like the
eigengap heuristic and redundant bit pruning. It preserves graph-injected
supervision through the pipeline such that Hamming distances among binary codes
approximate similarity relationships as defined by the graph.

Eigenvectors are oversampled during eigendecomposition, median-thresholded, and
then refined with a greedy bit-flipping routine that flips the bits of
individual nodes to maximize weighted adjacency agreement per-bit. The
bit-flipping routine operates on each column of bits in isolation.

Bit selection determines which of these binarized eigenvectors are retained
using reconstruction error minimization. Target distances from edge weights use
negative log transformation `target_dist = -log(weight + ε)` normalized to
`[0,1]`. Reconstruction error uses `error = |weight| * (target_norm -
hamming_norm)²`.

Signed, unsigned, weighted, normalized, and unnormalized graphs are supported by
this approach. Bit selection is particularly important for signed graphs because
eigenvectors can encode patterns that are not useful for reconstruction error or
Hamming distance metrics. This definition of reconstruction error ensures that
dissimilar nodes marked by negative edges remain distant in Hamming space.

## Landmark-Augmented Hash Learning

Landmark-augmented hash learning modifies the standard STH approach by
introducing an intermediate landmark-based representation. Landmarks are
in-sample data nodes (either the full in-sample set or an optimized subset) that
retain both their visible features and their final binary codes produced by the
bit selection process.

Rather than learning classifiers that map directly from *visible features* to
binary codes, this method learns to map from concatenated *landmark codes* to
binary codes. The transformation changes from `visible features → binary codes`
to `visible features → landmark neighbors → [concatenated neighbor codes →
binary codes]`, where the bracketed portion represents the learned mapping. This
provides an alternative framing of the hash learning problem as triangulation
from landmarks rather than direct feature-based mapping.

The `visible features → landmark neighbors` step leverages an indexing structure
(LSH, inverted index, etc.) to identify nearby landmark nodes based on visible
features. The binary codes of these k nearest landmarks are concatenated to form
a fixed-size input vector for k independent binary classifiers that predict
individual bits.

During training, per-bit classifiers learn to map from concatenated landmark
codes to target binary codes. For inference on unseen nodes, the trained
classifiers predict binary codes from the concatenated codes of retrieved
landmarks.

The fixed-size input space (k × code_length) remains constant regardless of
visible feature dimensionality, enabling more predictable training behavior.
Inference speed depends on the efficiency of the landmark indexing structure
rather than feature dimensionality.

## Reference Implementation Pipeline

1. **Graph construction**: Create virtual nodes for each unique label/category.
   Virtual nodes contain features only at their rank (e.g., category nodes have
   rank-0 features), while documents contain features at all ranks. Compute all
   edge weights via the unified similarity function with exponential rank decay,
   ensuring documents connecting to many virtual nodes don't overwhelm the
   graph. Build mutual k-NN edges between documents using visible features.

2. **Spectral decomposition**: Eigendecomposition with oversampling using random
   walk normalized Laplacian `L_rw = I - D^(-1)W`. The eigenvalue spectrum
   exhibits gaps corresponding to rank levels, with near-zero eigenvalues for
   category-level structure, intermediate values for author-level mixing, and
   larger eigenvalues capturing k-NN document similarity.

3. **Median thresholding**: Binarization via median split

4. **Coordinate descent refinement**: Iterative bit flipping to maximize
   weighted adjacency agreement, operating on each bit position across all nodes

5. **Bit selection**: Prefix testing with reconstruction error
   metric on the full heterogeneous graph to select final binary codes from
   binarized eigenvectors

6. **Redundant bit pruning**: Prefix selection followed by SFBS with swap
   operations for bit removal using reconstruction error

7. **Landmark indexing**: Fast set-based methods (e.g., LSH, inverted indices
   with Jaccard/overlap similarity)

8. **Hash function learning**: Multi-label binary classifier or ensemble
   training


## Experimental Variations

**Graph construction**:
- Mutual vs non-mutual k-NN
- Edge seeding strategies
- Decay parameter (1.5-2.0 for strong hierarchy)
- Laplacian variants (random walk strongly preferred for virtual node
  architectures)

**Binarization**:
- ITQ rotation
- Sign thresholding
- Skip coordinate descent refinement

**Bit selection**:
- Prefix + SFBS + swap (reference implementation)
- Fixed selection alternatives
- SFFS variants

**Landmark methods**:
- Variable landmark counts
- Selection via LSH or inverted indices

**Classifiers**:
- Multi-label vs independent binary
- Linear vs non-linear

## Future Directions

- Using bit-frequencies for landmark encoding instead of (or in addition to)
  concatenated codes for triangulation based on landmark bit statistics of
  neighbors.
- Mixing landmarks with visible features.
- Applying coreset sampling for landmark selection.
- Developing encoder-aware bit selection.
- Creating end-to-end optimization pipeline.
