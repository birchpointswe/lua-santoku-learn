# Graph-Supervised Self-Taught Hashing with Landmark Extension

An approach to supervised spectral hashing involving the use of virtual nodes to
encode supervision, reconstruction-error-based optimization to select the
optimal subset of bits, and landmark-based hash function learning to enable
out-of-sample extension and cross-domain inference.

## Background

Spectral hashing provides an efficient method to obtain binary embeddings for
nodes in a graph where hamming distances among embeddings approximate graph
distance among nodes in the graph. The process of generating embeddings for
unseen nodes is called out-of-sample extension. One method for enabling
out-of-sample extension described in Self-Taught Hashing (STH) involves training
k binary classifiers to predict individual hash code bits from arbitrary node
features, using the binary codes obtained for in-sample nodes as classifier
targets.

## Graph-Centric Supervision

In this approach to spectral hashing, supervision is injected during graph
construction through a virtual node architecture, where virtual nodes
representing latent concepts, classes, or other supervisory metadata are added
to the graph and connected to all data nodes that are associated with them. For
example, category and author supervision can be injected into a graph of text
documents by adding virtual nodes for each category and author and connecting
them to all relevant data nodes.

Virtual and data nodes are given binary features such that virtual nodes have
their ID as their (singleton) set of features and data nodes have the IDs of the
virtual nodes they connect to as their set of features. Additionally, data nodes
have a set of *visible* features which must be available at inference time (e.g.
binary text features). These features are assigned to ranks, such that
similarity (and therefore edge weight) between nodes can be computed as:

    w(i, j) = Σ[sim_r(i, j) * exp(-r * decay)] / Σ[exp(-r * decay)]

...where `sim_r(i, j)` computes similarity between nodes `i` and `j` considering
only features at rank `r`. Feature overlap in lower ranks (e.g. between
documents and their associated author/category nodes or between documents
sharing authors/categories thsemselves) results in expoentially higher
similarity scores than feature overlap strictly in higher ranks (e.g. between
documents sharing only text/visible features). This exponential decay pairs
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
probability proportional to `exp(-1 · decay)`, and to its strictly text-similar
neighbors with probability proportional to `exp(-2 · decay)`.

## Reconstruction-Optimized Bit Selection

Various eigenvector selection approaches exist (e.g. smallest-k, eigengap-based,
redundancy handling). This approach attempts to remain aligned with foundational
spectral hashing literature while addressing practical considerations like the
eigen-gap heuristic and redundant bit pruning. It preserves graph-injected
supervision through the pipeline such that Hamming distances among binary codes
approximate similarity relationships as defined by the graph.

Eigenvectors are oversampled during eigen-decomposition, median-thresholded, and
then refined with a greedy bit-flipping routine that flips the code-bits of
individual nodes to maximize weighted adjacency agreement per-bit. The
bit-flipping routine operates on each column of bits in isolation.

Bit selection determines which of these now-binary eigenvectors are retained
using reconstruction error minimization. Target distances from edge weights use
negative log transformation `target_dist = -log(weight + ε)` normalized to
`[0,1]`. Reconstruction error use `error = |weight| * (target_norm -
hamming_norm)²`.

Signed, unsigned, weighted, normalized, and unnormalized graphs are supported by
this approach. Bit selection is particularly important for signed graphs because
eigenvectors can encode patterns that are not useful for reconstruction error or
Hamming distance metrics. This definition of reconstruction error ensures that
dissimilar nodes marked by negative edges remain distant in Hamming space.

## Landmark-Augmented Hash Learning

TODO

## Reference Implementation Pipeline

1. **Graph construction**: Create virtual nodes for each unique label/category.
   Virtual nodes contain features only at their rank (e.g., title nodes have
   rank-0 features), while documents contain features at all ranks. Compute all
   edge weights via the unified similarity function with exponential rank decay,
   ensuring documents connecting to many virtual nodes don't overwhelm the
   graph. Build mutual k-NN edges between documents using observable features.

2. **Spectral decomposition**: Eigendecomposition with oversampling using random
   walk normalized Laplacian `L_rw = I - D^(-1)W`. The eigenvalue spectrum
   exhibits gaps corresponding to rank levels, with near-zero eigenvalues for
   category-level structure, intermediate values for author-level mixing, and
   larger eigenvalues capturing k-NN document similarity.

3. **Median thresholding**: Binarization via median split

4. **Coordinate descent refinement**: Iterative bit flipping to maximize
   weighted adjacency agreement, operating on individual node codes one bit at a
   time

5. **Bit selection**: Prefix testing with reconstruction error
   metric on the full heterogeneous graph

6. **Redundant bit pruning**: Prefix selection followed by SFBS with swap
   operations for bit removal using reconstruction error

7. **Landmark indexing**: Fast set-based methods (e.g., LSH, inverted indices
   with Jaccard/overlap similarity)

8. **Hash function learning**: Multi-label binary classifier or ensemble
   training

## Out-of-Sample Extension

**Training**: Select and index landmarks using fast set-based methods (e.g.,
LSH, inverted indices with Jaccard/overlap similarity). Compute k-NN using
observable features. Train classifiers on (landmark_codes to target_code) pairs.

**Inference**: Look up landmarks via simple indexing. Construct feature vector
from concatenated landmark codes. Apply trained classifiers to predict target
code.

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

**Classifiers**: multi-label vs independent binary, linear vs non-linear.

## Future Directions

- Using bit-frequencies for landmark encoding instead of (or in addition to)
  concatenated codes for triangulation based on landmark bit statistics of
  neighbors.

- Mixing landmarks with raw features.

- Applying coreset sampling for landmark selection.

- Developing encoder-aware bit selection.

- Creating end-to-end optimization pipeline.
