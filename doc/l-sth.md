# Graph-Supervised Self-Taught Hashing with Landmark Extension

## Abstract

An approach to spectral hashing with three contributions: (1) graph-centric
supervision via virtual nodes during construction, (2) reconstruction error
optimization for bit selection preserving graph relationships, and (3)
landmark-based out-of-sample extension enabling cross-domain inference. The
method supports different feature spaces for graph construction and inference
while maintaining fixed input dimensionality (n_hidden × n_landmarks) for
scalable deployment.

## Introduction & Motivation

This work builds on self-taught hashing with specific improvements to graph
construction (single point for supervision injection), bit selection
(reconstruction error optimization), and out-of-sample extension (landmark-based
triangulation). The approach targets consumer and edge hardware, enabling
training and deployment on laptops, Raspberry Pi, and embedded systems without
GPU requirements, enabling nightly retraining and CI/CD pipelines on standard
hardware.

## Background

Self-taught hashing (STH) addresses out-of-sample extension via per-bit
classifiers. Anchor graph hashing (AGH) uses anchor points and Nyström
extension. Hypergraph spectral hashing (HGH) captures higher-order relationships
beyond pairwise connections through hyperedge decomposition. STH and AGH require
the same feature space for graph construction and extension. Semi-supervised
spectral methods incorporate supervision through graph structure. This approach
enables different feature spaces through graph supervision and landmark
triangulation, while virtual nodes provide a flexible alternative to hyperedges
for encoding complex relationships.

## Key Principles

### Graph-Centric Supervision

All supervision is injected during graph construction through a virtual node
architecture. Virtual nodes represent latent concepts, classes, or bipartite
connections with features restricted to their designated rank level. Documents
connect to virtual nodes based on ground-truth relationships, creating a
heterogeneous graph (e.g., involving document and virtual node types like
category, author, etc.).

When multiple labelings exist (e.g., document categories and authors), each gets
its own virtual node type. The system assigns ranks to edge types based on
importance (lower rank = higher importance). Virtual node edges receive higher
priority (lower rank values) than document-document edges. For example, in a
system with text features and class labels: document-class edges are rank 0
(highest importance), document-author edges might be rank 1, and
document-document edges are rank 2 (lower importance). Edge weights are computed
via a unified function: `w(i, j) = Σ[sim_r(i, j) * exp(-r * decay)] / Σ[exp(-r *
decay)]`, where `sim_r(i, j)` computes similarity between nodes `i` and `j`
considering only features at rank `r`. This exponential decay pairs naturally with
the log-space mapping in reconstruction error, ensuring supervision shapes the
manifold structure hierarchically.

Each virtual node type operates at a specific rank level (e.g., class labels at
rank 0, authors at rank 1). When computing similarities, virtual nodes only
match features at their rank, while documents use features across all ranks.
This ensures clean semantic separation with hierarchical importance. Document to
document edge weights use symmetric similarity measures appropriate for the
data, such as set-based similarity over text features. Document to virtual node
weights are computed through the same unified function, automatically handling
rank-specific matching, potentially downweighting high-degree virtual nodes.
Higher-rank edges create progressively looser clustering (e.g., documents of the
same category cluster tighter than documents of the same author, followed by
documents sharing text features).

### Reconstruction-Optimized Bit Selection

Various eigenvector selection approaches exist (smallest-k, eigengap-based,
redundancy handling). This approach attempts to remain aligned with foundational
spectral hashing literature while addressing practical considerations like the
eigen-gap heuristic and redundant bit pruning. It preserves graph-injected
supervision through the pipeline such that Hamming distances among binary codes
approximate similarity relationships as defined by the graph.

The approach oversamples eigenvectors then applies median thresholding and
coordinate descent refinement (maximizing weighted adjacency agreement per bit).
Bit selection and pruning use reconstruction error minimization where target
distances are computed from edge weights via negative log transformation:
`target_dist = -log(weight + ε)`. Both target and Hamming distances are
normalized to `[0,1]`, and the weighted reconstruction error becomes `weight *
(target_norm - hamming_norm)²`. This weight-modulated error ensures higher
weight edges contribute more to error, with the log-space mapping naturally
pairing with the exponential rank decay from supervision. The reconstruction
error computation operates on the full heterogeneous graph, preserving virtual
node relationships through the weight×rank combination.

The approach handles signed, unsigned, weighted, normalized, and unnormalized
graphs intermixed. Bit selection is particularly important for signed graphs
because eigenvectors can encode patterns that are not useful for reconstruction
error or Hamming distance metrics. The reconstruction error ensures that
dissimilar nodes marked by negative edges remain distant in Hamming space.

### Landmark-Augmented Hash Learning

Addresses cross-domain out-of-sample extension. The graph construction leverages
both observable features (text, metadata) and domain knowledge (encoded via
virtual nodes and their relationships). Observable features form the
unsupervised backbone of the graph and are typically used for per-bit learners
in self-taught hashing, though any derived or alternative feature space can be
used for the learners. Landmark discovery uses observable features for
triangulation. Input dimensionality becomes fixed: n_hidden × n_landmarks.

## Reference Implementation Pipeline

1. **Graph construction**: Create virtual nodes for each unique label/category.
   Virtual nodes contain features only at their rank (e.g., title nodes have
   rank-0 features), while documents contain features at all ranks. Compute all
   edge weights via the unified similarity function with exponential rank decay,
   ensuring documents connecting to many virtual nodes don't overwhelm the
   graph. Build mutual k-NN edges between documents using observable features.

2. **Spectral decomposition**: Eigendecomposition with oversampling

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
