# Graph-Supervised Self-Taught Hashing with Landmark Extension

## Abstract

An approach to spectral hashing based on three principles: graph-centric
supervision where all supervision is injected during graph construction,
reconstruction-optimized bit selection that preserves graph-encoded
relationships, and landmark augmentation for cross-domain out-of-sample
extension. The method reduces the input dimensionality for out-of-sample
extension to a fixed size (n_hidden × n_landmarks) and enables different feature
spaces for graph construction and inference.

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
extension. Both require the same feature space for graph construction and
extension. Semi-supervised spectral methods incorporate supervision through
graph structure. This approach enables different feature spaces through graph
supervision and landmark triangulation.

## Key Principles

### Graph-Centric Supervision

All supervision is injected during graph construction through a virtual node
architecture. Virtual nodes represent latent concepts, classes, or bipartite
connections without observable features, existing only as structural
connections. Documents connect to virtual nodes based on ground-truth
relationships, creating a heterogeneous graph with multiple node types.

When multiple labelings exist (e.g., document categories and authors), each gets
its own virtual node type. The system assigns ranks to edge types:
document-virtual_A edges might be rank 0, document-virtual_B edges rank 1, and
document-document edges rank 2. Edge weights decay exponentially with rank,
normalized by the sum of per-rank weights. This exponential decay pairs
naturally with the log-space mapping in reconstruction error, ensuring
supervision shapes the manifold structure hierarchically.

Document to document edge weights use symmetric similarity measures appropriate
for the data, such as set-based similarity over text features with IDF
weighting. Document to virtual node weights can use similar approaches,
potentially downweighting high-degree virtual nodes. Higher-rank edges create
progressively looser clustering (e.g., documents of the same category cluster
tighter than documents of the same author, followed by documents sharing text
features).

### Reconstruction-Optimized Bit Selection

Various eigenvector selection approaches exist (smallest-k, eigengap-based,
redundancy handling). This approach attempts to remain aligned with foundational
spectral hashing literature while addressing practical considerations like the
eigen-gap heuristic and redundant bit pruning. It preserves graph-injected
supervision through the pipeline such that Hamming distances among binary codes
approximate similarity relationships as defined by the graph.

The approach oversamples eigenvectors then applies median thresholding and
coordinate descent refinement (maximizing weighted adjacency agreement per bit).
Bit selection and pruning use stress-based error minimization where target
distances are computed from edge weights via negative log transformation:
`target_dist = -log(weight + ε)`. Both target and Hamming distances are
normalized to `[0,1]`, and the weighted stress error becomes `weight *
(target_norm - hamming_norm)²`. This weight-modulated stress ensures higher
weight edges contribute more to error, with the log-space mapping naturally
pairing with the exponential rank decay from supervision. The stress computation
operates on the full heterogeneous graph, preserving virtual node relationships
through the weight×rank combination.

The approach handles signed, unsigned, weighted, normalized, and unnormalized
graphs intermixed. Bit selection is particularly important for signed graphs
because eigenvectors can encode patterns that are not useful for reconstruction
error or Hamming distance metrics. The reconstruction error ensures that
dissimilar nodes marked by negative edges remain distant in Hamming space.

### Landmark-Augmented Hash Learning

Addresses cross-domain out-of-sample extension. Feature Space A contains domain
features for graph topology (typically a superset of B with supervision).
Feature Space B contains observable features for landmark discovery. Feature
spaces can be identical when supervision is injected via edges or anchors.
Triangulation learns from concatenated landmark codes. Input dimensionality
becomes fixed: n_hidden × n_landmarks.

## Reference Implementation Pipeline

1. **Graph construction**: Create virtual nodes for each unique label/category.
   Treat virtual node IDs as features - virtual nodes have their own ID as their
   sole feature, while documents include virtual node IDs among their features.
   Compute document-virtual weights using weighted Jaccard or similar set-based
   similarity, ensuring documents connecting to many virtual nodes don't overwhelm
   the graph. Build mutual k-NN edges between documents using observable features.

2. **Spectral decomposition**: Eigendecomposition with oversampling

3. **Binarization**: Median thresholding

4. **Coordinate descent refinement**: Iterative bit flipping to maximize
   weighted adjacency agreement, operating on individual node codes one bit at a
   time

5. **Bit selection**: Prefix testing with stress-based reconstruction error
   metric on the full heterogeneous graph

6. **SFBS pruning**: Redundant bit removal via floating selection using
   stress-based error

7. **Landmark indexing**: Fast set-based methods

8. **Hash function learning**: Multi-label binary classifier or ensemble
   training

## Out-of-Sample Extension

**Training**: Select and index landmarks using fast set-based methods. Compute
k-NN in Feature Space B. Train classifiers on (landmark_codes to target_code)
pairs.

**Inference**: Look up landmarks via simple indexing. Construct feature vector
from concatenated landmark codes. Apply trained classifiers to predict target
code.

## Experimental Variations

**Graph construction**: sigma-k, non-mutual k-NN, edge seeding strategies,
rank-based supervision, class-weighted edges.

**Binarization**: ITQ, sign thresholding, skip coordinate descent.

**Bit selection**: SFFS/SFBS variants, fixed selection, prefix-only vs floating.

**Landmark lookup**: various fast set-based methods, variable landmark counts.

**Classifiers**: multi-label vs independent binary, linear vs non-linear.

## Future Directions

- Encoding landmark codes as bit-frequencies instead of (or in addition to)
  concatenated codes for triangulation based on landmark bit statistics of
  neighbors.

- Mixing landmarks with raw features.

- Coreset sampling for landmark selection.

- Encoder-aware bit selection.

- End-to-end optimization pipeline.
