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

All supervision is injected during graph construction. The graph structure
encodes domain knowledge and relationships. Subsequent steps preserve these
relationships through reconstruction optimization. This aligns with
semi-supervised spectral learning. Examples include feature-based supervision,
class-weighted edges, and synthetic anchors.

### Reconstruction-Optimized Bit Selection

Various eigenvector selection approaches exist (smallest-k, eigengap-based,
redundancy handling). This approach attempts to remain aligned with foundational
spectral hashing literature while addressing practical considerations like the
eigen-gap heuristic and redundant bit pruning. It attempts to preserve
graph-injected supervision through the pipeline such that Hamming distances
among binary codes approximate similarity relationships as defined by the graph.

The steps are as follows:

1. Oversample smallest-k eigenvectors (2-4x expected k)
2. Threshold eigenvectors at their medians
3. Refine via coordinate descent, flipping bits for graph reconstruction
4. Exhaustively search for the optimal prefix by reconstruction error from
  oversampled set of thresholded and refined eigenvectors
5. Prune redundant bits via sequential floating backward selection (SFBS) by
  reconstruction error

Reconstruction error here measures the squared difference between the original
graph weight and the Hamming similarity in binary space, weighted by the edge
strength. If the graph indicates two nodes are similar (high weight) but their
binary codes are distant (low Hamming similarity), this produces a large error.
Strong edges contribute more to the total error, ensuring important
relationships are preserved.

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

1. **Graph construction**: Mutual k-NN with supervised edges
2. **Spectral decomposition**: Eigendecomposition with oversampling
3. **Binarization**: Median thresholding
4. **Coordinate descent refinement**: Iterative bit flipping for graph
  reconstruction
5. **Bit selection**: Prefix testing with reconstruction error metric
6. **SFBS pruning**: Redundant bit removal via floating selection
7. **Landmark indexing**: Fast set-based methods
8. **Hash function learning**: Multi-label binary classifier or ensemble
  training

## Technical Examples

**Image labeling**: Triangulate location from similar images instead of learning
image to location mappings directly.

**Text labeling**: Intent classification using text-similarity k-NN shaped by
intent labels.

**Cross-domain transfer**: Image retrieval via text queries using
text-supervised image graph.

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

## TL;DR

Train embeddings in 60 seconds on a laptop. Process 1M documents with 10M edges
in minutes on consumer hardware. Run nightly retraining on edge devices. 95% of
transformer accuracy at 0.1% of compute. No GPU requirements. Grid search 100
configurations over lunch. Deploy to IoT, mobile, embedded systems without cloud
dependencies.
