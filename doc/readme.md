# Spectral Graph Hashing

A framework for learning binary codes from graph structure via spectral
decomposition. Binary codes enable fast Hamming-based similarity search and
clustering, with Tsetlin machine encoders extending retrieval to unseen queries.

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

## Out-of-Sample Extension

Two approaches extend learned embeddings to unseen data:

### Self-Taught Hashing (STH)

Direct feature-to-code prediction via per-dimension chi2 feature selection.

**Approach**: Select top-k features per output dimension using chi2 association,
then train a Tsetlin machine encoder on the selected features.

**Characteristics**:
- Per-dimension feature spaces (individualized)
- Direct token/pixel features as input
- Simpler pipeline with fewer hyperparameters
- Well-suited for high-dimensional sparse features

See [sth.md](sth.md) for details.

## Pipeline Components

### Feature Extraction and Ranking

Features are partitioned into **ranks** representing similarity hierarchy
levels:

| Rank | Signal Strength | Examples |
|------|-----------------|----------|
| 0 | Strong (exact match) | Categories, tags, document labels |
| 1 | Moderate | Structured metadata, entity types |
| 2+ | Weak | Text tokens, n-grams |

Each feature has optional IDF-style weights within its rank.

### Hierarchical Edge Weighting

Edge weights combine per-rank similarities with exponential decay:

    sim(i,j) = Σ[sim_r(i,j) * exp(-r * decay)] / Σ[exp(-r * decay)]

Where:
- sim_r(i,j) computes similarity using only features at rank r
- decay > 0 prioritizes lower ranks (categories matter most)
- decay < 0 prioritizes higher ranks (fine-grained features matter most)
- decay = 0 applies equal weighting across ranks

### Adjacency Construction

Builds weighted k-NN graph from feature indices.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| knn | int | - | Number of nearest neighbors |
| knn_alpha | float | 1.0 | Scaling for adaptive k modes |
| knn_mode | string | nil | Adaptive mode: cknn, sigma, or nil |
| knn_mutual | bool | false | Require bidirectional edges |
| weight_decay | float | 0.0 | Exponential decay for ranked features |
| bridge | string | mst | Component bridging: mst, largest, none |

### Spectral Decomposition

Extracts eigenvectors from graph Laplacian using PRIMME solver.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_dims | int | - | Number of eigenvectors to extract |
| laplacian | string | unnormalized | Laplacian type |
| eps | float | 1e-8 | Convergence tolerance |
| threshold | function | - | Binarization function |

### Thresholding Methods

| Method | Description | Best For |
|--------|-------------|----------|
| ITQ | Iterative quantization with learned rotation | High-quality codes |
| Otsu | Per-dimension threshold maximizing variance | Quality-aware selection |
| Median | Per-dimension median threshold | Balanced codes |
| Sign | Simple zero-threshold | Pre-centered embeddings |

### Evaluation

| Mode | Description |
|------|-------------|
| Retrieval | Score via ranking metrics (NDCG, Spearman, Pearson) |
| Clustering | Score via dendrogram analysis with elbow detection |

## Module Reference

| Module | Purpose |
|--------|---------|
| inv | Inverted index with ranked features and IDF weights |
| ann | Approximate nearest neighbor index for binary codes |
| hbi | Hamming ball index for exact Hamming distance search |
| graph | Build k-NN/CkNN adjacency graphs with bridging |
| spectral | Laplacian eigenvector computation via PRIMME |
| itq | Binarization: ITQ rotation, Otsu, median, sign |
| hlth | Landmark and nystrom-based hash encoding |
| eval | Retrieval and clustering evaluation |
| optimize | Hyperparameter search with adaptive sampling |
