# Two-Stage Spectral Embedding via Nyström Extension

## Overview

Spectral graph hashing with supervised edge weighting creates a warped manifold where label information distorts natural token-space geometry. Standard landmark-based out-of-sample extension conflates two distinct problems:

1. **Base positioning**: Where a sample belongs in unsupervised token-space
2. **Supervision warping**: How label information shifts that position

The Nyström approach separates these explicitly through dual spectral decomposition and learned warping functions.

## The Fundamental Asymmetry

Direct Nyström extension on supervised spectral codes fails because supervised graphs use label-weighted edge similarities. At test time, labels are unknown, making it impossible to compute the required similarities for proper extension.

The solution separates the observable manifold (tokens) from the unobservable manifold (labels).

## Architecture

### Training Phase

#### Dual Graph Construction

Build two k-NN graphs from identical token-space features:

**Unsupervised Graph**
- Pure token similarity (Jaccard, cosine, etc. on IDF-weighted tokens)
- Edge weights depend only on observable features
- Captures natural document similarity manifold

**Supervised Graph**
- Token + label similarity (hierarchical ranking: categories rank 0, tokens rank 1)
- Edge weights incorporate label information via weight_decay
- Captures label-aware warped manifold

#### Dual Spectral Decomposition

Compute independent spectral codes for each graph:

**Unsupervised Codes**: `spectral.encode(graph_unsup) → codes_unsup`
- Base position in token-space manifold
- Computable for new samples via Nyström extension

**Supervised Codes**: `spectral.encode(graph_sup) → codes_sup`
- Warped position reflecting label influence
- Target for out-of-sample prediction

#### Warping Function

Train binary classifier on the supervision-induced transformation:

```
Φ: codes_unsup → codes_sup
```

The classifier learns how label information systematically shifts positions in spectral space.

### Test Phase

#### Nyström Extension

For new sample x with token features f(x):

1. Compute token-space similarities to training samples: `w = sim(f(x), f(train))`
2. Extend unsupervised eigenvectors: `v_x^(k) = (1/λ_k) Σ_i w_i * v_i^(k)`
3. Apply binarization threshold → `code_unsup(x)`

#### Warping Application

```
code_sup(x) = Φ(code_unsup(x))
```

## Mathematical Foundation

Nyström extension projects new points into existing spectral spaces:
- Given eigenvectors V and eigenvalues Λ from training data
- For test point with edge weights w to training samples
- Extension: `v_new = (1/λ) * W^T * V`

This is exact for the unsupervised manifold because edge weights depend only on observable token features. The supervised manifold is not directly observable at test time since it requires labels, necessitating the learned warping function.

## Advantages

### vs Landmark Encoding

**Landmark approach**: `aggregate(sup_codes[neighbors]) → sup_code`
- Black box conflating positioning and warping
- Implicit warping pattern hidden in aggregation

**Nyström approach**: `nystrom(tokens) → unsup_code`, then `Φ(unsup_code) → sup_code`
- Explicit separation of concerns
- Interpretable warping function
- Modular components

### Computational Efficiency

**Test-time benefits:**
- k-NN search through ANN indices (same as landmarks)
- Compact binary-to-binary classifier (32 bits → 32 bits typical)
- No landmark code storage or retrieval

**Training benefits:**
- Modular warping functions (swap Tsetlin/MLP/etc without re-embedding)
- Retrain Φ without expensive spectral recomputation
- Separate optimization of positioning vs warping

### Interpretability

Decomposition enables analysis of supervision effects:
- Visualize unsupervised vs supervised embeddings
- Identify which bits flip due to label information
- Quantify label-driven geometry deformation
- Debug by separating positioning errors from warping errors

## Implementation

### Nyström Approximation Strategies

**Exact**: Compute similarities to all training samples
- O(N × d) per query for N training samples, d features
- Most accurate

**k-NN Approximate**: Use ANN index for nearest k training samples
- O(k × d) with O(log N) search
- Efficient with minimal error

**Anchored**: Pre-select landmark subset for similarities
- O(L × d) for L landmarks
- Fastest but more approximation error

### Graph Construction

Both graphs use identical topology (same neighbors) with different edge weights:
- Unsupervised: Token similarity only
- Supervised: Token + label similarity via hierarchical weighting

### Storage Requirements

Nyström extension requires:
- Continuous eigenvectors: N × K matrix (before binarization)
- Eigenvalues: K vector
- Training sample IDs
- Token feature index (for similarity computation)

### Binarization Consistency

Apply identical thresholding to both decompositions:
- Unsupervised codes: threshold(eigenvectors_unsup)
- Supervised codes: threshold(eigenvectors_sup)
- Test codes: threshold(nystrom_extension(test_sample))

Supports ITQ, Otsu, median, or sign thresholding.

## Validation

### Diagnostic Analysis

Analyze training data structure:

1. Compute both unsupervised and supervised codes for training samples
2. Calculate residual: `Δ = codes_sup ⊕ codes_unsup` (XOR)
3. Measure predictability of Δ from observable features
4. Compute mutual information I(Δ ; features)

Structured residuals indicate learnable warping patterns.

### Comparative Evaluation

Benchmark three configurations:

**Baseline**: Standard landmark encoder
**Nyström-only**: Unsupervised codes without warping
**Nyström+Warp**: Full two-stage approach

Metrics:
- Hamming distance to ground truth supervised codes
- k-NN classification accuracy
- Retrieval quality (NDCG, combined score)
- Sample efficiency (learning curves)

## Related Work

**Spectral Hashing**: Unsupervised binary codes via spectral decomposition. This approach extends with explicit supervision modeling.

**Self-Taught Hashing**: Trains classifiers to predict hash bits from features. Similar to Nyström warping but without explicit base positioning.

**Metric Learning**: Supervision warping resembles learned distance metrics. Implementation via spectral geometry deformation.

**Semi-Supervised Embedding**: Graph modification incorporating labels. This approach makes modifications explicit and learnable.

## Extensions

### Adaptive Warping

Position-dependent warping incorporating local context:
```
Φ(unsup_code, local_label_density, centroid_distance) → sup_code
```

### Multi-Task Warping

Compose warping functions for multiple supervision signals:
```
sup_code = Φ_category(Φ_tags(Φ_sentiment(unsup_code)))
```

### Alternative Classifiers

Replace binary classifier with neural architectures:
- MLP with learned representations
- Attention mechanisms over landmark contexts
- Graph neural networks on extended positions
