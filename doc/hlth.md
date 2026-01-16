# Landmark-based Tsetlin Hashing (HLTH)

Out-of-sample extension via landmark aggregation and code consensus.

## Overview

Landmark-based hashing extends binary embeddings to unseen data by aggregating
codes from retrieved landmarks. Rather than predicting codes directly from raw
features, the approach retrieves similar training samples and combines their
known codes into a representation for classification.

### Process

1. Compute spectral codes for training data via graph decomposition
2. Index landmarks with both observable features and binary codes
3. For new query: retrieve k-nearest landmarks by observable features
4. Aggregate landmark codes using selected mode (concat or frequency)
5. Train Tsetlin machine encoder on aggregated representation
6. Predict codes for new samples using learned encoder

## Landmark Aggregation Modes

### Concat Mode

Directly concatenates k landmark codes into a single feature vector:

- Output dimension: n_landmarks × n_hidden bits
- Position-sensitive: landmark order matters
- Direct representation of neighborhood
- Risk of overfitting: encodes exact landmark identities rather than aggregate patterns

Example: 4 landmarks × 32-bit codes = 128-bit feature vector

### Frequency Mode

Computes bit frequency statistics across landmarks with multi-threshold encoding:

1. For each bit position, count frequency across k landmarks
2. Apply multiple threshold levels to each frequency
3. Create binary feature indicating "bit appears in ≥ threshold% of landmarks"

- Output dimension: n_hidden × n_thresholds bits
- Position-invariant: landmark order doesn't matter
- Robust to noise via consensus encoding
- Better generalization: aggregates patterns rather than memorizing landmarks
- Preferred for out-of-sample extension where train/test gap is a concern

Example: 32 bits × 7 thresholds = 224-bit feature vector

Thresholds typically span [1/8, 2/8, ..., 7/8], encoding gradient of agreement
from "bit rare in neighborhood" to "bit ubiquitous in neighborhood".

## Parameters

### Landmark Encoding

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| mode | string | frequency | Aggregation mode: "concat" or "frequency" |
| n_landmarks | int | 16 | Number of landmarks to retrieve |
| n_thresholds | int | 8 | Threshold levels for frequency mode |
| quantile | bool | false | Use quantile-based thresholds |

### Landmark Retrieval

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| cmp | string | jaccard | Similarity metric for landmark retrieval |
| tversky_alpha | float | 0.5 | Tversky alpha parameter |
| tversky_beta | float | 0.5 | Tversky beta parameter |
| probe_radius | int | 2 | Search radius for ANN index |
| rank_filter | int | -1 | Limit to top-N ranks (-1 for all) |

### Tsetlin Machine

| Parameter | Type | Description |
|-----------|------|-------------|
| clauses | range | Clause count |
| clause_tolerance | range | Tolerance threshold |
| clause_maximum | range | Maximum threshold |
| target | range | Target activation |
| specificity | range | Feature specificity |
| include_bits | range | Bits to include in clauses |

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| search_rounds | int | 4 | Hyperparameter search rounds |
| search_trials | int | 10 | Trials per round |
| search_iterations | int | 20 | Training iterations per trial |
| final_iterations | int | 400 | Final training iterations |

## Lookup Modes

### Token-Space Lookup

Default mode using observable features (tokens, pixels) for landmark retrieval:

- Build inverted index over training tokens with IDF weights
- Query new sample tokens against index
- Retrieve landmarks by weighted similarity (Jaccard, Tversky, etc.)
- Aggregate landmark supervised codes

This is the standard approach when supervision is encoded in spectral codes.

### Unsupervised-Space Lookup

Alternative mode using unsupervised spectral codes for landmark retrieval:

- Compute both supervised and unsupervised spectral codes for training data
- Index landmarks by unsupervised codes (token-based similarity)
- Query new samples via Nyström extension to unsupervised space
- Retrieve landmarks in unsupervised code space
- Aggregate landmark supervised codes

This separates positioning (where a sample belongs) from supervision (how
labels shift that position). See Nyström Extension section below.

## Aggregation Modes

### Supervised Aggregation

Aggregate supervised spectral codes from retrieved landmarks:

```lua
encoder_args.codes_index = train.index_sup
```

This is the standard approach where landmark codes come from the supervised
spectral decomposition that incorporates category information.

### Unsupervised Aggregation

Aggregate unsupervised spectral codes from retrieved landmarks:

```lua
encoder_args.codes_index = train.index_unsup
```

Useful for transfer learning or when supervised codes are unavailable at
inference time.

## Nyström Extension

For unsupervised-space lookup, new samples must be projected into the
unsupervised spectral space. Nyström extension achieves this:

### Mathematical Foundation

Nyström extension projects new points into existing spectral spaces:
- Given eigenvectors V and eigenvalues Λ from training data
- For test point with edge weights w to training samples
- Extension: `v_new = (1/λ) * W^T * V`

This is exact for the unsupervised manifold because edge weights depend only
on observable token features.

### Process

For new sample x with token features f(x):

1. Compute token-space similarities to training samples: `w = sim(f(x), f(train))`
2. Extend unsupervised eigenvectors: `v_x^(k) = (1/λ_k) Σ_i w_i * v_i^(k)`
3. Apply binarization threshold → `code_unsup(x)`

### Approximation Strategies

**Exact**: Compute similarities to all training samples
- O(N × d) per query for N training samples, d features
- Most accurate

**k-NN Approximate**: Use ANN index for nearest k training samples
- O(k × d) with O(log N) search
- Efficient with minimal error

### Configuration

```lua
nystrom_encode = hlth.nystrom_encoder({
  features_index = train.node_features_graph,
  eigenvectors = train.raw_codes_unsup,
  eigenvalues = train.eigenvalues_unsup,
  ids = train.ids_unsup,
  n_dims = train.dims_unsup,
  threshold = threshold_fn,
  k_neighbors = cfg.nystrom.k_neighbors,
  cmp = cfg.nystrom.cmp,
  cmp_alpha = cfg.nystrom.cmp_alpha,
  probe_radius = cfg.nystrom.probe_radius,
})
```

### Nyström Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| k_neighbors | int | 0 | k for approximate (0 = exact) |
| cmp | string | jaccard | Similarity metric |
| cmp_alpha | float | 0.5 | Tversky alpha |
| cmp_beta | float | 0.5 | Tversky beta |
| probe_radius | int | 2 | ANN probe radius |
| rank_filter | int | -1 | Rank filter (-1 = all) |

### Two-Stage Warping

When using Nyström extension with supervised codes, the approach separates
concerns:

1. **Base positioning**: Nyström extension places sample in unsupervised space
2. **Supervision warping**: Encoder learns mapping from unsupervised to supervised

This enables analysis of supervision effects by comparing embeddings at each
stage.

## Landmark Selection

### IDF-Based (Default)

Use all IDF-filtered vocabulary tokens as landmarks:

```lua
train.tokens_landmark = train.tokens
landmark_weights = idf_weights
```

### Chi2-Based

Select tokens by chi2 association with spectral codes:

```lua
local landmark_vocab, chi2_scores = train.tokens:bits_top_chi2(
  train.codes_sup, train.n, n_top_v, train.dims_sup,
  cfg.landmark_selection.max_vocab)
tok:restrict(landmark_vocab)
train.tokens_landmark = tok:tokenize(train.problems)
```

### Coherence-Based

Select tokens by code coherence (consistency of codes among documents
containing the token):

```lua
local landmark_vocab, coherence_scores = train.tokens:bits_top_coherence(
  train.codes_sup, train.n, n_top_v, train.dims_sup,
  cfg.landmark_selection.max_vocab, cfg.landmark_selection.lambda)
```

## Landmark Filtering

Optional cluster-based authority selection reduces landmark set:

### Process

1. Cluster training samples using spectral codes
2. Compute authority scores (code consistency with token-space neighbors)
3. Select top-k samples per cluster as landmarks

### Configuration

```lua
landmark_filter = {
  enabled = true,
  k_neighbors = 8,
  k_per_cluster = 400,
  use_class_labels = true,  -- Use categories as clusters
  random_select = false,     -- Random vs authority-based
}
```

## Evaluation

### Search Metric

Validate encoder by predicting codes for held-out samples and measuring
retrieval quality:

1. Encode validation samples via landmark aggregation
2. Predict codes with current encoder
3. Index predicted codes in ANN structure
4. Build ground truth adjacency from category labels
5. Score retrieval via NDCG or other ranking metric

### Final Evaluation

After training, evaluate on train/validation/test splits:

```lua
local train_pred = encoder:predict(train_landmark_input, train.n)
local test_pred = encoder:predict(test_landmark_input, test.n)
```

## Comparison with Raw Approach

| Aspect | HLTH (Landmarks) | STH (Raw) |
|--------|------------------|-----------|
| Feature space | Shared | Per-dimension |
| Input type | Aggregated codes | Raw tokens/pixels |
| Feature selection | IDF/chi2 global | Chi2 per bit |
| Memory | Single automata | Per-dim automata |
| Complexity | O(k × n_thresholds) features | O(n_dims × k) features |
| Best for | Dense neighborhood patterns | Sparse high-dim features |
| Lookup modes | Token or unsupervised | Token only |

## Reference Pipeline

### Training

1. Build k-NN graph with category + feature edges
2. Extract supervised spectral codes
3. (Optional) Extract unsupervised spectral codes for Nyström lookup
4. Select landmark vocabulary (IDF, chi2, or coherence)
5. Build landmark index
6. (Optional) Filter landmarks by cluster authority
7. Create landmark encoder (concat or frequency mode)
8. Train Tsetlin encoder with hyperparameter search
9. Validate on held-out set using retrieval metrics

### Inference

1. Tokenize/featurize new sample
2. (If unsupervised lookup) Apply Nyström extension
3. Retrieve k-nearest landmarks
4. Aggregate landmark codes via mode
5. Pack with flip interleave
6. Predict with trained encoder
7. Index or compare predicted codes

## Module Reference

| Module | Purpose |
|--------|---------|
| inv | Inverted index for landmark retrieval |
| ann | ANN index for binary code lookup |
| hlth | Landmark encoder creation |
| tsetlin | Tsetlin machine encoder |
| optimize | Hyperparameter search for encoder |
| eval | Retrieval evaluation metrics |
| itq | Thresholding for Nyström extension |
