# Self-Taught Hashing (STH)

Out-of-sample extension via direct feature-to-code prediction with per-dimension
chi2 feature selection.

## Overview

Self-Taught Hashing learns binary classifiers that predict spectral hash bits
from observable features. Rather than using a shared global feature set, each
output dimension selects its own optimal feature subset using chi2 association.

### Process

1. Compute spectral codes for training data via graph decomposition
2. For each output dimension h, select top-k features by chi2 association
3. Union selected features across dimensions for vocabulary restriction
4. Individualize feature spaces per dimension
5. Train Tsetlin machine encoder on individualized features
6. Predict codes for new samples using learned encoder

## Per-Dimension Feature Selection

Standard feature selection chooses a single global vocabulary:

```
vocab, scores = tokens:bits_top_chi2(codes, n, n_visible, n_hidden, top_k)
tok:restrict(vocab)
toks = tok:tokenize(docs)
sentences = toks:bits_to_cvec(n, k, true)  -- All dims share k features
```

Individualized selection chooses features per output dimension:

```
ids_union, offsets, ids = tokens:bits_top_chi2_ind(codes, n, n_visible, n_hidden, top_k)
tok:restrict(ids_union)
toks = tok:tokenize(docs)
ind_toks, ind_offsets = toks:bits_individualize(offsets, ids, union_size)
bitmap, dim_offsets = ind_toks:bits_to_cvec_ind(ind_offsets, offsets, n, true)
```

### Data Structures

**ids_union**: Union of all per-dimension feature IDs (deduplicated)

**offsets**: CSR-style offsets into ids array: `[0, k_0, k_0+k_1, ...]`
- `offsets[h]` to `offsets[h+1]` gives feature indices for dimension h
- `offsets[n_hidden]` gives total feature count

**ids**: Concatenated per-dimension feature IDs (indices into ids_union)

**dim_offsets**: Byte offsets into the packed bitmap for each dimension
- Each dimension h has `ceil(2*k_h/8)` bytes per sample (flip-interleaved)
- `dim_offsets[h]` gives byte offset to dimension h's data

### Chi2 Feature Selection

For each output dimension h (bit position):
1. Treat bit h as binary label (0 or 1) across training samples
2. Compute chi2 statistic for each feature against this label
3. Select top-k features with highest chi2 scores
4. Store selected feature IDs in `ids[offsets[h]:offsets[h+1]]`

Chi2 measures association between feature presence and bit value:

    chi2 = Σ (observed - expected)² / expected

High chi2 indicates the feature is predictive of the bit.

### Flip Interleave

Input features use flip-interleaved encoding for Tsetlin machines:

For k_h features in dimension h:
- Bits 0 to k_h-1: Feature present (1) or absent (0)
- Bits k_h to 2*k_h-1: Feature absent (1) or present (0)

This doubles the feature space but provides explicit absence signals.

## Tsetlin Machine Integration

### Individualized Encoder Structure

Each output dimension h has its own automata sized to its k_h features:

```
tm->feat_sizes[h] = k_h
tm->input_chunks[h] = ceil(2*k_h / 8)  -- flip-interleaved
```

State and actions remain contiguous with per-dimension offsets:

```
state buffer:  [dim0_state][dim1_state]...[dim_{n-1}_state]
actions buffer: [dim0_actions][dim1_actions]...[dim_{n-1}_actions]
```

### Training

For each training iteration:
1. Shuffle sample order
2. For each dimension h in parallel:
   - Extract input from `sentences[dim_offsets[h] + sample*input_chunks[h]]`
   - Get target bit from codes
   - Update automata using standard Tsetlin update rules

### Prediction

For new sample x:
1. Tokenize and select features from restricted vocabulary
2. Individualize features into per-dimension spaces
3. Pack into flip-interleaved bitmap
4. For each dimension h:
   - Sum clause votes using dimension-specific automata
   - Set output bit based on vote sign

## Parameters

### Feature Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_vocab | int | 12288 | Features per dimension (k) |
| selection | string | chi2 | Selection method: chi2 or mi |
| individualized | bool | true | Per-dimension feature selection |

### Tsetlin Machine

| Parameter | Type | Description |
|-----------|------|-------------|
| clauses | range | Clause count per dimension |
| clause_tolerance | range | Tolerance threshold |
| clause_maximum | range | Maximum threshold |
| target | range | Target activation |
| specificity | range | Feature specificity |
| include_bits | range | Bits to include in clauses |

### Optimization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| search_rounds | int | 6 | Hyperparameter search rounds |
| search_trials | int | 10 | Trials per round |
| search_iterations | int | 20 | Training iterations per trial |
| final_iterations | int | 400 | Final training iterations |

## Evaluation

### Search Metric

Validate encoder by predicting codes for held-out samples and measuring
retrieval quality:

1. Predict codes for validation set
2. Index predicted codes in ANN structure
3. Build ground truth adjacency from category labels
4. Score retrieval via NDCG or other ranking metric

### Final Evaluation

After training, evaluate on train/validation/test splits:

```lua
local train_pred = encoder:predict(train_input, train_dim_offsets, train.n)
local test_pred = encoder:predict(test_input, test_dim_offsets, test.n)
local train_acc = eval.encoding_accuracy(train_pred, train_codes, train.n, n_dims)
```

## Comparison with Landmark Approach

| Aspect | STH (Raw) | HLTH (Landmarks) |
|--------|-----------|------------------|
| Feature space | Per-dimension | Shared |
| Input type | Raw tokens/pixels | Aggregated codes |
| Feature selection | Chi2 per bit | IDF/chi2 global |
| Memory | Per-dim automata | Single automata |
| Complexity | O(n_dims × k) features | O(k × n_thresholds) features |
| Best for | Sparse high-dim features | Dense neighborhood patterns |

## Reference Pipeline

### Training

1. Build k-NN graph with category + feature edges
2. Extract spectral codes via Laplacian decomposition
3. Apply ITQ/Otsu thresholding to produce binary codes
4. Run per-dimension chi2 feature selection
5. Union and restrict vocabulary
6. Individualize training samples
7. Train Tsetlin encoder with hyperparameter search
8. Validate on held-out set using retrieval metrics

### Inference

1. Tokenize/featurize new sample
2. Select features from restricted vocabulary
3. Individualize into per-dimension spaces
4. Pack with flip interleave
5. Predict with trained encoder
6. Index or compare predicted codes

## Module Reference

| Module | Purpose |
|--------|---------|
| ivec | Sparse feature vectors with chi2/MI selection |
| cvec | Packed binary vectors with flip interleave |
| tsetlin | Individualized Tsetlin encoder |
| optimize | Hyperparameter search for encoder |
| eval | Retrieval evaluation metrics |
