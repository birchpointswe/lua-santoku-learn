# Self-Taught Hashing (STH)

Out-of-sample extension via direct feature-to-code prediction with chi2 feature
selection.

## Overview

Self-Taught Hashing learns binary classifiers that predict spectral hash bits
from observable features. Features are selected using chi2 association to
identify the most predictive inputs.

### Process

1. Compute spectral codes for training data via graph decomposition
2. Select top-k features by chi2 association across all output dimensions
3. Restrict vocabulary to selected features
4. Train Tsetlin machine encoder on selected features
5. Predict codes for new samples using learned encoder

## Feature Selection

Feature selection chooses a global vocabulary based on chi2 scores:

```
vocab, scores = tokens:bits_top_chi2(codes, n, n_visible, n_hidden, top_k)
tok:restrict(vocab)
toks = tok:tokenize(docs)
sentences = toks:bits_to_cvec(n, k, true)  -- All dims share k features
```

### Chi2 Feature Selection

Chi2 measures association between feature presence and bit values:

    chi2 = Σ (observed - expected)² / expected

High chi2 indicates the feature is predictive of hash bits.

### Flip Interleave

Input features use flip-interleaved encoding for Tsetlin machines:

For k features:
- Bits 0 to k-1: Feature present (1) or absent (0)
- Bits k to 2k-1: Feature absent (1) or present (0)

This doubles the feature space but provides explicit absence signals.

## Tsetlin Machine Integration

### Encoder Structure

The encoder uses shared automata across all output dimensions:

```
tm->features = k
tm->input_chunks = ceil(2*k / 8)  -- flip-interleaved
```

### Training

For each training iteration:
1. Shuffle sample order
2. For each sample:
   - Extract input features
   - Get target bits from codes
   - Update automata using standard Tsetlin update rules

### Prediction

For new sample x:
1. Tokenize and select features from restricted vocabulary
2. Pack into flip-interleaved bitmap
3. Sum clause votes and set output bits based on vote sign

## Parameters

### Feature Selection

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_vocab | int | 12288 | Features to select (k) |
| selection | string | chi2 | Selection method: chi2 or mi |

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
local train_pred = encoder:predict(train_input, train.n)
local test_pred = encoder:predict(test_input, test.n)
local train_acc = eval.encoding_accuracy(train_pred, train_codes, train.n, n_dims)
```

## Reference Pipeline

### Training

1. Build k-NN graph with category + feature edges
2. Extract spectral codes via Laplacian decomposition
3. Apply ITQ/Otsu thresholding to produce binary codes
4. Run chi2 feature selection
5. Restrict vocabulary
6. Train Tsetlin encoder with hyperparameter search
7. Validate on held-out set using retrieval metrics

### Inference

1. Tokenize/featurize new sample
2. Select features from restricted vocabulary
3. Pack with flip interleave
4. Predict with trained encoder
5. Index or compare predicted codes

## Module Reference

| Module | Purpose |
|--------|---------|
| ivec | Sparse feature vectors with chi2/MI selection |
| cvec | Packed binary vectors with flip interleave |
| tsetlin | Tsetlin encoder |
| optimize | Hyperparameter search for encoder |
| eval | Retrieval evaluation metrics |
