# Santoku Learn

A Lua/C library for classification, regression, and retrieval built on
Tsetlin Machines with a spectral embedding pipeline for extreme
multi-label classification (XMLC).

## Components

**Tsetlin Machine** — Regression TM with bit-planar state
representation, soft clause evaluation, per-chunk squared-error
feedback, and sparse mode with per-class token mapping and
absorption-driven feature streaming. A separate SSL (Stochastic
Searching on the Line) path handles continuous-feature regression.
([doc/regressor.md](doc/regressor.md))

**Hyperparameter Search** — Round-based adaptive search with the 1/5th
success rule for jitter adaptation, deduplication, size preference at
equal accuracy, and batched early stopping for final training.
([doc/optimize.md](doc/optimize.md))

**Spectral Embedding** — Nystrom approximation via RPCholesky landmark
selection on a rank-weighted cosine similarity kernel defined over
sparse binary features. Supports asymmetric mode for cross-population
embedding (e.g., documents into label eigenspace).
([doc/pca.md](doc/pca.md))

**Binary Quantization and ANN** — SFBS (Sequential Forward Bit
Selection) quantizes continuous or binary embeddings into compact codes
optimizing NDCG against a ground-truth neighbor structure. Multi-index
hashing provides Hamming-distance nearest neighbor retrieval with
early termination. ([doc/ann.md](doc/ann.md))

**Evaluation and Bit Selection** — Classification, regression, ranking
(NDCG, Spearman, Pearson), and retrieval (P/R/F1 with optimal-k)
metrics. Greedy SFBS bit/dimension selection. Graph-constrained
agglomerative clustering with complete-linkage distance and
majority-vote centroids. ([doc/evaluation.md](doc/evaluation.md))

## Usage Patterns

The library supports five pipeline patterns from simple dense
classification to full XMLC with bipartite spectral embedding, two-round
SFBS quantization, and Hamming ANN retrieval.
([doc/usage.md](doc/usage.md))

## License

MIT License

Copyright 2025 Birch Point SWE

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
