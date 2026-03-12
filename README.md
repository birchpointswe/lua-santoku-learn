# Santoku Learn

A Lua/C library for classification, regression, and retrieval built on
spectral embedding and ridge regression for extreme multi-label
classification (XMLC).

## Components

**Spectral Embedding** — Kernel spectral embedding via RP-Cholesky
factorization. Supports cosine, arccos, and Hellinger kernels over
sparse, dense, or binary input. Float32 inference pipeline.
([doc/spectral.md](doc/spectral.md))

**Ridge Regression** — Centered ridge with intercept, propensity
weighting for tail labels, and precomputed gram matrix for fast
hyperparameter search. Dense regression and sparse label prediction
modes. ([doc/optimize.md](doc/optimize.md))

**Hyperparameter Search** — GP-BO search with LHS initialization,
cost-cooled Expected Improvement, and Latin Hypercube candidate
generation. ([doc/optimize.md](doc/optimize.md))

**Binary Quantization and ANN** — Multi-index hashing for
Hamming-distance nearest neighbor retrieval with optional float
reranking. ([doc/ann.md](doc/ann.md))

**Evaluation** — Classification, regression, ranking (NDCG, Spearman,
Pearson), and retrieval (P/R/F1 with optimal-k) metrics. Graph-
constrained agglomerative clustering.
([doc/evaluation.md](doc/evaluation.md))

**GFM** — General F-Measure optimization for per-label threshold
selection.

## Usage Patterns

The library supports pipeline patterns from dense regression to full
XMLC with spectral embedding, ANN shortlisting, ridge classification,
and GFM thresholding.
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
