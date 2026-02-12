# Nystrom Spectral Embedding

## Overview

`spectral.c` produces continuous d-dimensional embeddings from a
rank-weighted cosine similarity kernel defined over sparse binary feature
vectors. The pipeline is: RPCholesky landmark selection, centered
eigendecomposition, Nystrom projection. An optional asymmetric mode uses
one inverted index for landmarks and another for projection, enabling
cross-population embedding.

The module exposes two Lua functions: `encode` (build embeddings and
return a reusable encoder) and `load` (restore a persisted encoder).

## Kernel Function

The kernel is defined by the inverted index (`inv.h`). Each sample is a
set of binary features with per-feature weights and per-feature rank
assignments. Similarity between two samples is:

    rank_sim[r] = intersection_weight[r] / sqrt(query_weight[r] * entry_weight[r])

per rank, where intersection/query/entry weights are sums of feature
weights over features present in both/first/second sample at rank r.
This is cosine similarity restricted to each rank tier.

Ranks are combined via exponentially decaying weights:

    avg_sim = sum_r(exp(-r * decay) * rank_sim[r]) / sum_r(exp(-r * decay))

When `bandwidth >= 0`, the result is RBF-transformed:

    k(a, b) = exp(-(1 - avg_sim) * bandwidth)

When `bandwidth < 0` (default -1), raw `avg_sim` is returned.

Self-similarity is not always 1.0: ranks with no features contribute 0
to the numerator but still contribute their weight to the denominator.
The residual diagonal in RPCholesky reflects this.

## RPCholesky Landmark Selection

`tk_spectral_sample_landmarks` implements the randomly pivoted Cholesky
algorithm (Chen, Epperly, Tropp, Webber 2023). It produces a low-rank
factorization K ~ FF^T using adaptive column sampling.

### Algorithm

Initialize residual `d[i] = k(i, i)` for all n documents and
`initial_trace = sum(d)`.

For each landmark j = 0..m-1:

1. Compute `trace = sum(d[i] for d[i] > 0)`. Stop if
   `trace < trace_tol * initial_trace`.
2. Sample pivot index proportional to residual:
   `P{pivot = i} = d[i] / trace`.
3. Evaluate kernel column: `g[i] = k(i, pivot)` for all i. This is the
   dominant cost: n kernel evaluations per landmark.
4. Orthogonalize: `g[i] -= dot(F[i, 0:j-1], F[pivot, 0:j-1])` via
   `cblas_ddot`.
5. Normalize: `F[i, j] = g[i] / sqrt(d[pivot])`.
6. Update residual: `d[i] -= F[i, j]^2`, clamped to zero.

The kernel column evaluation uses `tk_inv_similarity_fast_cached`, which
takes precomputed per-node weight-by-rank sums for query and entry norms
and only computes intersection weights fresh. The main loop is
parallelized with OpenMP (`omp for schedule(guided)` on steps 3-6, `omp
single` on steps 1-2).

### Output

- `landmark_ids` (ivec, m): UIDs of selected pivot documents.
- `chol` (dvec, m*m): F restricted to landmark rows. This is the
  Cholesky factor L of K_mm (the landmark-landmark kernel matrix,
  K_mm = L L^T).
- `full_chol` (dvec, n*m): the full factor F for all n documents.
- `full_chol_ids` (ivec, n): UIDs in the row order of full_chol.
- `actual_landmarks`: may be less than requested if trace converged.
- `trace_ratio`: remaining trace / initial trace after all landmarks.

Total kernel evaluations: n per landmark plus n for the initial
diagonal, so (m+1)*n. Additional arithmetic: O(m^2 * n) from the
orthogonalization dot products.

## Eigendecomposition

`tm_encode` takes the RPCholesky output and produces d-dimensional
embeddings via centered kernel PCA.

### Centering

The m*m landmark Cholesky factor (chol) is column-mean-centered:

    cmeans[j] = mean of chol[:, j]       (m landmarks)
    cw[i, j]  = chol[i, j] - cmeans[j]

This corresponds to centering the kernel matrix. Centering is computed
only on the m landmark rows; the full n*m factor is centered implicitly
during projection by subtracting the projected mean.

### Gram matrix and eigendecomposition

    gram = cw^T * cw    (m*m, via cblas_dgemm)

`LAPACKE_dsyevr` computes the top-d eigenpairs of gram (eigenvalue
indices m-d+1 through m, reversed to descending order). The eigenvectors
V (m*d) define directions in the Cholesky factor space.

### Projection

Training embeddings:

    ccodes = full_chol * V    (n*d, via cblas_dgemm)

This is the kernel PCA embedding E = FV. The eigenvalues cancel:
the standard kernel PCA formula U * Lambda^{1/2} reduces to FV because
U = F * V * Lambda^{-1/2} and U * Lambda^{1/2} = F * V.

Centering adjustment:

    adjustment = V^T * cmeans    (d-vector, via cblas_dgemv)
    ccodes[i] -= adjustment      for all i

This is equivalent to centering the full n*m factor before projection,
but avoids materializing the centered n*m matrix.

### Out-of-sample projection matrix

    projection = L^{-T} * V    (m*d, via cblas_dtrsm)

where L is the m*m landmark Cholesky factor. This precomputes the matrix
needed for Nystrom extension: for a new sample with kernel vector
k = [k(x, l_1), ..., k(x, l_m)]^T, the embedding is:

    embedding(x) = projection^T * k - adjustment

which is one matrix-vector multiply (m*d gemv) plus a d-vector subtract.

## Asymmetric Mode

When `landmarks_inv` is provided in the arguments table and differs from
`inv`, the pipeline runs RPCholesky on `landmarks_inv` (selecting
landmarks and computing the Cholesky factor from that population's
kernel) but projects documents from `inv`.

Landmarks are looked up in `inv` by UID. Documents whose UIDs appear in
the Cholesky output reuse their precomputed embeddings directly.
Documents not present in the landmark population are encoded via
cross-similarity: their kernel vector against the landmark set is
computed using `inv`'s kernel function, then projected through the
precomputed projection matrix.

This enables embedding one population (e.g., documents) into the
eigenspace of another population (e.g., labels or a bipartite graph's
node set). The landmark kernel defines the coordinate system; the cross-
similarity defines how new samples project into it.

## Encoder Object

`tm_encode` returns three values: the training embeddings (dvec, n*d),
the corresponding UIDs (ivec, n), and a `tk_nystrom_encoder_t` userdata.

The encoder stores:

| Field | Content |
|---|---|
| `projection` | m*d matrix (L^{-T} V) |
| `adjustment` | d-vector (V^T cmeans) |
| `lm_sids` | m landmark SIDs in the feature inv |
| `m` | number of landmarks |
| `d` | embedding dimensionality |
| `bandwidth` | kernel bandwidth used |
| `decay` | kernel decay used |
| `trace_ratio` | approximation quality |

The encoder holds a reference to the feature inv (via fenv) for kernel
evaluation at encode time.

### Methods

- `encoder:encode(sparse_bits, n_samples, n_features)` — encode new
  samples. Input is a flat ivec of `sample_index * n_features + feature_id`
  packed entries (same format as inv:get with multiple UIDs). For each
  sample: binary-search to extract its features, partition by rank,
  compute similarity to each landmark, project via gemv, subtract
  adjustment. Parallelized with OpenMP (dynamic scheduling, chunk 16).

- `encoder:dims()` — returns d.

- `encoder:n_landmarks()` — returns m.

- `encoder:landmark_ids()` — returns the landmark UID ivec.

- `encoder:trace_ratio()` — returns remaining/initial trace ratio.

- `encoder:persist(path_or_true)` — serialize to file or string. Format:
  magic `TKny`, version byte (1), m, d, bandwidth, decay, trace_ratio,
  projection (m*d doubles), adjustment (d doubles), lm_sids (m int64s),
  landmark_ids ivec.

### Loading

`spectral.load(path_or_data, inv, is_string)` restores a persisted
encoder. The inv argument provides the feature index for subsequent
encode calls.

## Parameters

| Parameter | Role | Default |
|---|---|---|
| `n_landmarks` | Maximum RPCholesky pivots | 0 (= all documents) |
| `n_dims` | Embedding dimensionality (capped at actual landmarks) | required |
| `decay` | Rank weight decay (0 = equal ranks) | 0.0 |
| `bandwidth` | RBF kernel bandwidth (-1 = raw cosine) | -1.0 |
| `trace_tol` | Early stopping: stop when trace ratio falls below this | 1e-15 |
| `inv` | Feature inverted index (required) | — |
| `landmarks_inv` | Landmark inverted index (optional, defaults to inv) | inv |
