# Kernel Spectral Embedding

## Overview

`spectral.c` produces continuous d-dimensional embeddings via RP-Cholesky
factorization of a kernel matrix defined over sparse, dense, or binary
input. The module computes kernel similarities directly between samples
using one of four kernel functions.

Two Lua functions: `encode` (build embeddings and return a reusable
encoder) and `load` (restore a persisted encoder).

## Kernel Functions

| Kernel | Formula | Notes |
|---|---|---|
| `cosine` | `dot(a,b) / (‖a‖·‖b‖)` | Default. Clamped to [-1,1]. |
| `arccos0` | `1 - acos(cosine(a,b)) / π` | Arc-cosine order 0. |
| `arccos1` | `‖a‖·‖b‖·(sin(θ) + (π-θ)·cos(θ)) / π` | Arc-cosine order 1. |
| `hellinger` | `dot(√|a|, √|b|)` | Input values sqrt-transformed before dot product. No norm division. |

For CSR and dense modalities, per-sample L2 norms are precomputed and
used in the denominator.

## Input Modalities

Exactly one modality per call:

**CSR** (`offsets`, `tokens`, `n_tokens` required; `values` optional):
Sparse token vectors. Values default to 1.0 if omitted. Tokens are
sorted per sample for CSC construction. Accepts `fvec` or `dvec` values.

**Dense** (`codes` required; `d_input` optional): Continuous dvec. If
`d_input` is omitted, inferred from `codes.n / n_samples`.

**Bits** (`bits`, `d_bits` required): Binary cvec. Kernel similarity is
`1 - hamming / d_bits`.

## RP-Cholesky Factorization

`tk_spectral_sample_landmarks` implements randomly pivoted Cholesky
(Chen, Epperly, Tropp, Webber 2023) with blocked processing
(`TK_CHOL_BLOCK = 64`).

### Algorithm

Initialize residual `d[i] = k(i,i)` for all n samples and
`initial_trace = sum(d)`.

For each block of landmarks:

1. Compute `trace = sum(d[i] for d[i] > 0)`. Stop if
   `trace < trace_tol * initial_trace`.
2. Sample `block_size` pivots proportional to residual with rejection
   sampling (reject if `d[pivot] ≤ 0` or already selected).
3. For each pivot in the block:
   a. Compute kernel column `g[i] = k(i, pivot)` for all samples.
      Parallelized with OpenMP.
   b. Orthogonalize against all prior columns via `cblas_dgemv`.
   c. Normalize: `L[i, j] = g[i] / sqrt(g[pivot])`.
   d. Update residual: `d[i] -= L[i, j]²`, clamped to zero.

### Output

- `landmark_ids` (ivec): indices of selected pivot samples.
- `L_mat` (col-major double matrix, n × m): the full Cholesky factor.
- `actual_landmarks`: may be less than requested if trace converged.
- `trace_ratio`: remaining trace / initial trace.

## Training Embeddings

After RP-Cholesky:

1. Compute projection matrix: `projection = inv(L_lm^T)` where `L_lm`
   is the m×m landmark submatrix, via `cblas_dtrsm`. Stored as float.
2. Transpose `L_mat` from col-major to row-major float `train_codes`
   (n × m fvec). Each sample's row is its m-dimensional embedding.
3. The encoder stores projection (float) and landmark data for inference.

Return values: `(fvec train_codes, encoder)`.

Note: `d = m` — the embedding dimensionality equals the number of
landmarks. There is no eigendecomposition or dimensionality reduction
step.

## Encoder Object

`tk_nystrom_encoder_t` stores:

| Field | Content |
|---|---|
| `projection` | m×d float matrix (inv(L_lm^T)) |
| `m` | number of landmarks |
| `d` | embedding dimensionality (= m) |
| `kernel` | kernel type |
| `mod_type` | modality (CSR, dense, or bits) |
| `trace_ratio` | approximation quality |
| landmark data | modality-specific landmark vectors |

For CSR modality, landmark data is stored as both CSR and CSC (for fast
inference via sparse dot products).

### Methods

- `encoder:encode({table})` — encode new samples. Table has
  `n_samples` plus modality-specific fields matching the training call.
  Optional `output` fvec for reusable buffer.

- `encoder:dims()` — returns d.

- `encoder:n_landmarks()` — returns m.

- `encoder:landmark_ids()` — returns the landmark index ivec.

- `encoder:trace_ratio()` — returns remaining/initial trace ratio.

- `encoder:restrict(keep_ivec)` — keep only specified dimensions.
  Reindexes projection matrix in-place.

- `encoder:shrink()` — release projection matrix to free memory.

- `encoder:persist(path_or_true)` — serialize to file or string.

### Inference

`encoder:encode({n_samples, ...})` computes embeddings in tiles (capped
at 256MB working memory):

1. For each tile of samples:
   a. Compute kernel similarities to all m landmarks → float sims
      matrix (tile × m).
   b. Multiply `sims × projection → output` via `cblas_sgemm`.

CSR inference uses CSC landmark representation for fast sparse dot
products. Dense inference uses `cblas_sgemm` for the similarity
computation. Bits inference computes Hamming similarities directly.

### Persist Format

Magic `TKny`, version 21. Writes:
- kernel (uint8), mod_type (uint8), m (uint64), d (uint64),
  trace_ratio (double), projection (m×d floats)
- Modality-specific landmark data
- landmark_ids (ivec)

Loads versions 19, 20, and 21.

## Parameters

| Parameter | Role | Default |
|---|---|---|
| `n_samples` | Number of input samples | required |
| `n_landmarks` | Maximum RP-Cholesky pivots (0 = all) | 0 |
| `trace_tol` | Early stopping threshold on trace ratio | 0.0 |
| `kernel` | `"cosine"`, `"arccos0"`, `"arccos1"`, `"hellinger"` | `"cosine"` |
| `offsets` | CSR offsets (ivec) | modality-specific |
| `tokens` | CSR tokens (ivec) | modality-specific |
| `n_tokens` | Token vocabulary size | modality-specific |
| `values` | CSR values (fvec or dvec, optional) | 1.0 |
| `codes` | Dense embeddings (dvec) | modality-specific |
| `d_input` | Dense input dimensionality | inferred |
| `bits` | Binary codes (cvec) | modality-specific |
| `d_bits` | Number of bits per code | modality-specific |
