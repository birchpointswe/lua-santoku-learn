# Multi-Index Hashing ANN

## Overview

`ann.h` implements multi-index hashing (MIH) for Hamming-distance
nearest neighbor retrieval over binary codes, with optional float
reranking via dot product on continuous codes.

## Index Structure

`tk_ann_flat_t` stores:

| Field | Content |
|---|---|
| `N` | number of indexed vectors |
| `features` | bits per code |
| `m` | number of hash tables = `ceil(features / 16)` |
| `data` | pointer to packed binary codes (cvec) |
| `bytes_per_vec` | bytes per binary vector |
| `sorted_sids` | per-table sorted sample arrays |
| `bucket_off` | per-table bucket offset arrays |
| `codes` | optional float codes for reranking (fvec) |
| `n_dims` | dimensionality of float codes |

Each hash table indexes a 16-bit substring of the binary code. Table
`ti` covers bits `[ti*16, ti*16+16)` (the last table may cover fewer).
Bucket offsets are precomputed via counting sort for O(1) bucket lookup.

## Creation

```lua
local mih = ann.create({
  data = binary_cvec,
  features = n_bits,
  codes = float_fvec,   -- optional, for reranking
  n_dims = d,           -- required if codes provided
})
```

The `data` cvec must contain `N * ceil(features/8)` bytes of packed
binary codes. The `codes` fvec, if provided, must contain `N * n_dims`
floats for dot-product reranking.

## Search Algorithm

`tk_ann_flat_query` probes hash tables at increasing Hamming radii:

For radius r on table ti with query substring h:

1. Enumerate all `C(sub_bits, r)` bit-flip masks via combinatorial
   iteration.
2. For each mask: `probe_h = h ^ mask`. Look up the bucket via
   `bucket_off`.
3. For each sample in the bucket (skipping self if applicable and
   already-seen samples): compute full Hamming distance via
   `tk_cvec_bits_hamming_serial`, insert into max-heap of k results.

Outer loop: `r = 0, 1, ..., max_radius`, probing all m tables at each
radius before advancing.

### Early Termination

After processing all tables at radius r, if k results found and the
worst result has distance `< m * (r + 1)`, search stops. An unseen code
must differ by more than r bits in every substring.

## Methods

### `mih:neighborhoods(k [, do_rerank [, radius]])`

All-vs-all query. Queries every indexed code against all others (self
excluded).

- `k`: number of nearest neighbors per query.
- `do_rerank`: boolean, defaults to true if float codes provided at
  creation. When true, Hamming candidates are reranked by descending
  `cblas_sdot` score using float codes.
- `radius`: maximum probe radius per substring (default 3).

Returns `(offsets_ivec, neighbors_ivec, weights_dvec)`.

### `mih:neighborhoods_by_vecs(query_cvec, k [, query_fvec [, radius]])`

Query external binary codes against the index.

- `query_cvec`: packed binary query vectors.
- `k`: neighbors per query.
- `query_fvec`: optional float codes for reranking against indexed
  float codes.
- `radius`: maximum probe radius (default 3).

Returns `(offsets_ivec, neighbors_ivec, weights_dvec)`.

## Output Format

All query methods return CSR-style output:

- `offsets` (ivec, nq+1): start/end positions per query.
- `neighbors` (ivec): 0-based sample indices of neighbors.
- `weights` (dvec): similarity scores. When reranking, these are
  `cblas_sdot` values. Without reranking, these are
  `1 - hamming / features`.

Neighbors are sorted ascending by Hamming distance (no rerank) or
descending by dot product score (rerank).
