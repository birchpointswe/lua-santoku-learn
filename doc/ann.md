# Binary Quantization and ANN Indexing

## Overview

`quantizer.c` converts continuous or binary embeddings into compact binary
codes. `ann.h` indexes those codes for fast Hamming-distance nearest
neighbor retrieval via multi-index hashing.

The quantizer provides three paths: SFBS (Sequential Forward Bit Selection)
from continuous embeddings, SFBS from existing binary codes, and
thermometer encoding. All three produce a `tk_sfbs_encoder_t` that maps
input to binary codes. The ANN index stores binary codes in substring hash
tables and answers k-nearest-neighbor queries by probing at increasing
Hamming radii.

## SFBS Quantization

### From Continuous Embeddings

`quantizer.create` with `raw_codes` (dvec) and ground-truth neighbor
structure. Each output bit is a threshold test on one input dimension:
bit k is 1 iff `x[dim[k]] > threshold[k]`.

### Candidate Generation

Initial candidates: one per input dimension, threshold = median of that
dimension's values across all samples. The candidate pool is stored as
per-sample bitmaps (`cand_bins[c]`, one bit per sample indicating
above/below threshold).

When a candidate is selected, `tk_sfbs_spawn_children` generates two
child candidates on the same dimension: one at the median of values below
the parent threshold, one at the median of values above. This creates a
binary-tree refinement of each dimension. Maximum candidates is
`n_dims + 16 * target_bits`.

`max_dims` limits the number of distinct source dimensions used. A
candidate from a new dimension is skipped if `n_used_dims >= max_dims`.
Dimension reference counts track how many selected bits use each dimension.

### Scoring

Each candidate is scored by mean NDCG across query nodes. The ground
truth is a weighted neighbor adjacency: `expected_ids`, `expected_offsets`,
`expected_neighbors`, `expected_weights`.

For each query node with neighbors at offsets `[st, en)`:

    node_score = DCG(distance_ranking) / IDCG

where DCG uses the standard `1/log2(rank+1)` discount. Neighbors are
bucketed by current Hamming distance (distance = number of selected bits
that differ). Bucket weights are sums of neighbor weights within each
distance tier. This avoids sorting: `tk_sfbs_dcg` walks buckets in
distance order, accumulating discount-weighted sums.

IDCG is precomputed per node by sorting neighbor weights descending and
applying the same discount.

Distance tracking is incremental: `tk_sfbs_commit` updates the pairwise
distance array by XORing the candidate's per-sample bits for each pair.
Adding a candidate increments distances where the pair disagrees on that
bit; removing decrements.

### Three-Phase Selection

Phase 0 (add): Score all unselected candidates. If the best improves over
`current_score`, select it, commit distances, spawn children. Repeat.
If no improvement, move to phase 1.

Phase 1 (remove): Score all selected candidates for removal (subtract
their contribution). If the best-after-removal improves the score, remove
it and return to phase 0. If removal doesn't improve but a swap might
help, tentatively remove the weakest selected candidate and move to
phase 2.

Phase 2 (swap-add): Score all unselected candidates as replacements. If
the best exceeds `pre_swap_score`, accept the swap (commit add, spawn
children, return to phase 0). Otherwise, restore the removed candidate
and terminate.

The parallel structure uses OpenMP: candidate scoring is parallelized
across query nodes (`omp for schedule(static)` with reduction on score),
while phase transitions run in `omp single` blocks.

### Pruning Pass

After the three-phase selection, the selected bits are re-encoded into a
compact binary matrix and passed through `tk_sfbs_binary_select` as a
second pass. This removes redundant bits whose information is captured by
other selected bits in combination. The pruning is accepted only if the
resulting score is within `tolerance` of the first-pass score.

### From Binary Codes

`quantizer.create` with `codes` (cvec) instead of `raw_codes`. Selects
a subset of input bits that best preserves the neighbor ranking.

`tk_sfbs_binary_select` precomputes XOR codes: for each neighbor pair,
`xor_codes[j] = codes[a[j]] ^ codes[b[j]]`. The XOR bit at position b
is 1 iff the pair disagrees on bit b, so distance updates are single-bit
lookups via `tk_sfbs_commit_xor` instead of per-sample bitmap walks.

The same three-phase loop applies: add bits that improve NDCG, try
removing, try swapping. The scoring is parallelized identically with
per-node NDCG reduction.

A second pruning pass follows, operating on the re-encoded selected bits.

## ITQ Quantization

`quantizer.create` with `mode = "itq"`. Iterative Quantization produces
binary codes via centered alternating sign+SVD rotation.

Algorithm:
1. Center input: subtract column means.
2. Alternate for `iterations` rounds (default 50):
   a. Binary codes B = sign(X * R)
   b. SVD of B' * X -> rotation R = V * U'
3. Final codes: sign(X_centered * R)

Uses all input dimensions (k = n_dims). No dimension selection or
weighting -- the answer was always "max out k".

Parameters:
- `raw_codes` (dvec): continuous embeddings
- `n_samples`: number of samples
- `iterations` (optional): rotation iterations (default 50)

Encoder: `encode(raw_dvec)` -> cvec (center -> rotate -> sign).
Methods: `n_bits()`, `persist()`.

Persist format: magic `TKqi` + version 1 + k + rotation (dvec) + means
(dvec). `quantizer.load` dispatches on magic: `TKqi` -> ITQ, `TKqt` ->
SFBS.

## Thermometer Quantization

`quantizer.create` with `mode = "thermometer"`. Produces one bit per
threshold per dimension: bit is 1 iff `x[dim] > threshold`.

When `n_bins = 0`: thresholds are all unique values of each dimension
(sorted ascending). Total bits = sum of unique values across dimensions.

When `n_bins > 0`: `B` evenly-spaced quantile thresholds per dimension.
Threshold b is the value at quantile `(b+1)/(B+1)`. Total bits = `B * n_dims`.

No neighbor-aware optimization. Useful as a baseline or when the input
dimensions are already meaningful and uniform thresholding suffices.

## Encoder Object

`tk_sfbs_encoder_t` stores:

| Field | Content |
|---|---|
| `bit_dims` | ivec of source dimension indices, length n_bits |
| `bit_thresholds` | dvec of thresholds, length n_bits |
| `n_dims` | number of input dimensions |
| `n_bits` | number of output bits |

For continuous input, `encode` tests `x[dims[k]] > thresholds[k]`. For
binary input (cvec), `encode` copies bit `dims[k]` from the source to
output bit k (thresholds are unused, stored as 0.0).

### Methods

- `encoder:encode(input [, out])` -- input is dvec (continuous) or cvec
  (binary). Returns cvec of `n_samples * ceil(n_bits/8)` bytes.
  Parallelized with `omp parallel for schedule(static)`.

- `encoder:n_bits()` -- returns number of output bits.

- `encoder:n_dims()` -- returns number of input dimensions.

- `encoder:dims()` -- returns the bit_dims ivec.

- `encoder:thresholds()` -- returns the bit_thresholds dvec.

- `encoder:used_dims()` -- returns sorted unique ivec of source dimensions
  referenced by the selected bits.

- `encoder:restrict(kept_dims)` -- remaps bit_dims to new indices defined
  by `kept_dims`. Bits referencing dimensions not in `kept_dims` are
  dropped. Updates `n_bits` and `n_dims`. Used when the upstream
  embedding is restricted to fewer dimensions.

- `encoder:restrict_bits(kept_bits)` -- keeps only the bits at the given
  indices. Compacts bit_dims and bit_thresholds in-place.

- `encoder:persist(path_or_true)` -- serialize to file or string.

### Persist Format

Magic `TKqt`, version byte (1), n_dims (uint64), n_bits (uint64),
bit_dims (ivec), bit_thresholds (dvec).

`quantizer.load(path_or_data, is_string)` restores a persisted encoder.

## Multi-Index Hashing

`ann.h` implements multi-index hashing (Norouzi, Punjani, Fleet 2012) for
exact and approximate Hamming-distance search over binary codes.

### Index Structure

`tk_ann_t` stores:

| Field | Content |
|---|---|
| `features` | number of bits per code |
| `m` | number of hash tables = `ceil(features / 16)` |
| `tables` | array of m hash maps (uint32 key -> ivec of SIDs) |
| `vectors` | cvec of all stored binary codes, contiguous |
| `uid_sid` | iumap: user ID -> internal slot ID |
| `sid_to_uid` | ivec: slot ID -> user ID (-1 if deleted) |

Each hash table indexes a 16-bit substring (`TK_ANN_SUBSTR_BITS = 16`)
of the binary code. Table `ti` covers bits `[ti*16, ti*16+16)` (the last
table may cover fewer bits). The hash key is the raw 16-bit value of
that substring, stored as a uint32.

### Indexing

`tk_ann_add(ids, data)` inserts binary codes. For each code:

1. Allocate a slot ID (SID) via `tk_ann_uid_sid(uid, REPLACE)`. If the
   UID already exists, the old SID is marked deleted and a new SID is
   appended.
2. Extract 16-bit substrings via `tk_ann_substring` for each table.
3. Insert the SID into the corresponding bucket (hash map value is an
   ivec of SIDs).
4. Copy the code into the vectors array at the SID's offset.

Deletion (`tk_ann_remove`) marks the SID as deleted in `sid_to_uid` and
removes the UID from `uid_sid`. The SID remains in bucket posting lists
until `shrink` compacts them.

`tk_ann_shrink` compacts the index: reassigns consecutive SIDs to active
entries, updates vectors, posting lists, and both UID/SID maps. Called
periodically to reclaim space after deletions.

### Search Algorithm

`tk_ann_query_mih` searches for k nearest neighbors by probing hash
tables at increasing Hamming radii.

For radius r on table ti with query substring h:

1. Enumerate all `C(sub_bits, r)` bit-flip masks via combinatorial
   iteration over `pos[0..r-1]`.
2. For each mask: `probe_h = h ^ mask`. Look up the bucket.
3. For each SID in the bucket (skipping the query itself and previously
   seen SIDs): compute full Hamming distance via
   `tk_cvec_bits_hamming_serial`, then either:
   - If k > 0: insert into a max-heap of k results (`tk_pvec_hmax`).
   - If k = 0: append to output (unbounded collection).

Outer loop: `r = 0, 1, ..., max_probe_radius`, probing all m tables at
each radius before advancing.

### Early Termination

After processing all tables at radius r, if `k` results are found and
the worst result in the heap has distance `< m * (r + 1)`, search stops.
The bound `m * (r+1)` is the maximum possible Hamming distance for any
code that hasn't been seen yet: an unseen code must differ from the query
by more than r bits in every substring, so its total distance is at least
`m * (r+1)`.

### Neighborhood Queries

Three bulk query methods, all parallelized with
`omp parallel for schedule(guided)`:

**`neighborhoods(k, radius)`** -- all-vs-all. Queries every indexed code
against all others. Returns `(hoods, uids)` where `hoods` is a vec of
pvecs and `uids` is the corresponding ID ordering. Hood entries are
`(positional_index, hamming_distance)`.

**`neighborhoods_by_ids(query_ids, k, radius)`** -- subset query. Queries
only the specified UIDs. UIDs not in the index are filtered. Returns
`(hoods, all_uids)` with positional indices into `all_uids`.

**`neighborhoods_by_vecs(query_vecs, k, radius)`** -- external query.
Queries arbitrary binary codes not necessarily in the index. No self-skip.
Returns `(hoods, all_uids)` with positional indices into `all_uids`.

All three use `tk_ann_prepare_universe_map` to build a positional index
(`sid_to_pos`) mapping internal SIDs to dense output positions. Hood
entries use positional indices (not UIDs directly); dereference via the
returned `uids` ivec.

**`neighbors(id_or_vec, k, radius, out)`** -- single query returning a
pvec of `(uid, hamming_distance)` pairs, sorted ascending by distance.

### Similarity and Distance

    similarity(a, b) = 1 - hamming(a, b) / features
    distance(a, b) = hamming(a, b) / features

Computed via `tk_cvec_bits_hamming_serial` (byte-by-byte popcount of XOR).

### Get

`ann:get(uid_or_uids [, out, dest_sample, dest_stride])` extracts stored
binary codes. When `dest_stride > 0`, codes are bit-packed at arbitrary
bit offsets (not byte-aligned), enabling assembly of multi-source code
matrices where different encoders contribute different bit ranges.

### Persist Format

Magic `TKan`, version byte (1), destroyed flag, next_sid, features, m,
then for each table: bucket count followed by (hash_key, has_posting,
[posting_length, posting_sids]) per bucket. Then uid_sid (iumap),
sid_to_uid (ivec), vectors byte count and raw bytes.

`ann.load(path_or_data, is_string)` restores a persisted index.

## Parameters

### Quantizer

| Parameter | Role | Default |
|---|---|---|
| `raw_codes` | Continuous embeddings (dvec), mutually exclusive with codes | --|
| `codes` | Binary codes (cvec), mutually exclusive with raw_codes | --|
| `n_samples` | Number of samples | required |
| `n_dims` | Input dimensionality | required |
| `target_bits` | Maximum output bits | n_dims |
| `max_dims` | Maximum distinct source dimensions (0 = unlimited) | 0 |
| `tolerance` | Score tolerance for pruning pass acceptance | 0.0 |
| `mode` | `"thermometer"` for thermometer mode, nil for SFBS | nil |
| `n_bins` | Thermometer bins per dimension (0 = all unique values) | 0 |
| `ids` | Sample UIDs (ivec) | required |
| `expected_ids` | Adjacency node UIDs (ivec) | required (SFBS) |
| `expected_offsets` | CSR offsets into neighbor list (ivec) | required (SFBS) |
| `expected_neighbors` | Neighbor positional indices (ivec) | required (SFBS) |
| `expected_weights` | Neighbor weights (dvec) | required (SFBS) |
| `each` | Progress callback | nil |

### ANN Index

| Parameter | Role | Default |
|---|---|---|
| `features` | Number of bits per code (at creation) | required |
| `k` | Number of nearest neighbors to return | 0 (unlimited) |
| `radius` | Maximum probe radius (Hamming radius per substring) | 3 |
