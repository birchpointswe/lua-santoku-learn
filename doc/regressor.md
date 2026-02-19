# Regression Tsetlin Machine with Sparse Absorption

## Overview

A unified regression Tsetlin Machine that handles classification,
continuous-target regression, and binary encoding through a single training
path. Sparse mode adds per-class token mapping with absorption-driven
feature streaming for large-vocabulary problems.

All training goes through `tk_tsetlin_train_regressor`. Classification
targets (ivec labels or cvec binary codes) are converted to per-class
regression targets internally: +1.0 for positive, -1.0 for negative.

## State Representation

Each Tsetlin Automaton is stored as a multi-bit counter plus an action bit,
packed in bit-planar layout. For `state_bits=8`: 7 counter planes and 1
action plane, each `input_chunks` bytes wide. A clause's full state is the
concatenation of its planes across all literal positions.

Input features are pairs `(x_i, ~x_i)` packed as 2 bits per feature, so
`input_bits = 2 * features` and `input_chunks = ceil(input_bits / 8)`.

The bit-planar layout enables bulk TA updates via carry-chain arithmetic
across all literals in a clause simultaneously. `tk_automata_inc` ripple-
carries through counter planes then into the action plane; overflow saturates
at maximum state. `tk_automata_dec` does the borrow-chain equivalent;
underflow saturates at zero. Both have AVX-512 paths that process 512 TAs
per SIMD instruction when available. Early-exit checks (`carry_zero`) skip
remaining planes when no more bits need updating.

## Clause Evaluation

Clause output is not strict Boolean AND. The vote for a clause is:

- Empty clause (no included literals): votes 1 (matches everything).
- Otherwise: `min(n_included, clause_tolerance) - n_failed`, clamped to
  zero. A clause fires (contributes to the chunk vote) when this is positive.

`n_included` is the popcount of the action bits. `n_failed` is the popcount
of `action & ~input` -- included literals whose input is 0.

This is a soft/fuzzy evaluation: a clause with 10 included literals and 2
failures still fires with vote 8 (assuming tolerance >= 10). The
`clause_tolerance` parameter caps the maximum vote contribution per clause.
`clause_maximum` separately limits how many literals can receive Type I
reward, acting as a literal budget.

Chunks of `TK_CVEC_BITS` (8) clauses are evaluated together. Half the
clause chunks per class have positive polarity, half negative. Negative
polarity chunk votes are subtracted from the class sum.

## Feedback

Feedback probability is proportional to squared error:

    error = (chunk_vote - ideal_chunk_vote) / vote_target
    probability = error^2

Where `ideal_chunk_vote` is derived from the target value, the per-class
`y_min`/`y_max` range, and the chunk's polarity. Each clause in the chunk
independently receives feedback with this probability.

The `want_more` flag (chunk_vote < ideal) selects the feedback type:

### Positive feedback (want_more = true)

When the clause fires and is below the literal budget:
`tk_automata_inc(clause, input)` -- increment all TAs where the input bit is
1. Rewards Include for matching literals.

Then unconditionally when the clause fires:
`tk_automata_dec_not_excluded(clause, input)` -- decrement TAs where both
the input bit and the action bit are 0. Penalizes Include (rewards Exclude)
for non-matching, currently-excluded literals.

When the clause does not fire: sample `specificity_threshold` random input
bit positions and decrement each. This is the weak-probability decay that
coarsens infrequent patterns. `specificity_threshold = 2 * features /
floor(specificity)`, so larger `s` means fewer random decrements and
finer-grained patterns.

### Negative feedback (want_more = false)

When the clause fires:
`tk_automata_inc_not_excluded(clause, input)` -- increment excluded TAs
where the input bit is 0 (`~input & ~actions`). Forces inclusion of
literals that would block the clause from firing on this input.

When the clause does not fire: no action.

### Deviation from standard TM feedback

Standard TM uses `(T +/- clamp(v)) / 2T` as the probability of applying
feedback to each clause, based on the aggregate class vote sum. This
implementation replaces that with per-chunk squared error, which is more
directly tied to the regression objective. The resource allocation effect
still holds: chunks near their ideal receive almost no feedback, freeing
learning capacity for underperforming chunks.

## Imbalanced Label Handling

For classification targets (ivec or cvec), per-class skip thresholds
downsample negative examples:

    skip_prob = 1 - pos_count / neg_count

For each negative sample presented to a class, a random draw against
`skip_prob` decides whether to skip it entirely. This gives each class
effective 50/50 positive/negative exposure regardless of label frequency.

## Sparse Mode

Enabled by passing `n_tokens` at creation. Designed for large-vocabulary
input where materializing `2 * n_tokens` literal bits per clause is
infeasible.

### Per-class token mapping

Each class maintains a mapping of `features` slots to token IDs from a
vocabulary of `n_tokens`:

    mapping[class * features + slot] -> token_id

An `active` bitmap (per class, `ceil(n_tokens / 8)` bytes) tracks which
tokens are currently mapped, giving O(1) collision checks during
mapping initialization and absorption replacement.

### Densification

The managed dense buffer has layout `[sample * classes * bytes_per_class]`
where `bytes_per_class = ceil(features * 2 / 8)`. Each slot occupies 2 bits:
`0b10` = token absent, `0b01` = token present. The buffer is initialized to
`0xAA` (all absent).

`tk_tsetlin_densify_all` walks the CSC (Compressed Sparse Column) index to
set presence bits for all mapped tokens across all samples. This is
parallelized across classes.

`tk_tsetlin_densify_slot` does incremental single-slot updates: clears the
old token's presence bits (walking its CSC column), then sets the new
token's. Used during absorption to avoid full re-densification.

### Grouped input

When in sparse mode, `grouped=true` is automatic. Each class sees a
different Boolean input vector for the same sample because each class has
its own token-to-slot mapping. Class 0's clauses operate on tokens
{37, 142, 8903, ...} while class 1 sees {55, 203, 7621, ...}. This is
per-class embedded feature selection through the mapping mechanism.

### Initialization rankings

Two ranking structures control which tokens each class starts with and
what replaces absorbed tokens:

**absorb_ranking** (+ optional **absorb_ranking_offsets**): Per-class
initialization priority. When offsets are provided, `absorb_ranking` is a
concatenated array with per-class segments indexed by
`absorb_ranking_offsets[c]..absorb_ranking_offsets[c+1]`. Each class fills
its slots with the highest-priority tokens from its segment first, falling
back to random fill for remaining slots.

**absorb_ranking_global**: The streaming pool for absorption replacement.
Typically a global token ordering (all indices, or global chi-squared). When
a token is absorbed-excluded from a class, its replacement comes from this
pool via a per-class circular cursor.

When `absorb_ranking_offsets` is absent, both rankings alias the same
array. When offsets are present and no explicit global ranking is given, a
copy is made to prevent the init ranking's per-class structure from
interfering with the global cursor walk. The destroy path checks
`absorb_ranking_global != absorb_ranking` before freeing to avoid
double-free.

When no ranking is provided at all, a Fisher-Yates shuffle of
`[0, n_tokens)` is generated as a fallback.

### Absorption

`tk_tsetlin_absorb_class` runs every `absorb_interval` training iterations,
per class, parallelized with `omp for schedule(dynamic)`.

For each slot, it computes the maximum full-state across all clauses and
both literal polarities (positive and negated):

    full_state = action_is_include ? (max_counter + 1 + counter) : counter

This maps the TA's (counter, action) pair into a single `0..2*max_counter+1`
range. A slot is eligible for replacement only if its max full-state across
every clause is at or below `absorb_threshold`.

This is stricter than published Contracting TM work (Bhattarai 2023) where
each TA absorbs independently. Here, a token survives if any single clause
in any polarity finds it useful. This prevents premature replacement of
tokens that matter for a few clauses but are irrelevant to most.

Eligible slots are sorted by max_state ascending (weakest first). Up to
`absorb_maximum` slots are replaced per class per absorption pass. For each:

1. Old token is deactivated in the bitmap.
2. The per-class cursor walks `absorb_ranking_global` to find the next
   inactive token. Wraps around; if all tokens are active (vocab exhaustion),
   the slot is skipped.
3. New token is activated and mapped to the slot.
4. `tk_tsetlin_densify_slot` incrementally updates the dense buffer.
5. `tk_tsetlin_reset_slot_automata` resets the slot's TAs to
   `absorb_insert` state across all clauses.

`absorb_insert` defaults to `absorb_threshold + 1`, placing new TAs just
above the absorb boundary. They get a chance to prove useful before they
themselves might be absorbed. Setting it higher gives new tokens more
runway; setting it just above threshold makes the system more aggressive
about cycling.

### Absorption scratch memory

Each thread gets `features * 2 * sizeof(unsigned int)` bytes of scratch for
the max_states and eligible arrays. Allocated once at creation based on
`omp_get_max_threads()`.

## Hyperparameters

| Parameter | Role | Typical range |
|---|---|---|
| `clauses` | Per-class clause count (chunks per polarity, N -> N*16) | 1-32 |
| `clause_tolerance` | Max vote contribution per clause | 8-1024 |
| `clause_maximum` | Max literals receiving Type I reward | 8-1024 |
| `target` | Vote target for regression scaling | 8-1024 |
| `specificity` | Controls pattern granularity (higher = finer) | 2-2000 |
| `state_bits` | TA counter width including action bit | 8 (default) |
| `absorb_interval` | Iterations between absorption passes | 1-40 |
| `absorb_threshold` | Full-state threshold for eligibility | 0-126 (for 8-bit) |
| `absorb_maximum` | Max slots replaced per class per pass | 0 (unlimited)-features |
| `absorb_insert` | Initial state for newly streamed TAs | absorb_threshold+1 to 127 |

`clause_tolerance` and `clause_maximum` are auto-swapped if given in wrong
order. Both are capped at `2 * features` (total input bits). Target is
capped at `8 * clause_tolerance` during optimization. Specificity is capped
at `2 * features`.

Specificity threshold computation uses float division:
`specificity_threshold = (unsigned int)((2.0 * features) / specificity)`.

## Per-Output Hyperparameters

When `output_weights` (dvec, typically eigenvalues from spectral
embedding) is provided at training time, each output dimension receives
its own tolerance, maximum, specificity threshold, and target via
exponential modulation:

    value_for_dim_i = base * exp(alpha * (w_norm_i - 0.5))

where `w_norm_i = (weight_i - w_min) / (w_max - w_min)` normalizes
the weight to [0,1].

Four alpha parameters control modulation strength:
- `alpha_tolerance` -- modulates clause_tolerance per dim
- `alpha_maximum` -- modulates clause_maximum per dim
- `alpha_specificity` -- modulates specificity per dim
- `alpha_target` -- modulates target per dim

Alpha = 0 means no modulation (all dims get the base value). Positive
alpha gives higher values to dims with larger weights (leading
eigenvalues). Negative alpha gives higher values to dims with smaller
weights (trailing eigenvalues).

Per-dim values are stored as ivec arrays (`per_class_tolerances`,
`per_class_maximums`, `per_class_spec_thresholds`, `per_class_targets`)
in the TM struct. Ordering constraints are enforced per-dim:
tolerance <= maximum, target <= 8 * tolerance.

When `output_weights` is nil, no per-class ivecs are created and the
scalar parameters broadcast to all classes (backward compatible).

## Inference

### Classification (`classify`)

Per-sample: accumulate chunk votes into per-class sums (positive polarity
adds, negative subtracts). Return argmax class. Parallelized across samples.

### Regression (`regress`)

Per-sample, per-class:

    output = (votes_sum / (clause_chunks * target) + 0.5) * y_range + y_min

Maps the vote sum from `[-clause_chunks*target, +clause_chunks*target]` to
`[y_min, y_max]`. Parallelized across samples.

### Encoding (`encode`)

Per-sample: 1 bit per class. Bit is 1 if that class's vote sum is positive.
Produces a binary hash code suitable for Hamming-distance ANN retrieval.

### Sparse inference

All three modes accept either a cvec (pre-densified input) or a table with
a `tokens` field (ivec of `sample_index * n_tokens + token_id` packed
entries). The latter path calls `tk_tsetlin_densify_tokens`, which:

1. Bins tokens by sample (CSR-style bucketing)
2. Builds a reverse lookup `token_id -> slot` per class
3. Sets presence bits in a temporary dense buffer
4. Cleans up the reverse lookup per class (reset to -1)

This is used for dev/test inference where the managed dense buffer is only
available for the training set.

## Checkpoint and Restore

`checkpoint` saves actions + mapping into a cvec buffer. Does not save
counter state planes -- only action bits (sufficient for inference) and the
mapping (needed for sparse routing). `restore` recovers both, rebuilds the
active bitmap from the mapping, and re-densifies if a managed buffer exists.

Used by the batched training loop for early stopping with patience: the
best-seen model is checkpointed, and if the metric doesn't improve for
`patience` batches, training halts and restores to the checkpoint.

## Reconfigure

Allows changing all core hyperparameters between training runs without
reallocating the Lua userdata. Requires `reusable=true` at creation.
Reallocates state and action arrays if the new configuration needs more
memory (never shrinks allocations). Resets all sparse mode bookkeeping:
clears managed_dense, active bitmap, and cursors. Rankings are preserved
across reconfigure -- they are owned memory that persists until destroy.

Used by the hyperparameter search to run many configurations with a single
TM object: `reconfigure` then `train` on each trial, avoiding repeated
malloc/free cycles.

## Restrict

Reduces the model to a subset of classes after training. Compacts the action
and state arrays in-place by copying selected class slices into a temporary
buffer and writing back. Used when a trained model needs to serve only a
subset of its original output dimensions.

## Persist and Load

Binary format with magic `TKtm`, version byte, all scalar parameters, then
the raw action array and per-class y_min/y_max. Does not persist state
planes (inference-only). Does not persist sparse mode state (mapping,
rankings, dense buffer). A loaded model has `has_state=false` and cannot be
trained further.
