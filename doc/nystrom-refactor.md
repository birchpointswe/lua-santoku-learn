# Nyström Extension Implementation

## Prerequisites

### optimize.spectral Return Values

The `optimize.spectral` function returns `(model, params, metrics)` where model contains:
- `codes`: Binary spectral codes (cvec)
- `raw_codes`: Continuous eigenvectors before binarization (dvec)
- `eigs`: Eigenvalues from spectral decomposition (dvec)
- `dims`: Number of dimensions (may be less than requested if dimension selection is enabled)
- `ids`: Sample IDs (ivec)
- `index`: ANN index over binary codes
- `threshold_fn`: Binarization function
- `threshold_params`: Parameters used for threshold method
- `threshold_state`: Learned threshold parameters for reuse - added by modification below
  - For ITQ: `{rotation = dvec, mean = dvec}`
  - For median: `{thresholds = dvec}`
  - For otsu: `{thresholds = dvec, indices = ivec}`
  - For sign: `{}` (no learned parameters)

For Nyström extension, we use `model.raw_codes`, `model.eigs`, `model.dims`, and `model.threshold_state` directly.

### Dimension Selection Support

**IMPORTANT**: Dimension selection must occur BEFORE binarization when using stateful threshold methods (ITQ, median, otsu).

**Why**: Stateful methods learn parameters (rotation matrices, threshold values, dimension permutations) from the data. If dimension selection happens after learning these parameters, the parameter dimensions won't match the selected code dimensions.

**Current Implementation**: Looking at `optimize.lua` lines 991-1042, dimension selection happens AFTER thresholding. This works for immediate use but breaks Nyström extension because:
- ITQ learns K×K rotation and K-length mean
- Selection produces D < K dimensions
- Applying K×K rotation to D-dimensional test codes fails

**Required Change**: Modify `M.score_spectral_eval` in `optimize.lua` to:
1. Perform dimension selection on `raw_codes` BEFORE calling `M.apply_threshold`
2. Subset `raw_codes` and `eigs` to selected D dimensions
3. Call `M.apply_threshold` on the D-dimensional subset
4. Return `model.dims = D` with rotation/mean of correct size

This ensures learned threshold parameters match the final code dimensionality.

### ITQ Rotation Matrix Support

**CRITICAL REQUIREMENT - DO NOT REMOVE**

ITQ (Iterative Quantization) learns two parameters during binarization:
1. **Mean vector** (K-dimensional): Per-dimension mean computed during centering
2. **Rotation matrix** (K×K): Orthogonal rotation learned via alternating optimization

Currently, each call to `itq.itq()` computes **new parameters** specific to the input codes. This makes it impossible to correctly binarize Nyström-extended test codes because they would be centered and rotated differently than training codes, producing incompatible binary embeddings.

**Why this cannot be skipped**: The `model.threshold_fn` returned by `optimize.spectral` does NOT capture the learned rotation matrix or mean - it only captures hyperparameters like iteration count. Each call to `threshold_fn(new_codes, dims)` recomputes ITQ from scratch on the new codes with different parameters.

**Solution**: Modify ITQ to support:
1. **Returning the rotation matrix AND mean vector** after learning
2. **Accepting pre-learned rotation and mean** and skipping the learning phase

This allows training codes and test codes to be centered and rotated with the **same parameters**, ensuring compatibility.

#### Modified ITQ API

```c
// In lib/santoku/tsetlin/itq.c
int tk_itq_lua(lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  tk_dvec_t *codes = tk_lua_fget(L, 1, "itq", "codes", tk_dvec_peek);
  uint64_t n_dims = tk_lua_fgetunsigned(L, 1, "itq", "n_dims");
  uint64_t iterations = tk_lua_foptunsigned(L, 1, "itq", "iterations", 50);
  double tolerance = tk_lua_foptnumber(L, 1, "itq", "tolerance", 1e-8);

  // Input validation
  if (n_dims == 0)
    return luaL_error(L, "itq: n_dims must be > 0");
  if (codes->n % n_dims != 0)
    return luaL_error(L, "itq: codes size (%llu) not divisible by n_dims (%llu)",
                      (unsigned long long)codes->n, (unsigned long long)n_dims);
  uint64_t n_samples = codes->n / n_dims;
  if (n_samples == 0)
    return luaL_error(L, "itq: no samples");

  // NEW: optionally return rotation matrix and mean
  bool return_rotation = tk_lua_foptboolean(L, 1, "itq", "return_rotation", false);

  // NEW: optionally accept pre-learned rotation
  lua_getfield(L, 1, "rotation");
  tk_dvec_t *rotation = NULL;
  if (lua_type(L, -1) != LUA_TNIL) {
    rotation = tk_dvec_peek(L, -1);
  }
  lua_pop(L, 1);

  // NEW: optionally accept pre-computed mean for centering
  lua_getfield(L, 1, "mean");
  tk_dvec_t *mean = NULL;
  if (lua_type(L, -1) != LUA_TNIL) {
    mean = tk_dvec_peek(L, -1);
  }
  lua_pop(L, 1);

  uint64_t n_samples = codes->size / n_dims;
  tk_dvec_t *R;
  tk_dvec_t *M;
  bool allocated_R = false;
  bool allocated_M = false;

  if (rotation && mean) {
    // Use provided rotation and mean, skip learning
    if (rotation->size != n_dims * n_dims)
      return luaL_error(L, "itq: rotation matrix size mismatch");
    if (mean->size != n_dims)
      return luaL_error(L, "itq: mean vector size mismatch");
    R = rotation;
    M = mean;
  } else if (!rotation && !mean) {
    // Learn both rotation and mean via ITQ algorithm
    R = tk_dvec_create(L, n_dims * n_dims, 0, 0);
    M = tk_dvec_create(L, n_dims, 0, 0);
    allocated_R = true;
    allocated_M = true;
    tk_itq_fit(codes, R, M, n_dims, n_samples, iterations, tolerance, L);
  } else {
    return luaL_error(L, "itq: must provide both rotation and mean, or neither");
  }

  // Apply centering and rotation: B = sign((V - mean) × R)
  tk_cvec_t *binary_codes = tk_cvec_create(L, n_samples * TK_CVEC_BITS_BYTES(n_dims), 0, 0);
  tk_itq_transform(codes, R, M, binary_codes, n_dims, n_samples, L);

  // Stack management for return values
  // Note: tk_dvec_create and tk_cvec_create push objects to stack
  if (return_rotation) {
    if (allocated_R) {
      // Stack positions: [1:args, 2:R, 3:M, 4:binary_codes]
      // lua_insert(L, -3) takes element at index -1 and inserts at -3
      // This moves binary_codes from index -1 to index -3, shifting R and M
      lua_insert(L, -3);
      // Result: [1:args, 2:binary_codes, 3:R, 4:M]
      return 3;  // Returns elements at indices 2, 3, 4
    } else {
      // Stack: [1:args, 2:binary_codes]
      // Fetch provided rotation and mean from args table
      lua_getfield(L, 1, "rotation");
      lua_getfield(L, 1, "mean");
      // Stack: [1:args, 2:binary_codes, 3:rotation, 4:mean]
      return 3;
    }
  }

  // Only return binary_codes (default behavior for backward compatibility)
  if (allocated_R) {
    // Stack: [1:args, 2:R, 3:M, 4:binary_codes]
    // lua_replace(L, -3) pops top and replaces element at index -3
    lua_replace(L, -3);
    // Result: [1:args, 2:binary_codes, 3:M]
    lua_pop(L, 1);
    // Result: [1:args, 2:binary_codes]
  }
  // Stack: [1:args, 2:binary_codes] in all cases
  return 1;
}
```

#### Helper Functions

```c
// Fit ITQ rotation matrix and mean via alternating optimization
static void tk_itq_fit(tk_dvec_t *codes, tk_dvec_t *R, tk_dvec_t *mean,
                       uint64_t n_dims, uint64_t n_samples,
                       uint64_t iterations, double tolerance, lua_State *L) {
  const uint64_t K = n_dims;
  const uint64_t N = n_samples;

  // Allocate temporary buffers (from itq.h lines 254-263)
  size_t total_size = (N * K * 3 + K * K * 3 + K + K - 1) * sizeof(double);
  double *mem = tk_malloc(L, total_size);
  double *X = mem;
  double *V0 = X + N * K;
  double *B = V0 + N * K;
  double *BtV = B + N * K;
  double *U = BtV + K * K;
  double *VT = U + K * K;
  double *S = VT + K * K;
  double *superb = S + K;

  // Copy codes to working buffer (line 265)
  memcpy(X, codes->a, N * K * sizeof(double));

  // Center the codes and STORE the mean (replaces line 266)
  for (uint64_t k = 0; k < K; k++) {
    double sum = 0.0;
    for (uint64_t n = 0; n < N; n++) {
      sum += X[n * K + k];
    }
    double m = sum / (double)N;
    mean->a[k] = m;  // STORE MEAN
    for (uint64_t n = 0; n < N; n++) {
      X[n * K + k] -= m;  // CENTER
    }
  }

  // Initialize R to identity (lines 267-270)
  #pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < K; i++)
    for (uint64_t j = 0; j < K; j++)
      R->a[i * K + j] = (i == j ? 1.0 : 0.0);

  // ITQ alternating optimization (lines 273-297)
  double last_obj = DBL_MAX;
  for (uint64_t it = 0; it < iterations; it++) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, K, K, 1.0, X, K, R->a, K, 0.0, V0, K);
    double obj = 0.0;
    #pragma omp parallel for reduction(+:obj)
    for (size_t idx = 0; idx < N * K; idx++) {
      double v = V0[idx];
      double b = (v >= 0.0 ? 1.0 : -1.0);
      B[idx] = b;
      double d = b - v;
      obj += d * d;
    }
    if (it > 0 && fabs(last_obj - obj) < tolerance * fabs(obj))
      break;
    last_obj = obj;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, K, N, 1.0, B, K, X, K, 0.0, BtV, K);
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', K, K, BtV, K, S, U, K, VT, K, superb);
    if (info != 0) {
      free(mem);
      luaL_error(L, "ITQ SVD failed to converge (info=%d)", info);
      return;
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, K, K, K, 1.0, VT, K, U, K, 0.0, R->a, K);
  }

  free(mem);
}

// Apply centering, rotation, and binarization
static void tk_itq_transform(tk_dvec_t *codes, tk_dvec_t *R, tk_dvec_t *mean,
                             tk_cvec_t *binary_codes,
                             uint64_t n_dims, uint64_t n_samples, lua_State *L) {
  const uint64_t K = n_dims;
  const uint64_t N = n_samples;

  // Allocate temporary buffers using Lua-safe allocation
  size_t buffer_size = N * K * 2 * sizeof(double);
  double *mem = tk_malloc(L, buffer_size);
  double *X_centered = mem;
  double *V0 = X_centered + N * K;

  // Center using training mean (parallelized for performance)
  #pragma omp parallel for
  for (uint64_t n = 0; n < N; n++) {
    for (uint64_t k = 0; k < K; k++) {
      X_centered[n * K + k] = codes->a[n * K + k] - mean->a[k];
    }
  }

  // Apply rotation: V0 = X_centered × R (using BLAS)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, K, K,
              1.0, X_centered, K, R->a, K,
              0.0, V0, K);

  // Binarize using optimized byte-packing (reuse tk_itq_sign from itq.h)
  tk_itq_sign(binary_codes->a, V0, N, K);

  free(mem);
}
```

#### Usage Example

```lua
-- Training: learn rotation and mean
local train_codes_unsup, rotation_matrix, mean_vector = itq.itq({
  codes = raw_codes_unsup,
  n_dims = 32,
  iterations = 100,
  tolerance = 1e-8,
  return_rotation = true,  -- Get codes, rotation matrix, and mean
})

-- Test: reuse learned rotation and mean
local test_codes_unsup = itq.itq({
  codes = nystrom_raw_codes,
  n_dims = 32,
  rotation = rotation_matrix,  -- Apply same rotation as training
  mean = mean_vector,          -- Apply same centering as training
})
```

#### Backward Compatibility

**Existing code works unchanged:**
```lua
local codes = itq.itq({ codes = raw_codes, n_dims = 32 })
-- Returns 1 value (codes) when return_rotation not specified
```

**However, code that explicitly checks return value count may need updating:**
```lua
-- This pattern would break:
assert(select('#', itq.itq({codes = raw, n_dims = 32})) == 1)

-- This pattern is fine (Lua allows ignoring extra returns):
local codes = itq.itq({codes = raw, n_dims = 32, return_rotation = true})
-- codes gets first return value, rotation and mean are discarded
```

**Recommended migration:** Existing code calling `itq.itq` will continue to work. Only code that needs Nyström extension should use `return_rotation = true`.

**API Signature Impact:**
- Current signature: `itq.itq({...})` returns 1 value
- New signature: `itq.itq({...})` returns 1 value by default, 3 values with `return_rotation = true`
- This is backward compatible unless code uses `select('#', ...)` to count return values
- Similar changes apply to `itq.median` (2 values) and `itq.otsu` (3 values with thresholds instead of scores)

### Median Threshold Parameter Preservation

**CRITICAL REQUIREMENT**

The `itq.median` method learns per-dimension median thresholds from training data. Without preserving these thresholds, Nyström-extended test codes would compute their own medians, producing incompatible embeddings.

#### Modified Median API

```lua
-- Training: learn median thresholds
local train_codes, median_thresholds = itq.median({
  codes = raw_codes,
  n_dims = 32,
  return_thresholds = true,  -- Get median values
})

-- Test: reuse learned thresholds
local test_codes = itq.median({
  codes = nystrom_raw_codes,
  n_dims = 32,
  thresholds = median_thresholds,  -- Apply same thresholds as training
})
```

The median vector is K-dimensional (one threshold per dimension).

### Otsu Threshold Parameter Preservation

**CRITICAL REQUIREMENT**

The `itq.otsu` method learns per-dimension optimal thresholds AND reorders dimensions by quality scores. Without preserving these, test codes would have incompatible thresholds and wrong bit ordering.

#### Modified Otsu API

```lua
-- Training: learn thresholds and dimension permutation
local train_codes, otsu_thresholds, dimension_indices = itq.otsu({
  codes = raw_codes,
  n_dims = 32,
  metric = "variance",
  return_thresholds = true,  -- Get learned parameters
})

-- Test: reuse learned thresholds and permutation
local test_codes = itq.otsu({
  codes = nystrom_raw_codes,
  n_dims = 32,
  thresholds = otsu_thresholds,      -- Apply same thresholds
  indices = dimension_indices,        -- Apply same dimension ordering
})
```

The thresholds vector is K-dimensional, and indices vector is K-dimensional (permutation).

### optimize.spectral Threshold Parameter Return

**Required Change**: Modify `optimize.spectral` to capture and return learned threshold parameters (rotation/mean for ITQ, thresholds for median/otsu).

In `lib/santoku/tsetlin/optimize.lua`, the `M.apply_threshold` function (lines 911-937) needs to capture learned parameters for all stateful threshold methods:

```lua
M.apply_threshold = function (args)
  local raw_model = args.raw_model
  local threshold_params = args.threshold_params
  local bucket_size = args.bucket_size

  local threshold_fn = M.build_threshold_fn(threshold_params)

  -- Capture learned parameters from stateful threshold methods
  local codes, threshold_state = nil, {}
  local method = threshold_params.method

  if method == "itq" then
    local rotation, mean
    codes, rotation, mean = itq.itq({
      codes = raw_model.raw_codes,
      n_dims = raw_model.dims,
      iterations = threshold_params.iterations or 100,
      tolerance = threshold_params.tolerance or 1e-8,
      return_rotation = true,
    })
    threshold_state.rotation = rotation
    threshold_state.mean = mean
  elseif method == "median" then
    local thresholds
    codes, thresholds = itq.median({
      codes = raw_model.raw_codes,
      n_dims = raw_model.dims,
      return_thresholds = true,
    })
    threshold_state.thresholds = thresholds
  elseif method == "otsu" then
    local thresholds, indices
    codes, thresholds, indices = itq.otsu({
      codes = raw_model.raw_codes,
      n_dims = raw_model.dims,
      metric = threshold_params.metric or "variance",
      n_bins = threshold_params.n_bins or 32,
      minimize = threshold_params.minimize or false,
      return_thresholds = true,
    })
    threshold_state.thresholds = thresholds
    threshold_state.indices = indices
  else
    -- Stateless methods (sign, custom functions)
    codes = threshold_fn(raw_model.raw_codes, raw_model.dims)
  end

  -- Create ANN index (existing code)
  local ann_index = ann.create({
    expected_size = raw_model.ids:size(),
    features = raw_model.dims,
    bucket_size = bucket_size,
  })
  ann_index:add(codes, raw_model.ids)

  return {
    ids = raw_model.ids,
    codes = codes,
    raw_codes = raw_model.raw_codes,
    threshold_fn = threshold_fn,
    threshold_params = threshold_params,
    threshold_state = threshold_state,  -- Learned parameters for reuse
    eigs = raw_model.eigs,
    dims = raw_model.dims,
    index = ann_index,
  }
end
```

Additionally, modify `M.score_spectral_eval` in `optimize.lua` to perform dimension selection BEFORE thresholding:

```lua
M.score_spectral_eval = function (args)
  -- ... existing code to get raw_model ...

  local eval_dims = raw_model.dims
  local selected_raw_codes = raw_model.raw_codes
  local selected_eigs = raw_model.eigs

  -- MOVE dimension selection HERE (before thresholding)
  if eval_params.select_elbow and eval_params.select_metric then
    -- Compute dimension scores (use existing logic from lines 996-1017)
    local indices, scores
    if select_metric == "entropy" then
      indices, scores = raw_model.raw_codes:mtx_top_entropy(n_samples, eval_dims, eval_dims, malpha.n_bins)
    elseif select_metric == "variance" then
      indices, scores = raw_model.raw_codes:mtx_top_variance(n_samples, eval_dims, eval_dims)
    -- ... other metrics ...
    end

    local _, elbow_idx = scores:scores_elbow(eval_params.select_elbow, eval_params.select_elbow_alpha)
    scores:destroy()

    if elbow_idx > 0 and elbow_idx < eval_dims then
      -- Subset raw codes using existing mtx_select method
      local temp_raw = dvec.create()
      indices:setn(elbow_idx)
      raw_model.raw_codes:mtx_select(indices, nil, eval_dims, temp_raw)
      selected_raw_codes = temp_raw

      -- Subset eigenvalues (manual loop - no built-in function)
      local temp_eigs = dvec.create()
      for i = 0, elbow_idx - 1 do
        temp_eigs:push(raw_model.eigs:get(indices:get(i)))
      end
      selected_eigs = temp_eigs

      eval_dims = elbow_idx
    end
  end

  -- Now apply threshold to selected dimensions
  local threshold_model = M.apply_threshold({
    raw_model = {
      raw_codes = selected_raw_codes,
      dims = eval_dims,
      ids = raw_model.ids,
      eigs = selected_eigs,
    },
    threshold_params = threshold_params,
    bucket_size = bucket_size,
  })

  -- ...
end
```

This ensures learned threshold parameters match the final code dimensionality.

## Validation and Error Handling

All modifications include comprehensive input validation:

**ITQ modifications:**
- Validates `n_dims > 0`
- Validates `codes->n % n_dims == 0`
- Validates `n_samples > 0`
- Validates rotation matrix and mean vector sizes when provided
- Enforces all-or-none for rotation/mean parameters

**nystrom_encoder:**
- Validates `n_dims > 0`
- Validates eigenvalues size matches n_dims
- Validates eigenvectors size divisible by n_dims
- Validates ids size matches eigenvector row count
- Validates IDs are non-negative (required for hash map)
- Validates features_index type (inv/ann/hbi)
- Checks hash map creation success

**Error handling patterns:**
- Uses `luaL_error(L, "message", ...)` for validation failures (supports printf formatting)
- Uses `tk_error(L, "message", errno)` for allocation failures
- Proper memory cleanup before errors (frees allocated buffers)

## Implementation Summary

The Nyström extension refactor requires modifications to three modules:

### 1. ITQ Module (`lib/santoku/tsetlin/itq.c`, `lib/santoku/tsetlin/itq.h`)

**Modify all threshold methods to support parameter preservation:**

| Method | Learned Parameters | Return Values with Flag | Input Parameters for Reuse |
|--------|-------------------|------------------------|----------------------------|
| `itq.itq` | Rotation (K×K), Mean (K) | `(codes, rotation, mean)` | `rotation=dvec, mean=dvec` |
| `itq.median` | Thresholds (K) | `(codes, thresholds)` | `thresholds=dvec` |
| `itq.otsu` | Thresholds (K), Permutation (K) | `(codes, thresholds, indices)` | `thresholds=dvec, indices=ivec` |
| `itq.sign` | None | `(codes)` | None |

**Key requirements:**
- Add `return_rotation`/`return_thresholds` flag to each method
- Add parameter inputs for applying learned transformations to new data
- Extract fit and transform logic into separate helper functions
- Ensure backward compatibility (default behavior unchanged)

### 2. Optimize Module (`lib/santoku/tsetlin/optimize.lua`)

**Modify `M.apply_threshold` to:**
- Detect threshold method type (itq, median, otsu, sign)
- Call with `return_*=true` to capture learned parameters
- Store in `threshold_state` table
- Return `threshold_state` in model

**Modify `M.score_spectral_eval` to:**
- Perform dimension selection BEFORE calling `M.apply_threshold`
- Subset `raw_codes` and `eigs` to selected dimensions first
- Apply thresholding to already-selected codes
- Ensures learned parameters match final code dimensionality

### 3. HLTH Module (`lib/santoku/tsetlin/hlth.c`)

**Add `hlth.nystrom_encoder` function:**
- Performs Nyström extension using eigenvectors and eigenvalues
- Applies threshold function with preserved parameters
- Batched neighbor lookup + OpenMP parallelization
- Returns encoder closure matching `landmark_encoder` pattern

## Required New Function: hlth.nystrom_encoder

Creates an encoder that performs Nyström extension to produce unsupervised codes from token features.

```lua
local encode_fn, n_dims = hlth.nystrom_encoder({
  features_index = inv_or_ann_or_hbi,  -- Index for token similarity computation
  eigenvectors = dvec,                  -- N×K continuous eigenvectors from spectral.encode
  eigenvalues = dvec,                   -- K eigenvalues from spectral.encode
  ids = ivec,                           -- N training sample IDs
  n_dims = integer,                     -- K dimensions
  threshold = function,                 -- Binarization function (captures learned params like ITQ rotation)
  k_neighbors = integer,                -- Number of training samples to use (0 = all)
  cmp = string,                         -- Similarity metric ("jaccard", "overlap", "dice", "tversky")
  cmp_alpha = number,                   -- Comparison alpha parameter (optional, default=0.5)
  cmp_beta = number,                    -- Comparison beta parameter (optional, default=0.5)
  probe_radius = integer,               -- ANN search radius (optional, default=2)
  rank_filter = integer,                -- Rank filter for similarity computation (optional, default=-1)
})
```

**Parameters:**
- `features_index`: Inverted index (inv), ANN index (ann), or Hamming ball index (hbi) over token features
- `eigenvectors`: Continuous eigenvectors (N×K matrix, row-major)
- `eigenvalues`: Eigenvalues (K elements)
- `ids`: Training sample IDs (N elements)
- `n_dims`: Number of dimensions K
- `threshold`: Binarization function that captures learned parameters (e.g., ITQ rotation matrix)
- `k_neighbors`: Number of training samples to use (0 = all samples, k > 0 = k nearest)
- `cmp`: Similarity metric for token-space neighbors
- `cmp_alpha`, `cmp_beta`: Comparison parameters (used for tversky similarity)
- `probe_radius`: Radius for ANN index probing
- `rank_filter`: Filter features by rank during similarity computation

**Returns**: `(encode_fn, n_dims)`
- `encode_fn(tokens_cvec, n_samples)` → `binary_codes_cvec`
- `n_dims` → dimension count

**Behavior**: For each test sample:
1. Query `features_index` for neighbors in token space (all or k-NN)
2. Compute token similarities to neighbors using `cmp` metric
3. Extend eigenvectors: `v_new[k] = (1/λ[k]) × Σ_i w[i] × V[i,k]`
4. Apply `threshold` function to binarize (reusing learned parameters)
5. Return binary codes

## Complete Implementation: newsgroups-nystrom.lua

```lua
local dvec = require("santoku.dvec")
local cvec = require("santoku.cvec")
local test = require("santoku.test")
local ds = require("santoku.tsetlin.dataset")
local utc = require("santoku.utc")
local ivec = require("santoku.ivec")
local str = require("santoku.string")
local eval = require("santoku.tsetlin.evaluator")
local inv = require("santoku.tsetlin.inv")
local ann = require("santoku.tsetlin.ann")
local graph = require("santoku.tsetlin.graph")
local hlth = require("santoku.tsetlin.hlth")
local optimize = require("santoku.tsetlin.optimize")
local spectral = require("santoku.tsetlin.spectral")
local itq = require("santoku.tsetlin.itq")
local tokenizer = require("santoku.tokenizer")

local cfg = {
  data = {
    max_per_class = nil,
    n_classes = 20,
  },
  tokenizer = {
    max_len = 20,
    min_len = 2,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 0,
    cgrams_max = 0,
    skips = 1,
    negations = 0,
  },
  feature_selection = {
    min_df = -2,
    max_df = 1.0,
    max_vocab = 16384,
  },
  ann = {
    bucket_size = nil,
  },
  nystrom = {
    k_neighbors = 0,       -- Use all training samples (0 = all, k > 0 = k nearest)
    cmp = "jaccard",       -- Similarity metric ("jaccard", "overlap", "dice", "tversky")
    cmp_alpha = 0.5,       -- Comparison alpha parameter (used for tversky)
    cmp_beta = 0.5,        -- Comparison beta parameter (used for tversky)
    probe_radius = 2,      -- ANN search radius (for ann/hbi indices)
    rank_filter = -1,      -- Rank filter for similarity computation (-1 = all ranks)
  },
  cluster = {
    enabled = true,
    verbose = true,
    knn = 64,
  },
  encoder = {
    enabled = true,
    verbose = true,
  },
  tm = {
    clauses = 8, -- { def = 8, min = 4, max = 16, log = true, int = true },
    clause_tolerance = { def = 7, min = 4, max = 16, log = true, int = true },
    clause_maximum = { def = 22, min = 8, max = 32, log = true, int = true },
    target = { def = 8, min = 2, max = 16, log = true, int = true },
    specificity = { def = 12, min = 2, max = 1000, log = true, int = true },
  },
  tm_search = {
    patience = 3,
    rounds = 6,
    trials = 6,
    tolerance = 1e-4,
    iterations = 8,
  },
  training = {
    patience = 20,
    iterations = 100,
  },
  search = {
    rounds = 1,
    patience = 3,
    adjacency_samples = 1,
    spectral_samples = 1,
    eval_samples = 1,
    adjacency_unsup = {
      knn = 29,
      knn_alpha = 14,
      knn_mutual = false,
      knn_mode = "cknn",
      knn_cache = 128,
      bridge = "mst",
    },
    adjacency_sup = {
      knn = 29,
      knn_alpha = 14,
      weight_decay = 4.04,
      knn_mutual = false,
      knn_mode = "cknn",
      knn_cache = 128,
      bridge = "mst",
    },
    spectral = {
      laplacian = "unnormalized",
      n_dims = 32,
      eps = 1e-8,
      threshold = {
        method = "itq",
        itq_iterations = 500,
        itq_tolerance = 1e-8,
        otsu_metric = "variance",
        otsu_minimize = false,
        otsu_n_bins = 32,
      },
    },
    eval = {
      knn = 32,
      anchors = 16,
      pairs = 64,
      ranking = "ndcg",
      metric = "min",
      elbow = "first_gap",
      elbow_alpha = 4.4,
      target = "combined",
      max_consecutive_zeros = 3,
    },
    cluster_eval = {
      elbow = "plateau",
      elbow_alpha = 0.01,
      elbow_target = "quality",
    },
    verbose = true,
  },
}

test("newsgroups-nystrom", function()

  local stopwatch = utc.stopwatch()

  print("Reading data")
  local train, test = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max_per_class)

  str.printf("  Train: %6d (%d categories)\n", train.n, train.n_labels)
  str.printf("  Test:  %6d (%d categories)\n", test.n, test.n_labels)

  print("\nTraining tokenizer")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  str.printf("  Vocabulary: %d tokens\n", n_tokens)

  print("\nTokenizing train")
  train.tokens = tok:tokenize(train.problems)

  print("\nApplying DF filter")
  local vocab, idf_weights = train.tokens:bits_top_df(
    train.n, n_tokens, nil, cfg.feature_selection.min_df, cfg.feature_selection.max_df)
  local n_vocab = vocab:size()
  str.printf("  Vocabulary: %d features (from %d total)\n", n_vocab, n_tokens)
  str.printf("  IDF range: %.3f - %.3f\n", idf_weights:get(n_vocab - 1), idf_weights:get(0))

  print("\nRe-tokenizing with filtered vocabulary")
  tok:restrict(vocab)
  train.tokens = tok:tokenize(train.problems)
  test.tokens = tok:tokenize(test.problems)

  print("\nCreating IDs")
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  test.ids = ivec.create(test.n)
  test.ids:fill_indices()
  test.ids:add(train.n)

  -- === UNSUPERVISED INDEX: tokens only ===
  print("\nBuilding unsupervised index (tokens, IDF-weighted)")
  train.index_unsup = inv.create({ features = idf_weights })
  train.index_unsup:add(train.tokens, train.ids)
  str.printf("  Index: %d tokens\n", n_vocab)

  -- === SUPERVISED INDEX: categories + tokens ===
  print("\nBuilding supervised index (categories + tokens)")
  local sup_problems = ivec.create()
  train.tokens:bits_select(nil, train.ids, n_vocab, sup_problems)
  local cat_ext = ivec.create()
  cat_ext:copy(train.solutions)
  cat_ext:add_scaled(cfg.data.n_classes)
  sup_problems:bits_extend(cat_ext, n_vocab, cfg.data.n_classes)

  local sup_weights = dvec.create(n_vocab + cfg.data.n_classes)
  for i = 0, n_vocab - 1 do
    sup_weights:set(i, idf_weights:get(i))
  end
  sup_weights:fill(1.0, n_vocab, n_vocab + cfg.data.n_classes)

  local sup_ranks = ivec.create(n_vocab + cfg.data.n_classes)
  sup_ranks:fill(1, 0, n_vocab)
  sup_ranks:fill(0, n_vocab, n_vocab + cfg.data.n_classes)

  train.index_sup = inv.create({
    features = sup_weights,
    ranks = sup_ranks,
    n_ranks = 2,
  })
  train.index_sup:add(sup_problems, train.ids)
  str.printf("  Index: %d features (%d tokens + %d categories)\n",
    n_vocab + cfg.data.n_classes, n_vocab, cfg.data.n_classes)

  -- === BUILD GROUND TRUTH ===
  print("\nBuilding category ground truth")
  local function build_category_ground_truth(ids, solutions)
    local cat_index = inv.create({ features = cfg.data.n_classes })
    local data = ivec.create()
    data:copy(solutions)
    data:add_scaled(cfg.data.n_classes)
    cat_index:add(data, ids)
    data:destroy()

    local adj_expected_ids, adj_expected_offsets, adj_expected_neighbors, adj_expected_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.search.eval.anchors,
        random_pairs = cfg.search.eval.pairs,
      })

    return cat_index, {
      retrieval = {
        ids = adj_expected_ids,
        offsets = adj_expected_offsets,
        neighbors = adj_expected_neighbors,
        weights = adj_expected_weights,
      },
    }
  end

  train.cat_index, train.ground_truth = build_category_ground_truth(train.ids, train.solutions)

  -- === OPTIMIZE UNSUPERVISED SPECTRAL CODES ===
  print("\nOptimizing unsupervised spectral pipeline")
  train.unsup_result = optimize.spectral({
    index = train.index_unsup,
    knn_index = train.index_unsup,
    bucket_size = cfg.ann.bucket_size,
    rounds = cfg.search.rounds,
    patience = cfg.search.patience,
    adjacency_samples = cfg.search.adjacency_samples,
    spectral_samples = cfg.search.spectral_samples,
    eval_samples = cfg.search.eval_samples,
    adjacency = cfg.search.adjacency_unsup,
    spectral = cfg.search.spectral,
    eval = cfg.search.eval,
    expected_ids = train.ground_truth.retrieval.ids,
    expected_offsets = train.ground_truth.retrieval.offsets,
    expected_neighbors = train.ground_truth.retrieval.neighbors,
    expected_weights = train.ground_truth.retrieval.weights,
    adjacency_each = cfg.search.verbose and function (ns, cs, es, stg)
      if stg == "kruskal" or stg == "done" then
        str.printf("    [unsup] graph: %d nodes, %d edges, %d components\n", ns, es, cs)
      end
    end or nil,
    spectral_each = cfg.search.verbose and function (t, s)
      if t == "done" then
        str.printf("    [unsup] spectral: %d matvecs\n", s)
      end
    end or nil,
    each = cfg.search.verbose and function(info)
      if info.event == "round_start" then
        str.printf("\n=== Unsupervised Round %d/%d ===\n", info.round, info.rounds)
      elseif info.event == "round_end" then
        str.printf("\n--- Unsup Round %d complete: best=%.4f (global=%.4f) ---\n",
          info.round, info.round_best_score, info.global_best_score)
      elseif info.event == "stage" and info.stage == "adjacency" then
        local p = info.params.adjacency
        if info.is_final then
          str.printf("\n  [Unsup Final] knn=%d alpha=%d mutual=%s mode=%s\n",
            p.knn, p.knn_alpha, tostring(p.knn_mutual), p.knn_mode)
        else
          str.printf("\n  [Unsup %d] knn=%d alpha=%d mutual=%s mode=%s\n",
            info.sample, p.knn, p.knn_alpha, tostring(p.knn_mutual), p.knn_mode)
        end
      elseif info.event == "stage" and info.stage == "spectral" then
        local p = info.params.spectral
        local thresh_str = "unknown"
        if p.threshold_params and p.threshold_params.method then
          thresh_str = p.threshold_params.method
        elseif type(p.threshold) == "string" then
          thresh_str = p.threshold
        end
        str.printf("    [unsup] lap=%s dims=%d thresh=%s\n", p.laplacian, p.n_dims, thresh_str)
      elseif info.event == "eval" then
        local e = info.params.eval
        local m = info.metrics
        local alpha_str = e.elbow_alpha and str.format("%.1f", e.elbow_alpha) or "-"
        str.printf("    [unsup] elbow=%s(%s) knn=%d score=%.4f quality=%.4f combined=%.4f\n",
          e.elbow, alpha_str, e.knn, m.score, m.quality, m.combined)
      end
    end or nil,
  })

  train.codes_unsup = train.unsup_result.codes
  train.ids_unsup = train.unsup_result.ids
  train.raw_codes_unsup = train.unsup_result.raw_codes
  train.eigenvalues_unsup = train.unsup_result.eigs
  train.dims_unsup = train.unsup_result.dims
  train.threshold_state_unsup = train.unsup_result.threshold_state

  -- === OPTIMIZE SUPERVISED SPECTRAL CODES ===
  print("\nOptimizing supervised spectral pipeline")
  train.sup_result = optimize.spectral({
    index = train.index_sup,
    knn_index = train.index_unsup,
    bucket_size = cfg.ann.bucket_size,
    rounds = cfg.search.rounds,
    patience = cfg.search.patience,
    adjacency_samples = cfg.search.adjacency_samples,
    spectral_samples = cfg.search.spectral_samples,
    eval_samples = cfg.search.eval_samples,
    adjacency = cfg.search.adjacency_sup,
    spectral = cfg.search.spectral,
    eval = cfg.search.eval,
    expected_ids = train.ground_truth.retrieval.ids,
    expected_offsets = train.ground_truth.retrieval.offsets,
    expected_neighbors = train.ground_truth.retrieval.neighbors,
    expected_weights = train.ground_truth.retrieval.weights,
    adjacency_each = cfg.search.verbose and function (ns, cs, es, stg)
      if stg == "kruskal" or stg == "done" then
        str.printf("    [sup] graph: %d nodes, %d edges, %d components\n", ns, es, cs)
      end
    end or nil,
    spectral_each = cfg.search.verbose and function (t, s)
      if t == "done" then
        str.printf("    [sup] spectral: %d matvecs\n", s)
      end
    end or nil,
    each = cfg.search.verbose and function(info)
      if info.event == "round_start" then
        str.printf("\n=== Supervised Round %d/%d ===\n", info.round, info.rounds)
      elseif info.event == "round_end" then
        str.printf("\n--- Sup Round %d complete: best=%.4f (global=%.4f) ---\n",
          info.round, info.round_best_score, info.global_best_score)
      elseif info.event == "stage" and info.stage == "adjacency" then
        local p = info.params.adjacency
        if info.is_final then
          str.printf("\n  [Sup Final] decay=%.2f knn=%d alpha=%d mutual=%s mode=%s\n",
            p.weight_decay, p.knn, p.knn_alpha, tostring(p.knn_mutual), p.knn_mode)
        else
          str.printf("\n  [Sup %d] decay=%.2f knn=%d alpha=%d mutual=%s mode=%s\n",
            info.sample, p.weight_decay, p.knn, p.knn_alpha, tostring(p.knn_mutual), p.knn_mode)
        end
      elseif info.event == "stage" and info.stage == "spectral" then
        local p = info.params.spectral
        local thresh_str = "unknown"
        if p.threshold_params and p.threshold_params.method then
          thresh_str = p.threshold_params.method
        elseif type(p.threshold) == "string" then
          thresh_str = p.threshold
        end
        str.printf("    [sup] lap=%s dims=%d thresh=%s\n", p.laplacian, p.n_dims, thresh_str)
      elseif info.event == "eval" then
        local e = info.params.eval
        local m = info.metrics
        local alpha_str = e.elbow_alpha and str.format("%.1f", e.elbow_alpha) or "-"
        str.printf("    [sup] elbow=%s(%s) knn=%d score=%.4f quality=%.4f combined=%.4f\n",
          e.elbow, alpha_str, e.knn, m.score, m.quality, m.combined)
      end
    end or nil,
  })

  train.codes_sup = train.sup_result.codes
  train.ids_sup = train.sup_result.ids
  train.index_sup_spectral = train.sup_result.index
  train.dims_sup = train.sup_result.dims

  -- === NYSTRÖM ENCODER ===
  print("\nCreating Nyström encoder")

  -- Create threshold function that reuses learned parameters from optimize.spectral
  local threshold_fn = function(codes, n_dims)
    local state = train.threshold_state_unsup
    local method = train.unsup_result.threshold_params.method

    if method == "itq" and state.rotation and state.mean then
      return itq.itq({
        codes = codes,
        n_dims = n_dims,
        rotation = state.rotation,
        mean = state.mean,
      })
    elseif method == "median" and state.thresholds then
      return itq.median({
        codes = codes,
        n_dims = n_dims,
        thresholds = state.thresholds,
      })
    elseif method == "otsu" and state.thresholds and state.indices then
      return itq.otsu({
        codes = codes,
        n_dims = n_dims,
        thresholds = state.thresholds,
        indices = state.indices,
      })
    else
      -- Fallback to stateless or recompute
      return train.unsup_result.threshold_fn(codes, n_dims)
    end
  end

  local nystrom_encode, n_latent = hlth.nystrom_encoder({
    features_index = train.index_unsup,
    eigenvectors = train.raw_codes_unsup,
    eigenvalues = train.eigenvalues_unsup,
    ids = train.ids_unsup,
    n_dims = train.dims_unsup,
    threshold = threshold_fn,
    k_neighbors = cfg.nystrom.k_neighbors,
    cmp = cfg.nystrom.cmp,
    cmp_alpha = cfg.nystrom.cmp_alpha,
    cmp_beta = cfg.nystrom.cmp_beta,
    probe_radius = cfg.nystrom.probe_radius,
    rank_filter = cfg.nystrom.rank_filter,
  })
  str.printf("  Encoder: %d dims\n", n_latent)

  -- === WARPING ENCODER ===
  if cfg.encoder.enabled then
    print("\nTraining warping encoder: unsup → sup")

    local warping_codes_unsup = train.codes_unsup
    warping_codes_unsup:bits_flip_interleave(train.dims_unsup)

    train.warping_tm, train.warping_accuracy, train.warping_params = optimize.encoder({
      hidden = train.dims_sup,
      sentences = warping_codes_unsup,
      visible = train.dims_unsup,
      codes = train.codes_sup,
      samples = train.n,
      clauses = cfg.tm.clauses,
      clause_tolerance = cfg.tm.clause_tolerance,
      clause_maximum = cfg.tm.clause_maximum,
      target = cfg.tm.target,
      specificity = cfg.tm.specificity,
      search_patience = cfg.tm_search.patience,
      search_rounds = cfg.tm_search.rounds,
      search_trials = cfg.tm_search.trials,
      search_iterations = cfg.tm_search.iterations,
      search_tolerance = cfg.tm_search.tolerance,
      final_patience = cfg.training.patience,
      final_iterations = cfg.training.iterations,
      search_metric = function (t, enc_info)
        local predicted = t:predict(enc_info.sentences, enc_info.samples)
        local accuracy = eval.encoding_accuracy(predicted, train.codes_sup, enc_info.samples, train.dims_sup)
        return accuracy.mean_hamming, accuracy
      end,
      each = function (_, is_final, train_accuracy, params, epoch, round, trial)
        local d, dd = stopwatch()
        if is_final then
          str.printf("  Time %3.2f %3.2f  Finalizing  C=%d LF=%d L=%d T=%d S=%.2f  Epoch  %d\n",
            d, dd, params.clauses, params.clause_tolerance, params.clause_maximum,
            params.target, params.specificity, epoch)
        else
          str.printf("  Time %3.2f %3.2f  Exploring  C=%d LF=%d L=%d T=%d S=%.2f  R=%d T=%d  Epoch  %d\n",
            d, dd, params.clauses, params.clause_tolerance, params.clause_maximum,
            params.target, params.specificity, round, trial, epoch)
        end
        str.printi("    Warping | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)",
          train_accuracy)
      end,
    })

    print("\nFinal warping encoder performance")
    str.printi("  Train | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)",
      train.warping_accuracy)
    print("\n  Best TM params:")
    str.printf("    clauses=%d clause_tolerance=%d clause_maximum=%d target=%d specificity=%.2f\n",
      train.warping_params.clauses,
      train.warping_params.clause_tolerance,
      train.warping_params.clause_maximum,
      train.warping_params.target,
      train.warping_params.specificity)

    -- === OUT-OF-SAMPLE EXTENSION ===
    print("\n=== OUT-OF-SAMPLE EXTENSION ===")

    print("\nStep 1: Nyström extension (tokens → unsup codes)")
    local test_codes_unsup = nystrom_encode(test.tokens, test.n)
    str.printf("  Test unsupervised codes: %d samples × %d dims\n", test.n, train.dims_unsup)

    print("\nStep 2: Warping (unsup codes → sup codes)")
    test_codes_unsup:bits_flip_interleave(train.dims_unsup)
    local test_codes_sup = train.warping_tm:predict(test_codes_unsup, test.n)
    str.printf("  Test supervised codes: %d samples × %d dims\n", test.n, train.dims_sup)

    -- === EVALUATION ===
    print("\nIndexing predicted codes")
    -- Use the optimized supervised index from optimize.spectral
    local idx_train_sup = train.index_sup_spectral

    local idx_test_pred = ann.create({
      features = train.dims_sup,
      expected_size = test.n,
    })
    idx_test_pred:add(test_codes_sup, test.ids)

    print("\nBuilding ground truth for test")
    test.cat_index = inv.create({ features = cfg.data.n_classes })
    local test_cat_data = ivec.create()
    test_cat_data:copy(test.solutions)
    test_cat_data:add_scaled(cfg.data.n_classes)
    test.cat_index:add(test_cat_data, test.ids)

    local test_expected_ids, test_expected_offsets, test_expected_neighbors, test_expected_weights =
      graph.adjacency({
        category_index = test.cat_index,
        category_anchors = cfg.search.eval.anchors,
        random_pairs = cfg.search.eval.pairs,
      })

    print("\nBuilding retrieved adjacency for test")
    local test_retrieved_ids, test_retrieved_offsets, test_retrieved_neighbors, test_retrieved_weights =
      graph.adjacency({
        weight_index = idx_test_pred,
        seed_ids = test_expected_ids,
        seed_offsets = test_expected_offsets,
        seed_neighbors = test_expected_neighbors,
      })

    print("\nEvaluating test predictions")
    local test_stats = eval.score_retrieval({
      retrieved_ids = test_retrieved_ids,
      retrieved_offsets = test_retrieved_offsets,
      retrieved_neighbors = test_retrieved_neighbors,
      retrieved_weights = test_retrieved_weights,
      expected_ids = test_expected_ids,
      expected_offsets = test_expected_offsets,
      expected_neighbors = test_expected_neighbors,
      expected_weights = test_expected_weights,
      ranking = "ndcg",
      metric = "min",
      elbow = "lmethod",
      n_dims = train.dims_sup,
    })

    str.printf("\nNyström+Warping Retrieval Results\n")
    str.printf("  Test | Score: %.4f | Quality: %.4f | Combined: %.4f\n",
      test_stats.score, test_stats.quality, test_stats.combined)

    print("\nSpot-check: Test k-NN Classification")
    local max_k = 10
    local knn_ids, knn_offsets, knn_neighbors = graph.adjacency({
      knn_index = idx_train_sup,
      query_index = idx_test_pred,
      knn = max_k,
      bridge = "none",
    })

    local id_to_idx = {}
    for i = 0, knn_ids:size() - 1 do
      id_to_idx[knn_ids:get(i)] = i
    end

    local k_values = { 1, 3, 5, 10 }
    for _, k in ipairs(k_values) do
      local correct = 0
      local total_precision = 0
      local counted = 0
      for i = 0, test.n - 1 do
        local query_id = test.ids:get(i)
        local idx = id_to_idx[query_id]
        if idx then
          local query_label = test.solutions:get(i)
          local offset_start = knn_offsets:get(idx)
          local offset_end = knn_offsets:get(idx + 1)
          local votes = {}
          local same_class = 0
          local neighbor_count = 0
          for j = offset_start, offset_end - 1 do
            local neighbor_idx = knn_neighbors:get(j)
            local neighbor_id = knn_ids:get(neighbor_idx)
            if neighbor_id >= 0 and neighbor_id < train.n then
              neighbor_count = neighbor_count + 1
              if neighbor_count <= k then
                local neighbor_label = train.solutions:get(neighbor_id)
                votes[neighbor_label] = (votes[neighbor_label] or 0) + 1
                if neighbor_label == query_label then
                  same_class = same_class + 1
                end
              end
            end
          end
          if neighbor_count > 0 then
            local best_label, best_count = -1, 0
            for lbl, cnt in pairs(votes) do
              if cnt > best_count then
                best_label, best_count = lbl, cnt
              end
            end
            if best_label == query_label then
              correct = correct + 1
            end
            total_precision = total_precision + same_class / k
            counted = counted + 1
          end
        end
      end
      if counted > 0 then
        str.printf("  k=%2d: Accuracy = %.2f%% | Precision = %.2f%%\n",
          k, 100 * correct / counted, 100 * total_precision / counted)
      end
    end
  end

  collectgarbage("collect")

end)
```

## C Implementation: hlth.nystrom_encoder

### Data Structure

```c
typedef struct {
  tk_inv_t *feat_inv;
  tk_ann_t *feat_ann;
  tk_hbi_t *feat_hbi;
  tk_dvec_t *eigenvectors;
  tk_dvec_t *eigenvalues;
  tk_ivec_t *ids;
  uint64_t n_dims;
  uint64_t k_neighbors;
  tk_ivec_sim_type_t cmp;
  double cmp_alpha;
  double cmp_beta;
  uint64_t probe_radius;
  int64_t rank_filter;
  int threshold_fn_ref;
  bool destroyed;
} tk_nystrom_encoder_t;
```

### Garbage Collection

```c
static int tk_nystrom_encoder_gc(lua_State *L) {
  tk_nystrom_encoder_t *enc = tk_nystrom_encoder_peek(L, 1);
  enc->feat_inv = NULL;
  enc->feat_ann = NULL;
  enc->feat_hbi = NULL;
  enc->eigenvectors = NULL;
  enc->eigenvalues = NULL;
  enc->ids = NULL;
  enc->destroyed = true;
  return 0;
}

static inline tk_nystrom_encoder_t *tk_nystrom_encoder_peek(lua_State *L, int i) {
  return (tk_nystrom_encoder_t *)luaL_checkudata(L, i, TK_NYSTROM_ENCODER_MT);
}
```

### Encoder Creation Function

```c
static int tk_hlth_nystrom_encoder_lua(lua_State *L) {
  lua_settop(L, 1);
  luaL_checktype(L, 1, LUA_TTABLE);

  // Parse features_index (inv, ann, or hbi)
  lua_getfield(L, 1, "features_index");
  tk_inv_t *feat_inv = tk_inv_peekopt(L, -1);
  tk_ann_t *feat_ann = tk_ann_peekopt(L, -1);
  tk_hbi_t *feat_hbi = tk_hbi_peekopt(L, -1);
  lua_pop(L, 1);
  if (!feat_inv && !feat_ann && !feat_hbi)
    return luaL_error(L, "nystrom_encoder: features_index must be inv, ann, or hbi");

  // Parse eigenvectors, eigenvalues, ids
  tk_dvec_t *eigenvectors = tk_lua_fget(L, 1, "nystrom_encoder", "eigenvectors", tk_dvec_peek);
  tk_dvec_t *eigenvalues = tk_lua_fget(L, 1, "nystrom_encoder", "eigenvalues", tk_dvec_peek);
  tk_ivec_t *ids = tk_lua_fget(L, 1, "nystrom_encoder", "ids", tk_ivec_peek);
  uint64_t n_dims = tk_lua_fgetunsigned(L, 1, "nystrom_encoder", "n_dims");

  // Validate dimensions
  if (n_dims == 0)
    return luaL_error(L, "nystrom_encoder: n_dims must be > 0");
  if (eigenvalues->n != n_dims)
    return luaL_error(L, "nystrom_encoder: eigenvalues size (%llu) != n_dims (%llu)",
                      (unsigned long long)eigenvalues->n, (unsigned long long)n_dims);
  if (eigenvectors->n % n_dims != 0)
    return luaL_error(L, "nystrom_encoder: eigenvectors size (%llu) not divisible by n_dims (%llu)",
                      (unsigned long long)eigenvectors->n, (unsigned long long)n_dims);
  uint64_t n_training_samples = eigenvectors->n / n_dims;
  if (ids->n != n_training_samples)
    return luaL_error(L, "nystrom_encoder: ids size (%llu) != eigenvector rows (%llu)",
                      (unsigned long long)ids->n, (unsigned long long)n_training_samples);

  // Parse threshold function
  lua_getfield(L, 1, "threshold");
  if (!lua_isfunction(L, -1))
    return luaL_error(L, "nystrom_encoder: threshold must be a function");
  int threshold_ref = luaL_ref(L, LUA_REGISTRYINDEX);

  // Parse optional params
  uint64_t k_neighbors = tk_lua_foptunsigned(L, 1, "nystrom_encoder", "k_neighbors", 0);
  const char *cmp_str = tk_lua_foptstring(L, 1, "nystrom_encoder", "cmp", "jaccard");
  tk_ivec_sim_type_t cmp = tk_hlth_parse_similarity_type(cmp_str);
  double cmp_alpha = tk_lua_foptnumber(L, 1, "nystrom_encoder", "cmp_alpha", 0.5);
  double cmp_beta = tk_lua_foptnumber(L, 1, "nystrom_encoder", "cmp_beta", 0.5);
  uint64_t probe_radius = tk_lua_foptunsigned(L, 1, "nystrom_encoder", "probe_radius", 2);
  int64_t rank_filter = tk_lua_foptinteger(L, 1, "nystrom_encoder", "rank_filter", -1);

  // Create encoder
  tk_nystrom_encoder_t *enc = tk_lua_newuserdata(L, tk_nystrom_encoder_t,
    TK_NYSTROM_ENCODER_MT, tk_nystrom_encoder_mt_fns, tk_nystrom_encoder_gc);
  int Ei = lua_gettop(L);

  enc->feat_inv = feat_inv;
  enc->feat_ann = feat_ann;
  enc->feat_hbi = feat_hbi;
  enc->eigenvectors = eigenvectors;
  enc->eigenvalues = eigenvalues;
  enc->ids = ids;
  enc->n_dims = n_dims;
  enc->k_neighbors = k_neighbors;
  enc->cmp = cmp;
  enc->cmp_alpha = cmp_alpha;
  enc->cmp_beta = cmp_beta;
  enc->probe_radius = probe_radius;
  enc->rank_filter = rank_filter;
  enc->threshold_fn_ref = threshold_ref;
  enc->destroyed = false;

  // Create ephemeron table references to keep objects alive
  lua_getfield(L, 1, "features_index");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "eigenvectors");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "eigenvalues");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "ids");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  lua_getfield(L, 1, "threshold");
  tk_lua_add_ephemeron(L, TK_NYSTROM_ENCODER_EPH, Ei, -1);
  lua_pop(L, 1);

  // Create encoder closure
  lua_pushcclosure(L, tk_nystrom_encode_lua, 1);
  lua_pushinteger(L, n_dims);

  return 2;
}
```

### Encode Function

```c
static int tk_nystrom_encode_lua(lua_State *L) {
  tk_nystrom_encoder_t *enc = lua_touserdata(L, lua_upvalueindex(1));

  if (enc->destroyed)
    return luaL_error(L, "nystrom_encode: encoder has been destroyed");

  tk_ivec_t *query_ivec = tk_ivec_peekopt(L, 1);
  tk_cvec_t *query_cvec = query_ivec ? NULL : tk_cvec_peekopt(L, 1);

  if (!query_ivec && !query_cvec)
    return luaL_error(L, "nystrom_encode: expected ivec or cvec query");

  uint64_t n_samples = tk_lua_checkunsigned(L, 2, "n_samples");

  tk_inv_t *feat_inv = enc->feat_inv;
  tk_ann_t *feat_ann = enc->feat_ann;
  tk_hbi_t *feat_hbi = enc->feat_hbi;

  // === BATCH NEIGHBOR LOOKUP (like landmark_encoder) ===
  tk_ann_hoods_t *ann_hoods = NULL;
  tk_hbi_hoods_t *hbi_hoods = NULL;
  tk_inv_hoods_t *inv_hoods = NULL;
  tk_ivec_t *nbr_ids = NULL;

  if (feat_inv && query_ivec) {
    tk_inv_neighborhoods_by_vecs(L, feat_inv, query_ivec, enc->k_neighbors,
      0.0, 1.0, enc->cmp, enc->cmp_alpha, enc->cmp_beta,
      0.0, enc->rank_filter, &inv_hoods, &nbr_ids);
  } else if (feat_ann && query_cvec) {
    tk_ann_neighborhoods_by_vecs(L, feat_ann, query_cvec, enc->k_neighbors,
      enc->probe_radius, 0, ~0ULL, &ann_hoods, &nbr_ids);
  } else if (feat_hbi && query_cvec) {
    tk_hbi_neighborhoods_by_vecs(L, feat_hbi, query_cvec, enc->k_neighbors,
      0, ~0ULL, &hbi_hoods, &nbr_ids);
  } else {
    return luaL_error(L, "nystrom_encode: index/query type mismatch");
  }

  int stack_before_out = lua_gettop(L);

  // Create dvec for raw codes
  tk_dvec_t *raw_codes = tk_dvec_create(L, n_samples * enc->n_dims, 0, 0);
  raw_codes->n = n_samples * enc->n_dims;

  // Validate IDs are non-negative (iumap requires uint64_t keys)
  for (uint64_t i = 0; i < enc->ids->n; i++) {
    if (enc->ids->a[i] < 0)
      return luaL_error(L, "nystrom_encode: negative ID at index %llu: %lld",
                        (unsigned long long)i, (long long)enc->ids->a[i]);
  }

  // Build ID-to-index hash map for O(1) lookup in hot loop
  tk_iumap_t *id_to_idx = tk_iumap_from_ivec(NULL, enc->ids);
  if (!id_to_idx)
    tk_error(L, "nystrom_encode: failed to create ID mapping", ENOMEM);

  // === PARALLEL NYSTRÖM EXTENSION ===
  #pragma omp parallel
  {
    tk_ivec_t *tmp = tk_ivec_create(NULL, 0, 0, 0);
    tk_dvec_t *sims = tk_dvec_create(NULL, 0, 0, 0);

    #pragma omp for schedule(static)
    for (uint64_t i = 0; i < n_samples; i++) {
      tk_ivec_clear(tmp);
      tk_dvec_clear(sims);

      // Gather neighbors and extract similarities from hood structures
      int64_t nbr_idx, nbr_uid;
      TK_GRAPH_FOREACH_HOOD_NEIGHBOR(feat_inv, feat_ann, feat_hbi,
        inv_hoods, ann_hoods, hbi_hoods, i, 1.0, nbr_ids, nbr_idx, nbr_uid, {
          tk_ivec_push(tmp, nbr_uid);
        });

      // Extract similarities from hood structures
      if (inv_hoods && i < inv_hoods->n) {
        tk_rvec_t *hood = inv_hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j++) {
          double distance = hood->a[j].d;
          double sim = 1.0 - distance;  // Convert distance to similarity
          tk_dvec_push(sims, sim);
        }
      } else if (ann_hoods && i < ann_hoods->n) {
        tk_pvec_t *hood = ann_hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j++) {
          int64_t hamming = hood->a[j].p;
          double sim = 1.0 - ((double)hamming / (double)feat_ann->features);
          tk_dvec_push(sims, sim);
        }
      } else if (hbi_hoods && i < hbi_hoods->n) {
        tk_pvec_t *hood = hbi_hoods->a[i];
        for (uint64_t j = 0; j < hood->n; j++) {
          int64_t hamming = hood->a[j].p;
          double sim = 1.0 - ((double)hamming / (double)feat_hbi->features);
          tk_dvec_push(sims, sim);
        }
      }

      // Nyström extension: v_new[k] = (1/λ[k]) × Σ_j sim[j] × V[neighbor_j, k]
      for (uint64_t k = 0; k < enc->n_dims; k++) {
        double sum = 0.0;
        for (uint64_t j = 0; j < tmp->n && j < sims->n; j++) {
          uint64_t train_id = tmp->a[j];

          // O(1) hash map lookup instead of O(N) linear search
          khint_t khi = tk_iumap_get(id_to_idx, train_id);
          if (khi != tk_iumap_end(id_to_idx)) {
            uint64_t train_idx = tk_iumap_val(id_to_idx, khi);
            double v_jk = enc->eigenvectors->a[train_idx * enc->n_dims + k];
            sum += sims->a[j] * v_jk;
          }
        }
        double lambda_k = enc->eigenvalues->a[k];
        raw_codes->a[i * enc->n_dims + k] = sum / (lambda_k + 1e-10);
      }
    }

    tk_ivec_destroy(tmp);
    tk_dvec_destroy(sims);
  }

  // Cleanup hash map
  tk_iumap_destroy(id_to_idx);

  // Apply threshold function (calls Lua function with rotation matrix captured)
  lua_rawgeti(L, LUA_REGISTRYINDEX, enc->threshold_fn_ref);
  lua_pushvalue(L, stack_before_out + 1);  // raw_codes dvec
  lua_pushinteger(L, enc->n_dims);
  lua_call(L, 2, 1);  // threshold(raw_codes, n_dims) → binary_codes

  lua_replace(L, stack_before_out);
  lua_settop(L, stack_before_out);

  return 1;
}
```

### Helper Functions

```c
// Parse similarity type string
static tk_ivec_sim_type_t tk_hlth_parse_similarity_type(const char *cmp_str) {
  if (!strcmp(cmp_str, "jaccard")) return TK_IVEC_JACCARD;
  if (!strcmp(cmp_str, "overlap")) return TK_IVEC_OVERLAP;
  if (!strcmp(cmp_str, "dice")) return TK_IVEC_DICE;
  if (!strcmp(cmp_str, "tversky")) return TK_IVEC_TVERSKY;
  return TK_IVEC_JACCARD;
}
```

**Note**: The linear search helper `tk_hlth_find_id_index` has been removed in favor of `tk_iumap_from_ivec` for O(1) hash map lookup, which is critical for performance in the Nyström extension hot loop.

### Constants and Metatable

```c
#define TK_NYSTROM_ENCODER_MT "tk_nystrom_encoder_t"
#define TK_NYSTROM_ENCODER_EPH "tk_nystrom_encoder_eph"

static luaL_Reg tk_nystrom_encoder_mt_fns[] = {
  { NULL, NULL }
};

static luaL_Reg tk_hlth_fns[] = {
  { "landmark_encoder", tk_hlth_landmark_encoder_lua },
  { "nystrom_encoder", tk_hlth_nystrom_encoder_lua },
  { NULL, NULL }
};
```

## Implementation Notes

### Batched Neighbor Lookup
The encoder uses batch neighborhood queries (`tk_inv_neighborhoods_by_vecs`, `tk_ann_neighborhoods_by_vecs`, `tk_hbi_neighborhoods_by_vecs`) to find neighbors for all test samples simultaneously before the parallel loop. This is more efficient than per-sample queries and enables better cache utilization.

### OpenMP Parallelization
The Nyström extension loop is parallelized over test samples using `#pragma omp parallel for schedule(static)`. Each thread has thread-local allocations (tmp, sims) to avoid contention. This matches the pattern used in `landmark_encoder`.

### Similarity Extraction from Hoods
Hood structures store neighbors with distances/dissimilarities:
- `inv_hoods->a[i]`: `tk_rvec_t` with `tk_rank_t` elements containing `.d` (distance, 0.0 = identical)
- `ann_hoods->a[i]`: `tk_pvec_t` with `tk_pair_t` elements containing `.p` (Hamming distance)
- `hbi_hoods->a[i]`: `tk_pvec_t` with `tk_pair_t` elements containing `.p` (Hamming distance)

Convert to similarity: `sim = 1.0 - distance` for inv, `sim = 1.0 - (hamming / features)` for ann/hbi.

### Eigenvector Storage
Eigenvectors are stored row-major: `eigenvectors[sample_idx * n_dims + dim_idx]`. The Nyström extension computes weighted sums over neighbor eigenvectors for each dimension.

### Threshold Application
The threshold function captures the learned ITQ rotation matrix and mean vector (if ITQ was used) in a closure. When test codes are produced via Nyström extension, they are:
1. Centered using the **training mean** (same mean used for training codes)
2. Rotated using the **training rotation matrix** (same rotation used for training codes)
3. Binarized via sign thresholding

This ensures test codes are transformed identically to training codes, producing compatible binary embeddings. The C code stores the threshold function as a Lua registry reference and calls it once for the entire batch of raw codes.

### Memory Management
The encoder uses ephemeron tables (`tk_lua_add_ephemeron` with `TK_NYSTROM_ENCODER_EPH`) to store references to all input objects (index, eigenvectors, eigenvalues, ids). This prevents garbage collection while the encoder is alive, matching the pattern used in `landmark_encoder`.

### k_neighbors = 0 Behavior
When `k_neighbors = 0`, the batch neighborhood functions return all training samples instead of limiting to k neighbors. The inv/ann/hbi modules handle this internally.

### Dimension Selection Compatibility
When dimension selection is enabled in `optimize.spectral`, the returned `model.dims` may be less than the requested dimensions. The Nyström encoder uses `model.dims`, `model.raw_codes`, and `model.eigs` which are already the selected subset, so dimension selection works transparently.
