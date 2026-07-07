# Using santoku-learn

Worked examples for the modelling flow. See the [README](../README.md) for orientation. **The
tests are the spec**: every section names the test that covers the rest. Data types
(`csr`/`mtx`/`spans`) are santoku-matrix types; see its
[usage guide](../../lua-santoku-matrix/doc/usage.md).

## The supervised spine

Almost every task is the same shape: **features (csr) → weight → `optimize.krr` → deploy with the
returned encoder + ridge + decider.** Features flow in through a `rebuild(params)` callback: the spec
owns assembly (weights → select → hcat → standardize → L2), `optimize.krr` owns the search and hands
the callback a bundle of the knobs it searches (`thresh`/`scales`/`exponent`, per block). Here it is
end to end (the classifiers in `regress/{newsgroups,imdb,eurlex}.lua` do the same per-block over a
byte+word bundle via `util.fold_blocks`):

```lua
local tokenizer = require("santoku.learn.tokenizer")
local optimize  = require("santoku.learn.optimize")
local spectral  = require("santoku.learn.spectral")
local util      = require("santoku.learn.util")

-- 1. text -> sparse features (csr); fit supervised weights (bns scales X in place, leaves it raw)
local tok = tokenizer.create({ ngram_min = 5, ngram_max = 5 })
local X   = tok:fit({ texts = train_texts })
local bns = X:bns(train_labels)
local Xv  = tok:tokenize({ texts = val_texts }); Xv:bns(bns)

-- 2. searched feature weighting/pruning (per-block thresh/scales/exponent knobs over weighted
--    blocks) is baked by util.fold_blocks; see the regress specs for the full pool_blocks flow

-- 3. one call: per-trial rebuild(params) -> Nyström encode -> ridge solve -> decider calibrated on val
local enc, ridge, _, best, decider = optimize.krr({
  y = train_labels, val_y = val_labels,
  x = X, val_x = Xv,
  kernel = { "matern", "cosine", "arccos" }, n_landmarks = 8192,
  lambda = { def = 1e-2 }, k = 1, search_trials = 12,    -- 0 = locked at the defs
})

-- 4. persist what you need to deploy
enc:persist("model.nystrom"); ridge:persist("model.ridge"); decider:persist("model.decide")
tok:persist("model.tok")

-- 5. deploy on new data: re-encode through the same tok + bns
enc = spectral.load("model.nystrom")
local codes = enc:encode(tok:tokenize({ texts = test_texts }))   -- -> mtx
local P     = ridge:label(codes, 1)                              -- -> P csr (top-k labels)
local _, m  = decider:score({ pred = P, expected = test_labels, n_samples = #test_texts })
```

When you don't need searchable assembly, pass `x`/`val_x` directly instead of `rebuild` + knobs
(`regress/{housing,mnist}.lua` wrap their fixed features in a no-knob `rebuild` for one uniform path).

`optimize.krr` operates on a **baked** representation only: pass `x`/`val_x` (csr/dense) or `blocks`, or a
`rebuild(params[, fold])` that returns baked views, plus `y`/`val_y` (and for CV `folds`+`fold_y`/
`fold_val_y`/`fold_val_n`). It never tokenizes or weights features. The harness bakes (tokenize → bns →
sort) and builds that fold split with a `util` helper that returns the engine args (incl. the per-fold
`rebuild`), so the call reads `optimize.krr(util.fold_blocks({ ... }))`:

- `util.fold_blocks` — baked per-family `pool_blocks`/`test_blocks` (weights per fold, per-block
  thresh/scales/exponent knobs).
- `util.fold_dense` — a dense `pool_codes`/`test_codes` matrix (no bns).
- anything more custom (e.g. conll-full's 9-block type head) builds `fold_y`/`rebuild` itself and calls
  `optimize.krr` directly — that's the base interface the helpers above just emit.

CV engages only when `search_trials>0` and `K>1`; otherwise a single full-pool/test fit. Either way
`optimize.krr` returns `enc, ridge, val_codes, best, decider`. Read any `regress/*.lua` for the harness.

## Variations (same spine, read the test for specifics)

- **Per-block feature knobs** (`regress/newsgroups.lua`, `imdb.lua`, `eurlex.lua`): weight each
  block (byte, word) once, then each block gets its own searched `thresh`/`scales`/`exponent`.
  Declare per-block vectors like `thresh = { def = { t_byte, t_word } }` (frozen at
  `search_trials = 0`) or `{ {min,max,log}, ... }` to search each block independently.
  `conll-full.lua`'s type head does this over many blocks.
- **Multiclass, single-label** (`regress/newsgroups.lua`, `regress/mnist.lua`): pass an `n_labels>1`
  one-label-per-row `y`; `optimize.krr` auto-selects single-label decode. Deploy with
  `ridge:regress(codes)` → dense scores → `decider:score({ scores = ..., expected = ... })`.
  Report **both** an uncalibrated argmax (`decide.create({ n_labels, single = true })`) and the
  calibrated decider; argmax often wins on test.
- **Extreme multi-label / XMLC** (`regress/eurlex.lua`): multi-label `y`; set `k` (top-k per row)
  and pass mmap buffers (`chol_buf`/`xtx_buf`/`xty_buf`/`w_buf`) to bound RAM.
- **Regression** (`regress/housing.lua`): pass `targets`/`val_targets` (dvec) instead of `y`/`val_y`;
  score with `eval.regress_accuracy(ridge:regress(codes), targets)`. Mixed cat+continuous features
  are assembled with matrix ops (`bits:hcat(continuous:to_sparse())`, block standardize).
- **Retrieval / ANN** (`ann.lua`): encode a corpus to codes (mtx), `ann.create({ codes = M })`
  (signs internally), then `idx:neighborhoods_by_vecs(Q, k)` → P csr. Compare to exact
  `corpus:topk(Q, k)` (matrix) for ground truth.
- **Span NER** (`regress/conll-full.lua`, `regress/conll-gaz.lua`): a two-stage pipeline:
  `segmenter` learns byte-class segments, `aho` adds gazetteer candidates (→ spans), stage 1 tags
  inner tokens, candidates are enumerated/unioned (spans ops), stage 2 types them; `decide` runs
  span-mode (NMS) at both ends.

## `optimize.krr` contract

The front door. Inputs are matrix objects:
- `x`, `val_x`: feature `csr` (or dense `mtx` codes); **or** a `rebuild(params)` callback returning
  `{ x = ..., val_x = ... }` that assembles features at the trial's searched knobs (the spec owns
  bns/select/hcat/standardize/L2). `y`, `val_y`: label `csr` (or `targets`/`val_targets` dvec for
  regression).
- `kernel` (list to search over: `cosine`/`matern`/`arccos`), `n_landmarks`, `lambda`, `k`,
  `search_trials`, optional mmap `chol_buf`/`xtx_buf`/`xty_buf`/`w_buf`,
  `each` (progress callback via `util.make_ridge_log`).
- **Rebuild knobs** (the `params` bundle): `thresh` (per-block relevance floor), `scales`
  (per-block energy, gauge-normalized), `exponent` (per-block weight sharpening). Forms: scalar
  `{ min, max, log, def }`; per-block vector `{ {min,max,log}, ... }` or `{ def = { v1, v2, ... } }`
  (one searched dim per block, handed to `rebuild` as `p.<knob>[i]`). A bare `{ def = ... }` is the
  frozen value at `search_trials = 0`. Within a 1e-4 score tie the search prefers the smaller
  feature cost.
- Returns `enc, ridge, val_codes, params, decider, dec_metrics`.

**Locked vs search:** `search_trials = 0` runs locked at the `{ def = ... }` values (fast,
reproducible), the deployment mode. `search_trials > 0` runs GP-BO over kernel + `lambda`
+ any declared rebuild knobs; take the printed best and write it back as the new `def`s.

## The decide layer

`decide` is the single decision authority: the metric you *search* on is the metric you *deploy*.

```lua
local decide = require("santoku.learn.decide")
local d = decide.create({ n_labels = nl })          -- + single=true | span=true, reject=
d:calibrate({ pred = P, expected = E })             -- multilabel: learn a threshold on val
-- single: calibrate({ scores=, expected= }) ;  span: calibrate({ scores=, cand=, gold= })
local ks    = d:predict({ pred = P, n_samples = n })     -- per-sample accepts
local _, m  = d:score({ pred = P, expected = E, n_samples = n })   -- metric on test
d:persist(path); d = decide.load(path)
```

Modes: **single** (argmax / per-class offsets, macro-F1), **multilabel/binary** (threshold sweep,
micro-F1), **span** (NMS-DP over candidate spans, span-F1). An *uncalibrated* decider is the
baseline: single → pure argmax, span → reject_offset 0. Multilabel has no argmax baseline.

## Deploy / persistence checklist

Save the tokenizer(s), the encoder (`spectral`), the ridge, the decider, and any fit weights
(`bns`/`idf`/`standardize`). On load, re-encode new text through the same tokenizer + weights, then
`ridge:label`/`:regress` + `decider`. For large corpora, encode straight into an mmap `mtx`
(`enc:encode(X, out)`) to bound RAM.
