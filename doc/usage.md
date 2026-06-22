# Using santoku-learn

A by-example guide to the modelling flow. The [README](../README.md) is the map; this is the
cookbook. **The tests are the spec** — every section names the test that covers the rest. Data
types (`csr`/`mtx`/`spans`) are santoku-matrix types — see its
[usage guide](../../lua-santoku-matrix/doc/usage.md).

## The supervised spine

Almost every task is the same shape: **features (csr) → weight → `optimize.krr` → deploy with the
returned encoder + ridge + decider.** Here it is end to end (binary classification, from
`test/spec/santoku/learn/regress/imdb.lua`):

```lua
local tokenizer = require("santoku.learn.tokenizer")
local optimize  = require("santoku.learn.optimize")
local spectral  = require("santoku.learn.spectral")

-- 1. text -> sparse features (csr); supervised weight (fit); L2 normalize
local tok = tokenizer.create({ ngram_min = 5, ngram_max = 5 })
local X   = tok:fit({ texts = train_texts })
local bns = X:bns(train_labels)                 -- fit weights from labels csr
X:normalize()
local Xv  = tok:tokenize({ texts = val_texts }); Xv:bns(bns); Xv:normalize()

-- 2. one call: Nyström encode -> ridge solve -> decider calibrated on val
local enc, ridge, _, params, decider = optimize.krr({
  x = X, y = train_labels, val_x = Xv, val_y = val_labels,
  kernel = { "matern", "cosine", "arccos" }, n_landmarks = 8192,
  lambda = { def = 1e-2 }, k = 1, search_trials = 12,    -- 0 = locked at the defs
})

-- 3. persist what you need to deploy
enc:persist("model.nystrom"); ridge:persist("model.ridge"); decider:persist("model.decide")
tok:persist("model.tok")

-- 4. deploy on new data (reload, or reuse the live objects)
enc = spectral.load("model.nystrom")
local codes = enc:encode(tok:tokenize({ texts = test_texts }))   -- -> mtx
local P     = ridge:label(codes, 1)                              -- -> P csr (top-k labels)
local _, m  = decider:score({ pred = P, expected = test_labels, n_samples = #test_texts })
```

## Variations (same spine, read the test for specifics)

- **Multiclass, single-label** (`regress/newsgroups.lua`, `regress/mnist.lua`): pass an `n_labels>1`
  one-label-per-row `y`; `optimize.krr` auto-selects single-label decode. Deploy with
  `ridge:regress(codes)` → dense scores → `decider:score({ scores = ..., expected = ... })`.
  Report **both** an uncalibrated argmax (`decide.create({ n_labels, single = true })`) and the
  calibrated decider — argmax often wins on test.
- **Extreme multi-label / XMLC** (`regress/eurlex.lua`): multi-label `y`; set `k` (top-k per row),
  `propensity_a/b` for tail labels, and pass mmap buffers (`chol_buf`/`w_buf`/`pqty_buf`) to bound
  RAM. `eval.oracle_f1(P, E)` gives the achievable-F1 ceiling for diagnostics.
- **Regression** (`regress/housing.lua`): pass `targets`/`val_targets` (dvec) instead of `y`/`val_y`;
  score with `eval.regress_accuracy(ridge:regress(codes), targets)`. Mixed cat+continuous features
  are assembled with matrix ops (`bits:hcat(continuous:to_sparse())`, block standardize).
- **Retrieval / ANN** (`ann.lua`): encode a corpus to codes (mtx), `ann.create({ codes = M })`
  (signs internally), then `idx:neighborhoods_by_vecs(Q, k)` → P csr. Compare to exact
  `corpus:topk(Q, k)` (matrix) for ground truth.
- **Span NER** (`regress/conll.lua`): a two-stage pipeline — `segmenter` learns byte-class
  segments, `aho` adds gazetteer candidates (→ spans), stage 1 tags inner tokens, candidates are
  enumerated/unioned (spans ops), stage 2 types them; `decide` runs span-mode (NMS) at both ends.

## `optimize.krr` contract

The front door. Inputs are matrix objects:
- `x`, `val_x` — feature `csr` (or dense `mtx` codes); `y`, `val_y` — label `csr` (or `targets`/
  `val_targets` dvec for regression).
- `kernel` (list to search over — `cosine`/`matern`/`arccos`), `n_landmarks`, `lambda`, `k`,
  `propensity_a/b` (multilabel), `search_trials`, optional mmap `chol_buf`/`w_buf`/`pqty_buf`,
  `each` (progress callback via `util.make_ridge_log`).
- Returns `enc, ridge, val_codes, params, decider, dec_metrics`.

**Locked vs search:** `search_trials = 0` runs locked at the `{ def = ... }` values — fast,
reproducible, the deployment mode. `search_trials > 0` runs GP-BO over kernel + `lambda`
(+ propensity); take the printed best and write it back as the new `def`s. `optimize.ridge` is the
same minus the spectral step (you supply codes/gram).

## The decide layer

`decide` is the single decision authority — the metric you *search* on is the metric you *deploy*.

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
