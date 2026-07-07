# santoku-learn

A KRR-centric ML toolkit: turn text/features into a **spectral (Nyström) embedding**, fit
**ridge regression** over it, and make **calibrated decisions** for binary, multiclass,
extreme multi-label (XMLC), regression, span NER, and ANN retrieval. The whole modelling
path is wired together by `optimize.krr`.

This README is a usage guide, not an API reference. **The tests are the spec**: each
component points at the test that exercises its full surface.

Data structures (`csr` sparse features, `mtx` dense codes, `spans` labelled intervals) are
**santoku-matrix** types; this doc uses them but doesn't re-explain them; see the
[matrix README](../lua-santoku-matrix/README.md). The shorthand: features in/out are `csr`,
embeddings are `mtx`, predictions are a `csr` of top-k `(label, score)`, span sets are `spans`.

## The core pipeline

`optimize.krr` is the front door: it ties spectral encode + ridge + a calibrated decider into
one call that either runs locked at fixed params or searches (GP-BO) over them.

```lua
local tokenizer = require("santoku.learn.tokenizer")
local optimize  = require("santoku.learn.optimize")
local util      = require("santoku.learn.util")

-- text -> sparse features (csr); fit supervised weights (bns scales X in place, leaving it raw)
local tok = tokenizer.create({ ngram_min = 5, ngram_max = 5 })
local X   = tok:fit({ texts = train_texts })
local bns = X:bns(train_labels)        -- fit weights from labels csr; apply later with the same w
local Xv  = tok:tokenize({ texts = val_texts }); Xv:bns(bns)

-- searched feature weighting/pruning (per-block thresh/scales/exponent knobs over weighted blocks)
-- is baked by util.fold_blocks; see the regress specs for the full pool_blocks flow

-- one call: per-trial rebuild(params) -> Nyström encode -> ridge solve -> decider calibrated on val
local enc, ridge, _, best, decider = optimize.krr({
  y = train_labels, val_y = val_labels,
  x = X, val_x = Xv,
  kernel = { "matern", "cosine", "arccos" }, n_landmarks = 8192,
  lambda = { def = 1e-2 }, k = 1, search_trials = 12,        -- 0 = locked at the defs
})

-- deploy on new data: re-encode through the same tok + bns
local codes = enc:encode(tok:tokenize({ texts = test_texts }))   -- -> mtx
local P     = ridge:label(codes, 1)                                                -- -> P csr (top-k labels)
local _, m  = decider:score({ pred = P, expected = test_labels, n_samples = #test_texts })
```

`rebuild(params)` is the universal feature-assembly hook: the spec owns assembly (weights → select →
hcat → standardize → L2), `optimize.krr` owns the search. `params` is a bundle of the searched knobs
(`thresh`/`scales`/`exponent`, scalar or per-block vector specs). Pass `x`/`val_x` directly instead
when you don't need searchable assembly. See the `regress/` suite: `conll-gaz.lua`,
`conll-full.lua` (per-block vectors), `newsgroups`/`imdb`/`eurlex` (per-block byte+word).

## Components

| Module | Role | Anchor test |
|--------|------|-------------|
| `tokenizer` | text → n-gram features (`csr`); `focus`/`types` are `spans` | `tokenizer.lua` |
| `spectral`  | kernel Nyström / RP-Cholesky embedding (`csr`\|`mtx` → `mtx` codes) | regress suite |
| `ridge`     | ridge solve; `:label`→P csr, `:regress`→fvec, `:gram` | regress suite |
| `decide`    | the sole decision layer: `calibrate`/`predict`/`score` (single \| multilabel \| span) | `regress/mnist.lua`, `regress/conll-full.lua` |
| `optimize`  | `krr`: GP-BO search + encode/solve/decide over a **baked** representation (`x`/`blocks`/`rebuild`); folds via `fold_y`/`rebuild(p,fold)` | regress suite |
| `util`      | logging + `tokenize_blocks` / `fold_blocks` / `fold_dense` (bake pool→fold split + `rebuild` for `optimize.krr`) | regress suite |
| `evaluator` | `regress_accuracy` | `regress/housing.lua` |
| `ann`       | binary-LSH retrieval over codes, optional float rerank | `ann.lua` |
| `aho`       | Aho-Corasick gazetteer matching → `spans` | `aho.lua` |
| `segmenter` | learned byte-class segmentation (the shape signal) | `regress/conll-full.lua` |
| `booleanizer` | categorical + continuous featurizer → bits `csr` [+ dense `mtx`] | `booleanizer.lua` |
| `ner`       | span-NER helpers (extends the `spans` metatable) | `ner.lua`, `regress/conll-full.lua` |
| `dataset`   | dataset loaders (imdb/newsgroups/mnist/eurlex/conll/housing) | regress suite |
| `util`      | metric formatting + search logging | - |

## Methodology & ergonomics

- **`optimize.krr` is the front door.** Pass objects (`x`/`y`/`val_x`/`val_y`), or a
  `rebuild(params)` callback + searched knob specs in place of `x`/`val_x`; it returns
  `enc, ridge, val_codes, params, decider, dec_metrics`.
- **Locked vs search.** `search_trials = 0` runs locked at the `{ def = ... }` params (fast,
  reproducible), the deployment mode. `search_trials > 0` runs GP-BO over kernel + `lambda`
  + any declared rebuild knobs (`thresh`/`scales`/`exponent`); lock the winning params
  back as the new defs.
- **Search == deploy.** Every search trial scores through the **same `decide`** used at
  deployment, no separate selection metric. `decide` is the one decision authority (threshold
  for multilabel, argmax for single, NMS for span); you `calibrate` it on val and
  `score`/`predict` with it on test.
- **argmax vs decide.** For single-label and span, report both an uncalibrated argmax baseline
  and the calibrated decider (argmax often wins on test). Multilabel has no argmax baseline.
- **fit / apply.** Feature transforms return their learned params: `local w = X:bns(Y)` then
  `Xv:bns(w)` (same for `:idf`, `:standardize`). (These are matrix `csr` methods.)
- **persist / load.** `tokenizer`, `spectral` encoders, `ridge`, and `decide` all round-trip to
  disk; large code matrices can be encoded straight into an mmap `mtx` (`enc:encode(X, out)`).

## Scenarios → which test to read

| Task | Test |
|------|------|
| binary classification | `regress/imdb.lua` |
| multiclass (single-label) | `regress/newsgroups.lua`, `regress/mnist.lua` |
| extreme multi-label (XMLC) | `regress/eurlex.lua` |
| regression | `regress/housing.lua` |
| span NER (segment → tag → type) | `regress/conll-full.lua` |
| retrieval / ANN | `ann.lua` |

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
