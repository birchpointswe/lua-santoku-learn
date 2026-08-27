<p align="center">
  <img src="https://santoku.dev/logo-santoku-learn.png" height="64" alt="santoku-learn">
</p>

# santoku-learn

A kernel ridge regression toolkit for text. Turn documents into sparse ngram features,
project them into a spectral (Nystrom) embedding, fit ridge regression over it, and make
calibrated decisions for binary, multiclass, extreme multi-label, regression, span NER,
and approximate nearest neighbour retrieval.

## Install

```sh
luarocks install santoku-learn
```

## Example

```lua
local tokenizer = require("santoku.learn.tokenizer")
local optimize = require("santoku.learn.optimize")

local tok = tokenizer.create({ ngram_min = 5, ngram_max = 5 })

local X = tok:fit({ texts = train_texts })
local Xv = tok:tokenize({ texts = val_texts })

local enc, ridge, _, _, decider = optimize.krr({
  x = X, y = train_labels,
  val_x = Xv, val_y = val_labels,
  kernel = { "matern", "cosine" }, n_landmarks = 8192,
  lambda = { def = 1e-2 }, k = 1, search_trials = 12,
})

local codes = enc:encode(tok:tokenize({ texts = test_texts }))
local P = ridge:label(codes, 1)
local _, metrics = decider:score({ pred = P, expected = test_labels })
```

Features in and out are `csr` sparse matrices, embeddings are dense `mtx`, and predictions
come back as a `csr` of top-k label and score pairs, all from
[santoku-matrix](https://santoku.dev/#santoku-matrix).

## Documentation

Runnable examples and the full API: [santoku.dev](https://santoku.dev/#santoku-learn).

For agents and LLM tooling: [llms.txt](https://santoku.dev/llms.txt) for the index,
[llms-full.txt](https://santoku.dev/llms-full.txt) for every documented example.

## Tests

The tests are the spec. For the exhaustive surface, read them under
[`test/spec/santoku/learn`](test/spec/santoku/learn):
[`tokenizer.lua`](test/spec/santoku/learn/tokenizer.lua),
[`extract.lua`](test/spec/santoku/learn/extract.lua),
[`aho.lua`](test/spec/santoku/learn/aho.lua),
[`ann.lua`](test/spec/santoku/learn/ann.lua),
[`ard.lua`](test/spec/santoku/learn/ard.lua),
[`ner.lua`](test/spec/santoku/learn/ner.lua),
[`decide.lua`](test/spec/santoku/learn/decide.lua), and the end-to-end regressions in
[`test/spec/santoku/learn/regress`](test/spec/santoku/learn/regress).

## License

MIT, see [LICENSE](LICENSE).

## More examples

```lua
local test = require("santoku.test")

local err = require("santoku.error")
local assert = err.assert

local validate = require("santoku.validate")
local eq = validate.isequal

local tbl = require("santoku.table")
local teq = tbl.equals

local re = require("santoku.re")
local aho = require("santoku.learn.aho")
local tokenizer = require("santoku.learn.tokenizer")

test("split a corpus into byte spans in one parallel pass", function ()
  local off, s, e = tokenizer.extract({
    n = 3,
    texts = { "U.N. official", "", "ab cd12" },
    pattern = re.prog("[A-Za-z0-9]+")
  })
  assert(teq({ 0, 3, 3, 5 }, off:table()))
  assert(teq({ 0, 2, 5, 0, 3 }, s:table()))
  assert(teq({ 1, 3, 13, 2, 7 }, e:table()))
end)

test("named groups tag every token as it is matched", function ()
  local pattern = "{:caps: [A-Z]+ :} ![A-Za-z0-9]"
    .. " / {:num: [0-9]+ :} ![A-Za-z0-9]"
    .. " / [A-Za-z0-9]+ / [^A-Za-z0-9 ]+"
  local tags = re.tags(pattern)
  local _, _, _, ty = tokenizer.extract({
    n = 1, texts = { "UN 1996 x" }, pattern = re.prog(pattern) })
  assert(teq({ tags.caps, tags.num, 2 }, ty:table()))
end)

test("match a whole gazetteer against a corpus at once", function ()
  local ac = aho.create({ patterns = { "foo", "bar", "baz" } })
  local S = ac:predict({ texts = { "foo bar baz" } })
  assert(eq(3, S:col("id"):size()))
  assert(eq(1, S:col("id"):get(1)))
  assert(eq(4, S:col("s"):get(1)))
  assert(eq(7, S:col("e"):get(1)))
end)

test("tokenize into sparse ngram features, then freeze the vocabulary", function ()
  local tk = tokenizer.create({ ngram_min = 3, ngram_max = 3 })
  local X = tk:fit({ texts = { "hello", "world" } })
  assert(eq(2, (X:shape())))
  local n = tk:n_tokens()
  assert(eq(1, (tk:tokenize({ texts = { "zzzzzzzz" } }):shape())))
  assert(eq(n, tk:n_tokens()))
end)
```
