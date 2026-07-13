require("santoku.error")
local tokenizer = require("santoku.learn.tokenizer")
local re = require("santoku.re")
local ivec = require("santoku.ivec")
require("santoku.fvec")  -- installs fvec metatable for tokenize_raw's returned values vec
local spans = require("santoku.spans")
local test = require("santoku.test")

local word_prog = re.prog("[A-Za-z0-9]+")

local texts = { "Hello, w\195\182rld 42!", "a.b c-d  x", "", "  lead trail  ", "one" }
local n = #texts

local function word_tokens ()
  local woff, ws, we = tokenizer.extract({ n = n, texts = texts, pattern = word_prog })
  return spans.create({ offsets = woff, s = ws, e = we })
end

test("tokenizer", function ()

  test("chars: grow then frozen", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 3 })
    local X = tk:fit({ texts = { "hello", "world" } })
    assert((X:shape()) == 2)              -- one row per doc
    local nt = tk:n_tokens()
    assert(nt > 0)
    -- frozen pass: unseen grams are dropped, vocab unchanged
    local X2 = tk:tokenize({ texts = { "zzzzzzzz" } })
    assert(tk:n_tokens() == nt)
    assert((X2:shape()) == 1)
  end)

  test("frozen before any grow errors", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 3 })
    assert(not pcall(function () tk:tokenize({ texts = { "x" } }) end))
  end)

  test("focus brackets: one row per focus span", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 5, terminals = true, focus = true })
    local F = spans.create({ offsets = ivec.create({ 0, 2 }), s = ivec.create({ 0, 4 }), e = ivec.create({ 3, 7 }) })
    local X = tk:fit({ texts = { "abc def" }, focus = F })
    assert((X:shape()) == 2)              -- 2 focus spans -> 2 rows
    assert(tk:n_tokens() > 0)
  end)

  test("mode=words/flat: extract-fed, grow==frozen", function ()
    local T = word_tokens()
    local F = spans.create({ offsets = ivec.create({ 0, 1, 3, 3, 4, 5 }),
      s = ivec.create({ 2, 1, 4, 3, 0 }), e = ivec.create({ 9, 3, 9, 8, 3 }) })
    local w = tokenizer.create({ ngram_min = 1, ngram_max = 3, mode = "words",
      terminals = true, focus = true })
    local X = w:fit({ texts = texts, focus = F, tokens = T })
    assert((X:shape()) == 5)              -- one row per focus span
    assert(w:n_tokens() > 0)
    assert(X:eq(w:tokenize({ texts = texts, focus = F, tokens = T })))
    local f = tokenizer.create({ ngram_min = 1, ngram_max = 4, mode = "flat",
      terminals = true })
    local Y = f:fit({ texts = texts, tokens = T })
    assert((Y:shape()) == 5)              -- one row per doc
    assert(f:n_tokens() > 0)
    assert(Y:eq(f:tokenize({ texts = texts, tokens = T })))
  end)

  test("mode=tags: tokens carry the ty column", function ()
    local F = spans.create({ offsets = ivec.create({ 0, 1, 3, 3, 4, 5 }),
      s = ivec.create({ 2, 1, 4, 3, 0 }), e = ivec.create({ 9, 3, 9, 8, 3 }) })
    local C = spans.create({ offsets = ivec.create({ 0, 2, 4, 4, 5, 6 }),
      s = ivec.create({ 0, 7, 0, 4, 2, 0 }), e = ivec.create({ 5, 12, 3, 7, 6, 3 }),
      ty = ivec.create({ 0, 2, 1, 0, 2, 1 }) })
    local tk = tokenizer.create({ ngram_min = 1, ngram_max = 3, mode = "tags",
      n_tags = 3, terminals = true, focus = true })
    local Z = tk:fit({ texts = texts, focus = F, tokens = C })
    assert((Z:shape()) == 5)
    assert(tk:n_tokens() > 0)
    assert(Z:eq(tk:tokenize({ texts = texts, focus = F, tokens = C })))
  end)

  test("validity errors", function ()
    -- mode=tags requires n_tags; normalize is chars-only
    assert(not pcall(function () tokenizer.create({ ngram_max = 3, mode = "tags" }) end))
    assert(not pcall(function () tokenizer.create({ ngram_max = 3, mode = "flat", normalize = true }) end))
    assert(not pcall(function () tokenizer.create({ ngram_max = 3, mode = "tags", n_tags = 2, normalize = true }) end))
    -- flat/words/tags tokenize without tokens spans
    local f = tokenizer.create({ ngram_max = 3, mode = "flat" })
    assert(not pcall(function () f:fit({ texts = { "x y" } }) end))
  end)

  test("persist/load round-trips to an identical CSR", function ()
    local T = word_tokens()
    local tk = tokenizer.create({ ngram_min = 1, ngram_max = 3, mode = "words",
      terminals = true, focus = true })
    local F = spans.create({ offsets = ivec.create({ 0, 1, 1, 1, 1, 1 }), s = ivec.create({ 0 }), e = ivec.create({ 8 }) })
    tk:fit({ texts = texts, focus = F, tokens = T })
    local path = os.tmpname()
    tk:persist(path)
    local tk2 = tokenizer.load(path)
    os.remove(path)
    assert(tk2:n_tokens() == tk:n_tokens())
    local X1 = tk:tokenize({ texts = texts, focus = F, tokens = T })
    local X2 = tk2:tokenize({ texts = texts, focus = F, tokens = T })
    assert(X1:eq(X2))
  end)

  -- tokenize_raw: stateless, fit-free byte char-ngrams -> (offsets, raw 64-bit hash keys, counts).
  -- No vocab/fit, no modes/focus/regions/terminals/persist -- just the raw hashing bag (littlelist).
  test("tokenize_raw: raw ngram-hash csr, counts", function ()
    local off, tok, val = tokenizer.tokenize_raw({
      texts = { "hello" }, n_samples = 1, ngram_min = 3, ngram_max = 3 })
    assert(off:size() == 2 and off:get(0) == 0 and off:get(1) == 3)   -- 5-3+1 = 3 trigrams
    assert(tok:size() == 3 and val:size() == 3)
    assert(val:sum() == 3)                                            -- hel/ell/llo distinct -> each 1
  end)

  test("tokenize_raw: dedups repeated ngrams into counts", function ()
    local _, tok, val = tokenizer.tokenize_raw({
      texts = { "ababab" }, n_samples = 1, ngram_min = 2, ngram_max = 2 })
    assert(tok:size() == 2)                    -- "ab", "ba"
    assert(val:sum() == 5)                     -- 6-2+1 = 5 bigrams
    assert(val:max() == 3 and val:min() == 2)  -- ab x3, ba x2
  end)

  test("tokenize_raw: shared ngram -> shared column id across docs", function ()
    local off, tok, val = tokenizer.tokenize_raw({
      texts = { "abc", "abc" }, n_samples = 2, ngram_min = 3, ngram_max = 3 })
    assert(off:size() == 3 and off:get(1) == 1 and off:get(2) == 2)
    assert(tok:get(0) == tok:get(1))           -- same hash for the shared trigram
    assert(val:get(0) == 1 and val:get(1) == 1)
  end)

  test("tokenize_raw: ngram range unions sizes", function ()
    local _, tok, val = tokenizer.tokenize_raw({
      texts = { "abc" }, n_samples = 1, ngram_min = 1, ngram_max = 2 })
    assert(tok:size() == 5)                     -- a,b,c + ab,bc, all distinct
    assert(val:sum() == 5)
  end)

  test("tokenize_raw: normalize collapses whitespace", function ()
    local _, a = tokenizer.tokenize_raw({
      texts = { "a  b" }, n_samples = 1, ngram_min = 1, ngram_max = 3, normalize = true })
    local _, b = tokenizer.tokenize_raw({
      texts = { "a b" }, n_samples = 1, ngram_min = 1, ngram_max = 3, normalize = true })
    assert(a:eq(b))                             -- "a  b" -> "a b" under normalize
    local _, c = tokenizer.tokenize_raw({
      texts = { "a  b" }, n_samples = 1, ngram_min = 1, ngram_max = 3, normalize = false })
    assert(not a:eq(c))                         -- raw keeps the double space -> different ngrams
  end)

  test("tokenize_raw: bad ngram range errors", function ()
    assert(not pcall(function ()
      tokenizer.tokenize_raw({ texts = { "x" }, n_samples = 1, ngram_min = 3, ngram_max = 2 })
    end))
  end)

end)
