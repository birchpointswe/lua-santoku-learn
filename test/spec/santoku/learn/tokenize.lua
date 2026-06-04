require("santoku.error")
local csr = require("santoku.learn.csr")
local ivec = require("santoku.ivec")
local cvec = require("santoku.cvec")
local test = require("santoku.test")

test("tokenize", function ()

  test("basic char ngrams", function ()
    local _, off, tok, val, n = csr.tokenize({
      texts = { "hello", "world" }, ngram_min = 3, ngram_max = 3, n_samples = 2,
    })
    assert(off:size() == 3)
    assert(off:get(0) == 0)
    assert(tok:size() == off:get(2))
    assert(val:size() == tok:size())
    assert(n > 0)
  end)

  test("ngram range", function ()
    local _, off, tok = csr.tokenize({
      texts = { "hello" }, ngram_min = 1, ngram_max = 3, n_samples = 1,
    })
    assert(off:get(1) > 0)
    assert(tok:size() == off:get(1))
  end)

  test("ngram_map reuse: identical ids, unknown dropped", function ()
    local args = { texts = { "hello world" }, ngram_min = 3, ngram_max = 3, n_samples = 1 }
    local map, o1, t1, _, n1 = csr.tokenize(args)
    args.ngram_map = map
    local _, o2, t2, _, n2 = csr.tokenize(args)
    assert(n1 == n2)
    assert(o1:get(1) == o2:get(1))
    for i = 0, t1:size() - 1 do assert(t1:get(i) == t2:get(i)) end
    local _, o3 = csr.tokenize({
      texts = { "zzz qqq" }, ngram_min = 3, ngram_max = 3, ngram_map = map, n_samples = 1,
    })
    assert(o3:get(1) == 0)
  end)

  test("normalize folds case", function ()
    local map = csr.tokenize({
      texts = { "hello world" }, ngram_min = 3, ngram_max = 3, normalize = true, n_samples = 1,
    })
    local _, o_up = csr.tokenize({
      texts = { "HELLO WORLD" }, ngram_min = 3, ngram_max = 3, normalize = true,
      ngram_map = map, n_samples = 1,
    })
    local _, o_lo = csr.tokenize({
      texts = { "hello world" }, ngram_min = 3, ngram_max = 3, normalize = true,
      ngram_map = map, n_samples = 1,
    })
    assert(o_up:get(1) > 0)
    assert(o_up:get(1) == o_lo:get(1))
  end)

  test("terminals add features", function ()
    local _, o_no = csr.tokenize({ texts = { "hello" }, ngram_min = 3, ngram_max = 3, n_samples = 1 })
    local _, o_t = csr.tokenize({
      texts = { "hello" }, ngram_min = 3, ngram_max = 3, terminals = true, n_samples = 1,
    })
    assert(o_t:get(1) > o_no:get(1))
  end)

  test("ngrams beyond 8 hash (no cap)", function ()
    local _, off, tok, _, n = csr.tokenize({
      texts = { "abcdefghijklmnop" }, ngram_min = 10, ngram_max = 12, n_samples = 1,
    })
    assert(off:get(1) > 0)
    assert(tok:size() == off:get(1))
    assert(n > 0)
  end)

  test("ngram>8 reuse consistency", function ()
    local args = { texts = { "abcdefghijklmnop" }, ngram_min = 12, ngram_max = 12, n_samples = 1 }
    local map, o1, t1 = csr.tokenize(args)
    args.ngram_map = map
    local _, o2, t2 = csr.tokenize(args)
    assert(o1:get(1) == o2:get(1))
    for i = 0, t1:size() - 1 do assert(t1:get(i) == t2:get(i)) end
  end)

  test("sequences cvec (compact bytes)", function ()
    local seq = cvec.from_raw("hello world")
    local off = ivec.create({ 0, 11 })
    local _, o, t, _, n = csr.tokenize({
      sequences = seq, sequence_offsets = off, ngram_min = 3, ngram_max = 3, n_samples = 1,
    })
    assert(o:get(1) > 0)
    assert(t:size() == o:get(1))
    assert(n > 0)
  end)

end)
