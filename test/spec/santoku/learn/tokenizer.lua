require("santoku.error")
local tokenizer = require("santoku.learn.tokenizer")
local ivec = require("santoku.ivec")
local test = require("santoku.test")

local function iv (...)
  local v = ivec.create()
  for _, x in ipairs({ ... }) do v:push(x) end
  return v
end

test("tokenizer", function ()

  test("plain text: grow then frozen", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 3 })
    local off = tk:tokenize({ texts = { "hello", "world" }, n_samples = 2, grow = true })
    assert(off:size() == 3)               -- one row per doc + 1
    local nt = tk:n_tokens()
    assert(nt > 0)
    -- frozen pass: unseen grams are dropped, vocab unchanged
    local o2 = tk:tokenize({ texts = { "zzzzzzzz" }, n_samples = 1 })
    assert(tk:n_tokens() == nt)
    assert(o2:size() == 2)
  end)

  test("frozen before any grow errors", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 3 })
    assert(not pcall(function () tk:tokenize({ texts = { "x" }, n_samples = 1 }) end))
  end)

  test("focus brackets: one row per focus span", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 5, terminals = true, focus = true })
    local off = tk:tokenize({
      texts = { "abc def" }, n_samples = 1,
      focus = { offsets = iv(0, 2), starts = iv(0, 4), ends = iv(3, 7) }, grow = true,
    })
    assert(off:size() == 3)               -- 2 focus spans -> 2 rows + 1
    assert(tk:n_tokens() > 0)
  end)

  test("types=true renders a per-token type skeleton", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 5, n_types = 4, terminals = true, focus = true, types = true })
    local off = tk:tokenize({
      texts = { "the quick brown fox" }, n_samples = 1,
      focus = { offsets = iv(0, 1), starts = iv(4), ends = iv(9) },
      types = { offsets = iv(0, 4), starts = iv(0, 4, 10, 16), ends = iv(3, 9, 15, 19), types = iv(4, 0, 1, 4) },
      grow = true,    -- O PER ORG O
    })
    assert(off:size() == 2)
    assert(tk:n_tokens() > 0)
  end)

  test("shapes=true derives a per-token shape skeleton (no spans passed)", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 5, terminals = true, focus = true, shapes = true })
    local off = tk:tokenize({
      texts = { "Visit New York now" }, n_samples = 1,
      focus = { offsets = iv(0, 1), starts = iv(6), ends = iv(14) }, grow = true,
    })
    assert(off:size() == 2)
    assert(tk:n_tokens() > 0)
  end)

  test("validity errors", function ()
    assert(not pcall(function () tokenizer.create({ ngram_min = 3, ngram_max = 5, normalize = true, types = true, n_types = 2 }) end))
    assert(not pcall(function () tokenizer.create({ ngram_min = 3, ngram_max = 5, types = true, shapes = true, n_types = 2 }) end))   -- mutually exclusive
    assert(not pcall(function () tokenizer.create({ ngram_min = 3, ngram_max = 5, types = true }) end))   -- n_types required
  end)

  test("persist/load round-trips to an identical CSR", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 4, terminals = true, focus = true })
    local focus = { offsets = iv(0, 1), starts = iv(0), ends = iv(5) }
    tk:tokenize({ texts = { "hello world" }, n_samples = 1, focus = focus, grow = true })
    local path = os.tmpname()
    tk:persist(path)
    local tk2 = tokenizer.load(path)
    os.remove(path)
    assert(tk2:n_tokens() == tk:n_tokens())
    local o1, t1, v1 = tk:tokenize({ texts = { "hello world" }, n_samples = 1, focus = focus })
    local o2, t2, v2 = tk2:tokenize({ texts = { "hello world" }, n_samples = 1, focus = focus })
    assert(o1:size() == o2:size() and t1:size() == t2:size() and v1:size() == v2:size())
    for i = 0, t1:size() - 1 do assert(t1:get(i) == t2:get(i)) end
  end)

end)
