require("santoku.error")
local tokenizer = require("santoku.learn.tokenizer")
local ivec = require("santoku.ivec")
local spans = require("santoku.spans")
local test = require("santoku.test")

local function iv (...)
  local v = ivec.create()
  for _, x in ipairs({ ... }) do v:push(x) end
  return v
end

local function iveq (a, b)
  if a:size() ~= b:size() then return false end
  for i = 0, a:size() - 1 do
    if a:get(i) ~= b:get(i) then return false end
  end
  return true
end

test("tokenizer", function ()

  test("plain text: grow then frozen", function ()
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
    local F = spans.create({ offsets = iv(0, 2), s = iv(0, 4), e = iv(3, 7) })
    local X = tk:fit({ texts = { "abc def" }, focus = F })
    assert((X:shape()) == 2)              -- 2 focus spans -> 2 rows
    assert(tk:n_tokens() > 0)
  end)

  test("types=true renders a per-token type skeleton", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 5, n_types = 4, terminals = true, focus = true, types = true })
    local F = spans.create({ offsets = iv(0, 1), s = iv(4), e = iv(9) })
    local T = spans.create({ offsets = iv(0, 4), s = iv(0, 4, 10, 16), e = iv(3, 9, 15, 19), ty = iv(4, 0, 1, 4) })
    local X = tk:fit({ texts = { "the quick brown fox" }, focus = F, types = T })   -- O PER ORG O
    assert((X:shape()) == 1)
    assert(tk:n_tokens() > 0)
  end)

  test("validity errors", function ()
    assert(not pcall(function () tokenizer.create({ ngram_min = 3, ngram_max = 5, normalize = true, types = true, n_types = 2 }) end))
    assert(not pcall(function () tokenizer.create({ ngram_min = 3, ngram_max = 5, types = true }) end))   -- n_types required
  end)

  test("persist/load round-trips to an identical CSR", function ()
    local tk = tokenizer.create({ ngram_min = 3, ngram_max = 4, terminals = true, focus = true })
    local F = spans.create({ offsets = iv(0, 1), s = iv(0), e = iv(5) })
    tk:fit({ texts = { "hello world" }, focus = F })
    local path = os.tmpname()
    tk:persist(path)
    local tk2 = tokenizer.load(path)
    os.remove(path)
    assert(tk2:n_tokens() == tk:n_tokens())
    local X1 = tk:tokenize({ texts = { "hello world" }, focus = F })
    local X2 = tk2:tokenize({ texts = { "hello world" }, focus = F })
    assert(X1:eq(X2))
  end)

  test("words: raw byte spans, both modes", function ()
    local W = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    local off, st, en = tokenizer.words({ n = 2,
      texts = { "U.N. official", "" }, word_characters = W })
    assert(iveq(off, iv(0, 3, 3)))
    assert(iveq(st, iv(0, 2, 5)))
    assert(iveq(en, iv(1, 3, 13)))
    local off2, st2, en2 = tokenizer.words({ n = 1,
      texts = { "U.N. official" }, word_characters = W, punctuation = true })
    assert(iveq(off2, iv(0, 5)))
    assert(iveq(st2, iv(0, 1, 2, 3, 5)))
    assert(iveq(en2, iv(1, 2, 3, 4, 13)))
    assert(not pcall(function () tokenizer.words({ n = 1, texts = { "x" } }) end))
  end)

end)
