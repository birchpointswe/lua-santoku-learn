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
