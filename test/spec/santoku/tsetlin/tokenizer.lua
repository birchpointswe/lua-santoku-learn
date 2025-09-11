-- luacheck: push ignore

local tokenizer = require("santoku.tsetlin.tokenizer")
local test = require("santoku.test")
local serialize = require("santoku.serialize")

test("tokenizer", function ()

  local tok = tokenizer.create({
    max_df = 1.0,
    min_df = 0.0,
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 0,
    cgrams_max = 0,
    skips = 0,
    negations = 0,
  })

  tok.train({ corpus = { "this is a test the lib\'s tokenizer" } })
  tok.finalize()
  local words = tok.index()
  -- print(serialize(words))
  -- for id in tok.tokenize("this is a test"):each() do
  --   print(id, words[id + 1])
  -- end

end)

