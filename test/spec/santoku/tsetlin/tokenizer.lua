-- luacheck: push ignore

require("santoku.ivec")
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
    ngrams = 0,
    cgrams_min = 3,
    cgrams_max = 3,
    skips = 0,
    negations = 0,
  })

  local corpus = {
    "this is a test the lib\'s tokenizer",
    "this is another test",
    "And one more a thing",
  }

  tok:train({ corpus = corpus })
  tok:finalize()
  local tokens = tok:tokenize(corpus)
  local top, weights = tokens:bits_top_df(#corpus, tok:features())
  tok:restrict(top)

  local words = tok:index()

  print(serialize(weights:table()))
  print(serialize(words))
  for id in tok:tokenize("this is a test"):each() do
    print(id, words[id + 1], weights:get(id))
  end

end)

