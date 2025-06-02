local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local fs = require("santoku.fs")
local it = require("santoku.iter")
local ivec = require("santoku.ivec")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local tokenizer = require("santoku.tsetlin.tokenizer")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = nil
local THREADS = nil
local EVALUATE_EVERY = 1
local ITERATIONS = 20

local CLASSES = 2
local CLAUSES = 8192
local TARGET = 32
local SPECIFICITY = 29
local NEGATIVE = 0.5

local TOP_ALGO = "mi"
local TOP_K = 1024

local TOKENIZER_CONFIG = {
  max_df = 0.95,
  min_df = 0.01,
  max_len = 20,
  min_len = 1,
  max_run = 2,
  ngrams = 2,
  cgrams_min = 0,
  cgrams_max = 0,
  skips = 2,
  negations = 4,
  align = tm.align
}

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_imdb("test/res/imdb.dev", MAX)
  local train, test = ds.split_imdb(dataset, TTR)

  print("Train", train.n)
  print("Test", test.n)

  print("\nTraining tokenizer\n")
  local tokenizer = tokenizer.create(TOKENIZER_CONFIG)
  tokenizer.train({ corpus = train.problems })
  tokenizer.finalize()
  dataset.n_features = tokenizer.features()
  str.printf("Feat\t\t%d\t\t\n", dataset.n_features)

  print("Tokenizing train")
  train.problems0 = tokenizer.tokenize(train.problems)
  local top_v =
    TOP_ALGO == "chi2" and ivec.top_chi2(train.problems0, train.solutions, train.n, dataset.n_features, 2, TOP_K, THREADS) or -- luacheck: ignore
    TOP_ALGO == "mi" and ivec.top_mi(train.problems0, train.solutions, train.n, dataset.n_features, 2, TOP_K, THREADS) or (function () -- luacheck: ignore
      -- Fallback to all words
      local t = ivec.create(dataset.n_features)
      ivec.fill_indices(t)
      return t
    end)()
  local n_top_v = ivec.size(top_v)
  print("After top k filter", n_top_v)

  -- Show top words
  local words = tokenizer.index()
  for id in it.take(32, ivec.each(top_v)) do
    print(id, words[id + 1])
  end

  print("Re-encoding train/test with top features")
  tokenizer.restrict(top_v)
  train.problems = tokenizer.tokenize(train.problems);
  test.problems = tokenizer.tokenize(test.problems);

  print("Prepping for classifier")
  ivec.flip_interleave(train.problems, train.n, n_top_v)
  ivec.flip_interleave(test.problems, test.n, n_top_v)
  train.problems = ivec.raw_bitmap(train.problems, train.n, n_top_v * 2)
  test.problems = ivec.raw_bitmap(test.problems, test.n, n_top_v * 2)
  train.solutions = ivec.raw(train.solutions, "u32")
  test.solutions = ivec.raw(test.solutions, "u32")

  print("Creating")
  local t = tm.classifier({
    features = n_top_v,
    classes = CLASSES,
    clauses = CLAUSES,
    target = TARGET,
    negative = NEGATIVE,
    specificity = SPECIFICITY,
    threads = THREADS,
  })

  print("Training")
  local stopwatch = utc.stopwatch()
  t.train({
    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,
    iterations = ITERATIONS,
    each = function (epoch)
      local train_pred = t.predict(train.problems, train.n)
      local test_pred = t.predict(test.problems, test.n)
      local duration = stopwatch()
      if epoch == ITERATIONS or epoch % EVALUATE_EVERY == 0 then
        local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, CLASSES)
        local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, CLASSES)
        str.printf("Epoch %-4d  Time %4.2f  Test %4.2f  Train %4.2f\n",
          epoch, duration, test_stats.f1, train_stats.f1)
      else
        str.printf("Epoch %-4d  Time %4.2f\n",
          epoch, duration)
      end
    end
  })

  print()
  print("Persisting")
  fs.rm("model.bin", true)
  t.persist("model.bin", true)

  print("Testing restore")
  t = tm.load("model.bin", nil, true)
  local train_pred = t.predict(train.problems, train.n)
  local test_pred = t.predict(test.problems, test.n)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, CLASSES)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, CLASSES)
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_stats.f1, train_stats.f1)

end)
