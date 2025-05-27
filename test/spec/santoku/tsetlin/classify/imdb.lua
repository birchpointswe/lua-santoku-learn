local arr = require("santoku.array")
local eval = require("santoku.tsetlin.evaluator")
local fs = require("santoku.fs")
local it = require("santoku.iter")
local mtx = require("santoku.matrix.integer")
local num = require("santoku.num")
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
local FEATURES = 784

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
  skips = 1,
  negations = 4,
  align = tm.align
}

local function read_data (dir, max)
  local problems = {}
  local solutions = {}
  local pos = it.paste(1, it.take(max or math.huge, fs.files(dir .. "/pos")))
  local neg = it.paste(0, it.take(max or math.huge, fs.files(dir .. "/neg")))
  local samples = it.map(function (label, text)
    return label, fs.readfile(text)
  end, it.chain(pos, neg))
  for label, review in samples do
    solutions[#solutions + 1] = label
    problems[#problems + 1] = review
  end
  arr.shuffle(problems, solutions)
  return {
    problems = problems,
    solutions = solutions
  }
end

local function split_dataset (dataset, s, e)
  local ps = arr.copy({}, dataset.problems, 1, s, e)
  local ss = arr.copy({}, dataset.solutions, 1, s, e)
  ss = mtx.create({ ss })
  return ps, ss
end

test("tsetlin", function ()

  print("Reading data")
  local dataset = read_data("test/res/imdb.dev", MAX)

  print("Splitting & packing")
  local n_train = num.floor(#dataset.problems * TTR)
  local n_test = #dataset.problems - n_train
  FEATURES = num.round(FEATURES, tm.align)
  local train_problems_raw, train_solutions = split_dataset(dataset, 1, n_train)
  local test_problems_raw, test_solutions = split_dataset(dataset, n_train + 1, n_train + n_test)
  str.printf("Train %d  Test %d\n", n_train, n_test)

  print("Train", n_train)
  print("Test", n_test)

  print("Creating tokenizer")
  local tokenizer = tokenizer.create(TOKENIZER_CONFIG)
  tokenizer.train({ corpus = train_problems_raw })
  tokenizer.finalize()
  dataset.n_features = tokenizer.features()
  str.printf("Feat\t\t%d\t\t\n", dataset.n_features)

  print("Tokenizing train")
  local train_problems = tokenizer.tokenize(train_problems_raw)
  local top_v =
    TOP_ALGO == "chi2" and mtx.top_chi2(train_problems, train_solutions, n_train, dataset.n_features, 2, TOP_K) or
    TOP_ALGO == "mi" and mtx.top_mi(train_problems, train_solutions, n_train, dataset.n_features, 2, TOP_K) or nil
  local n_top_v = mtx.columns(top_v)
  print("After top k filter", n_top_v)

  -- Show top words
  local words = tokenizer.index()
  for id in it.take(32, mtx.each(top_v)) do
    print(id, words[id + 1])
  end

  print("Re-encoding train/test with top features")
  tokenizer.restrict(top_v)
  local train_problems = tokenizer.tokenize(train_problems_raw);
  local test_problems = tokenizer.tokenize(test_problems_raw);

  print("Prepping for classifier")
  mtx.flip_interleave(train_problems, n_train, n_top_v)
  train_problems = mtx.raw_bitmap(train_problems, n_train, n_top_v * 2)
  train_solutions = mtx.raw(train_solutions, nil, nil, "u32");
  mtx.flip_interleave(test_problems, n_test, n_top_v)
  test_problems = mtx.raw_bitmap(test_problems, n_test, n_top_v * 2)
  test_solutions = mtx.raw(test_solutions, nil, nil, "u32");

  print("Creating")
  local t = tm.classifier({
    features = FEATURES,
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
    samples = n_train,
    problems = train_problems,
    solutions = train_solutions,
    iterations = ITERATIONS,
    each = function (epoch)
      local train_pred = t.predict(train_problems, n_train)
      local test_pred = t.predict(test_problems, n_test)
      local duration = stopwatch()
      if epoch == ITERATIONS or epoch % EVALUATE_EVERY == 0 then
        local train_stats = eval.class_accuracy(train_pred, train_solutions, n_train, CLASSES)
        local test_stats = eval.class_accuracy(test_pred, test_solutions, n_test, CLASSES)
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
  local train_pred = t.predict(train_problems, n_train)
  local test_pred = t.predict(test_problems, n_test)
  local train_stats = eval.class_accuracy(train_pred, train_solutions, n_train, CLASSES)
  local test_stats = eval.class_accuracy(test_pred, test_solutions, n_test, CLASSES)
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_stats.f1, train_stats.f1)

end)
