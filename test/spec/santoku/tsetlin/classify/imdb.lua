local fs = require("santoku.fs")
local it = require("santoku.iter")
local err = require("santoku.error")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local tokenizer = require("santoku.tsetlin.tokenizer")

local TTR = 0.9
local MAX = nil
local NEGATIVE = 0.5

local CLASSES = 2
local CLAUSES = { def = 8, min = 8, max = 32, int = true, log = true }
local CLAUSE_TOLERANCE = { def = 8, min = 8, max = 64, int = true, log = true }
local CLAUSE_MAXIMUM = { def = 8, min = 8, max = 64, int = true, log = true }
local TARGET = { def = 4, min = 8, max = 64, int = true, log = true }
local SPECIFICITY = { def = 1000, min = 100, max = 2000, int = true, log = true }

local SEARCH_PATIENCE = 3
local SEARCH_ROUNDS = 10
local SEARCH_TRIALS = 4
local SEARCH_ITERATIONS = 10
local FINAL_ITERATIONS = 20

local TOP_ALGO = "chi2"
local TOP_K = 8192

local TOKENIZER_CONFIG = {
  max_len = 20,
  min_len = 1,
  max_run = 2,
  ngrams = 2,
  cgrams_min = 3,
  cgrams_max = 4,
  skips = 2,
  negations = 4,
}

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_imdb("test/res/imdb.50k", MAX)
  local train, test = ds.split_imdb(dataset, TTR)

  print("Train", train.n)
  print("Test", test.n)

  print("\nTraining tokenizer\n")
  local tokenizer = tokenizer.create(TOKENIZER_CONFIG)
  tokenizer:train({ corpus = train.problems })
  tokenizer:finalize()
  dataset.n_features = tokenizer:features()
  str.printf("Feat\t\t%d\t\t\n", dataset.n_features)

  print("Tokenizing train")
  train.problems0 = tokenizer:tokenize(train.problems)
  train.solutions:add_scaled(CLASSES)
  local top_v =
    TOP_ALGO == "chi2" and train.problems0:bits_top_chi2(train.solutions, train.n, dataset.n_features, CLASSES, TOP_K) or -- luacheck: ignore
    TOP_ALGO == "mi" and train.problems0:bits_top_mi(train.solutions, train.n, dataset.n_features, CLASSES, TOP_K) or
    err.error("TOP_ALGO must bo chi2 or mi")
  train.solutions:add_scaled(-CLASSES)
  local n_top_v = top_v:size()
  print("After top k filter", n_top_v)

  -- Show top words
  local words = tokenizer:index()
  for id in it.take(32, top_v:each()) do
    print(id, words[id + 1])
  end

  print("Re-encoding train/test with top features")
  tokenizer:restrict(top_v)
  train.problems = tokenizer:tokenize(train.problems);
  test.problems = tokenizer:tokenize(test.problems);

  print("Prepping for classifier")
  train.problems = train.problems:bits_to_cvec(train.n, n_top_v, true)
  test.problems = test.problems:bits_to_cvec(test.n, n_top_v, true)

  print("Optimizing Classifier")
  local stopwatch = utc.stopwatch()
  local t = tm.optimize_classifier({

    features = n_top_v,

    classes = CLASSES,
    clauses = CLAUSES,
    clause_tolerance = CLAUSE_TOLERANCE,
    clause_maximum = CLAUSE_MAXIMUM,
    target = TARGET,
    negative = NEGATIVE,
    specificity = SPECIFICITY,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    search_patience = SEARCH_PATIENCE,
    search_rounds = SEARCH_ROUNDS,
    search_trials = SEARCH_TRIALS,
    search_iterations = SEARCH_ITERATIONS,
    final_iterations = FINAL_ITERATIONS,

    search_metric = function (t)
      local predicted = t.predict(train.problems, train.n)
      local accuracy = eval.class_accuracy(predicted, train.solutions, train.n, CLASSES)
      return accuracy.f1, accuracy
    end,

    each = function (t, is_final, train_accuracy, params, epoch, round, trial)
      local test_predicted = t.predict(test.problems, test.n)
      local test_accuracy = eval.class_accuracy(test_predicted, test.solutions, test.n, CLASSES)
      local d, dd = stopwatch()
      -- luacheck: push ignore
      if is_final then
        str.printf("  Time %3.2f %3.2f  Finalizing  C=%d LF=%d L=%d T=%.2f S=%.2f  F1=(%.2f,%.2f)  Epoch  %d\n\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, train_accuracy.f1, test_accuracy.f1, epoch)
      else
        str.printf("  Time %3.2f %3.2f  Exploring  C=%d LF=%d L=%d T=%.2f S=%.2f  R=%d T=%d  F1=(%.2f,%.2f)  Epoch  %d\n\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, round, trial, train_accuracy.f1, test_accuracy.f1, epoch)
      end
      -- luacheck: pop
    end

  })

  print()
  print("Persisting")
  fs.rm("model.bin", true)
  t.persist("model.bin", true)

  print("Testing restore")
  t = tm.load("model.bin")
  local train_pred = t.predict(train.problems, train.n)
  local test_pred = t.predict(test.problems, test.n)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, CLASSES)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, CLASSES)
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_stats.f1, train_stats.f1)

end)
