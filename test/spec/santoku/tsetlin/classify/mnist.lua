local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local fs = require("santoku.fs")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = nil
local ITERATIONS = 100
local THREADS = nil

local CLASSES = 10
local NEGATIVE = 0.1
local CLAUSES = { def = 8, min = 8, max = 32, log = true, int = true }
local CLAUSE_TOLERANCE = { def = 8, min = 8, max = 64, int = true }
local CLAUSE_MAXIMUM = { def = 8, min = 8, max = 64, int = true }
local TARGET = { def = 4, min = 8, max = 64 }
local SPECIFICITY = { def = 1000, min = 100, max = 2000 }

local SEARCH_PATIENCE = 3
local SEARCH_ROUNDS = 10
local SEARCH_TRIALS = 4
local SEARCH_ITERATIONS = 10

local FEATURES = 784

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", FEATURES, MAX)
  print("Splitting")
  local train, test = ds.split_binary_mnist(dataset, TTR)

  str.printf("Transforming train\t%d\n", train.n)
  train.problems = train.problems:raw_bitmap(train.n, dataset.n_features * 2, true)
  train.solutions = train.solutions:raw("u32")

  str.printf("Transforming test\t%d\n", test.n)
  test.problems = test.problems:raw_bitmap(test.n, dataset.n_features * 2, true)
  test.solutions = test.solutions:raw("u32")

  print("Training\n")
  local stopwatch = utc.stopwatch()
  local t = tm.optimize_classifier({

    features = dataset.n_features,
    classes = CLASSES,
    negative = NEGATIVE,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    clauses = CLAUSES,
    clause_tolerance = CLAUSE_TOLERANCE,
    clause_maximum = CLAUSE_MAXIMUM,
    target = TARGET,
    specificity = SPECIFICITY,

    search_patience = SEARCH_PATIENCE,
    search_rounds = SEARCH_ROUNDS,
    search_trials = SEARCH_TRIALS,
    search_iterations = SEARCH_ITERATIONS,
    final_iterations = ITERATIONS,
    threads = THREADS,

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
  t = tm.load("model.bin", nil, true)
  local train_pred = t.predict(train.problems, train.n)
  local test_pred = t.predict(test.problems, test.n)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, CLASSES)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, CLASSES)
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_stats.f1, train_stats.f1)

end)
