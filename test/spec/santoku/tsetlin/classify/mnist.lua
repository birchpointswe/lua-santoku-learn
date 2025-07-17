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
local EVALUATE_EVERY = 1
local ITERATIONS = 100

local CLASSES = 10
local CLAUSES = 4096
local TARGET = 32
local SPECIFICITY = 10
local NEGATIVE = 0.1

local FEATURES = 784

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", FEATURES, MAX)
  local train, test = ds.split_binary_mnist(dataset, TTR)

  print("Transforming train")
  train.problems:flip_interleave(train.n, dataset.n_features)
  train.problems = train.problems:raw_bitmap(train.n, dataset.n_features * 2)
  train.solutions = train.solutions:raw("u32")

  print("Transforming test")
  test.problems:flip_interleave(test.n, dataset.n_features)
  test.problems = test.problems:raw_bitmap(test.n, dataset.n_features * 2)
  test.solutions = test.solutions:raw("u32")

  print("Creating")
  local stopwatch = utc.stopwatch()
  local t = tm.optimize_classifier({

    features = dataset.n_features,
    classes = CLASSES,
    negative = NEGATIVE,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    clauses = { def = 1024, min = 512, max = 4096, log = true },
    target = { def = 0.1, min = 0.05, max = 0.25 },
    specificity = { def = 10, min = 2, max = 20 },

    search_patience = 3,
    search_rounds = 4,
    search_trials = 10,
    search_iterations = 10,
    final_iterations = ITERATIONS,
    search_metric = function (t)
      local predicted = t.predict(train.problems, train.n)
      local accuracy = eval.class_accuracy(predicted, train.solutions, train.n, CLASSES)
      return accuracy.f1, accuracy
    end,

    each = function (t, is_final, train_accuracy, params, epoch, round, trial)
      local test_predicted = t.predict(test.problems, test.n)
      local test_accuracy = eval.class_accuracy(test_predicted, test.solutions, test.n, CLASSES)
      local d, dd = stopwatch()
      if is_final then
        str.printf("  Time %3.2f %3.2f  Finalizing  C=%d T=%.2f S=%.2f  F1=(%.2f,%.2f)  Epoch  %d\n",
          d, dd, params.clauses, params.target, params.specificity, train_accuracy.f1, test_accuracy.f1, epoch)
        print()
      else
        str.printf("  Time %3.2f %3.2f  Exploring  C=%d T=%.2f S=%.2f  R=%d T=%d  F1=(%.2f,%.2f)  Epoch  %d\n",
          d, dd, params.clauses, params.target, params.specificity, round, trial, train_accuracy.f1, test_accuracy.f1, epoch)
        print()
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
