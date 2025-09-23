local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local ivec = require("santoku.ivec")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = nil
local ITERATIONS = 400

local CLASSES = 10
local CLAUSES = 8
local CLAUSE_TOLERANCE = 8
local CLAUSE_MAXIMUM = 8
local SPECIFICITY = 1000
local NEGATIVE = nil
local TARGET = 4

local FEATURES = 784

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", FEATURES, MAX)
  print("Splitting")
  local train, test = ds.split_binary_mnist(dataset, TTR)
  train.problems = ivec.create()
  train.problems:bits_copy(dataset.problems, nil, train.ids, dataset.n_features)
  test.problems = ivec.create()
  test.problems:bits_copy(dataset.problems, nil, test.ids, dataset.n_features)

  str.printf("Transforming train\t%d\n", train.n)
  train.problems = train.problems:bits_to_cvec(train.n, dataset.n_features, true)

  str.printf("Transforming test\t%d\n", test.n)
  test.problems = test.problems:bits_to_cvec(test.n, dataset.n_features, true)

  print("Creating\n")
  local stopwatch = utc.stopwatch()
  local t = tm.classifier({
    features = dataset.n_features,
    classes = CLASSES,
    clauses = CLAUSES,
    clause_tolerance = CLAUSE_TOLERANCE,
    clause_maximum = CLAUSE_MAXIMUM,
    target = TARGET,
    negative = NEGATIVE,
    specificity = SPECIFICITY,
  })

  print("Training\n")
  t.train({
    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,
    iterations = ITERATIONS,
    each = function (epoch)
      local train_predicted = t.predict(train.problems, train.n)
      local test_predicted = t.predict(test.problems, test.n)
      local train_accuracy = eval.class_accuracy(train_predicted, train.solutions, train.n, CLASSES)
      local test_accuracy = eval.class_accuracy(test_predicted, test.solutions, test.n, CLASSES)
      local d, dd = stopwatch()
      str.printf("  Time %3.2f %3.2f  F1=(%.2f,%.2f)  Epoch  %d\n",
        d, dd, train_accuracy.f1, test_accuracy.f1, epoch)
      print()
    end
  })

end)
