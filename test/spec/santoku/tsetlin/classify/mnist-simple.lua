local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local ivec = require("santoku.ivec")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local cfg = {
  data = {
    ttr = 0.9,
    max = nil,
    features = 784,
  },
  tm = {
    classes = 10,
    clauses = 8,
    clause_tolerance = 8,
    clause_maximum = 8,
    target = 4,
    negative = nil,
    specificity = 1000,
  },
  training = {
    iterations = 5,
  },
  threads = nil,
}

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.features, cfg.data.max)
  print("Splitting")
  local train, test = ds.split_binary_mnist(dataset, cfg.data.ttr)
  train.problems = ivec.create()
  dataset.problems:bits_select(nil, train.ids, dataset.n_features, train.problems)
  test.problems = ivec.create()
  dataset.problems:bits_select(nil, test.ids, dataset.n_features, test.problems)

  str.printf("Transforming train\t%d\n", train.n)
  train.problems = train.problems:bits_to_cvec(train.n, dataset.n_features, true)

  str.printf("Transforming test\t%d\n", test.n)
  test.problems = test.problems:bits_to_cvec(test.n, dataset.n_features, true)

  print("Creating\n")
  local stopwatch = utc.stopwatch()
  local t = tm.classifier({
    features = dataset.n_features,
    classes = cfg.tm.classes,
    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    negative = cfg.tm.negative,
    specificity = cfg.tm.specificity,
  })

  print("Training\n")
  t:train({
    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,
    iterations = cfg.training.iterations,
    each = function (epoch)
      local train_predicted = t:predict(train.problems, train.n, cfg.threads)
      local test_predicted = t:predict(test.problems, test.n, cfg.threads)
      local train_accuracy = eval.class_accuracy(train_predicted, train.solutions, train.n, cfg.tm.classes, cfg.threads)
      local test_accuracy = eval.class_accuracy(test_predicted, test.solutions, test.n, cfg.tm.classes, cfg.threads)
      local d, dd = stopwatch()
      str.printf("  Time %3.2f %3.2f  F1=(%.2f,%.2f)  Epoch  %d\n",
        d, dd, train_accuracy.f1, test_accuracy.f1, epoch)
      print()
    end
  })

end)
