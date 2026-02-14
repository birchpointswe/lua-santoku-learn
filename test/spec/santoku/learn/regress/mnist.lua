local arr = require("santoku.array")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.learn.evaluator")
local fs = require("santoku.fs")
local ivec = require("santoku.ivec")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.learn")
local utc = require("santoku.utc")
local util = require("santoku.learn.util")

local cfg = {
  data = {
    ttr = 0.8,
    tvr = 0.1,
    max = nil,
    features = 784,
  },
  tm = {
    classes = 10,
    clauses = { def = 2, min = 1, max = 4, int = true },
    clause_maximum = { def = 8, min = 8, max = 1024, int = true },
    clause_tolerance_fraction = { def = 0.5, min = 0.01, max = 1.0 },
    target_fraction = { def = 0.25, min = 0.01, max = 2.0 },
    specificity = { def = 2, min = 2, max = 2000, int = true },
  },
  search = {
    trials = 120,
    iterations = 40,
    subsample_samples = 0.2,
  },
  training = {
    patience = 10,
    batch = 20,
    iterations = 800,
  },
}

test("mnist classifier", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.features, cfg.data.max)
  print("Splitting")
  local train, test_set, validate = ds.split_binary_mnist(dataset, cfg.data.ttr, cfg.data.tvr)
  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test_set.n)

  print("\nConverting to TM representation")
  train.tokens = ivec.create()
  dataset.problems:bits_select(nil, train.ids, cfg.data.features, train.tokens)
  train.problems = train.tokens:bits_to_cvec(train.n, cfg.data.features, true)

  validate.tokens = ivec.create()
  dataset.problems:bits_select(nil, validate.ids, cfg.data.features, validate.tokens)
  validate.problems = validate.tokens:bits_to_cvec(validate.n, cfg.data.features, true)

  test_set.tokens = ivec.create()
  dataset.problems:bits_select(nil, test_set.ids, cfg.data.features, test_set.tokens)
  test_set.problems = test_set.tokens:bits_to_cvec(test_set.n, cfg.data.features, true)

  local output_weights = dvec.create(cfg.tm.classes)
  for i = 0, train.n - 1 do
    local c = train.solutions:get(i)
    output_weights:set(c, output_weights:get(c) + 1)
  end

  print("\nTraining")
  local stopwatch = utc.stopwatch()
  local predicted_buf = ivec.create()
  local t = optimize.regressor({

    features = cfg.data.features,
    outputs = cfg.tm.classes,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    output_weights = output_weights,

    clauses = cfg.tm.clauses,
    clause_maximum = cfg.tm.clause_maximum,
    clause_tolerance_fraction = cfg.tm.clause_tolerance_fraction,
    target_fraction = cfg.tm.target_fraction,
    specificity = cfg.tm.specificity,

    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    search_subsample_samples = cfg.search.subsample_samples,
    final_batch = cfg.training.batch,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (regressor)
      local predicted = regressor:classify(validate.problems, validate.n, nil, predicted_buf)
      local stats = eval.class_accuracy(predicted, validate.solutions, validate.n, cfg.tm.classes)
      return stats.f1, stats
    end,

    each = util.make_classifier_log(stopwatch)

  })

  print()
  print("Persisting")
  fs.rm("regressor.bin", true)
  t:persist("regressor.bin", true)

  print("Testing restore")
  t = tm.load("regressor.bin", nil, true)

  print("\nClassification metrics:")
  local train_scores = t:classify(train.problems, train.n)
  local val_scores = t:classify(validate.problems, validate.n)
  local test_scores = t:classify(test_set.problems, test_set.n)

  local train_stats = eval.class_accuracy(train_scores, train.solutions, train.n, cfg.tm.classes)
  local val_stats = eval.class_accuracy(val_scores, validate.solutions, validate.n, cfg.tm.classes)
  local test_stats = eval.class_accuracy(test_scores, test_set.solutions, test_set.n, cfg.tm.classes)
  str.printf("  F1:   Train=%.2f  Val=%.2f  Test=%.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

  print("\nPer-class Test Accuracy (sorted by difficulty):\n")
  local class_order = arr.range(1, cfg.tm.classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    str.printf("  digit_%-2d  F1=%.2f  P=%.2f  R=%.2f\n", c - 1, ts.f1, ts.precision, ts.recall)
  end

end)
