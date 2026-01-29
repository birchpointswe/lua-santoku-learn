local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local optimize = require("santoku.tsetlin.optimize")
local fs = require("santoku.fs")
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")
local util = require("santoku.tsetlin.util")

local cfg = {
  data = {
    ttr = 0.8,
    tvr = 0.10,
    max = nil,
    n_thresholds = 32,
  },
  tm = {
    outputs = 1,
    clauses = 64,
    clause_tolerance = { def = 73, min = 8, max = 128, int = true },
    clause_maximum = { def = 79, min = 16, max = 128, int = true },
    target = { def = 37, min = 4, max = 64, int = true },
    specificity = { def = 925, min = 50, max = 2000 },
    include_bits = { def = 5, min = 1, max = 6, int = true },
  },
  search = {
    patience = 4,
    rounds = 6,
    trials = 20,
    iterations = 10,
    metric = "nmae",
  },
  training = {
    patience = 40,
    iterations = 400,
  },
}

test("tsetlin regressor", function ()

  print("Reading data")
  local dataset = ds.read_california_housing("test/res/california-housing.csv", {
    n_thresholds = cfg.data.n_thresholds,
    max = cfg.data.max,
  })
  str.printf("  Loaded %d samples with %d features\n", dataset.n, dataset.n_features)

  print("Splitting")
  local train, test_set, validate = ds.split_california_housing(dataset, cfg.data.ttr, cfg.data.tvr)
  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test_set.n)

  local target_max = train.targets:max()
  local target_min = train.targets:min()
  str.printf("  Target range: %.0f - %.0f\n", target_min, target_max)

  local n_features = dataset.n_features
  train.problems = train.bits:bits_to_cvec(train.n, n_features, true)
  validate.problems = validate.bits:bits_to_cvec(validate.n, n_features, true)
  test_set.problems = test_set.bits:bits_to_cvec(test_set.n, n_features, true)

  print("\nTraining")
  local stopwatch = utc.stopwatch()
  local t = optimize.regressor({

    features = dataset.n_features,
    outputs = cfg.tm.outputs,

    samples = train.n,
    problems = train.problems,
    targets = train.targets,

    balanced = false,

    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    specificity = cfg.tm.specificity,
    include_bits = cfg.tm.include_bits,

    search_patience = cfg.search.patience,
    search_rounds = cfg.search.rounds,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (regressor)
      local predicted = regressor:predict(validate.problems, validate.n)
      local stats = eval.regression_accuracy(predicted, validate.targets)
      return -stats[cfg.search.metric], stats
    end,

    each = util.make_regressor_acc_log(stopwatch)

  })

  print()
  print("Persisting")
  fs.rm("regressor.bin", true)
  t:persist("regressor.bin", true)

  print("Testing restore")
  t = tm.load("regressor.bin", nil, true)

  local train_pred = t:predict(train.problems, train.n)
  local val_pred = t:predict(validate.problems, validate.n)
  local test_pred = t:predict(test_set.problems, test_set.n)

  local train_stats = eval.regression_accuracy(train_pred, train.targets)
  local val_stats = eval.regression_accuracy(val_pred, validate.targets)
  local test_stats = eval.regression_accuracy(test_pred, test_set.targets)

  str.printf("\nResults (Accuracy):\n")
  str.printf("  Train:    %.1f%%\n", (1 - train_stats.nmae) * 100)
  str.printf("  Validate: %.1f%%\n", (1 - val_stats.nmae) * 100)
  str.printf("  Test:     %.1f%%\n", (1 - test_stats.nmae) * 100)

end)
