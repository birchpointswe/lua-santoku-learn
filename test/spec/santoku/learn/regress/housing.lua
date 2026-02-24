local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local optimize = require("santoku.learn.optimize")
local fs = require("santoku.fs")
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.learn")
local utc = require("santoku.utc")
local util = require("santoku.learn.util")

local cfg = {
  data = {
    ttr = 0.8,
    tvr = 0.10,
    max = nil,
    n_thresholds = 64,
  },
  tm = {
    outputs = 1,
    clauses = { def = 1, min = 1, max = 8, int = true },
    clause_maximum_fraction = { def = 0.048 },
    clause_tolerance_fraction = { def = 0.92 },
    target_fraction = { def = 0.063 },
    specificity_fraction = { def = 0.0011 },
  },
  search = {
    trials = 200,
    iterations = 40,
    subsample_samples = nil,
  },
  training = {
    patience = 4,
    batch = 40,
    iterations = 800,
  },
}

test("regressor", function ()

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

    clauses = cfg.tm.clauses,
    clause_maximum_fraction = cfg.tm.clause_maximum_fraction,
    clause_tolerance_fraction = cfg.tm.clause_tolerance_fraction,
    target_fraction = cfg.tm.target_fraction,
    specificity_fraction = cfg.tm.specificity_fraction,

    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    search_subsample_samples = cfg.search.subsample_samples,
    final_batch = cfg.training.batch,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (regressor)
      local score, nmae = regressor:regress_nmae(validate.problems, validate.n, validate.targets)
      return score, { nmae = nmae }
    end,

    each = util.make_regressor_acc_log(stopwatch)

  })

  print()
  print("Persisting")
  fs.rm("regressor.bin", true)
  t:persist("regressor.bin", true)

  print("Testing restore")
  t = tm.load("regressor.bin", nil, true)

  local train_pred = t:regress(train.problems, train.n)
  local val_pred = t:regress(validate.problems, validate.n)
  local test_pred = t:regress(test_set.problems, test_set.n)

  local train_stats = eval.regression_accuracy(train_pred, train.targets)
  local val_stats = eval.regression_accuracy(val_pred, validate.targets)
  local test_stats = eval.regression_accuracy(test_pred, test_set.targets)

  str.printf("\nResults (Accuracy):\n")
  str.printf("  Train:    %.1f%%\n", (1 - train_stats.nmae) * 100)
  str.printf("  Validate: %.1f%%\n", (1 - val_stats.nmae) * 100)
  str.printf("  Test:     %.1f%%\n", (1 - test_stats.nmae) * 100)

end)
