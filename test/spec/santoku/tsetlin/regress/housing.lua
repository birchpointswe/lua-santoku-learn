local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local optimize = require("santoku.tsetlin.optimize")
local fs = require("santoku.fs")
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local cfg = {
  data = {
    ttr = 0.7,
    tvr = 0.15,
    max = nil,
    n_thresholds = 99999,
  },
  tm = {
    outputs = 1,
    clauses = { def = 160, min = 32, max = 512, round = 8 },
    clause_tolerance = { def = 40, min = 8, max = 128, int = true },
    clause_maximum = { def = 127, min = 16, max = 128, int = true },
    target = { def = 47, min = 4, max = 64, int = true },
    specificity = { def = 64, min = 50, max = 2000 },
    include_bits = { def = 1, min = 1, max = 6, int = true },
  },
  search = {
    patience = 5,
    rounds = 0,
    trials = 8,
    iterations = 30,
  },
  training = {
    patience = 30,
    iterations = 300,
  },
  threads = nil,
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

  print("Training\n")
  local stopwatch = utc.stopwatch()
  local t = optimize.regressor({

    features = dataset.n_features,
    outputs = cfg.tm.outputs,

    samples = train.n,
    problems = train.problems,
    targets = train.targets,

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
      local predicted = regressor:predict(validate.problems, validate.n, cfg.threads)
      local stats = eval.regression_accuracy(predicted, validate.targets)
      local score = -stats.mean
      return score, stats
    end,

    each = function (_, is_final, val_stats, params, epoch, round, trial)
      local d, dd = stopwatch()
      local phase = is_final and "F" or str.format("R%d T%d", round, trial)
      str.printf("[REGRESS %s E%d] C=%d L=%d/%d T=%d S=%.0f IB=%d ACC=%.1f%% (%.2fs +%.2fs)\n",
        phase, epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
        params.target, params.specificity, params.include_bits, (1 - val_stats.nmae) * 100, d, dd)
    end

  })

  print()
  print("Persisting")
  fs.rm("regressor.bin", true)
  t:persist("regressor.bin", true)

  print("Testing restore")
  t = tm.load("regressor.bin", nil, true)

  local train_pred = t:predict(train.problems, train.n, cfg.threads)
  local val_pred = t:predict(validate.problems, validate.n, cfg.threads)
  local test_pred = t:predict(test_set.problems, test_set.n, cfg.threads)

  local train_stats = eval.regression_accuracy(train_pred, train.targets)
  local val_stats = eval.regression_accuracy(val_pred, validate.targets)
  local test_stats = eval.regression_accuracy(test_pred, test_set.targets)

  str.printf("\nResults (Accuracy):\n")
  str.printf("  Train:    %.1f%%\n", (1 - train_stats.nmae) * 100)
  str.printf("  Validate: %.1f%%\n", (1 - val_stats.nmae) * 100)
  str.printf("  Test:     %.1f%%\n", (1 - test_stats.nmae) * 100)

end)
