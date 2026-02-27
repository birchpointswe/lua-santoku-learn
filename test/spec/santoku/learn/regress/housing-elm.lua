local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

local cfg = {
  data = {
    ttr = 0.8,
    tvr = 0.10,
    max = nil,
    n_thresholds = 64,
  },
  elm = {
    n_hidden = 8192,
    seed = 42,
    lambda = { def = 1.0 },
    search_trials = 400,
    n_models = 1,
  },
}

test("housing elm regressor", function ()

  print("Reading data")
  local dataset = ds.read_california_housing("test/res/california-housing.csv", {
    n_thresholds = cfg.data.n_thresholds,
    max = cfg.data.max,
  })
  str.printf("  Loaded %d samples with %d features (%d continuous)\n",
    dataset.n, dataset.n_features, #dataset.feature_cols)

  print("Splitting")
  local train, test_set, validate = ds.split_california_housing(dataset, cfg.data.ttr, cfg.data.tvr)
  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test_set.n)

  local target_max = train.targets:max()
  local target_min = train.targets:min()
  str.printf("  Target range: %.0f - %.0f\n", target_min, target_max)

  local n_features = dataset.n_features
  local n_continuous = train.n_continuous
  local train_csc_off, train_csc_idx = csr.to_csc(train.bits, train.n, n_features)
  local val_csc_off, val_csc_idx = csr.to_csc(validate.bits, validate.n, n_features)
  local test_csc_off, test_csc_idx = csr.to_csc(test_set.bits, test_set.n, n_features)

  print("\nTraining ELM")
  local stopwatch = utc.stopwatch()
  local elm_obj, elm_params, _, train_scores = optimize.elm({
    n_samples = train.n,
    n_tokens = n_features,
    n_hidden = cfg.elm.n_hidden,
    seed = cfg.elm.seed,
    csc_offsets = train_csc_off,
    csc_indices = train_csc_idx,
    dense_features = train.continuous,
    n_dense = n_continuous,
    targets = train.targets,
    n_targets = 1,
    val_csc_offsets = val_csc_off,
    val_csc_indices = val_csc_idx,
    val_n_samples = validate.n,
    val_dense_features = validate.continuous,
    val_targets = validate.targets,
    lambda = cfg.elm.lambda,
    search_trials = cfg.elm.search_trials,
    n_models = cfg.elm.n_models,
    each = util.make_elm_log(stopwatch),
  })
  str.printf("\nBest: H=%d lambda=%.4e\n", elm_params.n_hidden, elm_params.lambda)
  str.printf("Time: %.1fs\n", stopwatch())

  print("\nEvaluating splits")
  local train_stats = eval.regression_accuracy(train_scores, train.targets)
  local val_pred = elm_obj:transform(val_csc_off, val_csc_idx, validate.n, validate.continuous)
  local val_stats = eval.regression_accuracy(val_pred, validate.targets)
  local test_pred = elm_obj:transform(test_csc_off, test_csc_idx, test_set.n, test_set.continuous)
  local test_stats = eval.regression_accuracy(test_pred, test_set.targets)

  str.printf("\nResults (Accuracy):\n")
  str.printf("  Train:    %.1f%%\n", (1 - train_stats.nmae) * 100)
  str.printf("  Validate: %.1f%%\n", (1 - val_stats.nmae) * 100)
  str.printf("  Test:     %.1f%%\n", (1 - test_stats.nmae) * 100)

end)
