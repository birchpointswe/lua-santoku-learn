local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local dvec = require("santoku.dvec")
local elm = require("santoku.learn.elm")
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
    mode = "sin",
    n_hidden = 8192,
    lambda = { def = 1.0, min = 0, max = 1 },
    search_trials = 400,
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

  local encoder, train_h = elm.create({
    mode = cfg.elm.mode,
    n_samples = train.n, n_tokens = n_features, n_hidden = cfg.elm.n_hidden,
    csc_offsets = train_csc_off, csc_indices = train_csc_idx,
  })

  local train_dense = dvec.create():copy(train.continuous)
  local dense_mean, dense_istd = train_dense:mtx_standardize(n_continuous)

  local scale = math.sqrt(cfg.elm.n_hidden / n_continuous)
  train_dense:scale(scale)
  local dims = cfg.elm.n_hidden + n_continuous
  train_h:mtx_extend(train_dense, cfg.elm.n_hidden, n_continuous)

  local val_h = encoder:encode({
    csc_offsets = val_csc_off, csc_indices = val_csc_idx, n_samples = validate.n,
  })
  local val_dense = dvec.create():copy(validate.continuous)
  val_dense:mtx_standardize(n_continuous, dense_mean, dense_istd)
  val_dense:scale(scale)
  val_h:mtx_extend(val_dense, cfg.elm.n_hidden, n_continuous)

  local _, ridge_obj, best_params = optimize.ridge({
    n_samples = train.n, n_dims = dims, codes = train_h,
    targets = train.targets, n_targets = 1,
    lambda = cfg.elm.lambda,
    search_trials = cfg.elm.search_trials,
    val_codes = val_h, val_n_samples = validate.n,
    val_targets = validate.targets,
    each = util.make_ridge_log(stopwatch),
  })
  str.printf("\nBest: H=%d lambda=%.4e\n", cfg.elm.n_hidden, best_params.lambda)
  str.printf("Time: %.1fs\n", stopwatch())

  local function encode_split(csc_off, csc_idx, dense_dv, n)
    local h = encoder:encode({
      csc_offsets = csc_off, csc_indices = csc_idx, n_samples = n,
    })
    local d = dvec.create():copy(dense_dv)
    d:mtx_standardize(n_continuous, dense_mean, dense_istd)
    d:scale(scale)
    h:mtx_extend(d, cfg.elm.n_hidden, n_continuous)
    return h
  end

  print("\nEvaluating splits")
  local train_stats = eval.regression_accuracy(ridge_obj:regress(train_h, train.n), train.targets)
  local val_stats = eval.regression_accuracy(ridge_obj:regress(val_h, validate.n), validate.targets)
  local test_h = encode_split(test_csc_off, test_csc_idx, test_set.continuous, test_set.n)
  local test_stats = eval.regression_accuracy(ridge_obj:regress(test_h, test_set.n), test_set.targets)

  str.printf("\nResults (Accuracy):\n")
  str.printf("  Train:    %.1f%%\n", (1 - train_stats.nmae) * 100)
  str.printf("  Validate: %.1f%%\n", (1 - val_stats.nmae) * 100)
  str.printf("  Test:     %.1f%%\n", (1 - test_stats.nmae) * 100)

end)
