local optimize = require("santoku.learn.optimize")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  landmark_rounds = 32,
  search_landmark_rounds = 1,
  data = { ttr = 0.8 },
  n_landmarks = 1024 * 8,
  kernel = { "matern" },
  nu = { def = 0 },
  gamma = { def = 0.059538 },
  lambda = { def = 8.30506e-05 },
  scales = { def = { 34.6414, 45.6015, 0.25601, 0.358376, 0.0384308, 0.290168, 0.00997914, 0.688452, 90.0605 } },
  search_trials = 0,
  folds = 5,
}

test("housing CV", function ()
  local stopwatch = utc.stopwatch()
  str.printf("[Data] Loading\n")
  local dataset = ds.read_california_housing("test/res/california-housing.csv", {})
  local train, test_set = ds.split_california_housing(dataset, cfg.data.ttr)
  local n_cont = train.n_continuous

  local cont_mean = train.continuous:center()
  test_set.continuous:center(cont_mean)
  local Xc = train.continuous:to_sparse():i32()
  local sp = Xc:standardize()
  local Xt = test_set.continuous:to_sparse():i32()
  Xt:standardize(sp)
  local bits = train.bits:i32(); local bits_std = bits:standardize()
  local bits_t = test_set.bits:i32(); bits_t:standardize(bits_std)
  local go = {}
  for g = 0, n_cont do go[g + 1] = g end
  local pool_blocks = { { x = Xc, group_offsets = go }, bits }
  local test_blocks = { { x = Xt, group_offsets = go }, bits_t }
  str.printf("[Data] pool=%d test=%d gauge_dims=%d folds=%d trials=%d\n",
    train.n, test_set.n, n_cont + 1, cfg.folds, cfg.search_trials)

  local _, ridge_obj, deploy = optimize.krr({
    pool_blocks = pool_blocks,
    pool_targets = train.targets,
    n_targets = 1,
    pool_n = train.n,
    scales = cfg.scales,
    folds = cfg.folds,
    kernel = cfg.kernel,
    nu = cfg.nu,
    gamma = cfg.gamma,
    lambda = cfg.lambda,
    n_landmarks = cfg.n_landmarks,
    search_landmarks = cfg.search_landmarks,
    landmark_rounds = cfg.landmark_rounds,
    search_landmark_rounds = cfg.search_landmark_rounds,
    search_trials = cfg.search_trials,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  })

  local _, test_scores = util.predict_tiled({ deploy = deploy, ridge = ridge_obj,
    blocks = test_blocks, n = test_set.n, scores = true, n_labels = 1 })
  local ts = eval.regress_accuracy(test_scores, test_set.targets)
  local _, total = stopwatch()
  str.printf("[Result] test=%.4f\nTotal: %.1fs\n", 1 - ts.nmae, total)
end)
