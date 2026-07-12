local optimize = require("santoku.learn.optimize")
local ds = require("santoku.learn.dataset")
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
  data = { ttr = 0.8, features = 784 },
  n_landmarks = 1024 * 8,
  kernel = { "matern" },
  nu = { def = 3 },
  gamma = { def = 2.91919 },
  lambda = { def = 2.13137e-07 },
  relevance = { "auc" },
  exponent = { def = { 0.278601 } },
  classes = 10,
  k = 1,
  search_trials = 0,
  folds = 5,
}

test("mnist CV", function ()
  local stopwatch = utc.stopwatch()
  str.printf("[Data] Loading\n")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.features)
  local train, test_set = ds.split_binary_mnist(dataset, cfg.data.ttr)
  local pool_codes = dataset.problems:rows(train.ids):normalize()
  local test_codes = dataset.problems:rows(test_set.ids):normalize()
  str.printf("[Data] pool=%d test=%d features=%d classes=%d folds=%d trials=%d\n",
    train.n, test_set.n, cfg.data.features, cfg.classes, cfg.folds, cfg.search_trials)

  local _, ridge_obj, deploy, best, decider = optimize.krr({
    pool_blocks = { pool_codes:i32() },
    pool_labels = train.labels,
    pool_class = train.labels:neighbors(),
    n_labels = cfg.classes,
    folds = cfg.folds,
    kernel = cfg.kernel,
    nu = cfg.nu,
    gamma = cfg.gamma,
    lambda = cfg.lambda,
    relevance = cfg.relevance,
    exponent = cfg.exponent,
    n_landmarks = cfg.n_landmarks,
    search_landmarks = cfg.search_landmarks,
    landmark_rounds = cfg.landmark_rounds,
    search_landmark_rounds = cfg.search_landmark_rounds,
    k = cfg.k,
    search_trials = cfg.search_trials,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  })

  local _, test_scores = util.predict_tiled({ deploy = deploy, ridge = ridge_obj,
    blocks = { test_codes:i32() }, n = test_set.n, scores = true, n_labels = cfg.classes })
  local _, m = decider:score({ scores = test_scores,
    n_samples = test_set.n, expected = test_set.labels })
  local _, total = stopwatch()
  str.printf("[Result] lambda=%.8g | test %s\nTotal: %.1fs\n", best.lambda or 0, util.fmt_metrics(m), total)
end)
