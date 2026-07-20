local optimize = require("santoku.learn.optimize")
local ds = require("santoku.learn.dataset")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

-- oracle: test acc=0.982143 maF1=0.982017
-- best: matern nu=inf (def=3) gamma=2.91919 lambda=2.13137e-07 exp={0.278601}
local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
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
  scratch_path = "test/res/mnist-scratch",
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

  local sp_enc, ridge_obj, deploy, best, decider = optimize.krr(util.merged(cfg, {
    pool_blocks = { pool_codes:i32() },
    pool_labels = train.labels,
    pool_class = train.labels:neighbors(),
    n_labels = cfg.classes,
    each = util.make_ridge_log(stopwatch),
  }))

  local _, test_scores = util.predict_tiled({ deploy = deploy, ridge = ridge_obj,
    blocks = { test_codes:i32() }, n = test_set.n, scores = true, n_labels = cfg.classes })
  local _, m = decider:score({ scores = test_scores,
    n_samples = test_set.n, expected = test_set.labels })
  local _, total = stopwatch()
  str.printf("[Result] lambda=%.8g | test %s\nTotal: %.1fs\n", best.lambda or 0, util.fmt_metrics(m), total)

  local bundle = require("santoku.learn.bundle")
  local bdir = os.tmpname() .. ".bundle"
  bundle.persist({ dir = bdir, encoder = sp_enc, ridge = ridge_obj, decider = decider })
  local dep = util.fmt_metrics(m)
  sp_enc, ridge_obj, deploy, decider, test_scores, pool_codes = nil -- luacheck: ignore
  collectgarbage("collect")
  local b = bundle.load(bdir)
  local _, sb = util.predict_tiled({ deploy = b.encode, ridge = b.ridge,
    blocks = { test_codes:i32() }, n = test_set.n, scores = true, n_labels = cfg.classes })
  local _, mb = b.decider:score({ scores = sb, n_samples = test_set.n, expected = test_set.labels })
  str.printf("[Bundle] reload test %s (deploy %s)\n", util.fmt_metrics(mb), dep)
  assert(util.fmt_metrics(mb) == dep, "reloaded bundle metrics diverge from deploy")
  for _, f in ipairs({ "encoder.bin", "ridge.bin", "decider.bin", "manifest.lua" }) do
    os.remove(bdir .. "/" .. f)
  end
  os.remove(bdir)
end)
