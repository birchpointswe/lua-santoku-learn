local optimize = require("santoku.learn.optimize")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

-- oracle: test 1-nmae=0.8710 (cold mint 2048/8192/1200 seed=5)
-- best: matern nu=1/2 (def=0) gamma=0.94471917 lambda=0.00020409608
-- seed_ensemble: K=1 test=0.8710, K=8 test=0.8748
local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  data = { ttr = 0.8 },
  n_landmarks = 1024 * 8,
  kernel = { "matern" },
  nu = { def = 0 },
  gamma = { def = 0.94471917 },
  lambda = { def = 0.00020409608 },
  scales = { def = { 38.400037, 55.697265, 0.25857085, 0.20780899, 0.044250118, 0.12342444, 0.017418369, 0.72738569, 125.74848 } },
  search_trials = 0,
  seed_ensemble = 1,
  scratch_path = "test/res/housing-scratch",
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

  local sp_enc, ridge_obj, deploy = optimize.krr(util.merged(cfg, {
    pool_blocks = pool_blocks,
    pool_targets = train.targets,
    n_targets = 1,
    pool_n = train.n,
    each = util.make_ridge_log(stopwatch),
  }))

  local _, test_scores = util.predict_tiled({ deploy = deploy, ridge = ridge_obj,
    blocks = test_blocks, n = test_set.n, scores = true, n_labels = 1 })
  local ts = eval.regress_accuracy(test_scores, test_set.targets)
  local _, total = stopwatch()
  str.printf("[Result] test=%.4f\nTotal: %.1fs\n", 1 - ts.nmae, total)

  local bundle = require("santoku.learn.bundle")
  local bdir = os.tmpname() .. ".bundle"
  bundle.persist({ dir = bdir, encoder = sp_enc, ridge = ridge_obj })
  local dep = str.format("%.4f", 1 - ts.nmae)
  sp_enc, ridge_obj, deploy, test_scores, pool_blocks = nil -- luacheck: ignore
  collectgarbage("collect")
  local b = bundle.load(bdir)
  local _, sb = util.predict_tiled({ deploy = b.encode, ridge = b.ridge,
    blocks = test_blocks, n = test_set.n, scores = true, n_labels = 1 })
  local tsb = eval.regress_accuracy(sb, test_set.targets)
  str.printf("[Bundle] reload test=%.4f (deploy %s)\n", 1 - tsb.nmae, dep)
  assert(str.format("%.4f", 1 - tsb.nmae) == dep, "reloaded bundle metric diverges from deploy")
  util.rmbundle(bdir)
end)
