local optimize = require("santoku.learn.optimize")
local ds = require("santoku.learn.dataset")
local util = require("santoku.learn.util")
local mtx = require("santoku.mtx")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")
local fs = require("santoku.fs")

fs.stdout:setvbuf("line")

local cfg = {
  search_landmarks = 1024 * 2,
  n_landmarks = 1024 * 8,
  kernel = { "matern" },
  nu = { def = 0 },
  gamma = { def = 0.33884277 },
  lambda = { def = 0.0044132295 },
  scales = { def = { 0.80902091, 0.006489361, 465.91994, 0.31863918, 0.31888871, 5.3659771, 0.74979041 } },
  classes = 2,
  k = 1,
  search_trials = 0,
  folds = 5,
}

local function dense_block (d, n_dense, mean)
  local X = mtx.create({ data = d.dense:data():to_fvec(), n_rows = d.n, n_cols = n_dense })
  if mean then X:center(mean) else mean = X:center() end
  return X:to_sparse():i32(), mean
end

test("adult CV", function ()
  local stopwatch = utc.stopwatch()
  local train, test_set, meta = ds.read_adult("test/res/adult")
  str.printf("[Data] pool=%d test=%d bits=%d dense=%d folds=%d trials=%d\n",
    train.n, test_set.n, meta.n_bits, meta.n_dense, cfg.folds, cfg.search_trials)
  assert(train.n == 32561 and test_set.n == 16281)

  local Xc, mean = dense_block(train, meta.n_dense)
  local Xt = dense_block(test_set, meta.n_dense, mean)
  local st = Xc:standardize(); Xt:scale_cols(st)
  local bits = train.bits:i32(); local bst = bits:standardize()
  local bits_t = test_set.bits:i32(); bits_t:scale_cols(bst)
  local go = {}
  for g = 0, meta.n_dense do go[g + 1] = g end
  local pool_blocks = { { x = Xc, group_offsets = go }, bits }
  local test_blocks = { { x = Xt, group_offsets = go }, bits_t }

  local sp_enc, ridge_obj, deploy, best, decider = optimize.krr(util.merged(cfg, {
    pool_blocks = pool_blocks,
    pool_labels = train.labels,
    pool_class = train.labels:neighbors(),
    n_labels = cfg.classes,
    each = util.make_ridge_log(stopwatch),
  }))

  local _, test_scores = util.predict_tiled({ deploy = deploy, ridge = ridge_obj,
    blocks = test_blocks, n = test_set.n, scores = true, n_labels = cfg.classes })
  local _, m = decider:score({ scores = test_scores, n_samples = test_set.n, expected = test_set.labels })
  local _, total = stopwatch()
  str.printf("[Result] scales=%s lambda=%.8g | test %s\nTotal: %.1fs\n",
    util.vecstr(best.scales), best.lambda or 0, util.fmt_metrics(m), total)

  local bundle = require("santoku.learn.bundle")
  local bdir = fs.tmpname() .. ".bundle"
  bundle.persist({ dir = bdir, encoder = sp_enc, ridge = ridge_obj, decider = decider })
  local dep = util.fmt_metrics(m)
  sp_enc, ridge_obj, deploy, decider, test_scores, pool_blocks = nil -- luacheck: ignore
  collectgarbage("collect")
  local b = bundle.load(bdir)
  local _, sb = util.predict_tiled({ deploy = b.encode, ridge = b.ridge,
    blocks = test_blocks, n = test_set.n, scores = true, n_labels = cfg.classes })
  local _, mb = b.decider:score({ scores = sb, n_samples = test_set.n, expected = test_set.labels })
  str.printf("[Bundle] reload test %s (deploy %s)\n", util.fmt_metrics(mb), dep)
  assert(util.fmt_metrics(mb) == dep, "reloaded bundle metrics diverge from deploy")
  util.rmbundle(bdir)
end)
