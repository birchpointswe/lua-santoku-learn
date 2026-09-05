local optimize = require("santoku.learn.optimize")
local ds = require("santoku.learn.dataset")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")
local fs = require("santoku.fs")

fs.stdout:setvbuf("line")

local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  data = { max = nil },
  blocks = {
    { ngram_min = 1, ngram_max = 5, mode = "flat" },
    { ngram_min = 1, ngram_max = 3, mode = "words" },
  },
  relevance = { "bns", "bns" },
  scales = { def = { 0.94523942, 1.057933 } },
  exponent = { def = { 1.7400224, 5.625631 } },
  n_landmarks = 1024 * 8,
  kernel = { "cosine" },
  lambda = { def = 1.5092313e-06 },
  classes = 20,
  k = 1,
  search_trials = 0,
  seed_ensemble = 1,
  scratch_path = "test/res/newsgroups-scratch",
  folds = 5,
}

test("newsgroups CV", function ()
  local stopwatch = utc.stopwatch()
  str.printf("[Data] Loading\n")
  local pool = ds.read_20newsgroups("test/res/20news-bydate-train", nil, nil, cfg.data.max)
  local test_set = ds.read_20newsgroups("test/res/20news-bydate-test")
  str.printf("[Data] pool=%d test=%d classes=%d folds=%d trials=%d\n",
    pool.n, test_set.n, cfg.classes, cfg.folds, cfg.search_trials)

  local Wtr = util.word_spans(pool.problems, pool.n)
  local Wte = util.word_spans(test_set.problems, test_set.n)
  local toks, pool_blocks = util.tokenize_blocks(cfg.blocks, pool.problems, { tokens = Wtr })
  local _, test_blocks = util.tokenize_blocks(cfg.blocks, test_set.problems, { toks = toks, tokens = Wte })

  local sp_enc, ridge_obj, deploy, best, decider = optimize.krr(util.merged(cfg, {
    pool_blocks = pool_blocks,
    pool_labels = pool.labels,
    pool_class = pool.labels:neighbors(),
    n_labels = cfg.classes,
    each = util.make_ridge_log(stopwatch),
  }))

  local _, test_scores = util.predict_tiled({ deploy = deploy, ridge = ridge_obj,
    blocks = test_blocks, n = test_set.n, scores = true, n_labels = cfg.classes })
  local _, m = decider:score({ scores = test_scores,
    n_samples = test_set.n, expected = test_set.labels })
  local _, total = stopwatch()
  str.printf("[Result] scales=%s lambda=%.8g | test %s\nTotal: %.1fs\n",
    util.vecstr(best.scales), best.lambda or 0, util.fmt_metrics(m), total)

  local bundle = require("santoku.learn.bundle")
  local bdir = fs.tmpname() .. ".bundle"
  bundle.persist({ dir = bdir, tokenizers = toks, encoder = sp_enc, ridge = ridge_obj,
    decider = decider })
  local dep = util.fmt_metrics(m)
  sp_enc, ridge_obj, deploy, decider, toks, test_scores, test_blocks = nil -- luacheck: ignore
  collectgarbage("collect")
  local b = bundle.load(bdir)
  local _, Xb = util.tokenize_blocks(cfg.blocks, test_set.problems, { toks = b.tokenizers, tokens = Wte })
  local _, sb = util.predict_tiled({ deploy = b.encode, ridge = b.ridge,
    blocks = Xb, n = test_set.n, scores = true, n_labels = cfg.classes })
  local _, mb = b.decider:score({ scores = sb,
    n_samples = test_set.n, expected = test_set.labels })
  str.printf("[Bundle] reload test %s (deploy %s)\n", util.fmt_metrics(mb), dep)
  assert(util.fmt_metrics(mb) == dep, "reloaded bundle metrics diverge from deploy")
  util.rmbundle(bdir)
end)
