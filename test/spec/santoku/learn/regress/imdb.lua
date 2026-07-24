local optimize = require("santoku.learn.optimize")
local ds = require("santoku.learn.dataset")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

-- oracle: test miF1=0.897999 (miP=0.886629 miR=0.909665) (cold mint 2048/8192/1200 seed=5)
-- best: cosine lambda=0.025471643 decode_offset=0.49393576 scales={0.082008995,12.193784} exp={6.6175246,1.1081682}
-- footnote: the 0.9127 config is achievable but CV-unselectable (gauge-blind); see project_selectability_gauge
-- seed_ensemble: K=1 miF1=0.897999, K=8 miF1=0.903856
local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  data = { ttr = 0.5 },
  blocks = {
    { ngram_min = 1, ngram_max = 5, mode = "flat" },
    { ngram_min = 1, ngram_max = 3, mode = "words" },
  },
  relevance = { "bns", "bns" },
  scales = { def = { 0.082008995, 12.193784 } },
  exponent = { def = { 6.6175246, 1.1081682 } },
  decode_offset = { def = 0.49393576 },
  n_landmarks = 1024 * 8,
  kernel = { "cosine" },
  lambda = { def = 0.025471643 },
  classes = 1,
  k = 1,
  search_trials = 0,
  seed_ensemble = 1,
  scratch_path = "test/res/imdb-scratch",
  folds = 5,
}

test("imdb CV", function ()
  local stopwatch = utc.stopwatch()
  str.printf("[Data] Loading\n")
  local dataset = ds.read_imdb("test/res/imdb.50k")
  local train, test_set = ds.split_imdb(dataset, cfg.data.ttr)
  str.printf("[Data] pool=%d test=%d folds=%d trials=%d\n",
    train.n, test_set.n, cfg.folds, cfg.search_trials)

  local Wtr = util.word_spans(train.problems, train.n)
  local Wte = util.word_spans(test_set.problems, test_set.n)
  local toks, pool_blocks = util.tokenize_blocks(cfg.blocks, train.problems, { tokens = Wtr })
  local _, test_blocks = util.tokenize_blocks(cfg.blocks, test_set.problems, { toks = toks, tokens = Wte })

  local sp_enc, ridge_obj, deploy, best, decider = optimize.krr(util.merged(cfg, {
    pool_blocks = pool_blocks,
    pool_labels = train.labels,
    n_labels = cfg.classes,
    each = util.make_ridge_log(stopwatch),
  }))

  local P = util.predict_tiled({ deploy = deploy, ridge = ridge_obj,
    blocks = test_blocks, n = test_set.n, k = 1 })
  local _, m = decider:score({ pred = P, expected = test_set.labels, n_samples = test_set.n })
  local _, total = stopwatch()
  str.printf("[Result] scales=%s lambda=%.8g offset=%.8g | test %s\nTotal: %.1fs\n",
    util.vecstr(best.scales), best.lambda or 0, decider:offset(), util.fmt_metrics(m), total)

  local bundle = require("santoku.learn.bundle")
  local bdir = os.tmpname() .. ".bundle"
  bundle.persist({ dir = bdir, tokenizers = toks, encoder = sp_enc, ridge = ridge_obj, decider = decider })
  local dep = util.fmt_metrics(m)
  sp_enc, ridge_obj, deploy, decider, toks, pool_blocks, test_blocks, P = nil -- luacheck: ignore
  collectgarbage("collect")
  local b = bundle.load(bdir)
  local _, Xb = util.tokenize_blocks(cfg.blocks, test_set.problems, { toks = b.tokenizers, tokens = Wte })
  local Pb = util.predict_tiled({ deploy = b.encode, ridge = b.ridge, blocks = Xb, n = test_set.n, k = 1 })
  local _, mb = b.decider:score({ pred = Pb, expected = test_set.labels, n_samples = test_set.n })
  str.printf("[Bundle] reload test %s (deploy %s)\n", util.fmt_metrics(mb), dep)
  assert(util.fmt_metrics(mb) == dep, "reloaded bundle metrics diverge from deploy")
  util.rmbundle(bdir)
end)
