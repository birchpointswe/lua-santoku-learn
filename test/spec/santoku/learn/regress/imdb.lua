local optimize = require("santoku.learn.optimize")
local ds = require("santoku.learn.dataset")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

-- oracle: test miF1=0.913670 (miP=0.895707 miR=0.932369)
-- best: cosine lambda=0.15053999 decode_offset=0.47083597
local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  data = { ttr = 0.5 },
  blocks = {
    { ngram_min = 1, ngram_max = 5, mode = "flat" },
    { ngram_min = 1, ngram_max = 3, mode = "words" },
  },
  relevance = { "bns", "bns" },
  scales = { def = { 2.1061338, 0.47480364 } },
  exponent = { def = { 2.3789142, 1.4690152 } },
  decode_offset = { def = 0.47083597 },
  n_landmarks = 1024 * 8,
  kernel = { "cosine" },
  lambda = { def = 0.15053999 },
  classes = 1,
  k = 1,
  search_trials = 0,
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
  local files = { "encoder.bin", "ridge.bin", "decider.bin", "manifest.lua" }
  for i = 1, #cfg.blocks do files[#files + 1] = "tokenizer_" .. i .. ".bin" end
  for _, f in ipairs(files) do os.remove(bdir .. "/" .. f) end
  os.remove(bdir)
end)
