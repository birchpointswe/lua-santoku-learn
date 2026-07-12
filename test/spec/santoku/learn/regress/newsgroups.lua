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
  data = { max = nil },
  blocks = {
    { ngram_min = 1, ngram_max = 5, mode = "flat" },
    { ngram_min = 1, ngram_max = 3, mode = "words" },
  },
  relevance = { "bns", "bns" },
  scales = { def = { 0.875402, 1.14233 } },
  exponent = { def = { 1.68339, 3.44989 } },
  n_landmarks = 1024 * 8,
  kernel = { "cosine" },
  lambda = { def = 1.58158e-06 },
  classes = 20,
  k = 1,
  search_trials = 0,
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

  local sp_enc, ridge_obj, deploy, best, decider, _, bake = optimize.krr({
    pool_blocks = pool_blocks,
    pool_labels = pool.labels,
    pool_class = pool.labels:neighbors(),
    n_labels = cfg.classes,
    folds = cfg.folds,
    relevance = cfg.relevance,
    scales = cfg.scales,
    exponent = cfg.exponent,
    kernel = cfg.kernel,
    lambda = cfg.lambda,
    n_landmarks = cfg.n_landmarks,
    search_landmarks = cfg.search_landmarks,
    landmark_rounds = cfg.landmark_rounds,
    search_landmark_rounds = cfg.search_landmark_rounds,
    k = cfg.k,
    search_trials = cfg.search_trials,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  })

  local test_codes = deploy(test_blocks)
  local _, m = decider:score({ scores = ridge_obj:regress(test_codes),
    n_samples = test_set.n, expected = test_set.labels })
  local _, total = stopwatch()
  str.printf("[Result] scales=%s lambda=%.8g | test %s\nTotal: %.1fs\n",
    util.vecstr(best.scales), best.lambda or 0, util.fmt_metrics(m), total)

  local baked = bake(test_blocks)
  local bundle = require("santoku.learn.bundle")
  local bdir = os.tmpname() .. ".bundle"
  bundle.persist({ dir = bdir, tokenizers = toks, encoder = sp_enc, ridge = ridge_obj,
    decider = decider, blocks = baked })
  local b = bundle.load(bdir)
  local _, test_blocks_b = util.tokenize_blocks(cfg.blocks, test_set.problems, { toks = b.tokenizers, tokens = Wte })
  local test_codes_b = b.encode(test_blocks_b)
  assert(test_codes:eq(test_codes_b), "bundle deploy codes diverge")
  assert(ridge_obj:regress(test_codes):eq(b.ridge:regress(test_codes_b)), "bundle scores diverge")
  assert(b.decider:offset() == decider:offset(), "bundle decider diverges")
  str.printf("[Bundle] round-trip bit-identical\n")
  local files = { "encoder.bin", "ridge.bin", "decider.bin", "manifest.lua" }
  for i = 1, #cfg.blocks do
    files[#files + 1] = "tokenizer_" .. i .. ".bin"
    files[#files + 1] = "colscale_" .. i .. ".bin"
  end
  for _, f in ipairs(files) do os.remove(bdir .. "/" .. f) end
  os.remove(bdir)
end)
