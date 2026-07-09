local optimize = require("santoku.learn.optimize")
local ds = require("santoku.learn.dataset")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local word_characters = util.WORD_CHARACTERS

local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  data = { max = nil },
  blocks = {
    { ngram_min = 1, ngram_max = 5, word_characters = word_characters },
    { ngram_min = 1, ngram_max = 3, word_characters = word_characters, words = true },
  },
  relevance = { "bns", "bns" },
  scales = { def = { 0.762539, 1.31141 } },
  exponent = { def = { 1.41983, 3.88377 } },
  n_landmarks = 1024 * 8,
  kernel = { "cosine" },
  lambda = { def = 8.05697e-06 },
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

  local toks, pool_blocks = util.tokenize_blocks(cfg.blocks, pool.problems)
  local _, test_blocks = util.tokenize_blocks(cfg.blocks, test_set.problems, toks)

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
    k = cfg.k,
    search_trials = cfg.search_trials,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  })

  local test_codes = deploy(test_blocks)
  local _, m = decider:score({ scores = ridge_obj:regress(test_codes),
    n_samples = test_set.n, expected = test_set.labels })
  local _, total = stopwatch()
  str.printf("[Result] scales=%s lambda=%.4g | test %s\nTotal: %.1fs\n",
    util.vecstr(best.scales), best.lambda or 0, util.fmt_metrics(m), total)

  local baked = bake(test_blocks)
  local bundle = require("santoku.learn.bundle")
  local bdir = os.tmpname() .. ".bundle"
  bundle.persist({ dir = bdir, tokenizers = toks, encoder = sp_enc, ridge = ridge_obj,
    decider = decider, blocks = baked })
  local b = bundle.load(bdir)
  local _, test_blocks_b = util.tokenize_blocks(cfg.blocks, test_set.problems, b.tokenizers)
  local test_codes_b = b.encode(test_blocks_b)
  assert(test_codes:eq(test_codes_b), "bundle deploy codes diverge")
  assert(ridge_obj:regress(test_codes):eq(b.ridge:regress(test_codes_b)), "bundle scores diverge")
  assert(b.decider:offset() == decider:offset(), "bundle decider diverges")
  str.printf("[Bundle] round-trip bit-identical\n")
  for _, f in ipairs({ "tokenizer_1.bin", "tokenizer_2.bin", "encoder.bin", "ridge.bin",
      "decider.bin", "colscale_1.bin", "colscale_2.bin", "manifest.lua" }) do
    os.remove(bdir .. "/" .. f)
  end
  os.remove(bdir)
end)
