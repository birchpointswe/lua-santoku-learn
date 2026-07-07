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
  search_landmarks = 2048,
  data = { ttr = 0.5 },
  blocks = {
    { ngram_min = 1, ngram_max = 5, word_characters = word_characters },
    { ngram_min = 1, ngram_max = 3, word_characters = word_characters, words = true },
  },
  relevance = { "bns", "bns" },
  scales = { def = {2.20435,0.453649} },
  exponent = { def = {2.50189,1.73248} },
  decode_offset = { def = 0.467859 },
  n_landmarks = 1024 * 8,
  kernel = { "cosine" },
  lambda = { def = 0.0104708 },
  classes = 1,
  k = 1,
  search_trials = 0,
  folds = 5,
}

test("imdb CV", function ()
  local stopwatch = utc.stopwatch()
  str.printf("[Data] Loading\n")
  local dataset = ds.read_imdb("test/res/imdb.50k")
  local train, test_set = ds.split_imdb(dataset, cfg.data.ttr)
  str.printf("[Data] pool=%d test=%d folds=%d trials=%d\n",
    train.n, test_set.n, cfg.folds, cfg.search_trials)

  local toks, pool_blocks = util.tokenize_blocks(cfg.blocks, train.problems)
  local _, test_blocks = util.tokenize_blocks(cfg.blocks, test_set.problems, toks)

  local _, ridge_obj, deploy, best, decider = optimize.krr({
    pool_blocks = pool_blocks,
    pool_labels = train.labels,
    n_labels = cfg.classes,
    folds = cfg.folds,
    relevance = cfg.relevance,
    scales = cfg.scales,
    exponent = cfg.exponent,
    kernel = cfg.kernel,
    nu = cfg.nu,
    gamma = cfg.gamma,
    lambda = cfg.lambda,
    n_landmarks = cfg.n_landmarks,
    search_landmarks = cfg.search_landmarks,
    k = cfg.k,
    search_trials = cfg.search_trials,
    decode_offset = cfg.decode_offset,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  })

  local test_codes = deploy(test_blocks)
  local P = ridge_obj:label(test_codes, 1)
  local _, m = decider:score({ pred = P, expected = test_set.labels, n_samples = test_set.n })
  local _, total = stopwatch()
  str.printf("[Result] scales=%s lambda=%.4g offset=%.6g | test %s\nTotal: %.1fs\n",
    util.vecstr(best.scales), best.lambda or 0, decider:offset(), util.fmt_metrics(m), total)
end)
