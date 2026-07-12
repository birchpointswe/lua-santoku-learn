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
  data = { ttr = 0.5 },
  blocks = {
    { ngram_min = 1, ngram_max = 5, mode = "flat" },
    { ngram_min = 1, ngram_max = 3, mode = "words" },
  },
  relevance = { "bns", "bns" },
  scales = { def = { 1.844, 0.542301 } },
  exponent = { def = { 1.94251, 1.73427 } },
  decode_offset = { def = 0.460847 },
  n_landmarks = 1024 * 8,
  kernel = { "cosine" },
  lambda = { def = 0.0298522 },
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

  local Wtr = util.word_spans(train.problems, train.n)
  local Wte = util.word_spans(test_set.problems, test_set.n)
  local toks, pool_blocks = util.tokenize_blocks(cfg.blocks, train.problems, { tokens = Wtr })
  local _, test_blocks = util.tokenize_blocks(cfg.blocks, test_set.problems, { toks = toks, tokens = Wte })

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
    landmark_rounds = cfg.landmark_rounds,
    search_landmark_rounds = cfg.search_landmark_rounds,
    k = cfg.k,
    search_trials = cfg.search_trials,
    decode_offset = cfg.decode_offset,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  })

  local P = util.predict_tiled({ deploy = deploy, ridge = ridge_obj,
    blocks = test_blocks, n = test_set.n, k = 1 })
  local _, m = decider:score({ pred = P, expected = test_set.labels, n_samples = test_set.n })
  local _, total = stopwatch()
  str.printf("[Result] scales=%s lambda=%.8g offset=%.8g | test %s\nTotal: %.1fs\n",
    util.vecstr(best.scales), best.lambda or 0, decider:offset(), util.fmt_metrics(m), total)
end)
