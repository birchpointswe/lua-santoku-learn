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
  landmark_buf_path = "test/res/eurlex-lm",
  blocks = {
    { ngram_min = 1, ngram_max = 5, mode = "flat" },
    -- { ngram_min = 1, ngram_max = 3, mode = "words" },
  },
  relevance = { "bns"--[[, "bns"]] },
  -- scales = { def = { 81.8926, 0.0122111 } },
  exponent = { def = { 3.0130623 } },
  decode_offset = { def = 0.35471344 },
  n_landmarks = 1024 * 8,
  kernel = { "cosine" },
  lambda = { def = 1.01227e-07 },
  k = 256,
  search_trials = 0,
  folds = 5,
}

local function materialize (iter)
  local t, x = {}, iter()
  while x do t[#t + 1] = x; x = iter() end
  return t
end

test("eurlex CV", function ()
  local stopwatch = utc.stopwatch()
  str.printf("[Data] Loading\n")
  local train, _, test_set = ds.read_eurlex57k("test/res/eurlex57k")
  local n_labels = train.n_labels
  local pool_texts, test_texts = materialize(train.text_iter()), materialize(test_set.text_iter())
  str.printf("[Data] pool=%d test=%d labels=%d folds=%d trials=%d\n",
    train.n, test_set.n, n_labels, cfg.folds, cfg.search_trials)

  local Wtr = util.word_spans(pool_texts, train.n)
  local Wte = util.word_spans(test_texts, test_set.n)
  local toks, pool_blocks = util.tokenize_blocks(cfg.blocks, pool_texts, { tokens = Wtr })
  local _, test_blocks = util.tokenize_blocks(cfg.blocks, test_texts, { toks = toks, tokens = Wte })

  local _, ridge_obj, deploy, best, decider = optimize.krr({
    pool_blocks = pool_blocks,
    pool_labels = train.labels,
    n_labels = n_labels,
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
    landmark_buf_path = cfg.landmark_buf_path,
    k = cfg.k,
    decode_offset = cfg.decode_offset,
    search_trials = cfg.search_trials,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  })

  local P = util.predict_tiled({ deploy = deploy, ridge = ridge_obj,
    blocks = test_blocks, n = test_set.n, k = cfg.k })
  local _, m = decider:score({ pred = P, expected = test_set.labels, n_samples = test_set.n })
  local _, total = stopwatch()
  str.printf("[Result] scales=%s offset=%.8g | test %s\nTotal: %.1fs\n",
    util.vecstr(best.scales), decider:offset(), util.fmt_metrics(m), total)
end)
