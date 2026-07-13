local ds = require("santoku.learn.dataset")
local optimize = require("santoku.learn.optimize")
local ner = require("santoku.learn.ner")
local csr = require("santoku.csr")
local spans = require("santoku.spans")
local ivec = require("santoku.ivec")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local N_TYPES = 4

local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  landmark_rounds = 32,
  search_landmark_rounds = 1,
  data = { dir = "test/res/conll2003", max = nil },
  tok = { ngram_min = 1, ngram_max = 5 },
  blocks = {
    { ngram_min = 1, ngram_max = 5, normalize = false, regions = true },
    { ngram_min = 1, ngram_max = 5, mode = "tags", n_tags = util.N_SHAPES, normalize = false, regions = true },
  },
  emb = { n_landmarks = 1024 * 8 },
  head = {
    kernel = { "matern" },
    nu = { def = 1 },
    gamma = { def = 0.53607869 },
    lambda = { def = 2.9299902e-05 },
    relevance = { "bns", "bns", "auc" },
    scales = { def = { 0.0045204084, 0.0045204084, 0.49983253, 15.168852, 0.0045204084,
      0.0045204084, 20.022861, 56.54299, 44.639002, 14.48332, 431.54347 } },
    exponent = { def = { 3.2694916, 6.8227349, 7.5514373, 1.0992432, 1.4892054,
      7.9947992, 1.5709097, 5.3751002, 0.79283053, 5.7866596, 5.2936447 } },
    decode_offset = { def = -0.46597821 },
    search_trials = 200,
    folds = 5,
  },
}

local function candidates (ac, pat_type, split, T)
  local S = ac:predict({ texts = split.texts, longest = true, tokens = T })
  local id = S:col("id")
  local ty = ivec.create(id:size()):copy(pat_type, id)
  return spans.create({ offsets = S:offsets(), s = S:col("s"), e = S:col("e"), ty = ty })
end

local function cand_labels (Scand, Sgold)
  return csr.from_mask(Scand:match_labels(Sgold))
end

test("conll-gaz CV", function ()
  local stopwatch = utc.stopwatch()
  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  local ac, pat_type = util.surface_gaz({ train, dev, test_set }, N_TYPES, false)
  local pool = ds.merge_conll2003(train, dev)
  local Ttr = util.shape_spans(pool.texts, pool.n)
  local Tte = util.shape_spans(test_set.texts, test_set.n)
  local Gtr, Gte = pool.gold, test_set.gold
  local Ctr, Cte = candidates(ac, pat_type, pool, Ttr), candidates(ac, pat_type, test_set, Tte)
  local n_pool, n_test = Ctr:col("s"):size(), Cte:col("s"):size()
  str.printf("[Cands] pool=%d test=%d | test coverage=%.4f folds=%d trials=%d\n",
    n_pool, n_test, Cte:coverage(Gte), cfg.head.folds, cfg.head.search_trials)

  local toks, Xtr = util.tokenize_blocks(cfg.blocks, pool.texts, { focus = Ctr, tokens = Ttr })
  local _, Xte = util.tokenize_blocks(cfg.blocks, test_set.texts, { toks = toks, focus = Cte, tokens = Tte })
  local n_sparse = #cfg.blocks

  local K = cfg.head.folds
  local df = util.doc_folds(Ctr, Gtr, K)  -- shared with krr so the cross-fit aligns to CV
  local function build_cgaz (g)
    return ner.build_char_gaz({ texts = pool.texts, gold = g, n_types = N_TYPES,
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max })
  end
  Xtr[n_sparse + 1] = util.gaz_block_oof({ folds = K, doc_fold = df, texts = pool.texts, cand = Ctr, gold = Gtr, build = build_cgaz })
  Xte[n_sparse + 1] = build_cgaz(Gtr):block(test_set.texts, Cte, nil)
  util.rms_scale_blocks(Xtr, { Xte }, n_sparse + 1)

  local Ytr = cand_labels(Ctr, Gtr)

  local _, rg, deploy, best, decider = optimize.krr({
    pool_blocks = Xtr,
    pool_labels = Ytr,
    pool_n = n_pool,
    n_labels = 1,
    reject = N_TYPES,
    folds = cfg.head.folds,
    doc_fold = df,
    cand = Ctr,
    gold = Gtr,
    relevance = cfg.head.relevance,
    scales = cfg.head.scales,
    exponent = cfg.head.exponent,
    kernel = cfg.head.kernel,
    nu = cfg.head.nu,
    gamma = cfg.head.gamma,
    lambda = cfg.head.lambda,
    n_landmarks = cfg.emb.n_landmarks,
    search_landmarks = cfg.search_landmarks,
    landmark_rounds = cfg.landmark_rounds,
    search_landmark_rounds = cfg.search_landmark_rounds,
    k = 1,
    decode_offset = cfg.head.decode_offset,
    search_trials = cfg.head.search_trials,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  })

  local _, test_scores = util.predict_tiled({ deploy = deploy, ridge = rg,
    blocks = Xte, n = n_test, scores = true, n_labels = 1 })
  local _, m = decider:score({ scores = test_scores,
    n_samples = test_set.n, cand = Cte, gold = Gte })
  local _, total = stopwatch()
  str.printf("[Span] lambda=%.8g offset=%.8g | test %s\nTotal: %.1fs\n",
    best.lambda or 0, decider:offset(), util.fmt_metrics(m), total)
end)
