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
local fvec = require("santoku.fvec")
local fs = require("santoku.fs")

io.stdout:setvbuf("line")

local N_TYPES = 4

-- oracle: test spF1=0.969541 (P=0.969713 R=0.969370)
-- best: matern nu=5/2 (def=2) gamma=0.033925952 lambda=3.7829147e-05 decode_offset=-0.49673414
local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  data = { dir = "test/res/conll2003", max = nil },
  tok = { ngram_min = 1, ngram_max = 5 },
  blocks = {
    { ngram_min = 1, ngram_max = 5, normalize = false, regions = true },
    { ngram_min = 1, ngram_max = 5, mode = "tags", n_tags = util.N_SHAPES, normalize = false, regions = true },
  },
  emb = { n_landmarks = 1024 * 8 },
  head = {
    kernel = { "matern" },
    nu = { def = 2 },
    gamma = { def = 0.033925952 },
    lambda = { def = 3.7829147e-05 },
    relevance = { "bns", "bns", "auc" },
    scales = { def = { 0.16700725, 750.25667, 0.022722455, 285.64255, 0.012953823, 0.0075025667, 0.0075025667, 0.058747138, 246.25134, 0.19469259, 598.74823 } },
    exponent = { def = { 1.3153303, 1.7915035, 3.0779086, 0.84781049, 6.1936501, 7.0719099, 7.8323064, 0.51783619, 2.3249964, 0.21921147, 7.9105983 } },
    decode_offset = { def = -0.49673414 },
    search_trials = 0,
    scratch_path = "test/res/conll-gaz-scratch",
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
  local df = util.doc_folds(Ctr, Gtr, K)
  local function build_cgaz (g)
    return ner.build_char_gaz({ texts = pool.texts, gold = g, n_types = N_TYPES,
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max })
  end
  local serve_gaz = build_cgaz(Gtr)
  Xtr[n_sparse + 1] = serve_gaz:block(pool.texts, Ctr, Ctr:type_labels(Gtr, N_TYPES))
  Xte[n_sparse + 1] = serve_gaz:block(test_set.texts, Cte, nil)
  local rms_w = util.rms_scale_blocks(Xtr, { Xte }, n_sparse + 1)

  local Ytr = cand_labels(Ctr, Gtr)

  local bdir = os.tmpname() .. ".bundle"
  fs.mkdirp(bdir)
  local w_path, chol_path = bdir .. "/w.mmap", bdir .. "/chol.mmap"
  local w_buf = fvec.mmap_create(w_path, cfg.emb.n_landmarks)
  local enc_chol_buf = fvec.mmap_create(chol_path, cfg.emb.n_landmarks * cfg.emb.n_landmarks)

  local enc, rg, deploy, best, decider = optimize.krr(util.merged(cfg.head, {
    pool_blocks = Xtr,
    pool_labels = Ytr,
    pool_n = n_pool,
    n_labels = 1,
    reject = N_TYPES,
    doc_fold = df,
    cand = Ctr,
    gold = Gtr,
    n_landmarks = cfg.emb.n_landmarks,
    w_buf = w_buf,
    enc_chol_buf = enc_chol_buf,
    search_landmarks = cfg.search_landmarks,
    k = 1,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  }))

  local _, test_scores = util.predict_tiled({ deploy = deploy, ridge = rg,
    blocks = Xte, n = n_test, scores = true, n_labels = 1 })
  local _, m = decider:score({ scores = test_scores,
    n_samples = test_set.n, cand = Cte, gold = Gte })
  local _, total = stopwatch()
  str.printf("[Span] lambda=%.8g offset=%.8g | test %s\nTotal: %.1fs\n",
    best.lambda or 0, decider:offset(), util.fmt_metrics(m), total)

  local bundle = require("santoku.learn.bundle")
  bundle.persist({ dir = bdir, tokenizers = toks, gaz = serve_gaz, gaz_rms = rms_w[n_sparse + 1],
    encoder = enc, ridge = rg, decider = decider, w_path = w_path, chol_path = chol_path })
  local b = bundle.load(bdir)
  local _, Xte_b = util.tokenize_blocks(cfg.blocks, test_set.texts,
    { toks = b.tokenizers, focus = Cte, tokens = Tte })
  Xte_b[n_sparse + 1] = b.gaz:block(test_set.texts, Cte, nil)
  Xte_b[n_sparse + 1]:bns(b.gaz_rms)
  local _, sb = util.predict_tiled({ deploy = b.encode, ridge = b.ridge,
    blocks = Xte_b, n = n_test, scores = true, n_labels = 1 })
  local maxd = 0
  for i = 0, n_test - 1 do
    local d = math.abs(test_scores:get(i) - sb:get(i))
    if d > maxd then maxd = d end
  end
  str.printf("[Bundle] serve-vs-deploy max score diff = %.3e\n", maxd)
  assert(maxd < 1e-3, str.format("bundle serve path diverges from deploy (%.3e)", maxd))
  for _, f in ipairs({ "tokenizer_1.bin", "tokenizer_2.bin", "encoder.bin", "ridge.bin",
      "decider.bin", "gaz.bin", "gaz_rms.bin", "w.mmap", "chol.mmap", "manifest.lua" }) do
    os.remove(bdir .. "/" .. f)
  end
  os.remove(bdir)
end)
