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

-- oracle: test spF1=0.966330 (P=0.967187 R=0.965475) (cold mint config, 0/0-verified seed=5)
-- best: matern nu=1/2 (def=0) gamma=0.20602912 lambda=1.6712232e-05 decode_offset=-0.56368208
-- footnote: warm champion tested spF1=0.968011 (+0.0008, upper-tail, not cold-reachable)
-- seed_ensemble: K=1 spF1=0.966330, K=8 spF1=0.966955
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
    nu = { def = 0 },
    gamma = { def = 0.20602912 },
    lambda = { def = 1.6712232e-05 },
    relevance = { "bns", "bns", "auc" },
    scales = { def = { 0.17283993, 58.126236, 2.557732, 0.26508607, 0.028053508, 0.72071547, 0.14748457, 0.92141463, 5.4495301, 0.017593886, 557.27326 } },
    exponent = { def = { 7.92257, 3.678052, 1.3781166, 6.7906309, 4.8926748, 6.2641423, 2.3517682, 4.7486136, 6.459379, 4.0679645, 2.3755379 } },
    decode_offset = { def = -0.56368208 },
    search_trials = 0,
    seed_ensemble = 1,
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

  -- `scratch` mmaps the tokenizer output CSRs to disk instead of RAM; comment it
  -- to A/B against the all-RAM path (results are bit-identical, only RSS differs).
  local toks, Xtr = util.tokenize_blocks(cfg.blocks, pool.texts,
    { focus = Ctr, tokens = Ttr, scratch = "test/res/conll-gaz-blocks" })
  local _, Xte = util.tokenize_blocks(cfg.blocks, test_set.texts,
    { toks = toks, focus = Cte, tokens = Tte, scratch = "test/res/conll-gaz-blocks.te" })
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
  local dep = util.fmt_metrics(m)
  enc, rg, deploy, decider, toks, serve_gaz, Xtr, Xte, test_scores, rms_w = nil -- luacheck: ignore
  collectgarbage("collect")
  local b = bundle.load(bdir)
  local _, Xb = util.tokenize_blocks(cfg.blocks, test_set.texts, { toks = b.tokenizers, focus = Cte, tokens = Tte })
  Xb[n_sparse + 1] = b.gaz:block(test_set.texts, Cte, nil)
  Xb[n_sparse + 1]:bns(b.gaz_rms)
  local _, sb = util.predict_tiled({ deploy = b.encode, ridge = b.ridge,
    blocks = Xb, n = n_test, scores = true, n_labels = 1 })
  local _, mb = b.decider:score({ scores = sb, n_samples = test_set.n, cand = Cte, gold = Gte })
  str.printf("[Bundle] reload test %s (deploy %s)\n", util.fmt_metrics(mb), dep)
  assert(util.fmt_metrics(mb) == dep, "reloaded bundle metrics diverge from deploy")
  util.rmbundle(bdir)
  for _, base in ipairs({ "test/res/conll-gaz-blocks", "test/res/conll-gaz-blocks.te" }) do
    for i = 1, n_sparse do
      for _, sfx in ipairs({ ".off", ".toks", ".vals" }) do os.remove(base .. "." .. i .. sfx) end
    end
  end
end)
