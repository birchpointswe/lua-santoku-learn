local ds = require("santoku.learn.dataset")
local csr = require("santoku.csr")
local spans = require("santoku.spans")
local ner = require("santoku.learn.ner")
local optimize = require("santoku.learn.optimize")
local ivec = require("santoku.ivec")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local N_TYPES = 4
local MAX_SPAN = 1000000000

-- oracle: test spF1=0.817037 (P=0.824176 R=0.810021) (cold mint config, 0/0-verified seed=5)
-- tag:  matern nu=inf (def=3) gamma=0.79742469 lambda=5.9753931e-07 decode_offset=0.52165604
-- type: matern nu=inf (def=3) gamma=1.652384 lambda=9.0469132e-07 decode_offset=0.18440428
-- footnote: warm champion tested spF1=0.822491 (+0.0054, not cold-reachable)
-- seed_ensemble (both heads): K=1 spF1=0.817037, K=8 spF1=0.821719
local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  data = {
    dir = "test/res/conll2003",
    max = nil
  },
  tok = { ngram_min = 1, ngram_max = 5 },
  emb = { n_landmarks = 1024 * 8, },
  tag = {
    kernel = { "matern" },
    nu = { def = 3 },
    gamma = { def = 0.79742469 },
    lambda = { def = 5.9753931e-07 },
    blocks = {
      { ngram_min = 1, ngram_max = 5, normalize = false },
      { ngram_min = 1, ngram_max = 3, mode = "words", normalize = false },
      { ngram_min = 1, ngram_max = 5, mode = "tags", n_tags = util.N_SHAPES, normalize = false },
    },
    relevance = { "bns", "bns", "bns" },
    scales = { def = { 7.6690343, 0.0067839924, 19.220911 } },
    exponent = { def = { 4.9334456, 4.6089473, 1.4923638 } },
    decode_offset = { def = 0.52165604 },
    search_trials = 0,
    seed_ensemble = 1,
    scratch_path = "test/res/conll-tag-scratch",
    folds = 5,
  },
  type = {
    kernel = { "matern" },
    nu = { def = 3 },
    gamma = { def = 1.652384 },
    lambda = { def = 9.0469132e-07 },
    blocks = {
      { ngram_min = 1, ngram_max = 5, normalize = false, regions = true },
      { ngram_min = 1, ngram_max = 5, mode = "tags", n_tags = util.N_SHAPES, normalize = false, regions = true },
    },
    relevance = { "bns", "bns" },
    scales = { def = { 0.027032086, 12.111778, 0.066024415, 99.018754, 0.13420805, 0.0030055239, 25.498869, 0.040503204, 180.79393, 0.072337035, 85.750553 } },
    exponent = { def = { 4.6987127, 0.042934613, 3.4189395, 3.9041192, 4.5004177, 3.0824886, 0.90713743, 4.9708287, 5.4551861, 3.394801, 1.8496327 } },
    decode_offset = { def = 0.18440428 },
    search_trials = 0,
    seed_ensemble = 1,
    scratch_path = "test/res/conll-type-scratch",
    folds = 5,
  },
}

local function build_segments (split)
  local S = util.shape_spans(split.texts, split.n)
  return { n = split.n, seg = S, n_seg = S:col("s"):size(),
    inner_mask = S:contained_labels(split.gold), gold = split.gold }
end

local function inner_to_ctx (pred)
  local n = pred:size()
  local out = ivec.create(n)
  for i = 0, n - 1 do
    out:set(i, pred:get(i) == 1 and 0 or 1)
  end
  return out
end

local function seg_ceiling (S, G, n)
  local soff, sstart, send = S:offsets(), S:col("s"), S:col("e")
  local eoff, es, ee = G:offsets(), G:col("s"), G:col("e")
  local ok, tot = 0, 0
  for d = 1, n do
    local ss, se = {}, {}
    for si = soff:get(d - 1), soff:get(d) - 1 do ss[sstart:get(si)] = true; se[send:get(si)] = true end
    for gi = eoff:get(d - 1), eoff:get(d) - 1 do
      tot = tot + 1
      if ss[es:get(gi)] and se[ee:get(gi)] then ok = ok + 1 end
    end
  end
  return tot > 0 and ok / tot or 0
end

test("conll-full", function ()

  local stopwatch = utc.stopwatch()
  local function sw () local d, dd = stopwatch(); return str.format("(%.1fs +%.1fs)", d, dd) end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  local n_train, n_dev = train.n, dev.n
  train = ds.merge_conll2003(train, dev)
  str.printf("[Data] pool=%d (train=%d + dev=%d) test=%d %s\n", train.n, n_train, n_dev, test_set.n, sw())

  local TR, TE = build_segments(train), build_segments(test_set)
  str.printf("[SegCeil] boundary-aligned gold (oracle recall): test=%.4f | n_seg train=%d test=%d %s\n",
    seg_ceiling(TE.seg, TE.gold, TE.n), TR.n_seg, TE.n_seg, sw())

  local function zeros (n) local z = ivec.create(n); z:zero(); return z end

  local tr_inner, te_inner
  do
    local toks, Xtr = util.tokenize_blocks(cfg.tag.blocks, train.texts,
      { focus = TR.seg, tokens = TR.seg })
    local Ytr = csr.from_mask(TR.inner_mask)
    str.printf("[Tag] Encoding\n")
    local _, rg, deploy, best, decider = optimize.krr(util.merged(cfg.tag, {
      pool_blocks = Xtr,
      pool_labels = Ytr,
      pool_n = TR.n_seg,
      n_labels = 1,
      n_landmarks = cfg.emb.n_landmarks,
      search_landmarks = cfg.search_landmarks,
      k = 1,
      verbose = cfg.verbose,
      each = util.make_ridge_log(stopwatch),
    }))
    str.printf("[Tag] kernel=%s lambda=%.8g offset=%.8g %s\n",
      best.kernel, best.lambda, decider:offset(), sw())
    local function tag_decode (split, B)
      local _, X = util.tokenize_blocks(cfg.tag.blocks, split.texts,
        { toks = toks, focus = B.seg, tokens = B.seg })
      local P = util.predict_tiled({ deploy = deploy, ridge = rg, blocks = X, n = B.n_seg, k = 1 })
      return decider:predict({ pred = P, n_samples = B.n_seg })
    end
    tr_inner = tag_decode(train, TR)
    te_inner = tag_decode(test_set, TE)
  end

  local function ctx_spans (S, pred)
    return spans.create({ offsets = S:offsets(), s = S:col("s"), e = S:col("e"),
      ty = inner_to_ctx(pred) })
  end

  local Sctx_tr = ctx_spans(TR.seg, tr_inner)
  local Sctx_te = ctx_spans(TE.seg, te_inner)
  local Sruns_tr = Sctx_tr:enumerate_subspans(MAX_SPAN, 1)
  local Sruns_te = Sctx_te:enumerate_subspans(MAX_SPAN, 1)

  local ac = util.surface_gaz({ train }, N_TYPES, true)
  local Sg_tr = ac:predict({ texts = train.texts, longest = true, tokens = TR.seg })
  local Sg_te = ac:predict({ texts = test_set.texts, longest = true, tokens = TE.seg })

  local function cand_union (Sruns, Sg)
    local A = spans.create({ offsets = Sruns:offsets(), s = Sruns:col("s"), e = Sruns:col("e"),
      ty = zeros(Sruns:col("s"):size()) })
    local B = spans.create({ offsets = Sg:offsets(), s = Sg:col("s"), e = Sg:col("e"),
      ty = zeros(Sg:col("s"):size()) })
    return A:union(B)
  end
  local Scand_tr = cand_union(Sruns_tr, Sg_tr)
  local Scand_te = cand_union(Sruns_te, Sg_te)
  local n_trc, n_tec = Scand_tr:col("s"):size(), Scand_te:col("s"):size()

  local tr_tlab = Scand_tr:type_labels(TR.gold, N_TYPES)
  local tr_tloff = ivec.create(n_trc + 1); tr_tloff:fill_indices()

  local te_gold = TE.gold:col("s"):size()
  local n_te_docs = TE.n
  local Sgaz_empty = spans.create({ offsets = zeros(n_te_docs + 1),
    s = ivec.create(0), e = ivec.create(0), ty = ivec.create(0) })
  local miss = ner.miss_report({ gaz = Sgaz_empty, bio = Scand_te, gold = TE.gold, n_types = N_TYPES })
  local ubt = miss.under_by_type
  str.printf("[Cands] train=%d test=%d | test span-coverage=%.4f"
    .. " | miss over=%d under=%d cross=%d none=%d"
    .. " | under by-type[PER=%d ORG=%d LOC=%d MISC=%d] %s\n",
    n_trc, n_tec, (miss.covered + miss.wrong_type) / te_gold,
    miss.over, miss.under, miss.cross, miss.none,
    ubt[0], ubt[1], ubt[2], ubt[3], sw())

  local Ytype = csr.create({ offsets = tr_tloff, neighbors = tr_tlab, n_cols = N_TYPES + 1 })
  local n_sparse = #cfg.type.blocks
  local ty_toks
  local function type_blocks (split, Scand, B)
    local toks, X = util.tokenize_blocks(cfg.type.blocks, split.texts,
      { toks = ty_toks, focus = Scand, tokens = B.seg })
    ty_toks = toks
    return X
  end
  local ty_all_tr = type_blocks(train, Scand_tr, TR)
  local ty_all_te = type_blocks(test_set, Scand_te, TE)

  local K = cfg.type.folds
  local df = util.doc_folds(Scand_tr, TR.gold, K)
  local function build_cgaz (g)
    return ner.build_char_gaz({ texts = train.texts, gold = g, n_types = N_TYPES,
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max })
  end
  local ty_serve_gaz = build_cgaz(TR.gold)
  ty_all_tr[#ty_all_tr + 1] = ty_serve_gaz:block(train.texts, Scand_tr, tr_tlab)
  ty_all_te[#ty_all_te + 1] = ty_serve_gaz:block(test_set.texts, Scand_te, nil)

  local ty_rms_w = util.rms_scale_blocks(ty_all_tr, { ty_all_te }, n_sparse + 1)
  local ty_relevance = {}
  for i = 1, n_sparse do ty_relevance[i] = cfg.type.relevance[i] end
  ty_relevance[n_sparse + 1] = "auc"
  str.printf("[Type] CV folds=%d trials=%d (full-gold + LOO gaz)\n", K, cfg.type.search_trials)
  local ty_enc, ridge_ty, deploy, _, ty_decider = optimize.krr(util.merged(cfg.type, {
    pool_blocks = ty_all_tr, pool_labels = Ytype, pool_n = n_trc,
    folds = K, doc_fold = df, cand = Scand_tr, gold = TR.gold,
    relevance = ty_relevance,
    n_labels = N_TYPES + 1,
    n_landmarks = cfg.emb.n_landmarks, search_landmarks = cfg.search_landmarks, k = 1,
    verbose = cfg.verbose, each = util.make_ridge_log(stopwatch),
  }))
  collectgarbage("collect")

  local te_lab_csr, te_scores = util.predict_tiled({ deploy = deploy, ridge = ridge_ty,
    blocks = ty_all_te, n = n_tec, k = 2, scores = true, n_labels = N_TYPES + 1 })
  local te_lab = te_lab_csr:neighbors()
  local _, te_m = ty_decider:score({ scores = te_scores, n_samples = n_te_docs, cand = Scand_te, gold = TE.gold })
  str.printf("[Span] test %s offset=%.8g %s\n", util.fmt_metrics(te_m), ty_decider:offset(), sw())

  local bundle = require("santoku.learn.bundle")
  local bdir = os.tmpname() .. ".bundle"
  bundle.persist({ dir = bdir, tokenizers = ty_toks, gaz = ty_serve_gaz, gaz_rms = ty_rms_w[n_sparse + 1],
    encoder = ty_enc, ridge = ridge_ty, decider = ty_decider })
  local dep = util.fmt_metrics(te_m)
  ty_enc, ridge_ty, deploy, ty_decider, ty_toks, ty_serve_gaz, te_scores, ty_all_tr, ty_all_te, ty_rms_w = nil -- luacheck: ignore
  collectgarbage("collect")
  local b = bundle.load(bdir)
  local _, Xb = util.tokenize_blocks(cfg.type.blocks, test_set.texts, { toks = b.tokenizers, focus = Scand_te, tokens = TE.seg })
  Xb[n_sparse + 1] = b.gaz:block(test_set.texts, Scand_te, nil)
  Xb[n_sparse + 1]:bns(b.gaz_rms)
  local _, sb = util.predict_tiled({ deploy = b.encode, ridge = b.ridge,
    blocks = Xb, n = n_tec, k = 2, scores = true, n_labels = N_TYPES + 1 })
  local _, mb = b.decider:score({ scores = sb, n_samples = n_te_docs, cand = Scand_te, gold = TE.gold })
  str.printf("[Bundle] reload test %s (deploy %s)\n", util.fmt_metrics(mb), dep)
  assert(util.fmt_metrics(mb) == dep, "reloaded bundle metrics diverge from deploy")
  util.rmbundle(bdir)

  local tdr = ner.decode_report({
    cand = Scand_te, pred = te_lab, pred_stride = 2,
    gold = TE.gold, n_types = N_TYPES,
  })
  local rbt, mbt, cf = tdr.reject_by_type, tdr.mistype_by_type, tdr.confusion
  str.printf("[Type Decode] gold=%d in_pool=%d not_in_pool=%d | correct=%d false_reject=%d mistype=%d\n",
    tdr.gold, tdr.in_pool, tdr.not_in_pool, tdr.correct, tdr.false_reject, tdr.mistype)
  str.printf("[Type Decode] false_reject by-type[PER=%d ORG=%d LOC=%d MISC=%d]"
    .. " mistype by-type[PER=%d ORG=%d LOC=%d MISC=%d]\n",
    rbt[0], rbt[1], rbt[2], rbt[3], mbt[0], mbt[1], mbt[2], mbt[3])
  local TN = { [0] = "PER", [1] = "ORG", [2] = "LOC", [3] = "MISC" }
  for t = 0, N_TYPES - 1 do
    str.printf("[Type Decode] mistype %s -> [PER=%d ORG=%d LOC=%d MISC=%d]\n",
      TN[t], cf[t * N_TYPES + 0], cf[t * N_TYPES + 1], cf[t * N_TYPES + 2], cf[t * N_TYPES + 3])
  end

  local _, total = stopwatch()
  str.printf("Total: %.1fs\n", total)

end)
