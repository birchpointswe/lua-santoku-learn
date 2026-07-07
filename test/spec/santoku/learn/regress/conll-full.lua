local ds = require("santoku.learn.dataset")
local csr = require("santoku.csr")
local spans = require("santoku.spans")
local ner = require("santoku.learn.ner")
local tokenizer = require("santoku.learn.tokenizer")
local aho = require("santoku.learn.aho")
local segmenter = require("santoku.learn.segmenter")
local optimize = require("santoku.learn.optimize")
local ivec = require("santoku.ivec")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  verbose = false,
  search_landmarks = 2048,
  data = {
    dir = "test/res/conll2003",
    max = nil
  },
  tok = { ngram_min = 1, ngram_max = 5, normalize = false },
  emb = { n_landmarks = 1024 * 8, },
  tag = {
    kernel = { "matern" },
    nu = { def = 2 },
    gamma = { def = 0.0324322 },
    lambda = { def = 0.000676847 },
    blocks = {
      { ngram_min = 1, ngram_max = 5, normalize = false },
      { ngram_min = 1, ngram_max = 3, words = true, word_characters = util.WORD_CHARACTERS, normalize = false },
    },
    relevance = { "bns", "bns" },
    scales = { def = { 0.492386, 2.03093 } },
    exponent = { def = { 3.48892, 4.10881 } },
    decode_offset = { def = 0.489454 },
    search_trials = 0,
    folds = 5,
  },
  type = {
    kernel = { "arccos" },
    order = { def = 2 },
    depth = { def = 1 },
    tangent = { def = 0 },
    lambda = { def = 5.36224e-06 },
    scales = { def = { 13.4525, 7.14652, 0.0434604, 0.0281398, 0.0653057, 2.03616, 4.33692, 4.58321, 3.21792 } },
    exponent = { def = { 5.78839, 4.49708, 0.071323, 3.7165, 7.86543, 2.85348, 3.14215, 0.39562, 0.000953421 } },
    decode_offset = { def = -0.00797415 },
    search_trials = 0,
    folds = 5,
  },
  shape = {
    n_cuts = 5
  },
}

local N_TYPES = 4
local MAX_SPAN = 1000000000
local word_characters = util.WORD_CHARACTERS

local function merge_splits (a, b)
  local m = { n = a.n + b.n, texts = {}, sent_tokens = {}, sent_ents = {},
    n_pos = a.n_pos, pos_names = a.pos_names, n_types = a.n_types, type_names = a.type_names }
  for i = 1, a.n do m.texts[i] = a.texts[i]; m.sent_tokens[i] = a.sent_tokens[i]; m.sent_ents[i] = a.sent_ents[i] end
  for i = 1, b.n do local j = a.n + i
    m.texts[j] = b.texts[i]; m.sent_tokens[j] = b.sent_tokens[i]; m.sent_ents[j] = b.sent_ents[i] end
  return m
end

local function build_gazetteer (train)
  local seen, patterns = {}, {}
  for d = 1, train.n do
    local text = train.texts[d]
    for _, e in ipairs(train.sent_ents[d]) do
      local surf = text:sub(e.s + 1, e.e)
      if not seen[surf] then seen[surf] = true; patterns[#patterns + 1] = surf end
    end
  end
  return aho.create({ patterns = patterns, normalize = true })
end

local function gaz_candidates (ac, split)
  return ac:predict({ texts = split.texts, longest = true, word_characters = word_characters })
end

local function gold_spans (split)
  local eoff, es, ee, ety = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  eoff:push(0)
  for d = 1, split.n do
    for _, e in ipairs(split.sent_ents[d]) do es:push(e.s); ee:push(e.e); ety:push(e.t) end
    eoff:push(es:size())
  end
  return spans.create({ offsets = eoff, s = es, e = ee, ty = ety })
end

local function is_inner_labels (S, G, n)
  local soff, sstart, send = S:offsets(), S:col("s"), S:col("e")
  local eoff, es, ee = G:offsets(), G:col("s"), G:col("e")
  local off, nbr = ivec.create(), ivec.create()
  off:push(0)
  for d = 1, n do
    for si = soff:get(d - 1), soff:get(d) - 1 do
      local ss, se = sstart:get(si), send:get(si)
      local inner = false
      for gi = eoff:get(d - 1), eoff:get(d) - 1 do
        if ss >= es:get(gi) and se <= ee:get(gi) then inner = true; break end
      end
      if inner then nbr:push(0) end
      off:push(nbr:size())
    end
  end
  return off, nbr
end

local function build_segments (split, seg)
  local soff, sstart, send = seg:segment({ texts = split.texts, n = split.n, drop_sep = true })
  local S = spans.create({ offsets = soff, s = sstart, e = send })
  local G = gold_spans(split)
  local ioff, inbr = is_inner_labels(S, G, split.n)
  return { n = split.n, seg = S, n_seg = sstart:size(),
    inner_off = ioff, inner_nbr = inbr, gold = G }
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

local function tokenize (split, F, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      terminals = true, focus = true, normalize = cfg.tok.normalize,
    })
  end
  local X = grow and tok:fit({ texts = split.texts, focus = F })
    or tok:tokenize({ texts = split.texts, focus = F })
  return tok, X
end

local function tokenize_ctx (split, F, T, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = 1, terminals = true, focus = true, types = true,
    })
  end
  local X = grow and tok:fit({ texts = split.texts, focus = F, types = T })
    or tok:tokenize({ texts = split.texts, focus = F, types = T })
  return tok, X
end

local function tokenize_shape (split, F, T, k, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = k + 1, terminals = true, focus = true, types = true,
    })
  end
  local X = grow and tok:fit({ texts = split.texts, focus = F, types = T })
    or tok:tokenize({ texts = split.texts, focus = F, types = T })
  return tok, X
end

test("conll-full", function ()

  local stopwatch = utc.stopwatch()
  local function sw () local d, dd = stopwatch(); return str.format("(%.1fs +%.1fs)", d, dd) end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  local n_train, n_dev = train.n, dev.n
  train = merge_splits(train, dev)
  str.printf("[Data] pool=%d (train=%d + dev=%d) test=%d %s\n", train.n, n_train, n_dev, test_set.n, sw())

  local seg = segmenter.create({ context = "left" })
  local coarse_k
  do
    local G = gold_spans(train)
    local ck, rec, mx, p95, sep = seg:train({ texts = train.texts, n = train.n,
      gold_offsets = G:offsets(), gold_starts = G:col("s"), gold_ends = G:col("e") })
    coarse_k = ck
    str.printf("[Seg] coarse_k=%d boundary-recall=%.4f segs/gold(max=%d p95=%d) sep=%d %s\n",
      ck, rec, mx, p95, sep, sw())
  end

  do
    local curve, nseen = seg:compression_curve({ texts = train.texts, n = train.n })
    cfg.shape.ks = optimize.plateaus(curve, cfg.shape.n_cuts, coarse_k + 1)
    local parts = {}
    for k = 1, nseen do parts[#parts + 1] = str.format("%d=%.2f", k, curve[k]) end
    str.printf("[Comp] train avg bytes/cells by k(clusters): %s\n", table.concat(parts, " "))
    str.printf("[Comp] jenks SHAPE cuts (n=%d): %s %s\n",
      cfg.shape.n_cuts, table.concat(cfg.shape.ks, ","), sw())
  end

  local TR, TE = build_segments(train, seg), build_segments(test_set, seg)
  str.printf("[SegCeil] boundary-aligned gold (oracle recall): test=%.4f %s\n",
    seg_ceiling(TE.seg, TE.gold, TE.n), sw())

  local function shape_runs (split)
    local out = {}
    for i, k in ipairs(cfg.shape.ks) do
      local off, st, en, cl = seg:segment({ texts = split.texts, n = split.n, k = k })
      out[i] = spans.create({ offsets = off, s = st, e = en, ty = cl })
    end
    return out
  end

  TR.sh, TE.sh = shape_runs(train), shape_runs(test_set)

  local function zeros (n) local z = ivec.create(n); z:zero(); return z end

  local tr_inner, te_inner
  do
    local toks, Xtr = util.tokenize_focus_blocks(cfg.tag.blocks, train.texts, TR.seg)
    local Ytr = csr.create({ offsets = TR.inner_off, neighbors = TR.inner_nbr, n_cols = 1 })
    str.printf("[Tag] Encoding\n")
    local _, rg, deploy, best, decider = optimize.krr({
      pool_blocks = Xtr,
      pool_labels = Ytr,
      pool_n = TR.n_seg,
      n_labels = 1,
      folds = cfg.tag.folds,
      relevance = cfg.tag.relevance,
      scales = cfg.tag.scales,
      exponent = cfg.tag.exponent,
      kernel = cfg.tag.kernel,
      nu = cfg.tag.nu,
      gamma = cfg.tag.gamma,
      lambda = cfg.tag.lambda,
      n_landmarks = cfg.emb.n_landmarks,
      search_landmarks = cfg.search_landmarks,
      k = 1,
      search_trials = cfg.tag.search_trials,
      cv_buf_path = cfg.data.dir .. "/cv_tag_",
      decode_offset = cfg.tag.decode_offset,
      verbose = cfg.verbose,
      each = util.make_ridge_log(stopwatch),
    })
    str.printf("[Tag] kernel=%s lambda=%.4e offset=%.6g %s\n",
      best.kernel, best.lambda, decider:offset(), sw())
    local function tag_decode (split, seg, n)
      local _, X = util.tokenize_focus_blocks(cfg.tag.blocks, split.texts, seg, toks)
      return decider:predict({ pred = rg:label(deploy(X), 1), n_samples = n })
    end
    tr_inner = tag_decode(train, TR.seg, TR.n_seg)
    te_inner = tag_decode(test_set, TE.seg, TE.n_seg)
  end

  local function ctx_spans (S, pred)
    return spans.create({ offsets = S:offsets(), s = S:col("s"), e = S:col("e"),
      ty = inner_to_ctx(pred) })
  end

  local Sctx_tr = ctx_spans(TR.seg, tr_inner)
  local Sctx_te = ctx_spans(TE.seg, te_inner)
  local Sruns_tr = Sctx_tr:enumerate_subspans(MAX_SPAN, 1)
  local Sruns_te = Sctx_te:enumerate_subspans(MAX_SPAN, 1)

  local ac = build_gazetteer(train)
  local function cand_union (split, Sruns, G)
    local Sg = gaz_candidates(ac, split)
    local A = spans.create({ offsets = Sruns:offsets(), s = Sruns:col("s"), e = Sruns:col("e"),
      ty = zeros(Sruns:col("s"):size()) })
    local B = spans.create({ offsets = Sg:offsets(), s = Sg:col("s"), e = Sg:col("e"),
      ty = zeros(Sg:col("s"):size()) })
    local Gz = spans.create({ offsets = G:offsets(), s = G:col("s"), e = G:col("e"),
      ty = zeros(G:col("s"):size()) })
    return A:union(B, Gz)
  end
  local Scand_tr = cand_union(train, Sruns_tr, TR.gold)
  local Scand_te = cand_union(test_set, Sruns_te, TE.gold)
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

  local ty = { models = {},
    tok = tokenize, tok_ctx = tokenize_ctx, tok_shape = tokenize_shape, shape_ks = cfg.shape.ks }
  local gaz = ner.build_typed_gaz({ texts = train.texts, gold = TR.gold, n_types = N_TYPES })
  local cgaz = ner.build_char_gaz({ texts = train.texts, gold = TR.gold, n_types = N_TYPES,
    ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max })
  local Ytype = csr.create({ offsets = tr_tloff, neighbors = tr_tlab, n_cols = N_TYPES + 1 })
  local ty_all_tr, n_sparse = util.build_type_blocks(ty, train, Scand_tr, Sctx_tr, TR.sh, true, gaz, cgaz, tr_tlab)
  local ty_all_te = util.build_type_blocks(ty, test_set, Scand_te, Sctx_te, TE.sh, false, gaz, cgaz, nil)
  local n_ty_blocks = #ty_all_tr
  local te_gaz_r = { [n_sparse + 1] = gaz:block(test_set.texts, Scand_te, nil),
    [n_sparse + 2] = cgaz:block(test_set.texts, Scand_te, nil) }
  util.rms_scale_blocks(ty_all_tr, { ty_all_te, te_gaz_r }, n_sparse + 1)
  local edef = (cfg.type.exponent or {}).def
  local ty_relevance, ty_exponent_spec = {}, {}
  for b = 1, n_ty_blocks do
    ty_relevance[b] = (b > n_sparse) and "auc" or "bns"
    local d = edef
    if type(edef) == "table" then d = edef[b] end
    ty_exponent_spec[b] = { def = (d ~= nil and d ~= false) and d or 1 }
  end
  local sty = cfg.type.scales or {}
  local sdef = sty.def
  local ty_scales_spec = {}
  for j = 1, n_ty_blocks do
    local d = sdef
    if type(sdef) == "table" then d = sdef[j] end
    if d == nil or d == false then
      ty_scales_spec[j] = false
    else
      ty_scales_spec[j] = { def = (type(d) == "number") and d or 1 }
    end
  end
  local K = cfg.type.folds
  str.printf("[Type] CV folds=%d trials=%d\n", K, cfg.type.search_trials)
  local cv_buf = cfg.data.dir .. "/cv_type_"
  local _, ridge_ty, deploy, _, ty_decider = optimize.krr({
    pool_blocks = ty_all_tr, pool_labels = Ytype, pool_n = n_trc,
    folds = K, cand = Scand_tr, gold = TR.gold,
    relevance = ty_relevance, scales = ty_scales_spec,
    exponent = ty_exponent_spec,
    n_labels = N_TYPES + 1,
    kernel = cfg.type.kernel, nu = cfg.type.nu, gamma = cfg.type.gamma,
    order = cfg.type.order, depth = cfg.type.depth, tangent = cfg.type.tangent,
    lambda = cfg.type.lambda,
    n_landmarks = cfg.emb.n_landmarks, search_landmarks = cfg.search_landmarks, k = 1, cv_buf_path = cv_buf, decode_offset = cfg.type.decode_offset,
    search_trials = cfg.type.search_trials, verbose = cfg.verbose, each = util.make_ridge_log(stopwatch),
  })
  collectgarbage("collect")

  local te_codes = deploy(ty_all_te)
  local te_scores = ridge_ty:regress(te_codes)
  local te_lab = ridge_ty:label(te_codes, 2):neighbors()
  local _, te_m = ty_decider:score({ scores = te_scores, n_samples = n_te_docs, cand = Scand_te, gold = TE.gold })
  str.printf("[Span] test %s offset=%.6g %s\n", util.fmt_metrics(te_m), ty_decider:offset(), sw())

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
  str.printf("\nTotal: %.1fs\n", total)

end)
