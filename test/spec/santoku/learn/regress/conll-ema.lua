local ds = require("santoku.learn.dataset")
local csr = require("santoku.learn.csr")
local aho = require("santoku.learn.aho")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local ridge = require("santoku.learn.ridge")
local eval = require("santoku.learn.evaluator")
local ivec = require("santoku.ivec")
local fvec = require("santoku.fvec")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

-- Span NER as two stages: TAG -> (enumerate) -> TYPE. EMA segment-pool variant.
--  1. TAG  (per-token type): char n-grams pooled into 4 PCNN segments (lead/inner_fwd/inner_bwd/trail)
--     around each token, fed to a krr (N_TYPES+1)-class head. Emits SOFT per-token scores, OOF on train.
--  enumerate: every 1..max_span subspan within each non-O run (argmax of the soft scores).
--  2. TYPE (per-span type-or-reject): the per-token soft scores pooled into the same 4 segments around
--     each candidate span (dense), fed to a krr head; argmax decode, DP overlap resolution.

local cfg = {
  data = { dir = "test/res/conll2003", max = nil },
  tok = { ngram_min = 3, ngram_max = 5, normalize = false },
  emb = { n_landmarks = 1024 * 8, trace_tol = 0.01 },
  tag = { kernel = { "cosine", "expcos", "geolaplace", "matern52", "rq", "arccos1" },
    lambda = { min = 1e-4, max = 1e1, log = true, def = 1.1980e-02 },
    propensity_a = { min = 0, max = 8, def = 0.2607 }, propensity_b = { min = 0, max = 16, def = 8.6397 },
    search_trials = 0, max_span = 5, alpha = 0.5 },          -- alpha decays per byte
  type = { kernel = { "arccos1", "cosine", "expcos", "geolaplace", "matern52", "rq" },
    lambda = { min = 1e-4, max = 1e1, log = true, def = 6.8798e-04 },
    propensity_a = { min = 0, max = 8, def = 0.4179 }, propensity_b = { min = 0, max = 16, def = 5.2162 },
    search_trials = 200, alpha = 0.6, blocks = "char" },     -- alpha per token; blocks = dense|char|both
  stack = { k = 5 },
}

local N_TYPES = 4
local P = N_TYPES + 1
local boundary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
local SEGS = { "lead", "inner_fwd", "inner_bwd", "trail" }
local NSEG = 4

-- gazetteer = aho automaton over the distinct train entity surface forms; an extra (untyped) candidate
-- source for surfaces the tagger tags all-O. Type is left to the TYPE stage.
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
  local off, _, starts, ends = ac:predict({ texts = split.texts, longest = true, boundary = boundary })
  return off, starts, ends
end

-- per-token spans + gold entity spans; BIO x type labels via csr.bio_encode
local function build_tokens (split)
  local off, starts, ends, types = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  local eoff, es, ee, ety = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  off:push(0); eoff:push(0)
  for d = 1, split.n do
    for _, t in ipairs(split.sent_tokens[d]) do starts:push(t.s); ends:push(t.e); types:push(0) end
    off:push(starts:size())
    for _, e in ipairs(split.sent_ents[d]) do es:push(e.s); ee:push(e.e); ety:push(e.t) end
    eoff:push(es:size())
  end
  local lab_off, lab_nbr = csr.bio_encode(off, starts, ends, eoff, es, ee, ety, N_TYPES)
  return off, starts, ends, types, lab_off, lab_nbr, starts:size(), eoff, es, ee, ety
end

local function build_fold (off, n, k)
  local fold = ivec.create(n)
  for d = 0, off:size() - 2 do fold:fill(d % k, off:get(d), off:get(d + 1)) end
  return fold
end

-- ===== EMA pooling helpers =====

-- sequence-tokenize a split's docs into a positioned char-gram stream (one vocab)
local function seq_tokenize (split, ngram_map)
  return csr.tokenize({
    texts = split.texts, n_samples = split.n,
    ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
    normalize = cfg.tok.normalize, sequence = true, ngram_map = ngram_map,
  })  -- ngram_map, s_off, s_tok, s_pos, V
end

-- 4 sparse char-pool blocks; focus = the given spans (V-wide each)
local function pool_sparse (s_off, s_tok, s_pos, V, foc_off, foc_s, foc_e, alpha)
  local blocks = {}
  for i = 1, NSEG do
    local o, t, v = csr.segment_pool({
      segment = SEGS[i], alpha = alpha,
      doc_span_offsets = foc_off, span_starts = foc_s, span_ends = foc_e,
      stream_offsets = s_off, stream_tokens = s_tok, stream_positions = s_pos, n_stream = V })
    blocks[i] = { o, t, v }
  end
  return blocks
end

-- 4 dense soft-pool blocks; focus = candidate spans, stream = per-token feature rows (d-wide each)
local function pool_dense (feats, d, tok_off, tok_s, tok_e, foc_off, foc_s, foc_e, alpha)
  local blocks = {}
  for i = 1, NSEG do
    local o, t, v = csr.segment_pool({
      segment = SEGS[i], alpha = alpha, features = feats, d = d,
      token_offsets = tok_off, token_starts = tok_s, token_ends = tok_e,
      doc_span_offsets = foc_off, span_starts = foc_s, span_ends = foc_e })
    blocks[i] = { o, t, v }
  end
  return blocks
end

local function gather_blocks (blocks, idx)
  local out = {}
  for i = 1, #blocks do
    local o, t, v = csr.gather_rows(blocks[i][1], blocks[i][2], blocks[i][3], idx)
    out[i] = { o, t, v }
  end
  return out
end

-- merge + (optional BNS) + per-block scale + normalize, IN PLACE (consumes blocks once).
-- learn=true learns bns[]/block into `learned`; learn=false applies them.
local function assemble (blocks, U, n_rows, do_bns, learn, learned, label_off, label_nbr, n_labels)
  local nb = #blocks
  if do_bns then
    if learn then
      learned.bns = {}
      for i = 1, nb do
        learned.bns[i] = csr.apply_bns(blocks[i][1], blocks[i][2], blocks[i][3], nil,
          label_off, label_nbr, U, n_labels)
      end
    else
      for i = 1, nb do
        csr.apply_bns(blocks[i][1], blocks[i][2], blocks[i][3], learned.bns[i])
      end
    end
  end
  local o, t, v = blocks[1][1], blocks[1][2], blocks[1][3]
  for i = 2, nb do
    o, t, v = csr.merge(o, t, v, blocks[i][1], blocks[i][2], blocks[i][3], (i - 1) * U)
  end
  local ntok = nb * U
  if learn then
    local bounds = {}
    for i = 0, nb do bounds[i + 1] = i * U end
    local ss = csr.block_sumsq(t, v, bounds)
    local bs = fvec.create(ntok)
    for i = 0, nb - 1 do
      local s = ss:get(i)
      bs:fill(s > 0 and math.sqrt(n_rows / s) or 0.0, i * U, (i + 1) * U)
    end
    learned.block = bs
  end
  csr.standardize(o, t, v, learned.block)
  csr.normalize(o, v)
  return o, t, v, ntok
end

-- row-wise argmax of an n x np score matrix -> ivec of class ids (O = N_TYPES)
local function argmax_rows (scores, n, np)
  local out = ivec.create(n)
  for i = 0, n - 1 do
    local best, bi = -math.huge, 0
    for c = 0, np - 1 do
      local s = scores:get(i * np + c)
      if s > best then best = s; bi = c end
    end
    out:set(i, bi)
  end
  return out
end

test("conll-ema", function ()

  local stopwatch = utc.stopwatch()
  local function sw () local d, dd = stopwatch(); return str.format("(%.1fs +%.1fs)", d, dd) end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  str.printf("[Data] train=%d dev=%d test=%d %s\n", train.n, dev.n, test_set.n, sw())

  -- ===== token + gold spans =====
  local tr_off, tr_s, tr_e, _, _, tr_lnbr, tr_n, tr_eoff, tr_es, tr_ee, tr_ety = build_tokens(train)
  local dv_off, dv_s, dv_e, _, _, dv_lnbr, dv_n, dv_eoff, dv_es, dv_ee, dv_ety = build_tokens(dev)
  local te_off, te_s, te_e, _, _, _, te_n, te_eoff, te_es, te_ee, te_ety = build_tokens(test_set)

  -- ===== Stage 1: TAG (per-token type; soft scores, OOF on train) =====
  local tr_types, dv_types, te_types
  local tr_scores, dv_scores, te_scores
  local s_off, s_tok, s_pos, V, ds_off, ds_tok, ds_pos, es_off, es_tok, es_pos
  do
    local fold = build_fold(tr_off, tr_n, cfg.stack.k)
    local tg_tr = csr.bio_token_type(tr_lnbr, N_TYPES)   -- per-token class id: type 0..N_TYPES-1, O=N_TYPES
    local tg_dv = csr.bio_token_type(dv_lnbr, N_TYPES)
    local tr_tgoff = ivec.create(tr_n + 1); tr_tgoff:fill_indices()
    local dv_tgoff = ivec.create(dv_n + 1); dv_tgoff:fill_indices()

    str.printf("[Tag] Pooling + encoding\n")
    local ng, _
    ng, s_off, s_tok, s_pos, V = seq_tokenize(train, nil)
    _, ds_off, ds_tok, ds_pos = seq_tokenize(dev, ng)
    _, es_off, es_tok, es_pos = seq_tokenize(test_set, ng)

    local tr_blocks = pool_sparse(s_off, s_tok, s_pos, V, tr_off, tr_s, tr_e, cfg.tag.alpha)
    local learned = {}
    -- full-train model assembles from a COPY (gather all) so the in-place BNS doesn't mutate tr_blocks,
    -- which the OOF below re-gathers per fold (must stay pristine for leak-safe refit).
    local all = ivec.create(tr_n); all:fill_indices()
    local o, t, v, ntok = assemble(gather_blocks(tr_blocks, all), V, tr_n, true, true, learned, tr_tgoff, tg_tr, P)

    local dv_blocks = pool_sparse(ds_off, ds_tok, ds_pos, V, dv_off, dv_s, dv_e, cfg.tag.alpha)
    local dvo, dvt, dvv = assemble(dv_blocks, V, dv_n, true, false, learned)

    local sp, rg, _, best = optimize.krr({
      offsets = o, tokens = t, values = v, n_samples = tr_n, n_tokens = ntok,
      kernel = cfg.tag.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
      label_offsets = tr_tgoff, label_neighbors = tg_tr, n_labels = P,
      val_offsets = dvo, val_tokens = dvt, val_values = dvv, val_n_samples = dv_n,
      val_expected_offsets = dv_tgoff, val_expected_neighbors = tg_dv,
      lambda = cfg.tag.lambda, propensity_a = cfg.tag.propensity_a, propensity_b = cfg.tag.propensity_b,
      k = 1, search_trials = cfg.tag.search_trials,
      each = util.make_ridge_log(stopwatch),
      trial_fn = function (gram, params)
        local f1, p, r = gram:label_accuracy(params.lambda, 1,
          params.propensity_a, params.propensity_b, dv_tgoff, tg_dv, 1)
        return f1, { f1 = f1, precision = p, recall = r }
      end,
    })
    str.printf("[Tag] kernel=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
      best.kernel, best.lambda, best.propensity_a, best.propensity_b, sw())

    -- soft decode dev/test (full model)
    local function decode (lo, lt, lv, n)
      local codes = sp:encode({ offsets = lo, tokens = lt, values = lv, n_samples = n })
      return rg:regress(codes, n)  -- n x P
    end
    dv_scores = decode(dvo, dvt, dvv, dv_n)
    local te_blocks = pool_sparse(es_off, es_tok, es_pos, V, te_off, te_s, te_e, cfg.tag.alpha)
    local teo, tet, tev = assemble(te_blocks, V, te_n, true, false, learned)
    te_scores = decode(teo, tet, tev, te_n)

    -- OOF soft scores on train (leak-safe; pooling is per-doc, label-free)
    tr_scores = optimize.oof({
      n = tr_n, k = cfg.stack.k, fold = fold, width = P, out = fvec.create(tr_n * P),
      each = function (ev) str.printf("[Tag OOF] fold %d/%d %s\n", ev.fold, ev.folds, sw()) end,
      fit = function (idx)
        local gb = gather_blocks(tr_blocks, idx)
        local lnbr = ivec.create(); lnbr:copy(tg_tr, idx)
        local loff = ivec.create(idx:size() + 1); loff:fill_indices()
        local ll = {}
        local fo, ft, fv, fntok = assemble(gb, V, idx:size(), true, true, ll, loff, lnbr, P)
        local _, spf, gram = spectral.encode({
          offsets = fo, tokens = ft, values = fv, n_tokens = fntok, n_samples = idx:size(),
          kernel = best.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
          label_offsets = loff, label_neighbors = lnbr, n_labels = P,
          solve_lambda = best.lambda,
          solve_propensity_a = best.propensity_a, solve_propensity_b = best.propensity_b,
        })
        return { sp = spf, r = ridge.create({ gram = gram }), learned = ll }
      end,
      predict = function (h, idx)
        local gb = gather_blocks(tr_blocks, idx)
        local lo, lt, lv = assemble(gb, V, idx:size(), true, false, h.learned)
        local codes = h.sp:encode({ offsets = lo, tokens = lt, values = lv, n_samples = idx:size() })
        return h.r:regress(codes, idx:size())
      end,
    })

    tr_types = argmax_rows(tr_scores, tr_n, P)
    dv_types = argmax_rows(dv_scores, dv_n, P)
    te_types = argmax_rows(te_scores, te_n, P)
  end

  -- enumerate candidate spans: every 1..max_span subspan within each contiguous non-O run
  local en_tro, en_trs, en_tre = csr.enumerate_subspans(tr_off, tr_s, tr_e, tr_types, cfg.tag.max_span, N_TYPES)
  local en_dvo, en_dvs, en_dve = csr.enumerate_subspans(dv_off, dv_s, dv_e, dv_types, cfg.tag.max_span, N_TYPES)
  local en_teo, en_tes, en_tee = csr.enumerate_subspans(te_off, te_s, te_e, te_types, cfg.tag.max_span, N_TYPES)

  -- union the enumerated spans with gazetteer surface matches, deduped on (start,end); both untyped
  local ac = build_gazetteer(train)
  local function zeros (n) local z = ivec.create(n); z:zero(); return z end
  local function cand_union (split, eoff, es, ee, goff, gs, ge)
    local g_off, g_s, g_e = gaz_candidates(ac, split)
    return csr.union_spans({
      a_offsets = eoff, a_starts = es, a_ends = ee, a_types = zeros(es:size()),
      b_offsets = g_off, b_starts = g_s, b_ends = g_e, b_types = zeros(g_s:size()),
      gold_offsets = goff, gold_starts = gs, gold_ends = ge, gold_types = zeros(gs:size()),
    })
  end
  local tr_co, tr_cs, tr_ce = cand_union(train, en_tro, en_trs, en_tre, tr_eoff, tr_es, tr_ee)
  local dv_co, dv_cs, dv_ce = cand_union(dev, en_dvo, en_dvs, en_dve, dv_eoff, dv_es, dv_ee)
  local te_co, te_cs, te_ce = cand_union(test_set, en_teo, en_tes, en_tee, te_eoff, te_es, te_ee)
  local n_trc, n_dvc, n_tec = tr_cs:size(), dv_cs:size(), te_cs:size()

  -- per-candidate class: gold type if (start,end) matches a gold span, else REJECT (= N_TYPES)
  local tr_tlab = csr.type_labels(tr_co, tr_cs, tr_ce, tr_eoff, tr_es, tr_ee, tr_ety, N_TYPES)
  local dv_tlab = csr.type_labels(dv_co, dv_cs, dv_ce, dv_eoff, dv_es, dv_ee, dv_ety, N_TYPES)
  local tr_tloff = ivec.create(n_trc + 1); tr_tloff:fill_indices()
  local dv_tloff = ivec.create(n_dvc + 1); dv_tloff:fill_indices()

  local te_gold = te_eoff:get(te_eoff:size() - 1)
  local n_te_docs = te_off:size() - 1
  local miss = csr.span_miss_report({
    gaz_offsets = zeros(n_te_docs + 1), gaz_starts = ivec.create(0),
    gaz_ends = ivec.create(0), gaz_types = ivec.create(0),
    bio_offsets = te_co, bio_starts = te_cs, bio_ends = te_ce, bio_types = zeros(n_tec),
    gold_offsets = te_eoff, gold_starts = te_es, gold_ends = te_ee, gold_types = te_ety,
    n_types = N_TYPES,
  })
  local ubt = miss.under_by_type
  str.printf("[Cands] train=%d dev=%d test=%d | test span-coverage=%.4f"
    .. " | miss over=%d under=%d cross=%d none=%d"
    .. " | under by-type[PER=%d ORG=%d LOC=%d MISC=%d] %s\n",
    n_trc, n_dvc, n_tec, (miss.covered + miss.wrong_type) / te_gold,
    miss.over, miss.under, miss.cross, miss.none,
    ubt[0], ubt[1], ubt[2], ubt[3], sw())

  -- ===== Stage 2: TYPE (per-span type-or-reject) =====
  -- cfg.type.blocks selects the feature blocks: "dense" = 4 soft-label pools (width P, no BNS),
  -- "char" = 4 char-pools over each candidate (width V, BNS), "both" = all 8. Per-block scaled.
  local ty_learned = {}
  local function type_feats (scores, tok_off, tok_s, tok_e, c_off, c_tok, c_pos, co, cs, ce, n, learn)
    local mode = cfg.type.blocks
    local blocks, widths, want_bns = {}, {}, {}
    local function add (bs, w, b)
      for i = 1, NSEG do blocks[#blocks + 1] = bs[i]; widths[#widths + 1] = w; want_bns[#want_bns + 1] = b end
    end
    if mode ~= "char" then add(pool_dense(scores, P, tok_off, tok_s, tok_e, co, cs, ce, cfg.type.alpha), P, false) end
    if mode ~= "dense" then add(pool_sparse(c_off, c_tok, c_pos, V, co, cs, ce, cfg.tag.alpha), V, true) end
    if learn then ty_learned.bns = {} end
    for i = 1, #blocks do
      if want_bns[i] then
        if learn then
          ty_learned.bns[i] = csr.apply_bns(blocks[i][1], blocks[i][2], blocks[i][3], nil, tr_tloff, tr_tlab, widths[i], P)
        else
          csr.apply_bns(blocks[i][1], blocks[i][2], blocks[i][3], ty_learned.bns[i])
        end
      end
    end
    local bounds = { 0 }
    local o, t, v = blocks[1][1], blocks[1][2], blocks[1][3]
    local shift = widths[1]; bounds[#bounds + 1] = shift
    for i = 2, #blocks do
      o, t, v = csr.merge(o, t, v, blocks[i][1], blocks[i][2], blocks[i][3], shift)
      shift = shift + widths[i]; bounds[#bounds + 1] = shift
    end
    local ntok = shift
    if learn then
      local ss = csr.block_sumsq(t, v, bounds)
      local bs = fvec.create(ntok)
      for i = 1, #blocks do
        local s = ss:get(i - 1)
        bs:fill(s > 0 and math.sqrt(n / s) or 0.0, bounds[i], bounds[i + 1])
      end
      ty_learned.block = bs
    end
    csr.standardize(o, t, v, ty_learned.block)
    csr.normalize(o, v)
    return o, t, v, ntok
  end
  local ty_off, ty_tok, ty_val, ty_ntok =
    type_feats(tr_scores, tr_off, tr_s, tr_e, s_off, s_tok, s_pos, tr_co, tr_cs, tr_ce, n_trc, true)
  local ty_dvo, ty_dvt, ty_dvv =
    type_feats(dv_scores, dv_off, dv_s, dv_e, ds_off, ds_tok, ds_pos, dv_co, dv_cs, dv_ce, n_dvc, false)

  str.printf("[Type] Encoding\n")
  local sp_ty, ridge_ty, _, ty_best = optimize.krr({
    offsets = ty_off, tokens = ty_tok, values = ty_val, n_samples = n_trc, n_tokens = ty_ntok,
    kernel = cfg.type.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_tloff, label_neighbors = tr_tlab, n_labels = P,
    val_offsets = ty_dvo, val_tokens = ty_dvt, val_values = ty_dvv, val_n_samples = n_dvc,
    val_expected_offsets = dv_tloff, val_expected_neighbors = dv_tlab,
    lambda = cfg.type.lambda, propensity_a = cfg.type.propensity_a, propensity_b = cfg.type.propensity_b,
    k = 1, search_trials = cfg.type.search_trials,
    each = util.make_ridge_log(stopwatch),
    trial_fn = function (gram, params)
      local f1, p, r = gram:label_accuracy(params.lambda, 1,
        params.propensity_a, params.propensity_b, dv_tloff, dv_tlab, 1)
      return f1, { f1 = f1, precision = p, recall = r }
    end,
  })
  str.printf("[Type] kernel=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
    ty_best.kernel, ty_best.lambda, ty_best.propensity_a, ty_best.propensity_b, sw())

  -- top-2 type scores per test candidate; DP overlap resolution over non-reject candidates
  local ty_teo, ty_tet, ty_tev =
    type_feats(te_scores, te_off, te_s, te_e, es_off, es_tok, es_pos, te_co, te_cs, te_ce, n_tec, false)
  local te_codes = sp_ty:encode({ offsets = ty_teo, tokens = ty_tet, values = ty_tev, n_samples = n_tec })
  local _, te_lab, te_sco = ridge_ty:label(te_codes, n_tec, 2)
  local keep, te_cls = csr.nms_dp(te_co, te_cs, te_ce, te_lab, te_sco, 2, N_TYPES)
  local p_off, p_s, p_e, p_ty = csr.filter_spans(te_co, te_cs, te_ce, te_cls, keep)
  local f1, p, r = eval.span_f1(p_off, p_s, p_e, p_ty, te_eoff, te_es, te_ee, te_ety)
  str.printf("[Hybrid Test] span: f1=%.4f p=%.4f r=%.4f %s\n", f1, p, r, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
