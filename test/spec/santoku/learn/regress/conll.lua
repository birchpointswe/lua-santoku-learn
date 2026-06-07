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

-- Hybrid NER with a STACKED BIO candidate generator. The BIO branch is two taggers: L1 tags tokens
-- from the focus-marked raw sentence (collapse=none, frozen), decode -> L1 spans; L2 re-tags with
-- L1's spans as collapsed context (collapse=spans) and a Viterbi transition layer -> stacked spans.
-- These (higher-recall) stacked spans are UNIONed with gazetteer matches, then span-level accept/
-- reject prunes to precision. Stacked BIO lifts union recall above the single tagger's ~0.61.
-- (A doc-context re-scoring layer was tried and removed: it never beat plain accept/reject, since the
-- accept/reject features already see the whole sentence.)
--
-- COMPROMISE (train candidates): dev/test BIO candidates are the honest full stacked tagger (no
-- leak). Train candidates apply the main L2 to OOF-L1 context: the L2 INPUT context is out-of-fold
-- (honest), but L2's PARAMETERS saw those rows, so train candidates are mildly optimistic. The
-- rigorous fix is outer-OOF-of-the-whole-stack (retrain L1+L2 per fold); deferred as it ~triples
-- training time. TODO: outer-OOF stacked train candidates if this v1 shows promise.

local cfg = {
  data = { dir = "test/res/conll2003", max = nil },
  tok = { ngram_min = 3, ngram_max = 5, collapse = "none", normalize = false },
  emb = { n_landmarks = 1024 * 8, trace_tol = 0.01 },
  -- BIO is 9-class (O + B/I x 4 types) with heavy imbalance (O dominates, MISC rare) -> search
  -- propensity to weight the rare entity classes. Accept/reject is binary (nl=1) -> propensity inert.
  -- Each stage carries its own winner-first kernel list (locked at search_trials=0).
  bio_l1 = { kernel = { "cosine", "expcos", "geolaplace", "matern52", "rq", "arccos1" },
    lambda = { min = 1e-4, max = 1e1, log = true, def = 2.5631e-04 },
    propensity_a = { min = 0, max = 4, def = 0.2351 }, propensity_b = { min = 0, max = 8, def = 5.8075 },
    search_trials = 0 },                              -- BIO L1 (best: cosine)
  bio_l2 = { kernel = { "rq", "cosine", "expcos", "geolaplace", "matern52", "arccos1" },
    lambda = { min = 1e-4, max = 1e1, log = true, def = 1.1447e-04 },
    propensity_a = { min = 0, max = 4, def = 0.2611 }, propensity_b = { min = 0, max = 8, def = 7.0661 },
    search_trials = 0, collapse = "spans" },          -- stacked BIO L2 (best: rq)
  ar = { kernel = { "cosine", "expcos", "geolaplace", "matern52", "rq", "arccos1" },
    lambda = { min = 1e-4, max = 1e1, log = true, def = 2.3105e-02 },
    search_trials = 0 },                              -- accept/reject, binary (best: cosine)
  stack = { k = 5 },
}

local boundary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
local N_TYPES = 4
local N_CLASSES = 2 * N_TYPES + 1

local function build_gazetteer (train)
  local counts = {}
  for d = 1, train.n do
    local text = train.texts[d]
    for _, e in ipairs(train.sent_ents[d]) do
      local surf = text:sub(e.s + 1, e.e)
      local c = counts[surf]; if not c then c = { 0, 0, 0, 0 }; counts[surf] = c end
      c[e.t + 1] = c[e.t + 1] + 1
    end
  end
  local patterns, ids = {}, ivec.create()
  for surf, c in pairs(counts) do
    local bt, bn = 0, -1
    for t = 0, 3 do if c[t + 1] > bn then bn = c[t + 1]; bt = t end end
    patterns[#patterns + 1] = surf; ids:push(bt)
  end
  return aho.create({ ids = ids, patterns = patterns, normalize = true })
end

local function gaz_candidates (ac, split)
  local off, mids, starts, ends = ac:predict({ texts = split.texts, longest = true, boundary = boundary })
  return off, starts, ends, mids
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

local function tokenize (split, off, starts, ends, types, ngram_map)
  return csr.tokenize_annotated({
    texts = split.texts, doc_span_offsets = off,
    span_starts = starts, span_ends = ends, span_types = types,
    collapse = cfg.tok.collapse, ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
    normalize = cfg.tok.normalize, terminals = true, ngram_map = ngram_map,
  })
end

-- render each focus span against the collapsed backdrop of context spans
local function tokenize_ctx (split, off, starts, ends, types, coff, cs, ce, cty, ngram_map, collapse)
  return csr.tokenize_annotated({
    texts = split.texts, doc_span_offsets = off,
    span_starts = starts, span_ends = ends, span_types = types,
    context_offsets = coff, context_starts = cs, context_ends = ce, context_types = cty,
    collapse = collapse, ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
    normalize = cfg.tok.normalize, terminals = true, ngram_map = ngram_map,
  })
end

local function build_fold (off, n, k)
  local fold = ivec.create(n)
  for d = 0, off:size() - 2 do fold:fill(d % k, off:get(d), off:get(d + 1)) end
  return fold
end

test("conll", function ()

  local stopwatch = utc.stopwatch()
  local function sw () local d, dd = stopwatch(); return str.format("(%.1fs +%.1fs)", d, dd) end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  str.printf("[Data] train=%d dev=%d test=%d %s\n", train.n, dev.n, test_set.n, sw())

  -- ===== token + gold spans =====
  local tr_off, tr_s, tr_e, tr_ty, tr_loff, tr_lnbr, tr_n, tr_eoff, tr_es, tr_ee, tr_ety = build_tokens(train)
  local dv_off, dv_s, dv_e, dv_ty, dv_loff, dv_lnbr, dv_n, dv_eoff, dv_es, dv_ee, dv_ety = build_tokens(dev)
  local te_off, te_s, te_e, te_ty, _, _, te_n, te_eoff, te_es, te_ee, te_ety = build_tokens(test_set)

  -- ===== stacked BIO: L1 tagger (cosine, search lambda) =====
  -- wrapped in a do-block so its locals free up (Lua 5.1 caps active locals per function)
  local tr_bo, tr_bs, tr_be, tr_bty
  local dv_bo, dv_bs, dv_be, dv_bty
  local te_bo, te_bs, te_be, te_bty
  do
  local l1_ngram, b_off, b_tok, b_val, b_ntok = tokenize(train, tr_off, tr_s, tr_e, tr_ty, nil)
  local raw_off, raw_tok, raw_val = b_off, b_tok, fvec.create(b_val)  -- pristine for OOF
  local b_bns = csr.apply_bns(b_off, b_tok, b_val, nil, tr_loff, tr_lnbr, b_ntok, N_CLASSES)
  csr.normalize(b_off, b_val)
  local _, l1_dvo, l1_dvt, l1_dvv = tokenize(dev, dv_off, dv_s, dv_e, dv_ty, l1_ngram)
  csr.apply_bns(l1_dvo, l1_dvt, l1_dvv, b_bns); csr.normalize(l1_dvo, l1_dvv)
  str.printf("[BIO L1] Encoding\n")
  local sp1, ridge1, _, l1_best = optimize.krr({
    offsets = b_off, tokens = b_tok, values = b_val, n_samples = tr_n, n_tokens = b_ntok,
    kernel = cfg.bio_l1.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_loff, label_neighbors = tr_lnbr, n_labels = N_CLASSES,
    val_offsets = l1_dvo, val_tokens = l1_dvt, val_values = l1_dvv, val_n_samples = dv_n,
    val_expected_offsets = dv_loff, val_expected_neighbors = dv_lnbr,
    lambda = cfg.bio_l1.lambda, propensity_a = cfg.bio_l1.propensity_a, propensity_b = cfg.bio_l1.propensity_b,
    k = 1, search_trials = cfg.bio_l1.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  str.printf("[BIO L1] kernel=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
    l1_best.kernel, l1_best.lambda, l1_best.propensity_a, l1_best.propensity_b, sw())

  -- L1 argmax-decoded backdrop spans (context for L2) for a split
  local function l1_context (split, off, s, e, ty, n)
    local _, o, t, v = tokenize(split, off, s, e, ty, l1_ngram)
    csr.apply_bns(o, t, v, b_bns); csr.normalize(o, v)
    local codes = sp1:encode({ offsets = o, tokens = t, values = v, n_samples = n })
    local _, amax = ridge1:label(codes, n, 1)
    return csr.bio_decode(off, s, e, amax, N_TYPES)
  end
  local dv_c1o, dv_c1s, dv_c1e, dv_c1y = l1_context(dev, dv_off, dv_s, dv_e, dv_ty, dv_n)
  local te_c1o, te_c1s, te_c1e, te_c1y = l1_context(test_set, te_off, te_s, te_e, te_ty, te_n)

  -- OOF L1 argmax tags on train -> train backdrop spans (honest context for L2 train)
  local fold = build_fold(tr_off, tr_n, cfg.stack.k)
  local oof_tags = optimize.oof({
    n = tr_n, k = cfg.stack.k, fold = fold, out = ivec.create(tr_n),
    each = function (ev) str.printf("[BIO L1 OOF] fold %d/%d %s\n", ev.fold, ev.folds, sw()) end,
    fit = function (idx)
      local so, st, sv = csr.gather_rows(raw_off, raw_tok, raw_val, idx)
      local slab = ivec.create(); slab:copy(tr_lnbr, idx)
      local sloff = ivec.create(idx:size() + 1); sloff:fill_indices()
      local sbns = csr.apply_bns(so, st, sv, nil, sloff, slab, b_ntok, N_CLASSES)
      csr.normalize(so, sv)
      local _, sp, gram = spectral.encode({
        offsets = so, tokens = st, values = sv, n_tokens = b_ntok, n_samples = idx:size(),
        kernel = l1_best.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
        label_offsets = sloff, label_neighbors = slab, n_labels = N_CLASSES,
      })
      return { sp = sp, r = ridge.create({ gram = gram, lambda = l1_best.lambda,
        propensity_a = l1_best.propensity_a, propensity_b = l1_best.propensity_b }), bns = sbns }
    end,
    predict = function (h, idx)
      local so, st, sv = csr.gather_rows(raw_off, raw_tok, raw_val, idx)
      csr.apply_bns(so, st, sv, h.bns); csr.normalize(so, sv)
      local codes = h.sp:encode({ offsets = so, tokens = st, values = sv, n_samples = idx:size() })
      local _, nbr = h.r:label(codes, idx:size(), 1)
      return nbr
    end,
  })
  local tr_c1o, tr_c1s, tr_c1e, tr_c1y = csr.bio_decode(tr_off, tr_s, tr_e, oof_tags, N_TYPES)

  -- ===== stacked BIO: L2 tagger (re-tag with L1-span context) =====
  local l2_ngram, l2_off, l2_tok, l2_val, l2_ntok =
    tokenize_ctx(train, tr_off, tr_s, tr_e, tr_ty, tr_c1o, tr_c1s, tr_c1e, tr_c1y, nil, cfg.bio_l2.collapse)
  local l2_bns = csr.apply_bns(l2_off, l2_tok, l2_val, nil, tr_loff, tr_lnbr, l2_ntok, N_CLASSES)
  csr.normalize(l2_off, l2_val)
  local _, l2_dvo, l2_dvt, l2_dvv =
    tokenize_ctx(dev, dv_off, dv_s, dv_e, dv_ty, dv_c1o, dv_c1s, dv_c1e, dv_c1y, l2_ngram, cfg.bio_l2.collapse)
  csr.apply_bns(l2_dvo, l2_dvt, l2_dvv, l2_bns); csr.normalize(l2_dvo, l2_dvv)

  str.printf("[BIO L2] Encoding (collapse=%s)\n", cfg.bio_l2.collapse)
  local sp2, ridge2, _, bio_best = optimize.krr({
    offsets = l2_off, tokens = l2_tok, values = l2_val, n_samples = tr_n, n_tokens = l2_ntok,
    kernel = cfg.bio_l2.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_loff, label_neighbors = tr_lnbr, n_labels = N_CLASSES,
    val_offsets = l2_dvo, val_tokens = l2_dvt, val_values = l2_dvv, val_n_samples = dv_n,
    val_expected_offsets = dv_loff, val_expected_neighbors = dv_lnbr,
    lambda = cfg.bio_l2.lambda, propensity_a = cfg.bio_l2.propensity_a, propensity_b = cfg.bio_l2.propensity_b,
    k = 1, search_trials = cfg.bio_l2.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  str.printf("[BIO L2] kernel=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
    bio_best.kernel, bio_best.lambda, bio_best.propensity_a, bio_best.propensity_b, sw())

  -- stacked spans for a split: L1 context -> L2 re-tag -> argmax -> decode
  local function stacked_spans (split, off, s, e, ty, n, co, cs, ce, cy)
    local _, lo, lt, lv =
      tokenize_ctx(split, off, s, e, ty, co, cs, ce, cy, l2_ngram, cfg.bio_l2.collapse)
    csr.apply_bns(lo, lt, lv, l2_bns); csr.normalize(lo, lv)
    local codes = sp2:encode({ offsets = lo, tokens = lt, values = lv, n_samples = n })
    local _, amax = ridge2:label(codes, n, 1)
    return csr.bio_decode(off, s, e, amax, N_TYPES)
  end
  dv_bo, dv_bs, dv_be, dv_bty = stacked_spans(dev, dv_off, dv_s, dv_e, dv_ty, dv_n, dv_c1o, dv_c1s, dv_c1e, dv_c1y)
  te_bo, te_bs, te_be, te_bty = stacked_spans(test_set, te_off, te_s, te_e, te_ty, te_n, te_c1o, te_c1s, te_c1e, te_c1y)
  -- train candidates: main L2 over OOF-L1 context (honest context, L2 params saw rows -> mild leak)
  tr_bo, tr_bs, tr_be, tr_bty = stacked_spans(train, tr_off, tr_s, tr_e, tr_ty, tr_n, tr_c1o, tr_c1s, tr_c1e, tr_c1y)

  -- stacked BIO own span-F1 on test (in-run baseline for the candidate generator)
  local f1, p, r = eval.span_f1(te_bo, te_bs, te_be, te_bty, te_eoff, te_es, te_ee, te_ety)
  str.printf("[BIO stacked Test] span: f1=%.4f p=%.4f r=%.4f %s\n", f1, p, r, sw())
  end

  -- ===== candidates = gazetteer UNION stacked-BIO, labeled by gold =====
  local ac = build_gazetteer(train)
  local function union (g_off, g_s, g_e, g_ty, b_o, b_s2, b_e2, b_ty2, eoff, es, ee, ety)
    return csr.union_spans({
      a_offsets = g_off, a_starts = g_s, a_ends = g_e, a_types = g_ty,
      b_offsets = b_o, b_starts = b_s2, b_ends = b_e2, b_types = b_ty2,
      gold_offsets = eoff, gold_starts = es, gold_ends = ee, gold_types = ety,
    })
  end
  local g_tro, g_trs, g_tre, g_trty = gaz_candidates(ac, train)
  local g_dvo, g_dvs, g_dve, g_dvty = gaz_candidates(ac, dev)
  local g_teo, g_tes, g_tee, g_tety = gaz_candidates(ac, test_set)
  local u_tro, u_trs, u_tre, u_trty, u_trlab =
    union(g_tro, g_trs, g_tre, g_trty, tr_bo, tr_bs, tr_be, tr_bty, tr_eoff, tr_es, tr_ee, tr_ety)
  local u_dvo, u_dvs, u_dve, u_dvty, u_dvlab =
    union(g_dvo, g_dvs, g_dve, g_dvty, dv_bo, dv_bs, dv_be, dv_bty, dv_eoff, dv_es, dv_ee, dv_ety)
  local u_teo, u_tes, u_tee, u_tety, u_telab =
    union(g_teo, g_tes, g_tee, g_tety, te_bo, te_bs, te_be, te_bty, te_eoff, te_es, te_ee, te_ety)
  local te_gold = te_eoff:get(te_eoff:size() - 1)
  str.printf("[Union] cands train=%d dev=%d test=%d | test coverage=%.4f %s\n",
    u_trlab:size(), u_dvlab:size(), u_telab:size(), u_telab:sum() / te_gold, sw())

  -- ===== accept/reject L1 on union candidates (frozen cosine/lambda) =====
  local n_trc, n_dvc, n_tec = u_trlab:size(), u_dvlab:size(), u_telab:size()
  local ar_ngram, ar_off, ar_tok, ar_val, ar_ntok = tokenize(train, u_tro, u_trs, u_tre, u_trty, nil)
  local tr_aloff, tr_alnbr = csr.binary_label_csr(u_trlab)
  local ar_bns = csr.apply_bns(ar_off, ar_tok, ar_val, nil, tr_aloff, tr_alnbr, ar_ntok, 1)
  csr.normalize(ar_off, ar_val)

  local _, ar_dvo, ar_dvt, ar_dvv = tokenize(dev, u_dvo, u_dvs, u_dve, u_dvty, ar_ngram)
  csr.apply_bns(ar_dvo, ar_dvt, ar_dvv, ar_bns)
  csr.normalize(ar_dvo, ar_dvv)
  local dv_aloff, dv_alnbr = csr.binary_label_csr(u_dvlab)

  str.printf("[Accept/Reject] Encoding\n")
  local sp_ar, ridge_ar, ar_dvcodes, ar_best = optimize.krr({
    offsets = ar_off, tokens = ar_tok, values = ar_val, n_samples = n_trc, n_tokens = ar_ntok,
    kernel = cfg.ar.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_aloff, label_neighbors = tr_alnbr, n_labels = 1,
    val_offsets = ar_dvo, val_tokens = ar_dvt, val_values = ar_dvv, val_n_samples = n_dvc,
    val_expected_offsets = dv_aloff, val_expected_neighbors = dv_alnbr,
    lambda = cfg.ar.lambda, k = 1, search_trials = cfg.ar.search_trials,
    each = util.make_ridge_log(stopwatch),
    trial_fn = function (gram, params)
      local f1, p, r = gram:label_accuracy(params.lambda, 1, nil, nil, dv_aloff, dv_alnbr, "gfm")
      return f1, { f1 = f1, precision = p, recall = r }
    end,
  })
  str.printf("[Accept/Reject] kernel=%s lambda=%.4e %s\n", ar_best.kernel, ar_best.lambda, sw())
  local d_off, d_nbr, d_sco = ridge_ar:label(ar_dvcodes, n_dvc, 1)
  local gfm_ar = optimize.gfm({
    n_labels = 1, val_offsets = d_off, val_neighbors = d_nbr, val_scores = d_sco,
    val_n_samples = n_dvc, val_expected_offsets = dv_aloff, val_expected_neighbors = dv_alnbr,
  })

  local _, ar_teo, ar_tet, ar_tev = tokenize(test_set, u_teo, u_tes, u_tee, u_tety, ar_ngram)
  csr.apply_bns(ar_teo, ar_tet, ar_tev, ar_bns)
  csr.normalize(ar_teo, ar_tev)
  local te_codes = sp_ar:encode({ offsets = ar_teo, tokens = ar_tet, values = ar_tev, n_samples = n_tec })
  local t_off, t_nbr, t_sco = ridge_ar:label(te_codes, n_tec, 1)
  local te_ks = gfm_ar:predict({ offsets = t_off, neighbors = t_nbr, scores = t_sco, n_samples = n_tec })
  do
    local p_off, p_s, p_e, p_ty = csr.filter_spans(u_teo, u_tes, u_tee, u_tety, te_ks)
    local f1, p, r = eval.span_f1(p_off, p_s, p_e, p_ty, te_eoff, te_es, te_ee, te_ety)
    str.printf("[Hybrid Test] span: f1=%.4f p=%.4f r=%.4f %s\n", f1, p, r, sw())
  end

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
