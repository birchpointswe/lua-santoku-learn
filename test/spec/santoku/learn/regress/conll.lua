local ds = require("santoku.learn.dataset")
local csr = require("santoku.learn.csr")
local tokenizer = require("santoku.learn.tokenizer")
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

-- Span NER as two stages: TAG -> (enumerate) -> TYPE. Enumerate-and-classify.
--  1. TAG   (per-token type): an (N_TYPES+1)-class token classifier (O + the 4 entity types), argmax.
--     OOF'd so the train candidates below are honest.
--  enumerate: within each contiguous non-O run, emit EVERY subspan of 1..max_span tokens. Over-generate
--     the candidate spans instead of committing to boundaries (no segment, no over-split).
--  2. TYPE  (per-span type-or-reject): the same (N_TYPES+1)-class head (4 types + REJECT) classifies each
--     candidate span by argmax; non-REJECT survivors, typed by their predicted class, are the output.
-- Both stages: collapse=none, no context, propensity for the dominant O/REJECT class, searched.

local cfg = {
  data = { dir = "test/res/conll2003", max = nil },
  tok = { ngram_min = 3, ngram_max = 5, normalize = false },
  emb = { n_landmarks = 1024 * 8, trace_tol = 0.01 },
  -- both heads are (N_TYPES+1)-class, decided by argmax (topk=1), propensity for the O/REJECT imbalance.
  -- TAG and TYPE locked to their searched winners (kernel/lambda/prop_a/prop_b all from the optimizer).
  -- max_span is a candidate-generation knob, not a model hyperparameter.
  tag = { kernel = { "cosine", "expcos", "geolaplace", "matern52", "rq", "arccos1" },
    lambda = { min = 1e-4, max = 1e1, log = true, def = 1.1980e-02 },
    propensity_a = { min = 0, max = 8, def = 0.2607 }, propensity_b = { min = 0, max = 16, def = 8.6397 },
    search_trials = 0, max_span = 5 },                -- stage 1: per-token type (O + N_TYPES)
  type = { kernel = { "arccos1", "cosine", "expcos", "geolaplace", "matern52", "rq" },
    lambda = { min = 1e-4, max = 1e1, log = true, def = 4.5416e-04 },
    propensity_a = { min = 0, max = 8, def = 7.5553 }, propensity_b = { min = 0, max = 16, def = 6.0956 },
    search_trials = 0 },                              -- stage 2: locked winner = 0.8131 (solver-agnostic; re-search overfits)
  stack = { k = 5 },
  exp = { typed_focus = false, shape = true, tag_shape = false },   -- best = stage-2 shape only (0.8141)
}

local N_TYPES = 4
local boundary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

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

-- typed gazetteer: lowercased train surface -> per-type occurrence counts, for P(type|surface) features.
local function build_typed_gaz (train)
  local gaz = {}
  for d = 1, train.n do
    local text = train.texts[d]
    for _, e in ipairs(train.sent_ents[d]) do
      local surf = text:sub(e.s + 1, e.e):lower()
      local c = gaz[surf]
      if not c then c = { total = 0 }; for ty = 0, N_TYPES - 1 do c[ty] = 0 end; gaz[surf] = c end
      c[e.t] = c[e.t] + 1
      c.total = c.total + 1
    end
  end
  return gaz
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

-- TEXT block (old collapse="none" -> focus=true). The threaded value `tok` is now the
-- tokenizer object: nil => create + grow the vocab (train); object => frozen (dev/test).
local function tokenize (split, off, starts, ends, _, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      terminals = true, focus = true, normalize = cfg.tok.normalize,
    })
  end
  local o, t, v = tok:tokenize({
    texts = split.texts, n_samples = split.n,
    doc_span_offsets = off, span_starts = starts, span_ends = ends, grow = grow,
  })
  return tok, o, t, v, tok:n_tokens()
end

-- Experiment (a): typed-focus TEXT block. Same as `tokenize` but the focus brackets
-- carry the candidate's predicted type (fty: per-candidate, -1=UNK / t / N_TYPES=O),
-- injecting the tagger's signal into the TYPE-stage text features.
local function tokenize_typed (split, off, starts, ends, fty, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = N_TYPES, terminals = true, focus = "typed", normalize = cfg.tok.normalize,
    })
  end
  local o, t, v = tok:tokenize({
    texts = split.texts, n_samples = split.n,
    doc_span_offsets = off, span_starts = starts, span_ends = ends,
    focus_types = fty, grow = grow,
  })
  return tok, o, t, v, tok:n_tokens()
end

-- CONTEXT block (old collapse="context" -> stream="type"): focus span over the per-token
-- type skeleton. `collapse` arg retained for call-site compatibility (always "context").
local function tokenize_ctx (split, off, starts, ends, _, coff, cs, ce, cty, tok, _)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = N_TYPES, terminals = true, focus = true, stream = "type",
    })
  end
  local o, t, v = tok:tokenize({
    texts = split.texts, n_samples = split.n,
    doc_span_offsets = off, span_starts = starts, span_ends = ends,
    context_offsets = coff, context_starts = cs, context_ends = ce, context_types = cty,
    grow = grow,
  })
  return tok, o, t, v, tok:n_tokens()
end

-- Experiment (c): SHAPE block. Per-candidate focus over the doc's per-token shape
-- skeleton (Xx/XX/xx/dd/punct/mix, classified on original bytes -- casing/digit signal
-- the text/type blocks lack). Same span plumbing as tokenize_ctx; no context_types.
local function tokenize_shape (split, off, starts, ends, coff, cs, ce, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      terminals = true, focus = true, stream = "shape",
    })
  end
  local o, t, v = tok:tokenize({
    texts = split.texts, n_samples = split.n,
    doc_span_offsets = off, span_starts = starts, span_ends = ends,
    context_offsets = coff, context_starts = cs, context_ends = ce,
    grow = grow,
  })
  return tok, o, t, v, tok:n_tokens()
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
  local tr_off, tr_s, tr_e, tr_ty, _, tr_lnbr, tr_n, tr_eoff, tr_es, tr_ee, tr_ety = build_tokens(train)
  local dv_off, dv_s, dv_e, dv_ty, _, dv_lnbr, dv_n, dv_eoff, dv_es, dv_ee, dv_ety = build_tokens(dev)
  local te_off, te_s, te_e, te_ty, _, _, te_n, te_eoff, te_es, te_ee, te_ety = build_tokens(test_set)

  -- Combine the tag stage's text block (+ optional shape block, stage-1 experiment)
  -- into one BNS'd, block-scaled, L2-normalized CSR. fit=true computes & returns the
  -- bns + per-block scale; fit=false applies the supplied bns/blk. Mutates inputs.
  -- When tag_shape is off (sb=nil) this is byte-identical to the old single-block path.
  local function tag_combine (to, tt, tv, tk, so, st, sv, sk, nl, loff, lnbr, n, fit, bns_in, blk_in)
    if not so then
      if fit then
        local bns = csr.apply_bns(to, tt, tv, nil, loff, lnbr, tk, nl)
        csr.normalize(to, tv)
        return to, tt, tv, tk, bns, nil
      end
      csr.apply_bns(to, tt, tv, bns_in); csr.normalize(to, tv)
      return to, tt, tv, tk, bns_in, nil
    end
    local o, t, v = csr.merge(to, tt, tv, so, st, sv, tk)
    local ntok = tk + sk
    local bns, blk
    if fit then
      bns = csr.apply_bns(o, t, v, nil, loff, lnbr, ntok, nl)
      local ss = csr.block_sumsq(t, v, { 0, tk, ntok })
      blk = fvec.create(ntok)
      blk:fill(ss:get(0) > 0 and math.sqrt(n / ss:get(0)) or 0.0, 0, tk)
      blk:fill(ss:get(1) > 0 and math.sqrt(n / ss:get(1)) or 0.0, tk, ntok)
    else
      bns, blk = bns_in, blk_in
      csr.apply_bns(o, t, v, bns)
    end
    csr.standardize(o, t, v, blk)
    csr.normalize(o, v)
    return o, t, v, ntok, bns, blk
  end

  -- shared multiclass-head scaffold (tag = per-token type; type = per-span type-or-reject): krr with
  -- propensity, selected and decoded by argmax (topk=1). tb/sb = text/shape block bundles {o,t,v,k}.
  local function train_head (label, scfg, tb, sb, n, nl, loff, lnbr, dtb, dsb, dvn, dloff, dlnbr)
    local o, t, v, ntok, bns, blk = tag_combine(tb.o, tb.t, tb.v, tb.k,
      sb and sb.o, sb and sb.t, sb and sb.v, sb and sb.k, nl, loff, lnbr, n, true)
    local dvo, dvt, dvv = tag_combine(dtb.o, dtb.t, dtb.v, dtb.k,
      dsb and dsb.o, dsb and dsb.t, dsb and dsb.v, dsb and dsb.k, nl, nil, nil, nil, false, bns, blk)
    str.printf("[%s] Encoding\n", label)
    local sp, rg, _, best = optimize.krr({
      offsets = o, tokens = t, values = v, n_samples = n, n_tokens = ntok,
      kernel = scfg.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
      label_offsets = loff, label_neighbors = lnbr, n_labels = nl,
      val_offsets = dvo, val_tokens = dvt, val_values = dvv, val_n_samples = dvn,
      val_expected_offsets = dloff, val_expected_neighbors = dlnbr,
      lambda = scfg.lambda, propensity_a = scfg.propensity_a, propensity_b = scfg.propensity_b,
      k = 1, search_trials = scfg.search_trials,
      each = util.make_ridge_log(stopwatch),
      -- argmax (fixed_k=1) classification F1 to match the argmax decode (default would use gfm)
      trial_fn = function (gram, params)
        local f1, p, r = gram:label_accuracy(params.lambda, 1,
          params.propensity_a, params.propensity_b, dloff, dlnbr, 1)
        return f1, { f1 = f1, precision = p, recall = r }
      end,
    })
    str.printf("[%s] kernel=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
      label, best.kernel, best.lambda, best.propensity_a, best.propensity_b, sw())
    return sp, rg, bns, blk, best
  end

  local function head_argmax (sp, rg, bns, blk, tb, sb, n)
    local o, t, v = tag_combine(tb.o, tb.t, tb.v, tb.k,
      sb and sb.o, sb and sb.t, sb and sb.v, sb and sb.k, nil, nil, nil, nil, false, bns, blk)
    local codes = sp:encode({ offsets = o, tokens = t, values = v, n_samples = n })
    local _, cls, sco = rg:label(codes, n, 1)
    return cls, sco
  end

  local function oof_head_argmax (label, best, nl, rtb, rsb, labvec, n, fold)
    return optimize.oof({
      n = n, k = cfg.stack.k, fold = fold, out = ivec.create(n),
      each = function (ev) str.printf("[%s OOF] fold %d/%d %s\n", label, ev.fold, ev.folds, sw()) end,
      fit = function (idx)
        local to, tt, tv = csr.gather_rows(rtb.o, rtb.t, rtb.v, idx)
        local so, st, sv
        if rsb then so, st, sv = csr.gather_rows(rsb.o, rsb.t, rsb.v, idx) end
        local lnbr = ivec.create(); lnbr:copy(labvec, idx)
        local loff = ivec.create(idx:size() + 1); loff:fill_indices()
        local o, t, v, ntok, bns, blk = tag_combine(to, tt, tv, rtb.k,
          so, st, sv, rsb and rsb.k, nl, loff, lnbr, idx:size(), true)
        local _, sp, gram = spectral.encode({
          offsets = o, tokens = t, values = v, n_tokens = ntok, n_samples = idx:size(),
          kernel = best.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
          label_offsets = loff, label_neighbors = lnbr, n_labels = nl,
          solve_lambda = best.lambda,
          solve_propensity_a = best.propensity_a, solve_propensity_b = best.propensity_b,
        })
        return { sp = sp, r = ridge.create({ gram = gram }), bns = bns, blk = blk }
      end,
      predict = function (h, idx)
        local to, tt, tv = csr.gather_rows(rtb.o, rtb.t, rtb.v, idx)
        local so, st, sv
        if rsb then so, st, sv = csr.gather_rows(rsb.o, rsb.t, rsb.v, idx) end
        local o, t, v = tag_combine(to, tt, tv, rtb.k,
          so, st, sv, rsb and rsb.k, nil, nil, nil, nil, false, h.bns, h.blk)
        local codes = h.sp:encode({ offsets = o, tokens = t, values = v, n_samples = idx:size() })
        local _, cls = h.r:label(codes, idx:size(), 1)
        return cls
      end,
    })
  end

  local function zeros (n) local z = ivec.create(n); z:zero(); return z end

  -- ===== Stage 1: TAG (per-token type, O + N_TYPES classes; argmax) =====
  local tr_types, dv_types, te_types
  do
    local fold = build_fold(tr_off, tr_n, cfg.stack.k)
    local tg_tr = csr.bio_token_type(tr_lnbr, N_TYPES)   -- per-token class: type 0..N_TYPES-1, or O=N_TYPES
    local tg_dv = csr.bio_token_type(dv_lnbr, N_TYPES)
    local tr_tgoff = ivec.create(tr_n + 1); tr_tgoff:fill_indices()
    local dv_tgoff = ivec.create(dv_n + 1); dv_tgoff:fill_indices()
    -- TEXT block (per token: focus over each token within its sentence)
    local ng, e_off, e_tok, e_val, e_ntok = tokenize(train, tr_off, tr_s, tr_e, tr_ty, nil)
    local raw_o, raw_t, raw_v = e_off, e_tok, fvec.create(e_val)  -- pristine for OOF
    local _, dvo, dvt, dvv = tokenize(dev, dv_off, dv_s, dv_e, dv_ty, ng)
    -- SHAPE block (stage-1 experiment): per token, focus over the token in the doc's shape skeleton
    local sng, se_ntok, raw_so, raw_st, raw_sv, dse_o, dse_t, dse_v
    local sb, dsb
    if cfg.exp.tag_shape then
      local se_off, se_tok, se_val
      sng, se_off, se_tok, se_val, se_ntok = tokenize_shape(train, tr_off, tr_s, tr_e, tr_off, tr_s, tr_e, nil)
      raw_so, raw_st, raw_sv = se_off, se_tok, fvec.create(se_val)
      _, dse_o, dse_t, dse_v = tokenize_shape(dev, dv_off, dv_s, dv_e, dv_off, dv_s, dv_e, sng)
      sb = { o = se_off, t = se_tok, v = se_val, k = se_ntok }
      dsb = { o = dse_o, t = dse_t, v = dse_v, k = se_ntok }
    end
    local sp, rg, bns, blk, best = train_head("Tag", cfg.tag,
      { o = e_off, t = e_tok, v = e_val, k = e_ntok }, sb, tr_n, N_TYPES + 1, tr_tgoff, tg_tr,
      { o = dvo, t = dvt, v = dvv, k = e_ntok }, dsb, dv_n, dv_tgoff, tg_dv)
    -- predict dev/test: fresh tokenizations (train_head mutated the val blocks)
    local _, ddo, ddt, ddv = tokenize(dev, dv_off, dv_s, dv_e, dv_ty, ng)
    local _, teo, tet, tev = tokenize(test_set, te_off, te_s, te_e, te_ty, ng)
    local dds, tes
    if cfg.exp.tag_shape then
      local _, do_, dt_, dv_ = tokenize_shape(dev, dv_off, dv_s, dv_e, dv_off, dv_s, dv_e, sng)
      local _, to_, tt_, tv_ = tokenize_shape(test_set, te_off, te_s, te_e, te_off, te_s, te_e, sng)
      dds = { o = do_, t = dt_, v = dv_, k = se_ntok }
      tes = { o = to_, t = tt_, v = tv_, k = se_ntok }
    end
    dv_types = head_argmax(sp, rg, bns, blk, { o = ddo, t = ddt, v = ddv, k = e_ntok }, dds, dv_n)
    te_types = head_argmax(sp, rg, bns, blk, { o = teo, t = tet, v = tev, k = e_ntok }, tes, te_n)
    local rsb = cfg.exp.tag_shape and { o = raw_so, t = raw_st, v = raw_sv, k = se_ntok } or nil
    tr_types = oof_head_argmax("Tag", best, N_TYPES + 1,
      { o = raw_o, t = raw_t, v = raw_v, k = e_ntok }, rsb, tg_tr, tr_n, fold)
  end

  -- enumerate candidate spans: every 1..max_span subspan within each contiguous non-O run
  local en_tro, en_trs, en_tre = csr.enumerate_subspans(tr_off, tr_s, tr_e, tr_types, cfg.tag.max_span, N_TYPES)
  local en_dvo, en_dvs, en_dve = csr.enumerate_subspans(dv_off, dv_s, dv_e, dv_types, cfg.tag.max_span, N_TYPES)
  local en_teo, en_tes, en_tee = csr.enumerate_subspans(te_off, te_s, te_e, te_types, cfg.tag.max_span, N_TYPES)

  -- union the enumerated spans with gazetteer surface matches, deduped on (start,end); both untyped
  local ac = build_gazetteer(train)
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

  -- experiment (a): per-candidate predicted type for typed focus (OOF tags on train, predicted on dev/test)
  local tr_fty = csr.candidate_focus_types(tr_co, tr_cs, tr_ce, tr_off, tr_s, tr_e, tr_types, N_TYPES)
  local dv_fty = csr.candidate_focus_types(dv_co, dv_cs, dv_ce, dv_off, dv_s, dv_e, dv_types, N_TYPES)
  local te_fty = csr.candidate_focus_types(te_co, te_cs, te_ce, te_off, te_s, te_e, te_types, N_TYPES)

  -- per-candidate class: gold type if (start,end) matches a gold span, else REJECT (= N_TYPES)
  local tr_tlab = csr.type_labels(tr_co, tr_cs, tr_ce, tr_eoff, tr_es, tr_ee, tr_ety, N_TYPES)
  local dv_tlab = csr.type_labels(dv_co, dv_cs, dv_ce, dv_eoff, dv_es, dv_ee, dv_ety, N_TYPES)
  local tr_tloff = ivec.create(n_trc + 1); tr_tloff:fill_indices()
  local dv_tloff = ivec.create(n_dvc + 1); dv_tloff:fill_indices()

  -- candidate (start,end) coverage, type-agnostic (covered + wrong_type, since candidate types are 0);
  -- under/over/none are boundary geometry, under_by_type uses real gold types
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

  -- ===== Stage 2: TYPE (per-span type-or-reject; argmax) =====
  -- Features per candidate = block 1 TEXT (focus) + block 2 the tag's per-token type signature
  -- (stream=type) + [exp (c): block SHAPE, per-token shape skeleton] + the typed-gazetteer prior
  -- P(type|surface). The tokenized blocks are BNS'd; gaz (continuous) is not. All merged and per-block
  -- scaled (housing pattern). Train type-context uses OOF tag types; gaz uses leave-one-out so a train
  -- mention can't read off its own answer.
  -- BNS'd blocks: 1 TEXT, 2 type-context, (3 SHAPE iff cfg.exp.shape); then gaz (not BNS'd).
  -- ty_nt1/ty_nt2 = block-1/block-2 ends; ty_ntok = end of the BNS'd group (== ty_nt2 if no shape).
  local ty_ng1, ty_ng2, ty_ng3, ty_nt1, ty_nt2, ty_ntok, ty_bns, ty_block
  local function type_feats (split, co, cs, ce, n, toff, ts, te, ttypes, fty, is_train)
    local m1, o1, t1, v1, k1
    if cfg.exp.typed_focus then
      m1, o1, t1, v1, k1 = tokenize_typed(split, co, cs, ce, fty, ty_ng1)
    else
      m1, o1, t1, v1, k1 = tokenize(split, co, cs, ce, zeros(n), ty_ng1)
    end
    local m2, o2, t2, v2, k2 = tokenize_ctx(split, co, cs, ce, zeros(n), toff, ts, te, ttypes, ty_ng2, "context")
    local o, t, v = csr.merge(o1, t1, v1, o2, t2, v2, k1)
    local k3 = 0
    if cfg.exp.shape then
      local m3, o3, t3, v3, kk3 = tokenize_shape(split, co, cs, ce, toff, ts, te, ty_ng3)
      o, t, v = csr.merge(o, t, v, o3, t3, v3, k1 + k2)
      k3 = kk3
      if is_train then ty_ng3 = m3 end
    end
    if is_train then ty_ng1, ty_ng2, ty_nt1, ty_nt2, ty_ntok = m1, m2, k1, k1 + k2, k1 + k2 + k3 end
    return o, t, v
  end
  -- block 3: per-candidate P(type|surface) over N_TYPES columns (empty if the surface is unseen in train).
  -- tlab (train only) drives leave-one-out: drop the candidate's own gold occurrence from the counts.
  local gaz_counts = build_typed_gaz(train)
  local function gaz_block (split, co, cs, ce, tlab)
    local off, tok, val = ivec.create(), ivec.create(), fvec.create()
    off:push(0)
    local nd = co:size() - 1
    for d = 1, nd do
      local text = split.texts[d]
      for i = co:get(d - 1), co:get(d) - 1 do
        local c = gaz_counts[text:sub(cs:get(i) + 1, ce:get(i)):lower()]
        if c then
          local g = tlab and tlab:get(i) or N_TYPES
          local den = c.total - (g < N_TYPES and 1 or 0)
          for ty = 0, N_TYPES - 1 do
            local cnt = c[ty] - (ty == g and 1 or 0)
            if cnt > 0 and den > 0 then tok:push(ty); val:push(cnt / den) end
          end
        end
        off:push(tok:size())
      end
    end
    return off, tok, val
  end
  local ty_off, ty_tok, ty_val = type_feats(train, tr_co, tr_cs, tr_ce, n_trc, tr_off, tr_s, tr_e, tr_types, tr_fty, true)
  ty_bns = csr.apply_bns(ty_off, ty_tok, ty_val, nil, tr_tloff, tr_tlab, ty_ntok, N_TYPES + 1)
  local g_off, g_tok, g_val = gaz_block(train, tr_co, tr_cs, tr_ce, tr_tlab)
  ty_off, ty_tok, ty_val = csr.merge(ty_off, ty_tok, ty_val, g_off, g_tok, g_val, ty_ntok)
  local ty_ntok_all = ty_ntok + N_TYPES
  -- per-block scale (housing pattern): TEXT, type-context, SHAPE (empty if off), gaz.
  local ss = csr.block_sumsq(ty_tok, ty_val, { 0, ty_nt1, ty_nt2, ty_ntok, ty_ntok_all })
  local ss0, ss1, ss2, ss3 = ss:get(0), ss:get(1), ss:get(2), ss:get(3)
  ty_block = fvec.create(ty_ntok_all)
  ty_block:fill(ss0 > 0 and math.sqrt(n_trc / ss0) or 0.0, 0, ty_nt1)
  ty_block:fill(ss1 > 0 and math.sqrt(n_trc / ss1) or 0.0, ty_nt1, ty_nt2)
  ty_block:fill(ss2 > 0 and math.sqrt(n_trc / ss2) or 0.0, ty_nt2, ty_ntok)
  ty_block:fill(ss3 > 0 and math.sqrt(n_trc / ss3) or 0.0, ty_ntok, ty_ntok_all)
  csr.standardize(ty_off, ty_tok, ty_val, ty_block)
  csr.normalize(ty_off, ty_val)
  local function ty_apply (split, co, cs, ce, n, toff, ts, te, ttypes, fty)
    local off, tok, val = type_feats(split, co, cs, ce, n, toff, ts, te, ttypes, fty, false)
    csr.apply_bns(off, tok, val, ty_bns)
    local go, gt, gv = gaz_block(split, co, cs, ce, nil)
    off, tok, val = csr.merge(off, tok, val, go, gt, gv, ty_ntok)
    csr.standardize(off, tok, val, ty_block)
    csr.normalize(off, val)
    return off, tok, val
  end
  local ty_dvo, ty_dvt, ty_dvv = ty_apply(dev, dv_co, dv_cs, dv_ce, n_dvc, dv_off, dv_s, dv_e, dv_types, dv_fty)

  str.printf("[Type] Encoding\n")
  local sp_ty, ridge_ty, _, ty_best = optimize.krr({
    offsets = ty_off, tokens = ty_tok, values = ty_val, n_samples = n_trc, n_tokens = ty_ntok_all,
    kernel = cfg.type.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_tloff, label_neighbors = tr_tlab, n_labels = N_TYPES + 1,
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

  -- top-2 type scores per test candidate; resolve overlaps by DP (weighted interval scheduling) over
  -- non-reject candidates, weighted by the margin (argmax - runner-up). Optimal max-total-margin set.
  local ty_teo, ty_tet, ty_tev = ty_apply(test_set, te_co, te_cs, te_ce, n_tec, te_off, te_s, te_e, te_types, te_fty)
  local te_codes = sp_ty:encode({ offsets = ty_teo, tokens = ty_tet, values = ty_tev, n_samples = n_tec })
  local _, te_lab, te_sco = ridge_ty:label(te_codes, n_tec, 2)
  local keep, te_cls = csr.nms_dp(te_co, te_cs, te_ce, te_lab, te_sco, 2, N_TYPES)
  local p_off, p_s, p_e, p_ty = csr.filter_spans(te_co, te_cs, te_ce, te_cls, keep)
  local f1, p, r = eval.span_f1(p_off, p_s, p_e, p_ty, te_eoff, te_es, te_ee, te_ety)
  str.printf("[Hybrid Test] span: f1=%.4f p=%.4f r=%.4f %s\n", f1, p, r, sw())

  -- decompose the TYPE conversion loss: of golds present in the candidate pool, how many does the head
  -- get correct vs reject (false_reject) vs assign the wrong type (mistype), with a gold->pred confusion
  -- matrix. cand_pred = top-1 class per candidate (te_lab stride 2; class N_TYPES == REJECT).
  local tdr = csr.type_decode_report({
    cand_offsets = te_co, cand_starts = te_cs, cand_ends = te_ce,
    cand_pred = te_lab, pred_stride = 2,
    gold_offsets = te_eoff, gold_starts = te_es, gold_ends = te_ee, gold_types = te_ety,
    n_types = N_TYPES,
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
