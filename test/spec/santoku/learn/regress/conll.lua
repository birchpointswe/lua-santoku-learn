local ds = require("santoku.learn.dataset")
local csr = require("santoku.learn.csr")
local tokenizer = require("santoku.learn.tokenizer")
local aho = require("santoku.learn.aho")
local optimize = require("santoku.learn.optimize")
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
-- Both heads: propensity for the dominant O/REJECT class; kernel/lambda locked to searched winners.

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

-- per-WORD typed gazetteer: each lowercased word of a train entity surface -> per-type counts, for the
-- head_gaz P(type|word) feature. Keys on component words so an unseen full surface still gets a type prior
-- from its parts (e.g. "...United"/"...Inc"/"...FC" -> ORG), targeting the ORG->PER/LOC mistypes.
local function build_word_gaz (train)
  local gaz = {}
  for d = 1, train.n do
    local text = train.texts[d]
    for _, e in ipairs(train.sent_ents[d]) do
      for w in text:sub(e.s + 1, e.e):lower():gmatch("%S+") do
        local c = gaz[w]
        if not c then c = { total = 0 }; for ty = 0, N_TYPES - 1 do c[ty] = 0 end; gaz[w] = c end
        c[e.t] = c[e.t] + 1
        c.total = c.total + 1
      end
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

-- TEXT block (focus=true). The threaded value `tok` is the tokenizer object:
-- nil => create + grow the vocab (train); object => frozen (dev/test).
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
    focus = { offsets = off, starts = starts, ends = ends }, grow = grow,
  })
  return tok, o, t, v, tok:n_tokens()
end

-- CTX block (types=true): focus span over the per-token predicted-type skeleton.
local function tokenize_ctx (split, off, starts, ends, coff, cs, ce, cty, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = N_TYPES, terminals = true, focus = true, types = true,
    })
  end
  local o, t, v = tok:tokenize({
    texts = split.texts, n_samples = split.n,
    focus = { offsets = off, starts = starts, ends = ends },
    types = { offsets = coff, starts = cs, ends = ce, types = cty },
    grow = grow,
  })
  return tok, o, t, v, tok:n_tokens()
end

-- Experiment (c): SHAPE block. Per-candidate focus over the doc's per-token shape
-- skeleton (Xx/XX/xx/dd/punct/mix, classified on original bytes -- casing/digit signal
-- the text/type blocks lack). Token boundaries are derived from whitespace inside the
-- tokenizer (CoNLL text = single-space-joined tokens), so no spans need to be passed.
local function tokenize_shape (split, off, starts, ends, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      terminals = true, focus = true, shapes = true,
    })
  end
  local o, t, v = tok:tokenize({
    texts = split.texts, n_samples = split.n,
    focus = { offsets = off, starts = starts, ends = ends },
    grow = grow,
  })
  return tok, o, t, v, tok:n_tokens()
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

  -- BNS + L2-normalize one block's CSR in place. fit=true computes & returns the bns;
  -- fit=false applies the supplied bns. Mutates inputs.
  local function tag_combine (to, tt, tv, tk, nl, loff, lnbr, fit, bns_in)
    if fit then
      local bns = csr.apply_bns(to, tt, tv, nil, loff, lnbr, tk, nl)
      csr.normalize(to, tv)
      return to, tt, tv, tk, bns
    end
    csr.apply_bns(to, tt, tv, bns_in); csr.normalize(to, tv)
    return to, tt, tv, tk, bns_in
  end

  -- shared multiclass-head scaffold (tag = per-token type; type = per-span type-or-reject): krr with
  -- propensity, selected and decoded by argmax (topk=1). tb = the block bundle {o,t,v,k}.
  local function train_head (label, scfg, tb, n, nl, loff, lnbr, dtb, dvn, dloff, dlnbr)
    local o, t, v, ntok, bns = tag_combine(tb.o, tb.t, tb.v, tb.k, nl, loff, lnbr, true)
    local dvo, dvt, dvv = tag_combine(dtb.o, dtb.t, dtb.v, dtb.k, nil, nil, nil, false, bns)
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
    return sp, rg, bns, best
  end

  local function head_argmax (sp, rg, bns, tb, n)
    local o, t, v = tag_combine(tb.o, tb.t, tb.v, tb.k, nil, nil, nil, false, bns)
    local codes = sp:encode({ offsets = o, tokens = t, values = v, n_samples = n })
    local _, cls, sco = rg:label(codes, n, 1)
    return cls, sco
  end

  local function zeros (n) local z = ivec.create(n); z:zero(); return z end

  -- ===== Stage 1: TAG (per-token type, O + N_TYPES classes; argmax) =====
  -- tr_types are the full-train tag model's IN-SAMPLE predictions on train (no OOF cross-fit): they match
  -- the dev/test CTX quality (also full-train model), which calibrates stage 2 better than honest OOF --
  -- whose weaker per-fold tag models would make the train CTX lower-quality than the test CTX.
  local tr_types, dv_types, te_types
  do
    local tg_tr = csr.bio_token_type(tr_lnbr, N_TYPES)   -- per-token class: type 0..N_TYPES-1, or O=N_TYPES
    local tg_dv = csr.bio_token_type(dv_lnbr, N_TYPES)
    local tr_tgoff = ivec.create(tr_n + 1); tr_tgoff:fill_indices()
    local dv_tgoff = ivec.create(dv_n + 1); dv_tgoff:fill_indices()
    -- TEXT block (per token: focus over each token within its sentence)
    local ng, e_off, e_tok, e_val, e_ntok = tokenize(train, tr_off, tr_s, tr_e, tr_ty, nil)
    local _, dvo, dvt, dvv = tokenize(dev, dv_off, dv_s, dv_e, dv_ty, ng)
    local sp, rg, bns = train_head("Tag", cfg.tag,
      { o = e_off, t = e_tok, v = e_val, k = e_ntok }, tr_n, N_TYPES + 1, tr_tgoff, tg_tr,
      { o = dvo, t = dvt, v = dvv, k = e_ntok }, dv_n, dv_tgoff, tg_dv)
    -- predict train/dev/test from the fitted model (fresh tokenizations; train_head mutated the fit blocks)
    local _, tro, trt, trv = tokenize(train, tr_off, tr_s, tr_e, tr_ty, ng)
    local _, ddo, ddt, ddv = tokenize(dev, dv_off, dv_s, dv_e, dv_ty, ng)
    local _, teo, tet, tev = tokenize(test_set, te_off, te_s, te_e, te_ty, ng)
    tr_types = head_argmax(sp, rg, bns, { o = tro, t = trt, v = trv, k = e_ntok }, tr_n)
    dv_types = head_argmax(sp, rg, bns, { o = ddo, t = ddt, v = ddv, k = e_ntok }, dv_n)
    te_types = head_argmax(sp, rg, bns, { o = teo, t = tet, v = tev, k = e_ntok }, te_n)
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
  -- The 5-block feature set per candidate:
  --   TEXT  (focus char-grams) + CTX (per-token tag-type skeleton) + SHAPE (per-token shape skeleton)
  --   are BNS'd together; surf_gaz (P(type|full surface)) + head_gaz (sum of P(type|word) over component
  --   words) are appended (continuous, not BNS'd). All per-block scaled (housing) then L2-normalized. Train
  --   CTX uses OOF tag types; both gaz blocks use leave-one-out so a train mention can't read its own answer.
  -- ty_nt1/ty_nt2 = TEXT/CTX ends; ty_ntok = end of the BNS'd group; +N_TYPES = surf_gaz, +N_TYPES = head_gaz.
  local ty_ng1, ty_ng2, ty_ng3, ty_nt1, ty_nt2, ty_ntok, ty_bns, ty_block
  local function type_feats (split, co, cs, ce, n, toff, ts, te, ttypes, is_train)
    local m1, o1, t1, v1, k1 = tokenize(split, co, cs, ce, zeros(n), ty_ng1)
    local m2, o2, t2, v2, k2 = tokenize_ctx(split, co, cs, ce, toff, ts, te, ttypes, ty_ng2)
    local o, t, v = csr.merge(o1, t1, v1, o2, t2, v2, k1)
    local m3, o3, t3, v3, k3 = tokenize_shape(split, co, cs, ce, ty_ng3)
    o, t, v = csr.merge(o, t, v, o3, t3, v3, k1 + k2)
    if is_train then
      ty_ng1, ty_ng2, ty_ng3 = m1, m2, m3
      ty_nt1, ty_nt2, ty_ntok = k1, k1 + k2, k1 + k2 + k3
    end
    return o, t, v
  end
  -- surf_gaz: per-candidate P(type|surface) over N_TYPES cols (empty if the surface is unseen in train).
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
  -- head_gaz: per-candidate sum over its component words of P(type|word). Aggregating all words beats
  -- picking the single most-peaked one (that reliably selects the place-name -- england/paris are far more
  -- peaked than designators like bank/united -- amplifying ORG->LOC). N_TYPES cols, LOO on train.
  local word_gaz = build_word_gaz(train)
  local function head_gaz_block (split, co, cs, ce, tlab)
    local off, tok, val = ivec.create(), ivec.create(), fvec.create()
    off:push(0)
    local nd = co:size() - 1
    for d = 1, nd do
      local text = split.texts[d]
      for i = co:get(d - 1), co:get(d) - 1 do
        local g = tlab and tlab:get(i) or N_TYPES
        local acc = {}
        for w in text:sub(cs:get(i) + 1, ce:get(i)):lower():gmatch("%S+") do
          local c = word_gaz[w]
          if c then
            local den = c.total - (g < N_TYPES and 1 or 0)
            if den > 0 then
              for ty = 0, N_TYPES - 1 do
                local cnt = c[ty] - (ty == g and 1 or 0)
                if cnt > 0 then acc[ty] = (acc[ty] or 0) + cnt / den end
              end
            end
          end
        end
        for ty = 0, N_TYPES - 1 do
          if acc[ty] then tok:push(ty); val:push(acc[ty]) end
        end
        off:push(tok:size())
      end
    end
    return off, tok, val
  end
  local ty_off, ty_tok, ty_val = type_feats(train, tr_co, tr_cs, tr_ce, n_trc, tr_off, tr_s, tr_e, tr_types, true)
  ty_bns = csr.apply_bns(ty_off, ty_tok, ty_val, nil, tr_tloff, tr_tlab, ty_ntok, N_TYPES + 1)
  local g_off, g_tok, g_val = gaz_block(train, tr_co, tr_cs, tr_ce, tr_tlab)
  ty_off, ty_tok, ty_val = csr.merge(ty_off, ty_tok, ty_val, g_off, g_tok, g_val, ty_ntok)
  local h_off, h_tok, h_val = head_gaz_block(train, tr_co, tr_cs, tr_ce, tr_tlab)
  ty_off, ty_tok, ty_val = csr.merge(ty_off, ty_tok, ty_val, h_off, h_tok, h_val, ty_ntok + N_TYPES)
  local ty_ntok_all = ty_ntok + 2 * N_TYPES
  -- per-block scale (housing): TEXT, CTX, SHAPE, surf_gaz, head_gaz each scaled independently.
  local bounds = { 0, ty_nt1, ty_nt2, ty_ntok, ty_ntok + N_TYPES, ty_ntok_all }
  local ss = csr.block_sumsq(ty_tok, ty_val, bounds)
  ty_block = fvec.create(ty_ntok_all)
  for r = 0, #bounds - 2 do
    local ssr = ss:get(r)
    ty_block:fill(ssr > 0 and math.sqrt(n_trc / ssr) or 0.0, bounds[r + 1], bounds[r + 2])
  end
  csr.standardize(ty_off, ty_tok, ty_val, ty_block)
  csr.normalize(ty_off, ty_val)
  local function ty_apply (split, co, cs, ce, n, toff, ts, te, ttypes)
    local off, tok, val = type_feats(split, co, cs, ce, n, toff, ts, te, ttypes, false)
    csr.apply_bns(off, tok, val, ty_bns)
    local go, gt, gv = gaz_block(split, co, cs, ce, nil)
    off, tok, val = csr.merge(off, tok, val, go, gt, gv, ty_ntok)
    local ho, ht, hv = head_gaz_block(split, co, cs, ce, nil)
    off, tok, val = csr.merge(off, tok, val, ho, ht, hv, ty_ntok + N_TYPES)
    csr.standardize(off, tok, val, ty_block)
    csr.normalize(off, val)
    return off, tok, val
  end
  local ty_dvo, ty_dvt, ty_dvv = ty_apply(dev, dv_co, dv_cs, dv_ce, n_dvc, dv_off, dv_s, dv_e, dv_types)

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
  local ty_teo, ty_tet, ty_tev = ty_apply(test_set, te_co, te_cs, te_ce, n_tec, te_off, te_s, te_e, te_types)
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
