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

-- Span NER as a TWO-LAYER OOF-stacked typed-IOB sequence tagger + Viterbi, to make the probe
-- apples-to-apples with the span pipeline's two-stage (TAG -> TYPE) 0.8127.
--   Layer 1: maximally-enriched 9-class KRR (text+shape + gaz-match + word-gaz), Viterbi-decoded.
--   Stacking signal: layer-1's Viterbi-decoded per-token TYPE skeleton, OOF'd on train (K-fold,
--     model refit per fold) so it's honest -- mirrors conll.lua's oof_head_argmax. The decode->span
--     ->type-skeleton is the discretization (nonlinearity) the next layer can't reconstruct from codes.
--   Layer 2: the same blocks + a CTX (types=true) block over the layer-1 skeleton; 9-class KRR, Viterbi.
-- Probe: does the stacked tagger clear 0.8127? If still ~0.79, the residual is the enumerate-and-
-- classify candidate-recall mechanism (which the per-token tagger lacks), not stacking depth.

local cfg = {
  data = { dir = "test/res/conll2003", max = nil },
  tok = { ngram_min = 3, ngram_max = 5, normalize = false },
  emb = { n_landmarks = 1024 * 8, trace_tol = 0.01 },
  -- search lambda/prop (and kernel) on dev SPAN-F1 (via a Viterbi-decoding trial_fn), not token acc.
  -- stage 1 FROZEN to the searched winner while we iterate on stage 2 (set search_trials=300 + restore
  -- the full kernel list to re-search). Locked path uses spec.def / the single kernel.
  tag = { kernel = { "expcos" },
    lambda = { min = 1e-4, max = 1e1, log = true, def = 1.1207e-03 },
    propensity_a = { min = 0, max = 8, def = 0.1181 }, propensity_b = { min = 0, max = 16, def = 1.0305 },
    search_trials = 0 },
  type = { kernel = { "cosine", "expcos", "geolaplace", "matern52", "rq", "arccos1" },   -- layer 2
    lambda = { min = 1e-4, max = 1e1, log = true },
    propensity_a = { min = 0, max = 8 }, propensity_b = { min = 0, max = 16 },
    search_trials = 100 },   -- L2 metric is near-flat; 4-way best was found by trial ~50 (was 300)
  vit = { min = 0, max = 1.5, tol = 0.01 },   -- post-hoc Viterbi-weight: golden-section on dev span-F1
  stack = { k = 5 },   -- OOF folds for the layer-1 -> layer-2 skeleton (k>=2 = honest)
  ctx = { iob = true },   -- L2 CTX skeleton: true = 9-way IOB (B/I/O preserved); false = 4-way type-only
}

local N_TYPES = 4
local NL = 2 * N_TYPES + 1   -- 9: B-t=2t, I-t=2t+1, O=8
local O_LABEL = 2 * N_TYPES
local CTX_NT = cfg.ctx.iob and NL or N_TYPES   -- L2 CTX skeleton cardinality (9-way IOB vs 4-way type)
local boundary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

local function build_tokens (split)
  local off, starts, ends = ivec.create(), ivec.create(), ivec.create()
  local eoff, es, ee, ety = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  off:push(0); eoff:push(0)
  for d = 1, split.n do
    for _, t in ipairs(split.sent_tokens[d]) do starts:push(t.s); ends:push(t.e) end
    off:push(starts:size())
    for _, e in ipairs(split.sent_ents[d]) do es:push(e.s); ee:push(e.e); ety:push(e.t) end
    eoff:push(es:size())
  end
  return off, starts, ends, starts:size(), eoff, es, ee, ety
end

local function build_iob_labels (split)
  local lab = ivec.create()
  for d = 1, split.n do
    local toks = split.sent_tokens[d]
    local n = #toks
    local tl = {}
    for i = 1, n do tl[i] = O_LABEL end
    for _, e in ipairs(split.sent_ents[d]) do
      local started = false
      for i = 1, n do
        if toks[i].s >= e.s and toks[i].e <= e.e then
          tl[i] = started and (2 * e.t + 1) or (2 * e.t)
          started = true
        end
      end
    end
    for i = 1, n do lab:push(tl[i]) end
  end
  return lab
end

-- gazetteer = aho automaton over distinct train entity surfaces; typed gaz = P(type|surface) counts.
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

local function build_typed_gaz (train)
  local gaz = {}
  for d = 1, train.n do
    local text = train.texts[d]
    for _, e in ipairs(train.sent_ents[d]) do
      local surf = text:sub(e.s + 1, e.e):lower()
      local c = gaz[surf]
      if not c then c = { total = 0 }; for ty = 0, N_TYPES - 1 do c[ty] = 0 end; gaz[surf] = c end
      c[e.t] = c[e.t] + 1; c.total = c.total + 1
    end
  end
  return gaz
end

-- per-WORD IOB-class counts over ALL train tokens (entity AND non-entity), for P(IOB|word). Unlike
-- the type-only version this carries the boundary role (B vs I) and the O base rate of each word.
local function build_word_gaz (train)
  local gaz = {}
  for d = 1, train.n do
    local toks = train.sent_tokens[d]
    local n = #toks
    local tl = {}
    for i = 1, n do tl[i] = O_LABEL end
    for _, e in ipairs(train.sent_ents[d]) do
      local started = false
      for i = 1, n do
        if toks[i].s >= e.s and toks[i].e <= e.e then
          tl[i] = started and (2 * e.t + 1) or (2 * e.t); started = true
        end
      end
    end
    local text = train.texts[d]
    for i = 1, n do
      local w = text:sub(toks[i].s + 1, toks[i].e):lower()
      local c = gaz[w]
      if not c then c = { total = 0 }; for k = 0, NL - 1 do c[k] = 0 end; gaz[w] = c end
      c[tl[i]] = c[tl[i]] + 1; c.total = c.total + 1
    end
  end
  return gaz
end

local function tokenize (split, off, starts, ends, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({ ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      terminals = true, focus = true, normalize = cfg.tok.normalize })
  end
  local o, t, v = tok:tokenize({ texts = split.texts, n_samples = split.n,
    focus = { offsets = off, starts = starts, ends = ends }, grow = grow })
  return tok, o, t, v, tok:n_tokens()
end

local function tokenize_shape (split, off, starts, ends, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({ ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      terminals = true, focus = true, shapes = true })
  end
  local o, t, v = tok:tokenize({ texts = split.texts, n_samples = split.n,
    focus = { offsets = off, starts = starts, ends = ends }, grow = grow })
  return tok, o, t, v, tok:n_tokens()
end

-- CTX block (types=true): per-token focus over the layer-1 decoded per-token skeleton (ttypes in
-- [0, n_types)). n_types = NL for the 9-way IOB skeleton (B/I/O preserved) or N_TYPES for type-only.
local function tokenize_ctx (split, off, starts, ends, ttypes, n_types, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({ ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = n_types, terminals = true, focus = true, types = true })
  end
  local o, t, v = tok:tokenize({ texts = split.texts, n_samples = split.n,
    focus = { offsets = off, starts = starts, ends = ends },
    types = { offsets = off, starts = starts, ends = ends, types = ttypes }, grow = grow })
  return tok, o, t, v, tok:n_tokens()
end

local function bio_to_spans (off, starts, ends, lab)
  local p_off, p_s, p_e, p_ty = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  p_off:push(0)
  local nd = off:size() - 1
  for d = 0, nd - 1 do
    local s, e = off:get(d), off:get(d + 1)
    local i = s
    while i < e do
      local c = lab:get(i)
      if c < O_LABEL and c % 2 == 0 then
        local t = math.floor(c / 2)
        local sp_s = starts:get(i)
        local j = i + 1
        while j < e and lab:get(j) == 2 * t + 1 do j = j + 1 end
        p_s:push(sp_s); p_e:push(ends:get(j - 1)); p_ty:push(t)
        i = j
      else
        i = i + 1
      end
    end
    p_off:push(p_s:size())
  end
  return p_off, p_s, p_e, p_ty
end

-- L2 CTX skeleton from per-token IOB labels (0..NL-1). cfg.ctx.iob: keep the full 9-way IOB labels
-- (B/I/O preserved -> injects L1's exact boundary decisions); else collapse to 4-way type (O = N_TYPES).
local function ctx_skeleton (iob, n)
  if cfg.ctx.iob then return iob end
  local ty = ivec.create(n)
  for i = 0, n - 1 do
    local c = iob:get(i)
    ty:set(i, c < O_LABEL and math.floor(c / 2) or N_TYPES)
  end
  return ty
end

test("conll-iob", function ()

  local stopwatch = utc.stopwatch()
  local function sw () local d, dd = stopwatch(); return str.format("(%.1fs +%.1fs)", d, dd) end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  str.printf("[Data] train=%d dev=%d test=%d %s\n", train.n, dev.n, test_set.n, sw())

  local tr_off, tr_s, tr_e, tr_n, tr_eoff, tr_es, tr_ee, tr_ety = build_tokens(train)
  local dv_off, dv_s, dv_e, dv_n, dv_eoff, dv_es, dv_ee, dv_ety = build_tokens(dev)
  local te_off, te_s, te_e, te_n, te_eoff, te_es, te_ee, te_ety = build_tokens(test_set)
  local tr_iob = build_iob_labels(train)
  local dv_iob = build_iob_labels(dev)
  local tr_loff = ivec.create(tr_n + 1); tr_loff:fill_indices()
  local dv_loff = ivec.create(dv_n + 1); dv_loff:fill_indices()

  local ac = build_gazetteer(train)
  local gaz_counts = build_typed_gaz(train)
  local word_gaz = build_word_gaz(train)

  -- per-token P(type | gazetteer-matched surface), placed in the B-slot for the match's first token
  -- and the I-slot for the rest (the longest-match gives the boundary structure). 8 cols, LOO on train.
  local function gaz_match_block (split, t_off, t_s, t_e, lab)
    local g_off, _, g_s, g_e = ac:predict({ texts = split.texts, longest = true, boundary = boundary })
    local off, tok, val = ivec.create(), ivec.create(), fvec.create()
    off:push(0)
    local nd = t_off:size() - 1
    for d = 0, nd - 1 do
      local text = split.texts[d + 1]
      local cov = {}
      for gi = g_off:get(d), g_off:get(d + 1) - 1 do
        local c = gaz_counts[text:sub(g_s:get(gi) + 1, g_e:get(gi)):lower()]
        if c then
          local first = true
          for ti = t_off:get(d), t_off:get(d + 1) - 1 do
            if t_s:get(ti) >= g_s:get(gi) and t_e:get(ti) <= g_e:get(gi) then
              cov[ti] = { c = c, first = first }; first = false
            end
          end
        end
      end
      for ti = t_off:get(d), t_off:get(d + 1) - 1 do
        local m = cov[ti]
        if m then
          local g = (lab and lab:get(ti) < O_LABEL) and math.floor(lab:get(ti) / 2) or N_TYPES
          local den = m.c.total - (g < N_TYPES and 1 or 0)
          for ty = 0, N_TYPES - 1 do
            local cnt = m.c[ty] - (ty == g and 1 or 0)
            if cnt > 0 and den > 0 then
              tok:push(m.first and (2 * ty) or (2 * ty + 1)); val:push(cnt / den)
            end
          end
        end
        off:push(tok:size())
      end
    end
    return off, tok, val
  end

  -- per-token P(IOB-class | word) over all occurrences (incl O base rate + B/I role). 9 cols, LOO.
  local function word_gaz_block (split, t_off, t_s, t_e, lab)
    local off, tok, val = ivec.create(), ivec.create(), fvec.create()
    off:push(0)
    local nd = t_off:size() - 1
    for d = 1, nd do
      local text = split.texts[d]
      for i = t_off:get(d - 1), t_off:get(d) - 1 do
        local c = word_gaz[text:sub(t_s:get(i) + 1, t_e:get(i)):lower()]
        if c then
          local g = lab and lab:get(i)          -- this token's own IOB class, for LOO
          local den = c.total - (g and 1 or 0)
          if den > 0 then
            for k = 0, NL - 1 do
              local cnt = c[k] - ((g == k) and 1 or 0)
              if cnt > 0 then tok:push(k); val:push(cnt / den) end
            end
          end
        end
        off:push(tok:size())
      end
    end
    return off, tok, val
  end

  -- ===== feature assembly =====
  -- Two halves, kept SEPARATE until after BNS (apply_bns sizes its tables to the BNS boundary and
  -- indexes by token id, so it must only ever see tokenize columns -- merging gaz first overflows it):
  --   tok_feats = text + shape [+ CTX skeleton], the BNS'd discrete group ([0, nt_ctx)).
  --   gaz_feats = gaz-match (2*N_TYPES) + word-gaz (NL), continuous, columns [0, 2*N_TYPES + NL).
  -- finalize BNS's the tokenize CSR, merges gaz at nt_ctx, then per-block scale + L2-normalize the whole.
  -- State `st` holds the tokenizer vocabs + bounds (one per layer). The OOF gathers the two raw halves and
  -- re-finalizes per fold, so BNS still only sees tokenize columns.
  local function tok_feats (st, split, t_off, t_s, t_e, ctx_types, is_train)
    local m1, o1, x1, v1, k1 = tokenize(split, t_off, t_s, t_e, st.ng1)
    local m3, o3, x3, v3, k3 = tokenize_shape(split, t_off, t_s, t_e, st.ng3)
    local o, x, v = csr.merge(o1, x1, v1, o3, x3, v3, k1)
    local base = k1 + k3
    if ctx_types ~= nil then
      local mc, oc, xc, vc, kc = tokenize_ctx(split, t_off, t_s, t_e, ctx_types, CTX_NT, st.ng_ctx)
      o, x, v = csr.merge(o, x, v, oc, xc, vc, base)
      if is_train then st.ng_ctx = mc end
      base = base + kc
    end
    if is_train then
      st.ng1, st.ng3 = m1, m3
      st.nt1, st.nt_ts, st.nt_ctx = k1, k1 + k3, base                     -- text end; +shape; +ctx (=gaz start)
      st.ntok_all = base + 2 * N_TYPES + NL
    end
    return o, x, v
  end
  local function gaz_feats (split, t_off, t_s, t_e, lab)   -- lab = LOO on train, nil on dev/test
    local go, gt2, gv = gaz_match_block(split, t_off, t_s, t_e, lab)
    local wo, wt, wv = word_gaz_block(split, t_off, t_s, t_e, lab)
    return csr.merge(go, gt2, gv, wo, wt, wv, 2 * N_TYPES)                 -- word-gaz shifted past gaz-match
  end
  -- BNS the tokenize CSR (fit if bns_in nil, else apply), merge gaz at nt_ctx, block-scale + normalize.
  local function finalize (st, to_, tx_, tv_, go_, gx_, gv_, n, loff, lab, bns_in, block_in)
    local bns = bns_in
    if bns_in == nil then bns = csr.apply_bns(to_, tx_, tv_, nil, loff, lab, st.nt_ctx, NL)
    else csr.apply_bns(to_, tx_, tv_, bns_in) end
    local o, x, v = csr.merge(to_, tx_, tv_, go_, gx_, gv_, st.nt_ctx)
    local bl = block_in
    if block_in == nil then
      local bounds = { 0, st.nt1, st.nt_ts, st.nt_ctx, st.nt_ctx + 2 * N_TYPES, st.ntok_all }
      local ss = csr.block_sumsq(x, v, bounds)
      bl = fvec.create(st.ntok_all)
      for r = 0, #bounds - 2 do
        local ssr = ss:get(r)
        bl:fill(ssr > 0 and math.sqrt(n / ssr) or 0.0, bounds[r + 1], bounds[r + 2])
      end
    end
    csr.standardize(o, x, v, bl); csr.normalize(o, v)
    return o, x, v, bns, bl
  end

  -- ===== Layer 1: enriched 9-class tagger =====
  local L1 = {}
  -- pristine pre-BNS raw halves (tokenize + gaz), kept for the OOF gather/refit
  local raw_ts_o, raw_ts_x, raw_ts_v, raw_gz_o, raw_gz_x, raw_gz_v
  local to, tx, tv, l1_bns, l1_block
  do
    local tso, tsx, tsv = tok_feats(L1, train, tr_off, tr_s, tr_e, nil, true)
    local gzo, gzx, gzv = gaz_feats(train, tr_off, tr_s, tr_e, tr_iob)
    raw_ts_o, raw_ts_x, raw_ts_v = tso, tsx, fvec.create(tsv)              -- BNS mutates only values
    raw_gz_o, raw_gz_x, raw_gz_v = gzo, gzx, fvec.create(gzv)
    to, tx, tv, l1_bns, l1_block = finalize(L1, tso, tsx, tsv, gzo, gzx, gzv, tr_n, tr_loff, tr_iob, nil, nil)
  end
  local ntok_all = L1.ntok_all
  local dvo, dvx, dvv
  do
    local tso, tsx, tsv = tok_feats(L1, dev, dv_off, dv_s, dv_e, nil, false)
    local gzo, gzx, gzv = gaz_feats(dev, dv_off, dv_s, dv_e, nil)
    dvo, dvx, dvv = finalize(L1, tso, tsx, tsv, gzo, gzx, gzv, dv_n, nil, nil, l1_bns, l1_block)
  end

  local vit = optimize.viterbi({ train_offsets = tr_off, train_neighbors = tr_iob, n_labels = NL })

  str.printf("[IOB] Encoding (n_tokens=%d)\n", ntok_all)
  -- select kernel/lambda/prop on dev per-token accuracy (label_accuracy); w via a post-hoc sweep.
  local sp, rg, _, best = optimize.krr({
    offsets = to, tokens = tx, values = tv, n_samples = tr_n, n_tokens = ntok_all,
    kernel = cfg.tag.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_loff, label_neighbors = tr_iob, n_labels = NL,
    val_offsets = dvo, val_tokens = dvx, val_values = dvv, val_n_samples = dv_n,
    val_expected_offsets = dv_loff, val_expected_neighbors = dv_iob,
    lambda = cfg.tag.lambda, propensity_a = cfg.tag.propensity_a, propensity_b = cfg.tag.propensity_b,
    k = 1, search_trials = cfg.tag.search_trials, each = util.make_ridge_log(stopwatch),
    trial_fn = function (gram, params)
      local f1, p, r = gram:label_accuracy(params.lambda, 1, params.propensity_a, params.propensity_b,
        dv_loff, dv_iob, 1)
      return f1, { f1 = f1, precision = p, recall = r }
    end,
  })
  str.printf("[IOB] kernel=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
    best.kernel, best.lambda, best.propensity_a, best.propensity_b, sw())

  -- encode+regress a split's emissions for a given layer (st/encoder/regressor/normalization, optional ctx).
  local function emit_for (st, encoder, regr, bns_, block_, split, t_off, t_s, t_e, n, ctx)
    local tso, tsx, tsv = tok_feats(st, split, t_off, t_s, t_e, ctx, false)
    local gzo, gzx, gzv = gaz_feats(split, t_off, t_s, t_e, nil)
    local o, x, v = finalize(st, tso, tsx, tsv, gzo, gzx, gzv, n, nil, nil, bns_, block_)
    return regr:regress(encoder:encode({ offsets = o, tokens = x, values = v, n_samples = n }), n)
  end
  -- post-hoc Viterbi-weight via golden-section on dev span-F1 (caller owns BIO->spans + span-F1).
  local function best_weight (tag, dv_emit)
    return vit:select_w(dv_off, dv_emit, dv_n, cfg.vit, function (lab)
      local po, ps, pe, pt = bio_to_spans(dv_off, dv_s, dv_e, lab)
      return eval.span_f1(po, ps, pe, pt, dv_eoff, dv_es, dv_ee, dv_ety)
    end, function (w, f1) str.printf("[%s w] w=%.4f dev_f1=%.4f %s\n", tag, w, f1, sw()) end)
  end
  local function report (tag, te_emit, w)
    local te_lab = vit:decode(te_off, te_emit, te_n, w)
    local po, ps, pe, pt = bio_to_spans(te_off, te_s, te_e, te_lab)
    local f1, p, r = eval.span_f1(po, ps, pe, pt, te_eoff, te_es, te_ee, te_ety)
    str.printf("[%s] span: f1=%.4f p=%.4f r=%.4f (w=%.2f) %s\n", tag, f1, p, r, w, sw())
  end

  local dv_emit = emit_for(L1, sp, rg, l1_bns, l1_block, dev, dv_off, dv_s, dv_e, dv_n)
  local te_emit = emit_for(L1, sp, rg, l1_bns, l1_block, test_set, te_off, te_s, te_e, te_n)
  local best_w, best_dev = best_weight("IOB", dv_emit)
  str.printf("[IOB] best dev w=%.4f f1=%.4f\n", best_w, best_dev)
  report("IOB Test", te_emit, best_w)

  -- ===== OOF layer-1 TYPE skeleton on train (honest stacking signal) =====
  -- K-fold at the DOCUMENT level: refit the (locked) layer-1 KRR on the other folds, Viterbi-decode
  -- the held-out fold at best_w. Mirrors conll.lua's oof_head_argmax (model refit per fold). NOTE: the
  -- gaz tables stay full-train + per-token LOO (same leakage tolerance as the span pipeline's surf_gaz);
  -- the [OOF train] span-F1 below vs [IOB dev] flags any residual gaz inflation of the skeleton.
  local tr_oof_iob = ivec.create(tr_n); tr_oof_iob:zero()   -- OOF IOB labels (the train CTX skeleton + diagnostic)
  do
    local K = cfg.stack.k
    local nd = tr_off:size() - 1
    for f = 0, K - 1 do
      local tr_idx, ev_idx = ivec.create(), ivec.create()
      local ev_off, ev_glob = ivec.create(), {}
      ev_off:push(0)
      for d = 0, nd - 1 do
        local s, e = tr_off:get(d), tr_off:get(d + 1)
        if d % K == f then
          for g = s, e - 1 do ev_idx:push(g); ev_glob[#ev_glob + 1] = g end
          ev_off:push(ev_idx:size())
        else
          for g = s, e - 1 do tr_idx:push(g) end
        end
      end
      str.printf("[IOB OOF] fold %d/%d train=%d eval=%d %s\n", f + 1, K, tr_idx:size(), ev_idx:size(), sw())
      local fts_o, fts_x, fts_v = csr.gather_rows(raw_ts_o, raw_ts_x, raw_ts_v, tr_idx)
      local fgz_o, fgz_x, fgz_v = csr.gather_rows(raw_gz_o, raw_gz_x, raw_gz_v, tr_idx)
      local fl = ivec.create(); fl:copy(tr_iob, tr_idx)
      local floff = ivec.create(tr_idx:size() + 1); floff:fill_indices()
      local fo, fx, fv, fbns, fblk = finalize(L1, fts_o, fts_x, fts_v, fgz_o, fgz_x, fgz_v,
        tr_idx:size(), floff, fl, nil, nil)
      local _, fsp, fgram = spectral.encode({
        offsets = fo, tokens = fx, values = fv, n_tokens = ntok_all, n_samples = tr_idx:size(),
        kernel = best.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
        label_offsets = floff, label_neighbors = fl, n_labels = NL,
        solve_lambda = best.lambda,
        solve_propensity_a = best.propensity_a, solve_propensity_b = best.propensity_b,
      })
      local frg = ridge.create({ gram = fgram })
      local ets_o, ets_x, ets_v = csr.gather_rows(raw_ts_o, raw_ts_x, raw_ts_v, ev_idx)
      local egz_o, egz_x, egz_v = csr.gather_rows(raw_gz_o, raw_gz_x, raw_gz_v, ev_idx)
      local eo, ex, ev = finalize(L1, ets_o, ets_x, ets_v, egz_o, egz_x, egz_v,
        ev_idx:size(), nil, nil, fbns, fblk)
      local emit = frg:regress(fsp:encode({ offsets = eo, tokens = ex, values = ev, n_samples = ev_idx:size() }),
        ev_idx:size())
      local elab = vit:decode(ev_off, emit, ev_idx:size(), best_w)
      for i = 0, ev_idx:size() - 1 do tr_oof_iob:set(ev_glob[i + 1], elab:get(i)) end
      collectgarbage("collect")
    end
  end
  -- honesty check: OOF-train span-F1 vs dev. A big gap = the full-train gaz tables inflate the train
  -- skeleton (would make layer 2 over-trust CTX on train); ~equal = the OOF skeleton transfers.
  do
    local po, ps, pe, pt = bio_to_spans(tr_off, tr_s, tr_e, tr_oof_iob)
    local f1 = eval.span_f1(po, ps, pe, pt, tr_eoff, tr_es, tr_ee, tr_ety)
    str.printf("[IOB OOF] train skeleton span_f1=%.4f (vs dev %.4f) %s\n", f1, best_dev, sw())
  end

  -- layer-1 skeletons: train = OOF (honest), dev/test = full-train model. cfg.ctx.iob picks 9-way vs 4-way.
  local tr_ctx = ctx_skeleton(tr_oof_iob, tr_n)
  local dv_ctx = ctx_skeleton(vit:decode(dv_off, dv_emit, dv_n, best_w), dv_n)
  local te_ctx = ctx_skeleton(vit:decode(te_off, te_emit, te_n, best_w), te_n)

  -- ===== Layer 2: same blocks + CTX (types=true over the layer-1 skeleton) =====
  local L2 = {}
  local l2_to, l2_tx, l2_tv, l2_bns, l2_block
  do
    local tso, tsx, tsv = tok_feats(L2, train, tr_off, tr_s, tr_e, tr_ctx, true)
    local gzo, gzx, gzv = gaz_feats(train, tr_off, tr_s, tr_e, tr_iob)
    l2_to, l2_tx, l2_tv, l2_bns, l2_block = finalize(L2, tso, tsx, tsv, gzo, gzx, gzv, tr_n, tr_loff, tr_iob, nil, nil)
  end
  local l2_ntok_all = L2.ntok_all
  local l2_dvo, l2_dvx, l2_dvv
  do
    local tso, tsx, tsv = tok_feats(L2, dev, dv_off, dv_s, dv_e, dv_ctx, false)
    local gzo, gzx, gzv = gaz_feats(dev, dv_off, dv_s, dv_e, nil)
    l2_dvo, l2_dvx, l2_dvv = finalize(L2, tso, tsx, tsv, gzo, gzx, gzv, dv_n, nil, nil, l2_bns, l2_block)
  end

  str.printf("[IOB-L2] Encoding (n_tokens=%d)\n", l2_ntok_all)
  local sp2, rg2, _, best2 = optimize.krr({
    offsets = l2_to, tokens = l2_tx, values = l2_tv, n_samples = tr_n, n_tokens = l2_ntok_all,
    kernel = cfg.type.kernel, n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_loff, label_neighbors = tr_iob, n_labels = NL,
    val_offsets = l2_dvo, val_tokens = l2_dvx, val_values = l2_dvv, val_n_samples = dv_n,
    val_expected_offsets = dv_loff, val_expected_neighbors = dv_iob,
    lambda = cfg.type.lambda, propensity_a = cfg.type.propensity_a, propensity_b = cfg.type.propensity_b,
    k = 1, search_trials = cfg.type.search_trials, each = util.make_ridge_log(stopwatch),
    trial_fn = function (gram, params)
      local f1, p, r = gram:label_accuracy(params.lambda, 1, params.propensity_a, params.propensity_b,
        dv_loff, dv_iob, 1)
      return f1, { f1 = f1, precision = p, recall = r }
    end,
  })
  str.printf("[IOB-L2] kernel=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
    best2.kernel, best2.lambda, best2.propensity_a, best2.propensity_b, sw())

  local dv_emit2 = emit_for(L2, sp2, rg2, l2_bns, l2_block, dev, dv_off, dv_s, dv_e, dv_n, dv_ctx)
  local te_emit2 = emit_for(L2, sp2, rg2, l2_bns, l2_block, test_set, te_off, te_s, te_e, te_n, te_ctx)
  local best_w2, best_dev2 = best_weight("IOB-L2", dv_emit2)
  str.printf("[IOB-L2] best dev w=%.4f f1=%.4f\n", best_w2, best_dev2)
  report("IOB-L2 Test", te_emit2, best_w2)

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
