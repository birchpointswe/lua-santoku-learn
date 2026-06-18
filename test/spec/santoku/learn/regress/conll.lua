local ds = require("santoku.learn.dataset")
local csr = require("santoku.learn.csr")
local tokenizer = require("santoku.learn.tokenizer")
local aho = require("santoku.learn.aho")
local segmenter = require("santoku.learn.segmenter")
local optimize = require("santoku.learn.optimize")
local ivec = require("santoku.ivec")
local fvec = require("santoku.fvec")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = {
    dir = "test/res/conll2003",
    max = nil
  },
  tok = {
    ngram_min = 3,
    ngram_max = 5,
    normalize = false
  },
  emb = {
    n_landmarks = 1024 * 8,
    trace_tol = 0.01
  },
  tag = {
    kernel = { "rbf", "arccos1", "cosine", "expcos", "geolaplace", "matern52", "rq" },
    gamma = { def = 0.1129 },
    lambda = { def = 7.8706e-04 },
    search_trials = 0
  },
  type = {
    kernel = { "expcos", "arccos1", "cosine", "geolaplace", "matern52", "rq", "rbf" },
    lambda = { def = 1.1338e-04 },
    propensity_a = { def = 0.4040 },
    propensity_b = { def = 15.5572 },
    search_trials = 0
  },
  shape = {
    n_cuts = 5
  },
}

local N_TYPES = 4
local boundary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

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

local function char_ngrams (s, nmin, nmax)
  local out, L = {}, #s
  for n = nmin, nmax do
    for i = 1, L - n + 1 do out[#out + 1] = s:sub(i, i + n - 1) end
  end
  return out
end
local function build_char_gaz (train, nmin, nmax)
  local gaz = {}
  for d = 1, train.n do
    local text = train.texts[d]
    for _, e in ipairs(train.sent_ents[d]) do
      for _, gram in ipairs(char_ngrams(text:sub(e.s + 1, e.e):lower(), nmin, nmax)) do
        local c = gaz[gram]
        if not c then c = { total = 0 }; for ty = 0, N_TYPES - 1 do c[ty] = 0 end; gaz[gram] = c end
        c[e.t] = c[e.t] + 1
        c.total = c.total + 1
      end
    end
  end
  return gaz
end

local function gold_spans (split)
  local eoff, es, ee, ety = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  eoff:push(0)
  for d = 1, split.n do
    for _, e in ipairs(split.sent_ents[d]) do es:push(e.s); ee:push(e.e); ety:push(e.t) end
    eoff:push(es:size())
  end
  return eoff, es, ee, ety
end

local function is_inner_labels (soff, sstart, send, eoff, es, ee, n)
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
  local eoff, es, ee, ety = gold_spans(split)
  local ioff, inbr = is_inner_labels(soff, sstart, send, eoff, es, ee, split.n)
  return soff, sstart, send, sstart:size(), ioff, inbr, eoff, es, ee, ety
end

local function enumerate_runs (soff, sstart, send, pred)
  local off, st, en = ivec.create(), ivec.create(), ivec.create()
  off:push(0)
  for d = 1, soff:size() - 1 do
    local si, hi = soff:get(d - 1), soff:get(d)
    while si < hi do
      if pred:get(si) ~= 1 then si = si + 1
      else
        local rlo = si
        while si < hi and pred:get(si) == 1 do si = si + 1 end
        local rhi = si - 1
        for i = rlo, rhi do
          for j = i, rhi do st:push(sstart:get(i)); en:push(send:get(j)) end
        end
      end
    end
    off:push(st:size())
  end
  return off, st, en
end

local function inner_to_ctx (pred)
  local n = pred:size()
  local out = ivec.create(n)
  for i = 0, n - 1 do
    out:set(i, pred:get(i) == 1 and 0 or 1)
  end
  return out
end

local function seg_ceiling (soff, sstart, send, eoff, es, ee, n)
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

local function tokenize_ctx (split, off, starts, ends, coff, cs, ce, cty, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = 1, terminals = true, focus = true, types = true,
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

local function tokenize_shape (split, off, starts, ends, shp, k, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = k + 1, terminals = true, focus = true, types = true,
    })
  end
  local o, t, v = tok:tokenize({
    texts = split.texts, n_samples = split.n,
    focus = { offsets = off, starts = starts, ends = ends },
    types = { offsets = shp.off, starts = shp.st, ends = shp.en, types = shp.cl },
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

  local seg = segmenter.create({ context = "left" })
  local coarse_k
  do
    local g_off, g_s, g_e = gold_spans(train)
    local ck, rec, mx, p95, sep = seg:train({ texts = train.texts, n = train.n,
      gold_offsets = g_off, gold_starts = g_s, gold_ends = g_e })
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

  local tr_off, tr_s, tr_e, tr_n, tr_ioff, tr_inbr, tr_eoff, tr_es, tr_ee, tr_ety = build_segments(train, seg)
  local dv_off, dv_s, dv_e, dv_n, dv_ioff, dv_inbr, dv_eoff, dv_es, dv_ee, dv_ety = build_segments(dev, seg)
  local te_off, te_s, te_e, te_n, _, _, te_eoff, te_es, te_ee, te_ety = build_segments(test_set, seg)
  str.printf("[SegCeil] boundary-aligned gold (oracle recall): dev=%.4f test=%.4f %s\n",
    seg_ceiling(dv_off, dv_s, dv_e, dv_eoff, dv_es, dv_ee, dev.n),
    seg_ceiling(te_off, te_s, te_e, te_eoff, te_es, te_ee, test_set.n), sw())

  local function shape_runs (split)
    local out = {}
    for i, k in ipairs(cfg.shape.ks) do
      local off, st, en, cl = seg:segment({ texts = split.texts, n = split.n, k = k })
      out[i] = { off = off, st = st, en = en, cl = cl }
    end
    return out
  end
  local tr_sh, dv_sh, te_sh = shape_runs(train), shape_runs(dev), shape_runs(test_set)

  local function tag_combine (to, tt, tv, tk, nl, loff, lnbr, fit, bns_in)
    if fit then
      local bns = csr.apply_bns(to, tt, tv, nil, loff, lnbr, tk, nl)
      csr.normalize(to, tv)
      return to, tt, tv, tk, bns
    end
    csr.apply_bns(to, tt, tv, bns_in)
    csr.normalize(to, tv)
    return to, tt, tv, tk, bns_in
  end

  local function train_head (label, scfg, tb, n, nl, loff, lnbr, dtb, dvn, dloff, dlnbr)
    local o, t, v, ntok, bns = tag_combine(tb.o, tb.t, tb.v, tb.k, nl, loff, lnbr, true)
    local dvo, dvt, dvv = tag_combine(dtb.o, dtb.t, dtb.v, dtb.k, nil, nil, nil, false, bns)
    str.printf("[%s] Encoding\n", label)
    local sp, rg, _, best, decider = optimize.krr({
      offsets = o, tokens = t, values = v, n_samples = n, n_tokens = ntok,
      kernel = scfg.kernel, rbf_gamma = scfg.gamma,
      n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
      label_offsets = loff, label_neighbors = lnbr, n_labels = nl,
      val_offsets = dvo, val_tokens = dvt, val_values = dvv, val_n_samples = dvn,
      val_expected_offsets = dloff, val_expected_neighbors = dlnbr,
      lambda = scfg.lambda,
      k = 1, search_trials = scfg.search_trials,
      each = util.make_ridge_log(stopwatch),
    })
    str.printf("[%s] kernel=%s lambda=%.4e %s\n", label, best.kernel, best.lambda, sw())
    return sp, rg, bns, best, decider
  end

  local function head_decode (sp, rg, decider, bns, tb, n)
    local o, t, v = tag_combine(tb.o, tb.t, tb.v, tb.k, nil, nil, nil, false, bns)
    local codes = sp:encode({ offsets = o, tokens = t, values = v, n_samples = n })
    local loff, lnbr, lsco = rg:label(codes, n, 1)
    return decider:predict({ offsets = loff, neighbors = lnbr, scores = lsco, n_samples = n })
  end

  local function zeros (n) local z = ivec.create(n); z:zero(); return z end

  local tr_inner, dv_inner, te_inner
  do
    local ng, e_off, e_tok, e_val, e_ntok = tokenize(train, tr_off, tr_s, tr_e, nil, nil)
    local _, dvo, dvt, dvv = tokenize(dev, dv_off, dv_s, dv_e, nil, ng)
    local sp, rg, bns, _, decider = train_head("Tag", cfg.tag,
      { o = e_off, t = e_tok, v = e_val, k = e_ntok }, tr_n, 1, tr_ioff, tr_inbr,
      { o = dvo, t = dvt, v = dvv, k = e_ntok }, dv_n, dv_ioff, dv_inbr)
    local _, tro, trt, trv = tokenize(train, tr_off, tr_s, tr_e, nil, ng)
    local _, ddo, ddt, ddv = tokenize(dev, dv_off, dv_s, dv_e, nil, ng)
    local _, teo, tet, tev = tokenize(test_set, te_off, te_s, te_e, nil, ng)
    tr_inner = head_decode(sp, rg, decider, bns, { o = tro, t = trt, v = trv, k = e_ntok }, tr_n)
    dv_inner = head_decode(sp, rg, decider, bns, { o = ddo, t = ddt, v = ddv, k = e_ntok }, dv_n)
    te_inner = head_decode(sp, rg, decider, bns, { o = teo, t = tet, v = tev, k = e_ntok }, te_n)
  end

  local en_tro, en_trs, en_tre = enumerate_runs(tr_off, tr_s, tr_e, tr_inner)
  local en_dvo, en_dvs, en_dve = enumerate_runs(dv_off, dv_s, dv_e, dv_inner)
  local en_teo, en_tes, en_tee = enumerate_runs(te_off, te_s, te_e, te_inner)

  local tr_cty, dv_cty, te_cty = inner_to_ctx(tr_inner), inner_to_ctx(dv_inner), inner_to_ctx(te_inner)

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

  local ty_ng1, ty_ng2, ty_ng3s, ty_ntok, ty_bns, ty_block, ty_bounds
  local function type_feats (split, co, cs, ce, n, toff, ts, te, ttypes, shps, is_train)
    local m1, o1, t1, v1, k1 = tokenize(split, co, cs, ce, zeros(n), ty_ng1)
    local m2, o2, t2, v2, k2 = tokenize_ctx(split, co, cs, ce, toff, ts, te, ttypes, ty_ng2)
    local o, t, v = csr.merge(o1, t1, v1, o2, t2, v2, k1)
    local ksum = k1 + k2
    local bnds, ng3 = { 0, k1, ksum }, {}
    for i, kk in ipairs(cfg.shape.ks) do
      local mi, oi, ti, vi, ki = tokenize_shape(split, co, cs, ce, shps[i], kk, ty_ng3s and ty_ng3s[i])
      o, t, v = csr.merge(o, t, v, oi, ti, vi, ksum)
      ksum = ksum + ki
      ng3[i] = mi
      bnds[#bnds + 1] = ksum
    end
    if is_train then
      ty_ng1, ty_ng2, ty_ng3s = m1, m2, ng3
      ty_ntok, ty_bounds = ksum, bnds
    end
    return o, t, v
  end
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
  local char_gaz = build_char_gaz(train, cfg.tok.ngram_min, cfg.tok.ngram_max)
  local function char_gaz_block (split, co, cs, ce, tlab)
    local off, tok, val = ivec.create(), ivec.create(), fvec.create()
    off:push(0)
    local nd = co:size() - 1
    for d = 1, nd do
      local text = split.texts[d]
      for i = co:get(d - 1), co:get(d) - 1 do
        local g = tlab and tlab:get(i) or N_TYPES
        local own = (g < N_TYPES) and 1 or 0
        local acc = {}
        for _, gram in ipairs(char_ngrams(text:sub(cs:get(i) + 1, ce:get(i)):lower(),
          cfg.tok.ngram_min, cfg.tok.ngram_max)) do
          local c = char_gaz[gram]
          if c then
            local den = c.total - own
            if den > 0 then
              for ty = 0, N_TYPES - 1 do
                local cnt = c[ty] - (ty == g and own or 0)
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
  local function ty_apply (split, co, cs, ce, n, toff, ts, te, ttypes, shps)
    local off, tok, val = type_feats(split, co, cs, ce, n, toff, ts, te, ttypes, shps, false)
    csr.apply_bns(off, tok, val, ty_bns)
    local go, gt, gv = gaz_block(split, co, cs, ce, nil)
    off, tok, val = csr.merge(off, tok, val, go, gt, gv, ty_ntok)
    local cgo, cgt, cgv = char_gaz_block(split, co, cs, ce, nil)
    off, tok, val = csr.merge(off, tok, val, cgo, cgt, cgv, ty_ntok + N_TYPES)
    csr.standardize(off, tok, val, ty_block)
    csr.normalize(off, val)
    return off, tok, val
  end

  local ty_off, ty_tok, ty_val = type_feats(train, tr_co, tr_cs, tr_ce, n_trc, tr_off, tr_s, tr_e, tr_cty, tr_sh, true)
  ty_bns = csr.apply_bns(ty_off, ty_tok, ty_val, nil, tr_tloff, tr_tlab, ty_ntok, N_TYPES + 1)
  local g_off, g_tok, g_val = gaz_block(train, tr_co, tr_cs, tr_ce, tr_tlab)
  ty_off, ty_tok, ty_val = csr.merge(ty_off, ty_tok, ty_val, g_off, g_tok, g_val, ty_ntok)
  local cg_off, cg_tok, cg_val = char_gaz_block(train, tr_co, tr_cs, tr_ce, tr_tlab)
  ty_off, ty_tok, ty_val = csr.merge(ty_off, ty_tok, ty_val, cg_off, cg_tok, cg_val, ty_ntok + N_TYPES)
  local ty_ntok_all = ty_ntok + 2 * N_TYPES
  local bounds = {}
  for _, b in ipairs(ty_bounds) do bounds[#bounds + 1] = b end
  bounds[#bounds + 1] = ty_ntok + N_TYPES
  bounds[#bounds + 1] = ty_ntok_all
  local ss = csr.block_sumsq(ty_tok, ty_val, bounds)
  ty_block = fvec.create(ty_ntok_all)
  for r = 0, #bounds - 2 do
    local ssr = ss:get(r)
    ty_block:fill(ssr > 0 and math.sqrt(n_trc / ssr) or 0.0, bounds[r + 1], bounds[r + 2])
  end
  csr.standardize(ty_off, ty_tok, ty_val, ty_block)
  csr.normalize(ty_off, ty_val)
  local ty_dvo, ty_dvt, ty_dvv = ty_apply(dev, dv_co, dv_cs, dv_ce, n_dvc, dv_off, dv_s, dv_e, dv_cty, dv_sh)

  str.printf("[Type] Encoding\n")
  local sp_ty, ridge_ty, _, _, decider_ty = optimize.krr({
    offsets = ty_off, tokens = ty_tok, values = ty_val, n_samples = n_trc, n_tokens = ty_ntok_all,
    kernel = cfg.type.kernel, rbf_gamma = cfg.type.gamma,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_tloff, label_neighbors = tr_tlab, n_labels = N_TYPES + 1,
    val_offsets = ty_dvo, val_tokens = ty_dvt, val_values = ty_dvv, val_n_samples = n_dvc,
    val_expected_offsets = dv_tloff, val_expected_neighbors = dv_tlab,
    val_spans = { cand_offsets = dv_co, cand_starts = dv_cs, cand_ends = dv_ce,
      gold_offsets = dv_eoff, gold_starts = dv_es, gold_ends = dv_ee, gold_types = dv_ety },
    lambda = cfg.type.lambda, propensity_a = cfg.type.propensity_a, propensity_b = cfg.type.propensity_b,
    k = 1, search_trials = cfg.type.search_trials,
    each = util.make_ridge_log(stopwatch),
  })

  local decide = require("santoku.learn.decide")
  local argmax_ty = decide.create({ n_labels = N_TYPES + 1, span = true })
  local n_dv_docs = dv_co:size() - 1
  local function span_score (d, scores, ndocs, co, cs, ce, geoff, ges, gee, gety)
    local _, m = d:score({ scores = scores, n_samples = ndocs,
      cand_offsets = co, cand_starts = cs, cand_ends = ce,
      expected_offsets = geoff, expected_starts = ges, expected_ends = gee, expected_types = gety })
    return m
  end
  local ty_teo, ty_tet, ty_tev = ty_apply(test_set, te_co, te_cs, te_ce, n_tec, te_off, te_s, te_e, te_cty, te_sh)
  local te_codes = sp_ty:encode({ offsets = ty_teo, tokens = ty_tet, values = ty_tev, n_samples = n_tec })
  local _, te_lab = ridge_ty:label(te_codes, n_tec, 2)
  local te_scores = ridge_ty:regress(te_codes, n_tec)
  local dev_codes = sp_ty:encode({ offsets = ty_dvo, tokens = ty_dvt, values = ty_dvv, n_samples = n_dvc })
  local dev_scores = ridge_ty:regress(dev_codes, n_dvc)
  str.printf("[Span] argmax dev %s | test %s %s\n",
    util.fmt_metrics(span_score(argmax_ty, dev_scores, n_dv_docs, dv_co, dv_cs, dv_ce, dv_eoff, dv_es, dv_ee, dv_ety)),
    util.fmt_metrics(span_score(argmax_ty, te_scores, n_te_docs, te_co, te_cs, te_ce, te_eoff, te_es, te_ee, te_ety)), sw())
  str.printf("[Span] decide dev %s | test %s %s\n",
    util.fmt_metrics(span_score(decider_ty, dev_scores, n_dv_docs, dv_co, dv_cs, dv_ce, dv_eoff, dv_es, dv_ee, dv_ety)),
    util.fmt_metrics(span_score(decider_ty, te_scores, n_te_docs, te_co, te_cs, te_ce, te_eoff, te_es, te_ee, te_ety)), sw())

  local tdr = csr.type_decode_report({
    cand_offsets = te_co, cand_starts = te_cs, cand_ends = te_ce, cand_pred = te_lab, pred_stride = 2,
    gold_offsets = te_eoff, gold_starts = te_es, gold_ends = te_ee, gold_types = te_ety, n_types = N_TYPES,
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
