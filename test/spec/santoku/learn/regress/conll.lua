local ds = require("santoku.learn.dataset")
local csr = require("santoku.csr")
local spans = require("santoku.spans")
local svec = require("santoku.svec")
local ner = require("santoku.learn.ner")
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
  },
  tag = {
    kernel = { "arccos", "matern", "cosine" },
    nu = { def = 3 },
    gamma = { def = 0.05076 },
    order = { def = 1 },
    depth = { def = 1 },
    tangent = { def = 0 },
    lambda = { def = 8.4893e-04 },
    arccos_order = { 1, 0, 2, 3, 4, 5, 6 },
    arccos_depth = { 1, 2, 3 },
    arccos_tangent = { 0, 1 },
    search_trials = 0
  },
  type = {
    kernel = { "arccos", "matern", "cosine" },
    nu = { def = 1 },
    gamma = { def = 0.3787 },
    order = { def = 3 },
    depth = { def = 1 },
    tangent = { def = 0 },
    lambda = { def = 8.2422e-04 },
    propensity_a = { def = 0.1159 },
    propensity_b = { def = 8.3703 },
    arccos_order = { 3, 0, 1, 2, 4, 5, 6 },
    arccos_depth = { 1, 2, 3 },
    arccos_tangent = { 0, 1 },
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
  local F = spans.create({ offsets = off, s = starts, e = ends })
  local X = grow and tok:fit({ texts = split.texts, focus = F })
    or tok:tokenize({ texts = split.texts, focus = F })
  return tok, X
end

local function tokenize_ctx (split, off, starts, ends, coff, cs, ce, cty, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = 1, terminals = true, focus = true, types = true,
    })
  end
  local F = spans.create({ offsets = off, s = starts, e = ends })
  local T = spans.create({ offsets = coff, s = cs, e = ce, ty = cty })
  local X = grow and tok:fit({ texts = split.texts, focus = F, types = T })
    or tok:tokenize({ texts = split.texts, focus = F, types = T })
  return tok, X
end

local function tokenize_shape (split, off, starts, ends, shp, k, tok)
  local grow = tok == nil
  if grow then
    tok = tokenizer.create({
      ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_types = k + 1, terminals = true, focus = true, types = true,
    })
  end
  local F = spans.create({ offsets = off, s = starts, e = ends })
  local T = spans.create({ offsets = shp.off, s = shp.st, e = shp.en, ty = shp.cl })
  local X = grow and tok:fit({ texts = split.texts, focus = F, types = T })
    or tok:tokenize({ texts = split.texts, focus = F, types = T })
  return tok, X
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

  local function tag_combine (X, Y, bns_in)
    local bns
    if bns_in then X:bns(bns_in); bns = bns_in
    else bns = X:bns(Y) end
    X:normalize()
    return bns
  end

  local function train_head (label, scfg, X, Y, Xdv, Ydv)
    local bns = tag_combine(X, Y)
    tag_combine(Xdv, nil, bns)
    str.printf("[%s] Encoding\n", label)
    local sp, rg, _, best, decider = optimize.krr({
      x = X, y = Y, val_x = Xdv, val_y = Ydv,
      kernel = scfg.kernel, nu = scfg.nu, gamma = scfg.gamma,
      n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
      lambda = scfg.lambda,
      k = 1, search_trials = scfg.search_trials, matern_trials = scfg.matern_trials,
      order = scfg.order, depth = scfg.depth, tangent = scfg.tangent,
      arccos_order = scfg.arccos_order, arccos_depth = scfg.arccos_depth,
      arccos_tangent = scfg.arccos_tangent,
      each = util.make_ridge_log(stopwatch),
    })
    str.printf("[%s] kernel=%s lambda=%.4e %s\n", label, best.kernel, best.lambda, sw())
    return sp, rg, bns, best, decider
  end

  local function head_decode (sp, rg, decider, bns, X, n)
    tag_combine(X, nil, bns)
    local P = rg:label(sp:encode(X), 1)
    return decider:predict({ pred = P, n_samples = n })
  end

  local function zeros (n) local z = ivec.create(n); z:zero(); return z end

  local tr_inner, dv_inner, te_inner
  do
    local ng, Xtr = tokenize(train, tr_off, tr_s, tr_e)
    local _, Xdv = tokenize(dev, dv_off, dv_s, dv_e, nil, ng)
    local Ytr = csr.create({ offsets = tr_ioff, neighbors = tr_inbr, n_cols = 1 })
    local Ydv = csr.create({ offsets = dv_ioff, neighbors = dv_inbr, n_cols = 1 })
    local sp, rg, bns, _, decider = train_head("Tag", cfg.tag, Xtr, Ytr, Xdv, Ydv)
    local _, Xtr2 = tokenize(train, tr_off, tr_s, tr_e, nil, ng)
    local _, Xdv2 = tokenize(dev, dv_off, dv_s, dv_e, nil, ng)
    local _, Xte2 = tokenize(test_set, te_off, te_s, te_e, nil, ng)
    tr_inner = head_decode(sp, rg, decider, bns, Xtr2, tr_n)
    dv_inner = head_decode(sp, rg, decider, bns, Xdv2, dv_n)
    te_inner = head_decode(sp, rg, decider, bns, Xte2, te_n)
  end

  local en_tro, en_trs, en_tre = enumerate_runs(tr_off, tr_s, tr_e, tr_inner)
  local en_dvo, en_dvs, en_dve = enumerate_runs(dv_off, dv_s, dv_e, dv_inner)
  local en_teo, en_tes, en_tee = enumerate_runs(te_off, te_s, te_e, te_inner)

  local tr_cty, dv_cty, te_cty = inner_to_ctx(tr_inner), inner_to_ctx(dv_inner), inner_to_ctx(te_inner)

  local ac = build_gazetteer(train)
  local function cand_union (split, eoff, es, ee, goff, gs, ge)
    local g_off, g_s, g_e = gaz_candidates(ac, split)
    local A = spans.create({ offsets = eoff, s = es, e = ee, ty = zeros(es:size()) })
    local B = spans.create({ offsets = g_off, s = g_s, e = g_e, ty = zeros(g_s:size()) })
    local G = spans.create({ offsets = goff, s = gs, e = ge, ty = zeros(gs:size()) })
    return A:union(B, G)
  end
  local Scand_tr = cand_union(train, en_tro, en_trs, en_tre, tr_eoff, tr_es, tr_ee)
  local Scand_dv = cand_union(dev, en_dvo, en_dvs, en_dve, dv_eoff, dv_es, dv_ee)
  local Scand_te = cand_union(test_set, en_teo, en_tes, en_tee, te_eoff, te_es, te_ee)
  local tr_co, tr_cs, tr_ce = Scand_tr:offsets(), Scand_tr:col("s"), Scand_tr:col("e")
  local dv_co, dv_cs, dv_ce = Scand_dv:offsets(), Scand_dv:col("s"), Scand_dv:col("e")
  local te_co, te_cs, te_ce = Scand_te:offsets(), Scand_te:col("s"), Scand_te:col("e")
  local n_trc, n_dvc, n_tec = tr_cs:size(), dv_cs:size(), te_cs:size()

  local Sgold_tr = spans.create({ offsets = tr_eoff, s = tr_es, e = tr_ee, ty = tr_ety })
  local Sgold_dv = spans.create({ offsets = dv_eoff, s = dv_es, e = dv_ee, ty = dv_ety })
  local Sgold_te = spans.create({ offsets = te_eoff, s = te_es, e = te_ee, ty = te_ety })
  local tr_tlab = Scand_tr:type_labels(Sgold_tr, N_TYPES)
  local dv_tlab = Scand_dv:type_labels(Sgold_dv, N_TYPES)
  local tr_tloff = ivec.create(n_trc + 1); tr_tloff:fill_indices()
  local dv_tloff = ivec.create(n_dvc + 1); dv_tloff:fill_indices()

  local te_gold = te_eoff:get(te_eoff:size() - 1)
  local n_te_docs = te_off:size() - 1
  local Sgaz_empty = spans.create({ offsets = zeros(n_te_docs + 1),
    s = ivec.create(0), e = ivec.create(0), ty = ivec.create(0) })
  local miss = ner.miss_report({ gaz = Sgaz_empty, bio = Scand_te, gold = Sgold_te, n_types = N_TYPES })
  local ubt = miss.under_by_type
  str.printf("[Cands] train=%d dev=%d test=%d | test span-coverage=%.4f"
    .. " | miss over=%d under=%d cross=%d none=%d"
    .. " | under by-type[PER=%d ORG=%d LOC=%d MISC=%d] %s\n",
    n_trc, n_dvc, n_tec, (miss.covered + miss.wrong_type) / te_gold,
    miss.over, miss.under, miss.cross, miss.none,
    ubt[0], ubt[1], ubt[2], ubt[3], sw())

  local ty_ng1, ty_ng2, ty_ng3s, ty_ntok, ty_bns, ty_block, ty_bounds
  local function type_feats (split, co, cs, ce, n, toff, ts, te, ttypes, shps, is_train)
    local m1, X = tokenize(split, co, cs, ce, zeros(n), ty_ng1)
    local _, k1 = X:shape()
    local m2, X2 = tokenize_ctx(split, co, cs, ce, toff, ts, te, ttypes, ty_ng2)
    local _, k2 = X2:shape()
    X:hcat(X2)
    local ksum = k1 + k2
    local bnds, ng3 = { 0, k1, ksum }, {}
    for i, kk in ipairs(cfg.shape.ks) do
      local mi, Xi = tokenize_shape(split, co, cs, ce, shps[i], kk, ty_ng3s and ty_ng3s[i])
      local _, ki = Xi:shape()
      X:hcat(Xi)
      ksum = ksum + ki
      ng3[i] = mi
      bnds[#bnds + 1] = ksum
    end
    if is_train then
      ty_ng1, ty_ng2, ty_ng3s = m1, m2, ng3
      ty_ntok, ty_bounds = ksum, bnds
    end
    return X
  end
  local gaz_counts = build_typed_gaz(train)
  local function gaz_block (split, co, cs, ce, tlab)
    local off, tok, val = ivec.create(), svec.create(), fvec.create()
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
    return csr.create({ offsets = off, neighbors = tok, values = val, n_cols = N_TYPES })
  end
  local char_gaz = build_char_gaz(train, cfg.tok.ngram_min, cfg.tok.ngram_max)
  local function char_gaz_block (split, co, cs, ce, tlab)
    local off, tok, val = ivec.create(), svec.create(), fvec.create()
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
    return csr.create({ offsets = off, neighbors = tok, values = val, n_cols = N_TYPES })
  end
  local function ty_apply (split, co, cs, ce, n, toff, ts, te, ttypes, shps)
    local X = type_feats(split, co, cs, ce, n, toff, ts, te, ttypes, shps, false)
    X:bns(ty_bns)
    X:hcat(gaz_block(split, co, cs, ce, nil))
    X:hcat(char_gaz_block(split, co, cs, ce, nil))
    X:standardize(ty_block)
    X:normalize()
    return X
  end

  local Xty = type_feats(train, tr_co, tr_cs, tr_ce, n_trc, tr_off, tr_s, tr_e, tr_cty, tr_sh, true)
  local Ytype = csr.create({ offsets = tr_tloff, neighbors = tr_tlab, n_cols = N_TYPES + 1 })
  ty_bns = Xty:bns(Ytype)
  Xty:hcat(gaz_block(train, tr_co, tr_cs, tr_ce, tr_tlab))
  Xty:hcat(char_gaz_block(train, tr_co, tr_cs, tr_ce, tr_tlab))
  local ty_ntok_all = ty_ntok + 2 * N_TYPES
  local bounds = {}
  for _, b in ipairs(ty_bounds) do bounds[#bounds + 1] = b end
  bounds[#bounds + 1] = ty_ntok + N_TYPES
  bounds[#bounds + 1] = ty_ntok_all
  local ss = Xty:sumsq_cols(ivec.create(bounds))
  ty_block = fvec.create(ty_ntok_all)
  for r = 0, #bounds - 2 do
    local ssr = ss:get(r)
    ty_block:fill(ssr > 0 and math.sqrt(n_trc / ssr) or 0.0, bounds[r + 1], bounds[r + 2])
  end
  Xty:standardize(ty_block)
  Xty:normalize()
  local Xdv_ty = ty_apply(dev, dv_co, dv_cs, dv_ce, n_dvc, dv_off, dv_s, dv_e, dv_cty, dv_sh)

  str.printf("[Type] Encoding\n")
  local sp_ty, ridge_ty, _, _, decider_ty = optimize.krr({
    x = Xty, val_x = Xdv_ty,
    y = csr.create({ offsets = tr_tloff, neighbors = tr_tlab, n_cols = N_TYPES + 1 }),
    val_y = csr.create({ offsets = dv_tloff, neighbors = dv_tlab, n_cols = N_TYPES + 1 }),
    kernel = cfg.type.kernel, nu = cfg.type.nu, gamma = cfg.type.gamma,
    order = cfg.type.order, depth = cfg.type.depth, tangent = cfg.type.tangent,
    arccos_order = cfg.type.arccos_order, arccos_depth = cfg.type.arccos_depth,
    arccos_tangent = cfg.type.arccos_tangent,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    val_spans = { cand_offsets = dv_co, cand_starts = dv_cs, cand_ends = dv_ce,
      gold_offsets = dv_eoff, gold_starts = dv_es, gold_ends = dv_ee, gold_types = dv_ety },
    lambda = cfg.type.lambda, propensity_a = cfg.type.propensity_a, propensity_b = cfg.type.propensity_b,
    k = 1, search_trials = cfg.type.search_trials,
    each = util.make_ridge_log(stopwatch),
  })

  local decide = require("santoku.learn.decide")
  local argmax_ty = decide.create({ n_labels = N_TYPES + 1, span = true })
  local n_dv_docs = dv_co:size() - 1
  local function span_score (d, scores, ndocs, Scand, Sgold)
    local _, m = d:score({ scores = scores, n_samples = ndocs, cand = Scand, gold = Sgold })
    return m
  end
  local Xte_ty = ty_apply(test_set, te_co, te_cs, te_ce, n_tec, te_off, te_s, te_e, te_cty, te_sh)
  local te_codes = sp_ty:encode(Xte_ty)
  local te_lab = ridge_ty:label(te_codes, 2):neighbors()
  local te_scores = ridge_ty:regress(te_codes)
  local dev_scores = ridge_ty:regress(sp_ty:encode(Xdv_ty))
  str.printf("[Span] argmax dev %s | test %s %s\n",
    util.fmt_metrics(span_score(argmax_ty, dev_scores, n_dv_docs, Scand_dv, Sgold_dv)),
    util.fmt_metrics(span_score(argmax_ty, te_scores, n_te_docs, Scand_te, Sgold_te)), sw())
  str.printf("[Span] decide dev %s | test %s %s\n",
    util.fmt_metrics(span_score(decider_ty, dev_scores, n_dv_docs, Scand_dv, Sgold_dv)),
    util.fmt_metrics(span_score(decider_ty, te_scores, n_te_docs, Scand_te, Sgold_te)), sw())

  local tdr = ner.decode_report({
    cand = Scand_te, pred = te_lab, pred_stride = 2,
    gold = Sgold_te, n_types = N_TYPES,
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
