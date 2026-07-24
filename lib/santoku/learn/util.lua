local str = require("santoku.string")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local fvec = require("santoku.fvec")
local svec = require("santoku.svec")
local spans = require("santoku.spans")
local re = require("santoku.re")
local fs = require("santoku.fs")

local M = {}

local word_prog = re.prog("[A-Za-z0-9]+")

local SHAPE_PATTERN = table.concat({
  '{:caps: [A-Z][A-Z]+ :} ![A-Za-z0-9]',
  '{:icap: [A-Z][a-z]+ [A-Z] [A-Za-z]* :} ![A-Za-z0-9]',
  '{:cap: [A-Z][a-z]+ :} ![A-Za-z0-9]',
  '{:u1: [A-Z] :} ![A-Za-z0-9]',
  '{:low: [a-z]+ :} ![A-Za-z0-9]',
  '{:d4: [0-9][0-9][0-9][0-9] :} ![A-Za-z0-9]',
  '{:d2: [0-9][0-9] :} ![A-Za-z0-9]',
  '{:num: [0-9]+ :} ![A-Za-z0-9]',
  '{:dsuf: [0-9]+ [a-z]+ :} ![A-Za-z0-9]',
  '{:alnum: [A-Za-z0-9]+ :}',
  '{:dot: "."+ :} ![^A-Za-z0-9%s]',
  '{:comma: ","+ :} ![^A-Za-z0-9%s]',
  '{:hyph: "-"+ :} ![^A-Za-z0-9%s]',
  '{:apos: "\'"+ :} ![^A-Za-z0-9%s]',
  '{:quot: ["`]+ :} ![^A-Za-z0-9%s]',
  '{:paren: [()]+ :} ![^A-Za-z0-9%s]',
  '{:amp: "&"+ :} ![^A-Za-z0-9%s]',
  '{:slash: "/"+ :} ![^A-Za-z0-9%s]',
  '{:punct: [^A-Za-z0-9%s]+ :}',
}, " / ")
local shape_prog = re.prog(SHAPE_PATTERN)

M.SHAPE_TAGS = re.tags(SHAPE_PATTERN)
M.N_SHAPES = 0
for _ in pairs(M.SHAPE_TAGS) do M.N_SHAPES = M.N_SHAPES + 1 end

function M.rmbundle (dir)
  for f in fs.files(dir) do fs.rm(f) end
  fs.rmdirs(dir)
end

function M.merged (...)
  local out = {}
  for i = 1, select("#", ...) do
    local t = select(i, ...)
    if t then for k, v in pairs(t) do out[k] = v end end
  end
  return out
end

function M.word_spans (texts, n)
  local tokenizer = require("santoku.learn.tokenizer")
  local off, s, e = tokenizer.extract({ n = n, texts = texts, pattern = word_prog })
  return spans.create({ offsets = off, s = s, e = e })
end

function M.shape_spans (texts, n)
  local tokenizer = require("santoku.learn.tokenizer")
  local off, s, e, ty = tokenizer.extract({ n = n, texts = texts, pattern = shape_prog })
  return spans.create({ offsets = off, s = s, e = e, ty = ty })
end

function M.surface_gaz (splits, n_types, normalize)
  local aho = require("santoku.learn.aho")
  local counts = {}
  for _, sp in ipairs(splits) do
    local go, gs, ge, gt = sp.gold:offsets(), sp.gold:col("s"), sp.gold:col("e"), sp.gold:col("ty")
    for d = 1, sp.n do
      local text = sp.texts[d]
      for j = go:get(d - 1), go:get(d) - 1 do
        local surf = text:sub(gs:get(j) + 1, ge:get(j))
        local c = counts[surf]
        if not c then c = {}; counts[surf] = c end
        local ty = gt:get(j)
        c[ty] = (c[ty] or 0) + 1
      end
    end
  end
  local patterns, pat_type = {}, ivec.create()
  for surf, c in pairs(counts) do
    patterns[#patterns + 1] = surf
    local bt, bc = 0, -1
    for ty = 0, n_types - 1 do
      local k = c[ty] or 0
      if k > bc then bc, bt = k, ty end
    end
    pat_type:push(bt)
  end
  return aho.create({ patterns = patterns, normalize = normalize }), pat_type
end

function M.vecstr (v)
  if type(v) ~= "table" then v = { v } end
  local p = {}
  for i = 1, #v do local x = v[i]
    p[i] = (x == math.floor(x)) and str.format("%d", x) or str.format("%.8g", x) end
  return "[" .. table.concat(p, ",") .. "]"
end

function M.fmt_metrics (m)
  if m.micro_f1 ~= nil then
    return str.format("miF1=%.6f miP=%.6f miR=%.6f", m.micro_f1, m.micro_precision or 0, m.micro_recall or 0)
  elseif m.precision ~= nil and m.f1 ~= nil then
    return str.format("miF1=%.6f miP=%.6f miR=%.6f", m.f1, m.precision, m.recall or 0)
  elseif m.macro_f1 ~= nil then
    return str.format("maF1=%.6f acc=%.6f", m.macro_f1, m.accuracy or 0)
  elseif m.span_f1 ~= nil then
    return str.format("spF1=%.6f P=%.6f R=%.6f", m.span_f1, m.precision or 0, m.recall or 0)
  end
  return ""
end

local function format_phase (ev)
  if ev.is_final then return "F" end
  local tag = ev.phase or "lhs"
  return str.format("%s %d/%d", tag, ev.trial or 1, ev.trials or 1)
end


local NU_NAME = { [0] = "1/2", [1] = "3/2", [2] = "5/2", [3] = "inf" }

local function format_kernel (p)
  if not p.kernel then
    return p.activation and str.format(" act=%s", p.activation) or ""
  end
  if p.kernel == "matern" then
    local nu = p.nu ~= nil and (NU_NAME[p.nu] or tostring(p.nu)) or "?"
    local g = p.gamma and str.format(" gamma=%.8g", p.gamma) or ""
    return str.format(" kernel=matern nu=%s%s", nu, g)
  elseif p.kernel == "arccos" then
    return str.format(" kernel=arccos n=%d depth=%d %s",
      p.order or 1, p.depth or 1, (p.tangent == 1) and "ntk" or "nngp")
  end
  local g = p.gamma and str.format(" gamma=%.8g", p.gamma) or ""
  return str.format(" kernel=%s%s", p.kernel, g)
end

local function fmt_exponent (ex)
  if type(ex) == "number" then return str.format(" exp=%.8g", ex) end
  if type(ex) ~= "table" then return "" end
  local parts, any = {}, false
  for i = 1, #ex do
    if ex[i] == false or ex[i] == nil then parts[i] = ""
    else parts[i] = str.format("%.8g", ex[i]); any = true end
  end
  if not any then return "" end
  return " exp=[" .. table.concat(parts, ",") .. "]"
end

local function fmt_scales (sc)
  if sc == nil then return "" end
  if type(sc) == "table" then
    local parts = {}
    for i = 1, #sc do parts[i] = sc[i] == false and "1" or str.format("%.8g", sc[i]) end
    return " scales=[" .. table.concat(parts, ",") .. "]"
  end
  return str.format(" scales=%.8g", sc)
end

function M.make_ridge_log (stopwatch, metric_fmt)
  -- accumulate every trial's CV score so [Ridge Done] can report the plateau width:
  -- how many configs are CV-indistinguishable from the winner (selection is a draw from
  -- that plateau, so a pin's last decimals are noise, not signal).
  local scores = {}
  return function (ev)
    if ev.event == "done" then
      local p = ev.params or {}
      local emb = ev.emb_d and str.format(" emb_d=%d", ev.emb_d) or ""
      local md = p.mode and str.format(" mode=%s", p.mode) or ""
      local kdesc = format_kernel(p)
      local solve = ev.solve and str.format(" solve=%s", ev.solve) or ""
      local sc = ev.score and str.format(" score=%.6f", ev.score) or ""
      local fsd = ev.fold_std and str.format(" fold_std=%.6f", ev.fold_std) or ""
      local scl = fmt_scales(p.scales)
      local exf = fmt_exponent(p.exponent)
      local timing = ""
      if stopwatch then
        local d, dd = stopwatch()
        timing = str.format(" (%.1fs +%.1fs)", d, dd)
      end
      str.printf("[Ridge Done]%s%s%s%s%s%s lambda=%.8g%s%s%s\n",
        emb, md, kdesc, scl, exf, solve, p.lambda or 0, sc, fsd, timing)
      if #scores > 1 then
        table.sort(scores, function (a, b) return a > b end)
        local n = #scores
        local best = scores[1]
        local band = best - scores[math.min(32, n)]
        local w1, w5 = 0, 0
        for i = 1, n do
          if scores[i] >= best - 0.001 then w1 = w1 + 1 end
          if scores[i] >= best - 0.005 then w5 = w5 + 1 end
        end
        str.printf("[Plateau] trials=%d best=%.6f band32=%.6f within1e-3=%d(%.0f%%) within5e-3=%d\n",
          n, best, band, w1, 100 * w1 / n, w5)
      end
      return
    end
    if ev.event == "profile" then
      local total = ev.total or 0
      local leaves, spans = {}, {}
      for kk in pairs(ev.stats) do
        if kk:sub(1, 1) == "~" then spans[#spans + 1] = kk else leaves[#leaves + 1] = kk end
      end
      table.sort(leaves, function (a, b) return ev.stats[a].time > ev.stats[b].time end)
      table.sort(spans, function (a, b) return ev.stats[a].time > ev.stats[b].time end)
      str.printf("[Profile] total=%.2fs\n", total)
      local leafsum = 0
      for _, kk in ipairs(leaves) do
        local e = ev.stats[kk]; leafsum = leafsum + e.time
        str.printf("  %-20s %9.2fs %5.1f%%  n=%-8d avg=%.4gms\n",
          kk, e.time, total > 0 and 100 * e.time / total or 0, e.count, e.count > 0 and 1000 * e.time / e.count or 0)
      end
      str.printf("  %-20s %9.2fs %5.1f%%\n", "(unaccounted)",
        total - leafsum, total > 0 and 100 * (total - leafsum) / total or 0)
      for _, kk in ipairs(spans) do
        local e = ev.stats[kk]
        str.printf("  [span] %-13s %9.2fs %5.1f%%  n=%d\n",
          kk:sub(2), e.time, total > 0 and 100 * e.time / total or 0, e.count)
      end
      return
    end
    local phase = format_phase(ev)
    local p = ev.params or {}
    local m = ev.metrics or {}
    local score = ev.score or 0
    if ev.event == "trial" and not m.failed then scores[#scores + 1] = score end
    local best = (ev.best and ev.best ~= -math.huge)
      and str.format(" (best=%.6f%s)", ev.best, ev.is_new_best and " ++" or "") or ""
    local md = metric_fmt and metric_fmt(m) or M.fmt_metrics(m)
    local detail = (md ~= "") and (" " .. md) or ""
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      timing = str.format(" (%.1fs +%.1fs)", d, dd)
    end
    local kdesc = format_kernel(p)
    local embd = ev.emb_d and str.format(" emb_d=%d", ev.emb_d) or ""
    local scl = fmt_scales(p.scales)
    local exf = fmt_exponent(p.exponent)
    local lambda = p.lambda or m.lambda or 0
    local off = (m.offset ~= nil) and str.format(" offset=%.8g", m.offset) or ""
    str.printf("[Ridge %s]%s%s%s%s lambda=%.8g score=%.6f%s%s%s%s\n",
      phase, kdesc, embd, scl, exf, lambda, score, detail, off, best, timing)
  end
end

-- Buffer factory the tokenizer leaf calls to materialize its output CSR (off/toks/vals). The
-- mid layer owns the RAM-vs-mmap choice: with a scratch base -> mmap_create to disk (RAM-lean),
-- else the leaf plain-creates in RAM. Keeps the C leaf buffer-pure (it never mmaps a path).
local TOK_VEC = { off = ivec, toks = svec, vals = fvec }
local function mmap_alloc (base)
  return function (kind, n) return TOK_VEC[kind].mmap_create(base .. "." .. kind, n) end
end

function M.tokenize_blocks (specs, texts, o)
  o = o or {}
  local tokenizer = require("santoku.learn.tokenizer")
  local toks = o.toks
  local grow = toks == nil
  if grow then
    toks = {}
    for i, b in ipairs(specs) do
      toks[i] = tokenizer.create({ ngram_min = b.ngram_min, ngram_max = b.ngram_max,
        mode = b.mode, n_tags = b.n_tags, normalize = b.normalize,
        terminals = o.focus ~= nil, focus = o.focus ~= nil, regions = b.regions })
    end
  end
  local X = {}
  for i = 1, #specs do
    local tk = o.tokens
    if type(tk) == "table" then tk = tk[i] end
    -- ivec.mmap_create is nil on WASM (no mmap) -> alloc stays nil -> RAM everywhere
    local alloc = (o.scratch and ivec.mmap_create) and mmap_alloc(o.scratch .. "." .. i) or nil
    local targs = { texts = texts, focus = o.focus, tokens = tk, alloc = alloc }
    local csr = grow and toks[i]:fit(targs) or toks[i]:tokenize(targs)
    local go = specs[i].regions and toks[i]:group_offsets() or nil
    X[i] = go and { x = csr, group_offsets = go } or csr
  end
  return toks, X
end

local function fold_assign (class, n, K)
  local foldof = ivec.create(n)
  if class then
    local by = {}
    for i = 0, n - 1 do local c = class:get(i); local t = by[c]; if not t then t = {}; by[c] = t end; t[#t + 1] = i end
    for _, docs in pairs(by) do for j = 1, #docs do foldof:set(docs[j], (j - 1) % K) end end
  else
    for i = 0, n - 1 do foldof:set(i, i % K) end
  end
  return foldof
end

local function fold_assign_reg (targets, n, K)
  local order = targets:argsort()
  local foldof = ivec.create(n)
  for r = 0, n - 1 do foldof:set(order:get(r), r % K) end
  return foldof
end

local function derive_class (labels, n)
  if not labels then return nil end
  local off, nbr = labels:offsets(), labels:neighbors()
  local single = true
  for i = 0, n - 1 do if off:get(i + 1) - off:get(i) ~= 1 then single = false; break end end
  local cls = ivec.create(n)
  if single then
    for i = 0, n - 1 do cls:set(i, nbr:get(off:get(i))) end
    return cls
  end
  local freq = {}
  for j = 0, nbr:size() - 1 do local l = nbr:get(j); freq[l] = (freq[l] or 0) + 1 end
  for i = 0, n - 1 do
    local lo, hi = off:get(i), off:get(i + 1)
    if hi <= lo then cls:set(i, -1)
    else
      local best, bestf = nbr:get(lo), freq[nbr:get(lo)]
      for j = lo + 1, hi - 1 do
        local l = nbr:get(j); local f = freq[l]
        if f < bestf or (f == bestf and l < best) then best, bestf = l, f end
      end
      cls:set(i, best)
    end
  end
  return cls
end

local SPAN_STRAT_CAP = 8
local function doc_strata (gold, ndocs)
  local go = gold:offsets()
  local buckets, order = {}, {}
  for d = 0, ndocs - 1 do
    local gc = go:get(d + 1) - go:get(d)
    if gc > SPAN_STRAT_CAP then gc = SPAN_STRAT_CAP end
    local b = buckets[gc]; if not b then b = {}; buckets[gc] = b; order[#order + 1] = gc end
    b[#b + 1] = d
  end
  return buckets, order
end

function M.doc_folds (cand, gold, K)
  local co = cand:offsets()
  local ndocs = co:size() - 1
  local buckets, order = doc_strata(gold, ndocs)
  local docfold = ivec.create(ndocs)
  for _, gc in ipairs(order) do
    local docs = buckets[gc]
    for j = 1, #docs do docfold:set(docs[j], (j - 1) % K) end
  end
  return docfold
end

local REG_STRATA_BINS = 16
local function reg_strata (targets, _n, B)
  return targets:quantile_bins(B)
end

local function pool_strata (a, n)
  if a.cand then
    local ty = a.cand:col("ty")
    local s = ivec.create(n)
    for i = 0, n - 1 do s:set(i, ty:get(i)) end
    return s
  elseif a.pool_targets then
    return reg_strata(a.pool_targets, n, REG_STRATA_BINS)
  else
    return derive_class(a.pool_labels, n)
  end
end

-- never-val stratum size: rows that no fold validates on serve two roles -- leak-free
-- weight-fit set and CV landmark basis -- so the count is driven by the CV landmark
-- budget (search_landmarks) with ~1.3x headroom, capped at 40% of the pool so the folds
-- keep enough training mass. No dependence on fold count (K); the stratum is orthogonal.
local STRAT_HEADROOM = 1.3
local STRAT_CAP = 0.4
local function strat_size (a, n)
  local m = a.search_landmarks or a.n_landmarks or math.floor(n / 3)
  return math.max(1, math.min(math.ceil(m * STRAT_HEADROOM), math.floor(n * STRAT_CAP)))
end

local function weight_fit (blocks, y, metrics, is_targets)
  local w = {}
  for i = 1, #blocks do
    local m = metrics and metrics[i]
    if m and y then
      if is_targets and m ~= "auc" then
        error("weight_fit: continuous targets require the auc metric")
      end
      w[i] = (m == "auc") and blocks[i]:auc(y) or blocks[i]:bns(y, true)
    end
  end
  return w
end

function M.rms_scale_blocks (train_blocks, eval_block_lists, from, to)
  local weights = {}
  for i = from or 1, to or #train_blocks do
    local X = train_blocks[i]
    local n, nc = X:shape()
    local ssq = X:sumsq_cols()
    local w = fvec.create(nc)
    for c = 0, nc - 1 do
      local s = ssq:get(c)
      w:set(c, s > 0 and math.sqrt(n / s) or 0)
    end
    X:bns(w)
    for _, ebl in ipairs(eval_block_lists) do ebl[i]:bns(w) end
    weights[i] = w
  end
  return weights
end

local WEIGHT_FLOOR = 1e-6
-- cs_cache (optional): pool the non-grouped colscale fvec across trials. For a non-grouped
-- weighted block, colscale = (wi:colscale(pcs, e, floor)) depends ONLY on the fixed weights
-- wi, fixed sumsq pcs[i], and the exponent e (the per-group `scale` feeds the scalar `sc`,
-- never the colscale vector), so it is safe to memoize by (block index, e) -- identical to
-- fold_dense's cs_cache.
-- gg_cache (optional): pool the GROUPED colscale fvec across trials. group_gauge's output is a
-- per-column vector sized to the block's group column span (trial-invariant), and every entry
-- is rewritten in place each call, so we keep one persistent fvec per grouped block and pass it
-- as the group_gauge out-buffer -- fully overwritten => bit-identical, and no per-trial fvec
-- churn. The grouped VALUES still vary per trial (scales/exps), so this is a buffer pool, not a
-- value memoization. encode only READS colscale, so sharing one fvec across trials is safe.
function M.build_blocks (blocks, scale, exponent, n, w, pcs, groups, cs_cache, gg_cache)
  local bl = {}
  local g0 = 0
  for i = 1, #blocks do
    local nc = select(2, blocks[i]:shape())
    local go = groups and groups[i]
    local wi = w and w[i]
    if go then
      local ng = go:size() - 1
      local scs, exs = {}, {}
      for g = 1, ng do
        local s = scale and scale[g0 + g]
        scs[g] = type(s) == "number" and s or 1.0
        local e = exponent and exponent[g0 + g]
        exs[g] = type(e) == "number" and e or 1.0
      end
      local out = gg_cache and gg_cache[i]
      local cs = pcs[i]:group_gauge(go, scs, n,
        { w = wi, exps = wi and exs or nil, floor = WEIGHT_FLOOR }, out)
      if gg_cache and not out then gg_cache[i] = cs end
      bl[i] = { x = blocks[i], n_tokens = nc, scale = 1.0, colscale = cs }
      g0 = g0 + ng
    else
      local sc = scale and scale[g0 + 1] or 1.0
      local cs, wssq
      if wi and pcs and pcs[i] then
        local e = exponent and exponent[g0 + 1]
        if e == nil or e == false then e = 1.0 end
        -- single-slot-per-block cache: reuse the pooled colscale when e is unchanged
        -- (fixed-exponent search = every-trial hit), else recompute IN PLACE into the pooled
        -- fvec (out-buffer). Bounded to one fvec per block regardless of trial count -- without
        -- the out-buffer a searched exponent misses every trial and leaks a fresh vocab-sized
        -- fvec each time (finalizer-only; the flat Lua heap means GC never reclaims it).
        local slot = cs_cache and cs_cache[i]
        if slot and slot.e == e then
          cs, wssq = slot.cs, slot.wssq
        else
          cs, wssq = wi:colscale(pcs[i], e, WEIGHT_FLOOR, slot and slot.cs)
          if cs_cache then cs_cache[i] = { e = e, cs = cs, wssq = wssq } end
        end
      elseif pcs and pcs[i] then
        wssq = pcs[i]:sum()
      end
      if wssq then
        sc = (wssq > 0) and (math.sqrt(n / wssq) * sc) or 0
      end
      bl[i] = { x = blocks[i], n_tokens = nc, scale = sc, colscale = cs }
      g0 = g0 + 1
    end
  end
  return bl
end

function M.predict_tiled (o)
  local mtx = require("santoku.mtx")
  local n = o.n
  local tile = o.tile or 4096
  local nl = o.n_labels
  -- single-model tiled prediction. deploy = encode fn, ridge = ridge.
  local function predict_one (deploy, ridge, want_label, want_scores)
    local out = mtx.create({ n_rows = 1, n_cols = 1, type = "f32" })
    local pred, scores, sbuf
    if want_scores then scores = fvec.create(n * nl) end
    for base = 0, n - 1, tile do
      local bs = math.min(tile, n - base)
      local idx = ivec.create(bs)
      for i = 0, bs - 1 do idx:set(i, base + i) end
      local codes
      if o.blocks then
        local ext = {}
        for bi = 1, #o.blocks do
          local b = o.blocks[bi]
          if type(b) == "table" then ext[bi] = { x = b.x:rows(idx), group_offsets = b.group_offsets }
          else ext[bi] = b:rows(idx) end
        end
        codes = deploy(ext, out)
      else
        codes = deploy(o.x:rows(idx), out)
      end
      if want_label then
        local p = ridge:label(codes, o.k)
        if pred then pred:append(p) else pred = p end
      end
      if want_scores then
        sbuf = ridge:regress(codes, sbuf)
        scores:copy(sbuf, 0, bs * nl, base * nl)
      end
      collectgarbage("collect")
    end
    return pred, scores
  end
  -- seed-ensemble path: E.build(s) -> (deploy, ridge) for base s; predict scores over the
  -- full test, average across K bases, then top-k the average for the label heads. Models
  -- are built-predicted-discarded one at a time (peak = 1 model + accumulator + one score
  -- matrix). E is passed as o.ridge (o.ridge.is_ensemble) so specs stay transparent.
  local E = (o.ridge and type(o.ridge) == "table" and o.ridge.is_ensemble) and o.ridge or o.ensemble
  if E then
    nl = nl or E.n_labels -- label heads pass only k; E carries n_labels from krr args
    local S, topk_ridge
    for s = 0, E.K - 1 do
      local deploy, ridge = E.build(s)
      topk_ridge = ridge
      local _, sc = predict_one(deploy, ridge, false, true)
      if not S then S = sc else S:addv(sc) end
      E.release(s)
    end
    S:scale(1.0 / E.K)
    local pred
    if o.k then pred = topk_ridge:topk(S, n, o.k) end
    return pred, (o.scores and S or nil)
  end
  return predict_one(o.deploy, o.ridge, o.k ~= nil, o.scores ~= nil)
end

local function slice_targets (t, idx)
  return (dvec.create(idx:size()):copy(t, idx))
end

local function split_fold (foldof, f, n)
  local tr_idx, va_idx = ivec.create(), ivec.create()
  for i = 0, n - 1 do if foldof:get(i) == f then va_idx:push(i) else tr_idx:push(i) end end
  return tr_idx, va_idx
end

function M.fold_dense (a)
  local K = a.folds or 1
  local n = a.pool_n or select(1, a.pool_codes:shape())
  local reg = a.pool_targets ~= nil
  local use_folds = (a.search_trials or 0) >= 1 and K > 1
  local rel_metric = type(a.relevance) == "table" and a.relevance[1] or a.relevance
  local rel = rel_metric == "auc" and (a.pool_labels ~= nil or a.pool_targets ~= nil)
  local is_csr = a.pool_codes.neighbors ~= nil
  a.pool_strata = pool_strata(a, n)
  -- never-val stratum (fold id -1): leak-free rows that no fold validates on, serving as
  -- both the leak-free relevance-weight fit set and the CV landmark basis. Sized to the CV
  -- landmark budget (strat_size), picked by even index spread so it is identical across
  -- fold seeds/sets/regimes. Fold grams are assembled by moment subtraction from the one
  -- full-pool encode (gram:fold), so no train-side X slicing exists.
  local strat_idx = ivec.create()
  local instrat = {}
  local ssize = strat_size(a, n)
  for i = 0, ssize - 1 do
    local r = math.floor(i * n / ssize)
    if not instrat[r] then instrat[r] = true; strat_idx:push(r) end
  end
  a.stratum_rows = strat_idx
  local function make_split (Kn, user_foldof)
    local foldof0 = user_foldof
      or ((reg and a.pool_targets) and fold_assign_reg(a.pool_targets, n, Kn))
      or fold_assign(a.pool_class or derive_class(a.pool_labels, n), n, Kn)
    local foldof = ivec.create()
    foldof:copy(foldof0)
    for r = 0, n - 1 do if instrat[r] then foldof:set(r, -1) end end
    local fvy, fvn, fvt = {}, {}, {}
    for f = 0, Kn - 1 do
      local _, va_idx = split_fold(foldof, f, n)
      fvn[f + 1] = va_idx:size()
      if reg then fvt[f + 1] = slice_targets(a.pool_targets, va_idx)
      else fvy[f + 1] = a.pool_labels:rows(va_idx) end
    end
    return { assign = foldof, val_y = reg and nil or fvy, val_n = fvn,
      val_targets = reg and fvt or nil }
  end
  local rel_w, cs_cache = {}, {}
  local function rel_weights (cv)
    local key = cv and "c" or "f"
    local hit = rel_w[key]; if hit then return hit end
    local xs = cv and a.pool_codes:rows(strat_idx) or a.pool_codes
    local wy
    if a.pool_labels then wy = cv and a.pool_labels:rows(strat_idx) or a.pool_labels
    else wy = cv and slice_targets(a.pool_targets, strat_idx) or a.pool_targets end
    local w = is_csr and xs:auc(wy) or xs:to_sparse():auc(wy)
    rel_w[key] = w
    return w
  end
  -- CV/cal builds fit relevance weights on the never-val stratum (leak-free); the deploy
  -- build fits on the full pool (no held-out set, nothing to leak into)
  local function dense_colscale (e, cv)
    if type(e) == "table" then e = e[1] end
    if not rel or e == nil or e == false then return nil end
    local key = (cv and "c" or "f") .. tostring(e)
    local hit = cs_cache[key]
    if hit then return hit end
    local w = rel_weights(cv)
    local nc = w:size()
    local logsum = 0
    for c = 0, nc - 1 do
      local v = w:get(c); if v < WEIGHT_FLOOR then v = WEIGHT_FLOOR end
      logsum = logsum + math.log(v)
    end
    local g = math.exp(logsum / nc)
    local cs = fvec.create(nc)
    for c = 0, nc - 1 do
      local v = w:get(c); if v < WEIGHT_FLOOR then v = WEIGHT_FLOOR end
      cs:set(c, (v / g) ^ e)
    end
    cs_cache[key] = cs
    return cs
  end
  a.rebuild = function (p, cv)
    local cs = dense_colscale(p and p.exponent, cv)
    local nc = is_csr and select(2, a.pool_codes:shape()) or nil
    if cs and is_csr then
      return { blocks = { { x = a.pool_codes, colscale = cs, n_tokens = nc } }, n_samples = n }
    end
    return { x = a.pool_codes, n_samples = n, colscale = cs }
  end
  a.folds = use_folds and K or 1
  a.n_samples = a.n_samples or n
  a.y = a.pool_labels
  a.targets = a.pool_targets
  if use_folds then
    a.fold_split = make_split(K, a.fold_assign)
  end
  return a
end

function M.fold_blocks (a)
  local K = a.folds or 1
  local pool, groups = {}, nil
  for i, b in ipairs(a.pool_blocks) do
    if type(b) == "table" then
      local go = b.group_offsets
      if go and not go.size then
        local iv = ivec.create(#go)
        for j = 1, #go do iv:set(j - 1, go[j]) end
        go = iv
      end
      pool[i] = b.x
      if go then groups = groups or {}; groups[i] = go end
    else
      pool[i] = b
    end
  end
  local gd = 0
  for i = 1, #pool do gd = gd + ((groups and groups[i]) and (groups[i]:size() - 1) or 1) end
  a.gauge_dims = gd
  local reg = a.pool_targets ~= nil
  local n = a.pool_n or select(1, pool[1]:shape())
  local metrics = a.relevance
  local wtargets = a.pool_labels == nil and reg
  if metrics and not (a.pool_labels or reg) then metrics = nil end
  local use_folds = K > 1 and (a.search_trials or 0) >= 1
  a.pool_strata = pool_strata(a, n)
  -- never-val weight stratum: label-supervised relevance weights are fit ONLY on rows no
  -- fold ever validates on (fold id -1 in every fold set), so fold-CV scores cannot be
  -- inflated through the weights (the full-pool weight leak let 400-trial searches walk
  -- away from test-good configs). The same rows are the CV landmark basis, so the count is
  -- sized to the CV landmark budget (strat_size): span picks whole docs at an even stride,
  -- dense picks rows at an even index spread; identical across fold seeds/sets/regimes.
  local strat_idx = ivec.create()
  local instrat = {}
  local ssize = strat_size(a, n)
  if a.cand then
    local co = a.cand:offsets()
    local ndocs = co:size() - 1
    local target_docs = math.max(1, math.min(math.ceil(ssize * ndocs / n), math.floor(ndocs * STRAT_CAP)))
    local stride = math.max(1, math.floor(ndocs / target_docs + 0.5))
    for dd = 0, ndocs - 1 do
      if dd % stride == 0 then
        instrat[dd] = true
        for r = co:get(dd), co:get(dd + 1) - 1 do strat_idx:push(r) end
      end
    end
  else
    for i = 0, ssize - 1 do
      local r = math.floor(i * n / ssize)
      if not instrat[r] then instrat[r] = true; strat_idx:push(r) end
    end
  end
  a.stratum_rows = strat_idx
  -- CV downdate: one fold split = an assignment vector + val-side slices; fold grams are
  -- assembled by moment subtraction from the one full-pool encode (gram:fold). The 1/3
  -- never-val stratum (fold id -1) is both leak-free weight rows and the CV landmark basis.
  local function make_split (Kn, user_docfold, user_foldof)
    local foldof, fvc, fvg
    if a.cand then
      local co = a.cand:offsets()
      local ndocs = co:size() - 1
      local docfold0 = user_docfold or M.doc_folds(a.cand, a.gold, Kn)
      local docfold = ivec.create()
      docfold:copy(docfold0)
      for dd = 0, ndocs - 1 do
        if instrat[dd] then docfold:set(dd, -1) end
      end
      foldof = ivec.create(n)
      foldof:fill_segments(co, docfold)
      fvc, fvg = {}, {}
      for f = 0, Kn - 1 do
        local docs = ivec.create()
        for dd = 0, ndocs - 1 do if docfold:get(dd) == f then docs:push(dd) end end
        fvc[f + 1] = a.cand:docs(docs)
        fvg[f + 1] = a.gold:docs(docs)
      end
    else
      local foldof0 = user_foldof
        or ((reg and a.pool_targets) and fold_assign_reg(a.pool_targets, n, Kn))
        or fold_assign(a.pool_class or derive_class(a.pool_labels, n), n, Kn)
      foldof = ivec.create()
      foldof:copy(foldof0)
      for r = 0, n - 1 do if instrat[r] then foldof:set(r, -1) end end
    end
    local fvy, fvn, fvt = {}, {}, {}
    for f = 0, Kn - 1 do
      local _, va_idx = split_fold(foldof, f, n)
      fvn[f + 1] = va_idx:size()
      if reg then fvt[f + 1] = slice_targets(a.pool_targets, va_idx)
      else fvy[f + 1] = a.pool_labels:rows(va_idx) end
    end
    return { assign = foldof, val_y = reg and nil or fvy, val_n = fvn,
      val_targets = reg and fvt or nil, val_cand = fvc, val_gold = fvg }
  end
  -- relevance weights: CV/cal fit on the never-val stratum (leak-free), deploy fits on the
  -- full pool (no held-out set, nothing to leak into)
  local sy = a.pool_labels and a.pool_labels:rows(strat_idx)
    or (wtargets and slice_targets(a.pool_targets, strat_idx)) or nil
  local spool = {}
  for i = 1, #pool do spool[i] = pool[i]:rows(strat_idx) end
  local p_w = weight_fit(spool, sy, metrics, wtargets)
  local p_w_full = weight_fit(pool, a.pool_labels or (wtargets and a.pool_targets), metrics, wtargets)
  local p_pcs = {}
  for i = 1, #pool do p_pcs[i] = pool[i]:sumsq_cols() end
  local exp_tbl
  if a.exponent then
    if a.exponent[1] ~= nil then exp_tbl = a.exponent
    elseif type(a.exponent.def) == "table" then exp_tbl = a.exponent.def end
  end
  if exp_tbl then
    local g0 = 0
    for i = 1, #pool do
      local go = groups and groups[i]
      local ng = go and (go:size() - 1) or 1
      for g = 1, ng do
        local cols = go and (go:get(g) - go:get(g - 1)) or select(2, pool[i]:shape())
        if not p_w_full[i] or cols <= 1 then exp_tbl[g0 + g] = false end
      end
      g0 = g0 + ng
    end
  end
  a.bake_external = function (ext, params)
    local xs = {}
    for i, b in ipairs(ext) do xs[i] = type(b) == "table" and b.x or b end
    return (M.build_blocks(xs, params.scales, params.exponent, n, p_w_full, p_pcs, groups))
  end
  -- separate caches per weight regime (never-val stratum vs full pool) since the pooled
  -- non-grouped colscale is a function of the fit weights. gg caches pool the grouped colscale
  -- OUT-buffer per regime (one persistent fvec per grouped block, fully overwritten each trial)
  -- so the grouped path no longer allocates a fresh fvec every trial.
  local cs_cache_cv, cs_cache_full = {}, {}
  local gg_cache_cv, gg_cache_full = {}, {}
  a.rebuild = function (params, cv)
    local w = cv and p_w or p_w_full
    local cache = cv and cs_cache_cv or cs_cache_full
    local ggc = cv and gg_cache_cv or gg_cache_full
    return { blocks = M.build_blocks(pool, params.scales, params.exponent, n, w, p_pcs, groups, cache, ggc),
      n_samples = n }
  end
  a.folds = use_folds and K or 1
  a.y = a.pool_labels
  a.targets = a.pool_targets
  if use_folds then
    a.fold_split = make_split(K, a.doc_fold, a.fold_assign)
  end
  return a
end

return M
