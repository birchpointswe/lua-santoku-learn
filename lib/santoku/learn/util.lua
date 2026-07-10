local str = require("santoku.string")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local fvec = require("santoku.fvec")

local M = {}

M.WORD_CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

function M.vecstr (v)
  if type(v) ~= "table" then v = { v } end
  local p = {}
  for i = 1, #v do local x = v[i]
    p[i] = (x == math.floor(x)) and str.format("%d", x) or str.format("%.6g", x) end
  return "[" .. table.concat(p, ",") .. "]"
end

function M.build_type_blocks (st, split, Scand, Sctx, is_train, gaz, cgaz, tlab)
  local blocks = {}
  local m1, X1 = st.tok(split, Scand, st.models[1])
  if is_train then st.models[1] = m1 end
  blocks[1] = X1
  local m2, X2 = st.tok_ctx(split, Scand, Sctx, st.models[2])
  if is_train then st.models[2] = m2 end
  blocks[2] = X2
  local n_sparse = #blocks
  blocks[#blocks + 1] = gaz:block(split.texts, Scand, tlab)
  blocks[#blocks + 1] = cgaz:block(split.texts, Scand, tlab)
  return blocks, n_sparse
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
    local g = p.gamma and str.format(" gamma=%.6g", p.gamma) or ""
    return str.format(" kernel=matern nu=%s%s", nu, g)
  elseif p.kernel == "arccos" then
    return str.format(" kernel=arccos n=%d depth=%d %s",
      p.order or 1, p.depth or 1, (p.tangent == 1) and "ntk" or "nngp")
  end
  local g = p.gamma and str.format(" gamma=%.6g", p.gamma) or ""
  return str.format(" kernel=%s%s", p.kernel, g)
end

local function fmt_exponent (ex)
  if type(ex) == "number" then return str.format(" exp=%.6g", ex) end
  if type(ex) ~= "table" then return "" end
  local parts, any = {}, false
  for i = 1, #ex do
    if ex[i] == false or ex[i] == nil then parts[i] = ""
    else parts[i] = str.format("%.6g", ex[i]); any = true end
  end
  if not any then return "" end
  return " exp=[" .. table.concat(parts, ",") .. "]"
end

local function fmt_scales (sc)
  if sc == nil then return "" end
  if type(sc) == "table" then
    local parts = {}
    for i = 1, #sc do parts[i] = sc[i] == false and "1" or str.format("%.6g", sc[i]) end
    return " scales=[" .. table.concat(parts, ",") .. "]"
  end
  return str.format(" scales=%.6g", sc)
end

function M.make_ridge_log (stopwatch, metric_fmt)
  return function (ev)
    if ev.event == "done" then
      local p = ev.params or {}
      local emb = ev.emb_d and str.format(" emb_d=%d", ev.emb_d) or ""
      local md = p.mode and str.format(" mode=%s", p.mode) or ""
      local kdesc = format_kernel(p)
      local solve = ev.solve and str.format(" solve=%s", ev.solve) or ""
      local sc = ev.score and str.format(" score=%.6f", ev.score) or ""
      local scl = fmt_scales(p.scales)
      local exf = fmt_exponent(p.exponent)
      local timing = ""
      if stopwatch then
        local d, dd = stopwatch()
        timing = str.format(" (%.1fs +%.1fs)", d, dd)
      end
      str.printf("[Ridge Done]%s%s%s%s%s%s lambda=%.6g%s%s\n",
        emb, md, kdesc, scl, exf, solve, p.lambda or 0, sc, timing)
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
    -- best = the running-best per-config LCB (durable incumbent; ++ = new best); score = this trial's
    -- raw fold-aggregated CV. detail = the fold-averaged metrics behind that score.
    local best = (ev.best and ev.best ~= -math.huge)
      and str.format(" (best=%.6f%s)", ev.best, ev.is_new_best and " ++" or "") or ""
    local md = metric_fmt and metric_fmt(m) or M.fmt_metrics(m)
    local detail = (md ~= "") and (" " .. md) or ""
    local timing = ""
    if stopwatch then
      local d, dd = stopwatch()
      if ev.gpbo then
        local chol = d - ev.gpbo
        if chol < 0 then chol = 0 end
        timing = str.format(" (%.1fs %.1fs +%.1fs)", chol, ev.gpbo, dd)
      else
        timing = str.format(" (%.1fs +%.1fs)", d, dd)
      end
    end
    local kdesc = format_kernel(p)
    local embd = ev.emb_d and str.format(" emb_d=%d", ev.emb_d) or ""
    local scl = fmt_scales(p.scales)
    local exf = fmt_exponent(p.exponent)
    local lambda = p.lambda or m.lambda or 0
    local off = (m.offset ~= nil) and str.format(" offset=%.4g", m.offset) or ""
    str.printf("[Ridge %s]%s%s%s%s lambda=%.6g score=%.6f%s%s%s%s\n",
      phase, kdesc, embd, scl, exf, lambda, score, detail, off, best, timing)
  end
end

local function tokenize_block_list (specs, texts, toks, focus, extra)
  local tokenizer = require("santoku.learn.tokenizer")
  local grow = toks == nil
  if grow then
    toks = {}
    for i, b in ipairs(specs) do
      local cfg = { ngram_min = b.ngram_min, ngram_max = b.ngram_max,
        word_characters = b.word_characters, words = b.words, normalize = b.normalize }
      if extra then for kk, vv in pairs(extra) do cfg[kk] = vv end end
      toks[i] = tokenizer.create(cfg)
    end
  end
  local X = {}
  for i = 1, #specs do
    local targs = { texts = texts, focus = focus }
    X[i] = grow and toks[i]:fit(targs) or toks[i]:tokenize(targs)
  end
  return toks, X
end

function M.tokenize_blocks (specs, texts, toks)
  return tokenize_block_list(specs, texts, toks)
end

function M.tokenize_focus_blocks (specs, texts, F, toks)
  return tokenize_block_list(specs, texts, toks, F, { terminals = true, focus = true })
end

local function label_is_multilabel (labels, n, n_labels)
  if (n_labels or 0) <= 1 then return true end
  if not labels then return false end
  local eo = labels:offsets()
  for i = 0, n - 1 do
    if eo:get(i + 1) - eo:get(i) ~= 1 then return true end
  end
  return false
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

-- regression folds: interleave by target rank (fold = rank % K over targets sorted ascending) so each
-- fold spans the whole target distribution -- quantile stratification. Replaces the arbitrary i%K index
-- partition, which leaves fold means at the mercy of any ordering in the data and inflates cross-fold
-- variance (the noise the search overfits). Continuous-target analogue of class stratification.
local function fold_assign_reg (targets, n, K)
  local order = targets:argsort()
  local foldof = ivec.create(n)
  for r = 0, n - 1 do foldof:set(order:get(r), r % K) end
  return foldof
end

-- Per-row stratum id from a labels csr, for classification fold stratification (fed to fold_assign):
--   single-label (every row has exactly one label) -> that label id.
--   multilabel/presence -> the RAREST label present in the row (its least-frequent label globally).
-- The rarest-label proxy is the greedy core of iterative multilabel stratification: balancing the
-- scarcest labels across folds is what dominates multilabel CV variance. Rows with no labels get their
-- own stratum (-1). Returns nil when there are no labels (caller falls back to i%K).
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

-- span-mode doc stratum: bucket docs by gold-span count (capped), so folds get a balanced share of
-- dense vs sparse documents rather than an arbitrary doc-index round-robin.
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

-- Per-row landmark stratum id, mirroring the fold-stratification target logic so uniform landmark
-- selection can guarantee representation of every target class/bin (not just proportional-in-expectation):
--   span (cand) -> candidate type; regression -> quantile bin; else -> derive_class (single/multilabel).
-- Returns nil when there is no target signal (caller selects landmarks unstratified).
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
  end
end

local WEIGHT_FLOOR = 1e-6
function M.build_blocks (blocks, scale, exponent, n, w, pcs)
  local bl = {}
  for i = 1, #blocks do
    local nc = select(2, blocks[i]:shape())
    local sc = scale and scale[i] or 1.0
    local wi = w and w[i]
    local cs, wssq
    if wi and pcs and pcs[i] then
      local e = exponent and exponent[i]
      if e == nil or e == false then e = 1.0 end
      cs, wssq = wi:colscale(pcs[i], e, WEIGHT_FLOOR)
    elseif pcs and pcs[i] then
      wssq = pcs[i]:sum()
    end
    if wssq then
      sc = (wssq > 0) and (math.sqrt(n / wssq) * sc) or 0
    end
    bl[i] = { x = blocks[i], n_tokens = nc, scale = sc, colscale = cs }
  end
  return bl
end

-- Encode + predict in row tiles so deploy never materializes an n x emb_d code matrix.
-- o: { deploy, ridge, n, blocks = {csr,...} (or x = csr), k = top-k labels (optional),
--      scores = true (optional; requires n_labels), n_labels, tile = rows per tile (default 4096) }
-- Returns pred csr (when k), scores fvec (n x n_labels, when scores).
function M.predict_tiled (o)
  local mtx = require("santoku.mtx")
  local n = o.n
  local tile = o.tile or 4096
  local out = mtx.create({ n_rows = 1, n_cols = 1, type = "f32" })
  local pred, scores, sbuf
  if o.scores then scores = fvec.create(n * o.n_labels) end
  for base = 0, n - 1, tile do
    local bs = math.min(tile, n - base)
    local idx = ivec.create(bs)
    for i = 0, bs - 1 do idx:set(i, base + i) end
    local codes
    if o.blocks then
      local ext = {}
      for bi = 1, #o.blocks do ext[bi] = o.blocks[bi]:rows(idx) end
      codes = o.deploy(ext, out)
    else
      codes = o.deploy(o.x:rows(idx), out)
    end
    if o.k then
      local p = o.ridge:label(codes, o.k)
      if pred then pred:append(p) else pred = p end
    end
    if scores then
      sbuf = o.ridge:regress(codes, sbuf)
      scores:copy(sbuf, 0, bs * o.n_labels, base * o.n_labels)
    end
    collectgarbage("collect")
  end
  return pred, scores
end

function M.columns_as_blocks (m)
  local _, n_cols = m:shape()
  local blks = {}
  local id = ivec.create(); id:push(0)
  for j = 0, n_cols - 1 do
    id:set(0, j)
    blks[j + 1] = m:cols(id):to_sparse():i32()
  end
  return blks
end

local function slice_targets (t, idx)
  return (dvec.create(idx:size()):copy(t, idx))
end

local function slice_ivec (s, idx)
  if not s then return nil end
  return (ivec.create(idx:size()):copy(s, idx))
end

local function split_fold (foldof, f, n)
  local tr_idx, va_idx = ivec.create(), ivec.create()
  for i = 0, n - 1 do if foldof:get(i) == f then va_idx:push(i) else tr_idx:push(i) end end
  return tr_idx, va_idx
end

-- Dense-input CV path (pass pool_codes -- a dense feature/embedding matrix or code CSR -- instead of
-- tokenized pool_blocks). Row-slices the codes into K folds; AUC-based per-slot colscale; all-CV, no
-- external dev set (finalize builds on the full pool; the caller deploys via the returned closure).
function M.fold_dense (a)
  local K = a.folds or 1
  local n = a.pool_n or select(1, a.pool_codes:shape())
  local reg = a.pool_targets ~= nil
  local use_folds = (a.search_trials or 0) > 0 and K > 1
  local rel_metric = type(a.relevance) == "table" and a.relevance[1] or a.relevance
  local rel = rel_metric == "auc" and (a.pool_labels ~= nil or a.pool_targets ~= nil)
  local is_csr = a.pool_codes.neighbors ~= nil
  local F = {}
  local fy, fvy, fvn, ft, fvt, fs = {}, {}, {}, {}, {}, {}
  a.pool_strata = pool_strata(a, n)
  if use_folds then
    local foldof = a.fold_assign
      or ((reg and a.pool_targets) and fold_assign_reg(a.pool_targets, n, K))
      or fold_assign(a.pool_class or derive_class(a.pool_labels, n), n, K)
    for f = 0, K - 1 do
      local tr_idx, va_idx = split_fold(foldof, f, n)
      local e = { tr = a.pool_codes:rows(tr_idx), va = a.pool_codes:rows(va_idx),
        n = tr_idx:size(), vn = va_idx:size() }
      if reg then e.t = slice_targets(a.pool_targets, tr_idx); e.vt = slice_targets(a.pool_targets, va_idx)
      else e.y = a.pool_labels:rows(tr_idx); e.vy = a.pool_labels:rows(va_idx) end
      fs[f + 1] = slice_ivec(a.pool_strata, tr_idx)
      F[f + 1] = e
    end
    for f = 1, K do
      fvn[f] = F[f].vn
      if reg then ft[f] = F[f].t; fvt[f] = F[f].vt else fy[f] = F[f].y; fvy[f] = F[f].vy end
    end
  end
  local rel_slot, cs_cache = {}, {}
  local function dense_colscale (slot, e)
    if type(e) == "table" then e = e[1] end
    if not rel or e == nil or e == false then return nil end
    local key = slot .. ":" .. tostring(e)
    local hit = cs_cache[key]
    if hit then return hit end
    local w = rel_slot[slot]
    if not w then
      local codes = slot == 0 and a.pool_codes or F[slot].tr
      local y = slot == 0 and (a.pool_labels or a.pool_targets) or (F[slot].y or F[slot].t)
      w = is_csr and codes:auc(y) or codes:to_sparse():auc(y)
      rel_slot[slot] = w
    end
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
  a.rebuild = function (p, f)
    -- f = CV fold (encode held-out d.va); f = nil = finalize on full pool (no held-out).
    local d = f and F[f] or { tr = a.pool_codes, n = n }
    local cs = dense_colscale(f or 0, p and p.exponent)
    local nc = is_csr and select(2, a.pool_codes:shape()) or nil
    local out = cs and is_csr
      and { blocks = { { x = d.tr, colscale = cs, n_tokens = nc } }, n_samples = d.n }
      or { x = d.tr, n_samples = d.n, colscale = cs }
    if d.va then
      if cs and is_csr then out.val_blocks = { { x = d.va, colscale = cs, n_tokens = nc } }
      else out.val_x = d.va end
      out.val_n_samples = d.vn
    end
    return out
  end
  a.folds = use_folds and K or 1
  a.n_samples = a.n_samples or n
  a.y = a.pool_labels
  a.targets = a.pool_targets
  a.fold_y = reg and nil or fy; a.fold_val_y = reg and nil or fvy; a.fold_val_n = fvn
  a.fold_targets = reg and ft or nil; a.fold_val_targets = reg and fvt or nil
  a.fold_strata = fs
  return a
end

function M.fold_blocks (a)
  local K = a.folds or 1
  local pool = a.pool_blocks
  local reg = a.pool_targets ~= nil
  local n = a.pool_n or select(1, pool[1]:shape())
  local metrics = a.relevance
  local wtargets = a.pool_labels == nil and reg
  if metrics and not (a.pool_labels or reg) then metrics = nil end
  local searching = (a.search_trials or 0) > 0
  local off = a.decode_offset
  if type(off) == "table" then off = (not searching) and off.def or nil end
  local use_folds = K > 1 and (searching
    or (off == nil and (a.val_cand or a.cand
      or (not reg and label_is_multilabel(a.pool_labels, n, a.n_labels)))))
  local F = {}
  local fs = {}
  a.pool_strata = pool_strata(a, n)
  if use_folds then
    local foldof
    if a.cand then
      local co = a.cand:offsets()
      local ndocs = co:size() - 1
      -- doc->fold via gold-count-stratified round-robin (balances dense/sparse docs across folds)
      local buckets, order = doc_strata(a.gold, ndocs)
      local docfold = ivec.create(ndocs)
      for _, gc in ipairs(order) do
        local docs = buckets[gc]
        for j = 1, #docs do docfold:set(docs[j], (j - 1) % K) end
      end
      foldof = ivec.create(n)
      foldof:fill_segments(co, docfold)
      local fvc, fvg = {}, {}
      for f = 0, K - 1 do
        local docs = ivec.create()
        for d = 0, ndocs - 1 do if docfold:get(d) == f then docs:push(d) end end
        fvc[f + 1] = a.cand:docs(docs)
        fvg[f + 1] = a.gold:docs(docs)
      end
      a.fold_val_cand, a.fold_val_gold = fvc, fvg
    else
      foldof = a.fold_assign
        or ((reg and a.pool_targets) and fold_assign_reg(a.pool_targets, n, K))
        or fold_assign(a.pool_class or derive_class(a.pool_labels, n), n, K)
    end
    for f = 0, K - 1 do
      local tr_idx, va_idx = split_fold(foldof, f, n)
      -- slices are transient here (w/pcs fit only); rebuild() re-slices lazily per call so at most
      -- one fold's copies are live at a time instead of all K held for the whole search
      local trb = {}
      for i = 1, #pool do trb[i] = pool[i]:rows(tr_idx) end
      local ty = a.pool_labels and a.pool_labels:rows(tr_idx)
      local vy = a.pool_labels and a.pool_labels:rows(va_idx)
      local ft, fvt
      if reg then ft = slice_targets(a.pool_targets, tr_idx); fvt = slice_targets(a.pool_targets, va_idx) end
      local w_f = weight_fit(trb, ty or (wtargets and ft), metrics, wtargets)
      local pcs_f = {}
      for i = 1, #trb do pcs_f[i] = trb[i]:sumsq_cols() end
      trb = nil -- luacheck: ignore
      local e = { tr_idx = tr_idx, va_idx = va_idx, w = w_f, pcs = pcs_f,
        n = tr_idx:size(), vn = va_idx:size(), y = ty, vy = vy, t = ft, vt = fvt }
      fs[f + 1] = slice_ivec(a.pool_strata, tr_idx)
      F[f + 1] = e
      collectgarbage("collect")
    end
  end
  local p_w = weight_fit(pool, a.pool_labels or (wtargets and a.pool_targets), metrics, wtargets)
  local p_pcs = {}
  for i = 1, #pool do p_pcs[i] = pool[i]:sumsq_cols() end
  local exp_tbl
  if a.exponent then
    if a.exponent[1] ~= nil then exp_tbl = a.exponent
    elseif type(a.exponent.def) == "table" then exp_tbl = a.exponent.def end
  end
  if exp_tbl then
    for i = 1, #pool do
      if not p_w[i] or select(2, pool[i]:shape()) <= 1 then exp_tbl[i] = false end
    end
  end
  a.bake_external = function (ext, params)
    return (M.build_blocks(ext, params.scales, params.exponent, n, p_w, p_pcs))
  end
  a.rebuild = function (params, f)
    -- f = a CV fold (train on the fold's row slices, encode the held-out slice); f = nil = finalize on
    -- the full pool (no held-out -- deploy encodes the caller's test set outside krr). Fold slices are
    -- materialized here per call and dropped with the previous trial's garbage.
    local d = f and F[f] or { w = p_w, pcs = p_pcs, n = n }
    local trb, vab
    if f then
      trb, vab = {}, {}
      for i = 1, #pool do trb[i] = pool[i]:rows(d.tr_idx); vab[i] = pool[i]:rows(d.va_idx) end
    else
      trb = pool
    end
    local out = { blocks = M.build_blocks(trb, params.scales, params.exponent, d.n, d.w, d.pcs),
      n_samples = d.n }
    if vab then
      out.val_blocks = M.build_blocks(vab, params.scales, params.exponent, d.n, d.w, d.pcs)
      out.val_n_samples = d.vn
    end
    return out
  end
  local fy, fvy, fvn, ft, fvt = {}, {}, {}, {}, {}
  if use_folds then for f = 1, K do
    fvn[f] = F[f].vn
    if reg then ft[f] = F[f].t; fvt[f] = F[f].vt else fy[f] = F[f].y; fvy[f] = F[f].vy end
  end end
  a.folds = use_folds and K or 1
  a.y = a.pool_labels
  a.targets = a.pool_targets
  a.fold_y = reg and nil or fy; a.fold_val_y = reg and nil or fvy; a.fold_val_n = fvn
  a.fold_targets = reg and ft or nil; a.fold_val_targets = reg and fvt or nil
  a.fold_strata = fs
  return a
end

return M
