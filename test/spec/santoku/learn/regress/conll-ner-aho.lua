local ds = require("santoku.learn.dataset")
local csr = require("santoku.learn.csr")
local aho = require("santoku.learn.aho")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local ridge = require("santoku.learn.ridge")
local ivec = require("santoku.ivec")
local fvec = require("santoku.fvec")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { dir = "test/res/conll2003", max = nil },
  tok = { ngram_min = 3, ngram_max = 5, collapse = "none", normalize = true },
  emb = { n_landmarks = 1024 * 8, trace_tol = 0.01, kernel = { "cosine", "expcos", "geolaplace", "angular", "matern32", "matern52", "rq", "arccos1" } },
  ridge = {
    lambda = { min = 1e-4, max = 1e1, log = true, def = 1.72e-2 },
    search_trials = 100,
  },
  stack = { enabled = true, k = 5, collapse = "spans" },
}

local boundary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

local function build_gazetteer (train)
  local counts = {}
  for d = 1, train.n do
    local text = train.texts[d]
    for _, e in ipairs(train.sent_ents[d]) do
      local surf = text:sub(e.s + 1, e.e)
      local c = counts[surf]
      if not c then c = { 0, 0, 0, 0 }; counts[surf] = c end
      c[e.t + 1] = c[e.t + 1] + 1
    end
  end
  local patterns = {}
  local ids = ivec.create()
  for surf, c in pairs(counts) do
    local bt, bn = 0, -1
    for t = 0, 3 do if c[t + 1] > bn then bn = c[t + 1]; bt = t end end
    patterns[#patterns + 1] = surf
    ids:push(bt)
  end
  return aho.create({ ids = ids, patterns = patterns, normalize = true })
end

local function build_candidates (ac, split)
  local off, mids, starts, ends = ac:predict({
    texts = split.texts, longest = true, boundary = boundary,
  })
  local labels = ivec.create()
  for d = 0, split.n - 1 do
    local gold = {}
    for _, e in ipairs(split.sent_ents[d + 1]) do
      gold[e.s .. ":" .. e.e] = true
    end
    for j = off:get(d), off:get(d + 1) - 1 do
      labels:push(gold[starts:get(j) .. ":" .. ends:get(j)] and 1 or 0)
    end
  end
  return off, starts, ends, mids, labels
end

local function gold_count (split)
  local g = 0
  for d = 1, split.n do g = g + #split.sent_ents[d] end
  return g
end

local function tokenize (split, off, starts, ends, types, ngram_map)
  return csr.tokenize_annotated({
    texts = split.texts, doc_span_offsets = off,
    span_starts = starts, span_ends = ends, span_types = types,
    collapse = cfg.tok.collapse,
    ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
    normalize = cfg.tok.normalize, terminals = true, ngram_map = ngram_map,
  })
end

-- L2: render each candidate (focus) against the doc backdrop of L1-accepted spans (context)
local function tokenize_ctx (split, off, starts, ends, types, coff, cs, ce, cty, ngram_map)
  return csr.tokenize_annotated({
    texts = split.texts, doc_span_offsets = off,
    span_starts = starts, span_ends = ends, span_types = types,
    context_offsets = coff, context_starts = cs, context_ends = ce, context_types = cty,
    collapse = cfg.stack.collapse,
    ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
    normalize = cfg.tok.normalize, terminals = true, ngram_map = ngram_map,
  })
end

local function build_fold (off, n, k)
  local fold = ivec.create(n)
  for d = 0, off:size() - 2 do
    fold:fill(d % k, off:get(d), off:get(d + 1))
  end
  return fold
end

test("conll ner aho", function ()

  local stopwatch = utc.stopwatch()
  local function sw ()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  str.printf("[Data] train=%d dev=%d test=%d %s\n", train.n, dev.n, test_set.n, sw())

  local ac = build_gazetteer(train)
  local tr_off, tr_s, tr_e, tr_ty, tr_lab = build_candidates(ac, train)
  local dv_off, dv_s, dv_e, dv_ty, dv_lab = build_candidates(ac, dev)
  local te_off, te_s, te_e, te_ty, te_lab = build_candidates(ac, test_set)
  local tr_n, dv_n, te_n = tr_lab:size(), dv_lab:size(), te_lab:size()
  str.printf("[Cand] train=%d (pos=%d) dev=%d test=%d %s\n",
    tr_n, tr_lab:sum(), dv_n, te_n, sw())
  str.printf("[Coverage] train=%.4f dev=%.4f test=%.4f (gazetteer recall ceiling)\n",
    tr_lab:sum() / gold_count(train),
    dv_lab:sum() / gold_count(dev),
    te_lab:sum() / gold_count(test_set))

  local ngram_map, offsets, tokens, values, n_tokens =
    tokenize(train, tr_off, tr_s, tr_e, tr_ty, nil)
  local tr_loff, tr_lnbr = csr.binary_label_csr(tr_lab)
  -- L2 OOF refits L1 on raw counts per fold; snapshot before BNS mutates `values` in place
  local stack_on = cfg.stack and cfg.stack.enabled
  local raw_off, raw_tok, raw_val
  if stack_on then raw_off, raw_tok, raw_val = offsets, tokens, fvec.create(values) end
  local bns = csr.apply_bns(offsets, tokens, values, nil, tr_loff, tr_lnbr, n_tokens, 1)
  csr.normalize(offsets, values)
  str.printf("[Tokenize] tokens=%d %s\n", n_tokens, sw())

  local _, dv_o, dv_t, dv_v = tokenize(dev, dv_off, dv_s, dv_e, dv_ty, ngram_map)
  csr.apply_bns(dv_o, dv_t, dv_v, bns)
  csr.normalize(dv_o, dv_v)
  local dv_loff, dv_lnbr = csr.binary_label_csr(dv_lab)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local sp_enc, ridge_obj, dv_codes, best_params = optimize.krr({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = tr_n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_loff, label_neighbors = tr_lnbr, n_labels = 1,
    val_offsets = dv_o, val_tokens = dv_t, val_values = dv_v,
    val_n_samples = dv_n,
    val_expected_offsets = dv_loff, val_expected_neighbors = dv_lnbr,
    lambda = cfg.ridge.lambda,
    k = 1, search_trials = 0,  -- L1 frozen at known-best (kernel=cosine, lambda=def); no search
    each = util.make_ridge_log(stopwatch),
    trial_fn = function (gram, params)
      local f1, p, r = gram:label_accuracy(params.lambda, 1, nil, nil,
        dv_loff, dv_lnbr, "gfm")
      return f1, { f1 = f1, precision = p, recall = r }
    end,
  })

  do  -- persist/load parity: round-tripped encoder must produce identical codes
    local pth = os.tmpname()
    sp_enc:persist(pth)
    local enc2 = spectral.load(pth)
    os.remove(pth)
    local dvc2 = enc2:encode({ offsets = dv_o, tokens = dv_t, values = dv_v, n_samples = dv_n })
    local nchk = dv_codes:size()
    if nchk > 100000 then nchk = 100000 end
    for i = 0, nchk - 1 do
      assert(dv_codes:get(i) == dvc2:get(i), "persist/load parity mismatch at " .. i)
    end
    str.printf("[Persist] load parity OK (%d codes)\n", nchk)
  end

  local d_off, d_nbr, d_sco = ridge_obj:label(dv_codes, dv_n, 1)
  local gfm_obj, gfm_m = optimize.gfm({
    n_labels = 1,
    val_offsets = d_off, val_neighbors = d_nbr, val_scores = d_sco,
    val_n_samples = dv_n,
    val_expected_offsets = dv_loff, val_expected_neighbors = dv_lnbr,
  })
  str.printf("[Params] kernel=%s lambda=%.4e\n", best_params.kernel, best_params.lambda)
  str.printf("[GFM] dev: f1=%.4f p=%.4f r=%.4f %s\n",
    gfm_m.f1, gfm_m.precision, gfm_m.recall, sw())

  local tr_codes = sp_enc:encode({
    offsets = offsets, tokens = tokens, values = values, n_samples = tr_n,
  })
  local trp_off, trp_nbr, trp_sco = ridge_obj:label(tr_codes, tr_n, 1)
  local _, tr_stats = gfm_obj:score({
    offsets = trp_off, neighbors = trp_nbr, scores = trp_sco,
    expected_offsets = tr_loff, expected_neighbors = tr_lnbr, n_samples = tr_n,
  })
  str.printf("[Train] f1=%.4f p=%.4f r=%.4f %s\n",
    tr_stats.micro_f1, tr_stats.micro_precision, tr_stats.micro_recall, sw())
  offsets = nil; tokens = nil; values = nil; tr_codes = nil -- luacheck: ignore
  collectgarbage("collect")

  local _, te_o, te_t, te_v = tokenize(test_set, te_off, te_s, te_e, te_ty, ngram_map)
  csr.apply_bns(te_o, te_t, te_v, bns)
  csr.normalize(te_o, te_v)
  local te_codes = sp_enc:encode({
    offsets = te_o, tokens = te_t, values = te_v, n_samples = te_n,
  })
  local t_off, t_nbr, t_sco = ridge_obj:label(te_codes, te_n, 1)
  local te_loff, te_lnbr = csr.binary_label_csr(te_lab)
  local _, te_stats = gfm_obj:score({
    offsets = t_off, neighbors = t_nbr, scores = t_sco,
    expected_offsets = te_loff, expected_neighbors = te_lnbr,
    n_samples = te_n,
  })
  local cov = te_lab:sum() / gold_count(test_set)
  local p, rc = te_stats.micro_precision, te_stats.micro_recall
  local rt = rc * cov
  local f1t = (p + rt > 0) and (2 * p * rt / (p + rt)) or 0
  local f1ceil = 2 * cov / (1 + cov)
  str.printf("[Test] cand: f1=%.4f p=%.4f r=%.4f %s\n",
    te_stats.micro_f1, p, rc, sw())
  str.printf("[Test] true-gold: f1=%.4f p=%.4f r=%.4f | coverage=%.4f ceiling-f1=%.4f\n",
    f1t, p, rt, cov, f1ceil)

  if stack_on then
    local kk = cfg.stack.k

    -- L1 out-of-fold scores on train (fold by document; BNS refit per fold to avoid leakage)
    local fold = build_fold(tr_off, tr_n, kk)
    local oof_scores = optimize.oof({
      n = tr_n, k = kk, fold = fold,
      each = function (ev)
        str.printf("[OOF] fold %d/%d n_train=%d n_eval=%d %s\n",
          ev.fold, ev.folds, ev.n_train, ev.n_eval, sw())
      end,
      fit = function (train_idx)
        local s_off, s_tok, s_val = csr.gather_rows(raw_off, raw_tok, raw_val, train_idx)
        local s_lab = ivec.create(); s_lab:copy(tr_lab, train_idx)
        local s_loff, s_lnbr = csr.binary_label_csr(s_lab)
        local s_bns = csr.apply_bns(s_off, s_tok, s_val, nil, s_loff, s_lnbr, n_tokens, 1)
        csr.normalize(s_off, s_val)
        local _, sp, gram = spectral.encode({
          offsets = s_off, tokens = s_tok, values = s_val,
          n_tokens = n_tokens, n_samples = train_idx:size(),
          kernel = best_params.kernel,
          n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
          label_offsets = s_loff, label_neighbors = s_lnbr, n_labels = 1,
        })
        return { sp = sp, r = ridge.create({ gram = gram, lambda = best_params.lambda }), bns = s_bns }
      end,
      predict = function (h, eval_idx)
        local s_off, s_tok, s_val = csr.gather_rows(raw_off, raw_tok, raw_val, eval_idx)
        csr.apply_bns(s_off, s_tok, s_val, h.bns)
        csr.normalize(s_off, s_val)
        local codes = h.sp:encode({ offsets = s_off, tokens = s_tok, values = s_val, n_samples = eval_idx:size() })
        local _, _, sco = h.r:label(codes, eval_idx:size(), 1)
        return sco
      end,
    })

    -- threshold to accepts with the final-L1 GFM (single threshold for all splits)
    local oof_off = ivec.create(tr_n + 1); oof_off:fill_indices()
    local oof_nbr = ivec.create(tr_n); oof_nbr:zero()
    local tr_ks = gfm_obj:predict({ offsets = oof_off, neighbors = oof_nbr, scores = oof_scores, n_samples = tr_n })
    local dv_ks = gfm_obj:predict({ offsets = d_off, neighbors = d_nbr, scores = d_sco, n_samples = dv_n })
    local te_ks = gfm_obj:predict({ offsets = t_off, neighbors = t_nbr, scores = t_sco, n_samples = te_n })
    str.printf("[Stack] accepts train=%d dev=%d test=%d (cands %d/%d/%d) %s\n",
      tr_ks:sum(), dv_ks:sum(), te_ks:sum(), tr_n, dv_n, te_n, sw())

    -- context backdrop per split, then L2 features (focus = all candidates)
    local tr_co, tr_cs, tr_ce, tr_cy = csr.filter_spans(tr_off, tr_s, tr_e, tr_ty, tr_ks)
    local dv_co, dv_cs, dv_ce, dv_cy = csr.filter_spans(dv_off, dv_s, dv_e, dv_ty, dv_ks)
    local te_co, te_cs, te_ce, te_cy = csr.filter_spans(te_off, te_s, te_e, te_ty, te_ks)

    local ngm2, l2_off, l2_tok, l2_val, l2_ntok =
      tokenize_ctx(train, tr_off, tr_s, tr_e, tr_ty, tr_co, tr_cs, tr_ce, tr_cy, nil)
    local l2_bns = csr.apply_bns(l2_off, l2_tok, l2_val, nil, tr_loff, tr_lnbr, l2_ntok, 1)
    csr.normalize(l2_off, l2_val)

    local _, l2_dvo, l2_dvt, l2_dvv =
      tokenize_ctx(dev, dv_off, dv_s, dv_e, dv_ty, dv_co, dv_cs, dv_ce, dv_cy, ngm2)
    csr.apply_bns(l2_dvo, l2_dvt, l2_dvv, l2_bns)
    csr.normalize(l2_dvo, l2_dvv)

    str.printf("[L2 KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
    local sp2, ridge2, l2_dvcodes, bp2 = optimize.krr({
      offsets = l2_off, tokens = l2_tok, values = l2_val,
      n_samples = tr_n, n_tokens = l2_ntok,
      kernel = cfg.emb.kernel,
      n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
      label_offsets = tr_loff, label_neighbors = tr_lnbr, n_labels = 1,
      val_offsets = l2_dvo, val_tokens = l2_dvt, val_values = l2_dvv, val_n_samples = dv_n,
      val_expected_offsets = dv_loff, val_expected_neighbors = dv_lnbr,
      lambda = cfg.ridge.lambda, k = 1, search_trials = 0,  -- L2 frozen at known-best (cosine, lambda=def)
      each = util.make_ridge_log(stopwatch),
      trial_fn = function (gram, params)
        local f1, pp, rr = gram:label_accuracy(params.lambda, 1, nil, nil, dv_loff, dv_lnbr, "gfm")
        return f1, { f1 = f1, precision = pp, recall = rr }
      end,
    })

    local l2_do, l2_dn, l2_ds = ridge2:label(l2_dvcodes, dv_n, 1)
    local gfm2, gfm2_m = optimize.gfm({
      n_labels = 1,
      val_offsets = l2_do, val_neighbors = l2_dn, val_scores = l2_ds, val_n_samples = dv_n,
      val_expected_offsets = dv_loff, val_expected_neighbors = dv_lnbr,
    })
    str.printf("[L2 Params] kernel=%s lambda=%.4e\n", bp2.kernel, bp2.lambda)
    str.printf("[L2 GFM] dev: f1=%.4f p=%.4f r=%.4f %s\n",
      gfm2_m.f1, gfm2_m.precision, gfm2_m.recall, sw())

    local _, l2_teo, l2_tet, l2_tev =
      tokenize_ctx(test_set, te_off, te_s, te_e, te_ty, te_co, te_cs, te_ce, te_cy, ngm2)
    csr.apply_bns(l2_teo, l2_tet, l2_tev, l2_bns)
    csr.normalize(l2_teo, l2_tev)
    local l2_tecodes = sp2:encode({ offsets = l2_teo, tokens = l2_tet, values = l2_tev, n_samples = te_n })
    local l2_to, l2_tn, l2_ts = ridge2:label(l2_tecodes, te_n, 1)
    local _, l2_stats = gfm2:score({
      offsets = l2_to, neighbors = l2_tn, scores = l2_ts,
      expected_offsets = te_loff, expected_neighbors = te_lnbr, n_samples = te_n,
    })
    local p2, rc2 = l2_stats.micro_precision, l2_stats.micro_recall
    local rt2 = rc2 * cov
    local f1t2 = (p2 + rt2 > 0) and (2 * p2 * rt2 / (p2 + rt2)) or 0
    str.printf("[L2 Test] cand: f1=%.4f p=%.4f r=%.4f %s\n",
      l2_stats.micro_f1, p2, rc2, sw())
    str.printf("[L2 Test] true-gold: f1=%.4f p=%.4f r=%.4f | coverage=%.4f ceiling-f1=%.4f\n",
      f1t2, p2, rt2, cov, f1ceil)
  end

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
