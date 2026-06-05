local ds = require("santoku.learn.dataset")
local csr = require("santoku.learn.csr")
local aho = require("santoku.learn.aho")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local ivec = require("santoku.ivec")
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
    lambda = { min = 1e-4, max = 1e1, log = true, def = 1e-1 },
    search_trials = 100,
  },
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

local function build_label_csr (labels, n)
  local off, nbr = ivec.create(), ivec.create()
  off:push(0)
  for i = 0, n - 1 do
    if labels:get(i) == 1 then nbr:push(0) end
    off:push(nbr:size())
  end
  return off, nbr
end

local function count_pos (labels)
  local p = 0
  for i = 0, labels:size() - 1 do if labels:get(i) == 1 then p = p + 1 end end
  return p
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

test("conll ner", function ()

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
    tr_n, count_pos(tr_lab), dv_n, te_n, sw())
  str.printf("[Coverage] train=%.4f dev=%.4f test=%.4f (gazetteer recall ceiling)\n",
    count_pos(tr_lab) / gold_count(train),
    count_pos(dv_lab) / gold_count(dev),
    count_pos(te_lab) / gold_count(test_set))

  local ngram_map, offsets, tokens, values, n_tokens =
    tokenize(train, tr_off, tr_s, tr_e, tr_ty, nil)
  local tr_loff, tr_lnbr = build_label_csr(tr_lab, tr_n)
  local bns = csr.apply_bns(offsets, tokens, values, nil, tr_loff, tr_lnbr, n_tokens, 1)
  csr.normalize(offsets, values)
  str.printf("[Tokenize] tokens=%d %s\n", n_tokens, sw())

  local _, dv_o, dv_t, dv_v = tokenize(dev, dv_off, dv_s, dv_e, dv_ty, ngram_map)
  csr.apply_bns(dv_o, dv_t, dv_v, bns)
  csr.normalize(dv_o, dv_v)
  local dv_loff, dv_lnbr = build_label_csr(dv_lab, dv_n)

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
    k = 1, search_trials = cfg.ridge.search_trials,
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
  local te_loff, te_lnbr = build_label_csr(te_lab, te_n)
  local _, te_stats = gfm_obj:score({
    offsets = t_off, neighbors = t_nbr, scores = t_sco,
    expected_offsets = te_loff, expected_neighbors = te_lnbr,
    n_samples = te_n,
  })
  local cov = count_pos(te_lab) / gold_count(test_set)
  local p, rc = te_stats.micro_precision, te_stats.micro_recall
  local rt = rc * cov
  local f1t = (p + rt > 0) and (2 * p * rt / (p + rt)) or 0
  local f1ceil = 2 * cov / (1 + cov)
  str.printf("[Test] cand: f1=%.4f p=%.4f r=%.4f %s\n",
    te_stats.micro_f1, p, rc, sw())
  str.printf("[Test] true-gold: f1=%.4f p=%.4f r=%.4f | coverage=%.4f ceiling-f1=%.4f\n",
    f1t, p, rt, cov, f1ceil)

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
