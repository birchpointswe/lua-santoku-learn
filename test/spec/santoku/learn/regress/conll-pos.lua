local ds = require("santoku.learn.dataset")
local csr = require("santoku.learn.csr")
local optimize = require("santoku.learn.optimize")
local eval = require("santoku.learn.evaluator")
local ivec = require("santoku.ivec")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { dir = "test/res/conll2003", max = nil },
  tok = { ngram_min = 3, ngram_max = 5, collapse = "none" },
  emb = { n_landmarks = 1024 * 8, trace_tol = 0.01, kernel = { "cosine", "expcos", "geolaplace", "angular", "matern32", "matern52", "rq", "arccos1" } },
  ridge = { lambda = { def = 2.7e-03 }, search_trials = 100 },
}

local function build_tokens (split)
  local off, starts, ends, types = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  local lab_off, lab_nbr = ivec.create(), ivec.create()
  off:push(0)
  lab_off:push(0)
  for d = 1, split.n do
    for _, t in ipairs(split.sent_tokens[d]) do
      starts:push(t.s)
      ends:push(t.e)
      types:push(0)
      lab_nbr:push(t.pos)
      lab_off:push(lab_nbr:size())
    end
    off:push(starts:size())
  end
  return off, starts, ends, types, lab_off, lab_nbr, starts:size()
end

local function tokenize (split, off, starts, ends, types, ngram_map)
  return csr.tokenize_annotated({
    texts = split.texts, doc_span_offsets = off,
    span_starts = starts, span_ends = ends, span_types = types,
    collapse = cfg.tok.collapse,
    ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
    normalize = true, terminals = true, ngram_map = ngram_map,
  })
end

test("conll pos", function ()

  local stopwatch = utc.stopwatch()
  local function sw ()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  local n_pos = train.n_pos
  str.printf("[Data] train=%d dev=%d test=%d n_pos=%d %s\n",
    train.n, dev.n, test_set.n, n_pos, sw())

  local tr_off, tr_s, tr_e, tr_ty, tr_loff, tr_lnbr, tr_n = build_tokens(train)
  local dv_off, dv_s, dv_e, dv_ty, dv_loff, dv_lnbr, dv_n = build_tokens(dev)
  local te_off, te_s, te_e, te_ty, te_loff, te_lnbr, te_n = build_tokens(test_set)
  str.printf("[Tokens] train=%d dev=%d test=%d %s\n", tr_n, dv_n, te_n, sw())

  local ngram_map, offsets, tokens, values, n_tokens =
    tokenize(train, tr_off, tr_s, tr_e, tr_ty, nil)
  local bns = csr.apply_bns(offsets, tokens, values, nil, tr_loff, tr_lnbr, n_tokens, n_pos)
  str.printf("[Tokenize] tokens=%d %s\n", n_tokens, sw())

  local _, dv_o, dv_t, dv_v = tokenize(dev, dv_off, dv_s, dv_e, dv_ty, ngram_map)
  csr.apply_bns(dv_o, dv_t, dv_v, bns)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local sp_enc, ridge_obj, dv_codes = optimize.krr({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = tr_n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_loff, label_neighbors = tr_lnbr, n_labels = n_pos,
    val_offsets = dv_o, val_tokens = dv_t, val_values = dv_v,
    val_n_samples = dv_n,
    val_expected_offsets = dv_loff, val_expected_neighbors = dv_lnbr,
    lambda = cfg.ridge.lambda,
    k = 1, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  offsets = nil; tokens = nil; values = nil -- luacheck: ignore
  collectgarbage("collect")

  local dvp_off, dvp_nbr = ridge_obj:label(dv_codes, dv_n, 1)
  local _, dv_stats = eval.label_accuracy({
    pred_offsets = dvp_off, pred_neighbors = dvp_nbr,
    expected_offsets = dv_loff, expected_neighbors = dv_lnbr, ks = 1,
  })

  local _, te_o, te_t, te_v = tokenize(test_set, te_off, te_s, te_e, te_ty, ngram_map)
  csr.apply_bns(te_o, te_t, te_v, bns)
  local te_codes = sp_enc:encode({
    offsets = te_o, tokens = te_t, values = te_v, n_samples = te_n,
  })
  local tep_off, tep_nbr = ridge_obj:label(te_codes, te_n, 1)
  local _, te_stats = eval.label_accuracy({
    pred_offsets = tep_off, pred_neighbors = tep_nbr,
    expected_offsets = te_loff, expected_neighbors = te_lnbr, ks = 1,
  })

  str.printf("[POS] accuracy: dev=%.4f test=%.4f %s\n",
    dv_stats.micro_f1, te_stats.micro_f1, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
