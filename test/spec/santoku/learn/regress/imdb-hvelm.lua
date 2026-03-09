local arr = require("santoku.array")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local hdc = require("santoku.learn.hdc")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { max = nil, ttr = 0.5, tvr = 0.1 },
  hdc = { d = 2^17, ngram = 5 },
  emb = { n_landmarks = 8192, trace_tol = 0.01, cholesky = true, n_dims = nil },
  ridge = {
    lambda = { def = 6.6967e-03 },
    propensity_a = { def = 3.0326 },
    propensity_b = { def = 4.2558 },
    classes = 2,
    search_trials = 200,
    k = 1,
  },
}

local class_names = { "negative", "positive" }

test("imdb hdc", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test_set, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)
  local n_classes = cfg.ridge.classes
  local label_off, label_nbr = train.solutions:bits_to_csr(train.n, n_classes)
  local val_label_off, val_label_nbr = validate.solutions:bits_to_csr(validate.n, n_classes)
  str.printf("[Data] train=%d val=%d test=%d classes=%d %s\n",
    train.n, validate.n, test_set.n, n_classes, sw())

  local ngram_map, set_bits, n_hdc_tokens = hdc.tokenize({
    texts = train.problems, hdc_ngram = cfg.hdc.ngram, n_samples = train.n,
  })
  local bns_ids, bns_scores = set_bits:bits_top_bns(
    train.solutions, train.n, n_hdc_tokens, n_classes)
  str.printf("[Tokenize] ngram=%d tokens=%d bns=%d %s\n",
    cfg.hdc.ngram, n_hdc_tokens, bns_ids:size(), sw())

  str.printf("[HDC] Encoding d=%d\n", cfg.hdc.d)
  local hdc_enc, train_cvec = hdc.create({
    texts = train.problems, n_samples = train.n,
    d = cfg.hdc.d, hdc_ngram = cfg.hdc.ngram,
    weight_map = ngram_map,
    weight_ids = bns_ids, weights = bns_scores,
  })
  str.printf("[HDC] d=%d encoded %s\n", cfg.hdc.d, sw())

  str.printf("[Spectral] Cholesky trace_tol=%s\n", tostring(cfg.emb.trace_tol))
  local _, _, sp_enc, _, xtx, xty, col_mean, y_mean, label_counts, pre_mean, pre_istd =
    spectral.encode({
      bits = train_cvec, n_samples = train.n, d_bits = cfg.hdc.d,
      n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
      cholesky = cfg.emb.cholesky, n_dims = cfg.emb.n_dims,
      label_offsets = label_off, label_neighbors = label_nbr, n_labels = n_classes,
    })
  local emb_d = sp_enc:dims()
  local val_codes = sp_enc:encode(hdc_enc:encode({ texts = validate.problems, n_samples = validate.n }), validate.n)
  str.printf("[Spectral] emb_d=%d %s\n", emb_d, sw())

  str.printf("[Ridge] Training\n")
  local _, ridge_obj, best_params, _, _, _, std = optimize.ridge({
    XtX = xtx, XtY = xty, col_mean = col_mean, y_mean = y_mean,
    label_counts = label_counts, pre_mean = pre_mean, pre_istd = pre_istd,
    n_samples = train.n, n_dims = emb_d, n_labels = n_classes,
    val_codes = val_codes, val_n_samples = validate.n,
    val_expected_offsets = val_label_off, val_expected_neighbors = val_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  str.printf("[Ridge] lambda=%.4e pa=%.4f pb=%.4f %s\n",
    best_params.lambda, best_params.propensity_a, best_params.propensity_b, sw())

  str.printf("[Eval] Labeling splits\n")
  local _, val_labels = ridge_obj:label(val_codes, validate.n, 1)
  local test_cvec = hdc_enc:encode({ texts = test_set.problems, n_samples = test_set.n })
  local test_codes = sp_enc:encode(test_cvec, test_set.n)
  test_codes:mtx_standardize(emb_d, std.pre_mean, std.pre_istd)
  local _, test_labels = ridge_obj:label(test_codes, test_set.n, 1)
  str.printf("[Eval] Labels done %s\n", sw())

  local val_stats = eval.class_accuracy(val_labels, validate.solutions, validate.n, n_classes)
  local test_stats = eval.class_accuracy(test_labels, test_set.solutions, test_set.n, n_classes)
  str.printf("[Class] F1: val=%.2f test=%.2f %s\n",
    val_stats.f1, test_stats.f1, sw())

  str.printf("\n[Per-class Test Accuracy]\n")
  local class_order = arr.range(1, n_classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    local cat = class_names[c] or ("class_" .. (c - 1))
    str.printf("  %-12s  F1=%.2f  P=%.2f  R=%.2f\n", cat, ts.f1, ts.precision, ts.recall)
  end

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
