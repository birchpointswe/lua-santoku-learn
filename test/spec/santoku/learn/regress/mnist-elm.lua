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
  data = { ttr = 0.8, tvr = 0.1, max = nil, features = 784 },
  hdc = { d = 2^16, ngram = 5, row_length = 28 },
  emb = { n_landmarks = 8192, trace_tol = 0.01, cholesky = true, n_dims = nil },
  ridge = {
    lambda = { def = 2.3163e-05 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    classes = 10,
    search_trials = 200,
    k = 1,
  },
}

test("mnist hdc", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.features, cfg.data.max)
  local train, test_set, validate = ds.split_binary_mnist(dataset, cfg.data.ttr, cfg.data.tvr)
  local n_features = cfg.data.features
  local n_classes = cfg.ridge.classes
  local label_off, label_nbr = train.solutions:bits_to_csr(train.n, n_classes)
  local val_label_off, val_label_nbr = validate.solutions:bits_to_csr(validate.n, n_classes)
  str.printf("[Data] train=%d val=%d test=%d features=%d classes=%d %s\n",
    train.n, validate.n, test_set.n, n_features, n_classes, sw())

  str.printf("[HDC] Encoding train\n")
  local train_bits = require("santoku.ivec").create()
  dataset.problems:bits_select(nil, train.ids, n_features, train_bits)
  local train_bmp = train_bits:bits_to_cvec(train.n, n_features)
  train_bits = nil -- luacheck: ignore
  local hdc_enc, train_cvec = hdc.create({
    bits = train_bmp, n_dims = n_features,
    n_samples = train.n, d = cfg.hdc.d,
    hdc_ngram = cfg.hdc.ngram, row_length = cfg.hdc.row_length,
  })
  str.printf("[HDC] d=%d %s\n", cfg.hdc.d, sw())

  str.printf("[Spectral] Cholesky trace_tol=%s\n", tostring(cfg.emb.trace_tol))
  local _, _, sp_enc, _, xtx, xty, col_mean, y_mean, label_counts, pre_mean, pre_istd =
    spectral.encode({
      bits = train_cvec, n_samples = train.n, d_bits = cfg.hdc.d,
      n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
      cholesky = cfg.emb.cholesky, n_dims = cfg.emb.n_dims,
      label_offsets = label_off, label_neighbors = label_nbr, n_labels = n_classes,
    })
  local emb_d = sp_enc:dims()

  str.printf("[HDC] Encoding val\n")
  local val_bits = require("santoku.ivec").create()
  dataset.problems:bits_select(nil, validate.ids, n_features, val_bits)
  local val_bmp = val_bits:bits_to_cvec(validate.n, n_features)
  val_bits = nil -- luacheck: ignore
  local val_codes = sp_enc:encode(hdc_enc:encode({ bits = val_bmp, n_samples = validate.n }), validate.n)
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
  local val_off, val_labels = ridge_obj:label(val_codes, validate.n, 1)
  local test_bits = require("santoku.ivec").create()
  dataset.problems:bits_select(nil, test_set.ids, n_features, test_bits)
  local test_bmp = test_bits:bits_to_cvec(test_set.n, n_features)
  test_bits = nil -- luacheck: ignore
  local test_codes = sp_enc:encode(hdc_enc:encode({ bits = test_bmp, n_samples = test_set.n }), test_set.n)
  test_codes:mtx_standardize(emb_d, std.pre_mean, std.pre_istd)
  local _, test_labels = ridge_obj:label(test_codes, test_set.n, 1)
  str.printf("[Eval] Labels done %s\n", sw())

  local val_stats = eval.class_accuracy(val_labels, validate.solutions, validate.n, n_classes)
  local test_stats = eval.class_accuracy(test_labels, test_set.solutions, test_set.n, n_classes)
  str.printf("[Class] F1: val=%.2f test=%.2f %s\n",
    val_stats.f1, test_stats.f1, sw())

  local _, val_oracle = eval.retrieval_ks({
    pred_offsets = val_off, pred_neighbors = val_labels,
    expected_offsets = val_label_off, expected_neighbors = val_label_nbr,
  })
  str.printf("[Retrieval] val: saF1=%.4f miF1=%.4f %s\n",
    val_oracle.sample_f1, val_oracle.micro_f1, sw())

  str.printf("\n[Per-class Test Accuracy]\n")
  local class_order = arr.range(1, n_classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    str.printf("  digit_%-2d  F1=%.2f  P=%.2f  R=%.2f\n", c - 1, ts.f1, ts.precision, ts.recall)
  end

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
