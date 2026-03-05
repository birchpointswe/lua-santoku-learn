local arr = require("santoku.array")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local hdc = require("santoku.learn.hdc")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

local cfg = {
  data = { max = nil, ttr = 0.5, tvr = 0.1, },
  hdc = { d = 8192, ngram = 5 },
  elm = {
    mode = nil,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.5 },
    propensity_b = { def = 1.5 },
    classes = 2,
    search_trials = 200,
    k = 1,
  },
}

local class_names = { "negative", "positive" }

local function eval_classifier (label, ridge_obj, hdc_enc, train_h, train, validate, test_set,
                                label_off, label_nbr, val_label_off, val_label_nbr, n_classes)
  print("\n" .. label .. " — Evaluating splits")
  local train_off, train_labels = ridge_obj:label(train_h, train.n, 1)
  local val_codes = hdc_enc:encode({ texts = validate.problems, n_samples = validate.n })
  local val_off, val_labels = ridge_obj:label(val_codes, validate.n, 1)
  local test_codes = hdc_enc:encode({ texts = test_set.problems, n_samples = test_set.n })
  local _, test_labels = ridge_obj:label(test_codes, test_set.n, 1)

  print("\nClassification metrics (class_accuracy):")
  local train_stats = eval.class_accuracy(train_labels, train.solutions, train.n, n_classes)
  local val_stats = eval.class_accuracy(val_labels, validate.solutions, validate.n, n_classes)
  local test_stats = eval.class_accuracy(test_labels, test_set.solutions, test_set.n, n_classes)
  str.printf("  F1:   Train=%.2f  Val=%.2f  Test=%.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

  print("\nRetrieval metrics:")
  local _, train_oracle = eval.retrieval_ks({ -- luacheck: ignore
    pred_offsets = train_off, pred_neighbors = train_labels,
    expected_offsets = label_off, expected_neighbors = label_nbr,
  })
  local _, val_oracle = eval.retrieval_ks({
    pred_offsets = val_off, pred_neighbors = val_labels,
    expected_offsets = val_label_off, expected_neighbors = val_label_nbr,
  })
  str.printf("  Train: saF1=%.4f miF1=%.4f\n", train_oracle.sample_f1, train_oracle.micro_f1)
  str.printf("  Val:   saF1=%.4f miF1=%.4f\n", val_oracle.sample_f1, val_oracle.micro_f1)

  print("\nPer-class Test Accuracy (sorted by difficulty):\n")
  local class_order = arr.range(1, n_classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    local cat = class_names[c] or ("class_" .. (c - 1))
    str.printf("  %-12s  F1=%.2f  P=%.2f  R=%.2f\n", cat, ts.f1, ts.precision, ts.recall)
  end
  return { train = train_stats.f1, val = val_stats.f1, test = test_stats.f1 }
end

test("imdb hdc", function ()

  print("Reading data")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test_set, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)

  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test_set.n)

  local n_classes = cfg.elm.classes
  local label_off, label_nbr = train.solutions:bits_to_csr(train.n, n_classes)
  local val_label_off, val_label_nbr = validate.solutions:bits_to_csr(validate.n, n_classes)

  str.printf("  d=%d  ngram=%d\n", cfg.hdc.d, cfg.hdc.ngram)

  local ngram_map, train_tok, n_hdc_tokens = hdc.tokenize({
    texts = train.problems, hdc_ngram = cfg.hdc.ngram, n_samples = train.n,
  })
  str.printf("  HDC tokens (char): %d\n", n_hdc_tokens)
  local bns_ids, bns_scores = train_tok:bits_top_bns(
    train.solutions, train.n, n_hdc_tokens, n_classes)
  str.printf("  BNS selected: %d\n", bns_ids:size())

  local stopwatch = utc.stopwatch()
  local hdc_enc, train_h = hdc.create({
    texts = train.problems,
    n_samples = train.n,
    d = cfg.hdc.d,
    hdc_ngram = cfg.hdc.ngram,
    weight_map = ngram_map,
    weight_ids = bns_ids,
    weights = bns_scores,
  })
  local val_h = hdc_enc:encode({ texts = validate.problems, n_samples = validate.n })
  local hdc_out_d = hdc_enc:out_d()
  local _, ridge_obj, elm_params = optimize.ridge({
    n_samples = train.n,
    n_dims = hdc_out_d,
    codes = train_h,
    n_labels = n_classes,
    label_offsets = label_off,
    label_neighbors = label_nbr,
    expected_offsets = label_off,
    expected_neighbors = label_nbr,
    val_codes = val_h,
    val_n_samples = validate.n,
    val_expected_offsets = val_label_off,
    val_expected_neighbors = val_label_nbr,
    lambda = cfg.elm.lambda,
    propensity_a = cfg.elm.propensity_a,
    propensity_b = cfg.elm.propensity_b,
    k = cfg.elm.k,
    search_trials = cfg.elm.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  str.printf("\nBest: lambda=%.4e\n", elm_params.lambda)
  str.printf("Time: %.1fs\n", stopwatch())

  local r = eval_classifier("HDC+Ridge BNS (char)", ridge_obj, hdc_enc, train_h, train, validate, test_set,
    label_off, label_nbr, val_label_off, val_label_nbr, n_classes)
  str.printf("\n  %-28s  Train=%.2f  Val=%.2f  Test=%.2f  Time=%.1fs\n",
    "HDC+Ridge BNS (char)", r.train, r.val, r.test, stopwatch())

end)
