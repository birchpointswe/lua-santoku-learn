local arr = require("santoku.array")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local ivec = require("santoku.ivec")
local optimize = require("santoku.learn.optimize")
local ridge = require("santoku.learn.ridge")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

local cfg = {
  data = {
    ttr = 0.5,
    tvr = 0.1,
    max = nil,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 1,
    cgrams_min = 3,
    cgrams_max = 5,
    cgrams_cross = true,
    skips = 0,
  },
  elm = {
    classes = 2,
    n_hidden = 8192,
    seed = 42,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 400,
    k = 1,
    n_models = 1,
  },
}

local class_names = { "negative", "positive" }

test("imdb elm classifier", function ()

  print("Reading data")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test_set, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)

  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test_set.n)

  print("\nTraining tokenizer\n")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens_raw = tok:features()
  str.printf("  Vocabulary: %d\n", n_tokens_raw)

  print("\nTokenizing")
  train.tokens = tok:tokenize(train.problems)
  validate.tokens = tok:tokenize(validate.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore

  print("\nFeature selection (BNS)")
  local bns_ids, bns_scores = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens_raw, cfg.elm.classes, nil, nil, "max")
  train.tokens:bits_select(bns_ids, nil, n_tokens_raw)
  validate.tokens:bits_select(bns_ids, nil, n_tokens_raw)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens_raw)
  local n_tokens = bns_ids:size()
  str.printf("  Selected %d features\n", n_tokens)

  local train_csc_off, train_csc_idx = csr.to_csc(train.tokens, train.n, n_tokens)
  local val_csc_off, val_csc_idx = csr.to_csc(validate.tokens, validate.n, n_tokens)
  local test_csc_off, test_csc_idx = csr.to_csc(test_set.tokens, test_set.n, n_tokens)

  print("\nBuilding label CSR")
  local label_off, label_nbr = train.solutions:bits_to_csr(train.n, cfg.elm.classes)
  local val_label_off, val_label_nbr = validate.solutions:bits_to_csr(validate.n, cfg.elm.classes)

  print("\nTraining ELM")
  local stopwatch = utc.stopwatch()
  local elm_obj, elm_params, _, train_scores = optimize.elm({
    n_samples = train.n,
    n_tokens = n_tokens,
    n_hidden = cfg.elm.n_hidden,
    seed = cfg.elm.seed,
    csc_offsets = train_csc_off,
    csc_indices = train_csc_idx,
    feature_weights = bns_scores,
    n_labels = cfg.elm.classes,
    label_offsets = label_off,
    label_neighbors = label_nbr,
    expected_offsets = label_off,
    expected_neighbors = label_nbr,
    val_csc_offsets = val_csc_off,
    val_csc_indices = val_csc_idx,
    val_n_samples = validate.n,
    val_expected_offsets = val_label_off,
    val_expected_neighbors = val_label_nbr,
    lambda = cfg.elm.lambda,
    propensity_a = cfg.elm.propensity_a,
    propensity_b = cfg.elm.propensity_b,
    k = cfg.elm.k,
    search_trials = cfg.elm.search_trials,
    n_models = cfg.elm.n_models,
    each = util.make_elm_log(stopwatch),
  })
  str.printf("\nBest: n_hidden=%d lambda=%.4e\n", elm_params.n_hidden, elm_params.lambda)
  str.printf("Time: %.1fs\n", stopwatch())

  local n_classes = cfg.elm.classes

  print("\nEvaluating splits")
  local train_off, train_labels = ridge.label_from_scores(train_scores, train.n, n_classes, 1)
  local val_off, val_labels = elm_obj:label(val_csc_off, val_csc_idx, validate.n, 1)
  local _, test_labels = elm_obj:label(test_csc_off, test_csc_idx, test_set.n, 1)

  print("\nClassification metrics (class_accuracy):")
  local train_stats = eval.class_accuracy(train_labels, train.solutions, train.n, n_classes)
  local val_stats = eval.class_accuracy(val_labels, validate.solutions, validate.n, n_classes)
  local test_stats = eval.class_accuracy(test_labels, test_set.solutions, test_set.n, n_classes)
  str.printf("  F1:   Train=%.2f  Val=%.2f  Test=%.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

  print("\nRetrieval metrics:")
  local _, train_oracle = eval.retrieval_ks({
    pred_offsets = train_off, pred_neighbors = train_labels,
    expected_offsets = label_off, expected_neighbors = label_nbr,
  })
  local _, val_oracle = eval.retrieval_ks({
    pred_offsets = val_off, pred_neighbors = val_labels,
    expected_offsets = val_label_off, expected_neighbors = val_label_nbr,
  })
  str.printf("  Train: maF1=%.4f miF1=%.4f\n", train_oracle.macro_f1, train_oracle.micro_f1)
  str.printf("  Val:   maF1=%.4f miF1=%.4f\n", val_oracle.macro_f1, val_oracle.micro_f1)

  print("\nPer-class Test Accuracy (sorted by difficulty):\n")
  local class_order = arr.range(1, n_classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    str.printf("  %-12s  F1=%.2f  P=%.2f  R=%.2f\n", class_names[c], ts.f1, ts.precision, ts.recall)
  end

end)
