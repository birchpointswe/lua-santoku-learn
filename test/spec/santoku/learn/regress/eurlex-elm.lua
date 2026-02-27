local ann = require("santoku.learn.ann")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.learn.evaluator")
local inv = require("santoku.learn.inv")
local ivec = require("santoku.ivec")
local optimize = require("santoku.learn.optimize")
local quantizer = require("santoku.learn.quantizer")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { max = nil },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 3,
    cgrams_max = 5,
    cgrams_cross = true,
    skips = 1,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 1024,
    decay = 0,
  },
  augmented = {
    knn = 32,
  },
  label_eval = {
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 20,
    k = 32,
  },
  elm_direct = {
    n_hidden = 8192*4,
    seed = 42,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 20,
    k = 32,
    n_models = 1,
  },
  elm_spectral = {
    n_hidden = 8192*4,
    seed = 42,
    lambda = { def = 1.0 },
    search_trials = 20,
    n_models = 1,
  },
  elm_fused = {
    n_hidden = 8192*4,
    seed = 42,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 20,
    k = 32,
    n_models = 1,
  },
}

test("eurlex-aligned", function ()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  str.printf("  Train: %d  Dev: %d  Test: %d  Labels: %d\n", train.n, dev.n, test_set.n, n_labels)

  print("\nBuilding label CSR")
  local train_label_off, train_label_nbr = train.solutions:bits_to_csr(train.n, n_labels)
  local dev_label_off, dev_label_nbr = dev.solutions:bits_to_csr(dev.n, n_labels)
  local test_label_off, test_label_nbr = test_set.solutions:bits_to_csr(test_set.n, n_labels)
  local train_ids = ivec.create(train.n):fill_indices()

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens_raw = tok:features()
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil
  dev.problems = nil
  test_set.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens_raw)

  print("\nFeature selection (BNS/max) for text->labels")
  local bns_ids, bns_scores = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens_raw, n_labels, nil, nil, "max")
  local n_bns_tokens = bns_ids:size()
  str.printf("  BNS features: %d\n", n_bns_tokens)

  local function make_bns_csc(tokens, n)
    local bits = ivec.create():copy(tokens)
    bits:bits_select(bns_ids, nil, n_tokens_raw)
    return csr.to_csc(bits, n, n_bns_tokens)
  end
  local train_bns_off, train_bns_idx = make_bns_csc(train.tokens, train.n)
  local dev_bns_off, dev_bns_idx = make_bns_csc(dev.tokens, dev.n)
  local test_bns_off, test_bns_idx = make_bns_csc(test_set.tokens, test_set.n)

  print("\n#1: Direct ELM text->labels (baseline)")
  local elm_direct_obj, elm_direct_params, _, d1_train_scores = optimize.elm({
    n_samples = train.n,
    n_tokens = n_bns_tokens,
    n_hidden = cfg.elm_direct.n_hidden,

    seed = cfg.elm_direct.seed,
    csc_offsets = train_bns_off,
    csc_indices = train_bns_idx,
    feature_weights = bns_scores,
    n_labels = n_labels,
    label_offsets = train_label_off,
    label_neighbors = train_label_nbr,
    expected_offsets = train_label_off,
    expected_neighbors = train_label_nbr,
    val_csc_offsets = dev_bns_off,
    val_csc_indices = dev_bns_idx,
    val_n_samples = dev.n,
    val_expected_offsets = dev_label_off,
    val_expected_neighbors = dev_label_nbr,
    lambda = cfg.elm_direct.lambda,
    propensity_a = cfg.elm_direct.propensity_a,
    propensity_b = cfg.elm_direct.propensity_b,
    k = cfg.elm_direct.k,
    search_trials = cfg.elm_direct.search_trials,
    n_models = cfg.elm_direct.n_models,
    each = util.make_elm_log(stopwatch),
  })
  str.printf("  lambda=%.4e\n", elm_direct_params.lambda)

  local d1_dev_scores = elm_direct_obj:transform(dev_bns_off, dev_bns_idx, dev.n)
  local d1_test_scores = elm_direct_obj:transform(test_bns_off, test_bns_idx, test_set.n)
  local d1_thresholds, d1_tr_lt = eval.label_thresholds({
    scores = d1_train_scores, n_samples = train.n, n_labels = n_labels,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
  })
  local d1_dv_lt = eval.apply_label_thresholds({
    scores = d1_dev_scores, n_samples = dev.n, n_labels = n_labels,
    thresholds = d1_thresholds,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  local d1_ts_lt = eval.apply_label_thresholds({
    scores = d1_test_scores, n_samples = test_set.n, n_labels = n_labels,
    thresholds = d1_thresholds,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
  })
  str.printf("  PerLabel:  Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d1_tr_lt.macro_f1, d1_dv_lt.macro_f1, d1_ts_lt.macro_f1)

  elm_direct_obj = nil -- luacheck: ignore
  d1_train_scores = nil; d1_dev_scores = nil; d1_test_scores = nil -- luacheck: ignore
  collectgarbage("collect")

  print("\nLabel spectral embedding")
  local label_sols = ivec.create():copy(train.solutions)
  local label_idf_ids, label_idf_scores = label_sols:bits_top_idf(train.n, n_labels)
  label_sols:bits_select(label_idf_ids, nil, n_labels)
  local label_index = inv.create({ features = label_idf_scores })
  label_index:add(label_sols, train_ids)
  str.printf("  Label index: %d IDF feats, %d docs\n", label_idf_ids:size(), train.n)
  local label_model = optimize.spectral({
    index = label_index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    each = util.make_spectral_log(stopwatch),
  })
  local label_dims = label_model.dims
  str.printf("  Label spectral: %d dims\n", label_dims)
  local label_codes = dvec.create():mtx_extend(label_model.raw_codes,
    label_model.ids:set_intersect(ivec.create(train.n):fill_indices()),
    label_model.ids, 0, label_dims, true)
  label_index = nil; label_model = nil; label_sols = nil -- luacheck: ignore

  print("\n#2: Ridge label spectral -> labels (ceiling, train only)")
  local _, _, ceiling_metrics = optimize.ridge({
    n_samples = train.n, n_dims = label_dims,
    n_labels = n_labels, codes = label_codes,
    label_offsets = train_label_off, label_neighbors = train_label_nbr,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
    lambda = cfg.label_eval.lambda,
    propensity_a = cfg.label_eval.propensity_a,
    propensity_b = cfg.label_eval.propensity_b,
    k = cfg.label_eval.k,
    search_trials = cfg.label_eval.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  local ceiling = ceiling_metrics.oracle
  str.printf("  Ceiling: maF1=%.4f miF1=%.4f\n", ceiling.macro_f1, ceiling.micro_f1)

  print("\nFeature selection for text->spectral")
  local r2_ids, r2_scores = train.tokens:bits_top_reg_auc(
    label_codes, train.n, n_tokens_raw, label_dims, nil, nil, "sum")
  local n_r2_tokens = r2_ids:size()
  str.printf("  Selected features: %d\n", n_r2_tokens)

  local function make_r2_csc(tokens, n)
    local bits = ivec.create():copy(tokens)
    bits:bits_select(r2_ids, nil, n_tokens_raw)
    return csr.to_csc(bits, n, n_r2_tokens)
  end
  local train_r2_off, train_r2_idx = make_r2_csc(train.tokens, train.n)
  local dev_r2_off, dev_r2_idx = make_r2_csc(dev.tokens, dev.n)
  local test_r2_off, test_r2_idx = make_r2_csc(test_set.tokens, test_set.n)

  print("\n#3: ELM text->spectral + neighbor augmentation")

  local elm_spec_obj, elm_spec_params, _, train_pred = optimize.elm({
    n_samples = train.n,
    n_tokens = n_r2_tokens,
    n_hidden = cfg.elm_spectral.n_hidden,

    seed = cfg.elm_spectral.seed,
    csc_offsets = train_r2_off,
    csc_indices = train_r2_idx,
    feature_weights = r2_scores,
    targets = label_codes,
    n_targets = label_dims,
    lambda = cfg.elm_spectral.lambda,
    search_trials = cfg.elm_spectral.search_trials,
    n_models = cfg.elm_spectral.n_models,
    each = util.make_elm_log(stopwatch),
  })
  str.printf("  H=%d lambda=%.4e\n", elm_spec_params.n_hidden, elm_spec_params.lambda)

  local dev_pred = elm_spec_obj:transform(dev_r2_off, dev_r2_idx, dev.n)
  local test_pred = elm_spec_obj:transform(test_r2_off, test_r2_idx, test_set.n)

  local pred_mae = eval.regression_accuracy(train_pred, label_codes)
  str.printf("  ELM->spectral MAE: %.6f\n", pred_mae.mean)

  str.printf("  Building ANN on train predicted codes\n")
  local pred_itq = quantizer.create({ mode = "itq", raw_codes = train_pred, n_samples = train.n })
  local train_bin = pred_itq:encode(train_pred)
  local pred_ann = ann.create({ features = label_dims })
  pred_ann:add(train_bin, train_ids)

  local function build_neighbor_avg(pred)
    local bin = pred_itq:encode(pred)
    local hood_ids, ann_hoods = pred_ann:neighborhoods_by_vecs(bin, cfg.augmented.knn)
    local nn_off, nn_nbr, nn_w = ann_hoods:to_csr(label_dims)
    return csr.neighbor_average(nn_off, nn_nbr, nn_w, hood_ids, label_codes, train_ids, label_dims)
  end

  local train_avg = build_neighbor_avg(train_pred)
  local dev_avg = build_neighbor_avg(dev_pred)
  local test_avg = build_neighbor_avg(test_pred)

  local train_combined = dvec.create():copy(train_pred)
  train_combined:mtx_extend(train_avg, label_dims, label_dims)
  local dev_combined = dvec.create():copy(dev_pred)
  dev_combined:mtx_extend(dev_avg, label_dims, label_dims)
  local test_combined = dvec.create():copy(test_pred)
  test_combined:mtx_extend(test_avg, label_dims, label_dims)
  local combined_dims = label_dims * 2
  str.printf("  Combined: %d dims (pred=%d + avg=%d, k=%d)\n",
    combined_dims, label_dims, label_dims, cfg.augmented.knn)

  train_pred = nil; dev_pred = nil; test_pred = nil -- luacheck: ignore
  elm_spec_obj = nil; pred_itq = nil; pred_ann = nil -- luacheck: ignore
  collectgarbage("collect")

  print("  Training ridge combined->labels")
  local aug_ridge, _, _ = optimize.ridge({
    n_samples = train.n, n_dims = combined_dims,
    n_labels = n_labels, codes = train_combined,
    label_offsets = train_label_off, label_neighbors = train_label_nbr,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
    lambda = cfg.label_eval.lambda,
    propensity_a = cfg.label_eval.propensity_a,
    propensity_b = cfg.label_eval.propensity_b,
    k = cfg.label_eval.k,
    search_trials = cfg.label_eval.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  local a3_train_scores = aug_ridge:transform(train_combined, train.n)
  local a3_dev_scores = aug_ridge:transform(dev_combined, dev.n)
  local a3_test_scores = aug_ridge:transform(test_combined, test_set.n)
  local a3_thresholds, a3_tr_lt = eval.label_thresholds({
    scores = a3_train_scores, n_samples = train.n, n_labels = n_labels,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
  })
  local a3_dv_lt = eval.apply_label_thresholds({
    scores = a3_dev_scores, n_samples = dev.n, n_labels = n_labels,
    thresholds = a3_thresholds,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  local a3_ts_lt = eval.apply_label_thresholds({
    scores = a3_test_scores, n_samples = test_set.n, n_labels = n_labels,
    thresholds = a3_thresholds,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
  })
  str.printf("  PerLabel: Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    a3_tr_lt.macro_f1, a3_dv_lt.macro_f1, a3_ts_lt.macro_f1)

  train_combined = nil; dev_combined = nil; test_combined = nil -- luacheck: ignore
  aug_ridge = nil -- luacheck: ignore
  a3_train_scores = nil; a3_dev_scores = nil; a3_test_scores = nil -- luacheck: ignore
  collectgarbage("collect")

  print("\n#4: Fused ELM text+avg->labels")
  local fused_obj, fused_params, _, a4_train_scores = optimize.elm({
    n_samples = train.n,
    n_tokens = n_bns_tokens,
    n_hidden = cfg.elm_fused.n_hidden,

    seed = cfg.elm_fused.seed,
    csc_offsets = train_bns_off,
    csc_indices = train_bns_idx,
    feature_weights = bns_scores,
    dense_features = train_avg,
    n_dense = label_dims,
    n_labels = n_labels,
    label_offsets = train_label_off,
    label_neighbors = train_label_nbr,
    expected_offsets = train_label_off,
    expected_neighbors = train_label_nbr,
    val_csc_offsets = dev_bns_off,
    val_csc_indices = dev_bns_idx,
    val_n_samples = dev.n,
    val_dense_features = dev_avg,
    val_expected_offsets = dev_label_off,
    val_expected_neighbors = dev_label_nbr,
    lambda = cfg.elm_fused.lambda,
    propensity_a = cfg.elm_fused.propensity_a,
    propensity_b = cfg.elm_fused.propensity_b,
    k = cfg.elm_fused.k,
    search_trials = cfg.elm_fused.search_trials,
    n_models = cfg.elm_fused.n_models,
    each = util.make_elm_log(stopwatch),
  })
  str.printf("  lambda=%.4e\n", fused_params.lambda)

  local a4_dev_scores = fused_obj:transform(dev_bns_off, dev_bns_idx, dev.n, dev_avg)
  local a4_test_scores = fused_obj:transform(test_bns_off, test_bns_idx, test_set.n, test_avg)
  local a4_thresholds, a4_tr_lt = eval.label_thresholds({
    scores = a4_train_scores, n_samples = train.n, n_labels = n_labels,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
  })
  local a4_dv_lt = eval.apply_label_thresholds({
    scores = a4_dev_scores, n_samples = dev.n, n_labels = n_labels,
    thresholds = a4_thresholds,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  local a4_ts_lt = eval.apply_label_thresholds({
    scores = a4_test_scores, n_samples = test_set.n, n_labels = n_labels,
    thresholds = a4_thresholds,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
  })
  str.printf("  PerLabel: Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    a4_tr_lt.macro_f1, a4_dv_lt.macro_f1, a4_ts_lt.macro_f1)

  fused_obj = nil -- luacheck: ignore
  a4_train_scores = nil; a4_dev_scores = nil; a4_test_scores = nil -- luacheck: ignore
  collectgarbage("collect")

  print("\n" .. string.rep("=", 60))
  print("RESULTS")
  print(string.rep("=", 60))
  str.printf("\n  #1 Direct ELM (baseline):\n")
  str.printf("    Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d1_tr_lt.macro_f1, d1_dv_lt.macro_f1, d1_ts_lt.macro_f1)
  str.printf("  #2 Ceiling (train):  maF1=%.4f miF1=%.4f\n", ceiling.macro_f1, ceiling.micro_f1)
  str.printf("  #3 Augmented:\n")
  str.printf("    Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    a3_tr_lt.macro_f1, a3_dv_lt.macro_f1, a3_ts_lt.macro_f1)
  str.printf("  #4 Fused ELM:\n")
  str.printf("    Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    a4_tr_lt.macro_f1, a4_dv_lt.macro_f1, a4_ts_lt.macro_f1)
  str.printf("  ELM->spectral MAE:   %.6f\n", pred_mae.mean)
  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
