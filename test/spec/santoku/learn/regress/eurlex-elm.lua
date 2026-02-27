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
  data = { max = nil, toks_per_class = 4096, toks_overall = nil },
  tokenizer = {
    max_len = 20, min_len = 1, max_run = 2,
    ngrams = 2, cgrams_min = 3, cgrams_max = 5,
    cgrams_cross = true, skips = 1,
  },
  nystrom = {
    n_landmarks = 4096, n_dims = nil, decay = 0,
  },
  augmented = { knn = 32 },
  ridge_label = {
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 200, k = 32,
  },
  elm_direct = {
    mode = "relu",
    n_hidden = 8192, seed = 42,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 200, k = 32,
  },
  elm_spectral = {
    mode = "relu",
    n_hidden = 8192, seed = 42,
    lambda = { def = 1.0 },
    search_trials = 200,
  },
  elm_sidecar = {
    mode = "relu",
    n_hidden = 8192, seed = 42,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 200, k = 32,
  },
  elm_kpred = {
    mode = "relu",
    n_hidden = 8192, seed = 42,
    lambda = { def = 1.0 },
    search_trials = 200,
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

  local function eval_oracle(pred_off, pred_nbr, exp_off, exp_nbr, ks)
    local _, oracle = eval.retrieval_ks({
      pred_offsets = pred_off, pred_neighbors = pred_nbr,
      expected_offsets = exp_off, expected_neighbors = exp_nbr,
      ks = ks,
    })
    return oracle
  end

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
  local bns_ids, bns_scores = train.tokens:bits_top_chi2(
    train.solutions, train.n, n_tokens_raw, n_labels, cfg.data.toks_per_class, cfg.data.toks_overall, "max")
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

  local function eval_elm(obj, train_h, k, dense_dv, dense_ts)
    local tr_off, tr_nbr = obj.ridge:label(train_h, train.n, k)
    local tr_oracle = eval_oracle(tr_off, tr_nbr, train_label_off, train_label_nbr)
    local dv_off, dv_nbr = obj:label(dev_bns_off, dev_bns_idx, dev.n, k, dense_dv)
    local dv_oracle = eval_oracle(dv_off, dv_nbr, dev_label_off, dev_label_nbr)
    local ts_off, ts_nbr = obj:label(test_bns_off, test_bns_idx, test_set.n, k, dense_ts)
    local ts_oracle = eval_oracle(ts_off, ts_nbr, test_label_off, test_label_nbr)
    str.printf("  Oracle:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
      tr_oracle.macro_f1, dv_oracle.macro_f1, ts_oracle.macro_f1)
    local tr_scores = obj.ridge:transform(train_h, train.n)
    local dv_scores = obj:transform(dev_bns_off, dev_bns_idx, dev.n, dense_dv)
    local ts_scores = obj:transform(test_bns_off, test_bns_idx, test_set.n, dense_ts)
    local thresholds = eval.label_thresholds({
      scores = dv_scores, n_samples = dev.n, n_labels = n_labels,
      expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
    })
    local tr_thr = eval.apply_label_thresholds({
      scores = tr_scores, n_samples = train.n, n_labels = n_labels,
      thresholds = thresholds,
      expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
    })
    local dv_thr = eval.apply_label_thresholds({
      scores = dv_scores, n_samples = dev.n, n_labels = n_labels,
      thresholds = thresholds,
      expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
    })
    local ts_thr = eval.apply_label_thresholds({
      scores = ts_scores, n_samples = test_set.n, n_labels = n_labels,
      thresholds = thresholds,
      expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
    })
    str.printf("  Thresh:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
      tr_thr.macro_f1, dv_thr.macro_f1, ts_thr.macro_f1)
    local _, kp_params, ts_pred_ks = util.train_kpred({
      k = k, n_labels = n_labels,
      n_tokens = n_bns_tokens, feature_weights = bns_scores,
      train_csc_offsets = dev_bns_off, train_csc_indices = dev_bns_idx,
      test_csc_offsets = test_bns_off, test_csc_indices = test_bns_idx,
      train_scores = dv_scores, train_pred_off = dv_off, train_pred_nbr = dv_nbr,
      train_exp_off = dev_label_off, train_exp_nbr = dev_label_nbr, train_n = dev.n,
      test_scores = ts_scores, test_pred_off = ts_off, test_pred_nbr = ts_nbr, test_n = test_set.n,
      n_hidden = cfg.elm_kpred.n_hidden, seed = cfg.elm_kpred.seed,
      mode = cfg.elm_kpred.mode,
      lambda = cfg.elm_kpred.lambda, search_trials = cfg.elm_kpred.search_trials,
      each = util.make_kpred_log(stopwatch),
    })
    local ts_kpred = eval_oracle(ts_off, ts_nbr, test_label_off, test_label_nbr, ts_pred_ks)
    str.printf("  K-pred:   Test maF1=%.4f (lambda=%.4e)\n", ts_kpred.macro_f1, kp_params.lambda)
    return {
      tr_oracle = tr_oracle, dv_oracle = dv_oracle, ts_oracle = ts_oracle,
      tr_thr = tr_thr, dv_thr = dv_thr, ts_thr = ts_thr,
      ts_kpred = ts_kpred,
    }
  end

  print("\n#1: Direct ELM text->labels (baseline)")
  local elm_direct_obj, elm_direct_params, _, d1_train_h = optimize.elm({
    n_samples = train.n,
    n_tokens = n_bns_tokens,
    n_hidden = cfg.elm_direct.n_hidden,
    seed = cfg.elm_direct.seed,
    mode = cfg.elm_direct.mode,
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
    each = util.make_elm_log(stopwatch),
  })
  str.printf("  lambda=%.4e\n", elm_direct_params.lambda)
  local d1 = eval_elm(elm_direct_obj, d1_train_h, cfg.elm_direct.k)
  elm_direct_obj = nil; d1_train_h = nil -- luacheck: ignore
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
  local ceiling_ridge, _, ceiling_metrics = optimize.ridge({
    n_samples = train.n, n_dims = label_dims,
    n_labels = n_labels, codes = label_codes,
    label_offsets = train_label_off, label_neighbors = train_label_nbr,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
    lambda = cfg.ridge_label.lambda,
    propensity_a = cfg.ridge_label.propensity_a,
    propensity_b = cfg.ridge_label.propensity_b,
    k = cfg.ridge_label.k,
    search_trials = cfg.ridge_label.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  local ceiling_oracle = ceiling_metrics.oracle
  ceiling_ridge = nil -- luacheck: ignore
  str.printf("  Oracle:   maF1=%.4f miF1=%.4f\n", ceiling_oracle.macro_f1, ceiling_oracle.micro_f1)

  print("\nFeature selection for text->spectral")
  local r2_ids, r2_scores = train.tokens:bits_top_reg_auc(
    label_codes, train.n, n_tokens_raw, label_dims, cfg.data.toks_per_class, cfg.data.toks_overall, "max")
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

  print("\nDistillation: ELM text->spectral")
  local elm_spec_obj, elm_spec_params, _, spec_train_h = optimize.elm({
    n_samples = train.n,
    n_tokens = n_r2_tokens,
    n_hidden = cfg.elm_spectral.n_hidden,
    seed = cfg.elm_spectral.seed,
    mode = cfg.elm_spectral.mode,
    csc_offsets = train_r2_off,
    csc_indices = train_r2_idx,
    feature_weights = r2_scores,
    targets = label_codes,
    n_targets = label_dims,
    lambda = cfg.elm_spectral.lambda,
    search_trials = cfg.elm_spectral.search_trials,
    each = util.make_elm_log(stopwatch),
  })
  str.printf("  H=%d lambda=%.4e\n", elm_spec_params.n_hidden, elm_spec_params.lambda)

  local train_pred = elm_spec_obj.ridge:transform(spec_train_h, train.n)
  spec_train_h = nil -- luacheck: ignore
  local dev_pred = elm_spec_obj:transform(dev_r2_off, dev_r2_idx, dev.n)
  local test_pred = elm_spec_obj:transform(test_r2_off, test_r2_idx, test_set.n)
  elm_spec_obj = nil -- luacheck: ignore

  local pred_mae = eval.regression_accuracy(train_pred, label_codes)
  str.printf("  MAE: %.6f\n", pred_mae.mean)

  print("\nBuilding neighbor averages")
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
  pred_itq = nil; pred_ann = nil; train_bin = nil -- luacheck: ignore
  str.printf("  knn=%d, dims=%d\n", cfg.augmented.knn, label_dims)

  local train_both = dvec.create():copy(train_pred)
  train_both:mtx_extend(train_avg, label_dims, label_dims)
  local dev_both = dvec.create():copy(dev_pred)
  dev_both:mtx_extend(dev_avg, label_dims, label_dims)
  local test_both = dvec.create():copy(test_pred)
  test_both:mtx_extend(test_avg, label_dims, label_dims)
  local both_dims = label_dims * 2
  collectgarbage("collect")

  local function train_sidecar_elm(dense_tr, dense_dv, dense_ts, n_dense)
    local sc = cfg.elm_sidecar
    local obj, params, _, train_h = optimize.elm({
      n_samples = train.n,
      n_tokens = n_bns_tokens,
      n_hidden = sc.n_hidden,
      seed = sc.seed,
      mode = sc.mode,
      csc_offsets = train_bns_off,
      csc_indices = train_bns_idx,
      feature_weights = bns_scores,
      dense_features = dense_tr,
      n_dense = n_dense,
      n_labels = n_labels,
      label_offsets = train_label_off,
      label_neighbors = train_label_nbr,
      expected_offsets = train_label_off,
      expected_neighbors = train_label_nbr,
      val_csc_offsets = dev_bns_off,
      val_csc_indices = dev_bns_idx,
      val_n_samples = dev.n,
      val_dense_features = dense_dv,
      val_expected_offsets = dev_label_off,
      val_expected_neighbors = dev_label_nbr,
      lambda = sc.lambda,
      propensity_a = sc.propensity_a,
      propensity_b = sc.propensity_b,
      k = sc.k,
      search_trials = sc.search_trials,
      each = util.make_elm_log(stopwatch),
    })
    str.printf("  lambda=%.4e\n", params.lambda)
    local result = eval_elm(obj, train_h, sc.k, dense_dv, dense_ts)
    obj = nil; train_h = nil -- luacheck: ignore
    collectgarbage("collect")
    return result
  end

  print("\n#3: Distilled (sparse + predicted spectral sidecar)")
  local d3 = train_sidecar_elm(train_pred, dev_pred, test_pred, label_dims)

  print("\n#4: Augmented (sparse + neighbor avg sidecar)")
  local d4 = train_sidecar_elm(train_avg, dev_avg, test_avg, label_dims)

  print("\n#5: Combined (sparse + pred + avg sidecar)")
  local d5 = train_sidecar_elm(train_both, dev_both, test_both, both_dims)

  train_pred = nil; dev_pred = nil; test_pred = nil -- luacheck: ignore
  train_avg = nil; dev_avg = nil; test_avg = nil -- luacheck: ignore
  train_both = nil; dev_both = nil; test_both = nil -- luacheck: ignore
  collectgarbage("collect")

  print("\n" .. string.rep("=", 60))
  print("RESULTS")
  print(string.rep("=", 60))
  str.printf("\n  #1 Direct ELM (baseline):\n")
  str.printf("    Oracle:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d1.tr_oracle.macro_f1, d1.dv_oracle.macro_f1, d1.ts_oracle.macro_f1)
  str.printf("    Thresh:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d1.tr_thr.macro_f1, d1.dv_thr.macro_f1, d1.ts_thr.macro_f1)
  str.printf("    K-pred:   Test maF1=%.4f\n", d1.ts_kpred.macro_f1)
  str.printf("  #2 Ceiling (train):\n")
  str.printf("    Oracle:   maF1=%.4f miF1=%.4f\n", ceiling_oracle.macro_f1, ceiling_oracle.micro_f1)
  str.printf("  #3 Distilled (sparse + pred spectral):\n")
  str.printf("    Oracle:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d3.tr_oracle.macro_f1, d3.dv_oracle.macro_f1, d3.ts_oracle.macro_f1)
  str.printf("    Thresh:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d3.tr_thr.macro_f1, d3.dv_thr.macro_f1, d3.ts_thr.macro_f1)
  str.printf("    K-pred:   Test maF1=%.4f\n", d3.ts_kpred.macro_f1)
  str.printf("  #4 Augmented (sparse + neighbor avg):\n")
  str.printf("    Oracle:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d4.tr_oracle.macro_f1, d4.dv_oracle.macro_f1, d4.ts_oracle.macro_f1)
  str.printf("    Thresh:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d4.tr_thr.macro_f1, d4.dv_thr.macro_f1, d4.ts_thr.macro_f1)
  str.printf("    K-pred:   Test maF1=%.4f\n", d4.ts_kpred.macro_f1)
  str.printf("  #5 Combined (sparse + pred + avg):\n")
  str.printf("    Oracle:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d5.tr_oracle.macro_f1, d5.dv_oracle.macro_f1, d5.ts_oracle.macro_f1)
  str.printf("    Thresh:   Train maF1=%.4f  Dev maF1=%.4f  Test maF1=%.4f\n",
    d5.tr_thr.macro_f1, d5.dv_thr.macro_f1, d5.ts_thr.macro_f1)
  str.printf("    K-pred:   Test maF1=%.4f\n", d5.ts_kpred.macro_f1)
  str.printf("  Distillation MAE: %.6f\n", pred_mae.mean)
  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
