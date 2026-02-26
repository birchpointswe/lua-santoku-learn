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
  text_index = "bns",
  elm_weights = "bns",
  bns = { per_class_k = nil },
  data = { max = nil, },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 0,
    cgrams_max = 0,
    cgrams_cross = false,
    skips = 1,
  },
  feature_selection = {
    n_selected = 65536,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 1024,
    decay = 0,
  },
  align = {
    lambda = { def = 1.0 },
    search_trials = 80,
  },
  augmented = {
    knn = 32,
  },
  label_eval = {
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 80,
    k = 32,
  },
  elm_direct = {
    n_hidden = 256, --{ min = 32, max = 1024, pow2 = true },
    seed = 42,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 80,
    search_subsample = nil,
    k = 32,
  },
  elm = {
    n_hidden = 8192, --{ min = 32, max = 1024, pow2 = true },
    seed = 42,
    lambda = { def = 1.0 },
    search_trials = 80,
    search_subsample = nil,
  },
  regressor = {
    flat = true,
    flat_evict = false,
    flat_encoding = "hadamard",
    flat_skip = nil,
    cost_beta = nil,
    features = 8192,
    clauses = 16,
    clause_maximum_fraction = { def = 0.03 },
    clause_tolerance_fraction = { def = 0.57 },
    specificity_fraction = { def = 0.00095 },
    absorb_threshold = { def = 9 },
    absorb_maximum_fraction = { def = 0.029 },
    absorb_insert_offset = { def = 49 },
    absorb_ranking_fraction = { def = 0.125 },
    absorb_ranking_limit = { def = 0.125 },
    target_fraction = { def = 0.10 },
    search_trials = 200,
    search_iterations = 20,
    search_subsample = 0.05,
    final_patience = 2,
    final_batch = 40,
    final_iterations = 800,
  },
}

test("eurlex-aligned", function ()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  str.printf("  Train: %d  Labels: %d\n", train.n, n_labels)

  print("\nBuilding label CSR")
  local label_offsets, label_neighbors = train.solutions:bits_to_csr(train.n, n_labels)

  train.ids = ivec.create(train.n):fill_indices()

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens_raw = tok:features()
  train.tokens = tok:tokenize(train.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens_raw)

  print("\nComputing text scores")
  local text_idf_ids, text_idf_scores = train.tokens:bits_top_idf(train.n, n_tokens_raw)
  str.printf("  IDF features: %d\n", text_idf_ids:size())
  local text_bns_ids, text_bns_scores
  if cfg.text_index == "bns" or cfg.elm_weights == "bns" then
    text_bns_ids, text_bns_scores = train.tokens:bits_top_bns(
      train.solutions, train.n, n_tokens_raw, n_labels, cfg.bns.per_class_k, nil, "max")
    str.printf("  BNS features: %d\n", text_bns_ids:size())
  end
  print("\nBuilding ELM CSC")
  local elm_feature_weights, n_elm_tokens, elm_csc_offsets, elm_csc_indices
  if cfg.elm_weights ~= "none" then
    local elm_feat_ids = cfg.elm_weights == "bns" and text_bns_ids or text_idf_ids
    elm_feature_weights = cfg.elm_weights == "bns" and text_bns_scores or text_idf_scores
    local elm_bits = ivec.create():copy(train.tokens)
    elm_bits:bits_select(elm_feat_ids, nil, n_tokens_raw)
    n_elm_tokens = elm_feat_ids:size()
    elm_csc_offsets, elm_csc_indices = csr.to_csc(elm_bits, train.n, n_elm_tokens)
    str.printf("  ELM tokens: %d (%s weighted)\n", n_elm_tokens, cfg.elm_weights:upper())
  else
    n_elm_tokens = n_tokens_raw
    elm_csc_offsets, elm_csc_indices = csr.to_csc(train.tokens, train.n, n_elm_tokens)
  end

  print("\nDirect ELM text->labels")
  local elm_direct_obj, elm_direct_params, elm_direct_metrics = optimize.elm({
    n_samples = train.n,
    n_tokens = n_elm_tokens,
    n_hidden = cfg.elm_direct.n_hidden,
    seed = cfg.elm_direct.seed,
    csc_offsets = elm_csc_offsets,
    csc_indices = elm_csc_indices,
    feature_weights = elm_feature_weights,
    n_labels = n_labels,
    label_offsets = label_offsets,
    label_neighbors = label_neighbors,
    expected_offsets = label_offsets,
    expected_neighbors = label_neighbors,
    lambda = cfg.elm_direct.lambda,
    propensity_a = cfg.elm_direct.propensity_a,
    propensity_b = cfg.elm_direct.propensity_b,
    k = cfg.elm_direct.k,
    search_trials = cfg.elm_direct.search_trials,
    search_subsample = cfg.elm_direct.search_subsample,
    stratify_offsets = label_offsets,
    stratify_neighbors = label_neighbors,
    stratify_labels = n_labels,
    each = util.make_elm_log(),
  })
  local elm_direct_mi, elm_direct_ma = elm_direct_obj:label_f1(
    elm_csc_offsets, elm_csc_indices, train.n, label_offsets, label_neighbors)
  str.printf("  Direct ELM: maF1=%.4f miF1=%.4f H=%d lambda=%.4e\n",
    elm_direct_ma, elm_direct_mi, elm_direct_params.n_hidden, elm_direct_params.lambda)

  print("\nBuilding label index")
  local label_sols = ivec.create():copy(train.solutions)
  local label_idf_ids, label_idf_scores = label_sols:bits_top_idf(train.n, n_labels)
  label_sols:bits_select(label_idf_ids, nil, n_labels)
  local label_index = inv.create({ features = label_idf_scores })
  label_index:add(label_sols, train.ids)
  str.printf("  Label index: %d IDF feats, %d docs\n", label_idf_ids:size(), train.n)

  print("\nLabel spectral embedding")
  local label_model = optimize.spectral({
    index = label_index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    each = function (ev) util.spectral_log(ev) end,
  })
  str.printf("  Label spectral: %d dims\n", label_model.dims)

  print("\nBuilding text index")
  local text_bits = ivec.create():copy(train.tokens)
  local text_feat_ids = cfg.text_index == "bns" and text_bns_ids or text_idf_ids
  local text_feat_scores = cfg.text_index == "bns" and text_bns_scores or text_idf_scores
  text_bits:bits_select(text_feat_ids, nil, n_tokens_raw)
  local text_index = inv.create({ features = text_feat_scores })
  text_index:add(text_bits, train.ids)
  str.printf("  Text index: %d %s feats, %d docs\n", text_feat_ids:size(), cfg.text_index:upper(), train.n)

  print("\nText spectral embedding")
  local text_model = optimize.spectral({
    index = text_index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    each = function (ev) util.spectral_log(ev) end,
  })
  str.printf("  Text spectral: %d dims\n", text_model.dims)

  local label_dims = label_model.dims
  local text_dims = text_model.dims
  str.printf("  Label dims: %d  Text dims: %d\n", label_dims, text_dims)

  local label_codes = dvec.create():mtx_extend(label_model.raw_codes,
    label_model.ids:set_intersect(ivec.create(train.n):fill_indices()),
    label_model.ids, 0, label_dims, true)

  local text_codes = dvec.create():mtx_extend(text_model.raw_codes,
    text_model.ids:set_intersect(ivec.create(train.n):fill_indices()),
    text_model.ids, 0, text_dims, true)

  local train_ids = ivec.create(train.n):fill_indices()

  print("\nBuilding teachers")
  local label_itq = quantizer.create({ mode = "itq", raw_codes = label_codes, n_samples = train.n })
  local label_bin = label_itq:encode(label_codes)
  local label_teacher_ann = ann.create({ features = label_dims })
  label_teacher_ann:add(label_bin, train_ids)

  local text_itq = quantizer.create({ mode = "itq", raw_codes = text_codes, n_samples = train.n })
  local text_bin = text_itq:encode(text_codes)
  local text_teacher_ann = ann.create({ features = text_dims })
  text_teacher_ann:add(text_bin, train_ids)
  str.printf("  Label teacher: %d bits  Text teacher: %d bits\n", label_dims, text_dims)

  local mae_score_fn = function (r, data)
    local transformed = r:transform(data.codes, data.n_samples)
    local ra = eval.regression_accuracy(transformed, data.targets)
    return -ra.mean, { mae = ra.mean }
  end

  print("\nRidge alignment: text -> label (MAE)")
  local t2l_ridge = optimize.ridge({
    n_samples = train.n, n_dims = text_dims, n_targets = label_dims,
    codes = text_codes, targets = label_codes,
    lambda = cfg.align.lambda,
    search_trials = cfg.align.search_trials,
    score_fn = mae_score_fn,
    each = util.make_ridge_log(),
  })
  local aligned_t2l = t2l_ridge:transform(text_codes, train.n)
  local t2l_mae = eval.regression_accuracy(aligned_t2l, label_codes)
  str.printf("  text->label MAE: %.6f\n", t2l_mae.mean)

  print("\nRidge alignment: label -> text (MAE)")
  local l2t_ridge = optimize.ridge({
    n_samples = train.n, n_dims = label_dims, n_targets = text_dims,
    codes = label_codes, targets = text_codes,
    lambda = cfg.align.lambda,
    search_trials = cfg.align.search_trials,
    score_fn = mae_score_fn,
    each = util.make_ridge_log(),
  })
  local aligned_l2t = l2t_ridge:transform(label_codes, train.n)
  local l2t_mae = eval.regression_accuracy(aligned_l2t, text_codes)
  str.printf("  label->text MAE: %.6f\n", l2t_mae.mean)

  local function label_eval (name, codes, dims)
    str.printf("\n  Label eval: %s\n", name)
    local _, _, metrics = optimize.ridge({
      n_samples = train.n, n_dims = dims,
      n_labels = n_labels, codes = codes,
      label_offsets = label_offsets, label_neighbors = label_neighbors,
      expected_offsets = label_offsets, expected_neighbors = label_neighbors,
      lambda = cfg.label_eval.lambda,
      propensity_a = cfg.label_eval.propensity_a,
      propensity_b = cfg.label_eval.propensity_b,
      k = cfg.label_eval.k,
      search_trials = cfg.label_eval.search_trials,
      each = util.make_ridge_log(),
    })
    local t = metrics.thresh
    str.printf("  %-50s maF1=%.4f miF1=%.4f P=%.4f R=%.4f\n",
      name .. ":", t.macro_f1, t.micro_f1, t.macro_precision, t.macro_recall)
    return t
  end

  local function augment (codes, dims, teacher_itq, teacher_ann, teacher_codes)
    local bin = teacher_itq:encode(codes)
    local hood_ids, ann_hoods = teacher_ann:neighborhoods_by_vecs(bin, cfg.augmented.knn)
    local nn_off, nn_nbr, nn_w = ann_hoods:to_csr(dims)
    local avg = csr.neighbor_average(nn_off, nn_nbr, nn_w, hood_ids, teacher_codes, train_ids, dims)
    local aug = dvec.create():copy(codes)
    aug:mtx_extend(avg, dims, dims)
    return aug
  end

  local function eval_pair (name, codes, dims, teacher_itq, teacher_ann, teacher_codes)
    local direct = label_eval(name, codes, dims)
    local aug = augment(codes, dims, teacher_itq, teacher_ann, teacher_codes)
    local augmented = label_eval(name .. " +aug", aug, 2 * dims)
    return direct, augmented
  end

  print("\nPre-TM label evaluations")
  local t2l_f1, t2l_f1_aug = eval_pair("text->label",
    aligned_t2l, label_dims, label_itq, label_teacher_ann, label_codes)
  local l2t_f1, l2t_f1_aug = eval_pair("label->text",
    aligned_l2t, text_dims, text_itq, text_teacher_ann, text_codes)

  print("\nELM spectral evaluation")

  local function elm_eval (name, target_codes, dims, teacher_itq, teacher_ann, teacher_codes)
    str.printf("\n  ELM %s\n", name)
    local elm_obj, elm_params = optimize.elm({
      n_samples = train.n,
      n_tokens = n_elm_tokens,
      n_hidden = cfg.elm.n_hidden,
      seed = cfg.elm.seed,
      csc_offsets = elm_csc_offsets,
      csc_indices = elm_csc_indices,
      feature_weights = elm_feature_weights,
      targets = target_codes,
      n_targets = dims,
      lambda = cfg.elm.lambda,
      search_trials = cfg.elm.search_trials,
      search_subsample = cfg.elm.search_subsample,
      score_fn = mae_score_fn,
      each = util.make_elm_log(),
    })
    local neg_mae, mae = elm_obj:regress_mae(elm_csc_offsets, elm_csc_indices, train.n, target_codes)
    str.printf("  %-50s MAE=%.6f H=%d lambda=%.4e\n",
      "ELM " .. name .. ":", mae, elm_params.n_hidden, elm_params.lambda)
    local predicted = elm_obj:regress(elm_csc_offsets, elm_csc_indices, train.n)
    local direct, augmented = eval_pair(
      "ELM " .. name, predicted, dims, teacher_itq, teacher_ann, teacher_codes)
    return neg_mae, mae, direct, augmented
  end

  local _, elm_t2l_mae, elm_t2l_f1, elm_t2l_f1_aug = elm_eval(
    "text->label", aligned_t2l, label_dims, label_itq, label_teacher_ann, label_codes)
  local _, elm_l2t_mae, elm_l2t_f1, elm_l2t_f1_aug = elm_eval(
    "label->text", aligned_l2t, text_dims, text_itq, text_teacher_ann, text_codes)
  local _, elm_lab_mae, elm_lab_f1, elm_lab_f1_aug = elm_eval(
    "label", label_codes, label_dims, label_itq, label_teacher_ann, label_codes)

  local n_selected = cfg.feature_selection.n_selected

  local function train_and_eval (name, target_codes, dims, teacher_itq, teacher_ann, teacher_codes)
    str.printf("\n%s\n%s\n", string.rep("=", 60), name)
    local tokens = ivec.create():copy(train.tokens)
    local n_tokens = n_tokens_raw
    local union_ids, _, class_offsets, class_feat_ids = tokens:bits_top_reg_f(
      target_codes, train.n, n_tokens, dims, n_selected, nil, "max")
    tokens:bits_select(union_ids, nil, n_tokens)
    class_offsets, class_feat_ids = csr.bits_select(class_offsets, class_feat_ids, union_ids)
    n_tokens = union_ids:size()
    str.printf("  Feature selection: %d features\n", n_tokens)
    local csc_offsets, csc_indices = csr.to_csc(tokens, train.n, n_tokens)
    local tm_obj = optimize.regressor({
      flat = cfg.regressor.flat,
      flat_evict = cfg.regressor.flat_evict,
      flat_encoding = cfg.regressor.flat_encoding,
      flat_skip = cfg.regressor.flat_skip,
      cost_beta = cfg.regressor.cost_beta,
      outputs = dims,
      samples = train.n,
      features = cfg.regressor.features,
      n_tokens = n_tokens,
      absorb_threshold = cfg.regressor.absorb_threshold,
      absorb_maximum_fraction = cfg.regressor.absorb_maximum_fraction,
      absorb_insert_offset = cfg.regressor.absorb_insert_offset,
      absorb_ranking_fraction = cfg.regressor.absorb_ranking_fraction,
      absorb_ranking_limit = cfg.regressor.absorb_ranking_limit,
      clauses = cfg.regressor.clauses,
      clause_maximum_fraction = cfg.regressor.clause_maximum_fraction,
      clause_tolerance_fraction = cfg.regressor.clause_tolerance_fraction,
      target_fraction = cfg.regressor.target_fraction,
      specificity_fraction = cfg.regressor.specificity_fraction,
      tokens = tokens,
      csc_offsets = csc_offsets,
      csc_indices = csc_indices,
      absorb_ranking = class_feat_ids,
      absorb_ranking_offsets = class_offsets,
      absorb_ranking_global = ivec.create(n_tokens):fill_indices(),
      targets = target_codes,
      search_trials = cfg.regressor.search_trials,
      search_iterations = cfg.regressor.search_iterations,
      search_subsample = cfg.regressor.search_subsample,
      stratify_offsets = label_offsets,
      stratify_neighbors = label_neighbors,
      stratify_labels = n_labels,
      final_batch = cfg.regressor.final_batch,
      final_patience = cfg.regressor.final_patience,
      final_iterations = cfg.regressor.final_iterations,
      each = util.make_regressor_log(stopwatch),
    })
    print("\nEvaluating")
    local predicted = tm_obj:regress(
      { tokens = tokens, n_samples = train.n }, train.n, true)
    local mae = eval.regression_accuracy(predicted, target_codes)
    str.printf("  %-50s MAE=%.6f\n", name .. ":", mae.mean)
    local direct, augmented = eval_pair(
      "TM " .. name, predicted, dims, teacher_itq, teacher_ann, teacher_codes)
    tm_obj:destroy()
    collectgarbage("collect")
    return mae, direct, augmented
  end

  local tm_t2l_mae, tm_t2l_f1, tm_t2l_f1_aug = train_and_eval(
    "text->label", aligned_t2l, label_dims, label_itq, label_teacher_ann, label_codes)
  local tm_l2t_mae, tm_l2t_f1, tm_l2t_f1_aug = train_and_eval(
    "label->text", aligned_l2t, text_dims, text_itq, text_teacher_ann, text_codes)
  local tm_lab_mae, tm_lab_f1, tm_lab_f1_aug = train_and_eval(
    "label", label_codes, label_dims, label_itq, label_teacher_ann, label_codes)

  print("\n" .. string.rep("=", 60))
  print("RESULTS")
  print(string.rep("=", 60))
  str.printf("\n  Label dims: %d  Text dims: %d\n", label_dims, text_dims)
  str.printf("\n  Alignment MAE:\n")
  str.printf("  %-50s %.6f\n", "text->label:", t2l_mae.mean)
  str.printf("  %-50s %.6f\n", "label->text:", l2t_mae.mean)
  str.printf("\n  ELM MAE:\n")
  str.printf("  %-50s %.6f\n", "ELM text->label:", elm_t2l_mae)
  str.printf("  %-50s %.6f\n", "ELM label->text:", elm_l2t_mae)
  str.printf("  %-50s %.6f\n", "ELM label:", elm_lab_mae)
  str.printf("\n  TM MAE:\n")
  str.printf("  %-50s %.6f\n", "TM text->label:", tm_t2l_mae.mean)
  str.printf("  %-50s %.6f\n", "TM label->text:", tm_l2t_mae.mean)
  str.printf("  %-50s %.6f\n", "TM label:", tm_lab_mae.mean)
  str.printf("\n  Label prediction (macro F1):\n")
  str.printf("  %-50s %.4f (miF1=%.4f)\n", "ELM direct:", elm_direct_ma, elm_direct_mi)
  str.printf("  %-50s %.4f\n", "text->label:", t2l_f1.macro_f1)
  str.printf("  %-50s %.4f\n", "text->label +aug:", t2l_f1_aug.macro_f1)
  str.printf("  %-50s %.4f\n", "label->text:", l2t_f1.macro_f1)
  str.printf("  %-50s %.4f\n", "label->text +aug:", l2t_f1_aug.macro_f1)
  str.printf("  %-50s %.4f\n", "ELM text->label:", elm_t2l_f1.macro_f1)
  str.printf("  %-50s %.4f\n", "ELM text->label +aug:", elm_t2l_f1_aug.macro_f1)
  str.printf("  %-50s %.4f\n", "ELM label->text:", elm_l2t_f1.macro_f1)
  str.printf("  %-50s %.4f\n", "ELM label->text +aug:", elm_l2t_f1_aug.macro_f1)
  str.printf("  %-50s %.4f\n", "ELM label:", elm_lab_f1.macro_f1)
  str.printf("  %-50s %.4f\n", "ELM label +aug:", elm_lab_f1_aug.macro_f1)
  str.printf("  %-50s %.4f\n", "TM text->label:", tm_t2l_f1.macro_f1)
  str.printf("  %-50s %.4f\n", "TM text->label +aug:", tm_t2l_f1_aug.macro_f1)
  str.printf("  %-50s %.4f\n", "TM label->text:", tm_l2t_f1.macro_f1)
  str.printf("  %-50s %.4f\n", "TM label->text +aug:", tm_l2t_f1_aug.macro_f1)
  str.printf("  %-50s %.4f\n", "TM label:", tm_lab_f1.macro_f1)
  str.printf("  %-50s %.4f\n", "TM label +aug:", tm_lab_f1_aug.macro_f1)
  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
