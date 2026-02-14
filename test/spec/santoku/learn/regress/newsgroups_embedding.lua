local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.learn.evaluator")
local inv = require("santoku.learn.inv")
local ivec = require("santoku.ivec")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = {
    max = nil,
    tvr = 0.1,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 1,
    cgrams_min = 3,
    cgrams_max = 5,
    cgrams_cross = true,
    skips = 1,
  },
  feature_selection = {
    n_bns = nil,
    n_selected = 65536,
  },
  graph = {
    decay = -10,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 256,
    bandwidth = -1,
  },
  eval = {
    knn = 16,
    random_pairs = 16,
  },
  regressor = {
    features = 4096,
    absorb_interval = 1, --{ def = 10, min = 1, max = 40 },
    absorb_threshold = { def = 0, min = 0, max = 256, int = true },
    absorb_maximum = { def = 256, min = 1, max = 256, int = true },
    absorb_insert_offset = { def = 1, min = 1, max = 256, int = true },
    clauses = 2,
    clause_maximum = { def = 16, min = 8, max = 1024, int = true },
    clause_tolerance_fraction = { def = 0.5, min = 0.01, max = 1.0 },
    target_fraction = { def = 0.25, min = 0.01, max = 2.0 },
    specificity = { def = 800, min = 2, max = 2000 },
    search_trials = 60,
    search_iterations = 10,
    search_subsample_samples = 0.2,
    search_subsample_targets = 8,
    final_patience = 20,
    final_batch = 40,
    final_iterations = 400,
  },
}

test("newsgroups-embedding", function ()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train, test_set, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max,
    nil,
    cfg.data.tvr)
  str.printf("  Train: %d  Validate: %d  Test: %d\n", train.n, validate.n, test_set.n)

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  validate.tokens = tok:tokenize(validate.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil
  validate.problems = nil
  test_set.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens)

  local n_classes = train.n_labels
  print("\nBNS feature selection")
  train.solutions:add_scaled(n_classes)
  local bns_ids, bns_scores = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_classes,
    cfg.feature_selection.n_bns, nil, "max")
  train.solutions:add_scaled(-n_classes)
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  validate.tokens:bits_select(bns_ids, nil, n_tokens)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens)
  local n_features = bns_ids:size()
  n_tokens = n_features
  str.printf("  %d features selected\n", n_features)

  print("\nBuilding doc-doc index (text + label ranks)")
  train.ids = ivec.create(train.n):fill_indices()
  local label_sols = ivec.create():copy(train.solutions)
  local label_idf_ids, label_idf_scores = label_sols:bits_top_df(train.n, n_classes)
  label_sols:bits_select(label_idf_ids, nil, n_classes)
  local n_label_feats = label_idf_ids:size()
  local graph_tokens = ivec.create():copy(train.tokens)
  graph_tokens:bits_extend(label_sols, n_tokens, n_label_feats)
  local graph_weights = dvec.create():copy(bns_scores)
  graph_weights:copy(label_idf_scores)
  local graph_ranks = ivec.create(n_tokens + n_label_feats)
  graph_ranks:fill(0, 0, n_tokens)
  graph_ranks:fill(1, n_tokens, n_tokens + n_label_feats)
  local index = inv.create({ features = graph_weights, ranks = graph_ranks, n_ranks = 2 })
  index:add(graph_tokens, train.ids)
  str.printf("  %d docs indexed, %d text features + %d IDF-weighted label features, decay=%.1f\n",
    train.n, n_tokens, n_label_feats, cfg.graph.decay)

  print("\nBuilding eval adjacency")
  local eval_uids, inv_hoods = index:neighborhoods(cfg.eval.knn, cfg.graph.decay, -1)
  local eval_off, eval_nbr, eval_w = inv_hoods:to_csr(eval_uids)
  local rp_off, rp_nbr, rp_w = csr.random_pairs(eval_uids, cfg.eval.random_pairs)
  csr.weight_from_index(eval_uids, rp_off, rp_nbr, rp_w, index, cfg.graph.decay, -1)
  csr.merge(eval_off, eval_nbr, eval_w, rp_off, rp_nbr, rp_w)
  csr.symmetrize(eval_off, eval_nbr, eval_w, eval_uids:size())
  str.printf("  %d nodes, %d edges\n", eval_uids:size(), eval_nbr:size())

  print("\nSpectral embedding (Nystrom)")
  local model = optimize.spectral({
    index = index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.graph.decay,
    bandwidth = cfg.nystrom.bandwidth,
    expected_ids = eval_uids,
    expected_offsets = eval_off,
    expected_neighbors = eval_nbr,
    expected_weights = eval_w,
    each = function (ev)
      util.spectral_log(ev)
    end,
  })
  local spectral_dims = model.dims
  local all_raw_codes = model.raw_codes
  local embedded_ids = model.ids
  str.printf("  dims=%d, embedded=%d\n", spectral_dims, embedded_ids:size())

  local train_raw_codes = dvec.create():mtx_extend(
    all_raw_codes, train.ids, embedded_ids, 0, spectral_dims, true)

  local train_bns_tokens = ivec.create():copy(train.tokens)
  local val_bns_tokens = ivec.create():copy(validate.tokens)
  local test_bns_tokens = ivec.create():copy(test_set.tokens)

  print("\nFeature selection for regressor (F-score)")
  local n_selected = cfg.feature_selection.n_selected
  local union_ids, _, class_offsets, class_feat_ids = train.tokens:bits_top_reg_f(
    train_raw_codes, train.n, n_tokens, spectral_dims, n_selected, nil, "sum")
  train.tokens:bits_select(union_ids, nil, n_tokens)
  validate.tokens:bits_select(union_ids, nil, n_tokens)
  test_set.tokens:bits_select(union_ids, nil, n_tokens)
  class_offsets, class_feat_ids = csr.bits_select(class_offsets, class_feat_ids, union_ids)
  n_tokens = union_ids:size()
  str.printf("  %d features selected\n", n_tokens)

  print("\nBuilding CSC index")
  local csc_offsets, csc_indices = csr.to_csc(train.tokens, train.n, n_tokens)
  str.printf("  Tokens: %d  Samples: %d\n", n_tokens, train.n)

  local absorb_ranking_global = ivec.create(n_tokens):fill_indices()

  print("\nTraining regressor")
  local predicted_buf = dvec.create()
  local tm = optimize.regressor({
    outputs = spectral_dims,
    samples = train.n,
    features = cfg.regressor.features,
    n_tokens = n_tokens,
    absorb_interval = cfg.regressor.absorb_interval,
    absorb_threshold = cfg.regressor.absorb_threshold,
    absorb_maximum = cfg.regressor.absorb_maximum,
    absorb_insert_offset = cfg.regressor.absorb_insert_offset,
    clauses = cfg.regressor.clauses,
    clause_maximum = cfg.regressor.clause_maximum,
    clause_tolerance_fraction = cfg.regressor.clause_tolerance_fraction,
    target_fraction = cfg.regressor.target_fraction,
    specificity = cfg.regressor.specificity,
    output_weights = model.eigenvalues,
    tokens = train.tokens,
    csc_offsets = csc_offsets,
    csc_indices = csc_indices,
    absorb_ranking = class_feat_ids,
    absorb_ranking_offsets = class_offsets,
    absorb_ranking_global = absorb_ranking_global,
    targets = train_raw_codes,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    search_subsample_samples = cfg.regressor.search_subsample_samples,
    search_subsample_targets = cfg.regressor.search_subsample_targets,
    final_batch = cfg.regressor.final_batch,
    final_patience = cfg.regressor.final_patience,
    final_iterations = cfg.regressor.final_iterations,
    search_metric = function (t, targs)
      local input = { tokens = targs.tokens, n_samples = targs.samples }
      local predicted = t:regress(input, targs.samples, true, predicted_buf)
      local stats = eval.regression_accuracy(predicted, targs.targets)
      return -stats.mean, stats
    end,
    each = util.make_regressor_log(stopwatch),
  })

  print("\n" .. string.rep("=", 60))
  print("FINAL EVALUATION")
  print(string.rep("=", 60))

  local function build_split_eval(tokens, n)
    local ids = ivec.create(n):fill_indices()
    local idx = inv.create({ features = bns_scores })
    idx:add(tokens, ids)
    local uids, hoods = idx:neighborhoods(cfg.eval.knn, 0, -1)
    local off, nbr, w = hoods:to_csr(uids)
    local rp_off, rp_nbr, rp_w = csr.random_pairs(uids, cfg.eval.random_pairs)
    csr.weight_from_index(uids, rp_off, rp_nbr, rp_w, idx, 0, -1)
    csr.merge(off, nbr, w, rp_off, rp_nbr, rp_w)
    csr.symmetrize(off, nbr, w, uids:size())
    return uids, off, nbr, w
  end

  print("\nBuilding per-split evaluation indexes")
  local tr_uids, tr_off, tr_nbr, tr_w = build_split_eval(train_bns_tokens, train.n)
  local va_uids, va_off, va_nbr, va_w = build_split_eval(val_bns_tokens, validate.n)
  local te_uids, te_off, te_nbr, te_w = build_split_eval(test_bns_tokens, test_set.n)
  str.printf("  Train: %d nodes, %d edges\n", tr_uids:size(), tr_nbr:size())
  str.printf("  Validate: %d nodes, %d edges\n", va_uids:size(), va_nbr:size())
  str.printf("  Test: %d nodes, %d edges\n", te_uids:size(), te_nbr:size())

  print("\nPredicting embeddings")
  local train_predicted = tm:regress(
    { tokens = train.tokens, n_samples = train.n }, train.n, true)
  local val_predicted = tm:regress(
    { tokens = validate.tokens, n_samples = validate.n }, validate.n, true)
  local test_predicted = tm:regress(
    { tokens = test_set.tokens, n_samples = test_set.n }, test_set.n, true)

  local train_ids = ivec.create(train.n):fill_indices()
  local val_ids = ivec.create(validate.n):fill_indices()
  local test_ids = ivec.create(test_set.n):fill_indices()

  local spectral_raw = eval.ranking_accuracy({
    raw_codes = train_raw_codes, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = tr_uids, eval_offsets = tr_off,
    eval_neighbors = tr_nbr, eval_weights = tr_w,
  })

  local train_ranking = eval.ranking_accuracy({
    raw_codes = train_predicted, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = tr_uids, eval_offsets = tr_off,
    eval_neighbors = tr_nbr, eval_weights = tr_w,
  })

  local val_ranking = eval.ranking_accuracy({
    raw_codes = val_predicted, ids = val_ids,
    n_dims = spectral_dims,
    eval_ids = va_uids, eval_offsets = va_off,
    eval_neighbors = va_nbr, eval_weights = va_w,
  })

  local test_ranking = eval.ranking_accuracy({
    raw_codes = test_predicted, ids = test_ids,
    n_dims = spectral_dims,
    eval_ids = te_uids, eval_offsets = te_off,
    eval_neighbors = te_nbr, eval_weights = te_w,
  })

  local train_reg = eval.regression_accuracy(train_predicted, train_raw_codes)

  print("\nEigenvalue spectrum")
  do
    local ev = model.eigenvalues
    local n_ev = ev:size()
    local top = math.min(n_ev, 10)
    for i = 0, top - 1 do
      str.printf("  EV[%d] = %.6f\n", i, ev:get(i))
    end
    if n_ev > 10 then
      str.printf("  ... (%d total)\n", n_ev)
      str.printf("  EV[%d] = %.6f (last)\n", n_ev - 1, ev:get(n_ev - 1))
    end
  end

  print("\nPer-dimension regression analysis")
  do
    local D = spectral_dims
    local pd = eval.regression_per_dim(train_predicted, train_raw_codes, train.n, D)
    str.printf("  %-6s %10s %10s %10s %12s\n", "Dim", "MAE", "Pearson r", "Var ratio", "Eigenvalue")
    local ev = model.eigenvalues
    local bands = { { 0, math.min(7, D - 1) } }
    if D > 8 then bands[#bands + 1] = { 8, math.min(31, D - 1) } end
    if D > 32 then bands[#bands + 1] = { 32, math.min(63, D - 1) } end
    if D > 64 then bands[#bands + 1] = { 64, math.min(127, D - 1) } end
    if D > 128 then bands[#bands + 1] = { 128, math.min(255, D - 1) } end
    if D > 256 then bands[#bands + 1] = { 256, D - 1 } end
    for _, band in ipairs(bands) do
      local lo, hi = band[1], band[2]
      local cnt = hi - lo + 1
      local s_mae, s_corr, s_vr, s_ev = 0, 0, 0, 0
      for d = lo, hi do
        s_mae = s_mae + pd.mae:get(d)
        s_corr = s_corr + pd.corr:get(d)
        s_vr = s_vr + pd.var_ratio:get(d)
        if d < ev:size() then s_ev = s_ev + ev:get(d) end
      end
      str.printf("  [%3d-%3d] %8.6f %10.4f %10.4f %12.6f\n",
        lo, hi, s_mae / cnt, s_corr / cnt, s_vr / cnt, s_ev / cnt)
    end
    local worst_mae_d = pd.mae:rmaxargs(D):get(0)
    local best_corr_d = pd.corr:rmaxargs(D):get(0)
    local worst_corr_d = pd.corr:rminargs(D):get(0)
    str.printf("  Best corr:  dim %d = %.4f\n", best_corr_d, pd.corr:get(best_corr_d))
    str.printf("  Worst corr: dim %d = %.4f\n", worst_corr_d, pd.corr:get(worst_corr_d))
    str.printf("  Worst MAE:  dim %d = %.6f\n", worst_mae_d, pd.mae:get(worst_mae_d))
  end

  print("\nAsymmetric predicted->spectral evaluation (train)")
  do
    local n = train.n
    local pred_ids = ivec.create(n):fill_indices()
    local spec_ids = ivec.create(n):fill_indices():add(n)

    local asym_gt_off = ivec.create(n + 1):fill_indices()
    local asym_gt_nbr = ivec.create(n):fill_indices()
    local asym_ids, asym_off, asym_nbr, asym_w = csr.bipartite_neg(
      asym_gt_off, asym_gt_nbr, pred_ids, spec_ids, cfg.eval.random_pairs)
    str.printf("  Asymmetric eval: %d nodes, %d edges\n", asym_ids:size(), asym_nbr:size())

    local asym_all_ids = ivec.create(n):fill_indices()
    asym_all_ids:copy(spec_ids)

    local spec_spec_raw = dvec.create()
    spec_spec_raw:copy(train_raw_codes)
    spec_spec_raw:copy(train_raw_codes)
    local spec_to_spec = eval.ranking_accuracy({
      raw_codes = spec_spec_raw, ids = asym_all_ids, n_dims = spectral_dims,
      eval_ids = asym_ids, eval_offsets = asym_off,
      eval_neighbors = asym_nbr, eval_weights = asym_w,
    })

    local pred_spec_raw = dvec.create()
    pred_spec_raw:copy(train_predicted)
    pred_spec_raw:copy(train_raw_codes)
    local pred_to_spec = eval.ranking_accuracy({
      raw_codes = pred_spec_raw, ids = asym_all_ids, n_dims = spectral_dims,
      eval_ids = asym_ids, eval_offsets = asym_off,
      eval_neighbors = asym_nbr, eval_weights = asym_w,
    })

    str.printf("  %-28s %8.4f\n", "Spectral->spectral:", spec_to_spec.score)
    str.printf("  %-28s %8.4f\n", "Predicted->spectral:", pred_to_spec.score)
  end

  str.printf("\n  Spectral dims: %d\n", spectral_dims)
  str.printf("  %-28s %8.4f\n", "Spectral raw (train ceil):", spectral_raw.score)
  str.printf("  %-28s %8.4f\n", "Predicted train:", train_ranking.score)
  str.printf("  %-28s %8.4f\n", "Predicted validate:", val_ranking.score)
  str.printf("  %-28s %8.4f\n", "Predicted test:", test_ranking.score)
  str.printf("\n  Regression MAE (train): %.4f\n", train_reg.mean)

  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
