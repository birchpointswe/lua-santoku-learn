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
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 1,
    cgrams_min = 0,
    cgrams_max = 0,
    cgrams_cross = false,
    skips = 0,
  },
  feature_selection = {
    n_bns = 8192,
    n_selected = 65536,
  },
  nystrom = {
    idf = false,
    n_landmarks = 4096,
    n_dims = 256,
    decay = 0,
    bandwidth = -1,
  },
  regressor = {
    features = 256, --{ def = 4096, min = 512, max = 4096, pow2 = true },
    absorb_interval = 1,
    absorb_threshold = { def = 58, min = 0, max = 256, int = true },
    absorb_maximum = { def = 120, min = 1, max = 256, int = true },
    absorb_insert_offset = { def = 17, min = 1, max = 256, int = true },
    clauses = { def = 4, min = 1, max = 8, int = true },
    clause_maximum = { def = 102, min = 8, max = 512, int = true },
    clause_tolerance_fraction = { def = 0.76, min = 0.01, max = 1.0 },
    target_fraction = { def = 0.32, min = 0.01, max = 2.0 },
    specificity = { def = 512, min = 2, max = 2000 },
    alpha_tolerance = { def = -0.3, min = -3, max = 3 },
    alpha_maximum = { def = 0.7, min = -3, max = 3 },
    alpha_target = { def = 1.2, min = -3, max = 3 },
    alpha_specificity = { def = 2.1, min = -3, max = 3 },
    search_trials = 100,
    search_iterations = 20,
    search_subsample_targets = 8,
    final_patience = 2,
    final_batch = 20,
    final_iterations = 300,
  },
  ridge = {
    lambda = { def = 0.5, min = 0.01, max = 10 },
    propensity_a = { def = 0.55, min = 0.1, max = 2.0 },
    propensity_b = { def = 1.5, min = 0.1, max = 5.0 },
    k = 32,
    search_trials = 30,
  },
}

test("eurlex", function()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local train_lc = train.label_counts:to_dvec()
  train_lc:asc()
  str.printf("  Train: %d  Labels: %d  Labels/doc: %.1f median\n",
    train.n, n_labels, train_lc:get(train_lc:size() / 2))
  train_lc = nil -- luacheck: ignore

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens)

  local n_classes = n_labels
  print("\nBNS feature selection")
  train.solutions:add_scaled(n_classes)
  local bns_ids = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_classes,
    cfg.feature_selection.n_bns, nil, "max")
  train.solutions:add_scaled(-n_classes)
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  n_tokens = bns_ids:size()
  str.printf("  %d features selected\n", n_tokens)

  print("\nBuilding label CSR")
  local train_label_offsets, train_label_neighbors = train.solutions:bits_to_csr(train.n, n_labels)
  train.label_csr = { offsets = train_label_offsets, neighbors = train_label_neighbors }

  print("\nBuilding doc-only label index")
  train.ids = ivec.create(train.n):fill_indices()
  local label_sols = ivec.create():copy(train.solutions)
  local label_idf_ids, label_idf_scores = label_sols:bits_top_df(train.n, n_labels)
  label_sols:bits_select(label_idf_ids, nil, n_labels)
  local n_label_feats = label_idf_ids:size()
  local label_index = inv.create({ features = cfg.nystrom.idf and label_idf_scores or label_idf_scores:size() })
  label_index:add(label_sols, train.ids)
  str.printf("  Label index: %d label features, %d docs\n", n_label_feats, train.n)

  print("\nSpectral embedding (Nystrom)")
  local model = optimize.spectral({
    index = label_index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    bandwidth = cfg.nystrom.bandwidth,
    each = function(ev) util.spectral_log(ev) end,
  })
  local spectral_dims = model.dims
  local train_raw_codes = dvec.create():mtx_extend(model.raw_codes,
    model.ids:set_intersect(ivec.create(train.n):fill_indices()),
    model.ids, 0, spectral_dims, true)
  str.printf("  Spectral dims: %d, embedded: %d\n", spectral_dims, model.ids:size())

  print("\nFeature selection for regressor (F-score)")
  local n_selected = cfg.feature_selection.n_selected
  local union_ids, _, class_offsets, class_feat_ids = train.tokens:bits_top_reg_f(
    train_raw_codes, train.n, n_tokens, spectral_dims, n_selected, nil, "sum")
  train.tokens:bits_select(union_ids, nil, n_tokens)
  class_offsets, class_feat_ids = csr.bits_select(class_offsets, class_feat_ids, union_ids)
  n_tokens = union_ids:size()
  str.printf("  %d features selected\n", n_tokens)

  print("\nBuilding CSC index")
  local csc_offsets, csc_indices = csr.to_csc(train.tokens, train.n, n_tokens)
  str.printf("  Tokens: %d  Samples: %d\n", n_tokens, train.n)

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
    alpha_tolerance = cfg.regressor.alpha_tolerance,
    alpha_maximum = cfg.regressor.alpha_maximum,
    alpha_target = cfg.regressor.alpha_target,
    alpha_specificity = cfg.regressor.alpha_specificity,
    output_weights = model.eigenvalues,
    tokens = train.tokens,
    csc_offsets = csc_offsets,
    csc_indices = csc_indices,
    absorb_ranking = class_feat_ids,
    absorb_ranking_offsets = class_offsets,
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

  print("\nPredicting embeddings (train)")
  local train_predicted = tm:regress(
    { tokens = train.tokens, n_samples = train.n }, train.n, true)

  local pred_stats = eval.regression_accuracy(train_predicted, train_raw_codes)
  str.printf("  Regression MAE: %.6f\n", pred_stats.mean)

  local pd = eval.regression_per_dim(train_predicted, train_raw_codes, train.n, spectral_dims)
  local ev = model.eigenvalues
  str.printf("  %-10s %10s %10s %10s %12s\n", "Dims", "MAE", "Pearson r", "Var ratio", "Eigenvalue")
  local bands = { { 0, math.min(7, spectral_dims - 1) } }
  if spectral_dims > 8 then bands[#bands + 1] = { 8, math.min(31, spectral_dims - 1) } end
  if spectral_dims > 32 then bands[#bands + 1] = { 32, math.min(63, spectral_dims - 1) } end
  if spectral_dims > 64 then bands[#bands + 1] = { 64, math.min(127, spectral_dims - 1) } end
  if spectral_dims > 128 then bands[#bands + 1] = { 128, math.min(255, spectral_dims - 1) } end
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
    str.printf("  [%3d-%3d] %10.6f %10.4f %10.4f %12.6f\n",
      lo, hi, s_mae / cnt, s_corr / cnt, s_vr / cnt, s_ev / cnt)
  end

  local function run_ridge(name, args)
    print("\n" .. name)
    local t0 = utc.time(true)
    args.label_offsets = train.label_csr.offsets
    args.label_neighbors = train.label_csr.neighbors
    args.n_labels = n_labels
    args.lambda = cfg.ridge.lambda
    args.propensity_a = cfg.ridge.propensity_a
    args.propensity_b = cfg.ridge.propensity_b
    args.expected_offsets = train.label_csr.offsets
    args.expected_neighbors = train.label_csr.neighbors
    args.search_trials = cfg.ridge.search_trials
    args.k = cfg.ridge.k
    args.each = function (ev0)
      str.printf("  [%d/%d] F1=%.4f best=%.4f\n",
        ev0.trial, ev0.trials, ev0.score, ev0.global_best_score or 0)
    end
    local _, p, m = optimize.ridge(args)
    local dt = utc.time(true) - t0
    local th = m.thresh
    local orc = m.oracle
    str.printf("  -> thresh=%.4f F1=%.4f (oracle=%.4f) lambda=%.4f a=%.2f b=%.2f (%.1fs)\n",
      th.threshold, th.macro_f1, orc.macro_f1, p.lambda, p.propensity_a, p.propensity_b, dt)
    return { name = name, params = p, oracle = orc, thresh = th, time = dt }
  end

  local results = {}

  results[#results + 1] = run_ridge("Ridge on spectral codes (ceiling)", {
    codes = train_raw_codes, n_samples = train.n, n_dims = spectral_dims,
  })
  collectgarbage("collect")

  results[#results + 1] = run_ridge("Ridge on predicted codes (pipeline)", {
    codes = train_predicted, n_samples = train.n, n_dims = spectral_dims,
  })
  collectgarbage("collect")

  print("\n" .. string.rep("=", 90))
  print("COMPARISON")
  print(string.rep("=", 90))
  str.printf("  %-40s %8s %8s %8s %8s %8s %6s\n",
    "Approach", "micro F1", "macro F1", "orc miF1", "orc maF1", "thresh", "Time")
  str.printf("  %-40s %8s %8s %8s %8s %8s %6s\n",
    string.rep("-", 40), "--------", "--------", "--------", "--------", "--------", "-----")
  for _, r in ipairs(results) do
    str.printf("  %-40s %8.4f %8.4f %8.4f %8.4f %8.4f %5.0fs\n",
      r.name, r.thresh.micro_f1, r.thresh.macro_f1,
      r.oracle.micro_f1, r.oracle.macro_f1, r.thresh.threshold, r.time)
  end
  str.printf("\n  Total time: %.1fs\n", stopwatch())

end)
