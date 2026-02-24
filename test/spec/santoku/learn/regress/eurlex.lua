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
    ngrams = 2,
    cgrams_min = 3,
    cgrams_max = 5,
    cgrams_cross = true,
    skips = 1,
  },
  feature_selection = {
    n_selected = 65536,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 256,
    decay = 0,
  },
  regressor = {
    class_batch = nil,
    cost_beta = nil,
    features = 8192, --{ def = 1024, min = 256, max = 2048, pow2 = true },
    clauses = 8, --{ def = 4, min = 2, max = 6, int = true },
    absorb_threshold = { def = 9 },
    absorb_maximum_fraction = { def = 0.029 },
    absorb_insert_offset = { def = 49 },
    absorb_ranking_fraction = { def = 0.125 },
    absorb_ranking_limit = { def = 0.125 },
    clause_maximum_fraction = { def = 0.03 },
    clause_tolerance_fraction = { def = 0.57 },
    target_fraction = { def = 0.10 },
    specificity_fraction = { def = 0.00095 },
    alpha_tolerance = { def = -0.3, min = -3, max = 3 },
    alpha_maximum = { def = 0.7, min = -3, max = 3 },
    alpha_target = { def = 1.2, min = -3, max = 3 },
    alpha_specificity = { def = 2.1, min = -3, max = 3 },
    search_trials = 200,
    search_iterations = 20,
    search_subsample_samples = 0.1,
    final_patience = 2,
    final_batch = 20,
    final_iterations = 300,
  },
  ridge = {
    lambda = { def = 0.5, min = 0.01, max = 10 },
    propensity_a = { def = 0.55, min = 0.1, max = 2.0 },
    propensity_b = { def = 1.5, min = 0.1, max = 5.0 },
    alpha_lambda = { def = 0, min = -3, max = 3 },
    k = 32,
    n_used_dims = { def = 256 },
    cost_beta = nil,
    search_trials = 200,
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

  print("\nBuilding label CSR")
  local train_label_offsets, train_label_neighbors = train.solutions:bits_to_csr(train.n, n_labels)
  train.label_csr = { offsets = train_label_offsets, neighbors = train_label_neighbors }

  train.ids = ivec.create(train.n):fill_indices()

  print("\nBuilding label index")
  local label_sols = ivec.create():copy(train.solutions)
  local label_idf_ids, label_idf_scores = label_sols:bits_top_idf(train.n, n_labels)
  label_sols:bits_select(label_idf_ids, nil, n_labels)
  local n_label_feats = label_idf_ids:size()
  local label_index = inv.create({ features = label_idf_scores:size() })
  label_index:add(label_sols, train.ids)
  str.printf("  Label index: %d IDF feats, %d docs, decay=%.1f\n",
    n_label_feats, train.n, cfg.nystrom.decay)

  print("\nSpectral embedding")
  local model = optimize.spectral({
    index = label_index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    each = function(ev) util.spectral_log(ev) end,
  })

  local working_dims = model.dims
  local train_codes = dvec.create():mtx_extend(model.raw_codes,
    model.ids:set_intersect(ivec.create(train.n):fill_indices()),
    model.ids, 0, working_dims, true)
  local eigenvalues = model.eigenvalues
  str.printf("  Spectral: %d dims\n", working_dims)

  local results = {}

  local function run_ridge(name, args)
    print("\n" .. name)
    local t0 = utc.time(true)
    args.label_offsets = train.label_csr.offsets
    args.label_neighbors = train.label_csr.neighbors
    args.n_labels = n_labels
    args.lambda = cfg.ridge.lambda
    args.propensity_a = cfg.ridge.propensity_a
    args.propensity_b = cfg.ridge.propensity_b
    args.alpha_lambda = args.output_weights and cfg.ridge.alpha_lambda or nil
    args.expected_offsets = train.label_csr.offsets
    args.expected_neighbors = train.label_csr.neighbors
    args.search_trials = cfg.ridge.search_trials
    args.search_subsample = cfg.ridge.search_subsample
    args.stratify_offsets = train.label_csr.offsets
    args.stratify_neighbors = train.label_csr.neighbors
    args.stratify_labels = n_labels
    args.k = cfg.ridge.k
    args.each = function (ev0)
      local p = ev0.params
      if ev0.event == "trial" then
        local marker = ev0.is_new_best and " ++" or ""
        local dims_str = p.n_used_dims and string.format(" d=%d", p.n_used_dims) or ""
        local al_str = p.alpha_lambda and string.format(" al=%+.2f", p.alpha_lambda) or ""
        str.printf("  [%s %d/%d] F1=%.4f (best=%.4f%s) lam=%.2f a=%.2f b=%.2f%s%s\n",
          ev0.phase, ev0.trial, ev0.trials, ev0.score, ev0.global_best_score, marker,
          p.lambda, p.propensity_a, p.propensity_b, dims_str, al_str)
      end
    end
    local ridge_obj, p, m = optimize.ridge(args)
    local dt = utc.time(true) - t0
    local rth = m.thresh
    local rorc = m.oracle
    local dims_str = p.n_used_dims and string.format(" d=%d", p.n_used_dims) or ""
    local al_str = p.alpha_lambda and string.format(" al=%+.2f", p.alpha_lambda) or ""
    str.printf("  -> thresh=%.4f F1=%.4f (oracle=%.4f) lam=%.4f a=%.2f b=%.2f%s%s (%.1fs)\n",
      rth.threshold, rth.macro_f1, rorc.macro_f1, p.lambda, p.propensity_a, p.propensity_b, dims_str, al_str, dt)
    return { name = name, params = p, oracle = rorc, thresh = rth, time = dt, ridge = ridge_obj }
  end

  local ceiling = run_ridge("Ridge on spectral codes (ceiling)", {
    codes = train_codes, n_samples = train.n, n_dims = working_dims,
    output_weights = eigenvalues,
  })
  results[#results + 1] = ceiling

  print("\nCeiling per-dim stats")
  local ceiling_dw = ceiling.ridge:dim_weights()
  str.printf("  %4s %10s %10s\n", "Dim", "Eigenvalue", "Ridge |W|")
  for d = 0, working_dims - 1 do
    local ev = d < eigenvalues:size() and eigenvalues:get(d) or 0
    str.printf("  %4d %10.6f %10.6f\n", d, ev, ceiling_dw:get(d))
  end

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens)

  print("\nFeature selection (F-score)")
  local n_selected = cfg.feature_selection.n_selected
  local union_ids, _, class_offsets, class_feat_ids = train.tokens:bits_top_reg_f(
    train_codes, train.n, n_tokens, working_dims, n_selected, nil, "sum")
  train.tokens:bits_select(union_ids, nil, n_tokens)
  class_offsets, class_feat_ids = csr.bits_select(class_offsets, class_feat_ids, union_ids)
  n_tokens = union_ids:size()
  str.printf("  %d features selected\n", n_tokens)

  print("\nBuilding CSC index")
  local csc_offsets, csc_indices = csr.to_csc(train.tokens, train.n, n_tokens)
  str.printf("  Tokens: %d  Samples: %d\n", n_tokens, train.n)

  print("\nTraining regressor")
  local tm = optimize.regressor({
    class_batch = cfg.regressor.class_batch,
    cost_beta = cfg.regressor.cost_beta,
    outputs = working_dims,
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
    alpha_tolerance = cfg.regressor.alpha_tolerance,
    alpha_maximum = cfg.regressor.alpha_maximum,
    alpha_target = cfg.regressor.alpha_target,
    alpha_specificity = cfg.regressor.alpha_specificity,
    output_weights = eigenvalues,
    tokens = train.tokens,
    csc_offsets = csc_offsets,
    csc_indices = csc_indices,
    absorb_ranking = class_feat_ids,
    absorb_ranking_offsets = class_offsets,
    targets = train_codes,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    search_subsample_samples = cfg.regressor.search_subsample_samples,
    search_subsample_targets = cfg.regressor.search_subsample_targets,
    stratify_offsets = train.label_csr.offsets,
    stratify_neighbors = train.label_csr.neighbors,
    stratify_labels = n_labels,
    final_batch = cfg.regressor.final_batch,
    final_patience = cfg.regressor.final_patience,
    final_iterations = cfg.regressor.final_iterations,
    each = util.make_regressor_log(stopwatch),
  })

  print("\nPredicting embeddings (train)")
  local train_predicted = tm:regress(
    { tokens = train.tokens, n_samples = train.n }, train.n, true)

  local pred_stats = eval.regression_accuracy(train_predicted, train_codes)
  str.printf("  Regression MAE: %.6f\n", pred_stats.mean)

  local pd = eval.regression_per_dim(train_predicted, train_codes, train.n, working_dims)
  local corr_order = pd.corr:rdesc(pd.corr:size())
  local reordered_predicted = dvec.create()
  train_predicted:mtx_select(corr_order, nil, working_dims, reordered_predicted)

  local reordered_corr = dvec.create():copy(pd.corr, corr_order)
  local pipeline = run_ridge("Ridge on predicted (pipeline)", {
    codes = reordered_predicted, n_samples = train.n, n_dims = working_dims,
    n_used_dims = cfg.ridge.n_used_dims, cost_beta = cfg.ridge.cost_beta,
    output_weights = reordered_corr,
  })
  results[#results + 1] = pipeline

  print("\nPer-dim regression stats (ordered by Pearson r)")
  local pipeline_dw = pipeline.ridge:dim_weights()
  local pipeline_nd = pipeline.ridge:n_dims()
  str.printf("  %4s %4s %10s %10s %10s %10s %10s\n",
    "Rank", "Dim", "Pearson r", "MAE", "Var ratio", "Eigenvalue", "Ridge |W|")
  for rank = 0, working_dims - 1 do
    local d = corr_order:get(rank)
    local ev = d < eigenvalues:size() and eigenvalues:get(d) or 0
    local rw = rank < pipeline_nd and pipeline_dw:get(rank) or 0
    str.printf("  %4d %4d %10.4f %10.6f %10.4f %10.6f %10.6f\n",
      rank, d, pd.corr:get(d), pd.mae:get(d), pd.var_ratio:get(d), ev, rw)
  end

  if pipeline.params.n_used_dims then
    local keep = pipeline.params.n_used_dims
    str.printf("  Keeping %d/%d dims (corr %.4f..%.4f)\n",
      keep, working_dims,
      pd.corr:get(corr_order:get(0)),
      pd.corr:get(corr_order:get(keep - 1)))
    corr_order:setn(keep)
    tm:restrict(corr_order)
    eigenvalues = dvec.create():copy(eigenvalues, corr_order)
    local rp = tm:regress({ tokens = train.tokens, n_samples = train.n }, train.n, true)
    local rr = dvec.create()
    train_codes:mtx_select(corr_order, nil, working_dims, rr)
    local rs = eval.regression_accuracy(rp, rr)
    str.printf("  Restricted TM to %d dims, MAE: %.6f\n", keep, rs.mean)
  end

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
