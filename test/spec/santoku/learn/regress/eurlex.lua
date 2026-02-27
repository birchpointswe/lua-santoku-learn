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
    cgrams_cross = false,
    skips = 1,
  },
  feature_selection = {
    n_selected = 8192,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 1024,
    decay = 0,
  },
  eval = {
    knn = 16,
    random_pairs = 16,
  },
  augmented = {
    knn = 32,
  },
  regressor = {
    class_batch = nil,
    cost_beta = nil,
    features = { def = 4096, min = 256, max = 4096, pow2 = true },
    clauses = { def = 4, min = 1, max = 8, int = true },
    clause_maximum_fraction = { def = 0.03 },
    clause_tolerance_fraction = { def = 0.57 },
    specificity_fraction = { def = 0.00095 },
    absorb_threshold = { def = 9 },
    absorb_maximum_fraction = { def = 0.029 },
    absorb_insert_offset = { def = 49 },
    absorb_ranking_fraction = { def = 0.125 },
    absorb_ranking_limit = { def = 0.125 },
    target_fraction = { def = 0.10 },
    alpha_tolerance = { def = -0.3, min = -3, max = 3 },
    alpha_maximum = { def = 0.7, min = -3, max = 3 },
    alpha_target = { def = 1.2, min = -3, max = 3 },
    alpha_specificity = { def = 2.1, min = -3, max = 3 },
    search_trials = 200,
    search_iterations = 20,
    search_subsample = 0.1,
    final_patience = 2,
    final_batch = 40,
    final_iterations = 800,
  },
  ridge = {
    lambda = { def = 0.5, min = 0.01, max = 10 },
    propensity_a = { def = 0.55, min = 0.1, max = 2.0 },
    propensity_b = { def = 1.5, min = 0.1, max = 5.0 },
    k = 32,
    search_trials = 200,
    search_subsample = 0.1,
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
  local label_index = inv.create({ features = label_idf_scores })
  label_index:add(label_sols, train.ids)
  str.printf("  Label index: %d IDF feats, %d docs, decay=%.1f\n",
    n_label_feats, train.n, cfg.nystrom.decay)

  print("\nBuilding eval adjacency")
  local eval_uids, inv_hoods = label_index:neighborhoods(cfg.eval.knn, cfg.nystrom.decay, -1)
  local eval_off, eval_nbr, eval_w = inv_hoods:to_csr(eval_uids)
  local rp_off, rp_nbr, rp_w = csr.random_pairs(eval_uids, cfg.eval.random_pairs)
  csr.weight_from_index(eval_uids, rp_off, rp_nbr, rp_w, label_index, cfg.nystrom.decay)
  csr.merge(eval_off, eval_nbr, eval_w, rp_off, rp_nbr, rp_w)
  csr.symmetrize(eval_off, eval_nbr, eval_w, eval_uids:size())
  str.printf("  %d nodes, %d edges\n", eval_uids:size(), eval_nbr:size())

  print("\nSpectral embedding")
  local model = optimize.spectral({
    index = label_index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    each = util.make_spectral_log(stopwatch),
  })

  local spectral_dims = model.dims
  local wide_codes = dvec.create():mtx_extend(model.raw_codes,
    model.ids:set_intersect(ivec.create(train.n):fill_indices()),
    model.ids, 0, spectral_dims, true)
  str.printf("  Spectral: %d dims (full width)\n", spectral_dims)

  local train_codes = wide_codes
  local working_dims = spectral_dims

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
    args.expected_offsets = train.label_csr.offsets
    args.expected_neighbors = train.label_csr.neighbors
    args.search_trials = cfg.ridge.search_trials
    args.search_subsample = cfg.ridge.search_subsample
    args.stratify_offsets = train.label_csr.offsets
    args.stratify_neighbors = train.label_csr.neighbors
    args.stratify_labels = n_labels
    args.k = cfg.ridge.k
    args.each = util.make_ridge_log(stopwatch)
    local ridge_obj, p, m = optimize.ridge(args)
    local dt = utc.time(true) - t0
    local rth = m.thresh
    local rorc = m.oracle
    str.printf("  -> thresh=%.4f F1=%.4f (oracle=%.4f) lam=%.4f a=%.2f b=%.2f (%.1fs)\n",
      rth.threshold, rth.macro_f1, rorc.macro_f1, p.lambda, p.propensity_a, p.propensity_b, dt)
    return { name = name, params = p, oracle = rorc, thresh = rth, time = dt, ridge = ridge_obj }
  end

  local ceiling = run_ridge("Ridge on spectral codes (ceiling)", {
    codes = wide_codes, n_samples = train.n, n_dims = spectral_dims,
  })
  results[#results + 1] = ceiling

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
    train_codes, train.n, n_tokens, working_dims, n_selected)
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
    output_weights = model.eigenvalues,
    tokens = train.tokens,
    csc_offsets = csc_offsets,
    csc_indices = csc_indices,
    absorb_ranking = class_feat_ids,
    absorb_ranking_offsets = class_offsets,
    targets = train_codes,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    search_subsample = cfg.regressor.search_subsample,
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

  local train_ids = ivec.create(train.n):fill_indices()

  local spectral_ranking = eval.ranking_accuracy({
    raw_codes = train_codes, ids = train_ids, n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })
  local predicted_ranking = eval.ranking_accuracy({
    raw_codes = train_predicted, ids = train_ids, n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local spectral_itq = quantizer.create({
    mode = "itq", raw_codes = train_codes, n_samples = train.n })
  local spectral_bin = spectral_itq:encode(train_codes)
  local pred_bin_sitq = spectral_itq:encode(train_predicted)

  local spectral_ranking_bin = eval.ranking_accuracy({
    codes = spectral_bin, ids = train_ids, n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })
  local pred_ranking_bin = eval.ranking_accuracy({
    codes = pred_bin_sitq, ids = train_ids, n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  str.printf("\n  Ranking (spectral, cont):  %.4f\n", spectral_ranking.score)
  str.printf("  Ranking (predicted, cont): %.4f\n", predicted_ranking.score)
  str.printf("  Ranking (spectral, bin):   %.4f\n", spectral_ranking_bin.score)
  str.printf("  Ranking (predicted, bin):  %.4f\n", pred_ranking_bin.score)

  local pipeline = run_ridge("Ridge on predicted (pipeline)", {
    codes = train_predicted, n_samples = train.n, n_dims = working_dims,
  })
  results[#results + 1] = pipeline

  print("\nBuilding augmented codes")
  local codes_ann = ann.create({ features = working_dims })
  codes_ann:add(spectral_bin, train_ids)
  local hood_ids, ann_hoods = codes_ann:neighborhoods_by_vecs(pred_bin_sitq, cfg.augmented.knn)
  local nn_off, nn_nbr, nn_w = ann_hoods:to_csr(working_dims)

  local pseudo_bits = csr.label_union(nn_off, nn_nbr, hood_ids,
    train_label_offsets, train_label_neighbors, n_labels)
  pseudo_bits:bits_select(label_idf_ids, nil, n_labels)
  local pseudo_nystrom = model.encoder:encode(pseudo_bits, train.n, n_label_feats)
  local augmented_codes = dvec.create():copy(train_predicted)
  augmented_codes:mtx_extend(pseudo_nystrom, working_dims, working_dims)

  local augmented = run_ridge("Ridge on augmented (pseudo doc)", {
    codes = augmented_codes, n_samples = train.n, n_dims = 2 * working_dims,
  })
  results[#results + 1] = augmented

  local pseudo_avg = csr.neighbor_average(nn_off, nn_nbr, nn_w, hood_ids, train_codes, train_ids, working_dims)
  local aug_avg_codes = dvec.create():copy(train_predicted)
  aug_avg_codes:mtx_extend(pseudo_avg, working_dims, working_dims)

  local aug_cavg = run_ridge("Ridge on augmented (code avg)", {
    codes = aug_avg_codes, n_samples = train.n, n_dims = 2 * working_dims,
  })
  results[#results + 1] = aug_cavg

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
