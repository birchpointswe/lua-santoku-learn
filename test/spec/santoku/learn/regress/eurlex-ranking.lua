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
local quantizer = require("santoku.learn.quantizer")
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
    n_selected = 65536,
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
  regressor = {
    class_batch = nil,
    cost_beta = nil,
    features = 256, --{ def = 4096, min = 256, max = 4096, pow2 = true },
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
    search_subsample_samples = 0.05,
    search_subsample_targets = nil,
    final_patience = 2,
    final_batch = 40,
    final_iterations = 800,
  },
}

test("eurlex-embedding", function ()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  str.printf("  Train: %d  Labels: %d\n", train.n, n_labels)

  print("\nBuilding label CSR")
  local label_offsets, label_neighbors = train.solutions:bits_to_csr(train.n, n_labels)

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
    each = function (ev) util.spectral_log(ev) end,
  })
  local spectral_dims = model.dims
  local train_raw_codes = dvec.create():mtx_extend(model.raw_codes,
    model.ids:set_intersect(ivec.create(train.n):fill_indices()),
    model.ids, 0, spectral_dims, true)
  str.printf("  Spectral: %d dims\n", spectral_dims)

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
    train_raw_codes, train.n, n_tokens, spectral_dims, n_selected, nil, "sum")
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
    outputs = spectral_dims,
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
    targets = train_raw_codes,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    search_subsample_samples = cfg.regressor.search_subsample_samples,
    search_subsample_targets = cfg.regressor.search_subsample_targets,
    stratify_offsets = label_offsets,
    stratify_neighbors = label_neighbors,
    stratify_labels = n_labels,
    final_batch = cfg.regressor.final_batch,
    final_patience = cfg.regressor.final_patience,
    final_iterations = cfg.regressor.final_iterations,
    each = util.make_regressor_log(stopwatch),
  })

  print("\n" .. string.rep("=", 60))
  print("FINAL EVALUATION")
  print(string.rep("=", 60))

  print("\nPredicting embeddings (train)")
  local train_predicted = tm:regress(
    { tokens = train.tokens, n_samples = train.n }, train.n, true)

  local train_ids = ivec.create(train.n):fill_indices()

  local spectral_ranking_cont = eval.ranking_accuracy({
    raw_codes = train_raw_codes, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local pred_ranking_cont = eval.ranking_accuracy({
    raw_codes = train_predicted, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local n_half = math.floor(train.n / 2)
  local shuffled = ivec.create(train.n):fill_indices():shuffle()
  local spec_half = ivec.create():copy(shuffled, 0, n_half, 0)
  local pred_half = ivec.create():copy(shuffled, n_half, train.n, 0)
  local spec_rows = dvec.create():copy(train_raw_codes):mtx_select(nil, spec_half, spectral_dims)
  local pred_rows = dvec.create():copy(train_predicted):mtx_select(nil, pred_half, spectral_dims)
  local mixed_codes = dvec.create():copy(spec_rows):copy(pred_rows)
  local mixed_ids = ivec.create():copy(spec_half):copy(pred_half)

  local mixed_ranking_cont = eval.ranking_accuracy({
    raw_codes = mixed_codes, ids = mixed_ids,
    n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local spectral_itq = quantizer.create({
    mode = "itq", raw_codes = train_raw_codes, n_samples = train.n })
  local predicted_itq = quantizer.create({
    mode = "itq", raw_codes = train_predicted, n_samples = train.n })

  local spectral_bin = spectral_itq:encode(train_raw_codes)
  local pred_bin_sitq = spectral_itq:encode(train_predicted)
  local pred_bin_pitq = predicted_itq:encode(train_predicted)

  local spectral_ranking_bin = eval.ranking_accuracy({
    codes = spectral_bin, ids = train_ids, n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local pred_ranking_bin_sitq = eval.ranking_accuracy({
    codes = pred_bin_sitq, ids = train_ids, n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local pred_ranking_bin_pitq = eval.ranking_accuracy({
    codes = pred_bin_pitq, ids = train_ids, n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local joint_itq = quantizer.create({
    mode = "itq", raw_codes = mixed_codes, n_samples = mixed_ids:size() })
  local mixed_bin_jitq = joint_itq:encode(mixed_codes)

  local mixed_ranking_bin_jitq = eval.ranking_accuracy({
    codes = mixed_bin_jitq, ids = mixed_ids, n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  str.printf("\n  Spectral dims: %d\n", spectral_dims)
  str.printf("  %-50s %8.4f\n", "Spectral (cont):", spectral_ranking_cont.score)
  str.printf("  %-50s %8.4f\n", "Predicted (cont):", pred_ranking_cont.score)
  str.printf("  %-50s %8.4f\n", "Mixed (cont):", mixed_ranking_cont.score)
  str.printf("  %-50s %8.4f\n", "Spectral (bin):", spectral_ranking_bin.score)
  str.printf("  %-50s %8.4f\n", "Predicted (bin, spectral ITQ):", pred_ranking_bin_sitq.score)
  str.printf("  %-50s %8.4f\n", "Predicted (bin, predicted ITQ):", pred_ranking_bin_pitq.score)
  str.printf("  %-50s %8.4f\n", "Mixed (bin, joint ITQ):", mixed_ranking_bin_jitq.score)

  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
