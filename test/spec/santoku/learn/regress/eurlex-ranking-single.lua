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
  align = {
    lambda = { def = 1.0, min = 0.001, max = 1000, log = true },
    search_trials = 80,
  },
  regressor = {
    flat = true,
    flat_evict = false,
    flat_encoding = "hadamard",
    flat_skip = nil, --{ def = 0.9, min = 0.9, max = 1.0 },
    cost_beta = nil,
    features = 1024, --{ def = 4096, min = 256, max = 8192, pow2 = true },
    clauses = 4, --{ def = 4, min = 1, max = 16, int = true },
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

test("eurlex-embedding-single", function ()

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
  local wide_codes = dvec.create():mtx_extend(model.raw_codes,
    model.ids:set_intersect(ivec.create(train.n):fill_indices()),
    model.ids, 0, spectral_dims, true)
  str.printf("  Spectral: %d dims (full width)\n", spectral_dims)

  local train_codes = wide_codes
  local working_dims = spectral_dims

  local train_ids = ivec.create(train.n):fill_indices()

  print("\nPre-training baselines")
  local spectral_ranking_cont = eval.ranking_accuracy({
    raw_codes = train_codes, ids = train_ids,
    n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })
  local spectral_itq = quantizer.create({
    mode = "itq", raw_codes = train_codes, n_samples = train.n })
  local spectral_bin = spectral_itq:encode(train_codes)
  local spectral_ranking_bin = eval.ranking_accuracy({
    codes = spectral_bin, ids = train_ids, n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })
  str.printf("  %-50s %8.4f\n", "Spectral (cont):", spectral_ranking_cont.score)
  str.printf("  %-50s %8.4f\n", "Spectral (bin):", spectral_ranking_bin.score)

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
    train_codes, train.n, n_tokens, working_dims, n_selected, nil, "max")
  train.tokens:bits_select(union_ids, nil, n_tokens)
  class_offsets, class_feat_ids = csr.bits_select(class_offsets, class_feat_ids, union_ids)
  n_tokens = union_ids:size()
  str.printf("  %d features selected\n", n_tokens)

  print("\nBuilding CSC index")
  local csc_offsets, csc_indices = csr.to_csc(train.tokens, train.n, n_tokens)
  str.printf("  Tokens: %d  Samples: %d\n", n_tokens, train.n)

  print("\nTraining flat regressor")
  local tm_obj = optimize.regressor({
    flat = cfg.regressor.flat,
    flat_evict = cfg.regressor.flat_evict,
    flat_encoding = cfg.regressor.flat_encoding,
    flat_skip = cfg.regressor.flat_skip,
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
    tokens = train.tokens,
    csc_offsets = csc_offsets,
    csc_indices = csc_indices,
    absorb_ranking = class_feat_ids,
    absorb_ranking_offsets = class_offsets,
    absorb_ranking_global = ivec.create(n_tokens):fill_indices(),
    targets = train_codes,
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

  print("\n" .. string.rep("=", 60))
  print("FINAL EVALUATION")
  print(string.rep("=", 60))

  print("\nPredicting embeddings (train)")
  local train_predicted = tm_obj:regress(
    { tokens = train.tokens, n_samples = train.n }, train.n, true)

  print("\nFitting ridge alignment")
  local align_model = optimize.ridge({
    n_samples = train.n, n_dims = working_dims, n_targets = working_dims,
    codes = train_predicted, targets = train_codes,
    lambda = cfg.align.lambda,
    search_trials = cfg.align.search_trials,
    score_fn = function (r, data)
      local transformed = r:transform(data.codes, data.n_samples)
      local ids = ivec.create(data.n_samples):fill_indices()
      local ra = eval.ranking_accuracy({
        raw_codes = transformed, ids = ids, n_dims = working_dims,
        eval_ids = eval_uids, eval_offsets = eval_off,
        eval_neighbors = eval_nbr, eval_weights = eval_w,
      })
      return ra.score, { score = ra.score }
    end,
    each = util.make_ridge_log(),
  })
  local train_aligned = align_model:transform(train_predicted, train.n)

  local pred_ranking_cont = eval.ranking_accuracy({
    raw_codes = train_predicted, ids = train_ids,
    n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local aligned_ranking_cont = eval.ranking_accuracy({
    raw_codes = train_aligned, ids = train_ids,
    n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local predicted_itq = quantizer.create({
    mode = "itq", raw_codes = train_predicted, n_samples = train.n })

  local pred_bin_sitq = spectral_itq:encode(train_predicted)
  local pred_bin_pitq = predicted_itq:encode(train_predicted)

  local pred_ranking_bin_sitq = eval.ranking_accuracy({
    codes = pred_bin_sitq, ids = train_ids, n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local pred_ranking_bin_pitq = eval.ranking_accuracy({
    codes = pred_bin_pitq, ids = train_ids, n_dims = working_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  str.printf("\n  Spectral dims: %d\n", working_dims)
  str.printf("  %-50s %8.4f\n", "Spectral (cont):", spectral_ranking_cont.score)
  str.printf("  %-50s %8.4f\n", "Predicted (cont):", pred_ranking_cont.score)
  str.printf("  %-50s %8.4f\n", "Aligned (cont):", aligned_ranking_cont.score)
  str.printf("  %-50s %8.4f\n", "Spectral (bin):", spectral_ranking_bin.score)
  str.printf("  %-50s %8.4f\n", "Predicted (bin, spectral ITQ):", pred_ranking_bin_sitq.score)
  str.printf("  %-50s %8.4f\n", "Predicted (bin, predicted ITQ):", pred_ranking_bin_pitq.score)

  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
