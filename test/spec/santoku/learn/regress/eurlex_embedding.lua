local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.learn.evaluator")
local inv = require("santoku.learn.inv")
local ivec = require("santoku.ivec")
local normalizer = require("santoku.learn.normalizer")
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
    n_bns = 8192,
    n_selected = 65536,
  },
  graph = {
    decay = 2,
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
    features = 2048,
    absorb_interval = 1,
    absorb_threshold = { def = 0, min = 0, max = 256, int = true },
    absorb_maximum = { def = 256, min = 1, max = 256, int = true },
    absorb_insert_offset = { def = 1, min = 1, max = 256, int = true },
    clauses = { def = 1, min = 1, max = 4, int = true },
    clause_maximum = { def = 16, min = 8, max = 256, int = true },
    clause_tolerance_fraction = { def = 0.5, min = 0.01, max = 1.0 },
    target_fraction = { def = 0.25, min = 0.01, max = 2.0 },
    specificity = { def = 800, min = 2, max = 2000 },
    search_trials = 200,
    search_iterations = 20,
    search_subsample_samples = 0.2,
    search_subsample_targets = 8,
    final_patience = 2,
    final_batch = 40,
    final_iterations = 200,
  },
}

test("eurlex-embedding", function ()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  str.printf("  Train: %d\n", train.n)

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens)

  local n_classes = train.n_labels
  print("\nBNS feature selection")
  train.solutions:add_scaled(n_classes)
  local bns_ids, bns_scores = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_classes,
    cfg.feature_selection.n_bns, nil, "max")
  train.solutions:add_scaled(-n_classes)
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  local n_features = bns_ids:size()
  n_tokens = n_features
  str.printf("  %d features selected\n", n_features)

  print("\nBuilding doc-doc index (labels only)")
  train.ids = ivec.create(train.n):fill_indices()
  local label_sols = ivec.create():copy(train.solutions)
  local label_idf_ids, label_idf_scores = label_sols:bits_top_df(train.n, n_classes)
  label_sols:bits_select(label_idf_ids, nil, n_classes)
  local n_label_feats = label_idf_ids:size()
  local index = inv.create({ features = label_idf_scores })
  index:add(label_sols, train.ids)
  str.printf("  %d docs indexed, %d IDF-weighted label features, decay=%.1f\n",
    train.n, n_label_feats, cfg.graph.decay)

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

  print("\nBuilding train text-only evaluation index")
  local tr_ids = ivec.create(train.n):fill_indices()
  local tr_idx = inv.create({ features = bns_scores })
  tr_idx:add(train_bns_tokens, tr_ids)
  local tr_uids, tr_hoods = tr_idx:neighborhoods(cfg.eval.knn, 0, -1)
  local tr_off, tr_nbr, tr_w = tr_hoods:to_csr(tr_uids)
  local rp_off, rp_nbr, rp_w = csr.random_pairs(tr_uids, cfg.eval.random_pairs)
  csr.weight_from_index(tr_uids, rp_off, rp_nbr, rp_w, tr_idx, 0, -1)
  csr.merge(tr_off, tr_nbr, tr_w, rp_off, rp_nbr, rp_w)
  csr.symmetrize(tr_off, tr_nbr, tr_w, tr_uids:size())
  str.printf("  Train: %d nodes, %d edges\n", tr_uids:size(), tr_nbr:size())

  print("\nPredicting embeddings (train)")
  local train_predicted_raw = tm:regress(
    { tokens = train.tokens, n_samples = train.n }, train.n, true)
  local retrieval_norm = normalizer.create({
    source = train_predicted_raw,
    target = train_raw_codes,
    n_samples_source = train.n,
    n_samples_target = train.n,
    n_dims = spectral_dims,
  })
  local train_predicted = retrieval_norm:encode(train_predicted_raw)

  local train_ids = ivec.create(train.n):fill_indices()

  local spectral_vs_text = eval.ranking_accuracy({
    raw_codes = train_raw_codes, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = tr_uids, eval_offsets = tr_off,
    eval_neighbors = tr_nbr, eval_weights = tr_w,
  })

  local spectral_vs_labels = eval.ranking_accuracy({
    raw_codes = train_raw_codes, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local pred_raw_vs_labels = eval.ranking_accuracy({
    raw_codes = train_predicted_raw, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local pred_norm_vs_labels = eval.ranking_accuracy({
    raw_codes = train_predicted, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = eval_uids, eval_offsets = eval_off,
    eval_neighbors = eval_nbr, eval_weights = eval_w,
  })

  local pred_norm_vs_text = eval.ranking_accuracy({
    raw_codes = train_predicted, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = tr_uids, eval_offsets = tr_off,
    eval_neighbors = tr_nbr, eval_weights = tr_w,
  })

  str.printf("\n  Spectral dims: %d\n", spectral_dims)
  str.printf("  %-35s %8.4f\n", "Spectral vs labels:", spectral_vs_labels.score)
  str.printf("  %-35s %8.4f\n", "Spectral vs text:", spectral_vs_text.score)
  str.printf("  %-35s %8.4f\n", "Predicted (raw) vs labels:", pred_raw_vs_labels.score)
  str.printf("  %-35s %8.4f\n", "Predicted (norm) vs labels:", pred_norm_vs_labels.score)
  str.printf("  %-35s %8.4f\n", "Predicted (norm) vs text:", pred_norm_vs_text.score)

  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
