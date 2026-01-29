local _ -- luacheck: ignore
local ann = require("santoku.tsetlin.ann")
local ds = require("santoku.tsetlin.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.tsetlin.evaluator")
local graph = require("santoku.tsetlin.graph")
local inv = require("santoku.tsetlin.inv")
local itq = require("santoku.tsetlin.itq")
local ivec = require("santoku.ivec")
local optimize = require("santoku.tsetlin.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.tsetlin.util")
local utc = require("santoku.utc")

local cfg = {
  exit_after = nil, --"spectral",
  data = {
    max = nil,
  },
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
    min_df = 2,
    max_df = 0.5,
  },
  regressor = {
    grouped = true,
    balanced = false,
    grouped_vocab = 4096,
    clauses = 8,
    clause_tolerance = { def = 25, min = 8, max = 128, int = true },
    clause_maximum = { def = 27, min = 16, max = 128, int = true },
    target = { def = 55, min = 4, max = 64, int = true },
    specificity = { def = 135, min = 50, max = 2000 },
    include_bits = { def = 3, min = 1, max = 6, int = true },
    search_frac = 0.1,
    search_patience = 4,
    search_rounds = 6,
    search_trials = 20,
    search_iterations = 40,
    final_patience = 40,
    final_iterations = 400,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 32,
    cmp = "cosine",
    combine = "exponential",
    decay = 2.0,
    rounds = 0,
    samples = 20,
  },
  eval = {
    knn = 16,
    pairs = 16,
    ranking = "ndcg",
    cmp = "cosine",
  },
  verbose = true,
}

test("eurlex-docs", function()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local train_lc = train.label_counts:to_dvec()
  train_lc:asc()
  str.printf("  Train: %d  Dev: %d  Test: %d  Labels: %d  Labels/doc: %.1f median\n",
    train.n, dev.n, test_set.n, n_labels, train_lc:get(train_lc:size() / 2))

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil -- luacheck: ignore
  dev.problems = nil -- luacheck: ignore
  test_set.problems = nil -- luacheck: ignore

  print("\nCreating IDs")
  train.ids = ivec.create(train.n):fill_indices()
  dev.ids = ivec.create(dev.n):fill_indices():add(train.n)
  test_set.ids = ivec.create(test_set.n):fill_indices():add(train.n + dev.n)

  print("\nBuilding label CSR for lookups")
  local train_label_offsets, train_label_neighbors = train.solutions:bits_to_csr(train.n, n_labels)
  train.label_csr = { offsets = train_label_offsets, neighbors = train_label_neighbors }
  local dev_label_offsets, dev_label_neighbors = dev.solutions:bits_to_csr(dev.n, n_labels)
  dev.label_csr = { offsets = dev_label_offsets, neighbors = dev_label_neighbors }
  local test_label_offsets, test_label_neighbors = test_set.solutions:bits_to_csr(test_set.n, n_labels)
  test_set.label_csr = { offsets = test_label_offsets, neighbors = test_label_neighbors }

  print("\nFeature selection (DF filter)")
  local min_df = cfg.feature_selection.min_df or 2
  local max_df = cfg.feature_selection.max_df or 0.5
  local df_ids = train.tokens:bits_top_df(train.n, n_tokens, nil, -min_df, max_df)
  str.printf("  DF filter (min=%d, max=%.0f%%): %d -> %d\n", min_df, max_df * 100, n_tokens, df_ids:size())
  train.tokens:bits_select(df_ids, nil, n_tokens)
  dev.tokens:bits_select(df_ids, nil, n_tokens)
  test_set.tokens:bits_select(df_ids, nil, n_tokens)
  n_tokens = df_ids:size()
  df_ids = nil -- luacheck: ignore

  print("\nKeeping full token space for regressor")
  local n_full_tokens = n_tokens
  train.full_tokens = ivec.create():copy(train.tokens)
  dev.full_tokens = ivec.create():copy(dev.tokens)
  test_set.full_tokens = ivec.create():copy(test_set.tokens)

  print("\nBuilding graph_index (docs only, two-rank: labels + tokens, unweighted)")
  local n_graph_features = n_labels + n_tokens
  local graph_features = ivec.create():copy(train.solutions):bits_extend(train.tokens, n_labels, n_tokens)
  local graph_ranks = ivec.create(n_graph_features):fill(0, 0, n_labels):fill(1, n_labels, n_graph_features)
  train.graph_index = inv.create({
    features = n_graph_features,
    ranks = graph_ranks,
    n_ranks = 2,
  })
  train.graph_index:add(graph_features, train.ids)
  graph_features = nil -- luacheck: ignore
  str.printf("  Docs: %d  Features: %d (labels=%d, tokens=%d)\n",
    train.n, n_graph_features, n_labels, n_tokens)

  print("\nBuilding label ANN for evaluation")
  local train_labels_cvec = train.solutions:bits_to_cvec(train.n, n_labels)
  train.label_ann = ann.create({ features = n_labels })
  train.label_ann:add(train_labels_cvec, train.ids)

  print("\nBuilding evaluation adjacency (knn from label ANN, weights from graph_index)")
  local train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights =
    graph.adjacency({
      knn_index = train.label_ann,
      knn = cfg.eval.knn,
      knn_cache = cfg.eval.knn,
      weight_index = train.graph_index,
      weight_cmp = cfg.nystrom.cmp,
      weight_combine = cfg.nystrom.combine,
      weight_decay = cfg.nystrom.decay,
      bridge = "none",
    })
  math.randomseed(12345)
  local spot_check_ids = {}
  for i = 1, 5 do
    spot_check_ids[i] = train_eval_ids:get(math.random(train_eval_ids:size()) - 1)
  end
  if cfg.verbose then
    util.spot_check_adjacency(train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights, "train eval")
    util.spot_check_neighbors_with_labels(train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights,
      train.label_csr, 0, "eval adj (knn+random)", spot_check_ids, 10)
  end

  print("\nRunning spectral embedding (Nystrom)")
  local model = optimize.spectral({
    index = train.graph_index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    cmp = cfg.nystrom.cmp,
    combine = cfg.nystrom.combine,
    decay = cfg.nystrom.decay,
    rounds = cfg.nystrom.rounds,
    samples = cfg.nystrom.samples,
    expected = {
      ids = train_eval_ids,
      offsets = train_eval_offsets,
      neighbors = train_eval_neighbors,
      weights = train_eval_weights,
    },
    eval = { ranking = cfg.eval.ranking },
    each = cfg.verbose and util.spectral_log or nil,
  })
  train.dims = model.dims
  local train_raw_codes = model.raw_codes
  str.printf("  Spectral dims: %d, samples: %d\n", train.dims, train.n)

  print("\nEvaluating spectral codes against eval adjacency")
  local spectral_eval_stats = eval.ranking_accuracy({
    raw_codes = train_raw_codes,
    ids = model.ids,
    eval_ids = train_eval_ids,
    eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors,
    eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Spectral codes ranking: %.4f\n", spectral_eval_stats.score)

  if cfg.exit_after == "spectral" then
    return
  end

  print("\nRegression feature selection for regressor (from full token space)")
  local n_regressor_features, make_regressor_sentences

  if cfg.regressor.grouped then
    local grouped_vocab = cfg.regressor.grouped_vocab
    local class_offsets, class_feat_ids, class_scores = train.full_tokens:bits_top_reg_f_grouped(
      train_raw_codes, train.n, n_full_tokens, train.dims, grouped_vocab)
    str.printf("  Regression features (grouped): %d total (requested %d per class x %d classes)\n",
      class_feat_ids:size(), grouped_vocab, train.dims)
    str.printf("  F-scores: min=%.2f max=%.2f mean=%.2f\n",
      class_scores:min(), class_scores:max(), class_scores:sum() / class_scores:size())
    make_regressor_sentences = function(full_tokens, n_samples)
      local cvec, max_k = full_tokens:bits_to_cvec(n_samples, n_full_tokens, class_offsets, class_feat_ids, true)
      return cvec, max_k
    end
  else
    local max_regressor_features = cfg.regressor.grouped_vocab * train.dims
    local reg_ids, reg_scores = train.full_tokens:bits_top_reg_f(
      train_raw_codes, train.n, n_full_tokens, max_regressor_features)
    n_regressor_features = reg_ids:size()
    str.printf("  Regression features (non-grouped): %d\n", n_regressor_features)
    str.printf("  F-scores: min=%.2f max=%.2f mean=%.2f\n",
      reg_scores:min(), reg_scores:max(), reg_scores:sum() / reg_scores:size())
    make_regressor_sentences = function(full_tokens, n_samples)
      return ivec.create():copy(full_tokens):bits_select(reg_ids, nil, n_full_tokens)
        :bits_to_cvec(n_samples, n_regressor_features, true)
    end
  end

  local search_n = math.floor(train.n * cfg.regressor.search_frac)
  local search_ids = ivec.create(train.n)
  search_ids:fill_indices()
  search_ids:shuffle()
  search_ids:setn(search_n)
  local search_full_tokens = ivec.create()
  train.full_tokens:bits_select(nil, search_ids, n_full_tokens, search_full_tokens)
  local search_problems, max_k = make_regressor_sentences(search_full_tokens, search_n)
  local search_targets = dvec.create()
  train_raw_codes:mtx_select(nil, search_ids, train.dims, search_targets)
  str.printf("\nSearch subset: %d / %d samples (%.0f%%)\n", search_n, train.n, cfg.regressor.search_frac * 100)

  local train_regressor_sentences
  train_regressor_sentences, max_k = make_regressor_sentences(train.full_tokens, train.n)
  if cfg.regressor.grouped then
    n_regressor_features = max_k
    str.printf("  Regressor features (padded max per class): %d\n", n_regressor_features)
  end

  print("\nTraining regressor (binary mode)")
  train.regressor = optimize.regressor({
    outputs = train.dims,
    targets = train_raw_codes,
    samples = train.n,
    problems = train_regressor_sentences,
    features = n_regressor_features,
    grouped = cfg.regressor.grouped,
    balanced = cfg.regressor.balanced,
    search_samples = search_n,
    search_problems = search_problems,
    search_targets = search_targets,
    clauses = cfg.regressor.clauses,
    clause_tolerance = cfg.regressor.clause_tolerance,
    clause_maximum = cfg.regressor.clause_maximum,
    target = cfg.regressor.target,
    specificity = cfg.regressor.specificity,
    include_bits = cfg.regressor.include_bits,
    search_patience = cfg.regressor.search_patience,
    search_rounds = cfg.regressor.search_rounds,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    final_patience = cfg.regressor.final_patience,
    final_iterations = cfg.regressor.final_iterations,
    search_metric = function (t)
      local predicted = t:predict(search_problems, search_n)
      local stats = eval.regression_accuracy(predicted, search_targets)
      return -stats.mean, stats
    end,
    each = cfg.verbose and util.make_regressor_log(stopwatch) or nil
  })

  print("\nPredicting train codes")
  local train_predicted_raw = train.regressor:predict(train_regressor_sentences, train.n)
  train_regressor_sentences = nil -- luacheck: ignore
  train_raw_codes = nil -- luacheck: ignore

  print("\nLearning ITQ on train codes")
  local train_predicted, itq_means, itq_rotation = itq.itq({
    codes = train_predicted_raw,
    n_dims = train.dims,
    iterations = 100,
    tolerance = 1e-8,
  })
  train_predicted_raw:destroy()
  train_predicted_raw = nil -- luacheck: ignore
  util.spot_check_codes(train_predicted, train.n, train.dims, "train predicted (ITQ binarized)")

  print("\nEvaluating predicted codes against eval adjacency")
  local pred_eval_stats = eval.ranking_accuracy({
    codes = train_predicted,
    ids = train.ids,
    eval_ids = train_eval_ids,
    eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors,
    eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Predicted codes ranking score: %.4f (spectral: %.4f)\n",
    pred_eval_stats.score, spectral_eval_stats.score)

  train_eval_ids = nil -- luacheck: ignore
  train_eval_offsets = nil -- luacheck: ignore
  train_eval_neighbors = nil -- luacheck: ignore
  train_eval_weights = nil -- luacheck: ignore

  print("\nPredicting dev codes")
  local dev_regressor_sentences = make_regressor_sentences(dev.full_tokens, dev.n)
  local dev_predicted_raw = train.regressor:predict(dev_regressor_sentences, dev.n)
  local dev_predicted = itq.apply({
    codes = dev_predicted_raw,
    means = itq_means,
    rotation = itq_rotation,
    n_dims = train.dims,
  })
  dev_predicted_raw:destroy()
  dev_predicted_raw = nil -- luacheck: ignore
  util.spot_check_codes(dev_predicted, dev.n, train.dims, "dev predicted (ITQ applied)")

  print("\nBuilding dev graph_index, label ANN, and evaluation adjacency")
  local dev_graph_features = ivec.create():copy(dev.solutions):bits_extend(dev.tokens, n_labels, n_tokens)
  dev.graph_index = inv.create({
    features = n_graph_features,
    ranks = graph_ranks,
    n_ranks = 2,
  })
  dev.graph_index:add(dev_graph_features, dev.ids)
  dev_graph_features = nil -- luacheck: ignore
  local dev_labels_cvec = dev.solutions:bits_to_cvec(dev.n, n_labels)
  dev.label_ann = ann.create({ features = n_labels })
  dev.label_ann:add(dev_labels_cvec, dev.ids)
  local dev_eval_ids, dev_eval_offsets, dev_eval_neighbors, dev_eval_weights =
    graph.adjacency({
      knn_index = dev.label_ann,
      knn = cfg.eval.knn,
      knn_cache = cfg.eval.knn,
      weight_index = dev.graph_index,
      weight_cmp = cfg.nystrom.cmp,
      weight_combine = cfg.nystrom.combine,
      weight_decay = cfg.nystrom.decay,
      bridge = "none",
    })
  math.randomseed(23456)
  local dev_spot_check_ids = {}
  for i = 1, 5 do
    dev_spot_check_ids[i] = dev_eval_ids:get(math.random(dev_eval_ids:size()) - 1)
  end
  if cfg.verbose then
    util.spot_check_adjacency(dev_eval_ids, dev_eval_offsets, dev_eval_neighbors, dev_eval_weights, "dev eval")
    util.spot_check_neighbors_with_labels(dev_eval_ids, dev_eval_offsets, dev_eval_neighbors, dev_eval_weights,
      dev.label_csr, train.n, "dev eval adj (knn+random)", dev_spot_check_ids, 10)
  end

  print("\nEvaluating dev predicted codes")
  local dev_pred_stats = eval.ranking_accuracy({
    codes = dev_predicted,
    ids = dev.ids,
    eval_ids = dev_eval_ids,
    eval_offsets = dev_eval_offsets,
    eval_neighbors = dev_eval_neighbors,
    eval_weights = dev_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Dev ranking score: %.4f\n", dev_pred_stats.score)


  dev_regressor_sentences = nil -- luacheck: ignore
  dev_predicted = nil -- luacheck: ignore
  dev_eval_ids = nil -- luacheck: ignore
  dev_eval_offsets = nil -- luacheck: ignore
  dev_eval_neighbors = nil -- luacheck: ignore
  dev_eval_weights = nil -- luacheck: ignore

  print("\nPredicting test codes")
  local test_regressor_sentences = make_regressor_sentences(test_set.full_tokens, test_set.n)
  local test_predicted_raw = train.regressor:predict(test_regressor_sentences, test_set.n)
  local test_predicted = itq.apply({
    codes = test_predicted_raw,
    means = itq_means,
    rotation = itq_rotation,
    n_dims = train.dims,
  })
  test_predicted_raw:destroy()
  test_predicted_raw = nil -- luacheck: ignore
  util.spot_check_codes(test_predicted, test_set.n, train.dims, "test predicted (ITQ applied)")

  print("\nBuilding test graph_index, label ANN, and evaluation adjacency")
  local test_graph_features = ivec.create():copy(test_set.solutions):bits_extend(test_set.tokens, n_labels, n_tokens)
  test_set.graph_index = inv.create({
    features = n_graph_features,
    ranks = graph_ranks,
    n_ranks = 2,
  })
  test_set.graph_index:add(test_graph_features, test_set.ids)
  test_graph_features = nil -- luacheck: ignore
  local test_labels_cvec = test_set.solutions:bits_to_cvec(test_set.n, n_labels)
  test_set.label_ann = ann.create({ features = n_labels })
  test_set.label_ann:add(test_labels_cvec, test_set.ids)
  local test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights =
    graph.adjacency({
      knn_index = test_set.label_ann,
      knn = cfg.eval.knn,
      knn_cache = cfg.eval.knn,
      weight_index = test_set.graph_index,
      weight_cmp = cfg.nystrom.cmp,
      weight_combine = cfg.nystrom.combine,
      weight_decay = cfg.nystrom.decay,
      bridge = "none",
    })
  math.randomseed(34567)
  local test_spot_check_ids = {}
  for i = 1, 5 do
    test_spot_check_ids[i] = test_eval_ids:get(math.random(test_eval_ids:size()) - 1)
  end
  if cfg.verbose then
    util.spot_check_adjacency(test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights, "test eval")
    util.spot_check_neighbors_with_labels(test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights,
      test_set.label_csr, train.n + dev.n, "test eval adj (knn+random)", test_spot_check_ids, 10)
  end

  print("\nEvaluating test predicted codes")
  local test_pred_stats = eval.ranking_accuracy({
    codes = test_predicted,
    ids = test_set.ids,
    eval_ids = test_eval_ids,
    eval_offsets = test_eval_offsets,
    eval_neighbors = test_eval_neighbors,
    eval_weights = test_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Test ranking score: %.4f\n", test_pred_stats.score)
  test_regressor_sentences = nil -- luacheck: ignore
  test_predicted = nil -- luacheck: ignore

  print("\n" .. string.rep("=", 60))
  print("SUMMARY")
  print(string.rep("=", 60))
  str.printf("  Dims: %d\n", train.dims)
  str.printf("  Train spectral score: %.4f\n", spectral_eval_stats.score)
  str.printf("  Train predicted score: %.4f\n", pred_eval_stats.score)
  str.printf("  Dev predicted score: %.4f\n", dev_pred_stats.score)
  str.printf("  Test predicted score: %.4f\n", test_pred_stats.score)
  str.printf("  Time: %.1fs\n", stopwatch())

end)
