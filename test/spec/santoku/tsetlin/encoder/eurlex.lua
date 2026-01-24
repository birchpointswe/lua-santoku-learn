local ann = require("santoku.tsetlin.ann")
local cvec = require("santoku.cvec")
local ds = require("santoku.tsetlin.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.tsetlin.evaluator")
local graph = require("santoku.tsetlin.graph")
local inv = require("santoku.tsetlin.inv")
local ivec = require("santoku.ivec")
local optimize = require("santoku.tsetlin.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.tsetlin.util")
local utc = require("santoku.utc")

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
    expected_vocab = 1300000,
  },
  feature_selection = {
    min_df = -2,
    max_df = 0.98,
    max_vocab = 8192,
  },
  encoder = {
    clauses = { def = 248, min = 16, max = 256, round = 8 },
    clause_tolerance = { def = 20, min = 8, max = 128, int = true },
    clause_maximum = { def = 119, min = 16, max = 128, int = true },
    target = { def = 17, min = 4, max = 64, int = true },
    specificity = { def = 816, min = 50, max = 2000 },
    include_bits = { def = 2, min = 1, max = 6, int = true },
    search_patience = 4,
    search_rounds = 4,
    search_trials = 10,
    search_iterations = 10,
    final_patience = 40,
    final_iterations = 400,
  },
  bit_pruning = {
    enabled = true,
    metric = "retrieval",
    ranking = "ndcg",
    tolerance = 1e-6,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 24,
    cmp = "jaccard",
    decay = { def = 0.67, min = 0.0, max = 2.0 },
    rounds = 4,
    samples = 10,
  },
  eval = {
    anchors = 16,
    pairs = 64,
    ranking = "ndcg",
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

  print("\nFeature selection (DF filter)")
  local df_sorted = train.tokens:bits_top_df(
    train.n, n_tokens, nil,
    cfg.feature_selection.min_df, cfg.feature_selection.max_df)
  local n_top_v = df_sorted:size()
  str.printf("  Vocab: %d -> %d (DF filtered)\n", n_tokens, n_top_v)
  tok:restrict(df_sorted)
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)

  print("\nCreating IDs")
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  dev.ids = ivec.create(dev.n)
  dev.ids:fill_indices()
  dev.ids:add(train.n)
  test_set.ids = ivec.create(test_set.n)
  test_set.ids:fill_indices()
  test_set.ids:add(train.n + dev.n)

  print("\nBuilding label CSR for lookups")
  local train_label_offsets, train_label_neighbors = train.solutions:bits_to_csr(train.n, n_labels)
  train.label_csr = { offsets = train_label_offsets, neighbors = train_label_neighbors }
  local dev_label_offsets, dev_label_neighbors = dev.solutions:bits_to_csr(dev.n, n_labels)
  dev.label_csr = { offsets = dev_label_offsets, neighbors = dev_label_neighbors }
  local test_label_offsets, test_label_neighbors = test_set.solutions:bits_to_csr(test_set.n, n_labels)
  test_set.label_csr = { offsets = test_label_offsets, neighbors = test_label_neighbors }

  local n_graph_features = n_labels + n_top_v

  print("\nComputing BNS weights for tokens")
  local bns_top_ids, bns_top_scores = train.tokens:bits_top_bns(
    train.solutions, train.n, n_top_v, n_labels, n_top_v)
  local graph_weights = dvec.create(n_graph_features)
  graph_weights:fill(1.0, 0, n_labels)
  for i = 0, bns_top_ids:size() - 1 do
    local tok_id = bns_top_ids:get(i)
    graph_weights:set(n_labels + tok_id, bns_top_scores:get(i))
  end
  str.printf("  BNS weights: min=%.4f max=%.4f (for %d tokens)\n",
    bns_top_scores:min(), bns_top_scores:max(), bns_top_ids:size())

  print("\nBuilding graph_index (docs only, two-rank: labels + tokens)")
  local graph_features = ivec.create()
  graph_features:copy(train.solutions)
  graph_features:bits_extend(train.tokens, n_labels, n_top_v)
  local graph_ranks = ivec.create(n_graph_features)
  graph_ranks:fill(0, 0, n_labels)
  graph_ranks:fill(1, n_labels, n_graph_features)
  train.graph_index = inv.create({
    features = graph_weights,
    ranks = graph_ranks,
    n_ranks = 2,
  })
  train.graph_index:add(graph_features, train.ids)
  str.printf("  Docs: %d  Features: %d (labels=%d, tokens=%d)\n",
    train.n, n_graph_features, n_labels, n_top_v)

  print("\nBuilding eval_index (labels only) and evaluation adjacency")
  train.eval_index = inv.create({ features = n_labels, expected_size = train.n })
  train.eval_index:add(train.solutions, train.ids)
  local train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights =
    graph.adjacency({
      category_index = train.eval_index,
      category_anchors = cfg.eval.anchors,
      random_pairs = cfg.eval.pairs,
    })
  math.randomseed(12345)
  local spot_check_ids = {}
  for i = 1, 5 do
    spot_check_ids[i] = train_eval_ids:get(math.random(train_eval_ids:size()) - 1)
  end
  if cfg.verbose then
    util.spot_check_adjacency(train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights, "train eval")
    util.spot_check_neighbors_with_labels(train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights,
      train.label_csr, 0, "eval adj (anchors+random)", spot_check_ids, 10)
  end

  print("\nRunning spectral embedding (Nyström)")
  local model = optimize.spectral({
    index = train.graph_index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    cmp = cfg.nystrom.cmp,
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
  train.index = model.index
  train.dims = model.dims

  print("\nExtracting spectral codes for training data")
  local train_target_codes = train.index:get(train.ids)
  util.spot_check_codes(train_target_codes, train.n, train.dims, "spectral codes")

  if cfg.verbose then
    print("\nSpot-checking spectral code KNN (compare to eval adj above)")
    local spectral_knn_ids, spectral_knn_offsets, spectral_knn_neighbors, spectral_knn_weights =
      graph.adjacency({
        index = train.index,
        knn_index = train.index,
        knn = 32,
        knn_cache = 32,
        bridge = "none",
      })
    util.spot_check_neighbors_with_labels(spectral_knn_ids, spectral_knn_offsets, spectral_knn_neighbors, spectral_knn_weights,
      train.label_csr, 0, "spectral code KNN", spot_check_ids, 10)
  end

  print("\nEvaluating spectral codes against eval adjacency")
  local spectral_eval_stats = eval.ranking_accuracy({
    index = train.index,
    ids = model.ids,
    eval_ids = train_eval_ids,
    eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors,
    eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Spectral codes ranking score: %.4f\n", spectral_eval_stats.score)

  print("\nPer-dim Chi2 feature selection for encoder")
  local chi2_vocab = train.tokens:bits_top_chi2_ind(
    train_target_codes, train.n, n_top_v, train.dims, cfg.feature_selection.max_vocab)
  local train_encoder_visible = chi2_vocab:size()
  str.printf("  Chi2: %d features (union across %d dims)\n", train_encoder_visible, train.dims)
  tok:restrict(chi2_vocab)
  local train_toks = tok:tokenize(train.problems)
  local train_encoder_sentences = train_toks:bits_to_cvec(train.n, train_encoder_visible, true)

  print("\nTraining encoder")
  local encoder_args = {
    hidden = train.dims,
    codes = train_target_codes,
    samples = train.n,
    sentences = train_encoder_sentences,
    visible = train_encoder_visible,
    clauses = cfg.encoder.clauses,
    clause_tolerance = cfg.encoder.clause_tolerance,
    clause_maximum = cfg.encoder.clause_maximum,
    target = cfg.encoder.target,
    specificity = cfg.encoder.specificity,
    include_bits = cfg.encoder.include_bits,
    search_patience = cfg.encoder.search_patience,
    search_rounds = cfg.encoder.search_rounds,
    search_trials = cfg.encoder.search_trials,
    search_iterations = cfg.encoder.search_iterations,
    final_patience = cfg.encoder.final_patience,
    final_iterations = cfg.encoder.final_iterations,
    search_metric = function (t, enc_info)
      local predicted = t:predict(enc_info.sentences, enc_info.samples)
      local accuracy = eval.encoding_accuracy(predicted, train_target_codes, enc_info.samples, train.dims)
      return accuracy.mean_hamming, accuracy
    end,
    each = cfg.verbose and function (_, is_final, metrics, params, epoch, round, trial)
      local phase = is_final and "F" or str.format("R%d T%d", round, trial)
      str.printf("[ENCODER %s E%d] C=%d L=%d/%d T=%d S=%.0f IB=%d ham=%.4f\n",
        phase, epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
        params.target, params.specificity, params.include_bits, metrics.mean_hamming)
    end or nil,
  }
  train.encoder = optimize.encoder(encoder_args)

  print("\nPredicting train codes")
  local train_predicted = train.encoder:predict(train_encoder_sentences, train.n)
  util.spot_check_codes(train_predicted, train.n, train.dims, "train predicted")

  local train_ham = eval.encoding_accuracy(train_predicted, train_target_codes, train.n, train.dims).mean_hamming
  str.printf("  Train hamming: %.4f\n", train_ham)

  print("\nEvaluating predicted codes against eval adjacency")
  local train_pred_ann = ann.create({ features = train.dims, expected_size = train.n })
  train_pred_ann:add(train_predicted, train.ids)

  local pred_eval_stats = eval.ranking_accuracy({
    index = train_pred_ann,
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

  local dims_final = train.dims
  local active_bits = nil
  if cfg.bit_pruning and cfg.bit_pruning.enabled then
    print("\nOptimizing bits")
    active_bits = eval.optimize_bits({
      index = train_pred_ann,
      expected_ids = train_eval_ids,
      expected_offsets = train_eval_offsets,
      expected_neighbors = train_eval_neighbors,
      expected_weights = train_eval_weights,
      n_dims = train.dims,
      start_prefix = train.dims,
      metric = cfg.bit_pruning.metric,
      ranking = cfg.bit_pruning.ranking,
      tolerance = cfg.bit_pruning.tolerance,
      each = cfg.verbose and function (bit, gain, score, event)
        str.printf("  %s bit=%d gain=%.6f score=%.6f\n", event, bit, gain, score)
      end or nil,
    })
    local n_active = active_bits:size()
    str.printf("  Active bits: %d / %d (%.0f%% pruned)\n",
      n_active, train.dims, 100 * (1 - n_active / train.dims))
    if n_active < train.dims then
      local train_pruned = cvec.create()
      train_predicted:bits_select(active_bits, nil, train.dims, train_pruned)
      train_predicted = train_pruned
      dims_final = n_active
      util.spot_check_codes(train_predicted, train.n, dims_final, "train pruned")
    end
  end

  print("\nPredicting dev codes")
  local dev_toks = tok:tokenize(dev.problems)
  local dev_encoder_sentences = dev_toks:bits_to_cvec(dev.n, train_encoder_visible, true)
  local dev_predicted = train.encoder:predict(dev_encoder_sentences, dev.n)
  if active_bits then
    local dev_pruned = cvec.create()
    dev_predicted:bits_select(active_bits, nil, train.dims, dev_pruned)
    dev_predicted = dev_pruned
  end
  util.spot_check_codes(dev_predicted, dev.n, dims_final, "dev predicted")

  print("\nBuilding dev eval_index (labels only) and evaluation adjacency")
  dev.eval_index = inv.create({ features = n_labels, expected_size = dev.n })
  dev.eval_index:add(dev.solutions, dev.ids)
  local dev_eval_ids, dev_eval_offsets, dev_eval_neighbors, dev_eval_weights =
    graph.adjacency({
      category_index = dev.eval_index,
      category_anchors = cfg.eval.anchors,
      random_pairs = cfg.eval.pairs,
    })
  math.randomseed(23456)
  local dev_spot_check_ids = {}
  for i = 1, 5 do
    dev_spot_check_ids[i] = dev_eval_ids:get(math.random(dev_eval_ids:size()) - 1)
  end
  if cfg.verbose then
    util.spot_check_adjacency(dev_eval_ids, dev_eval_offsets, dev_eval_neighbors, dev_eval_weights, "dev eval")
    util.spot_check_neighbors_with_labels(dev_eval_ids, dev_eval_offsets, dev_eval_neighbors, dev_eval_weights,
      dev.label_csr, train.n, "dev eval adj (anchors+random)", dev_spot_check_ids, 10)
  end

  print("\nEvaluating dev predicted codes")
  local dev_pred_ann = ann.create({ features = dims_final, expected_size = dev.n })
  dev_pred_ann:add(dev_predicted, dev.ids)
  local dev_pred_stats = eval.ranking_accuracy({
    index = dev_pred_ann,
    ids = dev.ids,
    eval_ids = dev_eval_ids,
    eval_offsets = dev_eval_offsets,
    eval_neighbors = dev_eval_neighbors,
    eval_weights = dev_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = dims_final,
  })
  str.printf("  Dev ranking score: %.4f\n", dev_pred_stats.score)

  print("\nPredicting test codes")
  local test_toks = tok:tokenize(test_set.problems)
  local test_encoder_sentences = test_toks:bits_to_cvec(test_set.n, train_encoder_visible, true)
  local test_predicted = train.encoder:predict(test_encoder_sentences, test_set.n)
  if active_bits then
    local test_pruned = cvec.create()
    test_predicted:bits_select(active_bits, nil, train.dims, test_pruned)
    test_predicted = test_pruned
  end
  util.spot_check_codes(test_predicted, test_set.n, dims_final, "test predicted")

  print("\nBuilding test eval_index (labels only) and evaluation adjacency")
  test_set.eval_index = inv.create({ features = n_labels, expected_size = test_set.n })
  test_set.eval_index:add(test_set.solutions, test_set.ids)
  local test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights =
    graph.adjacency({
      category_index = test_set.eval_index,
      category_anchors = cfg.eval.anchors,
      random_pairs = cfg.eval.pairs,
    })
  math.randomseed(34567)
  local test_spot_check_ids = {}
  for i = 1, 5 do
    test_spot_check_ids[i] = test_eval_ids:get(math.random(test_eval_ids:size()) - 1)
  end
  if cfg.verbose then
    util.spot_check_adjacency(test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights, "test eval")
    util.spot_check_neighbors_with_labels(test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights,
      test_set.label_csr, train.n + dev.n, "test eval adj (anchors+random)", test_spot_check_ids, 10)
  end

  print("\nEvaluating test predicted codes")
  local test_pred_ann = ann.create({ features = dims_final, expected_size = test_set.n })
  test_pred_ann:add(test_predicted, test_set.ids)
  local test_pred_stats = eval.ranking_accuracy({
    index = test_pred_ann,
    ids = test_set.ids,
    eval_ids = test_eval_ids,
    eval_offsets = test_eval_offsets,
    eval_neighbors = test_eval_neighbors,
    eval_weights = test_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = dims_final,
  })
  str.printf("  Test ranking score: %.4f\n", test_pred_stats.score)

  print("\n" .. string.rep("=", 60))
  print("SUMMARY")
  print(string.rep("=", 60))
  str.printf("  Spectral dims: %d -> %d (after pruning)\n", train.dims, dims_final)
  str.printf("  Train spectral score: %.4f\n", spectral_eval_stats.score)
  str.printf("  Train predicted score: %.4f  hamming: %.4f\n", pred_eval_stats.score, train_ham)
  str.printf("  Dev predicted score: %.4f\n", dev_pred_stats.score)
  str.printf("  Test predicted score: %.4f\n", test_pred_stats.score)
  str.printf("  Time: %.1fs\n", stopwatch())

end)
