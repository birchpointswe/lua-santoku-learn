local _ -- luacheck: ignore
local ann = require("santoku.tsetlin.ann")
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
  exit_after = nil, --"spectral",
  data = {
    max = 4000,
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
    grouped = false,
    max_vocab_spectral = 2^16,
    max_vocab_encoder = 2^15,
    features_per_class = 2^13,
  },
  encoder = {
    clauses = 8, --{ def = 8, min = 8, max = 64, round = 8 },
    clause_tolerance = { def = 27, min = 8, max = 128, int = true },
    clause_maximum = { def = 46, min = 16, max = 128, int = true },
    target = { def = 7, min = 4, max = 64, int = true },
    specificity = { def = 153, min = 50, max = 2000 },
    include_bits = { def = 1, min = 1, max = 6, int = true },
    search_frac = 0.2,
    search_patience = 4,
    search_rounds = 4,
    search_trials = 10,
    search_iterations = 10,
    final_patience = 40,
    final_iterations = 400,
  },
  nystrom = {
    invert_ranks = false,
    n_landmarks = 2048,
    n_dims = 48,
    cmp = "cosine",
    combine = "exponential",
    decay = { def = 2.0, min = 0.0, max = 3.0 },
    binarize = "median",
    rounds = 0,
    samples = 20,
  },
  eval = {
    anchors = 16,
    pairs = 64,
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
  local tok_index = tok:index()
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil -- luacheck: ignore
  dev.problems = nil -- luacheck: ignore
  test_set.problems = nil -- luacheck: ignore

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

  print("\nFeature selection")
  local df_ids, df_scores = train.tokens:bits_top_df(train.n, n_tokens)
  str.printf("  IDF: %d -> %d\n", n_tokens, df_ids:size())
  train.tokens:bits_select(df_ids, nil, n_tokens)
  dev.tokens:bits_select(df_ids, nil, n_tokens)
  test_set.tokens:bits_select(df_ids, nil, n_tokens)
  n_tokens = df_ids:size()
  local bns_ids, bns_scores = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_labels,
    cfg.feature_selection.max_vocab_spectral, "max")
  str.printf("  BNS: %d -> %d\n", n_tokens, bns_ids:size())
  local idf_gathered = dvec.create()
  idf_gathered:copy(df_scores, bns_ids)
  bns_scores:scalev(idf_gathered)
  local tok_id_map = ivec.create(bns_ids:size())
  for i = 0, bns_ids:size() - 1 do
    tok_id_map:set(i, df_ids:get(bns_ids:get(i)))
  end
  df_ids = nil -- luacheck: ignore
  df_scores = nil -- luacheck: ignore
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  dev.tokens:bits_select(bns_ids, nil, n_tokens)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens)
  local n_top_v = bns_ids:size()
  bns_ids = nil -- luacheck: ignore
  local label_df_ids, label_idf_scores = train.solutions:bits_top_df(train.n, n_labels)
  local n_graph_features = n_labels + n_top_v
  local graph_weights = dvec.create(n_graph_features)
  graph_weights:fill(0.0, 0, n_labels)
  graph_weights:copy(label_idf_scores, label_df_ids, true)
  label_df_ids = nil -- luacheck: ignore
  label_idf_scores = nil -- luacheck: ignore
  graph_weights:copy(bns_scores, n_labels)
  bns_scores = nil -- luacheck: ignore

  print("\nBuilding graph_index (docs only, two-rank: labels + tokens)")
  local graph_features = ivec.create()
  graph_features:copy(train.solutions)
  graph_features:bits_extend(train.tokens, n_labels, n_top_v)
  local graph_ranks = ivec.create(n_graph_features)
  if cfg.nystrom.invert_ranks then
    graph_ranks:fill(1, 0, n_labels)
    graph_ranks:fill(0, n_labels, n_graph_features)
  else
    graph_ranks:fill(0, 0, n_labels)
    graph_ranks:fill(1, n_labels, n_graph_features)
  end
  train.graph_index = inv.create({
    features = graph_weights,
    ranks = graph_ranks,
    n_ranks = 2,
  })
  train.graph_index:add(graph_features, train.ids)
  graph_features = nil -- luacheck: ignore
  graph_ranks = nil -- luacheck: ignore
  str.printf("  Docs: %d  Features: %d (labels=%d, tokens=%d)\n",
    train.n, n_graph_features, n_labels, n_top_v)

  print("\nBuilding eval_index (labels only) and evaluation adjacency")
  train.eval_index = inv.create({ features = n_labels })
  train.eval_index:add(train.solutions, train.ids)
  local train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights =
    graph.adjacency({
      category_index = train.eval_index,
      category_cmp = cfg.eval.cmp,
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
    combine = cfg.nystrom.combine,
    decay = cfg.nystrom.decay,
    binarize = cfg.nystrom.binarize,
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

  print("\nBit entropy stats for spectral codes:")
  local entropy_ids, entropy_scores = train_target_codes:bits_top_entropy(train.n, train.dims, train.dims)
  local zero_entropy_count = 0
  for i = 0, entropy_ids:size() - 1 do
    local dim = entropy_ids:get(i)
    local ent = entropy_scores:get(i)
    if ent < 0.01 then
      zero_entropy_count = zero_entropy_count + 1
      if zero_entropy_count <= 10 then
        str.printf("  dim %d: entropy=%.6f (near-constant)\n", dim, ent)
      end
    end
  end
  str.printf("  Total near-constant dims (entropy < 0.01): %d / %d\n", zero_entropy_count, train.dims)
  str.printf("  Entropy range: min=%.6f max=%.6f\n", entropy_scores:min(), entropy_scores:max())
  entropy_ids = nil -- luacheck: ignore
  entropy_scores = nil -- luacheck: ignore

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
  local spectral_raw_stats = eval.ranking_accuracy({
    raw_codes = model.raw_codes,
    ids = model.ids,
    eval_ids = train_eval_ids,
    eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors,
    eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = model.spectral_dims or train.dims,
  })
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
  str.printf("  Spectral codes ranking: raw=%.4f binary=%.4f\n", spectral_raw_stats.score, spectral_eval_stats.score)
  model.raw_codes = nil -- luacheck: ignore

  if cfg.exit_after == "spectral" then
    return
  end

  local n_encoder_features, encoder_feat_ids, encoder_class_offsets

  if cfg.feature_selection.grouped then

    print("\nEncoder feature selection (GROUPED mode)")
    local features_per_class = cfg.feature_selection.features_per_class
    encoder_class_offsets, encoder_feat_ids, _ = train.tokens:bits_top_chi2_grouped(
      train_target_codes, train.n, n_top_v, train.dims, features_per_class)
    n_encoder_features = features_per_class
    str.printf("  Chi2 grouped: %d features total (%d per class x %d classes)\n",
      encoder_feat_ids:size(), features_per_class, train.dims)

    print("\nTop 10 tokens per encoding bit:")
    for c = 0, train.dims - 1 do
      local tokens = {}
      for i = 0, math.min(10, features_per_class) - 1 do
        local fid = encoder_feat_ids:get(c * features_per_class + i)
        local orig_id = tok_id_map:get(fid)
        local token = tok_index[orig_id + 1] or ("?" .. orig_id)
        tokens[#tokens + 1] = token
      end
      str.printf("  bit %2d: %s\n", c, table.concat(tokens, ", "))
    end

    local bytes_per_class = math.ceil(features_per_class * 2 / 8)
    str.printf("  Per-class layout: %d bytes/class, %d bytes/sample\n", bytes_per_class, train.dims * bytes_per_class)

  else

    print("\nEncoder feature selection (REGULAR mode)")
    encoder_feat_ids = train.tokens:bits_top_chi2(
      train_target_codes, train.n, n_top_v, train.dims,
      cfg.feature_selection.max_vocab_encoder, "max")
    n_encoder_features = encoder_feat_ids:size()
    str.printf("  Chi2: %d features selected\n", n_encoder_features)

    print("\nTop 20 tokens selected:")
    local tokens = {}
    for i = 0, math.min(20, n_encoder_features) - 1 do
      local fid = encoder_feat_ids:get(i)
      local orig_id = tok_id_map:get(fid)
      local token = tok_index[orig_id + 1] or ("?" .. orig_id)
      tokens[#tokens + 1] = token
    end
    str.printf("  %s\n", table.concat(tokens, ", "))

    encoder_class_offsets = nil

  end

  local function make_encoder_sentences(tokens, n_samples)
    if cfg.feature_selection.grouped then
      return tokens:bits_to_cvec(n_samples, n_top_v, encoder_class_offsets, encoder_feat_ids, true)
    else
      local selected = ivec.create()
      selected:copy(tokens)
      selected:bits_select(encoder_feat_ids, nil, n_top_v)
      return selected:bits_to_cvec(n_samples, n_encoder_features, true)
    end
  end

  local train_encoder_sentences, actual_n_encoder_features = make_encoder_sentences(train.tokens, train.n)
  if cfg.feature_selection.grouped then
    n_encoder_features = actual_n_encoder_features
  end

  local search_n = math.floor(train.n * cfg.encoder.search_frac)
  local search_ids = ivec.create(search_n)
  search_ids:copy(train.ids, 0, search_n)
  local search_codes = train.index:get(search_ids)
  local search_sentences = make_encoder_sentences(train.tokens, search_n)
  str.printf("\nSearch subset: %d / %d samples (%.0f%%)\n", search_n, train.n, cfg.encoder.search_frac * 100)

  print("\nTraining encoder")
  train.encoder = optimize.encoder({
    hidden = train.dims,
    codes = train_target_codes,
    samples = train.n,
    sentences = train_encoder_sentences,
    visible = n_encoder_features,
    grouped = cfg.feature_selection.grouped,
    search_samples = search_n,
    search_sentences = search_sentences,
    search_codes = search_codes,
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
      local accuracy = eval.encoding_accuracy(predicted, search_codes, enc_info.samples, train.dims)
      return accuracy.mean_hamming, accuracy
    end,
    each = cfg.verbose and util.encoder_log or nil,
  })

  print("\nPredicting train codes")
  local train_predicted = train.encoder:predict(train_encoder_sentences, train.n)
  util.spot_check_codes(train_predicted, train.n, train.dims, "train predicted (full)")

  local train_ham = eval.encoding_accuracy(train_predicted, train_target_codes, train.n, train.dims).mean_hamming
  str.printf("  Train hamming: %.4f\n", train_ham)
  train_target_codes = nil -- luacheck: ignore
  train_encoder_sentences = nil -- luacheck: ignore

  print("\nEvaluating predicted codes against eval adjacency")
  local train_pred_ann = ann.create({ features = train.dims })
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

  train_eval_ids = nil -- luacheck: ignore
  train_eval_offsets = nil -- luacheck: ignore
  train_eval_neighbors = nil -- luacheck: ignore
  train_eval_weights = nil -- luacheck: ignore
  train_pred_ann = nil -- luacheck: ignore

  print("\nPredicting dev codes")
  local dev_encoder_sentences = make_encoder_sentences(dev.tokens, dev.n)
  local dev_predicted = train.encoder:predict(dev_encoder_sentences, dev.n)
  util.spot_check_codes(dev_predicted, dev.n, train.dims, "dev predicted")

  print("\nBuilding dev eval_index (labels only) and evaluation adjacency")
  dev.eval_index = inv.create({ features = n_labels })
  dev.eval_index:add(dev.solutions, dev.ids)
  local dev_eval_ids, dev_eval_offsets, dev_eval_neighbors, dev_eval_weights =
    graph.adjacency({
      category_index = dev.eval_index,
      category_cmp = cfg.eval.cmp,
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
  local dev_pred_ann = ann.create({ features = train.dims })
  dev_pred_ann:add(dev_predicted, dev.ids)
  local dev_pred_stats = eval.ranking_accuracy({
    index = dev_pred_ann,
    ids = dev.ids,
    eval_ids = dev_eval_ids,
    eval_offsets = dev_eval_offsets,
    eval_neighbors = dev_eval_neighbors,
    eval_weights = dev_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Dev ranking score: %.4f\n", dev_pred_stats.score)
  dev_encoder_sentences = nil -- luacheck: ignore
  dev_predicted = nil -- luacheck: ignore
  dev_pred_ann = nil -- luacheck: ignore
  dev_eval_ids = nil -- luacheck: ignore
  dev_eval_offsets = nil -- luacheck: ignore
  dev_eval_neighbors = nil -- luacheck: ignore
  dev_eval_weights = nil -- luacheck: ignore

  print("\nPredicting test codes")
  local test_encoder_sentences = make_encoder_sentences(test_set.tokens, test_set.n)
  local test_predicted = train.encoder:predict(test_encoder_sentences, test_set.n)
  util.spot_check_codes(test_predicted, test_set.n, train.dims, "test predicted")

  print("\nBuilding test eval_index (labels only) and evaluation adjacency")
  test_set.eval_index = inv.create({ features = n_labels })
  test_set.eval_index:add(test_set.solutions, test_set.ids)
  local test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights =
    graph.adjacency({
      category_index = test_set.eval_index,
      category_cmp = cfg.eval.cmp,
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
  local test_pred_ann = ann.create({ features = train.dims })
  test_pred_ann:add(test_predicted, test_set.ids)
  local test_pred_stats = eval.ranking_accuracy({
    index = test_pred_ann,
    ids = test_set.ids,
    eval_ids = test_eval_ids,
    eval_offsets = test_eval_offsets,
    eval_neighbors = test_eval_neighbors,
    eval_weights = test_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Test ranking score: %.4f\n", test_pred_stats.score)
  test_encoder_sentences = nil -- luacheck: ignore
  test_predicted = nil -- luacheck: ignore
  test_pred_ann = nil -- luacheck: ignore

  print("\n" .. string.rep("=", 60))
  print("SUMMARY")
  print(string.rep("=", 60))
  str.printf("  Spectral dims: %d\n", train.dims)
  str.printf("  Train spectral score: %.4f\n", spectral_eval_stats.score)
  str.printf("  Train predicted score: %.4f  hamming: %.4f\n", pred_eval_stats.score, train_ham)
  str.printf("  Dev predicted score: %.4f\n", dev_pred_stats.score)
  str.printf("  Test predicted score: %.4f\n", test_pred_stats.score)
  str.printf("  Time: %.1fs\n", stopwatch())

end)
