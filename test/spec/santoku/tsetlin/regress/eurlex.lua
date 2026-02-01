local _ -- luacheck: ignore
local ann = require("santoku.tsetlin.ann")
local ds = require("santoku.tsetlin.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.tsetlin.evaluator")
local graph = require("santoku.tsetlin.graph")
local hlth = require("santoku.tsetlin.hlth")
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
    bns_tokens = 65536,
    per_class = 1024,
  },
  regressor = {
    clauses = 8,
    clause_tolerance = { def = 749, min = 8, max = 1024, int = true },
    clause_maximum = { def = 105, min = 8, max = 1024, int = true },
    target = { def = 82, min = 8, max = 1024, int = true },
    specificity = { def = 526, min = 2, max = 2000 },
    search_rounds = 0,
    search_trials = 20,
    search_iterations = 40,
    final_patience = 20,
    final_batch = 40,
    final_iterations = 400,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 256,
    cmp = "cosine",
    combine = "rbf",
    decay = 1.0,
    rounds = 0,
    samples = 20,
  },
  itq = {
    iterations = 50,
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
  local token_index = tok:index()
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

  print("\nBNS feature selection")
  local bns_ids = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_labels, nil, cfg.feature_selection.bns_tokens, "max")
  local n_bns_tokens = bns_ids:size()
  str.printf("  BNS selection: %d -> %d tokens\n", n_tokens, n_bns_tokens)

  print("\nApplying BNS selection to all splits")
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  dev.tokens:bits_select(bns_ids, nil, n_tokens)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens)
  n_tokens = n_bns_tokens

  local n_graph_features = n_labels + n_bns_tokens

  print("\nBuilding graph_index (labels + tokens, unweighted)")
  local graph_features = ivec.create():copy(train.solutions):bits_extend(train.tokens, n_labels, n_bns_tokens)
  train.graph_index = inv.create({
    features = n_graph_features,
  })
  train.graph_index:add(graph_features, train.ids)
  graph_features = nil -- luacheck: ignore
  str.printf("  Docs: %d  Features: %d (labels=%d, BNS tokens=%d)\n",
    train.n, n_graph_features, n_labels, n_bns_tokens)

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
  local spectral_metrics
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
    each = function(ev)
      if cfg.verbose then util.spectral_log(ev) end
      if ev.event == "eval" or ev.event == "done" then
        spectral_metrics = ev.metrics or ev.best_metrics
      end
    end,
  })
  train.dims = model.dims
  local train_raw_codes = model.raw_codes
  str.printf("  Spectral dims: %d, samples: %d\n", train.dims, train.n)

  if cfg.exit_after == "spectral" then
    return
  end

  print("\nRegression feature selection for regressor")
  local n_regressor_features, make_regressor_sentences

  local _, _, class_offsets, class_feat_ids, class_scores = train.tokens:bits_top_reg_f(
    train_raw_codes, train.n, n_tokens, train.dims, cfg.feature_selection.per_class, nil, "max")
  str.printf("  Regression features (grouped): %d total (requested %d per class x %d classes)\n",
    class_feat_ids:size(), cfg.feature_selection.per_class, train.dims)
  str.printf("  F-scores: min=%.2f max=%.2f mean=%.2f\n",
    class_scores:min(), class_scores:max(), class_scores:sum() / class_scores:size())

  print("\nTop 5 features per spectral dimension:")
  for d = 1, train.dims do
    local start_idx = class_offsets:get(d - 1)
    local end_idx = class_offsets:get(d)
    local top_n = math.min(5, end_idx - start_idx)
    local feats = {}
    for i = 0, top_n - 1 do
      local fid = class_feat_ids:get(start_idx + i)
      local score = class_scores:get(start_idx + i)
      local bns_id = bns_ids:get(fid)
      local orig_id = df_ids:get(bns_id)
      local token = token_index[orig_id + 1] or tostring(orig_id)
      feats[i + 1] = string.format("%s(%.1f)", token, score)
    end
    str.printf("  dim_%02d: %s\n", d - 1, table.concat(feats, ", "))
  end

  make_regressor_sentences = function(tokens, n_samples)
    local cvec, max_k = tokens:bits_to_cvec(n_samples, n_tokens, class_offsets, class_feat_ids, true)
    return cvec, max_k
  end

  local train_regressor_sentences, max_k
  train_regressor_sentences, max_k = make_regressor_sentences(train.tokens, train.n)
  n_regressor_features = max_k
  str.printf("  Regressor features (padded max per class): %d\n", n_regressor_features)

  print("\nTraining TM regressor")
  local predicted_buf = dvec.create()
  local tm_train_args = {
    outputs = train.dims,
    samples = train.n,
    problems = train_regressor_sentences,
    features = n_regressor_features,
    grouped = true,
    clauses = cfg.regressor.clauses,
    clause_tolerance = cfg.regressor.clause_tolerance,
    clause_maximum = cfg.regressor.clause_maximum,
    target = cfg.regressor.target,
    specificity = cfg.regressor.specificity,
    search_rounds = cfg.regressor.search_rounds,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    final_batch = cfg.regressor.final_batch,
    final_patience = cfg.regressor.final_patience,
    final_iterations = cfg.regressor.final_iterations,
    targets = train_raw_codes,
    search_metric = function (t)
      local predicted = t:regress(train_regressor_sentences, train.n, true, predicted_buf)
      local stats = eval.regression_accuracy(predicted, train_raw_codes)
      return -stats.mean, stats
    end,
    each = cfg.verbose and util.make_regressor_log(stopwatch) or nil,
  }
  train.tm = optimize.regressor(tm_train_args)
  train_raw_codes = nil -- luacheck: ignore

  print("\nPredicting train codes")
  local train_predicted_raw = train.tm:regress(train_regressor_sentences, train.n, true)
  local train_raw_stats = eval.ranking_accuracy({
    raw_codes = train_predicted_raw, ids = train.ids, n_dims = train.dims,
    eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
  })
  local train_predicted_raw_score = train_raw_stats.score

  print("\nLearning ITQ rotation on predicted codes")
  local itq_rotation = optimize.itq({
    codes = train_predicted_raw,
    n_samples = train.n,
    n_dims = train.dims,
    iterations = cfg.itq.iterations,
  })

  print("\nCreating ITQ encoder")
  local itq_encode, itq_n_bits = hlth.itq_encoder({
    rotation = itq_rotation,
    n_dims = train.dims,
  })
  str.printf("  ITQ: %d input dims -> %d output bits\n", train.dims, itq_n_bits)

  local train_predicted = itq_encode(train_predicted_raw)
  train_predicted_raw:destroy()
  train_predicted_raw = nil
  train_regressor_sentences = nil -- luacheck: ignore
  util.spot_check_codes(train_predicted, train.n, itq_n_bits, "train predicted")
  local train_entropy = eval.entropy_stats(train_predicted, train.n, itq_n_bits)
  str.printf("  Train predicted entropy: mean=%.4f min=%.4f max=%.4f std=%.4f\n",
    train_entropy.mean, train_entropy.min, train_entropy.max, train_entropy.std)
  util.cluster_stats({
    codes = train_predicted, ids = train.ids, n_dims = itq_n_bits, knn = cfg.eval.knn,
    eval_offsets = train_eval_offsets, eval_neighbors = train_eval_neighbors,
    eval_weights = train_eval_weights, label = "train predicted",
  })

  print("\nPredicting dev codes")
  local dev_regressor_sentences = make_regressor_sentences(dev.tokens, dev.n)
  local dev_predicted_raw = train.tm:regress(dev_regressor_sentences, dev.n, true)
  local dev_predicted = itq_encode(dev_predicted_raw)
  util.spot_check_codes(dev_predicted, dev.n, itq_n_bits, "dev predicted")
  local dev_entropy = eval.entropy_stats(dev_predicted, dev.n, itq_n_bits)
  str.printf("  Dev predicted entropy: mean=%.4f min=%.4f max=%.4f std=%.4f\n",
    dev_entropy.mean, dev_entropy.min, dev_entropy.max, dev_entropy.std)

  print("\nBuilding dev graph_index, label ANN, and evaluation adjacency")
  local dev_graph_features = ivec.create():copy(dev.solutions):bits_extend(dev.tokens, n_labels, n_bns_tokens)
  dev.graph_index = inv.create({ features = n_graph_features, })
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
  local dev_raw_stats = eval.ranking_accuracy({
    raw_codes = dev_predicted_raw, ids = dev.ids, n_dims = train.dims,
    eval_ids = dev_eval_ids, eval_offsets = dev_eval_offsets,
    eval_neighbors = dev_eval_neighbors, eval_weights = dev_eval_weights,
    ranking = cfg.eval.ranking,
  })
  local dev_predicted_raw_score = dev_raw_stats.score
  dev_predicted_raw:destroy()
  dev_predicted_raw = nil
  util.cluster_stats({
    codes = dev_predicted, ids = dev.ids, n_dims = itq_n_bits, knn = cfg.eval.knn,
    eval_offsets = dev_eval_offsets, eval_neighbors = dev_eval_neighbors,
    eval_weights = dev_eval_weights, label = "dev predicted",
  })

  print("\nEvaluating train predicted codes (ITQ binary)")
  local train_full_stats = eval.ranking_accuracy({
    codes = train_predicted,
    ids = train.ids,
    eval_ids = train_eval_ids,
    eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors,
    eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = itq_n_bits,
  })
  str.printf("  Train predicted (full): %.4f\n", train_full_stats.score)

  print("\nEvaluating dev predicted codes (ITQ binary)")
  local dev_full_stats = eval.ranking_accuracy({
    codes = dev_predicted,
    ids = dev.ids,
    eval_ids = dev_eval_ids,
    eval_offsets = dev_eval_offsets,
    eval_neighbors = dev_eval_neighbors,
    eval_weights = dev_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = itq_n_bits,
  })
  str.printf("  Dev predicted (ITQ binary): %.4f\n", dev_full_stats.score)

  train_eval_ids = nil -- luacheck: ignore
  train_eval_offsets = nil -- luacheck: ignore
  train_eval_neighbors = nil -- luacheck: ignore
  train_eval_weights = nil -- luacheck: ignore


  dev_regressor_sentences = nil -- luacheck: ignore
  dev_predicted = nil -- luacheck: ignore
  dev_eval_ids = nil -- luacheck: ignore
  dev_eval_offsets = nil -- luacheck: ignore
  dev_eval_neighbors = nil -- luacheck: ignore
  dev_eval_weights = nil -- luacheck: ignore

  print("\nPredicting test codes")
  local test_regressor_sentences = make_regressor_sentences(test_set.tokens, test_set.n)
  local test_predicted_raw = train.tm:regress(test_regressor_sentences, test_set.n, true)
  local test_predicted = itq_encode(test_predicted_raw)
  util.spot_check_codes(test_predicted, test_set.n, itq_n_bits, "test predicted")
  local test_entropy = eval.entropy_stats(test_predicted, test_set.n, itq_n_bits)
  str.printf("  Test predicted entropy: mean=%.4f min=%.4f max=%.4f std=%.4f\n",
    test_entropy.mean, test_entropy.min, test_entropy.max, test_entropy.std)

  print("\nBuilding test graph_index, label ANN, and evaluation adjacency")
  local test_graph_features = ivec.create():copy(test_set.solutions):bits_extend(test_set.tokens, n_labels, n_bns_tokens)
  test_set.graph_index = inv.create({ features = n_graph_features, })
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
  local test_raw_stats = eval.ranking_accuracy({
    raw_codes = test_predicted_raw, ids = test_set.ids, n_dims = train.dims,
    eval_ids = test_eval_ids, eval_offsets = test_eval_offsets,
    eval_neighbors = test_eval_neighbors, eval_weights = test_eval_weights,
    ranking = cfg.eval.ranking,
  })
  local test_predicted_raw_score = test_raw_stats.score
  test_predicted_raw:destroy()
  test_predicted_raw = nil
  util.cluster_stats({
    codes = test_predicted, ids = test_set.ids, n_dims = itq_n_bits, knn = cfg.eval.knn,
    eval_offsets = test_eval_offsets, eval_neighbors = test_eval_neighbors,
    eval_weights = test_eval_weights, label = "test predicted",
  })

  print("\nEvaluating test predicted codes (ITQ binary)")
  local test_pred_stats = eval.ranking_accuracy({
    codes = test_predicted,
    ids = test_set.ids,
    eval_ids = test_eval_ids,
    eval_offsets = test_eval_offsets,
    eval_neighbors = test_eval_neighbors,
    eval_weights = test_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = itq_n_bits,
  })
  str.printf("  Test ranking score: %.4f\n", test_pred_stats.score)
  test_regressor_sentences = nil -- luacheck: ignore
  test_predicted = nil -- luacheck: ignore

  print("\n" .. string.rep("=", 60))
  print("SUMMARY")
  print(string.rep("=", 60))
  str.printf("  Spectral dims: %d  ITQ bits: %d\n", train.dims, itq_n_bits)
  str.printf("\n  Ranking Accuracy (%s):\n", cfg.eval.ranking)
  str.printf("                                 Train      Dev      Test\n")
  str.printf("    Spectral kernel:            %.4f\n", spectral_metrics.kernel_score)
  str.printf("    Spectral raw:               %.4f\n", spectral_metrics.raw_score)
  str.printf("    Predicted raw:              %.4f   %.4f   %.4f\n",
    train_predicted_raw_score, dev_predicted_raw_score, test_predicted_raw_score)
  str.printf("    Predicted binary (ITQ):    %.4f   %.4f   %.4f\n",
    train_full_stats.score, dev_full_stats.score, test_pred_stats.score)
  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
