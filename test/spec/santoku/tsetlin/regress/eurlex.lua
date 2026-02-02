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
    min_df = 2,
    max_df = 0.5,
    bns_tokens = 65536,
    per_class = 8192,
  },
  regressor = {
    clauses = 256,
    clause_tolerance = { def = 869, min = 8, max = 1024, int = true },
    clause_maximum = { def = 765, min = 8, max = 1024, int = true },
    target = { def = 119, min = 8, max = 1024, int = true },
    specificity = { def = 1182, min = 2, max = 2000 },
    search_rounds = 0,
    search_trials = 20,
    search_iterations = 40,
    final_patience = 20,
    final_batch = 40,
    final_iterations = 400,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 32,
    decay = 1.0,
    bandwidth = 1.0,
    rounds = 0,
    samples = 20,
  },
  rp = {
    max_bits = 1024
  },
  eval = {
    knn = 16,
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
  local token_index = tok:index()
  tok = nil
  train.problems = nil
  dev.problems = nil
  test_set.problems = nil

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
  local min_df = cfg.feature_selection.min_df
  local max_df = cfg.feature_selection.max_df
  local df_ids, idf_weights = train.tokens:bits_top_df(train.n, n_tokens, nil, -min_df, max_df)
  str.printf("  DF filter (min=%d, max=%.0f%%): %d -> %d\n", min_df, max_df * 100, n_tokens, df_ids:size())
  train.tokens:bits_select(df_ids, nil, n_tokens)
  dev.tokens:bits_select(df_ids, nil, n_tokens)
  test_set.tokens:bits_select(df_ids, nil, n_tokens)
  n_tokens = df_ids:size()

  print("\nBNS feature selection")
  local bns_ids, bns_weights = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_labels, nil, cfg.feature_selection.bns_tokens, "max")
  local n_bns_tokens = bns_ids:size()
  str.printf("  BNS selection: %d -> %d tokens\n", n_tokens, n_bns_tokens)

  print("\nComputing IDF×BNS token weights")
  local idf_at_bns = dvec.create():copy(idf_weights, bns_ids)
  local token_weights = dvec.create():copy(idf_at_bns):scalev(bns_weights)
  str.printf("  Token weights: min=%.4f max=%.4f mean=%.4f\n",
    token_weights:min(), token_weights:max(), token_weights:sum() / n_bns_tokens)

  print("\nApplying BNS selection to all splits")
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  dev.tokens:bits_select(bns_ids, nil, n_tokens)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens)
  n_tokens = n_bns_tokens

  local n_graph_features = n_labels + n_bns_tokens

  print("\nBuilding graph_index (labels=rank0, tokens=rank1)")
  local graph_ranks = ivec.create(n_graph_features):fill(1)
  for i = 0, n_labels - 1 do graph_ranks:set(i, 0) end
  local graph_weights = dvec.create(n_graph_features):fill(1.0)
  graph_weights:copy(token_weights, 0, n_bns_tokens, n_labels)
  local graph_features = ivec.create():copy(train.solutions):bits_extend(train.tokens, n_labels, n_bns_tokens)
  train.graph_index = inv.create({
    features = graph_weights,
    ranks = graph_ranks,
    n_ranks = 2,
  })
  train.graph_index:add(graph_features, train.ids)
  train.graph_features = graph_features
  str.printf("  Docs: %d  Features: %d (labels=%d, tokens=%d)\n",
    train.n, n_graph_features, n_labels, n_bns_tokens)

  print("\nBuilding label ANN for evaluation")
  local train_labels_cvec = train.solutions:bits_to_cvec(train.n, n_labels)
  train.label_ann = ann.create({ features = n_labels })
  train.label_ann:add(train_labels_cvec, train.ids)

  print("\nBuilding evaluation adjacency")
  local train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights =
    graph.adjacency({
      knn_index = train.label_ann,
      knn = cfg.eval.knn,
      knn_cache = cfg.eval.knn,
      weight_index = train.graph_index,
      weight_decay = cfg.nystrom.decay,
      weight_bandwidth = cfg.nystrom.bandwidth,
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
    train_tokens = train.graph_features,
    train_ids = train.ids,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    bandwidth = cfg.nystrom.bandwidth,
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
  local spectral_dims = model.dims
  local train_raw_codes = model.raw_codes
  str.printf("  Spectral dims: %d, samples: %d\n", spectral_dims, train.n)

  print("\nOptimizing SignRP configuration")
  local rp_ranks, rp_scores, rp_dims, rp_bits = optimize.rp({
    raw_codes = train_raw_codes,
    ids = train.ids,
    n_samples = train.n,
    n_dims = spectral_dims,
    max_bits = cfg.rp.max_bits,
    tolerance = cfg.rp.tolerance or 0.01,
    eval = {
      ids = train_eval_ids,
      offsets = train_eval_offsets,
      neighbors = train_eval_neighbors,
      weights = train_eval_weights,
    },
    ranking = cfg.eval.ranking,
  })
  local best_idx, _ = rp_ranks:get(0)
  local best_score = rp_scores:get(best_idx)
  local best_dims = rp_dims:get(best_idx)
  local best_bits = rp_bits:get(best_idx)
  str.printf("  Best: dims=%d bits=%d score=%.4f\n", best_dims, best_bits, best_score)
  str.printf("  Scores (by preference):\n")
  for i = 0, rp_ranks:size() - 1 do
    local idx, rank = rp_ranks:get(i)
    local marker = (rank == 0) and " *" or ""
    str.printf("    dims=%d bits=%d: %.4f%s\n", rp_dims:get(idx), rp_bits:get(idx), rp_scores:get(idx), marker)
  end

  rp_ranks:destroy()
  rp_scores:destroy()
  rp_dims:destroy()
  rp_bits:destroy()

  print("\nTruncating spectral codes to best dims and creating SignRP encoder")
  if best_dims < spectral_dims then
    local truncated = dvec.create()
    local selected_cols = ivec.create(best_dims):fill_indices()
    train_raw_codes:mtx_select(selected_cols, nil, spectral_dims, truncated)
    selected_cols:destroy()
    train_raw_codes:destroy()
    train_raw_codes = truncated
  end
  train.dims = best_dims
  train.rp_bits = best_bits

  print("\nFeature selection for regressor (F-score)")
  local union_feat_ids, _, class_offsets, class_feat_ids, class_scores = train.tokens:bits_top_reg_f(
    train_raw_codes, train.n, n_tokens, train.dims, cfg.feature_selection.per_class)
  str.printf("  Features: %d union, %d grouped (%d per dim x %d dims)\n",
    union_feat_ids:size(), class_feat_ids:size(), cfg.feature_selection.per_class, train.dims)
  str.printf("  F-scores: min=%.2f max=%.2f mean=%.2f\n",
    class_scores:min(), class_scores:max(), class_scores:sum() / class_scores:size())

  print("\nTop 5 features per dimension:")
  for d = 1, train.dims do
    local start_idx = class_offsets:get(d - 1)
    local end_idx = class_offsets:get(d)
    local top_n = math.min(5, end_idx - start_idx)
    local feats = {}
    for i = 0, top_n - 1 do
      local fid = class_feat_ids:get(start_idx + i)
      local bns_id = bns_ids:get(fid)
      local orig_id = df_ids:get(bns_id)
      feats[i + 1] = token_index[orig_id + 1] or tostring(orig_id)
    end
    str.printf("  dim_%02d: %s\n", d - 1, table.concat(feats, ", "))
  end

  local make_regressor_sentences = function(tokens, n_samples)
    return tokens:bits_to_cvec(n_samples, n_tokens, class_offsets, class_feat_ids, true)
  end

  local train_regressor_sentences, max_k = make_regressor_sentences(train.tokens, train.n)
  local n_regressor_features = max_k
  str.printf("  Regressor features: %d\n", n_regressor_features)

  print("\nTraining TM regressor")
  local predicted_buf = dvec.create()
  train.tm = optimize.regressor({
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
  })

  print("\nPredicting train codes")
  local train_predicted_raw = train.tm:regress(train_regressor_sentences, train.n, true)
  local train_raw_stats = eval.ranking_accuracy({
    raw_codes = train_predicted_raw, ids = train.ids, n_dims = train.dims,
    eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
  })

  print("\nNormalizing and creating SignRP encoder")
  local normalize = hlth.normalizer({
    codes = train_predicted_raw,
    n_dims = train.dims,
    n_samples = train.n,
  })
  local train_predicted_norm = normalize(train_predicted_raw)

  local rp_encode, rp_n_bits = hlth.rp_encoder({
    n_dims = train.dims,
    rp_dims = train.rp_bits,
  })

  local train_predicted_rp = rp_encode(train_predicted_norm)
  train_predicted_raw:destroy()
  train_predicted_norm:destroy()

  local train_rp_entropy = eval.entropy_stats(train_predicted_rp, train.n, rp_n_bits)
  str.printf("  Train RP entropy: mean=%.4f min=%.4f max=%.4f std=%.4f\n",
    train_rp_entropy.mean, train_rp_entropy.min, train_rp_entropy.max, train_rp_entropy.std)

  local train_rp_stats = eval.ranking_accuracy({
    codes = train_predicted_rp, ids = train.ids, n_dims = rp_n_bits,
    eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
  })

  local function evaluate_split(split, name)
    print("\nEvaluating " .. name)
    local sentences = make_regressor_sentences(split.tokens, split.n)
    local predicted_raw = train.tm:regress(sentences, split.n, true)
    local predicted_norm = normalize(predicted_raw)
    local predicted_rp = rp_encode(predicted_norm)

    local local_ids = ivec.create(split.n):fill_indices()
    local split_graph_features = ivec.create():copy(split.solutions):bits_extend(
      split.graph_tokens or split.tokens, n_labels, n_bns_tokens)
    local split_graph_index = inv.create({
      features = graph_weights,
      ranks = graph_ranks,
      n_ranks = 2,
    })
    split_graph_index:add(split_graph_features, local_ids)

    local labels_cvec = split.solutions:bits_to_cvec(split.n, n_labels)
    local label_ann = ann.create({ features = n_labels })
    label_ann:add(labels_cvec, local_ids)

    local eval_ids, eval_offsets, eval_neighbors, eval_weights = graph.adjacency({
      knn_index = label_ann,
      knn = cfg.eval.knn,
      knn_cache = cfg.eval.knn,
      weight_index = split_graph_index,
      weight_decay = cfg.nystrom.decay,
      weight_bandwidth = cfg.nystrom.bandwidth,
      bridge = "none",
    })

    if cfg.verbose then
      local split_spot_ids = {}
      for i = 1, 5 do
        split_spot_ids[i] = eval_ids:get(math.random(eval_ids:size()) - 1)
      end
      util.spot_check_adjacency(eval_ids, eval_offsets, eval_neighbors, eval_weights, name .. " eval")
      util.spot_check_neighbors_with_labels(eval_ids, eval_offsets, eval_neighbors, eval_weights,
        split.label_csr, 0, name .. " eval adj", split_spot_ids, 10)
    end

    local raw_stats = eval.ranking_accuracy({
      raw_codes = predicted_raw, ids = local_ids, n_dims = train.dims,
      eval_ids = eval_ids, eval_offsets = eval_offsets,
      eval_neighbors = eval_neighbors, eval_weights = eval_weights,
      ranking = cfg.eval.ranking,
    })

    local rp_stats = eval.ranking_accuracy({
      codes = predicted_rp, ids = local_ids, n_dims = rp_n_bits,
      eval_ids = eval_ids, eval_offsets = eval_offsets,
      eval_neighbors = eval_neighbors, eval_weights = eval_weights,
      ranking = cfg.eval.ranking,
    })

    local rp_entropy = eval.entropy_stats(predicted_rp, split.n, rp_n_bits)
    str.printf("  %s RP entropy: mean=%.4f min=%.4f max=%.4f std=%.4f\n",
      name, rp_entropy.mean, rp_entropy.min, rp_entropy.max, rp_entropy.std)

    predicted_raw:destroy()
    predicted_norm:destroy()
    predicted_rp:destroy()

    return raw_stats.score, rp_stats.score
  end

  local dev_raw, dev_rp = evaluate_split(dev, "dev")
  local test_raw, test_rp = evaluate_split(test_set, "test")

  print("\n" .. string.rep("=", 60))
  print("SUMMARY")
  print(string.rep("=", 60))
  str.printf("  Spectral dims: %d  RP bits: %d\n", train.dims, rp_n_bits)
  str.printf("\n  Ranking Accuracy (%s):\n", cfg.eval.ranking)
  str.printf("                          Train      Dev      Test\n")
  str.printf("    Spectral kernel:     %.4f\n", spectral_metrics.kernel_score)
  str.printf("    Spectral raw:        %.4f\n", spectral_metrics.raw_score)
  str.printf("    Spectral SignRP:     %.4f\n", best_score)
  str.printf("    Predicted raw:       %.4f   %.4f   %.4f\n",
    train_raw_stats.score, dev_raw, test_raw)
  str.printf("    Predicted SignRP:    %.4f   %.4f   %.4f\n",
    train_rp_stats.score, dev_rp, test_rp)
  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
