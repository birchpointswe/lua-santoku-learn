local dvec = require("santoku.dvec")
local cvec = require("santoku.cvec")
local test = require("santoku.test")
local ds = require("santoku.tsetlin.dataset")
local utc = require("santoku.utc")
local ivec = require("santoku.ivec")
local str = require("santoku.string")
local eval = require("santoku.tsetlin.evaluator")
local inv = require("santoku.tsetlin.inv")
local ann = require("santoku.tsetlin.ann")
local graph = require("santoku.tsetlin.graph")
local optimize = require("santoku.tsetlin.optimize")
local tokenizer = require("santoku.tokenizer")
local hlth = require("santoku.tsetlin.hlth")

local cfg; cfg = {
  embeddings = "spectral",
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
    min_df = -2,
    max_df = 0.98,
    max_vocab = nil,
  },
  retrieval = {
    knn = 64,
  },
  regressor = {
    n_bins = 8,
    clauses = { def = 64, min = 16, max = 256, round = 8 },
    clause_tolerance = { def = 32, min = 8, max = 128, int = true },
    clause_maximum = { def = 64, min = 16, max = 128, int = true },
    target = { def = 16, min = 4, max = 64, int = true },
    specificity = { def = 400, min = 50, max = 2000 },
    include_bits = { def = 2, min = 1, max = 6, int = true },
    search_patience = 5,
    search_rounds = 3,
    search_trials = 6,
    search_iterations = 20,
    final_patience = 20,
    final_iterations = 100,
  },
  search = {
    rounds = 0,
    adjacency_samples = 4,
    spectral_samples = 4,
    prone_samples = 10,
    select_samples = 4,
    eval_samples = 4,
    adjacency = {
      spectral = {
        knn = { def = 20, min = 10, max = 40, int = true },
        knn_alpha = { def = 12, min = 8, max = 20, int = true },
        weight_decay = { def = 3.0, min = 2, max = 6 },
        knn_mutual = { def = true, false, true },
        knn_mode = "cknn",
        knn_cache = 64,
        bridge = "mst",
      },
      prone = {
        knn = { def = 20, min = 8, max = 64, int = true },
        knn_alpha = { def = 10, min = 2, max = 32, int = true },
        weight_decay = { def = 2.5, min = 2, max = 6 },
        knn_mutual = { def = false, false, true },
        knn_mode = "cknn",
        knn_cache = 64,
        bridge = "mst",
      },
    },
    seeds = {
      enabled = true,
      knn = 64,
      knn_cache = 64,
    },
    spectral = {
      laplacian = "unnormalized",
      n_dims = 64,
      eps = 1e-8,
      threshold = {
        method = "itq",
        itq_iterations = 500,
        itq_tolerance = 1e-8,
      },
    },
    prone = {
      n_dims = { def = 32, min = 8, max = 64, int = true, round = 8 },
      n_iter = { def = 5, min = 3, max = 10, int = true },
      step = { def = 10, min = 5, max = 20, int = true },
      mu = { def = 0.2, min = 0.0, max = 0.5 },
      theta = { def = 0.5, min = 0.1, max = 1.0 },
      neg_samples = { def = 5, min = 1, max = 10, int = true },
      propagate = true,
      threshold = {
        method = "itq",
        itq_iterations = 500,
        itq_tolerance = 1e-8,
      },
    },
    eval = {
      knn = 32,
      anchors = 16,
      pairs = 64,
      ranking = "ndcg",
      metric = "avg",
    },
    verbose = true,
  },
}

test("eurlex-multilabel", function()

  local stopwatch = utc.stopwatch()

  print("Reading EURLEX57K data")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels

  str.printf("  Train: %6d (%d labels)\n", train.n, n_labels)
  str.printf("  Dev:   %6d\n", dev.n)
  str.printf("  Test:  %6d\n", test_set.n)

  print("\nTraining tokenizer")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  str.printf("  Vocabulary: %d tokens\n", n_tokens)

  print("\nTokenizing")
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)

  print("\nFeature selection (IDF filtering)")
  local idf_sorted, idf_weights = train.tokens:bits_top_df(
    train.n, n_tokens, cfg.feature_selection.max_vocab,
    cfg.feature_selection.min_df, cfg.feature_selection.max_df)
  local n_top_v = idf_sorted:size()
  str.printf("  DF filtered: %d features\n", n_top_v)
  str.printf("  IDF range: %.3f - %.3f\n", idf_weights:min(), idf_weights:max())
  tok:restrict(idf_sorted)

  print("\nRe-tokenizing with IDF-filtered vocabulary")
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)

  print("\nCreating document IDs")
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  dev.ids = ivec.create(dev.n)
  dev.ids:fill_indices()
  dev.ids:add(train.n)
  test_set.ids = ivec.create(test_set.n)
  test_set.ids:fill_indices()
  test_set.ids:add(train.n + dev.n)
  local n_docs = train.n + dev.n + test_set.n

  print("\nBuilding heterogeneous graph (documents + labels)")
  local n_graph_features = n_top_v + n_labels
  do
    local graph_problems = ivec.create()
    train.tokens:bits_select(nil, train.ids, n_top_v, graph_problems)
    graph_problems:bits_extend(train.solutions, n_top_v, n_labels)
    local graph_weights = dvec.create(n_graph_features)
    for i = 0, n_top_v - 1 do
      graph_weights:set(i, idf_weights:get(i))
    end
    graph_weights:fill(1.0, n_top_v, n_graph_features)
    local graph_ranks = ivec.create(n_graph_features)
    graph_ranks:fill(1, 0, n_top_v)
    graph_ranks:fill(0, n_top_v, n_graph_features)
    train.index_graph = inv.create({
      features = graph_weights,
      ranks = graph_ranks,
      n_ranks = 2,
    })
    train.index_graph:add(graph_problems, train.ids)
    graph_problems:destroy()
    str.printf("  Graph: %d document nodes, %d features (%d tokens + %d labels)\n",
      train.n, n_graph_features, n_top_v, n_labels)
  end

  print("\nAdding label nodes to graph")
  do
    local label_features = ivec.create()
    for i = 0, n_labels - 1 do
      label_features:push(i * n_graph_features + n_top_v + i)
    end
    local label_ids = ivec.create(n_labels)
    label_ids:fill_indices()
    label_ids:add(n_docs)
    train.index_graph:add(label_features, label_ids)
    label_features:destroy()
    label_ids:destroy()
    str.printf("  Added %d label nodes (IDs %d to %d)\n", n_labels, n_docs, n_docs + n_labels - 1)
  end

  print("\nBuilding knn index (IDF-weighted tokens only)")
  train.node_features_graph = inv.create({ features = idf_weights })
  train.node_features_graph:add(train.tokens, train.ids)
  str.printf("  KNN index: %d tokens with IDF weights\n", n_top_v)

  local function build_label_ground_truth (ids, solutions, n_samples)
    local offsets, neighbors = solutions:bits_to_csr(n_samples, n_labels)
    return {
      ids = ids,
      offsets = offsets,
      neighbors = neighbors,
    }
  end

  print("\nBuilding label ground truth (CSR format)")
  train.label_csr = build_label_ground_truth(train.ids, train.solutions, train.n)
  dev.label_csr = build_label_ground_truth(dev.ids, dev.solutions, dev.n)
  test_set.label_csr = build_label_ground_truth(test_set.ids, test_set.solutions, test_set.n)
  str.printf("  Train: %d labels total\n", train.label_csr.neighbors:size())
  str.printf("  Dev:   %d labels total\n", dev.label_csr.neighbors:size())
  str.printf("  Test:  %d labels total\n", test_set.label_csr.neighbors:size())

  local adj_cfg = cfg.search.adjacency[cfg.embeddings]
  local model, best_params, best_metrics

  local function build_ranking_ground_truth (knn_index, ids, knn)
    local adj_ids, adj_offsets, adj_neighbors, adj_weights = graph.adjacency({
      knn_index = knn_index,
      seed_ids = ids,
      knn = knn,
      knn_cache = knn,
      bridge = "none",
    })
    return {
      ids = adj_ids,
      offsets = adj_offsets,
      neighbors = adj_neighbors,
      weights = adj_weights,
    }
  end

  print("\nBuilding ranking ground truth for embedding eval")
  train.ranking_gt = build_ranking_ground_truth(train.index_graph, train.ids, cfg.search.eval.knn)

  if cfg.embeddings == "spectral" then
    print("\nOptimizing spectral pipeline on heterogeneous graph")
    model, best_params, best_metrics = optimize.spectral({
      index = train.index_graph,
      knn_index = train.node_features_graph,
      search_rounds = cfg.search.rounds,
      adjacency_samples = cfg.search.adjacency_samples,
      spectral_samples = cfg.search.spectral_samples,
      select_samples = cfg.search.select_samples,
      eval_samples = cfg.search.eval_samples,
      adjacency = adj_cfg,
      spectral = cfg.search.spectral,
      eval = cfg.search.eval,
      expected_ids = train.ranking_gt.ids,
      expected_offsets = train.ranking_gt.offsets,
      expected_neighbors = train.ranking_gt.neighbors,
      expected_weights = train.ranking_gt.weights,
      each = cfg.search.verbose and function (info)
        if info.event == "round_start" then
          str.printf("\n[SPECTRAL R%d] Starting\n", info.round)
        elseif info.event == "round_end" then
          str.printf("[SPECTRAL R%d] best=%.4f global=%.4f\n",
            info.round, info.round_best_score, info.global_best_score)
        elseif info.event == "stage" and info.stage == "adjacency" then
          local p = info.params.adjacency
          local phase = info.is_final and "F" or str.format("R%d S%d", info.round, info.sample)
          str.printf("[SPECTRAL %s ADJ] knn=%d alpha=%d decay=%.2f\n",
            phase, p.knn, p.knn_alpha, p.weight_decay)
        elseif info.event == "adjacency_result" then
          str.printf("[SPECTRAL] nodes=%d edges=%d\n", info.n_nodes, info.n_edges)
        elseif info.event == "eval" then
          str.printf("[SPECTRAL] score=%.4f\n", info.score)
        end
      end or nil,
    })
    str.printf("\nSpectral: dims=%d score=%.4f\n", model.dims, best_metrics.score)
  else
    local seed_ids, seed_offsets, seed_neighbors
    if cfg.search.seeds and cfg.search.seeds.enabled then
      print("\nPrecomputing seed edges for ProNE")
      seed_ids, seed_offsets, seed_neighbors = graph.adjacency({
        index = train.index_graph,
        knn_index = train.node_features_graph,
        knn = cfg.search.seeds.knn,
        knn_cache = cfg.search.seeds.knn_cache,
      })
      str.printf("  Seeds: %d nodes, %d edges\n", seed_ids:size(), seed_neighbors:size())
    end

    print("\nOptimizing ProNE pipeline on heterogeneous graph")
    model, best_params, best_metrics = optimize.prone({
      index = train.index_graph,
      knn_index = train.node_features_graph,
      seed_ids = seed_ids,
      seed_offsets = seed_offsets,
      seed_neighbors = seed_neighbors,
      search_rounds = cfg.search.rounds,
      adjacency_samples = cfg.search.adjacency_samples,
      prone_samples = cfg.search.prone_samples,
      select_samples = cfg.search.select_samples,
      eval_samples = cfg.search.eval_samples,
      adjacency = adj_cfg,
      prone = cfg.search.prone,
      eval = cfg.search.eval,
      expected_ids = train.ranking_gt.ids,
      expected_offsets = train.ranking_gt.offsets,
      expected_neighbors = train.ranking_gt.neighbors,
      expected_weights = train.ranking_gt.weights,
      each = cfg.search.verbose and function (info)
        if info.event == "round_start" then
          str.printf("\n[PRONE R%d] Starting\n", info.round)
        elseif info.event == "round_end" then
          str.printf("[PRONE R%d] best=%.4f global=%.4f\n",
            info.round, info.round_best_score, info.global_best_score)
        elseif info.event == "stage" and info.stage == "prone" then
          local p = info.params.prone
          str.printf("[PRONE] dims=%d step=%d mu=%.2f theta=%.2f\n",
            p.n_dims, p.step or 10, p.mu or 0.2, p.theta or 0.5)
        elseif info.event == "eval" then
          str.printf("[PRONE] score=%.4f\n", info.score)
        end
      end or nil,
    })
    str.printf("\nProNE: dims=%d score=%.4f\n", model.dims, best_metrics.score)
  end

  train.dims = model.dims
  train.index = model.index

  print("\nExtracting label embeddings")
  local label_ids = ivec.create(n_labels)
  label_ids:fill_indices()
  label_ids:add(n_docs)
  local label_codes = train.index:get(label_ids)
  str.printf("  Label codes: %d labels × %d dims\n", n_labels, train.dims)

  print("\nBuilding label ANN")
  local label_ann = ann.create({ features = train.dims, expected_size = n_labels })
  label_ann:add(label_codes, label_ids)

  local function get_label_hoods (split, split_name)
    str.printf("\nQuerying label ANN for %s documents\n", split_name)
    local doc_codes = train.index:get(split.ids)
    local hood_ids, hoods = label_ann:neighborhoods_by_vecs(doc_codes, cfg.retrieval.knn)
    str.printf("  Retrieved %d hoods of size %d\n", hoods.n, cfg.retrieval.knn)
    doc_codes:destroy()
    return hood_ids, hoods
  end

  train.hood_ids, train.hoods = get_label_hoods(train, "train")
  dev.hood_ids, dev.hoods = get_label_hoods(dev, "dev")
  test_set.hood_ids, test_set.hoods = get_label_hoods(test_set, "test")

  print("\nComputing optimal k values (training targets)")
  local train_exp_neighbors_shifted = ivec.create()
  train_exp_neighbors_shifted:copy(train.label_csr.neighbors)
  train_exp_neighbors_shifted:add(n_docs)
  local train_ks = eval.retrieval_ks({
    hoods = train.hoods,
    hood_ids = train.hood_ids,
    expected_offsets = train.label_csr.offsets,
    expected_neighbors = train_exp_neighbors_shifted,
  })
  train_exp_neighbors_shifted:destroy()

  local dev_exp_neighbors_shifted = ivec.create()
  dev_exp_neighbors_shifted:copy(dev.label_csr.neighbors)
  dev_exp_neighbors_shifted:add(n_docs)
  local dev_ks = eval.retrieval_ks({
    hoods = dev.hoods,
    hood_ids = dev.hood_ids,
    expected_offsets = dev.label_csr.offsets,
    expected_neighbors = dev_exp_neighbors_shifted,
  })
  dev_exp_neighbors_shifted:destroy()

  local test_exp_neighbors_shifted = ivec.create()
  test_exp_neighbors_shifted:copy(test_set.label_csr.neighbors)
  test_exp_neighbors_shifted:add(n_docs)
  local test_ks = eval.retrieval_ks({
    hoods = test_set.hoods,
    hood_ids = test_set.hood_ids,
    expected_offsets = test_set.label_csr.offsets,
    expected_neighbors = test_exp_neighbors_shifted,
  })
  test_exp_neighbors_shifted:destroy()

  str.printf("  Train k range: %d - %d (mean %.1f)\n",
    train_ks:min(), train_ks:max(), train_ks:to_dvec():mean())
  str.printf("  Dev k range:   %d - %d (mean %.1f)\n",
    dev_ks:min(), dev_ks:max(), dev_ks:to_dvec():mean())
  str.printf("  Test k range:  %d - %d (mean %.1f)\n",
    test_ks:min(), test_ks:max(), test_ks:to_dvec():mean())

  print("\nEncoding KNN features (similarities mode)")
  local hood_encoder = hlth.landmark_encoder({
    mode = "similarities",
    n_landmarks = cfg.retrieval.knn,
    n_bins = cfg.regressor.n_bins,
  })
  local train_features = hood_encoder(train.hoods)
  local dev_features = hood_encoder(dev.hoods)
  local test_features = hood_encoder(test_set.hoods)
  local n_regressor_features = cfg.retrieval.knn * cfg.regressor.n_bins
  str.printf("  Features: %d bins × %d positions = %d bits\n",
    cfg.regressor.n_bins, cfg.retrieval.knn, n_regressor_features)

  print("\nTraining k-regressor")
  local train_targets = train_ks:to_dvec()
  local dev_targets = dev_ks:to_dvec()
  local regressor = optimize.regressor({
    features = n_regressor_features,
    outputs = 1,
    samples = train.n,
    problems = train_features,
    targets = train_targets,
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
    search_metric = function (r)
      local predicted = r:predict(dev_features, dev.n)
      local stats = eval.regression_accuracy(predicted, dev_targets)
      return -stats.mean, stats
    end,
    each = function (_, is_final, val_stats, params, epoch, round, trial)
      local d, dd = stopwatch()
      local phase = is_final and "F" or str.format("R%d T%d", round, trial)
      str.printf("[REGRESS %s E%d] C=%d T=%d S=%.0f ACC=%.1f%% (%.2fs +%.2fs)\n",
        phase, epoch, params.clauses, params.target, params.specificity,
        (1 - val_stats.nmae) * 100, d, dd)
    end,
  })

  print("\nPredicting k values")
  local train_pred_ks = regressor:predict(train_features, train.n)
  local dev_pred_ks = regressor:predict(dev_features, dev.n)
  local test_pred_ks = regressor:predict(test_features, test_set.n)

  train_pred_ks = train_pred_ks:round():to_ivec()
  dev_pred_ks = dev_pred_ks:round():to_ivec()
  test_pred_ks = test_pred_ks:round():to_ivec()

  train_pred_ks:clamp(1, cfg.retrieval.knn)
  dev_pred_ks:clamp(1, cfg.retrieval.knn)
  test_pred_ks:clamp(1, cfg.retrieval.knn)

  str.printf("  Train predicted k: %d - %d (mean %.1f)\n",
    train_pred_ks:min(), train_pred_ks:max(), train_pred_ks:to_dvec():mean())
  str.printf("  Dev predicted k:   %d - %d (mean %.1f)\n",
    dev_pred_ks:min(), dev_pred_ks:max(), dev_pred_ks:to_dvec():mean())
  str.printf("  Test predicted k:  %d - %d (mean %.1f)\n",
    test_pred_ks:min(), test_pred_ks:max(), test_pred_ks:to_dvec():mean())

  print("\nTruncating hoods to predicted k")
  train.hoods:select_topk(train_pred_ks)
  dev.hoods:select_topk(dev_pred_ks)
  test_set.hoods:select_topk(test_pred_ks)

  print("\nConverting hoods to CSR for evaluation")
  local function hoods_to_csr (hoods, hood_ids, ids)
    local adj_offsets, adj_neighbors = graph.adj_hoods(ids, hoods, train.dims)
    adj_neighbors:lookup(hood_ids)
    adj_neighbors:add(-n_docs)
    return adj_offsets, adj_neighbors
  end

  local train_pred_offsets, train_pred_neighbors = hoods_to_csr(train.hoods, train.hood_ids, train.ids)
  local dev_pred_offsets, dev_pred_neighbors = hoods_to_csr(dev.hoods, dev.hood_ids, dev.ids)
  local test_pred_offsets, test_pred_neighbors = hoods_to_csr(test_set.hoods, test_set.hood_ids, test_set.ids)

  print("\nEvaluating multi-label retrieval")
  local function eval_retrieval (pred_offsets, pred_neighbors, label_csr, label)
    local stats = eval.retrieval_accuracy({
      predicted_offsets = pred_offsets,
      predicted_neighbors = pred_neighbors,
      expected_offsets = label_csr.offsets,
      expected_neighbors = label_csr.neighbors,
      metric = "f1",
    })
    str.printf("  %s: micro_f1=%.4f micro_prec=%.4f micro_rec=%.4f\n",
      label, stats.micro_f1, stats.micro_precision, stats.micro_recall)
    str.printf("         macro_f1=%.4f macro_prec=%.4f macro_rec=%.4f\n",
      stats.macro_f1, stats.macro_precision, stats.macro_recall)
    return stats
  end

  local train_stats = eval_retrieval(train_pred_offsets, train_pred_neighbors, train.label_csr, "Train")
  local dev_stats = eval_retrieval(dev_pred_offsets, dev_pred_neighbors, dev.label_csr, "Dev  ")
  local test_stats = eval_retrieval(test_pred_offsets, test_pred_neighbors, test_set.label_csr, "Test ")

  str.printf("\nFinal Results (Micro F1):\n")
  str.printf("  Train: %.2f%%\n", train_stats.micro_f1 * 100)
  str.printf("  Dev:   %.2f%%\n", dev_stats.micro_f1 * 100)
  str.printf("  Test:  %.2f%%\n", test_stats.micro_f1 * 100)

  collectgarbage("collect")

end)
