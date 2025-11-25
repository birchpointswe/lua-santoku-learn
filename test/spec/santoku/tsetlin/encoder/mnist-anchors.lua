require("santoku.dvec")
local cvec = require("santoku.cvec")
local test = require("santoku.test")
local ds = require("santoku.tsetlin.dataset")
local utc = require("santoku.utc")
local ivec = require("santoku.ivec")
local str = require("santoku.string")
local eval = require("santoku.tsetlin.evaluator")
local inv = require("santoku.tsetlin.inv")
local ann = require("santoku.tsetlin.ann")
local itq = require("santoku.tsetlin.itq")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local simhash = require("santoku.tsetlin.simhash")
local hlth = require("santoku.tsetlin.hlth")
local tm = require("santoku.tsetlin")

local cfg; cfg = {
  data = {
    ttr = 0.9,
    max = nil,
    max_class = nil,
    visible = 784,
    mode = "spectral", -- "spectral" or "simhash"
  },
  spectral = {
    laplacian = "unnormalized",
    n_dims = 24,
    eps = 1e-8,
    threshold = function (codes, dims)
      local codes, ids, vals = itq.otsu({
        codes = codes,
        n_dims = dims,
        metric = "variance",
        minimize = false,
      })
      print("\nThreshold scores")
      local _, idx = vals:scores_lmethod()
      for i = 0, vals:size() - 1 do
        str.printf("  Rank = %3d  Eig = %3d  Score: %.8f  %s\n", i, ids:get(i), vals:get(i), i == idx and "<- elbow" or "")
      end
      return codes, dims
    end,
    select = function (codes, n, dims)
      local eids, escores = codes:mtx_top_entropy(n, dims)
      print("\nRanked Eigenvectors")
      local _, eidx = escores:scores_lmethod()
      for i = 0, dims - 1 do
        str.printf("  Rank = %3d  Eig = %3d  Score: %.8f  %s\n", i, eids:get(i), escores:get(i), i == eidx and "<- elbow" or "")
      end
      eids:setn(eidx)
      return eids
    end,
  },
  simhash = {
    n_dims =  64,
    ranks = nil,
    quantiles = nil,
  },
  graph = {
    decay = 8.0,
    knn = 24,
    knn_mode = "cknn",
    knn_alpha = 2.8,
    knn_mutual = nil,
    knn_min = nil,
    knn_cache = nil,
    bridge = "mst",
  },
  ann = {
    bucket_size = nil,
  },
  eval = {
    knn = 32  ,
    anchors = 16,
    pairs = 16,
    retrieval_metric = "min",
    clustering_metric = "min",
    verbose = true,
    retrieval = function (d)
      local idx = d:scores_plateau(1e-4)
      return idx - 1
    end,
    clustering = function (d)
      local idx = d:scores_plateau(1e-4)
      return idx - 1
    end,
  },
  cluster = {
    enabled = true,
    knn = 64,
  },
  encoder = {
    enabled = false,
  },
  tm = {
    clauses = { def = 8, min = 8, max = 32, int = true, log = true, pow2 = true },
    clause_tolerance = { def = 8, min = 8, max = 256, int = true, log = true, pow2 = true },
    clause_maximum = { def = 8, min = 8, max = 256, int = true, log = true, pow2 = true },
    target = { def = 4, min = 2, max = 256, int = true, log = true, pow2 = true },
    specificity = { def = 10, min = 2, max = 400, int = true, log = true },
  },
  search = {
    patience = 3,
    rounds = 20,
    trials = 4,
    tolerance = 1e-6,
    iterations = 10,
  },
  training = {
    patience = 10,
    iterations = 200,
  },
}

test("mnist-anchors", function()

  local stopwatch = utc.stopwatch()

  -- Load MNIST dataset
  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.visible, cfg.data.max, cfg.data.max_class)
  dataset.n_visible = cfg.data.visible
  dataset.n_hidden = cfg.data.mode == "spectral" and cfg.spectral.n_dims or cfg.simhash.n_dims

  print("\nSplitting")
  local train, test = ds.split_binary_mnist(dataset, cfg.data.ttr)
  str.printf("  Train: %6d\n", train.n)
  str.printf("  Test:  %6d\n", test.n)

  -- Build pixel similarity graph for spectral encoding
  print("\nBuilding pixel similarity graph")
  do
    local graph_ids = ivec.create()
    graph_ids:copy(train.ids)
    local graph_problems = ivec.create()
    dataset.problems:bits_select(nil, graph_ids, cfg.data.visible, graph_problems)
    -- Add category labels with high priority for graph construction
    local problems_ext = ivec.create()
    problems_ext:copy(train.solutions)
    problems_ext:add_scaled(10)
    graph_problems:bits_extend(problems_ext, cfg.data.visible, 10)
    local graph_ranks = ivec.create(cfg.data.visible + 10)
    graph_ranks:fill(1, 0, cfg.data.visible)
    graph_ranks:fill(0, cfg.data.visible, cfg.data.visible + 10)
    train.index_graph = inv.create({
      features = cfg.data.visible + 10,
      ranks = graph_ranks,
      n_ranks = 2,
      decay = cfg.graph.decay
    })
    train.index_graph:add(graph_problems, graph_ids)
  end

  if cfg.data.mode == "spectral" or cfg.encoder.enabled then
    -- Build node features index from raw pixels
    print("\nBuilding node features index (raw pixels)")
    train.node_features = ann.create({ features = dataset.n_visible, expected_size = train.n, bucket_size = cfg.ann.bucket_size })
    train.train_raw = ivec.create()
    dataset.problems:bits_select(nil, train.ids, dataset.n_visible, train.train_raw)
    train.train_raw = train.train_raw:bits_to_cvec(train.n, dataset.n_visible)
    train.node_features:add(train.train_raw, train.ids)
  end

  -- Encode using spectral or simhash
  if cfg.data.mode == "spectral" then
    -- Create adjacency for spectral encoding
    print("\nCreating adjacency for spectral\n")
    do
      train.adj_ids_spectral,
      train.adj_offsets_spectral,
      train.adj_neighbors_spectral,
      train.adj_weights_spectral = graph.adjacency({
        weight_index = train.index_graph,
        knn_index = train.node_features,
        knn = cfg.graph.knn,
        knn_mode = cfg.graph.knn_mode,
        knn_alpha = cfg.graph.knn_alpha,
        knn_cache = cfg.graph.knn_cache,
        knn_mutual = cfg.graph.knn_mutual,
        knn_min = cfg.graph.knn_min,
        bridge = cfg.graph.bridge,
        each = function (ns, cs, es, stg)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | Stage: %-10s  Nodes: %7d  Components: %5d  Edges: %8d\n",
            d, dd, stg, ns, cs, es)
        end
      })
    end
    if train.adj_weights_spectral then
      str.printf("\n  Min: %f  Max: %f  Mean: %f\n",
        train.adj_weights_spectral:min(),
        train.adj_weights_spectral:max(),
        train.adj_weights_spectral:sum() / train.adj_weights_spectral:size())
    end
    print("\nSpectral eigendecomposition")
    train.ids_spectral, train.codes_spectral, train.scale_spectral, train.eigs_spectral = spectral.encode({
      type = cfg.spectral.laplacian,
      n_hidden = cfg.spectral.n_dims,
      eps = cfg.spectral.eps,
      ids = train.adj_ids_spectral,
      offsets = train.adj_offsets_spectral,
      neighbors = train.adj_neighbors_spectral,
      weights = train.adj_weights_spectral,
      each = function (t, s, v, k)
        local d, dd = stopwatch()
        if t == "done" then
          str.printf("  Time: %6.2f %6.2f  Stage: %s  matvecs = %d\n", d, dd, t, s)
        elseif t == "eig" then
          local gap = train.eig_last and v - train.eig_last or 0
          train.eig_last = v
          str.printf("  Time: %6.2f %6.2f  Stage: %s  %3d = %.8f   Gap = %.8f   %s\n",
            d, dd, t, s, v, gap, k and "" or "drop")
        else
          str.printf("  Time: %6.2f %6.2f  Stage: %s\n", d, dd, t)
        end
      end
    })
    train.dims_spectral = cfg.spectral.n_dims
    if cfg.spectral.select then
      local kept = cfg.spectral.select(train.codes_spectral, train.adj_ids_spectral:size(), train.dims_spectral)
      train.codes_spectral:mtx_select(kept, nil, train.dims_spectral)
      train.dims_spectral = kept:size()
    end
    print("\nThresholding")
    if cfg.spectral.threshold then
      train.codes_spectral = cfg.spectral.threshold(train.codes_spectral, train.dims_spectral)
    else
      train.codes_spectral = itq.median({ codes = train.codes_spectral, n_dims = train.dims_spectral })
    end
  elseif cfg.data.mode == "simhash" then
    print("\nSimhashing")
    local idx = inv.create({ features = 10, expected_size = train.n })
    local data = ivec.create()
    data:copy(train.solutions)
    data:add_scaled(10)
    idx:add(data, train.ids)
    data:destroy()
    train.ids_spectral, train.codes_spectral = simhash.encode(idx, cfg.simhash.n_dims, cfg.simhash.ranks, cfg.simhash.quantiles)
    train.dims_spectral = cfg.simhash.n_dims
    idx:destroy()
  else
    error("Unknown mode: " .. cfg.data.mode)
  end

  print("\nCreating class index")
  do
    train.cat_index = inv.create({ features = 10 })
    local data = ivec.create()
    data:copy(train.solutions)
    data:add_scaled(10)
    train.cat_index:add(data, train.ids_spectral)
    data:destroy()
  end

  -- Build category-based ground truth adjacency
  print("\nBuilding expected adjacency (category-based)")
  do
    train.adj_expected_ids,
    train.adj_expected_offsets,
    train.adj_expected_neighbors,
    train.adj_expected_weights = graph.adjacency({
      category_index = train.cat_index,
      category_anchors = cfg.eval.anchors,
      random_pairs = cfg.eval.pairs,
    })
    train.codes_expected = cvec.create()
    train.codes_expected:bits_extend(train.codes_spectral, train.adj_expected_ids, train.ids_spectral, 0, train.dims_spectral, true)
    train.category_index = train.cat_index
    str.printf("  Min: %f  Max: %f  Mean: %f\n",
      train.adj_expected_weights:min(),
      train.adj_expected_weights:max(),
      train.adj_expected_weights:sum() / train.adj_expected_weights:size())
  end

  -- Index the codes
  print("\nIndexing codes")
  train.index_spectral = ann.create({
    expected_size = train.n,
    bucket_size = cfg.ann.bucket_size,
    features = train.dims_spectral
  })
  train.index_spectral:add(train.codes_spectral, train.ids_spectral)

  print("\nCodebook stats")
  train.entropy = eval.entropy_stats(train.codes_spectral, train.ids_spectral:size(), train.dims_spectral)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)", train.entropy)

  -- Build code-space KNN adjacency (image-to-image)
  print("\nBuilding retrieved adjacency")
  train.adj_retrieved_ids,
  train.adj_retrieved_offsets,
  train.adj_retrieved_neighbors,
  train.adj_retrieved_weights = graph.adjacency({
    weight_index = train.index_spectral,
    seed_ids = train.adj_expected_ids,
    seed_offsets = train.adj_expected_offsets,
    seed_neighbors = train.adj_expected_neighbors
  })
  str.printf("  Min: %f  Max: %f  Mean: %f  Mean Size: %d\n",
    train.adj_retrieved_weights:min(),
    train.adj_retrieved_weights:max(),
    train.adj_retrieved_weights:sum() / train.adj_retrieved_weights:size(),
    train.adj_retrieved_weights:size() / train.adj_retrieved_ids:size())

  -- Score retrieval
  print("\nRetrieval stats")
  train.retrieval_stats = eval.score_retrieval({
    retrieved_ids = train.adj_retrieved_ids,
    retrieved_offsets = train.adj_retrieved_offsets,
    retrieved_neighbors = train.adj_retrieved_neighbors,
    retrieved_weights = train.adj_retrieved_weights,
    expected_ids = train.adj_expected_ids,
    expected_offsets = train.adj_expected_offsets,
    expected_neighbors = train.adj_expected_neighbors,
    expected_weights = train.adj_expected_weights,
    metric = cfg.eval.retrieval_metric,
    n_dims = train.dims_spectral,
  })

  if cfg.eval.verbose then
    for m = 0, train.retrieval_stats.quality:size() - 1 do
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | Margin: %2d | Quality: %.2f | Recall: %.2f | F1: %.2f\n",
        d, dd, m, train.retrieval_stats.quality:get(m), train.retrieval_stats.recall:get(m), train.retrieval_stats.f1:get(m))
    end
  end

  local best_margin = cfg.eval.retrieval(train.retrieval_stats.quality)
  str.printf("Best\n  Margin: %2d | Quality: %.2f | Recall: %.2f | F1: %.2f\n",
    best_margin, train.retrieval_stats.quality:get(best_margin), train.retrieval_stats.recall:get(best_margin), train.retrieval_stats.f1:get(best_margin))

  -- Optional clustering
  if cfg.cluster.enabled then
    print("\nSetting up clustering adjacency")
    local adj_cluster_ids, adj_cluster_offsets, adj_cluster_neighbors =
      graph.adjacency({
        knn_index = train.index_spectral,
        knn = cfg.cluster.knn,
        each = function (ns, cs, es, stg)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | Stage: %-10s  Nodes: %5d  Components: %5s  Edges: %5s\n",
            d, dd, stg, ns, cs, es)
        end
      })

    print("\nClustering")
    train.codes_clusters = eval.cluster({
      codes = train.index_spectral:get(adj_cluster_ids),
      n_dims = train.dims_spectral,
      ids = adj_cluster_ids,
      offsets = adj_cluster_offsets,
      neighbors = adj_cluster_neighbors,
    })

    train.cluster_stats = eval.score_clustering({
      ids = train.codes_clusters.ids,
      offsets = train.codes_clusters.offsets,
      merges = train.codes_clusters.merges,
      expected_ids = train.adj_expected_ids,
      expected_offsets = train.adj_expected_offsets,
      expected_neighbors = train.adj_expected_neighbors,
      expected_weights = train.adj_expected_weights,
      metric = cfg.eval.clustering_metric,
    })

    for step = 0, train.cluster_stats.n_steps do
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | Step: %2d | Quality: %.2f | Recall: %.2f | F1: %.2f | Clusters: %d\n",
        d, dd, step, train.cluster_stats.quality:get(step), train.cluster_stats.recall:get(step),
        train.cluster_stats.f1:get(step), train.cluster_stats.n_clusters:get(step))
    end

    local best_step = cfg.eval.clustering(train.cluster_stats.quality)
    local best_n_clusters = train.cluster_stats.n_clusters:get(best_step)
    str.printf("Best\n  Step: %2d | Quality: %.2f | Recall: %.2f | F1: %.2f | Clusters: %d\n",
      best_step, train.cluster_stats.quality:get(best_step), train.cluster_stats.recall:get(best_step),
      train.cluster_stats.f1:get(best_step), best_n_clusters)
  end

  collectgarbage("collect")

  -- Optional encoder training
  if cfg.encoder.enabled then

    -- Create landmark encoder
    print("\nCreating landmark encoder")
    local encode_landmarks, n_latent = hlth.landmark_encoder({
      landmarks_index = train.node_features,
      codes_index = train.index_spectral,
      n_landmarks = cfg.data.landmarks
    })

    -- Transform training data to landmark features
    print("\nTransforming training data to landmark features")
    local train_landmark_feats = encode_landmarks(train.train_raw, train.n)
    train_landmark_feats:bits_flip_interleave(n_latent)
    local train_solutions = train.index_spectral:get(train.ids)

    -- Transform test data to landmark features
    print("Transforming test data to landmark features")
    local test_raw = ivec.create()
    dataset.problems:bits_select(nil, test.ids, dataset.n_visible, test_raw)
    test_raw = test_raw:bits_to_cvec(test.n, dataset.n_visible)
    local test_landmark_feats = encode_landmarks(test_raw, test.n)
    test_landmark_feats:bits_flip_interleave(n_latent)

    -- Train TM encoder to predict codes from landmark features
    print("\nTraining encoder")
    train.encoder, train.encoder_accuracy = tm.optimize_encoder({
      visible = n_latent,
      hidden = train.dims_spectral,
      sentences = train_landmark_feats,
      codes = train_solutions,
      samples = train.n,
      clauses = cfg.tm.clauses,
      clause_tolerance = cfg.tm.clause_tolerance,
      clause_maximum = cfg.tm.clause_maximum,
      target = cfg.tm.target,
      specificity = cfg.tm.specificity,
      search_patience = cfg.search.patience,
      search_rounds = cfg.search.rounds,
      search_trials = cfg.search.trials,
      search_iterations = cfg.search.iterations,
      search_tolerance = cfg.search.tolerance,
      final_patience = cfg.training.patience,
      final_iterations = cfg.training.iterations,
      search_metric = function (t)
        local predicted = t:predict(train_landmark_feats, train.n)
        local accuracy = eval.encoding_accuracy(predicted, train_solutions, train.n, train.dims_spectral)
        return accuracy.mean_hamming, accuracy
      end,
      each = function (_, is_final, train_accuracy, params, epoch, round, trial)
        local d, dd = stopwatch()
        if is_final then
          str.printf("  Time %3.2f %3.2f  Finalizing  C=%d LF=%d L=%d T=%d S=%.2f  Epoch  %d\n",
            d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, epoch)
        else
          str.printf("  Time %3.2f %3.2f  Exploring  C=%d LF=%d L=%d T=%d S=%.2f  R=%d T=%d  Epoch  %d\n",
            d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, round, trial, epoch)
        end
        str.printi("    Train | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)", train_accuracy)
      end,
    })

    print("\nFinal encoder performance")
    str.printi("  Train | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)", train.encoder_accuracy)

    -- Predict codes for train and test
    print("\nPredicting codes with encoder")
    local train_predicted = train.encoder:predict(train_landmark_feats, train.n)
    local test_predicted = train.encoder:predict(test_landmark_feats, test.n)

    -- Index predicted codes
    print("Indexing predicted codes")
    local idx_train_pred = ann.create({ features = train.dims_spectral, expected_size = train.n })
    idx_train_pred:add(train_predicted, train.ids_spectral)

    local idx_test_pred = ann.create({ features = train.dims_spectral, expected_size = test.n })
    idx_test_pred:add(test_predicted, test.ids)

    -- Build retrieved adjacency for train predicted codes (reuse expected topology)
    print("\nBuilding retrieved adjacency for predicted codes (train)")
    local train_pred_retrieved_ids, train_pred_retrieved_offsets, train_pred_retrieved_neighbors, train_pred_retrieved_weights =
      graph.adjacency({
        weight_index = idx_train_pred,
        seed_ids = train.adj_expected_ids,
        seed_offsets = train.adj_expected_offsets,
        seed_neighbors = train.adj_expected_neighbors,
      })

    -- Build test expected adjacency
    print("Building expected adjacency for test")
    test.cat_index = inv.create({
      features = 10,
      expected_size = test.n,
      decay = cfg.graph.decay
    })
    local test_data = ivec.create()
    test_data:copy(test.solutions)
    test_data:add_scaled(10)
    test.cat_index:add(test_data, test.ids)
    test_data:destroy()

    local test_adj_expected_ids, test_adj_expected_offsets, test_adj_expected_neighbors, test_adj_expected_weights =
      graph.adjacency({
        category_index = test.cat_index,
        category_anchors = cfg.eval.anchors,
        random_pairs = cfg.eval.pairs,
      })

    -- Build retrieved adjacency for test predicted codes (reuse test expected topology)
    print("Building retrieved adjacency for predicted codes (test)")
    local test_pred_retrieved_ids, test_pred_retrieved_offsets, test_pred_retrieved_neighbors, test_pred_retrieved_weights =
      graph.adjacency({
        weight_index = idx_test_pred,
        seed_ids = test_adj_expected_ids,
        seed_offsets = test_adj_expected_offsets,
        seed_neighbors = test_adj_expected_neighbors,
      })

    -- Evaluate train predicted codes
    print("\nEvaluating train predicted codes")
    local train_pred_stats = eval.score_retrieval({
      retrieved_ids = train_pred_retrieved_ids,
      retrieved_offsets = train_pred_retrieved_offsets,
      retrieved_neighbors = train_pred_retrieved_neighbors,
      retrieved_weights = train_pred_retrieved_weights,
      expected_ids = train.adj_expected_ids,
      expected_offsets = train.adj_expected_offsets,
      expected_neighbors = train.adj_expected_neighbors,
      expected_weights = train.adj_expected_weights,
      metric = cfg.eval.retrieval_metric,
      n_dims = train.dims_spectral,
    })

    -- Evaluate test predicted codes
    print("Evaluating test predicted codes")
    local test_pred_stats = eval.score_retrieval({
      retrieved_ids = test_pred_retrieved_ids,
      retrieved_offsets = test_pred_retrieved_offsets,
      retrieved_neighbors = test_pred_retrieved_neighbors,
      retrieved_weights = test_pred_retrieved_weights,
      expected_ids = test_adj_expected_ids,
      expected_offsets = test_adj_expected_offsets,
      expected_neighbors = test_adj_expected_neighbors,
      expected_weights = test_adj_expected_weights,
      metric = cfg.eval.retrieval_metric,
      n_dims = train.dims_spectral,
    })

    -- Compare results
    local orig_best_margin = cfg.eval.retrieval(train.retrieval_stats.quality)
    local train_pred_best_margin = cfg.eval.retrieval(train_pred_stats.quality)
    local test_pred_best_margin = cfg.eval.retrieval(test_pred_stats.quality)

    str.printi("  Original | Margin: %.0f#(1) | Quality: %.2f#(2) | Recall: %.2f#(3) | F1: %.2f#(4)",
      { orig_best_margin, train.retrieval_stats.quality:get(orig_best_margin), train.retrieval_stats.recall:get(orig_best_margin),  train.retrieval_stats.f1:get(orig_best_margin) })
    str.printi("  Train    | Margin: %.0f#(1) | Quality: %.2f#(2) | Recall: %.2f#(3) | F1: %.2f#(4)",
      { train_pred_best_margin, train_pred_stats.quality:get(train_pred_best_margin), train_pred_stats.recall:get(train_pred_best_margin), train_pred_stats.f1:get(train_pred_best_margin) })
    str.printi("  Test     | Margin: %.0f#(1) | Quality: %.2f#(2) | Recall: %.2f#(3) | F1: %.2f#(4)",
      { test_pred_best_margin, test_pred_stats.quality:get(test_pred_best_margin), test_pred_stats.recall:get(test_pred_best_margin), test_pred_stats.f1:get(test_pred_best_margin) })

    idx_train_pred:destroy()
    idx_test_pred:destroy()
  end

  collectgarbage("collect")

end)
