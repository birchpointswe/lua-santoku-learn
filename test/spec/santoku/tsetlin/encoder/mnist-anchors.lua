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
local simhash = require("santoku.tsetlin.simhash")
local hlth = require("santoku.tsetlin.hlth")
local optimize = require("santoku.tsetlin.optimize")

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
    elbow_select = "lmethod",
    elbow_alpha_select = nil,
    threshold = function (codes, dims)
      local codes, _, _ = itq.otsu({
        codes = codes,
        n_dims = dims,
        metric = "variance",
        minimize = false
      })
      return codes, dims
    end,
    select = function (codes, _, n, dims)
      local eids, escores = codes:mtx_top_entropy(n, dims)
      local _, eidx = escores:scores_elbow(cfg.spectral.elbow_select, cfg.spectral.elbow_alpha_select)
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
    decay = 4, --{ def = 12, min = 0, max = 8 },
    knn = 24, --{ def = 12, min = 4, max = 24, int = true, log = true },
    knn_alpha = 20, --{ def = 30, min = 1, max = 40, int = true, log = true },
    knn_mode = "cknn",
    knn_mutual = true,
    knn_min = nil,
    knn_cache = 32,
    bridge = "mst",
  },
  ann = {
    bucket_size = nil,
  },
  eval = {
    knn = 32,
    anchors = 16,
    pairs = 16,
    ranking = "ndcg",
    metric = "min",
    elbow_retrieval = "lmethod",
    elbow_clustering = "plateau",
    elbow_alpha_retrieval = nil,
    elbow_alpha_clustering = 0.01,
    elbow_target_clustering = "quality",
    verbose = true,
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
  spectral_search = {
    rounds = 3,
    trials = 5,
  },
  training = {
    patience = 10,
    iterations = 200,
  },
}

test("mnist-anchors", function()

  local stopwatch = utc.stopwatch()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.visible, cfg.data.max, cfg.data.max_class)
  dataset.n_visible = cfg.data.visible
  dataset.n_hidden = cfg.data.mode == "spectral" and cfg.spectral.n_dims or cfg.simhash.n_dims

  print("\nSplitting")
  local train, test = ds.split_binary_mnist(dataset, cfg.data.ttr)
  str.printf("  Train: %6d\n", train.n)
  str.printf("  Test:  %6d\n", test.n)

  print("\nBuilding pixel similarity graph")
  do
    local graph_ids = ivec.create()
    graph_ids:copy(train.ids)
    local graph_problems = ivec.create()
    dataset.problems:bits_select(nil, graph_ids, cfg.data.visible, graph_problems)
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
    })
    train.index_graph:add(graph_problems, graph_ids)
  end

  if cfg.data.mode == "spectral" or cfg.encoder.enabled then
    print("\nBuilding node features index (raw pixels)")
    train.node_features = ann.create({ features = dataset.n_visible, expected_size = train.n, bucket_size = cfg.ann.bucket_size })
    train.train_raw = ivec.create()
    dataset.problems:bits_select(nil, train.ids, dataset.n_visible, train.train_raw)
    train.train_raw = train.train_raw:bits_to_cvec(train.n, dataset.n_visible)
    train.node_features:add(train.train_raw, train.ids)
  end

  local function build_category_ground_truth (ids_spectral)
    local cat_index = inv.create({ features = 10 })
    local data = ivec.create()
    data:copy(train.solutions)
    data:add_scaled(10)
    cat_index:add(data, ids_spectral)
    data:destroy()

    local adj_expected_ids, adj_expected_offsets, adj_expected_neighbors, adj_expected_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.eval.anchors,
        random_pairs = cfg.eval.pairs,
      })

    return cat_index, {
      ids = adj_expected_ids,
      offsets = adj_expected_offsets,
      neighbors = adj_expected_neighbors,
      weights = adj_expected_weights,
    }
  end

  if cfg.data.mode == "spectral" then
    print("\nOptimizing spectral pipeline")
    local model = optimize.spectral({
      index = train.index_graph,
      knn_index = train.node_features,
      n_dims = cfg.spectral.n_dims,
      laplacian = cfg.spectral.laplacian,
      eps = cfg.spectral.eps,
      select = cfg.spectral.select,
      threshold = cfg.spectral.threshold,
      knn_cache = cfg.graph.knn_cache,
      knn_mode = cfg.graph.knn_mode,
      bridge = cfg.graph.bridge,
      bucket_size = cfg.ann.bucket_size,
      knn = cfg.graph.knn,
      knn_alpha = cfg.graph.knn_alpha,
      knn_mutual = cfg.graph.knn_mutual,
      weight_decay = cfg.graph.decay,
      search_rounds = cfg.spectral_search.rounds,
      search_trials = cfg.spectral_search.trials,
      adjacency_each = cfg.eval.verbose and function (ns, cs, es, stg)
        if stg == "kruskal" or stg == "done" then
          str.printf("    graph: %d nodes, %d edges, %d components\n", ns, es, cs)
        end
      end or nil,
      spectral_each = cfg.eval.verbose and function (t, s)
        if t == "done" then
          str.printf("    spectral: %d matvecs\n", s)
        end
      end or nil,
      search_metric = function (m)
        local cat_index, ground_truth = build_category_ground_truth(m.ids)
        local adj_retrieved_ids, adj_retrieved_offsets, adj_retrieved_neighbors, adj_retrieved_weights =
          graph.adjacency({
            weight_index = m.index,
            seed_ids = ground_truth.ids,
            seed_offsets = ground_truth.offsets,
            seed_neighbors = ground_truth.neighbors,
          })
        local stats = eval.score_retrieval({
          retrieved_ids = adj_retrieved_ids,
          retrieved_offsets = adj_retrieved_offsets,
          retrieved_neighbors = adj_retrieved_neighbors,
          retrieved_weights = adj_retrieved_weights,
          expected_ids = ground_truth.ids,
          expected_offsets = ground_truth.offsets,
          expected_neighbors = ground_truth.neighbors,
          expected_weights = ground_truth.weights,
          ranking = cfg.eval.ranking,
          metric = cfg.eval.metric,
          elbow = cfg.eval.elbow_retrieval,
          elbow_alpha = cfg.eval.elbow_alpha_retrieval,
          n_dims = m.dims,
        })
        cat_index:destroy()
        ground_truth.ids:destroy()
        ground_truth.offsets:destroy()
        ground_truth.neighbors:destroy()
        ground_truth.weights:destroy()
        adj_retrieved_ids:destroy()
        adj_retrieved_offsets:destroy()
        adj_retrieved_neighbors:destroy()
        adj_retrieved_weights:destroy()
        return stats.f1, {
          score = stats.score,
          quality = stats.quality,
          recall = stats.recall,
          f1 = stats.f1,
        }
      end,
      each = cfg.eval.verbose and function (info)
        if info.event == "stage" and info.stage == "adjacency" then
          local p = info.params
          if info.is_final then
            str.printf("\n  [Final] knn=%d knn_alpha=%d knn_mutual=%s\n",
              p.knn, p.knn_alpha, tostring(p.knn_mutual))
          else
            str.printf("\n  [R%d T%d] knn=%d knn_alpha=%d knn_mutual=%s\n",
              info.round, info.trial, p.knn, p.knn_alpha, tostring(p.knn_mutual))
          end
        elseif info.event == "eval" then
          local m = info.metrics
          str.printf("    => S=%.4f Q=%.4f R=%.4f F1=%.4f\n",
            m.score, m.quality, m.recall, m.f1)
        elseif info.event == "round" then
          str.printf("\n  ---- Round %d complete | Best: %.4f | Global: %.4f ----\n",
            info.round, info.round_best_score, info.global_best_score)
        end
      end or nil,
    })
    train.ids_spectral = model.ids
    train.codes_spectral = model.codes
    train.dims_spectral = model.dims
    train.index_spectral = model.index
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

    -- Index the codes
    print("\nIndexing codes")
    train.index_spectral = ann.create({
      expected_size = train.n,
      bucket_size = cfg.ann.bucket_size,
      features = train.dims_spectral
    })
    train.index_spectral:add(train.codes_spectral, train.ids_spectral)
  else
    error("Unknown mode: " .. cfg.data.mode)
  end

  print("\nCreating class index")
  train.cat_index, train.ground_truth = build_category_ground_truth(train.ids_spectral)
  train.adj_expected_ids = train.ground_truth.ids
  train.adj_expected_offsets = train.ground_truth.offsets
  train.adj_expected_neighbors = train.ground_truth.neighbors
  train.adj_expected_weights = train.ground_truth.weights

  print("\nBuilding expected adjacency (category-based)")
  do
    train.codes_expected = cvec.create()
    train.codes_expected:bits_extend(train.codes_spectral, train.adj_expected_ids, train.ids_spectral, 0, train.dims_spectral, true)
    train.category_index = train.cat_index
    str.printf("  Min: %f  Max: %f  Mean: %f\n",
      train.adj_expected_weights:min(),
      train.adj_expected_weights:max(),
      train.adj_expected_weights:sum() / train.adj_expected_weights:size())
  end

  print("\nCodebook stats")
  train.entropy = eval.entropy_stats(train.codes_spectral, train.ids_spectral:size(), train.dims_spectral)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)", train.entropy)

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

  train.retrieval_stats = eval.score_retrieval({
    retrieved_ids = train.adj_retrieved_ids,
    retrieved_offsets = train.adj_retrieved_offsets,
    retrieved_neighbors = train.adj_retrieved_neighbors,
    retrieved_weights = train.adj_retrieved_weights,
    expected_ids = train.adj_expected_ids,
    expected_offsets = train.adj_expected_offsets,
    expected_neighbors = train.adj_expected_neighbors,
    expected_weights = train.adj_expected_weights,
    ranking = cfg.eval.ranking,
    metric = cfg.eval.metric,
    elbow = cfg.eval.elbow_retrieval,
    elbow_alpha = cfg.eval.elbow_alpha_retrieval,
    n_dims = train.dims_spectral,
  })

  str.printf("\nRetrieval\n  Score: %.4f | Quality: %.4f | Recall: %.4f | F1: %.4f\n",
    train.retrieval_stats.score, train.retrieval_stats.quality,
    train.retrieval_stats.recall, train.retrieval_stats.f1)

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
      metric = cfg.eval.metric,
      elbow = cfg.eval.elbow_clustering,
      elbow_target = cfg.eval.elbow_target_clustering,
      elbow_alpha = cfg.eval.elbow_alpha_clustering,
    })

    if cfg.eval.verbose then
      for step = 0, train.cluster_stats.n_steps do
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Quality: %.2f | Recall: %.2f | F1: %.2f | Clusters: %d\n",
          d, dd, step, train.cluster_stats.quality_curve:get(step), train.cluster_stats.recall_curve:get(step),
          train.cluster_stats.f1_curve:get(step), train.cluster_stats.n_clusters_curve:get(step))
      end
    end

    str.printf("Clustering\n  Best Step: %d | Quality: %.4f | Recall: %.4f | F1: %.4f | Clusters: %d\n",
      train.cluster_stats.best_step, train.cluster_stats.quality, train.cluster_stats.recall,
      train.cluster_stats.f1, train.cluster_stats.n_clusters)
  end

  collectgarbage("collect")

  if cfg.encoder.enabled then

    print("\nCreating landmark encoder")
    local encode_landmarks, n_latent = hlth.landmark_encoder({
      landmarks_index = train.node_features,
      codes_index = train.index_spectral,
      n_landmarks = cfg.data.landmarks
    })

    print("\nTransforming training data to landmark features")
    local train_landmark_feats = encode_landmarks(train.train_raw, train.n)
    train_landmark_feats:bits_flip_interleave(n_latent)
    local train_solutions = train.index_spectral:get(train.ids)

    print("Transforming test data to landmark features")
    local test_raw = ivec.create()
    dataset.problems:bits_select(nil, test.ids, dataset.n_visible, test_raw)
    test_raw = test_raw:bits_to_cvec(test.n, dataset.n_visible)
    local test_landmark_feats = encode_landmarks(test_raw, test.n)
    test_landmark_feats:bits_flip_interleave(n_latent)

    print("\nTraining encoder")
    train.encoder, train.encoder_accuracy = optimize.encoder({
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

    print("\nPredicting codes with encoder")
    local train_predicted = train.encoder:predict(train_landmark_feats, train.n)
    local test_predicted = train.encoder:predict(test_landmark_feats, test.n)

    print("Indexing predicted codes")
    local idx_train_pred = ann.create({ features = train.dims_spectral, expected_size = train.n })
    idx_train_pred:add(train_predicted, train.ids_spectral)

    local idx_test_pred = ann.create({ features = train.dims_spectral, expected_size = test.n })
    idx_test_pred:add(test_predicted, test.ids)

    print("\nBuilding retrieved adjacency for predicted codes (train)")
    local train_pred_retrieved_ids, train_pred_retrieved_offsets, train_pred_retrieved_neighbors, train_pred_retrieved_weights =
      graph.adjacency({
        weight_index = idx_train_pred,
        seed_ids = train.adj_expected_ids,
        seed_offsets = train.adj_expected_offsets,
        seed_neighbors = train.adj_expected_neighbors,
      })

    print("Building expected adjacency for test")
    test.cat_index = inv.create({
      features = 10,
      expected_size = test.n,
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

    print("Building retrieved adjacency for predicted codes (test)")
    local test_pred_retrieved_ids, test_pred_retrieved_offsets, test_pred_retrieved_neighbors, test_pred_retrieved_weights =
      graph.adjacency({
        weight_index = idx_test_pred,
        seed_ids = test_adj_expected_ids,
        seed_offsets = test_adj_expected_offsets,
        seed_neighbors = test_adj_expected_neighbors,
      })

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
      ranking = cfg.eval.ranking,
      metric = cfg.eval.metric,
      elbow = cfg.eval.elbow_retrieval,
      elbow_alpha = cfg.eval.elbow_alpha_retrieval,
      n_dims = train.dims_spectral,
    })

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
      ranking = cfg.eval.ranking,
      metric = cfg.eval.metric,
      elbow = cfg.eval.elbow_retrieval,
      elbow_alpha = cfg.eval.elbow_alpha_retrieval,
      n_dims = train.dims_spectral,
    })

    str.printf("  Original | Score: %.4f | Quality: %.4f | Recall: %.4f | F1: %.4f\n",
      train.retrieval_stats.score, train.retrieval_stats.quality,
      train.retrieval_stats.recall, train.retrieval_stats.f1)
    str.printf("  Train    | Score: %.4f | Quality: %.4f | Recall: %.4f | F1: %.4f\n",
      train_pred_stats.score, train_pred_stats.quality,
      train_pred_stats.recall, train_pred_stats.f1)
    str.printf("  Test     | Score: %.4f | Quality: %.4f | Recall: %.4f | F1: %.4f\n",
      test_pred_stats.score, test_pred_stats.quality,
      test_pred_stats.recall, test_pred_stats.f1)

    idx_train_pred:destroy()
    idx_test_pred:destroy()
  end

  collectgarbage("collect")

end)
