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
  simhash = {
    n_dims = 64,
    ranks = nil,
    quantiles = nil,
  },
  ann = {
    bucket_size = nil,
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
  tm_search = {
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
  search = {
    rounds = 1,
    adjacency_samples = 1,
    spectral_samples = 1,
    eval_samples = 1,
    adjacency = {
      knn = 24,
      knn_alpha = 20,
      weight_decay = 8,
      knn_mutual = true,
      knn_mode = "cknn",
      knn_cache = 32,
      bridge = "mst",
    },
    spectral = {
      laplacian = "unnormalized",
      n_dims = 24,
      eps = 1e-12,
      threshold = {
        method = "otsu",
        otsu_metric = "variance",
        otsu_minimize = false,
        otsu_n_bins = 32,
      },
    },
    eval = {
      knn = 32,
      anchors = 16,
      pairs = 64,
      ranking = "ndcg",
      metric = "min",
      elbow = "lmethod",
      elbow_alpha = nil,
      select_elbow = "lmethod",
      select_metric = "entropy",
      target = "combined",
    },
    cluster_eval = {
      elbow = "plateau",
      elbow_alpha = 0.01,
      elbow_target = "quality",
    },
    verbose = true,
  },
}

test("mnist-anchors", function()

  local stopwatch = utc.stopwatch()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.visible, cfg.data.max, cfg.data.max_class)
  dataset.n_visible = cfg.data.visible
  dataset.n_hidden = cfg.data.mode == "spectral" and cfg.search.spectral.n_dims or cfg.simhash.n_dims

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

  local function build_category_ground_truth (ids)
    local cat_index = inv.create({ features = 10 })
    local data = ivec.create()
    data:copy(train.solutions)
    data:add_scaled(10)
    cat_index:add(data, ids)
    data:destroy()

    local adj_expected_ids, adj_expected_offsets, adj_expected_neighbors, adj_expected_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.search.eval.anchors,
        random_pairs = cfg.search.eval.pairs,
      })

    local adj_cluster_ids, adj_cluster_offsets, adj_cluster_neighbors, adj_cluster_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.search.eval.anchors,
      })

    return cat_index, {
      retrieval = {
        ids = adj_expected_ids,
        offsets = adj_expected_offsets,
        neighbors = adj_expected_neighbors,
        weights = adj_expected_weights,
      },
      clustering = {
        ids = adj_cluster_ids,
        offsets = adj_cluster_offsets,
        neighbors = adj_cluster_neighbors,
        weights = adj_cluster_weights,
      },
    }
  end

  print("\nBuilding category ground truth")
  train.cat_index, train.ground_truth = build_category_ground_truth(train.ids)

  if cfg.data.mode == "spectral" then
    print("\nOptimizing spectral pipeline")
    local model = optimize.spectral({
      index = train.index_graph,
      knn_index = train.node_features,
      bucket_size = cfg.ann.bucket_size,
      adjacency_samples = cfg.search.adjacency_samples,
      spectral_samples = cfg.search.spectral_samples,
      eval_samples = cfg.search.eval_samples,
      adjacency = cfg.search.adjacency,
      spectral = cfg.search.spectral,
      eval = cfg.search.eval,
      expected_ids = train.ground_truth.retrieval.ids,
      expected_offsets = train.ground_truth.retrieval.offsets,
      expected_neighbors = train.ground_truth.retrieval.neighbors,
      expected_weights = train.ground_truth.retrieval.weights,
      adjacency_each = cfg.search.verbose and function (ns, cs, es, stg)
        if stg == "kruskal" or stg == "done" then
          str.printf("    graph: %d nodes, %d edges, %d components\n", ns, es, cs)
        end
      end or nil,
      spectral_each = cfg.search.verbose and function (t, s)
        if t == "done" then
          str.printf("    spectral: %d matvecs\n", s)
        end
      end or nil,
      each = cfg.search.verbose and function (info)
        if info.event == "round_start" then
          str.printf("\n=== Round %d/%d ===\n", info.round, info.rounds)
        elseif info.event == "round_end" then
          str.printf("\n--- Round %d complete: best=%.4f (global=%.4f) success=%.0f%% adapt=%.2f ---\n",
            info.round, info.round_best_score, info.global_best_score,
            (info.success_rate or 0) * 100, info.adapt_factor or 1)
        elseif info.event == "stage" and info.stage == "adjacency" then
          local p = info.params.adjacency
          if info.is_final then
            str.printf("\n  [Final] decay=%.2f knn=%d knn_alpha=%d knn_mutual=%s\n",
              p.weight_decay, p.knn, p.knn_alpha, tostring(p.knn_mutual))
          else
            str.printf("\n  [%d] decay=%.2f knn=%d knn_alpha=%d knn_mutual=%s\n",
              info.sample, p.weight_decay, p.knn, p.knn_alpha, tostring(p.knn_mutual))
          end
        elseif info.event == "stage" and info.stage == "spectral" then
          local p = info.params.spectral
          local thresh_str = "unknown"
          if p.threshold_params and p.threshold_params.method then
            local m = p.threshold_params.method
            if m == "itq" then
              local iters = p.threshold_params.iterations or 100
              local tol = p.threshold_params.tolerance or 1e-8
              thresh_str = str.format("itq(i=%d,t=%.0e)", iters, tol)
            elseif m == "otsu" then
              local metric = p.threshold_params.metric or "variance"
              local minimize = p.threshold_params.minimize and "min" or "max"
              thresh_str = str.format("otsu(%s,%s)", metric, minimize)
            else
              thresh_str = m
            end
          elseif type(p.threshold) == "string" then
            thresh_str = p.threshold
          end
          str.printf("    lap=%s dims=%d thresh=%s\n", p.laplacian, p.n_dims, thresh_str)
        elseif info.event == "eval" then
          local e = info.params.eval
          local m = info.metrics
          local alpha_str = e.elbow_alpha and str.format("%.1f", e.elbow_alpha) or "-"
          local select_str = ""
          if e.select_metric and e.select_metric ~= "none" then
            local sel_alpha = e.select_elbow_alpha and str.format("%.1f", e.select_elbow_alpha) or "-"
            local dims_str
            if m.selected_elbow and m.selected_elbow <= 0 then
              dims_str = str.format("FAIL->%d", m.n_dims)
            elseif m.selected_elbow and m.selected_elbow ~= m.n_dims then
              dims_str = str.format("%d->%d", m.selected_elbow, m.n_dims)
            else
              dims_str = str.format("%d", m.n_dims)
            end
            select_str = str.format(" [%s/%s(%s) %s]", e.select_metric, e.select_elbow, sel_alpha, dims_str)
          end
          str.printf("    elbow=%s(%s) knn=%d score=%.4f quality=%.4f combined=%.4f%s\n",
            e.elbow, alpha_str, e.knn, m.score, m.quality, m.combined, select_str)
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

  train.adj_expected_ids = train.ground_truth.retrieval.ids
  train.adj_expected_offsets = train.ground_truth.retrieval.offsets
  train.adj_expected_neighbors = train.ground_truth.retrieval.neighbors
  train.adj_expected_weights = train.ground_truth.retrieval.weights

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

  print("\nBuilding retrieved adjacency (spectral KNN)")
  train.adj_retrieved_ids,
  train.adj_retrieved_offsets,
  train.adj_retrieved_neighbors,
  train.adj_retrieved_weights = graph.adjacency({
    weight_index = train.index_spectral,
    seed_ids = train.adj_expected_ids,
    seed_offsets = train.adj_expected_offsets,
    seed_neighbors = train.adj_expected_neighbors,
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
    ranking = cfg.search.eval.ranking,
    metric = cfg.search.eval.metric,
    elbow = cfg.search.eval.elbow,
    elbow_alpha = cfg.search.eval.elbow_alpha,
    n_dims = train.dims_spectral,
  })

  str.printf("\nRetrieval\n  Score: %.4f | Quality: %.4f | Combined: %.4f\n",
    train.retrieval_stats.score, train.retrieval_stats.quality,
    train.retrieval_stats.combined)

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
      expected_ids = train.ground_truth.retrieval.ids,
      expected_offsets = train.ground_truth.retrieval.offsets,
      expected_neighbors = train.ground_truth.retrieval.neighbors,
      expected_weights = train.ground_truth.retrieval.weights,
      metric = cfg.search.eval.metric,
      elbow = cfg.search.cluster_eval.elbow,
      elbow_target = cfg.search.cluster_eval.elbow_target,
      elbow_alpha = cfg.search.cluster_eval.elbow_alpha,
    })

    if cfg.search.verbose then
      for step = 0, train.cluster_stats.n_steps do
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Quality: %.2f | Clusters: %d\n",
          d, dd, step, train.cluster_stats.quality_curve:get(step),
          train.cluster_stats.n_clusters_curve:get(step))
      end
    end

    str.printf("Clustering\n  Best Step: %d | Quality: %.4f | Clusters: %d\n",
      train.cluster_stats.best_step, train.cluster_stats.quality,
      train.cluster_stats.n_clusters)
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
      search_patience = cfg.tm_search.patience,
      search_rounds = cfg.tm_search.rounds,
      search_trials = cfg.tm_search.trials,
      search_iterations = cfg.tm_search.iterations,
      search_tolerance = cfg.tm_search.tolerance,
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
        category_anchors = cfg.search.eval.anchors,
        random_pairs = cfg.search.eval.pairs,
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
      ranking = cfg.search.eval.ranking,
      metric = cfg.search.eval.metric,
      elbow = cfg.search.eval.elbow,
      elbow_alpha = cfg.search.eval.elbow_alpha,
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
      ranking = cfg.search.eval.ranking,
      metric = cfg.search.eval.metric,
      elbow = cfg.search.eval.elbow,
      elbow_alpha = cfg.search.eval.elbow_alpha,
      n_dims = train.dims_spectral,
    })

    str.printf("  Original | Score: %.4f | Quality: %.4f | Combined: %.4f\n",
      train.retrieval_stats.score, train.retrieval_stats.quality,
      train.retrieval_stats.combined)
    str.printf("  Train    | Score: %.4f | Quality: %.4f | Combined: %.4f\n",
      train_pred_stats.score, train_pred_stats.quality,
      train_pred_stats.combined)
    str.printf("  Test     | Score: %.4f | Quality: %.4f | Combined: %.4f\n",
      test_pred_stats.score, test_pred_stats.quality,
      test_pred_stats.combined)

    idx_train_pred:destroy()
    idx_test_pred:destroy()
  end

  collectgarbage("collect")

end)
