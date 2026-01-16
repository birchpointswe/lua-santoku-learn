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
local simhash = require("santoku.tsetlin.simhash")
local hlth = require("santoku.tsetlin.hlth")
local optimize = require("santoku.tsetlin.optimize")

local cfg; cfg = {
  data = {
    ttr = 0.9,
    tvr = 0.1,
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
  landmarks = {
    n_landmarks = { def = 8, min = 4, max = 16, int = true },
    n_thresholds = { def = 7, min = 3, max = 12, int = true },
    landmark_mode = { "frequency", "weighted" },
    quantile = false,
  },
  ann = {
    bucket_size = nil,
  },
  cluster = {
    enabled = true,
    verbose = true,
    knn = 64,
  },
  encoder = {
    enabled = false,
    verbose = true,
  },
  tm = {
    clauses = { def = 16, min = 8, max = 32, round = 8 },
    clause_tolerance = { def = 64, min = 16, max = 128, int = true },
    clause_maximum = { def = 64, min = 16, max = 128, int = true },
    target = { def = 32, min = 16, max = 128, int = true },
    specificity = { def = 1000, min = 400, max = 4000 },
    include_bits = { def = 1, min = 1, max = 4, int = true },
  },
  tm_search = {
    patience = 2,
    rounds = 4,
    trials = 10,
    iterations = 20,
  },
  training = {
    patience = 2,
    iterations = 200,
  },
  classifier = {
    enabled = true,
    clauses = { def = 8, min = 8, max = 32, round = 8 },
    clause_tolerance = { def = 64, min = 16, max = 128, int = true },
    clause_maximum = { def = 64, min = 16, max = 128, int = true },
    target = { def = 32, min = 16, max = 128, int = true },
    specificity = { def = 1000, min = 400, max = 4000 },
    include_bits = { def = 1, min = 1, max = 4, int = true },
    negative = 0.5,
    search_patience = 2,
    search_rounds = 4,
    search_trials = 10,
    search_iterations = 20,
    final_iterations = 100,
  },
  search = {
    rounds = 4,  -- 0 = use baselines (def values), >0 = explore
    adjacency_samples = 4,
    spectral_samples = 2,
    eval_samples = 2,
    adjacency = {
      knn = { def = 24, min = 16, max = 32, int = true },
      knn_alpha = { def = 20, min = 12, max = 28, int = true },
      weight_decay = { def = 8, min = 2, max = 16 },
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
      metric = "avg",
      select_elbow = "lmethod",
      select_metric = "entropy",
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
  local train, test, validate = ds.split_binary_mnist(dataset, cfg.data.ttr, cfg.data.tvr)
  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test.n)

  print("\nCreating IDs")
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  validate.ids = ivec.create(validate.n)
  validate.ids:fill_indices()
  validate.ids:add(train.n)
  test.ids = ivec.create(test.n)
  test.ids:fill_indices()
  test.ids:add(train.n + validate.n)

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
      rounds = cfg.search.rounds,
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
          str.printf("    knn=%d score=%.4f%s\n", e.knn, m.score, select_str)
        end
      end or nil,
    })
    train.ids_sup = model.ids
    train.codes_sup = model.codes
    train.dims_sup = model.dims
    train.index_sup = model.index
  elseif cfg.data.mode == "simhash" then
    print("\nSimhashing")
    local idx = inv.create({ features = 10, expected_size = train.n })
    local data = ivec.create()
    data:copy(train.solutions)
    data:add_scaled(10)
    idx:add(data, train.ids)
    data:destroy()
    train.ids_sup, train.codes_sup = simhash.encode(idx, cfg.simhash.n_dims, cfg.simhash.ranks, cfg.simhash.quantiles)
    train.dims_sup = cfg.simhash.n_dims
    idx:destroy()
    print("\nIndexing codes")
    train.index_sup = ann.create({
      expected_size = train.n,
      bucket_size = cfg.ann.bucket_size,
      features = train.dims_sup
    })
    train.index_sup:add(train.codes_sup, train.ids_sup)
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
    train.codes_expected:bits_extend(train.codes_sup, train.adj_expected_ids, train.ids_sup, 0, train.dims_sup, true)
    train.category_index = train.cat_index
    str.printf("  Min: %f  Max: %f  Mean: %f\n",
      train.adj_expected_weights:min(),
      train.adj_expected_weights:max(),
      train.adj_expected_weights:sum() / train.adj_expected_weights:size())
  end

  print("\nCodebook stats")
  train.entropy = eval.entropy_stats(train.codes_sup, train.ids_sup:size(), train.dims_sup)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)", train.entropy)

  print("\nBuilding retrieved adjacency (spectral KNN)")
  train.adj_retrieved_ids,
  train.adj_retrieved_offsets,
  train.adj_retrieved_neighbors,
  train.adj_retrieved_weights = graph.adjacency({
    weight_index = train.index_sup,
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
    n_dims = train.dims_sup,
  })

  str.printf("\nRetrieval\n  Score: %.4f\n", train.retrieval_stats.score)

  if cfg.cluster.enabled then
    print("\nSetting up clustering adjacency")
    local adj_cluster_ids, adj_cluster_offsets, adj_cluster_neighbors =
      graph.adjacency({
        knn_index = train.index_sup,
        knn = cfg.cluster.knn,
        each = function (ns, cs, es, stg)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | Stage: %-10s  Nodes: %5d  Components: %5s  Edges: %5s\n",
            d, dd, stg, ns, cs, es)
        end
      })

    print("\nClustering")
    train.codes_clusters = eval.cluster({
      codes = train.index_sup:get(adj_cluster_ids),
      n_dims = train.dims_sup,
      ids = adj_cluster_ids,
      offsets = adj_cluster_offsets,
      neighbors = adj_cluster_neighbors,
      quality = true,
    })

    local cost_curve = dvec.create()
    cost_curve:copy(train.codes_clusters.quality_curve)
    cost_curve:log()
    cost_curve:scale(-1)
    local _, best_step = cost_curve:scores_elbow("lmethod")
    train.codes_clusters.best_step = best_step
    train.codes_clusters.quality = train.codes_clusters.quality_curve:get(best_step)
    train.codes_clusters.n_clusters = train.codes_clusters.n_clusters_curve:get(best_step)

    if cfg.search.verbose then
      for step = 0, train.codes_clusters.n_steps do
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Quality: %.2f | Clusters: %d\n",
          d, dd, step, train.codes_clusters.quality_curve:get(step),
          train.codes_clusters.n_clusters_curve:get(step))
      end
    end

    str.printf("Clustering\n  Best Step: %d | Quality: %.4f | Clusters: %d\n",
      train.codes_clusters.best_step, train.codes_clusters.quality,
      train.codes_clusters.n_clusters)
  end

  collectgarbage("collect")

  if cfg.encoder.enabled then

    print("\nCreating landmark encoder")
    local encode_landmarks, n_latent = hlth.landmark_encoder({
      landmarks_index = train.node_features,
      codes_index = train.index_sup,
      n_landmarks = cfg.data.landmarks
    })

    print("\nTransforming training data to landmark features")
    local train_landmark_feats = encode_landmarks(train.train_raw, train.n)
    train_landmark_feats:bits_flip_interleave(n_latent)
    local train_solutions = train.index_sup:get(train.ids)

    print("Transforming validate data to landmark features")
    local validate_raw = ivec.create()
    dataset.problems:bits_select(nil, validate.ids, dataset.n_visible, validate_raw)
    validate_raw = validate_raw:bits_to_cvec(validate.n, dataset.n_visible)
    local validate_landmark_feats = encode_landmarks(validate_raw, validate.n)
    validate_landmark_feats:bits_flip_interleave(n_latent)

    print("Transforming test data to landmark features")
    local test_raw = ivec.create()
    dataset.problems:bits_select(nil, test.ids, dataset.n_visible, test_raw)
    test_raw = test_raw:bits_to_cvec(test.n, dataset.n_visible)
    local test_landmark_feats = encode_landmarks(test_raw, test.n)
    test_landmark_feats:bits_flip_interleave(n_latent)

    print("Building expected adjacency for validate")
    validate.cat_index = inv.create({
      features = 10,
      expected_size = validate.n,
    })
    local val_data = ivec.create()
    val_data:copy(validate.solutions)
    val_data:add_scaled(10)
    validate.cat_index:add(val_data, validate.ids)
    val_data:destroy()
    local val_adj_expected_ids, val_adj_expected_offsets, val_adj_expected_neighbors, val_adj_expected_weights =
      graph.adjacency({
        category_index = validate.cat_index,
        category_anchors = cfg.search.eval.anchors,
        random_pairs = cfg.search.eval.pairs,
      })

    print("\nTraining encoder")
    train.encoder, train.encoder_accuracy = optimize.encoder({
      visible = n_latent,
      hidden = train.dims_sup,
      sentences = train_landmark_feats,
      codes = train_solutions,
      samples = train.n,
      clauses = cfg.tm.clauses,
      clause_tolerance = cfg.tm.clause_tolerance,
      clause_maximum = cfg.tm.clause_maximum,
      target = cfg.tm.target,
      specificity = cfg.tm.specificity,
      include_bits = cfg.tm.include_bits,
      search_patience = cfg.tm_search.patience,
      search_rounds = cfg.tm_search.rounds,
      search_trials = cfg.tm_search.trials,
      search_iterations = cfg.tm_search.iterations,
      search_tolerance = cfg.tm_search.tolerance,
      final_patience = cfg.training.patience,
      final_iterations = cfg.training.iterations,
      search_metric = function (t)
        local val_pred = t:predict(validate_landmark_feats, validate.n)
        local idx_val = ann.create({ features = train.dims_sup, expected_size = validate.n })
        idx_val:add(val_pred, validate.ids)
        local val_retrieved_ids, val_retrieved_offsets, val_retrieved_neighbors, val_retrieved_weights =
          graph.adjacency({
            weight_index = idx_val,
            seed_ids = val_adj_expected_ids,
            seed_offsets = val_adj_expected_offsets,
            seed_neighbors = val_adj_expected_neighbors,
          })
        local val_stats = eval.score_retrieval({
          retrieved_ids = val_retrieved_ids,
          retrieved_offsets = val_retrieved_offsets,
          retrieved_neighbors = val_retrieved_neighbors,
          retrieved_weights = val_retrieved_weights,
          expected_ids = val_adj_expected_ids,
          expected_offsets = val_adj_expected_offsets,
          expected_neighbors = val_adj_expected_neighbors,
          expected_weights = val_adj_expected_weights,
          ranking = cfg.search.eval.ranking,
          metric = cfg.search.eval.metric,
          n_dims = train.dims_sup,
        })
        idx_val:destroy()
        val_retrieved_ids:destroy()
        val_retrieved_offsets:destroy()
        val_retrieved_neighbors:destroy()
        val_retrieved_weights:destroy()
        return val_stats.score, val_stats
      end,
      each = function (_, is_final, val_accuracy, params, epoch, round, trial)
        local d, dd = stopwatch()
        local phase = is_final and "[F]" or str.format("[R%d T%d]", round, trial)
        str.printf("  [E%d]%s %.2f %.2f C=%d L=%d/%d T=%d S=%.0f IB=%d score=%.4f\n",
          epoch, phase, d, dd, params.clauses, params.clause_tolerance, params.clause_maximum,
          params.target, params.specificity, params.include_bits, val_accuracy.score)
      end,
    })

    print("\nFinal encoder performance")
    str.printf("  Val | Score: %.4f\n", train.encoder_accuracy.score)

    print("\nPredicting codes with encoder")
    local train_predicted = train.encoder:predict(train_landmark_feats, train.n)
    local validate_predicted = train.encoder:predict(validate_landmark_feats, validate.n)
    local test_predicted = train.encoder:predict(test_landmark_feats, test.n)

    print("Indexing predicted codes")
    local idx_train_pred = ann.create({ features = train.dims_sup, expected_size = train.n })
    idx_train_pred:add(train_predicted, train.ids_sup)

    local idx_test_pred = ann.create({ features = train.dims_sup, expected_size = test.n })
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
      n_dims = train.dims_sup,
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
      n_dims = train.dims_sup,
    })

    str.printf("  Original | Score: %.4f\n", train.retrieval_stats.score)
    str.printf("  Train    | Score: %.4f\n", train_pred_stats.score)
    str.printf("  Test     | Score: %.4f\n", test_pred_stats.score)

    idx_train_pred:destroy()
    idx_test_pred:destroy()

    if cfg.classifier.enabled then
      print("\nClassifier")
      train_predicted:bits_flip_interleave(train.dims_sup)
      validate_predicted:bits_flip_interleave(train.dims_sup)
      test_predicted:bits_flip_interleave(train.dims_sup)

      local classifier = optimize.classifier({
        features = train.dims_sup,
        classes = 10,
        clauses = cfg.classifier.clauses,
        clause_tolerance = cfg.classifier.clause_tolerance,
        clause_maximum = cfg.classifier.clause_maximum,
        target = cfg.classifier.target,
        negative = cfg.classifier.negative,
        specificity = cfg.classifier.specificity,
        include_bits = cfg.classifier.include_bits,
        samples = train.n,
        problems = train_predicted,
        solutions = train.solutions,
        search_patience = cfg.classifier.search_patience,
        search_rounds = cfg.classifier.search_rounds,
        search_trials = cfg.classifier.search_trials,
        search_iterations = cfg.classifier.search_iterations,
        final_iterations = cfg.classifier.final_iterations,
        search_metric = function (t)
          local predicted = t:predict(validate_predicted, validate.n)
          local accuracy = eval.class_accuracy(predicted, validate.solutions, validate.n, 10)
          return accuracy.f1, accuracy
        end,
        each = function (t, is_final, val_accuracy, params, epoch, round, trial)
          local test_pred = t:predict(test_predicted, test.n)
          local test_accuracy = eval.class_accuracy(test_pred, test.solutions, test.n, 10)
          local d, dd = stopwatch()
          local phase = is_final and "[F]" or str.format("[R%d T%d]", round, trial)
          str.printf("  [E%d]%s %.2f %.2f C=%d L=%d/%d T=%d S=%.0f IB=%d f1=(%.2f,%.2f)\n",
            epoch, phase, d, dd, params.clauses, params.clause_tolerance, params.clause_maximum,
            params.target, params.specificity, params.include_bits, val_accuracy.f1, test_accuracy.f1)
        end,
      })

      local train_class_pred = classifier:predict(train_predicted, train.n)
      local val_class_pred = classifier:predict(validate_predicted, validate.n)
      local test_class_pred = classifier:predict(test_predicted, test.n)
      local train_class_stats = eval.class_accuracy(train_class_pred, train.solutions, train.n, 10)
      local val_class_stats = eval.class_accuracy(val_class_pred, validate.solutions, validate.n, 10)
      local test_class_stats = eval.class_accuracy(test_class_pred, test.solutions, test.n, 10)

      str.printf("\nClassifier F1: train=%.2f val=%.2f test=%.2f\n",
        train_class_stats.f1, val_class_stats.f1, test_class_stats.f1)
    end
  end

  collectgarbage("collect")

end)
