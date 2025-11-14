require("santoku.dvec")
require("santoku.pvec")
local ds = require("santoku.tsetlin.dataset")
local err = require("santoku.error")
local eval = require("santoku.tsetlin.evaluator")
local tm = require("santoku.tsetlin")
local inv = require("santoku.tsetlin.inv")
local ann = require("santoku.tsetlin.ann")
local hbi = require("santoku.tsetlin.hbi")
local ivec = require("santoku.ivec")
local cvec = require("santoku.cvec")
local graph = require("santoku.tsetlin.graph")
local simhash = require("santoku.tsetlin.simhash")
local spectral = require("santoku.tsetlin.spectral")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local hlth = require("santoku.tsetlin.hlth")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local cfg; cfg = {
  data = {
    ttr = 0.9,
    max = nil,
    max_class = nil,
    visible = 784,
    hidden = 24,
    landmarks = 24,
  },
  mode = {
    encoder = true,
    cluster = true,
    codes = "spectral", -- simhash, spectral
    mode = "landmarks",
    binarize = "median",
    tch = false,
  },
  tch = {
    iterations = 10000,
  },
  index = {
    ann = true,
  },
  spectral = {
    laplacian = "random",
    method = "jdqr",
    precondition = "ic",
    primme_eps = 1e-6,
  },
  simhash = {
    ranks = 1,
    quantiles = nil,
  },
  sr = {
    eps = 1e-12,
    iterations = 10000,
  },
  itq = {
    eps = 1e-12,
    iterations = 1000,
  },
  graph = {
    reweight = nil,
    weight_cmp = nil,
    weight_alpha = nil,
    weight_beta = nil,
    knn = 32,
    knn_min = nil,
    knn_cache = nil,
    knn_mutual = false,
    category_anchors = nil,
    category_knn = nil,
    category_knn_decay = nil,
    sigma_k = nil,
    decay = 4.0,
    bridge = "mst",
  },
  clustering = {
    knn = 256,
  },
  eval = {
    knn = 16,
    pairs = 16,
    anchors = 16,
    bits_metric = "pearson",
    retrieval_metric = "min",
    cluster_metric = "min",
    tolerance = 2e-2,
    retrieval = function (d)
      return d:max()
      -- return d:scores_plateau(cfg.eval.tolerance)
    end,
    clustering = function (d)
      return d:max()
      -- return d:scores_plateau(cfg.eval.tolerance)
    end,
  },
  cluster = {
    knn = 256
  },
  bits = {
    sel = false,
    keep_prefix = nil,
    start_prefix = nil,
    sffs_tolerance = nil,
    sffs_fixed = nil,
  },
  tm = {
    clauses = 8, --{ def = 8, min = 8, max = 32, int = true, log = true, pow2 = true },
    clause_tolerance =  8, --{ def = 8, min = 8, max = 256, int = true, log = true, pow2 = true },
    clause_maximum = 16, --{ def = 8, min = 8, max = 256, int = true, log = true, pow2 = true },
    target = 8, --{ def = 4, min = 2, max = 256, int = true, log = true, pow2 = true },
    specificity = 400, --{ def = 10, min = 2, max = 400, int = true, log = true },
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

test("tsetlin", function ()

  print("Reading data")

  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.visible, cfg.data.max, cfg.data.max_class)
  dataset.n_visible = cfg.data.visible
  dataset.n_hidden = cfg.data.hidden
  dataset.n_landmarks = cfg.data.landmarks
  collectgarbage("collect")

  print("Splitting")

  local train, test = ds.split_binary_mnist(dataset, cfg.data.ttr)

  do
    train.node_features = ann.create({ features = dataset.n_visible, expected_size = train.n })
    local data = ivec.create()
    dataset.problems:bits_select(nil, train.ids, dataset.n_visible, data)
    data = data:bits_to_cvec(train.n, dataset.n_visible)
    train.node_features:add(data, train.ids)
    data:destroy()
  end

  do
    local problems_ext = ivec.create()
    problems_ext:copy(train.solutions)
    problems_ext:add_scaled(10)
    local graph_ids = ivec.create()
    graph_ids:copy(train.ids)
    local graph_problems = ivec.create()
    dataset.problems:bits_select(nil, graph_ids, dataset.n_visible, graph_problems)
    graph_problems:bits_extend(problems_ext, dataset.n_visible, 10)
    local graph_ranks = ivec.create(dataset.n_visible + 10)
    graph_ranks:fill(1, 0, dataset.n_visible)
    graph_ranks:fill(0, dataset.n_visible, dataset.n_visible + 10)
    train.node_combined = inv.create({ features = dataset.n_visible + 10, ranks = graph_ranks, decay = cfg.graph.decay, n_ranks = 2 })
    train.node_combined:add(graph_problems, graph_ids)
  end

  print("Creating graph")
  local stopwatch = utc.stopwatch()
  train.adj_ids,
  train.adj_offsets,
  train.adj_neighbors,
  train.adj_weights =
    graph.adjacency({
      reweight = cfg.graph.reweight,
      weight_index = train.node_combined,
      weight_cmp = cfg.graph.weight_cmp,
      weight_alpha = cfg.graph.weight_alpha,
      weight_beta = cfg.graph.weight_beta,
      category_index = train.node_combined,
      category_anchors = cfg.graph.category_anchors,
      category_knn = cfg.graph.category_knn,
      category_knn_decay = cfg.graph.category_knn_decay,
      category_ranks = 1,
      random_pairs = cfg.graph.random_pairs,
      knn_index = train.node_features,
      knn = cfg.graph.knn,
      knn_min = cfg.graph.knn_min,
      knn_mutual = cfg.graph.knn_mutual,
      knn_cache = cfg.graph.knn_cache,
      knn_rank = 1,
      sigma_k = cfg.graph.sigma_k,
      bridge = cfg.graph.bridge,
      each = function (ids, s, b, dt)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Nodes: %-6d  Components: %-6d  Edges: %-6d\n", d, dd, dt, ids, s, b)
      end
    })

  train.node_features:destroy()
  collectgarbage("collect")

  print("Weight stats")
  str.printf("  Max: %.6f  Min: %.6f:  Avg: %.6f\n",
    (train.adj_weights:max()),
    (train.adj_weights:min()),
    (train.adj_weights:sum() / train.adj_weights:size()))

  if cfg.mode.codes == "spectral" then
    print("Spectral eigendecomposition")
    train.ids_spectral, train.codes_spectral = spectral.encode({
      method = cfg.spectral.method,
      precondition = cfg.spectral.precondition,
      type = cfg.spectral.laplacian,
      eps = cfg.spectral.primme_eps,
      ids = train.adj_ids,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      n_hidden = dataset.n_hidden,
      each = function (t, s, v, k)
        local d, dd = stopwatch()
        if t == "done" then
          str.printf("  Time: %6.2f %6.2f  Stage: %s  matvecs = %d\n", d, dd, t, s)
        elseif t == "eig" then
          local gap = train.eig_last and v - train.eig_last or 0
          train.eig_last = v
          str.printf("  Time: %6.2f %6.2f  Stage: %s  %3d = %.8f   Gap = %.8f   %s\n", d, dd, t, s, v, gap, k and "" or "drop")
        else
          str.printf("  Time: %6.2f %6.2f  Stage: %s\n", d, dd, t)
        end
      end
    })
    train.node_combined:destroy()
    collectgarbage("collect")
  elseif cfg.mode.codes == "simhash" then
    print("Simhash")
    train.ids_simhash, train.codes_simhash = simhash.encode(train.node_combined, dataset.n_hidden, cfg.simhash.ranks, cfg.simhash.quantiles)
    train.ids_spectral = ivec.create()
    train.ids_spectral:copy(train.adj_ids)
    train.codes_spectral = cvec.create()
    train.codes_spectral:bits_extend(train.codes_simhash, train.ids_spectral, train.ids_simhash, 0, dataset.n_hidden, true)
    train.node_combined:destroy()
    collectgarbage("collect")
  end

  if cfg.mode.codes == "spectral" then
    train.codes_spectral_cont = train.codes_spectral
    if cfg.mode.binarize == "itq" then
      print("Iterative Quantization")
      train.codes_spectral = itq.encode({
        codes = train.codes_spectral,
        n_dims = dataset.n_hidden,
        tolerance = cfg.itq.eps,
        iterations = cfg.itq.iterations,
        each = function (i, a, b)
          str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
        end
      })
    elseif cfg.mode.binarize == "sr-ranking" then
      print("Spectral Rotation (Rank-Aware)")
      train.codes_spectral = itq.sr_ranking({
        codes = train.codes_spectral,
        n_dims = dataset.n_hidden,
        ids = train.adj_ids,
        offsets = train.adj_offsets,
        neighbors = train.adj_neighbors,
        weights = train.adj_weights,
        tolerance = cfg.sr.eps,
        iterations = cfg.sr.iterations,
        each = function (i, a, b)
          str.printf("  SR-Ranking completed in %s itrs. Objective %f → %f\n", i, a, b)
        end
      })
    elseif cfg.mode.binarize == "sr" then
      print("Spectral Rotation")
      train.codes_spectral = itq.sr({
        codes = train.codes_spectral,
        n_dims = dataset.n_hidden,
        tolerance = cfg.sr.eps,
        iterations = cfg.sr.iterations,
        each = function (i, a, b)
          str.printf("  SR completed in %s itrs. Objective %f → %f\n", i, a, b)
        end
      })
    elseif cfg.mode.binarize == "median" then
      print("Median thresholding")
      train.codes_spectral = itq.median({
        codes = train.codes_spectral,
        n_dims = dataset.n_hidden,
      })
    elseif cfg.mode.binarize == "sign" then
      print("Sign thresholding")
      train.codes_spectral = itq.sign({
        codes = train.codes_spectral,
        n_dims = dataset.n_hidden,
      })
    elseif cfg.mode.binarize == "dbq" then
      print("DBQ thresholding")
      train.codes_spectral, dataset.n_hidden = itq.dbq({
        codes = train.codes_spectral,
        n_dims = dataset.n_hidden,
      })
    end
    collectgarbage("collect")
  end

  if cfg.mode.tch then
    print("Flipping bits")
    tch.refine({
      ids = train.adj_ids,
      iterations = cfg.tch.iterations,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end
  collectgarbage("collect")

  print("Rearranging codes to match graph adjacency")
  local codes_adj_order = cvec.create()
  train.codes_spectral:bits_select(nil, train.adj_ids, dataset.n_hidden, codes_adj_order)
  collectgarbage("collect")

  print("Optimizing bit selection")
  if cfg.bits.sffs_fixed or not cfg.bits.sel then
    train.kept_bits = ivec.create(cfg.bits.sffs_fixed or dataset.n_hidden)
    train.kept_bits:fill_indices()
  else
    train.kept_bits = eval.optimize_bits({
      codes = codes_adj_order,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      n_dims = dataset.n_hidden,
      keep_prefix = cfg.bits.keep_prefix,
      start_prefix = cfg.bits.start_prefix,
      tolerance = cfg.bits.sffs_tolerance,
      metric = cfg.eval.bits_metric,
      each = function (bit, gain, score, action)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Bit  %-3d  %-12s | Gain: %2.12f | Score: %.12f\n",
          d, dd, bit, action, gain, score)
      end
    })
    collectgarbage("collect")
  end
  codes_adj_order:destroy()

  train.codes_spectral:bits_select(train.kept_bits, nil, dataset.n_hidden)
  dataset.n_hidden = train.kept_bits:size()
  dataset.n_latent = dataset.n_hidden * dataset.n_landmarks
  collectgarbage("collect")

  print("Creating spectral codes index")
  train.idx_spectral = cfg.index.ann
    and ann.create({ features = dataset.n_hidden, expected_size = train.ids_spectral:size() })
    or hbi.create({ features = dataset.n_hidden })
  train.idx_spectral:add(train.codes_spectral, train.ids_spectral)
  collectgarbage("collect")

  print("Setting up eval data")
  print("  Creating categorical adjacency for train")
  do
    local cat_index = inv.create({ features = 10, expected_size = train.ids_spectral:size(), decay = cfg.graph.decay })
    local data = ivec.create()
    data:copy(train.solutions)
    data:add_scaled(10)
    cat_index:add(data, train.ids_spectral)
    data:destroy()
    train.adj_eval_ids,
    train.adj_eval_offsets,
    train.adj_eval_neighbors,
    train.adj_eval_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.eval.anchors,
        random_pairs = cfg.eval.pairs,
      })
    cat_index:destroy()
  end
  print("  Creating categorical adjacency for test")
  do
    local cat_index = inv.create({ features = 10, expected_size = test.ids:size(), decay = cfg.graph.decay })
    local data = ivec.create()
    data:copy(test.solutions)
    data:add_scaled(10)
    cat_index:add(data, test.ids)
    data:destroy()
    test.adj_eval_ids,
    test.adj_eval_offsets,
    test.adj_eval_neighbors,
    test.adj_eval_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.eval.anchors,
        random_pairs = cfg.eval.pairs,
      })
    cat_index:destroy()
  end
  collectgarbage("collect")

  print("Codebook stats")
  train.entropy = eval.entropy_stats(train.codes_spectral, train.ids_spectral:size(), dataset.n_hidden)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    train.entropy)
  collectgarbage("collect")

  print("Retrieval stats (in-sample)")
  train.retrieval_scores = eval.score_retrieval({
    index = train.idx_spectral,
    ids = train.adj_eval_ids,
    offsets = train.adj_eval_offsets,
    neighbors = train.adj_eval_neighbors,
    weights = train.adj_eval_weights,
    metric = cfg.spectral.retrieval_metric,
  })
  for m = 0, train.retrieval_scores:size() - 1 do
    local d, dd = stopwatch()
    str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.10f\n",
      d, dd, m, train.retrieval_scores:get(m))
  end
  local best_score, best_idx = cfg.eval.retrieval(train.retrieval_scores)
  str.printf("Best\n  Margin: %d | Score: %+.10f\n", best_idx, best_score)
  collectgarbage("collect")

  local sth_n, sth_ids, sth_problems, sth_solutions, sth_visible
  local test_ids, test_problems
  local idx_train, idx_test

  if cfg.mode.encoder then

    if cfg.mode.mode == "raw" then
      print("Setting up raw features for STH")

      sth_ids = train.ids_spectral
      sth_n = sth_ids:size()
      sth_visible = dataset.n_visible

      sth_solutions = train.codes_spectral
      sth_problems = ivec.create()
      dataset.problems:bits_select(nil, sth_ids, dataset.n_visible, sth_problems)
      sth_problems = sth_problems:bits_to_cvec(sth_n, dataset.n_visible, true)

      test_ids = test.ids
      test_problems = ivec.create()
      dataset.problems:bits_select(nil, test.ids, dataset.n_visible, test_problems)
      test_problems = test_problems:bits_to_cvec(test.n, dataset.n_visible, true)
      collectgarbage("collect")

    elseif cfg.mode.mode == "landmarks" then
      print("Setting up landmark features for L-STH")

      sth_ids = train.ids_spectral
      sth_n = sth_ids:size()
      local sth_raw = ivec.create()
      dataset.problems:bits_select(nil, train.ids_spectral, dataset.n_visible, sth_raw)
      sth_raw = sth_raw:bits_to_cvec(sth_n, dataset.n_visible)
      local sth_idx = ann.create({ features = dataset.n_visible, expected_size = sth_ids:size() })
      sth_idx:add(sth_raw, sth_ids)

      local enc, n_latent = hlth.landmark_encoder({
        landmarks_index = sth_idx,
        codes_index = train.idx_spectral,
        n_landmarks = dataset.n_landmarks
      })
      dataset.n_latent = n_latent
      sth_visible = n_latent

      sth_solutions = train.idx_spectral:get(sth_ids)
      sth_problems = enc(sth_raw, sth_n)
      sth_problems:bits_flip_interleave(n_latent)
      test_ids = test.ids
      local test_vecs = ivec.create()
      dataset.problems:bits_select(nil, test.ids, dataset.n_visible, test_vecs)
      test_vecs = test_vecs:bits_to_cvec(test.n, dataset.n_visible)
      test_problems = enc(test_vecs, test.n)
      test_problems:bits_flip_interleave(n_latent)
      sth_idx:destroy()
      collectgarbage("collect")

    else
      err.error("Unexpected mode", cfg.mode.mode)
    end
    collectgarbage("collect")

    print("Creating encoder")
    stopwatch = utc.stopwatch()
    train.encoder, train.accuracy_predicted = tm.optimize_encoder({
      visible = sth_visible,
      hidden = dataset.n_hidden,
      sentences = sth_problems,
      codes = sth_solutions,
      samples = sth_n,
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
        local predicted = t:predict(sth_problems, sth_n)
        local accuracy = eval.encoding_accuracy(predicted, sth_solutions, sth_n, dataset.n_hidden)
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
    collectgarbage("collect")

    print("Final encoder performance (best checkpoint)")
    str.printi("  Train | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)", train.accuracy_predicted)

    -- Create encoder prediction indices once for reuse
    print("Creating encoder prediction indices")
    idx_train = cfg.index.ann
      and ann.create({ features = dataset.n_hidden, expected_size = sth_n })
      or hbi.create({ features = dataset.n_hidden })
    local sth_predicted = train.encoder:predict(sth_problems, sth_n)
    idx_train:add(sth_predicted, sth_ids)

    idx_test = cfg.index.ann
      and ann.create({ features = dataset.n_hidden, expected_size = test.n })
      or hbi.create({ features = dataset.n_hidden })
    local test_predicted = train.encoder:predict(test_problems, test.n)
    idx_test:add(test_predicted, test_ids)
    collectgarbage("collect")

    train.retrieval_scores_predicted = eval.score_retrieval({
      index = idx_train,
      ids = train.adj_eval_ids,
      offsets = train.adj_eval_offsets,
      weights = train.adj_eval_weights,
      neighbors = train.adj_eval_neighbors,
      metric = cfg.eval.retrieval_metric,
    })

    test.retrieval_scores_predicted = eval.score_retrieval({
      index = idx_test,
      ids = test.adj_eval_ids,
      offsets = test.adj_eval_offsets,
      weights = test.adj_eval_weights,
      neighbors = test.adj_eval_neighbors,
      metric = cfg.eval.retrieval_metric,
    })

    str.printi("  Codes  | Margin: %.2f#(2) | Score: %+.2f#(1)", { cfg.eval.retrieval(train.retrieval_scores) })
    str.printi("  Train  | Margin: %.2f#(2) | Score: %+.2f#(1)", { cfg.eval.retrieval(train.retrieval_scores_predicted) })
    str.printi("  Test   | Margin: %.2f#(2) | Score: %+.2f#(1)", { cfg.eval.retrieval(test.retrieval_scores_predicted) })
    print()
    collectgarbage("collect")

  end

  if cfg.mode.cluster then

    print("Clustering (in-sample)")
    train.adj_cluster_ids, train.adj_cluster_offsets, train.adj_cluster_neighbors =
      graph.adjacency({
        knn_index = train.idx_spectral,
        knn = cfg.cluster.knn
      })
    local codes_clusters = eval.cluster({
      codes = train.idx_spectral:get(train.adj_cluster_ids),
      n_dims = dataset.n_hidden,
      ids = train.adj_cluster_ids,
      offsets = train.adj_cluster_offsets,
      neighbors = train.adj_cluster_neighbors,
    })
    local codes_stats = eval.score_clustering({
      ids = codes_clusters.ids,
      offsets = codes_clusters.offsets,
      merges = codes_clusters.merges,
      eval_ids = train.adj_eval_ids,
      eval_offsets = train.adj_eval_offsets,
      eval_neighbors = train.adj_eval_neighbors,
      eval_weights = train.adj_eval_weights,
      metric = cfg.eval.cluster_metric,
    })
    for step = 0, codes_stats.n_steps do
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | Step: %2d | Score: %+.10f | Clusters: %d\n",
        d, dd, step, codes_stats.scores:get(step), codes_stats.n_clusters:get(step))
    end
    local best_score, best_step = cfg.eval.clustering(codes_stats.scores)
    local best_n_clusters = codes_stats.n_clusters:get(best_step)
    str.printf("Best\n  Step: %2d | Score: %+.10f | Clusters: %d\n", best_step, best_score, best_n_clusters)
    collectgarbage("collect")

    if cfg.mode.encoder then

      print("Clustering (train)")
      train.adj_pred_cluster_ids, train.adj_pred_cluster_offsets, train.adj_pred_cluster_neighbors =
        graph.adjacency({
          knn_index = idx_train,
          knn = cfg.cluster.knn
        })
      local train_clusters = eval.cluster({
        codes = idx_train:get(train.adj_pred_cluster_ids),
        n_dims = dataset.n_hidden,
        ids = train.adj_pred_cluster_ids,
        offsets = train.adj_pred_cluster_offsets,
        neighbors = train.adj_pred_cluster_neighbors,
      })
      local train_stats = eval.score_clustering({
        ids = train_clusters.ids,
        offsets = train_clusters.offsets,
        merges = train_clusters.merges,
        eval_ids = train.adj_eval_ids,
        eval_offsets = train.adj_eval_offsets,
        eval_neighbors = train.adj_eval_neighbors,
        eval_weights = train.adj_eval_weights,
        metric = cfg.eval.cluster_metric,
      })
      for step = 0, train_stats.n_steps do
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Score: %+.10f | Clusters: %d\n",
          d, dd, step, train_stats.scores:get(step), train_stats.n_clusters:get(step))
      end
      local best_score, best_step = cfg.eval.clustering(train_stats.scores)
      local best_n_clusters = train_stats.n_clusters:get(best_step)
      str.printf("Best\n  Step: %2d | Score: %+.10f | Clusters: %d\n", best_step, best_score, best_n_clusters)
      collectgarbage("collect")

      print("Clustering (test)")
      test.adj_pred_cluster_ids, test.adj_pred_cluster_offsets, test.adj_pred_cluster_neighbors =
        graph.adjacency({
          knn_index = idx_test,
          knn = cfg.cluster.knn
        })
      local test_clusters = eval.cluster({
        codes = idx_test:get(test.adj_pred_cluster_ids),
        n_dims = dataset.n_hidden,
        ids = test.adj_pred_cluster_ids,
        offsets = test.adj_pred_cluster_offsets,
        neighbors = test.adj_pred_cluster_neighbors,
      })
      local test_stats = eval.score_clustering({
        ids = test_clusters.ids,
        offsets = test_clusters.offsets,
        merges = test_clusters.merges,
        eval_ids = test.adj_eval_ids,
        eval_offsets = test.adj_eval_offsets,
        eval_neighbors = test.adj_eval_neighbors,
        eval_weights = test.adj_eval_weights,
        metric = cfg.eval.cluster_metric,
      })
      for step = 0, test_stats.n_steps do
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Score: %+.10f | Clusters: %d\n",
          d, dd, step, test_stats.scores:get(step), test_stats.n_clusters:get(step))
      end
      local best_score, best_step = cfg.eval.clustering(test_stats.scores)
      local best_n_clusters = test_stats.n_clusters:get(best_step)
      str.printf("Best\n  Step: %2d | Score: %+.10f | Clusters: %d\n", best_step, best_score, best_n_clusters)
      collectgarbage("collect")

    end
  end

end)
