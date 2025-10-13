require("santoku.dvec")
local ds = require("santoku.tsetlin.dataset")
local err = require("santoku.error")
local eval = require("santoku.tsetlin.evaluator")
local tm = require("santoku.tsetlin")
local inv = require("santoku.tsetlin.inv")
local ann = require("santoku.tsetlin.ann")
local hbi = require("santoku.tsetlin.hbi")
local pvec = require("santoku.pvec")
local ivec = require("santoku.ivec")
local cvec = require("santoku.cvec")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local cfg; cfg = {
  data = {
    ttr = 0.9,
    max = nil,
    max_class = nil,
    visible = 784,
    hidden = 32,
    landmarks = 24,
  },
  mode = {
    encoder = true,
    cluster = true,
    mode = "landmarks",
    binarize = "median",
    tch = true,
    ranks = true,
  },
  index = {
    ann = false,
  },
  spectral = {
    laplacian = "unnormalized",
    decay = 2.0,
    primme_eps = 1e-12,
  },
  itq = {
    eps = 1e-8,
    iterations = 200,
  },
  graph = {
    knn = 32,
    knn_eps = nil,
    knn_min = nil,
    knn_mutual = false,
    bridge = true,
  },
  clustering = {
    linkage = "simhash",
    knn = 32,
    knn_min = nil,
    knn_mutual = false,
    min_pts = 32,
  },
  eval = {
    bits_metric = "correlation",
    retrieval_metric = "precision",
    cluster_metric = "precision",
    sampled_anchors = 16,
    tolerance = 1e-3,
    retrieval = function (d)
      return d:max()
      -- return d:scores_plateau(cfg.eval.tolerance)
    end,
    clustering = function (d)
      return d:max()
      -- return d:scores_plateau(cfg.eval.tolerance)
    end,
  },
  bits = {
    keep_prefix = nil,
    start_prefix = nil,
    sffs_tolerance = nil,
    sffs_fixed = nil,
  },
  tm = {
    clauses = { def = 8, min = 8, max = 128, int = true, log = true, pow2 = true },
    clause_tolerance = { def = 8, min = 8, max = 256, int = true, log = true, pow2 = true },
    clause_maximum = { def = 8, min = 8, max = 256, int = true, log = true, pow2 = true },
    target = { def = 4, min = 2, max = 256, int = true, log = true, pow2 = true },
    specificity = { def = 1000, min = 2, max = 4000, int = true, log = true },
  },
  search = {
    patience = 20,
    rounds = 20,
    trials = 4,
    iterations = 20,
  },
  training = {
    iterations = 100,
  },
  threads = nil,
}

test("tsetlin", function ()

  print("Reading data")

  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.visible, cfg.data.max, cfg.data.max_class)
  dataset.n_visible = cfg.data.visible
  dataset.n_hidden = cfg.data.hidden
  dataset.n_landmarks = cfg.data.landmarks
  dataset.n_latent = cfg.data.hidden * cfg.data.landmarks
  collectgarbage("collect")

  print("Splitting")

  local train, test = ds.split_binary_mnist(dataset, cfg.data.ttr)
  if cfg.graph.knn then
    print("  Indexing original")
    local idx = ann.create({ features = dataset.n_visible, expected_size = train.n })
    local data = ivec.create()
    dataset.problems:bits_select(nil, train.ids, dataset.n_visible, data)
    idx:add(data:bits_to_cvec(train.n, dataset.n_visible), train.ids)
    print("  Neighborhoods")
    local ids, hoods = idx:neighborhoods(cfg.graph.knn, nil, 0, cfg.graph.knn_eps, cfg.graph.knn_min, cfg.graph.knn_mutual, cfg.threads)
    train.seed = graph.star_hoods(ids, hoods, cfg.threads)
    idx:destroy()
    collectgarbage("collect")
  else
    train.seed = pvec.create()
  end
  collectgarbage("collect")

  print("Building the graph index")

  local graph_index
  if cfg.mode.ranks then

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

    print("  Indexing")
    graph_index = inv.create({ features = dataset.n_visible + 10, ranks = graph_ranks, decay = cfg.spectral.decay, n_ranks = 2 })
    graph_index:add(graph_problems, graph_ids)

  else

    print("  Indexing")
    local graph_problems = ivec.create()
    dataset.problems:bits_select(nil, train.ids, dataset.n_visible, graph_problems)
    graph_index = inv.create({ features = dataset.n_visible })
    graph_index:add(graph_problems, train.ids)
    collectgarbage("collect")

  end

  print("Creating graph")

  local stopwatch = utc.stopwatch()
  train.adj_ids,
  train.adj_offsets,
  train.adj_neighbors,
  train.adj_weights =
    graph.adjacency({
      edges = train.seed,
      index = graph_index,
      bridge = cfg.graph.bridge,
      threads = cfg.threads,
      each = function (ids, s, b, dt)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Nodes: %-6d  Components: %-6d  Edges: %-6d\n", d, dd, dt, ids, s, b)
      end
    })
  graph_index:destroy()
  collectgarbage("collect")

  print("Spectral eigendecomposition")

  train.ids_spectral, train.codes_spectral = spectral.encode({
    type = cfg.spectral.laplacian,
    eps = cfg.spectral.primme_eps,
    ids = train.adj_ids,
    offsets = train.adj_offsets,
    neighbors = train.adj_neighbors,
    weights = train.adj_weights,
    n_hidden = dataset.n_hidden,
    threads = cfg.threads,
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
  collectgarbage("collect")

  train.codes_spectral_cont = train.codes_spectral
  if cfg.mode.binarize == "itq" then
    print("Iterative Quantization")
    train.codes_spectral = itq.encode({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      tolerance = cfg.itq.eps,
      iterations = cfg.itq.iterations,
      threads = cfg.threads,
      each = function (i, a, b)
        str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
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
  end
  collectgarbage("collect")

  if cfg.mode.tch then
    print("Flipping bits")
    tch.refine({
      ids = train.adj_ids,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      threads = cfg.threads,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end
  collectgarbage("collect")

  print("Setting up eval data")
  print("  Known labels")
  train.solutions_spectral = ivec.create()
  train.solutions_spectral:copy(train.solutions, train.ids_spectral)
  print("  Sampling train")
  train.ids_sampled,
  train.pos_sampled,
  train.neg_sampled
    = graph.multiclass_pairs(train.ids_spectral, train.solutions_spectral, cfg.eval.sampled_anchors, cfg.eval.sampled_anchors, cfg.threads)
  train.adj_sampled_ids,
  train.adj_sampled_offsets,
  train.adj_sampled_neighbors,
  train.adj_sampled_weights
    = graph.adj_pairs(train.ids_sampled, train.pos_sampled, train.neg_sampled, cfg.threads)
  print("  Sampling test")
  test.ids_sampled,
  test.pos_sampled,
  test.neg_sampled
    = graph.multiclass_pairs(test.ids, test.solutions, cfg.eval.sampled_anchors, cfg.eval.sampled_anchors, cfg.threads)
  test.adj_sampled_ids,
  test.adj_sampled_offsets,
  test.adj_sampled_neighbors,
  test.adj_sampled_weights
    = graph.adj_pairs(test.ids, test.pos_sampled, test.neg_sampled, cfg.threads)
  collectgarbage("collect")

  print("Optimizing bit selection")
  if cfg.bits.sffs_fixed then
    train.kept_bits = ivec.create(cfg.bits.sffs_fixed)
    train.kept_bits:fill_indices()
  else
    train.kept_bits = eval.optimize_bits({
      ids = train.ids_spectral,
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      keep_prefix = cfg.bits.keep_prefix,
      start_prefix = cfg.bits.start_prefix,
      tolerance = cfg.bits.sffs_tolerance,
      metric = cfg.eval.bits_metric,
      threads = cfg.threads,
      each = function (bit, gain, score, action)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Bit  %-3d  %-12s | Gain: %2.12f | Score: %.12f\n",
          d, dd, bit, action, gain, score)
      end
    })
    collectgarbage("collect")
  end

  train.codes_spectral:bits_select(train.kept_bits, nil, dataset.n_hidden)
  dataset.n_hidden = train.kept_bits:size()
  dataset.n_latent = dataset.n_hidden * dataset.n_landmarks
  collectgarbage("collect")

  print("Codebook stats")
  train.entropy = eval.entropy_stats(train.codes_spectral, train.ids_spectral:size(), dataset.n_hidden, cfg.threads)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    train.entropy)
  collectgarbage("collect")

  print("Retrieval stats (graph edges)")
  train.retrieval_scores = eval.optimize_retrieval({
    codes = train.codes_spectral,
    n_dims = dataset.n_hidden,
    ids = train.adj_sampled_ids,
    offsets = train.adj_offsets,
    neighbors = train.adj_neighbors,
    weights = train.adj_weights,
    metric = cfg.eval.retrieval_metric,
    threads = cfg.threads,
    each = function (acc)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.6f\n",
        d, dd, acc.margin, acc.score)
    end
  })
  local best_score, best_idx = train.retrieval_scores:max()
  str.printf("Best\n  Margin: %d | Score: %+.6f\n", best_idx, best_score)
  collectgarbage("collect")

  print("Retrieval stats (sampled binary adjacency)")
  train.retrieval_scores = eval.optimize_retrieval({
    codes = train.codes_spectral,
    n_dims = dataset.n_hidden,
    ids = train.adj_sampled_ids,
    offsets = train.adj_sampled_offsets,
    neighbors = train.adj_sampled_neighbors,
    weights = train.adj_sampled_weights,
    metric = cfg.eval.retrieval_metric,
    threads = cfg.threads,
    each = function (acc)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.6f\n",
        d, dd, acc.margin, acc.score)
    end
  })
  local best_score, best_idx = train.retrieval_scores:max()
  str.printf("Best\n  Margin: %d | Score: %+.6f\n", best_idx, best_score)
  collectgarbage("collect")

  if cfg.mode.encoder or cfg.mode.cluster then
    train.idx_spectral = cfg.index.ann
      and ann.create({ features = dataset.n_hidden, expected_size = train.ids_spectral:size() })
      or hbi.create({ features = dataset.n_hidden })
    train.idx_spectral:add(train.codes_spectral, train.ids_spectral)
    collectgarbage("collect")
  end

  local sth_n, sth_ids, sth_problems, sth_solutions, sth_visible
  local test_ids, test_problems

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
      sth_visible = dataset.n_latent

      local sth_raw = ivec.create()
      dataset.problems:bits_select(nil, train.ids_spectral, dataset.n_visible, sth_raw)
      sth_raw = sth_raw:bits_to_cvec(sth_n, dataset.n_visible)

      local sth_idx = ann.create({ features = dataset.n_visible, expected_size = sth_ids:size() })
      sth_idx:add(sth_raw, sth_ids)

      local sth_hoods
      sth_ids, sth_hoods = sth_idx:neighborhoods(dataset.n_landmarks, nil, nil, nil, nil, cfg.threads)

      sth_solutions = train.idx_spectral:get(sth_ids)
      sth_problems = cvec.create()
      local tmp = ivec.create()
      for i0, hood in sth_hoods:ieach() do
        hood:keys(tmp)
        tmp:lookup(sth_ids)
        train.idx_spectral:get(tmp, sth_problems, i0, dataset.n_latent)
      end
      sth_problems:bits_flip_interleave(dataset.n_latent)

      test_ids = test.ids
      test_problems = cvec.create()
      local test_vecs = ivec.create()
      dataset.problems:bits_select(nil, test.ids, dataset.n_visible, test_vecs)
      test_vecs = test_vecs:bits_to_cvec(test.n, dataset.n_visible)
      local nbr_ids, nbr_hoods = sth_idx:neighborhoods_by_vecs(test_vecs, dataset.n_landmarks, nil, nil, nil, cfg.threads)
      for i0, hood in nbr_hoods:ieach() do
        hood:keys(tmp)
        tmp:lookup(nbr_ids)
        train.idx_spectral:get(tmp, test_problems, i0, dataset.n_latent)
      end
      test_problems:bits_flip_interleave(dataset.n_latent)
      sth_idx:destroy()
      collectgarbage("collect")

    else
      err.error("Unexpected mode", cfg.mode.mode)
    end
    collectgarbage("collect")

    print("Creating encoder")
    stopwatch = utc.stopwatch()
    train.encoder = tm.optimize_encoder({
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
      final_iterations = cfg.training.iterations,
      threads = cfg.threads,
      search_metric = function (t)
        local predicted = t:predict(sth_problems, sth_n, cfg.threads)
        local accuracy = eval.encoding_accuracy(predicted, sth_solutions, sth_n, dataset.n_hidden, cfg.threads)
        return accuracy.mean_hamming, accuracy
      end,
      each = function (t, is_final, train_accuracy, params, epoch, round, trial)
        local d, dd = stopwatch()
        if is_final then
          str.printf("  Time %3.2f %3.2f  Finalizing  C=%d LF=%d L=%d T=%d S=%.2f  Epoch  %d\n",
            d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, epoch)
        else
          str.printf("  Time %3.2f %3.2f  Exploring  C=%d LF=%d L=%d T=%d S=%.2f  R=%d T=%d  Epoch  %d\n",
            d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, round, trial, epoch)
        end
        train.accuracy_predicted = train_accuracy
        str.printi("    Train | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)", train.accuracy_predicted)
        if is_final then
          local sth_predicted = t:predict(sth_problems, sth_n, cfg.threads)
          train.retrieval_scores = eval.optimize_retrieval({
            codes = sth_predicted,
            n_dims = dataset.n_hidden,
            ids = sth_ids,
            offsets = train.adj_offsets,
            weights = train.adj_weights,
            neighbors = train.adj_neighbors,
            metric = cfg.eval.retrieval_metric,
            threads = cfg.threads,
          })
          local test_predicted = t:predict(test_problems, test.n, cfg.threads)
          test.retrieval_scores_predicted = eval.optimize_retrieval({
            codes = test_predicted,
            n_dims = dataset.n_hidden,
            ids = test_ids,
            offsets = train.adj_sampled_offsets,
            weights = train.adj_sampled_weights,
            neighbors = train.adj_sampled_neighbors,
            metric = cfg.eval.retrieval_metric,
            threads = cfg.threads,
          })
          train.retrieval_scores = eval.optimize_retrieval({
            codes = test_predicted,
            n_dims = dataset.n_hidden,
            ids = test_ids,
            offsets = test.adj_sampled_offsets,
            weights = test.adj_sampled_weights,
            neighbors = test.adj_sampled_neighbors,
            metric = cfg.eval.retrieval_metric,
            threads = cfg.threads,
          })
          str.printi("    Codes  | Margin: %.2f#(2) | Score: %+.2f#(1)", { train.retrieval_scores:max() })
          str.printi("    Train  | Margin: %.2f#(2) | Score: %+.2f#(1)", { train.retrieval_scores_predicted:max() })
          str.printi("    Test   | Margin: %.2f#(2) | Score: %+.2f#(1)", { test.retrieval_scores:max() })
          print()
        end
      end,
    })
    collectgarbage("collect")

  end

  if cfg.mode.cluster then

    print("Clustering (codes) (graph edges)")
    local codes_stats = eval.optimize_clustering({
      index = train.idx_spectral,
      ids = train.ids_spectral,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      linkage = cfg.clustering.linkage,
      knn = cfg.clustering.knn,
      knn_min = cfg.clustering.knn_min,
      knn_mutual = cfg.clustering.knn_mutual,
      min_pts = cfg.clustering.min_pts,
      assign_noise = true,
      metric = cfg.eval.cluster_metric,
      threads = cfg.threads,
      each = function (acc)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Score: %+.6f | Clusters: %d\n",
          d, dd, acc.step, acc.score, acc.n_clusters)
      end
    })
    if codes_stats.scores then
      local best_score, best_step = cfg.eval.clustering(codes_stats.scores)
      local best_n_clusters = codes_stats.n_clusters:get(best_step)
      str.printf("Best\n  Step: %2d | Score: %+.6f | Clusters: %d\n", best_step, best_score, best_n_clusters)
      -- print("Validating dendrogram cuts.")
      -- for step, _, cut_assignments in eval.dendro_each(codes_stats.offsets, codes_stats.merges) do
      --   local cut_result = eval.clustering_accuracy({
      --     assignments = cut_assignments,
      --     offsets = train.adj_offsets,
      --     neighbors = train.adj_neighbors,
      --     weights = train.adj_weights,
      --     metric = cfg.eval.cluster_metric,
      --     threads = cfg.threads
      --   })
      --   local expected_score = codes_stats.scores:get(step)
      --   local actual_score = cut_result.score
      --   str.printf("  Step %d:  expected %.6f  got %.6f\n", step, expected_score, actual_score)
      -- end
    end
    collectgarbage("collect")

    print("Clustering (codes) (sampled binary adjacency)")
    local codes_stats = eval.optimize_clustering({
      index = train.idx_spectral,
      ids = train.ids_spectral,
      offsets = train.adj_sampled_offsets,
      neighbors = train.adj_sampled_neighbors,
      weights = train.adj_sampled_weights,
      linkage = cfg.clustering.linkage,
      knn = cfg.clustering.knn,
      knn_min = cfg.clustering.knn_min,
      knn_mutual = cfg.clustering.knn_mutual,
      min_pts = cfg.clustering.min_pts,
      assign_noise = true,
      metric = cfg.eval.cluster_metric,
      threads = cfg.threads,
      each = function (acc)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Score: %+.6f | Clusters: %d\n",
          d, dd, acc.step, acc.score, acc.n_clusters)
      end
    })
    if codes_stats.scores then
      local best_score, best_step = cfg.eval.clustering(codes_stats.scores)
      local best_n_clusters = codes_stats.n_clusters:get(best_step)
      str.printf("Best\n  Step: %2d | Score: %+.6f | Clusters: %d\n", best_step, best_score, best_n_clusters)
    end
    collectgarbage("collect")

    if cfg.mode.encoder then

      print("Clustering (train)")
      local idx_train = cfg.index.ann
        and ann.create({ features = dataset.n_hidden, expected_size = train.n })
        or hbi.create({ features = dataset.n_hidden })
      idx_train:add(train.encoder:predict(sth_problems, sth_n, cfg.threads), sth_ids)
      local train_stats = eval.optimize_clustering({
        index = idx_train,
        ids = train.ids_spectral,
        linkage = cfg.clustering.linkage,
        assign_noise = true,
        knn = cfg.clustering.knn,
        knn_min = cfg.clustering.knn_min,
        knn_mutual = cfg.clustering.knn_mutual,
        min_pts = cfg.clustering.min_pts,
        metric = cfg.eval.cluster_metric,
        threads = cfg.threads,
        each = function (acc)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.6f | Clusters: %d\n",
            d, dd, acc.margin, acc.score, acc.n_clusters)
        end
      })
      if train_stats.scores then
        local best_score, best_step = cfg.eval.clustering(train_stats.scores)
        local best_n_clusters = train_stats.n_clusters:get(best_step)
        str.printf("Best\n  Step: %2d | Score: %+.6f | Clusters: %d\n", best_step, best_score, best_n_clusters)
      end
      collectgarbage("collect")

      print("Clustering (test)")
      local idx_test = cfg.index.ann
        and ann.create({ features = dataset.n_hidden, expected_size = test.n })
        or hbi.create({ features = dataset.n_hidden })
      idx_test:add(train.encoder:predict(test_problems, test.n, cfg.threads), test_ids)
      local test_stats = eval.optimize_clustering({
        index = idx_test,
        ids = test.ids,
        linkage = cfg.clustering.linkage,
        assign_noise = true,
        knn = cfg.clustering.knn,
        knn_min = cfg.clustering.knn_min,
        knn_mutual = cfg.clustering.knn_mutual,
        min_pts = cfg.clustering.min_pts,
        metric = cfg.eval.cluster_metric,
        threads = cfg.threads,
        each = function (acc)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.6f | Clusters: %d\n",
            d, dd, acc.margin, acc.score, acc.n_clusters)
        end
      })
      if test_stats.scores then
        local best_step, best_score = cfg.eval.clustering(test_stats.scores)
        local best_n_clusters = test_stats.n_clusters:get(best_step)
        str.printf("Best\n  Step: %2d | Score: %+.6f | Clusters: %d\n", best_step, best_score, best_n_clusters)
      end
      collectgarbage("collect")

    end
  end

end)
