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

local TTR = 0.9
local MAX = nil
local MAX_CLASS = nil
local THREADS = nil

local PRIMME_EPS = 1e-12
local VISIBLE = 784
local HIDDEN = 32
local LANDMARKS = 24
local MODE = "landmarks" -- or raw

local ENCODER = false
local CLUSTER = true
local BINARIZE = "median" -- or itq, sign

local TCH = true
local ITQ_EPS = 1e-8
local ITQ_ITERATIONS = 200

local CORRELATION = "spearman" -- spearman, kendall, spearman-weighted, kendall-weighted
local QUALITY = "biserial" -- biserial, variance
local LAPLACIAN = "unnormalized" -- unnormalized, normalized, random
local DECAY = 2.0

local KNN = 32
local KNN_EPS = nil
local KNN_MIN = nil
local KNN_MUTUAL = false
local CLUSTER_METHOD = "agglo" -- or prefix, graph
-- local CLUSTER_KNN = nil
-- local CLUSTER_KNN_MUTUAL = nil
local MIN_PTS = 32
local BRIDGE = true
local ANN = false -- true: ann, false: hbi

local SAMPLED_ANCHORS = 16

local RANKS = true
local KEEP_PREFIX = nil
local START_PREFIX = nil
local SFFS_TOLERANCE = nil
local SFFS_FIXED = nil

local CLAUSES = { def = 8, min = 8, max = 128, int = true, log = true, pow2 = true }
local CLAUSE_TOLERANCE = { def = 8, min = 8, max = 256, int = true, log = true, pow2 = true }
local CLAUSE_MAXIMUM = { def = 8, min = 8, max = 256, int = true, log = true, pow2 = true }
local TARGET = { def = 4, min = 2, max = 256, int = true, log = true, pow2 = true }
local SPECIFICITY = { def = 1000, min = 2, max = 4000, int = true, log = true }

local SEARCH_PATIENCE = 20
local SEARCH_ROUNDS = 20
local SEARCH_TRIALS = 4
local SEARCH_ITERATIONS = 20
local ITERATIONS = 100

test("tsetlin", function ()

  print("Reading data")

  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", VISIBLE, MAX, MAX_CLASS)
  dataset.n_visible = VISIBLE
  dataset.n_hidden = HIDDEN
  dataset.n_landmarks = LANDMARKS
  dataset.n_latent = HIDDEN * LANDMARKS
  collectgarbage("collect")

  print("Splitting")

  local train, test = ds.split_binary_mnist(dataset, TTR)
  if KNN then
    print("  Indexing original")
    local idx = ann.create({ features = dataset.n_visible, expected_size = train.n })
    local data = ivec.create()
    dataset.problems:bits_select(nil, train.ids, dataset.n_visible, data)
    idx:add(data:bits_to_cvec(train.n, dataset.n_visible), train.ids)
    print("  Neighborhoods")
    local ids, hoods = idx:neighborhoods(KNN, nil, KNN_EPS, KNN_MIN, KNN_MUTUAL, THREADS)
    train.seed = graph.star_hoods(ids, hoods, THREADS)
    idx:destroy()
    collectgarbage("collect")
  else
    train.seed = pvec.create()
  end
  collectgarbage("collect")

  print("Building the graph index")

  local graph_index
  if RANKS then

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
    graph_index = inv.create({ features = dataset.n_visible + 10, ranks = graph_ranks, decay = DECAY, n_ranks = 2 })
    graph_index:add(graph_problems, graph_ids)

  else

    -- Fully unsupervised
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
      bridge = BRIDGE,
      threads = THREADS,
      each = function (ids, s, b, dt)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Nodes: %-6d  Components: %-6d  Edges: %-6d\n", d, dd, dt, ids, s, b)
      end
    })
  graph_index:destroy()
  collectgarbage("collect")

  print("Spectral eigendecomposition")

  train.ids_spectral, train.codes_spectral = spectral.encode({
    type = LAPLACIAN,
    eps = PRIMME_EPS,
    ids = train.adj_ids,
    offsets = train.adj_offsets,
    neighbors = train.adj_neighbors,
    weights = train.adj_weights,
    n_hidden = dataset.n_hidden,
    threads = THREADS,
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
  if BINARIZE == "itq" then
    print("Iterative Quantization")
    train.codes_spectral = itq.encode({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      tolerance = ITQ_EPS,
      iterations = ITQ_ITERATIONS,
      threads = THREADS,
      each = function (i, a, b)
        str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
      end
    })
  elseif BINARIZE == "median" then
    print("Median thresholding")
    train.codes_spectral = itq.median({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
    })
  elseif BINARIZE == "sign" then
    print("Sign thresholding")
    train.codes_spectral = itq.sign({
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
    })
  end
  collectgarbage("collect")


  if TCH then
    print("Flipping bits")
    tch.refine({
      ids = train.adj_ids,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      threads = THREADS,
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
    = graph.multiclass_pairs(train.ids_spectral, train.solutions_spectral, SAMPLED_ANCHORS, SAMPLED_ANCHORS, THREADS)
  train.adj_sampled_ids, -- == train.ids_spectral
  train.adj_sampled_offsets,
  train.adj_sampled_neighbors,
  train.adj_sampled_weights
    = graph.adj_pairs(train.ids_sampled, train.pos_sampled, train.neg_sampled, THREADS)
  print("  Sampling test")
  test.ids_sampled,
  test.pos_sampled,
  test.neg_sampled
    = graph.multiclass_pairs(test.ids, test.solutions, SAMPLED_ANCHORS, SAMPLED_ANCHORS, THREADS)
  test.adj_sampled_ids, -- == test.ids
  test.adj_sampled_offsets,
  test.adj_sampled_neighbors,
  test.adj_sampled_weights
    = graph.adj_pairs(test.ids, test.pos_sampled, test.neg_sampled, THREADS)
  collectgarbage("collect")

  print("Optimizing bit selection")
  if SFFS_FIXED then
    train.kept_bits = ivec.create(SFFS_FIXED)
    train.kept_bits:fill_indices()
  else
    train.kept_bits = eval.optimize_bits({
      ids = train.ids_spectral,
      codes = train.codes_spectral,
      n_dims = dataset.n_hidden,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      keep_prefix = KEEP_PREFIX,
      start_prefix = START_PREFIX,
      tolerance = SFFS_TOLERANCE,
      correlation = CORRELATION,
      threads = THREADS,
      each = function (bit, gain, score, action)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Bit  %-3d  %-12s | Gain: %2.12f | Score: %.12f\n",
          d, dd, bit, action, gain, score)
      end
    })
    collectgarbage("collect")
  end

  -- print("AUC stats")
  -- train.auc_continuous = eval.auc(
  --   train.adj_sampled_ids, -- == train.ids_spectral
  --   train.adj_sampled_offsets,
  --   train.adj_sampled_neighbors,
  --   train.adj_sampled_weights,
  --   train.codes_spectral_cont,
  --   dataset.n_hidden, nil, THREADS)
  -- train.auc_binary_full = eval.auc(
  --   train.adj_sampled_ids, -- == train.ids_spectral
  --   train.adj_sampled_offsets,
  --   train.adj_sampled_neighbors,
  --   train.adj_sampled_weights,
  --   train.codes_spectral,
  --   dataset.n_hidden, nil, THREADS)
  train.codes_spectral:bits_select(train.kept_bits, nil, dataset.n_hidden)
  -- dataset.n_hidden_full = dataset.n_hidden
  dataset.n_hidden = train.kept_bits:size()
  dataset.n_latent = dataset.n_hidden * dataset.n_landmarks  -- Recalculate after bit selection
  -- train.auc_binary = eval.auc(train.ids_spectral, train.codes_spectral, train.adj_sampled_offsets, train.adj_sampled_neighbors, train.adj_sampled_weights, dataset.n_hidden, nil, THREADS)
  -- str.printf("  AUC (continuous):        %.4f\n", train.auc_continuous)
  -- str.printf("  AUC (binary, all bits):  %.4f  %d bits\n", train.auc_binary_full, dataset.n_hidden_full)
  -- str.printf("  AUC (binary, subset):    %.4f  %d bits\n", train.auc_binary, dataset.n_hidden)
  -- collectgarbage("collect")

  print("Codebook stats")
  train.entropy = eval.entropy_stats(train.codes_spectral, train.ids_spectral:size(), dataset.n_hidden, THREADS)
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
    quality = QUALITY,
    threads = THREADS,
    each = function (acc)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.6f\n",
        d, dd, acc.margin, acc.score)
    end
  })
  local best_score, best_idx = train.retrieval_scores:max()
  str.printf("  Best | Margin: %d | Score: %+.6f\n", best_idx, best_score)
  collectgarbage("collect")

  print("Retrieval stats (sampled binary adjacency)")
  train.retrieval_scores = eval.optimize_retrieval({
    codes = train.codes_spectral,
    n_dims = dataset.n_hidden,
    ids = train.adj_sampled_ids,
    offsets = train.adj_sampled_offsets,
    neighbors = train.adj_sampled_neighbors,
    weights = train.adj_sampled_weights,
    quality = QUALITY,
    threads = THREADS,
    each = function (acc)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.6f\n",
        d, dd, acc.margin, acc.score)
    end
  })
  local best_score, best_idx = train.retrieval_scores:max()
  str.printf("  Best | Margin: %d | Score: %+.6f\n", best_idx, best_score)
  collectgarbage("collect")

  if ENCODER or CLUSTER then
    train.idx_spectral = ANN
      and ann.create({ features = dataset.n_hidden, expected_size = train.ids_spectral:size() })
      or hbi.create({ features = dataset.n_hidden })
    train.idx_spectral:add(train.codes_spectral, train.ids_spectral)
    collectgarbage("collect")
  end

  local sth_n, sth_ids, sth_problems, sth_solutions, sth_visible
  local test_ids, test_problems

  if ENCODER then

    if MODE == "raw" then
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

    elseif MODE == "landmarks" then
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
      sth_ids, sth_hoods = sth_idx:neighborhoods(dataset.n_landmarks, nil, nil, nil, nil, THREADS)

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
      local nbr_ids, nbr_hoods = sth_idx:neighborhoods_by_vecs(test_vecs, dataset.n_landmarks, nil, nil, nil, THREADS)
      for i0, hood in nbr_hoods:ieach() do
        hood:keys(tmp)
        tmp:lookup(nbr_ids)
        train.idx_spectral:get(tmp, test_problems, i0, dataset.n_latent)
      end
      test_problems:bits_flip_interleave(dataset.n_latent)
      sth_idx:destroy()
      collectgarbage("collect")

    else
      err.error("Unexpected mode", MODE)
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
      clauses = CLAUSES,
      clause_tolerance = CLAUSE_TOLERANCE,
      clause_maximum = CLAUSE_MAXIMUM,
      target = TARGET,
      specificity = SPECIFICITY,
      search_patience = SEARCH_PATIENCE,
      search_rounds = SEARCH_ROUNDS,
      search_trials = SEARCH_TRIALS,
      search_iterations = SEARCH_ITERATIONS,
      final_iterations = ITERATIONS,
      threads = THREADS,
      search_metric = function (t)
        local predicted = t:predict(sth_problems, sth_n, THREADS)
        local accuracy = eval.encoding_accuracy(predicted, sth_solutions, sth_n, dataset.n_hidden, THREADS)
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
          local sth_predicted = t:predict(sth_problems, sth_n, THREADS)
          -- train.auc_predicted = eval.auc(sth_ids, sth_predicted, train.adj_sampled_offsets, train.adj_sampled_neighbors, train.adj_sampled_weights, dataset.n_hidden, nil, THREADS)
          train.retrieval_scores = eval.optimize_retrieval({
            codes = sth_predicted,
            n_dims = dataset.n_hidden,
            ids = sth_ids,
            offsets = train.adj_offsets,
            weights = train.adj_weights,
            neighbors = train.adj_neighbors,
            quality = QUALITY,
            threads = THREADS,
          })
          local test_predicted = t:predict(test_problems, test.n, THREADS)
          -- test.auc_predicted = eval.auc(test_ids, test_predicted, test.adj_sampled_offsets, test.adj_sampled_neighbors, test.adj_sampled_weights, dataset.n_hidden, nil, THREADS)
          test.retrieval_scores_predicted = eval.optimize_retrieval({
            codes = test_predicted,
            n_dims = dataset.n_hidden,
            ids = test_ids,
            offsets = train.adj_sampled_offsets,
            weights = train.adj_sampled_weights,
            neighbors = train.adj_sampled_neighbors,
            quality = QUALITY,
            threads = THREADS,
          })
          train.retrieval_scores = eval.optimize_retrieval({
            codes = test_predicted,
            n_dims = dataset.n_hidden,
            ids = test_ids,
            offsets = test.adj_sampled_offsets,
            weights = test.adj_sampled_weights,
            neighbors = test.adj_sampled_neighbors,
            quality = QUALITY,
            threads = THREADS,
          })
          -- train.similarity.auc = train.auc_binary
          -- train.similarity_predicted.auc = train.auc_predicted
          -- test.similarity_predicted.auc = test.auc_predicted
          str.printi("    Codes  | Margin: %.2f#(2) | Score: %+.2f#(1)", { train.retrieval_scores:max() })
          str.printi("    Train  | Margin: %.2f#(2) | Score: %+.2f#(1)", { train.retrieval_scores_predicted:max() })
          str.printi("    Test   | Margin: %.2f#(2) | Score: %+.2f#(1)", { test.retrieval_scores:max() })
          -- str.printi("    Codes | AUC: %.2f#(auc) | Margin: %.2f#(margin)", train.similarity)
          -- str.printi("    Train | AUC: %.2f#(auc) | Margin: %.2f#(margin)", train.similarity_predicted)
          -- str.printi("    Test  | AUC: %.2f#(auc) | Margin: %.2f#(margin)", test.similarity_predicted)
          print()
        end
      end,
    })
    collectgarbage("collect")

  end

  if CLUSTER then

    print("Clustering (codes) (graph edges)")
    local codes_stats = eval.optimize_clustering({
      index = train.idx_spectral,
      ids = train.ids_spectral,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      method = CLUSTER_METHOD,
      -- knn = CLUSTER_KNN,
      -- knn_mutual = CLUSTER_KNN_MUTUAL,
      min_pts = MIN_PTS,
      assign_noise = true,
      quality = QUALITY,
      threads = THREADS,
      each = function (acc)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Score: %+.6f | Clusters: %d\n",
          d, dd, acc.step, acc.score, acc.n_clusters)
      end
    })
    if codes_stats.scores then
      local best_score, best_step = codes_stats.scores:max()
      local best_n_clusters = codes_stats.n_clusters:get(best_step)
      str.printf("  Best | Step: %2d | Score: %+.6f | Clusters: %d\n", best_step, best_score, best_n_clusters)
    end
    collectgarbage("collect")

    print("Clustering (codes) (sampled binary adjacency)")
    local codes_stats = eval.optimize_clustering({
      index = train.idx_spectral,
      ids = train.ids_spectral,
      offsets = train.adj_sampled_offsets,
      neighbors = train.adj_sampled_neighbors,
      weights = train.adj_sampled_weights,
      method = CLUSTER_METHOD,
      knn = CLUSTER_KNN,
      knn_mutual = CLUSTER_KNN_MUTUAL,
      min_pts = MIN_PTS,
      assign_noise = true,
      quality = QUALITY,
      threads = THREADS,
      each = function (acc)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Score: %+.6f | Clusters: %d\n",
          d, dd, acc.step, acc.score, acc.n_clusters)
      end
    })
    if codes_stats.scores then
      local best_score, best_step = codes_stats.scores:max()
      local best_n_clusters = codes_stats.n_clusters:get(best_step)
      str.printf("  Best | Step: %2d | Score: %+.6f | Clusters: %d\n", best_step, best_score, best_n_clusters)
    end
    collectgarbage("collect")

    if ENCODER then

      print("Clustering (train)")
      local idx_train = ANN
        and ann.create({ features = dataset.n_hidden, expected_size = train.n })
        or hbi.create({ features = dataset.n_hidden })
      idx_train:add(train.encoder:predict(sth_problems, sth_n, THREADS), sth_ids)
      local train_stats = eval.optimize_clustering({
        index = idx_train,
        ids = train.ids_spectral,
        method = CLUSTER_METHOD,
        knn = CLUSTER_KNN,
        knn_mutual = CLUSTER_KNN_MUTUAL,
        min_pts = MIN_PTS,
        assign_noise = true,
        quality = QUALITY,
        threads = THREADS,
        each = function (acc)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.6f | Clusters: %d\n",
            d, dd, acc.margin, acc.score, acc.n_clusters)
        end
      })
      if train_stats.scores then
        local best_score, best_step = train_stats.scores:max()
        local best_n_clusters = train_stats.n_clusters:get(best_step)
        str.printf("  Best | Step: %2d | Score: %+.6f | Clusters: %d\n", best_step, best_score, best_n_clusters)
      end
      collectgarbage("collect")

      -- TODO: Support clustering_accuracy_graph for test by constructing a
      -- parallel test graph
      print("Clustering (test)")
      local idx_test = ANN
        and ann.create({ features = dataset.n_hidden, expected_size = test.n })
        or hbi.create({ features = dataset.n_hidden })
      idx_test:add(train.encoder:predict(test_problems, test.n, THREADS), test_ids)
      local test_stats = eval.optimize_clustering({
        index = idx_test,
        ids = test.ids,
        method = CLUSTER_METHOD,
        knn = CLUSTER_KNN,
        knn_mutual = CLUSTER_KNN_MUTUAL,
        min_pts = MIN_PTS,
        assign_noise = true,
        quality = QUALITY,
        threads = THREADS,
        each = function (acc)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.6f | Clusters: %d\n",
            d, dd, acc.margin, acc.score, acc.n_clusters)
        end
      })
      if test_stats.scores then
        local best_step, best_score = test_stats.scores:argmax()
        local best_n_clusters = test_stats.n_clusters:get(best_step)
        str.printf("  Best | Step: %2d | Score: %+.6f | Clusters: %d\n", best_step, best_score, best_n_clusters)
      end
      collectgarbage("collect")

    end
  end

end)
