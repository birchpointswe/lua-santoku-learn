local ds = require("santoku.tsetlin.dataset")
local err = require("santoku.error")
local eval = require("santoku.tsetlin.evaluator")
local tm = require("santoku.tsetlin")
local inv = require("santoku.tsetlin.inv")
local ann = require("santoku.tsetlin.ann")
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

local VISIBLE = 784
local HIDDEN = 32
local LANDMARKS = 24
local MODE = "landmarks" -- "landmarks" for L-STH, "raw" for STH

local ENCODER = true
local CLUSTER = true
local BINARIZE = "median" -- "itq", "median", or "sign"

local TCH = true
local SPECTRAL_EPS = 1e-5
local ITQ_EPS = 1e-8
local ITQ_ITERATIONS = 200

local LAPLACIAN = "random"
local DECAY = 1.0

local KNN = 24
local KNN_EPS = nil
local KNN_MIN = nil
local KNN_MUTUAL = true
local BRIDGE = true
local MIN_PTS = 256

local SEED_PAIRS = 0
local SEED_ANCHORS = 0
local SEED_CLASS_ANCHORS = 0
local SAMPLED_ANCHORS = 16
local SIGMA_K = nil

local RANKS = true
local SFFS_FIXED = nil

local CLAUSES = { def = 8, min = 8, max = 128, int = true, log = true, pow2 = true }
local CLAUSE_TOLERANCE = { def = 8, min = 8, max = 256, int = true, log = true, pow2 = true }
local CLAUSE_MAXIMUM = { def = 8, min = 8, max = 256, int = true, log = true, pow2 = true }
local TARGET = { def = 4, min = 2, max = 256, int = true, log = true, pow2 = true }
local SPECIFICITY = { def = 1000, min = 1, max = 4000, int = true, log = true }

local SEARCH_PATIENCE = 10
local SEARCH_ROUNDS = 20
local SEARCH_TRIALS = 4
local SEARCH_ITERATIONS = 10
local ITERATIONS = 40

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
    local idx = ann.create({ features = dataset.n_visible, expected_size = train.n })
    local data = ivec.create()
    data:bits_copy(dataset.problems, nil, train.ids, dataset.n_visible)
    idx:add(data:bits_to_cvec(train.n, dataset.n_visible), train.ids)
    local ids, hoods = idx:neighborhoods(KNN, KNN_EPS, KNN_MIN, KNN_MUTUAL)
    train.seed = ds.star_hoods(ids, hoods)
  else
    train.seed = pvec.create()
  end
  if SEED_PAIRS then
    train.seed = train.seed or pvec.create()
    ds.random_pairs(train.ids, SEED_PAIRS, train.seed)
  end
  if SEED_ANCHORS then
    train.seed = train.seed or pvec.create()
    ds.anchor_pairs(train.ids, SEED_ANCHORS, train.seed)
  end
  if SEED_CLASS_ANCHORS then
    local pos, neg = ds.multiclass_pairs(train.ids, train.solutions, SEED_CLASS_ANCHORS, SEED_CLASS_ANCHORS)
    train.seed = train.seed or pvec.create()
    train.seed:copy(pos)
    train.seed:copy(neg)
  end
  collectgarbage("collect")

  print("Building the graph index")
  local graph_index
  if RANKS then

    local virtual_ids = ivec.create(10)
    virtual_ids:fill_indices()
    virtual_ids:add(dataset.ids:size())

    for id in train.ids:each() do
      local class_label = dataset.solutions:get(id)
      train.seed:push(id, dataset.ids:size() + class_label)
    end

    train.problems = ivec.create()
    train.problems:bits_copy(dataset.problems, nil, train.ids, dataset.n_visible)

    for idx, id in train.ids:ieach() do
      local class_label = dataset.solutions:get(id)
      local bit_idx = idx * (dataset.n_visible + 10) + dataset.n_visible + class_label
      train.problems:push(bit_idx)
    end

    local graph_corpus = ivec.create()
    local graph_ids = ivec.create()

    graph_ids:copy(train.ids)
    graph_corpus:copy(train.problems)

    graph_ids:copy(virtual_ids)

    local n_train_samples = train.ids:size()
    local n_features = dataset.n_visible + 10

    for i = 0, 9 do
      local sample_idx = n_train_samples + i
      local feature_idx = dataset.n_visible + i
      local bit_idx = sample_idx * n_features + feature_idx
      graph_corpus:push(bit_idx)
    end

    local graph_ranks = ivec.create(dataset.n_visible + 10)
    graph_ranks:fill(1, 0, dataset.n_visible)
    graph_ranks:fill(0, dataset.n_visible, dataset.n_visible + 10)

    graph_index = inv.create({
      features = dataset.n_visible + 10,
      ranks = graph_ranks,
      decay = DECAY,
      n_ranks = 2
    })
    graph_index:add(graph_corpus, graph_ids)

  else

    -- Fully unsupervised
    local graph_problems = ivec.create()
    graph_problems:bits_copy(dataset.problems, nil, train.ids, dataset.n_visible)
    graph_index = inv.create({ features = dataset.n_visible })
    graph_index:add(graph_problems, train.ids)
    collectgarbage("collect")

  end

  print("Creating graph")
  local stopwatch = utc.stopwatch()
  train.graph = graph.create({
    edges = train.seed,
    index = graph_index,
    bridge = BRIDGE,
    sigma_k = SIGMA_K,
    threads = THREADS,
    each = function (ids, s, b, dt)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Nodes: %-6d  Components: %-6d  Edges: %-6d\n", d, dd, dt, ids, s, b)
    end
  })
  train.adj_ids,
  train.adj_offsets,
  train.adj_neighbors,
  train.adj_weights =
    train.graph:adjacency()
  collectgarbage("collect")

  print("Spectral eigendecomposition")
  train.ids_spectral, train.codes_spectral = spectral.encode({
    type = LAPLACIAN,
    ids = train.adj_ids,
    offsets = train.adj_offsets,
    neighbors = train.adj_neighbors,
    weights = train.adj_weights,
    n_hidden = dataset.n_hidden,
    eps_primme = SPECTRAL_EPS,
    threads = THREADS,
    each = function (t, s, v, k)
      local d, dd = stopwatch()
      if t == "done" then
        str.printf("  Time: %6.2f %6.2f  Spectral Steps: %3d\n", d, dd, s)
      elseif t == "eig" then
        local gap = train.eig_last and v - train.eig_last or 0
        train.eig_last = v
        str.printf("  Time: %6.2f %6.2f  Eig: %3d = %.8f   Gap = %.8f   %s\n", d, dd, s, v, gap, k and "" or "drop")
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
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end
  collectgarbage("collect")

  train.solutions_spectral = ivec.create()
  train.solutions_spectral:copy(train.solutions, train.ids_spectral)
  train.pos_sampled, train.neg_sampled = ds.multiclass_pairs(train.ids_spectral, train.solutions_spectral, SAMPLED_ANCHORS, SAMPLED_ANCHORS)
  test.pos_sampled, test.neg_sampled = ds.multiclass_pairs(test.ids, test.solutions, SAMPLED_ANCHORS, SAMPLED_ANCHORS)
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
      threads = THREADS,
      each = function (bit, gain, score, action)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Bit  %-3d  %-12s | Gain: %2.12f | Score: %.12f\n",
          d, dd, bit, action, gain, score)
      end
    })
    collectgarbage("collect")
  end

  print("AUC stats")
  train.auc_continuous = eval.auc(train.ids_spectral, train.codes_spectral_cont, train.pos_sampled, train.neg_sampled, dataset.n_hidden, nil, THREADS)
  train.auc_binary_full = eval.auc(train.ids_spectral, train.codes_spectral, train.pos_sampled, train.neg_sampled, dataset.n_hidden, nil, THREADS)
  train.codes_spectral:bits_filter(train.kept_bits, dataset.n_hidden)
  dataset.n_hidden_full = dataset.n_hidden
  dataset.n_hidden = train.kept_bits:size()
  dataset.n_latent = dataset.n_hidden * dataset.n_landmarks  -- Recalculate after bit selection
  train.auc_binary = eval.auc(train.ids_spectral, train.codes_spectral, train.pos_sampled, train.neg_sampled, dataset.n_hidden, nil, THREADS)
  str.printf("  AUC (continuous):        %.4f\n", train.auc_continuous)
  str.printf("  AUC (binary, all bits):  %.4f  %d bits\n", train.auc_binary_full, dataset.n_hidden_full)
  str.printf("  AUC (binary, subset):    %.4f  %d bits\n", train.auc_binary, dataset.n_hidden)
  collectgarbage("collect")

  print("Codebook stats")
  train.entropy = eval.entropy_stats(train.codes_spectral, train.ids_spectral:size(), dataset.n_hidden, THREADS)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    train.entropy)
  collectgarbage("collect")

  print("Retrieval stats")
  train.similarity = eval.optimize_retrieval({
    codes = train.codes_spectral,
    n_dims = dataset.n_hidden,
    ids = train.ids_spectral,
    pos = train.pos_sampled,
    neg = train.neg_sampled,
    threads = THREADS,
    each = function (f, p, r, m)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d\n",
        d, dd, f, p, r, m)
    end
  })
  str.printi("Best\n  BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)",
    train.similarity)
  collectgarbage("collect")

  if ENCODER or CLUSTER then
    print("Indexing spectral codes")
    train.idx_spectral = ann.create({ features = dataset.n_hidden, expected_size = train.ids_spectral:size() })
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
      sth_problems:bits_copy(dataset.problems, nil, sth_ids, dataset.n_visible)
      sth_problems = sth_problems:bits_to_cvec(sth_n, dataset.n_visible, true)

      test_ids = test.ids
      test_problems = ivec.create()
      test_problems:bits_copy(dataset.problems, nil, test.ids, dataset.n_visible)
      test_problems = test_problems:bits_to_cvec(test.n, dataset.n_visible, true)

    elseif MODE == "landmarks" then
      print("Setting up landmark features for L-STH")

      sth_ids = train.ids_spectral
      sth_n = sth_ids:size()
      sth_visible = dataset.n_latent

      local sth_raw = ivec.create()
      sth_raw:bits_copy(dataset.problems, nil, train.ids_spectral, dataset.n_visible)
      sth_raw = sth_raw:bits_to_cvec(sth_n, dataset.n_visible)

      local sth_idx = ann.create({ features = dataset.n_visible, expected_size = sth_ids:size() })
      sth_idx:add(sth_raw, sth_ids)

      local sth_hoods
      sth_ids, sth_hoods = sth_idx:neighborhoods(dataset.n_landmarks)

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
      test_vecs:bits_copy(dataset.problems, nil, test.ids, dataset.n_visible)
      test_vecs = test_vecs:bits_to_cvec(test.n, dataset.n_visible)
      local nbr_ids, nbr_hoods = sth_idx:neighborhoods_by_vecs(test_vecs, dataset.n_landmarks)
      for i0, hood in nbr_hoods:ieach() do
        hood:keys(tmp)
        tmp:lookup(nbr_ids)
        train.idx_spectral:get(tmp, test_problems, i0, dataset.n_latent)
      end
      test_problems:bits_flip_interleave(dataset.n_latent)

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
        local predicted = t.predict(sth_problems, sth_n)
        local accuracy = eval.encoding_accuracy(predicted, sth_solutions, sth_n, dataset.n_hidden)
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
        str.printi("    Train (acc) | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)", train.accuracy_predicted)
        if is_final then
          local sth_predicted = t.predict(sth_problems, sth_n)
          train.auc_predicted = eval.auc(sth_ids, sth_predicted, train.pos_sampled, train.neg_sampled, dataset.n_hidden)
          train.similarity_predicted = eval.optimize_retrieval({
            codes = sth_predicted,
            n_dims = dataset.n_hidden,
            ids = sth_ids,
            pos = train.pos_sampled,
            neg = train.neg_sampled,
          })
          local test_predicted = t.predict(test_problems, test.n)
          test.auc_predicted = eval.auc(test_ids, test_predicted, test.pos_sampled, test.neg_sampled, dataset.n_hidden)
          test.similarity_predicted = eval.optimize_retrieval({
            codes = test_predicted,
            n_dims = dataset.n_hidden,
            ids = test_ids,
            pos = test.pos_sampled,
            neg = test.neg_sampled,
          })
          train.similarity.auc = train.auc_binary
          train.similarity_predicted.auc = train.auc_predicted
          test.similarity_predicted.auc = test.auc_predicted
          str.printi("    Codes (sim) | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", train.similarity)
          str.printi("    Train (sim) | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", train.similarity_predicted)
          str.printi("    Test (sim)  | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", test.similarity_predicted)
          print()
        end
      end,
    })
    collectgarbage("collect")

  end

  if CLUSTER then

    print("Clustering (codes)")
    local codes_stats, _, _, codes_nc = eval.optimize_clustering({
      index = train.idx_spectral,
      ids = train.ids_spectral,
      pos = train.pos_sampled,
      neg = train.neg_sampled,
      min_pts = MIN_PTS,
      assign_noise = true,
      each = function (f, p, r, m, c)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d | Clusters: %d\n",
          d, dd, f, p, r, m, c)
      end
    })
    codes_stats.n_clusters = codes_nc
    str.printi("Best\n  BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin) | Clusters: %d#(n_clusters)",
      codes_stats)
    collectgarbage("collect")

    if ENCODER then

      print("Clustering (train)")
      local idx_train = ann.create({ features = dataset.n_hidden, expected_size = train.n })
      idx_train:add(train.encoder.predict(sth_problems, sth_n), sth_ids)
      local train_stats, _, _, train_nc = eval.optimize_clustering({
        index = idx_train,
        ids = train.ids_spectral,
        pos = train.pos_sampled,
        neg = train.neg_sampled,
        each = function (f, p, r, m, c)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d | Clusters: %d\n",
            d, dd, f, p, r, m, c)
        end
      })
      train_stats.n_clusters = train_nc
      str.printi("Best\n  BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin) | Clusters: %d#(n_clusters)",
        train_stats)
      collectgarbage("collect")

      print("Clustering (test)")
      local idx_test = ann.create({ features = dataset.n_hidden, expected_size = test.n })
      idx_test:add(train.encoder.predict(test_problems, test.n), test_ids)
      local test_stats, _, _, test_nc = eval.optimize_clustering({
        index = idx_test,
        ids = test_ids,
        pos = test.pos_sampled,
        neg = test.neg_sampled,
        each = function (f, p, r, m, c)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d | Clusters: %d\n",
            d, dd, f, p, r, m, c)
        end
      })
      test_stats.n_clusters = test_nc
      str.printi("Best\n BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin) | Clusters: %d#(n_clusters)",
        test_stats)
      collectgarbage("collect")

    end
  end

end)
