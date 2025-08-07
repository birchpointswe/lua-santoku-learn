local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local hbi = require("santoku.tsetlin.hbi")
local tm = require("santoku.tsetlin")
local ivec = require("santoku.ivec")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = 10000
local MAX_CLASS = nil
local FEATURES = 784
local THREADS = nil

local BINARIZE = "itq"
local TCH = true
local HIDDEN = 10
local EPS_SPECTRAL = 1e-5
local NORMALIZED = true
local POS_SCALE = 1.0
local NEG_SCALE = -0.1
local POS_SIGMA = -1
local NEG_SIGMA = -1
local MST = false
local BRIDGE = true
local KNN_POS = 0
local KNN_NEG = 0
local KNN_CACHE = 0
local CLUSTER_MIN = 0
local CLUSTER_MAX = 4
local POS_ANCHORS = 6
local NEG_ANCHORS = 2

local SEARCH_PATIENCE = 3
local SEARCH_ROUNDS = 4
local SEARCH_TRIALS = 0
local SEARCH_ITERATIONS = 0
local ITERATIONS = 10
local TOP_K = 1000

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", FEATURES, MAX, MAX_CLASS)

  print("Splitting")
  local train, test = ds.split_binary_mnist(dataset, TTR)
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  test.ids = ivec.create(test.n)
  test.ids:fill_indices()
  dataset.n_features = FEATURES

  print("\nSampling pairs")
  train.pos, train.neg = ds.multiclass_pairs(train.ids, train.solutions, POS_ANCHORS, NEG_ANCHORS)
  test.pos, test.neg = ds.multiclass_pairs(test.ids, test.solutions, POS_ANCHORS, NEG_ANCHORS)
  str.printf("  Positives: %d  %d\n", train.pos:size(), test.pos:size())
  str.printf("  Negatives: %d  %d\n", train.neg:size(), test.neg:size())

  print("\nCreating graph")
  local stopwatch = utc.stopwatch()
  train.graph = graph.create({
    ids = train.ids,
    labels = train.solutions,
    pos = train.pos,
    neg = train.neg,
    mst = MST,
    bridge = BRIDGE,
    knn_pos = KNN_POS,
    knn_neg = KNN_NEG,
    knn_cache = KNN_CACHE,
    pos_scale = POS_SCALE,
    neg_scale = NEG_SCALE,
    pos_sigma = POS_SIGMA,
    neg_sigma = NEG_SIGMA,
    threads = THREADS,
    each = function (s, b, n, dt)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Components: %-6d  Positives: %-6d  Negatives: %-6d\n", d, dd, dt, s, b, n) -- luacheck: ignore
    end
  })
  train.adj_ids,
  train.adj_offsets,
  train.adj_neighbors,
  train.adj_weights =
    train.graph:adjacency()

  print("\nSpectral eigendecomposition")
  train.ids_spectral, train.codes_spectral = spectral.encode({
    ids = train.adj_ids,
    offsets = train.adj_offsets,
    neighbors = train.adj_neighbors,
    weights = train.adj_weights,
    n_hidden = HIDDEN,
    normalized = NORMALIZED,
    eps_primme = EPS_SPECTRAL,
    each = function (t, s, v, k)
      local d, dd = stopwatch()
      if t == "done" then
        str.printf("  Time: %6.2f %6.2f  Spectral Steps: %3d\n", d, dd, s)
      elseif t == "eig" then
        str.printf("  Time: %6.2f %6.2f  Eig: %3d = %.5f  %s\n", d, dd, s, v, k and "" or "drop")
      end
    end
  })

  train.codes_spectral_cont = train.codes_spectral
  if BINARIZE == "itq" then
    print("\nIterative Quantization")
    train.codes_spectral = itq.encode({
      codes = train.codes_spectral,
      n_dims = HIDDEN,
      threads = THREADS,
      each = function (i, a, b)
        str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
      end
    })
  elseif BINARIZE == "median" then
    print("\nMedian thresholding")
    train.codes_spectral = itq.median({
      codes = train.codes_spectral,
      n_dims = HIDDEN,
    })
  elseif BINARIZE == "sign" then
    print("\nSign thresholding")
    train.codes_spectral = itq.sign({
      codes = train.codes_spectral,
      n_dims = HIDDEN,
    })
  end

  if TCH then
    print("\nFlipping bits")
    tch.refine({
      ids = train.adj_ids,
      offsets = train.adj_offsets,
      neighbors = train.adj_neighbors,
      weights = train.adj_weights,
      codes = train.codes_spectral,
      n_dims = HIDDEN,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end

  train.codes_spectral = train.codes_spectral:raw_bitmap(train.ids_spectral:size(), HIDDEN)

  print("\nCodebook stats (general)")
  train.pos, train.neg = train.graph:pairs()
  train.entropy = eval.entropy_stats(train.codes_spectral, train.n, HIDDEN, THREADS)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    train.entropy)
  train.auc_binary = eval.auc(train.ids_spectral, train.codes_spectral, train.pos, train.neg, HIDDEN, nil, THREADS) -- luacheck: ignore
  train.auc_continuous = eval.auc(train.ids_spectral, train.codes_spectral_cont, train.pos, train.neg, HIDDEN, nil, THREADS) -- luacheck: ignore
  str.printi("  AUC (continuous): %.4f#(auc_continuous) | AUC (binary): %.4f#(auc_binary)", train)

  print("\nRetrieval stats (graph)")
  train.similarity = eval.optimize_retrieval({
    codes = train.codes_spectral,
    n_dims = HIDDEN,
    ids = train.ids_spectral,
    pos = train.pos,
    neg = train.neg,
    each = function (f, p, r, m)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m)
    end
  })
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)", -- luacheck: ignore
    train.similarity)

  print("\nCreating index")
  train.index_codes = hbi.create({ features = HIDDEN })
  train.index_codes:add(train.codes_spectral, train.ids_spectral)

  print("\nClustering (graph)\n")
  stopwatch()
  train.cluster_score,
  train.cluster_ids,
  train.cluster_assignments,
  train.n_clusters = eval.optimize_clustering({ -- luacheck: ignore
    index = train.index_codes,
    pos = train.pos,
    neg = train.neg,
    min_margin = CLUSTER_MIN,
    max_margin = CLUSTER_MAX,
    each = function (f, p, r, m, c)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d | Clusters: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m, c)
    end
  })
  train.cluster_score.n_clusters = train.n_clusters
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin) | Clusters: %d#(n_clusters)", -- luacheck: ignore
    train.cluster_score)

  print("\nSelecting features")
  local top_v = train.problems:top_chi2(train.codes_spectral, train.n, dataset.n_features, HIDDEN, TOP_K)

  print("\nPrepping for encoder")
  train.problems:filter(top_v, dataset.n_features)
  train.problems:bits_rearrange(train.ids_spectral, top_v:size())
  train.n = train.ids_spectral:size()
  train.problems:flip_interleave(train.n, top_v:size())
  train.problems = train.problems:raw_bitmap(train.n, top_v:size() * 2)
  test.ids = ivec.create(test.n)
  test.ids:fill_indices()
  test.problems:filter(top_v, dataset.n_features)
  test.problems:flip_interleave(test.n, top_v:size())
  test.problems = test.problems:raw_bitmap(test.n, top_v:size() * 2)
  dataset.n_features = top_v:size()

  print()
  str.printf("Input Features    %d\n", dataset.n_features * 2)
  str.printf("Encoded Features  %d\n", HIDDEN)
  str.printf("Train problems    %d\n", train.n)
  str.printf("Test problems     %d\n", test.n)

  print("\nCreating encoder\n")
  stopwatch = utc.stopwatch()
  --[[local t = ]] tm.optimize_encoder({

    visible = dataset.n_features,
    hidden = HIDDEN,
    sentences = train.problems,
    codes = train.codes_spectral,
    samples = train.n,

    clauses = { def = 1024, min = 512, max = 4096, log = true, int = true },
    target = { def = 0.1, min = 0.05, max = 0.25 },
    specificity = { def = 10, min = 2, max = 20 },

    search_patience = SEARCH_PATIENCE,
    search_rounds = SEARCH_ROUNDS,
    search_trials = SEARCH_TRIALS,
    search_iterations = SEARCH_ITERATIONS,
    final_iterations = ITERATIONS,

    search_metric = function (t)
      local predicted = t.predict(train.problems, train.n)
      local accuracy = eval.encoding_accuracy(predicted, train.codes_spectral, train.n, HIDDEN)
      return accuracy.mean_hamming, accuracy
    end,

    each = function (t, is_final, train_accuracy, params, epoch, round, trial)
      local d, dd = stopwatch()
      if is_final then
        str.printf("  Time %3.2f %3.2f  Finalizing  C=%d T=%.2f S=%.2f  Epoch  %d\n",
          d, dd, params.clauses, params.target, params.specificity, epoch)
        print()
      else
        str.printf("  Time %3.2f %3.2f  Exploring  C=%d T=%.2f S=%.2f  R=%d T=%d  Epoch  %d\n",
          d, dd, params.clauses, params.target, params.specificity, round, trial, epoch)
        print()
      end
      train.accuracy_predicted = train_accuracy
      str.printi("    Train (acc) | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)\n", train.accuracy_predicted) -- luacheck: ignore
      if is_final then
        train.codes_predicted = t.predict(train.problems, train.n)
        train.auc_predicted = eval.auc(train.ids, train.codes_predicted, train.pos, train.neg, HIDDEN)
        train.similarity_predicted = eval.optimize_retrieval({
          codes = train.codes_predicted,
          n_dims = HIDDEN,
          ids = train.ids_spectral,
          pos = train.pos,
          neg = train.neg,
        })
        test.codes_predicted = t.predict(test.problems, test.n)
        test.auc_predicted = eval.auc(test.ids, test.codes_predicted, test.pos, test.neg, HIDDEN)
        test.similarity_predicted = eval.optimize_retrieval({
          codes = test.codes_predicted,
          n_dims = HIDDEN,
          ids = test.ids,
          pos = test.pos,
          neg = test.neg,
        })
        train.similarity.auc = train.auc_binary
        train.similarity_predicted.auc = train.auc_predicted
        test.similarity_predicted.auc = test.auc_predicted
        str.printi("    Codes (sim) | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", train.similarity) -- luacheck: ignore
        str.printi("    Train (sim) | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", train.similarity_predicted) -- luacheck: ignore
        str.printi("    Test (sim)  | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", test.similarity_predicted) -- luacheck: ignore
        print()
      end
    end,
  })

end)
