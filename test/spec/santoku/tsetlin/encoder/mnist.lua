local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local ann = require("santoku.tsetlin.ann")
local tm = require("santoku.tsetlin")
local ivec = require("santoku.ivec")
local hbi = require("santoku.tsetlin.hbi")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = nil
local MAX_CLASS = 1000
local FEATURES = 784
local SAMPLED = false

local BINARIZE = "itq"
local TCH = true
local HIDDEN = 32
local FIXED = -1
local NORMALIZED = false
local NEGATIVES = -1.0
local MST = false
local BRIDGE = true
local KNN_POS = 0
local KNN_NEG = 0
local KNN_CACHE = 0
local CLUSTER_MIN = 0
local CLUSTER_MAX = 1
local POS_ANCHORS = 2
local NEG_ANCHORS = 1

local TM_ITERS = 100
-- local CLAUSES = 1024
-- local TARGET = 0.1
-- local SPECIFICITY = 10
local TOP_K = 1000

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", FEATURES, MAX, MAX_CLASS)

  print("Splitting")
  local train, test = ds.split_binary_mnist(dataset, TTR)
  dataset.n_features = FEATURES

  print("\nIndexing raw features")
  train.index_raw = ann.create({ expected_size = train.n, features = dataset.n_features })
  train.index_raw:add(train.problems:raw_bitmap(train.n, dataset.n_features), 0, train.n)

  str.printf("  Classes\t%d\n", 10)
  str.printf("  Samples\t%d\n", train.n)

  print("\nCreating graph")
  local stopwatch = utc.stopwatch()
  local spos, sneg = ds.multiclass_pairs(train.solutions, POS_ANCHORS, NEG_ANCHORS)
  train.graph = graph.create({
    labels = train.solutions,
    pos = spos,
    neg = sneg,
    index = train.index_raw,
    mst = MST,
    bridge = BRIDGE,
    knn_pos = KNN_POS,
    knn_neg = KNN_NEG,
    knn_cache = KNN_CACHE,
    each = function (s, b, n, dt)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Components: %-6d  Positives: %-6d  Negatives: %-6d\n", d, dd, dt, s, b, n) -- luacheck: ignore
    end
  })

  print("\nSpectral eigendecomposition")
  train.ids_spectral,
  train.codes_spectral,
  train.scale_spectral,
  train.dims_spectral,
  train.neg_scale = spectral.encode({
    graph = train.graph,
    n_hidden = HIDDEN,
    n_fixed = FIXED,
    negatives = NEGATIVES,
    normalized = NORMALIZED,
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
      n_dims = train.dims_spectral,
      each = function (i, a, b)
        str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
      end
    })
  elseif BINARIZE == "median" then
    print("\nMedian thresholding")
    train.codes_spectral = itq.median({
      codes = train.codes_spectral,
      n_dims = train.dims_spectral,
    })
  elseif BINARIZE == "sign" then
    print("\nSign thresholding")
    train.codes_spectral = itq.sign({
      codes = train.codes_spectral,
      n_dims = train.dims_spectral,
    })
  end

  if TCH then
    print("\nFlipping bits")
    tch.refine({
      codes = train.codes_spectral,
      graph = train.graph,
      scale = train.scale_spectral,
      negatives = train.neg_scale,
      n_dims = train.dims_spectral,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end

  train.codes_spectral = train.codes_spectral:raw_bitmap(train.ids_spectral:size(), train.dims_spectral)

  print("\nCodebook stats (general)")
  train.pos_graph, train.neg_graph = train.graph:pairs()
  train.entropy = eval.entropy_stats(train.codes_spectral, train.n, train.dims_spectral)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    train.entropy)
  train.auc_binary = eval.auc(train.codes_spectral, train.pos_graph, train.neg_graph, train.dims_spectral) -- luacheck: ignore
  train.auc_continuous = eval.auc(train.codes_spectral_cont, train.pos_graph, train.neg_graph, train.dims_spectral) -- luacheck: ignore
  str.printi("  AUC (continuous): %.4f#(auc_continuous) | AUC (binary): %.4f#(auc_binary)", train)

  print("\nRetrieval stats (graph)")
  train.similarity_graph = eval.optimize_retrieval({
    codes = train.codes_spectral,
    n_dims = train.dims_spectral,
    ids = train.ids_spectral,
    pos = train.pos_graph,
    neg = train.neg_graph,
    each = function (f, p, r, m)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m)
    end
  })
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)", -- luacheck: ignore
    train.similarity_graph)

  if SAMPLED then
    print("\nRetrieval stats (sampled)")
    train.pos_sampled, train.neg_sampled = ds.multiclass_pairs(train.solutions)
    train.similarity_sampled = eval.optimize_retrieval({
      codes = train.codes_spectral,
      n_dims = train.dims_spectral,
      ids = train.ids_spectral,
      pos = train.pos_sampled,
      neg = train.neg_sampled,
      each = function (a, f, p, r, m)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d\n", -- luacheck: ignore
          d, dd, a, f, p, r, m)
      end
    })
    str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)", -- luacheck: ignore
      train.similarity_sampled)
  end

  print("\nCreating index")
  train.index_codes = hbi.create({ features = train.dims_spectral })
  train.index_codes:add(train.codes_spectral, train.ids_spectral)
  -- train.index_codes = ann.create({ features = train.dims_spectral, expected_size = train.ids_spectral:size() })
  -- train.index_codes:add(train.codes_spectral, train.ids_spectral)

  print("\nClustering (graph)\n")
  stopwatch()
  train.cluster_score_graph,
  train.cluster_ids_graph,
  train.cluster_assignments_graph,
  train.n_clusters_graph = eval.optimize_clustering({ -- luacheck: ignore
    index = train.index_codes,
    pos = train.pos_graph,
    neg = train.neg_graph,
    min_margin = CLUSTER_MIN,
    max_margin = CLUSTER_MAX,
    each = function (f, p, r, m, c)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d | Clusters: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m, c)
    end
  })
  train.cluster_score_graph.n_clusters = train.n_clusters_graph
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin) | Clusters: %d#(n_clusters)", -- luacheck: ignore
    train.cluster_score_graph)

  if SAMPLED then
    print("\nClustering (sampled)\n")
    stopwatch()
    train.cluster_score_sampled,
    train.cluster_ids_sampled,
    train.cluster_assignments_sampled,
    train.n_clusters_sampled = eval.optimize_clustering({ -- luacheck: ignore
      index = train.index_codes,
      pos = train.pos_sampled,
      neg = train.neg_sampled,
      min_margin = CLUSTER_MIN,
      max_margin = CLUSTER_MAX,
      each = function (f, p, r, m, c)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d | Clusters: %d\n", -- luacheck: ignore
          d, dd, f, p, r, m, c)
      end
    })
    train.cluster_score_sampled.n_clusters = train.n_clusters_sampled
    str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin) | Clusters: %d#(n_clusters)", -- luacheck: ignore
      train.cluster_score_sampled)
  end

  print("\nSelecting features")
  local top_v = train.problems:top_chi2(train.codes_spectral, train.n, dataset.n_features, train.dims_spectral, TOP_K)

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
  test.pos_sampled, test.neg_sampled = ds.multiclass_pairs(test.solutions, POS_ANCHORS, NEG_ANCHORS)
  dataset.n_features = top_v:size()

  print()
  str.printf("Input Features    %d\n", dataset.n_features * 2)
  str.printf("Encoded Features  %d\n", train.dims_spectral)
  str.printf("Train problems    %d\n", train.n)
  str.printf("Test problems     %d\n", test.n)

  print("\nCreating encoder\n")
  stopwatch = utc.stopwatch()
  --[[local t = ]] tm.optimize_encoder({

    visible = dataset.n_features,
    hidden = train.dims_spectral,
    sentences = train.problems,
    codes = train.codes_spectral,
    samples = train.n,

    clauses = 128, --{ def = 1024, min = 512, max = 2048, log = true, int = true },
    target = { def = 0.15, min = 0.05, max = 0.25 },
    specificity = { def = 5.18, min = 2, max = 20 },

    search_patience = 1,
    search_rounds = 4,
    search_trials = 10,
    search_iterations = 10,
    final_iterations = TM_ITERS,
    search_metric = function (t)
      local predicted = t.predict(train.problems, train.n)
      local accuracy = eval.encoding_accuracy(predicted, train.codes_spectral, train.n, train.dims_spectral)
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
        train.auc_predicted = eval.auc(train.codes_predicted, train.pos_graph, train.neg_graph, train.dims_spectral)
        train.similarity_predicted = eval.optimize_retrieval({
          codes = train.codes_predicted,
          n_dims = train.dims_spectral,
          ids = train.ids_spectral,
          pos = train.pos_graph,
          neg = train.neg_graph,
        })
        test.codes_predicted = t.predict(test.problems, test.n)
        test.auc_predicted = eval.auc(test.codes_predicted, test.pos_sampled, test.neg_sampled, train.dims_spectral)
        test.similarity_predicted = eval.optimize_retrieval({
          codes = test.codes_predicted,
          n_dims = train.dims_spectral,
          ids = test.ids,
          pos = test.pos_sampled,
          neg = test.neg_sampled,
        })
        train.similarity_graph.auc = train.auc_binary
        train.similarity_predicted.auc = train.auc_predicted
        test.similarity_predicted.auc = test.auc_predicted
        str.printi("    Codes (sim) | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", train.similarity_graph) -- luacheck: ignore
        str.printi("    Train (sim) | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", train.similarity_predicted) -- luacheck: ignore
        str.printi("    Test (sim)  | AUC: %.2f#(auc) | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %.2f#(margin)", test.similarity_predicted) -- luacheck: ignore
        print()
      end
    end,
  })

end)
