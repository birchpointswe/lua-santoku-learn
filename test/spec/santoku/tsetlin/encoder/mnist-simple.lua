local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
-- local ann = require("santoku.tsetlin.ann")
-- local inv = require("santoku.tsetlin.inv")
local hbi = require("santoku.tsetlin.hbi")
local ivec = require("santoku.ivec")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

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

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", FEATURES, MAX, MAX_CLASS)
  dataset = ds.split_binary_mnist(dataset, 1.0)
  dataset.n_features = FEATURES
  dataset.ids = ivec.create(dataset.n)
  dataset.ids:fill_indices()

  print("\nSampling pairs")
  local spos, sneg = ds.multiclass_pairs(dataset.ids, dataset.solutions, POS_ANCHORS, NEG_ANCHORS)
  str.printf("  Positives: %d\n", spos:size())
  str.printf("  Negatives: %d\n", sneg:size())

  print("\nCreating graph")
  local stopwatch = utc.stopwatch()
  dataset.graph = graph.create({
    ids = dataset.ids,
    labels = dataset.solutions,
    pos = spos,
    neg = sneg,
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

  print("\nSpectral eigendecomposition")
  dataset.ids_spectral, dataset.codes_spectral = spectral.encode({
    graph = dataset.graph,
    n_hidden = HIDDEN,
    normalized = NORMALIZED,
    eps_primme = EPS_SPECTRAL,
    threads = THREADS,
    each = function (t, s, v, k)
      local d, dd = stopwatch()
      if t == "done" then
        str.printf("  Time: %6.2f %6.2f  Spectral Steps: %3d\n", d, dd, s)
      elseif t == "eig" then
        str.printf("  Time: %6.2f %6.2f  Eig: %3d = %.8f  %s\n", d, dd, s, v, k and "" or "drop")
      end
    end
  })

  dataset.codes_spectral_cont = dataset.codes_spectral
  if BINARIZE == "itq" then
    print("\nIterative Quantization")
    dataset.codes_spectral = itq.encode({
      codes = dataset.codes_spectral,
      n_dims = HIDDEN,
      threads = THREADS,
      each = function (i, a, b)
        str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
      end
    })
  elseif BINARIZE == "median" then
    print("\nMedian thresholding")
    dataset.codes_spectral = itq.median({
      codes = dataset.codes_spectral,
      n_dims = HIDDEN,
    })
  elseif BINARIZE == "sign" then
    print("\nSign thresholding")
    dataset.codes_spectral = itq.sign({
      codes = dataset.codes_spectral,
      n_dims = HIDDEN,
    })
  end

  if TCH then
    print("\nFlipping bits")
    tch.refine({
      codes = dataset.codes_spectral,
      graph = dataset.graph,
      scale = dataset.scale_spectral,
      pos_scale = POS_SCALE,
      neg_scale = NEG_SCALE,
      n_dims = HIDDEN,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end

  dataset.codes_spectral = dataset.codes_spectral:raw_bitmap(dataset.ids_spectral:size(), HIDDEN)

  print("\nCodebook stats (general)")
  dataset.pos_graph, dataset.neg_graph = dataset.graph:pairs()
  dataset.entropy = eval.entropy_stats(dataset.codes_spectral, dataset.n, HIDDEN, THREADS)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    dataset.entropy)
  dataset.auc_binary = eval.auc(dataset.ids_spectral, dataset.codes_spectral, dataset.pos_graph, dataset.neg_graph, HIDDEN, nil, THREADS) -- luacheck: ignore
  dataset.auc_continuous = eval.auc(dataset.ids_spectral, dataset.codes_spectral_cont, dataset.pos_graph, dataset.neg_graph, HIDDEN, nil, THREADS) -- luacheck: ignore
  str.printi("  AUC (continuous): %.4f#(auc_continuous) | AUC (binary): %.4f#(auc_binary)", dataset)

  print("\nRetrieval stats (graph)")
  dataset.similarity_graph = eval.optimize_retrieval({
    codes = dataset.codes_spectral,
    n_dims = HIDDEN,
    ids = dataset.ids_spectral,
    pos = dataset.pos_graph,
    neg = dataset.neg_graph,
    threads = THREADS,
    each = function (f, p, r, m)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m)
    end
  })
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)", -- luacheck: ignore
    dataset.similarity_graph)

  print("\nCreating index")
  dataset.index_codes = hbi.create({ features = HIDDEN, threads = THREADS })
  dataset.index_codes:add(dataset.codes_spectral, dataset.ids_spectral)

  print("\nClustering (graph)\n")
  stopwatch()
  dataset.cluster_score_graph,
  dataset.cluster_ids_graph,
  dataset.cluster_assignments_graph,
  dataset.n_clusters_graph = eval.optimize_clustering({ -- luacheck: ignore
    index = dataset.index_codes,
    pos = dataset.pos_graph,
    neg = dataset.neg_graph,
    min_margin = CLUSTER_MIN,
    max_margin = CLUSTER_MAX,
    threads = THREADS,
    each = function (f, p, r, m, c)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d | Clusters: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m, c)
    end
  })
  dataset.cluster_score_graph.n_clusters = dataset.n_clusters_graph
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin) | Clusters: %d#(n_clusters)", -- luacheck: ignore
    dataset.cluster_score_graph)

end)
