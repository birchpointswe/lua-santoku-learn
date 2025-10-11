local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local hbi = require("santoku.tsetlin.hbi")
local ivec = require("santoku.ivec")
local pvec = require("santoku.pvec")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local cfg = {
  data = {
    max = nil,
    max_class = nil,
    features = 784,
    hidden = 8,
  },
  mode = {
    binarize = "itq",
    tch = true,
  },
  spectral = {
    laplacian = "unnormalized",
    eps = 1e-3,
  },
  itq = {
    eps = 1e-8,
    iterations = 200,
  },
  clustering = {
    min_margin = 0,
    max_margin = 4,
  },
  eval = {
    anchors = 8,
  },
  threads = nil,
}

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.features, cfg.data.max, cfg.data.max_class)
  dataset = ds.split_binary_mnist(dataset, 1.0)
  dataset.n_features = cfg.data.features
  dataset.ids = ivec.create(dataset.n)
  dataset.ids:fill_indices()

  print("\nSampling seed pairs")
  local pos_seed, neg_seed = ds.multiclass_pairs(dataset.ids, dataset.solutions, cfg.eval.anchors, cfg.eval.anchors)
  local seed = pvec.create()
  seed:copy(pos_seed)
  seed:copy(neg_seed)
  local index = ds.classes_index(dataset.ids, dataset.solutions)
  str.printf("  Edges: %d\n", seed:size())

  print("\nCreating graph")
  local stopwatch = utc.stopwatch()
  dataset.graph_adj_ids,
  dataset.graph_adj_offsets,
  dataset.graph_adj_neighbors,
  dataset.graph_adj_weights =
    graph.adjacency({
      edges = seed,
      index = index,
      threads = cfg.threads,
      each = function (ids, s, b, dt)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Nodes: %-6d  Components: %-6d  Edges: %-6d\n", d, dd, dt, ids, s, b)
      end
    })

  print("\nSpectral eigendecomposition")
  dataset.ids_spectral, dataset.codes_spectral = spectral.encode({
    type = cfg.spectral.laplacian,
    eps = cfg.spectral.eps,
    ids = dataset.graph_adj_ids,
    offsets = dataset.graph_adj_offsets,
    neighbors = dataset.graph_adj_neighbors,
    weights = dataset.graph_adj_weights,
    n_hidden = cfg.data.hidden,
    threads = cfg.threads,
    each = function (t, s, v, k)
      local d, dd = stopwatch()
      if t == "done" then
        str.printf("  Time: %6.2f %6.2f  Stage: %s  matvecs = %d\n", d, dd, t, s)
      elseif t == "eig" then
        local gap = dataset.eig_last and v - dataset.eig_last or 0
        dataset.eig_last = v
        str.printf("  Time: %6.2f %6.2f  Stage: %s  %3d = %.8f   Gap = %.8f   %s\n", d, dd, t, s, v, gap, k and "" or "drop")
      else
        str.printf("  Time: %6.2f %6.2f  Stage: %s\n", d, dd, t)
      end
    end
  })

  dataset.codes_spectral_cont = dataset.codes_spectral
  if cfg.mode.binarize == "itq" then
    print("\nIterative Quantization")
    dataset.codes_spectral = itq.encode({
      codes = dataset.codes_spectral,
      n_dims = cfg.data.hidden,
      tolerance = cfg.itq.eps,
      iterations = cfg.itq.iterations,
      threads = cfg.threads,
      each = function (i, a, b)
        str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
      end
    })
  elseif cfg.mode.binarize == "median" then
    print("\nMedian thresholding")
    dataset.codes_spectral = itq.median({
      codes = dataset.codes_spectral,
      n_dims = cfg.data.hidden,
    })
  elseif cfg.mode.binarize == "sign" then
    print("\nSign thresholding")
    dataset.codes_spectral = itq.sign({
      codes = dataset.codes_spectral,
      n_dims = cfg.data.hidden,
    })
  end

  if cfg.mode.tch then
    print("\nFlipping bits")
    tch.refine({
      ids = dataset.graph_adj_ids,
      offsets = dataset.graph_adj_offsets,
      neighbors = dataset.graph_adj_neighbors,
      weights = dataset.graph_adj_weights,
      codes = dataset.codes_spectral,
      n_dims = cfg.data.hidden,
      threads = cfg.threads,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end

  print("\nCodebook stats")
  dataset.pos_sampled, dataset.neg_sampled = ds.multiclass_pairs(dataset.ids, dataset.solutions, cfg.eval.anchors, cfg.eval.anchors)
  dataset.entropy = eval.entropy_stats(dataset.codes_spectral, dataset.n, cfg.data.hidden, cfg.threads)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    dataset.entropy)
  dataset.auc_binary = eval.auc(dataset.ids_spectral, dataset.codes_spectral, dataset.pos_sampled, dataset.neg_sampled, cfg.data.hidden, nil, cfg.threads)
  dataset.auc_continuous = eval.auc(dataset.ids_spectral, dataset.codes_spectral_cont, dataset.pos_sampled, dataset.neg_sampled, cfg.data.hidden, nil, cfg.threads)
  str.printi("  AUC (continuous): %.4f#(auc_continuous) | AUC (binary): %.4f#(auc_binary)", dataset)

  print("\nRetrieval stats (sampled)")
  dataset.similarity_sampled = eval.optimize_retrieval({
    codes = dataset.codes_spectral,
    n_dims = cfg.data.hidden,
    ids = dataset.ids_spectral,
    pos = dataset.pos_sampled,
    neg = dataset.neg_sampled,
    threads = cfg.threads,
    each = function (f, p, r, m)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m)
    end
  })
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)", -- luacheck: ignore
    dataset.similarity_sampled)

  print("\nCreating index")
  dataset.index_codes = hbi.create({ features = cfg.data.hidden })
  dataset.index_codes:add(dataset.codes_spectral, dataset.ids_spectral)

  print("\nClustering (sampled)\n")
  stopwatch()
  dataset.cluster_score_sampled,
  dataset.cluster_ids_sampled,
  dataset.cluster_assignments_sampled,
  dataset.n_clusters_sampled = eval.optimize_clustering({
    index = dataset.index_codes,
    ids = dataset.ids_spectral,
    pos = dataset.pos_sampled,
    neg = dataset.neg_sampled,
    min_margin = cfg.clustering.min_margin,
    max_margin = cfg.clustering.max_margin,
    threads = cfg.threads,
    each = function (f, p, r, m, c)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d | Clusters: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m, c)
    end
  })
  dataset.cluster_score_sampled.n_clusters = dataset.n_clusters_sampled
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin) | Clusters: %d#(n_clusters)", -- luacheck: ignore
    dataset.cluster_score_sampled)

end)
