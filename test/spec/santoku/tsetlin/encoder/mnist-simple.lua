require("santoku.dvec")
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
    hidden = 32,
  },
  mode = {
    binarize = "median",
    tch = true,
  },
  spectral = {
    laplacian = "unnormalized",
    eps = 1e-12,
  },
  clustering = {
    linkage = "simhash",
    knn = 32,
    knn_min = nil,
    knn_mutual = false,
    min_pts = 32,
  },
  eval = {
    anchors = 8,
    retrieval_metric = "biserial",
    cluster_metric = "biserial",
    tolerance = 1e-3,
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

  print("\nCreating categorical graph")
  local stopwatch = utc.stopwatch()
  do
    local cat_index = ds.classes_index(dataset.ids, dataset.solutions)
    dataset.graph_adj_ids,
    dataset.graph_adj_offsets,
    dataset.graph_adj_neighbors,
    dataset.graph_adj_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.eval.anchors,
        category_negatives = cfg.eval.anchors,
        threads = cfg.threads,
        each = function (ids, s, b, dt)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Nodes: %-6d  Components: %-6d  Edges: %-6d\n", d, dd, dt, ids, s, b)
        end
      })
    cat_index:destroy()
  end

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

  print("\nMedian thresholding")
  dataset.codes_spectral_cont = dataset.codes_spectral
  dataset.codes_spectral = itq.median({
    codes = dataset.codes_spectral,
    n_dims = cfg.data.hidden,
  })

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
  dataset.entropy = eval.entropy_stats(dataset.codes_spectral, dataset.n, cfg.data.hidden, cfg.threads)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    dataset.entropy)

  print("\nCreating spectral codes index")
  dataset.index_codes = hbi.create({ features = cfg.data.hidden })
  dataset.index_codes:add(dataset.codes_spectral, dataset.ids_spectral)
  collectgarbage("collect")

  print("\nCreating categorical adjacency for evaluation")
  do
    local cat_index = ds.classes_index(dataset.ids, dataset.solutions)
    dataset.adj_sampled_ids,
    dataset.adj_sampled_offsets,
    dataset.adj_sampled_neighbors,
    dataset.adj_sampled_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.eval.anchors,
        category_negatives = cfg.eval.anchors,
        threads = cfg.threads
      })
    cat_index:destroy()
  end
  collectgarbage("collect")

  print("\nRetrieval stats (categorical adjacency)")
  dataset.retrieval_scores = eval.optimize_retrieval({
    index = dataset.index_codes,
    ids = dataset.adj_sampled_ids,
    offsets = dataset.adj_sampled_offsets,
    neighbors = dataset.adj_sampled_neighbors,
    weights = dataset.adj_sampled_weights,
    metric = cfg.eval.retrieval_metric,
    threads = cfg.threads,
    each = function (acc)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | Margin: %d | Score: %+.6f\n",
        d, dd, acc.margin, acc.score)
    end
  })
  local best_score, best_idx = dataset.retrieval_scores:max()
  str.printf("Best\n  Margin: %d | Score: %+.6f\n", best_idx, best_score)
  collectgarbage("collect")

  print("\nClustering (codes) (graph edges)\n")
  stopwatch()
  local cluster_stats = eval.optimize_clustering({
    index = dataset.index_codes,
    ids = dataset.graph_adj_ids,
    offsets = dataset.graph_adj_offsets,
    neighbors = dataset.graph_adj_neighbors,
    weights = dataset.graph_adj_weights,
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
  if cluster_stats.scores then
    local best_score, best_step = cluster_stats.scores:scores_plateau(cfg.eval.tolerance)
    local best_n_clusters = cluster_stats.n_clusters:get(best_step)
    str.printf("Best\n  Step: %2d | Score: %+.6f | Clusters: %d\n", best_step, best_score, best_n_clusters)
  end

end)
