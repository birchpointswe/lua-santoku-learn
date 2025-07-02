local ds = require("santoku.tsetlin.dataset")
local varg = require("santoku.varg")
local err = require("santoku.error")
local eval = require("santoku.tsetlin.evaluator")
local it = require("santoku.iter")
local ann = require("santoku.tsetlin.ann")
local hbi = require("santoku.tsetlin.hbi")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local booleanizer = require("santoku.tsetlin.booleanizer")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local serialize = require("santoku.serialize") -- luacheck: ignore
local tbl = require("santoku.table")
local str = require("santoku.string")
local arr = require("santoku.array")
local test = require("santoku.test")
local utc = require("santoku.utc")
local fs = require("santoku.fs")
local sys = require("santoku.system")

local MAX = nil
local MAX_CLASS = nil
local FEATURES = 784

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

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", FEATURES, MAX, MAX_CLASS)

  print("Splitting")
  dataset = tbl.assign(ds.split_binary_mnist(dataset, 1), dataset)
  dataset.n_features = FEATURES

  print("\nIndexing raw features")
  dataset.index_raw = ann.create({ expected_size = dataset.n, features = dataset.n_features })
  dataset.index_raw:add(dataset.problems:raw_bitmap(dataset.n, dataset.n_features), 0, dataset.n)

  str.printf("  Classes\t%d\n", 10)
  str.printf("  Samples\t%d\n", dataset.n)

  print("\nCreating graph")
  local stopwatch = utc.stopwatch()
  local spos, sneg = ds.multiclass_pairs(dataset.solutions, 2, 1)
  dataset.graph = graph.create({
    labels = dataset.solutions,
    pos = spos,
    neg = sneg,
    index = dataset.index_raw,
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

  -- Generate evaluation pairs
  dataset.pos_graph, dataset.neg_graph = dataset.graph:pairs()
  print("\nEnriched graph:\n")
  local ids = {}
  local cnt = {}
  local npos = {}
  local nneg = {}
  for l, a, b in it.chain(
    it.paste(true, dataset.pos_graph:each()),
    it.paste(false, dataset.neg_graph:each()))
  do
    ids[a] = true
    ids[b] = true
    local sa = dataset.solutions:get(a)
    local sb = dataset.solutions:get(b)
    if sa > sb then
      sa, sb = sb, sa
    end
    tbl.update(cnt, sa, sb, l, function (x)
      if l then
        npos[a] = (npos[a] or 0) + 1
        npos[b] = (npos[b] or 0) + 1
      else
        nneg[a] = (nneg[a] or 0) + 1
        nneg[b] = (nneg[b] or 0) + 1
      end
      return (x or 0) + 1
    end)
  end
  local ids = it.collect(it.take(100, it.keys(ids)))
  arr.sort(ids)
  for i in it.ivals(ids) do
    str.printf("  node:%-4d  pos:%-4d  neg:%-4d  total:%-4d\n", i, npos[i] or 0, nneg[i] or 0, (npos[i] or 0) + (nneg[i] or 0))
  end
  str.printf("  ")
  for i = 0, 9 do
    str.printf("         %-6d", i)
  end
  str.printf("\n")
  for i = 0, 9 do
    str.printf("  %1d", i)
    for j = 0, 9 do
      str.printf("  %6d %-6d", (tbl.get(cnt, i, j, true) or 0), (tbl.get(cnt, i, j, false) or 0))
    end
    str.printf("\n")
  end

  print("\nSpectral eigendecomposition")
  dataset.ids_spectral, dataset.codes_spectral, dataset.scale_spectral, dataset.dims_spectral, dataset.neg_scale = spectral.encode({
    graph = dataset.graph,
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

  dataset.codes_spectral_cont = dataset.codes_spectral
  if BINARIZE == "itq" then
    print("\nIterative Quantization")
    dataset.codes_spectral = itq.encode({
      codes = dataset.codes_spectral,
      n_dims = dataset.dims_spectral,
      each = function (i, a, b)
        str.printf("  ITQ completed in %s itrs. Objective %f → %f\n", i, a, b)
      end
    })
  elseif BINARIZE == "median" then
    print("\nSign thresholding")
    dataset.codes_spectral = itq.median({
      codes = dataset.codes_spectral,
      n_dims = dataset.dims_spectral,
    })
  elseif BINARIZE == "sign" then
    print("\nSign thresholding")
    dataset.codes_spectral = itq.sign({
      codes = dataset.codes_spectral,
      n_dims = dataset.dims_spectral,
    })
  end

  if TCH then
    print("\nFlipping bits")
    tch.refine({
      codes = dataset.codes_spectral,
      graph = dataset.graph,
      scale = dataset.scale_spectral,
      negatives = dataset.neg_scale,
      n_dims = dataset.dims_spectral,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end

  dataset.pos_sampled, dataset.neg_sampled = ds.multiclass_pairs(dataset.solutions)
  dataset.codes_spectral = dataset.codes_spectral:raw_bitmap(dataset.ids_spectral:size(), dataset.dims_spectral)

  print("\nCodebook stats (general)")
  dataset.entropy = eval.entropy_stats(dataset.codes_spectral, dataset.n, dataset.dims_spectral)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    dataset.entropy)
  str.printi("  AUC (continuous): %.4f#(auc_continuous) | AUC (binary): %.4f#(auc_binary)",
    { auc_binary = eval.auc(dataset.codes_spectral, dataset.pos_graph, dataset.neg_graph, dataset.dims_spectral)
    , auc_continuous = eval.auc(dataset.codes_spectral_cont, dataset.pos_graph, dataset.neg_graph, dataset.dims_spectral)
    })

  print("\nRetrieval stats (graph)")
  dataset.similarity_graph = eval.optimize_retrieval({
    codes = dataset.codes_spectral,
    n_dims = dataset.dims_spectral,
    ids = dataset.ids_spectral,
    pos = dataset.pos_graph,
    neg = dataset.neg_graph,
    each = function (f, p, r, m)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m)
    end
  })
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)", -- luacheck: ignore
    dataset.similarity_graph)

  print("\nRetrieval stats (sampled)")
  dataset.similarity_sampled = eval.optimize_retrieval({
    codes = dataset.codes_spectral,
    n_dims = dataset.dims_spectral,
    ids = dataset.ids_spectral,
    pos = dataset.pos_sampled,
    neg = dataset.neg_sampled,
    each = function (a, f, p, r, m)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d\n", -- luacheck: ignore
        d, dd, a, f, p, r, m)
    end
  })
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin)", -- luacheck: ignore
    dataset.similarity_sampled)

  print("\nCreating index")
  dataset.index_codes = hbi.create({ features = dataset.dims_spectral })
  dataset.index_codes:add(dataset.codes_spectral, dataset.ids_spectral)
  -- dataset.index_codes = ann.create({ features = dataset.dims_spectral, expected_size = dataset.ids_spectral:size() })
  -- dataset.index_codes:add(dataset.codes_spectral, dataset.ids_spectral)

  print("\nClustering (graph)\n")
  stopwatch()
  dataset.cluster_score_graph,
  dataset.cluster_ids_graph,
  dataset.cluster_assignments_graph,
  dataset.n_clusters_graph = eval.optimize_clustering({ -- luacheck: ignore
    index = dataset.index_codes,
    pos = dataset.pos_graph,
    neg = dataset.neg_graph,
    min_margin = 0,
    max_margin = 3,
    each = function (f, p, r, m, c)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | BACC: %.2f | TPR: %.2f | TNR: %.2f | Margin: %d | Clusters: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m, c)
    end
  })
  dataset.cluster_score_graph.n_clusters = dataset.n_clusters_graph
  str.printi("\n  Best | BACC: %.2f#(bacc) | TPR: %.2f#(tpr) | TNR: %.2f#(tnr) | Margin: %d#(margin) | Clusters: %d#(n_clusters)", -- luacheck: ignore
    dataset.cluster_score_graph)

  print("\nClustering (sampled)\n")
  stopwatch()
  dataset.cluster_score_sampled,
  dataset.cluster_ids_sampled,
  dataset.cluster_assignments_sampled,
  dataset.n_clusters_sampled = eval.optimize_clustering({ -- luacheck: ignore
    index = dataset.index_codes,
    pos = dataset.pos_sampled,
    neg = dataset.neg_sampled,
    min_margin = 0,
    max_margin = 3,
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
