local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local ann = require("santoku.tsetlin.ann")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local booleanizer = require("santoku.tsetlin.booleanizer")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local serialize = require("santoku.serialize") -- luacheck: ignore
local tbl = require("santoku.table")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local MAX = 40000
local FEATURES = 784

local ITQ = true
local TCH = true
local HIDDEN = 8
local KNN = 0
local STAR_CENTERS = 1
local STAR_NEGATIVES = 1
local TRANS_HOPS = 0
local TRANS_POS = 0
local TRANS_NEG = 0
local DBSCAN_MIN = 0

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", FEATURES, MAX)

  print("Splitting")
  dataset = tbl.assign(ds.split_binary_mnist(dataset, 1), dataset)
  dataset.n_features = FEATURES
  dataset.n_hidden = HIDDEN

  print("\nIndexing raw features")
  dataset.index_raw = ann.create({
    expected_size = dataset.n,
    features = FEATURES,
  })
  dataset.problems_raw = dataset.problems:raw_bitmap(dataset.n, FEATURES)
  dataset.index_raw:add(dataset.problems_raw, 0, dataset.n)

  ds.add_binary_mnist_pairs(dataset, STAR_CENTERS, STAR_NEGATIVES)
  str.printf("  Pos\t%d\n", dataset.pos:size() / 2)
  str.printf("  Neg\t%d\n", dataset.neg:size() / 2)
  str.printf("  Sent\t%d\n", dataset.n)

  print("\nCreating graph")
  local stopwatch = utc.stopwatch()
  dataset.graph = graph.create({
    pos = dataset.pos,
    neg = dataset.neg,
    index = dataset.index_raw,
    labels = dataset.solutions, -- TODO: ensure that alignment agrees with classes
    knn = KNN,
    trans_hops = TRANS_HOPS,
    trans_pos = TRANS_POS,
    trans_neg = TRANS_NEG,
    each = function (s, b, n, dt)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f  Stage: %-12s  Components: %-6d  Positives: %3d  Negatives: %3d\n", d, dd, dt, s, b, n) -- luacheck: ignore
    end
  })

  print("\nSpectral eigendecomposition")
  dataset.ids_spectral, dataset.codes_spectral = spectral.encode({
    graph = dataset.graph,
    n_hidden = HIDDEN,
    each = function (s)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f  Spectral Steps: %3d\n", d, dd, s)
    end
  })

  print("\nBooleanizing")
  if ITQ then
    dataset.n_hidden = HIDDEN
    dataset.codes_spectral = itq.encode({
      codes = dataset.codes_spectral,
      n_hidden = HIDDEN,
      each = function (i, j)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  ITQ Iter %d  Objective: %6.2f\n", d, dd, i, j)
      end
    })
  else
    local bzr = booleanizer.create({ n_thresholds = 1 })
    bzr:observe(dataset.codes, HIDDEN)
    bzr:finalize()
    dataset.codes_spectral = bzr:encode(dataset.codes_spectral, HIDDEN)
    dataset.n_hidden = bzr:bits()
  end

  if TCH then
    print("\nFlipping bits")
    tch.refine({
      codes = dataset.codes_spectral,
      graph = dataset.graph,
      n_hidden = dataset.n_hidden,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end

  print("\nCodebook stats")
  dataset.codes_spectral = dataset.codes_spectral:raw_bitmap(dataset.ids_spectral:size(), dataset.n_hidden)
  dataset.similarity = eval.optimize_retrieval({
    codes = dataset.codes_spectral,
    ids = dataset.ids_spectral,
    pos = dataset.pos,
    neg = dataset.neg,
    n_hidden = dataset.n_hidden,
    each = function (a, f, p, r, m)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | AUC: %6.2f | F1: %.2f | Precision: %.2f | Recall: %.2f | Margin: %.2f\n", -- luacheck: ignore
        d, dd, a, f, p, r, m)
    end
  })
  str.printi("\n  Best | AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", -- luacheck: ignore
    dataset.similarity)
  dataset.entropy = eval.entropy_stats(dataset.codes_spectral, dataset.n, dataset.n_hidden)
  str.printi("       | Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)\n",
    dataset.entropy)

  print("Creating index")
  dataset.index_codes = ann.create({
    expected_size = dataset.ids_spectral:size(),
    features = dataset.n_hidden,
  })
  print("  Adding codebook")
  dataset.index_codes:add(dataset.codes_spectral, dataset.ids_spectral)

  print("\nClustering\n")
  stopwatch()
  dataset.cluster_score, dataset.cluster_ids, dataset.cluster_assignments, dataset.n_clusters = eval.optimize_clustering({ -- luacheck: ignore
    index = dataset.index_codes,
    pos = dataset.pos,
    neg = dataset.neg,
    min = DBSCAN_MIN,
    each = function (f, p, r, m, c)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f | F1: %.2f | Precision: %.2f | Recall: %.2f | Margin: %.2f | Clusters: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m, c)
    end
  })
  dataset.cluster_score.n_clusters = dataset.n_clusters
  str.printi("\n  Best | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin) | Clusters: %d#(n_clusters)\n", -- luacheck: ignore
    dataset.cluster_score)

end)
