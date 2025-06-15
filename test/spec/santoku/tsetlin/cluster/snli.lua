local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local tm = require("santoku.tsetlin")
local ds = require("santoku.tsetlin.dataset")
local inv = require("santoku.tsetlin.inv")
local ann = require("santoku.tsetlin.ann")
local eval = require("santoku.tsetlin.evaluator")
local graph = require("santoku.tsetlin.graph")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local tokenizer = require("santoku.tsetlin.tokenizer")
local spectral = require("santoku.tsetlin.spectral")

local MAX_RECORDS = nil
local THREADS = nil
local HIDDEN = 64
local KNN = 0
local TRANS_HOPS = 0
local TRANS_POS = 0
local TRANS_NEG = 0
local ANN_BUCKET_TARGET = 10
local DBSCAN_MIN = 1

local TOKENIZER_CONFIG = {
  max_df = 0.95,
  min_df = 0.001,
  max_len = 20,
  min_len = 1,
  max_run = 1,
  ngrams = 3,
  cgrams_min = 3,
  cgrams_max = 4,
  skips = 2,
  negations = 4,
  align = tm.align
}

test("tsetlin", function ()

  print("Creating tokenizer")
  local tokenizer = tokenizer.create(TOKENIZER_CONFIG)

  print("Reading data")
  local dataset = ds.read_snli_pairs("test/res/snli.10k.txt", MAX_RECORDS, true)

  print()
  str.printf("Pos\t%d\n", dataset.n_pos)
  str.printf("Neg\t%d\n", dataset.n_neg)
  str.printf("Sent\t%d\n", dataset.n_sentences)

  print("\nTraining tokenizer\n")
  tokenizer.train({ corpus = dataset.raw_sentences })
  tokenizer.finalize()
  dataset.n_features = tokenizer.features()
  str.printf("Feat\t\t%d\t\t\n", dataset.n_features)

  print("\nTokenizing train")
  dataset.sentences = tokenizer.tokenize(dataset.raw_sentences)
  local stopwatch = utc.stopwatch()

  print("\nIndexing train")
  dataset.index = inv.create({ features = dataset.n_features })
  dataset.index:add(dataset.sentences, 0, dataset.n_sentences)

  print("Creating graph\n")
  dataset.graph = graph.create({
    pos = dataset.pos,
    neg = dataset.neg,
    index = dataset.index,
    knn = KNN,
    trans_hops = TRANS_HOPS,
    trans_pos = TRANS_POS,
    trans_neg = TRANS_NEG,
    threads = THREADS,
    each = function (s, b, n, dt)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  Graph: %-9s  Components: %-6d  Positives: %3d  Negatives: %3d\n", d, dd, dt, s, b, n) -- luacheck: ignore
    end
  })

  print("\nSpectral hashing\n")
  dataset.ids_spectral, dataset.codes_spectral = spectral.encode({
    graph = dataset.graph,
    n_hidden = HIDDEN,
    threads = THREADS,
    each = function (s)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  Spectral Steps: %3d\n", d, dd, s)
    end
  })

  print("\nBooleanizing\n")
  dataset.codes_spectral = itq.encode({
    codes = dataset.codes_spectral,
    n_hidden = HIDDEN,
    threads = THREADS,
    each = function (i, j)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  ITQ Iter %d  Objective: %6.2f\n", d, dd, i, j)
    end
  })

  print("\nRefining\n")
  tch.refine({
    codes = dataset.codes_spectral,
    graph = dataset.graph,
    n_hidden = HIDDEN,
    threads = THREADS,
    each = function (s)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
    end
  })

  dataset.codes_raw = dataset.codes_spectral:raw_bitmap(dataset.n_sentences, HIDDEN)
  dataset.similarity0 = eval.optimize_retrieval({
    codes = dataset.codes_raw,
    ids = dataset.ids_spectral,
    pos = dataset.pos,
    neg = dataset.neg,
    n_hidden = HIDDEN, THREADS
  })
  str.printi("AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", -- luacheck: ignore
    dataset.similarity0)
  dataset.entropy0 = eval.entropy_stats(dataset.codes_raw, dataset.n_sentences, HIDDEN, THREADS)
  str.printi("Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)\n",
    dataset.entropy0)

  print("Creating index")
  local index = ann.create({
    expected_size = dataset.n_sentences,
    bucket_target = ANN_BUCKET_TARGET,
    features = HIDDEN,
    threads = THREADS,
  })
  index:add(dataset.codes_raw, dataset.ids_spectral)

  print("Clustering\n")
  stopwatch()
  dataset.cluster_score, dataset.ids_cluster, dataset.clusters = eval.optimize_clustering({
    index = index,
    pos = dataset.pos,
    neg = dataset.neg,
    min = DBSCAN_MIN,
    threads = THREADS,
    each = function (f, p, r, m, c)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f | F1: %.2f | Precision: %.2f | Recall: %.2f | Margin: %.2f | Clusters: %d\n", -- luacheck: ignore
        d, dd, f, p, r, m, c)
    end
  })
  dataset.cluster_score.n_clusters = #dataset.clusters
  str.printi("\nBest | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin) | Clusters: %d#(n_clusters)\n", -- luacheck: ignore
    dataset.cluster_score)

  local avg_size = 0
  for i = 1, #dataset.clusters do
    if i < 10 then
      str.printf("Cluster %d, (%d members)\n", i, dataset.clusters[i]:size())
      local n = 0
      for m in dataset.clusters[i]:each() do
        m = m + 1
        n = n + 1
        if n > 10 then
          break
        end
        str.printf("  %s\n", dataset.raw_sentences[m])
      end
    end
    avg_size = avg_size + dataset.clusters[i]:size()
  end
  avg_size = avg_size / #dataset.clusters
  str.printf("Average size %d\n", avg_size)

end)
