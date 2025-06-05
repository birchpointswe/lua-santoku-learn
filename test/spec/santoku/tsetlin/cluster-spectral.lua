local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

local tm = require("santoku.tsetlin")
local ds = require("santoku.tsetlin.dataset")
-- local inv = require("santoku.tsetlin.inv")
local booleanizer = require("santoku.tsetlin.booleanizer")
local ann = require("santoku.tsetlin.ann")
local cluster = require("santoku.tsetlin.cluster")
local eval = require("santoku.tsetlin.evaluator")
local graph = require("santoku.tsetlin.graph")
local tch = require("santoku.tsetlin.tch")
local tokenizer = require("santoku.tsetlin.tokenizer")
local spectral = require("santoku.tsetlin.spectral")

local TRAIN_TEST_RATIO = 0.8
local MAX_RECORDS = nil
local THREADS = nil
local HIDDEN = 64
local GRAPH_KNN = 1
local TRANS_HOPS = 1
local TRANS_POS = 1
local TRANS_NEG = 1
local ANN_BUCKET_TARGET = 2

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
  local dataset = ds.read_snli_pairs("test/res/snli.10k.txt", MAX_RECORDS)
  local train, test = ds.split_snli_pairs(dataset, TRAIN_TEST_RATIO)

  print()
  str.printf("\tAll\tTrain\tTest\n\n")
  str.printf("Pos\t%d\t%d\t%d\n", dataset.n_pos, train.n_pos, test.n_pos)
  str.printf("Neg\t%d\t%d\t%d\n", dataset.n_neg, train.n_neg, test.n_neg)
  str.printf("Sent\t%d\t%d\t%d\n", dataset.n_sentences, train.n_sentences, test.n_sentences)

  print("\nTraining tokenizer\n")
  tokenizer.train({ corpus = train.raw_sentences })
  tokenizer.finalize()
  dataset.n_features = tokenizer.features()
  str.printf("Feat\t\t%d\t\t\n", dataset.n_features)

  print("\nTokenizing train")
  train.sentences = tokenizer.tokenize(train.raw_sentences)
  local stopwatch = utc.stopwatch()

  -- print("\nIndexing train")
  -- train.index = inv.create({ features = dataset.n_features })
  -- train.index:add(train.sentences)

  print("Creating graph")
  train.graph = graph.create({
    pos = train.pos,
    neg = train.neg,
    -- index = train.index,
    nodes = train.sentences,
    n_nodes = train.n_sentences,
    n_features = dataset.n_features,
    knn = GRAPH_KNN,
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
  train.codes0 = spectral.encode({
    graph = train.graph,
    n_hidden = HIDDEN,
    threads = THREADS,
    each = function (s)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  Spectral Steps: %3d\n", d, dd, s)
    end
  })

  print("\nBooleanizing\n")
  local bzr = booleanizer.create({ n_thresholds = 1 })
  bzr:observe(train.codes0, HIDDEN)
  bzr:finalize()
  train.codes0 = bzr:encode(train.codes0, HIDDEN)
  train.n_hidden = bzr:bits()

  print("\nRefining\n")
  tch.refine({
    codes = train.codes0,
    graph = train.graph,
    n_hidden = train.n_hidden,
    threads = THREADS,
    each = function (s)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
    end
  })

  train.codes0_raw = train.codes0:raw_bitmap(train.n_sentences, train.n_hidden)
  train.similarity0 = eval.encoding_similarity(train.codes0_raw, train.pos, train.neg, train.n_hidden, THREADS)
  str.printi("AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", -- luacheck: ignore
    train.similarity0)
  train.entropy0 = eval.codebook_stats(train.codes0_raw, train.n_sentences, train.n_hidden, THREADS)
  str.printi("Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)\n",
    train.entropy0)

  print("Creating index")
  local index = ann.create({
    features = train.n_hidden,
    bucket_target = ANN_BUCKET_TARGET,
    probe_radius = 0,
    samples = train.n_sentences,
    guidance = train.codes0_raw,
    -- exhaustive = true,
    threads = THREADS,
  })
  index:add(0, train.codes0_raw, train.n_sentences)

  print("Clustering")
  local clusters = cluster.dbscan(index, 2, 10)
  for i = 1, #clusters do
    if i > 10 then
      break
    end
    str.printf("Cluster %d\n", i)
    local n = 0
    for m in clusters[i]:each() do
      m = m + 1
      n = n + 1
      if n > 10 then
        break
      end
      str.printf("  %s\n", train.raw_sentences[m])
    end
  end

end)
