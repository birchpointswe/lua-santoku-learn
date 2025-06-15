local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local ann = require("santoku.tsetlin.ann")
local graph = require("santoku.tsetlin.graph")
local spectral = require("santoku.tsetlin.spectral")
local booleanizer = require("santoku.tsetlin.booleanizer")
local tch = require("santoku.tsetlin.tch")
local itq = require("santoku.tsetlin.itq")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = nil
local TM_ITERS = 10

local ITQ = true
local TCH = true
local HIDDEN = 8
local KNN = 0
local STAR_CENTERS = 1
local STAR_NEGATIVES = 1
local ANN_BUCKET_TARGET = 4
local TRANS_HOPS = 1
local TRANS_POS = 1
local TRANS_NEG = 1
local CLAUSES = 256
local TARGET = 32
local SPECIFICITY = 10

local FEATURES = 784

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/BinarizedMNISTData/MNISTTraining.txt", FEATURES, MAX)
  local train, test = ds.split_binary_mnist(dataset, TTR)

  str.printf("\nIndexing\tTrain\tTest\n")
  train.index = ann.create({
    expected_size = train.n,
    bucket_target = ANN_BUCKET_TARGET,
    features = FEATURES,
  })
  train.index:add(train.problems:raw_bitmap(train.n, FEATURES), 0, train.n)
  ds.add_binary_mnist_pairs(train, STAR_CENTERS, STAR_NEGATIVES)
  ds.add_binary_mnist_pairs(test, STAR_CENTERS, STAR_NEGATIVES)
  str.printf("  Positives\t%d\t%d\n", train.pos:size() / 2, test.pos:size() / 2)
  str.printf("  Negatives\t%d\t%d\n", train.neg:size() / 2, test.neg:size() / 2)
  str.printf("  Samples\t%d\t%d\n", train.n, test.n)

  print("\nCreating graph")
  local stopwatch = utc.stopwatch()
  train.graph = graph.create({
    pos = train.pos,
    neg = train.neg,
    index = train.index,
    knn = KNN,
    labels = dataset.solutions, -- TODO: ensure that alignment agrees with classes
    trans_hops = TRANS_HOPS,
    trans_pos = TRANS_POS,
    trans_neg = TRANS_NEG,
    each = function (s, b, n, dt)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f  Graph: %-9s  Components: %-6d  Positives: %3d  Negatives: %3d\n", d, dd, dt, s, b, n) -- luacheck: ignore
    end
  })

  print("\nSpectral hashing")
  train.ids0, train.codes0 = spectral.encode({
    graph = train.graph,
    n_hidden = HIDDEN,
    each = function (s)
      local d, dd = stopwatch()
      str.printf("  Time: %6.2f %6.2f  Spectral Steps: %3d\n", d, dd, s)
    end
  })

  print("\nBooleanizing")
  if ITQ then
    train.n_hidden = HIDDEN
    train.codes0 = itq.encode({
      codes = train.codes0,
      n_hidden = HIDDEN,
      each = function (i, j)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  ITQ Iter %d  Objective: %6.2f\n", d, dd, i, j)
      end
    })
  else
    local bzr = booleanizer.create({ n_thresholds = 1 })
    bzr:observe(train.codes0, HIDDEN)
    bzr:finalize()
    train.codes0 = bzr:encode(train.codes0, HIDDEN)
    train.n_hidden = bzr:bits()
  end

  if TCH then
    print("\nRefining")
    tch.refine({
      codes = train.codes0,
      graph = train.graph,
      n_hidden = train.n_hidden,
      each = function (s)
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f  TCH Steps: %3d\n", d, dd, s)
      end
    })
  end

  train.codes0_raw = train.codes0:raw_bitmap(train.n, train.n_hidden)
  train.similarity0 = eval.optimize_retrieval({
    codes = train.codes0_raw,
    ids = train.ids0,
    pos = train.pos,
    neg = train.neg,
    n_hidden = train.n_hidden
  })

  print("\nCodebook Stats")
  str.printi("  AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", -- luacheck: ignore
    train.similarity0)
  train.entropy0 = eval.entropy_stats(train.codes0_raw, train.n, train.n_hidden)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)",
    train.entropy0)

  print("\nPrepping for encoder")
  train.problems:bits_rearrange(train.ids0, dataset.n_features)
  train.n = train.ids0:size()
  train.problems:flip_interleave(train.n, FEATURES)
  test.problems:flip_interleave(test.n, FEATURES)
  train.problems = train.problems:raw_bitmap(train.n, FEATURES * 2)
  test.problems = test.problems:raw_bitmap(test.n, FEATURES * 2)

  print()
  str.printf("Input Features    %d\n", FEATURES * 2)
  str.printf("Encoded Features  %d\n", train.n_hidden)
  str.printf("Train problems    %d\n", train.n)
  str.printf("Test problems     %d\n", test.n)

  print("\nCreating encoder")
  local t = tm.encoder({
    visible = FEATURES,
    hidden = train.n_hidden,
    clauses = CLAUSES,
    target = TARGET,
    specificity = SPECIFICITY,
  })
  stopwatch = utc.stopwatch()
  t.train({
    sentences = train.problems,
    codes = train.codes0_raw,
    samples = train.n,
    iterations = TM_ITERS,
    each = function (epoch)
      train.codes1 = t.predict(train.problems, train.n)
      test.codes1 = t.predict(test.problems, test.n)
      train.accuracy0 = eval.encoding_accuracy(train.codes1, train.codes0_raw, train.n, train.n_hidden) -- luacheck: ignore
      train.similarity1 = eval.optimize_retrieval({
        codes = train.codes1,
        ids = train.ids0,
        pos = train.pos,
        neg = train.neg,
        n_hidden = train.n_hidden
      })
      test.similarity1 = eval.optimize_retrieval({
        codes = test.codes1,
        ids = train.ids0,
        pos = test.pos,
        neg = test.neg,
        n_hidden = train.n_hidden
      })
      print()
      str.printf("  Epoch %3d  Time %3.2f %3.2f\n",
        epoch, stopwatch())
      print()
      -- print(serialize(train.accuracy0))
      str.printi("    Train (acc) |           | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | F1 Spread: %.2f#(f1_min) %.2f#(f1_max) %.2f#(f1_std)", train.accuracy0) -- luacheck: ignore
      str.printi("    Codes (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity0) -- luacheck: ignore
      str.printi("    Train (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity1) -- luacheck: ignore
      str.printi("    Test (sim)  | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", test.similarity1) -- luacheck: ignore
    end
  })

end)
