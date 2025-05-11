local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local utc = require("santoku.utc")
local tm = require("santoku.tsetlin")
local codes = require("santoku.tsetlin.codebook")
local eval = require("santoku.tsetlin.evaluator")
local bm = require("santoku.bitmap")
local imtx = require("santoku.matrix.integer")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")
local err = require("santoku.error")

local TRAIN_TEST_RATIO = 0.8
local HIDDEN = 64
local HOPS_POS = 0
local HOPS_NEG = 1
local GROW_POS = 0
local GROW_NEG = 1
local CLAUSES = 512
local TARGET = 0.2
local STATE = 8
local BOOST = false
local ACTIVE = 0.8
local SPECIFICITY = 40
local THREADS = nil
local EVALUATE_EVERY = 1
local NEGATIVE = 0.0
local MAX_RECORDS = 100000
local MAX_EPOCHS = 500
local KNN = 32
local LBFGS_ITERS = 1000

local function tokenize (words, text)
  local m = {}
  local s = {}
  for w in str.gmatch(str.lower(text), "%S+") do
    local i = words[w]
    if not i then
      i = #words + 1
      words[w] = i
      words[i] = w
    end
    if not m[i] then
      arr.push(s, i - 1)
    end
    m[i] = true
  end
  err.assert(#s > 0, "empty sentence!")
  local m = imtx.create({ s })
  imtx.reshape(m, #s, 1)
  return m
end

local function read_data (fp, max)
  max = max or num.huge
  local words = {}
  local sentences = {}
  local pos = {}
  local neg = {}
  local ns = 0
  for line in it.take(max, it.drop(1, fs.lines(fp))) do
    local chunks = str.gmatch(line, "[^\t]+")
    local label = chunks()
    chunks = it.drop(4, chunks)
    local a = chunks()
    local b = chunks()
    err.assert(label and a and b, "unable to parse line", line)
    label = label == "entailment" and 1 or label == "contradiction" and 0 or nil
    if label then
      local na = sentences[a]
      local nb = sentences[b]
      if not na then
        ns = ns + 1
        na = ns
        sentences[a] = na
        sentences[na] = tokenize(words, a)
      end
      if not nb then
        ns = ns + 1
        nb = ns
        sentences[b] = nb
        sentences[nb] = tokenize(words, b)
      end
      if label == 1 then
        arr.push(pos, na, nb)
      else
        arr.push(neg, na, nb)
      end
    end
  end
  return {
    pos = pos,
    neg = neg,
    n_pos = #pos / 2,
    n_neg = #neg / 2,
    n_features = num.round(#words, tm.align),
    sentences = sentences,
    n_sentences = ns
  }
end

local function split_pairs (dataset, split, prop, start, size)
  local pairs_to = {}
  local pairs_from = dataset[prop]
  for i = start, start + size - 1 do
    local ia, ib =
      pairs_from[(i - start) * 2 + 1],
      pairs_from[(i - start) * 2 + 2]
    local na, nb =
      split.sentences_map[ia],
      split.sentences_map[ib]
    if not na then
      split.n_sentences = split.n_sentences + 1
      na = split.n_sentences
      split.sentences_map[ia] = na
      local s0 = imtx.create(dataset.sentences[ia])
      imtx.add(s0, (na - 1) * dataset.n_features)
      imtx.extend(split.sentences, s0)
    end
    if not nb then
      split.n_sentences = split.n_sentences + 1
      nb = split.n_sentences
      split.sentences_map[ib] = nb
      local s0 = imtx.create(dataset.sentences[ib])
      imtx.add(s0, (nb - 1) * dataset.n_features)
      imtx.extend(split.sentences, s0)
    end
    arr.push(pairs_to, na - 1, nb - 1)
  end
  split[prop] = imtx.create(pairs_to)
end

local function split_dataset (dataset, ratio)

  local test, train = {}, {}

  train.n_pos = num.floor(dataset.n_pos * ratio)
  train.n_neg = num.floor(dataset.n_neg * ratio)
  train.sentences = imtx.create(0, 1)
  train.sentences_map = {}
  train.n_sentences = 0

  test.n_pos = dataset.n_pos - train.n_pos
  test.n_neg = dataset.n_neg - train.n_neg
  test.sentences = imtx.create(0, 1)
  test.sentences_map = {}
  test.n_sentences = 0

  split_pairs(dataset, train, "pos", 1, train.n_pos)
  split_pairs(dataset, train, "neg", 1, train.n_neg)

  split_pairs(dataset, test, "pos", train.n_pos + 1, dataset.n_pos)
  split_pairs(dataset, test, "neg", train.n_neg + 1, dataset.n_neg)

  return train, test

end

test("tsetlin", function ()

  print("Reading data")
  local dataset = read_data("test/res/snli.train.txt", MAX_RECORDS)

  print("Raw pos", dataset.n_pos)
  print("Raw neg", dataset.n_neg)
  print("Raw sentences", dataset.n_sentences)
  print("Raw features", dataset.n_features)
  local train, test = split_dataset(dataset, TRAIN_TEST_RATIO)

  print("Generating codebook")
  local stopwatch = utc.stopwatch()
  train.codes0 = codes.codeify({
    pos = train.pos,
    neg = train.neg,
    sentences = train.sentences,
    n_sentences = train.n_sentences,
    n_hidden = HIDDEN,
    n_features = dataset.n_features,
    n_hops_pos = HOPS_POS,
    n_hops_neg = HOPS_NEG,
    n_grow_pos = GROW_POS,
    n_grow_neg = GROW_NEG,
    knn = KNN,
    lbfgs = LBFGS_ITERS,
    each = function (e, t, s, b, n)
      local d, dd = stopwatch()
      if t == "densify" then
        str.printf("Epoch: %3d  Time: %3.2f %3.2f  Components: %3d  Positives: %3d  Negatives: %3d\n", e, d, dd, s, b, n)
      elseif t == "spectral" then
        str.printf("Epoch: %3d  Time: %3.2f %3.2f  Spectral Steps: %3d\n", e, d, dd, s)
      elseif t == "lbfgs" then
        str.printf("Epoch: %3d  Time: %3.2f %3.2f  LBFGS Loss: %.4f\n", e, d, dd, s)
      elseif t == "tch" then
        str.printf("Epoch: %3d  Time: %3.2f %3.2f  Bit: %d  Steps: %3d\n", e, d, dd, s, b)
      end
    end
  })

  train.similarity0 = eval.encoding_similarity(
    train.codes0, train.pos, train.neg, train.n_sentences, HIDDEN)
  str.printi("AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)\n",
    train.similarity0)

  os.exit(0)

  imtx.flip_interleave(train.sentences, train.n_sentences, dataset.n_features)
  imtx.flip_interleave(test.sentences, test.n_sentences, dataset.n_features)

  -- TODO: Bitmap from list of bits
  train.sentences = bm.raw(train.sentences)
  test.sentenes = bm.raw(test.sentences)

  print("Input Features", dataset.n_features * 2)
  print("Encoded Features", HIDDEN)
  print("Train Pairs", train.n_pairs)
  print("Train Sentences", train.n_sentences)
  print("Test Pairs", test.n_pairs)
  print("Test Sentences", test.n_sentences)

  print("Creating encoder")
  local t = tm.encoder({
    visible = dataset.n_features,
    hidden = HIDDEN,
    clauses = CLAUSES,
    state = STATE,
    target = TARGET,
    boost = BOOST,
    specificity = SPECIFICITY,
    threads = THREADS,
  })

  print("Training encoder")
  stopwatch = utc.stopwatch()
  t.train({
    sentences = train.sentences,
    codes = train.codes,
    n_sentences = train.n_sentences,
    active = ACTIVE,
    margin = train.similarity0.margin,
    negative = NEGATIVE,
    iterations = MAX_EPOCHS,
    each = function (epoch)
      if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
        train.codes1 = t.predict(train.sentences, train.n_sentences)
        test.codes1 = t.predict(test.sentences, test.n_sentences)
        train.accuracy = eval.encoding_accuracy(
          train.codes1, train.codes, dataset.n_features, train.n_sentences)
        train.similarity1 = eval.encoding_similarity(
          train.codes1, train.pos, train.neg, train.n_sentences, HIDDEN)
        test.similarity1 = eval.encoding_similarity(
          test.codes1, test.pos, test.neg, test.n_sentences, HIDDEN)
        str.printf("Epoch %3d  Time %3.2f %3.2f\n",
          epoch, stopwatch())
        str.printi("  Train (acc) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall)", train.accuracy)
        str.printi("  Train (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity1)
        str.printi("  Test (sim)  | AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity1)
      else
        str.printf("Epoch %3d  Time %3.2f %3.2f\n",
          epoch, stopwatch())
      end
    end
  })

  print()
  print("Persisting")
  fs.rm("model.bin", true)
  t.persist("model.bin")
  print("Restoring")
  t = tm.load("model.bin")
  -- local test_score = t.evaluate({ pairs = test_pairs, labels = test_labels, samples = n_test, margin = train_stats.margin })
  -- local train_score = t.evaluate({ pairs = train_pairs, labels = train_labels, samples = n_train, margin = train_stats.margin })
  -- str.printf("Evaluate Test %4.2f  Train %4.2f\n", test_score, train_score)

end)
