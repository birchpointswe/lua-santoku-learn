local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local utc = require("santoku.utc")
local json = require("cjson")
local tm = require("santoku.tsetlin")
local tokenizer = require("santoku.tsetlin.tokenizer")
local codes = require("santoku.tsetlin.codebook")
local eval = require("santoku.tsetlin.evaluator")
local imtx = require("santoku.matrix.integer")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")
local err = require("santoku.error")

local TRAIN_TEST_RATIO = 0.8
local MAX_EPOCHS = 200
local MAX_RECORDS = 2000
-- local MAX_RECORDS = 16000
local HIDDEN = 64
local CLAUSES = 512
-- local CLAUSES = 4096
local TARGET = 0.25
local STATE = 8
local BOOST = true
local SPECTRAL = true
local TCH = true
local HOPS = 0
local GROW_POS = 0
local GROW_NEG = 1
local KNN = 32
-- local TOP_CHI2 = nil
local TOP_CHI2 = 1024
-- local TOP_CHI2 = 4096
-- local TOP_MI = 1024
local TOP_MI = nil
local TOP_STRATEGY = "round-robin"
-- local TOP_STRATEGY = "mi-proportional"
local TOKENIZER_CONFIG = {
  max_df = 0.95,
  min_df = 0.001,
  max_len = 20,
  min_len = 1,
  max_run = 2,
  ngrams = 3,
  cgrams_min = 4,
  cgrams_max = 4,
  skips = 2,
  negations = 3,
  align = tm.align
}

local EVALUATE_EVERY = 1
local THREADS = nil

local function read_data (fp, max)
  max = max or num.huge
  local sentences = {}
  local pos = {}
  local neg = {}
  local ns = 0
  -- for line in it.drop(1, fs.lines(fp)) do
  --   local data = json.decode(line)
  --   local label = data.label
  --   local a = data.text1
  --   local b = data.text1
  --   err.assert(label and a and b, "unable to parse line", line)
  --   label = label == 1 and 1 or label == 0 and 0 or nil
  for line in it.drop(1, fs.lines(fp)) do
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
        sentences[na] = a
      end
      if not nb then
        ns = ns + 1
        nb = ns
        sentences[b] = nb
        sentences[nb] = b
      end
      if label == 1 then
        arr.push(pos, na, nb)
      else
        arr.push(neg, na, nb)
      end
      if (#pos + #neg) / 2 >= max then
        break
      end
    end
  end
  return {
    pos = pos,
    neg = neg,
    n_pos = #pos / 2,
    n_neg = #neg / 2,
    raw_sentences = sentences,
    n_sentences = ns,
  }
end

local function split_pairs (dataset, split, prop, start, size)
  local pairs_to = {}
  local pairs_from = dataset[prop]
  for i = start, start + size - 1 do
    local ia, ib =
      pairs_from[(i - start) * 2 + 1],
      pairs_from[(i - start) * 2 + 2]
    local sa, sb =
      dataset.raw_sentences[ia],
      dataset.raw_sentences[ib]
    local na, nb =
      split.raw_sentences[sa],
      split.raw_sentences[sb]
    if not na then
      split.n_sentences = split.n_sentences + 1
      na = split.n_sentences
      split.raw_sentences[na] = sa
      split.raw_sentences[sa] = na
      -- -- TODO: implement imtx.extend_add(a, b, v) which copies b to a while
      -- -- adding v to each of the copied values
      -- local s0 = imtx.create(dataset.sentences[ia])
      -- imtx.add(s0, (na - 1) * dataset.n_features)
      -- imtx.extend(split.sentences, s0)
    end
    if not nb then
      split.n_sentences = split.n_sentences + 1
      nb = split.n_sentences
      split.raw_sentences[nb] = sb
      split.raw_sentences[sb] = nb
      -- local s0 = imtx.create(dataset.sentences[ib])
      -- imtx.add(s0, (nb - 1) * dataset.n_features)
      -- imtx.extend(split.sentences, s0)
    end
    arr.push(pairs_to, na - 1, nb - 1)
  end
  split[prop] = imtx.create(pairs_to)
end

local function split_dataset (dataset, ratio)

  local test, train = {}, {}

  train.n_pos = num.floor(dataset.n_pos * ratio)
  train.n_neg = num.floor(dataset.n_neg * ratio)
  train.n_sentences = 0
  train.raw_sentences = {}

  test.n_pos = dataset.n_pos - train.n_pos
  test.n_neg = dataset.n_neg - train.n_neg
  test.n_sentences = 0
  test.raw_sentences = {}

  split_pairs(dataset, train, "pos", 1, train.n_pos)
  split_pairs(dataset, train, "neg", 1, train.n_neg)

  split_pairs(dataset, test, "pos", train.n_pos + 1, test.n_pos)
  split_pairs(dataset, test, "neg", train.n_neg + 1, test.n_neg)

  return train, test

end

test("tsetlin", function ()

  print("Creating tokenizer")
  local tokenizer = tokenizer.create(TOKENIZER_CONFIG)

  print("Reading data")
  -- local dataset = read_data("test/res/qqp.train.jsonl", MAX_RECORDS)
  local dataset = read_data("test/res/snli.train.txt", MAX_RECORDS)

  print("Splitting")
  local train, test = split_dataset(dataset, TRAIN_TEST_RATIO)

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

  print("Generating codebook\n")
  local stopwatch = utc.stopwatch()
  train.codes0 = codes.codeify({
    pos = train.pos,
    neg = train.neg,
    sentences = train.sentences,
    n_sentences = train.n_sentences,
    n_features = dataset.n_features,
    n_hidden = HIDDEN,
    n_hops = HOPS,
    n_grow_pos = GROW_POS,
    n_grow_neg = GROW_NEG,
    knn = KNN,
    tch = TCH,
    spectral = SPECTRAL,
    learnability = LEARNABILITY,
    oversample = OVERSAMPLE,
    each = function (e, t, s, b, n, dt)
      local d, dd = stopwatch()
      if t == "densify" then
        str.printf("Epoch: %3d  Time: %6.2f %6.2f  Densify: %-9s  Components: %-6d  Positives: %3d  Negatives: %3d\n", e, d, dd, dt, s, b, n) -- luacheck: ignore
      elseif t == "spectral" then
        str.printf("Epoch: %3d  Time: %6.2f %6.2f  Spectral Steps: %3d\n", e, d, dd, s) -- luacheck: ignore
      elseif t == "tch" then
        str.printf("Epoch: %3d  Time: %6.2f %6.2f  TCH Steps: %3d\n", e, d, dd, s) -- luacheck: ignore
      elseif t == "downsample" then
        str.printf("Epoch: %3d  Time: %6.2f %6.2f  Downsample AUC Min: %6.4f Max: %6.4f\n", e, d, dd, s, b) -- luacheck: ignore
      end
    end
  })

  train.similarity0 = eval.encoding_similarity(
    train.codes0, train.pos, train.neg, train.n_sentences, HIDDEN)
  str.printi("AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", -- luacheck: ignore
    train.similarity0)
  train.entropy0 = eval.codebook_stats(train.codes0, train.n_sentences, HIDDEN)
  str.printi("Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)\n",
    train.entropy0)

  local top_v, n_top_v

  if TOP_CHI2 then
    -- chi2 filtering per-bit, round-robin union of per-bit top-k
    top_v = imtx.top_chi2(train.sentences, train.codes0, train.n_sentences, dataset.n_features, HIDDEN, TOP_CHI2, TOP_STRATEGY) -- luacheck: ignore
    n_top_v = imtx.columns(top_v)
    print("After top Chi2 filter", n_top_v)
  elseif TOP_MI then
    -- mi filtering per-bit, round-robin union of per-bit top-k
    top_v = imtx.top_mi(train.sentences, train.codes0, train.n_sentences, dataset.n_features, HIDDEN, TOP_MI, TOP_STRATEGY) -- luacheck: ignore
    n_top_v = imtx.columns(top_v)
    print("After top MI filter", n_top_v)
  else
    top_v = imtx.create(1, dataset.n_features)
    n_top_v = dataset.n_features
    imtx.fill_indices(top_v)
    print("Using unfiltered features")
  end

  -- Show top words
  local words = tokenizer.index()
  for id in it.take(32, imtx.each(top_v)) do
    print(id, words[id + 1])
  end

  print("Re-encoding train/test with top features")
  tokenizer.restrict(top_v)
  train.sentences = tokenizer.tokenize(train.raw_sentences)
  test.sentences = tokenizer.tokenize(test.raw_sentences)

  print("Prepping for encoder")
  imtx.flip_interleave(train.sentences, train.n_sentences, n_top_v)
  imtx.flip_interleave(test.sentences, test.n_sentences, n_top_v)
  train.sentences = imtx.raw_bitmap(train.sentences, train.n_sentences, n_top_v * 2)
  test.sentences = imtx.raw_bitmap(test.sentences, test.n_sentences, n_top_v * 2)

  print()
  print("Input Features", n_top_v * 2)
  print("Encoded Features", HIDDEN)
  print("Train Sentences", train.n_sentences)
  print("Test Sentences", test.n_sentences)

  print("Creating encoder")
  local t = tm.encoder({
    visible = n_top_v,
    hidden = HIDDEN,
    clauses = CLAUSES,
    state = STATE,
    target = TARGET,
    boost = BOOST,
    threads = THREADS,
  })

  print("Training encoder")
  stopwatch = utc.stopwatch()
  t.train({
    sentences = train.sentences,
    codes = train.codes0,
    samples = train.n_sentences,
    iterations = MAX_EPOCHS,
    each = function (epoch)
      if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
        train.codes1 = t.predict(train.sentences, train.n_sentences)
        test.codes1 = t.predict(test.sentences, test.n_sentences)
        train.accuracy0 = eval.encoding_accuracy(
          train.codes1, train.codes0, train.n_sentences, HIDDEN)
        -- for i = 1, #train.accuracy0.classes do
        --   str.printf("Bit %d:   F1: %.2f  Precision: %.2f  Recall: %.2f\n", i, train.accuracy0.classes[i].f1, train.accuracy0.classes[i].precision, train.accuracy0.classes[i].recall)
        -- end
        train.similarity1 = eval.encoding_similarity(
          train.codes1, train.pos, train.neg, train.n_sentences, HIDDEN)
        test.similarity1 = eval.encoding_similarity(
          test.codes1, test.pos, test.neg, test.n_sentences, HIDDEN)
        print()
        str.printf("Epoch %3d  Time %3.2f %3.2f\n",
          epoch, stopwatch())
        -- print()
        -- for i = 1, HIDDEN do
        --   str.printf("Bit %2d  F1 %.2f  ", i, train.accuracy0.classes[i].f1)
        --   -- if i % 4 == 0 then
        --     str.printf("\n")
        --   -- end
        -- end
        print()
        str.printi("  Train (acc) |           | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | F1 Spread: %.2f#(f1_min) %.2f#(f1_max) %.2f#(f1_std)", train.accuracy0) -- luacheck: ignore
        str.printi("  Codes (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity0) -- luacheck: ignore
        str.printi("  Train (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity1) -- luacheck: ignore
        str.printi("  Test (sim)  | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", test.similarity1) -- luacheck: ignore
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
  -- local test_score = t.evaluate({ pairs = test_pairs, labels = test_labels, samples = n_test, margin = train_stats.margin }) -- luacheck: ignore
  -- local train_score = t.evaluate({ pairs = train_pairs, labels = train_labels, samples = n_train, margin = train_stats.margin }) -- luacheck: ignore
  -- str.printf("Evaluate Test %4.2f  Train %4.2f\n", test_score, train_score)

end)
