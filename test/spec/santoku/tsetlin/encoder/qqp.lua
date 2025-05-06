local serialize = require("santoku.serialize") -- luacheck: ignore
local json = require("cjson")
local test = require("santoku.test")
local utc = require("santoku.utc")
local tbl = require("santoku.table")
local tm = require("santoku.tsetlin")
local codes = require("santoku.tsetlin.codebook")
local eval = require("santoku.tsetlin.evaluator")
local bm = require("santoku.bitmap")
local mtx = require("santoku.matrix.integer")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local rand = require("santoku.random")
local num = require("santoku.num")
local err = require("santoku.error")

local TRAIN_TEST_RATIO = 0.8
local HIDDEN = 64
local CLAUSES = 512
local TARGET = 0.2
local NEGATIVE = 0.0
local STATE = 8
local BOOST = false
local ACTIVE = 0.8
local SPECIFICITY = 40
local THREADS = nil
local EVALUATE_EVERY = 1
local MAX_RECORDS = 10000
local MAX_EPOCHS = 500
local SPECTRAL_ITERS = 10000
local LBFGS_ITERS = 0

local function tokenize (words, text)
  local b = bm.create()
  for w in str.gmatch(str.gsub(str.lower(text), "[^%a]", ""), "%S+") do
    local i = words[w]
    if not i then
      i = #words + 1
      words[w] = i
      words[i] = w
    end
    bm.set(b, i)
  end
  return b
end

local function read_data (fp, max)
  max = max or num.huge
  local sentences = {}
  local graph = {}
  local words = {}
  -- local data = it.collect(it.take(max, fs.lines(fp)))
  local data = it.collect(it.map(json.decode, it.take(max, fs.lines(fp))))
  for line in it.ivals(data) do
    -- local chunks = str.splits(line, "\t")
    -- local label = str.sub(chunks())
    local label = line.label
    if label == 0 or label == 1 then
    -- if label == "entailment" or label == "contradiction" then
      -- chunks = it.drop(4, chunks)
      -- local a = str.sub(chunks())
      -- local b = str.sub(chunks())
      -- label = label == "entailment" and 1 or 0
      local a = line.text1
      local b = line.text2
      local n1 = sentences[a]
      local n2 = sentences[b]
      if not n1 then
        arr.push(sentences, tokenize(words, a))
        n1 = #sentences
        sentences[a] = n1
      end
      if not n2 then
        arr.push(sentences, tokenize(words, b))
        n2 = #sentences
        sentences[b] = n2
      end
      tbl.set(graph, n1, n2, label)
      tbl.update(graph, n1, "n", function (n)
        return (n or 0) + 1
      end)
    end
  end
  local pairs = {}
  local labels = {}
  for n1, n2s in it.pairs(graph) do
    if tbl.get(graph, n1, "n") > 1 then
      for n2, label in it.pairs(n2s) do
        if n2 ~= "n" then
          arr.push(pairs, n1, n2)
          arr.push(labels, label)
        end
      end
    end
  end
  return {
    sentences = sentences,
    pairs = pairs,
    labels = labels,
    n_features = num.round(#words, tm.align),
    n_pairs = #pairs / 2,
  }
end

local function split_dataset (dataset, s, e)
  local n = e - s + 1
  local ns = 0
  local split_sentences_seen = {}
  local split_sentences = bm.create()
  local split_pairs = mtx.create(n, 2)
  local split_labels = bm.create()
  for i = s, e do
    local ia = dataset.pairs[(i - 1) * 2 + 1]
    local ib = dataset.pairs[(i - 1) * 2 + 2]
    local sa = dataset.sentences[ia]
    local sb = dataset.sentences[ib]
    local l = dataset.labels[i]
    local n1 = split_sentences_seen[ia]
    local n2 = split_sentences_seen[ib]
    if not n1 then
      n1 = ns
      ns = ns + 1
      split_sentences_seen[ia] = n1
      bm.extend(split_sentences, sa, n1 * dataset.n_features + 1)
    end
    if not n2 then
      n2 = ns
      ns = ns + 1
      split_sentences_seen[ib] = n2
      bm.extend(split_sentences, sb, n2 * dataset.n_features + 1)
    end
    mtx.set(split_pairs, i - s + 1, 1, n1)
    mtx.set(split_pairs, i - s + 1, 2, n2)
    if l == 1 then
      bm.set(split_labels, i - s + 1)
    elseif l ~= 0 then
      err.error("Unexpected label while splitting dataset", l)
    end
  end
  split_sentences = bm.flip_interleave(split_sentences, ns, dataset.n_features)
  split_sentences = bm.raw(split_sentences, ns * 2 * dataset.n_features)
  return ns, split_sentences, split_pairs, split_labels
end

local function evaluate (l, e, sw, t, ss, cs, ps, ls, nps, nss, h)
  str.printf("Epoch: %3d  Time: %3.2f %3.2f\n", e, sw())
  local cs0
  if ss and not cs then
    cs = t.predict(ss, nss)
  elseif ss and cs then
    cs0 = t.predict(ss, nss)
  end
  str.printf("  %s (sim)\t|  ", l)
  str.printi(
    "AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)",
      eval.encoding_similarity(cs, ls, h, nps, nss))
  if cs0 then
    str.printf("  %s (enc)\t|  ", l)
    str.printi(
      "F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall)",
        eval.encoding_accuracy(cs0, cs, h, nss))
  end
end

test("tsetlin", function ()

  print("Reading data")
  -- local dataset = read_data("test/res/snli.train.txt", MAX_RECORDS)
  local dataset = read_data("test/res/qqp.train.jsonl", MAX_RECORDS)

  print("Total pairs", dataset.n_pairs)
  print("Splitting")
  local n_train_pairs = num.floor(dataset.n_pairs * TRAIN_TEST_RATIO)
  local n_test_pairs = dataset.n_pairs - n_train_pairs
  local n_train_sentences, train_sentences, train_pairs, train_labels =
    split_dataset(dataset, 1, n_train_pairs)
  local n_test_sentences, test_sentences, test_pairs, test_labels =
    split_dataset(dataset, n_train_pairs + 1, n_train_pairs + n_test_pairs)

  print("Densifying")
  print("Initial Components", codes.components(train_pairs, n_train_sentences))
  codes.densify(train_pairs, train_labels, n_train_sentences)
  n_train_pairs = mtx.rows(train_pairs);
  print("Dense Components", codes.components(train_pairs, n_train_sentences))

  print("Raw")
  train_pairs = mtx.raw(train_pairs, nil, nil, "u32");
  train_labels = bm.raw(train_labels, num.round(n_train_pairs, tm.align))
  test_pairs = mtx.raw(test_pairs, nil, nil, "u32");
  test_labels = bm.raw(test_labels, num.round(n_test_pairs, tm.align))

  print("Generating codebook")
  local stopwatch = utc.stopwatch()
  local train_codes = codes.codeify({
    pairs = train_pairs,
    labels = train_labels,
    n_pairs = n_train_pairs,
    n_features = HIDDEN,
    n_codes = n_train_sentences,
    spectral_iterations = SPECTRAL_ITERS,
    lbfgs_iterations = LBFGS_ITERS,
    each = function (e, s)
      local d, t = stopwatch()
      if e == 1 then
        str.printf("Epoch: %3d  Time: %3.2f %3.2f  Lanczos Steps: %3d\n", e, d, t, s)
      else
        str.printf("Epoch: %3d  Time: %3.2f %3.2f  Loss: %.4f\n", e, d, t, s)
      end
    end
  })

  local train_stats = eval.encoding_similarity(train_codes, train_pairs, train_labels, HIDDEN, n_train_pairs, n_train_sentences)
  str.printi("AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)\n", train_stats)

  print("Creating encoder")
  print("Input Features", dataset.n_features * 2)
  print("Encoded Features", HIDDEN)
  print("Train Pairs", n_train_pairs)
  print("Train Sentences", n_train_sentences)
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
    sentences = train_sentences,
    codes = train_codes,
    samples = n_train_sentences,
    active = ACTIVE,
    margin = train_stats.margin,
    negative = NEGATIVE,
    iterations = MAX_EPOCHS,
    each = function (epoch)
      if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
        evaluate("Test", epoch, stopwatch, t,
          test_sentences, nil, test_pairs, test_labels, n_test_pairs, n_test_sentences, HIDDEN)
        evaluate("Train", epoch, stopwatch, t,
          train_sentences, train_codes, train_pairs, train_labels, n_train_pairs, n_train_sentences, HIDDEN)
      else
        str.printf("Epoch: %3d  Time %3.2f %3.2f\n",
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
