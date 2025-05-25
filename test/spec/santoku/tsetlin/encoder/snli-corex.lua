local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local utc = require("santoku.utc")
local tm = require("santoku.tsetlin")
local ds = require("santoku.tsetlin.dataset")
local tokenizer = require("santoku.tsetlin.tokenizer")
local graph = require("santoku.tsetlin.graph")
local threshold = require("santoku.tsetlin.threshold")
local corex = require("santoku.corex")
local eval = require("santoku.tsetlin.evaluator")
local mtx = require("santoku.matrix.integer")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")

local TRAIN_TEST_RATIO = 0.8
local TM_ITERS = 10
local COREX_ITERS = 100
local MAX_RECORDS = 1000
local EVALUATE_EVERY = 1
local THREADS = nil

local HIDDEN = 64
local CLAUSES = 512
local TARGET = 0.1
local SPECIFICITY = 60

local GRAPH_KNN = 1
local TRANS_HOPS = 1
local TRANS_POS = 1
local TRANS_NEG = 1
local TOP_ALGO = "chi2"
local TOP_K = 1024

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
  train.pos_enriched = mtx.create(train.pos)
  train.neg_enriched = mtx.create(train.neg)

  print("Enriching graph")
  graph.enrich({
    sentences = train.sentences,
    n_sentences = train.n_sentences,
    n_features = dataset.n_features,
    pos = train.pos_enriched,
    neg = train.neg_enriched,
    knn = GRAPH_KNN,
    trans_hops = TRANS_HOPS,
    trans_pos = TRANS_POS,
    trans_neg = TRANS_NEG,
    each = function (s, b, n, dt)
      local d, dd = stopwatch()
      str.printf("Time: %6.2f %6.2f  Graph: %-9s  Components: %-6d  Positives: %3d  Negatives: %3d\n", d, dd, dt, s, b, n) -- luacheck: ignore
    end
  })

  print("Creating Corex")
  train.graph_corpus = graph.render(train.pos_enriched, train.n_sentences)
  local cor = corex.create({
    visible = train.n_sentences,
    hidden = HIDDEN,
    threads = THREADS,
  })

  print("Training Corex")
  cor.train({
    corpus = train.graph_corpus,
    samples = train.n_sentences,
    iterations = COREX_ITERS,
    each = function (epoch, tc, dev)
      local duration, total = stopwatch()
      str.printf("Epoch  %-4d   Time  %6.3f  %6.3f   Convergence  %4.6f  %4.6f\n",
        epoch, duration, total, tc, dev)
    end
  })

  print("Generating codes\n")
  train.codes0 = cor.compress(train.graph_corpus, train.n_sentences)

  print("Finetuning codes\n")
  threshold.tch({
    codes = train.codes0, -- NOTE: updated in-place
    pos = train.pos_enriched,
    neg = train.neg_enriched,
    n_sentences = train.n_sentences,
    n_hidden = HIDDEN,
    each = function (e, s)
      local d, dd = stopwatch()
      str.printf("Epoch: %3d  Time: %6.2f %6.2f  TCH Steps: %3d\n", e, d, dd, s)
    end
  })

  train.codes0 = mtx.raw_bitmap(train.codes0, train.n_sentences, HIDDEN)
  train.similarity0 = eval.encoding_similarity(
    train.codes0, train.pos, train.neg, train.n_sentences, HIDDEN)
  str.printi("AUC: %.2f#(auc) | F1: %.2f#(f1) | Precision: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", -- luacheck: ignore
    train.similarity0)

  train.entropy0 = eval.codebook_stats(train.codes0, train.n_sentences, HIDDEN)
  str.printi("Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)\n",
    train.entropy0)

  local top_v =
    TOP_ALGO == "chi2" and mtx.top_chi2(train.sentences, train.codes0, train.n_sentences, dataset.n_features, HIDDEN, TOP_K) or -- luacheck: ignore
    TOP_ALGO == "mi" and mtx.top_mi(train.sentences, train.codes0, train.n_sentences, dataset.n_features, HIDDEN, TOP_K) or (function () -- luacheck: ignore
      -- Fallback to all words
      local t = mtx.create(1, dataset.n_features)
      mtx.fill_indices(t)
      return t
    end)()

  local n_top_v = mtx.values(top_v)
  print("After top k filter", n_top_v)

  -- Show top words
  local words = tokenizer.index()
  for id in it.take(32, mtx.each(top_v)) do
    print(id, words[id + 1])
  end

  print("Re-encoding train/test with top features")
  tokenizer.restrict(top_v)
  train.sentences = tokenizer.tokenize(train.raw_sentences)
  test.sentences = tokenizer.tokenize(test.raw_sentences)

  print("Prepping for encoder")
  mtx.flip_interleave(train.sentences, train.n_sentences, n_top_v)
  mtx.flip_interleave(test.sentences, test.n_sentences, n_top_v)
  train.sentences = mtx.raw_bitmap(train.sentences, train.n_sentences, n_top_v * 2)
  test.sentences = mtx.raw_bitmap(test.sentences, test.n_sentences, n_top_v * 2)

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
    target = TARGET,
    specificity = SPECIFICITY,
    threads = THREADS,
  })

  print("Training encoder")
  stopwatch = utc.stopwatch()
  t.train({
    sentences = train.sentences,
    codes = train.codes0,
    samples = train.n_sentences,
    iterations = TM_ITERS,
    each = function (epoch)
      if epoch == TM_ITERS or epoch % EVALUATE_EVERY == 0 then
        train.codes1 = t.predict(train.sentences, train.n_sentences)
        test.codes1 = t.predict(test.sentences, test.n_sentences)
        train.accuracy0 = eval.encoding_accuracy(
          train.codes1, train.codes0, train.n_sentences, HIDDEN)
        train.similarity1 = eval.encoding_similarity(
          train.codes1, train.pos, train.neg, train.n_sentences, HIDDEN)
        test.similarity1 = eval.encoding_similarity(
          test.codes1, test.pos, test.neg, test.n_sentences, HIDDEN)
        print()
        str.printf("Epoch %3d  Time %3.2f %3.2f\n",
          epoch, stopwatch())
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

  do
    train.codes1 = t.predict(train.sentences, train.n_sentences)
    test.codes1 = t.predict(test.sentences, test.n_sentences)
    train.accuracy0 = eval.encoding_accuracy(
      train.codes1, train.codes0, train.n_sentences, HIDDEN)
    train.similarity1 = eval.encoding_similarity(
      train.codes1, train.pos, train.neg, train.n_sentences, HIDDEN)
    test.similarity1 = eval.encoding_similarity(
      test.codes1, test.pos, test.neg, test.n_sentences, HIDDEN)
    print()
    str.printi("  Train (acc) |           | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | F1 Spread: %.2f#(f1_min) %.2f#(f1_max) %.2f#(f1_std)", train.accuracy0) -- luacheck: ignore
    str.printi("  Codes (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity0) -- luacheck: ignore
    str.printi("  Train (sim) | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", train.similarity1) -- luacheck: ignore
    str.printi("  Test (sim)  | AUC: %.2f#(auc) | F1: %.2f#(f1) | Prec: %.2f#(precision) | Recall: %.2f#(recall) | Margin: %.2f#(margin)", test.similarity1) -- luacheck: ignore
  end

end)

