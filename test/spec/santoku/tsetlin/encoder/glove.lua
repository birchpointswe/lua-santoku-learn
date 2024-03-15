-- Step 1: Train a TM to convert a word embedding vector to a bitmap where
-- hamming distance approximates cosine similarity.
--
-- Dataset & training:
-- - Randomly sample a subset of vector pairs and calculate cosine similarity
-- - Run each vector through and calculate hamming distance
-- - Compare the hamming distance of the bitmaps to the cosine similarity of the
--   vectors
-- - If too high, reinforce outputting 0s
-- - If too low, reinforce outputting 1s
--
-- Testing:
-- - Convert all word vectors to bitmaps, building a hash table from word to
--   bitmap (many words will have a hamming distance of zero, allowing for
--   re-use of bitmaps)
-- - Evaluate sentence similarity using the sts-benchmark dataset, representing
--   sentences as a word-bitmap bloom filter and similarity as hamming distance

-- Step 2: Train a TM to add semantic understanding to the sentence bloom
-- filters using the sts-benchmark dataset
--
-- Datset & training:
-- - Convert the [sentence, sentence, score] records in the sts-benchmark
--   dataset to [sentence-bloom, sentence-bloom, score]
-- - Perform the same training as in Step 1, this time comparing hamming
--   distance to the similarity score
--
-- Usage:
-- - Evaluate sentence similarity as in Step 1, representing sentences as
--   word-bitmap bloom filters that have been processed by this second TM

local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local booleanizer = require("santoku.tsetlin.booleanizer")
local bm = require("santoku.bitmap")
local mtx = require("santoku.matrix")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local rand = require("santoku.random")
local num = require("santoku.num")
local err = require("santoku.error")

local ENCODED_BITS = 64
local THRESHOLD_LEVELS = 10
local TRAIN_TEST_RATIO = 0.5

local CLAUSES = 40
local STATE_BITS = 8
local THRESHOLD = 80
local SPECIFICITY = 3
local DROP_CLAUSE = 0.75
local BOOST_TRUE_POSITIVE = true

local EVALUATE_EVERY = 5
local MAX_RECORDS = nil
local MAX_EPOCHS = 1000

local function read_data (fp, max)

  local as = {}
  local bs = {}
  local scores = {}

  local bits = {}
  local floats = {}
  local observations = {}
  local n_dims = nil

  local lines = fs.lines(fp)

  if max then
    lines = it.take(max, lines)
  end

  local ms = it.collect(it.map(function (l, s, e)

    it.collect(it.map(str.number, it.drop(1, str.matches(l, "%S+", false, s, e))), floats, 1)

    if n_dims == nil then
      n_dims = #floats
    elseif #floats ~= n_dims then
      err.error("mismatch in number of dimensions", #floats, n_dims)
    end

    for i = 1, #floats do
      observations[floats[i]] = true
    end

    local m = mtx.create(1, n_dims)
    mtx.set(m, 1, floats)

    return m

  end, lines))

  local thresholds = booleanizer.thresholds(observations, THRESHOLD_LEVELS)

  for i = 1, #ms do
    mtx.normalize(ms[i])
    for j = 1, mtx.columns(ms[i]) do
      for k = 1, #thresholds do
        local t = thresholds[k]
        local v = mtx.get(ms[i], 1, j)
        if v <= t.value then
          bits[(j - 1) * #thresholds * 2 + t.bit] = true
          bits[(j - 1) * #thresholds * 2 + t.bit + #thresholds] = false
        else
          bits[(j - 1) * #thresholds * 2 + t.bit] = false
          bits[(j - 1) * #thresholds * 2 + t.bit + #thresholds] = true
        end
      end
    end
    arr.push(as, bm.create(bits, 2 * #thresholds * n_dims))
  end

  for i = 1, #as do
    local i0 = rand.fast_random() % #as + 1
    bs[i] = as[i0]
    scores[i] = mtx.dot(ms[i], ms[i0])
  end

  return {
    as = as,
    bs = bs,
    scores = scores,
    n_features = #thresholds * n_dims,
    n_pairs = #as,
  }

end

local function split_dataset (dataset, s, e)
  local as = bm.raw_matrix(dataset.as, dataset.n_features * 2, s, e)
  local bs = bm.raw_matrix(dataset.bs, dataset.n_features * 2, s, e)
  local scores = mtx.raw(mtx.create(dataset.scores, s, e))
  return as, bs, scores
end

test("tsetlin", function ()

  print("Reading data")
  local dataset = read_data("test/res/santoku/tsetlin/glove.2500.txt", MAX_RECORDS)

  print("Shuffling")
  rand.seed()
  arr.shuffle(dataset.as, dataset.bs, dataset.scores)

  print("Splitting & packing")
  local n_train = num.floor(dataset.n_pairs * TRAIN_TEST_RATIO)
  local n_test = dataset.n_pairs - n_train
  local train_as, train_bs, train_scores = split_dataset(dataset, 1, n_train)
  local test_as, test_bs, test_scores = split_dataset(dataset, n_train + 1, n_train + n_test)

  print("Input Features", dataset.n_features * 2)
  print("Encoded Features", ENCODED_BITS)
  print("Train", n_train)
  print("Test", n_test)

  local t = tm.encoder(ENCODED_BITS, dataset.n_features, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  print("Training")
  for epoch = 1, MAX_EPOCHS do

    local start = os.clock()
    tm.train(t, n_train, train_as, train_bs, train_scores, SPECIFICITY, DROP_CLAUSE)
    local duration = os.clock() - start

    if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
      local test_score, nh, nl = tm.evaluate(t, n_test, test_as, test_bs, test_scores)
      local train_score = tm.evaluate(t, n_train, train_as, train_bs, train_scores)
      str.printf("Epoch\t%-4d\tTime\t%f\tTest\t%4.2f\tTrain\t%4.2f\tHigh\t%d\tLow\t%d\n", epoch, duration, test_score, train_score, nh, nl)
    else
      str.printf("Epoch\t%-4d\tTime\t%f\n", epoch, duration)
    end

  end

end)
