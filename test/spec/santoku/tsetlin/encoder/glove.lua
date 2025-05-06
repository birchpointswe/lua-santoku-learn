local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local utc = require("santoku.utc")
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

local THRESHOLD_LEVELS = 8
local TRAIN_TEST_RATIO = 0.8
local SIM_POS = 0.5
local SIM_NEG = 0.5

local HIDDEN = 64
local CLAUSES = 128
local TARGET = 0.5
local NEGATIVE = 0.1

local STATE = 8
local BOOST = true
local ACTIVE = 0.8
local SPECIFICITY = 3.9
local THREADS = nil

local EVALUATE_EVERY = 1
local MAX_RECORDS = 10000
local MAX_EPOCHS = 500

local function read_data (fp, max)

  local recs = {}

  local as = {}
  local bs = {}
  local ss = {}

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

    arr.push(recs, bm.create(bits, 2 * #thresholds * n_dims))

  end

  for i = 1, #recs do

    local i_n, i_p

    for _ = 1, #recs do
      local i0 = rand.fast_random() % #recs + 1
      if mtx.dot(ms[i], ms[i0]) < SIM_NEG then
        i_n = i0
        break
      end
    end

    if not i_n then
      break
    end

    for _ = 1, #recs do
      local i0 = rand.fast_random() % #recs + 1
      if mtx.dot(ms[i], ms[i0]) > SIM_POS then
        i_p = i0
        break
      end
    end

    if not i_p then
      break
    end

    as[#as + 1] = recs[i]
    bs[#bs + 1] = recs[i_n]
    ss[#ss + 1] = 0

    as[#as + 1] = recs[i]
    bs[#bs + 1] = recs[i_p]
    ss[#ss + 1] = 1

  end

  return {
    as = as,
    bs = bs,
    ss = ss,
    n_features = num.round(#thresholds * n_dims, tm.align),
    n_pairs = #as,
  }

end

local function split_dataset (dataset, s, e)
  local all_labels = bm.create()
  local all_pairs = {}
  for i = s, e do
    arr.push(all_pairs, dataset.as[i], dataset.bs[i])
    if dataset.ss[i] == 1 then
      bm.set(all_labels, i - s + 1)
    end
  end
  return
    bm.raw_matrix(all_pairs, dataset.n_features * 2),
    bm.raw(all_labels, #all_pairs)
end

test("tsetlin", function ()

  print("Reading data")
  local dataset = read_data(os.getenv("GLOVE") or "test/res/santoku/tsetlin/glove.txt", MAX_RECORDS)

  print("Shuffling")
  rand.seed()
  arr.shuffle(dataset.as, dataset.bs, dataset.ss)

  print("Splitting & packing")
  local n_train = num.floor(dataset.n_pairs * TRAIN_TEST_RATIO)
  local n_test = dataset.n_pairs - n_train
  local train_pairs, train_labels = split_dataset(dataset, 1, n_train)
  local test_pairs, test_labels = split_dataset(dataset, n_train + 1, n_train + n_test)

  print("Input Features", dataset.n_features * 2)
  print("Encoded Features", HIDDEN)
  print("Train", n_train)
  print("Test", n_test)

  print("Creating")
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

  print("Training")
  local stopwatch = utc.stopwatch()
  t.train({
    pairs = train_pairs,
    labels = train_labels,
    samples = n_train,
    active = ACTIVE,
    negative = NEGATIVE,
    iterations = MAX_EPOCHS,
    each = function (epoch)
      local duration, total = stopwatch()
      if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
        local train_score = t.evaluate({ pairs = train_pairs, labels = train_labels, samples = n_train })
        local test_score = t.evaluate({ pairs = test_pairs, labels = test_labels, samples = n_test })
        str.printf("Epoch %-4d   Time %6.3f   %6.3f   Test %4.2f   Train %4.2f\n",
          epoch, duration, total, test_score, train_score)
      else
        str.printf("Epoch %-4d   Time %6.3f   %6.3f\n",
          epoch, duration, total)
      end
    end
  })

  print()
  print("Persisting")
  fs.rm("model.bin", true)
  t.persist("model.bin")
  print("Restoring")
  t = tm.load("model.bin")
  local test_score = t.evaluate({ pairs = test_pairs, labels = test_labels, samples = n_test })
  local train_score = t.evaluate({ pairs = train_pairs, labels = train_labels, samples = n_train })
  str.printf("Evaluate Test %4.2f  Train %4.2f\n", test_score, train_score)

end)
