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

local ENCODED_BITS = 256
local THRESHOLD_LEVELS = 10
local TRAIN_TEST_RATIO = 0.5
local MARGIN = 0.1
local SIMILARITY_CUTOFF = 0.5
local DISTANCE_CUTOFF = 0.5

local CLAUSES = 80
local STATE_BITS = 8
local THRESHOLD = 160
local SPECIFICITY = 2
local LOSS_SCALE = 0.5
local LOSS_SCALE_MIN = 0
local LOSS_SCALE_MAX = 1
local DROP_CLAUSE = 0.5
local BOOST_TRUE_POSITIVE = false

local EVALUATE_EVERY = 10
local MAX_RECORDS = nil
local MAX_EPOCHS = 5

local function read_data (fp, max)

  local recs = {}

  local as = {}
  local ns = {}
  local ps = {}

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
      if mtx.dot(ms[i], ms[i0]) < DISTANCE_CUTOFF then
        i_n = i0
        break
      end
    end

    if not i_n then
      break
    end

    for _ = 1, #recs do
      local i0 = rand.fast_random() % #recs + 1
      if mtx.dot(ms[i], ms[i0]) > SIMILARITY_CUTOFF then
        i_p = i0
        break
      end
    end

    if not i_p then
      break
    end

    as[#as + 1] = recs[i]
    ns[#as] = recs[i_n]
    ps[#as] = recs[i_p]

  end

  return {
    as = as,
    ns = ns,
    ps = ps,
    n_features = #thresholds * n_dims,
    n_triplets = #as,
  }

end

local function split_dataset (dataset, s, e)
  local as = bm.raw_matrix(dataset.as, dataset.n_features * 2, s, e)
  local ns = bm.raw_matrix(dataset.ns, dataset.n_features * 2, s, e)
  local ps = bm.raw_matrix(dataset.ps, dataset.n_features * 2, s, e)
  return as, ns, ps
end

test("tsetlin", function ()

  print("Reading data")
  local dataset = read_data("test/res/santoku/tsetlin/glove.2500.txt", MAX_RECORDS)

  print("Shuffling")
  rand.seed()
  arr.shuffle(dataset.as, dataset.ns, dataset.ps)

  print("Splitting & packing")
  local n_train = num.floor(dataset.n_triplets * TRAIN_TEST_RATIO)
  local n_test = dataset.n_triplets - n_train
  local train_as, train_ns, train_ps = split_dataset(dataset, 1, n_train)
  local test_as, test_ns, test_ps = split_dataset(dataset, n_train + 1, n_train + n_test)

  print("Input Features", dataset.n_features * 2)
  print("Encoded Features", ENCODED_BITS)
  print("Train", n_train)
  print("Test", n_test)

  local t = tm.encoder(ENCODED_BITS, dataset.n_features, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  print("Training")
  for epoch = 1, MAX_EPOCHS do

    local start = os.time()
    tm.train(t, n_train, train_as, train_ns, train_ps,
      SPECIFICITY, DROP_CLAUSE, MARGIN,
      LOSS_SCALE, LOSS_SCALE_MIN, LOSS_SCALE_MAX)
    local duration = os.time() - start

    if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
      local test_score = tm.evaluate(t, n_test, test_as, test_ns, test_ps, MARGIN)
      local train_score = tm.evaluate(t, n_train, train_as, train_ns, train_ps, MARGIN)
      str.printf("Epoch %-4d  Time %d  Test %4.2f  Train %4.2f\n",
        epoch, duration, test_score, train_score)
    else
      str.printf("Epoch %-4d  Time %d\n",
        epoch, duration)
    end

  end

end)
