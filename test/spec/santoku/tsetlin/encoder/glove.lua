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

local ENCODED_BITS = 128
local THRESHOLD_LEVELS = 20
local TRAIN_TEST_RATIO = 0.8
local SIM_POS = 0.7
local SIM_NEG = 0.5

local CLAUSES = 512
local STATE_BITS = 8
local THRESHOLD = 512
local BOOST_TRUE_POSITIVE = false
local ACTIVE_CLAUSE = 0.75
local MARGIN = 0.05
local LOSS_ALPHA = 0

local EVALUATE_EVERY = 1
local MAX_RECORDS = 1000
local MAX_EPOCHS = 200

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
  local tokens = {}
  for i = s, e do
    arr.push(tokens, dataset.as[i], dataset.ns[i], dataset.ps[i])
  end
  return bm.raw_matrix(tokens, dataset.n_features * 2)
end

test("tsetlin", function ()

  print("Reading data")
  local dataset = read_data(os.getenv("GLOVE") or "test/res/santoku/tsetlin/glove.txt", MAX_RECORDS)

  print("Shuffling")
  rand.seed()
  arr.shuffle(dataset.as, dataset.ns, dataset.ps)

  print("Splitting & packing")
  local n_train = num.floor(dataset.n_triplets * TRAIN_TEST_RATIO)
  local n_test = dataset.n_triplets - n_train
  local train_tokens = split_dataset(dataset, 1, n_train)
  local test_tokens = split_dataset(dataset, n_train + 1, n_train + n_test)

  print("Input Features", dataset.n_features * 2)
  print("Encoded Features", ENCODED_BITS)
  print("Train", n_train)
  print("Test", n_test)

  print("Training")
  local t = tm.encoder(ENCODED_BITS, dataset.n_features, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  for epoch = 1, MAX_EPOCHS do

    local start = os.time()
    tm.train(t, n_train, train_tokens, ACTIVE_CLAUSE, MARGIN, LOSS_ALPHA)
    local duration = os.time() - start

    if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
      local test_score = tm.evaluate(t, n_test, test_tokens, MARGIN)
      local train_score = tm.evaluate(t, n_train, train_tokens, MARGIN)
      str.printf("Epoch %-4d  Time %d  Test %4.2f  Train %4.2f\n",
        epoch, duration, test_score, train_score)
    else
      str.printf("Epoch %-4d  Time %d\n",
        epoch, duration)
    end

  end

end)
