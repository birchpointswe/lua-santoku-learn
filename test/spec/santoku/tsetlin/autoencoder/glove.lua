local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local booleanizer = require("santoku.tsetlin.booleanizer")
local bm = require("santoku.bitmap")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local rand = require("santoku.random")
local num = require("santoku.num")
local err = require("santoku.error")

local ENCODED_BITS = 10
local THRESHOLD_LEVELS = 2
local TRAIN_TEST_RATIO = 0.5

local CLAUSES = 40
local STATE_BITS = 8
local THRESHOLD = 10
local SPECIFICITY = 2
local DROP_CLAUSE = 0.75
local BOOST_TRUE_POSITIVE = false

local MAX_RECORDS = 100
local MAX_EPOCHS = 10

local function read_data (fp, max)

  local problems = {}
  local bits = {}
  local observations = {}
  local n_dims = nil

  local lines = fs.lines(fp)

  if max then
    lines = it.take(max, lines)
  end

  local data = it.collect(it.map(function (l, s, e)

    local floats = it.collect(it.map(str.number, it.drop(1, str.matches(l, "%S+", false, s, e))))

    if n_dims == nil then
      n_dims = #floats
    elseif #floats ~= n_dims then
      err.error("mismatch in number of dimensions", #floats, n_dims)
    end

    for i = 1, #floats do
      observations[floats[i]] = true
    end

    return floats

  end, lines))

  local thresholds = booleanizer.thresholds(observations, THRESHOLD_LEVELS)

  for i = 1, #data do
    for j = 1, #data[i] do
      for k = 1, #thresholds do
        local t = thresholds[k]
        if data[i][j] <= t.value then
          bits[(j - 1) * #thresholds * 2 + t.bit] = true
          bits[(j - 1) * #thresholds * 2 + t.bit + #thresholds] = false
        else
          bits[(j - 1) * #thresholds * 2 + t.bit] = false
          bits[(j - 1) * #thresholds * 2 + t.bit + #thresholds] = true
        end
      end
    end
    arr.push(problems, bm.create(bits, 2 * #thresholds * n_dims))
  end

  return problems, #thresholds * 2 * n_dims

end

test("tsetlin", function ()

  print("Reading data")
  local data, n_features = read_data("test/res/santoku/tsetlin/glove.2500.txt", MAX_RECORDS)

  print("Shuffling")
  rand.seed()
  arr.shuffle(data)

  print("Splitting & packing")
  local n_train = num.floor(#data * TRAIN_TEST_RATIO)
  local n_test = #data - n_train
  local train = bm.raw_matrix(data, n_features * 2, 1, n_train)
  local test = bm.raw_matrix(data, n_features * 2, n_train + 1, #data)

  print("Features", n_features)
  print("Train", n_train)
  print("Test", n_test)

  local t = tm.autoencoder(ENCODED_BITS, n_features, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  local times = {}

  print("Training")
  for epoch = 1, MAX_EPOCHS do

    local start = os.clock()
    tm.train(t, n_train, train, SPECIFICITY, DROP_CLAUSE)
    local stop = os.clock()
    arr.push(times, stop - start)
    local avg_duration = arr.mean(times)

    local test_score = tm.evaluate(t, n_test, test)
    local train_score = tm.evaluate(t, n_train, train)

    str.printf("Epoch\t%-4d\tTest\t%4.2f\tTrain\t%4.2f\tTime\t%f\n", epoch, test_score, train_score, avg_duration)

  end

end)
