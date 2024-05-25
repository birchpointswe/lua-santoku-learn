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

local ENCODED_BITS = 128
local THRESHOLD_LEVELS = 2
local TRAIN_TEST_RATIO = 0.2

local CLAUSES = 80
local STATE_BITS = 8
local THRESHOLD = 200
local SPECIFICITY = { 1.6, 1.6, 0.1 }
local LOSS_ALPHA = 1
local ACTIVE_CLAUSE = 0.75
local BOOST_TRUE_POSITIVE = false

local EVALUATE_EVERY = 5
local MAX_RECORDS = 1000
local MAX_EPOCHS = 400

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

  return problems, #thresholds * n_dims

end

test("tsetlin", function ()

  print("Reading data")
  local data, n_features = read_data(os.getenv("GLOVE") or "test/res/santoku/tsetlin/glove.txt", MAX_RECORDS)

  print("Shuffling")
  rand.seed()
  arr.shuffle(data)

  print("Splitting & packing")
  local n_train = num.floor(#data * TRAIN_TEST_RATIO)
  local n_test = #data - n_train
  local train = bm.raw_matrix(data, n_features * 2, 1, n_train)
  local test = bm.raw_matrix(data, n_features * 2, n_train + 1, #data)

  print("Input Features", n_features * 2)
  print("Encoded Features", ENCODED_BITS)
  print("Train", n_train)
  print("Test", n_test)

  local t = tm.auto_encoder(ENCODED_BITS, n_features, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  print("Training")
  for SPEC = SPECIFICITY[1], SPECIFICITY[2], SPECIFICITY[3] do

    for epoch = 1, MAX_EPOCHS do

      local start = os.time()
      tm.train(t, n_train, train, SPEC, ACTIVE_CLAUSE, LOSS_ALPHA)
      local duration = os.time() - start

      if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
        local test_score = tm.evaluate(t, n_test, test)
        local train_score = tm.evaluate(t, n_train, train)
        str.printf("Epoch  %-4d  Spec %.2f Time  %d  Test  %4.2f  Train  %4.2f\n",
          epoch, SPEC, duration, test_score, train_score)
      else
        str.printf("Epoch  %-4d  Spec %.2f Time  %d\n",
          epoch, SPEC, duration)
      end

    end

  end

end)
