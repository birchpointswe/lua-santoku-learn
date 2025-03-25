local serialize = require("santoku.serialize") -- luacheck: ignore
local utc = require("santoku.utc")
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

local ENCODED_BITS = 32
local THRESHOLD_LEVELS = 2
local TRAIN_TEST_RATIO = 0.8

local CLAUSES = 512
local STATE_BITS = 8
local TARGET = 256
local BOOST_TRUE_POSITIVE = false
local ACTIVE_CLAUSE = 0.85
local LOSS_ALPHA = 0.5
local SPECIFICITY_LOW = 2
local SPECIFICITY_HIGH = 200

local EVALUATE_EVERY = 5
local MAX_RECORDS = 200
local MAX_EPOCHS = 20

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

  local t = tm.auto_encoder({
    visible = n_features,
    hidden = ENCODED_BITS,
    clauses = CLAUSES,
    state_bits = STATE_BITS,
    target = TARGET,
    boost_true_positive = BOOST_TRUE_POSITIVE,
    spec_low = SPECIFICITY_LOW,
    spec_high = SPECIFICITY_HIGH,
    threads = 4
  })

  local stopwatch = utc.stopwatch()
  print("Training")
  t.train({
    corpus = train,
    samples = n_train,
    active_clause = ACTIVE_CLAUSE,
    loss_alpha = LOSS_ALPHA,
    iterations = MAX_EPOCHS,
    each = function (epoch)
      local duration, avg_duration = stopwatch(0.1)
      if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
        local test_score = t.evaluate({ corpus = test, samples = n_test })
        local train_score = t.evaluate({ corpus = train, samples = n_train })
        str.printf("Epoch %-4d  Time %6.3f (%6.3f)  Test %4.2f  Train %4.2f\n",
          epoch, duration, avg_duration, test_score, train_score)
      else
        str.printf("Epoch %-4d  Time %6.3f (%6.3f)\n",
          epoch, duration, avg_duration)
      end
    end
  })

end)
