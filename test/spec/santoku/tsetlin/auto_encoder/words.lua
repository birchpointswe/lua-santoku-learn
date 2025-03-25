local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")
local bm = require("santoku.bitmap")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local rand = require("santoku.random")
local num = require("santoku.num")

local MAX_RECORDS = 200
local MAX_VOCAB = 512

local ENCODED_BITS = 32
local TRAIN_TEST_RATIO = 0.8

local CLAUSES = 1024
local STATE_BITS = 8
local TARGET = 256
local BOOST_TRUE_POSITIVE = false
local ACTIVE_CLAUSE = 0.85
local LOSS_ALPHA = 0.5
local SPECIFICITY_LOW = 2
local SPECIFICITY_HIGH = 200

local EVALUATE_EVERY = 1
local MAX_EPOCHS = 10

local function read_data (fp, max, max_vocab)

  local next_id = 1
  local ids = {}
  local problems = {}

  for line in fs.lines(fp) do
    line = str.lower(line)
    line = str.gsub(line, "[%s%p]+", " ")
    line = str.gsub(line, "[^a-zA-Z0-9 ]", "")
    local p = {}
    arr.push(problems, p)
    for word in str.gmatch(line, "%S+") do
      local id = ids[word]
      if not id and (not max_vocab or next_id <= max_vocab) then
        id = next_id
        ids[word] = id
        next_id = next_id + 1
      end
      if id then
        p[id] = true
      end
    end
    if max and #problems > max then
      break
    end
  end

  for i = 1, #problems do
    problems[i] = bm.create(problems[i], next_id - 1)
  end

  return problems, next_id - 1

end

test("tsetlin", function ()

  print("Reading data")
  local data, n_features =
    read_data(os.getenv("CORPUS") or "test/res/santoku/tsetlin/corpus.txt",
      MAX_RECORDS, MAX_VOCAB)

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
