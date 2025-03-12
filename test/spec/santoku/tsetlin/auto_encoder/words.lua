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

local MAX_RECORDS = 1000
local MAX_VOCAB = 512

local ENCODED_BITS = 64
local TRAIN_TEST_RATIO = 0.8

local CLAUSES = 1024
local STATE_BITS = 8
local THRESHOLD = 256
local BOOST_TRUE_POSITIVE = false
local ACTIVE_CLAUSE = 0.85
local LOSS_ALPHA = 0.5
local SPECIFICITY_LOW = 2
local SPECIFICITY_HIGH = 200

local EVALUATE_EVERY = 1
local MAX_EPOCHS = 400

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
  local data, n_features = read_data(os.getenv("CORPUS") or "test/res/santoku/tsetlin/corpus.txt", MAX_RECORDS, MAX_VOCAB)

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

  local t = tm.auto_encoder(
    ENCODED_BITS, n_features, CLAUSES,
    STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE,
    SPECIFICITY_LOW, SPECIFICITY_HIGH)

  print("Training")
  for epoch = 1, MAX_EPOCHS do

    local start = os.time()
    tm.train(t, n_train, train, ACTIVE_CLAUSE, LOSS_ALPHA)
    local duration = os.time() - start

    local total_weight_n = (#data - (n_train + 1)) * ENCODED_BITS
    local total_weight = 0
    for i = n_train + 1, #data do
      local x = bm.from_raw(tm.predict(t, bm.raw(data[i])), ENCODED_BITS)
      total_weight = total_weight + bm.cardinality(x)
    end

    if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
      local test_score = tm.evaluate(t, n_test, test)
      local train_score = tm.evaluate(t, n_train, train)
      str.printf("Epoch  %-4d  Time  %d  Test  %4.4f  Train  %4.4f  Avg. Weight  %4.4f\n",
        epoch, duration, test_score, train_score, total_weight / total_weight_n)
    else
      str.printf("Epoch  %-4d  Time  %d\n",
        epoch, duration)
    end

  end

end)
