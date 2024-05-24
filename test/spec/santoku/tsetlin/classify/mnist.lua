local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local num = require("santoku.num")
local bm = require("santoku.bitmap")
local mtx = require("santoku.matrix")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local rand = require("santoku.random")

local CLASSES = 10
local FEATURES = 784
local TRAIN_TEST_RATIO = 0.5
local CLAUSES = 2000
local STATE_BITS = 8
local THRESHOLD = 50
local SPECIFICITY = { 10, 10, 1 }
local DROP_CLAUSE = 0.75
local BOOST_TRUE_POSITIVE = true
local EVALUATE_EVERY = 1
local MAX_EPOCHS = 400

local function read_data (fp, max)
  local problems = {}
  local solutions = {}
  local bits = {}
  for l in fs.lines(fp) do
    local n = 0
    for bit in str.gmatch(l, "%S+") do
      n = n + 1
      if n == FEATURES + 1 then
        solutions[#solutions + 1] = tonumber(bit)
        break
      end
      bit = bit == "1"
      if bit then
        bits[n] = true
        bits[n + FEATURES] = false
      else
        bits[n] = false
        bits[n + FEATURES] = true
      end
      if max and n > max then
        break
      end
    end
    if n ~= FEATURES + 1 then
      error("bitmap length mismatch")
    else
      problems[#problems + 1] = bm.create(bits, FEATURES * 2)
    end
  end
  return {
    problems = problems,
    solutions = solutions
  }
end

local function split_dataset (dataset, s, e)
  local ps, ss = {}, {}
  for i = s, e do
    arr.push(ps, dataset.problems[i])
    arr.push(ss, dataset.solutions[i])
  end
  local b = bm.raw_matrix(ps, FEATURES * 2)
  local m = mtx.create(1, #ss)
  mtx.set(m, 1, ss)
  return b, mtx.raw(m, 1, 1, "u32")
end

test("tsetlin", function ()

  local MAX = nil

  print("Reading data")
  local dataset = read_data("test/res/santoku/tsetlin/BinarizedMNISTData/MNISTTest.txt", MAX)

  -- print("Shuffling")
  -- rand.seed()
  -- arr.shuffle(dataset.problems, dataset.solutions)

  print("Splitting & packing")
  local n_train = num.floor(#dataset.problems * TRAIN_TEST_RATIO)
  local n_test = #dataset.problems - n_train
  local train_problems, train_solutions = split_dataset(dataset, 1, n_train)
  local test_problems, test_solutions = split_dataset(dataset, n_train + 1, n_train + n_test)

  print("Train", n_train)
  print("Test", n_test)

  print("Training")
  for SPEC = SPECIFICITY[1], SPECIFICITY[2], SPECIFICITY[3] do

    local t = tm.classifier(CLASSES, FEATURES, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

    for epoch = 1, MAX_EPOCHS do
      local start = os.time()
      tm.train(t, n_train, train_problems, train_solutions, SPEC, DROP_CLAUSE)
      local stop = os.time()
      local duration = stop - start
      if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
        local test_score =
          tm.evaluate(t, n_test, test_problems, test_solutions, epoch == MAX_EPOCHS)
        local train_score =
          tm.evaluate(t, n_train, train_problems, train_solutions)
        str.printf("Epoch %-4d  Spec %.2f Time %d  Test %4.2f  Train %4.2f\n",
          epoch, SPEC, duration, test_score, train_score)
      else
        str.printf("Epoch %-4d  Spec %.2f Time %d\n",
          epoch, SPEC, duration)
      end
    end

  end

end)
