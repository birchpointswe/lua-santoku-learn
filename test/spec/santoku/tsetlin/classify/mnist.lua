local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local num = require("santoku.num")
local bm = require("santoku.bitmap")
local mtx = require("santoku.matrix")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")

local CLASSES = 10
local FEATURES = 784
local TRAIN_TEST_RATIO = 0.5
local CLAUSES = 2000
local STATE_BITS = 8
local THRESHOLD = 50
local BOOST_TRUE_POSITIVE = true
local ACTIVE_CLAUSE = 1 --0.75
local SPECIFICITY = 10
local EVALUATE_EVERY = 1
local MAX_EPOCHS = 20

local function read_data (fp, skip, max)
  local problems = {}
  local solutions = {}
  local bits = {}
  local skip = skip or 0
  for l in fs.lines(fp) do
    if skip > 0 then
      skip = skip - 1
    else
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
          bits[n + FEATURES] = nil
        else
          bits[n] = nil
          bits[n + FEATURES] = true
        end
      end
      if n ~= FEATURES + 1 then
        error("bitmap length mismatch")
      else
        problems[#problems + 1] = bm.create(bits, FEATURES * 2)
      end
      if max and #problems >= max then
        break
      end
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

  local SKIP = 0
  local MAX = 10000

  print("Reading data")
  local dataset = read_data("test/res/santoku/tsetlin/BinarizedMNISTData/MNISTTest.txt", SKIP, MAX)

  print("Splitting & packing")
  local n_train = num.floor(#dataset.problems * TRAIN_TEST_RATIO)
  local n_test = #dataset.problems - n_train
  local train_problems, train_solutions = split_dataset(dataset, 1, n_train)
  local test_problems, test_solutions = split_dataset(dataset, n_train + 1, n_train + n_test)

  print("Train", n_train)
  print("Test", n_test)

  print("Training")
  local t = tm.classifier(CLASSES, FEATURES, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  for epoch = 1, MAX_EPOCHS do
    local start = os.time()
    tm.train(t, n_train, train_problems, train_solutions, ACTIVE_CLAUSE, SPECIFICITY)
    local stop = os.time()
    local duration = stop - start
    if epoch == MAX_EPOCHS or epoch % EVALUATE_EVERY == 0 then
      local test_score =
        tm.evaluate(t, n_test, test_problems, test_solutions, epoch == MAX_EPOCHS)
      local train_score =
        tm.evaluate(t, n_train, train_problems, train_solutions)
      str.printf("Epoch %-4d  Time %d  Test %4.2f  Train %4.2f\n",
        epoch, duration, test_score, train_score)
    else
      str.printf("Epoch %-4d  Time %d\n",
        epoch, duration)
    end
  end

end)
