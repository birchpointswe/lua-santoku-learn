local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local num = require("santoku.num")
local tm = require("santoku.tsetlin")
local bm = require("santoku.bitmap")
local mtx = require("santoku.matrix")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")

local CLASSES = 2
local FEATURES = 12
local TRAIN_TEST_RATIO = 0.5
local CLAUSES = 80
local STATE_BITS = 8
local THRESHOLD = 200
local SPECIFICITY = 3.9
local ACTIVE_CLAUSE = 1
local BOOST_TRUE_POSITIVE = false
local MAX_EPOCHS = 20
local MAX_RECORDS = nil

local function read_data (fp, max)
  local problems = {}
  local solutions = {}
  for l in fs.lines(fp) do
    local b0 = bm.create()
    local n = 0
    for bit in str.gmatch(l, "%S+") do
      n = n + 1
      if n == FEATURES + 1 then
        solutions[#solutions + 1] = tonumber(bit)
        break
      end
      bit = bit == "1"
      if bit then
        bm.set(b0, n)
      else
        bm.set(b0, n + FEATURES)
      end
    end
    if n ~= FEATURES + 1 then
      error("bitmap length mismatch")
    else
      problems[#problems + 1] = b0
    end
    if max and #problems >= max then
      break
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

  print("Reading data")
  local dataset = read_data("test/res/santoku/tsetlin/NoisyXORTrainingData.txt", MAX_RECORDS)

  print("Splitting & packing")
  local n_train = num.floor(#dataset.problems * TRAIN_TEST_RATIO)
  local n_test = #dataset.problems - n_train
  local train_problems, train_solutions = split_dataset(dataset, 1, n_train)
  local test_problems, test_solutions = split_dataset(dataset, n_train + 1, n_train + n_test)

  print("Train", n_train)
  print("Test", n_test)

  local t = tm.classifier(CLASSES, FEATURES, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  local times = {}

  print("Training")
  for epoch = 1, MAX_EPOCHS do

    local start = os.time()
    tm.train(t, n_train, train_problems, train_solutions, SPECIFICITY, ACTIVE_CLAUSE)
    local stop = os.time()
    arr.push(times, stop - start)
    local avg_duration = arr.mean(times)

    local test_score, confusion, predictions =
      tm.evaluate(t, n_test, test_problems, test_solutions, epoch == MAX_EPOCHS)

    local train_score =
      tm.evaluate(t, n_train, train_problems, train_solutions)

    str.printf("Epoch\t%-4d\tTest\t%4.2f\tTrain\t%4.2f\tTime\t%d\n", epoch, test_score, train_score, avg_duration)

    if epoch == MAX_EPOCHS then

      if #confusion == 0 then
        print()
        print("  No confusion")
      else
        print()
        print("  Observed / Predicted  Ratio")
        print()
        for i = 1, #confusion do
          local s = confusion[i]
          str.printf("%10d / %-10d %-.4f\n", s.expected, s.predicted, s.ratio, s.count)
        end
      end

      print()
      print("  Observed / Predicted  Class Frequency")
      print()
      for i = 1, #predictions do
        local p = predictions[i]
        str.printf("    %.4f / %-6.4f     %s \n", p.frequency_observed, p.frequency_predicted, p.class)
      end

    end

  end

  --print("Persisting")
  --fs.rm("model.bin", true)
  --tm.persist(t, "model.bin")

  --print("Testing restore")
  --t = tm.load("model.bin")
  --local test_score =
  --  tm.evaluate(t, #test_problems, test_problems, test_solutions)
  --local train_score =
  --  tm.evaluate(t, #train_problems, train_problems, train_solutions)
  --str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_score, train_score)


end)
