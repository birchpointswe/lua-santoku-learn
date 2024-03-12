local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local bm = require("santoku.bitmap")
local mtx = require("santoku.matrix")
local str = require("santoku.string")
local arr = require("santoku.array")
local rand = require("santoku.random")

local CLASSES = 2
local FEATURES = 1214
local CLAUSES = 200
local STATE_BITS = 8
local THRESHOLD = 40
local SPECIFICITY = 3.9
local DROP_CLAUSE = 0.85
local BOOST_TRUE_POSITIVE = false
local MAX_EPOCHS = 10

local function read_data (fp, max)
  local problems = {}
  local solutions = {}
  local bits = {}
  for l in io.lines(fp) do
    local n = 0
    for bit in l:gmatch("%S+") do
      n = n + 1
      bit = bit == "1"
      if n == FEATURES + 1 then
        solutions[#solutions + 1] = bit and 1 or 0
        break
      elseif bit then
        bits[n] = true
        bits[n + FEATURES] = nil
      else
        bits[n] = nil
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
  return problems, solutions
end

local function pack_data (ps, ss)
  local b = bm.raw_matrix(ps, FEATURES * 2)
  -- TODO: Simply `local m = mtx.raw(ss)` would be nice
  local m = mtx.create(1, #ss)
  mtx.set(m, 1, ss)
  return b, mtx.raw(m, 1, 1, "u32")
end

test("tsetlin", function ()

  local MAX = nil

  print("Reading data")
  local train_problems, train_solutions =
    read_data("test/res/santoku/tsetlin/NoisyXORTrainingData.txt", MAX)
  local test_problems, test_solutions =
    read_data("test/res/santoku/tsetlin/NoisyXORTestData.txt", MAX)

  print("Shuffling")
  rand.seed()
  arr.shuffle(train_problems, train_solutions)
  arr.shuffle(test_problems, test_solutions)

  print("Packing data")
  local train_problems_packed, train_solutions_packed = pack_data(train_problems, train_solutions)
  local test_problems_packed, test_solutions_packed = pack_data(test_problems, test_solutions)

  local t = tm.classifier(CLASSES, FEATURES, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  local times = {}

  print("Training")
  for epoch = 1, MAX_EPOCHS do

    local start = os.clock()
    tm.train(t, #train_problems, train_problems_packed, train_solutions_packed, SPECIFICITY, DROP_CLAUSE)
    local stop = os.clock()
    arr.push(times, stop - start)
    local avg_duration = arr.mean(times)

    local test_score, confusion, predictions =
      tm.evaluate(t, #test_problems, test_problems_packed, test_solutions_packed, epoch == MAX_EPOCHS)

    local train_score =
      tm.evaluate(t, #train_problems, train_problems_packed, train_solutions_packed)

    str.printf("Epoch\t%-4d\tTest\t%4.2f\tTrain\t%4.2f\tTime\t%f\n", epoch, test_score, train_score, avg_duration)

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

end)
