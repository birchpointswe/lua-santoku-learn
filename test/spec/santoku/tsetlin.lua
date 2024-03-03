local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local bm = require("santoku.bitmap")
local fs = require("santoku.fs")
local it = require("santoku.iter")
local str = require("santoku.string")
local arr = require("santoku.array")
local rand = require("santoku.random")

local CLASSES = 2
local FEATURES = 384
local CLAUSES = 10
local STATE_BITS = 8
local THRESHOLD = 40
local SPECIFICITY = 3.9
local BOOST_TRUE_POSITIVE = false
local MAX_EPOCHS = 20

local function read_data (fp, max)
  local problems = {}
  local solutions = {}
  local records = it.map(function (l, s, e)
    return it.map(str.number, str.match(l, "%S+", false, s, e))
  end, fs.lines(fp))
  if max then
    records = it.take(max, records)
  end
  for bits in records do
    local b = bm.create()
    for i = 1, FEATURES do
      if bits() == 1 then
        bm.set(b, i)
      else
        -- NOTE: Setting the inverted features here. This is essential!
        -- TODO: Consider throwing an error if this is not done.
        bm.set(b, i + FEATURES)
      end
    end
    arr.push(problems, bm.raw(b, FEATURES * 2))
    arr.push(solutions, bits())
  end
  return problems, solutions
end

test("tsetlin", function ()

  local train_problems, train_solutions =
    read_data("test/res/santoku/tsetlin/NoisyXORTrainingData.txt")

  local test_problems, test_solutions =
    read_data("test/res/santoku/tsetlin/NoisyXORTestData.txt")

  rand.seed()
  arr.shuffle(train_problems, train_solutions)
  arr.shuffle(test_problems, test_solutions)

  local t = tm.create(CLASSES, FEATURES, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  for epoch = 1, MAX_EPOCHS do

    local start = os.clock()
    tm.train(t, train_problems, train_solutions, SPECIFICITY)
    local stop = os.clock()

    local test_score, confusion, predictions = tm.evaluate(t, test_problems, test_solutions, epoch == MAX_EPOCHS)
    local train_score = tm.evaluate(t, train_problems, train_solutions)

    str.printf("Epoch\t%-4d\tTest\t%4.2f\tTrain\t%4.2f\tTime\t%f\n", epoch, test_score, train_score, stop - start)

    if epoch == MAX_EPOCHS then

      if #confusion == 0 then
        print()
        print("  No confusion")
      else
        print()
        print("  Observed / Predicted  Ratio")
        print()
        for s in it.ivals(confusion) do
          str.printf("%10d / %-10d %-.2f\n", s.expected, s.predicted, s.ratio, s.count)
        end
      end

      print()
      print("  Observed / Predicted  Class Frequency")
      print()
      for p in it.ivals(predictions) do
        str.printf("    %.4f / %-6.4f     %s \n", p.frequency_observed, p.frequency_predicted, p.class)
      end

    end

  end

end)
