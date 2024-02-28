local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local bm = require("santoku.bitmap")
local fs = require("santoku.fs")
local it = require("santoku.iter")
local str = require("santoku.string")
local arr = require("santoku.array")

local CLASSES = 2
local FEATURES = 12
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

  local t = tm.create(CLASSES, FEATURES, CLAUSES, STATE_BITS, THRESHOLD, BOOST_TRUE_POSITIVE)

  for epoch = 1, MAX_EPOCHS do

    tm.train(t, train_problems, train_solutions, SPECIFICITY)

    local test_score --[[, cfx]] = tm.evaluate(t, test_problems, test_solutions, true)
    local train_score = tm.evaluate(t, train_problems, train_solutions)

    str.printf("Epoch\t%-4d\tTest\t%4.2f\tTrain\t%4.2f\n", epoch, test_score, train_score)

    -- print()
    -- for i = 0, CLASSES - 1 do
    --   for j = 0, CLASSES - 1 do
    --     if cfx[i] and cfx[i][j] then
    --       str.printf("%4d\t", cfx[i][j])
    --     else
    --       str.printf("%4d\t", 0)
    --     end
    --   end
    --   print()
    -- end
    -- print()

  end

end)
