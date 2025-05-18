local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local utc = require("santoku.utc")
local eval = require("santoku.tsetlin.evaluator")
local tm = require("santoku.tsetlin")
local num = require("santoku.num")
local mtx = require("santoku.matrix.integer")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")

local TTR = 0.9
local THREADS = nil
local EVALUATE_EVERY = 1
local ITERATIONS = 20

local CLASSES = 10
local CLAUSES = 1024
local TARGET = 0.1
local SPECIFICITY = 10
local NEGATIVE = 0.1

local FEATURES = 784

local function read_data (fp, skip, max)
  local problems = {}
  local solutions = {}
  local bits = {}
  local skip = skip or 0
  for l in fs.lines(fp) do
    if skip > 0 then
      skip = skip - 1
    else
      arr.clear(bits)
      local n = 0
      for bit in str.gmatch(l, "%S+") do
        if n == FEATURES then
          local s = tonumber(bit)
          solutions[#solutions + 1] = s
          break
        end
        bit = bit == "1"
        if bit then
          arr.push(bits, n)
        end
        n = n + 1
      end
      if n ~= FEATURES then
        error("bitmap length mismatch")
      else
        local p = mtx.create(bits)
        mtx.reshape(p, mtx.values(p), 1);
        problems[#problems + 1] = p
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
  local ps = mtx.create(0, 1)
  local ss = {}
  for i = s, e do
    local p = mtx.create(dataset.problems[i])
    mtx.add(p, (i - s) * FEATURES)
    mtx.extend(ps, p)
    arr.push(ss, dataset.solutions[i])
  end
  mtx.flip_interleave(ps, e - s + 1, FEATURES)
  ps = mtx.raw_bitmap(ps, e - s + 1, FEATURES * 2)
  ss = mtx.create(ss)
  ss = mtx.raw(ss, 1, 1, "u32")
  return ps, ss
end

test("tsetlin", function ()

  local SKIP = 0
  local MAX = 10000

  print("Reading data")
  local dataset = read_data("test/res/santoku/tsetlin/BinarizedMNISTData/MNISTTraining.txt", SKIP, MAX)

  print("Splitting & packing")
  local n_train = num.floor(#dataset.problems * TTR)
  local n_test = #dataset.problems - n_train
  FEATURES = num.round(FEATURES, tm.align)
  local train_problems, train_solutions = split_dataset(dataset, 1, n_train)
  local test_problems, test_solutions = split_dataset(dataset, n_train + 1, n_train + n_test)
  str.printf("Train %d  Test %d\n", n_train, n_test)

  print("Train", n_train)
  print("Test", n_test)

  print("Creating")
  local t = tm.classifier({
    features = FEATURES,
    classes = CLASSES,
    clauses = CLAUSES,
    target = TARGET,
    specificity = SPECIFICITY,
    threads = THREADS,
  })

  print("Training")
  local stopwatch = utc.stopwatch()
  t.train({
    samples = n_train,
    problems = train_problems,
    solutions = train_solutions,
    iterations = ITERATIONS,
    negative = NEGATIVE,
    each = function (epoch)
      local train_pred = t.predict(train_problems, n_train)
      local test_pred = t.predict(test_problems, n_test)
      local duration = stopwatch()
      if epoch == ITERATIONS or epoch % EVALUATE_EVERY == 0 then
        local train_stats = eval.class_accuracy(train_pred, train_solutions, CLASSES, n_train)
        local test_stats = eval.class_accuracy(test_pred, test_solutions, CLASSES, n_test)
        str.printf("Epoch %-4d  Time %4.2f  Test %4.2f  Train %4.2f\n",
          epoch, duration, test_stats.f1, train_stats.f1)
      else
        str.printf("Epoch %-4d  Time %4.2f\n",
          epoch, duration)
      end
    end
  })

  print()
  print("Persisting")
  fs.rm("model.bin", true)
  t.persist("model.bin", true)

  print("Testing restore")
  t = tm.load("model.bin", nil, true)
  local train_pred = t.predict(train_problems, n_train)
  local test_pred = t.predict(test_problems, n_test)
  local train_stats = eval.class_accuracy(train_pred, train_solutions, CLASSES, n_train)
  local test_stats = eval.class_accuracy(test_pred, test_solutions, CLASSES, n_test)
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_stats.f1, train_stats.f1)

end)
