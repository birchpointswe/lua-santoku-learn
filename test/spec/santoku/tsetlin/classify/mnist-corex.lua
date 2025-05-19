local arr = require("santoku.array")
local corex = require("santoku.corex")
local eval = require("santoku.tsetlin.evaluator")
local fs = require("santoku.fs")
local mtx = require("santoku.matrix.integer")
local num = require("santoku.num")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = 6000
local THREADS = nil
local EVALUATE_EVERY = 1
local ITERATIONS = 100

local CLASSES = 10
local CLAUSES = 4096
local TARGET = 32
local SPECIFICITY = 10
local NEGATIVE = 0.1

local VISIBLE = 784
local HIDDEN = 128

local function read_data (fp, max)
  local problems = {}
  local solutions = {}
  local bits = {}
  for l in fs.lines(fp) do
    arr.clear(bits)
    local n = 0
    for bit in str.gmatch(l, "%S+") do
      if n == VISIBLE then
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
    if n ~= VISIBLE then
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
    mtx.add(p, (i - s) * VISIBLE)
    mtx.extend(ps, p)
    arr.push(ss, dataset.solutions[i])
  end
  ss = mtx.create(ss)
  return ps, ss
end

test("tsetlin", function ()

  print("Reading data")
  local dataset = read_data("test/res/BinarizedMNISTData/MNISTTraining.txt", MAX)

  print("Splitting & packing")
  local n_train = num.floor(#dataset.problems * TTR)
  local n_test = #dataset.problems - n_train
  local train_problems, train_solutions = split_dataset(dataset, 1, n_train)
  local test_problems, test_solutions = split_dataset(dataset, n_train + 1, n_train + n_test)
  str.printf("Train %d  Test %d\n", n_train, n_test)

  print("Creating Corex")
  local cor = corex.create({
    visible = VISIBLE,
    hidden = HIDDEN,
    threads = nil,
  })

  print("Training")
  local stopwatch = utc.stopwatch()
  cor.train({
    corpus = train_problems,
    samples = n_train,
    spa = 5.0,
    iterations = ITERATIONS,
    each = function (epoch, tc, dev)
      local duration, total = stopwatch()
      str.printf("Epoch  %-4d   Time  %6.3f  %6.3f   Convergence  %4.6f  %4.6f\n",
        epoch, duration, total, tc, dev)
    end
  })

  print("Transforming train")
  cor.compress(train_problems, n_train)
  mtx.flip_interleave(train_problems, n_train, HIDDEN)
  train_problems = mtx.raw_bitmap(train_problems, n_train, HIDDEN * 2)
  train_solutions = mtx.raw(train_solutions, nil, nil, "u32")

  print("Transforming test")
  cor.compress(test_problems, n_test)
  mtx.flip_interleave(test_problems, n_test, HIDDEN)
  test_problems = mtx.raw_bitmap(test_problems, n_test, HIDDEN * 2)
  test_solutions = mtx.raw(test_solutions, nil, nil, "u32")

  print("Train", n_train)
  print("Test", n_test)

  print("Creating")
  local t = tm.classifier({
    features = HIDDEN,
    classes = CLASSES,
    clauses = CLAUSES,
    target = TARGET,
    specificity = SPECIFICITY,
    negative = NEGATIVE,
    threads = THREADS,
  })

  print("Training")
  local stopwatch = utc.stopwatch()
  t.train({
    samples = n_train,
    problems = train_problems,
    solutions = train_solutions,
    iterations = ITERATIONS,
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
