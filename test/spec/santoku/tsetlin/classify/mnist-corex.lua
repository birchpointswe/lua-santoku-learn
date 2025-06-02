local ds = require("santoku.tsetlin.dataset")
local corex = require("santoku.corex")
local eval = require("santoku.tsetlin.evaluator")
local fs = require("santoku.fs")
local ivec = require("santoku.ivec")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local TTR = 0.9
local MAX = 10000
local THREADS = nil
local EVALUATE_EVERY = 1
local TM_ITERS = 100
local COREX_ITERS = 100

local CLASSES = 10
local CLAUSES = 1024
local TARGET = 32
local SPECIFICITY = 10
local NEGATIVE = 0.1

local VISIBLE = 784
local HIDDEN = 128

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/BinarizedMNISTData/MNISTTraining.txt", VISIBLE, MAX)
  local train, test = ds.split_binary_mnist(dataset, TTR)

  print("Creating Corex")
  local cor = corex.create({
    visible = dataset.n_features,
    hidden = HIDDEN,
    threads = THREADS,
  })

  print("Training")
  local stopwatch = utc.stopwatch()
  cor.train({
    corpus = train.problems,
    samples = train.n,
    iterations = COREX_ITERS,
    each = function (epoch, tc, dev)
      local duration, total = stopwatch()
      str.printf("Epoch  %-4d   Time  %6.3f  %6.3f   Convergence  %4.6f  %4.6f\n",
        epoch, duration, total, tc, dev)
    end
  })

  print("Transforming train")
  cor.compress(train.problems, train.n)
  ivec.flip_interleave(train.problems, train.n, HIDDEN)
  train.problems = ivec.raw_bitmap(train.problems, train.n, HIDDEN * 2)
  train.solutions = ivec.raw(train.solutions, "u32")

  print("Transforming test")
  cor.compress(test.problems, test.n)
  ivec.flip_interleave(test.problems, test.n, HIDDEN)
  test.problems = ivec.raw_bitmap(test.problems, test.n, HIDDEN * 2)
  test.solutions = ivec.raw(test.solutions, "u32")

  print("Train", train.n)
  print("Test", test.n)

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
    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,
    iterations = TM_ITERS,
    each = function (epoch)
      local train_pred = t.predict(train.problems, train.n)
      local test_pred = t.predict(test.problems, test.n)
      local duration = stopwatch()
      if epoch == TM_ITERS or epoch % EVALUATE_EVERY == 0 then
        local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, CLASSES)
        local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, CLASSES)
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
  local train_pred = t.predict(train.problems, train.n)
  local test_pred = t.predict(test.problems, test.n)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, CLASSES)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, CLASSES)
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_stats.f1, train_stats.f1)

end)
