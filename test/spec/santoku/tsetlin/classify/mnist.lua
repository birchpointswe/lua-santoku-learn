local serialize = require("santoku.serialize") -- luacheck: ignore
local test = require("santoku.test")
local utc = require("santoku.utc")
local tm = require("santoku.tsetlin")
local num = require("santoku.num")
local bm = require("santoku.bitmap")
local bmc = require("santoku.bitmap.compressor")
local mtx = require("santoku.matrix")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")

local CLASSES = 10
local FEATURES = 784
local FEATURES_CMP = 128
local CMP_ITERS = 10
local CMP_EPS = 1e-6
local TRAIN_TEST_RATIO = 0.9
local CLAUSES = 4096
local STATE = 8
local TARGET = 32
local SPECIFICITY = 20
local BOOST = true
local ACTIVE = 0.85
local THREADS = nil
local EVALUATE_EVERY = 1
local ITERATIONS = 20

local function prep_fingerprint (fingerprint, bits)
  local flipped = bm.copy(fingerprint)
  bm.flip(flipped, 1, bits)
  bm.extend(fingerprint, flipped, bits + 1)
  return fingerprint
end

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
        else
          bits[n] = nil
        end
      end
      if n ~= FEATURES + 1 then
        error("bitmap length mismatch")
      else
        problems[#problems + 1] = bm.create(bits, FEATURES)
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

  do
    print("Running compress")
    local function split_compress (i0, i1)
      print("Splitting")
      local out = {}
      for i = i0, i1 do
        arr.push(out, dataset.problems[i])
      end
      return bm.matrix(out, FEATURES)
    end
    local n_train = num.floor(#dataset.problems * TRAIN_TEST_RATIO)
    local n_test = #dataset.problems - n_train
    local cmp_train = split_compress(1, n_train)
    local cmp_test = split_compress(n_train + 1, #dataset.problems)
    local compressor = bmc.create({
      visible = FEATURES,
      hidden = FEATURES_CMP,
      threads = THREADS,
    })
    print("Fitting")
    local stopwatch = utc.stopwatch()
    local mavg = num.mavg(0.2)
    compressor.train({
      corpus = cmp_train,
      samples = n_train,
      iterations = CMP_ITERS,
      each = function (epoch, tc)
        local duration, total = stopwatch()
        local tc0 = mavg(tc)
        str.printf("Epoch  %-4d  Time  %6.3f (%6.3f)  Convergence  %-4.6f (%-4.6f)\n",
          epoch, duration, total, tc, tc0)
        return epoch < 10 or num.abs(tc - tc0) > CMP_EPS
      end
    })
    print("Transforming train")
    cmp_train = compressor.compress(cmp_train, n_train)
    print(">", bm.tostring(cmp_train, FEATURES_CMP))
    print("Transforming test")
    cmp_test = compressor.compress(cmp_test, n_test)
    print("Recreating bitmaps")
    local cmp_all = bm.copy(cmp_train)
    bm.extend(cmp_all, cmp_test, n_train * FEATURES_CMP + 1)
    for i = 1, #dataset.problems do
      bm.clear(dataset.problems[i])
      for j = 1, FEATURES_CMP do
        if bm.get(cmp_all, (i - 1) * FEATURES_CMP + j) then
          bm.set(dataset.problems[i], j)
        end
      end
      dataset.problems[i] = prep_fingerprint(dataset.problems[i], FEATURES_CMP)
    end
  end
  FEATURES = FEATURES_CMP

  print("Splitting & packing")
  local n_train = num.floor(#dataset.problems * TRAIN_TEST_RATIO)
  local n_test = #dataset.problems - n_train
  local train_problems, train_solutions = split_dataset(dataset, 1, n_train)
  local test_problems, test_solutions = split_dataset(dataset, n_train + 1, n_train + n_test)

  print("Train", n_train)
  print("Test", n_test)

  print("Creating")
  local t = tm.classifier({
    classes = CLASSES,
    features = FEATURES,
    clauses = CLAUSES,
    state = STATE,
    target = TARGET,
    boost = BOOST,
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
    active = ACTIVE,
    each = function (epoch)
      local duration = stopwatch()
      if epoch == ITERATIONS or epoch % EVALUATE_EVERY == 0 then
        local test_score =
          t.evaluate({
            problems = test_problems,
            solutions = test_solutions,
            samples = n_test,
          })
        local train_score --[[, confusion, observed, predicted]] =
          t.evaluate({
            problems = train_problems,
            solutions = train_solutions,
            samples = n_train
          })
        str.printf("Epoch %-4d  Time %4.2f  Test %4.2f  Train %4.2f\n",
          epoch, duration, test_score, train_score)
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
  local test_score  =
    t.evaluate({
      problems = test_problems,
      solutions = test_solutions,
      samples = n_test
    })
  local train_score--[[, confusion, predictions]] =
    t.evaluate({
      problems = train_problems,
      solutions = train_solutions,
      samples = n_train,
      -- stats = true,
    })
  -- print()
  -- print("Confusion:")
  -- print(require("santoku.serialize")(confusion))
  -- print()
  -- print("Predictions:")
  -- print(require("santoku.serialize")(predictions))
  -- print()
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_score, train_score)

end)
