local arr = require("santoku.array")
local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local fs = require("santoku.fs")
local ivec = require("santoku.ivec")
local optimize = require("santoku.tsetlin.optimize")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local cfg = {
  data = {
    ttr = 0.8,
    tvr = 0.1,
    max = nil,
    features = 784,
  },
  tm = {
    classes = 10,
    clauses = 256,
    clause_tolerance = { def = 78, min = 16, max = 256, int = true },
    clause_maximum = { def = 91, min = 16, max = 256, int = true },
    target = { def = 40, min = 16, max = 256, int = true },
    specificity = { def = 3267, min = 400, max = 4000, int = true },
    include_bits = { def = 4, min = 1, max = 4, int = true },
  },
  search = {
    patience = 4,
    rounds = 6,
    trials = 20,
    iterations = 40,
  },
  training = {
    patience = 40,
    iterations = 400,
  },
}

test("mnist", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.features, cfg.data.max)
  print("Splitting")
  local train, test, validate = ds.split_binary_mnist(dataset, cfg.data.ttr, cfg.data.tvr)
  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test.n)

  train.problems = ivec.create()
  dataset.problems:bits_select(nil, train.ids, dataset.n_features, train.problems)
  validate.problems = ivec.create()
  dataset.problems:bits_select(nil, validate.ids, dataset.n_features, validate.problems)
  test.problems = ivec.create()
  dataset.problems:bits_select(nil, test.ids, dataset.n_features, test.problems)

  str.printf("Transforming train\t%d\n", train.n)
  train.problems = train.problems:bits_to_cvec(train.n, dataset.n_features, true)

  str.printf("Transforming validate\t%d\n", validate.n)
  validate.problems = validate.problems:bits_to_cvec(validate.n, dataset.n_features, true)

  str.printf("Transforming test\t%d\n", test.n)
  test.problems = test.problems:bits_to_cvec(test.n, dataset.n_features, true)

  print("Training\n")
  local stopwatch = utc.stopwatch()
  local t = optimize.classifier({

    features = dataset.n_features,
    classes = cfg.tm.classes,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    specificity = cfg.tm.specificity,
    include_bits = cfg.tm.include_bits,

    search_patience = cfg.search.patience,
    search_rounds = cfg.search.rounds,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (t)
      local predicted = t:predict(validate.problems, validate.n)
      local accuracy = eval.class_accuracy(predicted, validate.solutions, validate.n, cfg.tm.classes)
      return accuracy.f1, accuracy
    end,

    each = function (_, is_final, val_accuracy, params, epoch, round, trial)
      local d, dd = stopwatch()
      local phase = is_final and "F" or str.format("R%d T%d", round, trial)
      str.printf("[CLASSIFY %s E%d] C=%d L=%d/%d T=%d S=%.0f IB=%d F1=%.2f (%.2fs +%.2fs)\n",
        phase, epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
        params.target, params.specificity, params.include_bits, val_accuracy.f1, d, dd)
    end

  })

  print()
  print("Persisting")
  fs.rm("model.bin", true)
  t:persist("model.bin", true)

  print("Testing restore")
  t = tm.load("model.bin", nil, true)
  local train_pred = t:predict(train.problems, train.n)
  local val_pred = t:predict(validate.problems, validate.n)
  local test_pred = t:predict(test.problems, test.n)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, cfg.tm.classes)
  local val_stats = eval.class_accuracy(val_pred, validate.solutions, validate.n, cfg.tm.classes)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, cfg.tm.classes)
  str.printf("Evaluate\tTrain\t%4.2f\tVal\t%4.2f\tTest\t%4.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

  print("\nPer-class Test Accuracy (sorted by difficulty):\n")
  local class_order = arr.range(1, cfg.tm.classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    str.printf("  digit_%-2d  F1=%.2f  P=%.2f  R=%.2f\n", c - 1, ts.f1, ts.precision, ts.recall)
  end

end)
