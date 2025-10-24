local ds = require("santoku.tsetlin.dataset")
local corex = require("santoku.corex")
local ivec = require("santoku.ivec")
local eval = require("santoku.tsetlin.evaluator")
local fs = require("santoku.fs")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local cfg = {
  data = {
    ttr = 0.9,
    max = 1000,
    visible = 784,
    hidden = 128,
  },
  corex = {
    iterations = 10,
  },
  tm = {
    classes = 10,
    negative = 0.1,
    clauses = { def = 8, min = 8, max = 32, log = true, int = true },
    clause_tolerance = { def = 8, min = 8, max = 64, int = true },
    clause_maximum = { def = 8, min = 8, max = 64, int = true },
    target = { def = 4, min = 8, max = 64 },
    specificity = { def = 1000, min = 1, max = 2000, int = true, log = true },
  },
  search = {
    patience = 3,
    rounds = 10,
    trials = 4,
    iterations = 10,
  },
  training = {
    iterations = 100,
  },
  threads = nil,
}

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.visible, cfg.data.max)
  local train, test = ds.split_binary_mnist(dataset, cfg.data.ttr)
  train.problems = ivec.create()
  dataset.problems:bits_select(nil, train.ids, dataset.n_features, train.problems)
  test.problems = ivec.create()
  dataset.problems:bits_select(nil, test.ids, dataset.n_features, test.problems)

  print("Creating Corex")
  local cor = corex.create({
    visible = dataset.n_features,
    hidden = cfg.data.hidden,
  })

  print("Training")
  local stopwatch = utc.stopwatch()
  cor:train({
    corpus = train.problems,
    samples = train.n,
    iterations = cfg.corex.iterations,
    each = function (epoch, tc, dev)
      local duration, total = stopwatch()
      str.printf("Epoch  %-4d   Time  %6.3f  %6.3f   Convergence  %4.6f  %4.6f\n",
        epoch, duration, total, tc, dev)
    end
  })

  print("Transforming train")
  cor:compress(train.problems, train.n)
  train.problems = train.problems:bits_to_cvec(train.n, cfg.data.hidden, true)

  print("Transforming test")
  cor:compress(test.problems, test.n)
  test.problems = test.problems:bits_to_cvec(test.n, cfg.data.hidden, true)

  print("Train", train.n)
  print("Test", test.n)

  print("Optimizing classifier")
  local t = tm.optimize_classifier({

    features = cfg.data.hidden,
    classes = cfg.tm.classes,
    negative = cfg.tm.negative,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    specificity = cfg.tm.specificity,

    search_patience = cfg.search.patience,
    search_rounds = cfg.search.rounds,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    final_iterations = cfg.training.iterations,

    search_metric = function (t)
      local predicted = t:predict(train.problems, train.n, cfg.threads)
      local accuracy = eval.class_accuracy(predicted, train.solutions, train.n, cfg.tm.classes, cfg.threads)
      return accuracy.f1, accuracy
    end,

    each = function (t, is_final, train_accuracy, params, epoch, round, trial)
      local test_predicted = t:predict(test.problems, test.n, cfg.threads)
      local test_accuracy = eval.class_accuracy(test_predicted, test.solutions, test.n, cfg.tm.classes, cfg.threads)
      local d, dd = stopwatch()
      -- luacheck: push ignore
      if is_final then
        str.printf("  Time %3.2f %3.2f  Finalizing  C=%d LF=%d L=%d T=%.2f S=%.2f  F1=(%.2f,%.2f)  Epoch  %d\n\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, train_accuracy.f1, test_accuracy.f1, epoch)
      else
        str.printf("  Time %3.2f %3.2f  Exploring  C=%d LF=%d L=%d T=%.2f S=%.2f  R=%d T=%d  F1=(%.2f,%.2f)  Epoch  %d\n\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, round, trial, train_accuracy.f1, test_accuracy.f1, epoch)
      end
      -- luacheck: pop
    end

  })

  print()
  print("Persisting")
  fs.rm("model.bin", true)
  t:persist("model.bin", true)

  print("Testing restore")
  t = tm.load("model.bin")
  local train_pred = t:predict(train.problems, train.n, cfg.threads)
  local test_pred = t:predict(test.problems, test.n, cfg.threads)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, cfg.tm.classes, cfg.threads)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, cfg.tm.classes, cfg.threads)
  str.printf("Evaluate\tTest\t%4.2f\tTrain\t%4.2f\n", test_stats.f1, train_stats.f1)

end)
