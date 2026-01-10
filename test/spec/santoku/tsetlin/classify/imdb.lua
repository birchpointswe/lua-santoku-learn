local fs = require("santoku.fs")
local err = require("santoku.error")
local serialize = require("santoku.serialize") -- luacheck: ignore
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.tsetlin")
local utc = require("santoku.utc")

local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local optimize = require("santoku.tsetlin.optimize")
local tokenizer = require("santoku.tokenizer")

local cfg = {
  data = {
    ttr = 0.9,
    max = nil,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 3,
    cgrams_max = 4,
    skips = 2,
    negations = 4,
  },
  feature_selection = {
    algo = "chi2",
    top_k = 2048,
  },
  tm = {
    classes = 2,
    negative = 0.5,
    clauses = { def = 8, min = 8, max = 32, int = true, log = true },
    clause_tolerance = { def = 8, min = 8, max = 64, int = true, log = true },
    clause_maximum = { def = 8, min = 8, max = 64, int = true, log = true },
    target = { def = 4, min = 8, max = 64, int = true, log = true },
    specificity = { def = 1000, min = 2, max = 2000, int = true, log = true },
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
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test = ds.split_imdb(dataset, cfg.data.ttr)

  print("Train", train.n)
  print("Test", test.n)

  print("\nTraining tokenizer\n")
  local tokenizer = tokenizer.create(cfg.tokenizer)
  tokenizer:train({ corpus = train.problems })
  tokenizer:finalize()
  dataset.n_features = tokenizer:features()
  str.printf("Feat\t\t%d\t\t\n", dataset.n_features)

  print("Tokenizing train")
  train.problems0 = tokenizer:tokenize(train.problems)
  train.solutions:add_scaled(cfg.tm.classes)
  local top_v =
    cfg.feature_selection.algo == "chi2" and train.problems0:bits_top_chi2(train.solutions, train.n, dataset.n_features, cfg.tm.classes, cfg.feature_selection.top_k) or
    cfg.feature_selection.algo == "mi" and train.problems0:bits_top_mi(train.solutions, train.n, dataset.n_features, cfg.tm.classes, cfg.feature_selection.top_k) or
    err.error("feature_selection.algo must be chi2 or mi")
  train.solutions:add_scaled(-cfg.tm.classes)
  local n_top_v = top_v:size()
  print("After top k filter", n_top_v)

  -- Show top words
  local words = tokenizer:index()
  local nw = 0
  for id in top_v:each() do
    print(id, words[id + 1])
    nw = nw + 1
    if nw >= 32 then break end
  end

  print("Re-encoding train/test with top features")
  tokenizer:restrict(top_v)
  train.problems = tokenizer:tokenize(train.problems);
  test.problems = tokenizer:tokenize(test.problems);

  print("Prepping for classifier")
  train.problems = train.problems:bits_to_cvec(train.n, n_top_v, true)
  test.problems = test.problems:bits_to_cvec(test.n, n_top_v, true)

  print("Optimizing Classifier")
  local stopwatch = utc.stopwatch()
  local t = optimize.classifier({

    features = n_top_v,

    classes = cfg.tm.classes,
    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    negative = cfg.tm.negative,
    specificity = cfg.tm.specificity,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

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
