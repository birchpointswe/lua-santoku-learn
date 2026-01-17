local str = require("santoku.string")
local test = require("santoku.test")
local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local optimize = require("santoku.tsetlin.optimize")
local tokenizer = require("santoku.tokenizer")
local utc = require("santoku.utc")

local cfg = {
  data = {
    ttr = 0.5,
    tvr = 0.1,
    max = nil,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 3,
    cgrams_max = 4,
    skips = 1,
    negations = 0,
  },
  feature_selection = {
    algo = "chi2",
    top_k = 12800,
  },
  tm = {
    classes = 2,
    negative = 0.5,
    clauses = { def = 16, min = 8, max = 32, round = 8 },
    clause_tolerance = { def = 64, min = 16, max = 128, int = true },
    clause_maximum = { def = 64, min = 16, max = 128, int = true },
    target = { def = 32, min = 16, max = 128, int = true },
    specificity = { def = 1000, min = 400, max = 4000 },
    include_bits = { def = 1, min = 1, max = 4, int = true },
  },
  search = {
    patience = 2,
    rounds = 6,
    trials = 10,
    iterations = 20,
  },
  training = {
    iterations = 200,
  },
  threads = nil,
}

test("tsetlin", function ()

  print("Reading data")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)

  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test.n)

  print("\nTraining tokenizer\n")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_features = tok:features()
  str.printf("Feat\t\t%d\t\t\n", n_features)

  print("Tokenizing train")
  train.problems0 = tok:tokenize(train.problems)
  train.solutions:add_scaled(cfg.tm.classes)
  local top_v, chi2_weights
  if cfg.feature_selection.algo == "chi2" then
    top_v, chi2_weights = train.problems0:bits_top_chi2(train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.top_k)
  else
    top_v, chi2_weights = train.problems0:bits_top_mi(train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.top_k)
  end
  train.solutions:add_scaled(-cfg.tm.classes)
  local n_top_v = top_v:size()
  print("After top k filter", n_top_v)
  train.problems0 = nil

  local words = tok:index()
  print("\nTop 30 by Chi2 score:")
  for i = 0, 29 do
    local id = top_v:get(i)
    local score = chi2_weights:get(i)
    str.printf("  %6d  %-24s  %.4f\n", id, words[id + 1] or "?", score)
  end

  print("\nBottom 30 (of selected subset) by Chi2 score:")
  for i = 0, 29 do
    local id = top_v:get(n_top_v - i - 1)
    local score = chi2_weights:get(n_top_v - i - 1)
    str.printf("  %6d  %-24s  %.4f\n", id, words[id + 1] or "?", score)
  end

  tok:restrict(top_v)

  print("\nRe-encoding train/validate/test with top features")
  train.problems = tok:tokenize(train.problems)
  validate.problems = tok:tokenize(validate.problems)
  test.problems = tok:tokenize(test.problems)
  tok:destroy()

  print("Prepping for classifier")
  train.problems = train.problems:bits_to_cvec(train.n, n_top_v, true)
  validate.problems = validate.problems:bits_to_cvec(validate.n, n_top_v, true)
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
    include_bits = cfg.tm.include_bits,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    search_patience = cfg.search.patience,
    search_rounds = cfg.search.rounds,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    final_iterations = cfg.training.iterations,

    search_metric = function (t)
      local predicted = t:predict(validate.problems, validate.n, cfg.threads)
      local accuracy = eval.class_accuracy(predicted, validate.solutions, validate.n, cfg.tm.classes, cfg.threads)
      return accuracy.f1, accuracy
    end,

    each = function (t, is_final, val_accuracy, params, epoch, round, trial)
      local test_predicted = t:predict(test.problems, test.n, cfg.threads)
      local test_accuracy = eval.class_accuracy(test_predicted, test.solutions, test.n, cfg.tm.classes, cfg.threads)
      local d, dd = stopwatch()
      if is_final then
        str.printf("  Time %3.2f %3.2f  Finalizing  C=%d LF=%d L=%d T=%.2f S=%.2f IB=%d  F1=(val=%.2f,test=%.2f)  Epoch  %d\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, params.include_bits, val_accuracy.f1, test_accuracy.f1, epoch)
      else
        str.printf("  Time %3.2f %3.2f  Exploring  C=%d LF=%d L=%d T=%.2f S=%.2f IB=%d  R=%d T=%d  F1=(val=%.2f,test=%.2f)  Epoch  %d\n",
          d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, params.include_bits, round, trial, val_accuracy.f1, test_accuracy.f1, epoch)
      end
    end

  })

  print()
  print("Final Evaluation")
  local train_pred = t:predict(train.problems, train.n, cfg.threads)
  local val_pred = t:predict(validate.problems, validate.n, cfg.threads)
  local test_pred = t:predict(test.problems, test.n, cfg.threads)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, cfg.tm.classes, cfg.threads)
  local val_stats = eval.class_accuracy(val_pred, validate.solutions, validate.n, cfg.tm.classes, cfg.threads)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, cfg.tm.classes, cfg.threads)
  str.printf("Evaluate\tTrain\t%4.2f\tVal\t%4.2f\tTest\t%4.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

end)
