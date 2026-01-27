local arr = require("santoku.array")
local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local optimize = require("santoku.tsetlin.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local utc = require("santoku.utc")

local cfg = {
  data = {
    max = nil,
    tvr = 0.1,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 3,
    cgrams_max = 5,
    cgrams_cross = true,
    skips = 1,
  },
  feature_selection = {
    max_vocab = 2^15,
  },
  tm = {
    classes = 20,
    clauses = 64,
    clause_tolerance = { def = 41, min = 16, max = 128, int = true },
    clause_maximum = { def = 108, min = 16, max = 128, int = true },
    target = { def = 98, min = 16, max = 128, int = true },
    specificity = { def = 534, min = 400, max = 4000 },
    include_bits = { def = 3, min = 1, max = 4, int = true },
  },
  search = {
    patience = 4,
    rounds = 6,
    trials = 20,
    iterations = 10,
  },
  training = {
    patience = 40,
    iterations = 400,
  },
}

test("newsgroups", function ()

  print("Reading data")
  local train, test, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max,
    nil,
    cfg.data.tvr)

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

  print("\nFeature selection")
  local chi2_ids = train.problems0:bits_top_chi2(
    train.solutions, train.n, n_features, cfg.tm.classes,
    cfg.feature_selection.max_vocab, "max")
  local n_top_v = chi2_ids:size()
  str.printf("  Chi2: %d features\n", n_top_v)
  train.solutions:add_scaled(-cfg.tm.classes)
  train.problems0 = nil -- luacheck: ignore
  tok:restrict(chi2_ids)
  chi2_ids = nil -- luacheck: ignore

  local function to_bitmap (split)
    local toks = tok:tokenize(split.problems)
    local bitmap = toks:bits_to_cvec(split.n, n_top_v, true)
    toks = nil -- luacheck: ignore
    return bitmap
  end

  train.problems = to_bitmap(train)
  validate.problems = to_bitmap(validate)
  test.problems = to_bitmap(test)
  tok = nil -- luacheck: ignore

  print("Optimizing Classifier")
  local stopwatch = utc.stopwatch()
  local t = optimize.classifier({

    features = n_top_v,
    classes = cfg.tm.classes,
    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    specificity = cfg.tm.specificity,
    include_bits = cfg.tm.include_bits,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    search_patience = cfg.search.patience,
    search_rounds = cfg.search.rounds,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (t0, _)
      local predicted = t0:predict(validate.problems, validate.n)
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
  print("Final Evaluation")
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
    local cat = train.categories[c] or ("class_" .. (c - 1))
    str.printf("  %-28s  F1=%.2f  P=%.2f  R=%.2f\n", cat, ts.f1, ts.precision, ts.recall)
  end

end)
