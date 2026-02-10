local arr = require("santoku.array")
local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local ivec = require("santoku.ivec")
local optimize = require("santoku.tsetlin.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local utc = require("santoku.utc")
local util = require("santoku.tsetlin.util")

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
    per_class = 8192,
  },
  tm = {
    classes = 20,
    clauses = 256,
    clause_tolerance = { def = 8, min = 8, max = 1024, int = true },
    clause_maximum = { def = 8, min = 8, max = 1024, int = true },
    target = { def = 8, min = 8, max = 1024, int = true },
    specificity = { def = 2, min = 2, max = 4000, int = true },
  },
  search = {
    rounds = 6,
    trials = 20,
    iterations = 40,
    subsample = 0.2,
  },
  training = {
    patience = 10,
    batch = 40,
    iterations = 400,
  },
}

test("newsgroups regressor", function ()

  print("Reading data")
  local train, test_set, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max,
    nil,
    cfg.data.tvr)

  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test_set.n)

  print("\nTraining tokenizer\n")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_features = tok:features()
  str.printf("  Vocabulary: %d\n", n_features)

  print("\nTokenizing train for feature selection")
  train.tokens = tok:tokenize(train.problems)
  train.solutions:add_scaled(cfg.tm.classes)

  local token_index = tok:index()

  print("\nFeature selection (union BNS)")
  local union_ids, union_scores
  union_ids, union_scores = train.tokens:bits_top_bns(
    train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.per_class, nil, "max")
  str.printf("  Selected %d union features\n", union_ids:size())
  str.printf("  BNS scores: min=%.2f max=%.2f mean=%.2f\n",
    union_scores:min(), union_scores:max(), union_scores:sum() / union_scores:size())
  train.solutions:add_scaled(-cfg.tm.classes)

  print("\nTop 20 union features:")
  local top_n = math.min(20, union_ids:size())
  local feats = {}
  for i = 0, top_n - 1 do
    local fid = union_ids:get(i)
    local score = union_scores:get(i)
    local token = token_index[fid + 1] or tostring(fid)
    feats[i + 1] = string.format("%s(%.2f)", token, score)
  end
  str.printf("  %s\n", table.concat(feats, ", "))

  print("\nRestricting tokenizer to union features")
  tok:restrict(union_ids)
  n_features = union_ids:size()
  str.printf("  Restricted vocab: %d\n", n_features)

  print("\nTokenizing all splits with restricted tokenizer")
  train.tokens = tok:tokenize(train.problems)
  validate.tokens = tok:tokenize(validate.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore

  print("\nConverting to TM representation")
  train.problems = train.tokens:bits_to_cvec(train.n, n_features, true)
  validate.problems = validate.tokens:bits_to_cvec(validate.n, n_features, true)
  test_set.problems = test_set.tokens:bits_to_cvec(test_set.n, n_features, true)
  local tm_features = n_features
  str.printf("  TM features: %d\n", tm_features)

  train.tokens = nil -- luacheck: ignore
  validate.tokens = nil -- luacheck: ignore
  test_set.tokens = nil -- luacheck: ignore

  print("\nOptimizing Regressor")
  local stopwatch = utc.stopwatch()
  local predicted_buf = ivec.create()
  local t = optimize.regressor({

    features = tm_features,
    outputs = cfg.tm.classes,
    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    specificity = cfg.tm.specificity,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    search_rounds = cfg.search.rounds,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    search_subsample = cfg.search.subsample,
    final_batch = cfg.training.batch,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (regressor)
      local scores = regressor:classify(validate.problems, validate.n, false, predicted_buf)
      local stats = eval.class_accuracy(scores, validate.solutions, validate.n, cfg.tm.classes)
      return stats.f1, stats
    end,

    each = util.make_classifier_log(stopwatch)

  })

  print()
  print("Final Evaluation")

  print("\nClassification metrics:")
  local train_scores = t:classify(train.problems, train.n)
  local val_scores = t:classify(validate.problems, validate.n)
  local test_scores = t:classify(test_set.problems, test_set.n)

  local train_stats = eval.class_accuracy(train_scores, train.solutions, train.n, cfg.tm.classes)
  local val_stats = eval.class_accuracy(val_scores, validate.solutions, validate.n, cfg.tm.classes)
  local test_stats = eval.class_accuracy(test_scores, test_set.solutions, test_set.n, cfg.tm.classes)
  str.printf("  F1:   Train=%.2f  Val=%.2f  Test=%.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

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
