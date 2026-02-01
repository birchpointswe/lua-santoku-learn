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
    cgrams_max = 5,
    cgrams_cross = true,
    skips = 1,
  },
  feature_selection = {
    per_class = 4096,
    union = true,
  },
  tm = {
    classes = 2,
    clauses = 8,
    clause_tolerance = { def = 8, min = 8, max = 1024, int = true },
    clause_maximum = { def = 8, min = 8, max = 1024, int = true },
    target = { def = 8, min = 8, max = 1024, int = true },
    specificity = { def = 2, min = 2, max = 2000, int = true },
  },
  search = {
    rounds = 6,
    trials = 20,
    iterations = 40,
  },
  training = {
    patience = 4,
    batch = 10,
    iterations = 400,
  },
}

test("imdb regressor", function ()

  print("Reading data")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test_set, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)

  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test_set.n)

  print("\nTraining tokenizer\n")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_features = tok:features()
  str.printf("Feat\t\t%d\t\t\n", n_features)

  print("Tokenizing train for feature selection")
  train.tokens = tok:tokenize(train.problems)
  train.solutions:add_scaled(cfg.tm.classes)

  local token_index = tok:index()
  local class_names = { "negative", "positive" }
  local bns_offsets, bns_ids, bns_scores
  local tm_features, grouped

  if cfg.feature_selection.union then
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

    print("\nConverting to TM representation (flat)")
    train.problems = train.tokens:bits_to_cvec(train.n, n_features, true)
    validate.problems = validate.tokens:bits_to_cvec(validate.n, n_features, true)
    test_set.problems = test_set.tokens:bits_to_cvec(test_set.n, n_features, true)
    tm_features = n_features
    grouped = false
    str.printf("  TM features: %d (flat)\n", tm_features)

  else
    print("\nFeature selection (grouped BNS)")
    _, _, bns_offsets, bns_ids, bns_scores = train.tokens:bits_top_bns(
      train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.per_class, nil, "max")
    str.printf("  Selected %d features (%d per class x %d classes)\n",
      bns_ids:size(), cfg.feature_selection.per_class, cfg.tm.classes)
    str.printf("  BNS scores: min=%.2f max=%.2f mean=%.2f\n",
      bns_scores:min(), bns_scores:max(), bns_scores:sum() / bns_scores:size())
    train.solutions:add_scaled(-cfg.tm.classes)

    print("\nTop 10 features per class:")
    for c = 1, cfg.tm.classes do
      local start_idx = bns_offsets:get(c - 1)
      local end_idx = bns_offsets:get(c)
      local top_n = math.min(10, end_idx - start_idx)
      local feats = {}
      for i = 0, top_n - 1 do
        local fid = bns_ids:get(start_idx + i)
        local score = bns_scores:get(start_idx + i)
        local token = token_index[fid + 1] or tostring(fid)
        feats[i + 1] = string.format("%s(%.2f)", token, score)
      end
      str.printf("  %-12s %s\n", class_names[c] .. ":", table.concat(feats, ", "))
    end

    print("\nTokenizing all splits")
    validate.tokens = tok:tokenize(validate.problems)
    test_set.tokens = tok:tokenize(test_set.problems)
    tok = nil -- luacheck: ignore

    print("\nConverting to TM representation (grouped)")
    local max_k
    train.problems, max_k = train.tokens:bits_to_cvec(train.n, n_features, bns_offsets, bns_ids, true)
    validate.problems = validate.tokens:bits_to_cvec(validate.n, n_features, bns_offsets, bns_ids, true)
    test_set.problems = test_set.tokens:bits_to_cvec(test_set.n, n_features, bns_offsets, bns_ids, true)
    tm_features = max_k
    grouped = true
    str.printf("  TM features: %d per class (max_k=%d x %d classes)\n", max_k, max_k, cfg.tm.classes)
  end

  train.tokens = nil -- luacheck: ignore
  validate.tokens = nil -- luacheck: ignore
  test_set.tokens = nil -- luacheck: ignore

  print("\nOptimizing Regressor")
  local stopwatch = utc.stopwatch()
  local predicted_buf = ivec.create()
  local t = optimize.regressor({

    features = tm_features,
    outputs = cfg.tm.classes,
    grouped = grouped,
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
    final_batch = cfg.training.batch,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (regressor)
      local scores = regressor:classify(validate.problems, validate.n, true, predicted_buf)
      local stats = eval.class_accuracy(scores, validate.solutions, validate.n, cfg.tm.classes)
      return stats.f1, stats
    end,

    each = util.make_classifier_log(stopwatch)

  })

  print()
  print("Final Evaluation")

  print("\nClassification metrics:")
  local train_scores = t:classify(train.problems, train.n, true)
  local val_scores = t:classify(validate.problems, validate.n, true)
  local test_scores = t:classify(test_set.problems, test_set.n, true)

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
    str.printf("  %-12s  F1=%.2f  P=%.2f  R=%.2f\n", class_names[c], ts.f1, ts.precision, ts.recall)
  end

end)
