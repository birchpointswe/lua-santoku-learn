local arr = require("santoku.array")
local ds = require("santoku.tsetlin.dataset")
local dvec = require("santoku.dvec")
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
    cgrams_min = 0,
    cgrams_max = 0,
    cgrams_cross = true,
    skips = 0,
  },
  feature_selection = {
    max_vocab = 2^13,
  },
  tm = {
    classes = 2,
    clauses = 8,
    clause_tolerance = { def = 8, min = 8, max = 1024, int = true },
    clause_maximum = { def = 8, min = 8, max = 1024, int = true },
    target = { def = 8, min = 8, max = 1024, int = true },
    specificity = { def = 2, min = 2, max = 2000, int = true },
    include_bits = { def = 1, min = 1, max = 4, int = true },
  },
  search = {
    frac = 0.2,
    patience = 20,
    rounds = 6,
    trials = 20,
    iterations = 40,
  },
  training = {
    patience = 40,
    iterations = 400,
  },
}

local function labels_to_onehot(labels, n_samples, n_classes)
  local out = dvec.create(n_samples * n_classes)
  out:fill(0.0)
  for i = 0, n_samples - 1 do
    local label = labels:get(i)
    out:set(i * n_classes + label, 1.0)
  end
  return out
end

local function predict_class(scores, n_samples, n_classes)
  local predictions = ivec.create(n_samples)
  for i = 0, n_samples - 1 do
    local best_class = 0
    local best_score = scores:get(i * n_classes)
    for c = 1, n_classes - 1 do
      local score = scores:get(i * n_classes + c)
      if score > best_score then
        best_score = score
        best_class = c
      end
    end
    predictions:set(i, best_class)
  end
  return predictions
end

local function class_accuracy(predictions, labels, n_samples, n_classes)
  local classes = {}
  for c = 1, n_classes do
    classes[c] = { tp = 0, fp = 0, fn = 0 }
  end
  local correct = 0
  for i = 0, n_samples - 1 do
    local pred = predictions:get(i)
    local actual = labels:get(i)
    if pred == actual then
      correct = correct + 1
      classes[pred + 1].tp = classes[pred + 1].tp + 1
    else
      classes[pred + 1].fp = classes[pred + 1].fp + 1
      classes[actual + 1].fn = classes[actual + 1].fn + 1
    end
  end
  local f1_sum = 0
  for c = 1, n_classes do
    local tp, fp, fn = classes[c].tp, classes[c].fp, classes[c].fn
    local precision = tp > 0 and tp / (tp + fp) or 0
    local recall = tp > 0 and tp / (tp + fn) or 0
    local f1 = (precision + recall) > 0 and 2 * precision * recall / (precision + recall) or 0
    classes[c].precision = precision
    classes[c].recall = recall
    classes[c].f1 = f1
    f1_sum = f1_sum + f1
  end
  return {
    accuracy = correct / n_samples,
    f1 = f1_sum / n_classes,
    classes = classes,
  }
end

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

  print("Tokenizing all splits")
  train.tokens = tok:tokenize(train.problems)
  validate.tokens = tok:tokenize(validate.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore

  local search_n = math.floor(train.n * cfg.search.frac)
  local search_ids = ivec.create(train.n)
  search_ids:fill_indices()
  search_ids:shuffle()
  search_ids:setn(search_n)
  local search_tokens = ivec.create()
  train.tokens:bits_select(nil, search_ids, n_top_v, search_tokens)

  print("Converting labels to one-hot targets")
  train.targets = labels_to_onehot(train.solutions, train.n, cfg.tm.classes)
  validate.targets = labels_to_onehot(validate.solutions, validate.n, cfg.tm.classes)
  test_set.targets = labels_to_onehot(test_set.solutions, test_set.n, cfg.tm.classes)

  local search_solutions = ivec.create()
  search_solutions:copy(train.solutions, search_ids)
  local search_targets = labels_to_onehot(search_solutions, search_n, cfg.tm.classes)
  str.printf("\nSearch subset: %d / %d samples (%.0f%%)\n", search_n, train.n, cfg.search.frac * 100)

  print("\nConverting to TM representation")
  train.problems = train.tokens:bits_to_cvec(train.n, n_top_v, true)
  validate.problems = validate.tokens:bits_to_cvec(validate.n, n_top_v, true)
  test_set.problems = test_set.tokens:bits_to_cvec(test_set.n, n_top_v, true)
  local search_problems = search_tokens:bits_to_cvec(search_n, n_top_v, true)
  train.tokens = nil -- luacheck: ignore
  validate.tokens = nil -- luacheck: ignore
  test_set.tokens = nil -- luacheck: ignore
  search_tokens = nil -- luacheck: ignore

  print("\nOptimizing Regressor")
  local stopwatch = utc.stopwatch()
  local t = optimize.regressor({

    features = n_top_v,
    outputs = cfg.tm.classes,
    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    specificity = cfg.tm.specificity,
    include_bits = cfg.tm.include_bits,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,
    targets = train.targets,

    search_samples = search_n,
    search_problems = search_problems,
    search_solutions = search_solutions,
    search_targets = search_targets,

    search_patience = cfg.search.patience,
    search_rounds = cfg.search.rounds,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (regressor, info)
      local scores = regressor:predict(info.problems, info.samples)
      local stats = eval.scores_class_accuracy(scores, info.solutions, info.samples, cfg.tm.classes)
      return stats.f1, stats
    end,

    each = util.make_classifier_log(stopwatch)

  })

  print()
  print("Final Evaluation")

  print("\nRegression metrics:")
  local train_scores = t:predict(train.problems, train.n)
  local val_scores = t:predict(validate.problems, validate.n)
  local test_scores = t:predict(test_set.problems, test_set.n)

  local train_reg = eval.regression_accuracy(train_scores, train.targets)
  local val_reg = eval.regression_accuracy(val_scores, validate.targets)
  local test_reg = eval.regression_accuracy(test_scores, test_set.targets)
  str.printf("  MAE:  Train=%.4f  Val=%.4f  Test=%.4f\n", train_reg.mean, val_reg.mean, test_reg.mean)

  print("\nClassification metrics:")
  local train_pred = predict_class(train_scores, train.n, cfg.tm.classes)
  local val_pred = predict_class(val_scores, validate.n, cfg.tm.classes)
  local test_pred = predict_class(test_scores, test_set.n, cfg.tm.classes)

  local train_stats = class_accuracy(train_pred, train.solutions, train.n, cfg.tm.classes)
  local val_stats = class_accuracy(val_pred, validate.solutions, validate.n, cfg.tm.classes)
  local test_stats = class_accuracy(test_pred, test_set.solutions, test_set.n, cfg.tm.classes)
  str.printf("  F1:   Train=%.2f  Val=%.2f  Test=%.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

  print("\nPer-class Test Accuracy (sorted by difficulty):\n")
  local class_names = { "negative", "positive" }
  local class_order = arr.range(1, cfg.tm.classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    str.printf("  %-12s  F1=%.2f  P=%.2f  R=%.2f\n", class_names[c], ts.f1, ts.precision, ts.recall)
  end

  print("\nSample confidence scores (first 5 test samples):")
  for i = 0, 4 do
    local actual = test_set.solutions:get(i)
    local pred = test_pred:get(i)
    local scores = {}
    for c = 0, cfg.tm.classes - 1 do
      scores[c + 1] = str.format("%.2f", test_scores:get(i * cfg.tm.classes + c))
    end
    str.printf("  [%d] actual=%d pred=%d scores=[%s]\n", i, actual, pred, table.concat(scores, ", "))
  end

end)
