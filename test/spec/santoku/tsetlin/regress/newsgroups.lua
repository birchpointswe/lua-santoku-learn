local _ -- luacheck: ignore
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
    max_vocab = 16384,
    grouped = true,
    grouped_vocab = 8192,
  },
  tm = {
    classes = 20,
    clauses = 64,
    clause_tolerance = { def = 41, min = 16, max = 128, int = true },
    clause_maximum = { def = 108, min = 16, max = 128, int = true },
    target = { def = 98, min = 16, max = 128, int = true },
    specificity = { def = 200, min = 100, max = 4000, int = true },
    include_bits = { def = 1, min = 1, max = 4, int = true },
  },
  search = {
    frac = 0.4,
    patience = 6,
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
  str.printf("Feat\t\t%d\t\t\n", n_features)

  print("Tokenizing train")
  train.tokens = tok:tokenize(train.problems)
  train.solutions:add_scaled(cfg.tm.classes)

  local n_top_v, feat_ids, class_offsets

  local tok_index = tok:index()

  local search_n = math.floor(train.n * cfg.search.frac)
  local search_ids = ivec.create(train.n)
  search_ids:fill_indices()
  search_ids:shuffle()
  search_ids:setn(search_n)
  local search_tokens = ivec.create()

  if cfg.feature_selection.grouped then

    print("\nFeature selection (GROUPED mode)")
    class_offsets, feat_ids, _ = train.tokens:bits_top_chi2_grouped(
      train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.grouped_vocab)
    n_top_v = cfg.feature_selection.grouped_vocab
    train.solutions:add_scaled(-cfg.tm.classes)

    print("\nTop 10 tokens per class:")
    for c = 0, cfg.tm.classes - 1 do
      local cat = train.categories[c + 1] or ("class_" .. c)
      local tokens = {}
      for i = 0, math.min(10, cfg.feature_selection.grouped_vocab) - 1 do
        local fid = feat_ids:get(c * cfg.feature_selection.grouped_vocab + i)
        local token = tok_index[fid + 1] or ("?" .. fid)
        tokens[#tokens + 1] = token
      end
      str.printf("  %2d %-24s: %s\n", c, cat, table.concat(tokens, ", "))
    end

    local bytes_per_class = math.ceil(cfg.feature_selection.grouped_vocab * 2 / 8)
    str.printf("  Per-class layout: %d bytes/class, %d bytes/sample\n", bytes_per_class, cfg.tm.classes * bytes_per_class)

    print("Tokenizing all splits")
    validate.tokens = tok:tokenize(validate.problems)
    test_set.tokens = tok:tokenize(test_set.problems)
    train.tokens:bits_select(nil, search_ids, n_features, search_tokens)
    tok = nil -- luacheck: ignore

  else

    print("\nFeature selection (REGULAR mode)")
    local chi2_ids = train.tokens:bits_top_chi2(
      train.solutions, train.n, n_features, cfg.tm.classes,
      cfg.feature_selection.max_vocab, "max")
    n_top_v = chi2_ids:size()
    str.printf("  Chi2: %d features selected\n", n_top_v)
    train.solutions:add_scaled(-cfg.tm.classes)
    tok:restrict(chi2_ids)
    chi2_ids = nil -- luacheck: ignore

    print("Tokenizing all splits")
    train.tokens = tok:tokenize(train.problems)
    validate.tokens = tok:tokenize(validate.problems)
    test_set.tokens = tok:tokenize(test_set.problems)
    train.tokens:bits_select(nil, search_ids, n_top_v, search_tokens)
    tok = nil -- luacheck: ignore

  end

  print("Converting labels to one-hot targets")
  train.targets = labels_to_onehot(train.solutions, train.n, cfg.tm.classes)
  validate.targets = labels_to_onehot(validate.solutions, validate.n, cfg.tm.classes)
  test_set.targets = labels_to_onehot(test_set.solutions, test_set.n, cfg.tm.classes)

  local search_solutions = ivec.create()
  search_solutions:copy(train.solutions, search_ids)
  local search_targets = labels_to_onehot(search_solutions, search_n, cfg.tm.classes)

  print("\nConverting to TM representation")
  local search_problems
  if cfg.feature_selection.grouped then
    train.problems, n_top_v = train.tokens:bits_to_cvec(train.n, n_features, class_offsets, feat_ids, true)
    validate.problems = validate.tokens:bits_to_cvec(validate.n, n_features, class_offsets, feat_ids, true)
    test_set.problems = test_set.tokens:bits_to_cvec(test_set.n, n_features, class_offsets, feat_ids, true)
    search_problems = search_tokens:bits_to_cvec(search_n, n_features, class_offsets, feat_ids, true)
  else
    train.problems = train.tokens:bits_to_cvec(train.n, n_top_v, true)
    validate.problems = validate.tokens:bits_to_cvec(validate.n, n_top_v, true)
    test_set.problems = test_set.tokens:bits_to_cvec(test_set.n, n_top_v, true)
    search_problems = search_tokens:bits_to_cvec(search_n, n_top_v, true)
  end
  train.tokens = nil -- luacheck: ignore
  validate.tokens = nil -- luacheck: ignore
  test_set.tokens = nil -- luacheck: ignore
  search_tokens = nil -- luacheck: ignore
  str.printf("\nSearch subset: %d / %d samples (%.0f%%)\n", search_n, train.n, cfg.search.frac * 100)

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
    grouped = cfg.feature_selection.grouped,

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
  local class_order = arr.range(1, cfg.tm.classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    local cat = train.categories[c] or ("class_" .. (c - 1))
    str.printf("  %-28s  F1=%.2f  P=%.2f  R=%.2f\n", cat, ts.f1, ts.precision, ts.recall)
  end

  print("\nSample confidence scores (first 5 test samples):")
  for i = 0, 4 do
    local actual = test_set.solutions:get(i)
    local pred = test_pred:get(i)
    local best_score = test_scores:get(i * cfg.tm.classes + pred)
    str.printf("  [%d] actual=%d pred=%d score=%.2f\n", i, actual, pred, best_score)
  end

end)
