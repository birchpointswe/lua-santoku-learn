local arr = require("santoku.array")
local str = require("santoku.string")
local test = require("santoku.test")
local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
require("santoku.ivec")
local optimize = require("santoku.tsetlin.optimize")
local tokenizer = require("santoku.tokenizer")
local utc = require("santoku.utc")

local cfg = {
  data = {
    max_per_class = nil,
    tvr = 0.1,
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
    top_k = 12288,
    individualized = true,
  },
  tm = {
    classes = 20,
    negative = 0.05,
    clauses = { def = 16, min = 8, max = 256, round = 8 },
    clause_tolerance = { def = 64, min = 16, max = 128, int = true },
    clause_maximum = { def = 64, min = 16, max = 128, int = true },
    target = { def = 32, min = 16, max = 128, int = true },
    specificity = { def = 1000, min = 400, max = 4000 },
    include_bits = { def = 1, min = 1, max = 4, int = true },
  },
  search = {
    patience = 6,
    rounds = 6,
    trials = 10,
    iterations = 20,
  },
  training = {
    patience = 20,
    iterations = 400,
  },
  threads = nil,
}

test("tsetlin", function ()

  print("Reading data")
  local train, test, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max_per_class,
    nil,
    cfg.data.tvr)

  print("Train", train.n)
  print("Validate", validate.n)
  print("Test", test.n)

  print("\nTraining tokenizer\n")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_features = tok:features()
  str.printf("Feat\t\t%d\t\t\n", n_features)

  print("Tokenizing train")
  train.problems0 = tok:tokenize(train.problems)
  train.solutions:add_scaled(cfg.tm.classes)

  local n_top_v, feat_offsets, train_dim_offsets, validate_dim_offsets, test_dim_offsets
  local use_ind = cfg.feature_selection.individualized

  if use_ind then
    str.printf("\nPer-class chi2 feature selection (individualized=%s)\n", tostring(use_ind))
    local ids_union, feat_offs, feat_ids = train.problems0:bits_top_chi2_ind(
      train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.top_k)
    feat_offsets = feat_offs
    local union_size = ids_union:size()
    local total_features = feat_offsets:get(cfg.tm.classes)
    str.printf("  Per-class chi2: union=%d total=%d (%.1fx expansion)\n",
      union_size, total_features, total_features / union_size)
    train.solutions:add_scaled(-cfg.tm.classes)
    train.problems0 = nil
    tok:restrict(ids_union)
    local function to_ind_bitmap (split)
      local toks = tok:tokenize(split.problems)
      local ind, ind_off = toks:bits_individualize(feat_offsets, feat_ids, union_size)
      local bitmap, dim_off = ind:bits_to_cvec_ind(ind_off, feat_offsets, split.n, true)
      toks:destroy()
      ind:destroy()
      ind_off:destroy()
      return bitmap, dim_off
    end
    train.problems, train_dim_offsets = to_ind_bitmap(train)
    validate.problems, validate_dim_offsets = to_ind_bitmap(validate)
    test.problems, test_dim_offsets = to_ind_bitmap(test)
    n_top_v = union_size
  else
    local top_v, chi2_weights
    if cfg.feature_selection.algo == "chi2" then
      top_v, chi2_weights = train.problems0:bits_top_chi2(train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.top_k)
    else
      top_v, chi2_weights = train.problems0:bits_top_mi(train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.top_k)
    end
    train.solutions:add_scaled(-cfg.tm.classes)
    n_top_v = top_v:size()
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

    print("Prepping for classifier")
    train.problems = train.problems:bits_to_cvec(train.n, n_top_v, true)
    validate.problems = validate.problems:bits_to_cvec(validate.n, n_top_v, true)
    test.problems = test.problems:bits_to_cvec(test.n, n_top_v, true)
  end
  tok:destroy()

  print("Optimizing Classifier")
  local stopwatch = utc.stopwatch()
  local t = optimize.classifier({

    features = n_top_v,
    individualized = use_ind,
    feat_offsets = feat_offsets,
    dim_offsets = train_dim_offsets,

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

    search_metric = function (t0, _)
      local predicted
      if validate_dim_offsets then
        predicted = t0:predict(validate.problems, validate_dim_offsets, validate.n, cfg.threads)
      else
        predicted = t0:predict(validate.problems, validate.n, cfg.threads)
      end
      local accuracy = eval.class_accuracy(predicted, validate.solutions, validate.n, cfg.tm.classes, cfg.threads)
      return accuracy.f1, accuracy
    end,

    each = function (t0, is_final, val_accuracy, params, epoch, round, trial)
      local test_predicted
      if test_dim_offsets then
        test_predicted = t0:predict(test.problems, test_dim_offsets, test.n, cfg.threads)
      else
        test_predicted = t0:predict(test.problems, test.n, cfg.threads)
      end
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
  local train_pred, val_pred, test_pred
  if use_ind then
    train_pred = t:predict(train.problems, train_dim_offsets, train.n, cfg.threads)
    val_pred = t:predict(validate.problems, validate_dim_offsets, validate.n, cfg.threads)
    test_pred = t:predict(test.problems, test_dim_offsets, test.n, cfg.threads)
  else
    train_pred = t:predict(train.problems, train.n, cfg.threads)
    val_pred = t:predict(validate.problems, validate.n, cfg.threads)
    test_pred = t:predict(test.problems, test.n, cfg.threads)
  end
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, cfg.tm.classes, cfg.threads)
  local val_stats = eval.class_accuracy(val_pred, validate.solutions, validate.n, cfg.tm.classes, cfg.threads)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, cfg.tm.classes, cfg.threads)
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
